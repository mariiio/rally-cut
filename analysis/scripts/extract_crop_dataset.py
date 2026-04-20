"""Extract per-candidate crops for Phase 1 crop-head probe.

Outputs: outputs/crop_dataset/<video_id>/<rally-short>_<frame>_<source>.npz with
  - rally_id, frame, label, gbm_conf, source
  - player_crop: (T=9, 3, 64, 64) float32 normalized [0,1]
  - ball_patch: (T=9, 3, 32, 32) float32 normalized [0,1]

Samples per rally:
  - Positives: one per GT contact frame (label=1, source="gt_positive")
  - Hard negatives: up to 5 candidate frames with GBM conf in [0.20, 0.30]
    AND >= 8 frames from any GT (label=0, source="hard_negative")
  - Random negatives: up to 10 candidate frames with GBM conf < 0.20
    AND >= 8 frames from any GT (label=0, source="random_negative")

Usage (cd analysis):
    uv run python scripts/extract_crop_dataset.py
    uv run python scripts/extract_crop_dataset.py --video <video_id>
    uv run python scripts/extract_crop_dataset.py --limit 5
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rallycut.evaluation.tracking.db import get_video_path
from rallycut.tracking.contact_classifier import load_contact_classifier
from rallycut.tracking.contact_detector import ContactDetectionConfig
from rallycut.tracking.player_tracker import PlayerPosition as PlayerPos
from scripts.eval_action_detection import RallyData, load_rallies_with_action_gt
from scripts.eval_loo_video import _precompute

console = Console()

T_FRAMES = 9
HALF = T_FRAMES // 2
PLAYER_CROP_SIZE = 64
BALL_PATCH_SIZE = 32
BALL_PATCH_HALF_NORM = 0.04  # ~34px at 854 width
PLAYER_PAD_FRAC = 0.10

GT_EXCLUDE_FRAMES = 8
HARD_CONF_LO = 0.20
HARD_CONF_HI = 0.30
RND_CONF_HI = 0.20
MAX_HARD_PER_RALLY = 5
MAX_RND_PER_RALLY = 10

OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "outputs" / "crop_dataset"


@dataclass
class SampleSpec:
    rally_id: str
    video_id: str
    target_frame: int  # rally-local
    label: int
    gbm_conf: float
    source: str
    track_id: int  # for player bbox lookup


def _crop_player(img: np.ndarray, bbox_xywh_norm: tuple[float, float, float, float]) -> np.ndarray:
    h, w = img.shape[:2]
    cx, cy, bw, bh = bbox_xywh_norm
    pad_w = bw * PLAYER_PAD_FRAC
    pad_h = bh * PLAYER_PAD_FRAC
    x0 = max(0, int((cx - bw / 2 - pad_w) * w))
    x1 = min(w, int((cx + bw / 2 + pad_w) * w))
    y0 = max(0, int((cy - bh / 2 - pad_h) * h))
    y1 = min(h, int((cy + bh / 2 + pad_h) * h))
    if x1 <= x0 or y1 <= y0:
        return np.zeros((PLAYER_CROP_SIZE, PLAYER_CROP_SIZE, 3), dtype=np.float32)
    crop = img[y0:y1, x0:x1]
    crop = cv2.resize(crop, (PLAYER_CROP_SIZE, PLAYER_CROP_SIZE))
    return crop.astype(np.float32) / 255.0


def _crop_ball(img: np.ndarray, ball_xy_norm: tuple[float, float] | None) -> np.ndarray:
    if ball_xy_norm is None:
        return np.zeros((BALL_PATCH_SIZE, BALL_PATCH_SIZE, 3), dtype=np.float32)
    h, w = img.shape[:2]
    bx, by = ball_xy_norm
    x0 = max(0, int((bx - BALL_PATCH_HALF_NORM) * w))
    x1 = min(w, int((bx + BALL_PATCH_HALF_NORM) * w))
    y0 = max(0, int((by - BALL_PATCH_HALF_NORM) * h))
    y1 = min(h, int((by + BALL_PATCH_HALF_NORM) * h))
    if x1 <= x0 or y1 <= y0:
        return np.zeros((BALL_PATCH_SIZE, BALL_PATCH_SIZE, 3), dtype=np.float32)
    patch = img[y0:y1, x0:x1]
    patch = cv2.resize(patch, (BALL_PATCH_SIZE, BALL_PATCH_SIZE))
    return patch.astype(np.float32) / 255.0


def _build_bbox_lookup(
    player_positions: list[PlayerPos], track_id: int
) -> dict[int, tuple[float, float, float, float]]:
    """Map rally-local frame → (cx, cy, w, h) for the given track_id."""
    out: dict[int, tuple[float, float, float, float]] = {}
    for p in player_positions:
        if p.track_id == track_id:
            out[p.frame_number] = (p.x, p.y, p.width, p.height)
    return out


def _build_ball_lookup(ball_positions_json: list[dict]) -> dict[int, tuple[float, float]]:
    out: dict[int, tuple[float, float]] = {}
    for bp in ball_positions_json or []:
        x = bp.get("x", 0.0)
        y = bp.get("y", 0.0)
        if (x > 0 or y > 0) and bp.get("confidence", 1.0) >= 0.3:
            out[bp["frameNumber"]] = (x, y)
    return out


def _nearest_track_id_at_frame(
    player_positions: list[PlayerPos],
    frame: int,
    ball_xy: tuple[float, float] | None,
    window: int = 2,
) -> int | None:
    """Find the track_id of the player nearest to the ball at the target frame
    (or within ±window frames). Returns None if no player/ball available."""
    if ball_xy is None:
        return None
    candidates: list[tuple[float, int]] = []
    for p in player_positions:
        if abs(p.frame_number - frame) <= window:
            dx = p.x - ball_xy[0]
            dy = p.y - ball_xy[1]
            d = (dx * dx + dy * dy) ** 0.5
            candidates.append((d, p.track_id))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]


def _extract_sample(
    cap: cv2.VideoCapture,
    spec: SampleSpec,
    rally_start_frame: int,
    bbox_by_frame: dict[int, tuple[float, float, float, float]],
    ball_by_frame: dict[int, tuple[float, float]],
) -> dict | None:
    player_crops = np.zeros((T_FRAMES, 3, PLAYER_CROP_SIZE, PLAYER_CROP_SIZE), dtype=np.float32)
    ball_patches = np.zeros((T_FRAMES, 3, BALL_PATCH_SIZE, BALL_PATCH_SIZE), dtype=np.float32)

    for i, offset in enumerate(range(-HALF, HALF + 1)):
        rally_f = spec.target_frame + offset
        if rally_f < 0:
            continue
        abs_f = rally_start_frame + rally_f
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(abs_f))
        ok, img = cap.read()
        if not ok or img is None:
            continue

        bbox = bbox_by_frame.get(rally_f)
        if bbox is None:
            for d in (1, -1, 2, -2):
                bbox = bbox_by_frame.get(rally_f + d)
                if bbox is not None:
                    break
        if bbox is not None:
            crop_hwc = _crop_player(img, bbox)
            player_crops[i] = crop_hwc.transpose(2, 0, 1)

        ball = ball_by_frame.get(rally_f)
        if ball is None:
            for d in (1, -1, 2, -2):
                ball = ball_by_frame.get(rally_f + d)
                if ball is not None:
                    break
        if ball is not None:
            patch_hwc = _crop_ball(img, ball)
            ball_patches[i] = patch_hwc.transpose(2, 0, 1)

    return {
        "rally_id": spec.rally_id,
        "frame": spec.target_frame,
        "player_crop": player_crops,
        "ball_patch": ball_patches,
        "label": spec.label,
        "gbm_conf": spec.gbm_conf,
        "source": spec.source,
    }


def _build_sample_specs_for_rally(
    rally: RallyData,
    contact_cfg: ContactDetectionConfig,
    classifier,
    rng: random.Random,
):
    """Enumerate positives + hard-negs + random-negs for one rally.

    Returns (specs, pre, ball_lookup) or None.
    """
    pre = _precompute(rally, contact_cfg)
    if pre is None:
        return None

    # Candidate-frame → GBM confidence
    probas: list[float] = []
    if len(pre.candidate_features) > 0 and classifier is not None and classifier.is_trained:
        x_mat = pre.candidate_features
        expected = classifier.model.n_features_in_
        if x_mat.shape[1] > expected:
            x_mat = x_mat[:, :expected]
        elif x_mat.shape[1] < expected:
            pad = np.zeros((x_mat.shape[0], expected - x_mat.shape[1]))
            x_mat = np.hstack([x_mat, pad])
        probas = classifier.model.predict_proba(x_mat)[:, 1].tolist()

    # Recover candidate frames in the same order (frames_since_last is computed
    # against GT-matched candidates, so we re-run extract to get the frame list
    # — _precompute hid this. Use cand_features[i].frame instead.)
    # Actually cand_feats are CandidateFeatures objects in _precompute; but we
    # stored them as numpy. We don't have .frame. Re-derive from labels + pre:
    # simplest is to re-run extract_candidate_features for the frame list.
    from scripts.train_contact_classifier import extract_candidate_features
    cand_feats, cand_frames = extract_candidate_features(
        rally,
        config=contact_cfg,
        gt_frames=[gt.frame for gt in rally.gt_labels],
        sequence_probs=pre.sequence_probs,
    )
    if not cand_frames or len(cand_frames) != len(probas):
        # classifier missing or mismatched — abort this rally
        return None

    gt_frames = sorted(gt.frame for gt in rally.gt_labels)
    ball_lookup = _build_ball_lookup(rally.ball_positions_json or [])

    def _min_abs_gap(f: int) -> int:
        if not gt_frames:
            return 10_000
        return min(abs(f - g) for g in gt_frames)

    specs: list[SampleSpec] = []

    # --- Positives: one per GT contact ---
    for gt in rally.gt_labels:
        tid = gt.player_track_id
        if tid is None or tid < 0:
            tid_inferred = _nearest_track_id_at_frame(
                pre.player_positions, gt.frame, ball_lookup.get(gt.frame)
            )
            if tid_inferred is None:
                continue
            tid = tid_inferred
        specs.append(SampleSpec(
            rally_id=rally.rally_id,
            video_id=rally.video_id,
            target_frame=gt.frame,
            label=1,
            gbm_conf=0.0,  # positives: conf not used by label
            source="gt_positive",
            track_id=tid,
        ))

    # --- Hard-neg + random-neg pools ---
    hard_pool: list[tuple[int, float]] = []  # (frame, conf)
    rnd_pool: list[tuple[int, float]] = []
    for frame, conf in zip(cand_frames, probas, strict=True):
        if _min_abs_gap(frame) < GT_EXCLUDE_FRAMES:
            continue
        if HARD_CONF_LO <= conf <= HARD_CONF_HI:
            hard_pool.append((frame, conf))
        elif conf < RND_CONF_HI:
            rnd_pool.append((frame, conf))

    rng.shuffle(hard_pool)
    rng.shuffle(rnd_pool)
    for frame, conf in hard_pool[:MAX_HARD_PER_RALLY]:
        tid = _nearest_track_id_at_frame(pre.player_positions, frame, ball_lookup.get(frame))
        if tid is None:
            continue
        specs.append(SampleSpec(
            rally_id=rally.rally_id, video_id=rally.video_id,
            target_frame=frame, label=0, gbm_conf=conf, source="hard_negative", track_id=tid,
        ))
    for frame, conf in rnd_pool[:MAX_RND_PER_RALLY]:
        tid = _nearest_track_id_at_frame(pre.player_positions, frame, ball_lookup.get(frame))
        if tid is None:
            continue
        specs.append(SampleSpec(
            rally_id=rally.rally_id, video_id=rally.video_id,
            target_frame=frame, label=0, gbm_conf=conf, source="random_negative", track_id=tid,
        ))

    return specs, pre, ball_lookup


def _process_video(
    video_id: str,
    rallies: list[RallyData],
    contact_cfg: ContactDetectionConfig,
    classifier,
    output_root: Path,
    rng: random.Random,
    dry_run: bool,
) -> tuple[int, int, int]:
    """Process all rallies for one video. Returns (n_pos, n_hard, n_rnd)."""
    video_path = get_video_path(video_id)
    if video_path is None:
        console.print(f"  [yellow]no video path for {video_id[:8]}, skipping[/yellow]")
        return 0, 0, 0

    # Build samples across all rallies BEFORE opening VideoCapture
    all_work: list[tuple[RallyData, list[SampleSpec], object, dict]] = []
    for rally in rallies:
        result = _build_sample_specs_for_rally(rally, contact_cfg, classifier, rng)
        if result is None:
            continue
        specs, pre, ball_lookup = result
        if not specs:
            continue
        all_work.append((rally, specs, pre, ball_lookup))

    vid_out = output_root / video_id
    if not dry_run:
        vid_out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        console.print(f"  [red]failed to open {video_path}[/red]")
        return 0, 0, 0

    n_pos = n_hard = n_rnd = 0
    try:
        for rally, specs, pre, ball_lookup in all_work:
            rally_start_frame = round(rally.start_ms * rally.fps / 1000)
            specs_sorted = sorted(specs, key=lambda s: s.target_frame)
            for spec in specs_sorted:
                bbox_lookup = _build_bbox_lookup(pre.player_positions, spec.track_id)
                if not bbox_lookup:
                    continue
                sample = _extract_sample(cap, spec, rally_start_frame, bbox_lookup, ball_lookup)
                if sample is None:
                    continue
                short = rally.rally_id.split("-")[0]
                fname = f"{short}_f{spec.target_frame:06d}_{spec.source}.npz"
                if not dry_run:
                    np.savez(
                        vid_out / fname,
                        rally_id=sample["rally_id"],
                        frame=sample["frame"],
                        player_crop=sample["player_crop"],
                        ball_patch=sample["ball_patch"],
                        label=sample["label"],
                        gbm_conf=sample["gbm_conf"],
                        source=sample["source"],
                    )
                if spec.source == "gt_positive":
                    n_pos += 1
                elif spec.source == "hard_negative":
                    n_hard += 1
                else:
                    n_rnd += 1
    finally:
        cap.release()

    return n_pos, n_hard, n_rnd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=str, default=None,
                        help="Process only this video_id (smoke test)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N videos after grouping")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute specs and print counts but do not write .npz")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=str, default=str(OUTPUT_ROOT))
    args = parser.parse_args()

    rng = random.Random(args.seed)
    output_root = Path(args.output_root)

    console.print("[bold]Loading rallies with action GT...[/bold]")
    rallies = load_rallies_with_action_gt()
    console.print(f"  {len(rallies)} rallies across {len({r.video_id for r in rallies})} videos")

    if args.video:
        rallies = [r for r in rallies if r.video_id == args.video]
        console.print(f"  filtered to video {args.video[:8]}: {len(rallies)} rallies")

    by_video: dict[str, list[RallyData]] = defaultdict(list)
    for r in rallies:
        by_video[r.video_id].append(r)
    video_ids = sorted(by_video.keys())
    if args.limit is not None:
        video_ids = video_ids[: args.limit]

    console.print("[bold]Loading default contact classifier...[/bold]")
    classifier = load_contact_classifier()
    if classifier is None or not classifier.is_trained:
        console.print("[red]No trained contact classifier available — aborting.[/red]")
        sys.exit(1)

    contact_cfg = ContactDetectionConfig()
    output_root.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    total_pos = total_hard = total_rnd = 0
    for i, vid in enumerate(video_ids, start=1):
        t0 = time.time()
        n_pos, n_hard, n_rnd = _process_video(
            vid, by_video[vid], contact_cfg, classifier, output_root, rng, args.dry_run,
        )
        total_pos += n_pos
        total_hard += n_hard
        total_rnd += n_rnd
        dt = time.time() - t0
        console.print(
            f"  [{i}/{len(video_ids)}] {vid[:8]}: "
            f"{n_pos} pos + {n_hard} hard + {n_rnd} rnd saved ({dt:.1f}s)"
        )

    total = total_pos + total_hard + total_rnd
    total_dt = time.time() - t_start
    console.print(
        f"\n[bold]Done.[/bold] {total} samples total "
        f"({total_pos} pos + {total_hard} hard + {total_rnd} rnd) "
        f"across {len(video_ids)} videos in {total_dt/60:.1f} min"
    )
    if args.dry_run:
        console.print("[yellow](dry-run — nothing written)[/yellow]")


if __name__ == "__main__":
    main()
