"""Probe: within-team cosine separation across 43 GT rallies.

For each rally, computes the median learned-ReID embedding per primary
track, then cosine similarity between same-team pairs. Assesses whether
"anchor-based identity propagation" is viable: if enough rallies have
clear within-team separation, the best rally per video can anchor
identity across the match.

Usage:
    # Default: reads from retrack cache's pre-computed learned_store
    uv run python scripts/probe_anchor_viability.py

    # With custom checkpoint: extracts crops from video and runs through
    # DINOv2 backbone + MLP head inline
    uv run python scripts/probe_anchor_viability.py --checkpoint weights/within_team_reid/best.pt
"""

from __future__ import annotations

import argparse
import copy as _copy
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

# Ensure WEIGHT_LEARNED_REID > 0 so config_hash matches the cache built
# during Sessions 4-8 (which stored learned embeddings).
os.environ.setdefault("WEIGHT_LEARNED_REID", "0.05")
# Disable optional passes that might not be relevant here
os.environ.setdefault("ENABLE_OCCLUSION_RESOLVER", "0")

ANALYSIS_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ANALYSIS_ROOT / "reports" / "merge_veto"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
logger = logging.getLogger("probe_anchor_viability")

if TYPE_CHECKING:
    import torch


@dataclass
class RallyRow:
    rally_id: str
    video_id: str
    team0_pair: tuple[int, int]
    team1_pair: tuple[int, int]
    team0_cos: float | None
    team1_cos: float | None


def _load_post_processed_state(rally_id: str):
    """Load retrack cache + run apply_post_processing.

    Uses the monkey-patch trick from eval_occlusion_resolver.py so the
    learned_store is mutated in-place and its IDs match the post-processed
    primary track IDs.

    Returns:
        (result, learned_store) or None if not cached.
    """
    from rallycut.cli.commands.evaluate_tracking import _compute_tracker_config_hash
    from rallycut.evaluation.tracking.retrack_cache import RetrackCache
    from rallycut.tracking.player_filter import PlayerFilterConfig
    from rallycut.tracking.player_tracker import PlayerTracker

    rc = RetrackCache()
    entry = rc.get(rally_id, _compute_tracker_config_hash())
    if entry is None:
        return None

    cached_data, color, app, learned = entry
    if learned is None:
        return None

    filter_config = PlayerFilterConfig().scaled_for_fps(cached_data.video_fps)

    # Monkey-patch deepcopy so apply_post_processing's internal "deep copy"
    # of LearnedEmbeddingStore/ColorHistogramStore is a no-op. This means
    # the remap_ids calls inside post-processing mutate our local `learned`
    # reference directly, giving us the post-processed IDs after the call.
    original_deepcopy = _copy.deepcopy

    def _shallow_when_store(obj, memo=None):  # type: ignore[no-untyped-def]
        from rallycut.tracking.appearance_descriptor import AppearanceDescriptorStore
        from rallycut.tracking.color_repair import ColorHistogramStore, LearnedEmbeddingStore
        if isinstance(obj, (ColorHistogramStore, AppearanceDescriptorStore, LearnedEmbeddingStore)):
            return obj
        return original_deepcopy(obj, memo) if memo is not None else original_deepcopy(obj)

    _copy.deepcopy = _shallow_when_store  # type: ignore[assignment]
    try:
        result = PlayerTracker.apply_post_processing(
            positions=cached_data.positions,
            raw_positions=list(cached_data.positions),
            color_store=color,
            appearance_store=app,
            ball_positions=cached_data.ball_positions,
            video_fps=cached_data.video_fps,
            video_width=cached_data.video_width,
            video_height=cached_data.video_height,
            frame_count=cached_data.frame_count,
            start_frame=0,
            filter_enabled=True,
            filter_config=filter_config,
            learned_store=learned,
        )
    finally:
        _copy.deepcopy = original_deepcopy  # type: ignore[assignment]

    return result, learned


def _frames_for_track(result, track_id: int) -> set[int]:
    """Collect all frame numbers where track_id appears in result.positions."""
    frames: set[int] = set()
    for fp in result.positions:
        if fp.track_id == track_id:
            frames.add(fp.frame_number)
    return frames


def _cosine_pair(
    learned_store,
    tid_a: int,
    frames_a: set[int],
    tid_b: int,
    frames_b: set[int],
) -> float | None:
    """Return cosine similarity between two tracks' median embeddings, or None."""
    from rallycut.tracking.merge_veto import _segment_median_embedding

    emb_a = _segment_median_embedding(learned_store, tid_a, frames_a)
    emb_b = _segment_median_embedding(learned_store, tid_b, frames_b)
    if emb_a is None or emb_b is None:
        return None
    return float(np.dot(emb_a, emb_b))


def _compute_track_embedding_from_crops(
    result,
    track_id: int,
    frames: set[int],
    video_path: Path,
    start_ms: int,
    backbone,
    head,
    device: torch.device,
    max_samples: int = 30,
    min_valid: int = 5,
) -> np.ndarray | None:
    """Extract player crops from video and compute median embedding via backbone+head.

    Args:
        result: Post-processed tracking result (positions).
        track_id: Track ID to extract crops for.
        frames: Set of rally-relative frame numbers for this track.
        video_path: Path to the source video.
        start_ms: Rally start time in milliseconds.
        backbone: BackboneRunner instance.
        head: MLPHead instance (already eval mode).
        device: Torch device.
        max_samples: Maximum number of frames to sample.
        min_valid: Minimum valid crops required; returns None if fewer.

    Returns:
        L2-normalized median embedding (128-d float32 ndarray), or None.
    """
    import cv2
    import torch

    from training.within_team_reid.data.augment import augment_eval

    if not frames:
        return None

    # Build position lookup: frame_number → (x, y, w, h) normalized center-format
    pos_map: dict[int, tuple[float, float, float, float]] = {}
    for fp in result.positions:
        if fp.track_id == track_id and fp.frame_number in frames:
            pos_map[fp.frame_number] = (fp.x, fp.y, fp.width, fp.height)

    if not pos_map:
        return None

    # Uniform sample of up to max_samples frames
    sorted_frames = sorted(pos_map.keys())
    if len(sorted_frames) > max_samples:
        indices = np.linspace(0, len(sorted_frames) - 1, max_samples, dtype=int)
        target_frames = [sorted_frames[i] for i in indices]
    else:
        target_frames = sorted_frames

    target_set = set(target_frames)

    # Open video and seek to rally start
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Cannot open video: %s", video_path)
        return None

    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(start_ms))

        crops_bgr: list[np.ndarray] = []
        current_frame = 0

        while target_set:
            ret, frame = cap.read()
            if not ret:
                break

            if current_frame in target_set:
                target_set.discard(current_frame)
                x, y, w, h = pos_map[current_frame]
                fh, fw = frame.shape[:2]
                x1 = max(0, int((x - w / 2) * fw))
                y1 = max(0, int((y - h / 2) * fh))
                x2 = min(fw, int((x + w / 2) * fw))
                y2 = min(fh, int((y + h / 2) * fh))
                if x2 > x1 and y2 > y1:
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        crops_bgr.append(crop)

            current_frame += 1
            # Stop early if we've passed the last needed frame
            if current_frame > max(pos_map.keys()) + 1:
                break
    finally:
        cap.release()

    if len(crops_bgr) < min_valid:
        return None

    # Preprocess crops using eval augmentation → (3, 224, 224) BGR uint8 tensors
    tensors = [augment_eval(crop) for crop in crops_bgr]
    batch = torch.stack(tensors).to(device)  # (N, 3, 224, 224) uint8

    # Run backbone + head
    with torch.no_grad():
        feats = backbone.forward(batch)   # (N, 384) L2-normed float32
        embeddings = head(feats)          # (N, 128) L2-normed float32

    emb_np = embeddings.cpu().numpy()  # (N, 128)
    median_emb = np.median(emb_np, axis=0).astype(np.float32)
    norm = np.linalg.norm(median_emb)
    if norm < 1e-8:
        return None
    return median_emb / norm


def _cosine_pair_from_crops(
    result,
    tid_a: int,
    frames_a: set[int],
    tid_b: int,
    frames_b: set[int],
    video_path: Path,
    start_ms: int,
    backbone,
    head,
    device: torch.device,
) -> float | None:
    """Compute cosine similarity between two tracks' embeddings extracted from video."""
    emb_a = _compute_track_embedding_from_crops(
        result, tid_a, frames_a, video_path, start_ms, backbone, head, device,
    )
    emb_b = _compute_track_embedding_from_crops(
        result, tid_b, frames_b, video_path, start_ms, backbone, head, device,
    )
    if emb_a is None or emb_b is None:
        return None
    return float(np.dot(emb_a, emb_b))


def _process_rally(
    rally_id: str,
    video_id: str,
    idx: int,
    total: int,
    checkpoint_args: dict | None = None,
) -> RallyRow | None:
    """Process a single rally: load, post-process, compute same-team cosines.

    Args:
        rally_id: Rally ID.
        video_id: Video ID.
        idx: Progress index (1-based).
        total: Total number of rallies.
        checkpoint_args: If set, dict with keys: backbone, head, device,
            video_path, start_ms. When provided, crops are extracted from
            video instead of reading from the retrack cache's learned_store.
    """
    state = _load_post_processed_state(rally_id)
    if state is None:
        print(f"[{idx}/{total}] {rally_id[:8]}: SKIP (no cache or no learned store)")
        return None

    result, learned = state

    primary = result.primary_track_ids or []
    teams = result.team_assignments or {}

    if len(primary) < 4:
        print(f"[{idx}/{total}] {rally_id[:8]}: SKIP (only {len(primary)} primary tracks)")
        return None

    # Group primary tracks by team
    team0 = [t for t in primary if teams.get(t) == 0]
    team1 = [t for t in primary if teams.get(t) == 1]

    if len(team0) < 2 or len(team1) < 2:
        print(
            f"[{idx}/{total}] {rally_id[:8]}: SKIP "
            f"(team0={len(team0)} team1={len(team1)} primaries)"
        )
        return None

    # Use first two tracks per team (typically exactly 2 after filtering)
    t0a, t0b = team0[0], team0[1]
    t1a, t1b = team1[0], team1[1]

    frames_t0a = _frames_for_track(result, t0a)
    frames_t0b = _frames_for_track(result, t0b)
    frames_t1a = _frames_for_track(result, t1a)
    frames_t1b = _frames_for_track(result, t1b)

    if checkpoint_args is not None:
        backbone = checkpoint_args["backbone"]
        head = checkpoint_args["head"]
        device = checkpoint_args["device"]
        video_path = checkpoint_args["video_path"]
        start_ms = checkpoint_args["start_ms"]

        if video_path is None:
            print(f"[{idx}/{total}] {rally_id[:8]}: SKIP (no video path for {video_id[:8]})")
            return None

        cos0 = _cosine_pair_from_crops(
            result, t0a, frames_t0a, t0b, frames_t0b,
            video_path, start_ms, backbone, head, device,
        )
        cos1 = _cosine_pair_from_crops(
            result, t1a, frames_t1a, t1b, frames_t1b,
            video_path, start_ms, backbone, head, device,
        )
    else:
        cos0 = _cosine_pair(learned, t0a, frames_t0a, t0b, frames_t0b)
        cos1 = _cosine_pair(learned, t1a, frames_t1a, t1b, frames_t1b)

    cos0_str = f"{cos0:.3f}" if cos0 is not None else "N/A"
    cos1_str = f"{cos1:.3f}" if cos1 is not None else "N/A"
    print(f"[{idx}/{total}] {rally_id[:8]}: team0 cos={cos0_str}, team1 cos={cos1_str}")

    return RallyRow(
        rally_id=rally_id,
        video_id=video_id,
        team0_pair=(t0a, t0b),
        team1_pair=(t1a, t1b),
        team0_cos=cos0,
        team1_cos=cos1,
    )


def _render_report(rows: list[RallyRow], out_path: Path) -> None:
    # Collect all non-None cosine values (one per team-pair)
    cosines: list[float] = []
    for row in rows:
        if row.team0_cos is not None:
            cosines.append(row.team0_cos)
        if row.team1_cos is not None:
            cosines.append(row.team1_cos)

    n_pairs = len(cosines)
    if n_pairs == 0:
        out_path.write_text("# Anchor Viability Probe\n\nNo valid pairs found.\n")
        return

    arr = np.array(cosines, dtype=np.float32)
    mean_cos = float(np.mean(arr))
    median_cos = float(np.median(arr))
    p25_cos = float(np.percentile(arr, 25))
    p75_cos = float(np.percentile(arr, 75))

    # Per-rally stats: a rally has "clear separation" if ANY pair < 0.5
    # and "ambiguous" if ALL pairs > 0.7
    clear_sep_rallies = 0
    ambiguous_rallies = 0
    for row in rows:
        pair_cosines = [c for c in [row.team0_cos, row.team1_cos] if c is not None]
        if not pair_cosines:
            continue
        if any(c < 0.5 for c in pair_cosines):
            clear_sep_rallies += 1
        if all(c > 0.7 for c in pair_cosines):
            ambiguous_rallies += 1

    n_rallies = len(rows)

    # Per-VIDEO best anchor: for each video, lowest within-team cosine across all rallies
    by_video: dict[str, list[RallyRow]] = defaultdict(list)
    for row in rows:
        by_video[row.video_id].append(row)

    video_best: dict[str, float | None] = {}
    for vid, vrows in by_video.items():
        mins: list[float] = []
        for row in vrows:
            pair_cosines = [c for c in [row.team0_cos, row.team1_cos] if c is not None]
            if pair_cosines:
                mins.append(min(pair_cosines))
        video_best[vid] = min(mins) if mins else None

    n_videos = len(by_video)
    n_viable_videos = sum(1 for v in video_best.values() if v is not None and v < 0.5)
    pct_viable = (n_viable_videos / n_videos * 100) if n_videos > 0 else 0.0

    if pct_viable >= 70:
        verdict = "**VIABLE** — ≥70% of videos have a usable anchor rally (cosine < 0.5)."
    elif pct_viable >= 30:
        verdict = "**MARGINAL** — 30-70% of videos have a usable anchor rally (cosine < 0.5)."
    else:
        verdict = "**NOT VIABLE** — <30% of videos have a usable anchor rally (cosine < 0.5)."

    lines = [
        "# Anchor Viability Probe",
        "",
        "Measures within-team cosine similarity of Session-3 learned-ReID embeddings",
        "across 43 GT rallies. A low cosine between teammates means the head can distinguish",
        "them — a prerequisite for anchor-based identity propagation.",
        "",
        "## Distribution of within-team cosine similarities",
        "",
        f"- **N pairs**: {n_pairs} (from {n_rallies} rallies)",
        f"- **Mean**: {mean_cos:.3f}",
        f"- **Median**: {median_cos:.3f}",
        f"- **P25**: {p25_cos:.3f}",
        f"- **P75**: {p75_cos:.3f}",
        "",
        "## Separation counts",
        "",
        f"- **Clear separation** (≥1 pair with cosine < 0.5): **{clear_sep_rallies} / {n_rallies} rallies** "
        f"({clear_sep_rallies / n_rallies * 100:.0f}%)",
        f"- **Ambiguous** (all pairs with cosine > 0.7): **{ambiguous_rallies} / {n_rallies} rallies** "
        f"({ambiguous_rallies / n_rallies * 100:.0f}%)",
        "",
        "## Per-video best anchor",
        "",
        f"For each of {n_videos} videos, the best (lowest cosine) rally across all rallies in that video.",
        "",
        "| Video (short) | Best cosine | Viable? |",
        "|:--------------|:------------|:-------:|",
    ]

    for vid, best in sorted(video_best.items(), key=lambda kv: (kv[1] is None, kv[1] or 1.0)):
        if best is None:
            lines.append(f"| `{vid[:8]}` | N/A | — |")
        else:
            viable = "yes" if best < 0.5 else "no"
            lines.append(f"| `{vid[:8]}` | {best:.3f} | {viable} |")

    lines += [
        "",
        f"**Videos with viable anchor (best cosine < 0.5): {n_viable_videos} / {n_videos} ({pct_viable:.0f}%)**",
        "",
        "## Verdict",
        "",
        verdict,
        "",
        "## Per-rally detail",
        "",
        "| Rally (short) | Video (short) | Team-0 cos | Team-1 cos |",
        "|:--------------|:--------------|:----------:|:----------:|",
    ]

    for row in rows:
        cos0 = f"{row.team0_cos:.3f}" if row.team0_cos is not None else "N/A"
        cos1 = f"{row.team1_cos:.3f}" if row.team1_cos is not None else "N/A"
        lines.append(f"| `{row.rally_id[:8]}` | `{row.video_id[:8]}` | {cos0} | {cos1} |")

    out_path.write_text("\n".join(lines) + "\n")


def _load_checkpoint(checkpoint_path: Path):
    """Load backbone + head from a checkpoint file.

    Returns:
        (backbone, head, device) tuple.
    """
    import torch

    from training.within_team_reid.model.backbone import BackboneRunner
    from training.within_team_reid.model.head import MLPHead

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading checkpoint {checkpoint_path} on {device}")

    backbone = BackboneRunner(device=device)

    head = MLPHead(in_dim=384, hidden_dim=192, out_dim=128).to(device)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    head.load_state_dict(payload["head_state_dict"])
    head.eval()

    print(f"Checkpoint loaded (epoch={payload.get('epoch', '?')})")
    return backbone, head, device


def _build_video_path_map(rallies) -> dict[str, tuple[Path | None, int]]:
    """Build a map from rally_id → (video_path, start_ms) for all rallies.

    Deduplicates video path lookups by video_id.
    """
    from rallycut.evaluation.tracking.db import get_video_path

    video_paths: dict[str, Path | None] = {}
    result: dict[str, tuple[Path | None, int]] = {}

    for rally in rallies:
        vid = rally.video_id
        if vid not in video_paths:
            vpath = get_video_path(vid)
            video_paths[vid] = vpath
            if vpath is None:
                logger.warning("No video path for video_id=%s", vid)
        result[rally.rally_id] = (video_paths[vid], rally.start_ms)

    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe within-team cosine separation across GT rallies."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Path to a head checkpoint (.pt). When set, extracts crops from video "
            "and runs through DINOv2 backbone + MLP head instead of using the "
            "retrack cache's pre-computed learned_store."
        ),
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=None,
        help=(
            "Output path for the markdown report. "
            "Defaults to anchor_viability_probe.md (no checkpoint) or "
            "anchor_viability_probe_v2.md (with checkpoint)."
        ),
    )
    args = parser.parse_args()

    # Resolve default report output path
    if args.report_out is not None:
        report_out = args.report_out
    elif args.checkpoint is not None:
        report_out = OUT_DIR / "anchor_viability_probe_v2.md"
    else:
        report_out = OUT_DIR / "anchor_viability_probe.md"

    from rallycut.evaluation.tracking.db import load_labeled_rallies

    rallies = load_labeled_rallies()
    total = len(rallies)
    print(f"Loaded {total} GT rallies")

    # Load checkpoint if requested
    checkpoint_backbone = None
    checkpoint_head = None
    checkpoint_device = None
    video_path_map: dict[str, tuple[Path | None, int]] = {}

    if args.checkpoint is not None:
        if not args.checkpoint.exists():
            print(f"ERROR: checkpoint not found: {args.checkpoint}", file=sys.stderr)
            return 1
        checkpoint_backbone, checkpoint_head, checkpoint_device = _load_checkpoint(
            args.checkpoint
        )
        print("Building video path map...")
        video_path_map = _build_video_path_map(rallies)

    rows: list[RallyRow] = []
    for i, rally in enumerate(rallies, 1):
        if args.checkpoint is not None:
            video_path, start_ms = video_path_map.get(rally.rally_id, (None, 0))
            ckpt_args: dict | None = {
                "backbone": checkpoint_backbone,
                "head": checkpoint_head,
                "device": checkpoint_device,
                "video_path": video_path,
                "start_ms": start_ms,
            }
        else:
            ckpt_args = None

        row = _process_rally(rally.rally_id, rally.video_id, i, total, ckpt_args)
        if row is not None:
            rows.append(row)

    print(f"\nProcessed {len(rows)} / {total} rallies successfully")

    # Summary stats
    cosines: list[float] = []
    for row in rows:
        if row.team0_cos is not None:
            cosines.append(row.team0_cos)
        if row.team1_cos is not None:
            cosines.append(row.team1_cos)

    if cosines:
        arr = np.array(cosines)
        print("\n--- Distribution ---")
        print(f"  N pairs:  {len(cosines)}")
        print(f"  Mean:     {np.mean(arr):.3f}")
        print(f"  Median:   {np.median(arr):.3f}")
        print(f"  P25:      {np.percentile(arr, 25):.3f}")
        print(f"  P75:      {np.percentile(arr, 75):.3f}")
        print(f"  Min:      {np.min(arr):.3f}")
        print(f"  Max:      {np.max(arr):.3f}")

        n_clear = sum(
            1 for row in rows
            if any(c < 0.5 for c in [row.team0_cos, row.team1_cos] if c is not None)
        )
        n_ambig = sum(
            1 for row in rows
            if all(c > 0.7 for c in [row.team0_cos, row.team1_cos] if c is not None)
        )
        by_video: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            for c in [row.team0_cos, row.team1_cos]:
                if c is not None:
                    by_video[row.video_id].append(c)
        video_bests = {v: min(cs) for v, cs in by_video.items()}
        n_viable = sum(1 for b in video_bests.values() if b < 0.5)
        n_vids = len(video_bests)
        pct = n_viable / n_vids * 100 if n_vids > 0 else 0.0

        print("\n--- Separation ---")
        print(f"  Clear (any pair cos<0.5): {n_clear} / {len(rows)} rallies")
        print(f"  Ambiguous (all pairs cos>0.7): {n_ambig} / {len(rows)} rallies")
        print("\n--- Per-video anchor coverage ---")
        print(f"  Viable videos (best cos<0.5): {n_viable} / {n_vids} ({pct:.0f}%)")
        if pct >= 70:
            print("  VERDICT: VIABLE")
        elif pct >= 30:
            print("  VERDICT: MARGINAL")
        else:
            print("  VERDICT: NOT VIABLE")

    _render_report(rows, report_out)
    print(f"\nReport written to {report_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
