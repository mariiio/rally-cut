"""Feasibility probe — does pose-keypoint geometry add signal at swap events?

For each pred-exchange swap, we run YOLO-Pose at swap_frame ± 15 frames and
measure in-rally pose-descriptor distances:

    A = pred_new's pose descriptor from frames BEFORE the swap
    B = pred_new's pose descriptor from frames AFTER the swap
    C = pred_old's pose descriptor from frames BEFORE the swap

The question: does pose separate A (same body as B) from C (different body)?

Verdicts per swap:
    pose_agrees_with_hsv_correct   cost(B,A) < cost(B,C) by Δ ≥ 0.08
                                   (pose says pred_new is still the same body →
                                   swap was wrong, pose would have prevented it)
    pose_says_swap_was_recovery    cost(B,C) < cost(B,A) by Δ ≥ 0.08
                                   (pose says pred_new's post-swap body matches
                                   pred_old's pre-swap body — the "swap" is
                                   actually a correct re-association and what we
                                   classified as a tracker error was a fix)
    pose_blind                     |Δ| < 0.05 — pose can't distinguish
    no_pose_data                   not enough high-confidence keypoints

Usage:
    uv run python scripts/check_pose_signal_at_swaps.py                 # default trio
    uv run python scripts/check_pose_signal_at_swaps.py --all-swap-rallies
    uv run python scripts/check_pose_signal_at_swaps.py --rally <id>
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np

from rallycut.evaluation.db import get_connection
from rallycut.evaluation.tracking.db import get_video_path, load_labeled_rallies
from rallycut.tracking.pose_attribution.pose_cache import extract_pose_for_rally

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("pose-probe")

DEFAULT_RALLIES = [
    "fad29c31-6e2a-4a8d-86f1-9064b2f1f425",
    "209be896-b680-44dc-bf31-693f4e287149",
    "d724bbf0-bd0c-44e8-93d5-135aa07df5a1",
]

MIN_KPT_CONFIDENCE = 0.5
WINDOW = 15
DELTA_HAS_SIGNAL = 0.08
DELTA_BLIND = 0.05

# COCO-17 indices we care about
KPT_LEFT_SHOULDER = 5
KPT_RIGHT_SHOULDER = 6
KPT_LEFT_WRIST = 9
KPT_RIGHT_WRIST = 10
KPT_LEFT_HIP = 11
KPT_RIGHT_HIP = 12
KPT_LEFT_ANKLE = 15
KPT_RIGHT_ANKLE = 16
KPT_NOSE = 0


def _pair_midpoint(kps: np.ndarray, i: int, j: int) -> np.ndarray | None:
    """Mean of two keypoints, or None if either is low-confidence."""
    if kps[i, 2] < MIN_KPT_CONFIDENCE or kps[j, 2] < MIN_KPT_CONFIDENCE:
        return None
    return (kps[i, :2] + kps[j, :2]) / 2.0


def _dist(p1: np.ndarray | None, p2: np.ndarray | None) -> float | None:
    if p1 is None or p2 is None:
        return None
    return float(np.linalg.norm(p1 - p2))


def compute_pose_descriptor(kps: np.ndarray) -> np.ndarray | None:
    """6-dim scale-invariant body descriptor from COCO-17 keypoints.

    Returns None if the torso length is undefined (we can't normalise).
    """
    shoulder_mid = _pair_midpoint(kps, KPT_LEFT_SHOULDER, KPT_RIGHT_SHOULDER)
    hip_mid = _pair_midpoint(kps, KPT_LEFT_HIP, KPT_RIGHT_HIP)
    torso = _dist(shoulder_mid, hip_mid)
    # Keypoints are normalised to [0, 1]; require a minimum torso length of
    # ~2% of image height so tiny/partial detections don't produce noisy ratios.
    if torso is None or torso < 0.02:
        return None

    def _norm(d: float | None) -> float | None:
        return d / torso if d is not None else None

    shoulder_width = _dist(
        kps[KPT_LEFT_SHOULDER, :2] if kps[KPT_LEFT_SHOULDER, 2] >= MIN_KPT_CONFIDENCE else None,
        kps[KPT_RIGHT_SHOULDER, :2] if kps[KPT_RIGHT_SHOULDER, 2] >= MIN_KPT_CONFIDENCE else None,
    )
    hip_width = _dist(
        kps[KPT_LEFT_HIP, :2] if kps[KPT_LEFT_HIP, 2] >= MIN_KPT_CONFIDENCE else None,
        kps[KPT_RIGHT_HIP, :2] if kps[KPT_RIGHT_HIP, 2] >= MIN_KPT_CONFIDENCE else None,
    )
    ankle_mid = _pair_midpoint(kps, KPT_LEFT_ANKLE, KPT_RIGHT_ANKLE)
    leg = _dist(hip_mid, ankle_mid)
    wrist_left = kps[KPT_LEFT_WRIST, :2] if kps[KPT_LEFT_WRIST, 2] >= MIN_KPT_CONFIDENCE else None
    wrist_right = kps[KPT_RIGHT_WRIST, :2] if kps[KPT_RIGHT_WRIST, 2] >= MIN_KPT_CONFIDENCE else None
    left_arm = _dist(kps[KPT_LEFT_SHOULDER, :2] if kps[KPT_LEFT_SHOULDER, 2] >= MIN_KPT_CONFIDENCE else None, wrist_left)
    right_arm = _dist(kps[KPT_RIGHT_SHOULDER, :2] if kps[KPT_RIGHT_SHOULDER, 2] >= MIN_KPT_CONFIDENCE else None, wrist_right)
    arm = left_arm if left_arm is not None else right_arm
    head = kps[KPT_NOSE, :2] if kps[KPT_NOSE, 2] >= MIN_KPT_CONFIDENCE else None
    head_to_shoulder = _dist(head, shoulder_mid)

    desc = [
        _norm(shoulder_width) or np.nan,
        _norm(hip_width) or np.nan,
        _norm(leg) or np.nan,
        _norm(arm) or np.nan,
        _norm(head_to_shoulder) or np.nan,
        (shoulder_width / hip_width) if (shoulder_width and hip_width) else np.nan,
    ]
    out = np.array(desc, dtype=np.float32)
    if np.all(np.isnan(out)):
        return None
    return out


def _mean_desc(descs: list[np.ndarray]) -> np.ndarray | None:
    """Mean descriptor, ignoring NaN entries. Returns None if empty."""
    if not descs:
        return None
    stack = np.stack(descs)
    mean = np.nanmean(stack, axis=0)
    if np.all(np.isnan(mean)):
        return None
    return mean


def _desc_cost(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    """L1 distance on the shared (non-NaN) feature set, normalised to [0,1]."""
    if a is None or b is None:
        return None
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() == 0:
        return None
    diff = np.abs(a[mask] - b[mask])
    # Each feature is a ratio typically in [0, 3]; bound diff to [0, 1].
    return float(np.clip(diff.mean() / 1.0, 0.0, 1.0))


def _load_swap_events(audit_path: Path) -> list[dict]:
    """Same swap extraction as debug_reid_at_swaps.py, inlined for standalone use."""
    audit = json.loads(audit_path.read_text())
    pred_history: dict[int, list[tuple[int, int, int]]] = {}
    for g in audit.get("perGt", []):
        for s, e, pid in g.get("predIdSpans", []):
            pred_history.setdefault(pid, []).append((g["gtTrackId"], s, e))
    for h in pred_history.values():
        h.sort(key=lambda t: t[1])

    def prior_gt_of(pid: int, before: int) -> int | None:
        last_gt: int | None = None
        for gt_id, s, _e in pred_history.get(pid, []):
            if s >= before:
                break
            last_gt = gt_id
        return last_gt

    events = []
    for g in audit.get("perGt", []):
        spans = g.get("predIdSpans", [])
        for prev, cur in zip(spans, spans[1:]):
            _, prev_end, prev_pred = prev
            cur_start, _, cur_pred = cur
            if prev_pred == cur_pred or prev_pred < 0 or cur_pred < 0:
                continue
            incoming = prior_gt_of(cur_pred, cur_start)
            if incoming is None or incoming == g["gtTrackId"]:
                continue
            events.append({
                "rally_id": audit["rallyId"],
                "video_id": audit["videoId"],
                "swap_frame": cur_start,
                "gt_track_id": g["gtTrackId"],
                "pred_old": prev_pred,
                "pred_new": cur_pred,
            })
    return events


def _rally_ctx(rally_id: str) -> tuple[float, float] | None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT r.start_ms, v.fps
                   FROM rallies r JOIN videos v ON v.id = r.video_id
                   WHERE r.id = %s""",
                [rally_id],
            )
            row = cur.fetchone()
    if not row:
        return None
    return float(row[0] or 0), float(row[1] or 30.0)


def _classify_event(
    cost_same: float | None,
    cost_other: float | None,
) -> str:
    if cost_same is None or cost_other is None:
        return "no_pose_data"
    delta = cost_other - cost_same  # +ve → pose says same body (swap was wrong)
    if delta >= DELTA_HAS_SIGNAL:
        return "pose_agrees_with_hsv_correct"
    if delta <= -DELTA_HAS_SIGNAL:
        return "pose_says_swap_was_recovery"
    if abs(delta) < DELTA_BLIND:
        return "pose_blind"
    return "pose_weak_signal"


def run_for_rally(rally_id: str, audit_dir: Path, pose_model: object) -> list[dict]:
    audit_path = audit_dir / f"{rally_id}.json"
    if not audit_path.exists():
        logger.warning(f"  no audit for {rally_id}")
        return []
    events = _load_swap_events(audit_path)
    if not events:
        return []

    rallies = load_labeled_rallies(rally_id=rally_id)
    if not rallies or rallies[0].predictions is None:
        logger.warning(f"  no predictions for {rally_id}")
        return []
    rally = rallies[0]

    ctx = _rally_ctx(rally_id)
    if ctx is None:
        return []
    rally_start_ms, fps = ctx

    video_path = get_video_path(rally.video_id)
    if video_path is None:
        logger.warning(f"  cannot fetch video for {rally_id}")
        return []

    # positions_json in dict form (pose extractor expects dicts).
    positions_json = [
        {"frameNumber": p.frame_number, "trackId": p.track_id,
         "x": p.x, "y": p.y, "width": p.width, "height": p.height}
        for p in rally.predictions.positions
    ]

    swap_frames = [ev["swap_frame"] for ev in events]
    logger.info(f"  {len(events)} swap events, running YOLO-pose at {len(swap_frames)} swap centres (window ±{WINDOW})")

    pose_data = extract_pose_for_rally(
        video_path=str(video_path),
        rally_start_ms=int(rally_start_ms),
        fps=fps,
        contact_frames=swap_frames,
        positions_json=positions_json,
        window_half=WINDOW,
        iou_threshold=0.3,
        pose_model=pose_model,
    )
    logger.info(f"  pose_data: {len(pose_data['frames'])} samples, "
                f"track_ids={sorted(set(pose_data['track_ids'].tolist())) if len(pose_data['track_ids']) else '[]'}, "
                f"frame range=[{pose_data['frames'].min() if len(pose_data['frames']) else '—'}, "
                f"{pose_data['frames'].max() if len(pose_data['frames']) else '—'}]")

    # Index pose by (frame, track_id) → keypoints
    pose_by_ft: dict[tuple[int, int], np.ndarray] = {}
    for i in range(len(pose_data["frames"])):
        f = int(pose_data["frames"][i])
        t = int(pose_data["track_ids"][i])
        pose_by_ft[(f, t)] = pose_data["keypoints"][i]  # (17, 3)

    def gather_descs(pred_id: int, frame_range: range) -> list[np.ndarray]:
        out = []
        for f in frame_range:
            kps = pose_by_ft.get((f, pred_id))
            if kps is None:
                continue
            d = compute_pose_descriptor(kps)
            if d is not None:
                out.append(d)
        return out

    results = []
    for ev in events:
        swap_frame = ev["swap_frame"]
        pred_new = ev["pred_new"]
        pred_old = ev["pred_old"]
        pre_range = range(max(0, swap_frame - WINDOW), swap_frame)
        post_range = range(swap_frame, swap_frame + WINDOW)

        # Cache the gather_descs results once per (pred, range); samples count
        # reuses the same list instead of re-scanning pose_by_ft.
        new_pre_descs = gather_descs(pred_new, pre_range)
        new_post_descs = gather_descs(pred_new, post_range)
        old_pre_descs = gather_descs(pred_old, pre_range)

        desc_new_pre = _mean_desc(new_pre_descs)
        desc_new_post = _mean_desc(new_post_descs)
        desc_old_pre = _mean_desc(old_pre_descs)

        # cost(post, pre_of_new) = "same body" hypothesis
        # cost(post, pre_of_old) = "body changed to what pred_old had" hypothesis
        cost_same = _desc_cost(desc_new_post, desc_new_pre)
        cost_other = _desc_cost(desc_new_post, desc_old_pre)

        verdict = _classify_event(cost_same, cost_other)
        results.append({
            "rally_id": rally_id,
            "swap_frame": swap_frame,
            "pred_old": pred_old,
            "pred_new": pred_new,
            "gt_track_id": ev["gt_track_id"],
            "samples_new_pre": len(new_pre_descs),
            "samples_new_post": len(new_post_descs),
            "samples_old_pre": len(old_pre_descs),
            "cost_same_body": cost_same,
            "cost_different_body": cost_other,
            "verdict": verdict,
        })
        logger.info(
            f"  swap@{swap_frame} pred {pred_old}→{pred_new} on GT {ev['gt_track_id']}: "
            f"cost(same)={cost_same if cost_same is None else f'{cost_same:.3f}'}  "
            f"cost(other)={cost_other if cost_other is None else f'{cost_other:.3f}'}  "
            f"→ {verdict}"
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=Path("reports/tracking_audit"))
    parser.add_argument("--out", type=Path, default=Path("reports/tracking_audit/reid_debug/pose_signal.md"))
    parser.add_argument("--rally", type=str, default=None)
    parser.add_argument("--all-swap-rallies", action="store_true")
    args = parser.parse_args()

    if args.rally:
        rally_ids = [args.rally]
    elif args.all_swap_rallies:
        rally_ids = []
        for p in sorted(args.audit_dir.glob("*.json")):
            if p.name == "_summary.json":
                continue
            if _load_swap_events(p):
                rally_ids.append(json.loads(p.read_text())["rallyId"])
    else:
        rally_ids = DEFAULT_RALLIES

    # Load pose model once and share across rallies
    from ultralytics import YOLO
    logger.info("loading yolo11s-pose...")
    pose_model = YOLO("yolo11s-pose.pt")

    all_results: list[dict] = []
    for idx, rid in enumerate(rally_ids, start=1):
        logger.info(f"[{idx}/{len(rally_ids)}] {rid[:8]}")
        all_results.extend(run_for_rally(rid, audit_dir=args.audit_dir, pose_model=pose_model))

    counts = Counter(r["verdict"] for r in all_results)
    total = len(all_results)

    lines = [
        "# Pose-signal feasibility probe at swap events",
        "",
        f"Probed **{total}** swap events across {len(rally_ids)} rally(s).",
        "",
        "## Verdict counts",
        "",
        "| Verdict | Count | Share | What it means |",
        "|---|---:|---:|---|",
    ]
    descriptions = {
        "pose_agrees_with_hsv_correct":
            "pose(same_body) < pose(other_body) by ≥ 0.08 — pose would have prevented the swap",
        "pose_says_swap_was_recovery":
            "pose(other_body) < pose(same_body) by ≥ 0.08 — pose says pred_new is now tracking "
            "pred_old's prior body, so the 'swap' is actually a correct re-association",
        "pose_blind":
            "|Δ| < 0.05 — pose descriptors are too close to help",
        "pose_weak_signal":
            "0.05 ≤ |Δ| < 0.08 — marginal signal, would need combining with HSV to decide",
        "no_pose_data":
            "missing pose keypoints for pred_new pre/post or pred_old pre (YOLO-pose didn't "
            "match or confidence too low)",
    }
    for v, desc in descriptions.items():
        n = counts.get(v, 0)
        pct = f"{100 * n / total:.1f}%" if total else "0.0%"
        lines.append(f"| `{v}` | {n} | {pct} | {desc} |")

    lines.extend([
        "",
        "## Interpretation",
        "",
    ])
    helpful = counts.get("pose_agrees_with_hsv_correct", 0)
    recovery = counts.get("pose_says_swap_was_recovery", 0)
    blind = counts.get("pose_blind", 0) + counts.get("no_pose_data", 0)
    weak = counts.get("pose_weak_signal", 0)

    if helpful + recovery >= (total - blind) * 0.5 and (helpful + recovery) >= 5:
        lines.append(
            f"- **Pose has meaningful signal on {helpful + recovery}/{total} events.** "
            f"Proceed with structural integration: add pose descriptor to `compute_appearance_similarity()` "
            f"with an eval-gated merge."
        )
    elif (helpful + recovery) <= blind / 2:
        lines.append(
            f"- **Pose signal is weak ({helpful + recovery} useful vs {blind} blind/no-data).** "
            f"Structural pose integration would not materially improve ReID. "
            f"Next option: trajectory-continuity priors, role/position priors, or a trained "
            f"within-team ReID head."
        )
    else:
        lines.append(
            f"- **Mixed signal ({helpful + recovery} useful, {weak} weak, {blind} blind/no-data).** "
            f"Inspect the raw per-event table below to decide."
        )

    lines.extend(["", "## Per-event detail", "",
                  "| Rally | Swap frame | pred_old→pred_new | GT | N(new_pre/post, old_pre) | cost(same_body) | cost(other_body) | verdict |",
                  "|---|---:|---|---:|---:|---:|---:|---|"])
    for r in all_results:
        cs = f"{r['cost_same_body']:.3f}" if r['cost_same_body'] is not None else "—"
        co = f"{r['cost_different_body']:.3f}" if r['cost_different_body'] is not None else "—"
        lines.append(
            f"| `{r['rally_id'][:8]}` | {r['swap_frame']} | "
            f"{r['pred_old']}→{r['pred_new']} | {r['gt_track_id']} | "
            f"{r['samples_new_pre']}/{r['samples_new_post']}/{r['samples_old_pre']} | "
            f"{cs} | {co} | `{r['verdict']}` |"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    logger.info(f"\nReport written to {args.out}")


if __name__ == "__main__":
    main()
