"""Feasibility probe — does trajectory continuity add signal at swap events?

Idea: at each swap, extrapolate both pred_new's and pred_old's pre-swap
trajectories forward to the swap frame, then see which extrapolation matches
pred_new's ACTUAL post-swap position.

    extrap_new_pre @ swap = where pred_new would be IF it kept its own path
    extrap_old_pre @ swap = where the target body would be IF pred_old had
                             continued tracking it (the "correct re-association" hypothesis)
    actual_post = where pred_new actually is after the swap

Verdicts:

    trajectory_says_no_swap           dist(actual, extrap_new_pre) << dist(actual, extrap_old_pre)
                                       → pred_new is continuing its own trajectory; the
                                         global-identity match to the other GT was wrong
                                         (typical Hungarian crossover near convergence)
    trajectory_says_swap_was_recovery dist(actual, extrap_old_pre) << dist(actual, extrap_new_pre)
                                       → pred_new jumped to where pred_old's body was going,
                                         so the "swap" we flagged is actually a correct
                                         re-association — pred_old died and pred_new took over
    trajectory_blind                  |Δ| < 0.02 — both extrapolations converge at swap frame
    no_trajectory_data                insufficient pre-swap or post-swap positions

No YOLO-Pose, no feature extraction — pure kinematics on stored predictions.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np

from rallycut.evaluation.tracking.db import load_labeled_rallies
from rallycut.tracking.player_tracker import PlayerPosition

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("traj-probe")

DEFAULT_RALLIES = [
    "fad29c31-6e2a-4a8d-86f1-9064b2f1f425",
    "209be896-b680-44dc-bf31-693f4e287149",
    "d724bbf0-bd0c-44e8-93d5-135aa07df5a1",
]

WINDOW = 10              # frames pre/post to use
DELTA_HAS_SIGNAL = 0.03  # normalised distance: ~3% of frame width
DELTA_BLIND = 0.02


def _extrapolate_linear(
    positions: list[tuple[int, float, float]],  # [(frame, x, y), ...]
    target_frame: int,
) -> tuple[float, float] | None:
    """Linear extrapolation of (x, y) to target_frame from the last ≤ 5 samples.

    Returns (x, y) or None if we have < 2 samples.
    """
    if len(positions) < 2:
        return None
    # Use the most recent up to 5 positions for velocity estimation
    tail = sorted(positions, key=lambda t: t[0])[-5:]
    frames = np.array([p[0] for p in tail], dtype=np.float64)
    xs = np.array([p[1] for p in tail], dtype=np.float64)
    ys = np.array([p[2] for p in tail], dtype=np.float64)
    # Fit a line through the tail (minimum least squares; constant velocity model).
    if frames[-1] == frames[0]:
        return float(xs[-1]), float(ys[-1])
    slope_x = (xs[-1] - xs[0]) / (frames[-1] - frames[0])
    slope_y = (ys[-1] - ys[0]) / (frames[-1] - frames[0])
    dt = target_frame - frames[-1]
    return float(xs[-1] + slope_x * dt), float(ys[-1] + slope_y * dt)


def _dist(a: tuple[float, float] | None, b: tuple[float, float] | None) -> float | None:
    if a is None or b is None:
        return None
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _positions_for_pred(
    predictions: list[PlayerPosition],
    pred_id: int,
    frame_range: range,
) -> list[tuple[int, float, float]]:
    out: list[tuple[int, float, float]] = []
    for p in predictions:
        if p.track_id == pred_id and p.frame_number in frame_range:
            out.append((p.frame_number, p.x, p.y))
    return out


def _actual_position_of_pred(
    predictions: list[PlayerPosition],
    pred_id: int,
    target_frame: int,
) -> tuple[float, float] | None:
    for p in predictions:
        if p.track_id == pred_id and p.frame_number == target_frame:
            return (p.x, p.y)
    # fall back: nearest frame within ±2
    best = None
    best_df = 3
    for p in predictions:
        if p.track_id != pred_id:
            continue
        df = abs(p.frame_number - target_frame)
        if df < best_df:
            best_df = df
            best = (p.x, p.y)
    return best


def _classify(
    dist_to_new_pre: float | None,
    dist_to_old_pre: float | None,
) -> str:
    if dist_to_new_pre is None or dist_to_old_pre is None:
        return "no_trajectory_data"
    delta = dist_to_old_pre - dist_to_new_pre  # +ve → actual closer to new_pre (no swap)
    if delta >= DELTA_HAS_SIGNAL:
        return "trajectory_says_no_swap"
    if delta <= -DELTA_HAS_SIGNAL:
        return "trajectory_says_swap_was_recovery"
    if abs(delta) < DELTA_BLIND:
        return "trajectory_blind"
    return "trajectory_weak_signal"


def _load_swap_events(audit_path: Path) -> list[dict]:
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


def run_for_rally(rally_id: str, audit_dir: Path) -> list[dict]:
    audit_path = audit_dir / f"{rally_id}.json"
    if not audit_path.exists():
        return []
    events = _load_swap_events(audit_path)
    if not events:
        return []
    rallies = load_labeled_rallies(rally_id=rally_id)
    if not rallies or rallies[0].predictions is None:
        return []
    predictions = rallies[0].predictions.positions

    results = []
    for ev in events:
        swap_frame = ev["swap_frame"]
        pred_new = ev["pred_new"]
        pred_old = ev["pred_old"]

        pre_range = range(max(0, swap_frame - WINDOW), swap_frame)
        new_pre_pts = _positions_for_pred(predictions, pred_new, pre_range)
        old_pre_pts = _positions_for_pred(predictions, pred_old, pre_range)

        extrap_new = _extrapolate_linear(new_pre_pts, swap_frame)
        extrap_old = _extrapolate_linear(old_pre_pts, swap_frame)
        actual = _actual_position_of_pred(predictions, pred_new, swap_frame)

        d_new = _dist(actual, extrap_new)
        d_old = _dist(actual, extrap_old)
        verdict = _classify(d_new, d_old)

        results.append({
            "rally_id": rally_id,
            "swap_frame": swap_frame,
            "pred_old": pred_old,
            "pred_new": pred_new,
            "gt_track_id": ev["gt_track_id"],
            "samples_new_pre": len(new_pre_pts),
            "samples_old_pre": len(old_pre_pts),
            "actual_xy": actual,
            "extrap_new_pre_xy": extrap_new,
            "extrap_old_pre_xy": extrap_old,
            "dist_to_new_pre": d_new,
            "dist_to_old_pre": d_old,
            "verdict": verdict,
        })
        logger.info(
            f"  swap@{swap_frame} pred {pred_old}→{pred_new} on GT {ev['gt_track_id']}: "
            f"dist(new_pre)={d_new if d_new is None else f'{d_new:.3f}'}  "
            f"dist(old_pre)={d_old if d_old is None else f'{d_old:.3f}'}  → {verdict}"
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=Path("reports/tracking_audit"))
    parser.add_argument("--out", type=Path, default=Path("reports/tracking_audit/reid_debug/trajectory_signal.md"))
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

    all_results: list[dict] = []
    for idx, rid in enumerate(rally_ids, start=1):
        logger.info(f"[{idx}/{len(rally_ids)}] {rid[:8]}")
        all_results.extend(run_for_rally(rid, audit_dir=args.audit_dir))

    counts = Counter(r["verdict"] for r in all_results)
    total = len(all_results)

    lines = [
        "# Trajectory-continuity feasibility probe at swap events",
        "",
        f"Probed **{total}** swap events across {len(rally_ids)} rally(s) "
        f"using {WINDOW}-frame pre-swap extrapolation.",
        "",
        "## Verdict counts",
        "",
        "| Verdict | Count | Share | What it means |",
        "|---|---:|---:|---|",
    ]
    descriptions = {
        "trajectory_says_no_swap":
            f"actual ≈ pred_new's own extrapolation (Δ ≥ {DELTA_HAS_SIGNAL}); "
            "Hungarian match to the wrong GT — trajectory prior would have caught it.",
        "trajectory_says_swap_was_recovery":
            f"actual ≈ pred_old's extrapolation (Δ ≥ {DELTA_HAS_SIGNAL}); "
            "the event we flagged as a swap is actually a correct re-association.",
        "trajectory_blind":
            f"both extrapolations converge at swap frame (|Δ| < {DELTA_BLIND}); "
            "players genuinely crossing paths — trajectory can't decide.",
        "trajectory_weak_signal":
            f"{DELTA_BLIND} ≤ |Δ| < {DELTA_HAS_SIGNAL} — marginal, would need combining with another signal.",
        "no_trajectory_data":
            "insufficient pre-swap samples for either pred (< 2 positions).",
    }
    for v, desc in descriptions.items():
        n = counts.get(v, 0)
        pct = f"{100 * n / total:.1f}%" if total else "0.0%"
        lines.append(f"| `{v}` | {n} | {pct} | {desc} |")

    lines.extend(["", "## Interpretation", ""])
    helpful = counts.get("trajectory_says_no_swap", 0)
    recovery = counts.get("trajectory_says_swap_was_recovery", 0)
    blind = counts.get("trajectory_blind", 0) + counts.get("no_trajectory_data", 0)
    weak = counts.get("trajectory_weak_signal", 0)

    if (helpful + recovery) >= total * 0.3 and (helpful + recovery) >= 5:
        lines.append(
            f"- **Trajectory continuity carries real signal on {helpful + recovery}/{total} events "
            f"({helpful} no-swap + {recovery} recovery).** "
            f"Structural integration is worth prototyping: add a trajectory-continuity term to "
            f"`global_identity` cost — penalise assignments that require >> expected kinematics jump. "
            f"Eval-gate on HOTA/MOTA/oracles before merging."
        )
    elif (helpful + recovery) <= blind / 2:
        lines.append(
            f"- **Trajectory signal is weak ({helpful + recovery} useful vs {blind} blind/no-data).** "
            f"At the swap frame itself, players are too close for position alone to decide. "
            f"Pivot to learned within-team embedding or multi-signal fusion."
        )
    else:
        lines.append(
            f"- **Mixed signal ({helpful + recovery} useful, {weak} weak, {blind} blind).** "
            f"Worth combining with HSV: maybe the HSV-blind events become decidable when "
            f"trajectory breaks the tie."
        )
    if recovery > 0:
        lines.append(
            f"- **Note**: {recovery} event(s) classified as `trajectory_says_swap_was_recovery` — "
            f"our 'pred-exchange swap' count may include some CORRECT tracker re-associations. "
            f"Worth sanity-checking these before treating them all as failures."
        )

    lines.extend(["", "## Per-event detail", "",
                  "| Rally | Swap frame | pred_old→pred_new | GT | N(new_pre/old_pre) | dist(new_pre) | dist(old_pre) | verdict |",
                  "|---|---:|---|---:|---:|---:|---:|---|"])
    for r in all_results:
        dn = f"{r['dist_to_new_pre']:.3f}" if r['dist_to_new_pre'] is not None else "—"
        do = f"{r['dist_to_old_pre']:.3f}" if r['dist_to_old_pre'] is not None else "—"
        lines.append(
            f"| `{r['rally_id'][:8]}` | {r['swap_frame']} | {r['pred_old']}→{r['pred_new']} | "
            f"{r['gt_track_id']} | {r['samples_new_pre']}/{r['samples_old_pre']} | "
            f"{dn} | {do} | `{r['verdict']}` |"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    logger.info(f"\nReport written to {args.out}")


if __name__ == "__main__":
    main()
