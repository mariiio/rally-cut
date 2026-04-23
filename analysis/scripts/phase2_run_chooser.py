"""Phase 2 — offline attribution chooser (confidence-gated + positional team check).

Reads the locked baseline JSON + Phase 1.1 tracking-stability audit and re-picks
`playerTrackId` for each pipeline action using the Phase-1-derived policies:

1. **Swap-aware abstention (1.1a)** — if a candidate tid has a swap event
   within ±5 frames of the contact, drop it from the candidate pool.
2. **Low-coverage abstention (1.1a)** — if a candidate tid has <40% rally
   coverage, drop it.
3. **Contact-time positional team (1.3a)** — at the contact frame, each
   primary tid's foot-y relative to the per-rally midline determines its
   current side. `teamAssignments` is not consulted.
4. **Ball-side soft preference (1.4a)** — when `contact.courtSide` is known,
   prefer candidates on that side IF it leaves at least one candidate. Soft
   filter, not hard gate.
5. **Confidence margin gate (2.2)** — among surviving candidates, margin =
   (d_2 - d_1) / d_1. If margin < THR_MARGIN, emit `playerTrackId = None`
   (abstain — converts to `missing` in the bench metric, aligned with
   "prefer miss over wrong").

Outputs:
- `reports/attribution_rebuild/phase2_{margin}.json` — phase2 result JSON in
  baseline schema. Scored via `scripts/bench_attribution.py`.
- Stdout: Pareto table over margin thresholds.

Usage:
    uv run python scripts/phase2_run_chooser.py
    uv run python scripts/phase2_run_chooser.py --sweep 0.0,0.1,0.2,0.3
    uv run python scripts/phase2_run_chooser.py --margin 0.20 --out reports/attribution_rebuild/phase2_m020.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

from rallycut.evaluation.attribution_bench import (
    WRONG_CATEGORIES,
    aggregate,
    score_rally,
)
from rallycut.evaluation.db import get_connection

BASELINE_PATH = (
    Path(__file__).resolve().parent.parent
    / "reports"
    / "attribution_rebuild"
    / "baseline_2026_04_24.json"
)
TRACKING_AUDIT_PATH = (
    Path(__file__).resolve().parent.parent
    / "reports"
    / "attribution_rebuild"
    / "phase1_1_tracking_stability.json"
)
OUT_DIR = (
    Path(__file__).resolve().parent.parent
    / "reports"
    / "attribution_rebuild"
)

SWAP_WINDOW = 5
COVERAGE_MIN = 0.40
DEFAULT_SWEEP = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]


def _fetch_positions(rally_ids: list[str]) -> dict[str, list]:
    positions_by_rally: dict[str, list] = {}
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT rally_id, positions_json FROM player_tracks WHERE rally_id = ANY(%s)",
            (rally_ids,),
        )
        for rid, pos in cur.fetchall():
            positions_by_rally[rid] = pos or []
    return positions_by_rally


def _compute_midline(positions: list[dict]) -> float:
    foot_ys: list[float] = []
    for p in positions:
        if p.get("trackId") in (1, 2, 3, 4) and p.get("frameNumber", 0) < 15:
            foot_ys.append(p["y"] + p.get("height", 0) / 2)
    if len(foot_ys) < 4:
        foot_ys = [
            p["y"] + p.get("height", 0) / 2
            for p in positions
            if p.get("trackId") in (1, 2, 3, 4)
        ]
    return median(foot_ys) if foot_ys else 0.5


def _rally_start_team(positions: list[dict], midline: float) -> dict[int, str]:
    """Return {pid: 'near' | 'far'} based on rally-start positions (first 15
    frames). This is the player's physical team — stable for the rally, unlike
    per-frame positional which shifts when players cross the midline during
    play."""
    from collections import defaultdict
    per_tid: dict[int, list[float]] = defaultdict(list)
    for p in positions:
        if p.get("trackId") in (1, 2, 3, 4) and p.get("frameNumber", 0) < 15:
            per_tid[p["trackId"]].append(p["y"] + p.get("height", 0) / 2)
    if len(per_tid) < 4:
        per_tid = defaultdict(list)
        for p in positions:
            if p.get("trackId") in (1, 2, 3, 4):
                per_tid[p["trackId"]].append(p["y"] + p.get("height", 0) / 2)
    return {
        pid: ("near" if median(ys) > midline else "far")
        for pid, ys in per_tid.items() if ys
    }


def _side_at_frame(
    positions: list[dict],
    pid: int,
    frame: int,
    midline: float,
) -> str | None:
    """Foot-y side for a specific pid at a specific frame (nearest ±3f)."""
    candidates = [
        p for p in positions
        if p.get("trackId") == pid
        and p.get("frameNumber") is not None
        and abs(p["frameNumber"] - frame) <= 3
    ]
    if not candidates:
        return None
    best = min(candidates, key=lambda p: abs(p["frameNumber"] - frame))
    foot_y = best["y"] + best.get("height", 0) / 2
    return "near" if foot_y > midline else "far"


def _has_swap_near(
    swap_events: list[dict],
    pid: int,
    frame: int,
    window: int = SWAP_WINDOW,
) -> bool:
    return any(
        s["tid"] == pid and abs(s["frame"] - frame) <= window
        for s in swap_events
    )


def _is_low_coverage(coverage_map: dict, pid: int) -> bool:
    entry = coverage_map.get(str(pid)) or coverage_map.get(pid) or {}
    return entry.get("coverage_ratio", 1.0) < COVERAGE_MIN


def choose(
    contact: dict,
    action: dict,
    positions: list[dict],
    midline: float,
    rally_start_team: dict[int, str],
    swap_events: list[dict],
    coverage: dict,
    margin_thr: float,
    use_ball_side: bool = True,
    use_filters: bool = True,
) -> tuple[int | None, float | None, str]:
    """Corrective overlay chooser. Starts from pipeline's existing action pick,
    overrides only on specific signals:

    1. Swap or low-coverage on pipeline's pick → abstain.
    2. Rank-1 candidate's rally-start team disagrees with ball_side AND
       rank-2's agrees → override to rank-2 (label-flip correction).
    3. Margin gate on candidate distances.
    """
    frame = contact.get("frame")
    pipeline_pid = action.get("playerTrackId")
    candidates = contact.get("playerCandidates") or []

    def _unpack(c):
        if isinstance(c, list):
            return int(c[0]), float(c[1])
        if isinstance(c, dict):
            return int(c.get("playerTrackId")), float(c.get("distance"))
        return None, None

    ranked = [
        _unpack(c) for c in candidates
        if _unpack(c)[0] is not None and _unpack(c)[1] is not None
    ]
    ranked.sort(key=lambda x: x[1])

    # Swap-aware + coverage abstention on the pipeline's current pick
    if use_filters and pipeline_pid is not None:
        if _has_swap_near(swap_events, pipeline_pid, frame):
            return None, None, "swap_near_pipeline_pick"
        if _is_low_coverage(coverage, pipeline_pid):
            return None, None, "low_coverage_pipeline_pick"

    # Label-flip correction: rank-1 vs ball_side check
    override_pid = None
    if use_ball_side and len(ranked) >= 2:
        ball_side = contact.get("courtSide")
        if ball_side in ("near", "far"):
            rank1_pid, rank1_dist = ranked[0]
            rank2_pid, rank2_dist = ranked[1]
            rank1_side = rally_start_team.get(rank1_pid)
            rank2_side = rally_start_team.get(rank2_pid)
            # Override only when rank-1 disagrees with ball AND rank-2 agrees.
            if (
                rank1_side is not None
                and rank2_side is not None
                and rank1_side != ball_side
                and rank2_side == ball_side
            ):
                override_pid = rank2_pid
                # Apply abstention filters to override candidate too
                if use_filters:
                    if _has_swap_near(swap_events, rank2_pid, frame):
                        return None, None, "swap_near_override_pick"
                    if _is_low_coverage(coverage, rank2_pid):
                        return None, None, "low_coverage_override_pick"

    # Margin gate on base candidates (uses raw distances, not team-filtered)
    margin = None
    if len(ranked) >= 2:
        d1, d2 = ranked[0][1], ranked[1][1]
        margin = (d2 - d1) / d1 if d1 > 1e-9 else 1.0
        if margin < margin_thr:
            return None, margin, f"margin_below_{margin_thr}"

    if override_pid is not None:
        return override_pid, margin, "label_flip_override"
    if pipeline_pid is not None:
        return pipeline_pid, margin, "pipeline_pick_kept"
    # Fallback to rank-1 if no pipeline pick
    if ranked:
        return ranked[0][0], margin, "fallback_closest"
    return None, None, "no_candidates"


def run_chooser(
    baseline: dict,
    tracking_audit: dict,
    positions_by_rally: dict[str, list],
    margin_thr: float,
    use_ball_side: bool = True,
    use_filters: bool = True,
) -> dict:
    """Rebuild baseline-schema output with new chooser picks."""
    out_rallies = []
    abstain_reasons: dict[str, int] = defaultdict(int)
    total_actions = 0

    for rally in baseline["rallies"]:
        rid = rally["rally_id"]
        positions = positions_by_rally.get(rid, [])
        midline = _compute_midline(positions)
        rally_start_team = _rally_start_team(positions, midline)
        rally_audit = tracking_audit["per_rally"].get(rid, {})
        swap_events = rally_audit.get("swap_events") or []
        coverage = rally_audit.get("coverage") or {}

        # Build contact-by-frame lookup
        contacts_by_frame = {
            c["frame"]: c for c in rally.get("pipeline_contacts", [])
        }

        new_actions = []
        for action in rally.get("pipeline_actions", []):
            total_actions += 1
            frame = action["frame"]
            contact = contacts_by_frame.get(frame) or action
            new_pid, margin, reason = choose(
                contact, action, positions, midline, rally_start_team,
                swap_events, coverage, margin_thr,
                use_ball_side=use_ball_side, use_filters=use_filters,
            )
            if new_pid is None:
                abstain_reasons[reason] += 1
            new_action = dict(action)
            new_action["playerTrackId"] = new_pid
            new_action["phase2_margin"] = (
                round(margin, 3) if margin is not None else None
            )
            new_action["phase2_reason"] = reason
            new_actions.append(new_action)

        new_rally = dict(rally)
        new_rally["pipeline_actions"] = new_actions
        # Re-score against GT using new actions
        scored = score_rally(new_rally)
        new_rally["matches"] = scored["matches"]
        new_rally["rally_totals"] = scored["rally_totals"]
        out_rallies.append(new_rally)

    out = {
        "generated_at": "2026-04-24",
        "source": (
            f"Phase 2 offline chooser (margin_thr={margin_thr}) over baseline_2026_04_24.json"
        ),
        "phase2_config": {
            "margin_thr": margin_thr,
            "swap_window": SWAP_WINDOW,
            "coverage_min": COVERAGE_MIN,
        },
        "rallies": out_rallies,
        "abstain_reasons": dict(abstain_reasons),
        "total_actions": total_actions,
    }
    return out


def _print_summary(agg: dict, label: str) -> None:
    c = agg["combined"]["counts"]
    r = agg["combined"]["rates"]
    wrong = sum(c[k] for k in WRONG_CATEGORIES)
    print(
        f"  {label:14s} correct={c['correct']:>3d} ({r['correct_rate']:5.1%})  "
        f"wrong={wrong:>3d} ({r['wrong_rate']:5.1%})  "
        f"missing={c['missing']:>3d} ({r['missing_rate']:5.1%})  "
        f"abstain={c['abstained']:>3d} ({r['abstained_rate']:5.1%})"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin", type=float, default=None,
                    help="Single margin threshold (omit to sweep)")
    ap.add_argument("--sweep", type=str, default=None,
                    help="Comma-separated margin thresholds")
    ap.add_argument("--no-ball-side", action="store_true",
                    help="Disable ball-side soft filter")
    ap.add_argument("--no-filters", action="store_true",
                    help="Disable swap + coverage abstention (floor: pure closest)")
    ap.add_argument("--out", type=Path, default=None,
                    help="Single-run output path (only with --margin)")
    args = ap.parse_args()

    baseline = json.loads(BASELINE_PATH.read_text())
    tracking_audit = json.loads(TRACKING_AUDIT_PATH.read_text())
    rally_ids = [r["rally_id"] for r in baseline["rallies"]]
    positions_by_rally = _fetch_positions(rally_ids)

    # Baseline agg for comparison
    baseline_agg = aggregate(baseline["rallies"])
    print("BASELINE:")
    _print_summary(baseline_agg, "baseline")
    print()

    use_ball_side = not args.no_ball_side
    use_filters = not args.no_filters
    if args.margin is not None:
        result = run_chooser(
            baseline, tracking_audit, positions_by_rally, args.margin,
            use_ball_side=use_ball_side, use_filters=use_filters,
        )
        agg = aggregate(result["rallies"])
        print(f"PHASE 2 margin={args.margin}:")
        _print_summary(agg, f"m={args.margin}")
        out = args.out or (OUT_DIR / f"phase2_m{int(args.margin * 100):03d}.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        print(f"\nwrote {out}")
        print(f"abstain reasons: {result['abstain_reasons']}")
        return 0

    sweep = [float(x) for x in args.sweep.split(",")] if args.sweep else DEFAULT_SWEEP
    print(f"SWEEP over margin thresholds: {sweep}")
    print()
    print(f"{'margin':>8s} {'correct':>10s} {'wrong':>10s} {'missing':>10s} {'abstain':>10s}")
    print(f"{'baseline':>8s} {baseline_agg['combined']['counts']['correct']:>4d} "
          f"({baseline_agg['combined']['rates']['correct_rate']:5.1%})  "
          f"{sum(baseline_agg['combined']['counts'][k] for k in WRONG_CATEGORIES):>4d} "
          f"({baseline_agg['combined']['rates']['wrong_rate']:5.1%})  "
          f"{baseline_agg['combined']['counts']['missing']:>4d} "
          f"({baseline_agg['combined']['rates']['missing_rate']:5.1%})  "
          f"{0:>4d} (  0.0%)")
    best = None
    for mt in sweep:
        result = run_chooser(
            baseline, tracking_audit, positions_by_rally, mt,
            use_ball_side=use_ball_side, use_filters=use_filters,
        )
        agg = aggregate(result["rallies"])
        c = agg["combined"]["counts"]
        r = agg["combined"]["rates"]
        w = sum(c[k] for k in WRONG_CATEGORIES)
        print(f"{mt:>8.2f} {c['correct']:>4d} ({r['correct_rate']:5.1%})  "
              f"{w:>4d} ({r['wrong_rate']:5.1%})  "
              f"{c['missing']:>4d} ({r['missing_rate']:5.1%})  "
              f"{c['abstained']:>4d} ({r['abstained_rate']:5.1%})")
        out = OUT_DIR / f"phase2_m{int(mt * 100):03d}.json"
        out.write_text(json.dumps(result, indent=2))
        # Track best: wrong_rate minimum subject to correct_rate not collapsing
        if best is None or r["wrong_rate"] < best[1]:
            best = (mt, r["wrong_rate"], r["correct_rate"], r["missing_rate"] + r["abstained_rate"])
    print()
    if best:
        print(f"BEST margin={best[0]}: wrong={best[1]:.1%}, correct={best[2]:.1%}, miss+abs={best[3]:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
