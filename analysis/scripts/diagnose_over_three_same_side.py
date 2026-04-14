"""Net-cross diagnostic on `over_three_same_side` rallies.

The 2026-04-14 audit (`action_anomaly_audit_2026_04_14.md`) flagged 97
rallies in the clean GT pool (38.3%) where the predicted action sequence
has a run of more than 3 consecutive contacts on the same court side —
a structural impossibility in beach volleyball outside of stacked
blocks. Only 16 of those 97 are explained by missed-block detection;
the remaining 80 are unexplained.

This script walks each over_three rally and, for every consecutive
predicted contact pair *inside the long same-side run*, calls
`ball_crossed_net()` (the same function `classify_rally` uses to
decide possession changes) and bins the result:

- `cross_detected`: `ball_crossed_net` returned True. The signal *was*
  there but downstream possession state didn't reset → grammar /
  classify_rally logic bug.
- `no_signal`: `ball_crossed_net` returned None (too few ball
  positions in the inter-contact window). Ball-tracking dropout.
- `no_cross`: `ball_crossed_net` returned False. Either the run is
  legitimate or the ball Y displacement threshold is mis-tuned for
  this video.

Cross-references each pair with GT: if the GT actions at the two
nearest GT frames are on opposite sides, we *know* a real cross
happened, regardless of what `ball_crossed_net` returned. That lets us
separate "cross_detected_but_ignored" (grammar bug) from
"cross_present_but_undetected" (signal bug).

Output is stdout + a markdown summary.

Usage:
    cd analysis
    uv run python scripts/diagnose_over_three_same_side.py \\
        --skip-session 6f599a0e-b8ea-4bf0-a331-ce7d9ef88164 \\
        --output outputs/over_three_diagnostic_2026_04_14.md
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from rich.console import Console
from rich.table import Table

from rallycut.evaluation.db import get_connection
from rallycut.tracking.ball_tracker import BallPosition
from rallycut.tracking.contact_detector import ball_crossed_net

# Import the audit's loader + detectors so we use the same definitions.
from scripts.audit_action_sequence_anomalies import (
    _load_rallies,
    _run_detectors,
    _tag_low_quality,
)

console = Console()


@dataclass
class PairResult:
    """One same-side consecutive predicted-contact pair inside a run."""

    rally_id: str
    video_id: str
    rally_order: int
    from_frame: int
    to_frame: int
    side: str
    cross_detected: bool | None
    n_ball_positions: int
    y_delta: float | None
    gt_says_crossed: bool | None  # True/False/None depending on GT availability
    pred_actions: tuple[str, str]


def _load_ball_positions(rally_id: str) -> list[BallPosition] | None:
    """Read raw ball positions for a rally from the DB."""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT ball_positions_json, court_split_y "
            "FROM player_tracks WHERE rally_id = %s",
            (rally_id,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    raw: Any = row[0]
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not raw:
        return None
    out: list[BallPosition] = []
    for bp in raw:
        if not isinstance(bp, dict):
            continue
        out.append(BallPosition(
            frame_number=int(bp["frameNumber"]),
            x=float(bp["x"]),
            y=float(bp["y"]),
            confidence=float(bp.get("confidence", 1.0)),
        ))
    return out


def _net_y_for_rally(ball_positions: list[BallPosition]) -> float:
    """Estimate net_y as the median Y across ball positions.

    `classify_rally` derives net_y from ball trajectory in production
    (see action_classifier ContactSequence). Median Y is a stable
    proxy that doesn't require re-running the trajectory analyzer.
    """
    ys = [bp.y for bp in ball_positions if bp.confidence >= 0.3]
    if not ys:
        return 0.5
    return statistics.median(ys)


def _gt_says_crossed(
    gt_actions: list[dict[str, Any]],
    from_frame: int,
    to_frame: int,
    pred_side: str,
    tol: int = 8,
) -> bool | None:
    """Did GT have an opposite-side contact between from_frame and to_frame?

    If GT has actions inside (from_frame ± tol, to_frame ± tol) on the
    opposite side from pred_side, the ball *really did* cross — the
    pipeline's same-side run is masking it. Returns None if no GT
    contact lies inside the window.
    """
    opposite = "far" if pred_side == "near" else "near"
    found_opposite = False
    found_any = False
    for g in gt_actions:
        gf = int(g.get("frame", -1))
        if from_frame - tol <= gf <= to_frame + tol:
            found_any = True
            # GT doesn't store courtSide directly; infer from ballY relative
            # to median GT ballY. Use the simpler proxy: ballY < 0.5 ≈ far.
            by = g.get("ballY")
            if by is None:
                continue
            inferred = "far" if float(by) < 0.5 else "near"
            if inferred == opposite:
                found_opposite = True
                break
    if not found_any:
        return None
    return found_opposite


def _run_y_delta(
    ball_positions: list[BallPosition], from_frame: int, to_frame: int,
) -> tuple[float | None, int]:
    """Return (y_delta_or_None, n_ball_positions_in_range) for diagnostic."""
    in_range = [
        bp for bp in ball_positions
        if from_frame < bp.frame_number < to_frame
    ]
    n = len(in_range)
    if n < 4:
        return None, n
    half = max(2, n // 2)
    start_median = statistics.median(bp.y for bp in in_range[:half])
    end_median = statistics.median(bp.y for bp in in_range[-half:])
    return abs(end_median - start_median), n


def _walk_run(
    actions: list[dict[str, Any]],
    ball_positions: list[BallPosition],
    gt_actions: list[dict[str, Any]],
    rally_id: str,
    video_id: str,
    rally_order: int,
    net_y: float,
) -> list[PairResult]:
    """Walk a rally's predicted actions; emit a PairResult for every
    consecutive same-side same-action-type-allowed pair inside any
    same-side run of length > 3 (excluding blocks)."""
    out: list[PairResult] = []
    # Identify maximal same-side runs (excluding blocks).
    runs: list[list[int]] = []
    cur_run: list[int] = []
    cur_side: str | None = None
    for i, a in enumerate(actions):
        side = a.get("courtSide")
        if side not in ("near", "far") or a.get("action") == "block":
            if len(cur_run) > 3:
                runs.append(cur_run)
            cur_run = []
            cur_side = None
            continue
        if side != cur_side:
            if len(cur_run) > 3:
                runs.append(cur_run)
            cur_side = side
            cur_run = [i]
        else:
            cur_run.append(i)
    if len(cur_run) > 3:
        runs.append(cur_run)

    for run in runs:
        side = cast(str, actions[run[0]].get("courtSide"))
        for j in range(1, len(run)):
            a_prev = actions[run[j - 1]]
            a_cur = actions[run[j]]
            from_frame = int(a_prev.get("frame", -1))
            to_frame = int(a_cur.get("frame", -1))
            cross = ball_crossed_net(ball_positions, from_frame, to_frame, net_y)
            y_delta, n_in_range = _run_y_delta(
                ball_positions, from_frame, to_frame,
            )
            gt_crossed = _gt_says_crossed(
                gt_actions, from_frame, to_frame, side,
            )
            out.append(PairResult(
                rally_id=rally_id,
                video_id=video_id,
                rally_order=rally_order,
                from_frame=from_frame,
                to_frame=to_frame,
                side=side,
                cross_detected=cross,
                n_ball_positions=n_in_range,
                y_delta=y_delta,
                gt_says_crossed=gt_crossed,
                pred_actions=(
                    cast(str, a_prev.get("action")),
                    cast(str, a_cur.get("action")),
                ),
            ))
    return out


def _bucket(p: PairResult) -> str:
    """Categorize a pair by what likely went wrong."""
    if p.cross_detected is True:
        return "cross_detected_but_ignored"
    if p.cross_detected is None:
        return "no_signal_ball_dropout"
    # cross_detected is False
    if p.gt_says_crossed is True:
        return "cross_missed_by_threshold"
    if p.gt_says_crossed is False:
        return "legit_same_side_run"
    return "no_signal_no_gt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-session",
        default="6f599a0e-b8ea-4bf0-a331-ce7d9ef88164",
        help="Session id of bad-tracking videos to exclude (default: poor)",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("outputs/over_three_diagnostic_2026_04_14.md"),
    )
    args = parser.parse_args()

    console.print("[bold]Loading rallies…[/bold]")
    rallies = _load_rallies(skip_session_id=args.skip_session)
    _tag_low_quality(rallies)
    _run_detectors(rallies)
    clean = [r for r in rallies if not r.low_quality]
    over3 = [r for r in clean if r.flags.get("over_three_same_side")]
    console.print(f"  {len(clean)} clean rallies, {len(over3)} with over_three_same_side")

    pairs: list[PairResult] = []
    for r in over3:
        balls = _load_ball_positions(r.rally_id)
        if not balls:
            continue
        net_y = _net_y_for_rally(balls)
        pairs.extend(_walk_run(
            r.pred_actions, balls, r.gt_actions,
            r.rally_id, r.video_id, r.rally_order, net_y,
        ))

    if not pairs:
        console.print("[yellow]No same-side pairs found.[/yellow]")
        return

    # Bucket histogram
    counter = Counter(_bucket(p) for p in pairs)
    n_total = len(pairs)
    table = Table(title=f"Net-cross diagnostic — {n_total} same-side pairs in {len(over3)} rallies")
    table.add_column("bucket")
    table.add_column("n", justify="right")
    table.add_column("%", justify="right")
    for bucket in [
        "cross_detected_but_ignored",
        "no_signal_ball_dropout",
        "cross_missed_by_threshold",
        "legit_same_side_run",
        "no_signal_no_gt",
    ]:
        n = counter.get(bucket, 0)
        pct = n / n_total * 100.0 if n_total else 0.0
        table.add_row(bucket, str(n), f"{pct:.1f}%")
    console.print(table)

    # Per-rally counts (top 10 most affected)
    by_rally: dict[str, list[PairResult]] = {}
    for p in pairs:
        by_rally.setdefault(p.rally_id, []).append(p)
    top_affected = sorted(
        by_rally.items(), key=lambda kv: -len(kv[1])
    )[:10]
    console.print("\n[bold]Top 10 rallies by # of same-side pairs:[/bold]")
    for rid, ps in top_affected:
        b = Counter(_bucket(p) for p in ps)
        console.print(
            f"  {rid[:8]} ({len(ps)} pairs)  "
            f"detected_ignored={b['cross_detected_but_ignored']}  "
            f"no_signal={b['no_signal_ball_dropout']}  "
            f"threshold_miss={b['cross_missed_by_threshold']}  "
            f"legit={b['legit_same_side_run']}"
        )

    # Markdown output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Net-cross diagnostic — over_three_same_side rallies")
    lines.append("")
    lines.append(f"- Clean rallies: **{len(clean)}**")
    lines.append(f"- With over_three_same_side: **{len(over3)}**")
    lines.append(f"- Same-side consecutive pairs analyzed: **{n_total}**")
    lines.append("")
    lines.append("## Bucket histogram")
    lines.append("")
    lines.append("| Bucket | n | % | Implication |")
    lines.append("|---|---:|---:|---|")
    impls = {
        "cross_detected_but_ignored":
            "Grammar/possession-state bug. `ball_crossed_net` saw the cross — `classify_rally`'s state didn't reset.",
        "no_signal_ball_dropout":
            "Ball-tracking dropout. WASB lost the ball between contacts; cannot infer crossing.",
        "cross_missed_by_threshold":
            "Threshold mis-tuned: GT confirms the cross happened but Y-displacement < `_NET_CROSSING_Y_THRESHOLD`.",
        "legit_same_side_run":
            "GT also has same-side touches — the run is real (legal long defensive sequence or block).",
        "no_signal_no_gt":
            "Inconclusive (no signal AND no GT in the window).",
    }
    for bucket, impl in impls.items():
        n = counter.get(bucket, 0)
        pct = n / n_total * 100.0 if n_total else 0.0
        lines.append(f"| `{bucket}` | {n} | {pct:.1f}% | {impl} |")
    lines.append("")
    lines.append("## Per-rally pair detail")
    lines.append("")
    for rid, ps in sorted(by_rally.items(), key=lambda kv: -len(kv[1])):
        first = ps[0]
        lines.append(
            f"### {first.video_id[:8]} · rally #{first.rally_order} (`{rid[:8]}`)"
        )
        for p in ps:
            lines.append(
                f"- f{p.from_frame}→f{p.to_frame} {p.pred_actions[0]}→{p.pred_actions[1]} "
                f"on {p.side}: cross={p.cross_detected} "
                f"(n_ball={p.n_ball_positions}, y_delta="
                f"{'NA' if p.y_delta is None else f'{p.y_delta:.3f}'}) "
                f"· gt_crossed={p.gt_says_crossed} → **{_bucket(p)}**"
            )
        lines.append("")

    args.output.write_text("\n".join(lines))
    console.print(f"\n[green]Wrote {args.output}[/green]")


if __name__ == "__main__":
    main()
