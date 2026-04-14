"""Sweep alternative `ball_crossed_net` signal variants on labeled pairs.

The 2026-04-14 net-cross diagnostic showed the current Y-displacement
threshold has 105 FN + 73 FP across 500 same-side pairs in the GT pool.
This script measures whether a richer signal — adding a "ball Y line
crossed `net_y`" boolean alongside the existing Y-delta — can reduce
both error types simultaneously without regressing the legit-same-side
positive class.

Variants measured (cross_predicted = ...):

A. **current**: |y_delta| > τ_delta
B. **line_cross**: any consecutive pair of confident frames has the ball
   crossing net_y (sign change in `y - net_y`)
C. **n_frames_opposite**: ≥N confident frames in the window are on the
   opposite side of net_y from the trajectory's starting frame
D. **delta_or_line**: A OR B
E. **delta_and_line**: A AND B
F. **delta_or_n3**: A OR (n_frames_opposite ≥ 3)

Ground truth pool:
- Positives (real cross): every (gt_action[i], gt_action[i+1]) pair where
  the two GT actions are on opposite sides (per a per-rally net_y
  derived from the rally's ball trajectory).
- Negatives (no cross): every same-side GT-confirmed pair from the
  over_three diagnostic where `gt_says_crossed=False`.

For each variant we compute TP, FP, FN, TN, precision, recall, F1.
Aim: maximize F1 while keeping FN ≤ baseline.

Usage:
    cd analysis
    uv run python scripts/sweep_net_crossing_signal.py
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any

from rich.console import Console
from rich.table import Table

from rallycut.tracking.ball_tracker import BallPosition
from scripts.audit_action_sequence_anomalies import (
    _load_rallies,
    _run_detectors,
    _tag_low_quality,
)
from scripts.diagnose_over_three_same_side import (
    _load_ball_positions,
    _net_y_for_rally,
)

console = Console()


@dataclass
class Pair:
    """A pair of contacts to evaluate the cross signal on."""

    rally_id: str
    from_frame: int
    to_frame: int
    is_real_cross: bool  # ground-truth label
    ball_positions_in_window: list[BallPosition]
    net_y: float
    start_y: float  # ball y at first frame in window (or median of first n)


# --------------------------------------------------------------------------- #
# Signal variants                                                             #
# --------------------------------------------------------------------------- #


def _y_delta(pair: Pair) -> float | None:
    n = len(pair.ball_positions_in_window)
    if n < 4:
        return None
    half = max(2, n // 2)
    start = statistics.median(bp.y for bp in pair.ball_positions_in_window[:half])
    end = statistics.median(bp.y for bp in pair.ball_positions_in_window[-half:])
    return abs(end - start)


def _line_crossed(pair: Pair) -> bool:
    """Did `y - net_y` change sign at any consecutive pair?"""
    bps = pair.ball_positions_in_window
    if len(bps) < 2:
        return False
    for a, b in zip(bps[:-1], bps[1:], strict=False):
        sa = a.y - pair.net_y
        sb = b.y - pair.net_y
        if sa == 0 or sb == 0:
            continue
        if (sa > 0) != (sb > 0):
            return True
    return False


def _n_frames_opposite(pair: Pair) -> int:
    """Count frames whose Y is on the opposite side of net_y vs start_y."""
    if not pair.ball_positions_in_window:
        return 0
    start_above = pair.start_y > pair.net_y
    return sum(
        1 for bp in pair.ball_positions_in_window
        if (bp.y > pair.net_y) != start_above
    )


def _predict(variant: str, pair: Pair, tau_delta: float = 0.15) -> bool:
    """Return whether the variant predicts a cross for this pair."""
    yd = _y_delta(pair)
    yd_pos = yd is not None and yd > tau_delta
    if variant == "A_current":
        return yd_pos
    if variant == "B_line_cross":
        return _line_crossed(pair)
    if variant == "C_n3_opposite":
        return _n_frames_opposite(pair) >= 3
    if variant == "D_delta_or_line":
        return yd_pos or _line_crossed(pair)
    if variant == "E_delta_and_line":
        return yd_pos and _line_crossed(pair)
    if variant == "F_delta_or_n3":
        return yd_pos or _n_frames_opposite(pair) >= 3
    raise ValueError(f"unknown variant: {variant}")


# --------------------------------------------------------------------------- #
# Pair construction                                                           #
# --------------------------------------------------------------------------- #


def _positions_in_window(
    ball_positions: list[BallPosition],
    from_frame: int,
    to_frame: int,
) -> list[BallPosition]:
    return [
        bp for bp in ball_positions
        if from_frame < bp.frame_number < to_frame
    ]


def _build_pairs(rally) -> list[Pair]:  # type: ignore[no-untyped-def]
    """Build labeled pairs for a single rally.

    GT label = whether the two adjacent GT actions are on opposite sides.
    Side comes from GT.ballY relative to per-rally net_y proxy.
    """
    balls = _load_ball_positions(rally.rally_id)
    if not balls:
        return []
    net_y = _net_y_for_rally(balls)

    pairs: list[Pair] = []
    gt = rally.gt_actions
    for i in range(1, len(gt)):
        a = gt[i - 1]
        b = gt[i]
        af = int(a.get("frame", -1))
        bf = int(b.get("frame", -1))
        ay = a.get("ballY")
        by = b.get("ballY")
        if af < 0 or bf <= af or ay is None or by is None:
            continue
        side_a = "far" if float(ay) < net_y else "near"
        side_b = "far" if float(by) < net_y else "near"
        is_real_cross = side_a != side_b
        in_range = _positions_in_window(balls, af, bf)
        if len(in_range) < 4:
            continue
        start_y = statistics.median(
            bp.y for bp in in_range[:max(2, len(in_range) // 2)]
        )
        pairs.append(Pair(
            rally_id=rally.rally_id,
            from_frame=af,
            to_frame=bf,
            is_real_cross=is_real_cross,
            ball_positions_in_window=in_range,
            net_y=net_y,
            start_y=start_y,
        ))
    return pairs


# --------------------------------------------------------------------------- #
# Reporting                                                                   #
# --------------------------------------------------------------------------- #


def _confusion(variant: str, pairs: list[Pair], **kw: Any) -> dict[str, int]:
    tp = fp = fn = tn = 0
    for p in pairs:
        pred = _predict(variant, p, **kw)
        if p.is_real_cross and pred:
            tp += 1
        elif p.is_real_cross and not pred:
            fn += 1
        elif (not p.is_real_cross) and pred:
            fp += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _f1(c: dict[str, int]) -> tuple[float, float, float]:
    p = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 0.0
    r = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def main() -> None:
    console.print("[bold]Loading rallies (skip 'poor' session)…[/bold]")
    rallies = _load_rallies(skip_session_id="6f599a0e-b8ea-4bf0-a331-ce7d9ef88164")
    _tag_low_quality(rallies)
    _run_detectors(rallies)
    clean = [r for r in rallies if not r.low_quality]
    console.print(f"  {len(clean)} clean rallies")

    pairs: list[Pair] = []
    for rally in clean:
        pairs.extend(_build_pairs(rally))

    n_pos = sum(1 for p in pairs if p.is_real_cross)
    n_neg = len(pairs) - n_pos
    console.print(
        f"  {len(pairs)} GT-labeled adjacent pairs  "
        f"(positives={n_pos}, negatives={n_neg})"
    )

    if not pairs:
        console.print("[yellow]No pairs.[/yellow]")
        return

    variants = [
        "A_current",
        "B_line_cross",
        "C_n3_opposite",
        "D_delta_or_line",
        "E_delta_and_line",
        "F_delta_or_n3",
    ]

    table = Table(title=f"Net-cross signal variants — {len(pairs)} pairs")
    table.add_column("variant")
    table.add_column("TP", justify="right")
    table.add_column("FP", justify="right")
    table.add_column("FN", justify="right")
    table.add_column("TN", justify="right")
    table.add_column("P", justify="right")
    table.add_column("R", justify="right")
    table.add_column("F1", justify="right")
    for v in variants:
        c = _confusion(v, pairs)
        p, r, f1 = _f1(c)
        table.add_row(
            v, str(c["tp"]), str(c["fp"]), str(c["fn"]), str(c["tn"]),
            f"{p*100:.1f}%", f"{r*100:.1f}%", f"{f1*100:.1f}%",
        )
    console.print(table)

    # Threshold sweep on the current signal
    console.print("\n[bold]Threshold sweep on A_current (Y-delta only):[/bold]")
    sweep_table = Table()
    sweep_table.add_column("τ_delta", justify="right")
    sweep_table.add_column("TP", justify="right")
    sweep_table.add_column("FP", justify="right")
    sweep_table.add_column("FN", justify="right")
    sweep_table.add_column("TN", justify="right")
    sweep_table.add_column("F1", justify="right")
    for tau in [0.05, 0.075, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25]:
        c = _confusion("A_current", pairs, tau_delta=tau)
        _, _, f1 = _f1(c)
        sweep_table.add_row(
            f"{tau:.3f}", str(c["tp"]), str(c["fp"]),
            str(c["fn"]), str(c["tn"]), f"{f1*100:.1f}%",
        )
    console.print(sweep_table)

    # Threshold sweep with combined D variant
    console.print("\n[bold]Threshold sweep on D_delta_or_line:[/bold]")
    d_table = Table()
    d_table.add_column("τ_delta", justify="right")
    d_table.add_column("TP", justify="right")
    d_table.add_column("FP", justify="right")
    d_table.add_column("FN", justify="right")
    d_table.add_column("F1", justify="right")
    for tau in [0.05, 0.075, 0.10, 0.12, 0.15, 0.18, 0.20]:
        c = _confusion("D_delta_or_line", pairs, tau_delta=tau)
        _, _, f1 = _f1(c)
        d_table.add_row(
            f"{tau:.3f}", str(c["tp"]), str(c["fp"]),
            str(c["fn"]), f"{f1*100:.1f}%",
        )
    console.print(d_table)


if __name__ == "__main__":
    main()
