"""Sweep alternative `ball_crossed_net` signal variants on labeled pairs.

The 2026-04-14 net-cross diagnostic showed the current Y-displacement
threshold has 105 FN + 73 FP across 500 same-side pairs in the GT pool.
This script measures whether a richer signal — the existing image-space
Y-delta, alternative image-space signals, and homography-derived
court-space signals — can reduce both error types simultaneously
without regressing the legit-same-side positive class.

Image-space variants (legacy):

A. **current**: |y_delta| > τ_delta
B. **line_cross**: any consecutive pair of confident frames has the ball
   crossing net_y (sign change in `y - net_y`)
C. **n_frames_opposite**: ≥N confident frames in the window are on the
   opposite side of net_y from the trajectory's starting frame
D. **delta_or_line**: A OR B
E. **delta_and_line**: A AND B
F. **delta_or_n3**: A OR (n_frames_opposite ≥ 3)

Homography-derived variants (2026-04-15 extension, plan
`joyful-swimming-teapot.md`):

G. **court_y_endpoint**: project first-N and last-N confident ball
   positions via homography, compare median court_Y to midcourt=8m.
   Cross ⇔ endpoints on opposite sides of 8m (optionally gated by
   τ_court slack from the line).
H. **netline_image**: project court points (0, 8) and (W, 8) back to
   image space via inverse homography to get the true net line (with
   slope). Cross ⇔ signed perpendicular distance from ball to that
   line changes sign between window endpoints.
I. **court_y_or_netline**: G OR H.
J. **delta_or_court_y**: A OR G (keeps current signal as fallback for
   pairs where homography can't project).
K. **court_y_and_delta**: A AND G (high-precision variant).
L. **netline_replaces_scalar**: baseline B but with scalar net_y
   replaced by image-space net line — isolates Direction 2 signal.

Ground truth pool:
- Positives (real cross): every (gt_action[i], gt_action[i+1]) pair where
  the two GT actions are on opposite sides (per a per-rally net_y
  derived from the rally's ball trajectory).
- Negatives (no cross): every same-side GT-confirmed pair from the
  over_three diagnostic where `gt_says_crossed=False`.

Reporting:
- For every variant, emit two rows: `clean` (pairs whose video has a
  trusted calibration source — `manual` or `keypoint`) and `full`
  (entire pool). No calibration confidence field is persisted in DB;
  `court_calibration_source` is used as a coarse-grained proxy.
- Include `Δ_vs_A_current` (delta to legacy baseline on same row) and
  projection-success counts per variant.

For each variant we compute TP, FP, FN, TN, precision, recall, F1.
Aim: maximize F1 while keeping FN ≤ baseline.

Usage:
    cd analysis
    uv run python scripts/sweep_net_crossing_signal.py
    uv run python scripts/sweep_net_crossing_signal.py --phase2   # τ_court + gating sweep
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass, field
from typing import Any, cast

from rich.console import Console
from rich.table import Table

from rallycut.court.calibration import CourtCalibrator
from rallycut.evaluation.db import get_connection
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

# Net midline in court space (beach volleyball is 16m long, net at mid).
_NET_COURT_Y = 8.0
_COURT_WIDTH = 8.0

# Trust each calibration source at a coarse level (no per-video confidence
# is persisted in DB; this is the best proxy available without re-running
# the keypoint detector inline).
_SOURCE_CONFIDENCE: dict[str, float] = {
    "manual": 1.0,
    "keypoint": 0.8,
    "auto": 0.5,
}
_CLEAN_SOURCES = {"manual", "keypoint"}


@dataclass
class CalibrationInfo:
    """Per-video calibration proxy pulled from DB.

    No confidence score is persisted in `courtCalibrationJson`; the
    `court_calibration_source` column is used as a coarse-grained
    trust proxy (manual > keypoint > auto).
    """

    calibrator: CourtCalibrator | None
    source: str | None
    confidence: float  # 0-1 derived from source
    netline_endpoints: tuple[tuple[float, float], tuple[float, float]] | None = None


@dataclass
class Pair:
    """A pair of contacts to evaluate the cross signal on."""

    rally_id: str
    video_id: str
    from_frame: int
    to_frame: int
    is_real_cross: bool  # ground-truth label
    ball_positions_in_window: list[BallPosition]
    net_y: float
    start_y: float  # ball y at first frame in window (or median of first n)
    calibrator: CourtCalibrator | None = None
    calibration_source: str | None = None
    calibration_confidence: float = 0.0
    netline_endpoints: tuple[tuple[float, float], tuple[float, float]] | None = None
    # Cached projections to avoid per-variant recomputation
    court_positions: list[tuple[float, float]] | None = field(default=None, repr=False)

    def has_trusted_calibration(self) -> bool:
        return (
            self.calibrator is not None
            and self.calibration_source in _CLEAN_SOURCES
        )


# --------------------------------------------------------------------------- #
# Calibration loading                                                         #
# --------------------------------------------------------------------------- #


def _load_video_calibrations(video_ids: set[str]) -> dict[str, CalibrationInfo]:
    """Load per-video calibration for the given video ids.

    `courtCalibrationJson` stores 4 normalized-image corners; no
    confidence is persisted. The `court_calibration_source` column
    gives a coarse trust tier that we map to a synthetic confidence.
    """
    if not video_ids:
        return {}

    placeholders = ", ".join(["%s"] * len(video_ids))
    query = (
        f"SELECT id, court_calibration_json, court_calibration_source "
        f"FROM videos WHERE id IN ({placeholders})"
    )
    out: dict[str, CalibrationInfo] = {}
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(query, list(video_ids))
        for row in cur.fetchall():
            vid = cast(str, row[0])
            cal_json: Any = row[1]
            source = cast("str | None", row[2])
            if isinstance(cal_json, str):
                cal_json = json.loads(cal_json)
            calibrator: CourtCalibrator | None = None
            netline: tuple[tuple[float, float], tuple[float, float]] | None = None
            if (
                cal_json
                and isinstance(cal_json, list)
                and len(cal_json) == 4
            ):
                try:
                    calibrator = CourtCalibrator()
                    calibrator.calibrate(
                        [(float(c["x"]), float(c["y"])) for c in cal_json],
                    )
                    if not calibrator.is_calibrated:
                        calibrator = None
                    else:
                        # Precompute net endpoints in image space
                        p1 = calibrator.court_to_image(
                            (0.0, _NET_COURT_Y), 0, 0,
                        )
                        p2 = calibrator.court_to_image(
                            (_COURT_WIDTH, _NET_COURT_Y), 0, 0,
                        )
                        netline = (p1, p2)
                except Exception:
                    calibrator = None
            confidence = _SOURCE_CONFIDENCE.get(source or "", 0.0)
            if calibrator is None:
                confidence = 0.0
            out[vid] = CalibrationInfo(
                calibrator=calibrator,
                source=source,
                confidence=confidence,
                netline_endpoints=netline,
            )
    # Ensure every requested video appears even if row missing
    for vid in video_ids:
        out.setdefault(
            vid, CalibrationInfo(
                calibrator=None, source=None, confidence=0.0,
                netline_endpoints=None,
            ),
        )
    return out


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


def _court_positions(pair: Pair) -> list[tuple[float, float]] | None:
    """Project ball image positions → court coords. Cached on pair."""
    if pair.calibrator is None:
        return None
    if pair.court_positions is not None:
        return pair.court_positions
    projected: list[tuple[float, float]] = []
    for bp in pair.ball_positions_in_window:
        try:
            projected.append(
                pair.calibrator.image_to_court((bp.x, bp.y), 0, 0),
            )
        except Exception:
            continue
    pair.court_positions = projected
    return projected


def _court_y_endpoint_cross(
    pair: Pair, tau_court: float = 0.0, endpoint_frames: int = 3,
) -> bool | None:
    """Court_Y at window start vs end; sign-change = cross.

    Takes the FIRST and LAST `endpoint_frames` projected points (not
    half the window) to stay as close as possible to contact moments
    where ball altitude is lowest. Mid-flight peaks at z=5-10m break
    the z=0 homography assumption; contact-adjacent positions are
    closer to player hand height (z≈2-3m) which is still > 0 but
    much less severe.
    """
    cps = _court_positions(pair)
    if cps is None or len(cps) < 2 * endpoint_frames:
        return None
    start_median = statistics.median(
        p[1] for p in cps[:endpoint_frames]
    )
    end_median = statistics.median(
        p[1] for p in cps[-endpoint_frames:]
    )
    start_delta = start_median - _NET_COURT_Y
    end_delta = end_median - _NET_COURT_Y
    if abs(start_delta) < tau_court or abs(end_delta) < tau_court:
        return False
    if start_delta == 0 or end_delta == 0:
        return False
    return (start_delta > 0) != (end_delta > 0)


def _signed_distance_to_netline(
    pair: Pair, bp: BallPosition,
) -> float | None:
    """Signed perpendicular distance from ball to image-space net line."""
    endpoints = pair.netline_endpoints
    if endpoints is None:
        return None
    (x1, y1), (x2, y2) = endpoints
    # Numerator of signed-distance formula (denominator is positive
    # constant; sign is preserved).
    return (x2 - x1) * (y1 - bp.y) - (x1 - bp.x) * (y2 - y1)


def _netline_image_cross(pair: Pair) -> bool | None:
    """Did ball cross the image-space net line between window endpoints?

    Uses median signed distance at first-N / last-N window frames,
    checks for sign change. Falls back to None if netline unavailable
    or too few positions.
    """
    if pair.netline_endpoints is None:
        return None
    bps = pair.ball_positions_in_window
    if len(bps) < 4:
        return None
    half = max(2, len(bps) // 2)
    start_vals = [
        _signed_distance_to_netline(pair, bp) for bp in bps[:half]
    ]
    end_vals = [
        _signed_distance_to_netline(pair, bp) for bp in bps[-half:]
    ]
    start_clean = [v for v in start_vals if v is not None]
    end_clean = [v for v in end_vals if v is not None]
    if not start_clean or not end_clean:
        return None
    s = statistics.median(start_clean)
    e = statistics.median(end_clean)
    if s == 0 or e == 0:
        return False
    return (s > 0) != (e > 0)


def _netline_image_line_crossed(pair: Pair) -> bool | None:
    """Did signed-distance-to-netline change sign at any consecutive pair?

    This is the direct analog of `_line_crossed` but using the
    image-space net line derived from calibration instead of the
    scalar `net_y` proxy. Variant L.
    """
    if pair.netline_endpoints is None:
        return None
    bps = pair.ball_positions_in_window
    if len(bps) < 2:
        return False
    prev_sign: int | None = None
    for bp in bps:
        d = _signed_distance_to_netline(pair, bp)
        if d is None or d == 0:
            continue
        sign = 1 if d > 0 else -1
        if prev_sign is not None and sign != prev_sign:
            return True
        prev_sign = sign
    return False


def _predict(
    variant: str, pair: Pair,
    tau_delta: float = 0.15,
    tau_court: float = 0.0,
) -> bool | None:
    """Return whether the variant predicts a cross for this pair.

    Returns None for homography variants when projection is
    unavailable (calibrator missing or too few projected points).
    """
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

    # Homography-derived variants
    if variant == "G_court_y_endpoint":
        return _court_y_endpoint_cross(pair, tau_court=tau_court)
    if variant == "H_netline_image":
        return _netline_image_cross(pair)
    if variant == "I_court_y_or_netline":
        g = _court_y_endpoint_cross(pair, tau_court=tau_court)
        h = _netline_image_cross(pair)
        if g is None and h is None:
            return None
        return bool(g) or bool(h)
    if variant == "J_delta_or_court_y":
        g = _court_y_endpoint_cross(pair, tau_court=tau_court)
        if g is None:
            # fall back to A_current when homography unavailable
            return yd_pos
        return yd_pos or g
    if variant == "K_court_y_and_delta":
        g = _court_y_endpoint_cross(pair, tau_court=tau_court)
        if g is None:
            return None
        return yd_pos and g
    if variant == "L_netline_replaces_scalar":
        return _netline_image_line_crossed(pair)
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


def _build_pairs(
    rally: Any,
    cal_info: CalibrationInfo | None = None,
) -> list[Pair]:
    """Build labeled pairs for a single rally.

    GT label = whether the two adjacent GT actions are on opposite sides.
    Side comes from GT.ballY relative to per-rally net_y proxy.
    """
    balls = _load_ball_positions(rally.rally_id)
    if not balls:
        return []
    net_y = _net_y_for_rally(balls)

    calibrator = cal_info.calibrator if cal_info else None
    source = cal_info.source if cal_info else None
    confidence = cal_info.confidence if cal_info else 0.0
    netline = cal_info.netline_endpoints if cal_info else None

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
            video_id=rally.video_id,
            from_frame=af,
            to_frame=bf,
            is_real_cross=is_real_cross,
            ball_positions_in_window=in_range,
            net_y=net_y,
            start_y=start_y,
            calibrator=calibrator,
            calibration_source=source,
            calibration_confidence=confidence,
            netline_endpoints=netline,
        ))
    return pairs


# --------------------------------------------------------------------------- #
# Reporting                                                                   #
# --------------------------------------------------------------------------- #


def _confusion(
    variant: str, pairs: list[Pair], **kw: Any,
) -> dict[str, int]:
    """Confusion with `none` bucket counting pairs where projection failed.

    Variants that can return None (homography unavailable, too few
    projections) emit `none`; legacy variants never do. `none` pairs
    are EXCLUDED from precision/recall/F1 — they represent
    "signal unavailable" rather than a prediction.
    """
    tp = fp = fn = tn = none = 0
    for p in pairs:
        pred = _predict(variant, p, **kw)
        if pred is None:
            none += 1
            continue
        if p.is_real_cross and pred:
            tp += 1
        elif p.is_real_cross and not pred:
            fn += 1
        elif (not p.is_real_cross) and pred:
            fp += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "none": none}


def _f1(c: dict[str, int]) -> tuple[float, float, float]:
    p = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 0.0
    r = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


# --------------------------------------------------------------------------- #
# Variant registry + reporting                                                #
# --------------------------------------------------------------------------- #

_LEGACY_VARIANTS = [
    "A_current",
    "B_line_cross",
    "C_n3_opposite",
    "D_delta_or_line",
    "E_delta_and_line",
    "F_delta_or_n3",
]

_HOMOGRAPHY_VARIANTS = [
    "G_court_y_endpoint",
    "H_netline_image",
    "I_court_y_or_netline",
    "J_delta_or_court_y",
    "K_court_y_and_delta",
    "L_netline_replaces_scalar",
]


def _variant_row(
    variant: str, pairs: list[Pair], ref_f1: float | None = None,
    **kw: Any,
) -> tuple[dict[str, int], float, float, float]:
    c = _confusion(variant, pairs, **kw)
    p, r, f1 = _f1(c)
    return c, p, r, f1


def _render_variants_table(
    title: str, pairs: list[Pair],
    variants: list[str],
    ref_variant: str = "A_current",
) -> None:
    """Print a confusion table for every variant on a pair pool."""
    ref_c, _, _, ref_f1 = _variant_row(ref_variant, pairs)
    n = len(pairs)
    table = Table(title=f"{title} — n={n}")
    table.add_column("variant")
    table.add_column("TP", justify="right")
    table.add_column("FP", justify="right")
    table.add_column("FN", justify="right")
    table.add_column("TN", justify="right")
    table.add_column("none", justify="right")  # projection-unavailable
    table.add_column("P", justify="right")
    table.add_column("R", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("ΔF1 vs A", justify="right")
    table.add_column("cover%", justify="right")
    for v in variants:
        c, p, r, f1 = _variant_row(v, pairs)
        delta = (f1 - ref_f1) * 100.0
        cover = (n - c["none"]) / n * 100.0 if n else 0.0
        row_style = None
        if v != ref_variant and delta >= 5.0:
            row_style = "bold green"
        elif v != ref_variant and delta <= -1.0:
            row_style = "dim"
        table.add_row(
            v,
            str(c["tp"]), str(c["fp"]), str(c["fn"]),
            str(c["tn"]), str(c["none"]),
            f"{p*100:.1f}%", f"{r*100:.1f}%", f"{f1*100:.1f}%",
            f"{delta:+.1f}pp",
            f"{cover:.0f}%",
            style=row_style,
        )
    console.print(table)


def _sanity_check_and_report(
    clean_pairs: list[Pair], full_pairs: list[Pair],
) -> bool:
    """Run the mandatory pre-kill sanity checks.

    Returns True iff all checks pass (i.e., a "no signal" declaration
    would be trustworthy). If any check fails, report details so the
    caller can investigate wiring bugs before re-running.
    """
    console.print("\n[bold]Pre-kill sanity checks[/bold]")
    n_clean = len(clean_pairs)
    console.print(f"  (full pool n={len(full_pairs)})")
    clean_with_cal = sum(1 for p in clean_pairs if p.calibrator is not None)
    pct_cal = (
        clean_with_cal / n_clean * 100.0 if n_clean else 0.0
    )
    n_clean_pos = sum(1 for p in clean_pairs if p.is_real_cross)
    n_clean_neg = n_clean - n_clean_pos

    check1 = pct_cal >= 80.0
    check2 = n_clean_pos >= 200 and n_clean_neg >= 400

    console.print(
        f"  [{'green' if check1 else 'red'}]✓[/] calibrator loaded for "
        f"{clean_with_cal}/{n_clean} clean pairs ({pct_cal:.1f}%) — "
        f"needs ≥80% · {'PASS' if check1 else 'FAIL'}"
    )
    console.print(
        f"  [{'green' if check2 else 'red'}]✓[/] clean subset has "
        f"{n_clean_pos} pos + {n_clean_neg} neg pairs — "
        f"needs ≥200 pos & ≥400 neg · {'PASS' if check2 else 'FAIL'}"
    )

    # Spot-check projections — uses the SAME strict 3-frame endpoint
    # projection as variant G so the diagnostic matches the signal.
    console.print(
        "\n[bold]Projection spot-checks[/bold] "
        "(strict 3-frame endpoints, same as variant G)"
    )
    spot_pos = [p for p in clean_pairs if p.is_real_cross and p.calibrator][:5]
    spot_neg = [p for p in clean_pairs if not p.is_real_cross and p.calibrator][:5]
    endpoint_frames = 3
    for label, samples in [("real-cross", spot_pos), ("same-side", spot_neg)]:
        for p in samples:
            cps = _court_positions(p)
            if not cps or len(cps) < 2 * endpoint_frames:
                console.print(
                    f"  [yellow]{label}[/] {p.rally_id[:8]} "
                    f"f{p.from_frame}→f{p.to_frame}: "
                    f"projection failed ({len(cps) if cps else 0} pts)"
                )
                continue
            start_y = statistics.median(
                cp[1] for cp in cps[:endpoint_frames]
            )
            end_y = statistics.median(
                cp[1] for cp in cps[-endpoint_frames:]
            )
            expect = "≠" if label == "real-cross" else "="
            # Court y=0 is the near baseline (closer to camera), y=16 is
            # the far baseline. Label halves of the court accordingly;
            # pairs with |y| far outside [0, 16] are flagged extrapolated.
            got_start = "near" if start_y < _NET_COURT_Y else "far"
            got_end = "near" if end_y < _NET_COURT_Y else "far"
            extrap_flag = ""
            if start_y < -2 or start_y > 18 or end_y < -2 or end_y > 18:
                extrap_flag = " [extrapolated]"
            match = (got_start != got_end) if label == "real-cross" \
                else (got_start == got_end)
            tag = "[green]✓[/]" if match else "[red]✗[/]"
            console.print(
                f"  {tag} {label} {p.rally_id[:8]} "
                f"cy_start={start_y:.2f} ({got_start}) "
                f"cy_end={end_y:.2f} ({got_end}) "
                f"expect {expect}{extrap_flag} "
                f"· src={p.calibration_source} · n_proj={len(cps)}"
            )

    return check1 and check2


def _phase2_sweeps(pairs: list[Pair]) -> None:
    """Phase 2 — threshold + confidence-gating sweeps on winners."""
    console.print("\n[bold]Phase 2: τ_court sensitivity on G_court_y_endpoint[/bold]")
    tau_table = Table()
    tau_table.add_column("τ_court (m)", justify="right")
    tau_table.add_column("TP", justify="right")
    tau_table.add_column("FP", justify="right")
    tau_table.add_column("FN", justify="right")
    tau_table.add_column("none", justify="right")
    tau_table.add_column("P", justify="right")
    tau_table.add_column("R", justify="right")
    tau_table.add_column("F1", justify="right")
    for tau in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
        c, pre, rec, f1 = _variant_row(
            "G_court_y_endpoint", pairs, tau_court=tau,
        )
        tau_table.add_row(
            f"{tau:.2f}", str(c["tp"]), str(c["fp"]), str(c["fn"]),
            str(c["none"]),
            f"{pre*100:.1f}%", f"{rec*100:.1f}%", f"{f1*100:.1f}%",
        )
    console.print(tau_table)

    # Confidence-gated fallback: use G when calibration trusted, else A
    console.print(
        "\n[bold]Phase 2: confidence-gated fallback "
        "(G if trusted else A)[/bold]"
    )
    gate_table = Table()
    gate_table.add_column("gate source", justify="left")
    gate_table.add_column("TP", justify="right")
    gate_table.add_column("FP", justify="right")
    gate_table.add_column("FN", justify="right")
    gate_table.add_column("F1", justify="right")
    for gate_sources in [
        {"manual"},
        {"manual", "keypoint"},
        {"manual", "keypoint", "auto"},
    ]:
        tp = fp = fn = tn = 0
        for p in pairs:
            if p.calibration_source in gate_sources and p.calibrator:
                pred = _court_y_endpoint_cross(p, tau_court=0.0)
                if pred is None:
                    yd = _y_delta(p)
                    pred = yd is not None and yd > 0.15
            else:
                yd = _y_delta(p)
                pred = yd is not None and yd > 0.15
            if p.is_real_cross and pred:
                tp += 1
            elif p.is_real_cross and not pred:
                fn += 1
            elif (not p.is_real_cross) and pred:
                fp += 1
            else:
                tn += 1
        c = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "none": 0}
        _, _, f1 = _f1(c)
        gate_table.add_row(
            "|".join(sorted(gate_sources)),
            str(tp), str(fp), str(fn), f"{f1*100:.1f}%",
        )
    console.print(gate_table)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase2", action="store_true",
        help="Also run τ_court and confidence-gating sweeps on winners",
    )
    args = parser.parse_args()

    console.print("[bold]Loading rallies (skip 'poor' session)…[/bold]")
    rallies = _load_rallies(skip_session_id="6f599a0e-b8ea-4bf0-a331-ce7d9ef88164")
    _tag_low_quality(rallies)
    _run_detectors(rallies)
    clean_rallies = [r for r in rallies if not r.low_quality]
    console.print(f"  {len(clean_rallies)} clean rallies")

    video_ids = {r.video_id for r in clean_rallies}
    console.print(
        f"[bold]Loading calibration for {len(video_ids)} videos…[/bold]"
    )
    calibrations = _load_video_calibrations(video_ids)
    by_source: dict[str, int] = {}
    for loaded_cinfo in calibrations.values():
        key = loaded_cinfo.source or "none"
        by_source[key] = by_source.get(key, 0) + 1
    src_summary = ", ".join(
        f"{k}={v}" for k, v in sorted(by_source.items())
    )
    console.print(f"  calibration sources: {src_summary}")

    pairs: list[Pair] = []
    for rally in clean_rallies:
        rally_cinfo: CalibrationInfo | None = calibrations.get(rally.video_id)
        pairs.extend(_build_pairs(rally, cal_info=rally_cinfo))

    n_pos = sum(1 for p in pairs if p.is_real_cross)
    n_neg = len(pairs) - n_pos
    console.print(
        f"  {len(pairs)} GT-labeled adjacent pairs  "
        f"(positives={n_pos}, negatives={n_neg})"
    )

    if not pairs:
        console.print("[yellow]No pairs.[/yellow]")
        return

    clean_pairs = [p for p in pairs if p.has_trusted_calibration()]
    console.print(
        f"  clean subset (source ∈ {sorted(_CLEAN_SOURCES)}): "
        f"{len(clean_pairs)} pairs "
        f"(pos={sum(1 for p in clean_pairs if p.is_real_cross)}, "
        f"neg={sum(1 for p in clean_pairs if not p.is_real_cross)})"
    )

    variants = _LEGACY_VARIANTS + _HOMOGRAPHY_VARIANTS

    console.print("\n[bold]━━━ Full pool ━━━[/bold]")
    _render_variants_table(
        "All signal variants (full pool)", pairs, variants,
    )

    console.print("\n[bold]━━━ Clean subset (trusted calibration) ━━━[/bold]")
    _render_variants_table(
        "All signal variants (clean subset)", clean_pairs, variants,
    )

    _sanity_check_and_report(clean_pairs, pairs)

    # Legacy threshold sweep on A_current (unchanged)
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

    if args.phase2:
        _phase2_sweeps(clean_pairs)


if __name__ == "__main__":
    main()
