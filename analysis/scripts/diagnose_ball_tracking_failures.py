"""Diagnose ball tracking failure modes for `ball_dropout` contact FNs.

Session 3 pre-work diagnostic. Categorizes each `ball_dropout` FN (no ball
detection within ±5 frames of a GT contact) into failure-mode buckets, and
probes candidate fixes (raw retrack, filtered retrack, optical-flow gap fill)
to decide which architecture change — if any — is justified.

NO production code is modified. This script only reads. Output is a bucket
histogram + a per-bucket fix-rate table + an optional JSON dump.

Usage:
    cd analysis
    uv run python scripts/diagnose_ball_tracking_failures.py
    uv run python scripts/diagnose_ball_tracking_failures.py --rally <id>
    uv run python scripts/diagnose_ball_tracking_failures.py --output outputs/ball_dropout.json
    uv run python scripts/diagnose_ball_tracking_failures.py --no-optical-flow
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from rich.console import Console
from rich.table import Table

from rallycut.evaluation.tracking.db import get_video_path
from rallycut.tracking.ball_filter import (
    BallFilterConfig,
    BallTemporalFilter,
    get_wasb_filter_config,
)
from rallycut.tracking.ball_tracker import BallPosition, create_ball_tracker
from rallycut.tracking.contact_classifier import load_contact_classifier
from rallycut.tracking.contact_detector import detect_contacts
from scripts.diagnose_fn_contacts import _reconstruct_ball_player_data
from scripts.eval_action_detection import (
    GtLabel,
    RallyData,
    load_rallies_with_action_gt,
    match_contacts,
)

console = Console()

BALL_SEARCH_WINDOW = 5
CATEGORIZE_WINDOW = 15
HIT_WINDOW = 5
WARMUP_FRAMES = 120
SPIKE_VEL_THRESHOLD = 0.15
LOW_CONF_THRESHOLD = 0.20
FAR_COURT_Y = 0.33
OCCLUSION_RADIUS = 0.06

BUCKETS = [
    "serve_warmup",
    "off_screen",
    "hard_spike",
    "motion_blur",
    "far_court",
    "occlusion",
    "other",
]


@dataclass
class DropoutDiag:
    rally_id: str
    video_id: str = ""
    rally_start_ms: int = 0
    fps: float = 30.0
    gt_frame: int = 0
    gt_action: str = ""
    bucket: str = ""
    # probes
    retrack_raw_hit: bool = False
    retrack_filt_hit: bool = False
    seg_only_hit: bool = False
    seg_outlier_hit: bool = False
    optical_flow_hit: bool = False
    warmup_protection_hit: bool = False
    # raw signals
    raw_detections_in_window: int = 0
    nearest_raw_conf: float = 0.0
    bracket_gap: int = 9999
    bracket_before_frame: int | None = None
    bracket_after_frame: int | None = None


@dataclass
class RallyProbeCache:
    """Per-rally cached retrack results (video decode is expensive)."""

    raw_frames: dict[int, BallPosition] = field(default_factory=dict)
    filtered_frames: dict[int, BallPosition] = field(default_factory=dict)
    seg_only_frames: dict[int, BallPosition] = field(default_factory=dict)
    seg_outlier_frames: dict[int, BallPosition] = field(default_factory=dict)
    video_path: Path | None = None
    start_ms: int = 0
    fps: float = 30.0


def _make_seg_only_config(base: BallFilterConfig) -> BallFilterConfig:
    """Mirrors diagnose_ball_dropout.py's stage A: segment pruning only."""
    return BallFilterConfig(
        warmup_protect_frames=0,
        enable_motion_energy_filter=False,
        enable_stationarity_filter=False,
        enable_segment_pruning=True,
        segment_jump_threshold=base.segment_jump_threshold,
        min_segment_frames=base.min_segment_frames,
        min_output_confidence=base.min_output_confidence,
        max_chain_gap=base.max_chain_gap,
        enable_exit_ghost_removal=False,
        enable_oscillation_pruning=False,
        enable_outlier_removal=False,
        enable_blip_removal=False,
        enable_interpolation=False,
    )


def _make_seg_outlier_config(base: BallFilterConfig) -> BallFilterConfig:
    """Mirrors stage B: segment + exit ghost + outlier (no interp / blip / osc)."""
    return BallFilterConfig(
        warmup_protect_frames=0,
        enable_motion_energy_filter=base.enable_motion_energy_filter,
        enable_stationarity_filter=base.enable_stationarity_filter,
        enable_segment_pruning=True,
        segment_jump_threshold=base.segment_jump_threshold,
        min_segment_frames=base.min_segment_frames,
        min_output_confidence=base.min_output_confidence,
        max_chain_gap=base.max_chain_gap,
        enable_exit_ghost_removal=True,
        exit_edge_zone=base.exit_edge_zone,
        exit_approach_frames=base.exit_approach_frames,
        exit_min_approach_speed=base.exit_min_approach_speed,
        exit_max_ghost_frames=base.exit_max_ghost_frames,
        enable_oscillation_pruning=False,
        enable_outlier_removal=True,
        enable_blip_removal=False,
        enable_interpolation=False,
    )


def _retrack_rally(rally: RallyData, stage_isolation: bool = False) -> RallyProbeCache | None:
    """Re-run WASB on a rally: returns {frame: BallPosition} for raw and filtered."""
    video_path = get_video_path(rally.video_id)
    if video_path is None:
        return None

    try:
        tracker = create_ball_tracker()
        end_ms = rally.start_ms + int(1000 * (rally.frame_count or 0) / (rally.fps or 30.0))
        raw_result = tracker.track_video(
            str(video_path),
            start_ms=rally.start_ms,
            end_ms=end_ms,
            enable_filtering=False,
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"  [yellow]retrack failed for {rally.rally_id[:8]}: {exc}[/yellow]")
        return None

    raw = list(raw_result.positions)
    filt_cfg = get_wasb_filter_config()
    filtered = BallTemporalFilter(filt_cfg).filter_batch(list(raw))

    raw_map = {p.frame_number: p for p in raw if p.confidence > 0}
    filt_map = {p.frame_number: p for p in filtered if p.confidence > 0}

    seg_only_map: dict[int, BallPosition] = {}
    seg_outlier_map: dict[int, BallPosition] = {}
    if stage_isolation:
        seg_only = BallTemporalFilter(_make_seg_only_config(filt_cfg)).filter_batch(list(raw))
        seg_outlier = BallTemporalFilter(_make_seg_outlier_config(filt_cfg)).filter_batch(list(raw))
        seg_only_map = {p.frame_number: p for p in seg_only if p.confidence > 0}
        seg_outlier_map = {p.frame_number: p for p in seg_outlier if p.confidence > 0}

    return RallyProbeCache(
        raw_frames=raw_map,
        filtered_frames=filt_map,
        seg_only_frames=seg_only_map,
        seg_outlier_frames=seg_outlier_map,
        video_path=video_path,
        start_ms=rally.start_ms,
        fps=rally.fps or 30.0,
    )


def _optical_flow_bridge(
    cache: RallyProbeCache,
    before_frame: int,
    after_frame: int,
    before_pos: BallPosition,
    after_pos: BallPosition,
    gt_frame: int,
) -> bool:
    """Probe: can we bridge a gap by chaining Farneback flow from before->after?

    Returns True if the chained position at gt_frame lands inside the frame
    (roughly) AND the flow magnitude at the ball location is non-trivial.
    """
    if cache.video_path is None or after_frame <= before_frame:
        return False
    if not (before_frame <= gt_frame <= after_frame):
        return False
    if after_frame - before_frame > CATEGORIZE_WINDOW * 2:
        return False

    try:
        cap = cv2.VideoCapture(str(cache.video_path))
        if not cap.isOpened():
            return False

        # Seek to the absolute frame number in the full video.
        start_frame_abs = int(cache.start_ms * cache.fps / 1000.0)

        def read_at(local_fn: int) -> np.ndarray | None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_abs + local_fn)
            ok, f = cap.read()
            if not ok:
                return None
            f = cv2.resize(f, (320, 180))
            return cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)

        prev = read_at(before_frame)
        nxt = read_at(after_frame)
        cap.release()
        if prev is None or nxt is None:
            return False

        flow = cv2.calcOpticalFlowFarneback(
            prev, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0,
        )
        # Sample flow at the (normalized) ball position of `before_pos`
        h, w = prev.shape
        bx, by = int(before_pos.x * w), int(before_pos.y * h)
        bx = max(0, min(w - 1, bx))
        by = max(0, min(h - 1, by))
        patch = flow[max(0, by - 3) : by + 4, max(0, bx - 3) : bx + 4]
        if patch.size == 0:
            return False
        dx = float(np.mean(patch[..., 0])) / w
        dy = float(np.mean(patch[..., 1])) / h

        # Chain: linear share of flow up to gt_frame
        t = (gt_frame - before_frame) / (after_frame - before_frame)
        # Blend: flow-guided prediction from `before` + linear-interp from `after`
        pred_x = before_pos.x + dx * t + (after_pos.x - before_pos.x) * t
        pred_y = before_pos.y + dy * t + (after_pos.y - before_pos.y) * t
        pred_x *= 0.5
        pred_y *= 0.5

        flow_mag = (dx * dx + dy * dy) ** 0.5
        in_frame = 0.0 <= pred_x <= 1.0 and 0.0 <= pred_y <= 1.0
        return in_frame and flow_mag > 1e-4
    except Exception:  # noqa: BLE001
        return False


def _categorize(
    rally: RallyData,
    gt: GtLabel,
    cache: RallyProbeCache | None,
    player_positions: list,
) -> tuple[str, dict]:
    """Assign one bucket (first match wins) + return raw signal metadata."""
    gt_frame = gt.frame
    meta: dict = {
        "raw_detections_in_window": 0,
        "nearest_raw_conf": 0.0,
        "bracket_before_frame": None,
        "bracket_after_frame": None,
        "bracket_gap": 9999,
    }

    raw_map = cache.raw_frames if cache else {}
    window_frames = [
        f for f in raw_map
        if abs(f - gt_frame) <= CATEGORIZE_WINDOW and raw_map[f].confidence > 0
    ]
    meta["raw_detections_in_window"] = len(window_frames)
    if window_frames:
        nearest = min(window_frames, key=lambda f: abs(f - gt_frame))
        meta["nearest_raw_conf"] = float(raw_map[nearest].confidence)

    # Bracketing detections (for flow probe + hard spike)
    before_frames = sorted(f for f in raw_map if f < gt_frame and raw_map[f].confidence > 0)
    after_frames = sorted(f for f in raw_map if f > gt_frame and raw_map[f].confidence > 0)
    b_f = before_frames[-1] if before_frames else None
    a_f = after_frames[0] if after_frames else None
    meta["bracket_before_frame"] = b_f
    meta["bracket_after_frame"] = a_f
    if b_f is not None and a_f is not None:
        meta["bracket_gap"] = a_f - b_f

    # 1. serve_warmup
    if gt.action == "serve" and gt_frame < WARMUP_FRAMES and len(window_frames) > 0:
        return "serve_warmup", meta

    # 2. off_screen
    if not window_frames:
        return "off_screen", meta

    # 3. hard_spike — fast ball between bracketing detections
    if b_f is not None and a_f is not None and (a_f - b_f) > 1:
        dx = raw_map[a_f].x - raw_map[b_f].x
        dy = raw_map[a_f].y - raw_map[b_f].y
        vel = ((dx * dx + dy * dy) ** 0.5) / (a_f - b_f)
        if vel > SPIKE_VEL_THRESHOLD:
            return "hard_spike", meta

    # 4. motion_blur — low raw confidence in window
    if meta["nearest_raw_conf"] < LOW_CONF_THRESHOLD:
        return "motion_blur", meta

    # 5. far_court
    nearest_frame = min(window_frames, key=lambda f: abs(f - gt_frame))
    if raw_map[nearest_frame].y < FAR_COURT_Y:
        return "far_court", meta

    # 6. occlusion — player bbox center overlaps nearest ball pos
    if player_positions:
        bx, by = raw_map[nearest_frame].x, raw_map[nearest_frame].y
        for pp in player_positions:
            if pp.frame_number != gt_frame:
                continue
            if abs(pp.x - bx) < OCCLUSION_RADIUS and abs(pp.y - by) < OCCLUSION_RADIUS:
                return "occlusion", meta

    return "other", meta


def _probe_hit(frame_map: dict[int, BallPosition], gt_frame: int) -> bool:
    return any(
        abs(f - gt_frame) <= HIT_WINDOW and frame_map[f].confidence > 0
        for f in frame_map
    )


def diagnose_rally(
    rally: RallyData,
    fn_frames: list[GtLabel],
    run_flow: bool,
    stage_isolation: bool = False,
) -> list[DropoutDiag]:
    if not fn_frames:
        return []

    cache = _retrack_rally(rally, stage_isolation=stage_isolation)
    _, player_positions = _reconstruct_ball_player_data(rally)

    diags: list[DropoutDiag] = []
    for gt in fn_frames:
        bucket, meta = _categorize(rally, gt, cache, player_positions)

        retrack_raw = _probe_hit(cache.raw_frames, gt.frame) if cache else False
        retrack_filt = _probe_hit(cache.filtered_frames, gt.frame) if cache else False
        seg_only = (
            _probe_hit(cache.seg_only_frames, gt.frame)
            if cache and cache.seg_only_frames
            else False
        )
        seg_outlier = (
            _probe_hit(cache.seg_outlier_frames, gt.frame)
            if cache and cache.seg_outlier_frames
            else False
        )

        flow_hit = False
        if (
            run_flow
            and cache is not None
            and meta["bracket_before_frame"] is not None
            and meta["bracket_after_frame"] is not None
        ):
            b_f = meta["bracket_before_frame"]
            a_f = meta["bracket_after_frame"]
            flow_hit = _optical_flow_bridge(
                cache, b_f, a_f, cache.raw_frames[b_f], cache.raw_frames[a_f], gt.frame,
            )

        warmup_hit = False
        if bucket == "serve_warmup" and cache is not None:
            # Was a high-conf detection present but killed by the filter?
            nearby_raw_hiconf = any(
                abs(f - gt.frame) <= HIT_WINDOW
                and cache.raw_frames[f].confidence >= 0.50
                for f in cache.raw_frames
            )
            warmup_hit = nearby_raw_hiconf and not retrack_filt

        diags.append(DropoutDiag(
            rally_id=rally.rally_id,
            video_id=rally.video_id,
            rally_start_ms=rally.start_ms,
            fps=rally.fps or 30.0,
            gt_frame=gt.frame,
            gt_action=gt.action,
            bucket=bucket,
            retrack_raw_hit=retrack_raw,
            retrack_filt_hit=retrack_filt,
            seg_only_hit=seg_only,
            seg_outlier_hit=seg_outlier,
            optical_flow_hit=flow_hit,
            warmup_protection_hit=warmup_hit,
            raw_detections_in_window=meta["raw_detections_in_window"],
            nearest_raw_conf=meta["nearest_raw_conf"],
            bracket_gap=meta["bracket_gap"],
            bracket_before_frame=meta["bracket_before_frame"],
            bracket_after_frame=meta["bracket_after_frame"],
        ))

    return diags


def _find_ball_dropout_fns(rally: RallyData, classifier, tolerance_frames: int) -> list[GtLabel]:
    """Reproduce diagnose_fn_contacts's `ball_dropout` slice for one rally."""
    ball_positions, player_positions = _reconstruct_ball_player_data(rally)
    pred_actions: list[dict] = []
    if ball_positions:
        contacts = detect_contacts(
            ball_positions=ball_positions,
            player_positions=player_positions,
            net_y=rally.court_split_y,
            frame_count=rally.frame_count or None,
            classifier=classifier,
            use_classifier=True,
        )
        pred_actions = [c.to_dict() for c in contacts.contacts]

    matches, _ = match_contacts(rally.gt_labels, pred_actions, tolerance=tolerance_frames)

    # Build confident ball frame set (matches diagnose_fn_contacts's ball_dropout
    # definition: no ball within ±BALL_SEARCH_WINDOW of GT frame).
    from rallycut.tracking.contact_detector import _CONFIDENCE_THRESHOLD
    confident_frames = sorted(
        bp.frame_number for bp in ball_positions if bp.confidence >= _CONFIDENCE_THRESHOLD
    )

    dropouts: list[GtLabel] = []
    for m in matches:
        if m.pred_frame is not None:
            continue
        gap = min((abs(f - m.gt_frame) for f in confident_frames), default=9999)
        if gap > BALL_SEARCH_WINDOW:
            dropouts.append(GtLabel(
                frame=m.gt_frame,
                action=m.gt_action,
                player_track_id=-1,
                ball_x=next((g.ball_x for g in rally.gt_labels if g.frame == m.gt_frame), None),
                ball_y=next((g.ball_y for g in rally.gt_labels if g.frame == m.gt_frame), None),
            ))
    return dropouts


def print_bucket_table(diags: list[DropoutDiag]) -> None:
    table = Table(title="Ball-dropout FN bucket histogram", show_lines=False)
    table.add_column("Category")
    table.add_column("N", justify="right")
    table.add_column("%", justify="right")
    counts: dict[str, int] = defaultdict(int)
    for d in diags:
        counts[d.bucket] += 1
    total = max(1, len(diags))
    for b in BUCKETS:
        n = counts.get(b, 0)
        table.add_row(b, str(n), f"{100 * n / total:.1f}%")
    table.add_row("TOTAL", str(len(diags)), "100.0%", style="bold")
    console.print(table)


def print_fix_rate_table(diags: list[DropoutDiag]) -> None:
    table = Table(title="Fix-rate by bucket", show_lines=False)
    table.add_column("Category")
    table.add_column("N", justify="right")
    table.add_column("retrack_raw", justify="right")
    table.add_column("retrack_filt", justify="right")
    table.add_column("optical_flow", justify="right")
    table.add_column("warmup_fix", justify="right")
    by_bucket: dict[str, list[DropoutDiag]] = defaultdict(list)
    for d in diags:
        by_bucket[d.bucket].append(d)

    def pct(xs: list[bool]) -> str:
        if not xs:
            return "—"
        return f"{100 * sum(xs) / len(xs):.0f}%"

    for b in BUCKETS:
        group = by_bucket.get(b, [])
        if not group:
            continue
        table.add_row(
            b,
            str(len(group)),
            pct([d.retrack_raw_hit for d in group]),
            pct([d.retrack_filt_hit for d in group]),
            pct([d.optical_flow_hit for d in group]),
            pct([d.warmup_protection_hit for d in group]),
        )
    table.add_row(
        "OVERALL",
        str(len(diags)),
        pct([d.retrack_raw_hit for d in diags]),
        pct([d.retrack_filt_hit for d in diags]),
        pct([d.optical_flow_hit for d in diags]),
        pct([d.warmup_protection_hit for d in diags]),
        style="bold",
    )
    console.print(table)


def print_stage_isolation_table(diags: list[DropoutDiag]) -> None:
    """For each bucket, show how many raw hits survive each filter stage.

    Stages: raw → +segment_pruning → +segment+outlier → +full_pipeline.
    Use this to pinpoint which stage drops detections (esp. for far_court).
    """
    table = Table(title="Stage isolation: raw-hit survival per filter stage", show_lines=False)
    table.add_column("Category")
    table.add_column("N", justify="right")
    table.add_column("raw", justify="right")
    table.add_column("+seg", justify="right")
    table.add_column("+seg+outlier", justify="right")
    table.add_column("+full", justify="right")

    by_bucket: dict[str, list[DropoutDiag]] = defaultdict(list)
    for d in diags:
        by_bucket[d.bucket].append(d)

    def cell(group: list[DropoutDiag], attr: str) -> str:
        n = sum(getattr(d, attr) for d in group)
        return f"{n}/{len(group)}"

    for b in BUCKETS:
        group = by_bucket.get(b, [])
        if not group:
            continue
        table.add_row(
            b,
            str(len(group)),
            cell(group, "retrack_raw_hit"),
            cell(group, "seg_only_hit"),
            cell(group, "seg_outlier_hit"),
            cell(group, "retrack_filt_hit"),
        )
    table.add_row(
        "OVERALL",
        str(len(diags)),
        cell(diags, "retrack_raw_hit"),
        cell(diags, "seg_only_hit"),
        cell(diags, "seg_outlier_hit"),
        cell(diags, "retrack_filt_hit"),
        style="bold",
    )
    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rally", type=str, default=None, help="Single rally ID")
    parser.add_argument("--tolerance-ms", type=int, default=167)
    parser.add_argument("--no-optical-flow", action="store_true",
                        help="Skip the Farneback probe (faster)")
    parser.add_argument("--stage-isolation", action="store_true",
                        help="Run segment-only / segment+outlier / full filter stages "
                             "and report per-bucket raw-hit survival")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", type=str, default=None,
                        help="JSON dump of per-FN diagnostics")
    args = parser.parse_args()

    classifier = load_contact_classifier()
    if classifier is None:
        console.print("[yellow]No trained contact classifier — using hand-tuned gates[/yellow]")

    rallies = load_rallies_with_action_gt(rally_id=args.rally)
    if not rallies:
        console.print("[red]No rallies with action GT found[/red]")
        return
    console.print(f"[bold]Diagnosing ball_dropout FNs across {len(rallies)} rallies[/bold]\n")

    all_diags: list[DropoutDiag] = []
    run_flow = not args.no_optical_flow

    for i, rally in enumerate(rallies):
        tol = max(1, round((rally.fps or 30) * args.tolerance_ms / 1000))
        dropouts = _find_ball_dropout_fns(rally, classifier, tol)
        console.print(
            f"  [{i + 1}/{len(rallies)}] {rally.rally_id[:8]} "
            f"dropout_fns={len(dropouts)}"
        )
        if not dropouts:
            continue
        diags = diagnose_rally(rally, dropouts, run_flow, stage_isolation=args.stage_isolation)
        all_diags.extend(diags)

    console.print()
    if not all_diags:
        console.print("[green]No ball_dropout FNs found.[/green]")
        return

    console.print(f"[bold]Total ball_dropout FNs diagnosed: {len(all_diags)}[/bold]\n")
    print_bucket_table(all_diags)
    print_fix_rate_table(all_diags)
    if args.stage_isolation:
        print_stage_isolation_table(all_diags)

    if args.verbose:
        detail = Table(title="Per-FN details")
        for col in ("rally", "frame", "action", "bucket", "raw_n", "raw_conf",
                    "bracket_gap", "retr_raw", "retr_filt", "flow"):
            detail.add_column(col)
        for d in all_diags:
            detail.add_row(
                d.rally_id[:8], str(d.gt_frame), d.gt_action, d.bucket,
                str(d.raw_detections_in_window), f"{d.nearest_raw_conf:.2f}",
                str(d.bracket_gap),
                "✓" if d.retrack_raw_hit else "·",
                "✓" if d.retrack_filt_hit else "·",
                "✓" if d.optical_flow_hit else "·",
            )
        console.print(detail)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump([asdict(d) for d in all_diags], f, indent=2)
        console.print(f"[dim]Wrote {out_path}[/dim]")


if __name__ == "__main__":
    main()
