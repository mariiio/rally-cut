"""Per-rally tracking-audit classifiers and builders.

Layer 1 of the tracking-audit stack (see plan iridescent-floating-sky):
pure, testable functions that classify missed GT frames by root cause and
emit real identity-switch events from match data.

Aggregate metrics (HOTA, MOTA, IdentityMetrics counts) remain in metrics.py.
This module adds per-GT-track life-cycle detail that the evaluation harness
currently does not dump.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from rallycut.cli.commands.compare_tracking import _match_detections
from rallycut.evaluation.tracking.metrics import (
    _compute_iou,
    smart_interpolate_gt,
)
from rallycut.labeling.ground_truth import GroundTruthPosition, GroundTruthResult
from rallycut.tracking.player_tracker import PlayerPosition, PlayerTrackingResult

# Thresholds — tuned to match existing pipeline semantics.
EDGE_MARGIN = 0.05
OCCLUSION_DIST = 0.08
FILTER_IOU_THRESHOLD = 0.3
NET_BAND = 0.05
MIN_SEGMENT_FRAMES = 5
CONVERGENCE_DISTANCE = 0.05
CONVERGENCE_WINDOW = 5


class MissCause(StrEnum):
    OUT_OF_FRAME = "out_of_frame"
    EDGE_PROXIMITY = "edge_proximity"
    OCCLUSION = "occlusion"
    FILTER_DROP = "filter_drop"
    DETECTOR_MISS = "detector_miss"
    UNCLASSIFIED = "unclassified"


class SwitchCause(StrEnum):
    NET_CROSSING = "net_crossing"
    SAME_TEAM_SWAP = "same_team_swap"
    CROSS_TEAM_SWAP = "cross_team_swap"
    FRAGMENT_GAP = "fragment_gap"
    UNCLASSIFIED = "unclassified"


@dataclass
class SwitchEvent:
    frame: int
    pred_id: int
    gt_old: int
    gt_new: int
    cause: SwitchCause = SwitchCause.UNCLASSIFIED


def _iou_pos(a: GroundTruthPosition | PlayerPosition, b: GroundTruthPosition | PlayerPosition) -> float:
    return _compute_iou((a.x, a.y, a.width, a.height), (b.x, b.y, b.width, b.height))


def _centroid_dist(
    a: GroundTruthPosition | PlayerPosition,
    b: GroundTruthPosition | PlayerPosition,
) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    return float((dx * dx + dy * dy) ** 0.5)


def classify_miss_cause(
    gt_pos: GroundTruthPosition,
    other_gt_in_frame: list[GroundTruthPosition],
    raw_pred_in_frame: list[PlayerPosition],
    primary_pred_in_frame: list[PlayerPosition],
    raw_pred_in_window: list[PlayerPosition] | None = None,
    edge_margin: float = EDGE_MARGIN,
    occlusion_dist: float = OCCLUSION_DIST,
    filter_iou_threshold: float = FILTER_IOU_THRESHOLD,
) -> MissCause:
    """Classify why a single GT frame was missed, in priority order.

    Priority: OUT_OF_FRAME > EDGE_PROXIMITY > OCCLUSION > FILTER_DROP > DETECTOR_MISS.
    The priority is structural-before-behavioural: geometry of the frame first
    (we can't fix out-of-frame), then GT-level interactions (occlusion), then
    pipeline state (raw detection present but filtered vs no detection at all).

    Caller is responsible for only passing GT frames that were genuinely
    missed (Hungarian-unmatched at the eval IoU threshold).
    """
    if gt_pos.x < 0 or gt_pos.x > 1 or gt_pos.y < 0 or gt_pos.y > 1:
        return MissCause.OUT_OF_FRAME
    if (
        gt_pos.x < edge_margin
        or gt_pos.x > 1 - edge_margin
        or gt_pos.y < edge_margin
        or gt_pos.y > 1 - edge_margin
    ):
        return MissCause.EDGE_PROXIMITY
    for other in other_gt_in_frame:
        if other.track_id == gt_pos.track_id:
            continue
        if _centroid_dist(gt_pos, other) < occlusion_dist:
            return MissCause.OCCLUSION
    # YOLO may run at a stride > 1; GT is annotated at arbitrary frames, so an
    # exact-frame lookup can miss a raw detection that occurred on the adjacent
    # sampled frame. Fall back to a per-rally window (±1 frame) when provided.
    raw_candidates = list(raw_pred_in_frame)
    if raw_pred_in_window:
        raw_candidates.extend(raw_pred_in_window)
    for raw in raw_candidates:
        if _iou_pos(gt_pos, raw) >= filter_iou_threshold:
            return MissCause.FILTER_DROP
    return MissCause.DETECTOR_MISS


def group_missed_frame_ranges(
    frames_with_cause: list[tuple[int, MissCause]],
) -> dict[MissCause, list[tuple[int, int]]]:
    """Group (frame, cause) pairs into contiguous (start, end) ranges per cause.

    A run is broken by either a frame gap (frame != prev+1) or a cause change.
    Returns dict[cause, list of (start_frame, end_frame)] — end inclusive.
    """
    if not frames_with_cause:
        return {}

    ordered = sorted(frames_with_cause, key=lambda p: p[0])
    result: dict[MissCause, list[tuple[int, int]]] = defaultdict(list)

    start_frame, cur_cause = ordered[0]
    prev_frame = start_frame
    for frame, cause in ordered[1:]:
        if frame == prev_frame + 1 and cause == cur_cause:
            prev_frame = frame
        else:
            result[cur_cause].append((start_frame, prev_frame))
            start_frame = frame
            prev_frame = frame
            cur_cause = cause
    result[cur_cause].append((start_frame, prev_frame))
    return dict(result)


def _is_convergence_ambiguity(
    gt_old: int,
    gt_new: int,
    switch_frame: int,
    gt_positions: dict[int, dict[int, tuple[float, float]]],
    window: int = CONVERGENCE_WINDOW,
    dist: float = CONVERGENCE_DISTANCE,
) -> bool:
    if not gt_positions:
        return False
    pos_old = gt_positions.get(gt_old, {})
    pos_new = gt_positions.get(gt_new, {})
    for f in range(switch_frame - window, switch_frame + window + 1):
        if f in pos_old and f in pos_new:
            dx = pos_old[f][0] - pos_new[f][0]
            dy = pos_old[f][1] - pos_new[f][1]
            if (dx * dx + dy * dy) ** 0.5 < dist:
                return True
    return False


def _classify_switch_cause(
    gt_old: int,
    gt_new: int,
    switch_frame: int,
    gt_by_frame: dict[int, list[tuple[int, float, float, float, float]]] | None,
    net_y: float | None,
    net_band: float = NET_BAND,
) -> SwitchCause:
    if gt_by_frame is None or net_y is None:
        return SwitchCause.UNCLASSIFIED

    # Look up y-positions of both GT tracks at or near the switch frame.
    def _y_at(track_id: int) -> float | None:
        for offset in range(0, CONVERGENCE_WINDOW + 1):
            for f in (switch_frame - offset, switch_frame + offset):
                for gid, _x, y, _w, _h in gt_by_frame.get(f, []):
                    if gid == track_id:
                        return y
        return None

    y_old = _y_at(gt_old)
    y_new = _y_at(gt_new)
    if y_old is None or y_new is None:
        return SwitchCause.UNCLASSIFIED

    old_at_net = abs(y_old - net_y) < net_band
    new_at_net = abs(y_new - net_y) < net_band
    if old_at_net and new_at_net:
        return SwitchCause.NET_CROSSING

    old_side = 0 if y_old < net_y else 1
    new_side = 0 if y_new < net_y else 1
    if old_side == new_side:
        return SwitchCause.SAME_TEAM_SWAP
    return SwitchCause.CROSS_TEAM_SWAP


def iter_real_switch_events(
    pred_by_frame: dict[int, list[tuple[int, float, float, float, float]]],
    matches_by_frame: dict[int, list[tuple[int, int]]],
    gt_by_frame: dict[int, list[tuple[int, float, float, float, float]]] | None = None,
    net_y: float | None = None,
    team_assignments: dict[int, int] | None = None,
    min_segment_frames: int = MIN_SEGMENT_FRAMES,
    overlap_iou_threshold: float = 0.05,
) -> Iterator[SwitchEvent]:
    """Emit one SwitchEvent per real identity switch.

    Mirrors compute_identity_metrics segment-analysis logic so aggregate counts
    match, but yields a per-switch event with cause classification. Convergence
    ambiguities are filtered out (not emitted), matching the aggregate metric.
    """
    all_frames = sorted(matches_by_frame.keys())
    if not all_frames:
        return

    frame_info: list[tuple[int, dict[int, int], bool]] = []
    for frame in all_frames:
        pred_boxes = pred_by_frame.get(frame, [])
        matches = matches_by_frame[frame]
        p2g = {pred_id: gt_id for gt_id, pred_id in matches}

        is_overlap = False
        for i in range(len(pred_boxes)):
            for j in range(i + 1, len(pred_boxes)):
                iou = _compute_iou(
                    (pred_boxes[i][1], pred_boxes[i][2], pred_boxes[i][3], pred_boxes[i][4]),
                    (pred_boxes[j][1], pred_boxes[j][2], pred_boxes[j][3], pred_boxes[j][4]),
                )
                if iou > overlap_iou_threshold:
                    is_overlap = True
                    break
            if is_overlap:
                break
        frame_info.append((frame, p2g, is_overlap))

    gt_positions: dict[int, dict[int, tuple[float, float]]] = {}
    if gt_by_frame is not None:
        for frame, boxes in gt_by_frame.items():
            for gt_id, x, y, _w, _h in boxes:
                gt_positions.setdefault(gt_id, {})[frame] = (x, y)

    pred_ids: set[int] = set()
    for _, p2g, _ in frame_info:
        pred_ids.update(p2g.keys())

    for pred_id in pred_ids:
        clean: list[tuple[int, int]] = []
        for frame, p2g, is_overlap in frame_info:
            if pred_id in p2g and not is_overlap:
                clean.append((frame, p2g[pred_id]))
        if len(clean) < min_segment_frames:
            continue

        segments: list[tuple[int, int, int]] = []
        seg_gt = clean[0][1]
        seg_count = 1
        seg_start = clean[0][0]
        for i in range(1, len(clean)):
            _, gt_id = clean[i]
            if gt_id == seg_gt:
                seg_count += 1
            else:
                segments.append((seg_gt, seg_count, seg_start))
                seg_gt = gt_id
                seg_count = 1
                seg_start = clean[i][0]
        segments.append((seg_gt, seg_count, seg_start))

        real_segs = [s for s in segments if s[1] >= min_segment_frames]
        if len(real_segs) <= 1:
            continue

        # Emit first real switch per pred track (match compute_identity_metrics).
        for i in range(1, len(real_segs)):
            gt_old = real_segs[i - 1][0]
            gt_new = real_segs[i][0]
            switch_frame = real_segs[i][2]
            if gt_old == gt_new:
                continue
            if _is_convergence_ambiguity(gt_old, gt_new, switch_frame, gt_positions):
                continue
            cause = _classify_switch_cause(
                gt_old=gt_old,
                gt_new=gt_new,
                switch_frame=switch_frame,
                gt_by_frame=gt_by_frame,
                net_y=net_y,
            )
            yield SwitchEvent(
                frame=switch_frame,
                pred_id=pred_id,
                gt_old=gt_old,
                gt_new=gt_new,
                cause=cause,
            )
            break


# ---------------------------------------------------------------------------
# Rally-level audit assembly
# ---------------------------------------------------------------------------


@dataclass
class GtTrackAudit:
    """Per-GT-track life-cycle summary within a single rally."""

    gt_track_id: int
    gt_label: str
    gt_frame_count: int
    matched_frames: int
    coverage: float
    distinct_pred_ids: list[int]
    real_switches: list[SwitchEvent]
    missed_by_cause: dict[MissCause, list[tuple[int, int]]]
    time_at_net_pct: float
    net_crossings: int
    # Compact timeline: which pred_id was matched across which contiguous span.
    # Each entry: (start_frame, end_frame_inclusive, pred_id). Unmatched gaps
    # between spans are the MissCause ranges above.
    pred_id_spans: list[tuple[int, int, int]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gtTrackId": self.gt_track_id,
            "gtLabel": self.gt_label,
            "gtFrameCount": self.gt_frame_count,
            "matchedFrames": self.matched_frames,
            "coverage": self.coverage,
            "distinctPredIds": self.distinct_pred_ids,
            "realSwitches": [
                {
                    "frame": ev.frame,
                    "predId": ev.pred_id,
                    "gtOld": ev.gt_old,
                    "gtNew": ev.gt_new,
                    "cause": ev.cause.value,
                }
                for ev in self.real_switches
            ],
            "missedByCause": {
                cause.value: [list(r) for r in ranges]
                for cause, ranges in self.missed_by_cause.items()
            },
            "timeAtNetPct": self.time_at_net_pct,
            "netCrossings": self.net_crossings,
            "predIdSpans": [list(s) for s in self.pred_id_spans],
        }


@dataclass
class ConventionDriftAudit:
    """Detect whether GT-label ↔ pred-track mapping preserves the spatial/team convention."""

    gt_label_to_pred_id_mode: dict[str, int]
    gt_label_mean_y: dict[str, float]
    pred_mean_y: dict[int, float]
    team_assignments: dict[int, int]
    court_side_flip: bool
    team_label_flip: bool
    net_y: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "gtLabelToPredIdMode": self.gt_label_to_pred_id_mode,
            "gtLabelMeanY": self.gt_label_mean_y,
            "predMeanY": {str(k): v for k, v in self.pred_mean_y.items()},
            "teamAssignments": {str(k): v for k, v in self.team_assignments.items()},
            "courtSideFlip": self.court_side_flip,
            "teamLabelFlip": self.team_label_flip,
            "netY": self.net_y,
        }


@dataclass
class RallyAudit:
    """Complete per-rally audit — summaries, per-GT detail, convention, raw-det stats."""

    rally_id: str
    video_id: str
    frame_count: int
    video_fps: float
    # Aggregate metrics (derived)
    hota: float | None
    mota: float
    aggregate_real_switches: int
    # Per-GT detail
    per_gt: list[GtTrackAudit]
    convention: ConventionDriftAudit
    raw_detection_summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "rallyId": self.rally_id,
            "videoId": self.video_id,
            "frameCount": self.frame_count,
            "videoFps": self.video_fps,
            "hota": self.hota,
            "mota": self.mota,
            "aggregateRealSwitches": self.aggregate_real_switches,
            "perGt": [g.to_dict() for g in self.per_gt],
            "convention": self.convention.to_dict(),
            "rawDetectionSummary": self.raw_detection_summary,
        }


def _match_frames(
    gt: GroundTruthResult,
    predictions: PlayerTrackingResult,
    iou_threshold: float,
) -> tuple[
    dict[int, list[tuple[int, float, float, float, float]]],
    dict[int, list[tuple[int, float, float, float, float]]],
    dict[int, list[tuple[int, int]]],
    dict[int, list[int]],
]:
    """Build per-frame GT/pred index + matches + unmatched-GT-id list.

    Returns (gt_by_frame, pred_by_frame, matches_by_frame, unmatched_gt_ids_by_frame).
    Each of the per-frame dicts is keyed by frame number. Tuple format for boxes is
    (track_id, x, y, w, h).
    """
    gt_by_frame: dict[int, list[tuple[int, float, float, float, float]]] = defaultdict(list)
    pred_by_frame: dict[int, list[tuple[int, float, float, float, float]]] = defaultdict(list)

    for gt_p in gt.player_positions:
        gt_by_frame[gt_p.frame_number].append((gt_p.track_id, gt_p.x, gt_p.y, gt_p.width, gt_p.height))
    for pred_p in predictions.positions:
        pred_by_frame[pred_p.frame_number].append(
            (pred_p.track_id, pred_p.x, pred_p.y, pred_p.width, pred_p.height)
        )

    all_frames = sorted(set(gt_by_frame.keys()) | set(pred_by_frame.keys()))
    matches_by_frame: dict[int, list[tuple[int, int]]] = {}
    unmatched_by_frame: dict[int, list[int]] = {}

    for frame in all_frames:
        gt_boxes = gt_by_frame.get(frame, [])
        pred_boxes = pred_by_frame.get(frame, [])
        if not gt_boxes:
            continue
        if not pred_boxes:
            unmatched_by_frame[frame] = [b[0] for b in gt_boxes]
            continue
        matches, unmatched_gt, _unmatched_pred = _match_detections(
            gt_boxes, pred_boxes, iou_threshold
        )
        if matches:
            matches_by_frame[frame] = matches
        if unmatched_gt:
            unmatched_by_frame[frame] = unmatched_gt

    return dict(gt_by_frame), dict(pred_by_frame), matches_by_frame, unmatched_by_frame


def _raw_positions_by_frame(
    raw_positions: list[PlayerPosition] | None,
) -> dict[int, list[PlayerPosition]]:
    if not raw_positions:
        return {}
    idx: dict[int, list[PlayerPosition]] = defaultdict(list)
    for p in raw_positions:
        idx[p.frame_number].append(p)
    return dict(idx)


def _primary_positions_by_frame(
    predictions: PlayerTrackingResult,
) -> dict[int, list[PlayerPosition]]:
    idx: dict[int, list[PlayerPosition]] = defaultdict(list)
    for p in predictions.positions:
        idx[p.frame_number].append(p)
    return dict(idx)


def _gt_positions_by_frame(gt: GroundTruthResult) -> dict[int, list[GroundTruthPosition]]:
    idx: dict[int, list[GroundTruthPosition]] = defaultdict(list)
    for p in gt.player_positions:
        idx[p.frame_number].append(p)
    return dict(idx)


def build_per_gt_audits(
    gt: GroundTruthResult,
    predictions: PlayerTrackingResult,
    raw_positions: list[PlayerPosition] | None,
    matches_by_frame: dict[int, list[tuple[int, int]]],
    unmatched_by_frame: dict[int, list[int]],
    net_y: float | None,
    team_assignments: dict[int, int] | None,
) -> list[GtTrackAudit]:
    """Assemble the per-GT-track audit list.

    Caller supplies pre-computed matches + unmatched (from _match_frames). This
    keeps the matching side-effect centralised.
    """
    gt_positions_by_frame = _gt_positions_by_frame(gt)
    raw_by_frame = _raw_positions_by_frame(raw_positions)
    primary_by_frame = _primary_positions_by_frame(predictions)

    # Index GT positions by track
    gt_by_track: dict[int, list[GroundTruthPosition]] = defaultdict(list)
    for p in gt.player_positions:
        gt_by_track[p.track_id].append(p)

    # Index matches per GT track (frame-ordered for span building)
    matched_frames_by_gt: dict[int, set[int]] = defaultdict(set)
    pred_ids_by_gt: dict[int, list[int]] = defaultdict(list)
    matched_pairs_by_gt: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for frame in sorted(matches_by_frame.keys()):
        for gt_id, pred_id in matches_by_frame[frame]:
            matched_frames_by_gt[gt_id].add(frame)
            pred_ids_by_gt[gt_id].append(pred_id)
            matched_pairs_by_gt[gt_id].append((frame, pred_id))

    # All switch events indexed by the GT track that was abandoned.
    all_switch_events = list(
        iter_real_switch_events(
            pred_by_frame={
                f: [(b.track_id, b.x, b.y, b.width, b.height) for b in bs]
                for f, bs in primary_by_frame.items()
            },
            matches_by_frame=matches_by_frame,
            gt_by_frame={
                f: [(p.track_id, p.x, p.y, p.width, p.height) for p in ps]
                for f, ps in gt_positions_by_frame.items()
            },
            net_y=net_y,
            team_assignments=team_assignments,
        )
    )

    audits: list[GtTrackAudit] = []
    for gt_id in sorted(gt_by_track):
        gt_frames = gt_by_track[gt_id]
        gt_frame_numbers = sorted({p.frame_number for p in gt_frames})
        matched = matched_frames_by_gt.get(gt_id, set())
        coverage = len(matched) / len(gt_frame_numbers) if gt_frame_numbers else 0.0

        # Missed frames → classify each. YOLO may only run every N frames, so
        # supply a ±1-frame window of raw detections for the FILTER_DROP test.
        frames_with_cause: list[tuple[int, MissCause]] = []
        for p in gt_frames:
            if p.frame_number in matched:
                continue
            window_raw: list[PlayerPosition] = []
            for df in (-1, 1):
                window_raw.extend(raw_by_frame.get(p.frame_number + df, []))
            cause = classify_miss_cause(
                gt_pos=p,
                other_gt_in_frame=[
                    o for o in gt_positions_by_frame.get(p.frame_number, [])
                    if o.track_id != p.track_id
                ],
                raw_pred_in_frame=raw_by_frame.get(p.frame_number, []),
                primary_pred_in_frame=primary_by_frame.get(p.frame_number, []),
                raw_pred_in_window=window_raw,
            )
            frames_with_cause.append((p.frame_number, cause))
        missed_by_cause = group_missed_frame_ranges(frames_with_cause)

        distinct_preds = sorted(set(pred_ids_by_gt.get(gt_id, [])))
        # Switches where this GT was the old side
        gt_switches = [ev for ev in all_switch_events if ev.gt_old == gt_id]

        # Net proximity stats (GT perspective)
        if net_y is not None and gt_frames:
            at_net = sum(1 for p in gt_frames if abs(p.y - net_y) < NET_BAND)
            time_at_net_pct = at_net / len(gt_frames)
            crossings = 0
            sorted_gt = sorted(gt_frames, key=lambda p: p.frame_number)
            prev_side: int | None = None
            for p in sorted_gt:
                cur_side = 0 if p.y < net_y else 1
                if prev_side is not None and cur_side != prev_side:
                    crossings += 1
                prev_side = cur_side
        else:
            time_at_net_pct = 0.0
            crossings = 0

        # Label: use the first GT position's label, fall back to player_<id>
        label = gt_frames[0].label if gt_frames else f"player_{gt_id}"

        # Build compact pred_id spans from the ordered matched pairs.
        # A new span starts whenever the pred_id changes or there is a
        # frame gap > 1 (so downstream viewers can distinguish "continuous
        # ownership" from "dropped and re-acquired by the same pred_id").
        spans: list[tuple[int, int, int]] = []
        pairs = matched_pairs_by_gt.get(gt_id, [])
        if pairs:
            span_start, span_pred = pairs[0][0], pairs[0][1]
            prev_frame = pairs[0][0]
            for frame, pred_id in pairs[1:]:
                if pred_id != span_pred or frame != prev_frame + 1:
                    spans.append((span_start, prev_frame, span_pred))
                    span_start, span_pred = frame, pred_id
                prev_frame = frame
            spans.append((span_start, prev_frame, span_pred))

        audits.append(
            GtTrackAudit(
                gt_track_id=gt_id,
                gt_label=label,
                gt_frame_count=len(gt_frame_numbers),
                matched_frames=len(matched),
                coverage=coverage,
                distinct_pred_ids=distinct_preds,
                real_switches=gt_switches,
                missed_by_cause=missed_by_cause,
                time_at_net_pct=time_at_net_pct,
                net_crossings=crossings,
                pred_id_spans=spans,
            )
        )

    return audits


def build_convention_drift(
    gt: GroundTruthResult,
    predictions: PlayerTrackingResult,
    matches_by_frame: dict[int, list[tuple[int, int]]],
    net_y: float | None,
    team_assignments: dict[int, int] | None,
) -> ConventionDriftAudit:
    """Detect whether GT-label ↔ pred-track mapping reverses expected conventions.

    court_side_flip fires when GT labels that should be near-side (y < net_y)
    map predominantly to pred tracks whose median y is far-side (or vice versa).
    team_label_flip fires when team_assignments[majority_pred] disagrees with
    the GT side on 2+ labels.
    """
    # Majority pred-id per GT label
    label_to_pred_counts: dict[str, Counter[int]] = defaultdict(Counter)
    gt_label_y: dict[str, list[float]] = defaultdict(list)
    gt_label_of_id: dict[int, str] = {}
    for gt_p in gt.player_positions:
        gt_label_of_id[gt_p.track_id] = gt_p.label
        gt_label_y[gt_p.label].append(gt_p.y)

    pred_label_counts_by_frame_label: dict[str, Counter[int]] = defaultdict(Counter)
    for frame, matches in matches_by_frame.items():
        for gt_id, pred_id in matches:
            label = gt_label_of_id.get(gt_id, f"player_{gt_id}")
            pred_label_counts_by_frame_label[label][pred_id] += 1
    label_to_pred_counts = pred_label_counts_by_frame_label

    gt_label_to_pred_id_mode: dict[str, int] = {}
    for label, counts in label_to_pred_counts.items():
        if counts:
            gt_label_to_pred_id_mode[label] = counts.most_common(1)[0][0]

    gt_label_mean_y = {label: sum(ys) / len(ys) for label, ys in gt_label_y.items() if ys}

    # Pred mean y
    pred_y: dict[int, list[float]] = defaultdict(list)
    for pred_p in predictions.positions:
        pred_y[pred_p.track_id].append(pred_p.y)
    pred_mean_y = {tid: sum(ys) / len(ys) for tid, ys in pred_y.items() if ys}

    team_assignments = team_assignments or {}

    # Flip detection
    court_side_flip = False
    team_label_flip = False
    if net_y is not None:
        side_disagreements = 0
        team_disagreements = 0
        total = 0
        for label, pred_id in gt_label_to_pred_id_mode.items():
            if label not in gt_label_mean_y or pred_id not in pred_mean_y:
                continue
            total += 1
            gt_side = 0 if gt_label_mean_y[label] < net_y else 1
            pred_side = 0 if pred_mean_y[pred_id] < net_y else 1
            if gt_side != pred_side:
                side_disagreements += 1
            if pred_id in team_assignments:
                pred_team = team_assignments[pred_id]
                if pred_team != gt_side:
                    team_disagreements += 1
        if total > 0:
            court_side_flip = side_disagreements >= max(2, total // 2 + 1) - 1
            team_label_flip = team_disagreements >= max(2, total // 2 + 1) - 1

    return ConventionDriftAudit(
        gt_label_to_pred_id_mode=gt_label_to_pred_id_mode,
        gt_label_mean_y=gt_label_mean_y,
        pred_mean_y=pred_mean_y,
        team_assignments=dict(team_assignments),
        court_side_flip=court_side_flip,
        team_label_flip=team_label_flip,
        net_y=net_y,
    )


def build_rally_audit(
    rally_id: str,
    video_id: str,
    ground_truth: GroundTruthResult,
    predictions: PlayerTrackingResult,
    raw_positions: list[PlayerPosition] | None = None,
    iou_threshold: float = 0.5,
    smart_interpolate: bool = True,
    min_keyframe_iou_rate: float = 0.5,
) -> RallyAudit:
    """Assemble a RallyAudit from GT + predictions for one rally.

    Applies the same smart-interpolation policy as evaluate_rally() so aggregate
    numbers line up with the existing harness.
    """
    if smart_interpolate and predictions.frame_count > 0:
        ground_truth = smart_interpolate_gt(
            ground_truth, predictions, predictions.frame_count, min_keyframe_iou_rate
        )

    # Drop GT positions with out-of-range frame numbers. The LS-fps → tracking-fps
    # conversion in db._parse_ground_truth can yield negative frames (from the
    # 1-indexed → 0-indexed shift on the first rally keyframe); they'd otherwise
    # leak into missed-range output as spurious (-1, -1) entries.
    max_frame = predictions.frame_count - 1 if predictions.frame_count > 0 else 10**9
    ground_truth = GroundTruthResult(
        positions=[p for p in ground_truth.positions if 0 <= p.frame_number <= max_frame],
        frame_count=ground_truth.frame_count,
        video_width=ground_truth.video_width,
        video_height=ground_truth.video_height,
    )

    gt_by_frame, pred_by_frame, matches_by_frame, unmatched_by_frame = _match_frames(
        ground_truth, predictions, iou_threshold
    )

    net_y = predictions.court_split_y
    team_assignments = predictions.team_assignments or {}

    per_gt = build_per_gt_audits(
        gt=ground_truth,
        predictions=predictions,
        raw_positions=raw_positions,
        matches_by_frame=matches_by_frame,
        unmatched_by_frame=unmatched_by_frame,
        net_y=net_y,
        team_assignments=team_assignments,
    )

    convention = build_convention_drift(
        gt=ground_truth,
        predictions=predictions,
        matches_by_frame=matches_by_frame,
        net_y=net_y,
        team_assignments=team_assignments,
    )

    # Aggregate HOTA/MOTA — compute fresh here rather than depending on the
    # caller running evaluate_rally; cheap given we already have matches.
    total_gt = sum(len(v) for v in gt_by_frame.values())
    total_pred = sum(len(v) for v in pred_by_frame.values())
    total_tp = sum(len(m) for m in matches_by_frame.values())
    total_fp = total_pred - total_tp
    total_fn = total_gt - total_tp
    mota = 1.0 - (total_fn + total_fp) / total_gt if total_gt > 0 else 0.0

    # AssA (same formula as compute_hota_metrics)
    gt_to_pred_counts: dict[int, Counter[int]] = defaultdict(Counter)
    for matches in matches_by_frame.values():
        for gt_id, pred_id in matches:
            gt_to_pred_counts[gt_id][pred_id] += 1
    correctly_associated = sum(
        max(pc.values()) for pc in gt_to_pred_counts.values() if pc
    )
    assa = correctly_associated / total_tp if total_tp > 0 else 0.0
    deta = total_tp / (total_gt + total_pred - total_tp) if (total_gt + total_pred - total_tp) > 0 else 0.0
    hota = (deta * assa) ** 0.5 if deta > 0 and assa > 0 else 0.0

    # Raw detection summary
    raw_total = len(raw_positions) if raw_positions else 0
    raw_frames = len({p.frame_number for p in raw_positions}) if raw_positions else 0
    raw_per_frame = raw_total / predictions.frame_count if predictions.frame_count else 0.0

    real_switch_count = sum(len(g.real_switches) for g in per_gt)

    return RallyAudit(
        rally_id=rally_id,
        video_id=video_id,
        frame_count=predictions.frame_count,
        video_fps=predictions.video_fps,
        hota=hota if total_tp > 0 else None,
        mota=mota,
        aggregate_real_switches=real_switch_count,
        per_gt=per_gt,
        convention=convention,
        raw_detection_summary={
            "rawTotal": raw_total,
            "rawFrames": raw_frames,
            "rawPerFrame": raw_per_frame,
        },
    )
