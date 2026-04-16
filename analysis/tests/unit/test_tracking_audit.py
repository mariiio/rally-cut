"""Unit tests for tracking audit classifiers (miss cause + switch cause).

These are pure-logic tests: hand-crafted GT + pred fixtures, no ML calls, no DB.
"""

from __future__ import annotations

from rallycut.evaluation.tracking.audit import (
    MissCause,
    SwitchCause,
    SwitchEvent,
    classify_miss_cause,
    group_missed_frame_ranges,
    iter_real_switch_events,
)
from rallycut.labeling.ground_truth import GroundTruthPosition
from rallycut.tracking.player_tracker import PlayerPosition


def _gt(frame: int, track_id: int, x: float, y: float, w: float = 0.05, h: float = 0.15) -> GroundTruthPosition:
    return GroundTruthPosition(
        frame_number=frame,
        track_id=track_id,
        label=f"player_{track_id}",
        x=x,
        y=y,
        width=w,
        height=h,
    )


def _pp(frame: int, track_id: int, x: float, y: float, w: float = 0.05, h: float = 0.15, conf: float = 0.8) -> PlayerPosition:
    return PlayerPosition(
        frame_number=frame,
        track_id=track_id,
        x=x,
        y=y,
        width=w,
        height=h,
        confidence=conf,
    )


class TestClassifyMissCause:
    """classify_miss_cause: classify why a single GT frame was missed."""

    def test_out_of_frame_x_beyond_one(self) -> None:
        gt = _gt(100, 1, x=1.05, y=0.5)
        cause = classify_miss_cause(
            gt_pos=gt,
            other_gt_in_frame=[],
            raw_pred_in_frame=[],
            primary_pred_in_frame=[],
        )
        assert cause == MissCause.OUT_OF_FRAME

    def test_out_of_frame_y_below_zero(self) -> None:
        gt = _gt(100, 1, x=0.5, y=-0.02)
        cause = classify_miss_cause(
            gt_pos=gt,
            other_gt_in_frame=[],
            raw_pred_in_frame=[],
            primary_pred_in_frame=[],
        )
        assert cause == MissCause.OUT_OF_FRAME

    def test_edge_proximity_left(self) -> None:
        # Within 5% of left edge
        gt = _gt(100, 1, x=0.03, y=0.5)
        cause = classify_miss_cause(
            gt_pos=gt,
            other_gt_in_frame=[],
            raw_pred_in_frame=[],
            primary_pred_in_frame=[],
        )
        assert cause == MissCause.EDGE_PROXIMITY

    def test_edge_proximity_top(self) -> None:
        gt = _gt(100, 1, x=0.5, y=0.04)
        cause = classify_miss_cause(
            gt_pos=gt,
            other_gt_in_frame=[],
            raw_pred_in_frame=[],
            primary_pred_in_frame=[],
        )
        assert cause == MissCause.EDGE_PROXIMITY

    def test_occlusion_another_gt_nearby(self) -> None:
        # Another GT player at distance < 0.08
        gt = _gt(100, 1, x=0.5, y=0.5)
        other = _gt(100, 2, x=0.52, y=0.52)  # dist ~ 0.028
        cause = classify_miss_cause(
            gt_pos=gt,
            other_gt_in_frame=[other],
            raw_pred_in_frame=[],
            primary_pred_in_frame=[],
        )
        assert cause == MissCause.OCCLUSION

    def test_filter_drop_raw_exists_primary_does_not(self) -> None:
        # Raw YOLO detection has good IoU, but primary set does not
        gt = _gt(100, 1, x=0.5, y=0.5)
        raw_match = _pp(100, 999, x=0.51, y=0.5)  # IoU ≥ 0.3
        cause = classify_miss_cause(
            gt_pos=gt,
            other_gt_in_frame=[],
            raw_pred_in_frame=[raw_match],
            primary_pred_in_frame=[],
        )
        assert cause == MissCause.FILTER_DROP

    def test_detector_miss_no_raw_no_primary(self) -> None:
        gt = _gt(100, 1, x=0.5, y=0.5)
        cause = classify_miss_cause(
            gt_pos=gt,
            other_gt_in_frame=[],
            raw_pred_in_frame=[],
            primary_pred_in_frame=[],
        )
        assert cause == MissCause.DETECTOR_MISS

    def test_detector_miss_raw_far_away(self) -> None:
        # Raw detection exists but too far (IoU < 0.3)
        gt = _gt(100, 1, x=0.5, y=0.5)
        raw_far = _pp(100, 999, x=0.9, y=0.9)
        cause = classify_miss_cause(
            gt_pos=gt,
            other_gt_in_frame=[],
            raw_pred_in_frame=[raw_far],
            primary_pred_in_frame=[],
        )
        assert cause == MissCause.DETECTOR_MISS

    def test_priority_out_of_frame_over_edge(self) -> None:
        # x = 1.02 is both "out of frame" and "near edge" — out-of-frame wins
        gt = _gt(100, 1, x=1.02, y=0.5)
        cause = classify_miss_cause(
            gt_pos=gt,
            other_gt_in_frame=[],
            raw_pred_in_frame=[_pp(100, 999, x=1.02, y=0.5)],
            primary_pred_in_frame=[],
        )
        assert cause == MissCause.OUT_OF_FRAME

    def test_priority_edge_over_occlusion(self) -> None:
        # Near edge AND near another GT — edge wins (structural constraint)
        gt = _gt(100, 1, x=0.02, y=0.5)
        other = _gt(100, 2, x=0.04, y=0.5)
        cause = classify_miss_cause(
            gt_pos=gt,
            other_gt_in_frame=[other],
            raw_pred_in_frame=[],
            primary_pred_in_frame=[],
        )
        assert cause == MissCause.EDGE_PROXIMITY

    def test_filter_drop_detected_on_adjacent_frame_via_window(self) -> None:
        # YOLO runs at stride 2: GT on an odd frame, raw detection on the
        # adjacent even frame. The window lookup should find it and classify
        # as FILTER_DROP, not DETECTOR_MISS.
        gt = _gt(101, 1, x=0.5, y=0.5)
        raw_adjacent = _pp(100, 999, x=0.5, y=0.5)
        cause = classify_miss_cause(
            gt_pos=gt,
            other_gt_in_frame=[],
            raw_pred_in_frame=[],            # none on GT's exact frame
            primary_pred_in_frame=[],
            raw_pred_in_window=[raw_adjacent],
        )
        assert cause == MissCause.FILTER_DROP

    def test_priority_occlusion_over_filter_drop(self) -> None:
        # Occlusion and filter drop both apply — occlusion wins (explains the drop)
        gt = _gt(100, 1, x=0.5, y=0.5)
        other = _gt(100, 2, x=0.52, y=0.5)
        raw_match = _pp(100, 999, x=0.5, y=0.5)
        cause = classify_miss_cause(
            gt_pos=gt,
            other_gt_in_frame=[other],
            raw_pred_in_frame=[raw_match],
            primary_pred_in_frame=[],
        )
        assert cause == MissCause.OCCLUSION


class TestGroupMissedFrameRanges:
    """group_missed_frame_ranges: contiguous frames with same cause → (start, end)."""

    def test_empty(self) -> None:
        assert group_missed_frame_ranges([]) == {}

    def test_single_frame(self) -> None:
        result = group_missed_frame_ranges([(100, MissCause.DETECTOR_MISS)])
        assert result == {MissCause.DETECTOR_MISS: [(100, 100)]}

    def test_contiguous_same_cause(self) -> None:
        result = group_missed_frame_ranges([
            (100, MissCause.DETECTOR_MISS),
            (101, MissCause.DETECTOR_MISS),
            (102, MissCause.DETECTOR_MISS),
        ])
        assert result == {MissCause.DETECTOR_MISS: [(100, 102)]}

    def test_gap_breaks_range(self) -> None:
        # frames 100-101 contiguous, then 105-106 — two ranges
        result = group_missed_frame_ranges([
            (100, MissCause.DETECTOR_MISS),
            (101, MissCause.DETECTOR_MISS),
            (105, MissCause.DETECTOR_MISS),
            (106, MissCause.DETECTOR_MISS),
        ])
        assert result == {MissCause.DETECTOR_MISS: [(100, 101), (105, 106)]}

    def test_cause_change_breaks_range(self) -> None:
        # Same frames contiguous, but cause flips — two ranges under different keys
        result = group_missed_frame_ranges([
            (100, MissCause.OCCLUSION),
            (101, MissCause.OCCLUSION),
            (102, MissCause.DETECTOR_MISS),
        ])
        assert result == {
            MissCause.OCCLUSION: [(100, 101)],
            MissCause.DETECTOR_MISS: [(102, 102)],
        }

    def test_unsorted_input_is_sorted(self) -> None:
        result = group_missed_frame_ranges([
            (102, MissCause.DETECTOR_MISS),
            (100, MissCause.DETECTOR_MISS),
            (101, MissCause.DETECTOR_MISS),
        ])
        assert result == {MissCause.DETECTOR_MISS: [(100, 102)]}


class TestIterRealSwitchEvents:
    """iter_real_switch_events: extract SwitchEvent list from pred/match frames.

    Mirrors the logic of compute_identity_metrics but emits events instead of
    aggregate counts. Each event carries (frame, pred_id, gt_old, gt_new).
    """

    def test_no_switches_all_consistent(self) -> None:
        # Pred 10 follows gt 1 for 20 frames consistently
        pred_by_frame = {f: [(10, 0.3, 0.5, 0.05, 0.15)] for f in range(20)}
        matches_by_frame = {f: [(1, 10)] for f in range(20)}
        events = list(iter_real_switch_events(
            pred_by_frame=pred_by_frame,
            matches_by_frame=matches_by_frame,
            gt_by_frame=None,
        ))
        assert events == []

    def test_single_real_switch(self) -> None:
        # Pred 10 follows gt 1 for frames 0-9, then gt 2 for frames 10-19
        # No overlap between pred tracks, not a convergence (gt 1 and gt 2 far apart)
        pred_by_frame: dict[int, list[tuple[int, float, float, float, float]]] = {}
        matches_by_frame: dict[int, list[tuple[int, int]]] = {}
        gt_by_frame: dict[int, list[tuple[int, float, float, float, float]]] = {}
        for f in range(10):
            pred_by_frame[f] = [(10, 0.3, 0.5, 0.05, 0.15)]
            matches_by_frame[f] = [(1, 10)]
            gt_by_frame[f] = [(1, 0.3, 0.5, 0.05, 0.15), (2, 0.8, 0.5, 0.05, 0.15)]
        for f in range(10, 20):
            pred_by_frame[f] = [(10, 0.8, 0.5, 0.05, 0.15)]
            matches_by_frame[f] = [(2, 10)]
            gt_by_frame[f] = [(1, 0.3, 0.5, 0.05, 0.15), (2, 0.8, 0.5, 0.05, 0.15)]

        events = list(iter_real_switch_events(
            pred_by_frame=pred_by_frame,
            matches_by_frame=matches_by_frame,
            gt_by_frame=gt_by_frame,
        ))
        assert len(events) == 1
        ev = events[0]
        assert ev.pred_id == 10
        assert ev.gt_old == 1
        assert ev.gt_new == 2
        assert ev.frame == 10

    def test_convergence_switch_filtered(self) -> None:
        # GT 1 and GT 2 within _CONVERGENCE_DISTANCE of each other at switch point
        # — should be filtered as ambiguous, not emitted
        pred_by_frame: dict[int, list[tuple[int, float, float, float, float]]] = {}
        matches_by_frame: dict[int, list[tuple[int, int]]] = {}
        gt_by_frame: dict[int, list[tuple[int, float, float, float, float]]] = {}
        for f in range(10):
            pred_by_frame[f] = [(10, 0.50, 0.5, 0.05, 0.15)]
            matches_by_frame[f] = [(1, 10)]
            gt_by_frame[f] = [(1, 0.50, 0.5, 0.05, 0.15), (2, 0.52, 0.5, 0.05, 0.15)]
        for f in range(10, 20):
            pred_by_frame[f] = [(10, 0.52, 0.5, 0.05, 0.15)]
            matches_by_frame[f] = [(2, 10)]
            gt_by_frame[f] = [(1, 0.50, 0.5, 0.05, 0.15), (2, 0.52, 0.5, 0.05, 0.15)]

        events = list(iter_real_switch_events(
            pred_by_frame=pred_by_frame,
            matches_by_frame=matches_by_frame,
            gt_by_frame=gt_by_frame,
        ))
        assert events == []

    def test_event_aggregate_matches_identity_metrics(self) -> None:
        """Aggregate count of events should equal IdentityMetrics.num_switches."""
        from rallycut.evaluation.tracking.metrics import compute_identity_metrics

        # Two separate pred tracks, one with a switch, one without
        pred_by_frame: dict[int, list[tuple[int, float, float, float, float]]] = {}
        matches_by_frame: dict[int, list[tuple[int, int]]] = {}
        gt_by_frame: dict[int, list[tuple[int, float, float, float, float]]] = {}
        # Pred 10 on GT 1 for 20 frames (no switch), Pred 20 switches 1→2 at frame 10
        for f in range(20):
            preds = [(10, 0.1, 0.5, 0.05, 0.15)]
            matches = [(1, 10)]
            if f < 10:
                preds.append((20, 0.3, 0.5, 0.05, 0.15))
                matches.append((3, 20))
            else:
                preds.append((20, 0.8, 0.5, 0.05, 0.15))
                matches.append((4, 20))
            pred_by_frame[f] = preds
            matches_by_frame[f] = matches
            gt_by_frame[f] = [
                (1, 0.1, 0.5, 0.05, 0.15),
                (3, 0.3, 0.5, 0.05, 0.15),
                (4, 0.8, 0.5, 0.05, 0.15),
            ]

        events = list(iter_real_switch_events(
            pred_by_frame=pred_by_frame,
            matches_by_frame=matches_by_frame,
            gt_by_frame=gt_by_frame,
        ))
        id_metrics = compute_identity_metrics(
            pred_by_frame=pred_by_frame,
            matches_by_frame=matches_by_frame,
            gt_by_frame=gt_by_frame,
        )
        assert len(events) == id_metrics.num_switches


class TestClassifySwitchCause:
    """iter_real_switch_events should annotate each event with a SwitchCause."""

    def test_cause_net_crossing(self) -> None:
        # GT 1 starts above net (y<0.5), moves to net region at switch frame
        # GT 2 also at net region — both within ±0.05 of net_y=0.5
        pred_by_frame: dict[int, list[tuple[int, float, float, float, float]]] = {}
        matches_by_frame: dict[int, list[tuple[int, int]]] = {}
        gt_by_frame: dict[int, list[tuple[int, float, float, float, float]]] = {}
        for f in range(10):
            pred_by_frame[f] = [(10, 0.3, 0.30, 0.05, 0.15)]
            matches_by_frame[f] = [(1, 10)]
            gt_by_frame[f] = [(1, 0.3, 0.30, 0.05, 0.15), (2, 0.7, 0.70, 0.05, 0.15)]
        # At switch both players move to the net region (y ≈ 0.5)
        for f in range(10, 20):
            pred_by_frame[f] = [(10, 0.7, 0.48, 0.05, 0.15)]
            matches_by_frame[f] = [(2, 10)]
            gt_by_frame[f] = [(1, 0.3, 0.48, 0.05, 0.15), (2, 0.7, 0.52, 0.05, 0.15)]

        events = list(iter_real_switch_events(
            pred_by_frame=pred_by_frame,
            matches_by_frame=matches_by_frame,
            gt_by_frame=gt_by_frame,
            net_y=0.5,
            team_assignments={10: 0, 20: 1},
        ))
        assert len(events) == 1
        assert events[0].cause == SwitchCause.NET_CROSSING

    def test_cause_same_team_swap(self) -> None:
        # Both GT players on same side of net, pred swaps between them
        pred_by_frame: dict[int, list[tuple[int, float, float, float, float]]] = {}
        matches_by_frame: dict[int, list[tuple[int, int]]] = {}
        gt_by_frame: dict[int, list[tuple[int, float, float, float, float]]] = {}
        for f in range(10):
            pred_by_frame[f] = [(10, 0.3, 0.2, 0.05, 0.15)]
            matches_by_frame[f] = [(1, 10)]
            gt_by_frame[f] = [(1, 0.3, 0.2, 0.05, 0.15), (2, 0.7, 0.2, 0.05, 0.15)]
        for f in range(10, 20):
            pred_by_frame[f] = [(10, 0.7, 0.2, 0.05, 0.15)]
            matches_by_frame[f] = [(2, 10)]
            gt_by_frame[f] = [(1, 0.3, 0.2, 0.05, 0.15), (2, 0.7, 0.2, 0.05, 0.15)]

        events = list(iter_real_switch_events(
            pred_by_frame=pred_by_frame,
            matches_by_frame=matches_by_frame,
            gt_by_frame=gt_by_frame,
            net_y=0.5,
        ))
        assert len(events) == 1
        assert events[0].cause == SwitchCause.SAME_TEAM_SWAP

    def test_cause_cross_team_swap(self) -> None:
        # GT players on OPPOSITE sides of net, not near net
        pred_by_frame: dict[int, list[tuple[int, float, float, float, float]]] = {}
        matches_by_frame: dict[int, list[tuple[int, int]]] = {}
        gt_by_frame: dict[int, list[tuple[int, float, float, float, float]]] = {}
        for f in range(10):
            pred_by_frame[f] = [(10, 0.3, 0.2, 0.05, 0.15)]
            matches_by_frame[f] = [(1, 10)]
            gt_by_frame[f] = [(1, 0.3, 0.2, 0.05, 0.15), (2, 0.7, 0.8, 0.05, 0.15)]
        for f in range(10, 20):
            pred_by_frame[f] = [(10, 0.7, 0.8, 0.05, 0.15)]
            matches_by_frame[f] = [(2, 10)]
            gt_by_frame[f] = [(1, 0.3, 0.2, 0.05, 0.15), (2, 0.7, 0.8, 0.05, 0.15)]

        events = list(iter_real_switch_events(
            pred_by_frame=pred_by_frame,
            matches_by_frame=matches_by_frame,
            gt_by_frame=gt_by_frame,
            net_y=0.5,
        ))
        assert len(events) == 1
        assert events[0].cause == SwitchCause.CROSS_TEAM_SWAP


class TestInterpolatePrimaryTracks:
    """interpolate_primary_tracks: fills frame-parity gaps for primary tracks.

    This exists because production rallies can contain secondary primary tracks
    (promoted after Stage 10 interpolation) that stay at YOLO's stride, causing
    parity mismatch with GT annotations. The helper mirrors the production
    interpolate_player_gaps logic but is applied at eval time.
    """

    def _build_predictions(self, positions, primary_ids):
        from rallycut.tracking.player_tracker import PlayerTrackingResult
        return PlayerTrackingResult(
            positions=positions,
            frame_count=100,
            primary_track_ids=primary_ids,
        )

    def test_fills_odd_frame_gap_for_primary_track(self) -> None:
        from rallycut.evaluation.tracking.audit import interpolate_primary_tracks
        # Pred on frames 10, 12, 14 — odd frames missing
        pred = [_pp(10, 5, x=0.5, y=0.5), _pp(12, 5, x=0.6, y=0.5), _pp(14, 5, x=0.7, y=0.5)]
        result = interpolate_primary_tracks(self._build_predictions(pred, primary_ids=[5]))
        frames = sorted(p.frame_number for p in result.positions if p.track_id == 5)
        assert frames == [10, 11, 12, 13, 14]
        # Interpolated values should be halfway between neighbors
        got = {p.frame_number: p for p in result.positions if p.track_id == 5}
        assert abs(got[11].x - 0.55) < 1e-6
        assert abs(got[13].x - 0.65) < 1e-6

    def test_non_primary_track_not_interpolated(self) -> None:
        from rallycut.evaluation.tracking.audit import interpolate_primary_tracks
        pred = [_pp(10, 99, x=0.5, y=0.5), _pp(12, 99, x=0.6, y=0.5)]
        result = interpolate_primary_tracks(self._build_predictions(pred, primary_ids=[5]))
        frames = sorted(p.frame_number for p in result.positions if p.track_id == 99)
        assert frames == [10, 12]  # unchanged

    def test_gap_over_max_is_left_alone(self) -> None:
        from rallycut.evaluation.tracking.audit import interpolate_primary_tracks
        # Gap of 50 frames — do not interpolate (that's suspicious, likely track break)
        pred = [_pp(10, 5, x=0.5, y=0.5), _pp(60, 5, x=0.6, y=0.5)]
        result = interpolate_primary_tracks(
            self._build_predictions(pred, primary_ids=[5]), max_gap=10
        )
        frames = sorted(p.frame_number for p in result.positions if p.track_id == 5)
        assert frames == [10, 60]

    def test_empty_primary_is_noop(self) -> None:
        from rallycut.evaluation.tracking.audit import interpolate_primary_tracks
        pred = [_pp(10, 5, x=0.5, y=0.5)]
        result = interpolate_primary_tracks(self._build_predictions(pred, primary_ids=[]))
        assert len(result.positions) == 1


class TestBuildRallyAudit:
    """build_rally_audit: assemble a full RallyAudit from GT + predictions.

    These are integration-ish tests at the audit layer, but still pure (no DB,
    no video). They cover:
      - per-GT coverage + distinct_pred_ids (fragmentation signal)
      - miss-cause aggregation from the whole rally
      - convention drift flags
    """

    def _build(
        self,
        gt_positions: list[GroundTruthPosition],
        pred_positions: list[PlayerPosition],
        *,
        court_split_y: float | None = None,
        team_assignments: dict[int, int] | None = None,
        frame_count: int = 100,
        raw_positions: list[PlayerPosition] | None = None,
    ):
        from rallycut.evaluation.tracking.audit import build_rally_audit
        from rallycut.labeling.ground_truth import GroundTruthResult
        from rallycut.tracking.player_tracker import PlayerTrackingResult

        gt = GroundTruthResult(positions=gt_positions, frame_count=frame_count)
        predictions = PlayerTrackingResult(
            positions=pred_positions,
            frame_count=frame_count,
            court_split_y=court_split_y,
            team_assignments=team_assignments or {},
        )
        return build_rally_audit(
            rally_id="r1",
            video_id="v1",
            ground_truth=gt,
            predictions=predictions,
            raw_positions=raw_positions,
            smart_interpolate=False,  # keep fixtures simple
        )

    def test_full_coverage_single_gt_track(self) -> None:
        gt = [_gt(f, 1, x=0.3, y=0.5) for f in range(20)]
        pred = [_pp(f, 100, x=0.3, y=0.5) for f in range(20)]
        audit = self._build(gt, pred)

        assert len(audit.per_gt) == 1
        g = audit.per_gt[0]
        assert g.gt_track_id == 1
        assert g.gt_frame_count == 20
        assert g.matched_frames == 20
        assert g.coverage == 1.0
        assert g.distinct_pred_ids == [100]
        assert g.real_switches == []
        assert g.missed_by_cause == {}

    def test_fragmentation_two_pred_ids_cover_one_gt(self) -> None:
        gt = [_gt(f, 1, x=0.3, y=0.5) for f in range(20)]
        # Pred switches from id 100 (frames 0-9) to id 200 (frames 10-19).
        pred = (
            [_pp(f, 100, x=0.3, y=0.5) for f in range(10)]
            + [_pp(f, 200, x=0.3, y=0.5) for f in range(10, 20)]
        )
        audit = self._build(gt, pred)

        g = audit.per_gt[0]
        assert g.distinct_pred_ids == [100, 200]

    def test_detector_miss_gets_classified(self) -> None:
        # GT exists for 10 frames; predictions only for first 5 → 5 missed.
        gt = [_gt(f, 1, x=0.3, y=0.5) for f in range(10)]
        pred = [_pp(f, 100, x=0.3, y=0.5) for f in range(5)]
        audit = self._build(gt, pred)

        g = audit.per_gt[0]
        assert g.matched_frames == 5
        # 5 consecutive frames (5..9) missed, all DETECTOR_MISS
        assert MissCause.DETECTOR_MISS in g.missed_by_cause
        ranges = g.missed_by_cause[MissCause.DETECTOR_MISS]
        assert ranges == [(5, 9)]

    def test_filter_drop_uses_raw_positions(self) -> None:
        # Primary missing on frame 5, but raw detection present at same box.
        gt = [_gt(f, 1, x=0.3, y=0.5) for f in range(10)]
        pred = [_pp(f, 100, x=0.3, y=0.5) for f in range(10) if f != 5]
        raw = pred + [_pp(5, 999, x=0.3, y=0.5)]
        audit = self._build(gt, pred, raw_positions=raw)

        g = audit.per_gt[0]
        assert MissCause.FILTER_DROP in g.missed_by_cause
        assert g.missed_by_cause[MissCause.FILTER_DROP] == [(5, 5)]

    def test_convention_drift_court_side_flip(self) -> None:
        # GT player_1 is near side (y=0.25), player_2 is far side (y=0.75).
        # Pred swaps their matched sides: pred 100 is far, pred 200 is near.
        gt_p1 = [_gt(f, 1, x=0.3, y=0.25) for f in range(20)]
        gt_p2 = [_gt(f, 2, x=0.3, y=0.75) for f in range(20)]
        # Pred 100 lands near player_1's position but on opposite side geometry
        pred_100 = [_pp(f, 100, x=0.3, y=0.25) for f in range(20)]
        pred_200 = [_pp(f, 200, x=0.3, y=0.75) for f in range(20)]
        # No flip in this case — simulate the flip by reversing the team_assignments
        audit = self._build(
            gt_positions=gt_p1 + gt_p2,
            pred_positions=pred_100 + pred_200,
            court_split_y=0.5,
            team_assignments={100: 1, 200: 0},  # flipped vs side geometry
        )
        # court_side_flip stays False (predictions themselves are on correct side).
        assert audit.convention.court_side_flip is False
        # team_label_flip True: pred_100 is near (y=0.25 < 0.5) but assigned team 1.
        assert audit.convention.team_label_flip is True

    def test_no_drift_when_convention_consistent(self) -> None:
        gt_p1 = [_gt(f, 1, x=0.3, y=0.25) for f in range(20)]
        gt_p2 = [_gt(f, 2, x=0.3, y=0.75) for f in range(20)]
        pred_100 = [_pp(f, 100, x=0.3, y=0.25) for f in range(20)]
        pred_200 = [_pp(f, 200, x=0.3, y=0.75) for f in range(20)]
        audit = self._build(
            gt_positions=gt_p1 + gt_p2,
            pred_positions=pred_100 + pred_200,
            court_split_y=0.5,
            team_assignments={100: 0, 200: 1},
        )
        assert audit.convention.court_side_flip is False
        assert audit.convention.team_label_flip is False
        assert audit.convention.gt_label_to_pred_id_mode == {
            "player_1": 100,
            "player_2": 200,
        }

    def test_serializable_to_dict(self) -> None:
        gt = [_gt(f, 1, x=0.3, y=0.5) for f in range(5)]
        pred = [_pp(f, 100, x=0.3, y=0.5) for f in range(5)]
        audit = self._build(gt, pred)
        d = audit.to_dict()
        assert d["rallyId"] == "r1"
        assert d["videoId"] == "v1"
        assert isinstance(d["perGt"], list)
        assert isinstance(d["convention"], dict)
        assert isinstance(d["rawDetectionSummary"], dict)


class TestSwitchEventFields:
    def test_event_has_core_fields(self) -> None:
        ev = SwitchEvent(
            frame=100,
            pred_id=10,
            gt_old=1,
            gt_new=2,
            cause=SwitchCause.NET_CROSSING,
        )
        assert ev.frame == 100
        assert ev.pred_id == 10
        assert ev.gt_old == 1
        assert ev.gt_new == 2
        assert ev.cause == SwitchCause.NET_CROSSING
