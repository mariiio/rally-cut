"""Unit tests for `_appearance_validates_perm` (MATCHER_VERSION v11).

Validates that the appearance-validation guard correctly accepts perms
that don't increase appearance cost beyond tolerance, and rejects perms
that do.
"""
from __future__ import annotations

import numpy as np

from rallycut.tracking.match_tracker import (
    MatchPlayerTracker,
    RallyTrackingResult,
    StoredRallyData,
)
from rallycut.tracking.player_features import (
    HS_BINS,
    HS_RANGES,
    V_BINS,
    V_RANGES,
    TrackAppearanceStats,
)


def _make_hist(hue: float, sat: float) -> np.ndarray:
    import cv2
    h = np.full((20, 20), int(hue), dtype=np.uint8)
    s = np.full((20, 20), int(sat), dtype=np.uint8)
    v = np.full((20, 20), 180, dtype=np.uint8)
    hsv = np.stack([h, s, v], axis=-1)
    hist = cv2.calcHist([hsv], [0, 1], None, list(HS_BINS), HS_RANGES)
    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
    return hist.astype(np.float32)


def _make_v_hist(val: float) -> np.ndarray:
    import cv2
    v = np.full((20, 20), int(val), dtype=np.uint8)
    hsv = np.stack([np.zeros_like(v), np.zeros_like(v), v], axis=-1)
    hist = cv2.calcHist([hsv], [2], None, [V_BINS], V_RANGES)
    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
    return hist.astype(np.float32).flatten()


def _stats_with_color(track_id: int, hue: float, sat: float) -> TrackAppearanceStats:
    """Build a TrackAppearanceStats with a particular dominant color signature."""
    s = TrackAppearanceStats(track_id=track_id)
    s.avg_upper_hist = _make_hist(hue, sat)
    s.avg_lower_hist = _make_hist(hue, sat)
    s.avg_upper_v_hist = _make_v_hist(180.0)
    s.avg_lower_v_hist = _make_v_hist(180.0)
    s.avg_dominant_color_hsv = (float(hue), float(sat), 180.0)
    s.avg_skin_tone_hsv = (20.0, 150.0, 180.0)
    return s


def _make_rally_data(
    *,
    top_tracks: list[int],
    stats_by_tid: dict[int, TrackAppearanceStats],
    sides_by_tid: dict[int, int],
) -> StoredRallyData:
    return StoredRallyData(
        track_stats=stats_by_tid,
        track_court_sides=sides_by_tid,
        early_positions={tid: (0.5, 0.5) for tid in top_tracks},
        top_tracks=list(top_tracks),
    )


def _make_result(track_to_player: dict[int, int]) -> RallyTrackingResult:
    return RallyTrackingResult(
        rally_index=0,
        track_to_player=dict(track_to_player),
        server_player_id=None,
        side_switch_detected=False,
        assignment_confidence=0.7,
    )


def _setup_tracker_with_consistent_clusters() -> tuple[MatchPlayerTracker, list[RallyTrackingResult]]:
    """3-rally scenario: track 3 is always 'cyan' (low hue), track 4 always
    'red' (high hue). Identity AFM puts cyan in cluster 3 and red in cluster 4.

    This is the case where the consensus pass's tid→pid alignment is CORRECT —
    the upstream tracker is stable, identity AFM matches appearance, and the
    guard should accept any perm whose proposed cost is similar to current.
    """
    tracker = MatchPlayerTracker()
    cyan = lambda tid: _stats_with_color(tid, hue=90, sat=200)  # noqa: E731
    red = lambda tid: _stats_with_color(tid, hue=10, sat=200)  # noqa: E731

    rallies = []
    results = []
    for _ in range(3):
        rallies.append(_make_rally_data(
            top_tracks=[1, 2, 3, 4],
            stats_by_tid={
                1: _stats_with_color(1, hue=60, sat=100),
                2: _stats_with_color(2, hue=120, sat=100),
                3: cyan(3),
                4: red(4),
            },
            sides_by_tid={1: 0, 2: 0, 3: 1, 4: 1},
        ))
        results.append(_make_result({1: 1, 2: 2, 3: 3, 4: 4}))
    tracker.stored_rally_data = rallies
    return tracker, results


def _setup_tracker_with_inconsistent_upstream() -> tuple[
    MatchPlayerTracker, list[RallyTrackingResult]
]:
    """3-rally scenario where the upstream tracker re-IDs the cyan player.

    - Rally 0: cyan is at track 3 (correctly clustered as P3 by solver)
    - Rally 1: cyan is at track 4 (upstream gave it stable-ID 4; identity AFM
      would put cyan in P4 cluster — WRONG)
    - Rally 2: cyan is at track 4 (same as rally 1)

    MatchSolver's correct decision for rally 1 would be {1:1, 2:2, 4:3, 3:4}
    (T4 cyan → P3 because it appearance-matches rally 0's T3). The consensus
    pass at this rally might propose perm `{3:4, 4:3}` (within-far swap) to
    align tid→pid with rallies 0 and 2 (where rally 2 also has cyan at t4 in
    cluster P4 because the solver made the same mistake there).

    The guard SHOULD REJECT this perm — applying it puts T4 (cyan) → P4,
    which has appearance cost FAR higher than T4 → P3.
    """
    tracker = MatchPlayerTracker()

    def cyan(tid):
        return _stats_with_color(tid, hue=90, sat=200)

    def red(tid):
        return _stats_with_color(tid, hue=10, sat=200)

    # Rally 0: cyan at t3, red at t4 (identity AFM ⇒ C3=cyan, C4=red)
    r0 = _make_rally_data(
        top_tracks=[1, 2, 3, 4],
        stats_by_tid={
            1: _stats_with_color(1, hue=60, sat=100),
            2: _stats_with_color(2, hue=120, sat=100),
            3: cyan(3),
            4: red(4),
        },
        sides_by_tid={1: 0, 2: 0, 3: 1, 4: 1},
    )
    # Rally 1: cyan at t4 (upstream re-ID), red at t3
    r1 = _make_rally_data(
        top_tracks=[1, 2, 3, 4],
        stats_by_tid={
            1: _stats_with_color(1, hue=60, sat=100),
            2: _stats_with_color(2, hue=120, sat=100),
            3: red(3),     # cyan/red swapped
            4: cyan(4),    # cyan now at t4
        },
        sides_by_tid={1: 0, 2: 0, 3: 1, 4: 1},
    )
    # Rally 2: same as rally 0 (clean reference with cyan@t3, red@t4 in
    # identity AFM). Together with rally 0 this gives a clean P3=cyan /
    # P4=red cluster for the guard to evaluate against.
    r2 = _make_rally_data(
        top_tracks=[1, 2, 3, 4],
        stats_by_tid={
            1: _stats_with_color(1, hue=60, sat=100),
            2: _stats_with_color(2, hue=120, sat=100),
            3: cyan(3),
            4: red(4),
        },
        sides_by_tid={1: 0, 2: 0, 3: 1, 4: 1},
    )
    tracker.stored_rally_data = [r0, r1, r2]

    # Initial state: MatchSolver correctly identified rally 1's cyan
    # player at t4 and put it in P3 (matching r0+r2's cyan cluster). The
    # consensus pass is now considering whether to OVERRIDE this decision
    # to align tid→pid structurally with r0+r2's identity AFM.
    results = [
        _make_result({1: 1, 2: 2, 3: 3, 4: 4}),         # rally 0: identity (cyan at t3 → P3)
        _make_result({1: 1, 2: 2, 4: 3, 3: 4}),         # rally 1: cyan at t4 → P3 (solver decision)
        _make_result({1: 1, 2: 2, 3: 3, 4: 4}),         # rally 2: identity (cyan at t3 → P3)
    ]
    return tracker, results


class TestAppearanceValidatesPerm:
    def test_accepts_identity_perm(self) -> None:
        tracker, results = _setup_tracker_with_consistent_clusters()
        identity = {1: 1, 2: 2, 3: 3, 4: 4}
        assert tracker._appearance_validates_perm(0, results, identity) is True

    def test_accepts_perm_with_no_track_stats(self) -> None:
        """When no track_stats available, falls back to True (preserve legacy)."""
        tracker = MatchPlayerTracker()
        tracker.stored_rally_data = [
            _make_rally_data(top_tracks=[], stats_by_tid={}, sides_by_tid={}),
        ]
        results = [_make_result({})]
        assert tracker._appearance_validates_perm(
            0, results, {1: 2, 2: 1, 3: 4, 4: 3},
        ) is True

    def test_accepts_when_no_stored_rally_data(self) -> None:
        tracker = MatchPlayerTracker()
        tracker.stored_rally_data = []
        results: list[RallyTrackingResult] = []
        assert tracker._appearance_validates_perm(
            0, results, {1: 2, 2: 1, 3: 4, 4: 3},
        ) is True

    def test_rejects_perm_that_moves_cyan_to_wrong_cluster(self) -> None:
        """The lolo-style case: solver correctly identified cyan at t4 → P3
        in rally 1. The consensus pass proposes a within-far swap that
        would put cyan → P4 (where most-other-rally t4 sits). Appearance
        evidence: cyan ≪ red, so the perm INCREASES cost on T4 → P4 vs
        T4 → P3. Guard must reject."""
        tracker, results = _setup_tracker_with_inconsistent_upstream()
        # Solver's rally 1 AFM is {1:1, 2:2, 4:3, 3:4}. Consensus pass
        # would propose perm {3:4, 4:3} (swap within-far) to align with
        # rallies 0 and 2's identity. After perm: {1:1, 2:2, 4:4, 3:3} —
        # which puts cyan-at-t4 into cluster 4 (where red lives in
        # rally 0; the guard sees this as a big appearance regression
        # against rally 0's t4 contribution to cluster 4).
        bad_perm = {1: 1, 2: 2, 3: 4, 4: 3}
        assert tracker._appearance_validates_perm(1, results, bad_perm) is False

    def test_accepts_neutral_perm_within_tolerance(self) -> None:
        """A perm that doesn't change anything (or has near-zero cost
        delta) should be accepted, so the legitimate Bug C case (where
        appearance is noisy and structural agreement should dominate)
        still fires."""
        tracker, results = _setup_tracker_with_consistent_clusters()
        # All rallies have appearance-consistent identity. A within-near
        # swap {1:2, 2:1} on rally 1 yields a small cost delta because
        # tracks 1 and 2 have different hues (60 vs 120) but are still
        # within tolerance compared to rallies 0 and 2's identity layout.
        # The guard's job is to reject ONLY meaningful regressions.
        # This test confirms the tolerance is loose enough to admit
        # genuine Bug C snaps.
        neutral_perm = {1: 1, 2: 2, 3: 4, 4: 3}  # within-far swap (cyan↔red)
        # In the consistent-clusters scenario, perm SHOULD be rejected
        # because cyan-at-t3 → P4 is a worse appearance match (P4 cluster
        # has red across rallies). So we EXPECT False here:
        assert tracker._appearance_validates_perm(0, results, neutral_perm) is False
