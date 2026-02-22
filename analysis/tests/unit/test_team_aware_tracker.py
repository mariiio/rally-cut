"""Tests for team-aware BoT-SORT association penalty."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import numpy as np
import pytest

from rallycut.tracking.team_aware_tracker import (
    TeamAwareConfig,
    TeamSideTracker,
    get_court_split_y_from_calibration,
    patch_tracker_with_team_awareness,
)


@dataclass
class FakePosition:
    """Minimal position for testing (matches PlayerPosition interface)."""

    track_id: int
    y: float
    x: float = 0.5


@dataclass
class FakeSTrack:
    """Minimal STrack mock for penalty matrix tests."""

    track_id: int
    tlwh: tuple[float, float, float, float]  # (top_left_x, top_left_y, width, height) in pixels


class TestBootstrap:
    """Test team assignment bootstrap logic."""

    def _make_tracker(
        self,
        split_y: float = 0.50,
        bootstrap_frames: int = 30,
        min_tracks: int = 3,
    ) -> TeamSideTracker:
        config = TeamAwareConfig(
            enabled=True,
            bootstrap_frames=bootstrap_frames,
            min_tracks_for_bootstrap=min_tracks,
        )
        return TeamSideTracker(court_split_y=split_y, config=config)

    def test_assigns_teams_after_bootstrap(self) -> None:
        """4 tracks over 30 frames should produce correct team assignments."""
        tracker = self._make_tracker(split_y=0.50, bootstrap_frames=30)

        for _ in range(30):
            positions = [
                FakePosition(track_id=1, y=0.70),  # near
                FakePosition(track_id=2, y=0.65),  # near
                FakePosition(track_id=3, y=0.35),  # far
                FakePosition(track_id=4, y=0.40),  # far
            ]
            tracker.update(positions)

        assert tracker.is_active
        assert tracker.team_assignments[1] == 0  # near
        assert tracker.team_assignments[2] == 0  # near
        assert tracker.team_assignments[3] == 1  # far
        assert tracker.team_assignments[4] == 1  # far

    def test_not_active_before_bootstrap(self) -> None:
        """Should not be active before bootstrap_frames."""
        tracker = self._make_tracker(bootstrap_frames=30)

        for _ in range(29):
            positions = [
                FakePosition(track_id=1, y=0.70),
                FakePosition(track_id=2, y=0.65),
                FakePosition(track_id=3, y=0.35),
                FakePosition(track_id=4, y=0.40),
            ]
            tracker.update(positions)

        assert not tracker.is_active

    def test_not_active_with_too_few_tracks(self) -> None:
        """Should not activate if fewer than min_tracks_for_bootstrap."""
        tracker = self._make_tracker(bootstrap_frames=10, min_tracks=3)

        for _ in range(15):
            # Only 2 tracks
            positions = [
                FakePosition(track_id=1, y=0.70),
                FakePosition(track_id=2, y=0.35),
            ]
            tracker.update(positions)

        assert not tracker.is_active

    def test_new_track_assigned_after_bootstrap(self) -> None:
        """A new track appearing after bootstrap gets the correct team."""
        tracker = self._make_tracker(split_y=0.50, bootstrap_frames=10)

        # Bootstrap with 3 tracks
        for _ in range(10):
            positions = [
                FakePosition(track_id=1, y=0.70),
                FakePosition(track_id=2, y=0.35),
                FakePosition(track_id=3, y=0.60),
            ]
            tracker.update(positions)

        assert tracker.is_active

        # New track appears on far side
        tracker.update([FakePosition(track_id=5, y=0.30)])
        assert tracker.team_assignments[5] == 1  # far


class TestNetMarginBand:
    """Test the net-margin dead zone and penalty ramp."""

    def _make_active_tracker(
        self,
        split_y: float = 0.50,
        net_band: float = 0.05,
        penalty_scale: float = 2.0,
        max_penalty: float = 0.20,
    ) -> TeamSideTracker:
        config = TeamAwareConfig(
            enabled=True,
            bootstrap_frames=12,
            net_band=net_band,
            penalty_scale=penalty_scale,
            max_penalty=max_penalty,
            min_tracks_for_bootstrap=2,
        )
        tracker = TeamSideTracker(court_split_y=split_y, config=config)

        # Bootstrap: need 12 frames with 10+ observations per track
        for _ in range(12):
            tracker.update([
                FakePosition(track_id=1, y=0.70),
                FakePosition(track_id=2, y=0.30),
            ])

        assert tracker.is_active
        return tracker

    def test_no_penalty_within_dead_band(self) -> None:
        """Detection within net_band of split should get zero penalty."""
        tracker = self._make_active_tracker(split_y=0.50, net_band=0.05)

        # Near team track (id=1), detection at y=0.46 (4% past split, within band)
        # det_cy = (tlwh[1] + tlwh[3]/2) / img_height = (410 + 50) / 1000 = 0.46
        tracks = [FakeSTrack(track_id=1, tlwh=(0, 700, 50, 100))]
        dets = [FakeSTrack(track_id=-1, tlwh=(0, 410, 50, 100))]

        penalty = tracker.compute_penalty_matrix(tracks, dets, img_height=1000)
        assert penalty[0, 0] == 0.0

    def test_penalty_past_dead_band(self) -> None:
        """Detection clearly past the dead band should get penalty > 0."""
        tracker = self._make_active_tracker(split_y=0.50, net_band=0.05)

        # Near team track (id=1), detection at y=0.35 (15% past split, 10% past band)
        # cy=0.35 at h=1000: tlwh[1] + tlwh[3]/2 = 350
        tracks = [FakeSTrack(track_id=1, tlwh=(0, 700, 50, 100))]
        dets = [FakeSTrack(track_id=-1, tlwh=(0, 300, 50, 100))]

        penalty = tracker.compute_penalty_matrix(tracks, dets, img_height=1000)
        # signed_dist = 0.50 - 0.35 = 0.15, penetration = 0.15 - 0.05 = 0.10
        # penalty = min(0.10 * 2.0, 0.20) = 0.20
        assert penalty[0, 0] == pytest.approx(0.20, abs=0.01)

    def test_penalty_ramps_linearly(self) -> None:
        """Penalty should increase linearly with penetration depth."""
        tracker = self._make_active_tracker(
            split_y=0.50, net_band=0.05, penalty_scale=2.0, max_penalty=1.0,
        )

        # Near team track
        tracks = [FakeSTrack(track_id=1, tlwh=(0, 700, 50, 100))]

        penalties = []
        for det_y_pct in [0.44, 0.42, 0.40, 0.38]:
            # cy = det_y_pct → tlwh[1] + tlwh[3]/2 = det_y_pct * 1000
            tl_y = det_y_pct * 1000 - 50
            dets = [FakeSTrack(track_id=-1, tlwh=(0, tl_y, 50, 100))]
            p = tracker.compute_penalty_matrix(tracks, dets, img_height=1000)
            penalties.append(p[0, 0])

        # Should be monotonically increasing
        for i in range(len(penalties) - 1):
            assert penalties[i + 1] > penalties[i], (
                f"Penalty should increase: {penalties}"
            )

    def test_penalty_capped_at_max(self) -> None:
        """Penalty should not exceed max_penalty."""
        tracker = self._make_active_tracker(
            split_y=0.50, net_band=0.05, penalty_scale=2.0, max_penalty=0.20,
        )

        # Near team track, detection way on the wrong side (y=0.10)
        tracks = [FakeSTrack(track_id=1, tlwh=(0, 700, 50, 100))]
        dets = [FakeSTrack(track_id=-1, tlwh=(0, 50, 50, 100))]

        penalty = tracker.compute_penalty_matrix(tracks, dets, img_height=1000)
        assert penalty[0, 0] == pytest.approx(0.20, abs=0.01)

    def test_no_penalty_for_same_side(self) -> None:
        """Detection on the correct side should have zero penalty."""
        tracker = self._make_active_tracker(split_y=0.50, net_band=0.05)

        # Near team track (id=1), detection at y=0.70 (correct side)
        tracks = [FakeSTrack(track_id=1, tlwh=(0, 700, 50, 100))]
        dets = [FakeSTrack(track_id=-1, tlwh=(0, 650, 50, 100))]

        penalty = tracker.compute_penalty_matrix(tracks, dets, img_height=1000)
        assert penalty[0, 0] == 0.0

    def test_far_team_penalized_on_near_side(self) -> None:
        """Far team track with detection on near side should be penalized."""
        tracker = self._make_active_tracker(split_y=0.50, net_band=0.05)

        # Far team track (id=2), detection at y=0.65 (past band on near side)
        tracks = [FakeSTrack(track_id=2, tlwh=(0, 250, 50, 100))]
        dets = [FakeSTrack(track_id=-1, tlwh=(0, 600, 50, 100))]

        penalty = tracker.compute_penalty_matrix(tracks, dets, img_height=1000)
        # signed_dist = 0.65 - 0.50 = 0.15, penetration = 0.15 - 0.05 = 0.10
        assert penalty[0, 0] > 0.0


class TestTrustGating:
    """Test calibration and separation trust gates."""

    def test_insufficient_separation_stays_inactive(self) -> None:
        """Teams too close together should not activate penalty."""
        config = TeamAwareConfig(
            enabled=True,
            bootstrap_frames=10,
            min_tracks_for_bootstrap=2,
            min_team_separation=0.08,
        )
        tracker = TeamSideTracker(court_split_y=0.50, config=config)

        # Near and far teams only 0.04 apart (below threshold)
        for _ in range(10):
            tracker.update([
                FakePosition(track_id=1, y=0.52),  # barely near
                FakePosition(track_id=2, y=0.48),  # barely far
            ])

        assert not tracker.is_active

    def test_sufficient_separation_activates(self) -> None:
        """Teams well separated should activate penalty."""
        config = TeamAwareConfig(
            enabled=True,
            bootstrap_frames=10,
            min_tracks_for_bootstrap=2,
            min_team_separation=0.08,
        )
        tracker = TeamSideTracker(court_split_y=0.50, config=config)

        # Near and far teams 0.30 apart
        for _ in range(10):
            tracker.update([
                FakePosition(track_id=1, y=0.65),
                FakePosition(track_id=2, y=0.35),
            ])

        assert tracker.is_active

    def test_near_team_too_close_to_split_stays_inactive(self) -> None:
        """Near team median close to split_y should fail clearance gate."""
        config = TeamAwareConfig(
            enabled=True,
            bootstrap_frames=12,
            min_tracks_for_bootstrap=2,
            min_team_separation=0.08,
            min_clearance_from_split=0.05,
        )
        # Simulates compressed camera: split_y=0.58, near team at y=0.62
        # (clearance = 0.04 < 0.05), far team at y=0.45
        tracker = TeamSideTracker(court_split_y=0.58, config=config)

        for _ in range(12):
            tracker.update([
                FakePosition(track_id=1, y=0.62),  # near, clearance=0.04
                FakePosition(track_id=2, y=0.45),  # far, clearance=0.13
            ])

        # Inter-team gap=0.17 passes, but near clearance=0.04 < 0.05 fails
        assert not tracker.is_active

    def test_far_team_too_close_to_split_stays_inactive(self) -> None:
        """Far team median close to split_y should fail clearance gate."""
        config = TeamAwareConfig(
            enabled=True,
            bootstrap_frames=12,
            min_tracks_for_bootstrap=2,
            min_team_separation=0.08,
            min_clearance_from_split=0.05,
        )
        tracker = TeamSideTracker(court_split_y=0.50, config=config)

        for _ in range(12):
            tracker.update([
                FakePosition(track_id=1, y=0.65),  # near, clearance=0.15
                FakePosition(track_id=2, y=0.47),  # far, clearance=0.03 < 0.05
            ])

        assert not tracker.is_active


class TestPatchMechanism:
    """Test monkey-patching of BoT-SORT's get_dists."""

    def test_patch_adds_penalty_to_original(self) -> None:
        """Patched get_dists should return original + penalty."""
        # Create active tracker
        config = TeamAwareConfig(
            enabled=True,
            bootstrap_frames=12,
            min_tracks_for_bootstrap=2,
            net_band=0.05,
            penalty_scale=2.0,
            max_penalty=0.20,
        )
        tracker = TeamSideTracker(court_split_y=0.50, config=config)
        for _ in range(12):
            tracker.update([
                FakePosition(track_id=1, y=0.70),
                FakePosition(track_id=2, y=0.30),
            ])
        assert tracker.is_active

        # Use a simple namespace instead of MagicMock to avoid __call__ interception
        original_dists = np.array([[0.3, 0.5], [0.6, 0.2]])

        class FakeBoTSORT:
            def get_dists(self, tracks: list[FakeSTrack], detections: list[FakeSTrack]) -> np.ndarray:
                return original_dists.copy()

        fake_botsort = FakeBoTSORT()
        patch_tracker_with_team_awareness(fake_botsort, tracker, img_height=1000)

        # Track 1 (near), track 2 (far)
        tracks = [
            FakeSTrack(track_id=1, tlwh=(0, 700, 50, 100)),
            FakeSTrack(track_id=2, tlwh=(0, 250, 50, 100)),
        ]
        # det 0: cy=0.35 (wrong side for near), det 1: cy=0.65 (wrong side for far)
        dets = [
            FakeSTrack(track_id=-1, tlwh=(0, 300, 50, 100)),
            FakeSTrack(track_id=-1, tlwh=(0, 600, 50, 100)),
        ]

        result = fake_botsort.get_dists(tracks, dets)
        assert result.shape == (2, 2)

        # Cross-team associations should have penalty added (0.20 each)
        assert result[0, 0] == pytest.approx(0.50, abs=0.01)  # 0.3 + 0.20
        assert result[1, 1] == pytest.approx(0.40, abs=0.01)  # 0.2 + 0.20
        # Same-team associations should be unchanged
        assert result[0, 1] == pytest.approx(0.50, abs=0.01)  # 0.5 + 0.00
        assert result[1, 0] == pytest.approx(0.60, abs=0.01)  # 0.6 + 0.00

    def test_patch_noop_without_get_dists(self) -> None:
        """Should not crash if instance lacks get_dists."""
        tracker = TeamSideTracker(court_split_y=0.50)
        mock_obj = MagicMock(spec=[])  # No get_dists attribute
        del mock_obj.get_dists
        # Should not raise
        patch_tracker_with_team_awareness(mock_obj, tracker, img_height=1000)


class TestPreBootstrapPenalty:
    """Test that penalty is zero before bootstrap."""

    def test_penalty_zeros_before_bootstrap(self) -> None:
        """compute_penalty_matrix should return all zeros before active."""
        config = TeamAwareConfig(
            enabled=True,
            bootstrap_frames=30,
            min_tracks_for_bootstrap=3,
        )
        tracker = TeamSideTracker(court_split_y=0.50, config=config)

        # Feed some data but not enough for bootstrap
        for _ in range(10):
            tracker.update([
                FakePosition(track_id=1, y=0.70),
                FakePosition(track_id=2, y=0.30),
                FakePosition(track_id=3, y=0.60),
            ])

        assert not tracker.is_active

        tracks = [FakeSTrack(track_id=1, tlwh=(0, 700, 50, 100))]
        dets = [FakeSTrack(track_id=-1, tlwh=(0, 200, 50, 100))]
        penalty = tracker.compute_penalty_matrix(tracks, dets, img_height=1000)
        assert np.all(penalty == 0.0)


@dataclass
class TestGetCourtSplitY:
    """Test calibration-based court split Y extraction."""

    def test_returns_y_for_calibrated(self) -> None:
        """Should return net Y from calibration."""
        from rallycut.court.calibration import CourtCalibrator

        calibrator = CourtCalibrator()
        # Typical beach volleyball camera: near corners at bottom, far at top
        corners = [
            (0.15, 0.85),  # near-left
            (0.85, 0.85),  # near-right
            (0.70, 0.45),  # far-right
            (0.30, 0.45),  # far-left
        ]
        calibrator.calibrate(corners)

        split_y = get_court_split_y_from_calibration(calibrator)
        assert split_y is not None
        # Net should be between far line (0.45) and near line (0.85)
        assert 0.45 < split_y < 0.85

    def test_returns_none_for_uncalibrated(self) -> None:
        """Should return None if calibrator is not calibrated."""
        from rallycut.court.calibration import CourtCalibrator

        calibrator = CourtCalibrator()
        split_y = get_court_split_y_from_calibration(calibrator)
        assert split_y is None
