"""Tests for tracking quality report."""

from __future__ import annotations

from rallycut.tracking.player_tracker import PlayerPosition
from rallycut.tracking.quality_report import (
    TrackingQualityReport,
    compute_quality_report,
)


def _make_positions(
    track_ids: list[int],
    frame_range: range,
) -> list[PlayerPosition]:
    """Create positions for multiple tracks across a frame range."""
    positions: list[PlayerPosition] = []
    for f in frame_range:
        for tid in track_ids:
            positions.append(PlayerPosition(
                frame_number=f,
                track_id=tid,
                x=0.5,
                y=0.5,
                width=0.05,
                height=0.15,
                confidence=0.9,
            ))
    return positions


class TestTrackingQualityReport:
    def test_to_dict(self) -> None:
        report = TrackingQualityReport(
            ball_detection_rate=0.8,
            primary_track_count=4,
            trackability_score=0.75,
            suggestions=["test suggestion"],
        )
        d = report.to_dict()
        assert d["ballDetectionRate"] == 0.8
        assert d["primaryTrackCount"] == 4
        assert d["trackabilityScore"] == 0.75
        assert d["suggestions"] == ["test suggestion"]


class TestComputeQualityReport:
    def test_perfect_tracking(self) -> None:
        """Perfect tracking (4 players in all frames) should score near 1.0."""
        frames = range(0, 300)
        positions = _make_positions([1, 2, 3, 4], frames)
        raw_positions = _make_positions([1, 2, 3, 4], frames)

        report = compute_quality_report(
            positions=positions,
            raw_positions=raw_positions,
            frame_count=300,
            video_fps=30.0,
            primary_track_ids=[1, 2, 3, 4],
            ball_detection_rate=0.9,
            ball_positions_xy=[(0.3, 0.3), (0.7, 0.7), (0.5, 0.4)],
        )

        assert report.trackability_score > 0.8
        assert report.primary_track_count == 4
        assert len(report.suggestions) == 0  # No issues

    def test_no_ball_data(self) -> None:
        """Missing ball data should reduce score and suggest checking video quality."""
        frames = range(0, 300)
        positions = _make_positions([1, 2, 3, 4], frames)

        report = compute_quality_report(
            positions=positions,
            raw_positions=positions,
            frame_count=300,
            video_fps=30.0,
            primary_track_ids=[1, 2, 3, 4],
            ball_detection_rate=0.0,
        )

        # Ball score is 0, so overall score is reduced
        assert report.trackability_score < 0.9
        assert any("ball" in s.lower() for s in report.suggestions)

    def test_high_fragmentation(self) -> None:
        """Many ID switches should lower stability score."""
        frames = range(0, 300)
        positions = _make_positions([1, 2, 3, 4], frames)

        report = compute_quality_report(
            positions=positions,
            raw_positions=positions,
            frame_count=300,
            video_fps=30.0,
            primary_track_ids=[1, 2, 3, 4],
            ball_detection_rate=0.8,
            id_switch_count=15,
            color_split_count=5,
            swap_fix_count=3,
        )

        # 23 total switches in 10 seconds = ~2.3/sec should tank stability
        assert report.trackability_score < 0.8
        assert any("ID switch" in s for s in report.suggestions)

    def test_few_primary_tracks(self) -> None:
        """Fewer primary tracks than expected should produce a suggestion."""
        frames = range(0, 300)
        positions = _make_positions([1, 2, 3], frames)

        report = compute_quality_report(
            positions=positions,
            raw_positions=positions,
            frame_count=300,
            video_fps=30.0,
            primary_track_ids=[1, 2, 3],  # Only 3, expected 4
            ball_detection_rate=0.8,
        )

        assert report.primary_track_count == 3
        assert any("3 primary tracks" in s for s in report.suggestions)

    def test_empty_positions(self) -> None:
        """Empty input should not crash."""
        report = compute_quality_report(
            positions=[],
            raw_positions=[],
            frame_count=300,
            video_fps=30.0,
            primary_track_ids=[],
            ball_detection_rate=0.0,
        )

        assert report.trackability_score >= 0.0
        assert report.primary_track_count == 0

    def test_repair_counts_in_report(self) -> None:
        """Repair counts should be correctly recorded."""
        report = compute_quality_report(
            positions=[],
            raw_positions=[],
            frame_count=100,
            video_fps=30.0,
            primary_track_ids=[],
            id_switch_count=3,
            color_split_count=2,
            swap_fix_count=1,
        )

        assert report.id_switch_count == 3
        assert report.color_split_count == 2
        assert report.swap_fix_count == 1
