"""Video tracking quality assessment for player tracking.

Computes a trackability score (0-1) based on detection coverage, track
stability, and ball tracking quality. Provides actionable suggestions
for improving tracking results on difficult videos.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from rallycut.tracking.player_tracker import PlayerPosition

if TYPE_CHECKING:
    from rallycut.court.detector import CourtDetectionInsights

logger = logging.getLogger(__name__)


@dataclass
class TrackingQualityReport:
    """Assessment of tracking quality for a video segment."""

    # Ball tracking
    ball_detection_rate: float = 0.0  # % of frames with ball detected (0-1)
    ball_trajectory_spread: float = 0.0  # Geometric mean of X/Y std

    # Player detection
    avg_detections_per_frame: float = 0.0  # Raw YOLO detections inside ROI
    primary_track_count: int = 0  # Identified primary tracks

    # Track stability
    track_creation_rate: float = 0.0  # New tracks per second
    track_destruction_rate: float = 0.0  # Tracks lost per second
    avg_track_lifespan_frames: float = 0.0

    # Repair metrics
    id_switch_count: int = 0  # From enforce_spatial_consistency
    color_split_count: int = 0  # From split_tracks_by_color
    appearance_link_count: int = 0  # From link_tracklets_by_appearance

    # Distractor detection
    unique_raw_track_count: int = 0  # Unique tracks before filtering
    stationary_bg_removed_count: int = 0  # Tracks removed by stationary background filter
    calibration_recommended: bool = False  # True if calibration would likely help

    # Court detection
    court_detected: bool = True  # default True = no court detection was attempted
    court_confidence: float = 0.0

    # Court identity resolution
    court_identity_interactions: int = 0  # Net interactions detected
    court_identity_swaps: int = 0  # Swaps applied by court identity
    uncertain_identity_count: int = 0  # Ambiguous interactions

    # Team classification
    team_classification_skipped: bool = False  # True when split_confidence != "high"

    # Global identity optimization
    global_identity_segments: int = 0  # Segments after splitting at interactions
    global_identity_remapped: int = 0  # Positions remapped to canonical IDs

    # Contact detection readiness
    contact_readiness_score: float = 0.0  # 0-1, decreases per issue
    contact_readiness_issues: list[str] = field(default_factory=list)

    # Overall score
    trackability_score: float = 0.0  # 0-1 composite score
    suggestions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to dict for JSON output."""
        return {
            "ballDetectionRate": self.ball_detection_rate,
            "ballTrajectorySpread": self.ball_trajectory_spread,
            "avgDetectionsPerFrame": self.avg_detections_per_frame,
            "primaryTrackCount": self.primary_track_count,
            "trackCreationRate": self.track_creation_rate,
            "trackDestructionRate": self.track_destruction_rate,
            "avgTrackLifespanFrames": self.avg_track_lifespan_frames,
            "idSwitchCount": self.id_switch_count,
            "colorSplitCount": self.color_split_count,
            "appearanceLinkCount": self.appearance_link_count,
            "uniqueRawTrackCount": self.unique_raw_track_count,
            "stationaryBgRemovedCount": self.stationary_bg_removed_count,
            "calibrationRecommended": self.calibration_recommended,
            "courtDetected": self.court_detected,
            "courtConfidence": self.court_confidence,
            "courtIdentityInteractions": self.court_identity_interactions,
            "courtIdentitySwaps": self.court_identity_swaps,
            "uncertainIdentityCount": self.uncertain_identity_count,
            "teamClassificationSkipped": self.team_classification_skipped,
            "globalIdentitySegments": self.global_identity_segments,
            "globalIdentityRemapped": self.global_identity_remapped,
            "contactReadinessScore": self.contact_readiness_score,
            "contactReadinessIssues": self.contact_readiness_issues,
            "trackabilityScore": self.trackability_score,
            "suggestions": self.suggestions,
        }


def compute_quality_report(
    positions: list[PlayerPosition],
    raw_positions: list[PlayerPosition],
    frame_count: int,
    video_fps: float,
    primary_track_ids: list[int],
    ball_detection_rate: float = 0.0,
    ball_positions_xy: list[tuple[float, float]] | None = None,
    id_switch_count: int = 0,
    color_split_count: int = 0,
    appearance_link_count: int = 0,
    expected_players: int = 4,
    has_court_calibration: bool = False,
    court_identity_interactions: int = 0,
    court_identity_swaps: int = 0,
    uncertain_identity_count: int = 0,
    court_detection_insights: CourtDetectionInsights | None = None,
    contact_readiness_issues: list[str] | None = None,
    stationary_bg_removed_count: int = 0,
    global_identity_segments: int = 0,
    global_identity_remapped: int = 0,
    team_classification_skipped: bool = False,
) -> TrackingQualityReport:
    """Compute a tracking quality report from tracking results.

    Args:
        positions: Final filtered positions.
        raw_positions: Raw positions before filtering.
        frame_count: Total frames in the segment.
        video_fps: Video frame rate.
        primary_track_ids: Identified primary track IDs.
        ball_detection_rate: Ball detection rate (0-1).
        ball_positions_xy: Ball (x, y) positions for spread calculation.
        id_switch_count: Number of jump-based track splits.
        color_split_count: Number of color-based splits.
        appearance_link_count: Number of tracklet appearance-based merges.
        expected_players: Expected number of court players.
        has_court_calibration: Whether court calibration is available.
        court_identity_interactions: Number of net interactions detected.
        court_identity_swaps: Number of swaps applied by court identity.
        uncertain_identity_count: Number of ambiguous interactions.
        court_detection_insights: Court detection diagnostic info.
        contact_readiness_issues: Issues affecting contact detection readiness.
        stationary_bg_removed_count: Tracks removed by stationary background filter.
        global_identity_segments: Segments from global identity optimization.
        global_identity_remapped: Positions remapped by global identity.
        team_classification_skipped: Whether team classification was skipped
            due to low split confidence.

    Returns:
        TrackingQualityReport with score and suggestions.
    """
    report = TrackingQualityReport()
    report.ball_detection_rate = ball_detection_rate
    report.primary_track_count = len(primary_track_ids)
    report.id_switch_count = id_switch_count
    report.color_split_count = color_split_count
    report.appearance_link_count = appearance_link_count
    report.court_identity_interactions = court_identity_interactions
    report.court_identity_swaps = court_identity_swaps
    report.uncertain_identity_count = uncertain_identity_count
    report.stationary_bg_removed_count = stationary_bg_removed_count
    report.global_identity_segments = global_identity_segments
    report.global_identity_remapped = global_identity_remapped
    report.team_classification_skipped = team_classification_skipped

    duration_sec = frame_count / video_fps if video_fps > 0 else 0.0

    # Ball trajectory spread
    if ball_positions_xy and len(ball_positions_xy) > 1:
        xs = [p[0] for p in ball_positions_xy]
        ys = [p[1] for p in ball_positions_xy]
        x_std = max(float(np.std(xs)), 1e-6)
        y_std = max(float(np.std(ys)), 1e-6)
        report.ball_trajectory_spread = (x_std * y_std) ** 0.5
    else:
        report.ball_trajectory_spread = 0.0

    # Average raw detections per frame
    if raw_positions and frame_count > 0:
        frames_with_detections: dict[int, int] = defaultdict(int)
        for p in raw_positions:
            frames_with_detections[p.frame_number] += 1
        if frames_with_detections:
            report.avg_detections_per_frame = (
                sum(frames_with_detections.values()) / len(frames_with_detections)
            )

    # Unique raw track count (before filtering)
    unique_raw_tracks: set[int] = set()
    if raw_positions:
        unique_raw_tracks = {p.track_id for p in raw_positions if p.track_id >= 0}
        report.unique_raw_track_count = len(unique_raw_tracks)

    # Track lifespan statistics (from raw positions)
    if raw_positions:
        tracks: dict[int, list[int]] = defaultdict(list)
        for p in raw_positions:
            if p.track_id >= 0:
                tracks[p.track_id].append(p.frame_number)

        if tracks:
            lifespans = [max(fns) - min(fns) + 1 for fns in tracks.values()]
            report.avg_track_lifespan_frames = sum(lifespans) / len(lifespans)

            if duration_sec > 0:
                report.track_creation_rate = len(tracks) / duration_sec
                # Tracks that end before the last frame
                last_frame = max(p.frame_number for p in raw_positions)
                destroyed = sum(
                    1 for fns in tracks.values()
                    if max(fns) < last_frame - 5  # 5 frame tolerance
                )
                report.track_destruction_rate = destroyed / duration_sec

    # Compute sub-scores
    total_switches = id_switch_count + color_split_count

    # Detection score (0.30): primary tracks per frame vs expected
    detection_score = 0.0
    if positions and frame_count > 0:
        primary_set = set(primary_track_ids)
        frame_primary_counts: dict[int, int] = defaultdict(int)
        for p in positions:
            if p.track_id in primary_set:
                frame_primary_counts[p.frame_number] += 1
        if frame_primary_counts:
            avg_primary = sum(frame_primary_counts.values()) / len(frame_primary_counts)
            detection_score = min(avg_primary / expected_players, 1.0)

    # Stability score (0.25): penalize switches
    stability_score = 1.0
    if duration_sec > 0:
        switches_per_sec = total_switches / duration_sec
        stability_score = max(0.0, 1.0 - switches_per_sec / 2.0)

    # Coverage score (0.25): frames with >= expected primary tracks
    coverage_score = 0.0
    if positions and frame_count > 0:
        covered_frames = sum(
            1 for count in frame_primary_counts.values()
            if count >= expected_players
        )
        total_pos_frames = len(frame_primary_counts)
        if total_pos_frames > 0:
            coverage_score = covered_frames / total_pos_frames

    # Ball score (0.20)
    ball_score = min(ball_detection_rate, 1.0)

    # Contact detection readiness
    cr_issues = contact_readiness_issues or []
    report.contact_readiness_issues = cr_issues
    report.contact_readiness_score = max(0.0, 1.0 - 0.33 * len(cr_issues))

    # Weighted composite
    report.trackability_score = (
        0.30 * detection_score
        + 0.25 * stability_score
        + 0.25 * coverage_score
        + 0.20 * ball_score
    )

    # Generate suggestions
    suggestions: list[str] = []

    if detection_score < 0.6:
        suggestions.append(
            "Consider --yolo-model yolov8m or --imgsz 1920 for far-court detection"
        )

    if stability_score < 0.5:
        suggestions.append(
            "High ID switch rate — check if players frequently cross at the net"
        )

    if coverage_score < 0.7:
        suggestions.append(
            "Primary tracks lost in many frames — players may leave frame frequently"
        )

    if ball_score < 0.3:
        suggestions.append(
            "Poor ball detection (<30%) — check video quality and lighting"
        )

    if report.primary_track_count < expected_players:
        suggestions.append(
            f"Only {report.primary_track_count} primary tracks found "
            f"(expected {expected_players})"
        )

    # Calibration recommendation: many raw tracks or high ID switches
    # suggest background distractors that court ROI masking would fix
    excess_tracks = report.unique_raw_track_count > expected_players * 2
    high_switches = total_switches > 3
    if not has_court_calibration and (excess_tracks or high_switches):
        report.calibration_recommended = True
        suggestions.append(
            f"Label court corners for this video — "
            f"{report.unique_raw_track_count} raw tracks detected "
            f"(expected ~{expected_players}), calibration ROI would "
            f"filter background distractors"
        )

    # Court detection insights
    if court_detection_insights is not None:
        report.court_detected = court_detection_insights.detected
        report.court_confidence = court_detection_insights.confidence
        if not court_detection_insights.detected:
            report.calibration_recommended = True
            for tip in court_detection_insights.recording_tips:
                if tip not in suggestions:
                    suggestions.append(tip)

    report.suggestions = suggestions

    return report
