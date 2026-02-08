"""
Player tracking using YOLO + ByteTrack for volleyball videos.

Based on "Visual Tracking of Athletes in Beach Volleyball Using a Single Camera"
(Mauthner et al.) - modernized to use YOLO detection with ByteTrack association
instead of particle filters.
"""

import json
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from rallycut.court.calibration import CourtCalibrator, HomographyResult

logger = logging.getLogger(__name__)

# Model configuration
DEFAULT_MODEL = "yolov8n.pt"  # Nano model - fast and sufficient for person detection
PERSON_CLASS_ID = 0  # COCO class ID for 'person'

# Progress reporting configuration
PROGRESS_INTERVAL_SECONDS = 3.0  # Report progress every 3 seconds

# Player filtering thresholds
COURT_MARGIN = 4.0  # Meters outside court boundary to allow (servers, boundary plays)
BALL_PROXIMITY_THRESHOLD = 0.15  # Normalized distance (fraction of frame diagonal)
BALL_CONFIDENCE_MIN = 0.5  # Only use high-confidence ball detections
BALL_COVERAGE_MIN = 0.3  # Minimum ball detection coverage to use ball filtering

# Track merging thresholds (video-agnostic: times in seconds, distances normalized)
MERGE_MAX_GAP_SECONDS = 2.0  # Max temporal gap to consider merging
MERGE_MAX_SPATIAL_DIST = 0.20  # 20% of frame - accounts for prediction error
MERGE_MAX_COURT_DIST = 4.0  # 4 meters on court (with prediction tolerance)
MERGE_MIN_SCORE = 0.45  # Minimum score to accept merge
MERGE_SIZE_TOLERANCE = 0.50  # Bbox dimensions within 50% (far players have more size variation)
MERGE_MIN_FRAGMENT_FRAMES = 5  # Ignore fragments shorter than this
MERGE_VELOCITY_WINDOW = 5  # Frames to average for velocity estimation
MERGE_MAX_ITERATIONS = 20  # Prevent runaway merging

# Smart player identification thresholds (volleyball-context-aware)
MIN_COURT_ENGAGEMENT = 0.15  # Must engage with court at least 15% of time
MIN_POSITION_SPREAD = 1.0  # Minimum spread in meters (tracking noise ~0.5m)
MAX_POSITION_SPREAD = 4.0  # Spread of active player covering court area


@dataclass
class PlayerPosition:
    """Single player detection/tracking result for one frame."""

    frame_number: int
    x: float  # Normalized bbox center x (0-1)
    y: float  # Normalized bbox center y (0-1)
    w: float  # Normalized bbox width (0-1)
    h: float  # Normalized bbox height (0-1)
    confidence: float  # Detection confidence 0-1
    court_x: float | None = None  # Court x coordinate in meters (if calibrated)
    court_y: float | None = None  # Court y coordinate in meters (if calibrated)

    def to_dict(self) -> dict[str, Any]:
        result = {
            "frame": self.frame_number,
            "x": round(self.x, 4),
            "y": round(self.y, 4),
            "w": round(self.w, 4),
            "h": round(self.h, 4),
            "confidence": round(self.confidence, 3),
        }
        if self.court_x is not None:
            result["courtX"] = round(self.court_x, 2)
        if self.court_y is not None:
            result["courtY"] = round(self.court_y, 2)
        return result


@dataclass
class PlayerTrack:
    """Complete track for a single player across frames."""

    track_id: int
    positions: list[PlayerPosition] = field(default_factory=list)

    @property
    def frame_count(self) -> int:
        """Number of frames this player was tracked."""
        return len(self.positions)

    @property
    def first_frame(self) -> int | None:
        """First frame where player was detected."""
        return self.positions[0].frame_number if self.positions else None

    @property
    def last_frame(self) -> int | None:
        """Last frame where player was detected."""
        return self.positions[-1].frame_number if self.positions else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "trackId": self.track_id,
            "positions": [p.to_dict() for p in self.positions],
        }


@dataclass
class PlayerTrackingResult:
    """Complete tracking result for a video segment."""

    tracks: list[PlayerTrack] = field(default_factory=list)
    frame_count: int = 0
    video_fps: float = 30.0
    video_width: int = 0
    video_height: int = 0
    processing_time_ms: float = 0.0
    model_version: str = DEFAULT_MODEL

    @property
    def player_count(self) -> int:
        """Number of unique players tracked."""
        return len(self.tracks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tracks": [t.to_dict() for t in self.tracks],
            "frameCount": self.frame_count,
            "fps": self.video_fps,
            "videoWidth": self.video_width,
            "videoHeight": self.video_height,
            "playerCount": self.player_count,
            "processingTimeMs": round(self.processing_time_ms, 1),
            "modelVersion": self.model_version,
        }

    def to_json(self, path: Path) -> None:
        """Write result to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: Path) -> "PlayerTrackingResult":
        """Load result from JSON file."""
        with open(path) as f:
            data = json.load(f)

        tracks = []
        for t in data.get("tracks", []):
            positions = [
                PlayerPosition(
                    frame_number=p["frame"],
                    x=p["x"],
                    y=p["y"],
                    w=p["w"],
                    h=p["h"],
                    confidence=p["confidence"],
                    court_x=p.get("courtX"),
                    court_y=p.get("courtY"),
                )
                for p in t.get("positions", [])
            ]
            tracks.append(PlayerTrack(track_id=t["trackId"], positions=positions))

        return cls(
            tracks=tracks,
            frame_count=data.get("frameCount", 0),
            video_fps=data.get("fps", 30.0),
            video_width=data.get("videoWidth", 0),
            video_height=data.get("videoHeight", 0),
            processing_time_ms=data.get("processingTimeMs", 0.0),
            model_version=data.get("modelVersion", DEFAULT_MODEL),
        )


class PlayerTracker:
    """
    Player tracker using YOLO + ByteTrack.

    Detects and tracks players in volleyball videos, optionally projecting
    positions to court coordinates using homography calibration.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        calibration: CourtCalibrator | None = None,
        confidence_threshold: float = 0.42,
        max_players: int = 4,  # Beach volleyball = 4 players
    ):
        """
        Initialize player tracker.

        Args:
            model: YOLO model name or path
            calibration: Optional court calibrator for position projection
            confidence_threshold: Minimum detection confidence
            max_players: Expected maximum number of players (for filtering)
        """
        self.model_name = model
        self.calibration = calibration
        self.confidence_threshold = confidence_threshold
        self.max_players = max_players
        self._model: Any = None
        self._warmed_up: bool = False

    def _load_model(self) -> Any:
        """Load YOLO model with ByteTrack."""
        if self._model is not None:
            return self._model

        from ultralytics import YOLO

        logger.info(f"Loading YOLO model: {self.model_name}")
        self._model = YOLO(self.model_name)

        return self._model

    def warmup(self) -> None:
        """Warm up model with dummy inference to avoid cold-start latency."""
        if self._warmed_up:
            return

        model = self._load_model()
        logger.info("Warming up YOLO model...")
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        model.predict(dummy, verbose=False)
        self._warmed_up = True
        logger.info("Model warmup complete")

    def _get_foot_position(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
    ) -> tuple[float, float]:
        """
        Estimate foot position from bounding box.

        For court projection, we use the bottom-center of the bounding box
        as an approximation of where the player's feet are.

        Args:
            x, y: Normalized center of bounding box
            w, h: Normalized width/height of bounding box

        Returns:
            (x, y) foot position in normalized coordinates
        """
        foot_x = x  # Same x as center
        foot_y = y + h / 2  # Bottom of bbox
        return foot_x, foot_y

    def _compute_court_presence_ratio(self, track: PlayerTrack) -> float:
        """
        Compute ratio of positions with valid court coordinates.

        A high ratio indicates the player was on-court most of the time,
        helping to filter out referees/spectators who are off-court.

        Args:
            track: Player track to analyze

        Returns:
            Ratio from 0.0 to 1.0
        """
        if not track.positions:
            return 0.0
        in_court = sum(
            1
            for p in track.positions
            if p.court_x is not None and p.court_y is not None
        )
        return in_court / len(track.positions)

    def _build_ball_lookup(
        self,
        ball_positions: list[dict[str, Any]],
    ) -> dict[int, tuple[float, float]]:
        """
        Build frame -> ball position lookup from ball detections.

        Only includes high-confidence detections.

        Args:
            ball_positions: List of ball detections with frame, x, y, confidence

        Returns:
            Dict mapping frame number to (x, y) ball position
        """
        ball_by_frame: dict[int, tuple[float, float]] = {}
        for bp in ball_positions:
            if bp.get("confidence", 0) >= BALL_CONFIDENCE_MIN:
                frame = bp.get("frame")
                if frame is not None:
                    ball_by_frame[frame] = (bp["x"], bp["y"])
        return ball_by_frame

    def _compute_ball_proximity_score(
        self,
        track: PlayerTrack,
        ball_by_frame: dict[int, tuple[float, float]],
        proximity_threshold: float = BALL_PROXIMITY_THRESHOLD,
    ) -> float:
        """
        Compute fraction of frames where player was near the ball.

        Players actively participating in the game should be near the ball
        frequently. Spectators/referees typically stay far from ball action.

        Args:
            track: Player track to analyze
            ball_by_frame: Pre-computed frame -> (x, y) ball position lookup
            proximity_threshold: Normalized distance threshold (fraction of diagonal)

        Returns:
            Ratio from 0.0 to 1.0, or 0.5 if no ball data available
        """
        near_count = 0
        matched = 0

        for pos in track.positions:
            if pos.frame_number not in ball_by_frame:
                continue
            matched += 1
            bx, by = ball_by_frame[pos.frame_number]
            # Use Euclidean distance in normalized coordinates
            dist = math.sqrt((pos.x - bx) ** 2 + (pos.y - by) ** 2)
            if dist < proximity_threshold:
                near_count += 1

        # Return 0.5 (neutral) if no matched frames
        return near_count / matched if matched > 0 else 0.5

    def _compute_court_engagement_score(self, track: PlayerTrack) -> float:
        """
        Score based on engagement with actual playing area.

        Players enter the court interior. Non-players stay in margins or outside.
        Interior positions count full, marginal positions count half.

        Args:
            track: Player track to analyze

        Returns:
            Score from 0.0 to 1.0 where 1.0 = always inside court
        """
        if not track.positions or self.calibration is None:
            return 0.5

        interior = 0
        marginal = 0
        outside = 0

        for p in track.positions:
            if p.court_x is None or p.court_y is None:
                outside += 1
                continue

            # Check court interior (strict bounds: 0 to 16m X, 0 to 8m Y)
            in_interior = 0 <= p.court_x <= 16 and 0 <= p.court_y <= 8

            if in_interior:
                interior += 1
            elif self.calibration.is_point_in_court(
                (p.court_x, p.court_y), margin=COURT_MARGIN
            ):
                # In margins (within COURT_MARGIN of court)
                marginal += 1
            else:
                outside += 1

        total = len(track.positions)
        if total == 0:
            return 0.0

        # Interior counts full, marginal counts half
        engagement = (interior + 0.5 * marginal) / total
        return min(1.0, engagement)

    def _compute_movement_spread_score(self, track: PlayerTrack) -> float:
        """
        Score based on how spread out positions are on court.

        Active players cover their court half (~4m x 4m area).
        Non-players (referees, spectators) stay in one spot with minimal spread.

        Args:
            track: Player track to analyze

        Returns:
            Score from 0.0 to 1.0 where 1.0 = high spread (active player)
        """
        court_positions = [
            (p.court_x, p.court_y)
            for p in track.positions
            if p.court_x is not None and p.court_y is not None
        ]

        if len(court_positions) < 10:
            return 0.5  # Insufficient data

        xs = [p[0] for p in court_positions]
        ys = [p[1] for p in court_positions]

        # Standard deviation in each dimension
        std_x = float(np.std(xs))
        std_y = float(np.std(ys))

        # Geometric mean captures spread in both dimensions
        spread = math.sqrt(std_x * std_y)

        # Normalize to 0-1 range
        score = (spread - MIN_POSITION_SPREAD) / (
            MAX_POSITION_SPREAD - MIN_POSITION_SPREAD
        )
        return max(0.0, min(1.0, score))

    def _compute_ball_reactivity_score(
        self,
        track: PlayerTrack,
        ball_by_frame: dict[int, tuple[float, float]],
    ) -> float:
        """
        Score based on movement correlation with ball position.

        Players react to ball (move toward it or away after hitting).
        Non-players don't correlate with ball movement.

        Uses cosine similarity between player velocity and direction to ball.
        Takes absolute value since moving toward OR away indicates reactivity.

        Args:
            track: Player track to analyze
            ball_by_frame: Pre-computed frame -> (x, y) ball position lookup

        Returns:
            Score from 0.0 to 1.0 where 1.0 = high ball reactivity
        """
        if len(ball_by_frame) < 10 or len(track.positions) < 10:
            return 0.5  # Insufficient data

        # Build frame-aligned reactivity measurements
        reactions: list[float] = []
        prev_pos: PlayerPosition | None = None

        for p in track.positions:
            if p.frame_number not in ball_by_frame:
                prev_pos = p
                continue

            if prev_pos is None:
                prev_pos = p
                continue

            ball_x, ball_y = ball_by_frame[p.frame_number]

            # Player movement vector
            player_dx = p.x - prev_pos.x
            player_dy = p.y - prev_pos.y
            player_mag = math.sqrt(player_dx**2 + player_dy**2)

            # Direction to ball
            ball_dx = ball_x - p.x
            ball_dy = ball_y - p.y
            ball_mag = math.sqrt(ball_dx**2 + ball_dy**2)

            if player_mag > 0.001 and ball_mag > 0.001:
                # Cosine similarity: positive = moving toward ball
                dot = player_dx * ball_dx + player_dy * ball_dy
                cos_sim = dot / (player_mag * ball_mag)
                # abs: moving toward OR away is reactive
                reactions.append(abs(cos_sim))

            prev_pos = p

        if not reactions:
            return 0.5

        # Average reactivity
        avg_reactivity = sum(reactions) / len(reactions)

        # Normalize: random movement = ~0.5 avg, reactive = ~0.7+
        # 0.4 baseline (random), 0.7 = score 1.0
        score = (avg_reactivity - 0.4) / 0.3
        return max(0.0, min(1.0, score))

    def _estimate_track_velocity(
        self,
        track: PlayerTrack,
        window: int = MERGE_VELOCITY_WINDOW,
    ) -> tuple[float, float]:
        """
        Estimate track velocity from last N positions.

        Returns:
            (vx, vy) in normalized coords per frame
        """
        if len(track.positions) < 2:
            return (0.0, 0.0)

        # Use last N positions (or all if fewer)
        positions = track.positions[-window:]
        if len(positions) < 2:
            return (0.0, 0.0)

        # Average velocity over the window
        total_dx = positions[-1].x - positions[0].x
        total_dy = positions[-1].y - positions[0].y
        frame_span = positions[-1].frame_number - positions[0].frame_number

        if frame_span <= 0:
            return (0.0, 0.0)

        return (total_dx / frame_span, total_dy / frame_span)

    def _predict_position_after_gap(
        self,
        track: PlayerTrack,
        gap_frames: int,
        fps: float = 30.0,
    ) -> tuple[float, float]:
        """
        Predict where a track would be after a gap, based on velocity.

        Applies velocity decay to avoid over-extrapolation.
        """
        if not track.positions:
            return (0.5, 0.5)  # Center fallback

        end_pos = track.positions[-1]
        vx, vy = self._estimate_track_velocity(track)

        # Apply velocity decay for longer gaps (players slow down/stop)
        # Decay by 20% per second, fps-aware
        decay = 0.8 ** (gap_frames / fps)

        pred_x = end_pos.x + vx * gap_frames * decay
        pred_y = end_pos.y + vy * gap_frames * decay

        # Clamp to valid range
        pred_x = max(0.0, min(1.0, pred_x))
        pred_y = max(0.0, min(1.0, pred_y))

        return (pred_x, pred_y)

    def _compute_velocity_consistency(
        self,
        track_a: PlayerTrack,
        track_b: PlayerTrack,
    ) -> float:
        """
        Score how consistent the velocities are between two tracks.

        Returns:
            0.0 (opposite directions) to 1.0 (identical velocity)
        """
        va = self._estimate_track_velocity(track_a)
        vb = self._estimate_track_velocity(track_b)

        # If both stationary, consider consistent
        mag_a = math.sqrt(va[0] ** 2 + va[1] ** 2)
        mag_b = math.sqrt(vb[0] ** 2 + vb[1] ** 2)

        if mag_a < 0.001 and mag_b < 0.001:
            return 1.0
        if mag_a < 0.001 or mag_b < 0.001:
            return 0.5  # One stationary, inconclusive

        # Cosine similarity for direction
        dot = va[0] * vb[0] + va[1] * vb[1]
        cos_sim = dot / (mag_a * mag_b)

        # Convert from [-1, 1] to [0, 1]
        direction_score = (cos_sim + 1) / 2

        # Speed similarity
        speed_ratio = min(mag_a, mag_b) / max(mag_a, mag_b)

        return 0.6 * direction_score + 0.4 * speed_ratio

    def _tracks_overlap(self, track_a: PlayerTrack, track_b: PlayerTrack) -> bool:
        """Check if two tracks have overlapping frame ranges."""
        if not track_a.positions or not track_b.positions:
            return False
        # Since we check positions above, these are guaranteed to be non-None
        a_last = track_a.positions[-1].frame_number
        b_first = track_b.positions[0].frame_number
        b_last = track_b.positions[-1].frame_number
        a_first = track_a.positions[0].frame_number
        return not (a_last < b_first or b_last < a_first)

    def _tracks_coexist_spatially(
        self,
        track_a: PlayerTrack,
        track_b: PlayerTrack,
    ) -> bool:
        """
        Check if tracks have overlapping frames at different positions.

        If track A and B both have detections at the same frame but different
        positions, they're definitely different players - don't merge.

        Returns:
            True if tracks coexist at different positions (don't merge)
        """
        frames_a = {p.frame_number: (p.x, p.y) for p in track_a.positions}
        frames_b = {p.frame_number: (p.x, p.y) for p in track_b.positions}

        common_frames = set(frames_a.keys()) & set(frames_b.keys())
        if not common_frames:
            return False  # No overlap, can't determine

        # Check if positions are far apart in any common frame
        for frame in common_frames:
            pos_a = frames_a[frame]
            pos_b = frames_b[frame]
            dist = math.sqrt((pos_a[0] - pos_b[0]) ** 2 + (pos_a[1] - pos_b[1]) ** 2)
            if dist > 0.05:  # 5% of frame = definitely different people
                return True

        return False

    def _compute_merge_score(
        self,
        earlier: PlayerTrack,
        later: PlayerTrack,
        max_gap_frames: int,
        max_dist: float,
        fps: float,
    ) -> float | None:
        """
        Score merge candidate using velocity-aware prediction.

        Returns:
            Merge score (0-1) or None if not mergeable
        """
        if not earlier.positions or not later.positions:
            return None

        # Reject if tracks coexist at different positions (different players)
        if self._tracks_coexist_spatially(earlier, later):
            logger.debug(
                f"  Reject {earlier.track_id}->{later.track_id}: "
                "coexist at different positions"
            )
            return None

        end_pos = earlier.positions[-1]
        start_pos = later.positions[0]

        # Temporal gap (use frame numbers directly since we checked positions above)
        earlier_last_frame = end_pos.frame_number
        later_first_frame = start_pos.frame_number
        gap = later_first_frame - earlier_last_frame
        if gap <= 0 or gap > max_gap_frames:
            logger.debug(
                f"  Reject {earlier.track_id}->{later.track_id}: "
                f"gap={gap} (max={max_gap_frames})"
            )
            return None

        # Predict where earlier track would be after the gap
        pred_x, pred_y = self._predict_position_after_gap(earlier, gap, fps)

        # Distance from prediction to actual start of later track
        dist = math.sqrt((pred_x - start_pos.x) ** 2 + (pred_y - start_pos.y) ** 2)
        if dist > max_dist:
            logger.debug(
                f"  Reject {earlier.track_id}->{later.track_id}: "
                f"dist={dist:.3f} (max={max_dist}), "
                f"pred=({pred_x:.2f},{pred_y:.2f}), actual=({start_pos.x:.2f},{start_pos.y:.2f})"
            )
            return None

        # Size similarity
        w_ratio = (
            min(end_pos.w, start_pos.w) / max(end_pos.w, start_pos.w)
            if max(end_pos.w, start_pos.w) > 0
            else 0
        )
        h_ratio = (
            min(end_pos.h, start_pos.h) / max(end_pos.h, start_pos.h)
            if max(end_pos.h, start_pos.h) > 0
            else 0
        )
        size_sim = (w_ratio + h_ratio) / 2

        if size_sim < (1 - MERGE_SIZE_TOLERANCE):
            logger.debug(
                f"  Reject {earlier.track_id}->{later.track_id}: "
                f"size_sim={size_sim:.2f} (min={1 - MERGE_SIZE_TOLERANCE})"
            )
            return None

        # Score components (higher = better)
        gap_score = 1.0 - (gap / max_gap_frames)
        dist_score = 1.0 - (dist / max_dist)
        velocity_score = self._compute_velocity_consistency(earlier, later)

        # Use court distance if available (more accurate)
        if (
            end_pos.court_x is not None
            and end_pos.court_y is not None
            and start_pos.court_x is not None
            and start_pos.court_y is not None
        ):
            court_dist = math.sqrt(
                (end_pos.court_x - start_pos.court_x) ** 2
                + (end_pos.court_y - start_pos.court_y) ** 2
            )
            if court_dist > MERGE_MAX_COURT_DIST:
                logger.debug(
                    f"  Reject {earlier.track_id}->{later.track_id}: "
                    f"court_dist={court_dist:.1f}m (max={MERGE_MAX_COURT_DIST}m)"
                )
                return None
            court_score = 1.0 - (court_dist / MERGE_MAX_COURT_DIST)
            # With court data: emphasize physical distance
            score = (
                0.20 * gap_score
                + 0.35 * court_score
                + 0.15 * dist_score
                + 0.15 * size_sim
                + 0.15 * velocity_score
            )
            logger.debug(
                f"  Candidate {earlier.track_id}->{later.track_id}: "
                f"score={score:.2f} (gap={gap}, court_dist={court_dist:.1f}m)"
            )
            return score

        # Without court data: rely on image coords and velocity
        score = (
            0.25 * gap_score
            + 0.35 * dist_score
            + 0.20 * size_sim
            + 0.20 * velocity_score
        )
        logger.debug(
            f"  Candidate {earlier.track_id}->{later.track_id}: "
            f"score={score:.2f} (gap={gap}, dist={dist:.3f}, size={size_sim:.2f}, vel={velocity_score:.2f})"
        )
        return score

    def _merge_two_tracks(
        self, earlier: PlayerTrack, later: PlayerTrack
    ) -> PlayerTrack:
        """Combine two non-overlapping tracks into one."""
        merged_positions = sorted(
            earlier.positions + later.positions, key=lambda p: p.frame_number
        )
        return PlayerTrack(track_id=earlier.track_id, positions=merged_positions)

    def _merge_fragmented_tracks(
        self,
        tracks: list[PlayerTrack],
        fps: float,
    ) -> list[PlayerTrack]:
        """
        Merge fragmented tracks that likely belong to the same player.

        Uses velocity-aware prediction to handle moving players.
        Video-agnostic: works with any fps and resolution.
        """
        # Early exit if not enough tracks to have fragmentation
        if len(tracks) <= 4:
            return tracks

        # Conditionally filter short fragments:
        # - If we have few tracks (<=max_players*2), keep ALL fragments
        #   to avoid losing detections of poorly-detected players
        # - If we have many tracks, filter noise to reduce merge complexity
        if len(tracks) <= self.max_players * 2:
            # Few tracks - keep all, including short fragments
            valid_tracks = tracks
            noise_tracks: list[PlayerTrack] = []
            logger.debug(
                f"Track merging: keeping all {len(tracks)} tracks "
                "(few tracks, preserving short fragments)"
            )
        else:
            # Many tracks - filter noise
            valid_tracks = [
                t for t in tracks if len(t.positions) >= MERGE_MIN_FRAGMENT_FRAMES
            ]
            noise_tracks = [
                t for t in tracks if len(t.positions) < MERGE_MIN_FRAGMENT_FRAMES
            ]

        if len(valid_tracks) <= 4:
            # Not enough valid tracks, return originals
            logger.debug(
                f"Track merging skipped: only {len(valid_tracks)} valid tracks "
                f"(need >4 to merge)"
            )
            return tracks

        logger.info(
            f"Track merging: considering {len(valid_tracks)} tracks "
            f"({len(noise_tracks)} noise fragments filtered)"
        )
        for t in sorted(valid_tracks, key=lambda x: x.first_frame or 0):
            logger.debug(
                f"  Track {t.track_id}: frames {t.first_frame}-{t.last_frame}, "
                f"{len(t.positions)} positions"
            )

        max_gap_frames = int(MERGE_MAX_GAP_SECONDS * fps)
        tracks_by_id: dict[int, PlayerTrack] = {t.track_id: t for t in valid_tracks}
        merge_count = 0

        # Greedy merging loop with iteration limit
        for _iteration in range(MERGE_MAX_ITERATIONS):
            # Find best merge candidate
            best_score = MERGE_MIN_SCORE
            best_pair: tuple[int, int] | None = None

            # Sort by first_frame for efficient pairwise comparison
            track_list = sorted(
                tracks_by_id.values(), key=lambda t: t.first_frame or 0
            )

            for i, track_a in enumerate(track_list):
                for track_b in track_list[i + 1 :]:
                    if self._tracks_overlap(track_a, track_b):
                        continue

                    # Determine which is earlier
                    if (track_a.last_frame or 0) < (track_b.first_frame or 0):
                        earlier, later = track_a, track_b
                    else:
                        earlier, later = track_b, track_a

                    score = self._compute_merge_score(
                        earlier, later, max_gap_frames, MERGE_MAX_SPATIAL_DIST, fps
                    )
                    if score is not None and score > best_score:
                        best_score = score
                        best_pair = (earlier.track_id, later.track_id)

            if best_pair is None:
                break  # No more valid merges

            # Perform merge
            earlier = tracks_by_id[best_pair[0]]
            later = tracks_by_id[best_pair[1]]
            merged = self._merge_two_tracks(earlier, later)

            tracks_by_id[best_pair[0]] = merged
            del tracks_by_id[best_pair[1]]
            merge_count += 1

            gap_frames = (
                later.positions[0].frame_number - earlier.positions[-1].frame_number
            )
            logger.debug(
                f"Merged track {best_pair[1]} into {best_pair[0]} "
                f"(score={best_score:.2f}, gap={gap_frames})"
            )

        if merge_count > 0:
            logger.info(
                f"Track merging: {len(valid_tracks)} -> {len(tracks_by_id)} tracks "
                f"({merge_count} merges, {len(noise_tracks)} noise fragments ignored)"
            )

        return list(tracks_by_id.values())

    def track_video(
        self,
        video_path: Path | str,
        start_ms: int | None = None,
        end_ms: int | None = None,
        video_fps: float | None = None,
        video_width: int | None = None,
        video_height: int | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
        ball_positions: list[dict[str, Any]] | None = None,
    ) -> PlayerTrackingResult:
        """
        Track players in a video segment.

        Args:
            video_path: Path to video file
            start_ms: Start time in milliseconds (optional)
            end_ms: End time in milliseconds (optional)
            video_fps: Video FPS (optional, avoids re-reading video)
            video_width: Video width in pixels (optional)
            video_height: Video height in pixels (optional)
            progress_callback: Optional callback(progress: float, message: str) for updates
            ball_positions: Optional ball tracking data for proximity filtering

        Returns:
            PlayerTrackingResult with all tracked players
        """
        start_time = time.time()
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Report loading progress
        if progress_callback:
            progress_callback(0.0, "Loading model...")

        # Warmup model before tracking
        self.warmup()

        # Load model
        model = self._load_model()

        # Use passed video properties or read from file
        fps = video_fps
        width = video_width
        height = video_height
        total_frames = None

        if fps is None or width is None or height is None:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            try:
                if fps is None:
                    fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if width is None:
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                if height is None:
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            finally:
                cap.release()

        # Get total frames if not already read
        if total_frames is None:
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
            else:
                total_frames = 0

        # At this point, fps/width/height are guaranteed to be set
        assert fps is not None
        assert width is not None
        assert height is not None

        # Calculate frame range
        start_frame = 0
        end_frame = total_frames

        if start_ms is not None:
            start_frame = int(start_ms / 1000 * fps)

        if end_ms is not None:
            end_frame = min(int(end_ms / 1000 * fps), total_frames)

        frames_to_process = end_frame - start_frame
        logger.info(
            f"Tracking players in frames {start_frame}-{end_frame} "
            f"({frames_to_process} frames, {fps:.1f} fps)"
        )

        # Pre-cache calibration flag to avoid repeated method calls in hot loop
        has_calibration = self.calibration is not None and self.calibration.is_calibrated

        # Prepare video source for YOLO
        # YOLO's track() handles video reading internally
        video_source = str(video_path)

        # Report tracking start
        if progress_callback:
            progress_callback(0.0, f"Tracking {frames_to_process} frames...")

        # Run tracking with ByteTrack
        # persist=True maintains track IDs across frames
        results = model.track(
            source=video_source,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[PERSON_CLASS_ID],
            conf=self.confidence_threshold,
            stream=True,
            verbose=False,
        )

        # Collect tracks
        tracks_dict: dict[int, PlayerTrack] = {}
        frame_idx = 0
        processed_frames = 0
        last_progress_time = time.time()

        for result in results:
            # Skip frames outside range
            if frame_idx < start_frame:
                frame_idx += 1
                # Report skipping progress periodically
                now = time.time()
                if progress_callback and (now - last_progress_time >= PROGRESS_INTERVAL_SECONDS):
                    progress_callback(0.0, f"Seeking to frame {frame_idx}/{start_frame}...")
                    last_progress_time = now
                continue
            if frame_idx >= end_frame:
                break

            # Process detections
            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xywhn.cpu().numpy()  # Normalized xywh
                track_ids = result.boxes.id.cpu().numpy().astype(int)
                confidences = result.boxes.conf.cpu().numpy()

                for box, track_id, conf in zip(boxes, track_ids, confidences):
                    x, y, w, h = box

                    # Get foot position for court projection
                    foot_x, foot_y = self._get_foot_position(x, y, w, h)

                    # Project to court coordinates if calibrated
                    court_x, court_y = None, None
                    if has_calibration and self.calibration is not None:
                        try:
                            court_x, court_y = self.calibration.image_to_court(
                                (foot_x, foot_y),
                                width,
                                height,
                            )
                            # Filter out-of-bounds players (COURT_MARGIN for serves/boundary plays)
                            if not self.calibration.is_point_in_court(
                                (court_x, court_y), margin=COURT_MARGIN
                            ):
                                court_x, court_y = None, None
                        except Exception as e:
                            logger.warning(f"Court projection failed: {e}")

                    # Create position
                    position = PlayerPosition(
                        frame_number=frame_idx - start_frame,  # Relative frame number
                        x=float(x),
                        y=float(y),
                        w=float(w),
                        h=float(h),
                        confidence=float(conf),
                        court_x=court_x,
                        court_y=court_y,
                    )

                    # Add to track
                    if track_id not in tracks_dict:
                        tracks_dict[track_id] = PlayerTrack(track_id=track_id)
                    tracks_dict[track_id].positions.append(position)

            frame_idx += 1
            processed_frames += 1

            # Time-based progress callback (every 2 seconds)
            now = time.time()
            if progress_callback and (now - last_progress_time >= PROGRESS_INTERVAL_SECONDS):
                progress = processed_frames / frames_to_process if frames_to_process > 0 else 0
                message = f"Tracking frame {processed_frames}/{frames_to_process}"
                progress_callback(progress, message)
                last_progress_time = now

        # Final progress
        if progress_callback:
            progress_callback(1.0, "Tracking complete")

        raw_tracks = list(tracks_dict.values())

        # Merge fragmented tracks before filtering (fps-aware)
        merged_tracks = self._merge_fragmented_tracks(raw_tracks, fps=fps)

        # Filter tracks to keep only the most prominent players
        tracks = self._filter_tracks(merged_tracks, frames_to_process, ball_positions)

        processing_time_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Completed tracking: {len(tracks)} players, "
            f"{processed_frames} frames in {processing_time_ms/1000:.1f}s"
        )

        return PlayerTrackingResult(
            tracks=tracks,
            frame_count=frames_to_process,
            video_fps=fps,
            video_width=width,
            video_height=height,
            processing_time_ms=processing_time_ms,
            model_version=self.model_name,
        )

    def _filter_tracks(
        self,
        tracks: list[PlayerTrack],
        total_frames: int,
        ball_positions: list[dict[str, Any]] | None = None,
    ) -> list[PlayerTrack]:
        """
        Filter and sort tracks to keep most prominent players.

        Uses volleyball-context-aware scoring to positively identify active players
        based on court engagement, movement spread, and ball reactivity.

        Multi-stage filtering:
        1. Hard filter: minimum track length (10% of video)
        2. Compute all scores (length, court, ball, engagement, spread, reactivity)
        3. Hard filter: court engagement >= 15% (must enter playing area)
        4. Combined volleyball-weighted scoring
        5. Keep top max_players by score

        Args:
            tracks: All detected tracks
            total_frames: Total number of frames processed
            ball_positions: Optional ball tracking data for proximity filtering

        Returns:
            Filtered list of player tracks
        """
        min_track_length = max(10, total_frames * 0.1)  # At least 10% of video
        has_calibration = self.calibration is not None and self.calibration.is_calibrated

        # Build ball lookup once for all tracks (if ball data available)
        ball_by_frame: dict[int, tuple[float, float]] = {}
        has_ball_data = False
        if ball_positions:
            ball_by_frame = self._build_ball_lookup(ball_positions)
            ball_coverage = len(ball_by_frame) / total_frames if total_frames > 0 else 0
            has_ball_data = ball_coverage >= BALL_COVERAGE_MIN

        logger.debug(
            f"Filtering {len(tracks)} tracks: "
            f"calibration={has_calibration}, ball_data={has_ball_data}"
        )

        # Stage 1: Hard filter by minimum length
        candidates = [t for t in tracks if len(t.positions) >= min_track_length]
        logger.debug(f"After length filter: {len(candidates)} tracks")

        # Stage 2: Compute all scores for candidates
        scored_tracks: list[tuple[PlayerTrack, dict[str, float]]] = []

        for track in candidates:
            scores: dict[str, float] = {
                # Core scores
                "length": min(1.0, len(track.positions) / total_frames),
                "court_presence": (
                    self._compute_court_presence_ratio(track)
                    if has_calibration
                    else 0.5
                ),
                "ball_proximity": (
                    self._compute_ball_proximity_score(track, ball_by_frame)
                    if has_ball_data
                    else 0.5
                ),
                # Volleyball-context scores
                "engagement": (
                    self._compute_court_engagement_score(track)
                    if has_calibration
                    else 0.5
                ),
                "spread": (
                    self._compute_movement_spread_score(track)
                    if has_calibration
                    else 0.5
                ),
                "reactivity": (
                    self._compute_ball_reactivity_score(track, ball_by_frame)
                    if has_ball_data
                    else 0.5
                ),
            }
            scored_tracks.append((track, scores))

        # Stage 3: Hard filter by court engagement (must enter playing area)
        if has_calibration:
            before_count = len(scored_tracks)
            scored_tracks = [
                (t, s) for t, s in scored_tracks if s["engagement"] >= MIN_COURT_ENGAGEMENT
            ]
            filtered_count = before_count - len(scored_tracks)
            if filtered_count > 0:
                logger.info(
                    f"Filtered {filtered_count} tracks with no court engagement"
                )

        # Stage 4: Compute combined scores with volleyball-aware weights
        # Note: Spread is used in scoring but not as a hard filter, since short
        # rallies may have limited player movement
        final_scored: list[tuple[PlayerTrack, float, dict[str, float]]] = []

        for track, scores in scored_tracks:
            if has_calibration and has_ball_data:
                # Full data: volleyball-context weighted
                combined = (
                    0.10 * scores["length"]
                    + 0.15 * scores["court_presence"]
                    + 0.20 * scores["engagement"]
                    + 0.20 * scores["spread"]
                    + 0.20 * scores["ball_proximity"]
                    + 0.15 * scores["reactivity"]
                )
            elif has_calibration:
                # No ball data: emphasize court metrics
                combined = (
                    0.15 * scores["length"]
                    + 0.25 * scores["court_presence"]
                    + 0.30 * scores["engagement"]
                    + 0.30 * scores["spread"]
                )
            elif has_ball_data:
                # No calibration: use ball metrics
                combined = (
                    0.30 * scores["length"]
                    + 0.40 * scores["ball_proximity"]
                    + 0.30 * scores["reactivity"]
                )
            else:
                # No court or ball data: just use length
                combined = scores["length"]

            final_scored.append((track, combined, scores))

        # Stage 5: Sort by score and keep top players
        final_scored.sort(key=lambda x: x[1], reverse=True)
        result = [track for track, _, _ in final_scored[: self.max_players]]

        # Log kept vs rejected with detailed score breakdown
        for i, (track, combined, scores) in enumerate(final_scored):
            status = "KEPT" if i < self.max_players else "REJECTED"
            logger.info(
                f"Track {track.track_id}: engage={scores['engagement']:.2f}, "
                f"spread={scores['spread']:.2f}, react={scores['reactivity']:.2f}, "
                f"score={combined:.2f} -> {status}"
            )

        # Re-assign track IDs to be 1-indexed and sequential
        for i, track in enumerate(result, start=1):
            track.track_id = i

        return result

    def track_rally(
        self,
        video_path: Path | str,
        start_ms: int,
        end_ms: int,
        calibration_data: dict[str, Any] | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
        ball_positions: list[dict[str, Any]] | None = None,
    ) -> PlayerTrackingResult:
        """
        Track players in a single rally.

        Convenience method that accepts calibration data as dict
        (from database) and handles loading.

        Args:
            video_path: Path to video file
            start_ms: Rally start time in milliseconds
            end_ms: Rally end time in milliseconds
            calibration_data: Optional calibration from database
            progress_callback: Optional progress callback
            ball_positions: Optional ball tracking data for filtering

        Returns:
            PlayerTrackingResult for the rally
        """
        # Load calibration if provided
        if calibration_data is not None:
            from rallycut.court.calibration import CourtCalibrator, CourtType

            court_type_str = calibration_data.get("courtType", "beach").lower()
            court_type = CourtType(court_type_str)
            self.calibration = CourtCalibrator(court_type)

            if calibration_data.get("homographyMatrix"):
                homography = HomographyResult.from_dict(
                    {
                        "matrix": calibration_data["homographyMatrix"],
                        "inverseMatrix": calibration_data.get(
                            "inverseMatrix",
                            np.linalg.inv(
                                np.array(calibration_data["homographyMatrix"]).reshape(
                                    3, 3
                                )
                            )
                            .flatten()
                            .tolist(),
                        ),
                        "reprojectionError": calibration_data.get(
                            "reprojectionError", 0.0
                        ),
                        "isValid": calibration_data.get("isValid", True),
                    }
                )
                self.calibration.load_calibration(homography)

        return self.track_video(
            video_path,
            start_ms=start_ms,
            end_ms=end_ms,
            progress_callback=progress_callback,
            ball_positions=ball_positions,
        )
