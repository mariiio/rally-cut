"""
Player tracking using YOLOv8n + ByteTrack for volleyball videos.

Uses lightweight YOLOv8n for person detection with ByteTrack for temporal tracking.
Optimized for CPU (~30-40 FPS on proxy video).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

if TYPE_CHECKING:
    from rallycut.tracking.ball_tracker import BallPosition
    from rallycut.tracking.player_filter import PlayerFilterConfig

logger = logging.getLogger(__name__)

# Model configuration
MODEL_NAME = "yolov8n.pt"  # YOLOv8 nano - fastest, 6MB
PERSON_CLASS_ID = 0  # COCO class ID for person
DEFAULT_CONFIDENCE = 0.25  # Lower threshold for detection (filter later)
DEFAULT_IOU = 0.45  # NMS IoU threshold

# Custom ByteTrack config for better tracking stability
BYTETRACK_CONFIG = Path(__file__).parent / "bytetrack_volleyball.yaml"


def _get_model_cache_dir() -> Path:
    """Get the cache directory for player tracking models."""
    from platformdirs import user_cache_dir

    cache_dir = Path(user_cache_dir("rallycut")) / "models" / "player_tracking"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


@dataclass
class PlayerPosition:
    """Single player detection result."""

    frame_number: int
    track_id: int  # ByteTrack assigned ID (-1 if no tracking)
    x: float  # Normalized 0-1 (bbox center, relative to video width)
    y: float  # Normalized 0-1 (bbox center, relative to video height)
    width: float  # Normalized bbox width
    height: float  # Normalized bbox height
    confidence: float  # Detection confidence 0-1

    def to_dict(self) -> dict:
        return {
            "frameNumber": self.frame_number,
            "trackId": self.track_id,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "confidence": self.confidence,
        }


@dataclass
class BallPhaseInfo:
    """Serializable ball phase event for API output."""

    phase: str  # "serve", "attack", "defense", "transition", "unknown"
    frame_start: int
    frame_end: int
    velocity: float
    ball_x: float
    ball_y: float

    def to_dict(self) -> dict:
        return {
            "phase": self.phase,
            "frameStart": self.frame_start,
            "frameEnd": self.frame_end,
            "velocity": self.velocity,
            "ballX": self.ball_x,
            "ballY": self.ball_y,
        }


@dataclass
class ServerInfo:
    """Detected server information."""

    track_id: int  # -1 if not detected
    confidence: float
    serve_frame: int
    serve_velocity: float
    is_near_court: bool

    def to_dict(self) -> dict:
        return {
            "trackId": self.track_id,
            "confidence": self.confidence,
            "serveFrame": self.serve_frame,
            "serveVelocity": self.serve_velocity,
            "isNearCourt": self.is_near_court,
        }


@dataclass
class PlayerTrackingResult:
    """Complete player tracking result for a video segment."""

    positions: list[PlayerPosition] = field(default_factory=list)
    frame_count: int = 0
    video_fps: float = 30.0
    video_width: int = 0
    video_height: int = 0
    processing_time_ms: float = 0.0
    model_version: str = MODEL_NAME

    # Court split Y for two-team filtering (debug overlay)
    # Camera is always behind baseline, so teams split by horizontal line
    court_split_y: float | None = None  # 0-1 normalized Y coordinate
    primary_track_ids: list[int] = field(default_factory=list)  # Stable track IDs
    filter_method: str | None = None  # Filter method used (e.g., "bbox_size+track_stability+two_team")

    # Ball phase detection
    ball_phases: list[BallPhaseInfo] = field(default_factory=list)
    server_info: ServerInfo | None = None

    # Ball positions for trajectory overlay
    ball_positions: list[BallPosition] = field(default_factory=list)

    @property
    def avg_players_per_frame(self) -> float:
        """Average number of players detected per frame."""
        if self.frame_count == 0:
            return 0.0
        # Count positions by frame
        frame_counts: dict[int, int] = {}
        for p in self.positions:
            frame_counts[p.frame_number] = frame_counts.get(p.frame_number, 0) + 1
        if not frame_counts:
            return 0.0
        return sum(frame_counts.values()) / len(frame_counts)

    @property
    def unique_track_count(self) -> int:
        """Number of unique track IDs assigned."""
        return len({p.track_id for p in self.positions if p.track_id >= 0})

    @property
    def detection_rate(self) -> float:
        """Percentage of frames with at least one player detected."""
        if self.frame_count == 0:
            return 0.0
        frames_with_players = len({p.frame_number for p in self.positions})
        return frames_with_players / self.frame_count

    def to_dict(self) -> dict:
        result = {
            "positions": [p.to_dict() for p in self.positions],
            "frameCount": self.frame_count,
            "videoFps": self.video_fps,
            "videoWidth": self.video_width,
            "videoHeight": self.video_height,
            "avgPlayersPerFrame": self.avg_players_per_frame,
            "uniqueTrackCount": self.unique_track_count,
            "detectionRate": self.detection_rate,
            "processingTimeMs": self.processing_time_ms,
            "modelVersion": self.model_version,
        }
        # Include court split Y for debug overlay (horizontal line)
        if self.court_split_y is not None:
            result["courtSplitY"] = self.court_split_y
        if self.primary_track_ids:
            result["primaryTrackIds"] = self.primary_track_ids
        if self.filter_method:
            result["filterMethod"] = self.filter_method

        # Ball phase detection results
        if self.ball_phases:
            result["ballPhases"] = [bp.to_dict() for bp in self.ball_phases]
        if self.server_info is not None:
            result["serverInfo"] = self.server_info.to_dict()

        # Ball positions for trajectory overlay
        if self.ball_positions:
            result["ballPositions"] = [
                {
                    "frameNumber": bp.frame_number,
                    "x": bp.x,
                    "y": bp.y,
                    "confidence": bp.confidence,
                }
                for bp in self.ball_positions
            ]

        return result

    def to_json(self, path: Path) -> None:
        """Write result to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: Path) -> PlayerTrackingResult:
        """Load result from JSON file."""
        with open(path) as f:
            data = json.load(f)

        positions = [
            PlayerPosition(
                frame_number=p["frameNumber"],
                track_id=p["trackId"],
                x=p["x"],
                y=p["y"],
                width=p["width"],
                height=p["height"],
                confidence=p["confidence"],
            )
            for p in data.get("positions", [])
        ]

        # Parse ball phases
        ball_phases = [
            BallPhaseInfo(
                phase=bp["phase"],
                frame_start=bp["frameStart"],
                frame_end=bp["frameEnd"],
                velocity=bp["velocity"],
                ball_x=bp["ballX"],
                ball_y=bp["ballY"],
            )
            for bp in data.get("ballPhases", [])
        ]

        # Parse server info
        server_data = data.get("serverInfo")
        server_info = None
        if server_data:
            server_info = ServerInfo(
                track_id=server_data["trackId"],
                confidence=server_data["confidence"],
                serve_frame=server_data["serveFrame"],
                serve_velocity=server_data["serveVelocity"],
                is_near_court=server_data["isNearCourt"],
            )

        return cls(
            positions=positions,
            frame_count=data.get("frameCount", 0),
            video_fps=data.get("videoFps", 30.0),
            video_width=data.get("videoWidth", 0),
            video_height=data.get("videoHeight", 0),
            processing_time_ms=data.get("processingTimeMs", 0.0),
            model_version=data.get("modelVersion", MODEL_NAME),
            court_split_y=data.get("courtSplitY"),
            primary_track_ids=data.get("primaryTrackIds", []),
            ball_phases=ball_phases,
            server_info=server_info,
        )

    def to_api_format(self) -> dict:
        """Convert to format expected by API/UI.

        Groups positions by frame for efficient frontend rendering.
        """
        # Group by frame
        frames: dict[int, list] = {}
        for pos in self.positions:
            if pos.frame_number not in frames:
                frames[pos.frame_number] = []
            frames[pos.frame_number].append({
                "trackId": pos.track_id,
                "x": pos.x,
                "y": pos.y,
                "width": pos.width,
                "height": pos.height,
                "confidence": pos.confidence,
            })

        # Average confidence across all positions
        avg_confidence = (
            sum(p.confidence for p in self.positions) / len(self.positions)
            if self.positions
            else 0.0
        )

        return {
            "trackingData": [
                {"frameNumber": fn, "players": players}
                for fn, players in sorted(frames.items())
            ],
            "frameCount": self.frame_count,
            "detectionRate": self.detection_rate,
            "avgConfidence": avg_confidence,
            "avgPlayerCount": self.avg_players_per_frame,
            "uniqueTrackCount": self.unique_track_count,
        }


class PlayerTracker:
    """
    Player tracker using YOLOv8n + ByteTrack.

    Uses YOLOv8 nano for fast person detection with ByteTrack for
    temporal tracking across frames.
    """

    def __init__(
        self,
        model_path: Path | None = None,
        confidence: float = DEFAULT_CONFIDENCE,
        iou: float = DEFAULT_IOU,
    ):
        """
        Initialize player tracker.

        Args:
            model_path: Optional path to YOLOv8 model. If not provided,
                       downloads to cache on first use.
            confidence: Detection confidence threshold.
            iou: NMS IoU threshold.
        """
        self.model_path = model_path
        self.confidence = confidence
        self.iou = iou
        self._model: Any = None

    def _ensure_model(self) -> Path:
        """Ensure model is available, downloading if necessary."""
        if self.model_path and self.model_path.exists():
            return self.model_path

        cache_dir = _get_model_cache_dir()
        cached_path = cache_dir / MODEL_NAME

        # YOLOv8 downloads models automatically on first use
        # We just return the expected cache path
        self.model_path = cached_path
        return cached_path

    def _load_model(self) -> Any:
        """Load YOLOv8 model with ByteTrack."""
        if self._model is not None:
            return self._model

        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics package is required for player tracking. "
                "Install with: pip install ultralytics>=8.2.0"
            )

        # Load model - ultralytics handles download automatically
        model_path = self._ensure_model()

        # Try to load from cache first, fallback to auto-download
        if model_path.exists():
            self._model = YOLO(str(model_path))
        else:
            # Download yolov8n.pt automatically
            logger.info("Downloading YOLOv8n model...")
            self._model = YOLO("yolov8n.pt")
            # Save to cache for future use
            # (ultralytics caches in its own location, but we track it)

        # Configure for CPU/GPU
        # ultralytics auto-detects available hardware
        logger.info(f"Loaded YOLOv8n model: {model_path.name}")

        return self._model

    def _decode_results(
        self,
        results: Any,
        frame_number: int,
        video_width: int,
        video_height: int,
    ) -> list[PlayerPosition]:
        """
        Decode YOLO results to PlayerPosition list.

        Args:
            results: YOLO inference results.
            frame_number: Current frame index.
            video_width: Video frame width.
            video_height: Video frame height.

        Returns:
            List of PlayerPosition for this frame.
        """
        positions: list[PlayerPosition] = []

        if results is None or len(results) == 0:
            return positions

        result = results[0]  # Single image result

        # Get boxes - format depends on whether tracking is enabled
        if hasattr(result, "boxes") and result.boxes is not None:
            boxes = result.boxes

            # Get number of detections
            n_detections = len(boxes.xyxy) if hasattr(boxes, "xyxy") and boxes.xyxy is not None else 0

            for i in range(n_detections):
                try:
                    # Get class ID - only process person class
                    cls = int(boxes.cls[i].item()) if hasattr(boxes, "cls") and boxes.cls is not None and i < len(boxes.cls) else 0
                    if cls != PERSON_CLASS_ID:
                        continue

                    # Get confidence
                    conf = float(boxes.conf[i].item()) if hasattr(boxes, "conf") and boxes.conf is not None and i < len(boxes.conf) else 1.0

                    # Get track ID if available (may have fewer elements than boxes)
                    track_id = -1
                    if hasattr(boxes, "id") and boxes.id is not None and i < len(boxes.id):
                        track_id = int(boxes.id[i].item())

                    # Get bounding box (xyxy format)
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = xyxy

                    # Convert to normalized center coordinates
                    cx = (x1 + x2) / 2 / video_width
                    cy = (y1 + y2) / 2 / video_height
                    w = (x2 - x1) / video_width
                    h = (y2 - y1) / video_height

                    positions.append(PlayerPosition(
                        frame_number=frame_number,
                        track_id=track_id,
                        x=float(cx),
                        y=float(cy),
                        width=float(w),
                        height=float(h),
                        confidence=conf,
                    ))
                except (IndexError, RuntimeError) as e:
                    # Skip this detection if there's an indexing issue
                    logger.debug(f"Skipping detection {i} due to error: {e}")
                    continue

        return positions

    def track_video(
        self,
        video_path: Path | str,
        start_ms: int | None = None,
        end_ms: int | None = None,
        stride: int = 1,
        progress_callback: Callable[[float], None] | None = None,
        ball_positions: list[BallPosition] | None = None,
        filter_enabled: bool = False,
        filter_config: PlayerFilterConfig | None = None,
    ) -> PlayerTrackingResult:
        """
        Track players in a video segment.

        Args:
            video_path: Path to video file.
            start_ms: Start time in milliseconds (optional).
            end_ms: End time in milliseconds (optional).
            stride: Process every Nth frame (1=all, 3=every 3rd for faster processing).
            progress_callback: Optional callback(progress: float) for progress updates.
            ball_positions: Ball tracking results for court player filtering.
            filter_enabled: If True, filter to court players only.
            filter_config: Configuration for player filtering (court type, thresholds).

        Returns:
            PlayerTrackingResult with all detected positions.
        """
        import time

        start_time = time.time()
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Load YOLO model
        model = self._load_model()

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        try:
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Calculate frame range
            start_frame = 0
            end_frame = total_frames

            if start_ms is not None:
                start_frame = int(start_ms / 1000 * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            if end_ms is not None:
                end_frame = min(int(end_ms / 1000 * fps), total_frames)

            total_frames_in_range = end_frame - start_frame
            # With stride, we process fewer frames
            frames_to_process = (total_frames_in_range + stride - 1) // stride
            logger.info(
                f"Processing frames {start_frame}-{end_frame} "
                f"({total_frames_in_range} frames, stride={stride}, processing {frames_to_process} frames, {fps:.1f} fps)"
            )

            positions: list[PlayerPosition] = []
            frame_idx = start_frame
            frames_processed = 0

            while frame_idx < end_frame:
                ret, frame = cap.read()
                if not ret:
                    logger.warning(
                        f"Frame read failed at index {frame_idx} "
                        f"(expected {total_frames_in_range} frames)"
                    )
                    break

                # Only process every Nth frame (stride)
                if (frame_idx - start_frame) % stride == 0:
                    # Run YOLO with ByteTrack
                    # persist=True enables tracking across frames
                    try:
                        results = model.track(
                            frame,
                            persist=True,
                            tracker=str(BYTETRACK_CONFIG),
                            conf=self.confidence,
                            iou=self.iou,
                            classes=[PERSON_CLASS_ID],
                            verbose=False,
                        )

                        # Decode results
                        frame_positions = self._decode_results(
                            results, frame_idx, video_width, video_height
                        )
                        positions.extend(frame_positions)
                    except (IndexError, RuntimeError, ValueError) as e:
                        # Handle any errors from YOLO/ByteTrack internals
                        logger.debug(f"Frame {frame_idx} tracking failed: {e}")

                    frames_processed += 1

                    # Progress callback
                    if progress_callback and frames_processed % 30 == 0:
                        progress = frames_processed / frames_to_process
                        progress_callback(progress)

                frame_idx += 1

            # Final progress
            if progress_callback:
                progress_callback(1.0)

            # Court split Y for debug overlay (horizontal line)
            court_split_y: float | None = None
            primary_track_ids: list[int] = []
            filter_method: str | None = None

            # Apply court player filtering if enabled (per-frame with track stability)
            if filter_enabled:
                from rallycut.tracking.player_filter import (
                    PlayerFilter,
                    PlayerFilterConfig,
                    stabilize_track_ids,
                )

                # Get config (or create default)
                config = filter_config or PlayerFilterConfig()

                # Step 1: Stabilize track IDs before filtering
                # This merges tracks that represent the same player
                positions, id_mapping = stabilize_track_ids(positions, config)

                player_filter = PlayerFilter(
                    ball_positions=ball_positions,
                    total_frames=total_frames_in_range,
                    config=config,
                )

                # Step 2: Analyze all positions to identify stable tracks
                # This must be done before per-frame filtering
                player_filter.analyze_tracks(positions)

                # Capture court split Y for debug overlay
                court_split_y = player_filter.court_split_y
                primary_track_ids = sorted(player_filter.primary_tracks)

                # Step 3: Group positions by frame
                frames: dict[int, list[PlayerPosition]] = {}
                for p in positions:
                    if p.frame_number not in frames:
                        frames[p.frame_number] = []
                    frames[p.frame_number].append(p)

                # Step 4: Filter each frame separately (uses track stability)
                original_count = len(positions)
                filtered_positions: list[PlayerPosition] = []
                for frame_num in sorted(frames.keys()):
                    frame_players = frames[frame_num]
                    filtered_frame = player_filter.filter(frame_players)
                    filtered_positions.extend(filtered_frame)

                positions = filtered_positions
                filter_method = player_filter.filter_method
                logger.info(
                    f"Filtered {original_count} -> {len(positions)} detections "
                    f"using {filter_method}"
                )

            processing_time_ms = (time.time() - start_time) * 1000
            effective_fps = frames_processed / (processing_time_ms / 1000) if processing_time_ms > 0 else 0
            logger.info(
                f"Completed tracking {frames_processed} frames in "
                f"{processing_time_ms/1000:.1f}s ({effective_fps:.1f} FPS)"
            )

            return PlayerTrackingResult(
                positions=positions,
                frame_count=total_frames_in_range,  # Total frames in video range (for time mapping)
                video_fps=fps,
                video_width=video_width,
                video_height=video_height,
                processing_time_ms=processing_time_ms,
                model_version=MODEL_NAME,
                court_split_y=court_split_y,
                primary_track_ids=primary_track_ids,
                filter_method=filter_method,
            )

        finally:
            cap.release()
            # Reset tracker state for next video
            if hasattr(model, "predictor") and model.predictor is not None:
                model.predictor.trackers = []

    def track_frames(
        self,
        frames: Iterator[np.ndarray],
        video_fps: float,
        video_width: int,
        video_height: int,
    ) -> Iterator[list[PlayerPosition]]:
        """
        Track players in a stream of frames.

        Args:
            frames: Iterator of BGR frames.
            video_fps: Video frame rate.
            video_width: Frame width.
            video_height: Frame height.

        Yields:
            List of PlayerPosition for each frame.
        """
        model = self._load_model()
        frame_idx = 0

        for frame in frames:
            try:
                # Run YOLO with ByteTrack
                results = model.track(
                    frame,
                    persist=True,
                    tracker=str(BYTETRACK_CONFIG),
                    conf=self.confidence,
                    iou=self.iou,
                    classes=[PERSON_CLASS_ID],
                    verbose=False,
                )

                # Decode results
                frame_positions = self._decode_results(
                    results, frame_idx, video_width, video_height
                )
            except (IndexError, RuntimeError, ValueError) as e:
                # Handle any errors from YOLO/ByteTrack internals
                logger.debug(f"Frame {frame_idx} tracking failed: {e}")
                frame_positions = []

            yield frame_positions
            frame_idx += 1

        # Reset tracker state
        if hasattr(model, "predictor") and model.predictor is not None:
            model.predictor.trackers = []
