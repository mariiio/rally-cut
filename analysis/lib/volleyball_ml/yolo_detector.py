"""
YOLO adapter for ball detection.

Adapted from volleyball_analytics for RallyCut CLI use.
Original: https://github.com/masouduut94/volleyball_analytics
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from rallycut.core.models import BallPosition

if TYPE_CHECKING:
    from ultralytics import YOLO

logger = logging.getLogger(__name__)


class BallDetector:
    """
    Specialized YOLO-based ball detection.

    Uses a model trained specifically for volleyball ball detection.
    Supports multi-candidate detection with aspect ratio filtering.
    """

    def __init__(
        self,
        model_path: Path | None = None,
        device: str = "cpu",
        confidence_threshold: float = 0.5,
        max_candidates: int = 5,
        min_aspect_ratio: float = 0.7,
        max_aspect_ratio: float = 1.4,
    ):
        self.model_path = model_path
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.max_candidates = max_candidates
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self._model: YOLO | None = None
        self._ball_class_id: int | None = None

    def _load_model(self) -> None:
        """Lazy load the YOLO model and detect ball class ID."""
        if self._model is not None:
            return

        from ultralytics import YOLO

        if self.model_path and self.model_path.exists():
            model_source = str(self.model_path)
            logger.info(f"Loading volleyball ball detector from {model_source}")
        else:
            # Fallback to pretrained model - warn user
            model_source = "yolov8n.pt"
            logger.warning(
                f"Volleyball ball detector weights not found at {self.model_path}. "
                f"Falling back to generic {model_source} - ball detection may be unreliable!"
            )

        self._model = YOLO(model_source)

        # Dynamically detect ball class ID
        self._ball_class_id = self._find_ball_class_id()
        if self._ball_class_id is None:
            logger.warning(
                f"Could not find 'ball' or 'volleyball' class in model. "
                f"Available classes: {self._model.names}. Using class 0."
            )
            self._ball_class_id = 0
        else:
            logger.info(
                f"Using class {self._ball_class_id} "
                f"('{self._model.names[self._ball_class_id]}') for ball detection"
            )

    def _find_ball_class_id(self) -> int | None:
        """Find the class ID for ball/volleyball in the model."""
        if self._model is None:
            return None

        names = self._model.names
        # Look for ball-related class names (case-insensitive)
        ball_keywords = ["ball", "volleyball", "volley"]

        for class_id, class_name in names.items():
            class_name_lower = class_name.lower()
            for keyword in ball_keywords:
                if keyword in class_name_lower:
                    return class_id

        return None

    def detect_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
    ) -> BallPosition | None:
        """
        Detect ball in a single frame (returns best candidate).

        Args:
            frame: BGR frame from OpenCV
            frame_idx: Frame index

        Returns:
            BallPosition if ball detected, None otherwise
        """
        candidates = self.detect_frame_candidates(frame, frame_idx)
        return candidates[0] if candidates else None

    def detect_frame_candidates(
        self,
        frame: np.ndarray,
        frame_idx: int,
    ) -> list[BallPosition]:
        """
        Detect all ball candidates in a single frame.

        Returns multiple candidates sorted by confidence for temporal validation.

        Args:
            frame: BGR frame from OpenCV
            frame_idx: Frame index

        Returns:
            List of BallPosition candidates sorted by confidence (highest first)
        """
        self._load_model()
        assert self._model is not None  # For type checker

        results = self._model.predict(
            frame,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
            classes=[self._ball_class_id],
        )

        candidates = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:  # type: ignore[attr-defined]
                conf = float(box.conf)
                bbox_coords = box.xyxy[0].tolist()

                # Calculate bounding box dimensions
                width = bbox_coords[2] - bbox_coords[0]
                height = bbox_coords[3] - bbox_coords[1]

                # Aspect ratio filtering - volleyball should be roughly circular
                if height > 0:
                    aspect_ratio = width / height
                    if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
                        # Skip non-circular detections (likely false positives)
                        continue

                # Get center point
                center_x = (bbox_coords[0] + bbox_coords[2]) / 2
                center_y = (bbox_coords[1] + bbox_coords[3]) / 2

                candidates.append(
                    BallPosition(
                        frame_idx=frame_idx,
                        x=center_x,
                        y=center_y,
                        confidence=conf,
                        is_predicted=False,
                        bbox_width=width,
                        bbox_height=height,
                    )
                )

        # Sort by confidence and return top candidates
        candidates.sort(key=lambda p: p.confidence, reverse=True)
        return candidates[: self.max_candidates]

    def detect_batch(
        self,
        frames: list[np.ndarray],
        frame_indices: list[int],
    ) -> list[BallPosition]:
        """
        Detect ball in a batch of frames.

        Args:
            frames: List of BGR frames
            frame_indices: Corresponding frame indices

        Returns:
            List of BallPosition for frames where ball was detected
        """
        self._load_model()
        assert self._model is not None  # For type checker

        positions = []

        # Process in batches
        batch_size = 8 if self.device == "cuda" else 1

        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i : i + batch_size]
            batch_indices = frame_indices[i : i + batch_size]

            results = self._model.predict(
                batch_frames,
                conf=self.confidence_threshold,
                device=self.device,
                verbose=False,
                classes=[self._ball_class_id],
            )

            for result, frame_idx in zip(results, batch_indices):
                # Find best valid detection in this frame
                best_conf = 0.0
                best_pos = None
                best_size = None

                if result.boxes is None:
                    continue
                for box in result.boxes:  # type: ignore[attr-defined]
                    conf = float(box.conf)
                    bbox_coords = box.xyxy[0].tolist()

                    # Calculate bounding box dimensions
                    width = bbox_coords[2] - bbox_coords[0]
                    height = bbox_coords[3] - bbox_coords[1]

                    # Aspect ratio filtering
                    if height > 0:
                        aspect_ratio = width / height
                        if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
                            continue

                    if conf > best_conf:
                        best_conf = conf
                        center_x = (bbox_coords[0] + bbox_coords[2]) / 2
                        center_y = (bbox_coords[1] + bbox_coords[3]) / 2
                        best_pos = (center_x, center_y)
                        best_size = (width, height)

                if best_pos:
                    positions.append(
                        BallPosition(
                            frame_idx=frame_idx,
                            x=best_pos[0],
                            y=best_pos[1],
                            confidence=best_conf,
                            is_predicted=False,
                            bbox_width=best_size[0] if best_size else None,
                            bbox_height=best_size[1] if best_size else None,
                        )
                    )

        return positions
