"""
YOLO adapter for ball detection.

Adapted from volleyball_analytics for RallyCut CLI use.
Original: https://github.com/masouduut94/volleyball_analytics
"""

from pathlib import Path
from typing import Optional

import numpy as np

from rallycut.core.models import BallPosition


class BallDetector:
    """
    Specialized YOLO-based ball detection.

    Uses a model trained specifically for volleyball ball detection.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        device: str = "cpu",
        confidence_threshold: float = 0.3,
    ):
        self.model_path = model_path
        self.device = device
        self.confidence_threshold = confidence_threshold
        self._model = None

    def _load_model(self) -> None:
        """Lazy load the YOLO model."""
        if self._model is not None:
            return

        from ultralytics import YOLO

        if self.model_path and self.model_path.exists():
            model_source = str(self.model_path)
        else:
            # Fallback to pretrained model
            model_source = "yolov8n.pt"

        self._model = YOLO(model_source)

    def detect_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
    ) -> Optional[BallPosition]:
        """
        Detect ball in a single frame.

        Args:
            frame: BGR frame from OpenCV
            frame_idx: Frame index

        Returns:
            BallPosition if ball detected, None otherwise
        """
        self._load_model()

        results = self._model.predict(
            frame,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
            classes=[0],  # Assuming ball is class 0
        )

        # Find best ball detection (highest confidence)
        best_detection = None
        best_confidence = 0.0

        for result in results:
            for box in result.boxes:
                conf = float(box.conf)
                if conf > best_confidence:
                    best_confidence = conf
                    bbox_coords = box.xyxy[0].tolist()
                    # Get center point
                    center_x = (bbox_coords[0] + bbox_coords[2]) / 2
                    center_y = (bbox_coords[1] + bbox_coords[3]) / 2
                    best_detection = (center_x, center_y, conf)

        if best_detection:
            return BallPosition(
                frame_idx=frame_idx,
                x=best_detection[0],
                y=best_detection[1],
                confidence=best_detection[2],
                is_predicted=False,
            )

        return None

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
                classes=[0],
            )

            for result, frame_idx in zip(results, batch_indices):
                # Find best detection in this frame
                best_conf = 0.0
                best_pos = None

                for box in result.boxes:
                    conf = float(box.conf)
                    if conf > best_conf:
                        best_conf = conf
                        bbox_coords = box.xyxy[0].tolist()
                        center_x = (bbox_coords[0] + bbox_coords[2]) / 2
                        center_y = (bbox_coords[1] + bbox_coords[3]) / 2
                        best_pos = (center_x, center_y)

                if best_pos:
                    positions.append(
                        BallPosition(
                            frame_idx=frame_idx,
                            x=best_pos[0],
                            y=best_pos[1],
                            confidence=best_conf,
                            is_predicted=False,
                        )
                    )

        return positions
