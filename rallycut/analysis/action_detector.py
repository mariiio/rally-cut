"""Action detection for RallyCut."""

from pathlib import Path
from typing import Callable, Optional

from rallycut.core.config import get_config
from rallycut.core.models import Action, ActionType
from rallycut.core.video import Video


class ActionAnalyzer:
    """Analyzes video to detect volleyball actions (serve, attack, block, etc.)."""

    def __init__(
        self,
        device: Optional[str] = None,
        model_path: Optional[Path] = None,
        confidence_threshold: Optional[float] = None,
    ):
        config = get_config()
        self.device = device or config.device
        self.model_path = model_path or config.action_detector_path
        self.confidence_threshold = confidence_threshold or config.yolo_confidence
        self._detector = None

    def _get_detector(self):
        """Lazy load the detector."""
        if self._detector is None:
            from lib.volleyball_ml.yolo_detector import ActionDetector

            self._detector = ActionDetector(
                model_path=self.model_path,
                device=self.device,
                confidence_threshold=self.confidence_threshold,
            )
        return self._detector

    def analyze_video(
        self,
        video: Video,
        stride: int = 8,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        limit_seconds: Optional[float] = None,
        batch_size: int = 8,
    ) -> list[Action]:
        """
        Analyze video to detect actions.

        Args:
            video: Video to analyze
            stride: Frames to skip between detections
            progress_callback: Callback for progress updates
            limit_seconds: Only analyze first N seconds
            batch_size: Frames to process per batch

        Returns:
            List of detected actions
        """
        detector = self._get_detector()

        total_frames = video.info.frame_count
        fps = video.info.fps

        # Apply limit
        if limit_seconds is not None:
            max_frames = min(total_frames, int(limit_seconds * fps))
        else:
            max_frames = total_frames

        # Collect frame positions
        frame_positions = list(range(0, max_frames, stride))
        total_positions = len(frame_positions)

        if total_positions == 0:
            return []

        all_actions = []

        # Process in batches
        for batch_idx in range(0, total_positions, batch_size):
            batch_positions = frame_positions[batch_idx:batch_idx + batch_size]

            # Read frames
            batch_frames = []
            for frame_idx in batch_positions:
                frame = video.read_frame(frame_idx)
                if frame is not None:
                    batch_frames.append(frame)

            if not batch_frames:
                break

            # Detect actions
            actions = detector.detect_batch(
                batch_frames,
                batch_positions[:len(batch_frames)],
                fps=fps,
            )
            all_actions.extend(actions)

            # Progress
            if progress_callback:
                processed = min(batch_idx + batch_size, total_positions)
                progress = processed / total_positions
                progress_callback(progress, f"Frame {batch_positions[-1]}/{max_frames}")

        return all_actions

    def get_action_summary(self, actions: list[Action]) -> dict[ActionType, int]:
        """
        Summarize actions by type.

        Args:
            actions: List of detected actions

        Returns:
            Dict mapping ActionType to count
        """
        summary = {action_type: 0 for action_type in ActionType if action_type != ActionType.BALL}

        for action in actions:
            if action.action_type in summary:
                summary[action.action_type] += 1

        return summary

    def filter_by_confidence(
        self,
        actions: list[Action],
        min_confidence: float = 0.5,
    ) -> list[Action]:
        """Filter actions by minimum confidence."""
        return [a for a in actions if a.confidence >= min_confidence]

    def group_by_rally(
        self,
        actions: list[Action],
        rally_gap_seconds: float = 5.0,
    ) -> list[list[Action]]:
        """
        Group actions into rallies based on time gaps.

        Args:
            actions: List of actions sorted by timestamp
            rally_gap_seconds: Time gap that separates rallies

        Returns:
            List of rallies (each rally is a list of actions)
        """
        if not actions:
            return []

        # Sort by timestamp
        sorted_actions = sorted(actions, key=lambda a: a.timestamp)

        rallies = []
        current_rally = [sorted_actions[0]]

        for action in sorted_actions[1:]:
            if action.timestamp - current_rally[-1].timestamp > rally_gap_seconds:
                # Start new rally
                rallies.append(current_rally)
                current_rally = [action]
            else:
                current_rally.append(action)

        # Don't forget last rally
        if current_rally:
            rallies.append(current_rally)

        return rallies
