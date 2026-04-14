"""Continuous per-frame player identity classification.

Classifies each tracked player detection as player 1-4 every frame using
DINOv2 ViT-S/14 features compared against user-provided reference crops.
Temporal smoothing (EMA) stabilizes noisy per-frame predictions into reliable
per-track identity assignments.

Architecture: identity-first attribution. Identity is resolved during/after
tracking (not post-hoc), so at contact time the player's identity and team
are already known. This replaces serve-seeded team chains and proximity-based
attribution.

Usage:
    classifier = FrameIdentityClassifier(device="mps")
    classifier.train(crops_by_player)  # {player_id: [BGR crops]}

    # Per-frame classification during tracking
    for frame_num, frame, detections in tracking_loop:
        labels = classifier.classify_frame(frame_num, frame, detections)
        # labels: {track_id: IdentityLabel(player_id, confidence, team)}

    # At contact time: look up identity
    identity = classifier.get_identity(track_id, frame)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from rallycut.tracking.player_features import extract_bbox_crop
from rallycut.tracking.reid_embeddings import PlayerReIDClassifier

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Temporal smoothing: EMA decay for per-track identity probabilities.
# Lower = smoother (more history). 0.15 = ~7 frame effective window.
IDENTITY_EMA_ALPHA = 0.15

# Minimum confidence to consider an identity assignment reliable.
IDENTITY_MIN_CONFIDENCE = 0.40

# Subsample rate: classify every Nth frame to save compute.
# Identity is interpolated for skipped frames.
CLASSIFY_EVERY_N_FRAMES = 3

# Minimum crop size to attempt classification
MIN_CROP_HEIGHT = 32
MIN_CROP_WIDTH = 16


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class IdentityLabel:
    """Identity assignment for a player detection."""

    player_id: int  # Assigned player ID (from reference crops)
    confidence: float  # Confidence in assignment (0-1)
    team: int  # Team (0=near/A, 1=far/B), derived from player-team mapping
    probs: dict[int, float] = field(default_factory=dict)  # Full probability dist


@dataclass
class TrackIdentityState:
    """Per-track EMA state for temporal smoothing."""

    track_id: int
    # Running EMA of class probabilities: {player_id: smoothed_prob}
    smoothed_probs: dict[int, float] = field(default_factory=dict)
    # Number of frames classified for this track
    n_classified: int = 0
    # Last frame classified
    last_frame: int = -1


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------


class FrameIdentityClassifier:
    """Continuous per-frame player identity classifier.

    Wraps PlayerReIDClassifier (DINOv2 + linear head) with:
    - Per-frame crop extraction from player bounding boxes
    - Batched backbone inference (all players in one forward pass)
    - EMA temporal smoothing per track
    - Team derivation from player-team mapping
    """

    def __init__(
        self,
        player_teams: dict[int, int] | None = None,
        device: str | None = None,
        ema_alpha: float = IDENTITY_EMA_ALPHA,
        classify_every_n: int = CLASSIFY_EVERY_N_FRAMES,
    ) -> None:
        """Initialize the classifier.

        Args:
            player_teams: {player_id: team} mapping. If None, all players
                assigned team 0.
            device: Torch device. Auto-detected if None.
            ema_alpha: EMA decay factor for temporal smoothing.
            classify_every_n: Classify every Nth frame (1 = every frame).
        """
        self._reid = PlayerReIDClassifier(device=device)
        self._player_teams = player_teams or {}
        self._ema_alpha = ema_alpha
        self._classify_every_n = max(1, classify_every_n)

        # Per-track EMA state
        self._track_states: dict[int, TrackIdentityState] = {}

        # Cache of identity labels: {(track_id, frame): IdentityLabel}
        self._label_cache: dict[tuple[int, int], IdentityLabel] = {}

    @property
    def is_trained(self) -> bool:
        return self._reid.is_trained

    @property
    def player_ids(self) -> list[int]:
        return self._reid.player_ids

    def train(
        self,
        crops_by_player: dict[int, list[NDArray[np.uint8]]],
        **kwargs: Any,
    ) -> dict[str, float]:
        """Train the underlying ReID classifier on reference crops.

        Args:
            crops_by_player: {player_id: [BGR crop images]}.
            **kwargs: Passed to PlayerReIDClassifier.train().

        Returns:
            Training stats dict.
        """
        stats = self._reid.train(crops_by_player, **kwargs)

        # Infer teams from player IDs if not explicitly provided.
        # Convention: first 2 sorted player IDs = team 0, next 2 = team 1.
        if not self._player_teams and len(self._reid.player_ids) >= 2:
            sorted_ids = sorted(self._reid.player_ids)
            half = len(sorted_ids) // 2
            for i, pid in enumerate(sorted_ids):
                self._player_teams[pid] = 0 if i < half else 1

        return stats

    def set_player_teams(self, player_teams: dict[int, int]) -> None:
        """Set or update the player-to-team mapping."""
        self._player_teams = dict(player_teams)

    def classify_frame(
        self,
        frame_number: int,
        frame: NDArray[np.uint8],
        detections: list[dict[str, Any]],
    ) -> dict[int, IdentityLabel]:
        """Classify all player detections in a single frame.

        Args:
            frame_number: Current frame number.
            frame: BGR video frame (H, W, 3).
            detections: List of player detections, each with keys:
                trackId (int), x, y, width, height (float, normalized 0-1).

        Returns:
            {track_id: IdentityLabel} for each detection with valid crop.
        """
        if not self._reid.is_trained or not detections:
            return {}

        frame_h, frame_w = frame.shape[:2]

        # Decide whether to run classification on this frame
        should_classify = (frame_number % self._classify_every_n) == 0

        results: dict[int, IdentityLabel] = {}

        if should_classify:
            # Extract crops and track IDs for all detections
            crops: list[NDArray[np.uint8]] = []
            crop_track_ids: list[int] = []

            for det in detections:
                tid = det.get("trackId", det.get("track_id", -1))
                if tid < 0:
                    continue

                bbox = (
                    float(det["x"]),
                    float(det["y"]),
                    float(det["width"]),
                    float(det["height"]),
                )

                crop = extract_bbox_crop(
                    frame, bbox, frame_w, frame_h,
                    min_height=MIN_CROP_HEIGHT,
                    min_width=MIN_CROP_WIDTH,
                )
                if crop is not None:
                    crops.append(crop)
                    crop_track_ids.append(tid)

            if crops:
                # Batch predict all crops at once
                probs_list = self._reid.predict(crops)

                for tid, probs in zip(crop_track_ids, probs_list):
                    label = self._update_track_ema(tid, frame_number, probs)
                    results[tid] = label
                    self._label_cache[(tid, frame_number)] = label

        # For tracks not classified this frame, return last known label
        for det in detections:
            tid = det.get("trackId", det.get("track_id", -1))
            if tid < 0 or tid in results:
                continue
            state = self._track_states.get(tid)
            if state is not None and state.n_classified > 0:
                label = self._label_from_smoothed(tid, state.smoothed_probs)
                results[tid] = label

        return results

    def classify_detections_batch(
        self,
        positions_json: list[dict[str, Any]],
        video_path: str,
        rally_start_frame: int = 0,
        rally_end_frame: int | None = None,
    ) -> dict[int, dict[int, IdentityLabel]]:
        """Classify all detections in a rally from stored positions + video.

        This is the batch mode for post-tracking identity classification.
        Reads video frames and classifies player crops in order.

        Args:
            positions_json: List of position dicts (frameNumber, trackId, x, y, w, h).
            video_path: Path to video file.
            rally_start_frame: First frame of the rally.
            rally_end_frame: Last frame (inclusive). If None, uses max from positions.

        Returns:
            {track_id: {frame: IdentityLabel}} nested dict.
        """
        if not self._reid.is_trained or not positions_json:
            return {}

        # Group positions by frame
        positions_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for p in positions_json:
            fn = p.get("frameNumber", p.get("frame_number", -1))
            if fn >= 0:
                positions_by_frame[fn] = positions_by_frame.get(fn, [])
                positions_by_frame[fn].append(p)

        if not positions_by_frame:
            return {}

        # Reset state for fresh batch classification
        self._track_states.clear()
        self._label_cache.clear()

        sorted_frames = sorted(positions_by_frame.keys())
        if rally_end_frame is None:
            rally_end_frame = sorted_frames[-1]

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning("Cannot open video: %s", video_path)
            return {}

        # Seek to rally start
        cap.set(cv2.CAP_PROP_POS_FRAMES, rally_start_frame)

        results: dict[int, dict[int, IdentityLabel]] = defaultdict(dict)
        current_frame = rally_start_frame

        for target_frame in sorted_frames:
            if target_frame < rally_start_frame or target_frame > rally_end_frame:
                continue

            # Advance to target frame
            while current_frame < target_frame:
                cap.grab()
                current_frame += 1

            ret, frame = cap.read()
            if not ret or frame is None:
                break
            current_frame += 1

            # Classify this frame's detections
            dets = positions_by_frame[target_frame]
            frame_arr = np.asarray(frame, dtype=np.uint8)
            labels = self.classify_frame(target_frame, frame_arr, dets)

            for tid, label in labels.items():
                results[tid][target_frame] = label

        cap.release()

        # Log summary
        for tid in sorted(results):
            frames = results[tid]
            if frames:
                last_label = max(frames.items(), key=lambda x: x[0])[1]
                logger.debug(
                    "Track %d: %d frames classified, final identity=%d (%.1f%%)",
                    tid, len(frames), last_label.player_id,
                    last_label.confidence * 100,
                )

        return dict(results)

    def get_identity(
        self,
        track_id: int,
        frame: int | None = None,
    ) -> IdentityLabel | None:
        """Get the identity label for a track at a specific frame.

        If frame is None, returns the latest smoothed identity.

        Args:
            track_id: Player track ID.
            frame: Frame number (optional).

        Returns:
            IdentityLabel or None if track has no identity data.
        """
        if frame is not None:
            cached = self._label_cache.get((track_id, frame))
            if cached is not None:
                return cached

        state = self._track_states.get(track_id)
        if state is None or state.n_classified == 0:
            return None

        return self._label_from_smoothed(track_id, state.smoothed_probs)

    def get_track_identity(self, track_id: int) -> IdentityLabel | None:
        """Get the final (most recent) identity for a track.

        Uses the full EMA history — best for post-tracking attribution.
        """
        return self.get_identity(track_id)

    def get_all_track_identities(self) -> dict[int, IdentityLabel]:
        """Get final identity labels for all classified tracks."""
        result: dict[int, IdentityLabel] = {}
        for tid, state in self._track_states.items():
            if state.n_classified > 0:
                result[tid] = self._label_from_smoothed(tid, state.smoothed_probs)
        return result

    def to_identity_map(self) -> dict[int, int]:
        """Return {track_id: player_id} for all tracks above min confidence.

        Useful for integration with existing attribution code.
        """
        result: dict[int, int] = {}
        for tid, label in self.get_all_track_identities().items():
            if label.confidence >= IDENTITY_MIN_CONFIDENCE:
                result[tid] = label.player_id
        return result

    def to_team_map(self) -> dict[int, int]:
        """Return {track_id: team} for all tracks above min confidence.

        Useful for integration with existing attribution code that expects
        team_assignments: {track_id: team (0=near, 1=far)}.
        """
        result: dict[int, int] = {}
        for tid, label in self.get_all_track_identities().items():
            if label.confidence >= IDENTITY_MIN_CONFIDENCE:
                result[tid] = label.team
        return result

    # -----------------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------------

    def _update_track_ema(
        self,
        track_id: int,
        frame_number: int,
        raw_probs: dict[int, float],
    ) -> IdentityLabel:
        """Update EMA probabilities for a track and return the label."""
        state = self._track_states.get(track_id)
        if state is None:
            state = TrackIdentityState(track_id=track_id)
            self._track_states[track_id] = state

        if state.n_classified == 0:
            # First observation: initialize with raw probs
            state.smoothed_probs = dict(raw_probs)
        else:
            # EMA update: smoothed = (1-alpha) * old + alpha * new
            alpha = self._ema_alpha
            for pid in raw_probs:
                old = state.smoothed_probs.get(pid, 0.0)
                state.smoothed_probs[pid] = (1 - alpha) * old + alpha * raw_probs[pid]

        state.n_classified += 1
        state.last_frame = frame_number

        return self._label_from_smoothed(track_id, state.smoothed_probs)

    def _label_from_smoothed(
        self,
        track_id: int,
        smoothed_probs: dict[int, float],
    ) -> IdentityLabel:
        """Create an IdentityLabel from smoothed probabilities."""
        if not smoothed_probs:
            return IdentityLabel(player_id=-1, confidence=0.0, team=-1)

        # Normalize smoothed probs (EMA can drift from sum=1)
        total = sum(smoothed_probs.values())
        if total > 0:
            normed = {pid: p / total for pid, p in smoothed_probs.items()}
        else:
            normed = smoothed_probs

        best_pid = max(normed, key=lambda pid: normed[pid])
        best_conf = normed[best_pid]

        # Margin: difference between best and second-best
        sorted_probs = sorted(normed.values(), reverse=True)
        margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 1.0

        # Confidence combines probability and margin
        confidence = min(best_conf, best_conf * (0.5 + 0.5 * margin))

        team = self._player_teams.get(best_pid, 0)

        return IdentityLabel(
            player_id=best_pid,
            confidence=confidence,
            team=team,
            probs=normed,
        )
