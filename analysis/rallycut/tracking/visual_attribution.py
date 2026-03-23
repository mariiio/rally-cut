"""Per-player visual action attribution for beach volleyball.

Replaces proximity-based player attribution (~68% accuracy) with a visual
classifier that determines which player is performing an action from their
body movement in a short video clip around the contact frame.

Architecture:
- Input: 16-frame clip of a player's bounding box crop (224x224)
- Backbone: Frozen VideoMAE encoder → 768-dim CLS token
- Head: LogisticRegression (binary: acting vs not-acting)

At each detected contact, extract clips for all same-side candidates,
classify each, and attribute to the highest-scoring player.

Requires: VideoMAE weights at weights/videomae/game_state_classifier/
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

if TYPE_CHECKING:
    from rallycut.tracking.action_classifier import ClassifiedAction
    from rallycut.tracking.contact_detector import Contact

logger = logging.getLogger(__name__)

# Temporal window: 16 frames biased pre-contact to capture approach motion
WINDOW_BEFORE = 14
WINDOW_AFTER = 1
FRAME_WINDOW = WINDOW_BEFORE + WINDOW_AFTER + 1  # 16
CROP_PAD = 0.2  # 20% padding around bounding box
CROP_SIZE = 224  # VideoMAE input resolution
MIN_CROP_PX = 16  # Minimum crop dimension in pixels

DEFAULT_MODEL_PATH = Path(__file__).parent.parent.parent / (
    "weights/visual_attribution/visual_attribution.pkl"
)


@dataclass
class PlayerClip:
    """A 16-frame video clip of a single player's bounding box."""

    track_id: int
    frames: list[np.ndarray]  # 16 BGR frames, each 224x224
    geom_features: list[float]  # [vert_disp, ar_change, h_change, horiz_disp]

    @property
    def is_valid(self) -> bool:
        return len(self.frames) == FRAME_WINDOW


def extract_player_clip(
    video_cap: cv2.VideoCapture,
    positions_by_frame: dict[int, dict[str, float]],
    contact_frame: int,
    rally_start_frame: int,
    frame_w: int,
    frame_h: int,
) -> list[np.ndarray] | None:
    """Extract a 16-frame crop clip for a player around a contact frame.

    Args:
        video_cap: Open cv2.VideoCapture.
        positions_by_frame: {frame_number: {x, y, width, height}} for this track.
        contact_frame: Rally-relative contact frame number.
        rally_start_frame: Absolute frame number of rally start.
        frame_w: Video frame width in pixels.
        frame_h: Video frame height in pixels.

    Returns:
        List of 16 BGR frames (224x224) or None if extraction fails.
    """
    frames: list[np.ndarray] = []

    for offset in range(-WINDOW_BEFORE, WINDOW_AFTER + 1):
        rel_frame = contact_frame + offset

        # Find player position with ±1 frame tolerance
        pos = None
        for f_search in [rel_frame, rel_frame - 1, rel_frame + 1]:
            pos = positions_by_frame.get(f_search)
            if pos is not None:
                break

        if pos is None:
            # Interpolate from nearest known position
            for delta in range(2, 10):
                pos = positions_by_frame.get(rel_frame - delta)
                if pos is not None:
                    break
                pos = positions_by_frame.get(rel_frame + delta)
                if pos is not None:
                    break

        if pos is None:
            return None  # Can't locate player

        # Read video frame
        abs_frame = rally_start_frame + rel_frame
        video_cap.set(cv2.CAP_PROP_POS_FRAMES, abs_frame)
        ret, frame = video_cap.read()
        if not ret or frame is None:
            return None

        # Crop with padding
        cx = pos["x"] * frame_w
        cy = pos["y"] * frame_h
        bw = pos["width"] * frame_w
        bh = pos["height"] * frame_h
        pad_w = bw * CROP_PAD
        pad_h = bh * CROP_PAD

        x1 = max(0, int(cx - bw / 2 - pad_w))
        y1 = max(0, int(cy - bh / 2 - pad_h))
        x2 = min(frame_w, int(cx + bw / 2 + pad_w))
        y2 = min(frame_h, int(cy + bh / 2 + pad_h))

        if x2 - x1 < MIN_CROP_PX or y2 - y1 < MIN_CROP_PX:
            return None

        crop = frame[y1:y2, x1:x2]
        crop_resized = cv2.resize(crop, (CROP_SIZE, CROP_SIZE))
        frames.append(crop_resized)

    return frames if len(frames) == FRAME_WINDOW else None


def compute_geom_features(
    positions_by_frame: dict[int, dict[str, float]],
    contact_frame: int,
    lookback: int = 10,
) -> list[float]:
    """Compute geometric features for a player at a contact frame.

    Returns [vertical_disp, aspect_ratio_change, height_change, horiz_disp].
    """
    pos_now = positions_by_frame.get(contact_frame)
    if pos_now is None:
        # Search nearby frames
        for delta in range(1, 3):
            pos_now = positions_by_frame.get(contact_frame + delta)
            if pos_now is None:
                pos_now = positions_by_frame.get(contact_frame - delta)
            if pos_now is not None:
                break

    ref_frame = contact_frame - lookback
    pos_ref = None
    for f in range(ref_frame - 2, ref_frame + 3):
        pos_ref = positions_by_frame.get(f)
        if pos_ref is not None:
            break

    if pos_now is None or pos_ref is None:
        return [0.0, 1.0, 1.0, 0.0]

    vert_disp = pos_now["y"] - pos_ref["y"]
    ar_now = pos_now["height"] / max(pos_now["width"], 1e-6)
    ar_ref = pos_ref["height"] / max(pos_ref["width"], 1e-6)
    ar_change = ar_now / max(ar_ref, 1e-6)
    h_change = pos_now["height"] / max(pos_ref["height"], 1e-6)
    horiz_disp = abs(pos_now["x"] - pos_ref["x"])

    return [vert_disp, ar_change, h_change, horiz_disp]


def build_positions_by_frame(
    positions_json: list[dict[str, Any]],
    track_id: int,
) -> dict[int, dict[str, float]]:
    """Build frame→position dict for a single track."""
    result: dict[int, dict[str, float]] = {}
    for p in positions_json:
        if p.get("trackId") != track_id:
            continue
        fn = p.get("frameNumber")
        if fn is None:
            continue
        result[fn] = {
            "x": p.get("x", 0.0),
            "y": p.get("y", 0.0),
            "width": p.get("width", 0.0),
            "height": p.get("height", 0.0),
        }
    return result


class VisualAttributionClassifier:
    """Binary classifier: is this player performing the action?

    Uses frozen VideoMAE encoder features (768-dim CLS token) plus
    4 geometric bbox features, fed through a logistic regression head.
    """

    def __init__(self) -> None:
        self._model: Any = None  # sklearn LogisticRegression
        self._scaler: Any = None  # sklearn StandardScaler
        self._videomae: Any = None  # GameStateClassifier (lazy-loaded)
        self._use_geom: bool = True

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    def _get_videomae(self) -> Any:
        """Lazy-load VideoMAE model."""
        if self._videomae is None:
            from lib.volleyball_ml.video_mae import GameStateClassifier
            self._videomae = GameStateClassifier()
        return self._videomae

    def extract_features(
        self,
        clips: list[PlayerClip],
    ) -> np.ndarray:
        """Extract features from player clips.

        Args:
            clips: List of PlayerClip objects with valid 16-frame clips.

        Returns:
            Feature matrix of shape (n_clips, 768 + 4) or (n_clips, 768).
        """
        if not clips:
            return np.array([])

        videomae = self._get_videomae()

        # Extract VideoMAE features in batch
        batch_frames = [clip.frames for clip in clips]
        visual_features = videomae.get_encoder_features_batch(
            batch_frames, pooling="cls",
        )  # (n_clips, 768)

        if self._use_geom:
            geom_features = np.array([clip.geom_features for clip in clips])
            result: np.ndarray = np.concatenate([visual_features, geom_features], axis=1)
            return result

        return np.asarray(visual_features)

    def train(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        use_geom: bool = True,
    ) -> dict[str, Any]:
        """Train the binary classifier.

        Args:
            features: Feature matrix (n_samples, n_features).
            labels: Binary labels (1=acting, 0=not-acting).
            use_geom: Whether geometric features are included.

        Returns:
            Training metrics dict.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        self._use_geom = use_geom
        self._scaler = StandardScaler()
        x_scaled = self._scaler.fit_transform(features)

        self._model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42,
        )
        self._model.fit(x_scaled, labels)

        # Training metrics
        train_probs = self._model.predict_proba(x_scaled)[:, 1]
        train_preds = (train_probs >= 0.5).astype(int)
        accuracy = float(np.mean(train_preds == labels))

        return {
            "n_samples": len(labels),
            "n_positive": int(labels.sum()),
            "n_negative": int((1 - labels).sum()),
            "train_accuracy": accuracy,
        }

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Predict P(acting) for each sample.

        Args:
            features: Feature matrix (n_samples, n_features).

        Returns:
            Array of probabilities, shape (n_samples,).
        """
        if not self.is_trained:
            return np.full(len(features), 0.5)

        x_scaled = self._scaler.transform(features)
        proba: np.ndarray = self._model.predict_proba(x_scaled)[:, 1]
        return proba

    def save(self, path: str | Path | None = None) -> None:
        """Save trained model to disk."""
        path = Path(path) if path else DEFAULT_MODEL_PATH
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "model": self._model,
            "scaler": self._scaler,
            "use_geom": self._use_geom,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

        logger.info("Saved visual attribution classifier to %s", path)

    @classmethod
    def load(cls, path: str | Path | None = None) -> VisualAttributionClassifier:
        """Load trained model from disk."""
        path = Path(path) if path else DEFAULT_MODEL_PATH
        with open(path, "rb") as f:
            data = pickle.load(f)  # noqa: S301

        instance = cls()
        instance._model = data["model"]
        instance._scaler = data["scaler"]
        instance._use_geom = data.get("use_geom", True)
        return instance


def load_visual_attribution_classifier(
    model_path: str | Path | None = None,
) -> VisualAttributionClassifier | None:
    """Load visual attribution classifier from default or specified path.

    Returns None if no trained model exists.
    """
    path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    if path.exists():
        try:
            return VisualAttributionClassifier.load(path)
        except Exception as e:
            logger.warning(
                "Failed to load visual attribution classifier from %s: %s", path, e,
            )
    return None


def get_same_side_track_ids(
    positions_json: list[dict[str, Any]],
    target_track_id: int,
    contact_frame: int,
    team_assignments: dict[int, int] | None,
    court_split_y: float | None,
) -> list[int]:
    """Get track IDs of same-side players at a contact frame.

    Uses team assignments if available, else falls back to court_split_y.
    """
    # Get all tracks visible at contact frame (±2 frames)
    frame_tracks: dict[int, dict[str, float]] = {}
    for p in positions_json:
        fn = p.get("frameNumber")
        tid = p.get("trackId")
        if fn is None or tid is None:
            continue
        if abs(fn - contact_frame) <= 2:
            if tid not in frame_tracks:
                frame_tracks[tid] = {
                    "x": p.get("x", 0.0),
                    "y": p.get("y", 0.0),
                }

    if target_track_id not in frame_tracks:
        return []

    target_team = None
    if team_assignments and target_track_id in team_assignments:
        target_team = team_assignments[target_track_id]

    same_side: list[int] = []
    for tid in frame_tracks:
        if tid == target_track_id:
            continue

        if target_team is not None and team_assignments:
            if team_assignments.get(tid) == target_team:
                same_side.append(tid)
        elif court_split_y is not None:
            target_y = frame_tracks[target_track_id]["y"]
            cand_y = frame_tracks[tid]["y"]
            if (target_y < court_split_y) == (cand_y < court_split_y):
                same_side.append(tid)

    return same_side


def visual_reattribute(
    actions: list[ClassifiedAction],
    contacts: list[Contact],
    positions_json: list[dict[str, Any]],
    video_cap: cv2.VideoCapture,
    rally_start_frame: int,
    frame_w: int,
    frame_h: int,
    classifier: VisualAttributionClassifier,
    team_assignments: dict[int, int] | None = None,
    court_split_y: float | None = None,
    min_confidence: float = 0.6,
    min_margin: float = 0.15,
) -> int:
    """Re-attribute actions using visual per-player classification.

    For each action/contact pair with same-side candidates, extracts 16-frame
    clips, runs the visual classifier, and re-attributes to the highest-scoring
    player if confident.

    Args:
        actions: ClassifiedAction list (modified in place).
        contacts: Contact list.
        positions_json: Raw player positions.
        video_cap: Open cv2.VideoCapture for the video.
        rally_start_frame: Absolute frame of rally start.
        frame_w: Video frame width.
        frame_h: Video frame height.
        classifier: Trained VisualAttributionClassifier.
        team_assignments: Optional track_id → team mapping.
        court_split_y: Optional net Y for side detection.
        min_confidence: Minimum P(acting) to re-attribute.
        min_margin: Minimum margin between top-2 candidates.

    Returns:
        Number of actions re-attributed.
    """
    if not classifier.is_trained:
        return 0

    contact_by_frame: dict[int, Contact] = {c.frame: c for c in contacts}

    # Pre-build position indices per track (avoids re-scanning positions_json
    # for each candidate in each contact)
    all_track_positions: dict[int, dict[int, dict[str, float]]] = {}
    for p in positions_json:
        tid = p.get("trackId")
        fn = p.get("frameNumber")
        if tid is None or fn is None:
            continue
        if tid not in all_track_positions:
            all_track_positions[tid] = {}
        all_track_positions[tid][fn] = {
            "x": p.get("x", 0.0),
            "y": p.get("y", 0.0),
            "width": p.get("width", 0.0),
            "height": p.get("height", 0.0),
        }

    n_changed = 0

    for action in actions:
        contact = contact_by_frame.get(action.frame)
        if contact is None or not contact.player_candidates:
            continue

        current_tid = action.player_track_id
        if current_tid < 0:
            continue

        same_side = get_same_side_track_ids(
            positions_json, current_tid, action.frame,
            team_assignments, court_split_y,
        )
        if not same_side:
            continue

        candidate_tids = [current_tid] + same_side

        clips: list[PlayerClip] = []
        clip_tids: list[int] = []

        for tid in candidate_tids:
            pos_by_frame = all_track_positions.get(tid, {})
            if not pos_by_frame:
                continue

            frames = extract_player_clip(
                video_cap, pos_by_frame, action.frame,
                rally_start_frame, frame_w, frame_h,
            )
            if frames is None:
                continue

            geom = compute_geom_features(pos_by_frame, action.frame)
            clips.append(PlayerClip(
                track_id=tid, frames=frames, geom_features=geom,
            ))
            clip_tids.append(tid)

        if len(clips) < 2:
            # Only one candidate extractable — keep current attribution
            continue

        features = classifier.extract_features(clips)
        probs = classifier.predict_proba(features)

        sorted_indices = np.argsort(probs)[::-1]
        best_idx = sorted_indices[0]
        second_idx = sorted_indices[1]
        best_prob = float(probs[best_idx])
        margin = float(probs[best_idx] - probs[second_idx])
        best_tid = clip_tids[best_idx]

        if best_prob >= min_confidence and margin >= min_margin:
            if best_tid != current_tid:
                action.player_track_id = best_tid
                n_changed += 1

    return n_changed
