"""Learned per-contact action classifier for beach volleyball.

Replaces the rule-based state machine (~48% accuracy) with a trained
multiclass GBM that uses trajectory features + sequence context to classify
each validated contact into: serve, receive, set, attack, dig.

Serve and block detection remain heuristic (structurally identifiable).
The classifier handles the remaining dig/set/attack ambiguity where
missed net crossings and camera-angle artifacts cause wrong touch counts.

Model: scikit-learn GradientBoostingClassifier (multiclass, one-vs-rest).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from rallycut.tracking.contact_detector import ball_crossed_net

if TYPE_CHECKING:
    from rallycut.tracking.ball_tracker import BallPosition
    from rallycut.tracking.contact_detector import Contact
    from rallycut.tracking.player_tracker import PlayerPosition

logger = logging.getLogger(__name__)

_CONFIDENCE_THRESHOLD = 0.3


@dataclass
class ActionFeatures:
    """Feature vector for action classification of a single validated contact."""

    # From Contact
    velocity: float
    direction_change_deg: float
    ball_x: float
    ball_y: float
    ball_y_relative_net: float
    player_distance: float
    arc_fit_residual: float
    confidence: float

    # Sequence context
    contact_index_in_rally: int
    contact_count_on_current_side: int  # 1-3

    # Trajectory displacement
    post_contact_dy: float
    post_contact_dx: float
    pre_contact_dy: float

    # Inter-contact
    frames_since_last_contact: int
    distance_from_last_contact: float

    # Bbox motion of attributed player (peak changes in ±5 frames)
    max_d_y: float  # Peak frame-to-frame Y shift (jumps/dives)
    max_d_height: float  # Peak frame-to-frame height change (arm swings/crouching)

    # Team-based transition and player context (v4 features)
    next_contact_team_transition: float  # 1.0=different team, 0.0=same, 0.5=unknown
    player_y_relative_net: float  # Player bbox center Y minus net_y
    post_contact_speed: float  # Mean ball speed over 5 frames post-contact

    # Previous-action context (v3 features, two-pass classification)
    # Set by second pass; defaults = no context (first pass / unknown)
    prev_action_encoded: float = -1.0  # serve=0,..,dig=4, -1=unknown
    prev_action_confidence: float = 0.0

    # Speed profile features (v4)
    ball_vertical_velocity: float = 0.0  # Signed Y velocity at contact (+ = down)
    pre_contact_speed: float = 0.0  # Mean ball speed 5 frames before contact

    def to_array(self) -> np.ndarray:
        """Convert to numpy feature array for classifier input."""
        player_dist = self.player_distance if math.isfinite(self.player_distance) else 1.0
        return np.array([
            self.velocity,
            self.direction_change_deg,
            self.ball_x,
            self.ball_y,
            self.ball_y_relative_net,
            player_dist,
            self.arc_fit_residual,
            self.confidence,
            float(self.contact_index_in_rally),
            float(self.contact_count_on_current_side),
            self.post_contact_dy,
            self.post_contact_dx,
            self.pre_contact_dy,
            float(self.frames_since_last_contact),
            self.distance_from_last_contact,
            self.max_d_y,
            self.max_d_height,
            self.next_contact_team_transition,
            self.player_y_relative_net,
            self.post_contact_speed,
            self.prev_action_encoded,
            self.prev_action_confidence,
            self.ball_vertical_velocity,
            self.pre_contact_speed,
        ], dtype=np.float64)

    @staticmethod
    def feature_names() -> list[str]:
        return [
            "velocity",
            "direction_change_deg",
            "ball_x",
            "ball_y",
            "ball_y_relative_net",
            "player_distance",
            "arc_fit_residual",
            "confidence",
            "contact_index_in_rally",
            "contact_count_on_current_side",
            "post_contact_dy",
            "post_contact_dx",
            "pre_contact_dy",
            "frames_since_last_contact",
            "distance_from_last_contact",
            "max_d_y",
            "max_d_height",
            "next_contact_team_transition",
            "player_y_relative_net",
            "post_contact_speed",
            "prev_action_encoded",
            "prev_action_confidence",
            "ball_vertical_velocity",
            "pre_contact_speed",
        ]


def _build_ball_by_frame(
    ball_positions: list[BallPosition],
) -> dict[int, BallPosition]:
    """Build frame-indexed lookup from ball positions (confidence-filtered)."""
    return {
        bp.frame_number: bp
        for bp in ball_positions
        if bp.confidence >= _CONFIDENCE_THRESHOLD
    }


def _compute_displacement(
    ball_by_frame: dict[int, BallPosition],
    contact_frame: int,
    look_frames: int,
    direction: str = "forward",
) -> tuple[float, float]:
    """Compute ball displacement from contact position.

    Looks ahead (or back) in ball_by_frame and measures displacement
    at ~70% of the look window (to catch the main trajectory, not noise).

    Args:
        ball_by_frame: Pre-built frame-indexed ball position lookup.
        contact_frame: Frame number of the contact.
        look_frames: Number of frames to look ahead/back.
        direction: "forward" or "backward".

    Returns:
        (dx, dy) displacement. (0, 0) if insufficient data.
    """
    contact_bp = ball_by_frame.get(contact_frame)
    if contact_bp is None:
        return 0.0, 0.0

    # Target frame at ~70% of window
    target_offset = int(look_frames * 0.7)
    if direction == "forward":
        target_frame = contact_frame + target_offset
    else:
        target_frame = contact_frame - target_offset

    # Search for closest frame to target within window
    best_bp: BallPosition | None = None
    best_dist = look_frames
    search_range = (
        range(contact_frame + 1, contact_frame + look_frames + 1)
        if direction == "forward"
        else range(contact_frame - 1, contact_frame - look_frames - 1, -1)
    )
    for f in search_range:
        candidate = ball_by_frame.get(f)
        if candidate is not None:
            dist = abs(f - target_frame)
            if dist < best_dist:
                best_dist = dist
                best_bp = candidate

    if best_bp is None:
        return 0.0, 0.0

    return best_bp.x - contact_bp.x, best_bp.y - contact_bp.y


def _count_contacts_on_side(
    contacts: list[Contact],
    index: int,
    ball_positions: list[BallPosition] | None,
    net_y: float,
    team_assignments: dict[int, int] | None = None,
) -> int:
    """Count consecutive same-side contacts ending at index.

    Scans backward from contacts[index], counting contacts on the same
    side. Resets when possession changes. Detection priority:
    1. Team membership (when both contacts have known teams)
    2. Ball crossing net (endpoint displacement)
    3. Court side comparison (fallback)
    Capped at 3 (beach volleyball max).
    """
    if index == 0:
        return 1

    current = contacts[index]
    count = 1

    for j in range(index - 1, -1, -1):
        prev = contacts[j]
        # Priority 1: team membership (most reliable when available)
        if team_assignments:
            cur_team = team_assignments.get(current.player_track_id)
            prev_team = team_assignments.get(prev.player_track_id)
            if cur_team is not None and prev_team is not None:
                if cur_team != prev_team:
                    break
                # Same team — skip net-crossing/court_side checks
                count += 1
                current = prev
                continue
        # Priority 2: net crossing between contacts
        crossed: bool | None = None
        if ball_positions:
            crossed = ball_crossed_net(
                ball_positions, prev.frame, current.frame, net_y,
            )
        if crossed is True:
            break
        # Priority 3: court side comparison (fallback when trajectory
        # is absent or ambiguous)
        if prev.court_side != current.court_side:
            break
        count += 1
        current = prev

    return min(count, 3)


def _get_player_y_relative_net(
    player_positions: list[PlayerPosition] | None,
    track_id: int,
    frame: int,
    net_y: float,
    max_gap: int = 5,
) -> float:
    """Get player's Y position relative to net at a given frame.

    Returns 0.0 if no matching position is found within *max_gap* frames.
    """
    if not player_positions:
        return 0.0
    best_pp = None
    best_gap = max_gap + 1
    for pp in player_positions:
        if pp.track_id == track_id:
            gap = abs(pp.frame_number - frame)
            if gap < best_gap:
                best_gap = gap
                best_pp = pp
    if best_pp is None:
        return 0.0
    return (best_pp.y + best_pp.height / 2) - net_y


def extract_action_features(
    contact: Contact,
    index: int,
    all_contacts: list[Contact],
    ball_positions: list[BallPosition] | None,
    net_y: float,
    rally_start_frame: int = 0,
    team_assignments: dict[int, int] | None = None,
    player_positions: list[PlayerPosition] | None = None,
) -> ActionFeatures:
    """Compute features for a single contact for action classification.

    Args:
        contact: The contact to extract features for.
        index: Index of this contact in all_contacts.
        all_contacts: All validated contacts in the rally (ordered by frame).
        ball_positions: Ball positions for trajectory features.
        net_y: Estimated net Y position.
        rally_start_frame: First frame of the rally.
        team_assignments: Track ID → team index mapping.
        player_positions: Player positions for player-Y feature.

    Returns:
        ActionFeatures for this contact.
    """
    ball_by_frame = _build_ball_by_frame(ball_positions) if ball_positions else {}

    # Post-contact trajectory (look ahead 15 frames)
    post_dx, post_dy = _compute_displacement(ball_by_frame, contact.frame, 15, "forward")

    # Pre-contact trajectory (look back 10 frames)
    _, pre_dy = _compute_displacement(ball_by_frame, contact.frame, 10, "backward")

    # Contact count on current side
    side_count = _count_contacts_on_side(
        all_contacts, index, ball_positions, net_y, team_assignments,
    )

    # Bbox motion of attributed player
    bbox_motion = contact.candidate_bbox_motion.get(contact.player_track_id)
    max_d_y = bbox_motion[0] if bbox_motion else 0.0
    max_d_height = bbox_motion[1] if bbox_motion else 0.0

    # Inter-contact features
    if index > 0:
        prev = all_contacts[index - 1]
        frames_since_last = contact.frame - prev.frame
        dist_from_last = math.sqrt(
            (contact.ball_x - prev.ball_x) ** 2
            + (contact.ball_y - prev.ball_y) ** 2
        )
    else:
        frames_since_last = 0
        dist_from_last = 0.0

    # --- v4 features ---

    # Team-based transition: does the next contact belong to a different team?
    team_transition = 0.5  # unknown
    if team_assignments and index < len(all_contacts) - 1:
        cur_team = team_assignments.get(contact.player_track_id)
        next_contact = all_contacts[index + 1]
        next_team = team_assignments.get(next_contact.player_track_id)
        if cur_team is not None and next_team is not None:
            team_transition = 0.0 if cur_team == next_team else 1.0

    # Player Y relative to net (attacks happen closer to net)
    player_y_rel = _get_player_y_relative_net(
        player_positions, contact.player_track_id, contact.frame, net_y,
    )

    # Post-contact ball speed (attacks are faster than sets)
    post_speed = 0.0
    if ball_by_frame:
        speeds: list[float] = []
        for f in range(contact.frame + 1, contact.frame + 6):
            bp_cur = ball_by_frame.get(f)
            bp_prev = ball_by_frame.get(f - 1)
            if bp_cur is not None and bp_prev is not None:
                dx = bp_cur.x - bp_prev.x
                dy = bp_cur.y - bp_prev.y
                speeds.append(math.sqrt(dx * dx + dy * dy))
        if speeds:
            post_speed = sum(speeds) / len(speeds)

    # Ball vertical velocity at contact (signed, + = downward in image)
    ball_vert_vel = 0.0
    if ball_by_frame:
        bp_at = ball_by_frame.get(contact.frame)
        bp_after = ball_by_frame.get(contact.frame + 3)
        if bp_at is not None and bp_after is not None:
            ball_vert_vel = (bp_after.y - bp_at.y) / 3.0

    # Pre-contact ball speed (mean over 5 frames before contact)
    pre_speed = 0.0
    if ball_by_frame:
        pre_speeds: list[float] = []
        for f in range(contact.frame - 5, contact.frame):
            bp_cur = ball_by_frame.get(f)
            bp_prev = ball_by_frame.get(f - 1)
            if bp_cur is not None and bp_prev is not None:
                dx = bp_cur.x - bp_prev.x
                dy = bp_cur.y - bp_prev.y
                pre_speeds.append(math.sqrt(dx * dx + dy * dy))
        if pre_speeds:
            pre_speed = sum(pre_speeds) / len(pre_speeds)

    return ActionFeatures(
        velocity=contact.velocity,
        direction_change_deg=contact.direction_change_deg,
        ball_x=contact.ball_x,
        ball_y=contact.ball_y,
        ball_y_relative_net=contact.ball_y - net_y,
        player_distance=contact.player_distance,
        arc_fit_residual=contact.arc_fit_residual,
        confidence=contact.confidence,
        contact_index_in_rally=index,
        contact_count_on_current_side=side_count,
        post_contact_dy=post_dy,
        post_contact_dx=post_dx,
        pre_contact_dy=pre_dy,
        frames_since_last_contact=frames_since_last,
        distance_from_last_contact=dist_from_last,
        max_d_y=max_d_y,
        max_d_height=max_d_height,
        next_contact_team_transition=team_transition,
        player_y_relative_net=player_y_rel,
        post_contact_speed=post_speed,
        ball_vertical_velocity=ball_vert_vel,
        pre_contact_speed=pre_speed,
    )


ACTION_CLASSES = ["serve", "receive", "set", "attack", "dig"]

# Encoding for prev_action_encoded feature
_ACTION_ENCODING: dict[str, float] = {
    "serve": 0.0,
    "receive": 1.0,
    "set": 2.0,
    "attack": 3.0,
    "dig": 4.0,
    "block": 5.0,
}


def set_prev_action_context(
    features: ActionFeatures,
    prev_action: str,
    prev_confidence: float,
    same_side: bool | None = None,
) -> None:
    """Set previous-action context on a feature vector (mutates in place)."""
    features.prev_action_encoded = _ACTION_ENCODING.get(prev_action, -1.0)
    features.prev_action_confidence = prev_confidence


# Bump when feature vector changes (forces retrain of stale pickles).
FEATURE_VERSION = 5


class ActionTypeClassifier:
    """Multiclass action classifier for validated contacts.

    Wraps scikit-learn GradientBoostingClassifier. Trained from labeled
    action data (GT matches) and used to replace the rule-based state
    machine for dig/set/attack classification.
    """

    def __init__(self, model: Any = None):
        self.model = model
        self._feature_names = ActionFeatures.feature_names()

    @property
    def is_trained(self) -> bool:
        return self.model is not None

    def predict(self, features: list[ActionFeatures]) -> list[tuple[str, float]]:
        """Predict action type for each contact.

        Args:
            features: List of action feature vectors.

        Returns:
            List of (action_class, confidence) tuples.
        """
        if not self.is_trained or not features:
            return [("unknown", 0.0)] * len(features)

        x_mat = np.array([f.to_array() for f in features])
        probas = self.model.predict_proba(x_mat)
        classes = list(self.model.classes_)

        results: list[tuple[str, float]] = []
        for row in probas:
            best_idx = int(np.argmax(row))
            results.append((classes[best_idx], float(row[best_idx])))
        return results

    def train(
        self,
        x_train: np.ndarray,
        y: np.ndarray,
    ) -> dict[str, Any]:
        """Train the multiclass classifier on labeled data.

        Args:
            x_train: Feature matrix (n_samples, n_features).
            y: String class labels (e.g. "serve", "dig", ...).

        Returns:
            Dictionary of training metrics.
        """
        from sklearn.ensemble import GradientBoostingClassifier

        self.model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            min_samples_leaf=3,
            subsample=0.8,
            random_state=42,
        )
        self.model.fit(x_train, y)

        train_pred = self.model.predict(x_train)
        accuracy = float(np.mean(train_pred == y))
        n_samples = len(y)

        # Per-class accuracy
        per_class: dict[str, dict[str, float]] = {}
        for cls in ACTION_CLASSES:
            mask = y == cls
            if np.sum(mask) > 0:
                cls_acc = float(np.mean(train_pred[mask] == y[mask]))
                per_class[cls] = {
                    "accuracy": cls_acc,
                    "count": int(np.sum(mask)),
                }

        return {
            "train_accuracy": accuracy,
            "n_samples": n_samples,
            "per_class": per_class,
        }

    def loo_cv(
        self,
        x_all: np.ndarray,
        y: np.ndarray,
        rally_ids: np.ndarray,
    ) -> dict[str, Any]:
        """Leave-one-rally-out cross-validation.

        Args:
            x_all: Feature matrix.
            y: String class labels.
            rally_ids: Rally ID per sample (for grouping folds).

        Returns:
            Dictionary with LOO CV metrics.
        """
        from sklearn.ensemble import GradientBoostingClassifier

        unique_rallies = np.unique(rally_ids)
        all_preds = np.empty(len(y), dtype=y.dtype)

        for rally in unique_rallies:
            test_mask = rally_ids == rally
            train_mask = ~test_mask

            if np.sum(train_mask) < 10:
                all_preds[test_mask] = "unknown"
                continue

            model = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.1,
                min_samples_leaf=3,
                subsample=0.8,
                random_state=42,
            )
            model.fit(x_all[train_mask], y[train_mask])
            all_preds[test_mask] = model.predict(x_all[test_mask])

        accuracy = float(np.mean(all_preds == y))

        # Per-class metrics
        per_class: dict[str, dict[str, Any]] = {}
        for cls in ACTION_CLASSES:
            gt_mask = y == cls
            pred_mask = all_preds == cls
            tp = int(np.sum(gt_mask & pred_mask))
            fp = int(np.sum(~gt_mask & pred_mask))
            fn = int(np.sum(gt_mask & ~pred_mask))
            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            f1 = 2 * precision * recall / max(1e-9, precision + recall)
            per_class[cls] = {
                "tp": tp, "fp": fp, "fn": fn,
                "precision": precision, "recall": recall, "f1": f1,
                "count": int(np.sum(gt_mask)),
            }

        # Confusion matrix
        confusion: dict[str, dict[str, int]] = {}
        for cls_gt in ACTION_CLASSES:
            confusion[cls_gt] = {}
            gt_mask = y == cls_gt
            for cls_pred in ACTION_CLASSES:
                pred_mask = all_preds == cls_pred
                confusion[cls_gt][cls_pred] = int(np.sum(gt_mask & pred_mask))

        return {
            "loo_accuracy": accuracy,
            "n_rallies": len(unique_rallies),
            "n_samples": len(y),
            "per_class": per_class,
            "confusion": confusion,
        }

    def feature_importance(self) -> dict[str, float]:
        """Get feature importance from trained model."""
        if not self.is_trained:
            return {}
        importances = self.model.feature_importances_
        return dict(zip(self._feature_names, [float(v) for v in importances]))

    def save(self, path: str | Path) -> None:
        """Save trained model to disk."""
        import pickle

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "model": self.model,
            "feature_names": self._feature_names,
            "action_classes": ACTION_CLASSES,
            "feature_version": FEATURE_VERSION,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"Saved action type classifier to {path}")

    @classmethod
    def load(cls, path: str | Path) -> ActionTypeClassifier:
        """Load trained model from disk."""
        import pickle

        with open(path, "rb") as f:
            data = pickle.load(f)  # noqa: S301

        stored_version = data.get("feature_version", 1)
        if stored_version != FEATURE_VERSION:
            logger.warning(
                "Action classifier feature version mismatch: model has v%d, "
                "code expects v%d. Please retrain.",
                stored_version, FEATURE_VERSION,
            )
            return cls()  # Untrained — forces fallback

        classifier = cls(model=data["model"])
        classifier._feature_names = data.get(
            "feature_names", ActionFeatures.feature_names()
        )
        return classifier


DEFAULT_ACTION_MODEL_PATH = Path("weights/action_classifier/action_classifier.pkl")


def load_action_type_classifier(
    model_path: str | Path | None = None,
) -> ActionTypeClassifier | None:
    """Load action type classifier from default or specified path.

    Returns None if no trained model exists.
    """
    path = Path(model_path) if model_path else DEFAULT_ACTION_MODEL_PATH
    if path.exists():
        try:
            return ActionTypeClassifier.load(path)
        except Exception as e:
            logger.warning(f"Failed to load action type classifier from {path}: {e}")
    return None
