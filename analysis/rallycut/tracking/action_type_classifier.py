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

if TYPE_CHECKING:
    from rallycut.tracking.ball_tracker import BallPosition
    from rallycut.tracking.contact_detector import Contact

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
    is_at_net: float  # 1.0 / 0.0
    court_side: float  # 1.0 = near, 0.0 = far
    arc_fit_residual: float
    confidence: float

    # Sequence context
    contact_index_in_rally: int
    contact_count_on_current_side: int  # 1-3
    is_first_contact_on_side: float  # 1.0 / 0.0

    # Trajectory displacement
    post_contact_dy: float
    post_contact_dx: float
    pre_contact_dy: float

    # Inter-contact
    frames_since_last_contact: int
    distance_from_last_contact: float

    # Serve-specific
    is_in_serve_window: float  # 1.0 / 0.0
    is_at_baseline: float  # 1.0 / 0.0

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
            self.is_at_net,
            self.court_side,
            self.arc_fit_residual,
            self.confidence,
            float(self.contact_index_in_rally),
            float(self.contact_count_on_current_side),
            self.is_first_contact_on_side,
            self.post_contact_dy,
            self.post_contact_dx,
            self.pre_contact_dy,
            float(self.frames_since_last_contact),
            self.distance_from_last_contact,
            self.is_in_serve_window,
            self.is_at_baseline,
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
            "is_at_net",
            "court_side",
            "arc_fit_residual",
            "confidence",
            "contact_index_in_rally",
            "contact_count_on_current_side",
            "is_first_contact_on_side",
            "post_contact_dy",
            "post_contact_dx",
            "pre_contact_dy",
            "frames_since_last_contact",
            "distance_from_last_contact",
            "is_in_serve_window",
            "is_at_baseline",
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
) -> int:
    """Count consecutive same-side contacts ending at index.

    Scans backward from contacts[index], counting contacts on the same
    side. Resets when ball crosses net between consecutive contacts.
    Capped at 3 (beach volleyball max).
    """
    if index == 0:
        return 1

    from rallycut.tracking.action_classifier import _ball_crossed_net

    current = contacts[index]
    count = 1

    for j in range(index - 1, -1, -1):
        prev = contacts[j]
        # Check for net crossing between contacts
        if ball_positions:
            crossed = _ball_crossed_net(
                ball_positions, prev.frame, current.frame, net_y,
            )
            if crossed is True:
                break
        elif prev.court_side != current.court_side:
            break
        count += 1
        current = prev

    return min(count, 3)


def extract_action_features(
    contact: Contact,
    index: int,
    all_contacts: list[Contact],
    ball_positions: list[BallPosition] | None,
    net_y: float,
    rally_start_frame: int = 0,
    serve_window_frames: int = 60,
) -> ActionFeatures:
    """Compute features for a single contact for action classification.

    Args:
        contact: The contact to extract features for.
        index: Index of this contact in all_contacts.
        all_contacts: All validated contacts in the rally (ordered by frame).
        ball_positions: Ball positions for trajectory features.
        net_y: Estimated net Y position.
        rally_start_frame: First frame of the rally.
        serve_window_frames: Serve detection window size.

    Returns:
        ActionFeatures for this contact.
    """
    ball_by_frame = _build_ball_by_frame(ball_positions) if ball_positions else {}

    # Post-contact trajectory (look ahead 15 frames)
    post_dx, post_dy = _compute_displacement(ball_by_frame, contact.frame, 15, "forward")

    # Pre-contact trajectory (look back 10 frames)
    _, pre_dy = _compute_displacement(ball_by_frame, contact.frame, 10, "backward")

    # Contact count on current side
    side_count = _count_contacts_on_side(all_contacts, index, ball_positions, net_y)

    # Is first contact on this side?
    is_first_on_side = side_count == 1

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

    # Serve-specific
    is_in_serve_window = float(
        (contact.frame - rally_start_frame) < serve_window_frames
    )

    # Dynamic baselines (same formula as _find_serve_index)
    baseline_near = net_y + (1.0 - net_y) * 0.64
    baseline_far = net_y * 0.36
    is_at_baseline = float(
        contact.ball_y >= baseline_near or contact.ball_y <= baseline_far
    )

    return ActionFeatures(
        velocity=contact.velocity,
        direction_change_deg=contact.direction_change_deg,
        ball_x=contact.ball_x,
        ball_y=contact.ball_y,
        ball_y_relative_net=contact.ball_y - net_y,
        player_distance=contact.player_distance,
        is_at_net=float(contact.is_at_net),
        court_side=1.0 if contact.court_side == "near" else 0.0,
        arc_fit_residual=contact.arc_fit_residual,
        confidence=contact.confidence,
        contact_index_in_rally=index,
        contact_count_on_current_side=side_count,
        is_first_contact_on_side=float(is_first_on_side),
        post_contact_dy=post_dy,
        post_contact_dx=post_dx,
        pre_contact_dy=pre_dy,
        frames_since_last_contact=frames_since_last,
        distance_from_last_contact=dist_from_last,
        is_in_serve_window=is_in_serve_window,
        is_at_baseline=is_at_baseline,
    )


ACTION_CLASSES = ["serve", "receive", "set", "attack", "dig"]


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
