"""Learned server identification classifier for beach volleyball rallies.

The current heuristic server detector (_identify_server in action_classifier.py)
caps out at ~78% serve side accuracy. Oracle analysis proved that correctly
identifying the server gives 100% correct serve side, so the entire problem is
server identification.

This module trains a binary GBM classifier on per-player features extracted
from the first ~60 frames of a rally. For each rally, every tracked player
gets a P(server) score; the highest-scoring player is selected as the server,
and their court side determines the serve side.

Features per player (~24 total):
- Position (image-space): foot Y, distance from net, teammate separation,
  side, X position, bbox height + relative bbox height
- Position (court-space): court Y, distance from baseline, teammate separation
  in meters (when court calibration available)
- Motion: total displacement, position variance, max velocity,
  first-frame presence (captures late-arriving servers)
- Pose (YOLO-Pose keypoints): max wrist elevation, peak arm asymmetry,
  single-arm-raise fraction, max wrist velocity (toss + swing)
- Ball: distance to first ball detection in window
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from rallycut.court.calibration import CourtCalibrator
    from rallycut.tracking.ball_tracker import BallPosition
    from rallycut.tracking.player_tracker import PlayerPosition

logger = logging.getLogger(__name__)

# COCO keypoint indices used for pose features
_KPT_L_SHOULDER = 5
_KPT_R_SHOULDER = 6
_KPT_L_WRIST = 9
_KPT_R_WRIST = 10
_KPT_MIN_CONF = 0.3

# Court constants (meters)
_COURT_NET_Y = 8.0
_COURT_LENGTH = 16.0
_COURT_Y_MIN = -3.0
_COURT_Y_MAX = 19.0

FEATURE_VERSION = 1
DEFAULT_WINDOW_FRAMES = 60


@dataclass
class ServerFeatures:
    """Per-player feature vector for server identification."""

    track_id: int
    n_frames: int  # Number of frames the player was tracked in window

    # --- Position (image-space) ---
    foot_y_mean: float           # Mean foot Y (bbox bottom)
    foot_y_std: float            # Std of foot Y
    dist_from_net: float         # |foot_y_mean - net_y|
    teammate_separation: float   # foot_y distance from same-side teammate
    is_near_side: float          # 1.0 if foot_y_mean > net_y else 0.0
    mean_x: float                # Mean bbox center X
    bbox_height_mean: float      # Mean bbox height (smaller = farther from camera)
    bbox_height_relative: float  # Ratio to same-side teammate's mean height (1.0 if solo)

    # --- Position (court-space) ---
    court_y: float               # Mean court Y in meters
    court_dist_from_baseline: float  # Distance from nearest baseline (meters)
    court_dist_from_net: float   # |court_y - 8| in meters
    court_teammate_sep: float    # Court-space teammate separation (meters)
    has_court_space: float       # 1.0 if court projection valid, else 0.0

    # --- Motion ---
    total_displacement: float    # Sum of frame-to-frame movement (image-space)
    position_variance: float     # foot_y_std^2 + foot_x_std^2
    max_velocity: float          # Max frame-to-frame displacement
    first_frame_presence: float  # 1.0 if tracked from window start, else first_offset/window

    # --- Pose ---
    max_wrist_elevation: float   # Peak wrist elevation above shoulder / bbox_h
    peak_arm_asymmetry: float    # Peak |left_elev - right_elev|
    single_arm_raise_frac: float # Fraction of frames with exactly one arm raised
    max_wrist_velocity: float    # Peak wrist displacement / bbox_h

    # --- Ball proximity ---
    dist_to_first_ball: float    # Distance to first ball detection (-1 if no ball)

    def to_array(self) -> np.ndarray:
        return np.array([
            self.foot_y_mean,
            self.foot_y_std,
            self.dist_from_net,
            self.teammate_separation,
            self.is_near_side,
            self.mean_x,
            self.bbox_height_mean,
            self.bbox_height_relative,
            self.court_y,
            self.court_dist_from_baseline,
            self.court_dist_from_net,
            self.court_teammate_sep,
            self.has_court_space,
            self.total_displacement,
            self.position_variance,
            self.max_velocity,
            self.first_frame_presence,
            self.max_wrist_elevation,
            self.peak_arm_asymmetry,
            self.single_arm_raise_frac,
            self.max_wrist_velocity,
            self.dist_to_first_ball,
        ], dtype=np.float64)

    @staticmethod
    def feature_names() -> list[str]:
        return [
            "foot_y_mean",
            "foot_y_std",
            "dist_from_net",
            "teammate_separation",
            "is_near_side",
            "mean_x",
            "bbox_height_mean",
            "bbox_height_relative",
            "court_y",
            "court_dist_from_baseline",
            "court_dist_from_net",
            "court_teammate_sep",
            "has_court_space",
            "total_displacement",
            "position_variance",
            "max_velocity",
            "first_frame_presence",
            "max_wrist_elevation",
            "peak_arm_asymmetry",
            "single_arm_raise_frac",
            "max_wrist_velocity",
            "dist_to_first_ball",
        ]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def _extract_pose_features(
    frames_data: list[tuple[int, list[list[float]], float]],
) -> tuple[float, float, float, float]:
    """Compute pose features (elev, asym, single_frac, wrist_vel) for one player.

    frames_data: list of (frame_number, keypoints, bbox_height).
    Mirrors the logic of _score_server_candidates_by_pose() in action_classifier.py.
    """
    if len(frames_data) < 3:
        return 0.0, 0.0, 0.0, 0.0

    frames_data = sorted(frames_data, key=lambda x: x[0])

    max_elev = 0.0
    max_asym = 0.0
    max_wrist_vel = 0.0
    single_arm_frames = 0
    valid_frames = 0

    prev_l: tuple[float, float] | None = None
    prev_r: tuple[float, float] | None = None

    for _frame, kpts, bbox_h in frames_data:
        if not kpts or len(kpts) < 11 or bbox_h <= 1e-6:
            prev_l = None
            prev_r = None
            continue

        ls = kpts[_KPT_L_SHOULDER]
        rs = kpts[_KPT_R_SHOULDER]
        lw = kpts[_KPT_L_WRIST]
        rw = kpts[_KPT_R_WRIST]

        if ls[2] < _KPT_MIN_CONF or rs[2] < _KPT_MIN_CONF:
            prev_l = None
            prev_r = None
            continue

        l_ok = lw[2] >= _KPT_MIN_CONF
        r_ok = rw[2] >= _KPT_MIN_CONF
        if not (l_ok or r_ok):
            prev_l = None
            prev_r = None
            continue

        valid_frames += 1
        shoulder_y = (ls[1] + rs[1]) / 2.0

        l_elev = (shoulder_y - lw[1]) / bbox_h if l_ok else 0.0
        r_elev = (shoulder_y - rw[1]) / bbox_h if r_ok else 0.0
        frame_elev = max(l_elev, r_elev)
        if frame_elev > max_elev:
            max_elev = frame_elev

        if l_ok and r_ok:
            asym = abs(l_elev - r_elev)
            if asym > max_asym:
                max_asym = asym
            l_above = l_elev > 0.1
            r_above = r_elev > 0.1
            if l_above != r_above:
                single_arm_frames += 1

        if l_ok and prev_l is not None:
            dx = lw[0] - prev_l[0]
            dy = lw[1] - prev_l[1]
            vel = math.sqrt(dx * dx + dy * dy) / bbox_h
            if vel > max_wrist_vel:
                max_wrist_vel = vel
        if r_ok and prev_r is not None:
            dx = rw[0] - prev_r[0]
            dy = rw[1] - prev_r[1]
            vel = math.sqrt(dx * dx + dy * dy) / bbox_h
            if vel > max_wrist_vel:
                max_wrist_vel = vel

        prev_l = (lw[0], lw[1]) if l_ok else None
        prev_r = (rw[0], rw[1]) if r_ok else None

    if valid_frames < 3:
        return 0.0, 0.0, 0.0, 0.0

    return max_elev, max_asym, max_wrist_vel, single_arm_frames / valid_frames


def extract_server_features(
    player_positions: list[PlayerPosition],
    ball_positions: list[BallPosition] | None,
    net_y: float,
    start_frame: int,
    window_frames: int = DEFAULT_WINDOW_FRAMES,
    calibrator: CourtCalibrator | None = None,
) -> dict[int, ServerFeatures]:
    """Extract per-player server features from the first window_frames of a rally.

    Returns {track_id: ServerFeatures} for every player tracked at least 3
    frames within [start_frame, start_frame + window_frames).
    """
    end_frame = start_frame + window_frames

    # Group positions by track_id
    by_track: dict[int, list[Any]] = defaultdict(list)
    for p in player_positions:
        if start_frame <= p.frame_number < end_frame:
            by_track[p.track_id].append(p)

    if not by_track:
        return {}

    # Compute basic per-track stats
    track_stats: dict[int, dict[str, float]] = {}
    track_pose_inputs: dict[int, list[tuple[int, list[list[float]], float]]] = {}

    for tid, positions in by_track.items():
        if len(positions) < 3:
            continue
        positions.sort(key=lambda p: p.frame_number)
        foot_ys = [p.y + p.height / 2.0 for p in positions]
        xs = [p.x for p in positions]
        heights = [p.height for p in positions]

        foot_y_mean = float(np.mean(foot_ys))
        foot_y_std = float(np.std(foot_ys))
        x_std = float(np.std(xs))
        mean_x = float(np.mean(xs))
        bbox_h_mean = float(np.mean(heights))

        # Motion features
        total_disp = 0.0
        max_vel = 0.0
        for i in range(1, len(positions)):
            dx = positions[i].x - positions[i - 1].x
            dy = positions[i].y - positions[i - 1].y
            d = math.sqrt(dx * dx + dy * dy)
            total_disp += d
            if d > max_vel:
                max_vel = d

        first_offset = positions[0].frame_number - start_frame
        first_presence = 1.0 - min(1.0, first_offset / max(1, window_frames))

        track_stats[tid] = {
            "foot_y_mean": foot_y_mean,
            "foot_y_std": foot_y_std,
            "x_std": x_std,
            "mean_x": mean_x,
            "bbox_h_mean": bbox_h_mean,
            "total_disp": total_disp,
            "max_vel": max_vel,
            "first_presence": first_presence,
            "n_frames": float(len(positions)),
        }

        # Pose inputs (only frames with keypoints)
        pose_frames: list[tuple[int, list[list[float]], float]] = []
        for p in positions:
            if p.keypoints is not None and p.height > 0.01:
                pose_frames.append((p.frame_number, p.keypoints, p.height))
        track_pose_inputs[tid] = pose_frames

    if not track_stats:
        return {}

    # Side assignment (image space)
    near_tids: list[int] = []
    far_tids: list[int] = []
    for tid, st in track_stats.items():
        if st["foot_y_mean"] > net_y:
            near_tids.append(tid)
        else:
            far_tids.append(tid)

    # Court-space projection (when calibrator available)
    track_court_y: dict[int, float] = {}
    court_ok = calibrator is not None and calibrator.is_calibrated
    if court_ok:
        for tid, st in track_stats.items():
            try:
                _cx, cy = calibrator.image_to_court(  # type: ignore[union-attr]
                    (st["mean_x"], st["foot_y_mean"]), 1, 1,
                )
            except Exception:
                court_ok = False
                track_court_y = {}
                break
            if cy < _COURT_Y_MIN or cy > _COURT_Y_MAX:
                court_ok = False
                track_court_y = {}
                break
            track_court_y[tid] = float(cy)

    # First ball position (for ball proximity feature)
    first_ball_xy: tuple[float, float] | None = None
    if ball_positions:
        for b in ball_positions:
            if start_frame <= b.frame_number < end_frame and b.confidence >= 0.3:
                first_ball_xy = (b.x, b.y)
                break

    # Build features per track
    result: dict[int, ServerFeatures] = {}
    for tid, st in track_stats.items():
        foot_y_mean = st["foot_y_mean"]
        is_near = foot_y_mean > net_y
        same_side = near_tids if is_near else far_tids

        # Teammate separation (image-space): distance from net to self minus
        # distance from net to closest same-side teammate
        self_dist_from_net = abs(foot_y_mean - net_y)
        teammate_sep = self_dist_from_net  # default for solo
        bbox_h_relative = 1.0
        if len(same_side) > 1:
            best_other = -1.0
            for other in same_side:
                if other == tid:
                    continue
                other_dist = abs(track_stats[other]["foot_y_mean"] - net_y)
                if other_dist > best_other:
                    best_other = other_dist
                    bbox_h_relative = (
                        st["bbox_h_mean"] / max(1e-6, track_stats[other]["bbox_h_mean"])
                    )
            # Positive = this player is further from the net than its
            # teammate (i.e. closer to the baseline = stronger server signal).
            teammate_sep = self_dist_from_net - best_other

        # Court-space features
        court_y_val = 0.0
        court_dist_baseline = 0.0
        court_dist_net = 0.0
        court_teammate_sep_val = 0.0
        has_court = 0.0
        if court_ok and tid in track_court_y:
            cy = track_court_y[tid]
            court_y_val = cy
            court_dist_baseline = cy if cy < _COURT_NET_Y else _COURT_LENGTH - cy
            court_dist_net = abs(cy - _COURT_NET_Y)
            # Court-space teammate separation
            self_court_dist_net = abs(cy - _COURT_NET_Y)
            other_court_dist_net = self_court_dist_net  # default for solo
            for other in same_side:
                if other == tid or other not in track_court_y:
                    continue
                d = abs(track_court_y[other] - _COURT_NET_Y)
                if d < other_court_dist_net or other_court_dist_net == self_court_dist_net:
                    other_court_dist_net = d
            court_teammate_sep_val = self_court_dist_net - other_court_dist_net
            has_court = 1.0

        # Pose features
        elev, asym, wrist_vel, single_frac = _extract_pose_features(
            track_pose_inputs.get(tid, []),
        )

        # Ball distance
        if first_ball_xy is not None:
            dx = st["mean_x"] - first_ball_xy[0]
            dy = foot_y_mean - first_ball_xy[1]
            ball_dist = math.sqrt(dx * dx + dy * dy)
        else:
            ball_dist = -1.0

        result[tid] = ServerFeatures(
            track_id=tid,
            n_frames=int(st["n_frames"]),
            foot_y_mean=foot_y_mean,
            foot_y_std=st["foot_y_std"],
            dist_from_net=self_dist_from_net,
            teammate_separation=teammate_sep,
            is_near_side=1.0 if is_near else 0.0,
            mean_x=st["mean_x"],
            bbox_height_mean=st["bbox_h_mean"],
            bbox_height_relative=bbox_h_relative,
            court_y=court_y_val,
            court_dist_from_baseline=court_dist_baseline,
            court_dist_from_net=court_dist_net,
            court_teammate_sep=court_teammate_sep_val,
            has_court_space=has_court,
            total_displacement=st["total_disp"],
            position_variance=st["foot_y_std"] ** 2 + st["x_std"] ** 2,
            max_velocity=st["max_vel"],
            first_frame_presence=st["first_presence"],
            max_wrist_elevation=elev,
            peak_arm_asymmetry=asym,
            single_arm_raise_frac=single_frac,
            max_wrist_velocity=wrist_vel,
            dist_to_first_ball=ball_dist,
        )

    return result


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class ServerClassifier:
    """Binary GBM classifier for server identification.

    For each rally, every tracked player gets a P(server) score from this
    model. The argmax player is selected as the server, and their court
    side determines the serve side. See predict() for the inference call
    used at runtime.
    """

    def __init__(self, model: Any = None):
        self.model = model
        self._feature_names = ServerFeatures.feature_names()

    @property
    def is_trained(self) -> bool:
        return self.model is not None

    def predict(
        self,
        features: dict[int, ServerFeatures],
        net_y: float | None = None,
    ) -> tuple[int, str, float]:
        """Predict server from per-player features.

        Returns (server_track_id, court_side, confidence). Confidence is the
        probability gap between the top-1 and top-2 candidates (0..1).
        Returns (-1, "", 0.0) if untrained or no features.
        """
        if not self.is_trained or not features:
            return -1, "", 0.0

        tids = list(features.keys())
        x_mat = np.array([features[t].to_array() for t in tids])
        probas = self.model.predict_proba(x_mat)[:, 1]

        order = np.argsort(probas)[::-1]
        top_idx = int(order[0])
        top_tid = tids[top_idx]
        top_p = float(probas[top_idx])
        runner_p = float(probas[order[1]]) if len(order) > 1 else 0.0
        confidence = max(0.0, top_p - runner_p)

        # Court side: prefer is_near_side feature; if net_y supplied, derive
        # from foot_y_mean (more robust if features were stale).
        feat = features[top_tid]
        if net_y is not None:
            side = "near" if feat.foot_y_mean > net_y else "far"
        else:
            side = "near" if feat.is_near_side >= 0.5 else "far"

        return top_tid, side, confidence

    def train(
        self,
        x: np.ndarray,
        y: np.ndarray,
        positive_weight: float = 3.0,
    ) -> dict[str, float]:
        """Train binary GBM. y=1 for the GT server, y=0 for non-servers."""
        from sklearn.ensemble import GradientBoostingClassifier

        self.model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=42,
        )
        sample_weights = np.where(y == 1, positive_weight, 1.0)
        self.model.fit(x, y, sample_weight=sample_weights)

        proba = self.model.predict_proba(x)[:, 1]
        pred = (proba >= 0.5).astype(int)
        tp = int(np.sum((pred == 1) & (y == 1)))
        fp = int(np.sum((pred == 1) & (y == 0)))
        fn = int(np.sum((pred == 0) & (y == 1)))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        return {
            "train_f1": f1,
            "train_precision": precision,
            "train_recall": recall,
            "n_samples": int(len(y)),
            "n_positive": int(np.sum(y)),
        }

    def loo_cv(
        self,
        x: np.ndarray,
        y: np.ndarray,
        rally_ids: np.ndarray,
        track_ids: np.ndarray,
        gt_server_tids: dict[str, int],
        gt_serve_sides: dict[str, str],
        foot_y_means: np.ndarray,
        net_ys: dict[str, float],
        positive_weight: float = 3.0,
        progress: bool = True,
    ) -> dict[str, Any]:
        """Leave-one-rally-out cross-validation with rally-level metrics.

        For each held-out rally:
          1. Train on all other rallies' samples
          2. Predict P(server) for each player in held-out rally
          3. Pick argmax as predicted server
          4. Compare to GT server track_id (server identification)
          5. Derive court side from predicted server's foot_y_mean vs net_y
          6. Compare court side to GT serve side
        """
        from sklearn.ensemble import GradientBoostingClassifier

        unique_rallies = list(np.unique(rally_ids))
        n_rallies = len(unique_rallies)

        n_id_correct = 0
        n_side_correct = 0
        n_evaluated = 0

        per_rally: list[dict[str, Any]] = []

        for i, rally in enumerate(unique_rallies):
            test_mask = rally_ids == rally
            train_mask = ~test_mask

            if np.sum(train_mask) < 10 or np.sum(y[train_mask]) < 3:
                continue

            model = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                min_samples_leaf=5,
                subsample=0.8,
                random_state=42,
            )
            train_w = np.where(y[train_mask] == 1, positive_weight, 1.0)
            model.fit(x[train_mask], y[train_mask], sample_weight=train_w)

            test_x = x[test_mask]
            test_tids = track_ids[test_mask]
            test_foot_ys = foot_y_means[test_mask]
            probas = model.predict_proba(test_x)[:, 1]

            top_idx = int(np.argmax(probas))
            pred_tid = int(test_tids[top_idx])
            pred_foot_y = float(test_foot_ys[top_idx])

            net_y = net_ys.get(str(rally), 0.5)
            pred_side = "near" if pred_foot_y > net_y else "far"

            gt_tid = gt_server_tids.get(str(rally), -1)
            gt_side = gt_serve_sides.get(str(rally), "")

            id_correct = pred_tid == gt_tid and gt_tid >= 0
            side_correct = (gt_side != "") and pred_side == gt_side

            if gt_tid >= 0:
                n_evaluated += 1
                if id_correct:
                    n_id_correct += 1
                if side_correct:
                    n_side_correct += 1

            per_rally.append({
                "rally_id": str(rally),
                "gt_tid": gt_tid,
                "pred_tid": pred_tid,
                "gt_side": gt_side,
                "pred_side": pred_side,
                "id_correct": id_correct,
                "side_correct": side_correct,
                "top_p": float(probas[top_idx]),
            })

            if progress and (i + 1) % 25 == 0:
                logger.info(
                    "[%d/%d] LOO progress: id_acc=%.1f%% side_acc=%.1f%%",
                    i + 1, n_rallies,
                    100.0 * n_id_correct / max(1, n_evaluated),
                    100.0 * n_side_correct / max(1, n_evaluated),
                )

        return {
            "n_rallies": n_rallies,
            "n_evaluated": n_evaluated,
            "id_accuracy": n_id_correct / max(1, n_evaluated),
            "side_accuracy": n_side_correct / max(1, n_evaluated),
            "n_id_correct": n_id_correct,
            "n_side_correct": n_side_correct,
            "per_rally": per_rally,
        }

    def feature_importance(self) -> dict[str, float]:
        if not self.is_trained:
            return {}
        importances = self.model.feature_importances_
        return dict(zip(self._feature_names, [float(v) for v in importances]))

    def save(self, path: str | Path) -> None:
        import pickle

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "model": self.model,
            "feature_names": self._feature_names,
            "feature_version": FEATURE_VERSION,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info("Saved server classifier to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> ServerClassifier:
        import pickle

        with open(path, "rb") as f:
            data = pickle.load(f)  # noqa: S301

        version = data.get("feature_version", 0)
        if version != FEATURE_VERSION:
            logger.warning(
                "Server classifier feature version mismatch: file=%d expected=%d. "
                "Returning untrained classifier; please retrain.",
                version, FEATURE_VERSION,
            )
            return cls()

        clf = cls(model=data["model"])
        clf._feature_names = data.get(
            "feature_names", ServerFeatures.feature_names(),
        )
        return clf


# Default model path
DEFAULT_SERVER_MODEL_PATH = Path("weights/server_classifier/server_classifier.pkl")


def load_server_classifier(
    model_path: str | Path | None = None,
) -> ServerClassifier | None:
    """Load server classifier from default or specified path. Returns None if not found."""
    path = Path(model_path) if model_path else DEFAULT_SERVER_MODEL_PATH
    if path.exists():
        try:
            return ServerClassifier.load(path)
        except Exception as e:
            logger.warning("Failed to load server classifier from %s: %s", path, e)
    return None
