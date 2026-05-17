"""Dynamic-feature attribution scorer.

Per-action-type GBM that picks the most likely player for a contact among
candidate primary tracks. Combines static features (bbox distance, area,
aspect ratio, inside-frame) with dynamic features computed across a small
temporal window around the contact (velocity, top-y change, height change).

Training data: trusted player-attribution GT (`rally_action_ground_truth`)
across the trusted-14 corpus (~678 positive labels).

Validated on 2026-05-14:
  - LOO CV (corpus-wide): +12.4pp vs current pipeline picker on 678 contacts
  - User-flagged keke r1: all 4 actions correctly attributed (serve P2,
    receive P3, set P4, attack P3) when trained on full corpus

Spec: docs/superpowers/specs/2026-05-14-dynamic-attribution-scorer-design.md
(to be written if shipping production).

Production training: scripts/train_and_save_dynamic_scorer_2026_05_14.py
Honest LOO measurement: scripts/train_dynamic_attribution_scorer_2026_05_14.py
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Feature names in the order expected by trained models.
FEATURE_NAMES = (
    "bbox_dist",
    "bbox_area",
    "bbox_aspect_ratio",
    "bbox_inside_frame",
    "velocity_mag",
    "velocity_toward_ball",
    "top_y_at_contact",
    "top_y_change",
    "height_change",
    "same_as_prev",  # 1.0 if candidate.tid == previous action's playerTrackId else 0.0
    # Pose-dynamics features (v2, 2026-05-15). All zeros if keypoints unavailable.
    "wrist_velocity_max",
    "wrist_to_ball_min",
    "body_orientation_diff",
    "arms_raised",
    "wrist_post_alignment",
    "pose_confidence_mean",
    # v2.1 (2026-05-15): target ATTACK contest residual
    "wrist_y_velocity",  # vertical wrist velocity at contact; attacker descends, blocker stays
    # v3 (2026-05-17): team-awareness — 63% of v2 ATTACK errors were CROSS_TEAM
    # picks where the scorer chose the blocker on the opposite team. Without
    # a team-context signal the scorer can't distinguish "geometrically
    # similar near-ball candidate" from "actual attacker on the right team".
    # 1.0 if candidate's team matches the team-chain expected team for this
    # action, 0.0 if mismatched, 0.5 if uninformative (no team chain or no
    # team assignment for this candidate).
    "team_matches_expected",
)

# COCO 17-keypoint indices used in pose features.
KPT_LEFT_SHOULDER = 5
KPT_RIGHT_SHOULDER = 6
KPT_LEFT_WRIST = 9
KPT_RIGHT_WRIST = 10
KPT_VIS_THRESHOLD = 0.3

# Default location of trained models relative to repo root.
_DEFAULT_MODELS_DIR = (
    Path(__file__).resolve().parents[2] / "weights" / "dynamic_attribution_scorer"
)


@dataclass(frozen=True)
class CandidateFeatures:
    """Per-candidate feature vector at a single contact."""

    track_id: int
    bbox_dist: float
    bbox_area: float
    bbox_aspect_ratio: float
    bbox_inside_frame: float
    velocity_mag: float
    velocity_toward_ball: float
    top_y_at_contact: float
    top_y_change: float
    height_change: float
    same_as_prev: float = 0.0
    # Pose features (v2). Default to 0 / safe values when keypoints unavailable.
    wrist_velocity_max: float = 0.0
    wrist_to_ball_min: float = 1.0
    body_orientation_diff: float = math.pi
    arms_raised: float = 0.0
    wrist_post_alignment: float = 0.0
    pose_confidence_mean: float = 0.0
    # v2.1
    wrist_y_velocity: float = 0.0
    # v3 — default 0.5 = uninformative (no team chain or no assignment)
    team_matches_expected: float = 0.5

    def as_vector(self) -> list[float]:
        return [
            self.bbox_dist,
            self.bbox_area,
            self.bbox_aspect_ratio,
            self.bbox_inside_frame,
            self.velocity_mag,
            self.velocity_toward_ball,
            self.top_y_at_contact,
            self.top_y_change,
            self.height_change,
            self.same_as_prev,
            self.wrist_velocity_max,
            self.wrist_to_ball_min,
            self.body_orientation_diff,
            self.arms_raised,
            self.wrist_post_alignment,
            self.pose_confidence_mean,
            self.wrist_y_velocity,
            self.team_matches_expected,
        ]


@dataclass(frozen=True)
class PlayerPositionLike:
    """Minimal player-position record used by the scorer.

    Compatible with `rallycut.tracking.contact_detector.PlayerPosition` (the
    fields used here are a subset). Also accepts plain dicts via the
    `from_dict` helper so the scorer can be exercised from probes.
    """

    frame_number: int
    track_id: int
    x: float          # bbox center x (normalized)
    y: float          # bbox center y (normalized)
    width: float
    height: float
    # COCO 17-keypoint pose data (x, y, conf) per keypoint; None when not
    # enriched. When present, pose features are computed.
    keypoints: list[list[float]] | None = None


def position_from_dict(d: dict[str, Any]) -> PlayerPositionLike:
    kps = d.get("keypoints")
    return PlayerPositionLike(
        frame_number=int(d.get("frameNumber", d.get("frame_number", -1))),
        track_id=int(d.get("trackId", d.get("track_id", -1))),
        x=float(d.get("x", 0.0)),
        y=float(d.get("y", 0.0)),
        width=float(d.get("width", 0.0)),
        height=float(d.get("height", 0.0)),
        keypoints=kps if (kps and len(kps) >= 17) else None,
    )


def _find_pos(
    positions: list[PlayerPositionLike],
    track_id: int,
    target_frame: int,
    tolerance: int = 5,
) -> PlayerPositionLike | None:
    best: PlayerPositionLike | None = None
    best_delta = tolerance + 1
    for p in positions:
        if p.track_id != track_id:
            continue
        delta = abs(p.frame_number - target_frame)
        if delta < best_delta:
            best_delta = delta
            best = p
    return best


def _bbox_upper_quarter_dist(p: PlayerPositionLike, ball_x: float, ball_y: float) -> float:
    """Image-space distance using the bbox upper-quarter (mirrors
    `contact_detector._player_to_ball_dist`'s bbox fallback)."""
    px = p.x
    py = p.y - p.height * 0.25
    return math.hypot(px - ball_x, py - ball_y)


def _team_match_feature(
    track_id: int,
    expected_team: int | None,
    team_assignments: dict[int, int] | None,
) -> float:
    """0.5 = uninformative, 1.0 = candidate's team matches expected, 0.0 = mismatch.

    The default-0.5 fallback ensures the scorer gets no team signal (rather
    than a misleading one) when the team-chain is broken or the candidate
    has no team assignment.
    """
    if expected_team is None or team_assignments is None:
        return 0.5
    cand_team = team_assignments.get(track_id)
    if cand_team is None:
        return 0.5
    return 1.0 if int(cand_team) == int(expected_team) else 0.0


def extract_features(
    positions: list[PlayerPositionLike],
    track_id: int,
    contact_frame: int,
    ball_x: float,
    ball_y: float,
    prev_action_tid: int = -1,
    post_ball_x: float | None = None,
    post_ball_y: float | None = None,
    expected_team: int | None = None,
    team_assignments: dict[int, int] | None = None,
) -> CandidateFeatures | None:
    """Compute the 9-feature vector for one candidate at one contact.

    Returns None if the candidate has no bbox within ±2 frames of contact.

    Mirrors the feature extraction in
    `scripts/train_and_save_dynamic_scorer_2026_05_14.py` exactly — any
    change in either side must be mirrored in the other.
    """
    p_at = _find_pos(positions, track_id, contact_frame, tolerance=5)
    if p_at is None:
        return None
    p_prev = _find_pos(positions, track_id, contact_frame - 5, tolerance=5)
    p_next = _find_pos(positions, track_id, contact_frame + 5, tolerance=5)
    p_pre_extend = _find_pos(positions, track_id, contact_frame - 3, tolerance=5)
    p_post_extend = _find_pos(positions, track_id, contact_frame + 3, tolerance=5)

    x = p_at.x
    y = p_at.y
    w = p_at.width
    h = p_at.height
    bbox_dist = _bbox_upper_quarter_dist(p_at, ball_x, ball_y)
    bbox_area = w * h
    bbox_aspect_ratio = w / max(h, 1e-6)
    inside = 1.0 if (x >= 0 and y >= 0 and x + w <= 1.0 and y + h <= 1.0) else 0.0

    if p_prev is not None and p_next is not None:
        cx_prev = p_prev.x + p_prev.width / 2
        cy_prev = p_prev.y + p_prev.height / 2
        cx_next = p_next.x + p_next.width / 2
        cy_next = p_next.y + p_next.height / 2
        dx = cx_next - cx_prev
        dy = cy_next - cy_prev
        velocity_mag = math.hypot(dx, dy)
        cx_at = x + w / 2
        cy_at = y + h / 2
        to_ball_x = ball_x - cx_at
        to_ball_y = ball_y - cy_at
        to_ball_mag = math.hypot(to_ball_x, to_ball_y) + 1e-6
        velocity_toward_ball = (dx * to_ball_x + dy * to_ball_y) / to_ball_mag
    else:
        velocity_mag = 0.0
        velocity_toward_ball = 0.0

    if p_prev is not None:
        top_y_change = y - p_prev.y
    else:
        top_y_change = 0.0

    if p_pre_extend is not None and p_post_extend is not None:
        height_change = p_post_extend.height - p_pre_extend.height
    else:
        height_change = 0.0

    same_as_prev = 1.0 if (prev_action_tid >= 0 and track_id == prev_action_tid) else 0.0

    # Pose features (v2). Compute when keypoints are available.
    pose = _compute_pose_features(
        positions, track_id, contact_frame, ball_x, ball_y,
        post_ball_x, post_ball_y,
    )
    team_matches_expected = _team_match_feature(
        track_id, expected_team, team_assignments,
    )
    return CandidateFeatures(
        track_id=track_id,
        bbox_dist=bbox_dist,
        bbox_area=bbox_area,
        bbox_aspect_ratio=bbox_aspect_ratio,
        bbox_inside_frame=inside,
        velocity_mag=velocity_mag,
        velocity_toward_ball=velocity_toward_ball,
        top_y_at_contact=y,
        top_y_change=top_y_change,
        height_change=height_change,
        same_as_prev=same_as_prev,
        team_matches_expected=team_matches_expected,
        **pose,
    )


def _wrist_xy(p: PlayerPositionLike, which: str) -> tuple[float, float, float] | None:
    """Return (x, y, conf) of left or right wrist; None if not visible."""
    if p.keypoints is None or len(p.keypoints) < 17:
        return None
    idx = KPT_LEFT_WRIST if which == "left" else KPT_RIGHT_WRIST
    kx, ky, kc = p.keypoints[idx]
    if kc < KPT_VIS_THRESHOLD:
        return None
    return float(kx), float(ky), float(kc)


def _shoulder_xy(p: PlayerPositionLike, which: str) -> tuple[float, float, float] | None:
    if p.keypoints is None or len(p.keypoints) < 17:
        return None
    idx = KPT_LEFT_SHOULDER if which == "left" else KPT_RIGHT_SHOULDER
    kx, ky, kc = p.keypoints[idx]
    if kc < KPT_VIS_THRESHOLD:
        return None
    return float(kx), float(ky), float(kc)


_POSE_WINDOW = 5


def _compute_pose_features(
    positions: list[PlayerPositionLike],
    track_id: int,
    contact_frame: int,
    ball_x: float,
    ball_y: float,
    post_ball_x: float | None = None,
    post_ball_y: float | None = None,
) -> dict[str, float]:
    """Compute 7 pose-based features for one candidate.

    Returns dict with defaults when keypoints are unavailable. Mirrors
    `scripts/probe_pose_features_2026_05_15.py::compute_pose_features` —
    keep in lockstep.
    """
    track_positions = sorted(
        [p for p in positions
         if p.track_id == track_id
         and abs(p.frame_number - contact_frame) <= _POSE_WINDOW],
        key=lambda p: p.frame_number,
    )
    wrist_pos: dict[int, tuple[float, float]] = {}
    confs: list[float] = []
    arms_raised_at_contact = 0.0
    for p in track_positions:
        lw = _wrist_xy(p, "left")
        rw = _wrist_xy(p, "right")
        best_w: tuple[float, float] | None = None
        best_d = float("inf")
        for w in (lw, rw):
            if w is None:
                continue
            d = math.hypot(w[0] - ball_x, w[1] - ball_y)
            if d < best_d:
                best_d = d
                best_w = (w[0], w[1])
            confs.append(w[2])
        if best_w is not None:
            wrist_pos[p.frame_number] = best_w
        ls = _shoulder_xy(p, "left")
        rs = _shoulder_xy(p, "right")
        for s in (ls, rs):
            if s is not None:
                confs.append(s[2])
        if abs(p.frame_number - contact_frame) <= 2:
            if lw and rw and ls and rs:
                if lw[1] < ls[1] and rw[1] < rs[1]:
                    arms_raised_at_contact = 1.0

    # If we have no pose data at all, return defaults
    if not wrist_pos and not confs:
        return {
            "wrist_velocity_max": 0.0,
            "wrist_to_ball_min": 1.0,
            "body_orientation_diff": math.pi,
            "arms_raised": 0.0,
            "wrist_post_alignment": 0.0,
            "pose_confidence_mean": 0.0,
            "wrist_y_velocity": 0.0,
        }

    sorted_frames = sorted(wrist_pos.keys())
    wrist_velocity_max = 0.0
    wrist_y_velocity_at_contact = 0.0
    best_vel: tuple[float, float, float] | None = None
    for i in range(len(sorted_frames) - 1):
        f1, f2 = sorted_frames[i], sorted_frames[i + 1]
        if f2 - f1 > 3:
            continue
        x1, y1 = wrist_pos[f1]
        x2, y2 = wrist_pos[f2]
        gap = max(1, f2 - f1)
        dx_t, dy_t = (x2 - x1), (y2 - y1)
        d = math.hypot(dx_t, dy_t) / gap
        if d > wrist_velocity_max:
            wrist_velocity_max = d
        # Y-velocity near the contact frame (positive y = descending in image)
        if min(abs(f1 - contact_frame), abs(f2 - contact_frame)) <= 2:
            dy_per_frame = dy_t / gap
            if abs(dy_per_frame) > abs(wrist_y_velocity_at_contact):
                wrist_y_velocity_at_contact = dy_per_frame
        # Track best velocity vector (raw, not /gap) for post-alignment
        d_t = math.hypot(dx_t, dy_t)
        if best_vel is None or d_t > best_vel[2]:
            best_vel = (dx_t, dy_t, d_t)

    wrist_to_ball_min = float("inf")
    for f, (wx, wy) in wrist_pos.items():
        if abs(f - contact_frame) <= 2:
            d = math.hypot(wx - ball_x, wy - ball_y)
            if d < wrist_to_ball_min:
                wrist_to_ball_min = d
    if not math.isfinite(wrist_to_ball_min):
        wrist_to_ball_min = 1.0

    # Body orientation: angle between body-perpendicular and direction-to-ball
    body_orientation_diff = math.pi
    p_contact = next(
        (p for p in track_positions if abs(p.frame_number - contact_frame) <= 1),
        None,
    )
    if p_contact is not None:
        ls = _shoulder_xy(p_contact, "left")
        rs = _shoulder_xy(p_contact, "right")
        if ls is not None and rs is not None:
            sx = rs[0] - ls[0]
            sy = rs[1] - ls[1]
            facing_x = -sy
            facing_y = sx
            torso_x = (ls[0] + rs[0]) / 2
            torso_y = (ls[1] + rs[1]) / 2
            to_ball_x = ball_x - torso_x
            to_ball_y = ball_y - torso_y
            mag_f = math.hypot(facing_x, facing_y) + 1e-6
            mag_b = math.hypot(to_ball_x, to_ball_y) + 1e-6
            cos_theta = (facing_x * to_ball_x + facing_y * to_ball_y) / (mag_f * mag_b)
            cos_theta = max(-1.0, min(1.0, cos_theta))
            body_orientation_diff = math.acos(cos_theta)

    wrist_post_alignment = 0.0
    if (post_ball_x is not None and post_ball_y is not None
            and best_vel is not None and best_vel[2] > 0):
        ball_dx = post_ball_x - ball_x
        ball_dy = post_ball_y - ball_y
        ball_mag = math.hypot(ball_dx, ball_dy) + 1e-6
        wrist_post_alignment = (
            (best_vel[0] * ball_dx + best_vel[1] * ball_dy)
            / (best_vel[2] * ball_mag)
        )

    pose_confidence_mean = (sum(confs) / len(confs)) if confs else 0.0

    return {
        "wrist_velocity_max": wrist_velocity_max,
        "wrist_to_ball_min": wrist_to_ball_min,
        "body_orientation_diff": body_orientation_diff,
        "arms_raised": arms_raised_at_contact,
        "wrist_post_alignment": wrist_post_alignment,
        "pose_confidence_mean": pose_confidence_mean,
        "wrist_y_velocity": wrist_y_velocity_at_contact,
    }


class DynamicAttributionScorer:
    """Loads per-action-type GBMs and scores candidates for a contact.

    Thread-unsafe; create one instance per process. Models are lazy-loaded
    on first use of each action type.
    """

    def __init__(self, models_dir: Path | str | None = None, version: str = "v1") -> None:
        self.models_dir = Path(models_dir) if models_dir is not None else _DEFAULT_MODELS_DIR
        self.version = version
        self._models: dict[str, Any] = {}
        self._missing: set[str] = set()
        self._manifest: dict[str, Any] | None = None

    @property
    def is_available(self) -> bool:
        """True iff the models directory exists with a manifest."""
        return (self.models_dir / "manifest.json").exists()

    def _load_manifest(self) -> dict[str, Any]:
        if self._manifest is None:
            mp = self.models_dir / "manifest.json"
            if mp.exists():
                self._manifest = json.loads(mp.read_text())
            else:
                self._manifest = {}
        return self._manifest

    def _get_model(self, action: str) -> Any | None:
        key = action.upper()
        if key in self._models:
            return self._models[key]
        if key in self._missing:
            return None
        path = self.models_dir / f"{key}_{self.version}.joblib"
        if not path.exists():
            self._missing.add(key)
            logger.debug("No scorer model for %s at %s", key, path)
            return None
        try:
            import joblib
            model = joblib.load(path)
            self._models[key] = model
            return model
        except Exception as exc:
            logger.warning("Failed to load scorer model for %s: %s", key, exc)
            self._missing.add(key)
            return None

    def score(
        self,
        action: str,
        candidates: list[CandidateFeatures],
    ) -> list[float] | None:
        """Return per-candidate P(this candidate is the GT).

        Returns None when the per-action model is unavailable. Caller should
        fall back to the existing picker in that case.
        """
        if not candidates:
            return []
        model = self._get_model(action)
        if model is None:
            return None
        feature_matrix = [c.as_vector() for c in candidates]
        try:
            probs = model.predict_proba(feature_matrix)
            # Probability of class 1 (positive: this candidate is the GT).
            return [float(p[1]) for p in probs]
        except Exception as exc:
            logger.warning("scorer.score failed for action=%s: %s", action, exc)
            return None

    def pick(
        self,
        action: str,
        candidates: list[CandidateFeatures],
    ) -> int | None:
        """Return the track_id of the highest-scoring candidate.

        Returns None when no per-action model is loaded (caller falls back to
        the existing nearest-player picker).
        """
        if not candidates:
            return None
        probs = self.score(action, candidates)
        if probs is None:
            return None
        best_idx = max(range(len(candidates)), key=lambda i: probs[i])
        return candidates[best_idx].track_id

    def pick_with_probs(
        self,
        action: str,
        candidates: list[CandidateFeatures],
    ) -> tuple[int, list[float]] | None:
        """Return (best track_id, per-candidate probs). For debugging."""
        if not candidates:
            return None
        probs = self.score(action, candidates)
        if probs is None:
            return None
        best_idx = max(range(len(candidates)), key=lambda i: probs[i])
        return candidates[best_idx].track_id, probs


# ----------------------------------------------------------------------------
# Convenience: process-wide singleton (lazy-init), gated by env flag.
# ----------------------------------------------------------------------------

_singleton: DynamicAttributionScorer | None = None


def get_scorer(models_dir: Path | str | None = None) -> DynamicAttributionScorer | None:
    """Return the process-wide scorer if env flag enables it AND models exist.

    Env flag: `USE_DYNAMIC_ATTRIBUTION_SCORER` — default "1" (ON) as of v3.1
    ship (2026-05-17). Set to "0" / "false" / "no" to disable for rollback.
    """
    import os

    enabled = os.environ.get("USE_DYNAMIC_ATTRIBUTION_SCORER", "1").lower() in ("1", "true", "yes")
    if not enabled:
        return None
    global _singleton
    if _singleton is None:
        _singleton = DynamicAttributionScorer(models_dir=models_dir)
        if not _singleton.is_available:
            logger.warning(
                "USE_DYNAMIC_ATTRIBUTION_SCORER=1 but no manifest at %s",
                _singleton.models_dir,
            )
            _singleton = None
    return _singleton


__all__ = [
    "CandidateFeatures",
    "PlayerPositionLike",
    "FEATURE_NAMES",
    "DynamicAttributionScorer",
    "extract_features",
    "get_scorer",
    "position_from_dict",
]
