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
)

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
    x: float          # bbox top-left x (normalized)
    y: float          # bbox top-left y (normalized)
    width: float
    height: float


def position_from_dict(d: dict[str, Any]) -> PlayerPositionLike:
    return PlayerPositionLike(
        frame_number=int(d.get("frameNumber", d.get("frame_number", -1))),
        track_id=int(d.get("trackId", d.get("track_id", -1))),
        x=float(d.get("x", 0.0)),
        y=float(d.get("y", 0.0)),
        width=float(d.get("width", 0.0)),
        height=float(d.get("height", 0.0)),
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


def extract_features(
    positions: list[PlayerPositionLike],
    track_id: int,
    contact_frame: int,
    ball_x: float,
    ball_y: float,
    prev_action_tid: int = -1,
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
    )


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

    Env flag: `USE_DYNAMIC_ATTRIBUTION_SCORER` ∈ {"1", "true"} to enable.
    """
    import os

    enabled = os.environ.get("USE_DYNAMIC_ATTRIBUTION_SCORER", "0").lower() in ("1", "true", "yes")
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
