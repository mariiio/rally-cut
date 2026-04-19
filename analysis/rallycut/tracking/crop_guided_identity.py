"""Crop-guided player identity resolver.

User provides 1-N reference crops per player (4 players per match). This module
builds DINOv2 prototypes and exposes scoring primitives the post-processing
chain uses to anchor identity at both per-track and per-segment granularity.

Pipeline integration:
- Prototypes built once per match from `PlayerReferenceCrop` rows.
- ``IdentityAnchors`` is threaded through ``apply_post_processing`` as an
  optional kwarg; when ``None``, the legacy code path runs bit-identically.
- Post-processing's final step performs Hungarian assignment from primary
  tracks to player prototypes, overriding any residual same-team swaps.
- ``validate_prototypes()`` is the UX-facing pre-flight check that blocks
  "Re-run Matching" when user-selected crops cannot produce reliable
  prototypes (identical players, misassigned crops, single-angle coverage).

Prototype recipe (from SOTA research @ N=1-5 references):
Average DINOv2 ViT-S/14 (384-d) embeddings per player, L2-normalize. Argmax
cosine similarity with margin-thresholded confidence. Session 3 learned head
and OSNet ensemble are reserved as fallback paths if this saturates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from rallycut.tracking.reid_embeddings import extract_backbone_features

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Thresholds (tune on 43-rally eval set)
# ---------------------------------------------------------------------------

# Minimum margin (top_sim - runner_up_sim) for a "confident" assignment.
# Used by score_crop and the entry-seed+lock step in apply_post_processing.
LOCK_MARGIN_THRESHOLD = 0.15

# Minimum per-crop similarity to the player's own prototype. Below this the
# crop is treated as uninformative (occluded, blurry, wrong frame).
MIN_CROP_SELF_SIM = 0.45

# Quality validator thresholds.
MIN_CROPS_PER_PLAYER = 1
RECOMMENDED_CROPS_PER_PLAYER = 2
# Minimum mean pairwise cosine distance across a player's own crops. Below
# this the crops all look the same (one-angle / near-duplicate selection).
MIN_INTRA_PLAYER_SPREAD = 0.04
# Minimum pairwise cosine distance between any two player prototypes. Below
# this the two players look identical in embedding space — impossible to
# separate with this crop set. User must add a more discriminating crop.
MIN_INTER_PLAYER_DISTANCE = 0.08
# Multiplier applied to "own-prototype" similarity when flagging misassigned
# crops. A crop whose similarity to its assigned player's prototype is lower
# than `factor *` its similarity to any other player's prototype is likely
# misassigned by the user.
MISASSIGN_FACTOR = 1.10


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class IdentityAnchors:
    """Crop-guided identity anchors for a match.

    Attributes:
        prototypes: ``{player_id: (384,) float32 L2-normalized vector}``.
        per_crop_embeddings: ``{player_id: (N, 384) float32}`` retained so
            downstream callers (e.g., segment-split) can decide whether to
            use prototype-only or per-crop voting.
        lock_margin: Margin threshold for hard locks (default
            :data:`LOCK_MARGIN_THRESHOLD`).
        source: Short human-readable origin string for logs (``"user"`` or
            ``"dogfood"``).
    """

    prototypes: dict[int, NDArray[np.float32]]
    per_crop_embeddings: dict[int, NDArray[np.float32]] = field(default_factory=dict)
    lock_margin: float = LOCK_MARGIN_THRESHOLD
    source: str = "user"

    @property
    def player_ids(self) -> list[int]:
        return sorted(self.prototypes.keys())

    @property
    def prototype_matrix(self) -> NDArray[np.float32]:
        """(P, 384) matrix of prototypes, rows ordered by :attr:`player_ids`."""
        return np.stack([self.prototypes[pid] for pid in self.player_ids], axis=0)


@dataclass(frozen=True)
class CropScore:
    """Result of scoring a single crop / track embedding against prototypes."""

    player_id: int
    margin: float
    confidence: float
    similarities: dict[int, float]

    @property
    def is_confident(self) -> bool:
        return self.margin >= LOCK_MARGIN_THRESHOLD


@dataclass
class ValidationIssue:
    """A problem with user-provided reference crops.

    Consumed by the web client to render inline per-player guidance.
    """

    code: str
    message: str
    player_id: int | None = None


@dataclass
class ValidationResult:
    ok: bool
    issues: list[ValidationIssue]

    def to_dict(self) -> dict[str, Any]:
        return {
            "pass": self.ok,
            "issues": [
                {
                    "code": issue.code,
                    "message": issue.message,
                    "playerId": issue.player_id,
                }
                for issue in self.issues
            ],
        }


# ---------------------------------------------------------------------------
# Prototype construction
# ---------------------------------------------------------------------------


def _load_crop_bgr(path: Path) -> NDArray[np.uint8] | None:
    """Load a JPEG crop as BGR uint8. Returns None on failure."""
    try:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load crop %s: %s", path, exc)
        return None
    if img is None or img.size == 0 or img.shape[0] < 16 or img.shape[1] < 8:
        return None
    return np.asarray(img, dtype=np.uint8)


def build_anchors_from_crops(
    crops_by_player: dict[int, list[NDArray[np.uint8]]],
    *,
    device: str | None = None,
    source: str = "user",
) -> IdentityAnchors:
    """Build :class:`IdentityAnchors` from in-memory crop images.

    Prototypes are mean-pooled and L2-normalized DINOv2 ViT-S/14 embeddings.
    Empty inputs are silently dropped (downstream validator surfaces the
    "no crops for player" issue to the user).
    """
    prototypes: dict[int, NDArray[np.float32]] = {}
    per_crop: dict[int, NDArray[np.float32]] = {}

    for pid, crops in crops_by_player.items():
        valid = [c for c in crops if c is not None and c.size]
        if not valid:
            continue
        feats = extract_backbone_features(valid, device=device)
        if feats.shape[0] == 0:
            continue
        proto = feats.mean(axis=0)
        norm = float(np.linalg.norm(proto))
        if norm < 1e-6:
            continue
        prototypes[pid] = (proto / norm).astype(np.float32)
        per_crop[pid] = feats.astype(np.float32)

    return IdentityAnchors(
        prototypes=prototypes,
        per_crop_embeddings=per_crop,
        source=source,
    )


def build_anchors_from_paths(
    crops_by_player: dict[int, list[Path | str]],
    *,
    device: str | None = None,
    source: str = "user",
) -> IdentityAnchors:
    """Build :class:`IdentityAnchors` from on-disk JPEG paths.

    Used by the CLI / API integration when the backend has pre-downloaded
    S3-hosted reference crops into a temp directory.
    """
    loaded: dict[int, list[NDArray[np.uint8]]] = {}
    for pid, paths in crops_by_player.items():
        crops: list[NDArray[np.uint8]] = []
        for p in paths:
            img = _load_crop_bgr(Path(p))
            if img is not None:
                crops.append(img)
        if crops:
            loaded[pid] = crops
    return build_anchors_from_crops(loaded, device=device, source=source)


# ---------------------------------------------------------------------------
# Scoring primitives
# ---------------------------------------------------------------------------


def _cosine_similarities(
    embeddings: NDArray[np.float32],
    prototypes: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Cosine similarity between (N, D) embeddings and (P, D) prototypes.

    Inputs are expected to be L2-normalized (DINOv2 backbone + mean-pool
    already normalize), so this is a plain dot product.

    Returns:
        (N, P) float32.
    """
    if embeddings.ndim == 1:
        embeddings = embeddings[None, :]
    sims = embeddings @ prototypes.T
    return sims.astype(np.float32)


def score_embedding(
    embedding: NDArray[np.float32],
    anchors: IdentityAnchors,
) -> CropScore:
    """Score a single L2-normalized embedding against prototypes."""
    pids = anchors.player_ids
    if not pids:
        return CropScore(player_id=-1, margin=0.0, confidence=0.0, similarities={})

    sims_row = _cosine_similarities(embedding, anchors.prototype_matrix)[0]
    sims = {pid: float(sims_row[i]) for i, pid in enumerate(pids)}
    best_idx = int(np.argmax(sims_row))
    best_pid = pids[best_idx]
    best_sim = float(sims_row[best_idx])

    if len(pids) >= 2:
        # Margin vs the best non-self similarity.
        sorted_sims = np.sort(sims_row)[::-1]
        runner_up = float(sorted_sims[1])
        margin = best_sim - runner_up
    else:
        margin = best_sim

    # Confidence combines raw similarity with margin to penalize close calls.
    # Clamped to [0, 1]; tuned informally so `is_confident` ~= >0.55.
    confidence = max(0.0, min(1.0, 0.5 * (best_sim + 1.0) * min(1.0, margin / 0.30)))

    return CropScore(
        player_id=best_pid,
        margin=float(margin),
        confidence=float(confidence),
        similarities=sims,
    )


def score_crops(
    crops: list[NDArray[np.uint8]],
    anchors: IdentityAnchors,
    *,
    device: str | None = None,
) -> list[CropScore]:
    """Score a batch of BGR crops. Empty list returns empty list."""
    if not crops or not anchors.prototypes:
        return []
    embeddings = extract_backbone_features(crops, device=device)
    return [score_embedding(embeddings[i], anchors) for i in range(embeddings.shape[0])]


# ---------------------------------------------------------------------------
# Pre-flight quality validation (UX gate)
# ---------------------------------------------------------------------------


def validate_prototypes(
    anchors: IdentityAnchors,
    *,
    expected_player_ids: list[int] | None = None,
    min_crops: int = MIN_CROPS_PER_PLAYER,
    recommended_crops: int = RECOMMENDED_CROPS_PER_PLAYER,
    min_intra_spread: float = MIN_INTRA_PLAYER_SPREAD,
    min_inter_distance: float = MIN_INTER_PLAYER_DISTANCE,
    misassign_factor: float = MISASSIGN_FACTOR,
    partial_ok: bool = True,
) -> ValidationResult:
    """Run the UX-facing pre-flight check on anchors.

    Emits structured issues for the web client to render per-player guidance.
    A failing validation blocks the "Re-run Matching" button.

    Checks:
        1. Coverage — ``partial_ok=True`` (default): at least one expected
           player has crops; missing players surface as non-blocking warnings.
           ``partial_ok=False``: every expected player must have ``min_crops``.
        2. Each labeled player's crops cover enough of the embedding space
           (not all selected from one frame / angle).
        3. Every pair of labeled player prototypes is separable in embedding
           space.
        4. Each individual crop is closer to its own prototype than to the
           best-matching other labeled player's prototype.

    Args:
        anchors: Built via :func:`build_anchors_from_paths`.
        expected_player_ids: If provided, enforces coverage semantics over
            this list. Defaults to ``anchors.player_ids``.
        min_crops: Hard minimum crops per player present in anchors.
        recommended_crops: Soft minimum — emits a non-blocking warning.
        min_intra_spread: Minimum mean intra-player pairwise cosine distance.
        min_inter_distance: Minimum inter-player prototype distance.
        misassign_factor: Threshold multiplier for flagging likely mis-
            assigned crops (see module-level constant docstring).
        partial_ok: When True (default), allow the user to label a strict
            subset of the expected players. The downstream override path
            fixes same-team swaps involving labeled players only; unlabeled
            players keep their baseline assignment.
    """
    expected = sorted(expected_player_ids or anchors.player_ids)
    issues: list[ValidationIssue] = []

    # --- Check 1: coverage --------------------------------------------------
    for pid in expected:
        crops = anchors.per_crop_embeddings.get(pid)
        n = 0 if crops is None else int(crops.shape[0])
        if n < min_crops:
            # Missing crops are blocking only when partial labeling is off.
            # In partial mode they surface as informational guidance.
            issues.append(
                ValidationIssue(
                    code="missing_crops_optional" if partial_ok else "missing_crops",
                    player_id=pid,
                    message=(
                        f"Player {pid} has no reference crops — "
                        + (
                            "skipped by the crop-guided matcher."
                            if partial_ok
                            else f"add at least {min_crops} crop before re-running matching."
                        )
                    ),
                )
            )
        elif n < recommended_crops:
            issues.append(
                ValidationIssue(
                    code="few_crops",
                    player_id=pid,
                    message=(
                        f"Player {pid} has only {n} crop — add one more from a "
                        f"different angle for more reliable matching."
                    ),
                )
            )

    labeled_pids = sorted(
        pid for pid in expected
        if anchors.per_crop_embeddings.get(pid) is not None
        and anchors.per_crop_embeddings[pid].shape[0] >= min_crops
    )

    if not labeled_pids:
        # No player has crops — nothing to validate, nothing to override.
        issues.append(
            ValidationIssue(
                code="no_crops",
                player_id=None,
                message="Add at least one reference crop for any player.",
            )
        )
        return ValidationResult(ok=False, issues=issues)

    # In strict mode, any missing_crops is fatal.
    if not partial_ok and any(i.code == "missing_crops" for i in issues):
        return ValidationResult(ok=False, issues=issues)

    # --- Check 2: intra-player diversity -----------------------------------
    for pid in labeled_pids:
        feats = anchors.per_crop_embeddings.get(pid)
        if feats is None or feats.shape[0] < 2:
            continue
        sims = feats @ feats.T
        n = feats.shape[0]
        mask = ~np.eye(n, dtype=bool)
        mean_pair_sim = float(sims[mask].mean())
        spread = 1.0 - mean_pair_sim
        if spread < min_intra_spread:
            issues.append(
                ValidationIssue(
                    code="low_diversity",
                    player_id=pid,
                    message=(
                        f"Player {pid}'s crops all look nearly identical. Add a "
                        f"crop from a different rally or body angle."
                    ),
                )
            )

    # --- Check 3: inter-player separability --------------------------------
    # Only checks pairs among labeled players. Single-player labeling skips
    # this pairwise check because there's nothing to separate against.
    pids = labeled_pids
    if len(pids) >= 2:
        proto_mat = anchors.prototype_matrix
        pair_sims = proto_mat @ proto_mat.T
        for i, pid_a in enumerate(pids):
            for j in range(i + 1, len(pids)):
                pid_b = pids[j]
                distance = 1.0 - float(pair_sims[i, j])
                if distance < min_inter_distance:
                    issues.append(
                        ValidationIssue(
                            code="players_look_alike",
                            player_id=pid_a,
                            message=(
                                f"Player {pid_a} and Player {pid_b}'s crops are "
                                f"too visually similar to tell apart. Add a more "
                                f"distinctive crop for one of them "
                                f"(distance={distance:.3f} < {min_inter_distance:.3f})."
                            ),
                        )
                    )

    # --- Check 4: per-crop misassignment -----------------------------------
    for pid in pids:
        feats = anchors.per_crop_embeddings.get(pid)
        if feats is None or feats.shape[0] == 0:
            continue
        own_proto = anchors.prototypes[pid]
        own_sims = feats @ own_proto
        for crop_idx in range(feats.shape[0]):
            other_pids = [other for other in pids if other != pid]
            if not other_pids:
                continue
            other_sims = [
                float(feats[crop_idx] @ anchors.prototypes[other])
                for other in other_pids
            ]
            best_other = max(other_sims)
            own = float(own_sims[crop_idx])
            # Crop is likely misassigned if another prototype is notably
            # closer than its own.
            if best_other > own * misassign_factor and best_other - own > 0.05:
                best_other_pid = other_pids[int(np.argmax(other_sims))]
                issues.append(
                    ValidationIssue(
                        code="crop_likely_misassigned",
                        player_id=pid,
                        message=(
                            f"Crop #{crop_idx + 1} assigned to Player {pid} looks "
                            f"more like Player {best_other_pid}. Re-check or "
                            f"remove it."
                        ),
                    )
                )

    # Blocking set: quality problems with the crops themselves.
    # `missing_crops_optional` and `few_crops` are informational in partial
    # mode — the matcher simply won't override those players' tracks.
    blocking_codes = {
        "missing_crops",
        "no_crops",
        "low_diversity",
        "players_look_alike",
        "crop_likely_misassigned",
    }
    ok = not any(i.code in blocking_codes for i in issues)
    return ValidationResult(ok=ok, issues=issues)
