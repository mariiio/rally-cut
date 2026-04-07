"""Production runtime for the MS-TCN++ sequence action override.

The action classification pipeline produces per-contact actions via heuristics
+ a GBM classifier. When MS-TCN++ weights are available, we override the
non-serve action types with the model's per-frame argmax (serve is exempt
because structural rally constraints are stronger than per-frame predictions).

This module is the single source of truth for that runtime path. Both
`track-players --actions` and `analyze classify-actions` import from here so
the two CLI commands cannot drift apart on action classification behavior.

Public API:
    get_sequence_probs(...)        — compute (NUM_CLASSES, T) per-frame probs
    apply_sequence_override(...)   — mutate rally_actions in place

The model is loaded lazily and cached per process. If the weights file is
missing, a one-time WARNING is logged and `get_sequence_probs` returns None,
which is a no-op for callers — the heuristic + GBM result stands.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import torch

    from rallycut.temporal.ms_tcn.model import MSTCN
    from rallycut.tracking.ball_tracker import BallPosition
    from rallycut.tracking.player_tracker import PlayerPosition

logger = logging.getLogger(__name__)

_sequence_model_cache: dict[str, Any] = {}


def _load_sequence_model() -> tuple[MSTCN, torch.device] | None:
    """Lazy-load MS-TCN++ production model. Returns (model, device) or None.

    Caches the result (including the missing-weights case) so we don't re-check
    disk on every rally. Logs a one-time WARNING when weights are missing —
    without this, the action pipeline silently degrades to non-MS-TCN++
    classification, hiding production drift.
    """
    if "model" in _sequence_model_cache:
        cached = _sequence_model_cache["model"]
        return cached if cached is not None else None

    import torch

    # parents[3] = analysis/ root (sequence_action_runtime.py is in
    # rallycut/tracking/, so parents are: tracking → rallycut → analysis).
    local_path = (
        Path(__file__).resolve().parents[2]
        / "weights" / "sequence_action" / "ms_tcn_production.pt"
    )
    modal_path = Path("/app/weights/sequence_action/ms_tcn_production.pt")
    weights_path = local_path if local_path.exists() else modal_path
    if not weights_path.exists():
        logger.warning(
            "MS-TCN++ weights not found at %s or %s. Action classification "
            "will skip the sequence-model override and use heuristics + GBM "
            "only. Run `rallycut train sequence-action --modal` to retrain, "
            "or `rallycut train pull-weights` to fetch from S3.",
            local_path, modal_path,
        )
        _sequence_model_cache["model"] = None
        return None

    from rallycut.temporal.ms_tcn.model import MSTCN, MSTCNConfig

    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    config = MSTCNConfig(**checkpoint["config"])
    model = MSTCN(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    _sequence_model_cache["model"] = (model, device)
    return (model, device)


def get_sequence_probs(
    ball_positions: list[BallPosition],
    player_positions: list[PlayerPosition],
    court_split_y: float | None,
    frame_count: int,
    team_assignments: dict[int, int] | None,
    calibrator: Any = None,
) -> np.ndarray | None:
    """Compute MS-TCN++ per-frame action probabilities.

    Returns (NUM_CLASSES, T) array or None if the model is unavailable or
    the rally is too short. None is a valid no-op for callers — apply
    `apply_sequence_override` only when this returns a non-None array.

    `calibrator` is optional. When provided and calibrated, the homography
    matrix is used to enrich trajectory features with court-space signal.
    """
    loaded = _load_sequence_model()
    if loaded is None:
        return None
    if not frame_count or frame_count < 10:
        return None

    import torch

    from rallycut.actions.trajectory_features import extract_trajectory_features

    homography = None
    if calibrator is not None:
        h = getattr(calibrator, "homography", None)
        if h is not None:
            homography = h.homography

    features = extract_trajectory_features(
        ball_positions, player_positions, court_split_y, frame_count,
        team_assignments=team_assignments,
        homography=homography,
    )

    model, device = loaded
    feat = torch.from_numpy(features).float().unsqueeze(0).transpose(1, 2).to(device)
    mask = torch.ones(1, 1, frame_count, device=device)
    with torch.no_grad():
        logits = model(feat, mask)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    return probs


def apply_sequence_override(
    rally_actions: Any,
    sequence_probs: np.ndarray,
) -> None:
    """Override non-serve action types with MS-TCN++ argmax predictions.

    Mutates `rally_actions.actions` in place. Serve is exempt — structural
    rally constraints (first action, baseline position, arc crossing) are
    stronger than per-frame model predictions. Synthetic actions are also
    skipped because they have no ground frame to look up in the probs array.
    """
    from rallycut.actions.trajectory_features import ACTION_TYPES
    from rallycut.tracking.action_classifier import ActionType

    for action in rally_actions.actions:
        if action.is_synthetic or action.action_type == ActionType.SERVE:
            continue
        frame = action.frame
        if 0 <= frame < sequence_probs.shape[1]:
            cls = int(np.argmax(sequence_probs[1:, frame]))
            action.action_type = ActionType(ACTION_TYPES[cls])
