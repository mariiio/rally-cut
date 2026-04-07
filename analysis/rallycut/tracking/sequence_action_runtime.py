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


# Dig-preserving guard threshold for apply_sequence_override.
#
# When the GBM action classifier predicts `dig` but MS-TCN++'s argmax wants
# to overwrite it with `set`, refuse the override unless MS-TCN++ is at least
# DIG_GUARD_RATIO times more confident in `set` than in `dig`. The 2026-04-07
# contact-classifier audit confusion matrix found MS-TCN++ over-confidently
# rewrites 9 GT-dig contacts as `set` (a low set and a defensive dig produce
# nearly identical trajectory features), while removing the override entirely
# costs +13.8pp on block F1. The guard targets only the dig→set leak and
# leaves every other class transition untouched.
#
# Chosen value: 2.5 — see sweep results in sequence_action_classifier.md
# (2026-04-07). Sweep over {1.0, 1.5, 2.0, 2.5, 3.0, 4.0} chose 2.5 as the
# smallest τ that maximised action_accuracy on the 339-rally dashboard while
# meeting the dig F1 ≥ baseline + 0.5pp hard constraint. τ ∈ {2.5, 3.0, 4.0}
# all produce identical metrics, so 2.5 is the minimal-intervention choice.
# Hardcoded by intent: this is a production parameter, not a runtime knob.
DIG_GUARD_RATIO: float = 2.5


def apply_sequence_override(
    rally_actions: Any,
    sequence_probs: np.ndarray,
    dig_guard_ratio: float | None = None,
) -> None:
    """Override non-serve action types with MS-TCN++ argmax predictions.

    Mutates `rally_actions.actions` in place. Serve is exempt — structural
    rally constraints (first action, baseline position, arc crossing) are
    stronger than per-frame model predictions. Synthetic actions are also
    skipped because they have no ground frame to look up in the probs array.

    Dig-preserving guard: when the existing GBM prediction is `dig` and
    MS-TCN++'s argmax would overwrite it with `set`, the guard refuses the
    overwrite unless `seq_probs[set] >= dig_guard_ratio * seq_probs[dig]`.
    This recovers GBM digs that MS-TCN++ misclassifies as low sets without
    affecting any other class transition.
    """
    from rallycut.actions.trajectory_features import ACTION_TYPES
    from rallycut.tracking.action_classifier import ActionType

    # Read the global at call time (not via default arg) so test/sweep
    # harnesses can monkey-patch DIG_GUARD_RATIO between runs.
    ratio = dig_guard_ratio if dig_guard_ratio is not None else DIG_GUARD_RATIO

    # Index of `set` and `dig` inside sequence_probs[1:, :] (NUM_CLASSES − 1
    # offset because index 0 is background).
    set_idx = ACTION_TYPES.index("set")
    dig_idx = ACTION_TYPES.index("dig")

    for action in rally_actions.actions:
        if action.is_synthetic or action.action_type == ActionType.SERVE:
            continue
        frame = action.frame
        if not (0 <= frame < sequence_probs.shape[1]):
            continue
        per_frame = sequence_probs[1:, frame]
        cls = int(np.argmax(per_frame))
        new_type = ActionType(ACTION_TYPES[cls])

        # Dig guard: GBM said dig, MS-TCN++ wants set — only override if
        # MS-TCN++ is much more confident in set than in dig.
        if (
            action.action_type == ActionType.DIG
            and new_type == ActionType.SET
        ):
            seq_set = float(per_frame[set_idx])
            seq_dig = float(per_frame[dig_idx])
            if seq_set < ratio * seq_dig:
                continue

        action.action_type = new_type
