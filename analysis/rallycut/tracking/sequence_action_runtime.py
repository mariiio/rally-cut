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


# Relative-confidence gate for apply_sequence_override.
#
# Skip the MS-TCN++ override whenever its argmax probability is not at least
# `OVERRIDE_RELATIVE_CONF_K` times the GBM's top-1 confidence at the same
# contact. Intuition: if MS-TCN++ is only marginally more certain than the
# GBM, the GBM's decision (trained on labelled contacts with pose and
# trajectory features) usually wins. The 2026-04-14 error-origin diagnostic
# (`action_error_origin_2026-04-14.md`) found that 58% of action-type
# errors (56/96 on the clean pool) were override regressions: MS-TCN++
# demoted correct GBM predictions with median peak-prob 0.872 while the
# GBM had median confidence 0.763. This gate closes that gap without
# touching any guard-free transition.
#
# Chosen value: 1.2 — see `override_guard_sweep_2026-04-14.json`. The
# cross-product sweep on the 1668-contact probe set found K=1.2 combined
# with ATTACK_PRESERVE_RATIO=2.5 at the top of the Pareto frontier
# (+1.68pp action_accuracy, +4.72pp dig F1, +2.43pp attack F1, 0 regression
# on receive F1). Lower K (1.0) let marginal overrides through; higher K
# (≥1.5 alone) started blocking correct receive overrides (−3.47pp
# receive F1 at K=1.5).
OVERRIDE_RELATIVE_CONF_K: float = 1.2


# Attack-preserving guard ratio for apply_sequence_override.
#
# When the GBM action classifier predicts `attack` but MS-TCN++'s argmax
# wants to overwrite it with `set` or `dig`, refuse the override unless
# MS-TCN++ is at least ATTACK_PRESERVE_RATIO times more confident in the
# argmax class than in `attack`. Mirrors the DIG_GUARD_RATIO pattern.
#
# Attacks (especially tips, rolls, and soft hits) look trajectory-similar
# to sets and digs, and MS-TCN++ is particularly aggressive about
# demoting them. The 2026-04-14 diagnostic recorded 29 of 56 bucket-A
# errors as `attack → {set, dig}` demotions. This guard recovers the
# majority of those while leaving every non-attack origin untouched.
#
# Chosen value: 2.5 — see `override_guard_sweep_2026-04-14.json`. The
# cross-product sweep picked 2.5 as the Pareto-optimal pair-value when
# used alongside `OVERRIDE_RELATIVE_CONF_K=1.2`. Identical value to
# `DIG_GUARD_RATIO` by coincidence: both pairs share the same
# "trajectory-similar contact, ambiguous class" failure mode, and the
# sweep grid {1.5, 2.0, 2.5, 3.0} converged on 2.5 for each.
ATTACK_PRESERVE_RATIO: float = 2.5


def apply_sequence_override(
    rally_actions: Any,
    sequence_probs: np.ndarray,
    dig_guard_ratio: float | None = None,
    override_relative_conf_k: float | None = None,
    attack_preserve_ratio: float | None = None,
) -> None:
    """Override non-serve action types with MS-TCN++ argmax predictions.

    Mutates `rally_actions.actions` in place. Serve is exempt — structural
    rally constraints (first action, baseline position, arc crossing) are
    stronger than per-frame model predictions. Synthetic actions are also
    skipped because they have no ground frame to look up in the probs array.

    Three guards constrain the override. All three read module-level
    constants at call time so sweep harnesses can monkey-patch them:

    1. Relative-confidence gate (``OVERRIDE_RELATIVE_CONF_K``) — skip the
       override whenever MS-TCN++'s argmax probability is not at least K
       times the GBM's top-1 confidence (``action.confidence``). Catches
       the bulk of the 2026-04-14 override regressions where MS-TCN++
       was only marginally more certain than a correct GBM call.
    2. Attack-preserving guard (``ATTACK_PRESERVE_RATIO``) — when the GBM
       said ``attack`` and MS-TCN++ wants ``set``/``dig``, refuse unless
       MS-TCN++ is that ratio more confident in the argmax than in
       ``attack``. Mirrors the dig-guard pattern for attack demotions.
    3. Dig-preserving guard (``DIG_GUARD_RATIO``) — existing 2026-04-07
       guard for GBM=dig / MS-TCN++=set. Held fixed by design.
    """
    from rallycut.actions.trajectory_features import ACTION_TYPES
    from rallycut.tracking.action_classifier import ActionType

    # Read globals at call time so test/sweep harnesses can monkey-patch.
    ratio_dig = (
        dig_guard_ratio if dig_guard_ratio is not None else DIG_GUARD_RATIO
    )
    k_rel = (
        override_relative_conf_k
        if override_relative_conf_k is not None
        else OVERRIDE_RELATIVE_CONF_K
    )
    ratio_attack = (
        attack_preserve_ratio
        if attack_preserve_ratio is not None
        else ATTACK_PRESERVE_RATIO
    )

    # Index of `set`, `dig`, `attack` inside sequence_probs[1:, :]
    # (NUM_CLASSES − 1 offset because index 0 is background).
    set_idx = ACTION_TYPES.index("set")
    dig_idx = ACTION_TYPES.index("dig")
    attack_idx = ACTION_TYPES.index("attack")

    for action in rally_actions.actions:
        if action.is_synthetic or action.action_type == ActionType.SERVE:
            continue
        frame = action.frame
        if not (0 <= frame < sequence_probs.shape[1]):
            continue
        per_frame = sequence_probs[1:, frame]
        cls = int(np.argmax(per_frame))
        new_type = ActionType(ACTION_TYPES[cls])
        argmax_prob = float(per_frame[cls])

        # Serve is heuristic-only: classify_rally picks exactly one
        # serve per rally via _find_serve_index. The override must
        # never manufacture additional serves (see 2026-04-14 audit —
        # the double-serve symptom came from this leak).
        if new_type == ActionType.SERVE:
            continue

        # Relative-confidence gate: MS-TCN++ must beat the GBM's top-1
        # confidence by `k_rel`. Guards against the 2026-04-14
        # override-regression mode where MS-TCN++ wins the argmax by a
        # narrow margin over a high-confidence, correct GBM prediction.
        gbm_conf = float(action.confidence)
        if argmax_prob < k_rel * gbm_conf:
            continue

        # Attack-preserving guard: GBM said attack, MS-TCN++ wants set or
        # dig — only override if MS-TCN++ is much more confident in the
        # argmax than in attack.
        if (
            action.action_type == ActionType.ATTACK
            and new_type in (ActionType.SET, ActionType.DIG)
        ):
            seq_attack = float(per_frame[attack_idx])
            if argmax_prob < ratio_attack * seq_attack:
                continue

        # Dig guard: GBM said dig, MS-TCN++ wants set — only override if
        # MS-TCN++ is much more confident in set than in dig.
        if (
            action.action_type == ActionType.DIG
            and new_type == ActionType.SET
        ):
            seq_set = float(per_frame[set_idx])
            seq_dig = float(per_frame[dig_idx])
            if seq_set < ratio_dig * seq_dig:
                continue

        action.action_type = new_type


# --------------------------------------------------------------------------- #
# Sequence-based contact recovery (shipped 2026-04-07).                       #
#                                                                             #
# Empirical basis (see memory/fn_sequence_signal_2026_04.md): on the 339-     #
# rally canonical pool, MS-TCN++ puts a non-background peak >= 0.80 within    #
# +-5 frames of 99.4% of GT contacts the trajectory+pose classifier           #
# confidently rejects. Per-class peak matches GT 93.4% of the time. Control   #
# frame trigger rate at the same threshold is 17.8% (discriminative ratio     #
# 5.6x). The two signals are complementary: trajectory features see single-  #
# frame physics, the sequence model sees temporal action patterns.            #
#                                                                             #
# Integration: `contact_detector.detect_contacts()` uses the two constants    #
# below to rescue trajectory candidates the GBM confidently rejects. A       #
# trajectory candidate at frame f is rescued iff:                             #
#   1. max(sequence_probs[1:, f-5:f+6]) >= SEQ_RECOVERY_TAU   (sequence       #
#      model endorses the frame as non-background action)                     #
#   2. GBM(features_at_f) >= SEQ_RECOVERY_CLF_FLOOR           (classifier     #
#      still gives it a non-trivial score)                                    #
#                                                                             #
# This is a two-signal agreement gate — not a threshold hack. It is the       #
# principled way to combine complementary detectors with asymmetric FP rates: #
# relax each detector's single-source confidence requirement only when an     #
# independent signal already endorses the frame. No new candidates are        #
# injected — only existing trajectory candidates are rescued, so dedup,       #
# player attribution, and pose features remain unchanged.                     #
# --------------------------------------------------------------------------- #

SEQ_RECOVERY_TAU: float = 0.80
# Two-signal gate: trajectory candidates the GBM rejected are rescued only
# if the GBM still gave them a non-trivial score >= this floor. Read at call
# time by `contact_detector.detect_contacts` so sweep harnesses can
# monkey-patch it between runs without plumbing arguments.
SEQ_RECOVERY_CLF_FLOOR: float = 0.20

# --------------------------------------------------------------------------- #
# Arm B — multi-generator rescue (Pattern A, 2026-04-17).                     #
#                                                                             #
# Arm A (the two constants above) rescues candidates with conf >= 0.20 and    #
# sequence endorsement. Corpus audit after the Phase-3 sequence wiring found  #
# 71 residual FN_contacts with conf in [0.05, 0.20) where >=3 trajectory      #
# generators fired (velocity_peak / inflection / reversal / deceleration /    #
# parabolic / net_crossing) and a player sat within 0.15 of the ball — the    #
# classifier was skeptical but three independent trajectory detectors agreed  #
# and a real player was present. Four visually-confirmed rescues are in this  #
# band.                                                                       #
#                                                                             #
# Arm B rescues a candidate iff:                                              #
#   1. max(sequence_probs[1:, f-5:f+6]) >= SEQ_RECOVERY_TAU                   #
#   2. GBM(features_at_f) >= SEQ_RECOVERY_CLF_FLOOR_MULTIGEN                  #
#   3. >= SEQ_RECOVERY_MIN_GENERATORS trajectory generators claimed the frame #
#   4. nearest player <= SEQ_RECOVERY_MAX_PLAYER_DIST                         #
#                                                                             #
# The four-signal conjunction (sequence + classifier + multi-generator agree- #
# ment + spatial proximity) widens the two-signal Arm A without lowering the  #
# single-signal reliance any further — each additional required condition    #
# offsets the lower classifier floor. Defaults below are placeholders to be   #
# locked in by `scripts/sweep_rescue_gate.py` against the 364-rally corpus.   #
# --------------------------------------------------------------------------- #

# Defaults are DORMANT. The 364-rally sweep at 2026-04-17 (see
# reports/pattern_a_rescue_sweep_2026_04_17.md) measured every grid cell
# (FLOOR_MULTIGEN × MIN_GENERATORS); no cell cleared the precision budget
# (Δextra_pred / ΔTP ≤ 0.5). The most conservative cell (0.15, 3) added
# +8 TP at the cost of +19 extra_pred and 4 per-rally regressions —
# net-neutral on action_accuracy but +0.9pp on contact_recall. Setting
# MIN_GENERATORS=999 (unreachable) keeps Arm B disabled by default; the
# gate code, tests, and sweep harness stay in place for future retuning
# if a precision-improving follow-up lands. Matches the dormant-ship
# precedent established by `LEARNED_MERGE_VETO_COS` (merge_veto.py) and
# Session-4 of the within-team-ReID workstream.
SEQ_RECOVERY_CLF_FLOOR_MULTIGEN: float = 0.08
SEQ_RECOVERY_MIN_GENERATORS: int = 999
SEQ_RECOVERY_MAX_PLAYER_DIST: float = 0.15


# --------------------------------------------------------------------------- #
# Rally-start serve anchor (Pattern C, 2026-04-17).                           #
#                                                                             #
# Late ball-track start: when the WASB ball tracker does not lock onto the    #
# ball until after the serve contact (ball_gap >= 4 frames over the first    #
# 60-90 frames of a rally), no trajectory generator fires near the serve and #
# the rescue gate has nothing to rescue. Corpus audit: 11 early-rally serve  #
# FNs have `ball_present=False` AND `ball_gap_frames >= 4` AND no candidate  #
# within reasonable distance of GT.                                           #
#                                                                             #
# Fix: when no validated contact exists in the first                          #
# `SERVE_ANCHOR_MAX_FRAME` frames AND MS-TCN++'s serve-class probability     #
# peaks above `SERVE_ANCHOR_TAU` in that window, inject a synthetic          #
# serve contact at the peak frame attributed to the nearest player. This is  #
# a rally-invariant + sequence-class conjunction — MS-TCN++ places the      #
# serve temporally, the rally-invariant ("a rally begins with a serve")      #
# justifies the single-point injection. No arbitrary back-extrapolation.    #
# --------------------------------------------------------------------------- #

# Defaults are DORMANT. The 364-rally validation at 2026-04-17 (see
# reports/pattern_c_serve_anchor_2026_04_17.md) measured anchor on vs off:
# ΔTP=-3, ΔFN=-1, Δwrong_action=+6, Δwrong_player=-2, Δextra_pred=+18, with
# only 3/11 late-track-start targets rescued and 12 per-rally regressions.
# The anchor synthesizes more noise than signal in its current form. Setting
# SERVE_ANCHOR_TAU=1.1 (unreachable) keeps Pattern C disabled by default;
# the helper, constants, tests, and validation harness stay in place for
# future retuning (e.g. combining the sequence peak with a trajectory-based
# ball-backextrapolation step that better discriminates serve-start from
# mid-rally serve-class leakage).
SERVE_ANCHOR_TAU: float = 1.1

# Search window in frames relative to the rally start (~1.5 s at 60 fps).
SERVE_ANCHOR_MAX_FRAME: int = 90
