"""Factor weights for the Joint Attribution PGM.

Single source of truth for tuning. Hand-tuned starting values; calibrated
on the 22-rally fresh-GT panel via
`analysis/scripts/calibrate_joint_attribution_weights.py`.

Spec: docs/superpowers/specs/2026-05-12-joint-attribution-pgm-design.md
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FactorWeights:
    """Weights for unary (evidence) and pairwise/higher-order (rule) factors.

    All weights are non-negative and applied as log-likelihood multipliers.
    Higher weight = stronger factor influence on the joint MAP.
    """

    # Unary (per-contact evidence)
    # Calibrated 2026-05-12 via coordinate ascent on the 22-rally fresh-GT panel.
    # See reports/joint_attribution_calibration_2026_05_12.json for trajectory.
    w_proximity: float = 1.0       # playerCandidates rank (was 2.0)
    w_dist: float = 0.5            # playerCandidates distance (per-player) (was 1.0)
    w_dist_team: float = 0.5       # absent-state penalty proportional to team's nearest tracked player
    w_visual: float = 1.5          # cross-rally PID profile cosine similarity
    w_pose: float = 1.5            # pose model P(touching)
    w_prior: float = 0.5           # action_classifier initial PID as soft prior
    w_action: float = 1.0          # action-type prior for absent states

    # Pairwise (rule)
    w_back_to_back: float = 3.0    # penalty for same-player consecutive contacts (non-absent)
    w_alternation: float = 3.0     # penalty for same-team consecutive across a net crossing
    w_team_consistency: float = 3.0  # penalty for cross-team consecutive without a net crossing (was 2.0)
    w_absent_pair: float = 1.125   # penalty for two consecutive absent-* states (was 1.5)

    # Higher-order (rule)
    w_3_contact: float = 4.0       # penalty per extra contact beyond 3 same-team same-side
    w_serve_first: float = 3.0     # penalty if first contact's team != serving_team


DEFAULT_WEIGHTS = FactorWeights()
