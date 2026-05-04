"""Unit tests for the swap-acceptance gate in `identity_repair._accept_swap`.

The gate has two pathways:
  1. Strong total: total improvement >= full threshold.
  2. Substantial both halves: each individual shift >= INDIVIDUAL_MIN_SHIFT
     AND total >= half threshold.

Both pathways require both individual shifts to be positive (asymmetric
one-sided wins are rejected as artifacts).

Motivating case (rally 19 of dd042609): own=0.031, partner=0.039,
total=0.070, threshold=0.120 (same-team). Path 1 fails (0.070 < 0.120),
but path 2 passes (both >= 0.020 AND 0.070 >= 0.060).
"""

from __future__ import annotations

from rallycut.tracking.identity_repair import (
    INDIVIDUAL_MIN_SHIFT,
    MIN_IMPROVEMENT_CROSS_TEAM,
    MIN_IMPROVEMENT_SAME_TEAM,
    _accept_swap,
)


def test_path1_strong_total_accepts() -> None:
    """Total >= full threshold accepts even if one half barely contributes."""
    # Cross-team: full threshold = 0.06
    assert _accept_swap(
        shift_improvement=0.080,
        partner_improvement=0.005,
        threshold=MIN_IMPROVEMENT_CROSS_TEAM,
    ) is True


def test_path2_substantial_both_halves_accepts_below_full_threshold() -> None:
    """Rally 19 of dd042609: total below full but both halves substantial."""
    accepted = _accept_swap(
        shift_improvement=0.031,
        partner_improvement=0.039,
        threshold=MIN_IMPROVEMENT_SAME_TEAM,  # 0.12
    )
    assert accepted is True


def test_one_sided_large_shift_rejected_when_partner_marginal() -> None:
    """3814294e (t3): own=0.051, partner=0.002. Partner < INDIVIDUAL_MIN_SHIFT
    so path 2 fails; total=0.053 < threshold=0.12 so path 1 fails too."""
    assert _accept_swap(
        shift_improvement=0.051,
        partner_improvement=0.002,
        threshold=MIN_IMPROVEMENT_SAME_TEAM,
    ) is False


def test_negative_partner_always_rejected() -> None:
    """When the partner regresses under the swap, reject regardless of own
    shift magnitude. This is the core asymmetric-evidence guard."""
    assert _accept_swap(
        shift_improvement=0.200,
        partner_improvement=-0.050,
        threshold=MIN_IMPROVEMENT_CROSS_TEAM,
    ) is False


def test_negative_own_always_rejected() -> None:
    assert _accept_swap(
        shift_improvement=-0.010,
        partner_improvement=0.080,
        threshold=MIN_IMPROVEMENT_CROSS_TEAM,
    ) is False


def test_zero_partner_rejected_path1() -> None:
    """Strict positive — zero partner shift fails the bidirectional gate."""
    assert _accept_swap(
        shift_improvement=0.150,
        partner_improvement=0.0,
        threshold=MIN_IMPROVEMENT_CROSS_TEAM,
    ) is False


def test_path2_total_below_half_threshold_rejected() -> None:
    """3814294e (t4): own=0.010, partner=0.022, total=0.032. Both positive but
    own is below INDIVIDUAL_MIN_SHIFT (0.020), so path 2 fails. Total < full
    threshold (0.12), so path 1 fails. Skip."""
    assert _accept_swap(
        shift_improvement=0.010,
        partner_improvement=0.022,
        threshold=MIN_IMPROVEMENT_SAME_TEAM,
    ) is False


def test_path2_both_at_individual_minimum_accepts() -> None:
    """Boundary case: each half at exactly INDIVIDUAL_MIN_SHIFT, total
    at exactly half threshold."""
    assert _accept_swap(
        shift_improvement=INDIVIDUAL_MIN_SHIFT,  # 0.020
        partner_improvement=0.040,  # total = 0.060 = same-team threshold * 0.5
        threshold=MIN_IMPROVEMENT_SAME_TEAM,
    ) is True


def test_path2_just_below_half_threshold_rejected() -> None:
    """Both halves substantial but total just below half threshold — reject."""
    assert _accept_swap(
        shift_improvement=0.025,
        partner_improvement=0.030,  # total = 0.055 < 0.060
        threshold=MIN_IMPROVEMENT_SAME_TEAM,
    ) is False


def test_cross_team_path2_higher_bar() -> None:
    """Cross-team threshold is 0.06; path 2 half threshold is 0.030. Same-shape
    case fires for cross-team too if total is above half."""
    assert _accept_swap(
        shift_improvement=INDIVIDUAL_MIN_SHIFT,
        partner_improvement=0.020,  # total = 0.040 > 0.030 = cross-team * 0.5
        threshold=MIN_IMPROVEMENT_CROSS_TEAM,
    ) is True
