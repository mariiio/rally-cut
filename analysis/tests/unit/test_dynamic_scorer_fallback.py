"""Tests the chain-context fallback inside _apply_dynamic_scorer_attribution.

When the scorer's chain-aware pick disagrees with the no-chain
(expected_team=None) pick, prefer the higher-confidence of the two.
Validates the Sub-lever 1 guardrail for Branch A (chain-context confound).
"""
from __future__ import annotations

import pytest

from rallycut.tracking.action_classifier import (
    _scorer_chain_aware_fallback_pick,
)


def test_prefers_no_chain_when_higher_confidence():
    pick, prob = _scorer_chain_aware_fallback_pick(
        chain_pick_tid=3, chain_pick_prob=0.45,
        no_chain_pick_tid=2, no_chain_pick_prob=0.62,
    )
    assert pick == 2
    assert prob == pytest.approx(0.62)


def test_keeps_chain_pick_when_higher_confidence():
    pick, prob = _scorer_chain_aware_fallback_pick(
        chain_pick_tid=3, chain_pick_prob=0.72,
        no_chain_pick_tid=2, no_chain_pick_prob=0.55,
    )
    assert pick == 3
    assert prob == pytest.approx(0.72)


def test_keeps_chain_pick_when_picks_agree():
    pick, prob = _scorer_chain_aware_fallback_pick(
        chain_pick_tid=3, chain_pick_prob=0.55,
        no_chain_pick_tid=3, no_chain_pick_prob=0.60,
    )
    assert pick == 3
    assert prob == pytest.approx(0.60)
