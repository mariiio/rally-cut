"""Tests for the binary beach-VB classifier.

The classifier returns a single per-frame `beach_vb_prob` (softmax prob of the
beach-volleyball prompt vs. the not-beach-volleyball prompt). A video blocks
when the average across sampled frames falls below BEACH_VB_BLOCK_THRESHOLD.
"""
from rallycut.quality.beach_vb_classifier import (
    BEACH_VB_BLOCK_THRESHOLD,
    classify_is_beach_vb,
)
from rallycut.quality.types import Tier


def test_high_beach_vb_probability_passes():
    # All frames clearly look like beach VB
    probs = [0.95, 0.92, 0.90, 0.88, 0.94]
    result = classify_is_beach_vb(probs)
    assert result.issues == []
    assert result.metrics["avgBeachVbProb"] > 0.85


def test_very_low_beach_vb_probability_blocks():
    # All frames look like non-beach content (indoor, or not volleyball)
    probs = [0.12, 0.18, 0.15, 0.21, 0.19]
    result = classify_is_beach_vb(probs)
    assert any(i.id == "not_beach_volleyball" for i in result.issues)
    block = next(i for i in result.issues if i.id == "not_beach_volleyball")
    assert block.tier == Tier.BLOCK
    assert block.data["avgBeachVbProb"] < BEACH_VB_BLOCK_THRESHOLD


def test_ambiguous_does_not_block():
    # Right at the threshold — must NOT fire (favor false-accept)
    probs = [BEACH_VB_BLOCK_THRESHOLD + 0.01] * 5
    result = classify_is_beach_vb(probs)
    assert result.issues == []


def test_empty_input_is_noop():
    result = classify_is_beach_vb([])
    assert result.issues == []
    assert result.metrics == {}
