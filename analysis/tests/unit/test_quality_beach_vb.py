from rallycut.quality.beach_vb_classifier import classify_is_beach_vb, BeachVBProbabilities
from rallycut.quality.types import Tier


def test_high_beach_vb_probability_passes():
    probs = BeachVBProbabilities(beach_vb=0.93, indoor_vb=0.04, other=0.03)
    result = classify_is_beach_vb([probs] * 5)
    assert result.issues == []


def test_very_low_beach_vb_probability_blocks():
    probs = BeachVBProbabilities(beach_vb=0.05, indoor_vb=0.10, other=0.85)
    result = classify_is_beach_vb([probs] * 5)
    issue = next(i for i in result.issues if i.id == "wrong_angle_or_not_volleyball")
    assert issue.tier == Tier.BLOCK


def test_ambiguous_does_not_block():
    # We only block on confident-not-beach-VB. Ambiguous stays silent.
    probs = BeachVBProbabilities(beach_vb=0.4, indoor_vb=0.3, other=0.3)
    result = classify_is_beach_vb([probs] * 5)
    assert result.issues == []
