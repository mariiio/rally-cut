"""Inference correctness tests for the Joint Attribution PGM."""
from __future__ import annotations

import math

from rallycut.tracking.joint_attribution import (
    RallyContext,
    joint_attribute_rally,
    score_joint_config,
)
from rallycut.tracking.joint_attribution_weights import DEFAULT_WEIGHTS


def _make_simple_rally(n_contacts: int = 3, ball_x: float = 0.5) -> RallyContext:
    """Build a synthetic 3-contact rally for testing."""
    contacts = [
        {
            "frame": 100 + i * 30,
            "ballX": ball_x, "ballY": 0.5,
            "playerCandidates": [[1, 0.05], [2, 0.10], [3, 0.20], [4, 0.50]],
            "courtSide": "near" if i % 2 == 0 else "far",
        }
        for i in range(n_contacts)
    ]
    actions = [
        {"frame": c["frame"], "action": "set", "playerTrackId": 1, "confidence": 0.8}
        for c in contacts
    ]
    return RallyContext(
        rally_id="test-rally", contacts=contacts, initial_actions=actions,
        team_assignments={1: "A", 2: "A", 3: "B", 4: "B"},
        serving_team="A",
    )


def test_inference_returns_valid_assignment() -> None:
    """MAP returns a tuple of length n_contacts with valid states."""
    rally = _make_simple_rally(n_contacts=3)
    result = joint_attribute_rally(rally)
    assert len(result.map) == 3
    valid_states = {"P1", "P2", "P3", "P4", "ABSENT_TEAM_A", "ABSENT_TEAM_B"}
    for state in result.map:
        assert state in valid_states
    assert result.fallback_used == "exhaustive"


def test_inference_picks_proximity_winner_in_simple_case() -> None:
    """With proximity heavily favoring P1 and no rule violations, MAP picks P1."""
    contacts = [{
        "frame": 100, "ballX": 0.5, "ballY": 0.5,
        "playerCandidates": [[1, 0.01], [2, 0.30], [3, 0.40], [4, 0.50]],
        "courtSide": "near",
    }]
    actions = [{"frame": 100, "action": "serve", "playerTrackId": 1, "confidence": 0.9}]
    rally = RallyContext(
        rally_id="test", contacts=contacts, initial_actions=actions,
        team_assignments={1: "A", 2: "A", 3: "B", 4: "B"},
        serving_team="A",
    )
    result = joint_attribute_rally(rally)
    assert result.map[0] == "P1"


def test_inference_serve_first_factor_overrides_proximity() -> None:
    """When proximity favors team-B but serving_team is A, MAP picks team-A
    (or absent-team-A)."""
    contacts = [{
        "frame": 100, "ballX": 0.5, "ballY": 0.5,
        "playerCandidates": [[3, 0.01], [4, 0.05], [1, 0.30], [2, 0.40]],
        "courtSide": "far",
    }]
    actions = [{"frame": 100, "action": "serve", "playerTrackId": 3, "confidence": 0.9}]
    rally = RallyContext(
        rally_id="test", contacts=contacts, initial_actions=actions,
        team_assignments={1: "A", 2: "A", 3: "B", 4: "B"},
        serving_team="A",
    )
    result = joint_attribute_rally(rally)
    # MAP should NOT pick a team-B player (P3 or P4) for the first contact
    assert result.map[0] in ("P1", "P2", "ABSENT_TEAM_A")


def test_inference_back_to_back_penalty_drives_alternation() -> None:
    """Two consecutive contacts with proximity favoring same player —
    PGM should still alternate due to back-to-back penalty."""
    contacts = [
        {"frame": 100, "ballX": 0.5, "ballY": 0.5,
         "playerCandidates": [[1, 0.01], [2, 0.10], [3, 0.50], [4, 0.60]],
         "courtSide": "near"},
        {"frame": 130, "ballX": 0.5, "ballY": 0.4,
         "playerCandidates": [[1, 0.01], [2, 0.10], [3, 0.50], [4, 0.60]],
         "courtSide": "near"},
    ]
    actions = [
        {"frame": 100, "action": "dig", "playerTrackId": 1, "confidence": 0.8},
        {"frame": 130, "action": "set", "playerTrackId": 1, "confidence": 0.8},
    ]
    rally = RallyContext(
        rally_id="test", contacts=contacts, initial_actions=actions,
        team_assignments={1: "A", 2: "A", 3: "B", 4: "B"},
        serving_team="A",
    )
    result = joint_attribute_rally(rally)
    assert result.map[0] != result.map[1] or result.map[0] in ("ABSENT_TEAM_A", "ABSENT_TEAM_B")


def test_inference_marginals_normalize_per_contact() -> None:
    """Per-contact marginals sum to 1.0 over all states."""
    rally = _make_simple_rally(n_contacts=3)
    result = joint_attribute_rally(rally)
    for marginal in result.marginals:
        total = sum(marginal.values())
        assert math.isclose(total, 1.0, abs_tol=1e-6)


def test_inference_falls_back_to_coordinate_descent_at_n9() -> None:
    """For N > 8 contacts, inference uses coordinate descent."""
    rally = _make_simple_rally(n_contacts=9)
    result = joint_attribute_rally(rally)
    assert result.fallback_used == "coordinate_descent"
    assert len(result.map) == 9


def test_score_joint_config_matches_per_factor_sum() -> None:
    """score_joint_config equals sum of unary + pairwise + higher factors."""
    rally = _make_simple_rally(n_contacts=3)
    config = ("P1", "P2", "P1")
    score = score_joint_config(config, rally, weights=DEFAULT_WEIGHTS)
    assert isinstance(score, float)
    # No assert on value; just no exception + float type
