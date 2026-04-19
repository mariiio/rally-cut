"""Tests for crop-guided identity resolver.

Stubs out the DINOv2 backbone by constructing ``IdentityAnchors`` directly
with synthetic embeddings. This keeps unit tests fast (no 500MB hub
download) and deterministic.
"""

from __future__ import annotations

import numpy as np

from rallycut.tracking.crop_guided_identity import (
    LOCK_MARGIN_THRESHOLD,
    IdentityAnchors,
    ValidationIssue,
    ValidationResult,
    score_embedding,
    validate_prototypes,
)

# ---------------------------------------------------------------------------
# Synthetic helpers
# ---------------------------------------------------------------------------


def _unit(vec: np.ndarray) -> np.ndarray:
    out: np.ndarray = (vec / np.linalg.norm(vec)).astype(np.float32)
    return out


def _basis_prototype(axis: int, dim: int = 384) -> np.ndarray:
    """Prototype that's ``+1`` on ``axis`` — orthogonal across players."""
    v = np.zeros(dim, dtype=np.float32)
    v[axis] = 1.0
    return v


def _mixed_prototype(axis_a: int, axis_b: int, blend: float, dim: int = 384) -> np.ndarray:
    """Prototype that mixes two axes — lets us tune inter-player distance."""
    v = np.zeros(dim, dtype=np.float32)
    v[axis_a] = 1.0 - blend
    v[axis_b] = blend
    return _unit(v)


def _anchors_for_four_players(dim: int = 384) -> IdentityAnchors:
    """Four orthogonal prototypes with 3 matching crops each."""
    prototypes: dict[int, np.ndarray] = {}
    per_crop: dict[int, np.ndarray] = {}
    for pid in (1, 2, 3, 4):
        proto = _basis_prototype(pid - 1, dim)
        prototypes[pid] = proto
        # Three crops, each slightly perturbed to create non-zero intra spread.
        rng = np.random.default_rng(pid)
        perturbations = rng.normal(0, 0.05, size=(3, dim)).astype(np.float32)
        crops = np.stack([proto] * 3, axis=0) + perturbations
        crops = np.stack([_unit(c) for c in crops], axis=0)
        per_crop[pid] = crops
    return IdentityAnchors(prototypes=prototypes, per_crop_embeddings=per_crop)


# ---------------------------------------------------------------------------
# score_embedding
# ---------------------------------------------------------------------------


class TestScoreEmbedding:
    def test_returns_argmax_player_id(self) -> None:
        anchors = _anchors_for_four_players()
        query = anchors.prototypes[2]
        score = score_embedding(query, anchors)
        assert score.player_id == 2

    def test_confident_score_passes_lock_threshold(self) -> None:
        anchors = _anchors_for_four_players()
        query = anchors.prototypes[1]
        score = score_embedding(query, anchors)
        assert score.margin >= LOCK_MARGIN_THRESHOLD
        assert score.is_confident

    def test_ambiguous_score_fails_lock_threshold(self) -> None:
        # Query equidistant from players 1 and 2 → near-zero margin.
        p1 = _basis_prototype(0)
        p2 = _basis_prototype(1)
        anchors = IdentityAnchors(
            prototypes={1: p1, 2: p2},
            per_crop_embeddings={1: p1[None, :], 2: p2[None, :]},
        )
        query = _unit(0.5 * p1 + 0.5 * p2)
        score = score_embedding(query, anchors)
        assert score.margin < LOCK_MARGIN_THRESHOLD
        assert not score.is_confident

    def test_similarities_include_all_players(self) -> None:
        anchors = _anchors_for_four_players()
        score = score_embedding(anchors.prototypes[3], anchors)
        assert set(score.similarities.keys()) == {1, 2, 3, 4}

    def test_empty_anchors_returns_sentinel(self) -> None:
        empty = IdentityAnchors(prototypes={}, per_crop_embeddings={})
        score = score_embedding(np.ones(384, dtype=np.float32), empty)
        assert score.player_id == -1
        assert score.confidence == 0.0


# ---------------------------------------------------------------------------
# validate_prototypes
# ---------------------------------------------------------------------------


class TestValidatePrototypes:
    def test_clean_four_player_set_passes(self) -> None:
        anchors = _anchors_for_four_players()
        result = validate_prototypes(anchors)
        assert isinstance(result, ValidationResult)
        assert result.ok, result.issues

    def test_missing_player_informational_in_partial_mode(self) -> None:
        # Partial mode (default): missing-player is non-blocking; the
        # matcher simply skips that player's override.
        anchors = _anchors_for_four_players()
        anchors.prototypes.pop(3)
        anchors.per_crop_embeddings.pop(3)
        result = validate_prototypes(
            anchors, expected_player_ids=[1, 2, 3, 4], partial_ok=True,
        )
        assert result.ok
        missing = [i for i in result.issues if i.code == "missing_crops_optional"]
        assert missing
        assert missing[0].player_id == 3

    def test_missing_player_blocks_in_strict_mode(self) -> None:
        anchors = _anchors_for_four_players()
        anchors.prototypes.pop(3)
        anchors.per_crop_embeddings.pop(3)
        result = validate_prototypes(
            anchors, expected_player_ids=[1, 2, 3, 4], partial_ok=False,
        )
        assert not result.ok
        missing = [i for i in result.issues if i.code == "missing_crops"]
        assert missing
        assert missing[0].player_id == 3

    def test_single_player_labeling_passes(self) -> None:
        anchors = _anchors_for_four_players()
        for pid in (2, 3, 4):
            anchors.prototypes.pop(pid)
            anchors.per_crop_embeddings.pop(pid)
        result = validate_prototypes(
            anchors, expected_player_ids=[1, 2, 3, 4], partial_ok=True,
        )
        assert result.ok, result.issues

    def test_no_crops_at_all_blocks(self) -> None:
        empty = IdentityAnchors(prototypes={}, per_crop_embeddings={})
        result = validate_prototypes(
            empty, expected_player_ids=[1, 2, 3, 4], partial_ok=True,
        )
        assert not result.ok
        assert any(i.code == "no_crops" for i in result.issues)

    def test_single_crop_emits_warning_but_passes(self) -> None:
        # few_crops is non-blocking; validation should still pass.
        anchors = _anchors_for_four_players()
        anchors.per_crop_embeddings[1] = anchors.per_crop_embeddings[1][:1]
        result = validate_prototypes(anchors)
        warnings = [i for i in result.issues if i.code == "few_crops"]
        assert warnings
        assert warnings[0].player_id == 1
        assert result.ok

    def test_identical_crops_flagged_as_low_diversity(self) -> None:
        anchors = _anchors_for_four_players()
        proto = anchors.prototypes[1]
        # Replace all of player 1's crops with the exact same vector.
        anchors.per_crop_embeddings[1] = np.stack([proto, proto, proto], axis=0)
        result = validate_prototypes(anchors)
        assert not result.ok
        assert any(
            i.code == "low_diversity" and i.player_id == 1 for i in result.issues
        )

    def test_identical_prototypes_flagged_as_players_look_alike(self) -> None:
        # Two players with nearly-identical prototypes.
        shared = _basis_prototype(0)
        p3 = _basis_prototype(2)
        p4 = _basis_prototype(3)
        # Perturb p1/p2 slightly differently so their own crops have spread
        # but their prototypes still collide.
        rng = np.random.default_rng(0)
        noise = lambda: rng.normal(0, 0.05, size=(3, 384)).astype(np.float32)  # noqa: E731
        c1 = np.stack([_unit(shared + n) for n in noise()], axis=0)
        c2 = np.stack([_unit(shared + n) for n in noise()], axis=0)
        c3 = np.stack([_unit(p3 + n) for n in noise()], axis=0)
        c4 = np.stack([_unit(p4 + n) for n in noise()], axis=0)
        anchors = IdentityAnchors(
            prototypes={1: shared, 2: shared.copy(), 3: p3, 4: p4},
            per_crop_embeddings={1: c1, 2: c2, 3: c3, 4: c4},
        )
        result = validate_prototypes(anchors)
        assert not result.ok
        assert any(i.code == "players_look_alike" for i in result.issues)

    def test_misassigned_crop_flagged(self) -> None:
        anchors = _anchors_for_four_players()
        # Plant a player-3-looking crop inside player-1's set.
        p3_proto = anchors.prototypes[3]
        hijacked = np.stack(
            [
                anchors.per_crop_embeddings[1][0],
                anchors.per_crop_embeddings[1][1],
                p3_proto,
            ],
            axis=0,
        )
        anchors.per_crop_embeddings[1] = hijacked
        result = validate_prototypes(anchors)
        assert not result.ok
        assert any(
            i.code == "crop_likely_misassigned" and i.player_id == 1
            for i in result.issues
        )

    def test_expected_player_ids_enforced_strict(self) -> None:
        # Two-player anchor object but caller expects four players, strict.
        anchors = IdentityAnchors(
            prototypes={1: _basis_prototype(0), 2: _basis_prototype(1)},
            per_crop_embeddings={
                1: _basis_prototype(0)[None, :],
                2: _basis_prototype(1)[None, :],
            },
        )
        result = validate_prototypes(
            anchors, expected_player_ids=[1, 2, 3, 4], partial_ok=False,
        )
        assert not result.ok
        missing_pids = {
            i.player_id for i in result.issues if i.code == "missing_crops"
        }
        assert missing_pids == {3, 4}

    def test_to_dict_shape_is_stable(self) -> None:
        anchors = _anchors_for_four_players()
        result = validate_prototypes(anchors)
        payload = result.to_dict()
        assert isinstance(payload, dict)
        assert "pass" in payload
        assert "issues" in payload
        for issue in payload["issues"]:
            assert set(issue.keys()) == {"code", "message", "playerId"}


# ---------------------------------------------------------------------------
# IdentityAnchors convenience props
# ---------------------------------------------------------------------------


class TestIdentityAnchors:
    def test_player_ids_sorted(self) -> None:
        anchors = IdentityAnchors(
            prototypes={
                4: _basis_prototype(3),
                1: _basis_prototype(0),
                2: _basis_prototype(1),
            },
            per_crop_embeddings={},
        )
        assert anchors.player_ids == [1, 2, 4]

    def test_prototype_matrix_row_order_matches_player_ids(self) -> None:
        anchors = _anchors_for_four_players()
        matrix = anchors.prototype_matrix
        assert matrix.shape == (4, 384)
        for row_idx, pid in enumerate(anchors.player_ids):
            np.testing.assert_allclose(matrix[row_idx], anchors.prototypes[pid])


def test_validation_issue_is_plain_dataclass() -> None:
    issue = ValidationIssue(code="foo", message="bar", player_id=1)
    assert issue.code == "foo"
    assert issue.player_id == 1
