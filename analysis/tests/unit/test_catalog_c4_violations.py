"""Tests for the pure signal-computation helpers in catalog_c4_violations.

The DB-read + CSV-write orchestration in that script is integration-only
(verified by Task 7's smoke run); the pure helpers below are unit-tested.
"""

from __future__ import annotations

import math

import pytest

from scripts.catalog_c4_violations import (
    EXPECTED_TRANSITIONS,
    best_same_team_alt_ratio,
    placeholder_repair_recommendation,
    signal_team_geometry,
    signal_type_fit,
)


class TestSignalTypeFit:
    def test_known_ok_pair(self) -> None:
        # Serve → receive is in the table with label 'ok'.
        assert signal_type_fit("serve", "receive") == "ok"

    def test_known_wrong_pair(self) -> None:
        # Serve → set is in the table with label 'wrong'.
        assert signal_type_fit("serve", "set") == "wrong"

    def test_unknown_pair_returns_unknown(self) -> None:
        assert signal_type_fit("attack", "set") == "unknown"  # not in table

    def test_block_prev_always_ok(self) -> None:
        # block → anything is 'ok' (C-4 detector exempts the pair anyway,
        # but the type-fit signal must agree, not return 'wrong').
        assert signal_type_fit("block", "receive") == "ok"
        assert signal_type_fit("block", "set") == "ok"
        assert signal_type_fit("block", "attack") == "ok"  # fallback for any block→X

    def test_expected_transitions_constant_has_required_keys(self) -> None:
        # Sanity: the table must include the volleyball happy-path
        # transitions that the spec's expected-transitions section listed.
        required = [
            ("serve", "receive"),
            ("serve", "dig"),
            ("receive", "set"),
            ("set", "attack"),
            ("attack", "dig"),
            ("attack", "block"),
            ("dig", "set"),
        ]
        for key in required:
            assert key in EXPECTED_TRANSITIONS, f"missing transition {key}"


class TestBestSameTeamAltRatio:
    def test_no_same_team_alt_returns_nan(self) -> None:
        # Only opposite-team candidates → no alt → NaN.
        candidates = [(101, 0.50, "B"), (102, 0.55, "B")]
        ratio = best_same_team_alt_ratio(
            candidates=candidates,
            current_dist=0.45,
            current_team="A",
            current_tid=100,
        )
        assert math.isnan(ratio)

    def test_same_team_closer_returns_ratio_below_one(self) -> None:
        # Same-team candidate at 0.20 vs current at 0.40 → ratio 0.5.
        candidates = [(200, 0.20, "A"), (201, 0.30, "B")]
        ratio = best_same_team_alt_ratio(
            candidates=candidates,
            current_dist=0.40,
            current_team="A",
            current_tid=100,
        )
        assert ratio == pytest.approx(0.5)

    def test_same_team_farther_returns_ratio_above_one(self) -> None:
        candidates = [(200, 0.80, "A"), (201, 0.30, "B")]
        ratio = best_same_team_alt_ratio(
            candidates=candidates,
            current_dist=0.40,
            current_team="A",
            current_tid=100,
        )
        assert ratio == pytest.approx(2.0)

    def test_excludes_current_tid(self) -> None:
        # The current player is in the candidate list (rank 1) but must be
        # excluded from the "alternative" search.
        candidates = [(100, 0.20, "A"), (200, 0.50, "A")]
        ratio = best_same_team_alt_ratio(
            candidates=candidates,
            current_dist=0.20,
            current_team="A",
            current_tid=100,
        )
        # Alt is tid=200 at 0.50, ratio = 0.50/0.20 = 2.5.
        assert ratio == pytest.approx(2.5)


class TestSignalTeamGeometry:
    def test_rank1_on_expected_team_matches(self) -> None:
        candidates = [(101, 0.30, "A"), (200, 0.50, "B")]
        result = signal_team_geometry(
            candidates=candidates, expected_team="A",
        )
        assert result == "matches"

    def test_rank1_wrong_team_rank2_within_2x_violates(self) -> None:
        # Rank1 is on team B (wrong), rank2 is team A at 0.50/0.30 = 1.67x → violates.
        candidates = [(101, 0.30, "B"), (200, 0.50, "A")]
        result = signal_team_geometry(
            candidates=candidates, expected_team="A",
        )
        assert result == "violates"

    def test_rank1_wrong_team_rank2_beyond_2x_ambiguous(self) -> None:
        # Rank1 team B at 0.20, rank2 team A at 0.50 → 2.5x → outside 2x → ambiguous.
        candidates = [(101, 0.20, "B"), (200, 0.50, "A")]
        result = signal_team_geometry(
            candidates=candidates, expected_team="A",
        )
        assert result == "ambiguous"

    def test_empty_candidates_returns_ambiguous(self) -> None:
        assert signal_team_geometry(candidates=[], expected_team="A") == "ambiguous"

    def test_no_expected_team_returns_ambiguous(self) -> None:
        candidates = [(101, 0.30, "A")]
        assert signal_team_geometry(candidates=candidates, expected_team=None) == "ambiguous"


class TestPlaceholderRepairRecommendation:
    def _row(self, **overrides: object) -> dict[str, object]:
        defaults: dict[str, object] = {
            "signal_type_fit_prev": "ok",
            "signal_type_fit_curr": "ok",
            "signal_team_geometry_prev": "matches",
            "signal_team_geometry_curr": "matches",
            "prev_best_same_team_alt_ratio": float("nan"),
            "curr_best_same_team_alt_ratio": float("nan"),
            "conf_prev": 0.9,
            "conf_curr": 0.9,
        }
        defaults.update(overrides)
        return defaults

    def test_all_signals_ok_recommends_skip(self) -> None:
        row = self._row()
        assert placeholder_repair_recommendation(row) == "skip"

    def test_two_strong_against_curr_recommends_repair_curr(self) -> None:
        row = self._row(
            signal_type_fit_curr="wrong",
            signal_team_geometry_curr="violates",
        )
        assert placeholder_repair_recommendation(row) == "repair_curr"

    def test_two_strong_against_prev_recommends_repair_prev(self) -> None:
        row = self._row(
            signal_type_fit_prev="wrong",
            conf_prev=0.3,
        )
        assert placeholder_repair_recommendation(row) == "repair_prev"

    def test_both_strong_returns_ambiguous(self) -> None:
        row = self._row(
            signal_type_fit_prev="wrong",
            signal_team_geometry_prev="violates",
            signal_type_fit_curr="wrong",
            signal_team_geometry_curr="violates",
        )
        assert placeholder_repair_recommendation(row) == "ambiguous"

    def test_alt_ratio_under_06_counts_against_side(self) -> None:
        # Curr has a same-team alt at 0.5x current — that's one strike. Plus
        # type_fit=wrong → 2 strong, recommend repair_curr.
        row = self._row(
            signal_type_fit_curr="wrong",
            curr_best_same_team_alt_ratio=0.4,
        )
        assert placeholder_repair_recommendation(row) == "repair_curr"
