"""Tests for the multi-signal team-pair partition determination (v3).

`_propose_team_partitions` collects candidate 2v2 partitions from all
available position signals (y-side, bbox-side, upstream team_assignments).
`_assign_with_team_pair` then runs constrained Hungarian under each
candidate (and both pair-to-side orientations) and picks the lowest-cost
result. Together these replace the legacy unanimity gate that disengaged
the team-pair constraint whenever any single track's two side classifiers
disagreed.

Robustness contract this file pins:
  - Each signal contributes at most one candidate (its 2v2 split).
  - Degenerate signals (3v1, fewer than 4 tracks classified) contribute
    nothing — they don't poison the candidate pool.
  - Two signals proposing the same partition collapse to one candidate
    (no spurious duplicates).
  - When 0 signals propose, the caller falls back to unconstrained
    Hungarian (tested via the call site, not here).
  - When 2+ signals propose disjoint partitions, both are tried and
    appearance cost decides — proven in `_assign_with_team_pair` test.
"""
from __future__ import annotations

import numpy as np

from rallycut.tracking.match_solver import (
    HARD_TEAM_PAIR_COST,
    TEAM_HI_PAIR,
    TEAM_LO_PAIR,
    MatchSolver,
    _propose_team_partitions,
    _signal_proposes_partition,
)


class TestSignalProposesPartition:
    def test_clean_2v2_returns_partition(self) -> None:
        sig = {1: 0, 2: 0, 3: 1, 4: 1}
        out = _signal_proposes_partition(sig, {1, 2, 3, 4})
        assert out == frozenset({frozenset({1, 2}), frozenset({3, 4})})

    def test_3v1_returns_none(self) -> None:
        # d934f57a's y-side classifier shape: 3 near, 1 far.
        sig = {1: 0, 2: 0, 101: 0, 3: 1}
        out = _signal_proposes_partition(sig, {1, 2, 101, 3})
        assert out is None

    def test_4v0_returns_none(self) -> None:
        sig = {1: 0, 2: 0, 3: 0, 4: 0}
        out = _signal_proposes_partition(sig, {1, 2, 3, 4})
        assert out is None

    def test_empty_signal_returns_none(self) -> None:
        assert _signal_proposes_partition({}, {1, 2, 3, 4}) is None
        assert _signal_proposes_partition(None, {1, 2, 3, 4}) is None

    def test_signal_with_extra_track_ids_ignored(self) -> None:
        # Signal classifies tracks outside the requested set; only the
        # requested 4 should be considered.
        sig = {1: 0, 2: 0, 3: 1, 4: 1, 99: 0}
        out = _signal_proposes_partition(sig, {1, 2, 3, 4})
        assert out == frozenset({frozenset({1, 2}), frozenset({3, 4})})

    def test_partial_classification_returns_none(self) -> None:
        # Only 3 of the 4 requested tracks have classifications.
        sig = {1: 0, 2: 0, 3: 1}
        out = _signal_proposes_partition(sig, {1, 2, 3, 4})
        assert out is None


class TestProposeTeamPartitions:
    def test_two_signals_agree_returns_one_candidate(self) -> None:
        # Best case: both signals see the same 2v2 split.
        y_side = {1: 0, 2: 0, 3: 1, 4: 1}
        bbox_side = {1: 0, 2: 0, 3: 1, 4: 1}
        out = _propose_team_partitions([1, 2, 3, 4], y_side, bbox_side)
        assert len(out) == 1
        assert out[0] == frozenset({frozenset({1, 2}), frozenset({3, 4})})

    def test_two_signals_disagree_returns_two_candidates(self) -> None:
        # 09553ef1 shape: both 2v2 but disagree on T2 and T4.
        y_side = {1: 0, 2: 0, 3: 1, 4: 1}     # {T1,T2} vs {T3,T4}
        bbox_side = {1: 0, 4: 0, 2: 1, 3: 1}  # {T1,T4} vs {T2,T3}
        out = _propose_team_partitions([1, 2, 3, 4], y_side, bbox_side)
        assert len(out) == 2

    def test_one_degenerate_one_clean_returns_one_candidate(self) -> None:
        # d934f57a shape: y-side is 3v1, bbox-side is clean 2v2.
        y_side = {1: 0, 2: 0, 3: 0, 4: 1}      # 3v1 — degenerate
        bbox_side = {1: 0, 4: 0, 2: 1, 3: 1}   # 2v2
        out = _propose_team_partitions([1, 2, 3, 4], y_side, bbox_side)
        assert len(out) == 1
        assert out[0] == frozenset({frozenset({1, 4}), frozenset({2, 3})})

    def test_all_degenerate_returns_empty(self) -> None:
        # No signal proposes anything — caller falls back to unconstrained.
        y_side = {1: 0, 2: 0, 3: 0, 4: 0}    # 4v0
        bbox_side = {1: 0, 2: 0, 3: 0}       # incomplete
        out = _propose_team_partitions([1, 2, 3, 4], y_side, bbox_side)
        assert out == []

    def test_three_signals_supported(self) -> None:
        # team_assignments adds a third vote.
        y_side = {1: 0, 2: 0, 3: 1, 4: 1}
        bbox_side = {1: 0, 4: 0, 2: 1, 3: 1}
        team = {1: 0, 2: 0, 3: 1, 4: 1}  # agrees with y-side
        out = _propose_team_partitions([1, 2, 3, 4], y_side, bbox_side, team)
        # Two distinct candidates (y/team agree → 1; bbox → 1).
        assert len(out) == 2

    def test_wrong_track_count_returns_empty(self) -> None:
        # Solver only invokes team-pair logic for exactly 4 tracks; this
        # asserts the helper's defensive guard.
        out = _propose_team_partitions([1, 2, 3], {1: 0, 2: 0, 3: 1})
        assert out == []

    def test_none_signal_skipped(self) -> None:
        # Caller passes None when a signal isn't available — must not
        # raise or contribute a spurious candidate.
        y_side = {1: 0, 2: 0, 3: 1, 4: 1}
        out = _propose_team_partitions([1, 2, 3, 4], y_side, None, None)
        assert len(out) == 1


class TestAssignWithTeamPairCostDecided:
    """When two candidate partitions disagree, appearance cost decides."""

    def _solver(self) -> MatchSolver:
        return MatchSolver()

    def test_single_candidate_picks_lowest_cost_orientation(self) -> None:
        # 4 tracks, partition {{1,2}, {3,4}}, cost matrix designed so
        # {1,2}→{P1,P2}, {3,4}→{P3,P4} is much cheaper than the mirror.
        top = [1, 2, 3, 4]
        cluster_ids = list(TEAM_LO_PAIR + TEAM_HI_PAIR)  # [1, 2, 3, 4]
        # Diagonal-low cost: T_i → P_i is cheap (0.0); off-diagonal high.
        cost = np.full((4, 4), 0.9, dtype=float)
        np.fill_diagonal(cost, 0.0)
        partition = frozenset({frozenset({1, 2}), frozenset({3, 4})})

        assignment, _ = self._solver()._assign_with_team_pair(
            top, cluster_ids, cost, [partition],
        )
        assert assignment == {1: 1, 2: 2, 3: 3, 4: 4}

    def test_two_candidates_data_picks_better(self) -> None:
        # Two candidate partitions; appearance data favors one cleanly.
        top = [1, 2, 3, 4]
        cluster_ids = list(TEAM_LO_PAIR + TEAM_HI_PAIR)
        # Cost matrix that strongly prefers {1,4} together and {2,3} together.
        # T1 → P1 cheap, T4 → P2 cheap, T2 → P3 cheap, T3 → P4 cheap.
        cost = np.array([
            [0.0, 0.9, 0.9, 0.9],
            [0.9, 0.9, 0.0, 0.9],
            [0.9, 0.9, 0.9, 0.0],
            [0.9, 0.0, 0.9, 0.9],
        ])
        partition_a = frozenset({frozenset({1, 2}), frozenset({3, 4})})
        partition_b = frozenset({frozenset({1, 4}), frozenset({2, 3})})

        assignment, _ = self._solver()._assign_with_team_pair(
            top, cluster_ids, cost, [partition_a, partition_b],
        )
        # Cost-decided: partition_b's orientation wins.
        assert assignment[1] == 1
        assert assignment[4] == 2
        assert assignment[2] == 3
        assert assignment[3] == 4

    def test_uniform_cost_picks_side_consistent_orientation(self) -> None:
        """Iter-0 fix: when appearance cost is uniform (no profiles yet),
        break the orientation tie using `track_court_sides`.

        Mechanism: at MatchSolver iter 0 the cluster profiles are empty,
        so ``_build_appearance_cost`` returns 0.5 in every cell. The
        partition is correct but all 4 (orientation × pair-to-target)
        combinations have equal total cost. Without a tie-breaker, the
        first combination Python iterates wins — which on b5fb0594
        rally 1 (be3134ba, 2026-05-07) placed both near-court tracks
        into HI_PAIR PIDs, baking a cross-team error into iter-0 cluster
        members and propagating it to subsequent iterations.

        Fix: prefer the orientation where near-classified tracks
        (``track_court_sides[tid] == 0``) land in LO_PAIR (P1, P2) and
        far-classified tracks (``track_court_sides[tid] == 1``) land in
        HI_PAIR (P3, P4). The tie-breaker uses a tiny epsilon so it
        never overrides real appearance differences.
        """
        top = [1, 18, 3, 26]
        cluster_ids = list(TEAM_LO_PAIR + TEAM_HI_PAIR)
        # Uniform cost — every (track, cluster) is equally good.
        cost = np.full((4, 4), 0.5, dtype=float)
        # Side classification: T1, T18 are near; T3, T26 are far.
        # The correct orientation is T1, T18 → P1, P2 (LO_PAIR) and
        # T3, T26 → P3, P4 (HI_PAIR).
        track_court_sides = {1: 0, 18: 0, 3: 1, 26: 1}
        partition = frozenset({frozenset({1, 18}), frozenset({3, 26})})

        assignment, _ = self._solver()._assign_with_team_pair(
            top, cluster_ids, cost, [partition],
            track_court_sides=track_court_sides,
        )
        # Near tracks must end up in LO_PAIR (P1, P2).
        assert assignment[1] in TEAM_LO_PAIR, f"T1 should be in LO_PAIR, got {assignment}"
        assert assignment[18] in TEAM_LO_PAIR, f"T18 should be in LO_PAIR, got {assignment}"
        # Far tracks must end up in HI_PAIR (P3, P4).
        assert assignment[3] in TEAM_HI_PAIR, f"T3 should be in HI_PAIR, got {assignment}"
        assert assignment[26] in TEAM_HI_PAIR, f"T26 should be in HI_PAIR, got {assignment}"

    def test_appearance_overrides_side_hint_when_signals_disagree(self) -> None:
        """Tie-breaker is small enough that real appearance always wins.

        Without this guarantee, the side-classifier would be promoted
        from a tie-breaker to a hard rule — overriding genuine appearance
        evidence. The tie-breaker must only act when totals are tied.
        """
        top = [1, 18, 3, 26]
        cluster_ids = list(TEAM_LO_PAIR + TEAM_HI_PAIR)
        # Appearance STRONGLY prefers the side-INCONSISTENT orientation:
        # T1, T18 → P3, P4 (cost 0.0) vs P1, P2 (cost 0.9).
        cost = np.array([
            [0.9, 0.9, 0.0, 0.9],   # T1 → P3 cheapest
            [0.9, 0.9, 0.9, 0.0],   # T18 → P4 cheapest
            [0.0, 0.9, 0.9, 0.9],   # T3 → P1 cheapest
            [0.9, 0.0, 0.9, 0.9],   # T26 → P2 cheapest
        ])
        # Side-classifier hints the opposite — but appearance must win.
        track_court_sides = {1: 0, 18: 0, 3: 1, 26: 1}
        partition = frozenset({frozenset({1, 18}), frozenset({3, 26})})

        assignment, _ = self._solver()._assign_with_team_pair(
            top, cluster_ids, cost, [partition],
            track_court_sides=track_court_sides,
        )
        # Appearance wins: T1, T18 in HI_PAIR (the cheap cells).
        assert assignment[1] in TEAM_HI_PAIR
        assert assignment[18] in TEAM_HI_PAIR
        assert assignment[3] in TEAM_LO_PAIR
        assert assignment[26] in TEAM_LO_PAIR

    def test_partition_constraint_actually_enforced(self) -> None:
        # If the optimal unconstrained Hungarian would violate the
        # partition, the constrained version must NOT pick it.
        top = [1, 2, 3, 4]
        cluster_ids = list(TEAM_LO_PAIR + TEAM_HI_PAIR)
        # Cost matrix where unconstrained best is {1:1, 2:3, 3:2, 4:4}
        # (which puts T2 in HI_PAIR and T3 in LO_PAIR — violates {1,2}/{3,4}
        # under partition {{1,2},{3,4}}).
        cost = np.array([
            [0.0, 0.9, 0.9, 0.9],
            [0.9, 0.9, 0.0, 0.9],   # T2 cheapest at P3
            [0.9, 0.0, 0.9, 0.9],   # T3 cheapest at P2
            [0.9, 0.9, 0.9, 0.0],
        ])
        partition = frozenset({frozenset({1, 2}), frozenset({3, 4})})

        assignment, total_cost = self._solver()._assign_with_team_pair(
            top, cluster_ids, cost, [partition],
        )
        # T2 must be in {P1, P2} (lo pair); T3 must be in {P3, P4} (hi pair)
        # OR mirror (T2 in hi, T3 in lo). NOT a cross.
        assert (assignment[1] in TEAM_LO_PAIR) == (assignment[2] in TEAM_LO_PAIR), (
            f"T1, T2 must be in same pair (got {assignment})"
        )
        assert (assignment[3] in TEAM_LO_PAIR) == (assignment[4] in TEAM_LO_PAIR), (
            f"T3, T4 must be in same pair (got {assignment})"
        )
        # And the total cost must NOT contain a HARD penalty (otherwise
        # the partition wasn't enforced — the solver picked a forbidden cell).
        assert total_cost < HARD_TEAM_PAIR_COST
