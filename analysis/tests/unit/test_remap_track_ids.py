"""Tests for remap_track_ids idempotency helpers."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from rallycut.cli.commands.remap_track_ids import (
    _build_full_mapping,
    _invert_mapping,
    _remap_actions,
    _remap_contacts,
    _remap_positions,
    _should_reverse,
)


class TestInvertMapping:
    def test_simple(self) -> None:
        mapping = {3: 1, 7: 2, 12: 3, 15: 4}
        assert _invert_mapping(mapping) == {1: 3, 2: 7, 3: 12, 4: 15}

    def test_with_collision_shifts(self) -> None:
        """Collision-shifted IDs (101+) should also invert."""
        mapping = {3: 1, 7: 2, 12: 3, 15: 4, 1: 101, 2: 102}
        inverse = _invert_mapping(mapping)
        assert inverse == {1: 3, 2: 7, 3: 12, 4: 15, 101: 1, 102: 2}

    def test_identity(self) -> None:
        mapping = {1: 1, 2: 2, 3: 3}
        assert _invert_mapping(mapping) == {1: 1, 2: 2, 3: 3}

    def test_rejects_non_bijective(self) -> None:
        mapping = {3: 1, 7: 1}  # Both map to 1
        with pytest.raises(ValueError, match="Non-bijective"):
            _invert_mapping(mapping)


class TestShouldReverse:
    def test_remapped_positions_match(self) -> None:
        """Positions with remapped IDs should trigger reversal."""
        positions = [{"trackId": 1}, {"trackId": 2}, {"trackId": 3}, {"trackId": 4}]
        applied = {3: 1, 7: 2, 12: 3, 15: 4}
        assert _should_reverse(positions, applied) is True

    def test_original_positions_no_reverse(self) -> None:
        """Positions with original tracker IDs should not trigger reversal."""
        positions = [{"trackId": 3}, {"trackId": 7}, {"trackId": 12}, {"trackId": 15}]
        applied = {3: 1, 7: 2, 12: 3, 15: 4}
        # Original IDs {3, 7, 12, 15} are NOT a subset of output values {1, 2, 3, 4}
        # (7, 12, 15 are not in output values) → should NOT reverse
        assert _should_reverse(positions, applied) is False

    def test_empty_mapping(self) -> None:
        positions = [{"trackId": 1}]
        assert _should_reverse(positions, {}) is False

    def test_empty_positions(self) -> None:
        applied = {3: 1, 7: 2}
        assert _should_reverse([], applied) is False

    def test_with_collision_shifted_ids(self) -> None:
        """Positions with collision-shifted IDs are also remapped output."""
        positions = [
            {"trackId": 1}, {"trackId": 2}, {"trackId": 3}, {"trackId": 4},
            {"trackId": 101},
        ]
        applied = {3: 1, 7: 2, 12: 3, 15: 4, 1: 101}
        assert _should_reverse(positions, applied) is True

    def test_after_retrack_fresh_ids(self) -> None:
        """After re-tracking, fresh IDs don't match old mapping output."""
        positions = [{"trackId": 20}, {"trackId": 25}, {"trackId": 30}]
        applied = {3: 1, 7: 2, 12: 3, 15: 4}
        assert _should_reverse(positions, applied) is False

    def test_identity_mapping_returns_false(self) -> None:
        """Identity mapping (no actual remap) should not trigger reversal."""
        positions = [{"trackId": 1}, {"trackId": 2}]
        applied = {1: 1, 2: 2}  # Identity
        assert _should_reverse(positions, applied) is False

    def test_overlapping_input_output_non_identity(self) -> None:
        """Non-identity mapping where IDs overlap both input and output ranges."""
        # Mapping: {3→1, 1→3} (swap). IDs {1, 3} are subset of both.
        positions = [{"trackId": 1}, {"trackId": 3}]
        applied = {3: 1, 1: 3}
        # Non-identity, ambiguous → returns True (callers use remapApplied to gate)
        assert _should_reverse(positions, applied) is True


class TestBuildFullMappingBijective:
    def test_no_collisions(self) -> None:
        mapping = _build_full_mapping({3: 1, 7: 2}, {3, 7, 20})
        # Should be bijective
        _invert_mapping(mapping)  # Raises if not bijective
        assert mapping[3] == 1
        assert mapping[7] == 2
        assert mapping[20] == 20  # No collision

    def test_with_collisions(self) -> None:
        mapping = _build_full_mapping({3: 1, 7: 2}, {1, 2, 3, 7})
        # 1 and 2 collide with output player IDs → shifted to 101+
        _invert_mapping(mapping)  # Must be bijective
        assert mapping[3] == 1
        assert mapping[7] == 2
        assert mapping[1] >= 101
        assert mapping[2] >= 101


class TestRemapRoundtrip:
    def _make_positions(self, track_ids: list[int]) -> list[dict[str, Any]]:
        return [
            {"trackId": tid, "frame": i, "x": 0.5, "y": 0.5}
            for i, tid in enumerate(track_ids)
        ]

    def test_apply_then_invert_restores(self) -> None:
        """Remap then inverse-remap should restore original IDs."""
        original_ids = [3, 7, 12, 15, 3, 7]
        positions = self._make_positions(original_ids)

        track_to_player = {3: 1, 7: 2, 12: 3, 15: 4}
        all_ids = set(original_ids)
        mapping = _build_full_mapping(track_to_player, all_ids)

        # Apply forward
        _remap_positions(positions, mapping)
        assert [p["trackId"] for p in positions] == [1, 2, 3, 4, 1, 2]

        # Apply inverse
        inverse = _invert_mapping(mapping)
        _remap_positions(positions, inverse)
        assert [p["trackId"] for p in positions] == original_ids


class TestRemapIdempotent:
    """Simulate two full pipeline runs and verify identical result."""

    def _make_rally_data(
        self,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], list[int]]:
        positions = [
            {"trackId": 3, "frame": 0, "x": 0.1, "y": 0.2},
            {"trackId": 7, "frame": 0, "x": 0.3, "y": 0.4},
            {"trackId": 12, "frame": 1, "x": 0.5, "y": 0.6},
            {"trackId": 15, "frame": 1, "x": 0.7, "y": 0.8},
            {"trackId": 1, "frame": 2, "x": 0.9, "y": 0.1},  # Collides with player 1
        ]
        contacts = {"contacts": [{"playerTrackId": 3, "playerCandidates": [[7, 0.9]]}]}
        actions = {
            "actions": [{"playerTrackId": 12}],
            "teamAssignments": {"3": "near", "7": "far"},
        }
        primary_ids = [3, 7, 12, 15]
        return positions, contacts, actions, primary_ids

    def _run_remap(
        self,
        track_to_player: dict[int, int],
        positions: list[dict[str, Any]],
        contacts: dict[str, Any],
        actions: dict[str, Any],
        primary_ids: list[int],
        applied_full_mapping: dict[int, int] | None,
        remap_applied: bool = False,
    ) -> dict[int, int]:
        """Simulate one remap run. Returns the new appliedFullMapping."""
        # Step 1: Reverse if needed (requires both flag AND subset check)
        if applied_full_mapping and remap_applied:
            if _should_reverse(positions, applied_full_mapping):
                inverse = _invert_mapping(applied_full_mapping)
                _remap_positions(positions, inverse)
                _remap_contacts(contacts, inverse)
                _remap_actions(actions, inverse)
                primary_ids[:] = [inverse.get(t, t) for t in primary_ids]

        # Step 2: Build new mapping
        all_ids = {p["trackId"] for p in positions} | set(primary_ids)
        mapping = _build_full_mapping(track_to_player, all_ids)

        # Step 3: Apply
        _remap_positions(positions, mapping)
        _remap_contacts(contacts, mapping)
        _remap_actions(actions, mapping)
        primary_ids[:] = [mapping.get(t, t) for t in primary_ids]

        return mapping

    def test_two_runs_same_result(self) -> None:
        track_to_player = {3: 1, 7: 2, 12: 3, 15: 4}

        # --- Run 1 (no prior remap) ---
        pos1, con1, act1, pri1 = self._make_rally_data()
        afm1 = self._run_remap(track_to_player, pos1, con1, act1, pri1, None)

        snapshot_after_run1 = {
            "positions": copy.deepcopy(pos1),
            "contacts": copy.deepcopy(con1),
            "actions": copy.deepcopy(act1),
            "primary_ids": list(pri1),
        }

        # --- Run 2 (data already remapped, remap_applied=True) ---
        afm2 = self._run_remap(
            track_to_player, pos1, con1, act1, pri1,
            afm1, remap_applied=True,
        )

        snapshot_after_run2 = {
            "positions": pos1,
            "contacts": con1,
            "actions": act1,
            "primary_ids": list(pri1),
        }

        # Results must be identical
        assert snapshot_after_run1 == snapshot_after_run2
        # appliedFullMapping should be the same
        assert afm1 == afm2

    def test_reversal_skipped_after_retrack(self) -> None:
        """After re-tracking, positions have fresh IDs that don't match old mapping."""
        track_to_player = {3: 1, 7: 2, 12: 3, 15: 4}

        # Run 1 with original data
        pos1, con1, act1, pri1 = self._make_rally_data()
        afm1 = self._run_remap(track_to_player, pos1, con1, act1, pri1, None)

        # Simulate re-tracking: completely fresh IDs
        fresh_positions = [
            {"trackId": 20, "frame": 0, "x": 0.1, "y": 0.2},
            {"trackId": 25, "frame": 0, "x": 0.3, "y": 0.4},
            {"trackId": 30, "frame": 1, "x": 0.5, "y": 0.6},
            {"trackId": 35, "frame": 1, "x": 0.7, "y": 0.8},
        ]
        fresh_contacts: dict[str, Any] = {"contacts": [{"playerTrackId": 20}]}
        fresh_actions: dict[str, Any] = {"actions": [{"playerTrackId": 30}]}
        fresh_primary = [20, 25, 30, 35]

        new_mapping = {20: 1, 25: 2, 30: 3, 35: 4}

        # Run 2 with old afm but fresh data → subset check fails → skip reversal
        afm2 = self._run_remap(
            new_mapping, fresh_positions, fresh_contacts, fresh_actions,
            fresh_primary, afm1, remap_applied=True,
        )

        # Fresh data should be remapped with new mapping directly
        assert [p["trackId"] for p in fresh_positions] == [1, 2, 3, 4]
        assert fresh_primary == [1, 2, 3, 4]
        assert afm2 == {20: 1, 25: 2, 30: 3, 35: 4}

    def test_no_reversal_without_remap_applied_flag(self) -> None:
        """Even with valid appliedFullMapping, no reversal without remapApplied."""
        track_to_player = {3: 1, 7: 2, 12: 3, 15: 4}

        # Run 1
        pos1, con1, act1, pri1 = self._make_rally_data()
        afm1 = self._run_remap(track_to_player, pos1, con1, act1, pri1, None)

        # Run 2 WITHOUT remap_applied flag — no reversal, so the mapping
        # is applied on top of already-remapped data → corruption
        pos2 = copy.deepcopy(pos1)
        con2, act2, pri2 = copy.deepcopy(con1), copy.deepcopy(act1), list(pri1)
        self._run_remap(
            track_to_player, pos2, con2, act2, pri2,
            afm1, remap_applied=False,  # Flag not set
        )

        # Double-mapping corruption: IDs 1,2,4 collide with player IDs
        # and get shifted to 102,103,104. Only ID 3 maps correctly to 1.
        # This proves the flag is needed to prevent corruption.
        assert [p["trackId"] for p in pos2] == [102, 103, 1, 104, 101]

    def test_retrack_with_low_ids_no_false_reversal(self) -> None:
        """After re-tracking with IDs 1-4 (same as player IDs), no false reversal.

        This is the fragile case: tracker naturally assigns IDs that overlap with
        player IDs. The remapApplied flag prevents false reversal even when the
        subset check would be ambiguous.
        """
        # Original mapping: high tracker IDs → player IDs
        original_mapping = {3: 1, 7: 2, 12: 3, 15: 4}
        pos1 = [{"trackId": 3}, {"trackId": 7}, {"trackId": 12}, {"trackId": 15}]
        contacts1: dict[str, Any] = {"contacts": []}
        actions1: dict[str, Any] = {"actions": []}
        primary1 = [3, 7, 12, 15]

        afm1 = self._run_remap(
            original_mapping, pos1, contacts1, actions1, primary1, None,
        )
        # After run 1: positions have IDs 1,2,3,4

        # Simulate re-tracking: tracker happens to assign IDs 1,2,3,4
        fresh_positions = [{"trackId": 1}, {"trackId": 2}, {"trackId": 3}, {"trackId": 4}]
        fresh_contacts: dict[str, Any] = {"contacts": []}
        fresh_actions: dict[str, Any] = {"actions": []}
        fresh_primary = [1, 2, 3, 4]

        # New match-players produces identity mapping (IDs already correct)
        identity_mapping = {1: 1, 2: 2, 3: 3, 4: 4}

        # Without remapApplied flag (re-tracking clears it), no reversal
        self._run_remap(
            identity_mapping, fresh_positions, fresh_contacts, fresh_actions,
            fresh_primary, afm1, remap_applied=False,
        )

        # Fresh data should remain as-is (identity mapping applied = no change)
        assert [p["trackId"] for p in fresh_positions] == [1, 2, 3, 4]
        assert fresh_primary == [1, 2, 3, 4]
