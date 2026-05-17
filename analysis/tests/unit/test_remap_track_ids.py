"""Tests for remap_track_ids idempotency helpers."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from rallycut.cli.commands.remap_track_ids import (
    _build_full_mapping,
    _capture_snapshot,
    _check_snapshot_covered_by_mapping,
    _deepcopy_json,
    _invert_mapping,
    _load_canonical_per_rally,
    _remap_actions,
    _remap_contacts,
    _remap_positions,
    _resolve_raw_mapping_source,
    _should_reverse,
)
from rallycut.tracking.match_tracker import MATCHER_VERSION


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
    def test_unmapped_non_colliding_track_excluded(self) -> None:
        """Tracks with no PID and no collision are deliberately excluded.

        Pre-fix behaviour was an identity passthrough (`mapping[20] = 20`)
        which surfaced as junk PID labels in the editor (see
        chimera_stitching_dd042609_2026_05_04 memo). Now those tracks
        resolve to UNLABELED_TRACK_ID via `_resolve_with_overrides` and
        their positions get dropped in `_remap_positions`.
        """
        mapping = _build_full_mapping({3: 1, 7: 2}, {3, 7, 20})
        _invert_mapping(mapping)  # Must still be bijective on what's mapped
        assert mapping[3] == 1
        assert mapping[7] == 2
        assert 20 not in mapping  # Excluded — no PID, no collision

    def test_with_collisions(self) -> None:
        mapping = _build_full_mapping({3: 1, 7: 2}, {1, 2, 3, 7})
        # 1 and 2 collide with output player IDs → shifted to 101+ so
        # downstream tools (action attribution candidate lists, etc.)
        # can still reach them under unique IDs even though they have
        # no PID assignment.
        _invert_mapping(mapping)  # Must be bijective
        assert mapping[3] == 1
        assert mapping[7] == 2
        assert mapping[1] >= 101
        assert mapping[2] >= 101


class TestContractEnforcement:
    """Regression tests for the dd042609 r19/r21 fix (2026-05-04).

    Ensures that tracks present in `positions_json` but absent from
    `track_to_player` (i.e. no PID assignment from match-players) are
    dropped instead of leaked as identity passthroughs that surfaced
    as junk PID labels in the editor.
    """

    def test_orphan_position_dropped(self) -> None:
        # Reproduces the dd042609 r19 shape: track 7 in positions_json
        # but no PID assigned by match-players.
        positions = [
            {"trackId": 1, "frame": 0, "x": 0.1, "y": 0.2},
            {"trackId": 2, "frame": 0, "x": 0.3, "y": 0.4},
            {"trackId": 3, "frame": 0, "x": 0.5, "y": 0.6},
            {"trackId": 4, "frame": 0, "x": 0.7, "y": 0.8},
            # Orphan: track 7 has frames but no PID assignment.
            {"trackId": 7, "frame": 1, "x": 0.5, "y": 0.5},
            {"trackId": 7, "frame": 2, "x": 0.5, "y": 0.5},
        ]
        track_to_player = {1: 1, 2: 2, 3: 4, 4: 3}  # No entry for track 7
        all_ids = {1, 2, 3, 4, 7}

        mapping = _build_full_mapping(track_to_player, all_ids)
        # Track 7 must NOT be in the mapping — pre-fix it would be
        # `mapping[7] = 7` (identity passthrough).
        assert 7 not in mapping

        n_remapped, n_dropped = _remap_positions(positions, mapping)
        # All 4 PID-assigned positions get remapped to PID space; the
        # 2 orphan track-7 positions get dropped.
        track_ids = [p["trackId"] for p in positions]
        assert track_ids == [1, 2, 4, 3]
        assert n_dropped == 2
        assert n_remapped >= 1

    def test_collision_still_shifted_not_dropped(self) -> None:
        # Tracks that collide with PIDs but have no PID assignment
        # still get shifted to 101+ (so action attribution candidate
        # lists can resolve them under unique IDs).
        positions = [
            {"trackId": 1, "frame": 0, "x": 0.1, "y": 0.2},  # raw 1, collides with PID 1
            {"trackId": 7, "frame": 0, "x": 0.5, "y": 0.5},  # raw 7 → PID 1
        ]
        track_to_player = {7: 1}  # Only track 7 has a PID
        all_ids = {1, 7}

        mapping = _build_full_mapping(track_to_player, all_ids)
        assert mapping[7] == 1
        assert mapping[1] >= 101  # Shifted, not dropped

        n_remapped, n_dropped = _remap_positions(positions, mapping)
        # Both positions kept — orphan-with-collision became 101+.
        assert n_dropped == 0
        track_ids = sorted(p["trackId"] for p in positions)
        assert track_ids == [1, 101]

    def test_primary_track_ids_excludes_collision_shifts(self) -> None:
        """Regression: primary_track_ids must contain real PIDs only.

        Pre-fix: filter used `mapping.keys()` which includes collision-
        shifted IDs (e.g., track 4 with no PID becomes mapping[4]=101).
        That leaked 101 into primary_track_ids, which the editor
        rendered as a junk PID label. Post-fix: filter against
        `raw_mapping.keys()` (matcher's track_to_player keys), so only
        tracks that got a real PID assignment survive.

        Reproduces the post-recovery state of rally 8449bc3b
        (627c1add/caca): old primary_track_ids=[3, 4, 101, 1] with
        match-players track_to_player={1:2, 2:1, 13:4} →
        new primary_track_ids must be [2] (only track 1 was in
        track_to_player), NOT [2, 101] (which would happen if we
        let collision-shifted 4→101 leak in).
        """
        old_primary = [3, 4, 101, 1]
        track_to_player = {1: 2, 2: 1, 13: 4}  # matcher's PID assignments
        all_ids = set(old_primary) | set(track_to_player.keys())

        mapping = _build_full_mapping(track_to_player, all_ids)
        # Track 4 collides with PID 4 (in used_ids) → shifts to 101+.
        assert mapping[4] >= 101
        # Real PID-assigned tracks unchanged.
        assert mapping[1] == 2
        assert mapping[2] == 1
        assert mapping[13] == 4
        # Tracks 3, 101 have no PID and don't collide → excluded.
        assert 3 not in mapping
        assert 101 not in mapping

        # Contract enforcement: primary_track_ids must be subset of
        # raw_mapping.values() (real PIDs).
        new_primary = [
            track_to_player[t] for t in old_primary if t in track_to_player
        ]
        assert new_primary == [2]
        for pid in new_primary:
            assert pid in track_to_player.values()
            # And NOT a collision-shifted ID:
            assert pid < 100

    def test_identity_mapping_still_drops_orphan_primary(self) -> None:
        """Regression: identity-mapping shortcut must not bypass cleanup.

        Pre-fix: when match-players returned an identity track_to_player
        (e.g. {1:1, 2:2, 3:3}), `remap-track-ids` took a shortcut that
        skipped position+primary_track_ids cleanup. Stale orphans like
        track 18 stayed in DB. Post-fix: cleanup runs ALWAYS, regardless
        of mapping shape.

        Reproduces rally be3134ba (b5fb0594/ruru): old primary=[1, 3, 18, 2]
        with track_to_player={1:1, 2:2, 3:3} (identity).
        """
        old_primary = [1, 3, 18, 2]
        track_to_player = {1: 1, 2: 2, 3: 3}  # identity mapping
        all_ids = set(old_primary) | set(track_to_player.keys())

        mapping = _build_full_mapping(track_to_player, all_ids)
        # mapping is identity for matched tracks; track 18 absent
        # (no PID, no collision since 18 ∉ used_ids={1,2,3}).
        assert all(k == v for k, v in mapping.items())
        assert 18 not in mapping

        new_primary = [
            track_to_player[t] for t in old_primary if t in track_to_player
        ]
        assert new_primary == [1, 3, 2]  # Track 18 dropped
        assert 18 not in new_primary


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
        # The 5th position originally had track_id=1 (collision-shifted to
        # 101 in run 1); on run 2 it lands on 101 which has no PID
        # assignment and no collision — under the new contract enforcement
        # (remap-track-ids drops orphan positions instead of leaking
        # them as identity passthroughs) the position is dropped.
        # Pre-fix expected `[102, 103, 1, 104, 101]`; the trailing 101
        # was the contract leak this fix removes.
        assert [p["trackId"] for p in pos2] == [102, 103, 1, 104]

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


class TestLoadCanonicalPerRally:
    """`_load_canonical_per_rally` must invalidate stale-version canonical
    maps so a `MATCHER_VERSION` bump auto-clears the cache. Mirrors the
    assignmentAnchor stale-version pattern (commit `2df8af4`)."""

    def test_none_input_returns_empty(self) -> None:
        out, stale = _load_canonical_per_rally(None)
        assert out == {}
        assert stale is None

    def test_empty_dict_returns_empty(self) -> None:
        out, stale = _load_canonical_per_rally({})
        assert out == {}
        assert stale is None

    def test_current_version_loads_normally(self) -> None:
        payload = {
            "version": 1,
            "matcherVersion": MATCHER_VERSION,
            "rallies": {
                "rid-a": {"1": 2, "2": 1, "3": 4, "4": 3},
                "rid-b": {"1": 1, "2": 2, "3": 3, "4": 4},
            },
        }
        out, stale = _load_canonical_per_rally(payload)
        assert stale is None
        assert out == {
            "rid-a": {1: 2, 2: 1, 3: 4, 4: 3},
            "rid-b": {1: 1, 2: 2, 3: 3, 4: 4},
        }

    def test_stale_version_invalidates(self) -> None:
        payload = {
            "version": 1,
            "matcherVersion": "v0-ancient",
            "rallies": {"rid-a": {"1": 2, "2": 1}},
        }
        out, stale = _load_canonical_per_rally(payload)
        assert out == {}
        assert stale == "v0-ancient"

    def test_missing_version_with_data_treated_as_stale(self) -> None:
        """Pre-2026-05-03 canonical maps lack matcherVersion. Treat them
        as stale so the next remap-track-ids run falls through to the
        fresh trackToPlayer instead of trusting a possibly-wrong override."""
        payload = {
            "version": 1,
            "rallies": {"rid-a": {"1": 2}},
        }
        out, stale = _load_canonical_per_rally(payload)
        assert out == {}
        assert stale == "<unset>"

    def test_missing_version_without_data_is_noop(self) -> None:
        """No rallies entries → nothing to load AND nothing to warn about."""
        payload = {"version": 1, "rallies": {}}
        out, stale = _load_canonical_per_rally(payload)
        assert out == {}
        assert stale is None

    def test_skips_empty_rally_entries(self) -> None:
        payload = {
            "version": 1,
            "matcherVersion": MATCHER_VERSION,
            "rallies": {
                "rid-a": {"1": 2},
                "rid-b": {},          # empty mapping → skipped
                "": {"1": 1},         # empty rally id → skipped
            },
        }
        out, stale = _load_canonical_per_rally(payload)
        assert stale is None
        assert out == {"rid-a": {1: 2}}


class TestSnapshotIdempotent:
    """Snapshot-based idempotency: two runs from snapshot produce identical output.

    Mirrors the new pre_remap_state_json flow in remap_track_ids_cmd:
        - Run 1 captures snapshot of pristine row state, applies mapping, writes outputs.
        - Run 2 restores from snapshot, applies same mapping, writes outputs.
        - positions / contacts / actions / primary_ids must be byte-identical.
    """

    def _make_pristine(self) -> dict[str, Any]:
        return {
            "positions": [
                {"trackId": 7, "frameNumber": 0, "x": 0.1, "y": 0.2, "width": 0.05, "height": 0.1, "confidence": 0.9},
                {"trackId": 12, "frameNumber": 0, "x": 0.3, "y": 0.4, "width": 0.05, "height": 0.1, "confidence": 0.9},
                {"trackId": 15, "frameNumber": 1, "x": 0.5, "y": 0.6, "width": 0.05, "height": 0.1, "confidence": 0.9},
                {"trackId": 23, "frameNumber": 1, "x": 0.7, "y": 0.8, "width": 0.05, "height": 0.1, "confidence": 0.9},
            ],
            "contacts": {"contacts": [{"playerTrackId": 7, "frame": 0, "playerCandidates": [[12, 0.9]]}]},
            "actions": {"actions": [{"playerTrackId": 15, "frame": 1}], "teamAssignments": {"7": "near", "12": "far"}},
            "primaryTrackIds": [7, 12, 15, 23],
        }

    def _apply_remap(self, snapshot: dict[str, Any], track_to_player: dict[int, int]) -> dict[str, Any]:
        """Replicates the per-rally body of remap_track_ids_cmd: deep-copy from
        snapshot, build full mapping (with collision shifts), apply to all four
        fields, return the resulting working copy."""
        pos = _deepcopy_json(snapshot["positions"])
        con = _deepcopy_json(snapshot["contacts"])
        act = _deepcopy_json(snapshot["actions"])
        pri = _deepcopy_json(snapshot["primaryTrackIds"])

        all_ids = {p["trackId"] for p in pos} | set(pri)
        mapping = _build_full_mapping(track_to_player, all_ids)
        _remap_positions(pos, mapping)
        _remap_contacts(con, mapping)
        _remap_actions(act, mapping)
        pri = [mapping.get(t, t) for t in pri]
        return {
            "positions": pos,
            "contacts": con,
            "actions": act,
            "primaryTrackIds": pri,
        }

    def test_two_runs_byte_identical(self) -> None:
        """Run 1 captures snapshot + applies mapping; Run 2 restores from snapshot
        + applies same mapping. Outputs must match byte-for-byte (the regression
        target the user verified on b5fb0594/r04)."""
        pristine = self._make_pristine()
        track_to_player = {7: 2, 12: 1, 15: 3, 23: 4}  # non-identity, exercises swap

        # Run 1: pre_remap_state_json IS NULL → capture snapshot from current row.
        snapshot = _capture_snapshot(
            pristine["positions"], pristine["contacts"], pristine["actions"],
            pristine["primaryTrackIds"],
        )
        run1_output = self._apply_remap(snapshot, track_to_player)

        # Snapshot itself must remain pristine (snapshot != run1_output).
        assert snapshot["positions"][0]["trackId"] == 7  # original
        assert run1_output["positions"][0]["trackId"] == 2  # remapped

        # Run 2: pre_remap_state_json EXISTS → restore from snapshot, apply mapping.
        # The snapshot variable plays the role of pre_remap_state_json read from DB.
        run2_output = self._apply_remap(snapshot, track_to_player)

        # Byte-identical via JSON round-trip (matches the DB hash check pattern).
        import json
        assert json.dumps(run1_output, sort_keys=True) == json.dumps(run2_output, sort_keys=True)

    def test_snapshot_immune_to_working_copy_mutation(self) -> None:
        """Mutating the working copy during remap must not leak back into the
        snapshot dict. Otherwise on first run we'd persist the mutated state
        instead of the pristine state, breaking idempotency."""
        pristine = self._make_pristine()
        track_to_player = {7: 2, 12: 1, 15: 3, 23: 4}

        snapshot = _capture_snapshot(
            pristine["positions"], pristine["contacts"], pristine["actions"],
            pristine["primaryTrackIds"],
        )
        snapshot_serialized_before = self._serialize(snapshot)

        # Run remap (mutates working copy).
        self._apply_remap(snapshot, track_to_player)

        # Snapshot must be unchanged.
        assert self._serialize(snapshot) == snapshot_serialized_before

    @staticmethod
    def _serialize(payload: dict[str, Any]) -> str:
        import json
        return json.dumps(payload, sort_keys=True)


class TestResolveRawMappingSource:
    """Source-priority for the raw_id → canonical_pid mapping a remap will apply.

    Regression target: 2026-05-11 incident on video 5c756c41 where 7/10 rallies
    silently lost 1–2 players on retrack. Root cause was that re-application of
    remap (snapshot restore + apply mapping) was sourcing the mapping from
    ``trackToPlayer``, which gets overwritten to a canonical-space identity at
    the end of each remap run. With raw tracker IDs outside ``{1..4}`` (e.g.
    BoT-SORT had emitted ``{1, 3, 9, 10}`` after mid-rally ID switches),
    canonical-space identity dropped 9 and 10 — they were unmapped on the
    second remap. Fix: read ``appliedFullMapping`` when ``remapApplied=true``;
    fall back to ``trackToPlayer`` only on the first run.
    """

    def test_first_run_reads_track_to_player(self) -> None:
        rally_entry: dict[str, Any] = {
            "rallyId": "test-1",
            "trackToPlayer": {"1": 1, "3": 4, "9": 2, "10": 3},
            # remapApplied not set / False → first-run path
        }
        result = _resolve_raw_mapping_source(rally_entry, None)
        assert result == {1: 1, 3: 4, 9: 2, 10: 3}

    def test_re_apply_reads_applied_full_mapping_not_track_to_player(self) -> None:
        """Regression for 2026-05-11 incident.

        After the first remap, ``trackToPlayer`` is overwritten to canonical-
        space identity ``{1:1, 2:2, 3:3, 4:4}`` but ``appliedFullMapping``
        preserves the actual raw→canonical mapping. The re-apply must source
        from ``appliedFullMapping`` — sourcing from ``trackToPlayer`` would
        drop raw track IDs 9 and 10 (not in identity) and collapse the rally
        from 4 players to 2.
        """
        rally_entry: dict[str, Any] = {
            "rallyId": "test-1",
            "trackToPlayer": {"1": 1, "2": 2, "3": 3, "4": 4},
            "appliedFullMapping": {"1": 1, "3": 4, "9": 2, "10": 3},
            "remapApplied": True,
        }
        result = _resolve_raw_mapping_source(rally_entry, None)
        assert result == {1: 1, 3: 4, 9: 2, 10: 3}

    def test_re_apply_falls_back_to_track_to_player_when_afm_missing(self) -> None:
        """Legacy rallies pre-1c4ce3a have no ``appliedFullMapping``; the
        re-apply must still produce a usable mapping by falling through to
        ``trackToPlayer``."""
        rally_entry: dict[str, Any] = {
            "rallyId": "test-1",
            "trackToPlayer": {"1": 1, "3": 4, "9": 2, "10": 3},
            "remapApplied": True,
            # appliedFullMapping absent
        }
        result = _resolve_raw_mapping_source(rally_entry, None)
        assert result == {1: 1, 3: 4, 9: 2, 10: 3}

    def test_canonical_per_rally_wins_over_applied_full_mapping(self) -> None:
        rally_entry: dict[str, Any] = {
            "rallyId": "test-1",
            "trackToPlayer": {"1": 1, "2": 2, "3": 3, "4": 4},
            "appliedFullMapping": {"1": 1, "3": 4, "9": 2, "10": 3},
            "remapApplied": True,
        }
        canonical = {1: 2, 3: 1, 9: 4, 10: 3}
        result = _resolve_raw_mapping_source(rally_entry, canonical)
        assert result == canonical

    def test_synthetic_subtrack_ids_stripped(self) -> None:
        rally_entry: dict[str, Any] = {
            "rallyId": "test-1",
            "appliedFullMapping": {"1": 1, "3": 4, "-101": 2, "-102": 3},
            "remapApplied": True,
        }
        result = _resolve_raw_mapping_source(rally_entry, None)
        assert result == {1: 1, 3: 4}

    def test_returns_empty_when_no_mapping_anywhere(self) -> None:
        rally_entry: dict[str, Any] = {"rallyId": "test-1"}
        result = _resolve_raw_mapping_source(rally_entry, None)
        assert result == {}

    def test_snake_case_track_to_player_key_supported(self) -> None:
        """Legacy snake_case key support is preserved."""
        rally_entry: dict[str, Any] = {
            "rallyId": "test-1",
            "track_to_player": {"1": 1, "3": 4, "9": 2, "10": 3},
        }
        result = _resolve_raw_mapping_source(rally_entry, None)
        assert result == {1: 1, 3: 4, 9: 2, 10: 3}


class TestCheckSnapshotCoveredByMapping:
    """Fail-closed invariant for remap-track-ids.

    Regression target: ff175026 case where ``pre_remap_state_json.primaryTrackIds``
    held raw BoT-SORT IDs like ``[1, 5, 8, 11]`` while ``appliedFullMapping`` had
    been corrupted to canonical-space identity ``{1:1, 2:2, 3:3, 4:4}`` by an
    older pipeline bug. The snapshot-restore + apply loop then silently dropped
    tracks 5, 8, 11 from positions and primary_track_ids on every re-analyze.

    The fix (commit 5a521701, 2026-05-11) made remap prefer ``appliedFullMapping``
    over ``trackToPlayer`` — but does not protect against an already-corrupted
    ``appliedFullMapping``. This invariant adds a fail-closed guard: when the
    snapshot's real primary tracks aren't covered by the mapping the remap is
    about to apply, refuse to mutate. Better to leave the rally in its current
    canonical state than to silently destroy it.
    """

    def test_ok_when_snapshot_primary_subset_of_mapping(self) -> None:
        snapshot_primary = [1, 5, 8, 11]
        raw_mapping = {1: 1, 5: 2, 8: 3, 11: 4, 10: 101}
        ok, missing = _check_snapshot_covered_by_mapping(snapshot_primary, raw_mapping)
        assert ok is True
        assert missing == []

    def test_violation_when_mapping_missing_primary_ids(self) -> None:
        """The ff175026 / rally 10 shape — appliedFullMapping corrupted to identity."""
        snapshot_primary = [1, 5, 8, 11]
        raw_mapping = {1: 1, 2: 2, 3: 3, 4: 4}
        ok, missing = _check_snapshot_covered_by_mapping(snapshot_primary, raw_mapping)
        assert ok is False
        assert sorted(missing) == [5, 8, 11]

    def test_ok_when_snapshot_empty(self) -> None:
        """No primary to protect → nothing to check."""
        ok, missing = _check_snapshot_covered_by_mapping([], {1: 1})
        assert ok is True
        assert missing == []

    def test_ok_when_snapshot_none(self) -> None:
        """Legacy rallies that never had a snapshot captured → nothing to check."""
        ok, missing = _check_snapshot_covered_by_mapping(None, {1: 1})
        assert ok is True
        assert missing == []

    def test_violation_when_mapping_empty(self) -> None:
        snapshot_primary = [1, 5, 8, 11]
        ok, missing = _check_snapshot_covered_by_mapping(snapshot_primary, {})
        assert ok is False
        assert sorted(missing) == [1, 5, 8, 11]

    def test_negative_sentinel_ids_in_snapshot_ignored(self) -> None:
        """BoT-SORT's ``-1`` sentinel can leak into legacy snapshots; it must
        not trigger a false violation."""
        snapshot_primary = [-1, 1, 5]
        raw_mapping = {1: 1, 5: 2}
        ok, missing = _check_snapshot_covered_by_mapping(snapshot_primary, raw_mapping)
        assert ok is True
        assert missing == []
