"""Tests for pid_invariants module."""

from __future__ import annotations

from rallycut.tracking.pid_invariants import check_i1_primary_set_size


class TestCheckI1PrimarySetSize:
    def test_clean_size_4_passes(self) -> None:
        violations = check_i1_primary_set_size(rally_id="r1", primary_track_ids=[3, 7, 12, 15])
        assert violations == []

    def test_clean_size_0_passes(self) -> None:
        # filter disabled is allowed
        violations = check_i1_primary_set_size(rally_id="r1", primary_track_ids=[])
        assert violations == []

    def test_size_3_fails(self) -> None:
        violations = check_i1_primary_set_size(rally_id="r1", primary_track_ids=[3, 7, 12])
        assert len(violations) == 1
        assert violations[0].invariant == "I-1"
        assert violations[0].rally_id == "r1"
        assert "size 3" in violations[0].detail

    def test_size_5_fails(self) -> None:
        violations = check_i1_primary_set_size(rally_id="r2", primary_track_ids=[3, 7, 12, 15, 22])
        assert len(violations) == 1
        assert violations[0].invariant == "I-1"
