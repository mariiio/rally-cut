"""Tests for the within-rally appearance-based ID-switch detector.

Tests use monkeypatched `_pairwise_cost` so the test fixtures encode
APPEARANCE COSTS DIRECTLY — independent of how `compute_track_similarity`
maps from histogram fixtures to cost. This isolates the detector logic
(gate, changepoint selection, re-assignment) from the underlying cost
function's behavior.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from rallycut.tracking import _within_rally_id_switch as wris
from rallycut.tracking._subtrack import SubTrackCandidate
from rallycut.tracking._within_rally_id_switch import (
    NUM_WINDOWS,
    _clip_overlapping_sub_tracks,
    _detect_split_candidates,
    _reassign_split_halves,
)
from rallycut.tracking.player_features import TrackAppearanceStats
from rallycut.tracking.player_tracker import PlayerPosition


@dataclass
class _StubWindow:
    """Minimal stand-in for `_WindowStats`. The detector only uses
    `.window_idx` and `.stats` (passed to `_pairwise_cost`). We use the
    `track_id` field on the stats as a key so a monkeypatched cost
    function can look up predetermined pairwise costs."""
    track_id: int
    window_idx: int
    f_start: int
    f_end: int
    n_frames: int
    stats: TrackAppearanceStats


def _empty_stats(key: int) -> TrackAppearanceStats:
    """Build a TrackAppearanceStats whose only meaningful field is
    `track_id` — used as a lookup key by the monkeypatched cost."""
    return TrackAppearanceStats(track_id=key, features=[])


def _make_window(tid: int, w_idx: int, key: int) -> _StubWindow:
    return _StubWindow(
        track_id=tid, window_idx=w_idx,
        f_start=w_idx * 100, f_end=(w_idx + 1) * 100 - 1,
        n_frames=100,
        stats=_empty_stats(key),
    )


def _patch_costs(
    monkeypatch: pytest.MonkeyPatch,
    cost_table: dict[tuple[int, int], float],
) -> None:
    """Install a cost function that looks up (key_a, key_b) ordered."""
    def _cost(a: TrackAppearanceStats, b: TrackAppearanceStats) -> float:
        if a.track_id == b.track_id:
            return 0.0
        k = (min(a.track_id, b.track_id), max(a.track_id, b.track_id))
        return cost_table[k]
    monkeypatch.setattr(wris, "_pairwise_cost", _cost)


class TestDetectSplitCandidates:
    """Relative-gate (`RELATIVE_GATE_K * median(inter-track)`) means a
    track is a candidate only when its own window-to-window appearance
    discontinuity exceeds the typical between-track distance — adapts
    to videos with similar vs distinct uniforms."""

    def test_clean_track_does_not_trigger(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Track 1: all 3 windows use key=10 (intra-window cost = 0).
        # Track 2: all 3 windows use key=20.
        # Inter-track cost = 0.4. Intra = 0 → way below gate.
        by_track = {
            1: [_make_window(1, w, key=10) for w in range(NUM_WINDOWS)],
            2: [_make_window(2, w, key=20) for w in range(NUM_WINDOWS)],
        }
        _patch_costs(monkeypatch, {(10, 20): 0.4})
        out = _detect_split_candidates(by_track)
        assert out == {}

    def test_id_switch_track_triggers_with_correct_changepoint(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Track 1: W0+W1 = key 10 (cyan), W2 = key 11 (different player).
        # Cost(10, 11) = 0.6 (HIGH — clear switch).
        # Other tracks have key 20-40, inter-track cost = 0.4.
        # Median inter-track cost ≈ 0.4; gate = 0.95 * 0.4 = 0.38.
        # Intra-max = 0.6 > 0.38 → triggers.
        by_track = {
            1: [_make_window(1, 0, key=10),
                _make_window(1, 1, key=10),
                _make_window(1, 2, key=11)],
            2: [_make_window(2, w, key=20) for w in range(NUM_WINDOWS)],
            3: [_make_window(3, w, key=30) for w in range(NUM_WINDOWS)],
            4: [_make_window(4, w, key=40) for w in range(NUM_WINDOWS)],
        }
        _patch_costs(monkeypatch, {
            (10, 11): 0.6,
            (10, 20): 0.4, (10, 30): 0.4, (10, 40): 0.4,
            (11, 20): 0.4, (11, 30): 0.4, (11, 40): 0.4,
            (20, 30): 0.4, (20, 40): 0.4, (30, 40): 0.4,
        })
        out = _detect_split_candidates(by_track)
        assert 1 in out, f"Track 1 should be flagged; got {out}"
        _intra_max, boundary = out[1]
        assert boundary == 2, f"Expected boundary at W1/W2; got {boundary}"

    def test_id_switch_in_first_window_picks_correct_changepoint(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Track 1: W0 = key 11 (different), W1+W2 = key 10 (same).
        # Boundary should be at W0/W1 = 1.
        by_track = {
            1: [_make_window(1, 0, key=11),
                _make_window(1, 1, key=10),
                _make_window(1, 2, key=10)],
            2: [_make_window(2, w, key=20) for w in range(NUM_WINDOWS)],
            3: [_make_window(3, w, key=30) for w in range(NUM_WINDOWS)],
            4: [_make_window(4, w, key=40) for w in range(NUM_WINDOWS)],
        }
        _patch_costs(monkeypatch, {
            (10, 11): 0.6,
            (10, 20): 0.4, (10, 30): 0.4, (10, 40): 0.4,
            (11, 20): 0.4, (11, 30): 0.4, (11, 40): 0.4,
            (20, 30): 0.4, (20, 40): 0.4, (30, 40): 0.4,
        })
        out = _detect_split_candidates(by_track)
        assert 1 in out
        _intra_max, boundary = out[1]
        assert boundary == 1, f"Expected boundary at W0/W1; got {boundary}"

    def test_borderline_intra_below_relative_gate_does_not_trigger(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Intra-max = 0.30, inter-track median = 0.40.
        # Gate = 0.95 * 0.40 = 0.38. Intra 0.30 < 0.38 → no fire.
        by_track = {
            1: [_make_window(1, 0, key=10),
                _make_window(1, 1, key=10),
                _make_window(1, 2, key=11)],
            2: [_make_window(2, w, key=20) for w in range(NUM_WINDOWS)],
            3: [_make_window(3, w, key=30) for w in range(NUM_WINDOWS)],
            4: [_make_window(4, w, key=40) for w in range(NUM_WINDOWS)],
        }
        _patch_costs(monkeypatch, {
            (10, 11): 0.30,
            (10, 20): 0.4, (10, 30): 0.4, (10, 40): 0.4,
            (11, 20): 0.4, (11, 30): 0.4, (11, 40): 0.4,
            (20, 30): 0.4, (20, 40): 0.4, (30, 40): 0.4,
        })
        out = _detect_split_candidates(by_track)
        assert 1 not in out, f"Borderline shouldn't fire; got {out}"

    def test_too_few_tracks_returns_empty(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        by_track = {
            1: [_make_window(1, w, key=10) for w in range(NUM_WINDOWS)],
        }
        _patch_costs(monkeypatch, {})
        out = _detect_split_candidates(by_track)
        assert out == {}

    def test_track_with_fewer_than_n_windows_skipped(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        by_track = {
            1: [_make_window(1, 0, key=10),
                _make_window(1, 1, key=11)],  # only 2 windows
            2: [_make_window(2, w, key=20) for w in range(NUM_WINDOWS)],
            3: [_make_window(3, w, key=30) for w in range(NUM_WINDOWS)],
        }
        _patch_costs(monkeypatch, {
            (10, 11): 0.6, (10, 20): 0.4, (10, 30): 0.4,
            (11, 20): 0.4, (11, 30): 0.4, (20, 30): 0.4,
        })
        out = _detect_split_candidates(by_track)
        assert 1 not in out


class TestReassignSplitHalves:
    """The re-assignment Hungarian decides which half keeps the parent
    PID and which gets re-assigned. A clean re-assignment requires the
    halves to be distinguishable from each other AND from at least one
    other track."""

    def test_returns_distinct_pids(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # First half (key=1) matches T11 (key=30) at cost 0.6 (far) and
        # T10 (key=20) at cost 0.6 (far) → distinctive vs others.
        # Second half (key=2) matches T10 (key=20) at cost 0.1 (close).
        # → First half keeps parent PID (1); second re-assigns to T10's
        # PID (4).
        first = _empty_stats(1)
        second = _empty_stats(2)
        other_stats = {10: _empty_stats(20), 11: _empty_stats(30)}
        other_pids = {10: 4, 11: 3}

        _patch_costs(monkeypatch, {
            (1, 2): 0.5,  # Halves differ — passes the inter_half sanity gate.
            (1, 20): 0.6, (1, 30): 0.6,
            (2, 20): 0.1, (2, 30): 0.6,
        })
        out = _reassign_split_halves(
            parent_tid=5, parent_pid=1,
            first_half_stats=first, second_half_stats=second,
            other_track_pids=other_pids,
            other_track_stats=other_stats,
        )
        assert out is not None
        first_pid, second_pid = out
        assert first_pid != second_pid
        assert second_pid == 4, f"Second half should map to PID 4; got {second_pid}"

    def test_no_other_tracks_returns_none(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        first = _empty_stats(1)
        second = _empty_stats(2)
        _patch_costs(monkeypatch, {})
        out = _reassign_split_halves(
            parent_tid=5, parent_pid=1,
            first_half_stats=first, second_half_stats=second,
            other_track_pids={}, other_track_stats={},
        )
        assert out is None

    def test_both_halves_prefer_same_pid_returns_none(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Both halves equally close to T10 → both prefer the same
        # re-assignment → no useful split.
        same = _empty_stats(1)
        other_stats = {10: _empty_stats(20)}
        _patch_costs(monkeypatch, {(1, 20): 0.5})
        out = _reassign_split_halves(
            parent_tid=5, parent_pid=1,
            first_half_stats=same, second_half_stats=same,
            other_track_pids={10: 4}, other_track_stats=other_stats,
        )
        assert out is None


def _make_subtrack(
    parent: int, segment: int, f_start: int, f_end: int, pid: int,
) -> SubTrackCandidate:
    return SubTrackCandidate(
        parent_track_id=parent,
        segment_index=segment,
        f_start=f_start,
        f_end=f_end,
        appearance_stats=_empty_stats(parent * 100 + segment),
        aggregated_argmax_pid=pid,
    )


def _make_pos(track_id: int, frame: int) -> PlayerPosition:
    return PlayerPosition(
        track_id=track_id, frame_number=frame,
        x=0.5, y=0.5, width=0.1, height=0.2, confidence=0.9,
    )


class TestClipOverlappingSubTracks:
    """Phase 2: cross-track same-PID overlap deduplication.

    The bug pattern: Phase 1 splits T1 into halves; T1's first-half
    sub-track gets PID K via re-Hungarian; another track T8 already has
    PID K via the matcher's whole-track assignment; T1-first-half and
    T8 overlap in some frame range. Without dedup, two bboxes labeled
    PID K render in the overlap window — visible identity duplication.
    """

    def test_subtrack_clipped_when_other_track_overlaps_at_end(self) -> None:
        # The 09553ef1 case: T1-seg0 = [0, 209] PID 1; T8 = [181, 315]
        # PID 1. They overlap in [181, 209]. Sub-track yields → clipped
        # to [0, 180].
        emitted = [_make_subtrack(parent=1, segment=0,
                                  f_start=0, f_end=209, pid=1)]
        track_pids = {1: 2, 8: 1}  # T1's matcher pid, T8's pid
        positions = {
            1: [_make_pos(1, f) for f in range(0, 316)],
            8: [_make_pos(8, f) for f in range(181, 316)],
        }
        out = _clip_overlapping_sub_tracks(
            emitted=emitted, track_pids=track_pids,
            by_track_positions=positions, rally_id="test",
        )
        assert len(out) == 1
        assert out[0].f_start == 0
        assert out[0].f_end == 180  # T8 starts at 181

    def test_subtrack_clipped_when_other_track_overlaps_at_start(self) -> None:
        # T1-seg1 = [100, 200] PID 1; T8 = [0, 150] PID 1.
        # Overlap [100, 150]. Sub-track clipped to [151, 200].
        emitted = [_make_subtrack(parent=1, segment=1,
                                  f_start=100, f_end=200, pid=1)]
        track_pids = {1: 2, 8: 1}
        positions = {
            1: [_make_pos(1, f) for f in range(0, 201)],
            8: [_make_pos(8, f) for f in range(0, 151)],
        }
        out = _clip_overlapping_sub_tracks(
            emitted=emitted, track_pids=track_pids,
            by_track_positions=positions, rally_id="test",
        )
        assert len(out) == 1
        assert out[0].f_start == 151
        assert out[0].f_end == 200

    def test_subtrack_dropped_when_fully_overlapped(self) -> None:
        # Sub-track fully inside another track's range with same PID.
        emitted = [_make_subtrack(parent=1, segment=0,
                                  f_start=100, f_end=200, pid=1)]
        track_pids = {1: 2, 8: 1}
        positions = {
            1: [_make_pos(1, f) for f in range(0, 316)],
            8: [_make_pos(8, f) for f in range(0, 316)],
        }
        out = _clip_overlapping_sub_tracks(
            emitted=emitted, track_pids=track_pids,
            by_track_positions=positions, rally_id="test",
        )
        # Sub-track gets dropped entirely.
        assert out == []

    def test_subtrack_split_into_two_when_other_track_in_middle(self) -> None:
        # T1-seg0 = [0, 300] PID 1; T8 = [100, 200] PID 1.
        # Sub-track must split into [0, 99] and [201, 300].
        emitted = [_make_subtrack(parent=1, segment=0,
                                  f_start=0, f_end=300, pid=1)]
        track_pids = {1: 2, 8: 1}
        positions = {
            1: [_make_pos(1, f) for f in range(0, 301)],
            8: [_make_pos(8, f) for f in range(100, 201)],
        }
        out = _clip_overlapping_sub_tracks(
            emitted=emitted, track_pids=track_pids,
            by_track_positions=positions, rally_id="test",
        )
        assert len(out) == 2
        ranges = sorted((s.f_start, s.f_end) for s in out)
        assert ranges == [(0, 99), (201, 300)]
        # Synthetic track ids must be distinct (uses segment_index).
        assert out[0].synthetic_track_id != out[1].synthetic_track_id

    def test_no_overlap_passes_through(self) -> None:
        # Sub-track and other-track frame ranges disjoint → no clip.
        emitted = [_make_subtrack(parent=1, segment=0,
                                  f_start=0, f_end=100, pid=1)]
        track_pids = {1: 2, 8: 1}
        positions = {
            1: [_make_pos(1, f) for f in range(0, 101)],
            8: [_make_pos(8, f) for f in range(200, 301)],
        }
        out = _clip_overlapping_sub_tracks(
            emitted=emitted, track_pids=track_pids,
            by_track_positions=positions, rally_id="test",
        )
        assert len(out) == 1
        assert (out[0].f_start, out[0].f_end) == (0, 100)

    def test_other_track_with_different_pid_no_clip(self) -> None:
        # PID mismatch → no conflict even if frame ranges overlap.
        emitted = [_make_subtrack(parent=1, segment=0,
                                  f_start=0, f_end=200, pid=1)]
        track_pids = {1: 2, 8: 3}  # T8 has PID 3, sub-track has PID 1
        positions = {
            1: [_make_pos(1, f) for f in range(0, 201)],
            8: [_make_pos(8, f) for f in range(50, 151)],
        }
        out = _clip_overlapping_sub_tracks(
            emitted=emitted, track_pids=track_pids,
            by_track_positions=positions, rally_id="test",
        )
        assert len(out) == 1
        assert (out[0].f_start, out[0].f_end) == (0, 200)

    def test_parent_track_excluded_from_other_set(self) -> None:
        # The split parent T1 itself shouldn't count as an "other track"
        # against its own sub-track. (T1's matcher PID was 1, but its
        # sub-track gets PID 1 too — without this exclusion the
        # sub-track would be self-conflicting and dropped erroneously.)
        emitted = [_make_subtrack(parent=1, segment=0,
                                  f_start=0, f_end=200, pid=1)]
        track_pids = {1: 1}  # T1 mapped to PID 1 (the same as its sub-track)
        positions = {
            1: [_make_pos(1, f) for f in range(0, 201)],
        }
        out = _clip_overlapping_sub_tracks(
            emitted=emitted, track_pids=track_pids,
            by_track_positions=positions, rally_id="test",
        )
        assert len(out) == 1
        assert (out[0].f_start, out[0].f_end) == (0, 200)

    def test_multiple_other_tracks_merged_correctly(self) -> None:
        # Two other tracks with same PID, both in sub-track's range.
        # Their conflict ranges should merge correctly when overlapping.
        emitted = [_make_subtrack(parent=1, segment=0,
                                  f_start=0, f_end=300, pid=1)]
        track_pids = {1: 2, 8: 1, 9: 1}
        positions = {
            1: [_make_pos(1, f) for f in range(0, 301)],
            8: [_make_pos(8, f) for f in range(100, 201)],
            9: [_make_pos(9, f) for f in range(180, 251)],  # overlaps T8
        }
        out = _clip_overlapping_sub_tracks(
            emitted=emitted, track_pids=track_pids,
            by_track_positions=positions, rally_id="test",
        )
        # Merged conflict range = [100, 250]. Kept fragments: [0, 99]
        # and [251, 300].
        assert len(out) == 2
        ranges = sorted((s.f_start, s.f_end) for s in out)
        assert ranges == [(0, 99), (251, 300)]
