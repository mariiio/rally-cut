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
from rallycut.tracking._within_rally_id_switch import (
    NUM_WINDOWS,
    _detect_split_candidates,
    _reassign_split_halves,
)
from rallycut.tracking.player_features import TrackAppearanceStats


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
