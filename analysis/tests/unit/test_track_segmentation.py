"""Tests for within-track appearance segmentation (Task 2-5)."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from rallycut.cli.commands.match_players import _build_per_frame_pid_map
from rallycut.tracking._subtrack import SubTrackCandidate
from rallycut.tracking.match_tracker import (
    SEGMENT_MIN_PER_SEGMENT_MARGIN,
    MatchPlayerTracker,
)
from rallycut.tracking.player_features import TrackAppearanceStats
from rallycut.tracking.player_tracker import PlayerPosition


def _make_stats(track_id: int) -> TrackAppearanceStats:
    return TrackAppearanceStats(track_id=track_id)


def test_subtrack_candidate_carries_parent_and_window() -> None:
    parent_stats = _make_stats(track_id=2)
    sub = SubTrackCandidate(
        parent_track_id=2,
        segment_index=0,
        f_start=0,
        f_end=110,
        appearance_stats=parent_stats,
        aggregated_argmax_pid=2,
        aggregated_margin=0.18,
    )
    assert sub.parent_track_id == 2
    assert sub.segment_index == 0
    assert sub.f_start == 0
    assert sub.f_end == 110
    assert sub.appearance_stats is parent_stats
    # Formula: -1000 * (segment_index + 1) - parent_track_id - 2
    # For parent=2, seg=0: -1000 * 1 - 2 - 2 = -1004
    assert sub.synthetic_track_id == -1004
    assert sub.aggregated_argmax_pid == 2
    assert sub.aggregated_margin == pytest.approx(0.18)


def test_subtrack_candidate_synthetic_track_id_is_unique_per_segment() -> None:
    sub_a = SubTrackCandidate(
        parent_track_id=2,
        segment_index=0,
        f_start=0,
        f_end=100,
        appearance_stats=_make_stats(2),
    )
    sub_b = SubTrackCandidate(
        parent_track_id=2,
        segment_index=1,
        f_start=100,
        f_end=200,
        appearance_stats=_make_stats(2),
    )
    assert sub_a.synthetic_track_id != sub_b.synthetic_track_id
    # Synthetic ids must not collide with real positive track_ids
    assert sub_a.synthetic_track_id < 0
    assert sub_b.synthetic_track_id < 0


def test_subtrack_candidate_overlap_detection() -> None:
    a = SubTrackCandidate(
        parent_track_id=2,
        segment_index=0,
        f_start=0,
        f_end=100,
        appearance_stats=_make_stats(2),
    )
    b = SubTrackCandidate(
        parent_track_id=3,
        segment_index=0,
        f_start=50,
        f_end=150,
        appearance_stats=_make_stats(3),
    )
    c = SubTrackCandidate(
        parent_track_id=4,
        segment_index=0,
        f_start=200,
        f_end=300,
        appearance_stats=_make_stats(4),
    )
    assert a.overlaps(b)
    assert b.overlaps(a)
    assert not a.overlaps(c)
    assert not c.overlaps(b)


# ---------------------------------------------------------------------------
# Task 3: _segment_tracks_by_appearance detection logic
# ---------------------------------------------------------------------------


def _fake_position(track_id: int, frame: int, x: float = 0.5, y: float = 0.5) -> PlayerPosition:
    return PlayerPosition(
        track_id=track_id,
        frame_number=frame,
        x=x, y=y, width=0.05, height=0.15,
        confidence=0.9,
    )


class _FakeClassifier:
    """Stub for PlayerReIDClassifier — returns scripted probabilities by frame range."""

    def __init__(self, scripts: dict[tuple[int, int, int], dict[int, float]]):
        # Key: (parent_track_id, frame_start, frame_end) -> probs dict.
        self.scripts = scripts
        self.player_ids = sorted({pid for probs in scripts.values() for pid in probs})
        self._trained = True

    @property
    def is_trained(self) -> bool:
        return self._trained

    def predict_single(self, crop: np.ndarray) -> dict[int, float]:
        track_id, frame = int(crop[0, 0, 0]), int(crop[0, 0, 1])
        for (tid, fs, fe), probs in self.scripts.items():
            if tid == track_id and fs <= frame <= fe:
                return probs
        return {pid: 1.0 / len(self.player_ids) for pid in self.player_ids}

    def predict(self, crops: list[np.ndarray]) -> list[dict[int, float]]:
        return [self.predict_single(c) for c in crops]


def _stub_crop(track_id: int, frame: int) -> np.ndarray:
    arr = np.zeros((20, 20, 3), dtype=np.uint8)
    arr[0, 0, 0] = track_id
    arr[0, 0, 1] = frame
    return arr


def test_segment_consistent_track_does_not_split() -> None:
    tracker = MatchPlayerTracker.__new__(MatchPlayerTracker)
    tracker.frozen_player_ids = {1, 2, 3, 4}
    n_frames = 200
    positions = [_fake_position(1, f, x=0.5 + 0.001 * f) for f in range(n_frames)]
    classifier = _FakeClassifier({
        (1, 0, n_frames - 1): {1: 0.85, 2: 0.05, 3: 0.05, 4: 0.05},
    })
    track_stats = {1: TrackAppearanceStats(track_id=1)}
    sub_tracks = tracker._segment_tracks_by_appearance(
        track_ids=[1],
        track_stats=track_stats,
        positions=positions,
        classifier=classifier,
        crop_extractor=lambda tid, frame: _stub_crop(tid, frame),
    )
    assert len(sub_tracks) == 0  # No splits, fall through to original tracks.


def test_segment_clear_flip_with_strong_margins_splits() -> None:
    tracker = MatchPlayerTracker.__new__(MatchPlayerTracker)
    tracker.frozen_player_ids = {1, 2, 3, 4}
    n_frames = 240
    positions = [_fake_position(2, f) for f in range(n_frames)]
    classifier = _FakeClassifier({
        (2, 0, 119): {1: 0.10, 2: 0.78, 3: 0.07, 4: 0.05},
        (2, 120, 239): {1: 0.82, 2: 0.08, 3: 0.06, 4: 0.04},
    })
    track_stats = {2: TrackAppearanceStats(track_id=2)}
    sub_tracks = tracker._segment_tracks_by_appearance(
        track_ids=[2], track_stats=track_stats, positions=positions,
        classifier=classifier, crop_extractor=_stub_crop,
    )
    assert len(sub_tracks) == 2
    assert sub_tracks[0].parent_track_id == 2 and sub_tracks[0].segment_index == 0
    assert sub_tracks[0].aggregated_argmax_pid == 2
    assert sub_tracks[1].parent_track_id == 2 and sub_tracks[1].segment_index == 1
    assert sub_tracks[1].aggregated_argmax_pid == 1
    assert sub_tracks[0].aggregated_margin is not None
    assert sub_tracks[0].aggregated_margin >= SEGMENT_MIN_PER_SEGMENT_MARGIN
    assert sub_tracks[1].aggregated_margin is not None
    assert sub_tracks[1].aggregated_margin >= SEGMENT_MIN_PER_SEGMENT_MARGIN


def test_segment_weak_post_segment_abstains() -> None:
    tracker = MatchPlayerTracker.__new__(MatchPlayerTracker)
    tracker.frozen_player_ids = {1, 2, 3, 4}
    n_frames = 240
    positions = [_fake_position(2, f) for f in range(n_frames)]
    classifier = _FakeClassifier({
        (2, 0, 119): {1: 0.10, 2: 0.78, 3: 0.07, 4: 0.05},  # strong
        (2, 120, 239): {1: 0.30, 2: 0.28, 3: 0.22, 4: 0.20},  # weak — margin ~0.02
    })
    track_stats = {2: TrackAppearanceStats(track_id=2)}
    sub_tracks = tracker._segment_tracks_by_appearance(
        track_ids=[2], track_stats=track_stats, positions=positions,
        classifier=classifier, crop_extractor=_stub_crop,
    )
    assert len(sub_tracks) == 0  # Abstain — post segment too weak.


# ---------------------------------------------------------------------------
# Task 4: flag-gated wrapper + direct sub-track assignment dispatch
# ---------------------------------------------------------------------------


def test_track_split_flag_off_is_byte_identical(monkeypatch):
    """ENABLE_REF_CROP_TRACK_SPLIT=0 → splitter is never called."""
    monkeypatch.delenv("ENABLE_REF_CROP_TRACK_SPLIT", raising=False)
    tracker = MatchPlayerTracker.__new__(MatchPlayerTracker)
    tracker.frozen_player_ids = {1, 2, 3, 4}
    def boom(*args, **kwargs):
        raise AssertionError("splitter should not run when flag is 0")
    tracker._segment_tracks_by_appearance = boom  # type: ignore[method-assign]
    result = tracker._maybe_segment_tracks_by_appearance(
        track_ids=[1, 2], track_stats={}, positions=[],
        classifier=MagicMock(), crop_extractor=lambda tid, f: None,
    )
    assert result == []


def test_track_split_flag_on_calls_splitter(monkeypatch):
    monkeypatch.setenv("ENABLE_REF_CROP_TRACK_SPLIT", "1")
    tracker = MatchPlayerTracker.__new__(MatchPlayerTracker)
    tracker.frozen_player_ids = {1, 2, 3, 4}
    sentinel = []
    def fake_segment(track_ids, track_stats, positions, classifier, crop_extractor):
        sentinel.append(True)
        return []
    tracker._segment_tracks_by_appearance = fake_segment  # type: ignore[method-assign]
    tracker._maybe_segment_tracks_by_appearance(
        track_ids=[1, 2], track_stats={}, positions=[],
        classifier=MagicMock(spec=["is_trained", "predict"], is_trained=True),
        crop_extractor=lambda tid, f: None,
    )
    assert sentinel == [True]


def test_track_split_no_classifier_is_noop(monkeypatch):
    monkeypatch.setenv("ENABLE_REF_CROP_TRACK_SPLIT", "1")
    tracker = MatchPlayerTracker.__new__(MatchPlayerTracker)
    tracker.frozen_player_ids = {1, 2, 3, 4}
    result = tracker._maybe_segment_tracks_by_appearance(
        track_ids=[1, 2], track_stats={}, positions=[], classifier=None,
        crop_extractor=lambda tid, f: None,
    )
    assert result == []


def test_track_split_no_frozen_profiles_is_noop(monkeypatch):
    monkeypatch.setenv("ENABLE_REF_CROP_TRACK_SPLIT", "1")
    tracker = MatchPlayerTracker.__new__(MatchPlayerTracker)
    tracker.frozen_player_ids = set()  # No frozen ref-crop profiles
    classifier = MagicMock(spec=["is_trained"], is_trained=True)
    result = tracker._maybe_segment_tracks_by_appearance(
        track_ids=[1, 2], track_stats={}, positions=[], classifier=classifier,
        crop_extractor=lambda tid, f: None,
    )
    assert result == []


def test_apply_subtrack_assignments_with_no_conflict():
    """Helper that takes sub_tracks and returns (direct_assignments, remaining_track_ids, remaining_pids)."""
    tracker = MatchPlayerTracker.__new__(MatchPlayerTracker)
    sub_a = SubTrackCandidate(parent_track_id=2, segment_index=0, f_start=0, f_end=100,
                              appearance_stats=TrackAppearanceStats(track_id=2),
                              aggregated_argmax_pid=2, aggregated_margin=0.20)
    sub_b = SubTrackCandidate(parent_track_id=2, segment_index=1, f_start=100, f_end=240,
                              appearance_stats=TrackAppearanceStats(track_id=2),
                              aggregated_argmax_pid=1, aggregated_margin=0.40)
    top_tracks = [1, 2, 3, 4]
    all_pids = [1, 2, 3, 4]
    direct, remaining_tracks, remaining_pids = tracker._apply_subtrack_assignments(
        sub_tracks=[sub_a, sub_b], top_tracks=top_tracks, all_pids=all_pids,
    )
    assert direct == {sub_a.synthetic_track_id: 2, sub_b.synthetic_track_id: 1}
    # Parent tid=2 was split → removed from remaining real tracks.
    assert set(remaining_tracks) == {1, 3, 4}
    # pid 1 and 2 claimed by sub-tracks → only 3 and 4 remain for Hungarian.
    assert set(remaining_pids) == {3, 4}


def test_apply_subtrack_assignments_with_conflict_keeps_higher_margin():
    """Two sub-tracks claiming the same pid: higher-margin wins direct assignment;
    loser gets dropped (its frames will go unlabeled per Task 5's frame-level pass)."""
    tracker = MatchPlayerTracker.__new__(MatchPlayerTracker)
    sub_a = SubTrackCandidate(parent_track_id=2, segment_index=0, f_start=0, f_end=100,
                              appearance_stats=TrackAppearanceStats(track_id=2),
                              aggregated_argmax_pid=1, aggregated_margin=0.10)
    sub_b = SubTrackCandidate(parent_track_id=3, segment_index=0, f_start=50, f_end=150,
                              appearance_stats=TrackAppearanceStats(track_id=3),
                              aggregated_argmax_pid=1, aggregated_margin=0.40)
    top_tracks = [1, 2, 3, 4]
    all_pids = [1, 2, 3, 4]
    direct, remaining_tracks, remaining_pids = tracker._apply_subtrack_assignments(
        sub_tracks=[sub_a, sub_b], top_tracks=top_tracks, all_pids=all_pids,
    )
    # Higher-margin sub_b wins pid=1.
    assert direct.get(sub_b.synthetic_track_id) == 1
    # Lower-margin sub_a is dropped from direct.
    assert sub_a.synthetic_track_id not in direct
    # Parents 2 and 3 are removed from remaining (both had splits).
    assert set(remaining_tracks) == {1, 4}
    # pid 1 claimed; pids 2/3/4 remain.
    assert set(remaining_pids) == {2, 3, 4}


# ---------------------------------------------------------------------------
# Task 5: per-frame pid map translator (CLI-side helper)
# ---------------------------------------------------------------------------


def _make_sub(parent_tid: int, seg_idx: int, f_start: int, f_end: int,
              argmax_pid: int, margin: float) -> SubTrackCandidate:
    return SubTrackCandidate(
        parent_track_id=parent_tid, segment_index=seg_idx,
        f_start=f_start, f_end=f_end,
        appearance_stats=TrackAppearanceStats(track_id=parent_tid),
        aggregated_argmax_pid=argmax_pid, aggregated_margin=margin,
    )


def test_per_frame_pid_map_real_tracks_only():
    """No sub-tracks: every (real_tid, anything) maps to the assigned pid."""
    track_to_player = {1: 1, 2: 2, 3: 3, 4: 4}
    pf = _build_per_frame_pid_map(track_to_player, sub_tracks=[])
    # Real-track entries use the (-1) sentinel for "any frame"
    assert pf.get((1, -1)) == 1
    assert pf.get((2, -1)) == 2
    assert pf.get((3, -1)) == 3
    assert pf.get((4, -1)) == 4


def test_per_frame_pid_map_split_track_gives_per_segment_pids():
    """Split parent: pre-segment gets pid_pre, post-segment gets pid_post."""
    sub_pre = _make_sub(2, 0, 0, 99, argmax_pid=2, margin=0.20)
    sub_post = _make_sub(2, 1, 100, 240, argmax_pid=1, margin=0.40)
    track_to_player = {
        sub_pre.synthetic_track_id: 2,
        sub_post.synthetic_track_id: 1,
        # other (un-split) real tracks
        3: 3,
        4: 4,
    }
    pf = _build_per_frame_pid_map(track_to_player, sub_tracks=[sub_pre, sub_post])
    # Pre-segment frames map to pid=2
    assert pf.get((2, 0)) == 2
    assert pf.get((2, 50)) == 2
    assert pf.get((2, 99)) == 2
    # Post-segment frames map to pid=1
    assert pf.get((2, 100)) == 1
    assert pf.get((2, 200)) == 1
    assert pf.get((2, 240)) == 1
    # Non-split real tracks still use the sentinel
    assert pf.get((3, -1)) == 3
    assert pf.get((4, -1)) == 4
    # Frames outside both segments aren't mapped (would happen if positions exist
    # outside [f_start, f_end] of any sub-track for parent_tid=2, but Task 3
    # ensures sub-tracks cover all of the parent's frames).
    # Verify there's no sentinel for the split parent (its pid is per-segment, not constant).
    assert pf.get((2, -1)) is None


def test_per_frame_pid_map_lost_subtrack_leaves_frames_unlabeled():
    """If a sub-track is dropped from track_to_player (conflict loser), its
    frames are NOT in the per-frame map → CLI writes no label for them."""
    sub_pre = _make_sub(2, 0, 0, 99, argmax_pid=1, margin=0.10)
    sub_post = _make_sub(2, 1, 100, 240, argmax_pid=1, margin=0.40)
    # _apply_subtrack_assignments would keep only sub_post (higher margin).
    track_to_player = {
        sub_post.synthetic_track_id: 1,
        # Note: sub_pre is NOT in track_to_player.
        3: 3,
        4: 4,
    }
    pf = _build_per_frame_pid_map(track_to_player, sub_tracks=[sub_pre, sub_post])
    # Pre-segment frames are unlabeled (sub_pre lost)
    assert pf.get((2, 0)) is None
    assert pf.get((2, 50)) is None
    assert pf.get((2, 99)) is None
    # Post-segment frames mapped to pid=1
    assert pf.get((2, 100)) == 1
    assert pf.get((2, 240)) == 1
