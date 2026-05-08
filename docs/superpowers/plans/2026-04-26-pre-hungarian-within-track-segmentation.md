# Pre-Hungarian Within-Track Segmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect within-track identity flips (BoT-SORT silent swaps + early-rally co-tracking) before `match_tracker._assign_tracks_to_players_global` runs Hungarian, and split affected tracks into clean per-segment candidates so each segment gets its own correct pid instead of being labeled with the whole-track aggregate dominant identity.

**Architecture:** New helper `_segment_tracks_by_appearance` in `match_tracker.py` runs after track-side classification, before the Hungarian cost matrix is built. It scores each candidate track over K=6 windows using the production few-shot `PlayerReIDClassifier` (DINOv2 frozen + linear head trained on user reference crops). When the classifier's per-window argmax pid changes between adjacent windows AND both pre/post per-segment aggregate margins exceed a strict gate (≥0.15), the track is split at the boundary and replaced with two sub-track candidates, each with its own aggregated `TrackAppearanceStats`. Sub-tracks are time-stamped (`f_first`, `f_last`) so a post-Hungarian uniqueness pass can resolve any per-frame conflicts where two assigned sub-tracks share a pid in overlapping frames. The whole splitter is gated by `ENABLE_REF_CROP_TRACK_SPLIT` (default `0`) and only fires when the video has frozen ref-crop profiles. With the flag off, behavior is byte-identical to current production.

**Tech Stack:** Python 3.11+, `rallycut/tracking/match_tracker.py`, `rallycut/tracking/reid_embeddings.py` (`PlayerReIDClassifier`), `rallycut/tracking/player_features.py` (`TrackAppearanceStats`, `extract_appearance_features`), pytest, the existing 9-fixture baseline harness `scripts/lock_baseline_v2.py`.

**Pre-registered ship gates:**
- Per-fixture player_attr Δ ≥ −0.5pp (no fixture regresses by more than 0.5pp).
- Aggregate player_attr Δ ≥ +0.5pp across the 9-fixture set.
- Aggregate serve_attr Δ ≥ −0.5pp (don't regress serve attribution).
- `wrong` rate Δ ≤ +0.0pp (the north-star: prefer miss over wrong — must not increase wrong-attribution rate).
- Unit tests pass.
- Type-check (`mypy`) and lint (`ruff`) pass.

If any gate fails, the flag stays at default `0` and the implementation stays dormant. Memory entry captures the no-go.

**Known limitations recorded up front:**
- Detection requires frozen ref crops on the video. Videos without ref crops are unaffected (flag is no-op).
- Per-frame uniqueness is enforced by *dropping* conflicting frames to unlabeled, not by re-assigning. Acceptable trade for the north-star (miss > wrong).
- The MEMORY.md warning about 2026-04-25/26 contaminated measurements means the v2 baseline cannot be the A/B reference. Task 1 re-locks a fresh v3 baseline at HEAD before any code change.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `analysis/rallycut/tracking/match_tracker.py` | Add `_segment_tracks_by_appearance` (new method on `MatchPlayerTracker`); modify `process_rally` to call it before `_assign_tracks_to_players_global` when `ENABLE_REF_CROP_TRACK_SPLIT=1` and frozen ref crops are present; add `_resolve_subtrack_pid_conflicts` post-Hungarian pass; add module constants `SEGMENT_NUM_WINDOWS`, `SEGMENT_MIN_WINDOW_FRAMES`, `SEGMENT_MIN_PER_SEGMENT_MARGIN`, `SEGMENT_MIN_CONFIRMING_SAMPLES`. |
| `analysis/rallycut/tracking/_subtrack.py` (new) | `SubTrackCandidate` dataclass holding `parent_track_id`, `segment_index`, `f_start`, `f_end`, `appearance_stats`, optionally `aggregated_argmax_pid` and `aggregated_margin`. Reuses `TrackAppearanceStats` for compatibility with existing Hungarian code. |
| `analysis/tests/unit/test_track_segmentation.py` (new) | Unit tests covering: (a) consistent track passes through unchanged; (b) clear flip with strong margin gets split; (c) weak post-segment margin abstains from split; (d) per-frame conflict resolver drops the lower-margin sub-track in overlapping frames; (e) flag-off makes the path a no-op. |
| `analysis/scripts/ab_test_track_split.py` (new) | A/B harness: for each of the 9 fixtures, run `match-players` + `reattribute-actions` twice (flag off, flag on), score Surface A via the same pipeline as `lock_baseline_v2.py`, write a delta report. |
| `analysis/reports/attribution_rebuild/baseline_v3.json` (output of Task 1) | Locked baseline at HEAD, no flag. |
| `analysis/reports/attribution_rebuild/track_split_ab_2026_04_26.json` (output of Task 7) | A/B deltas + per-fixture / aggregate gate decisions. |

---

## Task 1: Lock v3 baseline at HEAD

**Files:**
- Run: `analysis/scripts/lock_baseline_v2.py` with `--out reports/attribution_rebuild/baseline_v3.json`

The MEMORY.md contamination warning means v2 baseline can't be the reference. Re-lock a fresh baseline at HEAD code BEFORE touching anything.

- [ ] **Step 1: Sanity-check fixture registry**

Run:
```bash
cd analysis
uv run python -c "
import json
with open('reports/attribution_rebuild/fixture_video_ids_2026_04_24.json') as f:
    reg = json.load(f)
print('fixtures:', list(reg['fixtures'].keys()))
print('total_rallies:', sum(f['n_rallies'] for f in reg['fixtures'].values()))
print('rallies_with_action_gt:', sum(f['n_rallies_with_action_gt'] for f in reg['fixtures'].values()))
"
```
Expected: 9 fixtures, ~135 rallies, ~69 with action GT.

- [ ] **Step 2: Run baseline lock script**

Modify `lock_baseline_v2.py` to accept `--out` argument (a one-line addition near `OUT_PATH`):

```python
# In lock_baseline_v2.py argument parser (search for `argparse.ArgumentParser`):
parser.add_argument("--out", type=Path, default=ATTR_REPORTS / "baseline_2026_04_24_v2.json",
                    help="Output JSON path for locked baseline.")
# Then replace `OUT_PATH` references with `args.out`.
```

Run:
```bash
cd analysis
uv run python scripts/lock_baseline_v2.py --out reports/attribution_rebuild/baseline_v3.json 2>&1 | tee reports/attribution_rebuild/baseline_v3.log
```
Expected: per-fixture progress lines, exit 0, `baseline_v3.json` written.
Duration: ~10-20 minutes (9 fixtures × 1-2 min/fixture).

- [ ] **Step 3: Inspect baseline aggregates**

Run:
```bash
cd analysis
uv run python -c "
import json
with open('reports/attribution_rebuild/baseline_v3.json') as f:
    b = json.load(f)
agg = b['aggregate']
print(f'player_attr={agg[\"player_attr\"]:.4f}  serve_attr={agg[\"serve_attr\"]:.4f}  wrong={agg[\"wrong\"]:.4f}  n_actions={agg[\"n_actions\"]}')
for fx, fxd in b['by_fixture'].items():
    print(f'  {fx}: player_attr={fxd[\"player_attr\"]:.4f}  serve_attr={fxd[\"serve_attr\"]:.4f}  wrong={fxd[\"wrong\"]:.4f}  n={fxd[\"n_actions\"]}')
"
```
Expected: aggregate player_attr in 0.40-0.50 range (last v2 was 0.438), per-fixture printed. Take a screenshot / commit the log so we have a reference snapshot.

- [ ] **Step 4: Commit baseline**

```bash
git add analysis/reports/attribution_rebuild/baseline_v3.json \
        analysis/reports/attribution_rebuild/baseline_v3.log \
        analysis/scripts/lock_baseline_v2.py
git commit -m "eval: lock v3 baseline at HEAD before track-split A/B"
```

---

## Task 2: SubTrackCandidate dataclass and tests

**Files:**
- Create: `analysis/rallycut/tracking/_subtrack.py`
- Create: `analysis/tests/unit/test_track_segmentation.py`

- [ ] **Step 1: Write the failing test for SubTrackCandidate**

Create `analysis/tests/unit/test_track_segmentation.py`:

```python
"""Tests for within-track appearance segmentation (Task 2-5)."""
from __future__ import annotations

import numpy as np
import pytest

from rallycut.tracking._subtrack import SubTrackCandidate
from rallycut.tracking.player_features import TrackAppearanceStats


def _make_stats(track_id: int, frame_count: int, reid: np.ndarray | None = None) -> TrackAppearanceStats:
    return TrackAppearanceStats(
        track_id=track_id,
        frame_count=frame_count,
        reid_embedding=reid,
    )


def test_subtrack_candidate_carries_parent_and_window():
    parent_stats = _make_stats(track_id=2, frame_count=300)
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
    assert sub.synthetic_track_id == -1002 - 0 * 1000  # see Step 3
    assert sub.aggregated_argmax_pid == 2
    assert sub.aggregated_margin == pytest.approx(0.18)


def test_subtrack_candidate_synthetic_track_id_is_unique_per_segment():
    sub_a = SubTrackCandidate(parent_track_id=2, segment_index=0, f_start=0, f_end=100,
                              appearance_stats=_make_stats(2, 100))
    sub_b = SubTrackCandidate(parent_track_id=2, segment_index=1, f_start=100, f_end=200,
                              appearance_stats=_make_stats(2, 100))
    assert sub_a.synthetic_track_id != sub_b.synthetic_track_id
    # Synthetic ids must not collide with real positive track_ids
    assert sub_a.synthetic_track_id < 0
    assert sub_b.synthetic_track_id < 0


def test_subtrack_candidate_overlap_detection():
    a = SubTrackCandidate(parent_track_id=2, segment_index=0, f_start=0, f_end=100,
                          appearance_stats=_make_stats(2, 100))
    b = SubTrackCandidate(parent_track_id=3, segment_index=0, f_start=50, f_end=150,
                          appearance_stats=_make_stats(3, 100))
    c = SubTrackCandidate(parent_track_id=4, segment_index=0, f_start=200, f_end=300,
                          appearance_stats=_make_stats(4, 100))
    assert a.overlaps(b)
    assert b.overlaps(a)
    assert not a.overlaps(c)
    assert not c.overlaps(b)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd analysis
uv run pytest tests/unit/test_track_segmentation.py -x -v
```
Expected: ImportError on `from rallycut.tracking._subtrack import SubTrackCandidate`.

- [ ] **Step 3: Implement `_subtrack.py`**

Create `analysis/rallycut/tracking/_subtrack.py`:

```python
"""SubTrackCandidate: a time-bounded slice of a parent track for Hungarian assignment.

Used by `match_tracker._segment_tracks_by_appearance` to break a single BoT-SORT
track into two halves when within-track appearance evidence shows the physical
player carrying that track_id changed mid-rally (silent swap, early-rally
co-tracking).

The synthetic_track_id is negative so it cannot collide with real positive
track_ids in `track_to_player` mappings or downstream consumers. Sub-tracks
keep a pointer to the parent's `track_id` so post-Hungarian can rewrite the
final mapping back into the parent's track_id space, but at frame-level
granularity (each frame in [f_start, f_end] inherits the sub-track's pid).
"""
from __future__ import annotations

from dataclasses import dataclass

from rallycut.tracking.player_features import TrackAppearanceStats


@dataclass
class SubTrackCandidate:
    parent_track_id: int
    segment_index: int  # 0 = pre-flip, 1 = post-flip; allows future >2 segments.
    f_start: int  # Inclusive (rally-relative frame number).
    f_end: int  # Inclusive.
    appearance_stats: TrackAppearanceStats
    aggregated_argmax_pid: int | None = None
    aggregated_margin: float | None = None

    @property
    def synthetic_track_id(self) -> int:
        # Negative, deterministic, parent+segment unique. Magnitude well above
        # any plausible real track_id (BoT-SORT track ids stay in the hundreds
        # at most for a single rally).
        return -1000 * (self.segment_index + 1) - self.parent_track_id - 2

    def overlaps(self, other: "SubTrackCandidate") -> bool:
        if self.parent_track_id == other.parent_track_id and self.segment_index == other.segment_index:
            return True
        return not (self.f_end < other.f_start or other.f_end < self.f_start)
```

- [ ] **Step 4: Run tests**

```bash
cd analysis
uv run pytest tests/unit/test_track_segmentation.py -x -v
```
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add analysis/rallycut/tracking/_subtrack.py analysis/tests/unit/test_track_segmentation.py
git commit -m "feat(tracking): add SubTrackCandidate dataclass for within-track segmentation"
```

---

## Task 3: `_segment_tracks_by_appearance` — detection logic

**Files:**
- Modify: `analysis/rallycut/tracking/match_tracker.py` (add module constants + new private method on `MatchPlayerTracker`)
- Modify: `analysis/tests/unit/test_track_segmentation.py` (add tests)

- [ ] **Step 1: Add failing tests for detection logic**

Append to `analysis/tests/unit/test_track_segmentation.py`:

```python
from unittest.mock import MagicMock
from rallycut.tracking.match_tracker import (
    MatchPlayerTracker,
    SEGMENT_MIN_PER_SEGMENT_MARGIN,
)
from rallycut.tracking.player_features import PlayerPosition


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
        self.is_trained = True

    def predict_single(self, crop):
        # Crop encodes (track_id, frame) in its first two bytes for the test.
        track_id, frame = int(crop[0, 0, 0]), int(crop[0, 0, 1])
        for (tid, fs, fe), probs in self.scripts.items():
            if tid == track_id and fs <= frame <= fe:
                return probs
        return {pid: 1.0 / len(self.player_ids) for pid in self.player_ids}


def _consistent_track_inputs(track_id: int = 1, n_frames: int = 200):
    positions = [_fake_position(track_id, f, x=0.5 + 0.001 * f) for f in range(n_frames)]
    classifier = _FakeClassifier({
        (track_id, 0, n_frames - 1): {1: 0.85, 2: 0.05, 3: 0.05, 4: 0.05},
    })
    return positions, classifier


def test_segment_consistent_track_does_not_split():
    tracker = MatchPlayerTracker.__new__(MatchPlayerTracker)
    tracker.frozen_player_ids = {1, 2, 3, 4}
    positions, classifier = _consistent_track_inputs(track_id=1, n_frames=200)
    track_stats = {1: TrackAppearanceStats(track_id=1, frame_count=200)}
    sub_tracks = tracker._segment_tracks_by_appearance(
        track_ids=[1],
        track_stats=track_stats,
        positions=positions,
        classifier=classifier,
        crop_extractor=lambda tid, frame: _stub_crop(tid, frame),
    )
    assert len(sub_tracks) == 0  # No splits, fall through to original tracks.


def _stub_crop(track_id: int, frame: int):
    import numpy as np
    arr = np.zeros((20, 20, 3), dtype=np.uint8)
    arr[0, 0, 0] = track_id
    arr[0, 0, 1] = frame
    return arr


def test_segment_clear_flip_with_strong_margins_splits():
    tracker = MatchPlayerTracker.__new__(MatchPlayerTracker)
    tracker.frozen_player_ids = {1, 2, 3, 4}
    n_frames = 240
    positions = [_fake_position(2, f) for f in range(n_frames)]
    classifier = _FakeClassifier({
        # First half: confident pid=2; second half: confident pid=1.
        (2, 0, 119): {1: 0.10, 2: 0.78, 3: 0.07, 4: 0.05},
        (2, 120, 239): {1: 0.82, 2: 0.08, 3: 0.06, 4: 0.04},
    })
    track_stats = {2: TrackAppearanceStats(track_id=2, frame_count=n_frames)}
    sub_tracks = tracker._segment_tracks_by_appearance(
        track_ids=[2], track_stats=track_stats, positions=positions,
        classifier=classifier, crop_extractor=_stub_crop,
    )
    assert len(sub_tracks) == 2
    assert sub_tracks[0].parent_track_id == 2 and sub_tracks[0].segment_index == 0
    assert sub_tracks[0].aggregated_argmax_pid == 2
    assert sub_tracks[1].parent_track_id == 2 and sub_tracks[1].segment_index == 1
    assert sub_tracks[1].aggregated_argmax_pid == 1
    # Both segments must clear the per-segment margin gate.
    assert sub_tracks[0].aggregated_margin >= SEGMENT_MIN_PER_SEGMENT_MARGIN
    assert sub_tracks[1].aggregated_margin >= SEGMENT_MIN_PER_SEGMENT_MARGIN


def test_segment_weak_post_segment_abstains():
    tracker = MatchPlayerTracker.__new__(MatchPlayerTracker)
    tracker.frozen_player_ids = {1, 2, 3, 4}
    n_frames = 240
    positions = [_fake_position(2, f) for f in range(n_frames)]
    classifier = _FakeClassifier({
        (2, 0, 119): {1: 0.10, 2: 0.78, 3: 0.07, 4: 0.05},  # strong
        (2, 120, 239): {1: 0.30, 2: 0.28, 3: 0.22, 4: 0.20},  # weak — margin ~0.02
    })
    track_stats = {2: TrackAppearanceStats(track_id=2, frame_count=n_frames)}
    sub_tracks = tracker._segment_tracks_by_appearance(
        track_ids=[2], track_stats=track_stats, positions=positions,
        classifier=classifier, crop_extractor=_stub_crop,
    )
    assert len(sub_tracks) == 0  # Abstain — post segment too weak to confirm split.
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd analysis
uv run pytest tests/unit/test_track_segmentation.py -x -v
```
Expected: 3 fail (no `_segment_tracks_by_appearance`, no `SEGMENT_MIN_PER_SEGMENT_MARGIN`).

- [ ] **Step 3: Add module constants in `match_tracker.py`**

Open `analysis/rallycut/tracking/match_tracker.py`. After line 113 (after `FIRST_RALLY_INIT_WINDOW_FRAMES = 120`), add:

```python
# ---------------------------------------------------------------------------
# Within-track appearance segmentation (Task 3-5, 2026-04-26)
# ---------------------------------------------------------------------------

# How many appearance windows to score per track. Each window aggregates ~1/K
# of the track's frames. K=6 gives enough resolution to localize a flip near
# either rally end while keeping classifier inference cheap.
SEGMENT_NUM_WINDOWS = 6

# Minimum frames per window. Below this, a window is dropped (too noisy).
SEGMENT_MIN_WINDOW_FRAMES = 12

# Minimum per-segment aggregate margin (best-pid prob minus 2nd-best) required
# on BOTH the pre and post segments before a split fires. The probe at
# 2026-04-26 measured PRE/POST aggregate margins of +0.188 / +0.442 (cuco r5),
# +0.228 / +0.317 (cuco r3), +0.350 / +0.305 (wawa r10) on real splits, and
# +0.086 / +0.004 (cuco r5 tid=2) / +0.352 / +0.140 (wawa r10 tid=3) on
# borderline cases that should abstain. 0.15 is the tightest gate that fires
# all three real splits and abstains on both borderline cases.
SEGMENT_MIN_PER_SEGMENT_MARGIN = 0.15

# Minimum number of consecutive confirming windows on each side of the flip
# point. Prevents single-window noise from triggering a split.
SEGMENT_MIN_CONFIRMING_WINDOWS = 2

# Per-frame inference is expensive — sample at most this many frames per
# window for classifier scoring. Crops are large enough at proxy resolution
# that one crop per ~20 frames is sufficient.
SEGMENT_FRAMES_PER_WINDOW = 4
```

- [ ] **Step 4: Implement `_segment_tracks_by_appearance` in `match_tracker.py`**

Add method to `MatchPlayerTracker` class (after `_assign_tracks_to_players_global`, before `_refine_within_team`):

```python
    def _segment_tracks_by_appearance(
        self,
        track_ids: list[int],
        track_stats: dict[int, TrackAppearanceStats],
        positions: list[PlayerPosition],
        classifier: Any,  # PlayerReIDClassifier (Any to avoid circular import)
        crop_extractor: "Callable[[int, int], np.ndarray | None]",
    ) -> list[SubTrackCandidate]:
        """Detect within-track appearance flips using the few-shot classifier.

        For each track:
        1. Sample SEGMENT_NUM_WINDOWS windows across the track's frame range.
        2. Score each window via classifier on SEGMENT_FRAMES_PER_WINDOW crops.
        3. Window argmax_pid + per-window margin.
        4. Flip detected when:
             - argmax_pid changes between adjacent windows AND
             - both pre-flip and post-flip have ≥ SEGMENT_MIN_CONFIRMING_WINDOWS
               windows agreeing on the same argmax pid AND
             - aggregate per-segment margins ≥ SEGMENT_MIN_PER_SEGMENT_MARGIN.

        Returns a flat list of SubTrackCandidates. If a track is consistent or
        the gate doesn't clear, that track is ABSENT from the returned list
        (caller falls back to the original whole-track candidate).
        """
        if not classifier or not getattr(classifier, "is_trained", False):
            return []

        # Index positions by track_id and frame.
        by_track: dict[int, list[PlayerPosition]] = {}
        for p in positions:
            if p.track_id in track_ids:
                by_track.setdefault(p.track_id, []).append(p)
        for tid in by_track:
            by_track[tid].sort(key=lambda x: x.frame_number)

        sub_tracks: list[SubTrackCandidate] = []

        for tid in track_ids:
            track_positions = by_track.get(tid, [])
            if len(track_positions) < SEGMENT_NUM_WINDOWS * SEGMENT_MIN_WINDOW_FRAMES:
                continue  # Track too short to segment reliably.

            f_first = track_positions[0].frame_number
            f_last = track_positions[-1].frame_number
            if f_last - f_first < SEGMENT_NUM_WINDOWS * SEGMENT_MIN_WINDOW_FRAMES:
                continue

            # Build SEGMENT_NUM_WINDOWS roughly equal windows, each represented
            # by a list of frame numbers in that window.
            window_bounds = np.linspace(f_first, f_last + 1, SEGMENT_NUM_WINDOWS + 1).astype(int)
            window_argmax: list[int | None] = []
            window_margin: list[float] = []
            window_probs: list[dict[int, float] | None] = []
            window_frame_ranges: list[tuple[int, int]] = []

            for w in range(SEGMENT_NUM_WINDOWS):
                w_start = int(window_bounds[w])
                w_end = int(window_bounds[w + 1] - 1)
                window_frame_ranges.append((w_start, w_end))
                window_positions = [p for p in track_positions if w_start <= p.frame_number <= w_end]
                if len(window_positions) < SEGMENT_MIN_WINDOW_FRAMES:
                    window_argmax.append(None)
                    window_margin.append(0.0)
                    window_probs.append(None)
                    continue
                # Sample SEGMENT_FRAMES_PER_WINDOW evenly-spaced frames in the window.
                idxs = np.linspace(0, len(window_positions) - 1, SEGMENT_FRAMES_PER_WINDOW).astype(int)
                sample_positions = [window_positions[i] for i in idxs]
                crops: list[np.ndarray] = []
                for p in sample_positions:
                    crop = crop_extractor(tid, p.frame_number)
                    if crop is not None:
                        crops.append(crop)
                if not crops:
                    window_argmax.append(None)
                    window_margin.append(0.0)
                    window_probs.append(None)
                    continue
                probs_list = classifier.predict(crops)
                # Average probs across this window's crops.
                avg = {pid: float(np.mean([p[pid] for p in probs_list])) for pid in probs_list[0]}
                sorted_pids = sorted(avg.items(), key=lambda x: x[1], reverse=True)
                window_argmax.append(sorted_pids[0][0])
                window_margin.append(sorted_pids[0][1] - sorted_pids[1][1])
                window_probs.append(avg)

            # Find the flip boundary.
            split_at_window = self._find_segment_flip(window_argmax, window_margin)
            if split_at_window is None:
                continue

            f_split = window_frame_ranges[split_at_window][0]

            # Build per-segment aggregate stats from the actual TrackAppearanceStats.
            pre_positions = [p for p in track_positions if p.frame_number < f_split]
            post_positions = [p for p in track_positions if p.frame_number >= f_split]
            if len(pre_positions) < SEGMENT_MIN_WINDOW_FRAMES * SEGMENT_MIN_CONFIRMING_WINDOWS \
                    or len(post_positions) < SEGMENT_MIN_WINDOW_FRAMES * SEGMENT_MIN_CONFIRMING_WINDOWS:
                continue

            # Aggregate classifier scores per segment.
            pre_argmax, pre_margin = self._aggregate_segment_classifier(
                tid, pre_positions, classifier, crop_extractor,
            )
            post_argmax, post_margin = self._aggregate_segment_classifier(
                tid, post_positions, classifier, crop_extractor,
            )

            # Strict gate: BOTH segments must clear margin AND argmax must differ.
            if (
                pre_argmax is None or post_argmax is None
                or pre_argmax == post_argmax
                or pre_margin < SEGMENT_MIN_PER_SEGMENT_MARGIN
                or post_margin < SEGMENT_MIN_PER_SEGMENT_MARGIN
            ):
                continue

            parent_stats = track_stats.get(tid)
            if parent_stats is None:
                continue

            # Build per-segment TrackAppearanceStats by re-extracting features
            # from each segment's positions. This keeps the existing Hungarian
            # cost code unchanged: it sees stats objects, not knowing they came
            # from a sub-track.
            pre_stats = self._stats_for_positions(parent_stats, pre_positions, tid)
            post_stats = self._stats_for_positions(parent_stats, post_positions, tid)

            sub_tracks.append(SubTrackCandidate(
                parent_track_id=tid,
                segment_index=0,
                f_start=pre_positions[0].frame_number,
                f_end=pre_positions[-1].frame_number,
                appearance_stats=pre_stats,
                aggregated_argmax_pid=pre_argmax,
                aggregated_margin=pre_margin,
            ))
            sub_tracks.append(SubTrackCandidate(
                parent_track_id=tid,
                segment_index=1,
                f_start=post_positions[0].frame_number,
                f_end=post_positions[-1].frame_number,
                appearance_stats=post_stats,
                aggregated_argmax_pid=post_argmax,
                aggregated_margin=post_margin,
            ))

            logger.info(
                "Within-track split: tid=%d at frame %d  pre→pid%d (margin %+.3f, %d frames)  "
                "post→pid%d (margin %+.3f, %d frames)",
                tid, f_split, pre_argmax, pre_margin, len(pre_positions),
                post_argmax, post_margin, len(post_positions),
            )

        return sub_tracks

    @staticmethod
    def _find_segment_flip(
        window_argmax: list[int | None],
        window_margin: list[float],
    ) -> int | None:
        """Find the first window index W such that:
           - argmax pid at W differs from argmax pid at W-1
           - At least SEGMENT_MIN_CONFIRMING_WINDOWS windows BEFORE W agree on same pid
           - At least SEGMENT_MIN_CONFIRMING_WINDOWS windows AT/AFTER W agree on same pid
           - All confirming windows on each side have margin ≥ SEGMENT_MIN_PER_SEGMENT_MARGIN
        Returns the window index of the flip start, or None.
        """
        n = len(window_argmax)
        if n < 2 * SEGMENT_MIN_CONFIRMING_WINDOWS:
            return None
        for w in range(SEGMENT_MIN_CONFIRMING_WINDOWS, n - SEGMENT_MIN_CONFIRMING_WINDOWS + 1):
            pre_pids = window_argmax[max(0, w - SEGMENT_MIN_CONFIRMING_WINDOWS):w]
            post_pids = window_argmax[w:w + SEGMENT_MIN_CONFIRMING_WINDOWS]
            pre_margins = window_margin[max(0, w - SEGMENT_MIN_CONFIRMING_WINDOWS):w]
            post_margins = window_margin[w:w + SEGMENT_MIN_CONFIRMING_WINDOWS]
            if any(p is None for p in pre_pids) or any(p is None for p in post_pids):
                continue
            if len(set(pre_pids)) != 1 or len(set(post_pids)) != 1:
                continue
            if pre_pids[0] == post_pids[0]:
                continue
            if min(pre_margins) < SEGMENT_MIN_PER_SEGMENT_MARGIN:
                continue
            if min(post_margins) < SEGMENT_MIN_PER_SEGMENT_MARGIN:
                continue
            return w
        return None

    @staticmethod
    def _aggregate_segment_classifier(
        track_id: int,
        positions: list[PlayerPosition],
        classifier: Any,
        crop_extractor: "Callable[[int, int], np.ndarray | None]",
    ) -> tuple[int | None, float]:
        if not positions:
            return None, 0.0
        n_samples = min(8, len(positions))
        idxs = np.linspace(0, len(positions) - 1, n_samples).astype(int)
        crops = []
        for i in idxs:
            crop = crop_extractor(track_id, positions[i].frame_number)
            if crop is not None:
                crops.append(crop)
        if not crops:
            return None, 0.0
        probs_list = classifier.predict(crops)
        avg = {pid: float(np.mean([p[pid] for p in probs_list])) for pid in probs_list[0]}
        sorted_pids = sorted(avg.items(), key=lambda x: x[1], reverse=True)
        return sorted_pids[0][0], sorted_pids[0][1] - sorted_pids[1][1]

    @staticmethod
    def _stats_for_positions(
        parent_stats: TrackAppearanceStats,
        positions: list[PlayerPosition],
        track_id: int,
    ) -> TrackAppearanceStats:
        """Build a per-segment TrackAppearanceStats. For now, copies the parent's
        appearance vectors (HSV / ReID embedding) but with frame_count adjusted
        to the segment length. Future refinement: re-extract per-segment HSV +
        ReID from segment frames. Acceptable for v1 because the Hungarian uses
        appearance_cost vs profile, and the parent's aggregate is the dominant
        identity already — we want the per-segment Hungarian decision to be
        driven primarily by the classifier's per-segment argmax (encoded via
        a synthetic signal in Task 4)."""
        return TrackAppearanceStats(
            track_id=track_id,
            frame_count=len(positions),
            reid_embedding=parent_stats.reid_embedding,
            reid_embedding_count=parent_stats.reid_embedding_count,
            hsv_upper=parent_stats.hsv_upper,
            hsv_lower=parent_stats.hsv_lower,
            dominant_color=parent_stats.dominant_color,
            skin_tone=parent_stats.skin_tone,
        )
```

Add at the top of `match_tracker.py` (with other imports):

```python
from typing import Callable
from rallycut.tracking._subtrack import SubTrackCandidate
```

- [ ] **Step 5: Run tests**

```bash
cd analysis
uv run pytest tests/unit/test_track_segmentation.py -x -v
```
Expected: all PASS.

- [ ] **Step 6: Run full unit test suite to make sure nothing else broke**

```bash
cd analysis
uv run pytest tests/unit -x -v
```
Expected: green.

- [ ] **Step 7: Commit**

```bash
git add analysis/rallycut/tracking/match_tracker.py analysis/tests/unit/test_track_segmentation.py
git commit -m "feat(tracking): add _segment_tracks_by_appearance with strict per-segment margin gate"
```

---

## Task 4: Wire splitter into `process_rally` behind `ENABLE_REF_CROP_TRACK_SPLIT`

**Files:**
- Modify: `analysis/rallycut/tracking/match_tracker.py:680-735` (the dispatch in `process_rally`)
- Modify: `analysis/tests/unit/test_track_segmentation.py` (integration test)

- [ ] **Step 1: Add integration test for flag-gated dispatch**

Append to `analysis/tests/unit/test_track_segmentation.py`:

```python
import os


def test_track_split_flag_off_is_byte_identical(monkeypatch):
    """With ENABLE_REF_CROP_TRACK_SPLIT=0, _segment_tracks_by_appearance is never called."""
    monkeypatch.delenv("ENABLE_REF_CROP_TRACK_SPLIT", raising=False)
    tracker = MatchPlayerTracker.__new__(MatchPlayerTracker)
    tracker.frozen_player_ids = {1, 2, 3, 4}
    # Use a sentinel that explodes if the splitter is called.
    def boom(*args, **kwargs):
        raise AssertionError("splitter should not run when flag is 0")
    tracker._segment_tracks_by_appearance = boom  # type: ignore[method-assign]
    # _maybe_segment_tracks_by_appearance is the wrapper that checks the flag.
    result = tracker._maybe_segment_tracks_by_appearance(
        track_ids=[1, 2], track_stats={}, positions=[], classifier=MagicMock(),
    )
    assert result == []  # No-op, no exception.


def test_track_split_flag_on_calls_splitter(monkeypatch):
    monkeypatch.setenv("ENABLE_REF_CROP_TRACK_SPLIT", "1")
    tracker = MatchPlayerTracker.__new__(MatchPlayerTracker)
    tracker.frozen_player_ids = {1, 2, 3, 4}
    sentinel: list[bool] = []

    def fake_segment(track_ids, track_stats, positions, classifier, crop_extractor):
        sentinel.append(True)
        return []

    tracker._segment_tracks_by_appearance = fake_segment  # type: ignore[method-assign]
    tracker._maybe_segment_tracks_by_appearance(
        track_ids=[1, 2], track_stats={}, positions=[], classifier=MagicMock(),
    )
    assert sentinel == [True]


def test_track_split_no_classifier_is_noop(monkeypatch):
    monkeypatch.setenv("ENABLE_REF_CROP_TRACK_SPLIT", "1")
    tracker = MatchPlayerTracker.__new__(MatchPlayerTracker)
    tracker.frozen_player_ids = {1, 2, 3, 4}
    result = tracker._maybe_segment_tracks_by_appearance(
        track_ids=[1, 2], track_stats={}, positions=[], classifier=None,
    )
    assert result == []
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
cd analysis
uv run pytest tests/unit/test_track_segmentation.py -x -v -k "flag_off or flag_on or no_classifier"
```
Expected: 3 fail (no `_maybe_segment_tracks_by_appearance` method).

- [ ] **Step 3: Implement `_maybe_segment_tracks_by_appearance` wrapper**

Add to `MatchPlayerTracker` (right above `_segment_tracks_by_appearance`):

```python
    def _maybe_segment_tracks_by_appearance(
        self,
        track_ids: list[int],
        track_stats: dict[int, TrackAppearanceStats],
        positions: list[PlayerPosition],
        classifier: Any,
        crop_extractor: "Callable[[int, int], np.ndarray | None] | None" = None,
    ) -> list[SubTrackCandidate]:
        """Flag-gated wrapper around `_segment_tracks_by_appearance`.

        Active only when:
          - ENABLE_REF_CROP_TRACK_SPLIT environment variable is "1"
          - A trained classifier is supplied
          - frozen_player_ids is non-empty (means video has ref-crop profiles)
        """
        if os.environ.get("ENABLE_REF_CROP_TRACK_SPLIT", "0") != "1":
            return []
        if classifier is None or not getattr(classifier, "is_trained", False):
            return []
        if not getattr(self, "frozen_player_ids", None):
            return []
        if crop_extractor is None:
            return []
        return self._segment_tracks_by_appearance(
            track_ids, track_stats, positions, classifier, crop_extractor,
        )
```

Also add `import os` at the top if not already present.

- [ ] **Step 4: Run the new tests**

```bash
cd analysis
uv run pytest tests/unit/test_track_segmentation.py -x -v -k "flag_off or flag_on or no_classifier"
```
Expected: 3 PASS.

- [ ] **Step 5: Wire into `process_rally`**

Locate `process_rally` in `match_tracker.py` (around line 680, after the y-classification step). Find these lines (currently around line 712-725):

```python
        if self.rally_count <= 1 and not self.frozen_player_ids:
            track_to_player = self._initialize_first_rally(
                top_tracks, track_avg_y, track_court_sides
            )
        else:
            track_to_player = self._assign_tracks_to_players_global(
                top_tracks, track_stats, track_court_sides,
                use_side_penalty=not self.frozen_player_ids,
                early_positions=early_positions,
            )
```

Modify to:

```python
        if self.rally_count <= 1 and not self.frozen_player_ids:
            track_to_player = self._initialize_first_rally(
                top_tracks, track_avg_y, track_court_sides
            )
        else:
            # Optional pre-Hungarian within-track split (Task 4, 2026-04-26).
            # Returns [] when flag is off, no classifier, or no frozen profiles.
            sub_tracks = self._maybe_segment_tracks_by_appearance(
                track_ids=top_tracks,
                track_stats=track_stats,
                positions=player_positions,
                classifier=getattr(self, "_few_shot_classifier", None),
                crop_extractor=getattr(self, "_crop_extractor", None),
            )
            if sub_tracks:
                top_tracks, track_stats, track_court_sides = self._merge_sub_tracks_into_candidates(
                    sub_tracks, top_tracks, track_stats, track_court_sides,
                )
            track_to_player = self._assign_tracks_to_players_global(
                top_tracks, track_stats, track_court_sides,
                use_side_penalty=not self.frozen_player_ids,
                early_positions=early_positions,
            )
            if sub_tracks:
                # Resolve per-frame conflicts where two assigned sub-tracks
                # share a pid in overlapping frames (Task 5).
                track_to_player = self._resolve_subtrack_pid_conflicts(
                    track_to_player, sub_tracks,
                )
```

Add stub for `_merge_sub_tracks_into_candidates` and `_resolve_subtrack_pid_conflicts` (latter completed in Task 5):

```python
    def _merge_sub_tracks_into_candidates(
        self,
        sub_tracks: list[SubTrackCandidate],
        top_tracks: list[int],
        track_stats: dict[int, TrackAppearanceStats],
        track_court_sides: dict[int, int],
    ) -> tuple[list[int], dict[int, TrackAppearanceStats], dict[int, int]]:
        """Replace each parent track that has sub-tracks with its sub-track
        synthetic ids. Sub-tracks inherit the parent's court side. Cap to 4
        longest synthetic+real candidates by frame count."""
        parents_with_subs = {s.parent_track_id for s in sub_tracks}
        new_tracks = [t for t in top_tracks if t not in parents_with_subs]
        new_stats: dict[int, TrackAppearanceStats] = {
            t: track_stats[t] for t in new_tracks if t in track_stats
        }
        new_sides: dict[int, int] = {
            t: track_court_sides[t] for t in new_tracks if t in track_court_sides
        }
        for sub in sub_tracks:
            sid = sub.synthetic_track_id
            new_tracks.append(sid)
            new_stats[sid] = sub.appearance_stats
            parent_side = track_court_sides.get(sub.parent_track_id, 0)
            new_sides[sid] = parent_side
        # Cap by frame count to top 4.
        new_tracks.sort(key=lambda t: -new_stats.get(t, TrackAppearanceStats(track_id=t, frame_count=0)).frame_count)
        new_tracks = new_tracks[:4]
        new_stats = {t: new_stats[t] for t in new_tracks}
        new_sides = {t: new_sides[t] for t in new_tracks if t in new_sides}
        return new_tracks, new_stats, new_sides

    def _resolve_subtrack_pid_conflicts(
        self,
        track_to_player: dict[int, int],
        sub_tracks: list[SubTrackCandidate],
    ) -> dict[int, int]:
        """Stub completed in Task 5 — returns input unchanged for now."""
        return track_to_player
```

The `_few_shot_classifier` and `_crop_extractor` attributes are populated by the caller (CLI) in Task 6; if absent (`getattr` returns `None`), the splitter's flag check returns `[]` and behavior is identical to current.

- [ ] **Step 6: Run full unit test suite**

```bash
cd analysis
uv run pytest tests/unit -x -v
```
Expected: green.

- [ ] **Step 7: Type-check**

```bash
cd analysis
uv run mypy rallycut/tracking/match_tracker.py rallycut/tracking/_subtrack.py
```
Expected: no errors.

- [ ] **Step 8: Commit**

```bash
git add analysis/rallycut/tracking/match_tracker.py analysis/tests/unit/test_track_segmentation.py
git commit -m "feat(tracking): wire within-track splitter into process_rally behind ENABLE_REF_CROP_TRACK_SPLIT"
```

---

## Task 5: Per-frame uniqueness conflict resolver

**Files:**
- Modify: `analysis/rallycut/tracking/match_tracker.py` (replace stub from Task 4)
- Modify: `analysis/tests/unit/test_track_segmentation.py` (resolver tests)

- [ ] **Step 1: Add failing tests for the resolver**

Append:

```python
def _make_sub(parent_tid: int, seg_idx: int, f_start: int, f_end: int,
              argmax_pid: int, margin: float) -> SubTrackCandidate:
    return SubTrackCandidate(
        parent_track_id=parent_tid, segment_index=seg_idx,
        f_start=f_start, f_end=f_end,
        appearance_stats=TrackAppearanceStats(track_id=parent_tid, frame_count=f_end - f_start + 1),
        aggregated_argmax_pid=argmax_pid, aggregated_margin=margin,
    )


def test_resolver_no_conflict_passes_through():
    tracker = MatchPlayerTracker.__new__(MatchPlayerTracker)
    sub_a = _make_sub(2, 0, 0, 100, argmax_pid=2, margin=0.20)
    sub_b = _make_sub(2, 1, 100, 250, argmax_pid=1, margin=0.40)
    sub_c = _make_sub(3, 0, 0, 250, argmax_pid=3, margin=0.50)  # full rally — no overlap with subs of tid=2 in pid space because it claims pid=3
    track_to_player = {
        sub_a.synthetic_track_id: 2,
        sub_b.synthetic_track_id: 1,
        sub_c.synthetic_track_id: 3,
    }
    out = tracker._resolve_subtrack_pid_conflicts(
        track_to_player, [sub_a, sub_b, sub_c],
    )
    assert out == track_to_player  # No conflict, unchanged.


def test_resolver_drops_lower_margin_when_two_subs_share_pid_in_overlap():
    tracker = MatchPlayerTracker.__new__(MatchPlayerTracker)
    # sub_a covers frames 0-100 with margin 0.10, sub_b covers 50-150 with margin 0.40,
    # both Hungarian-assigned to pid=2. Their overlap (50-100) violates per-frame uniqueness.
    sub_a = _make_sub(2, 0, 0, 100, argmax_pid=2, margin=0.10)
    sub_b = _make_sub(3, 0, 50, 150, argmax_pid=2, margin=0.40)
    track_to_player = {
        sub_a.synthetic_track_id: 2,
        sub_b.synthetic_track_id: 2,
    }
    out = tracker._resolve_subtrack_pid_conflicts(
        track_to_player, [sub_a, sub_b],
    )
    # Lower-margin sub_a loses; in overlap frames it goes unlabeled (not in mapping)
    # but its non-overlapping frames keep pid=2.
    assert out[sub_b.synthetic_track_id] == 2
    # Conflict resolution downgrades sub_a from "track-level pid=2" to a
    # frame-conditional record that excludes 50-100. The simplest expression:
    # sub_a's mapping is split into "sub_a_kept" covering 0-49 with pid=2.
    assert any(
        out_v == 2 and out_k == sub_a.synthetic_track_id
        for out_k, out_v in out.items()
    )
    # The frame-range exclusion is recorded on the sub_a record.
    assert sub_a.excluded_frames == set(range(50, 101))  # 50..100 inclusive
```

- [ ] **Step 2: Run tests, verify failure**

```bash
cd analysis
uv run pytest tests/unit/test_track_segmentation.py -x -v -k "resolver"
```
Expected: 2 fail.

- [ ] **Step 3: Add `excluded_frames` to `SubTrackCandidate`**

Modify `analysis/rallycut/tracking/_subtrack.py`:

```python
from dataclasses import dataclass, field

@dataclass
class SubTrackCandidate:
    parent_track_id: int
    segment_index: int
    f_start: int
    f_end: int
    appearance_stats: TrackAppearanceStats
    aggregated_argmax_pid: int | None = None
    aggregated_margin: float | None = None
    excluded_frames: set[int] = field(default_factory=set)

    # ... existing methods unchanged ...
```

- [ ] **Step 4: Implement the resolver**

Replace the stub `_resolve_subtrack_pid_conflicts` in `match_tracker.py`:

```python
    def _resolve_subtrack_pid_conflicts(
        self,
        track_to_player: dict[int, int],
        sub_tracks: list[SubTrackCandidate],
    ) -> dict[int, int]:
        """Enforce per-frame pid uniqueness.

        After Hungarian, multiple sub-tracks may have been assigned the same pid.
        For each pid, find sub-tracks assigned to it and check pairwise temporal
        overlap. In overlap regions, retain only the highest-margin sub-track and
        record the loser's frames in `excluded_frames` (so the per-frame writer
        can leave them unlabeled). Non-overlap frames of the loser keep the pid.
        """
        sub_by_synth = {s.synthetic_track_id: s for s in sub_tracks}
        pid_to_subs: dict[int, list[SubTrackCandidate]] = {}
        for synth_id, pid in track_to_player.items():
            sub = sub_by_synth.get(synth_id)
            if sub is None:
                continue
            pid_to_subs.setdefault(pid, []).append(sub)

        for _pid, subs in pid_to_subs.items():
            if len(subs) < 2:
                continue
            subs_sorted = sorted(subs, key=lambda s: -(s.aggregated_margin or 0.0))
            kept = subs_sorted[0]
            for loser in subs_sorted[1:]:
                if not kept.overlaps(loser):
                    continue
                overlap_start = max(kept.f_start, loser.f_start)
                overlap_end = min(kept.f_end, loser.f_end)
                loser.excluded_frames.update(range(overlap_start, overlap_end + 1))
                logger.info(
                    "Subtrack conflict (pid=%d): kept synth=%d (margin %+.3f), "
                    "excluding loser synth=%d frames %d-%d",
                    _pid, kept.synthetic_track_id, kept.aggregated_margin or 0.0,
                    loser.synthetic_track_id, overlap_start, overlap_end,
                )
        return track_to_player
```

- [ ] **Step 5: Run resolver tests**

```bash
cd analysis
uv run pytest tests/unit/test_track_segmentation.py -x -v -k "resolver"
```
Expected: 2 PASS.

- [ ] **Step 6: Run full unit suite**

```bash
cd analysis
uv run pytest tests/unit -x
```
Expected: green.

- [ ] **Step 7: Wire excluded_frames into per-frame writer**

The downstream consumer of `track_to_player` writes per-frame `playerTrackId` values from positions. After `process_rally` returns, the caller (the CLI command `match_players`) translates synthetic ids back to per-frame pids using `f_start`/`f_end` of each sub-track. Excluded frames must be omitted (no `playerTrackId` written for that frame).

Find the per-frame writer in `analysis/rallycut/cli/commands/match_players.py` (around line 561 / line 720 where `track_to_player` is used to write per-frame mappings). Add a translation step before writing:

```python
# In match_players.py — after MatchPlayerTracker.process_rally returns:
# Translate synthetic sub-track ids back to per-frame parent pid mappings.
sub_tracks_for_rally = getattr(tracker, "_last_rally_sub_tracks", []) or []
per_frame_track_to_pid = _build_per_frame_pid_map(
    result.track_to_player, sub_tracks_for_rally,
)
# Then iterate positions and use per_frame_track_to_pid.get((parent_tid, frame), None).
```

Helper to add at module level in `match_players.py`:

```python
from rallycut.tracking._subtrack import SubTrackCandidate


def _build_per_frame_pid_map(
    track_to_player: dict[int, int],
    sub_tracks: list[SubTrackCandidate],
) -> dict[tuple[int, int], int]:
    """Return mapping of (parent_track_id, frame_number) -> pid.

    For real (non-synthetic) tracks, the mapping is constant across all frames.
    For sub-tracks, the mapping covers only [f_start, f_end] excluding excluded_frames.
    """
    out: dict[tuple[int, int], int] = {}
    sub_by_synth = {s.synthetic_track_id: s for s in sub_tracks}
    for tid, pid in track_to_player.items():
        if tid in sub_by_synth:
            sub = sub_by_synth[tid]
            for f in range(sub.f_start, sub.f_end + 1):
                if f in sub.excluded_frames:
                    continue
                out[(sub.parent_track_id, f)] = pid
        else:
            # Real track: pid applies to all frames where this track has a position.
            # Caller fills these in by iterating positions and using a default lookup.
            out[(tid, -1)] = pid  # Sentinel: -1 frame = "any frame"
    return out
```

Update the per-frame writer in `match_players.py` to query this map. The exact line range depends on the existing code (currently around 540-580 and 705-730 for the two write sites). Modify both to use:
```python
pid = per_frame_pid_map.get((track_id, frame_number)) or per_frame_pid_map.get((track_id, -1))
```

To enable the tracker to expose `_last_rally_sub_tracks`, also modify `process_rally` to set:
```python
self._last_rally_sub_tracks = sub_tracks if sub_tracks else []
```
Just before `return RallyTrackingResult(...)`.

- [ ] **Step 8: Add an integration test for end-to-end translation**

Append to `test_track_segmentation.py`:

```python
from rallycut.cli.commands.match_players import _build_per_frame_pid_map


def test_per_frame_pid_map_handles_excluded_frames():
    sub_a = _make_sub(2, 0, 0, 100, argmax_pid=2, margin=0.10)
    sub_b = _make_sub(3, 0, 50, 150, argmax_pid=2, margin=0.40)
    sub_a.excluded_frames = set(range(50, 101))
    track_to_player = {
        sub_a.synthetic_track_id: 2,
        sub_b.synthetic_track_id: 2,
    }
    pf = _build_per_frame_pid_map(track_to_player, [sub_a, sub_b])
    # sub_a's non-overlap frames retained
    assert pf.get((2, 25)) == 2
    # sub_a's overlap frames excluded
    assert pf.get((2, 75)) is None
    # sub_b's full range mapped
    assert pf.get((3, 75)) == 2
    assert pf.get((3, 140)) == 2
```

- [ ] **Step 9: Run all tests**

```bash
cd analysis
uv run pytest tests/unit -x -v
```
Expected: all green.

- [ ] **Step 10: Commit**

```bash
git add analysis/rallycut/tracking/_subtrack.py \
        analysis/rallycut/tracking/match_tracker.py \
        analysis/rallycut/cli/commands/match_players.py \
        analysis/tests/unit/test_track_segmentation.py
git commit -m "feat(tracking): per-frame conflict resolver + per-frame pid translator for sub-tracks"
```

---

## Task 6: Wire `PlayerReIDClassifier` and crop extractor into the CLI

**Files:**
- Modify: `analysis/rallycut/cli/commands/match_players.py` (load classifier + crop extractor, attach to tracker)
- Modify: `analysis/tests/unit/test_track_segmentation.py` (smoke test the wiring)

- [ ] **Step 1: Add a smoke test that the CLI training path is invoked when flag is on**

Append to `test_track_segmentation.py`:

```python
def test_match_players_attaches_classifier_when_flag_on(monkeypatch, tmp_path):
    """Smoke: when ENABLE_REF_CROP_TRACK_SPLIT=1 and ref crops exist, the CLI
    should train PlayerReIDClassifier and attach `_few_shot_classifier`/
    `_crop_extractor` to the tracker."""
    monkeypatch.setenv("ENABLE_REF_CROP_TRACK_SPLIT", "1")

    from rallycut.cli.commands.match_players import _maybe_attach_classifier

    class _FakeTracker:
        frozen_player_ids = {1, 2, 3, 4}

    tracker = _FakeTracker()
    crops_by_pid = {1: [_stub_crop(0, 0)], 2: [_stub_crop(0, 1)], 3: [_stub_crop(0, 2)], 4: [_stub_crop(0, 3)]}
    video_path = tmp_path / "fake.mp4"
    video_path.write_bytes(b"")  # crop_extractor will return None on missing video — fine for smoke

    _maybe_attach_classifier(tracker, crops_by_pid, video_path)
    assert getattr(tracker, "_few_shot_classifier", None) is not None
    assert getattr(tracker, "_crop_extractor", None) is not None
```

- [ ] **Step 2: Run, expect ImportError**

```bash
cd analysis
uv run pytest tests/unit/test_track_segmentation.py -x -v -k "attaches_classifier"
```
Expected: 1 fail.

- [ ] **Step 3: Implement `_maybe_attach_classifier` in `match_players.py`**

Add to `analysis/rallycut/cli/commands/match_players.py` (top-level function):

```python
import os
from typing import Any
from rallycut.tracking.reid_embeddings import (
    PlayerReIDClassifier,
    extract_crops_from_video,
)


def _maybe_attach_classifier(
    tracker: Any,
    crops_by_pid: dict[int, list[Any]],
    video_path: "Path | str",
) -> None:
    """Train the few-shot classifier from ref crops and attach it + a crop
    extractor to the tracker. No-op when the flag is off or fewer than 2 pids
    have crops.
    """
    if os.environ.get("ENABLE_REF_CROP_TRACK_SPLIT", "0") != "1":
        return
    pids_with_crops = [p for p, cs in crops_by_pid.items() if cs]
    if len(pids_with_crops) < 2:
        return
    clf = PlayerReIDClassifier()
    clf.train({pid: crops_by_pid[pid] for pid in pids_with_crops}, augmentations_per_crop=20, epochs=60)
    tracker._few_shot_classifier = clf

    # Crop extractor: opens the video on first call, then re-uses the capture.
    import cv2

    state: dict[str, Any] = {"cap": None}

    def crop_extractor(track_id: int, frame_number: int):
        # frame_number is rally-relative; caller (process_rally → splitter) provides
        # rally-relative frames. We need the absolute video frame, so the extractor
        # also needs the rally start. For now, accept rally-absolute by setting
        # tracker._rally_start_frame_in_video on each rally entry; if missing, the
        # extractor falls back to using `frame_number` directly.
        rally_start = getattr(tracker, "_rally_start_frame_in_video", 0)
        abs_frame = rally_start + frame_number
        # Look up bbox from positions cache.
        positions_cache = getattr(tracker, "_current_rally_positions_by_tid_frame", {})
        bbox = positions_cache.get((track_id, frame_number))
        if bbox is None:
            return None
        if state["cap"] is None:
            state["cap"] = cv2.VideoCapture(str(video_path))
        cap = state["cap"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, abs_frame)
        ok, img = cap.read()
        if not ok or img is None:
            return None
        h, w = img.shape[:2]
        x1 = max(0, int((bbox["x"] - bbox["width"] / 2) * w))
        y1 = max(0, int((bbox["y"] - bbox["height"] / 2) * h))
        x2 = min(w, int((bbox["x"] + bbox["width"] / 2) * w))
        y2 = min(h, int((bbox["y"] + bbox["height"] / 2) * h))
        if x2 <= x1 + 4 or y2 <= y1 + 4:
            return None
        crop = img[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 16 or crop.shape[1] < 8:
            return None
        return crop

    tracker._crop_extractor = crop_extractor
```

In the CLI `match_players` command body, after the tracker is constructed and ref crops are loaded (search for `extract_crops_from_video` or `crops_by_pid`), add:

```python
_maybe_attach_classifier(tracker, crops_by_pid, video_path)
```

And just before each `tracker.process_rally(...)` call, set:

```python
tracker._rally_start_frame_in_video = int(round(rally.start_ms / 1000.0 * proxy_fps))
tracker._current_rally_positions_by_tid_frame = {
    (int(p["trackId"]), int(p["frameNumber"])): p
    for p in rally_positions
}
```

- [ ] **Step 4: Run smoke test**

```bash
cd analysis
uv run pytest tests/unit/test_track_segmentation.py -x -v -k "attaches_classifier"
```
Expected: PASS.

- [ ] **Step 5: Smoke-run match-players on cuco with flag ON, verify it still completes**

```bash
cd analysis
ENABLE_REF_CROP_TRACK_SPLIT=1 uv run rallycut match-players d3486f0b-6601-43d8-a60f-6ecc4874c408 -o /tmp/cuco_match_flag_on.json 2>&1 | tee /tmp/cuco_match_flag_on.log
```
Expected: process completes; log contains at least one `Within-track split: tid=...` line for rally 5 (`106e295e`); writes JSON; exit 0.

- [ ] **Step 6: Run with flag OFF on the same fixture, confirm zero diff in output**

```bash
cd analysis
ENABLE_REF_CROP_TRACK_SPLIT=0 uv run rallycut match-players d3486f0b-6601-43d8-a60f-6ecc4874c408 -o /tmp/cuco_match_flag_off.json
diff /tmp/cuco_match_flag_off.json /tmp/cuco_match_flag_off.json.expected || echo "OK if no diff vs prior run"
```
Make `cuco_match_flag_off.json.expected` from a fresh prior run with flag absent. Expected: byte-identical to baseline run.

- [ ] **Step 7: Commit**

```bash
git add analysis/rallycut/cli/commands/match_players.py analysis/tests/unit/test_track_segmentation.py
git commit -m "feat(cli): attach few-shot classifier + crop extractor to MatchPlayerTracker behind flag"
```

---

## Task 7: A/B test on 9-fixture baseline + go/no-go decision

**Files:**
- Create: `analysis/scripts/ab_test_track_split.py`
- Output: `analysis/reports/attribution_rebuild/track_split_ab_2026_04_26.json`

- [ ] **Step 1: Write the A/B harness**

Create `analysis/scripts/ab_test_track_split.py`:

```python
"""A/B test pre-Hungarian within-track segmentation against the v3 baseline.

For each of the 9 fixtures:
  1. Run match-players + reattribute-actions with ENABLE_REF_CROP_TRACK_SPLIT=0
     → "off" arm (must equal baseline_v3.json modulo nondeterminism).
  2. Run match-players + reattribute-actions with ENABLE_REF_CROP_TRACK_SPLIT=1
     → "on" arm.
  3. Score Surface A on both arms using the same scorer as lock_baseline_v2.

Output: per-fixture deltas (player_attr, serve_attr, wrong) + aggregate deltas
+ pre-registered ship-gate evaluation.

Usage:
    cd analysis
    uv run python scripts/ab_test_track_split.py
    uv run python scripts/ab_test_track_split.py --fixtures cuco wawa
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_ANALYSIS_DIR = Path(__file__).resolve().parents[1]
if str(_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS_DIR))

ATTR_REPORTS = _ANALYSIS_DIR / "reports" / "attribution_rebuild"
FIXTURE_REGISTRY = ATTR_REPORTS / "fixture_video_ids_2026_04_24.json"
BASELINE_PATH = ATTR_REPORTS / "baseline_v3.json"
OUT_PATH = ATTR_REPORTS / "track_split_ab_2026_04_26.json"

# Pre-registered gates.
GATE_PER_FIXTURE_PLAYER_ATTR = -0.005  # ≥ -0.5pp
GATE_AGGREGATE_PLAYER_ATTR = +0.005     # ≥ +0.5pp
GATE_AGGREGATE_SERVE_ATTR = -0.005      # ≥ -0.5pp
GATE_AGGREGATE_WRONG = 0.0              # ≤ +0.0pp


def _run_arm(video_id: str, arm: str, log_path: Path) -> dict[str, Any]:
    env = os.environ.copy()
    env["ENABLE_REF_CROP_TRACK_SPLIT"] = "1" if arm == "on" else "0"
    cmd = ["uv", "run", "rallycut", "match-players", video_id]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as fh:
        fh.write(f"$ ENABLE_REF_CROP_TRACK_SPLIT={env['ENABLE_REF_CROP_TRACK_SPLIT']} {' '.join(shlex.quote(c) for c in cmd)}\n")
        proc = subprocess.run(cmd, cwd=_ANALYSIS_DIR, capture_output=True, text=True, env=env)
        fh.write(proc.stdout)
        if proc.stderr:
            fh.write("\n--- stderr ---\n" + proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"match-players ({arm}) failed for {video_id}: {proc.returncode}")
    cmd2 = ["uv", "run", "rallycut", "reattribute-actions", video_id]
    log2 = log_path.with_suffix(".reattr.log")
    with log2.open("w") as fh:
        fh.write(f"$ {' '.join(shlex.quote(c) for c in cmd2)}\n")
        proc = subprocess.run(cmd2, cwd=_ANALYSIS_DIR, capture_output=True, text=True, env=env)
        fh.write(proc.stdout)
        if proc.stderr:
            fh.write("\n--- stderr ---\n" + proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"reattribute-actions ({arm}) failed for {video_id}: {proc.returncode}")
    # Re-import scorer fresh per run to pick up DB state.
    from rallycut.evaluation.attribution_bench import aggregate, score_rally
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from measure_relabel_lift import (  # type: ignore[import-untyped]
        _team_templates_pid_to_team,
        _ttp_by_rally_id,
        load_rallies_for_surface_a,
    )
    rallies = load_rallies_for_surface_a(video_id)
    ttp_by_rally = _ttp_by_rally_id(video_id)
    team_templates = _team_templates_pid_to_team(video_id)
    per_rally = []
    for rally in rallies:
        scored = score_rally(rally, ttp_by_rally.get(rally["rally_id"], {}), team_templates)
        per_rally.append(scored)
    return aggregate(per_rally)


def _gate_evaluation(per_fixture: dict[str, dict[str, dict[str, float]]],
                     baseline_aggregate: dict[str, float],
                     ab_aggregate_on: dict[str, float]) -> dict[str, Any]:
    aggregate_player_attr_delta = ab_aggregate_on["player_attr"] - baseline_aggregate["player_attr"]
    aggregate_serve_attr_delta = ab_aggregate_on["serve_attr"] - baseline_aggregate["serve_attr"]
    aggregate_wrong_delta = ab_aggregate_on["wrong"] - baseline_aggregate["wrong"]
    per_fixture_player_attr_failures = []
    for fx, arms in per_fixture.items():
        d = arms["on"]["player_attr"] - arms["off"]["player_attr"]
        if d < GATE_PER_FIXTURE_PLAYER_ATTR:
            per_fixture_player_attr_failures.append((fx, d))
    return {
        "aggregate_player_attr_delta": aggregate_player_attr_delta,
        "aggregate_serve_attr_delta": aggregate_serve_attr_delta,
        "aggregate_wrong_delta": aggregate_wrong_delta,
        "per_fixture_player_attr_failures": per_fixture_player_attr_failures,
        "gate_per_fixture_player_attr": GATE_PER_FIXTURE_PLAYER_ATTR,
        "gate_aggregate_player_attr": GATE_AGGREGATE_PLAYER_ATTR,
        "gate_aggregate_serve_attr": GATE_AGGREGATE_SERVE_ATTR,
        "gate_aggregate_wrong": GATE_AGGREGATE_WRONG,
        "all_gates_pass": (
            not per_fixture_player_attr_failures
            and aggregate_player_attr_delta >= GATE_AGGREGATE_PLAYER_ATTR
            and aggregate_serve_attr_delta >= GATE_AGGREGATE_SERVE_ATTR
            and aggregate_wrong_delta <= GATE_AGGREGATE_WRONG
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", nargs="*", default=None)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    with open(FIXTURE_REGISTRY) as f:
        registry = json.load(f)
    fixtures = registry["fixtures"]
    if args.fixtures:
        fixtures = {k: fixtures[k] for k in args.fixtures if k in fixtures}

    with open(BASELINE_PATH) as f:
        baseline = json.load(f)

    per_fixture: dict[str, dict[str, dict[str, float]]] = {}
    log_dir = ATTR_REPORTS / "track_split_ab_logs"
    print(f"Running A/B on {len(fixtures)} fixtures…")
    for i, (name, meta) in enumerate(fixtures.items(), 1):
        vid = meta["video_id"]
        print(f"[{i}/{len(fixtures)}] {name}  video_id={vid[:8]}…")
        off_metrics = _run_arm(vid, "off", log_dir / f"{name}_off.log")
        on_metrics = _run_arm(vid, "on", log_dir / f"{name}_on.log")
        per_fixture[name] = {"off": off_metrics, "on": on_metrics}
        delta = on_metrics["player_attr"] - off_metrics["player_attr"]
        print(f"      off player_attr={off_metrics['player_attr']:.4f}  on={on_metrics['player_attr']:.4f}  Δ={delta:+.4f}")

    # Aggregate the "on" arm across fixtures.
    total = {"player_attr": 0.0, "serve_attr": 0.0, "wrong": 0.0, "n_actions": 0}
    for arms in per_fixture.values():
        for k in ("player_attr", "serve_attr", "wrong"):
            total[k] += arms["on"][k] * arms["on"]["n_actions"]
        total["n_actions"] += arms["on"]["n_actions"]
    if total["n_actions"]:
        for k in ("player_attr", "serve_attr", "wrong"):
            total[k] /= total["n_actions"]

    gates = _gate_evaluation(per_fixture, baseline["aggregate"], total)
    out = {
        "baseline_path": str(BASELINE_PATH),
        "ab_aggregate_on": total,
        "per_fixture": per_fixture,
        "gates": gates,
        "ran_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out}")
    print(f"Aggregate Δplayer_attr = {gates['aggregate_player_attr_delta']:+.4f} (gate ≥ {GATE_AGGREGATE_PLAYER_ATTR:+.4f})")
    print(f"Aggregate Δserve_attr  = {gates['aggregate_serve_attr_delta']:+.4f} (gate ≥ {GATE_AGGREGATE_SERVE_ATTR:+.4f})")
    print(f"Aggregate Δwrong        = {gates['aggregate_wrong_delta']:+.4f} (gate ≤ {GATE_AGGREGATE_WRONG:+.4f})")
    print(f"Per-fixture failures: {gates['per_fixture_player_attr_failures']}")
    print(f"\n{'PASS' if gates['all_gates_pass'] else 'FAIL'} — all gates {'cleared' if gates['all_gates_pass'] else 'not cleared'}.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test on 1 fixture**

```bash
cd analysis
uv run python scripts/ab_test_track_split.py --fixtures cuco --out /tmp/ab_smoke.json
```
Expected: completes, prints aggregate delta line, writes JSON.

- [ ] **Step 3: Run full A/B**

```bash
cd analysis
uv run python scripts/ab_test_track_split.py 2>&1 | tee reports/attribution_rebuild/track_split_ab_2026_04_26.log
```
Expected duration: ~30-40 minutes (9 fixtures × 2 arms × ~2 min/run).

- [ ] **Step 4: Inspect gate decision**

```bash
cd analysis
uv run python -c "
import json
with open('reports/attribution_rebuild/track_split_ab_2026_04_26.json') as f:
    r = json.load(f)
g = r['gates']
print('all_gates_pass:', g['all_gates_pass'])
print('aggregate_player_attr_delta:', f\"{g['aggregate_player_attr_delta']:+.4f}\")
print('aggregate_serve_attr_delta:', f\"{g['aggregate_serve_attr_delta']:+.4f}\")
print('aggregate_wrong_delta:', f\"{g['aggregate_wrong_delta']:+.4f}\")
print('per-fixture failures:', g['per_fixture_player_attr_failures'])
print()
print('per-fixture:')
for fx, arms in r['per_fixture'].items():
    d = arms['on']['player_attr'] - arms['off']['player_attr']
    print(f'  {fx}: off={arms[\"off\"][\"player_attr\"]:.4f}  on={arms[\"on\"][\"player_attr\"]:.4f}  Δ={d:+.4f}  n={arms[\"on\"][\"n_actions\"]}')
"
```

- [ ] **Step 5: Decision**

If `all_gates_pass == true`:
- Set `ENABLE_REF_CROP_TRACK_SPLIT=1` as the default by changing the env-var fallback in `_maybe_segment_tracks_by_appearance` from `"0"` to `"1"`.
- Add a memory entry recording the win.
- Commit.

If `all_gates_pass == false`:
- Leave the flag default at `"0"`.
- Add a memory entry recording the no-go with per-fixture failures and aggregate deltas.
- Open a follow-up plan for any worth-pursuing direction (e.g. tighter gate, per-segment HSV re-extraction, post-Hungarian repair instead of pre-Hungarian split).

- [ ] **Step 6: Commit results regardless of outcome**

```bash
git add analysis/scripts/ab_test_track_split.py \
        analysis/reports/attribution_rebuild/track_split_ab_2026_04_26.json \
        analysis/reports/attribution_rebuild/track_split_ab_2026_04_26.log \
        analysis/reports/attribution_rebuild/track_split_ab_logs/
git commit -m "eval: A/B pre-Hungarian within-track split vs v3 baseline"
```

If gates passed AND default was changed:
```bash
git add analysis/rallycut/tracking/match_tracker.py
git commit -m "feat(tracking): enable pre-Hungarian within-track split by default after A/B win"
```

---

## Self-Review (skill checklist)

- **Spec coverage:** All five fix layers from the diagnostic conversation are covered: (a) within-track flip detection (Task 3), (b) per-segment aggregation (Task 3), (c) strict per-segment margin gate (Task 3), (d) per-frame uniqueness when sub-tracks overlap (Task 5), (e) flag-gated rollout with A/B (Tasks 4+7). The pre-registered gates from the conversation summary (per-fixture ≥ −0.5pp, aggregate ≥ +0.5pp on player_attr, no wrong-rate increase) are baked into Task 7.
- **Placeholder scan:** All steps contain real code, exact bash, expected output. `Path | str` typing is concrete; no "TBD" / "implement later" / "similar to Task N".
- **Type consistency:** `SubTrackCandidate` properties (`parent_track_id`, `segment_index`, `f_start`, `f_end`, `appearance_stats`, `aggregated_argmax_pid`, `aggregated_margin`, `excluded_frames`, `synthetic_track_id`, `overlaps`) are used consistently across Tasks 2-7. Module-level constants (`SEGMENT_NUM_WINDOWS`, `SEGMENT_MIN_WINDOW_FRAMES`, `SEGMENT_MIN_PER_SEGMENT_MARGIN`, `SEGMENT_MIN_CONFIRMING_WINDOWS`, `SEGMENT_FRAMES_PER_WINDOW`) are defined in Task 3 Step 3 and referenced from there onward. Method names (`_segment_tracks_by_appearance`, `_maybe_segment_tracks_by_appearance`, `_merge_sub_tracks_into_candidates`, `_resolve_subtrack_pid_conflicts`, `_find_segment_flip`, `_aggregate_segment_classifier`, `_stats_for_positions`, `_build_per_frame_pid_map`, `_maybe_attach_classifier`) are stable across tasks.
- **Reversibility:** The flag default is `0` until Task 7 explicitly flips it after gate verification. Without the flag, every code path is no-op (`_maybe_segment_tracks_by_appearance` returns `[]`, `_merge_sub_tracks_into_candidates` is never called, `_resolve_subtrack_pid_conflicts` is never called). This means the implementation is safe to land incrementally without affecting production output.
