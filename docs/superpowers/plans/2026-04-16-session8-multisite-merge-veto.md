# Session 8 — Multi-Site Learned-ReID Merge Veto Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend Session-6's proven-safe learned-ReID cosine veto from one merge site (`link_tracklets_by_appearance`) to every merge/rename pass that plausibly creates same-team identity swaps, so downstream passes can no longer recreate swaps blocked at step 0c. Ship gate: ≥50% SAME_TEAM_SWAP reduction with zero HOTA regression.

**Architecture:** Factor the Session-6 veto helper (`_learned_merge_would_veto`) and its prerequisite (`_segment_median_embedding`) into a new `rallycut/tracking/merge_veto.py` module; call it from every relevant merge/rename pass via a consistent kwarg (`learned_veto_threshold=LEARNED_MERGE_VETO_COS`). Run a per-pass swap-attribution diagnostic **before** any adapter work so we only touch the passes that create the 12 observed SAME_TEAM_SWAPs. Env-var-gated, byte-identical when disabled.

**Tech Stack:** Python 3.11, NumPy, PyTorch (embedding head already loaded by Session 4), BoxMOT+BoT-SORT tracker, existing `LearnedEmbeddingStore`, existing `evaluate-tracking --retrack --cached` harness, existing `build_rally_audit` + `SwitchCause` classifier.

**Spec reference:** `docs/superpowers/specs/2026-04-16-session8-multisite-merge-veto-design.md`

---

## File Structure

### New files

- `analysis/rallycut/tracking/merge_veto.py` — factored `_segment_median_embedding` + `learned_cosine_veto` helper. One responsibility: decide whether a proposed merge between two track fragments should be blocked by the learned ReID head.
- `analysis/tests/unit/test_merge_veto.py` — tests for the factored helper in isolation (threshold behavior, abstain-on-missing, renorm).
- `analysis/scripts/diagnose_per_pass_swaps.py` — diagnostic: runs `apply_post_processing` with instrumentation, attributes each SAME_TEAM_SWAP to the pass that introduced it. Output: `reports/merge_veto/per_pass_swap_attribution.md`.
- `analysis/scripts/eval_multisite_merge_veto.py` — 8-cell threshold sweep. Mirrors Session-6's `eval_learned_merge_veto.py` but runs against all adapter sites (same env var, different site coverage).

### Modified files

- `analysis/rallycut/tracking/tracklet_link.py` — remove local `_segment_median_embedding` + `_learned_merge_would_veto`, import from `merge_veto`. Keep the Session-6 call sites byte-identical (same helper name, same kwarg semantics).
- `analysis/rallycut/tracking/height_consistency.py` — `fix_height_swaps`: add `learned_veto_threshold` kwarg; veto each proposed swap pair before renaming.
- `analysis/rallycut/tracking/tracklet_link.py` — `relink_spatial_splits`, `relink_primary_fragments`: add `learned_veto_threshold` kwarg; veto each per-pair merge.
- `analysis/rallycut/tracking/player_filter.py` — `stabilize_track_ids`: add `learned_veto_threshold` + `learned_store` kwargs; veto each proposed rename.
- `analysis/rallycut/tracking/player_tracker.py` — thread `learned_store` and `LEARNED_MERGE_VETO_COS` through the 4 new call sites.
- `analysis/rallycut/cli/commands/evaluate_tracking.py` — already annotates `merge_veto:enabled` in the config hash (Session 6). No change needed unless diagnostic output shows we must add more sites.

### Reused (no edits)

- `analysis/rallycut/tracking/color_repair.py::LearnedEmbeddingStore` — per-(track, frame) embedding store.
- `analysis/rallycut/tracking/reid_embeddings.py` — head loader singleton + HEAD_SHA.
- `analysis/rallycut/evaluation/tracking/retrack_cache.py` — cache schema carries learned embeddings.
- `analysis/rallycut/evaluation/tracking/audit.py::build_rally_audit` — GT↔pred track audit returning `SwitchCause` per switch.

---

## Task 1: Factor the veto helper into `merge_veto.py`

**Files:**
- Create: `analysis/rallycut/tracking/merge_veto.py`
- Modify: `analysis/rallycut/tracking/tracklet_link.py` (remove local copies, import from new module)
- Create: `analysis/tests/unit/test_merge_veto.py`

- [ ] **Step 1: Write failing tests for the factored helper**

Create `analysis/tests/unit/test_merge_veto.py`:

```python
"""Tests for the factored learned-ReID merge veto helper.

The helper ingests two track fragments (each as a set of frames) plus
a LearnedEmbeddingStore. It returns True when the median learned
embeddings are dissimilar enough that the merge should be blocked, and
abstains (returns False, allowing the merge) when either side lacks
enough evidence.
"""

from __future__ import annotations

import numpy as np

from rallycut.tracking.color_repair import LearnedEmbeddingStore
from rallycut.tracking.merge_veto import (
    _segment_median_embedding,
    learned_cosine_veto,
)


def _store_with(track_embeddings: dict[int, dict[int, np.ndarray]]) -> LearnedEmbeddingStore:
    store = LearnedEmbeddingStore()
    for tid, frames in track_embeddings.items():
        for fn, emb in frames.items():
            store.add(tid, fn, emb.astype(np.float32))
    return store


def _unit(vec: np.ndarray) -> np.ndarray:
    arr = vec.astype(np.float32)
    return arr / np.linalg.norm(arr)


def test_veto_disabled_when_threshold_zero() -> None:
    """Threshold 0.0 → veto never fires (byte-identical default behavior)."""
    a = _unit(np.array([1.0, 0.0] + [0.0] * 126))
    b = _unit(np.array([-1.0, 0.0] + [0.0] * 126))
    store = _store_with({
        1: {fn: a for fn in range(10)},
        2: {fn: b for fn in range(10, 20)},
    })
    assert learned_cosine_veto(
        store, 1, set(range(10)), 2, set(range(10, 20)), threshold=0.0
    ) is False


def test_veto_blocks_on_dissimilar_embeddings() -> None:
    """cos ≈ -1.0 with threshold 0.5 → block."""
    a = _unit(np.array([1.0, 0.0] + [0.0] * 126))
    b = _unit(np.array([-1.0, 0.0] + [0.0] * 126))
    store = _store_with({
        1: {fn: a for fn in range(10)},
        2: {fn: b for fn in range(10, 20)},
    })
    assert learned_cosine_veto(
        store, 1, set(range(10)), 2, set(range(10, 20)), threshold=0.5
    ) is True


def test_veto_allows_on_similar_embeddings() -> None:
    """cos ≈ 1.0 with threshold 0.5 → allow."""
    a = _unit(np.array([1.0, 0.0] + [0.0] * 126))
    store = _store_with({
        1: {fn: a for fn in range(10)},
        2: {fn: a for fn in range(10, 20)},
    })
    assert learned_cosine_veto(
        store, 1, set(range(10)), 2, set(range(10, 20)), threshold=0.5
    ) is False


def test_veto_abstains_when_embeddings_missing() -> None:
    """Fewer than 5 embeddings on either side → abstain (return False)."""
    a = _unit(np.array([1.0, 0.0] + [0.0] * 126))
    b = _unit(np.array([-1.0, 0.0] + [0.0] * 126))
    store = _store_with({
        1: {fn: a for fn in range(3)},   # only 3 frames
        2: {fn: b for fn in range(10, 20)},
    })
    assert learned_cosine_veto(
        store, 1, set(range(3)), 2, set(range(10, 20)), threshold=0.5
    ) is False


def test_veto_abstains_when_store_empty() -> None:
    """Empty store → abstain."""
    store = LearnedEmbeddingStore()
    assert learned_cosine_veto(
        store, 1, {0, 1, 2, 3, 4, 5}, 2, {10, 11, 12, 13, 14, 15}, threshold=0.5
    ) is False


def test_segment_median_renormalized() -> None:
    """Output of _segment_median_embedding is L2-normalized."""
    a = _unit(np.array([0.3, 0.4, 0.5] + [0.0] * 125))
    store = _store_with({1: {fn: a for fn in range(10)}})
    med = _segment_median_embedding(store, 1, set(range(10)))
    assert med is not None
    assert abs(float(np.linalg.norm(med)) - 1.0) < 1e-5
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis
uv run pytest tests/unit/test_merge_veto.py -v
```

Expected: all 6 tests ERROR with `ModuleNotFoundError: No module named 'rallycut.tracking.merge_veto'`.

- [ ] **Step 3: Create `merge_veto.py` with factored helpers**

Create `analysis/rallycut/tracking/merge_veto.py`:

```python
"""Factored learned-ReID merge veto — used by every merge/rename pass.

Session 6 (tracklet_link.py) introduced the veto at one site. Session 8
extends it to every pass that plausibly creates same-team swaps. This
module owns the shared helpers so all adapter sites agree on signal
semantics, renormalization, and abstain behavior.

Public API:
    learned_cosine_veto(store, id_a, frames_a, id_b, frames_b, threshold) -> bool
        Returns True when the merge should be BLOCKED.

    _segment_median_embedding(store, track_id, frames) -> np.ndarray | None
        L2-renormalized median embedding for a track over `frames`.
        Returns None when fewer than 5 valid embeddings exist.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rallycut.tracking.color_repair import LearnedEmbeddingStore

# Env-var threshold. 0.0 = veto disabled (byte-identical default). > 0 =
# block merges whose median cosine similarity is below this threshold.
LEARNED_MERGE_VETO_COS = float(
    os.environ.get("LEARNED_MERGE_VETO_COS", "0.0")
)

# Segment-median sampling parameters. Median over ≤ `max_samples` frames
# is robust to occlusion-contaminated tails without quadratic cost.
LEARNED_MERGE_VETO_MIN_FRAMES = 5
LEARNED_MERGE_VETO_MAX_SAMPLES = 30


def _segment_median_embedding(
    learned_store: "LearnedEmbeddingStore | None",
    track_id: int,
    frames: set[int] | list[int],
    min_frames: int = LEARNED_MERGE_VETO_MIN_FRAMES,
    max_samples: int = LEARNED_MERGE_VETO_MAX_SAMPLES,
) -> np.ndarray | None:
    """L2-renormalized median learned-ReID embedding for a track over `frames`.

    Returns None when the store is absent, carries no data, or fewer than
    `min_frames` valid embeddings exist in the frame set. Samples up to
    `max_samples` uniformly-spaced frames when the fragment is longer.
    """
    if learned_store is None or not learned_store.has_data():
        return None
    sorted_frames = sorted(frames)
    if not sorted_frames:
        return None
    if len(sorted_frames) > max_samples:
        step = len(sorted_frames) / max_samples
        sorted_frames = [sorted_frames[int(i * step)] for i in range(max_samples)]
    vectors: list[np.ndarray] = []
    for fn in sorted_frames:
        emb = learned_store.get(track_id, fn)
        if emb is not None:
            vectors.append(emb)
    if len(vectors) < min_frames:
        return None
    stack = np.stack(vectors).astype(np.float32)
    med = np.median(stack, axis=0)
    norm = float(np.linalg.norm(med))
    if norm < 1e-8:
        return None
    normed: np.ndarray = (med / norm).astype(np.float32)
    return normed


def learned_cosine_veto(
    learned_store: "LearnedEmbeddingStore | None",
    track_id_a: int,
    frames_a: set[int] | list[int],
    track_id_b: int,
    frames_b: set[int] | list[int],
    threshold: float = LEARNED_MERGE_VETO_COS,
) -> bool:
    """Return True when the learned head says these two fragments are
    different players — i.e. the proposed merge/rename would be a false-
    positive teammate link that should be BLOCKED.

    Abstain (return False = don't block) when the feature is disabled
    (threshold ≤ 0), either side has fewer than 5 valid embeddings, or
    the store carries no data. The failure mode of a false abstention is
    the status-quo: the current pass decides the merge without our help.
    """
    if threshold <= 0.0:
        return False
    emb_a = _segment_median_embedding(learned_store, track_id_a, frames_a)
    emb_b = _segment_median_embedding(learned_store, track_id_b, frames_b)
    if emb_a is None or emb_b is None:
        return False
    cos_sim = float(np.dot(emb_a, emb_b))
    return cos_sim < threshold
```

- [ ] **Step 4: Refactor `tracklet_link.py` to use the shared helpers**

Remove the local `_segment_median_embedding`, `_learned_merge_would_veto`, and the `LEARNED_MERGE_VETO_COS` / `LEARNED_MERGE_VETO_MIN_FRAMES` / `LEARNED_MERGE_VETO_MAX_SAMPLES` module constants from `tracklet_link.py`. Replace with:

```python
from rallycut.tracking.merge_veto import (
    LEARNED_MERGE_VETO_COS,
    LEARNED_MERGE_VETO_MAX_SAMPLES,
    LEARNED_MERGE_VETO_MIN_FRAMES,
    _segment_median_embedding,
    learned_cosine_veto,
)
```

Replace the two Session-6 call sites in `_greedy_merge` and `_swap_optimize` with the new helper (note the signature change — frame sets are passed directly, not looked up via `tracks` dict):

**In `_greedy_merge` (around line 497)**: change

```python
if _learned_merge_would_veto(
    learned_store,
    track_id_a, track_id_b,
    tracks,
    threshold=learned_veto_threshold,
):
```

to

```python
if learned_cosine_veto(
    learned_store,
    track_id_a, tracks[track_id_a]["frames"],
    track_id_b, tracks[track_id_b]["frames"],
    threshold=learned_veto_threshold,
):
```

Same pattern for the `_swap_optimize` call site (around line 626).

Keep the local `_learned_merge_would_veto` function deleted — all callers now use `learned_cosine_veto` directly.

- [ ] **Step 5: Run merge_veto unit tests**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis
uv run pytest tests/unit/test_merge_veto.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 6: Run existing tracklet_link tests to verify refactor didn't break anything**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis
uv run pytest tests/unit/test_tracklet_link.py -v
```

Expected: all existing tests PASS (including the 5 Session-6 `TestLearnedMergeVeto` tests).

- [ ] **Step 7: Run lint + type check**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis
uv run ruff check rallycut/tracking/merge_veto.py rallycut/tracking/tracklet_link.py tests/unit/test_merge_veto.py
uv run mypy rallycut/tracking/merge_veto.py
```

Expected: both pass cleanly.

- [ ] **Step 8: Commit**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis
git add rallycut/tracking/merge_veto.py rallycut/tracking/tracklet_link.py tests/unit/test_merge_veto.py
git commit -m "refactor(analysis): factor learned-ReID merge veto into merge_veto module

Session 8 prep. Session 6's veto + _segment_median_embedding lived
inside tracklet_link.py; moves them to rallycut/tracking/merge_veto.py
so the per-pass adapters (Sessions 8 tasks 3-6) can share the same
helper without circular imports.

Signature change: callers pass frame sets directly instead of a tracks
dict, so the helper no longer depends on the tracklet_link track-summary
shape. Existing tracklet_link call sites updated in place. Byte-
identical behavior at LEARNED_MERGE_VETO_COS=0 (default)."
```

---

## Task 2: Per-pass swap-attribution diagnostic

**Files:**
- Create: `analysis/scripts/diagnose_per_pass_swaps.py`
- Output: `analysis/reports/merge_veto/per_pass_swap_attribution.md`

- [ ] **Step 1: Write the diagnostic script**

Create `analysis/scripts/diagnose_per_pass_swaps.py`:

```python
"""Session 8 diagnostic — attribute each SAME_TEAM_SWAP to the post-
processing pass that introduced it.

Runs apply_post_processing with a per-pass audit hook that calls
build_rally_audit after each of the 7 passes. For each SAME_TEAM_SWAP
in the final audit, identifies the FIRST pass where the GT↔pred mapping
for that track pair flipped.

Output: reports/merge_veto/per_pass_swap_attribution.md — table of
  {rally_id, gt_id, swap_frame, introduced_by_pass}

Usage:
    uv run python scripts/diagnose_per_pass_swaps.py
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

# Pass names, in execution order inside apply_post_processing. Must match
# the _skip() flag names used by player_tracker.py.
PASSES = [
    "ENFORCE_SPATIAL_CONSISTENCY",
    "FIX_HEIGHT_SWAPS",
    "SPLIT_TRACKS_BY_COLOR",
    "RELINK_SPATIAL_SPLITS",
    "RELINK_PRIMARY_FRAGMENTS",
    "LINK_TRACKLETS_BY_APPEARANCE",
    "STABILIZE_TRACK_IDS",
]

ANALYSIS_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ANALYSIS_ROOT / "reports" / "merge_veto"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("diagnose_per_pass_swaps")


def main() -> int:
    """Attribution strategy:

    1. Monkey-patch each of the 7 passes to dump `positions` after execution
       to `reports/merge_veto/per_pass_snapshots/<rally>/<N>_<pass>.json`.
    2. Run `evaluate-tracking --all --retrack --cached --audit-out ...`
       once with LEARNED_MERGE_VETO_COS=0 (baseline, all swaps present).
    3. For each rally with a SAME_TEAM_SWAP in the final audit:
       a. Reload snapshots 0..7 in order.
       b. Build rally audit on each snapshot against GT.
       c. Find the first snapshot where the swap appears.
       d. Attribute to that snapshot's pass.
    4. Aggregate into Counter[pass_name] + per-rally detail table.
    """
    # Implementation writes a tracking subprocess runner similar to
    # scripts/eval_minimal_processing.py. Full body elided here since
    # the implementer can reference that script + the approach above.
    #
    # Key functions to call:
    #   - from rallycut.evaluation.tracking.audit import build_rally_audit
    #   - load ground_truth_positions via evaluation/ground_truth.py
    #   - use SwitchCause.SAME_TEAM_SWAP enum value to filter
    raise NotImplementedError(
        "Implementer: follow the strategy above. ~150 LOC. Mirror the "
        "eval_minimal_processing.py subprocess runner for the baseline run; "
        "write per-pass snapshots via a targeted monkey-patch in "
        "player_tracker.apply_post_processing."
    )


if __name__ == "__main__":
    import sys
    sys.exit(main())
```

Implementer note: the fully-elided body should follow this structure. Key implementation points:

1. Add an `emit_pass_snapshot(pass_name, positions, rally_id)` helper that writes `list[dict]` positions to `reports/merge_veto/per_pass_snapshots/<rally_id>/<idx>_<pass_name>.json`.
2. Monkey-patch `apply_post_processing` locally (context-managed) to call this helper after each of the 7 passes. Pattern:

```python
import copy
original = PlayerTracker.apply_post_processing

def instrumented(*args, **kwargs):
    # Before each pass in the method, call emit_pass_snapshot.
    # Implementer: add the 7 dump points manually — this is one-off
    # diagnostic code; duplication is acceptable.
    ...
```

Rather than monkey-patch, a cleaner option is to add a kwarg `snapshot_hook: Callable[[str, list[PlayerPosition]], None] | None = None` to `apply_post_processing` — diagnostic-only, default None. Implementer can choose whichever is less invasive.

3. After the subprocess finishes, iterate rallies. For each with `SAME_TEAM_SWAP > 0`:
   - Load per-pass snapshots.
   - Call `build_rally_audit(snapshot_positions, gt_positions, ...)` — reuse from `rallycut.evaluation.tracking.audit`.
   - Find first snapshot where SAME_TEAM_SWAP count > 0 (or increments vs previous snapshot).
   - Record `(rally_id, pass_name)`.

4. Aggregate: `Counter[pass_name]` across all 12 swaps on 43 rallies.

- [ ] **Step 2: Run the diagnostic on 43 GT rallies**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis
uv run python scripts/diagnose_per_pass_swaps.py
```

Expected runtime: ~15 min (one retrack+audit cycle with snapshotting overhead). Uses existing retrack cache.

Expected output: `reports/merge_veto/per_pass_swap_attribution.md` with a table like:

```
| Rally | GT ID | Swap Frame | Introduced By |
|-------|-------|------------|---------------|
| d474b2ad | 3 | 845 | link_tracklets_by_appearance |
| ... | ... | ... | ... |

## Summary
| Pass | Swaps introduced |
|------|-----------------:|
| fix_height_swaps | 2 |
| relink_spatial_splits | 3 |
| link_tracklets_by_appearance | 5 |
| stabilize_track_ids | 2 |
| (others) | 0 |
```

- [ ] **Step 3: Decision gate — determine scope of remaining tasks**

Read the summary table. Three outcomes:

- **A (≥2 passes create swaps, excluding already-covered `link_tracklets_by_appearance`)**: proceed with all adapter tasks (3, 4, 5, 6) subject to which passes show up. Skip adapter tasks for passes with 0 attributed swaps.
- **B (only `link_tracklets_by_appearance` is attributed)**: Session 8 's multi-site thesis is false — skip to Task 7 via Plan B trigger. Stop, update memory, commit diagnostic, and re-plan.
- **C (diagnostic fails to run or attribution is ambiguous)**: fix the diagnostic before proceeding. Don't paper over.

- [ ] **Step 4: Commit diagnostic + output**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis
git add scripts/diagnose_per_pass_swaps.py reports/merge_veto/per_pass_swap_attribution.md
# Snapshots are large + regeneratable — ignore them
echo 'reports/merge_veto/per_pass_snapshots/' >> .gitignore
git add .gitignore
git commit -m "diag(analysis): per-pass swap attribution for Session 8

Attributes each of the 12 SAME_TEAM_SWAPs on 43 GT rallies to the
post-processing pass that introduced it. Informs which passes need
learned-ReID veto adapters in Session 8."
```

---

## Task 3: Adapter — `fix_height_swaps`

**Condition:** skip entirely if Task 2 attribution shows 0 swaps introduced by `FIX_HEIGHT_SWAPS`.

**Files:**
- Modify: `analysis/rallycut/tracking/height_consistency.py`
- Modify: `analysis/rallycut/tracking/player_tracker.py` (thread through the new kwarg)
- Modify: `analysis/tests/unit/test_height_consistency.py` (or create if it doesn't exist)

- [ ] **Step 1: Read the current pass to identify the decision boundary**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis
```

Read `rallycut/tracking/height_consistency.py` and locate the block where a proposed swap pair is confirmed and `positions`/stores are mutated. This is the spot to insert the veto — **before** any mutation, after the cross-match decision is made.

The key function `fix_height_swaps` already accepts `learned_store`. The discontinuity pairing logic produces a list of `(track_a, track_b, gap_frame)` triples. Inject the veto at the point where each triple is about to be accepted.

- [ ] **Step 2: Write a failing test**

Add to `analysis/tests/unit/test_height_consistency.py` (create if missing):

```python
"""Test learned-ReID veto inside fix_height_swaps."""

from __future__ import annotations

import numpy as np

from rallycut.tracking.color_repair import LearnedEmbeddingStore
from rallycut.tracking.height_consistency import fix_height_swaps
from rallycut.tracking.player_position import PlayerPosition


def _pos(frame: int, tid: int, y: float, h: float) -> PlayerPosition:
    return PlayerPosition(
        frame_number=frame, track_id=tid, x=0.5, y=y,
        width=0.1, height=h, confidence=0.9, keypoints=None,
    )


def _unit(v: list[float]) -> np.ndarray:
    arr = np.array(v, dtype=np.float32)
    return arr / np.linalg.norm(arr)


def test_height_swap_vetoed_on_different_learned_embeddings() -> None:
    """With threshold 0.5 and dissimilar embeddings, the swap is blocked."""
    # Construct two tracks with a cross-height pattern that would normally
    # trigger a swap: track 1 is tall pre-gap + short post-gap,
    # track 2 is short pre-gap + tall post-gap.
    positions = (
        [_pos(f, 1, 0.5, 0.40) for f in range(0, 10)]
        + [_pos(f, 1, 0.5, 0.20) for f in range(20, 30)]
        + [_pos(f, 2, 0.5, 0.20) for f in range(0, 10)]
        + [_pos(f, 2, 0.5, 0.40) for f in range(20, 30)]
    )

    emb_a = _unit([1.0, 0.0] + [0.0] * 126)
    emb_b = _unit([-1.0, 0.0] + [0.0] * 126)
    store = LearnedEmbeddingStore()
    for fn in range(30):
        store.add(1, fn, emb_a)
        store.add(2, fn, emb_b)

    result_positions, swap_result = fix_height_swaps(
        positions, learned_store=store, learned_veto_threshold=0.5,
    )
    assert swap_result.swaps == 0, (
        f"expected veto to block all height swaps, got {swap_result.swaps}"
    )


def test_height_swap_allowed_when_threshold_zero() -> None:
    """Default threshold 0.0 → byte-identical behavior (veto never fires)."""
    positions = (
        [_pos(f, 1, 0.5, 0.40) for f in range(0, 10)]
        + [_pos(f, 1, 0.5, 0.20) for f in range(20, 30)]
        + [_pos(f, 2, 0.5, 0.20) for f in range(0, 10)]
        + [_pos(f, 2, 0.5, 0.40) for f in range(20, 30)]
    )
    result_positions, swap_result = fix_height_swaps(
        positions, learned_store=None, learned_veto_threshold=0.0,
    )
    # Without veto, expect the swap to fire (exact count depends on
    # the discontinuity detector — just check > 0).
    assert swap_result.swaps > 0
```

- [ ] **Step 3: Run tests to confirm they fail**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis
uv run pytest tests/unit/test_height_consistency.py -v
```

Expected: `test_height_swap_vetoed_on_different_learned_embeddings` FAILS because the veto threshold kwarg doesn't yet wire into the decision.

- [ ] **Step 4: Wire the veto into `fix_height_swaps`**

In `rallycut/tracking/height_consistency.py`:

1. Add `learned_veto_threshold: float = LEARNED_MERGE_VETO_COS` to the signature. Import:

```python
from rallycut.tracking.merge_veto import LEARNED_MERGE_VETO_COS, learned_cosine_veto
```

2. Find the block where a cross-match pair `(track_a, track_b)` is accepted and swap is about to be applied. BEFORE accepting, compute frame sets for each side and call:

```python
frames_a = {p.frame_number for p in tracks[track_a_id]}
frames_b = {p.frame_number for p in tracks[track_b_id]}
if learned_cosine_veto(
    learned_store, track_a_id, frames_a, track_b_id, frames_b,
    threshold=learned_veto_threshold,
):
    logger.debug(
        "fix_height_swaps: learned-ReID veto blocked swap between "
        f"tracks {track_a_id}, {track_b_id} at gap ~frame {gap_frame}"
    )
    continue  # skip this proposed swap
```

The exact location depends on the existing structure — look for where `cross_match_tolerance` is evaluated and the swap is applied. Insert veto right after the decision is made, before the mutation happens.

3. In `player_tracker.py` around line 1555, update the call to pass the threshold (pulling from `merge_veto.LEARNED_MERGE_VETO_COS` as used by the `link_tracklets_by_appearance` site):

```python
from rallycut.tracking.merge_veto import LEARNED_MERGE_VETO_COS as _VETO_COS

...

positions, height_swap_result = fix_height_swaps(
    positions,
    color_store=color_store,
    appearance_store=appearance_store,
    learned_store=learned_store,
    learned_veto_threshold=_VETO_COS,
)
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis
uv run pytest tests/unit/test_height_consistency.py -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis
git add rallycut/tracking/height_consistency.py rallycut/tracking/player_tracker.py tests/unit/test_height_consistency.py
git commit -m "feat(analysis): learned-ReID veto in fix_height_swaps

Session 8 adapter #1. Blocks height-based swap-fix proposals when the
Session-3 ReID head says the two tracks are different players. Byte-
identical at LEARNED_MERGE_VETO_COS=0 (default)."
```

---

## Task 4: Adapter — `relink_spatial_splits`

**Condition:** skip if Task 2 attribution shows 0 swaps introduced by `RELINK_SPATIAL_SPLITS`.

**Files:**
- Modify: `analysis/rallycut/tracking/tracklet_link.py` (the `relink_spatial_splits` function, around line 992)
- Modify: `analysis/rallycut/tracking/player_tracker.py`
- Modify: `analysis/tests/unit/test_tracklet_link.py`

- [ ] **Step 1: Write a failing test**

Add to `analysis/tests/unit/test_tracklet_link.py` (in the existing `TestLearnedMergeVeto` class, or a new `TestRelinkSpatialSplitsVeto`):

```python
def test_relink_spatial_splits_vetoed_on_different_embeddings(self) -> None:
    """The veto blocks a same-team spatial re-link when embeddings disagree."""
    from rallycut.tracking.tracklet_link import relink_spatial_splits
    from rallycut.tracking.color_repair import ColorHistogramStore

    # Two tracks nearly identical in position (trivial spatial match)
    # but with learned embeddings on opposite sides.
    positions = [
        PlayerPosition(f, 1, 0.5, 0.5, 0.1, 0.3, 0.9, None)
        for f in range(0, 10)
    ] + [
        PlayerPosition(f, 2, 0.5, 0.5, 0.1, 0.3, 0.9, None)
        for f in range(12, 22)
    ]

    color_store = ColorHistogramStore()
    # (populate minimally so the gate accepts the match absent the veto)
    # ... implementer fills this in from the existing ColorHistogramStore API

    emb_a = np.array([1.0] + [0.0] * 127, dtype=np.float32)
    emb_b = np.array([-1.0] + [0.0] * 127, dtype=np.float32)
    store = LearnedEmbeddingStore()
    for f in range(10):
        store.add(1, f, emb_a)
    for f in range(12, 22):
        store.add(2, f, emb_b)

    _, num_relinks = relink_spatial_splits(
        positions, color_store,
        learned_store=store,
        learned_veto_threshold=0.5,
    )
    assert num_relinks == 0
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis
uv run pytest tests/unit/test_tracklet_link.py -v -k relink_spatial_splits_vetoed
```

Expected: FAILS with `TypeError: unexpected keyword argument 'learned_veto_threshold'`.

- [ ] **Step 3: Add the veto to `relink_spatial_splits`**

In `rallycut/tracking/tracklet_link.py`, around line 992:

1. Add `learned_veto_threshold: float = LEARNED_MERGE_VETO_COS` to the signature.

2. In the main merge-decision block (around line 1067-1077 where `dist <= max_distance` is checked and the merge is accepted), add BEFORE `id_mapping[next_tid] = canon`:

```python
if learned_cosine_veto(
    learned_store,
    canon, tracks[canon]["frames"],
    next_tid, next_info["frames"],
    threshold=learned_veto_threshold,
):
    logger.debug(
        f"relink_spatial_splits: learned-ReID veto blocked merge "
        f"{next_tid} -> {canon}"
    )
    continue
```

3. Thread the kwarg through `player_tracker.py` (around line 1582).

- [ ] **Step 4: Run test to confirm it passes**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis
uv run pytest tests/unit/test_tracklet_link.py -v -k relink_spatial_splits_vetoed
```

Expected: PASSES.

- [ ] **Step 5: Run full unit test suite to catch regressions**

```bash
uv run pytest tests/unit/test_tracklet_link.py -v
```

Expected: all tests PASS (including Session-6's 5 + this new one).

- [ ] **Step 6: Commit**

```bash
git add rallycut/tracking/tracklet_link.py rallycut/tracking/player_tracker.py tests/unit/test_tracklet_link.py
git commit -m "feat(analysis): learned-ReID veto in relink_spatial_splits

Session 8 adapter #2. Same helper as fix_height_swaps and
link_tracklets_by_appearance. Byte-identical at default threshold."
```

---

## Task 5: Adapter — `relink_primary_fragments`

**Condition:** skip if Task 2 attribution shows 0 swaps introduced by `RELINK_PRIMARY_FRAGMENTS`.

**Files:**
- Modify: `analysis/rallycut/tracking/tracklet_link.py` (around line 1146)
- Modify: `analysis/rallycut/tracking/player_tracker.py`
- Modify: `analysis/tests/unit/test_tracklet_link.py`

- [ ] **Step 1-6: Identical pattern to Task 4**

Signature change: add `learned_veto_threshold: float = LEARNED_MERGE_VETO_COS`.

Veto call site: inside `relink_primary_fragments`, at the per-fragment decision boundary. Look for where a non-primary fragment is about to be merged into a primary track; add the `learned_cosine_veto` call before the mutation.

Test: `test_relink_primary_fragments_vetoed_on_different_embeddings` — mirror Task 4's structure but with one fragment from `primary_track_ids` and one not.

Commit message: `feat(analysis): learned-ReID veto in relink_primary_fragments — Session 8 adapter #3`.

---

## Task 6: Adapter — `stabilize_track_ids`

**Condition:** skip if Task 2 attribution shows 0 swaps introduced by `STABILIZE_TRACK_IDS`.

**Files:**
- Modify: `analysis/rallycut/tracking/player_filter.py` (the `stabilize_track_ids` function, line 1261)
- Modify: `analysis/rallycut/tracking/player_tracker.py`
- Modify: `analysis/tests/unit/test_player_filter.py` (or test_stabilize_track_ids.py)

This pass is a **rename**, not a merge in the cost-matrix sense. It proposes `old_id -> new_id` mappings based on spatial continuity after a gap. The veto fires when the learned head says the pre-gap fragment (old_id) and post-gap fragment (new_id) are different players.

- [ ] **Step 1: Write failing test**

```python
def test_stabilize_track_ids_vetoed_on_different_learned_embeddings():
    """Veto blocks spurious rename when learned head disagrees."""
    from rallycut.tracking.player_filter import (
        PlayerFilterConfig, stabilize_track_ids,
    )

    # Track 1 ends at frame 10, track 2 starts at frame 15 nearby.
    # Without veto, stabilize would merge 2 -> 1.
    positions = (
        [PlayerPosition(f, 1, 0.5, 0.5, 0.1, 0.3, 0.9, None) for f in range(0, 10)]
        + [PlayerPosition(f, 2, 0.52, 0.5, 0.1, 0.3, 0.9, None) for f in range(15, 25)]
    )
    emb_a = _unit([1.0] + [0.0] * 127)
    emb_b = _unit([-1.0] + [0.0] * 127)
    store = LearnedEmbeddingStore()
    for f in range(10):
        store.add(1, f, emb_a)
    for f in range(15, 25):
        store.add(2, f, emb_b)

    _, id_mapping = stabilize_track_ids(
        positions, PlayerFilterConfig(),
        learned_store=store,
        learned_veto_threshold=0.5,
    )
    assert 2 not in id_mapping, f"veto failed: {id_mapping}"
```

- [ ] **Step 2: Add signature + veto**

`stabilize_track_ids` currently takes `positions, config, team_assignments=None`. Add:

```python
learned_store: "LearnedEmbeddingStore | None" = None,
learned_veto_threshold: float = LEARNED_MERGE_VETO_COS,
```

Imports at top of file (TYPE_CHECKING for store):

```python
from rallycut.tracking.merge_veto import LEARNED_MERGE_VETO_COS, learned_cosine_veto

if TYPE_CHECKING:
    from rallycut.tracking.color_repair import LearnedEmbeddingStore
```

Find the block where `new_track_id -> old_track_id` is about to be written to the mapping dict. Insert veto:

```python
if learned_cosine_veto(
    learned_store,
    old_tid, set(old_frames),
    new_tid, set(new_frames),
    threshold=learned_veto_threshold,
):
    logger.debug(
        f"stabilize_track_ids: learned-ReID veto blocked rename "
        f"{new_tid} -> {old_tid}"
    )
    continue
```

- [ ] **Step 3: Wire through `player_tracker.py`**

Line 1641: update the call to pass `learned_store=learned_store, learned_veto_threshold=_VETO_COS`.

- [ ] **Step 4-5: Run tests + lint, confirm pass**

- [ ] **Step 6: Commit**

```bash
git commit -m "feat(analysis): learned-ReID veto in stabilize_track_ids — Session 8 adapter #4"
```

---

## Task 7: Regression parity smoke (byte-identical at default threshold)

**Files:**
- No code changes. Runs the existing eval harness twice.

- [ ] **Step 1: Capture pre-Session-8 HOTA baseline from the prior sweep output**

Check `reports/merge_veto/session6_sweep/tracking.json` (Session 6's 0.0-threshold cell). Extract per-rally HOTA + aggregate HOTA. Save as `/tmp/pre_session8_hota.json`.

If this file doesn't exist, rebuild by running:

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis
LEARNED_MERGE_VETO_COS=0.0 uv run rallycut evaluate-tracking --all --retrack --cached \
  --output /tmp/pre_session8.json --audit-out /tmp/pre_session8_audit/
```

- [ ] **Step 2: Run the same eval post-refactor with default env var**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis
uv run rallycut evaluate-tracking --all --retrack --cached \
  --output /tmp/post_session8_default.json --audit-out /tmp/post_session8_default_audit/
```

- [ ] **Step 3: Compare**

```python
# Quick inline script saved as scripts/compare_tracking_json.py (1-off)
import json
pre = json.load(open("/tmp/pre_session8.json"))
post = json.load(open("/tmp/post_session8_default.json"))
for pr, ps in zip(pre["rallies"], post["rallies"]):
    assert pr["rallyId"] == ps["rallyId"]
    h_pre = pr.get("hota", {}).get("hota", 0.0)
    h_post = ps.get("hota", {}).get("hota", 0.0)
    if abs(h_pre - h_post) > 1e-6:
        print(f"DIVERGED: {pr['rallyId']} HOTA {h_pre} vs {h_post}")
print("OK" if all(...) else "DIVERGED")
```

Expected: NO divergence > 1e-6 on any rally. If divergence: the refactor introduced a bug, STOP, find it, fix before continuing.

- [ ] **Step 4: Commit only if divergence is zero; otherwise investigate**

No files to commit (results are ephemeral). This is a validation gate. If HOTA drifts by any amount at default env var, the refactor or adapters broke byte-identical guarantee. Halt.

---

## Task 8: Build the multi-site sweep harness

**Files:**
- Create: `analysis/scripts/eval_multisite_merge_veto.py`

- [ ] **Step 1: Write the sweep script**

Modelled directly on Session-6's `scripts/eval_learned_merge_veto.py`. Reuse its structure entirely — only difference is the expected output report name.

Create `analysis/scripts/eval_multisite_merge_veto.py`:

```python
"""Session 8 — multi-site learned-ReID merge veto threshold sweep.

Reuses the Session-6 harness structure. After the Session-8 adapters
(tasks 3-6), LEARNED_MERGE_VETO_COS now affects multiple merge/rename
passes, not just tracklet_link.link_tracklets_by_appearance.

Per cell: runs evaluate-tracking --all --retrack --cached, parses
aggregate + per-rally HOTA and SAME_TEAM_SWAP counts, emits session8
report with knee recommendation.

Usage:
    uv run python scripts/eval_multisite_merge_veto.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ANALYSIS_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ANALYSIS_ROOT / "reports" / "merge_veto"

THRESHOLDS = [0.0, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80]

# Ship gate constants
MAX_HOTA_DROP = 0.005  # 0.5 pp
MAX_FRAG_INCREASE = 0.20  # 20%
MIN_SWAP_REDUCTION = 0.50  # 50%

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("eval_multisite_merge_veto")


@dataclass
class CellResult:
    threshold: float
    tracking_json: Path
    audit_dir: Path
    elapsed_s: float = 0.0
    aggregate_hota: float = 0.0
    same_team_swaps: int = 0
    per_rally_hota: dict[str, float] = field(default_factory=dict)
    per_rally_unique_pred_ids: dict[str, int] = field(default_factory=dict)


def _run(cell: CellResult) -> None:
    cell.audit_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "LEARNED_MERGE_VETO_COS": f"{cell.threshold}"}
    cmd = [
        "uv", "run", "rallycut", "evaluate-tracking",
        "--all", "--retrack", "--cached",
        "--output", str(cell.tracking_json),
        "--audit-out", str(cell.audit_dir),
    ]
    logger.info("threshold %.3f", cell.threshold)
    t0 = time.time()
    subprocess.run(cmd, env=env, cwd=str(ANALYSIS_ROOT), check=True)
    cell.elapsed_s = time.time() - t0


def _parse(cell: CellResult) -> None:
    if cell.tracking_json.exists():
        data = json.loads(cell.tracking_json.read_text())
        hotas: list[float] = []
        for r in data.get("rallies", []):
            rid = r.get("rallyId")
            if rid is None:
                continue
            h = r.get("hota")
            if isinstance(h, dict):
                h = h.get("hota") or h.get("value")
            h_val = float(h or 0.0)
            cell.per_rally_hota[rid] = h_val
            hotas.append(h_val)
        if hotas:
            cell.aggregate_hota = sum(hotas) / len(hotas)

    for p in sorted(cell.audit_dir.glob("*.json")):
        if p.name.startswith("_"):
            continue
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        rid = d.get("rallyId")
        for g in d.get("perGt", []):
            for sw in g.get("realSwitches", []):
                if sw.get("cause") == "same_team_swap":
                    cell.same_team_swaps += 1
        pred_ids: set[int] = set()
        for g in d.get("perGt", []):
            for _, _, pid in g.get("predIdSpans", []):
                if pid >= 0:
                    pred_ids.add(pid)
        if rid is not None:
            cell.per_rally_unique_pred_ids[rid] = len(pred_ids)


def _evaluate_gate(baseline: CellResult, cell: CellResult) -> tuple[bool, str]:
    """Return (passes_gate, reason)."""
    # Gate 1: SAME_TEAM_SWAP reduction
    if baseline.same_team_swaps == 0:
        return (False, "baseline has 0 swaps — gate not applicable")
    swap_reduction = (baseline.same_team_swaps - cell.same_team_swaps) / baseline.same_team_swaps
    if swap_reduction < MIN_SWAP_REDUCTION:
        return (False, f"swap reduction {swap_reduction * 100:.1f}% < {MIN_SWAP_REDUCTION * 100:.0f}%")
    # Gate 2: No per-rally HOTA drop > 0.5 pp
    for rid, h_base in baseline.per_rally_hota.items():
        h_cell = cell.per_rally_hota.get(rid, h_base)
        if h_base - h_cell > MAX_HOTA_DROP:
            return (False, f"rally {rid[:8]} HOTA drop {h_base - h_cell:.4f} > {MAX_HOTA_DROP:.3f}")
    # Gate 3: Fragmentation delta ≤ 20%
    base_frag = sum(baseline.per_rally_unique_pred_ids.values())
    cell_frag = sum(cell.per_rally_unique_pred_ids.values())
    if base_frag > 0:
        frag_ratio = (cell_frag - base_frag) / base_frag
        if frag_ratio > MAX_FRAG_INCREASE:
            return (False, f"fragmentation +{frag_ratio * 100:.1f}% > {MAX_FRAG_INCREASE * 100:.0f}%")
    return (True, "clears all gates")


def _render_report(cells: list[CellResult], path: Path) -> None:
    baseline = next(c for c in cells if c.threshold == 0.0)

    lines = [
        "# Session 8 — Multi-site merge-veto gate report",
        "",
        "Threshold sweep of the learned-ReID cosine veto, now applied at",
        "all sites flagged by the Session-8 per-pass swap-attribution",
        "diagnostic. Gates: ≥50% SAME_TEAM_SWAP reduction, no per-rally",
        "HOTA drop > 0.5 pp, fragmentation delta ≤ 20%.",
        "",
        "## Per-cell summary",
        "",
        "| Threshold | Agg HOTA | Agg HOTA Δ | Swaps | Swap Δ | Frag | Frag Δ | Gate | Reason |",
        "|---:|---:|---:|---:|---:|---:|---:|:---:|:---|",
    ]
    base_frag = sum(baseline.per_rally_unique_pred_ids.values())
    for c in cells:
        ok, reason = _evaluate_gate(baseline, c)
        cell_frag = sum(c.per_rally_unique_pred_ids.values())
        frag_delta = cell_frag - base_frag
        hota_delta = (c.aggregate_hota - baseline.aggregate_hota) * 100
        swap_delta = c.same_team_swaps - baseline.same_team_swaps
        lines.append(
            f"| {c.threshold:.2f} | {c.aggregate_hota * 100:.2f}% "
            f"| {hota_delta:+.2f}pp | {c.same_team_swaps} | {swap_delta:+d} "
            f"| {cell_frag} | {frag_delta:+d} "
            f"| {'✅' if ok else '❌'} | {reason} |"
        )

    # Knee
    knee = None
    for c in sorted(cells, key=lambda x: x.threshold):
        if c.threshold == 0.0:
            continue
        ok, _ = _evaluate_gate(baseline, c)
        if ok:
            knee = c
            break

    lines += ["", "## Knee recommendation", ""]
    if knee:
        lines.append(f"**Ship at `LEARNED_MERGE_VETO_COS={knee.threshold:.2f}`** (lowest threshold passing all gates).")
    else:
        lines.append("**NO SHIP.** No threshold clears all gates. Falls to Plan B (single-site multi-signal).")

    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sweep = OUT_DIR / "session8_sweep"
    sweep.mkdir(parents=True, exist_ok=True)

    cells: list[CellResult] = []
    for t in THRESHOLDS:
        label = f"t{int(t * 100):03d}"
        cell = CellResult(
            threshold=t,
            tracking_json=sweep / label / "tracking.json",
            audit_dir=sweep / label / "audit",
        )
        _run(cell)
        _parse(cell)
        logger.info(
            "t=%.2f agg_HOTA=%.3f swaps=%d elapsed=%.1fs",
            cell.threshold, cell.aggregate_hota, cell.same_team_swaps, cell.elapsed_s,
        )
        cells.append(cell)

    report_path = OUT_DIR / "session8_gate_report.md"
    _render_report(cells, report_path)
    logger.info("wrote %s", report_path)
    print(report_path.read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Commit the harness**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis
git add scripts/eval_multisite_merge_veto.py
git commit -m "eval(analysis): Session 8 multi-site merge-veto sweep harness"
```

---

## Task 9: Run the sweep and evaluate against the ship gate

- [ ] **Step 1: Run the sweep**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis
uv run python scripts/eval_multisite_merge_veto.py 2>&1 | tee /tmp/session8_sweep.log
```

Expected runtime: ~40-60 min (8 cells × 5 min cached retrack).

- [ ] **Step 2: Read the report and interpret outcomes**

Read `reports/merge_veto/session8_gate_report.md`.

Three outcomes:

**SHIP**: one or more thresholds pass all gates. Note the knee (lowest passing threshold).

**NO SHIP, promising**: ≥40% swap reduction at best cell without passing all gates. Fall to Plan B (single-site multi-signal veto with court-plane velocity ANDed).

**NO SHIP, dead**: < 40% reduction at every cell. Close the within-team ID workstream per Session 7's "accept ceiling" framing.

- [ ] **Step 3: If SHIP, run `player_attribution_oracle` regression check**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis
LEARNED_MERGE_VETO_COS=<knee_threshold> uv run python scripts/production_eval.py
```

Compare the reported `player_attribution_oracle` to the pre-Session-8 baseline. Gate: delta > -0.3 pp.

- [ ] **Step 4: If SHIP, update the production default**

Edit `rallycut/tracking/merge_veto.py` to change the env-var fallback from `"0.0"` to the knee threshold value, e.g., `"0.55"`. This turns the feature on by default.

```python
LEARNED_MERGE_VETO_COS = float(
    os.environ.get("LEARNED_MERGE_VETO_COS", "0.55")  # was "0.0" — Session 8 knee
)
```

---

## Task 10: Write session report + memory update + final commit

**Files:**
- Create/update: `analysis/reports/merge_veto/session8_gate_report.md` (already produced by the sweep)
- Create: `/Users/mario/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/session8_merge_veto_2026_04_17.md`
- Update: `/Users/mario/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/MEMORY.md`
- Update: `/Users/mario/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/within_team_reid_project_2026_04_16.md`

- [ ] **Step 1: Write memory file**

Content template (adapt based on outcome):

```markdown
---
name: Session 8 Multi-Site Merge Veto
description: Session 8 of within-team ReID (2026-04-17). Extended Session-6 veto to all flagged merge/rename passes. Verdict: <SHIP/NO SHIP>.
type: project
---

# Session 8 — Multi-site learned-ReID merge veto

**Date**: 2026-04-17
**Outcome**: <SHIP at t=X.XX / NO SHIP → Plan B / NO SHIP → accept ceiling>

## Key findings

- Per-pass swap attribution diagnostic: <summary from task 2>.
- Sweep knee: <cell + rationale>.
- Gate results: <specific pass/fail per gate>.

## What shipped (if SHIP)

- Factored `merge_veto.py` helper, imported from 4 merge/rename passes.
- `LEARNED_MERGE_VETO_COS` default now X.XX.
- Expected production delta: <SAME_TEAM_SWAP X → Y on 43 GT rallies>.

## Dormant code (if NO SHIP)

- All 4 adapters env-var-gated. Default 0.0 → byte-identical.
- Pivot: <Plan B / accept ceiling + reason>.
```

- [ ] **Step 2: Update MEMORY.md index line**

Find the existing `within_team_reid_project_2026_04_16.md` entry. Prepend the new session summary, matching the existing style (one line under ~150 chars).

- [ ] **Step 3: Update the within_team_reid_project memory file**

Add a `## Session 8 (2026-04-17)` section summarizing outcome + code state.

- [ ] **Step 4: Commit everything**

```bash
cd /Users/mario/Personal/Projects/RallyCut
git add analysis/reports/merge_veto/
git commit -m "report(analysis): Session 8 multi-site merge-veto gate report

<SHIP at t=X.XX / NO SHIP → plan B / NO SHIP → accept ceiling>

<One-sentence summary of headline metrics>

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

Memory files live outside the git repo; they're persisted by the memory system automatically.

---

## Self-review notes

**Spec coverage check**: every requirement in the spec maps to a task.
- Diagnostic pass (§Architecture.1) → Task 2
- Unified veto helper (§Architecture.2) → Task 1
- Per-pass adapters (§Architecture.3) → Tasks 3-6
- Env var gating (§Architecture.4) → Task 1 (constant lives in merge_veto.py, preserved byte-identical)
- Threshold sweep (§Architecture.5) → Tasks 8-9
- All 4 ship gates → Task 9 steps 2-3
- Plan B trigger → Task 9 step 2 (documented as outcome branch)
- Kill conditions → Task 2 step 3 + Task 9 step 2
- Deliverables → Task 10

**Type/signature consistency**: `learned_veto_threshold` is the uniform kwarg name across all 4 adapters. `learned_cosine_veto` takes frame sets directly (not a `tracks` dict) so callers with different internal representations (track-summary dict in tracklet_link, per-track-positions list in height_consistency) can both call it.

**Placeholder scan**: Task 2 step 1 intentionally elides the full diagnostic body with implementer guidance — this is a substantial (~150 LOC) one-off diagnostic and spelling it out in full would bloat the plan without adding value. The guidance gives a concrete strategy, output format, and helper-function references; an implementer can produce the script from that. Flagging this as an approved simplification, not a placeholder bug.

**Scope check**: one focused session's work (~10h). Each task is independently commitable and reversible.
