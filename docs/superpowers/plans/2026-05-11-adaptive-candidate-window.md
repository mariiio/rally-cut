# Adaptive Candidate-Generation Window (v3.0) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover 3-4 same-team-server attribution errors on the 3 fresh-GT videos by extending `_find_nearest_players`' search window forward when the standard ±15 frame window returns fewer than `max_candidates` tracks.

**Architecture:** Surgical fallback inside `_find_nearest_player[s]` in `analysis/rallycut/tracking/contact_detector.py`. When the standard window returns underfull, run a forward-only second pass (+60 frames from contact) and merge unique track_ids in. Pass 1 entries take precedence (already closer to contact frame). Env flag `ADAPTIVE_PLAYER_SEARCH_WINDOW` (default ON) gates the new behavior; OFF restores byte-identical pre-v3 output.

**Tech Stack:** Python 3.11, pytest. No new dependencies. New CLI script applies the change to existing DB data without re-running the full tracking pipeline.

**Spec:** `docs/superpowers/specs/2026-05-11-adaptive-candidate-window-design.md`

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `analysis/rallycut/tracking/contact_detector.py` | Modify | Extract a `_collect_best_per_track` helper from inlined logic; add `_ADAPTIVE_FORWARD_FRAMES = 60`; wire env-flag-gated fallback into both `_find_nearest_player` and `_find_nearest_players`. |
| `analysis/tests/unit/test_contact_detector_adaptive_window.py` | Create | All new tests for the helper + adaptive fallback truth table + integration tests reproducing one of the panel cases. |
| `analysis/scripts/regenerate_contact_candidates.py` | Create | Reads `positions_json` + `contacts_json` from DB; re-derives `playerCandidates` using the new adaptive logic; writes updated `contacts_json`. Required because existing stored `Contact.player_candidates` were computed with the old window. |
| `analysis/scripts/measure_attribution_fresh_gt.py` | Existing | Baseline harness — reused unchanged. |
| `analysis/reports/attribution_baseline/adaptive_window_v30_2026_05_11.md` | Create | Measurement report (pre/post + gate verdicts). |
| `~/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/adaptive_candidate_window_v30_2026_05_11.md` | Create | Post-ship memory entry. |
| `~/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/MEMORY.md` | Modify | Add `[SHIPPED]` index entry. |

---

## Task 1: Extract `_collect_best_per_track` helper from `_find_nearest_players`

**Files:**
- Modify: `analysis/rallycut/tracking/contact_detector.py` (refactor — extract helper, no behavior change)
- Create: `analysis/tests/unit/test_contact_detector_adaptive_window.py`

- [ ] **Step 1: Create the test file with the helper truth-table tests**

Create `analysis/tests/unit/test_contact_detector_adaptive_window.py`:

```python
"""Unit tests for the v3.0 adaptive candidate-generation window.

Spec: docs/superpowers/specs/2026-05-11-adaptive-candidate-window-design.md
"""

from __future__ import annotations

from rallycut.tracking.contact_detector import _collect_best_per_track
from rallycut.tracking.player_tracker import PlayerPosition


def _pos(frame: int, tid: int, x: float = 0.5, y: float = 0.5) -> PlayerPosition:
    return PlayerPosition(
        frame_number=frame, track_id=tid,
        x=x, y=y, width=0.05, height=0.10, confidence=1.0,
    )


class TestCollectBestPerTrack:
    """Helper returns best (closest depth-corrected) entry per track in window."""

    def test_returns_all_tracks_within_window(self) -> None:
        """All primary tracks visible within the window are collected."""
        positions = [
            _pos(frame=100, tid=1, x=0.50, y=0.50),
            _pos(frame=100, tid=2, x=0.60, y=0.50),
            _pos(frame=100, tid=3, x=0.70, y=0.50),
            _pos(frame=100, tid=4, x=0.80, y=0.50),
        ]
        result = _collect_best_per_track(
            player_positions=positions, frame=100,
            search_frames=15, ball_x=0.5, ball_y=0.5,
            primary_track_ids=[1, 2, 3, 4],
            lower_bound_frame=85,
            upper_bound_frame=115,
            court_calibrator=None,
        )
        assert set(result.keys()) == {1, 2, 3, 4}

    def test_excludes_non_primary_tracks(self) -> None:
        """Non-primary track ids are filtered out."""
        positions = [
            _pos(frame=100, tid=1), _pos(frame=100, tid=99),
        ]
        result = _collect_best_per_track(
            player_positions=positions, frame=100,
            search_frames=15, ball_x=0.5, ball_y=0.5,
            primary_track_ids=[1, 2, 3, 4],
            lower_bound_frame=85,
            upper_bound_frame=115,
            court_calibrator=None,
        )
        assert set(result.keys()) == {1}

    def test_excludes_positions_outside_window(self) -> None:
        """Positions outside [lower_bound_frame, upper_bound_frame] excluded."""
        positions = [
            _pos(frame=84, tid=1),   # before lower bound
            _pos(frame=100, tid=2),  # in window
            _pos(frame=116, tid=3),  # after upper bound
        ]
        result = _collect_best_per_track(
            player_positions=positions, frame=100,
            search_frames=15, ball_x=0.5, ball_y=0.5,
            primary_track_ids=[1, 2, 3, 4],
            lower_bound_frame=85, upper_bound_frame=115,
            court_calibrator=None,
        )
        assert set(result.keys()) == {2}

    def test_keeps_best_per_track_when_multiple_frames(self) -> None:
        """For a single track with multiple positions, keep the one with smallest distance."""
        positions = [
            _pos(frame=98, tid=1, x=0.80, y=0.50),  # far from ball
            _pos(frame=100, tid=1, x=0.51, y=0.50),  # close to ball
            _pos(frame=102, tid=1, x=0.60, y=0.50),  # medium
        ]
        result = _collect_best_per_track(
            player_positions=positions, frame=100,
            search_frames=15, ball_x=0.5, ball_y=0.5,
            primary_track_ids=[1, 2, 3, 4],
            lower_bound_frame=85, upper_bound_frame=115,
            court_calibrator=None,
        )
        assert 1 in result
        # Best (closest to ball) entry kept; its img_dist should be the smallest.
        # The result dict's value is (rank_dist, img_dist, center_y).
        _rank_dist, img_dist, _y = result[1]
        # Closest position (frame=100, x=0.51) → img_dist ≈ |0.51 - 0.5| = 0.01 (after upper-quarter bbox shift adds a small offset).
        assert img_dist < 0.05
```

- [ ] **Step 2: Run the tests to verify they fail with import error**

Run: `cd analysis && uv run pytest tests/unit/test_contact_detector_adaptive_window.py -v`
Expected: ImportError — `_collect_best_per_track` not defined.

- [ ] **Step 3: Extract the helper from `_find_nearest_players`**

Open `analysis/rallycut/tracking/contact_detector.py`. Locate `_find_nearest_players` at line ~600. The existing function body has an inline loop that builds `best_per_track`. Extract it into a module-level helper:

```python
def _collect_best_per_track(
    player_positions: list[PlayerPosition],
    frame: int,
    search_frames: int,
    ball_x: float,
    ball_y: float,
    primary_track_ids: list[int] | None,
    lower_bound_frame: int,
    upper_bound_frame: int,
    court_calibrator: CourtCalibrator | None,
) -> dict[int, tuple[float, float, float]]:
    """Collect best (closest depth-corrected) position per track in a window.

    Used by `_find_nearest_players` (and the v3.0 adaptive-window fallback).
    Returns ``{track_id: (rank_dist, img_dist, center_y)}`` — the same shape
    as the dict that `_find_nearest_players` builds and ranks.

    `lower_bound_frame` and `upper_bound_frame` define the absolute frame
    range to consider (inclusive). The pre-v3 caller uses
    `frame ± search_frames`; the v3 adaptive fallback uses an asymmetric
    `[frame, frame + _ADAPTIVE_FORWARD_FRAMES]` for the forward-only second
    pass. `search_frames` is kept for backward compatibility with callers
    that don't pass bounds explicitly, but `lower/upper_bound_frame` take
    precedence.

    `primary_track_ids` filtering and depth-correction are unchanged from
    the inlined version.
    """
    primary_set = set(primary_track_ids) if primary_track_ids else None
    best_per_track: dict[int, tuple[float, float, float]] = {}

    for p in player_positions:
        if p.frame_number < lower_bound_frame or p.frame_number > upper_bound_frame:
            continue
        if primary_set is not None and p.track_id not in primary_set:
            continue

        img_dist = _player_to_ball_dist(p, ball_x, ball_y)

        # Depth-correction (perspective scaling), unchanged from prior inlined code.
        scale_y = p.y - p.height * 0.25
        scale = _depth_scale_at_y(scale_y, court_calibrator)
        rank_dist = img_dist * scale

        prior = best_per_track.get(p.track_id)
        if prior is None or rank_dist < prior[0]:
            best_per_track[p.track_id] = (rank_dist, img_dist, p.y)

    return best_per_track
```

Place this directly above `_find_nearest_players` (around line 597, between `_depth_scale_at_y` and `_find_nearest_players`).

Then replace the inline collection logic in `_find_nearest_players` body (currently lines ~632-654) with a single call:

```python
    # Pass 1: standard ±search_frames window (unchanged behavior).
    best_per_track = _collect_best_per_track(
        player_positions=player_positions,
        frame=frame,
        search_frames=search_frames,
        ball_x=ball_x, ball_y=ball_y,
        primary_track_ids=primary_track_ids,
        lower_bound_frame=frame - search_frames,
        upper_bound_frame=frame + search_frames,
        court_calibrator=court_calibrator,
    )

    ranked = sorted(best_per_track.items(), key=lambda x: x[1][0])
    return [
        (tid, img_dist, center_y)
        for tid, (_rank_dist, img_dist, center_y) in ranked[:max_candidates]
    ]
```

This refactor is a pure code-motion change — the helper does exactly what the inline loop did. No behavior change.

- [ ] **Step 4: Run the helper tests and the existing tests to verify no regression**

Run: `cd analysis && uv run pytest tests/unit/test_contact_detector_adaptive_window.py -v`
Expected: 4 tests PASS.

Run: `cd analysis && uv run pytest tests/unit -q`
Expected: same pass count as before this task (no new failures). Spot-check that `tests/integration/test_detect_contacts_snapshot.py` still passes — this is the snapshot guard.

- [ ] **Step 5: Run ruff + mypy on touched files**

Run: `cd analysis && uv run ruff check rallycut/tracking/contact_detector.py tests/unit/test_contact_detector_adaptive_window.py`
Run: `cd analysis && uv run mypy rallycut/tracking/contact_detector.py`
Expected: both clean.

- [ ] **Step 6: Commit**

```bash
git add analysis/rallycut/tracking/contact_detector.py analysis/tests/unit/test_contact_detector_adaptive_window.py
git commit -m "$(cat <<'EOF'
refactor(contact_detector): extract _collect_best_per_track helper

Pure code-motion refactor: extract the inline best-per-track collection
loop from _find_nearest_players into a module-level helper. Same logic,
same caller-visible behavior. Adds upper_bound_frame / lower_bound_frame
parameters that take precedence over the symmetric ±search_frames window
— used by the v3.0 adaptive-window fallback in the next task.

Spec: docs/superpowers/specs/2026-05-11-adaptive-candidate-window-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Add adaptive fallback to `_find_nearest_players`

**Files:**
- Modify: `analysis/rallycut/tracking/contact_detector.py`
- Modify: `analysis/tests/unit/test_contact_detector_adaptive_window.py`

- [ ] **Step 1: Add the fallback behavior tests**

Append to `analysis/tests/unit/test_contact_detector_adaptive_window.py`:

```python
from unittest.mock import patch

from rallycut.tracking.contact_detector import _find_nearest_players


class TestAdaptiveFallback:
    """Truth table for the adaptive forward-only fallback in _find_nearest_players."""

    def test_no_fallback_when_standard_window_full(self) -> None:
        """If standard ±15 returns all 4 primary tracks, no expansion occurs."""
        # All 4 primaries visible in the standard window.
        positions = [
            _pos(frame=100, tid=1, x=0.51),
            _pos(frame=100, tid=2, x=0.55),
            _pos(frame=100, tid=3, x=0.60),
            _pos(frame=100, tid=4, x=0.70),
            # A late-tracked candidate (tid=5) that WOULD be picked up by fallback.
            _pos(frame=150, tid=5, x=0.49),  # closest to ball, but should be excluded.
        ]
        with patch.dict("os.environ", {"ADAPTIVE_PLAYER_SEARCH_WINDOW": "1"}):
            result = _find_nearest_players(
                frame=100, ball_x=0.5, ball_y=0.5,
                player_positions=positions,
                search_frames=15, max_candidates=4,
                primary_track_ids=[1, 2, 3, 4, 5],
            )
        tids = [tid for tid, _d, _y in result]
        # tid=5 was outside the standard window AND fallback didn't fire
        # (because the standard window already had 4 candidates).
        assert tids == [1, 2, 3, 4]

    def test_fallback_fires_when_underfull(self) -> None:
        """When standard ±15 returns fewer than max_candidates, expand forward."""
        # Only 3 tracks visible in ±15; tid=4 only visible later.
        positions = [
            _pos(frame=100, tid=1, x=0.55),
            _pos(frame=100, tid=2, x=0.60),
            _pos(frame=100, tid=3, x=0.70),
            _pos(frame=150, tid=4, x=0.51),  # 50 frames forward — within +60.
        ]
        with patch.dict("os.environ", {"ADAPTIVE_PLAYER_SEARCH_WINDOW": "1"}):
            result = _find_nearest_players(
                frame=100, ball_x=0.5, ball_y=0.5,
                player_positions=positions,
                search_frames=15, max_candidates=4,
                primary_track_ids=[1, 2, 3, 4],
            )
        tids = [tid for tid, _d, _y in result]
        # tid=4 (closest by raw distance from frame 150) should now appear.
        # Pass 1 entries (tids 1, 2, 3) take precedence; tid=4 is added via fallback.
        # Final ranking is by distance — tid=4's bbox-corrected distance is smallest.
        assert 4 in tids
        assert set(tids) == {1, 2, 3, 4}

    def test_fallback_forward_only_excludes_backward_window(self) -> None:
        """Fallback expands FORWARD only — never looks earlier than the contact."""
        # Underfull standard window (only tid=1, 2 within ±15).
        # tid=3 is BEFORE contact (frame=70, gap=30 backward).
        # tid=4 is AFTER contact (frame=150, gap=50 forward) — within +60.
        positions = [
            _pos(frame=100, tid=1, x=0.55),
            _pos(frame=100, tid=2, x=0.60),
            _pos(frame=70, tid=3, x=0.51),   # backward — should be excluded
            _pos(frame=150, tid=4, x=0.52),  # forward — should be included
        ]
        with patch.dict("os.environ", {"ADAPTIVE_PLAYER_SEARCH_WINDOW": "1"}):
            result = _find_nearest_players(
                frame=100, ball_x=0.5, ball_y=0.5,
                player_positions=positions,
                search_frames=15, max_candidates=4,
                primary_track_ids=[1, 2, 3, 4],
            )
        tids = [tid for tid, _d, _y in result]
        # tid=3 (backward) excluded; tid=4 (forward) included.
        assert 3 not in tids
        assert 4 in tids

    def test_env_flag_off_disables_fallback(self) -> None:
        """ADAPTIVE_PLAYER_SEARCH_WINDOW=0 restores pre-v3 behavior."""
        positions = [
            _pos(frame=100, tid=1, x=0.55),
            _pos(frame=100, tid=2, x=0.60),
            _pos(frame=150, tid=4, x=0.51),  # only seen in forward window
        ]
        with patch.dict("os.environ", {"ADAPTIVE_PLAYER_SEARCH_WINDOW": "0"}):
            result = _find_nearest_players(
                frame=100, ball_x=0.5, ball_y=0.5,
                player_positions=positions,
                search_frames=15, max_candidates=4,
                primary_track_ids=[1, 2, 3, 4],
            )
        tids = [tid for tid, _d, _y in result]
        # Pre-v3 behavior: tid=4 NOT included (outside ±15 window).
        assert 4 not in tids

    def test_pass1_entries_take_precedence_over_pass2_for_same_track(self) -> None:
        """When the same track appears in both passes, Pass 1 (closer to contact) wins."""
        # tid=1 visible at frame=100 (Pass 1) AND frame=160 (Pass 2, +60).
        # Pass 1 entry should be retained.
        positions = [
            _pos(frame=100, tid=1, x=0.55, y=0.50),  # Pass 1 entry
            _pos(frame=100, tid=2, x=0.60),
            _pos(frame=160, tid=1, x=0.51, y=0.50),  # Pass 2 entry — closer to ball
        ]
        with patch.dict("os.environ", {"ADAPTIVE_PLAYER_SEARCH_WINDOW": "1"}):
            result = _find_nearest_players(
                frame=100, ball_x=0.5, ball_y=0.5,
                player_positions=positions,
                search_frames=15, max_candidates=4,
                primary_track_ids=[1, 2, 3, 4],
            )
        # tid=1's distance should reflect the Pass 1 frame=100 position (x=0.55),
        # not the Pass 2 frame=160 position (x=0.51, which would be closer).
        tid1_entry = next((t for t in result if t[0] == 1), None)
        assert tid1_entry is not None
        _tid, img_dist, _y = tid1_entry
        # Distance from (x=0.55, y=0.50-0.10*0.25) to (0.5, 0.5) is ~0.057, not 0.012.
        assert img_dist > 0.04, "Pass 1 entry should be retained, not overwritten by Pass 2"
```

- [ ] **Step 2: Run the tests to verify they fail (helper not yet aware of fallback)**

Run: `cd analysis && uv run pytest tests/unit/test_contact_detector_adaptive_window.py::TestAdaptiveFallback -v`
Expected: tests fail — the fallback behavior isn't implemented yet.

- [ ] **Step 3: Add `_ADAPTIVE_FORWARD_FRAMES` constant + adaptive fallback**

In `analysis/rallycut/tracking/contact_detector.py`, near the other module-level constants (around lines 30-60, depending on file layout — search for `_CROSS_SIDE_MIN_DISTANCE` or `_CONFIDENCE_THRESHOLD`), add:

```python
# v3.0 (2026-05-11): adaptive forward-only fallback window for
# _find_nearest_player(s). When the standard ±search_frames window returns
# fewer than max_candidates tracks, expand FORWARD by this many frames to
# catch late-tracked players (e.g., a server who's only detected AFTER
# entering the play frame). Validated against 4 panel cases on 2026-05-11.
# Gated by env var ADAPTIVE_PLAYER_SEARCH_WINDOW (default ON, set "0" to disable).
_ADAPTIVE_FORWARD_FRAMES = 60
```

Then update `_find_nearest_players`' body to add the fallback. Replace the post-refactor body (the version landed in Task 1) with:

```python
    # Pass 1: standard ±search_frames window (existing behavior).
    best_per_track = _collect_best_per_track(
        player_positions=player_positions,
        frame=frame,
        search_frames=search_frames,
        ball_x=ball_x, ball_y=ball_y,
        primary_track_ids=primary_track_ids,
        lower_bound_frame=frame - search_frames,
        upper_bound_frame=frame + search_frames,
        court_calibrator=court_calibrator,
    )

    # v3.0 adaptive fallback: if Pass 1 is underfull (fewer than max_candidates
    # primary tracks visible), expand forward-only to catch late-tracked
    # players. Validated cause of 7/8 absent-GT serves on the 3 GT panel.
    # Spec: docs/superpowers/specs/2026-05-11-adaptive-candidate-window-design.md
    adaptive_enabled = (
        os.environ.get("ADAPTIVE_PLAYER_SEARCH_WINDOW", "1") != "0"
    )
    if adaptive_enabled and len(best_per_track) < max_candidates:
        pass2_best = _collect_best_per_track(
            player_positions=player_positions,
            frame=frame,
            search_frames=_ADAPTIVE_FORWARD_FRAMES,
            ball_x=ball_x, ball_y=ball_y,
            primary_track_ids=primary_track_ids,
            lower_bound_frame=frame,  # forward-only — never look earlier
            upper_bound_frame=frame + _ADAPTIVE_FORWARD_FRAMES,
            court_calibrator=court_calibrator,
        )
        # Merge: Pass 1 entries take precedence (already closer to the contact
        # frame). Pass 2 only contributes track_ids missing from Pass 1.
        for tid, entry in pass2_best.items():
            if tid not in best_per_track:
                best_per_track[tid] = entry

    ranked = sorted(best_per_track.items(), key=lambda x: x[1][0])
    return [
        (tid, img_dist, center_y)
        for tid, (_rank_dist, img_dist, center_y) in ranked[:max_candidates]
    ]
```

Verify that `import os` is already at the top of the file. (It was added during the v1 team-chain workstream — should be present. If not, add it alphabetically in the import block.)

- [ ] **Step 4: Run the tests and verify they pass**

Run: `cd analysis && uv run pytest tests/unit/test_contact_detector_adaptive_window.py -v`
Expected: 4 (TestCollectBestPerTrack) + 5 (TestAdaptiveFallback) = 9 tests PASS.

- [ ] **Step 5: Run the existing suites to confirm no regression**

Run: `cd analysis && uv run pytest tests/unit -q`
Expected: same pass count as before plus the 5 new TestAdaptiveFallback tests. No new failures.

Spot-check: `cd analysis && uv run pytest tests/integration/test_detect_contacts_snapshot.py -v`
Expected: PASS. This is the snapshot guard for the contact detection pipeline; the adaptive fallback should not fire on the snapshot fixture (the snapshot was generated when all primary tracks were visible in the standard window).

If the snapshot test fails: investigate. The fix should preserve byte-identical behavior when the env flag is OFF AND when the standard window is full.

- [ ] **Step 6: Run mypy + ruff**

Run: `cd analysis && uv run mypy rallycut/tracking/contact_detector.py`
Run: `cd analysis && uv run ruff check rallycut/tracking/contact_detector.py tests/unit/test_contact_detector_adaptive_window.py`
Expected: both clean.

- [ ] **Step 7: Commit**

```bash
git add analysis/rallycut/tracking/contact_detector.py analysis/tests/unit/test_contact_detector_adaptive_window.py
git commit -m "$(cat <<'EOF'
feat(contact_detector): v3.0 adaptive forward-only fallback in _find_nearest_players

When the standard ±search_frames window returns fewer than max_candidates
primary tracks, run a forward-only second pass (+60 frames from contact)
to capture late-tracked players. Pass 1 entries take precedence (already
closer to contact frame); Pass 2 only fills in missing track_ids.

Validated against 4 of 4 panel cases with contacts (window=60 catches
3, gigi/bc9345c1 still needs window=120 — deferred to v3.0.1). 2 of 4
become rank-1 candidates with the wider window.

Gated by env var ADAPTIVE_PLAYER_SEARCH_WINDOW (default ON; "0" disables
for instant rollback to pre-v3 behavior).

Spec: docs/superpowers/specs/2026-05-11-adaptive-candidate-window-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Add adaptive fallback to `_find_nearest_player` (singular)

**Files:**
- Modify: `analysis/rallycut/tracking/contact_detector.py`
- Modify: `analysis/tests/unit/test_contact_detector_adaptive_window.py`

- [ ] **Step 1: Add the singular-form tests**

Append to `analysis/tests/unit/test_contact_detector_adaptive_window.py`:

```python
from rallycut.tracking.contact_detector import _find_nearest_player


class TestFindNearestPlayerAdaptive:
    """The singular _find_nearest_player gets the same fallback behavior."""

    def test_singular_fallback_finds_late_tracked_player_when_underfull(self) -> None:
        """When ±15 has no candidates (or fewer than max_candidates if filtered),
        the singular form's fallback catches a late-tracked one."""
        positions = [
            # Only one primary visible in ±15 — underfull
            _pos(frame=100, tid=1, x=0.70),
            # tid=2 only visible later, but closer to ball
            _pos(frame=150, tid=2, x=0.51),
        ]
        with patch.dict("os.environ", {"ADAPTIVE_PLAYER_SEARCH_WINDOW": "1"}):
            tid, dist, _center_y = _find_nearest_player(
                frame=100, ball_x=0.5, ball_y=0.5,
                player_positions=positions,
                search_frames=15,
                primary_track_ids=[1, 2, 3, 4],
            )
        # Without fallback: tid=1 wins (it's the only candidate).
        # With fallback: tid=2 wins because its forward-window position is closer.
        assert tid == 2

    def test_singular_env_off_returns_only_pass1(self) -> None:
        """With env flag OFF, singular form returns only Pass 1 result."""
        positions = [
            _pos(frame=100, tid=1, x=0.70),
            _pos(frame=150, tid=2, x=0.51),  # forward-only, would be ignored
        ]
        with patch.dict("os.environ", {"ADAPTIVE_PLAYER_SEARCH_WINDOW": "0"}):
            tid, _dist, _center_y = _find_nearest_player(
                frame=100, ball_x=0.5, ball_y=0.5,
                player_positions=positions,
                search_frames=15,
                primary_track_ids=[1, 2, 3, 4],
            )
        # Pre-v3 behavior: only tid=1 (the sole candidate in ±15).
        assert tid == 1

    def test_singular_no_primary_track_filter_uses_max_candidates(self) -> None:
        """When primary_track_ids is None, the fallback uses max_candidates as
        the underfull threshold (consistent with _find_nearest_players)."""
        # Singular form doesn't take max_candidates explicitly — it just picks
        # the single nearest. The adaptive fallback should still fire if Pass 1
        # is empty.
        positions = [
            _pos(frame=150, tid=1, x=0.51),  # forward-only
        ]
        with patch.dict("os.environ", {"ADAPTIVE_PLAYER_SEARCH_WINDOW": "1"}):
            tid, _dist, _y = _find_nearest_player(
                frame=100, ball_x=0.5, ball_y=0.5,
                player_positions=positions,
                search_frames=15,
                primary_track_ids=None,
            )
        # Pass 1 empty → fallback fires → tid=1 found.
        assert tid == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd analysis && uv run pytest tests/unit/test_contact_detector_adaptive_window.py::TestFindNearestPlayerAdaptive -v`
Expected: tests fail — singular function doesn't have the fallback yet.

- [ ] **Step 3: Add the fallback to `_find_nearest_player`**

In `analysis/rallycut/tracking/contact_detector.py`, locate `_find_nearest_player` (singular) at line ~525. Its current body iterates positions and tracks the single nearest. Replace with a delegation that reuses `_collect_best_per_track`:

```python
def _find_nearest_player(
    frame: int,
    ball_x: float,
    ball_y: float,
    player_positions: list[PlayerPosition],
    search_frames: int = 5,
    primary_track_ids: list[int] | None = None,
) -> tuple[int, float, float]:
    """Find nearest player to ball at given frame.

    Uses wrist keypoint distance when pose data is available, falling
    back to bbox upper-quarter distance. See _player_to_ball_dist().

    When ``primary_track_ids`` is provided, the search is restricted to
    those tracks — matches the filter applied in `_find_nearest_players`
    (plural) so the initial pick aligns with the candidate list.

    v3.0: when the standard ±search_frames window returns no candidates
    (or, with primary_track_ids, fewer than the primary count), expand
    forward-only by _ADAPTIVE_FORWARD_FRAMES to catch late-tracked
    players. Gated by env var ADAPTIVE_PLAYER_SEARCH_WINDOW (default ON).

    Returns:
        (track_id, distance, player_center_y). track_id=-1 if no player found.
        player_center_y is the bbox center Y (for court side determination).
    """
    # Pass 1: standard ±search_frames window.
    pass1 = _collect_best_per_track(
        player_positions=player_positions,
        frame=frame,
        search_frames=search_frames,
        ball_x=ball_x, ball_y=ball_y,
        primary_track_ids=primary_track_ids,
        lower_bound_frame=frame - search_frames,
        upper_bound_frame=frame + search_frames,
        court_calibrator=None,
    )

    # v3.0 adaptive fallback. Underfull threshold: when primary_track_ids
    # is provided, it's len(primary_track_ids); otherwise, fire only when
    # Pass 1 is completely empty.
    target = (
        len(primary_track_ids) if primary_track_ids else (1 if not pass1 else 0)
    )
    adaptive_enabled = (
        os.environ.get("ADAPTIVE_PLAYER_SEARCH_WINDOW", "1") != "0"
    )
    if adaptive_enabled and len(pass1) < target:
        pass2 = _collect_best_per_track(
            player_positions=player_positions,
            frame=frame,
            search_frames=_ADAPTIVE_FORWARD_FRAMES,
            ball_x=ball_x, ball_y=ball_y,
            primary_track_ids=primary_track_ids,
            lower_bound_frame=frame,  # forward-only
            upper_bound_frame=frame + _ADAPTIVE_FORWARD_FRAMES,
            court_calibrator=None,
        )
        for tid, entry in pass2.items():
            if tid not in pass1:
                pass1[tid] = entry

    if not pass1:
        return -1, float("inf"), 0.5

    # Find the single best (smallest img_dist — index 1 in the value tuple).
    best_tid = -1
    best_img_dist = float("inf")
    best_center_y = 0.5
    for tid, (_rank_dist, img_dist, center_y) in pass1.items():
        if img_dist < best_img_dist:
            best_tid = tid
            best_img_dist = img_dist
            best_center_y = center_y

    return best_tid, best_img_dist, best_center_y
```

- [ ] **Step 4: Run the tests and verify they pass**

Run: `cd analysis && uv run pytest tests/unit/test_contact_detector_adaptive_window.py::TestFindNearestPlayerAdaptive -v`
Expected: 3 tests PASS.

- [ ] **Step 5: Run full file and existing suites**

Run: `cd analysis && uv run pytest tests/unit/test_contact_detector_adaptive_window.py -v`
Expected: 12 tests PASS (4 + 5 + 3).

Run: `cd analysis && uv run pytest tests/unit -q`
Expected: no new failures.

- [ ] **Step 6: Run mypy + ruff**

Run: `cd analysis && uv run mypy rallycut/tracking/contact_detector.py`
Run: `cd analysis && uv run ruff check rallycut/tracking/contact_detector.py tests/unit/test_contact_detector_adaptive_window.py`
Expected: both clean.

- [ ] **Step 7: Commit**

```bash
git add analysis/rallycut/tracking/contact_detector.py analysis/tests/unit/test_contact_detector_adaptive_window.py
git commit -m "$(cat <<'EOF'
feat(contact_detector): v3.0 adaptive fallback in _find_nearest_player (singular)

Same forward-only fallback semantics as _find_nearest_players (plural):
when the standard ±search_frames window is underfull, expand forward by
+_ADAPTIVE_FORWARD_FRAMES to catch late-tracked players. Underfull
threshold is len(primary_track_ids) when provided, otherwise "any
non-empty" (i.e., fire only when standard window finds nothing).

Refactor: delegate the collection logic to _collect_best_per_track
(extracted in Task 1), then pick the single nearest from the merged
result. Behavior with env flag OFF is byte-identical to pre-v3.

Spec: docs/superpowers/specs/2026-05-11-adaptive-candidate-window-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Build `scripts/regenerate_contact_candidates.py`

**Files:**
- Create: `analysis/scripts/regenerate_contact_candidates.py`

This script applies the new adaptive candidate-generation to EXISTING DB rallies WITHOUT re-running the full tracking pipeline. It reads positions_json + contacts_json, re-derives each Contact's `playerCandidates` using the v3 code (with env flag ON), and writes updated contacts_json back.

- [ ] **Step 1: Create the regeneration script**

Create `analysis/scripts/regenerate_contact_candidates.py`:

```python
"""Regenerate Contact.player_candidates for stored rallies using the v3.0
adaptive candidate-generation window.

Reads `positions_json` + `contacts_json` from `player_tracks`; for each
contact, re-computes `playerCandidates` via the new
`_find_nearest_players` (with ADAPTIVE_PLAYER_SEARCH_WINDOW=1) and writes
the updated `contacts_json` back. Other Contact fields (ball position,
frame, court_side, etc.) are preserved.

This is the deploy companion for the v3.0 spec — without it, existing
rallies still carry the pre-v3 narrow-window candidates and the adaptive
fallback has no effect on attribution.

Run from analysis/:
    uv run python scripts/regenerate_contact_candidates.py <video-id> [--dry-run]
    uv run python scripts/regenerate_contact_candidates.py --all-with-gt [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from rallycut.evaluation.db import get_connection
from rallycut.tracking.contact_detector import _find_nearest_players
from rallycut.tracking.player_tracker import PlayerPosition


def _reconstruct_positions(positions_json: list[dict[str, Any]] | None) -> list[PlayerPosition]:
    if not positions_json:
        return []
    out: list[PlayerPosition] = []
    for p in positions_json:
        if not isinstance(p, dict):
            continue
        out.append(PlayerPosition(
            frame_number=p.get("frameNumber", 0),
            track_id=p.get("trackId", -1),
            x=p.get("x", 0.5),
            y=p.get("y", 0.5),
            width=p.get("width", 0.05),
            height=p.get("height", 0.10),
            confidence=p.get("confidence", 1.0),
            keypoints=p.get("keypoints"),
        ))
    return out


def _regenerate_one_rally(
    contacts_json: dict[str, Any],
    positions_json: list[dict[str, Any]],
    primary_track_ids: list[int] | None,
    max_candidates: int = 4,
) -> tuple[dict[str, Any], int]:
    """Return (new_contacts_json, n_contacts_changed).

    A contact is "changed" if its playerCandidates list differs (in set or
    order) from the pre-regeneration version.
    """
    # Ensure env flag is ON for the regeneration regardless of caller's env.
    os.environ["ADAPTIVE_PLAYER_SEARCH_WINDOW"] = "1"

    positions = _reconstruct_positions(positions_json)
    new_contacts_json = dict(contacts_json or {})
    new_list = []
    n_changed = 0

    for c in (contacts_json or {}).get("contacts", []):
        frame = c.get("frame", 0)
        ball_x = c.get("ballX", 0.5)
        ball_y = c.get("ballY", 0.5)

        new_candidates = _find_nearest_players(
            frame=frame, ball_x=ball_x, ball_y=ball_y,
            player_positions=positions,
            search_frames=15,
            max_candidates=max_candidates,
            court_calibrator=None,  # caller already applied calibration upstream
            primary_track_ids=primary_track_ids,
        )

        # Build the playerCandidates list in the same shape as the original.
        new_pc = [
            [tid, dist if dist != float("inf") else None]
            for tid, dist, _y in new_candidates
        ]

        old_pc = c.get("playerCandidates") or []
        if [list(x) for x in new_pc] != [list(x) for x in old_pc]:
            n_changed += 1

        c_copy = dict(c)
        c_copy["playerCandidates"] = new_pc
        new_list.append(c_copy)

    new_contacts_json["contacts"] = new_list
    return new_contacts_json, n_changed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video_id", nargs="?", default=None)
    parser.add_argument(
        "--all-with-gt",
        action="store_true",
        help="Regenerate for all videos with action_ground_truth_json (the 3 GT videos).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.video_id and not args.all_with_gt:
        sys.exit("Provide either <video-id> or --all-with-gt")

    with get_connection() as conn, conn.cursor() as cur:
        if args.all_with_gt:
            cur.execute(
                """SELECT DISTINCT v.id, v.filename FROM videos v
                   JOIN rallies r ON r.video_id = v.id
                   JOIN player_tracks pt ON pt.rally_id = r.id
                   WHERE pt.action_ground_truth_json IS NOT NULL
                   ORDER BY v.filename"""
            )
            video_rows = cur.fetchall()
        else:
            cur.execute("SELECT id, filename FROM videos WHERE id = %s", (args.video_id,))
            video_rows = cur.fetchall()

        total_rallies = 0
        total_changed = 0
        for vid, filename in video_rows:
            cur.execute(
                """SELECT pt.id, pt.contacts_json, pt.positions_json, pt.primary_track_ids
                   FROM player_tracks pt JOIN rallies r ON r.id = pt.rally_id
                   WHERE r.video_id = %s
                     AND pt.contacts_json IS NOT NULL
                     AND pt.positions_json IS NOT NULL
                   ORDER BY r.start_ms""",
                (vid,),
            )
            rallies = cur.fetchall()
            print(f"[{filename}] {len(rallies)} rallies", flush=True)

            for pt_id, contacts_json, positions_json, primary_track_ids in rallies:
                total_rallies += 1
                new_contacts_json, n_changed = _regenerate_one_rally(
                    contacts_json or {},
                    positions_json or [],
                    list(primary_track_ids) if primary_track_ids else None,
                )
                if n_changed > 0:
                    total_changed += 1
                    print(f"  pt_id={pt_id}: {n_changed} contacts changed", flush=True)
                    if not args.dry_run:
                        cur.execute(
                            "UPDATE player_tracks SET contacts_json = %s WHERE id = %s",
                            [json.dumps(new_contacts_json), pt_id],
                        )
        if not args.dry_run:
            conn.commit()
    print(f"\nTotal: {total_changed}/{total_rallies} rallies had at least one "
          f"contact's playerCandidates updated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Test the script in dry-run mode on one rally**

Run from `analysis/`:

```bash
uv run python scripts/regenerate_contact_candidates.py 950fbe5d-fdad-4862-b05d-8b374bdd5ec6 --dry-run
```

Expected: the script runs without errors and reports how many rallies would change. Some rallies should show changes; check the output prints reasonable counts.

If the dry-run errors out: debug. STOP and report BLOCKED with the specific error.

- [ ] **Step 3: Commit the script**

```bash
git add analysis/scripts/regenerate_contact_candidates.py
git commit -m "$(cat <<'EOF'
feat(candidate-window): regenerate_contact_candidates.py for v3.0 deploy

Applies the v3.0 adaptive-window candidate-generation to existing DB
rallies WITHOUT re-running the full tracking pipeline. Reads
positions_json + contacts_json, re-derives Contact.playerCandidates via
the new _find_nearest_players (with ADAPTIVE_PLAYER_SEARCH_WINDOW=1),
writes updated contacts_json back.

Supports a single video-id or --all-with-gt for the 3 GT-labeled videos.
--dry-run shows changes without writing.

Used by Tasks 5 (3 GT videos deploy) and 7 (fleet deploy).

Spec: docs/superpowers/specs/2026-05-11-adaptive-candidate-window-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: A/B measurement on 3 GT videos

**Files:**
- Create: `analysis/reports/attribution_baseline/adaptive_window_v30_2026_05_11.md`

- [ ] **Step 1: Create the report file with the baseline reference**

```bash
mkdir -p analysis/reports/attribution_baseline
cat > analysis/reports/attribution_baseline/adaptive_window_v30_2026_05_11.md <<'EOF'
# Adaptive Candidate Window v3.0 — A/B Measurement Report (2026-05-11)

Spec: docs/superpowers/specs/2026-05-11-adaptive-candidate-window-design.md
Plan: docs/superpowers/plans/2026-05-11-adaptive-candidate-window.md

## Pre-v3 baseline (DB read — post-v1, pre-v3)

n=136 GT actions across 22 rallies (cece + gigi + wawa):
- correct: 82 (60.3%)
- wrong: 26 (19.1%) [cross=17, same=9, unk=0]
- missing: 28 (20.6%)
- Per-action matched accuracy: serve 47%, set 67%, dig 82%, attack 83%, receive 94%
- Absent-GT cases: 8 (all serves; root cause: server tracked AFTER serve frame)

## Pre-deploy ranks for the 4 absent-server cases with contacts

(From the 2026-05-11 rank-of-GT validation:)

| Case | Pre-v3 GT rank | Window=60 GT rank |
|---|---|---|
| gigi/72c8229b f=94  | absent | rank 2 |
| gigi/bc9345c1 f=111 | absent | absent (gap=85, needs window=120) |
| gigi/5b6f0474 f=48  | absent | rank 1 |
| wawa/06c13117 f=184 | absent | rank 1 |

(3 of 4 unblock with window=60; 2 of those become rank 1 = best candidate.)

## Post-regeneration measurements (filled in after Task 5)

EOF
```

- [ ] **Step 2: Run the regeneration on the 3 GT videos**

Run from `analysis/`:

```bash
uv run python scripts/regenerate_contact_candidates.py --all-with-gt
```

Expected: the script writes updated `contacts_json` for rallies whose `playerCandidates` changed under the new adaptive logic. Capture the output (which rallies changed, how many contacts each).

Append the output to the report file under "## Post-regeneration measurements" section.

- [ ] **Step 3: Re-run the baseline measurement**

```bash
uv run python -u scripts/measure_attribution_fresh_gt.py
```

Expected: numbers differ from the pre-v3 baseline because `Contact.player_candidates` is now wider for the affected rallies. The downstream attribution path (`reattribute_players` Pass 2, etc.) operates on the new candidates.

Append the output to the report file.

- [ ] **Step 4: Re-run the rank-of-GT diagnostic**

Re-run the validation script from earlier (`/tmp/validate_uncertainty_rank.py`). Expected: the 4 absent-server cases with contacts (gigi/72c8229b, gigi/bc9345c1, gigi/5b6f0474, wawa/06c13117) should show updated ranks:

- gigi/72c8229b f=94: was absent → should be rank 2
- gigi/bc9345c1 f=111: was absent → still absent (gap=85, needs window=120 — expected)
- gigi/5b6f0474 f=48: was absent → should be rank 1
- wawa/06c13117 f=184: was absent → should be rank 1

Append these results to the report file under a "## Rank-of-GT diagnostic (post-v3)" section.

- [ ] **Step 5: Verify the 5 pre-ship gates**

Append a checklist to the report file:

```markdown
## Pre-ship gates (post-v3, DB read after regeneration)

- [ ] G-A: Combined `correct_rate` improves by ≥ +2pp (60.3% → ≥ 62.3%).
      Result: pre=60.3%, post=__%, delta=__pp.
- [ ] G-B: No per-fixture `correct` regression (cece ≥ 22, gigi ≥ 35, wawa ≥ 25).
      Result: cece __, gigi __, wawa __.
- [ ] G-C: `wrong_unknown_team` non-increasing (0 today).
      Result: pre=0, post=__.
- [ ] G-D: No new test failures in unit suites.
      Result: __ tests passed (was 180 baseline before v3).
- [ ] G-E: Of the 4 absent-server cases with contacts, ≥ 3 now have GT in candidates.
      Result: __ of 4.
```

Fill in result blanks. STOP if G-A, G-B, or G-E fails. Report DONE_WITH_CONCERNS.

- [ ] **Step 6: Commit the report**

```bash
git add analysis/reports/attribution_baseline/adaptive_window_v30_2026_05_11.md
git commit -m "$(cat <<'EOF'
report(candidate-window): v3.0 A/B measurement on 3 GT videos

Regeneration script writes updated Contact.playerCandidates for the 3
GT-labeled videos. Baseline harness re-run captures post-v3 accuracy.
Rank-of-GT diagnostic confirms absent-server cases now resolve to
ranked candidates per the spec's prediction.

Spec: docs/superpowers/specs/2026-05-11-adaptive-candidate-window-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Deploy to 3 GT videos via reattribute-actions

**Files:** None modified; only DB updates.

The regeneration in Task 5 already wrote the new `playerCandidates`. This task runs `reattribute-actions` to apply attribution downstream of those new candidates and verify the end-to-end attribution accuracy on DB.

- [ ] **Step 1: DB snapshot of the 3 GT videos' actions_json**

```bash
mkdir -p analysis/reports/attribution_baseline/db_snapshots
PGPASSWORD=postgres psql -h localhost -p 5436 -U postgres -d rallycut \
  -t -A -c "
SELECT json_build_object(
  'video_id', r.video_id,
  'rally_id', r.id,
  'pt_id', pt.id,
  'actions_json', pt.actions_json
)
FROM rallies r
JOIN player_tracks pt ON pt.rally_id = r.id
WHERE r.video_id IN (
  '950fbe5d-fdad-4862-b05d-8b374bdd5ec6',
  'b097dd2a-6953-4e0e-a603-5be3552f462e',
  '5c756c41-1cc1-4486-a95c-97398912cfbe'
)
ORDER BY r.video_id, r.start_ms" \
  > analysis/reports/attribution_baseline/db_snapshots/pre_adaptive_window_2026_05_11.jsonl
wc -l analysis/reports/attribution_baseline/db_snapshots/pre_adaptive_window_2026_05_11.jsonl
```

Expected: 22 lines.

- [ ] **Step 2: Run reattribute-actions on each of the 3 GT videos**

```bash
cd analysis
uv run rallycut reattribute-actions 950fbe5d-fdad-4862-b05d-8b374bdd5ec6
uv run rallycut reattribute-actions b097dd2a-6953-4e0e-a603-5be3552f462e
uv run rallycut reattribute-actions 5c756c41-1cc1-4486-a95c-97398912cfbe
```

Capture the per-video re-attribution counts. STOP if any errors.

- [ ] **Step 3: Re-run baseline harness against DB**

```bash
cd analysis && uv run python -u scripts/measure_attribution_fresh_gt.py
```

Append the output to `analysis/reports/attribution_baseline/adaptive_window_v30_2026_05_11.md` under "## Post-deploy (DB read, env flag ON)".

- [ ] **Step 4: Run coherence-invariants on the 3 GT videos**

```bash
cd analysis
uv run rallycut audit-coherence-invariants 950fbe5d-fdad-4862-b05d-8b374bdd5ec6
uv run rallycut audit-coherence-invariants b097dd2a-6953-4e0e-a603-5be3552f462e
uv run rallycut audit-coherence-invariants 5c756c41-1cc1-4486-a95c-97398912cfbe
```

Append C-1 / C-2 / C-3 counts to the report. Expected: C-2 should be NON-REGRESSING (v3.0 doesn't change attribution per se, only candidate pool). If C-2 increases, investigate.

- [ ] **Step 5: Commit the post-deploy verification**

```bash
git add analysis/reports/attribution_baseline/adaptive_window_v30_2026_05_11.md analysis/reports/attribution_baseline/db_snapshots/
git commit -m "$(cat <<'EOF'
report(candidate-window): v3.0 post-deploy verification on 3 GT videos

reattribute-actions re-run on cece/gigi/wawa with the regenerated
playerCandidates from Task 5. DB snapshot pre-deploy retained for
rollback. Final post-deploy baseline numbers + coherence C-2 counts
captured.

Spec: docs/superpowers/specs/2026-05-11-adaptive-candidate-window-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Fleet deploy

**Files:** None modified; DB updates fleet-wide.

- [ ] **Step 1: List all fleet videos**

```bash
PGPASSWORD=postgres psql -h localhost -p 5436 -U postgres -d rallycut -t -A -c "
SELECT v.id FROM videos v
WHERE EXISTS (SELECT 1 FROM rallies r JOIN player_tracks pt ON pt.rally_id = r.id WHERE r.video_id = v.id)
  AND v.match_analysis_json IS NOT NULL
ORDER BY v.filename" > /tmp/fleet_video_ids.txt
wc -l /tmp/fleet_video_ids.txt
```

Expected: ~70 video IDs.

- [ ] **Step 2: Fleet-wide regeneration of contacts_json**

```bash
cd analysis
: > /tmp/regenerate_fleet.log
while read -r vid; do
  echo "=== $vid ===" >> /tmp/regenerate_fleet.log
  uv run python scripts/regenerate_contact_candidates.py "$vid" >> /tmp/regenerate_fleet.log 2>&1
done < /tmp/fleet_video_ids.txt
```

Run with `run_in_background: true`. Each video takes ~5-30s depending on rally count. Expected total: ~10-30 minutes for 70 videos.

- [ ] **Step 3: Fleet-wide reattribute-actions**

```bash
cd analysis
: > /tmp/reattribute_fleet_v30.log
while read -r vid; do
  echo "=== $vid ===" >> /tmp/reattribute_fleet_v30.log
  uv run rallycut reattribute-actions "$vid" >> /tmp/reattribute_fleet_v30.log 2>&1
done < /tmp/fleet_video_ids.txt
```

Run with `run_in_background: true`. ~5-10 minutes for 70 videos.

- [ ] **Step 4: Post-deploy fleet coherence audit**

```bash
cd analysis
: > /tmp/coherence_post_v30.log
while read -r vid; do
  echo "=== $vid ===" >> /tmp/coherence_post_v30.log
  uv run rallycut audit-coherence-invariants "$vid" >> /tmp/coherence_post_v30.log 2>&1
done < /tmp/fleet_video_ids.txt
```

- [ ] **Step 5: Compute fleet C-1/C-2/C-3 deltas**

Compare fleet C-1/C-2/C-3 violation counts pre and post. The most recent pre-deploy fleet baseline is the team-chain v1 deploy (HEAD `b0ddf1d`): C-1=106, C-2=338, C-3=1. The expectation: v3.0 should not regress these (it adds candidates to the pool without changing how attribution is decided).

Append the deltas to `analysis/reports/attribution_baseline/adaptive_window_v30_2026_05_11.md` under "## Fleet deploy results".

- [ ] **Step 6: Commit the fleet report**

```bash
git add analysis/reports/attribution_baseline/adaptive_window_v30_2026_05_11.md
git commit -m "$(cat <<'EOF'
report(candidate-window): v3.0 fleet deploy on ~70 videos

regenerate_contact_candidates run fleet-wide; reattribute-actions run
fleet-wide; coherence-invariants audited pre/post. Per-video
re-attribution counts captured.

Spec: docs/superpowers/specs/2026-05-11-adaptive-candidate-window-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Memory entry + MEMORY.md index

**Files:**
- Create: `~/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/adaptive_candidate_window_v30_2026_05_11.md`
- Modify: `~/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/MEMORY.md`

- [ ] **Step 1: Write the memory entry**

Create `~/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/adaptive_candidate_window_v30_2026_05_11.md`:

```markdown
---
name: Adaptive candidate window v3.0 2026-05-11
description: Forward-only fallback in _find_nearest_player(s) when standard window is underfull. Catches late-tracked servers (the validated cause of 7/8 absent-GT serves on the 3 GT panel). Env flag ADAPTIVE_PLAYER_SEARCH_WINDOW (default ON).
type: project
---
# Adaptive Candidate Window v3.0 — 2026-05-11

**Shipped:** 2026-05-11

## What it does

`_find_nearest_player(s)` in `analysis/rallycut/tracking/contact_detector.py` now runs a forward-only fallback when the standard ±15 frame window returns fewer than `max_candidates` primary tracks. The fallback expands forward by `_ADAPTIVE_FORWARD_FRAMES = 60` from the contact frame, catches late-tracked players (Pass 2), and merges them into the Pass 1 result. Pass 1 entries take precedence (they're closer to the contact frame); Pass 2 only fills missing track_ids.

A new `scripts/regenerate_contact_candidates.py` applies the v3 logic to existing DB rallies WITHOUT re-running tracking — it reads positions_json + contacts_json and writes updated `Contact.playerCandidates` back.

## Why it works

The 2026-05-11 rank-of-GT validation diagnostic found that 8 of 26 wrong attributions had GT ABSENT from `Contact.player_candidates`. All 8 were serves. Detailed investigation: 7 of 8 were caused by the GT server being tracked only AFTER the serve contact frame — the player detector misses them at serve time (off-screen behind baseline, tossing the ball) but picks them up after they step forward.

Offline validation: widening the window to 60 frames forward recovers 3 of 4 panel cases with contacts. 2 of those become rank-1 candidates (best by proximity).

The fallback is FORWARD-ONLY by design (asymmetric) — the dominant failure mode is "server tracked AFTER serve", never BEFORE. Forward-only also prevents pulling in players who briefly appeared before the rally started.

## Why this works where v2 didn't

v2 (joint hard-rule beam search, NO-GO) tried to be globally smart but cascaded upstream noise. v3 is the opposite: a surgical fix to candidate generation, validated by data, with a clear ceiling.

This addresses the UPSTREAM bottleneck (GT absent from candidates) rather than trying to recover from it downstream.

## Env flag

`ADAPTIVE_PLAYER_SEARCH_WINDOW=0` restores byte-identical pre-v3 behavior. Default `"1"` (ON). Read at call time inside both `_find_nearest_player` and `_find_nearest_players`.

## Measured impact (fill in post Tasks 5-7)

| Metric | Pre-v3 | Post-v3 | Delta |
|---|---:|---:|---:|
| Combined correct_rate (3 GT) | 60.3% | __% | __pp |
| wrong cases with absent GT | 8 of 26 | __ of __ | __ |
| Fleet C-2 violations | 338 | __ | __ |

## Files touched

- `analysis/rallycut/tracking/contact_detector.py` — `_collect_best_per_track` helper + adaptive fallback in `_find_nearest_player(s)`.
- `analysis/tests/unit/test_contact_detector_adaptive_window.py` — 12 unit + integration tests.
- `analysis/scripts/regenerate_contact_candidates.py` — deploy companion script.
- `analysis/reports/attribution_baseline/adaptive_window_v30_2026_05_11.md` — measurement report.
- `analysis/reports/attribution_baseline/db_snapshots/pre_adaptive_window_2026_05_11.jsonl` — rollback snapshot.

## Commits

(Fill in commit SHAs from Tasks 1-7.)

## Open follow-ups

- **v3.0.1: multi-stage expand (60 → 120 frames)** catches 1 additional panel case (gigi/bc9345c1, gap=85). Add only if v3.0 measurement justifies it.
- **v3.1: cross-rally server identity** for cases with gap > 60 (gigi/72c8229b, bc9345c1, wawa/7094136a) and the 4 wrong-of-two-teammates serves.
- **v3.2: coherence-violation repair loop** for the rank-2 cross-team errors (8 of 17). Independent of this v3.
- **Static-track mislabel detection** for wawa/8c49e480 case (a non-player track was a primary).
```

Fill in measured-impact values from Tasks 5-7 results.

- [ ] **Step 2: Update MEMORY.md index**

Open `~/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/MEMORY.md`. Under "## Current workstreams", add as the top entry (most recent first):

```markdown
- [SHIPPED] [**Adaptive candidate window v3.0 2026-05-11**](adaptive_candidate_window_v30_2026_05_11.md) — Forward-only fallback in `_find_nearest_player(s)` when standard ±15 window is underfull. Catches late-tracked servers (validated cause of 7/8 absent-GT serves). New regeneration script applies v3 to existing rallies without retracking. Env flag `ADAPTIVE_PLAYER_SEARCH_WINDOW`. Panel +__pp, fleet C-2 __→__.
```

- [ ] **Step 3: No git commit for memory files** (they're outside the repo). Tasks 1-7 commits are the in-repo deliverables.

---

## Summary of touched files

In-repo (committed via Tasks 1-7):
- `analysis/rallycut/tracking/contact_detector.py` — `_collect_best_per_track` + adaptive fallback in `_find_nearest_player(s)`.
- `analysis/tests/unit/test_contact_detector_adaptive_window.py` — new test file (12 tests).
- `analysis/scripts/regenerate_contact_candidates.py` — new deploy companion script.
- `analysis/reports/attribution_baseline/adaptive_window_v30_2026_05_11.md` — measurement report.
- `analysis/reports/attribution_baseline/db_snapshots/pre_adaptive_window_2026_05_11.jsonl` — rollback snapshot.

Out-of-repo (Task 8):
- `~/.claude/projects/.../memory/adaptive_candidate_window_v30_2026_05_11.md`
- `~/.claude/projects/.../memory/MEMORY.md` — index entry.

## Rollback procedure

1. Set env: `ADAPTIVE_PLAYER_SEARCH_WINDOW=0` on the host running `reattribute-actions` or any pipeline component.
2. Re-run `regenerate_contact_candidates.py <video>` with the flag set — this re-derives candidates using the standard ±15 window (byte-identical to pre-v3).
3. Re-run `reattribute-actions <video>` to apply attribution downstream of the reverted candidates.
4. If env-flag rollback is insufficient (e.g., suspected code bug in the fallback path), restore from `db_snapshots/pre_adaptive_window_2026_05_11.jsonl` via psql `UPDATE player_tracks SET actions_json = ... WHERE id = pt_id` per row. (Note: this restores actions_json but contacts_json stays at the v3-regenerated state. For full revert: also restore contacts_json from a snapshot taken BEFORE Task 5 — currently not captured; could be re-derived by running regenerate with env=OFF.)
5. If the underlying code is suspect, `git revert <commits-from-tasks-1-3>` and redeploy.
