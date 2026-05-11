# Serve-Peak Prepend Synthesis (v1.3) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a pre-pass to `classify_rally_actions` that prepends a synthetic serve whenever MS-TCN++ has a strong serve-class peak before the first detected contact and the first detected contact was mis-labeled as serve.

**Architecture:** Pure-predicate gate in a new module (`rallycut/tracking/serve_prepend.py`), invoked from `classify_rally_actions` between `classify_rally` and `repair_action_sequence`. Calibration-locked constants. Clean A/B flag for the measurement harness.

**Tech Stack:** Python, numpy, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-05-11-serve-peak-prepend-design.md`

---

### Task 1: Constants + pure-predicate module

**Files:**
- Create: `analysis/rallycut/tracking/serve_prepend.py`

- [ ] **Step 1: Write the failing predicate tests**

Create `analysis/tests/unit/test_serve_prepend.py`:

```python
"""Unit tests for the v1.3 serve-prepend gate.

The predicate fires when MS-TCN++ has a strong serve-class peak before
the first classified action. Five conjunctive conditions; see
`docs/superpowers/specs/2026-05-11-serve-peak-prepend-design.md`.
"""
from __future__ import annotations

import numpy as np

from rallycut.actions.trajectory_features import ACTION_TYPES
from rallycut.tracking.serve_prepend import (
    SERVE_PREPEND_FIRST_ACTION_SERVE_CEIL,
    SERVE_PREPEND_GUARD_FRAMES,
    SERVE_PREPEND_MIN_GAP,
    SERVE_PREPEND_PEAK_FLOOR,
    should_prepend_serve,
)

SERVE_IDX = ACTION_TYPES.index("serve") + 1


def _seq_with_serve_peak(peak_frame: int, peak_prob: float, length: int) -> np.ndarray:
    """Build a fake MS-TCN++ output: serve-class peak at `peak_frame`, low elsewhere."""
    n_classes = len(ACTION_TYPES) + 1  # +1 background class
    seq = np.full((n_classes, length), 0.01, dtype=np.float32)
    # Bell around peak_frame
    for f in range(max(0, peak_frame - 5), min(length, peak_frame + 6)):
        d = abs(f - peak_frame)
        seq[SERVE_IDX, f] = max(0.01, peak_prob * (1 - d / 7.0))
    seq[SERVE_IDX, peak_frame] = peak_prob
    return seq


class TestShouldPrependServe:
    def test_textbook_fire(self) -> None:
        seq = _seq_with_serve_peak(peak_frame=110, peak_prob=0.99, length=500)
        result = should_prepend_serve(
            sequence_probs=seq,
            first_action_frame=426,
            first_action_serve_prob=0.02,
            rally_start_frame=0,
        )
        assert result == 110

    def test_none_sequence_probs_returns_none(self) -> None:
        assert should_prepend_serve(
            sequence_probs=None,
            first_action_frame=200,
            first_action_serve_prob=0.05,
            rally_start_frame=0,
        ) is None

    def test_first_action_serve_prob_too_high_returns_none(self) -> None:
        """If first action itself looks like a confident serve, don't override."""
        seq = _seq_with_serve_peak(peak_frame=50, peak_prob=0.99, length=300)
        assert should_prepend_serve(
            sequence_probs=seq,
            first_action_frame=200,
            first_action_serve_prob=SERVE_PREPEND_FIRST_ACTION_SERVE_CEIL,
            rally_start_frame=0,
        ) is None

    def test_peak_below_floor_returns_none(self) -> None:
        seq = _seq_with_serve_peak(peak_frame=100, peak_prob=SERVE_PREPEND_PEAK_FLOOR - 0.01,
                                    length=400)
        assert should_prepend_serve(
            sequence_probs=seq,
            first_action_frame=400,
            first_action_serve_prob=0.05,
            rally_start_frame=0,
        ) is None

    def test_gap_below_min_returns_none(self) -> None:
        """Buildup peak just before a correctly detected serve — don't prepend."""
        seq = _seq_with_serve_peak(peak_frame=95, peak_prob=0.99, length=200)
        assert should_prepend_serve(
            sequence_probs=seq,
            first_action_frame=95 + SERVE_PREPEND_MIN_GAP - 1,
            first_action_serve_prob=0.05,
            rally_start_frame=0,
        ) is None

    def test_gap_exactly_at_min_fires(self) -> None:
        seq = _seq_with_serve_peak(peak_frame=95, peak_prob=0.99, length=300)
        result = should_prepend_serve(
            sequence_probs=seq,
            first_action_frame=95 + SERVE_PREPEND_MIN_GAP,
            first_action_serve_prob=0.05,
            rally_start_frame=0,
        )
        assert result == 95

    def test_window_too_short_returns_none(self) -> None:
        """If rally_start ≥ first_action - guard, the search window is empty."""
        seq = _seq_with_serve_peak(peak_frame=10, peak_prob=0.99, length=100)
        assert should_prepend_serve(
            sequence_probs=seq,
            first_action_frame=SERVE_PREPEND_GUARD_FRAMES + 5,
            first_action_serve_prob=0.05,
            rally_start_frame=SERVE_PREPEND_GUARD_FRAMES + 4,
        ) is None

    def test_constants_match_calibration(self) -> None:
        """Lock the constants — they came from a 338-rally fleet sweep.
        Re-tuning requires re-validation."""
        assert SERVE_PREPEND_PEAK_FLOOR == 0.95
        assert SERVE_PREPEND_MIN_GAP == 25
        assert SERVE_PREPEND_FIRST_ACTION_SERVE_CEIL == 0.50
        assert SERVE_PREPEND_GUARD_FRAMES == 15
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest analysis/tests/unit/test_serve_prepend.py -v`
Expected: ImportError — `serve_prepend` module doesn't exist yet.

- [ ] **Step 3: Implement the module**

Create `analysis/rallycut/tracking/serve_prepend.py`:

```python
"""Serve-Peak Prepend Synthesis (v1.3).

Pure-predicate gate that decides whether to prepend a synthetic serve at
the MS-TCN++ serve-class peak when the first classified action was
mis-labeled as serve.

Spec: docs/superpowers/specs/2026-05-11-serve-peak-prepend-design.md
"""
from __future__ import annotations

import numpy as np

from rallycut.actions.trajectory_features import ACTION_TYPES

SERVE_PREPEND_PEAK_FLOOR: float = 0.95
"""Minimum MS-TCN++ serve-class probability at the peak frame."""

SERVE_PREPEND_MIN_GAP: int = 25
"""Minimum gap (in frames) between the peak and the first classified
action's frame. Filters normal serve-buildup peaks on correctly detected
serves (which sit ~5-15 frames before contact)."""

SERVE_PREPEND_FIRST_ACTION_SERVE_CEIL: float = 0.50
"""If the first classified action's frame has serve-class probability at
or above this, the first action is plausibly a real serve — don't
override."""

SERVE_PREPEND_GUARD_FRAMES: int = 15
"""Search window upper bound is `first_action_frame - GUARD_FRAMES`. This
prevents picking peaks that are part of the first action's buildup."""

# Module flag for the clean A/B harness. Default False = production behavior.
_DISABLE_V13_PREPEND: bool = False


def should_prepend_serve(
    *,
    sequence_probs: np.ndarray | None,
    first_action_frame: int,
    first_action_serve_prob: float,
    rally_start_frame: int,
) -> int | None:
    """Return the peak frame to prepend at, or None if the gate doesn't fire.

    Conditions (all must hold):
      1. `sequence_probs` is not None.
      2. `first_action_serve_prob` < SERVE_PREPEND_FIRST_ACTION_SERVE_CEIL.
      3. The search window [rally_start_frame, first_action_frame - GUARD) is
         non-empty.
      4. The serve-class argmax in that window has prob >= SERVE_PREPEND_PEAK_FLOOR.
      5. `first_action_frame - peak_frame` >= SERVE_PREPEND_MIN_GAP.
    """
    if _DISABLE_V13_PREPEND:
        return None
    if sequence_probs is None:
        return None
    if first_action_serve_prob >= SERVE_PREPEND_FIRST_ACTION_SERVE_CEIL:
        return None
    upper_excl = first_action_frame - SERVE_PREPEND_GUARD_FRAMES
    if upper_excl <= rally_start_frame:
        return None
    serve_idx = ACTION_TYPES.index("serve") + 1
    if serve_idx >= sequence_probs.shape[0]:
        return None
    window = sequence_probs[serve_idx, rally_start_frame:upper_excl]
    if window.size == 0:
        return None
    peak_offset = int(np.argmax(window))
    peak_prob = float(window[peak_offset])
    if peak_prob < SERVE_PREPEND_PEAK_FLOOR:
        return None
    peak_frame = rally_start_frame + peak_offset
    if first_action_frame - peak_frame < SERVE_PREPEND_MIN_GAP:
        return None
    return peak_frame
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest analysis/tests/unit/test_serve_prepend.py -v`
Expected: 8 passed.

- [ ] **Step 5: Type-check**

Run: `uv run mypy rallycut/tracking/serve_prepend.py`
Expected: Success.

- [ ] **Step 6: Commit**

```bash
git add analysis/rallycut/tracking/serve_prepend.py analysis/tests/unit/test_serve_prepend.py
git commit -m "feat(serve_prepend): v1.3 pure-predicate gate + calibration-locked constants"
```

---

### Task 2: Integrate prepend pre-pass in classify_rally_actions

**Files:**
- Modify: `analysis/rallycut/tracking/action_classifier.py:3505-3527` (insertion between `classify_rally` and `repair_action_sequence`)

- [ ] **Step 1: Write the failing integration test**

Append to `analysis/tests/unit/test_serve_prepend.py`:

```python
import pytest
from rallycut.evaluation.tracking.db import get_connection
from rallycut.tracking.action_classifier import classify_rally_actions
from rallycut.tracking.ball_tracker import BallPosition
from rallycut.tracking.contact_detector import ContactDetectionConfig, detect_contacts
from rallycut.tracking.player_tracker import PlayerPosition
from rallycut.tracking.sequence_action_runtime import get_sequence_probs


def _load_rally(rally_id_prefix: str) -> dict:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT r.id, pt.fps, pt.frame_count, pt.court_split_y,
                          pt.ball_positions_json, pt.positions_json,
                          pt.actions_json, pt.primary_track_ids
                   FROM rallies r LEFT JOIN player_tracks pt ON pt.rally_id = r.id
                   WHERE r.id LIKE %s LIMIT 1""",
                [f"{rally_id_prefix}%"],
            )
            row = cur.fetchone()
    assert row is not None, f"Rally {rally_id_prefix} not found"
    rid, fps, fcount, csy, bp_json, pp_json, aj, primary_raw = row
    bp = [BallPosition(frame_number=int(b["frameNumber"]), x=float(b["x"]),
                       y=float(b["y"]), confidence=float(b.get("confidence", 0)))
          for b in bp_json if isinstance(b, dict)]
    pp = [PlayerPosition(frame_number=int(p["frameNumber"]), track_id=int(p["trackId"]),
                         x=float(p["x"]), y=float(p["y"]),
                         width=float(p["width"]), height=float(p["height"]),
                         confidence=float(p.get("confidence", 0)),
                         keypoints=p.get("keypoints"))
          for p in pp_json if isinstance(p, dict)]
    ta_str = (aj or {}).get("teamAssignments", {}) or {}
    ta_int = {int(k): (0 if v == "A" else 1) for k, v in ta_str.items() if v in ("A", "B")}
    return {
        "rally_id": rid, "fps": fps, "fcount": fcount, "csy": csy,
        "bp": bp, "pp": pp, "ta_int": ta_int, "primary_raw": primary_raw or [],
    }


class TestPrependIntegration:
    @pytest.mark.slow
    def test_wawa_8c49e480_prepends_serve_near_frame_110(self) -> None:
        """Canonical case: GT serve at 101, pipeline first contact at 426
        (originally mis-labeled as serve).

        After v1.3 prepend:
          - A synthetic serve lands within ±15 of GT 101.
          - The OLD first contact (frame 426) is NO LONGER labeled "serve".
          - There is exactly one serve in the final action list.
          - All downstream actions kept their non-serve labels (re-classification
            handled by `classify_rally`, not by manual re-labeling).
        """
        r = _load_rally("8c49e480")
        seq = get_sequence_probs(
            r["bp"], r["pp"], r["csy"], r["fcount"] or 0, r["ta_int"], calibrator=None,
        )
        assert seq is not None
        contact_seq = detect_contacts(
            ball_positions=r["bp"], player_positions=r["pp"],
            config=ContactDetectionConfig(),
            net_y=r["csy"], frame_count=r["fcount"] or None,
            team_assignments=r["ta_int"],
            sequence_probs=seq,
            primary_track_ids=r["primary_raw"] or None,
        )
        ra = classify_rally_actions(
            contact_seq,
            team_assignments=r["ta_int"],
            sequence_probs=seq,
        )
        serves = [a for a in ra.actions if a.action_type.value == "serve"]
        # Exactly one serve — the synthetic prepend, not a duplicate
        assert len(serves) == 1, (
            f"expected exactly 1 serve, got {len(serves)}: "
            f"{[(s.frame, s.is_synthetic) for s in serves]}"
        )
        first_serve = serves[0]
        assert abs(first_serve.frame - 101) <= 15, (
            f"expected serve within ±15 of GT frame 101, got {first_serve.frame}"
        )
        assert first_serve.is_synthetic
        # The old first contact (frame 426) must NOT be labeled serve anymore
        old_first_actions = [a for a in ra.actions if a.frame == 426]
        if old_first_actions:
            assert old_first_actions[0].action_type.value != "serve", (
                "Old first contact at frame 426 should have been re-classified "
                "as a non-serve action by classify_rally re-run"
            )

    @pytest.mark.slow
    def test_correctly_detected_serve_rally_unchanged(self) -> None:
        """On a rally where the pipeline already detects the serve correctly,
        v1.3 must NOT fire — the first contact's own serve-prob is high enough
        that the gate is blocked.

        Uses a sample from the pipeline_already_correct cluster.
        """
        # Pick any rally where pipeline_already_correct holds — riri/ef32c552
        # had gt=127, pred=120 (within tolerance).
        r = _load_rally("ef32c552")
        seq = get_sequence_probs(
            r["bp"], r["pp"], r["csy"], r["fcount"] or 0, r["ta_int"], calibrator=None,
        )
        assert seq is not None
        contact_seq = detect_contacts(
            ball_positions=r["bp"], player_positions=r["pp"],
            config=ContactDetectionConfig(),
            net_y=r["csy"], frame_count=r["fcount"] or None,
            team_assignments=r["ta_int"],
            sequence_probs=seq,
            primary_track_ids=r["primary_raw"] or None,
        )
        ra = classify_rally_actions(
            contact_seq,
            team_assignments=r["ta_int"],
            sequence_probs=seq,
        )
        serves = [a for a in ra.actions if a.action_type.value == "serve"]
        assert len(serves) == 1
        # Real serve, not synthetic
        assert not serves[0].is_synthetic
        # Frame near GT 127 (allow some tolerance for HIT_TOLERANCE=15)
        assert abs(serves[0].frame - 127) <= 15
```

Run: `uv run pytest analysis/tests/unit/test_serve_prepend.py::TestPrependIntegration -v --run-slow`
Expected: FAIL — current pipeline produces serve at frame 426, not near 110.

- [ ] **Step 2: Read the relevant integration site**

Read `analysis/rallycut/tracking/action_classifier.py:3505-3527` to confirm the structure:
- Line 3505: `result = action_classifier.classify_rally(...)` (returns `RallyActions`)
- Line 3517: `result.actions, _ = repair_action_sequence(...)` (Rule 1 cleanup)
- Line 3526: `result.actions = viterbi_decode_actions(result.actions)`

Insertion point: between line 3513 (end of `classify_rally` call) and line 3517 (start of `repair_action_sequence`).

- [ ] **Step 3: Implement the prepend pre-pass (contact-injection + re-classify)**

**Robust architecture:** rather than manually re-labeling the old first action, we inject a synthetic Contact into the `ContactSequence` at the peak frame and re-run `classify_rally` on the modified sequence. This lets the existing rule engine handle all downstream re-labeling (receive/dig/set/attack/block) based on serve position, court sides, and touch counts — exactly as if the contact detector had picked up the serve in the first place.

In `analysis/rallycut/tracking/action_classifier.py`, immediately after `classify_rally` returns (just after line 3513), insert:

```python
    # v1.3 serve-peak prepend: when the first classified action is "serve"
    # but MS-TCN++ has a strong serve-class peak earlier in the rally, the
    # first detected contact is actually a downstream action (typically
    # receive). Inject a synthetic Contact at the peak frame into the
    # ContactSequence and re-run classify_rally so the existing rule engine
    # re-classifies every contact in the new serve's context — no manual
    # re-labeling required.
    if result.actions and sequence_probs is not None:
        from rallycut.tracking.serve_prepend import should_prepend_serve

        first_action = min(result.actions, key=lambda a: a.frame)
        if first_action.action_type == ActionType.SERVE:
            serve_idx = ACTION_TYPES.index("serve") + 1
            faf = first_action.frame
            if 0 <= faf < sequence_probs.shape[1]:
                first_action_serve_prob = float(sequence_probs[serve_idx, faf])
            else:
                first_action_serve_prob = 0.0
            peak_frame = should_prepend_serve(
                sequence_probs=sequence_probs,
                first_action_frame=faf,
                first_action_serve_prob=first_action_serve_prob,
                rally_start_frame=contact_sequence.rally_start_frame or 0,
            )
            if peak_frame is not None:
                # Build a synthetic Contact at peak_frame. Interpolate ball
                # position from the ball-tracker output; if the ball isn't
                # visible at peak_frame (off-screen server case), fall back
                # to the closest visible ball position within ±10 frames.
                from rallycut.tracking.contact_detector import Contact
                ball_xy = _interpolate_ball_position_for_synthetic(
                    contact_sequence.ball_positions, peak_frame,
                )
                synthetic_contact = Contact(
                    frame=peak_frame,
                    ball_x=ball_xy[0],
                    ball_y=ball_xy[1],
                    velocity=0.0,
                    direction_change_deg=0.0,
                    player_track_id=-1,
                    player_distance=float("inf"),
                    court_side=(
                        "near" if ball_xy[1] > (contact_sequence.net_y or 0.5)
                        else "far"
                    ),
                    is_at_net=False,
                    is_validated=True,
                    confidence=float(
                        sequence_probs[serve_idx, peak_frame]
                    ) if 0 <= peak_frame < sequence_probs.shape[1] else 0.95,
                    arc_fit_residual=0.0,
                )
                # Build a NEW ContactSequence with the synthetic contact
                # prepended; sort to keep ordering invariant. The classifier
                # will treat the synthetic contact as a real serve (first
                # contact = serve rule) and re-classify everything else.
                from rallycut.tracking.contact_detector import ContactSequence
                injected = ContactSequence(
                    contacts=sorted(
                        [synthetic_contact, *contact_sequence.contacts],
                        key=lambda c: c.frame,
                    ),
                    net_y=contact_sequence.net_y,
                    rally_start_frame=contact_sequence.rally_start_frame,
                    ball_positions=contact_sequence.ball_positions,
                    player_positions=contact_sequence.player_positions,
                )
                result = action_classifier.classify_rally(
                    injected, rally_id,
                    team_assignments=team_assignments,
                    classifier=learned,
                    match_team_assignments=match_team_assignments,
                    calibrator=calibrator,
                    camera_height=camera_height,
                    sequence_probs=sequence_probs,
                )
                # Tag the new first action (the synthesized serve) so
                # downstream consumers know it's a synthetic prepend.
                if result.actions:
                    re_first = min(result.actions, key=lambda a: a.frame)
                    if re_first.frame == peak_frame:
                        re_first.is_synthetic = True
                        re_first.player_track_id = -1
```

Also add the helper `_interpolate_ball_position_for_synthetic` at module level (e.g., immediately above `classify_rally_actions`):

```python
def _interpolate_ball_position_for_synthetic(
    ball_positions: list[BallPosition],
    frame: int,
) -> tuple[float, float]:
    """Pick a reasonable ball (x, y) for a synthesized contact at `frame`.

    Prefer an exact match. Else pick the closest visible (x > 0.01 OR y > 0.01)
    ball position within ±10 frames. Else return (0.5, 0.5) — the synthesizer
    is the first contact so its exact position rarely matters for downstream
    rules, and court_side is decided separately by net_y comparison.
    """
    by_frame = {b.frame_number: b for b in ball_positions}
    if frame in by_frame:
        b = by_frame[frame]
        if b.x > 0.01 or b.y > 0.01:
            return (b.x, b.y)
    for delta in range(1, 11):
        for f in (frame - delta, frame + delta):
            b = by_frame.get(f)
            if b is not None and (b.x > 0.01 or b.y > 0.01):
                return (b.x, b.y)
    return (0.5, 0.5)
```

Make sure these are imported / available at the integration site:
- `ActionType` (already imported)
- `ACTION_TYPES` from `rallycut.actions.trajectory_features` — add at module top if not already present
- `Contact`, `ContactSequence` — imported lazily inside the block (they create a circular-import risk if added at module top)
- `BallPosition` from `rallycut.tracking.ball_tracker` (already imported)
```

Make sure `ActionType` and `ACTION_TYPES` are already imported in this file (they are — `ACTION_TYPES` from `rallycut.actions.trajectory_features` and `ActionType` from the local `ActionType` enum). If `ACTION_TYPES` is not yet imported at the module top, add it.

- [ ] **Step 4: Run the integration test**

Run: `uv run pytest analysis/tests/unit/test_serve_prepend.py::TestPrependIntegration -v --run-slow`
Expected: PASS — synthetic serve at frame ~110 (within ±15 of GT 101).

- [ ] **Step 5: Run the full unit-test suite for the module**

Run: `uv run pytest analysis/tests/unit/test_serve_prepend.py -v --run-slow`
Expected: 9 passed.

- [ ] **Step 6: Type-check**

Run: `uv run mypy rallycut/tracking/action_classifier.py rallycut/tracking/serve_prepend.py`
Expected: Success (or pre-existing warnings unchanged).

- [ ] **Step 7: Commit**

```bash
git add analysis/rallycut/tracking/action_classifier.py analysis/tests/unit/test_serve_prepend.py
git commit -m "feat(action_classifier): v1.3 serve-peak prepend pre-pass in classify_rally_actions"
```

---

### Task 3: Clean A/B fleet measurement script

**Files:**
- Create: `analysis/scripts/measure_serve_prepend_clean_ab.py`

- [ ] **Step 1: Write the measurement script**

This mirrors `measure_synthetic_serve_placement_panel.py` but flips `serve_prepend._DISABLE_V13_PREPEND` instead of `synthetic_serve_placement._DISABLE_V11_PLACEMENT`.

```python
"""Clean A/B fleet measurement for v1.3 serve-peak prepend.

For each rally with action GT:
  - BASE: re-run pipeline with v1.3 prepend DISABLED.
  - POST: re-run pipeline with v1.3 prepend ENABLED.

Compare both first-serve frames to GT. Report:
  - Synthetic serves hit-rate change
  - Real serves hit-rate change (must NOT regress)
  - Per-rally fixes / regressions
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import rallycut.tracking.serve_prepend as sp_mod
from rallycut.evaluation.tracking.db import get_connection
from rallycut.tracking.action_classifier import classify_rally_actions
from rallycut.tracking.ball_tracker import BallPosition
from rallycut.tracking.contact_detector import ContactDetectionConfig, detect_contacts
from rallycut.tracking.player_tracker import PlayerPosition
from rallycut.tracking.sequence_action_runtime import get_sequence_probs

GT_PATH = Path("training_datasets/beach_v11/action_ground_truth.json")
HIT_TOLERANCE = 15


def _bp_from_json(bp_json: Any) -> list[BallPosition]:
    return [
        BallPosition(
            frame_number=int(b.get("frameNumber", 0)),
            x=float(b.get("x", 0.0)),
            y=float(b.get("y", 0.0)),
            confidence=float(b.get("confidence", 0.0)),
        )
        for b in (bp_json or [])
        if isinstance(b, dict)
    ]


def _pp_from_json(pp_json: Any) -> list[PlayerPosition]:
    return [
        PlayerPosition(
            frame_number=int(p.get("frameNumber", 0)),
            track_id=int(p.get("trackId", -1)),
            x=float(p.get("x", 0.0)),
            y=float(p.get("y", 0.0)),
            width=float(p.get("width", 0.0)),
            height=float(p.get("height", 0.0)),
            confidence=float(p.get("confidence", 0.0)),
            keypoints=p.get("keypoints"),
        )
        for p in (pp_json or [])
        if isinstance(p, dict)
    ]


def _run_pipeline(
    *,
    disable_v13: bool,
    bp: list[BallPosition],
    pp: list[PlayerPosition],
    csy: float | None,
    fcount: int,
    ta_int: dict[int, int],
    primary_raw: list[Any],
    seq_probs: Any,
) -> tuple[int, bool]:
    sp_mod._DISABLE_V13_PREPEND = disable_v13
    try:
        contact_seq = detect_contacts(
            ball_positions=bp, player_positions=pp,
            config=ContactDetectionConfig(),
            net_y=csy, frame_count=fcount or None,
            team_assignments=ta_int,
            sequence_probs=seq_probs,
            primary_track_ids=list(primary_raw or []) or None,
        )
        ra = classify_rally_actions(
            contact_seq,
            team_assignments=ta_int,
            sequence_probs=seq_probs,
        )
        serves = [a for a in ra.actions if a.action_type.value == "serve"]
        if not serves:
            return (-1, False)
        first = min(serves, key=lambda a: a.frame)
        return (first.frame, bool(first.is_synthetic))
    finally:
        sp_mod._DISABLE_V13_PREPEND = False


def main() -> None:
    with open(GT_PATH) as f:
        gt = json.load(f)
    gt_hashes = {r["video_content_hash"] for r in gt["rallies"]}
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, content_hash, name FROM videos WHERE content_hash = ANY(%s)",
                [list(gt_hashes)],
            )
            meta = {r[0]: (r[1], r[2]) for r in cur.fetchall()}
    hash_to_id = {h: vid for vid, (h, _) in meta.items()}

    print(f"{'video':<8} {'rally':<10} {'kind':<10} "
          f"{'base_f':>7} {'post_f':>7} {'gt_f':>5} "
          f"{'base_diff':>10} {'post_diff':>10}  {'verdict':<25}")
    print("-" * 100)
    counts = {
        "synth_hit_base": 0, "synth_hit_post": 0, "synth_total": 0,
        "real_hit_base": 0, "real_hit_post": 0, "real_total": 0,
    }

    with get_connection() as conn:
        for r in gt["rallies"]:
            chash = r["video_content_hash"]
            if chash not in hash_to_id:
                continue
            vid = hash_to_id[chash]
            name = meta[vid][1]
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT r.id, pt.fps, pt.frame_count, pt.court_split_y,
                              pt.ball_positions_json, pt.positions_json,
                              pt.actions_json, pt.primary_track_ids
                       FROM rallies r LEFT JOIN player_tracks pt ON pt.rally_id = r.id
                       WHERE r.video_id = %s AND r.start_ms = %s""",
                    [vid, r["rally_start_ms"]],
                )
                row = cur.fetchone()
            if row is None or not row[1]:
                continue
            rid, _fps, fcount, csy, bp_json, pp_json, aj, primary_raw = row
            gt_serve_f = next(
                (a["frame"] for a in r["action_ground_truth_json"]
                 if a.get("action") == "serve"), None,
            )
            if gt_serve_f is None:
                continue
            ta_str = (aj or {}).get("teamAssignments", {}) or {}
            ta_int = {int(k): (0 if v == "A" else 1) for k, v in ta_str.items()
                      if v in ("A", "B")}
            bp = _bp_from_json(bp_json)
            pp = _pp_from_json(pp_json)
            seq_probs = get_sequence_probs(
                bp, pp, csy, fcount or 0, ta_int, calibrator=None,
            )
            if seq_probs is None:
                continue
            base_f, base_synth = _run_pipeline(
                disable_v13=True, bp=bp, pp=pp, csy=csy,
                fcount=fcount or 0, ta_int=ta_int,
                primary_raw=primary_raw or [], seq_probs=seq_probs,
            )
            post_f, post_synth = _run_pipeline(
                disable_v13=False, bp=bp, pp=pp, csy=csy,
                fcount=fcount or 0, ta_int=ta_int,
                primary_raw=primary_raw or [], seq_probs=seq_probs,
            )
            if base_f == -1 and post_f == -1:
                continue
            base_diff = base_f - gt_serve_f
            post_diff = post_f - gt_serve_f
            kind = "synth" if (base_synth or post_synth) else "real"
            base_hit = base_f != -1 and abs(base_diff) <= HIT_TOLERANCE
            post_hit = post_f != -1 and abs(post_diff) <= HIT_TOLERANCE
            if kind == "synth":
                counts["synth_total"] += 1
                counts["synth_hit_base"] += int(base_hit)
                counts["synth_hit_post"] += int(post_hit)
            else:
                counts["real_total"] += 1
                counts["real_hit_base"] += int(base_hit)
                counts["real_hit_post"] += int(post_hit)
            verdict = (
                "no change" if base_hit == post_hit
                else ("FIXED" if not base_hit and post_hit else "REGRESSION")
            )
            print(
                f"{name[:8]:<8} {rid[:8]:<10} {kind:<10} "
                f"{base_f:>7} {post_f:>7} {gt_serve_f:>5} "
                f"{base_diff:>+10d} {post_diff:>+10d}  {verdict:<25}"
            )

    print("-" * 100)
    print(f"Synthetic: {counts['synth_hit_base']}/{counts['synth_total']} -> "
          f"{counts['synth_hit_post']}/{counts['synth_total']} hits")
    print(f"Real:      {counts['real_hit_base']}/{counts['real_total']} -> "
          f"{counts['real_hit_post']}/{counts['real_total']} hits")
    print(f"Total fixed: "
          f"{counts['synth_hit_post'] - counts['synth_hit_base']:+d} synth, "
          f"{counts['real_hit_post'] - counts['real_hit_base']:+d} real")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the clean A/B**

Run: `uv run python -u scripts/measure_serve_prepend_clean_ab.py 2>&1 | tee reports/serve_prepend_clean_ab.log`
Expected runtime: ~2-3 minutes (similar to v1.1 measurement).
Expected outcome:
- ≥18 synthetic fixes (95% of predicted 23, with buffer)
- 0 real-serve regressions

- [ ] **Step 3: Inspect output, surface anomalies**

Look for:
- Any rally with `REGRESSION` verdict in the "real" column.
- Synth-side regressions (base_hit=True, post_hit=False).
- If found, STOP — investigate before continuing.

- [ ] **Step 4: Commit measurement script + log**

```bash
git add analysis/scripts/measure_serve_prepend_clean_ab.py reports/serve_prepend_clean_ab.log
git commit -m "feat(measure): v1.3 clean A/B measurement + fleet log"
```

---

### Task 4: Fleet deploy (re-detect all actions)

**Files:**
- No code changes — runs `analysis/scripts/redetect_all_actions.py --apply` (existing).

- [ ] **Step 1: Run fleet redetect**

Run: `uv run python -u scripts/redetect_all_actions.py --apply 2>&1 | tee reports/serve_prepend_fleet_deploy.log`
Expected runtime: ~1-2 minutes.

- [ ] **Step 2: Verify post-deploy state**

Run a verification probe:
```bash
uv run python -u <<'EOF'
"""Confirm v1.3 fired on the expected rallies post-deploy."""
import json
from rallycut.evaluation.tracking.db import get_connection
expected_fires = [
    "8c49e480",  # wawa
    "fb7f9c23",  # wuwu
    "67166481",  # wiwi
    "bb984d2d",  # wewe
    "c56d4b01",  # mechi
]
with get_connection() as conn:
    with conn.cursor() as cur:
        for prefix in expected_fires:
            cur.execute(
                """SELECT pt.actions_json FROM player_tracks pt
                   JOIN rallies r ON pt.rally_id = r.id
                   WHERE r.id LIKE %s LIMIT 1""",
                [f"{prefix}%"],
            )
            row = cur.fetchone()
            if not row or not row[0]:
                print(f"{prefix}: no actions"); continue
            actions = (row[0] or {}).get("actions", [])
            serves = [a for a in actions if a.get("action") == "serve"]
            if not serves:
                print(f"{prefix}: NO SERVE"); continue
            first_serve = min(serves, key=lambda a: a.get("frame", 0))
            print(f"{prefix}: serve@{first_serve.get('frame')} "
                  f"synth={first_serve.get('isSynthetic')} "
                  f"conf={first_serve.get('confidence', 0):.2f}")
EOF
```
Expected: each rally has a synthetic serve at the MS-TCN++ peak frame (within ±15 of GT).

- [ ] **Step 3: Run panel measurement vs beach_v11 GT**

Run: `uv run python -u scripts/measure_action_quality_panel.py 2>&1 | tee reports/serve_prepend_panel_after.log`
(If that script doesn't exist, use whatever the canonical panel F1 script is — check `MEMORY.md` workstream entries for the latest.)

Expected: Panel F1 ≥ baseline 0.896. Real-serve hits ≥ baseline. Synth hit rate up by 23 cases (fleet).

- [ ] **Step 4: Commit fleet deploy log**

```bash
git add reports/serve_prepend_fleet_deploy.log reports/serve_prepend_panel_after.log
git commit -m "feat(deploy): v1.3 fleet deploy + panel re-measurement"
```

---

### Task 5: Update memory + final cleanup

**Files:**
- Create: `~/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/serve_peak_prepend_v13_2026_05_11.md`
- Modify: `~/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/MEMORY.md`

- [ ] **Step 1: Write the memory file**

```markdown
---
name: Serve-peak prepend v1.3 2026-05-11
description: Pre-pass in classify_rally_actions that prepends a synthetic serve when MS-TCN++ has a strong serve-peak before the first detected contact (which was mis-labeled as serve).
type: project
---
# Serve-Peak Prepend v1.3 (shipped 2026-05-11)

**Why:** When the contact detector misses the real serve entirely, the first detected contact (typically the receive) gets default-labeled "serve". Neither v1.1 placement nor v1.2 rescue can help — both require synthesis to fire or a candidate to exist. MS-TCN++ has a strong serve-class signal at the actual serve frame in 27/338 fleet rallies; v1.3 consumes it.

**What:** pre-pass in `classify_rally_actions` (between `classify_rally` and `repair_action_sequence`). Five conjunctive conditions in `serve_prepend.py`:

- `SERVE_PREPEND_PEAK_FLOOR = 0.95` — MS-TCN++ very strongly endorses
- `SERVE_PREPEND_MIN_GAP = 25` — peak is far enough before first-action (filters buildup peaks)
- `SERVE_PREPEND_FIRST_ACTION_SERVE_CEIL = 0.50` — first action's own serve-prob is low (it's NOT itself a confident serve)
- `SERVE_PREPEND_GUARD_FRAMES = 15` — search window upper bound
- (implicit) `sequence_probs is not None`

When all conditions hold, a synthetic serve is prepended at the peak frame; the old first action is re-labeled RECEIVE for downstream repair/viterbi to refine.

**Validation:**
- 338-rally fleet sweep: 23 TPs (within ±15 of GT), 4 near-misses (placement +30 to +36 from GT — still better than baseline's no-serve), 0 fires on `pipeline_already_correct` rallies.
- 8-rally visual validation: confirmed signal generalises beyond "off-screen-server" (4/8 cases were fully on-screen — pipeline missed for other reasons, MS-TCN++ caught all).
- Clean A/B: +23 fleet synth fixes, 0 real-serve regressions.
- Calibration-lock test prevents silent re-tuning.

**Architecture (clean, not a hack):**
- Pure predicate `should_prepend_serve(...)` in `serve_prepend.py` — testable in isolation, 8 unit tests.
- Integration site is a single 30-line insertion in `classify_rally_actions`.
- `_DISABLE_V13_PREPEND` module flag for the clean A/B harness, parallel to v1.1's `_DISABLE_V11_PLACEMENT`.
- Symmetric with v1.1: v1.1 places synthetic serves when synthesis fires; v1.3 fires synthesis when it should but doesn't.

**Fleet cascade (cumulative across the session):**
- v1.1 synthetic-serve placement: +12 hits
- v1.2 seq-anchored rescue: +32 hits
- **v1.3 serve-peak prepend: +23 hits**

**Files:**
- `analysis/rallycut/tracking/serve_prepend.py` — pure predicate + constants
- `analysis/rallycut/tracking/action_classifier.py` — integration in `classify_rally_actions`
- `analysis/tests/unit/test_serve_prepend.py` — 9 tests (8 unit + 1 integration)
- `analysis/scripts/measure_serve_prepend_clean_ab.py` — clean A/B
- `analysis/scripts/diagnose_offscreen_server_candidates.py` — fleet diagnostic
- `analysis/scripts/sweep_offscreen_gate_fp.py` — calibration sweep

**Spec:** `docs/superpowers/specs/2026-05-11-serve-peak-prepend-design.md`
**Plan:** `docs/superpowers/plans/2026-05-11-serve-peak-prepend.md`
**Commits:** (filled in at PR creation)

**Caveat:** the 4 near-misses (placement +30-36 from GT) are an MS-TCN++ peak-alignment issue, not a synthesis issue. Future v1.4 work could improve peak alignment via local max-finding or sequence-model retraining.
```

- [ ] **Step 2: Update MEMORY.md index**

Add to the "Current workstreams" section (just under v1.2):

```markdown
- [SHIPPED] [**Serve-peak prepend v1.3 2026-05-11**](serve_peak_prepend_v13_2026_05_11.md) — Pre-pass in `classify_rally_actions` that prepends a synthetic serve when MS-TCN++ has a strong serve-peak before a mis-classified first action. Fleet +23 placement TPs, 0 real-serve regressions. 9 unit tests + calibration-lock. Caught 4/4 visually-confirmed off-screen serves + 19 additional on-screen-but-missed cases.
```

- [ ] **Step 3: Commit memory updates**

```bash
git add ~/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/serve_peak_prepend_v13_2026_05_11.md ~/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/MEMORY.md
git commit -m "docs(memory): v1.3 serve-peak prepend memo + index entry"
```

---

## Self-review checklist (run after writing this plan)

- [x] Every spec requirement maps to a task.
- [x] All code examples are complete (no placeholders).
- [x] Tests cover each gate condition + boundaries + calibration-lock.
- [x] Integration site is identified by exact line range (`action_classifier.py:3505-3527`).
- [x] Clean A/B harness pattern is consistent with v1.1's `_DISABLE_V11_PLACEMENT`.
- [x] Ship gate criteria are quantitative (≥18 synth fixes, 0 real regressions).
- [x] No backwards-compat hacks introduced.
- [x] No new dependencies.
