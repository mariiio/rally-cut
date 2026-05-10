# Synthetic-Serve Frame Placement (v1.1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the placeholder `first_contact_frame - 30` formula in `_make_synthetic_serve` with two-signal evidence-based placement (MS-TCN++ serve-class peak + ball-trajectory direction-change) so synthetic serves land at the actual serve frame.

**Architecture:** New pure helper `pick_synthetic_serve_frame(*, sequence_probs, ball_positions, rally_start_frame, first_contact_frame)` returning a frame or None. `_make_synthetic_serve` calls it when both signals are available; falls back to current formula when not. Plumbing threads `sequence_probs` (already at `classify_rally_actions` level) and `ball_positions` (available via `ContactSequence`) down to the two existing call sites in `action_classifier.py:1894` and `:2377`.

**Tech Stack:** Python 3.11, NumPy. No new dependencies. Tests via pytest. Repo standards: mypy --strict, ruff.

**Spec:** `docs/superpowers/specs/2026-05-10-synthetic-serve-placement-design.md`

---

## File Structure

**New (1 module + 1 test + 1 script):**
- `analysis/rallycut/tracking/synthetic_serve_placement.py` — pure helper (~100 LOC).
- `analysis/tests/unit/test_synthetic_serve_placement.py` — unit tests for the helper.
- `analysis/scripts/measure_synthetic_serve_placement_panel.py` — pre/post measurement.

**Modified (1):**
- `analysis/rallycut/tracking/action_classifier.py` — `_make_synthetic_serve` signature + body, plumbing in `classify_rally`.

**No CLI changes** — production paths (`track_player.py`, `analyze.py`) already pass `sequence_probs` to `classify_rally_actions`. `ball_positions` may need to be threaded if not already; the implementer's first action in Task 3 is to verify and add the kwarg if missing.

---

## Task 1: Calibrate the trajectory-burst threshold

**Files:**
- Create: `analysis/scripts/calibrate_serve_burst_threshold.py`

**Why first**: the spec specifies a `BURST_THRESHOLD` for direction-change-based detection but leaves the value to calibration. Without this, the helper's Signal B is a guess. We measure the direction-change values at the 14 correctly-placed real serves on the panel and pick a threshold that separates them from non-serve frames in their search windows.

- [ ] **Step 1: Write the calibration script**

Create `analysis/scripts/calibrate_serve_burst_threshold.py` with this content:

```python
"""Calibrate the trajectory direction-change threshold for serve detection.

For each panel rally with a correctly-placed real (non-synthetic) serve:
  - Load ball_positions
  - At the GT serve frame, compute compute_direction_change(...) over a
    range of check_frames windows (4, 6, 8).
  - Print the value at the GT serve frame and the max value at any non-
    serve frame in [rally_start, first_real_contact_frame].

Use the distribution to pick BURST_THRESHOLD such that >=90% of GT serve
frames exceed it AND <10% of non-serve frames in the search windows do.
"""

from __future__ import annotations

import json
from pathlib import Path

from rallycut.evaluation.tracking.db import get_connection
from rallycut.tracking.ball_tracker import BallPosition
from rallycut.tracking.contact_detector import compute_direction_change

GT_PATH = Path("training_datasets/beach_v11/action_ground_truth.json")
# Use every video that has action GT — frame and action-type labels are
# clean; only player attribution is known-noisy and we don't compare it
# here. The 5-video "panel" was just a quick subset used during the prior
# diagnostic session.
WINDOW_VALUES = [4, 6, 8]


def _ball_by_frame(bp_json):
    out = {}
    for b in bp_json or []:
        if not isinstance(b, dict):
            continue
        f = int(b.get("frameNumber", b.get("frame", 0)))
        out[f] = BallPosition(
            frame_number=f,
            x=float(b.get("x", 0.0)),
            y=float(b.get("y", 0.0)),
            confidence=float(b.get("confidence", 0.0)),
        )
    return out


def main() -> None:
    with open(GT_PATH) as f:
        gt = json.load(f)

    # Derive video IDs from the GT file: any content_hash referenced is fair game.
    gt_hashes = {r["video_content_hash"] for r in gt["rallies"]}
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, content_hash, name FROM videos "
                "WHERE content_hash = ANY(%s)",
                [list(gt_hashes)],
            )
            meta = {r[0]: (r[1], r[2]) for r in cur.fetchall()}
    hash_to_id = {h: vid for vid, (h, _) in meta.items()}

    print(f"{'video':<8} {'rally':<10} {'gt_serve_f':>10} "
          f"{'dc@4':>7} {'dc@6':>7} {'dc@8':>7} "
          f"{'max_nonserve@6':>16}")
    print("-" * 80)

    serve_dc_values: list[float] = []
    nonserve_dc_values: list[float] = []

    with get_connection() as conn:
        for r in gt["rallies"]:
            chash = r["video_content_hash"]
            if chash not in hash_to_id:
                continue
            vid = hash_to_id[chash]
            name = meta[vid][1]
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT r.id, pt.ball_positions_json, pt.actions_json
                       FROM rallies r LEFT JOIN player_tracks pt
                       ON pt.rally_id = r.id
                       WHERE r.video_id = %s AND r.start_ms = %s""",
                    [vid, r["rally_start_ms"]],
                )
                row = cur.fetchone()
            if row is None or not row[1]:
                continue
            rid, bp_json, aj = row
            bbf = _ball_by_frame(bp_json)
            actions = (aj or {}).get("actions") or []
            pred_serve = next(
                (a for a in actions if a.get("action") == "serve"),
                None,
            )
            if pred_serve is None or pred_serve.get("isSynthetic", False):
                continue  # Only calibrate on REAL detected serves
            gt_serve_f = next(
                (a["frame"] for a in r["action_ground_truth_json"]
                 if a.get("action") == "serve"),
                None,
            )
            if gt_serve_f is None:
                continue
            # Only count as "correctly placed" if pred and gt agree within 15.
            if abs(pred_serve.get("frame", -999) - gt_serve_f) > 15:
                continue

            dcs = {w: compute_direction_change(bbf, gt_serve_f, w) for w in WINDOW_VALUES}
            # Max direction-change at non-serve frames in [0, gt_serve_f - 5].
            max_nonserve = 0.0
            for f in range(0, max(0, gt_serve_f - 5)):
                v = compute_direction_change(bbf, f, 6)
                if v > max_nonserve:
                    max_nonserve = v

            serve_dc_values.append(dcs[6])
            nonserve_dc_values.append(max_nonserve)

            print(
                f"{name:<6} {rid[:8]:<10} {gt_serve_f:>10} "
                f"{dcs[4]:>7.1f} {dcs[6]:>7.1f} {dcs[8]:>7.1f} "
                f"{max_nonserve:>16.1f}"
            )

    print("-" * 80)
    if serve_dc_values:
        print(f"Serve direction-change @ window=6: "
              f"min={min(serve_dc_values):.1f}  median={sorted(serve_dc_values)[len(serve_dc_values)//2]:.1f}  "
              f"max={max(serve_dc_values):.1f}  n={len(serve_dc_values)}")
    if nonserve_dc_values:
        print(f"Max non-serve dc in window @ window=6: "
              f"min={min(nonserve_dc_values):.1f}  median={sorted(nonserve_dc_values)[len(nonserve_dc_values)//2]:.1f}  "
              f"max={max(nonserve_dc_values):.1f}  n={len(nonserve_dc_values)}")
    # Pick threshold: max non-serve + 5deg margin, but cap at 30deg as a
    # sanity floor (typical contact threshold).
    if serve_dc_values and nonserve_dc_values:
        proposed = max(min(nonserve_dc_values) - 1, 30)
        # If serve values straddle this, dial it to the 25th percentile of serves.
        if any(s < proposed for s in serve_dc_values):
            sorted_serves = sorted(serve_dc_values)
            p25 = sorted_serves[len(sorted_serves) // 4]
            proposed = p25
        print(f"Proposed BURST_THRESHOLD: {proposed:.1f} deg")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the calibration**

```
cd /Users/mario/Personal/Projects/RallyCut/analysis
uv run python -u scripts/calibrate_serve_burst_threshold.py | tee reports/serve_burst_threshold_calibration_2026_05_10.log
```

The output ends with a "Proposed BURST_THRESHOLD" value. Use that value (rounded to 1 decimal) as the constant in Task 2's helper.

- [ ] **Step 3: Decide signal viability**

Read the per-rally lines. If the serve direction-change values cluster cleanly above the non-serve values, Signal B is viable — proceed with the proposed threshold.

If the distributions overlap heavily (e.g., min serve dc < max non-serve dc), Signal B isn't reliable — note this in the report and the helper should rely on Signal A only (drop the trajectory check). In that case, leave the placeholder constant as `BURST_THRESHOLD = 999.0` (effectively disabling Signal B) and rely on Signal A alone.

- [ ] **Step 4: Commit**

```bash
git add analysis/scripts/calibrate_serve_burst_threshold.py analysis/reports/serve_burst_threshold_calibration_2026_05_10.log
git commit -m "diag(serve): calibrate trajectory direction-change threshold for synthetic-serve placement"
```

(Use `git add -f` for the log if it's gitignored.)

---

## Task 2: Implement `pick_synthetic_serve_frame` helper

**Files:**
- Create: `analysis/rallycut/tracking/synthetic_serve_placement.py`
- Create: `analysis/tests/unit/test_synthetic_serve_placement.py`

- [ ] **Step 1: Write failing tests**

Create `analysis/tests/unit/test_synthetic_serve_placement.py`:

```python
"""Unit tests for synthetic_serve_placement.pick_synthetic_serve_frame."""

from __future__ import annotations

import numpy as np

from rallycut.tracking.ball_tracker import BallPosition
from rallycut.tracking.synthetic_serve_placement import (
    BURST_THRESHOLD,
    SEARCH_GUARD,
    SERVE_SEQ_FLOOR,
    pick_synthetic_serve_frame,
)


def _flat_ball_then_burst(burst_frame: int, n_frames: int) -> list[BallPosition]:
    """Ball stays put for `burst_frame` frames, then moves sharply.

    Generates a 90-degree direction change at burst_frame so
    compute_direction_change produces a large value there.
    """
    pos = []
    for f in range(n_frames):
        if f < burst_frame:
            pos.append(BallPosition(frame_number=f, x=0.50, y=0.50, confidence=1.0))
        else:
            # Move diagonally after the burst.
            dx = (f - burst_frame) * 0.01
            pos.append(BallPosition(frame_number=f, x=0.50 + dx, y=0.50 - dx, confidence=1.0))
    return pos


def test_returns_seq_peak_when_only_seq_signal() -> None:
    """Strong seq peak, weak ball signal -> return seq frame."""
    seq = np.zeros((7, 400))
    seq[1, 80] = 0.85  # serve-class peak at frame 80 (index 1 in seq_probs)
    ball = []  # empty -> no Signal B
    result = pick_synthetic_serve_frame(
        sequence_probs=seq,
        ball_positions=ball,
        rally_start_frame=0,
        first_contact_frame=200,
    )
    assert result == 80, result


def test_returns_burst_frame_when_only_ball_signal() -> None:
    """Weak seq, strong burst -> return burst frame."""
    seq = np.zeros((7, 400))  # all flat -> no Signal A
    ball = _flat_ball_then_burst(burst_frame=120, n_frames=200)
    result = pick_synthetic_serve_frame(
        sequence_probs=seq,
        ball_positions=ball,
        rally_start_frame=0,
        first_contact_frame=200,
    )
    assert result is not None
    assert abs(result - 120) <= 5, result


def test_returns_burst_when_both_agree() -> None:
    """Both signals strong + within 30 frames -> return burst (more precise)."""
    seq = np.zeros((7, 400))
    seq[1, 100] = 0.85
    ball = _flat_ball_then_burst(burst_frame=110, n_frames=200)
    result = pick_synthetic_serve_frame(
        sequence_probs=seq,
        ball_positions=ball,
        rally_start_frame=0,
        first_contact_frame=200,
    )
    # When agreeing, helper returns burst frame.
    assert result is not None
    assert abs(result - 110) <= 5, result


def test_returns_seq_when_signals_disagree_widely() -> None:
    """Both strong but >30 frames apart -> return seq (more reliable on class)."""
    seq = np.zeros((7, 400))
    seq[1, 60] = 0.85
    ball = _flat_ball_then_burst(burst_frame=180, n_frames=200)
    result = pick_synthetic_serve_frame(
        sequence_probs=seq,
        ball_positions=ball,
        rally_start_frame=0,
        first_contact_frame=200,
    )
    assert result == 60, result


def test_returns_none_when_neither_signal() -> None:
    """No seq peak, no ball burst -> None (caller falls back)."""
    seq = np.zeros((7, 400))
    ball = [BallPosition(frame_number=f, x=0.5, y=0.5, confidence=1.0) for f in range(200)]
    result = pick_synthetic_serve_frame(
        sequence_probs=seq,
        ball_positions=ball,
        rally_start_frame=0,
        first_contact_frame=200,
    )
    assert result is None


def test_returns_none_when_search_window_collapses() -> None:
    """rally_start ~ first_contact -> nothing to search."""
    seq = np.zeros((7, 400))
    seq[1, 50] = 0.85
    ball = []
    result = pick_synthetic_serve_frame(
        sequence_probs=seq,
        ball_positions=ball,
        rally_start_frame=100,
        first_contact_frame=100 + SEARCH_GUARD - 1,
    )
    assert result is None


def test_clamps_when_picked_frame_too_early() -> None:
    """Picked frame > 150 frames before first_contact -> clamp to first_contact - 150."""
    seq = np.zeros((7, 400))
    seq[1, 10] = 0.85  # very early peak
    ball = []
    result = pick_synthetic_serve_frame(
        sequence_probs=seq,
        ball_positions=ball,
        rally_start_frame=0,
        first_contact_frame=300,
    )
    # 300 - 150 = 150 max.
    assert result == 150, result


def test_constants_present() -> None:
    """Smoke check: the calibrated constants are exposed for monkey-patching."""
    assert isinstance(BURST_THRESHOLD, float)
    assert isinstance(SERVE_SEQ_FLOOR, float)
    assert isinstance(SEARCH_GUARD, int)
```

- [ ] **Step 2: Run failing tests**

```
cd /Users/mario/Personal/Projects/RallyCut/analysis
uv run pytest tests/unit/test_synthetic_serve_placement.py -v
```

Expected: FAIL (module not yet defined).

- [ ] **Step 3: Implement the helper**

Create `analysis/rallycut/tracking/synthetic_serve_placement.py` with this content (replace `BURST_THRESHOLD = 50.0` with the value from Task 1):

```python
"""Two-signal frame placement for synthetic serves.

Used by `_make_synthetic_serve` to land synthetic serves at the actual
serve frame instead of the placeholder `first_contact_frame - 30`. Two
independent signals:

  Signal A — MS-TCN++ serve-class peak in the search window.
  Signal B — Ball trajectory direction-change burst in the search window.

Decision matrix (see spec §"Architecture"):
  Both strong + |fA - fB| <= 30: return fB (trajectory more frame-precise).
  Both strong + disagree:        return fA (seq more reliable on class).
  Only A strong: return fA.
  Only B strong: return fB.
  Neither:       return None (caller falls back to placeholder formula).

Spec: docs/superpowers/specs/2026-05-10-synthetic-serve-placement-design.md
"""

from __future__ import annotations

import numpy as np

from rallycut.actions.trajectory_features import ACTION_TYPES
from rallycut.tracking.ball_tracker import BallPosition
from rallycut.tracking.contact_detector import compute_direction_change

# Constants (calibrated 2026-05-10 — see
# scripts/calibrate_serve_burst_threshold.py + the spec):
SERVE_SEQ_FLOOR: float = 0.50
BURST_THRESHOLD: float = 50.0  # degrees; replaced by Task 1's calibrated value
SEARCH_GUARD: int = 5  # don't pick a frame within SEARCH_GUARD of first_contact
MAX_PRESERVE_FRAMES: int = 150  # cap on how early a synthetic can be relative to first_contact
AGREEMENT_WINDOW: int = 30  # frames within which seq and burst signals "agree"

# Index of "serve" in the seq_probs array (offset by 1 for the bg row).
_SERVE_SEQ_INDEX: int = ACTION_TYPES.index("serve") + 1


def _seq_serve_peak(
    sequence_probs: np.ndarray, lo: int, hi: int
) -> tuple[int, float]:
    """Argmax frame and value in sequence_probs[SERVE, lo:hi+1]. (-1, 0.0) if invalid."""
    if (
        sequence_probs.ndim != 2
        or sequence_probs.shape[0] <= _SERVE_SEQ_INDEX
        or hi < lo
    ):
        return -1, 0.0
    t = sequence_probs.shape[1]
    hi_clip = min(t - 1, hi)
    if hi_clip < lo:
        return -1, 0.0
    slice_ = sequence_probs[_SERVE_SEQ_INDEX, lo:hi_clip + 1]
    if slice_.size == 0:
        return -1, 0.0
    rel = int(np.argmax(slice_))
    return lo + rel, float(slice_[rel])


def _trajectory_burst_peak(
    ball_positions: list[BallPosition], lo: int, hi: int
) -> tuple[int, float]:
    """Argmax direction-change frame in [lo, hi]. (-1, 0.0) if no signal."""
    if hi < lo or not ball_positions:
        return -1, 0.0
    bbf = {bp.frame_number: bp for bp in ball_positions}
    best_f, best_v = -1, 0.0
    for f in range(lo, hi + 1):
        if f not in bbf:
            continue
        v = compute_direction_change(bbf, f, 6)
        if v > best_v:
            best_f, best_v = f, v
    return best_f, best_v


def pick_synthetic_serve_frame(
    *,
    sequence_probs: np.ndarray,
    ball_positions: list[BallPosition],
    rally_start_frame: int,
    first_contact_frame: int,
) -> int | None:
    """Pick a frame for a synthetic serve using two-signal evidence.

    Returns a frame in `[rally_start, first_contact_frame - SEARCH_GUARD]`
    or None when neither signal is strong enough.
    """
    lo = max(0, rally_start_frame)
    hi = first_contact_frame - SEARCH_GUARD
    if hi < lo:
        return None

    f_seq, p_seq = _seq_serve_peak(sequence_probs, lo, hi)
    f_burst, v_burst = _trajectory_burst_peak(ball_positions, lo, hi)

    seq_strong = f_seq >= 0 and p_seq >= SERVE_SEQ_FLOOR
    burst_strong = f_burst >= 0 and v_burst >= BURST_THRESHOLD

    if seq_strong and burst_strong:
        if abs(f_seq - f_burst) <= AGREEMENT_WINDOW:
            picked = f_burst  # trajectory is more frame-precise
        else:
            picked = f_seq    # seq more reliable on action class
    elif seq_strong:
        picked = f_seq
    elif burst_strong:
        picked = f_burst
    else:
        return None

    # Sanity cap — don't pick a frame absurdly early relative to first_contact.
    earliest = first_contact_frame - MAX_PRESERVE_FRAMES
    if picked < earliest:
        picked = earliest
    return picked
```

- [ ] **Step 4: Run tests; expect PASS**

```
cd /Users/mario/Personal/Projects/RallyCut/analysis
uv run pytest tests/unit/test_synthetic_serve_placement.py -v
```

Expected: 8 passed.

- [ ] **Step 5: Lint + type-check**

```
cd /Users/mario/Personal/Projects/RallyCut/analysis
uv run mypy rallycut/tracking/synthetic_serve_placement.py
uv run ruff check rallycut/tracking/synthetic_serve_placement.py tests/unit/test_synthetic_serve_placement.py
```

Both expected: clean.

- [ ] **Step 6: Commit**

```bash
git add analysis/rallycut/tracking/synthetic_serve_placement.py analysis/tests/unit/test_synthetic_serve_placement.py
git commit -m "feat(serve): pick_synthetic_serve_frame two-signal helper"
```

---

## Task 3: Modify `_make_synthetic_serve` and plumb signals to call sites

**Files:**
- Modify: `analysis/rallycut/tracking/action_classifier.py`

The two `_make_synthetic_serve` call sites:
- `action_classifier.py:1894` (inside `classify_rally`'s main contact loop)
- `action_classifier.py:2377` (inside `_repair_serve_chain`'s Rule 0)

`classify_rally` already receives `sequence_probs`; we just need to ensure both call sites have access. `ContactSequence.ball_positions` is already populated.

- [ ] **Step 1: Update `_make_synthetic_serve` signature and body**

In `analysis/rallycut/tracking/action_classifier.py`, find the existing `_make_synthetic_serve` definition (around line 1412). Replace it with this version (preserving the existing docstring intent and adding the new behavior):

```python
def _make_synthetic_serve(
    serve_side: str,
    first_contact_frame: int,
    net_y: float,
    rally_start_frame: int | None = None,
    server_track_id: int = -1,
    sequence_probs: np.ndarray | None = None,
    ball_positions: list[BallPosition] | None = None,
) -> ClassifiedAction:
    """Create a synthetic serve action for a missed serve.

    When `sequence_probs` and `ball_positions` are provided, places the
    serve at a frame derived from two-signal evidence (MS-TCN++ serve-class
    peak + ball trajectory direction-change burst) via
    `pick_synthetic_serve_frame`. Falls back to the legacy placement
    formula (rally_start when close, else first_contact_frame - 30) when
    either signal is missing or both are weak.

    The synthetic's `confidence` reflects the placement source:
      0.40 — fully fallback (no signal-based frame).
      0.55 — fallback frame but server identified by position detection.
      0.60 — frame chosen via two-signal evidence (no server identity).
      0.70 — frame chosen via two-signal evidence + server identity.

    Args:
        serve_side: Court side of the serve ("near" or "far").
        first_contact_frame: Frame of the first detected contact.
        net_y: Net Y position.
        rally_start_frame: Frame when the rally segment starts.
        server_track_id: Track ID of the server. -1 if unknown.
        sequence_probs: Optional MS-TCN++ per-frame action probs (NUM_CLASSES, T).
        ball_positions: Optional ball positions used for the trajectory signal.

    Returns:
        A synthetic ClassifiedAction for the serve.
    """
    from rallycut.tracking.synthetic_serve_placement import (
        pick_synthetic_serve_frame,
    )

    baseline_near, baseline_far = _serve_baselines(net_y)

    serve_frame: int | None = None
    placement_confident = False
    if sequence_probs is not None and ball_positions is not None:
        serve_frame = pick_synthetic_serve_frame(
            sequence_probs=sequence_probs,
            ball_positions=ball_positions,
            rally_start_frame=rally_start_frame or 0,
            first_contact_frame=first_contact_frame,
        )
        placement_confident = serve_frame is not None

    if serve_frame is None:
        if (
            rally_start_frame is not None
            and rally_start_frame < first_contact_frame
            and (first_contact_frame - rally_start_frame) <= 90
        ):
            serve_frame = rally_start_frame
        else:
            serve_frame = max(0, first_contact_frame - 30)

    # Confidence reflects placement source AND server identity availability.
    if placement_confident:
        confidence = 0.70 if server_track_id >= 0 else 0.60
    else:
        confidence = 0.55 if server_track_id >= 0 else 0.40

    return ClassifiedAction(
        action_type=ActionType.SERVE,
        frame=serve_frame,
        ball_x=0.5,
        ball_y=baseline_near if serve_side == "near" else baseline_far,
        velocity=0.0,
        player_track_id=server_track_id,
        court_side=serve_side,
        confidence=confidence,
        is_synthetic=True,
    )
```

- [ ] **Step 2: Plumb the new args at both call sites**

Find the call site at line ~1894 (inside `classify_rally`). Change from:

```python
synth = _make_synthetic_serve(
    serve_side, contact.frame,
    contact_sequence.net_y,
    rally_start_frame=start_frame,
    server_track_id=server_pos_tid,
)
```

to:

```python
synth = _make_synthetic_serve(
    serve_side, contact.frame,
    contact_sequence.net_y,
    rally_start_frame=start_frame,
    server_track_id=server_pos_tid,
    sequence_probs=sequence_probs,
    ball_positions=contact_sequence.ball_positions,
)
```

`sequence_probs` is the parameter to `classify_rally` (verify by reading the method signature; if it isn't currently a parameter, thread it from `classify_rally_actions` per Task 4 first).

Find the call site at line ~2377 (inside `_repair_serve_chain`). Change from:

```python
synthetic = _make_synthetic_serve(
    opposite, serve.frame, net_y,
    rally_start_frame=rally_start_frame,
    server_track_id=server_track_id,
)
```

to:

```python
synthetic = _make_synthetic_serve(
    opposite, serve.frame, net_y,
    rally_start_frame=rally_start_frame,
    server_track_id=server_track_id,
    sequence_probs=sequence_probs,
    ball_positions=ball_positions,
)
```

`_repair_serve_chain` will need `sequence_probs` and `ball_positions` parameters. Add them to its signature with `= None` defaults so older callers don't break, then pass them at every internal `_repair_serve_chain` invocation.

- [ ] **Step 3: Verify imports**

At the top of `action_classifier.py`, confirm `import numpy as np` is present (it should be — used elsewhere). If not, add it. The `BallPosition` type is also already imported via `from rallycut.tracking.ball_tracker import BallPosition` (verify; add if missing).

- [ ] **Step 4: Run the existing test suite to ensure nothing regresses**

```
cd /Users/mario/Personal/Projects/RallyCut/analysis
uv run pytest tests/unit/ -q
```

Expected: 1270 passed (the count from this session's last verification). Any failure means the plumbing changed an existing behavior.

- [ ] **Step 5: Lint + type-check**

```
uv run mypy rallycut/tracking/action_classifier.py
uv run ruff check rallycut/tracking/action_classifier.py
```

Both expected: clean.

- [ ] **Step 6: Commit**

```bash
git add analysis/rallycut/tracking/action_classifier.py
git commit -m "feat(serve): two-signal frame placement in _make_synthetic_serve"
```

---

## Task 4: Verify `classify_rally_actions` already passes `sequence_probs` (and add `ball_positions` if needed)

**Files:**
- Read/possibly modify: `analysis/rallycut/tracking/action_classifier.py`

- [ ] **Step 1: Inspect the public entry point**

`classify_rally_actions` (around line 3384) is the public entry point. Confirm:
- It already accepts `sequence_probs: np.ndarray | None = None` (it does as of `apply_sequence_override` integration).
- It calls the inner `ActionClassifier.classify_rally` and threads `sequence_probs` through.

If `sequence_probs` isn't being threaded into `classify_rally`, add it.

- [ ] **Step 2: Inspect `classify_rally` for `ball_positions` access**

`classify_rally` accepts `contact_sequence: ContactSequence` which carries `ball_positions`. So `contact_sequence.ball_positions` is the source for the new arg at the call sites in Task 3. No further plumbing needed if `contact_sequence` is consistently available.

- [ ] **Step 3: Confirm CLI paths pass `sequence_probs`**

In `track_player.py` (around line 1014–1018) and `analyze.py` (around line 168–171), the CLI already passes `sequence_probs=sequence_probs` to `classify_rally_actions`. No change needed.

- [ ] **Step 4: If anything was missed in Steps 1-3**

Add the missing kwarg, run the test suite, commit:

```bash
git add analysis/rallycut/tracking/action_classifier.py
git commit -m "feat(serve): thread sequence_probs through classify_rally to synthetic-serve placement"
```

If nothing was missed (most likely), this task is a no-op verification step — proceed.

---

## Task 5: Integration test on `fb7f9c23`

**Files:**
- Modify: `analysis/tests/unit/test_synthetic_serve_placement.py` (add an integration test)

- [ ] **Step 1: Write the integration test**

Append to `analysis/tests/unit/test_synthetic_serve_placement.py`:

```python
import pytest


@pytest.mark.integration
def test_synthetic_serve_lands_at_real_frame_for_fb7f9c23() -> None:
    """End-to-end: re-running classify_rally_actions on fb7f9c23 should
    place the synthetic serve within +-15 frames of GT serve frame 154.

    This rally is the canonical "serve missed by detector, fallback fires"
    case. With the v1.1 placement, MS-TCN++ + trajectory-burst should
    catch the actual serve.
    """
    from rallycut.evaluation.tracking.db import get_connection
    from rallycut.tracking.action_classifier import classify_rally_actions
    from rallycut.tracking.ball_tracker import BallPosition
    from rallycut.tracking.contact_detector import (
        ContactDetectionConfig,
        detect_contacts,
    )
    from rallycut.tracking.player_tracker import PlayerPosition
    from rallycut.tracking.sequence_action_runtime import get_sequence_probs

    rally_id = "fb7f9c23-3544-48bd-910d-10a8f12fd594"

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT pt.fps, pt.frame_count, pt.court_split_y,
                          pt.ball_positions_json, pt.positions_json,
                          pt.actions_json, pt.primary_track_ids
                   FROM player_tracks pt JOIN rallies r ON pt.rally_id = r.id
                   WHERE r.id = %s""",
                [rally_id],
            )
            row = cur.fetchone()
    assert row is not None, "fixture rally fb7f9c23 missing from DB"
    fps, frame_count, court_split_y, bp_json, pp_json, aj, primary_raw = row

    bp = [
        BallPosition(
            frame_number=int(b.get("frameNumber", 0)),
            x=float(b.get("x", 0.0)),
            y=float(b.get("y", 0.0)),
            confidence=float(b.get("confidence", 0.0)),
        )
        for b in (bp_json or [])
        if isinstance(b, dict)
    ]
    pp = [
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
    ta_str = (aj or {}).get("teamAssignments", {}) or {}
    ta_int: dict[int, int] = {}
    for k, v in ta_str.items():
        if v == "A":
            ta_int[int(k)] = 0
        elif v == "B":
            ta_int[int(k)] = 1

    seq_probs = get_sequence_probs(
        bp, pp, court_split_y, frame_count or 0, ta_int, calibrator=None,
    )
    if seq_probs is None:
        pytest.skip("MS-TCN++ weights unavailable in this environment")

    contact_seq = detect_contacts(
        ball_positions=bp, player_positions=pp,
        config=ContactDetectionConfig(),
        net_y=court_split_y, frame_count=frame_count or None,
        team_assignments=ta_int, court_calibrator=None,
        sequence_probs=seq_probs,
        primary_track_ids=list(primary_raw or []) or None,
    )

    rally_actions = classify_rally_actions(
        contact_seq,
        team_assignments=ta_int,
        calibrator=None,
        sequence_probs=seq_probs,
    )
    serves = [a for a in rally_actions.actions if a.action_type.value == "serve"]
    assert serves, "No serve action in classified rally"
    serve = serves[0]
    # GT serve at frame 154; require the synthetic to land within +-15.
    assert abs(serve.frame - 154) <= 15, (
        f"Synthetic serve at frame {serve.frame}; GT 154 (off by "
        f"{serve.frame - 154})"
    )
    assert serve.is_synthetic, "Expected synthetic (since detector missed it)"
```

- [ ] **Step 2: Run the test**

```
cd /Users/mario/Personal/Projects/RallyCut/analysis
uv run pytest tests/unit/test_synthetic_serve_placement.py::test_synthetic_serve_lands_at_real_frame_for_fb7f9c23 -v
```

Expected: PASS. If it fails because the synthetic frame is not within ±15 of 154:
- Read the actual frame from the assertion message.
- Investigate whether Signal A or Signal B (or neither) fired.
- If neither, the helper returned None and the fallback formula ran — the calibration may need re-tuning OR fb7f9c23 is genuinely a hard case (the missing serve plus missing receive scenario described in the spec). Report DONE_WITH_CONCERNS with the specific failure detail.

- [ ] **Step 3: Commit**

```bash
git add analysis/tests/unit/test_synthetic_serve_placement.py
git commit -m "test(serve): integration test for fb7f9c23 synthetic-serve placement"
```

---

## Task 6: Panel measurement (pre/post comparison)

**Files:**
- Create: `analysis/scripts/measure_synthetic_serve_placement_panel.py`

- [ ] **Step 1: Write the measurement script**

Create `analysis/scripts/measure_synthetic_serve_placement_panel.py`:

```python
"""Panel measurement for synthetic-serve placement (v1.1).

For each panel rally with GT serve label:
  - Compare current pred serve frame vs GT (BASE).
  - Re-run classify_rally_actions with v1.1 placement (POST).
  - Print per-rally diff + aggregate.

Counts:
  - Synthetic serves: hit (within +-15 of GT) vs miss.
  - Real (detected) serves: hit vs miss (sanity check — must not regress).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rallycut.evaluation.tracking.db import get_connection
from rallycut.tracking.action_classifier import classify_rally_actions
from rallycut.tracking.ball_tracker import BallPosition
from rallycut.tracking.contact_detector import (
    ContactDetectionConfig,
    detect_contacts,
)
from rallycut.tracking.player_tracker import PlayerPosition
from rallycut.tracking.sequence_action_runtime import get_sequence_probs

GT_PATH = Path("training_datasets/beach_v11/action_ground_truth.json")
# Use every video that has action GT — frame and action-type labels are
# clean; only player attribution is known-noisy and we don't compare it
# here. The 5-video "panel" was just a quick subset used during the prior
# diagnostic session.
HIT_TOLERANCE = 15


def _ball_positions_from_json(bp_json: Any) -> list[BallPosition]:
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


def _player_positions_from_json(pp_json: Any) -> list[PlayerPosition]:
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


def main() -> None:
    with open(GT_PATH) as f:
        gt = json.load(f)

    # Use every video referenced in the GT file (66 videos / ~340 rallies)
    # — frame and action-type labels are clean across the whole pool;
    # only player attribution is known-noisy and we don't compare it here.
    gt_hashes = {r["video_content_hash"] for r in gt["rallies"]}
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, content_hash, name FROM videos "
                "WHERE content_hash = ANY(%s)",
                [list(gt_hashes)],
            )
            meta = {r[0]: (r[1], r[2]) for r in cur.fetchall()}
    hash_to_id = {h: vid for vid, (h, _) in meta.items()}

    print(f"{'video':<8} {'rally':<10} {'kind':<10} "
          f"{'base_f':>7} {'post_f':>7} {'gt_f':>5} "
          f"{'base_diff':>10} {'post_diff':>10}  {'verdict':<25}")
    print("-" * 100)

    counts = {"synth_hit_base": 0, "synth_hit_post": 0,
              "synth_total": 0,
              "real_hit_base": 0, "real_hit_post": 0,
              "real_total": 0}

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
                       FROM rallies r LEFT JOIN player_tracks pt
                       ON pt.rally_id = r.id
                       WHERE r.video_id = %s AND r.start_ms = %s""",
                    [vid, r["rally_start_ms"]],
                )
                row = cur.fetchone()
            if row is None or not row[1]:
                continue
            rid, fps, fcount, csy, bp_json, pp_json, aj, primary_raw = row

            actions = (aj or {}).get("actions") or []
            pred_serve = next(
                (a for a in actions if a.get("action") == "serve"), None,
            )
            gt_serve_f = next(
                (a["frame"] for a in r["action_ground_truth_json"]
                 if a.get("action") == "serve"),
                None,
            )
            if pred_serve is None or gt_serve_f is None:
                continue
            base_f = pred_serve.get("frame", -1)
            base_diff = base_f - gt_serve_f
            kind = "synth" if pred_serve.get("isSynthetic", False) else "real"

            # Re-run with v1.1 placement.
            ta_str = (aj or {}).get("teamAssignments", {}) or {}
            ta_int = {int(k): (0 if v == "A" else 1) for k, v in ta_str.items()
                      if v in ("A", "B")}
            bp = _ball_positions_from_json(bp_json)
            pp = _player_positions_from_json(pp_json)
            seq_probs = get_sequence_probs(
                bp, pp, csy, fcount or 0, ta_int, calibrator=None,
            )
            if seq_probs is None:
                post_f = base_f
            else:
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
                post_serve = next(
                    (a for a in ra.actions if a.action_type.value == "serve"),
                    None,
                )
                post_f = post_serve.frame if post_serve else -1
            post_diff = post_f - gt_serve_f

            base_hit = abs(base_diff) <= HIT_TOLERANCE
            post_hit = abs(post_diff) <= HIT_TOLERANCE
            if kind == "synth":
                counts["synth_total"] += 1
                if base_hit:
                    counts["synth_hit_base"] += 1
                if post_hit:
                    counts["synth_hit_post"] += 1
            else:
                counts["real_total"] += 1
                if base_hit:
                    counts["real_hit_base"] += 1
                if post_hit:
                    counts["real_hit_post"] += 1

            verdict = (
                "no change"
                if base_hit == post_hit
                else ("FIXED" if not base_hit and post_hit else "REGRESSION")
            )
            print(
                f"{name:<6} {rid[:8]:<10} {kind:<10} "
                f"{base_f:>7} {post_f:>7} {gt_serve_f:>5} "
                f"{base_diff:>+10d} {post_diff:>+10d}  {verdict:<25}"
            )

    print("-" * 100)
    print(
        f"Synthetic: {counts['synth_hit_base']}/{counts['synth_total']} "
        f"-> {counts['synth_hit_post']}/{counts['synth_total']} hits"
    )
    print(
        f"Real:      {counts['real_hit_base']}/{counts['real_total']} "
        f"-> {counts['real_hit_post']}/{counts['real_total']} hits"
    )
    print(
        f"Total fixed: {counts['synth_hit_post'] - counts['synth_hit_base']:+d} "
        f"synth, {counts['real_hit_post'] - counts['real_hit_base']:+d} real"
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the measurement**

```
cd /Users/mario/Personal/Projects/RallyCut/analysis
uv run python -u scripts/measure_synthetic_serve_placement_panel.py | tee reports/synthetic_serve_placement_panel_2026_05_10.log
```

- [ ] **Step 3: Apply ship-gate** (fleet-scale, not panel)

Per the spec §"Done criteria" (measured across the full 66-video GT pool):

- ≥ **30 of 51** currently-mis-placed synthetic serves now hit (within ±15 of GT). The summary line `Synthetic: 44/95 -> Y/95` must show `Y >= 74` (44 + 30).
- ≤ **5 regressions** on the 194 currently-correctly-placed real serves. The summary line `Real: 194/218 -> Z/218` must show `Z >= 189` (194 − 5).

If ship gate met → proceed to Task 7. If not → report DONE_WITH_CONCERNS with the actual numbers and per-rally diagnostic; do NOT commit downstream changes.

- [ ] **Step 4: Commit script + log**

```bash
git add analysis/scripts/measure_synthetic_serve_placement_panel.py
git add -f analysis/reports/synthetic_serve_placement_panel_2026_05_10.log
git commit -m "feat(serve): panel measurement script + ship-gate evidence for v1.1"
```

---

## Task 7: Memory + close (gated on Task 6 passing)

- [ ] **Step 1: Memory entry for the workstream**

Create `~/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/synthetic_serve_placement_v11_2026_05_10.md` with:

```markdown
---
name: Synthetic-serve frame placement v1.1 (Sub-2.B replacement)
description: Two-signal placement (MS-TCN++ peak + ball direction-change) for _make_synthetic_serve. Panel synth hit rate 40% -> [POST]%. Replaces abandoned coherence-driven-recovery v1.
type: project
---

# Synthetic-Serve Frame Placement v1.1 (shipped 2026-05-10)

**Why:** original 'coherence-driven contact recovery v1' (now removed) recovered 0 panel contacts because gap-windows were too narrow. Per-FN diagnostic + visual-check pivot revealed the real bottleneck: synthetic serves placed at `first_contact_frame - 30` (placeholder formula) — 60% mis-placed on the panel.

**What:** new helper `pick_synthetic_serve_frame` in `rallycut/tracking/synthetic_serve_placement.py`. `_make_synthetic_serve` calls it when `sequence_probs` + `ball_positions` available (always-on in production CLI paths). Two signals:
- Signal A — MS-TCN++ serve-class peak with floor 0.50 in `[rally_start, first_contact - 5]`.
- Signal B — ball direction-change >= [calibrated value] in same window.
- Decision matrix: agreeing -> burst frame (more precise); disagreeing -> seq frame (more reliable on class); single signal -> that frame; neither -> None (legacy fallback fires).

**Fleet measurement** (66 videos, 313 rallies with both GT and pred serve, 2026-05-10):
- Synthetic serves: 44/95 hit -> [POST]/95 hit ([+N] fixed).
- Real serves: 194/218 -> [POST]/218 (≤5 regressions allowed).

**Spec:** docs/superpowers/specs/2026-05-10-synthetic-serve-placement-design.md
**Plan:** docs/superpowers/plans/2026-05-10-synthetic-serve-placement.md
**Commits:** [list relevant SHAs from this session]

**Caveats:**
- BURST_THRESHOLD calibrated on 14 panel real serves; may need re-tuning for fleet diversity.
- Some FNs remain unrecoverable (genuine off-screen / ball-warmup serves) — see spec §"Out of scope" Pattern A2.
```

Replace `[BASE]/[POST]/[+N]/[POST]%` with the actual numbers from the Task 6 measurement.

- [ ] **Step 2: Update MEMORY.md index**

Add a one-line entry under "Current workstreams" in `~/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/MEMORY.md`:

```markdown
- [SHIPPED] [**Synthetic-serve placement v1.1 2026-05-10**](synthetic_serve_placement_v11_2026_05_10.md) — Two-signal frame placement (MS-TCN++ peak + ball trajectory burst) for _make_synthetic_serve. Panel synth hit rate +X. Replaces abandoned coherence-driven-recovery v1.
```

- [ ] **Step 3: Final regression check**

```
cd /Users/mario/Personal/Projects/RallyCut/analysis
uv run pytest tests/unit/ -q
uv run mypy rallycut/
uv run ruff check rallycut/
```

All three: clean.

- [ ] **Step 4: Final close**

No commit needed for memory files (they live outside the repo). Workstream is closed.

---

## Self-Review Checklist

After implementing, verify:

- [ ] Spec §"In scope" items are all covered: helper, _make_synthetic_serve change, plumbing, unit tests, integration test, panel measurement script.
- [ ] Spec §"Out of scope" items NOT smuggled in (no attribution changes, no MS-TCN++ retraining, no coherence-rule changes, no recovery-of-mid-rally-FNs).
- [ ] BURST_THRESHOLD constant in `synthetic_serve_placement.py` is the calibrated value from Task 1 (not the placeholder `50.0`).
- [ ] All tests pass; mypy --strict + ruff clean.
- [ ] Ship gate met before Task 7 fires (4+ synth fixes, 0 real regressions).
- [ ] No DB writes anywhere — synthetic serves are recomputed on every classify_rally_actions call.
