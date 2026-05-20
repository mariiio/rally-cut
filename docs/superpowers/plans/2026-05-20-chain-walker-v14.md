# Chain-Walker v14 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `_compute_expected_teams` (the team-chain walker in `action_classifier.py:2968`) to support two flag-gated interventions — BLOCK conditional flip (B.1) and ball-trajectory verifier (B.2) — both default OFF in v14 so behavior is byte-identical to v13 until A/B picks a winner.

**Architecture:** Replace the inline `_NET_CROSSING_ACTIONS` check inside the walker loop with a `_possession_flips_after(action, next_action, contacts, config)` decision function. New helper `_contact_side_at(contacts, frame)` looks up `Contact.court_side` by ±3-frame proximity. Two env flags (`WALKER_BLOCK_CONDITIONAL`, `WALKER_BALL_TRAJECTORY_VERIFIER`) read once at module load; pass through a frozen `ChainWalkerConfig` dataclass into the walker. Validation is two stages: a cheap walker-decision replay against the existing 1265-pair GT dataset, then an end-to-end 2×2 A/B on trusted-31.

**Tech Stack:** Python 3.11+, `pytest`, `psycopg`, existing `rallycut.tracking.action_classifier` + `contact_detector` modules.

**Spec reference:** `docs/superpowers/specs/2026-05-20-chain-walker-v14-design.md` (commit `d4e5232d`).

**Background (read before starting):**
- `analysis/rallycut/tracking/action_classifier.py:_compute_expected_teams` (line 2968-3015) currently has the inline rule `if action.action_type in _NET_CROSSING_ACTIONS: current_team = 1 - current_team`. `_NET_CROSSING_ACTIONS = {SERVE, ATTACK}` at line 2774.
- The walker is called from `reattribute_players` (line 3162) which already has `contact_by_frame` (dict[int, Contact]) in scope; we'll pass `contacts: list[Contact] | None = None` through to keep the signature backward-compatible.
- `Contact.court_side` is `"near" | "far" | "unknown"` (per `contact_detector.py:401`).
- The 1265-pair GT dataset lives in `analysis/reports/gt_net_crossings_2026_05_20/events.csv` (commit `e41bfa1c`) — Stage-1 replay reads from there.
- `Contact.is_synthetic` or equivalent: synthetic contacts have ball positions interpolated. The verifier degrades to rule when either contact is synthetic (per the spec's risk section).
- Pre-commit hook (`.claude/hooks/pre-commit-check.sh`) requires `ACTION_PIPELINE_VERSION` bump on changes to `action_classifier.py` unless `[no-version-bump]` is in the commit message.

---

## File Structure

| Path | Status | Responsibility |
|---|---|---|
| `analysis/rallycut/tracking/action_classifier.py` (lines `2960-3015` + flag constants near `_NET_CROSSING_ACTIONS` + `ACTION_PIPELINE_VERSION`) | MODIFY | Add `ChainWalkerConfig`, `_contact_side_at`, `_possession_flips_after` helpers; refactor `_compute_expected_teams` to use them; update `reattribute_players` to pass `contacts`; bump `ACTION_PIPELINE_VERSION` to `"v14"` with history comment |
| `analysis/tests/unit/test_chain_walker_v14.py` | CREATE | Unit tests for `_possession_flips_after` covering v13-default + B.1 + B.2 + synthetic-contact degradation |
| `analysis/scripts/measure_walker_accuracy_2026_05_20.py` | CREATE | Stage-1 replay: replays each of 4 flag configs against `events.csv`, reports walker-decision accuracy table |
| `analysis/scripts/sub_lever_2_ab_2026_05_20.sh` | CREATE | Stage-2 end-to-end A/B driver (4 redetect cycles + 4 measurements + compare) |

---

## Task 1: Walker refactor + flag wiring (v14)

**Files:**
- Modify: `analysis/rallycut/tracking/action_classifier.py`
- Test: `analysis/tests/unit/test_chain_walker_v14.py` (CREATE)

### Step 1: Write the failing tests

Create `/Users/mario/Personal/Projects/RallyCut/analysis/tests/unit/test_chain_walker_v14.py`:

```python
"""Tests for v14 chain-walker decision function.

Covers _possession_flips_after across all (flag, action_type) cases plus
synthetic-contact degradation behavior.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from rallycut.tracking.action_classifier import (
    ChainWalkerConfig,
    _contact_side_at,
    _possession_flips_after,
)
from rallycut.tracking.contact_detector import Contact


def _make_contact(frame: int, side: str, is_synthetic: bool = False) -> Contact:
    return Contact(
        frame=frame,
        ball_x=0.5,
        ball_y=0.5 if side == "near" else 0.3,
        velocity=0.0,
        court_side=side,
        is_at_net=False,
        confidence=0.8,
        is_validated=True,
        player_track_id=1,
        arc_fit_residual=0.0,
        player_distance=0.05,
        is_synthetic=is_synthetic,
    )


@dataclass
class _FakeAction:
    action_type: str  # use string for test simplicity; production uses ActionType enum
    frame: int
    is_synthetic: bool = False


# --- v13-equivalent behavior (both flags off) ----------------------------

def test_v13_default_serve_flips():
    cfg = ChainWalkerConfig(block_conditional=False, ball_trajectory_verifier=False)
    a = _FakeAction("serve", 100)
    nxt = _FakeAction("receive", 140)
    assert _possession_flips_after(a, nxt, [], cfg) is True


def test_v13_default_attack_flips():
    cfg = ChainWalkerConfig(block_conditional=False, ball_trajectory_verifier=False)
    a = _FakeAction("attack", 200)
    nxt = _FakeAction("dig", 220)
    assert _possession_flips_after(a, nxt, [], cfg) is True


def test_v13_default_block_stays():
    cfg = ChainWalkerConfig(block_conditional=False, ball_trajectory_verifier=False)
    a = _FakeAction("block", 300)
    nxt = _FakeAction("attack", 320)
    assert _possession_flips_after(a, nxt, [], cfg) is False


def test_v13_default_set_stays():
    cfg = ChainWalkerConfig(block_conditional=False, ball_trajectory_verifier=False)
    a = _FakeAction("set", 400)
    nxt = _FakeAction("attack", 420)
    assert _possession_flips_after(a, nxt, [], cfg) is False


# --- B.1 BLOCK conditional ------------------------------------------------

def test_b1_block_flips_when_next_side_differs():
    cfg = ChainWalkerConfig(block_conditional=True, ball_trajectory_verifier=False)
    a = _FakeAction("block", 100)
    nxt = _FakeAction("dig", 130)
    contacts = [_make_contact(100, "near"), _make_contact(130, "far")]
    assert _possession_flips_after(a, nxt, contacts, cfg) is True


def test_b1_block_stays_when_next_side_same():
    cfg = ChainWalkerConfig(block_conditional=True, ball_trajectory_verifier=False)
    a = _FakeAction("block", 100)
    nxt = _FakeAction("attack", 130)
    contacts = [_make_contact(100, "near"), _make_contact(130, "near")]
    assert _possession_flips_after(a, nxt, contacts, cfg) is False


def test_b1_block_falls_back_to_v13_when_contacts_missing():
    cfg = ChainWalkerConfig(block_conditional=True, ball_trajectory_verifier=False)
    a = _FakeAction("block", 100)
    nxt = _FakeAction("dig", 130)
    # No contacts at action frames -> _contact_side_at returns None -> fall back to v13 rule
    assert _possession_flips_after(a, nxt, [], cfg) is False  # v13 rule: block stays


# --- B.2 ball-trajectory verifier ----------------------------------------

def test_b2_overrides_rule_when_ball_crossed():
    """SET rule says stay, but contact court_sides differ -> verifier flips."""
    cfg = ChainWalkerConfig(block_conditional=False, ball_trajectory_verifier=True)
    a = _FakeAction("set", 100)
    nxt = _FakeAction("attack", 130)
    contacts = [_make_contact(100, "near"), _make_contact(130, "far")]
    assert _possession_flips_after(a, nxt, contacts, cfg) is True


def test_b2_overrides_rule_when_ball_stayed():
    """SERVE rule says flip, but contact court_sides match -> verifier stays."""
    cfg = ChainWalkerConfig(block_conditional=False, ball_trajectory_verifier=True)
    a = _FakeAction("serve", 100)
    nxt = _FakeAction("dig", 130)
    contacts = [_make_contact(100, "near"), _make_contact(130, "near")]
    assert _possession_flips_after(a, nxt, contacts, cfg) is False


def test_b2_degrades_to_rule_when_synthetic_contact():
    """Synthetic contact -> verifier returns None -> falls back to rule."""
    cfg = ChainWalkerConfig(block_conditional=False, ball_trajectory_verifier=True)
    a = _FakeAction("attack", 100)
    nxt = _FakeAction("dig", 130)
    contacts = [
        _make_contact(100, "near", is_synthetic=True),
        _make_contact(130, "far"),
    ]
    # Verifier sees a synthetic contact, declines; v13 rule applies (attack flips)
    assert _possession_flips_after(a, nxt, contacts, cfg) is True


def test_b2_degrades_to_rule_when_contact_side_unknown():
    cfg = ChainWalkerConfig(block_conditional=False, ball_trajectory_verifier=True)
    a = _FakeAction("attack", 100)
    nxt = _FakeAction("dig", 130)
    contacts = [_make_contact(100, "unknown"), _make_contact(130, "far")]
    # Verifier declines on unknown; v13 rule applies (attack flips)
    assert _possession_flips_after(a, nxt, contacts, cfg) is True


# --- _contact_side_at helper ---------------------------------------------

def test_contact_side_at_returns_side_within_tolerance():
    contacts = [_make_contact(100, "near"), _make_contact(150, "far")]
    assert _contact_side_at(contacts, 101) == "near"
    assert _contact_side_at(contacts, 152) == "far"


def test_contact_side_at_returns_none_outside_tolerance():
    contacts = [_make_contact(100, "near")]
    # ±3 default tolerance; 105 is too far
    assert _contact_side_at(contacts, 105) is None


def test_contact_side_at_returns_none_when_side_unknown():
    contacts = [_make_contact(100, "unknown")]
    assert _contact_side_at(contacts, 100) is None


def test_contact_side_at_skips_synthetic_contacts():
    contacts = [
        _make_contact(100, "near", is_synthetic=True),
        _make_contact(102, "far"),
    ]
    # Within ±3 of 100, synthetic skipped, falls through to next contact ("far" at 102)
    assert _contact_side_at(contacts, 100) == "far"
```

Note: tests use string `action_type` because the helpers will be written
to accept either the `ActionType` enum (production) or its `.value`
attribute. If the actual implementation strictly requires the enum, adapt
the helper to convert; the test layer will then need `ActionType` imports.

### Step 2: Run tests to verify they fail

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && uv run pytest tests/unit/test_chain_walker_v14.py -v
```

Expected: ImportError (`ChainWalkerConfig`, `_contact_side_at`, `_possession_flips_after` not defined yet).

### Step 3: Implement the helpers

In `/Users/mario/Personal/Projects/RallyCut/analysis/rallycut/tracking/action_classifier.py`, find the existing `_NET_CROSSING_ACTIONS = {ActionType.SERVE, ActionType.ATTACK}` line (line 2774). Immediately after that block, add:

```python
# v14 chain-walker config + helpers (Sub-lever 2 / chain-quality rewrite).
# Both flags default OFF; when both are off, _possession_flips_after is
# byte-identical to the v13 inline rule (action.action_type in
# _NET_CROSSING_ACTIONS).
import os
from dataclasses import dataclass

_WALKER_BLOCK_CONDITIONAL = (
    os.environ.get("WALKER_BLOCK_CONDITIONAL", "0").lower() in ("1", "true", "yes")
)
_WALKER_BALL_TRAJECTORY_VERIFIER = (
    os.environ.get("WALKER_BALL_TRAJECTORY_VERIFIER", "0").lower() in ("1", "true", "yes")
)


@dataclass(frozen=True)
class ChainWalkerConfig:
    """Knobs for _compute_expected_teams (v14)."""
    block_conditional: bool = False
    ball_trajectory_verifier: bool = False

    @classmethod
    def from_env(cls) -> "ChainWalkerConfig":
        return cls(
            block_conditional=_WALKER_BLOCK_CONDITIONAL,
            ball_trajectory_verifier=_WALKER_BALL_TRAJECTORY_VERIFIER,
        )


def _contact_side_at(
    contacts: list[Any],
    frame: int,
    tol: int = 3,
) -> str | None:
    """Return Contact.court_side (`"near"`/`"far"`) for the contact closest to
    `frame` within ±tol frames, skipping synthetic + unknown-side contacts.
    Returns None when no eligible contact is in range.
    """
    if not contacts:
        return None
    best = None
    best_delta = tol + 1
    for c in contacts:
        if getattr(c, "is_synthetic", False):
            continue
        side = getattr(c, "court_side", "unknown")
        if side not in ("near", "far"):
            continue
        d = abs(int(getattr(c, "frame", -10**9)) - int(frame))
        if d < best_delta:
            best_delta = d
            best = side
    return best


def _possession_flips_after(
    action: Any,
    next_action: Any | None,
    contacts: list[Any],
    config: ChainWalkerConfig,
) -> bool:
    """Whether possession flips after `action`.

    v13 baseline (both flags OFF): rule_says_flip = action.action_type in
    _NET_CROSSING_ACTIONS.

    B.1 (config.block_conditional): when action is BLOCK, use
    next_contact.court_side != action_contact.court_side as the rule.
    Falls back to v13 (block stays) when contacts are unavailable.

    B.2 (config.ball_trajectory_verifier): when both contact sides are
    available (non-synthetic, known side), use physical court_side change
    as the source of truth, overriding the rule.
    """
    # Normalise action_type to string (accept enum or raw string)
    a_type = getattr(action.action_type, "value", action.action_type)
    a_type = str(a_type).lower()

    rule_says_flip = a_type in ("serve", "attack")

    # B.1
    if config.block_conditional and a_type == "block":
        curr_side = _contact_side_at(contacts, action.frame)
        next_side = (
            _contact_side_at(contacts, next_action.frame)
            if next_action is not None else None
        )
        if curr_side and next_side:
            rule_says_flip = (curr_side != next_side)

    # B.2 — physical truth overrides rule
    if config.ball_trajectory_verifier and next_action is not None:
        curr_side = _contact_side_at(contacts, action.frame)
        next_side = _contact_side_at(contacts, next_action.frame)
        if curr_side and next_side:
            return curr_side != next_side

    return rule_says_flip
```

Then add `from typing import Any` at the top of the file if not already imported (likely already imported; verify).

### Step 4: Refactor `_compute_expected_teams` to use the decision function

Locate `_compute_expected_teams` (around line 2968). Replace the loop:

```python
    current_team = serve_team
    for i, action in enumerate(actions):
        if action.action_type == ActionType.UNKNOWN:
            continue
        if action.is_synthetic:
            if action.action_type == ActionType.SERVE:
                expected[i] = serve_team
                current_team = 1 - serve_team
            continue

        expected[i] = current_team

        # After net-crossing actions, flip to opposite team
        if action.action_type in _NET_CROSSING_ACTIONS:
            current_team = 1 - current_team

    return expected
```

with the decision-function variant. Update the function signature to
accept an optional `contacts` parameter:

```python
def _compute_expected_teams(
    actions: list[ClassifiedAction],
    team_assignments: dict[int, int],
    contacts: list[Any] | None = None,
    config: ChainWalkerConfig | None = None,
) -> list[int | None]:
    """[existing docstring]"""
    expected: list[int | None] = [None] * len(actions)
    if config is None:
        config = ChainWalkerConfig.from_env()
    if contacts is None:
        contacts = []

    serve_team: int | None = None
    for a in actions:
        if a.action_type == ActionType.SERVE and a.player_track_id >= 0:
            serve_team = team_assignments.get(a.player_track_id)
            break

    if serve_team is None:
        return expected

    current_team = serve_team
    for i, action in enumerate(actions):
        if action.action_type == ActionType.UNKNOWN:
            continue
        if action.is_synthetic:
            if action.action_type == ActionType.SERVE:
                expected[i] = serve_team
                current_team = 1 - serve_team
            continue

        expected[i] = current_team

        next_action = actions[i + 1] if i + 1 < len(actions) else None
        if _possession_flips_after(action, next_action, contacts, config):
            current_team = 1 - current_team

    return expected
```

### Step 5: Update the call site in `reattribute_players`

Around line 3162, the call is:

```python
    expected_teams = _compute_expected_teams(actions, team_assignments)
```

Update to pass contacts. The `reattribute_players` function already has
`contacts` in scope (verify by reading line 3108 — the parameter list).
If the parameter exists, change to:

```python
    expected_teams = _compute_expected_teams(actions, team_assignments, contacts=contacts)
```

If `contacts` is not directly available as a list, derive it from
`contact_by_frame.values()`:

```python
    expected_teams = _compute_expected_teams(
        actions, team_assignments,
        contacts=list(contact_by_frame.values()),
    )
```

### Step 6: Bump `ACTION_PIPELINE_VERSION` to v14

Find the constant (around line 176-189). Add a v14 history comment above
the constant and update:

```python
# v14 (2026-05-20): chain-walker decision-function refactor (Sub-lever 2).
#                   Two new env flags (WALKER_BLOCK_CONDITIONAL,
#                   WALKER_BALL_TRAJECTORY_VERIFIER), both default OFF.
#                   Behavior byte-identical to v13 when both flags are
#                   unset. Targets the 64 missed flips + 20 over-flips
#                   identified by probe_gt_net_crossings_2026_05_20
#                   (commit e41bfa1c). Spec:
#                   docs/superpowers/specs/2026-05-20-chain-walker-v14-design.md
ACTION_PIPELINE_VERSION = "v14"
```

### Step 7: Run all the new unit tests

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && uv run pytest tests/unit/test_chain_walker_v14.py -v
```

Expected: all tests pass (count depends on how many you wrote; the spec
example has 13 tests). If `Contact(...)` instantiation fails because of
missing required fields, adjust `_make_contact` to match the actual
Contact dataclass.

### Step 8: Run the broader scorer + classifier test suite to catch regressions

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && uv run pytest tests/unit/ -v -k "scorer or classify or attribution or chain or cascade"
```

Expected: all matching tests pass (plus the 13 new chain_walker_v14 tests). If a pre-existing test fails, investigate before committing — the v13-default-equivalent behavior MUST hold.

### Step 9: Verify ruff + mypy clean

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && uv run ruff check rallycut/tracking/action_classifier.py tests/unit/test_chain_walker_v14.py && uv run mypy rallycut/tracking/action_classifier.py
```

Expected: zero findings.

### Step 10: Commit

```bash
cd /Users/mario/Personal/Projects/RallyCut && git add analysis/rallycut/tracking/action_classifier.py analysis/tests/unit/test_chain_walker_v14.py && git commit -m "$(cat <<'EOF'
feat(walker): chain-walker v14 — flag-gated BLOCK conditional + ball-trajectory verifier

Refactor _compute_expected_teams to use a decision function that
supports two flag-gated interventions:
  WALKER_BLOCK_CONDITIONAL — when on, BLOCK flips iff next contact's
    court_side differs from this contact's. Empirically classifies
    17/17 prev=block events correctly on trusted-32 GT.
  WALKER_BALL_TRAJECTORY_VERIFIER — when on, contact-court-side
    transitions override the rule-based decision. Robust to upstream
    action mis-classification (the 51 set->attack / dig->attack /
    receive-over cases identified by the GT-pair probe).

Both flags default OFF in v14 -> byte-identical to v13 in production.
Winning combination flipped to default ON in v15 after A/B (spec at
commit d4e5232d, plan at TBD).

Bumps ACTION_PIPELINE_VERSION to v14.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Stage-1 walker accuracy replay script

**Files:**
- Create: `analysis/scripts/measure_walker_accuracy_2026_05_20.py`

### Step 1: Write the script

Create `/Users/mario/Personal/Projects/RallyCut/analysis/scripts/measure_walker_accuracy_2026_05_20.py`:

```python
#!/usr/bin/env python3
"""Stage-1 walker-decision accuracy replay (no scorer interaction).

Reads the 1265-pair GT dataset from probe_gt_net_crossings_2026_05_20
(events.csv) and replays each of 4 ChainWalkerConfig combinations
against it:

  cfg_00: both flags OFF (= v13 baseline)
  cfg_10: B.1 ON, B.2 OFF
  cfg_01: B.1 OFF, B.2 ON
  cfg_11: both ON

For each config, recomputes the walker's flip decision per pair and
tallies correct_flip / missed_flip / correct_stay / over_flip. Compares
against v13 baseline.

Output:
  reports/walker_accuracy_2026_05_20/summary.json
  reports/walker_accuracy_2026_05_20/summary.md
  Console: side-by-side table.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ANALYSIS_DIR))

from rallycut.tracking.action_classifier import (  # noqa: E402
    ChainWalkerConfig,
    _possession_flips_after,
)

IN_CSV = ANALYSIS_DIR / "reports" / "gt_net_crossings_2026_05_20" / "events.csv"
OUT_DIR = ANALYSIS_DIR / "reports" / "walker_accuracy_2026_05_20"
OUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class _Action:
    action_type: str
    frame: int
    is_synthetic: bool = False


@dataclass
class _Contact:
    frame: int
    court_side: str
    is_synthetic: bool = False


def _row_to_objs(row: dict) -> tuple[_Action, _Action, list[_Contact]]:
    """Reconstruct (curr_action, next_action, contacts) from an events.csv row.

    The events.csv has prev/curr action + frame + pid + team but NOT
    court_side per contact. We infer court_side from team_prev/team_curr +
    a fixed "team A = near" convention.

    NOTE: this is the diagnostic-time convention. Actual production
    court_side comes from Contact.court_side (ball-y-position-derived),
    which may differ from team-derived in edge cases. For Stage-1 replay
    purposes, this approximation correctly tracks whether contacts are
    on the same side or different sides — which is all the verifier
    needs.
    """
    # Map team -> side using the convention that team A is the near court
    # (only matters that prev and curr get consistent sides).
    side_of = {"A": "near", "B": "far"}
    prev_side = side_of.get(row["prev_team"], "unknown")
    curr_side = side_of.get(row["curr_team"], "unknown")
    a = _Action(action_type=row["prev_action"], frame=int(row["prev_frame"]))
    nxt = _Action(action_type=row["curr_action"], frame=int(row["curr_frame"]))
    contacts = [
        _Contact(frame=int(row["prev_frame"]), court_side=prev_side),
        _Contact(frame=int(row["curr_frame"]), court_side=curr_side),
    ]
    return a, nxt, contacts


def _accuracy_for_config(rows: list[dict], cfg: ChainWalkerConfig) -> dict:
    correct_flip = 0
    missed_flip = 0
    correct_stay = 0
    over_flip = 0
    for row in rows:
        a, nxt, contacts = _row_to_objs(row)
        walker_says_flip = _possession_flips_after(a, nxt, contacts, cfg)
        gt_says_flip = (row["event_type"] == "crossing")
        if gt_says_flip and walker_says_flip:
            correct_flip += 1
        elif gt_says_flip and not walker_says_flip:
            missed_flip += 1
        elif (not gt_says_flip) and (not walker_says_flip):
            correct_stay += 1
        else:
            over_flip += 1
    return {
        "correct_flip": correct_flip,
        "missed_flip": missed_flip,
        "correct_stay": correct_stay,
        "over_flip": over_flip,
        "total_correct": correct_flip + correct_stay,
        "total_pairs": len(rows),
    }


def main() -> int:
    if not IN_CSV.exists():
        print(f"ERROR: {IN_CSV} not found. Run probe_gt_net_crossings first.",
              file=sys.stderr)
        return 1

    with open(IN_CSV) as fh:
        rows = list(csv.DictReader(fh))
    print(f"Loaded {len(rows)} GT-pair events", flush=True)

    configs = {
        "cfg_00 (v13 baseline)": ChainWalkerConfig(False, False),
        "cfg_10 (B.1 only)":     ChainWalkerConfig(True, False),
        "cfg_01 (B.2 only)":     ChainWalkerConfig(False, True),
        "cfg_11 (B.1+B.2)":      ChainWalkerConfig(True, True),
    }

    results: dict[str, dict] = {}
    for name, cfg in configs.items():
        r = _accuracy_for_config(rows, cfg)
        results[name] = r
        print(f"  {name}: correct_flip={r['correct_flip']}, "
              f"missed_flip={r['missed_flip']}, "
              f"correct_stay={r['correct_stay']}, "
              f"over_flip={r['over_flip']}, "
              f"total_correct={r['total_correct']}/{r['total_pairs']}",
              flush=True)

    out = {"configs": results}
    (OUT_DIR / "summary.json").write_text(json.dumps(out, indent=2))

    md = ["# Walker Accuracy Replay — Summary (2026-05-20)", ""]
    md.append(f"Substrate: {len(rows)} GT-pair events from "
              "reports/gt_net_crossings_2026_05_20/events.csv.")
    md.append("")
    md.append("| Config | correct_flip | missed_flip | correct_stay | "
              "over_flip | total_correct |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for name, r in results.items():
        md.append(
            f"| {name} | {r['correct_flip']} | {r['missed_flip']} | "
            f"{r['correct_stay']} | {r['over_flip']} | "
            f"{r['total_correct']}/{r['total_pairs']} |"
        )
    md.append("")
    md.append("## Sanity check")
    md.append("")
    md.append("cfg_00 must produce numbers identical to the v13 baseline "
              "(482 correct_flip, 64 missed_flip, 699 correct_stay, "
              "20 over_flip). If not, the refactor broke v13-equivalent "
              "behavior — STOP and debug before A/B.")
    md.append("")
    (OUT_DIR / "summary.md").write_text("\n".join(md))
    print(f"\nWrote {OUT_DIR/'summary.md'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

### Step 2: Verify ruff + mypy clean

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && uv run ruff check scripts/measure_walker_accuracy_2026_05_20.py && uv run mypy scripts/measure_walker_accuracy_2026_05_20.py
```

### Step 3: Run + verify v13-baseline sanity

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && uv run python -u scripts/measure_walker_accuracy_2026_05_20.py 2>&1
```

Expected:
- `cfg_00` row matches **482 / 64 / 699 / 20** (the v13 baseline from `probe_gt_net_crossings_2026_05_20`).
  If it doesn't match, STOP — the refactor broke v13-equivalent behavior.
- `cfg_10` (B.1 only) should improve `missed_flip` by ~13 (BLOCK recoveries) with possibly +1-3 `over_flip` change.
- `cfg_01` (B.2 only) should improve `missed_flip` substantially (most of the 51 action-mis-class cases) AND improve `over_flip` (most of the 20).
- `cfg_11` should be at least as good as `cfg_01` (B.1 may add marginal extra on the few BLOCK cases B.2 missed).

### Step 4: Commit

```bash
cd /Users/mario/Personal/Projects/RallyCut && git add analysis/scripts/measure_walker_accuracy_2026_05_20.py analysis/reports/walker_accuracy_2026_05_20/ && git commit -m "$(cat <<'EOF'
diag(walker): Stage-1 walker-accuracy replay across 4 v14 flag configs

Replays each of 4 ChainWalkerConfig combinations against the 1265-pair
GT dataset (events.csv from probe_gt_net_crossings_2026_05_20). Cheap
deterministic ceiling — no scorer interaction. cfg_00 sanity-checks the
v13-equivalent path.

[no-version-bump]

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Stage-2 end-to-end A/B driver

**Files:**
- Create: `analysis/scripts/sub_lever_2_ab_2026_05_20.sh`

### Step 1: Write the driver

Create `/Users/mario/Personal/Projects/RallyCut/analysis/scripts/sub_lever_2_ab_2026_05_20.sh`:

```bash
#!/bin/bash
# Sub-lever 2 A/B driver: 2x2 flag grid for v14 chain-walker.
#
# Phase 1: cfg_00 redetect on 32 trusted videos (v13-equivalent baseline)
# Phase 2: measure attribution -> v14_cfg00.json
# Phase 3: cfg_10 redetect (B.1 ON)
# Phase 4: measure -> v14_cfg10.json + compare-to v14_cfg00
# Phase 5: cfg_01 redetect (B.2 ON)
# Phase 6: measure -> v14_cfg01.json + compare-to v14_cfg00
# Phase 7: cfg_11 redetect (both ON)
# Phase 8: measure -> v14_cfg11.json + compare-to v14_cfg00
# Phase 9: coherence audit (current state = cfg_11) for sanity check
#
# Runs from analysis/ dir; expects DB DSN in env or default localhost:5436.
# Total wall time estimate: ~3-4 hours on this corpus (4 full redetect
# cycles + 4 measurements).

set -euo pipefail

cd "$(dirname "$0")/.."  # analysis/

TRUSTED_NAMES="titi toto lulu wawa caco cece cici cuco gaga gigi kaka kiki keke koko kuku juju yeye gugu mame meme mimi moma mumu papa pepe pipi popo pupu veve vivi vovo haha"
echo "[setup] Resolving 32 video UUIDs..."
UUIDS=$(uv run python -c "
import os, psycopg
dsn = os.environ.get('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5436/rallycut')
names = '$TRUSTED_NAMES'.split()
with psycopg.connect(dsn) as c:
    rows = c.execute('SELECT id FROM videos WHERE name = ANY(%s) ORDER BY name', [names]).fetchall()
print('\n'.join(str(r[0]) for r in rows))
")
N_UUIDS=$(echo "$UUIDS" | wc -l | tr -d ' ')
echo "[setup] Resolved $N_UUIDS UUIDs (>=32 expected; duplicates by name OK)"
if [ "$N_UUIDS" -lt 32 ]; then
    echo "[ERROR] Expected at least 32 UUIDs, got $N_UUIDS" >&2
    exit 1
fi

run_redetect_loop () {
    local block_flag="$1"
    local verifier_flag="$2"
    local label="$3"
    echo
    echo "========================================="
    echo "[$label] redetect 32 videos, BLOCK_COND=$block_flag, BALL_TRAJ=$verifier_flag"
    echo "========================================="
    local i=0
    while IFS= read -r uid; do
        i=$((i+1))
        echo "  [$i/$N_UUIDS] redetect $uid"
        WALKER_BLOCK_CONDITIONAL=$block_flag \
            WALKER_BALL_TRAJECTORY_VERIFIER=$verifier_flag \
            USE_DYNAMIC_ATTRIBUTION_SCORER=1 \
            PYTHONUNBUFFERED=1 \
            uv run python -u scripts/redetect_all_actions.py --video "$uid" --apply
    done <<< "$UUIDS"
}

# Phase 1+2: cfg_00 baseline
run_redetect_loop 0 0 "PHASE 1: cfg_00 (v13 baseline)"
echo
echo "========================================="
echo "[PHASE 2] measure -> v14_cfg00"
echo "========================================="
PYTHONUNBUFFERED=1 uv run python -u scripts/measure_attribution_trusted_31_2026_05_20.py --label v14_cfg00

# Phase 3+4: cfg_10 (B.1 only)
run_redetect_loop 1 0 "PHASE 3: cfg_10 (B.1 only)"
echo
echo "========================================="
echo "[PHASE 4] measure -> v14_cfg10 (compare to v14_cfg00)"
echo "========================================="
PYTHONUNBUFFERED=1 uv run python -u scripts/measure_attribution_trusted_31_2026_05_20.py \
    --label v14_cfg10 --compare-to v14_cfg00

# Phase 5+6: cfg_01 (B.2 only)
run_redetect_loop 0 1 "PHASE 5: cfg_01 (B.2 only)"
echo
echo "========================================="
echo "[PHASE 6] measure -> v14_cfg01 (compare to v14_cfg00)"
echo "========================================="
PYTHONUNBUFFERED=1 uv run python -u scripts/measure_attribution_trusted_31_2026_05_20.py \
    --label v14_cfg01 --compare-to v14_cfg00

# Phase 7+8: cfg_11 (both ON)
run_redetect_loop 1 1 "PHASE 7: cfg_11 (B.1 + B.2)"
echo
echo "========================================="
echo "[PHASE 8] measure -> v14_cfg11 (compare to v14_cfg00)"
echo "========================================="
PYTHONUNBUFFERED=1 uv run python -u scripts/measure_attribution_trusted_31_2026_05_20.py \
    --label v14_cfg11 --compare-to v14_cfg00

# Phase 9: coherence audit on current state (cfg_11)
echo
echo "========================================="
echo "[PHASE 9] coherence audit (current state = cfg_11)"
echo "========================================="
PYTHONUNBUFFERED=1 uv run python -u scripts/audit_coherence_trusted_29_2026_05_17.py \
    --label v14_cfg11_coherence

echo
echo "========================================="
echo "A/B complete. Check reports/attribution_trusted_31_2026_05_20/v14_cfg*.json"
echo "and the compare-to outputs for the 3 non-baseline configs."
echo "========================================="
```

### Step 2: Make executable + run

```bash
chmod +x /Users/mario/Personal/Projects/RallyCut/analysis/scripts/sub_lever_2_ab_2026_05_20.sh
/Users/mario/Personal/Projects/RallyCut/analysis/scripts/sub_lever_2_ab_2026_05_20.sh 2>&1
```

Expected wall time: 3-4 hours. Run in background; monitor for phase boundaries + final-line "A/B complete".

### Step 3: Inspect compare-to outputs

```bash
ls /Users/mario/Personal/Projects/RallyCut/analysis/reports/attribution_trusted_31_2026_05_20/v14_cfg*.json
cat /Users/mario/Personal/Projects/RallyCut/analysis/reports/attribution_trusted_31_2026_05_20/v14_cfg10_compare.md 2>/dev/null || true
cat /Users/mario/Personal/Projects/RallyCut/analysis/reports/attribution_trusted_31_2026_05_20/v14_cfg01_compare.md 2>/dev/null || true
cat /Users/mario/Personal/Projects/RallyCut/analysis/reports/attribution_trusted_31_2026_05_20/v14_cfg11_compare.md 2>/dev/null || true
```

(`*_compare.md` is whatever output format the measure script's `--compare-to` produces; adapt the cat command to actual filenames if different.)

### Step 4: Commit driver + outputs

```bash
cd /Users/mario/Personal/Projects/RallyCut && git add analysis/scripts/sub_lever_2_ab_2026_05_20.sh analysis/reports/attribution_trusted_31_2026_05_20/v14_cfg00.json analysis/reports/attribution_trusted_31_2026_05_20/v14_cfg10.json analysis/reports/attribution_trusted_31_2026_05_20/v14_cfg01.json analysis/reports/attribution_trusted_31_2026_05_20/v14_cfg11.json && git commit -m "$(cat <<'EOF'
eval(walker): v14 chain-walker 2x2 A/B (cfg_00/10/01/11)

Driver + per-config attribution measurements on trusted-31. Shipping
gate per spec:
  1. Walker-decision accuracy strictly higher than v13 (Stage 1).
  2. End-to-end attribution accuracy >= v13 baseline - 0.5pp (net non-
     regressive).
  3. No protected-video regression > 2 contacts (mumu/keke/mame/veve/papa).

Spec at commit d4e5232d.

[no-version-bump]

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Decision + ship/no-ship (self-handled)

This task is controller-handled (not delegated). After Stage-1 + Stage-2 results are in:

- [ ] **Step 1: Read Stage-1 results.**
  ```bash
  cat /Users/mario/Personal/Projects/RallyCut/analysis/reports/walker_accuracy_2026_05_20/summary.md
  ```
  Verify cfg_00 matches v13 baseline (482/64/699/20). Note delta from cfg_00 for each of cfg_10/cfg_01/cfg_11.

- [ ] **Step 2: Read Stage-2 results.**
  ```bash
  python3 -c "
  import json
  for cfg in ('v14_cfg00', 'v14_cfg10', 'v14_cfg01', 'v14_cfg11'):
      d = json.load(open(f'/Users/mario/Personal/Projects/RallyCut/analysis/reports/attribution_trusted_31_2026_05_20/{cfg}.json'))
      t = d['totals']
      n_matched = t['total'] - t['unmatched']
      acc = 100 * t['correct'] / max(n_matched, 1)
      print(f'{cfg}: total={t[\"correct\"]}/{n_matched} matched ({acc:.2f}%), unmatched={t[\"unmatched\"]}')
  "
  ```

- [ ] **Step 3: Apply the ship gate per the decision tree from the spec.**
  - All 3 non-trivial configs PASS and one is clearly best → ship that config as v15 default.
  - Only B.1 passes → ship B.1-only as v15.
  - Only B.2 passes → ship B.2-only as v15.
  - All pass but lift <5 attribution wins → ship the simplest passing config (likely B.1).
  - None passes → NO-SHIP path: bump version back to v13-equivalent OR keep v14 with both flags off, write NO-SHIP memo, scope Class C as next attempt.

- [ ] **Step 4: If shipping, implement v15 commit.**
  Change the appropriate `_WALKER_*` env-default(s) from `"0"` to `"1"` in `action_classifier.py`. Bump `ACTION_PIPELINE_VERSION` to `"v15"` with history comment summarizing A/B numbers. Commit + fleet refresh:
  ```bash
  cd /Users/mario/Personal/Projects/RallyCut/analysis && uv run python scripts/redetect_all_actions.py --apply 2>&1
  ```

- [ ] **Step 5: Write findings memo.**
  Create `~/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/chain_walker_v14_<ship|no_ship>_2026_05_20.md` with:
  - Stage-1 walker accuracy table per config
  - Stage-2 attribution lift per config
  - Ship decision + rationale
  - Per-class oracle ceiling vs realistic delta (L6 oracle = 35; how close did we get?)
  - If NO-SHIP: scope Class C (chain-quality classifier) as next attempt

- [ ] **Step 6: Update MEMORY.md.**
  Add a [SHIPPED] or [NO-SHIP] entry referencing the findings memo and commit hashes.

---

## Out of scope

- **Action-classifier improvements** (WS-2 from the original four-prong campaign). The 51 action-mis-classification missed flips are PARTIALLY addressed by B.2 (verifier sees crossing despite mis-typed action), but the upstream GBM still produces the wrong action label. Fixing the GBM itself is a separate workstream.
- **Synthetic-serve team re-derivation** (~26 H3 cases from chain-asymmetry diagnostic) — separate spec.
- **Chain-quality classifier (Class C)** — only justified if v14 underdelivers vs L6's 35-violation oracle.
- **Fleet redetect outside trusted-31** — production refresh is a Phase-9 task contingent on shipping decision.

---

## Self-Review Notes

- **Spec coverage:** all 4 spec sections (B.1 design, B.2 design, dual-ceiling validation, ship gate) have a corresponding task or step. The decision tree from the spec's "Decision tree from A/B outcomes" section maps directly to Task 4 Step 3 branches.
- **Placeholder scan:** none. Task 4 references env-default flips and version bump explicitly; Task 2 references the actual v13 baseline numbers (482/64/699/20).
- **Type consistency:** `ChainWalkerConfig` in Task 1 step 3 matches the call sites in Task 2 step 1 (`ChainWalkerConfig(False, False)` positional args). `_possession_flips_after` signature stable across tests + replay + production.
- **The 1265-pair GT events.csv schema:** Task 2 step 1 inlines the `_row_to_objs` adapter that maps CSV columns to action/contact dataclasses. The court_side reconstruction note (team A = near convention) is explicit so a future reader understands the approximation.
