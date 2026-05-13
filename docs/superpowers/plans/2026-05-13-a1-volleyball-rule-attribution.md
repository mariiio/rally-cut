# A1 Volleyball-Rule Attribution Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** New attribution pass in `reattribute_players` that enforces the volleyball rule (`consecutive contacts ≠ same player; block exception only`) as a *hard* constraint. Flips offending pairs to the closest same-team alt; abstains when no alt is within 0.3 normalized court distance. Replaces parked Sub-2.B Phase 2 with a structurally different mechanism (no distance cap, no soft signal mix).

**Architecture:** Single forward pass over the action sequence, runs after Pass 2 (team re-attribution) and Pass 2c (within-team proximity swap) but before Pass 3 (ReID). Reads `contact.player_candidates` to find same-team alts. Env flag `USE_VOLLEYBALL_RULE_ATTRIBUTION` (default OFF). Marks abstentions with `attribution_uncertain=True` on the action dict for downstream consumers.

**Tech Stack:** Python 3.11+, pytest, existing `Contact`/`ClassifiedAction` dataclasses in `tracking/action_classifier.py` and `tracking/contact_detector.py`.

**Spec:** `docs/superpowers/specs/2026-05-13-action-attribution-root-causes-design.md` (workstream A1).

---

### Task 1: Confirm `ClassifiedAction` has (or can have) an `attribution_uncertain` field

**Files:**
- Read: `analysis/rallycut/tracking/action_classifier.py` — locate `ClassifiedAction` dataclass.

- [ ] **Step 1: Locate the dataclass**

Run: `grep -n "class ClassifiedAction" analysis/rallycut/tracking/action_classifier.py`.

Read the dataclass body to check existing fields.

- [ ] **Step 2: Add `attribution_uncertain` if missing**

If the dataclass doesn't have an `attribution_uncertain: bool = False` field, add it. It must default to `False` so existing call sites don't break.

If it already has the field, skip this step.

- [ ] **Step 3: Confirm `RallyActions.to_dict()` propagates the field**

Find the `to_dict()` method on `ClassifiedAction` (or `RallyActions` — whichever serializes actions to the `actions_json` shape). Ensure `attribution_uncertain` is included in the output dict. If not, add it (key: `"attributionUncertain"` to match camelCase conventions used elsewhere).

This is a *minor schema change* but does NOT require an `ACTION_PIPELINE_VERSION` bump because the field defaults to `False` and old rallies that didn't have the flag will read `False` on deserialization (additive change with safe default). Confirm by reading the post-classifier-change checklist in `analysis/CLAUDE.md` if uncertain.

If the bump *is* required, do it in Task 5's commit instead of separately.

- [ ] **Step 4: Commit if any field/serialization changes were made**

```
git add analysis/rallycut/tracking/action_classifier.py
git commit -m "feat(action-classifier): add attribution_uncertain field to ClassifiedAction"
```

If no changes were needed in Task 1, skip the commit.

---

### Task 2: Write failing tests for the new pass

**Files:**
- Create: `analysis/tests/unit/test_volleyball_rule_attribution.py`.

- [ ] **Step 1: Write the test file**

```python
"""Unit tests for the A1 volleyball-rule attribution pass.

Spec: docs/superpowers/specs/2026-05-13-action-attribution-root-causes-design.md
"""

from __future__ import annotations

from dataclasses import replace

import pytest


def _make_action(
    frame: int,
    action_type: str,
    player_track_id: int,
    confidence: float = 0.9,
    team: str | None = None,
):
    """Build a ClassifiedAction with sensible defaults for tests."""
    from rallycut.tracking.action_classifier import (
        ActionType, ClassifiedAction,
    )
    return ClassifiedAction(
        frame=frame,
        action_type=ActionType(action_type),
        confidence=confidence,
        player_track_id=player_track_id,
        court_side="far",
        ball_x=0.5,
        ball_y=0.5,
        velocity=0.01,
        is_synthetic=False,
        team=team,
    )


def _make_contact(frame: int, candidates: list[tuple[int, float]]):
    """Build a Contact with a player_candidates list (tid, distance)."""
    from rallycut.tracking.contact_detector import Contact
    return Contact(
        frame=frame,
        ball_x=0.5,
        ball_y=0.5,
        court_side="far",
        player_track_id=candidates[0][0] if candidates else -1,
        player_distance=candidates[0][1] if candidates else float("inf"),
        player_candidates=candidates,
        velocity=0.01,
    )


class TestVolleyballRulePass:
    """A1: anti-self-touch for SET/RECEIVE/DIG with block exception."""

    def test_cascade_dig_flips_to_same_team_alt(self):
        """Cascade frame 225: attack(p2 B) → dig(p2 B). Alt p1 (B)
        at 0.041 normalized is within abstention bound. Should flip.
        """
        from rallycut.tracking.action_classifier import (
            _attribution_volleyball_rule_pass,
        )
        actions = [
            _make_action(176, "attack", 2),
            _make_action(225, "dig", 2),
        ]
        contacts = [
            _make_contact(176, [(2, 0.052), (3, 0.125), (1, 0.146)]),
            _make_contact(225, [(2, 0.006), (1, 0.041), (3, 0.101)]),
        ]
        team_assignments = {1: 1, 2: 1, 3: 0, 4: 0}  # 1=team B, 0=team A

        n = _attribution_volleyball_rule_pass(
            actions=actions,
            contacts=contacts,
            team_assignments=team_assignments,
        )

        assert n == 1, "one flip expected"
        assert actions[0].player_track_id == 2, "prev attack unchanged"
        assert actions[1].player_track_id == 1, "dig flipped to track 1"
        assert actions[1].attribution_uncertain is False

    def test_cascade_set_flips_to_same_team_alt(self):
        """Cascade frame 128: receive(p2 B) → set(p2 B). Same shape."""
        from rallycut.tracking.action_classifier import (
            _attribution_volleyball_rule_pass,
        )
        actions = [
            _make_action(76, "receive", 2),
            _make_action(128, "set", 2),
        ]
        contacts = [
            _make_contact(76, [(2, 0.020), (1, 0.080), (3, 0.150)]),
            _make_contact(128, [(2, 0.009), (1, 0.087), (3, 0.097)]),
        ]
        team_assignments = {1: 1, 2: 1, 3: 0, 4: 0}

        n = _attribution_volleyball_rule_pass(
            actions=actions,
            contacts=contacts,
            team_assignments=team_assignments,
        )

        assert n == 1
        assert actions[1].player_track_id == 1

    def test_block_exception_no_flip(self):
        """block → same-player is legal. No flip."""
        from rallycut.tracking.action_classifier import (
            _attribution_volleyball_rule_pass,
        )
        actions = [
            _make_action(100, "block", 2),
            _make_action(110, "dig", 2),
        ]
        contacts = [
            _make_contact(100, [(2, 0.05), (1, 0.10)]),
            _make_contact(110, [(2, 0.01), (1, 0.04)]),
        ]
        team_assignments = {1: 1, 2: 1, 3: 0, 4: 0}

        n = _attribution_volleyball_rule_pass(
            actions=actions, contacts=contacts,
            team_assignments=team_assignments,
        )

        assert n == 0
        assert actions[1].player_track_id == 2  # unchanged

    def test_cross_team_no_flip(self):
        """Cross-team consecutive is not a C-4 violation. No flip."""
        from rallycut.tracking.action_classifier import (
            _attribution_volleyball_rule_pass,
        )
        actions = [
            _make_action(100, "attack", 2),
            _make_action(130, "dig", 3),
        ]
        contacts = [
            _make_contact(100, [(2, 0.05), (1, 0.10)]),
            _make_contact(130, [(3, 0.02), (4, 0.10)]),
        ]
        team_assignments = {1: 1, 2: 1, 3: 0, 4: 0}

        n = _attribution_volleyball_rule_pass(
            actions=actions, contacts=contacts,
            team_assignments=team_assignments,
        )

        assert n == 0
        assert actions[1].player_track_id == 3  # unchanged

    def test_abstention_when_alt_too_far(self):
        """No same-team alt within 0.3 normalized → abstain, mark
        attribution_uncertain=True, don't flip.
        """
        from rallycut.tracking.action_classifier import (
            _attribution_volleyball_rule_pass,
        )
        actions = [
            _make_action(100, "receive", 2),
            _make_action(130, "set", 2),
        ]
        contacts = [
            _make_contact(100, [(2, 0.05), (1, 0.40)]),
            _make_contact(130, [(2, 0.01), (1, 0.45)]),  # alt at 0.45 > 0.3
        ]
        team_assignments = {1: 1, 2: 1, 3: 0, 4: 0}

        n = _attribution_volleyball_rule_pass(
            actions=actions, contacts=contacts,
            team_assignments=team_assignments,
        )

        assert n == 0, "no flip — abstained"
        assert actions[1].player_track_id == 2  # unchanged
        assert actions[1].attribution_uncertain is True

    def test_no_same_team_alt_at_all_abstains(self):
        """Only 2 candidates and both are the same player → abstain."""
        from rallycut.tracking.action_classifier import (
            _attribution_volleyball_rule_pass,
        )
        actions = [
            _make_action(100, "receive", 2),
            _make_action(130, "set", 2),
        ]
        contacts = [
            _make_contact(100, [(2, 0.05), (3, 0.20)]),
            _make_contact(130, [(2, 0.01), (3, 0.20)]),  # 3 is team A
        ]
        team_assignments = {1: 1, 2: 1, 3: 0, 4: 0}

        n = _attribution_volleyball_rule_pass(
            actions=actions, contacts=contacts,
            team_assignments=team_assignments,
        )

        assert n == 0
        assert actions[1].attribution_uncertain is True

    def test_multiple_pairs_in_one_rally(self):
        """Cascade with three same-player back-to-back pairs.
        Each pair fires independently in forward order; flipping
        action[i] resolves pair (i-1, i) but may introduce/remove
        violations for pair (i, i+1).
        """
        from rallycut.tracking.action_classifier import (
            _attribution_volleyball_rule_pass,
        )
        # 128 set p2 → 176 attack p2 → 225 dig p2 → 276 set p2.
        # Forward iteration: flip 176? No — that would touch the
        # prev. The forward rule re-attributes the CURR side.
        # Pair (128, 176): same p2 → flip 176 to p1.
        # Pair (176, 225) after flip: p1 vs p2 — no violation.
        # But we still need to handle the original (225, 276): same p2 → flip 276.
        # Result expectation depends on the implementation's exact
        # behavior; the spec example says 3 flips reduces to (128→p1,
        # 176 unchanged, 225 flips → p1, 276 unchanged). Verify this
        # in the implementation; if your forward pass produces a
        # different fixed-point, update the assertion accordingly.
        actions = [
            _make_action(128, "set", 2),
            _make_action(176, "attack", 2),
            _make_action(225, "dig", 2),
            _make_action(276, "set", 2),
        ]
        contacts = [
            _make_contact(128, [(2, 0.009), (1, 0.087)]),
            _make_contact(176, [(2, 0.052), (1, 0.146)]),
            _make_contact(225, [(2, 0.006), (1, 0.041)]),
            _make_contact(276, [(2, 0.003), (1, 0.103)]),
        ]
        team_assignments = {1: 1, 2: 1, 3: 0, 4: 0}

        n = _attribution_volleyball_rule_pass(
            actions=actions, contacts=contacts,
            team_assignments=team_assignments,
        )

        # Expected: pair (128, 176) → 128 prev unchanged, 176 curr (attack)
        # flips to p1 (closest team-B alt at 0.146 < 0.3).
        # After 176 flips, pair (176, 225) is now (p1 attack, p2 dig) — no
        # violation. Pair (225, 276) is (p2 dig, p2 set) — fires; 276 flips
        # to p1 (0.103 < 0.3).
        # Total: 2 flips. Final ids: [2, 1, 2, 1].
        assert n == 2
        assert [a.player_track_id for a in actions] == [2, 1, 2, 1]

    def test_default_off_via_env(self, monkeypatch):
        """When USE_VOLLEYBALL_RULE_ATTRIBUTION is unset or 0, the
        pass is a no-op when called via reattribute_players.
        """
        from rallycut.tracking.action_classifier import reattribute_players
        monkeypatch.delenv(
            "USE_VOLLEYBALL_RULE_ATTRIBUTION", raising=False,
        )
        actions = [
            _make_action(176, "attack", 2),
            _make_action(225, "dig", 2),
        ]
        contacts = [
            _make_contact(176, [(2, 0.052), (1, 0.146)]),
            _make_contact(225, [(2, 0.006), (1, 0.041)]),
        ]
        team_assignments = {1: 1, 2: 1, 3: 0, 4: 0}

        result = reattribute_players(
            actions=actions,
            contacts=contacts,
            team_assignments=team_assignments,
        )

        # Without the env flag, dig stays as p2 (the cascade error
        # persists pre-A1).
        assert result[1].player_track_id == 2

    def test_enabled_via_env(self, monkeypatch):
        """When USE_VOLLEYBALL_RULE_ATTRIBUTION=1, the pass fires."""
        from rallycut.tracking.action_classifier import reattribute_players
        monkeypatch.setenv("USE_VOLLEYBALL_RULE_ATTRIBUTION", "1")

        actions = [
            _make_action(176, "attack", 2),
            _make_action(225, "dig", 2),
        ]
        contacts = [
            _make_contact(176, [(2, 0.052), (1, 0.146)]),
            _make_contact(225, [(2, 0.006), (1, 0.041)]),
        ]
        team_assignments = {1: 1, 2: 1, 3: 0, 4: 0}

        result = reattribute_players(
            actions=actions,
            contacts=contacts,
            team_assignments=team_assignments,
        )

        assert result[1].player_track_id == 1
```

**Important:** The fixture builders use field names from `ClassifiedAction` and `Contact`. Look up the actual dataclass definitions before pasting and adjust field names if they differ. Common gotchas:
- `ClassifiedAction.team` may not exist as a field — if not, drop it from `_make_action`.
- `Contact.player_candidates` field type and the dataclass constructor may differ.

Run `grep -n "class ClassifiedAction\|class Contact" analysis/rallycut/tracking/*.py` and read both to confirm.

- [ ] **Step 2: Run the tests, verify they fail**

```
uv run pytest tests/unit/test_volleyball_rule_attribution.py -v
```

Expected: all FAIL with `ImportError` (`_attribution_volleyball_rule_pass` not yet defined).

---

### Task 3: Implement `_attribution_volleyball_rule_pass`

**Files:**
- Modify: `analysis/rallycut/tracking/action_classifier.py` — add the function near other reattribute_players helpers.

- [ ] **Step 1: Add the constant**

Near the top of `action_classifier.py` (where other constants live), add:

```python
# Volleyball-rule attribution pass (A1, 2026-05-13). Abstention bound:
# if no same-team alt to the ball is within this normalized court
# distance, the rule abstains rather than flip. Picked as a sanity
# bound — values in [0.2, 0.4] produce identical panel outcomes.
_VOLLEYBALL_RULE_ABSTAIN_BOUND = 0.3
```

- [ ] **Step 2: Add the function**

Add this function near other reattribute_players helpers (e.g., right above the `def reattribute_players(` definition, around line 2959):

```python
def _attribution_volleyball_rule_pass(
    *,
    actions: list[ClassifiedAction],
    contacts: list[Contact],
    team_assignments: dict[int, int] | None,
) -> int:
    """A1: enforce the volleyball-rule (consecutive contacts ≠ same
    player, block exception only) as a hard constraint at attribution
    time.

    For each consecutive pair (prev, curr) in frame-order:
      - If same team AND same player AND prev.action_type != BLOCK:
        - Find the closest same-team alt to the ball at curr.frame
          from contact.player_candidates (excluding prev.player).
        - If alt_distance ≤ _VOLLEYBALL_RULE_ABSTAIN_BOUND: flip
          curr.player_track_id to alt.
        - Else: set curr.attribution_uncertain = True (abstain).

    Modifies `actions` in place. Returns count of flips (abstentions
    don't count).

    No distance cap (the 2x cap that parked Sub-2.B Phase 2 excluded
    cases like the cascade with alt_ratios up to 36x — the alt is
    still legitimate by the volleyball rule). No soft signal mix
    (signals were where Sub-2.B's hand-tuning happened).

    Spec: docs/superpowers/specs/2026-05-13-action-attribution-root-causes-design.md
    """
    if team_assignments is None or not team_assignments:
        return 0
    if len(actions) < 2:
        return 0

    # Index contacts by frame for O(1) lookup.
    contact_by_frame: dict[int, Contact] = {c.frame: c for c in contacts}

    # Sort actions by frame defensively (same convention as
    # coherence_invariants); we mutate in place, so build an index
    # of (frame → list-position).
    sorted_indices = sorted(
        range(len(actions)), key=lambda i: actions[i].frame
    )

    n_flips = 0
    for k in range(1, len(sorted_indices)):
        prev_idx = sorted_indices[k - 1]
        curr_idx = sorted_indices[k]
        prev = actions[prev_idx]
        curr = actions[curr_idx]

        if prev.player_track_id < 0 or curr.player_track_id < 0:
            continue
        if prev.player_track_id != curr.player_track_id:
            continue

        prev_team = team_assignments.get(prev.player_track_id)
        curr_team = team_assignments.get(curr.player_track_id)
        if prev_team is None or curr_team is None:
            continue
        if prev_team != curr_team:
            continue  # cross-team isn't a C-4 violation

        # Strict block exception: only prev=block exempts the pair.
        # Use ActionType.BLOCK to avoid string comparison.
        if prev.action_type == ActionType.BLOCK:
            continue

        contact = contact_by_frame.get(curr.frame)
        if contact is None or not contact.player_candidates:
            # No candidates to consider; abstain.
            curr.attribution_uncertain = True
            continue

        # Find closest same-team alt that isn't prev.player.
        best_tid = -1
        best_dist = float("inf")
        for tid, dist in contact.player_candidates:
            if tid == curr.player_track_id:
                continue  # current (== prev) — skip
            if team_assignments.get(tid) != curr_team:
                continue
            if dist < best_dist:
                best_tid = tid
                best_dist = dist

        if best_tid < 0 or best_dist > _VOLLEYBALL_RULE_ABSTAIN_BOUND:
            # No plausible alt → abstain.
            curr.attribution_uncertain = True
            continue

        logger.info(
            "volleyball_rule_flip frame=%d action=%s prev_action=%s "
            "old_pid=%d new_pid=%d team=%d alt_dist=%.3f",
            curr.frame, curr.action_type.value, prev.action_type.value,
            curr.player_track_id, best_tid, curr_team, best_dist,
        )
        curr.player_track_id = best_tid
        n_flips += 1

    return n_flips
```

- [ ] **Step 3: Run the unit tests**

```
uv run pytest tests/unit/test_volleyball_rule_attribution.py -v -k "not enabled_via_env and not default_off"
```

Expected: the non-integration tests PASS. (`enabled_via_env` and `default_off` will still fail until Task 4 wires the env flag.)

If any fail, read the test diff carefully — the most likely issue is fixture field-name mismatch.

- [ ] **Step 4: Commit**

```
git add analysis/rallycut/tracking/action_classifier.py analysis/tests/unit/test_volleyball_rule_attribution.py
git commit -m "feat(attribution): add _attribution_volleyball_rule_pass (A1)"
```

---

### Task 4: Wire into `reattribute_players` behind env flag

**Files:**
- Modify: `analysis/rallycut/tracking/action_classifier.py` — `reattribute_players` function.

- [ ] **Step 1: Add the env-flag-gated call**

In `reattribute_players` (around line 2959), find the section AFTER Pass 2c (within-team proximity swap, around line 3194 — search for `Within-team proximity swap`). Add A1 after Pass 2c, BEFORE Pass 3 (ReID at line 3196). Insert this block:

```python
    # Pass 2d (A1, 2026-05-13): volleyball-rule attribution.
    # Enforces the hard rule that consecutive contacts ≠ same player
    # (block exception). Drops Sub-2.B Phase 2's 2x distance cap;
    # abstains when no same-team alt is within
    # _VOLLEYBALL_RULE_ABSTAIN_BOUND. Default OFF; opt-in via env.
    # Spec: docs/superpowers/specs/2026-05-13-action-attribution-root-causes-design.md
    if os.environ.get("USE_VOLLEYBALL_RULE_ATTRIBUTION", "0") == "1":
        n_vb = _attribution_volleyball_rule_pass(
            actions=actions,
            contacts=contacts,
            team_assignments=team_assignments,
        )
        if n_vb > 0:
            logger.info(
                "Volleyball-rule attribution: re-attributed %d/%d actions",
                n_vb, len(actions),
            )
```

- [ ] **Step 2: Run the env-flag tests**

```
uv run pytest tests/unit/test_volleyball_rule_attribution.py -v
```

Expected: all PASS (including `default_off` and `enabled_via_env`).

- [ ] **Step 3: Run the full reattribute_players test suite (regression)**

```
uv run pytest tests/unit/ -k "reattribute or attribution" -v
```

Expected: no new failures.

- [ ] **Step 4: Commit**

```
git add analysis/rallycut/tracking/action_classifier.py
git commit -m "feat(reattribute): wire A1 volleyball-rule pass behind USE_VOLLEYBALL_RULE_ATTRIBUTION"
```

---

### Task 5: A/B measurement on the 22-rally panel

**Files:**
- No code changes; measurement.

- [ ] **Step 1: Reset matcher state for the panel**

```
uv run python scripts/reset_matcher_state.py --all-with-gt
```

Per `analysis/CLAUDE.md` "Cross-rally matcher validation protocol" — mandatory before any A/B.

- [ ] **Step 2: Baseline measurement (A1 off)**

```
unset USE_VOLLEYBALL_RULE_ATTRIBUTION
scripts/eval_cross_fixture.sh 2>&1 | tee analysis/reports/a1_volleyball_rule/baseline_2026_05_13.log
```

Make sure `analysis/reports/a1_volleyball_rule/` exists first (`mkdir -p analysis/reports/a1_volleyball_rule`).

Expected: panel summary with same_team / cross_team / missing counts.

- [ ] **Step 3: Treatment measurement (A1 on)**

```
USE_VOLLEYBALL_RULE_ATTRIBUTION=1 scripts/eval_cross_fixture.sh 2>&1 | tee analysis/reports/a1_volleyball_rule/a1_on_2026_05_13.log
```

- [ ] **Step 4: Compare**

Diff the two logs. Compute:
- `same_team` errors: baseline vs A1 (must drop by ≥ 2 per ship gate item 1).
- `cross_team` errors: A1 not worse.
- `missing` / `abstained`: track new abstentions.
- panel correct-rate: A1 not worse.
- abstention rate (count of `attribution_uncertain=True` divided by total contacts): ≤ 5% per ship gate item 4.

- [ ] **Step 5: Write the verdict**

Create `analysis/reports/a1_volleyball_rule/verdict_2026_05_13.md`:

```markdown
# A1 Volleyball-Rule Attribution — Panel A/B Verdict

| Metric | Baseline | A1 on | Δ |
|---|---:|---:|---:|
| correct | N | N | Δ |
| same_team | N | N | Δ |
| cross_team | N | N | Δ |
| missing / abstained | N | N | Δ |
| abstention rate | — | X% | — |

## Per-fixture deltas

<paste per-fixture rows>

## Ship gate

- [ ] same_team -≥2
- [ ] correct rate not worse
- [ ] abstention rate ≤ 5%
- [ ] cross_team not worse

## Verdict

SHIP / NO-SHIP / DONE-WITH-CONCERNS
```

Fill in the values from the two log files.

- [ ] **Step 6: Commit the verdict**

```
git add analysis/reports/a1_volleyball_rule/
git commit -m "docs(a1): panel A/B verdict 2026-05-13"
```

---

### Task 6: C-4 catalog regen and ship-gate check

**Files:**
- No code changes; measurement.

- [ ] **Step 1: Run the catalog with A1 on**

```
USE_VOLLEYBALL_RULE_ATTRIBUTION=1 uv run python scripts/catalog_c4_violations.py
```

Compare to the prior `2026-05-13_post_redetect.csv`.

Ship gate item 2: C-4 violation count drops by ≥ 80 % on rallies where at least one same-team back-to-back pair has `alt_ratio ∈ [2×, 50×]`.

- [ ] **Step 2: Compute the filtered drop**

From `analysis/`:

```
uv run python -c "
import csv
def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))

before = load('analysis/reports/coherence_c4_catalog/2026-05-13_post_redetect.csv')
# After is the same path freshly regenerated with A1 on. Re-name if needed.
after = load('analysis/reports/coherence_c4_catalog/2026-05-13_a1_on.csv')

def filtered(rows):
    out = []
    for r in rows:
        try:
            ratio = float(r['curr_best_same_team_alt_ratio'])
        except (ValueError, KeyError):
            continue
        if 2.0 <= ratio <= 50.0:
            out.append(r)
    return out

b = filtered(before)
a = filtered(after)
print(f'Before: {len(b)} filtered C-4 pairs')
print(f'After:  {len(a)} filtered C-4 pairs')
print(f'Drop:   {(len(b) - len(a)) / max(len(b), 1) * 100:.1f}%')
"
```

(You'll need to manually save the new catalog as `2026-05-13_a1_on.csv` before running this — re-run `catalog_c4_violations.py` with A1 on, then rename the output.)

Ship gate: drop ≥ 80%.

- [ ] **Step 3: Update the verdict file**

Append the catalog ship-gate result to `analysis/reports/a1_volleyball_rule/verdict_2026_05_13.md`.

- [ ] **Step 4: Commit**

```
git add analysis/reports/
git commit -m "docs(a1): C-4 catalog regen verdict + filtered drop measurement"
```

---

### Task 7: Decide ship vs no-ship

- [ ] **Step 1: Review all four ship gates**

1. 22-rally panel: same_team errors down by ≥ 2, correct-rate not worse. (Task 5.)
2. C-4 filtered drop ≥ 80%. (Task 6.)
3. Fleet F1 not worse than baseline. (Task 5 should produce this as part of the eval; if not, re-run `measure_pid_accuracy.py` on the panel.)
4. Abstention rate ≤ 5%. (Task 5.)

- [ ] **Step 2: If all gates pass**

Make A1 default-ON by changing the env-flag default in `reattribute_players` from `"0"` to `"1"`:

```python
    if os.environ.get("USE_VOLLEYBALL_RULE_ATTRIBUTION", "1") == "1":
```

Bump `ACTION_PIPELINE_VERSION` (per `analysis/CLAUDE.md` post-classifier-change checklist) — A1 changes serialized output (`attributionUncertain` field flips for some actions).

Commit:

```
git add analysis/rallycut/tracking/action_classifier.py
git commit -m "feat(a1): default-ON USE_VOLLEYBALL_RULE_ATTRIBUTION, bump ACTION_PIPELINE_VERSION"
```

Refresh fleet:

```
uv run python scripts/redetect_all_actions.py --apply
```

- [ ] **Step 3: If any gate fails — NO-SHIP**

Keep the infrastructure (env-flag-gated) but leave default OFF. Document the NO-SHIP in the verdict.

---

### Task 8: Memory update

- [ ] **Step 1: Create the topic file**

`/Users/mario/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/a1_volleyball_rule_attribution_2026_05_13.md`:

```markdown
---
name: a1-volleyball-rule-attribution-2026-05-13
description: "<SHIPPED / NO-SHIP — fill in> Hard volleyball-rule attribution pass in reattribute_players. <Panel delta + abstention rate from verdict>. Replaces parked Sub-2.B Phase 2 with no distance cap + abstention bound 0.3 norm court distance."
metadata: 
  type: project
---

# A1 Volleyball-Rule Attribution Pass — <SHIPPED / NO-SHIP> 2026-05-13

<Paste the verdict table from analysis/reports/a1_volleyball_rule/verdict_2026_05_13.md>

## Why this is not Sub-2.B Phase 2

- Sub-2.B's 2x distance cap was the binding constraint; A1 has no cap.
- Sub-2.B's "≥2 of {type_fit, team_geometry, alt_ratio, confidence}"
  soft signal mix was where the hand-tuning happened; A1 uses the
  hard volleyball rule.
- A1 abstains (marks attribution_uncertain) when no plausible alt
  exists, rather than flipping unconditionally.

Composes with [[coherence_repair_sub_2b_2026_05_13]] (replaces
parked Phase 2), [[adaptive_candidate_window_v30_2026_05_11]]
(Pass 2c runs before A1).
```

- [ ] **Step 2: Add MEMORY.md index line**

```
- [<SHIPPED/NO-SHIP>] [**A1 volleyball-rule attribution 2026-05-13**](a1_volleyball_rule_attribution_2026_05_13.md) — Hard rule in reattribute_players. <One-line panel/catalog result>.
```

---

## Ship gate verification

- [ ] Panel: same_team -≥2, correct rate not worse, abstention ≤ 5%.
- [ ] Catalog: filtered C-4 drop ≥ 80%.
- [ ] Fleet F1 not worse.
- [ ] Cascade rally `a0881d82` resolves: frames 128 and 225 flip to track 1.
