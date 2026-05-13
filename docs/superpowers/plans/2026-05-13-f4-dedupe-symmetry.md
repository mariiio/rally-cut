# F4 Dedupe Symmetry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `repair_action_sequence.Rule 3` in `action_classifier.py` to also dedupe consecutive serves when exactly one of the pair is synthetic — the missing symmetric branch alongside the existing non-synth duplicate dedupe.

**Architecture:** Single-rule extension at one site in `action_classifier.py`. The existing Rule 3 only checks `not a.is_synthetic` when picking the *anchor*; we widen the dedupe to also catch (anchor=synth, second=real) and (anchor=real, second=synth) pairs. The "anchor" remains the first serve found regardless of synth flag.

**Tech Stack:** Python 3.11+, pytest. No env flag — closed-form rule extension.

**Spec:** `docs/superpowers/specs/2026-05-13-action-attribution-root-causes-design.md` (workstream B).

---

### Task 1: Find the test file and confirm Rule 3 has tests

**Files:**
- Read: `analysis/tests/unit/test_action_classifier.py` (or wherever `repair_action_sequence` is tested — locate first).

- [ ] **Step 1: Locate Rule 3's existing tests**

Run from `analysis/`:

```
grep -rn "Rule 3\|repair_action_sequence\|duplicate serve" tests/unit/
```

Expected: at least one test in `test_action_classifier.py` exercising the non-synth duplicate-serve dedupe. Read it to understand the test fixture shape (how synthetic serves and `ClassifiedAction` objects are constructed in fixtures).

If no Rule 3 test exists, add one for the existing non-synth behavior in Task 2 as a regression guard before extending.

---

### Task 2: Write the failing test for synth+real dedupe

**Files:**
- Modify: `analysis/tests/unit/test_action_classifier.py` (or the file located in Task 1) — append a new test for the synth+real case.

- [ ] **Step 1: Add the failing test**

Find the existing Rule 3 test class (or the section grouping `repair_action_sequence` tests). Add this test:

```python
def test_rule_3_dedupes_synth_then_real_serve() -> None:
    """F4: synthetic-prepended serve followed immediately by a real
    classifier-produced serve. Second serve must become receive.

    Rally 88a04529 (keke) is the canonical fixture: synth serve at
    f106 for team B (correctly prepended by synthetic-serve placement),
    then a real action at f111 was classified as serve. The real one
    is the actual receive.
    """
    from rallycut.tracking.action_classifier import (
        ActionType, ClassifiedAction, repair_action_sequence,
    )

    actions = [
        ClassifiedAction(
            frame=106,
            action_type=ActionType.SERVE,
            confidence=0.7,
            player_track_id=3,
            court_side="far",
            ball_x=0.5,
            ball_y=0.8,
            velocity=0.0,
            is_synthetic=True,
            team="B",
        ),
        ClassifiedAction(
            frame=111,
            action_type=ActionType.SERVE,
            confidence=0.85,
            player_track_id=1,
            court_side="near",
            ball_x=0.55,
            ball_y=0.55,
            velocity=0.012,
            is_synthetic=False,
            team="A",
        ),
    ]

    # ball_positions and net_y are not used by Rule 3 itself, but
    # repair_action_sequence may need them for other rules. Pass minimal
    # plausible values.
    repaired = repair_action_sequence(
        actions=actions,
        ball_positions=[],  # adjust if your repair_action_sequence signature requires non-empty
        net_y=0.5,
    )

    assert len(repaired) == 2
    assert repaired[0].action_type == ActionType.SERVE
    assert repaired[0].is_synthetic is True, "anchor synth serve preserved"
    assert repaired[1].action_type == ActionType.RECEIVE, (
        "second serve (real) should be reclassified as receive"
    )


def test_rule_3_dedupes_real_then_synth_serve() -> None:
    """Reverse ordering: real serve first, then a synth serve appears
    later (hypothetically, e.g. from a later placement repair). The
    second one (synth) should still be reclassified.

    Note: in current pipeline this ordering doesn't occur (synth is
    always prepended), but Rule 3's symmetry should be order-agnostic
    to avoid silently re-introducing the bug if placement logic
    changes later.
    """
    from rallycut.tracking.action_classifier import (
        ActionType, ClassifiedAction, repair_action_sequence,
    )

    actions = [
        ClassifiedAction(
            frame=110,
            action_type=ActionType.SERVE,
            confidence=0.9,
            player_track_id=1,
            court_side="near",
            ball_x=0.5,
            ball_y=0.8,
            velocity=0.0,
            is_synthetic=False,
            team="A",
        ),
        ClassifiedAction(
            frame=125,
            action_type=ActionType.SERVE,
            confidence=0.7,
            player_track_id=3,
            court_side="far",
            ball_x=0.55,
            ball_y=0.55,
            velocity=0.012,
            is_synthetic=True,
            team="B",
        ),
    ]

    repaired = repair_action_sequence(
        actions=actions, ball_positions=[], net_y=0.5,
    )

    assert repaired[0].action_type == ActionType.SERVE
    assert repaired[1].action_type == ActionType.RECEIVE


def test_rule_3_non_synth_pair_still_works() -> None:
    """Regression guard: the existing non-synth + non-synth dedupe
    behavior must not change. Both are real classifier-produced
    serves; second becomes receive.
    """
    from rallycut.tracking.action_classifier import (
        ActionType, ClassifiedAction, repair_action_sequence,
    )

    actions = [
        ClassifiedAction(
            frame=100,
            action_type=ActionType.SERVE,
            confidence=0.9,
            player_track_id=1,
            court_side="near",
            ball_x=0.5,
            ball_y=0.8,
            velocity=0.0,
            is_synthetic=False,
            team="A",
        ),
        ClassifiedAction(
            frame=125,
            action_type=ActionType.SERVE,
            confidence=0.7,
            player_track_id=3,
            court_side="far",
            ball_x=0.55,
            ball_y=0.55,
            velocity=0.012,
            is_synthetic=False,
            team="B",
        ),
    ]

    repaired = repair_action_sequence(
        actions=actions, ball_positions=[], net_y=0.5,
    )

    assert repaired[0].action_type == ActionType.SERVE
    assert repaired[1].action_type == ActionType.RECEIVE
```

**Important:** Look up the actual `ClassifiedAction` dataclass signature and `repair_action_sequence` parameter list before pasting — adjust field names and the call signature to match what's actually in `action_classifier.py`. Use `grep -n "class ClassifiedAction\|def repair_action_sequence" analysis/rallycut/tracking/action_classifier.py` to find them.

- [ ] **Step 2: Run the synth tests, verify they fail**

```
uv run pytest tests/unit/test_action_classifier.py -k "synth" -v
```

Expected: the two synth tests FAIL (the second action stays as SERVE because Rule 3's current logic only counts non-synthetic serves as the anchor; a synth+real or real+synth pair has the first one consumed as anchor and the second not flagged because the existing loop continues without dedupe in the synth case).

The non-synth regression test should PASS (existing behavior).

---

### Task 3: Extend Rule 3 to handle synth+real symmetric case

**Files:**
- Modify: `analysis/rallycut/tracking/action_classifier.py` — Rule 3 block at line 2443-2459.

- [ ] **Step 1: Read the current Rule 3 implementation**

The current block (around line 2443):

```python
    # ------------------------------------------------------------------
    # Rule 3 (pre-pass): Duplicate non-synthetic serves → extras become receive.
    # The second detected "serve" is almost certainly the serve return.
    # Only non-synthetic serves count; the first real serve is the anchor.
    # ------------------------------------------------------------------
    first_real_serve_found = False
    if 3 not in _disabled:
        for i, a in enumerate(repaired):
            if a.action_type == ActionType.SERVE and not a.is_synthetic:
                if not first_real_serve_found:
                    first_real_serve_found = True
                else:
                    if repair_count >= _MAX_SEQUENCE_REPAIRS:
                        break
                    _triggers[3] += 1
                    repaired[i] = _reclassify(a, ActionType.RECEIVE)
                    repair_count += 1
                    logger.debug(
                        "Repair rule 3: duplicate serve at f%d → receive",
                        a.frame,
                    )
```

- [ ] **Step 2: Replace it with the symmetric version**

```python
    # ------------------------------------------------------------------
    # Rule 3 (pre-pass): Duplicate serves → extras become receive.
    # Symmetric over synthetic-ness: the FIRST serve found (regardless
    # of is_synthetic) is the anchor; every subsequent serve becomes
    # receive. Pre-2026-05-13 this rule only counted non-synthetic
    # serves as the anchor, which silently allowed a (synth, real)
    # pair to slip through as a double-serve. Spec WS-B in
    # docs/superpowers/specs/2026-05-13-action-attribution-root-causes-design.md
    # ------------------------------------------------------------------
    first_serve_found = False
    if 3 not in _disabled:
        for i, a in enumerate(repaired):
            if a.action_type == ActionType.SERVE:
                if not first_serve_found:
                    first_serve_found = True
                else:
                    if repair_count >= _MAX_SEQUENCE_REPAIRS:
                        break
                    _triggers[3] += 1
                    repaired[i] = _reclassify(a, ActionType.RECEIVE)
                    repair_count += 1
                    logger.debug(
                        "Repair rule 3: duplicate serve at f%d (synth=%s) → receive",
                        a.frame, a.is_synthetic,
                    )
```

The change is: rename `first_real_serve_found` → `first_serve_found` and drop the `and not a.is_synthetic` condition in the outer `if`.

- [ ] **Step 3: Run all three new tests + existing Rule 3 tests**

```
uv run pytest tests/unit/test_action_classifier.py -k "rule_3 or synth" -v
```

Expected: all PASS.

- [ ] **Step 4: Run the full action_classifier test file (regression check)**

```
uv run pytest tests/unit/test_action_classifier.py -v
```

Expected: no new failures. If any tests broke, the symmetry change broke an existing invariant — investigate before proceeding.

---

### Task 4: Verify on F4 rally fixture

**Files:**
- No code changes; verification only.

- [ ] **Step 1: Pull the F4 rally's current action sequence**

```
PGPASSWORD=postgres psql -h localhost -p 5436 -U postgres -d rallycut -At -c "SELECT pt.actions_json FROM player_tracks pt WHERE pt.rally_id LIKE '88a04529%' LIMIT 1;" | python3 -c "
import sys, json
raw = sys.stdin.read().strip()
d = json.loads(raw)
while isinstance(d, str):
    d = json.loads(d)
for a in d.get('actions', []):
    print(f\"frame={a['frame']:>4} type={a['action']:>8} pid={a.get('playerTrackId')} synth={a.get('isSynthetic', False)}\")
"
```

Expected: shows synth serve at f106 + real serve at f111 (pre-fix state).

- [ ] **Step 2: Re-run the rally through the action pipeline**

```
uv run python scripts/redetect_all_actions.py --apply --rally-id 88a04529-<full-uuid>
```

Get the full UUID with: `PGPASSWORD=postgres psql -h localhost -p 5436 -U postgres -d rallycut -At -c "SELECT id FROM rallies WHERE id LIKE '88a04529%';"`.

- [ ] **Step 3: Re-pull and verify**

Re-run the query from Step 1.

Expected: f106 still SERVE (synth), f111 now RECEIVE.

If f111 is still SERVE, the pipeline is using a cached classifier output or the rule didn't apply — investigate via logs.

- [ ] **Step 4: Bump `ACTION_PIPELINE_VERSION`**

Per `analysis/CLAUDE.md` "Post-classifier-change checklist", any change to `action_classifier.py` that alters `RallyActions.to_dict()` output requires bumping `ACTION_PIPELINE_VERSION` in the same commit.

In `action_classifier.py`, find `ACTION_PIPELINE_VERSION = "v<N>"` near the top of the file (grep `ACTION_PIPELINE_VERSION` if you can't spot it). Bump to the next integer, and add a history-comment line:

```python
# v<N+1> (2026-05-13): Rule 3 dedupes synth+real serve pairs symmetrically (WS-B).
ACTION_PIPELINE_VERSION = "v<N+1>"
```

Pre-commit hook will reject the commit if you don't bump.

---

### Task 5: Commit and refresh the fleet

- [ ] **Step 1: Commit the rule change + version bump**

```
git add analysis/rallycut/tracking/action_classifier.py analysis/tests/unit/test_action_classifier.py
git commit -m "$(cat <<'EOF'
feat(repair-rule-3): symmetric dedupe over synthetic-real serve pairs

Rule 3 was only counting non-synthetic serves as the dedupe anchor,
allowing F4 pattern (synth-prepended + real classifier serve) to
slip through as a double-serve. Make Rule 3 symmetric: the first
serve found (regardless of synth flag) is the anchor; subsequent
serves become receive.

Bumps ACTION_PIPELINE_VERSION.

Spec: docs/superpowers/specs/2026-05-13-action-attribution-root-causes-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 2: Refresh the fleet**

```
uv run python scripts/redetect_all_actions.py --apply
```

Expected: per-rally streaming logs, then a final tally. F4 rallies should report Rule 3 trigger.

- [ ] **Step 3: Re-run the C-4 catalog regen (sanity)**

```
uv run python scripts/catalog_c4_violations.py
```

Expected: F4 rally `88a04529` should no longer appear (or appear with fewer violations) in the new `2026-05-13_post_redetect.csv`. Compare to the previous post-redetect catalog (already in `analysis/reports/coherence_c4_catalog/2026-05-13_post_redetect.csv`).

If the C-4 count went UP, the symmetric rule introduced a regression — investigate.

---

### Task 6: Update memory index

**Files:**
- Modify: `MEMORY.md` — one index line.
- Create: `f4_dedupe_symmetry_2026_05_13.md` topic file (memory dir).

- [ ] **Step 1: Create the topic file**

`/Users/mario/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/f4_dedupe_symmetry_2026_05_13.md`:

```markdown
---
name: f4-dedupe-symmetry-2026-05-13
description: "Rule 3 in repair_action_sequence now dedupes synth+real serve pairs symmetrically. Pre-2026-05-13 it only counted non-synthetic serves as the anchor, letting F4 pattern slip through. WS-B of action-attribution-root-causes-2026-05-13 design."
metadata: 
  type: project
---

# F4 Dedupe Symmetry — SHIPPED 2026-05-13

Single-rule extension at `action_classifier.py` Rule 3 block.
Replaces the `not a.is_synthetic` anchor check with an unconditional
"first serve found is the anchor" rule. ACTION_PIPELINE_VERSION
bumped; fleet refreshed.

Composes with [[synthetic_serve_placement_v11_2026_05_10]] and
[[serve_peak_prepend_v13_2026_05_11]] — the prior workstreams that
created the synth+real co-occurrence path.

## Files

- `analysis/rallycut/tracking/action_classifier.py` (Rule 3 block)
- `analysis/tests/unit/test_action_classifier.py`
```

- [ ] **Step 2: Add MEMORY.md index line**

```
- [SHIPPED] [**F4 dedupe symmetry 2026-05-13**](f4_dedupe_symmetry_2026_05_13.md) — Rule 3 anchor check is now synthetic-agnostic; synth+real serve pairs reclassify the second as receive. ACTION_PIPELINE_VERSION bumped; fleet refreshed.
```

---

## Ship gate verification

- [ ] F4 rally `88a04529` re-runs cleanly: f106 SERVE (synth), f111 RECEIVE.
- [ ] Existing non-synth dedupe behavior unchanged (Task 3 Step 4 full file passes).
- [ ] Fleet C-4 catalog regen shows no regression.
