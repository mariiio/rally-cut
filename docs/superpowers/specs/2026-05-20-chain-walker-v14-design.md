# Chain-Walker v14 (Class B) — Design 2026-05-20

## TL;DR

Three convergent diagnostics ([[upstream_bottleneck_findings_2026_05_20]],
[[chain_asymmetry_findings_2026_05_20]], and the GT-derived net-crossing
probe at commit `e41bfa1c`) pinpoint the team-chain walker
(`_compute_expected_teams` at `action_classifier.py:2968`) as the top
attribution-accuracy lever. Walker is correct ~93% of the time on
net-crossing decisions but has 64 missed flips + 20 over-flips out of
1265 GT-derived pairs.

This spec scopes a flag-gated v14 walker update with two interventions:

- **B.1 — BLOCK conditional flip**: replaces "always stay on BLOCK"
  with "flip if next contact's court_side differs". Empirically
  classifies 17/17 prev=block events correctly on trusted-32 GT.
- **B.2 — Ball-trajectory verifier**: for any consecutive action pair,
  if `contacts[i].courtSide != contacts[i+1].courtSide`, override the
  rule-based decision with physical truth. Robust to upstream action
  mis-classification — addresses many of the 51 cases where the walker
  was rule-correct but action-type-wrong.

Both gated behind separate env flags (default OFF in v14 → byte-identical
to v13). Validation is a 2×2 A/B grid on trusted-31 + a pre-A/B
walker-decision-accuracy replay on the 1265 GT-pair dataset.

Target ceiling: L6 oracle is 35 attribution recoveries on trusted-31.
Realistic ceiling: depends on what fraction of walker-decision wins
translate to attribution wins.

## Background

### Why this design now

Sub-lever 1 ([[attribution_sub_lever_1_no_ship_2026_05_20]]) NO-SHIP'd
because its audit projected +28 violation recovery but realistic
intervention delivered +4 (projection trap: rank ≠ confidence-leader).
The L6 oracle in [[upstream_bottleneck_findings_2026_05_20]] is similarly
oracle-conditioned (perfect chain). To avoid the same trap, this spec
mandates dual-ceiling validation: a CHEAP walker-decision replay against
the 1265-pair GT dataset (deterministic, no scorer interaction), and an
END-TO-END A/B that measures actual attribution accuracy lift.

### GT-derived net-crossing probe — what we learned

`probe_gt_net_crossings_2026_05_20.py` (commit `e41bfa1c`) walks GT
actions in frame order per trusted-32 rally. For each consecutive pair,
infers whether ball crossed (`team_prev != team_curr` per
`teamAssignments`).

| Walker decision | Count | Rate |
|---|---:|---:|
| Correct flip (rule says flip, ball crossed) | 482 | 88.3% of 546 crossings |
| **Missed flip** (rule says stay, but ball crossed) | **64** | 11.7% |
| Correct stay (rule says stay, ball didn't cross) | 699 | 97.2% of 719 non-crossings |
| **Over flip** (rule says flip, but ball didn't cross) | **20** | 2.8% |

Missed flip breakdown by `prev_action`:

| prev_action | n | Likely cause |
|---|---:|---|
| set | 20 | Mostly `set→attack` — pipeline classified "set" but actually attack (dump / 2-touch over) |
| dig | 17 | Mostly `dig→attack` — pipeline classified "dig" but actually overpass/dig flat |
| receive | 14 | Receive that went over the net (free ball / overpass receive) |
| block | 13 | Block-cover deflection that crossed |

Over-flip breakdown:

| prev_action | n | Likely cause |
|---|---:|---|
| attack | 11 | `attack→dig same-team` — block-cover after own attack got blocked back |
| serve | 9 | `serve→receive same-team` — anomalies (likely team_assignments edge cases) |

BLOCK split: 17 prev=block events total, 13 crossed and 4 stayed.
Per-curr-action breakdown:

| curr_action | crossed | stayed | next_side rule | Verdict |
|---|---:|---:|---|---|
| dig | 12 | 0 | flip | ✓ 12/12 |
| attack | 1 | 3 | conditional on next-contact side | empirically separates cleanly |
| set | 0 | 1 | stay | ✓ 1/1 |

**Using `next_contact.courtSide` as the BLOCK rule classifies 17/17
correctly.** No ball trajectory needed for the BLOCK case specifically.

## Architecture

### Current walker (v13)

`analysis/rallycut/tracking/action_classifier.py:_compute_expected_teams`
(line 2968-3015):

```python
_NET_CROSSING_ACTIONS = {ActionType.SERVE, ActionType.ATTACK}

current_team = serve_team
for i, action in enumerate(actions):
    ...
    expected[i] = current_team
    if action.action_type in _NET_CROSSING_ACTIONS:
        current_team = 1 - current_team
```

### v14 walker

Decision function replaces inline rule:

```python
def _possession_flips_after(
    action: ClassifiedAction,
    next_action: ClassifiedAction | None,
    contacts: list[Contact],
    config: ChainWalkerConfig,
) -> bool:
    """Whether possession flips after this action."""
    rule_says_flip = action.action_type in _NET_CROSSING_ACTIONS

    # B.1: BLOCK conditional — next-contact court_side signal
    if config.block_conditional and action.action_type == ActionType.BLOCK:
        curr_side = _contact_side_at(contacts, action.frame)
        next_side = (_contact_side_at(contacts, next_action.frame)
                     if next_action is not None else None)
        if curr_side and next_side:
            rule_says_flip = (curr_side != next_side)

    # B.2: ball-trajectory verifier — physical truth overrides rule
    if config.ball_trajectory_verifier and next_action is not None:
        curr_side = _contact_side_at(contacts, action.frame)
        next_side = _contact_side_at(contacts, next_action.frame)
        if curr_side and next_side:
            return curr_side != next_side  # override

    return rule_says_flip
```

Where `_contact_side_at(contacts, frame)` finds the contact within ±3
frames of `frame` and returns its `court_side`. Returns `None` when
no matching contact (verifier degrades to rule).

### Env flags

| Flag | Default v14 | Behavior when set |
|---|---|---|
| `WALKER_BLOCK_CONDITIONAL` | `"0"` (off) | Enables B.1 |
| `WALKER_BALL_TRAJECTORY_VERIFIER` | `"0"` (off) | Enables B.2 |

Both OFF → byte-identical to v13. Production initial deploy: both off
in v14 stamp. After A/B, winning combination becomes the production
default in a separate v15 commit.

## Validation

### Stage 1 — Walker-decision accuracy replay (fast, ~1 min)

`analysis/scripts/measure_walker_accuracy_2026_05_20.py`: replays each
of 4 flag configurations against the 1265-pair GT dataset. Per config:

- `correct_flip` / `missed_flip` / `correct_stay` / `over_flip` counts
- Compared against v13 baseline (482 / 64 / 699 / 20)

This is the cheap, deterministic ceiling. No scorer interaction; no
attribution measurement. Tells us EXACTLY how many walker decisions
change under each config.

Pre-A/B gate: if a config doesn't improve walker decisions on the
GT-pair dataset, skip the end-to-end A/B for it.

### Stage 2 — End-to-end attribution A/B (slow, ~60-90 min)

`analysis/scripts/sub_lever_2_ab_2026_05_20.sh`: 4 redetect cycles on
trusted-31 (one per flag combination), 4 measurements via
`measure_attribution_trusted_31_2026_05_20.py`, comparison vs v13
baseline.

| Cfg | B.1 | B.2 |
|---|---|---|
| 00 | OFF | OFF |
| 10 | ON | OFF |
| 01 | OFF | ON |
| 11 | ON | ON |

Configs 00 must produce byte-identical results to v13 (sanity check).

### Ship gate

A configuration ships if all three:

1. **Walker-decision accuracy strictly higher** than v13 (`(correct_flip
   + correct_stay) > 482 + 699 = 1181`).
2. **End-to-end attribution accuracy** ≥ v13 baseline − 0.5pp (net
   non-regressive).
3. **No protected-video regression** ≥ 2 contacts (mumu, keke, mame,
   veve, papa — same 5-video gate from Sub-lever 1).

Ship the highest-accuracy configuration that passes all three. If none
passes: NO-SHIP, document findings, scope chain-quality classifier
(Class C from earlier brainstorm) as next attempt.

## File structure

| Path | Status | Responsibility |
|---|---|---|
| `analysis/rallycut/tracking/action_classifier.py` | MODIFY (`2968-3015`) | Add `_possession_flips_after`, `_contact_side_at`, `ChainWalkerConfig`; refactor `_compute_expected_teams`; bump `ACTION_PIPELINE_VERSION` to `v14` |
| `analysis/tests/unit/test_chain_walker_v14.py` | CREATE | Unit tests for the helper functions and per-flag behavior |
| `analysis/scripts/measure_walker_accuracy_2026_05_20.py` | CREATE | Stage-1 replay against 1265-pair GT dataset |
| `analysis/scripts/sub_lever_2_ab_2026_05_20.sh` | CREATE | Stage-2 end-to-end A/B driver |

## Decision tree from A/B outcomes

| Outcome | Action |
|---|---|
| All 3 non-trivial configs PASS, one is clearly best | Flip its env to default ON in v15, update `[[upstream_bottleneck_findings]]` to SHIPPED. |
| Only B.1 passes | Ship B.1-only as v15 default. B.2 stays as opt-in for future revisit. |
| Only B.2 passes | Ship B.2-only as v15. B.1 stays opt-in (the BLOCK rule turns out to be subsumed by ball-trajectory verifier). |
| All three pass but small lift (<5 attribution wins) | Ship the simplest passing config (likely B.1). Note residual ceiling in findings memo; scope Class C if user wants further investment. |
| None passes | NO-SHIP. Per plan Phase 9 in the prior Sub-lever 1 pattern: revert wiring, keep helpers as documented infrastructure, write a NO-SHIP memo with realistic-ceiling vs oracle-ceiling delta as the takeaway. Then scope Class C. |

## Risks

- **`courtSide` data quality.** `Contact.courtSide` is set by the
  contact detector based on ball position relative to `court_split_y`.
  Errors in court calibration or net detection would propagate. The v8
  net-top regression (recent SHIPPED per memory) addresses a chunk of
  this; pre-validation: spot-check `_contact_side_at` returns None
  rate on trusted-31.
- **Synthetic contacts.** `Contact.courtSide` for synthetic serves is
  set via `_interpolate_ball_position_for_synthetic`. If interpolated
  ball position lands on the wrong side, verifier flips incorrectly.
  Mitigate: in v14, the verifier returns None (degrades to rule) when
  EITHER contact is synthetic. Document in spec.
- **Sub-lever 1 projection-trap recurrence.** Walker-decision lift may
  not translate 1:1 to attribution lift. Mitigation: dual-ceiling
  validation, gate on attribution accuracy not walker accuracy.

## Out of scope

- **Action-classifier improvements** for the 51 set→attack /
  dig→attack / receive→over mis-classifications. WS-2 from the original
  four-prong campaign — separate workstream.
- **Synthetic-serve team re-derivation** (26 H3 cases from chain
  asymmetry diagnostic) — separate workstream.
- **Probabilistic walker / chain-quality classifier (Class C)** — only
  scoped if Class B underdelivers.
- **VLM probe** — out per user constraint.

## Related

- [[upstream_bottleneck_findings_2026_05_20]] — identified L6 chain
  accuracy as #1 lever (oracle 35 recoveries).
- [[chain_asymmetry_findings_2026_05_20]] — H1+H2 rejected, walker
  logic implicated (62% unexplained).
- GT-derived net-crossing probe at commit `e41bfa1c` —
  `analysis/reports/gt_net_crossings_2026_05_20/`.
- [[attribution_sub_lever_1_no_ship_2026_05_20]] — prior NO-SHIP that
  introduced the dual-ceiling methodology in response to the
  projection trap.
- `_compute_expected_teams` at
  `analysis/rallycut/tracking/action_classifier.py:2968` — the code to
  modify.
