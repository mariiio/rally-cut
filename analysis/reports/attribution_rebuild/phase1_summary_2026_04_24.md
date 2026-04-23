# Phase 1 — Primitive Audit: Summary

**Date:** 2026-04-24
**Scope:** 9 fixtures / 69 rallies / 468 GT actions (locked baseline).
**Plan:** `docs/superpowers/plans/2026-04-24-attribution-primitive-first.md`

## Audits completed

| Phase | Audit | Gate | Result | Verdict |
|---|---|---|---|---|
| 1.1 | Tracking-ID stability | ≥95% (retroactively ≥70%) | 55.1% | ⚠️ FAIL — policy response in place |
| 1.2 | Side-switch reliability | ≥98% | 100.0% | ✅ PASS |
| 1.3 | Team→side (identity-first vs positional) | ≥98% | 26.1% | ⚠️ FAIL — Phase 2 validates at contact-time |
| 1.4 | Ball court-side best method | ≥95% | 70.9% (Method A) | ⚠️ FAIL — treat as soft evidence only |
| 1.5 | Composite partition | — | 117 chooser-fixable / 43 primitive-fixable / 71 detection-limit / 31 irreducible | Phase 2 headroom quantified |

## Phase 2 handoff — three policies baked in

Rather than block Phase 2 behind upstream stage-2 fixes (memory documents multiple NO-GO'd ML attempts), the Phase 2 confidence-gated chooser must incorporate these **primitive-aware policies**:

### Phase 1.1a — Swap-aware abstention

Input: `reports/attribution_rebuild/phase1_1_tracking_stability.json` (`swap_events` + `coverage` per rally).

- If a candidate tid has a swap event within ±5 frames of a contact → chooser abstains (`player_track_id = None`).
- If a candidate tid has < 40% rally coverage → chooser abstains.
- Converts 25 within-rally swap events + 13 low-coverage tid cases from `wrong_rate` to `missing_rate` — aligned with "prefer miss over wrong".

### Phase 1.3a — Contact-time team validation

Input: `reports/attribution_rebuild/phase1_3_team_side.json` (per-rally positional pairing vs `teamAssignments`).

- Chooser does NOT trust `teamAssignments` for team-membership inference.
- At contact frame, compute positional pairs (2 nearest players together vs 2 farthest) and determine "team of candidate" from this local signal, not from stored team letter.
- Captures the 38 `team_pair_wrong` primitive errors without upstream stage-2 fix.
- Label flips (A↔B swapped uniformly) remain on the `servingTeam` / side-switch audit; these are cosmetic for attribution.

### Phase 1.4a — Ball-side as soft evidence

Input: `reports/attribution_rebuild/phase1_4_ball_court_side.json` (pipeline `courtSide` + trajectory median).

- Ball-side agreement is a margin multiplier, not a hard filter.
- `~70%` reliability (pipeline Method A) makes it a tiebreaker at best.
- Consistent with plan §3.3 "soft volleyball-rule priors — never hard filters".

## Phase 2 target

- **Baseline**: 44.0% correct / 17.7% wrong / 15.4% missing (468 actions).
- **Phase 2 kill gate** (plan §2): `wrong_rate ≤ 10%` (halve baseline's 17.7% → `≤ 8.85%`).
- **Expected move under above policies + confidence-gated chooser:**
  - Primitive-aware abstention converts ~31 wrongs → missing.
  - Contact-time team validation captures ~38 cross_team primitive fixables → correct.
  - Confidence gate on remaining 117 chooser_fixable converts ~60-80 → correct or abstain.
- **Ceiling at Phase 2 exit**: ~60-65% correct / ~8-10% wrong / ~25-30% missing.

## Artifacts emitted

- `reports/phase1_1_tracking_stability_2026_04_24.md` + `.json`
- `reports/phase1_2_side_switch_2026_04_24.md` + `.json`
- `reports/phase1_3_team_side_2026_04_24.md` + `.json`
- `reports/phase1_4_ball_court_side_2026_04_24.md` + `.json`
- `reports/phase1_5_composite_partition_2026_04_24.md` + `.json`

## Open follow-ups (non-blocking, logged)

- **wawa** (5 dropouts in 5 rallies) — systematic fragmentation investigation.
- **tata** (2 rallies with only 3 primary tids) — occlusion at serve; not fixable upstream.
- **cuco + lulu** (heavy swap concentration) — candidates for manual `swap-tracks` CLI if accuracy ceiling reached in Phase 3.
- **teamAssignments pair bug** root cause in `match-players` — left in place; Phase 1.3a provides a read-side workaround. Could be upstream-fixed as a separate workstream.
