# `player_contact_radius` loose (0.15 → 0.20) — NO-GO

**Date:** 2026-04-23
**Hypothesis:** per `memory/action_audit_2026_04_20.md`, 88 % of `no_player_nearby` FNs (57 / 291 FN pool, ~20 %) are seq-endorsed; loosening `player_contact_radius` 0.15 → 0.20 at candidate-generation sites would recover them while the existing `_apply_rescue_branch` (gbm<0.10 AND seq≥0.95) gates FPs.
**Result:** **NO-GO.** 3 / 4 pre-registered gates FAIL on the 68-fold LOO A/B.

## Gate results

| Gate | Threshold | Baseline (r=0.15) | Treatment (r=0.20) | Δ | Status |
|---|---|---:|---:|---:|---|
| Contact F1 Δ ≥ 0 pp | aggregate | 89.0 % | 88.9 % | −0.1 | **FAIL** |
| Action Accuracy Δ ≥ 0 pp | aggregate | 91.7 % | 91.6 % | −0.1 | **FAIL** |
| FP increase ≤ 15 abs | aggregate | 159 | 166 | +7 | PASS |
| FN reduction ≥ 30 on no_player_nearby | subset | — | — | aggregate FN −1 only | **FAIL** |

Per the pre-registered anti-rationalization rule, any gate fail closes the workstream. Not tuning to the gate.

## Per-class breakdown (signed Δ)

| class | F1 baseline | F1 treatment | Δ F1 | TP Δ | FP Δ | FN Δ |
|---|---:|---:|---:|---:|---:|---:|
| serve | 71.7 % | **72.3 %** | **+0.6** | +5 | +3 | −5 |
| receive | 82.7 % | 82.9 % | +0.2 | +1 | 0 | −1 |
| set | 87.2 % | 87.2 % | 0.0 | −2 | −2 | +2 |
| attack | 89.4 % | 88.6 % | **−0.8** | −3 | +6 | +3 |
| dig | 74.7 % | 74.3 % | −0.4 | −2 | 0 | +2 |
| block | 5.0 % | 4.8 % | −0.2 | 0 | +2 | 0 |

## Failure mechanism

The audit's core claim (seq-conjunctive rescue catches most loose-radius candidates) held for **serves only**. Serves gained +5 TP / −5 FN with only +3 FP — a real win (recall +1.4 pp).

The hypothesis did NOT hold for other classes:
- **Attack regressed (−0.8 pp F1)**: loose radius generated more proximity candidates in attack-adjacent frames that the GBM accepted as attacks but were actually spurious. Each attack FP is costly — attack precision dropped 1.2 pp.
- **Dig regressed (−0.4 pp F1)**: smaller than attack but net-negative.
- **Set tied**: TP and FP both dropped equally.

Root cause: the `_apply_rescue_branch` is *conjunctive* (gbm<0.10 AND seq≥0.95), so it rescues only hard-reject candidates. Most new proximity candidates introduced by the 0.20 radius hit the GBM at moderate confidence where the rescue branch doesn't apply — and the GBM accepts them at inflated rates because more candidates per-frame dilutes the precision/recall tradeoff.

Net: the audit was right that serves are recoverable, wrong that the lever is a uniform radius loosening.

## Code REMOVED (2026-04-23 cleanup)

Dormant env-flag-gated code was deleted during end-of-session housekeeping. This memo is the record of the attempt. If re-opened, rebuild minimally: add `player_contact_radius_loose: float` config field + 3 site substitutions (post-serve receive, proximity, fallback `has_player`) behind an env flag.

Artifacts: `reports/loo_baseline_radius015.{md,json}` + `reports/loo_loose_radius020.{md,json}` + `reports/loo_baseline_radius015_synth.{md,json}` + `reports/loo_loose_radius020_synth.{md,json}` — A/B data across no-synth / synth-enabled regimes.

## What NOT to do as follow-up

- **Do not** sweep intermediate radii (0.16, 0.17, 0.18) hoping to split the serve-gain / attack-loss tradeoff. The per-class asymmetry suggests the signal is class-specific, not magnitude-specific.
- **Do not** add a seq-conjunctive hard gate to the loose-radius branch. That would collapse the serve gains too (serves also have moderate-confidence candidates the rescue branch doesn't fire on).
- **Do not** ship flag-enabled on any video subset. The gains don't survive aggregate measurement.

## What the result actually suggests

**Serves are the recoverable class** (65.1 % baseline recall, 127 FN in the 364-rally corpus). A serve-specific prior — the original plan's **Phase B: rally-start serve prior** — is the right next lever:

- When `frames_since_rally_start < 90` (3 s at 30 fps) AND candidate has `direction_change_deg < 30` AND ball on one side for ≥ 1 s before candidate → boost serve-candidate acceptance.
- Doesn't touch attack / dig / set paths → no per-class regression risk.
- Doesn't depend on ball-side classification → independent of today's Probe 2 game-semantics NO-GO.
- ~2 hr implementation.

Phase B is the recommended next workstream.

## Audit trail

- Pre-registration memo: not written separately — gates inline in today's verbal plan, preserved in `MEMORY.md` entry when updated.
- Implementation commit: un-committed (dormant code; will be committed as part of the memo).
- Audit source cited: `memory/action_audit_2026_04_20.md`.
