# Phase B rally-start serve prior — NO-GO

**Date:** 2026-04-23
**Hypothesis:** Per the 2026-04-22 game-semantics scoping plan, a rescue branch that accepts serve-like candidates in the first 90 frames would recover the 86 remaining serve FNs (under production-aligned eval with synthetic-serve matching enabled).
**Result:** **NO-GO.** 4 / 4 pre-registered gates FAIL. Catastrophic serve F1 regression (−21.8 pp).

## Gate results

| Gate | Threshold | Baseline (synth) | Phase B (synth) | Δ | Status |
|---|---|---:|---:|---:|---|
| Contact F1 | ≥ 0 pp | 89.8 % | 86.8 % | **−3.0 pp** | **FAIL** |
| Action Acc | ≥ 0 pp | 91.3 % | 87.5 % | **−3.8 pp** | **FAIL** |
| Serve FN reduction | ≥ 15 abs | FN=86 | FN=159 | **+73** | **FAIL** |
| FP increase | ≤ 20 abs | FP=164 | FP=283 | **+119** | **FAIL** |

All four. Per the anti-rationalization rule: no retune.

## Per-class F1 (signed Δ)

| Class | Baseline | Phase B | Δ |
|---|---:|---:|---:|
| **serve** | 78.8 % | **57.0 %** | **−21.8 pp** |
| receive | 81.2 % | 71.3 % | −9.9 |
| set | 86.2 % | 84.7 % | −1.5 |
| attack | 89.0 % | 88.3 % | −0.7 |
| dig | 72.7 % | 72.5 % | −0.2 |
| block | 10.8 % | 10.8 % | 0.0 |

## Mechanism of failure

The pre-registered rescue condition:

```
if frame - first_frame < 90
   AND direction_change_deg < 30
   AND _ball_sustained_on_one_side(..., window=30, min_samples=10)
then accept even if GBM + stepback rejected
```

**Three design flaws were exposed at corpus scale:**

1. **The 90-frame window covers ~2-3 s, long enough that many receives (the contact AFTER the serve) fall inside it.** When GBM correctly rejected a receive-like candidate, this rescue over-rode. Receive FP +76 confirms.

2. **`_ball_sustained_on_one_side` is too permissive after a ball landed.** The check is "ball on one side for 30 f with ≥ 10 samples". Once the serve ball lands on the receive side, it stays there until receive contact — so any candidate with ball-on-receive-side-30-f qualifies. Mid-rally candidates (not serves) pass the gate.

3. **`direction_change_deg < 30` accepts noise-level candidates.** Serves DO have low direction change (ball moving in a nearly-straight line toward receive), but so does ball-in-mid-flight before any real contact. The GBM was correctly rejecting these as not-a-contact; the rescue overrides that rejection.

The cascading effect: rescued-false-serves displaced real serves in `classify_rally_actions` (which picks the first contact as serve). Serve TP dropped 278 → 205 (−73). The false rescues got labeled "serve" on first contact, so real serves got labeled "receive" downstream, triggering duplicate-receive repair rules, distorting the whole action chain.

## Code REMOVED (2026-04-23 cleanup)

Dormant env-flag-gated code was deleted during end-of-session housekeeping. This memo is the record of the attempt. If re-opened, the rescue logic (`_ball_sustained_on_one_side` helper + rally-start rescue wrapper + wiring) is straightforward to rebuild from the spec above — but don't re-open without a fundamentally better hypothesis. The mechanism failed on 4/4 pre-registered gates.

Artifacts: `reports/loo_phase_b_synth.{md,json}` — A/B data.

## What NOT to do as follow-up

- **Do not** narrow the window from 90 → 60 or 45 to "fit gate". The mechanism itself is flawed: a blunt OR-gate on GBM rejection was never going to work when the GBM's rejections are mostly correct.
- **Do not** tighten `direction_change < 30` to `< 20` or `< 15`. Same reason. The per-class damage spreads across 5 of 6 classes — it's not a threshold issue.
- **Do not** add a ball-velocity or ball-height filter. We'd be reinventing what the GBM already does better.

## What the mechanism SHOULD look like (if re-opened)

Two principled alternatives, both significantly bigger than a rescue branch:

1. **Serve-specific GBM head.** Train a separate classifier on first-90-frame candidates with serve-aware features (ball-in-server-toss window, server formation, serve-position-from-server). Treat serve detection as a separate problem from mid-rally contacts. ~1 week.

2. **Pose-based serve detection.** The server's unique motion (toss-reach-strike) has a stereotyped pose signature that generic contact pose features miss. Fine-tune or feature-engineer on serve poses specifically. ~1-2 weeks.

Neither is class-local or cheap. Both should only be opened after the next-best lever is exhausted.

## Residual FN budget (production-aligned)

Under `--include-synthetic` baseline:

- **serve**: 86 FN (33.9 % of total)
- **attack**: 91 FN (35.8 %)
- set: 62 FN (24.4 %)
- receive: 61 FN (24.0 %)
- dig: 87 FN (34.3 %)
- block: 28 FN (11.0 %)

**Dig is now at 87 FN, very close to serve's 86.** Serve is no longer the single largest lever. The 86-FN gap is real but no cheap mechanism addresses it without the 1-2 week investment above.

## Next-best levers (revised)

1. **Attack F1 is 89.0 % with 91 FN and 28 FP.** Already the best class. Residual gap is small; not high-ROI to chase.
2. **Dig improvement.** Dig F1 = 72.7 %, 87 FN, 92 FP — 4th-lowest F1 after serve/block. Worth an audit but no obvious single lever from today's data.
3. **Block remains a known data-starvation problem.** 28 FN and F1 at 10.8 %. Needs labeled training data, not a rescue branch.
4. **Ball-tracker recall gains** (user-rejected retrain path) — would help multiple classes simultaneously.

None of these have the 2-4 hr quick-win signature that Phase B and the radius-loose attempt promised (and failed). The v5 baseline at 89.8 % Contact F1 / 91.3 % Action Acc (with synth enabled) may be near the local maximum for this architecture.

## Commits / artifacts

- Code: un-committed (dormant env-flag-gated implementation in `contact_detector.py`).
- Reports: `analysis/reports/loo_phase_b_synth.{md,json}`, `analysis/reports/phase_b_rally_start_serve_prior_nogo_2026_04_23.md`.
- Memory: will be added to `memory/` and indexed in `MEMORY.md`.
