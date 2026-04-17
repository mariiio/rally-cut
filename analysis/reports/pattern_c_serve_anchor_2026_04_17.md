# Pattern C (rally-start serve anchor) validation — 2026-04-17

> **Status:** NO SHIP. Code reverted in the same session after dormant ship proved unnecessary. Kept as the decision record so future work doesn't re-investigate the same hypothesis without addressing the follow-up notes at the bottom.

## Context

Corpus audit at commit 65f4ca2 identified 11 early-rally serve FNs (`gt_frame < 90`, `ball_present=False`, `ball_gap_frames ≥ 4`, no candidate within reasonable distance of GT) where the WASB ball tracker had not yet locked onto the ball by the time the serve occurred. Pattern A's rescue gate can't help — there is no trajectory candidate to rescue — so a separate mechanism is needed.

## Design

`rallycut/tracking/contact_detector.py` — added `_maybe_anchor_rally_start_serve`. After dedup and before returning the `ContactSequence`, the helper synthesizes a serve `Contact` at the MS-TCN++ serve-class peak frame when **all** of these hold:

1. No existing validated contact lives in `[first_frame, first_frame + SERVE_ANCHOR_MAX_FRAME)` (90 frames ≈ 1.5s at 60fps).
2. `sequence_probs.shape[0] >= 7` (the full MS-TCN++ 7-class output — background + 6 actions; guards against 2-row test fixtures triggering the anchor).
3. `sequence_probs[1, :].max()` in the window ≥ `SERVE_ANCHOR_TAU`.
4. Some confident ball detection exists within `±SERVE_ANCHOR_MAX_FRAME` of the peak (otherwise the helper refuses to synthesize a `(0,0)` contact that would corrupt downstream features).
5. Nearest player at the peak frame is within 0.20 of the ball (loose radius; serve players can be a bit far).

The synthesized contact inherits the ball's position at the peak (or nearest neighbor), attributes to the nearest player, and sets `is_validated=True` with `confidence=peak_val` so it flows through `classify_rally_actions` like any other contact. Rationale: rally invariant ("every rally begins with a serve") + MS-TCN++ localization + ball-position guard is the strongest independent signal available when the trajectory pipeline has no candidate.

## Validation

Anchor on (`SERVE_ANCHOR_TAU=0.85`) vs anchor off (`SERVE_ANCHOR_TAU=1.1` — unreachable), Arm B disabled in both, full 364-rally corpus rebuild.

| Metric | Anchor off | Anchor on | Δ |
|---|---:|---:|---:|
| TP | 938 | 935 | **−3** |
| FN | 395 | 394 | −1 |
| wrong_action | 170 | 176 | **+6** |
| wrong_player | 595 | 593 | −2 |
| extra_pred | 286 | 304 | **+18** |
| n_rallies_regressed | — | 12 | — |
| Late-track-start rescues | — | 3/11 | — |

## Decision rule (pre-registered)

Ship if **all** of: ≥6 of the 11 late-track-start FNs rescued AND `Δextra_pred ≤ 3` AND no rally regresses.

Every hard constraint fails: only 3/11 targets rescued, `Δextra_pred=+18`, 12 rallies regress.

### Verdict: **NO SHIP (DORMANT)**

Root cause: MS-TCN++'s serve-class channel has meaningful baseline activity throughout a rally (mid-rally contacts sometimes score >0.85 on the serve class due to feature similarity with hard-hit contacts), so anchoring at the peak inside the first 90 frames fires for more than just true late-start serves. The 18 new extra_preds and 6 new wrong_action errors bear this out — the anchor is injecting serves in rallies where the existing first contact was already correctly classified.

Secondary issue: when the anchor DOES fire at the right rally, the ball position at the peak frame is often missing (ball tracker ramp-up), so `_maybe_anchor_rally_start_serve` refuses rather than synthesizing a `(0,0)` contact. That refusal path is correct — we would rather miss a serve than fabricate one with broken features — but it limits the recall ceiling.

`rallycut/tracking/sequence_action_runtime.py` sets `SERVE_ANCHOR_TAU = 1.1` (unreachable) as the module default. The helper, constants, unit tests, and validation baseline stay in place for future retuning.

## Follow-up directions

Two precision-raising changes could make the anchor shippable:

1. **Require serve-class peak prominence, not just magnitude.** A peak ≥ 0.85 is common; a peak that is 5× higher than the local serve-class median within the window is rare and almost always a real serve. Replace `max >= τ` with `max(window) / median(window) >= prominence_τ`.
2. **Gate on ball-trajectory shape, not just presence.** Require the first confident ball trajectory post-peak to follow a downward-then-upward arc consistent with a served ball, not an arbitrary continuation of mid-rally play. This adds a physics constraint that separates a ramp-up-start serve from a mid-rally spike falsely classified as serve.

## Files

- Implementation: `rallycut/tracking/contact_detector.py` — `_maybe_anchor_rally_start_serve` at line ~1468, wired after dedup in `detect_contacts`.
- Constants + dormant rationale: `rallycut/tracking/sequence_action_runtime.py:379-395`.
- Tests: `tests/unit/test_contact_detector.py` — `TestRallyStartServeAnchor` (5 tests).
- Validation harness: `scripts/analyze_sweep_and_pattern_c.py` (shared with Pattern A).
- Snapshots: `outputs/action_errors/sweep/baseline_no_armb_no_anchor/` vs `outputs/action_errors/sweep/baseline_no_armb_anchor_only/`.
