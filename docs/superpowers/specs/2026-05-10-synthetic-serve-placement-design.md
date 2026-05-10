# Synthetic-Serve Frame Placement (v1.1)

**Date:** 2026-05-10
**Status:** Design — pending implementation plan
**Workstream context:** Sub-2.B (replaces the original "coherence-driven contact recovery v1" which shipped scaffolding only and is being removed). Targets the highest-leverage FN bucket on the panel: **mis-placed synthetic serves**.

## Goal

When the action pipeline cannot detect a real serve and falls back to synthesizing one via `_make_synthetic_serve`, place that synthetic serve at the **frame the actual serve happened**, using two independent signals (MS-TCN++ serve-class peak + ball-trajectory burst), instead of the current placeholder formula `first_contact_frame - 30`.

Default-on production change. Behavior-preserving when neither signal is available (falls back to current logic).

## Motivation

### Fleet diagnostic, 2026-05-10

Across all 66 videos with action GT (338 rallies, 313 with a pred serve, 335 with a GT serve):

| Serve type | Count | Mis-placed (>15 frames off GT) | Hit rate |
|---|---:|---:|---:|
| Real (detected) serves | 218 | 24 | **89%** |
| Synthetic serves | **95** | **51** | **46%** |

(For reference — the 5-video panel + 073cb11b subset showed 14/15 real hits and 4/10 synthetic hits, consistent with the fleet rates within sampling noise.)

**~22 rallies have no pred serve at all** — the synthesis fallback didn't fire either; investigated separately, out-of-scope for this workstream.

The pipeline is excellent at *detecting* serves when they're detectable. The bottleneck is the synthesis fallback, which uses a placeholder formula:

```python
# rallycut/tracking/action_classifier.py:1447-1454
if (rally_start_frame is not None
    and rally_start_frame < first_contact_frame
    and (first_contact_frame - rally_start_frame) <= 90):
    serve_frame = rally_start_frame
else:
    serve_frame = max(0, first_contact_frame - 30)
```

The "30 frames before first contact" rule has no signal grounding. On `fb7f9c23`, this places the synthetic serve at frame 276 (= 306 − 30) when the actual serve was at frame 154 — **122 frames late**.

### Why mis-placed synthetics double-hurt the metric

A synthetic serve at the wrong frame:
- **Adds a FN** at the GT serve frame (real serve unmatched).
- **Adds an FP** at the synthetic frame (fake contact at a frame that has no GT contact).
- Cascades to downstream stats: `servingTeam`, score tracking, player attribution all use serve frame.

So fixing the placement saves **both** an FN and an FP per affected rally. **Fleet-wide: ~51 FN + ~51 FP recoverable** by this single fix, vastly more than the panel-only target. Conservative estimate: if the helper closes 60% of mis-placed synthetics, that's ~30 FN + ~30 FP recovered across 313 rallies with GT serves.

## Scope

**In scope:**
- New helper `pick_synthetic_serve_frame(*, sequence_probs, ball_positions, rally_start_frame, first_contact_frame, net_y) -> int | None` in `rallycut/tracking/action_classifier.py` (or a sibling module).
- Modify `_make_synthetic_serve` to accept optional `sequence_probs` + `ball_positions` and use `pick_synthetic_serve_frame` when available; keep the current formula as the fallback.
- Plumb `sequence_probs` and `ball_positions` from `classify_rally_actions` (where they're already available) down through `classify_rally` to the two `_make_synthetic_serve` call sites (`action_classifier.py:1894` and `:2377`).
- Default-on: production paths (`track-players`, `analyze actions`) already pass `sequence_probs` per `track_player.py:991-1018`. No flag.
- Unit tests for the new helper covering the four signal-availability cases (both signals available + agree, both available + disagree, only seq peak available, neither available).
- Integration test on `fb7f9c23` asserting that the synthetic serve lands within ±15 of GT serve frame (frame 154 ± 15).
- Panel re-measurement script: report per-rally synthetic-vs-real serve placement vs GT (extends the diagnostic written this session).

**Out of scope:**
- Detecting *whether* a serve was missed (existing logic is correct). Only changing *where* the synthetic goes.
- Improving `_find_post_serve_receive_candidate` or other receive-detection logic (separate workstream — receives are downstream of correctly-placed serves).
- Attribution of the synthetic serve's `playerTrackId` (existing position-based detector is unchanged; the helper only changes the **frame**).
- Re-training the contact GBM or MS-TCN++.
- Recovering missed contacts in mid-rally gaps (audit-clean Pattern B from the recovery diagnostic — separate v1.2).
- Investigation of the 1 mis-placed real serve (`wawa:fb6e37bf`, +142 frames off). Unrelated; deserves its own diagnostic.
- Modifying the C-1/C-2/C-3 coherence audit. (The audit's reads of "first action is serve" continue to work after this change because synthetic serves remain labeled SERVE — they just live at the right frame now.)

## Architecture

### Calibration finding (2026-05-10): Signal B dropped

A calibration pass on 194 correctly-placed real serves across the 66-video GT pool measured trajectory direction-change at serve vs non-serve frames. Distributions overlap heavily:

- Serve frames: median 12.3°, min 0.0°, max 179.2°.
- Pre-serve non-serve frames in the search window: median 175.5°, min 0.0°, max 180.0°.

Direction-change is NOT discriminative for serves in beach volleyball. The deeper reason: the synthesis fallback fires precisely *because* per-frame trajectory signals were weak at the actual serve frame. Adding trajectory-based logic on top is fundamentally limited by the same root cause. Dropped from v1.1 design; revisit only with a different trajectory metric (e.g., velocity envelope, net-crossing detection) in v1.2.

**v1.1 uses Signal A (MS-TCN++ serve-class peak) alone.**

### `pick_synthetic_serve_frame` algorithm

Given `sequence_probs: np.ndarray (NUM_CLASSES, T)`, `rally_start_frame: int`, `first_contact_frame: int`:

1. **Define the search window** as `[max(0, rally_start_frame), first_contact_frame - SEARCH_GUARD]` where `SEARCH_GUARD = 5` (don't pick a frame too close to the next detected contact — that creates dedup artifacts).
2. **Signal A — MS-TCN++ serve peak**: scan `sequence_probs[SERVE_SEQ_INDEX, :]` over the window; record the argmax frame `f_seq` and its probability `p_seq`. Strong signal iff `p_seq >= SERVE_SEQ_FLOOR = 0.50`. (Lower than the typical 0.80 because serves are inherently harder for MS-TCN++ — see `contact_detection_ceiling.md`.)
3. **Decision**: if Signal A is strong, return `f_seq`. Otherwise return `None` (caller falls back to the current `first_contact_frame - 30` placeholder formula).
4. **Sanity cap**: if `f_seq` is more than `MAX_PRESERVE_FRAMES = 150` before `first_contact_frame` (5s+ at 30fps, 2.5s+ at 60fps), clamp to `first_contact_frame - 150`. Long pre-rally windows risk picking up a previous rally's tail.

### `_make_synthetic_serve` signature change

Current:
```python
def _make_synthetic_serve(
    serve_side: str,
    first_contact_frame: int,
    net_y: float,
    rally_start_frame: int | None = None,
    server_track_id: int = -1,
) -> ClassifiedAction
```

New:
```python
def _make_synthetic_serve(
    serve_side: str,
    first_contact_frame: int,
    net_y: float,
    rally_start_frame: int | None = None,
    server_track_id: int = -1,
    sequence_probs: np.ndarray | None = None,
) -> ClassifiedAction
```

Body change: try `pick_synthetic_serve_frame` first when `sequence_probs` is not None. If that returns a frame, use it. Otherwise fall back to the existing formula.

When `pick_synthetic_serve_frame` returns a frame derived from a high-confidence signal, also bump the synthetic's `confidence` from the current `0.4 / 0.55` to `0.6 / 0.7` respectively — the synthetic now has signal-grounded placement, not a placeholder.

(`ball_positions` was originally proposed as the input for Signal B; Signal B was dropped after calibration, so it's no longer needed.)

### Plumbing changes

- `classify_rally_actions` already accepts `sequence_probs`. Add `ball_positions` as a kwarg (or thread `contact_sequence.ball_positions` since `ContactSequence` carries it).
- `classify_rally` (the inner method on `ActionClassifier`) accepts `sequence_probs` if not already; thread to the two `_make_synthetic_serve` call sites.
- Production paths (`track_player.py`, `analyze.py`) already compute and pass `sequence_probs` to `classify_rally_actions`. No additional plumbing in CLI.

### Failure modes (the helper returns None)

- Sequence model unavailable (cached miss in `get_sequence_probs`). Falls back to current formula.
- Both signals weak (no clear seq peak AND no ball-velocity burst). Falls back to current formula.
- Search window collapses (rally_start ≈ first_contact). Falls back.

In all None cases, behavior is identical to today.

## Rollout sequence

1. Land `pick_synthetic_serve_frame` + new `_make_synthetic_serve` signature (plumbing-only change to call sites; no behavior change yet — pass `None` for the new args from CLI). Unit-test the helper in isolation.
2. Thread `sequence_probs` + `ball_positions` to the two `_make_synthetic_serve` call sites in `classify_rally`. Now the helper is consulted at runtime.
3. Re-run the panel diagnostic from this session (the per-rally synthetic-vs-real placement table). Compare pre/post.
4. Visual spot-check on the panel rallies that flipped: confirm the new synthetic-serve frame is at or near the actual serve in the video.
5. If panel ship-gate met: roll out across the fleet (no DB writes — synthetic serves are recomputed on every action-classification pass; next `track-players` or `analyze actions` run picks up the change).

No DB migration. No retraining. No production-default flag.

## Done criteria (ship gate)

Measured on the **full GT pool** (66 videos, 313 rallies with both GT serve and pred serve):

- ≥ **30 of 51** currently-mis-placed synthetic serves now within ±15 frames of GT serve (≥ 60% catch rate on the addressable population).
- **≤ 5 regressions** on the 194 currently-correctly-placed real serves (allows for minor frame drift; no systemic displacement).
- Synthetic hit rate jumps from **46% → ≥ 75%**.
- Synthetic-serve `confidence` field reflects signal grounding (≥ 0.6 when picked from signals; legacy 0.4 when fallen back).
- Visual spot-check on 5 rallies (mix of fixed, regressed-if-any, and signal-disagreement low-confidence): the synthetic frame matches the visible serve.

## Risks

1. **MS-TCN++ serve-class peak is a known weak spot.** Per `contact_detection_ceiling.md`, the dominant FN class is serve at 74% F1, and "ball tracking starts late" is the documented failure mode. Lowering the threshold to 0.50 (vs the usual 0.80) helps but trades precision. **Mitigation**: Signal B (ball velocity burst) is the precision check. When signals disagree, we mark the synthetic as low-confidence rather than committing to a wrong frame.

2. **Ball velocity threshold is empirically fragile.** A `0.012` per-frame normalized-velocity threshold isn't a universal constant; serves at different camera distances and frame rates may need different values. **Mitigation**: the implementation plan must include calibration on the 14 correctly-placed real serves before shipping. If a single threshold doesn't separate serves from non-serves with > 90% precision in the search window, we either tune per-fps or drop Signal B and rely on Signal A only.

3. **Picked frame is in pre-rally dead time.** The rally bounds may include a few seconds before the actual serve (toss + walk-up). The picked frame might land in dead time if the pipeline didn't filter the rally bounds tightly. **Mitigation**: the `MAX_PRESERVE_FRAMES = 150` clamp prevents wildly-early picks; visual spot-check confirms.

4. **Player attribution stays stale.** This workstream changes only the synthetic's `frame`, not its `player_track_id`. If the original `server_track_id = -1` (off-screen server), the corrected-frame synthetic is still un-attributed. That's acceptable for v1.1 — it's a frame fix, not a server-identity fix. Attribution is downstream of frame correctness; future work can revisit.

5. **Cascade to receive detection.** Once the serve is at the right frame, `_find_post_serve_receive_candidate` may now produce different results — the receive-search window shifts. This might create new candidates (good — they were missed before) or new artifacts (less likely; the receive search is constrained to a forward window from the serve frame). **Mitigation**: panel measurement compares not just serve placement but full F1 — any net regression on receive accuracy is a ship gate fail.

6. **Synthetic confidence inflation.** Bumping `confidence` from 0.4→0.6+ when the placement is signal-grounded gives downstream stats consumers (e.g., player_attribution) more weight on these synthetics. **Mitigation**: `confidence` is calibrated; 0.6 is well below the typical detected-serve confidence (~0.55–0.95) so it doesn't dominate.

## File-change summary

- New: helper function `pick_synthetic_serve_frame(...)` (in `action_classifier.py` or a new `synthetic_serve_placement.py` — implementer's choice).
- Modified (1): `analysis/rallycut/tracking/action_classifier.py` — `_make_synthetic_serve` signature + body, plumbing through `classify_rally`/`classify_rally_actions`. ~40 LOC net.
- Modified (1): production CLI paths if any new plumbing is needed (`track_player.py`, `analyze.py`) — likely no change because `sequence_probs` is already passed; need to add `ball_positions` if it isn't.
- New tests:
  - Unit: `pick_synthetic_serve_frame` over the four decision-matrix cases.
  - Integration: `fb7f9c23` synthetic-serve placement (frame within ±15 of GT 154).
- New script: `analysis/scripts/measure_synthetic_serve_placement_panel.py` — pre/post diagnostic table.

**Total**: 4 files, ~150 LOC + tests.

## Notes on what this replaces

The original "coherence-driven contact recovery v1" is being removed (`contact_recovery.py` and friends). Its diagnostic value — confirming the FN distribution and revealing the GT-bounds-mismatch in `ruru:3655eb69` — is preserved in the GT fix and the saved diagnostic logs. The v1 module itself recovered 0 contacts and pivoting to this workstream gives up nothing the prior work was actually doing.
