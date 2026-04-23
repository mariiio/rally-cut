# Probe 0 — Coordination with attribution workstream

**Date:** 2026-04-23
**Scoping plan:** docs/superpowers/plans/2026-04-22-game-semantics-scoping.md
**Workstream touch-point:** player_attribution Day 3 roadmap Phase 4 (memory/player_attribution_day3_2026_04_22.md)

## Current state of attribution workstream

Day 3 is at Phase 2→3 (match_tracker seeding patch, feature-flagged `MATCH_TRACKER_GLOBAL_SEED=1`). Phase 4 of the attribution roadmap explicitly names:

> Serve → opposite-team receive (hard) / Three-touch bound (hard) / Net-crossing count = 1 per exchange (hard) / Consecutive-touch teammate / Setter → attacker teammate / Block + dig coupling

None of this is implemented yet. The roadmap calls these "rule-cost terms" that plug into a `α·identity + β·geometry + γ·rule_violation` attribution cost. A per-frame ball-side signal is the prerequisite input to the net-cross term.

## Decision

**This workstream builds Probes 1 + 2.** Attribution hasn't started on any ball-side primitive. Building it here with a schema attribution will consume is the right factorization per the plan's Probe 0 decision tree.

## Ball-side schema agreement (for attribution to consume)

A per-frame, per-rally record:

```json
{
  "rally_id": "uuid",
  "frame": 42,
  "ball_image_xy": [0.51, 0.43],
  "net_base_image_ys": { "left": 0.48, "right": 0.49 },
  "net_top_image_ys":  { "left": 0.28, "right": 0.29 },
  "ball_side": "far",          // "far" | "near" | "ambiguous"
  "side_confidence": 0.87,      // 0..1 — lower near at-net or partial-occlusion
  "ambiguity_reasons": []       // e.g. ["at_net_within_5px", "ball_interpolated"]
}
```

**Conventions (locked):**

1. **Decisive horizontal is net-TOP** (not net-base). A ball above net height but on the near side can have image-Y above (smaller value than) net-base. Net-top is the cleaner decision line in image space.
2. **Coordinate convention**: image-Y increases downward (standard image convention). "Far" = image-Y < net-top-line-Y-at-ball-X. "Near" = image-Y > net-top-line-Y-at-ball-X.
3. **At-net ambiguity**: when `|ball_y − net_top_y| < 5 px at 1280w-equivalent`, label as `ambiguous` rather than force a side.
4. **Interpolated ball**: if the underlying WASB track was filled by the ball filter's interpolation (not a real detection), flag via `ambiguity_reasons` so the attribution workstream can downweight the sample.
5. **Net-top height constant**: default 2.43 m (men's beach). Women's matches would need a flag; for this probe we use 2.43 universally and observe how noisy per-video it looks.

**Storage format:** not pre-committed. For Probes 1+2, emitted as CSV/JSONL under `analysis/outputs/game_semantics_probe/`. If probes pass, we'll propose a cached per-rally column or sidecar file at Phase A implementation.

## What the attribution workstream needs to acknowledge

When Phase 4 attribution lands, it consumes `ball_side` + `side_confidence` as inputs to the net-cross rule-cost. The attribution workstream should:

1. Treat `ambiguous` frames as abstentions for the net-cross cost — don't penalize either side.
2. Not trust `ball_side` on frames with `side_confidence < 0.5` — downweight or drop.
3. Not re-derive ball-side from first principles (attribution has no net-line detector of its own). Consume this workstream's output.

## Known limitations flagged upfront

- **Monocular 2D sidedness is architecturally limited.** A high ball on the near side can be image-Y above the net-top line and be labeled far in error. Probe 2 will measure how often this actually happens on real rallies. If it's frequent (>10% of mid-flight frames), the `side_confidence` flag needs to capture it and both this workstream and attribution need to treat `ambiguous` generously.
- **Net-top projection depends on court calibration.** Videos without calibration (see check_calibration_coverage.py coverage rate) fall back to the pure-2D midline heuristic from Probe 1's fallback. That fallback is worse; we measure it separately.
- **Net-height constant is a single value.** If women's beach (2.24 m) shows up in the corpus, the projected net-top will be systematically low for those videos; the probe will flag it.

## Sign-off

No meeting — attribution workstream owner (same user/owner as this workstream per `player_attribution_day3` memo) can veto this schema by reading this memo. Pre-registered: if they request a different ball-side schema BEFORE Probe 2 output lands, Probe 2 re-runs under that schema. If after, they consume Probe 2's current schema as-is.
