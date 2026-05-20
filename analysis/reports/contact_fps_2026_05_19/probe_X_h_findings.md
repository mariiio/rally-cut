# Probe X-H findings: GBM probabilities for the 13 H-A3 cases — REFRAMES X-G

**Date:** 2026-05-20
**Inputs:** the 13 H-A3 cases from X-G. Ran `detect_contacts` with classifier
threshold = 0.0 (accept anything with GBM > 0) AND with the production
pipeline (regressor + dedup) ON, then read contacts in the attack window.

## Headline

**H-A3 is mis-named.** The GBM is NOT confidently rejecting these candidates.
Sub-bucket distribution across the 6 cases that have any candidate in the
attack window at threshold=0:

| GBM bucket | Count | Verdict |
|---|---|---|
| GBM < 0.10 (truly rejected) | **0** | None |
| GBM in [0.10, 0.40) | **1** | gigi b8d333ae: 0.202 |
| GBM > 0.40 (well above 0.35 default) | **5** | caco 9452ee5a (0.963), juju acada27e (0.917), juju c89b346b (0.873), kiki a0aba15e (0.835), popo c1052008 (0.912) |

So GBM acceptance isn't the gate. Most "H-A3" candidates are ACCEPTED by
the GBM at a high probability. They just don't appear in the final
`detect_contacts` output.

## Where do they go?

7 of the 13 cases have NO candidate at all in the attack window in the
final permissive output (gigi 72c8229b, gigi 3e07342a, juju d810943e,
mimi f3695225, moma 753a4ec7, moma 9bb60892, toto 67b3e1ad), but X-G
reported `cand@att >= 1` for each — meaning a candidate DID survive
`_merge_candidates`.

Pipeline order (`contact_detector.py`):
1. Generators → raw frames
2. `_merge_candidates` chain (line 2267-2296), `min_peak_distance_frames=12`
3. Per-frame GBM accept/reject (line 2928)
4. `_snap_contacts_to_direction_change_max` (line 3141), window ±10
5. `refine_contacts_with_regressor` v4 (line 3158), bumps frames
6. `_deduplicate_contacts` (line 3165) with `adaptive=True`,
   `_CROSS_SIDE_MIN_DISTANCE=4`

For a candidate at attack frame T to disappear after surviving merge AND
GBM, only steps 4-6 can remove it. The dominant culprit is almost
certainly **the heuristic snap or the regressor (steps 4-5) bumping the
attack-frame candidate INTO the block window**, where it then either
deduplicates against the block candidate or simply appears in the
"block" window of probe X-G's final list.

## Worked example: juju acada27e

- GT block frame = 241, attack window = [226, 237]
- X-G base run: `det@blk=[238]`, `det@att=[]` — ONE contact in the
  ±15 around block frame, at frame 238.
- X-H permissive run: contact at frame **237** with GBM=0.917, at-net=True,
  ball.y=0.484, court=near.

Reading the two together: the same physical candidate is at frame 237 in
the permissive run (sorted by confidence in dedup; high-confidence wins
its slot) but at frame 238 in the base run (different dedup-survivor
choice when fewer candidates compete). Either way, the regressor + dedup
collapse what's likely TWO physical contacts (attacker at ~237, blocker
at ~241) into ONE contact whose location bounces between 237-238.

## True failure mode (revised hypothesis)

**The contact_frame_regressor (v4) is consolidating the attack+block
pair into a single contact**, because:

1. Pre-regressor, the merger has already kept only ONE candidate per
   12-frame window (the higher-priority direction-change frame wins).
2. The remaining candidate sits between the physical attack and physical
   block frame.
3. The regressor snaps it to the GT-frame estimate. The training data
   (full 74-video action GT) labels the BLOCK frame for these cases as
   the single GT contact, so the regressor preserves that bias.
4. There's no second candidate to recover the attacker's frame.

## Implications for Phase 3

The actual fix path is more upstream than the original plan considered:

- **Step 2 of the pipeline is the gate.** The merger consumes both
  attack and block candidates and only one survives.
- **Net-proximity-gated cross-side relaxation in the merger** remains
  the most surgical fix, but we'd need:
  - Compute approximate court_side or net-proximity at merge time
    (cheap — just check `abs(ball.y[T] - estimated_net_y) < 0.15`).
  - Allow two near-net candidates within (`_CROSS_SIDE_MIN_DISTANCE`,
    `min_peak_distance_frames`) to coexist when they likely span the
    net.
- **Don't lower GBM threshold** (probe shows GBM doesn't reject these).
- **Don't disable the regressor** (it improved action accuracy on the
  full corpus; disabling regresses other classes).

## Cases by final-state interpretation

| Case | X-G cand@att | X-H gbm | Likely state |
|---|---|---|---|
| caco 9452ee5a | 1 | 0.963 | candidate kept at attack frame in permissive |
| gigi 72c8229b | 1 | n/a | snapped into block window |
| gigi 3e07342a | 1 | n/a | snapped into block window |
| gigi b8d333ae | 2 | 0.202 | low GBM AND snap moved survivor |
| juju d810943e | 1 | n/a | snapped into block window |
| juju acada27e | 1 | 0.917 | candidate kept at attack frame in permissive |
| juju c89b346b | 2 | 0.873 | candidate kept in permissive; dedup removes in base |
| kiki a0aba15e | 2 | 0.835 | candidate kept in permissive; dedup removes in base |
| mimi f3695225 | 1 | n/a | snapped into block window |
| moma 753a4ec7 | 1 | n/a | snapped into block window |
| moma 9bb60892 | 1 | n/a | snapped into block window |
| popo c1052008 | 2 | 0.912 | candidate kept at attack frame in permissive |
| toto 67b3e1ad | 1 | n/a | snapped into block window |

## Next probes needed

- **P-I**: instrument `_snap_contacts_to_direction_change_max` and
  `refine_contacts_with_regressor` to log per-contact pre-snap vs
  post-snap frame deltas, for the 13 H-A3 rallies. Falsifies whether
  the regressor is the source of cross-pair collapse.
- **P-J**: try disabling the regressor on these 13 rallies and see
  whether two contacts emerge (one in attack window, one in block
  window). If yes, the path is regressor-gating.

## What this means for the original Block F1 plan

Phase 3 of the plan should now focus on:
1. Merger-level relaxation for cross-side near-net pairs (H-A1, ~8 cases)
2. Snap/regressor-induced pair-collapse mitigation (H-A3 reinterpreted,
   ~7-13 cases)

These are now both contact-detector-internal changes. The plan's
"H-A3 → out-of-scope 60fps GBM bug" branch is wrong (these are mostly
30fps videos). The decision tree needs rebuilding.

## Numbers reproduce

```bash
cd analysis
uv run python -u scripts/probe_X_h_gbm_rejected_attacks.py
```
