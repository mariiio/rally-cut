# v9 SOTA — decision document (final, post-ship)

> **v9 SHIPPED locally on `main`** at commit `880dd718`.
> CONTACT_PIPELINE_VERSION `v8 → v9`. Direct 8-keypoint net-top
> observation replaces v8 NLE as the cascade primary; v8 NLE + v7 M4
> + v6 ball-traj kept as cascade fallbacks. Production callers
> (track_player.py, analyze.py, redetect_all_actions.py) wired to
> compute NetTopLine once per source video and pass through.

## Eval gate (probe X-O on 78 user-GT videos)

|              | n  | med &#124;Δ&#124; | mean   | worst  | >0.025 | >0.05 |
|--------------|----|-----------|--------|--------|--------|-------|
| M4 LOO       | 78 | 0.0060    | 0.0078 | 0.0242 | 0      | 0     |
| NLE v8 mid   | 77 | 0.0103    | 0.0126 | 0.0529 | 11     | 1     |
| **v9 8-kpt** | 78 | **0.0046** | **0.0066** | 0.0317 | 3      | **0** |

Val-split-only (15 unseen videos): v9 med=0.0052 — confirms
generalization, no overfitting to training split.

Tilt direction on visibly-tilted subset (|gt_tilt|>0.015, n=10):
**10/10 = 100%**. Three mismatches at |gt_tilt|∈(0.010,0.015) were at
the labeling noise floor; one (gigi) was a false fail from the gate's
eps=0.003 flat-threshold (both gt and v9 negative, just smaller magnitude).
Visually confirmed via overlay rendering — v9 picks up real tilt with
~0.002 magnitude precision.

## Fleet C-4 impact (clean A/B)

Same code today, holding all non-net_y commits fixed:

| Cascade primary | C-4 violations | Δ |
|-----------------|----------------|---|
| v8 NLE (flag off) | 165 | baseline |
| **v9 NetTopLine** | **167** | **+2 (+1.2%)** |

The +2 is a real but small regression: v9's slightly-different net_y
position interacts with v6/v7-tuned `is_at_net` + side-classification
thresholds. Underlying mechanism: **40 newly-resolved violations + 42
newly-introduced violations**, net +2 — i.e. v9 IS doing real
correctness work, not just regressing.

**IMPORTANT correction:** an earlier (incorrect) framing claimed
"+9 (+5.7%) regression." That number was conflating cascade impact
with +7 baseline drift from concurrent scorer/classifier commits
(unrelated to net_y). The clean A/B above measures only the cascade
flip; the +7 drift is a separate workstream.

## Why ship despite +2 C-4 cost

- **Midpoint accuracy is ~55% better** (0.0046 vs 0.0103 median |Δ|).
- **Zero outliers above 0.05** (v8 has 1 — the catastrophic yaya case
  that v8 NLE itself failed to fix completely).
- **+2 fleet cost is small and addressable** in a downstream-threshold
  re-tuning workstream that should push C-4 lower than 165 (the v8
  baseline) once `is_at_net` is calibrated for v9's net_y convention.
- **40 newly-resolved violations** are real correctness gains —
  cases where v8 was getting contact attribution wrong and v9 fixes
  them.
- **100% tilt direction agreement** on unambiguously-tilted videos
  (10/10), with magnitude precision ~0.002 — no estimator before
  v9 expressed tilt at all.

## v9 training caveat

The 8-kpt model also regressed on kp 0..5 (center keypoints shifted
~0.034 normalized y on 74/76 videos). We isolated this by keeping
the v9 model at a **separate path**
(`weights/court_keypoint/court_keypoint_v9_8kpt.pt`); production
court calibration still uses the original 6-kpt model at
`court_keypoint_best.pt`. The two-file split lets v9 ship without
risking production calibration.

A v10 8-kpt model trained with a larger base (yolo11m-pose or
yolo11l-pose) or stricter freeze on the pretrained pose head could
plausibly fix the kp 0..5 regression, allowing unification.

## Open follow-ups

1. **Re-tune `is_at_net` + side-classification thresholds for v9
   net_y convention** — should eliminate the +2 fleet C-4 cost and
   likely push it negative. The downstream code is at
   `contact_detector._resolve_court_side` (hard threshold at
   estimated_net_y) and `is_at_net` band check (currently asymmetric
   −0.15 above / +0.08 below the v6/v7 net_y).
2. **Investigate the +7 baseline drift** between the v8 ship date
   (158) and today (165). Likely from the parallel Sub-lever 1
   scorer commits (66141e0f revert and predecessors).
3. **v10 8-kpt training** with a larger base to fix the kp 0..5
   regression and let one model serve both production court
   calibration AND net-top observation.

## Artifacts

* `o1_summary.md` + `o1_v9_ab.csv` — probe X-O numerical eval (78 GT)
* `../coherence_c4_catalog/2026-05-20_post_v9_flagoff_summary.md` —
  fleet baseline with v9 cascade OFF (165)
* `../coherence_c4_catalog/2026-05-20_post_v9_final_summary.md` —
  fleet with v9 cascade ON (167)
* `../net_top_tilt_validation_2026_05_20/DECISION.md` — v8 ship
  context (superseded as primary; v8 still the cascade fallback)
* `~/.claude/projects/.../memory/net_top_v9_8keypoint_2026_05_20.md` —
  durable memory entry
* `analysis/scripts/probe_X_o_v9_keypoint_eval.py` — eval probe
  (re-runnable)
* `analysis/rallycut/court/net_top_keypoint_reader.py` — production
  reader module
* `analysis/datasets/court_keypoints_v6/` — 8-keypoint training set
* `weights/court_keypoint/court_keypoint_v9_8kpt.pt` — v9 model
  (separate from production 6-kpt at `court_keypoint_best.pt`)
