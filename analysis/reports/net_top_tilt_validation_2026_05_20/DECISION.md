# Tilt-aware net_top — decision document (post-R3, N=77 clean)

> **2026-05-20 update — R4 SHIPPED locally** (not yet pushed).
> `CONTACT_PIPELINE_VERSION v7 → v8`. The contact-detection pipeline now
> uses `estimate_net_line()` solvePnP midpoint as the primary source of
> `estimated_net_y`, falling back to v7's M4 corners ridge, then to v6's
> ball-trajectory midpoint. Library: `analysis/rallycut/tracking/contact_detector.py`.
> Production wiring: `analysis/rallycut/cli/commands/track_player.py` (computes
> NetLine once on the full source video, stamps it into the output JSON,
> passes through both `detect_contacts` calls) and
> `analysis/rallycut/cli/commands/analyze.py` (reads the stamped JSON
> field, replays without re-running keypoint detection).
> Tilt awareness was *not* adopted — only the L/R midpoint scalar — because
> the visual catalog (this doc) showed solvePnP-recovered tilt is unreliable
> on visibly-flat nets. The tilt question stays closed as low-EV.
> Fleet refresh required:
> `uv run python scripts/redetect_all_actions.py --apply`.



Two read-only probes against the 77-video user-GT corpus. R3 (orphan
recovery) executed 2026-05-20: probe fetcher patched to resolve videos
via MinIO-wide index + vid-keyed local cache, plus support for raw
`.MOV` orphans. Net result: **N1 now has 77/77 paired comparisons** (up
from 61/77 in the first pass). Two upstream bugs found and fixed.

See `n1_summary.md` (numerical A/B) and `n2_catalog.md` (visual
catalog grouped by |tilt| bucket).

## N1 headline (N=77, all paired)

| Estimator    | n  | med &#124;Δ&#124; | mean &#124;Δ&#124; | worst | >0.025 | >0.05 | >0.10 |
|--------------|----|------|-------|-------|--------|-------|-------|
| M4 (LOO ridge) | 77 | 0.008 | 0.013 | 0.222 | 5 | 1 | 1 |
| NLE midpoint   | 77 | 0.009 | 0.012 | **0.047** | 9 | **0** | **0** |

Paired winner counts (ties within ±0.005): **M4 30 / NLE 19 / tie 28.**

Key observations:

* **Medians effectively tied** (0.008 vs 0.009 — well inside the
  +0.002 viability threshold from the plan).
* **Mean: NLE slightly better** (0.012 vs 0.013) thanks to no
  catastrophic outliers.
* **Worst case: NLE much better** (0.047 vs 0.222). M4's 0.222 is the
  known `yaya[0]` row (see N1 caveat below). Without that one row M4
  worst falls to ~0.034 — matching the original M4 ship numbers — but
  the catastrophic case is part of the production corpus and has to
  be counted.
* **NLE has zero |Δ|>0.05 errors and zero |Δ|>0.10.** M4 has one of
  each. NLE has more residue in the middle (9 vs 5 at >0.025).

## R3 artifacts (bugs fixed this pass)

* **Probe fetcher**: was constructing the S3 path as
  `videos/00000000-0000-0000-0000-000000000001/<vid>/<name>.mp4`,
  which silently failed for 16 videos whose actual MinIO file lives
  under a different `user_id` prefix (e.g.
  `videos/53991ae9-…/<vid>/<name>.mp4` or
  `videos/ec103cb6-…/<vid>/<name>.mp4`). Replaced with a `aws s3 ls
  --recursive` index of MinIO at probe startup, then vid-keyed lookup.
* **Local cache key**: the original fetcher used
  `/tmp/net_top_validation_videos/<name>.mp4` as the local cache
  filename. Two `Video` rows with name=`yaya` and two with `yeye` both
  resolved to the same local file, so the second row reused the first
  row's video for keypoint detection. This produced the spurious
  `yaya[1] NLE Δ=+0.232` in the first N1 pass. New key:
  `<vid>__<name>.mp4`.
* **Raw `.MOV` orphans**: 4 videos (yiyi, yoyo, yaya[1], yeye[1])
  exist in MinIO only as raw iPhone `IMG_*.MOV` (never processed into
  proxy/optimized). Suffix priority extended to include `.mov`/`.MOV`
  so they resolve at low priority. Adds ~5 GB to the local cache (the
  yoyo file alone is 3.3 GB).
* **Helper module**: `analysis/scripts/_net_top_probe_fetch.py` — the
  fetcher logic shared by both probes. Future probes that need a
  per-video file from MinIO can reuse it.

## N1 caveats that remain

* **`yaya` 30fps row** still has `M4 Δ=-0.222` (the catastrophic case).
  Two yaya rows in the DB share corners but have GTs 0.449 (30fps) and
  0.199 (59.9fps) — at least one of the two labels is wrong. After R3,
  both rows fetch their own files; M4's prediction is shared (0.227)
  because corners are identical; only one GT is consistent. Worth
  spot-checking visually whether the 30fps GT (0.449) or the 59.9fps GT
  (0.199) is the right one for the actual video. Not blocking the
  decision but a data-quality clean-up.
* **4 `sanity_failed` warnings** from `estimate_net_line` on match,
  matttch, yeye[1], yiyi — geometry between far/near baselines was
  outside the sanity band. NLE still returned a value; the warning
  says "trust judiciously." All 4 have small tilts.

## N2 tilt prevalence (N=77, post-R3)

| bucket     | &#124;tilt&#124; range | count | % |
|------------|--------|-------|---|
| flat       | <0.005 | 30 | 39% |
| mild       | 0.005–0.015 | 29 | 38% |
| notable    | 0.015–0.030 | 17 | 22% |
| pronounced | >0.030 | 1 | 1% |

The `notable + pronounced` group grew from 12→18 after R3 — well above
the "≤5 → low-EV" gate. The single pronounced case is `yoyo`
(tilt=-0.032), a night beach scene.

## N2 visual spot-check — 7 cases now (5 from first pass + 2 new)

| video | bucket    | NLE tilt | visible net tilt | NLE direction matches? |
|-------|-----------|----------|------------------|------------------------|
| jiji  | notable   | +0.030   | slight right-down  | **yes** — orange sits on net, rescues M4 |
| wawa  | notable   | -0.028   | ambiguous          | unclear at video resolution |
| cece  | notable   | -0.017   | appears flat       | **no** — NLE invents left-down tilt |
| jeje  | notable   | +0.026   | appears flat       | **no** — NLE invents right-down tilt |
| lala  | notable   | +0.019   | appears flat       | **no** — false right-down tilt, midpoint also wrong |
| dark  | notable   | +0.028   | clearly horizontal | **no** — false right-down tilt |
| yoyo  | pronounced | -0.032   | both nets look horizontal | **no** — solvePnP noise in dim lighting |

**Pattern confirmed**: NLE's |tilt| magnitude bucketing is reliable as
a "this might be tilted" filter, but the *direction* and *magnitude*
of the tilt are unreliable on visibly-flat nets. The pronounced case
is itself a false positive. ~2 of 7 spot-checks show NLE recovering
real tilt (jiji and arguably wawa); the other 5 are false-positive
tilts.

## Decision rule from the plan, applied to N=77

| N1 midpoint    | N2 tilt prevalence | N2 NLE tilt direction | Choose |
|----------------|--------------------|------------------------|--------|
| **NLE ≈ M4 (tied)** | **18 tilted** | **wrong / unreliable** | **C1** |

The clean N=77 data lands in the same decision cell as N=61: **C1**
(human GT) is the only path that produces a tilt-aware estimator whose
tilt output matches the user-visible net. C2 (replace M4 with NLE
outright) and C3 (use NLE as a teacher signal) would both propagate
false tilts on the majority of visibly-flat nets.

## Sharpened second-order check: is C1 worth the cost?

The clean numbers reframe the EV math:

* **Real-tilt cases**: spot-check rate of 2/7 = ~29% of "notable" + a
  pronounced false positive ⇒ projected ~5 truly-tilted videos in 77
  (down from my earlier 7-9 estimate).
* **M4's error on the truly-tilted ones**: from the top-10-by-|tilt|
  in the CSV, M4 |Δ| ranged 0.000–0.030 with median ~0.010. Headroom
  for a tilt-aware model on those ~5 videos is at most ~0.02
  improvement each.
* **Downstream impact**: `estimated_net_y` feeds `is_at_net`
  (zone ±0.04), side classification, `_check_net_crossing` window, and
  `ball_y_relative_net` GBM feature. A scalar improvement of ~0.02 on
  ~5/77 videos translates to a small fraction-of-a-percent lift on the
  contact-detection downstream metric.

Estimated EV: still **small**. The R3 finding sharpens the
denominator: ~5 truly-tilted videos, not 12 — so the headroom is even
smaller than the first DECISION.md suggested.

## Updated next-step options

* **(R1) Close as low-EV.** *Default recommendation given the data.*
  M4 LOO is already at med 0.008. NLE midpoint matches it on med, beats
  it on outliers (no |Δ|>0.05), but its tilt signal is unreliable. ~5
  truly-tilted videos in 77 — too small a slice to justify a UI +
  relabel + ridge + downstream-lift cycle.
* **(R2) C1 with reduced scope.** Decouple UI handles, add L/R
  columns, relabel ONLY the videos visually-confirmed as tilted (skim
  the N2 catalog — likely 4-8 videos), fit a sparse tilt ridge, gate
  ship on a measurable downstream metric. **Plan cost: ~1 hour user
  time + ~4-6 hours engineering.** Risk: ridge fit on ~5 non-zero
  examples has no statistical headroom; may not generalise.
* **(R4) Adopt NLE midpoint as production scalar (no tilt).** *New
  option that R3 surfaced.* NLE midpoint already beats M4 on means and
  catastrophic outliers (worst 0.047 vs 0.222, zero |Δ|>0.05 vs 1).
  Replace M4 in `contact_detector._prepare_candidates` with
  `estimate_net_line(...).top_*` averaged, fall back to M4 when
  `estimate_net_line` returns None or sanity-failed. No tilt support;
  just a more robust scalar. **Plan cost: ~2 hours engineering, zero
  user time, zero schema change.** Risk: 4 sanity-failed videos need
  fallback, and the keypoint detector may regress on future videos
  not in the 77-video set.

R1 and R4 are mutually compatible (R4 can ship while the tilt question
stays closed). R2 is the only path to actual tilt awareness.

## Adjacent data-quality finds worth fixing (independent of decision)

* **18 `Video` rows with NULL `original_s3_key` and `proxy_s3_key`** —
  their files exist in MinIO under different UUIDs but the DB lost the
  link. Fixable with a `UPDATE videos SET proxy_s3_key = ?,
  original_s3_key = ? WHERE id IN (...)` populated from the MinIO
  index. Affects much more than just this workstream: any consumer
  that reads `proxy_s3_key` from these rows will fail.
* **`yaya` duplicate-row mystery** — two rows share corners, different
  GTs (0.449 vs 0.199), one of them has `IMG_1820.MOV` as the only
  MinIO file. Either the two rows are different camera setups
  accidentally given the same court calibration corners, or one of the
  two GT labels is incorrect. Same likely for `yeye`.

## Artifacts

* `n1_summary.md` — full numerical summary (N=77)
* `n1_midpoint_ab.csv` — per-video table
* `n2_catalog.md` — full visual catalog (77 frames grouped by bucket)
* `frames/<bucket>/<name>.jpg` — per-video overlays
* `analysis/scripts/_net_top_probe_fetch.py` — reusable robust MinIO
  fetcher for downstream probes
