# Contact-Detector FPS-Agnostic A/B — REGRESSED

## Outcome

**A/B failed.** The fps-aware fix (CONTACT_PIPELINE_VERSION v4→v5) caused all four 60fps videos to LOSE contacts, not gain them.

## Cohort density

| Video | fps | Baseline density/s | Post density/s | Delta |
|-------|-----|--------------------|----------------|-------|
| haha  | 59.9 | 0.296 | 0.267 | **−0.029** |
| kuku  | 59.9 | 0.429 | 0.419 | **−0.010** |
| koko  | 30.0 | 0.349 | 0.349 | 0.000 (DB-recorded as 30fps) |
| lulu  | 59.9 | 0.361 | 0.353 | **−0.008** |
| wawa  | 59.9 | 0.362 | 0.344 | **−0.018** |
| titi  | 30.0 | 0.506 | 0.506 | 0.000 (byte-identical) |
| toto  | 29.9 | 0.464 | 0.464 | 0.000 (byte-identical) |
| jaja  | 30.0 | 0.333 | 0.333 | 0.000 (byte-identical) |

**60fps cohort avg: 0.362 → 0.346/s (−0.016/s, target was ≥0.39/s)**
**30fps cohort avg: 0.413 → 0.413/s (deadband works as designed)**

## Per-video total contact deltas (60fps)

- haha: 61 → 55 (−6)
- kuku: 90 → 87 (−3)
- lulu: 104 → 103 (−1)
- wawa: 46 → 45 (−1)

All four videos regressed; none improved.

## What worked

1. **Code is clean**: mypy + ruff pass, 8 new unit tests for `_scale_config_for_fps` pass, pre-commit hook accepted v5 bump.
2. **30fps deadband works as designed**: titi/toto/jaja outputs are byte-identical pre/post (modulo the `ver=v4`→`ver=v5` tag).
3. **All plumbing is correct**: fps threads from `Video.fps` / `PlayerTrack.fps` through `analyze.py`, `track_player.py` (both calls), and `redetect_all_actions.py` to `detect_contacts(fps=...)`.

## What failed

The 60fps cohort regressed. Two distinct failure modes in the per-rally diff (`baseline_haha.txt` vs `post_haha.txt`):

### Mode 1: Over-dedup from scaled `min_peak_distance_frames`

Plan Risk #2 came true.

- Rally `1e159b7e`: baseline `[158, 227, 233, 323, 415, 456]` → post `[226, 324, 417, 449]`. The pair `(227, 233)` is 6 frames apart at 60fps (100ms physical time). At v4 the dedup window was 12 frames (200ms at 60fps), so both survived. At v5 the dedup window scaled to 24 frames (400ms at 60fps), so they got merged. Correct in physical-time semantics, but it lost what was probably two real touches.

### Mode 2: GBM feature drift on scaled `direction_check_frames`

The plan agent flagged this as a BOTH case (gate + GBM feature input). I noted it but didn't act on it.

- `direction_check_frames` (8 → 16 at 60fps) is used both as a gate AND fed to the GBM as `CandidateFeatures.direction_change_deg`. Scaling the window changes the angle the GBM observes, even though the physical time is the same. The v4 GBM was trained on 8-frame windows; at 60fps it now sees 16-frame windows producing systematically different angle values.
- Rally `1a84b131` baseline `[311]` → post `[]` — the single contact vanished. Not dedup.
- Rally `2e667b9c` baseline `[187, 256]` → post `[]` — both contacts vanished, neither was dedup-adjacent. The frames 187 and 256 are 69 frames apart — well outside any dedup window.
- Rally `d98f9f86` lost frame 469 (42 frames from neighbors 377 and 511 — well outside any dedup window).

The GBM feature shift is the most likely cause for these sparse-contact losses.

## What to do

The DB is currently in v5 state with worse 60fps contacts. Three options:

1. **Roll back**: `git revert` the 4 code commits, then re-run `redetect_all_actions.py --apply` on the 8 videos with the restored v4 code to restore the baseline DB state. Keep the new diagnostic scripts (`dump_contacts.py`, `contact_density_cohort.py`) — they're useful regardless.

2. **Refine + retry**: edit `_scale_config_for_fps` to scale only the "pure gate" fields and leave the BOTH cases (direction_check, min_peak_distance) at their 30fps values. Pure gates likely safe to scale: `serve_window_frames`, `warmup_skip_frames`, `post_serve_search_window`, `proximity_search_window`, `deceleration_window`, `inflection_check_frames`, `player_search_frames`, `player_candidate_search_frames`. Bump v5→v6, re-A/B.

3. **Investigate deeper**: per-rally trace which gate/GBM-feature dropped each lost contact. May reveal that the user's premise (gates too tight at 60fps causing FN) is partly wrong — the actual mechanism might be upstream (ball_filter culling) or downstream (GBM threshold not fps-aware).

## Files

- `baseline_*.txt` — per-rally contact dumps pre-fix (v4 state)
- `post_*.txt` — per-rally contact dumps post-fix (v5 state)
- `baseline_density.txt` / `post_density.txt` — cohort density before/after
- `redetect_*.log` — per-video redetect logs from the --apply runs
