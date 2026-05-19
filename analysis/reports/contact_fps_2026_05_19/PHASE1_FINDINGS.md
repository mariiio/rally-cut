# Phase 1 Diagnosis — Contact-FN failure mode on 60fps cohort

**Date:** 2026-05-19
**Cohorts:** 60fps (kuku, lulu, wawa — 185 GT actions), 30fps (titi, toto, jaja — 232 GT actions)
**Method:** For each GT action frame, check if v4 contact exists within ±10 frames. If not, classify the FN by re-running upstream signals at the GT frame.

## Headline result

| Cohort | FN rate | C% (ball missing) | D% (no player near) | A% (GBM rejected) | B% (no candidate) |
|--------|---------|-------------------|---------------------|-------------------|-------------------|
| 60fps  | **15.1%** | 4.9% | 4.9% | **5.4%** | **0.0%** |
| 30fps  | 6.5%    | 3.0% | 2.6% | 0.9%      | 0.0%      |
| Δ      | +8.6pp  | +1.9pp | +2.3pp | **+4.5pp** | 0.0pp |

**Class B = 0 in BOTH cohorts.** Every single FN where ball+player were present at the GT frame had `_prepare_candidates` produce a valid candidate within ±5 frames. The candidate generators work fine on 60fps.

The dominant 60fps gap (+4.5pp class A) is **the GBM rejecting valid candidates** the gates already accepted.

## Per-video breakdown

| video | fps | GT | matched | FN | FN% | C | D | A | B |
|-------|-----|-----|---------|-----|------|---|---|---|---|
| kuku  | 59.9 | 97 | 86 | 11 | 11.3% | 1 | 4 | 6 | 0 |
| lulu  | 59.9 | 37 | 31 |  6 | 16.2% | 4 | 1 | 1 | 0 |
| wawa  | 59.9 | 51 | 40 | 11 | 21.6% | 4 | 4 | 3 | 0 |
| titi  | 30.0 | 86 | 79 |  7 |  8.1% | 3 | 3 | 1 | 0 |
| toto  | 29.9 | 58 | 51 |  7 | 12.1% | 4 | 3 | 0 | 0 |
| jaja  | 30.0 | 88 | 87 |  1 |  1.1% | 0 | 0 | 1 | 0 |

## Which generators produced the rejected class-A candidates

(Some candidates have multiple generators firing — counts sum to more than class A totals)

| Generator | 60fps | 30fps |
|-----------|-------|-------|
| inflection | 7 | 2 |
| direction-change | 5 | 2 |
| net-crossing | 5 | 2 |
| velocity peak | 3 | 1 |
| deceleration | 2 | 1 |
| parabolic | 1 | 0 |

Strong signal exists at the GT frame across multiple generators. The GBM rejects despite this.

## Implications

### What the v5 ship actually did

The v5 plan was "scale 13 frame-count GATE fields by fps/30." This affected:
- The gates that decide which candidates `_prepare_candidates` outputs (`direction_check_frames`, `inflection_check_frames`, etc.)
- The dedup window (`min_peak_distance_frames`)
- The GBM feature input (`direction_check_frames` is used both as a gate AND fed to the GBM)

**Class B = 0 means none of the gate-scaling could have helped** — gates were already firing at the right frames in v4. The scaling did:
1. Make dedup more aggressive at 60fps → real touches merged together (`min_peak_distance` 12→24)
2. Shift the GBM feature value (`direction_change_deg`) further from training distribution → MORE class A rejections, not fewer
3. No improvement to class B (because it was already 0)

Net: more class A FNs + no class B fix = regression. Exactly what we measured.

### What the proper fix is

**Retrain the contact_classifier GBM** with:

1. **Training corpus expanded to include 60fps videos.** Currently the GBM was trained on 2480 (candidate, GT) pairs presumably from the trusted-29 corpus (mostly 30fps). Adding 60fps videos directly addresses the feature-distribution gap.

2. **Features computed in physical-time units** (optional but cleaner). E.g. `direction_change_deg` computed over `window = round(0.27s * fps)` instead of `window = 8`. This makes the GBM see fps-invariant inputs. Cleanest design; the gate-scaling approach was attempting this but on the wrong layer.

3. **Per-fps cross-validation.** Hold out one 60fps and one 30fps video per fold; verify both cohorts improve.

Expected impact: a properly-trained GBM should close most of the +4.5pp class A gap. If we hit it fully, 60fps FN rate drops from 15.1% to ~10.6%, roughly matching 30fps.

### Upstream issues (smaller priority)

- **Class C +1.9pp** (60fps ball-tracker FN): Look at `ball_filter.py` time-semantic constants (`max_chain_gap=30`, `exit_max_ghost_frames=30`, `warmup_protect_frames=120`). These may be culling real ball segments at 60fps. Lower priority than GBM retrain but real.
- **Class D +2.3pp** (no player near at GT frame): Could be player-tracker fps fragility, OR GT labeling slop (the labeled ball position differs from the tracker's). Worth a per-FN visual check before fixing.

### What's NOT a problem

- Candidate generators (`_prepare_candidates`) work fine on 60fps. Class B = 0 proves this.
- The fps-deadband design (no-op below ±10% of 30fps) is correct.
- The fps plumbing through `analyze.py`/`track_player.py`/`redetect_all_actions.py` works.

## Recommendation

**Lift the "no GBM retrain" constraint** for the contact_classifier specifically.

Important distinction: the user's original out-of-scope item was "action-model retraining (see retrain_action_models_plan_2026_05_18.md)." That refers to the action_classifier GBM + MS-TCN++. **The contact_classifier GBM is a different model** (`analysis/rallycut/tracking/contact_classifier.py`, last touched in v4 history). Retraining contact_classifier doesn't trigger action_classifier rework.

Once retrained, the v5 work becomes meaningful IF features are made fps-invariant. Otherwise, the simpler ship is: just retrain on multi-fps corpus, no contact_detector code change.

## Scripts kept

- `analysis/scripts/diagnose_contact_fn_60fps.py` — this diagnostic. Re-runnable to track regression after any change.
- `analysis/scripts/dump_contacts.py` — per-rally inspection (Phase 0).
- `analysis/scripts/contact_density_cohort.py` — cohort density (Phase 0).
