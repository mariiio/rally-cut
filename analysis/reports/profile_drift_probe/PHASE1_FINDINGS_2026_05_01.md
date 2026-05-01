# Phase 1 — H-Profile-Drift probe findings (2026-05-01)

**Verdict: CASCADE FALSIFIED for the blind regime.**

Dropping every per-field EMA in `PlayerAppearanceProfile` (counterfactual
`EXPERIMENTAL_DROP_PROFILE_EMA=1` flag) had **zero observable effect** on the
13-rally panel verdict. 0 of 4 panel-BAD rallies flipped to GOOD; 0
GOOD/control regressions; both passes produced AGREES 12/13 with identical
per-rally outcomes. The verdict tool output for baseline and counterfactual is
**byte-identical** (`diff verdict_baseline.txt verdict_dropema.txt` empty),
including per-frame swap event counts and the slow-drift detector's exact
fractional measurements. The `baseline_restore` pass also produced a
byte-identical verdict file, confirming determinism.

## Setup

- 4 panel fixtures (`b5fb0594`, `5c756c41`, `854bb250`, `7d77980f`)
- `--no-ref-crops` forces the MatchSolver (blind-regime) path on all fixtures
  per `feedback_blind_regime_goal.md`
- `ENABLE_BBOX_SWAP_DETECTION=0` and `ENABLE_POSITION_JUMP_SWAP=0` to match the
  locked panel state
- Three passes per fixture: `baseline` (flag OFF), `dropema` (flag ON),
  `baseline_restore` (flag OFF — determinism check)
- Verdict tool: `analysis/scripts/panel_verdict_per_frame.py` reads
  `positions_json` directly (the artifact the editor renders)

Pre-run cleanup: stripped legacy synthetic sub-track entries (`-2004`, `-2005`,
etc.) from `appliedFullMapping` in `match_analysis_json` and from
`canonical_pid_map_json` for 5c756c41 and 7d77980f. These were leftovers from a
pre-revert sub-track code path; they broke `_invert_mapping` bijectivity in
match-players' reversal step. After cleanup the orchestrator ran clean.

## Results

### Verdict counts

| Pass | AGREES | Verdict file byte-identical to baseline? |
|---|---|---|
| Baseline (flag OFF) | **12/13** | (anchor) |
| Counterfactual (flag ON) | **12/13** | **YES** — `diff` produces no output |
| Baseline restore (flag OFF) | **12/13** | **YES** — confirms determinism |

The single disagreement under both passes — 5c756c41/r03 PANEL expected BAD,
derived GOOD — is **not** caused by the EMA. It is the same disagreement
under both flag values. Compared to the locked AGREES 13/13 baseline from
`panel_visual_verdict_2026_05_01.md`, the locked run was MIXED-MODE
(ref-crops on 5c756c41 and 7d77980f, blind on the others). My orchestrator
forces blind everywhere; the change in r03's verdict is a regime difference
(blind vs ref-crop), not an EMA effect.

### Panel BAD rallies

| Rally | Baseline | Counterfactual | Flipped to GOOD? |
|---|---|---|---|
| b5fb0594/r10 | BAD (slow_drift) | BAD (slow_drift, identical params) | NO |
| 5c756c41/r03 | GOOD (blind regime) | GOOD (blind regime) | (already GOOD; not a flip) |
| 5c756c41/r07 | BAD (slow_drift) | BAD (slow_drift, identical params) | NO |
| 7d77980f/r02 | BAD (within_rally_swap, 1 event) | BAD (within_rally_swap, 1 event) | NO |

**0 of 4 panel-BAD rallies flipped to GOOD.** Per the pre-defined verdict
gate (≥3 of 4 flips required for CASCADE CONFIRMED), this is the
**FALSIFIED** outcome.

### Probe sidecar drift validation

The probe captured per-rally profile L2 norms during the post-solve
`_update_profiles` sweep. Under baseline, profiles drift across rallies
(EMA blends each rally's track features with the existing profile state).
Under counterfactual, profiles freeze to first-sample state — drift = 0.

Lower-body histogram L2 range across rallies, per PID:

| Fixture | PID1 | PID2 | PID3 | PID4 |
|---|---|---|---|---|
| **5c756c41** baseline | 0.1721 | 0.0977 | 0.0780 | 0.0317 |
| 5c756c41 dropema | **0.0000** | **0.0000** | **0.0000** | **0.0000** |
| **854bb250** baseline | 0.1295 | 0.0718 | 0.0340 | 0.0442 |
| 854bb250 dropema | **0.0000** | **0.0000** | **0.0000** | **0.0000** |
| **7d77980f** baseline | 0.0951 | 0.1776 | 0.2586 | 0.0705 |
| 7d77980f dropema | **0.0000** | **0.0000** | **0.0000** | **0.0000** |
| **b5fb0594** baseline | 0.1060 | 0.1183 | 0.0644 | 0.1842 |
| b5fb0594 dropema | **0.0000** | **0.0000** | **0.0000** | **0.0000** |

Drift magnitudes under baseline are non-trivial (0.03–0.26 in L2 units).
Under counterfactual, drift is exactly zero — the freeze is wired correctly.
Yet the panel verdicts are identical between passes.

## Why the EMA didn't matter

The blind path runs `MatchSolver`, whose `_assign_rally` builds its cost
matrix from **track-vs-track pairwise costs** (`compute_track_similarity`
against other rallies' members), NOT from accumulated profiles. Profiles
are only rebuilt AFTER MatchSolver converges, in the post-solve
`_update_profiles` sweep at `match_tracker.py:3568-3572`. That sweep populates
profiles for downstream consumers (`team_templates`, `playerProfiles` JSON,
scratchpad replay) — but the per-rally `track_to_player` mapping that becomes
`positions_json` is already frozen by the time the sweep runs.

The flag does affect `process_rally`'s in-loop Hungarian (which DOES use
profiles), but `process_rally`'s `track_to_player` is **discarded and
overwritten** by MatchSolver in the spliced result at
`match_tracker.py:3538-3562`. So profile state never reaches the artifact
the verdict tool reads.

The b5fb0594/r04 OFF-OFF flip cited in
`regression_2026_05_01_7307c1d_revert.md` therefore must come from a
DIFFERENT cross-rally coupling pathway, not the profile EMA.

## What this rules in

The cascade observed in production lives somewhere downstream of the EMA.
Candidates ranked by likelihood:

1. **MatchSolver coordinate-descent coupling.** Iteration K's per-rally
   Hungarian uses iteration K-1's assignments for OTHER rallies (via
   `members_by_cluster`). A wrong assignment in rally M biases rally N's
   pairwise costs in the next iteration. Convergence is to a local optimum
   that depends on iteration order and seed.
2. **Pass-1 `process_rally` side effects** that persist through MatchSolver:
   side-switch detection, sub-track stash, server detection. Any of these
   could differ between runs depending on accumulated state and influence
   downstream pipeline steps that touch `positions_json` (e.g., remap-track-ids
   uses `subTracks` from process_rally output).
3. **Stale DB state** between runs: leftover `appliedFullMapping`, stale
   `canonical_pid_map_json`, sub-track entries. Today's session uncovered
   exactly this — 5c756c41 had legacy synth-ID corruption in canonicalPidMap
   from a pre-revert ref-crop run. Cleaning that changed match-players output.

## Phase 2 implication

Per the plan's acceptance gate:

> **CASCADE FALSIFIED** → Stop, drop EMA hypothesis, write NO-GO memo.

The `EXPERIMENTAL_DROP_PROFILE_EMA` flag and the per-field freeze should
NOT be productionized. Phase 2's "ENABLE_PER_RALLY_PROTOTYPE" rename does
not apply.

The architectural refactor's task 3 (per-rally prototype) needs to be
**redesigned** to target whatever pathway IS the cascade. Specifically:

- The probe sidecars contain per-iteration MatchSolver state (cost
  matrices, assignments, changed_from_prev). The next probe should compare
  this between two consecutive match-players runs on the same fixture
  (same DB state) and look for non-determinism. Any per-iteration state
  that differs between runs is the cascade source.
- The leading hypothesis is now MatchSolver coordinate-descent local
  optima depending on iteration order. Phase 2 candidate: replace the
  forward sweep with a pass that solves all rallies jointly without
  iteration-to-iteration coupling (e.g., spectral, SDP, or seeded global
  Hungarian).

## Probe code disposition

The Phase 1 probe (`MATCH_PLAYERS_PROBE=1`,
`analysis/rallycut/tracking/_profile_drift_probe.py`) is generic enough
to also instrument MatchSolver-internal state (it already captures
per-iteration cost matrices and assignments). Keep it shipped behind the
env flag. The `EXPERIMENTAL_DROP_PROFILE_EMA` flag specifically tests the
EMA hypothesis — keep it for now as a documented falsification artifact
(useful if anyone re-opens the EMA hypothesis with a different fixture
panel), but it should not be promoted to a production flag.

## Files & artifacts

- Probe sidecars: `analysis/reports/profile_drift_probe/*.json` (12 files,
  3 per fixture × 4 fixtures)
- Verdict snapshots: `verdict_baseline.txt`, `verdict_dropema.txt`,
  `verdict_baseline_restore.txt`
- Per-fixture match-players + remap logs: `<short>_<mode>_*.log`
- Analysis output: `analysis_output.md`
- Orchestrator: `analysis/scripts/run_phase1_drift_probe.sh`
- Probe-aware analyzer: `analysis/scripts/analyze_phase1_drift.py`

## Caveats

1. **Aggressive freeze semantics**. The flag freezes profiles after the
   FIRST per-field observation (i.e., the first frame's worth of features).
   A milder ablation (freeze after first RALLY's aggregated EMA) was not
   tested. But given the zero-effect outcome of the strict freeze,
   the milder version would be at most as impactful — strictly less likely
   to flip verdicts.
2. **W4 (`ENABLE_BBOX_SWAP_DETECTION`) was OFF**. Some panel rallies
   (5c756c41/r07's "slow_drift" with PID4 half-shift=0.58) might be
   addressable by W4. Not in scope for this falsification.
3. **Test was on 13 panel rallies across 4 fixtures**. A larger fixture set
   would tighten the falsification but the L2-drift table shows the EMA
   IS doing real work in baseline mode (non-zero drift) — it just doesn't
   propagate through MatchSolver to positions_json on this panel.
