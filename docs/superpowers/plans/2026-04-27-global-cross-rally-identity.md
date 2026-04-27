# Cross-Rally Identity — Problem Brief for Next Session

> **Date:** 2026-04-27
> **Status:** Problem brief, NOT a plan. The architecture, solver choice, phasing, and validation surfaces are intentionally left open. The next session should brainstorm against this brief and propose the approach — don't take this as a directive.
> **Why a brief instead of a plan:** the failure modes we hit on wawa rally 10 (and the Phase-1 patches I just shipped) are symptoms of a structural mismatch, but the optimal architecture depends on weighing tradeoffs that need a fresh look. Several plausible approaches are listed in §6 — none are recommended.

---

## 1. The empirical failure (anchor case)

User reported on wawa (`5c756c41`) after retracking:
- Rallies 1, 4-9 (1-indexed): correct identity.
- Rally 2: within-team swap (woman should be pid 3, was pid 4).
- Rally 3: cross-team swap pid 2 ↔ pid 4.
- Rally 10: complete swap "1/2 turned into 3/4" — visually-team-A bodies got pids 3 and 4.
- Confidence dialog showed 93% on jojo rally 1 even though tracks T1 and T26 were the same physical body.

**Note on top-tracks count.** The diagnostic dump reports `top_tracks=[1, 3, 4]` for rally 0 and 4 tracks for rally 10. Both rallies actually have 4 physical players present; the 3-track top set is a *primary-track filtering* artifact — rallies routinely start with the server off-screen or with one player occluded for a while, and `identify_primary_tracks` excludes tracks that don't meet duration/coverage thresholds. The "rally 0 has 3 top tracks" is a SYMPTOM of treating rally 0 as the seeding authority, not a video-quality problem. Any fixture's first rally can present this way.

The 2026-04-27 diagnostic (`scripts/diagnose_side_switch_detector.py --video-id 5c756c41 --rally-index 9`) traced rally 10 to a specific Hungarian state:

| pid | rally_count | upper_hist_count | reid_count | skin_h | dominant_h |
|---|---|---|---|---|---|
| 1 | 1 | 12 | 0 | 12.08 | 164.90 |
| **2** | **0** | **0** | **0** | **None** | **None** |
| 3 | 1 | 12 | 0 | 10.55 | 3.48 |
| 4 | 1 | 12 | 0 | 11.86 | 0.56 |

Pid 2 was a **wildcard slot with empty profile** entering Hungarian on rally 10. Trace:
- Rally 0 had `top_tracks=[1, 3, 4]` — only 3 tracks. Pids 1, 3, 4 got profile seeds; pid 2 got nothing.
- All subsequent rallies had `confidence ∈ [0.44, 0.62]`, every one below `MIN_PROFILE_UPDATE_CONFIDENCE = 0.80`. No profile updates fired.
- Pid 2's profile stayed empty for all 10 rallies. Hungarian then routed any "free" track to it.

Side-switch detection at rally 7 was correct (matches user's "rally 7→8 switch"). The detector is fine on this case. The bug is **upstream of side-switch detection, in the profile-accumulation path itself**.

The other failure modes (rally 2 within-team, rally 3 cross-team) are not yet diagnosed at the same depth but appear to be variants of the same shape: forward-incremental decisions made under uncertainty, with no later opportunity to revise them with full-match evidence.

---

## 2. What we know about the architecture today

The cross-rally identity layer is implemented in `analysis/rallycut/tracking/match_tracker.py` (3300 lines). Decision flow:

```
Pass 1 (sequential, per-rally):
  rally 0:  _initialize_first_rally  (Y-sort seed, only 4 tracks if available)
  rally 1+: _assign_tracks_to_players_global  (Hungarian vs accumulated profiles)
            _refine_within_team                (position continuity)
            _compute_assignment_confidence
            _update_profiles                   (gated by MIN_PROFILE_UPDATE_CONFIDENCE = 0.80)

Pass 2 (post-hoc, all rallies):
  Stage 0: _detect_side_switches_combinatorial  (binary orientation per rally, 1024-partition search)
  Stage 1: _assign_tracks_to_players_global     (re-Hungarian per rally with final profiles)
  Stage 2: _global_within_team_voting           (cross-rally pairwise agreement to fix within-team)

Phase-1 patches (just shipped, 2026-04-27):
  Step 1: _classify_sides_by_bbox_height        (perspective-based side signal)
  Step 2: _select_seed_rally + _within_team_permutation_from_seed  (re-anchor canonical pid layout)
  Step 3: _high_confidence_sides_for_team_pair + HARD_TEAM_PAIR_COST  (forbid cross-team Hungarian)
```

Every stage and constant in this stack exists to compensate for an earlier stage's commitment under uncertainty. `MIN_PROFILE_UPDATE_CONFIDENCE = 0.80` exists because forward-accumulation drifts. `_detect_side_switches_combinatorial` exists because per-rally orientation isn't trusted. `_global_within_team_voting` exists to undo Pass-1 within-team errors. My Phase-1 Steps 2 and 3 exist to undo rally-0 lock-in and margin-tight cross-team swaps. The structure is paying compound interest on the same wrong assumption.

---

## 3. Signals and evidence we already have

**Primary scope: blind analysis without reference crops must work first.** Reference crops are the post-label authority and reach 95% on Day-4, but the blind path is the foundation — if blind output is wrong, ref-crop relabel is fixing wrong primitives. The next session targets blind quality first; ref-crop integration is a follow-up after blind is solid.

The next session should leverage everything below before proposing an approach. **Critically: don't take any signal as automatically valid. Audit each one — some may be net-negative or redundant.**

**Per-rally extraction (already produced and serialized today):**
- `TrackAppearanceStats` — HSV upper/lower body histograms, V histograms, head histogram, dominant color, skin tone, optional 384-dim DINOv2 ReID embedding.
- `track_court_sides` — per-track {0=near, 1=far} from y-coordinate / court-split / team_assignments / calibration.
- `sides_by_bbox` — perspective clustering by median bbox height (Phase-1 step 1).
- `early_positions` — start-of-rally (x,y) per top track.
- Inter-rally gaps (`start_ms`, `end_ms`).
- Serve direction inferred from ball trajectory.
- `top_tracks` — selected primary tracks (up to 4).
- Rally-rally pairwise team appearance preference matrix (`_team_match_cost`-based, computed every match).

**Reference crops (when user provides):**
- DB-stored player_id → list of BGR crops, frozen profile anchors. Day-4 measurement: 95.22% direct accuracy on 8 click-GT fixtures (`memory/player_attribution_day4_2026_04_23.md`). Hard constraint authority.
- Today's machinery treats them as profile seeds for Pass-1; could be pinned anchors in any new architecture.

**Court calibration (when keypoints detected):**
- Homography for image→court projection. Hard side classification with `SIDE_PENALTY_CALIBRATED = 1.0`. Available on a meaningful subset of videos (`Video.calibration` table).

**Per-fixture anchors:**
- **cece** (`950fbe5d-fdad-4862-b05d-8b374bdd5ec6`) — 5 rallies, no side switches, perfect identity preservation in production. *Positive control — the easy case the architecture must not regress.*
- **2d105b7b** (`2d105b7b-12be-476e-b010-1b274380b891`) — 7 rallies, 1 GT side switch at index 3 (user rally 3→4), perfect identity preservation **even though the detector reports HIT=0/1 (missed the switch entirely)**. *Diagnostic gold: shows the rest of the pipeline (per-rally Hungarian + within-team voting) can compensate for a missed switch when appearance is strong enough. Tells us identity correctness and side-switch correctness are partially decoupled — rebuilding the side-switch detector alone wouldn't have fixed cases like this, and breaking it alone wouldn't have hurt them.*
- **wawa** (`5c756c41`) — 10 rallies, 1 GT switch at index 7 (detected correctly), "complete swap 1/2 → 3/4" at rally 10. *Failure case driving the rewrite.* Use the rally-9 cost-matrix dump as the primary diagnostic.
- **jojo** (`38f65800`) — 14 rallies, 1 GT switch at index 8 (detected at 7, off by 1), cross-team swaps in user-rallies 9-13, T1+T26 same-physical-body confidence inflation. *Multi-failure-mode case.*
- Additional fixtures in the click-GT and 9-fixture sets below.

**Validation surfaces (existing GT):**
- 9-fixture locked baseline: 468 actions across 9 videos (`analysis/reports/attribution_rebuild/baseline_2026_04_24.json`). Today: 43.8% blind player_attr.
- Click-GT direct accuracy on 8 fixtures (cece / rere / tata / lulu / yeye / wawa / wewe / 0a383519 / 7d77980f) — 314 actions. Today: 33.8% blind, 95.22% with ref crops (Day-4).
- 316 GT rallies with side-switch labels (46 videos, 57 switches) — `memory/diagnosis_2026-04-10.md`.
- Tracking GT for HOTA/IDsw evaluation (`evaluate-tracking`).

**Closed NO-GOs — re-open if there's high-confidence reason or a new idea.** "Closed" reflects the evidence at the time of closure, not a permanent verdict. If the new architectural framing changes the role of one of these (e.g. a NO-GO that failed as a primary signal might work as a secondary feature in a global solver), it's worth a fresh probe. Likewise if a new idea emerges that wasn't on the table during the original sessions.
- Within-team ReID heads (sessions 1-9, `memory/within_team_reid_project_2026_04_16.md`, `memory/session9_individual_reid_probe_2026_04_17.md`). DINOv2 lacked within-team individual signal *as an end-to-end veto*. Whether it adds value as a soft input to a global solver is a separate question.
- Crop-guided per-action classifier (`memory/crop_guided_identity_nogo_2026_04_19.md`). Net-negative on attribution *with the existing pipeline*. Same caveat applies.
- Per-frame Hungarian as universal architectural rebuild (`memory/per_frame_hungarian_probe_2026_04_26_PRELIMINARY.md`). Aggregate −9 to −11pp; FALSIFIED as a rebuild but ALIVE as a narrow rescue. Note this finding is partially contaminated — verify on fresh GT before relying on it either way.
- Phase-3 global k-means seeding (`memory/attribution_primitive_first_phase0_2026_04_24.md`). Closed; superseded by Day-4 ref crops in the post-label path. Whether it's useful as the *blind* path's seeding mechanism is independent of the closure reason.

**New ideas worth considering** (none of these have been probed yet):
- Pose-keypoints as an identity feature (height-ratio, build, gait) — orthogonal to color/texture which dominate today.
- Per-track motion signatures across frames — court-coverage patterns differ between players (server vs blocker, attacker vs defender).
- Trajectory continuity across rally boundaries — reusing the late-position-to-early-position bridge today's `_refine_within_team` uses, but globally.
- Track-fragmentation merge confidence as an explicit signal (today's pipeline merges silently).

**Sensing ceilings (architectural limits, not bugs):**
- Same-uniform teammates (regime 3, e.g., titi, 2e984c43, yeye). Day-4 hits 84.6% per-rally; no ML lever has moved this in 9 sessions. Document, don't try to solve in this rewrite.
- Contact-detector recall: 22% of actions have no candidate. Closed workstream after 4 prior NO-GOs. Identity layer cannot fix actions it never sees.

---

## 4. What the next session should answer

These are open questions, not preordained tasks:

1. **Is the architecture itself wrong, or are the patches wrong?** Today's failure modes can be read either way. Pid 2 being unseeded could be fixed by (a) bootstrapping empty profiles, (b) rebuilding pass-1 from a later anchor rally, or (c) replacing the sequential decision layer entirely. The "correct" answer depends on whether the same root cause affects fixtures we haven't examined yet — and on how much code we're willing to rewrite vs. patch.
2. **Which signals are underused, redundant, or actively hurting?** Don't take any signal as automatically valid. Examples to audit, not assume: HSV histograms might be unstable across lighting changes; ReID may not generalize beyond its training domain; the y-side classifier may be net-negative when court_split_y is noisy; the bbox-side classifier (Phase-1 step 1) may strictly subsume the y-side classifier (or vice versa); the side-switch detector's binary output is provably noisy (71%/60% recall/precision per the 2026-04-27 audit) but its preference matrix may still be useful. Also explore signals NOT used today: pose keypoints, per-track motion patterns, trajectory continuity at rally boundaries, calibration confidence per rally. If a signal is hurting, drop it; if missing, add it.
3. **What's the right validation surface?** click-GT direct accuracy is a tight per-action measure but only covers 8 fixtures. The 9-fixture locked baseline is broader but has its own contamination concerns (`memory/knowledge_state_2026_04_26.md`). The next session may decide a different surface is needed.
4. **Where does ref-crop authority sit in the new design?** This is a follow-up question, NOT the primary scope. Blind quality is the priority. Once blind is solid, decide: ref crops as hard constraints? Cluster anchors? Both? Contract when only 1-3 of 4 are provided? Day-4's 95% is the post-label ceiling once blind output is correct primitives.
5. **What's the right scope for "this session"?** Full rewrite? Targeted refactor of the decision layer with extraction kept verbatim? Patch the most-failing path? The brief above lists the empirical evidence for "structural", but it's not a foregone conclusion.

---

## 5. Constraints

- **Don't break existing production.** Run the new approach in parallel to the current sequential layer until it clears a measurable gate.
- **Don't re-open closed workstreams without new evidence.** §3 lists what's NO-GO.
- **Match-tracker reads/writes via `match_analysis_json.rallyScratchpad`** (already serialized; see `match_tracker.py:486` for `StoredRallyData.to_dict`). Schema changes are observable; coordinate with the relabel-with-crops worker (`match_players.py:18` JSON-file branch).
- **Reference-crop convention is authoritative**: pid labels VERBATIM from the user; never permute to align with legacy Hungarian (`memory/canonical_pid_no_alignment_2026_04_25.md`).

---

## 6. Approaches that have been mentioned (NOT recommended, just enumerated)

To save the next session from re-deriving these from scratch — these came up in 2026-04-27's discussion. Each has tradeoffs the next session should weigh independently.

- **Extract-then-decide with global joint identity + orientation co-clustering.** Run extraction per-rally with no chronology; solve cluster labels and orientation labels together as one optimization. Pros: principled, side-switch falls out of the orientation solve. Cons: 2-3 week rewrite; local-optima risk in iterative formulations.
- **Coordinate-descent / EM over the appearance graph.** Alternate fix-orientations-cluster, fix-clusters-flip-orientations. Pros: cheap, drop-in replacement for `refine_assignments`. Cons: needs careful initialization; may not be a strict improvement over today's pipeline.
- **ILP / MILP global solve.** Exact solution if it scales. Pros: explainable, ref crops as hard constraints are clean. Cons: encoding effort; uncertain scalability to long matches.
- **Targeted patches on top of the current architecture.** Empty-profile bootstrap + lower confidence gate for short clips + something for the rally-0-fragile case. Pros: minimal change, fast. Cons: more compound-interest on the wrong shape (per the §1-2 read), but maybe that read is wrong.
- **Per-frame Hungarian.** Already probed 2026-04-26, partially falsified as a rebuild. Re-probe with cleaner GT before considering.

---

## 7. Where to start a fresh session

1. Read `memory/MEMORY.md` top-down (the 2026-04-25/26 contamination notice matters).
2. Read the wawa diagnostic output: `scripts/diagnose_side_switch_detector.py --video-id 5c756c41 --rally-index 9`.
3. Read `match_tracker.py` end-to-end — it's the entire problem surface. ~3300 lines but heavily commented.
4. Brainstorm against §4's open questions. Don't take §6 as a recommendation.
5. Propose a session plan back to the user with: chosen approach, kill gate, fixtures, scope, time budget.

The Phase-1 Step 1/2/3 commits (`af73b61`, `a5d45f9`, `0f667fb`) are the most recent edits; they may or may not survive the next architecture call.
