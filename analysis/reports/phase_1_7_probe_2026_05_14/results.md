# Phase 1.7 probe: does player_motion surface missed blocks?

Corpus: 12 trusted-GT videos (titi, toto, lulu, wawa, caco, cece, cici, cuco, gaga, kaka, juju, yeye)
Frame tolerance: ±5

## Step 1 — flag mechanics

- Flag: `ContactDetectionConfig.enable_player_motion_candidates` (default `False`, defined at `analysis/rallycut/tracking/contact_detector.py:231`).
- Gated function: `_find_player_motion_candidates` (at `analysis/rallycut/tracking/contact_detector.py:824`); fired from `_prepare_candidates` step 5f at line 2259.
- Signal: for each ball frame, finds players within bbox-proximity (≤ `player_motion_max_ball_distance=0.20`), then checks peak Δy or Δheight over a ±5-frame window against `player_motion_min_d_y=0.015` / `player_motion_min_d_height=0.015`. Skips frames already within `min_peak_distance_frames=12` of an existing candidate. Returns frame numbers only — no pid attribution.
- Comment in code: "adds 265 candidates but only 9 TPs (3.4% hit rate), hurting classifier LOO CV."

## Step 2 — missed BLOCK cases identified

- Total GT BLOCK rows: 14
- Of those, nearest pipeline action (±5) IS `block`: 0
- Missed cases (probe targets): 13

| # | video | rally (8) | gt_frame | gt_pid | nearest pipe action | nearest pipe pid | Δframes |
|---|---|---|---|---|---|---|---|
| 1 | titi | 1e38daab | 185 | 4 | attack | 1 | 3 |
| 2 | juju | 6022138d | 220 | 1 | attack | 1 | 5 |
| 3 | kaka | f33d7ac8 | 462 | 4 | dig | 1 | 1 |
| 4 | toto | 70bd06c8 | 206 | 2 | attack | 2 | 2 |
| 5 | juju | d810943e | 390 | 2 | attack | 2 | 5 |
| 6 | juju | e03ef981 | 312 | 2 | None | None | — |
| 7 | juju | c89b346b | 227 | 1 | attack | 4 | 5 |
| 8 | juju | acada27e | 241 | 1 | attack | 1 | 4 |
| 9 | caco | 9452ee5a | 190 | 4 | attack | 4 | 5 |
| 10 | yeye | 2d3cb54b | 509 | 3 | attack | 1 | 2 |
| 11 | toto | 67b3e1ad | 173 | 4 | attack | 2 | 4 |
| 12 | caco | cfc464a7 | 336 | 4 | attack | 4 | 2 |
| 13 | cici | d362c7b2 | 241 | None | attack | 1 | 4 |

## Step 3 — probe results per case

| # | video | rally (8) | gt_frame | motion fired ±5? | fired frames (in window) | total motion cands in rally |
|---|---|---|---|---|---|---|
| 1 | titi | 1e38daab | 185 | no | [] | 3 |
| 2 | juju | 6022138d | 220 | no | [] | 0 |
| 3 | kaka | f33d7ac8 | 462 | no | [] | 2 |
| 4 | toto | 70bd06c8 | 206 | no | [] | 0 |
| 5 | juju | d810943e | 390 | no | [] | 2 |
| 6 | juju | e03ef981 | 312 | no | [] | 0 |
| 7 | juju | c89b346b | 227 | no | [] | 2 |
| 8 | juju | acada27e | 241 | no | [] | 0 |
| 9 | caco | 9452ee5a | 190 | YES | [189] | 2 |
| 10 | yeye | 2d3cb54b | 509 | no | [] | 0 |
| 11 | toto | 67b3e1ad | 173 | no | [] | 0 |
| 12 | caco | cfc464a7 | 336 | no | [] | 0 |
| 13 | cici | d362c7b2 | 241 | no | [] | 0 |

Summary:
- **Block recall lift: 1/13 = 7.7%** (cases where a player_motion candidate fires within ±5 of GT block frame)
- Block attribution lift: n/a — the raw generator returns frame numbers only. Attribution would be applied downstream by the standard pose/temporal attribution path. Not measured in this probe (per spec — Step 3 asks about candidate fire, attribution is the second-order question).

## Step 4 — overall FP volume

- Total player_motion candidates (12 videos): 84
- Already covered by existing pipeline contact (within ±5, would be deduped): 13
- Genuinely new candidates: 71
  - Of which, near a GT row (useful): 0
  - Of which, near no GT row (likely FP): 71

**FP rate on new candidates: 71/71 = 100.0%**

Per video:

| video | total motion cands | near pipe contact (dedup) | new near GT | new near no GT |
|---|---|---|---|---|
| titi | 17 | 6 | 0 | 11 |
| toto | 13 | 2 | 0 | 11 |
| lulu | 6 | 0 | 0 | 6 |
| wawa | 6 | 1 | 0 | 5 |
| caco | 3 | 1 | 0 | 2 |
| cece | 4 | 1 | 0 | 3 |
| cici | 4 | 1 | 0 | 3 |
| cuco | 1 | 0 | 0 | 1 |
| gaga | 2 | 0 | 0 | 2 |
| kaka | 12 | 1 | 0 | 11 |
| juju | 7 | 0 | 0 | 7 |
| yeye | 9 | 0 | 0 | 9 |

## Step 5 — root-cause check (re-interpretation)

The Step-3 result ("1/13 motion candidates fire ±5 of GT") looked at the
diff between cands ON and cands OFF. That diff is suppressed by the
generator's own dedup rule: "skip frames within
`min_peak_distance_frames=12` of an existing candidate." To distinguish
**generator can't see this frame** from **generator was suppressed by
dedup**, I re-ran `_find_player_motion_candidates` with an empty
`existing_candidates=[]` (raw, pre-dedup) on each of the 13 missed
cases.

Per-case (Δ ≤ 5 from gt_block_frame):

| # | video | rally (8) | gt | existing cands ±5 | raw motion ±5 |
|---|---|---|---|---|---|
| 1 | titi | 1e38daab | 185 | [182, 186] | [186-190] (5) |
| 2 | juju | 6022138d | 220 | [215, 219, 220] | [215-225] (11) |
| 3 | kaka | f33d7ac8 | 462 | [461] | [] (0) |
| 4 | toto | 70bd06c8 | 206 | [204, 207] | [201-211] (11) |
| 5 | juju | d810943e | 390 | [385, 388] | [385-389] (5) |
| 6 | juju | e03ef981 | 312 | [] | [307, 313-317] (6) |
| 7 | juju | c89b346b | 227 | [222, 223, 231] | [222-232] (11) |
| 8 | juju | acada27e | 241 | [237, 244] | [239-246] (8) |
| 9 | caco | 9452ee5a | 190 | [185, 187] | [185-195] (11) |
| 10 | yeye | 2d3cb54b | 509 | [507, 512] | [504-507, 509-514] (10) |
| 11 | toto | 67b3e1ad | 173 | [169, 172] | [168-178] (11) |
| 12 | caco | cfc464a7 | 336 | [334, 336] | [331-341] (11) |
| 13 | cici | d362c7b2 | 241 | [237, 245] | [236-246] (11) |

**12/13 missed blocks already have an existing candidate frame within
±5 of the GT block.** The pipeline is NOT missing the contact at the
candidate-generation level — it's mis-typing it as `attack` at the
action-classification stage. The 1 case with no existing candidate
(juju e03ef981) is also covered by raw motion (frames 313-317 near GT
312), so player_motion does see it pre-dedup.

In other words:
- The block-recall failure mode is **action-type confusion (attack vs
  block) on already-existing candidates**, not **contact-detector
  candidate-generation recall**.
- Enabling `player_motion` ON its own can't help, because the dedup
  rule suppresses it on 12 of 13 cases. Even if we disabled dedup, the
  duplicate motion frame would still be classified `attack` by the
  same downstream classifier.
- Phase 1.7's stated premise ("relax the candidate generator to surface
  the missed blocks") is empirically wrong. The blocks are already in
  the candidate list; the problem is classifier confusion downstream.

## Verdict

**NO-SHIP** — player_motion candidate-generation is not the right
lever for these 10-13 missed blocks. The pipeline already generates a
candidate at or within ±5 of the GT block frame in 12/13 cases; the
classifier downstream labels it as `attack`. Phase 1.7 as
originally framed (relax candidate generator) cannot help.

Recommended next steps (out of scope for this probe):
1. **Block re-classification** on existing candidates — diagnose what
   makes the classifier output `attack` instead of `block` at those
   12 candidate frames. The existing `block_reclassification.py`
   module exists; check whether its gates are firing on these cases.
2. Investigate `e03ef981` (the 1/13 with no existing candidate ±5)
   separately as a candidate-generator recall miss.

Decision rules from spec:
- block recall ≥5/10 AND FP rate <50% → SHIP-1.7
- block recall ≥5/10 AND FP rate ≥75% → NEEDS-CO-CONDITIONS
- block recall <5/10 → NO-SHIP (generator can't see these blocks)

Step-3 hits 1/13 in raw form, but Step-5 reveals the underlying cause
is **mistyping on existing candidates, not generator blindness**. NO-SHIP
applies, but for a different reason than the spec anticipated.
