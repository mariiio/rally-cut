# S4 (anti-self-touch + trajectory-integral) — labeling checklist

**Goal:** confirm whether each of S4's fleet-wide attribution flips matches
the actual toucher at the contact frame. Open each rally in the rally editor,
scrub to the source-video time, look at who is touching the ball, and:

- If **S4's proposed pick** is the actual toucher: leave the suggested change to s4_pid.
- If **pipeline's pick** (the current attribution) is correct: confirm it / leave unchanged.
- If **neither** is correct: change to whoever actually touched the ball.
- If the contact is a **false positive** (no real ball touch on that frame): delete the action.

The editor writes to `rally_action_ground_truth`. When you finish, tell me **"done with labeling"**.

**Total fleet S4 flip candidates:** 395 across 71 videos.
**This sample:** 28 cases stratified across buckets:

- `attack_after_attack`: 7
- `attack_after_set`: 7
- `same_team_chain`: 6
- `cross_team_prev`: 4
- `probe_b_cascade_anchor`: 2
- `same_team_other`: 2

**Stratification notes (anti-self-touch fires unless noted):**

- `probe_b_cascade_anchor` — the two titi/a0881d82 frames from Probe B (f128, f225). Already in your mental model.
- `attack_after_attack` — same-team ATTACK<-ATTACK flips (10 fleet-wide, we sample 7). The highest-risk failure mode for S4: "same player can't attack twice in a row" is heuristic, not a hard volleyball rule (recovery touch + spike CAN happen). If S4 wins here it's defensible; if it loses on most cases, the rule itself is questionable.
- `attack_after_set` — same-team ATTACK<-SET flips (39 fleet-wide, we sample 7). Classic setter-also-attacker pipeline confusion. This is the biggest within-team flip type and the most likely place S4 helps.
- `same_team_chain` — same-team SET/RECEIVE/DIG <- prev-same-team contact (175 fleet-wide, we sample 6). Sampled by descending alt_ratio (most decisive S4 trajectory-integral disagreement).
- `same_team_other` — same-team chain with other action_type pairs (31 fleet-wide, e.g. BLOCK<-SET, ATTACK<-DIG, ATTACK<-RECEIVE; we sample 2).
- `cross_team_prev` — previous action was OPPOSITE team (140 fleet-wide, we sample 4). Anti-self-touch is a no-op here; S4 differs from pipeline purely because trajectory-integral disagrees with ball-position-at-contact. Example: DIG after an opposing ATTACK.

**Time estimate:** ~30-45 minutes if you spend ~60s per case (browser navigation + frame inspection).

| # | Bucket | Video | Rally order | Source time | Frame | Action | Prev | Pipeline pick | S4 proposed | Rally UUID | Verdict |
|--:|:-------|:------|------------:|------------:|------:|:-------|:-----|:-------------|:------------|:-----------|:-------:|
| 1 | probe_b_cascade_anchor | titi | #20 | **10:32.506** | 128 | SET | f76 RECEIVE p2(B) | p2 (B) | **p1 (B)** | `a0881d82-bd3e-4664-bd11-4e672ca29aa6` |  |
| 2 | probe_b_cascade_anchor | titi | #20 | **10:35.739** | 225 | DIG | f176 ATTACK p2(B) | p2 (B) | **p1 (B)** | `a0881d82-bd3e-4664-bd11-4e672ca29aa6` |  |
| 3 | attack_after_attack | yaya | #5 | **3:48.333** | 304 | ATTACK | f251 ATTACK p1(B) | p1 (B) | **p2 (B)** | `1f5ff17d-014f-4093-b6e6-c85360fb1638` |  |
| 4 | attack_after_attack | matttch | #7 | **3:40.049** | 296 | ATTACK | f220 ATTACK p3(B) | p3 (B) | **p4 (B)** | `8d3205ed-b0dc-4c0c-bc24-fda34554e45f` |  |
| 5 | attack_after_attack | lulu | #8 | **3:46.542** | 558 | ATTACK | f439 ATTACK p1(A) | p1 (A) | **p2 (A)** | `55565c2b-49b2-4817-b046-7bef02569b0e` |  |
| 6 | attack_after_attack | yoyo | #2 | **1:37.234** | 528 | ATTACK | f506 ATTACK p1(A) | p1 (A) | **p2 (A)** | `21a9b203-dc92-48dc-8f19-d94835e0e226` |  |
| 7 | attack_after_attack | meme | #2 | **0:54.267** | 314 | ATTACK | f265 ATTACK p1(A) | p1 (A) | **p2 (A)** | `49582c29-f09c-4b39-8435-bd19451812c8` |  |
| 8 | attack_after_attack | jiji | #9 | **5:37.667** | 380 | ATTACK | f304 ATTACK p4(A) | p4 (A) | **p3 (A)** | `0793ebd2-5301-4317-a7ba-3fb3bf9fe368` |  |
| 9 | attack_after_attack | meme | #3 | **1:13.533** | 220 | ATTACK | f163 ATTACK p1(A) | p1 (A) | **p2 (A)** | `65c54107-6c0f-4888-9600-6ec06939c8b5` |  |
| 10 | attack_after_set | titi | #9 | **4:51.869** | 193 | ATTACK | f144 SET p1(B) | p1 (B) | **p2 (B)** | `594a832a-ab27-480e-976a-268c8f101559` |  |
| 11 | attack_after_set | lili | #15 | **9:00.839** | 230 | ATTACK | f181 SET p3(A) | p3 (A) | **p4 (A)** | `ded9504c-ce40-4808-8d12-8bcc3b974b62` |  |
| 12 | attack_after_set | match | #3 | **1:55.818** | 159 | ATTACK | f111 SET p2(A) | p2 (A) | **p1 (A)** | `920d4a33-ae9a-4299-ac19-8e773ca76e61` |  |
| 13 | attack_after_set | toto | #19 | **11:02.400** | 679 | ATTACK | f627 SET p1(B) | p1 (B) | **p2 (B)** | `a1a5baf7-e249-476e-9ab7-cf60fac04111` |  |
| 14 | attack_after_set | vivi | #6 | **3:02.056** | 533 | ATTACK | f461 SET p1(B) | p1 (B) | **p2 (B)** | `eeecf5a0-def4-43dd-8fc9-72e502165c2d` |  |
| 15 | attack_after_set | tata | #4 | **3:37.167** | 233 | ATTACK | f196 SET p2(A) | p2 (A) | **p1 (A)** | `e50f127e-3952-4ba2-9047-39fff14a2e25` |  |
| 16 | attack_after_set | matchop | #4 | **2:10.998** | 177 | ATTACK | f134 SET p4(B) | p4 (B) | **p3 (B)** | `f62bc819-d7c4-4287-b617-edac3f5194bc` |  |
| 17 | same_team_chain | yiyi | #4 | **2:07.666** | 389 | DIG | f356 ATTACK p4(B) | p4 (B) | **p3 (B)** | `227cf416-f217-4226-a9c6-f70d721ae8f4` |  |
| 18 | same_team_chain | mech | #4 | **2:17.671** | 59 | RECEIVE | f39 SERVE p3(A) | p3 (A) | **p4 (A)** | `b0dabe43-7ddb-4544-8d2c-e86032a8d8f5` |  |
| 19 | same_team_chain | gaga | #0 | **0:18.307** | 355 | SET | f301 DIG p3(B) | p3 (B) | **p4 (B)** | `d474b2ad-17e7-4c70-96f6-e52a38533d18` |  |
| 20 | same_team_chain | mimi | #4 | **2:25.366** | 231 | DIG | f199 ATTACK p4(A) | p4 (A) | **p3 (A)** | `bbb22a1a-0079-4172-9605-2cd34d2999b3` |  |
| 21 | same_team_chain | matchop | #6 | **2:53.977** | 100 | RECEIVE | f83 RECEIVE p4(A) | p4 (A) | **p3 (A)** | `f433967e-2c40-4169-b5cb-87f48cd0fa63` |  |
| 22 | same_team_chain | yoyo | #24 | **13:52.124** | 1030 | DIG | f1003 ATTACK p2(B) | p2 (B) | **p1 (B)** | `8ec18b68-81fa-42a0-9075-ed4e08074587` |  |
| 23 | same_team_other | rere | #6 | **3:15.167** | 413 | BLOCK | f358 SET p4(A) | p4 (A) | **p3 (A)** | `4f73e54c-b1b2-4fa8-a90d-044287987aa6` |  |
| 24 | same_team_other | mech | #7 | **3:33.613** | 127 | ATTACK | f70 RECEIVE p1(B) | p1 (B) | **p2 (B)** | `f8e251d8-c7ee-40fd-946c-502067343936` |  |
| 25 | cross_team_prev | wiwi | #1 | **0:47.883** | 509 | DIG | f436 ATTACK p4(A) | p2 (B) | **p1 (B)** | `7aef7188-0287-4cbb-bd47-3bf36c1121b6` |  |
| 26 | cross_team_prev | veve | #1 | **0:39.667** | 92 | RECEIVE | f55 SERVE p2(A) | p4 (B) | **p3 (B)** | `8276604f-9504-4116-94ef-96224feefca0` |  |
| 27 | cross_team_prev | mech | #0 | **0:24.193** | 302 | DIG | f283 ATTACK p1(A) | p3 (B) | **p4 (B)** | `fad29c31-6e2a-4a8d-86f1-9064b2f1f425` |  |
| 28 | cross_team_prev | vava | #6 | **3:21.367** | 239 | BLOCK | f228 ATTACK p3(A) | p2 (B) | **p1 (B)** | `84119722-400c-4e1f-9004-e1f052d5df84` |  |

## How to label each row

1. Open the rally in the editor (URL pattern: `/videos/<video>/rallies/<rally_uuid>`).
2. Scrub to the **source time** column (= rally start + frame/fps, this is the global video time).
3. Find the action at that frame in the editor's action list — its current player ID matches **Pipeline pick**.
4. Decide whether that player actually touched the ball at that frame.
    - If yes: leave it as Pipeline pick.
    - If S4 proposed pick is right: change the player to S4's pick.
    - If someone else: change to that player.
    - If it's a non-contact: delete the action.

## After you finish

I'll run a measurement script that looks up the GT `resolved_track_id` for each (rally_id, frame) within a +/-5-frame tolerance and computes three precision metrics:

- **S4 precision** = fraction of flips where GT matches S4's pick (the ship signal — high precision means S4's flip is correct).
- **Pipeline precision (on the same set)** = fraction where GT matches the pipeline's original pick (= the harm rate — high means S4 would break more than it fixes).
- **Neither precision** = fraction where GT is neither (= cases where S4 is wrong but in a different way than pipeline).

**Ship gate (preliminary, will refine after seeing data):** S4 precision >= 70% AND pipeline precision <= 30% AND lift (S4 - pipeline) >= +30 pp on this sample of 25-30 → SHIP S4. Otherwise NO-SHIP.
