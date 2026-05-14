# A3 BLOCK reclassification — labeling checklist

**Goal:** confirm whether each of A3's 49 fleet-wide ATTACK→BLOCK reclassification picks is actually a real block. Open each rally in the rally editor, scrub to the source-video time, look at who is touching the ball at that frame, and update the action type in the editor:

- If the contact at that frame is **actually a block**: change the action type from `attack` to `block` (or leave it as-is if it's already `block`).
- If it's actually an attack: leave it as `attack`.
- If it's something else entirely (dig, receive, mis-detected contact): change accordingly OR delete the action.

**Ship gate:** ≥ 92 % of these (≥ 45 / 49) confirmed as real blocks → A3 ships default-ON, ACTION_PIPELINE_VERSION v2→v3, fleet refreshed. F5 fixed for real (not just opt-in).

**Time estimate:** ~30-60 minutes if you spend ~30-60 s per case.

When you're done, tell me "done with labeling" and I'll re-run the fleet rate measurement against your labels to compute precision and decide ship vs no-ship.

| # | Video | Rally order | Source time | Frame | Rally UUID | Verdict |
|--:|:------|------------:|------------:|------:|:-----------|:-------:|
| 1 | caco | #2 | **1:00.733** | 151 | `cfc464a7-423f-4c65-aa9b-0f0484d1da20` |  |
| 2 | cuco | #2 | **1:16.233** | 205 | `f127f3d5-17ed-4641-8f5a-cbc76a09bdab` |  |
| 3 | cucu | #1 | **0:55.867** | 218 | `7d5fbfb2-0158-463e-bddc-0c38f077688f` |  |
| 4 | dark | #1 | **0:50.936** | 183 | `a154c554-a5bf-417a-8f53-8bdfac929f71` |  |
| 5 | dark | #2 | **1:22.885** | 179 | `98d30bb8-dd79-4925-9961-39b417a9e1ab` |  |
| 6 | dark | #3 | **1:54.588** | 171 | `ca026ea3-9520-45f4-ba70-f92c19a95210` |  |
| 7 | dark | #7 | **3:56.014** | 172 | `cf4cdd43-1ea2-484f-b7e8-d9985678feb0` |  |
| 8 | jaja | #8 | **4:02.467** | 476 | `3cd2782c-b497-48a2-8521-a2fa989d91dd` |  |
| 9 | jeje | #6 | **3:41.200** | 294 | `c42ee5eb-4d4d-4428-89df-f6af001c5fa9` |  |
| 10 | jeje | #12 | **1:07.275** | 352 | `878f9a31-70fd-4ec1-a28b-3eff16891ba1` |  |
| 11 | jiji | #1 | **1:27.733** | 631 | `935ead8c-2fb9-4f1f-8364-3e0af6ed0bca` |  |
| 12 | jojo | #10 | **4:28.435** | 250 | `c5fda2c2-735a-44b2-8415-da32f2b452e0` |  |
| 13 | keke (F5) | #3 | **1:28.333** | 184 | `99091ec6-9fa1-4950-999d-4b9bed0121db` |  |
| 14 | lala | #8 | **4:26.252** | 966 | `2eeb3ae6-cf97-4eeb-9400-28a8060a7636` |  |
| 15 | lili | #0 | **0:43.209** | 256 | `a7b017cd-98d9-40ab-bbe6-851d1630c6ec` |  |
| 16 | lili | #3 | **2:18.704** | 238 | `05f7dae1-287e-4292-89f9-3060fa90c2bc` |  |
| 17 | lili | #8 | **4:41.914** | 222 | `571b8cc2-0c2a-44ef-bf6b-95c6ec2f48cd` |  |
| 18 | lili | #17 | **10:16.549** | 207 | `fb6d01ea-54ac-4568-b28b-61d13bd5c798` |  |
| 19 | lili | #20  | **11:35.528** | 210 | `0b7a2461-3e86-4b80-a793-2b69342a69ad` |  |
| 20 | lolo | #17 | **10:26.067** | 356 | `008b5411-d00c-4678-ace1-93f82be90dc5` |  |
| 21 | lolo | #24 | **13:52.100** | 249 | `6935b412-5e38-4829-81e3-f2d3debfa1e2` |  |
| 22 | lolo | #26 | **14:59.400** | 264 | `47d53c40-8440-44d4-9e23-3a4ae3a80537` |  |
| 23 | mama | #0 | **0:23.533** | 224 | `87ce7bff-2dd3-434e-829c-365e0c53cfcb` |  |
| 24 | mimi | #3 | **1:55.766** | 207 | `f3695225-5958-4c74-bdc9-d1ab5d1a3876` |  |
| 25 | mochi | #2 | **1:33.862** | 197 | `39139435-78f0-4cc8-8ef1-5a5ed4ffe9fc` |  |
| 26 | muchi | #0 | **0:16.752** | 382 | `d5b199ea-f448-4ccf-a1b2-cc04de6720f8` |  |
| 27 | muchi | #1 | **0:51.017** | 310 | `ec483e08-01a2-4118-9dc2-5708ec203e7f` |  |
| 28 | natch | #6 | **6:10.929** | 168 | `f0371bb7-f17f-42b1-83f6-8c28b4e59f41` |  |
| 29 | papa | #3 | **2:28.733** | 412 | `e46a9741-14a5-4aa3-b4cd-d8cdc3552752` |  |
| 30 | pepe | #2 | **1:32.033** | 250 | `4ea1bfa2-8ecb-4e63-9128-ac8ff9edf0c6` |  |
| 31 | pepe | #6 | **3:32.567** | 239 | `6391fb0e-b734-47c4-a49e-0de91ca7e259` |  |
| 32 | rara | #9 | **4:50.500** | 273 | `5b9e0ef2-d3d5-4bea-9e82-73cf04c383e5` |  |
| 33 | ruru | #5 | **2:11.483** | 623 | `3655eb69-b01f-431b-8d99-911ef4c414d7` |  |
| 34 | titi | #4 | **2:03.201** | 173 | `caa96651-13d0-4240-908f-3f367dd32653` |  |
| 35 | titi | #6 | **2:52.702** | 170 | `43b849ec-214d-440b-8307-1938f98cafd1` |  |
| 36 | titi | #28 | **15:18.209** | 359 | `4ad457f6-6cbc-49c7-ab9b-5d2c3edb8ab2` |  |
| 37 | titi | #28 | **15:20.275** | 421 | `4ad457f6-6cbc-49c7-ab9b-5d2c3edb8ab2` |  |
| 38 | toto | #2 | **1:12.743** | 174 | `fcc5dcba-9f9f-4125-920b-46940845ca27` |  |
| 39 | toto | #12 | **6:48.659** | 373 | `1d316b85-b7d9-44fe-a570-10cf8a40bf4f` |  |
| 40 | toto | #17 | **9:31.971** | 302 | `f1f09039-2292-4fcd-8c16-ab03371df190` |  |
| 41 | tutu | #16 | **8:00.024** | 244 | `93b88976-af99-48e2-bf54-f885877fad76` |  |
| 42 | veve | #5 | **2:29.967** | 185 | `24b025fa-7221-488d-b5c0-b17df022a364` |  |
| 43 | vuvu | #0 | **0:10.595** | 574 | `86c9106d-5b30-459d-bab9-810d6a58699a` |  |
| 44 | wewe | #5 | **2:38.291** | 331 | `83790ce7-28fa-4749-b673-45eda804cf09` |  |
| 45 | wiwi | #1 | **0:46.667** | 436 | `7aef7188-0287-4cbb-bd47-3bf36c1121b6` |  |
| 46 | wiwi | #1 | **0:51.283** | 713 | `7aef7188-0287-4cbb-bd47-3bf36c1121b6` |  |
| 47 | wowo | #1 | **0:52.452** | 707 | `169292de-2bef-45ce-acad-32aa0f5e308e` |  |
| 48 | wowo | #3 | **2:05.658** | 391 | `b07b388b-ccaa-44a5-baf5-4826b541a663` |  |
| 49 | yoyo | #2 | **1:36.867** | 506 | `21a9b203-dc92-48dc-8f19-d94835e0e226` |  |

## After you finish

The editor writes to `rally_action_ground_truth`. When you're done, I'll run a measurement script that:

1. For each of these 49 (rally_id, frame) tuples, looks up the GT action type within a ±5-frame tolerance.
2. Counts how many have GT action type = `BLOCK`.
3. Computes precision = (# real blocks) / 49.
4. If precision ≥ 92% → A3 ships default-ON. Else NO-SHIP.

Just tell me **"done with labeling"** when you've worked through enough to give a confident signal. You don't have to label all 49 — even 20 stratified picks would give a strong precision estimate, but more is more decisive.
