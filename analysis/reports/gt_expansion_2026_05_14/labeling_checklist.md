# T2 GT-expansion labeling checklist

**Goal**: ~150 contacts labeled across ~30 rallies. Unlocks A1.v2 precision measurement, PGM Phase B per-action-type weights, contact-detector at-net FP retrain.

**Sample**: 150 contacts across 47 rallies and 37 videos.

**Time estimate**: ~60-90 s per contact = ~2.5-3.5 hours total.

**How to label**: open each rally in the rally editor at the listed source-video time, look at the contact, and either:
- Confirm pipeline pick (do nothing, GT row created as-is)
- Correct attribution (change player in editor → resolver updates GT row)
- Correct action type (change type)
- Delete (if the contact is a false positive / non-event)

**Counts per bucket:**

- `A_cascade`: 40
- `B_cross_team`: 30
- `C_at_net_attack`: 30
- `D_random_control`: 30
- `E_confidence_low`: 5
- `E_confidence_mid`: 5
- `E_confidence_high`: 10

## Bucket A — Cascade-shape (same-player back-to-back)

| # | Video | Rally (#order, uuid) | Frame | Source time | Pipeline pick (pid / action) | Prev (frame / pid / action) | Conf | Notes |
|--:|:------|:--------------------|------:|:------------|:-----------------------------|:----------------------------|----:|:------|
| 1 | titi | #22 `a0881d82` | 225 | **10:35.739** | p2 / dig | f176 / p2 / attack | 0.69 | anchor titi a0881d82 cascade |
| 2 | titi | #22 `a0881d82` | 128 | **10:32.505** | p2 / set | f76 / p2 / receive | 0.77 | anchor titi a0881d82 cascade |
| 3 | rara | #5 `88e3503c` | 179 | **2:17.366** | p2 / attack | f152 / p1 / receive | 0.80 | alt_ratio=7.55 |
| 4 | rara | #5 `88e3503c` | 274 | **2:20.533** | p2 / set | f179 / p2 / attack | 0.67 | alt_ratio=7.55 |
| 5 | rara | #5 `88e3503c` | 386 | **2:24.266** | p1 / dig | f351 / p1 / attack | 0.62 | alt_ratio=5.32 |
| 6 | rara | #5 `88e3503c` | 431 | **2:25.766** | p1 / set | f386 / p1 / dig | 0.78 | alt_ratio=5.32 |
| 7 | rara | #5 `88e3503c` | 480 | **2:27.400** | p1 / attack | f431 / p1 / set | 0.95 | alt_ratio=4.27 |
| 8 | yoyo | #25 `8ec18b68` | 340 | **13:40.623** | p1 / dig | f306 / p1 / attack | 0.60 | alt_ratio=4.32 |
| 9 | yoyo | #25 `8ec18b68` | 443 | **13:42.340** | p1 / set | f340 / p1 / dig | 0.60 | alt_ratio=4.32 |
| 10 | yoyo | #25 `8ec18b68` | 750 | **13:47.457** | p1 / dig | f443 / p1 / set | 0.43 | alt_ratio=3.80 |
| 11 | yoyo | #25 `8ec18b68` | 1003 | **13:51.673** | p2 / attack | f929 / p1 / set | 0.54 | alt_ratio=2.55 |
| 12 | yoyo | #25 `8ec18b68` | 1030 | **13:52.123** | p2 / dig | f1003 / p2 / attack | 0.52 | alt_ratio=2.55 |
| 13 | toto | #25 `974d4b76` | 593 | **14:01.256** | p3 / attack | f545 / p4 / set | 0.95 | alt_ratio=18.15 |
| 14 | toto | #25 `974d4b76` | 632 | **14:02.560** | p3 / dig | f593 / p3 / attack | 0.45 | alt_ratio=18.15 |
| 15 | toto | #25 `974d4b76` | 79 | **13:44.075** | p3 / serve | — | 0.70 | alt_ratio=7.07 |
| 16 | toto | #25 `974d4b76` | 115 | **13:45.279** | p3 / receive | f79 / p3 / serve | 0.90 | alt_ratio=7.07 |
| 17 | toto | #25 `974d4b76` | 429 | **13:55.774** | p2 / attack | f392 / p3 / set | 0.73 | alt_ratio=3.63 |
| 18 | matchop | #8 `f433967e` | 83 | **2:53.409** | p4 / receive | f50 / p4 / serve | 0.58 | alt_ratio=5.42 |
| 19 | matchop | #8 `f433967e` | 100 | **2:53.976** | p4 / receive | f83 / p4 / receive | 0.33 | alt_ratio=5.42 |
| 20 | matchop | #8 `f433967e` | 50 | **2:52.308** | p4 / serve | — | 0.60 | alt_ratio=3.22 |
| 21 | matchop | #8 `f433967e` | 122 | **2:54.710** | p4 / set | f100 / p4 / receive | 0.60 | alt_ratio=1.87 |
| 22 | matchop | #8 `f433967e` | 175 | **2:56.479** | p2 / attack | f122 / p4 / set | 0.58 | cascade-rally neighbor (densified) |
| 23 | yoyo | #10 `adf3e9c8` | 832 | **4:51.183** | p4 / set | f735 / p3 / attack | 0.50 | alt_ratio=5.20 |
| 24 | yoyo | #10 `adf3e9c8` | 926 | **4:52.750** | p4 / set | f832 / p4 / set | 0.81 | alt_ratio=5.20 |
| 25 | yoyo | #10 `adf3e9c8` | 522 | **4:46.017** | p2 / dig | f477 / p3 / attack | 0.84 | alt_ratio=4.09 |
| 26 | popo | #8 `2001b635` | 189 | **3:17.300** | p3 / dig | f169 / p3 / attack | 0.92 | alt_ratio=8.33 |
| 27 | popo | #8 `2001b635` | 219 | **3:18.300** | p3 / dig | f189 / p3 / dig | 0.58 | alt_ratio=8.33 |
| 28 | popo | #8 `2001b635` | 169 | **3:16.633** | p3 / attack | f126 / p4 / set | 0.91 | alt_ratio=3.82 |
| 29 | popo | #8 `2001b635` | 41 | **3:12.366** | p1 / serve | — | 0.90 | cascade-rally neighbor (densified) |
| 30 | popo | #8 `2001b635` | 74 | **3:13.466** | p3 / receive | f41 / p1 / serve | 0.85 | cascade-rally neighbor (densified) |
| 31 | titi | #16 `35aec2b6` | 64 | **7:53.571** | p3 / receive | f32 / p2 / serve | 0.70 | alt_ratio=7.58 |
| 32 | titi | #16 `35aec2b6` | 111 | **7:55.138** | p3 / set | f64 / p3 / receive | 0.78 | alt_ratio=7.58 |
| 33 | titi | #16 `35aec2b6` | 154 | **7:56.571** | p3 / attack | f111 / p3 / set | 0.60 | alt_ratio=3.29 |
| 34 | titi | #16 `35aec2b6` | 32 | **7:52.504** | p2 / serve | — | 0.70 | cascade-rally neighbor (densified) |
| 35 | titi | #16 `35aec2b6` | 181 | **7:57.471** | p2 / dig | f154 / p3 / attack | 0.44 | cascade-rally neighbor (densified) |
| 36 | caca | #3 `8449bc3b` | 694 | **1:23.366** | p1 / attack | f596 / p2 / set | 0.91 | alt_ratio=6.84 |
| 37 | caca | #3 `8449bc3b` | 737 | **1:24.083** | p1 / dig | f694 / p1 / attack | 0.43 | alt_ratio=6.84 |
| 38 | caca | #3 `8449bc3b` | 351 | **1:17.650** | p4 / set | f201 / p2 / serve | 0.21 | alt_ratio=3.28 |
| 39 | caca | #3 `8449bc3b` | 466 | **1:19.566** | p4 / attack | f351 / p4 / set | 0.60 | alt_ratio=3.28 |
| 40 | caca | #3 `8449bc3b` | 201 | **1:15.150** | p2 / serve | — | 0.70 | cascade-rally neighbor (densified) |

## Bucket B — Mid-rally cross-team (F3/F5 shape)

| # | Video | Rally (#order, uuid) | Frame | Source time | Pipeline pick (pid / action) | Prev (frame / pid / action) | Conf | Notes |
|--:|:------|:--------------------|------:|:------------|:-----------------------------|:----------------------------|----:|:------|
| 41 | vivi | #2 `8e9ce257` | 1867 | **1:01.258** | p1 / dig | f1699 / p4 / set | 0.38 | C-5 cross-team B->A prev=set non-transfer |
| 42 | vivi | #2 `8e9ce257` | 586 | **0:39.652** | p2 / set | f471 / p4 / dig | 0.85 | C-5 cross-team B->A prev=dig non-transfer |
| 43 | vivi | #2 `8e9ce257` | 395 | **0:36.431** | p4 / set | f287 / p1 / dig | 0.60 | C-5 cross-team A->B prev=dig non-transfer |
| 44 | vivi | #2 `8e9ce257` | 1350 | **0:52.538** | p3 / set | f1254 / p1 / receive | 0.59 | C-5 cross-team A->B prev=receive non-transfer |
| 45 | vivi | #2 `8e9ce257` | 287 | **0:34.609** | p1 / dig | f166 / p4 / receive | 0.40 | C-5 cross-team B->A prev=receive non-transfer |
| 46 | jeje | #5 `9bef4dda` | 191 | **1:41.337** | p3 / set | f133 / p1 / receive | 0.52 | C-5 cross-team A->B prev=receive non-transfer |
| 47 | jeje | #5 `9bef4dda` | 245 | **1:43.137** | p1 / attack | f191 / p3 / set | 0.93 | C-5 cross-team B->A prev=set non-transfer |
| 48 | jeje | #5 `9bef4dda` | 306 | **1:45.171** | p1 / set | f266 / p4 / dig | 0.71 | C-5 cross-team B->A prev=dig non-transfer |
| 49 | jeje | #5 `9bef4dda` | 588 | **1:54.571** | p2 / attack | f535 / p4 / set | 0.90 | C-5 cross-team B->A prev=set non-transfer |
| 50 | jeje | #5 `9bef4dda` | 414 | **1:48.771** | p1 / set | f382 / p3 / dig | 0.91 | C-5 cross-team B->A prev=dig non-transfer |
| 51 | titi | #25 `66571168` | 257 | **11:50.806** | p3 / attack | f158 / p2 / set | 0.78 | C-5 cross-team B->A prev=set non-transfer |
| 52 | michu | #6 `6efbf348` | 182 | **2:07.732** | p3 / attack | f137 / p2 / set | 0.96 | C-5 cross-team B->A prev=set non-transfer |
| 53 | michu | #6 `6efbf348` | 90 | **2:04.663** | p3 / receive | f55 / p1 / receive | 0.60 | C-5 cross-team B->A prev=receive non-transfer |
| 54 | michu | #6 `6efbf348` | 137 | **2:06.231** | p2 / set | f90 / p3 / receive | 0.84 | C-5 cross-team A->B prev=receive non-transfer |
| 55 | michu | #6 `6efbf348` | 55 | **2:03.495** | p1 / receive | f37 / p-1 / serve | 0.70 | cross-team rally neighbor (densified) |
| 56 | riri | #11 `e7d41f54` | 286 | **4:44.933** | p4 / set | f252 / p1 / dig | 0.83 | C-5 cross-team B->A prev=dig non-transfer |
| 57 | riri | #11 `e7d41f54` | 183 | **4:41.500** | p3 / set | f133 / p1 / receive | 0.96 | C-5 cross-team B->A prev=receive non-transfer |
| 58 | riri | #11 `e7d41f54` | 229 | **4:43.033** | p1 / attack | f183 / p3 / set | 0.91 | C-5 cross-team A->B prev=set non-transfer |
| 59 | riri | #11 `e7d41f54` | 106 | **4:38.933** | p4 / serve | — | 0.70 | cross-team rally neighbor (densified) |
| 60 | riri | #11 `e7d41f54` | 133 | **4:39.833** | p1 / receive | f106 / p4 / serve | 0.48 | cross-team rally neighbor (densified) |
| 61 | riri | #12 `9ddcfd09` | 337 | **5:13.833** | p2 / attack | f301 / p4 / dig | 0.37 | C-5 cross-team A->B prev=dig non-transfer |
| 62 | riri | #12 `9ddcfd09` | 468 | **5:18.200** | p2 / dig | f426 / p3 / set | 0.31 | C-5 cross-team A->B prev=set non-transfer |
| 63 | riri | #12 `9ddcfd09` | 426 | **5:16.800** | p3 / set | f379 / p1 / set | 0.90 | C-5 cross-team B->A prev=set non-transfer |
| 64 | toto | #21 `a1a5baf7` | 151 | **10:44.751** | p1 / set | f104 / p4 / receive | 0.94 | C-5 cross-team A->B prev=receive non-transfer |
| 65 | toto | #21 `a1a5baf7` | 416 | **10:53.609** | p3 / attack | f370 / p1 / set | 0.34 | C-5 cross-team B->A prev=set non-transfer |
| 66 | toto | #21 `a1a5baf7` | 627 | **11:00.662** | p1 / set | f585 / p4 / dig | 0.93 | C-5 cross-team A->B prev=dig non-transfer |
| 67 | kiki | #8 `0fc55658` | 385 | **3:17.230** | p4 / receive | f306 / p1 / set | 0.51 | C-5 cross-team B->A prev=set non-transfer |
| 68 | kiki | #8 `0fc55658` | 685 | **3:22.235** | p1 / dig | f568 / p3 / set | 0.60 | C-5 cross-team A->B prev=set non-transfer |
| 69 | kiki | #8 `0fc55658` | 862 | **3:25.188** | p3 / attack | f765 / p2 / set | 0.72 | C-5 cross-team B->A prev=set non-transfer |
| 70 | kiki | #8 `0fc55658` | 167 | **3:13.593** | p3 / serve | — | 0.70 | cross-team rally neighbor (densified) |

## Bucket C — At-net attacks (block candidates / FP candidates)

| # | Video | Rally (#order, uuid) | Frame | Source time | Pipeline pick (pid / action) | Prev (frame / pid / action) | Conf | Notes |
|--:|:------|:--------------------|------:|:------------|:-----------------------------|:----------------------------|----:|:------|
| 71 | matttch | #9 `8d3205ed` | 220 | **3:37.512** | p3 / ATTACK | f191 / p3 / DIG | 0.39 | s4 disagrees: pipeline=p3 vs s4=p4 |
| 72 | matttch | #9 `8d3205ed` | 296 | **3:40.048** | p3 / ATTACK | f220 / p3 / ATTACK | 0.85 | s4 disagrees: pipeline=p3 vs s4=p4 |
| 73 | matttch | #9 `8d3205ed` | 173 | **3:35.944** | p4 / attack | f120 / p1 / set | 0.33 | at-net rally neighbor (densified) |
| 74 | matttch | #9 `8d3205ed` | 531 | **3:47.889** | p4 / attack | f355 / p1 / dig | 0.71 | at-net rally neighbor (densified) |
| 75 | matttch | #9 `8d3205ed` | 38 | **3:31.439** | p4 / serve | — | 0.70 | at-net rally neighbor (densified) |
| 76 | caca | #1 `f0fdfcdb` | 510 | **0:16.900** | p4 / ATTACK | f420 / p4 / SET | 0.93 | s4 disagrees: pipeline=p4 vs s4=p3 |
| 77 | caca | #1 `f0fdfcdb` | 246 | **0:12.500** | p2 / serve | — | 0.42 | at-net rally neighbor (densified) |
| 78 | caca | #1 `f0fdfcdb` | 420 | **0:15.400** | p4 / set | f246 / p2 / serve | 0.51 | at-net rally neighbor (densified) |
| 79 | caco | #6 `281730b7` | 183 | **2:28.297** | p4 / ATTACK | f76 / p4 / RECEIVE | 0.58 | s4 disagrees: pipeline=p4 vs s4=p3 |
| 80 | caco | #6 `281730b7` | 46 | **2:23.730** | p2 / serve | — | 0.84 | at-net rally neighbor (densified) |
| 81 | caco | #6 `281730b7` | 76 | **2:24.730** | p4 / receive | f46 / p2 / serve | 0.90 | at-net rally neighbor (densified) |
| 82 | cucu | #2 `7d5fbfb2` | 218 | **0:55.866** | p1 / ATTACK | f179 / p3 / SET | 0.60 | s4 disagrees: pipeline=p1 vs s4=p2 |
| 83 | cucu | #2 `7d5fbfb2` | 85 | **0:51.433** | p6 / serve | — | 0.45 | at-net rally neighbor (densified) |
| 84 | cucu | #2 `7d5fbfb2` | 124 | **0:52.733** | p2 / receive | f85 / p6 / serve | 0.88 | at-net rally neighbor (densified) |
| 85 | cucu | #2 `7d5fbfb2` | 179 | **0:54.566** | p3 / set | f124 / p2 / receive | 0.90 | at-net rally neighbor (densified) |
| 86 | cucu | #2 `7d5fbfb2` | 248 | **0:56.866** | p6 / dig | f218 / p1 / attack | 0.42 | at-net rally neighbor (densified) |
| 87 | cucu | #3 `24d00889` | 218 | **1:17.466** | p3 / ATTACK | f167 / p1 / SET | 0.98 | s4 disagrees: pipeline=p3 vs s4=p4 |
| 88 | cucu | #3 `24d00889` | 79 | **1:12.833** | p1 / serve | — | 0.70 | at-net rally neighbor (densified) |
| 89 | cucu | #3 `24d00889` | 114 | **1:14.000** | p3 / receive | f79 / p1 / serve | 0.58 | at-net rally neighbor (densified) |
| 90 | gugu | #4 `2e8b3ce2` | 224 | **1:31.306** | p2 / ATTACK | f174 / p2 / SET | 0.60 | s4 disagrees: pipeline=p2 vs s4=p1 |
| 91 | gugu | #4 `2e8b3ce2` | 112 | **1:27.520** | p1 / serve | — | 0.55 | at-net rally neighbor (densified) |
| 92 | gugu | #4 `2e8b3ce2` | 174 | **1:29.616** | p2 / set | f112 / p1 / serve | 0.42 | at-net rally neighbor (densified) |
| 93 | gugu | #4 `2e8b3ce2` | 245 | **1:32.016** | p4 / dig | f224 / p2 / attack | 0.33 | at-net rally neighbor (densified) |
| 94 | jojo | #1 `de12d43e` | 272 | **0:20.519** | p4 / ATTACK | f225 / p4 / SET | 0.96 | s4 disagrees: pipeline=p4 vs s4=p3 |
| 95 | jojo | #1 `de12d43e` | 143 | **0:16.215** | p2 / serve | — | 0.49 | at-net rally neighbor (densified) |
| 96 | jojo | #1 `de12d43e` | 175 | **0:17.283** | p4 / receive | f143 / p2 / serve | 0.84 | at-net rally neighbor (densified) |
| 97 | jojo | #1 `de12d43e` | 225 | **0:18.951** | p4 / set | f175 / p4 / receive | 0.93 | at-net rally neighbor (densified) |
| 98 | juju | #12 `5f5292c2` | 205 | **5:04.633** | p1 / ATTACK | f152 / p1 / SET | 0.82 | s4 disagrees: pipeline=p1 vs s4=p2 |
| 99 | juju | #12 `5f5292c2` | 80 | **5:00.466** | p3 / serve | — | 0.53 | at-net rally neighbor (densified) |
| 100 | juju | #12 `5f5292c2` | 116 | **5:01.666** | p4 / receive | f80 / p3 / serve | 0.39 | at-net rally neighbor (densified) |

## Bucket D — Random / control (high-confidence)

| # | Video | Rally (#order, uuid) | Frame | Source time | Pipeline pick (pid / action) | Prev (frame / pid / action) | Conf | Notes |
|--:|:------|:--------------------|------:|:------------|:-----------------------------|:----------------------------|----:|:------|
| 101 | mame | #3 `52b0ca76` | 160 | **1:14.733** | p4 / receive | f128 / p1 / serve | 0.70 | high-conf control (conf=0.70) |
| 102 | mame | #3 `52b0ca76` | 128 | **1:13.666** | p1 / serve | — | 0.70 | high-conf control (conf=0.70) |
| 103 | mame | #3 `52b0ca76` | 254 | **1:17.866** | p2 / attack | f208 / p3 / set | 0.87 | high-conf control (conf=0.87) |
| 104 | vava | #2 `5f32d512` | 63 | **0:33.900** | p2 / serve | — | 0.70 | high-conf control (conf=0.70) |
| 105 | vava | #2 `5f32d512` | 94 | **0:34.933** | p2 / receive | f63 / p2 / serve | 0.90 | high-conf control (conf=0.90) |
| 106 | vava | #2 `5f32d512` | 143 | **0:36.566** | p4 / set | f94 / p2 / receive | 0.90 | high-conf control (conf=0.90) |
| 107 | wuwu | #1 `37e14e1e` | 157 | **0:17.650** | p3 / serve | — | 0.70 | high-conf control (conf=0.70) |
| 108 | wuwu | #1 `37e14e1e` | 396 | **0:21.637** | p2 / attack | f308 / p1 / set | 0.90 | high-conf control (conf=0.90) |
| 109 | wuwu | #1 `37e14e1e` | 308 | **0:20.169** | p1 / set | f211 / p2 / receive | 0.94 | high-conf control (conf=0.94) |
| 110 | wuwu | #1 `37e14e1e` | 211 | **0:18.551** | p2 / receive | f157 / p3 / serve | 0.85 | high-conf control (conf=0.85) |
| 111 | vava | #7 `84119722` | 177 | **3:19.300** | p4 / set | f132 / p3 / receive | 0.88 | high-conf control (conf=0.88) |
| 112 | vava | #7 `84119722` | 96 | **3:16.600** | p1 / serve | — | 0.70 | high-conf control (conf=0.70) |
| 113 | vava | #7 `84119722` | 132 | **3:17.800** | p3 / receive | f96 / p1 / serve | 0.82 | high-conf control (conf=0.82) |
| 114 | vava | #7 `84119722` | 228 | **3:21.000** | p3 / attack | f177 / p4 / set | 0.97 | high-conf control (conf=0.97) |
| 115 | meme | #11 `259887d1` | 165 | **3:47.433** | p1 / attack | f120 / p2 / receive | 0.95 | high-conf control (conf=0.95) |
| 116 | meme | #11 `259887d1` | 120 | **3:45.933** | p2 / receive | f81 / p4 / serve | 0.90 | high-conf control (conf=0.90) |
| 117 | meme | #11 `259887d1` | 81 | **3:44.633** | p4 / serve | — | 0.79 | high-conf control (conf=0.79) |
| 118 | lulu | #9 `bb90082c` | 626 | **4:07.296** | p2 / attack | f513 / p4 / attack | 0.88 | high-conf control (conf=0.88) |
| 119 | lulu | #9 `bb90082c` | 415 | **4:03.776** | p3 / set | f305 / p1 / receive | 0.93 | high-conf control (conf=0.93) |
| 120 | lulu | #9 `bb90082c` | 513 | **4:05.411** | p4 / attack | f415 / p3 / set | 0.89 | high-conf control (conf=0.89) |
| 121 | pipi | #2 `75295533` | 131 | **0:45.766** | p4 / set | f99 / p3 / receive | 0.78 | high-conf control (conf=0.78) |
| 122 | pipi | #2 `75295533` | 181 | **0:47.433** | p3 / attack | f131 / p4 / set | 0.79 | high-conf control (conf=0.79) |
| 123 | pipi | #2 `75295533` | 99 | **0:44.700** | p3 / receive | f54 / p2 / serve | 0.88 | high-conf control (conf=0.88) |
| 124 | pipi | #2 `75295533` | 54 | **0:43.200** | p2 / serve | — | 0.85 | high-conf control (conf=0.85) |
| 125 | moma | #3 `e1929103` | 192 | **1:31.800** | p1 / attack | f153 / p2 / set | 0.89 | high-conf control (conf=0.89) |
| 126 | moma | #3 `e1929103` | 104 | **1:28.866** | p1 / receive | f65 / p4 / serve | 0.74 | high-conf control (conf=0.74) |
| 127 | moma | #3 `e1929103` | 65 | **1:27.566** | p4 / serve | — | 0.70 | high-conf control (conf=0.70) |
| 128 | moma | #3 `e1929103` | 153 | **1:30.500** | p2 / set | f104 / p1 / receive | 0.91 | high-conf control (conf=0.91) |
| 129 | wewe | #1 `0b8c1d2b` | 170 | **0:05.855** | p3 / serve | — | 0.70 | high-conf control (conf=0.70) |
| 130 | wewe | #1 `0b8c1d2b` | 339 | **0:08.674** | p1 / set | f228 / p3 / receive | 0.89 | high-conf control (conf=0.89) |

## Bucket E.low — Confidence percentile (conf < 0.3)

| # | Video | Rally (#order, uuid) | Frame | Source time | Pipeline pick (pid / action) | Prev (frame / pid / action) | Conf | Notes |
|--:|:------|:--------------------|------:|:------------|:-----------------------------|:----------------------------|----:|:------|
| 131 | matahtach | #5 `349bbee9` | 111 | **2:17.271** | p3 / receive | f52 / p3 / receive | 0.15 | low-conf tier (conf=0.15) |
| 132 | yiyi | #8 `16a083a5` | 375 | **4:05.354** | p1 / attack | f263 / p2 / set | 0.28 | low-conf tier (conf=0.28) |
| 133 | matahtach | #5 `349bbee9` | 52 | **2:15.304** | p3 / receive | f45 / p1 / serve | 0.17 | low-conf tier (conf=0.17) |
| 134 | lele | #16 `62a4b02d` | 109 | **8:39.653** | p2 / serve | — | 0.20 | low-conf tier (conf=0.20) |
| 135 | mama | #3 `b7f92cdc` | 68 | **1:51.132** | p3 / serve | — | 0.11 | low-conf tier (conf=0.11) |

## Bucket E.mid — Confidence percentile (0.3 <= conf < 0.6)

| # | Video | Rally (#order, uuid) | Frame | Source time | Pipeline pick (pid / action) | Prev (frame / pid / action) | Conf | Notes |
|--:|:------|:--------------------|------:|:------------|:-----------------------------|:----------------------------|----:|:------|
| 136 | vava | #7 `84119722` | 239 | **3:21.366** | p2 / block | f228 / p3 / attack | 0.36 | mid-conf tier (conf=0.36) |
| 137 | moma | #3 `e1929103` | 231 | **1:33.100** | p1 / dig | f192 / p1 / attack | 0.32 | mid-conf tier (conf=0.32) |
| 138 | lulu | #9 `bb90082c` | 305 | **4:01.941** | p1 / receive | f244 / p-1 / serve | 0.31 | mid-conf tier (conf=0.31) |
| 139 | juju | #12 `5f5292c2` | 152 | **5:02.866** | p1 / set | f116 / p4 / receive | 0.53 | mid-conf tier (conf=0.53) |
| 140 | natch | #1 `e5e4c0b7` | 110 | **0:49.693** | p1 / receive | f85 / p4 / serve | 0.48 | mid-conf tier (conf=0.48) |

## Bucket E.high — Confidence percentile (conf >= 0.7)

| # | Video | Rally (#order, uuid) | Frame | Source time | Pipeline pick (pid / action) | Prev (frame / pid / action) | Conf | Notes |
|--:|:------|:--------------------|------:|:------------|:-----------------------------|:----------------------------|----:|:------|
| 141 | kiki | #8 `0fc55658` | 306 | **3:15.912** | p1 / set | f233 / p2 / receive | 0.89 | high-conf tier (conf=0.89) |
| 142 | wewe | #1 `0b8c1d2b` | 441 | **0:10.376** | p2 / attack | f339 / p1 / set | 0.97 | high-conf tier (conf=0.97) |
| 143 | lulu | #8 `55565c2b` | 172 | **3:40.102** | p3 / serve | — | 0.70 | high-conf tier (conf=0.70) |
| 144 | matchope | #6 `7cd50bd0` | 60 | **2:32.152** | p2 / receive | f32 / p1 / serve | 0.70 | high-conf tier (conf=0.70) |
| 145 | lolo | #11 `bb1c9802` | 140 | **4:43.266** | p3 / receive | f106 / p2 / serve | 0.81 | high-conf tier (conf=0.81) |
| 146 | ruru | #2 `57a588b2` | 469 | **0:32.016** | p2 / attack | f369 / p2 / set | 0.95 | high-conf tier (conf=0.95) |
| 147 | tete | #8 `adca6b54` | 167 | **4:19.766** | p4 / set | f120 / p2 / receive | 0.90 | high-conf tier (conf=0.90) |
| 148 | muchi | #5 `5668ebd9` | 118 | **2:29.884** | p2 / set | f68 / p1 / receive | 0.82 | high-conf tier (conf=0.82) |
| 149 | wiwi | #7 `eb4c4b8e` | 511 | **3:35.516** | p1 / attack | f406 / p2 / set | 0.88 | high-conf tier (conf=0.88) |
| 150 | rara | #9 `5b9e0ef2` | 167 | **4:46.966** | p1 / receive | f130 / p4 / serve | 0.70 | high-conf tier (conf=0.70) |

## After labeling

Tell Claude "done with T2 labeling" and I will:

1. Re-run A1.v2 + S4 + role-attribution probes against the new GT.
2. Compute per-bucket precision.
3. Measure contact-detector at-net FP rate.
4. Surface a precision-vs-confidence curve.
5. Decide which workstream(s) ship next based on cleaned data.
