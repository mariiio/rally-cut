# A1 Visual Verification Sample — 2026-05-13

User: scrub each contact frame in the source video. Mark each row ✅ (A1 picked the real toucher) / ❌ (A1 made it worse) / ⚠️ (ambiguous). Ship threshold: ≥ 8 / 10 ✅.

Conventions: `frame` is the frame index inside the rally clip (0-based, at the source-video fps). `Source-video time` = rally `start_ms` + `frame / fps * 1000`. Player IDs (`p1..p4`) are the post-`remap_track_ids` PIDs; teams are A/B as labeled in `actions_json.teamAssignments`.

| # | Video | Rally (short) | Source-video time | Frame | Prev action (frame, player, team) | Curr action | A1 before → after | alt_ratio | Verdict |
|--:|:------|:--------------|:-----------------:|:-----:|:-----------------------------------|:------------|:------------------|:---------:|:-------:|
| 1 | juju | d810943e | 00:46.900 | 345 | dig@252 p4(unknown) | set | curr p4 → p3 | 2.7x |  |
| 2 | gigi | 72c8229b | 00:17.500 | 399 | set@357 p3(B) | attack | prev p3 → p4 | 2.7x |  |
| 3 | caca | ae99ab2a | 00:42.483 | 353 | receive@241 p1(A) | set | curr p1 → p2 | 4.6x |  |
| 4 | machi | f406f4b3 | 01:09.060 | 333 | serve@242 p1(A) | attack | curr p1 → p2 | 4.5x |  |
| 5 | matchop | f433967e | 02:53.976 | 100 | receive@83 p4(A) | receive | prev p4 → p3 | 5.4x |  |
| 6 | lala | 2eeb3ae6 | 04:14.486 | 616 | dig@569 p1(B) | dig | curr p1 → p2 | 8.5x |  |
| 7 | mech | b0dabe43 | 02:17.670 | 59 | serve@39 p3(A) | receive | curr p3 → p4 | 19.7x |  |
| 8 | veve | 4c27b635 | 01:37.266 | 284 | attack@265 p3(B) | dig | curr p3 → p4 | 15.2x |  |
| 9 | natch | e5e4c0b7 | 00:53.160 | 214 | receive@110 p1(A) | attack | curr p1 → p2 | 2.0x |  |
| 10 | matttch | 8d3205ed | 03:40.048 | 296 | attack@220 p3(B) | attack | prev p3 → p4 | 2.4x |  |

---

## Per-fix details

### 1. juju / rally d810943e / pair_idx 4

- **video_id**: `06f0b063-b3f9-40aa-b13b-fba1edd74a85`
- **rally_id**: `d810943e-dd1c-4518-a6af-577b22555c3c`
- **filename**: `juju.mp4`
- **fps**: 30, rally start_ms: 35400
- **prev action**: `dig` frame=252 (t=00:43.800) player=p4(unknown) → **A1→ p4** conf=0.924 player_dist=0.0294
- **curr action**: `set` frame=345 (t=00:46.900) player=p4(unknown) → **A1→ p3** conf=0.655 player_dist=0.0122
- **curr_best_same_team_alt_ratio**: 2.70x
- **prev candidates (top 5)**: p2(A)@0.0282, p4(B)@0.0294, p3(B)@0.0416
- **curr candidates (top 5)**: p4(B)@0.0122, p3(B)@0.0330, p2(A)@0.0620

### 2. gigi / rally 72c8229b / pair_idx 8

- **video_id**: `b097dd2a-6953-4e0e-a603-5be3552f462e`
- **rally_id**: `72c8229b-2993-4310-9b61-cd6162cc27fa`
- **filename**: `gigi.mp4`
- **fps**: 30, rally start_ms: 4200
- **prev action**: `set` frame=357 (t=00:16.100) player=p3(B) → **A1→ p4** conf=0.679 player_dist=0.0291
- **curr action**: `attack` frame=399 (t=00:17.500) player=p3(B) → **A1→ p3** conf=0.921 player_dist=0.0454
- **curr_best_same_team_alt_ratio**: 2.66x
- **prev candidates (top 5)**: p3(B)@0.0291, p1(A)@0.0473, p2(A)@0.0496
- **curr candidates (top 5)**: p3(B)@0.0454, p1(A)@0.0720, p4(B)@0.1208

### 3. caca / rally ae99ab2a / pair_idx 2

- **video_id**: `627c1add-8a80-42ab-8278-3617880ebf81`
- **rally_id**: `ae99ab2a-e342-4096-9225-6cfbb3909d15`
- **filename**: `caca.mp4`
- **fps**: 60, rally start_ms: 36600
- **prev action**: `receive` frame=241 (t=00:40.616) player=p1(A) → **A1→ p1** conf=0.301 player_dist=0.0308
- **curr action**: `set` frame=353 (t=00:42.483) player=p1(A) → **A1→ p2** conf=0.537 player_dist=0.0332
- **curr_best_same_team_alt_ratio**: 4.58x
- **prev candidates (top 5)**: p1(A)@0.0308, p3(B)@0.0774, p2(A)@0.1878
- **curr candidates (top 5)**: p1(A)@0.0332, p3(B)@0.0644, p2(A)@0.1519

### 4. machi / rally f406f4b3 / pair_idx 1

- **video_id**: `23a5f798-78a0-4b3a-8647-b4a2166274b1`
- **rally_id**: `f406f4b3-95df-474a-aecc-a5ba7239ad9b`
- **filename**: `machi.mp4`
- **fps**: 25, rally start_ms: 55740
- **prev action**: `serve` frame=242 (t=01:05.420) player=p1(A) → **A1→ p1** conf=0.313 player_dist=0.1060
- **curr action**: `attack` frame=333 (t=01:09.060) player=p1(A) → **A1→ p2** conf=0.501 player_dist=0.0301
- **curr_best_same_team_alt_ratio**: 4.53x
- **prev candidates (top 5)**: p1(A)@0.1060, p2(A)@0.3075, p3(B)@0.3075
- **curr candidates (top 5)**: p1(A)@0.0301, p2(A)@0.1361, p4(B)@0.2362

### 5. matchop / rally f433967e / pair_idx 2

- **video_id**: `920ba69d-2526-4e6c-a357-c44af3bf5c99`
- **rally_id**: `f433967e-2c40-4169-b5cb-87f48cd0fa63`
- **filename**: `matchop.mp4`
- **fps**: 29.97, rally start_ms: 170640
- **prev action**: `receive` frame=83 (t=02:53.409) player=p4(A) → **A1→ p3** conf=0.576 player_dist=0.0448
- **curr action**: `receive` frame=100 (t=02:53.976) player=p4(A) → **A1→ p4** conf=0.334 player_dist=0.0162
- **curr_best_same_team_alt_ratio**: 5.42x
- **prev candidates (top 5)**: p4(A)@0.0448, p3(A)@0.1444, p2(B)@0.3561
- **curr candidates (top 5)**: p4(A)@0.0162, p3(A)@0.0876, p2(B)@0.2597

### 6. lala / rally 2eeb3ae6 / pair_idx 13

- **video_id**: `84e66e74-8d4f-420a-ad01-0ada95153ad0`
- **rally_id**: `2eeb3ae6-cf97-4eeb-9400-28a8060a7636`
- **filename**: `lala.mp4`
- **fps**: 29.75, rally start_ms: 233781
- **prev action**: `dig` frame=569 (t=04:12.907) player=p1(B) → **A1→ p1** conf=0.763 player_dist=0.0233
- **curr action**: `dig` frame=616 (t=04:14.486) player=p1(A) → **A1→ p2** conf=0.964 player_dist=0.0221
- **curr_best_same_team_alt_ratio**: 8.47x
- **prev candidates (top 5)**: p1(B)@0.0233, p3(A)@0.0626, p2(B)@0.1407
- **curr candidates (top 5)**: p3(A)@0.0123, p1(B)@0.0269, p2(B)@0.1872

### 7. mech / rally b0dabe43 / pair_idx 1

- **video_id**: `c6e4c876-beca-4cb8-9cce-4a4fc70553f1`
- **rally_id**: `b0dabe43-7ddb-4544-8d2c-e86032a8d8f5`
- **filename**: `mech.mp4`
- **fps**: 29.97, rally start_ms: 135702
- **prev action**: `serve` frame=39 (t=02:17.003) player=p3(A) → **A1→ p3** conf=0.882 player_dist=0.1779
- **curr action**: `receive` frame=59 (t=02:17.670) player=p3(A) → **A1→ p4** conf=0.853 player_dist=0.0066
- **curr_best_same_team_alt_ratio**: 19.74x
- **prev candidates (top 5)**: p3(A)@0.1779, p4(A)@0.3985, p1(B)@0.4606
- **curr candidates (top 5)**: p3(A)@0.0066, p4(A)@0.1312, p1(B)@0.1394

### 8. veve / rally 4c27b635 / pair_idx 6

- **video_id**: `43928971-2e07-4814-bb1a-3d91c7bf03b2`
- **rally_id**: `4c27b635-fbab-4bcb-a30e-f82a87c223c2`
- **filename**: `veve.mp4`
- **fps**: 30, rally start_ms: 87800
- **prev action**: `attack` frame=265 (t=01:36.633) player=p3(B) → **A1→ p3** conf=0.841 player_dist=0.0039
- **curr action**: `dig` frame=284 (t=01:37.266) player=p3(B) → **A1→ p4** conf=0.835 player_dist=0.0096
- **curr_best_same_team_alt_ratio**: 15.16x
- **prev candidates (top 5)**: p3(B)@0.0039, p2(A)@0.0626, p1(A)@0.2057
- **curr candidates (top 5)**: p3(B)@0.0096, p2(A)@0.0117, p4(B)@0.1453

### 9. natch / rally e5e4c0b7 / pair_idx 2

- **video_id**: `a7ee3d38-a3a9-4dcd-a2af-e0617997e708`
- **rally_id**: `e5e4c0b7-7f18-493f-b95b-574e51821452`
- **filename**: `natch.mp4`
- **fps**: 30, rally start_ms: 46027
- **prev action**: `receive` frame=110 (t=00:49.693) player=p1(A) → **A1→ p1** conf=0.476 player_dist=0.1508
- **curr action**: `attack` frame=214 (t=00:53.160) player=p1(A) → **A1→ p2** conf=0.775 player_dist=0.0759
- **curr_best_same_team_alt_ratio**: 2.02x
- **prev candidates (top 5)**: p4(B)@0.1508, p1(A)@0.1735, p3(B)@0.1950
- **curr candidates (top 5)**: p1(A)@0.0759, p2(A)@0.1533, p3(B)@0.1741

### 10. matttch / rally 8d3205ed / pair_idx 6

- **video_id**: `23b662ba-99e0-47d6-a9ac-90bb6fa9bdd1`
- **rally_id**: `8d3205ed-b0dc-4c0c-bc24-fda34554e45f`
- **filename**: `matttch.mp4`
- **fps**: 29.97, rally start_ms: 210172
- **prev action**: `attack` frame=220 (t=03:37.512) player=p3(B) → **A1→ p4** conf=0.388 player_dist=0.0250
- **curr action**: `attack` frame=296 (t=03:40.048) player=p3(B) → **A1→ p3** conf=0.853 player_dist=0.0777
- **curr_best_same_team_alt_ratio**: 2.41x
- **prev candidates (top 5)**: p3(B)@0.0250, p4(B)@0.1652, p1(A)@0.2713
- **curr candidates (top 5)**: p3(B)@0.0777, p4(B)@0.1874, p1(A)@0.3387

