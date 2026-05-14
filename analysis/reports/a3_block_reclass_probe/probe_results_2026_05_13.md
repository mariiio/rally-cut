# A3 BLOCK Reclassification Probe — 2026-05-13

Spec: `docs/superpowers/specs/2026-05-13-action-attribution-root-causes-design.md`

## Fleet scan

- ATTACK actions matching `(a) ∧ (c) ∧ (d_loose)` (cross-team prev, any type): **48**
- ATTACK actions matching `(a) ∧ (c) ∧ (d_strict)` (prev ∈ {attack, set}): **43**

> Note on (d): the design spec says `prev.action ∈ {attack, set}` (strict). The canonical F5 case has `prev=receive`, which fails strict-(d). We report both variants. F5 is included regardless.

## Candidates picked (10)

| # | F5 | video | rally | frame | src_time | suspect | prev | dc° | conf | d_strict | d_loose | all_strict | all_loose |
|---|----|-------|-------|-------|----------|---------|------|-----|------|----------|---------|------------|-----------|
| 1 | ★ | keke | 99091ec6 | 184 | 1:28.33 | p2(A) | f150 receive(B) | 13 | 0.69 | False | True | False | True |
| 2 |   | titi | 4ad457f6 | 359 | 15:18.21 | p2(A) | f312 set(B) | 90 | 0.81 | True | True | True | True |
| 3 |   | toto | fcc5dcba | 174 | 1:12.75 | p1(A) | f127 set(B) | 14 | 0.68 | True | True | False | False |
| 4 |   | lili | 0b7a2461 | 210 | 11:35.53 | p3(B) | f171 set(A) | 4 | 0.70 | True | True | False | False |
| 5 |   | lolo | 6935b412 | 249 | 13:52.10 | p3(A) | f201 set(B) | 1 | 0.58 | True | True | False | False |
| 6 |   | gigi | 3e07342a | 126 | 3:15.70 | p1(A) | f97 receive(B) | 0 | 0.63 | False | True | False | False |
| 7 |   | cece | 5c35e049 | 242 | 0:50.27 | p3(B) | f209 attack(A) | 11 | 0.85 | True | True | False | False |
| 8 |   | wawa | 7094136a | 306 | 0:15.73 | p4(B) | f211 set(A) | 26 | 0.82 | True | True | False | False |
| 9 |   | lala | 2eeb3ae6 | 966 | 4:26.25 | p1(B) | f912 set(A) | 27 | 0.36 | True | True | True | True |
| 10 |   | dark | cf4cdd43 | 172 | 3:56.00 | p1(B) | f122 set(A) | 14 | 0.40 | True | True | False | False |

## Per-case detail

### #1 keke/99091ec6 f184 (F5)

- suspect: p2(A) conf=0.694
- prev: frame 150 `receive`(B) p3
- direction_change_deg: `13.2` (≤ 90: `True`)
- player_court_xy: `(4.467877726027082, 5.339587978022004)` (a=True, source=`court`)
- net_y_image: `0.7283611878563798`
- wrist_y_image: `0.507104218006134` (which=`right`)
- (b) wrist-above-net: `True` (`ok`)
- (d) strict: `False` (`prev-type-receive`)
- (d) loose:  `True` (`ok`)
- all_pass_strict: `False`
- all_pass_loose:  `True`
- source-video time: **1:28.33** (source_frame=2650, fps=30.000, rally_start_ms=82200)

### #2 titi/4ad457f6 f359 

- suspect: p2(A) conf=0.812
- prev: frame 312 `set`(B) p4
- direction_change_deg: `89.5` (≤ 90: `True`)
- player_court_xy: `(0.40404141360650253, 9.789684165279489)` (a=True, source=`court`)
- net_y_image: `0.7010616674202232`
- wrist_y_image: `0.40456822514533997` (which=`left`)
- (b) wrist-above-net: `True` (`ok`)
- (d) strict: `True` (`ok`)
- (d) loose:  `True` (`ok`)
- all_pass_strict: `True`
- all_pass_loose:  `True`
- source-video time: **15:18.21** (source_frame=27546, fps=30.000, rally_start_ms=906242)

### #3 toto/fcc5dcba f174 

- suspect: p1(A) conf=0.678
- prev: frame 127 `set`(B) p4
- direction_change_deg: `14.3` (≤ 90: `True`)
- player_court_xy: `(2.8826418380859447, 2.6873774374369366)` (a=False, source=`court`)
- net_y_image: `0.6500983551389076`
- wrist_y_image: `0.575971245765686` (which=`left`)
- (b) wrist-above-net: `True` (`ok`)
- (d) strict: `True` (`ok`)
- (d) loose:  `True` (`ok`)
- all_pass_strict: `False`
- all_pass_loose:  `False`
- source-video time: **1:12.75** (source_frame=2176, fps=29.912, rally_start_ms=66927)

### #4 lili/0b7a2461 f210 

- suspect: p3(B) conf=0.703
- prev: frame 171 `set`(A) p4
- direction_change_deg: `3.7` (≤ 90: `True`)
- player_court_xy: `(-1.0315694685269214, 13.585721634750861)` (a=False, source=`court`)
- net_y_image: `0.7000879346670833`
- wrist_y_image: `0.46764233708381653` (which=`left`)
- (b) wrist-above-net: `True` (`ok`)
- (d) strict: `True` (`ok`)
- (d) loose:  `True` (`ok`)
- all_pass_strict: `False`
- all_pass_loose:  `False`
- source-video time: **11:35.53** (source_frame=20845, fps=29.970, rally_start_ms=688521)

### #5 lolo/6935b412 f249 

- suspect: p3(A) conf=0.576
- prev: frame 201 `set`(B) p1
- direction_change_deg: `0.9` (≤ 90: `True`)
- player_court_xy: `(2.0338526079456125, 4.2899409789749585)` (a=False, source=`court`)
- net_y_image: `0.6935949645849844`
- wrist_y_image: `0.5994341373443604` (which=`right`)
- (b) wrist-above-net: `True` (`ok`)
- (d) strict: `True` (`ok`)
- (d) loose:  `True` (`ok`)
- all_pass_strict: `False`
- all_pass_loose:  `False`
- source-video time: **13:52.10** (source_frame=24963, fps=30.000, rally_start_ms=823800)

### #6 gigi/3e07342a f126 

- suspect: p1(A) conf=0.628
- prev: frame 97 `receive`(B) p3
- direction_change_deg: `0.1` (≤ 90: `True`)
- player_court_xy: `(2.1048595983146985, 12.15418817844729)` (a=False, source=`court`)
- net_y_image: `0.5995549134969612`
- wrist_y_image: `None` (which=`None`)
- (b) wrist-above-net: `None` (`no-wrist-detected`)
- (d) strict: `False` (`prev-type-receive`)
- (d) loose:  `True` (`ok`)
- all_pass_strict: `False`
- all_pass_loose:  `False`
- source-video time: **3:15.70** (source_frame=5871, fps=30.000, rally_start_ms=191500)

### #7 cece/5c35e049 f242 

- suspect: p3(B) conf=0.846
- prev: frame 209 `attack`(A) p2
- direction_change_deg: `10.9` (≤ 90: `True`)
- player_court_xy: `(3.749267705343599, 12.81741072141872)` (a=False, source=`court`)
- net_y_image: `0.5737377606610319`
- wrist_y_image: `None` (which=`None`)
- (b) wrist-above-net: `None` (`no-wrist-detected`)
- (d) strict: `True` (`ok`)
- (d) loose:  `True` (`ok`)
- all_pass_strict: `False`
- all_pass_loose:  `False`
- source-video time: **0:50.27** (source_frame=1508, fps=30.000, rally_start_ms=42200)

### #8 wawa/7094136a f306 

- suspect: p4(B) conf=0.818
- prev: frame 211 `set`(A) p3
- direction_change_deg: `25.7` (≤ 90: `True`)
- player_court_xy: `(-0.615461073379773, 26.026844178486073)` (a=False, source=`court`)
- net_y_image: `0.5901362979650229`
- wrist_y_image: `None` (which=`None`)
- (b) wrist-above-net: `None` (`no-wrist-detected`)
- (d) strict: `True` (`ok`)
- (d) loose:  `True` (`ok`)
- all_pass_strict: `False`
- all_pass_loose:  `False`
- source-video time: **0:15.73** (source_frame=943, fps=59.940, rally_start_ms=10627)

### #9 lala/2eeb3ae6 f966 

- suspect: p1(B) conf=0.358
- prev: frame 912 `set`(A) p4
- direction_change_deg: `27.3` (≤ 90: `True`)
- player_court_xy: `(4.942891840752952, 10.4416406358063)` (a=True, source=`court`)
- net_y_image: `0.6411795436120595`
- wrist_y_image: `0.5226821303367615` (which=`left`)
- (b) wrist-above-net: `True` (`ok`)
- (d) strict: `True` (`ok`)
- (d) loose:  `True` (`ok`)
- all_pass_strict: `True`
- all_pass_loose:  `True`
- source-video time: **4:26.25** (source_frame=7921, fps=29.750, rally_start_ms=233781)

### #10 dark/cf4cdd43 f172 

- suspect: p1(B) conf=0.399
- prev: frame 122 `set`(A) p3
- direction_change_deg: `14.1` (≤ 90: `True`)
- player_court_xy: `(3.7363423757142775, 13.122149881761986)` (a=False, source=`court`)
- net_y_image: `0.6259139430025088`
- wrist_y_image: `0.3443840444087982` (which=`right`)
- (b) wrist-above-net: `True` (`ok`)
- (d) strict: `True` (`ok`)
- (d) loose:  `True` (`ok`)
- all_pass_strict: `False`
- all_pass_loose:  `False`
- source-video time: **3:56.00** (source_frame=7080, fps=30.000, rally_start_ms=230281)

## Claude vision first-pass verdicts (Phase 4)

- block: **5**
- attack: **4**
- ambiguous: **1**

**Threshold ≥ 7 blocks → NO-SHIP A3**

### Per-case verdicts

| # | video / rally | frame | verdict | note |
|---|---------------|-------|---------|------|
| 1 | keke/99091ec6 | 184 | 🟦 block | F5 canonical. Suspect p2(A) on right side jumping with arms above net; team B receivers on other side. Classic block pose — arms up high above net plane, body parallel to net. Ball directly above hands. |
| 2 | titi/4ad457f6 | 359 | 🟦 block | p2(A) at the net with arms raised in block formation; p3(B) immediately adjacent. Classic net duel — both arms above net level, ball right at the net. dc=89.5° is borderline but visually clearly a block. |
| 3 | toto/fcc5dcba | 174 | 🟧 attack | p1(A) in background-left near net but arms at hip level, NO jump, ball at chest/face level not above net. wrist clearly below net. Not a block — looks like an attack or receive. |
| 4 | lili/0b7a2461 | 210 | 🟦 block | p3(B) on left jumping high at the net with arms clearly above the net plane. Ball right at her hand. dc=3.7° = pure deflection. Court_y=13.6 mis-projects feet because she's mid-jump. Visually unambiguous block. |
| 5 | lolo/6935b412 | 249 | 🟧 attack | p3(A) standing at net with arms DOWN at sides, no jump, looking sideways. Ball just passing near him — wrist below net. Not a block pose at all. dc=0.9° suggests this could be a non-contact or soft tip. |
| 6 | gigi/3e07342a | 126 | 🟧 attack | p1(A) bent over with arms forward/down — receive/dig pose. NO arms above net. Ball at head height but player not jumping. Probably mis-typed as ATTACK but is more likely receive/dig — in any case NOT a block. |
| 7 | cece/5c35e049 | 242 | ⚠️ ambig | p3(B) is a very small/distant figure; wrist not detected by pose model. Ball at his head level but pose is indistinguishable at this scale. Can't confirm block-vs-attack. |
| 8 | wawa/7094136a | 306 | 🟦 block | p4(B) at the top of the net with BOTH ARMS RAISED above net plane. Classic block formation. Ball at his hands. Court_y=26 is a calibration artifact (high-fps proxy mismatch). Visually unmistakable block. |
| 9 | lala/2eeb3ae6 | 966 | 🟧 attack | p1(B) at net but arm extended LATERALLY to ball at shoulder/face level. Wrist below net. No jump visible. This is NOT a block — looks like a soft tip / set-over or low attack hit. |
| 10 | dark/cf4cdd43 | 172 | 🟦 block | p1(B) at the net jumping with arms raised; ball at top of his head/hands. Adjacent team B player (p2) also jumping. Classic net duel where the previous set(A) was met by a B-team blocker. dc=14.1° = clean deflection. |