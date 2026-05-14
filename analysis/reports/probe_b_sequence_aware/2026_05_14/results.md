# Probe B — Sequence-Aware Role Attribution (2026-05-14)

Test 3 hypotheses on whether pre-contact ball-trajectory signals
can disambiguate the within-team toucher better than ball-position-
at-contact (which today's pipeline + A1/A2 all use).

## Strategies

| S | Name | Description |
|---|------|-------------|
| S0 | baseline | pipeline pick (current production) |
| S1 | ball-pos | argmin dist(bbox_center, ball@contact) |
| S2 | traj-end | extrapolate ball traj over K=10; project to contact |
| S3 | traj-int | mean(dist(bbox_center, ball)) over [f-K, f-1] |
| S4 | anti-self | S3 excluding prev_toucher unless prev=BLOCK |

## Aggregate

- Cases run: **10**
- GT-evaluable: **10**

| Strategy | Correct (of evaluable) |
|----------|------------------------|
| S0 (baseline) | 1 / 10 |
| S1 (ball-pos) | 3 / 10 |
| S2 (traj-end) | 3 / 10 |
| S3 (traj-int) | 2 / 10 |
| S4 (anti-self) | 6 / 10 |

**Best trajectory strategy:** 6; lift over S1 = `3`

### Within-team subset (9 of 10)

Within-team subset = GT track is in the same-team candidate set,
so within-team disambiguation is in-scope. Cross-team errors (e.g. #3
F3-keke) are unreachable by this probe's same-team-only search.

| Strategy | Correct (within-team) |
|----------|------------------------|
| S0 | 1 / 9 |
| S1 | 3 / 9 |
| S2 | 3 / 9 |
| S3 | 2 / 9 |
| S4 | 6 / 9 |

### Verdict: **SHIP-A4-design (lift=3, threshold ≥ 3)**

## Per-case detail

| # | label | video/rally | f | type | GT | S0 | S1 | S2 | S3 | S4 | n_pre |
|---|-------|-------------|---|------|----|----|----|----|----|----|-------|
| 1 | cascade-f128 | titi/a0881d82 | 128 | SET | p1 | p2✗ | p1✓ | p1✓ | p2✗ | p1✓ | 10 |
| 2 | cascade-f225 | titi/a0881d82 | 225 | DIG | p1 | p2✗ | p2✗ | p2✗ | p2✗ | p1✓ | 10 |
| 3 | F3-keke | keke/0144acfb | 223 | ATTACK | p3 | p1✗ | p1✗ | p1✗ | p1✗ | p1✗ | 10 |
| 4 | F5-keke | keke/99091ec6 | 184 | ATTACK | p2 | p2✓ | p2✓ | p2✓ | p2✓ | p2✓ | 10 |
| 5 | stratified-jojo | jojo/36d3aa2c | 143 | RECEIVE | p3 | p4✗ | p4✗ | p4✗ | p4✗ | p4✗ | 10 |
| 6 | stratified-tutu-set | tutu/9064ba7b | 254 | SET | p2 | p1✗ | p2✓ | p2✓ | p1✗ | p2✓ | 10 |
| 7 | stratified-gugu-dig | gugu/62b6c286 | 230 | DIG | p4 | p3✗ | p3✗ | p3✗ | p4✓ | p4✓ | 10 |
| 8 | stratified-jaja | jaja/2ae99d01 | 146 | RECEIVE | p2 | p1✗ | p1✗ | p1✗ | p1✗ | p1✗ | 10 |
| 9 | stratified-natch | natch/e5e4c0b7 | 214 | ATTACK | p2 | p1✗ | p1✗ | p1✗ | p1✗ | p2✓ | 10 |
| 10 | stratified-lala-attack | lala/793625cd | 335 | ATTACK | p3 | p4✗ | p4✗ | p4✗ | p4✗ | p4✗ | 10 |

## Per-case detail (verbose)

### #1 cascade-f128 — titi/a0881d82 SET f=128

- note: cascade f128 — pipeline picked p2 (closest), GT=p1 (same-team B)
- GT resolved_track_id: `p1`
- pipeline (S0): `p2` correct=False
- S1 ball-pos: `p1` correct=True
- S2 traj-end: `p1` correct=True endpoint=(0.5788705675009677, 0.54620795140693)
- S3 traj-int: `p2` correct=False integrals={2: 0.15054975572358337, 1: 0.15768659113956424}
- S4 anti-self: `p1` correct=True prev_toucher=p2 prev_action=RECEIVE
- same_team_cands: [(2, 0.008662643324836987), (1, 0.019272894094226552)]
- n_pre_ball: 10
- visual: `visual_frames/01_titi_a0881d82_f128.jpg`

### #2 cascade-f225 — titi/a0881d82 DIG f=225

- note: cascade f225 — pipeline picked p2, GT=p1 (same-team B)
- GT resolved_track_id: `p1`
- pipeline (S0): `p2` correct=False
- S1 ball-pos: `p2` correct=False
- S2 traj-end: `p2` correct=False endpoint=(0.5916921909987698, 0.5292234507365476)
- S3 traj-int: `p2` correct=False integrals={2: 0.13084302995372457, 1: 0.14482660271722475}
- S4 anti-self: `p1` correct=True prev_toucher=p2 prev_action=ATTACK
- same_team_cands: [(2, 0.006088401102891127), (1, 0.04140246156133992)]
- n_pre_ball: 10
- visual: `visual_frames/02_titi_a0881d82_f225.jpg`

### #3 F3-keke — keke/0144acfb ATTACK f=223

- note: F3 occlusion case — likely no GT yet, will skip
- GT resolved_track_id: `p3`
- pipeline (S0): `p1` correct=False
- S1 ball-pos: `p1` correct=False
- S2 traj-end: `p1` correct=False endpoint=(0.23575365397920123, 0.433180171879918)
- S3 traj-int: `p1` correct=False integrals={1: 0.31409642929247045, 4: 0.3277167126128405}
- S4 anti-self: `p1` correct=False prev_toucher=p2 prev_action=SET
- same_team_cands: [(1, 0.004042790335685305), (4, 0.3268390814081336)]
- n_pre_ball: 10
- visual: `visual_frames/03_keke_0144acfb_f223.jpg`

### #4 F5-keke — keke/99091ec6 ATTACK f=184

- note: F5 mid-rally attack — pipeline currently matches GT (control)
- GT resolved_track_id: `p2`
- pipeline (S0): `p2` correct=True
- S1 ball-pos: `p2` correct=True
- S2 traj-end: `p2` correct=True endpoint=(0.5122163208600682, 0.4183703918723152)
- S3 traj-int: `p2` correct=True integrals={2: 0.30644428753478925, 1: 0.34470495946528307}
- S4 anti-self: `p2` correct=True prev_toucher=p3 prev_action=RECEIVE
- same_team_cands: [(2, 0.08794585350970062), (1, 0.1891639965234685)]
- n_pre_ball: 10
- visual: `visual_frames/04_keke_99091ec6_f184.jpg`

### #5 stratified-jojo — jojo/36d3aa2c RECEIVE f=143

- note: same-team error: pl=4 closest, gt=3
- GT resolved_track_id: `p3`
- pipeline (S0): `p4` correct=False
- S1 ball-pos: `p4` correct=False
- S2 traj-end: `p4` correct=False endpoint=(0.4181421662263797, 0.5638140160857594)
- S3 traj-int: `p4` correct=False integrals={4: 0.13450867043788445, 3: 0.19663300127209427}
- S4 anti-self: `p4` correct=False prev_toucher=p1 prev_action=SERVE
- same_team_cands: [(4, 0.01299605078234043), (3, 0.06781830420067496)]
- n_pre_ball: 10
- visual: `visual_frames/05_jojo_36d3aa2c_f143.jpg`

### #6 stratified-tutu-set — tutu/9064ba7b SET f=254

- note: same-team error: pl=1 closest, gt=2 (tight gap 0.009 vs 0.025)
- GT resolved_track_id: `p2`
- pipeline (S0): `p1` correct=False
- S1 ball-pos: `p2` correct=True
- S2 traj-end: `p2` correct=True endpoint=(0.5876082859394238, 0.5399785777454218)
- S3 traj-int: `p1` correct=False integrals={1: 0.1706253386697099, 2: 0.17616936991665771}
- S4 anti-self: `p2` correct=True prev_toucher=p1 prev_action=RECEIVE
- same_team_cands: [(1, 0.00949438292940989), (2, 0.024977958337911055)]
- n_pre_ball: 10
- visual: `visual_frames/06_tutu_9064ba7b_f254.jpg`

### #7 stratified-gugu-dig — gugu/62b6c286 DIG f=230

- note: same-team error: pl=3 closest, gt=4 (tight gap)
- GT resolved_track_id: `p4`
- pipeline (S0): `p3` correct=False
- S1 ball-pos: `p3` correct=False
- S2 traj-end: `p3` correct=False endpoint=(0.4510373200228731, 0.45600585731420384)
- S3 traj-int: `p4` correct=True integrals={3: 0.14629805391193315, 4: 0.128470793893625}
- S4 anti-self: `p4` correct=True prev_toucher=p1 prev_action=ATTACK
- same_team_cands: [(3, 0.009871887482971162), (4, 0.020062270009728343)]
- n_pre_ball: 10
- visual: `visual_frames/07_gugu_62b6c286_f230.jpg`

### #8 stratified-jaja — jaja/2ae99d01 RECEIVE f=146

- note: same-team error: pl=1, gt=2
- GT resolved_track_id: `p2`
- pipeline (S0): `p1` correct=False
- S1 ball-pos: `p1` correct=False
- S2 traj-end: `p1` correct=False endpoint=(0.330757202593831, 0.5032669899024527)
- S3 traj-int: `p1` correct=False integrals={1: 0.22841314045009026, 2: 0.3341520413412857}
- S4 anti-self: `p1` correct=False prev_toucher=p3 prev_action=SERVE
- same_team_cands: [(1, 0.025435093314759835), (2, 0.23387713732832535)]
- n_pre_ball: 10
- visual: `visual_frames/08_jaja_2ae99d01_f146.jpg`

### #9 stratified-natch — natch/e5e4c0b7 ATTACK f=214

- note: same-team error: pl=1 closest, gt=2
- GT resolved_track_id: `p2`
- pipeline (S0): `p1` correct=False
- S1 ball-pos: `p1` correct=False
- S2 traj-end: `p1` correct=False endpoint=(0.47059404027591295, 0.29389572159642796)
- S3 traj-int: `p1` correct=False integrals={1: 0.3618554822233596, 2: 0.38603362183447626}
- S4 anti-self: `p2` correct=True prev_toucher=p1 prev_action=RECEIVE
- same_team_cands: [(1, 0.0759147685986838), (2, 0.15331215314440697)]
- n_pre_ball: 10
- visual: `visual_frames/09_natch_e5e4c0b7_f214.jpg`

### #10 stratified-lala-attack — lala/793625cd ATTACK f=335

- note: same-team error: pl=4 closest, gt=3 (tight: 0.091 vs 0.092)
- GT resolved_track_id: `p3`
- pipeline (S0): `p4` correct=False
- S1 ball-pos: `p4` correct=False
- S2 traj-end: `p4` correct=False endpoint=(0.40110124008384873, 0.3185382153654919)
- S3 traj-int: `p4` correct=False integrals={4: 0.2906851367174693, 3: 0.33355489610425076}
- S4 anti-self: `p4` correct=False prev_toucher=p3 prev_action=SET
- same_team_cands: [(4, 0.09116244671556112), (3, 0.2525700754209652)]
- n_pre_ball: 10
- visual: `visual_frames/10_lala_793625cd_f335.jpg`
