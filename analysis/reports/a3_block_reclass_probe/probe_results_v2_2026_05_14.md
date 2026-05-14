# A3 BLOCK Reclassification Probe v2 — 2026-05-14

Spec: `docs/superpowers/specs/2026-05-13-action-attribution-root-causes-design.md`

## v2 refinements

- (a)′ HEAD-near-net in image coords (replaces v1 feet-court projection).
- (b)′ soft pose-fallback — Unknown counts as soft-yes when (a)+(c)+(d) all confirm.
- (d) loose: prev cross-team AND prev NOT a serve.

## Fleet scan (v2)

- v2 triage (a′ ∧ c ∧ d_loose-v2): **60**
- v2 strict (a′ ∧ c ∧ d_strict): **52**

> Note on (d): the design spec says `prev.action ∈ {attack, set}` (strict). The canonical F5 case has `prev=receive`, which fails strict-(d). We report both variants. F5 is included as a loose-only pick.

## Candidates picked (10)

| # | F5 | variant | conf-tier | video | rally | frame | src_time | suspect | prev | dc° | conf | sel_strict | sel_loose |
|---|----|---------|-----------|-------|-------|-------|----------|---------|------|-----|------|------------|-----------|
| 1 | ★ | loose-only | strong | keke | 99091ec6 | 184 | 1:28.33 | p2(A) | f150 receive(B) | 13 | 0.69 | False | True |
| 2 |   | strict | strong | caco | cfc464a7 | 151 | 1:00.73 | p1(A) | f121 set(B) | 36 | 0.67 | True | True |
| 3 |   | strict | moderate | cece | 5c35e049 | 242 | 0:50.27 | p3(B) | f209 attack(A) | 11 | 0.85 | True | True |
| 4 |   | strict | strong | cuco | f127f3d5 | 205 | 1:16.23 | p1(A) | f162 set(B) | 51 | 0.88 | True | True |
| 5 |   | strict | strong | cucu | 7d5fbfb2 | 218 | 0:55.87 | p1(A) | f179 set(B) | 7 | 0.60 | True | True |
| 6 |   | strict | strong | dark | ca026ea3 | 171 | 1:54.60 | p1(A) | f118 set(B) | 82 | 0.90 | True | True |
| 7 |   | loose-only | moderate | gigi | 3e07342a | 126 | 3:15.70 | p1(A) | f97 receive(B) | 0 | 0.63 | False | True |
| 8 |   | loose-only | strong | kiki | fb6e23ff | 796 | 3:55.72 | p1(A) | f720 dig(B) | 19 | 0.60 | False | True |
| 9 |   | loose-only | moderate | lele | 8c00e2cc | 114 | 9:27.45 | p2(B) | f73 receive(A) | 1 | 0.69 | False | True |
| 10 |   | loose-only | strong | lili | 05f7dae1 | 238 | 2:18.71 | p3(A) | f170 dig(B) | 13 | 0.57 | False | True |

## Per-case detail

### #1 keke/99091ec6 f184 (F5)

- variant: **loose-only** confidence: **strong**
- suspect: p2(A) conf=0.694
- prev: frame 150 `receive`(B) p3
- direction_change_deg: `13.2` (≤ 90: `True`)
- player_court_xy (feet-projection, diagnostic): `(4.467877726027082, 5.339587978022004)`
- head_y_image: `0.4775607277690793`
- net_y_image: `0.7283611878563798`
- |head - net| = `0.2508` (band: 0.050; pass: `False`)
- (a)′ pass: `True` source=`head-near-ball-at-net`
- wrist_y_image: `0.507104218006134` (which=`right`)
- (b)′ wrist-above-net: `True` (`ok`)
- (d) strict: `False` (`prev-type-receive`)
- (d) loose:  `True` (`ok`)
- selected_strict: `False`  selected_loose: `True`
- source-video time: **1:28.33** (source_frame=2650, fps=30.000, rally_start_ms=82200)

### #2 caco/cfc464a7 f151 

- variant: **strict** confidence: **strong**
- suspect: p1(A) conf=0.671
- prev: frame 121 `set`(B) p4
- direction_change_deg: `35.6` (≤ 90: `True`)
- player_court_xy (feet-projection, diagnostic): `(1.5659580672517923, 5.223524386137371)`
- head_y_image: `0.4203216392039197`
- net_y_image: `0.6703223147476646`
- |head - net| = `0.2500` (band: 0.050; pass: `False`)
- (a)′ pass: `True` source=`head-near-ball-at-net`
- wrist_y_image: `0.4342047870159149` (which=`right`)
- (b)′ wrist-above-net: `True` (`ok`)
- (d) strict: `True` (`ok`)
- (d) loose:  `True` (`ok`)
- selected_strict: `True`  selected_loose: `True`
- source-video time: **1:00.73** (source_frame=1822, fps=30.000, rally_start_ms=55700)

### #3 cece/5c35e049 f242 

- variant: **strict** confidence: **moderate**
- suspect: p3(B) conf=0.846
- prev: frame 209 `attack`(A) p2
- direction_change_deg: `10.9` (≤ 90: `True`)
- player_court_xy (feet-projection, diagnostic): `(3.749267705343599, 12.81741072141872)`
- head_y_image: `0.4266969973393587`
- net_y_image: `0.5737377606610319`
- |head - net| = `0.1470` (band: 0.050; pass: `False`)
- (a)′ pass: `True` source=`head-near-ball-at-net`
- wrist_y_image: `None` (which=`None`)
- (b)′ wrist-above-net: `None` (`no-wrist-detected`)
- (d) strict: `True` (`ok`)
- (d) loose:  `True` (`ok`)
- selected_strict: `True`  selected_loose: `True`
- source-video time: **0:50.27** (source_frame=1508, fps=30.000, rally_start_ms=42200)

### #4 cuco/f127f3d5 f205 

- variant: **strict** confidence: **strong**
- suspect: p1(A) conf=0.884
- prev: frame 162 `set`(B) p2
- direction_change_deg: `50.5` (≤ 90: `True`)
- player_court_xy (feet-projection, diagnostic): `(7.192719904333646, 9.692518722234304)`
- head_y_image: `0.2440843903479729`
- net_y_image: `0.5535156361277368`
- |head - net| = `0.3094` (band: 0.050; pass: `False`)
- (a)′ pass: `True` source=`head-near-ball-at-net`
- wrist_y_image: `0.2724480628967285` (which=`right`)
- (b)′ wrist-above-net: `True` (`ok`)
- (d) strict: `True` (`ok`)
- (d) loose:  `True` (`ok`)
- selected_strict: `True`  selected_loose: `True`
- source-video time: **1:16.23** (source_frame=2287, fps=30.000, rally_start_ms=69400)

### #5 cucu/7d5fbfb2 f218 

- variant: **strict** confidence: **strong**
- suspect: p1(A) conf=0.601
- prev: frame 179 `set`(B) p3
- direction_change_deg: `7.3` (≤ 90: `True`)
- player_court_xy (feet-projection, diagnostic): `(1.9268427653496736, 6.550145476909062)`
- head_y_image: `0.4064328880272555`
- net_y_image: `0.6157764112757514`
- |head - net| = `0.2093` (band: 0.050; pass: `False`)
- (a)′ pass: `True` source=`head-near-ball-at-net`
- wrist_y_image: `0.4097815752029419` (which=`right`)
- (b)′ wrist-above-net: `True` (`ok`)
- (d) strict: `True` (`ok`)
- (d) loose:  `True` (`ok`)
- selected_strict: `True`  selected_loose: `True`
- source-video time: **0:55.87** (source_frame=1676, fps=30.000, rally_start_ms=48600)

### #6 dark/ca026ea3 f171 

- variant: **strict** confidence: **strong**
- suspect: p1(A) conf=0.898
- prev: frame 118 `set`(B) p3
- direction_change_deg: `81.7` (≤ 90: `True`)
- player_court_xy (feet-projection, diagnostic): `(2.601936274889072, 10.451898006703267)`
- head_y_image: `0.3222132623195649`
- net_y_image: `0.6259139430025088`
- |head - net| = `0.3037` (band: 0.050; pass: `False`)
- (a)′ pass: `True` source=`head-near-ball-at-net`
- wrist_y_image: `0.3453868329524994` (which=`right`)
- (b)′ wrist-above-net: `True` (`ok`)
- (d) strict: `True` (`ok`)
- (d) loose:  `True` (`ok`)
- selected_strict: `True`  selected_loose: `True`
- source-video time: **1:54.60** (source_frame=3438, fps=30.000, rally_start_ms=108888)

### #7 gigi/3e07342a f126 

- variant: **loose-only** confidence: **moderate**
- suspect: p1(A) conf=0.628
- prev: frame 97 `receive`(B) p3
- direction_change_deg: `0.1` (≤ 90: `True`)
- player_court_xy (feet-projection, diagnostic): `(2.1048595983146985, 12.15418817844729)`
- head_y_image: `0.44406162202358246`
- net_y_image: `0.5995549134969612`
- |head - net| = `0.1555` (band: 0.050; pass: `False`)
- (a)′ pass: `True` source=`head-near-ball-at-net`
- wrist_y_image: `None` (which=`None`)
- (b)′ wrist-above-net: `None` (`no-wrist-detected`)
- (d) strict: `False` (`prev-type-receive`)
- (d) loose:  `True` (`ok`)
- selected_strict: `False`  selected_loose: `True`
- source-video time: **3:15.70** (source_frame=5871, fps=30.000, rally_start_ms=191500)

### #8 kiki/fb6e23ff f796 

- variant: **loose-only** confidence: **strong**
- suspect: p1(A) conf=0.600
- prev: frame 720 `dig`(B) p4
- direction_change_deg: `19.4` (≤ 90: `True`)
- player_court_xy (feet-projection, diagnostic): `(3.543732687523094, 7.728159350351707)`
- head_y_image: `0.4324673022604143`
- net_y_image: `0.7158696001886077`
- |head - net| = `0.2834` (band: 0.050; pass: `False`)
- (a)′ pass: `True` source=`head-near-ball-at-net`
- wrist_y_image: `0.4468136131763458` (which=`right`)
- (b)′ wrist-above-net: `True` (`ok`)
- (d) strict: `False` (`prev-type-dig`)
- (d) loose:  `True` (`ok`)
- selected_strict: `False`  selected_loose: `True`
- source-video time: **3:55.72** (source_frame=14129, fps=59.940, rally_start_ms=222438)

### #9 lele/8c00e2cc f114 

- variant: **loose-only** confidence: **moderate**
- suspect: p2(B) conf=0.687
- prev: frame 73 `receive`(A) p1
- direction_change_deg: `1.2` (≤ 90: `True`)
- player_court_xy (feet-projection, diagnostic): `(3.0101798708804233, 11.995441126752098)`
- head_y_image: `0.4987778449399354`
- net_y_image: `0.6783241446299115`
- |head - net| = `0.1795` (band: 0.050; pass: `False`)
- (a)′ pass: `True` source=`head-near-ball-at-net`
- wrist_y_image: `None` (which=`None`)
- (b)′ wrist-above-net: `None` (`no-wrist-detected`)
- (d) strict: `False` (`prev-type-receive`)
- (d) loose:  `True` (`ok`)
- selected_strict: `False`  selected_loose: `True`
- source-video time: **9:27.45** (source_frame=16703, fps=29.435, rally_start_ms=563595)

### #10 lili/05f7dae1 f238 

- variant: **loose-only** confidence: **strong**
- suspect: p3(A) conf=0.566
- prev: frame 170 `dig`(B) p1
- direction_change_deg: `13.4` (≤ 90: `True`)
- player_court_xy (feet-projection, diagnostic): `(5.418505165225556, 5.705629697365002)`
- head_y_image: `0.4554875719646899`
- net_y_image: `0.7000879346670833`
- |head - net| = `0.2446` (band: 0.050; pass: `False`)
- (a)′ pass: `True` source=`head-near-ball-at-net`
- wrist_y_image: `0.47565025091171265` (which=`right`)
- (b)′ wrist-above-net: `True` (`ok`)
- (d) strict: `False` (`prev-type-dig`)
- (d) loose:  `True` (`ok`)
- selected_strict: `False`  selected_loose: `True`
- source-video time: **2:18.71** (source_frame=4157, fps=29.970, rally_start_ms=130763)

## Claude vision first-pass verdicts (Phase 4)

- block: **7**
- attack: **3**
- ambiguous: **0**

**Threshold ≥ 7 blocks → SHIP A3**

### Per-variant breakdown

| variant | n | 🟦 block | 🟧 attack | ⚠️ ambig |
|---------|---|---------|----------|---------|
| strict | 5 | 4 | 1 | 0 |
| loose-only | 5 | 3 | 2 | 0 |

### Per-confidence breakdown

| confidence | n | 🟦 block | 🟧 attack | ⚠️ ambig |
|------------|---|---------|----------|---------|
| strong | 7 | 7 | 0 | 0 |
| moderate | 3 | 0 | 3 | 0 |

### Per-case verdicts

| # | video / rally | frame | variant | conf-tier | verdict | note |
|---|---------------|-------|---------|-----------|---------|------|
| 1 | keke/99091ec6 | 184 | loose-only | strong | 🟦 block | F5 canonical (loose-only, strong). Suspect p2(A) at the net with arms reaching up; ball directly above his hands. Team B player (p1 blue) adjacent. Classic net duel — wrist at net height. (a)′ pass head-near-ball; pose detected wrist above net. |
| 2 | caco/cfc464a7 | 151 | strict | strong | 🟦 block | STRICT, strong. p1(A) jumping high at the left side of the net with arms above the net plane; ball at wrist level. Team B player ducking right at the net opposite him. Textbook block. |
| 3 | cece/5c35e049 | 242 | strict | moderate | 🟧 attack | STRICT, moderate (b unknown). p3(B) is a small distant figure on back court holding ball at face level. NOT at net — the net is far in the background. dc=11° but no jump, no above-net arms. False positive — not a block. |
| 4 | cuco/f127f3d5 | 205 | strict | strong | 🟦 block | STRICT, strong. p1(A) at right side of net jumping with arms raised above the net plane; ball at hand at top of net. Team B red player adjacent at net opposite. Classic block, ball deflected (dc=50.5°). |
| 5 | cucu/7d5fbfb2 | 218 | strict | strong | 🟦 block | STRICT, strong. p2(A) jumping at the net with both arms raised high above net; ball directly above hands. Teammate p1(A) jumping with him (double block). dc=7° tight deflection. |
| 6 | dark/ca026ea3 | 171 | strict | strong | 🟦 block | STRICT, strong. p1(A) jumping HIGH at the net with arms straight up, ball at hands at top of jump. Classic block pose. dc=82° is near the 90° boundary but consistent with a hard-hit block where the ball reverses sharply at the net. prev=set(B) means team B's setter set their teammate; team A's blocker intercepted at net. |
| 7 | gigi/3e07342a | 126 | loose-only | moderate | 🟧 attack | LOOSE-only, moderate (b unknown). p1(A) is small figure at far back-left near net, bent over with arms forward — receive/dig posture, not block. dc=0.1° means ball did NOT deflect (block would deflect). Mis-typed attack; not a block. |
| 8 | kiki/fb6e23ff | 796 | loose-only | strong | 🟦 block | LOOSE-only, strong. p1(A) jumping high at center net with arms raised; ball at wrist at top of net. Three team B players stacked at net below — solo blocker against multiple opponents. Classic block. |
| 9 | lele/8c00e2cc | 114 | loose-only | moderate | 🟧 attack | LOOSE-only, moderate (b unknown). p2(B) small distant figure at back, ball is well ABOVE him (not contacting). dc=1.2° — no deflection. Likely no real contact at all; mis-classified attack. Not a block. |
| 10 | lili/05f7dae1 | 238 | loose-only | strong | 🟦 block | LOOSE-only, strong. p3(A) at right side of net jumping with arm extended upward; wrist at the top of the net. Ball at hand level. Two team B players (red) on opposing side of net in front of him. Single blocker. dc=13.4° clean deflection. |