# A2 Pose-Driven Attribution Probe — 2026-05-13

Spec: `docs/superpowers/specs/2026-05-13-action-attribution-root-causes-design.md`

## Aggregate

- Total cases run: **11**
- GT-anchored cases (idx 1–10): **10**
- Pose-evaluable subset (both same-team candidates with detected wrist): **9**
- `bbox_correct` (within pose-evaluable subset): **9** / 9
- `pose_correct` (within pose-evaluable subset): **9** / 9
- **Disambiguation improvement: `0.000`**  (threshold ≥ 0.5)
- Mean pose inference latency: 193.4 ms / frame

### Verdict: **NO-SHIP A2 (move to A3 or other)**

### Breakdown by kind

| kind | n | evaluable | bbox_correct | pose_correct |
|------|---|-----------|--------------|--------------|
| error | 7 | 6 | 6 | 6 |
| control | 3 | 3 | 3 | 3 |

## Per-case detail

### #1 [error] wawa/7094136a frame 306

- reason: wawa same-team ATTACK (A1 inspection flagged)
- gt_tid: `4` | pl_tid: `4` | bbox_pick: `4` | pose_pick: `None`
- bbox_dists: `{4: 0.040268986120506715, 3: 0.1780085663033003}`
- wrist_dists: `{4: None, 3: 0.1820324111372488}`
- wrist_which: `{4: None, 3: 'right'}`
- pose_evaluable: `False` | bbox_correct: `True` | pose_correct: `None` | pose_ms: `302`

### #2 [error] wawa/21d4cdf6 frame 485

- reason: wawa same-team ATTACK (A1 inspection flagged)
- gt_tid: `1` | pl_tid: `1` | bbox_pick: `1` | pose_pick: `1`
- bbox_dists: `{1: 0.04505653324751621, 2: 0.16408786774302747}`
- wrist_dists: `{1: 0.020197668795852382, 2: 0.18952968059466763}`
- wrist_which: `{1: 'right', 2: 'right'}`
- pose_evaluable: `True` | bbox_correct: `True` | pose_correct: `True` | pose_ms: `186`

### #3 [error] gigi/39e866fd frame 297

- reason: gigi same-team ATTACK
- gt_tid: `1` | pl_tid: `1` | bbox_pick: `1` | pose_pick: `1`
- bbox_dists: `{1: 0.05848492432486328, 2: 0.17041029503729585}`
- wrist_dists: `{1: 0.022731636987385433, 2: 0.198187492495662}`
- wrist_which: `{1: 'right', 2: 'right'}`
- pose_evaluable: `True` | bbox_correct: `True` | pose_correct: `True` | pose_ms: `177`

### #4 [error] cece/5c35e049 frame 501

- reason: cece same-team ATTACK
- gt_tid: `1` | pl_tid: `1` | bbox_pick: `1` | pose_pick: `1`
- bbox_dists: `{1: 0.032861400775751635, 2: 0.21247772523890418}`
- wrist_dists: `{1: 0.017124753252093474, 2: 0.23584717712641465}`
- wrist_which: `{1: 'right', 2: 'right'}`
- pose_evaluable: `True` | bbox_correct: `True` | pose_correct: `True` | pose_ms: `191`

### #5 [error] titi/43b849ec frame 344

- reason: titi (top C-5 video) same-team ATTACK
- gt_tid: `3` | pl_tid: `3` | bbox_pick: `3` | pose_pick: `3`
- bbox_dists: `{3: 0.07043281288643219, 4: 0.21127804352911236}`
- wrist_dists: `{3: 0.04293729637180848, 4: 0.24889364358599142}`
- wrist_which: `{3: 'left', 4: 'right'}`
- pose_evaluable: `True` | bbox_correct: `True` | pose_correct: `True` | pose_ms: `172`

### #6 [error] titi/a0881d82 frame 176

- reason: titi (cascade rally) ATTACK
- gt_tid: `2` | pl_tid: `2` | bbox_pick: `2` | pose_pick: `2`
- bbox_dists: `{2: 0.052017557769997835, 1: 0.1455463935871276}`
- wrist_dists: `{2: 0.02136439446647679, 1: 0.1785779569639687}`
- wrist_which: `{2: 'right', 1: 'left'}`
- pose_evaluable: `True` | bbox_correct: `True` | pose_correct: `True` | pose_ms: `174`

### #7 [error] titi/caa96651 frame 173

- reason: titi (top C-5 video) ATTACK
- gt_tid: `1` | pl_tid: `1` | bbox_pick: `1` | pose_pick: `1`
- bbox_dists: `{1: 0.06266344096041578, 2: 0.3147782929887423}`
- wrist_dists: `{1: 0.034285079977921865, 2: 0.33676734293628413}`
- wrist_which: `{1: 'right', 2: 'right'}`
- pose_evaluable: `True` | bbox_correct: `True` | pose_correct: `True` | pose_ms: `178`

### #8 [control] juju/57d1327c frame 211

- reason: juju ATTACK control
- gt_tid: `1` | pl_tid: `1` | bbox_pick: `1` | pose_pick: `1`
- bbox_dists: `{1: 0.04156299130737394, 2: 0.22645600073544295}`
- wrist_dists: `{1: 0.01024297873984624, 2: 0.2721245565279054}`
- wrist_which: `{1: 'right', 2: 'right'}`
- pose_evaluable: `True` | bbox_correct: `True` | pose_correct: `True` | pose_ms: `174`

### #9 [control] wawa/7f0f540a frame 477

- reason: wawa ATTACK control (tight gap)
- gt_tid: `2` | pl_tid: `2` | bbox_pick: `2` | pose_pick: `2`
- bbox_dists: `{2: 0.07733427768871459, 1: 0.24478067977508416}`
- wrist_dists: `{2: 0.017680809247719143, 1: 0.23543068598112682}`
- wrist_which: `{2: 'right', 1: 'left'}`
- pose_evaluable: `True` | bbox_correct: `True` | pose_correct: `True` | pose_ms: `191`

### #10 [control] lili/879a8cff frame 225

- reason: lili ATTACK control
- gt_tid: `2` | pl_tid: `2` | bbox_pick: `2` | pose_pick: `2`
- bbox_dists: `{2: 0.11850253148118084, 1: 0.38555458536595105}`
- wrist_dists: `{2: 0.06529064422831159, 1: 0.38438774999086994}`
- wrist_which: `{2: 'left', 1: 'right'}`
- pose_evaluable: `True` | bbox_correct: `True` | pose_correct: `True` | pose_ms: `175`

### #11 [f3_extra] keke/0144acfb frame 223

- reason: F3 canonical case (no GT). Logging for visual.
- gt_tid: `None` | pl_tid: `1` | bbox_pick: `1` | pose_pick: `1`
- bbox_dists: `{1: 0.004042790335685305, 4: 0.3268390814081336}`
- wrist_dists: `{1: 0.03020428237520555, 4: 0.37789481705452044}`
- wrist_which: `{1: 'left', 4: 'right'}`
- pose_evaluable: `True` | bbox_correct: `None` | pose_correct: `None` | pose_ms: `209`
