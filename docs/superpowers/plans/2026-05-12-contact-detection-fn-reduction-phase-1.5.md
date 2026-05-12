# Contact Detection FN Reduction Phase 1.5 — Implementation Plan

**Goal:** Test the generator-creation threshold relaxation hypothesis. Reuses Phase 1 infrastructure; adds 6 new `*_relaxed` fields, 3 env flags, per-flag unit tests, and one combined A/B run with diagnostic-count as primary success signal.

**Architecture:** Same env-flag mechanism as Phase 1. New flags swap generator-creation thresholds (in `_find_direction_change_candidates`, velocity-peak generator, `_find_parabolic_breakpoints`) rather than validation gates. Combined-first A/B; per-flag isolation only if combined passes G-A.

**Tech Stack:** Python 3.11, existing `contact_detector.py` + `_resolve_effective_config` + `measure_contact_recall_full.py` + `redetect_all_actions.py`.

**Source:** `docs/superpowers/specs/2026-05-12-contact-detection-fn-reduction-phase-1.5-design.md`

---

### Task 1: Add 6 generator-threshold `*_relaxed` fields

**File:** `analysis/rallycut/tracking/contact_detector.py` (modify `ContactDetectionConfig`)

- [ ] Append after the Phase 1 `*_relaxed` fields:

```python
    # Phase 1.5: generator-creation threshold relaxations
    direction_change_candidate_min_deg_relaxed: float = 15.0
    direction_change_candidate_prominence_relaxed: float = 5.0
    min_peak_prominence_relaxed: float = 0.0015
    min_candidate_velocity_relaxed: float = 0.0015
    parabolic_min_residual_relaxed: float = 0.010
    parabolic_min_prominence_relaxed: float = 0.004
```

- [ ] Run `cd analysis && uv run mypy rallycut/tracking/contact_detector.py` — expect 0 errors.
- [ ] Commit: `feat(contact-detection): add Phase 1.5 generator-threshold *_relaxed fields`

### Task 2: Extend `_resolve_effective_config` with 3 new flag blocks

**File:** `analysis/rallycut/tracking/contact_detector.py` (modify `_resolve_effective_config`)

- [ ] Add three new `if` blocks at the end of the function body:

```python
    if os.environ.get("RELAX_CONTACT_DIR_GEN", "0") == "1":
        new_cfg = dataclasses.replace(
            new_cfg,
            direction_change_candidate_min_deg=cfg.direction_change_candidate_min_deg_relaxed,
            direction_change_candidate_prominence=cfg.direction_change_candidate_prominence_relaxed,
        )
    if os.environ.get("RELAX_CONTACT_VEL_GEN", "0") == "1":
        new_cfg = dataclasses.replace(
            new_cfg,
            min_peak_prominence=cfg.min_peak_prominence_relaxed,
            min_candidate_velocity=cfg.min_candidate_velocity_relaxed,
        )
    if os.environ.get("RELAX_CONTACT_PARABOLIC_GEN", "0") == "1":
        new_cfg = dataclasses.replace(
            new_cfg,
            parabolic_min_residual=cfg.parabolic_min_residual_relaxed,
            parabolic_min_prominence=cfg.parabolic_min_prominence_relaxed,
        )
```

- [ ] Run `cd analysis && uv run mypy rallycut/tracking/contact_detector.py` — expect 0 errors.
- [ ] Run `cd analysis && uv run pytest tests/unit -k contact_detector -v` — expect 107 pass (no new tests yet; existing tests must remain green).
- [ ] Commit: `feat(contact-detection): wire 3 Phase 1.5 generator-threshold flags into _resolve_effective_config`

### Task 3: Per-flag + combined unit tests

**File:** `analysis/tests/unit/test_contact_detector_relaxations.py` (append)

- [ ] Add four tests following the established pattern:

```python
def test_dir_gen_flag_lowers_threshold(
    baseline_cfg: ContactDetectionConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RELAX_CONTACT_DIR_GEN", "1")
    resolved = _resolve_effective_config(baseline_cfg)
    assert resolved.direction_change_candidate_min_deg == baseline_cfg.direction_change_candidate_min_deg_relaxed
    assert resolved.direction_change_candidate_min_deg == 15.0
    assert resolved.direction_change_candidate_prominence == baseline_cfg.direction_change_candidate_prominence_relaxed
    assert resolved.direction_change_candidate_prominence == 5.0


def test_vel_gen_flag_lowers_thresholds(
    baseline_cfg: ContactDetectionConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RELAX_CONTACT_VEL_GEN", "1")
    resolved = _resolve_effective_config(baseline_cfg)
    assert resolved.min_peak_prominence == baseline_cfg.min_peak_prominence_relaxed
    assert resolved.min_peak_prominence == 0.0015
    assert resolved.min_candidate_velocity == baseline_cfg.min_candidate_velocity_relaxed


def test_parabolic_gen_flag_lowers_thresholds(
    baseline_cfg: ContactDetectionConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RELAX_CONTACT_PARABOLIC_GEN", "1")
    resolved = _resolve_effective_config(baseline_cfg)
    assert resolved.parabolic_min_residual == baseline_cfg.parabolic_min_residual_relaxed
    assert resolved.parabolic_min_residual == 0.010
    assert resolved.parabolic_min_prominence == baseline_cfg.parabolic_min_prominence_relaxed


def test_all_three_generator_flags_apply_together(
    baseline_cfg: ContactDetectionConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    for flag in ("RELAX_CONTACT_DIR_GEN", "RELAX_CONTACT_VEL_GEN", "RELAX_CONTACT_PARABOLIC_GEN"):
        monkeypatch.setenv(flag, "1")
    resolved = _resolve_effective_config(baseline_cfg)
    assert resolved.direction_change_candidate_min_deg == 15.0
    assert resolved.min_peak_prominence == 0.0015
    assert resolved.parabolic_min_residual == 0.010
```

Also update `test_default_no_flags_preserves_cfg` to delenv the 3 new flags as well.

- [ ] Run `cd analysis && uv run pytest tests/unit/test_contact_detector_relaxations.py -v` — expect 12 pass (8 from Phase 1 + 4 new).
- [ ] Commit: `test(contact-detection): unit tests for Phase 1.5 generator-threshold flags`

### Task 4: A/B test combined generator-threshold relaxation

This is the main Phase 1.5 measurement. Three flags ON simultaneously.

- [ ] Snapshot DB state for cheap rollback:

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && uv run python -c "
import json
from rallycut.evaluation.db import get_connection
with get_connection() as conn, open('/tmp/phase15_pre_snapshot.jsonl', 'w') as f:
    with conn.cursor() as cur:
        cur.execute('''
          SELECT r.id::text, pt.contacts_json, pt.actions_json
          FROM rallies r JOIN player_tracks pt ON pt.rally_id = r.id
          WHERE pt.ball_positions_json IS NOT NULL
        ''')
        for rid, c, a in cur.fetchall():
            f.write(json.dumps({'rally_id': rid, 'contacts_json': c, 'actions_json': a}, default=str) + chr(10))
print('snapshot written')
"
```

- [ ] Confirm baseline state: `env | grep RELAX_CONTACT` (must be empty) + quick re-measure to confirm recall=0.8909.

- [ ] Apply combined Phase 1.5 relaxation:

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && \
  RELAX_CONTACT_DIR_GEN=1 RELAX_CONTACT_VEL_GEN=1 RELAX_CONTACT_PARABOLIC_GEN=1 \
  uv run python -u scripts/redetect_all_actions.py --apply > /tmp/phase15_redetect.log 2>&1
```

Expected: ~1-2 min (warm models from prior runs).

- [ ] Re-measure with all 3 flags ON:

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && \
  RELAX_CONTACT_DIR_GEN=1 RELAX_CONTACT_VEL_GEN=1 RELAX_CONTACT_PARABOLIC_GEN=1 \
  uv run python -u scripts/measure_contact_recall_full.py \
    --label phase15_combined_on \
    --output reports/contact_detection_fn/measurement_phase15_combined_2026_05_12.json
```

- [ ] Run the diagnostic count (probe cases with contact at GT frame):

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && uv run python -c "
import json
from rallycut.evaluation.db import get_connection
probe = json.load(open('reports/contact_detection_fn/probe_2026_05_12.json'))
cases = probe['cases']
from collections import defaultdict
by_action = defaultdict(lambda: [0, 0])
total_hit = 0
with get_connection() as conn, conn.cursor() as cur:
    for c in cases:
        by_action[c['gt_action'] or '?'][0] += 1
        cur.execute('SELECT contacts_json FROM player_tracks WHERE rally_id::text = %s', (c['rally_id_full'],))
        row = cur.fetchone()
        if not row or not row[0]:
            continue
        contacts = (row[0] if isinstance(row[0], dict) else {}).get('contacts', []) or []
        for ct in contacts:
            if abs(ct.get('frame', -10000) - c['gt_frame']) <= 10:
                total_hit += 1
                by_action[c['gt_action'] or '?'][1] += 1
                break
print(f'Probe cases with contact within +/-10 of GT: {total_hit}/{len(cases)}')
for a, (tot, hit) in sorted(by_action.items()):
    print(f'  {a:<10} {hit:>3}/{tot:<3} ({hit/tot*100:>5.1f}%)')
"
```

- [ ] Run unit tests with all flags ON to confirm G-D:

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && \
  RELAX_CONTACT_DIR_GEN=1 RELAX_CONTACT_VEL_GEN=1 RELAX_CONTACT_PARABOLIC_GEN=1 \
  uv run pytest tests/unit/test_contact_detector_relaxations.py -v
```

- [ ] Apply decision matrix (from spec):

| Diagnostic | Recall Δ | Action |
|---|---|---|
| ≥80, ≥+3pp | PASS — ship | Default-ON in production |
| ≥80, <+3pp | DONE_WITH_CONCERNS | Ship as infrastructure; flag GBM next |
| 40-80, ≥+3pp | PASS — ship | Same |
| 40-80, <+3pp | Continue to T5 | Per-flag A/B for clarity |
| <40 | NO-SHIP | Restore snapshot; escalate to Phase 1.7 / Phase 2 |

- [ ] Write summary `analysis/reports/contact_detection_fn/measurement_phase15_combined_2026_05_12.md` with comparison table + verdict + diagnostic count.
- [ ] If NO-SHIP: restore from snapshot (same script pattern as Phase 1 Task 9 Step 7).
- [ ] If PASS or DONE_WITH_CONCERNS: leave DB in new state.
- [ ] Commit the verdict markdown.

### Task 5: Per-flag isolation (CONDITIONAL — only if T4 lands in 40-80 / <+3pp band)

Run individual flag A/Bs to identify which generator(s) drove the partial win.

For each flag in (DIR_GEN, VEL_GEN, PARABOLIC_GEN):

- [ ] Restore from snapshot (clean baseline before each per-flag test).
- [ ] Set ONLY that flag, run redetect.
- [ ] Measure recall + diagnostic count.
- [ ] Record verdict + delta vs T4-combined-PASS-baseline.

If T4 was PASS or NO-SHIP, skip this task — no isolation needed.

### Task 6: Final report + memory update

- [ ] Write `analysis/reports/contact_detection_fn/measurement_phase15_final_2026_05_12.md` covering:
  - Phase 1.5 verdict + ship list (which flags default-ON)
  - Diagnostic count vs Phase 1's 14/173 baseline
  - Per-action-type breakdown
  - Recommended Phase 1.7 (player-motion candidates for blocks) — referenced
  - Any new findings (e.g., one specific generator dominates the recovery)

- [ ] Update `~/.claude/projects/.../memory/contact_detection_fn_v1_2026_05_12.md` (append Phase 1.5 outcome section) OR create new `contact_detection_fn_phase15_2026_05_12.md` if outcome is meaningful.

- [ ] Update `MEMORY.md` index entry to reflect Phase 1.5 verdict.

- [ ] Commit.
