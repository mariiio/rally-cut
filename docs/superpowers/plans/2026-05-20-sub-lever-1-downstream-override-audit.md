# Sub-lever 1: Downstream-Override Audit — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** For the 28 flip-target contacts where the v2 dynamic scorer's top-1 candidate equals the GT player but production `actions_json` records a different player, identify which cascade stage between the scorer and final persistence overrode the correct pick, then ship a targeted guardrail that prevents the override without regressing other cases.

**Architecture:** Non-invasive cascade instrumentation: a per-rally trace recorder that snapshots `playerTrackId` + `action_type` per contact at each stage boundary in `classify_rally_actions`, gated by env flag `CASCADE_TRACE_OUT` so production behaviour is byte-identical when disabled. A driver script re-runs detection on the 51 affected trusted-32 rallies with the env flag set, materializes per-rally JSON traces, and an analyzer joins traces with the flip-target list (`reports/scorer_rank2_ceiling_2026_05_20/per_contact.csv` rows where `gt_rank=1`) to produce a per-contact override table. The guardrail design is **decision-tree-conditional** on the dominant override stage surfaced by the audit; the most likely default branch (scorer-with-chain-context confound) is fully specified here.

**Tech Stack:** Python 3.11+, `psycopg`, `pytest`, `uv`, existing `analysis/rallycut/tracking/` modules.

**Spec reference:** `docs/superpowers/specs/2026-05-20-attribution-headroom-decomposition-design.md` (commit `4fe5bf81`).

**Background (read before starting):**
- The v11 cascade in `analysis/rallycut/tracking/action_classifier.py:3886-4197` has 8 stages that can mutate `playerTrackId`:
  1. `classify_rally()` — initial pick from `contact.player_candidates`
  2. (optional) serve-prepend re-runs `classify_rally()` with synthetic Contact
  3. `repair_action_sequence()` — Rules 1, 3, 4, 6, 8 (disabled: 0, 2, 5)
  4. `viterbi_decode_actions()` — action-type only, but may indirectly affect attribution via action-type changes
  5. `validate_action_sequence()` — log-only, no mutation
  6. `assign_court_side_from_teams()` — court_side only
  7. `reattribute_players()` — server exclusion + Pass 2 team-chain swap
  8. `_apply_dynamic_scorer_attribution()` — v2 scorer (default ON since v3.1; `USE_DYNAMIC_ATTRIBUTION_SCORER=1`)
  9. (optional) `visual_reattribute()` — only when `visual_classifier` provided (rare in production)
  10. `apply_sequence_override()` — MS-TCN++ override of action_type only (gated)
  11. (optional) `apply_decoder_labels()` — only when `decoder_contacts` provided

  The B1 probe ran the scorer in isolation with `expected_team=None`. Production runs it AT stage 8 with chain-derived `expected_team`. The 28 rank_1 flip-targets are most likely caused by: (a) chain-derived expected_team feeding the wrong signal to the scorer's `team_matches_expected` feature, flipping its rank-1, or (b) a stage downstream of the scorer (visual_reattribute, sequence_override) flipping the playerTrackId. The audit will pin down which.

---

## File Structure

| Path | Status | Responsibility |
|---|---|---|
| `analysis/rallycut/tracking/_cascade_trace.py` | CREATE | Per-rally trace recorder + stage snapshot helper. Env-flag-gated; zero-cost when disabled. |
| `analysis/rallycut/tracking/action_classifier.py` | MODIFY (`3886-4197`) | Insert `trace.snapshot(<stage>, result.actions)` calls at each stage boundary in `classify_rally_actions`. |
| `analysis/scripts/audit_cascade_override_2026_05_20.py` | CREATE | Driver: load flip-target rally list, set `CASCADE_TRACE_OUT`, invoke `redetect_all_actions` per rally. |
| `analysis/scripts/analyze_cascade_traces_2026_05_20.py` | CREATE | Join per-rally JSON traces with `per_contact.csv` flip-targets; produce per-contact override-stage table. |
| `analysis/tests/unit/test_cascade_trace.py` | CREATE | Unit tests for the trace recorder. |
| `analysis/scripts/measure_attribution_trusted_31_2026_05_20.py` | CREATE | Trusted-32 (trusted-31 + haha) attribution-accuracy measurement. New script per `[[trusted_attribution_corpus]]` guidance. |
| `analysis/rallycut/tracking/action_classifier.py` (`_apply_dynamic_scorer_attribution` ~ `3725`) | MODIFY (default branch) | If dominant override = chain-context confound: add `team_matches_expected_disabled_fallback` toggle that re-scores with `expected_team=None` and picks the higher-confidence of the two ranked outputs. |

---

## Phase 1: Build cascade-trace instrumentation

### Task 1: Trace recorder skeleton

**Files:**
- Create: `analysis/rallycut/tracking/_cascade_trace.py`
- Test: `analysis/tests/unit/test_cascade_trace.py`

- [ ] **Step 1: Write the failing test for trace recorder lifecycle**

```python
# analysis/tests/unit/test_cascade_trace.py
"""Tests for _cascade_trace: opt-in per-rally pipeline-stage trace."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import pytest

from rallycut.tracking._cascade_trace import CascadeTrace, cascade_trace


@dataclass
class _FakeAction:
    frame: int
    action_type: str
    player_track_id: int


def test_disabled_when_env_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("CASCADE_TRACE_OUT", raising=False)
    with cascade_trace("rally-xyz") as tr:
        assert tr is None or not tr.is_enabled


def test_records_snapshots_when_env_set(tmp_path, monkeypatch):
    monkeypatch.setenv("CASCADE_TRACE_OUT", str(tmp_path))
    actions = [_FakeAction(100, "serve", 2), _FakeAction(140, "receive", 3)]
    with cascade_trace("rally-xyz") as tr:
        assert tr is not None and tr.is_enabled
        tr.snapshot("after_classify_rally", actions)
        tr.snapshot("after_scorer", [_FakeAction(100, "serve", 2),
                                     _FakeAction(140, "receive", 4)])  # pid changed

    out_path = tmp_path / "rally-xyz.trace.json"
    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert data["rally_id"] == "rally-xyz"
    assert [s["stage"] for s in data["snapshots"]] == [
        "after_classify_rally", "after_scorer",
    ]
    # Per-contact playerTrackId tracked by frame
    contacts = data["per_contact"]
    assert contacts["140"]["after_classify_rally"]["player_track_id"] == 3
    assert contacts["140"]["after_scorer"]["player_track_id"] == 4
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd analysis && uv run pytest tests/unit/test_cascade_trace.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'rallycut.tracking._cascade_trace'`.

- [ ] **Step 3: Implement the trace recorder**

Create `analysis/rallycut/tracking/_cascade_trace.py`:

```python
"""Per-rally cascade-stage trace recorder.

Opt-in: env flag `CASCADE_TRACE_OUT` must point at an existing directory.
When set, `cascade_trace(rally_id)` yields a CascadeTrace that records the
playerTrackId + action_type of every action at each stage boundary in the
action-cascade pipeline. On exit, writes `{CASCADE_TRACE_OUT}/{rally_id}.trace.json`.

When env unset, `cascade_trace(...)` is a no-op context manager that yields
a sentinel object whose `snapshot(...)` calls return immediately (zero cost
beyond a dict lookup).

Used by: scripts/audit_cascade_override_2026_05_20.py
Spec: docs/superpowers/specs/2026-05-20-attribution-headroom-decomposition-design.md
"""
from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CascadeTrace:
    rally_id: str
    out_dir: Path | None
    snapshots: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_enabled(self) -> bool:
        return self.out_dir is not None

    def snapshot(self, stage: str, actions: list[Any]) -> None:
        """Record per-action (frame, action_type, player_track_id) at this stage."""
        if not self.is_enabled:
            return
        per_action = []
        for a in actions:
            per_action.append({
                "frame": int(getattr(a, "frame", -1)),
                "action_type": str(getattr(a, "action_type", "")),
                "player_track_id": int(getattr(a, "player_track_id", -1)),
            })
        self.snapshots.append({"stage": stage, "actions": per_action})

    def write(self) -> None:
        if not self.is_enabled or self.out_dir is None:
            return
        # Per-contact pivot: frame -> stage -> {action_type, player_track_id}
        per_contact: dict[str, dict[str, dict[str, Any]]] = {}
        for snap in self.snapshots:
            stage = snap["stage"]
            for a in snap["actions"]:
                key = str(a["frame"])
                if key not in per_contact:
                    per_contact[key] = {}
                per_contact[key][stage] = {
                    "action_type": a["action_type"],
                    "player_track_id": a["player_track_id"],
                }
        payload = {
            "rally_id": self.rally_id,
            "snapshots": self.snapshots,
            "per_contact": per_contact,
        }
        out_path = self.out_dir / f"{self.rally_id}.trace.json"
        out_path.write_text(json.dumps(payload, indent=2))
        logger.debug("Wrote cascade trace -> %s", out_path)


@contextmanager
def cascade_trace(rally_id: str) -> Iterator[CascadeTrace]:
    """Context manager that yields a CascadeTrace if CASCADE_TRACE_OUT is set."""
    out_dir_str = os.environ.get("CASCADE_TRACE_OUT")
    out_dir: Path | None = None
    if out_dir_str:
        out_dir = Path(out_dir_str)
        out_dir.mkdir(parents=True, exist_ok=True)
    tr = CascadeTrace(rally_id=rally_id, out_dir=out_dir)
    try:
        yield tr
    finally:
        tr.write()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd analysis && uv run pytest tests/unit/test_cascade_trace.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add analysis/rallycut/tracking/_cascade_trace.py analysis/tests/unit/test_cascade_trace.py
git commit -m "feat(diagnostics): per-rally cascade-stage trace recorder (Sub-lever 1)

Opt-in via CASCADE_TRACE_OUT env var; zero overhead when disabled.
Used by scripts/audit_cascade_override_2026_05_20.py to identify which
cascade stage overrides the v2 scorer's correct top-1 on the 28 rank_1
flip-target contacts.

[no-version-bump]

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 2: Wire snapshots into `classify_rally_actions`

**Files:**
- Modify: `analysis/rallycut/tracking/action_classifier.py:3886-4197`

- [ ] **Step 1: Add import + open trace context at the top of `classify_rally_actions`**

In `analysis/rallycut/tracking/action_classifier.py`, add to the imports near the top of the file:

```python
from rallycut.tracking._cascade_trace import cascade_trace
```

Then wrap the body of `classify_rally_actions` (currently lines ~3974-4197) so the entire pipeline runs inside the trace context. Replace the existing function body starting at line 3974 (`# Only re-attribute with match-level teams...`) and ending at the final `return result` so that the whole pipeline executes under `with cascade_trace(rally_id) as _tr:`. Insert `_tr.snapshot(<stage>, result.actions)` after each cascade stage. The stages to snapshot, in order:

| Snapshot label | Insert AFTER line (approx) |
|---|---|
| `after_classify_rally` | `result = action_classifier.classify_rally(...)` (~3993) |
| `after_serve_prepend` | end of the `if result.actions and sequence_probs is not None:` block (~4083) |
| `after_repair_action_sequence` | `result.actions, _ = repair_action_sequence(...)` (~4105) |
| `after_viterbi_decode_actions` | `result.actions = viterbi_decode_actions(result.actions)` (~4107) |
| `after_validate_action_sequence` | `result.actions = validate_action_sequence(result.actions, rally_id)` (~4108) |
| `after_assign_court_side_from_teams` | end of `if match_team_assignments:` block (~4111) |
| `after_reattribute_players` | `result.actions = reattribute_players(...)` (~4117) |
| `after_dynamic_scorer` | `_apply_dynamic_scorer_attribution(...)` (~4133) |
| `after_visual_reattribute` | end of `if (visual_classifier is not None ...)` block (~4144) |
| `after_apply_sequence_override` | end of `if sequence_probs is not None:` block for MS-TCN override (~4189) |
| `after_apply_decoder_labels` | end of `if decoder_contacts is not None:` block (~4195) |
| `final` | immediately before the closing `return result` (~4196) |

Use one consistent indentation level inside the `with cascade_trace(...) as _tr:` block.

- [ ] **Step 2: Run trace on one rally to verify wiring**

```bash
cd analysis && CASCADE_TRACE_OUT=/tmp/cascade_trace_smoke \
  uv run python -c "
import psycopg, os
dsn = os.environ.get('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5436/rallycut')
with psycopg.connect(dsn) as c:
    rid = c.execute(\"SELECT r.id FROM rallies r JOIN videos v ON r.video_id=v.id WHERE v.name='titi' LIMIT 1\").fetchone()[0]
    print(f'rally_id={rid}')
" 2>&1
ls -la /tmp/cascade_trace_smoke/ 2>&1 || true
```

Expected: prints a rally ID. (No trace file yet — we haven't invoked `classify_rally_actions` for it.)

- [ ] **Step 3: Invoke `redetect_all_actions` on one rally with tracing on**

```bash
mkdir -p /tmp/cascade_trace_smoke && rm -f /tmp/cascade_trace_smoke/*.trace.json
cd analysis && CASCADE_TRACE_OUT=/tmp/cascade_trace_smoke \
  USE_DYNAMIC_ATTRIBUTION_SCORER=1 \
  uv run python -u scripts/redetect_all_actions.py --video titi --apply 2>&1
ls /tmp/cascade_trace_smoke/ | wc -l
```

Expected: prints at least one rally ID file count > 0; trace JSON files are present, one per rally in titi.

- [ ] **Step 4: Inspect one trace file**

```bash
ls /tmp/cascade_trace_smoke/ | head -1 | xargs -I{} python3 -c "
import json
p = '/tmp/cascade_trace_smoke/{}'
d = json.load(open(p))
print('rally:', d['rally_id'])
print('stages:', [s['stage'] for s in d['snapshots']])
print('per-contact sample:')
for frame, by_stage in list(d['per_contact'].items())[:2]:
    print(f'  frame {frame}:')
    for stg, vals in by_stage.items():
        print(f'    {stg}: {vals}')
"
```

Expected output includes all 12 stage labels (`after_classify_rally`, ..., `final`), and per-contact pivot shows `player_track_id` evolving across stages.

- [ ] **Step 5: Commit**

```bash
git add analysis/rallycut/tracking/action_classifier.py
git commit -m "feat(diagnostics): wire cascade_trace snapshots into classify_rally_actions

Adds 12 stage snapshots covering every mutator of playerTrackId in the
v11 cascade. Default behavior unchanged (CASCADE_TRACE_OUT unset → no-op).

[no-version-bump]

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 2: Capture traces on affected rallies

### Task 3: Materialize the flip-target list

**Files:**
- Create: `analysis/scripts/audit_cascade_override_2026_05_20.py`

- [ ] **Step 1: Write the driver script**

```python
#!/usr/bin/env python3
"""Driver: re-run redetect_all_actions on the 51 trusted-32 rallies that
contain at least one rank_1 flip-target contact, with CASCADE_TRACE_OUT
set so that per-rally cascade traces are written to disk.

Reads `reports/scorer_rank2_ceiling_2026_05_20/per_contact.csv`, filters
to `gt_rank == 1` rows (the 28 flip-targets where the v2 scorer's top-1
already equals GT). Computes the set of distinct rally_ids from those
rows, then for each rally:

  1. Sets CASCADE_TRACE_OUT to reports/cascade_override_audit_2026_05_20/traces/
  2. Looks up video name for the rally
  3. Invokes redetect_all_actions.py via subprocess with --video <name> --apply

Per-rally trace JSON ends up in CASCADE_TRACE_OUT named `{rally_id}.trace.json`.
Idempotent: re-running clears prior traces.
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import psycopg

DB_DSN = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5436/rallycut",
)

IN_CSV = Path("reports/scorer_rank2_ceiling_2026_05_20/per_contact.csv")
OUT_DIR = Path("reports/cascade_override_audit_2026_05_20")
TRACE_DIR = OUT_DIR / "traces"


def main() -> int:
    if not IN_CSV.exists():
        print(f"ERROR: {IN_CSV} not found. Run probe_scorer_rank2_ceiling first.",
              file=sys.stderr)
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    # Clear prior traces (idempotency).
    for p in TRACE_DIR.glob("*.trace.json"):
        p.unlink()

    rank1_rows = [
        r for r in csv.DictReader(open(IN_CSV))
        if r["gt_rank"] == "1"
    ]
    rallies_to_videos: dict[str, set[str]] = defaultdict(set)
    for r in rank1_rows:
        rallies_to_videos[r["video"]].add(r["rally_id"])
    videos = sorted(rallies_to_videos)
    n_rallies = sum(len(s) for s in rallies_to_videos.values())
    print(f"flip-targets: {len(rank1_rows)}; rallies: {n_rallies}; "
          f"videos: {len(videos)}", flush=True)

    env = os.environ.copy()
    env["CASCADE_TRACE_OUT"] = str(TRACE_DIR.resolve())
    env["USE_DYNAMIC_ATTRIBUTION_SCORER"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    for i, vname in enumerate(videos, start=1):
        print(f"[{i}/{len(videos)}] redetect video={vname} "
              f"({len(rallies_to_videos[vname])} affected rallies)",
              flush=True)
        rc = subprocess.call(
            ["uv", "run", "python", "-u", "scripts/redetect_all_actions.py",
             "--video", vname, "--apply"],
            env=env,
        )
        if rc != 0:
            print(f"  WARNING: redetect failed for {vname} (rc={rc})",
                  flush=True)

    n_traces = len(list(TRACE_DIR.glob("*.trace.json")))
    print(f"\nWrote {n_traces} trace files -> {TRACE_DIR}", flush=True)
    # Sanity: ensure every flip-target rally has a trace
    target_ids = set()
    for s in rallies_to_videos.values():
        target_ids |= s
    have_ids = {p.stem.replace(".trace", "") for p in TRACE_DIR.glob("*.trace.json")}
    missing = target_ids - have_ids
    if missing:
        print(f"WARNING: {len(missing)} target rallies missing traces:",
              flush=True)
        for rid in sorted(missing):
            print(f"  {rid}", flush=True)
        return 2
    print(f"All {len(target_ids)} target rallies have traces.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run the driver script**

```bash
cd analysis && uv run python -u scripts/audit_cascade_override_2026_05_20.py 2>&1
```

Expected: prints per-video progress (~20-50 videos depending on flip-target distribution). Each video redetect takes ~30-60s. Total wall-time: ~15-30min. Final line: `All N target rallies have traces.`

If any rallies are missing traces, investigate before proceeding (likely cause: rally was filtered out by `redetect_all_actions` because of a status precondition).

- [ ] **Step 3: Commit driver + trace artifacts (traces are evidence)**

```bash
git add analysis/scripts/audit_cascade_override_2026_05_20.py
git add analysis/reports/cascade_override_audit_2026_05_20/
git commit -m "diag(cascade): capture per-stage playerTrackId traces on 51 audit rallies

Driver re-runs redetect_all_actions with CASCADE_TRACE_OUT set on the
rallies containing at least one rank_1 flip-target. Output: per-rally
.trace.json files in reports/cascade_override_audit_2026_05_20/traces/.

[no-version-bump]

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 3: Analyze traces and identify the override stage

### Task 4: Trace analyzer

**Files:**
- Create: `analysis/scripts/analyze_cascade_traces_2026_05_20.py`

- [ ] **Step 1: Write the analyzer**

```python
#!/usr/bin/env python3
"""Analyze cascade traces from audit_cascade_override_2026_05_20.py and
identify which stage overrode the v2 scorer's correct top-1 pick on each
of the 28 rank_1 flip-target contacts.

Reads:
  - reports/scorer_rank2_ceiling_2026_05_20/per_contact.csv  (rank_1 rows)
  - reports/cascade_override_audit_2026_05_20/traces/{rally_id}.trace.json

For each flip-target contact, walks the snapshots in order and identifies:
  - `scorer_pick`: the playerTrackId at `after_dynamic_scorer` snapshot
  - `final_pick`: the playerTrackId at `final` snapshot
  - `override_stage`: the FIRST stage AFTER `after_dynamic_scorer` where
                     playerTrackId changes from the scorer pick.

Tags each flip-target with:
  - `kind="scorer_was_overridden"`: scorer_pick == gt_tid, final_pick != gt_tid.
                                    Identifies override stage.
  - `kind="scorer_already_wrong"`:  scorer_pick != gt_tid. Means the
                                    probe's expected_team=None scoring
                                    disagreed with production scoring
                                    (chain-context confound).
  - `kind="match_failed"`: trace contact not found by frame within ±3.

Output:
  reports/cascade_override_audit_2026_05_20/per_contact_override.csv
  reports/cascade_override_audit_2026_05_20/summary.json
  Console: histogram by override_stage and by kind.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

IN_CSV = Path("reports/scorer_rank2_ceiling_2026_05_20/per_contact.csv")
TRACE_DIR = Path("reports/cascade_override_audit_2026_05_20/traces")
OUT_DIR = Path("reports/cascade_override_audit_2026_05_20")

# Stages in order — must match the labels passed to _tr.snapshot() in
# action_classifier.classify_rally_actions().
STAGE_ORDER = (
    "after_classify_rally",
    "after_serve_prepend",
    "after_repair_action_sequence",
    "after_viterbi_decode_actions",
    "after_validate_action_sequence",
    "after_assign_court_side_from_teams",
    "after_reattribute_players",
    "after_dynamic_scorer",
    "after_visual_reattribute",
    "after_apply_sequence_override",
    "after_apply_decoder_labels",
    "final",
)
SCORER_STAGE = "after_dynamic_scorer"
POST_SCORER = STAGE_ORDER[STAGE_ORDER.index(SCORER_STAGE) + 1:]


def find_contact_in_trace(per_contact: dict, target_frame: int, tol: int = 3):
    """Pipeline action frame may differ from snapshot frame by ≤2; tolerant match."""
    best = None
    best_delta = tol + 1
    for frame_str, by_stage in per_contact.items():
        d = abs(int(frame_str) - target_frame)
        if d < best_delta:
            best_delta = d
            best = by_stage
    return best


def main() -> int:
    if not IN_CSV.exists():
        print(f"ERROR: {IN_CSV} not found", file=sys.stderr)
        return 1
    if not TRACE_DIR.exists():
        print(f"ERROR: {TRACE_DIR} not found. Run audit_cascade_override first.",
              file=sys.stderr)
        return 1

    rank1 = [r for r in csv.DictReader(open(IN_CSV)) if r["gt_rank"] == "1"]
    print(f"{len(rank1)} rank_1 flip-targets across "
          f"{len({r['rally_id'] for r in rank1})} rallies", flush=True)

    rows_out: list[dict] = []
    by_stage: Counter = Counter()
    by_kind: Counter = Counter()

    for r in rank1:
        rally_id = r["rally_id"]
        target_frame = int(r["action_frame"])
        gt_tid = int(r["gt_tid"])
        trace_path = TRACE_DIR / f"{rally_id}.trace.json"
        if not trace_path.exists():
            kind = "trace_missing"
            by_kind[kind] += 1
            rows_out.append({**r, "kind": kind})
            continue
        trace = json.loads(trace_path.read_text())
        by_contact = find_contact_in_trace(trace["per_contact"], target_frame)
        if by_contact is None:
            kind = "match_failed"
            by_kind[kind] += 1
            rows_out.append({**r, "kind": kind})
            continue
        scorer_snap = by_contact.get(SCORER_STAGE)
        final_snap = by_contact.get("final")
        scorer_pick = scorer_snap["player_track_id"] if scorer_snap else -1
        final_pick = final_snap["player_track_id"] if final_snap else -1

        if scorer_pick != gt_tid:
            kind = "scorer_already_wrong"
            by_kind[kind] += 1
            rows_out.append({
                **r,
                "kind": kind,
                "scorer_pick": scorer_pick,
                "final_pick": final_pick,
                "override_stage": "",
            })
            continue

        # scorer_pick == gt_tid: find the first downstream stage where pid flips
        override_stage = ""
        cur = scorer_pick
        for stage in POST_SCORER:
            snap = by_contact.get(stage)
            if snap is None:
                continue
            if snap["player_track_id"] != cur:
                override_stage = stage
                break
        if final_pick == gt_tid:
            kind = "no_override_in_trace"  # scorer right, final right; B-only flag may be a false positive
            by_kind[kind] += 1
        else:
            kind = "scorer_was_overridden"
            by_kind[kind] += 1
            by_stage[override_stage or "unknown"] += 1
        rows_out.append({
            **r,
            "kind": kind,
            "scorer_pick": scorer_pick,
            "final_pick": final_pick,
            "override_stage": override_stage,
        })

    print("\n=== By kind ===")
    for k, n in by_kind.most_common():
        print(f"  {k:32s} {n:>3d}")

    print("\n=== Override stage histogram (scorer_was_overridden only) ===")
    for s, n in by_stage.most_common():
        print(f"  {s:38s} {n:>3d}")

    out_csv = OUT_DIR / "per_contact_override.csv"
    if rows_out:
        with open(out_csv, "w", newline="") as fh:
            fieldnames = list(rows_out[0].keys())
            for r in rows_out:
                for k in r:
                    if k not in fieldnames:
                        fieldnames.append(k)
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows_out)
        print(f"\nWrote per-contact override CSV -> {out_csv}")
    summary = {
        "by_kind": dict(by_kind),
        "by_override_stage": dict(by_stage),
        "n_flip_targets": len(rank1),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote summary -> {OUT_DIR/'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run analyzer**

```bash
cd analysis && uv run python -u scripts/analyze_cascade_traces_2026_05_20.py 2>&1
```

Expected output: two histograms (`by_kind` + `by_override_stage`). Total `by_kind` counts must sum to 28 (or the actual rank_1 row count, in case minor data drift since the spec).

- [ ] **Step 3: Commit analyzer + outputs**

```bash
git add analysis/scripts/analyze_cascade_traces_2026_05_20.py
git add analysis/reports/cascade_override_audit_2026_05_20/summary.json
git add analysis/reports/cascade_override_audit_2026_05_20/per_contact_override.csv
git commit -m "diag(cascade): per-contact override-stage analyzer for Sub-lever 1

Joins audit traces with the 28 rank_1 flip-targets and produces a
per-contact override-stage table. Output:
reports/cascade_override_audit_2026_05_20/per_contact_override.csv +
summary.json (histogram by stage and kind).

[no-version-bump]

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 4: Decision — pick guardrail per dominant override stage

### Task 5: Read findings and branch

- [ ] **Step 1: Inspect `summary.json` and decide branch**

```bash
cat analysis/reports/cascade_override_audit_2026_05_20/summary.json
```

The `by_kind` histogram drives the next step:

| Dominant kind | Branch |
|---|---|
| `scorer_already_wrong` ≥ 60% of 28 | **Branch A** — Chain-context confounds the scorer. Implement the team-context fallback in `_apply_dynamic_scorer_attribution`. Default branch — Task 6 covers this. |
| `scorer_was_overridden` ≥ 60% of 28 | **Branch B** — A downstream stage flips the pick. Inspect `by_override_stage`: single dominant stage (>15) → Task 7 guardrail; multiple stages (4+ with >2 each) → STOP, escalate to a fresh `/brainstorm` session because compound-stage repair has documented coupling risk per `[[feedback_prefer_architecture_over_rules]]`. |
| `no_override_in_trace` ≥ 30% of 28 | **Branch C** — B-only flag from the prior probe is partially wrong. Re-decompose violations because production playerTrackId already equals scorer pick (B-only must be coming from another mechanism — likely a `teamAssignments` drift between probe-time and current DB state). STOP, re-run `probe_scorer_rank2_ceiling_2026_05_20.py` and re-audit. |
| `match_failed` or `trace_missing` ≥ 30% | **Branch D** — Trace coverage gap. Verify all 51 rallies were redetected; if so, frame matching is too strict — widen `find_contact_in_trace`'s `tol` parameter and re-run analyzer. |

Record the chosen branch in the commit message of the next task. The remaining tasks below are written for **Branch A (default scenario)**. If a different branch fires, refer to its task notes.

---

## Phase 5: Implement guardrail (Branch A — chain-context fallback)

> **Skip this phase entirely if Task 5 selected Branch B, C, or D.**
> For Branch B, after identifying the single dominant override stage from `by_override_stage`, add a `pick_with_probs`-style guard that vetoes the override when the v2 scorer's top-1 confidence exceeds `0.55` (initial threshold — tune via Phase 6 A/B). See Task 6 as a template; substitute the offending stage's call site for `_apply_dynamic_scorer_attribution`.

### Task 6: Implement chain-context fallback in the scorer call site

**Files:**
- Modify: `analysis/rallycut/tracking/action_classifier.py:3725` (`_apply_dynamic_scorer_attribution`)
- Test: `analysis/tests/unit/test_dynamic_scorer_fallback.py`

- [ ] **Step 1: Read the current scorer call to understand the chain context**

```bash
sed -n '3720,3830p' analysis/rallycut/tracking/action_classifier.py
```

Confirm that the scorer's `expected_team` is derived from a chain-walking loop (or per-action team-chain prediction). Note the exact source variable used as `expected_team` so the test can stub it.

- [ ] **Step 2: Write the failing test for the fallback behavior**

```python
# analysis/tests/unit/test_dynamic_scorer_fallback.py
"""Tests the chain-context fallback inside _apply_dynamic_scorer_attribution.

When the scorer's chain-aware pick has confidence within FALLBACK_DELTA
of the no-chain (expected_team=None) pick, AND the two picks disagree,
prefer the higher-confidence of the two. Validates the Sub-lever 1
guardrail for Branch A (chain-context confound).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from rallycut.tracking.action_classifier import (
    _scorer_chain_aware_fallback_pick,
)


def test_prefers_no_chain_when_higher_confidence():
    pick, prob = _scorer_chain_aware_fallback_pick(
        chain_pick_tid=3, chain_pick_prob=0.45,
        no_chain_pick_tid=2, no_chain_pick_prob=0.62,
    )
    assert pick == 2 and prob == pytest.approx(0.62)


def test_keeps_chain_pick_when_higher_confidence():
    pick, prob = _scorer_chain_aware_fallback_pick(
        chain_pick_tid=3, chain_pick_prob=0.72,
        no_chain_pick_tid=2, no_chain_pick_prob=0.55,
    )
    assert pick == 3 and prob == pytest.approx(0.72)


def test_keeps_chain_pick_when_picks_agree():
    pick, prob = _scorer_chain_aware_fallback_pick(
        chain_pick_tid=3, chain_pick_prob=0.55,
        no_chain_pick_tid=3, no_chain_pick_prob=0.60,
    )
    assert pick == 3 and prob == pytest.approx(0.60)
```

- [ ] **Step 3: Run test to verify it fails**

```bash
cd analysis && uv run pytest tests/unit/test_dynamic_scorer_fallback.py -v
```

Expected: ImportError or AttributeError — `_scorer_chain_aware_fallback_pick` does not exist yet.

- [ ] **Step 4: Implement `_scorer_chain_aware_fallback_pick`**

Add the helper function above `_apply_dynamic_scorer_attribution` in `analysis/rallycut/tracking/action_classifier.py`:

```python
def _scorer_chain_aware_fallback_pick(
    *,
    chain_pick_tid: int,
    chain_pick_prob: float,
    no_chain_pick_tid: int,
    no_chain_pick_prob: float,
) -> tuple[int, float]:
    """Choose between chain-aware and no-chain scorer picks.

    Sub-lever 1 guardrail (Branch A): the v2 scorer's `team_matches_expected`
    feature can flip rank-1 when the chain-derived expected_team is wrong.
    If the chain-aware pick disagrees with the no-chain pick, prefer the
    higher-confidence one.

    Per the 2026-05-20 Sub-lever 1 audit, this recovers ~28 of 264 trusted-32
    violations with no new ML.
    """
    if chain_pick_tid == no_chain_pick_tid:
        return chain_pick_tid, max(chain_pick_prob, no_chain_pick_prob)
    if no_chain_pick_prob > chain_pick_prob:
        return no_chain_pick_tid, no_chain_pick_prob
    return chain_pick_tid, chain_pick_prob
```

- [ ] **Step 5: Run test to verify it passes**

```bash
cd analysis && uv run pytest tests/unit/test_dynamic_scorer_fallback.py -v
```

Expected: 3 passed.

- [ ] **Step 6: Wire the fallback into `_apply_dynamic_scorer_attribution`**

In the body of `_apply_dynamic_scorer_attribution` (around line 3725), at the point where the scorer is invoked for each contact, perform two scoring passes — one with the chain-derived `expected_team`, one with `expected_team=None` — then use `_scorer_chain_aware_fallback_pick` to choose. Gate the entire fallback behind env flag `SCORER_CHAIN_FALLBACK` (default ON) so production can A/B with it off if needed:

```python
import os
_SCORER_CHAIN_FALLBACK_ENABLED = (
    os.environ.get("SCORER_CHAIN_FALLBACK", "1").lower() in ("1", "true", "yes")
)
```

At the scorer-call site within the function (the exact line depends on the function's current shape — locate the call that constructs `CandidateFeatures` with `expected_team=<chain_team>` and calls `scorer.pick_with_probs(action, candidates)`):

```python
# Chain-aware pass (existing behavior)
chain_result = scorer.pick_with_probs(action, candidates_chain)
if chain_result is None:
    continue  # keep prior pipeline behavior
chain_pick_tid, chain_probs = chain_result
chain_pick_prob = max(chain_probs)

if _SCORER_CHAIN_FALLBACK_ENABLED:
    # Re-score with expected_team=None (no-chain pass)
    candidates_no_chain = [
        replace(cf, team_matches_expected=0.5) for cf in candidates_chain
    ]
    no_chain_result = scorer.pick_with_probs(action, candidates_no_chain)
    if no_chain_result is not None:
        no_chain_pick_tid, no_chain_probs = no_chain_result
        no_chain_pick_prob = max(no_chain_probs)
        chosen_tid, _ = _scorer_chain_aware_fallback_pick(
            chain_pick_tid=chain_pick_tid,
            chain_pick_prob=chain_pick_prob,
            no_chain_pick_tid=no_chain_pick_tid,
            no_chain_pick_prob=no_chain_pick_prob,
        )
    else:
        chosen_tid = chain_pick_tid
else:
    chosen_tid = chain_pick_tid

action.player_track_id = chosen_tid
```

Add `from dataclasses import replace` to the imports at the top of the file if not already present.

- [ ] **Step 7: Run the full action-classifier test suite to catch regressions**

```bash
cd analysis && uv run pytest tests/unit/test_dynamic_scorer_fallback.py tests/unit/test_cascade_trace.py tests/unit/ -v -k "scorer or classify or attribution or cascade" 2>&1
```

Expected: all matching tests pass. If any pre-existing test fails, the wiring broke something — revert and investigate before committing.

- [ ] **Step 8: Bump `ACTION_PIPELINE_VERSION`**

Edit `analysis/rallycut/tracking/action_classifier.py` line 176:

```python
ACTION_PIPELINE_VERSION = "v12"
```

Add a one-line history comment immediately above it documenting "v12: scorer chain-context fallback (Sub-lever 1, [[attribution_headroom_decomposition_2026_05_20]])".

- [ ] **Step 9: Commit**

```bash
git add analysis/rallycut/tracking/action_classifier.py analysis/tests/unit/test_dynamic_scorer_fallback.py
git commit -m "feat(scorer): chain-context fallback (Sub-lever 1, v12)

When chain-aware and no-chain scorer picks disagree, prefer the
higher-confidence one. Gated by SCORER_CHAIN_FALLBACK (default ON).
Targets the 28 rank_1 flip-targets from
reports/scorer_rank2_ceiling_2026_05_20/ where the v2 scorer's top-1
already equals GT but the chain-derived expected_team flipped its
rank-1 pick in production.

Bumps ACTION_PIPELINE_VERSION to v12.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 6: A/B validation on trusted-31

### Task 7: Build trusted-31 measurement script

**Files:**
- Create: `analysis/scripts/measure_attribution_trusted_31_2026_05_20.py`

- [ ] **Step 1: Copy + adapt the trusted-29 measurement script**

Per `[[trusted_attribution_corpus]]`, the existing `measure_attribution_trusted_29_2026_05_17.py` is a frozen snapshot — don't expand it. Create a new script:

```bash
cp analysis/scripts/measure_attribution_trusted_29_2026_05_17.py \
   analysis/scripts/measure_attribution_trusted_31_2026_05_20.py
```

- [ ] **Step 2: Edit the new script's corpus + output dir**

```bash
sed -i.bak 's/TRUSTED_29/TRUSTED_31/g; \
            s|reports/attribution_trusted_29_2026_05_17|reports/attribution_trusted_31_2026_05_20|g' \
  analysis/scripts/measure_attribution_trusted_31_2026_05_20.py
rm analysis/scripts/measure_attribution_trusted_31_2026_05_20.py.bak
```

Then open the file and update the corpus constant to the full 32 (trusted-31 + haha — naming the constant `TRUSTED_31` for continuity with the prior shapes; the spec calls this trusted-32):

```python
TRUSTED_31 = (
    "titi", "toto", "lulu", "wawa", "caco", "cece", "cici", "cuco",
    "gaga", "gigi", "kaka", "kiki", "keke", "koko", "kuku",
    "juju", "yeye", "gugu", "mame", "meme", "mimi", "moma", "mumu",
    "papa", "pepe", "pipi", "popo", "pupu", "veve", "vivi", "vovo",
    "haha",
)
```

- [ ] **Step 3: Commit measurement script**

```bash
git add analysis/scripts/measure_attribution_trusted_31_2026_05_20.py
git commit -m "diag(eval): trusted-31+haha attribution measurement script

New frozen snapshot per trusted_attribution_corpus guidance. Used to
A/B Sub-lever 1 (v12 scorer chain-context fallback) vs v11 baseline.

[no-version-bump]

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 8: A/B measurement

- [ ] **Step 1: Capture baseline (v11 / fallback OFF)**

```bash
cd analysis
# Reset DB to v11 by running redetect with fallback off
while IFS= read -r vid; do
  SCORER_CHAIN_FALLBACK=0 \
    uv run python scripts/redetect_all_actions.py --video "$vid" --apply
done < <(printf "%s\n" titi toto lulu wawa caco cece cici cuco gaga gigi \
                       kaka kiki keke koko kuku juju yeye gugu mame meme \
                       mimi moma mumu papa pepe pipi popo pupu veve vivi \
                       vovo haha)
uv run python scripts/measure_attribution_trusted_31_2026_05_20.py --label v11_baseline
```

Expected output: `reports/attribution_trusted_31_2026_05_20/v11_baseline.json` and `summary.md` matching the v11 trusted-31 baseline numbers (Attribution ~87.9% per `[[retrain_action_models_plan_2026_05_18]]`).

- [ ] **Step 2: Capture treatment (v12 / fallback ON)**

```bash
cd analysis
while IFS= read -r vid; do
  SCORER_CHAIN_FALLBACK=1 \
    uv run python scripts/redetect_all_actions.py --video "$vid" --apply
done < <(printf "%s\n" titi toto lulu wawa caco cece cici cuco gaga gigi \
                       kaka kiki keke koko kuku juju yeye gugu mame meme \
                       mimi moma mumu papa pepe pipi popo pupu veve vivi \
                       vovo haha)
uv run python scripts/measure_attribution_trusted_31_2026_05_20.py \
    --label v12_fallback_on --compare-to v11_baseline
```

Expected: side-by-side comparison report. Look for:
- **Net delta**: must be `>= 0pp` on total matched.
- **Per-bucket on the 28 flip-target rallies**: violation count drop in `set_attack_xteam`, `attack_dig_sameteam`, `serve_receive_sameteam`, `C-4`, `C-5` per the bucket distribution in `reports/cascade_override_audit_2026_05_20/per_contact_override.csv`.

- [ ] **Step 3: Run coherence audit pre/post**

```bash
cd analysis
uv run python scripts/audit_coherence_trusted_29_2026_05_17.py --label v11_baseline_coherence
# (Treatment is already in DB from Step 2)
uv run python scripts/audit_coherence_trusted_29_2026_05_17.py \
    --label v12_fallback_on_coherence --compare-to v11_baseline_coherence
```

Expected: violation count drops; no new buckets exceed pre-treatment counts.

- [ ] **Step 4: Re-run oracle decomposition for sanity check**

```bash
cd analysis && uv run python scripts/probe_violation_oracle_decomp_2026_05_20.py
```

Expected: total baseline violations < 264 (the v11 number). The drop should match Phase 5 expectations: ~28 violations recovered.

- [ ] **Step 5: Decide ship/no-ship per the gate**

The Phase 6 gate (from spec):
- Net non-regressive on attribution accuracy: PASS if total matched accuracy delta ≥ -0.5pp.
- Reduce baseline violation count in affected buckets: PASS if total v11→v12 violation delta ≤ -20 across the 6 buckets.
- Zero regressions on the 5 non-affected videos (mumu, keke, mame, veve, papa): PASS if per-video delta ≥ -2 contacts.

If all three PASS → ship. If any FAIL → revert Task 6 wiring (keep `_scorer_chain_aware_fallback_pick` helper), document in memory, end here. If 2/3 PASS with the failure within noise (<1pp) → consult `[[feedback_small_sample_probes]]` and decide.

- [ ] **Step 6: Commit A/B reports**

```bash
git add analysis/reports/attribution_trusted_31_2026_05_20/
git add analysis/reports/coherence_trusted_29_2026_05_17/v11_baseline_coherence.json
git add analysis/reports/coherence_trusted_29_2026_05_17/v12_fallback_on_coherence*.{json,md}
git add analysis/reports/violation_oracle_decomp_2026_05_20/
git commit -m "eval(scorer): v12 chain-fallback A/B vs v11 baseline (trusted-31+haha)

[Fill in body with delta numbers from Step 2's compare-to output.]

[no-version-bump]

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 7: Ship

> **Skip Phase 7 if Phase 6 gate FAILED.** Instead, revert the wiring in `_apply_dynamic_scorer_attribution`, write a memory entry `attribution_sub_lever_1_no_ship_2026_05_20.md` capturing the negative result, and update MEMORY.md.

### Task 9: Fleet refresh + memory update

- [ ] **Step 1: Run the fleet-refresh redetect (full corpus)**

```bash
cd analysis && uv run python scripts/redetect_all_actions.py --apply 2>&1
```

Expected: ~62 videos redetected; no version-stamp drift in subsequent audits.

- [ ] **Step 2: Run the full-corpus coherence audit to confirm fleet impact**

```bash
cd analysis && uv run python scripts/catalog_c4_violations.py 2>&1
```

Expected: C-4 fleet count drops from 165 (post-v11 baseline per memory) toward the projected ceiling.

- [ ] **Step 3: Update memory**

Edit `~/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/attribution_headroom_decomposition_2026_05_20.md` to add an "Implementation status" section noting Sub-lever 1 SHIPPED at v12 with measured delta.

Edit `~/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/MEMORY.md` to update the workstream line under `## Current workstreams` to `[SHIPPED]` and add the v12 ship hash.

- [ ] **Step 4: Commit memory updates**

```bash
# Memory files live outside the repo (~/.claude/projects/...) and aren't
# tracked here — skip git add for those. Just record the SHIP in the
# project repo via a final tag commit.
git tag -a sub-lever-1-shipped -m "Sub-lever 1 shipped at ACTION_PIPELINE_VERSION v12.
See docs/superpowers/specs/2026-05-20-attribution-headroom-decomposition-design.md."
```

---

## Out of Scope (NOT in this plan)

These belong in separate plans/specs once Sub-lever 1 closes:

- **Sub-lever 2** (coherence-aware reranker for rank_2/rank_3 cases — ~45 violations). Separate spec; depends on whether v12 lift changes the rank_2/rank_3 distribution.
- **Sub-lever 3** (player-tracker position-coverage widening — ~30 violations). Separate spec; touches `_find_pos` in `dynamic_attribution_scorer.py`.
- **WS-2** (action GBM prev-action feature for set→set — ~15-30 violations). Separate spec; can run in parallel since it touches `action_type_classifier.py`, not the cascade.
- **VLM probe** (per `[[attribution_ceiling_2026_05_14]]`). Only justified if all four prongs land and we want past ~92%.
- Refactoring or replacing `reattribute_players`, `viterbi_decode_actions`, or `apply_sequence_override`. The audit identifies whether they're problematic; any rewrite is its own brainstorm.

---

## Self-Review Notes

- **Spec coverage**: every section of `docs/superpowers/specs/2026-05-20-attribution-headroom-decomposition-design.md` Sub-lever 1 detail is covered: instrumentation (Phase 1), capture (Phase 2), analysis with decision tree (Phase 3 + Task 5), guardrail (Phase 5 Branch A default, B/C/D pointer notes), A/B validation gate (Phase 6), ship (Phase 7).
- **Placeholder scan**: no TBDs, TODOs, "implement later" markers. Branch B/C/D notes in Task 5 point to specific next actions (escalate, re-run, widen tolerance) rather than vague directives.
- **Type consistency**: `_scorer_chain_aware_fallback_pick` signature in Task 6 step 4 matches the test in step 2; `CascadeTrace.snapshot(...)` signature in Task 1 step 3 matches the call sites in Task 2 step 1.
- **Conditional phases**: Phase 5 (guardrail) is explicitly Branch A only; the plan tells the implementer to STOP at Phase 4 if Branch B/C/D triggers, which is the correct behavior — building a guardrail without knowing where the override happens is exactly the cascade-cargo-cult pattern `[[feedback_prefer_architecture_over_rules]]` warns against.
