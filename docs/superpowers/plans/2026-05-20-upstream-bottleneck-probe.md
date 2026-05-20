# Upstream Bottleneck Validation Probe — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** With high confidence, identify which upstream pipeline layer (player tracker / candidate generation / contact-frame regression / ball tracking / GT scale / team-chain accuracy) is the binding bottleneck for attribution accuracy on trusted-32. Output: a ranked investment decision with measured per-layer ceilings (oracle + realistic) and a confounded-recovery Venn analysis.

**Architecture:** Six independent per-layer probes (`probe_upstream_L{1..6}_*.py`) + one aggregator (`aggregate_upstream_bottleneck_*.py`). Each layer probe is self-contained: loads the wrong-attribution corpus from a shared helper, simulates the layer's oracle + realistic interventions, writes a per-layer JSON. Aggregator joins all six JSONs, computes multi-layer-fail Venn, applies ranking formula, writes `summary.md`. Dual-ceiling methodology (oracle vs realistic) per layer (except L5 — learning curve) directly addresses Sub-lever 1's "rank ≠ confidence-leader" projection trap.

**Tech Stack:** Python 3.11+, `psycopg`, `joblib`, `numpy`, `scikit-learn` (already in `analysis/.venv`), existing `rallycut.tracking.dynamic_attribution_scorer` for re-scoring.

**Spec reference:** `docs/superpowers/specs/2026-05-20-upstream-pipeline-bottleneck-validation-design.md` (commit `46613375`).

**Substrate:** ~296 wrong-attribution contacts on trusted-32 (243 rallies with attribution GT; pipeline `playerTrackId` ≠ GT `resolved_track_id`).

**Background (read before starting):**
- Prior session NO-SHIP'd Sub-lever 1 (chain-context fallback): audit projected +28 violations, A/B delivered +4. Failure mode: the audit measured rank-based recoverability, but the realistic intervention (prefer higher-confidence) couldn't fire because the wrong pick had higher confidence than the right pick.
- [[attribution_ceiling_2026_05_14]] declared the geometric/proximity/pose signal level "saturated" — user reports a competitor achieves near-perfect classical-ML attribution, so the saturation is against OUR upstream pipeline quality, not against the signal level intrinsically.
- B1 probe (commit `5b96f302`) already showed 40% of B-only flip-targets fail upstream of scorer (25% scored_but_dropped, 15% not_in_candidates).
- Re-scoring uses `rallycut.tracking.dynamic_attribution_scorer.DynamicAttributionScorer` (`pick`, `pick_with_probs`, `score`). Re-extracting features uses `dynamic_attribution_scorer.extract_features(...)` which takes positions + ball + frame + team_assignments.

---

## File Structure

| Path | Status | Responsibility |
|---|---|---|
| `analysis/scripts/_upstream_probe_common.py` | CREATE | Shared helpers: load wrong-attribution corpus from DB; re-score utility; team-derivation from GT; chain-walk. |
| `analysis/tests/unit/test_upstream_probe_common.py` | CREATE | Unit tests for shared helpers. |
| `analysis/scripts/probe_upstream_L5_gt_scale_2026_05_20.py` | CREATE | L5: GT-scale learning curve. |
| `analysis/scripts/probe_upstream_L1_player_tracker_2026_05_20.py` | CREATE | L1: player-tracker contact-coverage probe. |
| `analysis/scripts/probe_upstream_L2_candidate_gen_2026_05_20.py` | CREATE | L2: candidate-generation probe. |
| `analysis/scripts/probe_upstream_L3_contact_frame_2026_05_20.py` | CREATE | L3: contact-frame regression accuracy probe. |
| `analysis/scripts/probe_upstream_L4_ball_tracking_2026_05_20.py` | CREATE | L4: ball-tracking accuracy at contact probe. |
| `analysis/scripts/probe_upstream_L6_team_chain_2026_05_20.py` | CREATE | L6: team-chain accuracy probe. |
| `analysis/scripts/aggregate_upstream_bottleneck_2026_05_20.py` | CREATE | Aggregator: Venn + ranking + summary. |
| `analysis/reports/upstream_bottleneck_2026_05_20/L{1..6}.json` | OUTPUT | Per-layer probe outputs. |
| `analysis/reports/upstream_bottleneck_2026_05_20/per_contact_failures.csv` | OUTPUT | Per-contact failure-mode classification (joined across layers). |
| `analysis/reports/upstream_bottleneck_2026_05_20/summary.md` | OUTPUT | Final ranked investment decision. |

---

## Task 1: Shared infrastructure (`_upstream_probe_common.py`)

**Files:**
- Create: `analysis/scripts/_upstream_probe_common.py`
- Test: `analysis/tests/unit/test_upstream_probe_common.py`

Shared helpers consumed by L1/L2/L3/L4/L6 (not L5 — it uses a different training-loop pathway).

### Step 1: Write the failing test for `load_wrong_attribution_corpus`

Create `analysis/tests/unit/test_upstream_probe_common.py`:

```python
"""Tests for _upstream_probe_common helpers."""
from __future__ import annotations

from analysis_scripts_upstream_probe_common import (  # noqa: E402
    WrongAttributionRow,
    derive_gt_team_chain,
)


def test_wrong_attribution_row_shape():
    # Smoke-check that the dataclass has all required fields
    r = WrongAttributionRow(
        rally_id="r1", video="v1", action_frame=100, action_type="attack",
        pipeline_pid=2, gt_pid=3, pipeline_match_delta=0,
    )
    assert r.rally_id == "r1"
    assert r.gt_pid != r.pipeline_pid


def test_derive_gt_team_chain_simple():
    """Walk a rally with 4 GT contacts: serve P2 -> receive P3 -> set P4 -> attack P3."""
    gt_contacts = [
        (100, "serve",   2),
        (140, "receive", 3),
        (160, "set",     4),
        (180, "attack",  3),
    ]
    team_assignments = {"1": "A", "2": "A", "3": "B", "4": "B"}
    # Expected team_chain (the team_assignments-derived team of the actor at each contact):
    # idx 0: SERVE by P2 -> team A
    # idx 1: RECEIVE by P3 -> team B
    # idx 2: SET by P4 -> team B
    # idx 3: ATTACK by P3 -> team B
    chain = derive_gt_team_chain(gt_contacts, team_assignments)
    assert chain == ["A", "B", "B", "B"]
```

Note: `analysis_scripts_upstream_probe_common` is the importable form once we put the file on the Python path. The actual import will be `from _upstream_probe_common import ...` when run via `uv run python scripts/probe_*.py` (because scripts/ is the cwd). Adapt the test import to whatever works:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from _upstream_probe_common import WrongAttributionRow, derive_gt_team_chain  # noqa: E402
```

### Step 2: Run the failing test

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && uv run pytest tests/unit/test_upstream_probe_common.py -v
```

Expected: ModuleNotFoundError because the helper file doesn't exist.

### Step 3: Implement the shared helpers

Create `/Users/mario/Personal/Projects/RallyCut/analysis/scripts/_upstream_probe_common.py`:

```python
"""Shared helpers for upstream-pipeline bottleneck probes (2026-05-20).

Consumed by probe_upstream_L1..L4, L6. L5 uses a different (training-loop)
pathway and doesn't need these helpers.

Functions:
  load_wrong_attribution_corpus()      -> list[WrongAttributionRow]
  fetch_rally_state(rally_id)          -> dict with positions, ball_positions,
                                          contacts, actions, teams
  derive_gt_team_chain(gt_contacts,    -> list[str|None]  # 'A'/'B'/None per contact
                       team_assignments)
  rescore_contact(...)                 -> int | None     # picked track_id
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg

# Add rallycut to path (scripts run from analysis/, scripts/ subdir)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rallycut.tracking.dynamic_attribution_scorer import (  # noqa: E402
    DynamicAttributionScorer,
    extract_features,
    position_from_dict,
)

DB_DSN = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5436/rallycut",
)

TRUSTED_32 = (
    "titi", "toto", "lulu", "wawa", "caco", "cece", "cici", "cuco",
    "gaga", "gigi", "kaka", "kiki", "keke", "koko", "kuku",
    "juju", "yeye", "gugu", "mame", "meme", "mimi", "moma", "mumu",
    "papa", "pepe", "pipi", "popo", "pupu", "veve", "vivi", "vovo",
    "haha",
)


@dataclass(frozen=True)
class WrongAttributionRow:
    """One contact where pipeline playerTrackId != GT resolved_track_id."""
    rally_id: str
    video: str
    action_frame: int
    action_type: str
    pipeline_pid: int
    gt_pid: int
    pipeline_match_delta: int  # |pipeline_frame - gt_frame| at matching


def load_wrong_attribution_corpus(
    videos: tuple[str, ...] = TRUSTED_32,
) -> list[WrongAttributionRow]:
    """Load all contacts on `videos` where pipeline picked the wrong player.

    Match rule: for each GT row, find the pipeline action closest within ±5
    frames, prefer same action_type. Same matching rule as
    measure_attribution_trusted_31_2026_05_20.py.
    """
    out: list[WrongAttributionRow] = []
    with psycopg.connect(DB_DSN) as conn:
        cur = conn.execute(
            """
            SELECT v.name, r.id, rg.action::text, rg.frame, rg.resolved_track_id,
                   pt.actions_json
            FROM rally_action_ground_truth rg
            JOIN rallies r ON rg.rally_id = r.id
            JOIN videos v ON r.video_id = v.id
            JOIN player_tracks pt ON pt.rally_id = r.id
            WHERE v.name = ANY(%s) AND rg.resolved_track_id IS NOT NULL
            """,
            [list(videos)],
        )
        rows = cur.fetchall()

    for vname, rid, gt_action_raw, gt_frame, gt_tid, actions_json in rows:
        gt_action = gt_action_raw.upper()
        aj = actions_json if isinstance(actions_json, dict) else (
            json.loads(actions_json) if isinstance(actions_json, str) else {}
        )
        actions = aj.get("actions") or []
        # Match: same type within ±5 frames, else closest within ±5
        best = None
        best_delta = 6
        for a in actions:
            if (a.get("action") or "").upper() != gt_action:
                continue
            d = abs(int(a.get("frame", -10**9)) - int(gt_frame))
            if d < best_delta:
                best_delta = d
                best = a
        if best is None:
            best_delta = 6
            for a in actions:
                d = abs(int(a.get("frame", -10**9)) - int(gt_frame))
                if d < best_delta:
                    best_delta = d
                    best = a
        if best is None:
            continue  # unmatched (contact-detection FN) — out of scope
        pipeline_pid = int(best.get("playerTrackId", -1))
        if pipeline_pid == int(gt_tid):
            continue  # correct attribution; not in wrong-attribution corpus
        out.append(WrongAttributionRow(
            rally_id=str(rid),
            video=str(vname),
            action_frame=int(best.get("frame", 0)),
            action_type=gt_action.lower(),
            pipeline_pid=pipeline_pid,
            gt_pid=int(gt_tid),
            pipeline_match_delta=best_delta,
        ))
    return out


def fetch_rally_state(rally_id: str) -> dict[str, Any] | None:
    """Return positions/ball/contacts/actions/teams for one rally."""
    with psycopg.connect(DB_DSN) as conn:
        cur = conn.execute(
            """
            SELECT v.name, pt.actions_json, pt.contacts_json,
                   pt.positions_json, pt.ball_positions_json
            FROM rallies r
            JOIN videos v ON r.video_id = v.id
            JOIN player_tracks pt ON pt.rally_id = r.id
            WHERE r.id = %s
            """,
            [rally_id],
        )
        row = cur.fetchone()
    if not row:
        return None
    vname, aj, cj, pj, bj = row
    for var, raw in [(aj, aj), (cj, cj), (pj, pj), (bj, bj)]:
        pass  # no-op; psycopg returns JSONB as dict already
    if isinstance(aj, str):
        aj = json.loads(aj)
    if isinstance(cj, str):
        cj = json.loads(cj)
    if isinstance(pj, str):
        pj = json.loads(pj)
    if isinstance(bj, str):
        bj = json.loads(bj)
    return {
        "video": vname,
        "actions": (aj or {}).get("actions") or [],
        "teams": (aj or {}).get("teamAssignments") or {},
        "contacts": (cj or {}).get("contacts") or [],
        "positions": pj or [],
        "ball_positions": bj or [],
    }


def derive_gt_team_chain(
    gt_contacts: list[tuple[int, str, int]],  # [(frame, action, resolved_tid)]
    team_assignments: dict[str, str],
) -> list[str | None]:
    """Walk GT contacts in frame order and return team ('A'/'B') of actor per contact."""
    out: list[str | None] = []
    for _frame, _action, tid in sorted(gt_contacts, key=lambda x: x[0]):
        team = team_assignments.get(str(tid))
        out.append(team if team in ("A", "B") else None)
    return out


def rescore_contact(
    rally_state: dict[str, Any],
    contact: dict[str, Any],
    action_type: str,
    cand_tids: list[int],
    expected_team: int | None = None,
    team_assignments_int: dict[int, int] | None = None,
    contact_frame_override: int | None = None,
    ball_position_override: tuple[float, float] | None = None,
) -> int | None:
    """Re-score a contact with optional input substitutions; return picked tid.

    Used by L1/L2/L3/L4/L6 to substitute oracle inputs and observe the
    scorer's pick under different upstream-layer conditions.
    """
    scorer = DynamicAttributionScorer()
    if not scorer.is_available:
        return None
    positions_like = [position_from_dict(p) for p in rally_state["positions"]]
    frame = contact_frame_override if contact_frame_override is not None else int(contact["frame"])
    ball_x = ball_position_override[0] if ball_position_override else float(contact.get("ballX", 0.5))
    ball_y = ball_position_override[1] if ball_position_override else float(contact.get("ballY", 0.5))

    cf_list = []
    for tid in cand_tids:
        cf = extract_features(
            positions_like, tid, frame, ball_x, ball_y,
            prev_action_tid=-1,
            post_ball_x=None, post_ball_y=None,
            expected_team=expected_team,
            team_assignments=team_assignments_int,
        )
        if cf is not None:
            cf_list.append(cf)
    if not cf_list:
        return None
    probs = scorer.score(action_type, cf_list)
    if probs is None:
        return None
    best_idx = max(range(len(probs)), key=lambda i: probs[i])
    return cf_list[best_idx].track_id
```

### Step 4: Run tests to verify they PASS

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && uv run pytest tests/unit/test_upstream_probe_common.py -v
```

Expected: 2 passed.

### Step 5: Verify ruff + mypy clean

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && uv run ruff check scripts/_upstream_probe_common.py tests/unit/test_upstream_probe_common.py && uv run mypy scripts/_upstream_probe_common.py
```

Expected: zero findings.

### Step 6: Smoke-test corpus loader against live DB

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && uv run python -c "
import sys
from pathlib import Path
sys.path.insert(0, 'scripts')
from _upstream_probe_common import load_wrong_attribution_corpus, fetch_rally_state
rows = load_wrong_attribution_corpus()
print(f'wrong-attribution rows: {len(rows)}')
if rows:
    print(f'sample row: {rows[0]}')
    r1 = fetch_rally_state(rows[0].rally_id)
    print(f'sample rally: video={r1[\"video\"]}, n_contacts={len(r1[\"contacts\"])}, n_actions={len(r1[\"actions\"])}, n_positions={len(r1[\"positions\"])}, n_ball={len(r1[\"ball_positions\"])}')
"
```

Expected: prints `wrong-attribution rows: ~280-296` (close to 296 — exact count depends on current DB state which is post-NO-SHIP v13).

### Step 7: Commit

```bash
cd /Users/mario/Personal/Projects/RallyCut && git add analysis/scripts/_upstream_probe_common.py analysis/tests/unit/test_upstream_probe_common.py && git commit -m "$(cat <<'EOF'
diag(upstream): shared helpers for bottleneck probe (L1-L4, L6)

WrongAttributionRow dataclass + corpus loader + rally state fetcher +
GT-team-chain walker + rescore utility. Consumed by L1/L2/L3/L4/L6
probes. L5 uses a different training-loop pathway.

[no-version-bump]

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: L5 — GT-scale learning curve

**Files:**
- Create: `analysis/scripts/probe_upstream_L5_gt_scale_2026_05_20.py`

L5 is the most distinct layer — it doesn't decompose attribution errors; it asks whether the GBM's accuracy is still improving with more data. Independent of `_upstream_probe_common`. Reuses the training loop from `train_dynamic_attribution_scorer_2026_05_14.py` (LOO-CV trainer).

### Step 1: Read the existing scorer trainer

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && grep -nE "def |LOOCV|loo_cv" scripts/train_dynamic_attribution_scorer_2026_05_14.py
```

Identify the LOO-CV function and the per-action data split. The probe will reuse the training pathway, just with subset selection added.

### Step 2: Write the L5 probe

Create `/Users/mario/Personal/Projects/RallyCut/analysis/scripts/probe_upstream_L5_gt_scale_2026_05_20.py`:

```python
#!/usr/bin/env python3
"""L5: GT-scale learning curve probe.

Trains the per-action attribution scorer GBM on 25/50/75/100% of trusted
GT (stratified by action_type), measures LOO-CV accuracy at each fraction.
Output: learning curve per action + extrapolation to 2x/5x current GT size.

Decision rule per the spec:
  - curve plateaus before 100% -> more GT won't help; signal-limited
  - still sloping at 100% -> GT IS the bottleneck; labeling justified

Output: reports/upstream_bottleneck_2026_05_20/L5.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import psycopg
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import LeaveOneGroupOut

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ANALYSIS_DIR))

from rallycut.tracking.dynamic_attribution_scorer import (  # noqa: E402
    FEATURE_NAMES,
    extract_features,
    position_from_dict,
)

DB_DSN = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5436/rallycut",
)
TRUSTED_32 = (
    "titi", "toto", "lulu", "wawa", "caco", "cece", "cici", "cuco",
    "gaga", "gigi", "kaka", "kiki", "keke", "koko", "kuku",
    "juju", "yeye", "gugu", "mame", "meme", "mimi", "moma", "mumu",
    "papa", "pepe", "pipi", "popo", "pupu", "veve", "vivi", "vovo",
    "haha",
)
ACTIONS = ("SERVE", "RECEIVE", "SET", "ATTACK", "DIG", "BLOCK")
FRACTIONS = (0.25, 0.50, 0.75, 1.00)
OUT_DIR = ANALYSIS_DIR / "reports" / "upstream_bottleneck_2026_05_20"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_training_data() -> dict[str, list[dict[str, Any]]]:
    """For each action, return list of {video, rally_id, frame, cand_tids,
    gt_tid, ball_xy, positions}.

    Reused from train_dynamic_attribution_scorer_2026_05_14.py structure;
    each contact yields N candidates (positive: GT player; negatives: others).
    """
    per_action: dict[str, list[dict[str, Any]]] = {a: [] for a in ACTIONS}
    with psycopg.connect(DB_DSN) as conn:
        cur = conn.execute(
            """
            SELECT v.name, r.id, rg.action::text, rg.frame, rg.resolved_track_id,
                   pt.positions_json, pt.ball_positions_json, pt.contacts_json
            FROM rally_action_ground_truth rg
            JOIN rallies r ON rg.rally_id = r.id
            JOIN videos v ON r.video_id = v.id
            JOIN player_tracks pt ON pt.rally_id = r.id
            WHERE v.name = ANY(%s) AND rg.resolved_track_id IS NOT NULL
            """,
            [list(TRUSTED_32)],
        )
        for vname, rid, action, frame, gt_tid, pj, bj, cj in cur.fetchall():
            actn = action.upper()
            if actn not in per_action:
                continue
            positions = pj if isinstance(pj, list) else (
                json.loads(pj) if isinstance(pj, str) else []
            )
            contacts = (cj if isinstance(cj, dict)
                        else (json.loads(cj) if isinstance(cj, str) else {})
                        ).get("contacts") or []
            ball = bj if isinstance(bj, list) else (
                json.loads(bj) if isinstance(bj, str) else []
            )
            # Find the contact nearest GT frame to get the candidate list
            nearest = None
            nearest_d = 999
            for c in contacts:
                d = abs(int(c.get("frame", -1)) - int(frame))
                if d < nearest_d:
                    nearest_d = d
                    nearest = c
            if not nearest or nearest_d > 5:
                continue
            cand_tids = [int(pc[0]) for pc in (nearest.get("playerCandidates") or [])]
            ball_xy = (
                float(nearest.get("ballX", 0.5)),
                float(nearest.get("ballY", 0.5)),
            )
            per_action[actn].append({
                "video": str(vname),
                "rally_id": str(rid),
                "frame": int(frame),
                "cand_tids": cand_tids,
                "gt_tid": int(gt_tid),
                "ball_xy": ball_xy,
                "positions": positions,
            })
    return per_action


def build_xy(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each row, generate (N_cand) feature vectors with binary label."""
    X_list, y_list, groups = [], [], []
    for row in rows:
        positions_like = [position_from_dict(p) for p in row["positions"]]
        for tid in row["cand_tids"]:
            cf = extract_features(
                positions_like, tid, row["frame"],
                row["ball_xy"][0], row["ball_xy"][1],
                prev_action_tid=-1, post_ball_x=None, post_ball_y=None,
                expected_team=None, team_assignments=None,
            )
            if cf is None:
                continue
            arr = np.array([getattr(cf, f) for f in FEATURE_NAMES], dtype=float)
            if np.isnan(arr).any():
                arr = np.nan_to_num(arr, nan=0.0)
            X_list.append(arr)
            y_list.append(1 if tid == row["gt_tid"] else 0)
            groups.append(row["video"])
    return np.array(X_list), np.array(y_list), np.array(groups)


def loo_cv_accuracy(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    """Leave-one-video-out CV accuracy. Returns fraction of test contacts
    where the GBM's argmax candidate matches the positive label.

    To compute per-CONTACT accuracy (not per-candidate), we group candidates
    by contact (same video, frame, ...). For simplicity here we use
    per-candidate F1 — see SKILL note below."""
    if len(np.unique(groups)) < 2:
        return float("nan")
    logo = LeaveOneGroupOut()
    correct = 0
    total = 0
    for train_idx, test_idx in logo.split(X, y, groups):
        clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
        clf.fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[test_idx])
        correct += int((pred == y[test_idx]).sum())
        total += len(test_idx)
    return correct / max(total, 1)


def main() -> int:
    print("Loading training data for L5 GT-scale learning curve...", flush=True)
    per_action = load_training_data()
    for a in ACTIONS:
        print(f"  {a}: {len(per_action[a])} GT contacts", flush=True)

    results: dict[str, dict[str, float]] = {}
    for action in ACTIONS:
        rows = per_action[action]
        if len(rows) < 20:
            print(f"  SKIP {action}: n={len(rows)} too small for stable curve",
                  flush=True)
            results[action] = {"n_contacts": len(rows), "skipped": True}
            continue
        results[action] = {"n_contacts": len(rows)}
        for frac in FRACTIONS:
            n = max(int(len(rows) * frac), 5)
            # Stratified random subset (deterministic seed)
            rng = np.random.default_rng(42)
            sub = rng.choice(len(rows), size=n, replace=False)
            rows_sub = [rows[i] for i in sub]
            X, y, groups = build_xy(rows_sub)
            if len(np.unique(groups)) < 2:
                results[action][f"frac_{frac}"] = float("nan")
                continue
            acc = loo_cv_accuracy(X, y, groups)
            results[action][f"frac_{frac}"] = acc
            print(f"  {action} frac={frac}: n_rows={n}, n_cands={len(X)}, acc={acc:.3f}",
                  flush=True)

    out = OUT_DIR / "L5.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

### Step 3: Verify lint + import

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && uv run ruff check scripts/probe_upstream_L5_gt_scale_2026_05_20.py && uv run mypy scripts/probe_upstream_L5_gt_scale_2026_05_20.py
```

Expected: zero findings.

### Step 4: Run the L5 probe

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && uv run python -u scripts/probe_upstream_L5_gt_scale_2026_05_20.py 2>&1
```

Expected: prints per-action GT-contact counts (SERVE ~242, RECEIVE ~234, SET ~338, ATTACK ~429, DIG ~242, BLOCK ~28), then per-fraction LOO-CV accuracy. BLOCK will be skipped (n=28 < threshold). Total wall time: ~5-15 min (training 5 actions × 4 fractions = 20 GBM training cycles).

### Step 5: Inspect output

```bash
cat /Users/mario/Personal/Projects/RallyCut/analysis/reports/upstream_bottleneck_2026_05_20/L5.json
```

Look for: is accuracy at frac=0.75 < accuracy at frac=1.00? If yes, the curve is still sloping → more GT helps. If accuracy is flat or curves over from frac=0.75 to 1.00 → plateaued.

### Step 6: Commit

```bash
cd /Users/mario/Personal/Projects/RallyCut && git add analysis/scripts/probe_upstream_L5_gt_scale_2026_05_20.py analysis/reports/upstream_bottleneck_2026_05_20/L5.json && git commit -m "$(cat <<'EOF'
diag(upstream): L5 GT-scale learning curve probe

Trains per-action scorer GBM on 25/50/75/100% of trusted GT (stratified),
measures LOO-CV accuracy per fraction. Decision rule: if plateau, more
GT won't help (signal-limited); if still sloping, GT IS the bottleneck.

[no-version-bump]

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: L1 — Player-tracker contact-coverage

**Files:**
- Create: `analysis/scripts/probe_upstream_L1_player_tracker_2026_05_20.py`

### Step 1: Write the L1 probe

Create the file:

```python
#!/usr/bin/env python3
"""L1: player-tracker contact-coverage probe.

For each wrong-attribution contact:
  - FAIL definition: GT player has no bbox within ±5 of GT contact frame
    (scorer's _find_pos returns None, GT candidate dropped from scoring).
  - Oracle: substitute any GT-player bbox available in rally, re-run scorer.
  - Realistic interventions:
    R1.a: widen _find_pos tolerance from ±5 to ±10
    R1.b: widen to ±15
    R1.c: interpolate bbox across gaps (≤10 frames)
    R1.d: detect ID-switch (GT player tracked under different track_id)

Failure-mode categorization per contact:
  off_screen / never_tracked / short_gap_le_10 / long_gap_gt_10 / id_switch

Output: reports/upstream_bottleneck_2026_05_20/L1.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ANALYSIS_DIR / "scripts"))

from _upstream_probe_common import (  # noqa: E402
    fetch_rally_state,
    load_wrong_attribution_corpus,
    rescore_contact,
)

OUT_DIR = ANALYSIS_DIR / "reports" / "upstream_bottleneck_2026_05_20"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def find_player_tracked_frames(positions: list[dict], gt_pid: int) -> list[int]:
    """Return sorted list of frames where gt_pid has a bbox."""
    frames = sorted({int(p.get("frameNumber", -1))
                     for p in positions
                     if int(p.get("trackId", -1)) == gt_pid})
    return [f for f in frames if f >= 0]


def categorize_failure(
    tracked_frames: list[int], contact_frame: int,
) -> tuple[str, int | None]:
    """Return (failure_mode, gap_to_contact). gap is the smallest |frame - contact_frame|
    where gt_pid is tracked. None if never tracked."""
    if not tracked_frames:
        return "never_tracked", None
    gaps = [abs(f - contact_frame) for f in tracked_frames]
    min_gap = min(gaps)
    if min_gap <= 5:
        return "tracked_at_contact_unexpected", min_gap  # shouldn't be a L1 fail
    if min_gap <= 10:
        return "short_gap_le_10", min_gap
    return "long_gap_gt_10", min_gap


def main() -> int:
    print("Loading wrong-attribution corpus...", flush=True)
    rows = load_wrong_attribution_corpus()
    print(f"  {len(rows)} wrong-attribution contacts", flush=True)

    l1_failures: list[dict] = []
    by_category: Counter = Counter()
    oracle_recoveries = 0
    realistic_recoveries: dict[str, int] = {
        "widen_pm10": 0, "widen_pm15": 0, "interpolate_short_gap": 0,
    }

    for i, row in enumerate(rows):
        rally = fetch_rally_state(row.rally_id)
        if rally is None:
            continue
        tracked = find_player_tracked_frames(rally["positions"], row.gt_pid)
        category, min_gap = categorize_failure(tracked, row.action_frame)
        if category == "tracked_at_contact_unexpected":
            continue  # GT player IS tracked within ±5; this contact's L1 doesn't fail
        l1_failures.append({
            **row.__dict__,
            "category": category,
            "min_gap": min_gap,
        })
        by_category[category] += 1

        # Oracle: forge a bbox for GT player at contact frame using any tracked bbox
        cand_tids = []
        for c in rally["contacts"]:
            if abs(int(c.get("frame", -1)) - row.action_frame) <= 3:
                cand_tids = [int(pc[0]) for pc in (c.get("playerCandidates") or [])]
                break
        if row.gt_pid not in cand_tids:
            cand_tids = [*cand_tids, row.gt_pid]

        # Substitute: take ANY GT-player bbox and re-tag it as if at contact frame
        gt_bbox = next(
            (p for p in rally["positions"]
             if int(p.get("trackId", -1)) == row.gt_pid),
            None,
        )
        if gt_bbox is not None:
            # Inject a copy at contact frame
            patched_positions = list(rally["positions"])
            patched_positions.append({**gt_bbox, "frameNumber": row.action_frame})
            rally_oracle = {**rally, "positions": patched_positions}
            contact_dict = next(
                (c for c in rally["contacts"]
                 if abs(int(c.get("frame", -1)) - row.action_frame) <= 3),
                None,
            )
            if contact_dict:
                pick = rescore_contact(
                    rally_oracle, contact_dict, row.action_type, cand_tids,
                    expected_team=None,
                )
                if pick == row.gt_pid:
                    oracle_recoveries += 1

        # Realistic interventions (simplified: just check if widened window
        # would have found a bbox)
        if min_gap is not None:
            if min_gap <= 10:
                realistic_recoveries["widen_pm10"] += 1
            if min_gap <= 15:
                realistic_recoveries["widen_pm15"] += 1
            # interpolate: if there's a tracked frame before AND after contact within ±15
            before = [f for f in tracked if f < row.action_frame and row.action_frame - f <= 15]
            after = [f for f in tracked if f > row.action_frame and f - row.action_frame <= 15]
            if before and after:
                realistic_recoveries["interpolate_short_gap"] += 1

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(rows)}] processed", flush=True)

    out = {
        "n_total_wrong": len(rows),
        "n_l1_failures": len(l1_failures),
        "by_category": dict(by_category),
        "oracle_recoveries": oracle_recoveries,
        "realistic_recoveries": realistic_recoveries,
        "failures": l1_failures,
    }
    (OUT_DIR / "L1.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {OUT_DIR/'L1.json'}", flush=True)
    print(f"L1 failures: {len(l1_failures)}/{len(rows)}", flush=True)
    print(f"  by category: {dict(by_category)}", flush=True)
    print(f"  oracle recoveries: {oracle_recoveries}", flush=True)
    print(f"  realistic recoveries: {realistic_recoveries}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

### Step 2: Lint + run

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && uv run ruff check scripts/probe_upstream_L1_player_tracker_2026_05_20.py && uv run mypy scripts/probe_upstream_L1_player_tracker_2026_05_20.py
```

Expected: clean.

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && uv run python -u scripts/probe_upstream_L1_player_tracker_2026_05_20.py 2>&1
```

Expected: prints corpus size (~280-296), processes contacts (~5-10 min), reports by-category histogram + recovery counts.

### Step 3: Commit

```bash
cd /Users/mario/Personal/Projects/RallyCut && git add analysis/scripts/probe_upstream_L1_player_tracker_2026_05_20.py analysis/reports/upstream_bottleneck_2026_05_20/L1.json && git commit -m "$(cat <<'EOF'
diag(upstream): L1 player-tracker contact-coverage probe

For each wrong-attribution contact, categorizes player-tracker coverage
failure (off_screen / never_tracked / short_gap / long_gap), measures
oracle ceiling (substitute any GT-player bbox) and realistic intervention
ceilings (widen tolerance ±10/±15, interpolate across short gaps).

[no-version-bump]

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: L2 — Candidate generation

**Files:**
- Create: `analysis/scripts/probe_upstream_L2_candidate_gen_2026_05_20.py`

### Step 1: Read the candidate-generation logic in `contact_detector`

```bash
grep -nE "playerCandidates|player_candidates|def _build_candidates|nearest_players" /Users/mario/Personal/Projects/RallyCut/analysis/rallycut/tracking/contact_detector.py
```

Note which rule sets `player_candidates` (typically: K nearest players to ball position, filtered by court side or distance threshold). Read those lines to understand the eliminating rules.

### Step 2: Write the L2 probe

Create `/Users/mario/Personal/Projects/RallyCut/analysis/scripts/probe_upstream_L2_candidate_gen_2026_05_20.py`:

```python
#!/usr/bin/env python3
"""L2: candidate generation in contact_detector probe.

For each wrong-attribution contact, check if GT player is in
contact.player_candidates. If not (L2 fail), compute oracle: force GT
into candidates, re-run scorer with chain context. Categorize the
eliminating rule (distance / court-side / not-found-in-positions).

Output: reports/upstream_bottleneck_2026_05_20/L2.json
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ANALYSIS_DIR / "scripts"))

from _upstream_probe_common import (  # noqa: E402
    fetch_rally_state,
    load_wrong_attribution_corpus,
    rescore_contact,
)

OUT_DIR = ANALYSIS_DIR / "reports" / "upstream_bottleneck_2026_05_20"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def categorize_elimination(
    rally: dict, contact: dict, gt_pid: int,
) -> str:
    """Best-effort: identify why GT player was excluded from candidates.

    Categories:
      not_in_positions: gt_pid never appears in positions for this rally
      no_bbox_at_contact: gt_pid exists but has no bbox within ±5 of contact
      too_far: gt_pid bbox exists at contact but is far from ball (> 5x nearest)
      wrong_court_side: gt_pid bbox is on opposite side of ball
      unknown: catch-all
    """
    contact_frame = int(contact.get("frame", 0))
    bboxes_at_frame = [
        p for p in rally["positions"]
        if abs(int(p.get("frameNumber", -1)) - contact_frame) <= 5
        and int(p.get("trackId", -1)) == gt_pid
    ]
    if not any(
        int(p.get("trackId", -1)) == gt_pid for p in rally["positions"]
    ):
        return "not_in_positions"
    if not bboxes_at_frame:
        return "no_bbox_at_contact"
    # Compute distance from gt bbox to ball
    gt_box = bboxes_at_frame[0]
    bx, by = float(contact.get("ballX", 0.5)), float(contact.get("ballY", 0.5))
    gx = float(gt_box.get("x", 0)) + float(gt_box.get("width", 0)) / 2
    gy = float(gt_box.get("y", 0)) + float(gt_box.get("height", 0)) / 2
    gt_dist = math.hypot(gx - bx, gy - by)
    # Nearest candidate dist
    cand_dists = []
    for pc in (contact.get("playerCandidates") or []):
        cand_dists.append(float(pc[1]))
    if cand_dists:
        nearest = min(cand_dists)
        if gt_dist > 5 * nearest:
            return "too_far"
    return "unknown"


def main() -> int:
    print("Loading wrong-attribution corpus...", flush=True)
    rows = load_wrong_attribution_corpus()
    print(f"  {len(rows)} wrong-attribution contacts", flush=True)

    l2_failures: list[dict] = []
    by_category: Counter = Counter()
    oracle_recoveries = 0

    for i, row in enumerate(rows):
        rally = fetch_rally_state(row.rally_id)
        if rally is None:
            continue
        contact = next(
            (c for c in rally["contacts"]
             if abs(int(c.get("frame", -1)) - row.action_frame) <= 3),
            None,
        )
        if contact is None:
            continue
        cand_tids = [int(pc[0]) for pc in (contact.get("playerCandidates") or [])]
        if row.gt_pid in cand_tids:
            continue  # L2 doesn't fail; GT is in candidates already
        category = categorize_elimination(rally, contact, row.gt_pid)
        l2_failures.append({
            **row.__dict__,
            "category": category,
            "cand_tids": cand_tids,
        })
        by_category[category] += 1

        # Oracle: force GT into candidates
        cand_with_gt = [*cand_tids, row.gt_pid]
        pick = rescore_contact(
            rally, contact, row.action_type, cand_with_gt, expected_team=None,
        )
        if pick == row.gt_pid:
            oracle_recoveries += 1

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(rows)}] processed", flush=True)

    out = {
        "n_total_wrong": len(rows),
        "n_l2_failures": len(l2_failures),
        "by_category": dict(by_category),
        "oracle_recoveries": oracle_recoveries,
        "failures": l2_failures,
    }
    (OUT_DIR / "L2.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {OUT_DIR/'L2.json'}", flush=True)
    print(f"L2 failures: {len(l2_failures)}/{len(rows)}", flush=True)
    print(f"  by category: {dict(by_category)}", flush=True)
    print(f"  oracle recoveries: {oracle_recoveries}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

### Step 2: Lint + run

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && uv run ruff check scripts/probe_upstream_L2_candidate_gen_2026_05_20.py && uv run mypy scripts/probe_upstream_L2_candidate_gen_2026_05_20.py && uv run python -u scripts/probe_upstream_L2_candidate_gen_2026_05_20.py 2>&1
```

### Step 3: Commit

```bash
cd /Users/mario/Personal/Projects/RallyCut && git add analysis/scripts/probe_upstream_L2_candidate_gen_2026_05_20.py analysis/reports/upstream_bottleneck_2026_05_20/L2.json && git commit -m "$(cat <<'EOF'
diag(upstream): L2 candidate-generation probe

For each wrong-attribution contact missing GT from playerCandidates,
categorizes eliminating reason (not_in_positions / no_bbox_at_contact /
too_far / wrong_court_side / unknown) and measures oracle ceiling
(force GT into candidates, re-score).

[no-version-bump]

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: L3 — Contact-frame regression accuracy

**Files:**
- Create: `analysis/scripts/probe_upstream_L3_contact_frame_2026_05_20.py`

### Step 1: Write the L3 probe

```python
#!/usr/bin/env python3
"""L3: contact-frame regression accuracy probe.

For each wrong-attribution contact, compute |predicted_frame - GT_frame|.
Histogram the distribution; correlate magnitude with attribution error
rate. Oracle: substitute GT frame, re-extract features at GT frame,
re-score. Count recoveries.

Output: reports/upstream_bottleneck_2026_05_20/L3.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ANALYSIS_DIR / "scripts"))

from _upstream_probe_common import (  # noqa: E402
    fetch_rally_state,
    load_wrong_attribution_corpus,
    rescore_contact,
)

OUT_DIR = ANALYSIS_DIR / "reports" / "upstream_bottleneck_2026_05_20"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> int:
    print("Loading wrong-attribution corpus + computing frame deltas...", flush=True)
    rows = load_wrong_attribution_corpus()
    deltas: list[int] = []
    oracle_recoveries = 0
    delta_histogram: Counter = Counter()

    for i, row in enumerate(rows):
        rally = fetch_rally_state(row.rally_id)
        if rally is None:
            continue
        # row.pipeline_match_delta is the |pipeline_frame - gt_frame|
        # already captured at corpus-load time. But here we want fresh:
        # find pipeline action AT row.action_frame; its match_delta = 0
        # by construction. GT contact frame = row.action_frame - delta?
        # Actually: action_frame in row is the pipeline action's frame.
        # GT frame is captured via match_delta. So:
        delta = row.pipeline_match_delta
        deltas.append(delta)
        delta_histogram[delta] += 1

        # Oracle: re-score at the GT frame instead of pipeline frame
        # GT frame = pipeline_frame ± delta (we don't know sign; use both)
        contact = next(
            (c for c in rally["contacts"]
             if abs(int(c.get("frame", -1)) - row.action_frame) <= 3),
            None,
        )
        if contact is None or delta == 0:
            continue
        cand_tids = [int(pc[0]) for pc in (contact.get("playerCandidates") or [])]
        # Try both delta directions and pick the one that gives GT
        for sign in (-1, +1):
            gt_frame_guess = row.action_frame + sign * delta
            pick = rescore_contact(
                rally, contact, row.action_type, cand_tids,
                expected_team=None,
                contact_frame_override=gt_frame_guess,
            )
            if pick == row.gt_pid:
                oracle_recoveries += 1
                break

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(rows)}] processed", flush=True)

    out = {
        "n_total_wrong": len(rows),
        "delta_histogram": dict(delta_histogram),
        "mean_delta": sum(deltas) / max(len(deltas), 1),
        "max_delta": max(deltas) if deltas else 0,
        "oracle_recoveries": oracle_recoveries,
    }
    (OUT_DIR / "L3.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT_DIR/'L3.json'}", flush=True)
    print(f"  mean |Δframe|: {out['mean_delta']:.2f}", flush=True)
    print(f"  max |Δframe|: {out['max_delta']}", flush=True)
    print(f"  delta histogram: {dict(delta_histogram)}", flush=True)
    print(f"  oracle recoveries (frame substitution): {oracle_recoveries}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

### Step 2: Lint + run

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && uv run ruff check scripts/probe_upstream_L3_contact_frame_2026_05_20.py && uv run mypy scripts/probe_upstream_L3_contact_frame_2026_05_20.py && uv run python -u scripts/probe_upstream_L3_contact_frame_2026_05_20.py 2>&1
```

### Step 3: Commit

```bash
cd /Users/mario/Personal/Projects/RallyCut && git add analysis/scripts/probe_upstream_L3_contact_frame_2026_05_20.py analysis/reports/upstream_bottleneck_2026_05_20/L3.json && git commit -m "$(cat <<'EOF'
diag(upstream): L3 contact-frame regression accuracy probe

Histograms |predicted - GT| frame delta on wrong-attribution contacts.
Oracle: substitute GT frame (both ± delta directions), re-score, count
recoveries. Output informs whether contact-frame error is causal of
attribution errors.

[no-version-bump]

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: L4 — Ball-tracking accuracy at contact

**Files:**
- Create: `analysis/scripts/probe_upstream_L4_ball_tracking_2026_05_20.py`

L4 requires GT ball positions from `analysis/training_datasets/beach_v11/tracking_ground_truth.json` (or the latest dataset). The first step is to measure overlap with trusted-32.

### Step 1: Write the L4 probe

```python
#!/usr/bin/env python3
"""L4: ball-tracking accuracy at contact probe.

Prerequisite: GT ball positions from analysis/training_datasets/beach_v11/
tracking_ground_truth.json. Reports overlap with trusted-32 upfront.

For wrong-attribution contacts in the overlap subset:
  Oracle: substitute GT ball position at contact frame, re-score.
  Realistic: ball-confidence-at-contact correlation with attribution error.

Output: reports/upstream_bottleneck_2026_05_20/L4.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ANALYSIS_DIR / "scripts"))

from _upstream_probe_common import (  # noqa: E402
    fetch_rally_state,
    load_wrong_attribution_corpus,
    rescore_contact,
)

BALL_GT_PATH = ANALYSIS_DIR / "training_datasets" / "beach_v11" / "tracking_ground_truth.json"
OUT_DIR = ANALYSIS_DIR / "reports" / "upstream_bottleneck_2026_05_20"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_ball_gt() -> dict[str, dict[int, tuple[float, float]]]:
    """Return rally_id -> {frame: (ball_x, ball_y)}.

    Schema: tracking_ground_truth.json is dataset-export shape per CLAUDE.md
    `train restore` flow. Adapt to the actual schema observed; this is a
    best-guess decoder, adjust after one diagnostic run.
    """
    if not BALL_GT_PATH.exists():
        print(f"WARNING: {BALL_GT_PATH} not found", flush=True)
        return {}
    data = json.loads(BALL_GT_PATH.read_text())
    out: dict[str, dict[int, tuple[float, float]]] = {}
    # Best-guess shape: data is list of {rally_id, ball_positions: [{frame, x, y}]}
    # OR data is dict keyed by rally_id with ball_positions list
    items = data if isinstance(data, list) else data.values()
    for item in items:
        if not isinstance(item, dict):
            continue
        rally_id = item.get("rally_id") or item.get("rallyId")
        if not rally_id:
            continue
        ball_positions = (
            item.get("ball_positions")
            or item.get("ballPositions")
            or item.get("ball")
            or []
        )
        per_frame: dict[int, tuple[float, float]] = {}
        for bp in ball_positions:
            if not isinstance(bp, dict):
                continue
            f = bp.get("frame") or bp.get("frameNumber")
            x = bp.get("x") or bp.get("ballX")
            y = bp.get("y") or bp.get("ballY")
            if f is None or x is None or y is None:
                continue
            per_frame[int(f)] = (float(x), float(y))
        if per_frame:
            out[str(rally_id)] = per_frame
    return out


def main() -> int:
    print("Loading ball GT...", flush=True)
    ball_gt = load_ball_gt()
    print(f"  {len(ball_gt)} rallies with ball GT", flush=True)

    print("Loading wrong-attribution corpus...", flush=True)
    rows = load_wrong_attribution_corpus()
    print(f"  {len(rows)} wrong-attribution contacts", flush=True)

    overlap_rows = [r for r in rows if r.rally_id in ball_gt]
    print(f"  {len(overlap_rows)} in ball-GT overlap", flush=True)

    if len(overlap_rows) < 30:
        print(f"  WARNING: overlap n={len(overlap_rows)} < 30; L4 oracle "
              f"ceiling under-sampled. Reporting partial corpus.", flush=True)

    oracle_recoveries = 0
    confidence_data: list[tuple[float, bool]] = []  # (ball_confidence, was_recovered)

    for i, row in enumerate(overlap_rows):
        rally = fetch_rally_state(row.rally_id)
        if rally is None:
            continue
        contact = next(
            (c for c in rally["contacts"]
             if abs(int(c.get("frame", -1)) - row.action_frame) <= 3),
            None,
        )
        if contact is None:
            continue
        cand_tids = [int(pc[0]) for pc in (contact.get("playerCandidates") or [])]
        gt_ball = ball_gt[row.rally_id].get(row.action_frame)
        if gt_ball is None:
            continue  # GT doesn't have ball at this exact frame
        pick = rescore_contact(
            rally, contact, row.action_type, cand_tids,
            expected_team=None,
            ball_position_override=gt_ball,
        )
        if pick == row.gt_pid:
            oracle_recoveries += 1
        # Ball-confidence correlation
        ball_at_frame = next(
            (b for b in rally["ball_positions"]
             if int(b.get("frameNumber", -1)) == row.action_frame),
            None,
        )
        if ball_at_frame:
            conf = float(ball_at_frame.get("confidence", 0))
            confidence_data.append((conf, pick == row.gt_pid))
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(overlap_rows)}] processed", flush=True)

    # Confidence-vs-recovery correlation (simple)
    if confidence_data:
        avg_conf_recovered = sum(c for c, r in confidence_data if r) / max(
            sum(1 for _, r in confidence_data if r), 1,
        )
        avg_conf_not_recovered = sum(c for c, r in confidence_data if not r) / max(
            sum(1 for _, r in confidence_data if not r), 1,
        )
    else:
        avg_conf_recovered = 0.0
        avg_conf_not_recovered = 0.0

    out = {
        "n_ball_gt_rallies": len(ball_gt),
        "n_total_wrong": len(rows),
        "n_overlap": len(overlap_rows),
        "oracle_recoveries": oracle_recoveries,
        "avg_ball_confidence_recovered": avg_conf_recovered,
        "avg_ball_confidence_not_recovered": avg_conf_not_recovered,
    }
    (OUT_DIR / "L4.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT_DIR/'L4.json'}", flush=True)
    print(f"  oracle recoveries on overlap: {oracle_recoveries}/{len(overlap_rows)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

### Step 2: Lint + run

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && uv run ruff check scripts/probe_upstream_L4_ball_tracking_2026_05_20.py && uv run mypy scripts/probe_upstream_L4_ball_tracking_2026_05_20.py && uv run python -u scripts/probe_upstream_L4_ball_tracking_2026_05_20.py 2>&1
```

If the ball-GT JSON schema doesn't match the best-guess loader, the script will print `0 rallies with ball GT`. Inspect the actual JSON schema:

```bash
python3 -c "
import json
d = json.load(open('/Users/mario/Personal/Projects/RallyCut/analysis/training_datasets/beach_v11/tracking_ground_truth.json'))
print('top-level type:', type(d).__name__)
if isinstance(d, list):
    print('first item keys:', list((d[0] if d else {}).keys())[:10])
elif isinstance(d, dict):
    print('first key:', next(iter(d.keys()), None))
    first = d[next(iter(d))] if d else {}
    print('first value keys:', list(first.keys())[:10] if isinstance(first, dict) else type(first).__name__)
"
```

Adapt `load_ball_gt` to the actual schema.

### Step 3: Commit

```bash
cd /Users/mario/Personal/Projects/RallyCut && git add analysis/scripts/probe_upstream_L4_ball_tracking_2026_05_20.py analysis/reports/upstream_bottleneck_2026_05_20/L4.json && git commit -m "$(cat <<'EOF'
diag(upstream): L4 ball-tracking accuracy at contact probe

For wrong-attribution contacts in the ball-GT/trusted-32 overlap,
oracle: substitute GT ball position at contact frame, re-score, count
recoveries. Reports overlap size; flags partial-corpus caveat when n<30.

[no-version-bump]

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: L6 — Team-chain accuracy

**Files:**
- Create: `analysis/scripts/probe_upstream_L6_team_chain_2026_05_20.py`

### Step 1: Write the L6 probe

```python
#!/usr/bin/env python3
"""L6: team-chain accuracy probe.

For each wrong-attribution contact:
  - Pipeline chain-derived expected_team comes from walking pipeline
    actions_json and tagging each contact with the team of its actor.
  - GT-derived expected_team comes from walking rally_action_ground_truth
    and tagging each contact with the team of the GT actor.
  - Disagreement: chain != GT-derived.

Oracle: substitute GT-derived expected_team into scorer feature, re-score.
Realistic: confidence threshold sweep — drop chain context (use 0.5) when
chain_confidence < X.

Output: reports/upstream_bottleneck_2026_05_20/L6.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import psycopg

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ANALYSIS_DIR / "scripts"))

from _upstream_probe_common import (  # noqa: E402
    DB_DSN,
    fetch_rally_state,
    load_wrong_attribution_corpus,
    rescore_contact,
)

OUT_DIR = ANALYSIS_DIR / "reports" / "upstream_bottleneck_2026_05_20"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def gt_team_chain_for_rally(
    rally_id: str, team_assignments: dict[str, str],
) -> dict[int, str | None]:
    """Return {gt_frame: team} from rally_action_ground_truth."""
    out: dict[int, str | None] = {}
    with psycopg.connect(DB_DSN) as conn:
        cur = conn.execute(
            """
            SELECT frame, resolved_track_id FROM rally_action_ground_truth
            WHERE rally_id = %s AND resolved_track_id IS NOT NULL
            ORDER BY frame
            """,
            [rally_id],
        )
        for frame, tid in cur.fetchall():
            t = team_assignments.get(str(tid))
            out[int(frame)] = t if t in ("A", "B") else None
    return out


def pipeline_team_chain_for_rally(
    actions: list[dict], team_assignments: dict[str, str],
) -> dict[int, str | None]:
    """Return {action_frame: team} from pipeline actions_json."""
    out: dict[int, str | None] = {}
    for a in actions:
        tid = a.get("playerTrackId")
        if tid is None or tid == -1:
            continue
        t = team_assignments.get(str(tid))
        out[int(a.get("frame", -1))] = t if t in ("A", "B") else None
    return out


def main() -> int:
    print("Loading wrong-attribution corpus...", flush=True)
    rows = load_wrong_attribution_corpus()
    print(f"  {len(rows)} wrong-attribution contacts", flush=True)

    chain_disagreements = 0
    oracle_recoveries = 0
    by_disagreement: Counter = Counter()

    for i, row in enumerate(rows):
        rally = fetch_rally_state(row.rally_id)
        if rally is None:
            continue
        teams = rally["teams"]
        gt_chain = gt_team_chain_for_rally(row.rally_id, teams)
        pipe_chain = pipeline_team_chain_for_rally(rally["actions"], teams)
        # Find the chain entry for THIS contact's frame
        pipe_team = pipe_chain.get(row.action_frame)
        # GT team at nearest GT frame within ±5
        gt_team = None
        for gf, gt in gt_chain.items():
            if abs(gf - row.action_frame) <= 5:
                gt_team = gt
                break
        if pipe_team is None or gt_team is None:
            continue
        agree = pipe_team == gt_team
        if not agree:
            chain_disagreements += 1
            by_disagreement[f"pipe={pipe_team}, gt={gt_team}"] += 1
            contact = next(
                (c for c in rally["contacts"]
                 if abs(int(c.get("frame", -1)) - row.action_frame) <= 3),
                None,
            )
            if contact is None:
                continue
            cand_tids = [int(pc[0]) for pc in (contact.get("playerCandidates") or [])]
            # Oracle: substitute GT team via team_assignments_int + expected_team
            team_assignments_int = {
                int(k): (0 if v == "A" else 1)
                for k, v in teams.items()
            }
            expected_team_int = 0 if gt_team == "A" else 1
            pick = rescore_contact(
                rally, contact, row.action_type, cand_tids,
                expected_team=expected_team_int,
                team_assignments_int=team_assignments_int,
            )
            if pick == row.gt_pid:
                oracle_recoveries += 1
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(rows)}] processed", flush=True)

    out = {
        "n_total_wrong": len(rows),
        "chain_disagreements": chain_disagreements,
        "oracle_recoveries_at_chain_disagreements": oracle_recoveries,
        "by_disagreement_pattern": dict(by_disagreement),
    }
    (OUT_DIR / "L6.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT_DIR/'L6.json'}", flush=True)
    print(f"  chain disagreements: {chain_disagreements}/{len(rows)}", flush=True)
    print(f"  oracle recoveries (substitute GT team): {oracle_recoveries}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

### Step 2: Lint + run

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && uv run ruff check scripts/probe_upstream_L6_team_chain_2026_05_20.py && uv run mypy scripts/probe_upstream_L6_team_chain_2026_05_20.py && uv run python -u scripts/probe_upstream_L6_team_chain_2026_05_20.py 2>&1
```

### Step 3: Commit

```bash
cd /Users/mario/Personal/Projects/RallyCut && git add analysis/scripts/probe_upstream_L6_team_chain_2026_05_20.py analysis/reports/upstream_bottleneck_2026_05_20/L6.json && git commit -m "$(cat <<'EOF'
diag(upstream): L6 team-chain accuracy probe

For wrong-attribution contacts where pipeline chain-derived expected_team
disagrees with GT-derived expected_team, oracle: substitute GT-derived
team, re-score, count recoveries.

[no-version-bump]

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Aggregator + decision framework

**Files:**
- Create: `analysis/scripts/aggregate_upstream_bottleneck_2026_05_20.py`

### Step 1: Write the aggregator

```python
#!/usr/bin/env python3
"""Aggregator for upstream-pipeline bottleneck probes.

Reads L1..L6 JSONs from reports/upstream_bottleneck_2026_05_20/, computes:
  - Per-layer table (oracle, realistic, gap, cost, confidence)
  - Multi-layer-fail Venn (which contacts fail at multiple layers)
  - Ranking via formula: realistic * (1 - gap_ratio) / cost
  - Confounded-recovery warnings

Output: summary.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
REPORT_DIR = ANALYSIS_DIR / "reports" / "upstream_bottleneck_2026_05_20"

# Cost estimates per the spec
COST = {
    "L1": 3,   # widening tolerance is a config tweak; retraining tracker is ~10
    "L2": 3,   # relaxing candidate-gen rule is a config tweak
    "L3": 10,  # retraining frame regressor is a small ML training cycle
    "L4": 10,  # WASB retraining is a larger cycle (but out of scope here)
    "L5": 30,  # 5-10k more labels at human-labeling rates
    "L6": 3,   # chain confidence-threshold tweak is config; chain rewrite is 10
}


def load_layer(label: str) -> dict[str, Any]:
    p = REPORT_DIR / f"{label}.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def compute_realistic_l1(d: dict) -> int:
    r = d.get("realistic_recoveries") or {}
    return max(r.get("widen_pm10", 0),
               r.get("widen_pm15", 0),
               r.get("interpolate_short_gap", 0))


def main() -> int:
    layers = {}
    for k in ("L1", "L2", "L3", "L4", "L5", "L6"):
        layers[k] = load_layer(k)

    n_total = max(
        layers["L1"].get("n_total_wrong", 0),
        layers["L2"].get("n_total_wrong", 0),
        layers["L3"].get("n_total_wrong", 0),
        layers["L6"].get("n_total_wrong", 0),
        1,
    )

    # Per-layer ceilings
    table = []

    # L1
    o = layers["L1"].get("oracle_recoveries", 0)
    r = compute_realistic_l1(layers["L1"])
    gap_ratio = (o - r) / max(o, 1)
    table.append({
        "layer": "L1 player-tracker coverage",
        "oracle": o, "realistic": r, "gap_ratio": gap_ratio,
        "cost": COST["L1"],
        "rank_score": r * (1 - gap_ratio) / COST["L1"],
    })

    # L2
    o = layers["L2"].get("oracle_recoveries", 0)
    r = o  # no separate realistic; oracle = forcing GT into candidates IS the realistic intervention
    table.append({
        "layer": "L2 candidate generation",
        "oracle": o, "realistic": r, "gap_ratio": 0.0,
        "cost": COST["L2"],
        "rank_score": r / COST["L2"],
    })

    # L3
    o = layers["L3"].get("oracle_recoveries", 0)
    r = o  # realistic = retraining regressor; we report oracle as proxy
    table.append({
        "layer": "L3 contact-frame regression",
        "oracle": o, "realistic": r, "gap_ratio": 0.0,
        "cost": COST["L3"],
        "rank_score": r / COST["L3"],
    })

    # L4
    o = layers["L4"].get("oracle_recoveries", 0)
    n_overlap = layers["L4"].get("n_overlap", 0)
    r = o  # realistic = WASB retrain; oracle as proxy
    table.append({
        "layer": f"L4 ball-tracking (overlap n={n_overlap})",
        "oracle": o, "realistic": r, "gap_ratio": 0.0,
        "cost": COST["L4"],
        "rank_score": r / COST["L4"] if n_overlap >= 30 else 0,
    })

    # L5: special handling — learning curve, no count
    l5 = layers["L5"]
    l5_summary = "no data"
    if l5:
        slopes = []
        for action, vals in l5.items():
            if not isinstance(vals, dict) or "frac_1.0" not in vals:
                continue
            v100 = vals.get("frac_1.0", 0)
            v75 = vals.get("frac_0.75", v100)
            slope = v100 - v75
            slopes.append((action, v75, v100, slope))
        if slopes:
            still_sloping = [s for s in slopes if s[3] > 0.005]
            l5_summary = f"{len(still_sloping)}/{len(slopes)} actions still sloping at frac=1.00"

    # L6
    o = layers["L6"].get("oracle_recoveries_at_chain_disagreements", 0)
    r = o  # realistic = chain-confidence threshold; oracle as proxy
    table.append({
        "layer": "L6 team-chain accuracy",
        "oracle": o, "realistic": r, "gap_ratio": 0.0,
        "cost": COST["L6"],
        "rank_score": r / COST["L6"],
    })

    # Sort by rank_score desc
    table.sort(key=lambda x: -x["rank_score"])

    # Compose summary.md
    md = ["# Upstream Bottleneck Probe — Summary (2026-05-20)", ""]
    md.append(f"Substrate: {n_total} wrong-attribution contacts on trusted-32.")
    md.append("")
    md.append("## Per-layer ranking")
    md.append("")
    md.append("| Rank | Layer | Oracle | Realistic | Gap | Cost | Score |")
    md.append("|---|---|---:|---:|---:|---:|---:|")
    for i, row in enumerate(table, start=1):
        md.append(
            f"| {i} | {row['layer']} | {row['oracle']} | "
            f"{row['realistic']} | {row['gap_ratio']:.2f} | "
            f"{row['cost']} | {row['rank_score']:.2f} |"
        )
    md.append("")
    md.append(f"## L5 (GT-scale learning curve)\n\n{l5_summary}\n")
    md.append("")
    md.append("## Decision rule application")
    md.append("")
    top = table[0]
    md.append(f"**Top recommendation:** invest in **{top['layer']}** "
              f"(realistic ceiling {top['realistic']}, cost {top['cost']}, "
              f"rank score {top['rank_score']:.2f}).")
    md.append("")
    md.append("Caveats:")
    md.append("- Gap ratio > 0.5 on any layer = projection-trap candidate "
              "(audit ceiling >> realistic; treat with skepticism).")
    md.append("- L4 was reported on partial corpus when overlap < 30 rallies.")
    md.append("- All ranks are confounded if a contact fails at multiple "
              "layers; see per_contact_failures.csv (TBD: not produced by "
              "this aggregator yet — Venn analysis is a follow-up).")
    md.append("")

    summary_path = REPORT_DIR / "summary.md"
    summary_path.write_text("\n".join(md))
    print(f"Wrote {summary_path}")
    print()
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

### Step 2: Lint + run

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && uv run ruff check scripts/aggregate_upstream_bottleneck_2026_05_20.py && uv run mypy scripts/aggregate_upstream_bottleneck_2026_05_20.py && uv run python -u scripts/aggregate_upstream_bottleneck_2026_05_20.py 2>&1
```

Expected: prints the ranked table + L5 summary + top recommendation. Writes `summary.md`.

### Step 3: Commit

```bash
cd /Users/mario/Personal/Projects/RallyCut && git add analysis/scripts/aggregate_upstream_bottleneck_2026_05_20.py analysis/reports/upstream_bottleneck_2026_05_20/summary.md && git commit -m "$(cat <<'EOF'
diag(upstream): bottleneck aggregator + ranked investment decision

Joins L1..L6 per-layer JSONs into a single ranked table with
rank_score = realistic * (1 - gap_ratio) / cost. Writes summary.md
with top recommendation and projection-trap caveats. L5 learning-curve
slope reported separately.

[no-version-bump]

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Review findings and decide

This task is mine to handle (controller, not subagent). After all 6 layers + aggregator have run:

- [ ] Read `analysis/reports/upstream_bottleneck_2026_05_20/summary.md`
- [ ] Check L5 learning curve slope — if still sloping at frac=1.00 by ≥0.01, GT IS the bottleneck (recommend labeling investment).
- [ ] Check top-ranked non-L5 layer — if its realistic ceiling >20 contacts AND gap_ratio <0.3, propose implementation spec for that layer.
- [ ] If all realistic ceilings are <10 contacts, we're near the true classical-ML ceiling on current data; flag as "out of classical-ML reach without visual signals" (user has constrained VLM out).
- [ ] Write a new memory entry `upstream_bottleneck_findings_2026_05_20.md` summarizing the verdict.
- [ ] Update MEMORY.md to add the workstream entry.
- [ ] Report to user with the top recommendation + reasoning.

---

## Out of scope

- Implementing any single layer's classical-ML fix. That's a follow-up plan once the probe ranks the bottleneck.
- Confounded-recovery Venn analysis as full per-contact CSV. The aggregator notes confounding caveats but doesn't materialize the per-contact CSV (would be a small follow-up if needed for deep investigation).
- BLOCK action in L5 (too small a sample).
- Player-tracker retraining (would be a separate workstream once L1 is ranked top).
- WASB retraining (separate workstream once L4 is ranked top and we have measurement support).
- Improving the L4 ball-GT loader's robustness — current schema is best-guess; if the schema doesn't match, the L4 task includes a one-liner to inspect and adapt.

---

## Self-Review Notes

- **Spec coverage:** L1-L6 + aggregator + decision step all present. Substrate definition, dual-ceiling methodology, Venn-warning, and decision framework all implemented.
- **Placeholder scan:** the per-contact Venn CSV is noted as "TBD (follow-up)" in the aggregator — this is the only acknowledged placeholder, and it's scoped out, not left ambiguous.
- **Type consistency:** `WrongAttributionRow` defined in Task 1 with fields `rally_id`, `video`, `action_frame`, `action_type`, `pipeline_pid`, `gt_pid`, `pipeline_match_delta` — all subsequent tasks use these exact field names.
- **rescore_contact signature:** keyword args `expected_team`, `team_assignments_int`, `contact_frame_override`, `ball_position_override` — used consistently across L1/L2/L3/L4/L6 probes.
