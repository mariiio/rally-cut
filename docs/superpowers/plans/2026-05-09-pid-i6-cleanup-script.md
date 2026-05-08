# PID Invariant I-6 Cleanup Script Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a one-shot CLI command that rebuilds persisted `teamAssignments` per rally for a given video, clearing I-6 violations on legacy panel data without re-tracking.

**Architecture:** New Typer CLI `cleanup-team-assignments` reads each rally's `positions_json` + `actions_json["teamAssignments"]` + `court_split_y` from `player_tracks`, converts the legacy `"A"/"B"` dict to `int 0/1` for `classify_teams` (using existing precomputed-passthrough), runs `classify_teams` to reconstruct the dict, converts back to `"A"/"B"`, writes back to `actions_json`. One DB transaction per rally. Idempotent.

**Tech Stack:** Python 3.11, Typer, Rich, psycopg via existing `rallycut.evaluation.tracking.db.get_connection`.

**Spec:** `docs/superpowers/specs/2026-05-08-pid-i6-cleanup-script-design.md`

---

## File Structure

**New (1):**
- `analysis/rallycut/cli/commands/cleanup_team_assignments.py` — Typer command (~80 LOC).

**Modified (1):**
- `analysis/rallycut/cli/main.py` — register the new command (~2 lines: import + `app.command()`).

---

## Task 1: Build and register the CLI command

**Files:**
- Create: `analysis/rallycut/cli/commands/cleanup_team_assignments.py`
- Modify: `analysis/rallycut/cli/main.py`

- [ ] **Step 1: Create the CLI command**

Create `analysis/rallycut/cli/commands/cleanup_team_assignments.py` with this exact content:

```python
"""CLI: rallycut cleanup-team-assignments <video-id>

One-shot cleanup that rebuilds persisted `teamAssignments` per rally
using `classify_teams` precomputed-passthrough. Clears I-6 violations
on legacy data persisted before the player_tracker fix landed.

The CLI is idempotent: re-running on already-clean rallies is a no-op.
"""

from __future__ import annotations

import json
from typing import Any, cast

import typer
from rich.console import Console

from rallycut.evaluation.tracking.db import get_connection
from rallycut.tracking.player_filter import classify_teams
from rallycut.tracking.player_tracker import PlayerPosition

console = Console()

# Convention from action_classifier.py:138 — team 0 (near) = "A", team 1 (far) = "B".
_LABEL_TO_INT = {"A": 0, "B": 1}
_INT_TO_LABEL = {0: "A", 1: "B"}


def _convert_str_to_int(team_assignments: dict[str, str]) -> dict[int, int]:
    """Convert legacy {str: 'A'|'B'} → {int: 0|1}. Skip corrupt entries."""
    result: dict[int, int] = {}
    for tid_str, label in team_assignments.items():
        if label not in _LABEL_TO_INT:
            continue
        try:
            result[int(tid_str)] = _LABEL_TO_INT[label]
        except (ValueError, TypeError):
            continue
    return result


def _convert_int_to_str(team_assignments: dict[int, int]) -> dict[str, str]:
    """Convert classify_teams output {int: 0|1} → {str: 'A'|'B'}."""
    return {
        str(tid): _INT_TO_LABEL[team]
        for tid, team in team_assignments.items()
        if team in _INT_TO_LABEL
    }


def _reconstruct_positions(positions_json: list[dict[str, Any]]) -> list[PlayerPosition]:
    """Build PlayerPosition objects from persisted JSON.

    classify_teams only reads frame_number, track_id, and y; other fields
    use safe defaults.
    """
    out: list[PlayerPosition] = []
    for p in positions_json:
        try:
            out.append(
                PlayerPosition(
                    frame_number=int(p.get("frameNumber", 0)),
                    track_id=int(p.get("trackId", -1)),
                    x=float(p.get("x", 0.0)),
                    y=float(p.get("y", 0.0)),
                    width=float(p.get("width", 0.0)),
                    height=float(p.get("height", 0.0)),
                    confidence=float(p.get("confidence", 0.0)),
                )
            )
        except (TypeError, ValueError):
            continue
    return out


def cleanup_team_assignments_cmd(
    video_id: str = typer.Argument(..., help="Video UUID"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress per-rally info"),
) -> None:
    """Rebuild persisted teamAssignments per rally to close I-6 violations on legacy data."""
    if not quiet:
        console.print(f"[dim]Cleaning up teamAssignments for video {video_id}…[/dim]")

    rally_query = """
        SELECT
            r.id AS rally_id,
            pt.positions_json,
            pt.actions_json,
            pt.court_split_y
        FROM rallies r
        JOIN player_tracks pt ON pt.rally_id = r.id
        WHERE r.video_id = %s
        ORDER BY r.start_ms
    """

    n_updated = 0
    n_skipped = 0
    n_no_change = 0

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(rally_query, [video_id])
            rows = cur.fetchall()

        for row in rows:
            rally_id = cast(str, row[0])
            positions_json = row[1]
            actions_json = row[2]
            court_split_y = row[3]

            if (
                court_split_y is None
                or not isinstance(positions_json, list)
                or not positions_json
                or not isinstance(actions_json, dict)
            ):
                n_skipped += 1
                if not quiet:
                    console.print(
                        f"  [yellow][skip][/yellow] rally {rally_id}: missing court_split_y, "
                        f"positions, or actions_json"
                    )
                continue

            old_team_assignments_str = actions_json.get("teamAssignments")
            if not isinstance(old_team_assignments_str, dict):
                n_skipped += 1
                if not quiet:
                    console.print(
                        f"  [yellow][skip][/yellow] rally {rally_id}: no teamAssignments dict"
                    )
                continue

            int_precomputed = _convert_str_to_int(old_team_assignments_str)
            positions = _reconstruct_positions(positions_json)
            if not positions:
                n_skipped += 1
                if not quiet:
                    console.print(
                        f"  [yellow][skip][/yellow] rally {rally_id}: no usable positions"
                    )
                continue

            new_team_assignments_int = classify_teams(
                positions,
                float(court_split_y),
                precomputed_assignments=int_precomputed,
            )
            new_team_assignments_str = _convert_int_to_str(new_team_assignments_int)

            if new_team_assignments_str == old_team_assignments_str:
                n_no_change += 1
                if not quiet:
                    console.print(
                        f"  [dim][noop][/dim] rally {rally_id}: already clean"
                    )
                continue

            actions_json["teamAssignments"] = new_team_assignments_str

            try:
                with conn.cursor() as wcur:
                    wcur.execute(
                        "UPDATE player_tracks SET actions_json = %s WHERE rally_id = %s",
                        [json.dumps(actions_json), rally_id],
                    )
                conn.commit()
                n_updated += 1
                if not quiet:
                    console.print(
                        f"  [green][fix][/green]  rally {rally_id}: "
                        f"{old_team_assignments_str} → {new_team_assignments_str}"
                    )
            except Exception as exc:
                conn.rollback()
                if not quiet:
                    console.print(
                        f"  [red][err][/red]  rally {rally_id}: write failed: {exc}"
                    )

    console.print(
        f"\n[bold]Summary:[/bold] {n_updated} updated · "
        f"{n_no_change} no-change · {n_skipped} skipped"
    )
```

- [ ] **Step 2: Verify the module imports cleanly**

```bash
cd analysis && uv run python -c "from rallycut.cli.commands.cleanup_team_assignments import cleanup_team_assignments_cmd; print('ok')"
```

Expected: prints `ok`.

- [ ] **Step 3: Register in `cli/main.py`**

In `analysis/rallycut/cli/main.py`:

After the line:
```python
from rallycut.cli.commands.audit_pid_invariants import audit_pid_invariants_cmd
```

Add:
```python
from rallycut.cli.commands.cleanup_team_assignments import cleanup_team_assignments_cmd
```

After the line:
```python
app.command(name="audit-pid-invariants")(audit_pid_invariants_cmd)
```

Add:
```python
app.command(name="cleanup-team-assignments")(cleanup_team_assignments_cmd)
```

- [ ] **Step 4: Verify CLI registration**

```bash
uv run rallycut cleanup-team-assignments --help
```

Expected: Typer help output describing the command, including the line `Rebuild persisted teamAssignments per rally to close I-6 violations on legacy data.`.

- [ ] **Step 5: Lint + mypy**

```bash
cd analysis && uv run ruff check rallycut/cli/commands/cleanup_team_assignments.py rallycut/cli/main.py
cd analysis && uv run mypy rallycut/cli/commands/cleanup_team_assignments.py
```

Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add analysis/rallycut/cli/commands/cleanup_team_assignments.py analysis/rallycut/cli/main.py
git commit -m "feat(cli): cleanup-team-assignments — rebuild legacy teamAssignments (closes I-6)"
```

---

## Task 2: Run cleanup on panel + verify audit clean

**Files:** None edited; this is integration verification.

- [ ] **Step 1: Run cleanup on each panel video (in order)**

```bash
for v in 5c756c41-1cc1-4486-a95c-97398912cfbe \
         b5fb0594-d64f-4a0d-bad9-de8fc36414d0 \
         854bb250-3e91-47d2-944d-f62413e3cf45 \
         7d77980f-3006-40e0-adc0-db491a5bb659; do
  echo "===== $v ====="
  uv run rallycut cleanup-team-assignments "$v"
done
```

Expected per video: a Summary line showing some `updated` count (the rallies that had broken teamAssignments). Total updated across the 4 videos should be roughly the number of I-6 violations from the pre-cleanup audit (~27 rallies, but some rallies may have multiple violations sharing one rally id).

- [ ] **Step 2: Re-run audit per video; confirm I-6 → 0**

```bash
for v in 5c756c41-1cc1-4486-a95c-97398912cfbe \
         b5fb0594-d64f-4a0d-bad9-de8fc36414d0 \
         854bb250-3e91-47d2-944d-f62413e3cf45 \
         7d77980f-3006-40e0-adc0-db491a5bb659; do
  echo "===== $v ====="
  uv run rallycut audit-pid-invariants "$v"
done
```

Expected:
- 5c756c41: zero violations (✓ All invariants hold) — was 5 I-6, now 0.
- 854bb250: zero violations — was 2 I-6, now 0.
- b5fb0594: 17 violations remaining (was 24; should drop the 7 I-6, leaving 1 I-2 + 11 I-3 + 5 I-7).
- 7d77980f: 13 violations remaining (was 26; should drop the 13 I-6, leaving 8 I-3 + 5 I-7).

If a panel video shows MORE I-6 violations after cleanup than before, **stop** and report — that's a regression. If I-6 is zero on all 4 videos, proceed.

- [ ] **Step 3: No commit (this task is verification only)**

If everything is clean, proceed to Step 4. If not, surface findings to the user before continuing.

- [ ] **Step 4: Update memory with post-cleanup panel state**

Update `/Users/mario/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/MEMORY.md` — find the existing entry from Sub-1.1.A and add a sibling line near it (under "Current workstreams"):

```markdown
- [SHIPPED] [**PID I-6 cleanup CLI 2026-05-09**](pid_i6_cleanup_cli_2026_05_09.md) — `cleanup-team-assignments` command; panel I-6 cleared; I-3/I-7 residual on b5fb0594 + 7d77980f remain as legacy noise.
```

Then create the memory file at `/Users/mario/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/pid_i6_cleanup_cli_2026_05_09.md`:

```markdown
---
name: PID I-6 cleanup CLI 2026-05-09
description: Sub-1.1.B — one-shot cleanup-team-assignments CLI clears legacy I-6 violations on the panel without re-tracking
type: project
---

# PID I-6 Cleanup CLI (Sub-1.1.B)

**Shipped:** 2026-05-09

**Why:** Sub-1.1.A's player_tracker fix only takes effect on new tracking runs. Legacy panel data (tracked weeks ago) still had broken teamAssignments and continued to fail the I-6 audit after re-matching. Re-tracking the panel was the heavy alternative; this CLI is the lightweight one.

**What it does:** Reads each rally's persisted `positions_json` + existing `teamAssignments` + `court_split_y`, runs `classify_teams` with the existing dict as `precomputed_assignments`, writes the corrected dict back to `actions_json["teamAssignments"]`. Same logic as the player_tracker fix, applied retroactively. One DB transaction per rally. Idempotent.

**How to apply:** Use `uv run rallycut cleanup-team-assignments <video-id>` for any pre-fix-tracked video that fails the I-6 audit. The CLI is idempotent — safe to re-run.

**Spec:** `docs/superpowers/specs/2026-05-08-pid-i6-cleanup-script-design.md`
**Plan:** `docs/superpowers/plans/2026-05-09-pid-i6-cleanup-script.md`

**Post-cleanup panel state (2026-05-09):**
- 5c756c41: zero violations.
- 854bb250: zero violations.
- b5fb0594: 17 residual violations (1 I-2 + 11 I-3 + 5 I-7), all phantom-action leakage from before the Task 12 silent-skip fix.
- 7d77980f: 13 residual violations (8 I-3 + 5 I-7), same phantom-action class.
- Total residual: 30 violations on 2 panel videos. Treated as legacy noise; not addressed in Sub-1.1.B per scope decision. Sub-2 should filter these rallies if they cause coherence-validator false positives.
```

The memory directory at `/Users/mario/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/` is NOT a git repository — verify with `git status -C /Users/mario/.claude/projects/...` if uncertain. Just write the files; no commit needed there.

---

## Done criteria

- [ ] `uv run rallycut cleanup-team-assignments --help` shows the command help text.
- [ ] After running cleanup on all 4 panel videos, audit shows zero I-6 violations on each.
- [ ] I-3 and I-7 residual counts match the pre-cleanup baseline (no new violations introduced; legacy violations preserved per scope).
- [ ] Memory entry recorded.

## Out of scope

- I-3/I-7 phantom-action cleanup.
- Any change to `player_tracker.py` (already shipped in Sub-1.1.A).
- Re-tracking via Modal or local CPU.
- Unit tests for the CLI itself (logic covered by Sub-1.1.A's classify_teams tests; integration via audit).
