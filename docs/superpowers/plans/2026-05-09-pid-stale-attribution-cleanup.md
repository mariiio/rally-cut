# PID Stale Attribution Cleanup CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `cleanup-stale-attribution` CLI that filters legacy `positions_json` and `actions_json["actions"]` per rally, with `--dry-run` mode for safe preview.

**Architecture:** Typer CLI mirroring `cleanup-team-assignments` pattern. Per rally: load primary_track_ids + positions + actions, filter both to keep only entries referencing primary tracks (positions) or primary ∪ {-1} (actions), write back if changed. `--dry-run` runs the full filtering loop and prints diffs without committing.

**Tech Stack:** Python 3.11, Typer, Rich, psycopg via existing `rallycut.evaluation.tracking.db.get_connection`.

**Spec:** `docs/superpowers/specs/2026-05-09-pid-stale-attribution-cleanup-design.md`

---

## File Structure

**New (1):**
- `analysis/rallycut/cli/commands/cleanup_stale_attribution.py` — Typer command (~100 LOC).

**Modified (1):**
- `analysis/rallycut/cli/main.py` — register the new command (~2 lines: import + `app.command`).

---

## Task 1: Build and register the CLI command

**Files:**
- Create: `analysis/rallycut/cli/commands/cleanup_stale_attribution.py`
- Modify: `analysis/rallycut/cli/main.py`

- [ ] **Step 1: Create the CLI command**

Create `analysis/rallycut/cli/commands/cleanup_stale_attribution.py` with this EXACT content:

```python
"""CLI: rallycut cleanup-stale-attribution <video-id> [--dry-run]

One-shot cleanup that filters per-rally `positions_json` and
`actions_json["actions"]` to drop entries referencing track IDs not in
`primary_track_ids`. Closes I-2 (positions leakage), I-3 and I-7 (action
attribution leakage) on legacy data persisted before Task 12's silent-skip
fix in compute_match_stats.

Idempotent. Includes `--dry-run` mode for safe preview before mutating.
"""

from __future__ import annotations

import json
from typing import Any, cast

import typer
from rich.console import Console

from rallycut.evaluation.tracking.db import get_connection

console = Console()


def _filter_positions(
    positions_json: list[dict[str, Any]], primary: set[int]
) -> tuple[list[dict[str, Any]], int]:
    """Keep only positions whose trackId is in primary. Return (filtered, n_dropped)."""
    kept: list[dict[str, Any]] = []
    dropped = 0
    for p in positions_json:
        try:
            tid = int(p.get("trackId", -1))
        except (TypeError, ValueError):
            dropped += 1
            continue
        if tid in primary:
            kept.append(p)
        else:
            dropped += 1
    return kept, dropped


def _filter_actions(
    actions: list[dict[str, Any]], primary: set[int]
) -> tuple[list[dict[str, Any]], int]:
    """Keep only actions whose playerTrackId is in primary ∪ {-1}. Return (filtered, n_dropped)."""
    allowed = primary | {-1}
    kept: list[dict[str, Any]] = []
    dropped = 0
    for a in actions:
        try:
            tid = int(a.get("playerTrackId", -1))
        except (TypeError, ValueError):
            dropped += 1
            continue
        if tid in allowed:
            kept.append(a)
        else:
            dropped += 1
    return kept, dropped


def cleanup_stale_attribution_cmd(
    video_id: str = typer.Argument(..., help="Video UUID"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Preview changes without writing to DB"
    ),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Suppress per-rally info"
    ),
) -> None:
    """Drop legacy positions/actions referencing non-primary track IDs (I-2/I-3/I-7)."""
    prefix = "[DRY RUN] " if dry_run else ""
    if not quiet:
        console.print(
            f"[dim]{prefix}Cleaning up stale attribution for video {video_id}…[/dim]"
        )

    rally_query = """
        SELECT
            r.id AS rally_id,
            pt.primary_track_ids,
            pt.positions_json,
            pt.actions_json
        FROM rallies r
        JOIN player_tracks pt ON pt.rally_id = r.id
        WHERE r.video_id = %s
        ORDER BY r.start_ms
    """

    n_updated = 0
    n_no_change = 0
    n_skipped = 0
    total_positions_dropped = 0
    total_actions_dropped = 0

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(rally_query, [video_id])
            rows = cur.fetchall()

        for row in rows:
            rally_id = cast(str, row[0])
            primary_raw = row[1]
            positions_json = row[2]
            actions_json = row[3]

            if (
                not isinstance(primary_raw, list)
                or not primary_raw
                or not isinstance(positions_json, list)
                or not isinstance(actions_json, dict)
            ):
                n_skipped += 1
                if not quiet:
                    console.print(
                        f"  [yellow][skip][/yellow] rally {rally_id}: "
                        f"missing primary_track_ids, positions, or actions_json"
                    )
                continue

            primary = {int(t) for t in primary_raw}

            new_positions, n_pos_dropped = _filter_positions(positions_json, primary)
            actions_list = actions_json.get("actions")
            if isinstance(actions_list, list):
                new_actions, n_act_dropped = _filter_actions(actions_list, primary)
            else:
                new_actions = []
                n_act_dropped = 0

            if n_pos_dropped == 0 and n_act_dropped == 0:
                n_no_change += 1
                if not quiet:
                    console.print(
                        f"  [dim][noop][/dim] rally {rally_id}: already clean"
                    )
                continue

            total_positions_dropped += n_pos_dropped
            total_actions_dropped += n_act_dropped

            if dry_run:
                if not quiet:
                    console.print(
                        f"  [cyan][DRY][/cyan]   rally {rally_id}: "
                        f"would drop {n_pos_dropped} positions, "
                        f"{n_act_dropped} actions"
                    )
                continue

            new_actions_json = dict(actions_json)
            if isinstance(actions_list, list):
                new_actions_json["actions"] = new_actions

            try:
                with conn.cursor() as wcur:
                    wcur.execute(
                        "UPDATE player_tracks "
                        "SET positions_json = %s, actions_json = %s "
                        "WHERE rally_id = %s",
                        [
                            json.dumps(new_positions),
                            json.dumps(new_actions_json),
                            rally_id,
                        ],
                    )
                conn.commit()
                n_updated += 1
                if not quiet:
                    console.print(
                        f"  [green][fix][/green]  rally {rally_id}: "
                        f"dropped {n_pos_dropped} positions, "
                        f"{n_act_dropped} actions"
                    )
            except Exception as exc:
                conn.rollback()
                if not quiet:
                    console.print(
                        f"  [red][err][/red]  rally {rally_id}: "
                        f"write failed: {exc}"
                    )

    summary_label = "Dry-run" if dry_run else "Summary"
    console.print(
        f"\n[bold]{summary_label}:[/bold] "
        f"{n_updated} updated · {n_no_change} no-change · {n_skipped} skipped · "
        f"{total_positions_dropped} positions · {total_actions_dropped} actions "
        f"{'(would be) ' if dry_run else ''}dropped"
    )
```

- [ ] **Step 2: Verify the module imports cleanly**

```bash
cd analysis && uv run python -c "from rallycut.cli.commands.cleanup_stale_attribution import cleanup_stale_attribution_cmd; print('ok')"
```

Expected: prints `ok`.

- [ ] **Step 3: Register in `cli/main.py`**

In `analysis/rallycut/cli/main.py`:

After the line:
```python
from rallycut.cli.commands.cleanup_team_assignments import cleanup_team_assignments_cmd
```

Add:
```python
from rallycut.cli.commands.cleanup_stale_attribution import cleanup_stale_attribution_cmd
```

After the line:
```python
app.command(name="cleanup-team-assignments")(cleanup_team_assignments_cmd)
```

Add:
```python
app.command(name="cleanup-stale-attribution")(cleanup_stale_attribution_cmd)
```

- [ ] **Step 4: Verify CLI registration**

```bash
uv run rallycut cleanup-stale-attribution --help
```

Expected: Typer help output describing the command, including the line `Drop legacy positions/actions referencing non-primary track IDs (I-2/I-3/I-7).` and the `--dry-run` option.

- [ ] **Step 5: Lint + mypy**

```bash
cd analysis && uv run ruff check rallycut/cli/commands/cleanup_stale_attribution.py rallycut/cli/main.py
cd analysis && uv run mypy rallycut/cli/commands/cleanup_stale_attribution.py
```

Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add analysis/rallycut/cli/commands/cleanup_stale_attribution.py analysis/rallycut/cli/main.py
git commit -m "feat(cli): cleanup-stale-attribution — filter legacy positions/actions (closes I-2/I-3/I-7)"
```

---

## Task 2: Dry-run preview on the fleet

**Files:** None edited. This is verification.

The dry-run pass must precede any real run on the fleet. Aggregate totals are checked against the audit's pre-cleanup invariant counts.

- [ ] **Step 1: Ensure `/tmp/audit_videos.txt` exists with the 70 audit-able video IDs**

If the file doesn't exist (it was created earlier in the session), regenerate:

```bash
cd analysis && uv run python -c "
from rallycut.evaluation.tracking.db import get_connection
with get_connection() as conn:
    with conn.cursor() as cur:
        cur.execute('''SELECT DISTINCT v.id FROM videos v JOIN rallies r ON r.video_id = v.id 
                       JOIN player_tracks pt ON pt.rally_id = r.id WHERE pt.actions_json IS NOT NULL ORDER BY v.id''')
        for row in cur.fetchall():
            print(row[0])
" > /tmp/audit_videos.txt && wc -l /tmp/audit_videos.txt
```

Expected: 70 lines.

- [ ] **Step 2: Dry-run cleanup on all 70 videos, capturing summary lines**

```bash
while read vid; do
  uv run rallycut cleanup-stale-attribution "$vid" --dry-run --quiet
done < /tmp/audit_videos.txt > /tmp/dry_run_summary.txt 2>&1
```

This redirects ALL output (including per-rally lines if any leaked through, plus the per-video summary lines) to `/tmp/dry_run_summary.txt`. The `--quiet` flag suppresses per-rally lines, leaving only summaries.

- [ ] **Step 3: Aggregate dry-run totals**

```bash
grep "Dry-run:" /tmp/dry_run_summary.txt | awk '
  /positions/ {
    for (i=1; i<=NF; i++) {
      if ($i ~ /^[0-9]+$/ && $(i+1) == "positions") pos += $i;
      if ($i ~ /^[0-9]+$/ && $(i+1) == "actions") act += $i;
    }
  }
  END { printf "Total positions would-be-dropped: %d\nTotal actions would-be-dropped: %d\n", pos, act }
'
```

Expected approximate values (from pre-cleanup fleet audit):
- Positions: ≈ 200-300 (I-2 had 206 violations; one violation = one offending track_id which may have many position rows, so this could be larger).
- Actions: ≈ 100-150 (I-3 had 141 violations; one violation = one action, so this should be close to 141).

If values are within an order of magnitude of expected, proceed to Task 3. If wildly different (e.g., > 10x or < 0.1x), STOP and investigate.

- [ ] **Step 4: No commit (verification only)**

---

## Task 3: Real run on the fleet + re-audit

**Files:** None edited. This is execution + verification.

- [ ] **Step 1: Real cleanup pass on all 70 videos**

```bash
while read vid; do
  echo "===== ${vid:0:8} ====="
  uv run rallycut cleanup-stale-attribution "$vid" --quiet
done < /tmp/audit_videos.txt
```

Expected: per-video Summary lines showing some `updated` count. Total updated rallies should roughly match the pre-cleanup I-3 video count (since I-3 was per-action and most rallies have multiple offending actions, the rally count is likely smaller than 141).

- [ ] **Step 2: Re-run fleet audit**

```bash
cd analysis && uv run python scripts/fleet_pid_audit.py
```

Expected output (per the spec's Done criteria):
- I-2: 0
- I-3: 0
- I-7: 0
- I-6: ≤ 1 (unchanged from pre-cleanup; one stubborn case on 635dcba2)
- I-1: ~16 (unchanged)
- I-5: ~33 (unchanged)
- Total: ~49 violations on outlier videos.

If totals don't match (e.g., I-2/I-3/I-7 still > 0 anywhere), STOP and surface findings.

- [ ] **Step 3: Update memory**

The memory directory at `/Users/mario/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/` is NOT a git repo. Just write files; no commit needed.

Create `/Users/mario/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/pid_stale_attribution_cleanup_2026_05_09.md`:

```markdown
---
name: PID stale attribution cleanup 2026-05-09
description: Sub-1.1.C — cleanup-stale-attribution CLI clears legacy I-2/I-3/I-7 violations across the fleet
type: project
---

# PID Stale Attribution Cleanup CLI (Sub-1.1.C)

**Shipped:** 2026-05-09

**Why:** Sub-1.1.B's cleanup-team-assignments cleared I-6 across the fleet but left ~400 residual violations: I-2 (positions_json contains non-primary tracks), I-3 (actions attributed to non-primary tracks), I-7 (post-mapping mapped_track_id outside canonical range). All three are correlated symptoms of pre-Task-12 tracking-pipeline state where the filter to primary tracks was incomplete at storage time. Re-tracking is heavy; this CLI does the in-place filter.

**What it does:** Reads each rally's `positions_json` + `actions_json` + `primary_track_ids`, filters positions to keep only primary tracks, filters actions to keep only primary ∪ {-1}, writes back. Includes `--dry-run` for safe preview. One DB transaction per rally. Idempotent.

**How to apply:** Use `uv run rallycut cleanup-stale-attribution <video-id>` for any pre-fix-tracked video. Run `--dry-run` first to preview the diff. Safe to re-run.

**Spec:** `docs/superpowers/specs/2026-05-09-pid-stale-attribution-cleanup-design.md`
**Plan:** `docs/superpowers/plans/2026-05-09-pid-stale-attribution-cleanup.md`

**Post-cleanup fleet baseline (2026-05-09, after Sub-1.1.B + Sub-1.1.C):**
- 70 videos audited.
- I-1: ~16 violations on ~13 videos (primary set size != 4 — needs re-tracking, not cleanup).
- I-2/I-3/I-6/I-7: zero (all cleared via the two cleanup CLIs).
- I-5: ~33 violations on 2 outlier videos (073cb11b: 32, 635dcba2: 1) — investigate individually.
- I-6: ≤ 1 stubborn case on 635dcba2 — investigate alongside its I-5.

Total residual: ~49 violations on outliers. The fleet is otherwise audit-clean.
```

Then update `/Users/mario/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/MEMORY.md` — find the entry for the Sub-1 audit workstream (should be the first item under "Current workstreams") and update it to reference the new memory file. Use the Edit tool with this old_string:

```markdown
- [SHIPPED] [**PID leverage audit Sub-1 2026-05-08**](pid_i6_team_assignments_realign_2026_05_08.md) — `audit-pid-invariants` CLI + 7 invariants (I-1..I-7) + I-7 silent-skip in `compute_match_stats`. Sub-1.1.A `player_tracker.py` re-derive after `optimize_global_identity` closes I-6 for new tracking. Sub-1.1.B `cleanup-team-assignments` CLI ([cleanup CLI memo](pid_i6_cleanup_cli_2026_05_09.md)) clears legacy panel I-6. Post-cleanup: 5c756c41 + 854bb250 audit-clean; b5fb0594 + 7d77980f have residual phantom-action I-2/I-3/I-7 (29 total) deferred as legacy noise.
```

and this new_string:

```markdown
- [SHIPPED] [**PID leverage audit Sub-1 2026-05-08**](pid_i6_team_assignments_realign_2026_05_08.md) — `audit-pid-invariants` CLI + 7 invariants (I-1..I-7) + I-7 silent-skip in `compute_match_stats`. Sub-1.1.A `player_tracker.py` re-derive after `optimize_global_identity` closes I-6 for new tracking. Sub-1.1.B `cleanup-team-assignments` CLI ([memo](pid_i6_cleanup_cli_2026_05_09.md)) clears legacy I-6 fleet-wide (47 videos). Sub-1.1.C `cleanup-stale-attribution` CLI ([memo](pid_stale_attribution_cleanup_2026_05_09.md)) clears legacy I-2/I-3/I-7 (~400 violations across 49 videos) via in-place filter. Final fleet residual: ~49 I-1/I-5 outliers on a few videos requiring individual investigation.
```

- [ ] **Step 4: No commit (memory dir is not a git repo)**

---

## Done criteria

- [ ] `uv run rallycut cleanup-stale-attribution --help` works and shows the `--dry-run` flag.
- [ ] Dry-run pass on the fleet completes without errors and produces sane totals.
- [ ] Real run on the fleet completes; re-audit shows zero I-2/I-3/I-7 across all 70 videos.
- [ ] Final fleet residual is the expected I-1 + I-5 outlier set (~49 violations).
- [ ] Memory entry recorded.

## Out of scope

- I-1 cleanup (primary set size != 4 — needs re-tracking).
- I-5 cleanup (match-analysis coverage gap on 073cb11b — separate investigation).
- I-6 stubborn case on 635dcba2 (separate investigation).
- Sub-2 (team-coherence validator) and Sub-3 (web debug surface).
