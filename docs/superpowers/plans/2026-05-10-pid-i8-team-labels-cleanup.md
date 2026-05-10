# PID I-8 Team Labels Cleanup CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `cleanup-team-labels-by-majority` CLI that rewrites scrambled team partitions on legacy data using cross-rally per-PID majority, with `--dry-run` and strict 2v2 + existing-partition safety checks.

**Architecture:** Typer CLI mirroring `cleanup-team-assignments` and `cleanup-stale-attribution`. Per video: load all rallies, compute set of valid 2v2 partitions seen + per-PID majority labels across all rallies. Per rally: if invalid (not 2v2), build candidate by flipping each primary's label to its per-PID majority; commit only if candidate is 2v2 AND matches an existing valid partition in the video.

**Tech Stack:** Python 3.11, Typer, Rich, psycopg via existing `rallycut.evaluation.tracking.db.get_connection`.

**Spec:** `docs/superpowers/specs/2026-05-10-pid-i8-team-labels-cleanup-design.md`

---

## File Structure

**New (1):**
- `analysis/rallycut/cli/commands/cleanup_team_labels_by_majority.py` — Typer command (~150 LOC).

**Modified (1):**
- `analysis/rallycut/cli/main.py` — register the new command (~2 lines: import + `app.command`).

---

## Task 1: Build and register the CLI command

**Files:**
- Create: `analysis/rallycut/cli/commands/cleanup_team_labels_by_majority.py`
- Modify: `analysis/rallycut/cli/main.py`

- [ ] **Step 1: Create the CLI command**

Create `analysis/rallycut/cli/commands/cleanup_team_labels_by_majority.py` with this EXACT content:

```python
"""CLI: rallycut cleanup-team-labels-by-majority <video-id> [--dry-run]

One-shot cleanup that rewrites scrambled team partitions per rally using
cross-rally per-PID majority. Closes I-8 violations on legacy data without
re-tracking.

Conservative by design:
  - Only commits a fix when the candidate is exactly 2A+2B.
  - Only commits a fix when the candidate matches a partition that already
    exists as a valid 2v2 in another rally of the same video.
  - Skips PIDs whose A/B count is tied (no decisive majority) — typical for
    side-switched videos.

These gates make silent corruption impossible. Worst case: leaves a rally
unchanged.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, cast

import typer
from rich.console import Console

from rallycut.evaluation.tracking.db import get_connection

console = Console()


def _partition_for_primary(
    team_assignments: dict[str, str], primary: list[int]
) -> dict[int, str] | None:
    """Project teamAssignments onto the primary set; None if any label invalid."""
    out: dict[int, str] = {}
    for tid in primary:
        label = team_assignments.get(str(tid))
        if label not in ("A", "B"):
            return None
        out[tid] = label
    return out


def _is_2v2(partition: dict[int, str]) -> bool:
    a = sum(1 for v in partition.values() if v == "A")
    b = sum(1 for v in partition.values() if v == "B")
    return a == 2 and b == 2


def _per_pid_majority(
    rallies: list[tuple[list[int], dict[str, str]]],
) -> dict[int, str | None]:
    """Per-PID majority across all rallies. None for ties or no observations."""
    pid_counts: dict[int, Counter[str]] = {}
    for primary, ta in rallies:
        for tid in primary:
            label = ta.get(str(tid))
            if label not in ("A", "B"):
                continue
            pid_counts.setdefault(tid, Counter())[label] += 1
    result: dict[int, str | None] = {}
    for tid, counts in pid_counts.items():
        a, b = counts.get("A", 0), counts.get("B", 0)
        if a > b:
            result[tid] = "A"
        elif b > a:
            result[tid] = "B"
        else:
            result[tid] = None  # tie
    return result


def _candidate_from_majority(
    primary: list[int], majority: dict[int, str | None]
) -> dict[int, str] | None:
    """Build candidate by replacing each primary's label with its majority.
    Returns None if any primary's majority is a tie (None) or absent.
    """
    out: dict[int, str] = {}
    for tid in primary:
        m = majority.get(tid)
        if m not in ("A", "B"):
            return None
        out[tid] = m
    return out


def cleanup_team_labels_by_majority_cmd(
    video_id: str = typer.Argument(..., help="Video UUID"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Preview changes without writing to DB"
    ),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Suppress per-rally info"
    ),
) -> None:
    """Rewrite scrambled team partitions to per-PID majority (closes I-8 on legacy data)."""
    prefix = "[DRY RUN] " if dry_run else ""
    if not quiet:
        console.print(
            f"[dim]{prefix}Cleaning up team labels for video {video_id}…[/dim]"
        )

    rally_query = """
        SELECT
            r.id AS rally_id,
            pt.primary_track_ids,
            pt.actions_json
        FROM rallies r
        JOIN player_tracks pt ON pt.rally_id = r.id
        WHERE r.video_id = %s
        ORDER BY r.start_ms
    """

    n_fixed = 0
    n_noop = 0
    n_skipped = 0
    n_ambiguous = 0

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(rally_query, [video_id])
            rows = cur.fetchall()

        # First pass: collect rally-level data + valid 2v2 partitions seen.
        per_rally: list[
            tuple[str, list[int], dict[str, Any], dict[str, str]]
        ] = []
        valid_partitions: set[frozenset[tuple[int, str]]] = set()
        rallies_for_majority: list[tuple[list[int], dict[str, str]]] = []
        for row in rows:
            rally_id = cast(str, row[0])
            primary_raw = row[1]
            actions_json = row[2]
            if (
                not isinstance(primary_raw, list)
                or len(primary_raw) != 4
                or not isinstance(actions_json, dict)
            ):
                continue
            ta = actions_json.get("teamAssignments")
            if not isinstance(ta, dict):
                continue
            primary = [int(t) for t in primary_raw]
            partition = _partition_for_primary(ta, primary)
            if partition is None:
                continue  # I-6 territory
            per_rally.append((rally_id, primary, actions_json, ta))
            rallies_for_majority.append((primary, ta))
            if _is_2v2(partition):
                valid_partitions.add(frozenset(partition.items()))

        majority = _per_pid_majority(rallies_for_majority)

        # Second pass: per-rally fix attempt.
        for rally_id, primary, actions_json, ta in per_rally:
            partition = _partition_for_primary(ta, primary)
            if partition is None or _is_2v2(partition):
                n_noop += 1
                if not quiet:
                    console.print(
                        f"  [dim][noop][/dim] rally {rally_id}: already 2v2 or unscored"
                    )
                continue

            candidate = _candidate_from_majority(primary, majority)
            if candidate is None:
                n_ambiguous += 1
                if not quiet:
                    console.print(
                        f"  [yellow][skip-tie][/yellow] rally {rally_id}: "
                        f"some primary PID has tied majority (likely side-switched)"
                    )
                continue
            if not _is_2v2(candidate):
                n_ambiguous += 1
                if not quiet:
                    console.print(
                        f"  [yellow][ambiguous][/yellow] rally {rally_id}: "
                        f"per-PID majority does not produce 2v2"
                    )
                continue
            if frozenset(candidate.items()) not in valid_partitions:
                n_ambiguous += 1
                if not quiet:
                    console.print(
                        f"  [yellow][ambiguous][/yellow] rally {rally_id}: "
                        f"candidate {dict(sorted(candidate.items()))} "
                        f"matches no existing valid partition in this video"
                    )
                continue

            # Build new actions_json with the corrected teamAssignments.
            new_ta: dict[str, str] = dict(ta)
            for tid, label in candidate.items():
                new_ta[str(tid)] = label
            new_actions_json = dict(actions_json)
            new_actions_json["teamAssignments"] = new_ta

            old_view = {str(t): ta.get(str(t)) for t in sorted(primary)}
            new_view = {str(t): new_ta[str(t)] for t in sorted(primary)}

            if dry_run:
                if not quiet:
                    console.print(
                        f"  [cyan][DRY][/cyan]   rally {rally_id}: "
                        f"would fix {old_view} → {new_view}"
                    )
                n_fixed += 1
                continue

            try:
                with conn.cursor() as wcur:
                    wcur.execute(
                        "UPDATE player_tracks SET actions_json = %s "
                        "WHERE rally_id = %s",
                        [json.dumps(new_actions_json), rally_id],
                    )
                conn.commit()
                n_fixed += 1
                if not quiet:
                    console.print(
                        f"  [green][fix][/green]  rally {rally_id}: "
                        f"{old_view} → {new_view}"
                    )
            except Exception as exc:
                conn.rollback()
                if not quiet:
                    console.print(
                        f"  [red][err][/red]  rally {rally_id}: "
                        f"write failed: {exc}"
                    )

        n_skipped = len(rows) - len(per_rally)

    summary_label = "Dry-run" if dry_run else "Summary"
    console.print(
        f"\n[bold]{summary_label}:[/bold] "
        f"{n_fixed} fixed · {n_noop} no-op · "
        f"{n_ambiguous} ambiguous · {n_skipped} skipped"
    )
```

- [ ] **Step 2: Verify the module imports cleanly**

```bash
cd analysis && uv run python -c "from rallycut.cli.commands.cleanup_team_labels_by_majority import cleanup_team_labels_by_majority_cmd; print('ok')"
```

Expected: prints `ok`.

- [ ] **Step 3: Register in `cli/main.py`**

In `analysis/rallycut/cli/main.py`:

After the line:
```python
from rallycut.cli.commands.cleanup_stale_attribution import cleanup_stale_attribution_cmd
```

Add:
```python
from rallycut.cli.commands.cleanup_team_labels_by_majority import cleanup_team_labels_by_majority_cmd
```

After the line:
```python
app.command(name="cleanup-stale-attribution")(cleanup_stale_attribution_cmd)
```

Add:
```python
app.command(name="cleanup-team-labels-by-majority")(cleanup_team_labels_by_majority_cmd)
```

- [ ] **Step 4: Verify CLI registration**

```bash
uv run rallycut cleanup-team-labels-by-majority --help
```

Expected: Typer help output describing the command, including the line `Rewrite scrambled team partitions to per-PID majority (closes I-8 on legacy data).` and the `--dry-run` option.

- [ ] **Step 5: Lint + mypy**

```bash
cd analysis && uv run ruff check rallycut/cli/commands/cleanup_team_labels_by_majority.py rallycut/cli/main.py
cd analysis && uv run mypy rallycut/cli/commands/cleanup_team_labels_by_majority.py
```

Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add analysis/rallycut/cli/commands/cleanup_team_labels_by_majority.py analysis/rallycut/cli/main.py
git commit -m "feat(cli): cleanup-team-labels-by-majority — fix scrambled partitions (closes legacy I-8)"
```

---

## Task 2: Dry-run preview on the fleet

**Files:** None edited. This is verification.

The dry-run pass must precede any real run. Aggregate counts are checked against the audit's I-8 violation count (112).

- [ ] **Step 1: Get the list of videos with I-8 violations**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && uv run python -c "
from rallycut.tracking.pid_invariants import run_all
with open('/tmp/audit_videos.txt') as f:
    vids = [line.strip() for line in f if line.strip()]
i8_videos = []
for vid in vids:
    vs = run_all(video_id=vid)
    if any(v.invariant == 'I-8' for v in vs):
        i8_videos.append(vid)
print(f'videos with I-8: {len(i8_videos)}')
with open('/tmp/i8_videos.txt', 'w') as f:
    for vid in i8_videos:
        f.write(vid + '\n')
"
```

Expected: `videos with I-8: 42` (matching the prior fleet audit).

- [ ] **Step 2: Dry-run cleanup on all 42 videos**

```bash
while read vid; do
  uv run rallycut cleanup-team-labels-by-majority "$vid" --dry-run --quiet
done < /tmp/i8_videos.txt > /tmp/i8_dry_run.txt 2>&1
echo "exit=$?"
echo "---summary lines:---"
grep "Dry-run:" /tmp/i8_dry_run.txt | wc -l
echo "---aggregate:---"
grep "Dry-run:" /tmp/i8_dry_run.txt | awk '
  {
    for (i=1; i<=NF; i++) {
      if ($i ~ /^[0-9]+$/ && $(i+1) == "fixed") fixed += $i;
      if ($i ~ /^[0-9]+$/ && $(i+1) == "ambiguous") amb += $i;
    }
  }
  END { printf "fixed=%d ambiguous=%d (sum=%d, expected ~112)\n", fixed, amb, fixed+amb }
'
```

Expected: `summary lines: 42`. Aggregate `fixed + ambiguous` should be close to but not exceed 112 (the pre-cleanup I-8 fleet count). A `fixed` count of ~80-90 (≈75-80% of 112) and `ambiguous` count of ~22-32 is the reasonable expectation per spec.

- [ ] **Step 3: Sanity check before real run**

If aggregate `fixed` is unreasonably high (>112) or zero, STOP — investigate. Otherwise proceed to Task 3.

- [ ] **Step 4: No commit (verification only)**

---

## Task 3: Real run + re-audit

**Files:** None edited. This is execution + verification.

- [ ] **Step 1: Real cleanup pass on all 42 videos**

```bash
while read vid; do
  uv run rallycut cleanup-team-labels-by-majority "$vid" --quiet
done < /tmp/i8_videos.txt > /tmp/i8_real_run.txt 2>&1
echo "exit=$?"
grep "Summary:" /tmp/i8_real_run.txt | awk '
  {
    for (i=1; i<=NF; i++) {
      if ($i ~ /^[0-9]+$/ && $(i+1) == "fixed") fixed += $i;
      if ($i ~ /^[0-9]+$/ && $(i+1) == "ambiguous") amb += $i;
    }
  }
  END { printf "Real run total: fixed=%d ambiguous=%d\n", fixed, amb }
'
```

Expected: `fixed` and `ambiguous` totals match the dry-run aggregates from Task 2.

- [ ] **Step 2: Re-run fleet audit**

```bash
cd analysis && uv run python scripts/fleet_pid_audit.py 2>&1 > /tmp/fleet_after_i8_cleanup.txt
sed -n '70,$p' /tmp/fleet_after_i8_cleanup.txt
```

Expected:
- I-8 violations significantly reduced (target: <30, ideally <22 if all "fixed" candidates committed cleanly).
- I-1: 1 (unchanged from prior — the `4b7ad71f` outlier).
- All other invariants: 0 (regression check — must NOT have introduced new I-1/I-2/I-3/I-4/I-5/I-6/I-7 violations).
- Clean videos count should rise.

If any other invariant count rises above its prior value, STOP and surface — the cleanup introduced a regression.

- [ ] **Step 3: Update memory**

The memory directory at `/Users/mario/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/` is NOT a git repo. Just write files; no commit needed.

Create `/Users/mario/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/pid_i8_cleanup_2026_05_10.md`:

```markdown
---
name: PID I-8 cleanup CLI 2026-05-10
description: Sub-1.1.E — cleanup-team-labels-by-majority rewrites scrambled team partitions using cross-rally per-PID majority
type: project
---

# PID I-8 Team Labels Cleanup CLI (Sub-1.1.E)

**Shipped:** 2026-05-10

**Why:** I-8 (added 2026-05-10) detects scrambled team partitions (1A+3B etc.) caused by `classify_teams` mis-classifying late-arriver tracks or Y-ambiguous starts. Visual verification on b097dd2a confirmed PIDs are stable across rallies but team labels for those PIDs vary per rally. Since a player's team is invariant within a match, the deviant rallies' labels are wrong.

**What it does:** Reads each rally's primary_track_ids + teamAssignments, computes per-PID majority across all rallies in the video, builds candidate fix by flipping each primary's label to its majority. Commits ONLY when (a) candidate is exactly 2A+2B AND (b) candidate matches a partition that already exists as a valid 2v2 in another rally of the same video. Conservative gates make silent corruption impossible.

**How to apply:** `uv run rallycut cleanup-team-labels-by-majority <video-id>` for any video failing I-8. Run with `--dry-run` first to preview.

**Side switches:** Per-PID majority for a side-switched video is 50% A / 50% B per PID → tie → no fix attempted. Side-switched-video rallies remain visible I-8 violations. This is acknowledged scope.

**Spec:** `docs/superpowers/specs/2026-05-10-pid-i8-team-labels-cleanup-design.md`
**Plan:** `docs/superpowers/plans/2026-05-10-pid-i8-team-labels-cleanup.md`

**Post-cleanup fleet baseline (2026-05-10):**
[fill in actual numbers from the re-audit in Task 3 Step 2; example expected:]
- 70 videos audited.
- I-8: ~22-30 residual (down from 112).
- All other invariants unchanged from prior baseline.
- Remaining I-8 mostly on side-switched videos; will be addressed by future producer fix in classify_teams.
```

Update `/Users/mario/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/MEMORY.md` — find the existing PID-leverage workstream entry and append Sub-1.1.E reference using Edit tool with this old_string:

```markdown
- [SHIPPED] [**PID leverage audit Sub-1 2026-05-08/10**](pid_i6_team_assignments_realign_2026_05_08.md) — `audit-pid-invariants` CLI + 8 invariants (I-1..I-8) + I-7 silent-skip in `compute_match_stats`. Sub-1.1.A `player_tracker.py` re-derive after `optimize_global_identity` closes I-6 for new tracking. Sub-1.1.B [`cleanup-team-assignments`](pid_i6_cleanup_cli_2026_05_09.md) cleared legacy I-6. Sub-1.1.C [`cleanup-stale-attribution`](pid_stale_attribution_cleanup_2026_05_09.md) cleared legacy I-2/I-3/I-7. Sub-1.1.D retracked 16 I-1 rallies. I-8 added 2026-05-10 (`5a6968b`) — beach VB requires 2A+2B partition. **Fleet state on structural invariants (I-1..I-7):** 69/70 clean (98%); 1 residual = video-metadata edge case. **Fleet state on correctness (I-8):** 42/70 videos have ≥1 scrambled-partition rally; 112 violations total. Root cause = `classify_teams` mishandles late-arriver tracks + Y-ambiguous starts. Producer fix not yet shipped — I-8 is current visibility surface. Frontend `STALE_TIMEOUT_MS` 30→60 min in `4767ef3`. Audit schema-bug fix `9e6390f` (snake_case match_analysis keys).
```

and this new_string:

```markdown
- [SHIPPED] [**PID leverage audit Sub-1 2026-05-08/10**](pid_i6_team_assignments_realign_2026_05_08.md) — `audit-pid-invariants` CLI + 8 invariants (I-1..I-8) + I-7 silent-skip in `compute_match_stats`. Sub-1.1.A `player_tracker.py` re-derive after `optimize_global_identity` closes I-6 for new tracking. Sub-1.1.B [`cleanup-team-assignments`](pid_i6_cleanup_cli_2026_05_09.md), Sub-1.1.C [`cleanup-stale-attribution`](pid_stale_attribution_cleanup_2026_05_09.md), Sub-1.1.E [`cleanup-team-labels-by-majority`](pid_i8_cleanup_2026_05_10.md) — three cleanup CLIs covering legacy I-6 / I-2-I-3-I-7 / I-8 respectively. Sub-1.1.D retracked 16 I-1 rallies. I-8 added (`5a6968b`) — beach VB requires 2A+2B partition. **Fleet state on structural invariants (I-1..I-7):** 69/70 clean. **Fleet state on correctness (I-8):** ~22-30 residual after Sub-1.1.E cleanup (down from 112); remaining mostly side-switched videos awaiting producer fix in classify_teams. Frontend `STALE_TIMEOUT_MS` 30→60 min in `4767ef3`. Audit schema-bug fix `9e6390f`.
```

(Replace "~22-30 residual" with the actual count from the re-audit when filling in.)

- [ ] **Step 4: No commit (memory dir is not a git repo)**

---

## Done criteria

- [ ] `uv run rallycut cleanup-team-labels-by-majority --help` works and shows the `--dry-run` flag.
- [ ] Dry-run pass on the 42 affected videos completes without errors and produces sane totals.
- [ ] Real run completes; re-audit shows I-8 substantially reduced.
- [ ] No new I-1 / I-2 / I-3 / I-4 / I-5 / I-6 / I-7 violations introduced.
- [ ] Memory entry recorded.

## Out of scope

- Side-switch-aware per-segment majority (rejected per spec — silent-corruption risk).
- Producer fix in `classify_teams` (separate workstream).
- Re-tracking.
- Unit tests for the CLI itself (logic uses predicates already covered by I-8's unit tests; `--dry-run` is the verification tool).
- Sub-2 / Sub-3 work.
