"""End-to-end pipeline validation on 8 GT videos.

For each video: runs match-players → repair-identities → remap-track-ids →
reattribute-actions, then inspects DB results for correctness.

Reports per-video stats and spot-checks volleyball logic on 3 videos.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from rich.console import Console
from rich.table import Table

from rallycut.evaluation.db import get_connection

console = Console()

ANALYSIS_DIR = Path(__file__).resolve().parent.parent

VIDEOS = [
    "0a383519-ecaa-411a-8e5e-e0aadc835725",  # IMG_2313.MOV
    "7d77980f-3006-40e0-adc0-db491a5bb659",  # tata.mp4
    "2e984c43-cef6-4215-8d8e-50d892b510b9",  # titi.mp4
    "d5a6932f-e853-469b-ad9b-8c811a058526",  # tete.mp4
    "635dcba2-2645-49d1-ac62-072f05c957d9",  # toto.mp4
    "dd042609-e22e-4f60-83ed-038897c88c32",  # tete.mp4
    "ff175026-bd5b-4349-b390-188438da2005",  # lolo.mp4
    "84e66e74-8d4f-420a-ad01-0ada95153ad0",  # lala.mp4
]

# Spot-check these indices (0-based) — first, middle, last
SPOT_CHECK_INDICES = [0, 3, 6]
SPOT_CHECK_RALLIES = 5


@dataclass
class PipelineRunResult:
    video_id: str
    repair_count: int = 0
    stage_times: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


@dataclass
class VideoStats:
    video_id: str
    video_name: str = ""
    num_rallies: int = 0
    avg_confidence: float = 0.0
    min_confidence: float = 0.0
    side_switches: int = 0
    player_action_counts: dict[int, int] = field(default_factory=dict)
    total_actions: int = 0


@dataclass
class SpotCheckResult:
    rally_id: str
    rally_index: int
    issues: list[str] = field(default_factory=list)
    action_sequence: list[str] = field(default_factory=list)
    player_ids_used: set[int] = field(default_factory=set)
    team_assignments: dict[str, str] = field(default_factory=dict)
    num_actions: int = 0


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

def _run_stage(cmd: list[str], label: str) -> tuple[float, str, str]:
    """Run a pipeline stage. Returns (elapsed_s, stdout, stderr)."""
    t0 = time.time()
    result = subprocess.run(
        ["uv", "run", *cmd],
        capture_output=True, text=True,
        cwd=str(ANALYSIS_DIR),
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        raise RuntimeError(
            f"{label} failed (rc={result.returncode}): {result.stderr[:500]}"
        )
    return elapsed, result.stdout, result.stderr


def run_pipeline(video_id: str) -> PipelineRunResult:
    """Run all 4 post-tracking pipeline stages on a video."""
    run = PipelineRunResult(video_id=video_id)
    short = video_id[:8]

    stages = [
        ("match-players", ["rallycut", "match-players", video_id, "-q"]),
        ("repair-identities", ["rallycut", "repair-identities", video_id]),
        ("remap-track-ids", ["rallycut", "remap-track-ids", video_id]),
        ("reattribute-actions", ["rallycut", "reattribute-actions", video_id]),
    ]

    for label, cmd in stages:
        try:
            elapsed, stdout, stderr = _run_stage(cmd, label)
            run.stage_times[label] = elapsed

            if label == "repair-identities":
                m = re.search(r"Total repairs:\s*(\d+)", stdout + stderr)
                run.repair_count = int(m.group(1)) if m else -1

        except RuntimeError as e:
            run.errors.append(str(e))
            console.print(f"  [red]ERROR[/red] {label}: {e}")

    total = sum(run.stage_times.values())
    console.print(
        f"  {short}: {total:.0f}s "
        f"(match={run.stage_times.get('match-players', 0):.0f}s) "
        f"repairs={run.repair_count}"
    )
    return run


# ---------------------------------------------------------------------------
# Stats collection from DB
# ---------------------------------------------------------------------------

def collect_video_stats(video_id: str) -> VideoStats:
    """Query DB for pipeline results."""
    stats = VideoStats(video_id=video_id)

    with get_connection() as conn:
        with conn.cursor() as cur:
            # Video name + match analysis
            cur.execute(
                "SELECT filename, match_analysis_json FROM videos WHERE id = %s",
                [video_id],
            )
            row = cur.fetchone()
            if not row:
                stats.video_name = "NOT FOUND"
                return stats

            stats.video_name = str(row[0] or video_id[:8])
            match_json = row[1]

            if match_json:
                if isinstance(match_json, str):
                    match_json = json.loads(match_json)
                match_dict = cast(dict[str, Any], match_json)
                rallies_data = match_dict.get("rallies", [])
                confs = [
                    r.get("assignmentConfidence", 0.0)
                    for r in rallies_data
                    if r.get("assignmentConfidence") is not None
                ]
                if confs:
                    stats.avg_confidence = sum(confs) / len(confs)
                    stats.min_confidence = min(confs)
                stats.side_switches = sum(
                    1 for r in rallies_data if r.get("sideSwitchDetected")
                )

            # Rally count + actions
            cur.execute(
                """SELECT r.id, pt.actions_json, pt.primary_track_ids
                   FROM rallies r
                   JOIN player_tracks pt ON pt.rally_id = r.id
                   WHERE r.video_id = %s
                   ORDER BY r.start_ms""",
                [video_id],
            )
            rally_rows = cur.fetchall()
            stats.num_rallies = len(rally_rows)

            action_counts: dict[int, int] = defaultdict(int)
            total = 0
            for _rid, actions_json, _ptids in rally_rows:
                if not actions_json:
                    continue
                if isinstance(actions_json, str):
                    actions_json = json.loads(actions_json)
                actions_dict = cast(dict[str, Any], actions_json)
                for a in actions_dict.get("actions", []):
                    pid = a.get("playerTrackId", -1)
                    if pid >= 1:
                        action_counts[pid] += 1
                        total += 1

            stats.player_action_counts = dict(action_counts)
            stats.total_actions = total

    return stats


# ---------------------------------------------------------------------------
# Spot-check volleyball logic
# ---------------------------------------------------------------------------

def spot_check_video(video_id: str, max_rallies: int = 5) -> list[SpotCheckResult]:
    """Inspect action sequences for volleyball logic violations."""
    results: list[SpotCheckResult] = []

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT r.id, pt.actions_json, pt.primary_track_ids
                   FROM rallies r
                   JOIN player_tracks pt ON pt.rally_id = r.id
                   WHERE r.video_id = %s
                   ORDER BY r.start_ms
                   LIMIT %s""",
                [video_id, max_rallies],
            )
            rows = cur.fetchall()

    for i, (rally_id, actions_json, primary_track_ids) in enumerate(rows):
        sc = SpotCheckResult(rally_id=str(rally_id)[:8], rally_index=i)

        if not actions_json:
            sc.issues.append("No actions_json")
            results.append(sc)
            continue

        if isinstance(actions_json, str):
            actions_json = json.loads(actions_json)

        actions_dict = cast(dict[str, Any], actions_json)
        actions = actions_dict.get("actions", [])
        team_assignments = actions_dict.get("teamAssignments", {})
        sc.team_assignments = team_assignments
        sc.num_actions = len(actions)

        if not actions:
            sc.issues.append("Empty action list")
            results.append(sc)
            continue

        # Extract action sequence and player IDs
        for a in actions:
            sc.action_sequence.append(a.get("action", "?"))
            pid = a.get("playerTrackId", -1)
            if pid >= 0:
                sc.player_ids_used.add(pid)

        # Check 1: Serve first
        if sc.action_sequence[0] != "serve":
            sc.issues.append(f"First action is '{sc.action_sequence[0]}', not serve")

        # Check 2: No duplicate serves
        serve_count = sc.action_sequence.count("serve")
        if serve_count > 1:
            sc.issues.append(f"{serve_count} serves in rally")

        # Check 3: Reasonable rally length
        if sc.num_actions < 2:
            sc.issues.append(f"Only {sc.num_actions} action(s)")
        elif sc.num_actions > 20:
            sc.issues.append(f"Unusually long rally: {sc.num_actions} actions")

        # Check 4: Player IDs should be 1-4 for primary tracks
        high_ids = {pid for pid in sc.player_ids_used if pid > 4}
        if high_ids:
            sc.issues.append(f"Player IDs > 4 in actions: {high_ids}")

        # Check 5: Team pairing — P1-P2 same team, P3-P4 same team
        if team_assignments:
            t1 = team_assignments.get("1") or team_assignments.get(1)
            t2 = team_assignments.get("2") or team_assignments.get(2)
            t3 = team_assignments.get("3") or team_assignments.get(3)
            t4 = team_assignments.get("4") or team_assignments.get(4)
            if t1 and t2 and t1 != t2:
                sc.issues.append(f"P1({t1}) and P2({t2}) on different teams")
            if t3 and t4 and t3 != t4:
                sc.issues.append(f"P3({t3}) and P4({t4}) on different teams")

        # Check 6: Team alternation (≤3 consecutive same-team touches)
        if team_assignments:
            consecutive = 1
            max_consecutive = 1
            for j in range(1, len(actions)):
                prev_pid = str(actions[j - 1].get("playerTrackId", -1))
                curr_pid = str(actions[j].get("playerTrackId", -1))
                prev_team = team_assignments.get(prev_pid)
                curr_team = team_assignments.get(curr_pid)
                if prev_team and curr_team and prev_team == curr_team:
                    consecutive += 1
                    max_consecutive = max(max_consecutive, consecutive)
                else:
                    consecutive = 1
            if max_consecutive > 3:
                sc.issues.append(
                    f"{max_consecutive} consecutive touches by same team"
                )

        results.append(sc)

    return results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_summary_table(
    stats_list: list[VideoStats], runs: list[PipelineRunResult],
) -> None:
    """Print Rich summary table."""
    table = Table(title="Pipeline Validation (8 videos)")
    table.add_column("Video", style="cyan", no_wrap=True)
    table.add_column("Name", max_width=18)
    table.add_column("Rallies", justify="right")
    table.add_column("Avg Cf", justify="right")
    table.add_column("Min Cf", justify="right")
    table.add_column("Repairs", justify="right")
    table.add_column("Switches", justify="right")
    table.add_column("Actions P1/P2/P3/P4", justify="right")
    table.add_column("Total", justify="right")

    run_map = {r.video_id: r for r in runs}

    for s in stats_list:
        run = run_map.get(s.video_id)
        repairs = str(run.repair_count) if run else "?"
        if run and run.errors:
            repairs += " [red]ERR[/red]"

        ac = s.player_action_counts
        action_str = "/".join(str(ac.get(pid, 0)) for pid in [1, 2, 3, 4])

        min_cf_style = ""
        if s.min_confidence < 0.60:
            min_cf_style = "[red]"
        elif s.min_confidence < 0.70:
            min_cf_style = "[yellow]"

        min_cf_str = f"{min_cf_style}{s.min_confidence:.2f}"
        if min_cf_style:
            min_cf_str += "[/]"

        table.add_row(
            s.video_id[:8],
            s.video_name[:18],
            str(s.num_rallies),
            f"{s.avg_confidence:.2f}",
            min_cf_str,
            repairs,
            str(s.side_switches),
            action_str,
            str(s.total_actions),
        )

    console.print()
    console.print(table)


def print_spot_check_report(
    all_checks: list[tuple[str, list[SpotCheckResult]]],
) -> None:
    """Print spot-check results."""
    console.print("\n[bold]Spot-Check Report[/bold]")

    total_rallies = 0
    total_issues = 0

    for video_id, checks in all_checks:
        console.print(f"\n  [cyan]{video_id[:8]}[/cyan]")
        for sc in checks:
            total_rallies += 1
            seq = " → ".join(sc.action_sequence[:8])
            if len(sc.action_sequence) > 8:
                seq += " …"

            if sc.issues:
                total_issues += len(sc.issues)
                for issue in sc.issues:
                    console.print(f"    [yellow]![/yellow] rally {sc.rally_id} [{sc.num_actions} actions]: {issue}")
                console.print(f"      seq: {seq}")
            else:
                console.print(
                    f"    [green]OK[/green] rally {sc.rally_id}: "
                    f"{sc.num_actions} actions, players {sorted(sc.player_ids_used)}"
                )

    console.print(
        f"\n  Checked {total_rallies} rallies, "
        f"{'[green]0 issues[/green]' if total_issues == 0 else f'[yellow]{total_issues} issue(s)[/yellow]'}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    skip_pipeline = "--stats-only" in sys.argv

    console.print("[bold]End-to-end pipeline validation[/bold]")
    console.print(f"Videos: {len(VIDEOS)}")

    # Phase 1: Run pipeline on all videos
    runs: list[PipelineRunResult] = []
    if skip_pipeline:
        console.print("\n[dim]Skipping Phase 1 (--stats-only)[/dim]")
        runs = [PipelineRunResult(video_id=v) for v in VIDEOS]
    else:
        console.print("\n[bold]Phase 1: Running pipeline[/bold]")
        for i, vid in enumerate(VIDEOS):
            console.print(f"[{i + 1}/{len(VIDEOS)}] {vid[:8]}…", end=" ")
            run = run_pipeline(vid)
            runs.append(run)

    # Phase 2: Collect stats
    console.print("\n[bold]Phase 2: Collecting stats[/bold]")
    stats_list: list[VideoStats] = []
    for vid in VIDEOS:
        stats = collect_video_stats(vid)
        stats_list.append(stats)

    print_summary_table(stats_list, runs)

    # Phase 3: Spot-check
    console.print("\n[bold]Phase 3: Spot-checking[/bold]")
    all_checks: list[tuple[str, list[SpotCheckResult]]] = []
    for idx in SPOT_CHECK_INDICES:
        if idx < len(VIDEOS):
            vid = VIDEOS[idx]
            checks = spot_check_video(vid, max_rallies=SPOT_CHECK_RALLIES)
            all_checks.append((vid, checks))

    print_spot_check_report(all_checks)

    # Verdict
    console.print("\n[bold]Verdict[/bold]")
    total_errors = sum(len(r.errors) for r in runs)
    total_issues = sum(
        sum(len(sc.issues) for sc in checks) for _, checks in all_checks
    )
    low_conf = sum(1 for s in stats_list if s.min_confidence < 0.50)

    if total_errors > 0:
        console.print(f"  [red]FAIL[/red] — {total_errors} pipeline error(s)")
    elif low_conf > 0 or total_issues > 5:
        console.print(
            f"  [yellow]WARN[/yellow] — {total_issues} spot-check issue(s), "
            f"{low_conf} videos with min conf < 0.50"
        )
    else:
        console.print(f"  [green]PASS[/green] — {total_issues} issue(s)")


if __name__ == "__main__":
    main()
