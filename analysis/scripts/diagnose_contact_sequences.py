"""Diagnose invalid contact sequences across all rallies.

Categorizes failures by pattern (serve+receive same team, 5+ consecutive,
both) to determine whether they're caused by action classification errors
or player matching errors.

Usage:
    cd analysis
    uv run python scripts/diagnose_contact_sequences.py
"""

from __future__ import annotations

from collections import Counter, defaultdict

from rich.console import Console
from rich.table import Table

from rallycut.cli.commands.compute_match_stats import _load_rally_actions_and_positions
from rallycut.evaluation.db import get_connection
from rallycut.statistics.match_stats import _validate_contact_sequence
from rallycut.tracking.action_classifier import ActionType, RallyActions

console = Console()


def get_videos_with_actions() -> list[tuple[str, str, int]]:
    """Get all video IDs that have tracked rallies with actions."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT r.video_id, v.filename, COUNT(r.id)
                FROM rallies r
                JOIN player_tracks pt ON pt.rally_id = r.id
                JOIN videos v ON v.id = r.video_id
                WHERE pt.actions_json IS NOT NULL
                GROUP BY r.video_id, v.filename
                ORDER BY v.filename
            """)
            return [(str(r[0]), str(r[1] or "?"), int(r[2])) for r in cur.fetchall()]


def classify_failure(ra: RallyActions) -> dict:
    """Classify the failure pattern of an invalid rally.

    Returns dict with keys: rule1 (serve+receive same team),
    rule2 (5+ consecutive), team_pattern, details.
    """
    team_contacts = [
        (a.frame, a.team, a.action_type, a.player_track_id)
        for a in ra.actions
        if a.team in ("A", "B")
    ]
    team_contacts.sort(key=lambda x: x[0])

    if len(team_contacts) < 2:
        return {"rule1": False, "rule2": False, "team_pattern": "", "details": []}

    # Check rule 1: serve team == next non-serve team
    rule1 = False
    if team_contacts[0][2] == ActionType.SERVE:
        serve_team = team_contacts[0][1]
        for _, team, action_type, _ in team_contacts[1:]:
            if action_type != ActionType.SERVE:
                if team == serve_team:
                    rule1 = True
                break

    # Check rule 2: 5+ consecutive same-team contacts
    rule2 = False
    max_consecutive = 1
    consecutive = 1
    prev_team = team_contacts[0][1]
    for _, team, _, _ in team_contacts[1:]:
        if team == prev_team:
            consecutive += 1
            max_consecutive = max(max_consecutive, consecutive)
            if consecutive > 4:
                rule2 = True
        else:
            consecutive = 1
        prev_team = team

    # Build team pattern and detail strings
    team_pattern = " ".join(t[1] for t in team_contacts)
    details = [
        f"{t[2].value}({t[1]},tid={t[3]},f={t[0]})"
        for t in team_contacts
    ]

    # Count unknowns
    unknown_count = sum(1 for a in ra.actions if a.team == "unknown")

    return {
        "rule1": rule1,
        "rule2": rule2,
        "max_consecutive": max_consecutive,
        "team_pattern": team_pattern,
        "details": details,
        "unknown_count": unknown_count,
        "n_contacts": len(team_contacts),
    }


def main() -> None:
    videos = get_videos_with_actions()
    if not videos:
        console.print("[red]No videos with actions found.[/red]")
        return

    total_rallies = sum(n for _, _, n in videos)
    console.print(
        f"[bold]Scanning {len(videos)} videos, {total_rallies} rallies[/bold]\n"
    )

    # Collect all invalid rallies
    invalid_rallies: list[dict] = []
    valid_count = 0
    skip_count = 0
    video_invalid_counts: Counter[str] = Counter()

    for vid, filename, n_rallies in videos:
        rally_actions_list, *_ = _load_rally_actions_and_positions(vid)

        for ra in rally_actions_list:
            result = _validate_contact_sequence(ra)
            if result is None:
                skip_count += 1
            elif result:
                valid_count += 1
            else:
                info = classify_failure(ra)
                info["rally_id"] = ra.rally_id
                info["video_id"] = vid
                info["filename"] = filename
                invalid_rallies.append(info)
                video_invalid_counts[filename] += 1

    # --- Summary ---
    console.print(f"[bold green]Valid:[/bold green] {valid_count}")
    console.print(f"[bold red]Invalid:[/bold red] {len(invalid_rallies)}")
    console.print(f"[dim]Skipped (insufficient team info):[/dim] {skip_count}\n")

    if not invalid_rallies:
        console.print("[green]No invalid contact sequences found![/green]")
        return

    # --- Pattern categories ---
    rule1_only = [r for r in invalid_rallies if r["rule1"] and not r["rule2"]]
    rule2_only = [r for r in invalid_rallies if r["rule2"] and not r["rule1"]]
    both_rules = [r for r in invalid_rallies if r["rule1"] and r["rule2"]]

    table = Table(title="Failure Pattern Summary")
    table.add_column("Pattern", style="bold")
    table.add_column("Count", justify="right")
    table.add_column("% of Invalid", justify="right")
    table.add_column("Example Rally IDs")

    for label, group in [
        ("Same-team serve+receive (rule 1 only)", rule1_only),
        ("5+ consecutive same-team (rule 2 only)", rule2_only),
        ("Both rules", both_rules),
    ]:
        if group:
            pct = f"{len(group) / len(invalid_rallies) * 100:.0f}%"
            examples = ", ".join(r["rally_id"][:8] for r in group[:3])
            table.add_row(label, str(len(group)), pct, examples)

    console.print(table)

    # --- Sub-categorize rule 1 failures ---
    if rule1_only or both_rules:
        console.print("\n[bold]Rule 1 detail (serve+receive same team):[/bold]")
        r1_all = rule1_only + both_rules
        # Check if next action after serve is receive or something else
        missing_receive = 0
        wrong_team_receive = 0
        for r in r1_all:
            details = r["details"]
            if len(details) >= 2:
                # Parse first two actions
                first = details[0]  # serve(A,...)
                second = details[1]
                if "receive" in second or "dig" in second:
                    wrong_team_receive += 1
                else:
                    missing_receive += 1
        console.print(f"  Missing receive (serve→set/attack same team): {missing_receive}")
        console.print(f"  Wrong team receive (serve→receive same team): {wrong_team_receive}")

    # --- Video distribution ---
    console.print("\n[bold]Video distribution:[/bold]")
    vid_table = Table()
    vid_table.add_column("Video")
    vid_table.add_column("Invalid", justify="right")

    for filename, count in video_invalid_counts.most_common():
        vid_table.add_row(filename, str(count))
    console.print(vid_table)

    # --- Per-rally details ---
    console.print("\n[bold]Per-rally details:[/bold]")
    # Group by pattern
    by_pattern: defaultdict[str, list[dict]] = defaultdict(list)
    for r in invalid_rallies:
        if r["rule1"] and r["rule2"]:
            pat = "both"
        elif r["rule1"]:
            pat = "rule1"
        else:
            pat = "rule2"
        by_pattern[pat].append(r)

    for pat_name, label in [
        ("rule1", "RULE 1: Same-team serve+receive"),
        ("rule2", "RULE 2: 5+ consecutive same-team"),
        ("both", "BOTH RULES"),
    ]:
        group = by_pattern.get(pat_name, [])
        if not group:
            continue
        console.print(f"\n[bold underline]{label} ({len(group)})[/bold underline]")
        for r in group:
            console.print(f"\n  Rally: [cyan]{r['rally_id'][:8]}[/cyan]  "
                          f"Video: {r['filename']}  "
                          f"Contacts: {r['n_contacts']}  "
                          f"Unknown: {r['unknown_count']}  "
                          f"Max consec: {r['max_consecutive']}")
            console.print(f"  Team pattern: {r['team_pattern']}")
            console.print(f"  Actions: {' → '.join(r['details'])}")


if __name__ == "__main__":
    main()
