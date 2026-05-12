"""Main CLI entry point for RallyCut."""

import typer
from rich.console import Console

from rallycut.cli.commands.analyze import app as analyze_app
from rallycut.cli.commands.audit_action_gt import audit_action_gt_cmd
from rallycut.cli.commands.audit_coherence_invariants import audit_coherence_invariants_cmd
from rallycut.cli.commands.audit_pid_invariants import audit_pid_invariants_cmd
from rallycut.cli.commands.cleanup_stale_attribution import cleanup_stale_attribution_cmd
from rallycut.cli.commands.cleanup_team_assignments import cleanup_team_assignments_cmd
from rallycut.cli.commands.cleanup_team_labels_by_majority import (
    cleanup_team_labels_by_majority_cmd,
)
from rallycut.cli.commands.compare_tracking import compare_tracking as compare_tracking_command
from rallycut.cli.commands.compute_match_stats import compute_match_stats_cmd
from rallycut.cli.commands.cut import cut as cut_command
from rallycut.cli.commands.detect_court import detect_court as detect_court_command
from rallycut.cli.commands.evaluate import app as evaluate_app
from rallycut.cli.commands.evaluate_tracking import app as evaluate_tracking_app
from rallycut.cli.commands.label import app as label_app
from rallycut.cli.commands.match_players import match_players as match_players_command
from rallycut.cli.commands.migrate_action_gt import migrate_action_gt_cmd
from rallycut.cli.commands.preflight import preflight as preflight_command
from rallycut.cli.commands.preview_check import preview_check as preview_check_command
from rallycut.cli.commands.profile import profile as profile_command
from rallycut.cli.commands.reattribute_actions import reattribute_actions_cmd
from rallycut.cli.commands.reinterpolate_primary import reinterpolate_primary_cmd
from rallycut.cli.commands.remap_track_ids import remap_track_ids_cmd
from rallycut.cli.commands.repair_identities import repair_identities_cmd
from rallycut.cli.commands.tilt_detect import tilt_detect as tilt_detect_command
from rallycut.cli.commands.track_player import track_players as track_players_command
from rallycut.cli.commands.train import app as train_app

app = typer.Typer(
    name="rallycut",
    help="Volleyball video analysis CLI - rally detection",
    no_args_is_help=True,
)

console = Console()

# Register commands
app.command(name="cut")(cut_command)
app.command(name="profile")(profile_command)
app.command(name="track-players")(track_players_command)
app.command(name="compare-tracking")(compare_tracking_command)
app.command(name="match-players")(match_players_command)
app.command(name="detect-court")(detect_court_command)
app.command(name="preflight")(preflight_command)
app.command(name="preview-check")(preview_check_command)
app.command(name="tilt-detect")(tilt_detect_command)
app.command(name="compute-match-stats")(compute_match_stats_cmd)
app.command(name="reattribute-actions")(reattribute_actions_cmd)
app.command(name="reinterpolate-primary")(reinterpolate_primary_cmd)
app.command(name="remap-track-ids")(remap_track_ids_cmd)
app.command(name="repair-identities")(repair_identities_cmd)
app.command(name="audit-action-gt")(audit_action_gt_cmd)
app.command(name="audit-coherence-invariants")(audit_coherence_invariants_cmd)
app.command(name="audit-pid-invariants")(audit_pid_invariants_cmd)
app.command(name="migrate-action-gt")(migrate_action_gt_cmd)
app.command(name="cleanup-team-assignments")(cleanup_team_assignments_cmd)
app.command(name="cleanup-stale-attribution")(cleanup_stale_attribution_cmd)
app.command(name="cleanup-team-labels-by-majority")(cleanup_team_labels_by_majority_cmd)
app.add_typer(evaluate_tracking_app, name="evaluate-tracking")
app.add_typer(label_app, name="label")
app.add_typer(evaluate_app, name="evaluate")
app.add_typer(train_app, name="train")
app.add_typer(analyze_app, name="analyze")


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
) -> None:
    """RallyCut - Volleyball video analysis CLI."""
    if verbose:
        console.print("[dim]Verbose mode enabled[/dim]")


if __name__ == "__main__":
    app()
