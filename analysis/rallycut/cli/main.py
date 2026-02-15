"""Main CLI entry point for RallyCut."""

import typer
from rich.console import Console

from rallycut.cli.commands.analyze import app as analyze_app
from rallycut.cli.commands.compare_tracking import compare_tracking as compare_tracking_command
from rallycut.cli.commands.cut import cut as cut_command
from rallycut.cli.commands.evaluate import app as evaluate_app
from rallycut.cli.commands.evaluate_tracking import app as evaluate_tracking_app
from rallycut.cli.commands.label import app as label_app
from rallycut.cli.commands.match_players import match_players as match_players_command
from rallycut.cli.commands.profile import profile as profile_command
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
