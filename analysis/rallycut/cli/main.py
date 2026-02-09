"""Main CLI entry point for RallyCut."""

import typer
from rich.console import Console

from rallycut.cli.commands.cut import cut as cut_command
from rallycut.cli.commands.evaluate import app as evaluate_app
from rallycut.cli.commands.profile import profile as profile_command
from rallycut.cli.commands.track_ball import track_ball as track_ball_command
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
app.command(name="track-ball")(track_ball_command)
app.command(name="track-players")(track_players_command)
app.add_typer(evaluate_app, name="evaluate")
app.add_typer(train_app, name="train")


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
) -> None:
    """RallyCut - Volleyball video analysis CLI."""
    if verbose:
        console.print("[dim]Verbose mode enabled[/dim]")


if __name__ == "__main__":
    app()
