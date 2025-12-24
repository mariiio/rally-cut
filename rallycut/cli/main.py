"""Main CLI entry point for RallyCut."""

import typer
from rich.console import Console

from rallycut.cli.commands.cut import cut as cut_command
from rallycut.cli.commands.overlay import overlay as overlay_command
from rallycut.cli.commands.highlights import highlights as highlights_command
from rallycut.cli.commands.profile import profile as profile_command

app = typer.Typer(
    name="rallycut",
    help="Beach volleyball video analysis CLI - auto-cut, highlights, and ball tracking",
    no_args_is_help=True,
)

console = Console()

# Register commands
app.command(name="cut")(cut_command)
app.command(name="overlay")(overlay_command)
app.command(name="highlights")(highlights_command)
app.command(name="profile")(profile_command)


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
):
    """RallyCut - Beach volleyball video analysis CLI."""
    if verbose:
        console.print("[dim]Verbose mode enabled[/dim]")


if __name__ == "__main__":
    app()
