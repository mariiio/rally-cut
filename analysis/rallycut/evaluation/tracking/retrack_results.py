"""Persist last retrack evaluation run for delta comparison.

Stores per-rally metrics as JSON so the next --retrack run can show
Δ columns (improvement/regression) vs the previous run.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_RESULTS_DIR = Path.home() / ".cache" / "rallycut" / "retrack_results"
_LAST_RUN_PATH = _RESULTS_DIR / "last_run.json"


@dataclass
class RetrackRunResult:
    """Per-rally metrics from a single retrack evaluation."""

    rally_id: str
    hota: float | None = None
    f1: float = 0.0
    id_switches: int = 0
    identity_accuracy: float | None = None


def load_last_run() -> dict[str, RetrackRunResult] | None:
    """Load previous retrack run results.

    Returns:
        Dict of rally_id -> RetrackRunResult, or None if no previous run.
    """
    if not _LAST_RUN_PATH.exists():
        return None

    try:
        with open(_LAST_RUN_PATH) as f:
            data = json.load(f)

        return {
            rally_id: RetrackRunResult(
                rally_id=rally_id,
                hota=r.get("hota"),
                f1=r.get("f1", 0.0),
                id_switches=r.get("id_switches", 0),
                identity_accuracy=r.get("identity_accuracy"),
            )
            for rally_id, r in data.get("results", {}).items()
        }
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning(f"Failed to load last retrack run: {e}")
        return None


def save_run(results: dict[str, RetrackRunResult]) -> None:
    """Save current retrack run results for next delta comparison."""
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    data: dict[str, Any] = {
        "results": {
            rally_id: asdict(r)
            for rally_id, r in results.items()
        },
    }

    try:
        with open(_LAST_RUN_PATH, "w") as f:
            json.dump(data, f, indent=2)
    except OSError as e:
        logger.warning(f"Failed to save retrack results: {e}")


def format_delta(current: float, previous: float | None, fmt: str = ".1%") -> str:
    """Format a metric delta as a colored string for Rich.

    Args:
        current: Current value.
        previous: Previous value (None = no delta shown).
        fmt: Format spec for the delta value.

    Returns:
        String like "[green]+1.2%[/green]" or "[red]-0.5%[/red]" or "".
    """
    if previous is None:
        return ""
    delta = current - previous
    if abs(delta) < 5e-4:
        return "[dim]±0[/dim]"
    sign = "+" if delta > 0 else ""
    color = "green" if delta > 0 else "red"
    return f"[{color}]{sign}{delta:{fmt}}[/{color}]"


def format_delta_int(current: int, previous: int | None) -> str:
    """Format an integer metric delta (lower is better for IDsw)."""
    if previous is None:
        return ""
    delta = current - previous
    if delta == 0:
        return "[dim]±0[/dim]"
    sign = "+" if delta > 0 else ""
    # For ID switches, fewer is better → negative delta is green
    color = "red" if delta > 0 else "green"
    return f"[{color}]{sign}{delta}[/{color}]"
