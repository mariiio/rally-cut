"""Verify reattribute_actions stamps actions_pipeline_version on its SQL UPDATE.

The CLI does a direct SQL UPDATE bypassing the Prisma layer; the column
must be set in the same UPDATE statement so the version reflects the
current vintage of the action_classifier code.
"""

from __future__ import annotations

from pathlib import Path


def test_update_statement_includes_actions_pipeline_version() -> None:
    """The CLI's UPDATE statement sets actions_pipeline_version alongside actions_json."""
    src = Path(__file__).resolve().parents[2] / "rallycut" / "cli" / "commands" / "reattribute_actions.py"
    text = src.read_text()
    # The SQL may be split across lines, so check for both parts separately
    # in the context of UPDATE player_tracks.
    assert "UPDATE player_tracks SET actions_json = %s" in text, (
        "reattribute_actions.py UPDATE statement missing or altered"
    )
    assert "actions_pipeline_version = %s" in text, (
        "reattribute_actions.py UPDATE must set actions_pipeline_version"
    )
    # And that ACTION_PIPELINE_VERSION is imported/used.
    assert "ACTION_PIPELINE_VERSION" in text


def test_update_does_not_touch_contacts_pipeline_version() -> None:
    """reattribute_actions only writes actions_json — contacts_* columns are unchanged."""
    src = Path(__file__).resolve().parents[2] / "rallycut" / "cli" / "commands" / "reattribute_actions.py"
    text = src.read_text()
    assert "contacts_pipeline_version" not in text, (
        "reattribute_actions only writes actions_*; touching contacts_* would "
        "incorrectly bump the contact-pipeline vintage."
    )
