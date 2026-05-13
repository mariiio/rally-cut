"""Verify redetect_all_actions stamps both pipeline versions."""

from __future__ import annotations

from pathlib import Path


def test_update_includes_both_pipeline_versions() -> None:
    src = Path(__file__).resolve().parents[2] / "scripts" / "redetect_all_actions.py"
    text = src.read_text()
    # Both columns must appear in the UPDATE statement.
    assert "contacts_pipeline_version = %s" in text
    assert "actions_pipeline_version = %s" in text
    # And both constants must be imported / used.
    assert "ACTION_PIPELINE_VERSION" in text
    assert "CONTACT_PIPELINE_VERSION" in text
