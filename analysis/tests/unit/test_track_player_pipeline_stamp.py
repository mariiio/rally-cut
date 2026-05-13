"""Verify the track_player CLI stamps pipeline versions onto its JSON output.

The Modal tracking pipeline invokes `rallycut track-players ... --actions
--pose` which writes a JSON file that the TS saveTrackingResult reads.
The version stamps must travel in that JSON.

These tests exercise the PRODUCTION helper `build_actions_data_dict` from
track_player.py — not a tautological reconstruction in the test body — so
a future regression that, e.g., drops one of the two version fields or
points it at the wrong constant will fail here.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

from rallycut.cli.commands.track_player import build_actions_data_dict
from rallycut.tracking.action_classifier import ACTION_PIPELINE_VERSION
from rallycut.tracking.contact_detector import CONTACT_PIPELINE_VERSION


def _mock_contact_seq(payload: dict | None = None) -> Mock:
    seq = Mock()
    seq.to_dict.return_value = payload if payload is not None else {"numContacts": 0, "contacts": []}
    return seq


def _mock_rally_actions(payload: dict | None = None) -> Mock:
    actions = Mock()
    actions.to_dict.return_value = payload if payload is not None else {"numContacts": 0, "actions": []}
    return actions


def test_build_actions_data_dict_includes_both_versions() -> None:
    result = build_actions_data_dict(_mock_contact_seq(), _mock_rally_actions())
    assert set(result.keys()) == {
        "contacts",
        "actions",
        "contactsPipelineVersion",
        "actionsPipelineVersion",
    }


def test_build_actions_data_dict_stamps_current_constants() -> None:
    """The helper must stamp the CURRENT module constants, not literal strings.

    Catches a regression where someone hard-codes 'v1' instead of importing.
    """
    result = build_actions_data_dict(_mock_contact_seq(), _mock_rally_actions())
    assert result["actionsPipelineVersion"] == ACTION_PIPELINE_VERSION
    assert result["contactsPipelineVersion"] == CONTACT_PIPELINE_VERSION


def test_build_actions_data_dict_versions_are_not_v0_sentinel() -> None:
    result = build_actions_data_dict(_mock_contact_seq(), _mock_rally_actions())
    assert result["actionsPipelineVersion"] != "v0"
    assert result["contactsPipelineVersion"] != "v0"


def test_build_actions_data_dict_preserves_producer_payloads() -> None:
    """The dict's contacts/actions fields are exactly the producers' to_dict() output."""
    contact_payload = {"numContacts": 3, "contacts": [{"frame": 10}]}
    actions_payload = {"numContacts": 3, "actions": [{"frame": 10, "action": "serve"}]}
    result = build_actions_data_dict(
        _mock_contact_seq(contact_payload), _mock_rally_actions(actions_payload),
    )
    assert result["contacts"] == contact_payload
    assert result["actions"] == actions_payload


def test_track_player_json_output_carries_versions_after_serialize(tmp_path: Path) -> None:
    """End-to-end-ish: the helper's output survives JSON round-trip with all 4 keys."""
    out = tmp_path / "tracks.json"
    actions_data = build_actions_data_dict(_mock_contact_seq(), _mock_rally_actions())
    base_dict = {"positions": [], "frameCount": 0}
    out.write_text(json.dumps({**base_dict, **actions_data}))

    loaded = json.loads(out.read_text())
    assert loaded["contactsPipelineVersion"] == CONTACT_PIPELINE_VERSION
    assert loaded["actionsPipelineVersion"] == ACTION_PIPELINE_VERSION
