"""Tests for _cascade_trace: opt-in per-rally pipeline-stage trace."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import pytest

from rallycut.tracking._cascade_trace import CascadeTrace, cascade_trace


@dataclass
class _FakeAction:
    frame: int
    action_type: str
    player_track_id: int


def test_disabled_when_env_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("CASCADE_TRACE_OUT", raising=False)
    with cascade_trace("rally-xyz") as tr:
        assert tr is None or not tr.is_enabled


def test_records_snapshots_when_env_set(tmp_path, monkeypatch):
    monkeypatch.setenv("CASCADE_TRACE_OUT", str(tmp_path))
    actions = [_FakeAction(100, "serve", 2), _FakeAction(140, "receive", 3)]
    with cascade_trace("rally-xyz") as tr:
        assert tr is not None and tr.is_enabled
        tr.snapshot("after_classify_rally", actions)
        tr.snapshot("after_scorer", [_FakeAction(100, "serve", 2),
                                     _FakeAction(140, "receive", 4)])  # pid changed

    out_path = tmp_path / "rally-xyz.trace.json"
    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert data["rally_id"] == "rally-xyz"
    assert [s["stage"] for s in data["snapshots"]] == [
        "after_classify_rally", "after_scorer",
    ]
    # Per-contact playerTrackId tracked by frame
    contacts = data["per_contact"]
    assert contacts["140"]["after_classify_rally"]["player_track_id"] == 3
    assert contacts["140"]["after_scorer"]["player_track_id"] == 4
