"""Tests for _upstream_probe_common helpers."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from _upstream_probe_common import WrongAttributionRow, derive_gt_team_chain  # noqa: E402


def test_wrong_attribution_row_shape():
    """Smoke-check the dataclass has all required fields."""
    r = WrongAttributionRow(
        rally_id="r1", video="v1", action_frame=100, action_type="attack",
        pipeline_pid=2, gt_pid=3, pipeline_match_delta=0,
    )
    assert r.rally_id == "r1"
    assert r.gt_pid != r.pipeline_pid


def test_derive_gt_team_chain_simple():
    """Walk a rally with 4 GT contacts: serve P2 -> receive P3 -> set P4 -> attack P3.

    With team_assignments {1:A, 2:A, 3:B, 4:B}, the team chain by actor is:
      idx 0: SERVE by P2 -> A
      idx 1: RECEIVE by P3 -> B
      idx 2: SET by P4 -> B
      idx 3: ATTACK by P3 -> B
    """
    gt_contacts = [
        (100, "serve",   2),
        (140, "receive", 3),
        (160, "set",     4),
        (180, "attack",  3),
    ]
    team_assignments = {"1": "A", "2": "A", "3": "B", "4": "B"}
    chain = derive_gt_team_chain(gt_contacts, team_assignments)
    assert chain == ["A", "B", "B", "B"]
