"""Unit tests for score-tracking GT backup helpers."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def fake_conn() -> MagicMock:
    """Fake psycopg connection whose cursor returns scriptable rows."""
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cursor
    conn.cursor.return_value.__exit__.return_value = False
    return conn


def _set_rows(conn: MagicMock, rows: list[tuple]) -> MagicMock:
    cursor = conn.cursor.return_value.__enter__.return_value
    cursor.fetchall.return_value = rows
    return cursor


class TestExportScoreGroundTruth:
    def test_returns_entries_with_either_field(
        self, fake_conn: MagicMock
    ) -> None:
        from rallycut.cli.commands.train import _export_score_ground_truth

        # Rows shape: (content_hash, rally_id, video_id, gt_serving_team, gt_side_switch)
        _set_rows(
            fake_conn,
            [
                ("hashA", "rally-serving-only", "video-1", "A", None),
                ("hashA", "rally-switch-only", "video-1", None, True),
                ("hashB", "rally-both", "video-2", "B", False),
                ("hashC", "rally-other", "video-3", "A", True),  # not in dataset
            ],
        )

        with patch(
            "rallycut.evaluation.db.get_connection", return_value=fake_conn
        ):
            result = _export_score_ground_truth({"hashA", "hashB"})

        assert result is not None
        assert result["stats"] == {
            "total_rallies": 3,
            "total_videos": 2,
            "total_with_serving": 2,
            "total_with_side_switch": 2,
        }

        by_id = {r["rally_id"]: r for r in result["rallies"]}
        assert set(by_id) == {
            "rally-serving-only",
            "rally-switch-only",
            "rally-both",
        }

        serving_only = by_id["rally-serving-only"]
        assert serving_only["gt_serving_team"] == "A"
        assert "gt_side_switch" not in serving_only  # NULL in DB → omitted

        switch_only = by_id["rally-switch-only"]
        assert switch_only["gt_side_switch"] is True
        assert "gt_serving_team" not in switch_only  # NULL in DB → omitted

        both = by_id["rally-both"]
        assert both["gt_serving_team"] == "B"
        assert both["gt_side_switch"] is False

    def test_returns_none_when_no_rows(self, fake_conn: MagicMock) -> None:
        from rallycut.cli.commands.train import _export_score_ground_truth

        _set_rows(fake_conn, [])
        with patch(
            "rallycut.evaluation.db.get_connection", return_value=fake_conn
        ):
            assert _export_score_ground_truth({"hashA"}) is None

    def test_returns_none_when_all_filtered_out(
        self, fake_conn: MagicMock
    ) -> None:
        from rallycut.cli.commands.train import _export_score_ground_truth

        _set_rows(
            fake_conn, [("hashZ", "rally-9", "video-9", "A", None)]
        )
        with patch(
            "rallycut.evaluation.db.get_connection", return_value=fake_conn
        ):
            assert _export_score_ground_truth({"hashA"}) is None


class TestRestoreScoreGroundTruth:
    def _write_payload(self, tmp_path: Path) -> Path:
        payload = {
            "stats": {
                "total_rallies": 3,
                "total_videos": 1,
                "total_with_serving": 2,
                "total_with_side_switch": 2,
            },
            "rallies": [
                {
                    "rally_id": "rally-serving-writable",
                    "video_id": "vid-1",
                    "content_hash": "hash1",
                    "gt_serving_team": "A",
                    "gt_side_switch": True,
                },
                {
                    "rally_id": "rally-switch-only",
                    "video_id": "vid-1",
                    "content_hash": "hash1",
                    "gt_side_switch": False,
                },
                {
                    "rally_id": "rally-missing",
                    "video_id": "vid-1",
                    "content_hash": "hash1",
                    "gt_serving_team": "B",
                },
            ],
        }
        p = tmp_path / "score_ground_truth.json"
        p.write_text(json.dumps(payload))
        return p

    def test_updates_only_null_rows_and_records_misses(
        self, tmp_path: Path
    ) -> None:
        from typing import Any

        from rallycut.training.restore import (
            RestoreResult,
            _restore_score_ground_truth,
        )

        path = self._write_payload(tmp_path)

        # The helper runs, per rally:
        #   1) SELECT id FROM rallies WHERE id = %s   (existence check)
        #   2) UPDATE gt_serving_team (only if the field is in the JSON)
        #   3) UPDATE gt_side_switch  (only if the field is in the JSON)
        # Rally 1 has both fields: serving UPDATE rowcount=1, side_switch UPDATE
        #   rowcount=0 (already set) → counts as restored (serving wrote).
        # Rally 2 has only side_switch: side_switch UPDATE rowcount=1 → counts.
        # Rally 3 is missing from DB: existence SELECT returns None → error.
        existence_results = iter(
            [
                ("rally-serving-writable",),
                ("rally-switch-only",),
                None,
            ]
        )
        update_rowcounts = iter([1, 0, 1])

        cursor = MagicMock()

        def execute_side_effect(sql: str, _params: Any = None) -> None:
            sql_lower = sql.lower()
            if "select id from rallies" in sql_lower:
                cursor._last = "select"
            elif "update rallies" in sql_lower:
                cursor._last = "update"
                cursor.rowcount = next(update_rowcounts)
            else:
                cursor._last = "other"

        cursor.execute.side_effect = execute_side_effect
        cursor.fetchone.side_effect = lambda: next(existence_results)

        conn = MagicMock()
        conn.cursor.return_value.__enter__.return_value = cursor
        conn.cursor.return_value.__exit__.return_value = False

        result = RestoreResult()
        _restore_score_ground_truth(conn, path, result)

        assert result.score_gt_restored == 2
        assert any("rally-missing" in e for e in result.errors)

    def test_silent_skip_when_file_missing(self, tmp_path: Path) -> None:
        from rallycut.training.restore import (
            RestoreResult,
            _restore_score_ground_truth,
        )

        result = RestoreResult()
        _restore_score_ground_truth(MagicMock(), tmp_path / "missing.json", result)

        assert result.score_gt_restored == 0
        assert result.errors == []
