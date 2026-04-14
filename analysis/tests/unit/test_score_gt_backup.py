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

        # Rows: (content_hash, rally_id, video_id, start_ms, end_ms,
        #        gt_serving_team, gt_side_switch)
        # rally_id / video_id are still selected for the query but no longer
        # written to the JSON entry; the stable match key is content_hash +
        # start_ms + end_ms.
        _set_rows(
            fake_conn,
            [
                ("hashA", "rally-1", "video-1", 1000, 5000, "A", None),
                ("hashA", "rally-2", "video-1", 6000, 10000, None, True),
                ("hashB", "rally-3", "video-2", 2000, 7000, "B", False),
                ("hashC", "rally-4", "video-3", 0, 1000, "A", True),  # not in dataset
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

        for entry in result["rallies"]:
            assert set(entry.keys()) >= {
                "video_content_hash",
                "rally_start_ms",
                "rally_end_ms",
            }
            # Stable match-key fields only — no UUID keys
            assert "rally_id" not in entry
            assert "video_id" not in entry

        by_timing = {
            (e["video_content_hash"], e["rally_start_ms"], e["rally_end_ms"]): e
            for e in result["rallies"]
        }
        serving_only = by_timing[("hashA", 1000, 5000)]
        assert serving_only["gt_serving_team"] == "A"
        assert "gt_side_switch" not in serving_only

        switch_only = by_timing[("hashA", 6000, 10000)]
        assert switch_only["gt_side_switch"] is True
        assert "gt_serving_team" not in switch_only

        both = by_timing[("hashB", 2000, 7000)]
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
            fake_conn,
            [("hashZ", "rally-9", "video-9", 0, 1000, "A", None)],
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
                    "video_content_hash": "hash1",
                    "rally_start_ms": 1000,
                    "rally_end_ms": 5000,
                    "gt_serving_team": "A",
                    "gt_side_switch": True,
                },
                {
                    "video_content_hash": "hash1",
                    "rally_start_ms": 6000,
                    "rally_end_ms": 10000,
                    "gt_side_switch": False,
                },
                {
                    "video_content_hash": "hash1",
                    "rally_start_ms": 9999,
                    "rally_end_ms": 99999,
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

        # Helper, per rally: SELECT composite-key, then per-field NULL-only
        # UPDATE. Rally 1 (hash1, 1000, 5000) has both fields: serving UPDATE
        # rowcount=1, side_switch UPDATE rowcount=0 (already set) → counts as
        # restored. Rally 2 has only side_switch → UPDATE rowcount=1 → counts.
        # Rally 3 has no matching (start_ms, end_ms) → lookup returns None,
        # error recorded, no UPDATE attempted.
        lookup_results = iter([
            ("rally-A-db-uuid",),
            ("rally-B-db-uuid",),
            None,
        ])
        update_rowcounts = iter([1, 0, 1])

        cursor = MagicMock()

        def execute_side_effect(sql: str, _params: Any = None) -> None:
            sql_lower = sql.lower()
            if "select r.id" in sql_lower:
                cursor._last = "select"
            elif "update rallies" in sql_lower:
                cursor._last = "update"
                cursor.rowcount = next(update_rowcounts)
            else:
                cursor._last = "other"

        cursor.execute.side_effect = execute_side_effect
        cursor.fetchone.side_effect = lambda: next(lookup_results)

        conn = MagicMock()
        conn.cursor.return_value.__enter__.return_value = cursor
        conn.cursor.return_value.__exit__.return_value = False

        result = RestoreResult()
        _restore_score_ground_truth(conn, path, result)

        assert result.score_gt_restored == 2
        # Error mentions the timing (not a UUID)
        assert any("hash1" in e and "9999" in e for e in result.errors)

    def test_silent_skip_when_file_missing(self, tmp_path: Path) -> None:
        from rallycut.training.restore import (
            RestoreResult,
            _restore_score_ground_truth,
        )

        result = RestoreResult()
        _restore_score_ground_truth(MagicMock(), tmp_path / "missing.json", result)

        assert result.score_gt_restored == 0
        assert result.errors == []
