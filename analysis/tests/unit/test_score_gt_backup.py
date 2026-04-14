"""Unit tests for score-tracking GT backup helpers."""

from __future__ import annotations

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
