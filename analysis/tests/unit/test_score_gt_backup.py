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
    def test_returns_dict_with_filtered_rallies(self, fake_conn: MagicMock) -> None:
        from rallycut.cli.commands.train import _export_score_ground_truth

        _set_rows(
            fake_conn,
            [
                ("hashA", "rally-1", "video-1", "A"),
                ("hashB", "rally-2", "video-2", "B"),
                ("hashC", "rally-3", "video-3", "A"),  # not in dataset
            ],
        )

        with patch(
            "rallycut.cli.commands.train.get_connection", return_value=fake_conn
        ):
            result = _export_score_ground_truth({"hashA", "hashB"})

        assert result is not None
        assert result["stats"] == {"total_rallies": 2, "total_videos": 2}
        rally_ids = {r["rally_id"] for r in result["rallies"]}
        assert rally_ids == {"rally-1", "rally-2"}
        for entry in result["rallies"]:
            assert set(entry.keys()) == {
                "rally_id",
                "video_id",
                "content_hash",
                "gt_serving_team",
            }
            assert entry["gt_serving_team"] in {"A", "B"}

    def test_returns_none_when_no_rows(self, fake_conn: MagicMock) -> None:
        from rallycut.cli.commands.train import _export_score_ground_truth

        _set_rows(fake_conn, [])
        with patch(
            "rallycut.cli.commands.train.get_connection", return_value=fake_conn
        ):
            assert _export_score_ground_truth({"hashA"}) is None

    def test_returns_none_when_all_filtered_out(
        self, fake_conn: MagicMock
    ) -> None:
        from rallycut.cli.commands.train import _export_score_ground_truth

        _set_rows(fake_conn, [("hashZ", "rally-9", "video-9", "A")])
        with patch(
            "rallycut.cli.commands.train.get_connection", return_value=fake_conn
        ):
            assert _export_score_ground_truth({"hashA"}) is None
