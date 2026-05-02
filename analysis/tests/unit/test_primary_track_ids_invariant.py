"""Tests for the primary_track_ids persistence invariant.

The invariant: any primary_track_ids list that crosses the persist
boundary must be 0..max_players DISTINCT NON-NEGATIVE integers.
Negatives (BoT-SORT's `-1` unmatched-detection sentinel) and duplicates
silently cause the cross-rally matcher to emit fewer than 4 PIDs —
see commit 0296a7f for the original failure mode and
scripts/repair_primary_track_ids.py for the migration tool.

Three layers of defense, one test each:
  1. Write-side: `validate_primary_track_ids` raises on negatives/dups.
  2. Read-side: `load_rallies_for_video` auto-cleans legacy rows.
  3. Filter contract: `PlayerFilter.identify_primary_tracks` never
     emits negatives or duplicates by construction.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import pytest

from rallycut.tracking.player_filter import (
    PlayerFilterConfig,
    TrackStats,
    identify_primary_tracks,
    validate_primary_track_ids,
)


# ---------------------------------------------------------------------------
# Layer 1: write-side assertion
# ---------------------------------------------------------------------------


class TestValidatePrimaryTrackIds:
    def test_accepts_clean_list(self) -> None:
        assert validate_primary_track_ids([1, 3, 9, 10]) == [1, 3, 9, 10]

    def test_accepts_empty(self) -> None:
        assert validate_primary_track_ids([]) == []

    def test_rejects_negative_sentinel(self) -> None:
        with pytest.raises(ValueError, match="negatives"):
            validate_primary_track_ids([1, -1, -1, 10])

    def test_rejects_duplicates(self) -> None:
        with pytest.raises(ValueError, match="duplicates"):
            validate_primary_track_ids([1, 2, 2, 3])

    def test_error_includes_context(self) -> None:
        with pytest.raises(ValueError, match="my_caller"):
            validate_primary_track_ids([-1], context="my_caller")

    def test_accepts_set_input(self) -> None:
        # set is unordered but distinct → no dups by construction.
        out = validate_primary_track_ids({1, 3, 9, 10})
        assert sorted(out) == [1, 3, 9, 10]


# ---------------------------------------------------------------------------
# Layer 2: read-side auto-clean
# ---------------------------------------------------------------------------


class TestLoadRalliesAutoClean:
    """Test that load_rallies_for_video strips sentinels + dups in-memory.

    We don't touch the DB here — instead we patch get_connection's
    `cur.fetchall()` to return synthetic rows and assert the resulting
    RallyTrackData has clean primary_track_ids and a warning was logged.
    """

    def test_strips_negatives_and_dedupes(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    ) -> None:
        from rallycut.evaluation.tracking import db as db_mod

        # Synthetic row matching the SELECT in load_rallies_for_video.
        # Layout: (rally_id, video_id, start_ms, end_ms, positions_json,
        # primary_track_ids, court_split_y, ball_positions_json, actions_json)
        synth_rows = [(
            "rally-1", "video-1", 0, 1000,
            [{"frameNumber": 0, "trackId": 1, "x": 0.5, "y": 0.5,
              "width": 0.1, "height": 0.2, "confidence": 0.9}],
            [1, -1, -1, 10],  # corrupted: 2 sentinels
            0.5, None, None,
        )]

        class FakeCursor:
            def execute(self, *_a: Any, **_k: Any) -> None:
                pass

            def fetchall(self) -> list[tuple[Any, ...]]:
                return synth_rows

            def __enter__(self) -> "FakeCursor":
                return self

            def __exit__(self, *_a: Any) -> None:
                pass

        class FakeConn:
            def cursor(self) -> FakeCursor:
                return FakeCursor()

            def __enter__(self) -> "FakeConn":
                return self

            def __exit__(self, *_a: Any) -> None:
                pass

        monkeypatch.setattr(db_mod, "get_connection", lambda: FakeConn())

        with caplog.at_level("WARNING", logger="rallycut.evaluation.tracking.db"):
            results = db_mod.load_rallies_for_video("video-1")

        assert len(results) == 1
        # No -1, no dups, original positives preserved in order.
        assert results[0].primary_track_ids == [1, 10]
        assert any(
            "Auto-cleaned stale primary_track_ids" in rec.message
            for rec in caplog.records
        )

    def test_clean_rows_are_passthrough(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    ) -> None:
        from rallycut.evaluation.tracking import db as db_mod

        synth_rows = [(
            "rally-1", "video-1", 0, 1000,
            [{"frameNumber": 0, "trackId": 1, "x": 0.5, "y": 0.5,
              "width": 0.1, "height": 0.2, "confidence": 0.9}],
            [1, 3, 9, 10],
            0.5, None, None,
        )]

        class FakeCursor:
            def execute(self, *_a: Any, **_k: Any) -> None:
                pass

            def fetchall(self) -> list[tuple[Any, ...]]:
                return synth_rows

            def __enter__(self) -> "FakeCursor":
                return self

            def __exit__(self, *_a: Any) -> None:
                pass

        class FakeConn:
            def cursor(self) -> FakeCursor:
                return FakeCursor()

            def __enter__(self) -> "FakeConn":
                return self

            def __exit__(self, *_a: Any) -> None:
                pass

        monkeypatch.setattr(db_mod, "get_connection", lambda: FakeConn())

        with caplog.at_level("WARNING", logger="rallycut.evaluation.tracking.db"):
            results = db_mod.load_rallies_for_video("video-1")

        assert results[0].primary_track_ids == [1, 3, 9, 10]
        assert not any(
            "Auto-cleaned" in rec.message for rec in caplog.records
        )


# ---------------------------------------------------------------------------
# Layer 3: filter contract
# ---------------------------------------------------------------------------


def _make_track_stats(
    track_id: int,
    presence_rate: float,
    avg_x: float = 0.5,
    *,
    is_referee: bool = False,
    avg_bbox_area: float = 0.005,
    position_spread: float = 0.05,
    ball_proximity_score: float = 0.5,
    total_frames: int = 100,
) -> TrackStats:
    # presence_rate is a property derived from frame_count / total_frames.
    return TrackStats(
        track_id=track_id,
        frame_count=int(presence_rate * total_frames),
        total_frames=total_frames,
        avg_bbox_area=avg_bbox_area,
        avg_confidence=0.9,
        ball_proximity_score=ball_proximity_score,
        first_frame=0,
        last_frame=total_frames,
        position_spread=position_spread,
        avg_x=avg_x,
        is_likely_referee=is_referee,
    )


class TestFilterContract:
    def test_identify_primary_tracks_emits_only_non_negatives(self) -> None:
        # Even if (hypothetically) a -1 keyed entry slips into track_stats,
        # identify_primary_tracks must NOT include it. Today's filter excludes
        # negatives via sideline_min/max checks anyway; this asserts the
        # invariant at the function boundary so a future regression fails loud.
        track_stats = {
            -1: _make_track_stats(-1, 0.5, avg_x=0.4),
            1: _make_track_stats(1, 0.9, avg_x=0.5),
            2: _make_track_stats(2, 0.8, avg_x=0.45),
            3: _make_track_stats(3, 0.7, avg_x=0.55),
            4: _make_track_stats(4, 0.6, avg_x=0.5),
        }

        config = PlayerFilterConfig()
        result = identify_primary_tracks(track_stats, config)

        # Result is a set so dedup is automatic; assert the validator
        # accepts it (no negatives).
        validate_primary_track_ids(result, context="contract test")
        assert all(t >= 0 for t in result)
