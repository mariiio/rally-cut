from rallycut.cli.commands.migrate_action_gt import (
    _ball_at,
    _bbox_at,
    _team_for,
    build_label_row,
)


def test_bbox_at_finds_match_with_x1y1x2y2_shape() -> None:
    raw = [{"frameNumber": 50, "trackId": 7, "bbox": {"x1": 0.1, "y1": 0.1, "x2": 0.2, "y2": 0.3}}]
    assert _bbox_at(raw, 50, 7) == {"x1": 0.1, "y1": 0.1, "x2": 0.2, "y2": 0.3}


def test_bbox_at_converts_xywh_shape() -> None:
    raw = [{"frameNumber": 50, "trackId": 7, "x": 0.1, "y": 0.1, "width": 0.1, "height": 0.2}]
    result = _bbox_at(raw, 50, 7)
    assert result is not None
    assert abs(result["x1"] - 0.1) < 1e-9
    assert abs(result["y1"] - 0.1) < 1e-9
    assert abs(result["x2"] - 0.2) < 1e-9
    assert abs(result["y2"] - 0.3) < 1e-9


def test_bbox_at_returns_none_when_track_not_at_frame() -> None:
    raw = [{"frameNumber": 50, "trackId": 7, "bbox": {"x1": 0, "y1": 0, "x2": 1, "y2": 1}}]
    assert _bbox_at(raw, 50, 9) is None
    assert _bbox_at(raw, 100, 7) is None


def test_ball_at() -> None:
    ball = [{"frameNumber": 50, "x": 0.5, "y": 0.4}]
    assert _ball_at(ball, 50) == (0.5, 0.4)
    assert _ball_at(ball, 99) is None
    assert _ball_at(None, 50) is None


def test_team_for_only_returns_when_in_primary_ids_and_assignments() -> None:
    assert _team_for(7, [7, 8], {"7": "A", "8": "B"}) == "A"
    assert _team_for(7, [], {"7": "A"}) is None  # not in primary
    assert _team_for(7, [7], {}) is None  # not in assignments
    assert _team_for(7, [7], {"7": "X"}) is None  # invalid team value


def test_build_label_row_snapshot_exact() -> None:
    row = build_label_row(
        {"frame": 50, "action": "serve", "trackId": 7},
        raw_positions=[{"frameNumber": 50, "trackId": 7, "x": 0.1, "y": 0.1, "width": 0.1, "height": 0.2}],
        positions=None,
        ball_positions=[{"frameNumber": 50, "x": 0.5, "y": 0.4}],
        primary_track_ids=[7],
        team_assignments={"7": "A"},
    )
    assert row is not None
    assert row["action"] == "SERVE"
    assert row["snapshot_bbox_x1"] == 0.1
    assert abs(row["snapshot_bbox_x2"] - 0.2) < 1e-9
    assert row["snapshot_ball_x"] == 0.5
    assert row["snapshot_team"] == "A"
    assert row["snapshot_track_id"] == 7
    assert row["resolved_track_id"] == 7
    assert row["resolved_source"] == "SNAPSHOT_EXACT"


def test_build_label_row_unresolved_when_no_position_at_frame() -> None:
    row = build_label_row(
        {"frame": 50, "action": "serve", "trackId": 99},
        raw_positions=[],
        positions=None,
        ball_positions=[],
        primary_track_ids=[7],
        team_assignments={"7": "A"},
    )
    assert row is not None
    assert row["snapshot_bbox_x1"] is None
    assert row["snapshot_track_id"] == 99
    assert row["resolved_track_id"] is None
    assert row["resolved_source"] == "UNRESOLVED"


def test_build_label_row_falls_back_to_label_ball_xy_when_unsnapshotted() -> None:
    """Legacy labels with ballX/ballY directly on the GT object."""
    row = build_label_row(
        {"frame": 50, "action": "attack", "trackId": 7, "ballX": 0.7, "ballY": 0.6},
        raw_positions=None,
        positions=None,
        ball_positions=None,
        primary_track_ids=None,
        team_assignments=None,
    )
    assert row is not None
    assert row["snapshot_ball_x"] == 0.7
    assert row["snapshot_ball_y"] == 0.6


def test_build_label_row_returns_none_on_malformed() -> None:
    assert build_label_row({"action": "serve"}, None, None, None, None, None) is None  # no frame
    assert build_label_row({"frame": 0, "action": "bogus"}, None, None, None, None, None) is None  # invalid action


def test_build_label_row_falls_back_to_positions_by_player_track_id() -> None:
    """Legacy labels with only playerTrackId snapshot from positions_json."""
    row = build_label_row(
        {"frame": 50, "action": "serve", "playerTrackId": 2},
        raw_positions=[],
        positions=[{"frameNumber": 50, "trackId": 2, "x": 0.3, "y": 0.3, "width": 0.1, "height": 0.2}],
        ball_positions=[{"frameNumber": 50, "x": 0.5, "y": 0.4}],
        primary_track_ids=[1, 2, 3, 4],
        team_assignments={"2": "B"},
    )
    assert row is not None
    assert row["snapshot_bbox_x1"] == 0.3
    assert row["snapshot_team"] == "B"
    assert row["snapshot_track_id"] == 2
    assert row["resolved_track_id"] == 2
    assert row["resolved_source"] == "SNAPSHOT_EXACT"


def test_build_label_row_prefers_trackid_over_player_track_id_when_both_present() -> None:
    """When both ids are set and both could resolve, prefer trackId lookup in raw_positions."""
    row = build_label_row(
        {"frame": 50, "action": "serve", "trackId": 11, "playerTrackId": 2},
        raw_positions=[{"frameNumber": 50, "trackId": 11, "x": 0.1, "y": 0.1, "width": 0.1, "height": 0.2}],
        positions=[{"frameNumber": 50, "trackId": 2, "x": 0.9, "y": 0.9, "width": 0.05, "height": 0.05}],
        ball_positions=[],
        primary_track_ids=[1, 2, 3, 4],
        team_assignments={"2": "A"},
    )
    assert row is not None
    # Snapshot came from raw_positions[trackId=11], NOT positions[trackId=2]
    assert row["snapshot_bbox_x1"] == 0.1
    assert row["resolved_track_id"] == 11
