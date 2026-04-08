"""Unit tests for the shared player_matching_gt_json loader."""

from __future__ import annotations

from rallycut.evaluation.gt_loader import (
    IOU_THRESHOLD,
    load_player_matching_gt,
)


def _pos(frame: int, track_id: int, x: float, y: float,
         w: float = 0.05, h: float = 0.2) -> dict:
    return {
        "frameNumber": frame, "trackId": track_id,
        "x": x, "y": y, "width": w, "height": h,
        "confidence": 0.9,
    }


def test_resolves_labels_to_tracks_by_iou() -> None:
    positions = {
        "rally-1": [
            _pos(10, 7, 0.40, 0.60),  # player 1 is now track 7
            _pos(10, 8, 0.60, 0.60),  # player 2 is now track 8
            _pos(10, 9, 0.30, 0.30),  # player 3 is now track 9
            _pos(10, 11, 0.70, 0.30),  # player 4 is now track 11
        ],
    }
    gt_json = {
        "rallies": {
            "rally-1": {
                "labels": [
                    {"playerId": 1, "frame": 10,
                     "cx": 0.40, "cy": 0.60, "w": 0.05, "h": 0.2},
                    {"playerId": 2, "frame": 10,
                     "cx": 0.60, "cy": 0.60, "w": 0.05, "h": 0.2},
                    {"playerId": 3, "frame": 10,
                     "cx": 0.30, "cy": 0.30, "w": 0.05, "h": 0.2},
                    {"playerId": 4, "frame": 10,
                     "cx": 0.70, "cy": 0.30, "w": 0.05, "h": 0.2},
                ],
            },
        },
        "sideSwitches": [3],
        "excludedRallies": ["rally-X"],
    }
    gt = load_player_matching_gt(gt_json, positions_lookup=positions.get)
    assert gt.rallies == {"rally-1": {"7": 1, "8": 2, "9": 3, "11": 4}}
    assert gt.side_switches == [3]
    assert gt.excluded_rallies == ["rally-X"]
    assert gt.warnings == []


def test_snake_case_side_switches_accepted() -> None:
    positions = {"r": [_pos(0, 1, 0.5, 0.5)]}
    gt = load_player_matching_gt({
        "rallies": {"r": {"labels": [
            {"playerId": 1, "frame": 0,
             "cx": 0.5, "cy": 0.5, "w": 0.05, "h": 0.2},
        ]}},
        "side_switches": [7],
    }, positions_lookup=positions.get)
    assert gt.side_switches == [7]


def test_drops_label_when_iou_below_threshold() -> None:
    positions = {"r": [_pos(0, 1, 0.0, 0.0)]}  # far from label bbox
    gt = load_player_matching_gt({
        "rallies": {"r": {"labels": [
            {"playerId": 1, "frame": 0,
             "cx": 0.9, "cy": 0.9, "w": 0.05, "h": 0.2},
        ]}},
    }, positions_lookup=positions.get)
    assert "r" not in gt.rallies
    assert any("IoU" in w for w in gt.warnings)


def test_partial_resolution_keeps_rally() -> None:
    positions = {"r": [_pos(0, 1, 0.40, 0.60)]}
    gt = load_player_matching_gt({
        "rallies": {"r": {"labels": [
            {"playerId": 1, "frame": 0,
             "cx": 0.40, "cy": 0.60, "w": 0.05, "h": 0.2},
            {"playerId": 2, "frame": 0,
             "cx": 0.99, "cy": 0.99, "w": 0.05, "h": 0.2},
        ]}},
    }, positions_lookup=positions.get)
    assert gt.rallies == {"r": {"1": 1}}
    assert len(gt.warnings) == 1


def test_missing_positions_skips_rally() -> None:
    gt = load_player_matching_gt({
        "rallies": {"r": {"labels": [
            {"playerId": 1, "frame": 0,
             "cx": 0.5, "cy": 0.5, "w": 0.05, "h": 0.2},
        ]}},
    }, positions_lookup=lambda _rid: None)
    assert gt.rallies == {}
    assert any("no positions" in w for w in gt.warnings)


def test_none_returns_empty() -> None:
    gt = load_player_matching_gt(None)
    assert gt.rallies == {} and gt.side_switches == []


def test_iou_threshold_sane() -> None:
    assert 0 < IOU_THRESHOLD < 1


def test_two_labels_resolving_to_same_track_warns_and_drops_second() -> None:
    # Both labels at the same frame, both closest to track 1 — the second
    # should be dropped with a warning rather than silently overwriting.
    positions = {"r": [_pos(0, 1, 0.50, 0.50)]}
    gt = load_player_matching_gt({
        "rallies": {"r": {"labels": [
            {"playerId": 1, "frame": 0,
             "cx": 0.50, "cy": 0.50, "w": 0.05, "h": 0.2},
            {"playerId": 2, "frame": 0,
             "cx": 0.50, "cy": 0.50, "w": 0.05, "h": 0.2},
        ]}},
    }, positions_lookup=positions.get)
    assert gt.rallies == {"r": {"1": 1}}  # first wins
    assert any("resolved to track 1" in w for w in gt.warnings)
