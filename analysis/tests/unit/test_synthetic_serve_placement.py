"""Unit tests for synthetic_serve_placement.pick_synthetic_serve_frame."""

from __future__ import annotations

import numpy as np

from rallycut.tracking.synthetic_serve_placement import (
    MAX_PRESERVE_FRAMES,
    SEARCH_GUARD,
    SERVE_SEQ_FLOOR,
    pick_synthetic_serve_frame,
)


def test_returns_seq_peak_frame() -> None:
    """Strong seq peak in window -> return that frame."""
    seq = np.zeros((7, 400))
    seq[1, 80] = 0.85  # serve-class peak at frame 80 (index 1 in seq_probs)
    result = pick_synthetic_serve_frame(
        sequence_probs=seq,
        rally_start_frame=0,
        first_contact_frame=200,
    )
    assert result == 80, result


def test_returns_none_when_seq_peak_below_floor() -> None:
    """Seq peak below SERVE_SEQ_FLOOR -> None."""
    seq = np.zeros((7, 400))
    seq[1, 80] = SERVE_SEQ_FLOOR - 0.01
    result = pick_synthetic_serve_frame(
        sequence_probs=seq,
        rally_start_frame=0,
        first_contact_frame=200,
    )
    assert result is None


def test_returns_none_when_seq_probs_empty() -> None:
    """All-zero seq probs -> argmax falls below floor -> None."""
    seq = np.zeros((7, 400))
    result = pick_synthetic_serve_frame(
        sequence_probs=seq,
        rally_start_frame=0,
        first_contact_frame=200,
    )
    assert result is None


def test_returns_none_when_search_window_collapses() -> None:
    """rally_start within SEARCH_GUARD of first_contact -> nothing to search."""
    seq = np.zeros((7, 400))
    seq[1, 50] = 0.85
    result = pick_synthetic_serve_frame(
        sequence_probs=seq,
        rally_start_frame=100,
        first_contact_frame=100 + SEARCH_GUARD - 1,
    )
    assert result is None


def test_clamps_when_picked_frame_too_early() -> None:
    """Picked frame > MAX_PRESERVE_FRAMES before first_contact -> clamp."""
    seq = np.zeros((7, 400))
    seq[1, 10] = 0.85  # very early peak
    result = pick_synthetic_serve_frame(
        sequence_probs=seq,
        rally_start_frame=0,
        first_contact_frame=300,
    )
    assert result == 300 - MAX_PRESERVE_FRAMES, result


def test_picks_only_within_search_window() -> None:
    """Seq peaks outside [rally_start, first_contact - SEARCH_GUARD] are ignored."""
    seq = np.zeros((7, 400))
    seq[1, 350] = 0.95   # outside (after first_contact)
    seq[1, 80] = 0.60    # inside, below the 350 peak globally but should still be argmax inside window
    result = pick_synthetic_serve_frame(
        sequence_probs=seq,
        rally_start_frame=0,
        first_contact_frame=200,
    )
    assert result == 80, result


def test_constants_present() -> None:
    """Smoke check: the constants are exposed for monkey-patching."""
    assert isinstance(SERVE_SEQ_FLOOR, float)
    assert isinstance(SEARCH_GUARD, int)
    assert isinstance(MAX_PRESERVE_FRAMES, int)


import pytest


@pytest.mark.integration
def test_synthetic_serve_lands_at_real_frame_for_fb7f9c23() -> None:
    """End-to-end: re-running classify_rally_actions on fb7f9c23 should
    place the synthetic serve within +-15 frames of GT serve frame 154.

    This rally is the canonical "serve missed by detector, fallback fires"
    case. With the v1.1 placement, MS-TCN++ serve-class peak should
    pinpoint the actual serve frame.
    """
    from rallycut.evaluation.tracking.db import get_connection
    from rallycut.tracking.action_classifier import classify_rally_actions
    from rallycut.tracking.ball_tracker import BallPosition
    from rallycut.tracking.contact_detector import (
        ContactDetectionConfig,
        detect_contacts,
    )
    from rallycut.tracking.player_tracker import PlayerPosition
    from rallycut.tracking.sequence_action_runtime import get_sequence_probs

    rally_id = "fb7f9c23-3544-48bd-910d-10a8f12fd594"

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT pt.fps, pt.frame_count, pt.court_split_y,
                          pt.ball_positions_json, pt.positions_json,
                          pt.actions_json, pt.primary_track_ids
                   FROM player_tracks pt JOIN rallies r ON pt.rally_id = r.id
                   WHERE r.id = %s""",
                [rally_id],
            )
            row = cur.fetchone()
    assert row is not None, "fixture rally fb7f9c23 missing from DB"
    fps, frame_count, court_split_y, bp_json, pp_json, aj, primary_raw = row

    bp = [
        BallPosition(
            frame_number=int(b.get("frameNumber", 0)),
            x=float(b.get("x", 0.0)),
            y=float(b.get("y", 0.0)),
            confidence=float(b.get("confidence", 0.0)),
        )
        for b in (bp_json or [])
        if isinstance(b, dict)
    ]
    pp = [
        PlayerPosition(
            frame_number=int(p.get("frameNumber", 0)),
            track_id=int(p.get("trackId", -1)),
            x=float(p.get("x", 0.0)),
            y=float(p.get("y", 0.0)),
            width=float(p.get("width", 0.0)),
            height=float(p.get("height", 0.0)),
            confidence=float(p.get("confidence", 0.0)),
            keypoints=p.get("keypoints"),
        )
        for p in (pp_json or [])
        if isinstance(p, dict)
    ]
    ta_str = (aj or {}).get("teamAssignments", {}) or {}
    ta_int: dict[int, int] = {}
    for k, v in ta_str.items():
        if v == "A":
            ta_int[int(k)] = 0
        elif v == "B":
            ta_int[int(k)] = 1

    seq_probs = get_sequence_probs(
        bp, pp, court_split_y, frame_count or 0, ta_int, calibrator=None,
    )
    if seq_probs is None:
        pytest.skip("MS-TCN++ weights unavailable in this environment")

    contact_seq = detect_contacts(
        ball_positions=bp, player_positions=pp,
        config=ContactDetectionConfig(),
        net_y=court_split_y, frame_count=frame_count or None,
        team_assignments=ta_int, court_calibrator=None,
        sequence_probs=seq_probs,
        primary_track_ids=list(primary_raw or []) or None,
    )

    rally_actions = classify_rally_actions(
        contact_seq,
        team_assignments=ta_int,
        calibrator=None,
        sequence_probs=seq_probs,
    )
    serves = [a for a in rally_actions.actions if a.action_type.value == "serve"]
    assert serves, "No serve action in classified rally"
    serve = serves[0]
    # GT serve at frame 154; require the synthetic to land within +-15.
    assert abs(serve.frame - 154) <= 15, (
        f"Synthetic serve at frame {serve.frame}; GT 154 (off by "
        f"{serve.frame - 154})"
    )
    assert serve.is_synthetic, "Expected synthetic (since detector missed it)"
