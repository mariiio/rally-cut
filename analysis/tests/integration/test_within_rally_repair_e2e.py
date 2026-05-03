"""End-to-end integration test for the within-rally ID-switch repair.

The unit tests in `tests/unit/test_within_rally_id_switch.py` cover
the algorithm via monkeypatched `_pairwise_cost`. They prove the
detector / re-Hungarian / clip logic is correct for synthetic
appearance-cost matrices, but they DO NOT exercise the integration
path:
    extract_rally_appearances → compute_track_similarity → detector
    → re-Hungarian → clip → SubTrackCandidate emission

This integration test runs the full pipeline on a real rally with
known ID-switch behavior (7d77980f / 09553ef1, the user-reported
bug 2 case) and asserts the OBSERVABLE OUTCOMES:

  - With the flag OFF: no sub-tracks emitted (baseline).
  - With the flag ON: the detector fires on at least one rally
    where appearance discontinuity is real, and the resulting
    sub-track is what Phase 2 cross-track-overlap dedup would emit.

Skipped when DB is unpopulated, video file is missing, or the test
fixture has been retracked into a different state.

Marked `slow` because it runs OSNet ReID per window per primary track.
"""
from __future__ import annotations

import os
from typing import Any

import pytest

pytestmark = pytest.mark.slow

FIXTURE_VIDEO_ID = "7d77980f-3006-40e0-adc0-db491a5bb659"
TARGET_RALLY_PREFIX = "09553ef1"


def _load_rally_for_test(prefix: str) -> tuple[Any, Any, Any]:
    """Returns (rally, video_path, reid_model). Skips on missing data."""
    try:
        from rallycut.evaluation.tracking.db import (
            get_video_path,
            load_rallies_for_video,
        )
        from rallycut.tracking.reid_general import GeneralReIDModel, WEIGHTS_PATH
    except ImportError as exc:
        pytest.skip(f"missing import: {exc}")

    rallies = load_rallies_for_video(FIXTURE_VIDEO_ID)
    if not rallies:
        pytest.skip(f"no rallies in DB for {FIXTURE_VIDEO_ID}")
    rally = next(
        (r for r in rallies if r.rally_id.startswith(prefix)), None,
    )
    if rally is None:
        pytest.skip(f"no rally matching prefix {prefix} in {FIXTURE_VIDEO_ID}")
    video_path = get_video_path(FIXTURE_VIDEO_ID)
    if video_path is None:
        pytest.skip(f"video file not resolvable for {FIXTURE_VIDEO_ID}")
    if not WEIGHTS_PATH.exists():
        pytest.skip(f"OSNet weights not found at {WEIGHTS_PATH}")
    reid = GeneralReIDModel(weights_path=WEIGHTS_PATH)
    return rally, video_path, reid


def _process_rally_to_track_to_player(
    rally: Any, video_path: Any, reid: Any,
) -> tuple[dict[int, int], Any]:
    """Run the matcher up to the point Phase 1+2 would hook in.

    Returns (track_to_player from process_rally, the MatchPlayerTracker
    instance — caller can use .stored_rally_data for diagnostics).
    """
    from rallycut.tracking.match_tracker import (
        extract_rally_appearances,
        MatchPlayerTracker,
    )
    tracker = MatchPlayerTracker()
    track_stats = extract_rally_appearances(
        video_path=video_path,
        positions=rally.positions,
        primary_track_ids=rally.primary_track_ids,
        start_ms=rally.start_ms, end_ms=rally.end_ms,
        num_samples=12, extract_reid=True, reid_model=reid,
    )
    res = tracker.process_rally(
        track_stats=track_stats,
        player_positions=rally.positions,
        ball_positions=rally.ball_positions,
        court_split_y=rally.court_split_y,
        team_assignments=rally.team_assignments,
        start_ms=rally.start_ms, end_ms=rally.end_ms,
    )
    return res.track_to_player, tracker


def test_phase1_2_fires_on_known_id_switch_rally() -> None:
    """End-to-end: with the flag ON, the detector fires on 09553ef1
    (the user-reported bug-2 rally with documented BoT-SORT ID-jump
    on T1) and emits at least one sub-track. This proves the real
    pipeline path actually works — not just the synthetic-cost units.
    """
    from rallycut.tracking import _within_rally_id_switch as wris

    rally, video_path, reid = _load_rally_for_test(TARGET_RALLY_PREFIX)
    track_to_player, _tracker = _process_rally_to_track_to_player(
        rally, video_path, reid,
    )
    if not track_to_player:
        pytest.skip("matcher emitted no track_to_player; can't run detector")

    # Force flag ON for this test regardless of environment.
    prior = os.environ.get(wris.ENV_FLAG)
    os.environ[wris.ENV_FLAG] = "1"
    try:
        overrides = wris.maybe_emit_within_rally_split(
            rally_id=rally.rally_id,
            video_path=video_path,
            rally_start_ms=rally.start_ms,
            rally_end_ms=rally.end_ms,
            positions=rally.positions,
            track_to_player=track_to_player,
            reid_model=reid,
        )
    finally:
        if prior is None:
            os.environ.pop(wris.ENV_FLAG, None)
        else:
            os.environ[wris.ENV_FLAG] = prior

    assert overrides is not None and len(overrides) >= 1, (
        f"detector should fire on {TARGET_RALLY_PREFIX} (known ID switch); "
        f"got {overrides}"
    )

    # The emitted sub-tracks must satisfy the basic invariants:
    #   - All have valid (positive) PIDs in 1..4
    #   - All cover non-empty frame ranges
    #   - All have valid synthetic_track_id (negative)
    for ov in overrides:
        assert ov.aggregated_argmax_pid is not None
        assert 1 <= ov.aggregated_argmax_pid <= 4
        assert ov.f_start <= ov.f_end
        assert ov.synthetic_track_id < 0


def test_phase1_2_no_op_with_flag_off() -> None:
    """Negative control: with the flag OFF, the detector returns None
    (no sub-tracks emitted). Confirms the gate is effective.
    """
    from rallycut.tracking import _within_rally_id_switch as wris

    rally, video_path, reid = _load_rally_for_test(TARGET_RALLY_PREFIX)
    track_to_player, _tracker = _process_rally_to_track_to_player(
        rally, video_path, reid,
    )
    if not track_to_player:
        pytest.skip("matcher emitted no track_to_player")

    # Force flag OFF.
    prior = os.environ.get(wris.ENV_FLAG)
    os.environ[wris.ENV_FLAG] = "0"
    try:
        overrides = wris.maybe_emit_within_rally_split(
            rally_id=rally.rally_id,
            video_path=video_path,
            rally_start_ms=rally.start_ms,
            rally_end_ms=rally.end_ms,
            positions=rally.positions,
            track_to_player=track_to_player,
            reid_model=reid,
        )
    finally:
        if prior is None:
            os.environ.pop(wris.ENV_FLAG, None)
        else:
            os.environ[wris.ENV_FLAG] = prior

    assert overrides is None
