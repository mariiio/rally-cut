"""Unit tests for the v1.3 serve-prepend gate.

The predicate fires when MS-TCN++ has a strong serve-class peak before
the first classified action. Five conjunctive conditions; see
`docs/superpowers/specs/2026-05-11-serve-peak-prepend-design.md`.
"""
from __future__ import annotations

import numpy as np

from rallycut.actions.trajectory_features import ACTION_TYPES
from rallycut.tracking.serve_prepend import (
    SERVE_PREPEND_CLASSIFIER_CONF_CEIL,
    SERVE_PREPEND_FIRST_ACTION_SERVE_CEIL,
    SERVE_PREPEND_GUARD_FRAMES,
    SERVE_PREPEND_MIN_GAP,
    SERVE_PREPEND_PEAK_FLOOR,
    should_prepend_serve,
)

SERVE_IDX = ACTION_TYPES.index("serve") + 1


def _seq_with_serve_peak(peak_frame: int, peak_prob: float, length: int) -> np.ndarray:
    """Build a fake MS-TCN++ output: serve-class peak at `peak_frame`, low elsewhere."""
    n_classes = len(ACTION_TYPES) + 1  # +1 background class
    seq = np.full((n_classes, length), 0.01, dtype=np.float32)
    # Bell around peak_frame
    for f in range(max(0, peak_frame - 5), min(length, peak_frame + 6)):
        d = abs(f - peak_frame)
        seq[SERVE_IDX, f] = max(0.01, peak_prob * (1 - d / 7.0))
    seq[SERVE_IDX, peak_frame] = peak_prob
    return seq


class TestShouldPrependServe:
    def test_textbook_fire(self) -> None:
        seq = _seq_with_serve_peak(peak_frame=110, peak_prob=0.99, length=500)
        result = should_prepend_serve(
            sequence_probs=seq,
            first_action_frame=426,
            first_action_serve_prob=0.02,
            first_action_classifier_confidence=0.30,
            rally_start_frame=0,
        )
        assert result == 110

    def test_none_sequence_probs_returns_none(self) -> None:
        assert should_prepend_serve(
            sequence_probs=None,
            first_action_frame=200,
            first_action_serve_prob=0.05,
            first_action_classifier_confidence=0.30,
            rally_start_frame=0,
        ) is None

    def test_first_action_serve_prob_too_high_returns_none(self) -> None:
        """If MS-TCN serve probability at the first action is high, don't override."""
        seq = _seq_with_serve_peak(peak_frame=50, peak_prob=0.99, length=300)
        assert should_prepend_serve(
            sequence_probs=seq,
            first_action_frame=200,
            first_action_serve_prob=SERVE_PREPEND_FIRST_ACTION_SERVE_CEIL,
            first_action_classifier_confidence=0.30,
            rally_start_frame=0,
        ) is None

    def test_high_classifier_confidence_vetoes_fire(self) -> None:
        """If action_classifier was confident in its serve label
        (confidence >= SERVE_PREPEND_CLASSIFIER_CONF_CEIL), the gate must
        NOT fire even when MS-TCN serve probability is low.

        The gate's semantic precondition is "first action is mis-labeled
        as serve". Single-model proxies for mis-labeling are fragile: when
        MS-TCN itself is wrong, the existing first_action_serve_prob check
        under-vetoes. Requiring DUAL-SIGNAL agreement (both MS-TCN AND
        action_classifier indicate mis-labeling) makes the gate robust.

        Bug case (06f0b063 rally 8): real serve at f=94 with classifier
        confidence 0.65 was overridden by MS-TCN's false-positive serve
        peak at f=4 (prob >= 0.95), producing a duplicate serve.
        """
        seq = _seq_with_serve_peak(peak_frame=10, peak_prob=0.99, length=200)
        assert should_prepend_serve(
            sequence_probs=seq,
            first_action_frame=100,
            first_action_serve_prob=0.05,
            first_action_classifier_confidence=SERVE_PREPEND_CLASSIFIER_CONF_CEIL,
            rally_start_frame=0,
        ) is None

    def test_low_classifier_confidence_allows_fire(self) -> None:
        """When BOTH MS-TCN AND action_classifier indicate the first action
        is weakly-labeled (low MS-TCN serve prob AND low classifier
        confidence), the gate fires normally. Confirms the new dual-signal
        gate doesn't break the canonical TP case."""
        seq = _seq_with_serve_peak(peak_frame=110, peak_prob=0.99, length=500)
        result = should_prepend_serve(
            sequence_probs=seq,
            first_action_frame=426,
            first_action_serve_prob=0.02,
            first_action_classifier_confidence=SERVE_PREPEND_CLASSIFIER_CONF_CEIL - 0.01,
            rally_start_frame=0,
        )
        assert result == 110

    def test_peak_below_floor_returns_none(self) -> None:
        seq = _seq_with_serve_peak(peak_frame=100, peak_prob=SERVE_PREPEND_PEAK_FLOOR - 0.01,
                                    length=400)
        assert should_prepend_serve(
            sequence_probs=seq,
            first_action_frame=400,
            first_action_serve_prob=0.05,
            first_action_classifier_confidence=0.30,
            rally_start_frame=0,
        ) is None

    def test_gap_below_min_returns_none(self) -> None:
        """Buildup peak just before a correctly detected serve — don't prepend."""
        seq = _seq_with_serve_peak(peak_frame=95, peak_prob=0.99, length=200)
        assert should_prepend_serve(
            sequence_probs=seq,
            first_action_frame=95 + SERVE_PREPEND_MIN_GAP - 1,
            first_action_serve_prob=0.05,
            first_action_classifier_confidence=0.30,
            rally_start_frame=0,
        ) is None

    def test_gap_exactly_at_min_fires(self) -> None:
        seq = _seq_with_serve_peak(peak_frame=95, peak_prob=0.99, length=300)
        result = should_prepend_serve(
            sequence_probs=seq,
            first_action_frame=95 + SERVE_PREPEND_MIN_GAP,
            first_action_serve_prob=0.05,
            first_action_classifier_confidence=0.30,
            rally_start_frame=0,
        )
        assert result == 95

    def test_window_too_short_returns_none(self) -> None:
        """If rally_start ≥ first_action - guard, the search window is empty."""
        seq = _seq_with_serve_peak(peak_frame=10, peak_prob=0.99, length=100)
        assert should_prepend_serve(
            sequence_probs=seq,
            first_action_frame=SERVE_PREPEND_GUARD_FRAMES + 5,
            first_action_serve_prob=0.05,
            first_action_classifier_confidence=0.30,
            rally_start_frame=SERVE_PREPEND_GUARD_FRAMES + 4,
        ) is None

    def test_constants_match_calibration(self) -> None:
        """Lock the constants — they came from a 338-rally fleet sweep.
        Re-tuning requires re-validation."""
        assert SERVE_PREPEND_PEAK_FLOOR == 0.95
        assert SERVE_PREPEND_MIN_GAP == 25
        assert SERVE_PREPEND_FIRST_ACTION_SERVE_CEIL == 0.50
        assert SERVE_PREPEND_CLASSIFIER_CONF_CEIL == 0.50
        assert SERVE_PREPEND_GUARD_FRAMES == 15


import pytest
from rallycut.evaluation.tracking.db import get_connection
from rallycut.tracking.action_classifier import classify_rally_actions
from rallycut.tracking.ball_tracker import BallPosition
from rallycut.tracking.contact_detector import ContactDetectionConfig, detect_contacts
from rallycut.tracking.player_tracker import PlayerPosition
from rallycut.tracking.sequence_action_runtime import get_sequence_probs


def _load_rally(rally_id_prefix: str) -> dict:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT r.id, pt.fps, pt.frame_count, pt.court_split_y,
                          pt.ball_positions_json, pt.positions_json,
                          pt.actions_json, pt.primary_track_ids
                   FROM rallies r LEFT JOIN player_tracks pt ON pt.rally_id = r.id
                   WHERE r.id LIKE %s LIMIT 1""",
                [f"{rally_id_prefix}%"],
            )
            row = cur.fetchone()
    assert row is not None, f"Rally {rally_id_prefix} not found"
    rid, fps, fcount, csy, bp_json, pp_json, aj, primary_raw = row
    bp = [BallPosition(frame_number=int(b["frameNumber"]), x=float(b["x"]),
                       y=float(b["y"]), confidence=float(b.get("confidence", 0)))
          for b in bp_json if isinstance(b, dict)]
    pp = [PlayerPosition(frame_number=int(p["frameNumber"]), track_id=int(p["trackId"]),
                         x=float(p["x"]), y=float(p["y"]),
                         width=float(p["width"]), height=float(p["height"]),
                         confidence=float(p.get("confidence", 0)),
                         keypoints=p.get("keypoints"))
          for p in pp_json if isinstance(p, dict)]
    ta_str = (aj or {}).get("teamAssignments", {}) or {}
    ta_int = {int(k): (0 if v == "A" else 1) for k, v in ta_str.items() if v in ("A", "B")}
    return {
        "rally_id": rid, "fps": fps, "fcount": fcount, "csy": csy,
        "bp": bp, "pp": pp, "ta_int": ta_int, "primary_raw": primary_raw or [],
    }


class TestPrependIntegration:
    @pytest.mark.slow
    def test_wawa_8c49e480_prepends_serve_near_frame_110(self) -> None:
        """Canonical case: GT serve at 101, pipeline first contact at 426
        (originally mis-labeled as serve).

        After v1.3 prepend:
          - A synthetic serve lands within ±15 of GT 101.
          - The OLD first contact (frame 426) is NO LONGER labeled "serve".
          - There is exactly one serve in the final action list.
          - All downstream actions kept their non-serve labels (re-classification
            handled by `classify_rally`, not by manual re-labeling).
        """
        r = _load_rally("8c49e480")
        seq = get_sequence_probs(
            r["bp"], r["pp"], r["csy"], r["fcount"] or 0, r["ta_int"], calibrator=None,
        )
        assert seq is not None
        contact_seq = detect_contacts(
            ball_positions=r["bp"], player_positions=r["pp"],
            config=ContactDetectionConfig(),
            net_y=r["csy"], frame_count=r["fcount"] or None,
            team_assignments=r["ta_int"],
            sequence_probs=seq,
            primary_track_ids=r["primary_raw"] or None,
        )
        ra = classify_rally_actions(
            contact_seq,
            team_assignments=r["ta_int"],
            sequence_probs=seq,
        )
        serves = [a for a in ra.actions if a.action_type.value == "serve"]
        # Exactly one serve — the synthetic prepend, not a duplicate
        assert len(serves) == 1, (
            f"expected exactly 1 serve, got {len(serves)}: "
            f"{[(s.frame, s.is_synthetic) for s in serves]}"
        )
        first_serve = serves[0]
        assert abs(first_serve.frame - 101) <= 15, (
            f"expected serve within ±15 of GT frame 101, got {first_serve.frame}"
        )
        assert first_serve.is_synthetic
        # The old first contact (frame 426) must NOT be labeled serve anymore
        old_first_actions = [a for a in ra.actions if a.frame == 426]
        if old_first_actions:
            assert old_first_actions[0].action_type.value != "serve", (
                "Old first contact at frame 426 should have been re-classified "
                "as a non-serve action by classify_rally re-run"
            )

    @pytest.mark.slow
    def test_correctly_detected_serve_rally_unchanged(self) -> None:
        """On a rally where the pipeline already detects the serve correctly,
        v1.3 must NOT fire — the first contact's own serve-prob is high enough
        that the gate is blocked.

        Uses a sample from the pipeline_already_correct cluster.
        """
        # Pick any rally where pipeline_already_correct holds — riri/ef32c552
        # had gt=127, pred=120 (within tolerance).
        r = _load_rally("ef32c552")
        seq = get_sequence_probs(
            r["bp"], r["pp"], r["csy"], r["fcount"] or 0, r["ta_int"], calibrator=None,
        )
        assert seq is not None
        contact_seq = detect_contacts(
            ball_positions=r["bp"], player_positions=r["pp"],
            config=ContactDetectionConfig(),
            net_y=r["csy"], frame_count=r["fcount"] or None,
            team_assignments=r["ta_int"],
            sequence_probs=seq,
            primary_track_ids=r["primary_raw"] or None,
        )
        ra = classify_rally_actions(
            contact_seq,
            team_assignments=r["ta_int"],
            sequence_probs=seq,
        )
        serves = [a for a in ra.actions if a.action_type.value == "serve"]
        assert len(serves) == 1
        # Real serve, not synthetic
        assert not serves[0].is_synthetic
        # Frame near GT 127 (allow some tolerance for HIT_TOLERANCE=15)
        assert abs(serves[0].frame - 127) <= 15
