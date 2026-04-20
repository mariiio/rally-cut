"""Unit tests for decoder_runtime.run_decoder_over_rally."""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from rallycut.tracking.ball_tracker import BallPosition
from rallycut.tracking.candidate_decoder import DecodedContact
from rallycut.tracking.contact_detector import ContactDetectionConfig
from rallycut.tracking.decoder_runtime import run_decoder_over_rally
from rallycut.tracking.player_tracker import PlayerPosition


def _ball(frame: int, x: float = 0.5, y: float = 0.5) -> BallPosition:
    return BallPosition(frame_number=frame, x=x, y=y, confidence=0.9)


def _player(frame: int, track_id: int, x: float = 0.5, y: float = 0.6) -> PlayerPosition:
    return PlayerPosition(
        frame_number=frame, track_id=track_id, x=x, y=y,
        width=0.1, height=0.2, confidence=0.9,
    )


def _fake_classifier(n_features: int = 26, n_rows: int = 10) -> MagicMock:
    clf = MagicMock()
    clf.model = MagicMock()
    clf.model.n_features_in_ = n_features
    clf.model.predict_proba = MagicMock(
        return_value=np.tile([0.95, 0.05], (n_rows, 1))
    )
    return clf


def test_returns_empty_when_no_candidates() -> None:
    result = run_decoder_over_rally(
        ball_positions=[],
        player_positions=[],
        sequence_probs=None,
        classifier=_fake_classifier(),
        contact_config=ContactDetectionConfig(),
    )
    assert result == []


def test_returns_empty_when_classifier_untrained() -> None:
    clf = MagicMock()
    clf.model = None
    result = run_decoder_over_rally(
        ball_positions=[_ball(i) for i in range(30)],
        player_positions=[_player(i, 1) for i in range(30)],
        sequence_probs=None,
        classifier=clf,
        contact_config=ContactDetectionConfig(),
    )
    assert result == []


def test_returns_decoded_contacts_in_frame_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """When candidates exist, decoder returns DecodedContact in frame order."""
    from rallycut.tracking import decoder_runtime
    from rallycut.tracking.contact_classifier import CandidateFeatures

    fake_decoded = [
        DecodedContact(candidate_idx=0, frame=10, action="serve", action_idx=0, score=-1.0),
        DecodedContact(candidate_idx=1, frame=40, action="receive", action_idx=1, score=-2.0),
    ]
    monkeypatch.setattr(
        decoder_runtime, "decode_rally",
        lambda *a, **kw: fake_decoded,
    )
    import scripts.train_contact_classifier as tcc

    def _fake_extract(
        rally: Any, **kw: Any,
    ) -> tuple[list[CandidateFeatures], list[int]]:
        return (
            [
                CandidateFeatures(
                    frame=10, velocity=0.1, direction_change_deg=0.0,
                    arc_fit_residual=0.0, acceleration=0.0,
                    trajectory_curvature=0.0, velocity_y=0.0,
                    velocity_ratio=1.0, player_distance=0.1,
                    ball_x=0.5, ball_y=0.5, ball_y_relative_net=0.0,
                    is_net_crossing=False, frames_since_last=0,
                ),
                CandidateFeatures(
                    frame=40, velocity=0.1, direction_change_deg=0.0,
                    arc_fit_residual=0.0, acceleration=0.0,
                    trajectory_curvature=0.0, velocity_y=0.0,
                    velocity_ratio=1.0, player_distance=0.1,
                    ball_x=0.5, ball_y=0.5, ball_y_relative_net=0.0,
                    is_net_crossing=False, frames_since_last=30,
                ),
            ],
            [10, 40],
        )

    # Patches the tcc module attribute; works because run_decoder_over_rally
    # does a deferred `from scripts.train_contact_classifier import ...` at
    # call time, which resolves against the patched module attribute. If
    # that import is ever hoisted to module level, this patch will silently
    # no-op — update this test at that time.
    monkeypatch.setattr(tcc, "extract_candidate_features", _fake_extract)

    result = run_decoder_over_rally(
        ball_positions=[_ball(i) for i in range(50)],
        player_positions=[_player(i, 1) for i in range(50)],
        sequence_probs=None,
        classifier=_fake_classifier(n_rows=2),
        contact_config=ContactDetectionConfig(),
    )
    assert [d.frame for d in result] == [10, 40]
    assert [d.action for d in result] == ["serve", "receive"]


def test_pad_when_classifier_expects_fewer_features(monkeypatch: pytest.MonkeyPatch) -> None:
    """If classifier expects 10 features and the code produces 26, the first
    10 columns are used and the call still returns decoded contacts."""
    from rallycut.tracking import decoder_runtime
    from rallycut.tracking.contact_classifier import CandidateFeatures

    fake_decoded = [
        DecodedContact(candidate_idx=0, frame=5, action="attack", action_idx=3, score=-1.0),
    ]
    monkeypatch.setattr(decoder_runtime, "decode_rally", lambda *a, **kw: fake_decoded)

    import scripts.train_contact_classifier as tcc

    def _fake_extract(rally: Any, **kw: Any) -> tuple[list[CandidateFeatures], list[int]]:
        return (
            [CandidateFeatures(
                frame=5, velocity=0.1, direction_change_deg=0.0,
                arc_fit_residual=0.0, acceleration=0.0,
                trajectory_curvature=0.0, velocity_y=0.0,
                velocity_ratio=1.0, player_distance=0.1,
                ball_x=0.5, ball_y=0.5, ball_y_relative_net=0.0,
                is_net_crossing=False, frames_since_last=0,
            )],
            [5],
        )

    monkeypatch.setattr(tcc, "extract_candidate_features", _fake_extract)

    result = run_decoder_over_rally(
        ball_positions=[_ball(i) for i in range(10)],
        player_positions=[_player(i, 1) for i in range(10)],
        sequence_probs=None,
        classifier=_fake_classifier(n_features=10, n_rows=1),
        contact_config=ContactDetectionConfig(),
    )
    assert len(result) == 1
    assert result[0].action == "attack"
