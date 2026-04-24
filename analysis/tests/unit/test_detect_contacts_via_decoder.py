"""Unit tests for `detect_contacts_via_decoder` (Phase 2c).

Smoke-coverage at the unit-test layer:
1. Function exists with the right signature.
2. Returns ContactSequence on real rally data (DB-backed; skips when offline).
3. Decoder-path contacts carry `decoder_action` populated; GBM-path does not.
4. Falls back to `detect_contacts` cleanly when no classifier is available.

Apples-to-apples LOO-eval comparison (decoder vs baseline F1, Action Acc,
per-class F1) is the job of Phase 3 in
`docs/superpowers/plans/2026-04-24-parallel-decoder-ship.md`. This file
just guards the parallel-entry-point shape.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np  # noqa: F401  (used in return-type hint of _load_real_rally)
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_real_rally() -> tuple[list, list, np.ndarray | None, int]:
    """Load ball+player positions and seq_probs for a stable test rally.

    Skips the test if the DB is unreachable or the rally has no data.
    """
    try:
        from rallycut.tracking.ball_tracker import BallPosition
        from rallycut.tracking.sequence_action_runtime import get_sequence_probs
        from scripts.eval_action_detection import (
            _build_player_positions,
            load_rallies_with_action_gt,
        )
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Cannot import rally loaders: {e}")

    rally_id = "0102cbba-74c3-43ae-8d7b-caf604cae34e"
    try:
        rallies = load_rallies_with_action_gt(rally_id=rally_id)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Cannot reach DB: {e}")
    if not rallies:
        pytest.skip(f"Rally {rally_id} not in DB")
    rally = rallies[0]
    if not rally.ball_positions_json or not rally.positions_json:
        pytest.skip(f"Rally {rally_id} missing positions")

    players = _build_player_positions(
        rally.positions_json, rally_id=rally.rally_id, inject_pose=True,
    )
    balls = [
        BallPosition(
            frame_number=bp["frameNumber"],
            x=bp["x"],
            y=bp["y"],
            confidence=bp.get("confidence", 1.0),
        )
        for bp in rally.ball_positions_json
    ]
    seq_probs = get_sequence_probs(
        balls, players, rally.court_split_y, rally.frame_count or 0, None,
    )
    return balls, players, seq_probs, rally.frame_count or 0


def test_function_signature_and_imports() -> None:
    """Phase 2c entry point exists with required parameters."""
    import inspect

    from rallycut.tracking.contact_detector import detect_contacts_via_decoder
    sig = inspect.signature(detect_contacts_via_decoder)
    required_params = {
        "ball_positions",
        "player_positions",
        "config",
        "classifier",
        "use_classifier",
        "sequence_probs",
        "team_assignments",
        "court_calibrator",
        "frame_count",
        "primary_track_ids",
        "skip_penalty",
        "min_accept_prob",
    }
    missing = required_params - set(sig.parameters)
    assert not missing, f"Missing params: {missing}"


def test_returns_contact_sequence_on_real_rally() -> None:
    """Decoder path produces a ContactSequence with attributed contacts."""
    from rallycut.tracking.contact_detector import (
        ContactSequence,
        detect_contacts_via_decoder,
    )

    balls, players, seq_probs, frame_count = _load_real_rally()

    seq = detect_contacts_via_decoder(
        ball_positions=balls,
        player_positions=players,
        sequence_probs=seq_probs,
        frame_count=frame_count,
    )

    assert isinstance(seq, ContactSequence)
    # Stable test rally has 4 GT contacts; decoder should produce >0
    assert seq.num_contacts > 0, (
        "Decoder produced 0 contacts on a known-non-empty rally"
    )
    # Every accepted contact must be flagged validated and have an action
    for c in seq.contacts:
        assert c.is_validated, f"Decoder emitted unvalidated contact at f={c.frame}"
        assert c.decoder_action in {
            "serve", "receive", "set", "attack", "dig", "block",
        }, f"Bad decoder_action {c.decoder_action!r} at f={c.frame}"


def test_decoder_assigns_action_label_gbm_does_not() -> None:
    """The new field `decoder_action` is set only on the decoder path."""
    from rallycut.tracking.contact_detector import (
        detect_contacts,
        detect_contacts_via_decoder,
    )

    balls, players, seq_probs, frame_count = _load_real_rally()

    gbm_seq = detect_contacts(
        ball_positions=balls,
        player_positions=players,
        sequence_probs=seq_probs,
        frame_count=frame_count,
    )
    dec_seq = detect_contacts_via_decoder(
        ball_positions=balls,
        player_positions=players,
        sequence_probs=seq_probs,
        frame_count=frame_count,
    )

    # GBM path leaves decoder_action as None
    assert all(c.decoder_action is None for c in gbm_seq.contacts), (
        "GBM path should NOT populate decoder_action"
    )
    # Decoder path populates decoder_action on every contact
    assert all(c.decoder_action is not None for c in dec_seq.contacts), (
        "Decoder path must populate decoder_action on every contact"
    )


def test_falls_back_to_detect_contacts_without_classifier() -> None:
    """When no trained classifier is available, decoder path defers to GBM."""
    from rallycut.tracking.contact_classifier import ContactClassifier
    from rallycut.tracking.contact_detector import (
        detect_contacts,
        detect_contacts_via_decoder,
    )

    balls, players, seq_probs, frame_count = _load_real_rally()

    # Pass a freshly-constructed (untrained) classifier; decoder must fall back
    untrained = ContactClassifier()
    assert not untrained.is_trained

    fallback = detect_contacts_via_decoder(
        ball_positions=balls,
        player_positions=players,
        sequence_probs=seq_probs,
        frame_count=frame_count,
        classifier=untrained,
    )
    legacy = detect_contacts(
        ball_positions=balls,
        player_positions=players,
        sequence_probs=seq_probs,
        frame_count=frame_count,
        classifier=untrained,
        use_classifier=False,  # legacy hand-tuned gates
    )
    # Falling back means we get GBM-shaped output (no decoder_action)
    assert all(c.decoder_action is None for c in fallback.contacts), (
        "Fallback should produce GBM-style (no decoder_action) output"
    )
    # Both should produce some contacts via the legacy path
    assert fallback.num_contacts > 0
    assert legacy.num_contacts > 0


def test_empty_ball_positions_returns_empty_sequence() -> None:
    """Defensive: zero-input case should not crash."""
    from rallycut.tracking.contact_detector import (
        ContactSequence,
        detect_contacts_via_decoder,
    )
    seq = detect_contacts_via_decoder(ball_positions=[])
    assert isinstance(seq, ContactSequence)
    assert seq.num_contacts == 0
