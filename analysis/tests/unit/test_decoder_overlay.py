"""Unit tests for apply_decoder_labels overlay."""
from __future__ import annotations

from rallycut.tracking.action_classifier import ActionType, ClassifiedAction, RallyActions
from rallycut.tracking.candidate_decoder import DecodedContact
from rallycut.tracking.decoder_overlay import apply_decoder_labels


def _ca(frame: int, action: ActionType, tid: int = 1) -> ClassifiedAction:
    return ClassifiedAction(
        action_type=action,
        frame=frame,
        ball_x=0.5,
        ball_y=0.5,
        velocity=0.1,
        player_track_id=tid,
        court_side="near",
        confidence=0.8,
        team="A",
    )


def _dc(frame: int, action: str) -> DecodedContact:
    return DecodedContact(
        candidate_idx=0,
        frame=frame,
        action=action,
        action_idx=0,
        score=-1.0,
    )


def _rally(actions: list[ClassifiedAction]) -> RallyActions:
    return RallyActions(rally_id="r1", actions=actions)


def test_swaps_label_within_tolerance() -> None:
    ra = _rally([_ca(100, ActionType.RECEIVE)])
    decoder = [_dc(101, "set")]
    out, stat = apply_decoder_labels(ra, decoder, tol_frames=3)
    assert out.actions[0].action_type == ActionType.SET
    assert out.actions[0].player_track_id == 1
    assert stat.n_label_swapped == 1
    assert stat.n_matched == 1


def test_no_swap_outside_tolerance() -> None:
    ra = _rally([_ca(100, ActionType.RECEIVE)])
    decoder = [_dc(110, "set")]
    out, stat = apply_decoder_labels(ra, decoder, tol_frames=3)
    assert out.actions[0].action_type == ActionType.RECEIVE
    assert stat.n_label_swapped == 0
    assert stat.n_matched == 0


def test_no_swap_when_labels_agree() -> None:
    ra = _rally([_ca(100, ActionType.SET)])
    decoder = [_dc(100, "set")]
    out, stat = apply_decoder_labels(ra, decoder, tol_frames=3)
    assert stat.n_matched == 1
    assert stat.n_label_swapped == 0


def test_greedy_matching_closest_first() -> None:
    ra = _rally([_ca(100, ActionType.RECEIVE), _ca(105, ActionType.DIG)])
    decoder = [_dc(100, "set"), _dc(106, "attack")]
    out, _ = apply_decoder_labels(ra, decoder, tol_frames=3)
    assert out.actions[0].action_type == ActionType.SET
    assert out.actions[1].action_type == ActionType.ATTACK


def test_decoder_frame_with_no_detector_match_is_dropped() -> None:
    ra = _rally([_ca(100, ActionType.SERVE)])
    decoder = [_dc(100, "set"), _dc(500, "attack")]
    out, stat = apply_decoder_labels(ra, decoder, tol_frames=3)
    assert len(out.actions) == 1
    assert out.actions[0].action_type == ActionType.SET
    assert stat.n_matched == 1


def test_empty_decoder_is_noop() -> None:
    ra = _rally([_ca(100, ActionType.SERVE), _ca(120, ActionType.RECEIVE)])
    out, stat = apply_decoder_labels(ra, [], tol_frames=3)
    assert out.actions[0].action_type == ActionType.SERVE
    assert out.actions[1].action_type == ActionType.RECEIVE
    assert stat.n_label_swapped == 0


def test_empty_rally_is_noop() -> None:
    ra = _rally([])
    decoder = [_dc(100, "serve")]
    out, stat = apply_decoder_labels(ra, decoder, tol_frames=3)
    assert out.actions == []
    assert stat.n_label_swapped == 0
    assert stat.n_matched == 0


def test_all_non_action_fields_preserved() -> None:
    original = ClassifiedAction(
        action_type=ActionType.RECEIVE,
        frame=100,
        ball_x=0.42,
        ball_y=0.88,
        velocity=0.73,
        player_track_id=3,
        court_side="far",
        confidence=0.91,
        team="B",
        is_synthetic=True,
        action_zone=4,
        attack_direction="line",
        set_origin_zone=2,
        set_dest_zone=5,
    )
    ra = RallyActions(rally_id="r1", actions=[original])
    out, _ = apply_decoder_labels(ra, [_dc(100, "set")], tol_frames=3)
    swapped = out.actions[0]
    assert swapped.action_type == ActionType.SET
    for field_name in (
        "frame", "ball_x", "ball_y", "velocity", "player_track_id",
        "court_side", "confidence", "team", "is_synthetic",
        "action_zone", "attack_direction", "set_origin_zone", "set_dest_zone",
    ):
        assert getattr(swapped, field_name) == getattr(original, field_name), (
            f"field {field_name} changed"
        )


def test_unknown_decoder_action_string_is_ignored() -> None:
    """If the decoder emits an action label not in the mapping (shouldn't
    happen in practice, but be defensive), the match consumes the slot but
    no swap occurs - prevents a bad string from clobbering a good label."""
    ra = _rally([_ca(100, ActionType.SERVE)])
    decoder = [_dc(100, "bogus_action")]
    out, stat = apply_decoder_labels(ra, decoder, tol_frames=3)
    assert out.actions[0].action_type == ActionType.SERVE
    assert stat.n_matched == 1
    assert stat.n_label_swapped == 0


def test_tie_breaking_picks_first_in_list() -> None:
    """When two detected actions are equidistant from a decoder contact,
    the earlier one in rally_actions.actions wins. Deterministic behavior
    relied on by Task 3 callers."""
    ra = _rally([_ca(99, ActionType.RECEIVE), _ca(101, ActionType.DIG)])
    decoder = [_dc(100, "set")]
    out, stat = apply_decoder_labels(ra, decoder, tol_frames=3)
    assert out.actions[0].action_type == ActionType.SET
    assert out.actions[1].action_type == ActionType.DIG
    assert stat.n_matched == 1
    assert stat.n_label_swapped == 1


def test_unsorted_decoder_contacts_are_sorted_internally() -> None:
    """The function sorts decoder_contacts by frame before greedy matching.
    Pinning this so the sort can't silently disappear in a future refactor."""
    ra = _rally([_ca(100, ActionType.RECEIVE), _ca(200, ActionType.DIG)])
    decoder = [_dc(200, "attack"), _dc(100, "set")]  # reversed
    out, stat = apply_decoder_labels(ra, decoder, tol_frames=3)
    assert out.actions[0].action_type == ActionType.SET
    assert out.actions[1].action_type == ActionType.ATTACK
    assert stat.n_matched == 2
    assert stat.n_label_swapped == 2
