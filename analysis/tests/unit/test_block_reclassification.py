"""Unit tests for A3 BLOCK reclassification helper.

Covers ``should_reclassify_to_block`` production gate (firm wrist + loose-d).
"""
from __future__ import annotations

from typing import Any

import pytest

from rallycut.tracking.block_reclassification import (
    should_reclassify_to_block,
)

# Shared geometry that satisfies (a)′ HEAD-near-BALL-at-net.
# ball_y near net_y, head near ball — confirms (a)′ pass.
NET_Y = 0.70                # ground-net image-y
BALL_Y_NEAR_NET = 0.55      # ball is ~15% above ground net (in air, mid-jump block region)
HEAD_Y_NEAR_BALL = 0.50     # head within ±0.15 of ball
WRIST_Y_FIRM = 0.45         # wrist above net (smaller y = higher)


def _make_attack(team: str = "A", frame: int = 200) -> dict[str, Any]:
    return {
        "action": "attack",
        "frame": frame,
        "team": team,
        "playerTrackId": 1,
    }


def _make_prev(
    action_type: str = "set",
    team: str = "B",
    frame: int = 150,
) -> dict[str, Any]:
    return {
        "action": action_type,
        "frame": frame,
        "team": team,
        "playerTrackId": 2,
    }


# ---------------------------------------------------------------------------
# Positive cases
# ---------------------------------------------------------------------------


def test_f5_shape_reclassifies() -> None:
    """F5 canonical: prev=receive (cross-team), all gates firm + (a)′ pass."""
    action = _make_attack(team="A", frame=184)
    prev = _make_prev(action_type="receive", team="B", frame=150)
    assert should_reclassify_to_block(
        action=action,
        prev_action=prev,
        direction_change_deg=13.0,
        ball_y_image=BALL_Y_NEAR_NET,
        player_bbox_top_y_image=HEAD_Y_NEAR_BALL,
        net_y_image=NET_Y,
        wrist_y_image=WRIST_Y_FIRM,
    ) is True


def test_strict_shape_reclassifies() -> None:
    """Strict (d): prev=set cross-team, all gates firm."""
    action = _make_attack(team="A", frame=151)
    prev = _make_prev(action_type="set", team="B", frame=121)
    assert should_reclassify_to_block(
        action=action,
        prev_action=prev,
        direction_change_deg=35.6,
        ball_y_image=BALL_Y_NEAR_NET,
        player_bbox_top_y_image=HEAD_Y_NEAR_BALL,
        net_y_image=NET_Y,
        wrist_y_image=WRIST_Y_FIRM,
    ) is True


def test_strict_shape_with_prev_attack_reclassifies() -> None:
    """Strict (d): prev=attack cross-team also accepted under loose."""
    action = _make_attack(team="B", frame=242)
    prev = _make_prev(action_type="attack", team="A", frame=209)
    assert should_reclassify_to_block(
        action=action,
        prev_action=prev,
        direction_change_deg=10.9,
        ball_y_image=BALL_Y_NEAR_NET,
        player_bbox_top_y_image=HEAD_Y_NEAR_BALL,
        net_y_image=NET_Y,
        wrist_y_image=WRIST_Y_FIRM,
    ) is True


# ---------------------------------------------------------------------------
# Negative cases — one gate fails at a time
# ---------------------------------------------------------------------------


def test_a_fails_head_far_from_net_skips() -> None:
    """(a)′ fail: head image-y is far from ball / ball not in net region."""
    action = _make_attack(team="A")
    prev = _make_prev(action_type="set", team="B")
    # Head sits at 0.95 (deep back-court) and ball at 0.55 — abs diff > band (0.15).
    assert should_reclassify_to_block(
        action=action,
        prev_action=prev,
        direction_change_deg=20.0,
        ball_y_image=BALL_Y_NEAR_NET,
        player_bbox_top_y_image=0.95,   # head far below ball
        net_y_image=NET_Y,
        wrist_y_image=WRIST_Y_FIRM,
    ) is False


def test_b_unknown_no_wrist_detected_skips() -> None:
    """(b)′ Unknown (no wrist detected): firm gate REJECTS."""
    action = _make_attack(team="A")
    prev = _make_prev(action_type="set", team="B")
    assert should_reclassify_to_block(
        action=action,
        prev_action=prev,
        direction_change_deg=20.0,
        ball_y_image=BALL_Y_NEAR_NET,
        player_bbox_top_y_image=HEAD_Y_NEAR_BALL,
        net_y_image=NET_Y,
        wrist_y_image=None,             # NOT detected
    ) is False


def test_b_wrist_below_net_skips() -> None:
    """(b)′ wrist below net (image-y > net): rejected."""
    action = _make_attack(team="A")
    prev = _make_prev(action_type="set", team="B")
    assert should_reclassify_to_block(
        action=action,
        prev_action=prev,
        direction_change_deg=20.0,
        ball_y_image=BALL_Y_NEAR_NET,
        player_bbox_top_y_image=HEAD_Y_NEAR_BALL,
        net_y_image=NET_Y,
        wrist_y_image=0.85,             # below net (greater image-y)
    ) is False


def test_c_fails_direction_change_over_90_skips() -> None:
    """(c) fail: dc > 90° (ball fully reverses, not a block deflection)."""
    action = _make_attack(team="A")
    prev = _make_prev(action_type="set", team="B")
    assert should_reclassify_to_block(
        action=action,
        prev_action=prev,
        direction_change_deg=145.0,     # full reversal — not a block
        ball_y_image=BALL_Y_NEAR_NET,
        player_bbox_top_y_image=HEAD_Y_NEAR_BALL,
        net_y_image=NET_Y,
        wrist_y_image=WRIST_Y_FIRM,
    ) is False


def test_d_fails_prev_is_serve_skips() -> None:
    """(d) fail (loose-excluded): prev=serve cross-team — receive scenario, not block."""
    action = _make_attack(team="A")
    prev = _make_prev(action_type="serve", team="B")
    assert should_reclassify_to_block(
        action=action,
        prev_action=prev,
        direction_change_deg=20.0,
        ball_y_image=BALL_Y_NEAR_NET,
        player_bbox_top_y_image=HEAD_Y_NEAR_BALL,
        net_y_image=NET_Y,
        wrist_y_image=WRIST_Y_FIRM,
    ) is False


def test_d_fails_same_team_skips() -> None:
    """(d) fail: prev is the SAME team — same-side rally, not a block."""
    action = _make_attack(team="A")
    prev = _make_prev(action_type="set", team="A")  # SAME team
    assert should_reclassify_to_block(
        action=action,
        prev_action=prev,
        direction_change_deg=20.0,
        ball_y_image=BALL_Y_NEAR_NET,
        player_bbox_top_y_image=HEAD_Y_NEAR_BALL,
        net_y_image=NET_Y,
        wrist_y_image=WRIST_Y_FIRM,
    ) is False


def test_no_prev_action_skips() -> None:
    """No prev (first action in rally) — (d) cannot pass."""
    action = _make_attack(team="A")
    assert should_reclassify_to_block(
        action=action,
        prev_action=None,
        direction_change_deg=20.0,
        ball_y_image=BALL_Y_NEAR_NET,
        player_bbox_top_y_image=HEAD_Y_NEAR_BALL,
        net_y_image=NET_Y,
        wrist_y_image=WRIST_Y_FIRM,
    ) is False


# ---------------------------------------------------------------------------
# Strict (d) variant
# ---------------------------------------------------------------------------


def test_strict_d_variant_accepts_prev_set() -> None:
    """Strict (d) variant: prev=set cross-team accepted."""
    action = _make_attack(team="A")
    prev = _make_prev(action_type="set", team="B")
    assert should_reclassify_to_block(
        action=action,
        prev_action=prev,
        direction_change_deg=20.0,
        ball_y_image=BALL_Y_NEAR_NET,
        player_bbox_top_y_image=HEAD_Y_NEAR_BALL,
        net_y_image=NET_Y,
        wrist_y_image=WRIST_Y_FIRM,
        use_strict_d=True,
    ) is True


def test_strict_d_variant_rejects_prev_receive() -> None:
    """Strict (d) variant rejects F5-shape (prev=receive)."""
    action = _make_attack(team="A")
    prev = _make_prev(action_type="receive", team="B")
    assert should_reclassify_to_block(
        action=action,
        prev_action=prev,
        direction_change_deg=13.0,
        ball_y_image=BALL_Y_NEAR_NET,
        player_bbox_top_y_image=HEAD_Y_NEAR_BALL,
        net_y_image=NET_Y,
        wrist_y_image=WRIST_Y_FIRM,
        use_strict_d=True,
    ) is False


# ---------------------------------------------------------------------------
# Soft variant (require_firm_wrist=False) — diagnostic
# ---------------------------------------------------------------------------


def test_soft_variant_accepts_wrist_unknown() -> None:
    """require_firm_wrist=False allows (b)=Unknown — diagnostic mode only."""
    action = _make_attack(team="A")
    prev = _make_prev(action_type="set", team="B")
    assert should_reclassify_to_block(
        action=action,
        prev_action=prev,
        direction_change_deg=20.0,
        ball_y_image=BALL_Y_NEAR_NET,
        player_bbox_top_y_image=HEAD_Y_NEAR_BALL,
        net_y_image=NET_Y,
        wrist_y_image=None,             # Unknown
        require_firm_wrist=False,
    ) is True


def test_soft_variant_rejects_wrist_false() -> None:
    """require_firm_wrist=False still rejects explicit (b)=False."""
    action = _make_attack(team="A")
    prev = _make_prev(action_type="set", team="B")
    assert should_reclassify_to_block(
        action=action,
        prev_action=prev,
        direction_change_deg=20.0,
        ball_y_image=BALL_Y_NEAR_NET,
        player_bbox_top_y_image=HEAD_Y_NEAR_BALL,
        net_y_image=NET_Y,
        wrist_y_image=0.90,             # below-net wrist
        require_firm_wrist=False,
    ) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
