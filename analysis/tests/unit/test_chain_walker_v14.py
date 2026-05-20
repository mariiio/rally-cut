"""Tests for v14 chain-walker decision function."""
from __future__ import annotations

from dataclasses import dataclass

from rallycut.tracking.action_classifier import (
    ChainWalkerConfig,
    _contact_side_at,
    _possession_flips_after,
)
from rallycut.tracking.contact_detector import Contact


def _make_contact(frame: int, side: str, is_synthetic: bool = False) -> Contact:
    c = Contact(
        frame=frame,
        ball_x=0.5,
        ball_y=0.5 if side == "near" else 0.3,
        velocity=0.0,
        direction_change_deg=0.0,
        court_side=side,
        is_at_net=False,
        confidence=0.8,
        is_validated=True,
        player_track_id=1,
        arc_fit_residual=0.0,
        player_distance=0.05,
    )
    # Attach is_synthetic as a dynamic attribute since Contact doesn't have it natively.
    # _contact_side_at uses getattr(c, 'is_synthetic', False) so this works.
    object.__setattr__(c, "_is_synthetic_override", is_synthetic)
    return c


# Monkey-patch is_synthetic as a property-like override via a wrapper
@dataclass
class _ContactWithSynthetic:
    """Thin wrapper that adds is_synthetic to Contact for test use."""

    _contact: Contact
    is_synthetic: bool = False

    def __getattr__(self, name: str) -> object:
        return getattr(self._contact, name)


def _make_contact_obj(frame: int, side: str, is_synthetic: bool = False) -> _ContactWithSynthetic:
    c = Contact(
        frame=frame,
        ball_x=0.5,
        ball_y=0.5 if side == "near" else 0.3,
        velocity=0.0,
        direction_change_deg=0.0,
        court_side=side,
        is_at_net=False,
        confidence=0.8,
        is_validated=True,
        player_track_id=1,
        arc_fit_residual=0.0,
        player_distance=0.05,
    )
    return _ContactWithSynthetic(_contact=c, is_synthetic=is_synthetic)


@dataclass
class _FakeAction:
    action_type: str
    frame: int
    is_synthetic: bool = False


def test_v13_default_serve_flips() -> None:
    cfg = ChainWalkerConfig(False, False)
    assert _possession_flips_after(_FakeAction("serve", 100), _FakeAction("receive", 140), [], cfg) is True


def test_v13_default_attack_flips() -> None:
    cfg = ChainWalkerConfig(False, False)
    assert _possession_flips_after(_FakeAction("attack", 200), _FakeAction("dig", 220), [], cfg) is True


def test_v13_default_block_stays() -> None:
    cfg = ChainWalkerConfig(False, False)
    assert _possession_flips_after(_FakeAction("block", 300), _FakeAction("attack", 320), [], cfg) is False


def test_v13_default_set_stays() -> None:
    cfg = ChainWalkerConfig(False, False)
    assert _possession_flips_after(_FakeAction("set", 400), _FakeAction("attack", 420), [], cfg) is False


def test_b1_block_flips_when_next_side_differs() -> None:
    cfg = ChainWalkerConfig(block_conditional=True, ball_trajectory_verifier=False)
    contacts = [_make_contact_obj(100, "near"), _make_contact_obj(130, "far")]
    assert _possession_flips_after(_FakeAction("block", 100), _FakeAction("dig", 130), contacts, cfg) is True


def test_b1_block_stays_when_next_side_same() -> None:
    cfg = ChainWalkerConfig(block_conditional=True, ball_trajectory_verifier=False)
    contacts = [_make_contact_obj(100, "near"), _make_contact_obj(130, "near")]
    assert _possession_flips_after(_FakeAction("block", 100), _FakeAction("attack", 130), contacts, cfg) is False


def test_b1_block_falls_back_to_v13_when_contacts_missing() -> None:
    cfg = ChainWalkerConfig(block_conditional=True, ball_trajectory_verifier=False)
    # No contacts → v13 rule (block stays)
    assert _possession_flips_after(_FakeAction("block", 100), _FakeAction("dig", 130), [], cfg) is False


def test_b2_overrides_rule_when_ball_crossed() -> None:
    """SET rule says stay, but contact court_sides differ → verifier flips."""
    cfg = ChainWalkerConfig(block_conditional=False, ball_trajectory_verifier=True)
    contacts = [_make_contact_obj(100, "near"), _make_contact_obj(130, "far")]
    assert _possession_flips_after(_FakeAction("set", 100), _FakeAction("attack", 130), contacts, cfg) is True


def test_b2_overrides_rule_when_ball_stayed() -> None:
    """SERVE rule says flip, but court_sides match → verifier stays."""
    cfg = ChainWalkerConfig(block_conditional=False, ball_trajectory_verifier=True)
    contacts = [_make_contact_obj(100, "near"), _make_contact_obj(130, "near")]
    assert _possession_flips_after(_FakeAction("serve", 100), _FakeAction("dig", 130), contacts, cfg) is False


def test_b2_degrades_to_rule_when_synthetic_contact() -> None:
    """Synthetic contact → verifier returns None → falls back to rule."""
    cfg = ChainWalkerConfig(block_conditional=False, ball_trajectory_verifier=True)
    contacts = [_make_contact_obj(100, "near", is_synthetic=True), _make_contact_obj(130, "far")]
    # Verifier declines on synthetic; v13 rule applies (attack flips)
    assert _possession_flips_after(_FakeAction("attack", 100), _FakeAction("dig", 130), contacts, cfg) is True


def test_b2_degrades_to_rule_when_contact_side_unknown() -> None:
    cfg = ChainWalkerConfig(block_conditional=False, ball_trajectory_verifier=True)
    contacts = [_make_contact_obj(100, "unknown"), _make_contact_obj(130, "far")]
    # Verifier declines on unknown; v13 rule applies (attack flips)
    assert _possession_flips_after(_FakeAction("attack", 100), _FakeAction("dig", 130), contacts, cfg) is True


def test_contact_side_at_returns_side_within_tolerance() -> None:
    contacts = [_make_contact_obj(100, "near"), _make_contact_obj(150, "far")]
    assert _contact_side_at(contacts, 101) == "near"
    assert _contact_side_at(contacts, 152) == "far"


def test_contact_side_at_returns_none_outside_tolerance() -> None:
    contacts = [_make_contact_obj(100, "near")]
    assert _contact_side_at(contacts, 105) is None


def test_contact_side_at_returns_none_when_side_unknown() -> None:
    contacts = [_make_contact_obj(100, "unknown")]
    assert _contact_side_at(contacts, 100) is None


def test_contact_side_at_skips_synthetic_contacts() -> None:
    contacts = [
        _make_contact_obj(100, "near", is_synthetic=True),
        _make_contact_obj(102, "far"),
    ]
    # Within ±3 of 100, synthetic skipped → falls through to "far" at 102
    assert _contact_side_at(contacts, 100) == "far"
