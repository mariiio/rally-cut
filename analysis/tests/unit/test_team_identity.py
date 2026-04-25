"""Unit tests for team-template construction under canonical pids.

Pin the design decision: team membership in `team_templates` is derived
from per-rally positional team_assignments (mode-vote per pid → team),
NOT from the legacy `pid <= 2 → team 0` partition. The legacy partition
is the deterministic tiebreaker only.

Plan-doc: docs/superpowers/plans/2026-04-25-ref-crop-canonical-identity.md.
"""
from __future__ import annotations

import numpy as np

from rallycut.tracking.player_features import PlayerAppearanceProfile
from rallycut.tracking.team_identity import build_team_templates


def _profile(pid: int) -> PlayerAppearanceProfile:
    """Bare profile with a deterministic, distinguishable lower-body
    histogram so discriminability has signal in tests."""
    p = PlayerAppearanceProfile(player_id=pid, team=0)
    # Pid-distinct histogram so {1,2} / {3,4} discriminability is non-zero
    # under the legacy fallback path.
    p.avg_lower_hist = np.full(48, float(pid), dtype=np.float32)
    return p


def test_legacy_partition_when_no_positional_data() -> None:
    """No per-rally inputs → legacy `pid <= 2` partition is used."""
    profiles = {pid: _profile(pid) for pid in (1, 2, 3, 4)}
    t0, t1 = build_team_templates(profiles)
    assert t0.player_ids == [1, 2]
    assert t1.player_ids == [3, 4]


def test_positional_partition_overrides_legacy() -> None:
    """Per-rally votes say {1,3} on near, {2,4} on far → that wins."""
    profiles = {pid: _profile(pid) for pid in (1, 2, 3, 4)}
    # 3 rallies, all same partition: pid 1+3 near (team 0), pid 2+4 far (team 1)
    track_to_player_per_rally = [
        {10: 1, 20: 2, 30: 3, 40: 4},
        {10: 1, 20: 2, 30: 3, 40: 4},
        {10: 1, 20: 2, 30: 3, 40: 4},
    ]
    track_court_sides_per_rally = [
        {10: 0, 20: 1, 30: 0, 40: 1},
        {10: 0, 20: 1, 30: 0, 40: 1},
        {10: 0, 20: 1, 30: 0, 40: 1},
    ]
    t0, t1 = build_team_templates(
        profiles,
        track_to_player_per_rally=track_to_player_per_rally,
        track_court_sides_per_rally=track_court_sides_per_rally,
    )
    assert t0.player_ids == [1, 3]
    assert t1.player_ids == [2, 4]


def test_mode_vote_handles_per_rally_disagreement() -> None:
    """One stray rally puts pid 1 on the wrong side; majority still wins."""
    profiles = {pid: _profile(pid) for pid in (1, 2, 3, 4)}
    # Two rallies: pid 1 on team 0; one rally: pid 1 on team 1 (occlusion or
    # bad classification). Mode-vote → pid 1 stays team 0.
    track_to_player_per_rally = [
        {10: 1, 20: 2, 30: 3, 40: 4},
        {10: 1, 20: 2, 30: 3, 40: 4},
        {10: 1, 20: 2, 30: 3, 40: 4},
    ]
    track_court_sides_per_rally = [
        {10: 0, 20: 0, 30: 1, 40: 1},
        {10: 0, 20: 0, 30: 1, 40: 1},
        {10: 1, 20: 0, 30: 1, 40: 1},  # pid 1 anomalously on team 1 here
    ]
    t0, t1 = build_team_templates(
        profiles,
        track_to_player_per_rally=track_to_player_per_rally,
        track_court_sides_per_rally=track_court_sides_per_rally,
    )
    assert t0.player_ids == [1, 2]
    assert t1.player_ids == [3, 4]


def test_tie_vote_breaks_to_legacy_partition() -> None:
    """Equal team-0 and team-1 votes for pid → legacy partition wins.
    Keeps output bit-stable on degenerate inputs."""
    profiles = {pid: _profile(pid) for pid in (1, 2, 3, 4)}
    track_to_player_per_rally = [
        {10: 1, 20: 2, 30: 3, 40: 4},
        {10: 1, 20: 2, 30: 3, 40: 4},
    ]
    # Pid 1 votes split 1-1 between teams 0 and 1 → tie.
    track_court_sides_per_rally = [
        {10: 0, 20: 0, 30: 1, 40: 1},
        {10: 1, 20: 0, 30: 1, 40: 1},
    ]
    t0, t1 = build_team_templates(
        profiles,
        track_to_player_per_rally=track_to_player_per_rally,
        track_court_sides_per_rally=track_court_sides_per_rally,
    )
    # Legacy partition tiebreaker → pid 1 lands on team 0.
    assert 1 in t0.player_ids


def test_unobserved_pid_falls_back_to_legacy() -> None:
    """Pid never observed in any rally → legacy partition fills it in."""
    profiles = {pid: _profile(pid) for pid in (1, 2, 3, 4)}
    # pid 4 never appears in track_to_player → no votes → legacy default.
    track_to_player_per_rally = [{10: 1, 20: 2, 30: 3}]
    track_court_sides_per_rally = [{10: 0, 20: 0, 30: 1}]
    t0, t1 = build_team_templates(
        profiles,
        track_to_player_per_rally=track_to_player_per_rally,
        track_court_sides_per_rally=track_court_sides_per_rally,
    )
    # pid 4 keeps its legacy team (1).
    assert 4 in t1.player_ids


def test_input_length_mismatch_falls_back_to_legacy() -> None:
    """Mismatched per-rally list lengths → legacy partition (defensive)."""
    profiles = {pid: _profile(pid) for pid in (1, 2, 3, 4)}
    track_to_player_per_rally = [{10: 1, 20: 2, 30: 3, 40: 4}]
    track_court_sides_per_rally: list[dict[int, int]] = []  # length mismatch
    t0, t1 = build_team_templates(
        profiles,
        track_to_player_per_rally=track_to_player_per_rally,
        track_court_sides_per_rally=track_court_sides_per_rally,
    )
    assert t0.player_ids == [1, 2]
    assert t1.player_ids == [3, 4]


def test_deterministic_across_calls() -> None:
    """Same inputs → identical TeamTemplate output (no shuffle, no drift)."""
    profiles = {pid: _profile(pid) for pid in (1, 2, 3, 4)}
    track_to_player_per_rally = [{10: 1, 20: 2, 30: 3, 40: 4}]
    track_court_sides_per_rally = [{10: 0, 20: 1, 30: 0, 40: 1}]
    a0, a1 = build_team_templates(
        profiles,
        track_to_player_per_rally=track_to_player_per_rally,
        track_court_sides_per_rally=track_court_sides_per_rally,
    )
    b0, b1 = build_team_templates(
        profiles,
        track_to_player_per_rally=track_to_player_per_rally,
        track_court_sides_per_rally=track_court_sides_per_rally,
    )
    assert (a0.player_ids, a0.team_label) == (b0.player_ids, b0.team_label)
    assert (a1.player_ids, a1.team_label) == (b1.player_ids, b1.team_label)
