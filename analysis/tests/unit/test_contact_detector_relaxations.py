"""Unit tests for ContactDetectionConfig relaxation fields and the
_resolve_effective_config helper. Each RELAX_CONTACT_* env flag gets a
test verifying flag-OFF preserves baseline behavior and flag-ON applies
the corresponding _relaxed value.

Spec: docs/superpowers/specs/2026-05-12-contact-detection-fn-reduction-design.md
"""
from __future__ import annotations

import pytest

from rallycut.tracking.contact_detector import (
    ContactDetectionConfig,
    _resolve_effective_config,
)


@pytest.fixture
def baseline_cfg() -> ContactDetectionConfig:
    return ContactDetectionConfig()


def test_default_no_flags_preserves_cfg(
    baseline_cfg: ContactDetectionConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no RELAX_CONTACT_* env flags set, _resolve_effective_config
    returns the input cfg unchanged."""
    for flag in (
        "RELAX_CONTACT_DIR_CHANGE",
        "RELAX_CONTACT_VELOCITY",
        "RELAX_CONTACT_INFLECTION",
        "RELAX_CONTACT_WARMUP",
        "RELAX_CONTACT_PLAYER_RADIUS",
    ):
        monkeypatch.delenv(flag, raising=False)
    resolved = _resolve_effective_config(baseline_cfg)
    assert resolved == baseline_cfg


def test_dir_change_flag_lowers_threshold(
    baseline_cfg: ContactDetectionConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RELAX_CONTACT_DIR_CHANGE", "1")
    resolved = _resolve_effective_config(baseline_cfg)
    assert resolved.min_direction_change_deg == baseline_cfg.min_direction_change_deg_relaxed
    assert resolved.min_direction_change_deg == 12.0  # current default
    # Other fields unchanged
    assert resolved.min_peak_velocity == baseline_cfg.min_peak_velocity


def test_velocity_flag_lowers_two_thresholds(
    baseline_cfg: ContactDetectionConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RELAX_CONTACT_VELOCITY", "1")
    resolved = _resolve_effective_config(baseline_cfg)
    assert resolved.min_peak_velocity == baseline_cfg.min_peak_velocity_relaxed
    assert resolved.deceleration_min_speed_before == (
        baseline_cfg.deceleration_min_speed_before_relaxed
    )
    assert resolved.min_peak_velocity == 0.005


def test_inflection_flag_lowers_threshold(
    baseline_cfg: ContactDetectionConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RELAX_CONTACT_INFLECTION", "1")
    resolved = _resolve_effective_config(baseline_cfg)
    assert resolved.min_inflection_angle_deg == baseline_cfg.min_inflection_angle_deg_relaxed
    assert resolved.min_inflection_angle_deg == 10.0


def test_warmup_flag_lowers_skip_frames(
    baseline_cfg: ContactDetectionConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RELAX_CONTACT_WARMUP", "1")
    resolved = _resolve_effective_config(baseline_cfg)
    assert resolved.warmup_skip_frames == baseline_cfg.warmup_skip_frames_relaxed
    assert resolved.warmup_skip_frames == 2


def test_player_radius_flag_increases_radius(
    baseline_cfg: ContactDetectionConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RELAX_CONTACT_PLAYER_RADIUS", "1")
    resolved = _resolve_effective_config(baseline_cfg)
    assert resolved.player_contact_radius == baseline_cfg.player_contact_radius_relaxed
    assert resolved.player_contact_radius == 0.20


def test_combined_flags_apply_additively(
    baseline_cfg: ContactDetectionConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Multiple flags ON simultaneously all apply; no interaction."""
    for flag in (
        "RELAX_CONTACT_DIR_CHANGE",
        "RELAX_CONTACT_VELOCITY",
        "RELAX_CONTACT_INFLECTION",
        "RELAX_CONTACT_WARMUP",
        "RELAX_CONTACT_PLAYER_RADIUS",
    ):
        monkeypatch.setenv(flag, "1")
    resolved = _resolve_effective_config(baseline_cfg)
    assert resolved.min_direction_change_deg == baseline_cfg.min_direction_change_deg_relaxed
    assert resolved.min_peak_velocity == baseline_cfg.min_peak_velocity_relaxed
    assert resolved.deceleration_min_speed_before == (
        baseline_cfg.deceleration_min_speed_before_relaxed
    )
    assert resolved.min_inflection_angle_deg == baseline_cfg.min_inflection_angle_deg_relaxed
    assert resolved.warmup_skip_frames == baseline_cfg.warmup_skip_frames_relaxed
    assert resolved.player_contact_radius == baseline_cfg.player_contact_radius_relaxed


def test_flag_value_other_than_one_is_ignored(
    baseline_cfg: ContactDetectionConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only RELAX_CONTACT_*='1' activates relaxation. '0', 'true', empty are inert."""
    for value in ("0", "", "true", "yes"):
        monkeypatch.setenv("RELAX_CONTACT_DIR_CHANGE", value)
        resolved = _resolve_effective_config(baseline_cfg)
        assert resolved.min_direction_change_deg == baseline_cfg.min_direction_change_deg
