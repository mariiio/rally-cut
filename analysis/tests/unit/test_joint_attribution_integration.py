"""Integration tests for the USE_JOINT_ATTRIBUTION env-flag wiring."""
from __future__ import annotations

import os


def test_use_joint_attribution_env_flag_default_off() -> None:
    """With USE_JOINT_ATTRIBUTION unset or '0', the flag-checking helper returns False."""
    from rallycut.tracking.joint_attribution import use_joint_attribution_enabled
    os.environ.pop("USE_JOINT_ATTRIBUTION", None)
    assert use_joint_attribution_enabled() is False
    os.environ["USE_JOINT_ATTRIBUTION"] = "0"
    assert use_joint_attribution_enabled() is False
    os.environ.pop("USE_JOINT_ATTRIBUTION", None)


def test_use_joint_attribution_env_flag_enabled() -> None:
    """With USE_JOINT_ATTRIBUTION='1', the flag-checking helper returns True."""
    from rallycut.tracking.joint_attribution import use_joint_attribution_enabled
    os.environ["USE_JOINT_ATTRIBUTION"] = "1"
    try:
        assert use_joint_attribution_enabled() is True
    finally:
        os.environ.pop("USE_JOINT_ATTRIBUTION", None)


def test_use_joint_attribution_env_flag_other_values_inert() -> None:
    """Only '1' enables; 'true', 'yes', 'on' are inert (matching project pattern)."""
    from rallycut.tracking.joint_attribution import use_joint_attribution_enabled
    for value in ("true", "yes", "on", ""):
        os.environ["USE_JOINT_ATTRIBUTION"] = value
        assert use_joint_attribution_enabled() is False
    os.environ.pop("USE_JOINT_ATTRIBUTION", None)
