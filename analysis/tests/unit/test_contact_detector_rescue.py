"""Unit tests for the rescue branch in contact_detector.

Brief: docs/superpowers/briefs/2026-04-21-contact-rescue-stepback.md

The rescue rule accepts a GBM-rejected candidate iff:
    gbm_prob < _RESCUE_GBM_CEILING (0.10)
  AND
    seq_max_nonbg_within_5f >= _RESCUE_SEQ_FLOOR (0.95)

These tests lock the rule on the pure helper. Integration behaviour is
measured separately via the 68-fold A/B.
"""

from __future__ import annotations

import inspect

from rallycut.tracking.contact_detector import (
    _RESCUE_GBM_CEILING,
    _RESCUE_SEQ_FLOOR,
    _apply_rescue_branch,
    detect_contacts,
)


class TestApplyRescueBranch:
    """Tests for the pure helper `_apply_rescue_branch`."""

    def test_already_validated_stays_validated(self) -> None:
        """If the classifier already accepted, rescue is a no-op."""
        assert _apply_rescue_branch(
            is_validated=True, gbm_prob=0.5, seq_max_nonbg=0.0,
            enable_rescue=True,
        ) is True
        assert _apply_rescue_branch(
            is_validated=True, gbm_prob=0.5, seq_max_nonbg=0.0,
            enable_rescue=False,
        ) is True

    def test_flag_off_never_rescues(self) -> None:
        """When enable_rescue=False, rejected candidates stay rejected."""
        # Textbook rescue signature (gbm very low, seq very high)
        assert _apply_rescue_branch(
            is_validated=False, gbm_prob=0.02, seq_max_nonbg=0.99,
            enable_rescue=False,
        ) is False

    def test_rescue_fires_on_textbook_signature(self) -> None:
        """gbm << 0.10 and seq >> 0.95 → rescued when enabled.

        Values taken from brief's flagged cf4cdd43 f181 case
        (gbm=0.005, seq=0.978).
        """
        assert _apply_rescue_branch(
            is_validated=False, gbm_prob=0.005, seq_max_nonbg=0.978,
            enable_rescue=True,
        ) is True

    def test_rescue_boundary_just_below_gbm_ceiling(self) -> None:
        """gbm strictly below 0.10 passes the gbm condition."""
        # Just below ceiling and just at seq floor
        assert _apply_rescue_branch(
            is_validated=False,
            gbm_prob=_RESCUE_GBM_CEILING - 1e-6,
            seq_max_nonbg=_RESCUE_SEQ_FLOOR,
            enable_rescue=True,
        ) is True

    def test_rescue_does_not_fire_at_or_above_gbm_ceiling(self) -> None:
        """gbm in [0.10, threshold) no-man's land — NOT rescued.

        Mirrors the e2db8571 f357 case (gbm=0.161, seq=0.965) from the brief.
        """
        assert _apply_rescue_branch(
            is_validated=False, gbm_prob=_RESCUE_GBM_CEILING,
            seq_max_nonbg=0.99, enable_rescue=True,
        ) is False
        assert _apply_rescue_branch(
            is_validated=False, gbm_prob=0.161, seq_max_nonbg=0.965,
            enable_rescue=True,
        ) is False

    def test_rescue_does_not_fire_when_seq_below_floor(self) -> None:
        """seq < 0.95 — NOT rescued even with tiny gbm."""
        assert _apply_rescue_branch(
            is_validated=False, gbm_prob=0.02,
            seq_max_nonbg=_RESCUE_SEQ_FLOOR - 1e-6,
            enable_rescue=True,
        ) is False
        assert _apply_rescue_branch(
            is_validated=False, gbm_prob=0.02, seq_max_nonbg=0.90,
            enable_rescue=True,
        ) is False

    def test_rescue_constants_match_brief(self) -> None:
        """Lock the constants so they can't be silently re-tuned.

        The brief specifies exactly gbm<0.10 AND seq>=0.95; the kill-test
        verdict is a function of these numbers.
        """
        assert _RESCUE_GBM_CEILING == 0.10
        assert _RESCUE_SEQ_FLOOR == 0.95


class TestDetectContactsSignature:
    """Verify the enable_rescue kwarg is wired into the public API."""

    def test_enable_rescue_parameter_present_and_default_false(self) -> None:
        """enable_rescue must be on the public API and default to False."""
        sig = inspect.signature(detect_contacts)
        assert "enable_rescue" in sig.parameters, (
            "detect_contacts must expose enable_rescue for the A/B harness"
        )
        param = sig.parameters["enable_rescue"]
        assert param.default is False, (
            "enable_rescue must default to False so production and existing "
            "callers stay on the baseline path"
        )
