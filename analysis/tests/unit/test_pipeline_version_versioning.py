"""Pin the pipeline-version constants.

Mirrors tests/unit/test_assignment_anchor_versioning.py for MATCHER_VERSION.
This file pins the contract — bumping a pipeline version MUST invalidate
the previous one (we never write a version we've published before, except
v0 which is reserved for the migration sentinel).
"""

from __future__ import annotations

# Past published versions. Add entries here on every constant bump to
# prevent accidental reverts.
LEGACY_ACTION_VERSIONS: set[str] = set()
LEGACY_CONTACT_VERSIONS: set[str] = set()


def test_action_pipeline_version_is_nonempty_string() -> None:
    from rallycut.tracking.action_classifier import ACTION_PIPELINE_VERSION
    assert isinstance(ACTION_PIPELINE_VERSION, str)
    assert len(ACTION_PIPELINE_VERSION) > 0


def test_action_pipeline_version_is_not_v0_sentinel() -> None:
    from rallycut.tracking.action_classifier import ACTION_PIPELINE_VERSION
    assert ACTION_PIPELINE_VERSION != "v0", (
        "v0 is reserved for the migration backfill sentinel; never written by code"
    )


def test_action_pipeline_version_not_in_legacy() -> None:
    from rallycut.tracking.action_classifier import ACTION_PIPELINE_VERSION
    assert ACTION_PIPELINE_VERSION not in LEGACY_ACTION_VERSIONS, (
        f"ACTION_PIPELINE_VERSION={ACTION_PIPELINE_VERSION!r} is in the legacy set. "
        "Pick a fresh value rather than reverting."
    )


def test_contact_pipeline_version_is_nonempty_string() -> None:
    from rallycut.tracking.contact_detector import CONTACT_PIPELINE_VERSION
    assert isinstance(CONTACT_PIPELINE_VERSION, str)
    assert len(CONTACT_PIPELINE_VERSION) > 0


def test_contact_pipeline_version_is_not_v0_sentinel() -> None:
    from rallycut.tracking.contact_detector import CONTACT_PIPELINE_VERSION
    assert CONTACT_PIPELINE_VERSION != "v0"


def test_contact_pipeline_version_not_in_legacy() -> None:
    from rallycut.tracking.contact_detector import CONTACT_PIPELINE_VERSION
    assert CONTACT_PIPELINE_VERSION not in LEGACY_CONTACT_VERSIONS


def test_versions_are_independent() -> None:
    """Doc test: bumping one constant should not require bumping the other."""
    from rallycut.tracking.action_classifier import ACTION_PIPELINE_VERSION
    from rallycut.tracking.contact_detector import CONTACT_PIPELINE_VERSION
    # They will commonly be equal at v1 release time. The point of this
    # test is the docstring — kept as a guard against future code that
    # would assume strict equality.
    assert isinstance(ACTION_PIPELINE_VERSION, str)
    assert isinstance(CONTACT_PIPELINE_VERSION, str)
