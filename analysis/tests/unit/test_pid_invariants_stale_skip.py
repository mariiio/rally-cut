"""Stale-version skip semantics in pid_invariants.run_all.

I-3 + I-7 depend on actions_json -> skip if actions_pipeline_version is stale.
I-4 depends on contacts_json -> skip if contacts_pipeline_version is stale.
I-1, I-2, I-5, I-6, I-8 don't depend on that content -> unaffected.
"""

from __future__ import annotations

from rallycut.tracking.action_classifier import ACTION_PIPELINE_VERSION
from rallycut.tracking.contact_detector import CONTACT_PIPELINE_VERSION
from rallycut.tracking.pid_invariants import StaleVersionReport, Violation


def test_stale_version_report_dataclass_shape() -> None:
    report = StaleVersionReport(
        total_rallies=10,
        skipped_stale_actions=frozenset({"rally-A", "rally-B"}),
        skipped_stale_contacts=frozenset({"rally-A"}),
        current_actions_version=ACTION_PIPELINE_VERSION,
        current_contacts_version=CONTACT_PIPELINE_VERSION,
        observed_actions_versions={"v0": 2, "v1": 8},
        observed_contacts_versions={"v0": 1, "v1": 9},
    )
    assert report.total_rallies == 10
    assert "rally-A" in report.skipped_stale_actions
    assert report.has_stale is True


def test_run_all_returns_tuple_signature() -> None:
    """Smoke-test the signature: run_all returns (list, StaleVersionReport)."""
    from rallycut.tracking import pid_invariants

    # Use a known video id; if the DB is unreachable we get an exception
    # at the connection level, not a signature mismatch. Just verify the
    # function exists with the new return type by reading its source.
    import inspect

    sig = inspect.signature(pid_invariants.run_all)
    assert (
        "tuple[list[Violation], StaleVersionReport]" in str(sig.return_annotation)
        or "Tuple" in str(sig.return_annotation)
        or "tuple" in str(sig.return_annotation)
    ), f"run_all return type should be tuple; got {sig.return_annotation}"


def test_has_stale_false_when_no_skips() -> None:
    report = StaleVersionReport(
        total_rallies=5,
        skipped_stale_actions=frozenset(),
        skipped_stale_contacts=frozenset(),
        current_actions_version="v1",
        current_contacts_version="v1",
        observed_actions_versions={"v1": 5},
        observed_contacts_versions={"v1": 5},
    )
    assert report.has_stale is False
