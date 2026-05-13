"""Stale-actions skip for the coherence audit.

All four coherence invariants (C-1..C-4) read actions_json. A stale
actions_pipeline_version means the rally is excluded from the report.
"""

from __future__ import annotations

from unittest.mock import patch

from rallycut.tracking.action_classifier import ACTION_PIPELINE_VERSION
from rallycut.tracking.contact_detector import CONTACT_PIPELINE_VERSION
from rallycut.tracking.pid_invariants import StaleVersionReport


def _stale_report_with_skip(rally_ids: list[str]) -> StaleVersionReport:
    return StaleVersionReport(
        total_rallies=len(rally_ids) + 1,
        skipped_stale_actions=frozenset(rally_ids),
        skipped_stale_contacts=frozenset(),
        current_actions_version=ACTION_PIPELINE_VERSION,
        current_contacts_version=CONTACT_PIPELINE_VERSION,
        observed_actions_versions={"v0": len(rally_ids), ACTION_PIPELINE_VERSION: 1},
        observed_contacts_versions={CONTACT_PIPELINE_VERSION: len(rally_ids) + 1},
    )


def test_run_all_returns_tuple_signature() -> None:
    """coherence_invariants.run_all returns (violations, StaleVersionReport)."""
    import inspect

    from rallycut.tracking import coherence_invariants
    sig = inspect.signature(coherence_invariants.run_all)
    assert "tuple" in str(sig.return_annotation).lower()


def test_run_all_skips_stale_rallies_from_check_dispatch() -> None:
    """A rally listed in pid_stale.skipped_stale_actions is excluded from C-1..C-4."""
    from rallycut.tracking import coherence_invariants

    fake_pid_stale = _stale_report_with_skip(["rally-stale"])

    with patch(
        "rallycut.tracking.coherence_invariants.pid_run_all",
        return_value=([], fake_pid_stale),
    ), patch(
        "rallycut.tracking.coherence_invariants.get_connection",
    ) as mock_conn:
        # Two rallies: one stale, one current. Both have actions_json content.
        cursor_mock = mock_conn.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value
        cursor_mock.fetchall.return_value = [
            ("rally-stale", {"actions": [{"frame": 0, "playerTrackId": 1, "action": "serve"}],
                            "teamAssignments": {"1": "A"}}),
            ("rally-current", {"actions": [{"frame": 0, "playerTrackId": 1, "action": "serve"}],
                               "teamAssignments": {"1": "A"}}),
        ]

        violations, stale = coherence_invariants.run_all(video_id="vid")

    # Stale rally must not appear in any violation rally_id.
    rally_ids_in_violations = {v.rally_id for v in violations}
    assert "rally-stale" not in rally_ids_in_violations
    # Returned stale report is pass-through from pid_run_all.
    assert "rally-stale" in stale.skipped_stale_actions


def test_run_all_passes_through_stale_report_unchanged() -> None:
    """The returned StaleVersionReport is the same object pid_run_all produced."""
    from rallycut.tracking import coherence_invariants

    fake_pid_stale = _stale_report_with_skip([])

    with patch(
        "rallycut.tracking.coherence_invariants.pid_run_all",
        return_value=([], fake_pid_stale),
    ), patch(
        "rallycut.tracking.coherence_invariants.get_connection",
    ) as mock_conn:
        cursor_mock = mock_conn.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value
        cursor_mock.fetchall.return_value = []

        _violations, stale = coherence_invariants.run_all(video_id="vid")

    assert stale is fake_pid_stale  # identity check — pass-through
