"""The audit CLI shells render the StaleVersionReport header."""

from __future__ import annotations

from unittest.mock import patch

from typer.testing import CliRunner

from rallycut.cli.main import app
from rallycut.tracking.pid_invariants import StaleVersionReport, Violation

runner = CliRunner()


def _empty_stale() -> StaleVersionReport:
    return StaleVersionReport(
        total_rallies=0,
        skipped_stale_actions=frozenset(),
        skipped_stale_contacts=frozenset(),
        current_actions_version="v1",
        current_contacts_version="v1",
        observed_actions_versions={},
        observed_contacts_versions={},
    )


def test_audit_pid_invariants_renders_stale_header() -> None:
    fake_violations: list[Violation] = []
    fake_stale = StaleVersionReport(
        total_rallies=10,
        skipped_stale_actions=frozenset({"rally-A", "rally-B"}),
        skipped_stale_contacts=frozenset(),
        current_actions_version="v1",
        current_contacts_version="v1",
        observed_actions_versions={"v0": 2, "v1": 8},
        observed_contacts_versions={"v1": 10},
    )
    with patch(
        "rallycut.cli.commands.audit_pid_invariants.run_all",
        return_value=(fake_violations, fake_stale),
    ):
        result = runner.invoke(app, ["audit-pid-invariants", "test-video-id"])

    # Header is rendered before any green/red summary.
    assert "2 of 10 rallies skipped" in result.stdout
    assert "actions_pipeline_version" in result.stdout
    assert "v0" in result.stdout
    # Stale-only does NOT fail the audit.
    assert result.exit_code == 0


def test_audit_pid_invariants_fails_on_error_violation() -> None:
    fake_violations = [Violation(invariant="I-1", rally_id="r1", detail="bad")]
    fake_stale = _empty_stale()
    with patch(
        "rallycut.cli.commands.audit_pid_invariants.run_all",
        return_value=(fake_violations, fake_stale),
    ):
        result = runner.invoke(app, ["audit-pid-invariants", "test-video-id"])

    assert result.exit_code == 1


def test_audit_pid_invariants_no_stale_no_violations_exits_clean() -> None:
    with patch(
        "rallycut.cli.commands.audit_pid_invariants.run_all",
        return_value=([], _empty_stale()),
    ):
        result = runner.invoke(app, ["audit-pid-invariants", "test-video-id"])

    assert result.exit_code == 0
    # No stale header printed when nothing is stale.
    assert "skipped due to stale" not in result.stdout


def test_audit_coherence_invariants_renders_stale_header() -> None:
    fake_violations: list[Violation] = []
    fake_stale = StaleVersionReport(
        total_rallies=5,
        skipped_stale_actions=frozenset({"r1"}),
        skipped_stale_contacts=frozenset(),
        current_actions_version="v1",
        current_contacts_version="v1",
        observed_actions_versions={"v0": 1, "v1": 4},
        observed_contacts_versions={"v1": 5},
    )
    with patch(
        "rallycut.cli.commands.audit_coherence_invariants.run_all",
        return_value=(fake_violations, fake_stale),
    ):
        result = runner.invoke(app, ["audit-coherence-invariants", "test-video-id"])

    assert "1 of 5" in result.stdout
    assert "actions_pipeline_version" in result.stdout
    assert result.exit_code == 0


def test_stale_only_does_not_fail_audit() -> None:
    """A run with ONLY stale rallies and no error violations exits 0."""
    fake_stale = StaleVersionReport(
        total_rallies=3,
        skipped_stale_actions=frozenset({"r1", "r2"}),
        skipped_stale_contacts=frozenset({"r3"}),
        current_actions_version="v1",
        current_contacts_version="v1",
        observed_actions_versions={"v0": 2, "v1": 1},
        observed_contacts_versions={"v0": 1, "v1": 2},
    )
    with patch(
        "rallycut.cli.commands.audit_pid_invariants.run_all",
        return_value=([], fake_stale),
    ):
        result = runner.invoke(app, ["audit-pid-invariants", "test-video-id"])

    assert result.exit_code == 0
    # Both action and contact stale lines should appear.
    assert "actions_pipeline_version" in result.stdout
    assert "contacts_pipeline_version" in result.stdout
