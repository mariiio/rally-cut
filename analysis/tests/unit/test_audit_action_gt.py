"""Smoke test that the audit CLI module imports cleanly."""
from rallycut.cli.commands.audit_action_gt import audit_action_gt_cmd, audit_video


def test_imports_resolve() -> None:
    assert callable(audit_action_gt_cmd)
    assert callable(audit_video)
