"""Test the pipeline-version-bump pre-commit hook block.

The hook lives in .claude/hooks/pre-commit-check.sh and is wired in
.claude/settings.json as a PreToolUse hook on Bash. It blocks `git commit`
calls when action_classifier.py or contact_detector.py is staged without
the corresponding *_PIPELINE_VERSION constant being bumped in the same
commit, unless the commit message contains `[no-version-bump]`.

We test the hook by feeding it crafted JSON tool-input payloads and
checking exit code + stderr.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[3]
HOOK_SCRIPT = PROJECT_DIR / ".claude" / "hooks" / "pre-commit-check.sh"


def _run_hook(
    *,
    command: str,
    staged_files: list[str],
    file_diffs: dict[str, str],
    monkeypatch_git: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run the hook with a synthetic git environment.

    Uses a tmpdir as fake $PROJECT_DIR by overriding GIT_DIR / GIT_WORK_TREE
    and stubbing `git diff --cached` via a wrapper on PATH.
    """
    # The hook reads $INPUT from stdin (the Claude-tool-input JSON).
    payload = json.dumps({"tool_input": {"command": command}})

    # For the version-bump block we only need: $STAGED list + git diff output.
    # The simplest approach: set up a real git repo with the staged
    # changes already applied, then call the hook with that as PROJECT_DIR.
    raise NotImplementedError("see Step 2 — implement using a tmp git repo")


def test_action_classifier_change_without_bump_is_blocked(tmp_path: Path) -> None:
    """Modifying action_classifier.py without bumping the constant fails."""
    # Test setup: seed a commit with v1, then stage a change that does NOT
    # modify that line (only adds a comment). The constant must not appear
    # as a new '+' line in the diff.
    repo = _make_repo(tmp_path)
    file = repo / "analysis/rallycut/tracking/action_classifier.py"
    _seed_file(repo, file, 'ACTION_PIPELINE_VERSION = "v1"\n')
    # Modify the file without changing the constant line.
    file.write_text('ACTION_PIPELINE_VERSION = "v1"\n# new docstring change\n')
    _stage(repo, file)

    result = _invoke_hook(repo, command='git commit -m "docs: fix typo"')

    assert result.returncode == 2, result.stderr
    assert "action_classifier.py modified without bumping ACTION_PIPELINE_VERSION" in result.stderr


def test_action_classifier_change_with_marker_is_allowed(tmp_path: Path) -> None:
    """Same edit passes when commit message has [no-version-bump]."""
    repo = _make_repo(tmp_path)
    file = repo / "analysis/rallycut/tracking/action_classifier.py"
    _seed_file(repo, file, 'ACTION_PIPELINE_VERSION = "v1"\n')
    # Modify the file without changing the constant line.
    file.write_text('ACTION_PIPELINE_VERSION = "v1"\n# new docstring change\n')
    _stage(repo, file)

    result = _invoke_hook(repo, command='git commit -m "docs: fix typo [no-version-bump]"')

    assert result.returncode == 0, result.stderr


def test_action_classifier_change_with_bump_is_allowed(tmp_path: Path) -> None:
    """Bumping the constant in the same commit passes."""
    repo = _make_repo(tmp_path)
    file = repo / "analysis/rallycut/tracking/action_classifier.py"
    # Initial commit has v1; new commit bumps to v2.
    _seed_file(repo, file, 'ACTION_PIPELINE_VERSION = "v1"\n')
    file.write_text('ACTION_PIPELINE_VERSION = "v2"\n')
    _stage(repo, file)

    result = _invoke_hook(repo, command='git commit -m "feat: change classifier"')

    assert result.returncode == 0, result.stderr


def test_unrelated_file_change_is_allowed(tmp_path: Path) -> None:
    """A commit that doesn't touch the watched files passes."""
    repo = _make_repo(tmp_path)
    file = repo / "analysis/rallycut/tracking/other.py"
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text("# unrelated\n")
    _stage(repo, file)

    result = _invoke_hook(repo, command='git commit -m "chore: other change"')

    assert result.returncode == 0, result.stderr


def test_contact_detector_change_without_bump_is_blocked(tmp_path: Path) -> None:
    """Same enforcement for contact_detector.py."""
    repo = _make_repo(tmp_path)
    file = repo / "analysis/rallycut/tracking/contact_detector.py"
    _seed_file(repo, file, 'CONTACT_PIPELINE_VERSION = "v1"\n')
    # Modify the file without changing the constant line.
    file.write_text('CONTACT_PIPELINE_VERSION = "v1"\n# new comment\n')
    _stage(repo, file)

    result = _invoke_hook(repo, command='git commit -m "tweak"')

    assert result.returncode == 2, result.stderr
    assert "contact_detector.py modified without bumping CONTACT_PIPELINE_VERSION" in result.stderr


# --- Helpers ----------------------------------------------------------------


def _make_repo(tmp_path: Path) -> Path:
    """Create a bare git repo with the directory layout the hook expects."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    (repo / "analysis/rallycut/tracking").mkdir(parents=True)
    # Empty .claude dir is fine; the hook reads PROJECT_DIR from a hardcoded
    # path so we override it via env in _invoke_hook.
    return repo


def _seed_file(repo: Path, file: Path, content: str) -> None:
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(content)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "seed"], cwd=repo, check=True)


def _stage(repo: Path, file: Path) -> None:
    subprocess.run(["git", "add", str(file.relative_to(repo))], cwd=repo, check=True)


def _invoke_hook(repo: Path, *, command: str) -> subprocess.CompletedProcess[str]:
    """Run the hook with PROJECT_DIR pointing at a tmp repo.

    The hook hard-codes PROJECT_DIR. We patch it for the test run by
    writing a temporary hook copy that substitutes the path.
    """
    src = HOOK_SCRIPT.read_text()
    patched = src.replace(
        'PROJECT_DIR="/Users/mario/Personal/Projects/RallyCut"',
        f'PROJECT_DIR="{repo}"',
    )
    tmp_hook = repo / "pre-commit-check.sh"
    tmp_hook.write_text(patched)
    tmp_hook.chmod(0o755)

    payload = json.dumps({"tool_input": {"command": command}})
    return subprocess.run(
        ["bash", str(tmp_hook)],
        input=payload,
        capture_output=True,
        text=True,
        cwd=repo,
    )
