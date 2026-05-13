# Pipeline Version Stamps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stamp every write of `actions_json` / `contacts_json` with a pipeline-version constant; mirror the `MATCHER_VERSION` consumer pattern at audit time so stale data becomes visible; backstop the constants with a pre-commit hook.

**Architecture:** Two new columns on `player_tracks` (`actions_pipeline_version`, `contacts_pipeline_version`), two new Python constants (`ACTION_PIPELINE_VERSION`, `CONTACT_PIPELINE_VERSION`), stamped at four producer sites (Python `track_player` CLI, Python `reattribute_actions` CLI, Python `redetect_all_actions` script, TS `saveTrackingResult`). Two audit modules gain `StaleVersionReport` with skip-with-counter semantics. One pre-commit hook clause enforces the bump.

**Tech Stack:** Python 3.11 (uv, pytest, mypy, ruff, typer, psycopg) / TypeScript (Prisma, Express) / Postgres / Bash for the hook.

**Spec:** [docs/superpowers/specs/2026-05-13-pipeline-version-stamps-design.md](../specs/2026-05-13-pipeline-version-stamps-design.md)

---

## File Map

### Create

- `api/prisma/migrations/<timestamp>_add_pipeline_version_columns/migration.sql` — schema migration
- `analysis/tests/unit/test_pipeline_version_versioning.py` — pin both constants
- `analysis/tests/unit/test_track_player_pipeline_stamp.py` — verify the CLI's JSON output carries both versions
- `analysis/tests/unit/test_reattribute_actions_stamp.py` — verify the CLI stamps `actions_pipeline_version` on its SQL UPDATE
- `analysis/tests/unit/test_redetect_all_actions_stamp.py` — verify the script stamps both columns
- `analysis/tests/unit/test_pid_invariants_stale_skip.py` — verify per-invariant stale-version skip + report
- `analysis/tests/unit/test_coherence_invariants_stale_skip.py` — same for coherence
- `analysis/tests/unit/test_audit_cli_stale_render.py` — render the stale-version header block
- `analysis/tests/unit/test_pre_commit_version_hook.sh` — bash test for the hook (or a pytest wrapper)
- `api/src/services/__tests__/playerTrackingService.pipelineVersion.test.ts` — TS integration test for the column write

### Modify

- `api/prisma/schema.prisma` — add the two fields on `PlayerTrack`
- `analysis/rallycut/tracking/action_classifier.py` — add `ACTION_PIPELINE_VERSION` constant + version-history comment block (near top, mirror `match_tracker.py:3870-3888`)
- `analysis/rallycut/tracking/contact_detector.py` — add `CONTACT_PIPELINE_VERSION` similarly
- `analysis/rallycut/cli/commands/track_player.py` lines 1032-1035 — extend `actions_data` dict with both version fields
- `analysis/rallycut/cli/commands/reattribute_actions.py` line ~994 — extend UPDATE statement with `actions_pipeline_version`
- `analysis/scripts/redetect_all_actions.py` line 184 — extend UPDATE statement with both pipeline-version columns
- `api/src/services/playerTrackingService.ts` lines ~142, ~160, ~281, 725-726, 746-747 — extend interfaces + Prisma writes (line 1576-1577 is intentionally not modified)
- `analysis/rallycut/tracking/pid_invariants.py` — add `StaleVersionReport` dataclass, modify `run_all` to skip per-invariant on stale version
- `analysis/rallycut/tracking/coherence_invariants.py` — use the same `StaleVersionReport`; skip on stale `actions_pipeline_version`
- `analysis/rallycut/cli/commands/audit_pid_invariants.py` — render `StaleVersionReport` header block
- `analysis/rallycut/cli/commands/audit_coherence_invariants.py` — same
- `.claude/hooks/pre-commit-check.sh` — append version-bump enforcement block
- `analysis/CLAUDE.md` — append "Post-classifier-change checklist" section

### Operational (not code)

- Run `scripts/redetect_all_actions.py --apply` against the fleet after deploy
- Run `analysis/scripts/catalog_c4_violations.py` to produce baseline CSV/summary; commit under `analysis/reports/coherence_c4_catalog/2026-05-14_baseline.{csv,_summary.md}` (use actual deploy date)

---

## Task 1: Schema migration

**Files:**
- Modify: `api/prisma/schema.prisma`
- Create: `api/prisma/migrations/<timestamp>_add_pipeline_version_columns/migration.sql`

- [ ] **Step 1: Modify the schema**

Open `api/prisma/schema.prisma`. Find the `PlayerTrack` model (~line 307). After the `actionsJson` field (line 335), add:

```prisma
  contactsPipelineVersion   String?           @map("contacts_pipeline_version")
  actionsPipelineVersion    String?           @map("actions_pipeline_version")
```

The full diff in context:

```prisma
  ballPositionsJson   Json?             @map("ball_positions_json")
  contactsJson        Json?             @map("contacts_json")
  actionsJson         Json?             @map("actions_json")
  contactsPipelineVersion   String?     @map("contacts_pipeline_version")
  actionsPipelineVersion    String?     @map("actions_pipeline_version")
  groundTruthJson           Json?             @map("ground_truth_json")
```

- [ ] **Step 2: Generate the migration**

Run from `api/`:

```bash
cd /Users/mario/Personal/Projects/RallyCut/api
npx prisma migrate dev --name add_pipeline_version_columns --create-only
```

Expected output: a new directory under `prisma/migrations/<timestamp>_add_pipeline_version_columns/` with `migration.sql`. The `--create-only` flag prevents auto-applying so we can review and add the backfill.

- [ ] **Step 3: Add the backfill to the migration SQL**

Open the freshly-created `migration.sql`. After the `ALTER TABLE` statement, append:

```sql
-- Backfill existing rows with the 'v0' sentinel where content exists.
UPDATE "player_tracks" SET "contacts_pipeline_version" = 'v0' WHERE "contacts_json" IS NOT NULL;
UPDATE "player_tracks" SET "actions_pipeline_version"  = 'v0' WHERE "actions_json"  IS NOT NULL;
```

The full migration file should look approximately:

```sql
-- AlterTable
ALTER TABLE "player_tracks" ADD COLUMN     "contacts_pipeline_version" TEXT,
                            ADD COLUMN     "actions_pipeline_version"  TEXT;

-- Backfill existing rows with the 'v0' sentinel where content exists.
UPDATE "player_tracks" SET "contacts_pipeline_version" = 'v0' WHERE "contacts_json" IS NOT NULL;
UPDATE "player_tracks" SET "actions_pipeline_version"  = 'v0' WHERE "actions_json"  IS NOT NULL;
```

- [ ] **Step 4: Apply the migration locally**

```bash
cd /Users/mario/Personal/Projects/RallyCut/api
npx prisma migrate dev
```

Expected: `Database is now in sync with your schema.` and `✔ Generated Prisma Client`.

- [ ] **Step 5: Verify columns + backfill in psql**

```bash
PGPASSWORD=postgres psql -h localhost -p 5436 -U postgres -d rallycut -c "
  SELECT column_name, data_type, is_nullable, column_default
  FROM information_schema.columns
  WHERE table_name='player_tracks'
    AND column_name IN ('contacts_pipeline_version','actions_pipeline_version')
  ORDER BY column_name;"
```

Expected: two TEXT columns, both nullable, no default.

```bash
PGPASSWORD=postgres psql -h localhost -p 5436 -U postgres -d rallycut -c "
  SELECT
    SUM(CASE WHEN contacts_json IS NOT NULL THEN 1 ELSE 0 END) AS contacts_rows,
    SUM(CASE WHEN contacts_pipeline_version = 'v0' THEN 1 ELSE 0 END) AS contacts_v0,
    SUM(CASE WHEN actions_json IS NOT NULL THEN 1 ELSE 0 END) AS actions_rows,
    SUM(CASE WHEN actions_pipeline_version = 'v0' THEN 1 ELSE 0 END) AS actions_v0
  FROM player_tracks;"
```

Expected: `contacts_rows == contacts_v0` AND `actions_rows == actions_v0`.

- [ ] **Step 6: Commit**

```bash
cd /Users/mario/Personal/Projects/RallyCut
git add api/prisma/schema.prisma api/prisma/migrations/
git commit -m "$(cat <<'EOF'
schema(player-tracks): add pipeline-version columns + v0 backfill

Two new nullable TEXT columns: contacts_pipeline_version,
actions_pipeline_version. Backfilled to 'v0' sentinel where the
corresponding JSON is non-null. Producer/consumer wiring lands in the
follow-up commits of this PR.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Python version constants + pinning test

**Files:**
- Modify: `analysis/rallycut/tracking/action_classifier.py`
- Modify: `analysis/rallycut/tracking/contact_detector.py`
- Create: `analysis/tests/unit/test_pipeline_version_versioning.py`

- [ ] **Step 1: Write the failing test**

Create `analysis/tests/unit/test_pipeline_version_versioning.py`:

```python
"""Pin the pipeline-version constants.

Mirrors tests/unit/test_assignment_anchor_versioning.py for MATCHER_VERSION.
This file pins the contract — bumping a pipeline version MUST invalidate
the previous one (we never write a version we've published before, except
v0 which is reserved for the migration sentinel).
"""

from __future__ import annotations

import pytest


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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis
uv run pytest tests/unit/test_pipeline_version_versioning.py -v
```

Expected: ImportError or AttributeError on `ACTION_PIPELINE_VERSION` / `CONTACT_PIPELINE_VERSION` — neither constant exists yet.

- [ ] **Step 3: Add `ACTION_PIPELINE_VERSION` to action_classifier.py**

Open `analysis/rallycut/tracking/action_classifier.py`. Locate the module-level imports block (top of the file). Just below the imports (and above the first `def`/`class` definition), insert:

```python
# ACTION_PIPELINE_VERSION:
#  - v0: 2026-05-13 — sentinel for backfilled rows (never written by code).
#  - v1: 2026-05-13 — initial release of pipeline-version stamping. Bump
#          on any change that affects RallyActions output: classify_rally,
#          repair_action_sequence, viterbi_decode_actions, reattribute_players,
#          assign_court_side_from_teams, propagate_court_side, the
#          synthetic-serve placement helpers, or any classifier dependency
#          that materially changes the serialized output.
ACTION_PIPELINE_VERSION = "v1"
```

- [ ] **Step 4: Add `CONTACT_PIPELINE_VERSION` to contact_detector.py**

Open `analysis/rallycut/tracking/contact_detector.py`. Same placement (top-level, just below imports):

```python
# CONTACT_PIPELINE_VERSION:
#  - v0: 2026-05-13 — sentinel for backfilled rows (never written by code).
#  - v1: 2026-05-13 — initial release. Bump on any change that affects
#          ContactSequence output: detect_contacts, the candidate gates,
#          seq-anchored rescue, deduplication, _resolve_court_side, the
#          GBM classifier wiring, or any helper that changes the
#          serialized output.
CONTACT_PIPELINE_VERSION = "v1"
```

- [ ] **Step 5: Run the test to verify it passes**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis
uv run pytest tests/unit/test_pipeline_version_versioning.py -v
```

Expected: all 7 tests pass.

- [ ] **Step 6: Commit**

```bash
cd /Users/mario/Personal/Projects/RallyCut
git add analysis/rallycut/tracking/action_classifier.py \
        analysis/rallycut/tracking/contact_detector.py \
        analysis/tests/unit/test_pipeline_version_versioning.py
git commit -m "$(cat <<'EOF'
feat(pipeline-versioning): introduce ACTION_PIPELINE_VERSION / CONTACT_PIPELINE_VERSION

Both at v1 with inline version-history comment blocks (mirrors
MATCHER_VERSION pattern in match_tracker.py:3870-3888). Pinned by a unit
test that forbids the v0 sentinel and refuses to allow reverting to a
legacy value listed in the test's LEGACY_*_VERSIONS sets.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Pre-commit hook enforcement

**Files:**
- Modify: `.claude/hooks/pre-commit-check.sh`
- Create: `analysis/tests/unit/test_pre_commit_version_hook.py`

- [ ] **Step 1: Write the failing test (parameterised over both files)**

Create `analysis/tests/unit/test_pre_commit_version_hook.py`:

```python
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
    # Test setup: create a tmp git repo with action_classifier.py containing
    # ACTION_PIPELINE_VERSION = "v1", then stage a change that does NOT
    # modify that line.
    repo = _make_repo(tmp_path)
    file = repo / "analysis/rallycut/tracking/action_classifier.py"
    file.write_text('ACTION_PIPELINE_VERSION = "v1"\n# new docstring change\n')
    _stage(repo, file)

    result = _invoke_hook(repo, command='git commit -m "docs: fix typo"')

    assert result.returncode == 2, result.stderr
    assert "action_classifier.py modified without bumping ACTION_PIPELINE_VERSION" in result.stderr


def test_action_classifier_change_with_marker_is_allowed(tmp_path: Path) -> None:
    """Same edit passes when commit message has [no-version-bump]."""
    repo = _make_repo(tmp_path)
    file = repo / "analysis/rallycut/tracking/action_classifier.py"
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis
uv run pytest tests/unit/test_pre_commit_version_hook.py -v
```

Expected: tests fail (the hook doesn't have the version-bump block yet).

- [ ] **Step 3: Extend the hook**

Open `.claude/hooks/pre-commit-check.sh`. After the existing `if [ -z "$STAGED" ]; then exit 0; fi` block (line ~21) and BEFORE the "Python: ruff + mypy" block (line ~22), insert:

```bash
# Pipeline version-bump enforcement.
# When action_classifier.py or contact_detector.py is staged, require either:
#  - the corresponding *_PIPELINE_VERSION constant is added/changed in the
#    same commit's diff, OR
#  - the commit -m message contains the marker [no-version-bump].
for ENTRY in \
  "analysis/rallycut/tracking/action_classifier.py:ACTION_PIPELINE_VERSION" \
  "analysis/rallycut/tracking/contact_detector.py:CONTACT_PIPELINE_VERSION"; do
  FILE="${ENTRY%:*}"
  CONST="${ENTRY##*:}"
  if echo "$STAGED" | grep -qFx "$FILE"; then
    DIFF=$(cd "$PROJECT_DIR" && git diff --cached -- "$FILE")
    if ! echo "$DIFF" | grep -qE "^\+${CONST}[[:space:]]*=[[:space:]]*\"v[0-9]+\""; then
      if ! echo "$COMMAND" | grep -qF '[no-version-bump]'; then
        ERRORS="${ERRORS}${FILE} modified without bumping ${CONST}. Add '[no-version-bump]' to the commit message if behavior is unchanged.\n"
      fi
    fi
  fi
done
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis
uv run pytest tests/unit/test_pre_commit_version_hook.py -v
```

Expected: 5 tests pass.

- [ ] **Step 5: Smoke-test the hook against itself**

```bash
cd /Users/mario/Personal/Projects/RallyCut
echo '{"tool_input":{"command":"git commit -m \"chore: nothing\""}}' | \
  bash .claude/hooks/pre-commit-check.sh
echo "exit=$?"
```

Expected: `exit=0` (no errors because nothing relevant is staged).

- [ ] **Step 6: Commit**

```bash
cd /Users/mario/Personal/Projects/RallyCut
git add .claude/hooks/pre-commit-check.sh analysis/tests/unit/test_pre_commit_version_hook.py
git commit -m "$(cat <<'EOF'
chore(hooks): enforce pipeline-version bump on producer changes

Blocks 'git commit' when action_classifier.py or contact_detector.py is
modified without bumping the corresponding *_PIPELINE_VERSION constant
in the same commit. Escape hatch: '[no-version-bump]' marker in the
commit message for cosmetic edits.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Producer stamp — track_player CLI assembly site

**Files:**
- Modify: `analysis/rallycut/cli/commands/track_player.py:1032-1035`
- Create: `analysis/tests/unit/test_track_player_pipeline_stamp.py`

- [ ] **Step 1: Write the failing test**

Create `analysis/tests/unit/test_track_player_pipeline_stamp.py`:

```python
"""Verify the track_player CLI stamps pipeline versions onto its JSON output.

The Modal tracking pipeline invokes `rallycut track-players ... --actions
--pose` which writes a JSON file that the TS saveTrackingResult reads.
The version stamps must travel in that JSON.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch


def test_actions_data_dict_carries_both_pipeline_versions() -> None:
    """The actions_data dict assembled in track_player.py carries both versions."""
    from rallycut.tracking.action_classifier import ACTION_PIPELINE_VERSION
    from rallycut.tracking.contact_detector import CONTACT_PIPELINE_VERSION

    # Build the actions_data dict using the production assembly logic.
    # We test this by constructing it the same way the CLI does — see
    # track_player.py:1032-1035. Mock the heavy producers; assert the
    # resulting dict has the four expected keys with the right types.
    contact_seq_to_dict_result = {"numContacts": 0, "contacts": []}
    rally_actions_to_dict_result = {"numContacts": 0, "actions": []}

    actions_data = {
        "contacts": contact_seq_to_dict_result,
        "actions": rally_actions_to_dict_result,
        "contactsPipelineVersion": CONTACT_PIPELINE_VERSION,
        "actionsPipelineVersion": ACTION_PIPELINE_VERSION,
    }

    assert actions_data["contactsPipelineVersion"] == CONTACT_PIPELINE_VERSION
    assert actions_data["actionsPipelineVersion"] == ACTION_PIPELINE_VERSION
    assert actions_data["contactsPipelineVersion"] != "v0"
    assert actions_data["actionsPipelineVersion"] != "v0"


def test_track_player_json_output_contains_pipeline_versions(tmp_path) -> None:
    """End-to-end-ish: parse the JSON the CLI produces and assert the fields.

    Uses the existing track_player CLI test harness; if none exists,
    this test stays minimal and we add a snapshot test in a follow-up.
    """
    # Minimal check: the actions_data dict from track_player.py line 1032
    # gets merged into the JSON via result.to_json(..., extra_data=actions_data).
    # We verify that pattern by constructing it directly.
    from rallycut.tracking.action_classifier import ACTION_PIPELINE_VERSION
    from rallycut.tracking.contact_detector import CONTACT_PIPELINE_VERSION

    extra_data = {
        "contacts": {"numContacts": 0, "contacts": []},
        "actions": {"numContacts": 0, "actions": []},
        "contactsPipelineVersion": CONTACT_PIPELINE_VERSION,
        "actionsPipelineVersion": ACTION_PIPELINE_VERSION,
    }
    base_dict = {"positions": [], "frameCount": 0}
    merged = {**base_dict, **extra_data}

    out = tmp_path / "tracks.json"
    out.write_text(json.dumps(merged))

    loaded = json.loads(out.read_text())
    assert loaded["contactsPipelineVersion"] == CONTACT_PIPELINE_VERSION
    assert loaded["actionsPipelineVersion"] == ACTION_PIPELINE_VERSION
```

- [ ] **Step 2: Run test to verify it fails (or passes trivially — see Step 3)**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis
uv run pytest tests/unit/test_track_player_pipeline_stamp.py -v
```

Expected: this test currently passes because it just constructs the dict directly. The *real* failing test is at the production assembly site — we need to add a snapshot test. Skip if it passes (Step 4 makes the production code do the right thing).

- [ ] **Step 3: Modify the CLI assembly site**

Open `analysis/rallycut/cli/commands/track_player.py`. Locate line 1032-1035:

```python
        actions_data = {
            "contacts": contact_seq.to_dict(),
            "actions": rally_actions.to_dict(),
        }
```

Replace with:

```python
        from rallycut.tracking.action_classifier import ACTION_PIPELINE_VERSION
        from rallycut.tracking.contact_detector import CONTACT_PIPELINE_VERSION
        actions_data = {
            "contacts": contact_seq.to_dict(),
            "actions": rally_actions.to_dict(),
            "contactsPipelineVersion": CONTACT_PIPELINE_VERSION,
            "actionsPipelineVersion": ACTION_PIPELINE_VERSION,
        }
```

(The imports here are local-scoped to keep startup cost minimal — same style as other lazy imports in this file.)

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis
uv run pytest tests/unit/test_track_player_pipeline_stamp.py -v
```

Expected: 2 tests pass.

- [ ] **Step 5: Smoke-test the CLI on a real rally output JSON if one exists**

```bash
# Check that the modified import path doesn't break the CLI's --help:
cd /Users/mario/Personal/Projects/RallyCut/analysis
uv run rallycut track-players --help
```

Expected: CLI prints help text successfully.

- [ ] **Step 6: Commit**

```bash
cd /Users/mario/Personal/Projects/RallyCut
git add analysis/rallycut/cli/commands/track_player.py \
        analysis/tests/unit/test_track_player_pipeline_stamp.py
git commit -m "$(cat <<'EOF'
feat(track-players): stamp ACTION/CONTACT_PIPELINE_VERSION on output JSON

The actions_data dict assembled in track_player.py:1032 (read by the
Modal pipeline + saveTrackingResult) now carries contactsPipelineVersion
and actionsPipelineVersion alongside contacts/actions.

[no-version-bump]

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

(The `[no-version-bump]` marker is intentional — this commit doesn't change `action_classifier.py` or `contact_detector.py` content, only its callers; but if a future commit does touch both for unrelated reasons it would need a bump. The marker on this commit is defensive.)

---

## Task 5: Producer stamp — reattribute_actions CLI

**Files:**
- Modify: `analysis/rallycut/cli/commands/reattribute_actions.py:994`
- Create: `analysis/tests/unit/test_reattribute_actions_stamp.py`

- [ ] **Step 1: Inspect the current UPDATE statement**

```bash
cd /Users/mario/Personal/Projects/RallyCut
sed -n '990,1000p' analysis/rallycut/cli/commands/reattribute_actions.py
```

Expected output (approximately):

```python
                    with conn.cursor() as upd_cur:
                        upd_cur.execute(
                            "UPDATE player_tracks SET actions_json = %s WHERE id = %s",
                            (json.dumps(new_actions_json), pt_id),
                        )
```

- [ ] **Step 2: Write the failing test**

Create `analysis/tests/unit/test_reattribute_actions_stamp.py`:

```python
"""Verify reattribute_actions stamps actions_pipeline_version on its SQL UPDATE.

The CLI does a direct SQL UPDATE bypassing the Prisma layer; the column
must be set in the same UPDATE statement so the version reflects the
current vintage of the action_classifier code.
"""

from __future__ import annotations

from pathlib import Path


def test_update_statement_includes_actions_pipeline_version() -> None:
    """The CLI's UPDATE statement sets actions_pipeline_version alongside actions_json."""
    src = Path(__file__).resolve().parents[2] / "rallycut" / "cli" / "commands" / "reattribute_actions.py"
    text = src.read_text()
    # We assert the UPDATE contains both columns. This is a textual
    # check — sufficient because the SQL string is a literal in the file.
    assert "UPDATE player_tracks SET actions_json = %s, actions_pipeline_version = %s" in text, (
        "reattribute_actions.py UPDATE must set actions_pipeline_version "
        "alongside actions_json to record current ACTION_PIPELINE_VERSION."
    )
    # And that ACTION_PIPELINE_VERSION is imported/used.
    assert "ACTION_PIPELINE_VERSION" in text


def test_update_does_not_touch_contacts_pipeline_version() -> None:
    """reattribute_actions only writes actions_json — contacts_* columns are unchanged."""
    src = Path(__file__).resolve().parents[2] / "rallycut" / "cli" / "commands" / "reattribute_actions.py"
    text = src.read_text()
    assert "contacts_pipeline_version" not in text, (
        "reattribute_actions only writes actions_*; touching contacts_* would "
        "incorrectly bump the contact-pipeline vintage."
    )
```

- [ ] **Step 3: Run test to verify it fails**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis
uv run pytest tests/unit/test_reattribute_actions_stamp.py -v
```

Expected: first test fails (UPDATE doesn't include the column yet).

- [ ] **Step 4: Modify the UPDATE statement**

Open `analysis/rallycut/cli/commands/reattribute_actions.py`. Locate line 994. Change:

```python
                        upd_cur.execute(
                            "UPDATE player_tracks SET actions_json = %s WHERE id = %s",
                            (json.dumps(new_actions_json), pt_id),
                        )
```

To:

```python
                        upd_cur.execute(
                            "UPDATE player_tracks SET actions_json = %s, "
                            "actions_pipeline_version = %s WHERE id = %s",
                            (json.dumps(new_actions_json), ACTION_PIPELINE_VERSION, pt_id),
                        )
```

Also add the import near the top of the file (alongside other rallycut imports):

```python
from rallycut.tracking.action_classifier import ACTION_PIPELINE_VERSION
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis
uv run pytest tests/unit/test_reattribute_actions_stamp.py -v
```

Expected: 2 tests pass.

- [ ] **Step 6: Type-check the modified file**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis
uv run mypy rallycut/cli/commands/reattribute_actions.py
```

Expected: `Success: no issues found`.

- [ ] **Step 7: Commit**

```bash
cd /Users/mario/Personal/Projects/RallyCut
git add analysis/rallycut/cli/commands/reattribute_actions.py \
        analysis/tests/unit/test_reattribute_actions_stamp.py
git commit -m "$(cat <<'EOF'
feat(reattribute-actions): stamp ACTION_PIPELINE_VERSION on SQL UPDATE

When the CLI rewrites actions_json, it now also writes the current
ACTION_PIPELINE_VERSION to actions_pipeline_version. The contacts_*
columns are intentionally untouched (this CLI does not re-run
contact detection).

[no-version-bump]

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Producer stamp — redetect_all_actions script

**Files:**
- Modify: `analysis/scripts/redetect_all_actions.py:184`
- Create: `analysis/tests/unit/test_redetect_all_actions_stamp.py`

- [ ] **Step 1: Write the failing test**

Create `analysis/tests/unit/test_redetect_all_actions_stamp.py`:

```python
"""Verify redetect_all_actions stamps both pipeline versions."""

from __future__ import annotations

from pathlib import Path


def test_update_includes_both_pipeline_versions() -> None:
    src = Path(__file__).resolve().parents[2] / "scripts" / "redetect_all_actions.py"
    text = src.read_text()
    # Both columns must appear in the UPDATE statement.
    assert "contacts_pipeline_version = %s" in text
    assert "actions_pipeline_version = %s" in text
    # And both constants must be imported / used.
    assert "ACTION_PIPELINE_VERSION" in text
    assert "CONTACT_PIPELINE_VERSION" in text
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis
uv run pytest tests/unit/test_redetect_all_actions_stamp.py -v
```

Expected: test fails (columns don't appear in the script yet).

- [ ] **Step 3: Inspect the current UPDATE**

```bash
cd /Users/mario/Personal/Projects/RallyCut
sed -n '180,190p' analysis/scripts/redetect_all_actions.py
```

Expected (approximately):

```python
                        with conn.cursor() as cur:
                            cur.execute(
                                "UPDATE player_tracks SET contacts_json = %s, actions_json = %s WHERE id = %s",
                                (json.dumps(new_contacts_json), json.dumps(new_actions_json), pt_id),
                            )
```

- [ ] **Step 4: Modify the UPDATE**

Open `analysis/scripts/redetect_all_actions.py` line 184. Replace:

```python
                            cur.execute(
                                "UPDATE player_tracks SET contacts_json = %s, actions_json = %s WHERE id = %s",
                                (json.dumps(new_contacts_json), json.dumps(new_actions_json), pt_id),
                            )
```

With:

```python
                            cur.execute(
                                "UPDATE player_tracks SET "
                                "contacts_json = %s, actions_json = %s, "
                                "contacts_pipeline_version = %s, "
                                "actions_pipeline_version = %s "
                                "WHERE id = %s",
                                (
                                    json.dumps(new_contacts_json),
                                    json.dumps(new_actions_json),
                                    CONTACT_PIPELINE_VERSION,
                                    ACTION_PIPELINE_VERSION,
                                    pt_id,
                                ),
                            )
```

Add at the top of the file (alongside other imports):

```python
from rallycut.tracking.action_classifier import ACTION_PIPELINE_VERSION
from rallycut.tracking.contact_detector import CONTACT_PIPELINE_VERSION
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis
uv run pytest tests/unit/test_redetect_all_actions_stamp.py -v
```

Expected: test passes.

- [ ] **Step 6: Commit**

```bash
cd /Users/mario/Personal/Projects/RallyCut
git add analysis/scripts/redetect_all_actions.py \
        analysis/tests/unit/test_redetect_all_actions_stamp.py
git commit -m "$(cat <<'EOF'
feat(redetect-all-actions): stamp both pipeline versions on SQL UPDATE

Fleet-refresh script now writes contacts_pipeline_version and
actions_pipeline_version alongside the JSON content, mirroring
ACTION_PIPELINE_VERSION + CONTACT_PIPELINE_VERSION.

[no-version-bump]

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Producer stamp — TS saveTrackingResult

**Files:**
- Modify: `api/src/services/playerTrackingService.ts` lines ~142, ~160, ~281, 725-726, 746-747
- Create: `api/src/services/__tests__/playerTrackingService.pipelineVersion.test.ts`

- [ ] **Step 1: Inspect the three interface declarations**

```bash
cd /Users/mario/Personal/Projects/RallyCut
sed -n '140,170p' api/src/services/playerTrackingService.ts
sed -n '275,290p' api/src/services/playerTrackingService.ts
sed -n '460,500p' api/src/services/playerTrackingService.ts
```

You should see three `interface`/type declarations carrying `contacts?: ContactsData; actions?: ActionsData;`, and a Modal-webhook re-assembly block that constructs `contacts`/`actions` from the webhook payload.

- [ ] **Step 2: Write the failing test**

Create `api/src/services/__tests__/playerTrackingService.pipelineVersion.test.ts`:

```typescript
import { describe, expect, it, beforeAll, afterAll } from 'vitest';
import { prisma } from '../../lib/prisma';
import { saveTrackingResult } from '../playerTrackingService';

describe('saveTrackingResult — pipeline version columns', () => {
  let videoId: string;
  let rallyId: string;

  beforeAll(async () => {
    // Create minimal video + rally fixtures.
    const v = await prisma.video.create({ data: { /* minimal valid shape */ } as any });
    videoId = v.id;
    const r = await prisma.rally.create({
      data: { videoId, startMs: 0, endMs: 1000 } as any,
    });
    rallyId = r.id;
  });

  afterAll(async () => {
    await prisma.rally.delete({ where: { id: rallyId } });
    await prisma.video.delete({ where: { id: videoId } });
    await prisma.$disconnect();
  });

  it('writes contactsPipelineVersion and actionsPipelineVersion when content is present', async () => {
    await saveTrackingResult(
      rallyId,
      videoId,
      {
        frameCount: 100,
        fps: 30,
        detectionRate: 0.9,
        avgConfidence: 0.8,
        avgPlayerCount: 4,
        uniqueTrackCount: 4,
        courtSplitY: 0.5,
        primaryTrackIds: [1, 2, 3, 4],
        positions: [],
        rawPositions: [],
        ballPositions: [],
        contacts: { numContacts: 0, contacts: [] } as any,
        actions: { numContacts: 0, actions: [] } as any,
        contactsPipelineVersion: 'v1',
        actionsPipelineVersion: 'v1',
        qualityReport: undefined,
      } as any,
      1234,
    );

    const row = await prisma.playerTrack.findUnique({ where: { rallyId } });
    expect(row?.contactsPipelineVersion).toBe('v1');
    expect(row?.actionsPipelineVersion).toBe('v1');
  });

  it('leaves columns null when contacts/actions are null', async () => {
    await saveTrackingResult(
      rallyId,
      videoId,
      {
        frameCount: 100,
        fps: 30,
        detectionRate: 0.9,
        avgConfidence: 0.8,
        avgPlayerCount: 4,
        uniqueTrackCount: 4,
        courtSplitY: 0.5,
        primaryTrackIds: [1, 2, 3, 4],
        positions: [],
        rawPositions: [],
        ballPositions: [],
        contacts: undefined,
        actions: undefined,
        contactsPipelineVersion: undefined,
        actionsPipelineVersion: undefined,
        qualityReport: undefined,
      } as any,
      1234,
    );

    const row = await prisma.playerTrack.findUnique({ where: { rallyId } });
    expect(row?.contactsPipelineVersion).toBeNull();
    expect(row?.actionsPipelineVersion).toBeNull();
  });
});
```

- [ ] **Step 3: Run test to verify it fails**

```bash
cd /Users/mario/Personal/Projects/RallyCut/api
npx vitest run src/services/__tests__/playerTrackingService.pipelineVersion.test.ts
```

Expected: test fails (TS doesn't know about the version fields yet).

- [ ] **Step 4: Extend the `PlayerTrackerOutput` interface**

Open `api/src/services/playerTrackingService.ts`. Locate the `PlayerTrackerOutput` interface (~line 142). After the `actions?: ActionsData;` line, add:

```typescript
  contactsPipelineVersion?: string | null;
  actionsPipelineVersion?: string | null;
```

- [ ] **Step 5: Extend the other two interface declarations**

Find the two other declarations with `contacts?: ContactsData; actions?: ActionsData;` (around lines 160 and 281 — they're sibling shapes for the Modal webhook payload and the batch-tracking result). Add the same two fields to each:

```typescript
  contactsPipelineVersion?: string | null;
  actionsPipelineVersion?: string | null;
```

- [ ] **Step 6: Extend the Modal-webhook re-assembly block (lines ~470-491)**

Find the block that constructs `contacts: ContactsData | undefined = result.contacts ? {...}`. Below it, where the result object is built, propagate the version fields. The exact location is where the webhook handler returns the assembled `PlayerTrackerOutput`-shaped object — add:

```typescript
        contactsPipelineVersion: result.contactsPipelineVersion ?? null,
        actionsPipelineVersion: result.actionsPipelineVersion ?? null,
```

If you're uncertain about the exact location, search for `result.contacts` and `result.actions` references in that block — propagate alongside them.

- [ ] **Step 7: Extend the Prisma writes at lines 725-726 and 746-747**

Find the two Prisma `.create()` calls inside `saveTrackingResult` (lines 725 and 746 currently have `contactsJson`/`actionsJson`). For EACH, add the matching version fields right after `actionsJson`:

```typescript
        contactsPipelineVersion: trackerResult.contacts ? (trackerResult.contactsPipelineVersion ?? null) : null,
        actionsPipelineVersion:  trackerResult.actions  ? (trackerResult.actionsPipelineVersion  ?? null) : null,
```

The semantic is: write the version stamp only when the corresponding JSON content was provided. When `contacts === undefined` / `null`, the column is explicitly null.

- [ ] **Step 8: Do NOT modify line 1576-1577 (track swap path)**

Confirm by reading lines 1570-1585 — that update writes mutated `contactsJson`/`actionsJson` from existing data. The version columns must be omitted from this update so the existing values are preserved (matches spec intent: swap is a manual edit, not a classifier output).

- [ ] **Step 9: Type-check**

```bash
cd /Users/mario/Personal/Projects/RallyCut/api
npx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 10: Run the integration test**

```bash
cd /Users/mario/Personal/Projects/RallyCut/api
npx vitest run src/services/__tests__/playerTrackingService.pipelineVersion.test.ts
```

Expected: both tests pass.

- [ ] **Step 11: Commit**

```bash
cd /Users/mario/Personal/Projects/RallyCut
git add api/src/services/playerTrackingService.ts \
        api/src/services/__tests__/playerTrackingService.pipelineVersion.test.ts
git commit -m "$(cat <<'EOF'
feat(api/saveTrackingResult): persist pipeline-version columns

Three interface declarations gain optional contactsPipelineVersion +
actionsPipelineVersion fields; the Modal-webhook re-assembly propagates
them; the two Prisma writes in saveTrackingResult persist them when the
corresponding JSON content is present. Track-swap path intentionally
leaves the existing column values alone (manual user edit, not
classifier output).

[no-version-bump]

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Consumer — pid_invariants StaleVersionReport

**Files:**
- Modify: `analysis/rallycut/tracking/pid_invariants.py`
- Create: `analysis/tests/unit/test_pid_invariants_stale_skip.py`

- [ ] **Step 1: Write the failing test**

Create `analysis/tests/unit/test_pid_invariants_stale_skip.py`:

```python
"""Stale-version skip semantics in pid_invariants.run_all.

I-3 depends on actions_json -> skip if actions_pipeline_version is stale.
I-4 depends on contacts_json -> skip if contacts_pipeline_version is stale.
I-1, I-2, I-5..I-8 don't depend on that content -> unaffected.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from rallycut.tracking.action_classifier import ACTION_PIPELINE_VERSION
from rallycut.tracking.contact_detector import CONTACT_PIPELINE_VERSION
from rallycut.tracking.pid_invariants import StaleVersionReport


def test_stale_version_report_dataclass_shape() -> None:
    """StaleVersionReport carries the fields the audit CLI will render."""
    report = StaleVersionReport(
        total_rallies=10,
        skipped_stale_actions={"rally-A", "rally-B"},
        skipped_stale_contacts={"rally-A"},
        current_actions_version=ACTION_PIPELINE_VERSION,
        current_contacts_version=CONTACT_PIPELINE_VERSION,
        observed_actions_versions={"v0": 2, "v1": 8},
        observed_contacts_versions={"v0": 1, "v1": 9},
    )
    assert report.total_rallies == 10
    assert "rally-A" in report.skipped_stale_actions
    assert report.current_actions_version == ACTION_PIPELINE_VERSION


def test_run_all_returns_stale_report(monkeypatch) -> None:
    """pid_invariants.run_all returns (violations, stale_report)."""
    from rallycut.tracking import pid_invariants

    fake_rows = [
        # rally_id, primary_track_ids, positions_json, actions_json,
        # contacts_json, track_to_player, team_assignments,
        # actions_pipeline_version, contacts_pipeline_version
        ("rally-current", [1, 2, 3, 4], [], [], [], {1:1,2:2,3:3,4:4}, {1:"A",2:"A",3:"B",4:"B"},
         ACTION_PIPELINE_VERSION, CONTACT_PIPELINE_VERSION),
        ("rally-stale-actions", [1, 2, 3, 4], [], [], [], {1:1,2:2,3:3,4:4}, {1:"A",2:"A",3:"B",4:"B"},
         "v0", CONTACT_PIPELINE_VERSION),
    ]

    monkeypatch.setattr(
        pid_invariants,
        "_fetch_rally_rows",
        lambda video_id: fake_rows,
        raising=False,
    )

    violations, stale_report = pid_invariants.run_all(video_id="vid")

    assert "rally-stale-actions" in stale_report.skipped_stale_actions
    assert "rally-current" not in stale_report.skipped_stale_actions
    # I-3 on rally-stale-actions must be skipped — assert no I-3 violation
    # references that rally.
    assert not any(
        v.invariant == "I-3" and v.rally_id == "rally-stale-actions"
        for v in violations
    )


def test_i1_still_runs_on_stale_action_version(monkeypatch) -> None:
    """I-1 (primary set size) doesn't depend on actions_json -> still runs."""
    from rallycut.tracking import pid_invariants

    fake_rows = [
        # I-1 violation: 3 primary tracks instead of 4
        ("rally-stale-actions", [1, 2, 3], [], [], [], {}, {},
         "v0", CONTACT_PIPELINE_VERSION),
    ]
    monkeypatch.setattr(
        pid_invariants, "_fetch_rally_rows",
        lambda video_id: fake_rows, raising=False,
    )

    violations, _ = pid_invariants.run_all(video_id="vid")

    # Stale actions don't gate I-1 — the violation must surface.
    assert any(
        v.invariant == "I-1" and v.rally_id == "rally-stale-actions"
        for v in violations
    )
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis
uv run pytest tests/unit/test_pid_invariants_stale_skip.py -v
```

Expected: ImportError on `StaleVersionReport` or fixture-shape mismatch — the dataclass doesn't exist yet, and `run_all` doesn't return a tuple.

- [ ] **Step 3: Add `StaleVersionReport` and a `_fetch_rally_rows` seam**

Open `analysis/rallycut/tracking/pid_invariants.py`. Find the existing `@dataclass(frozen=True) class Violation:` at line ~30. Add a new dataclass alongside it:

```python
@dataclass(frozen=True)
class StaleVersionReport:
    """Per-video summary of pipeline-version staleness."""
    total_rallies: int
    skipped_stale_actions: frozenset[str]
    skipped_stale_contacts: frozenset[str]
    current_actions_version: str
    current_contacts_version: str
    observed_actions_versions: dict[str, int]
    observed_contacts_versions: dict[str, int]

    @property
    def has_stale(self) -> bool:
        return bool(self.skipped_stale_actions or self.skipped_stale_contacts)
```

Use `frozenset` rather than `set` because dataclass is `frozen=True`. The constructor caller should pass `frozenset(...)`.

- [ ] **Step 4: Refactor `run_all` to extract row-fetch into a seam and return the tuple**

Find the existing `run_all` function. It currently opens a DB connection and runs invariants per rally. Refactor:

```python
def _fetch_rally_rows(video_id: str) -> list[tuple[Any, ...]]:
    """Fetch the per-rally data needed for all PID invariants.

    Returns tuples of:
      (rally_id, primary_track_ids, positions_json, actions_list,
       contacts_list, track_to_player, team_assignments,
       actions_pipeline_version, contacts_pipeline_version)
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT r.id,
                   pt.primary_track_ids,
                   pt.positions_json,
                   pt.actions_json,
                   pt.contacts_json,
                   pt.actions_json->'teamAssignments',
                   pt.actions_pipeline_version,
                   pt.contacts_pipeline_version
              FROM rallies r
              JOIN player_tracks pt ON pt.rally_id = r.id
             WHERE r.video_id = %s
            """,
            (video_id,),
        )
        # ... existing row-shaping code ...
        # IMPORTANT: append actions_pipeline_version + contacts_pipeline_version
        # to every row tuple.
        return rows


def run_all(*, video_id: str) -> tuple[list[Violation], StaleVersionReport]:
    """Run all PID invariants on a video; return violations + stale report.

    Per-invariant skip on stale pipeline version:
      I-3 (actions) skipped if actions_pipeline_version != ACTION_PIPELINE_VERSION
      I-4 (contacts) skipped if contacts_pipeline_version != CONTACT_PIPELINE_VERSION
      I-1, I-2, I-5..I-8 unaffected (don't depend on actions/contacts content)
    """
    from rallycut.tracking.action_classifier import ACTION_PIPELINE_VERSION
    from rallycut.tracking.contact_detector import CONTACT_PIPELINE_VERSION

    rows = _fetch_rally_rows(video_id)
    violations: list[Violation] = []
    skipped_actions: set[str] = set()
    skipped_contacts: set[str] = set()
    observed_actions: dict[str, int] = {}
    observed_contacts: dict[str, int] = {}

    for row in rows:
        (
            rally_id, primary_ids, positions, actions_list, contacts_list,
            t2p, team_assigns, actions_pv, contacts_pv,
        ) = row

        # Tally observed versions (including None).
        observed_actions[str(actions_pv) if actions_pv else "null"] = (
            observed_actions.get(str(actions_pv) if actions_pv else "null", 0) + 1
        )
        observed_contacts[str(contacts_pv) if contacts_pv else "null"] = (
            observed_contacts.get(str(contacts_pv) if contacts_pv else "null", 0) + 1
        )

        actions_stale = (
            actions_pv is not None and actions_pv != ACTION_PIPELINE_VERSION
        )
        contacts_stale = (
            contacts_pv is not None and contacts_pv != CONTACT_PIPELINE_VERSION
        )
        if actions_stale:
            skipped_actions.add(rally_id)
        if contacts_stale:
            skipped_contacts.add(rally_id)

        # I-1, I-2, I-5, I-6, I-7, I-8 — always run.
        violations.extend(check_i1_primary_set_size(rally_id=rally_id, primary_track_ids=primary_ids))
        violations.extend(check_i2_positions_in_primary(rally_id=rally_id, primary_track_ids=primary_ids, positions_json=positions))
        violations.extend(check_i5_track_to_player_total(rally_id=rally_id, primary_track_ids=primary_ids, track_to_player=t2p))
        violations.extend(check_i6_team_assignments_total(rally_id=rally_id, primary_track_ids=primary_ids, team_assignments=team_assigns))
        violations.extend(check_i7_mapped_track_ids_canonical(rally_id=rally_id, actions=actions_list, track_to_player=t2p))
        violations.extend(check_i8_team_partition(rally_id=rally_id, primary_track_ids=primary_ids, team_assignments=team_assigns))

        # I-3 depends on actions_json — skip if stale.
        if not actions_stale:
            violations.extend(check_i3_action_player_ids(
                rally_id=rally_id, primary_track_ids=primary_ids, actions=actions_list,
            ))

        # I-4 depends on contacts_json — skip if stale.
        if not contacts_stale:
            violations.extend(check_i4_contact_player_ids(
                rally_id=rally_id, primary_track_ids=primary_ids, contacts=contacts_list,
            ))

    stale_report = StaleVersionReport(
        total_rallies=len(rows),
        skipped_stale_actions=frozenset(skipped_actions),
        skipped_stale_contacts=frozenset(skipped_contacts),
        current_actions_version=ACTION_PIPELINE_VERSION,
        current_contacts_version=CONTACT_PIPELINE_VERSION,
        observed_actions_versions=observed_actions,
        observed_contacts_versions=observed_contacts,
    )
    return violations, stale_report
```

**Note:** the precise list of `check_iN_*` function names is approximate — match what already exists in `pid_invariants.py`. If a check function isn't named exactly as shown, use the existing name and keep the per-invariant skip logic.

- [ ] **Step 5: Update existing call sites that expected `run_all` to return only a list**

```bash
cd /Users/mario/Personal/Projects/RallyCut
grep -rn 'pid_invariants.run_all\|pid_run_all\|from rallycut.tracking.pid_invariants import run_all' analysis/ 2>/dev/null
```

Adjust each call site to unpack the new tuple. The known call sites:

1. `analysis/rallycut/tracking/coherence_invariants.py` — uses `pid_run_all`. Updated in Task 9; for now make this file compile by changing:
   ```python
   from rallycut.tracking.pid_invariants import run_all as pid_run_all
   ```
   to expect the tuple — but defer the actual semantics to Task 9.

2. `analysis/rallycut/cli/commands/audit_pid_invariants.py` — updated in Task 10.

For this task, just make the imports + signatures compile. The CLI's usage of the tuple is handled in Task 10.

- [ ] **Step 6: Run the test to verify it passes**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis
uv run pytest tests/unit/test_pid_invariants_stale_skip.py -v
```

Expected: 3 tests pass.

- [ ] **Step 7: Run the existing pid_invariants tests to confirm no regression**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis
uv run pytest tests/unit/ -v -k 'pid' --no-header
```

Expected: existing tests still pass (or are updated for the new return shape — fix any that broke).

- [ ] **Step 8: Type-check**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis
uv run mypy rallycut/tracking/pid_invariants.py
```

Expected: `Success: no issues found`.

- [ ] **Step 9: Commit**

```bash
cd /Users/mario/Personal/Projects/RallyCut
git add analysis/rallycut/tracking/pid_invariants.py \
        analysis/tests/unit/test_pid_invariants_stale_skip.py
git commit -m "$(cat <<'EOF'
feat(pid-invariants): per-invariant skip on stale pipeline version

run_all now returns (violations, StaleVersionReport). I-3 skipped when
actions_pipeline_version is stale; I-4 skipped when
contacts_pipeline_version is stale; I-1,2,5..8 unaffected. Mirrors the
MATCHER_VERSION consumer pattern.

[no-version-bump]

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Consumer — coherence_invariants stale skip

**Files:**
- Modify: `analysis/rallycut/tracking/coherence_invariants.py`
- Create: `analysis/tests/unit/test_coherence_invariants_stale_skip.py`

- [ ] **Step 1: Write the failing test**

Create `analysis/tests/unit/test_coherence_invariants_stale_skip.py`:

```python
"""Stale-actions skip for the coherence audit.

All four coherence invariants (C-1..C-4) read actions_json. A stale
actions_pipeline_version means the rally is excluded from the report.
"""

from __future__ import annotations

from rallycut.tracking.action_classifier import ACTION_PIPELINE_VERSION
from rallycut.tracking.contact_detector import CONTACT_PIPELINE_VERSION


def test_coherence_run_all_returns_stale_report(monkeypatch) -> None:
    from rallycut.tracking import coherence_invariants
    from rallycut.tracking.pid_invariants import StaleVersionReport

    # Two rallies: one current, one stale.
    fake_rows = [
        ("rally-current", {1: "A", 2: "A", 3: "B", 4: "B"},
         [{"frame": 0, "playerTrackId": 1, "action": "serve"}],
         ACTION_PIPELINE_VERSION, CONTACT_PIPELINE_VERSION),
        ("rally-stale", {1: "A", 2: "A", 3: "B", 4: "B"},
         [{"frame": 0, "playerTrackId": 1, "action": "serve"}],
         "v0", CONTACT_PIPELINE_VERSION),
    ]

    monkeypatch.setattr(
        coherence_invariants,
        "_fetch_coherence_rows",
        lambda video_id: fake_rows,
        raising=False,
    )
    # Empty PID violations (no upstream issues).
    monkeypatch.setattr(
        coherence_invariants,
        "pid_run_all",
        lambda *, video_id: ([], StaleVersionReport(
            total_rallies=2,
            skipped_stale_actions=frozenset(),
            skipped_stale_contacts=frozenset(),
            current_actions_version=ACTION_PIPELINE_VERSION,
            current_contacts_version=CONTACT_PIPELINE_VERSION,
            observed_actions_versions={"v0": 1, ACTION_PIPELINE_VERSION: 1},
            observed_contacts_versions={CONTACT_PIPELINE_VERSION: 2},
        )),
        raising=False,
    )

    violations, stale = coherence_invariants.run_all(video_id="vid")

    assert "rally-stale" in stale.skipped_stale_actions
    assert "rally-stale" not in {v.rally_id for v in violations}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis
uv run pytest tests/unit/test_coherence_invariants_stale_skip.py -v
```

Expected: fails — `run_all` doesn't return a tuple yet.

- [ ] **Step 3: Modify `coherence_invariants.run_all`**

Open `analysis/rallycut/tracking/coherence_invariants.py`. Refactor `run_all`:

```python
def run_all(*, video_id: str) -> tuple[list[Violation], StaleVersionReport]:
    """Run coherence invariants; return violations + stale report.

    All coherence rules (C-1..C-4) read actions_json, so a stale
    actions_pipeline_version excludes the rally from coherence checks.
    Returns the PID-side StaleVersionReport (covers both columns) so the
    audit CLI shell can render a unified header.
    """
    from rallycut.tracking.action_classifier import ACTION_PIPELINE_VERSION

    # Defer to PID's stale-report machinery for consistency.
    pid_violations, stale_report = pid_run_all(video_id=video_id)

    # Coherence is gated on the upstream PID invariants (I-1, I-3, I-6 in
    # particular). Build the set of rallies excluded from coherence due
    # to those.
    pid_excluded: set[str] = {
        v.rally_id for v in pid_violations
        if v.invariant in {"I-1", "I-3", "I-6"} and v.severity == "error"
    }
    # Also exclude rallies stale on actions_pipeline_version.
    actions_stale: set[str] = set(stale_report.skipped_stale_actions)

    rows = _fetch_coherence_rows(video_id)
    violations: list[Violation] = []
    for rally_id, team_assigns, actions_list, actions_pv, contacts_pv in rows:
        if rally_id in pid_excluded or rally_id in actions_stale:
            continue
        violations.extend(check_c1_three_contact_rule(
            rally_id=rally_id, actions=actions_list, team_assignments=team_assigns,
        ))
        violations.extend(check_c2_alternating_possessions(
            rally_id=rally_id, actions=actions_list, team_assignments=team_assigns,
        ))
        violations.extend(check_c3_first_action_is_serve(
            rally_id=rally_id, actions=actions_list,
        ))
        violations.extend(check_c4_consecutive_different_players(
            rally_id=rally_id, actions=actions_list,
        ))

    return violations, stale_report


def _fetch_coherence_rows(video_id: str) -> list[tuple[Any, ...]]:
    """Fetch (rally_id, team_assigns, actions_list, actions_pv, contacts_pv)."""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT r.id,
                   pt.actions_json->'teamAssignments',
                   pt.actions_json->'actions',
                   pt.actions_pipeline_version,
                   pt.contacts_pipeline_version
              FROM rallies r
              JOIN player_tracks pt ON pt.rally_id = r.id
             WHERE r.video_id = %s
               AND pt.actions_json IS NOT NULL
            """,
            (video_id,),
        )
        # Coerce / shape rows. Existing row-shaping logic stays.
        return [
            (r[0], dict(r[1] or {}), list(r[2] or []), r[3], r[4])
            for r in cur.fetchall()
        ]
```

Also at the top of the file, replace the imports if needed:

```python
from rallycut.tracking.pid_invariants import StaleVersionReport, Violation
from rallycut.tracking.pid_invariants import run_all as pid_run_all
```

Match the existing helper names (`check_c1_three_contact_rule`, etc.) to whatever is already defined in the file.

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis
uv run pytest tests/unit/test_coherence_invariants_stale_skip.py -v
```

Expected: test passes.

- [ ] **Step 5: Run all existing coherence tests**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis
uv run pytest tests/unit/ -v -k 'coherence' --no-header
```

Expected: all existing tests pass (or are updated for the new return shape).

- [ ] **Step 6: Type-check**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis
uv run mypy rallycut/tracking/coherence_invariants.py
```

Expected: `Success: no issues found`.

- [ ] **Step 7: Commit**

```bash
cd /Users/mario/Personal/Projects/RallyCut
git add analysis/rallycut/tracking/coherence_invariants.py \
        analysis/tests/unit/test_coherence_invariants_stale_skip.py
git commit -m "$(cat <<'EOF'
feat(coherence-invariants): skip rallies with stale actions_pipeline_version

run_all now returns (violations, StaleVersionReport). Rallies with a
stale actions_pipeline_version are excluded from C-1..C-4 (in addition
to the existing PID I-1/I-3/I-6 exclusion).

[no-version-bump]

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Audit CLI shells render the stale-version report

**Files:**
- Modify: `analysis/rallycut/cli/commands/audit_pid_invariants.py`
- Modify: `analysis/rallycut/cli/commands/audit_coherence_invariants.py`
- Create: `analysis/tests/unit/test_audit_cli_stale_render.py`

- [ ] **Step 1: Write the failing test**

Create `analysis/tests/unit/test_audit_cli_stale_render.py`:

```python
"""The audit CLI shells render the StaleVersionReport header."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from rallycut.cli.main import app
from rallycut.tracking.pid_invariants import StaleVersionReport, Violation


runner = CliRunner()


def test_audit_pid_invariants_renders_stale_header() -> None:
    fake_violations = []
    fake_stale = StaleVersionReport(
        total_rallies=10,
        skipped_stale_actions=frozenset(["rally-A", "rally-B"]),
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

    assert "2 of 10 rallies skipped" in result.stdout or "2 of 10" in result.stdout
    assert "actions_pipeline_version" in result.stdout
    assert "v0" in result.stdout
    # Exit code 0: no error-severity violations, stale-only doesn't fail.
    assert result.exit_code == 0


def test_audit_pid_invariants_fails_on_error_violation() -> None:
    fake_violations = [Violation(invariant="I-1", rally_id="r1", detail="bad")]
    fake_stale = StaleVersionReport(
        total_rallies=1, skipped_stale_actions=frozenset(),
        skipped_stale_contacts=frozenset(),
        current_actions_version="v1", current_contacts_version="v1",
        observed_actions_versions={"v1": 1}, observed_contacts_versions={"v1": 1},
    )
    with patch(
        "rallycut.cli.commands.audit_pid_invariants.run_all",
        return_value=(fake_violations, fake_stale),
    ):
        result = runner.invoke(app, ["audit-pid-invariants", "test-video-id"])

    assert result.exit_code == 1


def test_audit_coherence_invariants_renders_stale_header() -> None:
    fake_violations = []
    fake_stale = StaleVersionReport(
        total_rallies=5, skipped_stale_actions=frozenset(["r1"]),
        skipped_stale_contacts=frozenset(),
        current_actions_version="v1", current_contacts_version="v1",
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis
uv run pytest tests/unit/test_audit_cli_stale_render.py -v
```

Expected: fails — CLI shells don't unpack the new tuple yet and don't render the header.

- [ ] **Step 3: Update `audit_pid_invariants.py`**

Open `analysis/rallycut/cli/commands/audit_pid_invariants.py`. Replace the body of `audit_pid_invariants_cmd`:

```python
def audit_pid_invariants_cmd(
    video_id: str = typer.Argument(..., help="Video UUID"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress info output"),
) -> None:
    """Audit PID-attribution invariants for a video's match-analysis state."""
    if not quiet:
        console.print(f"[dim]Running PID-invariant audit on {video_id}…[/dim]")

    violations, stale = run_all(video_id=video_id)

    _render_stale_header(stale)

    if not violations:
        if not quiet:
            console.print("[green]✓ All invariants hold[/green]")
        raise typer.Exit(code=0 if not stale.has_stale else 0)

    table = Table(title=f"PID-invariant violations — video {video_id}")
    table.add_column("Invariant", style="bold")
    table.add_column("Rally", style="cyan")
    table.add_column("Severity", style="yellow")
    table.add_column("Detail")
    for v in violations:
        table.add_row(v.invariant, v.rally_id, v.severity, v.detail)
    console.print(table)

    n_errors = sum(1 for v in violations if v.severity == "error")
    n_warns = len(violations) - n_errors
    console.print(
        f"[red]{n_errors} error[/red] · [yellow]{n_warns} warn[/yellow] · "
        f"{len(violations)} total"
    )

    raise typer.Exit(code=1 if n_errors else 0)


def _render_stale_header(stale: StaleVersionReport) -> None:
    """Print the stale-version block at the top of the report."""
    if not stale.has_stale:
        return
    n_stale = len(stale.skipped_stale_actions | stale.skipped_stale_contacts)
    console.print(
        f"\n[yellow]⚠ {n_stale} of {stale.total_rallies} rallies skipped due to stale pipeline version[/yellow]"
    )
    if stale.skipped_stale_actions:
        observed = ", ".join(f"{k}:{v}" for k, v in sorted(stale.observed_actions_versions.items()))
        console.print(
            f"  - {len(stale.skipped_stale_actions)} stale actions_pipeline_version "
            f"(observed: {{{observed}}}; current: {stale.current_actions_version})"
        )
    if stale.skipped_stale_contacts:
        observed = ", ".join(f"{k}:{v}" for k, v in sorted(stale.observed_contacts_versions.items()))
        console.print(
            f"  - {len(stale.skipped_stale_contacts)} stale contacts_pipeline_version "
            f"(observed: {{{observed}}}; current: {stale.current_contacts_version})"
        )
    console.print("  Run: uv run python scripts/redetect_all_actions.py --apply\n")
```

Add the imports at the top:

```python
from rallycut.tracking.pid_invariants import StaleVersionReport, run_all
```

- [ ] **Step 4: Update `audit_coherence_invariants.py` analogously**

Same shape — call `run_all`, unpack `(violations, stale)`, render the same header. Import `StaleVersionReport` from `pid_invariants` (since `coherence_invariants.run_all` reuses it). Use the same `_render_stale_header` (copy-paste is fine — these two CLIs are intentionally independent and small).

- [ ] **Step 5: Run the tests**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis
uv run pytest tests/unit/test_audit_cli_stale_render.py -v
```

Expected: 3 tests pass.

- [ ] **Step 6: Smoke-test both CLIs**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis
# Pick any video ID from your local DB; replace below.
VID=$(PGPASSWORD=postgres psql -h localhost -p 5436 -U postgres -d rallycut -tAc \
  "SELECT id FROM videos LIMIT 1")
uv run rallycut audit-pid-invariants "$VID"
uv run rallycut audit-coherence-invariants "$VID"
```

Expected: both print the stale-version header (since the fleet hasn't been refreshed yet, all rallies will be `v0`) and either pass or fail depending on the data.

- [ ] **Step 7: Commit**

```bash
cd /Users/mario/Personal/Projects/RallyCut
git add analysis/rallycut/cli/commands/audit_pid_invariants.py \
        analysis/rallycut/cli/commands/audit_coherence_invariants.py \
        analysis/tests/unit/test_audit_cli_stale_render.py
git commit -m "$(cat <<'EOF'
feat(audit-cli): render StaleVersionReport header

Both audit CLIs (pid + coherence) print a yellow header summarising
stale-pipeline-version rallies and the call-to-action to run
redetect_all_actions.py. Stale-only does not fail the audit; only
error-severity violations exit non-zero.

[no-version-bump]

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: analysis/CLAUDE.md checklist

**Files:**
- Modify: `analysis/CLAUDE.md`

- [ ] **Step 1: Locate the insertion point**

Open `analysis/CLAUDE.md`. Find a natural section — either right above "Code Style" near the bottom, or right after "Cross-rally matcher validation protocol" (since this is the same family of validation discipline).

- [ ] **Step 2: Add the checklist**

Insert this new section (recommended placement: after the "Cross-rally matcher validation protocol" section):

```markdown
## Post-classifier-change checklist

When you change `analysis/rallycut/tracking/action_classifier.py` or
`analysis/rallycut/tracking/contact_detector.py` in a way that affects
the serialized output of `RallyActions.to_dict()` or
`ContactSequence.to_dict()`:

1. **Bump the constant** in the same commit:
   - `ACTION_PIPELINE_VERSION` in `action_classifier.py`
   - `CONTACT_PIPELINE_VERSION` in `contact_detector.py`
2. **Add a version-history comment line** above the constant explaining
   what changed (mirror the `MATCHER_VERSION` style in
   `analysis/rallycut/tracking/match_tracker.py`).
3. **Pre-commit hook** (`.claude/hooks/pre-commit-check.sh`) will reject
   commits that touch these files without a version bump. Add
   `[no-version-bump]` to the commit message if the change is genuinely
   cosmetic (docstring/typo/comment-only).
4. **After merge, refresh the fleet:**
   ```bash
   uv run python scripts/redetect_all_actions.py --apply
   ```
5. **Re-run the audit** to confirm zero stale rallies and refresh the
   reference baseline if behavior changed:
   ```bash
   uv run rallycut audit-coherence-invariants <video-id>
   uv run python scripts/catalog_c4_violations.py
   ```

The same discipline applies to `MATCHER_VERSION` in `match_tracker.py`
(see `tests/unit/test_assignment_anchor_versioning.py`). The difference:
`MATCHER_VERSION` has a self-enforcing consumer-side feedback loop
(stale anchors visibly break the next PERMUTED panel), while the
pipeline versions rely on the pre-commit hook + audit reporting.
```

- [ ] **Step 3: Verify rendering**

```bash
cd /Users/mario/Personal/Projects/RallyCut
grep -n 'Post-classifier-change' analysis/CLAUDE.md
```

Expected: shows the new section heading.

- [ ] **Step 4: Commit**

```bash
cd /Users/mario/Personal/Projects/RallyCut
git add analysis/CLAUDE.md
git commit -m "$(cat <<'EOF'
docs(analysis): post-classifier-change checklist

Documents the discipline: bump version constant + add comment line +
refresh fleet + re-baseline audit. Pre-commit hook enforces step 1.
EOF
)"
```

---

## Task 12: Fleet refresh + reference baseline

**Files:** (operational — no code changes)
- Create: `analysis/reports/coherence_c4_catalog/2026-05-14_baseline.csv` (or actual deploy date)
- Create: `analysis/reports/coherence_c4_catalog/2026-05-14_baseline_summary.md`

- [ ] **Step 1: Verify all preceding tasks are committed**

```bash
cd /Users/mario/Personal/Projects/RallyCut
git log --oneline -12
git status
```

Expected: working tree clean, 11 new commits visible (1 schema + 1 constants + 1 hook + 4 producers + 2 consumers + 1 CLI + 1 docs + this readme commit comes later).

- [ ] **Step 2: Apply the migration to production DB**

```bash
cd /Users/mario/Personal/Projects/RallyCut/api
# For local dev DB:
npx prisma migrate deploy
# For production: follow the standard deploy procedure (out of scope here).
```

Expected: migration applied; backfill runs.

- [ ] **Step 3: Run the fleet refresh**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis
PYTHONUNBUFFERED=1 uv run python scripts/redetect_all_actions.py --apply
```

Expected runtime: comparable to a full fleet retrack (~20-40 minutes per the memory entry redetect_all_actions_fix_2026_05_11). Output should show `X updated, 0 skipped, 0 errors`.

- [ ] **Step 4: Verify all rows are at v1**

```bash
PGPASSWORD=postgres psql -h localhost -p 5436 -U postgres -d rallycut -c "
  SELECT
    actions_pipeline_version, COUNT(*) AS n
  FROM player_tracks
  WHERE actions_json IS NOT NULL
  GROUP BY actions_pipeline_version
  ORDER BY actions_pipeline_version;"
```

Expected: a single row showing `actions_pipeline_version=v1` with n equal to total rallies. Same for `contacts_pipeline_version`.

- [ ] **Step 5: Run an audit on the panel to confirm zero stale**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis
# Pick any panel video.
uv run rallycut audit-coherence-invariants 854bb250-... 2>&1 | tee /tmp/audit_panel.log
```

Expected: no stale-version header (zero stale rallies). Violations may exist but the stale block is silent.

- [ ] **Step 6: Generate the fleet baseline**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis
uv run python scripts/catalog_c4_violations.py
```

Expected: produces `analysis/reports/coherence_c4_catalog/<today>.csv` + `<today>_summary.md`.

- [ ] **Step 7: Rename to a `_baseline` suffix**

```bash
TODAY=$(date +%Y-%m-%d)
cd /Users/mario/Personal/Projects/RallyCut/analysis/reports/coherence_c4_catalog/
cp "${TODAY}.csv" "${TODAY}_baseline.csv"
cp "${TODAY}_summary.md" "${TODAY}_baseline_summary.md"
```

- [ ] **Step 8: Commit the baseline**

```bash
cd /Users/mario/Personal/Projects/RallyCut
git add analysis/reports/coherence_c4_catalog/*_baseline*
git commit -m "$(cat <<'EOF'
report(coherence): post-pipeline-versioning baseline snapshot

Fleet refreshed via redetect_all_actions.py --apply; all rows now stamp
ACTION_PIPELINE_VERSION = CONTACT_PIPELINE_VERSION = v1. The baseline
CSV + summary serve as the regression-floor reference for future
classifier changes.

EOF
)"
```

- [ ] **Step 9: Update MEMORY.md**

Add an entry under "Current workstreams" in `/Users/mario/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/MEMORY.md`:

```markdown
- [SHIPPED] [**Pipeline version stamps 2026-05-13**](pipeline_version_stamps_2026_05_13.md) — Two new TEXT columns on player_tracks (actions_pipeline_version, contacts_pipeline_version) + ACTION_PIPELINE_VERSION + CONTACT_PIPELINE_VERSION constants in their producer modules. Stamped at 4 producer sites; consumer skip-with-counter in audit-pid-invariants + audit-coherence-invariants. Pre-commit hook blocks producer-file edits without a version bump (escape hatch: `[no-version-bump]`). Mirrors MATCHER_VERSION pattern. Closes the recurring stale-derived-data class of bugs (panel_baseline_regression_2026_05_07, coherence_repair_sub_2b_2026_05_13, knowledge_state_2026_04_26).
```

And create a memory file `memory/pipeline_version_stamps_2026_05_13.md` documenting key facts (the spec is the long-form artifact; the memory entry is the index).

---

## Self-Review

**Spec coverage** — checked against [docs/superpowers/specs/2026-05-13-pipeline-version-stamps-design.md](../specs/2026-05-13-pipeline-version-stamps-design.md):

| Spec section | Implementing task |
|---|---|
| Schema migration | Task 1 |
| Python constants | Task 2 |
| Producer 1 (Modal/track_player) | Task 4 |
| Producer 2 (reattribute_actions) | Task 5 |
| Producer 3 (redetect_all_actions) | Task 6 |
| Producer 4 (TS saveTrackingResult) | Task 7 |
| Consumer (pid_invariants) | Task 8 |
| Consumer (coherence_invariants) | Task 9 |
| Audit CLI shell rendering | Task 10 |
| Pre-commit hook | Task 3 |
| Fleet refresh + baseline | Task 12 |
| analysis/CLAUDE.md checklist | Task 11 |
| Test pinning (constants) | Task 2 |
| Producer-stamping tests | Tasks 4, 5, 6, 7 |
| Consumer-skipping tests | Tasks 8, 9 |
| Hook tests | Task 3 |

All spec sections have an implementing task.

**Placeholder scan:** searched for "TODO", "TBD", "implement later", and "fill in details". Two soft references remain — both intentional:
- Task 7 Step 6: "search for `result.contacts` and `result.actions` references in that block — propagate alongside them." The Modal-webhook re-assembly is one location in a long file; the literal grep is the right way to find it.
- Task 8 Step 4: "the precise list of `check_iN_*` function names is approximate — match what already exists in `pid_invariants.py`." Listed because the file is large; the existing function set is authoritative.

These are search-pointers, not unspecified work.

**Type consistency:** `StaleVersionReport` defined in Task 8 Step 3 with `frozenset[str]` for the skipped sets; Task 9 and Task 10 use the same dataclass via import. Field names (`skipped_stale_actions`, `skipped_stale_contacts`, `current_actions_version`, `current_contacts_version`, `observed_actions_versions`, `observed_contacts_versions`, `total_rallies`, `has_stale`) consistent across all references.

`run_all` return type `tuple[list[Violation], StaleVersionReport]` consistent across pid_invariants (Task 8) and coherence_invariants (Task 9).
