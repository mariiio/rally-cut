# Score Tracking GT Backup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Round-trip `rallies.gt_serving_team` and `rallies.gt_side_switch` through S3 by extending the existing `train export-dataset` / `push` / `pull` / `restore` pipeline with a new optional `score_ground_truth.json` artifact.

**Architecture:** Mirror the existing optional-GT pattern (tracking, action, player-matching). Add an export helper, write the JSON when non-empty, append to push/pull metadata files, and add a NULL-only restore that never clobbers newer DB labels.

**Tech Stack:** Python 3.11+, psycopg, boto3, Typer, pytest. Files touched: `analysis/rallycut/cli/commands/train.py`, `analysis/rallycut/training/backup.py`, `analysis/rallycut/training/restore.py`, plus a new `analysis/tests/unit/test_score_gt_backup.py`.

**Spec:** `docs/superpowers/specs/2026-04-14-score-tracking-gt-backup-design.md`

---

## File Structure

| File | Responsibility |
|------|----------------|
| `analysis/rallycut/cli/commands/train.py` | Add `_export_score_ground_truth`, write JSON in `export_dataset`, print restore summary line. |
| `analysis/rallycut/training/backup.py` | Append `score_ground_truth.json` to push metadata files; add optional download in pull. |
| `analysis/rallycut/training/restore.py` | Add `score_gt_restored` field on `RestoreResult`, `_restore_score_ground_truth` helper, wire into `restore_dataset_to_db` + dry-run preview. |
| `analysis/tests/unit/test_score_gt_backup.py` (new) | Unit tests for the export helper shape and the NULL-only restore semantics. |

---

## Conventions

- Run all commands from the repo root unless noted.
- Python lives under `analysis/`; tests run with `cd analysis && uv run pytest …`.
- Use `psycopg` cursor mocking (no live DB) for unit tests, matching the pattern in `analysis/tests/unit/test_match_stats.py`.
- Each task ends with a commit. Use Conventional Commits (`feat:`, `test:`, `refactor:`).

---

## Task 1: Add `_export_score_ground_truth` helper

**Files:**
- Modify: `analysis/rallycut/cli/commands/train.py` (insert after `_export_action_ground_truth`, line 1720)
- Test: `analysis/tests/unit/test_score_gt_backup.py` (new)

> **Scope update (2026-04-14):** After Task 1's first pass landed, scope
> expanded to also back up `rallies.gt_side_switch` (co-labeled with
> `gt_serving_team` via the same Score GT UI). See Task 1b for the
> extension.

- [ ] **Step 1: Create the failing test file**

Create `analysis/tests/unit/test_score_gt_backup.py`:

```python
"""Unit tests for score-tracking GT backup helpers."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def fake_conn() -> MagicMock:
    """Fake psycopg connection whose cursor returns scriptable rows."""
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cursor
    conn.cursor.return_value.__exit__.return_value = False
    return conn


def _set_rows(conn: MagicMock, rows: list[tuple]) -> MagicMock:
    cursor = conn.cursor.return_value.__enter__.return_value
    cursor.fetchall.return_value = rows
    return cursor


class TestExportScoreGroundTruth:
    def test_returns_dict_with_filtered_rallies(self, fake_conn: MagicMock) -> None:
        from rallycut.cli.commands.train import _export_score_ground_truth

        _set_rows(
            fake_conn,
            [
                ("hashA", "rally-1", "video-1", "A"),
                ("hashB", "rally-2", "video-2", "B"),
                ("hashC", "rally-3", "video-3", "A"),  # not in dataset
            ],
        )

        with patch(
            "rallycut.cli.commands.train.get_connection", return_value=fake_conn
        ):
            result = _export_score_ground_truth({"hashA", "hashB"})

        assert result is not None
        assert result["stats"] == {"total_rallies": 2, "total_videos": 2}
        rally_ids = {r["rally_id"] for r in result["rallies"]}
        assert rally_ids == {"rally-1", "rally-2"}
        for entry in result["rallies"]:
            assert set(entry.keys()) == {
                "rally_id",
                "video_id",
                "content_hash",
                "gt_serving_team",
            }
            assert entry["gt_serving_team"] in {"A", "B"}

    def test_returns_none_when_no_rows(self, fake_conn: MagicMock) -> None:
        from rallycut.cli.commands.train import _export_score_ground_truth

        _set_rows(fake_conn, [])
        with patch(
            "rallycut.cli.commands.train.get_connection", return_value=fake_conn
        ):
            assert _export_score_ground_truth({"hashA"}) is None

    def test_returns_none_when_all_filtered_out(
        self, fake_conn: MagicMock
    ) -> None:
        from rallycut.cli.commands.train import _export_score_ground_truth

        _set_rows(fake_conn, [("hashZ", "rally-9", "video-9", "A")])
        with patch(
            "rallycut.cli.commands.train.get_connection", return_value=fake_conn
        ):
            assert _export_score_ground_truth({"hashA"}) is None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd analysis && uv run pytest tests/unit/test_score_gt_backup.py::TestExportScoreGroundTruth -v`

Expected: 3 failures with `ImportError: cannot import name '_export_score_ground_truth'`.

- [ ] **Step 3: Implement the helper**

In `analysis/rallycut/cli/commands/train.py`, insert immediately after the closing `}` of `_export_action_ground_truth` (after line 1720, before `@app.command("export-dataset")`):

```python
def _export_score_ground_truth(
    video_content_hashes: set[str],
) -> dict[str, Any] | None:
    """Export score-tracking GT (rallies.gt_serving_team) from DB.

    Returns rallies whose video is in the current dataset and whose
    ``gt_serving_team`` is non-NULL. Returns ``None`` if no such rallies exist.
    """
    from rallycut.evaluation.db import get_connection

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    v.content_hash,
                    r.id,
                    v.id AS video_id,
                    r.gt_serving_team
                FROM rallies r
                JOIN videos v ON v.id = r.video_id
                WHERE r.gt_serving_team IS NOT NULL
                  AND v.deleted_at IS NULL
                ORDER BY v.content_hash, r.start_ms
                """
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        return None

    rallies: list[dict[str, Any]] = []
    video_ids_seen: set[str] = set()

    for row in rows:
        content_hash = str(row[0])
        if content_hash not in video_content_hashes:
            continue
        rallies.append({
            "rally_id": str(row[1]),
            "video_id": str(row[2]),
            "content_hash": content_hash,
            "gt_serving_team": str(row[3]),
        })
        video_ids_seen.add(str(row[2]))

    if not rallies:
        return None

    return {
        "stats": {
            "total_rallies": len(rallies),
            "total_videos": len(video_ids_seen),
        },
        "rallies": rallies,
    }
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd analysis && uv run pytest tests/unit/test_score_gt_backup.py::TestExportScoreGroundTruth -v`

Expected: 3 passed.

- [ ] **Step 5: Type-check and lint**

Run: `cd analysis && uv run ruff check rallycut/cli/commands/train.py tests/unit/test_score_gt_backup.py && uv run mypy rallycut/cli/commands/train.py`

Expected: clean (no new errors introduced).

- [ ] **Step 6: Commit**

```bash
git add analysis/rallycut/cli/commands/train.py analysis/tests/unit/test_score_gt_backup.py
git commit -m "feat: add _export_score_ground_truth helper"
```

---

## Task 1b: Extend helper to include `gt_side_switch`

**Files:**
- Modify: `analysis/rallycut/cli/commands/train.py` (the `_export_score_ground_truth` helper added in Task 1)
- Modify: `analysis/tests/unit/test_score_gt_backup.py` (the `TestExportScoreGroundTruth` class)

- [ ] **Step 1: Update the failing tests first**

Replace the body of `TestExportScoreGroundTruth` in `analysis/tests/unit/test_score_gt_backup.py` with:

```python
class TestExportScoreGroundTruth:
    def test_returns_entries_with_either_field(
        self, fake_conn: MagicMock
    ) -> None:
        from rallycut.cli.commands.train import _export_score_ground_truth

        # Rows shape: (content_hash, rally_id, video_id, gt_serving_team, gt_side_switch)
        _set_rows(
            fake_conn,
            [
                ("hashA", "rally-serving-only", "video-1", "A", None),
                ("hashA", "rally-switch-only", "video-1", None, True),
                ("hashB", "rally-both", "video-2", "B", False),
                ("hashC", "rally-other", "video-3", "A", True),  # not in dataset
            ],
        )

        with patch(
            "rallycut.evaluation.db.get_connection", return_value=fake_conn
        ):
            result = _export_score_ground_truth({"hashA", "hashB"})

        assert result is not None
        assert result["stats"] == {
            "total_rallies": 3,
            "total_videos": 2,
            "total_with_serving": 2,
            "total_with_side_switch": 2,
        }

        by_id = {r["rally_id"]: r for r in result["rallies"]}
        assert set(by_id) == {
            "rally-serving-only",
            "rally-switch-only",
            "rally-both",
        }

        serving_only = by_id["rally-serving-only"]
        assert serving_only["gt_serving_team"] == "A"
        assert "gt_side_switch" not in serving_only  # NULL in DB → omitted

        switch_only = by_id["rally-switch-only"]
        assert switch_only["gt_side_switch"] is True
        assert "gt_serving_team" not in switch_only  # NULL in DB → omitted

        both = by_id["rally-both"]
        assert both["gt_serving_team"] == "B"
        assert both["gt_side_switch"] is False

    def test_returns_none_when_no_rows(self, fake_conn: MagicMock) -> None:
        from rallycut.cli.commands.train import _export_score_ground_truth

        _set_rows(fake_conn, [])
        with patch(
            "rallycut.evaluation.db.get_connection", return_value=fake_conn
        ):
            assert _export_score_ground_truth({"hashA"}) is None

    def test_returns_none_when_all_filtered_out(
        self, fake_conn: MagicMock
    ) -> None:
        from rallycut.cli.commands.train import _export_score_ground_truth

        _set_rows(
            fake_conn, [("hashZ", "rally-9", "video-9", "A", None)]
        )
        with patch(
            "rallycut.evaluation.db.get_connection", return_value=fake_conn
        ):
            assert _export_score_ground_truth({"hashA"}) is None
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `cd /Users/mario/Personal/Projects/RallyCut/analysis && uv run pytest tests/unit/test_score_gt_backup.py::TestExportScoreGroundTruth -v`

Expected: the first test fails because the current helper's SQL does not return `gt_side_switch` and does not omit NULL fields per-entry.

- [ ] **Step 3: Rewrite the helper**

Replace the body of `_export_score_ground_truth` in `analysis/rallycut/cli/commands/train.py` with:

```python
def _export_score_ground_truth(
    video_content_hashes: set[str],
) -> dict[str, Any] | None:
    """Export score-tracking GT from DB.

    Returns rallies whose video is in the current dataset and for which at
    least one of ``gt_serving_team`` / ``gt_side_switch`` is non-NULL.
    Per-entry, each column is omitted when its DB value is NULL so
    downstream restore never sees an explicit ``null``.

    Returns ``None`` if no qualifying rallies exist.
    """
    from rallycut.evaluation.db import get_connection

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    v.content_hash,
                    r.id,
                    v.id AS video_id,
                    r.gt_serving_team,
                    r.gt_side_switch
                FROM rallies r
                JOIN videos v ON v.id = r.video_id
                WHERE v.deleted_at IS NULL
                  AND (
                      r.gt_serving_team IS NOT NULL
                   OR r.gt_side_switch IS NOT NULL
                  )
                ORDER BY v.content_hash, r.start_ms
                """
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        return None

    rallies: list[dict[str, Any]] = []
    video_ids_seen: set[str] = set()
    total_with_serving = 0
    total_with_side_switch = 0

    for row in rows:
        content_hash = str(row[0])
        if content_hash not in video_content_hashes:
            continue

        entry: dict[str, Any] = {
            "rally_id": str(row[1]),
            "video_id": str(row[2]),
            "content_hash": content_hash,
        }
        serving = row[3]
        side_switch = row[4]
        if serving is not None:
            entry["gt_serving_team"] = str(serving)
            total_with_serving += 1
        if side_switch is not None:
            entry["gt_side_switch"] = bool(side_switch)
            total_with_side_switch += 1

        rallies.append(entry)
        video_ids_seen.add(str(row[2]))

    if not rallies:
        return None

    return {
        "stats": {
            "total_rallies": len(rallies),
            "total_videos": len(video_ids_seen),
            "total_with_serving": total_with_serving,
            "total_with_side_switch": total_with_side_switch,
        },
        "rallies": rallies,
    }
```

- [ ] **Step 4: Run tests to confirm they pass**

Run: `cd /Users/mario/Personal/Projects/RallyCut/analysis && uv run pytest tests/unit/test_score_gt_backup.py::TestExportScoreGroundTruth -v`

Expected: 3 passed.

- [ ] **Step 5: Lint and type-check**

Run: `cd /Users/mario/Personal/Projects/RallyCut/analysis && uv run ruff check rallycut/cli/commands/train.py tests/unit/test_score_gt_backup.py && uv run mypy rallycut/cli/commands/train.py`

Expected: clean.

- [ ] **Step 6: Commit**

```bash
cd /Users/mario/Personal/Projects/RallyCut
git add analysis/rallycut/cli/commands/train.py analysis/tests/unit/test_score_gt_backup.py
git commit -m "feat: include gt_side_switch in score_ground_truth export"
```

---

## Task 2: Wire score GT into `export_dataset`

**Files:**
- Modify: `analysis/rallycut/cli/commands/train.py:1905-1916` (after the action-GT block in `export_dataset`)

- [ ] **Step 1: Add the export block**

In `export_dataset()`, immediately after the existing action-GT block ending at line 1916 (the `rprint(f"  Action GT: …")` line), insert:

```python
    # Export score-tracking ground truth (gt_serving_team + gt_side_switch)
    score_gt = _export_score_ground_truth(content_hashes)
    if score_gt:
        score_gt_path = dataset_dir / "score_ground_truth.json"
        with open(score_gt_path, "w") as f:
            json.dump(score_gt, f, indent=2)
        sgt_stats = score_gt["stats"]
        rprint(f"Created [cyan]{score_gt_path}[/cyan]")
        rprint(
            f"  Score GT: [green]{sgt_stats['total_rallies']}[/green] rallies"
            f" across {sgt_stats['total_videos']} videos"
            f" ([cyan]{sgt_stats['total_with_serving']}[/cyan] serving,"
            f" [cyan]{sgt_stats['total_with_side_switch']}[/cyan] side-switch)"
        )
```

- [ ] **Step 2: Smoke-check by importing**

Run: `cd analysis && uv run python -c "from rallycut.cli.commands.train import export_dataset, _export_score_ground_truth; print('ok')"`

Expected: `ok`.

- [ ] **Step 3: Type-check and lint**

Run: `cd analysis && uv run ruff check rallycut/cli/commands/train.py && uv run mypy rallycut/cli/commands/train.py`

Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add analysis/rallycut/cli/commands/train.py
git commit -m "feat: write score_ground_truth.json in export-dataset"
```

---

## Task 3: Push and pull `score_ground_truth.json`

**Files:**
- Modify: `analysis/rallycut/training/backup.py` (push_dataset metadata list ~line 248; pull_dataset try/except blocks ~line 384)

- [ ] **Step 1: Add to push metadata files**

Locate the existing block in `push_dataset` (around line 248):

```python
        player_matching_gt_path = dataset_dir / "player_matching_ground_truth.json"
        if player_matching_gt_path.exists():
            metadata_files.append(
                (player_matching_gt_path, "player_matching_ground_truth.json")
            )
```

Append immediately after it:

```python
        score_gt_path = dataset_dir / "score_ground_truth.json"
        if score_gt_path.exists():
            metadata_files.append((score_gt_path, "score_ground_truth.json"))
```

- [ ] **Step 2: Add optional download to pull**

Locate the existing pull block at line ~384 ending the player_matching try/except:

```python
        # Download player matching ground truth (optional)
        try:
            player_matching_gt_path = dataset_dir / "player_matching_ground_truth.json"
            self.s3.download_file(
                self.bucket,
                self._dataset_key(name, "player_matching_ground_truth.json"),
                str(player_matching_gt_path),
            )
        except ClientError:
            pass  # Not present in older datasets
```

Append immediately after it:

```python
        # Download score ground truth (optional — older datasets may not have it)
        try:
            score_gt_path = dataset_dir / "score_ground_truth.json"
            self.s3.download_file(
                self.bucket,
                self._dataset_key(name, "score_ground_truth.json"),
                str(score_gt_path),
            )
        except ClientError:
            pass  # Not present in older datasets
```

- [ ] **Step 3: Update module docstring**

In `analysis/rallycut/training/backup.py`, edit the docstring (lines 9-23). Find:

```
        action_ground_truth.json             (optional)
        player_matching_ground_truth.json    (optional)
```

Replace with:

```
        action_ground_truth.json             (optional)
        player_matching_ground_truth.json    (optional)
        score_ground_truth.json              (optional)
```

- [ ] **Step 4: Type-check and lint**

Run: `cd analysis && uv run ruff check rallycut/training/backup.py && uv run mypy rallycut/training/backup.py`

Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add analysis/rallycut/training/backup.py
git commit -m "feat: push and pull score_ground_truth.json"
```

---

## Task 4: NULL-only restore helper

**Files:**
- Modify: `analysis/rallycut/training/restore.py` (add field on `RestoreResult` line 33-37; new `_restore_score_ground_truth` after `_restore_player_matching_gt` at line 645)
- Test: `analysis/tests/unit/test_score_gt_backup.py`

- [ ] **Step 1: Add the failing test**

Append to `analysis/tests/unit/test_score_gt_backup.py`:

```python
class TestRestoreScoreGroundTruth:
    def _write_payload(self, tmp_path: Path) -> Path:
        payload = {
            "stats": {
                "total_rallies": 3,
                "total_videos": 1,
                "total_with_serving": 2,
                "total_with_side_switch": 2,
            },
            "rallies": [
                {
                    "rally_id": "rally-serving-writable",
                    "video_id": "vid-1",
                    "content_hash": "hash1",
                    "gt_serving_team": "A",
                    "gt_side_switch": True,
                },
                {
                    "rally_id": "rally-switch-only",
                    "video_id": "vid-1",
                    "content_hash": "hash1",
                    "gt_side_switch": False,
                },
                {
                    "rally_id": "rally-missing",
                    "video_id": "vid-1",
                    "content_hash": "hash1",
                    "gt_serving_team": "B",
                },
            ],
        }
        p = tmp_path / "score_ground_truth.json"
        p.write_text(json.dumps(payload))
        return p

    def test_updates_only_null_rows_and_records_misses(
        self, tmp_path: Path
    ) -> None:
        from rallycut.training.restore import (
            RestoreResult,
            _restore_score_ground_truth,
        )

        path = self._write_payload(tmp_path)

        # The helper runs, per rally:
        #   1) SELECT id FROM rallies WHERE id = %s   (existence check)
        #   2) UPDATE gt_serving_team (only if the field is in the JSON)
        #   3) UPDATE gt_side_switch  (only if the field is in the JSON)
        # Rally 1 has both fields, rally 2 has only side_switch, rally 3 is
        # missing from DB (existence check fails → no UPDATEs).
        #
        # For rally 1: serving UPDATE rowcount=1, side_switch UPDATE rowcount=0
        #              (already set) → count as restored (serving wrote).
        # For rally 2: side_switch UPDATE rowcount=1 → count as restored.
        # For rally 3: existence SELECT returns None → error, no UPDATE.
        existence_results = iter(
            [
                ("rally-serving-writable",),  # rally 1 exists
                ("rally-switch-only",),       # rally 2 exists
                None,                          # rally 3 missing
            ]
        )
        update_rowcounts = iter([1, 0, 1])  # rally1.serving, rally1.switch, rally2.switch

        cursor = MagicMock()

        def execute_side_effect(sql: str, _params: Any = None) -> None:
            sql_lower = sql.lower()
            if "select id from rallies" in sql_lower:
                cursor._last = "select"
            elif "update rallies" in sql_lower:
                cursor._last = "update"
                cursor.rowcount = next(update_rowcounts)
            else:
                cursor._last = "other"

        cursor.execute.side_effect = execute_side_effect
        cursor.fetchone.side_effect = lambda: next(existence_results)

        conn = MagicMock()
        conn.cursor.return_value.__enter__.return_value = cursor
        conn.cursor.return_value.__exit__.return_value = False

        result = RestoreResult()
        _restore_score_ground_truth(conn, path, result)

        # Rally 1 (wrote serving) and rally 2 (wrote side_switch) count.
        assert result.score_gt_restored == 2
        assert any("rally-missing" in e for e in result.errors)

    def test_silent_skip_when_file_missing(self, tmp_path: Path) -> None:
        from rallycut.training.restore import (
            RestoreResult,
            _restore_score_ground_truth,
        )

        result = RestoreResult()
        _restore_score_ground_truth(MagicMock(), tmp_path / "missing.json", result)

        assert result.score_gt_restored == 0
        assert result.errors == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd analysis && uv run pytest tests/unit/test_score_gt_backup.py::TestRestoreScoreGroundTruth -v`

Expected: failures with `ImportError: cannot import name '_restore_score_ground_truth'`.

- [ ] **Step 3: Add the field on `RestoreResult`**

In `analysis/rallycut/training/restore.py`, edit the `RestoreResult` dataclass (lines 26-37). Find:

```python
    player_matching_gt_restored: int = 0
    session_created: str = ""
```

Replace with:

```python
    player_matching_gt_restored: int = 0
    score_gt_restored: int = 0
    session_created: str = ""
```

- [ ] **Step 4: Add the helper**

Append to `analysis/rallycut/training/restore.py` immediately after the closing of `_restore_player_matching_gt` (after line 645 where `result.player_matching_gt_restored = restored` ends its function body):

```python
def _restore_score_ground_truth(
    conn: psycopg.Connection[tuple[Any, ...]],
    score_gt_path: Path,
    result: RestoreResult,
) -> None:
    """Restore rallies.gt_serving_team and rallies.gt_side_switch.

    Per-field NULL-only: each column is updated independently, and only
    when the target column is currently NULL. Existing labels are
    preserved so a re-run never clobbers DB edits made after the snapshot
    was taken. A field absent from a JSON entry means it was NULL at
    export time — its UPDATE is skipped entirely.

    A rally counts as restored if at least one of its fields was actually
    written.
    """
    if not score_gt_path.exists():
        return

    with open(score_gt_path) as f:
        payload = json.load(f)

    entries = payload.get("rallies", [])
    if not entries:
        return

    restored = 0
    for entry in entries:
        rally_id = entry["rally_id"]

        with conn.cursor() as cur:
            cur.execute("SELECT id FROM rallies WHERE id = %s", (rally_id,))
            row = cur.fetchone()
            if row is None:
                result.errors.append(
                    f"Score GT: no matching rally id {rally_id}"
                )
                continue

            wrote_any = False
            if "gt_serving_team" in entry:
                cur.execute(
                    """
                    UPDATE rallies
                       SET gt_serving_team = %s
                     WHERE id = %s
                       AND gt_serving_team IS NULL
                    """,
                    (entry["gt_serving_team"], rally_id),
                )
                if cur.rowcount > 0:
                    wrote_any = True

            if "gt_side_switch" in entry:
                cur.execute(
                    """
                    UPDATE rallies
                       SET gt_side_switch = %s
                     WHERE id = %s
                       AND gt_side_switch IS NULL
                    """,
                    (bool(entry["gt_side_switch"]), rally_id),
                )
                if cur.rowcount > 0:
                    wrote_any = True

            if wrote_any:
                restored += 1

    result.score_gt_restored = restored
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `cd analysis && uv run pytest tests/unit/test_score_gt_backup.py -v`

Expected: all 5 tests pass.

- [ ] **Step 6: Type-check and lint**

Run: `cd analysis && uv run ruff check rallycut/training/restore.py tests/unit/test_score_gt_backup.py && uv run mypy rallycut/training/restore.py`

Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add analysis/rallycut/training/restore.py analysis/tests/unit/test_score_gt_backup.py
git commit -m "feat: NULL-only restore for score_ground_truth.json"
```

---

## Task 5: Wire restore + dry-run preview + CLI summary

**Files:**
- Modify: `analysis/rallycut/training/restore.py` (`restore_dataset_to_db` ~line 322; `_preview_restore` ~line 404)
- Modify: `analysis/rallycut/cli/commands/train.py` (`restore` command summary ~line 2143)

- [ ] **Step 1: Wire into `restore_dataset_to_db`**

In `analysis/rallycut/training/restore.py`, find (lines 322-325):

```python
            # Restore player matching ground truth if available
            pm_gt_path = dataset_dir / "player_matching_ground_truth.json"
            if pm_gt_path.exists():
                _restore_player_matching_gt(conn, pm_gt_path, result)
```

Append immediately after:

```python
            # Restore score-tracking ground truth (NULL-only) if available
            score_gt_path = dataset_dir / "score_ground_truth.json"
            if score_gt_path.exists():
                _restore_score_ground_truth(conn, score_gt_path, result)
```

- [ ] **Step 2: Add to dry-run preview**

In `_preview_restore` find (after line 413 where the player matching preview ends):

```python
        rprint(
            f"  Would restore: [green]{pm_stats.get('total_rallies', 0)}[/green]"
            f" player matching GT across {pm_stats.get('total_videos', 0)} videos"
        )
```

Append:

```python
    # Show score GT info if available
    score_gt_path = dataset_dir / "score_ground_truth.json"
    if score_gt_path.exists():
        with open(score_gt_path) as f:
            score_gt = json.load(f)
        sgt_stats = score_gt.get("stats", {})
        rprint(
            f"  Would restore: [green]{sgt_stats.get('total_rallies', 0)}[/green]"
            f" score GT (NULL-only) across {sgt_stats.get('total_videos', 0)} videos"
            f" ({sgt_stats.get('total_with_serving', 0)} serving,"
            f" {sgt_stats.get('total_with_side_switch', 0)} side-switch)"
        )
```

- [ ] **Step 3: Update module docstring**

In `analysis/rallycut/training/restore.py`, edit the module docstring (lines 7-8). Find:

```
Expected files: manifest.json, ground_truth.json, tracking_ground_truth.json (optional),
action_ground_truth.json (optional).
```

Replace with:

```
Expected files: manifest.json, ground_truth.json, tracking_ground_truth.json (optional),
action_ground_truth.json (optional), player_matching_ground_truth.json (optional),
score_ground_truth.json (optional).
```

- [ ] **Step 4: Add the CLI summary line**

In `analysis/rallycut/cli/commands/train.py`, find (lines 2140-2144 in the `restore` command):

```python
    if result.tracking_gt_restored:
        rprint(f"  Tracking GT restored: [green]{result.tracking_gt_restored}[/green]")
    if result.action_gt_restored:
        rprint(f"  Action GT restored: [green]{result.action_gt_restored}[/green]")
    rprint(f"  Session: [cyan]{result.session_created}[/cyan]")
```

Replace with:

```python
    if result.tracking_gt_restored:
        rprint(f"  Tracking GT restored: [green]{result.tracking_gt_restored}[/green]")
    if result.action_gt_restored:
        rprint(f"  Action GT restored: [green]{result.action_gt_restored}[/green]")
    if result.score_gt_restored:
        rprint(f"  Score GT restored: [green]{result.score_gt_restored}[/green]")
    rprint(f"  Session: [cyan]{result.session_created}[/cyan]")
```

- [ ] **Step 5: Smoke-import**

Run: `cd analysis && uv run python -c "from rallycut.training.restore import restore_dataset_to_db, _restore_score_ground_truth; from rallycut.cli.commands.train import restore; print('ok')"`

Expected: `ok`.

- [ ] **Step 6: Run all unit tests for the new module**

Run: `cd analysis && uv run pytest tests/unit/test_score_gt_backup.py -v`

Expected: 5 passed.

- [ ] **Step 7: Type-check and lint everything touched**

Run: `cd analysis && uv run ruff check rallycut/cli/commands/train.py rallycut/training/restore.py rallycut/training/backup.py && uv run mypy rallycut/cli/commands/train.py rallycut/training/restore.py rallycut/training/backup.py`

Expected: clean (no new errors).

- [ ] **Step 8: Commit**

```bash
git add analysis/rallycut/training/restore.py analysis/rallycut/cli/commands/train.py
git commit -m "feat: restore score GT and surface in CLI summary"
```

---

## Task 6: End-to-end live verification

**Files:** none modified — verification only.

This task confirms the full round-trip works against the real database and S3. Skip with explicit user approval if S3 creds are unavailable in this environment.

- [ ] **Step 1: Confirm the column exists and has data**

Run: `cd analysis && uv run python -c "from rallycut.evaluation.db import get_connection; conn = get_connection(); cur = conn.cursor(); cur.execute('SELECT COUNT(*) FROM rallies WHERE gt_serving_team IS NOT NULL'); print('non-null gt_serving_team rows:', cur.fetchone()[0])"`

Expected: a non-zero count (per memory, ~316 GT rallies are locked).

- [ ] **Step 2: Export a fresh dataset and confirm score GT file is written**

Run: `cd analysis && uv run rallycut train export-dataset --name score_gt_backup_test`

Expected output includes a line like `Created training_datasets/score_gt_backup_test/score_ground_truth.json` and `Score GT: <N> rallies across <M> videos`.

Verify the file: `ls -la analysis/training_datasets/score_gt_backup_test/score_ground_truth.json`

- [ ] **Step 3: Inspect the JSON shape**

Run: `cd analysis && uv run python -c "import json; d = json.load(open('training_datasets/score_gt_backup_test/score_ground_truth.json')); print('stats:', d['stats']); print('first:', d['rallies'][0] if d['rallies'] else None)"`

Expected: stats dict with `total_rallies` and `total_videos`; first entry has the four fields from the spec (`rally_id`, `video_id`, `content_hash`, `gt_serving_team`).

- [ ] **Step 4: Push to S3**

Run: `cd analysis && uv run rallycut train push --name score_gt_backup_test`

Expected: completes without errors. The command uploads `score_ground_truth.json` along with the other metadata files.

- [ ] **Step 5: Pull into a clean directory and confirm the file came back**

Run: `cd analysis && rm -rf /tmp/rallycut_pull_test && uv run rallycut train pull --name score_gt_backup_test --output /tmp/rallycut_pull_test`

(If `pull` does not accept `--output`, omit it and the default location is fine — then check that location.)

Verify: `ls -la /tmp/rallycut_pull_test/score_gt_backup_test/score_ground_truth.json` (or the corresponding default-location path).

Expected: the file exists and matches the pushed version.

- [ ] **Step 6: Restore dry-run shows score GT line**

Run: `cd analysis && uv run rallycut train restore --name score_gt_backup_test --dry-run`

Expected: output includes a `Would restore: <N> score GT (NULL-only) across <M> videos` line.

- [ ] **Step 7: Clean up the test dataset**

Decide with the user whether to delete `score_gt_backup_test` from S3. Do **not** auto-delete — confirm first.

If approved: `cd analysis && uv run aws s3 rm s3://$TRAINING_S3_BUCKET/$TRAINING_S3_PREFIX/datasets/score_gt_backup_test/ --recursive` (use the bucket/prefix from `rallycut.yaml` or env).

- [ ] **Step 8: Final commit if any docs touched**

If any incidental doc updates are needed (e.g., the `analysis/CLAUDE.md` table of dataset files), commit them here. Otherwise skip.

---

## Self-Review Notes

- **Spec coverage:** Goals 1–4, format/components/error-handling/testing/out-of-scope all map to Tasks 1–5. Task 6 confirms the live round-trip per spec's data-flow diagram.
- **Placeholders:** None — every step has either complete code, a complete command, or an explicit user-decision step (Task 6 cleanup).
- **Type consistency:** `_export_score_ground_truth` returns a dict whose `rallies[*]` keys (`rally_id`, `video_id`, `content_hash`, `gt_serving_team`) exactly match what `_restore_score_ground_truth` consumes. `RestoreResult.score_gt_restored: int` is referenced consistently across the dataclass field, the helper assignment, and the CLI summary.
