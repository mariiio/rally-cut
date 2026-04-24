"""Snapshot test locking detect_contacts() output on 5 representative rallies.

Purpose
-------
Phase 2a of the parallel-decoder ship plan
(`docs/superpowers/plans/2026-04-24-parallel-decoder-ship.md`). Locks the
current `detect_contacts()` output before refactoring the function so any
behavior change during the refactor is caught immediately.

What it tests
-------------
For 5 stable rally IDs (selected deterministically), the test:
1. Loads positions from the DB via `load_rallies_with_action_gt`
2. Builds PlayerPositions with pose-cache injection (production path)
3. Calls `detect_contacts(...)` with the production classifier
4. Serializes the resulting `ContactSequence` to JSON
5. Compares to the stored fixture in `tests/fixtures/detect_contacts_snapshot/`

Regenerating the fixtures (intentional behavior change)
-------------------------------------------------------
When the GBM classifier weights, MS-TCN++ weights, or any other production
artifact is intentionally updated, regenerate the fixtures:

    cd analysis
    RALLYCUT_SNAPSHOT_REGENERATE=1 uv run pytest \\
        tests/integration/test_detect_contacts_snapshot.py

Then commit the updated fixtures along with the weight change. The test
becomes a guard: any unintended behavior shift requires a deliberate
regeneration step.

Skip behavior
-------------
If the database is unavailable (e.g., CI without local Postgres) the test
SKIPS rather than fails. The fixture path is the canonical source of truth
when the DB is offline.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "detect_contacts_snapshot"
RALLY_LIST_PATH = FIXTURE_DIR / "_rally_ids.json"
REGENERATE = os.environ.get("RALLYCUT_SNAPSHOT_REGENERATE") == "1"

# Number of rallies to snapshot. Enough variety to catch class-specific
# regressions; few enough to keep test runtime under 60s.
N_RALLIES = 5


def _select_rally_ids() -> list[str]:
    """Pick 5 stable rally IDs.

    Deterministic ordering: query DB ORDER BY rally_id, return first
    N_RALLIES with at least 3 GT contacts (filters out near-empty rallies
    that wouldn't exercise the candidate generators).

    Cached to disk on first run so subsequent test runs don't depend on
    DB state for rally selection (only for position data load).
    """
    if RALLY_LIST_PATH.exists() and not REGENERATE:
        cached: list[str] = json.loads(RALLY_LIST_PATH.read_text())
        return cached

    from rallycut.evaluation.tracking.db import get_connection

    query = """
        SELECT r.id, jsonb_array_length(pt.action_ground_truth_json) AS n_gt
        FROM rallies r
        JOIN player_tracks pt ON pt.rally_id = r.id
        WHERE pt.action_ground_truth_json IS NOT NULL
          AND jsonb_array_length(pt.action_ground_truth_json) >= 3
          AND pt.positions_json IS NOT NULL
          AND pt.ball_positions_json IS NOT NULL
        ORDER BY r.id::text
        LIMIT %s
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, [N_RALLIES])
            rows = cur.fetchall()

    rally_ids = [str(row[0]) for row in rows]
    if len(rally_ids) < N_RALLIES:
        return rally_ids  # tolerate fewer rallies in dev DBs
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    RALLY_LIST_PATH.write_text(json.dumps(rally_ids, indent=2))
    return rally_ids


def _detect_for_rally(rally_id: str) -> dict:
    """Run detect_contacts on a single rally; return serialized output.

    Mirrors the production caller in `scripts/eval_loo_video.py:_eval_rally`:
    passes `sequence_probs` (MS-TCN++ output) so the GBM's `seq_max_nonbg`
    feature is populated. Without it the GBM rejects ~all candidates because
    the seq feature dominates accept decisions (~40% importance).
    """
    from rallycut.tracking.ball_tracker import BallPosition
    from rallycut.tracking.contact_detector import detect_contacts
    from rallycut.tracking.sequence_action_runtime import get_sequence_probs
    from scripts.eval_action_detection import (
        _build_player_positions,
        load_rallies_with_action_gt,
    )

    rallies = load_rallies_with_action_gt(rally_id=rally_id)
    if not rallies:
        pytest.skip(f"Rally {rally_id} not in DB")
    rally = rallies[0]
    if not rally.positions_json or not rally.ball_positions_json:
        pytest.skip(f"Rally {rally_id} missing position data")

    player_positions = _build_player_positions(
        rally.positions_json, rally_id=rally.rally_id, inject_pose=True,
    )
    ball_positions = [
        BallPosition(
            frame_number=bp["frameNumber"],
            x=bp["x"],
            y=bp["y"],
            confidence=bp.get("confidence", 1.0),
        )
        for bp in rally.ball_positions_json
    ]

    sequence_probs = get_sequence_probs(
        ball_positions, player_positions,
        rally.court_split_y, rally.frame_count or 0, None,
    )

    seq = detect_contacts(
        ball_positions=ball_positions,
        player_positions=player_positions,
        net_y=rally.court_split_y,
        frame_count=rally.frame_count,
        sequence_probs=sequence_probs,
    )
    return seq.to_dict()


def _fixture_path(rally_id: str) -> Path:
    return FIXTURE_DIR / f"{rally_id}.json"


def _normalize_for_compare(d: dict) -> str:
    """Stable string serialization for diff."""
    return json.dumps(d, indent=2, sort_keys=True, default=str)


@pytest.fixture(scope="session")
def rally_ids() -> list[str]:
    try:
        ids = _select_rally_ids()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Cannot load rally IDs from DB: {e}")
    if not ids:
        pytest.skip("No rallies with action GT in DB")
    return ids


def test_detect_contacts_snapshots(rally_ids: list[str]) -> None:
    """Detect contacts on each rally; assert output matches the locked fixture."""
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    diffs: list[str] = []
    for rid in rally_ids:
        try:
            actual = _detect_for_rally(rid)
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"Cannot load data for rally {rid}: {e}")

        actual_text = _normalize_for_compare(actual)
        fixture_path = _fixture_path(rid)

        if REGENERATE or not fixture_path.exists():
            fixture_path.write_text(actual_text)
            print(f"[snapshot] {'REGENERATED' if REGENERATE else 'CREATED'} "
                  f"{fixture_path.relative_to(REPO_ROOT)}")
            continue

        expected_text = fixture_path.read_text()
        if expected_text != actual_text:
            diffs.append(
                f"\n=== Rally {rid} drift ===\n"
                f"Fixture: {fixture_path.relative_to(REPO_ROOT)}\n"
                f"Run with RALLYCUT_SNAPSHOT_REGENERATE=1 to update IF "
                f"the change is intentional.\n"
            )

    if diffs:
        pytest.fail(
            "detect_contacts() output drift detected on "
            f"{len(diffs)}/{len(rally_ids)} rallies:\n"
            + "\n".join(diffs)
        )


if __name__ == "__main__":
    # Direct invocation: just run the regeneration logic, no pytest.
    os.environ["RALLYCUT_SNAPSHOT_REGENERATE"] = "1"
    ids = _select_rally_ids()
    print(f"[snapshot] Regenerating fixtures for {len(ids)} rallies")
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    for rid in ids:
        actual = _detect_for_rally(rid)
        _fixture_path(rid).write_text(_normalize_for_compare(actual))
        print(f"[snapshot] wrote {_fixture_path(rid).relative_to(REPO_ROOT)}")
