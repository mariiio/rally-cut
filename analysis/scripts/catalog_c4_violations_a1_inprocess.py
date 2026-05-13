"""In-process C-4 catalog with A1 volleyball-rule re-attribution.

Mirrors `scripts/catalog_c4_violations.py` but, before running the C-4
check on each rally's actions, re-runs `reattribute_players` in process
with `USE_VOLLEYBALL_RULE_ATTRIBUTION` enabled. This lets us measure
the post-A1 C-4 violation count WITHOUT writing A1 attributions to the
database (so we don't pre-commit the ship decision).

Only the C-4 invariant is recomputed (C-1/C-2/C-3 are not relevant to
the A1 evaluation). Output CSV has the same columns as the production
catalog so the existing comparison harness works.

Usage:
    cd analysis
    USE_VOLLEYBALL_RULE_ATTRIBUTION=1 uv run python \
        scripts/catalog_c4_violations_a1_inprocess.py \
        --output reports/coherence_c4_catalog/2026-05-13_a1_on.csv \
        --summary reports/coherence_c4_catalog/2026-05-13_a1_on_summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from rallycut.evaluation.tracking.db import get_connection
from rallycut.tracking.action_classifier import (
    ClassifiedAction,
    reattribute_players,
)
from rallycut.tracking.coherence_invariants import (
    check_c4_no_same_player_back_to_back,
)
from rallycut.tracking.contact_detector import Contact
from rallycut.tracking.pid_invariants import run_all as pid_run_all

# Reuse the row-builder + summary helpers from the production catalog.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from catalog_c4_violations import (  # noqa: E402
    CSV_COLUMNS,
    _contact_by_frame,  # noqa: F401  (kept for parity with prod)
    _row_from_violation,
    _summarize,
)

_UPSTREAM_BLOCKER_INVARIANTS = {"I-1", "I-3", "I-6"}


def _reconstruct_contacts(contacts_data: list[dict[str, Any]]) -> list[Contact]:
    out: list[Contact] = []
    for c in contacts_data:
        candidates_raw = c.get("player_candidates") or c.get("playerCandidates") or []
        candidates: list[tuple[int, float]] = []
        for entry in candidates_raw:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                tid, dist = entry[0], entry[1]
            elif isinstance(entry, dict):
                tid = entry.get("track_id") or entry.get("trackId") or entry.get("tid")
                dist = entry.get("distance") or entry.get("dist")
            else:
                continue
            if tid is None or dist is None:
                continue
            try:
                candidates.append((int(tid), float(dist)))
            except (TypeError, ValueError):
                continue
        out.append(Contact(
            frame=int(c.get("frame", 0)),
            ball_x=float(c.get("ballX", c.get("ball_x", 0.0))),
            ball_y=float(c.get("ballY", c.get("ball_y", 0.0))),
            velocity=float(c.get("velocity", 0.0)),
            direction_change_deg=float(c.get("directionChangeDeg", c.get("direction_change_deg", 0.0))),
            player_track_id=int(c.get("playerTrackId", c.get("player_track_id", -1))),
            player_distance=float(c.get("playerDistance", c.get("player_distance", float("inf")) or float("inf"))),
            player_candidates=candidates,
            court_side=str(c.get("courtSide", c.get("court_side", "unknown"))),
            is_at_net=bool(c.get("isAtNet", c.get("is_at_net", False))),
            is_validated=bool(c.get("isValidated", c.get("is_validated", False))),
            confidence=float(c.get("confidence", 0.0)),
            arc_fit_residual=float(c.get("arcFitResidual", c.get("arc_fit_residual", 0.0))),
        ))
    return out


def _reconstruct_actions(actions_data: list[dict[str, Any]]) -> list[ClassifiedAction]:
    from rallycut.tracking.action_classifier import ActionType
    out: list[ClassifiedAction] = []
    for a in actions_data:
        if not isinstance(a, dict):
            continue
        try:
            t = ActionType(a.get("action", "unknown"))
        except (KeyError, ValueError):
            t = ActionType.UNKNOWN
        out.append(ClassifiedAction(
            action_type=t,
            frame=int(a.get("frame", 0)),
            ball_x=float(a.get("ballX", 0.0)),
            ball_y=float(a.get("ballY", 0.0)),
            velocity=float(a.get("velocity", 0.0)),
            player_track_id=int(a.get("playerTrackId", -1)),
            court_side=str(a.get("courtSide", "unknown")),
            confidence=float(a.get("confidence", 0.0)),
            is_synthetic=bool(a.get("isSynthetic", False)),
            team=str(a.get("team", "unknown")),
        ))
    return out


def _team_assignments_int(team_assignments_raw: dict[str, Any]) -> dict[int, int]:
    """Convert {tid_str: 'A'/'B'} -> {tid_int: 0/1}."""
    out: dict[int, int] = {}
    for k, v in team_assignments_raw.items():
        try:
            tid = int(k)
        except (TypeError, ValueError):
            continue
        if v == "A":
            out[tid] = 0
        elif v == "B":
            out[tid] = 1
    return out


def _load_video_ids() -> list[str]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT video_id FROM rallies "
                "WHERE status = %s OR status IS NULL "
                "ORDER BY video_id",
                ["CONFIRMED"],
            )
            return [str(row[0]) for row in cur.fetchall()]


def _load_rallies_with_payloads(video_id: str) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    """Return [(rally_id, actions_json, contacts_json), ...] for video."""
    query = """
        SELECT
            r.id AS rally_id,
            pt.actions_json,
            pt.contacts_json
        FROM rallies r
        JOIN player_tracks pt ON pt.rally_id = r.id
        WHERE r.video_id = %s
          AND (r.status = 'CONFIRMED' OR r.status IS NULL)
        ORDER BY r.start_ms
    """
    out: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, [video_id])
            for rally_id, actions_json, contacts_json in cur.fetchall():
                if not isinstance(actions_json, dict):
                    continue
                cj = contacts_json if isinstance(contacts_json, dict) else {}
                out.append((str(rally_id), actions_json, cj))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="In-process C-4 catalog with A1 volleyball-rule re-attribution.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    # Set the env flag so reattribute_players activates A1.
    a1_on = os.environ.get("USE_VOLLEYBALL_RULE_ATTRIBUTION") == "1"
    print(f"[catalog-a1] USE_VOLLEYBALL_RULE_ATTRIBUTION={'1 (A1 ON)' if a1_on else 'unset (BASELINE)'}",
          flush=True)

    video_ids = _load_video_ids()
    print(f"[catalog-a1] fleet has {len(video_ids)} videos", flush=True)

    t_start = time.monotonic()
    all_rows: list[dict[str, Any]] = []
    total_pairs_processed = 0
    total_rallies_scanned = 0

    for idx, video_id in enumerate(video_ids, start=1):
        # Mirror the same upstream-blocker exclusion used by coherence_run_all
        try:
            pid_violations, pid_stale = pid_run_all(video_id=video_id)
        except Exception as exc:
            print(f"[{idx}/{len(video_ids)}] video={video_id} pid_run_all ERROR: {exc}",
                  flush=True)
            continue
        excluded_rallies: set[str] = {
            v.rally_id for v in pid_violations
            if v.invariant in _UPSTREAM_BLOCKER_INVARIANTS
        }
        stale_rallies = pid_stale.skipped_stale_actions if pid_stale else set()
        pid_by_rally: dict[str, list[str]] = {}
        for v in pid_violations:
            pid_by_rally.setdefault(v.rally_id, []).append(v.invariant)

        rallies = _load_rallies_with_payloads(video_id)
        video_c4_pairs = 0
        for rally_id, actions_json, contacts_json in rallies:
            if rally_id in excluded_rallies:
                continue
            if rally_id in stale_rallies:
                continue

            actions_raw = actions_json.get("actions") or []
            team_assignments_raw = actions_json.get("teamAssignments") or {}
            contacts_raw = contacts_json.get("contacts") or [] if isinstance(contacts_json, dict) else []

            actions = _reconstruct_actions(actions_raw)
            contacts = _reconstruct_contacts(contacts_raw)
            team_ints = _team_assignments_int(team_assignments_raw)

            if team_ints and actions:
                reattribute_players(actions, contacts, team_ints)

            # Serialise back to dict form for the C-4 detector + row builder.
            new_actions = [a.to_dict() for a in actions]

            # Build rally_payload using same shape as production catalog
            # consumers expect.
            rally_payload = {
                "actions": new_actions,
                "team_assignments": team_assignments_raw,
                "contacts": contacts_raw,
            }

            violations = check_c4_no_same_player_back_to_back(
                rally_id=rally_id,
                actions=new_actions,
                team_assignments=team_assignments_raw,
            )
            total_rallies_scanned += 1
            video_c4_pairs += len(violations)

            for v in violations:
                if v.payload is None:
                    continue
                row = _row_from_violation(
                    rally_id=rally_id,
                    video_id=video_id,
                    payload=v.payload,
                    rally_payload=rally_payload,
                    # Co-violations not measured in this minimal harness.
                    co_violations={},
                    co_pid_invariants=pid_by_rally.get(rally_id, []),
                )
                all_rows.append(row)
                total_pairs_processed += 1

        elapsed = time.monotonic() - t_start
        print(f"[{idx}/{len(video_ids)}] video={video_id} "
              f"c4_pairs={video_c4_pairs} (rallies={len(rallies)}, "
              f"elapsed={elapsed:.1f}s)", flush=True)

    # Write CSV.
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    print(f"[catalog-a1] wrote {len(all_rows)} rows to {args.output}", flush=True)

    args.summary.write_text(_summarize(all_rows))
    print(f"[catalog-a1] wrote summary to {args.summary}", flush=True)
    print(f"[catalog-a1] scanned {total_rallies_scanned} rallies", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
