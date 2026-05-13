"""Catalog harness for C-4 (no-same-player-back-to-back) coherence violations.

Walks the labeled fleet, runs `coherence_invariants.run_all` per video,
filters to C-4 violations, computes per-pair evidence signals, writes a
CSV row + a fleet-aggregate summary markdown. The CSV is the input to the
Phase 1 → Phase 2 gated review (see spec).

Pure signal-computation helpers live at module top so they're unit-testable
without DB access. The DB orchestration is in `main`.

Spec: docs/superpowers/specs/2026-05-13-sub-2b-coherence-repair-design.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from rallycut.evaluation.tracking.db import get_connection
from rallycut.tracking.coherence_invariants import run_all as coherence_run_all
from rallycut.tracking.pid_invariants import run_all as pid_run_all

# ---------------------------------------------------------------------------
# Expected-transitions table for signal_type_fit. Conservative: only known
# pairs are labeled; unknown pairs return 'unknown' (NOT 'wrong').
# ---------------------------------------------------------------------------
EXPECTED_TRANSITIONS: dict[tuple[str, str], tuple[str | None, str]] = {
    # (prev_type, curr_type): (expected_curr_team_relation, label)
    ("serve",   "receive"): ("other",  "ok"),
    ("serve",   "dig"):     ("other",  "ok"),
    ("serve",   "set"):     (None,     "wrong"),
    ("serve",   "attack"):  (None,     "wrong"),
    ("serve",   "block"):   ("other",  "ok"),
    ("receive", "set"):     ("same",   "ok"),
    ("receive", "attack"):  ("same",   "ok"),
    ("set",     "attack"):  ("same",   "ok"),
    ("set",     "set"):     (None,     "wrong"),
    ("attack",  "dig"):     ("other",  "ok"),
    ("attack",  "block"):   ("other",  "ok"),
    ("attack",  "receive"): ("other",  "ok"),
    ("dig",     "set"):     ("same",   "ok"),
    ("dig",     "attack"):  ("same",   "ok"),
}
# Block-prev fallback: any block→X is 'ok' (block exempts the C-4 pair).
_BLOCK_PREV_FALLBACK_LABEL = "ok"


def signal_type_fit(prev_action: str, curr_action: str) -> str:
    """Return 'ok' | 'wrong' | 'unknown' for the action-type transition.

    Block-prev pairs are always 'ok' (the C-4 detector exempts them).
    """
    if prev_action == "block":
        return _BLOCK_PREV_FALLBACK_LABEL
    entry = EXPECTED_TRANSITIONS.get((prev_action, curr_action))
    if entry is None:
        return "unknown"
    _, label = entry
    return label


def best_same_team_alt_ratio(
    *,
    candidates: list[tuple[int, float, str]],
    current_dist: float,
    current_team: str,
    current_tid: int,
) -> float:
    """Return best same-team alternative's distance / current_dist.

    < 1.0 means a same-team alternative is closer than the current player.
    NaN if no same-team alternative exists or current_dist is non-finite.
    """
    if not math.isfinite(current_dist) or current_dist <= 0:
        return math.nan
    best: float = math.inf
    for tid, dist, team in candidates:
        if tid == current_tid:
            continue
        if team != current_team:
            continue
        if dist < best:
            best = dist
    if not math.isfinite(best):
        return math.nan
    return best / current_dist


def signal_team_geometry(
    *,
    candidates: list[tuple[int, float, str]],
    expected_team: str | None,
) -> str:
    """Return 'matches' | 'violates' | 'ambiguous'.

    'matches' if rank-1 candidate is on the expected team.
    'violates' if rank-1 is wrong team but a rank-2+ same-team candidate
    is within 2x distance of rank-1.
    'ambiguous' otherwise (no candidates, no expected team, no nearby
    same-team alternative).
    """
    if expected_team is None or not candidates:
        return "ambiguous"
    rank1_tid, rank1_dist, rank1_team = candidates[0]
    if rank1_team == expected_team:
        return "matches"
    # rank1 is wrong team — does a same-team candidate sit within 2x?
    if not math.isfinite(rank1_dist) or rank1_dist <= 0:
        return "ambiguous"
    threshold = 2.0 * rank1_dist
    for tid, dist, team in candidates[1:]:
        if team == expected_team and math.isfinite(dist) and dist <= threshold:
            return "violates"
    return "ambiguous"


def placeholder_repair_recommendation(row: dict[str, Any]) -> str:
    """Hypothesis-only rule scoring evidence on each side of the pair.

    Counts "strong-against" signals per side: type_fit=='wrong',
    team_geometry=='violates', alt_ratio < 0.6 (closer same-team alt
    exists), confidence < 0.5. If one side has ≥2 strong-against and the
    other has ≤1, recommend repairing that side. If both have 0,
    recommend 'skip'. Otherwise 'ambiguous'.

    Not shipped to production. Exists so the gated review can falsify it
    against hand-classified root_cause labels.
    """
    def strong_against(prefix: str) -> int:
        count = 0
        if row.get(f"signal_type_fit_{prefix}") == "wrong":
            count += 1
        if row.get(f"signal_team_geometry_{prefix}") == "violates":
            count += 1
        alt = row.get(f"{prefix}_best_same_team_alt_ratio")
        if isinstance(alt, (int, float)) and math.isfinite(alt) and alt < 0.6:
            count += 1
        conf = row.get(f"conf_{prefix}")
        if isinstance(conf, (int, float)) and conf < 0.5:
            count += 1
        return count

    prev_count = strong_against("prev")
    curr_count = strong_against("curr")

    if prev_count >= 2 and curr_count <= 1:
        return "repair_prev"
    if curr_count >= 2 and prev_count <= 1:
        return "repair_curr"
    if prev_count == 0 and curr_count == 0:
        return "skip"
    return "ambiguous"


CSV_COLUMNS = [
    "rally_id", "video_id", "pair_idx",
    "frame_prev", "frame_curr",
    "action_prev_type", "action_curr_type",
    "player_id", "team_label",
    "conf_prev", "conf_curr",
    "prev_player_dist", "curr_player_dist",
    "prev_top3_candidates", "curr_top3_candidates",
    "prev_best_same_team_alt_ratio", "curr_best_same_team_alt_ratio",
    "signal_type_fit_prev", "signal_type_fit_curr",
    "signal_team_geometry_prev", "signal_team_geometry_curr",
    "co_c1_fires", "co_c2_fires", "co_c3_fires",
    "co_pid_invariant_fires",
    "repair_recommendation",
    "root_cause", "notes",
]


def _load_video_ids() -> list[str]:
    """All videos that have at least one CONFIRMED-or-NULL rally."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT video_id FROM rallies "
                "WHERE status = %s OR status IS NULL "
                "ORDER BY video_id",
                ["CONFIRMED"],
            )
            return [str(row[0]) for row in cur.fetchall()]


def _load_rally_payloads(video_id: str) -> dict[str, dict[str, Any]]:
    """For each rally in the video, load actions_json + contacts_json.

    Returns {rally_id: {"actions": ..., "team_assignments": ..., "contacts": ...}}.
    """
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
    out: dict[str, dict[str, Any]] = {}
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, [video_id])
            for rally_id, actions_json, contacts_json in cur.fetchall():
                if not isinstance(actions_json, dict):
                    continue
                actions = actions_json.get("actions") or []
                team_assignments = actions_json.get("teamAssignments") or {}
                contacts: list[dict[str, Any]] = []
                if isinstance(contacts_json, dict):
                    contacts = contacts_json.get("contacts") or []
                elif isinstance(contacts_json, list):
                    contacts = contacts_json
                out[str(rally_id)] = {
                    "actions": actions,
                    "team_assignments": team_assignments,
                    "contacts": contacts,
                }
    return out


def _contact_by_frame(contacts: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Index contacts by frame for O(1) lookup."""
    by_frame: dict[int, dict[str, Any]] = {}
    for c in contacts:
        frame = c.get("frame")
        if isinstance(frame, int):
            by_frame[frame] = c
    return by_frame


def _extract_top3_candidates(
    contact: dict[str, Any] | None,
    team_assignments: dict[str, str],
) -> list[tuple[int, float, str]]:
    """Return up to 3 (tid, dist, team_label) tuples from contact.player_candidates."""
    if contact is None:
        return []
    raw = contact.get("player_candidates") or contact.get("playerCandidates") or []
    out: list[tuple[int, float, str]] = []
    for entry in raw[:3]:
        if isinstance(entry, dict):
            tid = entry.get("track_id") or entry.get("trackId") or entry.get("tid")
            dist = entry.get("distance") or entry.get("dist")
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
            tid, dist = entry[0], entry[1]
        else:
            continue
        if tid is None or dist is None:
            continue
        try:
            tid_i = int(tid)
            dist_f = float(dist)
        except (TypeError, ValueError):
            continue
        team = team_assignments.get(str(tid_i), "?")
        out.append((tid_i, dist_f, team))
    return out


def _expected_team_for_curr(prev_action: str, prev_team: str) -> str | None:
    """Derive the expected curr-action team from prev's action + team using
    the EXPECTED_TRANSITIONS relations ('same' / 'other' / 'either' / None)."""
    if prev_action == "block":
        return None  # block exempts; geometry is 'ambiguous' for curr
    for (p_a, _c_a), (relation, _label) in EXPECTED_TRANSITIONS.items():
        # The relation is shared per prev row in the table; pick first 'ok' row.
        if p_a == prev_action and relation in ("same", "other"):
            return prev_team if relation == "same" else (
                "B" if prev_team == "A" else "A"
            )
    return None


def _row_from_violation(
    *,
    rally_id: str,
    video_id: str,
    payload: dict[str, Any],
    rally_payload: dict[str, Any],
    co_violations: dict[str, bool],
    co_pid_invariants: list[str],
) -> dict[str, Any]:
    """Build one CSV row for a C-4 violation."""
    actions: list[dict[str, Any]] = rally_payload["actions"]
    team_assignments: dict[str, str] = rally_payload["team_assignments"]
    contacts_by_frame = _contact_by_frame(rally_payload["contacts"])

    prev_idx = payload["prev_index"]
    curr_idx = payload["curr_index"]
    # Re-sort actions the same way the detector did, so indices align.
    sorted_actions = sorted(actions, key=lambda a: int(a.get("frame", 0)))
    prev = sorted_actions[prev_idx]
    curr = sorted_actions[curr_idx]

    prev_pid = int(prev.get("playerTrackId", -1))
    curr_pid = int(curr.get("playerTrackId", -1))
    team_label = team_assignments.get(str(prev_pid), "?")

    prev_contact = contacts_by_frame.get(int(prev.get("frame", -1)))
    curr_contact = contacts_by_frame.get(int(curr.get("frame", -1)))

    prev_top3 = _extract_top3_candidates(prev_contact, team_assignments)
    curr_top3 = _extract_top3_candidates(curr_contact, team_assignments)

    prev_player_dist = float(
        (prev_contact or {}).get(
            "player_distance",
            (prev_contact or {}).get("playerDistance", math.nan),
        )
    )
    curr_player_dist = float(
        (curr_contact or {}).get(
            "player_distance",
            (curr_contact or {}).get("playerDistance", math.nan),
        )
    )

    prev_alt_ratio = best_same_team_alt_ratio(
        candidates=prev_top3,
        current_dist=prev_player_dist,
        current_team=team_label,
        current_tid=prev_pid,
    )
    curr_alt_ratio = best_same_team_alt_ratio(
        candidates=curr_top3,
        current_dist=curr_player_dist,
        current_team=team_label,
        current_tid=curr_pid,
    )

    # Expected team for curr: derived from prev_action + prev_team.
    expected_curr_team = _expected_team_for_curr(
        prev_action=payload["prev_action"], prev_team=team_label,
    )
    # Expected team for prev: derived from action[i-2] when available.
    expected_prev_team: str | None = None
    if prev_idx >= 1:
        action_before_prev = sorted_actions[prev_idx - 1]
        before_prev_pid = action_before_prev.get("playerTrackId")
        if isinstance(before_prev_pid, int):
            before_prev_team = team_assignments.get(str(before_prev_pid))
            if before_prev_team:
                expected_prev_team = _expected_team_for_curr(
                    prev_action=str(action_before_prev.get("action", "")),
                    prev_team=before_prev_team,
                )

    # signal_type_fit for prev: needs action[i-2] → action[i-1] transition.
    signal_type_fit_prev_v = "unknown"  # cannot compute without action[i-2]
    if prev_idx >= 1:
        signal_type_fit_prev_v = signal_type_fit(
            str(sorted_actions[prev_idx - 1].get("action", "")),
            payload["prev_action"],
        )
    # signal_type_fit for curr: prev_action → curr_action transition.
    signal_type_fit_curr_v = signal_type_fit(
        payload["prev_action"], payload["curr_action"],
    )

    row: dict[str, Any] = {
        "rally_id": rally_id,
        "video_id": video_id,
        "pair_idx": curr_idx,
        "frame_prev": payload["prev_frame"],
        "frame_curr": payload["curr_frame"],
        "action_prev_type": payload["prev_action"],
        "action_curr_type": payload["curr_action"],
        "player_id": payload["player_id"],
        "team_label": team_label,
        "conf_prev": prev.get("confidence", math.nan),
        "conf_curr": curr.get("confidence", math.nan),
        "prev_player_dist": prev_player_dist,
        "curr_player_dist": curr_player_dist,
        "prev_top3_candidates": json.dumps(prev_top3),
        "curr_top3_candidates": json.dumps(curr_top3),
        "prev_best_same_team_alt_ratio": prev_alt_ratio,
        "curr_best_same_team_alt_ratio": curr_alt_ratio,
        "signal_type_fit_prev": signal_type_fit_prev_v,
        "signal_type_fit_curr": signal_type_fit_curr_v,
        "signal_team_geometry_prev": signal_team_geometry(
            candidates=prev_top3, expected_team=expected_prev_team,
        ),
        "signal_team_geometry_curr": signal_team_geometry(
            candidates=curr_top3, expected_team=expected_curr_team,
        ),
        "co_c1_fires": co_violations.get("C-1", False),
        "co_c2_fires": co_violations.get("C-2", False),
        "co_c3_fires": co_violations.get("C-3", False),
        "co_pid_invariant_fires": ",".join(co_pid_invariants),
        "root_cause": "",
        "notes": "",
    }
    row["repair_recommendation"] = placeholder_repair_recommendation(row)
    return row


def _summarize(rows: list[dict[str, Any]]) -> str:
    """Build the fleet-aggregate summary markdown."""
    total = len(rows)
    pair_counter: Counter[tuple[str, str]] = Counter(
        (r["action_prev_type"], r["action_curr_type"]) for r in rows
    )
    co_c2 = sum(1 for r in rows if r["co_c2_fires"])
    co_c3 = sum(1 for r in rows if r["co_c3_fires"])
    rec_counter: Counter[str] = Counter(r["repair_recommendation"] for r in rows)
    by_video: Counter[str] = Counter(r["video_id"] for r in rows)
    top_videos = by_video.most_common(10)

    lines: list[str] = []
    lines.append("# C-4 Fleet Catalog Summary\n")
    lines.append(f"**Total C-4 violations:** {total}\n")
    pct_c2 = 100 * co_c2 / total if total else 0
    pct_c3 = 100 * co_c3 / total if total else 0
    lines.append(
        f"**Co-violation rates:** C-2 fires on {co_c2} ({pct_c2:.1f}%), "
        f"C-3 fires on {co_c3} ({pct_c3:.1f}%).\n"
    )
    lines.append("## (prev_action, curr_action) breakdown\n")
    lines.append("| prev | curr | count |\n|---|---|---:|")
    for (prev, curr), count in pair_counter.most_common():
        lines.append(f"| {prev} | {curr} | {count} |")
    lines.append("\n## Placeholder repair_recommendation distribution\n")
    lines.append("| recommendation | count |\n|---|---:|")
    for rec, count in rec_counter.most_common():
        lines.append(f"| {rec} | {count} |")
    lines.append("\n## Top 10 worst videos\n")
    lines.append("| video_id | C-4 count |\n|---|---:|")
    for video_id, count in top_videos:
        lines.append(f"| {video_id} | {count} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Walk the fleet, write a per-pair C-4 violation catalog.",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Path to write the per-violation CSV.",
    )
    parser.add_argument(
        "--summary", type=Path, required=True,
        help="Path to write the fleet-aggregate markdown summary.",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    video_ids = _load_video_ids()
    print(f"[catalog] fleet has {len(video_ids)} videos", flush=True)

    all_rows: list[dict[str, Any]] = []
    for idx, video_id in enumerate(video_ids, start=1):
        try:
            coherence_violations = coherence_run_all(video_id=video_id)
        except Exception as exc:
            print(
                f"[{idx}/{len(video_ids)}] video={video_id} ERROR: {exc}",
                flush=True,
            )
            continue
        try:
            pid_violations, _stale = pid_run_all(video_id=video_id)
        except Exception:
            pid_violations = []

        rally_payloads = _load_rally_payloads(video_id)

        # Per-rally co-violation + PID-violation lookup tables.
        co_by_rally: dict[str, dict[str, bool]] = {}
        for v in coherence_violations:
            entry = co_by_rally.setdefault(v.rally_id, {
                "C-1": False, "C-2": False, "C-3": False, "C-4": False,
            })
            entry[v.invariant] = True
        pid_by_rally: dict[str, list[str]] = {}
        for v in pid_violations:
            pid_by_rally.setdefault(v.rally_id, []).append(v.invariant)

        c4_violations = [v for v in coherence_violations if v.invariant == "C-4"]
        rallies_with_c4 = {v.rally_id for v in c4_violations}
        print(
            f"[{idx}/{len(video_ids)}] video={video_id} "
            f"c4_pairs={len(c4_violations)} "
            f"rallies_with_c4={len(rallies_with_c4)}",
            flush=True,
        )

        for v in c4_violations:
            if v.payload is None:
                continue
            rally_payload = rally_payloads.get(v.rally_id)
            if rally_payload is None:
                continue
            row = _row_from_violation(
                rally_id=v.rally_id,
                video_id=video_id,
                payload=v.payload,
                rally_payload=rally_payload,
                co_violations=co_by_rally.get(v.rally_id, {}),
                co_pid_invariants=pid_by_rally.get(v.rally_id, []),
            )
            all_rows.append(row)

    # Write CSV.
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    print(f"[catalog] wrote {len(all_rows)} rows to {args.output}", flush=True)

    # Write summary markdown.
    args.summary.write_text(_summarize(all_rows))
    print(f"[catalog] wrote summary to {args.summary}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
