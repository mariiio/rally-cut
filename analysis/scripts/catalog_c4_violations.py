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

import math
import sys
from typing import Any

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


# ---------------------------------------------------------------------------
# DB orchestration lands in Task 7. Stub `main` for now so the module is
# importable but cannot be run as a script yet.
# ---------------------------------------------------------------------------
def main() -> int:
    print("catalog_c4_violations: DB orchestration not yet implemented (Task 7).",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
