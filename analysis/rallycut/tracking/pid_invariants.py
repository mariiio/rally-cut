"""PID-attribution invariants for the analysis pipeline.

Eval-time enforcement only: production never imports these checks. The audit
CLI (`rallycut audit-pid-invariants`) and the panel eval script wire them in.

Each invariant has a dedicated `check_iN_*` function returning a list of
Violation records. `run_all` orchestrates DB loading and aggregation.

Invariants (see docs/superpowers/specs/2026-05-08-pid-leverage-audit-sub1-design.md):
  I-1: len(primary_track_ids) == 4 (or 0 if filter disabled)
  I-2: every trackId in positionsJson ∈ primary_track_ids
  I-3: every action's playerTrackId ∈ primary_track_ids ∪ {-1}
  I-4: every contact's playerTrackId ∈ primary_track_ids ∪ {-1}
  I-5: trackToPlayer is total over primary_track_ids
  I-6: team_assignments is total over primary_track_ids
  I-7: after stats mapping, every action's player_track_id ∈ {1..4} ∪ {-1}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class Violation:
    invariant: str
    rally_id: str
    detail: str
    severity: Literal["error", "warn"] = "error"


def check_i1_primary_set_size(
    *,
    rally_id: str,
    primary_track_ids: list[int],
) -> list[Violation]:
    """I-1: primary_track_ids must have exactly 4 entries (or 0 if filter disabled)."""
    n = len(primary_track_ids)
    if n in (0, 4):
        return []
    return [
        Violation(
            invariant="I-1",
            rally_id=rally_id,
            detail=f"primary_track_ids has size {n}, expected 4 (or 0 if filter disabled)",
        )
    ]


def check_i2_positions_in_primary(
    *,
    rally_id: str,
    primary_track_ids: list[int],
    positions_json: list[dict[str, Any]] | None,
) -> list[Violation]:
    """I-2: every trackId in positions_json must be in primary_track_ids."""
    if not positions_json or not primary_track_ids:
        return []
    primary = set(primary_track_ids)
    offenders: set[int] = set()
    for p in positions_json:
        tid = p.get("trackId")
        if tid is None:
            continue
        if int(tid) not in primary:
            offenders.add(int(tid))
    return [
        Violation(
            invariant="I-2",
            rally_id=rally_id,
            detail=f"positions_json contains non-primary trackId={tid} (primary={sorted(primary)})",
        )
        for tid in sorted(offenders)
    ]


def check_i3_action_attribution(
    *,
    rally_id: str,
    primary_track_ids: list[int],
    actions_json: list[dict[str, Any]] | None,
) -> list[Violation]:
    """I-3: every action's playerTrackId must be in primary ∪ {-1}."""
    if not actions_json or not primary_track_ids:
        return []
    allowed = set(primary_track_ids) | {-1}
    violations: list[Violation] = []
    for idx, a in enumerate(actions_json):
        tid = a.get("playerTrackId")
        if tid is None:
            continue
        if int(tid) not in allowed:
            violations.append(
                Violation(
                    invariant="I-3",
                    rally_id=rally_id,
                    detail=(
                        f"action[{idx}] playerTrackId={tid} not in primary "
                        f"{sorted(primary_track_ids)} ∪ {{-1}} "
                        f"(action={a.get('action')!r}, frame={a.get('frame')})"
                    ),
                )
            )
    return violations


def check_i4_contact_attribution(
    *,
    rally_id: str,
    primary_track_ids: list[int],
    contacts_json: list[dict[str, Any]] | None,
) -> list[Violation]:
    """I-4: every contact's playerTrackId must be in primary ∪ {-1}."""
    if not contacts_json or not primary_track_ids:
        return []
    allowed = set(primary_track_ids) | {-1}
    violations: list[Violation] = []
    for idx, c in enumerate(contacts_json):
        tid = c.get("playerTrackId")
        if tid is None:
            continue
        if int(tid) not in allowed:
            violations.append(
                Violation(
                    invariant="I-4",
                    rally_id=rally_id,
                    detail=(
                        f"contact[{idx}] playerTrackId={tid} not in primary "
                        f"{sorted(primary_track_ids)} ∪ {{-1}} (frame={c.get('frame')})"
                    ),
                )
            )
    return violations


def check_i5_track_to_player_total(
    *,
    rally_id: str,
    primary_track_ids: list[int],
    track_to_player: dict[str, int] | None,
) -> list[Violation]:
    """I-5: trackToPlayer must map every primary_track_id to a PID in {1..4}."""
    if not primary_track_ids:
        return []
    mapping = track_to_player or {}
    # Normalize keys: JSON serializes int keys as strings.
    normalized = {int(k): int(v) for k, v in mapping.items()}
    violations: list[Violation] = []
    for tid in primary_track_ids:
        if tid not in normalized:
            violations.append(
                Violation(
                    invariant="I-5",
                    rally_id=rally_id,
                    detail=f"primary track {tid} missing from trackToPlayer (have keys {sorted(normalized.keys())})",
                )
            )
            continue
        pid = normalized[tid]
        if pid not in (1, 2, 3, 4):
            violations.append(
                Violation(
                    invariant="I-5",
                    rally_id=rally_id,
                    detail=f"trackToPlayer[{tid}]=pid={pid}, expected 1..4",
                )
            )
    return violations
