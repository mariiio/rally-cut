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
