"""Per-rally cascade-stage trace recorder.

Opt-in: env flag `CASCADE_TRACE_OUT` must point at an existing directory.
When set, `cascade_trace(rally_id)` yields a CascadeTrace that records the
playerTrackId + action_type of every action at each stage boundary in the
action-cascade pipeline. On exit, writes `{CASCADE_TRACE_OUT}/{rally_id}.trace.json`.

When env unset, `cascade_trace(...)` is a no-op context manager that yields
a sentinel object whose `snapshot(...)` calls return immediately (zero cost
beyond a dict lookup).

Used by: scripts/audit_cascade_override_2026_05_20.py
Spec: docs/superpowers/specs/2026-05-20-attribution-headroom-decomposition-design.md
"""
from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CascadeTrace:
    rally_id: str
    out_dir: Path | None
    snapshots: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_enabled(self) -> bool:
        return self.out_dir is not None

    def snapshot(self, stage: str, actions: list[Any]) -> None:
        """Record per-action (frame, action_type, player_track_id) at this stage."""
        if not self.is_enabled:
            return
        # Duck-typed: avoid importing action_classifier types to prevent a
        # circular dependency with the module this instruments.
        per_action = []
        for a in actions:
            per_action.append({
                "frame": int(getattr(a, "frame", -1)),
                "action_type": str(getattr(a, "action_type", "")),
                "player_track_id": int(getattr(a, "player_track_id", -1)),
            })
        self.snapshots.append({"stage": stage, "actions": per_action})

    def write(self) -> None:
        if self.out_dir is None:
            return
        # Per-contact pivot: frame -> stage -> {action_type, player_track_id}.
        # If two actions share a frame within one stage snapshot, last writer
        # wins — expected use case is one action per frame.
        per_contact: dict[str, dict[str, dict[str, Any]]] = {}
        for snap in self.snapshots:
            stage = snap["stage"]
            for a in snap["actions"]:
                key = str(a["frame"])
                if key not in per_contact:
                    per_contact[key] = {}
                per_contact[key][stage] = {
                    "action_type": a["action_type"],
                    "player_track_id": a["player_track_id"],
                }
        payload = {
            "rally_id": self.rally_id,
            "snapshots": self.snapshots,
            "per_contact": per_contact,
        }
        out_path = self.out_dir / f"{self.rally_id}.trace.json"
        out_path.write_text(json.dumps(payload, indent=2))
        logger.debug("Wrote cascade trace -> %s", out_path)


@contextmanager
def cascade_trace(rally_id: str) -> Iterator[CascadeTrace]:
    """Context manager that yields a CascadeTrace if CASCADE_TRACE_OUT is set."""
    out_dir_str = os.environ.get("CASCADE_TRACE_OUT")
    out_dir: Path | None = None
    if out_dir_str:
        out_dir = Path(out_dir_str)
        out_dir.mkdir(parents=True, exist_ok=True)
    tr = CascadeTrace(rally_id=rally_id, out_dir=out_dir)
    try:
        yield tr
    finally:
        tr.write()
