"""Helpers for the relabel-with-crops worker (Phase 1.1).

Convert match_analysis_json on-disk shape into the typed objects
refine_assignments / replay_refine_from_scratchpad need, and write the
new assignments back. The CLI command (relabel_with_crops.py) is
orchestration on top of these.
"""

from __future__ import annotations

from typing import Any

from rallycut.tracking.match_tracker import RallyTrackingResult
from rallycut.tracking.player_features import PlayerAppearanceProfile


def reconstruct_initial_results(
    rally_entries: list[dict[str, Any]],
) -> list[RallyTrackingResult]:
    """Rebuild RallyTrackingResult list from match_analysis_json `rallies`.

    Used by the relabel worker as the `initial_results` argument to
    replay_refine_from_scratchpad — only rally_index and server_player_id
    are propagated; track_to_player gets recomputed by the replay.
    """
    results: list[RallyTrackingResult] = []
    for entry in rally_entries:
        results.append(RallyTrackingResult(
            rally_index=int(entry["rallyIndex"]),
            track_to_player={
                int(tid): int(pid)
                for tid, pid in entry.get("trackToPlayer", {}).items()
            },
            server_player_id=entry.get("serverPlayerId"),
            side_switch_detected=bool(entry.get("sideSwitchDetected", False)),
            assignment_confidence=float(entry.get("assignmentConfidence", 0.0)),
        ))
    return results


def reconstruct_profiles(
    profiles_dict: dict[str, dict[str, Any]],
) -> dict[int, PlayerAppearanceProfile]:
    """Rebuild pid → PlayerAppearanceProfile from match_analysis_json
    `playerProfiles`."""
    return {
        int(pid): PlayerAppearanceProfile.from_dict(d)
        for pid, d in profiles_dict.items()
    }


def apply_relabel_to_rally_entries(
    original_entries: list[dict[str, Any]],
    refined: list[RallyTrackingResult],
) -> list[dict[str, Any]]:
    """Produce updated rally_entries reflecting the replay's new assignments.

    Updates trackToPlayer, assignmentConfidence, sideSwitchDetected from the
    refined result. Preserves all other fields (rallyId, startMs, endMs,
    rallyIndex, serverPlayerId) from the original entry. Does not mutate
    `original_entries`.
    """
    if len(original_entries) != len(refined):
        raise ValueError(
            f"length mismatch: {len(original_entries)} entries vs "
            f"{len(refined)} refined results"
        )
    new_entries: list[dict[str, Any]] = []
    for orig, ref in zip(original_entries, refined):
        new_entry = dict(orig)
        new_entry["trackToPlayer"] = {
            str(tid): int(pid) for tid, pid in ref.track_to_player.items()
        }
        new_entry["assignmentConfidence"] = float(ref.assignment_confidence)
        new_entry["sideSwitchDetected"] = bool(ref.side_switch_detected)
        new_entries.append(new_entry)
    return new_entries
