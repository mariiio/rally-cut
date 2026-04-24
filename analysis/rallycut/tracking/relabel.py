"""Helpers for the relabel-with-crops worker (Phase 1.1).

Convert match_analysis_json on-disk shape into the typed objects
refine_assignments / replay_refine_from_scratchpad need, and write the
new assignments back. The CLI command (relabel_with_crops.py) is
orchestration on top of these.
"""

from __future__ import annotations

from typing import Any

import numpy as np

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


def detect_anomalous_crops(
    embeddings_per_pid: dict[int, np.ndarray],
    threshold_sigma: float = 1.5,
    min_distance_floor: float = 0.05,
) -> dict[int, list[int]]:
    """Flag crops whose embedding is far from the rest of their pid cohort.

    Plan §5 Q1 refinement: when the user provides ≥2 crops for the same
    pid, surface a warning if one looks visually different from the others.
    A miss-click on the wrong player should land here.

    Algorithm:
        1. For each pid with ≥3 crops:
           a. Compute the (N, N) cosine-distance matrix.
           b. For each crop i, compute median distance to the other N-1 crops.
           c. Flag crops where BOTH:
              - per_crop_median[i] > cohort_median + threshold_sigma * cohort_std
              - per_crop_median[i] - cohort_median > min_distance_floor
              The second clause prevents false positives on visually-uniform
              cohorts where σ is tiny but no real outlier exists.
        2. pids with <3 crops return [] (no cohort to compare against).

    Args:
        embeddings_per_pid: pid → (N, D) array of L2-normalized embeddings
            (e.g. DINOv2 backbone features). Empty arrays / N<3 → [].
        threshold_sigma: stddev multiplier above cohort median that
            triggers a flag. Default 1.5 — surface enough to catch real
            misclicks without nagging on benign within-player variation.
        min_distance_floor: absolute cosine-distance gap (above cohort
            median) that a crop must clear before being flagged. Default
            0.05 — about 5% cosine distance, well above lighting/pose
            variation but below "different player" distance.

    Returns:
        pid → sorted list of crop indices flagged as outliers.
    """
    flagged: dict[int, list[int]] = {}
    for pid, embeddings in embeddings_per_pid.items():
        n = embeddings.shape[0]
        if n < 3:
            flagged[pid] = []
            continue
        # Cosine distance = 1 - cosine similarity. Embeddings assumed normalized.
        sims = embeddings @ embeddings.T
        dists = 1.0 - sims
        np.fill_diagonal(dists, np.nan)  # Ignore self-distance.
        per_crop_median = np.nanmedian(dists, axis=1)
        cohort_median = float(np.median(per_crop_median))
        cohort_std = float(np.std(per_crop_median))
        if cohort_std == 0.0:
            flagged[pid] = []
            continue
        sigma_threshold = cohort_median + threshold_sigma * cohort_std
        absolute_threshold = cohort_median + min_distance_floor
        flagged[pid] = sorted(
            int(i) for i in range(n)
            if per_crop_median[i] > sigma_threshold
            and per_crop_median[i] > absolute_threshold
        )
    return flagged


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
