"""Env-gated profile-drift probe for the MatchSolver path.

Activates when ``MATCH_PLAYERS_PROBE=1``. Captures per-iteration per-rally
Hungarian state inside ``MatchSolver`` and per-rally profile snapshots
before/after ``_update_profiles`` calls in the post-solve sweep. Writes a
single sidecar JSON per match-players invocation under
``analysis/reports/profile_drift_probe/``.

Probe is process-global (one match-players run per process). Call sites push
records via ``record_*``; the outer entry calls ``begin_probe`` and
``finalize_probe``.

This is NOT a replacement for the deleted ``pipeline_forensic.py`` (commit
``ce7b08c``). It is intentionally narrow, isolated to a single module, and
trivial to revert.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from rallycut.tracking.player_features import PlayerAppearanceProfile

logger = logging.getLogger(__name__)

ENV_FLAG = "MATCH_PLAYERS_PROBE"
ENV_DROP_EMA_FLAG = "EXPERIMENTAL_DROP_PROFILE_EMA"


def is_enabled() -> bool:
    return os.environ.get(ENV_FLAG, "0") == "1"


_state: dict[str, Any] | None = None


def begin_probe(
    video_id: str,
    *,
    rally_ids: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Initialize a probe session. No-op when ``MATCH_PLAYERS_PROBE`` is unset."""
    global _state
    if not is_enabled():
        _state = None
        return
    _state = {
        "video_id": video_id,
        "rally_ids": list(rally_ids) if rally_ids else [],
        "started_at": time.time(),
        "extra": dict(extra or {}),
        "iter_records": [],
        "update_records": [],
    }


def _l2(x: np.ndarray | None) -> float | None:
    """L2 norm — captures EMA peakedness changes for normalized histograms.

    For HSV histograms (sum=1.0), L1 is identically 1.0 across rallies and
    can't detect drift. L2 = sqrt(sum(bin^2)) varies with concentration as
    EMA blending shifts mass between bins, so two profiles with different
    distributions return different L2 even when both sum to 1.
    """
    if x is None:
        return None
    return float(np.linalg.norm(x))


def _checksum_profile(prof: PlayerAppearanceProfile) -> dict[str, Any]:
    return {
        "rally_count": int(prof.rally_count),
        "skin_sample_count": int(prof.skin_sample_count),
        "avg_skin_tone_hsv": (
            list(prof.avg_skin_tone_hsv) if prof.avg_skin_tone_hsv else None
        ),
        "upper_hist_count": int(prof.upper_hist_count),
        "upper_hist_l2": _l2(prof.avg_upper_hist),
        "lower_hist_count": int(prof.lower_hist_count),
        "lower_hist_l2": _l2(prof.avg_lower_hist),
        "upper_v_hist_count": int(prof.upper_v_hist_count),
        "upper_v_hist_l2": _l2(prof.avg_upper_v_hist),
        "lower_v_hist_count": int(prof.lower_v_hist_count),
        "lower_v_hist_l2": _l2(prof.avg_lower_v_hist),
        "dominant_color_count": int(prof.dominant_color_count),
        "avg_dominant_color_hsv": (
            list(prof.avg_dominant_color_hsv) if prof.avg_dominant_color_hsv else None
        ),
        "head_hist_count": int(prof.head_hist_count),
        "head_hist_l2": _l2(prof.avg_head_hist),
        "reid_embedding_count": int(prof.reid_embedding_count),
        "reid_embedding_norm": _l2(prof.reid_embedding),
    }


def checksum_profiles(
    profiles: dict[int, PlayerAppearanceProfile],
) -> dict[str, dict[str, Any]]:
    return {str(pid): _checksum_profile(p) for pid, p in profiles.items()}


def record_solver_iteration(
    *,
    iteration: int,
    rally_idx: int,
    top_tracks: list[int],
    cluster_ids: list[int],
    cost_matrix: np.ndarray,
    assignment: dict[int, int],
    prev_assignment: dict[int, int] | None,
) -> None:
    if _state is None:
        return
    cm = np.asarray(cost_matrix, dtype=float)
    margins: list[float | None] = []
    for r in range(cm.shape[0]):
        row_sorted = np.sort(cm[r])
        margins.append(
            float(row_sorted[1] - row_sorted[0]) if cm.shape[1] >= 2 else None
        )
    prev = prev_assignment or {}
    changed = {
        int(tid): {"prev": int(prev.get(tid, -1)), "new": int(cid)}
        for tid, cid in assignment.items()
        if prev.get(tid) != cid
    }
    _state["iter_records"].append({
        "iteration": int(iteration),
        "rally_idx": int(rally_idx),
        "top_tracks": [int(t) for t in top_tracks],
        "cluster_ids": [int(c) for c in cluster_ids],
        "cost_matrix": cm.tolist(),
        "row_margins": margins,
        "assignment": {str(int(tid)): int(cid) for tid, cid in assignment.items()},
        "changed_from_prev": changed,
    })


def record_update_profiles(
    *,
    rally_idx: int,
    track_to_player: dict[int, int],
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
    context: str = "post_solve",
) -> None:
    if _state is None:
        return
    _state["update_records"].append({
        "rally_idx": int(rally_idx),
        "context": context,
        "track_to_player": {
            str(int(tid)): int(pid) for tid, pid in track_to_player.items()
        },
        "before": before,
        "after": after,
    })


def finalize_probe() -> Path | None:
    """Write sidecar JSON and reset the probe. Returns the path written, or None."""
    global _state
    if _state is None:
        return None
    started_at = float(_state["started_at"])
    video_id = str(_state["video_id"])
    payload = {
        "video_id": video_id,
        "rally_ids": _state["rally_ids"],
        "started_at": started_at,
        "finished_at": time.time(),
        ENV_DROP_EMA_FLAG: os.environ.get(ENV_DROP_EMA_FLAG, "0"),
        "extra": _state["extra"],
        "iter_records": _state["iter_records"],
        "update_records": _state["update_records"],
    }
    output_dir = Path(__file__).resolve().parents[2] / "reports" / "profile_drift_probe"
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(started_at))
    short = video_id[:8] if len(video_id) >= 8 else video_id
    drop_tag = "dropema" if payload[ENV_DROP_EMA_FLAG] == "1" else "baseline"
    path = output_dir / f"{short}_{drop_tag}_{ts}.json"
    path.write_text(json.dumps(payload, indent=2))
    logger.info(
        "ProfileDriftProbe sidecar: %s (iter=%d, update=%d)",
        path, len(payload["iter_records"]), len(payload["update_records"]),
    )
    _state = None
    return path
