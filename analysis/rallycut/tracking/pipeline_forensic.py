"""Read-only post-tracking pipeline forensic logger.

Companion to `botsort_instrumented.py`: instead of capturing per-frame BoxMOT
state, this module captures per-stage state for the 13 post-tracking pipeline
stages (height-swap, color-split, tracklet-link, primary-filter, top-4,
per-frame, team-classification, within-rally global identity, drift, plus
W4 split + cross-rally Hungarian).

Activation: env var `PIPELINE_FORENSIC_LOG_DIR` AND `PIPELINE_FORENSIC_RALLY_TAG`
both set. When unset, every public function short-circuits via `is_active()` so
production behavior is byte-identical to a build without this module.

Sidecar layout: one JSONL per rally at
`<LOG_DIR>/<RALLY_TAG>.jsonl`. Line 1 is a `meta` record (rally tag + schema
version + expected stages). Subsequent lines are per-stage records tagged
`{"stage": "<n>_<name>", ...}`.

Free-function stages (`fix_height_swaps`, `split_tracks_by_color`,
`link_tracklets_by_appearance`, `detect_convergence_swaps`, etc.) are
instrumented at the call site in `apply_post_processing()` /
`process_rally()`. Stages whose return shape doesn't expose internal decisions
(e.g., per-pair veto outcomes inside `link_tracklets_by_appearance`) can use
the optional `attach_callback` channel below: a stage module sets a
module-global callback when forensic is active and calls it for each internal
decision. The callback is None when forensic is inactive.
"""
from __future__ import annotations

import json
import logging
import os
import statistics
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rallycut.tracking.player_tracker import PlayerPosition

logger = logging.getLogger(__name__)

ENV_LOG_DIR = "PIPELINE_FORENSIC_LOG_DIR"
ENV_RALLY_TAG = "PIPELINE_FORENSIC_RALLY_TAG"

SCHEMA_VERSION = 1

# Stages we expect to emit at least one record per rally (or `SKIPPED` if the
# stage's gate isn't satisfied). Listed in execution order so the analyzer
# walks them left-to-right when computing first-divergence.
EXPECTED_STAGES: tuple[str, ...] = (
    "1_w4_split",
    "2_height_swap",
    "3_color_split",
    "4a_link_spatial",
    "4b_link_primary_frag",
    "4c_link_appearance",
    "5_candidate_filter",
    "6_top4_selection",
    "7_per_frame_filter",
    "8_team_classification",
    "9a_within_global_id",
    "9b_cross_hungarian",
    "9b_hungarian_costs",
    "10_drift_convergence",
)

# Module-level state — set lazily by `is_active()` on first call per rally.
_state: dict[str, Any] = {
    "active": None,  # None = unchecked, True/False = checked
    "sidecar_path": None,
    "rally_tag": None,
    "fp": None,
    "stages_emitted": set(),
}


def is_active() -> bool:
    """Return True iff both env vars are set. One dict lookup when off.

    Called at every stage call site. Lazily opens the sidecar on first
    True; subsequent calls reuse the open fp. Sidecar is closed by the
    `finalize()` call that capture drivers should make after the rally.
    """
    cached = _state["active"]
    if cached is not None:
        return bool(cached)

    log_dir = os.environ.get(ENV_LOG_DIR)
    rally_tag = os.environ.get(ENV_RALLY_TAG)
    if not log_dir or not rally_tag:
        _state["active"] = False
        return False

    # Open sidecar lazily.
    sidecar_path = Path(log_dir) / f"{rally_tag}.jsonl"
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    if sidecar_path.exists():
        sidecar_path.unlink()

    fp = open(sidecar_path, "w", buffering=1, encoding="utf-8")
    meta = {
        "type": "meta",
        "rally_tag": rally_tag,
        "schema_version": SCHEMA_VERSION,
        "expected_stages": list(EXPECTED_STAGES),
    }
    fp.write(json.dumps(meta) + "\n")

    _state["active"] = True
    _state["sidecar_path"] = sidecar_path
    _state["rally_tag"] = rally_tag
    _state["fp"] = fp
    _state["stages_emitted"] = set()
    logger.info("pipeline_forensic active; writing sidecar to %s", sidecar_path)
    return True


def finalize() -> None:
    """Mark stages that did NOT emit as `SKIPPED` and close the sidecar.

    Capture driver should call this after each rally completes (in the
    finally block of the per-rally retrack). Safe to call when inactive.
    """
    if _state["active"] is not True:
        return
    fp = _state["fp"]
    emitted: set[str] = _state["stages_emitted"]
    for stage in EXPECTED_STAGES:
        if stage not in emitted:
            try:
                fp.write(json.dumps({"stage": stage, "status": "SKIPPED"}) + "\n")
            except (ValueError, OSError):
                pass
    try:
        fp.close()
    except (ValueError, OSError):
        pass
    _state["active"] = None
    _state["sidecar_path"] = None
    _state["rally_tag"] = None
    _state["fp"] = None
    _state["stages_emitted"] = set()


def log_stage(stage: str, payload: dict[str, Any]) -> None:
    """Append one JSONL record. No-op when inactive.

    `stage` should be one of EXPECTED_STAGES. Records are tagged with stage
    so the analyzer can replay rally-end-to-end.
    """
    if not is_active():
        return
    record = {"stage": stage, **payload}
    fp = _state["fp"]
    try:
        fp.write(json.dumps(record, default=_default_json) + "\n")
    except (ValueError, OSError) as exc:
        logger.warning("pipeline_forensic failed to write %s: %s", stage, exc)
        return
    _state["stages_emitted"].add(stage)


def _default_json(obj: Any) -> Any:
    """Fallback serializer for numpy scalars / dataclass-like objects."""
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
    return repr(obj)


# ---------------------------------------------------------------------------
# Capture helpers — turn pipeline objects into compact JSON-able dicts
# ---------------------------------------------------------------------------

def positions_summary(
    positions: list[PlayerPosition],
    primary_track_ids: list[int] | None = None,
) -> dict[str, Any]:
    """Compact per-track summary of a positions list.

    Returns: {
      "n_positions": int,
      "n_tracks": int,
      "per_track": {tid: {"n": int, "f_min": int, "f_max": int,
                          "y_med": float, "h_med": float, "w_med": float}},
      "primary_track_ids": list[int] | None,
    }
    """
    by_tid: dict[int, list[PlayerPosition]] = defaultdict(list)
    for p in positions:
        by_tid[int(p.track_id)].append(p)

    per_track: dict[str, dict[str, Any]] = {}
    for tid, ps in by_tid.items():
        ys = [p.y for p in ps]
        hs = [p.height for p in ps]
        ws = [p.width for p in ps]
        frames = [p.frame_number for p in ps]
        per_track[str(tid)] = {
            "n": len(ps),
            "f_min": int(min(frames)),
            "f_max": int(max(frames)),
            "y_med": float(statistics.median(ys)),
            "h_med": float(statistics.median(hs)),
            "w_med": float(statistics.median(ws)),
        }

    return {
        "n_positions": len(positions),
        "n_tracks": len(by_tid),
        "per_track": per_track,
        "primary_track_ids": (
            [int(t) for t in primary_track_ids]
            if primary_track_ids is not None
            else None
        ),
    }


def track_id_set(positions: list[PlayerPosition]) -> list[int]:
    """Sorted list of unique track ids in `positions`."""
    return sorted({int(p.track_id) for p in positions})


def positions_diff(
    pre: list[PlayerPosition],
    post: list[PlayerPosition],
) -> dict[str, Any]:
    """Diff helper: which track ids appeared/disappeared/changed between
    pre and post.

    Returns: {
      "added": list[int],     # tids in post but not pre
      "removed": list[int],   # tids in pre but not post
      "frame_count_changed": dict[tid -> (pre_n, post_n)],  # tids where count differs
    }
    """
    pre_counts: dict[int, int] = defaultdict(int)
    for p in pre:
        pre_counts[int(p.track_id)] += 1
    post_counts: dict[int, int] = defaultdict(int)
    for p in post:
        post_counts[int(p.track_id)] += 1

    pre_ids = set(pre_counts.keys())
    post_ids = set(post_counts.keys())
    added = sorted(post_ids - pre_ids)
    removed = sorted(pre_ids - post_ids)
    changed: dict[str, list[int]] = {}
    for tid in pre_ids & post_ids:
        if pre_counts[tid] != post_counts[tid]:
            changed[str(tid)] = [pre_counts[tid], post_counts[tid]]

    return {"added": added, "removed": removed, "frame_count_changed": changed}


# ---------------------------------------------------------------------------
# Optional per-decision callbacks (for stages with coarse return shape)
# ---------------------------------------------------------------------------

# `link_tracklets_by_appearance` and `detect_convergence_swaps` make per-pair
# decisions that don't surface in the return value. Their modules expose
# module-level globals `_FORENSIC_CB` (or similar) that the call sites set
# when forensic is active. The callback signature is:
#   fn(stage: str, decision: dict)
# The stage module short-circuits all overhead when the callback is None.

_callbacks: dict[str, Callable[[str, dict[str, Any]], None] | None] = {}


def register_callback(
    name: str,
    fn: Callable[[str, dict[str, Any]], None] | None,
) -> None:
    """Register a forensic callback by name. Used by stage modules that
    expose a `set_forensic_callback(fn)` hook for per-decision logging.

    `name` is informational (e.g., "tracklet_link_appearance"). Pass `fn=None`
    to clear.
    """
    _callbacks[name] = fn


def emit_decision(stage: str, decision: dict[str, Any]) -> None:
    """Emit one per-decision record under `stage`. Called by stage modules
    via their forensic callback. No-op when inactive.
    """
    if not is_active():
        return
    log_stage(stage, {"kind": "decision", **decision})


def appearance_cost_breakdown(profile: Any, stats: Any) -> dict[str, float]:
    """Per-feature breakdown of `compute_appearance_similarity(profile, stats)`.

    Replicates the per-feature contribution calculation inline (vs. modifying
    the production function signature). Returns a dict mapping each feature
    key to its weighted contribution to the total HSV cost. Sum of values
    ≈ the scalar HSV cost returned by `compute_appearance_similarity`.
    """
    from rallycut.tracking import player_features as _pf_mod

    scores: list[tuple[str, float, float]] = []
    sim = _pf_mod._histogram_similarity(profile.avg_lower_hist, stats.avg_lower_hist)
    if sim is not None:
        scores.append(("lower_hist", _pf_mod._WEIGHT_LOWER_HIST, sim))
    sim = _pf_mod._histogram_similarity(profile.avg_lower_v_hist, stats.avg_lower_v_hist)
    if sim is not None:
        scores.append(("lower_v_hist", _pf_mod._WEIGHT_LOWER_V_HIST, sim))
    sim = _pf_mod._histogram_similarity(profile.avg_upper_hist, stats.avg_upper_hist)
    if sim is not None:
        scores.append(("upper_hist", _pf_mod._WEIGHT_UPPER_HIST, sim))
    sim = _pf_mod._histogram_similarity(profile.avg_upper_v_hist, stats.avg_upper_v_hist)
    if sim is not None:
        scores.append(("upper_v_hist", _pf_mod._WEIGHT_UPPER_V_HIST, sim))
    if profile.avg_skin_tone_hsv is not None and stats.avg_skin_tone_hsv is not None:
        sim = _pf_mod._hsv_similarity(profile.avg_skin_tone_hsv, stats.avg_skin_tone_hsv)
        scores.append(("skin", _pf_mod._WEIGHT_SKIN, sim))
    if (
        profile.avg_dominant_color_hsv is not None
        and stats.avg_dominant_color_hsv is not None
    ):
        sim = _pf_mod._hsv_similarity(
            profile.avg_dominant_color_hsv, stats.avg_dominant_color_hsv,
        )
        scores.append(("dominant_color", _pf_mod._WEIGHT_DOMINANT_COLOR, sim))
    sim = _pf_mod._histogram_similarity(profile.avg_head_hist, stats.avg_head_hist)
    if sim is not None:
        scores.append(("head_hist", _pf_mod._WEIGHT_HEAD_HIST, sim))

    if not scores:
        return {}
    total_w = sum(w for _, w, _ in scores)
    return {key: float(w * (1.0 - s) / total_w) for key, w, s in scores}
