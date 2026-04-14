"""Canonical production-mirroring eval dashboard.

This script is the **one honest measurement instrument** for the RallyCut
`track-players --actions` action-classification path. It mirrors production
exactly by reusing the same shared helpers production imports — it does NOT
re-implement any pipeline stage. See
`memory/production_truth_2026_04.md` for the full production spec.

What it does
------------
For each of the ~339 GT rallies in the DB, it loads pre-computed tracked
ball/player JSON and runs stages 9–14 of `track_player.py` end-to-end:

    1. `verify_team_assignments`         (match_tracker.py)
    2. `get_sequence_probs`              (sequence_action_runtime.py)   [MS-TCN++]
    3. pose injection from pose_cache    (pre-stored — equivalent to the
                                          production two-pass enrichment's
                                          end state)
    4. `detect_contacts`                 (contact_detector.py)
    5. `classify_rally_actions`          (action_classifier.py)
    6. `apply_sequence_override`         (sequence_action_runtime.py)

Then it computes honest aggregate metrics with rerun variance bars and
reports any rallies production couldn't process in a `production_rejected`
list (we never silently skip).

Usage
-----
    cd analysis
    uv run python scripts/production_eval.py                     # full run, N=3 reruns
    uv run python scripts/production_eval.py --reruns 5
    uv run python scripts/production_eval.py --parity-check <rid>  # per-contact diff vs DB
    uv run python scripts/production_eval.py --ablate mstcn_override

Ablation registry
-----------------
See `ABLATIONS` below. To add a new ablation in <10 lines:

    1. Add a `PipelineContext` field (e.g. `skip_pose_enrich: bool = False`).
    2. Gate the relevant stage in `_run_rally()` on the field.
    3. Add `"pose_enrich": lambda ctx: setattr(ctx, "skip_pose_enrich", True)`
       to `ABLATIONS`.

Output JSON schema
------------------
Written to `analysis/outputs/production_eval/run_<YYYY-MM-DD-HHMMSS>.json`:

    {
      "run_id":             "2026-04-07-142310",
      "git_sha":            "abc1234",
      "n_reruns":           3,
      "ablations_active":   ["mstcn_override"],
      "weights": {
        "<name>": {"path": "...", "size_bytes": N, "sha256_prefix": "..."}
      },
      "n_rallies_loaded":   339,
      "n_rallies_evaluated": 331,
      "production_rejected": [
        {"rally_id": "...", "reason": "missing ball_positions_json"},
        {"rally_id": "...", "reason": "RuntimeError: ..."}
      ],
      "metrics": {
        "contact_f1":                  {"mean": 0.817, "std": 0.003, "values": [...]},
        "contact_recall":              {...},
        "contact_precision":           {...},
        "action_accuracy":             {...},
        "court_side_accuracy":         {...},
        "player_attribution_accuracy": {...},
        "serve_id_accuracy":           {...},
        "serve_attr_accuracy":         {...},
        "action_accuracy_per_class":   {"serve": {"mean": ..., "std": ...}, ...}
      },
      "variance_warning": null | "metric X std dev 1.3pp > 1pp — consider --reruns 5"
    }

Phase 3 (component contribution audit) consumes this schema mechanically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table

# Ensure sibling scripts (eval_action_detection.py) are importable.
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from eval_action_detection import (  # noqa: E402
    MatchResult,
    RallyData,
    _build_player_positions,
    _load_match_team_assignments,
    _load_track_to_player_maps,
    compute_metrics,
    load_rallies_with_action_gt,
    match_contacts,
)

from rallycut.evaluation.score_ground_truth import (  # noqa: E402
    ScoreMetrics,
    compute_score_metrics,
)
from rallycut.evaluation.tracking.db import get_connection  # noqa: E402
from rallycut.scoring.cross_rally_viterbi import (  # noqa: E402
    RallyObservation,
    RallyPositionData,
    decode_video_dual_hypothesis,
    detect_side_switches_from_positions,
)
from rallycut.tracking.action_classifier import (  # noqa: E402
    _find_serving_side_by_formation,
    classify_rally_actions,
)
from rallycut.tracking.ball_tracker import BallPosition  # noqa: E402
from rallycut.tracking.contact_detector import (  # noqa: E402
    Contact,
    ContactDetectionConfig,
    detect_contacts,
)
from rallycut.tracking.match_tracker import verify_team_assignments  # noqa: E402
from rallycut.tracking.player_tracker import PlayerPosition  # noqa: E402
from rallycut.tracking.sequence_action_runtime import (  # noqa: E402
    apply_sequence_override,
    get_sequence_probs,
)

console = Console()


# --------------------------------------------------------------------------- #
# Weights inventory — loud abort if MS-TCN++ weights missing (per spec)       #
# --------------------------------------------------------------------------- #

WEIGHTS_PATHS: dict[str, Path] = {
    "contact_classifier": Path("weights/contact_classifier/contact_classifier.pkl"),
    "pose_attribution":   Path("weights/pose_attribution/pose_attribution.joblib"),
    "action_classifier":  Path("weights/action_classifier/action_classifier.pkl"),
    "sequence_action":    Path("weights/sequence_action/ms_tcn_production.pt"),
}


def _weights_inventory(analysis_root: Path) -> dict[str, dict[str, Any]]:
    """Stat every production weight file; fail loudly if MS-TCN++ is missing.

    Returns a dict keyed by weight-group name. Each entry has `path`,
    `size_bytes`, and `sha256_prefix` (first 16 hex chars — enough to prove
    which file was loaded, cheap to compute).
    """
    inventory: dict[str, dict[str, Any]] = {}
    for name, rel in WEIGHTS_PATHS.items():
        abs_path = (analysis_root / rel).resolve()
        if not abs_path.exists():
            if name == "sequence_action":
                raise RuntimeError(
                    f"MS-TCN++ weights missing at {abs_path}. "
                    "Production path loads these unconditionally and the spec "
                    "forbids silent fallback. Run "
                    "`uv run rallycut train pull-weights` or retrain before "
                    "running this eval."
                )
            console.print(
                f"[yellow]warning:[/yellow] weight file missing: {abs_path} "
                f"(production would auto-load this; downstream stage may degrade)"
            )
            inventory[name] = {"path": str(abs_path), "size_bytes": 0, "sha256_prefix": ""}
            continue
        size = abs_path.stat().st_size
        h = hashlib.sha256()
        with abs_path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        inventory[name] = {
            "path": str(abs_path),
            "size_bytes": size,
            "sha256_prefix": h.hexdigest()[:16],
        }
    return inventory


# --------------------------------------------------------------------------- #
# Pipeline context + ablation registry                                        #
# --------------------------------------------------------------------------- #


@dataclass
class PipelineContext:
    """Runtime toggles consumed by `_run_rally`. Ablations mutate this."""

    skip_sequence_override: bool = False
    skip_verify_teams: bool = False
    skip_pose_attribution: bool = False
    skip_contact_classifier: bool = False
    skip_adaptive_dedup: bool = False
    skip_seq_enriched_contact_gbm: bool = False
    skip_sequence_recovery: bool = False
    # Session 11: when True, eval reverts to pre-fix literal-ID comparison
    # between gt.player_track_id and pred.playerTrackId (no trackId
    # normalization via match_analysis.trackToPlayer). Used to reproduce
    # the pre-Session-11 baseline bit-exactly.
    skip_trackid_normalization: bool = False
    skip_viterbi_scoring: bool = False


# Each ablation is a one-line mutator. Extension pattern is documented in the
# module docstring. Keep these under 10 lines apiece.
ABLATIONS: dict[str, Callable[[PipelineContext], None]] = {
    "mstcn_override":           lambda ctx: setattr(ctx, "skip_sequence_override", True),
    "verify_team_assignments":  lambda ctx: setattr(ctx, "skip_verify_teams", True),
    "pose_attribution":         lambda ctx: setattr(ctx, "skip_pose_attribution", True),
    "contact_classifier":       lambda ctx: setattr(ctx, "skip_contact_classifier", True),
    "adaptive_dedup":           lambda ctx: setattr(ctx, "skip_adaptive_dedup", True),
    "seq_enriched_contact_gbm": lambda ctx: setattr(ctx, "skip_seq_enriched_contact_gbm", True),
    "sequence_recovery":        lambda ctx: setattr(ctx, "skip_sequence_recovery", True),
    "literal_id_match":         lambda ctx: setattr(ctx, "skip_trackid_normalization", True),
    "viterbi_scoring":          lambda ctx: setattr(ctx, "skip_viterbi_scoring", True),
}


# --------------------------------------------------------------------------- #
# Court calibrators (match production enrichment)                             #
# --------------------------------------------------------------------------- #


def _build_calibrators(video_ids: set[str]) -> dict[str, Any]:
    """Build per-video `CourtCalibrator` from DB-stored court keypoints.

    Mirrors how `track_player.py` passes a real `CourtCalibrator` into
    `get_sequence_probs`, `detect_contacts`, and `classify_rally_actions`.
    Videos without calibration yield no entry (helpers treat `None` as
    "no homography enrichment", matching production's uncalibrated path).
    """
    from rallycut.court.calibration import CourtCalibrator  # noqa: PLC0415
    from rallycut.evaluation.tracking.db import load_court_calibration  # noqa: PLC0415

    out: dict[str, Any] = {}
    for vid in video_ids:
        corners = load_court_calibration(vid)
        if not corners or len(corners) != 4:
            continue
        cal = CourtCalibrator()
        cal.calibrate([(c["x"], c["y"]) for c in corners])
        if cal.is_calibrated:
            out[vid] = cal
    return out


def _build_camera_heights(
    video_ids: set[str],
    calibrators: dict[str, Any],
) -> dict[str, float]:
    """Compute per-video camera height from court calibration + video resolution."""
    from rallycut.court.camera_model import calibrate_camera  # noqa: PLC0415

    heights: dict[str, float] = {}
    if not video_ids:
        return heights
    placeholders = ", ".join(["%s"] * len(video_ids))
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"SELECT id, width, height FROM videos WHERE id IN ({placeholders})",
            list(video_ids),
        )
        resolutions = {vid: (int(w), int(h)) for vid, w, h in cur.fetchall() if w and h}
    for vid, cal in calibrators.items():
        res = resolutions.get(vid)
        if res is None or not cal.is_calibrated or cal.homography is None:
            continue
        cam = calibrate_camera(
            cal.homography.image_corners,
            cal.homography.court_corners,
            res[0], res[1],
        )
        if cam is not None and cam.is_valid:
            heights[vid] = float(cam.camera_position[2])
    return heights


def _load_team_templates_by_video(
    video_ids: set[str],
) -> dict[str, tuple[Any, Any]]:
    """Load team templates and player profiles from match_analysis_json.

    Returns {video_id: (template_0, template_1)} for videos that have
    teamTemplates in their match_analysis_json.
    """
    from rallycut.tracking.team_identity import TeamTemplate  # noqa: PLC0415

    if not video_ids:
        return {}
    placeholders = ", ".join(["%s"] * len(video_ids))
    query = f"""
        SELECT id, match_analysis_json
        FROM videos
        WHERE id IN ({placeholders})
          AND match_analysis_json IS NOT NULL
    """
    result: dict[str, tuple[Any, Any]] = {}
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(query, list(video_ids))
        rows = cur.fetchall()
    for video_id, ma_json in rows:
        if not isinstance(ma_json, dict):
            continue
        templates_data = ma_json.get("teamTemplates")
        if not templates_data or not isinstance(templates_data, dict):
            continue
        t0_data = templates_data.get("0")
        t1_data = templates_data.get("1")
        if not t0_data or not t1_data:
            continue
        t0 = TeamTemplate.from_dict(t0_data)
        t1 = TeamTemplate.from_dict(t1_data)
        result[video_id] = (t0, t1)
    return result


def _load_formation_semantic_flips(video_ids: set[str]) -> dict[str, bool]:
    """Compute per-rally `semantic_flip` from `match_analysis_json`.

    Reads each video's `sideSwitchDetected` flags in rally order and
    returns `{rally_id: flipped}` where `flipped=True` when the
    cumulative side-switch count BEFORE this rally is odd.

    Used by the formation-based serving_team predictor to map the
    physical-near team (team_assignments' team 0) to the correct semantic
    team on flipped rallies. +8.7pp on the 92-rally action_GT subset vs
    no flip (see score_tracking_architecture_2026_04.md).

    Falls back to all-False when match_analysis_json is missing.
    """
    if not video_ids:
        return {}
    placeholders = ", ".join(["%s"] * len(video_ids))
    query = f"""
        SELECT id, match_analysis_json
        FROM videos
        WHERE id IN ({placeholders})
          AND match_analysis_json IS NOT NULL
    """
    result: dict[str, bool] = {}
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(query, list(video_ids))
        rows = cur.fetchall()
    for _video_id, ma_json in rows:
        if not isinstance(ma_json, dict):
            continue
        rally_entries = ma_json.get("rallies") or []
        if not isinstance(rally_entries, list):
            continue
        count = 0
        for rally_entry in rally_entries:
            rid = rally_entry.get("rallyId") or rally_entry.get("rally_id")
            if rid:
                result[rid] = (count % 2 == 1)
            if rally_entry.get("sideSwitchDetected") or rally_entry.get(
                "side_switch_detected"
            ):
                count += 1
    return result


def _load_formation_semantic_flips_from_gt(
    video_ids: set[str],
) -> dict[str, bool]:
    """Compute per-rally ``semantic_flip`` from GT switches.

    Delegates to the shared ``resolve_side_flipped`` loader which merges
    pipeline-detected switches with per-rally manual overrides from the
    Score GT UI.  Uses ``gt_only=False`` so the flip chain spans all
    rallies in the video (not just GT-labeled ones).

    Falls back to pipeline-derived flips for videos without any switch data.
    """
    if not video_ids:
        return {}

    from rallycut.evaluation.switch_loader import resolve_side_flipped

    result = resolve_side_flipped(video_ids, gt_only=False)

    # Determine which videos are covered (have at least one rally resolved)
    covered_videos: set[str] = set()
    with get_connection() as conn, conn.cursor() as cur:
        if result:
            placeholders = ", ".join(["%s"] * len(result))
            cur.execute(f"""
                SELECT DISTINCT video_id FROM rallies
                WHERE id IN ({placeholders})
            """, list(result.keys()))
            covered_videos = {row[0] for row in cur.fetchall()}

    uncovered = video_ids - covered_videos
    if uncovered:
        result.update(_load_formation_semantic_flips(uncovered))

    return result


# --------------------------------------------------------------------------- #
# Per-rally production-mirrored pipeline                                      #
# --------------------------------------------------------------------------- #


@dataclass
class RallyOutcome:
    """Result of running the production path on one rally."""

    rally_id: str
    matches: list[MatchResult] = field(default_factory=list)
    unmatched_preds: list[dict] = field(default_factory=list)
    pred_actions: list[dict] = field(default_factory=list)  # kept for parity check


def _parse_positions(raw: list[dict]) -> list[PlayerPosition]:
    return [
        PlayerPosition(
            frame_number=pp["frameNumber"],
            track_id=pp["trackId"],
            x=pp["x"],
            y=pp["y"],
            width=pp.get("width", 0.05),
            height=pp.get("height", 0.10),
            confidence=pp.get("confidence", 1.0),
            keypoints=pp.get("keypoints"),
        )
        for pp in raw
    ]


def _parse_ball(raw: list[dict]) -> list[BallPosition]:
    return [
        BallPosition(
            frame_number=bp["frameNumber"],
            x=bp["x"],
            y=bp["y"],
            confidence=bp.get("confidence", 1.0),
        )
        for bp in raw
        if bp.get("x", 0) > 0 or bp.get("y", 0) > 0
    ]


def _run_rally(
    rally: RallyData,
    match_teams: dict[int, int] | None,
    calibrator: Any,
    ctx: PipelineContext,
    track_to_player: dict[int, int] | None = None,
    formation_semantic_flip: bool = False,
    camera_height: float = 0.0,
) -> tuple[list[dict], Any]:
    """Mirror `track_player.py:1011–1093` stages 9–14.

    Returns (pred_action_dicts, rally_actions). The structured `RallyActions`
    is kept so downstream score-metric code can chain rallies into
    `compute_match_scores` without re-parsing dicts.

    Any exception is raised to the caller, which records it in
    `production_rejected`. The caller also handles the "missing inputs"
    rejection gate before we get here.
    """
    ball_positions = _parse_ball(rally.ball_positions_json or [])
    # `inject_pose=True` reads `pose_cache` for this rally — equivalent end
    # state to production's two-pass `enrich_positions_with_pose`, which
    # mutates keypoints into `player_positions`. The cache is what the
    # production pipeline wrote during the track that populated positions_json.
    player_positions: list[PlayerPosition] = _build_player_positions(
        rally.positions_json or [], rally_id=rally.rally_id, inject_pose=True,
    )

    # Stage 9 — verify team assignments (per-rally bulk flip).
    teams: dict[int, int] | None = dict(match_teams) if match_teams else None
    if teams is not None and not ctx.skip_verify_teams:
        teams = verify_team_assignments(teams, player_positions)

    # Stage 10 — MS-TCN++ per-frame action probabilities. Calibrator is
    # forwarded so trajectory features get the homography-enriched signal
    # that production uses when court calibration is available
    # (track_player.py passes the verified CourtCalibrator).
    sequence_probs: np.ndarray | None = get_sequence_probs(
        ball_positions=ball_positions,
        player_positions=player_positions,
        court_split_y=rally.court_split_y,
        frame_count=rally.frame_count,
        team_assignments=teams,
        calibrator=calibrator,
    )

    # Stages 11+12 — pose-informed contact detection. Pose keypoints are
    # already injected via pose_cache above; production's default
    # ContactDetectionConfig is used implicitly unless an ablation toggles
    # a flag on it. When any contact-stage toggle is active we materialize
    # a ContactDetectionConfig() (same defaults) and mutate the specific
    # fields, so the non-ablated flags remain production-identical.
    cfg: ContactDetectionConfig | None = None
    use_classifier = True
    if (ctx.skip_pose_attribution or ctx.skip_adaptive_dedup
            or ctx.skip_contact_classifier):
        cfg = ContactDetectionConfig()
        if ctx.skip_pose_attribution:
            cfg.use_pose_attribution = False
        if ctx.skip_adaptive_dedup:
            cfg.adaptive_dedup = False
        if ctx.skip_contact_classifier:
            use_classifier = False

    # Sequence recovery is integrated into `detect_contacts` itself: the
    # ContactDetectionConfig flag `enable_sequence_recovery` gates whether
    # the two-signal agreement rescue fires. The ablation zeros the config
    # flag on a fresh cfg so the ablated run mirrors production minus the
    # one stage.
    if ctx.skip_sequence_recovery:
        if cfg is None:
            cfg = ContactDetectionConfig()
        cfg.enable_sequence_recovery = False

    contact_sequence = detect_contacts(
        ball_positions=ball_positions,
        player_positions=player_positions,
        config=cfg,
        use_classifier=use_classifier,
        frame_count=rally.frame_count or None,
        team_assignments=teams,
        court_calibrator=calibrator,
        sequence_probs=None if ctx.skip_seq_enriched_contact_gbm else sequence_probs,
    )

    # Stage 13 — action classification (state machine + GBM + repair + etc.).
    rally_actions = classify_rally_actions(
        contact_sequence,
        rally.rally_id,
        team_assignments=teams,
        match_team_assignments=teams,
        calibrator=calibrator,
        track_to_player=track_to_player,
        formation_semantic_flip=formation_semantic_flip,
        camera_height=camera_height,
    )

    # Stage 14 — MS-TCN++ hybrid override (serves exempt).
    if sequence_probs is not None and not ctx.skip_sequence_override:
        apply_sequence_override(rally_actions, sequence_probs)

    return [a.to_dict() for a in rally_actions.actions], rally_actions


# --------------------------------------------------------------------------- #
# Eval loop                                                                   #
# --------------------------------------------------------------------------- #


def _tolerance_frames(fps: float, tolerance_ms: int = 167) -> int:
    return max(1, round(fps * tolerance_ms / 1000))


def _rally_permutation_oracle(
    rally_matches: list[MatchResult],
) -> tuple[int, int, dict[int, int]]:
    """Compute the best within-rally permutation of pred→GT player IDs.

    For each rally we observe a set of ``(gt.player_track_id, pred_tid)``
    pairs. The literal ID compare in ``match_contacts`` assumes GT's canonical
    numbering matches the current tracker's canonical numbering — but that
    assumption breaks whenever a retrack rebases trackIds, because GT's
    integer labels are frozen against an older track run.

    This oracle asks a retrack-stable question instead: "within this rally,
    is there ANY consistent relabeling of predicted player IDs that aligns
    with GT?". We build a cost matrix ``cost[g, p] = -count(gt=g ∧ pred=p)``
    over evaluable matched contacts, solve a Hungarian assignment, and
    recount ``player_correct`` under that permutation. This measures the
    upper bound of within-rally attribution consistency — i.e., how well
    the pred attribution layer did, *modulo* GT's labeling convention.

    Canonical as of 2026-04-10: this IS the production player_attribution
    metric. The literal-compare ``player_attribution_accuracy`` is retained
    in the summary table as a drift-sensitive diagnostic only — do not use
    it for go/no-go decisions. Rationale: the literal compare is not
    retrack-stable (any rebase of tracker integer IDs silently shifts the
    number), so it cannot be trusted as a go-forward metric without a
    permutation-invariant wrapper like this one.

    This implementation only fixes the *measurement* side of the drift
    problem. The underlying tracker/GT integer convention is still brittle;
    the production architectural fix is DINOv2 user-crop canonical-identity
    anchoring in ``match_tracker.py`` (Workstream W0). Until W0 ships, the
    oracle metric prevents retrack accidents from contaminating go/no-go
    decisions.
    """
    evaluable = [
        m for m in rally_matches
        if m.player_evaluable and m.pred_frame is not None and m.gt_action != "block"
    ]
    if not evaluable:
        return 0, 0, {}

    # Collect unique GT and pred IDs observed in this rally.
    gt_ids: list[int] = []
    pred_ids: list[int] = []
    # MatchResult doesn't carry the raw tids — we reconstruct via a
    # caller-provided list. But we only have the booleans here, so use a
    # side-channel attached to each MatchResult by _run_once below.
    for m in evaluable:
        gt_tid = getattr(m, "_gt_tid", None)
        pred_tid = getattr(m, "_pred_tid", None)
        if gt_tid is None or pred_tid is None:
            continue
        if gt_tid not in gt_ids:
            gt_ids.append(gt_tid)
        if pred_tid not in pred_ids:
            pred_ids.append(pred_tid)

    if not gt_ids or not pred_ids:
        return 0, len(evaluable), {}

    size = max(len(gt_ids), len(pred_ids))
    cost = np.zeros((size, size), dtype=np.float64)
    for m in evaluable:
        gt_tid = getattr(m, "_gt_tid", None)
        pred_tid = getattr(m, "_pred_tid", None)
        if gt_tid is None or pred_tid is None:
            continue
        g_idx = gt_ids.index(gt_tid)
        p_idx = pred_ids.index(pred_tid)
        cost[g_idx, p_idx] -= 1.0  # maximize agreement → minimize negative count

    from scipy.optimize import linear_sum_assignment  # noqa: PLC0415
    row_ind, col_ind = linear_sum_assignment(cost)

    # Build the permutation: gt_id → pred_id that it's assigned to
    permutation: dict[int, int] = {}
    for r, c in zip(row_ind, col_ind):
        if r < len(gt_ids) and c < len(pred_ids):
            permutation[gt_ids[r]] = pred_ids[c]

    # Recount under the permutation.
    correct = 0
    for m in evaluable:
        gt_tid = getattr(m, "_gt_tid", None)
        pred_tid = getattr(m, "_pred_tid", None)
        if gt_tid is None or pred_tid is None:
            continue
        if permutation.get(gt_tid) == pred_tid:
            correct += 1
    return correct, len(evaluable), permutation


def _apply_viterbi_scoring(
    rallies: list[RallyData],
    pred_by_video: dict[str, list[tuple[int, Any]]],
    formation_flip_by_rally: dict[str, bool] | None,
    t2p_by_rally: dict[str, dict[int, int]] | None = None,
    team_templates_by_video: dict[str, tuple[Any, Any]] | None = None,
    calibrators: dict[str, Any] | None = None,
) -> None:
    """Post-process per-rally serving_team via dual-hypothesis Viterbi.

    Groups rallies by video, extracts physical side from formation,
    detects side switches from player positions (when track_to_player
    is available, falls back to match_analysis semantic_flip), runs
    ``decode_video_dual_hypothesis`` (tries both near=A and near=B),
    and overrides ``formation_serving_team`` on each ``RallyActions``.

    When team templates are available, adds per-rally team localization
    (which team is near/far) to break the dual-hypothesis tie using
    appearance consistency instead of the symmetric plausibility scorer.

    Production-viable: +10pp over baseline with zero user input.

    Modifies ``pred_by_video`` in place.
    """
    rally_data_by_id: dict[str, RallyData] = {r.rally_id: r for r in rallies}
    flips = formation_flip_by_rally or {}
    t2p = t2p_by_rally or {}
    templates_by_vid = team_templates_by_video or {}

    for video_id, rally_order in pred_by_video.items():
        rally_order.sort(key=lambda x: x[0])

        observations: list[RallyObservation] = []
        position_data: list[RallyPositionData] = []
        rally_actions_list: list[Any] = []
        team_near_labels: list[str | None] = []

        # Get team templates for this video (if available)
        video_templates = templates_by_vid.get(video_id)

        for order_i, (_start_ms, rally_actions) in enumerate(rally_order):
            rid = rally_actions.rally_id
            rd = rally_data_by_id.get(rid)
            rally_actions_list.append(rally_actions)

            # Formation physical side (bypasses team_assignments).
            formation_side: str | None = None
            formation_conf = 0.0
            positions: list[PlayerPosition] = []
            net_y = 0.5
            if rd and rd.positions_json:
                positions = _parse_positions(rd.positions_json)
                net_y = rd.court_split_y if rd.court_split_y else 0.5
                ball_pos = _parse_ball(rd.ball_positions_json or [])
                vid_cal = (calibrators or {}).get(video_id)
                # Parse first contact for adaptive window + fusion
                fc_frame: int | None = None
                first_contact_obj: Contact | None = None
                if rd.contacts_json and isinstance(rd.contacts_json, dict):
                    contacts_list = rd.contacts_json.get("contacts", [])
                    if contacts_list:
                        c = contacts_list[0]
                        fc_frame = c.get("frame")
                        first_contact_obj = Contact(
                            frame=c.get("frame", 0),
                            ball_x=c.get("ballX", 0.5),
                            ball_y=c.get("ballY", 0.5),
                            velocity=c.get("velocity", 0.0),
                            direction_change_deg=c.get("directionChangeDeg", 0.0),
                            player_track_id=c.get("playerTrackId", -1),
                            player_distance=c.get("playerDistance", float("inf")),
                            is_at_net=c.get("isAtNet", False),
                            court_side=c.get("courtSide", "unknown"),
                        )
                formation_side, formation_conf = _find_serving_side_by_formation(
                    positions, net_y=net_y, start_frame=0,
                    ball_positions=ball_pos or None,
                    calibrator=vid_cal,
                    first_contact_frame=fc_frame,
                    adaptive_window=True,
                    first_contact=first_contact_obj,
                )

            # Team localization: which team template is near?
            # Fallback chain: localize_team_near (Y-position) → GT semantic
            # flip (when available) → None (triggers Viterbi fallback).
            # localize_team_near returns None for narrow-angle cameras where
            # the Y gap is too small to trust. GT flips (from human-labeled
            # sideSwitches) fill that gap in evaluation contexts.
            team_near: str | None = None
            if video_templates is not None:
                from rallycut.tracking.team_identity import localize_team_near  # noqa: PLC0415
                rally_t2p = t2p.get(rid, {})
                team_near = localize_team_near(positions, rally_t2p, video_templates)
                if team_near is None and rid in flips:
                    team_near = "1" if flips[rid] else "0"
            team_near_labels.append(team_near)

            observations.append(RallyObservation(
                rally_id=rid,
                formation_side=formation_side,
                formation_confidence=formation_conf,
            ))

            position_data.append(RallyPositionData(
                positions=positions,
                track_to_player=t2p.get(rid, {}),
                court_split_y=rd.court_split_y if rd else None,
            ))

        # Per-rally team localization: directly determine serving team
        # from formation (which side serves) + track_to_player (which
        # team is on which side). No accumulated side-switch state needed.
        has_team_loc = any(tn is not None for tn in team_near_labels)

        if has_team_loc and video_templates is not None:
            from rallycut.tracking.team_identity import (  # noqa: PLC0415
                calibrate_convention_from_gt,
                resolve_serving_team,
            )

            # Convention: which template label = "A"?
            gt_teams: list[str | None] = []
            for ra in rally_actions_list:
                rd = rally_data_by_id.get(ra.rally_id)
                gt_teams.append(rd.gt_serving_team if rd else None)
            has_gt = any(gt is not None for gt in gt_teams)

            if has_gt:
                formation_sides = [obs.formation_side for obs in observations]
                label_a = calibrate_convention_from_gt(
                    gt_teams, formation_sides, team_near_labels, video_templates,
                )
            else:
                label_a = next(
                    (tn for tn in team_near_labels if tn is not None), "0",
                )

            for obs, ra, tn in zip(observations, rally_actions_list, team_near_labels):
                team = resolve_serving_team(
                    obs.formation_side, tn, video_templates, label_a,
                )
                if team is not None:
                    ra.formation_serving_team = team
        else:
            # Fallback: no team localization → use Viterbi with side switches
            has_t2p = any(pd.track_to_player for pd in position_data)
            if has_t2p:
                switch_indices = detect_side_switches_from_positions(position_data)
            else:
                switch_indices = set()
                for order_i in range(1, len(rally_actions_list)):
                    rid = rally_actions_list[order_i].rally_id
                    prev_rid = rally_actions_list[order_i - 1].rally_id
                    cur_flip = flips.get(rid, False)
                    prev_flip = flips.get(prev_rid, False)
                    if cur_flip != prev_flip:
                        switch_indices.add(order_i)

            decoded = decode_video_dual_hypothesis(
                observations, side_switch_rallies=switch_indices,
            )
            for ra, dec in zip(rally_actions_list, decoded):
                ra.formation_serving_team = dec.serving_team


def _run_once(
    rallies: list[RallyData],
    team_map: dict[str, dict[int, int]],
    calibrators: dict[str, Any],
    ctx: PipelineContext,
    t2p_by_rally: dict[str, dict[int, int]] | None = None,
    formation_flip_by_rally: dict[str, bool] | None = None,
    camera_heights: dict[str, float] | None = None,
    team_templates_by_video: dict[str, tuple[Any, Any]] | None = None,
    *,
    print_progress: bool = True,
) -> tuple[
    list[MatchResult],
    list[dict],
    list[dict[str, str]],
    dict[str, list[tuple[int, Any]]],
    dict[str, tuple[str | None, str | None]],
    tuple[int, int],
    tuple[int, int],
]:
    """Run the full 339-rally loop once.

    Returns (matches, unmatched_preds, rejections, pred_by_video, gt_lookup,
    oracle_counts, serve_oracle_counts). The two ``*_oracle_counts`` tuples
    are ``(correct, total)`` accumulators feeding the within-rally
    permutation-invariant canonical metrics ``player_attribution_oracle`` and
    ``serve_attr_oracle`` — see ``_rally_permutation_oracle`` for rationale.
    """
    all_matches: list[MatchResult] = []
    all_unmatched: list[dict] = []
    rejections: list[dict[str, str]] = []
    pred_by_video: dict[str, list[tuple[int, Any]]] = {}
    gt_lookup: dict[str, tuple[str | None, str | None]] = {}
    oracle_correct_total = 0
    oracle_evaluable_total = 0
    serve_oracle_correct_total = 0
    serve_oracle_evaluable_total = 0

    for idx, rally in enumerate(rallies, start=1):
        # Rejection gate — surface rather than silently skip.
        if not rally.ball_positions_json:
            rejections.append({"rally_id": rally.rally_id, "reason": "missing ball_positions_json"})
            continue
        if not rally.positions_json:
            rejections.append({"rally_id": rally.rally_id, "reason": "missing positions_json"})
            continue
        if not rally.frame_count or rally.frame_count < 10:
            rejections.append({
                "rally_id": rally.rally_id,
                "reason": f"frame_count={rally.frame_count}",
            })
            continue

        try:
            rally_t2p_for_formation = None
            if t2p_by_rally is not None:
                rally_t2p_for_formation = t2p_by_rally.get(rally.rally_id)
            rally_semantic_flip = False
            if formation_flip_by_rally is not None:
                rally_semantic_flip = formation_flip_by_rally.get(
                    rally.rally_id, False
                )
            pred_actions, rally_actions_obj = _run_rally(
                rally,
                team_map.get(rally.rally_id),
                calibrators.get(rally.video_id),
                ctx,
                track_to_player=rally_t2p_for_formation,
                formation_semantic_flip=rally_semantic_flip,
                camera_height=(camera_heights or {}).get(rally.video_id, 0.0),
            )
        except Exception as exc:  # noqa: BLE001 — we want to surface any prod failure
            rejections.append({
                "rally_id": rally.rally_id,
                "reason": f"{type(exc).__name__}: {exc}",
            })
            if print_progress:
                console.print(
                    f"  [red][{idx}/{len(rallies)}][/red] {rally.rally_id}: "
                    f"REJECTED — {type(exc).__name__}: {exc}"
                )
            continue

        real_pred = [a for a in pred_actions if not a.get("isSynthetic")]
        raw_avail_tids = {pp["trackId"] for pp in rally.positions_json}
        match_teams = team_map.get(rally.rally_id)

        # Session 11: normalize raw trackIds → canonical player IDs (1-4)
        # via match_analysis.trackToPlayer so the literal ID compare in
        # match_contacts becomes robust to trackId rebases. The ablation
        # flag `literal_id_match` reverts to the pre-fix path bit-exactly.
        rally_t2p: dict[int, int] | None = None
        if t2p_by_rally is not None and not ctx.skip_trackid_normalization:
            rally_t2p = t2p_by_rally.get(rally.rally_id) or None
        if rally_t2p:
            avail_tids = {rally_t2p.get(tid, tid) for tid in raw_avail_tids}
        else:
            avail_tids = raw_avail_tids

        matches, unmatched = match_contacts(
            rally.gt_labels,
            real_pred,
            tolerance=_tolerance_frames(rally.fps),
            available_track_ids=avail_tids,
            team_assignments=match_teams,
            track_id_map=rally_t2p,
        )

        # Attach raw (gt_tid, pred_tid) side-channel to each match for the
        # within-rally permutation oracle. match_contacts doesn't carry these
        # on MatchResult, so we recover them here by looking up the originals
        # on gt_labels / real_pred keyed on (frame, action).
        gt_by_frame_action: dict[tuple[int, str], int] = {
            (gl.frame, gl.action): gl.player_track_id for gl in rally.gt_labels
        }
        pred_by_frame_action: dict[tuple[int, str], int] = {
            (a.get("frame", -1), a.get("action", "")): a.get("playerTrackId", -1)
            for a in real_pred
        }
        for m in matches:
            gt_tid = gt_by_frame_action.get((m.gt_frame, m.gt_action))
            m._gt_tid = gt_tid if (gt_tid is not None and gt_tid >= 0) else None  # type: ignore[attr-defined]
            if m.pred_frame is not None and m.pred_action is not None:
                raw_pred_tid = pred_by_frame_action.get((m.pred_frame, m.pred_action))
                if raw_pred_tid is not None and raw_pred_tid >= 0:
                    # Apply the same Session-11 normalization the literal
                    # compare uses, so the oracle measures residual
                    # mismatch on top of the canonical mapping.
                    if rally_t2p is not None:
                        raw_pred_tid = rally_t2p.get(raw_pred_tid, raw_pred_tid)
                    m._pred_tid = raw_pred_tid  # type: ignore[attr-defined]
                else:
                    m._pred_tid = None  # type: ignore[attr-defined]
            else:
                m._pred_tid = None  # type: ignore[attr-defined]

        rc, rn, rally_permutation = _rally_permutation_oracle(matches)
        oracle_correct_total += rc
        oracle_evaluable_total += rn

        # Apply the same rally-local permutation to GT serves for
        # `serve_attr_oracle`. This keeps serve_attr drift-proof the same
        # way `player_attribution_oracle` is. Without this, serve_attr
        # still uses the literal-compare path via `m.player_correct` and
        # remains vulnerable to retrack label shuffling.
        for m in matches:
            if (
                m.gt_action == "serve"
                and m.pred_frame is not None
                and m.player_evaluable
            ):
                gt_tid = getattr(m, "_gt_tid", None)
                pred_tid = getattr(m, "_pred_tid", None)
                if gt_tid is None or pred_tid is None:
                    continue
                serve_oracle_evaluable_total += 1
                if rally_permutation.get(gt_tid) == pred_tid:
                    serve_oracle_correct_total += 1

        all_matches.extend(matches)
        all_unmatched.extend(unmatched)

        # Session 5 — capture predicted RallyActions and per-rally score GT
        # for the match-level score metric.
        pred_by_video.setdefault(rally.video_id, []).append(
            (rally.start_ms, rally_actions_obj)
        )
        gt_lookup[rally.rally_id] = (rally.gt_serving_team, rally.gt_point_winner)

        if print_progress and (idx % 20 == 0 or idx == len(rallies)):
            console.print(f"  [{idx}/{len(rallies)}] processed")

    # ---- Cross-rally Viterbi post-processing (Phase 2) ----
    # Override per-rally serving_team with dual-hypothesis Viterbi.
    # Tries both near=A and near=B, picks the more plausible score
    # progression. Production-viable: +7.3pp with no GT or user input.
    if not ctx.skip_viterbi_scoring:
        _apply_viterbi_scoring(
            rallies, pred_by_video, formation_flip_by_rally, t2p_by_rally,
            team_templates_by_video, calibrators=calibrators,
        )

    return (
        all_matches,
        all_unmatched,
        rejections,
        pred_by_video,
        gt_lookup,
        (oracle_correct_total, oracle_evaluable_total),
        (serve_oracle_correct_total, serve_oracle_evaluable_total),
    )


def _serve_metrics(matches: list[MatchResult]) -> tuple[float, int, float, int]:
    """Compute (serve_id_accuracy, serve_id_n, serve_attr_accuracy, serve_attr_n).

    - serve_id:   of all GT serves, what fraction have a predicted contact
                  classified as "serve" at that frame?
    - serve_attr: of all matched GT serves that have an evaluable track_id,
                  what fraction got the server's player_track_id right?
    """
    gt_serves = [m for m in matches if m.gt_action == "serve"]
    id_n = len(gt_serves)
    id_correct = sum(1 for m in gt_serves if m.pred_action == "serve")

    attr_pool = [m for m in gt_serves if m.pred_frame is not None and m.player_evaluable]
    attr_n = len(attr_pool)
    attr_correct = sum(1 for m in attr_pool if m.player_correct)

    id_acc = id_correct / max(1, id_n)
    attr_acc = attr_correct / max(1, attr_n)
    return id_acc, id_n, attr_acc, attr_n


def _flatten_run(
    matches: list[MatchResult],
    unmatched: list[dict],
    score_metrics: ScoreMetrics | None = None,
    oracle_counts: tuple[int, int] | None = None,
    serve_oracle_counts: tuple[int, int] | None = None,
) -> dict[str, float]:
    """Collapse a single run into scalar metrics consumed by the aggregator."""
    m = compute_metrics(matches, unmatched)
    serve_id, _, serve_attr, _ = _serve_metrics(matches)
    out: dict[str, float] = {
        "contact_f1":                  float(m["f1"]),
        "contact_recall":              float(m["recall"]),
        "contact_precision":           float(m["precision"]),
        "action_accuracy":             float(m["action_accuracy"]),
        "court_side_accuracy":         float(m["court_side_accuracy"]),
        "player_attribution_accuracy": float(m["player_evaluable_accuracy"]),
        "serve_id_accuracy":           float(serve_id),
        "serve_attr_accuracy":         float(serve_attr),
    }
    if oracle_counts is not None:
        oc, on = oracle_counts
        if on > 0:
            out["player_attribution_oracle"] = oc / on
    if serve_oracle_counts is not None:
        soc, son = serve_oracle_counts
        if son > 0:
            out["serve_attr_oracle"] = soc / son
    # Session 5 — score metric. Emitted whenever at least one rally has
    # `gt_serving_team` labeled; otherwise the key is absent and the summary
    # prints "n/a".
    if score_metrics is not None and score_metrics.n_rallies_scored > 0:
        out["score_accuracy"] = float(score_metrics.score_accuracy)
    # Per-class F1 goes in with a prefix so the aggregator treats each as
    # its own metric for variance purposes.
    for cls, stats in m["per_class"].items():
        out[f"per_class::{cls}::f1"] = float(stats["f1"])
    return out


def _aggregate_runs(run_scalars: list[dict[str, float]]) -> tuple[dict[str, dict[str, Any]], str | None]:
    """Compute mean/std/values and a variance warning if any std dev > 1pp."""
    if not run_scalars:
        return {}, None
    keys = run_scalars[0].keys()
    agg: dict[str, dict[str, Any]] = {}
    warn: str | None = None
    for k in keys:
        vals = [r[k] for r in run_scalars]
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=0))
        agg[k] = {"mean": mean, "std": std, "values": vals}
        if std > 0.01 and warn is None:
            warn = (
                f"metric {k!r} std dev {std * 100:.2f}pp > 1pp — "
                "consider increasing --reruns"
            )
    return agg, warn


# --------------------------------------------------------------------------- #
# Parity check                                                                #
# --------------------------------------------------------------------------- #


def _parity_check(rally_id: str) -> int:
    """Run the dashboard on one rally and diff per-contact against DB-stored production output.

    Comparison is against `rally.contacts_json` / `rally.actions_json` which
    were written by a prior `track-players --actions` run. Recency caveat: if
    those DB rows are stale, divergence may reflect pipeline progress rather
    than a dashboard bug.
    """
    rallies = load_rallies_with_action_gt(rally_id=rally_id)
    if not rallies:
        console.print(f"[red]rally {rally_id} not found or has no action GT[/red]")
        return 1
    rally = rallies[0]

    rally_pos_lookup: dict[str, list[PlayerPosition]] = {}
    if rally.positions_json:
        rally_pos_lookup[rally.rally_id] = _parse_positions(rally.positions_json)
    team_map = _load_match_team_assignments({rally.video_id}, rally_positions=rally_pos_lookup)

    calibrators = _build_calibrators({rally.video_id})
    cam_heights = _build_camera_heights({rally.video_id}, calibrators)
    ctx = PipelineContext()
    pred_actions, _ = _run_rally(
        rally,
        team_map.get(rally.rally_id),
        calibrators.get(rally.video_id),
        ctx,
        camera_height=cam_heights.get(rally.video_id, 0.0),
    )

    # DB-stored production output (from the last track-players run that
    # populated this row). May be stale.
    stored_actions: list[dict] = []
    if rally.actions_json and isinstance(rally.actions_json, dict):
        stored_actions = list(rally.actions_json.get("actions", []) or [])
    if not stored_actions:
        console.print(
            "[yellow]no actions_json stored for this rally — cannot diff. "
            "Pick a rally whose production run populated actions_json.[/yellow]"
        )
        return 2

    # Index stored by frame for a cheap nearest-match per GT contact.
    stored_by_frame = {int(a["frame"]): a for a in stored_actions if "frame" in a}
    diffs = 0
    total = 0
    table = Table(title=f"parity diff for {rally_id}", show_lines=False)
    table.add_column("gt_frame", justify="right")
    table.add_column("gt_action")
    table.add_column("dash (action, tid, side)")
    table.add_column("stored (action, tid, side)")
    table.add_column("match?")

    pred_by_frame: dict[int, dict] = {}
    for a in pred_actions:
        if a.get("isSynthetic"):
            continue
        pred_by_frame[int(a["frame"])] = a

    for gt in rally.gt_labels:
        total += 1
        # Nearest-frame within ±5 for both.
        dash = _nearest(pred_by_frame, gt.frame, 5)
        stored = _nearest(stored_by_frame, gt.frame, 5)
        if dash is None or stored is None:
            table.add_row(
                str(gt.frame), gt.action,
                _fmt(dash), _fmt(stored),
                "[yellow]missing[/yellow]",
            )
            diffs += 1
            continue
        same = (
            dash.get("action") == stored.get("action")
            and dash.get("playerTrackId") == stored.get("playerTrackId")
            and dash.get("courtSide") == stored.get("courtSide")
        )
        table.add_row(
            str(gt.frame), gt.action,
            _fmt(dash), _fmt(stored),
            "[green]ok[/green]" if same else "[red]DIFF[/red]",
        )
        if not same:
            diffs += 1

    console.print(table)
    console.print(
        f"parity: {total - diffs}/{total} GT contacts match DB-stored production output"
    )
    if diffs:
        console.print(
            "[yellow]Diffs present. May be (a) dashboard bug — fix call order/args, "
            "(b) stale DB row — pick a more recent rally, or (c) non-determinism "
            "in MS-TCN / contact GBM. If (a), stop and investigate before running "
            "full eval.[/yellow]"
        )
    return 0 if diffs == 0 else 3


def _nearest(by_frame: dict[int, dict], target: int, tol: int) -> dict | None:
    best: dict | None = None
    best_d = tol + 1
    for f, a in by_frame.items():
        d = abs(f - target)
        if d < best_d:
            best = a
            best_d = d
    return best


def _fmt(a: dict | None) -> str:
    if a is None:
        return "—"
    return f"{a.get('action')}, tid={a.get('playerTrackId')}, {a.get('courtSide')}"


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


def _git_sha() -> str:
    # Include a `-dirty` suffix when the working tree differs from HEAD, so
    # two runs with an uncommitted patch toggled in/out can't appear to
    # share a git_sha in the output JSON. The 2026-04-14 "1.6pp variance at
    # same sha" misdiagnosis (see action_audit_2026_04_14.md) was actually
    # a pre/post A/B of commit b59e264 being staged as a working-tree diff.
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        dirty = subprocess.call(
            ["git", "diff", "--quiet", "HEAD"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ) != 0
        return f"{sha}-dirty" if dirty else sha
    except Exception:  # noqa: BLE001
        return "unknown"


def _print_summary(
    agg: dict[str, dict[str, Any]],
    n_loaded: int,
    n_eval: int,
    rejections: list[dict[str, str]],
    warning: str | None,
    score_metrics: ScoreMetrics | None = None,
) -> None:
    # `player_attribution_oracle` is the CANONICAL player-attribution metric
    # going forward (2026-04-10). It is permutation-invariant over within-rally
    # pred trackIds — no retrack can produce the illusion of a regression by
    # reshuffling integer labels. The literal-compare metric
    # `player_attribution_accuracy` is kept as a drift-sensitive diagnostic
    # only; do NOT use it for go/no-go decisions. See
    # `memory/diagnosis_2026-04-10.md` §Oracle diagnostic for rationale.
    headline_keys = [
        "contact_f1", "contact_recall", "contact_precision",
        "action_accuracy", "court_side_accuracy",
        "player_attribution_oracle",
        "player_attribution_accuracy",  # diagnostic only — drift-sensitive
        "serve_id_accuracy",
        "serve_attr_oracle",
        "serve_attr_accuracy",          # diagnostic only — drift-sensitive
        "score_accuracy",
    ]
    table = Table(title="production_eval — aggregate (mean ± std across reruns)")
    table.add_column("metric")
    table.add_column("mean", justify="right")
    table.add_column("± std", justify="right")
    for k in headline_keys:
        if k not in agg:
            continue
        m = agg[k]
        table.add_row(k, f"{m['mean']:.1%}", f"{m['std'] * 100:.2f}pp")
    console.print(table)

    per_class = {k: v for k, v in agg.items() if k.startswith("per_class::")}
    if per_class:
        pc = Table(title="per-class F1 (mean ± std)")
        pc.add_column("class")
        pc.add_column("f1", justify="right")
        pc.add_column("± std", justify="right")
        for k, v in per_class.items():
            cls = k.split("::")[1]
            pc.add_row(cls, f"{v['mean']:.1%}", f"{v['std'] * 100:.2f}pp")
        console.print(pc)

    # Session 5 — score coverage sub-table.
    if score_metrics is not None:
        if score_metrics.n_rallies_scored > 0:
            st = Table(title="score breakdown (Session 5)")
            st.add_column("metric")
            st.add_column("value", justify="right")
            st.add_row("score_accuracy (pred serving == gt serving)",
                       f"{score_metrics.score_accuracy:.1%}")
            st.add_row("rallies scored", str(score_metrics.n_rallies_scored))
            st.add_row("videos with GT", str(score_metrics.n_videos_with_any_gt))
            console.print(st)
        else:
            console.print(
                "[yellow]score_accuracy: n/a[/yellow] — no rally has "
                "gt_serving_team labeled yet. Label in the editor's "
                "Ground Truth panel to populate."
            )

    console.print(
        f"[bold]rallies:[/bold] loaded={n_loaded}  evaluated={n_eval}  "
        f"rejected={len(rejections)}"
    )
    if rejections:
        console.print("[yellow]production_rejected (first 10):[/yellow]")
        for r in rejections[:10]:
            console.print(f"  - {r['rally_id']}: {r['reason']}")
    if warning:
        console.print(f"[yellow]variance warning:[/yellow] {warning}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument("--reruns", type=int, default=3,
                        help="Number of full-loop reruns for variance (default: 3)")
    parser.add_argument("--ablate", action="append", default=[],
                        choices=sorted(ABLATIONS.keys()),
                        help="Disable a production component (repeatable).")
    parser.add_argument("--parity-check", type=str, default=None, metavar="RALLY_ID",
                        help="Diff one rally's dashboard output vs DB-stored production output and exit.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N rallies (debug).")
    args = parser.parse_args()

    analysis_root = Path(__file__).resolve().parent.parent
    weights = _weights_inventory(analysis_root)

    if args.parity_check:
        return _parity_check(args.parity_check)

    console.print("[bold]Loading rallies with action GT...[/bold]")
    rallies = load_rallies_with_action_gt()
    if args.limit:
        rallies = rallies[: args.limit]
    n_loaded = len(rallies)
    console.print(f"  {n_loaded} rallies")

    # Build verified match team assignments exactly as production does
    # (pass `rally_positions=...` so `verify_team_assignments` runs).
    rally_pos_lookup: dict[str, list[PlayerPosition]] = {}
    for r in rallies:
        if r.positions_json:
            rally_pos_lookup[r.rally_id] = _parse_positions(r.positions_json)
    video_ids = {r.video_id for r in rallies}
    team_map = _load_match_team_assignments(video_ids, rally_positions=rally_pos_lookup)
    # Session 11: load per-rally trackId → canonical player-ID (1-4) mappings
    # so match_contacts can normalize predicted trackIds before comparing
    # against GT. See memory/housekeeping_retrack_2026_04_09.md.
    t2p_by_rally = _load_track_to_player_maps(video_ids)
    # Session W1: per-rally cumulative side-switch count, used by the
    # formation-based serving_team predictor to convert physical→semantic
    # team labels on flipped rallies.
    # See memory/score_tracking_architecture_2026_04.md.
    formation_flip_by_rally = _load_formation_semantic_flips_from_gt(video_ids)
    team_templates_by_video = _load_team_templates_by_video(video_ids)
    if team_templates_by_video:
        console.print(
            f"  team templates loaded for {len(team_templates_by_video)}/{len(video_ids)} videos"
        )

    # Production passes a real `CourtCalibrator` into the three enrichment
    # stages when calibration is available. Build them once up front.
    calibrators = _build_calibrators(video_ids)
    camera_heights = _build_camera_heights(video_ids, calibrators)
    console.print(
        f"  court calibration available for {len(calibrators)}/{len(video_ids)} videos"
        f", camera height for {len(camera_heights)}/{len(video_ids)}"
    )

    # Build PipelineContext + apply ablations.
    ctx = PipelineContext()
    for name in args.ablate:
        ABLATIONS[name](ctx)
    if args.ablate:
        console.print(f"[yellow]ablations active:[/yellow] {', '.join(args.ablate)}")

    # Run N times.
    run_scalars: list[dict[str, float]] = []
    last_rejections: list[dict[str, str]] = []
    last_score_metrics: ScoreMetrics | None = None
    n_eval = 0
    for run_i in range(args.reruns):
        console.print(f"[bold]Run {run_i + 1}/{args.reruns}[/bold]")
        t0 = time.time()
        (
            matches,
            unmatched,
            rejections,
            pred_by_video,
            gt_lookup,
            oracle_counts,
            serve_oracle_counts,
        ) = _run_once(
            rallies, team_map, calibrators, ctx, t2p_by_rally,
            formation_flip_by_rally,
            camera_heights=camera_heights,
            team_templates_by_video=team_templates_by_video,
        )
        dt = time.time() - t0
        console.print(f"  run {run_i + 1} done in {dt:.1f}s  "
                      f"(matches={len(matches)}, rejections={len(rejections)})")
        score_metrics = compute_score_metrics(pred_by_video, gt_lookup)
        run_scalars.append(_flatten_run(
            matches, unmatched, score_metrics, oracle_counts, serve_oracle_counts
        ))
        last_rejections = rejections
        last_score_metrics = score_metrics
        n_eval = n_loaded - len(rejections)

    agg, warning = _aggregate_runs(run_scalars)
    _print_summary(agg, n_loaded, n_eval, last_rejections, warning, last_score_metrics)

    # Write JSON.
    out_dir = analysis_root / "outputs" / "production_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    out_path = out_dir / f"run_{stamp}.json"

    # Regroup per_class:: keys under a nested dict for a cleaner schema.
    metrics_out: dict[str, Any] = {}
    per_class_out: dict[str, dict[str, Any]] = {}
    for k, v in agg.items():
        if k.startswith("per_class::"):
            _, cls, stat = k.split("::")
            per_class_out.setdefault(cls, {})[stat] = v
        else:
            metrics_out[k] = v
    if per_class_out:
        metrics_out["action_accuracy_per_class"] = per_class_out

    payload: dict[str, Any] = {
        "run_id": stamp,
        "git_sha": _git_sha(),
        "n_reruns": args.reruns,
        "ablations_active": list(args.ablate),
        "weights": weights,
        "n_rallies_loaded": n_loaded,
        "n_rallies_evaluated": n_eval,
        "production_rejected": last_rejections,
        "metrics": metrics_out,
        "variance_warning": warning,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    console.print(f"[green]wrote[/green] {out_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)
