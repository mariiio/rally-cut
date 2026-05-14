"""Diagnostic probe — does any candidate-level signal cleanly separate
GT BLOCKs from GT ATTACKs at existing pipeline contact frames?

Spec: ``/tmp/agent-prompt-snapshot`` (block-vs-attack diagnostic — 2026-05-14).

Premise: the 12-video trusted-GT validation
(`precision_trusted_videos_2026_05_14`) and the Phase 1.7 probe
(`phase_1_7_probe_2026_05_14`) together established that 12/13 GT blocks
have an existing pipeline candidate within ±5 frames — those candidates
are getting LABELED `attack` by the action classifier. So the block-recall
bottleneck is downstream: a 2-class block-vs-attack discrimination on
already-existing candidates.

This probe asks: is there a clean signal that separates real GT blocks from
GT attacks on those candidate frames? If yes (AUROC ≥ 0.75 on any single
signal, or any 2-signal combination), we can train/threshold a 2-class
block-vs-attack head. If no, we accept the block-recall floor.

Per-candidate signals extracted:
  Trajectory/ball:
    - direction_change_deg
    - velocity (pre-contact ball speed, frames in [f-5, f-1])
    - arc_fit_residual (deviation from quadratic fit on pre-contact y)
    - ball_y_at_contact (image-y, closer to 0 = higher in frame, near net)
    - pre_contact_ball_trajectory_dy_sign  (positive = ball was descending /
        typical attack incoming; negative = ascending / defense-style; near 0
        = flat / typical pre-block)
  Pose/player:
    - player_bbox_top_y  (image-y of player's head; smaller = higher)
    - wrist_above_net  (1 = highest detected wrist is above the net
        ground-line in image coords, 0 otherwise, NaN if no pose / no calib)
    - wrist_minus_net_y  (continuous version: wrist_y - net_y, negative is above)
    - body_center_minus_ball_y  (player bbox center minus ball-y; negative
        means player's body is above the ball, typical of an at-net player)
  Sequence/context:
    - prev_action_cross_team  (1 if prev pipeline action is opposing team; 0 same; -1 unknown)
  MS-TCN++ (if weights load):
    - ms_tcn_block_prob (per-frame BLOCK class probability at the contact)
    - ms_tcn_attack_prob (same for ATTACK)
    - ms_tcn_block_minus_attack (block - attack prob)

For each candidate:
  - GT BLOCK: 13 cases from `phase_1_7_probe_2026_05_14/results.json`.
  - GT ATTACK: every `ATTACK` row in `rally_action_ground_truth` in the
    12 videos. Use the closest pipeline action within ±5 frames as the
    "existing candidate frame" — same matching as the Phase 1.7 probe.

Analysis:
  - Per-signal block vs attack median/quartiles + AUROC.
  - Top 2-3 by AUROC.
  - Combined-signal: for the top-2, do a simple rule-based threshold scan
    and report the best precision/recall split.
  - Scatter plot of top-2 signals coloured by class.

Outputs:
  - analysis/reports/block_vs_attack_diagnostic_2026_05_14/per_case.json
  - analysis/reports/block_vs_attack_diagnostic_2026_05_14/results.md
  - analysis/reports/block_vs_attack_diagnostic_2026_05_14/scatter.png

Usage:
    cd analysis
    uv run python scripts/probe_block_vs_attack_signals_2026_05_14.py
"""
from __future__ import annotations

import json
import math
import sys
import time
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from rallycut.evaluation.tracking.db import get_connection  # noqa: E402
from rallycut.evaluation.video_resolver import VideoResolver  # noqa: E402
from rallycut.tracking.block_reclassification import (  # noqa: E402
    estimate_net_y_image,
)

VIDEO_NAMES = [
    "titi", "toto", "lulu", "wawa", "caco", "cece",
    "cici", "cuco", "gaga", "kaka", "juju", "yeye",
]
GT_FRAME_TOL = 5

REPORT_DIR = HERE.parent / "reports" / "block_vs_attack_diagnostic_2026_05_14"
PER_CASE_JSON = REPORT_DIR / "per_case.json"
RESULTS_MD = REPORT_DIR / "results.md"
SCATTER_PNG = REPORT_DIR / "scatter.png"
FRAMES_CACHE = REPORT_DIR / "_frame_cache"

# 13 missed GT block cases — pulled verbatim from
# analysis/reports/phase_1_7_probe_2026_05_14/results.json.
GT_BLOCK_CASES: list[dict[str, Any]] = [
    {"video": "titi", "rally_id": "1e38daab-c435-4298-8788-cafb19c18171", "gt_frame": 185, "gt_pid": 4},
    {"video": "juju", "rally_id": "6022138d-db87-465e-892d-aa58ea4e9d6f", "gt_frame": 220, "gt_pid": 1},
    {"video": "kaka", "rally_id": "f33d7ac8-d348-4525-9caf-5082020d534c", "gt_frame": 462, "gt_pid": 4},
    {"video": "toto", "rally_id": "70bd06c8-3a66-4e8c-b1ef-6d4a804891ef", "gt_frame": 206, "gt_pid": 2},
    {"video": "juju", "rally_id": "d810943e-dd1c-4518-a6af-577b22555c3c", "gt_frame": 390, "gt_pid": 2},
    {"video": "juju", "rally_id": "e03ef981-b1e1-43b3-93c9-6c4480e9492a", "gt_frame": 312, "gt_pid": 2},
    {"video": "juju", "rally_id": "c89b346b-5989-48fd-ab3a-f9b1596db8bd", "gt_frame": 227, "gt_pid": 1},
    {"video": "juju", "rally_id": "acada27e-5bf8-4f94-94aa-d615df953440", "gt_frame": 241, "gt_pid": 1},
    {"video": "caco", "rally_id": "9452ee5a-4ec7-40b6-904d-50c855dcd545", "gt_frame": 190, "gt_pid": 4},
    {"video": "yeye", "rally_id": "2d3cb54b-78f4-46fd-965a-ccb74b458f1b", "gt_frame": 509, "gt_pid": 3},
    {"video": "toto", "rally_id": "67b3e1ad-b994-4db9-8e2e-b41f53247f36", "gt_frame": 173, "gt_pid": 4},
    {"video": "caco", "rally_id": "cfc464a7-423f-4c65-aa9b-0f0484d1da20", "gt_frame": 336, "gt_pid": 4},
    {"video": "cici", "rally_id": "d362c7b2-2699-4edd-be83-5e5b563bf463", "gt_frame": 241, "gt_pid": None},
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CandidateRow:
    """One sample (block or attack) at a pipeline-contact frame."""

    klass: str                      # "block" | "attack"
    video: str
    rally_id: str
    rally_short: str
    gt_frame: int
    gt_pid: int | None
    # Pipeline contact pinned to this GT row.
    pl_frame: int | None
    pl_tid: int | None
    pl_action: str | None
    delta_frames: int | None
    # Trajectory/ball signals
    direction_change_deg: float | None = None
    velocity: float | None = None
    arc_fit_residual: float | None = None
    ball_y_at_contact: float | None = None
    pre_contact_ball_dy_sign: float | None = None  # mean(d_y) over pre-frames
    # Pose/player signals
    player_bbox_top_y: float | None = None
    player_bbox_center_y: float | None = None
    wrist_y_image: float | None = None
    wrist_minus_net_y: float | None = None
    wrist_above_net: float | None = None  # 0/1/None
    body_center_minus_ball_y: float | None = None
    net_y_image: float | None = None
    # Sequence
    prev_action: str | None = None
    prev_team_cross: float | None = None  # 1=cross, 0=same, None=unknown
    # MS-TCN++
    ms_tcn_block_prob: float | None = None
    ms_tcn_attack_prob: float | None = None
    # Diagnostic
    note: str = ""


# ---------------------------------------------------------------------------
# DB fetching
# ---------------------------------------------------------------------------

def _fetch_video_meta() -> dict[str, dict[str, Any]]:
    """video_name -> {id, fps, width, height, court_calibration_json, s3_key, ...}."""
    out: dict[str, dict[str, Any]] = {}
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT id, name, fps, width, height,
                      court_calibration_json,
                      s3_key, proxy_s3_key, processed_s3_key, content_hash
               FROM videos WHERE name = ANY(%s)""",
            [VIDEO_NAMES],
        )
        for r in cur.fetchall():
            out[str(r[1])] = {
                "id": str(r[0]),
                "fps": float(r[2]) if r[2] is not None else 30.0,
                "width": int(r[3] or 0),
                "height": int(r[4] or 0),
                "court_calibration_json": r[5],
                "s3_key": r[6],
                "proxy_s3_key": r[7],
                "processed_s3_key": r[8],
                "content_hash": r[9],
            }
    return out


def _fetch_rally_data(rally_id: str) -> dict[str, Any] | None:
    """Pull actions/contacts/positions/ball_positions for a rally."""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT actions_json, contacts_json, positions_json, ball_positions_json
               FROM player_tracks WHERE rally_id = %s""",
            [rally_id],
        )
        r = cur.fetchone()
    if not r:
        return None
    return {
        "actions_json": r[0] or {},
        "contacts_json": r[1] or {},
        "positions_json": r[2] or [],
        "ball_positions_json": r[3] or [],
    }


def _fetch_attacks_with_rallies(video_id: str) -> list[tuple[str, str, int, int | None]]:
    """Return list of (rally_id, rally_short, frame, resolved_track_id) for ATTACK GT rows."""
    out: list[tuple[str, str, int, int | None]] = []
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT g.rally_id, g.frame, g.resolved_track_id
               FROM rally_action_ground_truth g
               JOIN rallies r ON r.id = g.rally_id
               WHERE r.video_id = %s AND g.action::text = 'ATTACK'""",
            [video_id],
        )
        for r in cur.fetchall():
            rid = str(r[0])
            out.append((rid, rid.split("-", 1)[0], int(r[1]),
                        int(r[2]) if r[2] is not None else None))
    return out


# ---------------------------------------------------------------------------
# Signal extraction
# ---------------------------------------------------------------------------

def _find_pipeline_action_near(
    actions: list[dict[str, Any]], gt_frame: int, tol: int = GT_FRAME_TOL,
) -> tuple[dict[str, Any] | None, int]:
    """Find the action with frame closest to gt_frame (within tol). Returns (action_dict, index)."""
    best: dict[str, Any] | None = None
    best_idx = -1
    best_d = tol + 1
    for i, a in enumerate(sorted(actions, key=lambda x: int(x.get("frame", 0)))):
        d = abs(int(a.get("frame", 0)) - gt_frame)
        if d <= tol and d < best_d:
            best = a
            best_idx = i
            best_d = d
    return best, best_idx


def _ball_velocity(
    ball_by_frame: dict[int, tuple[float, float]],
    frame: int,
    window: int = 5,
) -> float | None:
    """Mean per-frame ball speed (normalized units) over frames [frame-window, frame-1]."""
    pts: list[tuple[int, float, float]] = []
    for f in range(frame - window, frame):
        if f in ball_by_frame:
            pts.append((f, *ball_by_frame[f]))
    if len(pts) < 2:
        return None
    total = 0.0
    for i in range(1, len(pts)):
        df = pts[i][0] - pts[i - 1][0]
        if df == 0:
            continue
        dx = pts[i][1] - pts[i - 1][1]
        dy = pts[i][2] - pts[i - 1][2]
        total += math.hypot(dx, dy) / max(df, 1)
    return total / max(1, len(pts) - 1)


def _arc_fit_residual(
    ball_by_frame: dict[int, tuple[float, float]],
    frame: int,
    window: int = 8,
) -> float | None:
    """Fit a quadratic y = a + b*t + c*t^2 over pre-contact frames; return RMS residual.

    Higher residual = less parabolic = more "kink"/deflection-like.
    """
    pts: list[tuple[int, float]] = []
    for f in range(frame - window, frame):
        if f in ball_by_frame:
            pts.append((f, ball_by_frame[f][1]))
    if len(pts) < 4:
        return None
    ts = np.array([p[0] - frame for p in pts], dtype=np.float64)
    ys = np.array([p[1] for p in pts], dtype=np.float64)
    try:
        coeffs = np.polyfit(ts, ys, 2)
        preds = np.polyval(coeffs, ts)
        return float(np.sqrt(np.mean((ys - preds) ** 2)))
    except Exception:  # noqa: BLE001
        return None


def _pre_dy_sign(
    ball_by_frame: dict[int, tuple[float, float]],
    frame: int,
    window: int = 5,
) -> float | None:
    """Mean Δy per frame over pre-contact window.

    Image-y is INVERTED (0 = top, 1 = bottom). Positive Δy ⇒ ball
    moving DOWN (descending, typical attack incoming). Negative ⇒
    moving UP (ascending, typical defense). ~0 ⇒ flat (typical pre-block).
    """
    pts: list[tuple[int, float]] = []
    for f in range(frame - window, frame):
        if f in ball_by_frame:
            pts.append((f, ball_by_frame[f][1]))
    if len(pts) < 2:
        return None
    dys: list[float] = []
    for i in range(1, len(pts)):
        df = pts[i][0] - pts[i - 1][0]
        if df == 0:
            continue
        dys.append((pts[i][1] - pts[i - 1][1]) / max(df, 1))
    if not dys:
        return None
    return float(np.mean(dys))


def _player_position(
    positions: list[dict[str, Any]], frame: int, tid: int,
) -> tuple[float, float, float, float] | None:
    """Return (cx, cy, w, h) at frame for tid, else None."""
    for p in positions:
        try:
            f = int(p.get("frameNumber", -1))
            t = int(p.get("trackId", -1))
        except (TypeError, ValueError):
            continue
        if f == frame and t == tid:
            return (
                float(p.get("x", 0.0)),
                float(p.get("y", 0.0)),
                float(p.get("width", 0.0)),
                float(p.get("height", 0.0)),
            )
    return None


# ---------------------------------------------------------------------------
# Pose inference (yolo11s-pose for wrist Y)
# ---------------------------------------------------------------------------

KPT_LEFT_WRIST = 9
KPT_RIGHT_WRIST = 10
WRIST_CONF_FLOOR = 0.30
POSE_TRACK_IOU_THRESHOLD = 0.30
_POSE_MODEL = None


def _get_pose_model():
    global _POSE_MODEL
    if _POSE_MODEL is None:
        from ultralytics import YOLO
        weight_path = HERE.parent / "yolo11s-pose.pt"
        _POSE_MODEL = YOLO(str(weight_path)) if weight_path.exists() else YOLO("yolo11s-pose.pt")
    return _POSE_MODEL


def _bbox_iou(b1: tuple[float, float, float, float], b2: tuple[float, float, float, float]) -> float:
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    u = a1 + a2 - inter
    return inter / u if u > 0 else 0.0


def _load_source_frame(video_path: Path, source_frame_index: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, 0, 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, source_frame_index)
    ok, frame = cap.read()
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if not ok or frame is None:
        return None, w, h
    return frame, w, h


def _wrist_y_for_player(
    video_path: Path,
    source_frame_index: int,
    player_bbox_norm: tuple[float, float, float, float],  # (cx, cy, w, h) in normalized 0-1
) -> tuple[float | None, str | None]:
    """Run yolo11s-pose at source_frame; return (higher-wrist y in normalized image-y,
    which='left'|'right'). Match by bbox IoU."""
    frame, img_w, img_h = _load_source_frame(video_path, source_frame_index)
    if frame is None:
        return None, None
    pose = _get_pose_model()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_list = pose.predict(frame, verbose=False, imgsz=1280)
    if not res_list:
        return None, None
    res = res_list[0]
    if res.keypoints is None or res.boxes is None:
        return None, None
    kps_all = res.keypoints.data.cpu().numpy()
    boxes_pix = res.boxes.xyxy.cpu().numpy()
    cx, cy, bw, bh = player_bbox_norm
    pbox = (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)
    best_iou = 0.0
    best_kp = None
    for di in range(len(kps_all)):
        bx = boxes_pix[di]
        det_box_norm = (bx[0] / img_w, bx[1] / img_h, bx[2] / img_w, bx[3] / img_h)
        iou = _bbox_iou(det_box_norm, pbox)
        if iou >= POSE_TRACK_IOU_THRESHOLD and iou > best_iou:
            best_iou = iou
            kp = kps_all[di].copy()
            kp[:, 0] /= img_w
            kp[:, 1] /= img_h
            best_kp = kp
    if best_kp is None:
        return None, None
    best_y = None
    best_which = None
    for idx, name in ((KPT_LEFT_WRIST, "left"), (KPT_RIGHT_WRIST, "right")):
        wx, wy, wc = float(best_kp[idx, 0]), float(best_kp[idx, 1]), float(best_kp[idx, 2])
        if wc < WRIST_CONF_FLOOR:
            continue
        if best_y is None or wy < best_y:
            best_y = wy
            best_which = name
    return best_y, best_which


# ---------------------------------------------------------------------------
# MS-TCN++ probs (per-rally; cached)
# ---------------------------------------------------------------------------

_MS_TCN_CACHE: dict[str, np.ndarray | None] = {}


def _ms_tcn_probs_for_rally(
    rally_id: str,
    rally_data: dict[str, Any],
) -> np.ndarray | None:
    """Return (NUM_CLASSES=7, T) probs from MS-TCN++ for this rally; or None.

    Cached per rally_id. ACTION_TYPES = [serve, receive, set, attack, dig, block]
    with idx 1..6; idx 0 = background. So block_idx = 6, attack_idx = 4.
    """
    if rally_id in _MS_TCN_CACHE:
        return _MS_TCN_CACHE[rally_id]
    try:
        from rallycut.tracking.ball_tracker import BallPosition as BP
        from rallycut.tracking.player_tracker import PlayerPosition as PP
        from rallycut.tracking.sequence_action_runtime import get_sequence_probs
    except Exception as e:  # noqa: BLE001
        print(f"  ms-tcn import failed: {e}")
        _MS_TCN_CACHE[rally_id] = None
        return None

    ball_positions = []
    max_frame = 0
    for bp in rally_data.get("ball_positions_json", []) or []:
        try:
            f = int(bp.get("frameNumber", -1))
            x = float(bp.get("x", 0.0)); y = float(bp.get("y", 0.0))
            c = float(bp.get("confidence", 0.0))
        except (TypeError, ValueError, KeyError):
            continue
        if f < 0:
            continue
        ball_positions.append(BP(frame_number=f, x=x, y=y, confidence=c))
        max_frame = max(max_frame, f)

    player_positions = []
    for p in rally_data.get("positions_json", []) or []:
        try:
            f = int(p.get("frameNumber", -1))
            t = int(p.get("trackId", -1))
            x = float(p.get("x", 0.0)); y = float(p.get("y", 0.0))
            w = float(p.get("width", 0.0)); h = float(p.get("height", 0.0))
            c = float(p.get("confidence", 0.0))
        except (TypeError, ValueError, KeyError):
            continue
        if f < 0 or t < 0:
            continue
        player_positions.append(PP(
            frame_number=f, track_id=t, x=x, y=y, width=w, height=h, confidence=c,
        ))
        max_frame = max(max_frame, f)

    team_ass_raw = (rally_data.get("actions_json") or {}).get("teamAssignments", {}) or {}
    team_ass: dict[int, int] = {}
    for k, v in team_ass_raw.items():
        try:
            team_ass[int(k)] = 0 if str(v) == "A" else 1
        except (TypeError, ValueError):
            continue

    frame_count = max_frame + 1
    if frame_count < 10:
        _MS_TCN_CACHE[rally_id] = None
        return None
    try:
        probs = get_sequence_probs(
            ball_positions=ball_positions,
            player_positions=player_positions,
            court_split_y=None,
            frame_count=frame_count,
            team_assignments=team_ass,
            calibrator=None,
        )
    except Exception as e:  # noqa: BLE001
        print(f"  ms-tcn inference failed for {rally_id[:8]}: {e}")
        probs = None
    _MS_TCN_CACHE[rally_id] = probs
    return probs


# ---------------------------------------------------------------------------
# Per-case processing
# ---------------------------------------------------------------------------

def _process_case(
    klass: str,
    video_name: str,
    rally_id: str,
    gt_frame: int,
    gt_pid: int | None,
    video_meta: dict[str, Any],
    video_path_cache: dict[str, Path | None],
    resolver: VideoResolver,
) -> CandidateRow:
    rally_data = _fetch_rally_data(rally_id)
    row = CandidateRow(
        klass=klass, video=video_name, rally_id=rally_id,
        rally_short=rally_id.split("-", 1)[0],
        gt_frame=gt_frame, gt_pid=gt_pid,
        pl_frame=None, pl_tid=None, pl_action=None, delta_frames=None,
    )
    if rally_data is None:
        row.note = "no_player_tracks_row"
        return row

    actions = (rally_data["actions_json"] or {}).get("actions", []) or []
    team_ass = (rally_data["actions_json"] or {}).get("teamAssignments", {}) or {}
    contacts = (rally_data["contacts_json"] or {}).get("contacts", []) or []
    positions = rally_data["positions_json"] or []
    ball_positions = rally_data["ball_positions_json"] or []

    # Index ball
    ball_by_frame: dict[int, tuple[float, float]] = {}
    for bp in ball_positions:
        try:
            f = int(bp["frameNumber"]); bx = float(bp.get("x", 0.0)); by = float(bp.get("y", 0.0))
        except (KeyError, TypeError, ValueError):
            continue
        if bx <= 0 and by <= 0:
            continue
        ball_by_frame[f] = (bx, by)

    # Index contacts by frame
    contacts_by_frame: dict[int, dict[str, Any]] = {}
    for c in contacts:
        try:
            contacts_by_frame[int(c.get("frame", -1))] = c
        except (TypeError, ValueError):
            continue

    # 1) Pin the closest pipeline action within ±5
    pl_action, pl_idx = _find_pipeline_action_near(actions, gt_frame, tol=GT_FRAME_TOL)
    if pl_action is None:
        row.note = "no_pipeline_action_within_pm5"
        return row
    pl_frame = int(pl_action.get("frame", -1))
    pl_tid_raw = pl_action.get("playerTrackId")
    try:
        pl_tid = int(pl_tid_raw) if pl_tid_raw is not None else None
    except (TypeError, ValueError):
        pl_tid = None
    row.pl_frame = pl_frame
    row.pl_tid = pl_tid
    row.pl_action = str(pl_action.get("action", "")).lower() or None
    row.delta_frames = pl_frame - gt_frame

    # 2) Contact-based signals
    c = contacts_by_frame.get(pl_frame)
    if c is not None:
        dc = c.get("directionChangeDeg")
        try:
            row.direction_change_deg = float(dc) if dc is not None else None
        except (TypeError, ValueError):
            row.direction_change_deg = None
        bx = c.get("ballX"); by = c.get("ballY")
        try:
            ball_y = float(by) if by is not None else None
        except (TypeError, ValueError):
            ball_y = None
        row.ball_y_at_contact = ball_y

    if row.ball_y_at_contact is None and pl_frame in ball_by_frame:
        row.ball_y_at_contact = ball_by_frame[pl_frame][1]

    row.velocity = _ball_velocity(ball_by_frame, pl_frame)
    row.arc_fit_residual = _arc_fit_residual(ball_by_frame, pl_frame)
    row.pre_contact_ball_dy_sign = _pre_dy_sign(ball_by_frame, pl_frame)

    # 3) Player bbox top/center y
    if pl_tid is not None:
        pos = _player_position(positions, pl_frame, pl_tid)
        if pos is not None:
            cx_p, cy_p, w_p, h_p = pos
            row.player_bbox_top_y = cy_p - h_p / 2.0
            row.player_bbox_center_y = cy_p
            if row.ball_y_at_contact is not None:
                row.body_center_minus_ball_y = cy_p - row.ball_y_at_contact

    # 4) Net y (from court calibration)
    cal = video_meta.get("court_calibration_json")
    net_y = estimate_net_y_image(cal)
    row.net_y_image = net_y

    # 5) Wrist y via pose inference
    if pl_tid is not None:
        pos = _player_position(positions, pl_frame, pl_tid)
        if pos is not None:
            # Need source-frame index: rally_start_ms / fps converted to frame + pl_frame
            with get_connection() as conn, conn.cursor() as cur:
                cur.execute("SELECT start_ms FROM rallies WHERE id = %s", [rally_id])
                rrow = cur.fetchone()
            start_ms = int(rrow[0]) if rrow and rrow[0] is not None else 0
            fps = float(video_meta.get("fps", 30.0))
            rally_start_frame = int(round(start_ms / 1000.0 * fps))
            source_frame_idx = rally_start_frame + pl_frame

            # Resolve / cache video path
            if video_name not in video_path_cache:
                vp = None
                for label_key in ("proxy_s3_key", "s3_key", "processed_s3_key"):
                    s3k = video_meta.get(label_key)
                    if not s3k:
                        continue
                    try:
                        vp = resolver.resolve(s3k, video_meta.get("content_hash"))
                        break
                    except Exception:  # noqa: BLE001
                        continue
                video_path_cache[video_name] = vp
            video_path = video_path_cache.get(video_name)
            if video_path is not None:
                try:
                    wy, _which = _wrist_y_for_player(video_path, source_frame_idx, pos)
                except Exception as e:  # noqa: BLE001
                    row.note = f"pose_err={e!r}"
                    wy = None
                row.wrist_y_image = wy
                if wy is not None and net_y is not None:
                    row.wrist_minus_net_y = wy - net_y
                    row.wrist_above_net = 1.0 if wy < net_y else 0.0

    # 6) Sequence context — prev action cross-team
    prev = actions[pl_idx - 1] if pl_idx > 0 else None
    if prev is not None:
        row.prev_action = str(prev.get("action", "")).lower() or None
        prev_team = prev.get("team")
        pl_team = pl_action.get("team")
        if (
            prev_team in (None, "unknown")
            or pl_team in (None, "unknown")
        ):
            row.prev_team_cross = None
        else:
            row.prev_team_cross = 1.0 if str(prev_team) != str(pl_team) else 0.0

    # 7) MS-TCN++ probs at pl_frame
    probs = _ms_tcn_probs_for_rally(rally_id, rally_data)
    if probs is not None and 0 <= pl_frame < probs.shape[1]:
        # ACTION_TYPES = [serve, receive, set, attack, dig, block] starting at idx 1.
        row.ms_tcn_attack_prob = float(probs[4, pl_frame])
        row.ms_tcn_block_prob = float(probs[6, pl_frame])

    return row


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def _auroc(pos: list[float], neg: list[float]) -> float | None:
    """Compute AUROC where higher value of signal = predicting positive class.

    Returns AUROC; if AUROC < 0.5 we keep it (caller can flip sign). Returns
    None if either class has fewer than 2 finite samples.
    """
    pos_f = [v for v in pos if v is not None and np.isfinite(v)]
    neg_f = [v for v in neg if v is not None and np.isfinite(v)]
    if len(pos_f) < 2 or len(neg_f) < 2:
        return None
    pos_arr = np.array(pos_f); neg_arr = np.array(neg_f)
    # Mann-Whitney U style: for each (pos, neg) pair, count pos > neg.
    # Use efficient rank-based computation.
    all_vals = np.concatenate([pos_arr, neg_arr])
    ranks = all_vals.argsort().argsort().astype(np.float64) + 1
    # Handle ties: average ranks.
    # Recompute ranks with avg-tie handling.
    order = np.argsort(all_vals)
    sorted_vals = all_vals[order]
    avg_ranks = np.empty_like(sorted_vals, dtype=np.float64)
    i = 0
    while i < len(sorted_vals):
        j = i
        while j + 1 < len(sorted_vals) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            avg_ranks[k] = avg_rank
        i = j + 1
    ranks_full = np.empty_like(avg_ranks)
    ranks_full[order] = avg_ranks
    pos_ranks = ranks_full[: len(pos_arr)]
    n_p = len(pos_arr); n_n = len(neg_arr)
    u = pos_ranks.sum() - n_p * (n_p + 1) / 2.0
    return float(u / (n_p * n_n))


def _quartiles(vals: list[float]) -> tuple[float, float, float]:
    arr = np.array([v for v in vals if v is not None and np.isfinite(v)])
    if arr.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    return (float(np.percentile(arr, 25)), float(np.median(arr)), float(np.percentile(arr, 75)))


def _format_q(q: tuple[float, float, float]) -> str:
    if not np.isfinite(q[1]):
        return "n/a"
    return f"{q[1]:.3f} [{q[0]:.3f}, {q[2]:.3f}]"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_CACHE.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Block-vs-attack signal diagnostic (2026-05-14)")
    print("=" * 70)

    video_meta = _fetch_video_meta()
    print(f"Loaded {len(video_meta)} video meta rows")
    for vn in VIDEO_NAMES:
        if vn not in video_meta:
            print(f"  WARN: video '{vn}' missing")

    # 1) Build the candidate list
    cases: list[tuple[str, str, str, int, int | None]] = []
    # blocks
    for c in GT_BLOCK_CASES:
        cases.append(("block", c["video"], c["rally_id"], c["gt_frame"], c.get("gt_pid")))
    # attacks: pull from DB for each video
    for vn in VIDEO_NAMES:
        if vn not in video_meta:
            continue
        for rid, _short, frame, gt_tid in _fetch_attacks_with_rallies(video_meta[vn]["id"]):
            cases.append(("attack", vn, rid, frame, gt_tid))

    n_blocks = sum(1 for k, *_ in cases if k == "block")
    n_attacks = sum(1 for k, *_ in cases if k == "attack")
    print(f"Building {len(cases)} candidate rows ({n_blocks} blocks + {n_attacks} attacks)")

    # 2) Process each case
    resolver = VideoResolver()
    video_path_cache: dict[str, Path | None] = {}
    rows: list[CandidateRow] = []
    t0_total = time.perf_counter()
    for i, (klass, vn, rid, gt_frame, gt_pid) in enumerate(cases, 1):
        t0 = time.perf_counter()
        try:
            row = _process_case(
                klass=klass, video_name=vn, rally_id=rid,
                gt_frame=gt_frame, gt_pid=gt_pid,
                video_meta=video_meta[vn], video_path_cache=video_path_cache,
                resolver=resolver,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[{i}/{len(cases)}] {klass} {vn}/{rid[:8]} f={gt_frame} ERROR: {e!r}")
            row = CandidateRow(
                klass=klass, video=vn, rally_id=rid,
                rally_short=rid.split("-", 1)[0], gt_frame=gt_frame, gt_pid=gt_pid,
                pl_frame=None, pl_tid=None, pl_action=None, delta_frames=None,
                note=f"err={e!r}",
            )
        rows.append(row)
        dt = time.perf_counter() - t0
        print(
            f"[{i}/{len(cases)}] {klass:6s} {vn}/{row.rally_short} "
            f"gt_f={gt_frame} pl_f={row.pl_frame} Δ={row.delta_frames} "
            f"act={row.pl_action} dc={row.direction_change_deg} "
            f"vel={row.velocity} wrist_above={row.wrist_above_net} "
            f"ms_block={row.ms_tcn_block_prob} ms_atk={row.ms_tcn_attack_prob} "
            f"note={row.note} dt={dt*1000:.0f}ms"
        )

    total_dt = time.perf_counter() - t0_total
    print(f"\nTotal processing time: {total_dt:.1f}s ({total_dt/max(len(cases),1)*1000:.0f}ms/case)")

    # 3) Persist per-case JSON
    per_case_dump = [asdict(r) for r in rows]
    PER_CASE_JSON.write_text(json.dumps(per_case_dump, indent=2))
    print(f"\nWrote {PER_CASE_JSON}")

    # 4) Analysis — AUROC per signal
    blocks_rows = [r for r in rows if r.klass == "block" and r.pl_frame is not None]
    attacks_rows = [r for r in rows if r.klass == "attack" and r.pl_frame is not None]
    print(f"\nRows with candidate-frame data: blocks={len(blocks_rows)} attacks={len(attacks_rows)}")

    signals: dict[str, str] = {
        # name -> direction: "high" means high value predicts BLOCK
        "direction_change_deg": "low",      # block deflections low-angle
        "velocity": "low",                  # blocks happen with lower pre-velocity (jump+wait)
        "arc_fit_residual": "high",         # blocks deviate from parabola
        "ball_y_at_contact": "low",         # smaller y = higher = near net for block
        "pre_contact_ball_dy_sign": "low",  # blocks: flat/ascending => low or negative
        "player_bbox_top_y": "low",         # blocks: head high in frame => low y
        "wrist_y_image": "low",             # block: wrist high => low y
        "wrist_minus_net_y": "low",         # negative = above net => low value good for block
        "wrist_above_net": "high",          # 1 means above net (block)
        "body_center_minus_ball_y": "high", # block: body BELOW ball at contact => positive
        "prev_team_cross": "high",          # cross-team prev => more likely block
        "ms_tcn_block_prob": "high",
        "ms_tcn_attack_prob": "low",
    }

    def _val(r: CandidateRow, name: str) -> float | None:
        v = getattr(r, name)
        if v is None:
            return None
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(f):
            return None
        return f

    # MS-TCN block - attack (composite)
    auroc_table: list[dict[str, Any]] = []
    for sig, direction in signals.items():
        pos_vals = [_val(r, sig) for r in blocks_rows]
        neg_vals = [_val(r, sig) for r in attacks_rows]
        # AUROC assumes higher = positive. If direction='low', flip sign.
        if direction == "low":
            pos_vals_a = [-v if v is not None else None for v in pos_vals]
            neg_vals_a = [-v if v is not None else None for v in neg_vals]
        else:
            pos_vals_a = pos_vals
            neg_vals_a = neg_vals
        au = _auroc(pos_vals_a, neg_vals_a)
        block_q = _quartiles([v for v in pos_vals if v is not None])
        attack_q = _quartiles([v for v in neg_vals if v is not None])
        n_block_def = sum(1 for v in pos_vals if v is not None)
        n_attack_def = sum(1 for v in neg_vals if v is not None)
        auroc_table.append({
            "signal": sig,
            "direction": direction,
            "block_q25": block_q[0], "block_median": block_q[1], "block_q75": block_q[2],
            "attack_q25": attack_q[0], "attack_median": attack_q[1], "attack_q75": attack_q[2],
            "n_block_defined": n_block_def,
            "n_attack_defined": n_attack_def,
            "auroc": au,
        })

    # Composite: block_prob - attack_prob
    pos_b_minus_a = []
    neg_b_minus_a = []
    for r in blocks_rows:
        if r.ms_tcn_block_prob is not None and r.ms_tcn_attack_prob is not None:
            pos_b_minus_a.append(r.ms_tcn_block_prob - r.ms_tcn_attack_prob)
    for r in attacks_rows:
        if r.ms_tcn_block_prob is not None and r.ms_tcn_attack_prob is not None:
            neg_b_minus_a.append(r.ms_tcn_block_prob - r.ms_tcn_attack_prob)
    au_diff = _auroc(pos_b_minus_a, neg_b_minus_a)
    auroc_table.append({
        "signal": "ms_tcn_block_minus_attack",
        "direction": "high",
        "block_q25": np.percentile(pos_b_minus_a, 25) if pos_b_minus_a else float("nan"),
        "block_median": np.median(pos_b_minus_a) if pos_b_minus_a else float("nan"),
        "block_q75": np.percentile(pos_b_minus_a, 75) if pos_b_minus_a else float("nan"),
        "attack_q25": np.percentile(neg_b_minus_a, 25) if neg_b_minus_a else float("nan"),
        "attack_median": np.median(neg_b_minus_a) if neg_b_minus_a else float("nan"),
        "attack_q75": np.percentile(neg_b_minus_a, 75) if neg_b_minus_a else float("nan"),
        "n_block_defined": len(pos_b_minus_a),
        "n_attack_defined": len(neg_b_minus_a),
        "auroc": au_diff,
    })

    # Sort by |AUROC - 0.5| descending (most informative regardless of direction)
    def _sortkey(d: dict[str, Any]) -> float:
        au = d.get("auroc")
        return abs((au if au is not None else 0.5) - 0.5)

    auroc_table_sorted = sorted(auroc_table, key=_sortkey, reverse=True)

    print("\n--- Per-signal AUROC (sorted by |AUROC-0.5|) ---")
    print(f"{'signal':30s} {'dir':5s} {'AUROC':>7s} {'n_b':>4s} {'n_a':>4s} "
          f"{'b_med':>10s} {'a_med':>10s}")
    for d in auroc_table_sorted:
        au = d["auroc"]
        au_str = f"{au:.3f}" if au is not None else "n/a"
        bq = d["block_median"]; aq = d["attack_median"]
        bq_str = f"{bq:.3f}" if bq is not None and np.isfinite(bq) else "n/a"
        aq_str = f"{aq:.3f}" if aq is not None and np.isfinite(aq) else "n/a"
        print(f"{d['signal']:30s} {d['direction']:5s} {au_str:>7s} "
              f"{d['n_block_defined']:>4d} {d['n_attack_defined']:>4d} "
              f"{bq_str:>10s} {aq_str:>10s}")

    # 5) Top 2-signal scatter + simple-rule threshold scan
    top_signals = [d for d in auroc_table_sorted
                   if d["auroc"] is not None
                   and d["n_block_defined"] >= 5
                   and d["n_attack_defined"] >= 20][:3]
    print("\n--- Top signals (n_block>=5, n_attack>=20) ---")
    for d in top_signals:
        print(f"  {d['signal']:30s} AUROC={d['auroc']:.3f}")

    # Combined rule scan on top 2 (if exist)
    best_rule: dict[str, Any] | None = None
    if len(top_signals) >= 2:
        s1 = top_signals[0]["signal"]; d1 = top_signals[0]["direction"]
        s2 = top_signals[1]["signal"]; d2 = top_signals[1]["direction"]
        block_pairs = []
        attack_pairs = []
        for r in blocks_rows:
            v1 = _val(r, s1); v2 = _val(r, s2)
            if v1 is not None and v2 is not None:
                block_pairs.append((v1, v2))
        for r in attacks_rows:
            v1 = _val(r, s1); v2 = _val(r, s2)
            if v1 is not None and v2 is not None:
                attack_pairs.append((v1, v2))
        print(f"\n--- Combined rule scan on ({s1}, {s2}) ---")
        print(f"    block_with_both={len(block_pairs)} attack_with_both={len(attack_pairs)}")
        # Threshold grid: per-signal quartile-based grid
        if block_pairs and attack_pairs:
            all_v1 = [p[0] for p in block_pairs + attack_pairs]
            all_v2 = [p[1] for p in block_pairs + attack_pairs]
            grid_v1 = np.percentile(all_v1, np.linspace(5, 95, 19))
            grid_v2 = np.percentile(all_v2, np.linspace(5, 95, 19))
            best_f1 = -1.0
            for t1 in grid_v1:
                for t2 in grid_v2:
                    # Rule: predict BLOCK if (signal1 cmp1 t1) AND (signal2 cmp2 t2)
                    def cmp(v, t, direction):
                        return v < t if direction == "low" else v > t
                    tp = sum(1 for (a, b) in block_pairs if cmp(a, t1, d1) and cmp(b, t2, d2))
                    fp = sum(1 for (a, b) in attack_pairs if cmp(a, t1, d1) and cmp(b, t2, d2))
                    fn = len(block_pairs) - tp
                    precision = tp / max(tp + fp, 1)
                    recall = tp / max(tp + fn, 1)
                    f1 = (2 * precision * recall / max(precision + recall, 1e-9)
                          if (precision + recall) > 0 else 0.0)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_rule = {
                            "s1": s1, "d1": d1, "t1": float(t1),
                            "s2": s2, "d2": d2, "t2": float(t2),
                            "tp": tp, "fp": fp, "fn": fn,
                            "precision": precision, "recall": recall, "f1": f1,
                        }
            if best_rule is not None:
                print(f"    Best rule: ({best_rule['s1']} {'<' if best_rule['d1']=='low' else '>'} "
                      f"{best_rule['t1']:.4f}) AND ({best_rule['s2']} {'<' if best_rule['d2']=='low' else '>'} "
                      f"{best_rule['t2']:.4f})  "
                      f"TP={best_rule['tp']} FP={best_rule['fp']} FN={best_rule['fn']} "
                      f"prec={best_rule['precision']:.3f} rec={best_rule['recall']:.3f} "
                      f"F1={best_rule['f1']:.3f}")

    # Single-signal best-threshold scan for the #1
    best_single_rule: dict[str, Any] | None = None
    if top_signals:
        s = top_signals[0]["signal"]; d = top_signals[0]["direction"]
        b_vals = [_val(r, s) for r in blocks_rows]
        a_vals = [_val(r, s) for r in attacks_rows]
        b_clean = [v for v in b_vals if v is not None]
        a_clean = [v for v in a_vals if v is not None]
        if b_clean and a_clean:
            grid = np.percentile(b_clean + a_clean, np.linspace(1, 99, 49))
            best_f1 = -1.0
            for t in grid:
                def cmp_s(v: float) -> bool:
                    return v < t if d == "low" else v > t
                tp = sum(1 for v in b_clean if cmp_s(v))
                fp = sum(1 for v in a_clean if cmp_s(v))
                fn = len(b_clean) - tp
                precision = tp / max(tp + fp, 1)
                recall = tp / max(tp + fn, 1)
                f1 = (2 * precision * recall / max(precision + recall, 1e-9)
                      if (precision + recall) > 0 else 0.0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_single_rule = {
                        "signal": s, "direction": d, "threshold": float(t),
                        "tp": tp, "fp": fp, "fn": fn,
                        "precision": precision, "recall": recall, "f1": f1,
                    }
            if best_single_rule is not None:
                print(f"\n--- Best single-signal rule on {s} ---")
                print(f"    {s} {'<' if d=='low' else '>'} {best_single_rule['threshold']:.4f}  "
                      f"TP={best_single_rule['tp']} FP={best_single_rule['fp']} FN={best_single_rule['fn']} "
                      f"prec={best_single_rule['precision']:.3f} rec={best_single_rule['recall']:.3f} "
                      f"F1={best_single_rule['f1']:.3f}")

    # 6) Scatter plot
    try:
        import matplotlib  # noqa
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if len(top_signals) >= 2:
            s1 = top_signals[0]["signal"]; s2 = top_signals[1]["signal"]
            bx = [getattr(r, s1) for r in blocks_rows
                  if getattr(r, s1) is not None and getattr(r, s2) is not None]
            by = [getattr(r, s2) for r in blocks_rows
                  if getattr(r, s1) is not None and getattr(r, s2) is not None]
            ax = [getattr(r, s1) for r in attacks_rows
                  if getattr(r, s1) is not None and getattr(r, s2) is not None]
            ay = [getattr(r, s2) for r in attacks_rows
                  if getattr(r, s1) is not None and getattr(r, s2) is not None]
            fig, ax_ = plt.subplots(figsize=(8, 6))
            ax_.scatter(ax, ay, c="tab:gray", s=15, alpha=0.5, label=f"attack (n={len(ax)})")
            ax_.scatter(bx, by, c="tab:red", s=60, marker="x", label=f"block (n={len(bx)})")
            ax_.set_xlabel(f"{s1} ({top_signals[0]['direction']}=block)")
            ax_.set_ylabel(f"{s2} ({top_signals[1]['direction']}=block)")
            ax_.set_title("Block vs Attack — top-2 candidate signals\n"
                          f"AUROC {s1}={top_signals[0]['auroc']:.3f} | {s2}={top_signals[1]['auroc']:.3f}")
            ax_.legend()
            fig.tight_layout()
            fig.savefig(SCATTER_PNG, dpi=120)
            print(f"\nWrote {SCATTER_PNG}")
        else:
            print("\nSkipped scatter (need ≥2 top signals).")
    except Exception as e:  # noqa: BLE001
        print(f"\nScatter plot failed: {e}")

    # 7) Markdown report
    verdict = _decide_verdict(auroc_table_sorted, best_single_rule, best_rule)
    _write_markdown(
        n_blocks=len(blocks_rows),
        n_attacks=len(attacks_rows),
        auroc_table=auroc_table_sorted,
        top_signals=top_signals,
        best_rule=best_rule,
        best_single_rule=best_single_rule,
        verdict=verdict,
    )
    print(f"\nWrote {RESULTS_MD}")
    print(f"\n=== VERDICT ===\n{verdict}")


def _decide_verdict(
    auroc_table: list[dict[str, Any]],
    best_single_rule: dict[str, Any] | None,
    best_combined_rule: dict[str, Any] | None,
) -> str:
    """SHIP-block-head if any single signal AUROC >= 0.75, otherwise NO-SIGNAL-block-floor."""
    top_au = 0.5
    top_sig = "none"
    for d in auroc_table:
        au = d.get("auroc")
        if au is None:
            continue
        if abs(au - 0.5) > abs(top_au - 0.5):
            top_au = au
            top_sig = d["signal"]
    msg_parts = [f"Top single-signal AUROC = {top_au:.3f} ({top_sig})."]
    if best_single_rule:
        msg_parts.append(
            f"Best single-signal threshold rule: "
            f"prec={best_single_rule['precision']:.3f} rec={best_single_rule['recall']:.3f} "
            f"F1={best_single_rule['f1']:.3f}."
        )
    if best_combined_rule:
        msg_parts.append(
            f"Best 2-signal combined rule: "
            f"prec={best_combined_rule['precision']:.3f} rec={best_combined_rule['recall']:.3f} "
            f"F1={best_combined_rule['f1']:.3f}."
        )
    if abs(top_au - 0.5) >= 0.25:
        return "SHIP-block-head — " + " ".join(msg_parts)
    return "NO-SIGNAL-block-floor — " + " ".join(msg_parts)


def _write_markdown(
    *,
    n_blocks: int,
    n_attacks: int,
    auroc_table: list[dict[str, Any]],
    top_signals: list[dict[str, Any]],
    best_rule: dict[str, Any] | None,
    best_single_rule: dict[str, Any] | None,
    verdict: str,
) -> None:
    def _fnum(v: float | None, n: int = 3) -> str:
        if v is None:
            return "n/a"
        try:
            if not np.isfinite(v):
                return "n/a"
        except TypeError:
            return "n/a"
        return f"{v:.{n}f}"

    lines: list[str] = []
    lines.append("# Block-vs-attack signal diagnostic — 2026-05-14")
    lines.append("")
    lines.append("## Class counts")
    lines.append(f"- GT blocks (with pipeline candidate ±5): {n_blocks} / 13")
    lines.append(f"- GT attacks (with pipeline candidate ±5): {n_attacks}")
    lines.append(f"- Both classes have candidate-frame data: "
                 f"{'yes' if n_blocks > 0 and n_attacks > 0 else 'no'}")
    lines.append("")
    lines.append("## Per-signal AUROC")
    lines.append("")
    lines.append("| Signal | dir | Block median [Q1, Q3] | Attack median [Q1, Q3] | n_b | n_a | AUROC |")
    lines.append("|---|---|---|---|---|---|---|")
    for d in auroc_table:
        bq = (d['block_q25'], d['block_median'], d['block_q75'])
        aq = (d['attack_q25'], d['attack_median'], d['attack_q75'])
        lines.append(
            f"| `{d['signal']}` | {d['direction']} | {_format_q(bq)} | {_format_q(aq)} | "
            f"{d['n_block_defined']} | {d['n_attack_defined']} | {_fnum(d['auroc'])} |"
        )
    lines.append("")
    lines.append("Direction column meaning: `low` ⇒ smaller value predicts BLOCK; "
                 "`high` ⇒ larger value predicts BLOCK. AUROC is computed in the "
                 "direction-aware sense (AUROC > 0.5 = signal as informative).")
    lines.append("")
    lines.append("## Top signals (by |AUROC − 0.5|)")
    if not top_signals:
        lines.append("")
        lines.append("_No signal had enough defined samples in both classes (n_block≥5, n_attack≥20)._")
    else:
        lines.append("")
        for d in top_signals:
            lines.append(
                f"- `{d['signal']}` (dir={d['direction']}, AUROC={_fnum(d['auroc'])}, "
                f"n_block={d['n_block_defined']}, n_attack={d['n_attack_defined']})"
            )
    lines.append("")
    lines.append("## Best single-signal threshold rule")
    if best_single_rule is None:
        lines.append("_no rule fit_")
    else:
        op = "<" if best_single_rule["direction"] == "low" else ">"
        lines.append(
            f"`{best_single_rule['signal']}` {op} {best_single_rule['threshold']:.4f}"
        )
        lines.append("")
        lines.append(
            f"- TP={best_single_rule['tp']}, FP={best_single_rule['fp']}, FN={best_single_rule['fn']}"
        )
        lines.append(
            f"- precision={best_single_rule['precision']:.3f}, "
            f"recall={best_single_rule['recall']:.3f}, F1={best_single_rule['f1']:.3f}"
        )
    lines.append("")
    lines.append("## Best combined-signal threshold rule")
    if best_rule is None:
        lines.append("_no rule fit_")
    else:
        op1 = "<" if best_rule["d1"] == "low" else ">"
        op2 = "<" if best_rule["d2"] == "low" else ">"
        lines.append(
            f"(`{best_rule['s1']}` {op1} {best_rule['t1']:.4f}) AND "
            f"(`{best_rule['s2']}` {op2} {best_rule['t2']:.4f})"
        )
        lines.append("")
        lines.append(
            f"- TP={best_rule['tp']}, FP={best_rule['fp']}, FN={best_rule['fn']}"
        )
        lines.append(
            f"- precision={best_rule['precision']:.3f}, "
            f"recall={best_rule['recall']:.3f}, F1={best_rule['f1']:.3f}"
        )
    lines.append("")
    lines.append("## Visualization")
    if top_signals and len(top_signals) >= 2:
        lines.append(f"`{SCATTER_PNG.name}` — top-2 signals coloured by class.")
    else:
        lines.append("_scatter skipped (insufficient signals)_")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(verdict)
    lines.append("")
    RESULTS_MD.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
