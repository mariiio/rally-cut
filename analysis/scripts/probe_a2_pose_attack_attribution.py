"""A2 pose-driven attribution probe (2026-05-13).

Measures whether pose-wrist-to-ball is more discriminating than
bbox-center-to-ball when picking the actual attacker between two
same-team candidates. Spec: ``docs/superpowers/specs/2026-05-13-action-attribution-root-causes-design.md``,
section A2.

For each of 10 hand-picked attacks (plus 1 extra F3 keke case for
visualization only — no GT, excluded from metric):

1. Load contact at attack frame from the pipeline output.
2. Load the GT attacker (resolved_track_id from rally_action_ground_truth).
3. From contact.player_candidates, gather same-team candidates (≥2).
4. Compute bbox_pick = same-team candidate with smallest bbox-center → ball distance.
5. Run yolo11s-pose on the source frame. Match each pose detection to
   a tracked player via bbox-IoU.
6. For each same-team candidate, take its matched pose's nearer wrist
   (left=KPT 9 or right=KPT 10, COCO 17-keypoint), compute wrist-to-ball
   distance.
7. pose_pick = same-team candidate with smallest wrist-to-ball distance.
   If either candidate has no detected wrist (conf<0.3 on both wrists),
   case is pose_unevaluable.
8. bbox_correct = (bbox_pick == gt_tid); pose_correct = (pose_pick == gt_tid).

Aggregate:
- bbox_correct_count over 10 GT cases
- pose_correct_count over pose_evaluable cases
- disambiguation_improvement = (pose_correct - bbox_correct) / pose_evaluable_n
  (where both are counted from the same pose_evaluable subset for fairness)

Ship threshold (per spec): improvement >= 0.5 → A2 worth building.

Outputs:
- analysis/reports/a2_pose_probe/probe_results_2026_05_13.json
- analysis/reports/a2_pose_probe/probe_results_2026_05_13.md
- analysis/reports/a2_pose_probe/visual_2026_05_13.html
- analysis/reports/a2_pose_probe/visual_frames/*.jpg

Usage:
    cd analysis
    uv run python scripts/probe_a2_pose_attack_attribution.py
"""
from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from rallycut.evaluation.tracking.db import get_connection  # noqa: E402
from rallycut.evaluation.video_resolver import VideoResolver  # noqa: E402

REPORT_DIR = HERE.parent / "reports" / "a2_pose_probe"
FRAMES_DIR = REPORT_DIR / "visual_frames"
JSON_PATH = REPORT_DIR / "probe_results_2026_05_13.json"
MD_PATH = REPORT_DIR / "probe_results_2026_05_13.md"
HTML_PATH = REPORT_DIR / "visual_2026_05_13.html"

# COCO 17-keypoint indices
KPT_LEFT_WRIST = 9
KPT_RIGHT_WRIST = 10
KPT_LEFT_SHOULDER = 5
KPT_RIGHT_SHOULDER = 6

# Confidence floor for considering a wrist detected.
WRIST_CONF_FLOOR = 0.30
# IoU threshold for matching pose detection to a tracked player bbox.
POSE_TRACK_IOU_THRESHOLD = 0.30


# ---------------------------------------------------------------------------
# Hand-picked probe set (10 GT-backed + 1 F3 visualization-only)
# ---------------------------------------------------------------------------

@dataclass
class ProbeCase:
    idx: int
    kind: str            # "error" | "control" | "f3_extra"
    video_name: str
    video_id: str
    rally_id: str
    rally_short: str
    pl_frame: int        # contact frame (rally-relative)
    gt_tid: int | None   # GT attacker's resolved_track_id; None for F3 extra
    pl_tid: int          # pipeline's current pick
    same_team_cands: list[tuple[int, float]]  # [(tid, bbox_dist_to_ball)]
    rally_start_ms: int
    ball_xy: tuple[float, float]  # action.ballX, action.ballY (normalized)
    reason: str
    team_assignments: dict[str, str] = field(default_factory=dict)


# Probe cases: just (idx, kind, video, rally_short, frame). All player-id
# fields (gt_tid, pl_tid, same_team_cands) are queried LIVE from the DB at
# probe time — so re-running after relabeling in the editor picks up the
# fresh state. Kind is recomputed: "error" if gt_tid != pl_tid, "control"
# if equal, "f3_extra" if gt_tid is null. The hardcoded `kind` here is
# just a label hint for grouping; the runtime kind is what's used.
PROBE_CASES: list[dict[str, Any]] = [
    dict(idx=1, kind="error", video_name="wawa", rally_short="7094136a",
         pl_frame=306,
         reason="wawa same-team ATTACK (A1 inspection flagged)"),
    dict(idx=2, kind="error", video_name="wawa", rally_short="21d4cdf6",
         pl_frame=485,
         reason="wawa same-team ATTACK (A1 inspection flagged)"),
    dict(idx=3, kind="error", video_name="gigi", rally_short="39e866fd",
         pl_frame=297,
         reason="gigi same-team ATTACK"),
    dict(idx=4, kind="error", video_name="cece", rally_short="5c35e049",
         pl_frame=501,
         reason="cece same-team ATTACK"),
    dict(idx=5, kind="error", video_name="titi", rally_short="43b849ec",
         pl_frame=344,
         reason="titi (top C-5 video) same-team ATTACK"),
    dict(idx=6, kind="error", video_name="titi", rally_short="a0881d82",
         pl_frame=176,
         reason="titi (cascade rally) ATTACK"),
    dict(idx=7, kind="error", video_name="titi", rally_short="caa96651",
         pl_frame=173,
         reason="titi (top C-5 video) ATTACK"),
    dict(idx=8, kind="control", video_name="juju", rally_short="57d1327c",
         pl_frame=211,
         reason="juju ATTACK control"),
    dict(idx=9, kind="control", video_name="wawa", rally_short="7f0f540a",
         pl_frame=477,
         reason="wawa ATTACK control (tight gap)"),
    dict(idx=10, kind="control", video_name="lili", rally_short="879a8cff",
         pl_frame=225,
         reason="lili ATTACK control"),
    dict(idx=11, kind="f3_extra", video_name="keke", rally_short="0144acfb",
         pl_frame=223,
         reason="F3 canonical case (no GT). Logging for visual."),
]


_GT_FRAME_TOLERANCE = 5  # match probe contact frame to GT within ±N frames


# ---------------------------------------------------------------------------
# DB queries
# ---------------------------------------------------------------------------

def _fetch_cases() -> list[ProbeCase]:
    """Hydrate PROBE_CASES with rally metadata + team_assignments from DB."""
    # Map video_name → vid for lookup
    video_names = sorted({c["video_name"] for c in PROBE_CASES})
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT name, id FROM videos WHERE name = ANY(%s)",
            [video_names],
        )
        name_to_vid = {r[0]: r[1] for r in cur.fetchall()}

    out: list[ProbeCase] = []
    with get_connection() as conn, conn.cursor() as cur:
        for c in PROBE_CASES:
            vid = name_to_vid.get(c["video_name"])
            if not vid:
                print(f"[{c['idx']}] WARNING: video {c['video_name']} not found")
                continue
            cur.execute(
                """
                SELECT r.id, r.start_ms, pt.actions_json, pt.contacts_json
                FROM rallies r JOIN player_tracks pt ON pt.rally_id = r.id
                WHERE r.video_id = %s AND r.id::text LIKE %s || '%%'
                """,
                [vid, c["rally_short"]],
            )
            row = cur.fetchone()
            if not row:
                print(f"[{c['idx']}] WARNING: rally {c['rally_short']} not found")
                continue
            rid, start_ms, actions_json, contacts_json = row
            actions_json = actions_json or {}
            contacts_json = contacts_json or {}
            actions = actions_json.get("actions", [])
            team_ass = actions_json.get("teamAssignments", {}) or {}

            # Pull pl_tid and ball_xy from actions_json at pl_frame.
            pl_tid = -1
            ball_xy = (0.0, 0.0)
            for a in actions:
                if int(a.get("frame", -1)) == c["pl_frame"]:
                    pl_tid = int(a.get("playerTrackId") or -1)
                    ball_xy = (float(a.get("ballX") or 0.0),
                               float(a.get("ballY") or 0.0))
                    break

            # Pull same_team candidates from contacts_json at pl_frame.
            same_team_cands: list[tuple[int, float]] = []
            if pl_tid > 0 and team_ass:
                pl_team = team_ass.get(str(pl_tid))
                for contact in contacts_json.get("contacts", []):
                    if int(contact.get("frame", -1)) != c["pl_frame"]:
                        continue
                    for cand in contact.get("playerCandidates", []):
                        if not cand or len(cand) < 2:
                            continue
                        try:
                            tid = int(cand[0])
                            dist = float(cand[1])
                        except (TypeError, ValueError):
                            continue
                        if team_ass.get(str(tid)) == pl_team:
                            same_team_cands.append((tid, dist))
                    break  # only one contact matching this frame

            # Pull gt_tid from rally_action_ground_truth — resolved_track_id,
            # match by closest frame within tolerance.
            cur.execute(
                """
                SELECT frame, resolved_track_id
                FROM rally_action_ground_truth
                WHERE rally_id = %s
                  AND action::text = 'ATTACK'
                  AND resolved_track_id IS NOT NULL
                  AND ABS(frame - %s) <= %s
                ORDER BY ABS(frame - %s) ASC
                LIMIT 1
                """,
                [rid, c["pl_frame"], _GT_FRAME_TOLERANCE, c["pl_frame"]],
            )
            gt_row = cur.fetchone()
            gt_tid: int | None = (
                int(gt_row[1]) if gt_row and gt_row[1] is not None else None
            )

            out.append(ProbeCase(
                idx=c["idx"],
                kind=c["kind"],
                video_name=c["video_name"],
                video_id=vid,
                rally_id=rid,
                rally_short=c["rally_short"],
                pl_frame=c["pl_frame"],
                gt_tid=gt_tid,
                pl_tid=pl_tid,
                same_team_cands=same_team_cands,
                rally_start_ms=int(start_ms or 0),
                ball_xy=ball_xy,
                reason=c["reason"],
                team_assignments=team_ass,
            ))
    return out


def _fetch_video_meta(video_id: str) -> dict[str, Any]:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT id, name, filename, fps, width, height,
                      s3_key, proxy_s3_key, processed_s3_key, content_hash
               FROM videos WHERE id = %s""",
            (video_id,),
        )
        r = cur.fetchone()
    if r is None:
        return {}
    return {
        "id": r[0], "name": r[1], "filename": r[2],
        "fps": float(r[3]) if r[3] is not None else 30.0,
        "width": int(r[4] or 0), "height": int(r[5] or 0),
        "s3_key": r[6], "proxy_s3_key": r[7],
        "processed_s3_key": r[8], "content_hash": r[9],
    }


def _fetch_positions_at_frame(
    rally_id: str, frame: int,
) -> list[dict[str, Any]]:
    """Return all player positions at the given rally-relative frame."""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT positions_json FROM player_tracks WHERE rally_id = %s",
            (rally_id,),
        )
        row = cur.fetchone()
    if not row or not row[0]:
        return []
    positions = row[0]
    return [p for p in positions if int(p.get("frameNumber", -1)) == frame]


# ---------------------------------------------------------------------------
# Video / pose
# ---------------------------------------------------------------------------

def _resolve_video(resolver: VideoResolver, vm: dict[str, Any]) -> Path | None:
    candidates: list[tuple[str, str]] = []
    if vm.get("proxy_s3_key"):
        candidates.append(("proxy", vm["proxy_s3_key"]))
    if vm.get("s3_key"):
        candidates.append(("original", vm["s3_key"]))
    if vm.get("processed_s3_key"):
        candidates.append(("processed", vm["processed_s3_key"]))
    for label, key in candidates:
        try:
            return resolver.resolve(key, vm["content_hash"])
        except Exception as e:  # noqa: BLE001
            print(f"    resolve {label} failed: {e}")
    return None


def _load_frame(video_path: Path, source_frame: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, 0, 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
    ok, frame = cap.read()
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if not ok or frame is None:
        return None, img_w, img_h
    return frame, img_w, img_h


_POSE_MODEL = None


def _get_pose_model():
    global _POSE_MODEL
    if _POSE_MODEL is None:
        from ultralytics import YOLO
        # The yolo11s-pose.pt weight is at analysis/yolo11s-pose.pt
        # Working from scripts/, prefer analysis-relative path.
        weight_path = HERE.parent / "yolo11s-pose.pt"
        if weight_path.exists():
            _POSE_MODEL = YOLO(str(weight_path))
        else:
            _POSE_MODEL = YOLO("yolo11s-pose.pt")
    return _POSE_MODEL


def _bbox_iou(
    box1: tuple[float, float, float, float],
    box2: tuple[float, float, float, float],
) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def _run_pose_on_frame(
    frame_bgr,
    img_w: int,
    img_h: int,
    tracked_players: list[dict[str, Any]],
    iou_floor: float = POSE_TRACK_IOU_THRESHOLD,
) -> dict[int, np.ndarray]:
    """Run yolo11s-pose on frame; match detections to tracked players by IoU.

    Returns dict {track_id: (17, 3) keypoints in normalized 0-1 coords}.
    """
    pose = _get_pose_model()
    results = pose.predict(frame_bgr, verbose=False, imgsz=1280)  # type: ignore[attr-defined]
    if not results:
        return {}
    res = results[0]
    if res.keypoints is None or res.boxes is None:
        return {}
    kps_all = res.keypoints.data.cpu().numpy()    # (N, 17, 3)
    boxes_pix = res.boxes.xyxy.cpu().numpy()       # (N, 4) pixel xyxy

    # Build track bboxes in normalized xyxy
    track_boxes: list[tuple[int, tuple[float, float, float, float]]] = []
    for p in tracked_players:
        tid = int(p.get("trackId", -1))
        if tid < 0:
            continue
        cx = float(p.get("x", 0.0))
        cy = float(p.get("y", 0.0))
        bw = float(p.get("width", 0.0))
        bh = float(p.get("height", 0.0))
        track_boxes.append((tid, (cx - bw / 2, cy - bh / 2,
                                  cx + bw / 2, cy + bh / 2)))

    out: dict[int, np.ndarray] = {}
    # Per-track, pick the pose detection with highest IoU >= floor.
    for tid, tbox in track_boxes:
        best_iou = 0.0
        best_kp = None
        for det_idx in range(len(kps_all)):
            bx = boxes_pix[det_idx]
            det_box_norm = (bx[0] / img_w, bx[1] / img_h,
                            bx[2] / img_w, bx[3] / img_h)
            iou = _bbox_iou(det_box_norm, tbox)
            if iou >= iou_floor and iou > best_iou:
                best_iou = iou
                # Normalize keypoint (x, y, conf) → x/W, y/H, conf
                kp = kps_all[det_idx].copy()
                kp[:, 0] /= img_w
                kp[:, 1] /= img_h
                best_kp = kp
        if best_kp is not None:
            out[tid] = best_kp
    return out


def _nearest_wrist(
    kp: np.ndarray,
    ball_xy: tuple[float, float],
    conf_floor: float = WRIST_CONF_FLOOR,
) -> tuple[float | None, str | None, tuple[float, float] | None]:
    """Return (distance, which_wrist, wrist_xy) — distance is None if no wrist
    above conf_floor."""
    bx, by = ball_xy
    best_d: float | None = None
    best_which: str | None = None
    best_xy: tuple[float, float] | None = None
    for idx, name in ((KPT_LEFT_WRIST, "left"), (KPT_RIGHT_WRIST, "right")):
        wx, wy, wc = float(kp[idx, 0]), float(kp[idx, 1]), float(kp[idx, 2])
        if wc < conf_floor:
            continue
        d = math.hypot(wx - bx, wy - by)
        if best_d is None or d < best_d:
            best_d = d
            best_which = name
            best_xy = (wx, wy)
    return best_d, best_which, best_xy


# ---------------------------------------------------------------------------
# Probe per-case
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    idx: int
    kind: str
    video: str
    rally_short: str
    pl_frame: int
    gt_tid: int | None
    pl_tid: int
    bbox_pick: int                 # rank-1 same-team by bbox-center distance
    pose_pick: int | None          # rank-1 same-team by wrist-to-ball distance
    pose_evaluable: bool           # both candidates have a wrist
    bbox_correct: bool | None      # vs gt_tid; None if no GT
    pose_correct: bool | None      # vs gt_tid; None if no GT or unevaluable
    same_team_bbox_dists: dict[int, float] = field(default_factory=dict)
    pose_wrist_dists: dict[int, float | None] = field(default_factory=dict)
    pose_wrist_xy: dict[int, list[float] | None] = field(default_factory=dict)
    pose_wrist_which: dict[int, str | None] = field(default_factory=dict)
    pose_inference_ms: float = 0.0
    visual_frame_path: str | None = None
    error: str | None = None
    reason: str = ""


def _probe_one(
    case: ProbeCase,
    resolver: VideoResolver,
) -> CaseResult:
    print(f"\n[{case.idx}/11] {case.kind:>7} {case.video_name}/{case.rally_short} "
          f"pl_frame={case.pl_frame} gt={case.gt_tid} pl={case.pl_tid}")

    vm = _fetch_video_meta(case.video_id)
    if not vm:
        return CaseResult(idx=case.idx, kind=case.kind,
                          video=case.video_name, rally_short=case.rally_short,
                          pl_frame=case.pl_frame, gt_tid=case.gt_tid,
                          pl_tid=case.pl_tid,
                          bbox_pick=-1, pose_pick=None,
                          pose_evaluable=False,
                          bbox_correct=None, pose_correct=None,
                          reason=case.reason,
                          error=f"video {case.video_id} not in DB")

    video_path = _resolve_video(resolver, vm)
    if not video_path:
        return CaseResult(idx=case.idx, kind=case.kind,
                          video=case.video_name, rally_short=case.rally_short,
                          pl_frame=case.pl_frame, gt_tid=case.gt_tid,
                          pl_tid=case.pl_tid,
                          bbox_pick=-1, pose_pick=None,
                          pose_evaluable=False,
                          bbox_correct=None, pose_correct=None,
                          reason=case.reason,
                          error="video could not be resolved")

    # Get video fps from container (could differ from DB stored fps).
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return CaseResult(idx=case.idx, kind=case.kind,
                          video=case.video_name, rally_short=case.rally_short,
                          pl_frame=case.pl_frame, gt_tid=case.gt_tid,
                          pl_tid=case.pl_tid,
                          bbox_pick=-1, pose_pick=None,
                          pose_evaluable=False,
                          bbox_correct=None, pose_correct=None,
                          reason=case.reason,
                          error=f"OpenCV cannot open {video_path}")
    vid_fps = cap.get(cv2.CAP_PROP_FPS) or vm.get("fps", 30.0)
    cap.release()

    rally_start_frame = int(round(case.rally_start_ms / 1000.0 * vid_fps))
    source_frame = rally_start_frame + case.pl_frame
    print(f"    fps={vid_fps:.3f} rally_start_frame={rally_start_frame} "
          f"source_frame={source_frame}")

    frame_bgr, img_w, img_h = _load_frame(video_path, source_frame)
    if frame_bgr is None:
        return CaseResult(idx=case.idx, kind=case.kind,
                          video=case.video_name, rally_short=case.rally_short,
                          pl_frame=case.pl_frame, gt_tid=case.gt_tid,
                          pl_tid=case.pl_tid,
                          bbox_pick=-1, pose_pick=None,
                          pose_evaluable=False,
                          bbox_correct=None, pose_correct=None,
                          reason=case.reason,
                          error="failed to read frame")

    tracked = _fetch_positions_at_frame(case.rally_id, case.pl_frame)
    print(f"    tracked_players_at_frame: {len(tracked)} "
          f"(tids={sorted({int(p.get('trackId', -1)) for p in tracked})})")

    t0 = time.perf_counter()
    tid_to_kp = _run_pose_on_frame(frame_bgr, img_w, img_h, tracked)
    pose_ms = (time.perf_counter() - t0) * 1000.0
    print(f"    pose inference: {pose_ms:.0f} ms, matched_tids={sorted(tid_to_kp.keys())}")

    # bbox_pick: same_team_cands is already in rank order (smallest first).
    same_team_bbox = dict(case.same_team_cands)
    sorted_st = sorted(same_team_bbox.items(), key=lambda x: x[1])
    bbox_pick = sorted_st[0][0] if sorted_st else -1

    # pose_pick: per-candidate wrist-to-ball distance.
    pose_wrist_dists: dict[int, float | None] = {}
    pose_wrist_xy: dict[int, list[float] | None] = {}
    pose_wrist_which: dict[int, str | None] = {}
    for tid, _ in case.same_team_cands:
        kp = tid_to_kp.get(tid)
        if kp is None:
            pose_wrist_dists[tid] = None
            pose_wrist_xy[tid] = None
            pose_wrist_which[tid] = None
            continue
        d, which, xy = _nearest_wrist(kp, case.ball_xy)
        pose_wrist_dists[tid] = d
        pose_wrist_xy[tid] = list(xy) if xy else None
        pose_wrist_which[tid] = which

    # Evaluable iff every same-team candidate has at least one wrist.
    pose_evaluable = all(
        pose_wrist_dists.get(tid) is not None
        for tid, _ in case.same_team_cands
    )
    pose_pick: int | None = None
    if pose_evaluable:
        # pick min wrist distance
        sorted_pose = sorted(
            [(tid, pose_wrist_dists[tid]) for tid, _ in case.same_team_cands
             if pose_wrist_dists[tid] is not None],
            key=lambda x: x[1],
        )
        pose_pick = sorted_pose[0][0] if sorted_pose else None

    # GT-anchored correctness
    if case.gt_tid is None:
        bbox_correct = None
        pose_correct = None
    else:
        bbox_correct = (bbox_pick == case.gt_tid)
        pose_correct = (pose_pick == case.gt_tid) if pose_pick is not None else None

    # Save annotated frame for visual page.
    visual_path = _save_annotated_frame(
        frame_bgr=frame_bgr,
        case=case,
        tracked=tracked,
        tid_to_kp=tid_to_kp,
        same_team_bbox=same_team_bbox,
        pose_wrist_xy=pose_wrist_xy,
        bbox_pick=bbox_pick,
        pose_pick=pose_pick,
    )

    print(f"    bbox_pick={bbox_pick} pose_pick={pose_pick} "
          f"pose_evaluable={pose_evaluable} "
          f"bbox_correct={bbox_correct} pose_correct={pose_correct}")
    print(f"    pose_wrist_dists: {pose_wrist_dists}")

    return CaseResult(
        idx=case.idx,
        kind=case.kind,
        video=case.video_name,
        rally_short=case.rally_short,
        pl_frame=case.pl_frame,
        gt_tid=case.gt_tid,
        pl_tid=case.pl_tid,
        bbox_pick=bbox_pick,
        pose_pick=pose_pick,
        pose_evaluable=pose_evaluable,
        bbox_correct=bbox_correct,
        pose_correct=pose_correct,
        same_team_bbox_dists={int(k): float(v) for k, v in same_team_bbox.items()},
        pose_wrist_dists={int(k): (float(v) if v is not None else None)
                          for k, v in pose_wrist_dists.items()},
        pose_wrist_xy=pose_wrist_xy,
        pose_wrist_which=pose_wrist_which,
        pose_inference_ms=pose_ms,
        visual_frame_path=str(visual_path.relative_to(REPORT_DIR)) if visual_path else None,
        error=None,
        reason=case.reason,
    )


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------

COLOR_TEAM_A = (255, 120, 50)
COLOR_TEAM_B = (60, 60, 235)
COLOR_BBOX_PICK = (0, 220, 220)     # cyan — bbox pick
COLOR_POSE_PICK = (40, 220, 40)     # green — pose pick
COLOR_GT = (255, 60, 255)           # magenta — GT
COLOR_BALL = (0, 255, 255)
COLOR_WRIST = (40, 220, 40)


def _team_color(team: str | None) -> tuple[int, int, int]:
    if team == "A":
        return COLOR_TEAM_A
    if team == "B":
        return COLOR_TEAM_B
    return (160, 160, 160)


def _save_annotated_frame(
    *,
    frame_bgr,
    case: ProbeCase,
    tracked: list[dict[str, Any]],
    tid_to_kp: dict[int, np.ndarray],
    same_team_bbox: dict[int, float],
    pose_wrist_xy: dict[int, list[float] | None],
    bbox_pick: int,
    pose_pick: int | None,
) -> Path | None:
    img = frame_bgr.copy()
    h, w = img.shape[:2]

    same_team_set = set(same_team_bbox.keys())

    # Draw all player bboxes (thin team-colored)
    for p in tracked:
        try:
            tid = int(p.get("trackId", -1))
        except (TypeError, ValueError):
            continue
        if tid < 0:
            continue
        cx = float(p.get("x", 0.0)) * w
        cy = float(p.get("y", 0.0)) * h
        bw = float(p.get("width", 0.0)) * w
        bh = float(p.get("height", 0.0)) * h
        x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
        x2, y2 = int(cx + bw / 2), int(cy + bh / 2)
        team = case.team_assignments.get(str(tid))
        color = _team_color(team)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Highlight same-team candidates
        if tid in same_team_set:
            # Stack overlays:
            #  - bbox_pick → cyan halo (offset -3)
            #  - pose_pick → green halo (offset -6)
            #  - gt_tid    → magenta thick (offset -9)
            offset = -3
            if tid == bbox_pick:
                cv2.rectangle(img, (x1 + offset, y1 + offset),
                              (x2 - offset, y2 - offset),
                              COLOR_BBOX_PICK, 3)
                offset -= 3
            if tid == pose_pick:
                cv2.rectangle(img, (x1 + offset, y1 + offset),
                              (x2 - offset, y2 - offset),
                              COLOR_POSE_PICK, 3)
                offset -= 3
            if case.gt_tid is not None and tid == case.gt_tid:
                cv2.rectangle(img, (x1 + offset, y1 + offset),
                              (x2 - offset, y2 - offset),
                              COLOR_GT, 3)

        label_parts = [f"p{tid}({team or '?'})"]
        if tid in same_team_set:
            label_parts.append("ST")
        if tid == bbox_pick:
            label_parts.append("bbox")
        if tid == pose_pick:
            label_parts.append("pose")
        if case.gt_tid is not None and tid == case.gt_tid:
            label_parts.append("GT")
        label = " ".join(label_parts)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ly = max(y1 - 6, 14)
        cv2.rectangle(img, (x1 - 2, ly - th - 4),
                      (x1 + tw + 4, ly + 2), (0, 0, 0), -1)
        cv2.putText(img, label, (x1, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Draw all wrist keypoints for same-team candidates
    for tid in same_team_set:
        wxy = pose_wrist_xy.get(tid)
        if wxy is None:
            continue
        wx, wy = int(wxy[0] * w), int(wxy[1] * h)
        cv2.circle(img, (wx, wy), 12, COLOR_WRIST, 2)
        cv2.circle(img, (wx, wy), 3, COLOR_WRIST, -1)
        # Connect wrist to ball with thin line for visibility
        bx, by = int(case.ball_xy[0] * w), int(case.ball_xy[1] * h)
        cv2.line(img, (wx, wy), (bx, by), COLOR_WRIST, 1, cv2.LINE_AA)
        # Label wrist with tid
        cv2.putText(img, f"w{tid}", (wx + 8, wy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WRIST, 1, cv2.LINE_AA)

    # Ball
    bx_n, by_n = case.ball_xy
    if bx_n > 0 or by_n > 0:
        bx, by = int(bx_n * w), int(by_n * h)
        cv2.circle(img, (bx, by), 16, COLOR_BALL, 3)
        cv2.circle(img, (bx, by), 2, COLOR_BALL, -1)

    # Title banner
    banner_h = 60
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.62, img, 0.38, 0, img)
    title = (f"#{case.idx} {case.kind.upper()}  {case.video_name}/{case.rally_short} "
             f"  ATTACK @ frame {case.pl_frame}")
    sub = (f"pl_tid={case.pl_tid}  gt_tid={case.gt_tid}  "
           f"bbox_pick=p{bbox_pick}  pose_pick=p{pose_pick}")
    cv2.putText(img, title, (12, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, sub, (12, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (210, 210, 210), 1, cv2.LINE_AA)

    # Legend
    legend = [
        ("bbox-pick (cyan halo)", COLOR_BBOX_PICK),
        ("pose-pick (green halo)", COLOR_POSE_PICK),
        ("GT (magenta halo)", COLOR_GT),
        ("ball (yellow)", COLOR_BALL),
        ("wrist→ball (green line)", COLOR_WRIST),
    ]
    lh = 18
    lw_box = 240
    ly0 = h - lh * len(legend) - 8
    overlay = img.copy()
    cv2.rectangle(overlay, (w - lw_box - 8, ly0 - 6),
                  (w - 4, h - 4), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    for i, (txt, col) in enumerate(legend):
        ly = ly0 + i * lh + 12
        cv2.rectangle(img, (w - lw_box, ly - 10),
                      (w - lw_box + 18, ly + 2), col, -1)
        cv2.putText(img, txt, (w - lw_box + 24, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)

    out_path = FRAMES_DIR / f"{case.idx:02d}_{case.video_name}_{case.rally_short}_f{case.pl_frame}.jpg"
    cv2.imwrite(str(out_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    print(f"    wrote {out_path.name}")
    return out_path


# ---------------------------------------------------------------------------
# Aggregation + reporting
# ---------------------------------------------------------------------------

def _aggregate(results: list[CaseResult]) -> dict[str, Any]:
    # Only GT-anchored cases count (idx 1..10)
    gt_anchored = [r for r in results if r.gt_tid is not None]
    pose_evaluable = [r for r in gt_anchored if r.pose_evaluable]

    bbox_correct_n = sum(1 for r in pose_evaluable if r.bbox_correct)
    pose_correct_n = sum(1 for r in pose_evaluable if r.pose_correct)
    pose_evaluable_n = len(pose_evaluable)

    # Disambiguation improvement is normalized by evaluable_n per spec.
    if pose_evaluable_n > 0:
        disambiguation = (pose_correct_n - bbox_correct_n) / pose_evaluable_n
    else:
        disambiguation = 0.0

    # Also report per-kind for transparency.
    by_kind: dict[str, dict[str, int]] = {}
    for r in gt_anchored:
        d = by_kind.setdefault(r.kind, {
            "n": 0, "evaluable": 0, "bbox_correct": 0, "pose_correct": 0
        })
        d["n"] += 1
        if r.pose_evaluable:
            d["evaluable"] += 1
            if r.bbox_correct:
                d["bbox_correct"] += 1
            if r.pose_correct:
                d["pose_correct"] += 1

    return {
        "n_total": len(results),
        "n_gt_anchored": len(gt_anchored),
        "n_pose_evaluable": pose_evaluable_n,
        "bbox_correct": bbox_correct_n,
        "pose_correct": pose_correct_n,
        "disambiguation_improvement": disambiguation,
        "ship_threshold": 0.5,
        "verdict": ("SHIP A2"
                    if disambiguation >= 0.5
                    else "NO-SHIP A2 (move to A3 or other)"),
        "by_kind": by_kind,
        "pose_inference_ms_mean": (
            sum(r.pose_inference_ms for r in results) / max(len(results), 1)
        ),
    }


def _write_markdown(results: list[CaseResult], agg: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# A2 Pose-Driven Attribution Probe — 2026-05-13")
    lines.append("")
    lines.append(f"Spec: `docs/superpowers/specs/2026-05-13-action-attribution-root-causes-design.md`")
    lines.append("")
    lines.append("## Aggregate")
    lines.append("")
    lines.append(f"- Total cases run: **{agg['n_total']}**")
    lines.append(f"- GT-anchored cases (idx 1–10): **{agg['n_gt_anchored']}**")
    lines.append(f"- Pose-evaluable subset (both same-team candidates with detected wrist): **{agg['n_pose_evaluable']}**")
    lines.append(f"- `bbox_correct` (within pose-evaluable subset): **{agg['bbox_correct']}** / {agg['n_pose_evaluable']}")
    lines.append(f"- `pose_correct` (within pose-evaluable subset): **{agg['pose_correct']}** / {agg['n_pose_evaluable']}")
    lines.append(f"- **Disambiguation improvement: `{agg['disambiguation_improvement']:.3f}`**  "
                 f"(threshold ≥ 0.5)")
    lines.append(f"- Mean pose inference latency: {agg['pose_inference_ms_mean']:.1f} ms / frame")
    lines.append("")
    lines.append(f"### Verdict: **{agg['verdict']}**")
    lines.append("")
    lines.append("### Breakdown by kind")
    lines.append("")
    lines.append("| kind | n | evaluable | bbox_correct | pose_correct |")
    lines.append("|------|---|-----------|--------------|--------------|")
    for k, v in agg["by_kind"].items():
        lines.append(f"| {k} | {v['n']} | {v['evaluable']} | "
                     f"{v['bbox_correct']} | {v['pose_correct']} |")
    lines.append("")
    lines.append("## Per-case detail")
    lines.append("")
    for r in results:
        lines.append(f"### #{r.idx} [{r.kind}] {r.video}/{r.rally_short} frame {r.pl_frame}")
        lines.append("")
        lines.append(f"- reason: {r.reason}")
        lines.append(f"- gt_tid: `{r.gt_tid}` | pl_tid: `{r.pl_tid}` | "
                     f"bbox_pick: `{r.bbox_pick}` | pose_pick: `{r.pose_pick}`")
        lines.append(f"- bbox_dists: `{r.same_team_bbox_dists}`")
        lines.append(f"- wrist_dists: `{r.pose_wrist_dists}`")
        lines.append(f"- wrist_which: `{r.pose_wrist_which}`")
        lines.append(f"- pose_evaluable: `{r.pose_evaluable}` | "
                     f"bbox_correct: `{r.bbox_correct}` | "
                     f"pose_correct: `{r.pose_correct}` | "
                     f"pose_ms: `{r.pose_inference_ms:.0f}`")
        if r.error:
            lines.append(f"- ERROR: {r.error}")
        lines.append("")
    MD_PATH.write_text("\n".join(lines))
    print(f"\nWrote markdown: {MD_PATH}")


# ---------------------------------------------------------------------------
# HTML visualization
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>A2 Pose Probe Visual Verification — 2026-05-13</title>
<style>
  :root {
    --bg:#0f0f10; --fg:#eaeaea; --muted:#9c9c9c; --card:#1a1a1c;
    --border:#2a2a2e; --accent:#5b8def;
    --ok:#2bb673; --bad:#e3534b; --warn:#f0a83b;
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
         line-height:1.45; }
  header { position:sticky; top:0; background:#000; border-bottom:1px solid #222;
           padding:14px 22px; display:flex; gap:18px; align-items:center; z-index:20; }
  header h1 { margin:0; font-size:17px; }
  header .status { color:var(--muted); font-size:13px; }
  header .grow { flex:1; }
  header .pill { padding:5px 12px; border-radius:12px; background:#1d1d1d;
                 color:#cfcfcf; font-size:12px;
                 font-family:ui-monospace,Menlo,monospace; }
  header .pill.ok   { background:#13361f; color:#9bdbb3; }
  header .pill.bad  { background:#3a1414; color:#f0a59f; }

  .container { max-width:1500px; margin:0 auto; padding:24px 22px 60px; }
  .agg { background:var(--card); border:1px solid var(--border); border-radius:10px;
         padding:18px 22px; margin-bottom:22px; }
  .agg dl { display:grid; grid-template-columns:auto 1fr; gap:6px 16px; margin:0; }
  .agg dt { color:var(--muted); }
  .agg dd { margin:0; font-family:ui-monospace,Menlo,monospace; }
  .verdict { margin-top:10px; font-size:18px; font-weight:700; }
  .verdict.ok  { color:var(--ok); }
  .verdict.bad { color:var(--bad); }

  .card { background:var(--card); border:1px solid var(--border); border-radius:10px;
          margin-bottom:22px; overflow:hidden; }
  .card-head { padding:13px 20px; display:flex; gap:12px; align-items:center;
               background:#161617; border-bottom:1px solid var(--border); }
  .card-head .num { font-size:20px; font-weight:700; color:var(--accent); min-width:34px; }
  .card-head .title { font-size:15px; font-weight:600; }
  .card-head .pill { font-size:11px; padding:3px 9px; border-radius:12px;
                     background:#222; color:#cfcfcf; }
  .card-head .pill.err  { background:#3a1414; color:#f0a59f; }
  .card-head .pill.ctrl { background:#13361f; color:#9bdbb3; }
  .card-head .pill.f3   { background:#2a210e; color:#f0c98a; }
  .card-head .pill.win  { background:#13361f; color:#9bdbb3; }
  .card-head .pill.tie  { background:#222; color:#cfcfcf; }
  .card-head .pill.lose { background:#3a1414; color:#f0a59f; }
  .card-head .grow { flex:1; }

  .card-body { display:grid; grid-template-columns:minmax(0,2.3fr) minmax(280px,1fr);
               gap:18px; padding:18px 20px; }
  @media (max-width:1000px) { .card-body { grid-template-columns:1fr; } }
  .frame { background:#000; border-radius:6px; overflow:hidden; }
  .frame img { display:block; width:100%; max-height:70vh; object-fit:contain;
               background:#000; cursor:zoom-in; }
  .frame.error { aspect-ratio:16/9; display:flex; align-items:center;
                 justify-content:center; color:var(--bad); padding:20px;
                 background:#1f0f0f; }

  .sidebar { font-size:13px; }
  .sidebar dl { margin:0; display:grid; grid-template-columns:auto 1fr;
                column-gap:10px; row-gap:5px; }
  .sidebar dt { color:var(--muted); }
  .sidebar dd { margin:0; font-family:ui-monospace,Menlo,monospace;
                font-size:12px; word-break:break-word; }
  .sidebar .group + .group { margin-top:12px; padding-top:12px;
                              border-top:1px dashed #2a2a2a; }

  .verdicts { display:grid; grid-template-columns:repeat(3,1fr); gap:8px;
              margin-top:16px; }
  .verdicts button { background:#1f1f22; color:#eee; border:2px solid #2a2a2e;
                     border-radius:8px; padding:14px 8px; font-size:14px;
                     font-weight:600; cursor:pointer; }
  .verdicts button:hover { transform:translateY(-1px); }
  .verdicts button.ok.sel   { border-color:var(--ok);   background:#0f261c; }
  .verdicts button.bad.sel  { border-color:var(--bad);  background:#291313; }
  .verdicts button.warn.sel { border-color:var(--warn); background:#2a210e; }

  #zoom-overlay { display:none; position:fixed; inset:0;
                  background:rgba(0,0,0,.94); z-index:99;
                  align-items:center; justify-content:center; cursor:zoom-out; }
  #zoom-overlay.open { display:flex; }
  #zoom-overlay img { max-width:96vw; max-height:96vh; }

  #copy-modal { display:none; position:fixed; inset:0;
                background:rgba(0,0,0,.7); z-index:99;
                align-items:center; justify-content:center; }
  #copy-modal.open { display:flex; }
  #copy-modal .box { background:#1a1a1c; border:1px solid #333; border-radius:10px;
                     padding:20px; max-width:680px; width:90%; }
  #copy-modal pre { background:#0a0a0a; color:#eaeaea; padding:14px;
                    border-radius:6px; font-size:14px; max-height:60vh;
                    overflow:auto; user-select:all; }
  #copy-modal .row { display:flex; gap:10px; margin-top:12px;
                     justify-content:flex-end; }
  #copy-modal button { background:#222; color:#eee; border:1px solid #333;
                       padding:8px 16px; border-radius:6px; cursor:pointer; }
  #copy-modal button.primary { background:var(--accent); color:#000;
                                border-color:transparent; }
</style>
</head>
<body>

<header>
  <h1>A2 Pose Probe — 11 attacks</h1>
  <span class="status" id="status">0 / 11 verdicts</span>
  <span class="grow"></span>
  <span class="pill" id="agg-pill">…</span>
  <button id="copy-btn" style="background:#222;color:#eee;border:1px solid #333;padding:8px 14px;border-radius:6px;font-weight:600;cursor:pointer;">Copy verdicts</button>
</header>

<div class="container">
  <div class="agg" id="agg-card"></div>
  <div id="cards"></div>
</div>

<div id="zoom-overlay"><img id="zoom-img"></div>

<div id="copy-modal">
  <div class="box">
    <h3 style="margin-top:0">Verdicts — copy and paste back to Claude</h3>
    <pre id="copy-text"></pre>
    <div class="row">
      <button id="copy-clipboard" class="primary">Copy to clipboard</button>
      <button id="copy-close">Close</button>
    </div>
  </div>
</div>

<script id="data" type="application/json">__DATA_JSON__</script>
<script>
const DATA = JSON.parse(document.getElementById('data').textContent);
const STORAGE_KEY = 'a2_pose_probe_2026_05_13';

function loadV() { try { return JSON.parse(localStorage.getItem(STORAGE_KEY)) || {}; } catch(_) { return {}; } }
function saveV(v) { localStorage.setItem(STORAGE_KEY, JSON.stringify(v)); }
let verdicts = loadV();
const $ = id => document.getElementById(id);

function emo(v) { return v==='ok'?'✅':v==='bad'?'❌':v==='warn'?'⚠️':'·'; }
function lbl(v) { return v==='ok'?'pose right':v==='bad'?'pose wrong':v==='warn'?'ambiguous':'—'; }

function aggHtml() {
  const a = DATA.agg;
  const verdict = a.verdict;
  const cls = a.disambiguation_improvement >= 0.5 ? 'ok' : 'bad';
  return `
    <dl>
      <dt>Total cases</dt><dd>${a.n_total}</dd>
      <dt>GT-anchored (idx 1–10)</dt><dd>${a.n_gt_anchored}</dd>
      <dt>Pose evaluable</dt><dd>${a.n_pose_evaluable} / ${a.n_gt_anchored}</dd>
      <dt>bbox correct</dt><dd>${a.bbox_correct} / ${a.n_pose_evaluable}</dd>
      <dt>pose correct</dt><dd>${a.pose_correct} / ${a.n_pose_evaluable}</dd>
      <dt>Disambiguation Δ</dt><dd>${a.disambiguation_improvement.toFixed(3)} (threshold ≥ 0.5)</dd>
      <dt>Mean inference</dt><dd>${a.pose_inference_ms_mean.toFixed(0)} ms / frame</dd>
    </dl>
    <div class="verdict ${cls}">Verdict: ${verdict}</div>`;
}

function cardHtml(r) {
  const v = verdicts[r.idx] || null;
  const kindPill = r.kind === 'error' ? 'err' : (r.kind === 'control' ? 'ctrl' : 'f3');
  let outcomePill = '';
  if (r.gt_tid !== null) {
    if (r.bbox_correct && !r.pose_correct) outcomePill = '<span class="pill lose">pose regressed</span>';
    else if (!r.bbox_correct && r.pose_correct) outcomePill = '<span class="pill win">pose fixed</span>';
    else if (r.bbox_correct && r.pose_correct)  outcomePill = '<span class="pill tie">both correct</span>';
    else if (r.pose_correct === null)           outcomePill = '<span class="pill tie">pose unevaluable</span>';
    else                                        outcomePill = '<span class="pill lose">both wrong</span>';
  } else {
    outcomePill = '<span class="pill tie">no GT (F3)</span>';
  }
  const frame = r.visual_frame_path ?
    `<div class="frame"><img src="${r.visual_frame_path}" onclick="zoomImg(this.src)"></div>` :
    `<div class="frame error">No frame: ${r.error || 'unknown'}</div>`;

  return `
    <div class="card" id="card-${r.idx}">
      <div class="card-head">
        <span class="num">${r.idx}</span>
        <span class="title">${r.video} / ${r.rally_short} / ATTACK f${r.pl_frame}</span>
        <span class="pill ${kindPill}">${r.kind}</span>
        ${outcomePill}
        <span class="grow"></span>
        <span class="pill">bbox p${r.bbox_pick} ➝ pose p${r.pose_pick === null ? '—' : r.pose_pick} ${r.gt_tid !== null ? '| GT p'+r.gt_tid : ''}</span>
      </div>
      <div class="card-body">
        ${frame}
        <div class="sidebar">
          <div class="group">
            <dl>
              <dt>video</dt><dd>${r.video}</dd>
              <dt>rally</dt><dd>${r.rally_short}</dd>
              <dt>frame</dt><dd>${r.pl_frame}</dd>
              <dt>pl_tid (current)</dt><dd>p${r.pl_tid}</dd>
              <dt>gt_tid</dt><dd>${r.gt_tid === null ? '—' : 'p'+r.gt_tid}</dd>
            </dl>
          </div>
          <div class="group">
            <dl>
              <dt>bbox dists</dt><dd>${JSON.stringify(r.same_team_bbox_dists)}</dd>
              <dt>wrist dists</dt><dd>${JSON.stringify(r.pose_wrist_dists)}</dd>
              <dt>wrist used</dt><dd>${JSON.stringify(r.pose_wrist_which)}</dd>
              <dt>pose_evaluable</dt><dd>${r.pose_evaluable}</dd>
              <dt>bbox_correct</dt><dd>${r.bbox_correct}</dd>
              <dt>pose_correct</dt><dd>${r.pose_correct}</dd>
              <dt>latency</dt><dd>${r.pose_inference_ms.toFixed(0)} ms</dd>
            </dl>
          </div>
          <div class="group">
            <dl>
              <dt>reason</dt><dd>${r.reason}</dd>
            </dl>
          </div>
          <div class="verdicts" data-idx="${r.idx}">
            <button class="ok ${v==='ok'?'sel':''}"   data-v="ok">✅ pose right</button>
            <button class="bad ${v==='bad'?'sel':''}" data-v="bad">❌ pose wrong</button>
            <button class="warn ${v==='warn'?'sel':''}" data-v="warn">⚠️ ambiguous</button>
          </div>
        </div>
      </div>
    </div>`;
}

function render() {
  $('agg-card').innerHTML = aggHtml();
  const a = DATA.agg;
  $('agg-pill').textContent =
    `Δ=${a.disambiguation_improvement.toFixed(2)} bbox=${a.bbox_correct} pose=${a.pose_correct} / ${a.n_pose_evaluable}`;
  $('agg-pill').className = 'pill ' + (a.disambiguation_improvement >= 0.5 ? 'ok' : 'bad');
  $('cards').innerHTML = DATA.cases.map(cardHtml).join('');
  document.querySelectorAll('.verdicts').forEach(div => {
    const idx = +div.dataset.idx;
    div.querySelectorAll('button').forEach(btn => {
      btn.addEventListener('click', () => setV(idx, btn.dataset.v));
    });
  });
  upd();
}

function setV(idx, v) {
  if (verdicts[idx] === v) delete verdicts[idx]; else verdicts[idx] = v;
  saveV(verdicts);
  const card = DATA.cases.find(c => c.idx === idx);
  const el = document.getElementById(`card-${idx}`);
  if (el && card) {
    const tmp = document.createElement('div');
    tmp.innerHTML = cardHtml(card);
    el.replaceWith(tmp.firstElementChild);
    document.querySelector(`[data-idx="${idx}"]`).querySelectorAll('button').forEach(b => {
      b.addEventListener('click', () => setV(idx, b.dataset.v));
    });
  }
  upd();
}

function upd() {
  const total = DATA.cases.length;
  let n = 0; for (const k in verdicts) if (verdicts[k]) n++;
  $('status').textContent = `${n} / ${total} verdicts`;
}

function compile() {
  const parts = [];
  for (const r of DATA.cases) parts.push(`${r.idx}:${emo(verdicts[r.idx])}`);
  return parts.join(' ') + '\n\n' + DATA.cases.map(r => {
    const v = verdicts[r.idx];
    return `  ${r.idx}. ${r.video}/${r.rally_short} f${r.pl_frame} [${r.kind}]: ${emo(v)} ${lbl(v)}`;
  }).join('\n');
}

$('copy-btn').addEventListener('click', () => {
  $('copy-text').textContent = compile();
  $('copy-modal').classList.add('open');
});
$('copy-close').addEventListener('click', () => $('copy-modal').classList.remove('open'));
$('copy-clipboard').addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText($('copy-text').textContent);
    $('copy-clipboard').textContent = 'Copied!';
    setTimeout(() => $('copy-clipboard').textContent = 'Copy to clipboard', 1200);
  } catch(e) { alert('Clipboard copy failed; select manually.'); }
});

function zoomImg(s) { $('zoom-img').src = s; $('zoom-overlay').classList.add('open'); }
$('zoom-overlay').addEventListener('click', () => $('zoom-overlay').classList.remove('open'));

render();
</script>
</body>
</html>
"""


def _write_html(results: list[CaseResult], agg: dict[str, Any]) -> None:
    payload = {
        "agg": agg,
        "cases": [asdict(r) for r in results],
    }
    html = _HTML_TEMPLATE.replace("__DATA_JSON__", json.dumps(payload, indent=2))
    HTML_PATH.write_text(html)
    print(f"Wrote HTML: {HTML_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== A2 Pose Attribution Probe ===")
    print(f"Report dir: {REPORT_DIR}")

    cases = _fetch_cases()
    print(f"\nLoaded {len(cases)} probe cases")

    resolver = VideoResolver()
    print(f"VideoResolver: endpoint={resolver.s3_endpoint} "
          f"bucket={resolver.bucket_name}")

    # Pre-load pose model so latency is honest on case 1.
    print("\nLoading yolo11s-pose…")
    _get_pose_model()
    print("Pose model ready.")

    results: list[CaseResult] = []
    for case in cases:
        try:
            r = _probe_one(case, resolver)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            r = CaseResult(
                idx=case.idx, kind=case.kind,
                video=case.video_name, rally_short=case.rally_short,
                pl_frame=case.pl_frame, gt_tid=case.gt_tid, pl_tid=case.pl_tid,
                bbox_pick=-1, pose_pick=None,
                pose_evaluable=False,
                bbox_correct=None, pose_correct=None,
                reason=case.reason,
                error=f"unexpected: {e}",
            )
        results.append(r)

    agg = _aggregate(results)
    print("\n=== Aggregate ===")
    print(json.dumps(agg, indent=2))

    JSON_PATH.write_text(json.dumps({
        "agg": agg,
        "cases": [asdict(r) for r in results],
    }, indent=2))
    print(f"\nWrote JSON: {JSON_PATH}")

    _write_markdown(results, agg)
    _write_html(results, agg)
    print(f"\nOpen visual page: open {HTML_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
