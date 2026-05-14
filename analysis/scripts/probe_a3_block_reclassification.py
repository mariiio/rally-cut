"""A3 BLOCK reclassification probe (2026-05-14).

Spec: ``docs/superpowers/specs/2026-05-13-action-attribution-root-causes-design.md``,
section A3.

Hypothesis: a contact predicted as ``ATTACK`` should be **reclassified as
BLOCK** iff ALL four conditions hold:

  (a) player at-net (court-y near net plane, or ball-y near net-y as fallback)
  (b) wrist above net-height (pose-keypoint Y < net image-y)
  (c) ball direction-change at contact ≤ 90° (deflected, not fully reversed)
  (d) prev action is opposing-team set/attack within the possession

Canonical target: **F5 — rally `99091ec6` (keke), frame 184 attack p2(A)**,
sandwiched between team-B receives. Note: F5's prev is ``receive(B)`` —
NOT set/attack — so it matches the LOOSE-(d) variant (cross-team prev,
any prev type) but NOT the STRICT-(d) variant. Both variants are
reported by the helper.

Probe protocol:
1. Pull all ATTACK actions fleet-wide.
2. For each, compute (a), (c), (d_strict), (d_loose) from cached DB state
   (no pose yet — wrist condition skipped for triage).
3. Triage candidates: a ∧ c ∧ d_loose (the superset most likely to need
   pose check).
4. Pick 10:
   - **F5 always** (mandatory, even if loose-(d) wasn't fired by the
     scan — it is, since the prev cross-team prerequisite holds).
   - 3-4 from top C-5 videos (titi, toto, juju, lili, lolo).
   - 2-3 from panel videos (gigi, cece, wawa) where GT exists.
   - Fill remaining slots from any-video matches.
5. For each picked candidate, run yolo11s-pose at the source frame,
   match to playerTrackId via bbox-IoU, take wrist-Y, evaluate (b).
6. Render annotated frame; emit visual HTML page; ask Claude to call
   each one block/attack/ambiguous.

Ship threshold (per spec): **≥ 7 / 10 actual blocks → SHIP A3.**

Outputs:
- analysis/reports/a3_block_reclass_probe/probe_results_2026_05_13.json
- analysis/reports/a3_block_reclass_probe/probe_results_2026_05_13.md
- analysis/reports/a3_block_reclass_probe/visual_2026_05_13.html
- analysis/reports/a3_block_reclass_probe/visual_frames/*.jpg

Usage:
    cd analysis
    uv run python scripts/probe_a3_block_reclassification.py
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
from rallycut.tracking.block_reclassification import (  # noqa: E402
    CandidateInputs,
    CandidateVerdict,
    estimate_net_y_image,
    evaluate_candidate,
    project_image_to_court,
)

REPORT_DIR = HERE.parent / "reports" / "a3_block_reclass_probe"
FRAMES_DIR = REPORT_DIR / "visual_frames_v2"
JSON_PATH = REPORT_DIR / "probe_results_v2_2026_05_14.json"
MD_PATH = REPORT_DIR / "probe_results_v2_2026_05_14.md"
HTML_PATH = REPORT_DIR / "visual_v2_2026_05_14.html"

# COCO 17-keypoint indices (same as A2 probe)
KPT_LEFT_WRIST = 9
KPT_RIGHT_WRIST = 10

# Confidence floor for considering a wrist detected.
WRIST_CONF_FLOOR = 0.30
# IoU threshold for matching pose detection to a tracked player bbox.
POSE_TRACK_IOU_THRESHOLD = 0.30

# Target candidate count.
TARGET_N = 10

# Top C-5 videos from analysis/reports/coherence_c5_baseline_2026_05_13.md.
TOP_C5_VIDEOS = ("titi", "toto", "juju", "lili", "lolo")

# Panel-GT videos.
PANEL_VIDEOS = ("gigi", "cece", "wawa")

# F5 canonical case (mandatory).
F5_VIDEO = "keke"
F5_RALLY_SHORT = "99091ec6"
F5_FRAME = 184


# ---------------------------------------------------------------------------
# Probe-set structures
# ---------------------------------------------------------------------------


@dataclass
class TriageCandidate:
    """Fleet-scan result: a single ATTACK that passes a∧c∧d_loose."""

    video_name: str
    video_id: str
    rally_id: str
    rally_short: str
    pl_frame: int
    pl_tid: int
    pl_team: str
    prev_frame: int | None
    prev_action: str | None
    prev_team: str | None
    prev_pid: int | None
    direction_change_deg: float
    ball_xy: tuple[float, float]
    is_at_net: bool                # from contact.isAtNet (legacy hint, not authoritative)
    rally_start_ms: int
    team_assignments: dict[str, str]
    confidence: float
    a_passes: bool
    a_source: str
    c_passes: bool
    d_strict_passes: bool
    d_loose_passes: bool
    is_f5: bool
    # v2 additions
    head_y_image: float | None = None
    net_y_image_video: float | None = None


@dataclass
class CaseResult:
    """Per-case probe output (after pose inference + annotated frame)."""

    idx: int
    is_f5: bool
    video: str
    rally_short: str
    rally_id: str
    pl_frame: int
    pl_tid: int
    pl_team: str
    prev_frame: int | None
    prev_action: str | None
    prev_team: str | None
    prev_pid: int | None
    direction_change_deg: float
    confidence: float
    # v2 picker labels
    variant: str            # "strict" | "loose-only" (where loose-only = passes loose but NOT strict)
    confidence_tier: str    # "strong" | "moderate" | "weak" | "none"
    # Condition outcomes
    a_passes: bool
    a_source: str
    b_passes: bool | None       # None = wrist not detected
    b_reason: str
    c_passes: bool
    d_strict_passes: bool
    d_strict_reason: str
    d_loose_passes: bool
    d_loose_reason: str
    all_pass_strict: bool
    all_pass_loose: bool
    # v2 picker fields
    selected_strict: bool
    selected_loose: bool
    # Geometry
    player_court_xy: tuple[float, float] | None
    head_y_image: float | None
    net_y_image: float | None
    wrist_y_image: float | None
    wrist_xy_image: tuple[float, float] | None
    wrist_which: str | None
    ball_xy: tuple[float, float]
    # Source-video time
    source_frame: int
    fps: float
    source_time_sec: float
    source_time_str: str
    rally_start_ms: int
    # Visual
    visual_frame_path: str | None
    error: str | None
    # Claude vision verdict (filled in Phase 4)
    claude_verdict: str | None = None     # "block" | "attack" | "ambiguous"
    claude_verdict_note: str | None = None
    # Other
    same_team_tracked: list[int] = field(default_factory=list)
    pose_inference_ms: float = 0.0
    team_assignments: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# DB scan
# ---------------------------------------------------------------------------


def _scan_fleet() -> list[TriageCandidate]:
    """Walk every undeleted video's rallies and surface ATTACK actions
    that pass v2 (a)′ ∧ (c) ∧ (d_loose-v2).

    v2 (a)′ uses the player's bbox-TOP (head, from ``positions_json``)
    against the video's net image-y (computed from
    ``court_calibration_json``). Falls back to ball-y proximity to net
    image-y when bbox-top or calibration is missing.

    v2 (d) loose: prev cross-team AND prev NOT a serve.

    Adds ``head_y_image`` to each TriageCandidate for downstream use.
    """
    out: list[TriageCandidate] = []
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT v.id, v.name, v.court_calibration_json,
                   r.id, r.start_ms,
                   pt.actions_json, pt.contacts_json, pt.positions_json
            FROM videos v
            JOIN rallies r ON r.video_id = v.id
            JOIN player_tracks pt ON pt.rally_id = r.id
            WHERE v.deleted_at IS NULL
            ORDER BY v.name, r.start_ms
            """
        )
        rows = cur.fetchall()

    print(f"Scanning {len(rows)} rallies fleet-wide (v2 a′ head-near-net)…")
    net_y_by_video: dict[str, float | None] = {}
    for (
        video_id, video_name, court_cal, rally_id, start_ms,
        actions_json, contacts_json, positions_json,
    ) in rows:
        actions_json = actions_json or {}
        contacts_json = contacts_json or {}
        positions_json = positions_json or []

        if video_id not in net_y_by_video:
            net_y_by_video[video_id] = estimate_net_y_image(court_cal)
        net_y_image = net_y_by_video[video_id]

        actions = sorted(
            actions_json.get("actions", []) or [],
            key=lambda a: int(a.get("frame", 0)),
        )
        team_ass = actions_json.get("teamAssignments", {}) or {}
        contacts_by_frame: dict[int, dict[str, Any]] = {}
        for c in contacts_json.get("contacts", []) or []:
            try:
                contacts_by_frame[int(c.get("frame", -1))] = c
            except (TypeError, ValueError):
                continue

        # Index positions by (frame, trackId) for O(1) lookup.
        positions_by_key: dict[tuple[int, int], dict[str, Any]] = {}
        for p in positions_json:
            try:
                fn = int(p.get("frameNumber", -1))
                tid = int(p.get("trackId", -1))
            except (TypeError, ValueError):
                continue
            if fn < 0 or tid < 0:
                continue
            positions_by_key[(fn, tid)] = p

        for i, action in enumerate(actions):
            if str(action.get("action", "")) != "attack":
                continue
            frame = int(action.get("frame", -1))
            if frame < 0:
                continue
            pl_tid = action.get("playerTrackId")
            if pl_tid is None or int(pl_tid) < 0:
                continue
            pl_tid = int(pl_tid)
            pl_team = action.get("team")
            if pl_team in (None, "unknown"):
                continue

            # Direction change at this frame (from contact).
            c = contacts_by_frame.get(frame)
            if not c:
                continue
            dc = c.get("directionChangeDeg")
            if dc is None:
                continue
            dc = float(dc)

            ball_x = float(action.get("ballX") or 0.0)
            ball_y = float(action.get("ballY") or 0.0)

            # (c) check
            if dc > 90.0:
                continue

            # v2 (a)′ check — head-near-ball + ball-in-net-region.
            # Mirror block_reclassification.check_a_at_net_v2 semantics.
            pos = positions_by_key.get((frame, pl_tid))
            head_y_image: float | None = None
            if pos is not None:
                cy = float(pos.get("y", 0.0))
                bh = float(pos.get("height", 0.0))
                head_y_image = cy - bh / 2.0  # bbox-top in normalized image-y
            a_pass = False
            a_source_v2 = "no-data"
            HEAD_BALL_BAND = 0.15
            BALL_BELOW_NET = 0.05
            BALL_ABOVE_NET = 0.35
            if (head_y_image is not None and net_y_image is not None):
                ball_dy = ball_y - net_y_image
                ball_in_net = -BALL_ABOVE_NET <= ball_dy <= BALL_BELOW_NET
                head_near_ball = abs(head_y_image - ball_y) <= HEAD_BALL_BAND
                a_pass = ball_in_net and head_near_ball
                a_source_v2 = "head-near-ball-at-net"
            elif net_y_image is not None:
                ball_dy = ball_y - net_y_image
                a_pass = -BALL_ABOVE_NET <= ball_dy <= BALL_BELOW_NET
                a_source_v2 = "ball-y-fallback"
            else:
                a_pass = abs(ball_y - 0.5) <= 0.20
                a_source_v2 = "ball-y-nominal"
            if not a_pass:
                continue

            # v2 (d_loose) check — cross-team, prev NOT serve.
            prev = actions[i - 1] if i > 0 else None
            if prev is None:
                continue
            prev_team = prev.get("team")
            if prev_team in (None, "unknown"):
                continue
            if prev_team == pl_team:
                continue  # same-team → not crossover candidate
            prev_action_type = str(prev.get("action", ""))
            if prev_action_type == "serve":
                continue  # v2 loose excludes prev=serve
            d_strict = prev_action_type in {"attack", "set"}

            is_f5 = (
                video_name == F5_VIDEO
                and str(rally_id).startswith(F5_RALLY_SHORT)
                and frame == F5_FRAME
            )

            out.append(TriageCandidate(
                video_name=video_name,
                video_id=video_id,
                rally_id=rally_id,
                rally_short=str(rally_id).split("-", 1)[0],
                pl_frame=frame,
                pl_tid=pl_tid,
                pl_team=str(pl_team),
                prev_frame=int(prev.get("frame", -1)),
                prev_action=prev_action_type,
                prev_team=str(prev_team),
                prev_pid=(int(prev.get("playerTrackId")) if prev.get("playerTrackId") is not None else None),
                direction_change_deg=dc,
                ball_xy=(ball_x, ball_y),
                is_at_net=bool(c.get("isAtNet", False)),
                rally_start_ms=int(start_ms or 0),
                team_assignments={str(k): str(v) for k, v in team_ass.items()},
                confidence=float(action.get("confidence") or 0.0),
                a_passes=True,             # confirmed at this point
                a_source=a_source_v2,
                c_passes=True,
                d_strict_passes=d_strict,
                d_loose_passes=True,
                is_f5=is_f5,
                head_y_image=head_y_image,
                net_y_image_video=net_y_image,
            ))

    return out


# ---------------------------------------------------------------------------
# Candidate selection (10 picks from the triage set)
# ---------------------------------------------------------------------------


def _pick_candidates(
    triage: list[TriageCandidate],
    target_n: int = TARGET_N,
) -> list[TriageCandidate]:
    """Pick exactly 10 candidates for the v2 probe:

    - **F5 always** (mandatory; it's loose-only by definition since its
      prev=receive).
    - **5 strict picks** (d_strict=True) — stratified across distinct
      videos, diversified by dc° (different deflection regimes).
    - **5 loose-only picks** (d_loose=True ∧ d_strict=False) including
      F5 — stratified across distinct videos.

    Stratification: at most 1 candidate per video per bucket. Diversity:
    rank-by-dc° (low/mid/high) within each video bucket.
    """
    picked: list[TriageCandidate] = []
    seen_keys: set[tuple[str, int]] = set()

    def _key(c: TriageCandidate) -> tuple[str, int]:
        return (c.rally_id, c.pl_frame)

    # 1. F5 (mandatory, loose-only).
    f5_candidates = [c for c in triage if c.is_f5]
    if f5_candidates:
        for c in f5_candidates:
            if _key(c) not in seen_keys:
                picked.append(c)
                seen_keys.add(_key(c))
                break
    else:
        print(f"WARNING: F5 ({F5_VIDEO}/{F5_RALLY_SHORT} f{F5_FRAME}) not in triage. "
              f"Cannot include as a pick.")

    strict_pool = [c for c in triage if c.d_strict_passes and _key(c) not in seen_keys]
    loose_only_pool = [
        c for c in triage
        if not c.d_strict_passes and c.d_loose_passes and _key(c) not in seen_keys
    ]

    def _spread_pick(pool: list[TriageCandidate], n_target: int) -> list[TriageCandidate]:
        """Pick n_target items from pool with one-per-video preference,
        diversified by dc°. Within a video, pick the candidate whose dc°
        is farthest from already-picked dc°'s (variety)."""
        out: list[TriageCandidate] = []
        by_vid: dict[str, list[TriageCandidate]] = {}
        for c in pool:
            by_vid.setdefault(c.video_name, []).append(c)
        chosen_dcs: list[float] = []
        # Round 1: 1 per video, prefer videos NOT already represented in `picked`.
        existing_videos = {p.video_name for p in picked}

        def _video_priority(vname: str) -> int:
            # Lower = higher priority. Underrepresented videos first.
            return 0 if vname not in existing_videos else 1
        ordered_videos = sorted(by_vid.keys(), key=lambda v: (_video_priority(v), v))
        for vname in ordered_videos:
            if len(out) >= n_target:
                break
            bucket = by_vid[vname]
            if not bucket:
                continue
            # Pick candidate maximizing min-distance to existing chosen_dcs
            # (diversity). Tie-break: lower confidence (more likely to be a
            # mis-typed attack).
            def _score(c: TriageCandidate) -> tuple[float, float]:
                if not chosen_dcs:
                    return (-c.direction_change_deg, c.confidence)
                d = min(abs(c.direction_change_deg - x) for x in chosen_dcs)
                return (-d, c.confidence)
            best = min(bucket, key=_score)
            if _key(best) not in seen_keys:
                out.append(best)
                seen_keys.add(_key(best))
                chosen_dcs.append(best.direction_change_deg)

        # Round 2: fill from any remaining (allow same-video) by diversity.
        remaining = [c for c in pool if _key(c) not in seen_keys]
        while len(out) < n_target and remaining:
            def _score2(c: TriageCandidate) -> tuple[float, float]:
                if not chosen_dcs:
                    return (-c.direction_change_deg, c.confidence)
                d = min(abs(c.direction_change_deg - x) for x in chosen_dcs)
                return (-d, c.confidence)
            best = min(remaining, key=_score2)
            out.append(best)
            seen_keys.add(_key(best))
            chosen_dcs.append(best.direction_change_deg)
            remaining = [c for c in remaining if _key(c) not in seen_keys]
        return out

    # 2. 5 strict picks.
    strict_picks = _spread_pick(strict_pool, 5)
    picked.extend(strict_picks)

    # 3. 5 loose-only picks (F5 already counted; need 4 more loose-only).
    loose_needed = target_n - len(picked)
    loose_picks = _spread_pick(loose_only_pool, loose_needed)
    picked.extend(loose_picks)

    return picked[:target_n]


# ---------------------------------------------------------------------------
# DB queries per-case (video meta, positions, court calibration)
# ---------------------------------------------------------------------------


def _fetch_video_meta(video_id: str) -> dict[str, Any]:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT id, name, filename, fps, width, height,
                      s3_key, proxy_s3_key, processed_s3_key, content_hash,
                      court_calibration_json
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
        "court_calibration_json": r[10],
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
# Video / pose (mirrors A2 probe)
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
        return None, 0, 0, 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
    ok, frame = cap.read()
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.release()
    if not ok or frame is None:
        return None, img_w, img_h, fps
    return frame, img_w, img_h, fps


_POSE_MODEL = None


def _get_pose_model():
    global _POSE_MODEL
    if _POSE_MODEL is None:
        from ultralytics import YOLO
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
    """Run yolo11s-pose; match detections to tracks by IoU.

    Returns {track_id: (17, 3) keypoints in normalized 0-1 coords}.
    """
    pose = _get_pose_model()
    results = pose.predict(frame_bgr, verbose=False, imgsz=1280)  # type: ignore[attr-defined]
    if not results:
        return {}
    res = results[0]
    if res.keypoints is None or res.boxes is None:
        return {}
    kps_all = res.keypoints.data.cpu().numpy()
    boxes_pix = res.boxes.xyxy.cpu().numpy()

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
                kp = kps_all[det_idx].copy()
                kp[:, 0] /= img_w
                kp[:, 1] /= img_h
                best_kp = kp
        if best_kp is not None:
            out[tid] = best_kp
    return out


def _higher_wrist(
    kp: np.ndarray,
    conf_floor: float = WRIST_CONF_FLOOR,
) -> tuple[float | None, str | None, tuple[float, float] | None]:
    """Return the wrist with the smallest image-y (highest in frame).

    (image-y is inverted — smaller = higher.)
    """
    best_y: float | None = None
    best_which: str | None = None
    best_xy: tuple[float, float] | None = None
    for idx, name in ((KPT_LEFT_WRIST, "left"), (KPT_RIGHT_WRIST, "right")):
        wx, wy, wc = float(kp[idx, 0]), float(kp[idx, 1]), float(kp[idx, 2])
        if wc < conf_floor:
            continue
        if best_y is None or wy < best_y:
            best_y = wy
            best_which = name
            best_xy = (wx, wy)
    return best_y, best_which, best_xy


# ---------------------------------------------------------------------------
# Per-case probe
# ---------------------------------------------------------------------------


def _hms(seconds: float) -> str:
    m = int(seconds // 60)
    s = seconds - m * 60
    return f"{m}:{s:05.2f}"


def _probe_one(
    idx: int,
    tc: TriageCandidate,
    resolver: VideoResolver,
) -> CaseResult:
    print(f"\n[{idx}/10] {tc.video_name}/{tc.rally_short} f{tc.pl_frame} "
          f"pl=p{tc.pl_tid}({tc.pl_team}) "
          f"prev=f{tc.prev_frame} {tc.prev_action}({tc.prev_team}) "
          f"dc={tc.direction_change_deg:.1f}° "
          f"d_strict={tc.d_strict_passes} f5={tc.is_f5}")

    vm = _fetch_video_meta(tc.video_id)
    if not vm:
        return _error_case(idx, tc, "video meta missing")

    video_path = _resolve_video(resolver, vm)
    if not video_path:
        return _error_case(idx, tc, "could not resolve video")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return _error_case(idx, tc, "cv2 cannot open video")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or vm.get("fps", 30.0))
    cap.release()
    rally_start_frame = int(round(tc.rally_start_ms / 1000.0 * fps))
    source_frame = rally_start_frame + tc.pl_frame
    source_time_sec = source_frame / fps
    print(f"    fps={fps:.3f} rally_start_frame={rally_start_frame} "
          f"source_frame={source_frame} time={_hms(source_time_sec)}")

    frame_bgr, img_w, img_h, _ = _load_frame(video_path, source_frame)
    if frame_bgr is None:
        return _error_case(idx, tc, "failed to read frame")

    cal = vm.get("court_calibration_json")
    net_y_image = estimate_net_y_image(cal)
    print(f"    court_calibration: {'present' if cal else 'missing'}, "
          f"net_y_image={net_y_image}")

    # Locate player bbox at this frame, compute head-y (v2 (a)′), and
    # also keep court projection for diagnostics.
    tracked = _fetch_positions_at_frame(tc.rally_id, tc.pl_frame)
    player_pos = None
    for p in tracked:
        if int(p.get("trackId", -1)) == tc.pl_tid:
            player_pos = p
            break
    player_court_xy: tuple[float, float] | None = None
    head_y_image: float | None = None
    feet_y_image: float | None = None
    if player_pos is not None:
        cx, cy = float(player_pos.get("x", 0.0)), float(player_pos.get("y", 0.0))
        bh = float(player_pos.get("height", 0.0))
        head_y_image = cy - bh / 2.0
        feet_y_image = cy + bh / 2.0
        # Court projection (diagnostic only in v2).
        player_court_xy = project_image_to_court((cx, feet_y_image), cal)
        print(f"    player p{tc.pl_tid} bbox-center=({cx:.3f}, {cy:.3f}) "
              f"head_y={head_y_image:.3f} feet_y={feet_y_image:.3f} "
              f"court_xy={player_court_xy}")
    else:
        print(f"    WARNING: no position for p{tc.pl_tid} at frame {tc.pl_frame}")

    # Build pl_action + prev_action dicts (compatible with helper).
    pl_action = {
        "frame": tc.pl_frame, "action": "attack",
        "team": tc.pl_team,
        "playerTrackId": tc.pl_tid,
    }
    prev_action: dict[str, Any] | None = None
    if tc.prev_action is not None:
        prev_action = {
            "frame": tc.prev_frame, "action": tc.prev_action,
            "team": tc.prev_team,
            "playerTrackId": tc.prev_pid,
        }

    # Pose inference for wrist-y.
    t0 = time.perf_counter()
    tid_to_kp = _run_pose_on_frame(frame_bgr, img_w, img_h, tracked)
    pose_ms = (time.perf_counter() - t0) * 1000.0
    print(f"    pose inference: {pose_ms:.0f} ms, "
          f"matched_tids={sorted(tid_to_kp.keys())}")

    wrist_y_image: float | None = None
    wrist_xy_image: tuple[float, float] | None = None
    wrist_which: str | None = None
    pl_kp = tid_to_kp.get(tc.pl_tid)
    if pl_kp is not None:
        wy, which, xy = _higher_wrist(pl_kp)
        wrist_y_image = wy
        wrist_xy_image = xy
        wrist_which = which
    print(f"    wrist_y_image={wrist_y_image} ({wrist_which}) "
          f"net_y_image={net_y_image}")

    # Build helper inputs (v2: passes head_y_image, feet_y_image).
    inputs = CandidateInputs(
        action_index=-1,
        action=pl_action,
        prev_action=prev_action,
        team_assignments=tc.team_assignments,
        player_court_xy=player_court_xy,
        net_y_image=net_y_image,
        wrist_y_image=wrist_y_image,
        direction_change_deg=tc.direction_change_deg,
        ball_y_image=tc.ball_xy[1],
        player_bbox_top_y_image=head_y_image,
        player_bbox_bottom_y_image=feet_y_image,
    )
    v = evaluate_candidate(inputs)
    print(f"    verdict: a={v.a_passes}({v.a_source}) "
          f"b={v.b_passes}({v.b_reason}) "
          f"c={v.c_passes} d_strict={v.d_strict_passes}({v.d_strict_reason}) "
          f"d_loose={v.d_loose_passes}({v.d_loose_reason}) "
          f"selected_strict={v.selected_strict} selected_loose={v.selected_loose} "
          f"confidence={v.confidence}")

    same_team_tracked = sorted({
        int(p.get("trackId", -1)) for p in tracked
        if tc.team_assignments.get(str(int(p.get("trackId", -1)))) == tc.pl_team
        and int(p.get("trackId", -1)) >= 0
    })

    visual_path = _save_annotated_frame(
        frame_bgr=frame_bgr,
        tc=tc,
        idx=idx,
        tracked=tracked,
        tid_to_kp=tid_to_kp,
        verdict=v,
        net_y_image=net_y_image,
        wrist_xy_image=wrist_xy_image,
        source_time_str=_hms(source_time_sec),
        player_court_xy=player_court_xy,
        head_y_image=head_y_image,
    )

    variant = "strict" if tc.d_strict_passes else "loose-only"
    return CaseResult(
        idx=idx,
        is_f5=tc.is_f5,
        video=tc.video_name,
        rally_short=tc.rally_short,
        rally_id=tc.rally_id,
        pl_frame=tc.pl_frame,
        pl_tid=tc.pl_tid,
        pl_team=tc.pl_team,
        prev_frame=tc.prev_frame,
        prev_action=tc.prev_action,
        prev_team=tc.prev_team,
        prev_pid=tc.prev_pid,
        direction_change_deg=tc.direction_change_deg,
        confidence=tc.confidence,
        variant=variant,
        confidence_tier=v.confidence,
        a_passes=v.a_passes,
        a_source=v.a_source,
        b_passes=v.b_passes,
        b_reason=v.b_reason,
        c_passes=v.c_passes,
        d_strict_passes=v.d_strict_passes,
        d_strict_reason=v.d_strict_reason,
        d_loose_passes=v.d_loose_passes,
        d_loose_reason=v.d_loose_reason,
        all_pass_strict=v.all_pass_strict,
        all_pass_loose=v.all_pass_loose,
        selected_strict=v.selected_strict,
        selected_loose=v.selected_loose,
        player_court_xy=player_court_xy,
        head_y_image=head_y_image,
        net_y_image=net_y_image,
        wrist_y_image=wrist_y_image,
        wrist_xy_image=wrist_xy_image,
        wrist_which=wrist_which,
        ball_xy=tc.ball_xy,
        source_frame=source_frame,
        fps=fps,
        source_time_sec=source_time_sec,
        source_time_str=_hms(source_time_sec),
        rally_start_ms=tc.rally_start_ms,
        visual_frame_path=(
            str(visual_path.relative_to(REPORT_DIR)) if visual_path else None
        ),
        error=None,
        same_team_tracked=same_team_tracked,
        pose_inference_ms=pose_ms,
        team_assignments=tc.team_assignments,
    )


def _error_case(idx: int, tc: TriageCandidate, msg: str) -> CaseResult:
    print(f"    ERROR: {msg}")
    variant = "strict" if tc.d_strict_passes else "loose-only"
    return CaseResult(
        idx=idx, is_f5=tc.is_f5, video=tc.video_name,
        rally_short=tc.rally_short, rally_id=tc.rally_id,
        pl_frame=tc.pl_frame, pl_tid=tc.pl_tid, pl_team=tc.pl_team,
        prev_frame=tc.prev_frame, prev_action=tc.prev_action,
        prev_team=tc.prev_team, prev_pid=tc.prev_pid,
        direction_change_deg=tc.direction_change_deg, confidence=tc.confidence,
        variant=variant, confidence_tier="none",
        a_passes=False, a_source="error", b_passes=None, b_reason=msg,
        c_passes=True, d_strict_passes=tc.d_strict_passes, d_strict_reason="error",
        d_loose_passes=tc.d_loose_passes, d_loose_reason="error",
        all_pass_strict=False, all_pass_loose=False,
        selected_strict=False, selected_loose=False,
        player_court_xy=None, head_y_image=None, net_y_image=None,
        wrist_y_image=None, wrist_xy_image=None, wrist_which=None,
        ball_xy=tc.ball_xy, source_frame=0, fps=30.0,
        source_time_sec=0.0, source_time_str="0:00.00",
        rally_start_ms=tc.rally_start_ms,
        visual_frame_path=None, error=msg,
        team_assignments=tc.team_assignments,
    )


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------

COLOR_TEAM_A = (255, 120, 50)   # BGR — orange-ish for team A
COLOR_TEAM_B = (60, 60, 235)    # red-ish for team B
COLOR_SUSPECT = (0, 165, 255)   # bright orange — the ATTACK we'd reclassify
COLOR_BALL = (0, 255, 255)      # yellow
COLOR_WRIST = (40, 220, 40)     # green
COLOR_NET = (200, 200, 200)     # light gray net line
COLOR_PASS = (40, 220, 40)
COLOR_FAIL = (40, 40, 220)


def _team_color(team: str | None) -> tuple[int, int, int]:
    if team == "A":
        return COLOR_TEAM_A
    if team == "B":
        return COLOR_TEAM_B
    return (160, 160, 160)


def _save_annotated_frame(
    *,
    frame_bgr,
    tc: TriageCandidate,
    idx: int,
    tracked: list[dict[str, Any]],
    tid_to_kp: dict[int, np.ndarray],
    verdict: CandidateVerdict,
    net_y_image: float | None,
    wrist_xy_image: tuple[float, float] | None,
    source_time_str: str,
    player_court_xy: tuple[float, float] | None,
    head_y_image: float | None = None,
) -> Path | None:
    img = frame_bgr.copy()
    h, w = img.shape[:2]

    # Net line + ±band (v2 a′ window) if available.
    if net_y_image is not None:
        nly = int(net_y_image * h)
        cv2.line(img, (0, nly), (w, nly), COLOR_NET, 2, cv2.LINE_AA)
        cv2.putText(img, "NET", (10, max(nly - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_NET, 1, cv2.LINE_AA)
        # Shaded ±5% band as a hint to (a)′.
        band_px = int(0.05 * h)
        overlay = img.copy()
        cv2.rectangle(overlay, (0, max(0, nly - band_px)),
                      (w, min(h, nly + band_px)), (200, 200, 200), -1)
        cv2.addWeighted(overlay, 0.10, img, 0.90, 0, img)

    # Player bboxes
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
        team = tc.team_assignments.get(str(tid))
        color = _team_color(team)

        if tid == tc.pl_tid:
            # Suspect — thick orange outline.
            cv2.rectangle(img, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2),
                          COLOR_SUSPECT, 4)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        label = f"p{tid}({team or '?'})"
        if tid == tc.pl_tid:
            label += " SUSPECT"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ly = max(y1 - 6, 14)
        cv2.rectangle(img, (x1 - 2, ly - th - 4),
                      (x1 + tw + 4, ly + 2), (0, 0, 0), -1)
        cv2.putText(img, label, (x1, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Wrist + line from wrist to ball.
    bx_n, by_n = tc.ball_xy
    bx, by = int(bx_n * w), int(by_n * h)
    if wrist_xy_image is not None:
        wx, wy = int(wrist_xy_image[0] * w), int(wrist_xy_image[1] * h)
        cv2.circle(img, (wx, wy), 12, COLOR_WRIST, 2)
        cv2.circle(img, (wx, wy), 3, COLOR_WRIST, -1)
        cv2.line(img, (wx, wy), (bx, by), COLOR_WRIST, 1, cv2.LINE_AA)
        cv2.putText(img, "wrist", (wx + 10, wy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WRIST, 1, cv2.LINE_AA)

    # Ball
    if bx_n > 0 or by_n > 0:
        cv2.circle(img, (bx, by), 16, COLOR_BALL, 3)
        cv2.circle(img, (bx, by), 2, COLOR_BALL, -1)
        cv2.putText(img, "ball", (bx + 18, by + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_BALL, 1, cv2.LINE_AA)

    # Head marker (v2 a′)
    if head_y_image is not None:
        # Mark head position with a thin orange line across the suspect bbox top
        # so the reader can compare head-y vs net-y band.
        suspect_pos = next(
            (p for p in tracked if int(p.get("trackId", -1)) == tc.pl_tid), None)
        if suspect_pos is not None:
            cx = float(suspect_pos.get("x", 0.0)) * w
            bw = float(suspect_pos.get("width", 0.0)) * w
            hy = int(head_y_image * h)
            cv2.line(img, (int(cx - bw / 2 - 8), hy),
                     (int(cx + bw / 2 + 8), hy),
                     (60, 200, 230), 2, cv2.LINE_AA)
            cv2.putText(img, "head", (int(cx + bw / 2 + 12), hy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 200, 230), 1,
                        cv2.LINE_AA)

    # Title banner
    banner_h = 112
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.62, img, 0.38, 0, img)
    variant_tag = "STRICT" if tc.d_strict_passes else "LOOSE-only"
    title = (f"#{idx} BLOCK CANDIDATE  {tc.video_name}/{tc.rally_short}  "
             f"ATTACK @ f{tc.pl_frame}  [{variant_tag}]")
    if tc.is_f5:
        title += "  [F5 canonical]"
    cv2.putText(img, title, (12, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)

    def _badge(cond: bool | None, label: str) -> str:
        if cond is True:
            return f"{label}=YES"
        if cond is False:
            return f"{label}=no"
        return f"{label}=?"

    # (a)' details
    a_detail = ""
    if head_y_image is not None and net_y_image is not None:
        a_detail = (f"|head-net|={abs(head_y_image - net_y_image):.3f}"
                    f" (band 0.050)")
    sub_a = (f"{_badge(verdict.a_passes, '(a) at-net')} ({verdict.a_source}) {a_detail}")
    cv2.putText(img, sub_a, (12, 47),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1, cv2.LINE_AA)
    sub_b = (
        f"{_badge(verdict.b_passes, '(b) wrist>net')}  ({verdict.b_reason})  "
        f"(c) dc={tc.direction_change_deg:.0f}°(<=90:{verdict.c_passes})  "
        f"conf={verdict.confidence}"
    )
    cv2.putText(img, sub_b, (12, 71),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1, cv2.LINE_AA)
    sub2 = (f"(d) prev=f{tc.prev_frame} {tc.prev_action}({tc.prev_team})  "
            f"strict={verdict.d_strict_passes}  loose={verdict.d_loose_passes}  "
            f"sel_strict={verdict.selected_strict}  sel_loose={verdict.selected_loose}  "
            f"src_time={source_time_str}")
    cv2.putText(img, sub2, (12, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)

    # Legend
    legend = [
        ("suspect (orange thick)", COLOR_SUSPECT),
        ("team A", COLOR_TEAM_A),
        ("team B", COLOR_TEAM_B),
        ("ball (yellow)", COLOR_BALL),
        ("wrist → ball (green)", COLOR_WRIST),
        ("net line (gray)", COLOR_NET),
    ]
    lh = 18
    lw_box = 260
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

    out_path = (
        FRAMES_DIR
        / f"{idx:02d}_{tc.video_name}_{tc.rally_short}_f{tc.pl_frame}.jpg"
    )
    cv2.imwrite(str(out_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    print(f"    wrote {out_path.name}")
    return out_path


# ---------------------------------------------------------------------------
# Markdown + HTML
# ---------------------------------------------------------------------------


def _write_markdown(results: list[CaseResult], triage_count: int,
                    strict_count: int) -> None:
    lines: list[str] = []
    lines.append("# A3 BLOCK Reclassification Probe v2 — 2026-05-14")
    lines.append("")
    lines.append("Spec: `docs/superpowers/specs/2026-05-13-action-attribution-root-causes-design.md`")
    lines.append("")
    lines.append("## v2 refinements")
    lines.append("")
    lines.append("- (a)′ HEAD-near-net in image coords (replaces v1 feet-court projection).")
    lines.append("- (b)′ soft pose-fallback — Unknown counts as soft-yes when (a)+(c)+(d) all confirm.")
    lines.append("- (d) loose: prev cross-team AND prev NOT a serve.")
    lines.append("")
    lines.append("## Fleet scan (v2)")
    lines.append("")
    lines.append(f"- v2 triage (a′ ∧ c ∧ d_loose-v2): **{triage_count}**")
    lines.append(f"- v2 strict (a′ ∧ c ∧ d_strict): **{strict_count}**")
    lines.append("")
    lines.append("> Note on (d): the design spec says `prev.action ∈ {attack, set}` "
                 "(strict). The canonical F5 case has `prev=receive`, which fails "
                 "strict-(d). We report both variants. F5 is included as a "
                 "loose-only pick.")
    lines.append("")
    lines.append("## Candidates picked (10)")
    lines.append("")
    lines.append("| # | F5 | variant | conf-tier | video | rally | frame | src_time | suspect | prev | dc° | conf | sel_strict | sel_loose |")
    lines.append("|---|----|---------|-----------|-------|-------|-------|----------|---------|------|-----|------|------------|-----------|")
    for r in results:
        f5 = "★" if r.is_f5 else " "
        suspect = f"p{r.pl_tid}({r.pl_team})"
        prev = f"f{r.prev_frame} {r.prev_action}({r.prev_team})" if r.prev_action else "—"
        lines.append(
            f"| {r.idx} | {f5} | {r.variant} | {r.confidence_tier} | "
            f"{r.video} | {r.rally_short} | {r.pl_frame} | "
            f"{r.source_time_str} | {suspect} | {prev} | "
            f"{r.direction_change_deg:.0f} | {r.confidence:.2f} | "
            f"{r.selected_strict} | {r.selected_loose} |"
        )
    lines.append("")
    lines.append("## Per-case detail")
    lines.append("")
    for r in results:
        lines.append(f"### #{r.idx} {r.video}/{r.rally_short} f{r.pl_frame} {'(F5)' if r.is_f5 else ''}")
        lines.append("")
        lines.append(f"- variant: **{r.variant}** confidence: **{r.confidence_tier}**")
        lines.append(f"- suspect: p{r.pl_tid}({r.pl_team}) conf={r.confidence:.3f}")
        lines.append(f"- prev: frame {r.prev_frame} `{r.prev_action}`({r.prev_team}) p{r.prev_pid}")
        lines.append(f"- direction_change_deg: `{r.direction_change_deg:.1f}` (≤ 90: `{r.c_passes}`)")
        lines.append(f"- player_court_xy (feet-projection, diagnostic): `{r.player_court_xy}`")
        lines.append(f"- head_y_image: `{r.head_y_image}`")
        lines.append(f"- net_y_image: `{r.net_y_image}`")
        if r.head_y_image is not None and r.net_y_image is not None:
            lines.append(f"- |head - net| = `{abs(r.head_y_image - r.net_y_image):.4f}` "
                         f"(band: 0.050; pass: `{abs(r.head_y_image - r.net_y_image) <= 0.05}`)")
        lines.append(f"- (a)′ pass: `{r.a_passes}` source=`{r.a_source}`")
        lines.append(f"- wrist_y_image: `{r.wrist_y_image}` (which=`{r.wrist_which}`)")
        lines.append(f"- (b)′ wrist-above-net: `{r.b_passes}` (`{r.b_reason}`)")
        lines.append(f"- (d) strict: `{r.d_strict_passes}` (`{r.d_strict_reason}`)")
        lines.append(f"- (d) loose:  `{r.d_loose_passes}` (`{r.d_loose_reason}`)")
        lines.append(f"- selected_strict: `{r.selected_strict}`  selected_loose: `{r.selected_loose}`")
        lines.append(f"- source-video time: **{r.source_time_str}** "
                     f"(source_frame={r.source_frame}, fps={r.fps:.3f}, "
                     f"rally_start_ms={r.rally_start_ms})")
        if r.error:
            lines.append(f"- ERROR: {r.error}")
        lines.append("")
    MD_PATH.write_text("\n".join(lines))
    print(f"Wrote markdown: {MD_PATH}")


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>A3 BLOCK Reclass Probe — 2026-05-13</title>
<style>
  :root {
    --bg:#0f0f10; --fg:#eaeaea; --muted:#9c9c9c; --card:#1a1a1c;
    --border:#2a2a2e; --accent:#f0a83b;
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
  .card-head .pill.f5 { background:#2a210e; color:#f0c98a; }
  .card-head .pill.strict { background:#13361f; color:#9bdbb3; }
  .card-head .pill.loose-only { background:#2a210e; color:#f0c98a; }
  .card-head .pill.neither { background:#3a1414; color:#f0a59f; }
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
  .cond-pass { color:#9bdbb3; }
  .cond-fail { color:#f0a59f; }
  .cond-unk  { color:#f0c98a; }

  .verdicts { display:grid; grid-template-columns:repeat(3,1fr); gap:8px;
              margin-top:16px; }
  .verdicts button { background:#1f1f22; color:#eee; border:2px solid #2a2a2e;
                     border-radius:8px; padding:14px 8px; font-size:14px;
                     font-weight:600; cursor:pointer; }
  .verdicts button:hover { transform:translateY(-1px); }
  .verdicts button.block.sel { border-color:#5b8def;   background:#0f1c33; }
  .verdicts button.attack.sel { border-color:var(--warn); background:#2a210e; }
  .verdicts button.ambig.sel  { border-color:#888;        background:#222; }

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
  <h1>A3 BLOCK Reclass Probe — 10 candidates</h1>
  <span class="status" id="status">0 / 10 verdicts</span>
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
    <h3 style="margin-top:0">Verdicts — copy/paste back to Claude</h3>
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
const STORAGE_KEY = 'a3_block_reclass_probe_2026_05_13';

function loadV() { try { return JSON.parse(localStorage.getItem(STORAGE_KEY)) || {}; } catch(_) { return {}; } }
function saveV(v) { localStorage.setItem(STORAGE_KEY, JSON.stringify(v)); }
let verdicts = loadV();
const $ = id => document.getElementById(id);

function emo(v) { return v==='block'?'🟦':v==='attack'?'🟧':v==='ambig'?'⚠️':'·'; }
function lbl(v) { return v==='block'?'actually block':v==='attack'?'actually attack':v==='ambig'?'ambiguous':'—'; }

function condIcon(b) { if (b===true) return '<span class="cond-pass">YES</span>'; if (b===false) return '<span class="cond-fail">no</span>'; return '<span class="cond-unk">?</span>'; }

function aggHtml() {
  const a = DATA.agg;
  const claudeTally = a.claude_tally || {block:0, attack:0, ambig:0, unset:0};
  const claudeShip = a.claude_block_count >= 7 ? 'SHIP A3' : 'NO-SHIP A3';
  const claudeCls = a.claude_block_count >= 7 ? 'ok' : 'bad';
  const bv = a.by_variant || {};
  const bc = a.by_confidence_tier || {};
  const rowV = (k) => bv[k] ? `${bv[k].block}🟦 ${bv[k].attack}🟧 ${bv[k].ambig}⚠️ (n=${bv[k].n})` : '—';
  const rowC = (k) => bc[k] ? `${bc[k].block}🟦 ${bc[k].attack}🟧 ${bc[k].ambig}⚠️ (n=${bc[k].n})` : '—';
  return `
    <dl>
      <dt>v2 triage (a′ ∧ c ∧ d_loose-v2) fleet count</dt><dd>${a.triage_count}</dd>
      <dt>v2 strict (a′ ∧ c ∧ d_strict) fleet count</dt><dd>${a.strict_count}</dd>
      <dt>Candidates picked</dt><dd>${DATA.cases.length} (strict=${a.n_strict_picks}, loose-only=${a.n_loose_only_picks})</dd>
      <dt>selected_strict picks</dt><dd>${a.selected_strict_count}</dd>
      <dt>selected_loose picks</dt><dd>${a.selected_loose_count}</dd>
      <dt>Variant: strict picks</dt><dd>${rowV('strict')}</dd>
      <dt>Variant: loose-only picks</dt><dd>${rowV('loose-only')}</dd>
      <dt>Confidence: strong</dt><dd>${rowC('strong')}</dd>
      <dt>Confidence: moderate</dt><dd>${rowC('moderate')}</dd>
      <dt>Confidence: weak</dt><dd>${rowC('weak')}</dd>
      <dt>Confidence: none</dt><dd>${rowC('none')}</dd>
      <dt>Claude vision: 🟦 block</dt><dd>${claudeTally.block}</dd>
      <dt>Claude vision: 🟧 attack</dt><dd>${claudeTally.attack}</dd>
      <dt>Claude vision: ⚠️ ambiguous</dt><dd>${claudeTally.ambig}</dd>
      <dt>Claude vision: unset</dt><dd>${claudeTally.unset}</dd>
    </dl>
    <div class="verdict ${claudeCls}">Claude first-pass verdict: ${claudeShip} (${a.claude_block_count} / 10 blocks; threshold ≥ 7)</div>`;
}

function cardHtml(r) {
  const v = verdicts[r.idx] || null;
  let bucketPill;
  if (r.variant === 'strict') bucketPill = '<span class="pill strict">STRICT-(d)</span>';
  else bucketPill = '<span class="pill loose-only">LOOSE-only-(d)</span>';
  const confPill = `<span class="pill">conf: ${r.confidence_tier}</span>`;
  const selPill = (r.selected_strict || r.selected_loose)
    ? '<span class="pill strict">selected</span>'
    : '<span class="pill neither">not-selected</span>';
  const f5Pill = r.is_f5 ? '<span class="pill f5">F5 canonical</span>' : '';
  const claudePill = r.claude_verdict
    ? `<span class="pill">${emo(r.claude_verdict)} Claude: ${lbl(r.claude_verdict)}</span>`
    : '';

  const frame = r.visual_frame_path ?
    `<div class="frame"><img src="${r.visual_frame_path}" onclick="zoomImg(this.src)"></div>` :
    `<div class="frame error">No frame: ${r.error || 'unknown'}</div>`;

  return `
    <div class="card" id="card-${r.idx}">
      <div class="card-head">
        <span class="num">${r.idx}</span>
        <span class="title">${r.video} / ${r.rally_short} / ATTACK f${r.pl_frame} (p${r.pl_tid} ${r.pl_team})</span>
        ${f5Pill}
        ${bucketPill}
        ${confPill}
        ${selPill}
        ${claudePill}
        <span class="grow"></span>
        <span class="pill">${r.source_time_str}</span>
      </div>
      <div class="card-body">
        ${frame}
        <div class="sidebar">
          <div class="group">
            <dl>
              <dt>video</dt><dd>${r.video}</dd>
              <dt>rally</dt><dd>${r.rally_short}</dd>
              <dt>frame</dt><dd>${r.pl_frame}</dd>
              <dt>source frame</dt><dd>${r.source_frame} @ ${r.fps.toFixed(2)} fps</dd>
              <dt>source time</dt><dd>${r.source_time_str}</dd>
              <dt>suspect</dt><dd>p${r.pl_tid} (${r.pl_team})</dd>
              <dt>confidence</dt><dd>${r.confidence.toFixed(3)}</dd>
            </dl>
          </div>
          <div class="group">
            <dl>
              <dt>prev frame</dt><dd>${r.prev_frame}</dd>
              <dt>prev action</dt><dd>${r.prev_action} (p${r.prev_pid} ${r.prev_team})</dd>
              <dt>direction_change_deg</dt><dd>${r.direction_change_deg.toFixed(1)}°</dd>
              <dt>player court (x,y)</dt><dd>${r.player_court_xy ? '['+r.player_court_xy[0].toFixed(2)+', '+r.player_court_xy[1].toFixed(2)+']' : '—'}</dd>
              <dt>net_y_image</dt><dd>${r.net_y_image !== null ? r.net_y_image.toFixed(3) : '—'}</dd>
              <dt>wrist_y_image</dt><dd>${r.wrist_y_image !== null ? r.wrist_y_image.toFixed(3) : '—'} (${r.wrist_which || '—'})</dd>
            </dl>
          </div>
          <div class="group">
            <dl>
              <dt>(a) at-net</dt><dd>${condIcon(r.a_passes)} (${r.a_source})</dd>
              <dt>(b) wrist > net</dt><dd>${condIcon(r.b_passes)} (${r.b_reason})</dd>
              <dt>(c) dc ≤ 90°</dt><dd>${condIcon(r.c_passes)}</dd>
              <dt>(d) strict</dt><dd>${condIcon(r.d_strict_passes)} (${r.d_strict_reason})</dd>
              <dt>(d) loose</dt><dd>${condIcon(r.d_loose_passes)} (${r.d_loose_reason})</dd>
            </dl>
          </div>
          ${r.claude_verdict ? `<div class="group"><dl><dt>Claude vision call</dt><dd>${emo(r.claude_verdict)} ${lbl(r.claude_verdict)}</dd>${r.claude_verdict_note ? '<dt>note</dt><dd>'+r.claude_verdict_note+'</dd>' : ''}</dl></div>` : ''}
          <div class="verdicts" data-idx="${r.idx}">
            <button class="block ${v==='block'?'sel':''}"   data-v="block">🟦 actually a block</button>
            <button class="attack ${v==='attack'?'sel':''}" data-v="attack">🟧 actually an attack</button>
            <button class="ambig ${v==='ambig'?'sel':''}"   data-v="ambig">⚠️ ambiguous</button>
          </div>
        </div>
      </div>
    </div>`;
}

function render() {
  $('agg-card').innerHTML = aggHtml();
  const a = DATA.agg;
  $('agg-pill').textContent =
    `claude blocks: ${a.claude_block_count} / 10 (need ≥7)`;
  $('agg-pill').className = 'pill ' + (a.claude_block_count >= 7 ? 'ok' : 'bad');
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
    return `  ${r.idx}. ${r.video}/${r.rally_short} f${r.pl_frame}: ${emo(v)} ${lbl(v)}`;
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


def _aggregate(results: list[CaseResult], triage_count: int,
               strict_count: int) -> dict[str, Any]:
    claude_tally = {"block": 0, "attack": 0, "ambig": 0, "unset": 0}
    # Per-variant breakdown.
    by_variant = {"strict": {"n": 0, "block": 0, "attack": 0, "ambig": 0, "unset": 0},
                  "loose-only": {"n": 0, "block": 0, "attack": 0, "ambig": 0, "unset": 0}}
    by_conf = {"strong": {"n": 0, "block": 0, "attack": 0, "ambig": 0, "unset": 0},
               "moderate": {"n": 0, "block": 0, "attack": 0, "ambig": 0, "unset": 0},
               "weak": {"n": 0, "block": 0, "attack": 0, "ambig": 0, "unset": 0},
               "none": {"n": 0, "block": 0, "attack": 0, "ambig": 0, "unset": 0}}
    for r in results:
        v = r.claude_verdict if r.claude_verdict in claude_tally else "unset"
        claude_tally[v] += 1
        if r.variant in by_variant:
            by_variant[r.variant]["n"] += 1
            by_variant[r.variant][v] += 1
        if r.confidence_tier in by_conf:
            by_conf[r.confidence_tier]["n"] += 1
            by_conf[r.confidence_tier][v] += 1
    return {
        "triage_count": triage_count,
        "strict_count": strict_count,
        "all_pass_strict_count": sum(1 for r in results if r.all_pass_strict),
        "all_pass_loose_count": sum(1 for r in results if r.all_pass_loose),
        "selected_strict_count": sum(1 for r in results if r.selected_strict),
        "selected_loose_count": sum(1 for r in results if r.selected_loose),
        "n_total": len(results),
        "n_strict_picks": by_variant["strict"]["n"],
        "n_loose_only_picks": by_variant["loose-only"]["n"],
        "by_variant": by_variant,
        "by_confidence_tier": by_conf,
        "claude_tally": claude_tally,
        "claude_block_count": claude_tally["block"],
        "ship_threshold": 7,
        "verdict": (
            "SHIP A3" if claude_tally["block"] >= 7 else "NO-SHIP A3"
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    print("=== A3 BLOCK Reclassification Probe ===")
    print(f"Report dir: {REPORT_DIR}")

    triage = _scan_fleet()
    strict_only = [c for c in triage if c.d_strict_passes]
    print(f"\nTriage (a∧c∧d_loose): {len(triage)} ATTACK actions")
    print(f"Strict (a∧c∧d_strict): {len(strict_only)} ATTACK actions")

    f5_in_triage = [c for c in triage if c.is_f5]
    print(f"F5 in triage: {len(f5_in_triage)}")

    if not triage:
        print("\nNo candidates in triage — A3 has no targets fleet-wide.")
        return 1

    picked = _pick_candidates(triage)
    print(f"\nPicked {len(picked)} candidates:")
    for i, c in enumerate(picked, 1):
        print(f"  [{i}] {c.video_name}/{c.rally_short} f{c.pl_frame} "
              f"d_strict={c.d_strict_passes} f5={c.is_f5}")

    print("\nLoading yolo11s-pose…")
    _get_pose_model()
    print("Pose model ready.")

    resolver = VideoResolver()
    print(f"VideoResolver: endpoint={resolver.s3_endpoint} "
          f"bucket={resolver.bucket_name}")

    results: list[CaseResult] = []
    for i, tc in enumerate(picked, 1):
        try:
            r = _probe_one(i, tc, resolver)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            r = _error_case(i, tc, f"unexpected: {e}")
        results.append(r)

    agg = _aggregate(results, len(triage), len(strict_only))
    print("\n=== Aggregate ===")
    print(json.dumps(agg, indent=2))

    JSON_PATH.write_text(json.dumps({
        "agg": agg,
        "cases": [asdict(r) for r in results],
    }, indent=2))
    print(f"\nWrote JSON: {JSON_PATH}")

    _write_markdown(results, len(triage), len(strict_only))
    _write_html(results, agg)
    print(f"\nOpen visual page: open {HTML_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
