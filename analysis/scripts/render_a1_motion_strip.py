"""Render multi-frame motion strips for the 10 A1 (volleyball-rule) catalog fixes.

For each case, extract source-video frames at offsets [-6, -3, 0, +3, +6] around
both the prev contact and the curr contact (10 frames per case). Annotate each:
  - All player bboxes (team colored, thin)
  - Ball position (yellow circle) when available
  - Tight zoom-crop bounding the action area (players + ball, with padding)
  - PREV strip:
      * red halo on the BEFORE prev pid (pipeline output)
      * green halo on the AFTER prev pid (A1's edit) if A1 changed prev
  - CURR strip:
      * cyan halo on the BEFORE curr pid (pipeline output)
      * green halo on the AFTER curr pid (A1's edit) if A1 changed curr

Composite each case into a single image:
  Top row: 5 prev frames horizontally.
  Bottom row: 5 curr frames horizontally.
  Header strip explaining the case.

Output:
  analysis/reports/a1_volleyball_rule/motion_strips_2026_05_14/{N:02d}_{video}_{rally_short}.jpg

Usage:
    cd analysis
    uv run python scripts/render_a1_motion_strip.py
    uv run python scripts/render_a1_motion_strip.py --only 4   # just case #4
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from rallycut.evaluation.tracking.db import get_connection
from rallycut.evaluation.video_resolver import VideoResolver

HERE = Path(__file__).resolve().parent
REPORT_DIR = HERE.parent / "reports" / "a1_volleyball_rule"
OUT_DIR = REPORT_DIR / "motion_strips_2026_05_14"


# 10 A1 fixes (idx, video_name, rally_id_short, video_id, rally_id, curr_frame,
#              prev_frame, prev_action_type, curr_action_type,
#              pipeline_prev_pid, pipeline_curr_pid, a1_prev_pid, a1_curr_pid,
#              prev_team, curr_team, alt_ratio).
# pipeline_*_pid: original pid before A1.
# a1_*_pid: A1's proposed pid AFTER edit (None means A1 didn't change that side).
A1_CASES: list[dict[str, Any]] = [
    {
        "idx": 1, "video_name": "juju", "rally_short": "d810943e",
        "video_id": "06f0b063-b3f9-40aa-b13b-fba1edd74a85",
        "rally_id": "d810943e-dd1c-4518-a6af-577b22555c3c",
        "prev_frame": 252, "curr_frame": 345,
        "prev_action_type": "DIG", "curr_action_type": "SET",
        "pipeline_prev_pid": 4, "pipeline_curr_pid": 4,
        "a1_prev_pid": None, "a1_curr_pid": 3,
        "prev_team": "?", "curr_team": "?",
        "alt_ratio": 2.7,
    },
    {
        "idx": 2, "video_name": "gigi", "rally_short": "72c8229b",
        "video_id": "b097dd2a-6953-4e0e-a603-5be3552f462e",
        "rally_id": "72c8229b-2993-4310-9b61-cd6162cc27fa",
        "prev_frame": 357, "curr_frame": 399,
        "prev_action_type": "SET", "curr_action_type": "ATTACK",
        "pipeline_prev_pid": 3, "pipeline_curr_pid": 3,
        "a1_prev_pid": 4, "a1_curr_pid": None,
        "prev_team": "B", "curr_team": "B",
        "alt_ratio": 2.7,
    },
    {
        "idx": 3, "video_name": "caca", "rally_short": "ae99ab2a",
        "video_id": "627c1add-8a80-42ab-8278-3617880ebf81",
        "rally_id": "ae99ab2a-e342-4096-9225-6cfbb3909d15",
        "prev_frame": 241, "curr_frame": 353,
        "prev_action_type": "RECEIVE", "curr_action_type": "SET",
        "pipeline_prev_pid": 1, "pipeline_curr_pid": 1,
        "a1_prev_pid": None, "a1_curr_pid": 2,
        "prev_team": "A", "curr_team": "A",
        "alt_ratio": 4.6,
    },
    {
        "idx": 4, "video_name": "machi", "rally_short": "f406f4b3",
        "video_id": "23a5f798-78a0-4b3a-8647-b4a2166274b1",
        "rally_id": "f406f4b3-95df-474a-aecc-a5ba7239ad9b",
        "prev_frame": 242, "curr_frame": 333,
        "prev_action_type": "SERVE", "curr_action_type": "ATTACK",
        "pipeline_prev_pid": 1, "pipeline_curr_pid": 1,
        "a1_prev_pid": None, "a1_curr_pid": 2,
        "prev_team": "A", "curr_team": "A",
        "alt_ratio": 4.5,
    },
    {
        "idx": 5, "video_name": "matchop", "rally_short": "f433967e",
        "video_id": "920ba69d-2526-4e6c-a357-c44af3bf5c99",
        "rally_id": "f433967e-2c40-4169-b5cb-87f48cd0fa63",
        "prev_frame": 83, "curr_frame": 100,
        "prev_action_type": "RECEIVE", "curr_action_type": "RECEIVE",
        "pipeline_prev_pid": 4, "pipeline_curr_pid": 4,
        "a1_prev_pid": 3, "a1_curr_pid": None,
        "prev_team": "A", "curr_team": "A",
        "alt_ratio": 5.4,
    },
    {
        "idx": 6, "video_name": "lala", "rally_short": "2eeb3ae6",
        "video_id": "84e66e74-8d4f-420a-ad01-0ada95153ad0",
        "rally_id": "2eeb3ae6-cf97-4eeb-9400-28a8060a7636",
        "prev_frame": 569, "curr_frame": 616,
        "prev_action_type": "DIG", "curr_action_type": "DIG",
        "pipeline_prev_pid": 1, "pipeline_curr_pid": 1,
        "a1_prev_pid": None, "a1_curr_pid": 2,
        "prev_team": "B", "curr_team": "A",
        "alt_ratio": 8.5,
    },
    {
        "idx": 7, "video_name": "mech", "rally_short": "b0dabe43",
        "video_id": "c6e4c876-beca-4cb8-9cce-4a4fc70553f1",
        "rally_id": "b0dabe43-7ddb-4544-8d2c-e86032a8d8f5",
        "prev_frame": 39, "curr_frame": 59,
        "prev_action_type": "SERVE", "curr_action_type": "RECEIVE",
        "pipeline_prev_pid": 3, "pipeline_curr_pid": 3,
        "a1_prev_pid": None, "a1_curr_pid": 4,
        "prev_team": "A", "curr_team": "A",
        "alt_ratio": 19.7,
    },
    {
        "idx": 8, "video_name": "veve", "rally_short": "4c27b635",
        "video_id": "43928971-2e07-4814-bb1a-3d91c7bf03b2",
        "rally_id": "4c27b635-fbab-4bcb-a30e-f82a87c223c2",
        "prev_frame": 265, "curr_frame": 284,
        "prev_action_type": "ATTACK", "curr_action_type": "DIG",
        "pipeline_prev_pid": 3, "pipeline_curr_pid": 3,
        "a1_prev_pid": None, "a1_curr_pid": 4,
        "prev_team": "B", "curr_team": "B",
        "alt_ratio": 15.2,
    },
    {
        "idx": 9, "video_name": "natch", "rally_short": "e5e4c0b7",
        "video_id": "a7ee3d38-a3a9-4dcd-a2af-e0617997e708",
        "rally_id": "e5e4c0b7-7f18-493f-b95b-574e51821452",
        "prev_frame": 110, "curr_frame": 214,
        "prev_action_type": "RECEIVE", "curr_action_type": "ATTACK",
        "pipeline_prev_pid": 1, "pipeline_curr_pid": 1,
        "a1_prev_pid": None, "a1_curr_pid": 2,
        "prev_team": "A", "curr_team": "A",
        "alt_ratio": 2.0,
    },
    {
        "idx": 10, "video_name": "matttch", "rally_short": "8d3205ed",
        "video_id": "23b662ba-99e0-47d6-a9ac-90bb6fa9bdd1",
        "rally_id": "8d3205ed-b0dc-4c0c-bc24-fda34554e45f",
        "prev_frame": 220, "curr_frame": 296,
        "prev_action_type": "ATTACK", "curr_action_type": "ATTACK",
        "pipeline_prev_pid": 3, "pipeline_curr_pid": 3,
        "a1_prev_pid": 4, "a1_curr_pid": None,
        "prev_team": "B", "curr_team": "B",
        "alt_ratio": 2.4,
    },
]


OFFSETS = [-6, -3, 0, 3, 6]
CROP_W = 600
CROP_H = 400
PAD_PX = 80  # padding around the players+ball bbox before zoom

# BGR colors
COLOR_PIPELINE_CURR = (255, 200, 60)   # cyan-ish (BEFORE on curr)
COLOR_PIPELINE_PREV = (60, 60, 255)    # red (BEFORE on prev)
COLOR_A1_AFTER = (60, 220, 60)         # green (AFTER on whichever side A1 flipped)
COLOR_TEAM_A = (200, 120, 50)
COLOR_TEAM_B = (80, 80, 200)
COLOR_OTHER = (160, 160, 160)
COLOR_BALL = (0, 255, 255)


def _fetch_video_meta(video_id: str) -> dict[str, Any]:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT id, name, filename, fps, width, height,
                      s3_key, proxy_s3_key, processed_s3_key, content_hash
               FROM videos WHERE id = %s""",
            (video_id,),
        )
        r = cur.fetchone()
    if not r:
        return {}
    return {
        "id": r[0], "name": r[1], "filename": r[2],
        "fps": float(r[3]) if r[3] is not None else 30.0,
        "width": int(r[4] or 0), "height": int(r[5] or 0),
        "s3_key": r[6], "proxy_s3_key": r[7],
        "processed_s3_key": r[8], "content_hash": r[9],
    }


def _resolve_video(resolver: VideoResolver, vm: dict[str, Any]) -> Path | None:
    for key_name in ("proxy_s3_key", "s3_key", "processed_s3_key"):
        sk = vm.get(key_name)
        if not sk:
            continue
        try:
            return resolver.resolve(sk, vm["content_hash"])
        except Exception:
            continue
    return None


def _fetch_rally_state(
    rally_id: str,
) -> tuple[
    list[dict[str, Any]],
    dict[int, dict[int, tuple[float, float, float, float]]],
    dict[int, tuple[float, float]],
    dict[str, str],
    int,
]:
    """Returns (actions_list, positions_by_frame, ball_by_frame, team_assignments, rally_start_ms)."""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT pt.positions_json, pt.actions_json, pt.ball_positions_json, "
            "       r.start_ms "
            "FROM player_tracks pt JOIN rallies r ON r.id = pt.rally_id "
            "WHERE pt.rally_id = %s",
            (rally_id,),
        )
        r = cur.fetchone()
    if not r:
        return [], {}, {}, {}, 0
    positions_json, actions_json, ball_json, start_ms = r

    actions_list: list[dict[str, Any]] = []
    team_assignments: dict[str, str] = {}
    if actions_json:
        actions_list = actions_json.get("actions", []) or []
        team_assignments = actions_json.get("teamAssignments", {}) or {}

    positions_by_frame: dict[int, dict[int, tuple[float, float, float, float]]] = {}
    for p in positions_json or []:
        try:
            f = int(p.get("frameNumber", -1))
            tid = int(p.get("trackId", -1))
        except (TypeError, ValueError):
            continue
        positions_by_frame.setdefault(f, {})[tid] = (
            float(p.get("x", 0.0)),
            float(p.get("y", 0.0)),
            float(p.get("width", 0.0)),
            float(p.get("height", 0.0)),
        )

    ball_by_frame: dict[int, tuple[float, float]] = {}
    for bp in ball_json or []:
        try:
            f = int(bp["frameNumber"])
            bx = float(bp.get("x", 0.0)); by = float(bp.get("y", 0.0))
        except (KeyError, TypeError, ValueError):
            continue
        if bx <= 0 and by <= 0:
            continue
        ball_by_frame[f] = (bx, by)

    return actions_list, positions_by_frame, ball_by_frame, team_assignments, int(start_ms or 0)


def _team_color(team: str | None) -> tuple[int, int, int]:
    if team == "A":
        return COLOR_TEAM_A
    if team == "B":
        return COLOR_TEAM_B
    return COLOR_OTHER


def _extract_frame(video_path: Path, source_frame: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return frame


def _find_action_at(actions: list[dict[str, Any]], frame: int) -> dict[str, Any] | None:
    for a in actions:
        try:
            f = int(a.get("frame", -1))
        except (TypeError, ValueError):
            continue
        if f == frame:
            return a
    return None


def _bboxes_to_pixel(
    positions: dict[int, tuple[float, float, float, float]],
    W: int,
    H: int,
) -> dict[int, tuple[int, int, int, int]]:
    out = {}
    for tid, (cx, cy, bw, bh) in positions.items():
        if bw <= 0 or bh <= 0:
            continue
        x1 = int((cx - bw / 2) * W); y1 = int((cy - bh / 2) * H)
        x2 = int((cx + bw / 2) * W); y2 = int((cy + bh / 2) * H)
        out[tid] = (x1, y1, x2, y2)
    return out


def _compute_crop_rect(
    bboxes_px: dict[int, tuple[int, int, int, int]],
    ball_px: tuple[int, int] | None,
    img_w: int,
    img_h: int,
    pad: int = PAD_PX,
) -> tuple[int, int, int, int]:
    """Bounding rect around players + ball, expanded by `pad` and clipped to image."""
    xs1: list[int] = []; ys1: list[int] = []; xs2: list[int] = []; ys2: list[int] = []
    for (x1, y1, x2, y2) in bboxes_px.values():
        xs1.append(x1); ys1.append(y1); xs2.append(x2); ys2.append(y2)
    if ball_px is not None:
        bx, by = ball_px
        xs1.append(bx - 5); ys1.append(by - 5); xs2.append(bx + 5); ys2.append(by + 5)
    if not xs1:
        target_ratio = CROP_W / CROP_H
        img_ratio = img_w / max(img_h, 1)
        if img_ratio > target_ratio:
            ch = img_h
            cw = int(ch * target_ratio)
        else:
            cw = img_w
            ch = int(cw / target_ratio)
        cx = img_w // 2; cy = img_h // 2
        x1 = max(0, cx - cw // 2); y1 = max(0, cy - ch // 2)
        x2 = min(img_w, x1 + cw); y2 = min(img_h, y1 + ch)
        return (x1, y1, x2, y2)

    x1 = max(0, min(xs1) - pad)
    y1 = max(0, min(ys1) - pad)
    x2 = min(img_w, max(xs2) + pad)
    y2 = min(img_h, max(ys2) + pad)

    target_ratio = CROP_W / CROP_H
    cw = x2 - x1
    ch = y2 - y1
    if cw / max(ch, 1) < target_ratio:
        needed = int(ch * target_ratio)
        extra = needed - cw
        x1 = max(0, x1 - extra // 2)
        x2 = min(img_w, x2 + (extra - extra // 2))
        cw = x2 - x1
        if cw < needed:
            if x1 == 0:
                x2 = min(img_w, x1 + needed)
            elif x2 == img_w:
                x1 = max(0, x2 - needed)
    else:
        needed = int(cw / target_ratio)
        extra = needed - ch
        y1 = max(0, y1 - extra // 2)
        y2 = min(img_h, y2 + (extra - extra // 2))
        ch = y2 - y1
        if ch < needed:
            if y1 == 0:
                y2 = min(img_h, y1 + needed)
            elif y2 == img_h:
                y1 = max(0, y2 - needed)

    return (x1, y1, x2, y2)


def _annotate_and_crop_frame(
    frame: np.ndarray,
    positions: dict[int, tuple[float, float, float, float]],
    team_assignments: dict[str, str],
    ball: tuple[float, float] | None,
    halos: list[tuple[int, tuple[int, int, int], int]],
    label: str,
    crop_rect_hint: tuple[int, int, int, int] | None = None,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Returns (cropped_resized_BGR, crop_rect_used).

    halos: list of (tid, color, offset_px) — bigger offset draws rectangle further inside.
    """
    H, W = frame.shape[:2]
    bboxes_px = _bboxes_to_pixel(positions, W, H)
    ball_px: tuple[int, int] | None = None
    if ball is not None:
        bx_n, by_n = ball
        if not (bx_n <= 0 and by_n <= 0):
            ball_px = (int(bx_n * W), int(by_n * H))

    for tid, (x1, y1, x2, y2) in bboxes_px.items():
        team = team_assignments.get(str(tid))
        color = _team_color(team)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        tag = f"p{tid}({team})" if team else f"p{tid}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        ly = max(y1 - 6, 14)
        cv2.rectangle(frame, (x1 - 2, ly - th - 4), (x1 + tw + 4, ly + 2), (0, 0, 0), -1)
        cv2.putText(frame, tag, (x1, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    for tid, color, offset in halos:
        if tid not in bboxes_px:
            continue
        x1, y1, x2, y2 = bboxes_px[tid]
        cv2.rectangle(frame, (x1 + offset, y1 + offset), (x2 - offset, y2 - offset), color, 4)

    if ball_px is not None:
        cv2.circle(frame, ball_px, 14, COLOR_BALL, 3)
        cv2.circle(frame, ball_px, 2, COLOR_BALL, -1)

    if crop_rect_hint is not None:
        crop_rect = crop_rect_hint
    else:
        crop_rect = _compute_crop_rect(bboxes_px, ball_px, W, H)
    x1c, y1c, x2c, y2c = crop_rect
    crop = frame[y1c:y2c, x1c:x2c]
    if crop.size == 0:
        crop = np.zeros((CROP_H, CROP_W, 3), dtype=np.uint8)
    else:
        crop = cv2.resize(crop, (CROP_W, CROP_H), interpolation=cv2.INTER_AREA)

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(crop, (0, 0), (tw + 14, th + 14), (0, 0, 0), -1)
    cv2.putText(crop, label, (7, th + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)

    return crop, crop_rect


def _placeholder() -> np.ndarray:
    img = np.zeros((CROP_H, CROP_W, 3), dtype=np.uint8)
    cv2.putText(img, "frame unavailable", (50, CROP_H // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (90, 90, 90), 2, cv2.LINE_AA)
    return img


def _dim_halos(
    halos_at_zero: list[tuple[int, tuple[int, int, int], int]],
) -> list[tuple[int, tuple[int, int, int], int]]:
    out = []
    for tid, (b, g, r), offset in halos_at_zero:
        dim = (int(b * 0.55), int(g * 0.55), int(r * 0.55))
        out.append((tid, dim, offset))
    return out


def _build_row(
    *,
    anchor_action_frame: int,
    anchor_action: dict[str, Any],
    positions_by_frame: dict[int, dict[int, tuple[float, float, float, float]]],
    ball_by_frame: dict[int, tuple[float, float]],
    team_assignments: dict[str, str],
    video_path: Path,
    rally_start_frame: int,
    halos_at_zero: list[tuple[int, tuple[int, int, int], int]],
) -> list[np.ndarray]:
    # Compute a shared crop rect using the first frame in the row that has positions.
    shared_crop_rect: tuple[int, int, int, int] | None = None
    for off in [0, -3, 3, -6, 6]:
        target_pl_frame = anchor_action_frame + off
        pos = positions_by_frame.get(target_pl_frame, {})
        if not pos:
            continue
        if off == 0:
            ball = ball_by_frame.get(target_pl_frame, (
                float(anchor_action.get("ballX", 0.0)),
                float(anchor_action.get("ballY", 0.0)),
            ))
        else:
            ball = ball_by_frame.get(target_pl_frame, (0.0, 0.0))
        source_frame = rally_start_frame + target_pl_frame
        extracted = _extract_frame(video_path, source_frame)
        if extracted is None:
            continue
        H, W = extracted.shape[:2]
        bboxes_px = _bboxes_to_pixel(pos, W, H)
        ball_px = None
        if not (ball[0] <= 0 and ball[1] <= 0):
            ball_px = (int(ball[0] * W), int(ball[1] * H))
        shared_crop_rect = _compute_crop_rect(bboxes_px, ball_px, W, H)
        break

    row: list[np.ndarray] = []
    for off in OFFSETS:
        target_pl_frame = anchor_action_frame + off
        source_frame = rally_start_frame + target_pl_frame
        extracted = _extract_frame(video_path, source_frame)
        if extracted is None:
            row.append(_placeholder())
            continue
        pos = positions_by_frame.get(target_pl_frame, {})
        if off == 0:
            ball = ball_by_frame.get(target_pl_frame, (
                float(anchor_action.get("ballX", 0.0)),
                float(anchor_action.get("ballY", 0.0)),
            ))
        else:
            ball = ball_by_frame.get(target_pl_frame, (0.0, 0.0))
        halos = halos_at_zero if off == 0 else _dim_halos(halos_at_zero)
        label = f"f={target_pl_frame}  off={off:+d}"
        cropped, _ = _annotate_and_crop_frame(
            extracted, pos, team_assignments,
            ball if not (ball[0] <= 0 and ball[1] <= 0) else None,
            halos, label,
            crop_rect_hint=shared_crop_rect,
        )
        row.append(cropped)
    return row


def _render_case(
    *,
    case: dict[str, Any],
    video_path: Path,
    video_meta: dict[str, Any],
    out_path: Path,
) -> dict[str, Any] | None:
    rally_id = case["rally_id"]
    prev_frame = case["prev_frame"]
    curr_frame = case["curr_frame"]
    actions, positions_by_frame, ball_by_frame, team_assignments, rally_start_ms = (
        _fetch_rally_state(rally_id)
    )
    if not actions:
        print(f"  [{case['idx']}] no actions for rally {rally_id[:8]}")
        return None

    prev_action = _find_action_at(actions, prev_frame)
    curr_action = _find_action_at(actions, curr_frame)
    if not prev_action or not curr_action:
        print(f"  [{case['idx']}] missing prev@f{prev_frame} or curr@f{curr_frame}")
        return None

    fps = float(video_meta.get("fps") or 30.0)
    rally_start_frame = int(round(rally_start_ms / 1000.0 * fps))

    pipeline_prev_pid = case["pipeline_prev_pid"]
    pipeline_curr_pid = case["pipeline_curr_pid"]
    a1_prev_pid = case["a1_prev_pid"]
    a1_curr_pid = case["a1_curr_pid"]

    # Halos on PREV row.
    prev_halos: list[tuple[int, tuple[int, int, int], int]] = []
    if pipeline_prev_pid is not None and pipeline_prev_pid >= 0:
        prev_halos.append((pipeline_prev_pid, COLOR_PIPELINE_PREV, -4))
    if a1_prev_pid is not None and a1_prev_pid >= 0:
        # If A1 same as pipeline (shouldn't happen but be safe), stack outside.
        offset = -10 if a1_prev_pid == pipeline_prev_pid else -4
        prev_halos.append((a1_prev_pid, COLOR_A1_AFTER, offset))

    # Halos on CURR row.
    curr_halos: list[tuple[int, tuple[int, int, int], int]] = []
    if pipeline_curr_pid is not None and pipeline_curr_pid >= 0:
        curr_halos.append((pipeline_curr_pid, COLOR_PIPELINE_CURR, -4))
    if a1_curr_pid is not None and a1_curr_pid >= 0:
        offset = -10 if a1_curr_pid == pipeline_curr_pid else -4
        curr_halos.append((a1_curr_pid, COLOR_A1_AFTER, offset))

    prev_row = _build_row(
        anchor_action_frame=prev_frame,
        anchor_action=prev_action,
        positions_by_frame=positions_by_frame,
        ball_by_frame=ball_by_frame,
        team_assignments=team_assignments,
        video_path=video_path,
        rally_start_frame=rally_start_frame,
        halos_at_zero=prev_halos,
    )
    curr_row = _build_row(
        anchor_action_frame=curr_frame,
        anchor_action=curr_action,
        positions_by_frame=positions_by_frame,
        ball_by_frame=ball_by_frame,
        team_assignments=team_assignments,
        video_path=video_path,
        rally_start_frame=rally_start_frame,
        halos_at_zero=curr_halos,
    )

    if len(prev_row) != len(OFFSETS) or len(curr_row) != len(OFFSETS):
        print(f"  [{case['idx']}] row build failed (prev={len(prev_row)} curr={len(curr_row)})")
        return None

    sep_v = np.full((CROP_H, 6, 3), 30, dtype=np.uint8)
    prev_concat = prev_row[0]
    for f in prev_row[1:]:
        prev_concat = cv2.hconcat([prev_concat, sep_v, f])
    curr_concat = curr_row[0]
    for f in curr_row[1:]:
        curr_concat = cv2.hconcat([curr_concat, sep_v, f])

    full_w = prev_concat.shape[1]
    sep_h = np.full((10, full_w, 3), 30, dtype=np.uint8)

    row_tag_h = 28

    def _row_tag(text: str, color: tuple[int, int, int]) -> np.ndarray:
        strip = np.full((row_tag_h, full_w, 3), 30, dtype=np.uint8)
        cv2.putText(strip, text, (10, row_tag_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
        return strip

    # PREV tag
    prev_pid_str = f"p{pipeline_prev_pid}({case['prev_team']})"
    prev_a1_str = (
        f" --> A1 p{a1_prev_pid}({case['prev_team']})"
        if a1_prev_pid is not None else "  (A1 unchanged)"
    )
    prev_tag = _row_tag(
        f"PREV  {case['prev_action_type']}@f{prev_frame}  pipeline={prev_pid_str}{prev_a1_str}",
        COLOR_PIPELINE_PREV if a1_prev_pid is None else COLOR_A1_AFTER,
    )

    # CURR tag
    curr_pid_str = f"p{pipeline_curr_pid}({case['curr_team']})"
    curr_a1_str = (
        f" --> A1 p{a1_curr_pid}({case['curr_team']})"
        if a1_curr_pid is not None else "  (A1 unchanged)"
    )
    curr_tag = _row_tag(
        f"CURR  {case['curr_action_type']}@f{curr_frame}  pipeline={curr_pid_str}{curr_a1_str}",
        COLOR_PIPELINE_CURR if a1_curr_pid is None else COLOR_A1_AFTER,
    )

    # Header
    header_h = 56
    header = np.full((header_h, full_w, 3), 20, dtype=np.uint8)
    edit_side = "PREV" if a1_prev_pid is not None else "CURR"
    edit_from = pipeline_prev_pid if a1_prev_pid is not None else pipeline_curr_pid
    edit_to = a1_prev_pid if a1_prev_pid is not None else a1_curr_pid
    title1 = (
        f"#{case['idx']}  {case['video_name']}/{case['rally_short']}  "
        f"alt_ratio={case['alt_ratio']:.1f}x  "
        f"A1 edit: {edit_side} p{edit_from} -> p{edit_to}"
    )
    title2 = (
        f"PREV: {case['prev_action_type']}@f{prev_frame}  "
        f"CURR: {case['curr_action_type']}@f{curr_frame}  "
        f"Strict-rule: no same-player consecutive (except after BLOCK)"
    )
    cv2.putText(header, title1, (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.putText(header, title2, (10, 47),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    final = cv2.vconcat([
        header,
        prev_tag,
        prev_concat,
        sep_h,
        curr_tag,
        curr_concat,
    ])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), final, [int(cv2.IMWRITE_JPEG_QUALITY), 88])

    return {
        "idx": case["idx"],
        "video_name": case["video_name"],
        "rally_short": case["rally_short"],
        "rally_id": rally_id,
        "prev_frame": prev_frame,
        "curr_frame": curr_frame,
        "edit_side": edit_side,
        "img": str(out_path.relative_to(REPORT_DIR)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=int, default=None,
                        help="Render only the case with this 1-based index")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    resolver = VideoResolver()
    video_meta_cache: dict[str, dict[str, Any]] = {}
    video_path_cache: dict[str, Path | None] = {}

    cases = A1_CASES
    if args.only is not None:
        cases = [c for c in cases if c["idx"] == args.only]
    print(f"Rendering {len(cases)} A1 motion strips...", flush=True)

    successes = 0
    failures: list[str] = []
    for case in cases:
        vid = case["video_id"]
        if vid not in video_meta_cache:
            vm = _fetch_video_meta(vid)
            video_meta_cache[vid] = vm
            video_path_cache[vid] = _resolve_video(resolver, vm) if vm else None
        vm = video_meta_cache[vid]
        vpath = video_path_cache[vid]
        if not vm or not vpath:
            msg = f"[{case['idx']}] no video for {case['video_name']}/{case['rally_short']}"
            print(f"  {msg}", flush=True)
            failures.append(msg)
            continue

        out_name = f"{case['idx']:02d}_{case['video_name']}_{case['rally_short']}.jpg"
        out_path = OUT_DIR / out_name
        rec = _render_case(
            case=case,
            video_path=vpath,
            video_meta=vm,
            out_path=out_path,
        )
        if rec is None:
            failures.append(f"[{case['idx']}] render failed")
            continue
        successes += 1
        print(f"  [{case['idx']:>2}/{len(A1_CASES)}] wrote: {out_path.name}", flush=True)

    print()
    print(f"Wrote {successes} / {len(cases)} cases")
    if failures:
        print("Failures:")
        for f in failures:
            print(f"  - {f}")
    print(f"Output dir: {OUT_DIR}")


if __name__ == "__main__":
    main()
