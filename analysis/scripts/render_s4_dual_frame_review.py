"""Render dual-frame (prev + curr) annotated strips for the 17 S4 "failure-mode" picks.

For each of the 17 cases, builds a side-by-side image:
  LEFT  = previous action's frame, with all 4 player bboxes drawn,
          the prev action's attributed player highlighted in RED, ball drawn yellow.
  RIGHT = current S4 candidate frame, with all 4 player bboxes drawn,
          pipeline pick highlighted CYAN, S4 pick GREEN,
          prev_toucher's track highlighted RED (the one S4 is excluding), ball yellow.

Outputs:
  analysis/reports/probe_b_sequence_aware/2026_05_14/s4_dual_frames/{N}_{video}_{rally}_prev_f{prev}_curr_f{curr}.jpg
  analysis/reports/probe_b_sequence_aware/2026_05_14/s4_dual_frame_review.html

Usage:
    cd analysis
    uv run python scripts/render_s4_dual_frame_review.py
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from rallycut.evaluation.tracking.db import get_connection
from rallycut.evaluation.video_resolver import VideoResolver

HERE = Path(__file__).resolve().parent
REPORT_DIR = HERE.parent / "reports" / "probe_b_sequence_aware" / "2026_05_14"
FRAMES_DIR = REPORT_DIR / "s4_dual_frames"
DEFAULT_FLIPS = REPORT_DIR / "s4_fleet_candidates.json"
DEFAULT_HTML = REPORT_DIR / "s4_dual_frame_review.html"

# The 17 target picks (rally_uuid, curr_frame). Order = display order.
TARGETS: list[tuple[str, int, str]] = [
    # probe_b_cascade_anchor
    ("a0881d82-bd3e-4664-bd11-4e672ca29aa6", 128, "probe_b_cascade_anchor"),
    ("a0881d82-bd3e-4664-bd11-4e672ca29aa6", 225, "probe_b_cascade_anchor"),
    # attack_after_attack
    ("1f5ff17d-014f-4093-b6e6-c85360fb1638", 304, "attack_after_attack"),
    ("8d3205ed-b0dc-4c0c-bc24-fda34554e45f", 296, "attack_after_attack"),
    ("55565c2b-49b2-4817-b046-7bef02569b0e", 558, "attack_after_attack"),
    ("21a9b203-dc92-48dc-8f19-d94835e0e226", 528, "attack_after_attack"),
    ("49582c29-f09c-4b39-8435-bd19451812c8", 314, "attack_after_attack"),
    ("0793ebd2-5301-4317-a7ba-3fb3bf9fe368", 380, "attack_after_attack"),
    ("65c54107-6c0f-4888-9600-6ec06939c8b5", 220, "attack_after_attack"),
    # attack_after_set
    ("594a832a-ab27-480e-976a-268c8f101559", 193, "attack_after_set"),
    ("ded9504c-ce40-4808-8d12-8bcc3b974b62", 230, "attack_after_set"),
    ("920d4a33-ae9a-4299-ac19-8e773ca76e61", 159, "attack_after_set"),
    ("a1a5baf7-e249-476e-9ab7-cf60fac04111", 679, "attack_after_set"),
    ("eeecf5a0-def4-43dd-8fc9-72e502165c2d", 533, "attack_after_set"),
    ("e50f127e-3952-4ba2-9047-39fff14a2e25", 233, "attack_after_set"),
    ("f62bc819-d7c4-4287-b617-edac3f5194bc", 177, "attack_after_set"),
    # same_team_other (attack after receive, mech/f8e251d8)
    ("f8e251d8-c7ee-40fd-946c-502067343936", 127, "same_team_other"),
]

# Colors (BGR for cv2)
COLOR_PIPELINE = (255, 200, 60)     # cyan-ish
COLOR_S4 = (60, 220, 60)            # green
COLOR_PREV = (60, 60, 255)          # red
COLOR_TEAM_A = (200, 120, 50)       # blue-ish
COLOR_TEAM_B = (80, 80, 200)        # red-ish
COLOR_OTHER = (160, 160, 160)
COLOR_BALL = (0, 255, 255)


@dataclass
class FrameInfo:
    positions: dict[int, tuple[float, float, float, float]]  # tid -> (x, y, w, h) normalized centers
    ball: tuple[float, float]
    frame_number: int


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


def _fetch_rally_state(rally_id: str) -> tuple[list[dict[str, Any]], dict[int, dict[int, tuple[float, float, float, float]]], dict[int, tuple[float, float]], dict[str, str], int]:
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

    # Positions: frame -> trackId -> (xc,yc,w,h) (all normalized)
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


def _draw_box(img: np.ndarray, bbox: tuple[float, float, float, float],
              color: tuple[int, int, int], thickness: int = 2,
              label: str | None = None) -> None:
    h, w = img.shape[:2]
    cx, cy, bw, bh = bbox
    if bw <= 0 or bh <= 0:
        return
    x1 = int((cx - bw / 2) * w)
    y1 = int((cy - bh / 2) * h)
    x2 = int((cx + bw / 2) * w)
    y2 = int((cy + bh / 2) * h)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ly = max(y1 - 6, 14)
        cv2.rectangle(img, (x1 - 2, ly - th - 4), (x1 + tw + 4, ly + 2), (0, 0, 0), -1)
        cv2.putText(img, label, (x1, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def _draw_halo(img: np.ndarray, bbox: tuple[float, float, float, float],
               color: tuple[int, int, int], thickness: int = 3,
               offset: int = -4) -> None:
    h, w = img.shape[:2]
    cx, cy, bw, bh = bbox
    if bw <= 0 or bh <= 0:
        return
    x1 = int((cx - bw / 2) * w) + offset
    y1 = int((cy - bh / 2) * h) + offset
    x2 = int((cx + bw / 2) * w) - offset
    y2 = int((cy + bh / 2) * h) - offset
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)


def _draw_ball(img: np.ndarray, ball: tuple[float, float], color: tuple[int, int, int] = COLOR_BALL) -> None:
    bx_n, by_n = ball
    if bx_n <= 0 and by_n <= 0:
        return
    h, w = img.shape[:2]
    bx = int(bx_n * w); by = int(by_n * h)
    cv2.circle(img, (bx, by), 14, color, 3)
    cv2.circle(img, (bx, by), 2, color, -1)


def _banner(img: np.ndarray, lines: list[str]) -> None:
    h, w = img.shape[:2]
    banner_h = 22 + 20 * len(lines)
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    y = 22
    for i, ln in enumerate(lines):
        scale = 0.6 if i == 0 else 0.5
        weight = 2 if i == 0 else 1
        cv2.putText(img, ln, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), weight, cv2.LINE_AA)
        y += 20


def _annotate_prev_frame(
    img: np.ndarray,
    positions: dict[int, tuple[float, float, float, float]],
    team_assignments: dict[str, str],
    ball: tuple[float, float],
    prev_pid: int,
    prev_action: str,
    prev_team: str,
    pl_frame: int,
    source_time: str,
) -> None:
    """Annotate the prev action frame."""
    # Bboxes per player track.
    for tid, bb in positions.items():
        team = team_assignments.get(str(tid))
        color = _team_color(team)
        label = f"p{tid}({team})" if team else f"p{tid}"
        _draw_box(img, bb, color, thickness=1, label=label)

    # Highlight prev attributed player with thick red border.
    if prev_pid in positions:
        _draw_halo(img, positions[prev_pid], COLOR_PREV, thickness=3, offset=-4)
        # Re-draw text for clarity
        h, w = img.shape[:2]
        cx, cy, bw, bh = positions[prev_pid]
        x1 = int((cx - bw / 2) * w); y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w); y2 = int((cy + bh / 2) * h)
        tag = f"PREV p{prev_pid}({prev_action.upper()})"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = y2 + th + 6
        cv2.rectangle(img, (x1 - 2, ty - th - 4), (x1 + tw + 4, ty + 2), (0, 0, 0), -1)
        cv2.putText(img, tag, (x1, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_PREV, 1, cv2.LINE_AA)

    _draw_ball(img, ball)
    _banner(img, [
        f"PREV: {prev_action.upper()} @ f{pl_frame}",
        f"source time: {source_time}  prev_player=p{prev_pid}({prev_team})",
    ])


def _annotate_curr_frame(
    img: np.ndarray,
    positions: dict[int, tuple[float, float, float, float]],
    team_assignments: dict[str, str],
    ball: tuple[float, float],
    pipeline_pid: int,
    s4_pid: int,
    prev_pid: int,
    pipeline_team: str,
    s4_team: str,
    prev_team: str,
    curr_action: str,
    pl_frame: int,
    source_time: str,
    prev_action: str,
    prev_frame: int,
) -> None:
    """Annotate the curr action frame."""
    for tid, bb in positions.items():
        team = team_assignments.get(str(tid))
        color = _team_color(team)
        label = f"p{tid}({team})" if team else f"p{tid}"
        _draw_box(img, bb, color, thickness=1, label=label)

    # Halos: prev (red), pipeline (cyan), s4 (green). Stack with growing offsets.
    halos: list[tuple[int, tuple[int, int, int], str]] = []
    if prev_pid in positions:
        halos.append((prev_pid, COLOR_PREV, f"PREV-TOUCHER p{prev_pid}"))
    if pipeline_pid in positions:
        halos.append((pipeline_pid, COLOR_PIPELINE, f"PIPELINE p{pipeline_pid}({pipeline_team})"))
    if s4_pid in positions and s4_pid != pipeline_pid:
        halos.append((s4_pid, COLOR_S4, f"S4 p{s4_pid}({s4_team})"))

    # Draw each halo at decreasing offset for stacked rings.
    halo_offsets: dict[int, int] = {}
    for pid, color, _ in halos:
        offset = halo_offsets.get(pid, -4)
        _draw_halo(img, positions[pid], color, thickness=3, offset=offset)
        halo_offsets[pid] = offset - 4

    # Labels below boxes.
    h, w = img.shape[:2]
    used_labels: dict[int, list[str]] = {}
    for pid, color, tag in halos:
        used_labels.setdefault(pid, []).append((tag, color))  # type: ignore[arg-type]
    for pid, items in used_labels.items():
        if pid not in positions:
            continue
        cx, cy, bw, bh = positions[pid]
        x1 = int((cx - bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        ty = y2 + 16
        for tag, color in items:  # type: ignore[misc]
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1 - 2, ty - th - 4), (x1 + tw + 4, ty + 2), (0, 0, 0), -1)
            cv2.putText(img, tag, (x1, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            ty += 18

    _draw_ball(img, ball)
    _banner(img, [
        f"CURR: {curr_action.upper()} @ f{pl_frame}  (prev was {prev_action.upper()}@f{prev_frame})",
        f"source time: {source_time}  PL=p{pipeline_pid}({pipeline_team}) vs S4=p{s4_pid}({s4_team})",
    ])


def _ms_to_str(total_ms: float) -> str:
    total_s = total_ms / 1000.0
    m = int(total_s // 60)
    s = total_s - m * 60
    return f"{m}:{s:06.3f}"


def _find_prev_action(actions: list[dict[str, Any]], curr_frame: int) -> dict[str, Any] | None:
    """Find the action with the largest frame strictly less than curr_frame."""
    best = None
    for a in actions:
        try:
            f = int(a.get("frame", -1))
        except (TypeError, ValueError):
            continue
        if f < curr_frame:
            if best is None or f > int(best["frame"]):
                best = a
    return best


def _find_curr_action(actions: list[dict[str, Any]], curr_frame: int) -> dict[str, Any] | None:
    """Find the action whose frame equals curr_frame."""
    for a in actions:
        try:
            f = int(a.get("frame", -1))
        except (TypeError, ValueError):
            continue
        if f == curr_frame:
            return a
    return None


def _compose_side_by_side(left: np.ndarray, right: np.ndarray,
                          target_height: int = 720) -> np.ndarray:
    """Scale both to `target_height` and stack horizontally with a separator."""
    def _scale(im: np.ndarray) -> np.ndarray:
        h, w = im.shape[:2]
        if h == target_height:
            return im
        new_w = int(round(w * target_height / max(h, 1)))
        return cv2.resize(im, (new_w, target_height), interpolation=cv2.INTER_AREA)

    L = _scale(left); R = _scale(right)
    sep = np.full((target_height, 8, 3), 30, dtype=np.uint8)
    return cv2.hconcat([L, sep, R])


def _render_case(
    *,
    idx: int,
    rally_id: str,
    curr_frame: int,
    bucket: str,
    flip: dict[str, Any],
    video_path: Path,
    video_meta: dict[str, Any],
    out_path: Path,
) -> dict[str, Any] | None:
    actions, positions_by_frame, ball_by_frame, team_assignments, rally_start_ms = (
        _fetch_rally_state(rally_id)
    )
    if not actions:
        print(f"  [{idx}] no actions for rally {rally_id[:8]}")
        return None

    curr_action = _find_curr_action(actions, curr_frame)
    prev_action = _find_prev_action(actions, curr_frame)
    if not curr_action or not prev_action:
        print(f"  [{idx}] missing curr/prev action at f={curr_frame}")
        return None

    prev_frame = int(prev_action["frame"])
    fps = float(video_meta.get("fps") or flip.get("fps") or 30.0)
    rally_start_frame = int(round(rally_start_ms / 1000.0 * fps))

    prev_source_frame = rally_start_frame + prev_frame
    curr_source_frame = rally_start_frame + curr_frame

    left_img = _extract_frame(video_path, prev_source_frame)
    right_img = _extract_frame(video_path, curr_source_frame)
    if left_img is None or right_img is None:
        print(f"  [{idx}] frame extraction failed (prev_src={prev_source_frame} curr_src={curr_source_frame})")
        return None

    prev_positions = positions_by_frame.get(prev_frame, {})
    curr_positions = positions_by_frame.get(curr_frame, {})

    prev_ball = ball_by_frame.get(prev_frame, (
        float(prev_action.get("ballX", 0.0)),
        float(prev_action.get("ballY", 0.0)),
    ))
    curr_ball = ball_by_frame.get(curr_frame, (
        float(curr_action.get("ballX", 0.0)),
        float(curr_action.get("ballY", 0.0)),
    ))

    prev_pid = int(prev_action.get("playerTrackId", -1))
    prev_team = str(prev_action.get("team") or team_assignments.get(str(prev_pid), "?"))
    prev_action_type = str(prev_action.get("action", "?")).upper()
    curr_action_type = str(curr_action.get("action", "?")).upper()

    pipeline_pid = int(flip["pipeline_pid"])
    s4_pid = int(flip["s4_pid"])
    pipeline_team = str(flip["pipeline_team"])
    s4_team = str(flip["s4_team"])

    prev_source_ms = rally_start_ms + prev_frame * 1000.0 / max(fps, 1e-6)
    curr_source_ms = rally_start_ms + curr_frame * 1000.0 / max(fps, 1e-6)

    _annotate_prev_frame(
        left_img,
        prev_positions,
        team_assignments,
        prev_ball,
        prev_pid,
        prev_action_type,
        prev_team,
        prev_frame,
        _ms_to_str(prev_source_ms),
    )

    _annotate_curr_frame(
        right_img,
        curr_positions,
        team_assignments,
        curr_ball,
        pipeline_pid,
        s4_pid,
        prev_pid,
        pipeline_team,
        s4_team,
        prev_team,
        curr_action_type,
        curr_frame,
        _ms_to_str(curr_source_ms),
        prev_action_type,
        prev_frame,
    )

    composed = _compose_side_by_side(left_img, right_img)

    # Top strip with overall title.
    h, w = composed.shape[:2]
    title_h = 36
    strip = np.full((title_h, w, 3), 30, dtype=np.uint8)
    title = (f"#{idx} [{bucket}] {flip['video_name']}/{flip['rally_short']}  "
             f"{prev_action_type}@f{prev_frame} -> {curr_action_type}@f{curr_frame}  "
             f"(prev p{prev_pid}({prev_team})  PL=p{pipeline_pid}({pipeline_team})  "
             f"S4=p{s4_pid}({s4_team}))")
    cv2.putText(strip, title, (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
    final = cv2.vconcat([strip, composed])

    cv2.imwrite(str(out_path), final, [int(cv2.IMWRITE_JPEG_QUALITY), 88])

    return {
        "idx": idx,
        "bucket": bucket,
        "video_name": flip["video_name"],
        "rally_short": flip["rally_short"],
        "rally_id": rally_id,
        "rally_order": flip.get("rally_order", -1),
        "fps": fps,
        "prev_frame": prev_frame,
        "prev_action": prev_action_type,
        "prev_pid": prev_pid,
        "prev_team": prev_team,
        "curr_frame": curr_frame,
        "curr_action": curr_action_type,
        "pipeline_pid": pipeline_pid,
        "pipeline_team": pipeline_team,
        "s4_pid": s4_pid,
        "s4_team": s4_team,
        "prev_source_time": _ms_to_str(prev_source_ms),
        "curr_source_time": _ms_to_str(curr_source_ms),
        "img": str(out_path.relative_to(REPORT_DIR)),
    }


_HTML_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>S4 dual-frame review — 2026-05-14</title>
<style>
  :root { --bg:#0f0f10; --card:#1a1a1c; --border:#2a2a2e; --text:#eaeaea; --muted:#aaa; }
  body { background:var(--bg); color:var(--text); font-family:-apple-system,sans-serif; margin:0; padding:24px; }
  h1 { margin:0 0 6px 0; }
  .summary { color:var(--muted); margin-bottom:14px; font-size:13px; }
  .legend { background:var(--card); border:1px solid var(--border); border-radius:8px; padding:10px 18px; margin-bottom:18px; font-size:13px; }
  .legend span.swatch { display:inline-block; width:12px; height:12px; vertical-align:middle; margin-right:4px; border-radius:2px; }
  .toolbar { position:sticky; top:0; background:var(--bg); padding:8px 0 14px 0; z-index:5; border-bottom:1px solid var(--border); }
  .toolbar button { background:#222; color:#eaeaea; border:1px solid #333; border-radius:6px; padding:6px 12px; margin-right:6px; cursor:pointer; }
  .toolbar button:hover { background:#2a2a2e; }
  .case { background:var(--card); border:1px solid var(--border); border-radius:8px; padding:14px 18px; margin:14px 0; display:grid; grid-template-columns: 1fr 280px; gap:14px; }
  .case h3 { margin:0 0 8px 0; font-size:15px; }
  .case img { max-width:100%; max-height:80vh; display:block; margin:6px 0; cursor:zoom-in; background:#000; border-radius:4px; }
  .meta { color:var(--muted); font-size:12px; line-height:1.5; }
  .meta b { color:#ddd; }
  .bucket { display:inline-block; padding:2px 8px; border-radius:4px; font-size:11px; font-weight:600; background:#222; color:#ddd; margin-left:6px; }
  .sidebar { display:flex; flex-direction:column; gap:10px; }
  .verdict-buttons { display:flex; flex-direction:column; gap:6px; }
  .verdict-buttons button { background:#222; color:#eaeaea; border:1px solid #333; border-radius:6px; padding:8px 10px; cursor:pointer; text-align:left; font-size:13px; }
  .verdict-buttons button:hover { background:#2a2a2e; }
  .verdict-buttons button.selected { box-shadow: inset 0 0 0 2px #60aaff; background:#1f3050; }
  .verdict-readout { margin-top:6px; color:#9cc; font-size:12px; min-height:18px; }
  #zoom-overlay { display:none; position:fixed; inset:0; background:rgba(0,0,0,.94); z-index:99; align-items:center; justify-content:center; cursor:zoom-out; }
  #zoom-overlay.open { display:flex; }
  #zoom-overlay img { max-width:97vw; max-height:97vh; }
  #paste-overlay { display:none; position:fixed; inset:0; background:rgba(0,0,0,.85); z-index:100; align-items:center; justify-content:center; }
  #paste-overlay.open { display:flex; }
  #paste-overlay .panel { background:#1a1a1c; border:1px solid #333; border-radius:8px; padding:18px; max-width:700px; width:90%; }
  #paste-overlay textarea { width:100%; height:240px; background:#0f0f10; color:#eaeaea; border:1px solid #333; border-radius:4px; padding:10px; font-family:ui-monospace,monospace; font-size:12px; }
  #paste-overlay .panel-actions { margin-top:10px; display:flex; gap:8px; }
</style></head><body>
<h1>S4 dual-frame review — failure-mode candidates</h1>
<div class="summary">17 cases where pipeline shows a same-player consecutive pair without a block between them. Strict volleyball rule says one side must be wrong.</div>
<div class="legend">
  <span class="swatch" style="background:rgb(50,120,200)"></span>Team A
  &nbsp;&nbsp;<span class="swatch" style="background:rgb(200,80,80)"></span>Team B
  &nbsp;&nbsp;<span class="swatch" style="background:rgb(60,200,255)"></span>Pipeline pick (cyan)
  &nbsp;&nbsp;<span class="swatch" style="background:rgb(60,220,60)"></span>S4 proposed (green)
  &nbsp;&nbsp;<span class="swatch" style="background:rgb(255,60,60)"></span>Prev attributed player / prev-toucher (red)
  &nbsp;&nbsp;<span class="swatch" style="background:rgb(255,255,0)"></span>Ball (yellow)
</div>
<div class="toolbar">
  <button onclick="copyVerdicts()">Copy verdicts to clipboard</button>
  <button onclick="showPaste()">Show paste-back string</button>
  <button onclick="resetVerdicts()">Reset all</button>
  <span id="progress" style="color:#9cc; margin-left:10px; font-size:13px;"></span>
</div>
<div id="cases"></div>
<div id="zoom-overlay"><img id="zoom-img"></div>
<div id="paste-overlay"><div class="panel">
  <div style="font-size:13px; color:#ccc; margin-bottom:8px;">Paste-back string:</div>
  <textarea id="paste-text" readonly></textarea>
  <div class="panel-actions">
    <button onclick="copyPaste()">Copy</button>
    <button onclick="hidePaste()">Close</button>
  </div>
</div></div>
<script id="data" type="application/json">__DATA_JSON__</script>
<script>
const DATA = JSON.parse(document.getElementById('data').textContent);
const VERDICT_OPTIONS = [
  { code: "prev_wrong",   label: "PREV wrong (prev's attribution is bad)",          icon: "[P]" },
  { code: "curr_wrong",   label: "CURR wrong (S4 is right)",                         icon: "[C]" },
  { code: "type_wrong",   label: "TYPE wrong (one is actually BLOCK)",               icon: "[T]" },
  { code: "false_pos",    label: "FALSE POSITIVE (one contact isn't real)",          icon: "[F]" },
  { code: "other",        label: "something else",                                   icon: "[?]" },
];
const verdicts = {};

function caseHtml(r) {
  const buttons = VERDICT_OPTIONS.map(o =>
    `<button data-idx="${r.idx}" data-code="${o.code}" onclick="setVerdict(${r.idx}, '${o.code}', this)">${o.icon} ${o.label}</button>`
  ).join('');
  return `
    <div class="case" id="case-${r.idx}">
      <div class="left">
        <h3>#${r.idx} ${r.video_name}/${r.rally_short} <span class="bucket">${r.bucket}</span></h3>
        <div class="meta">
          <b>prev:</b> ${r.prev_action}@f${r.prev_frame} (${r.prev_source_time}) p${r.prev_pid}(${r.prev_team})<br>
          <b>curr:</b> ${r.curr_action}@f${r.curr_frame} (${r.curr_source_time}) — PL=p${r.pipeline_pid}(${r.pipeline_team}) vs S4=p${r.s4_pid}(${r.s4_team})<br>
          <b>rally:</b> #${r.rally_order} <code style="font-size:11px;">${r.rally_id}</code>
        </div>
        <img src="${r.img}" onclick="zoom(this.src)">
      </div>
      <div class="sidebar">
        <div style="font-size:13px; color:#ccc;">Verdict</div>
        <div class="verdict-buttons" data-idx="${r.idx}">${buttons}</div>
        <div class="verdict-readout" id="vr-${r.idx}"></div>
      </div>
    </div>`;
}

function setVerdict(idx, code, btn) {
  verdicts[idx] = code;
  document.querySelectorAll(`.verdict-buttons[data-idx="${idx}"] button`).forEach(b => b.classList.remove('selected'));
  btn.classList.add('selected');
  const r = DATA.find(x => x.idx === idx);
  document.getElementById(`vr-${idx}`).textContent = `${code} — ${r.video_name}/${r.rally_short} f${r.curr_frame}`;
  updateProgress();
  localStorage.setItem('s4_dual_verdicts', JSON.stringify(verdicts));
}

function updateProgress() {
  const n = Object.keys(verdicts).length;
  document.getElementById('progress').textContent = `${n} / ${DATA.length} labeled`;
}

function buildPaste() {
  const rows = DATA.map(r => {
    const v = verdicts[r.idx] || "";
    return `${r.idx}\\t${r.bucket}\\t${r.video_name}/${r.rally_short}\\tf${r.prev_frame}->${r.curr_frame}\\t${r.prev_action}->${r.curr_action}\\tPL=p${r.pipeline_pid} S4=p${r.s4_pid} prev=p${r.prev_pid}\\t${v}`;
  });
  return ["idx\\tbucket\\tvideo/rally\\tframes\\taction_pair\\tpids\\tverdict", ...rows].join("\\n");
}

function copyVerdicts() {
  const text = buildPaste();
  navigator.clipboard.writeText(text).then(
    () => alert("Copied " + Object.keys(verdicts).length + " verdicts."),
    () => { showPaste(); }
  );
}
function showPaste() {
  document.getElementById('paste-text').value = buildPaste();
  document.getElementById('paste-overlay').classList.add('open');
}
function hidePaste() { document.getElementById('paste-overlay').classList.remove('open'); }
function copyPaste() {
  const ta = document.getElementById('paste-text');
  ta.select();
  document.execCommand('copy');
  alert("Copied.");
}
function resetVerdicts() {
  if (!confirm("Clear all verdicts?")) return;
  for (const k of Object.keys(verdicts)) delete verdicts[k];
  localStorage.removeItem('s4_dual_verdicts');
  document.querySelectorAll('.verdict-buttons button').forEach(b => b.classList.remove('selected'));
  document.querySelectorAll('.verdict-readout').forEach(e => e.textContent = '');
  updateProgress();
}
function zoom(s){const o=document.getElementById('zoom-overlay');document.getElementById('zoom-img').src=s;o.classList.add('open');}
document.getElementById('zoom-overlay').addEventListener('click', () => document.getElementById('zoom-overlay').classList.remove('open'));

document.getElementById('cases').innerHTML = DATA.map(caseHtml).join('');
// Restore from localStorage.
try {
  const saved = JSON.parse(localStorage.getItem('s4_dual_verdicts') || "{}");
  for (const [k, v] of Object.entries(saved)) {
    const btn = document.querySelector(`.verdict-buttons[data-idx="${k}"] button[data-code="${v}"]`);
    if (btn) { btn.click(); }
  }
} catch (e) {}
updateProgress();
</script></body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Render dual-frame review page")
    parser.add_argument("--flips", type=str, default=str(DEFAULT_FLIPS))
    parser.add_argument("--html", type=str, default=str(DEFAULT_HTML))
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    data = json.loads(Path(args.flips).read_text())
    flips = data["flips"]
    flip_by_key: dict[tuple[str, int], dict[str, Any]] = {
        (f["rally_id"], f["pl_frame"]): f for f in flips
    }

    resolver = VideoResolver()
    video_meta_cache: dict[str, dict[str, Any]] = {}
    video_path_cache: dict[str, Path | None] = {}
    payload: list[dict[str, Any]] = []
    failures: list[str] = []

    print(f"Rendering {len(TARGETS)} dual-frame cases...")
    for i, (rally_id, curr_frame, bucket) in enumerate(TARGETS, start=1):
        flip = flip_by_key.get((rally_id, curr_frame))
        if not flip:
            msg = f"[{i}] miss: ({rally_id[:8]}, {curr_frame})"
            print(f"  {msg}")
            failures.append(msg)
            continue
        vid = flip["video_id"]
        if vid not in video_meta_cache:
            vm = _fetch_video_meta(vid)
            video_meta_cache[vid] = vm
            video_path_cache[vid] = _resolve_video(resolver, vm) if vm else None
        vm = video_meta_cache[vid]
        vpath = video_path_cache[vid]
        if not vm or not vpath:
            msg = f"[{i}] no video for {flip['video_name']}/{flip['rally_short']}"
            print(f"  {msg}")
            failures.append(msg)
            continue

        out_name = (
            f"{i:02d}_{flip['video_name']}_{flip['rally_short']}_"
            f"prev_f{flip['prev_action_frame']}_curr_f{curr_frame}.jpg"
        )
        out_path = FRAMES_DIR / out_name
        rec = _render_case(
            idx=i,
            rally_id=rally_id,
            curr_frame=curr_frame,
            bucket=bucket,
            flip=flip,
            video_path=vpath,
            video_meta=vm,
            out_path=out_path,
        )
        if rec is None:
            failures.append(f"[{i}] render failed")
            continue
        payload.append(rec)
        print(f"  [{i:>2}/{len(TARGETS)}] wrote: {out_path.name}")

    html = _HTML_TEMPLATE.replace("__DATA_JSON__", json.dumps(payload, indent=2))
    Path(args.html).write_text(html)

    print()
    print(f"Wrote {len(payload)} / {len(TARGETS)} cases")
    if failures:
        print("Failures:")
        for f in failures:
            print(f"  - {f}")
    print(f"HTML: {args.html}")
    print(f"Frames dir: {FRAMES_DIR}")


if __name__ == "__main__":
    main()
