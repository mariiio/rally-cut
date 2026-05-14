"""Render multi-frame motion strips for the 17 S4 failure-mode cases.

For each case, extract source-video frames at offsets [-6, -3, 0, +3, +6] around
both the prev contact and the curr contact (10 frames per case). Annotate each:
  - All player bboxes (team colored, thin)
  - Ball position (yellow circle) when available
  - Tight zoom-crop bounding the action area (players + ball, with padding)
  - PREV strip: red halo on prev_attributed_pid
  - CURR strip: cyan halo on pipeline_pid, green halo on S4_pid (if different)
  - Frame label top-left: f={frame_num} offset={+/-N}

Composite each case into a single image:
  Top row: 5 prev frames horizontally.
  Bottom row: 5 curr frames horizontally.
  Header strip explaining the case.

Output:
  analysis/reports/probe_b_sequence_aware/2026_05_14/s4_motion_strips/{N:02d}_{video}_{rally_short}.jpg

Usage:
    cd analysis
    uv run python scripts/render_s4_motion_strip.py
    uv run python scripts/render_s4_motion_strip.py --only 10   # just case #10
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from rallycut.evaluation.tracking.db import get_connection
from rallycut.evaluation.video_resolver import VideoResolver

HERE = Path(__file__).resolve().parent
REPORT_DIR = HERE.parent / "reports" / "probe_b_sequence_aware" / "2026_05_14"
OUT_DIR = REPORT_DIR / "s4_motion_strips"
DEFAULT_FLIPS = REPORT_DIR / "s4_fleet_candidates.json"

# Order matches render_s4_dual_frame_review.py
TARGETS: list[tuple[str, int, str]] = [
    ("a0881d82-bd3e-4664-bd11-4e672ca29aa6", 128, "probe_b_cascade_anchor"),
    ("a0881d82-bd3e-4664-bd11-4e672ca29aa6", 225, "probe_b_cascade_anchor"),
    ("1f5ff17d-014f-4093-b6e6-c85360fb1638", 304, "attack_after_attack"),
    ("8d3205ed-b0dc-4c0c-bc24-fda34554e45f", 296, "attack_after_attack"),
    ("55565c2b-49b2-4817-b046-7bef02569b0e", 558, "attack_after_attack"),
    ("21a9b203-dc92-48dc-8f19-d94835e0e226", 528, "attack_after_attack"),
    ("49582c29-f09c-4b39-8435-bd19451812c8", 314, "attack_after_attack"),
    ("0793ebd2-5301-4317-a7ba-3fb3bf9fe368", 380, "attack_after_attack"),
    ("65c54107-6c0f-4888-9600-6ec06939c8b5", 220, "attack_after_attack"),
    ("594a832a-ab27-480e-976a-268c8f101559", 193, "attack_after_set"),
    ("ded9504c-ce40-4808-8d12-8bcc3b974b62", 230, "attack_after_set"),
    ("920d4a33-ae9a-4299-ac19-8e773ca76e61", 159, "attack_after_set"),
    ("a1a5baf7-e249-476e-9ab7-cf60fac04111", 679, "attack_after_set"),
    ("eeecf5a0-def4-43dd-8fc9-72e502165c2d", 533, "attack_after_set"),
    ("e50f127e-3952-4ba2-9047-39fff14a2e25", 233, "attack_after_set"),
    ("f62bc819-d7c4-4287-b617-edac3f5194bc", 177, "attack_after_set"),
    ("f8e251d8-c7ee-40fd-946c-502067343936", 127, "same_team_other"),
]

OFFSETS = [-6, -3, 0, 3, 6]
CROP_W = 600
CROP_H = 400
PAD_PX = 80  # padding around the players+ball bbox before zoom

# BGR colors
COLOR_PIPELINE = (255, 200, 60)   # cyan-ish
COLOR_S4 = (60, 220, 60)          # green
COLOR_PREV = (60, 60, 255)        # red
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


def _find_prev_action(actions: list[dict[str, Any]], curr_frame: int) -> dict[str, Any] | None:
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
        # No detections at all — fall back to centre crop matching target aspect.
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

    # Adjust to target aspect ratio (CROP_W / CROP_H = 1.5).
    target_ratio = CROP_W / CROP_H
    cw = x2 - x1
    ch = y2 - y1
    if cw / max(ch, 1) < target_ratio:
        # Too tall — widen.
        needed = int(ch * target_ratio)
        extra = needed - cw
        x1 = max(0, x1 - extra // 2)
        x2 = min(img_w, x2 + (extra - extra // 2))
        # If we hit image bounds, also pad the other side.
        cw = x2 - x1
        if cw < needed:
            if x1 == 0:
                x2 = min(img_w, x1 + needed)
            elif x2 == img_w:
                x1 = max(0, x2 - needed)
    else:
        # Too wide — heighten.
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

    halos: list of (tid, color, offset_px) — bigger offset draws the rectangle further inside.
    """
    H, W = frame.shape[:2]
    bboxes_px = _bboxes_to_pixel(positions, W, H)
    ball_px: tuple[int, int] | None = None
    if ball is not None:
        bx_n, by_n = ball
        if not (bx_n <= 0 and by_n <= 0):
            ball_px = (int(bx_n * W), int(by_n * H))

    # Draw thin team bboxes.
    for tid, (x1, y1, x2, y2) in bboxes_px.items():
        team = team_assignments.get(str(tid))
        color = _team_color(team)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        tag = f"p{tid}({team})" if team else f"p{tid}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        ly = max(y1 - 6, 14)
        cv2.rectangle(frame, (x1 - 2, ly - th - 4), (x1 + tw + 4, ly + 2), (0, 0, 0), -1)
        cv2.putText(frame, tag, (x1, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    # Halos.
    for tid, color, offset in halos:
        if tid not in bboxes_px:
            continue
        x1, y1, x2, y2 = bboxes_px[tid]
        cv2.rectangle(frame, (x1 + offset, y1 + offset), (x2 - offset, y2 - offset), color, 4)

    # Ball.
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

    # Top-left label.
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

    curr_action = _find_action_at(actions, curr_frame)
    prev_action = _find_prev_action(actions, curr_frame)
    if not curr_action or not prev_action:
        print(f"  [{idx}] missing curr/prev action at f={curr_frame}")
        return None

    prev_frame = int(prev_action["frame"])
    fps = float(video_meta.get("fps") or flip.get("fps") or 30.0)
    rally_start_frame = int(round(rally_start_ms / 1000.0 * fps))

    prev_pid = int(prev_action.get("playerTrackId", -1))
    prev_team = str(prev_action.get("team") or team_assignments.get(str(prev_pid), "?"))
    prev_action_type = str(prev_action.get("action", "?")).upper()
    curr_action_type = str(curr_action.get("action", "?")).upper()

    pipeline_pid = int(flip["pipeline_pid"])
    s4_pid = int(flip["s4_pid"])
    pipeline_team = str(flip["pipeline_team"])
    s4_team = str(flip["s4_team"])

    # Build the two strips (prev + curr) side-by-side as rows of 5 frames each.
    # First pass: pick a shared crop rect across all 5 frames in each row so the
    # zoom is steady. Use the contact-frame positions+ball as the anchor for
    # crop rect (most relevant frame; positions at +/-6 may be empty if the
    # rally is short or tracks drop out).
    def _build_row(
        anchor_frame: int,
        anchor_action_frame: int,
        anchor_action: dict[str, Any],
        halos_at_zero: list[tuple[int, tuple[int, int, int], int]],
    ) -> tuple[list[np.ndarray], int | None]:
        # Compute shared crop rect using ANY available frame in the row.
        # Prefer offset 0; fall back to nearest available.
        shared_crop_rect: tuple[int, int, int, int] | None = None
        # First pass — pick anchor frame to compute crop.
        for off in [0, -3, 3, -6, 6]:
            target_pl_frame = anchor_action_frame + off
            pos = positions_by_frame.get(target_pl_frame, {})
            if not pos:
                continue
            # Get ball for this frame.
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
            # Ball
            if off == 0:
                ball = ball_by_frame.get(target_pl_frame, (
                    float(anchor_action.get("ballX", 0.0)),
                    float(anchor_action.get("ballY", 0.0)),
                ))
            else:
                ball = ball_by_frame.get(target_pl_frame, (0.0, 0.0))
            halos = halos_at_zero if off == 0 else _halos_neutral(halos_at_zero)
            label = f"f={target_pl_frame}  off={off:+d}"
            cropped, _ = _annotate_and_crop_frame(
                extracted, pos, team_assignments,
                ball if not (ball[0] <= 0 and ball[1] <= 0) else None,
                halos, label,
                crop_rect_hint=shared_crop_rect,
            )
            row.append(cropped)
        return row, shared_crop_rect[0] if shared_crop_rect else None

    # PREV row: red halo on prev_pid.
    prev_halos: list[tuple[int, tuple[int, int, int], int]] = [
        (prev_pid, COLOR_PREV, -4)
    ]
    # CURR row: cyan halo on pipeline_pid + green halo on s4_pid (offset
    # nested so both visible if same).
    curr_halos: list[tuple[int, tuple[int, int, int], int]] = []
    if pipeline_pid >= 0:
        curr_halos.append((pipeline_pid, COLOR_PIPELINE, -4))
    if s4_pid >= 0 and s4_pid != pipeline_pid:
        curr_halos.append((s4_pid, COLOR_S4, -4))
    elif s4_pid == pipeline_pid and s4_pid >= 0:
        # Same pid: stack a green ring outside the cyan one.
        curr_halos.append((s4_pid, COLOR_S4, -10))

    prev_row, _ = _build_row(prev_frame, prev_frame, prev_action, prev_halos)
    curr_row, _ = _build_row(curr_frame, curr_frame, curr_action, curr_halos)

    if len(prev_row) != len(OFFSETS) or len(curr_row) != len(OFFSETS):
        print(f"  [{idx}] row build failed (prev={len(prev_row)} curr={len(curr_row)})")
        return None

    # Compose: each row hconcat with 8px separators; row separators 8px black; header bar.
    sep_v = np.full((CROP_H, 6, 3), 30, dtype=np.uint8)
    prev_concat = prev_row[0]
    for f in prev_row[1:]:
        prev_concat = cv2.hconcat([prev_concat, sep_v, f])
    curr_concat = curr_row[0]
    for f in curr_row[1:]:
        curr_concat = cv2.hconcat([curr_concat, sep_v, f])

    full_w = prev_concat.shape[1]
    sep_h = np.full((10, full_w, 3), 30, dtype=np.uint8)

    # Row tag strips on the left? Cleaner: a thin strip above each row labeling PREV / CURR.
    row_tag_h = 28

    def _row_tag(text: str, color: tuple[int, int, int]) -> np.ndarray:
        strip = np.full((row_tag_h, full_w, 3), 30, dtype=np.uint8)
        cv2.putText(strip, text, (10, row_tag_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        return strip

    prev_tag = _row_tag(
        f"PREV  {prev_action_type}@f{prev_frame}  prev_attributed_pid=p{prev_pid}({prev_team})",
        COLOR_PREV,
    )
    curr_tag_lines: list[str] = []
    curr_tag_lines.append(
        f"CURR  {curr_action_type}@f{curr_frame}  "
        f"PIPELINE=p{pipeline_pid}({pipeline_team})  "
        f"S4=p{s4_pid}({s4_team})"
    )
    curr_tag = _row_tag(curr_tag_lines[0], (220, 220, 220))

    # Header strip with case title.
    header_h = 44
    header = np.full((header_h, full_w, 3), 20, dtype=np.uint8)
    title = (
        f"#{idx} [{bucket}] {flip['video_name']}/{flip['rally_short']}  "
        f"PREV: {prev_action_type}@f{prev_frame} pid=p{prev_pid}({prev_team})  ->  "
        f"CURR: {curr_action_type}@f{curr_frame} "
        f"PL=p{pipeline_pid}({pipeline_team}) S4=p{s4_pid}({s4_team})"
    )
    cv2.putText(header, title, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (245, 245, 245), 1, cv2.LINE_AA)

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
        "idx": idx,
        "bucket": bucket,
        "video_name": flip["video_name"],
        "rally_short": flip["rally_short"],
        "rally_id": rally_id,
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
        "img": str(out_path.relative_to(REPORT_DIR)),
    }


def _halos_neutral(
    halos_at_zero: list[tuple[int, tuple[int, int, int], int]],
) -> list[tuple[int, tuple[int, int, int], int]]:
    """For non-contact frames, keep the same halos but make them thinner/lighter so the
    moment-of-contact frame visually stands out. We still want to track the player
    through the strip, so we keep the halos but dimmed."""
    out = []
    for tid, (b, g, r), offset in halos_at_zero:
        # Dim the color ~50%.
        dim = (int(b * 0.55), int(g * 0.55), int(r * 0.55))
        out.append((tid, dim, offset))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flips", type=str, default=str(DEFAULT_FLIPS))
    parser.add_argument("--only", type=int, default=None,
                        help="Render only the case with this 1-based index")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = json.loads(Path(args.flips).read_text())
    flips = data["flips"]
    flip_by_key: dict[tuple[str, int], dict[str, Any]] = {
        (f["rally_id"], f["pl_frame"]): f for f in flips
    }

    resolver = VideoResolver()
    video_meta_cache: dict[str, dict[str, Any]] = {}
    video_path_cache: dict[str, Path | None] = {}

    targets = list(enumerate(TARGETS, start=1))
    if args.only is not None:
        targets = [t for t in targets if t[0] == args.only]
    print(f"Rendering {len(targets)} motion strips...", flush=True)

    successes = 0
    failures: list[str] = []
    for i, (rally_id, curr_frame, bucket) in targets:
        flip = flip_by_key.get((rally_id, curr_frame))
        if not flip:
            msg = f"[{i}] miss: ({rally_id[:8]}, {curr_frame})"
            print(f"  {msg}", flush=True)
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
            print(f"  {msg}", flush=True)
            failures.append(msg)
            continue

        out_name = f"{i:02d}_{flip['video_name']}_{flip['rally_short']}.jpg"
        out_path = OUT_DIR / out_name
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
        successes += 1
        print(f"  [{i:>2}/{len(TARGETS)}] wrote: {out_path.name}", flush=True)

    print()
    print(f"Wrote {successes} / {len(targets)} cases")
    if failures:
        print("Failures:")
        for f in failures:
            print(f"  - {f}")
    print(f"Output dir: {OUT_DIR}")


if __name__ == "__main__":
    main()
