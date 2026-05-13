"""Build the A1 visual verification HTML page (2026-05-13).

For each of the 10 A1 fixes documented in
`analysis/reports/a1_volleyball_rule/visual_verify_sample_2026_05_13.md`:

1. Download the source video proxy from MinIO (via VideoResolver).
2. Seek to the contact frame in the source coordinate system
   (rally.start_ms * fps / 1000 + contact_frame_in_rally).
3. Annotate the frame with:
   - All player bboxes from `positions_json` at that frame, colored by team.
   - Thick RED outline on the pre-A1 ("before") player.
   - Thick GREEN outline on the post-A1 ("after") player.
   - Yellow ball circle at `actions_json.actions[i].ballX/ballY`.
   - Action label + frame number banner.
4. Also extract the prev-action frame with the same annotation conventions
   (for the optional 2-frame pre/post strip).
5. Save annotated frames to
   `analysis/reports/a1_volleyball_rule/visual_verify_frames/`.
6. Emit `analysis/reports/a1_volleyball_rule/visual_verify.html` — a
   single page with 10 cards, verdict buttons (✅/❌/⚠️), and a
   "Copy verdicts" button.

Usage:
    cd analysis
    uv run python scripts/build_a1_visual_verify_page.py
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from rallycut.evaluation.tracking.db import get_connection  # noqa: E402
from rallycut.evaluation.video_resolver import VideoResolver  # noqa: E402

REPORT_DIR = (
    HERE.parent / "reports" / "a1_volleyball_rule"
)
FRAMES_DIR = REPORT_DIR / "visual_verify_frames"
HTML_PATH = REPORT_DIR / "visual_verify.html"


# ---------------------------------------------------------------------------
# Hand-curated list (mirrors the markdown table).
# Re-running with the same markdown produces the same output.
# ---------------------------------------------------------------------------

@dataclass
class FixSpec:
    idx: int
    video_name: str
    video_id: str
    rally_id: str
    rally_short: str
    fps: float  # informational; we read from video metadata
    start_ms: int  # informational
    prev_frame: int
    prev_action: str
    prev_player_label: str
    prev_team_label: str
    curr_frame: int
    curr_action: str
    a1_target: str  # "prev" or "curr"
    pid_before: int
    pid_after: int
    alt_ratio: float
    prev_candidates: str
    curr_candidates: str


FIXES: list[FixSpec] = [
    FixSpec(1, "juju", "06f0b063-b3f9-40aa-b13b-fba1edd74a85",
            "d810943e-dd1c-4518-a6af-577b22555c3c", "d810943e",
            30.0, 35400,
            252, "dig", "p4", "unknown",
            345, "set", "curr", 4, 3, 2.70,
            "p2(A)@0.0282, p4(B)@0.0294, p3(B)@0.0416",
            "p4(B)@0.0122, p3(B)@0.0330, p2(A)@0.0620"),
    FixSpec(2, "gigi", "b097dd2a-6953-4e0e-a603-5be3552f462e",
            "72c8229b-2993-4310-9b61-cd6162cc27fa", "72c8229b",
            30.0, 4200,
            357, "set", "p3", "B",
            399, "attack", "prev", 3, 4, 2.66,
            "p3(B)@0.0291, p1(A)@0.0473, p2(A)@0.0496",
            "p3(B)@0.0454, p1(A)@0.0720, p4(B)@0.1208"),
    FixSpec(3, "caca", "627c1add-8a80-42ab-8278-3617880ebf81",
            "ae99ab2a-e342-4096-9225-6cfbb3909d15", "ae99ab2a",
            60.0, 36600,
            241, "receive", "p1", "A",
            353, "set", "curr", 1, 2, 4.58,
            "p1(A)@0.0308, p3(B)@0.0774, p2(A)@0.1878",
            "p1(A)@0.0332, p3(B)@0.0644, p2(A)@0.1519"),
    FixSpec(4, "machi", "23a5f798-78a0-4b3a-8647-b4a2166274b1",
            "f406f4b3-95df-474a-aecc-a5ba7239ad9b", "f406f4b3",
            25.0, 55740,
            242, "serve", "p1", "A",
            333, "attack", "curr", 1, 2, 4.53,
            "p1(A)@0.1060, p2(A)@0.3075, p3(B)@0.3075",
            "p1(A)@0.0301, p2(A)@0.1361, p4(B)@0.2362"),
    FixSpec(5, "matchop", "920ba69d-2526-4e6c-a357-c44af3bf5c99",
            "f433967e-2c40-4169-b5cb-87f48cd0fa63", "f433967e",
            29.97, 170640,
            83, "receive", "p4", "A",
            100, "receive", "prev", 4, 3, 5.42,
            "p4(A)@0.0448, p3(A)@0.1444, p2(B)@0.3561",
            "p4(A)@0.0162, p3(A)@0.0876, p2(B)@0.2597"),
    FixSpec(6, "lala", "84e66e74-8d4f-420a-ad01-0ada95153ad0",
            "2eeb3ae6-cf97-4eeb-9400-28a8060a7636", "2eeb3ae6",
            29.75, 233781,
            569, "dig", "p1", "B",
            616, "dig", "curr", 1, 2, 8.47,
            "p1(B)@0.0233, p3(A)@0.0626, p2(B)@0.1407",
            "p3(A)@0.0123, p1(B)@0.0269, p2(B)@0.1872"),
    FixSpec(7, "mech", "c6e4c876-beca-4cb8-9cce-4a4fc70553f1",
            "b0dabe43-7ddb-4544-8d2c-e86032a8d8f5", "b0dabe43",
            29.97, 135702,
            39, "serve", "p3", "A",
            59, "receive", "curr", 3, 4, 19.74,
            "p3(A)@0.1779, p4(A)@0.3985, p1(B)@0.4606",
            "p3(A)@0.0066, p4(A)@0.1312, p1(B)@0.1394"),
    FixSpec(8, "veve", "43928971-2e07-4814-bb1a-3d91c7bf03b2",
            "4c27b635-fbab-4bcb-a30e-f82a87c223c2", "4c27b635",
            30.0, 87800,
            265, "attack", "p3", "B",
            284, "dig", "curr", 3, 4, 15.16,
            "p3(B)@0.0039, p2(A)@0.0626, p1(A)@0.2057",
            "p3(B)@0.0096, p2(A)@0.0117, p4(B)@0.1453"),
    FixSpec(9, "natch", "a7ee3d38-a3a9-4dcd-a2af-e0617997e708",
            "e5e4c0b7-7f18-493f-b95b-574e51821452", "e5e4c0b7",
            30.0, 46027,
            110, "receive", "p1", "A",
            214, "attack", "curr", 1, 2, 2.02,
            "p4(B)@0.1508, p1(A)@0.1735, p3(B)@0.1950",
            "p1(A)@0.0759, p2(A)@0.1533, p3(B)@0.1741"),
    FixSpec(10, "matttch", "23b662ba-99e0-47d6-a9ac-90bb6fa9bdd1",
            "8d3205ed-b0dc-4c0c-bc24-fda34554e45f", "8d3205ed",
            29.97, 210172,
            220, "attack", "p3", "B",
            296, "attack", "prev", 3, 4, 2.41,
            "p3(B)@0.0250, p4(B)@0.1652, p1(A)@0.2713",
            "p3(B)@0.0777, p4(B)@0.1874, p1(A)@0.3387"),
]


# ---------------------------------------------------------------------------
# DB queries
# ---------------------------------------------------------------------------

def _query_video(video_id: str) -> dict[str, Any]:
    q = """SELECT id, name, filename, fps, width, height,
                  s3_key, proxy_s3_key, processed_s3_key, content_hash
           FROM videos WHERE id = %s"""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(q, (video_id,))
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


def _query_rally(rally_id: str) -> dict[str, Any]:
    q = """SELECT r.start_ms, r.end_ms, pt.positions_json, pt.actions_json,
                  pt.fps
           FROM rallies r JOIN player_tracks pt ON pt.rally_id = r.id
           WHERE r.id = %s"""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(q, (rally_id,))
        r = cur.fetchone()
    if r is None:
        return {}
    return {
        "start_ms": int(r[0] or 0), "end_ms": int(r[1] or 0),
        "positions": r[2] or [], "actions_json": r[3] or {},
        "fps": float(r[4]) if r[4] is not None else None,
    }


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------

# BGR colors
COLOR_TEAM_A = (255, 120, 50)   # blue-ish
COLOR_TEAM_B = (60, 60, 235)    # red-ish
COLOR_UNKNOWN = (160, 160, 160)
COLOR_BEFORE = (0, 0, 255)      # red — pre-A1 pick
COLOR_AFTER = (0, 230, 0)       # green — post-A1 pick
COLOR_BALL = (0, 255, 255)      # yellow


def _team_for_track(track_id: int, team_assignments: dict[str, Any]) -> str | None:
    """team_assignments maps str(trackId) -> team label.

    Observed serialization forms:
      - string label: "A" / "B" (current production schema)
      - int 0/1 (legacy from contact_detector internals)
    """
    if not team_assignments:
        return None
    raw = team_assignments.get(str(track_id))
    if raw is None:
        return None
    # String label path
    if isinstance(raw, str):
        s = raw.strip().upper()
        if s in {"A", "B"}:
            return s
        # tolerate "0"/"1" strings
        try:
            return "A" if int(s) == 0 else "B"
        except ValueError:
            return None
    # Int path
    try:
        return "A" if int(raw) == 0 else "B"
    except (TypeError, ValueError):
        return None


def _team_color(team: str | None) -> tuple[int, int, int]:
    if team == "A":
        return COLOR_TEAM_A
    if team == "B":
        return COLOR_TEAM_B
    return COLOR_UNKNOWN


def _ball_xy_for_action(
    actions_json: dict[str, Any], frame: int,
) -> tuple[float, float] | None:
    """Return (ballX, ballY) normalized for the action at `frame`."""
    for a in (actions_json.get("actions") or []):
        if int(a.get("frame", -1)) == frame:
            bx = a.get("ballX")
            by = a.get("ballY")
            if bx is None or by is None:
                return None
            try:
                return (float(bx), float(by))
            except (TypeError, ValueError):
                return None
    return None


def _annotate_frame(
    frame_bgr,
    *,
    positions_at_frame: list[dict[str, Any]],
    team_assignments: dict[str, int],
    pid_before: int,
    pid_after: int,
    ball_xy: tuple[float, float] | None,
    title_top: str,
    title_bottom: str,
):
    img = frame_bgr.copy()
    h, w = img.shape[:2]

    # Player bboxes
    for p in positions_at_frame:
        try:
            tid = int(p["trackId"])
        except (KeyError, TypeError, ValueError):
            continue
        if tid < 0:
            continue
        cx = float(p.get("x", 0.0))
        cy = float(p.get("y", 0.0))
        bw = float(p.get("width", 0.0))
        bh = float(p.get("height", 0.0))
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)

        team = _team_for_track(tid, team_assignments)
        color = _team_color(team)
        thickness = 2

        # Special outlines for before/after picks
        is_before = (tid == pid_before)
        is_after = (tid == pid_after)
        # Draw team-colored bbox first (thin)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        if is_before and is_after:
            # Same track is both — unlikely for a real A1 fix. Treat as AFTER.
            is_before = False

        if is_before:
            cv2.rectangle(img, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), COLOR_BEFORE, 5)
            label = f"BEFORE: p{tid} ({team or '?'})"
            label_color = COLOR_BEFORE
            label_y = max(y1 - 12, 18)
        elif is_after:
            cv2.rectangle(img, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), COLOR_AFTER, 5)
            label = f"AFTER: p{tid} ({team or '?'})"
            label_color = COLOR_AFTER
            label_y = max(y1 - 12, 18)
        else:
            label = f"p{tid} ({team or '?'})"
            label_color = (255, 255, 255)
            label_y = max(y1 - 6, 14)

        # Backdrop for legibility
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1 - 2, label_y - th - 4),
                      (x1 + tw + 4, label_y + 2), (0, 0, 0), -1)
        cv2.putText(img, label, (x1, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 1, cv2.LINE_AA)

    # Ball
    if ball_xy is not None:
        bx_n, by_n = ball_xy
        if bx_n > 0 or by_n > 0:
            bx = int(bx_n * w)
            by = int(by_n * h)
            cv2.circle(img, (bx, by), 16, COLOR_BALL, 3)
            cv2.circle(img, (bx, by), 2, COLOR_BALL, -1)

    # Title banner (top, two lines)
    banner_h = 56
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    cv2.putText(img, title_top, (12, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, title_bottom, (12, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    # Mini legend bottom-right
    legend_lines = [
        ("BEFORE (pre-A1)", COLOR_BEFORE),
        ("AFTER  (post-A1)", COLOR_AFTER),
        ("Team A           ", COLOR_TEAM_A),
        ("Team B           ", COLOR_TEAM_B),
        ("Ball             ", COLOR_BALL),
    ]
    lh = 18
    lw_box = 200
    ly0 = h - lh * len(legend_lines) - 8
    overlay = img.copy()
    cv2.rectangle(overlay, (w - lw_box - 8, ly0 - 6),
                  (w - 4, h - 4), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    for i, (txt, col) in enumerate(legend_lines):
        y = ly0 + i * lh + 12
        cv2.rectangle(img, (w - lw_box, y - 10), (w - lw_box + 18, y + 2), col, -1)
        cv2.putText(img, txt, (w - lw_box + 24, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)

    return img


def _positions_at_frame(positions: list[dict[str, Any]], frame: int) -> list[dict[str, Any]]:
    return [p for p in positions if int(p.get("frameNumber", -1)) == frame]


# ---------------------------------------------------------------------------
# Per-fix extraction
# ---------------------------------------------------------------------------

@dataclass
class CardData:
    idx: int
    fix: FixSpec
    video_meta: dict[str, Any]
    rally_meta: dict[str, Any]
    curr_frame_path: Path | None
    prev_frame_path: Path | None
    error: str | None


def _resolve_video(resolver: VideoResolver, vm: dict[str, Any]) -> Path | None:
    """Try proxy_s3_key first, fall back to s3_key."""
    candidates = []
    if vm.get("proxy_s3_key"):
        candidates.append(("proxy", vm["proxy_s3_key"]))
    if vm.get("s3_key"):
        candidates.append(("original", vm["s3_key"]))
    if vm.get("processed_s3_key"):
        candidates.append(("processed", vm["processed_s3_key"]))
    last_err = None
    for label, key in candidates:
        try:
            path = resolver.resolve(key, vm["content_hash"])
            print(f"    resolved via {label}: {path}")
            return path
        except Exception as e:  # noqa: BLE001
            last_err = f"{label}: {e}"
            print(f"    failed via {label}: {e}")
    if last_err:
        print(f"    all candidates failed; last: {last_err}")
    return None


def _extract_card(
    fix: FixSpec, resolver: VideoResolver,
) -> CardData:
    print(f"\n[{fix.idx}/10] {fix.video_name} rally {fix.rally_short} "
          f"frames prev={fix.prev_frame} curr={fix.curr_frame}")
    vm = _query_video(fix.video_id)
    if not vm:
        return CardData(fix.idx, fix, {}, {}, None, None,
                        error=f"video {fix.video_id} not found in DB")
    rm = _query_rally(fix.rally_id)
    if not rm:
        return CardData(fix.idx, fix, vm, {}, None, None,
                        error=f"rally {fix.rally_id} not found in DB")

    # Resolve video
    video_path = _resolve_video(resolver, vm)
    if video_path is None:
        return CardData(fix.idx, fix, vm, rm, None, None,
                        error="could not download any video variant from S3/MinIO")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return CardData(fix.idx, fix, vm, rm, None, None,
                        error=f"OpenCV could not open {video_path}")

    vid_fps = cap.get(cv2.CAP_PROP_FPS) or vm.get("fps", 30.0)
    # Rally-relative frame -> source frame
    rally_start_frame = int(round(rm["start_ms"] / 1000.0 * vid_fps))
    print(f"    vid_fps={vid_fps:.3f} rally_start_ms={rm['start_ms']} "
          f"rally_start_frame={rally_start_frame}")

    team_assignments = (rm["actions_json"].get("teamAssignments") or {})

    def _do_frame(frame_in_rally: int, kind: str) -> Path | None:
        source_frame = rally_start_frame + frame_in_rally
        cap.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
        ok, img = cap.read()
        if not ok or img is None:
            print(f"    [{kind}] FAILED to read source frame {source_frame}")
            return None

        positions = _positions_at_frame(rm["positions"], frame_in_rally)
        ball_xy = _ball_xy_for_action(rm["actions_json"], frame_in_rally)

        action_label = fix.curr_action if kind == "curr" else fix.prev_action
        sec = frame_in_rally / float(vid_fps)
        mm = int(sec // 60)
        ss = sec - mm * 60
        time_str = f"{mm:02d}:{ss:06.3f}"
        title_top = f"{action_label.upper()} @ rally frame {frame_in_rally} "
        title_top += f"(source f={source_frame}, t={time_str})"
        title_bottom = (
            f"{fix.video_name} / rally {fix.rally_short} | "
            f"alt_ratio {fix.alt_ratio:.2f}x | "
            f"BEFORE p{fix.pid_before} AFTER p{fix.pid_after} | "
            f"{'CURR FRAME (changed by A1)' if kind == 'curr' else 'PREV FRAME (context)'}"
        )
        annotated = _annotate_frame(
            img,
            positions_at_frame=positions,
            team_assignments=team_assignments,
            pid_before=fix.pid_before if kind == "curr" else -1,
            pid_after=fix.pid_after if kind == "curr" else -1,
            ball_xy=ball_xy,
            title_top=title_top,
            title_bottom=title_bottom,
        )
        out_name = f"{fix.idx:02d}_{fix.video_name}_{kind}_f{frame_in_rally}.jpg"
        out_path = FRAMES_DIR / out_name
        cv2.imwrite(str(out_path), annotated,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        print(f"    [{kind}] wrote {out_path.name} "
              f"(players={len(positions)}, ball={ball_xy})")
        return out_path

    # For prev frame, highlight the prev-action's player only as a thin "context" bbox
    # — but we still want to highlight the actor of the prev action so the user can see
    # "this is who the prev action was attributed to". We use the prev pid as both
    # before and after for that frame so it gets the green outline (no red, no diff).
    # Actually clearer: use the pid that A1 attributed prev (which may be pid_before
    # if A1 didn't move prev, or pid_after if A1 moved prev). For now: highlight
    # whichever pid was the prev action's player_track_id in the new attribution.
    # The fix metadata tells us a1_target — if "prev", A1 moved prev → new pid is
    # pid_after; otherwise prev kept its old attribution (pid_before).
    prev_pid_after_a1 = fix.pid_after if fix.a1_target == "prev" else fix.pid_before
    prev_pid_before_a1 = fix.pid_before if fix.a1_target == "prev" else fix.pid_before
    # For prev, only highlight the "after" pick (no diff drawing).

    def _do_prev_frame() -> Path | None:
        frame_in_rally = fix.prev_frame
        source_frame = rally_start_frame + frame_in_rally
        cap.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
        ok, img = cap.read()
        if not ok or img is None:
            print(f"    [prev] FAILED to read source frame {source_frame}")
            return None

        positions = _positions_at_frame(rm["positions"], frame_in_rally)
        ball_xy = _ball_xy_for_action(rm["actions_json"], frame_in_rally)

        sec = frame_in_rally / float(vid_fps)
        mm = int(sec // 60)
        ss = sec - mm * 60
        time_str = f"{mm:02d}:{ss:06.3f}"
        title_top = (
            f"{fix.prev_action.upper()} @ rally frame {frame_in_rally} "
            f"(source f={source_frame}, t={time_str})"
        )
        title_bottom = (
            f"{fix.video_name} / rally {fix.rally_short} | "
            f"PREV (context) | prev-action player after A1 = p{prev_pid_after_a1}"
        )
        # If A1 moved prev (a1_target == "prev"), show prev pid_before in RED
        # and pid_after in GREEN; otherwise just GREEN around the kept player.
        if fix.a1_target == "prev":
            pid_before_on_prev = fix.pid_before
            pid_after_on_prev = fix.pid_after
        else:
            pid_before_on_prev = -1
            pid_after_on_prev = prev_pid_after_a1

        annotated = _annotate_frame(
            img,
            positions_at_frame=positions,
            team_assignments=team_assignments,
            pid_before=pid_before_on_prev,
            pid_after=pid_after_on_prev,
            ball_xy=ball_xy,
            title_top=title_top,
            title_bottom=title_bottom,
        )
        out_name = f"{fix.idx:02d}_{fix.video_name}_prev_f{frame_in_rally}.jpg"
        out_path = FRAMES_DIR / out_name
        cv2.imwrite(str(out_path), annotated,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        print(f"    [prev] wrote {out_path.name} "
              f"(players={len(positions)}, ball={ball_xy})")
        return out_path

    # For curr frame, the "before"/"after" semantics apply directly.
    if fix.a1_target == "curr":
        curr_path = _do_frame(fix.curr_frame, "curr")
    else:
        # A1 moved prev — curr was not changed. Highlight curr's actor (pid_before
        # for both before/after, drawing only the GREEN outline).
        curr_path = None
        # We still want to render the curr frame for context.
        frame_in_rally = fix.curr_frame
        source_frame = rally_start_frame + frame_in_rally
        cap.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
        ok, img = cap.read()
        if ok and img is not None:
            positions = _positions_at_frame(rm["positions"], frame_in_rally)
            ball_xy = _ball_xy_for_action(rm["actions_json"], frame_in_rally)
            sec = frame_in_rally / float(vid_fps)
            mm = int(sec // 60)
            ss = sec - mm * 60
            time_str = f"{mm:02d}:{ss:06.3f}"
            title_top = (
                f"{fix.curr_action.upper()} @ rally frame {frame_in_rally} "
                f"(source f={source_frame}, t={time_str})"
            )
            title_bottom = (
                f"{fix.video_name} / rally {fix.rally_short} | "
                f"CURR (kept by A1) | curr-action player = p{fix.pid_before}"
            )
            annotated = _annotate_frame(
                img,
                positions_at_frame=positions,
                team_assignments=team_assignments,
                pid_before=-1,
                pid_after=fix.pid_before,
                ball_xy=ball_xy,
                title_top=title_top,
                title_bottom=title_bottom,
            )
            out_name = f"{fix.idx:02d}_{fix.video_name}_curr_f{frame_in_rally}.jpg"
            out_path = FRAMES_DIR / out_name
            cv2.imwrite(str(out_path), annotated,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            print(f"    [curr] wrote {out_path.name} "
                  f"(players={len(positions)}, ball={ball_xy})")
            curr_path = out_path

    prev_path = _do_prev_frame()
    cap.release()
    return CardData(fix.idx, fix, vm, rm, curr_path, prev_path, error=None)


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

def _render_html(cards: list[CardData]) -> str:
    # Build per-card JSON for client-side logic.
    rows = []
    for c in cards:
        fix = c.fix
        # Compute changed frame indicator (the one A1 changed)
        rows.append({
            "idx": c.idx,
            "video": fix.video_name,
            "rally_short": fix.rally_short,
            "action": fix.curr_action,
            "alt_ratio": fix.alt_ratio,
            "pid_before": fix.pid_before,
            "pid_after": fix.pid_after,
            "a1_target": fix.a1_target,
            "prev_action_desc": (
                f"{fix.prev_action} @ frame {fix.prev_frame}, "
                f"player p{fix.prev_player_label.lstrip('p')} ({fix.prev_team_label})"
            ),
            "curr_action_desc": (
                f"{fix.curr_action} @ frame {fix.curr_frame}"
            ),
            "prev_candidates": fix.prev_candidates,
            "curr_candidates": fix.curr_candidates,
            "curr_frame_path": (
                f"visual_verify_frames/{c.curr_frame_path.name}"
                if c.curr_frame_path else None
            ),
            "prev_frame_path": (
                f"visual_verify_frames/{c.prev_frame_path.name}"
                if c.prev_frame_path else None
            ),
            "error": c.error,
            "summary_line": (
                f"{'CURR' if fix.a1_target == 'curr' else 'PREV'} "
                f"p{fix.pid_before} → p{fix.pid_after}"
            ),
        })
    rows_json = json.dumps(rows, indent=2)

    return _HTML_TEMPLATE.replace("__ROWS_JSON__", rows_json)


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>A1 Visual Verification (10 fixes) — 2026-05-13</title>
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
           padding:14px 22px; display:flex; gap:18px; align-items:center;
           z-index:20; }
  header h1 { margin:0; font-size:17px; }
  header .status { color:var(--muted); font-size:13px; }
  header .grow { flex:1; }
  header button { background:#222; color:#eee; border:1px solid #333; padding:8px 14px;
                  border-radius:6px; font-weight:600; cursor:pointer; font-size:13px; }
  header button.primary { background:var(--accent); color:#000; border-color:transparent; }
  header button:hover { background:#2d2d2d; }
  header button.primary:hover { background:#789fef; }

  .container { max-width:1500px; margin:0 auto; padding:24px 22px 80px; }
  .card { background:var(--card); border:1px solid var(--border); border-radius:10px;
          margin-bottom:26px; overflow:hidden; }
  .card-head { padding:14px 20px; display:flex; align-items:center; gap:14px;
               background:#161617; border-bottom:1px solid var(--border); }
  .card-head .num { font-size:20px; font-weight:700; color:var(--accent);
                    min-width:34px; }
  .card-head .title { font-size:16px; font-weight:600; }
  .card-head .pill { font-size:11px; padding:3px 9px; border-radius:12px;
                     background:#222; color:#cfcfcf; }
  .card-head .pill.alt { background:#332; color:#ffd599; }
  .card-head .grow { flex:1; }
  .card-head .verdict-status { font-size:13px; color:var(--muted); }
  .card-head .verdict-status.ok   { color:var(--ok); font-weight:700; }
  .card-head .verdict-status.bad  { color:var(--bad); font-weight:700; }
  .card-head .verdict-status.warn { color:var(--warn); font-weight:700; }

  .card-body { display:grid; grid-template-columns:minmax(0,2.4fr) minmax(280px,1fr);
               gap:18px; padding:18px 20px; }
  @media (max-width:1000px) { .card-body { grid-template-columns:1fr; } }

  .frames { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
  @media (max-width:760px) { .frames { grid-template-columns:1fr; } }
  .frame { background:#000; border-radius:6px; overflow:hidden; position:relative; }
  .frame img { display:block; width:100%; max-height:62vh; object-fit:contain;
               background:#000; cursor:zoom-in; }
  .frame .frame-tag { position:absolute; top:8px; left:8px;
                      background:rgba(0,0,0,.75); color:#fff; font-size:11px;
                      padding:2px 8px; border-radius:10px; letter-spacing:.04em; }
  .frame.error { aspect-ratio:16/9; display:flex; align-items:center;
                 justify-content:center; color:var(--bad); padding:20px;
                 background:#1f0f0f; }

  .sidebar { font-size:13px; }
  .sidebar dl { margin:0; display:grid; grid-template-columns:auto 1fr;
                column-gap:10px; row-gap:5px; }
  .sidebar dt { color:var(--muted); }
  .sidebar dd { margin:0; font-family:ui-monospace,Menlo,monospace;
                font-size:12px; word-break:break-word; }
  .sidebar .group + .group { margin-top:12px;
                              padding-top:12px;
                              border-top:1px dashed #2a2a2a; }
  .sidebar .change-arrow { color:var(--warn); font-weight:700; }

  .verdicts { display:grid; grid-template-columns:repeat(3,1fr); gap:8px;
              margin-top:16px; }
  .verdicts button { background:#1f1f22; color:#eee; border:2px solid #2a2a2e;
                     border-radius:8px; padding:14px 8px; font-size:14px;
                     font-weight:600; cursor:pointer; }
  .verdicts button:hover { transform:translateY(-1px); }
  .verdicts button .icon { display:block; font-size:20px; margin-bottom:4px; }
  .verdicts button.ok:hover   { border-color:var(--ok);   background:#0f261c; }
  .verdicts button.bad:hover  { border-color:var(--bad);  background:#291313; }
  .verdicts button.warn:hover { border-color:var(--warn); background:#2a210e; }
  .verdicts button.ok.sel   { border-color:var(--ok);   background:#0f261c; }
  .verdicts button.bad.sel  { border-color:var(--bad);  background:#291313; }
  .verdicts button.warn.sel { border-color:var(--warn); background:#2a210e; }

  .summary-bar { position:fixed; bottom:0; left:0; right:0;
                 background:#0a0a0a; border-top:1px solid #222;
                 padding:14px 22px;
                 display:flex; gap:14px; align-items:center;
                 z-index:30; }
  .summary-bar .pill { background:#1d1d1d; color:#ddd; padding:6px 12px;
                       border-radius:14px; font-size:13px;
                       font-family:ui-monospace,Menlo,monospace; }
  .summary-bar .grow { flex:1; }
  .summary-bar button { background:var(--accent); color:#000; border:none;
                        padding:10px 18px; border-radius:6px; font-weight:700;
                        font-size:14px; cursor:pointer; }
  .summary-bar button.secondary { background:#222; color:#eee; }

  #zoom-overlay { display:none; position:fixed; inset:0;
                  background:rgba(0,0,0,.94); z-index:99;
                  align-items:center; justify-content:center; cursor:zoom-out; }
  #zoom-overlay.open { display:flex; }
  #zoom-overlay img { max-width:96vw; max-height:96vh; }

  #copy-modal { display:none; position:fixed; inset:0;
                background:rgba(0,0,0,.7); z-index:99;
                align-items:center; justify-content:center; }
  #copy-modal.open { display:flex; }
  #copy-modal .box { background:#1a1a1c; border:1px solid #333;
                     border-radius:10px; padding:20px; max-width:680px;
                     width:90%; }
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
  <h1>A1 Visual Verification — 10 fixes</h1>
  <span class="status" id="status">0 / 10 verdicts</span>
  <span class="grow"></span>
  <button id="copy-btn" class="primary">Copy verdicts</button>
  <button id="reset-btn">Reset</button>
</header>

<div class="container" id="container"></div>

<div class="summary-bar">
  <span class="pill" id="counts">✅ 0 &nbsp; ❌ 0 &nbsp; ⚠️ 0 &nbsp; (· 10 pending)</span>
  <span class="grow"></span>
  <button class="secondary" id="reset-btn-2">Reset all</button>
  <button id="copy-btn-2">Copy verdicts</button>
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

<script id="rows-data" type="application/json">__ROWS_JSON__</script>
<script>
const ROWS = JSON.parse(document.getElementById('rows-data').textContent);
const STORAGE_KEY = 'a1_visual_verify_2026_05_13';

function loadVerdicts() {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY)) || {}; }
  catch (_) { return {}; }
}
function saveVerdicts(v) { localStorage.setItem(STORAGE_KEY, JSON.stringify(v)); }
let verdicts = loadVerdicts();

const $ = (id) => document.getElementById(id);

function verdictEmoji(v) {
  if (v === 'ok')   return '✅';
  if (v === 'bad')  return '❌';
  if (v === 'warn') return '⚠️';
  return '·';
}
function verdictLabel(v) {
  if (v === 'ok')   return 'A1 right';
  if (v === 'bad')  return 'A1 wrong';
  if (v === 'warn') return 'ambiguous';
  return '—';
}

function renderCard(row) {
  const v = verdicts[row.idx] || null;
  const head = `
    <div class="card-head">
      <span class="num">${row.idx}</span>
      <span class="title">${row.video} / rally ${row.rally_short} / ${row.action.toUpperCase()}</span>
      <span class="pill alt">alt_ratio ${row.alt_ratio.toFixed(2)}×</span>
      <span class="pill">${row.summary_line}</span>
      <span class="grow"></span>
      <span class="verdict-status ${v || ''}" id="status-${row.idx}">${verdictEmoji(v)} ${verdictLabel(v)}</span>
    </div>`;

  const framesHtml = row.error
    ? `<div class="frame error">extraction failed: ${row.error}</div>`
    : `
      <div class="frame">
        <span class="frame-tag">PREV (context)</span>
        ${row.prev_frame_path
          ? `<img src="${row.prev_frame_path}" alt="prev frame" onclick="zoomImg(this.src)">`
          : '<div class="error">prev frame missing</div>'}
      </div>
      <div class="frame">
        <span class="frame-tag">CURR (A1 contact frame)</span>
        ${row.curr_frame_path
          ? `<img src="${row.curr_frame_path}" alt="curr frame" onclick="zoomImg(this.src)">`
          : '<div class="error">curr frame missing</div>'}
      </div>`;

  const sidebar = `
    <div class="sidebar">
      <div class="group">
        <dl>
          <dt>video</dt><dd>${row.video}</dd>
          <dt>rally</dt><dd>${row.rally_short}…</dd>
          <dt>action</dt><dd>${row.action}</dd>
          <dt>alt ratio</dt><dd>${row.alt_ratio.toFixed(2)}×</dd>
          <dt>A1 changed</dt><dd><span class="change-arrow">${row.a1_target.toUpperCase()}</span>: p${row.pid_before} → p${row.pid_after}</dd>
        </dl>
      </div>
      <div class="group">
        <dl>
          <dt>prev</dt><dd>${row.prev_action_desc}</dd>
          <dt>curr</dt><dd>${row.curr_action_desc}</dd>
        </dl>
      </div>
      <div class="group">
        <dl>
          <dt>prev cands</dt><dd>${row.prev_candidates}</dd>
          <dt>curr cands</dt><dd>${row.curr_candidates}</dd>
        </dl>
      </div>
      <div class="verdicts" data-idx="${row.idx}">
        <button class="ok ${v==='ok'?'sel':''}"   data-v="ok">
          <span class="icon">✅</span>A1 right</button>
        <button class="bad ${v==='bad'?'sel':''}" data-v="bad">
          <span class="icon">❌</span>A1 wrong</button>
        <button class="warn ${v==='warn'?'sel':''}" data-v="warn">
          <span class="icon">⚠️</span>ambiguous</button>
      </div>
    </div>`;

  return `<div class="card" id="card-${row.idx}">${head}
    <div class="card-body">
      <div class="frames">${framesHtml}</div>
      ${sidebar}
    </div></div>`;
}

function renderAll() {
  const container = $('container');
  container.innerHTML = ROWS.map(renderCard).join('');
  container.querySelectorAll('.verdicts').forEach(div => {
    const idx = +div.dataset.idx;
    div.querySelectorAll('button').forEach(btn => {
      btn.addEventListener('click', () => setVerdict(idx, btn.dataset.v));
    });
  });
  updateStatus();
}

function setVerdict(idx, v) {
  if (verdicts[idx] === v) {
    delete verdicts[idx];
  } else {
    verdicts[idx] = v;
  }
  saveVerdicts(verdicts);
  // Re-render the card row only
  const card = ROWS.find(r => r.idx === idx);
  const el = document.getElementById(`card-${idx}`);
  if (el && card) {
    const tmp = document.createElement('div');
    tmp.innerHTML = renderCard(card);
    el.replaceWith(tmp.firstElementChild);
    document.querySelector(`[data-idx="${idx}"]`).querySelectorAll('button').forEach(btn => {
      btn.addEventListener('click', () => setVerdict(idx, btn.dataset.v));
    });
  }
  updateStatus();
}

function updateStatus() {
  const total = ROWS.length;
  let ok=0, bad=0, warn=0;
  for (const idx in verdicts) {
    if (verdicts[idx] === 'ok') ok++;
    else if (verdicts[idx] === 'bad') bad++;
    else if (verdicts[idx] === 'warn') warn++;
  }
  const done = ok + bad + warn;
  $('status').textContent = `${done} / ${total} verdicts`;
  const pending = total - done;
  $('counts').textContent =
    `✅ ${ok}  ❌ ${bad}  ⚠️ ${warn}  (· ${pending} pending)`;
}

function compileVerdictString() {
  const parts = [];
  for (const r of ROWS) {
    const v = verdicts[r.idx];
    parts.push(`${r.idx}:${verdictEmoji(v)}`);
  }
  let line = parts.join(' ');
  // Plus a per-fix block for context
  let detail = ROWS.map(r => {
    const v = verdicts[r.idx];
    return `  ${r.idx}. ${r.video}/${r.rally_short} ${r.action} `
         + `(${r.summary_line}, alt ${r.alt_ratio.toFixed(2)}×): `
         + `${verdictEmoji(v)} ${verdictLabel(v)}`;
  }).join('\n');
  return line + '\n\n' + detail;
}

function showCopy() {
  $('copy-text').textContent = compileVerdictString();
  $('copy-modal').classList.add('open');
}

function hideCopy() { $('copy-modal').classList.remove('open'); }

async function copyClipboard() {
  try {
    await navigator.clipboard.writeText($('copy-text').textContent);
    $('copy-clipboard').textContent = 'Copied!';
    setTimeout(() => $('copy-clipboard').textContent = 'Copy to clipboard', 1200);
  } catch (e) {
    alert('Clipboard copy failed; select the text manually.');
  }
}

function zoomImg(src) {
  $('zoom-img').src = src;
  $('zoom-overlay').classList.add('open');
}

$('zoom-overlay').addEventListener('click', () => {
  $('zoom-overlay').classList.remove('open');
});
$('copy-btn').addEventListener('click', showCopy);
$('copy-btn-2').addEventListener('click', showCopy);
$('copy-close').addEventListener('click', hideCopy);
$('copy-clipboard').addEventListener('click', copyClipboard);
$('reset-btn').addEventListener('click', () => {
  if (confirm('Clear all verdicts?')) {
    verdicts = {}; saveVerdicts(verdicts); renderAll();
  }
});
$('reset-btn-2').addEventListener('click', () => {
  if (confirm('Clear all verdicts?')) {
    verdicts = {}; saveVerdicts(verdicts); renderAll();
  }
});

renderAll();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    resolver = VideoResolver()
    print(f"VideoResolver: endpoint={resolver.s3_endpoint} "
          f"bucket={resolver.bucket_name} cache={resolver.cache_dir}")

    cards: list[CardData] = []
    for fix in FIXES:
        try:
            card = _extract_card(fix, resolver)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            card = CardData(fix.idx, fix, {}, {}, None, None,
                            error=f"unexpected error: {e}")
        cards.append(card)

    print("\n--- Summary ---")
    n_curr = sum(1 for c in cards if c.curr_frame_path is not None)
    n_prev = sum(1 for c in cards if c.prev_frame_path is not None)
    n_err = sum(1 for c in cards if c.error)
    print(f"curr frames extracted: {n_curr}/10")
    print(f"prev frames extracted: {n_prev}/10")
    print(f"cards with error:      {n_err}/10")
    for c in cards:
        if c.error:
            print(f"  fix #{c.idx} ({c.fix.video_name}): {c.error}")

    html = _render_html(cards)
    HTML_PATH.write_text(html)
    print(f"\nWrote HTML: {HTML_PATH}")
    print(f"Open with: open {HTML_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
