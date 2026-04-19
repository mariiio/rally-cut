"""Audit ``action_ground_truth_json`` for a fixture and emit a review HTML.

Phase 0 of the crop-guided attribution research plan. The reviewer's only
job is **visual matching**: for each action moment, look at the full frame
(with the ball circled), then click the letter (A/B/C/D) of the reference
person who's the one performing the action.

Why this design:
* Reference crops are labeled with neutral letters (A/B/C/D) instead of
  canonical "Player 1-4" — the canonical IDs may have drifted since the
  reference crops were captured (see memory note on convention drift).
* Actor crops from the tracker are NOT shown by default — the tracker
  may be locked on the wrong player at the contact frame, which biases
  the reviewer.
* Filmstrip of full frames around the contact gives motion context so
  the reviewer can identify the actor by who jumps / swings / digs.

Usage:
    cd analysis
    uv run python scripts/audit_fixture_gt.py \\
        --video-id 0a383519-ecaa-411a-8e5e-e0aadc835725 \\
        --html-out ../reports/fixture_0a383519_gt_audit.html
"""

from __future__ import annotations

import argparse
import base64
import json
import string
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from rich.console import Console

from rallycut.evaluation.db import get_connection
from rallycut.evaluation.tracking.db import get_video_path

console = Console()

# Reference-crop display.
REF_HEIGHT = 140
MAX_REF_CROPS_PER_PERSON = 3

# Filmstrip of full frames around the contact for motion context.
FILMSTRIP_OFFSETS = [-12, -6, -2, 0, 4, 10]
FILMSTRIP_HEIGHT = 220
# Larger "main" frame at the contact instant.
MAIN_FRAME_HEIGHT = 540
# Ball marker drawn on the main frame (radius in pixels).
BALL_MARKER_RADIUS = 22
BALL_MARKER_COLOR = (0, 255, 255)  # yellow in BGR
BALL_MARKER_THICKNESS = 3


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class GtAction:
    rally_id: str
    rally_start_frame: int
    rally_index: int           # 1-based for display
    frame: int                 # rally-relative
    action: str
    gt_canonical_pid: int      # what the GT JSON says (may be wrong)
    pred_canonical_pid: int | None
    ball_x: float
    ball_y: float


@dataclass
class RallyContext:
    rally_id: str
    start_ms: int
    fps: float
    rally_start_frame: int
    positions_json: list[dict[str, Any]]
    actions_json: list[dict[str, Any]]
    track_to_player: dict[int, int]  # raw → canonical


# ---------------------------------------------------------------------------
# DB loaders
# ---------------------------------------------------------------------------


def load_match_analysis(video_id: str) -> dict[str, Any]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT match_analysis_json FROM videos WHERE id = %s",
                [video_id],
            )
            row = cur.fetchone()
    if not row or not row[0]:
        raise RuntimeError(
            f"Video {video_id} has no match_analysis_json. Run "
            "`rallycut match-players <video-id>` first."
        )
    return row[0]  # type: ignore[no-any-return]


def load_fixture_rallies(video_id: str) -> list[RallyContext]:
    """Load all GT-bearing rallies for the fixture, one ``RallyContext`` each."""
    query = """
        SELECT
            r.id, r.start_ms, pt.fps,
            pt.positions_json, pt.actions_json,
            pt.action_ground_truth_json
        FROM rallies r
        JOIN player_tracks pt ON pt.rally_id = r.id
        WHERE r.video_id = %s
          AND pt.action_ground_truth_json IS NOT NULL
        ORDER BY r.start_ms
    """
    contexts: list[RallyContext] = []
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, [video_id])
            rows = cur.fetchall()

    match_analysis = load_match_analysis(video_id)
    track_to_player_by_rally: dict[str, dict[int, int]] = {}
    for entry in match_analysis.get("rallies", []):
        rid = entry.get("rallyId") or entry.get("rally_id", "")
        ttp = entry.get("trackToPlayer") or entry.get("track_to_player", {})
        if rid and ttp:
            track_to_player_by_rally[rid] = {
                int(k): int(v) for k, v in ttp.items()
            }

    for row in rows:
        rid, start_ms, fps, positions_json, actions_json, _gt_json = row
        rid = str(rid)
        fps_f = float(fps or 60.0)
        rally_start_frame = int((start_ms or 0) / 1000.0 * fps_f)
        ttp = track_to_player_by_rally.get(rid, {})
        raw_actions = actions_json.get("actions", []) if isinstance(actions_json, dict) else []
        if isinstance(raw_actions, dict):
            raw_actions = raw_actions.get("actions", [])
        contexts.append(RallyContext(
            rally_id=rid,
            start_ms=int(start_ms or 0),
            fps=fps_f,
            rally_start_frame=rally_start_frame,
            positions_json=positions_json or [],
            actions_json=raw_actions or [],
            track_to_player=ttp,
        ))
    return contexts


def load_gt_actions(
    video_id: str,
    contexts_by_rally: dict[str, RallyContext],
    rally_index_by_id: dict[str, int],
) -> list[GtAction]:
    """Pair every GT action with the predicted canonical player at the same frame."""
    query = """
        SELECT r.id, pt.action_ground_truth_json
        FROM rallies r
        JOIN player_tracks pt ON pt.rally_id = r.id
        WHERE r.video_id = %s
          AND pt.action_ground_truth_json IS NOT NULL
        ORDER BY r.start_ms
    """
    gt_actions: list[GtAction] = []
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, [video_id])
            for rid, gt_json in cur.fetchall():
                rid = str(rid)
                ctx = contexts_by_rally.get(rid)
                if ctx is None:
                    continue
                pred_index: list[tuple[int, int]] = []
                for a in ctx.actions_json:
                    if not isinstance(a, dict):
                        continue
                    pred_index.append((
                        int(a.get("frame", 0)),
                        int(a.get("playerTrackId", -1)),
                    ))

                for label in gt_json:
                    if not isinstance(label, dict):
                        continue
                    gt_frame = int(label.get("frame", 0))
                    gt_canonical = int(label.get("playerTrackId", -1))
                    best_tid = None
                    best_dist = 4
                    for pred_frame, pred_tid in pred_index:
                        dist = abs(pred_frame - gt_frame)
                        if dist < best_dist:
                            best_tid = pred_tid
                            best_dist = dist
                    pred_canonical = (
                        ctx.track_to_player.get(best_tid)
                        if best_tid is not None and best_tid >= 0
                        else None
                    )
                    gt_actions.append(GtAction(
                        rally_id=rid,
                        rally_start_frame=ctx.rally_start_frame,
                        rally_index=rally_index_by_id[rid],
                        frame=gt_frame,
                        action=str(label.get("action", "")),
                        gt_canonical_pid=gt_canonical,
                        pred_canonical_pid=pred_canonical,
                        ball_x=float(label.get("ballX", 0.0)),
                        ball_y=float(label.get("ballY", 0.0)),
                    ))
    return gt_actions


def load_reference_crops(
    video_id: str,
    video_path: Path,
    max_per_player: int,
) -> dict[int, list[np.ndarray]]:
    """Extract user-labeled reference crops keyed by canonical pid."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT player_id, frame_ms, bbox_x, bbox_y, bbox_w, bbox_h
                   FROM player_reference_crops
                   WHERE video_id = %s
                   ORDER BY player_id, created_at""",
                [video_id],
            )
            rows = cur.fetchall()
    if not rows:
        return {}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {}
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    crops_by_player: dict[int, list[np.ndarray]] = {}
    sorted_rows = sorted(rows, key=lambda r: float(r[1]))  # type: ignore[call-overload]
    for pid, frame_ms, bx, by, bw, bh in sorted_rows:
        pid = int(pid)
        if len(crops_by_player.get(pid, [])) >= max_per_player:
            continue
        cap.set(cv2.CAP_PROP_POS_MSEC, float(frame_ms))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        crop = _crop_bbox(np.asarray(frame, dtype=np.uint8),
                          float(bx), float(by), float(bw), float(bh), fw, fh)
        if crop is not None:
            crops_by_player.setdefault(pid, []).append(crop)
    cap.release()
    return crops_by_player


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------


def _crop_bbox(
    frame: np.ndarray,
    bx: float, by: float, bw: float, bh: float,
    fw: int, fh: int,
    pad: float = 0.20,
) -> np.ndarray | None:
    pad_w = bw * pad
    pad_h = bh * pad
    x1 = max(0, int((bx - bw / 2 - pad_w) * fw))
    y1 = max(0, int((by - bh / 2 - pad_h) * fh))
    x2 = min(fw, int((bx + bw / 2 + pad_w) * fw))
    y2 = min(fh, int((by + bh / 2 + pad_h) * fh))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] < 16 or crop.shape[1] < 8:
        return None
    return crop  # type: ignore[no-any-return]


def extract_filmstrip(
    cap: cv2.VideoCapture,
    ctx: RallyContext,
    rel_frame: int,
    fw: int, fh: int,
    ball_x: float, ball_y: float,
    target_height: int,
    mark_ball: bool,
) -> list[np.ndarray]:
    """Extract a filmstrip of full frames around the contact instant.

    The middle frame (offset 0) optionally gets a yellow ball-position
    marker so the reviewer can spot the contact.
    """
    out: list[np.ndarray] = []
    for offset in FILMSTRIP_OFFSETS:
        abs_frame = ctx.rally_start_frame + rel_frame + offset
        if abs_frame < 0:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, abs_frame)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        frame = np.asarray(frame, dtype=np.uint8).copy()
        if mark_ball and offset == 0:
            cx = int(round(ball_x * fw))
            cy = int(round(ball_y * fh))
            cv2.circle(
                frame, (cx, cy),
                BALL_MARKER_RADIUS, BALL_MARKER_COLOR, BALL_MARKER_THICKNESS,
            )
        h, w = frame.shape[:2]
        if h != target_height:
            scale = target_height / h
            frame = cv2.resize(frame, (max(1, int(round(w * scale))), target_height))
        out.append(frame)
    return out


def extract_main_frame(
    cap: cv2.VideoCapture,
    ctx: RallyContext,
    rel_frame: int,
    fw: int, fh: int,
    ball_x: float, ball_y: float,
) -> np.ndarray | None:
    abs_frame = ctx.rally_start_frame + rel_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, abs_frame)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    frame = np.asarray(frame, dtype=np.uint8).copy()
    cx = int(round(ball_x * fw))
    cy = int(round(ball_y * fh))
    cv2.circle(
        frame, (cx, cy),
        BALL_MARKER_RADIUS, BALL_MARKER_COLOR, BALL_MARKER_THICKNESS,
    )
    h, w = frame.shape[:2]
    if h != MAIN_FRAME_HEIGHT:
        scale = MAIN_FRAME_HEIGHT / h
        frame = cv2.resize(frame, (max(1, int(round(w * scale))), MAIN_FRAME_HEIGHT))
    return frame


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------


def to_data_uri(crop: np.ndarray | None, target_height: int | None = None) -> str:
    if crop is None or crop.size == 0:
        return ""
    if target_height is not None and crop.shape[0] != target_height:
        h, w = crop.shape[:2]
        scale = target_height / h
        crop = cv2.resize(crop, (max(1, int(round(w * scale))), target_height))
    ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def render_html(
    video_id: str,
    rallies: list[RallyContext],
    actions: list[GtAction],
    main_frames: list[np.ndarray | None],
    ref_crops: dict[int, list[np.ndarray]],
) -> str:
    """Build the standalone HTML."""
    # Map canonical pid → letter (A, B, C, D). Sorted pids → letters in order.
    sorted_pids = sorted(ref_crops.keys())
    letter_for_pid = {pid: string.ascii_uppercase[i] for i, pid in enumerate(sorted_pids)}
    pid_for_letter = {v: k for k, v in letter_for_pid.items()}
    letter_mapping_json = json.dumps(
        {letter: pid for pid, letter in letter_for_pid.items()},
    )

    # Reference strips at top.
    ref_strip_html = ""
    for pid in sorted_pids:
        letter = letter_for_pid[pid]
        imgs = "".join(
            f'<img src="{to_data_uri(c, REF_HEIGHT)}" alt="Person {letter}">'
            for c in ref_crops[pid]
        )
        ref_strip_html += (
            f'<div class="ref-strip">'
            f'<div class="ref-letter">Person {letter}</div>'
            f'<div class="ref-imgs">{imgs}</div></div>'
        )

    # Pre-encode one thumbnail per person (first reference crop) for the
    # picker tiles below each row.
    person_thumb_uri: dict[str, str] = {}
    for pid, letter in letter_for_pid.items():
        crops = ref_crops.get(pid, [])
        if crops:
            person_thumb_uri[letter] = to_data_uri(crops[0], REF_HEIGHT)

    # Rows.
    rows_html = ""
    person_tiles_template = []
    for letter in sorted(letter_for_pid.values()):
        thumb = person_thumb_uri.get(letter, "")
        img = (
            f'<img src="{thumb}" alt="Person {letter}">'
            if thumb else '<div class="tile-missing">?</div>'
        )
        person_tiles_template.append((letter, f'Person {letter}', img))
    extra_tiles = [
        ("can't tell", "Can't tell", '<div class="tile-icon">?</div>'),
        ("no actor visible", "No clear actor", '<div class="tile-icon">∅</div>'),
    ]

    for idx, a in enumerate(actions):
        main_uri = to_data_uri(main_frames[idx])
        tiles = "".join(
            f'<label class="tile"><input type="radio" name="row-{idx}" value="{val}">'
            f'<div class="tile-img">{img}</div>'
            f'<div class="tile-label">{lbl}</div></label>'
            for val, lbl, img in (person_tiles_template + extra_tiles)
        )
        debug_line = (
            f'<span class="debug">DB says GT canonical = P{a.gt_canonical_pid} '
            f'(Person {letter_for_pid.get(a.gt_canonical_pid, "?")})'
            + (f' &middot; pipeline predicted P{a.pred_canonical_pid} '
               f'(Person {letter_for_pid.get(a.pred_canonical_pid, "?")})'
               if a.pred_canonical_pid is not None else '')
            + '</span>'
        )
        rows_html += f"""
<div class="row" data-idx="{idx}"
     data-rally-id="{a.rally_id}"
     data-rally-index="{a.rally_index}"
     data-frame="{a.frame}"
     data-action="{a.action}"
     data-gt-pid="{a.gt_canonical_pid}"
     data-pred-pid="{a.pred_canonical_pid if a.pred_canonical_pid is not None else ''}">
  <div class="row-head">
    <span class="row-num">#{idx + 1} / {len(actions)}</span>
    <span class="row-context">Rally {a.rally_index}/{len(rallies)} &middot;
      <b>{a.action}</b> &middot; frame {a.frame}</span>
    {debug_line}
  </div>
  <div class="row-main">
    {f'<img src="{main_uri}" class="main-frame">' if main_uri else '<div class="missing">no frame</div>'}
  </div>
  <div class="row-radios">
    <div class="radio-prompt">Pick the player who took this action:</div>
    <div class="tile-grid">{tiles}</div>
  </div>
</div>
"""

    js = """
<script>
const LETTER_TO_PID = __LETTER_MAPPING__;
function collect() {
  const rows = document.querySelectorAll('.row');
  const out = {
    videoId: '__VIDEO_ID__',
    generatedAt: new Date().toISOString(),
    letterToCanonicalPid: LETTER_TO_PID,
    actions: []
  };
  rows.forEach(r => {
    const idx = parseInt(r.dataset.idx, 10);
    const radio = r.querySelector(`input[name="row-${idx}"]:checked`);
    const value = radio ? radio.value : null;
    const trustedLetter = value && value.length === 1 ? value : null;
    const trustedCanonicalPid = trustedLetter ? LETTER_TO_PID[trustedLetter] : null;
    out.actions.push({
      rallyId: r.dataset.rallyId,
      rallyIndex: parseInt(r.dataset.rallyIndex, 10),
      frame: parseInt(r.dataset.frame, 10),
      action: r.dataset.action,
      originalCanonicalPid: parseInt(r.dataset.gtPid, 10),
      predictedCanonicalPid: r.dataset.predPid ? parseInt(r.dataset.predPid, 10) : null,
      reviewedAs: value,                  // 'A'|'B'|'C'|'D'|"can't tell"|"no actor visible"
      trustedCanonicalPid: trustedCanonicalPid,
      wasRelabeled: (trustedCanonicalPid !== null) &&
                    (trustedCanonicalPid !== parseInt(r.dataset.gtPid, 10))
    });
  });
  return out;
}
function updateProgress() {
  const rows = document.querySelectorAll('.row');
  const done = Array.from(rows).filter(r =>
    r.querySelector(`input[name="row-${r.dataset.idx}"]:checked`)
  ).length;
  document.getElementById('progress').textContent = `${done} / ${rows.length} reviewed`;
}
document.getElementById('save').addEventListener('click', () => {
  const data = collect();
  const total = data.actions.length;
  const labelled = data.actions.filter(a => a.reviewedAs).length;
  if (labelled < total) {
    if (!confirm(`Only ${labelled}/${total} reviewed. Save anyway?`)) return;
  }
  const blob = new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'fixture_trusted_gt.json';
  a.click();
});
document.addEventListener('change', e => {
  if (e.target.tagName === 'INPUT') updateProgress();
});
updateProgress();
</script>
""".replace("__VIDEO_ID__", video_id).replace("__LETTER_MAPPING__", letter_mapping_json)

    style = """
<style>
body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 0;
       padding: 0 20px 40px; background: #f5f5f7; color: #222; }
header { position: sticky; top: 0; background: #f5f5f7; padding: 14px 0;
         z-index: 20; border-bottom: 1px solid #d4d4d4; }
header h1 { margin: 0 0 8px; font-size: 17px; }
header p { margin: 0 0 8px; font-size: 13px; color: #555; max-width: 900px; }
.controls { display: flex; gap: 12px; align-items: center; margin-bottom: 8px; }
button { padding: 7px 14px; border: 1px solid #888; background: white;
         border-radius: 5px; cursor: pointer; font-size: 13px; font-weight: 500; }
button:hover { background: #eee; }
#progress { font-size: 13px; color: #555; }
.ref-strips { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;
              padding: 10px 0; background: #f5f5f7; }
.ref-strip { border: 2px solid #888; border-radius: 8px; padding: 8px;
             background: white; }
.ref-letter { font-size: 14px; font-weight: bold; margin-bottom: 6px;
              color: #333; }
.ref-imgs { display: flex; gap: 6px; align-items: flex-end; overflow-x: auto; }
.ref-imgs img { height: 140px; border-radius: 4px; border: 1px solid #ddd; }
.row { background: white; border: 1px solid #d4d4d4; border-radius: 8px;
       padding: 14px 16px; margin: 14px 0; }
.row-head { display: flex; gap: 14px; align-items: baseline; flex-wrap: wrap;
            margin-bottom: 10px; font-size: 14px; }
.row-num { font-weight: bold; color: #444; }
.row-context { color: #333; }
.debug { font-size: 11px; color: #888; margin-left: auto; font-family: monospace; }
.row-main { text-align: center; margin-bottom: 10px; }
.main-frame { max-width: 100%; max-height: 540px; border: 2px solid #aaa;
              border-radius: 4px; }
.missing { display: inline-block; width: 320px; height: 200px; background: #eee;
           color: #999; line-height: 200px; border-radius: 4px;
           border: 2px dashed #ccc; font-size: 13px; }
.row-radios { margin-top: 8px; }
.radio-prompt { font-size: 13px; font-weight: bold; color: #333; margin-bottom: 8px; }
.tile-grid { display: flex; gap: 10px; flex-wrap: wrap; }
.tile { display: flex; flex-direction: column; align-items: center;
        cursor: pointer; padding: 6px; border: 2px solid #bbb; border-radius: 8px;
        background: #fafafa; min-width: 110px; transition: all 0.1s; }
.tile:hover { background: #eef; border-color: #88a; }
.tile:has(input:checked) { background: #d8f0d8; border-color: #4a8;
                           box-shadow: 0 0 0 2px #4a8 inset; }
.tile input { display: none; }
.tile-img { display: flex; align-items: center; justify-content: center;
            height: 140px; }
.tile-img img { height: 140px; border-radius: 4px; border: 1px solid #ddd; }
.tile-icon, .tile-missing { width: 90px; height: 140px; display: flex;
                            align-items: center; justify-content: center;
                            background: #eee; color: #888; border-radius: 4px;
                            border: 1px dashed #ccc; font-size: 36px; }
.tile-label { font-size: 13px; font-weight: 500; color: #333; margin-top: 6px; }
</style>
"""

    instructions = (
        '<p><b>Your task:</b> for each row, look at the yellow ball marker in '
        'the main frame and click the player tile (with their reference photo) '
        'who took the action. Use "Can\'t tell" if ambiguous, or "No clear '
        'actor" if no one is near the ball.</p>'
    )

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>GT Audit — {video_id[:8]}</title>
{style}
</head>
<body>
<header>
  <h1>Action GT Audit — {video_id}</h1>
  {instructions}
  <div class="controls">
    <button id="save">Save JSON (download)</button>
    <span id="progress">0 / 0 reviewed</span>
  </div>
</header>
<section class="ref-strips">{ref_strip_html}</section>
<section>{rows_html}</section>
{js}
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--html-out", required=True, type=Path)
    parser.add_argument("--max-ref-crops", type=int, default=MAX_REF_CROPS_PER_PERSON)
    args = parser.parse_args(argv)

    video_id: str = args.video_id
    console.print(f"[bold]Auditing GT for video[/bold] {video_id}")

    video_path = get_video_path(video_id)
    if video_path is None:
        console.print(f"[red]Cannot resolve video path for {video_id}[/red]")
        return 1
    console.print(f"  video file: {video_path}")

    rallies = load_fixture_rallies(video_id)
    rally_index_by_id = {r.rally_id: i + 1 for i, r in enumerate(rallies)}
    contexts_by_rally = {r.rally_id: r for r in rallies}
    console.print(f"  {len(rallies)} GT-bearing rallies")

    actions = load_gt_actions(video_id, contexts_by_rally, rally_index_by_id)
    console.print(f"  {len(actions)} GT actions to audit")

    ref_crops = load_reference_crops(video_id, video_path, args.max_ref_crops)
    n_refs = sum(len(c) for c in ref_crops.values())
    console.print(f"  {n_refs} reference crops loaded for {len(ref_crops)} players")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        console.print("[red]Failed to open video[/red]")
        return 1
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    main_frames: list[np.ndarray | None] = []
    for i, action in enumerate(actions):
        ctx = contexts_by_rally[action.rally_id]
        main_frames.append(extract_main_frame(
            cap, ctx, action.frame, fw, fh, action.ball_x, action.ball_y,
        ))
        if (i + 1) % 10 == 0 or i == len(actions) - 1:
            console.print(f"  extracted [{i + 1}/{len(actions)}]")
    cap.release()

    html = render_html(video_id, rallies, actions, main_frames, ref_crops)
    args.html_out.parent.mkdir(parents=True, exist_ok=True)
    args.html_out.write_text(html)
    size_mb = args.html_out.stat().st_size / (1024 * 1024)
    console.print(f"  [green]wrote[/green] {args.html_out} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
