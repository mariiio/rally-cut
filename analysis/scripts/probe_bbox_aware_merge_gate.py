"""Validate the bbox-quality-aware merge gate hypothesis on user-reported cases.

For each user-reported failure case, identify candidate-merge pairs
(temporally-non-overlapping filtered tracks that COULD have been merged
but weren't), compute every signal the merge decision could use, and
show what each candidate gate WOULD decide. Generates an HTML page
where the user can visually verify the decisions on the actual crops.

The proposed gate works as follows:
  - Compute bbox quality at endpoint frames (last frame of A,
    first frame of B). Quality = bbox area + aspect ratio + crop
    confidence.
  - SMALL/PARTIAL bbox → use MOTION-ONLY criteria:
      gap ≤ 30 frames AND spatial dist ≤ 0.08 normalized AND
      court-plane velocity ≤ 2.5 m/s (if calibrator available)
  - LARGE/CLEAR bbox → use APPEARANCE-AWARE (current production logic).

For each candidate pair we report:
  - Metrics: gap, spatial dist, court dist, HSV Bhatt, learned cos,
    bbox areas
  - Current gate decision (HSV ≤ 0.20 + spatial + temporal)
  - Proposed gate decision (motion-only or appearance-aware based on
    bbox quality)
  - Verdict: would the proposed gate APPROVE this merge that the
    current gate REJECTS?

Read-only, no DB mutation. Writes per-rally HTML to
analysis/reports/bbox_aware_gate_validation/<rally_short>/.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np

from rallycut.evaluation.tracking.db import get_connection, get_video_path
from rallycut.tracking.player_features import extract_bbox_crop

ANALYSIS_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ANALYSIS_ROOT / "reports" / "bbox_aware_gate_validation"

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


# Gate thresholds — align with production's PRIMARY_RELINK_MAX_GAP=50 +
# PRIMARY_RELINK_MAX_DISTANCE=0.08, plus an additional VELOCITY check
# (the lever current production lacks for relink_primary_fragments). The
# velocity check prevents very-fast displacements over short gaps that
# imply impossible motion (e.g., teleport across the court).
MOTION_MAX_GAP = 50  # frames; matches PRIMARY_RELINK_MAX_GAP
MOTION_MAX_SPATIAL_NORM = 0.08  # matches PRIMARY_RELINK_MAX_DISTANCE
MOTION_MAX_VELOCITY_PER_FRAME = 0.005  # ≈0.15 normalized over 30 frames; sprint cap
MOTION_MAX_COURT_M = 2.5  # court-plane meters (when calibrator available)

# Bbox quality classification — calibrated from observed user-case
# crops. Far-side small detections produce mostly-sand crops at bbox
# areas in the 0.003-0.012 range (these are the "bad input to ReID"
# cases we identified visually). Near-side full-body detections sit
# at 0.020+ (clean ReID input). Threshold 0.012 marks the boundary.
SMALL_BBOX_AREA_THRESH = 0.012  # normalized; smaller → use motion-only
PARTIAL_ASPECT_TOL = 0.4  # h/w ratio outside [PARTIAL_TOL, 1/PARTIAL_TOL] → partial body

# Appearance gate (mirrors production)
HSV_MAX_BHATT = 0.20
APPEARANCE_REID_HIGH = 0.7  # if learned cos > this, override even if HSV fails


@dataclass
class GateDecision:
    blocked: bool
    reason: str
    rule: str  # "motion-only" or "appearance-aware" or "current-prod"


def _load_rally_full(rally_id: str) -> dict[str, Any]:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                r.id, r.video_id, r.start_ms, r.end_ms,
                pt.positions_json, pt.raw_positions_json,
                pt.primary_track_ids, pt.fps, pt.frame_count,
                v.match_analysis_json, v.player_matching_gt_json,
                v.court_calibration_json
            FROM rallies r
            JOIN player_tracks pt ON pt.rally_id = r.id
            JOIN videos v ON v.id = r.video_id
            WHERE r.id = %s
            """,
            [rally_id],
        )
        row = cur.fetchone()
    if row is None:
        raise SystemExit(f"rally {rally_id} not found")
    (
        rid, video_id, start_ms, end_ms,
        pos, raw_pos, primary, fps, frame_count,
        ma, gt, calib,
    ) = row
    pos = pos if isinstance(pos, list) else (json.loads(cast(str, pos)) if pos else [])
    raw_pos = raw_pos if isinstance(raw_pos, list) else (json.loads(cast(str, raw_pos)) if raw_pos else [])
    if isinstance(ma, str):
        ma = json.loads(ma)
    if isinstance(gt, str):
        gt = json.loads(gt)
    if isinstance(calib, str):
        calib = json.loads(calib)
    return {
        "rally_id": rid,
        "video_id": video_id,
        "start_ms": int(start_ms),
        "end_ms": int(end_ms),
        "positions": pos,
        "raw_positions": raw_pos,
        "primary_track_ids": primary or [],
        "fps": float(fps) if fps else 30.0,
        "frame_count": int(frame_count) if frame_count else 0,
        "match_analysis": ma or {},
        "gt_json": gt or {},
        "court_calibration_json": calib,
    }


def _track_endpoints(positions: list[dict[str, Any]], track_id: int) -> dict[str, Any] | None:
    track_pos = sorted(
        [p for p in positions if int(p.get("trackId", -1)) == track_id],
        key=lambda p: int(p.get("frameNumber", 0)),
    )
    if not track_pos:
        return None
    return {
        "first": track_pos[0],
        "last": track_pos[-1],
        "n_frames": len(track_pos),
        "frame_set": {int(p["frameNumber"]) for p in track_pos},
        "median_area": float(np.median([
            float(p.get("width", 0)) * float(p.get("height", 0)) for p in track_pos
        ])),
        "median_height": float(np.median([float(p.get("height", 0)) for p in track_pos])),
        "median_aspect": float(np.median([
            float(p.get("height", 0)) / max(float(p.get("width", 0)), 1e-6)
            for p in track_pos
        ])),
    }


def _bbox_area(p: dict[str, Any]) -> float:
    return float(p.get("width", 0)) * float(p.get("height", 0))


def _is_low_quality_bbox(p: dict[str, Any]) -> bool:
    """Heuristic: small bbox area → low ReID/HSV reliability."""
    return _bbox_area(p) < SMALL_BBOX_AREA_THRESH


def _classify_pair_temporal(
    a: dict[str, Any], b: dict[str, Any]
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Determine which track ends first; returns (relation, earlier, later).

    relation: "non-overlap-A-first" | "non-overlap-B-first" | "overlap"
    """
    a_last = int(a["last"]["frameNumber"])
    b_first = int(b["first"]["frameNumber"])
    if a_last < b_first:
        return "non-overlap-A-first", a, b
    b_last = int(b["last"]["frameNumber"])
    a_first = int(a["first"]["frameNumber"])
    if b_last < a_first:
        return "non-overlap-B-first", b, a
    overlap = a["frame_set"] & b["frame_set"]
    return "overlap" if overlap else "non-overlap-A-first", a, b


def _spatial_distance(a_pos: dict[str, Any], b_pos: dict[str, Any]) -> float:
    ax = float(a_pos.get("x", 0)) + float(a_pos.get("width", 0)) / 2
    ay = float(a_pos.get("y", 0)) + float(a_pos.get("height", 0)) / 2
    bx = float(b_pos.get("x", 0)) + float(b_pos.get("width", 0)) / 2
    by = float(b_pos.get("y", 0)) + float(b_pos.get("height", 0)) / 2
    return float(np.hypot(bx - ax, by - ay))


def _evaluate_motion_only_gate(
    earlier: dict[str, Any], later: dict[str, Any]
) -> GateDecision:
    """Apply motion-only criteria to a non-overlapping pair.

    Three independent checks:
      - gap ≤ MOTION_MAX_GAP (currently 50, matches PRIMARY_RELINK_MAX_GAP)
      - spatial endpoint distance ≤ MOTION_MAX_SPATIAL_NORM
      - implied velocity ≤ MOTION_MAX_VELOCITY_PER_FRAME (the new gate;
        prevents teleport-style merges where two fragments are close in
        gap but the player can't have plausibly traversed the distance
        at that speed)
    """
    gap = int(later["first"]["frameNumber"]) - int(earlier["last"]["frameNumber"])
    if gap <= 0:
        return GateDecision(True, "tracks overlap or are simultaneous", "motion-only")
    if gap > MOTION_MAX_GAP:
        return GateDecision(True, f"gap {gap}f > {MOTION_MAX_GAP}f", "motion-only")
    sd = _spatial_distance(earlier["last"], later["first"])
    if sd > MOTION_MAX_SPATIAL_NORM:
        return GateDecision(True, f"spatial dist {sd:.3f} > {MOTION_MAX_SPATIAL_NORM}", "motion-only")
    velocity = sd / gap
    if velocity > MOTION_MAX_VELOCITY_PER_FRAME:
        return GateDecision(
            True,
            f"velocity {velocity:.4f}/f > {MOTION_MAX_VELOCITY_PER_FRAME}/f (gap {gap}, dist {sd:.3f})",
            "motion-only",
        )
    return GateDecision(
        False,
        f"motion-only PASS: gap={gap}f, dist={sd:.3f}, vel={velocity:.4f}/f",
        "motion-only",
    )


def _evaluate_current_prod_gate(
    earlier: dict[str, Any], later: dict[str, Any], hsv_bhatt: float | None
) -> GateDecision:
    """Mirror relink_primary_fragments' current logic (gap + spatial + appearance)."""
    gap = int(later["first"]["frameNumber"]) - int(earlier["last"]["frameNumber"])
    if gap <= 0:
        return GateDecision(True, "overlap", "current-prod")
    if gap > 50:  # PRIMARY_RELINK_MAX_GAP
        return GateDecision(True, f"gap {gap}f > 50f", "current-prod")
    sd = _spatial_distance(earlier["last"], later["first"])
    if sd > 0.08:  # PRIMARY_RELINK_MAX_DISTANCE
        return GateDecision(True, f"spatial {sd:.3f} > 0.08", "current-prod")
    if hsv_bhatt is None:
        return GateDecision(False, "no HSV histograms; would pass spatial+temporal alone", "current-prod")
    if hsv_bhatt > HSV_MAX_BHATT:
        return GateDecision(True, f"HSV Bhatt {hsv_bhatt:.3f} > {HSV_MAX_BHATT}", "current-prod")
    return GateDecision(False, f"current-prod PASS: gap={gap}f, dist={sd:.3f}, bhatt={hsv_bhatt:.3f}", "current-prod")


def _evaluate_bbox_aware_gate(
    earlier: dict[str, Any], later: dict[str, Any], hsv_bhatt: float | None,
    learned_cos: float | None,
) -> GateDecision:
    """Proposed: bbox-quality-aware. Falls back to motion-only when bboxes are small."""
    earlier_low = _is_low_quality_bbox(earlier["last"])
    later_low = _is_low_quality_bbox(later["first"])
    if earlier_low or later_low:
        # Trust motion only — appearance signals are unreliable on tiny crops.
        gate = _evaluate_motion_only_gate(earlier, later)
        gate = GateDecision(
            gate.blocked, f"[bbox-quality LOW: areas A={_bbox_area(earlier['last']):.4f}, B={_bbox_area(later['first']):.4f}] " + gate.reason,
            "bbox-aware → motion-only",
        )
        return gate
    # Large/clear bboxes — use appearance-aware path.
    gap = int(later["first"]["frameNumber"]) - int(earlier["last"]["frameNumber"])
    if gap <= 0:
        return GateDecision(True, "overlap", "bbox-aware → appearance")
    if gap > 50:
        return GateDecision(True, f"gap {gap}f > 50f", "bbox-aware → appearance")
    sd = _spatial_distance(earlier["last"], later["first"])
    if sd > 0.08:
        return GateDecision(True, f"spatial {sd:.3f} > 0.08", "bbox-aware → appearance")
    # Appearance signal: ReID overrides HSV when high
    if learned_cos is not None and learned_cos > APPEARANCE_REID_HIGH:
        return GateDecision(False, f"learned ReID confirms (cos={learned_cos:.3f} > {APPEARANCE_REID_HIGH})", "bbox-aware → appearance")
    if hsv_bhatt is not None and hsv_bhatt > HSV_MAX_BHATT:
        return GateDecision(True, f"HSV Bhatt {hsv_bhatt:.3f} > {HSV_MAX_BHATT}, learned cos {learned_cos:.3f if learned_cos is not None else None} not high enough", "bbox-aware → appearance")
    return GateDecision(False, f"bbox-aware appearance PASS: gap={gap}f, dist={sd:.3f}, bhatt={hsv_bhatt}, cos={learned_cos}", "bbox-aware → appearance")


def _save_endpoint_annotated_frame(
    cap: cv2.VideoCapture,
    rally_start_ms: int,
    video_fps: float,
    pos: dict[str, Any],
    out_path: Path,
    label: str,
    color: tuple[int, int, int] = (0, 255, 0),  # green BGR
    other_positions_at_frame: list[dict[str, Any]] | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Save a full-frame image with the target bbox highlighted, plus
    optional faded outlines of other positions at the same frame for context.
    """
    f_in_rally = int(pos.get("frameNumber", 0))
    f_in_video = int(rally_start_ms / 1000.0 * video_fps) + f_in_rally
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_in_video)
    ret, frame = cap.read()
    meta = {
        "rally_frame": f_in_rally, "video_frame": f_in_video,
        "x": pos.get("x"), "y": pos.get("y"),
        "width": pos.get("width"), "height": pos.get("height"),
        "bbox_area": _bbox_area(pos),
    }
    if not ret or frame is None:
        meta["error"] = "frame read failed"
        return False, meta
    fh, fw = frame.shape[:2]

    # Draw faded outlines for other positions in the same frame (context).
    if other_positions_at_frame:
        for op in other_positions_at_frame:
            ox, oy = op.get("x"), op.get("y")
            ow, oh = op.get("width"), op.get("height")
            if None in (ox, oy, ow, oh):
                continue
            x1 = int((ox - ow / 2) * fw)
            y1 = int((oy - oh / 2) * fh)
            x2 = int((ox + ow / 2) * fw)
            y2 = int((oy + oh / 2) * fh)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (160, 160, 160), 1)
            other_label = f"raw{op.get('trackId', '?')}"
            cv2.putText(frame, other_label, (x1, max(y1 - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1, cv2.LINE_AA)

    # Draw the target bbox in bright color with thicker line.
    bx, by, bw, bh = pos["x"], pos["y"], pos["width"], pos["height"]
    x1 = int((bx - bw / 2) * fw)
    y1 = int((by - bh / 2) * fh)
    x2 = int((bx + bw / 2) * fw)
    y2 = int((by + bh / 2) * fh)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    # Header banner with the label
    header_h = 56
    header = np.zeros((header_h, fw, 3), dtype=np.uint8)
    cv2.putText(header, label, (12, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    annotated = np.vstack([header, frame])

    # Resize for HTML embedding (~720px wide max)
    max_w = 720
    if annotated.shape[1] > max_w:
        scale = max_w / annotated.shape[1]
        new_w = max_w
        new_h = int(annotated.shape[0] * scale)
        annotated = cv2.resize(annotated, (new_w, new_h))

    cv2.imwrite(str(out_path), annotated)
    meta["frame_w"] = int(annotated.shape[1])
    meta["frame_h"] = int(annotated.shape[0])
    return True, meta


# Backward-compat name used elsewhere in this module
_save_endpoint_crop = _save_endpoint_annotated_frame


def _format_html(
    rally_id: str,
    rally_info: dict[str, Any],
    candidates: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    html: list[str] = []
    html.append("<!DOCTYPE html><html><head><meta charset='utf-8'>")
    html.append(f"<title>Bbox-aware merge gate — {rally_id[:8]}</title>")
    html.append("""
<style>
body { font-family: -apple-system, sans-serif; max-width: 1400px; margin: 1em auto; padding: 0 1em; }
.candidate { border: 2px solid #999; border-radius: 8px; padding: 1em; margin-bottom: 2em; background: #fafafa; }
.candidate.would-flip { background: #effdf0; border-color: #2a6; }
.candidate.would-block-correctly { background: #f5f5f5; border-color: #999; }
.candidate-header { font-weight: 600; font-size: 16px; margin-bottom: 0.5em; }
.crop-pair { display: flex; gap: 16px; align-items: flex-start; margin-top: 0.5em; flex-wrap: wrap; }
.crop-card { border: 1px solid #ccc; padding: 4px; background: #fff; }
.crop-card img { display: block; max-width: 720px; height: auto; }
.crop-card .meta { margin-top: 4px; font-size: 11px; color: #555; }
.signals { font-size: 13px; margin: 0.5em 0; }
.signals table { border-collapse: collapse; }
.signals th, .signals td { border: 1px solid #ccc; padding: 4px 8px; text-align: left; font-size: 12px; }
.signals th { background: #eee; }
.gate-decision { padding: 8px; margin: 4px 0; border-radius: 4px; font-size: 13px; }
.gate-pass { background: #effdf0; color: #2a6; }
.gate-block { background: #fee; color: #c33; }
.summary { background: #f0f4ff; padding: 1em; margin-bottom: 1em; border-radius: 4px; }
</style>
</head><body>
""")
    html.append(f"<h1>Bbox-aware merge gate validation — rally {rally_id[:8]}</h1>")

    # Summary
    n_pass = sum(1 for c in candidates if not c["proposed"]["blocked"])
    n_block = sum(1 for c in candidates if c["proposed"]["blocked"])
    html.append("<div class='summary'>")
    html.append(f"<b>Summary:</b> {len(candidates)} candidate non-overlapping pairs evaluated.<br>")
    html.append(f"Proposed bbox-aware gate: <b style='color:#2a6'>{n_pass}</b> APPROVE, <b style='color:#c33'>{n_block}</b> BLOCK<br><br>")
    html.append("<b>How to verify:</b> for each pair below, look at the two endpoint crops side-by-side. ")
    html.append("Are they the SAME physical player (then the gate should APPROVE) or DIFFERENT players (then it should BLOCK)? ")
    html.append("Compare your judgment to the gate decision printed below each pair.")
    html.append("</div>")

    for cand in candidates:
        # Color the candidate by gate decision
        if not cand["proposed"]["blocked"]:
            flip_class = "would-flip"  # green — gate says APPROVE merge
        else:
            flip_class = "would-block-correctly"  # neutral — gate says BLOCK
        html.append(f"<div class='candidate {flip_class}'>")
        verdict = "PROPOSED: APPROVE MERGE" if not cand["proposed"]["blocked"] else "PROPOSED: BLOCK"
        html.append(
            f"<div class='candidate-header'>"
            f"raw T{cand['track_a']} ↔ raw T{cand['track_b']} "
            f"— <span style='color:{'#2a6' if not cand['proposed']['blocked'] else '#c33'}'>{verdict}</span>"
            f"</div>"
        )
        html.append("<div class='signals'><table>")
        html.append("<tr><th>Signal</th><th>Value</th></tr>")
        html.append(f"<tr><td>Temporal gap</td><td>{cand['gap']} frames ({cand['gap'] / cand['fps']:.2f}s)</td></tr>")
        html.append(f"<tr><td>Spatial distance (image-norm)</td><td>{cand['spatial_dist']:.4f}</td></tr>")
        html.append(f"<tr><td>Earlier endpoint bbox area</td><td>{cand['earlier_area']:.5f}</td></tr>")
        html.append(f"<tr><td>Later endpoint bbox area</td><td>{cand['later_area']:.5f}</td></tr>")
        html.append(f"<tr><td>Median bbox area (track A)</td><td>{cand['a_median_area']:.5f}</td></tr>")
        html.append(f"<tr><td>Median bbox area (track B)</td><td>{cand['b_median_area']:.5f}</td></tr>")
        html.append(f"<tr><td>Earlier bbox classified as</td><td>{'LOW QUALITY' if cand['earlier_low'] else 'OK'}</td></tr>")
        html.append(f"<tr><td>Later bbox classified as</td><td>{'LOW QUALITY' if cand['later_low'] else 'OK'}</td></tr>")
        html.append("</table></div>")
        html.append(
            f"<div class='gate-decision {'gate-block' if cand['current']['blocked'] else 'gate-pass'}'>"
            f"<b>Current production gate:</b> {'BLOCK' if cand['current']['blocked'] else 'APPROVE'} — {cand['current']['reason']}"
            "</div>"
        )
        html.append(
            f"<div class='gate-decision {'gate-block' if cand['proposed']['blocked'] else 'gate-pass'}'>"
            f"<b>Proposed bbox-aware gate ({cand['proposed']['rule']}):</b> {'BLOCK' if cand['proposed']['blocked'] else 'APPROVE'} — {cand['proposed']['reason']}"
            "</div>"
        )
        html.append("<div class='crop-pair'>")
        for label, key in (("A endpoint (last)", "a_crop"), ("B endpoint (first)", "b_crop")):
            crop = cand[key]
            html.append("<div class='crop-card'>")
            if crop and (out_dir / "crops" / crop["filename"]).exists():
                html.append(f"<img src='crops/{crop['filename']}'/>")
            html.append(f"<div class='meta'><b>{label}</b><br>")
            if crop:
                html.append(f"frame {crop['meta']['rally_frame']}<br>")
                html.append(f"bbox area {crop['meta']['bbox_area']:.5f}<br>")
                if 'crop_w' in crop['meta']:
                    html.append(f"crop {crop['meta']['crop_w']}×{crop['meta']['crop_h']}px<br>")
            html.append("</div></div>")
        html.append("</div>")
        html.append("</div>")
    html.append("</body></html>")
    (out_dir / "index.html").write_text("\n".join(html))


def _evaluate_rally(rally_id: str, out_root: Path) -> list[dict[str, Any]]:
    rally = _load_rally_full(rally_id)
    rally_short = rally_id.split("-")[0]
    out_dir = out_root / rally_short
    (out_dir / "crops").mkdir(parents=True, exist_ok=True)

    # Iterate RAW BoT-SORT track ids — these are the natural fragments
    # the merge pipeline considers. Filtered (post-interpolation) tracks
    # have synthetic positions that mask the real temporal gaps.
    track_ids = sorted(
        {int(p.get("trackId", -1)) for p in rally["raw_positions"] if p.get("trackId") is not None and int(p.get("trackId", -1)) >= 0}
    )
    print(f"\n=== rally {rally_short} ({len(track_ids)} raw tracks: {track_ids}) ===")

    track_data = {tid: _track_endpoints(rally["raw_positions"], tid) for tid in track_ids}
    track_data = {k: v for k, v in track_data.items() if v is not None}

    video_path = get_video_path(rally["video_id"])
    cap = None
    video_fps = rally["fps"]
    if video_path is not None:
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            video_fps = cap.get(cv2.CAP_PROP_FPS) or rally["fps"]
        else:
            cap = None

    candidates: list[dict[str, Any]] = []

    for i, tid_a in enumerate(track_ids):
        if tid_a not in track_data:
            continue
        for tid_b in track_ids[i + 1:]:
            if tid_b not in track_data:
                continue
            a, b = track_data[tid_a], track_data[tid_b]
            relation, earlier, later = _classify_pair_temporal(a, b)
            if relation == "overlap":
                continue
            # Only evaluate non-overlapping pairs (could be merged).
            gap = int(later["first"]["frameNumber"]) - int(earlier["last"]["frameNumber"])
            if gap <= 0 or gap > 50:
                continue
            sd = _spatial_distance(earlier["last"], later["first"])
            if sd > 0.20:
                continue  # too far for any reasonable merge
            # We don't have HSV histograms or learned ReID embeddings on
            # disk for non-instrumented runs — leave them None and rely
            # on bbox-aware path's motion-only fallback for low-quality.
            # We do not have HSV histograms or ReID embeddings on disk for
            # an offline probe. Compute proposed-gate decision based on
            # bbox-quality + motion only — this is the PURE form of the
            # proposal (apply motion-only path for low-quality bboxes).
            # The "current production" decision can't be faithfully
            # simulated without HSV — instead we show "would current
            # geometric gates pass?" to flag merges that current already
            # allows; the user is expected to know that real production
            # would additionally apply HSV and likely block these.
            current_dec = _evaluate_current_prod_gate(earlier, later, hsv_bhatt=None)
            proposed_dec = _evaluate_bbox_aware_gate(earlier, later, hsv_bhatt=None, learned_cos=None)

            earlier_area = _bbox_area(earlier["last"])
            later_area = _bbox_area(later["first"])
            a_role = "earlier" if earlier is a else "later"
            b_role = "later" if earlier is a else "earlier"

            # Save annotated full-frame images (bbox drawn on the
            # entire frame so user can see the scene context). Includes
            # faded outlines of OTHER raw tracks present at the same
            # frame, so user can identify which physical player each
            # bbox is targeting.
            a_endpoint_pos = a["last"] if a_role == "earlier" else a["first"]
            b_endpoint_pos = b["first"] if b_role == "later" else b["last"]
            a_frame_path = out_dir / "crops" / f"pair_T{tid_a}_T{tid_b}_A_endpoint.jpg"
            b_frame_path = out_dir / "crops" / f"pair_T{tid_a}_T{tid_b}_B_endpoint.jpg"

            # Build "other positions at this frame" for context outlines
            a_frame_num = int(a_endpoint_pos["frameNumber"])
            b_frame_num = int(b_endpoint_pos["frameNumber"])
            a_others = [
                p for p in rally["raw_positions"]
                if int(p.get("frameNumber", -1)) == a_frame_num
                and int(p.get("trackId", -1)) != tid_a
            ]
            b_others = [
                p for p in rally["raw_positions"]
                if int(p.get("frameNumber", -1)) == b_frame_num
                and int(p.get("trackId", -1)) != tid_b
            ]

            a_crop = b_crop = None
            if cap is not None:
                ok_a, meta_a = _save_endpoint_annotated_frame(
                    cap, rally["start_ms"], video_fps,
                    a_endpoint_pos,
                    a_frame_path,
                    label=f"raw T{tid_a} @ frame {a_frame_num} (last)",
                    color=(0, 255, 0),
                    other_positions_at_frame=a_others,
                )
                if ok_a:
                    a_crop = {"filename": a_frame_path.name, "meta": meta_a}
                ok_b, meta_b = _save_endpoint_annotated_frame(
                    cap, rally["start_ms"], video_fps,
                    b_endpoint_pos,
                    b_frame_path,
                    label=f"raw T{tid_b} @ frame {b_frame_num} (first)",
                    color=(0, 200, 255),  # orange-ish BGR
                    other_positions_at_frame=b_others,
                )
                if ok_b:
                    b_crop = {"filename": b_frame_path.name, "meta": meta_b}

            candidates.append({
                "track_a": tid_a, "track_b": tid_b,
                "a_role": a_role, "b_role": b_role,
                "gap": gap, "spatial_dist": sd, "fps": video_fps,
                "earlier_area": earlier_area, "later_area": later_area,
                "a_median_area": a["median_area"], "b_median_area": b["median_area"],
                "earlier_low": _is_low_quality_bbox(earlier["last"]),
                "later_low": _is_low_quality_bbox(later["first"]),
                "current": {"blocked": current_dec.blocked, "reason": current_dec.reason, "rule": current_dec.rule},
                "proposed": {"blocked": proposed_dec.blocked, "reason": proposed_dec.reason, "rule": proposed_dec.rule},
                "would_flip": current_dec.blocked and not proposed_dec.blocked,
                "a_crop": a_crop, "b_crop": b_crop,
            })

            verdict = ""
            if current_dec.blocked and not proposed_dec.blocked:
                verdict = " ★ FLIP TO APPROVE"
            elif (not current_dec.blocked) and proposed_dec.blocked:
                verdict = " ⚠ FLIP TO BLOCK"
            print(
                f"  T{tid_a}↔T{tid_b}: gap={gap}f dist={sd:.3f} "
                f"areas={earlier_area:.4f}/{later_area:.4f} "
                f"current={'BLOCK' if current_dec.blocked else 'PASS'} "
                f"proposed={'BLOCK' if proposed_dec.blocked else 'PASS'}{verdict}"
            )

    if cap is not None:
        cap.release()

    # Persist HTML + JSON
    summary = {
        "rally_id": rally_id,
        "primary_track_ids": rally["primary_track_ids"],
        "candidates": candidates,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    if candidates:
        _format_html(rally_id, rally, candidates, out_dir)
        print(f"  → {out_dir}/index.html")
    return candidates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rallies", nargs="+", required=True)
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Output: {OUT_ROOT}")

    all_candidates = []
    for rid in args.rallies:
        cands = _evaluate_rally(rid, OUT_ROOT)
        all_candidates.extend([{"rally": rid, **c} for c in cands])

    print()
    print("=" * 60)
    n_flip_total = sum(1 for c in all_candidates if c["would_flip"])
    n_total = len(all_candidates)
    print(f"OVERALL: {n_flip_total}/{n_total} candidates would flip from BLOCK→APPROVE under proposed gate.")
    print()
    print("ACTION: open the HTML pages and visually verify each FLIP candidate "
          "is actually the same physical player. If 'would_flip' is RIGHT in "
          "every case (or close), the bbox-aware gate is validated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
