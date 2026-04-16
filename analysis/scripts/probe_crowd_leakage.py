"""Crowded-scene diagnostic — detect non-player tracks and raw-detection crowd load.

For every rally in the DB (optionally filtered by --video-id / --all-labeled),
this probe quantifies three independent signals:

1. Raw-detection density — how many YOLO detections per frame on average, and
   how many frames have > MAX_REASONABLE raw dets (strong crowd signal).
2. Suspect primary tracks — any primary track whose median position/size
   looks non-player (outside calibrated court, too small, too close to frame
   edge).
3. Video metadata flags — crowdLevel / cameraAngle / sceneComplexity from
   `videos.quality_report_json` when present.

Outputs:
    reports/tracking_audit/crowd_leak/<rally_id>.json
    reports/tracking_audit/crowd_leak/_summary.md

Usage:
    uv run python scripts/probe_crowd_leakage.py --all-labeled
    uv run python scripts/probe_crowd_leakage.py --video-id <id>
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from rallycut.evaluation.db import get_connection

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("crowd-probe")

# Thresholds — picked conservatively so we surface the truly suspect, not marginal cases.
EXPECTED_MAX_PLAYERS = 4           # 4 on court + occasional ref/libero
CROWD_FRAME_RAW_COUNT = 8          # Per-frame raw dets above this ⇒ definitely crowd
CROWD_RALLY_MEAN_THRESHOLD = 5.5   # Rally mean raw dets/frame above this ⇒ crowd rally
MIN_MEDIAN_HEIGHT = 0.05           # Primary < 5% frame height is probably a distant bystander
EDGE_MARGIN = 0.015                # Primary median within 1.5% of frame edge
STATIC_TRACK_SPREAD = 0.01         # Track with total xy spread < 1% — stationary (ref/chair)
STATIC_MIN_FRAMES = 30             # Need ≥30 frames to reliably call a track stationary


def _list_rallies(
    video_id: str | None,
    all_labeled: bool,
    rally_id: str | None,
) -> list[tuple[str, str]]:
    """Return [(rally_id, video_id), ...] matching the filters."""
    clauses = ["pt.status = 'COMPLETED'", "pt.positions_json IS NOT NULL"]
    params: list[Any] = []
    if rally_id:
        clauses.append("pt.rally_id = %s")
        params.append(rally_id)
    elif video_id:
        clauses.append("r.video_id = %s")
        params.append(video_id)
    elif all_labeled:
        clauses.append("pt.ground_truth_json IS NOT NULL")
    sql = f"""
        SELECT r.id, r.video_id
        FROM rallies r JOIN player_tracks pt ON pt.rally_id = r.id
        WHERE {' AND '.join(clauses)}
        ORDER BY r.video_id, r.start_ms
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return [(str(r[0]), str(r[1])) for r in cur.fetchall()]


def _load_rally_data(rally_id: str) -> dict[str, Any] | None:
    sql = """
        SELECT pt.positions_json, pt.raw_positions_json, pt.primary_track_ids,
               pt.frame_count, pt.fps,
               v.quality_report_json, v.court_calibration_json,
               v.width, v.height, v.name, v.filename
        FROM player_tracks pt JOIN rallies r ON r.id = pt.rally_id
        JOIN videos v ON v.id = r.video_id
        WHERE pt.rally_id = %s
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, [rally_id])
            row = cur.fetchone()
    if not row:
        return None
    return {
        "positions_json": row[0] or [],
        "raw_positions_json": row[1] or [],
        "primary_track_ids": list(row[2] or []),
        "frame_count": int(row[3] or 0),
        "fps": float(row[4] or 30.0),
        "quality_report_json": row[5] or {},
        "court_calibration_json": row[6],
        "video_width": int(row[7] or 1920),
        "video_height": int(row[8] or 1080),
        "video_name": row[9] or row[10] or "—",
    }


def _analyze_primary_tracks(
    positions: list[dict],
    primary_ids: list[int],
) -> list[dict[str, Any]]:
    """Identify primary tracks with non-player signatures.

    Image-space heuristics only. Court-plane projection was tried first but
    the stored 4-corner homography extrapolates wildly outside the calibrated
    polygon, producing false positives on legitimate on-court positions.
    These signals are robust without it:
      - too_small: bbox median height < threshold (distant bystander)
      - near_edge_xy: track sits ≤ 1.5% from a frame edge
      - stationary: xy spread < 1% over ≥ 30 frames (ref / chair / umbrella)
    """
    per_track: dict[int, list[dict]] = {}
    for p in positions:
        tid = p["trackId"]
        if tid in primary_ids:
            per_track.setdefault(tid, []).append(p)

    suspect: list[dict[str, Any]] = []
    for tid, track_pts in per_track.items():
        if not track_pts:
            continue
        xs = [p["x"] for p in track_pts]
        ys = [p["y"] for p in track_pts]
        hs = [p["height"] for p in track_pts]
        med_x, med_y, med_h = median(xs), median(ys), median(hs)

        flags: list[str] = []
        if med_h < MIN_MEDIAN_HEIGHT:
            flags.append(f"too_small(h={med_h:.3f})")
        if med_x < EDGE_MARGIN or med_x > 1 - EDGE_MARGIN:
            flags.append(f"near_edge_x(x={med_x:.3f})")
        if med_y < EDGE_MARGIN or med_y > 1 - EDGE_MARGIN:
            flags.append(f"near_edge_y(y={med_y:.3f})")
        if len(track_pts) >= STATIC_MIN_FRAMES:
            xy_spread = (max(xs) - min(xs)) + (max(ys) - min(ys))
            if xy_spread < STATIC_TRACK_SPREAD:
                flags.append(f"stationary(spread={xy_spread:.4f})")

        if flags:
            suspect.append({
                "track_id": tid,
                "frames": len(track_pts),
                "median_xy": (round(med_x, 3), round(med_y, 3)),
                "median_height": round(med_h, 3),
                "flags": flags,
            })
    return suspect


def _analyze_raw_density(
    raw_positions: list[dict],
    frame_count: int,
) -> dict[str, Any]:
    per_frame: Counter[int] = Counter()
    for p in raw_positions:
        per_frame[p["frameNumber"]] += 1
    counts = [per_frame.get(f, 0) for f in range(frame_count)] if frame_count else list(per_frame.values())
    if not counts:
        return {"mean": 0.0, "max": 0, "frames_over_threshold": 0, "share_over_threshold": 0.0}
    arr = np.array(counts, dtype=np.float32)
    over = int((arr > CROWD_FRAME_RAW_COUNT).sum())
    return {
        "mean": round(float(arr.mean()), 2),
        "max": int(arr.max()),
        "frames_over_threshold": over,
        "share_over_threshold": round(over / len(counts), 3) if len(counts) else 0.0,
    }


def analyze_rally(rally_id: str, video_id: str) -> dict[str, Any] | None:
    data = _load_rally_data(rally_id)
    if data is None:
        return None
    raw_density = _analyze_raw_density(
        data["raw_positions_json"], data["frame_count"],
    )
    suspect_tracks = _analyze_primary_tracks(
        positions=data["positions_json"],
        primary_ids=data["primary_track_ids"],
    )
    qr = data["quality_report_json"]
    meta_flags = {
        k: qr[k] for k in ("crowdLevel", "sceneComplexity", "cameraAngle",
                            "cameraDistance", "brightness")
        if isinstance(qr, dict) and k in qr
    }
    is_crowded_rally = raw_density["mean"] > CROWD_RALLY_MEAN_THRESHOLD
    return {
        "rallyId": rally_id,
        "videoId": video_id,
        "videoName": data["video_name"],
        "frameCount": data["frame_count"],
        "rawDensity": raw_density,
        "isCrowdedByRawDensity": is_crowded_rally,
        "suspectPrimaryTracks": suspect_tracks,
        "videoMetaFlags": meta_flags,
    }


def _severity(result: dict[str, Any]) -> int:
    score = 0
    if result.get("isCrowdedByRawDensity"):
        score += 2
    score += len(result.get("suspectPrimaryTracks", []))
    score += int(result.get("rawDensity", {}).get("frames_over_threshold", 0) > 0)
    return score


def render_summary(results: list[dict[str, Any]]) -> str:
    total = len(results)
    crowded = sum(1 for r in results if r.get("isCrowdedByRawDensity"))
    with_suspect = sum(1 for r in results if r.get("suspectPrimaryTracks"))

    # Aggregate raw-density stats
    means = [r["rawDensity"]["mean"] for r in results if r.get("rawDensity")]
    agg_mean = round(sum(means) / len(means), 2) if means else 0.0

    # Flag counts across all suspect tracks
    flag_counts: Counter[str] = Counter()
    for r in results:
        for st in r.get("suspectPrimaryTracks", []):
            for f in st["flags"]:
                flag_counts[f.split("(")[0]] += 1

    lines = [
        "# Crowded-scene / non-player leakage probe",
        "",
        f"Analysed **{total}** rallies. Aggregate raw-detection mean "
        f"**{agg_mean}**/frame.",
        "",
        "## Headline",
        "",
        f"- Crowded rallies (raw mean > {CROWD_RALLY_MEAN_THRESHOLD}): "
        f"**{crowded} / {total}** ({100 * crowded / total:.1f}%).",
        f"- Rallies with suspect primary tracks: **{with_suspect} / {total}** "
        f"({100 * with_suspect / total:.1f}%).",
        "",
        "## Suspect-flag counts (across all flagged primary tracks)",
        "",
        "| Flag | Count |",
        "|---|---:|",
    ]
    for flag, count in flag_counts.most_common():
        lines.append(f"| `{flag}` | {count} |")
    lines.extend([
        "",
        "## Top-20 rallies by severity",
        "",
        "| Severity | Rally | Video | Raw mean | Raw max | Crowd frames | Suspect tracks | Meta flags |",
        "|---:|---|---|---:|---:|---:|---|---|",
    ])
    ranked = sorted(results, key=lambda r: -_severity(r))
    for r in ranked[:20]:
        density = r["rawDensity"]
        suspect_desc = (
            ", ".join(f"#{st['track_id']}:{','.join(f.split('(')[0] for f in st['flags'])}"
                      for st in r["suspectPrimaryTracks"])
            or "—"
        )
        meta_desc = ", ".join(f"{k}={v}" for k, v in r["videoMetaFlags"].items()) or "—"
        lines.append(
            f"| {_severity(r)} | `{r['rallyId'][:8]}` | {r['videoName']} | "
            f"{density['mean']} | {density['max']} | {density['frames_over_threshold']} | "
            f"{suspect_desc} | {meta_desc} |"
        )
    lines.append("")
    lines.extend([
        "## Interpretation",
        "",
        "- **High raw mean + suspect tracks** → crowd/referee leaking into primary set. ",
        "  Likely fixes: tighter court ROI, bbox-size filter, post-filter non-player scorer.",
        "- **High raw mean, no suspect tracks** → ROI + filters are catching crowd. ",
        "  No action needed for those rallies.",
        "- **Suspect tracks but low raw mean** → rare case where a legitimate player ",
        "  behaves oddly (e.g. serves from deep baseline). Verify before rejecting.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rally-id", type=str, default=None)
    parser.add_argument("--video-id", type=str, default=None)
    parser.add_argument("--all-labeled", action="store_true")
    parser.add_argument("--all", action="store_true", help="Every rally in DB")
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("reports/tracking_audit/crowd_leak"),
    )
    args = parser.parse_args()

    if args.all:
        rallies = _list_rallies(None, all_labeled=False, rally_id=None)
    elif args.all_labeled:
        rallies = _list_rallies(None, all_labeled=True, rally_id=None)
    elif args.video_id:
        rallies = _list_rallies(args.video_id, all_labeled=False, rally_id=None)
    elif args.rally_id:
        rallies = _list_rallies(None, all_labeled=False, rally_id=args.rally_id)
    else:
        rallies = _list_rallies(None, all_labeled=True, rally_id=None)

    if not rallies:
        logger.error("no rallies matched the filter")
        raise SystemExit(1)

    logger.info(f"Analysing {len(rallies)} rally(s)...")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for idx, (rid, vid) in enumerate(rallies, start=1):
        result = analyze_rally(rid, vid)
        if result is None:
            continue
        (args.out_dir / f"{rid}.json").write_text(json.dumps(result, indent=2))
        results.append(result)
        severity = _severity(result)
        tag = "⚠" if severity >= 2 else " "
        logger.info(
            f"  [{idx}/{len(rallies)}] {rid[:8]} {tag}  "
            f"raw_mean={result['rawDensity']['mean']}  "
            f"suspect_tracks={len(result['suspectPrimaryTracks'])}  "
            f"crowded={result['isCrowdedByRawDensity']}"
        )

    summary_path = args.out_dir / "_summary.md"
    summary_path.write_text(render_summary(results))
    logger.info(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
