"""Session 5 — enumerate within-team convergence events for human labelling.

Walks all DB rallies that have tracking ground truth. For each rally, runs
``detect_convergence_periods`` on the stored predictions, filters to pairs
where both tracks are (a) in ``primary_track_ids`` and (b) on the same team
via ``team_assignments``. Cross-references with the audit's SAME_TEAM_SWAP
switches (a hint for the labeller, not a label). Samples representative
crops for the pre / post windows and saves thumbnails for the mini-app.

Output:
    reports/occlusion_resolver/events.json
    reports/occlusion_resolver/crops/{rally_id}/{event_id}/{pre|post}_T{tid}_F{frame}.jpg

Usage:
    uv run python scripts/enumerate_convergence_events.py
    uv run python scripts/enumerate_convergence_events.py --limit 3  # smoke test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
from numpy.typing import NDArray

from rallycut.evaluation.tracking.audit import (
    SwitchCause,
    build_rally_audit,
)
from rallycut.evaluation.tracking.db import (
    TrackingEvaluationRally,
    get_video_path,
    load_labeled_rallies,
)
from rallycut.tracking.color_repair import (
    ConvergencePeriod,
    detect_convergence_periods,
)
from rallycut.tracking.player_features import extract_bbox_crop
from rallycut.tracking.player_tracker import PlayerPosition

ANALYSIS_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ANALYSIS_ROOT / "reports" / "occlusion_resolver"
CROPS_DIR = OUT_DIR / "crops"

# Session 4 retrack audits reflect the current tracking-pipeline state
# (still contains the SAME_TEAM_SWAPs that production post-cleanup passes
# have since resolved in DB). Use them for the `crossed_switch` hint.
RETRACK_AUDIT_DIR = (
    ANALYSIS_ROOT / "reports" / "within_team_reid" / "sweep" / "w0.00" / "audit"
)

# Window + sampling params (align with the plan)
WINDOW_FRAMES = 30          # frames before/after convergence to consider
SEPARATION_GAP = 10         # skip frames immediately around the convergence
NUM_SAMPLE_CROPS = 8        # per (track, pre|post) window
MIN_WINDOW_FRAMES = 6       # abstain-threshold hint for the labeller
CROSSED_SWITCH_SLACK = 15   # frames of slack around a SAME_TEAM_SWAP event
IOU_THRESHOLD = 0.3
MIN_OVERLAP_FRAMES = 5
CROP_THUMB_JPEG_QUALITY = 85

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("enumerate_convergence_events")


@dataclass
class ConvergenceEvent:
    event_id: str
    rally_id: str
    video_id: str
    track_a: int
    track_b: int
    team: int
    start_frame: int
    end_frame: int
    duration_frames: int
    crossed_switch: bool
    audit_same_team_switches: list[int]  # frame numbers
    pre_frames_a: list[int]
    pre_frames_b: list[int]
    post_frames_a: list[int]
    post_frames_b: list[int]
    crops: dict[str, list[str]] = field(default_factory=dict)
    court_split_y: float | None = None
    difficulty_score: float = 0.0  # sort key: min(pre,post) count, higher=easier
    source: str = "db_convergence"  # "db_convergence" | "retrack_audit_swap"


def _positions_by_track(
    positions: list[PlayerPosition],
) -> dict[int, list[PlayerPosition]]:
    out: dict[int, list[PlayerPosition]] = {}
    for p in positions:
        if p.track_id >= 0:
            out.setdefault(p.track_id, []).append(p)
    for tid in out:
        out[tid].sort(key=lambda p: p.frame_number)
    return out


def _sample_window_frames(
    positions: list[PlayerPosition],
    window_start: int,
    window_end: int,
    n_samples: int,
) -> list[int]:
    """Evenly-spaced up to n frames from positions whose frame_number is in [window_start, window_end)."""
    in_window = [
        p.frame_number for p in positions
        if window_start <= p.frame_number <= window_end
    ]
    if not in_window:
        return []
    if len(in_window) <= n_samples:
        return in_window
    import numpy as np
    idx = np.linspace(0, len(in_window) - 1, n_samples).astype(int)
    return [in_window[i] for i in idx]


def _build_events_for_rally(
    rally: TrackingEvaluationRally,
) -> list[ConvergenceEvent]:
    # Prefer retrack-post-processed predictions if available (matches the
    # Session 4 retrack audit namespace — track IDs resolve to the same
    # post-processed tracks the audit's realSwitches reference). Fall back
    # to DB-stored predictions.
    result = _retrack_post_processed(rally)
    if result is not None:
        positions, team_assignments, primary_track_ids, court_split_y = result
    else:
        if rally.predictions is None:
            return []
        positions = rally.predictions.positions
        team_assignments = dict(rally.predictions.team_assignments or {})
        primary_track_ids = set(rally.predictions.primary_track_ids or [])
        court_split_y = rally.predictions.court_split_y

    if not positions:
        return []
    if not primary_track_ids:
        logger.info(
            "  rally %s: no primary_track_ids — skipping", rally.rally_id[:8],
        )
        return []

    if not team_assignments and court_split_y is not None:
        from rallycut.tracking.player_filter import classify_teams

        primary_positions = [
            p for p in positions if p.track_id in primary_track_ids
        ]
        team_assignments = classify_teams(primary_positions, court_split_y)

    if not team_assignments:
        logger.info(
            "  rally %s: could not derive team_assignments — skipping",
            rally.rally_id[:8],
        )
        return []

    periods = detect_convergence_periods(
        positions,
        iou_threshold=IOU_THRESHOLD,
        min_duration=MIN_OVERLAP_FRAMES,
    )
    same_team_periods: list[ConvergencePeriod] = []
    for p in periods:
        a, b = p.track_a, p.track_b
        if a not in primary_track_ids or b not in primary_track_ids:
            continue
        ta = team_assignments.get(a)
        tb = team_assignments.get(b)
        if ta is None or tb is None or ta != tb:
            continue
        same_team_periods.append(p)

    # NOTE: do NOT early-return on empty same_team_periods — Set B (retrack
    # audit swap pairs) may still produce events even when no IoU-overlap
    # convergence is detected (same-team swaps often happen without bbox
    # overlap, e.g. at the ball or at net-joust).

    # Audit cross-reference for crossed_switch hint. Source priority:
    # (1) Session 4 retrack audit (reflects current tracking-pipeline state
    # including SAME_TEAM_SWAPs that DB post-cleanup passes have masked),
    # (2) fall back to running build_rally_audit on DB predictions.
    audit_switches: list[int] = []
    retrack_audit_path = RETRACK_AUDIT_DIR / f"{rally.rally_id}.json"
    if retrack_audit_path.exists():
        try:
            data = json.loads(retrack_audit_path.read_text())
            for g in data.get("perGt", []):
                for sw in g.get("realSwitches", []):
                    if sw.get("cause") == SwitchCause.SAME_TEAM_SWAP.value:
                        audit_switches.append(int(sw["frame"]))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "  rally %s: retrack audit read failed (%s)",
                rally.rally_id[:8], exc,
            )

    if not audit_switches:
        try:
            rally_audit = build_rally_audit(
                rally_id=rally.rally_id,
                video_id=rally.video_id,
                ground_truth=rally.ground_truth,
                predictions=rally.predictions,
            )
            for g in rally_audit.per_gt:
                for sw in g.real_switches:
                    if sw.cause == SwitchCause.SAME_TEAM_SWAP:
                        audit_switches.append(sw.frame)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "  rally %s: DB audit build failed (%s) — no crossed_switch hints",
                rally.rally_id[:8], exc,
            )

    by_track = _positions_by_track(positions)
    events: list[ConvergenceEvent] = []

    # -------------------------------------------------------------------
    # Set B: known same-team swap events mined from the retrack audit.
    # The audit records a SAME_TEAM_SWAP per GT track that changed its
    # pred-id assignment. A classic pred-exchange swap shows up as TWO
    # symmetric entries at ~the same frame: {predId=X, gtOld=A, gtNew=B}
    # on gt_A, and {predId=Y, gtOld=B, gtNew=A} on gt_B. Pair them to
    # recover the (pred_X, pred_Y) couple — that's what the resolver
    # will need to evaluate.
    # -------------------------------------------------------------------
    retrack_swap_pairs: list[tuple[int, int, int, int, int]] = []  # (frame, gt_a, gt_b, pred_a, pred_b)
    if retrack_audit_path.exists():
        try:
            data = json.loads(retrack_audit_path.read_text())
            same_team_switches_raw: list[dict] = []
            for g in data.get("perGt", []):
                for sw in g.get("realSwitches", []):
                    if sw.get("cause") == SwitchCause.SAME_TEAM_SWAP.value:
                        same_team_switches_raw.append({
                            "gtTrackId": int(g["gtTrackId"]),
                            "frame": int(sw["frame"]),
                            "predId": int(sw["predId"]),
                            "gtOld": int(sw["gtOld"]),
                            "gtNew": int(sw["gtNew"]),
                        })
            used: set[int] = set()
            for i, sw_a in enumerate(same_team_switches_raw):
                if i in used:
                    continue
                for j in range(i + 1, len(same_team_switches_raw)):
                    if j in used:
                        continue
                    sw_b = same_team_switches_raw[j]
                    if (
                        abs(sw_a["frame"] - sw_b["frame"]) <= 3
                        and sw_a["gtOld"] == sw_b["gtNew"]
                        and sw_a["gtNew"] == sw_b["gtOld"]
                        and sw_a["predId"] != sw_b["predId"]
                    ):
                        frame = (sw_a["frame"] + sw_b["frame"]) // 2
                        retrack_swap_pairs.append((
                            frame,
                            sw_a["gtTrackId"], sw_b["gtTrackId"],
                            sw_a["predId"], sw_b["predId"],
                        ))
                        used.add(i)
                        used.add(j)
                        break
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "  rally %s: retrack-audit pairing failed (%s)",
                rally.rally_id[:8], exc,
            )

    # -------------------------------------------------------------------
    # Set A: DB-predictions convergence periods (primary mostly-negatives
    # corpus).
    # -------------------------------------------------------------------
    for idx, cp in enumerate(same_team_periods):
        pre_start = cp.start_frame - WINDOW_FRAMES
        pre_end = cp.start_frame - 1
        post_start = cp.end_frame + SEPARATION_GAP
        post_end = post_start + WINDOW_FRAMES

        pre_a = _sample_window_frames(by_track.get(cp.track_a, []), pre_start, pre_end, NUM_SAMPLE_CROPS)
        pre_b = _sample_window_frames(by_track.get(cp.track_b, []), pre_start, pre_end, NUM_SAMPLE_CROPS)
        post_a = _sample_window_frames(by_track.get(cp.track_a, []), post_start, post_end, NUM_SAMPLE_CROPS)
        post_b = _sample_window_frames(by_track.get(cp.track_b, []), post_start, post_end, NUM_SAMPLE_CROPS)

        crossed_frames = [
            f for f in audit_switches
            if (cp.start_frame - CROSSED_SWITCH_SLACK) <= f <= (cp.end_frame + CROSSED_SWITCH_SLACK)
        ]
        event_id = f"e{idx:03d}_T{cp.track_a}T{cp.track_b}_f{cp.start_frame}-{cp.end_frame}"
        difficulty = float(min(len(pre_a), len(pre_b), len(post_a), len(post_b)))
        events.append(ConvergenceEvent(
            event_id=event_id,
            rally_id=rally.rally_id,
            video_id=rally.video_id,
            track_a=cp.track_a,
            track_b=cp.track_b,
            team=int(team_assignments[cp.track_a]),
            start_frame=cp.start_frame,
            end_frame=cp.end_frame,
            duration_frames=cp.end_frame - cp.start_frame + 1,
            crossed_switch=bool(crossed_frames),
            audit_same_team_switches=sorted(crossed_frames),
            pre_frames_a=pre_a,
            pre_frames_b=pre_b,
            post_frames_a=post_a,
            post_frames_b=post_b,
            court_split_y=court_split_y,
            difficulty_score=difficulty,
            source="db_convergence",
        ))

    # -------------------------------------------------------------------
    # Set B emission: retrack-audit swap spans that don't already overlap
    # an existing DB-convergence event we emitted above. These are
    # likely-positive events drawn from the retrack tracking state.
    # Tracks referenced here are from the retrack prediction namespace;
    # the labeller shows video evidence, not pred-id-numbered bboxes, so
    # the track-namespace mismatch doesn't confuse a human reviewer — but
    # we DO still need their bboxes to extract crops. Load them from the
    # retrack cache.
    # -------------------------------------------------------------------
    # Set B pairs reference post-processed pred-ids from the audit, so
    # look up positions from the SAME post-processed index used above.
    retrack_by_track = by_track

    for swap_idx, (frame, gt_a, gt_b, pred_a, pred_b) in enumerate(
        retrack_swap_pairs
    ):
        # Window around the swap frame
        start = max(0, frame - 3)
        end = frame + 3
        pre_a = _sample_window_frames(
            retrack_by_track.get(pred_a, []), start - WINDOW_FRAMES, start - 1, NUM_SAMPLE_CROPS,
        )
        pre_b = _sample_window_frames(
            retrack_by_track.get(pred_b, []), start - WINDOW_FRAMES, start - 1, NUM_SAMPLE_CROPS,
        )
        post_a = _sample_window_frames(
            retrack_by_track.get(pred_a, []), end + SEPARATION_GAP, end + SEPARATION_GAP + WINDOW_FRAMES, NUM_SAMPLE_CROPS,
        )
        post_b = _sample_window_frames(
            retrack_by_track.get(pred_b, []), end + SEPARATION_GAP, end + SEPARATION_GAP + WINDOW_FRAMES, NUM_SAMPLE_CROPS,
        )
        # Accept even partial windows (swap events are sparse — lower bar).
        if min(len(pre_a), len(pre_b), len(post_a), len(post_b)) < 2:
            continue
        event_id = f"rsw{swap_idx:03d}_gt{gt_a}gt{gt_b}_T{pred_a}T{pred_b}_f{frame}"
        events.append(ConvergenceEvent(
            event_id=event_id,
            rally_id=rally.rally_id,
            video_id=rally.video_id,
            track_a=pred_a,
            track_b=pred_b,
            team=int(team_assignments.get(pred_a, 0)),
            start_frame=start,
            end_frame=end,
            duration_frames=end - start + 1,
            crossed_switch=True,
            audit_same_team_switches=[frame],
            pre_frames_a=pre_a,
            pre_frames_b=pre_b,
            post_frames_a=post_a,
            post_frames_b=post_b,
            court_split_y=court_split_y,
            difficulty_score=float(min(len(pre_a), len(pre_b), len(post_a), len(post_b))),
            source="retrack_audit_swap",
        ))
    return events


def _load_retrack_positions(
    rally: TrackingEvaluationRally,
) -> list[PlayerPosition]:
    """Load raw (pre-post-processing) tracked positions from the retrack cache."""
    try:
        from rallycut.cli.commands.evaluate_tracking import (
            _compute_tracker_config_hash,
        )
        from rallycut.evaluation.tracking.retrack_cache import RetrackCache
    except Exception as exc:  # noqa: BLE001
        logger.warning("  retrack cache import failed (%s)", exc)
        return []
    rc = RetrackCache()
    entry = rc.get(rally.rally_id, _compute_tracker_config_hash())
    if entry is None:
        return []
    cached_data, _color, _app, _learned = entry
    return list(cached_data.positions)


_rerun_cache: dict[
    str, tuple[list[PlayerPosition], dict[int, int], set[int], float | None]
] = {}


def _retrack_post_processed(
    rally: TrackingEvaluationRally,
) -> tuple[list[PlayerPosition], dict[int, int], set[int], float | None] | None:
    """Run apply_post_processing on the cached retrack entry and return
    the final post-processed positions + team_assignments + primary_track_ids
    + court_split_y. Result is cached in-process.

    Matches the prediction namespace used to build the Session 4 retrack
    audit — so pred-ids here will line up with the audit's realSwitches.
    """
    if rally.rally_id in _rerun_cache:
        return _rerun_cache[rally.rally_id]
    try:
        from rallycut.cli.commands.evaluate_tracking import (
            _compute_tracker_config_hash,
        )
        from rallycut.evaluation.tracking.retrack_cache import RetrackCache
        from rallycut.tracking.player_filter import PlayerFilterConfig
        from rallycut.tracking.player_tracker import PlayerTracker
    except Exception as exc:  # noqa: BLE001
        logger.warning("  retrack post-proc import failed (%s)", exc)
        return None

    rc = RetrackCache()
    entry = rc.get(rally.rally_id, _compute_tracker_config_hash())
    if entry is None:
        return None
    cached_data, color_store, appearance_store, learned_store = entry
    try:
        filter_config = PlayerFilterConfig().scaled_for_fps(cached_data.video_fps)
        result = PlayerTracker.apply_post_processing(
            positions=cached_data.positions,
            raw_positions=list(cached_data.positions),
            color_store=color_store,
            appearance_store=appearance_store,
            ball_positions=cached_data.ball_positions,
            video_fps=cached_data.video_fps,
            video_width=cached_data.video_width,
            video_height=cached_data.video_height,
            frame_count=cached_data.frame_count,
            start_frame=0,
            filter_enabled=True,
            filter_config=filter_config,
            learned_store=learned_store,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "  rally %s: retrack post-proc failed (%s)",
            rally.rally_id[:8], exc,
        )
        return None

    positions = result.positions
    primary_track_ids = set(result.primary_track_ids or [])
    team_assignments = dict(result.team_assignments or {})
    court_split_y = result.court_split_y
    if not team_assignments and court_split_y is not None and primary_track_ids:
        from rallycut.tracking.player_filter import classify_teams

        primary_positions = [
            p for p in positions if p.track_id in primary_track_ids
        ]
        team_assignments = classify_teams(primary_positions, court_split_y)

    out = (positions, team_assignments, primary_track_ids, court_split_y)
    _rerun_cache[rally.rally_id] = out
    return out


def _read_rally_frames(
    video_path: Path,
    start_ms: int,
    video_fps: float,
    rally_frames: set[int],
) -> dict[int, NDArray]:
    """Read a set of rally-relative frames from video. Sequential read after
    seeking to the earliest-needed frame. Returns rally_frame → BGR ndarray.
    Lifted from scripts/probe_reid_models_on_swaps.py::_read_rally_frames.
    """
    import numpy as np
    if not rally_frames:
        return {}
    start_frame = int(round(start_ms / 1000.0 * video_fps))
    abs_needed = sorted(start_frame + f for f in rally_frames)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("  cannot open %s", video_path)
        return {}

    out: dict[int, NDArray] = {}
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, abs_needed[0])
        cur = abs_needed[0]
        idx = 0
        while idx < len(abs_needed):
            target = abs_needed[idx]
            while cur < target:
                cap.read()
                cur += 1
                if cur > target + 10_000:
                    break
            ok, frame = cap.read()
            cur += 1
            if not ok:
                break
            if target == abs_needed[idx]:
                rf = target - start_frame
                out[rf] = np.asarray(frame, dtype=np.uint8)
                idx += 1
    finally:
        cap.release()
    return out


def _extract_and_save_crops(
    rally: TrackingEvaluationRally,
    events: list[ConvergenceEvent],
    video_path: Path,
    positions: list[PlayerPosition],
) -> None:
    if not events:
        return
    # Collect all (frame, track) tuples we need crops for.
    frame_tid_set: set[tuple[int, int]] = set()
    for ev in events:
        for f in ev.pre_frames_a + ev.post_frames_a:
            frame_tid_set.add((f, ev.track_a))
        for f in ev.pre_frames_b + ev.post_frames_b:
            frame_tid_set.add((f, ev.track_b))
    if not frame_tid_set:
        return

    # Read the frames in one sequential pass.
    frames_needed = {f for f, _ in frame_tid_set}
    frames_by_number = _read_rally_frames(
        video_path=video_path,
        start_ms=rally.start_ms,
        video_fps=rally.video_fps,
        rally_frames=frames_needed,
    )
    if not frames_by_number:
        return

    # Index positions by (track, frame). Use the caller-supplied positions —
    # the events reference track IDs from the post-processed retrack
    # namespace, which differs from rally.predictions (DB) for Set B events.
    pos_by_key: dict[tuple[int, int], PlayerPosition] = {}
    for p in positions:
        if p.track_id >= 0:
            pos_by_key[(p.track_id, p.frame_number)] = p

    for ev in events:
        event_dir = CROPS_DIR / ev.rally_id / ev.event_id
        event_dir.mkdir(parents=True, exist_ok=True)
        crops_map: dict[str, list[str]] = {
            "pre_a": [], "pre_b": [], "post_a": [], "post_b": [],
        }
        for group, frames, tid in [
            ("pre_a", ev.pre_frames_a, ev.track_a),
            ("pre_b", ev.pre_frames_b, ev.track_b),
            ("post_a", ev.post_frames_a, ev.track_a),
            ("post_b", ev.post_frames_b, ev.track_b),
        ]:
            for f in frames:
                frame = frames_by_number.get(f)
                if frame is None:
                    continue
                pos = pos_by_key.get((tid, f))
                if pos is None:
                    continue
                crop = extract_bbox_crop(
                    frame,
                    (pos.x, pos.y, pos.width, pos.height),
                    rally.video_width,
                    rally.video_height,
                )
                if crop is None or crop.size == 0:
                    continue
                out_path = event_dir / f"{group}_T{tid}_F{f:05d}.jpg"
                cv2.imwrite(
                    str(out_path),
                    crop,
                    [int(cv2.IMWRITE_JPEG_QUALITY), CROP_THUMB_JPEG_QUALITY],
                )
                # Store relative path from OUT_DIR (labeller loads via relative URLs).
                crops_map[group].append(
                    str(out_path.relative_to(OUT_DIR))
                )
        ev.crops = crops_map


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N rallies (debug).",
    )
    parser.add_argument(
        "--skip-crops", action="store_true",
        help="Skip video reads + crop extraction (fast smoke test of counts).",
    )
    parser.add_argument(
        "--out", type=Path, default=OUT_DIR / "events.json",
        help="Output JSON path.",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("loading labelled rallies from DB...")
    rallies = load_labeled_rallies()
    if args.limit:
        rallies = rallies[: args.limit]
    logger.info("  %d rallies", len(rallies))

    all_events: list[ConvergenceEvent] = []
    n_with_events = 0
    for i, rally in enumerate(rallies, start=1):
        t0 = time.time()
        events = _build_events_for_rally(rally)
        if not events:
            logger.info(
                "[%d/%d] %s: 0 same-team convergences (%.1fs)",
                i, len(rallies), rally.rally_id[:8], time.time() - t0,
            )
            continue
        n_with_events += 1
        if not args.skip_crops:
            video_path = get_video_path(rally.video_id)
            if video_path and video_path.exists():
                # Use the same post-processed positions the events reference.
                cached_pp = _rerun_cache.get(rally.rally_id)
                if cached_pp is not None:
                    positions = cached_pp[0]
                elif rally.predictions is not None:
                    positions = rally.predictions.positions
                else:
                    positions = []
                _extract_and_save_crops(rally, events, video_path, positions)
            else:
                logger.warning(
                    "  rally %s: no local video — crops skipped",
                    rally.rally_id[:8],
                )
        all_events.extend(events)
        logger.info(
            "[%d/%d] %s: %d events (%d crossed_switch, %.1fs)",
            i, len(rallies), rally.rally_id[:8],
            len(events), sum(1 for e in events if e.crossed_switch),
            time.time() - t0,
        )

    # Sort events: higher difficulty (more evidence) first, then crossed_switch hints grouped
    all_events.sort(key=lambda e: (-e.difficulty_score, -int(e.crossed_switch)))

    payload = {
        "version": 1,
        "schema": "rallycut-session5-convergence-events-v1",
        "window_frames": WINDOW_FRAMES,
        "separation_gap": SEPARATION_GAP,
        "num_sample_crops": NUM_SAMPLE_CROPS,
        "min_window_frames": MIN_WINDOW_FRAMES,
        "n_rallies": len(rallies),
        "n_rallies_with_events": n_with_events,
        "n_events": len(all_events),
        "events": [asdict(e) for e in all_events],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))

    logger.info("")
    logger.info("=== summary ===")
    logger.info("  rallies inspected: %d", len(rallies))
    logger.info("  rallies with events: %d", n_with_events)
    logger.info("  total events: %d", len(all_events))
    logger.info(
        "  events with audit crossed_switch: %d",
        sum(1 for e in all_events if e.crossed_switch),
    )
    logger.info("  wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
