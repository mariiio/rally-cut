#!/usr/bin/env python3
"""Per-rally cost-matrix dump for the cross-rally matcher.

Unlike ``diagnose_match_players.py`` which only shows rallies that have GT,
this script prints the matcher's diagnostics for *every* rally in a video.

Usage:
    uv run python scripts/diagnose_pid_per_rally.py --video-id <uuid>
    uv run python scripts/diagnose_pid_per_rally.py --video-id <uuid> --rallies 6,9,22,23

Read-only: does NOT touch the DB. Runs ``match_players_across_rallies``
in-memory with ``collect_diagnostics=True`` and prints per-rally state.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `uv run python scripts/diagnose_pid_per_rally.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rallycut.court.calibration import CourtCalibrator  # noqa: E402
from rallycut.evaluation.db import get_connection  # noqa: E402
from rallycut.evaluation.tracking.db import (  # noqa: E402
    get_video_path,
    load_rallies_for_video,
)
from rallycut.tracking.match_tracker import (  # noqa: E402
    match_players_across_rallies,
)


def _load_court_calibrator(video_id: str) -> CourtCalibrator | None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT court_calibration_json FROM videos WHERE id = %s",
                [video_id],
            )
            row = cur.fetchone()
    if not row or not row[0]:
        return None
    cal = row[0]
    if not isinstance(cal, list) or len(cal) != 4:
        return None
    calibrator = CourtCalibrator()
    calibrator.calibrate([(c["x"], c["y"]) for c in cal])
    return calibrator if calibrator.is_calibrated else None


def _print_rally(idx_1based: int, rally, result, diag) -> None:
    print(f"\n{'=' * 78}")
    print(
        f"Rally [{idx_1based:2d}]  id={rally.rally_id[:8]}  "
        f"primary_track_ids={rally.primary_track_ids}  "
        f"start_ms={rally.start_ms} end_ms={rally.end_ms}"
    )
    print(f"  trackToPlayer (matcher output): {dict(sorted(result.track_to_player.items()))}")
    print(
        f"  confidence={result.assignment_confidence:.3f}  "
        f"side_switch={result.side_switch_detected}  "
        f"server_pid={result.server_player_id}"
    )
    if result.sub_tracks:
        print(f"  sub_tracks ({len(result.sub_tracks)}):")
        for st in result.sub_tracks:
            print(
                f"    parent={st.parent_track_id} synth={st.synthetic_track_id} "
                f"frames=[{st.f_start},{st.f_end}] reason={getattr(st, 'reason', '?')}"
            )

    if diag is None:
        print("  (no diagnostics — collect_diagnostics was False or rally skipped)")
        return

    # Court sides per track
    print("  track_court_sides:", diag.track_court_sides)

    # Cost matrix
    tids = diag.track_ids
    pids = diag.player_ids
    cm = diag.cost_matrix
    header = "      " + "".join(f"  P{pid:>2}   " for pid in pids) + "  margin"
    print(header)
    for i, tid in enumerate(tids):
        if i >= cm.shape[0]:
            break
        row = f"  T{tid:>3}:"
        assigned_pid = diag.assignment.get(tid)
        for j, pid in enumerate(pids):
            if j >= cm.shape[1]:
                break
            cost = cm[i, j]
            marker = " *" if pid == assigned_pid else "  "
            row += f" {cost:5.3f}{marker}"
        if assigned_pid in diag.assignment_margins:
            row += f"  {diag.assignment_margins[assigned_pid]:.3f}"
        print(row)


def _print_profile_discriminability(profiles) -> None:
    from rallycut.tracking.player_features import (
        TrackAppearanceStats,
        compute_appearance_similarity,
    )

    pids = sorted(profiles.keys())
    if not pids:
        return
    print(f"\n{'=' * 78}")
    print("Final player profile discriminability (Bhattacharyya distance, higher=more distinct):")
    header = "     " + "".join(f"  P{pid:>2}  " for pid in pids)
    print(header)
    for pid_a in pids:
        row = f"  P{pid_a}:"
        for pid_b in pids:
            if pid_a == pid_b:
                row += "  ---  "
            else:
                profile_b = profiles[pid_b]
                stats_b = TrackAppearanceStats(track_id=pid_b)
                stats_b.avg_skin_tone_hsv = profile_b.avg_skin_tone_hsv
                stats_b.avg_upper_hist = profile_b.avg_upper_hist
                stats_b.avg_lower_hist = profile_b.avg_lower_hist
                stats_b.avg_lower_v_hist = profile_b.avg_lower_v_hist
                stats_b.avg_upper_v_hist = profile_b.avg_upper_v_hist
                stats_b.avg_dominant_color_hsv = profile_b.avg_dominant_color_hsv
                cost = compute_appearance_similarity(profiles[pid_a], stats_b)
                row += f" {cost:5.3f}"
        print(row)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video-id", required=True, help="Video UUID")
    ap.add_argument(
        "--rallies",
        default="",
        help="Comma-separated 1-indexed rally indices to focus on (default: all)",
    )
    args = ap.parse_args()

    focus: set[int] = set()
    if args.rallies:
        focus = {int(x.strip()) for x in args.rallies.split(",") if x.strip()}

    rallies = load_rallies_for_video(args.video_id)
    if not rallies:
        print(f"No rallies found for video {args.video_id}")
        sys.exit(1)
    video_path = get_video_path(args.video_id)
    if not video_path:
        print(f"Could not resolve video path for {args.video_id}")
        sys.exit(1)
    calibrator = _load_court_calibrator(args.video_id)
    print(
        f"Loaded {len(rallies)} rallies for {args.video_id[:8]}  "
        f"calibrator={'yes' if calibrator else 'no'}  video={video_path.name}"
    )

    # Match the production CLI: load the general OSNet ReID model when its
    # weights exist. Without this, the margins reported here are *worse* than
    # production (ReID margin gate never passes), so the diagnostic mis-states
    # how confident the matcher really is.
    reid_model = None
    try:
        from rallycut.tracking.reid_general import WEIGHTS_PATH as REID_WEIGHTS_PATH
        if REID_WEIGHTS_PATH.exists():
            from rallycut.tracking.reid_general import GeneralReIDModel
            reid_model = GeneralReIDModel(weights_path=REID_WEIGHTS_PATH)
            print(f"  Loaded ReID weights: {REID_WEIGHTS_PATH.name}")
    except Exception as exc:  # noqa: BLE001
        print(f"  ReID load failed (continuing HSV-only): {exc}")

    print("Running match_players_across_rallies (collect_diagnostics=True)...")
    result = match_players_across_rallies(
        video_path=video_path,
        rallies=rallies,
        calibrator=calibrator,
        collect_diagnostics=True,
        extract_reid=reid_model is not None,
        reid_model=reid_model,
    )

    diag_by_idx = {d.rally_index: d for d in result.diagnostics}

    for i, (rally_data, rally_result) in enumerate(zip(rallies, result.rally_results)):
        idx_1 = i + 1
        if focus and idx_1 not in focus:
            continue
        diag = diag_by_idx.get(i)
        _print_rally(idx_1, rally_data, rally_result, diag)

    _print_profile_discriminability(result.player_profiles)

    # Summary
    print(f"\n{'=' * 78}")
    print(f"SUMMARY: {len(rallies)} rallies, "
          f"avg confidence={sum(r.assignment_confidence for r in result.rally_results)/len(rallies):.3f}")
    print(f"Side switches detected at rally indices: "
          f"{[i+1 for i, r in enumerate(result.rally_results) if r.side_switch_detected]}")


if __name__ == "__main__":
    main()
