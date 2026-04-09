"""Analyze team assignment accuracy as bottleneck for serve side detection.

Investigates whether wrong team assignments come from side switch detection
errors (FP cascades), appearance matching drift, or per-rally player matching.

Usage:
    cd analysis
    uv run python scripts/analyze_team_accuracy.py
"""

from __future__ import annotations

import logging
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.table import Table

from eval_sequence_enriched import RallyBundle, prepare_rallies
from rallycut.evaluation.db import get_connection
from rallycut.tracking.match_tracker import build_match_team_assignments

logging.getLogger("rallycut").setLevel(logging.WARNING)

console = Console()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TeamRecord:
    rally_id: str
    video_id: str
    gt_player_tid: int
    gt_action: str
    position_team: int  # 0=near, 1=far from Y position
    assigned_team: int | None  # from match_teams
    team_correct: bool
    rally_index: int  # position in match order


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_raw_match_analysis(video_ids: set[str]) -> dict[str, dict]:
    """Load raw match_analysis_json per video from DB."""
    if not video_ids:
        return {}
    placeholders = ", ".join(["%s"] * len(video_ids))
    query = f"""
        SELECT id, match_analysis_json
        FROM videos
        WHERE id IN ({placeholders})
          AND match_analysis_json IS NOT NULL
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, list(video_ids))
            rows = cur.fetchall()
    return {
        str(row[0]): row[1]
        for row in rows
        if isinstance(row[1], dict)
    }


def get_position_team(
    bundle: RallyBundle, track_id: int, max_frames: int = 60,
) -> int | None:
    """Derive team from player Y position (independent of match_teams).

    Returns 0 (near/high-Y) or 1 (far/low-Y), or None if no data.
    """
    split_y = bundle.rally.court_split_y
    if split_y is None:
        return None

    start_frame = 0
    ys = [
        p.y
        for p in bundle.player_positions
        if p.track_id == track_id
        and start_frame <= p.frame_number < start_frame + max_frames
    ]
    if not ys:
        # Try all frames as fallback
        ys = [p.y for p in bundle.player_positions if p.track_id == track_id]
    if not ys:
        return None

    med_y = median(ys)
    return 0 if med_y > split_y else 1


# ---------------------------------------------------------------------------
# Build records
# ---------------------------------------------------------------------------


def build_records(bundles: list[RallyBundle], raw_ma: dict[str, dict]) -> list[TeamRecord]:
    """Build per-rally team accuracy records using position-based GT."""
    # Build rally_id -> rally_index mapping from match_analysis_json
    rally_index_map: dict[str, int] = {}
    for _vid, ma_json in raw_ma.items():
        for i, entry in enumerate(ma_json.get("rallies", [])):
            rid = entry.get("rallyId") or entry.get("rally_id", "")
            if rid:
                rally_index_map[rid] = i

    records: list[TeamRecord] = []
    for bundle in bundles:
        gt_serves = [gt for gt in bundle.gt_labels if gt.action == "serve"]
        if not gt_serves:
            continue
        gt_s = gt_serves[0]
        if gt_s.player_track_id < 0:
            continue

        pos_team = get_position_team(bundle, gt_s.player_track_id)
        if pos_team is None:
            continue

        assigned_team = None
        if bundle.match_teams:
            assigned_team = bundle.match_teams.get(gt_s.player_track_id)

        team_correct = assigned_team == pos_team if assigned_team is not None else False
        rally_idx = rally_index_map.get(bundle.rally.rally_id, -1)

        records.append(TeamRecord(
            rally_id=bundle.rally.rally_id,
            video_id=bundle.rally.video_id,
            gt_player_tid=gt_s.player_track_id,
            gt_action="serve",
            position_team=pos_team,
            assigned_team=assigned_team,
            team_correct=team_correct,
            rally_index=rally_idx,
        ))

    return records


# ---------------------------------------------------------------------------
# Analysis 1: Overall team accuracy
# ---------------------------------------------------------------------------


def analyze_overall(records: list[TeamRecord]) -> None:
    console.print("\n[bold]1. Overall Team Assignment Accuracy (Serve Players)[/bold]")

    has_assignment = [r for r in records if r.assigned_team is not None]
    no_assignment = [r for r in records if r.assigned_team is None]

    correct = sum(1 for r in has_assignment if r.team_correct)
    total = len(has_assignment)

    console.print(f"  Rallies with position-based GT: {len(records)}")
    console.print(f"  Rallies with team assignment:   {total}")
    console.print(f"  Rallies without team assignment: {len(no_assignment)}")
    if total:
        console.print(
            f"  [bold]Team accuracy: {correct}/{total} ({correct/total:.1%})[/bold]"
        )
        wrong = total - correct
        console.print(f"  Wrong team assignments: {wrong}")


# ---------------------------------------------------------------------------
# Analysis 2: Root cause classification
# ---------------------------------------------------------------------------


def analyze_root_causes(
    records: list[TeamRecord], raw_ma: dict[str, dict],
) -> None:
    console.print("\n[bold]2. Root Cause of Wrong Team Assignments[/bold]")

    errors = [r for r in records if not r.team_correct and r.assigned_team is not None]
    if not errors:
        console.print("  No team errors found.")
        return

    # Group errors by video
    errors_by_video: dict[str, list[TeamRecord]] = defaultdict(list)
    for r in errors:
        errors_by_video[r.video_id].append(r)

    # For each video with errors, check if removing switches would fix them
    causes: Counter[str] = Counter()
    detail_rows: list[tuple[str, str, str, int, int, int | None]] = []

    for vid, vid_errors in errors_by_video.items():
        ma_json = raw_ma.get(vid)
        if not ma_json:
            for r in vid_errors:
                causes["no_match_data"] += 1
            continue

        # Build teams WITHOUT any switches (all switches removed)
        no_switch_json = dict(ma_json)
        no_switch_rallies = []
        for entry in ma_json.get("rallies", []):
            clean = dict(entry)
            clean["sideSwitchDetected"] = False
            clean["side_switch_detected"] = False
            no_switch_rallies.append(clean)
        no_switch_json["rallies"] = no_switch_rallies

        teams_no_switch = build_match_team_assignments(no_switch_json, min_confidence=0.0)
        teams_with_switch = build_match_team_assignments(ma_json, min_confidence=0.0)

        # Count detected switches in this video
        n_switches = sum(
            1
            for entry in ma_json.get("rallies", [])
            if entry.get("sideSwitchDetected") or entry.get("side_switch_detected")
        )

        for r in vid_errors:
            no_sw_team = None
            if r.rally_id in teams_no_switch:
                no_sw_team = teams_no_switch[r.rally_id].get(r.gt_player_tid)

            with_sw_team = None
            if r.rally_id in teams_with_switch:
                with_sw_team = teams_with_switch[r.rally_id].get(r.gt_player_tid)

            if no_sw_team == r.position_team and with_sw_team != r.position_team:
                # Switches made it wrong -> switch FP cascade
                cause = "switch_fp_cascade"
            elif no_sw_team != r.position_team and with_sw_team != r.position_team:
                # Wrong with AND without switches -> could be switch FN or player matching
                # If no switches are detected but team is still wrong, it's a matching error
                # If switches are detected but still wrong, could be FN (need more switches)
                if n_switches == 0:
                    cause = "player_matching"
                else:
                    # Has switches but still wrong -> likely need different switches (FN or wrong placement)
                    cause = "switch_fn_or_misplaced"
            elif no_sw_team != r.position_team and with_sw_team == r.position_team:
                # Switches made it right but we're in the error list? Shouldn't happen
                # This means match_teams (which uses verified assignments) differs
                cause = "verification_divergence"
            else:
                cause = "player_matching"

            causes[cause] += 1
            detail_rows.append((
                r.rally_id[:8], r.video_id[:8], cause,
                r.position_team,
                r.assigned_team if r.assigned_team is not None else -1,
                no_sw_team,
            ))

    table = Table(title=f"Root Causes ({len(errors)} errors)")
    table.add_column("Cause")
    table.add_column("Count", justify="right")
    table.add_column("% of Errors", justify="right")
    for cause, count in causes.most_common():
        table.add_row(cause, str(count), f"{count/len(errors):.0%}")
    console.print(table)

    # Show detail for switch cascade errors
    cascade_details = [d for d in detail_rows if d[2] == "switch_fp_cascade"]
    if cascade_details:
        dt = Table(title="Switch FP Cascade Details")
        dt.add_column("Rally", style="dim")
        dt.add_column("Video", style="dim")
        dt.add_column("GT Team", justify="right")
        dt.add_column("Assigned", justify="right")
        dt.add_column("No-Switch", justify="right")
        for rally, vid, _cause, gt_t, assigned, no_sw in cascade_details[:20]:
            dt.add_row(
                rally, vid,
                str(gt_t), str(assigned),
                str(no_sw) if no_sw is not None else "-",
            )
        console.print(dt)


# ---------------------------------------------------------------------------
# Analysis 3: Per-video concentration
# ---------------------------------------------------------------------------


def analyze_per_video(
    records: list[TeamRecord], raw_ma: dict[str, dict],
) -> None:
    console.print("\n[bold]3. Per-Video Team Accuracy[/bold]")

    by_video: dict[str, list[TeamRecord]] = defaultdict(list)
    for r in records:
        if r.assigned_team is not None:
            by_video[r.video_id].append(r)

    if not by_video:
        console.print("  No data.")
        return

    # Count switches per video
    switch_counts: dict[str, int] = {}
    for vid, ma_json in raw_ma.items():
        switch_counts[vid] = sum(
            1
            for entry in ma_json.get("rallies", [])
            if entry.get("sideSwitchDetected") or entry.get("side_switch_detected")
        )

    table = Table(title="Team Accuracy by Video")
    table.add_column("Video", style="dim")
    table.add_column("Rallies", justify="right")
    table.add_column("Correct", justify="right")
    table.add_column("Wrong", justify="right")
    table.add_column("Accuracy", justify="right")
    table.add_column("Switches", justify="right")
    table.add_column("Max Run", justify="right")

    rows: list[tuple[str, int, int, int, float, int, int]] = []
    for vid, vid_records in by_video.items():
        total = len(vid_records)
        correct = sum(1 for r in vid_records if r.team_correct)
        wrong = total - correct
        acc = correct / total if total else 0
        n_sw = switch_counts.get(vid, 0)
        max_run = _longest_error_run(vid_records)
        rows.append((vid[:8], total, correct, wrong, acc, n_sw, max_run))

    # Sort by accuracy ascending (worst first)
    rows.sort(key=lambda x: x[4])

    for vid, total, correct, wrong, acc, n_sw, max_run in rows:
        style = "red" if acc < 0.8 else ("yellow" if acc < 0.9 else "")
        table.add_row(
            vid, str(total), str(correct), str(wrong),
            f"{acc:.0%}", str(n_sw), str(max_run),
            style=style,
        )

    console.print(table)

    # Summary stats
    all_accs = [r[4] for r in rows if r[1] > 0]
    if all_accs:
        console.print(f"  Videos with <80% accuracy: {sum(1 for a in all_accs if a < 0.8)}/{len(all_accs)}")
        console.print(f"  Videos with 100% accuracy: {sum(1 for a in all_accs if a == 1.0)}/{len(all_accs)}")


def _longest_error_run(records: list[TeamRecord]) -> int:
    """Find longest consecutive run of wrong team assignments."""
    sorted_recs = sorted(records, key=lambda r: r.rally_index)
    max_run = 0
    current_run = 0
    for r in sorted_recs:
        if not r.team_correct:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    return max_run


# ---------------------------------------------------------------------------
# Analysis 4: Consecutive error runs (cascade detection)
# ---------------------------------------------------------------------------


def analyze_cascades(records: list[TeamRecord]) -> None:
    console.print("\n[bold]4. Consecutive Error Runs (Cascade Detection)[/bold]")

    by_video: dict[str, list[TeamRecord]] = defaultdict(list)
    for r in records:
        if r.assigned_team is not None:
            by_video[r.video_id].append(r)

    cascades: list[tuple[str, int, int, int]] = []  # (video, start_idx, length, start_rally_index)

    for vid, vid_records in by_video.items():
        sorted_recs = sorted(vid_records, key=lambda r: r.rally_index)
        run_start = -1
        run_len = 0
        for i, r in enumerate(sorted_recs):
            if not r.team_correct:
                if run_len == 0:
                    run_start = i
                run_len += 1
            else:
                if run_len >= 2:
                    cascades.append((
                        vid[:8], run_start, run_len,
                        sorted_recs[run_start].rally_index,
                    ))
                run_len = 0
        if run_len >= 2:
            cascades.append((vid[:8], run_start, run_len, sorted_recs[run_start].rally_index))

    if not cascades:
        console.print("  No consecutive error runs of length >= 2 found.")
        return

    cascades.sort(key=lambda x: -x[2])

    table = Table(title=f"Error Cascades (runs >= 2, total: {len(cascades)})")
    table.add_column("Video", style="dim")
    table.add_column("Start Rally Idx", justify="right")
    table.add_column("Run Length", justify="right")
    for vid, _start, length, rally_idx in cascades:
        style = "red" if length >= 4 else "yellow"
        table.add_row(vid, str(rally_idx), str(length), style=style)
    console.print(table)

    total_in_cascades = sum(c[2] for c in cascades)
    total_errors = sum(1 for r in records if not r.team_correct and r.assigned_team is not None)
    console.print(f"  Errors in cascades: {total_in_cascades}/{total_errors} "
                  f"({total_in_cascades/total_errors:.0%})" if total_errors else "")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def analyze_verification_impact(
    bundles: list[RallyBundle], raw_ma: dict[str, dict],
) -> None:
    """Compare team accuracy with and without per-rally position verification."""
    console.print("\n[bold]5. Impact of Per-Rally Position Verification[/bold]")

    # Build rally_positions dict from bundles
    rally_positions: dict[str, list] = {}
    for b in bundles:
        if b.player_positions:
            rally_positions[b.rally.rally_id] = b.player_positions

    # Rebuild teams WITH verification for each video
    verified_teams: dict[str, dict[int, int]] = {}
    unverified_teams: dict[str, dict[int, int]] = {}
    for vid, ma_json in raw_ma.items():
        verified_teams.update(
            build_match_team_assignments(ma_json, min_confidence=0.70,
                                        rally_positions=rally_positions)
        )
        unverified_teams.update(
            build_match_team_assignments(ma_json, min_confidence=0.70)
        )

    # Compare on serve players
    rally_index_map: dict[str, int] = {}
    for _vid, ma_json in raw_ma.items():
        for i, entry in enumerate(ma_json.get("rallies", [])):
            rid = entry.get("rallyId") or entry.get("rally_id", "")
            if rid:
                rally_index_map[rid] = i

    n_unverified_correct = 0
    n_verified_correct = 0
    n_total = 0
    flipped = 0
    flip_helped = 0
    flip_hurt = 0

    by_video_delta: dict[str, tuple[int, int, int]] = {}  # vid -> (total, unv_correct, ver_correct)

    for b in bundles:
        gt_serves = [gt for gt in b.gt_labels if gt.action == "serve"]
        if not gt_serves:
            continue
        gt_s = gt_serves[0]
        if gt_s.player_track_id < 0:
            continue

        pos_team = get_position_team(b, gt_s.player_track_id)
        if pos_team is None:
            continue

        rid = b.rally.rally_id
        unv = unverified_teams.get(rid, {}).get(gt_s.player_track_id)
        ver = verified_teams.get(rid, {}).get(gt_s.player_track_id)

        if unv is None or ver is None:
            continue

        n_total += 1
        unv_ok = unv == pos_team
        ver_ok = ver == pos_team
        if unv_ok:
            n_unverified_correct += 1
        if ver_ok:
            n_verified_correct += 1

        if unv != ver:
            flipped += 1
            if ver_ok and not unv_ok:
                flip_helped += 1
            elif not ver_ok and unv_ok:
                flip_hurt += 1

        vid = b.rally.video_id
        if vid not in by_video_delta:
            by_video_delta[vid] = (0, 0, 0)
        t, uc, vc = by_video_delta[vid]
        by_video_delta[vid] = (t + 1, uc + (1 if unv_ok else 0), vc + (1 if ver_ok else 0))

    console.print(f"  Rallies compared: {n_total}")
    console.print(f"  [dim]Unverified:[/dim] {n_unverified_correct}/{n_total} "
                  f"({n_unverified_correct/n_total:.1%})")
    console.print(f"  [bold]Verified:  [/bold] {n_verified_correct}/{n_total} "
                  f"({n_verified_correct/n_total:.1%})")
    delta = n_verified_correct - n_unverified_correct
    if delta > 0:
        console.print(f"  [green]Delta: {delta:+d} rallies ({delta/n_total:+.1%})[/green]")
    elif delta < 0:
        console.print(f"  [red]Delta: {delta:+d} rallies ({delta/n_total:+.1%})[/red]")
    else:
        console.print(f"  Delta: 0 rallies")
    console.print(f"  Flipped: {flipped} rallies (helped {flip_helped}, hurt {flip_hurt})")

    # Per-video delta table (only show videos where it changed)
    changed = {
        vid: (t, uc, vc) for vid, (t, uc, vc) in by_video_delta.items()
        if uc != vc
    }
    if changed:
        table = Table(title="Per-Video Verification Impact")
        table.add_column("Video", style="dim")
        table.add_column("Rallies", justify="right")
        table.add_column("Before", justify="right")
        table.add_column("After", justify="right")
        table.add_column("Delta", justify="right")
        for vid in sorted(changed, key=lambda v: changed[v][2] - changed[v][1], reverse=True):
            t, uc, vc = changed[vid]
            d = vc - uc
            s = "green" if d > 0 else "red"
            table.add_row(
                vid[:8], str(t),
                f"{uc}/{t} ({uc/t:.0%})",
                f"{vc}/{t} ({vc/t:.0%})",
                f"[{s}]{d:+d}[/{s}]",
            )
        console.print(table)


def analyze_court_space_teams(
    bundles: list[RallyBundle], raw_ma: dict[str, dict],
) -> None:
    """Test per-rally court-space team assignment as a switch-free alternative."""
    from rallycut.court.calibration import CourtCalibrator
    from rallycut.evaluation.tracking.db import load_court_calibration

    console.print("\n[bold]6. Court-Space Per-Rally Team Assignment (Switch-Free)[/bold]")

    video_ids = {b.rally.video_id for b in bundles}
    calibrators: dict[str, CourtCalibrator] = {}
    for vid in video_ids:
        corners = load_court_calibration(vid)
        if corners and len(corners) == 4:
            cal = CourtCalibrator()
            cal.calibrate([(c["x"], c["y"]) for c in corners])
            if cal.is_calibrated:
                calibrators[vid] = cal

    console.print(f"  Calibrations: {len(calibrators)}/{len(video_ids)} videos")

    NET_Y = 8.0  # meters — net position in court space

    def compute_court_teams(
        bundle: RallyBundle, cal: CourtCalibrator,
    ) -> dict[int, int] | None:
        """Compute team for ALL tracks in a rally from court-space positions.

        Returns dict[track_id, team (0=near, 1=far)] or None if insufficient data.
        """
        # Project all tracks to court space
        all_court_ys: dict[int, list[float]] = defaultdict(list)
        for p in bundle.player_positions:
            try:
                _cx, cy = cal.image_to_court((p.x, p.y), 1, 1)
                # Filter extreme projections (calibration error)
                if -5 < cy < 25:
                    all_court_ys[p.track_id].append(cy)
            except Exception:
                pass

        if len(all_court_ys) < 2:
            return None

        # Cluster tracks by side of net
        track_med_cy: dict[int, float] = {}
        for tid, cys in all_court_ys.items():
            if cys:
                track_med_cy[tid] = median(cys)

        track_sides: dict[int, int] = {}
        for tid, med_cy in track_med_cy.items():
            track_sides[tid] = 0 if med_cy < NET_Y else 1

        # Need tracks on both sides
        sides = set(track_sides.values())
        if len(sides) < 2:
            return None

        # Determine which cluster is "near" (team 0) using image Y
        side0_img_ys: list[float] = []
        side1_img_ys: list[float] = []
        for p in bundle.player_positions:
            s = track_sides.get(p.track_id)
            if s == 0:
                side0_img_ys.append(p.y)
            elif s == 1:
                side1_img_ys.append(p.y)

        if not side0_img_ys or not side1_img_ys:
            return None

        avg_img_0 = sum(side0_img_ys) / len(side0_img_ys)
        avg_img_1 = sum(side1_img_ys) / len(side1_img_ys)

        # Higher image Y = near court = team 0
        near_is_side0 = avg_img_0 > avg_img_1

        teams: dict[int, int] = {}
        for tid, side in track_sides.items():
            if near_is_side0:
                teams[tid] = side  # 0=near, 1=far
            else:
                teams[tid] = 1 - side
        return teams

    n_total = 0
    n_court_correct = 0
    n_match_correct = 0
    n_has_court = 0
    n_no_court = 0
    n_no_server_in_court = 0

    by_video: dict[str, list[tuple[bool, bool, bool]]] = defaultdict(list)
    # Debug: track why court teams fail
    fail_reasons: Counter[str] = Counter()

    for b in bundles:
        gt_serves = [gt for gt in b.gt_labels if gt.action == "serve"]
        if not gt_serves:
            continue
        gt_s = gt_serves[0]
        if gt_s.player_track_id < 0:
            continue

        pos_team = get_position_team(b, gt_s.player_track_id)
        if pos_team is None:
            continue

        # Match-teams baseline
        match_team = None
        if b.match_teams:
            match_team = b.match_teams.get(gt_s.player_track_id)

        # Court-space team for ALL tracks
        cal = calibrators.get(b.rally.video_id)
        court_team = None
        if cal:
            court_teams = compute_court_teams(b, cal)
            if court_teams is not None:
                court_team = court_teams.get(gt_s.player_track_id)
                if court_team is None:
                    n_no_server_in_court += 1
                    fail_reasons["server_not_in_court_tracks"] += 1
            else:
                fail_reasons["insufficient_court_data"] += 1
        else:
            fail_reasons["no_calibration"] += 1

        n_total += 1
        has_court = court_team is not None
        court_ok = court_team == pos_team if has_court else False
        match_ok = match_team == pos_team if match_team is not None else False

        if has_court:
            n_has_court += 1
            if court_ok:
                n_court_correct += 1
        else:
            n_no_court += 1

        if match_team is not None and match_ok:
            n_match_correct += 1

        by_video[b.rally.video_id].append((has_court, court_ok, match_ok))

    console.print(f"  Rallies evaluated: {n_total}")
    console.print(f"  With court-space team: {n_has_court} ({n_has_court/n_total:.0%})")
    console.print(f"  Without court-space team: {n_no_court}")
    for reason, count in fail_reasons.most_common():
        console.print(f"    - {reason}: {count}")
    console.print()

    # Compare accuracy
    n_with_both = sum(1 for v in by_video.values() for h, _, _ in v if h)
    court_acc = n_court_correct / n_has_court if n_has_court else 0
    # Match accuracy only on rallies where both are available
    match_on_court_subset = sum(
        1 for v in by_video.values() for h, _, m in v if h and m
    )
    match_acc_overall = n_match_correct / n_total if n_total else 0

    console.print(f"  [bold]Court-space team accuracy: {n_court_correct}/{n_has_court} "
                  f"({court_acc:.1%})[/bold]")
    console.print(f"  Match-teams accuracy:    {n_match_correct}/{n_total} "
                  f"({match_acc_overall:.1%})")
    delta = n_court_correct - match_on_court_subset
    if n_has_court:
        console.print(f"  Delta on court subset:   {delta:+d} rallies")

    # Hybrid: use court-space when available, match-teams otherwise
    n_hybrid_correct = 0
    for vid_results in by_video.values():
        for has_court_val, court_ok_val, match_ok_val in vid_results:
            if has_court_val:
                if court_ok_val:
                    n_hybrid_correct += 1
            else:
                if match_ok_val:
                    n_hybrid_correct += 1
    console.print(f"  [bold green]Hybrid (court||match): {n_hybrid_correct}/{n_total} "
                  f"({n_hybrid_correct/n_total:.1%})[/bold green]")

    # Per-video comparison (only where court-space changed outcome)
    table = Table(title="Per-Video: Court-Space vs Match-Teams")
    table.add_column("Video", style="dim")
    table.add_column("Rallies", justify="right")
    table.add_column("Court", justify="right")
    table.add_column("Match", justify="right")
    table.add_column("Delta", justify="right")

    vid_rows: list[tuple[str, int, int, int, int]] = []
    for vid, results in by_video.items():
        court_results = [(c, m) for h, c, m in results if h]
        if not court_results:
            continue
        n = len(court_results)
        nc = sum(1 for c, _ in court_results if c)
        nm = sum(1 for _, m in court_results if m)
        if nc != nm:
            vid_rows.append((vid[:8], n, nc, nm, nc - nm))

    vid_rows.sort(key=lambda x: x[4], reverse=True)
    for vid, n, nc, nm, d in vid_rows:
        s = "green" if d > 0 else "red"
        table.add_row(
            vid, str(n),
            f"{nc}/{n} ({nc/n:.0%})",
            f"{nm}/{n} ({nm/n:.0%})",
            f"[{s}]{d:+d}[/{s}]",
        )
    if vid_rows:
        console.print(table)


def analyze_player_count_asymmetry(bundles: list[RallyBundle]) -> None:
    """Test player-count asymmetry as a serve side signal.

    At rally start, if one side has fewer visible players than the other,
    the missing player is likely the server (off-screen at baseline).
    1 near : 2 far → near-side serve (near server off-screen)
    2 near : 1 far → far-side serve (or far player occluded — less reliable)
    """
    console.print("\n[bold]7. Player Count Asymmetry at Rally Start[/bold]")

    WINDOW = 45  # frames to consider at rally start

    @dataclass
    class AsymRecord:
        rally_id: str
        video_id: str
        n_near: int
        n_far: int
        gt_serve_side: str  # "near"/"far"
        asymmetry_pred: str  # "near"/"far"/""
        server_tracked: bool  # is GT server in the tracked players?

    records: list[AsymRecord] = []

    for b in bundles:
        gt_serves = [gt for gt in b.gt_labels if gt.action == "serve"]
        if not gt_serves:
            continue
        gt_s = gt_serves[0]
        if gt_s.player_track_id < 0:
            continue

        pos_team = get_position_team(b, gt_s.player_track_id)
        if pos_team is None:
            continue
        gt_side = "near" if pos_team == 0 else "far"

        split_y = b.rally.court_split_y
        if split_y is None:
            continue

        # Count unique tracks per side in first WINDOW frames
        near_tids: set[int] = set()
        far_tids: set[int] = set()
        server_tracked = False

        for p in b.player_positions:
            if p.frame_number >= WINDOW:
                break
            if p.track_id == gt_s.player_track_id:
                server_tracked = True
            if p.y > split_y:
                near_tids.add(p.track_id)
            else:
                far_tids.add(p.track_id)

        n_near = len(near_tids)
        n_far = len(far_tids)

        # Asymmetry prediction: fewer players on one side → serve from that side
        pred = ""
        if n_near < n_far and n_near <= 1:
            pred = "near"
        elif n_far < n_near and n_far <= 1:
            pred = "far"

        records.append(AsymRecord(
            rally_id=b.rally.rally_id,
            video_id=b.rally.video_id,
            n_near=n_near, n_far=n_far,
            gt_serve_side=gt_side,
            asymmetry_pred=pred,
            server_tracked=server_tracked,
        ))

    console.print(f"  Rallies evaluated: {len(records)}")

    # Distribution of player counts
    count_dist: Counter[str] = Counter()
    for r in records:
        count_dist[f"{r.n_near}n:{r.n_far}f"] += 1
    console.print("  Player count distribution (near:far):")
    for config in sorted(count_dist.keys()):
        console.print(f"    {config}: {count_dist[config]}")

    # Asymmetry signal accuracy
    has_pred = [r for r in records if r.asymmetry_pred]
    no_pred = [r for r in records if not r.asymmetry_pred]
    correct_pred = [r for r in has_pred if r.asymmetry_pred == r.gt_serve_side]

    console.print(f"\n  Asymmetry signal fires: {len(has_pred)}/{len(records)} "
                  f"({len(has_pred)/len(records):.0%})")
    if has_pred:
        console.print(f"  Asymmetry accuracy: {len(correct_pred)}/{len(has_pred)} "
                      f"({len(correct_pred)/len(has_pred):.1%})")

    # Break down by whether server is tracked (the hard cases are untracked)
    untracked = [r for r in records if not r.server_tracked]
    untracked_has_pred = [r for r in untracked if r.asymmetry_pred]
    untracked_correct = [r for r in untracked_has_pred
                         if r.asymmetry_pred == r.gt_serve_side]

    console.print(f"\n  Server NOT tracked (hard cases): {len(untracked)}")
    if untracked_has_pred:
        console.print(f"    Asymmetry fires: {len(untracked_has_pred)}/{len(untracked)} "
                      f"({len(untracked_has_pred)/len(untracked):.0%})")
        console.print(f"    Asymmetry accuracy: {len(untracked_correct)}/{len(untracked_has_pred)} "
                      f"({len(untracked_correct)/len(untracked_has_pred):.1%})")

    # Detail table: asymmetry predictions
    if has_pred:
        table = Table(title="Asymmetry Signal Detail")
        table.add_column("Config")
        table.add_column("Count", justify="right")
        table.add_column("GT Near", justify="right")
        table.add_column("GT Far", justify="right")
        table.add_column("Pred", justify="right")
        table.add_column("Accuracy", justify="right")

        for config in sorted(set(f"{r.n_near}n:{r.n_far}f" for r in has_pred)):
            subset = [r for r in has_pred if f"{r.n_near}n:{r.n_far}f" == config]
            n_gt_near = sum(1 for r in subset if r.gt_serve_side == "near")
            n_gt_far = sum(1 for r in subset if r.gt_serve_side == "far")
            pred_side = subset[0].asymmetry_pred
            correct = sum(1 for r in subset if r.asymmetry_pred == r.gt_serve_side)
            table.add_row(
                config, str(len(subset)),
                str(n_gt_near), str(n_gt_far),
                pred_side, f"{correct}/{len(subset)} ({correct/len(subset):.0%})",
            )
        console.print(table)


def analyze_y_distance_gating(bundles: list[RallyBundle]) -> None:
    """Test Y-distance gating: trust server position over team when clearly on one side.

    When the predicted server's Y is far from court_split_y, their position
    directly indicates their team. Only trust team overwrite when the server
    is near the net (ambiguous position).
    """
    console.print("\n[bold]8. Server Y-Distance Gating for Team Overwrite[/bold]")

    @dataclass
    class GateRecord:
        rally_id: str
        video_id: str
        gt_side: str
        server_tid: int
        server_y_dist: float  # absolute distance from court_split_y
        pos_side: str  # side from Y position
        team_side: str  # side from match_teams
        final_side_baseline: str  # team overwrite (current behavior)
        agree: bool

    records: list[GateRecord] = []

    for b in bundles:
        gt_serves = [gt for gt in b.gt_labels if gt.action == "serve"]
        if not gt_serves:
            continue
        gt_s = gt_serves[0]
        if gt_s.player_track_id < 0:
            continue

        pos_team = get_position_team(b, gt_s.player_track_id)
        if pos_team is None:
            continue
        gt_side = "near" if pos_team == 0 else "far"

        split_y = b.rally.court_split_y
        if split_y is None or not b.match_teams:
            continue

        match_team = b.match_teams.get(gt_s.player_track_id)
        if match_team is None:
            continue

        # Server's median Y position (early frames)
        ys = [p.y for p in b.player_positions
              if p.track_id == gt_s.player_track_id and p.frame_number < 60]
        if not ys:
            ys = [p.y for p in b.player_positions
                  if p.track_id == gt_s.player_track_id]
        if not ys:
            continue

        med_y = median(ys)
        y_dist = abs(med_y - split_y)
        pos_side = "near" if med_y > split_y else "far"
        team_side = "near" if match_team == 0 else "far"

        records.append(GateRecord(
            rally_id=b.rally.rally_id,
            video_id=b.rally.video_id,
            gt_side=gt_side,
            server_tid=gt_s.player_track_id,
            server_y_dist=y_dist,
            pos_side=pos_side,
            team_side=team_side,
            final_side_baseline=team_side,  # current: always trust team
            agree=pos_side == team_side,
        ))

    console.print(f"  Rallies with server tracked + team assignment: {len(records)}")

    # How often do position and team agree?
    agree = [r for r in records if r.agree]
    disagree = [r for r in records if not r.agree]
    console.print(f"  Position == Team: {len(agree)} ({len(agree)/len(records):.0%})")
    console.print(f"  Position != Team: {len(disagree)} ({len(disagree)/len(records):.0%})")

    if disagree:
        pos_right = sum(1 for r in disagree if r.pos_side == r.gt_side)
        team_right = sum(1 for r in disagree if r.team_side == r.gt_side)
        console.print(f"    When they disagree: position right {pos_right}, "
                      f"team right {team_right}")

    # Sweep Y-distance thresholds
    # For each threshold: if server Y-distance > threshold, trust position; else trust team
    thresholds = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 1.0]

    # Baseline: always trust team
    baseline_correct = sum(1 for r in records if r.team_side == r.gt_side)

    table = Table(title="Y-Distance Gating Sweep")
    table.add_column("Threshold")
    table.add_column("Gated (trust pos)", justify="right")
    table.add_column("Correct", justify="right")
    table.add_column("Accuracy", justify="right")
    table.add_column("Δ vs team-always", justify="right")

    for thresh in thresholds:
        correct = 0
        n_gated = 0
        for r in records:
            if r.server_y_dist > thresh and not r.agree:
                # Trust position over team
                n_gated += 1
                if r.pos_side == r.gt_side:
                    correct += 1
            else:
                # Trust team (default)
                if r.team_side == r.gt_side:
                    correct += 1

        delta = correct - baseline_correct
        label = f"≥{thresh:.2f}" if thresh < 1.0 else "never gate (baseline)"
        if delta > 0:
            delta_str = f"[green]{delta:+d}[/green]"
        elif delta < 0:
            delta_str = f"[red]{delta:+d}[/red]"
        else:
            delta_str = "0"
        table.add_row(
            label, str(n_gated), str(correct),
            f"{correct/len(records):.1%}",
            delta_str,
        )

    console.print(table)

    # Distribution of Y-distances for agree vs disagree
    if disagree:
        console.print("\n  Y-distance distribution (disagree cases):")
        pos_right_dists = [r.server_y_dist for r in disagree if r.pos_side == r.gt_side]
        team_right_dists = [r.server_y_dist for r in disagree if r.team_side == r.gt_side]
        if pos_right_dists:
            console.print(f"    Position right ({len(pos_right_dists)}): "
                          f"min={min(pos_right_dists):.3f} "
                          f"median={median(pos_right_dists):.3f} "
                          f"max={max(pos_right_dists):.3f}")
        if team_right_dists:
            console.print(f"    Team right ({len(team_right_dists)}): "
                          f"min={min(team_right_dists):.3f} "
                          f"median={median(team_right_dists):.3f} "
                          f"max={max(team_right_dists):.3f}")


def main() -> None:
    console.print("[bold]Team Assignment Accuracy Analysis[/bold]")
    bundles = prepare_rallies(label_spread=2)
    if not bundles:
        console.print("[red]No rallies loaded.[/red]")
        return

    video_ids = {b.rally.video_id for b in bundles}
    console.print(f"  {len(bundles)} rallies, {len(video_ids)} videos")

    raw_ma = load_raw_match_analysis(video_ids)
    console.print(f"  {len(raw_ma)} videos with match_analysis_json")

    records = build_records(bundles, raw_ma)
    console.print(f"  {len(records)} rallies with position-based GT team")

    analyze_overall(records)
    analyze_root_causes(records, raw_ma)
    analyze_per_video(records, raw_ma)
    analyze_cascades(records)
    analyze_verification_impact(bundles, raw_ma)
    analyze_court_space_teams(bundles, raw_ma)
    analyze_player_count_asymmetry(bundles)
    analyze_y_distance_gating(bundles)


if __name__ == "__main__":
    main()
