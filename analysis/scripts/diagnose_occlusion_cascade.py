"""Diagnose track ID cascade from occlusion in IMG_2313 rally #10.

At 04:50.5 player 2 gets occluded, loses tracking, then returns at 04:52.4
and steals track 3, cascading ID swaps across 3 players.

This script re-tracks the rally segment, captures raw BoT-SORT output vs
final pipeline output, and prints per-frame track assignments at the critical
timestamps to show exactly where the cascade starts and what post-processing
does/doesn't fix.

Usage:
    cd analysis
    uv run python scripts/diagnose_occlusion_cascade.py
    uv run python scripts/diagnose_occlusion_cascade.py --debug-video occlusion_debug.mp4
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from rallycut.tracking.player_filter import (
    PlayerFilter,
    PlayerFilterConfig,
    classify_teams,
    compute_court_split,
    interpolate_player_gaps,
    remove_stationary_background_tracks,
    stabilize_track_ids,
)
from rallycut.tracking.player_tracker import (
    PlayerPosition,
    PlayerTracker,
)

console = Console()

# Rally segment from render_boxmot_comparison.py (IMG2313_rally9_drop_0444)
VIDEO_PATH = Path("training_datasets/beach_v8/videos/IMG_2313.MOV")
START_MS = 277317
END_MS = 296520

# Key timestamps (absolute video seconds) and their rally-relative frame numbers.
# Frame numbers are approximate — the script auto-discovers exact FPS.
KEY_EVENTS = [
    (290.5, "player 2 last seen (before occlusion)"),
    (291.5, "mid-occlusion (player 2 gone)"),
    (292.4, "player 2 returns — takes track 3"),
    (292.6, "cascade: player 3 → track 4"),
    (294.5, "stabilization: original player 2 → track 3"),
]


def _ts_to_rally_frame(abs_seconds: float, fps: float) -> int:
    """Convert absolute video timestamp to rally-relative frame number."""
    rally_start_s = START_MS / 1000.0
    return round((abs_seconds - rally_start_s) * fps)


def _get_frame_positions(
    positions: list[PlayerPosition], frame: int
) -> list[PlayerPosition]:
    """Get all positions at a given frame, sorted by track_id."""
    return sorted(
        [p for p in positions if p.frame_number == frame],
        key=lambda p: p.track_id,
    )


def _track_existence_summary(
    positions: list[PlayerPosition],
    frame_start: int,
    frame_end: int,
) -> dict[int, tuple[int, int]]:
    """For each track, find first and last frame within the range."""
    tracks: dict[int, tuple[int, int]] = {}
    for p in positions:
        if p.track_id < 0:
            continue
        if frame_start <= p.frame_number <= frame_end:
            if p.track_id not in tracks:
                tracks[p.track_id] = (p.frame_number, p.frame_number)
            else:
                first, last = tracks[p.track_id]
                tracks[p.track_id] = (
                    min(first, p.frame_number),
                    max(last, p.frame_number),
                )
    return tracks


def _print_frame_table(
    label: str,
    positions: list[PlayerPosition],
    key_frames: list[tuple[int, str]],
) -> None:
    """Print a table of track assignments at key frames."""
    table = Table(title=label, show_lines=True)
    table.add_column("Frame", style="cyan", width=8)
    table.add_column("Event", style="yellow", width=42)
    table.add_column("Tracks (id @ x,y [conf])", style="white", width=60)

    for frame, event in key_frames:
        frame_pos = _get_frame_positions(positions, frame)
        if not frame_pos:
            # Try ±1 frame (FPS rounding)
            for offset in [1, -1, 2, -2]:
                frame_pos = _get_frame_positions(positions, frame + offset)
                if frame_pos:
                    break

        tracks_str = "  ".join(
            f"[bold]T{p.track_id}[/bold]@({p.x:.3f},{p.y:.3f})[{p.confidence:.2f}]"
            for p in frame_pos
        ) if frame_pos else "[red]no detections[/red]"

        table.add_row(str(frame), event, tracks_str)

    console.print(table)


def _print_occlusion_gap(
    raw_positions: list[PlayerPosition],
    gap_start: int,
    gap_end: int,
) -> None:
    """Print track existence during the occlusion gap."""
    existence = _track_existence_summary(raw_positions, gap_start, gap_end)

    console.print(f"\n[bold]Track existence during occlusion gap "
                  f"(frames {gap_start}–{gap_end}):[/bold]")
    for tid in sorted(existence):
        first, last = existence[tid]
        console.print(f"  Track {tid}: frames {first}–{last}")

    # Which tracks disappear?
    pre_tracks = {
        p.track_id for p in raw_positions
        if p.frame_number == gap_start - 1 and p.track_id >= 0
    }
    post_tracks = {
        p.track_id for p in raw_positions
        if p.frame_number == gap_end + 1 and p.track_id >= 0
    }
    disappeared = pre_tracks - {
        tid for tid, (f, l) in existence.items() if l >= gap_end - 5
    }
    appeared = post_tracks - pre_tracks
    if disappeared:
        console.print(f"  [red]Disappeared: {sorted(disappeared)}[/red]")
    if appeared:
        console.print(f"  [green]New tracks after gap: {sorted(appeared)}[/green]")


def _run_pipeline_stages(
    raw_positions: list[PlayerPosition],
    color_store,
    appearance_store,
    ball_positions,
    total_frames: int,
    key_frames: list[tuple[int, str]],
) -> list[PlayerPosition]:
    """Run post-processing stages one by one, printing state after each."""
    from rallycut.tracking.color_repair import split_tracks_by_color
    from rallycut.tracking.convergence_swap import detect_convergence_swaps
    from rallycut.tracking.global_identity import optimize_global_identity
    from rallycut.tracking.height_consistency import fix_height_swaps
    from rallycut.tracking.spatial_consistency import enforce_spatial_consistency
    from rallycut.tracking.tracklet_link import link_tracklets_by_appearance

    config = PlayerFilterConfig()
    positions = copy.deepcopy(raw_positions)
    cs = copy.deepcopy(color_store) if color_store else None
    aps = copy.deepcopy(appearance_store) if appearance_store else None

    stages = []

    # Stage 1: Stationary removal
    positions, removed = remove_stationary_background_tracks(
        positions, config, total_frames=total_frames
    )
    console.print(f"\n[dim]Stationary removal: {len(removed)} tracks removed[/dim]")
    stages.append(("After stationary removal", copy.deepcopy(positions)))

    # Stage 2: Spatial consistency
    positions, sc_result = enforce_spatial_consistency(
        positions, color_store=cs, appearance_store=aps
    )
    console.print(f"[dim]Spatial consistency: {sc_result.jump_splits} splits[/dim]")
    stages.append(("After spatial consistency", copy.deepcopy(positions)))

    # Stage 3: Height swaps
    positions, hs_result = fix_height_swaps(
        positions, color_store=cs, appearance_store=aps
    )
    console.print(f"[dim]Height swaps: {hs_result.swaps} swaps[/dim]")
    stages.append(("After height swaps", copy.deepcopy(positions)))

    # Stage 4: Color split
    num_color_splits = 0
    if cs is not None and cs.has_data():
        positions, num_color_splits = split_tracks_by_color(positions, cs)
    console.print(f"[dim]Color split: {num_color_splits} splits[/dim]")
    stages.append(("After color split", copy.deepcopy(positions)))

    # Stage 5: Tracklet linking
    num_links = 0
    if cs is not None and cs.has_data():
        positions, num_links = link_tracklets_by_appearance(
            positions, cs, appearance_store=aps
        )
    console.print(f"[dim]Tracklet linking: {num_links} merges[/dim]")
    stages.append(("After tracklet linking", copy.deepcopy(positions)))

    # Stage 6: Stabilize track IDs
    positions, id_mapping = stabilize_track_ids(positions, config)
    if id_mapping and cs is not None:
        cs.remap_ids(id_mapping)
    if id_mapping and aps is not None:
        aps.remap_ids(id_mapping)
    console.print(f"[dim]Stabilize: {len(id_mapping)} merges[/dim]")
    stages.append(("After stabilize", copy.deepcopy(positions)))

    # Stage 7: Per-frame filtering
    player_filter = PlayerFilter(
        ball_positions=ball_positions,
        total_frames=total_frames,
        config=config,
    )
    player_filter.analyze_tracks(positions)
    primary_tids = sorted(player_filter.primary_tracks)
    console.print(f"[dim]Primary tracks: {primary_tids}[/dim]")

    frames: dict[int, list[PlayerPosition]] = {}
    for p in positions:
        frames.setdefault(p.frame_number, []).append(p)
    filtered: list[PlayerPosition] = []
    for fn in sorted(frames):
        filtered.extend(player_filter.filter(frames[fn]))
    positions = filtered
    stages.append(("After per-frame filter", copy.deepcopy(positions)))

    # Stage 8: Team classification + global identity
    split_result = compute_court_split(
        ball_positions or [], config, player_positions=positions
    )
    split_y = split_result[0] if split_result else None
    split_conf = split_result[1] if split_result else None
    precomputed = split_result[2] if split_result else None
    team_assignments: dict[int, int] = {}

    if split_y is not None and split_conf == "high":
        team_assignments = classify_teams(
            positions, split_y, precomputed_assignments=precomputed
        )
    elif precomputed and len(set(precomputed.values())) >= 2:
        team_assignments = dict(precomputed)

    if team_assignments:
        console.print(f"[dim]Teams: {team_assignments}[/dim]")

    if cs is not None and cs.has_data() and team_assignments:
        positions, global_result = optimize_global_identity(
            positions, team_assignments, cs,
            court_split_y=split_y, appearance_store=aps,
        )
        console.print(
            f"[dim]Global identity: {global_result.num_segments} segments, "
            f"{global_result.num_remapped} remapped, "
            f"skipped={global_result.skipped}[/dim]"
        )
    stages.append(("After global identity", copy.deepcopy(positions)))

    # Stage 9: Convergence swap
    if len(primary_tids) >= 4:
        positions, n_swaps = detect_convergence_swaps(
            positions, primary_tids,
            color_store=cs, upstream_split_y=split_y,
            upstream_teams=team_assignments,
        )
        console.print(f"[dim]Convergence swaps: {n_swaps}[/dim]")
    stages.append(("After convergence swap", copy.deepcopy(positions)))

    # Print tables for stages that changed something
    # Always show raw, first change, and final
    console.print("\n")
    for label, stage_positions in stages:
        _print_frame_table(label, stage_positions, key_frames)

    return positions


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose occlusion cascade in IMG_2313")
    parser.add_argument("--debug-video", type=str, help="Render debug video to this path")
    parser.add_argument("--stages", action="store_true",
                        help="Run each pipeline stage individually and show intermediate state")
    args = parser.parse_args()

    if not VIDEO_PATH.exists():
        console.print(f"[red]Video not found: {VIDEO_PATH}[/red]")
        sys.exit(1)

    # Enable logging for tracking modules
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(name)s %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )
    # Reduce noise from non-tracking modules
    for name in ["urllib3", "PIL", "ultralytics", "boxmot"]:
        logging.getLogger(name).setLevel(logging.WARNING)

    console.print(f"[bold]Diagnosing occlusion cascade in IMG_2313[/bold]")
    console.print(f"Rally segment: {START_MS}ms – {END_MS}ms")
    console.print(f"Video: {VIDEO_PATH}\n")

    # Track the rally segment
    console.print("[bold]Step 1: Tracking rally segment...[/bold]")
    tracker = PlayerTracker()
    result = tracker.track_video(
        str(VIDEO_PATH),
        start_ms=START_MS,
        end_ms=END_MS,
        filter_enabled=True,
    )

    fps = result.video_fps
    console.print(f"FPS: {fps}, frames: {result.frame_count}")
    console.print(f"Primary tracks: {result.primary_track_ids}")
    console.print(f"Team assignments: {result.team_assignments}")

    # Compute rally-relative key frames
    key_frames = [
        (_ts_to_rally_frame(ts, fps), event)
        for ts, event in KEY_EVENTS
    ]
    console.print(f"\n[bold]Key frames (0-based within rally):[/bold]")
    for frame, event in key_frames:
        console.print(f"  Frame {frame}: {event}")

    # Occlusion gap bounds
    gap_start = _ts_to_rally_frame(290.5, fps)
    gap_end = _ts_to_rally_frame(292.4, fps)

    # Print raw BoT-SORT output
    console.print("\n[bold]Step 2: Raw BoT-SORT output[/bold]")
    raw = result.raw_positions
    if not raw:
        console.print("[red]No raw positions available![/red]")
        sys.exit(1)

    _print_frame_table("Raw BoT-SORT (before post-processing)", raw, key_frames)
    _print_occlusion_gap(raw, gap_start, gap_end)

    # Print final pipeline output
    console.print("\n[bold]Step 3: Final pipeline output[/bold]")
    _print_frame_table("Final (after all post-processing)", result.positions, key_frames)

    # Track count over time around the event
    console.print("\n[bold]Step 4: Track count around occlusion[/bold]")
    table = Table(title="Active tracks per frame (raw)", show_lines=False)
    table.add_column("Frame", style="cyan", width=8)
    table.add_column("Time", style="dim", width=10)
    table.add_column("Track IDs", style="white")

    sample_frames = list(range(gap_start - 5, gap_end + 15, 3))
    for f in sample_frames:
        abs_time = START_MS / 1000.0 + f / fps
        frame_pos = [p for p in raw if p.frame_number == f and p.track_id >= 0]
        tids = sorted(set(p.track_id for p in frame_pos))
        mm = int(abs_time // 60)
        ss = abs_time % 60
        table.add_row(str(f), f"{mm:02d}:{ss:05.2f}", str(tids))
    console.print(table)

    # Track position trajectories around the event
    console.print("\n[bold]Step 5: Track position trajectories (raw)[/bold]")
    # For each track, show x,y at sampled frames around the event
    event_range = range(gap_start - 10, gap_end + 60, 5)
    track_ids_of_interest = sorted({
        p.track_id for p in raw
        if gap_start - 10 <= p.frame_number <= gap_end + 60 and p.track_id >= 0
    })
    # Exclude stationary T5
    track_ids_of_interest = [t for t in track_ids_of_interest if t != 5]

    traj_table = Table(title="Position trajectories (raw, non-stationary tracks)")
    traj_table.add_column("Frame", style="cyan", width=6)
    traj_table.add_column("Time", style="dim", width=10)
    for tid in track_ids_of_interest:
        traj_table.add_column(f"T{tid}", width=16)

    for f in event_range:
        abs_time = START_MS / 1000.0 + f / fps
        mm = int(abs_time // 60)
        ss = abs_time % 60
        row = [str(f), f"{mm:02d}:{ss:05.2f}"]
        for tid in track_ids_of_interest:
            pos = [p for p in raw if p.frame_number == f and p.track_id == tid]
            if pos:
                p = pos[0]
                row.append(f"({p.x:.3f},{p.y:.3f})")
            else:
                row.append("[dim]—[/dim]")
        traj_table.add_row(*row)

    console.print(traj_table)

    # Show the tracklet linking merges from log
    console.print("\n[bold]Step 6: Tracklet linking merges (from pipeline):[/bold]")
    unique_raw = sorted({p.track_id for p in raw if p.track_id >= 0})
    unique_final = sorted({p.track_id for p in result.positions if p.track_id >= 0})
    console.print(f"  Raw track IDs: {unique_raw}")
    console.print(f"  Final track IDs: {unique_final}")

    # Map raw → final by finding which final track_id each raw track_id
    # position ends up closest to
    console.print("\n[bold]Raw → Final ID mapping (by position overlap):[/bold]")
    for raw_tid in unique_raw:
        raw_frames = {p.frame_number for p in raw if p.track_id == raw_tid}
        if len(raw_frames) < 3:
            continue
        # Sample some frames
        sample = sorted(raw_frames)[:10]
        matches: dict[int, int] = {}
        for f in sample:
            raw_p = [p for p in raw if p.frame_number == f and p.track_id == raw_tid]
            if not raw_p:
                continue
            rp = raw_p[0]
            # Find nearest final track at same frame
            final_p = [p for p in result.positions if p.frame_number == f and p.track_id >= 0]
            if not final_p:
                continue
            nearest = min(final_p, key=lambda p: (p.x - rp.x)**2 + (p.y - rp.y)**2)
            dist = ((nearest.x - rp.x)**2 + (nearest.y - rp.y)**2)**0.5
            if dist < 0.05:
                matches[nearest.track_id] = matches.get(nearest.track_id, 0) + 1
        if matches:
            best = max(matches, key=lambda k: matches[k])
            console.print(f"  Raw T{raw_tid} ({len(raw_frames)} frames) → Final T{best} ({matches[best]}/{len(sample)} frames matched)")

    # Per-stage analysis
    if args.stages:
        console.print("\n[bold]Step 5: Per-stage pipeline analysis[/bold]")
        # Re-run stages on raw positions with deepcopy of stores
        _run_pipeline_stages(
            raw,
            result.color_store,
            result.appearance_store,
            getattr(result, "ball_positions", None) or [],
            result.frame_count,
            key_frames,
        )

    # Debug video
    if args.debug_video:
        console.print(f"\n[bold]Rendering debug video to {args.debug_video}...[/bold]")
        _render_debug_video(
            str(VIDEO_PATH), result, key_frames, args.debug_video, fps
        )


def _render_debug_video(
    video_path: str,
    result,
    key_frames: list[tuple[int, str]],
    output_path: str,
    fps: float,
) -> None:
    """Render a debug video with bboxes and track IDs overlaid."""
    import cv2
    import numpy as np

    COLORS = {
        1: (0, 255, 0),
        2: (255, 100, 0),
        3: (0, 100, 255),
        4: (255, 255, 0),
        5: (0, 255, 255),
        6: (255, 0, 255),
    }

    def get_color(tid: int) -> tuple[int, int, int]:
        if tid in COLORS:
            return COLORS[tid]
        np.random.seed(tid * 31)
        return tuple(int(x) for x in np.random.randint(80, 255, 3))

    cap = cv2.VideoCapture(video_path)
    start_frame = round(START_MS / 1000.0 * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # Build frame lookup for raw + final
    raw_by_frame: dict[int, list[PlayerPosition]] = {}
    for p in result.raw_positions:
        raw_by_frame.setdefault(p.frame_number, []).append(p)

    final_by_frame: dict[int, list[PlayerPosition]] = {}
    for p in result.positions:
        final_by_frame.setdefault(p.frame_number, []).append(p)

    key_frame_set = {f for f, _ in key_frames}
    key_frame_labels = {f: e for f, e in key_frames}

    total_frames = result.frame_count
    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        abs_time = START_MS / 1000.0 + frame_idx / fps
        mm = int(abs_time // 60)
        ss = abs_time % 60

        # Draw final positions (filled bbox)
        for p in final_by_frame.get(frame_idx, []):
            if p.track_id < 0:
                continue
            color = get_color(p.track_id)
            # Convert normalized coords to pixel (bbox center + assumed size)
            cx, cy = int(p.x * w), int(p.y * h)
            bw = int(p.width * w) if p.width else 40
            bh = int(p.height * h) if p.height else 80
            x1, y1 = cx - bw // 2, cy - bh // 2
            x2, y2 = cx + bw // 2, cy + bh // 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame, f"T{p.track_id}",
                (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
            )

        # Timestamp overlay
        cv2.putText(
            frame, f"F{frame_idx} {mm:02d}:{ss:05.2f}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )

        # Highlight key frames
        if frame_idx in key_frame_set:
            label = key_frame_labels[frame_idx]
            cv2.putText(
                frame, f">>> {label}",
                (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )

        out.write(frame)

    cap.release()
    out.release()
    console.print(f"[green]Debug video saved to {output_path}[/green]")


if __name__ == "__main__":
    main()
