"""Measure court_side detection accuracy with different signals.

Compares ball-position-based court_side determination methods against
ground truth (GT player's team = correct court_side).

Signals tested:
1. Ball Y vs net_y (current baseline — simple threshold)
2. Ball trajectory direction (post-contact ball movement direction)
3. Pre-contact trajectory direction (where ball came from)
4. Combined trajectory + position
5. Player Y position of nearest candidate

Usage:
    cd analysis
    uv run python scripts/eval_court_side.py
"""

from __future__ import annotations

import argparse
from typing import Any

from rich.console import Console
from rich.table import Table

from rallycut.evaluation.db import get_connection
from rallycut.tracking.ball_tracker import BallPosition
from rallycut.tracking.contact_detector import detect_contacts
from rallycut.tracking.player_tracker import PlayerPosition

console = Console()


def _load_data() -> list[dict[str, Any]]:
    """Load rallies with action GT, ball positions, and match teams."""
    query = """
        SELECT
            r.id as rally_id,
            r.video_id,
            pt.action_ground_truth_json,
            pt.ball_positions_json,
            pt.positions_json,
            pt.fps,
            pt.court_split_y,
            v.match_analysis_json
        FROM rallies r
        JOIN player_tracks pt ON pt.rally_id = r.id
        JOIN videos v ON v.id = r.video_id
        WHERE pt.action_ground_truth_json IS NOT NULL
          AND pt.ball_positions_json IS NOT NULL
        ORDER BY r.video_id, r.start_ms
    """
    results: list[dict[str, Any]] = []
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            for row in cur.fetchall():
                (rally_id, video_id, gt_json, ball_json, pos_json,
                 fps, court_split_y, ma_json) = row

                if not gt_json or not ball_json:
                    continue

                # Get team assignments from match analysis
                team_assignments: dict[int, int] = {}
                if ma_json and isinstance(ma_json, dict):
                    for rally_entry in ma_json.get("rallies", []):
                        rid = rally_entry.get("rallyId") or rally_entry.get("rally_id", "")
                        if rid == rally_id:
                            t2p = rally_entry.get("trackToPlayer") or rally_entry.get("track_to_player", {})
                            profiles = ma_json.get("playerProfiles", {})
                            for tid_str, pid in t2p.items():
                                pid_str = str(pid)
                                if pid_str in profiles:
                                    team = profiles[pid_str].get("team")
                                    if team is not None:
                                        team_assignments[int(tid_str)] = int(team)
                            break

                results.append({
                    "rally_id": rally_id,
                    "video_id": video_id,
                    "gt_labels": gt_json,
                    "ball_positions_json": ball_json,
                    "positions_json": pos_json,
                    "fps": fps or 30.0,
                    "court_split_y": court_split_y,
                    "team_assignments": team_assignments,
                })

    return results


def _ball_y_court_side(ball_y: float, net_y: float) -> str:
    """Simple Y-threshold: ball below net = far, above = near."""
    return "far" if ball_y < net_y else "near"


def _trajectory_court_side(
    ball_positions: list[BallPosition],
    contact_frame: int,
    look_frames: int = 10,
) -> str:
    """Determine court_side from post-contact ball trajectory direction.

    After a near-court player hits the ball, it moves AWAY from near court
    (decreasing Y). After a far-court player, it moves toward near (increasing Y).
    """
    ball_by_frame: dict[int, BallPosition] = {}
    for bp in ball_positions:
        if bp.confidence >= 0.3:
            ball_by_frame[bp.frame_number] = bp

    contact_ball = ball_by_frame.get(contact_frame)
    if contact_ball is None:
        return "unknown"

    # Look at ball position after contact (at ~70% of look window)
    target_offset = int(look_frames * 0.7)
    post_ball = None
    for offset in range(target_offset, look_frames + 1):
        post_ball = ball_by_frame.get(contact_frame + offset)
        if post_ball is not None:
            break
    if post_ball is None:
        # Try smaller offsets
        for offset in range(target_offset - 1, 2, -1):
            post_ball = ball_by_frame.get(contact_frame + offset)
            if post_ball is not None:
                break

    if post_ball is None:
        return "unknown"

    dy = post_ball.y - contact_ball.y
    # Ball moving down (dy < 0) = moving toward far court = near player hit it
    # Ball moving up (dy > 0) = moving toward near court = far player hit it
    # Note: in image coords, y increases downward, so:
    # Near court = high Y, Far court = low Y
    # Ball dy > 0 = moving toward near = far player hit it
    # Ball dy < 0 = moving toward far = near player hit it

    if abs(dy) < 0.01:
        return "unknown"  # Too little movement to determine

    return "near" if dy < 0 else "far"


def _pre_trajectory_court_side(
    ball_positions: list[BallPosition],
    contact_frame: int,
    look_frames: int = 8,
) -> str:
    """Determine court_side from pre-contact ball trajectory.

    Before a near-court contact, ball was coming FROM far court (increasing Y).
    Before a far-court contact, ball was coming FROM near court (decreasing Y).
    """
    ball_by_frame: dict[int, BallPosition] = {}
    for bp in ball_positions:
        if bp.confidence >= 0.3:
            ball_by_frame[bp.frame_number] = bp

    contact_ball = ball_by_frame.get(contact_frame)
    if contact_ball is None:
        return "unknown"

    target_offset = int(look_frames * 0.7)
    pre_ball = None
    for offset in range(target_offset, look_frames + 1):
        pre_ball = ball_by_frame.get(contact_frame - offset)
        if pre_ball is not None:
            break
    if pre_ball is None:
        for offset in range(target_offset - 1, 2, -1):
            pre_ball = ball_by_frame.get(contact_frame - offset)
            if pre_ball is not None:
                break

    if pre_ball is None:
        return "unknown"

    dy = contact_ball.y - pre_ball.y
    # Ball was moving toward near (dy > 0) → near-court contact
    # Ball was moving toward far (dy < 0) → far-court contact
    if abs(dy) < 0.01:
        return "unknown"

    return "near" if dy > 0 else "far"


def _combined_court_side(
    ball_positions: list[BallPosition],
    contact_frame: int,
    net_y: float,
) -> str:
    """Combine post-trajectory + ball Y position for court_side.

    Uses trajectory direction as primary signal (more reliable for contacts
    near net). Falls back to ball Y position.
    """
    post = _trajectory_court_side(ball_positions, contact_frame)
    if post != "unknown":
        return post

    pre = _pre_trajectory_court_side(ball_positions, contact_frame)
    if pre != "unknown":
        return pre

    # Fall back to ball Y
    ball_by_frame = {bp.frame_number: bp for bp in ball_positions if bp.confidence >= 0.3}
    contact_ball = ball_by_frame.get(contact_frame)
    if contact_ball is not None:
        return _ball_y_court_side(contact_ball.y, net_y)

    return "unknown"


def _player_y_court_side(
    positions_json: list[dict[str, Any]] | None,
    contact_frame: int,
    nearest_track_id: int,
    net_y: float,
) -> str:
    """Use the nearest player's Y position to determine court_side.

    Near-court players have Y > net_y, far-court players Y < net_y.
    """
    if not positions_json or nearest_track_id < 0:
        return "unknown"

    # Find player position at contact frame
    for p in positions_json:
        if p.get("frameNumber") == contact_frame and p.get("trackId") == nearest_track_id:
            player_y = p["y"]
            return "near" if player_y > net_y else "far"

    # Try nearby frames
    for offset in range(1, 4):
        for f in (contact_frame + offset, contact_frame - offset):
            for p in positions_json:
                if p.get("frameNumber") == f and p.get("trackId") == nearest_track_id:
                    return "near" if p["y"] > net_y else "far"

    return "unknown"


def _player_y_court_side_split(
    positions_json: list[dict[str, Any]] | None,
    contact_frame: int,
    nearest_track_id: int,
    court_split_y: float | None,
    net_y: float,
) -> str:
    """Player Y using court_split_y (player midpoint) as threshold.

    Falls back to net_y if court_split_y unavailable.
    """
    split = court_split_y if court_split_y is not None else net_y
    if not positions_json or nearest_track_id < 0:
        return "unknown"

    for p in positions_json:
        if p.get("frameNumber") == contact_frame and p.get("trackId") == nearest_track_id:
            return "near" if p["y"] > split else "far"

    for offset in range(1, 4):
        for f in (contact_frame + offset, contact_frame - offset):
            for p in positions_json:
                if p.get("frameNumber") == f and p.get("trackId") == nearest_track_id:
                    return "near" if p["y"] > split else "far"

    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate court_side detection methods")
    parser.add_argument("--rally", type=str, help="Specific rally ID")
    args = parser.parse_args()

    rallies = _load_data()
    if args.rally:
        rallies = [r for r in rallies if r["rally_id"] == args.rally or r["rally_id"].startswith(args.rally)]

    if not rallies:
        console.print("[red]No rallies found with action GT.[/red]")
        return

    console.print(f"\n[bold]Court-Side Detection Accuracy ({len(rallies)} rallies)[/bold]\n")

    # For each signal, track (correct, total, unknown)
    signals = {
        "ball_y": {"correct": 0, "total": 0, "unknown": 0},
        "post_traj": {"correct": 0, "total": 0, "unknown": 0},
        "pre_traj": {"correct": 0, "total": 0, "unknown": 0},
        "combined": {"correct": 0, "total": 0, "unknown": 0},
        "player_y": {"correct": 0, "total": 0, "unknown": 0},
        # Regression check: court_split_y as player threshold (should stay <50%)
        "plyr_split": {"correct": 0, "total": 0, "unknown": 0},
    }

    for rally_idx, rally in enumerate(rallies):
        if not rally["team_assignments"]:
            continue

        ball_positions = [
            BallPosition(
                frame_number=bp["frameNumber"],
                x=bp["x"], y=bp["y"],
                confidence=bp.get("confidence", 1.0),
            )
            for bp in rally["ball_positions_json"]
            if bp.get("x", 0) > 0 or bp.get("y", 0) > 0
        ]

        # Detect contacts to get frame numbers
        player_positions = []
        if rally["positions_json"]:
            player_positions = [
                PlayerPosition(
                    frame_number=pp["frameNumber"],
                    track_id=pp["trackId"],
                    x=pp["x"], y=pp["y"],
                    width=pp["width"], height=pp["height"],
                    confidence=pp.get("confidence", 1.0),
                )
                for pp in rally["positions_json"]
            ]

        contacts = detect_contacts(
            ball_positions=ball_positions,
            player_positions=player_positions,
            net_y=rally["court_split_y"],
        )

        net_y = contacts.net_y

        # Determine per-rally team-to-side mapping from actual player positions.
        # team 0 isn't always "near" — it depends on the initial assignment and
        # whether a side switch has occurred.
        team_avg_y: dict[int, list[float]] = {0: [], 1: []}
        if rally["positions_json"]:
            for p in rally["positions_json"]:
                tid = p.get("trackId", -1)
                team = rally["team_assignments"].get(tid)
                if team is not None:
                    team_avg_y[team].append(p["y"])
        if team_avg_y[0] and team_avg_y[1]:
            avg_0 = sum(team_avg_y[0]) / len(team_avg_y[0])
            avg_1 = sum(team_avg_y[1]) / len(team_avg_y[1])
            # Higher Y = near court (closer to camera)
            if avg_0 > avg_1:
                team_to_side = {0: "near", 1: "far"}
            else:
                team_to_side = {0: "far", 1: "near"}
        else:
            team_to_side = {0: "near", 1: "far"}

        # Match GT labels to detected contacts
        tolerance = max(1, round(rally["fps"] * 167 / 1000))
        gt_labels = rally["gt_labels"]

        for gt in gt_labels:
            gt_frame = gt["frame"]
            gt_tid = gt.get("playerTrackId", -1)
            if gt_tid < 0:
                continue

            gt_team = rally["team_assignments"].get(gt_tid)
            if gt_team is None:
                continue

            gt_side = team_to_side[gt_team]

            # Find matching contact
            best_contact = None
            best_dist = tolerance + 1
            for c in contacts.contacts:
                dist = abs(c.frame - gt_frame)
                if dist <= tolerance and dist < best_dist:
                    best_dist = dist
                    best_contact = c

            if best_contact is None:
                continue

            frame = best_contact.frame

            # Test each signal
            # 1. Ball Y
            ball_y_side = _ball_y_court_side(best_contact.ball_y, net_y)
            if ball_y_side != "unknown":
                signals["ball_y"]["total"] += 1
                if ball_y_side == gt_side:
                    signals["ball_y"]["correct"] += 1
            else:
                signals["ball_y"]["unknown"] += 1

            # 2. Post-contact trajectory
            post_side = _trajectory_court_side(ball_positions, frame)
            if post_side != "unknown":
                signals["post_traj"]["total"] += 1
                if post_side == gt_side:
                    signals["post_traj"]["correct"] += 1
            else:
                signals["post_traj"]["unknown"] += 1

            # 3. Pre-contact trajectory
            pre_side = _pre_trajectory_court_side(ball_positions, frame)
            if pre_side != "unknown":
                signals["pre_traj"]["total"] += 1
                if pre_side == gt_side:
                    signals["pre_traj"]["correct"] += 1
            else:
                signals["pre_traj"]["unknown"] += 1

            # 4. Combined
            combined_side = _combined_court_side(ball_positions, frame, net_y)
            if combined_side != "unknown":
                signals["combined"]["total"] += 1
                if combined_side == gt_side:
                    signals["combined"]["correct"] += 1
            else:
                signals["combined"]["unknown"] += 1

            # 5. Player Y (using ball-trajectory net_y as split)
            player_side = _player_y_court_side(
                rally["positions_json"], frame,
                best_contact.player_track_id, net_y,
            )
            if player_side != "unknown":
                signals["player_y"]["total"] += 1
                if player_side == gt_side:
                    signals["player_y"]["correct"] += 1
            else:
                signals["player_y"]["unknown"] += 1

            # 6. Player Y (using court_split_y as split — correct threshold)
            plyr_split_side = _player_y_court_side_split(
                rally["positions_json"], frame,
                best_contact.player_track_id,
                rally["court_split_y"], net_y,
            )
            if plyr_split_side != "unknown":
                signals["plyr_split"]["total"] += 1
                if plyr_split_side == gt_side:
                    signals["plyr_split"]["correct"] += 1
            else:
                signals["plyr_split"]["unknown"] += 1

        if (rally_idx + 1) % 20 == 0:
            print(f"  [{rally_idx + 1}/{len(rallies)}]...")

    # Results table
    table = Table(title="Court-Side Signal Accuracy")
    table.add_column("Signal", style="bold")
    table.add_column("Correct", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Unknown", justify="right")
    table.add_column("Accuracy", justify="right")
    table.add_column("Coverage", justify="right")

    total_gt = max(s["total"] + s["unknown"] for s in signals.values())

    for name, stats in signals.items():
        acc = stats["correct"] / max(1, stats["total"])
        coverage = stats["total"] / max(1, total_gt)
        table.add_row(
            name,
            str(stats["correct"]),
            str(stats["total"]),
            str(stats["unknown"]),
            f"{acc:.1%}",
            f"{coverage:.1%}",
        )

    console.print(table)


if __name__ == "__main__":
    main()
