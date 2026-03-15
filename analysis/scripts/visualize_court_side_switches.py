#!/usr/bin/env python3
"""Visualize court-side violations to confirm whether they are real ID switches.

Extracts frames around each violation showing track positions, court_split_y line,
and team assignments. Focuses on mid-rally transitions (not frame-0 violations)
which are most likely real switches.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass

sys.path.insert(0, ".")

import cv2
import numpy as np
from pathlib import Path

from rallycut.evaluation.db import get_connection
from rallycut.evaluation.tracking.db import get_video_path, load_rallies_for_video
from rallycut.tracking.player_filter import classify_teams
from rallycut.tracking.player_tracker import PlayerPosition


WINDOW_SIZE = 30
NEAR_NET_MARGIN = 0.03
MIN_WRONG_WINDOWS = 1
# Only look at violations starting after this frame (skip frame-0 whole-rally issues)
MIN_TRANSITION_FRAME = 30


@dataclass
class Violation:
    rally_id: str
    video_id: str
    start_ms: int
    track_id: int
    team: int
    wrong_start: int
    wrong_end: int
    num_windows: int


def find_mid_rally_violations(
    rallies: list, video_id: str
) -> list[Violation]:
    """Find violations that start mid-rally (likely real switches)."""
    violations: list[Violation] = []

    for rally in rallies:
        if rally.court_split_y is None:
            continue

        team_assignments = classify_teams(rally.positions, rally.court_split_y)
        if not team_assignments:
            continue

        for track_id in rally.primary_track_ids:
            team = team_assignments.get(track_id)
            if team is None:
                continue

            # Get track positions sorted by frame
            track_pos = sorted(
                [p for p in rally.positions if p.track_id == track_id],
                key=lambda p: p.frame_number,
            )
            if not track_pos:
                continue

            # Compute per-window median Y
            wrong_start = None
            wrong_count = 0
            for i in range(0, len(track_pos), WINDOW_SIZE):
                chunk = track_pos[i : i + WINDOW_SIZE]
                if len(chunk) < 5:
                    continue
                ys = sorted([p.y for p in chunk])
                median_y = ys[len(ys) // 2]
                start_frame = chunk[0].frame_number
                end_frame = chunk[-1].frame_number

                if abs(median_y - rally.court_split_y) < NEAR_NET_MARGIN:
                    # Near net — flush
                    if wrong_count >= MIN_WRONG_WINDOWS and wrong_start is not None:
                        if wrong_start >= MIN_TRANSITION_FRAME:
                            violations.append(Violation(
                                rally_id=rally.rally_id,
                                video_id=video_id,
                                start_ms=rally.start_ms,
                                track_id=track_id,
                                team=team,
                                wrong_start=wrong_start,
                                wrong_end=end_frame,
                                num_windows=wrong_count,
                            ))
                    wrong_start = None
                    wrong_count = 0
                    continue

                on_near = median_y > rally.court_split_y
                expected_near = team == 0
                if on_near != expected_near:
                    if wrong_start is None:
                        wrong_start = start_frame
                    wrong_count += 1
                else:
                    if wrong_count >= MIN_WRONG_WINDOWS and wrong_start is not None:
                        if wrong_start >= MIN_TRANSITION_FRAME:
                            violations.append(Violation(
                                rally_id=rally.rally_id,
                                video_id=video_id,
                                start_ms=rally.start_ms,
                                track_id=track_id,
                                team=team,
                                wrong_start=wrong_start,
                                wrong_end=end_frame,
                                num_windows=wrong_count,
                            ))
                    wrong_start = None
                    wrong_count = 0

            # Flush
            if wrong_count >= MIN_WRONG_WINDOWS and wrong_start is not None:
                if wrong_start >= MIN_TRANSITION_FRAME:
                    violations.append(Violation(
                        rally_id=rally.rally_id,
                        video_id=video_id,
                        start_ms=rally.start_ms,
                        track_id=track_id,
                        team=team,
                        wrong_start=wrong_start,
                        wrong_end=track_pos[-1].frame_number,
                        num_windows=wrong_count,
                    ))

    return violations


def render_frame_with_tracks(
    frame: np.ndarray,
    positions: list[PlayerPosition],
    frame_num: int,
    court_split_y: float,
    team_assignments: dict[int, int],
    primary_ids: list[int],
    highlight_track: int | None = None,
    label: str = "",
) -> np.ndarray:
    """Draw tracks, court split line, and team labels on frame."""
    h, w = frame.shape[:2]
    img = frame.copy()

    # Draw court split line
    split_px = int(court_split_y * h)
    cv2.line(img, (0, split_px), (w, split_px), (0, 255, 255), 2)
    cv2.putText(img, "NEAR (team 0)", (10, split_px + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(img, "FAR (team 1)", (10, split_px - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Draw track positions
    frame_pos = [p for p in positions if p.frame_number == frame_num]
    team_colors = {0: (0, 200, 0), 1: (200, 0, 0)}  # near=green, far=blue

    for p in frame_pos:
        if p.track_id not in primary_ids:
            continue
        team = team_assignments.get(p.track_id, -1)
        color = team_colors.get(team, (128, 128, 128))

        if p.track_id == highlight_track:
            color = (0, 0, 255)  # Red for highlighted track
            thickness = 4
        else:
            thickness = 2

        x1 = int((p.x - p.width / 2) * w)
        y1 = int((p.y - p.height / 2) * h)
        x2 = int((p.x + p.width / 2) * w)
        y2 = int((p.y + p.height / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        team_label = f"T{p.track_id} (team {team})"
        on_side = "NEAR" if p.y > court_split_y else "FAR"
        expected = "NEAR" if team == 0 else "FAR"
        wrong = " WRONG!" if on_side != expected else ""
        cv2.putText(img, f"{team_label} {on_side}{wrong}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Label bar
    cv2.rectangle(img, (0, 0), (w, 45), (40, 40, 40), -1)
    cv2.putText(img, label, (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    return img


def main() -> None:
    out_dir = Path("analysis/outputs/court_side_switches")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get all video IDs
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT r.video_id, v.filename
                FROM rallies r
                JOIN player_tracks pt ON pt.rally_id = r.id
                JOIN videos v ON v.id = r.video_id
                WHERE pt.positions_json IS NOT NULL
                ORDER BY v.filename
            """)
            video_rows = cur.fetchall()

    print(f"Scanning {len(video_rows)} videos for mid-rally court-side violations...\n")

    all_violations: list[Violation] = []
    video_filenames: dict[str, str] = {}

    for video_id, filename in video_rows:
        video_filenames[video_id] = filename
        rallies = load_rallies_for_video(video_id)
        violations = find_mid_rally_violations(rallies, video_id)
        all_violations.extend(violations)

    # Group by rally and find rallies with opposite-team violations
    rally_violations: dict[str, list[Violation]] = defaultdict(list)
    for v in all_violations:
        rally_violations[v.rally_id].append(v)

    # Rank: prefer rallies with violations from both teams (likely real switches)
    ranked: list[tuple[str, list[Violation], bool]] = []
    for rally_id, violations in rally_violations.items():
        teams = {v.team for v in violations}
        has_both = len(teams) >= 2
        ranked.append((rally_id, violations, has_both))
    ranked.sort(key=lambda x: (-x[2], -max(v.num_windows for v in x[1])))

    print(f"Found {len(all_violations)} mid-rally violations in {len(rally_violations)} rallies")
    paired_count = sum(1 for _, _, both in ranked if both)
    print(f"  With opposite-team violations (likely real switches): {paired_count}")
    print(f"  Single-team violations: {len(rally_violations) - paired_count}")
    print()

    # Visualize top cases
    max_visualize = 10
    visualized = 0

    for rally_id, violations, has_both in ranked:
        if visualized >= max_visualize:
            break

        video_id = violations[0].video_id
        filename = video_filenames.get(video_id, "?")
        video_path = get_video_path(video_id)
        if not video_path:
            print(f"  Skipping {rally_id[:8]} — no video file")
            continue

        # Load rally data
        rallies = load_rallies_for_video(video_id)
        rally = next((r for r in rallies if r.rally_id == rally_id), None)
        if rally is None or rally.court_split_y is None:
            continue

        team_assignments = classify_teams(rally.positions, rally.court_split_y)
        if not team_assignments:
            continue

        tag = "PAIRED" if has_both else "SINGLE"
        print(f"[{visualized + 1}/{max_visualize}] Rally {rally_id[:8]} ({filename}) [{tag}]")
        for v in violations:
            side = "near" if v.team == 0 else "far"
            wrong = "far" if v.team == 0 else "near"
            print(f"  T{v.track_id} ({side}) on {wrong} side frames {v.wrong_start}-{v.wrong_end} ({v.num_windows} windows)")

        # Pick the earliest violation's transition frame
        earliest = min(violations, key=lambda v: v.wrong_start)
        switch_frame = earliest.wrong_start

        # Extract frames around the switch
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_MSEC, rally.start_ms)

        frames_to_grab = [
            max(0, switch_frame - 30),
            max(0, switch_frame - 10),
            switch_frame,
            switch_frame + 10,
            switch_frame + 30,
        ]
        labels = [
            f"30 BEFORE (frame {frames_to_grab[0]})",
            f"10 BEFORE (frame {frames_to_grab[1]})",
            f"SWITCH (frame {switch_frame})",
            f"10 AFTER (frame {frames_to_grab[3]})",
            f"30 AFTER (frame {frames_to_grab[4]})",
        ]

        grabbed: dict[int, np.ndarray] = {}
        max_frame = max(frames_to_grab) + 1
        for i in range(max_frame):
            ret, img = cap.read()
            if not ret:
                break
            if i in frames_to_grab:
                grabbed[i] = img
        cap.release()

        if not grabbed:
            print("  Could not extract frames")
            continue

        panels = []
        for fn, label in zip(frames_to_grab, labels):
            if fn not in grabbed:
                continue
            rendered = render_frame_with_tracks(
                grabbed[fn], rally.positions, fn,
                rally.court_split_y, team_assignments,
                rally.primary_track_ids,
                highlight_track=earliest.track_id,
                label=label,
            )
            # Scale down
            h, w = rendered.shape[:2]
            scale = 960 / w
            rendered = cv2.resize(rendered, (int(w * scale), int(h * scale)))
            panels.append(rendered)

        if panels:
            # Title bar
            pw = panels[0].shape[1]
            title = np.zeros((60, pw, 3), dtype=np.uint8)
            title_text = (
                f"{tag}: {rally_id[:8]} ({filename}) "
                f"T{earliest.track_id} switch @ frame {switch_frame}"
            )
            cv2.putText(title, title_text, (10, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            final = np.vstack([title] + panels)
            out_path = out_dir / f"{visualized + 1:02d}_{tag}_{rally_id[:8]}.png"
            cv2.imwrite(str(out_path), final)
            print(f"  Saved: {out_path}")

        visualized += 1

    print(f"\nSaved {visualized} visualizations to {out_dir}/")


if __name__ == "__main__":
    main()
