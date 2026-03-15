#!/usr/bin/env python3
"""Diagnose cross-team ID switches by detecting court-side violations.

Scans all tracked rallies and flags tracks that spend sustained time
on the wrong side of the court relative to their team assignment.
Identifies complementary swap pairs (two tracks from opposite teams
swapping court sides at approximately the same frame).
"""

from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass, field

sys.path.insert(0, ".")

from rallycut.evaluation.db import get_connection
from rallycut.evaluation.tracking.db import load_rallies_for_video
from rallycut.tracking.player_filter import classify_teams
from rallycut.tracking.player_tracker import PlayerPosition


WINDOW_SIZE = 30  # frames per temporal window
MIN_WRONG_SIDE_WINDOWS = 1  # minimum contiguous wrong-side windows to flag
NEAR_NET_MARGIN = 0.03  # ignore positions within this fraction of court_split_y
PAIR_FRAME_TOLERANCE = 15  # max frame difference for complementary swap pairing


@dataclass
class CourtSideViolation:
    """A track spending sustained time on the wrong court side."""

    rally_id: str
    track_id: int
    team: int  # 0=near (high Y), 1=far (low Y)
    wrong_side_start_frame: int
    wrong_side_end_frame: int
    wrong_side_window_count: int
    median_y_values: list[float] = field(default_factory=list, repr=False)


@dataclass
class SwapPair:
    """Two tracks from opposite teams swapping sides at ~same frame."""

    rally_id: str
    track_a_id: int
    track_a_team: int
    track_b_id: int
    track_b_team: int
    approx_switch_frame: int
    violation_a: CourtSideViolation
    violation_b: CourtSideViolation


def _compute_median_y_per_window(
    positions: list[PlayerPosition],
    track_id: int,
    window_size: int = WINDOW_SIZE,
) -> list[tuple[int, int, float]]:
    """Compute median Y per temporal window for a track.

    Returns list of (start_frame, end_frame, median_y).
    """
    track_pos = sorted(
        [p for p in positions if p.track_id == track_id],
        key=lambda p: p.frame_number,
    )
    if not track_pos:
        return []

    windows: list[tuple[int, int, float]] = []
    for i in range(0, len(track_pos), window_size):
        chunk = track_pos[i : i + window_size]
        if len(chunk) < 5:  # too few points for reliable median
            continue
        ys = [p.y for p in chunk]
        ys.sort()
        median_y = ys[len(ys) // 2]
        windows.append((chunk[0].frame_number, chunk[-1].frame_number, median_y))

    return windows


def _classify_side(y: float, court_split_y: float) -> int | None:
    """Classify Y position as near (0) or far (1) side.

    Returns None if too close to the split line.
    """
    if abs(y - court_split_y) < NEAR_NET_MARGIN:
        return None
    # Near team (team 0) is on the side with higher Y (bottom of frame)
    # Far team (team 1) is on the side with lower Y (top of frame)
    return 0 if y > court_split_y else 1


def detect_violations(
    positions: list[PlayerPosition],
    team_assignments: dict[int, int],
    court_split_y: float,
    primary_track_ids: list[int],
) -> list[CourtSideViolation]:
    """Detect tracks spending sustained time on the wrong court side."""
    violations: list[CourtSideViolation] = []

    for track_id in primary_track_ids:
        team = team_assignments.get(track_id)
        if team is None:
            continue

        windows = _compute_median_y_per_window(positions, track_id)
        if not windows:
            continue

        # Find contiguous runs of wrong-side windows
        wrong_run_start: int | None = None
        wrong_run_start_frame = 0
        wrong_count = 0
        wrong_medians: list[float] = []

        for start_frame, end_frame, median_y in windows:
            side = _classify_side(median_y, court_split_y)
            if side is not None and side != team:
                # Wrong side
                if wrong_run_start is None:
                    wrong_run_start = start_frame
                    wrong_run_start_frame = start_frame
                wrong_count += 1
                wrong_medians.append(median_y)
            else:
                # Right side or near-net: flush any accumulated violation
                if wrong_count >= MIN_WRONG_SIDE_WINDOWS:
                    violations.append(
                        CourtSideViolation(
                            rally_id="",  # filled by caller
                            track_id=track_id,
                            team=team,
                            wrong_side_start_frame=wrong_run_start_frame,
                            wrong_side_end_frame=end_frame,
                            wrong_side_window_count=wrong_count,
                            median_y_values=list(wrong_medians),
                        )
                    )
                wrong_run_start = None
                wrong_count = 0
                wrong_medians = []

        # Flush final run
        if wrong_count >= MIN_WRONG_SIDE_WINDOWS and windows:
            violations.append(
                CourtSideViolation(
                    rally_id="",
                    track_id=track_id,
                    team=team,
                    wrong_side_start_frame=wrong_run_start_frame,
                    wrong_side_end_frame=windows[-1][1],
                    wrong_side_window_count=wrong_count,
                    median_y_values=list(wrong_medians),
                )
            )

    return violations


def find_swap_pairs(
    violations: list[CourtSideViolation],
    team_assignments: dict[int, int],
) -> list[SwapPair]:
    """Find complementary swap pairs from violations."""
    pairs: list[SwapPair] = []
    used: set[int] = set()

    for i, va in enumerate(violations):
        if i in used:
            continue
        for j, vb in enumerate(violations):
            if j in used or j <= i:
                continue
            # Must be from opposite teams
            if va.team == vb.team:
                continue
            # Switch frames must be close
            if (
                abs(va.wrong_side_start_frame - vb.wrong_side_start_frame)
                <= PAIR_FRAME_TOLERANCE
            ):
                pairs.append(
                    SwapPair(
                        rally_id=va.rally_id,
                        track_a_id=va.track_id,
                        track_a_team=va.team,
                        track_b_id=vb.track_id,
                        track_b_team=vb.team,
                        approx_switch_frame=(
                            va.wrong_side_start_frame + vb.wrong_side_start_frame
                        )
                        // 2,
                        violation_a=va,
                        violation_b=vb,
                    )
                )
                used.add(i)
                used.add(j)
                break

    return pairs


def main() -> None:
    # Get all video IDs with tracked rallies
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT r.video_id, v.filename, COUNT(r.id)
                FROM rallies r
                JOIN player_tracks pt ON pt.rally_id = r.id
                JOIN videos v ON v.id = r.video_id
                WHERE pt.positions_json IS NOT NULL
                GROUP BY r.video_id, v.filename
                ORDER BY v.filename
            """)
            video_rows = cur.fetchall()

    print(f"Found {len(video_rows)} videos with tracked rallies")
    print("=" * 90)

    total_rallies = 0
    rallies_with_violations = 0
    total_violations = 0
    all_paired_swaps: list[SwapPair] = []
    all_unpaired_violations: list[CourtSideViolation] = []

    for vid_idx, (video_id, filename, rally_count) in enumerate(video_rows, 1):
        rallies = load_rallies_for_video(video_id)
        vid_violations = 0
        vid_rallies_with_violations = 0

        for rally in rallies:
            total_rallies += 1

            if rally.court_split_y is None:
                continue

            # Compute per-rally team assignments from Y position
            # (not match-level A/B which don't correspond to court side)
            team_assignments = classify_teams(
                rally.positions, rally.court_split_y
            )
            if not team_assignments:
                continue

            violations = detect_violations(
                rally.positions,
                team_assignments,
                rally.court_split_y,
                rally.primary_track_ids,
            )

            if not violations:
                continue

            # Set rally IDs
            for v in violations:
                v.rally_id = rally.rally_id

            vid_violations += len(violations)
            vid_rallies_with_violations += 1
            rallies_with_violations += 1
            total_violations += len(violations)

            # Find swap pairs
            pairs = find_swap_pairs(violations, team_assignments)
            all_paired_swaps.extend(pairs)

            # Track unpaired violations
            paired_indices = set()
            for pair in pairs:
                for idx, v in enumerate(violations):
                    if v is pair.violation_a or v is pair.violation_b:
                        paired_indices.add(idx)
            for idx, v in enumerate(violations):
                if idx not in paired_indices:
                    all_unpaired_violations.append(v)

            # Print details
            team_labels = {0: "near", 1: "far"}
            for v in violations:
                wrong_label = team_labels.get(1 - v.team, "?")
                print(
                    f"  Rally {v.rally_id[:8]}: "
                    f"T{v.track_id} ({team_labels.get(v.team, '?')}) "
                    f"on {wrong_label} side "
                    f"frames {v.wrong_side_start_frame}-{v.wrong_side_end_frame} "
                    f"({v.wrong_side_window_count} windows)"
                )

            for pair in pairs:
                print(
                    f"    -> SWAP PAIR: T{pair.track_a_id}<->T{pair.track_b_id} "
                    f"at ~frame {pair.approx_switch_frame}"
                )

        status = f"{vid_violations} violations in {vid_rallies_with_violations} rallies" if vid_violations else "clean"
        print(
            f"[{vid_idx}/{len(video_rows)}] {filename} "
            f"({len(rallies)} rallies): {status}"
        )

    # Summary
    print(f"\n{'=' * 90}")
    print("SUMMARY")
    print(f"{'=' * 90}")
    print(f"Total rallies scanned: {total_rallies}")
    print(f"Rallies with court-side violations: {rallies_with_violations}")
    print(f"Total violations: {total_violations}")
    print(f"Paired swaps (high confidence): {len(all_paired_swaps)}")
    print(f"Single-sided violations: {len(all_unpaired_violations)}")

    if all_paired_swaps:
        print(f"\nPaired swap details:")
        for idx, pair in enumerate(all_paired_swaps, 1):
            print(
                f"  {idx}. Rally {pair.rally_id[:8]}: "
                f"T{pair.track_a_id} (team {pair.track_a_team}) <-> "
                f"T{pair.track_b_id} (team {pair.track_b_team}) "
                f"at ~frame {pair.approx_switch_frame}"
            )


if __name__ == "__main__":
    main()
