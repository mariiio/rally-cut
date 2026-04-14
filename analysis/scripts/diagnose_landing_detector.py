"""Per-rally diagnostic for why landing heatmaps are empty.

For a given video, loads the same inputs that ``compute-match-stats``
uses, then enumerates — per rally — which gating condition in
``detect_rally_landings()`` is firing. Aggregates drop reasons at the end.

Usage
-----
    cd analysis
    uv run python scripts/diagnose_landing_detector.py <video-id>
    uv run python scripts/diagnose_landing_detector.py <video-id> --limit 10
"""

from __future__ import annotations

import argparse
from collections import Counter
from typing import Any

from rich.console import Console

from rallycut.cli.commands.compute_match_stats import (
    _load_rally_actions_and_positions,
)
from rallycut.statistics.landing_detector import (
    HALF_COURT_M,
    _find_ball_near_frame,
    _get_court_side,
    _player_feet_court_xy,
    detect_rally_landings,
    find_landing,
)
from rallycut.tracking.action_classifier import ActionType

console = Console()


def _classify_rally(
    ra: Any,
    ball_pos: list[Any],
    pos_raw: list[dict[str, Any]] | None,
    calibrator: Any,
) -> tuple[int, str, dict[str, int]]:
    """Run detect_rally_landings and return (count, drop_reason, team_counts).

    drop_reason is populated when count==0 and explains *why*.
    team_counts reports how many actions had team A/B/unknown.
    """
    team_counts: Counter[str] = Counter()
    for a in ra.actions:
        team_counts[a.team or "unknown"] += 1

    if calibrator is None or not calibrator.is_calibrated:
        return 0, "no_calibration", dict(team_counts)
    if not ra.actions:
        return 0, "no_actions", dict(team_counts)
    if not ball_pos:
        return 0, "no_ball_positions", dict(team_counts)

    landings = detect_rally_landings(
        ra, ball_pos, calibrator, 1920, 1080, positions_raw=pos_raw,
    )
    if landings:
        # Emit a compact summary of the passed landings for orientation check.
        details = ";".join(
            f"{lp.action_type}:team={lp.team},y={lp.court_y:.1f}"
            for lp in landings
            if lp.court_y is not None
        )
        return len(landings), f"OK({details})", dict(team_counts)

    # Zero landings despite inputs — figure out which gate dropped them.
    # Walk the same paths detect_rally_landings walks.
    reasons: list[str] = []
    actions = sorted(ra.actions, key=lambda a: a.frame)

    serve = ra.serve
    if serve is not None and serve.action_type == ActionType.SERVE:
        receive = None
        for a in actions:
            if a.frame > serve.frame and a.action_type == ActionType.RECEIVE:
                receive = a
                break
        if receive is None:
            reasons.append("serve_no_receive")
        elif receive.player_track_id < 0:
            reasons.append("serve_receive_trackid_neg")
        elif pos_raw is None:
            reasons.append("serve_no_positions_raw")
        else:
            court_pos = _player_feet_court_xy(
                pos_raw, receive.player_track_id, receive.frame, calibrator,
            )
            if court_pos is None:
                reasons.append("serve_feet_projection_failed")
            else:
                server_side = _get_court_side(serve.team)
                target_on_near = court_pos[1] > HALF_COURT_M
                # Also look up server's feet position to see ground truth.
                srv_court = None
                if serve.player_track_id >= 0:
                    srv_court = _player_feet_court_xy(
                        pos_raw, serve.player_track_id,
                        serve.frame, calibrator,
                    )
                srv_y_str = f"{srv_court[1]:.1f}" if srv_court else "na"
                if server_side == "unknown":
                    reasons.append("serve_team_unknown")
                elif not (
                    (server_side == "near" and not target_on_near)
                    or (server_side == "far" and target_on_near)
                ):
                    reasons.append(
                        f"serve_same_half_validation"
                        f"(team={serve.team} cs={serve.court_side} "
                        f"srv_y={srv_y_str} recv_y={court_pos[1]:.1f})"
                    )
                else:
                    reasons.append("serve_unexpected_drop")
    else:
        reasons.append("no_serve")

    # Attack diagnostics
    attack_count = sum(
        1 for a in actions if a.action_type == ActionType.ATTACK
    )
    if attack_count == 0:
        reasons.append("no_attacks")
    else:
        atk_drops: Counter[str] = Counter()
        for i, action in enumerate(actions):
            if action.action_type != ActionType.ATTACK:
                continue
            next_contact_court: tuple[float, float] | None = None
            next_frame: int | None = None
            drop: str | None = None
            for b in actions[i + 1 :]:
                if b.action_type == ActionType.UNKNOWN:
                    continue
                if b.player_track_id < 0:
                    drop = "attack_next_trackid_neg"
                    break
                if pos_raw is None:
                    drop = "attack_no_positions_raw"
                    break
                bc = _player_feet_court_xy(
                    pos_raw, b.player_track_id, b.frame, calibrator,
                )
                if bc is None:
                    drop = "attack_next_feet_failed"
                    break
                next_contact_court = bc
                next_frame = b.frame
                break

            if next_contact_court is not None and next_frame is not None:
                attacker_side = _get_court_side(action.team)
                landing_on_near = next_contact_court[1] > HALF_COURT_M
                if attacker_side == "unknown":
                    atk_drops["attack_team_unknown"] += 1
                elif not (
                    (attacker_side == "near" and not landing_on_near)
                    or (attacker_side == "far" and landing_on_near)
                ):
                    atk_drops["attack_same_half_validation"] += 1
                else:
                    atk_drops["attack_unexpected_drop"] += 1
                continue

            # Terminal attack path
            if drop is not None:
                atk_drops[drop] += 1
                continue
            stopped = find_landing(ball_pos, action.frame)
            if stopped is None:
                fallback_hit = False
                for dt in range(10, 31):
                    if _find_ball_near_frame(
                        ball_pos, action.frame + dt, radius=2,
                    ) is not None:
                        fallback_hit = True
                        break
                if not fallback_hit:
                    atk_drops["terminal_no_ball_trajectory"] += 1
                else:
                    t_side = _get_court_side(action.team)
                    if t_side == "unknown":
                        atk_drops["terminal_team_unknown"] += 1
                    else:
                        atk_drops["terminal_same_half_validation"] += 1
            else:
                t_side = _get_court_side(action.team)
                if t_side == "unknown":
                    atk_drops["terminal_team_unknown"] += 1
                else:
                    atk_drops["terminal_same_half_validation"] += 1
        for k, v in atk_drops.items():
            reasons.append(f"{k}x{v}")

    reason_str = ",".join(reasons) if reasons else "unknown"
    return 0, reason_str, dict(team_counts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_id", help="Video UUID to diagnose")
    parser.add_argument(
        "--limit", type=int, default=None, help="Only inspect first N rallies",
    )
    args = parser.parse_args()

    console.print(f"[bold]Loading inputs for video[/bold] {args.video_id[:8]}...")
    (
        rally_actions_list, _all_positions, _video_fps,
        ball_positions_map, calibrator, _match_analysis,
        video_width, video_height, positions_raw_map,
    ) = _load_rally_actions_and_positions(args.video_id)

    # Top-level summary
    team_totals: Counter[str] = Counter()
    total_actions = 0
    for ra in rally_actions_list:
        for a in ra.actions:
            team_totals[a.team or "unknown"] += 1
            total_actions += 1

    console.print(
        f"  rallies={len(rally_actions_list)}  actions={total_actions}  "
        f"team(A)={team_totals['A']} team(B)={team_totals['B']} "
        f"team(unknown)={team_totals['unknown']}"
    )
    console.print(
        f"  calibrated={bool(calibrator and calibrator.is_calibrated)}  "
        f"video={video_width}x{video_height}  "
        f"ball_rallies={len(ball_positions_map)}  "
        f"pos_raw_rallies={len(positions_raw_map)}"
    )
    console.print()

    rallies = rally_actions_list
    if args.limit is not None:
        rallies = rallies[: args.limit]

    drop_counter: Counter[str] = Counter()
    total_landings = 0
    zero_landing_rallies = 0

    for i, ra in enumerate(rallies, start=1):
        ball_pos = ball_positions_map.get(ra.rally_id, [])
        pos_raw = positions_raw_map.get(ra.rally_id)
        team_c: Counter[str] = Counter(
            a.team or "unknown" for a in ra.actions
        )
        n_serves = sum(
            1 for a in ra.actions if a.action_type == ActionType.SERVE
        )
        n_attacks = sum(
            1 for a in ra.actions if a.action_type == ActionType.ATTACK
        )
        count, drop_reason, _ = _classify_rally(
            ra, ball_pos, pos_raw, calibrator,
        )
        total_landings += count
        if count == 0:
            zero_landing_rallies += 1
            # Bucket into the dominant gate for aggregate.
            # Primary gates first, then action-level reasons.
            primary = drop_reason.split(",")[0]
            drop_counter[primary] += 1

        extra = drop_reason if drop_reason else ""
        console.print(
            f"[{i}/{len(rallies)}] rally={ra.rally_id[:8]} "
            f"actions={len(ra.actions)} ball={len(ball_pos)} "
            f"pos_raw={len(pos_raw) if pos_raw else 0} "
            f"S/A={n_serves}/{n_attacks} "
            f"teams(A={team_c['A']},B={team_c['B']},u={team_c['unknown']}) "
            f"landings={count} "
            f"{('drop=' + extra) if count == 0 else extra}"
        )

    console.print()
    console.print("[bold]Summary[/bold]")
    console.print(
        f"  total landings: {total_landings}  "
        f"zero-landing rallies: {zero_landing_rallies}/{len(rallies)}"
    )
    console.print("  dominant drop reasons (primary gate):")
    for reason, n in drop_counter.most_common():
        console.print(f"    {reason}: {n}")


if __name__ == "__main__":
    main()
