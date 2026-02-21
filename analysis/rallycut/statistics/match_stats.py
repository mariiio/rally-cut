"""Per-player and per-match statistics for beach volleyball.

Aggregates tracking and action classification data into meaningful
match statistics: per-player action counts, movement distance,
court zones, and overall match metrics.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from rallycut.court.calibration import CourtCalibrator
    from rallycut.tracking.action_classifier import ClassifiedAction, RallyActions
    from rallycut.tracking.player_tracker import PlayerPosition

logger = logging.getLogger(__name__)


@dataclass
class PlayerStats:
    """Statistics for a single player across a match or set."""

    track_id: int
    team: str = "unknown"  # "A" (near court), "B" (far court), or "unknown"
    # Action counts
    serves: int = 0
    receives: int = 0
    sets: int = 0
    attacks: int = 0
    blocks: int = 0
    digs: int = 0
    # Outcome counts (attributed from score progression)
    kills: int = 0  # Attacks that won the point
    attack_errors: int = 0  # Attacks that lost the point
    aces: int = 0  # Serves that won the point (no receive)
    serve_errors: int = 0  # Serves that lost the point
    # Movement
    total_distance_px: float = 0.0  # Total movement in pixels
    total_distance_m: float = 0.0  # Total movement in meters (if calibrated)
    avg_speed_px_per_frame: float = 0.0
    # Court zones (normalized 0-1, percentage of time in each zone)
    time_near_court_pct: float = 0.0  # % of time on near side
    time_far_court_pct: float = 0.0  # % of time on far side
    # Position heatmap (10x10 grid, normalized 0-1)
    position_heatmap: list[list[float]] = field(default_factory=list)
    # Frames active
    num_frames: int = 0
    court_side: str = "unknown"  # Primary court side ("near" or "far")

    @property
    def total_actions(self) -> int:
        return self.serves + self.receives + self.sets + self.attacks + self.blocks + self.digs

    @property
    def kill_pct(self) -> float:
        """Kill percentage: kills / attacks."""
        return self.kills / self.attacks if self.attacks > 0 else 0.0

    @property
    def attack_efficiency(self) -> float:
        """Attack efficiency: (kills - errors) / attacks."""
        return (self.kills - self.attack_errors) / self.attacks if self.attacks > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "trackId": self.track_id,
            "team": self.team,
            "serves": self.serves,
            "receives": self.receives,
            "sets": self.sets,
            "attacks": self.attacks,
            "blocks": self.blocks,
            "digs": self.digs,
            "kills": self.kills,
            "attackErrors": self.attack_errors,
            "aces": self.aces,
            "serveErrors": self.serve_errors,
            "killPct": round(self.kill_pct, 3),
            "attackEfficiency": round(self.attack_efficiency, 3),
            "totalActions": self.total_actions,
            "totalDistancePx": round(self.total_distance_px, 1),
            "avgSpeedPxPerFrame": round(self.avg_speed_px_per_frame, 4),
            "numFrames": self.num_frames,
            "courtSide": self.court_side,
        }
        if self.total_distance_m > 0:
            result["totalDistanceM"] = round(self.total_distance_m, 1)
        if self.position_heatmap:
            result["positionHeatmap"] = self.position_heatmap
        return result


@dataclass
class RallyStats:
    """Statistics for a single rally."""

    rally_id: str
    duration_frames: int
    duration_seconds: float
    num_contacts: int
    action_sequence: list[str]
    has_block: bool
    has_extended_exchange: bool  # 3+ contacts on a side
    max_rally_velocity: float  # Peak ball velocity during rally
    serving_side: str  # "near" or "far"
    # Outcome attribution (populated when score progression is available)
    terminal_action: str = "unknown"  # Last action type before point ends
    terminal_player_track_id: int = -1  # Player who made the terminal action
    point_winner: str = "unknown"  # "A", "B", or "unknown"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "rallyId": self.rally_id,
            "durationFrames": self.duration_frames,
            "durationSeconds": round(self.duration_seconds, 2),
            "numContacts": self.num_contacts,
            "actionSequence": self.action_sequence,
            "hasBlock": self.has_block,
            "hasExtendedExchange": self.has_extended_exchange,
            "maxRallyVelocity": round(self.max_rally_velocity, 4),
            "servingSide": self.serving_side,
        }
        if self.terminal_action != "unknown":
            d["terminalAction"] = self.terminal_action
            d["terminalPlayerTrackId"] = self.terminal_player_track_id
        if self.point_winner != "unknown":
            d["pointWinner"] = self.point_winner
        return d


@dataclass
class TeamStats:
    """Aggregated statistics for one team across a match."""

    team: str  # "A" or "B"
    player_ids: list[int] = field(default_factory=list)
    serves: int = 0
    receives: int = 0
    sets: int = 0
    attacks: int = 0
    blocks: int = 0
    digs: int = 0
    # Outcome counts (aggregated from players)
    kills: int = 0
    attack_errors: int = 0
    aces: int = 0
    serve_errors: int = 0
    # Score metrics
    points_won: int = 0
    side_out_pct: float = 0.0  # % of receiving rallies won
    serve_win_pct: float = 0.0  # % of serving rallies won

    @property
    def total_actions(self) -> int:
        return self.serves + self.receives + self.sets + self.attacks + self.blocks + self.digs

    @property
    def kill_pct(self) -> float:
        return self.kills / self.attacks if self.attacks > 0 else 0.0

    @property
    def attack_efficiency(self) -> float:
        return (self.kills - self.attack_errors) / self.attacks if self.attacks > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "team": self.team,
            "playerIds": self.player_ids,
            "serves": self.serves,
            "receives": self.receives,
            "sets": self.sets,
            "attacks": self.attacks,
            "blocks": self.blocks,
            "digs": self.digs,
            "kills": self.kills,
            "attackErrors": self.attack_errors,
            "aces": self.aces,
            "serveErrors": self.serve_errors,
            "killPct": round(self.kill_pct, 3),
            "attackEfficiency": round(self.attack_efficiency, 3),
            "totalActions": self.total_actions,
            "pointsWon": self.points_won,
            "sideOutPct": round(self.side_out_pct, 3),
            "serveWinPct": round(self.serve_win_pct, 3),
        }


@dataclass
class RallyScoreState:
    """Score state after a rally, inferred from serve ownership changes."""

    rally_id: str
    score_a: int
    score_b: int
    serving_team: str  # "A" or "B"
    point_winner: str  # "A", "B", or "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "rallyId": self.rally_id,
            "scoreA": self.score_a,
            "scoreB": self.score_b,
            "servingTeam": self.serving_team,
            "pointWinner": self.point_winner,
        }


@dataclass
class MatchStats:
    """Aggregated statistics for a full match."""

    # Per-player stats
    player_stats: list[PlayerStats] = field(default_factory=list)
    # Per-team stats
    team_stats: list[TeamStats] = field(default_factory=list)
    # Per-rally stats
    rally_stats: list[RallyStats] = field(default_factory=list)
    # Per-rally score progression
    score_progression: list[RallyScoreState] = field(default_factory=list)
    # Match-level aggregates
    total_rallies: int = 0
    total_contacts: int = 0
    avg_rally_duration_s: float = 0.0
    longest_rally_duration_s: float = 0.0
    avg_contacts_per_rally: float = 0.0
    side_out_rate: float = 0.0  # % of rallies won by receiving team
    final_score_a: int = 0
    final_score_b: int = 0
    # Video metadata
    video_fps: float = 30.0
    video_width: int = 1920
    video_height: int = 1080

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "totalRallies": self.total_rallies,
            "totalContacts": self.total_contacts,
            "avgRallyDurationS": round(self.avg_rally_duration_s, 2),
            "longestRallyDurationS": round(self.longest_rally_duration_s, 2),
            "avgContactsPerRally": round(self.avg_contacts_per_rally, 1),
            "sideOutRate": round(self.side_out_rate, 3),
            "playerStats": [p.to_dict() for p in self.player_stats],
            "rallyStats": [r.to_dict() for r in self.rally_stats],
        }
        if self.team_stats:
            d["teamStats"] = [t.to_dict() for t in self.team_stats]
        if self.score_progression:
            d["scoreProgression"] = [s.to_dict() for s in self.score_progression]
            d["finalScoreA"] = self.final_score_a
            d["finalScoreB"] = self.final_score_b
        return d


def compute_player_movement(
    positions: list[PlayerPosition],
    track_id: int,
    video_width: int = 1920,
    video_height: int = 1080,
    calibrator: CourtCalibrator | None = None,
) -> tuple[float, float, float]:
    """Compute total movement distance for a player track.

    Args:
        positions: All player positions.
        track_id: Track ID to compute movement for.
        video_width: Video width for pixel conversion.
        video_height: Video height for pixel conversion.
        calibrator: Optional court calibrator for real-world distance.

    Returns:
        (distance_px, distance_m, avg_speed_px_per_frame)
    """
    track_pos = sorted(
        [p for p in positions if p.track_id == track_id],
        key=lambda p: p.frame_number,
    )

    if len(track_pos) < 2:
        return 0.0, 0.0, 0.0

    total_px = 0.0
    total_m = 0.0
    num_moves = 0

    for i in range(1, len(track_pos)):
        prev = track_pos[i - 1]
        curr = track_pos[i]

        frame_gap = curr.frame_number - prev.frame_number
        if frame_gap <= 0 or frame_gap > 5:
            continue

        dx_px = (curr.x - prev.x) * video_width
        dy_px = (curr.y - prev.y) * video_height
        dist_px = math.sqrt(dx_px * dx_px + dy_px * dy_px)
        total_px += dist_px
        num_moves += 1

        if calibrator is not None and calibrator.is_calibrated:
            try:
                court_prev = calibrator.image_to_court(
                    (prev.x, prev.y + prev.height / 2), video_width, video_height
                )
                court_curr = calibrator.image_to_court(
                    (curr.x, curr.y + curr.height / 2), video_width, video_height
                )
                dist_m = math.sqrt(
                    (court_curr[0] - court_prev[0]) ** 2
                    + (court_curr[1] - court_prev[1]) ** 2
                )
                total_m += dist_m
            except (RuntimeError, ValueError):
                pass

    avg_speed = total_px / num_moves if num_moves > 0 else 0.0

    return total_px, total_m, avg_speed


def compute_position_heatmap(
    positions: list[PlayerPosition],
    track_id: int,
    grid_size: int = 10,
) -> list[list[float]]:
    """Compute position heatmap for a player on a grid.

    Args:
        positions: All player positions.
        track_id: Track ID.
        grid_size: Grid resolution (default 10x10).

    Returns:
        2D list of normalized occupancy values (0-1).
    """
    grid = np.zeros((grid_size, grid_size), dtype=np.float64)

    track_pos = [p for p in positions if p.track_id == track_id]
    if not track_pos:
        return grid.tolist()

    for p in track_pos:
        gx = min(int(p.x * grid_size), grid_size - 1)
        gy = min(int(p.y * grid_size), grid_size - 1)
        grid[gy, gx] += 1

    # Normalize
    total = grid.sum()
    if total > 0:
        grid /= total

    return grid.tolist()


def compute_match_scores(
    rally_actions_list: list[RallyActions],
) -> list[RallyScoreState]:
    """Compute running scores from serve ownership across consecutive rallies.

    Volleyball rule: the winner of the previous rally serves next.
    - If serving team changes between rallies → previous rally won by new server.
    - If serving team stays the same → they scored (side-out failed).
    - Last rally's winner is unknown (no next-rally info).

    Args:
        rally_actions_list: Classified actions for each rally (in match order).

    Returns:
        List of RallyScoreState with running scores.
    """
    scores: list[RallyScoreState] = []
    score_a = 0
    score_b = 0

    # Extract serving team per rally
    serving_teams: list[tuple[str, str]] = []  # (rally_id, serving_team)
    for ra in rally_actions_list:
        serving = ra.serving_team
        if serving:
            serving_teams.append((ra.rally_id, serving))

    skipped = len(rally_actions_list) - len(serving_teams)
    if skipped > 0:
        logger.warning(
            "Skipped %d/%d rallies with unknown serving team for score computation",
            skipped, len(rally_actions_list),
        )

    if not serving_teams:
        return scores

    for i, (rally_id, serving) in enumerate(serving_teams):
        point_winner = "unknown"

        if i < len(serving_teams) - 1:
            next_serving = serving_teams[i + 1][1]
            if next_serving != serving:
                # Server changed → new server won the previous point
                point_winner = next_serving
            else:
                # Same server → they scored (side-out failed for opponent)
                point_winner = serving

        if point_winner == "A":
            score_a += 1
        elif point_winner == "B":
            score_b += 1

        scores.append(RallyScoreState(
            rally_id=rally_id,
            score_a=score_a,
            score_b=score_b,
            serving_team=serving,
            point_winner=point_winner,
        ))

    return scores


def _find_terminal_action(ra: RallyActions) -> ClassifiedAction | None:
    """Find the terminal action in a rally (last non-UNKNOWN action).

    The terminal action is the last contact before the rally ends — typically
    an attack (kill or error), serve (ace or error), or block.
    """
    from rallycut.tracking.action_classifier import ActionType

    for action in reversed(ra.actions):
        if action.action_type not in (ActionType.UNKNOWN,):
            return action
    return None


def _attribute_outcomes(
    rally_stats: list[RallyStats],
    rally_actions_map: dict[str, RallyActions],
    player_stats_map: dict[int, PlayerStats],
    team_assignments: dict[int, int],
) -> None:
    """Attribute kills, aces, and errors to individual players.

    Logic:
    - If terminal action is ATTACK and the attacker's team won → kill
    - If terminal action is ATTACK and the attacker's team lost → attack error
    - If terminal action is SERVE and the server's team won → ace
    - If terminal action is SERVE and the server's team lost → serve error
    """
    for rs in rally_stats:
        if rs.point_winner == "unknown" or rs.terminal_action == "unknown":
            continue
        if rs.terminal_player_track_id < 0:
            continue

        player = player_stats_map.get(rs.terminal_player_track_id)
        if not player:
            continue

        # Determine terminal player's team
        team_int = team_assignments.get(rs.terminal_player_track_id)
        player_team = "A" if team_int == 0 else "B" if team_int == 1 else "unknown"
        if player_team == "unknown":
            continue

        won = player_team == rs.point_winner

        if rs.terminal_action == "attack":
            if won:
                player.kills += 1
            else:
                player.attack_errors += 1
        elif rs.terminal_action == "serve":
            if won:
                player.aces += 1
            else:
                player.serve_errors += 1


def compute_match_stats(
    rally_actions_list: list[RallyActions],
    player_positions: list[PlayerPosition],
    video_fps: float = 30.0,
    video_width: int = 1920,
    video_height: int = 1080,
    calibrator: CourtCalibrator | None = None,
) -> MatchStats:
    """Compute comprehensive match statistics.

    Args:
        rally_actions_list: Classified actions for each rally.
        player_positions: All player tracking positions across the match.
        video_fps: Video frame rate.
        video_width: Video width.
        video_height: Video height.
        calibrator: Optional court calibrator for real-world distances.

    Returns:
        MatchStats with per-player and per-rally statistics.
    """
    from rallycut.tracking.action_classifier import ActionType

    stats = MatchStats(
        video_fps=video_fps,
        video_width=video_width,
        video_height=video_height,
    )

    # Collect all unique track IDs from actions
    all_track_ids: set[int] = set()
    for ra in rally_actions_list:
        for action in ra.actions:
            if action.player_track_id >= 0:
                all_track_ids.add(action.player_track_id)

    # Also add track IDs from player positions (may have players without actions)
    track_frame_counts: dict[int, int] = {}
    for p in player_positions:
        if p.track_id >= 0:
            track_frame_counts[p.track_id] = track_frame_counts.get(p.track_id, 0) + 1

    # Keep top 4 tracks by frame count (beach volleyball = 4 players)
    sorted_tracks = sorted(track_frame_counts.items(), key=lambda x: -x[1])
    for track_id, _ in sorted_tracks[:4]:
        all_track_ids.add(track_id)

    # Build a merged team_assignments from all rallies
    merged_team_assignments: dict[int, int] = {}
    for ra in rally_actions_list:
        merged_team_assignments.update(ra.team_assignments)

    # Compute per-player stats
    for track_id in sorted(all_track_ids):
        # Derive team from merged assignments
        team_int = merged_team_assignments.get(track_id)
        team_label = "A" if team_int == 0 else "B" if team_int == 1 else "unknown"
        player = PlayerStats(track_id=track_id, team=team_label)

        # Count actions
        for ra in rally_actions_list:
            for action in ra.actions:
                if action.player_track_id != track_id:
                    continue
                if action.action_type == ActionType.SERVE:
                    player.serves += 1
                elif action.action_type == ActionType.RECEIVE:
                    player.receives += 1
                elif action.action_type == ActionType.SET:
                    player.sets += 1
                elif action.action_type == ActionType.ATTACK:
                    player.attacks += 1
                elif action.action_type == ActionType.BLOCK:
                    player.blocks += 1
                elif action.action_type == ActionType.DIG:
                    player.digs += 1

        # Compute movement
        dist_px, dist_m, avg_speed = compute_player_movement(
            player_positions, track_id, video_width, video_height, calibrator
        )
        player.total_distance_px = dist_px
        player.total_distance_m = dist_m
        player.avg_speed_px_per_frame = avg_speed

        # Compute court side and position heatmap
        track_pos = [p for p in player_positions if p.track_id == track_id]
        player.num_frames = len(track_pos)

        if track_pos:
            avg_y = np.mean([p.y for p in track_pos])
            player.court_side = "near" if avg_y >= 0.5 else "far"
            near_count = sum(1 for p in track_pos if p.y >= 0.5)
            player.time_near_court_pct = near_count / len(track_pos)
            player.time_far_court_pct = 1.0 - player.time_near_court_pct

        player.position_heatmap = compute_position_heatmap(
            player_positions, track_id
        )

        stats.player_stats.append(player)

    # Compute per-rally stats
    for ra in rally_actions_list:
        if not ra.actions:
            continue

        frames = [a.frame for a in ra.actions]
        duration_frames = max(frames) - min(frames) if len(frames) > 1 else 0
        duration_s = duration_frames / video_fps

        action_seq = [a.action_type.value for a in ra.actions]
        has_block = any(a.action_type == ActionType.BLOCK for a in ra.actions)

        # Extended exchange: any side has 3+ consecutive contacts
        has_extended = False
        consecutive = 1
        for i in range(1, len(ra.actions)):
            if ra.actions[i].court_side == ra.actions[i - 1].court_side:
                consecutive += 1
                if consecutive >= 3:
                    has_extended = True
                    break
            else:
                consecutive = 1

        max_velocity = max((a.velocity for a in ra.actions), default=0.0)

        serve = ra.serve
        serving_side = serve.court_side if serve else "unknown"

        # Terminal action: last non-UNKNOWN action before the point ends
        terminal = _find_terminal_action(ra)
        terminal_action = terminal.action_type.value if terminal else "unknown"
        terminal_player = terminal.player_track_id if terminal else -1

        stats.rally_stats.append(RallyStats(
            rally_id=ra.rally_id,
            duration_frames=duration_frames,
            duration_seconds=duration_s,
            num_contacts=ra.num_contacts,
            action_sequence=action_seq,
            has_block=has_block,
            has_extended_exchange=has_extended,
            max_rally_velocity=max_velocity,
            serving_side=serving_side,
            terminal_action=terminal_action,
            terminal_player_track_id=terminal_player,
        ))

    # Compute match-level aggregates
    stats.total_rallies = len(stats.rally_stats)
    stats.total_contacts = sum(r.num_contacts for r in stats.rally_stats)

    if stats.rally_stats:
        durations = [r.duration_seconds for r in stats.rally_stats]
        stats.avg_rally_duration_s = float(np.mean(durations))
        stats.longest_rally_duration_s = max(durations)
        stats.avg_contacts_per_rally = stats.total_contacts / stats.total_rallies

    # Compute score progression from serve ownership
    score_progression = compute_match_scores(rally_actions_list)
    stats.score_progression = score_progression

    # Build rally_id → point_winner lookup and attribute outcomes to rally stats
    point_winner_map: dict[str, str] = {}
    if score_progression:
        last = score_progression[-1]
        stats.final_score_a = last.score_a
        stats.final_score_b = last.score_b
        for sp in score_progression:
            point_winner_map[sp.rally_id] = sp.point_winner

    # Populate point_winner on rally stats
    for rs in stats.rally_stats:
        rs.point_winner = point_winner_map.get(rs.rally_id, "unknown")

    # Attribute outcomes to players: kills, aces, errors
    # Build rally_id → RallyActions lookup
    rally_actions_map: dict[str, RallyActions] = {
        ra.rally_id: ra for ra in rally_actions_list
    }
    # Build player_stats lookup by track_id
    player_stats_map: dict[int, PlayerStats] = {
        p.track_id: p for p in stats.player_stats
    }
    _attribute_outcomes(
        stats.rally_stats, rally_actions_map,
        player_stats_map, merged_team_assignments,
    )

    # Compute per-team stats by aggregating player stats (after outcome attribution)
    for team_label in ("A", "B"):
        team_players = [p for p in stats.player_stats if p.team == team_label]
        if not team_players:
            continue
        ts = TeamStats(
            team=team_label,
            player_ids=[p.track_id for p in team_players],
            serves=sum(p.serves for p in team_players),
            receives=sum(p.receives for p in team_players),
            sets=sum(p.sets for p in team_players),
            attacks=sum(p.attacks for p in team_players),
            blocks=sum(p.blocks for p in team_players),
            digs=sum(p.digs for p in team_players),
            kills=sum(p.kills for p in team_players),
            attack_errors=sum(p.attack_errors for p in team_players),
            aces=sum(p.aces for p in team_players),
            serve_errors=sum(p.serve_errors for p in team_players),
        )
        stats.team_stats.append(ts)

    # Compute side-out rate and serve-win rate per team
    if score_progression:
        for ts in stats.team_stats:
            serving_rallies = [
                s for s in score_progression if s.serving_team == ts.team
            ]
            receiving_rallies = [
                s for s in score_progression if s.serving_team != ts.team
            ]
            serve_wins = sum(
                1 for s in serving_rallies if s.point_winner == ts.team
            )
            side_outs = sum(
                1 for s in receiving_rallies if s.point_winner == ts.team
            )
            ts.points_won = serve_wins + side_outs
            if serving_rallies:
                ts.serve_win_pct = serve_wins / len(serving_rallies)
            if receiving_rallies:
                ts.side_out_pct = side_outs / len(receiving_rallies)

        # Overall side-out rate
        total_with_winner = [
            s for s in score_progression if s.point_winner != "unknown"
        ]
        if total_with_winner:
            side_outs_total = sum(
                1 for s in total_with_winner
                if s.point_winner != s.serving_team
            )
            stats.side_out_rate = side_outs_total / len(total_with_winner)

    logger.info(
        f"Match stats: {stats.total_rallies} rallies, "
        f"{stats.total_contacts} contacts, "
        f"{len(stats.player_stats)} players tracked"
    )

    return stats
