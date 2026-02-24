"""Per-player and per-match statistics for beach volleyball.

Aggregates tracking and action classification data into meaningful
match statistics: per-player action counts, movement distance,
court zones, and overall match metrics.

Advanced stats include serve outcome classification, attack direction/power,
reception quality rating, defensive heatmaps, duo sync scores, clutch
performance, scoring runs, and more.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from rallycut.court.calibration import CourtCalibrator
    from rallycut.tracking.action_classifier import ClassifiedAction, RallyActions
    from rallycut.tracking.ball_tracker import BallPosition
    from rallycut.tracking.player_tracker import PlayerPosition

logger = logging.getLogger(__name__)


class ServeOutcome(str, Enum):
    """Outcome of a serve from the server's perspective."""

    ACE = "ace"
    ERROR = "error"
    IN_SYSTEM = "in_system"  # Receiving team gets 3 contacts (pass-set-attack)
    OUT_OF_SYSTEM = "out_of_system"  # Receiving team gets 2 contacts
    FORCED_FREE = "forced_free"  # Receiving team returns immediately (1 contact)
    UNKNOWN = "unknown"


class AttackDirection(str, Enum):
    """Direction classification for attacks."""

    LINE = "line"
    CROSS = "cross"
    CUT = "cut"
    UNKNOWN = "unknown"


class AttackPower(str, Enum):
    """Power classification for attacks."""

    POWER = "power"  # Hard driven ball
    SHOT = "shot"  # Medium pace placement
    TIP = "tip"  # Soft touch / tip
    UNKNOWN = "unknown"


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
    block_kills: int = 0  # Blocks that won the point
    # Serve outcome counts
    serve_in_system: int = 0
    serve_out_of_system: int = 0
    serve_forced_free: int = 0
    serve_pressure_score: float = 0.0
    # Reception / pass quality (0-3 scale)
    reception_rating: float = 0.0
    reception_count: int = 0
    dig_quality: float = 0.0
    dig_count: int = 0
    # Attack direction
    attacks_line: int = 0
    attacks_cross: int = 0
    attacks_power: int = 0
    attacks_shot: int = 0
    attacks_tip: int = 0
    line_kill_pct: float = 0.0
    cross_kill_pct: float = 0.0
    # Ball speed
    avg_serve_speed: float = 0.0
    max_serve_speed: float = 0.0
    avg_attack_speed: float = 0.0
    max_attack_speed: float = 0.0
    # Derived totals
    total_points: int = 0  # kills + aces + block_kills
    total_errors: int = 0  # attack_errors + serve_errors
    error_pct: float = 0.0
    # Clutch (close-score stats, score diff <= 2)
    clutch_kill_pct: float = 0.0
    # Serve targeting (court coordinates, if calibrated)
    serve_targets: list[tuple[float, float]] = field(default_factory=list)
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
            "blockKills": self.block_kills,
            "killPct": round(self.kill_pct, 3),
            "attackEfficiency": round(self.attack_efficiency, 3),
            "totalActions": self.total_actions,
            "totalPoints": self.total_points,
            "totalErrors": self.total_errors,
            "errorPct": round(self.error_pct, 3),
            "totalDistancePx": round(self.total_distance_px, 1),
            "avgSpeedPxPerFrame": round(self.avg_speed_px_per_frame, 4),
            "numFrames": self.num_frames,
            "courtSide": self.court_side,
        }
        if self.total_distance_m > 0:
            result["totalDistanceM"] = round(self.total_distance_m, 1)
        if self.position_heatmap:
            result["positionHeatmap"] = self.position_heatmap
        # Serve stats (only if player served)
        if self.serves > 0:
            result["serveInSystem"] = self.serve_in_system
            result["serveOutOfSystem"] = self.serve_out_of_system
            result["serveForcedFree"] = self.serve_forced_free
            result["servePressureScore"] = round(self.serve_pressure_score, 2)
            result["avgServeSpeed"] = round(self.avg_serve_speed, 4)
            result["maxServeSpeed"] = round(self.max_serve_speed, 4)
            if self.serve_targets:
                result["serveTargets"] = [
                    {"x": round(x, 2), "y": round(y, 2)} for x, y in self.serve_targets
                ]
        # Reception stats (only if player received)
        if self.reception_count > 0:
            result["receptionRating"] = round(self.reception_rating, 2)
            result["receptionCount"] = self.reception_count
        if self.dig_count > 0:
            result["digQuality"] = round(self.dig_quality, 2)
            result["digCount"] = self.dig_count
        # Attack stats (only if player attacked)
        if self.attacks > 0:
            result["attacksLine"] = self.attacks_line
            result["attacksCross"] = self.attacks_cross
            result["attacksPower"] = self.attacks_power
            result["attacksShot"] = self.attacks_shot
            result["attacksTip"] = self.attacks_tip
            result["avgAttackSpeed"] = round(self.avg_attack_speed, 4)
            result["maxAttackSpeed"] = round(self.max_attack_speed, 4)
            if self.attacks_line > 0:
                result["lineKillPct"] = round(self.line_kill_pct, 3)
            if self.attacks_cross > 0:
                result["crossKillPct"] = round(self.cross_kill_pct, 3)
        # Clutch (only if meaningful)
        if self.clutch_kill_pct > 0:
            result["clutchKillPct"] = round(self.clutch_kill_pct, 3)
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
    # Advanced per-rally stats
    serve_outcome: str = "unknown"  # ServeOutcome value
    net_crossings: int = 0  # Number of side changes

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
            "netCrossings": self.net_crossings,
        }
        if self.terminal_action != "unknown":
            d["terminalAction"] = self.terminal_action
            d["terminalPlayerTrackId"] = self.terminal_player_track_id
        if self.point_winner != "unknown":
            d["pointWinner"] = self.point_winner
        if self.serve_outcome != "unknown":
            d["serveOutcome"] = self.serve_outcome
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
    block_kills: int = 0
    # Long rally stats
    long_rally_win_pct_8: float = 0.0  # Win % in rallies with 8+ contacts
    long_rally_win_pct_10: float = 0.0  # Win % in rallies with 10+ contacts
    # Defensive heatmap (4x4 grid on defending half-court)
    defensive_heatmap: list[list[float]] = field(default_factory=list)
    defensive_heatmap_calibrated: bool = False
    # Efficiency breakdowns
    sideout_attack_efficiency: float = 0.0  # Attack efficiency after receiving
    transition_attack_efficiency: float = 0.0  # Attack efficiency after digging
    # Free ball conversion
    free_ball_conversion_pct: float = 0.0
    # Scoring runs
    longest_scoring_run: int = 0
    avg_scoring_run: float = 0.0
    # Clutch (close-score, diff <= 2)
    clutch_side_out_pct: float = 0.0

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
        d: dict[str, Any] = {
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
            "blockKills": self.block_kills,
            "killPct": round(self.kill_pct, 3),
            "attackEfficiency": round(self.attack_efficiency, 3),
            "totalActions": self.total_actions,
            "pointsWon": self.points_won,
            "sideOutPct": round(self.side_out_pct, 3),
            "serveWinPct": round(self.serve_win_pct, 3),
            "longRallyWinPct8": round(self.long_rally_win_pct_8, 3),
            "longRallyWinPct10": round(self.long_rally_win_pct_10, 3),
            "sideoutAttackEfficiency": round(self.sideout_attack_efficiency, 3),
            "transitionAttackEfficiency": round(self.transition_attack_efficiency, 3),
            "longestScoringRun": self.longest_scoring_run,
            "avgScoringRun": round(self.avg_scoring_run, 1),
        }
        if self.defensive_heatmap:
            d["defensiveHeatmap"] = self.defensive_heatmap
            d["defensiveHeatmapCalibrated"] = self.defensive_heatmap_calibrated
        if self.free_ball_conversion_pct > 0:
            d["freeBallConversionPct"] = round(self.free_ball_conversion_pct, 3)
        if self.clutch_side_out_pct > 0:
            d["clutchSideOutPct"] = round(self.clutch_side_out_pct, 3)
        return d


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
class DuoStats:
    """Cross-rally statistics for a stable player identity (1-4).

    Requires match analysis (match-players) to map rally-specific track IDs
    to consistent player identities across the match.
    """

    player_id: int  # Stable player ID (1-4) from match analysis
    team: str = "unknown"  # "A" or "B"
    set_to_kill_pct: float = 0.0  # % of this player's sets leading to kills
    attack_win_pct: float = 0.0  # % of rallies won when this player attacks terminally
    sets_leading_to_kill: int = 0
    total_sets: int = 0
    attacks_that_won: int = 0
    total_terminal_attacks: int = 0

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "playerId": self.player_id,
            "team": self.team,
            "setToKillPct": round(self.set_to_kill_pct, 3),
            "attackWinPct": round(self.attack_win_pct, 3),
            "setsLeadingToKill": self.sets_leading_to_kill,
            "totalSets": self.total_sets,
            "attacksThatWon": self.attacks_that_won,
            "totalTerminalAttacks": self.total_terminal_attacks,
        }
        return d


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
    # Duo sync stats (cross-rally player identity)
    duo_stats: list[DuoStats] = field(default_factory=list)
    # Match-level aggregates
    total_rallies: int = 0
    total_contacts: int = 0
    avg_rally_duration_s: float = 0.0
    longest_rally_duration_s: float = 0.0
    avg_contacts_per_rally: float = 0.0
    side_out_rate: float = 0.0  # % of rallies won by receiving team
    final_score_a: int = 0
    final_score_b: int = 0
    momentum_shifts: int = 0  # Times scoring momentum changes teams
    # Calibration
    is_calibrated: bool = False
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
            "isCalibrated": self.is_calibrated,
            "playerStats": [p.to_dict() for p in self.player_stats],
            "rallyStats": [r.to_dict() for r in self.rally_stats],
        }
        if self.team_stats:
            d["teamStats"] = [t.to_dict() for t in self.team_stats]
        if self.score_progression:
            d["scoreProgression"] = [s.to_dict() for s in self.score_progression]
            d["finalScoreA"] = self.final_score_a
            d["finalScoreB"] = self.final_score_b
            d["momentumShifts"] = self.momentum_shifts
        if self.duo_stats:
            d["duoStats"] = [ds.to_dict() for ds in self.duo_stats]
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
    """Attribute kills, aces, block kills, and errors to individual players.

    Logic:
    - If terminal action is ATTACK and the attacker's team won → kill
    - If terminal action is ATTACK and the attacker's team lost → attack error
    - If terminal action is SERVE and the server's team won → ace
    - If terminal action is SERVE and the server's team lost → serve error
    - If terminal action is BLOCK and the blocker's team won → block kill
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
        elif rs.terminal_action == "block":
            if won:
                player.block_kills += 1


# ---------------------------------------------------------------------------
# Court projection helpers
# ---------------------------------------------------------------------------


def project_to_court(
    x: float,
    y: float,
    calibrator: CourtCalibrator | None,
    video_width: int,
    video_height: int,
) -> tuple[float, float] | None:
    """Project normalized image coordinates to court coordinates (meters).

    Uses calibrator homography when available, returns None on failure.
    """
    if calibrator is None or not calibrator.is_calibrated:
        return None
    try:
        return calibrator.image_to_court((x, y), video_width, video_height)
    except (RuntimeError, ValueError):
        return None


def estimate_court_projection(
    x: float,
    y: float,
    court_split_y: float,
    player_x_bounds: tuple[float, float] = (0.05, 0.95),
) -> tuple[float, float]:
    """Estimate court coordinates without calibration using affine mapping.

    Maps image x to court width [0, 8m] and image y to court length [0, 16m].
    Net is at court_split_y in image space → 8m in court space.
    Rough approximation without perspective correction.

    Returns:
        (court_x_m, court_y_m) in meters.
    """
    from rallycut.court.calibration import COURT_LENGTH, COURT_WIDTH

    x_min, x_max = player_x_bounds
    x_range = max(x_max - x_min, 0.1)
    court_x = ((x - x_min) / x_range) * COURT_WIDTH
    court_x = max(0.0, min(COURT_WIDTH, court_x))

    # Map y: court_split_y → 8m (net), 0.0 → 0m (far baseline), 1.0 → 16m (near baseline)
    if court_split_y > 0.01:
        if y <= court_split_y:
            # Far side: [0, court_split_y] → [0, 8]
            court_y = (y / court_split_y) * (COURT_LENGTH / 2)
        else:
            # Near side: [court_split_y, 1.0] → [8, 16]
            near_range = max(1.0 - court_split_y, 0.1)
            court_y = (COURT_LENGTH / 2) + (
                (y - court_split_y) / near_range
            ) * (COURT_LENGTH / 2)
    else:
        court_y = y * COURT_LENGTH

    court_y = max(0.0, min(COURT_LENGTH, court_y))
    return court_x, court_y


def _get_court_split_y(rally_actions_list: list[RallyActions]) -> float:
    """Estimate net Y position from action court_side transitions.

    Returns the average Y of actions near the net (transitions between sides).
    Falls back to 0.5 if insufficient data.
    """
    near_ys: list[float] = []
    far_ys: list[float] = []
    for ra in rally_actions_list:
        for a in ra.actions:
            if a.court_side == "near" and a.ball_y > 0:
                near_ys.append(a.ball_y)
            elif a.court_side == "far" and a.ball_y > 0:
                far_ys.append(a.ball_y)

    if near_ys and far_ys:
        # Net is approximately between the nearest far-side and nearest near-side actions
        return (min(near_ys) + max(far_ys)) / 2

    return 0.5


def _get_player_x_bounds(
    player_positions: list[PlayerPosition],
) -> tuple[float, float]:
    """Get 3rd/97th percentile X bounds of player positions."""
    if not player_positions:
        return (0.05, 0.95)
    xs = [p.x for p in player_positions]
    return (float(np.percentile(xs, 3)), float(np.percentile(xs, 97)))


# ---------------------------------------------------------------------------
# Serve outcome classification
# ---------------------------------------------------------------------------


def _classify_serve_outcome(
    ra: RallyActions,
    point_winner: str,
) -> ServeOutcome:
    """Classify the outcome of a serve in a rally.

    Logic:
    - ACE: serve is terminal action + server's team won point
    - ERROR: serve is terminal action + server's team lost point
    - IN_SYSTEM: 3+ contacts on receiving side after serve
    - OUT_OF_SYSTEM: 2 contacts on receiving side
    - FORCED_FREE: 1 contact on receiving side (immediate return)
    """
    from rallycut.tracking.action_classifier import ActionType

    serve = ra.serve
    if not serve:
        return ServeOutcome.UNKNOWN

    # Check if serve is terminal action (ACE or ERROR)
    terminal = _find_terminal_action(ra)
    if terminal and terminal.action_type == ActionType.SERVE:
        if serve.team != "unknown" and point_winner != "unknown":
            return ServeOutcome.ACE if serve.team == point_winner else ServeOutcome.ERROR
        return ServeOutcome.UNKNOWN

    # Count contacts on receiving side before ball crosses back
    serve_side = serve.court_side
    receiving_side = "far" if serve_side == "near" else "near"

    receiving_contacts = 0
    found_receive = False
    for action in ra.actions:
        if action.action_type == ActionType.SERVE:
            continue
        if action.court_side == receiving_side:
            receiving_contacts += 1
            found_receive = True
        elif found_receive:
            # Ball crossed back to serving side
            break

    if receiving_contacts >= 3:
        return ServeOutcome.IN_SYSTEM
    elif receiving_contacts == 2:
        return ServeOutcome.OUT_OF_SYSTEM
    elif receiving_contacts == 1:
        return ServeOutcome.FORCED_FREE
    return ServeOutcome.UNKNOWN


def _compute_serve_pressure_score(ps: PlayerStats) -> float:
    """Compute weighted serve pressure score for a player.

    Weights: ace=3, forced_free=2, out_of_system=1.5, in_system=0, error=-1.
    """
    total_outcomes = (
        ps.aces + ps.serve_errors + ps.serve_in_system
        + ps.serve_out_of_system + ps.serve_forced_free
    )
    if total_outcomes == 0:
        return 0.0
    weighted = (
        ps.aces * 3.0
        + ps.serve_forced_free * 2.0
        + ps.serve_out_of_system * 1.5
        + ps.serve_in_system * 0.0
        + ps.serve_errors * -1.0
    )
    return weighted / total_outcomes


# ---------------------------------------------------------------------------
# Reception / pass quality
# ---------------------------------------------------------------------------


def _compute_reception_quality(ra: RallyActions) -> float:
    """Rate reception quality after serve: 3.0=perfect, 2.0=OK, 1.0=poor, 0.0=ace.

    Counts contacts on the receiving side:
    - 3+ contacts (receive-set-attack) → 3.0
    - 2 contacts → 2.0
    - 1 contact → 1.0
    - 0 contacts (ace) → 0.0
    Returns -1.0 if quality cannot be determined.
    """
    from rallycut.tracking.action_classifier import ActionType

    serve = ra.serve
    if not serve:
        return -1.0

    receiving_side = "far" if serve.court_side == "near" else "near"
    contacts = 0
    for action in ra.actions:
        if action.action_type == ActionType.SERVE:
            continue
        if action.court_side == receiving_side:
            contacts += 1
        elif contacts > 0:
            break

    return float(min(contacts, 3))


def _compute_dig_quality(ra: RallyActions, dig_action: ClassifiedAction) -> float:
    """Rate dig quality: count contacts on the digger's side after the dig.

    Same 0-3 scale as reception quality.
    Returns -1.0 if quality cannot be determined.
    """
    dig_side = dig_action.court_side
    if dig_side not in ("near", "far"):
        return -1.0

    # Count contacts on dig_side starting from the dig
    contacts = 0
    found_dig = False
    for action in ra.actions:
        if action is dig_action:
            found_dig = True
            contacts = 1
            continue
        if not found_dig:
            continue
        if action.court_side == dig_side:
            contacts += 1
        else:
            break

    return float(min(contacts, 3))


# ---------------------------------------------------------------------------
# Attack direction and power classification
# ---------------------------------------------------------------------------


def _classify_attack(
    attack: ClassifiedAction,
    ball_positions: list[BallPosition] | None,
    calibrator: CourtCalibrator | None,
    video_width: int,
    video_height: int,
    court_split_y: float,
    player_x_bounds: tuple[float, float],
) -> tuple[AttackDirection, AttackPower]:
    """Classify attack direction (line/cross/cut) and power (power/shot/tip).

    Uses ball trajectory in the 3-15 frames after attack contact.
    Returns (direction, power).
    """
    # Power classification from velocity (always available)
    velocity = attack.velocity
    if velocity > 0.04:
        power = AttackPower.POWER
    elif velocity > 0.02:
        power = AttackPower.SHOT
    else:
        power = AttackPower.TIP

    # Direction needs ball positions
    if not ball_positions:
        return AttackDirection.UNKNOWN, power

    attack_frame = attack.frame
    post_attack = [
        bp for bp in ball_positions
        if attack_frame < bp.frame_number <= attack_frame + 15
        and bp.confidence > 0.1
    ]

    if len(post_attack) < 3:
        return AttackDirection.UNKNOWN, power

    start_x, start_y = attack.ball_x, attack.ball_y
    end = post_attack[-1]
    end_x, end_y = end.x, end.y

    # Project to court coordinates
    court_start = project_to_court(start_x, start_y, calibrator, video_width, video_height)
    court_end = project_to_court(end_x, end_y, calibrator, video_width, video_height)

    if court_start and court_end:
        dx = court_end[0] - court_start[0]
        dy = court_end[1] - court_start[1]
    else:
        # Fallback: estimate from normalized coords
        court_start_est = estimate_court_projection(
            start_x, start_y, court_split_y, player_x_bounds,
        )
        court_end_est = estimate_court_projection(
            end_x, end_y, court_split_y, player_x_bounds,
        )
        dx = court_end_est[0] - court_start_est[0]
        dy = court_end_est[1] - court_start_est[1]

    abs_dx = abs(dx)
    abs_dy = abs(dy)

    if abs_dx < 2.0 and abs_dy > 3.0:
        direction = AttackDirection.LINE
    elif abs_dx > 2.0:
        direction = AttackDirection.CROSS
    else:
        direction = AttackDirection.CUT

    return direction, power


# ---------------------------------------------------------------------------
# Net crossings
# ---------------------------------------------------------------------------


def _count_net_crossings(ra: RallyActions) -> int:
    """Count the number of times the ball crosses the net in a rally."""
    crossings = 0
    prev_side: str | None = None
    for action in ra.actions:
        if action.court_side in ("near", "far"):
            if prev_side is not None and action.court_side != prev_side:
                crossings += 1
            prev_side = action.court_side
    return crossings


# ---------------------------------------------------------------------------
# Scoring runs and momentum
# ---------------------------------------------------------------------------


def _compute_scoring_runs(
    score_progression: list[RallyScoreState],
) -> tuple[dict[str, list[int]], int]:
    """Compute scoring runs per team and momentum shifts.

    Returns:
        (runs_by_team, momentum_shifts) where runs_by_team maps team to
        list of consecutive-point run lengths.
    """
    runs: dict[str, list[int]] = {"A": [], "B": []}
    current_team: str | None = None
    current_run = 0
    momentum_shifts = 0

    for sp in score_progression:
        if sp.point_winner in ("A", "B"):
            if sp.point_winner == current_team:
                current_run += 1
            else:
                if current_team is not None and current_run > 0:
                    runs[current_team].append(current_run)
                    momentum_shifts += 1
                current_team = sp.point_winner
                current_run = 1

    # Flush last run
    if current_team is not None and current_run > 0:
        runs[current_team].append(current_run)

    # First run doesn't count as a momentum shift
    if momentum_shifts > 0:
        momentum_shifts -= 1

    return runs, momentum_shifts


# ---------------------------------------------------------------------------
# Long rally win %
# ---------------------------------------------------------------------------


def _compute_long_rally_win_pct(
    rally_stats: list[RallyStats],
    team: str,
    threshold: int,
) -> float:
    """Compute win % for rallies with >= threshold contacts."""
    long_rallies = [
        rs for rs in rally_stats
        if rs.num_contacts >= threshold and rs.point_winner != "unknown"
    ]
    if not long_rallies:
        return 0.0
    wins = sum(1 for rs in long_rallies if rs.point_winner == team)
    return wins / len(long_rallies)


# ---------------------------------------------------------------------------
# Defensive heatmap
# ---------------------------------------------------------------------------


def _compute_defensive_heatmap(
    rally_stats: list[RallyStats],
    rally_actions_map: dict[str, RallyActions],
    defending_team: str,
    calibrator: CourtCalibrator | None,
    video_width: int,
    video_height: int,
    court_split_y: float,
    player_x_bounds: tuple[float, float],
) -> tuple[list[list[float]], bool]:
    """Compute a 4x4 defensive heatmap for a team's half-court.

    Records court positions of attacks that scored against the defending team.
    Each cell is ~2m x 2m.

    Returns:
        (heatmap, is_calibrated)
    """
    from rallycut.court.calibration import COURT_LENGTH, COURT_WIDTH

    grid = np.zeros((4, 4), dtype=np.float64)
    is_calibrated = calibrator is not None and calibrator.is_calibrated

    # Determine which half-court to map
    # Defending team: "A" is near (y > 8m), "B" is far (y < 8m)
    half_y_start = (COURT_LENGTH / 2) if defending_team == "A" else 0.0
    half_y_end = COURT_LENGTH if defending_team == "A" else (COURT_LENGTH / 2)

    for rs in rally_stats:
        if rs.point_winner == "unknown":
            continue
        # Only count attacks that scored AGAINST the defending team
        if rs.point_winner == defending_team:
            continue
        if rs.terminal_action != "attack":
            continue

        ra = rally_actions_map.get(rs.rally_id)
        if not ra:
            continue

        terminal = _find_terminal_action(ra)
        if terminal is None:
            continue

        # Get court position of the attack landing
        court_pos = project_to_court(
            terminal.ball_x, terminal.ball_y, calibrator, video_width, video_height,
        )
        if court_pos is None:
            court_pos = estimate_court_projection(
                terminal.ball_x, terminal.ball_y, court_split_y, player_x_bounds,
            )

        cx, cy = court_pos
        # Map to grid on defending half-court
        if half_y_start <= cy <= half_y_end:
            gx = min(int(cx / COURT_WIDTH * 4), 3)
            gy = min(int((cy - half_y_start) / (COURT_LENGTH / 2) * 4), 3)
            gx = max(0, gx)
            gy = max(0, gy)
            grid[gy, gx] += 1

    total = grid.sum()
    if total > 0:
        grid /= total

    return grid.tolist(), is_calibrated


# ---------------------------------------------------------------------------
# Duo sync score (cross-rally player identity)
# ---------------------------------------------------------------------------


def _compute_duo_sync(
    rally_actions_list: list[RallyActions],
    rally_stats: list[RallyStats],
    match_analysis: dict[str, Any] | None,
    team_assignments: dict[int, int],
) -> list[DuoStats]:
    """Compute duo sync scores using cross-rally player identity mapping.

    Requires match_analysis from match-players command to map rally-specific
    track IDs to stable player IDs (1-4).
    """
    from rallycut.tracking.action_classifier import ActionType

    if not match_analysis or "rallies" not in match_analysis:
        return []

    # Build rally_id → trackToPlayer mapping
    rally_player_map: dict[str, dict[int, int]] = {}
    for rally_info in match_analysis["rallies"]:
        rid = rally_info.get("rallyId", "")
        ttp = rally_info.get("trackToPlayer", {})
        rally_player_map[rid] = {int(k): v for k, v in ttp.items()}

    # Build rally_id → point_winner lookup
    point_winner_map: dict[str, str] = {
        rs.rally_id: rs.point_winner for rs in rally_stats
    }

    # Per-player accumulators
    player_sets: dict[int, int] = {}  # player_id → total sets
    player_sets_to_kill: dict[int, int] = {}  # player_id → sets leading to kills
    player_terminal_attacks: dict[int, int] = {}  # player_id → terminal attacks
    player_attacks_won: dict[int, int] = {}  # player_id → terminal attacks that won
    player_teams: dict[int, str] = {}

    for ra in rally_actions_list:
        track_to_player = rally_player_map.get(ra.rally_id, {})
        if not track_to_player:
            continue

        point_winner = point_winner_map.get(ra.rally_id, "unknown")
        terminal = _find_terminal_action(ra)

        for i, action in enumerate(ra.actions):
            pid = track_to_player.get(action.player_track_id)
            if pid is None:
                continue

            # Track team for this player
            team_int = team_assignments.get(action.player_track_id)
            if team_int is not None:
                player_teams[pid] = "A" if team_int == 0 else "B"

            if action.action_type == ActionType.SET:
                player_sets[pid] = player_sets.get(pid, 0) + 1
                # Check if the next action is an attack that was a kill
                if i + 1 < len(ra.actions):
                    next_action = ra.actions[i + 1]
                    if next_action.action_type == ActionType.ATTACK:
                        if terminal and terminal is next_action and point_winner != "unknown":
                            # Terminal player's team
                            t_int = team_assignments.get(terminal.player_track_id)
                            t_team = "A" if t_int == 0 else "B" if t_int == 1 else "unknown"
                            if t_team == point_winner:
                                player_sets_to_kill[pid] = (
                                    player_sets_to_kill.get(pid, 0) + 1
                                )

            if action.action_type == ActionType.ATTACK:
                if terminal and terminal is action:
                    player_terminal_attacks[pid] = (
                        player_terminal_attacks.get(pid, 0) + 1
                    )
                    t_int = team_assignments.get(action.player_track_id)
                    t_team = "A" if t_int == 0 else "B" if t_int == 1 else "unknown"
                    if t_team == point_winner:
                        player_attacks_won[pid] = player_attacks_won.get(pid, 0) + 1

    # Build DuoStats
    all_pids = set(player_sets) | set(player_terminal_attacks)
    duo_stats: list[DuoStats] = []
    for pid in sorted(all_pids):
        total_s = player_sets.get(pid, 0)
        s_to_k = player_sets_to_kill.get(pid, 0)
        total_ta = player_terminal_attacks.get(pid, 0)
        ta_won = player_attacks_won.get(pid, 0)

        ds = DuoStats(
            player_id=pid,
            team=player_teams.get(pid, "unknown"),
            sets_leading_to_kill=s_to_k,
            total_sets=total_s,
            set_to_kill_pct=s_to_k / total_s if total_s > 0 else 0.0,
            attacks_that_won=ta_won,
            total_terminal_attacks=total_ta,
            attack_win_pct=ta_won / total_ta if total_ta > 0 else 0.0,
        )
        duo_stats.append(ds)

    return duo_stats


# ---------------------------------------------------------------------------
# Free ball conversion
# ---------------------------------------------------------------------------


def _is_free_ball_rally(ra: RallyActions) -> tuple[bool, str]:
    """Detect if a rally involved a free ball.

    A free ball occurs when one side has only 1 contact before the ball
    crosses back to the other side (excluding serve-receive).

    Returns:
        (is_free_ball, receiving_team) — team that received the free ball.
    """
    from rallycut.tracking.action_classifier import ActionType

    actions = [a for a in ra.actions if a.action_type != ActionType.UNKNOWN]
    if len(actions) < 3:
        return False, "unknown"

    # Skip the serve exchange — look at subsequent side changes
    prev_side: str | None = None
    side_contacts = 0

    for i, action in enumerate(actions):
        if action.action_type == ActionType.SERVE:
            prev_side = action.court_side
            side_contacts = 0
            continue

        if prev_side is None:
            prev_side = action.court_side
            side_contacts = 1
            continue

        if action.court_side == prev_side:
            side_contacts += 1
        else:
            # Side changed — check if previous side had only 1 contact
            # (and it wasn't the serve-receive exchange)
            if side_contacts == 1 and i >= 3:
                receiving_team = (
                    "A" if action.court_side == "near"
                    else ("B" if action.court_side == "far" else "unknown")
                )
                return True, receiving_team
            prev_side = action.court_side
            side_contacts = 1

    return False, "unknown"


# ---------------------------------------------------------------------------
# Clutch performance (close-score stats)
# ---------------------------------------------------------------------------


def _get_close_score_rally_ids(
    score_progression: list[RallyScoreState],
    max_diff: int = 2,
) -> set[str]:
    """Get rally IDs where score differential is <= max_diff."""
    close_rally_ids: set[str] = set()
    for sp in score_progression:
        # Use score BEFORE this rally's point (subtract the point just awarded)
        score_a = sp.score_a
        score_b = sp.score_b
        if sp.point_winner == "A":
            score_a -= 1
        elif sp.point_winner == "B":
            score_b -= 1
        if abs(score_a - score_b) <= max_diff:
            close_rally_ids.add(sp.rally_id)
    return close_rally_ids


# ---------------------------------------------------------------------------
# Transition vs side-out attack efficiency
# ---------------------------------------------------------------------------


def _classify_attack_context(
    ra: RallyActions,
    attack_action: ClassifiedAction,
) -> str:
    """Classify whether an attack is side-out or transition.

    - Side-out: attack preceded by receive (after serve)
    - Transition: attack preceded by dig (after opponent's attack)
    Returns "sideout", "transition", or "unknown".
    """
    from rallycut.tracking.action_classifier import ActionType

    attack_side = attack_action.court_side
    # Find the attack's index in the action list
    idx = -1
    for i, action in enumerate(ra.actions):
        if action is attack_action:
            idx = i
            break
    if idx < 0:
        return "unknown"

    # Walk backwards from attack to find what started this possession
    for i in range(idx - 1, -1, -1):
        prev = ra.actions[i]
        if prev.court_side != attack_side:
            break
        if prev.action_type == ActionType.RECEIVE:
            return "sideout"
        if prev.action_type == ActionType.DIG:
            return "transition"

    return "unknown"


# ---------------------------------------------------------------------------
# Ball speed aggregation
# ---------------------------------------------------------------------------


def _aggregate_ball_speeds(
    rally_actions_list: list[RallyActions],
    player_stats_map: dict[int, PlayerStats],
) -> None:
    """Aggregate serve and attack speeds per player from action velocities."""
    from rallycut.tracking.action_classifier import ActionType

    serve_speeds: dict[int, list[float]] = {}
    attack_speeds: dict[int, list[float]] = {}

    for ra in rally_actions_list:
        for action in ra.actions:
            pid = action.player_track_id
            if pid < 0:
                continue
            if action.velocity <= 0:
                continue
            if action.action_type == ActionType.SERVE:
                serve_speeds.setdefault(pid, []).append(action.velocity)
            elif action.action_type == ActionType.ATTACK:
                attack_speeds.setdefault(pid, []).append(action.velocity)

    for tid, ps in player_stats_map.items():
        if tid in serve_speeds:
            speeds = serve_speeds[tid]
            ps.avg_serve_speed = float(np.mean(speeds))
            ps.max_serve_speed = max(speeds)
        if tid in attack_speeds:
            speeds = attack_speeds[tid]
            ps.avg_attack_speed = float(np.mean(speeds))
            ps.max_attack_speed = max(speeds)


def compute_match_stats(
    rally_actions_list: list[RallyActions],
    player_positions: list[PlayerPosition],
    video_fps: float = 30.0,
    video_width: int = 1920,
    video_height: int = 1080,
    calibrator: CourtCalibrator | None = None,
    ball_positions_map: dict[str, list[BallPosition]] | None = None,
    match_analysis: dict[str, Any] | None = None,
) -> MatchStats:
    """Compute comprehensive match statistics.

    Args:
        rally_actions_list: Classified actions for each rally.
        player_positions: All player tracking positions across the match.
        video_fps: Video frame rate.
        video_width: Video width.
        video_height: Video height.
        calibrator: Optional court calibrator for real-world distances.
        ball_positions_map: Ball positions per rally_id (for attack/serve analysis).
        match_analysis: Cross-rally player identity from match-players command.

    Returns:
        MatchStats with per-player and per-rally statistics.
    """
    from rallycut.tracking.action_classifier import ActionType

    is_calibrated = calibrator is not None and calibrator.is_calibrated
    stats = MatchStats(
        video_fps=video_fps,
        video_width=video_width,
        video_height=video_height,
        is_calibrated=is_calibrated,
    )

    # Precompute court geometry for fallback projection
    court_split_y = _get_court_split_y(rally_actions_list)
    player_x_bounds = _get_player_x_bounds(player_positions)

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
            net_crossings=_count_net_crossings(ra),
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
            block_kills=sum(p.block_kills for p in team_players),
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

    # -----------------------------------------------------------------------
    # Advanced stats: serve outcomes, reception quality, attack analysis,
    # ball speed, long rally win %, scoring runs, duo sync, etc.
    # -----------------------------------------------------------------------

    # 1. Serve outcome classification + serve stats on players
    rally_stats_by_id: dict[str, RallyStats] = {
        r.rally_id: r for r in stats.rally_stats
    }
    for ra in rally_actions_list:
        pw = point_winner_map.get(ra.rally_id, "unknown")
        outcome = _classify_serve_outcome(ra, pw)

        # Set serve_outcome on the matching RallyStats
        rally_rs = rally_stats_by_id.get(ra.rally_id)
        if rally_rs:
            rally_rs.serve_outcome = outcome.value

        # Accumulate on server's PlayerStats
        serve = ra.serve
        if serve and serve.player_track_id >= 0:
            ps = player_stats_map.get(serve.player_track_id)
            if ps:
                if outcome == ServeOutcome.IN_SYSTEM:
                    ps.serve_in_system += 1
                elif outcome == ServeOutcome.OUT_OF_SYSTEM:
                    ps.serve_out_of_system += 1
                elif outcome == ServeOutcome.FORCED_FREE:
                    ps.serve_forced_free += 1

        # Serve targeting: record receive position projected to court
        if serve and outcome not in (ServeOutcome.ACE, ServeOutcome.ERROR,
                                     ServeOutcome.UNKNOWN):
            # Find the receive action
            for a in ra.actions:
                if a.action_type == ActionType.RECEIVE:
                    court_pos = project_to_court(
                        a.ball_x, a.ball_y, calibrator, video_width, video_height,
                    )
                    if court_pos is None:
                        court_pos = estimate_court_projection(
                            a.ball_x, a.ball_y, court_split_y, player_x_bounds,
                        )
                    if serve.player_track_id >= 0:
                        server_ps = player_stats_map.get(serve.player_track_id)
                        if server_ps:
                            server_ps.serve_targets.append(court_pos)
                    break

    # Compute serve pressure scores
    for ps in stats.player_stats:
        ps.serve_pressure_score = _compute_serve_pressure_score(ps)

    # 2. Reception quality + dig quality
    reception_ratings: dict[int, list[float]] = {}
    dig_ratings: dict[int, list[float]] = {}
    for ra in rally_actions_list:
        # Reception quality (receive after serve)
        quality = _compute_reception_quality(ra)
        if quality >= 0:
            for a in ra.actions:
                if a.action_type == ActionType.RECEIVE and a.player_track_id >= 0:
                    reception_ratings.setdefault(a.player_track_id, []).append(quality)
                    break

        # Dig quality
        for a in ra.actions:
            if a.action_type == ActionType.DIG and a.player_track_id >= 0:
                dq = _compute_dig_quality(ra, a)
                if dq >= 0:
                    dig_ratings.setdefault(a.player_track_id, []).append(dq)

    for tid, ps in player_stats_map.items():
        if tid in reception_ratings:
            ratings = reception_ratings[tid]
            ps.reception_rating = float(np.mean(ratings))
            ps.reception_count = len(ratings)
        if tid in dig_ratings:
            ratings = dig_ratings[tid]
            ps.dig_quality = float(np.mean(ratings))
            ps.dig_count = len(ratings)

    # 3. Attack direction + power classification
    line_kills: dict[int, int] = {}
    cross_kills: dict[int, int] = {}
    line_attacks: dict[int, int] = {}
    cross_attacks: dict[int, int] = {}
    for ra in rally_actions_list:
        ball_pos = (ball_positions_map or {}).get(ra.rally_id)
        pw = point_winner_map.get(ra.rally_id, "unknown")
        terminal = _find_terminal_action(ra)

        for a in ra.actions:
            if a.action_type != ActionType.ATTACK or a.player_track_id < 0:
                continue

            direction, power = _classify_attack(
                a, ball_pos, calibrator, video_width, video_height,
                court_split_y, player_x_bounds,
            )
            ps = player_stats_map.get(a.player_track_id)
            if ps is None:
                continue

            if direction == AttackDirection.LINE:
                ps.attacks_line += 1
                line_attacks[a.player_track_id] = line_attacks.get(a.player_track_id, 0) + 1
            elif direction == AttackDirection.CROSS:
                ps.attacks_cross += 1
                cross_attacks[a.player_track_id] = cross_attacks.get(a.player_track_id, 0) + 1

            if power == AttackPower.POWER:
                ps.attacks_power += 1
            elif power == AttackPower.SHOT:
                ps.attacks_shot += 1
            elif power == AttackPower.TIP:
                ps.attacks_tip += 1

            # Track kills by direction
            if terminal and terminal is a and pw != "unknown":
                t_int = merged_team_assignments.get(a.player_track_id)
                t_team = "A" if t_int == 0 else "B" if t_int == 1 else "unknown"
                if t_team == pw:  # It was a kill
                    if direction == AttackDirection.LINE:
                        line_kills[a.player_track_id] = (
                            line_kills.get(a.player_track_id, 0) + 1
                        )
                    elif direction == AttackDirection.CROSS:
                        cross_kills[a.player_track_id] = (
                            cross_kills.get(a.player_track_id, 0) + 1
                        )

    # Compute direction kill percentages
    for tid, ps in player_stats_map.items():
        la = line_attacks.get(tid, 0)
        ca = cross_attacks.get(tid, 0)
        if la > 0:
            ps.line_kill_pct = line_kills.get(tid, 0) / la
        if ca > 0:
            ps.cross_kill_pct = cross_kills.get(tid, 0) / ca

    # 4. Ball speed aggregation
    _aggregate_ball_speeds(rally_actions_list, player_stats_map)

    # 5. Derived totals on players
    for ps in stats.player_stats:
        ps.total_points = ps.kills + ps.aces + ps.block_kills
        ps.total_errors = ps.attack_errors + ps.serve_errors
        if ps.total_actions > 0:
            ps.error_pct = ps.total_errors / ps.total_actions

    # 6. Long rally win % per team
    for ts in stats.team_stats:
        ts.long_rally_win_pct_8 = _compute_long_rally_win_pct(
            stats.rally_stats, ts.team, 8,
        )
        ts.long_rally_win_pct_10 = _compute_long_rally_win_pct(
            stats.rally_stats, ts.team, 10,
        )

    # 7. Defensive heatmaps per team
    for ts in stats.team_stats:
        heatmap, hm_calibrated = _compute_defensive_heatmap(
            stats.rally_stats, rally_actions_map, ts.team,
            calibrator, video_width, video_height,
            court_split_y, player_x_bounds,
        )
        ts.defensive_heatmap = heatmap
        ts.defensive_heatmap_calibrated = hm_calibrated

    # 8. Scoring runs + momentum
    if score_progression:
        runs_by_team, momentum_shifts = _compute_scoring_runs(score_progression)
        stats.momentum_shifts = momentum_shifts
        for ts in stats.team_stats:
            team_runs = runs_by_team.get(ts.team, [])
            if team_runs:
                ts.longest_scoring_run = max(team_runs)
                ts.avg_scoring_run = float(np.mean(team_runs))

    # 9. Free ball conversion per team
    free_ball_opportunities: dict[str, int] = {"A": 0, "B": 0}
    free_ball_conversions: dict[str, int] = {"A": 0, "B": 0}
    for ra in rally_actions_list:
        is_free, receiving_team = _is_free_ball_rally(ra)
        if is_free and receiving_team in ("A", "B"):
            free_ball_opportunities[receiving_team] += 1
            pw = point_winner_map.get(ra.rally_id, "unknown")
            if pw == receiving_team:
                free_ball_conversions[receiving_team] += 1
    for ts in stats.team_stats:
        opp = free_ball_opportunities.get(ts.team, 0)
        if opp > 0:
            ts.free_ball_conversion_pct = free_ball_conversions.get(ts.team, 0) / opp

    # 10. Transition vs side-out attack efficiency per team
    sideout_kills: dict[str, int] = {"A": 0, "B": 0}
    sideout_errors: dict[str, int] = {"A": 0, "B": 0}
    sideout_attacks: dict[str, int] = {"A": 0, "B": 0}
    transition_kills: dict[str, int] = {"A": 0, "B": 0}
    transition_errors: dict[str, int] = {"A": 0, "B": 0}
    transition_attacks: dict[str, int] = {"A": 0, "B": 0}

    for ra in rally_actions_list:
        pw = point_winner_map.get(ra.rally_id, "unknown")
        terminal = _find_terminal_action(ra)

        for a in ra.actions:
            if a.action_type != ActionType.ATTACK or a.player_track_id < 0:
                continue
            t_int = merged_team_assignments.get(a.player_track_id)
            if t_int is None:
                continue
            t_team = "A" if t_int == 0 else "B"

            context = _classify_attack_context(ra, a)
            if context == "sideout":
                sideout_attacks[t_team] += 1
                if terminal and terminal is a and pw != "unknown":
                    if t_team == pw:
                        sideout_kills[t_team] += 1
                    else:
                        sideout_errors[t_team] += 1
            elif context == "transition":
                transition_attacks[t_team] += 1
                if terminal and terminal is a and pw != "unknown":
                    if t_team == pw:
                        transition_kills[t_team] += 1
                    else:
                        transition_errors[t_team] += 1

    for ts in stats.team_stats:
        sa = sideout_attacks.get(ts.team, 0)
        if sa > 0:
            ts.sideout_attack_efficiency = (
                (sideout_kills.get(ts.team, 0) - sideout_errors.get(ts.team, 0)) / sa
            )
        ta = transition_attacks.get(ts.team, 0)
        if ta > 0:
            ts.transition_attack_efficiency = (
                (transition_kills.get(ts.team, 0) - transition_errors.get(ts.team, 0)) / ta
            )

    # 11. Clutch performance (close-score stats)
    if score_progression:
        close_rally_ids = _get_close_score_rally_ids(score_progression)
        if close_rally_ids:
            # Per-player clutch kill %
            clutch_attacks: dict[int, int] = {}
            clutch_kills: dict[int, int] = {}
            for rs in stats.rally_stats:
                if rs.rally_id not in close_rally_ids:
                    continue
                if rs.terminal_action != "attack" or rs.terminal_player_track_id < 0:
                    continue
                tid = rs.terminal_player_track_id
                clutch_attacks[tid] = clutch_attacks.get(tid, 0) + 1
                t_int = merged_team_assignments.get(tid)
                t_team = "A" if t_int == 0 else "B" if t_int == 1 else "unknown"
                if t_team == rs.point_winner:
                    clutch_kills[tid] = clutch_kills.get(tid, 0) + 1

            for tid, ps in player_stats_map.items():
                ca = clutch_attacks.get(tid, 0)
                if ca > 0:
                    ps.clutch_kill_pct = clutch_kills.get(tid, 0) / ca

            # Per-team clutch side-out %
            for ts in stats.team_stats:
                close_receiving = [
                    sp for sp in score_progression
                    if sp.rally_id in close_rally_ids
                    and sp.serving_team != ts.team
                ]
                if close_receiving:
                    close_side_outs = sum(
                        1 for sp in close_receiving if sp.point_winner == ts.team
                    )
                    ts.clutch_side_out_pct = close_side_outs / len(close_receiving)

    # 12. Duo sync scores (cross-rally player identity)
    if match_analysis:
        stats.duo_stats = _compute_duo_sync(
            rally_actions_list, stats.rally_stats,
            match_analysis, merged_team_assignments,
        )

    logger.info(
        f"Match stats: {stats.total_rallies} rallies, "
        f"{stats.total_contacts} contacts, "
        f"{len(stats.player_stats)} players tracked"
        f"{', calibrated' if is_calibrated else ''}"
    )

    return stats
