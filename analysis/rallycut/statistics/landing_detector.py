"""Landing detection for serve and attack heatmaps.

Detects where on the court serves and attacks land by using **player
feet positions** (ground-plane, Z=0) as the primary proxy. The court
homography is geometrically exact at Z=0 (<2px reprojection error on
66 calibrated videos), so projecting feet gives accurate court
coordinates.

Ball positions in the air are NOT reliable through the homography
(the ball at arm height ≈ 1-2m above ground projects wildly off-court).
Ball-stopped detection (velocity threshold) is used only as a fallback
for terminal attacks where no receiving player exists.

Strategy:
  - **Serve target**: Receiving player's feet at the receive contact
    frame. The serve "lands" where the receiver plays it.
  - **Attack landing**: For mid-rally attacks, the next-contact player's
    feet. For terminal attacks (kill/error), the ball's stopped position
    on the sand (Z=0, homography exact) or a ball-trajectory fallback.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from rallycut.court.calibration import CourtCalibrator
    from rallycut.tracking.action_classifier import ClassifiedAction, RallyActions
    from rallycut.tracking.ball_tracker import BallPosition

# Velocity threshold for "stopped ball" detection (normalised image coords
# per frame). Matches the values calibrated in audit_ball_3d_tier1.py.
STOPPED_VEL_THRESHOLD = 0.004  # ~4 px/frame at 1920px
STOPPED_MIN_RUN = 5  # consecutive frames required

# Court dimensions (must match rallycut.court.calibration).
COURT_WIDTH_M = 8.0
COURT_LENGTH_M = 16.0

# Grid resolution for heatmap binning.
GRID_COLS = 8  # 1m per column on 8m court width
GRID_ROWS = 4  # 2m per row on 8m half-court
HALF_COURT_M = COURT_LENGTH_M / 2.0  # 8m

# Minimum confidence for ball positions to be considered.
_MIN_BALL_CONF = 0.3

# Maximum frame gap allowed within a "stopped" run.
_MAX_FRAME_GAP = 3

# Frame search radius for nearest player/ball lookups.
_NEAREST_RADIUS = 3

# COCO-17 keypoint indices for ankles.
_COCO_LEFT_ANKLE = 15
_COCO_RIGHT_ANKLE = 16
_MIN_ANKLE_CONF = 0.5

# Ball-trajectory fallback window for terminal attacks.
_FALLBACK_DT_MIN = 10
_FALLBACK_DT_MAX = 30


@dataclass
class LandingPoint:
    """A detected ball landing position on the court."""

    frame: int
    image_x: float  # normalized 0-1
    image_y: float  # normalized 0-1
    court_x: float | None  # metres (None if uncalibrated)
    court_y: float | None  # metres
    action_type: str  # "serve" or "attack"
    rally_id: str
    player_track_id: int
    team: str  # "A" or "B"
    court_side: str = "unknown"  # "near" or "far" — acting team's court side

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "type": self.action_type,
            "team": self.team,
            "courtSide": self.court_side,
            "rallyId": self.rally_id,
            "playerTrackId": self.player_track_id,
            "frame": self.frame,
            "imageX": round(self.image_x, 4),
            "imageY": round(self.image_y, 4),
        }
        if self.court_x is not None and self.court_y is not None:
            d["courtX"] = round(self.court_x, 2)
            d["courtY"] = round(self.court_y, 2)
        return d


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def find_landing(
    ball_positions: list[BallPosition],
    start_frame: int,
    end_frame: int | None = None,
) -> tuple[int, float, float] | None:
    """Find where the ball stops within a frame window.

    Walks backward from the end of the window, collecting consecutive
    frames where image-space displacement is below STOPPED_VEL_THRESHOLD.
    Returns the median position of the stopped segment.

    Returns:
        (frame, x, y) of the landing, or None if ball never stops.
    """
    pts = sorted(
        [
            (bp.frame_number, bp.x, bp.y)
            for bp in ball_positions
            if bp.confidence >= _MIN_BALL_CONF
            and bp.frame_number >= start_frame
            and (end_frame is None or bp.frame_number < end_frame)
        ],
        key=lambda t: t[0],
    )

    if len(pts) < STOPPED_MIN_RUN + 1:
        return None

    end = len(pts) - 1
    stopped_start = end
    while stopped_start > 0:
        f0, x0, y0 = pts[stopped_start - 1]
        f1, x1, y1 = pts[stopped_start]
        if f1 - f0 > _MAX_FRAME_GAP:
            break
        dist = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
        if dist > STOPPED_VEL_THRESHOLD:
            break
        stopped_start -= 1

    run_len = end - stopped_start + 1
    if run_len < STOPPED_MIN_RUN:
        return None

    stopped = pts[stopped_start : end + 1]
    median_x = float(np.median([r[1] for r in stopped]))
    median_y = float(np.median([r[2] for r in stopped]))
    return (stopped[len(stopped) // 2][0], median_x, median_y)


def _feet_image_xy(pos: dict[str, Any]) -> tuple[float, float] | None:
    """Image-space feet estimate from a raw position dict.

    Prefers ankle keypoints when confident, else falls back to bbox
    bottom-center (x, y + height/2).
    """
    kps = pos.get("keypoints")
    if kps and len(kps) > _COCO_RIGHT_ANKLE:
        try:
            left = kps[_COCO_LEFT_ANKLE]
            right = kps[_COCO_RIGHT_ANKLE]
            lx, ly, lc = float(left[0]), float(left[1]), float(left[2])
            rx, ry, rc = float(right[0]), float(right[1]), float(right[2])
            left_ok = lc >= _MIN_ANKLE_CONF
            right_ok = rc >= _MIN_ANKLE_CONF
            if left_ok and right_ok:
                return ((lx + rx) / 2.0, (ly + ry) / 2.0)
            if left_ok:
                return (lx, ly)
            if right_ok:
                return (rx, ry)
        except (IndexError, KeyError, TypeError, ValueError):
            pass
    try:
        px = float(pos["x"])
        py = float(pos["y"]) + float(pos.get("height", 0.10)) / 2.0
        return (px, py)
    except (KeyError, TypeError, ValueError):
        return None


def _player_feet_court_xy(
    positions_raw: list[dict[str, Any]],
    track_id: int,
    frame: int,
    calibrator: CourtCalibrator,
) -> tuple[float, float] | None:
    """Project a player's feet to court coordinates (metres).

    Searches within ±_NEAREST_RADIUS frames for the given track_id.
    Feet are on the ground plane (Z=0), so the homography is exact.
    """
    best_img: tuple[float, float] | None = None
    best_d = _NEAREST_RADIUS + 1
    for pp in positions_raw:
        try:
            if int(pp["trackId"]) != track_id:
                continue
            d = abs(int(pp["frameNumber"]) - frame)
        except (KeyError, TypeError, ValueError):
            continue
        if d <= _NEAREST_RADIUS and d < best_d:
            pt = _feet_image_xy(pp)
            if pt is not None:
                best_img = pt
                best_d = d
    if best_img is None:
        return None
    try:
        pc = calibrator.image_to_court(best_img, 1, 1)
        return (float(pc[0]), float(pc[1]))
    except Exception:  # noqa: BLE001
        return None


def _find_ball_near_frame(
    ball_positions: list[BallPosition],
    target_frame: int,
    radius: int = 3,
) -> tuple[float, float] | None:
    """Find the ball position closest to a target frame."""
    candidates = [
        bp for bp in ball_positions
        if abs(bp.frame_number - target_frame) <= radius
        and bp.confidence >= _MIN_BALL_CONF
    ]
    if not candidates:
        return None
    best = max(candidates, key=lambda bp: bp.confidence)
    return (best.x, best.y)


def _get_court_side(
    team: str,
    team_assignments: dict[str, int] | None,
) -> str:
    """Map a team label to a court side.

    After ``reattribute-actions``, team "A" is always mapped to near
    (team 0) and "B" to far (team 1).  If *team_assignments* is not
    available we fall back to the same convention directly from the
    team label.
    """
    if team == "A":
        return "near"
    if team == "B":
        return "far"
    return "unknown"


# ---------------------------------------------------------------------------
# Main detection
# ---------------------------------------------------------------------------


def detect_rally_landings(
    rally_actions: RallyActions,
    ball_positions: list[BallPosition],
    calibrator: CourtCalibrator | None,
    video_width: int,
    video_height: int,
    positions_raw: list[dict[str, Any]] | None = None,
) -> list[LandingPoint]:
    """Detect landing points in a rally for serve and attack heatmaps.

    **Serve target**: The receiving player's feet at the receive contact
    frame. The ball is in flight the entire time between serve and
    receive, so using ball image positions through the homography gives
    wrong results (ball at arm height ≠ Z=0). Player feet are on the
    ground plane and project accurately.

    **Attack landing**: For ALL attacks (not just terminal), the next-
    contact player's feet position is used as the landing point. For
    terminal attacks (no next contact), falls back to ball-stopped
    detection on the sand (Z=0, homography exact) or ball trajectory
    position some frames after contact.

    Args:
        rally_actions: Classified actions for the rally.
        ball_positions: WASB ball detections for the rally.
        calibrator: Court calibrator (may be None).
        video_width: Video width in pixels.
        video_height: Video height in pixels.
        positions_raw: Raw player position dicts (for feet projection).
            When provided, enables accurate ground-plane projection via
            player ankle/feet keypoints.

    Returns:
        List of detected LandingPoint instances.
    """
    from rallycut.tracking.action_classifier import ActionType

    if not rally_actions.actions or not ball_positions:
        return []
    if calibrator is None or not calibrator.is_calibrated:
        return []

    landings: list[LandingPoint] = []
    actions = sorted(rally_actions.actions, key=lambda a: a.frame)
    team_assignments = getattr(rally_actions, "team_assignments", None)

    def _make_landing(
        frame: int,
        court_x: float,
        court_y: float,
        image_x: float,
        image_y: float,
        action_type: str,
        player_track_id: int,
        team: str,
    ) -> LandingPoint:
        return LandingPoint(
            frame=frame,
            image_x=image_x,
            image_y=image_y,
            court_x=court_x,
            court_y=court_y,
            action_type=action_type,
            rally_id=rally_actions.rally_id,
            player_track_id=player_track_id,
            team=team,
            court_side=_get_court_side(team, team_assignments),
        )

    # --- Serve target (receiving player's feet) ---
    serve = rally_actions.serve
    if serve is not None and serve.action_type == ActionType.SERVE:
        receive: ClassifiedAction | None = None
        for a in actions:
            if a.frame > serve.frame and a.action_type == ActionType.RECEIVE:
                receive = a
                break

        if (
            receive is not None
            and receive.player_track_id >= 0
            and positions_raw is not None
        ):
            court_pos = _player_feet_court_xy(
                positions_raw, receive.player_track_id,
                receive.frame, calibrator,
            )
            if court_pos is not None:
                # Use ball position at receive frame for image coords.
                ball_at_recv = _find_ball_near_frame(
                    ball_positions, receive.frame,
                )
                img_x = ball_at_recv[0] if ball_at_recv else receive.ball_x
                img_y = ball_at_recv[1] if ball_at_recv else receive.ball_y
                landings.append(_make_landing(
                    receive.frame, court_pos[0], court_pos[1],
                    img_x, img_y,
                    "serve", serve.player_track_id, serve.team,
                ))

    # --- All attacks (mid-rally + terminal) ---
    for i, action in enumerate(actions):
        if action.action_type != ActionType.ATTACK:
            continue

        # Look for the next real contact after this attack.
        next_contact_court: tuple[float, float] | None = None
        next_img: tuple[float, float] | None = None
        next_frame: int | None = None
        for b in actions[i + 1 :]:
            if b.action_type in (ActionType.UNKNOWN,):
                continue
            if b.player_track_id < 0:
                continue
            if positions_raw is not None:
                bc = _player_feet_court_xy(
                    positions_raw, b.player_track_id,
                    b.frame, calibrator,
                )
                if bc is not None:
                    next_contact_court = bc
                    ball_at_contact = _find_ball_near_frame(
                        ball_positions, b.frame,
                    )
                    next_img = ball_at_contact if ball_at_contact else (
                        b.ball_x, b.ball_y,
                    )
                    next_frame = b.frame
            break  # only check first valid contact candidate

        if next_contact_court is not None and next_frame is not None:
            assert next_img is not None
            landings.append(_make_landing(
                next_frame, next_contact_court[0], next_contact_court[1],
                next_img[0], next_img[1],
                "attack", action.player_track_id, action.team,
            ))
            continue

        # Terminal attack: no next contact found.
        attack_court: tuple[float, float] | None = None
        img_x = action.ball_x
        img_y = action.ball_y
        det_frame = action.frame

        # Primary: ball stopped on sand (Z=0, homography exact).
        stopped = find_landing(ball_positions, action.frame)
        if stopped is not None:
            det_frame, sx, sy = stopped
            attack_court = _project_court_safe(
                sx, sy, calibrator,
            )
            img_x, img_y = sx, sy

        # Fallback: ball position 10-30 frames after contact.
        if attack_court is None:
            for dt in range(_FALLBACK_DT_MIN, _FALLBACK_DT_MAX + 1):
                pos = _find_ball_near_frame(
                    ball_positions, action.frame + dt, radius=2,
                )
                if pos is not None:
                    det_frame = action.frame + dt
                    cp = _project_court_safe(pos[0], pos[1], calibrator)
                    if cp is not None:
                        attack_court = cp
                        img_x, img_y = pos
                    break

        if attack_court is not None:
            landings.append(_make_landing(
                det_frame, attack_court[0], attack_court[1],
                img_x, img_y,
                "attack", action.player_track_id, action.team,
            ))

    return landings


def _project_court_safe(
    x: float,
    y: float,
    calibrator: CourtCalibrator,
) -> tuple[float, float] | None:
    """Project normalised image coords to court metres, returning None on error.

    Uses (1, 1) for width/height since coords are already normalised 0-1.
    """
    try:
        c = calibrator.image_to_court((x, y), 1, 1)
        return (float(c[0]), float(c[1]))
    except Exception:  # noqa: BLE001
        return None



# ---------------------------------------------------------------------------
# Heatmap aggregation
# ---------------------------------------------------------------------------


def compute_landing_heatmaps(
    landings: list[LandingPoint],
    rally_actions_list: list[Any] | None = None,
) -> dict[str, Any]:
    """Aggregate landing points into per-team half-court heatmap grids.

    Each team's landings are normalised to a canonical half-court view
    (team's own half at bottom, opponent half at top) so that side
    switches during the match are transparent.

    Returns a dict with keys ``"teamA"``, ``"teamB"`` (each containing
    ``"serve"``, ``"attack"``, ``"all"`` grids + ``"points"``), and
    ``"perRally"`` (raw court coords per rally for debug overlay).

    Grid is GRID_ROWS x GRID_COLS (4x8 = 2m x 1m cells on 8m half-court).
    """
    # Only include landings that project within the court bounds
    # (with a small margin for projection noise).
    margin = 1.0  # 1m tolerance outside court lines
    calibrated_landings = [
        lp for lp in landings
        if lp.court_x is not None
        and lp.court_y is not None
        and -margin <= lp.court_x <= COURT_WIDTH_M + margin
        and -margin <= lp.court_y <= COURT_LENGTH_M + margin
    ]

    def _build_half_grid(pts: list[LandingPoint]) -> list[list[float]]:
        """Build a normalised 4x8 grid on the opponent's half-court.

        Normalisation rule (canonical view: team's own half at bottom):
        - Team on "near" side (Y=8-16m): targets land on far side (Y=0-8m)
          — already canonical, no flip.
        - Team on "far" side (Y=0-8m): targets land on near side (Y=8-16m)
          — flip: y = 16 - y, x = 8 - x.
        After normalisation, clamp to half-court: x in [0, 8m), y in [0, 8m).
        """
        grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float64)
        for lp in pts:
            if lp.court_x is None or lp.court_y is None:
                continue
            cx, cy = lp.court_x, lp.court_y
            # Normalise based on acting team's court side.
            if lp.court_side == "far":
                cy = COURT_LENGTH_M - cy
                cx = COURT_WIDTH_M - cx
            # After normalisation the target should be on the opponent's
            # half (Y=0-8m).  Clamp to half-court bounds.
            cx = max(0.0, min(cx, COURT_WIDTH_M - 1e-9))
            cy = max(0.0, min(cy, HALF_COURT_M - 1e-9))
            gx = int(cx / COURT_WIDTH_M * GRID_COLS)
            gy = int(cy / HALF_COURT_M * GRID_ROWS)
            grid[gy, gx] += 1
        total = grid.sum()
        if total > 0:
            grid /= total
        return [[round(float(v), 4) for v in row] for row in grid]

    def _team_block(pts: list[LandingPoint]) -> dict[str, Any]:
        serve_pts = [lp for lp in pts if lp.action_type == "serve"]
        attack_pts = [lp for lp in pts if lp.action_type == "attack"]
        return {
            "serve": {
                "grid": _build_half_grid(serve_pts),
                "count": len(serve_pts),
            },
            "attack": {
                "grid": _build_half_grid(attack_pts),
                "count": len(attack_pts),
            },
            "all": {
                "grid": _build_half_grid(pts),
                "count": len(pts),
            },
            "points": [lp.to_dict() for lp in pts],
        }

    team_a_pts = [lp for lp in calibrated_landings if lp.team == "A"]
    team_b_pts = [lp for lp in calibrated_landings if lp.team == "B"]

    # Build per-rally dict with raw (non-normalised) landing coords.
    per_rally: dict[str, dict[str, Any]] = {}
    # Pre-compute serving team from rally_actions_list.
    serving_team_map: dict[str, str] = {}
    if rally_actions_list is not None:
        for ra in rally_actions_list:
            serve = getattr(ra, "serve", None)
            if serve is not None:
                serving_team_map[ra.rally_id] = serve.team
    for lp in calibrated_landings:
        entry = per_rally.setdefault(lp.rally_id, {
            "points": [],
            "servingTeam": serving_team_map.get(lp.rally_id, "unknown"),
        })
        entry["points"].append(lp.to_dict())

    result: dict[str, Any] = {
        "teamA": _team_block(team_a_pts),
        "teamB": _team_block(team_b_pts),
        "perRally": per_rally,
    }
    return result
