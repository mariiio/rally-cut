"""Visual verification for play_annotations — court overlay + ball trail.

Each annotated action → one PNG. The video contact frame is overlaid with:

  1. **Court lines via homography** — sidelines, baseline, net midline,
     and zone gridlines (1-5) projected onto the video frame so you see
     the court layout in perspective.
  2. **Ball trajectory trail** — the ball's flight path from contact
     onward (25 frames) drawn as a colored gradient trail on the same
     frame. Contact point marked with a big circle, landing with a star.

Title bar: ``ATTACK → LINE`` / ``SET zone 2 → 5`` etc. One glance shows
whether the trail direction matches the label, and the court overlay
tells you which zone the ball is in.

Usage
-----
    cd analysis
    uv run python scripts/visualize_play_annotations.py              # ~10 random rallies
    uv run python scripts/visualize_play_annotations.py --limit 20
    uv run python scripts/visualize_play_annotations.py --rally <id>

Writes PNGs to ``analysis/outputs/play_annotation_viz/``.
"""

from __future__ import annotations

import argparse
import random
import sys
import traceback
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from eval_action_detection import (  # noqa: E402
    _build_player_positions,
    _load_match_team_assignments,
    load_rallies_with_action_gt,
)
from production_eval import (  # noqa: E402
    PipelineContext,
    _build_calibrators,
    _parse_ball,
    _parse_positions,
)

from rallycut.statistics.play_annotations import (  # noqa: E402
    COURT_LENGTH_M,
    COURT_WIDTH_M,
    NET_Y_M,
    annotate_rally_actions,
)
from rallycut.tracking.action_classifier import ActionType, classify_rally_actions  # noqa: E402
from rallycut.tracking.contact_detector import detect_contacts  # noqa: E402
from rallycut.tracking.match_tracker import verify_team_assignments  # noqa: E402
from rallycut.tracking.sequence_action_runtime import (  # noqa: E402
    apply_sequence_override,
    get_sequence_probs,
)

console = Console()

DIR_BGR: dict[str, tuple[int, int, int]] = {
    "line": (255, 128, 31),
    "cross": (14, 127, 255),
    "cut": (40, 39, 214),
}
SET_BGR = (0, 200, 0)
NEUTRAL_BGR = (200, 200, 200)
COURT_LINE_BGR = (255, 255, 0)  # yellow (BGR)
ZONE_LINE_BGR = (200, 200, 0)   # faded yellow (BGR)
NET_LINE_BGR = (0, 255, 255)    # cyan (BGR)
TRAIL_LEN = 25  # frames of ball trail after contact


# ---------------------- pipeline ---------------------- #


def _run_rally(
    rally: Any, match_teams: dict[int, int] | None, calibrator: Any, ctx: PipelineContext
) -> tuple[Any, list]:
    ball_positions = _parse_ball(rally.ball_positions_json or [])
    player_positions = _build_player_positions(
        rally.positions_json or [], rally_id=rally.rally_id, inject_pose=True
    )
    teams = dict(match_teams) if match_teams else None
    if teams is not None and not ctx.skip_verify_teams:
        teams = verify_team_assignments(teams, player_positions)
    sequence_probs = get_sequence_probs(
        ball_positions=ball_positions,
        player_positions=player_positions,
        court_split_y=rally.court_split_y,
        frame_count=rally.frame_count,
        team_assignments=teams,
        calibrator=calibrator,
    )
    contact_sequence = detect_contacts(
        ball_positions=ball_positions,
        player_positions=player_positions,
        config=None,
        use_classifier=True,
        frame_count=rally.frame_count or None,
        team_assignments=teams,
        court_calibrator=calibrator,
        sequence_probs=sequence_probs,
    )
    rally_actions = classify_rally_actions(
        contact_sequence,
        rally.rally_id,
        team_assignments=teams,
        match_team_assignments=teams,
        calibrator=calibrator,
    )
    if sequence_probs is not None:
        apply_sequence_override(rally_actions, sequence_probs)
    return rally_actions, ball_positions


# ---------------------- helpers ---------------------- #


def _nearest_ball(ball_positions: list, frame: int, radius: int = 3) -> Any:
    best = None
    best_d = radius + 1
    for bp in ball_positions:
        d = abs(bp.frame_number - frame)
        if d <= radius and d < best_d:
            best = bp
            best_d = d
    return best


def _ball_trail(ball_positions: list, start_frame: int, length: int = TRAIL_LEN) -> list[tuple[float, float, int]]:
    """Return (x_norm, y_norm, frame) for ball samples in [start_frame, start_frame+length]."""
    trail: list[tuple[float, float, int]] = []
    for bp in ball_positions:
        if bp.frame_number < start_frame:
            continue
        if bp.frame_number > start_frame + length:
            break
        trail.append((bp.x, bp.y, bp.frame_number))
    return trail


def _setter_image_xy(positions_raw: list[dict], setter_tid: int, frame: int) -> tuple[float, float] | None:
    best = None
    best_d = 4
    for pp in positions_raw:
        try:
            if int(pp["trackId"]) != setter_tid:
                continue
            d = abs(int(pp["frameNumber"]) - frame)
        except (KeyError, TypeError, ValueError):
            continue
        if d <= 3 and d < best_d:
            try:
                best = (float(pp["x"]), float(pp["y"]))
                best_d = d
            except (KeyError, TypeError, ValueError):
                continue
    return best


class _FrameReader:
    def __init__(self, video_path: Path) -> None:
        self.cap = cv2.VideoCapture(str(video_path))
        self.ok = self.cap.isOpened()

    def read(self, absolute_frame: int) -> np.ndarray | None:
        if not self.ok:
            return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, absolute_frame)
        ok, frame = self.cap.read()
        return frame if ok else None

    def close(self) -> None:
        if self.ok:
            self.cap.release()


# ---------------------- court overlay ---------------------- #


def _court_to_px(calibrator: Any, court_x: float, court_y: float, w: int, h: int) -> tuple[int, int] | None:
    try:
        nx, ny = calibrator.court_to_image((court_x, court_y), 1, 1)
    except Exception:  # noqa: BLE001
        return None
    return (int(nx * w), int(ny * h))


def _draw_court_overlay(img: np.ndarray, calibrator: Any) -> np.ndarray:
    """Project court lines + zones onto the video frame."""
    out = img.copy()
    h, w = out.shape[:2]

    def _line(p1: tuple[float, float], p2: tuple[float, float], color: tuple[int, int, int], thickness: int = 2) -> None:
        a = _court_to_px(calibrator, *p1, w, h)
        b = _court_to_px(calibrator, *p2, w, h)
        if a is not None and b is not None:
            cv2.line(out, a, b, color, thickness, cv2.LINE_AA)

    # Court boundary (rectangle).
    _line((0, 0), (COURT_WIDTH_M, 0), COURT_LINE_BGR, 3)
    _line((COURT_WIDTH_M, 0), (COURT_WIDTH_M, COURT_LENGTH_M), COURT_LINE_BGR, 3)
    _line((COURT_WIDTH_M, COURT_LENGTH_M), (0, COURT_LENGTH_M), COURT_LINE_BGR, 3)
    _line((0, COURT_LENGTH_M), (0, 0), COURT_LINE_BGR, 3)

    # Net midline.
    _line((0, NET_Y_M), (COURT_WIDTH_M, NET_Y_M), NET_LINE_BGR, 3)

    # Zone gridlines (1.6 m intervals across width).
    for i in range(1, 5):
        x = i * 1.6
        _line((x, 0), (x, COURT_LENGTH_M), ZONE_LINE_BGR, 1)

    # Zone labels at one baseline.
    for z in range(1, 6):
        cx = (z - 0.5) * 1.6
        pt = _court_to_px(calibrator, cx, -0.3, w, h)
        if pt is not None:
            cv2.putText(out, str(z), pt, cv2.FONT_HERSHEY_SIMPLEX, 1.2, ZONE_LINE_BGR, 2, cv2.LINE_AA)

    return out


# ---------------------- ball trail ---------------------- #


def _draw_trail(
    img: np.ndarray,
    trail: list[tuple[float, float, int]],
    color_bgr: tuple[int, int, int],
    contact_frame: int,
) -> np.ndarray:
    """Draw the ball trajectory as a gradient-width trail on the frame.

    Trail goes from thick at contact to thin at landing. Contact = big
    circle, landing = filled diamond.
    """
    if len(trail) < 2:
        return img
    out = img.copy()
    h, w = out.shape[:2]
    pts = [(int(x * w), int(y * h)) for x, y, _f in trail]

    # Draw as segments with decreasing thickness.
    n = len(pts)
    for i in range(n - 1):
        t = max(2, 6 - int(4 * i / max(1, n - 1)))
        alpha = 1.0 - 0.5 * (i / max(1, n - 1))
        c = tuple(int(v * alpha) for v in color_bgr)
        cv2.line(out, pts[i], pts[i + 1], c, t, cv2.LINE_AA)

    # Contact: big open circle.
    cv2.circle(out, pts[0], 20, color_bgr, 4)
    # Landing: filled circle with white ring.
    cv2.circle(out, pts[-1], 16, (255, 255, 255), 3)
    cv2.circle(out, pts[-1], 12, color_bgr, -1)

    return out


# ---------------------- title bar ---------------------- #


def _draw_title(img: np.ndarray, text: str, color_bgr: tuple[int, int, int]) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    bar_h = max(70, int(h * 0.08))
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.7, out, 0.3, 0)
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = bar_h / 45.0
    thickness = max(2, int(scale * 2))
    cv2.putText(out, text, (20, bar_h - 18), font, scale, color_bgr, thickness, cv2.LINE_AA)
    return out


# ---------------------- render actions ---------------------- #


def _render_attack(
    reader: _FrameReader,
    rally: Any,
    rally_start_frame: int,
    action: Any,
    ball_positions: list,
    calibrator: Any,
    out_dir: Path,
) -> bool:
    direction = action.attack_direction or "unknown"
    color = DIR_BGR.get(direction, NEUTRAL_BGR)
    title = f"ATTACK -> {direction.upper()}    rally {rally.rally_id[:8]}  f={action.frame}"

    raw = reader.read(rally_start_frame + action.frame)
    if raw is None:
        return False

    # 1. Court overlay.
    frame = _draw_court_overlay(raw, calibrator)
    # 2. Ball trail.
    trail = _ball_trail(ball_positions, action.frame, TRAIL_LEN)
    frame = _draw_trail(frame, trail, color, action.frame)
    # 3. Title.
    frame = _draw_title(frame, title, color)

    out_path = out_dir / f"{rally.video_id[:8]}__{rally.rally_id[:8]}__f{action.frame:04d}__attack_{direction}.png"
    cv2.imwrite(str(out_path), frame)
    return True


def _render_set(
    reader: _FrameReader,
    rally: Any,
    rally_start_frame: int,
    set_action: Any,
    next_attack: Any,
    ball_positions: list,
    calibrator: Any,
    out_dir: Path,
) -> bool:
    o = set_action.set_origin_zone if set_action.set_origin_zone is not None else "?"
    d = set_action.set_dest_zone if set_action.set_dest_zone is not None else "?"
    title = f"SET zone {o} -> {d}    rally {rally.rally_id[:8]}  f={set_action.frame}"

    raw = reader.read(rally_start_frame + set_action.frame)
    if raw is None:
        return False

    frame = _draw_court_overlay(raw, calibrator)
    h, w = frame.shape[:2]

    # Ball trail from set to next attack.
    trail = _ball_trail(ball_positions, set_action.frame, next_attack.frame - set_action.frame + 5)
    frame = _draw_trail(frame, trail, SET_BGR, set_action.frame)

    # Circle the setter (green).
    setter_xy = (
        _setter_image_xy(rally.positions_json or [], set_action.player_track_id, set_action.frame)
        if set_action.player_track_id >= 0 else None
    )
    if setter_xy is not None:
        sx, sy = int(setter_xy[0] * w), int(setter_xy[1] * h)
        cv2.circle(frame, (sx, sy), 36, (0, 200, 0), 5)
        cv2.putText(frame, "SETTER", (sx - 40, sy - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2, cv2.LINE_AA)

    frame = _draw_title(frame, title, SET_BGR)

    out_path = out_dir / f"{rally.video_id[:8]}__{rally.rally_id[:8]}__f{set_action.frame:04d}__set_z{o}to{d}.png"
    cv2.imwrite(str(out_path), frame)
    return True


# ---------------------- main ---------------------- #


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--rally", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from rallycut.evaluation.tracking.db import get_video_path  # lazy

    out_dir = Path(__file__).resolve().parent.parent / "outputs" / "play_annotation_viz"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.png"):
        old.unlink()

    console.print("[bold]Loading rallies with action GT...[/bold]")
    rallies = load_rallies_with_action_gt(rally_id=args.rally if args.rally else None)
    if not args.rally:
        random.Random(args.seed).shuffle(rallies)
        rallies = rallies[: args.limit]
    console.print(f"  rendering {len(rallies)} rallies → {out_dir}")

    rally_pos_lookup: dict[str, list[Any]] = {
        r.rally_id: _parse_positions(r.positions_json) for r in rallies if r.positions_json
    }
    video_ids = {r.video_id for r in rallies}
    team_map = _load_match_team_assignments(video_ids, rally_positions=rally_pos_lookup)
    calibrators = _build_calibrators(video_ids)

    video_paths: dict[str, Path | None] = {}
    for vid in video_ids:
        p = get_video_path(vid)
        video_paths[vid] = Path(p) if p else None

    ctx = PipelineContext()
    n_png = 0
    for idx, rally in enumerate(rallies, start=1):
        if not rally.ball_positions_json or not rally.positions_json or not rally.frame_count:
            continue
        calibrator = calibrators.get(rally.video_id)
        if calibrator is None or not getattr(calibrator, "is_calibrated", False):
            continue
        vp = video_paths.get(rally.video_id)
        if vp is None or not vp.exists():
            continue
        try:
            rally_actions, ball_positions = _run_rally(
                rally, team_map.get(rally.rally_id), calibrator, ctx
            )
        except Exception as exc:  # noqa: BLE001
            console.print(f"  [red][{idx}/{len(rallies)}][/red] {rally.rally_id[:8]}: {type(exc).__name__}: {exc}")
            continue
        annotate_rally_actions(
            rally_actions, ball_positions, rally.positions_json or [], calibrator
        )

        reader = _FrameReader(vp)
        if not reader.ok:
            continue
        rally_start_frame = int(rally.start_ms / 1000.0 * rally.fps)
        actions_sorted = sorted(rally_actions.actions, key=lambda a: a.frame)
        try:
            for i, a in enumerate(actions_sorted):
                if a.action_type == ActionType.ATTACK:
                    if _render_attack(reader, rally, rally_start_frame, a, ball_positions, calibrator, out_dir):
                        n_png += 1
                elif a.action_type == ActionType.SET:
                    next_attack = None
                    for b in actions_sorted[i + 1 :]:
                        if b.action_type == ActionType.ATTACK:
                            next_attack = b
                            break
                    if next_attack is None:
                        continue
                    if _render_set(reader, rally, rally_start_frame, a, next_attack, ball_positions, calibrator, out_dir):
                        n_png += 1
        finally:
            reader.close()
        console.print(f"  [{idx}/{len(rallies)}] {rally.rally_id[:8]}: total={n_png}")

    console.print(f"[green]done[/green] total PNGs={n_png}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)
