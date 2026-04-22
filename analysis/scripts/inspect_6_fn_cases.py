"""Dump raw ball trajectory around the 6 user-flagged FN cases and compute direction
change over multiple windows + show the pipeline's direction_change_deg semantics
honestly.

Target cases (rally_id prefix, GT frame):
    fb7f9c23 f:230
    8ce36875 f:72
    a67c04fb f:143
    04ef801f f:228
    1a6e05d5 f:147
    f978201e f:92
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rallycut.evaluation.db import get_connection
from rallycut.tracking.ball_tracker import BallPosition
from rallycut.tracking.contact_detector import compute_direction_change

TARGETS = [
    ("fb7f9c23", 230),
    ("8ce36875", 72),
    ("a67c04fb", 143),
    ("04ef801f", 228),
    ("1a6e05d5", 147),
    ("f978201e", 92),
]

DB = "postgresql://postgres:postgres@localhost:5436/rallycut"


def angle_between(v1: tuple[float, float], v2: tuple[float, float]) -> float:
    """Angle between two 2D vectors in degrees."""
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return float("nan")
    cos_theta = max(-1.0, min(1.0, dot / (n1 * n2)))
    return math.degrees(math.acos(cos_theta))


def load_ball(rally_prefix: str) -> tuple[str, list[BallPosition]] | None:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT r.id, pt.ball_positions_json, pt.fps "
        "FROM rallies r JOIN player_tracks pt ON pt.rally_id = r.id "
        "WHERE r.id LIKE %s LIMIT 1",
        (rally_prefix + "%",),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    rid, ball_json, fps = row
    if ball_json is None:
        return None
    # Already a dict/list since JSONB.
    ball_list = ball_json if isinstance(ball_json, list) else json.loads(ball_json)
    balls = []
    for bp in ball_list:
        if bp.get("x", 0) > 0 or bp.get("y", 0) > 0:
            balls.append(BallPosition(
                frame_number=bp["frameNumber"],
                x=bp["x"], y=bp["y"],
                confidence=bp.get("confidence", 1.0),
            ))
    return rid, balls


def main() -> None:
    for prefix, gt_frame in TARGETS:
        result = load_ball(prefix)
        if result is None:
            print(f"=== {prefix} f:{gt_frame}: NO BALL DATA ===\n")
            continue
        rid, balls = result
        ball_by_frame = {b.frame_number: b for b in balls}

        print(f"=== {rid[:12]}... f:{gt_frame} (rally {prefix}) ===")
        # Show ±7 frames around GT
        for f in range(gt_frame - 7, gt_frame + 8):
            b = ball_by_frame.get(f)
            if b is None:
                print(f"  f={f:4d}: (no detection)")
            else:
                marker = "  <-- GT" if f == gt_frame else ""
                print(
                    f"  f={f:4d}: x={b.x:.4f} y={b.y:.4f} conf={b.confidence:.3f}{marker}"
                )

        # Compute direction change at GT frame using production's function
        # at several check-frame windows.
        print(f"  direction_change_deg at GT frame, over windows:")
        for window in (3, 5, 7, 10):
            dc = compute_direction_change(ball_by_frame, gt_frame, window)
            print(f"    window=±{window}f: {dc:.2f}°")

        # Also: compute a "visual" simple angle using frame f-5 vs f vs f+5
        def pt(f):
            return ball_by_frame.get(f)
        f = gt_frame
        visuals = []
        for w in (3, 5, 8):
            b_pre = pt(f - w)
            b_mid = pt(f)
            b_post = pt(f + w)
            # Fall back to nearest available frame if direct hit missing.
            if b_pre is None:
                for off in range(1, 4):
                    b_pre = pt(f - w - off) or pt(f - w + off)
                    if b_pre is not None:
                        break
            if b_mid is None:
                for off in range(1, 4):
                    b_mid = pt(f - off) or pt(f + off)
                    if b_mid is not None:
                        break
            if b_post is None:
                for off in range(1, 4):
                    b_post = pt(f + w + off) or pt(f + w - off)
                    if b_post is not None:
                        break
            if b_pre is None or b_mid is None or b_post is None:
                visuals.append((w, float("nan"), "missing detections"))
                continue
            v_in = (b_mid.x - b_pre.x, b_mid.y - b_pre.y)
            v_out = (b_post.x - b_mid.x, b_post.y - b_mid.y)
            # Direction change = 180° - angle_between_consecutive_vectors
            # (0° = no change, 180° = complete reversal).
            # compute_direction_change uses 180° - theta convention.
            theta = angle_between(v_in, v_out)
            dc_like = 180.0 - theta if not math.isnan(theta) else float("nan")
            visuals.append((w, dc_like, f"{b_pre.frame_number}->{b_mid.frame_number}->{b_post.frame_number}"))
        print("  visual direction change (|incoming-outgoing angle|, 0=straight, 180=reverse):")
        for w, dc_like, note in visuals:
            if math.isnan(dc_like):
                print(f"    ±{w}f: NaN ({note})")
            else:
                print(f"    ±{w}f: {dc_like:6.2f}° [{note}]")

        # Also show speed change.
        def speed_at(f_mid: int, w: int) -> float:
            b_pre = ball_by_frame.get(f_mid - w)
            b_mid = ball_by_frame.get(f_mid)
            b_post = ball_by_frame.get(f_mid + w)
            if b_pre is None or b_mid is None or b_post is None:
                return float("nan")
            s_in = math.hypot(b_mid.x - b_pre.x, b_mid.y - b_pre.y) / w
            s_out = math.hypot(b_post.x - b_mid.x, b_post.y - b_mid.y) / w
            return s_in, s_out
        for w in (3, 5):
            result_s = speed_at(f, w)
            if isinstance(result_s, tuple):
                s_in, s_out = result_s
                ratio = s_out / max(s_in, 1e-6)
                print(f"  speed at ±{w}f: in={s_in:.5f} out={s_out:.5f} (out/in={ratio:.2f})")
        print()


if __name__ == "__main__":
    main()
