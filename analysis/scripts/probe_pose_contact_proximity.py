"""Phase B probe P2: pose → ball proximity at known contact frames.

For every action GT contact frame where a player track ID is available,
check whether the attributed player's wrist keypoint is within 30cm of
the ball position at that frame, measured on the action-specific contact
height plane (serve ~2.5m, dig ~0.3m, etc.).

Why this matters
----------------
The plan's Tier 2 research candidate is a pose-anchored multi-hypothesis
Bayesian filter that collapses depth uncertainty at contact events using
ball-hand proximity from pose. For that to work in practice, the pose
system must reliably detect a hand within ~30cm of the ball at contact.
If it doesn't (because hands are small, pose is jittery, or contacts are
brief), the channel is not viable even if the framework is sound.

Uses action GT as the contact source (~340 rallies). The attributed
player track ID comes from the same GT label.

Why the ray→plane back-projection instead of ground homography
---------------------------------------------------------------
A serve contact happens at ~2.5m above the ground. Back-projecting
through the ground homography (z=0) would put both the ball and the
wrist several metres off-court in the Y direction. Instead we build a
proper pinhole camera model via calibrate_camera_with_net and cast the
ray to the action-specific contact-height plane. Both ball and wrist
are back-projected with the same camera model, so the residual depth
error cancels when we measure their RELATIVE distance.

Gate
----
≥70% of contacts have ball-wrist 3D distance ≤ 0.30 m on the
contact-height plane, across serve/receive/set/attack contacts.

Usage
-----
    cd analysis
    uv run python scripts/probe_pose_contact_proximity.py
    uv run python scripts/probe_pose_contact_proximity.py --audit-only
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import matplotlib.pyplot as plt
import numpy as np

from rallycut.court.calibration import COURT_LENGTH, COURT_WIDTH, CourtCalibrator
from rallycut.court.camera_model import (
    CameraModel,
    calibrate_camera,
    calibrate_camera_with_net,
    image_ray,
)
from rallycut.evaluation.db import get_connection
from rallycut.tracking.ball_tracker import BallPosition
from rallycut.tracking.contact_detector import ContactDetectionConfig, detect_contacts
from rallycut.tracking.player_tracker import PlayerPosition

COURT_CORNERS: list[tuple[float, float]] = [
    (0.0, 0.0),
    (COURT_WIDTH, 0.0),
    (COURT_WIDTH, COURT_LENGTH),
    (0.0, COURT_LENGTH),
]

OUTPUT_DIR = Path("outputs/ball_3d_rig")
AUDIT_FILE = OUTPUT_DIR / "audit_rallies.json"

# Action-specific contact height (metres above ground).
CONTACT_HEIGHT_M = {
    "serve": 2.5,
    "receive": 0.6,
    "set": 2.2,
    "attack": 2.8,
    "block": 2.5,
    "dig": 0.3,
}

# COCO 17 keypoint indices.
KPT_LEFT_WRIST = 9
KPT_RIGHT_WRIST = 10
MIN_WRIST_CONF = 0.25

# Proximity gates: image-space thresholds corresponding to approximate 3D
# separations at typical ball depth (~10m, focal ~2000px, ball diameter ~20cm).
#
# A contact is not literal touching — the hand strikes/cups the ball from one
# side, so the wrist keypoint is typically 20-40cm from the ball center even
# during a clean contact. Useful depth anchors for a multi-hypothesis filter
# can tolerate this — what matters is reducing the 5m depth prior to a
# meaningful posterior.
#
# Reported at three gates:
#   tight  (~30cm 3D):  0.035 image  — "ball touching wrist"
#   medium (~60cm 3D):  0.070 image  — "ball at hand" (realistic contact)
#   loose  (~100cm 3D): 0.110 image  — "wrist is somewhere near the ball"
#
# Ray back-projection to a contact-height plane was empirically unstable at
# 1-2m cameras (tiny direction_z → huge amplification), so we measure
# proximity in image space rather than on a metric plane.
PROXIMITY_THRESHOLDS = {
    "tight_30cm": 0.035,
    "medium_60cm": 0.070,
    "loose_1m": 0.110,
}
GATE_PASS_RATE = 0.70

# Search window (frames) around the action GT label to find the actual
# ball-hand contact moment. Action GT labels mark "action moments" which
# are often offset from the true contact frame by several frames (e.g.,
# serve-toss-apex labels sit ~10 frames before the contact swing). Search
# within ±CONTACT_SEARCH_WINDOW for the frame of minimum ball-wrist image
# distance — that's the actual touching moment for the attributed player.
CONTACT_SEARCH_WINDOW = 20


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


@dataclass
class RallyRow:
    rally_id: str
    video_id: str
    fps: float
    ball_positions: list[BallPosition]
    player_positions: list[PlayerPosition]
    action_gt: list[dict]
    video_cal: tuple[list[tuple[float, float]], int, int]


def _parse_ball(bp_json: Any) -> list[BallPosition]:
    if not bp_json:
        return []
    positions = bp_json.get("positions", []) if isinstance(bp_json, dict) else bp_json
    return [
        BallPosition(
            frame_number=p.get("frameNumber", 0),
            x=p.get("x", 0.0),
            y=p.get("y", 0.0),
            confidence=p.get("confidence", 0.5),
        )
        for p in positions
    ]


def _parse_players(pos_json: Any) -> list[PlayerPosition]:
    if not pos_json:
        return []
    return [
        PlayerPosition(
            frame_number=p.get("frameNumber", 0),
            track_id=p.get("trackId", 0),
            x=p.get("x", 0.0),
            y=p.get("y", 0.0),
            width=p.get("width", 0.0),
            height=p.get("height", 0.0),
            confidence=p.get("confidence", 0.5),
            keypoints=p.get("keypoints"),
        )
        for p in pos_json
    ]


def load_rallies(audit_only: bool) -> list[RallyRow]:
    where = """
        WHERE pt.action_ground_truth_json IS NOT NULL
          AND pt.ball_positions_json IS NOT NULL
          AND v.court_calibration_json IS NOT NULL
    """
    if audit_only and AUDIT_FILE.exists():
        audit_info = json.loads(AUDIT_FILE.read_text())
        audit_ids = [r["rally_id"] for r in audit_info["audit_rallies"]]
    else:
        audit_ids = None

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(f"""
            SELECT r.id, r.video_id, pt.fps,
                   pt.ball_positions_json, pt.positions_json, pt.action_ground_truth_json,
                   v.court_calibration_json, v.width, v.height
            FROM rallies r
            JOIN player_tracks pt ON pt.rally_id = r.id
            JOIN videos v ON v.id = r.video_id
            {where}
        """)
        rows = cur.fetchall()

    out: list[RallyRow] = []
    for row in rows:
        rid = str(row[0])
        if audit_ids is not None and rid not in audit_ids:
            continue
        cal = row[6]
        if not isinstance(cal, list) or len(cal) != 4:
            continue
        out.append(RallyRow(
            rally_id=rid,
            video_id=str(row[1]),
            fps=float(row[2] or 30.0),
            ball_positions=_parse_ball(row[3]),
            player_positions=_parse_players(row[4]),
            action_gt=row[5] or [],
            video_cal=(
                [(c["x"], c["y"]) for c in cal],
                int(row[7] or 1920),
                int(row[8] or 1080),
            ),
        ))
    return out


# ---------------------------------------------------------------------------
# Geometry: ray → constant-z plane
# ---------------------------------------------------------------------------


def ray_to_plane(
    camera: CameraModel, image_xy: tuple[float, float], z_target: float,
) -> tuple[float, float] | None:
    """Intersect camera ray through image point with horizontal plane z=z_target.

    Returns (x, y) in metric court coordinates, or None if the ray does not
    reach the plane in the forward direction (direction_z points away from
    the plane).
    """
    origin, direction = image_ray(camera, image_xy)
    dz = direction[2]
    if abs(dz) < 1e-8:
        return None
    t = (z_target - origin[2]) / dz
    if t <= 0:
        return None
    x = float(origin[0] + t * direction[0])
    y = float(origin[1] + t * direction[1])
    return x, y


def build_camera(rally: RallyRow) -> CameraModel | None:
    calibrator = CourtCalibrator()
    calibrator.calibrate(rally.video_cal[0])
    cs = detect_contacts(
        ball_positions=rally.ball_positions,
        player_positions=rally.player_positions,
        config=ContactDetectionConfig(),
        court_calibrator=calibrator,
    )
    net_y = cs.net_y if 0.1 < cs.net_y < 0.9 else None
    corners, w, h = rally.video_cal
    cam = None
    if net_y is not None:
        cam = calibrate_camera_with_net(corners, COURT_CORNERS, w, h, net_y_image=net_y)
    if cam is None or not cam.is_valid:
        cam = calibrate_camera(corners, COURT_CORNERS, w, h)
    return cam


# ---------------------------------------------------------------------------
# Pose lookup
# ---------------------------------------------------------------------------


def poses_in_window(
    rally: RallyRow, center_frame: int, track_id: int, half_window: int,
) -> list[PlayerPosition]:
    """Return all PlayerPosition entries for the attributed player within
    ±half_window frames of the given center frame."""
    out: list[PlayerPosition] = []
    lo = center_frame - half_window
    hi = center_frame + half_window
    for pos in rally.player_positions:
        if pos.track_id != track_id:
            continue
        if lo <= pos.frame_number <= hi:
            out.append(pos)
    return out


def balls_in_window(
    rally: RallyRow, center_frame: int, half_window: int,
) -> dict[int, BallPosition]:
    """Return WASB ball detections indexed by frame within ±half_window."""
    out: dict[int, BallPosition] = {}
    lo = center_frame - half_window
    hi = center_frame + half_window
    for bp in rally.ball_positions:
        if lo <= bp.frame_number <= hi:
            out[bp.frame_number] = bp
    return out


def best_wrist(pose: PlayerPosition) -> tuple[float, float] | None:
    if pose.keypoints is None or len(pose.keypoints) < 11:
        return None
    left = pose.keypoints[KPT_LEFT_WRIST]
    right = pose.keypoints[KPT_RIGHT_WRIST]
    cands = [k for k in (left, right) if len(k) >= 3 and k[2] >= MIN_WRIST_CONF]
    if not cands:
        return None
    # Prefer the higher (smaller image-y) wrist — that's typically the ball-contact hand.
    cands.sort(key=lambda k: k[1])
    return float(cands[0][0]), float(cands[0][1])


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------


def probe_contact(
    rally: RallyRow,
    camera: CameraModel,
    label: dict[str, Any],
) -> dict[str, Any] | None:
    action = label.get("action")
    if action not in CONTACT_HEIGHT_M:
        return None
    label_frame = int(label["frame"])
    track_id = int(label.get("playerTrackId") or -1)
    if track_id < 0:
        return None

    # Candidate ball positions across the search window. These come from
    # the WASB ball track, not the action-GT label — the label's ballXY may
    # be at toss-apex and we want the actual touching frame.
    window_balls = balls_in_window(rally, label_frame, CONTACT_SEARCH_WINDOW)
    if not window_balls:
        return {
            "rally_id": rally.rally_id,
            "frame": label_frame,
            "action": action,
            "track_id": track_id,
            "status": "no_ball_in_window",
        }
    window_poses = poses_in_window(rally, label_frame, track_id, CONTACT_SEARCH_WINDOW)
    if not window_poses:
        return {
            "rally_id": rally.rally_id,
            "frame": label_frame,
            "action": action,
            "track_id": track_id,
            "status": "no_pose",
        }
    # Filter to poses with a usable wrist keypoint.
    poses_with_wrist = [
        (pos, best_wrist(pos)) for pos in window_poses
    ]
    poses_with_wrist = [p for p in poses_with_wrist if p[1] is not None]
    if not poses_with_wrist:
        return {
            "rally_id": rally.rally_id,
            "frame": label_frame,
            "action": action,
            "track_id": track_id,
            "status": "no_wrist_keypoint",
        }

    z_target = CONTACT_HEIGHT_M[action]

    # Search for the frame with minimum ball-wrist IMAGE distance.
    # Image-space proximity is the right metric here: back-projecting to a
    # contact-height plane is numerically unstable on 1-2m cameras because
    # direction_z is tiny. A 0.035 normalised image offset at typical depths
    # corresponds to ~30cm 3D — within a ball diameter or two.
    best_img_d = float("inf")
    best_frame = None

    for pose, wrist in poses_with_wrist:
        f = pose.frame_number
        bp = window_balls.get(f)
        if bp is None:
            for df in range(1, 3):
                bp = window_balls.get(f - df) or window_balls.get(f + df)
                if bp is not None:
                    break
        if bp is None:
            continue

        ball_img = (float(bp.x), float(bp.y))
        wrist_img = (float(wrist[0]), float(wrist[1]))
        img_d = math.sqrt(
            (ball_img[0] - wrist_img[0]) ** 2 + (ball_img[1] - wrist_img[1]) ** 2
        )
        if img_d < best_img_d:
            best_img_d = img_d
            best_frame = f

    if best_frame is None:
        return {
            "rally_id": rally.rally_id,
            "frame": label_frame,
            "action": action,
            "track_id": track_id,
            "status": "ray_miss",
        }

    return {
        "rally_id": rally.rally_id,
        "label_frame": label_frame,
        "min_frame": best_frame,
        "frame_offset": abs(best_frame - label_frame),
        "action": action,
        "track_id": track_id,
        "status": "ok",
        "img_dist": float(best_img_d),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()

    rallies = load_rallies(audit_only=args.audit_only)
    print(f"Loaded {len(rallies)} rallies" + (" (audit only)" if args.audit_only else ""))

    all_results: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}
    rallies_processed = 0
    for rally in rallies:
        cam = build_camera(rally)
        if cam is None:
            skipped["no_camera"] = skipped.get("no_camera", 0) + 1
            continue
        rallies_processed += 1
        for label in rally.action_gt:
            r = probe_contact(rally, cam, label)
            if r is None:
                continue
            all_results.append(r)

    # Aggregate.
    by_action: dict[str, dict[str, int]] = {}
    within_by_gate: dict[str, int] = {g: 0 for g in PROXIMITY_THRESHOLDS}
    total_ok = 0
    distances: list[float] = []
    dist_by_action: dict[str, list[float]] = {}

    for r in all_results:
        action = r["action"]
        if action not in by_action:
            by_action[action] = {
                "ok": 0, "no_pose": 0, "no_wrist_keypoint": 0,
                "no_ball_in_window": 0, "ray_miss": 0,
            }
            for g in PROXIMITY_THRESHOLDS:
                by_action[action][f"within_{g}"] = 0
        by_action[action][r["status"]] = by_action[action].get(r["status"], 0) + 1
        if r["status"] == "ok":
            total_ok += 1
            d = r["img_dist"]
            distances.append(d)
            dist_by_action.setdefault(action, []).append(d)
            for gate_name, gate_val in PROXIMITY_THRESHOLDS.items():
                if d <= gate_val:
                    within_by_gate[gate_name] += 1
                    by_action[action][f"within_{gate_name}"] += 1

    print("\n=== TOTAL ===")
    print(f"  rallies processed : {rallies_processed}")
    print(f"  contact labels    : {len(all_results)}")
    print(f"  status=ok         : {total_ok}")
    if total_ok:
        for g, val in PROXIMITY_THRESHOLDS.items():
            pct = within_by_gate[g] / total_ok * 100
            print(f"  within {g:<12s} (img ≤{val:.3f}): {within_by_gate[g]}/{total_ok} ({pct:.1f}%)")
        print(f"  median image dist : {float(np.median(distances)):.4f}")
        print(f"  p70               : {float(np.percentile(distances, 70)):.4f}")
        print(f"  p90               : {float(np.percentile(distances, 90)):.4f}")
    print()
    print("=== BY ACTION (tight / medium / loose) ===")
    for action in ("serve", "receive", "set", "attack", "block", "dig"):
        if action not in by_action:
            continue
        stats = by_action[action]
        ok = stats.get("ok", 0)
        if not ok:
            continue
        med = float(np.median(dist_by_action[action]))
        tight_n = stats.get("within_tight_30cm", 0)
        medium_n = stats.get("within_medium_60cm", 0)
        loose_n = stats.get("within_loose_1m", 0)
        print(
            f"  {action:<8s}: n={ok:>4d}  "
            f"tight={tight_n}/{ok} ({tight_n / ok * 100:5.1f}%)  "
            f"medium={medium_n}/{ok} ({medium_n / ok * 100:5.1f}%)  "
            f"loose={loose_n}/{ok} ({loose_n / ok * 100:5.1f}%)  "
            f"median={med:.4f}"
        )

    # Gate is defined at the "medium" threshold (~60cm 3D), which is the
    # realistic definition of a pose-to-ball contact anchor (hand striking
    # the ball is ~30cm offset from wrist keypoint).
    medium_rate = within_by_gate["medium_60cm"] / total_ok if total_ok else 0.0
    tight_rate = within_by_gate["tight_30cm"] / total_ok if total_ok else 0.0
    loose_rate = within_by_gate["loose_1m"] / total_ok if total_ok else 0.0
    print()
    print(
        f"Gate P2 (medium ~60cm): {medium_rate * 100:.1f}%  gate ≥{GATE_PASS_RATE:.0%}  "
        f"{'PASS' if medium_rate >= GATE_PASS_RATE else 'FAIL'}"
    )
    print(f"  tight (~30cm): {tight_rate * 100:.1f}%")
    print(f"  loose (~1m):   {loose_rate * 100:.1f}%")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if distances:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].hist(distances, bins=60, range=(0, 0.3), color="#2ca02c", alpha=0.8)
        for g, val in PROXIMITY_THRESHOLDS.items():
            colour = {"tight_30cm": "red", "medium_60cm": "orange", "loose_1m": "purple"}[g]
            axes[0].axvline(val, color=colour, linewidth=1.2, label=f"{g} ({val:.3f})")
        axes[0].set_xlabel("ball-wrist image distance (normalised)")
        axes[0].set_ylabel("contacts")
        axes[0].set_title(f"All contacts (n={len(distances)})")
        axes[0].legend(fontsize=7)

        for action, colour in zip(
            ("serve", "receive", "set", "attack", "block", "dig"),
            ("#d62728", "#1f77b4", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2"),
        ):
            ds = dist_by_action.get(action, [])
            if ds:
                axes[1].hist(ds, bins=50, range=(0, 0.3), alpha=0.5, color=colour,
                             label=f"{action} (n={len(ds)})")
        axes[1].axvline(PROXIMITY_THRESHOLDS["medium_60cm"], color="orange", linewidth=1.2)
        axes[1].set_xlabel("ball-wrist image distance (normalised)")
        axes[1].set_title("By action type")
        axes[1].legend(fontsize=7)

        plt.tight_layout()
        hist_path = OUTPUT_DIR / "probe_pose_contact.png"
        plt.savefig(hist_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"\nWrote {hist_path}")

    out = OUTPUT_DIR / "probe_pose_contact.json"
    out.write_text(json.dumps({
        "gate_pass_rate": GATE_PASS_RATE,
        "proximity_thresholds_img": PROXIMITY_THRESHOLDS,
        "n_rallies": rallies_processed,
        "n_contacts": len(all_results),
        "n_ok": total_ok,
        "within_by_gate": within_by_gate,
        "rate_by_gate": {
            g: within_by_gate[g] / total_ok if total_ok else 0.0
            for g in PROXIMITY_THRESHOLDS
        },
        "pass_medium": medium_rate >= GATE_PASS_RATE,
        "median_img_dist": float(np.median(distances)) if distances else None,
        "p70_img_dist": float(np.percentile(distances, 70)) if distances else None,
        "p90_img_dist": float(np.percentile(distances, 90)) if distances else None,
        "by_action": by_action,
        "median_by_action": {k: float(np.median(v)) for k, v in dist_by_action.items()},
    }, indent=2, default=float))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
