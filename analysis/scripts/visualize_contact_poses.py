"""Visualize frames at TP, FP, and FN contact points for pose analysis.

Extracts video frames at detected and ground-truth contact points, annotated
with ball position, nearest player bbox, and contact metadata. Optionally
overlays YOLOv8-pose keypoints to assess whether player pose is a useful
signal for filtering false positives.

Usage:
    cd analysis
    uv run python scripts/visualize_contact_poses.py              # Frames + annotations
    uv run python scripts/visualize_contact_poses.py --pose        # + keypoint skeleton + stats
    uv run python scripts/visualize_contact_poses.py --stats       # Stats only (no images)
    uv run python scripts/visualize_contact_poses.py --stats --window 5  # Temporal pose window (±5 frames)
    uv run python scripts/visualize_contact_poses.py --rally <id>  # Single rally
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from rallycut.core.video import Video
from rallycut.evaluation.tracking.db import get_video_path
from rallycut.tracking.action_classifier import classify_rally_actions
from rallycut.tracking.ball_tracker import BallPosition as BallPos
from rallycut.tracking.contact_detector import (
    Contact,
    detect_contacts,
)
from rallycut.tracking.player_tracker import PlayerPosition as PlayerPos
from scripts.eval_action_detection import (
    GtLabel,
    RallyData,
    load_rallies_with_action_gt,
    match_contacts,
)

console = Console()

OUTPUT_DIR = Path("output/contact_poses")

# COCO 17-keypoint skeleton connections for pose overlay
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # head
    (5, 6),  # shoulders
    (5, 7), (7, 9),  # left arm
    (6, 8), (8, 10),  # right arm
    (5, 11), (6, 12),  # torso
    (11, 12),  # hips
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
]

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


@dataclass
class ContactFrame:
    """A contact point to visualize."""

    rally_id: str
    video_id: str
    category: str  # "TP", "FP", "FN"
    frame: int  # rally-relative frame
    abs_frame: int  # absolute video frame
    contact: Contact | None  # predicted contact (None for FN)
    gt_label: GtLabel | None  # GT label (None for FP)
    action: str  # predicted or GT action label
    # Player positions at this frame for bbox drawing
    player_positions: list[PlayerPos]


@dataclass
class PoseData:
    """Pose features extracted for a contact frame."""

    category: str  # "TP", "FP", "FN"
    action: str
    rally_id: str
    frame: int
    pose_detected: bool = False
    # Arm elevation angles (degrees from horizontal, positive = raised)
    left_arm_angle: float | None = None
    right_arm_angle: float | None = None
    max_arm_angle: float | None = None  # max of left/right
    # Wrist elevation relative to shoulder (positive = above shoulder)
    left_wrist_elevation: float | None = None
    right_wrist_elevation: float | None = None
    max_wrist_elevation: float | None = None
    # Torso bend (angle of shoulder-hip from vertical)
    torso_angle: float | None = None
    # Number of visible keypoints
    num_keypoints: int = 0
    # Player bbox area (proxy for near/far)
    player_area: float = 0.0


@dataclass
class TemporalPoseData:
    """Temporal pose features extracted across ±N frames around a contact."""

    category: str  # "TP", "FP", "FN"
    action: str
    rally_id: str
    frame: int
    num_valid_frames: int = 0  # frames with successful pose detection
    # Arm angle dynamics
    peak_arm_angle: float | None = None  # max arm angle in window
    arm_angle_range: float | None = None  # max - min arm angle
    arm_angle_std: float | None = None  # std dev of arm angles
    max_arm_angle_rate: float | None = None  # max frame-to-frame |delta angle|
    # Wrist elevation dynamics
    wrist_above_shoulder_count: int = 0  # frames with wrist elevation > 0
    wrist_elevation_range: float | None = None  # max - min wrist elevation
    # Temporal asymmetry
    arm_rise_fall_delta: float | None = None  # max(pre) - min(post) arm angle
    peak_arm_angle_offset: int | None = None  # frame offset of peak (negative = before)
    # Torso dynamics
    peak_torso_angle: float | None = None
    torso_angle_range: float | None = None


def _build_contact_frames(
    rallies: list[RallyData],
) -> list[ContactFrame]:
    """Run contact detection, match to GT, and build ContactFrame list."""
    frames: list[ContactFrame] = []

    for rally in rallies:
        if not rally.ball_positions_json:
            continue

        ball_positions = [
            BallPos(
                frame_number=bp["frameNumber"],
                x=bp["x"],
                y=bp["y"],
                confidence=bp.get("confidence", 1.0),
            )
            for bp in rally.ball_positions_json
            if bp.get("x", 0) > 0 or bp.get("y", 0) > 0
        ]

        player_positions: list[PlayerPos] = []
        if rally.positions_json:
            player_positions = [
                PlayerPos(
                    frame_number=pp["frameNumber"],
                    track_id=pp["trackId"],
                    x=pp["x"],
                    y=pp["y"],
                    width=pp["width"],
                    height=pp["height"],
                    confidence=pp.get("confidence", 1.0),
                )
                for pp in rally.positions_json
            ]

        contact_seq = detect_contacts(
            ball_positions=ball_positions,
            player_positions=player_positions,
            net_y=rally.court_split_y,
            frame_count=rally.frame_count or None,
        )

        rally_actions = classify_rally_actions(contact_seq, rally.rally_id)
        pred_actions = [a.to_dict() for a in rally_actions.actions]

        # Build player position lookup: frame -> list of PlayerPos
        player_by_frame: dict[int, list[PlayerPos]] = {}
        for pp in player_positions:
            player_by_frame.setdefault(pp.frame_number, []).append(pp)

        # Match GT to predictions
        tolerance_frames = max(1, round(rally.fps * 167 / 1000))
        matches, unmatched_preds = match_contacts(
            rally.gt_labels, pred_actions, tolerance=tolerance_frames,
        )

        # Build contact lookup: frame -> Contact
        contact_by_frame: dict[int, Contact] = {c.frame: c for c in contact_seq.contacts}

        # Compute absolute frame offset
        video_fps = rally.video_fps
        abs_offset = int(rally.start_ms / 1000 * video_fps)

        # TP and FN from matches
        for m in matches:
            if m.pred_frame is not None:
                # TP
                contact = contact_by_frame.get(m.pred_frame)
                frames.append(ContactFrame(
                    rally_id=rally.rally_id,
                    video_id=rally.video_id,
                    category="TP",
                    frame=m.pred_frame,
                    abs_frame=abs_offset + m.pred_frame,
                    contact=contact,
                    gt_label=_find_gt(rally.gt_labels, m.gt_frame),
                    action=m.pred_action or "unknown",
                    player_positions=player_by_frame.get(m.pred_frame, []),
                ))
            else:
                # FN
                frames.append(ContactFrame(
                    rally_id=rally.rally_id,
                    video_id=rally.video_id,
                    category="FN",
                    frame=m.gt_frame,
                    abs_frame=abs_offset + m.gt_frame,
                    contact=None,
                    gt_label=_find_gt(rally.gt_labels, m.gt_frame),
                    action=m.gt_action,
                    player_positions=player_by_frame.get(m.gt_frame, []),
                ))

        # FP from unmatched predictions
        for pred in unmatched_preds:
            pred_frame = pred.get("frame", 0)
            contact = contact_by_frame.get(pred_frame)
            frames.append(ContactFrame(
                rally_id=rally.rally_id,
                video_id=rally.video_id,
                category="FP",
                frame=pred_frame,
                abs_frame=abs_offset + pred_frame,
                contact=contact,
                gt_label=None,
                action=pred.get("action", "unknown"),
                player_positions=player_by_frame.get(pred_frame, []),
            ))

    return frames


def _find_gt(labels: list[GtLabel], frame: int) -> GtLabel | None:
    """Find GT label at the given frame."""
    for gt in labels:
        if gt.frame == frame:
            return gt
    return None


# Colors (BGR)
_COLOR_TP = (0, 200, 0)    # green
_COLOR_FP = (0, 0, 220)    # red
_COLOR_FN = (220, 140, 0)  # blue
_COLOR_BALL = (0, 255, 255)  # yellow
_COLOR_POSE = (255, 128, 0)  # orange


def _category_color(cat: str) -> tuple[int, int, int]:
    if cat == "TP":
        return _COLOR_TP
    elif cat == "FP":
        return _COLOR_FP
    return _COLOR_FN


def _annotate_frame(
    frame: np.ndarray,
    cf: ContactFrame,
) -> np.ndarray:
    """Draw ball, nearest player bbox, and text annotations."""
    h, w = frame.shape[:2]
    out = frame.copy()
    color = _category_color(cf.category)

    # Ball position
    if cf.contact:
        bx, by = int(cf.contact.ball_x * w), int(cf.contact.ball_y * h)
        cv2.circle(out, (bx, by), 10, _COLOR_BALL, -1)
        cv2.circle(out, (bx, by), 12, (0, 0, 0), 2)
    elif cf.gt_label and cf.gt_label.ball_x is not None and cf.gt_label.ball_y is not None:
        bx, by = int(cf.gt_label.ball_x * w), int(cf.gt_label.ball_y * h)
        cv2.circle(out, (bx, by), 10, _COLOR_BALL, -1)
        cv2.circle(out, (bx, by), 12, (0, 0, 0), 2)

    # Find nearest player to ball/contact
    nearest_pp = _find_nearest_player(cf)
    if nearest_pp:
        x1 = int((nearest_pp.x - nearest_pp.width / 2) * w)
        y1 = int((nearest_pp.y - nearest_pp.height / 2) * h)
        x2 = int((nearest_pp.x + nearest_pp.width / 2) * w)
        y2 = int((nearest_pp.y + nearest_pp.height / 2) * h)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)

    # Draw all other players with thin gray boxes
    for pp in cf.player_positions:
        if pp is nearest_pp:
            continue
        px1 = int((pp.x - pp.width / 2) * w)
        py1 = int((pp.y - pp.height / 2) * h)
        px2 = int((pp.x + pp.width / 2) * w)
        py2 = int((pp.y + pp.height / 2) * h)
        cv2.rectangle(out, (px1, py1), (px2, py2), (160, 160, 160), 1)

    # Text info
    lines = [
        f"{cf.category} | {cf.action} | f{cf.frame}",
    ]
    if cf.contact:
        lines.append(
            f"vel={cf.contact.velocity:.4f}  dir={cf.contact.direction_change_deg:.0f}deg  "
            f"pdist={cf.contact.player_distance:.3f}"
        )
        lines.append(f"side={cf.contact.court_side}  net={cf.contact.is_at_net}")
    if cf.category == "FN" and cf.gt_label:
        lines.append(f"GT action: {cf.gt_label.action}  GT track: {cf.gt_label.player_track_id}")

    # Draw text with background
    y_text = 30
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(out, (8, y_text - th - 4), (12 + tw, y_text + 4), (0, 0, 0), -1)
        cv2.putText(out, line, (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        y_text += th + 12

    return out


def _find_nearest_player(cf: ContactFrame) -> PlayerPos | None:
    """Find player closest to ball/contact position."""
    if not cf.player_positions:
        return None

    # Use contact's attributed track_id if available
    if cf.contact and cf.contact.player_track_id >= 0:
        for pp in cf.player_positions:
            if pp.track_id == cf.contact.player_track_id:
                return pp

    # Fallback: find nearest by distance to ball
    ref_x, ref_y = 0.5, 0.5
    if cf.contact:
        ref_x, ref_y = cf.contact.ball_x, cf.contact.ball_y
    elif cf.gt_label and cf.gt_label.ball_x is not None and cf.gt_label.ball_y is not None:
        ref_x, ref_y = cf.gt_label.ball_x, cf.gt_label.ball_y

    best = None
    best_dist = float("inf")
    for pp in cf.player_positions:
        d = ((pp.x - ref_x) ** 2 + (pp.y - ref_y) ** 2) ** 0.5
        if d < best_dist:
            best_dist = d
            best = pp
    return best


def _overlay_pose(
    frame: np.ndarray,
    cf: ContactFrame,
    pose_model: object,
) -> np.ndarray:
    """Run YOLOv8-pose on nearest player region and draw skeleton."""
    nearest = _find_nearest_player(cf)
    if nearest is None:
        return frame

    h, w = frame.shape[:2]
    out = frame.copy()

    # Expand bbox by 1.5x for pose context
    cx, cy = nearest.x * w, nearest.y * h
    bw, bh = nearest.width * w * 1.5, nearest.height * h * 1.5
    x1 = max(0, int(cx - bw / 2))
    y1 = max(0, int(cy - bh / 2))
    x2 = min(w, int(cx + bw / 2))
    y2 = min(h, int(cy + bh / 2))

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return out

    # Run pose detection on crop
    results = pose_model.predict(crop, verbose=False)  # type: ignore[attr-defined]
    if not results or len(results) == 0:
        return out

    result = results[0]
    if result.keypoints is None or len(result.keypoints) == 0:
        return out

    # Get keypoints of first detected person (most prominent in crop)
    kps = result.keypoints.data[0].cpu().numpy()  # shape (17, 3): x, y, conf

    # Draw skeleton
    for i, j in SKELETON:
        if kps[i][2] > 0.3 and kps[j][2] > 0.3:
            pt1 = (int(kps[i][0]) + x1, int(kps[i][1]) + y1)
            pt2 = (int(kps[j][0]) + x1, int(kps[j][1]) + y1)
            cv2.line(out, pt1, pt2, _COLOR_POSE, 2)

    # Draw keypoints
    for k in range(17):
        if kps[k][2] > 0.3:
            pt = (int(kps[k][0]) + x1, int(kps[k][1]) + y1)
            cv2.circle(out, pt, 4, _COLOR_POSE, -1)

    # Compute and display arm elevation angles
    arm_info = _compute_arm_angles(kps)
    if arm_info:
        y_pos = frame.shape[0] - 40
        for info in arm_info:
            (tw, th), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (8, y_pos - th - 4), (12 + tw, y_pos + 4), (0, 0, 0), -1)
            cv2.putText(out, info, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _COLOR_POSE, 1)
            y_pos -= th + 12

    return out


def _compute_arm_angles(kps: np.ndarray) -> list[str]:
    """Compute arm elevation angles from keypoints."""
    info: list[str] = []

    # Left arm: shoulder(5) -> elbow(7) -> wrist(9)
    if kps[5][2] > 0.3 and kps[7][2] > 0.3:
        dx = kps[7][0] - kps[5][0]
        dy = kps[7][1] - kps[5][1]  # y increases downward
        angle = np.degrees(np.arctan2(-dy, dx))  # elevation from horizontal
        info.append(f"L arm: {angle:.0f}deg")

    # Right arm: shoulder(6) -> elbow(8) -> wrist(10)
    if kps[6][2] > 0.3 and kps[8][2] > 0.3:
        dx = kps[8][0] - kps[6][0]
        dy = kps[8][1] - kps[6][1]
        angle = np.degrees(np.arctan2(-dy, dx))
        info.append(f"R arm: {angle:.0f}deg")

    return info


def _compute_crop_bbox(
    nearest: PlayerPos,
    h: int,
    w: int,
) -> tuple[int, int, int, int] | None:
    """Compute 1.5x-expanded bbox around a player in pixel coords."""
    cx, cy = nearest.x * w, nearest.y * h
    bw, bh = nearest.width * w * 1.5, nearest.height * h * 1.5
    x1 = max(0, int(cx - bw / 2))
    y1 = max(0, int(cy - bh / 2))
    x2 = min(w, int(cx + bw / 2))
    y2 = min(h, int(cy + bh / 2))

    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _extract_pose_for_crop(
    frame: np.ndarray,
    crop_bbox: tuple[int, int, int, int],
    pose_model: object,
) -> PoseData | None:
    """Run pose on a pre-defined crop region. Returns PoseData or None if no pose."""
    x1, y1, x2, y2 = crop_bbox
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    results = pose_model.predict(crop, verbose=False)  # type: ignore[attr-defined]
    if not results or len(results) == 0:
        return None

    result = results[0]
    if result.keypoints is None or len(result.keypoints) == 0:
        return None

    kps = result.keypoints.data[0].cpu().numpy()  # (17, 3)

    pd = PoseData(category="", action="", rally_id="", frame=0)
    pd.pose_detected = True
    pd.num_keypoints = int((kps[:, 2] > 0.3).sum())

    # Arm angles (shoulder -> elbow elevation from horizontal)
    if kps[5][2] > 0.3 and kps[7][2] > 0.3:
        dx = kps[7][0] - kps[5][0]
        dy = kps[7][1] - kps[5][1]
        pd.left_arm_angle = float(np.degrees(np.arctan2(-dy, dx)))

    if kps[6][2] > 0.3 and kps[8][2] > 0.3:
        dx = kps[8][0] - kps[6][0]
        dy = kps[8][1] - kps[6][1]
        pd.right_arm_angle = float(np.degrees(np.arctan2(-dy, dx)))

    angles = [a for a in [pd.left_arm_angle, pd.right_arm_angle] if a is not None]
    pd.max_arm_angle = max(angles) if angles else None

    # Wrist elevation relative to shoulder (negative dy = wrist above shoulder)
    if kps[5][2] > 0.3 and kps[9][2] > 0.3:
        pd.left_wrist_elevation = float(kps[5][1] - kps[9][1])  # positive = above
    if kps[6][2] > 0.3 and kps[10][2] > 0.3:
        pd.right_wrist_elevation = float(kps[6][1] - kps[10][1])

    elevations = [e for e in [pd.left_wrist_elevation, pd.right_wrist_elevation] if e is not None]
    pd.max_wrist_elevation = max(elevations) if elevations else None

    # Torso angle: midpoint(shoulders) to midpoint(hips) vs vertical
    if all(kps[i][2] > 0.3 for i in [5, 6, 11, 12]):
        shoulder_mid = (kps[5][:2] + kps[6][:2]) / 2
        hip_mid = (kps[11][:2] + kps[12][:2]) / 2
        dx = hip_mid[0] - shoulder_mid[0]
        dy = hip_mid[1] - shoulder_mid[1]
        pd.torso_angle = float(np.degrees(np.arctan2(abs(dx), abs(dy))))

    return pd


def _extract_pose_data(
    frame: np.ndarray,
    cf: ContactFrame,
    pose_model: object,
) -> PoseData:
    """Run pose detection and extract structured pose features (single frame)."""
    pd = PoseData(
        category=cf.category,
        action=cf.action,
        rally_id=cf.rally_id,
        frame=cf.frame,
    )

    nearest = _find_nearest_player(cf)
    if nearest is None:
        return pd

    pd.player_area = nearest.width * nearest.height

    h, w = frame.shape[:2]
    crop_bbox = _compute_crop_bbox(nearest, h, w)
    if crop_bbox is None:
        return pd

    result = _extract_pose_for_crop(frame, crop_bbox, pose_model)
    if result is None:
        return pd

    # Copy extracted fields
    pd.pose_detected = result.pose_detected
    pd.num_keypoints = result.num_keypoints
    pd.left_arm_angle = result.left_arm_angle
    pd.right_arm_angle = result.right_arm_angle
    pd.max_arm_angle = result.max_arm_angle
    pd.left_wrist_elevation = result.left_wrist_elevation
    pd.right_wrist_elevation = result.right_wrist_elevation
    pd.max_wrist_elevation = result.max_wrist_elevation
    pd.torso_angle = result.torso_angle

    return pd


def _compute_temporal_features(
    td: TemporalPoseData,
    frame_poses: list[PoseData | None],
    window: int,
) -> None:
    """Compute temporal metrics from a sequence of PoseData and fill into td.

    frame_poses has length 2*window+1, index `window` is the contact frame.
    """
    # Collect per-feature sequences from valid frames
    arm_angles: list[float] = []
    wrist_elevations: list[float] = []
    torso_angles: list[float] = []

    for p in frame_poses:
        if p is not None and p.pose_detected:
            if p.max_arm_angle is not None:
                arm_angles.append(p.max_arm_angle)
            if p.max_wrist_elevation is not None:
                wrist_elevations.append(p.max_wrist_elevation)
            if p.torso_angle is not None:
                torso_angles.append(p.torso_angle)

    td.num_valid_frames = sum(1 for p in frame_poses if p is not None and p.pose_detected)

    if len(arm_angles) >= 2:
        td.peak_arm_angle = max(arm_angles)
        td.arm_angle_range = max(arm_angles) - min(arm_angles)
        td.arm_angle_std = statistics.stdev(arm_angles)

        # Max frame-to-frame rate (arm_angles is already in temporal order)
        td.max_arm_angle_rate = max(
            abs(arm_angles[i] - arm_angles[i - 1])
            for i in range(1, len(arm_angles))
        )

        # Peak arm angle offset (relative to contact frame at index=window)
        peak_val = td.peak_arm_angle
        for i, p in enumerate(frame_poses):
            if p is not None and p.pose_detected and p.max_arm_angle == peak_val:
                td.peak_arm_angle_offset = i - window
                break

        # Arm rise/fall delta: max(pre-contact) - min(post-contact)
        pre_angles: list[float] = []
        post_angles: list[float] = []
        for i, p in enumerate(frame_poses):
            if p is not None and p.pose_detected and p.max_arm_angle is not None:
                if i < window:
                    pre_angles.append(p.max_arm_angle)
                elif i > window:
                    post_angles.append(p.max_arm_angle)
        if pre_angles and post_angles:
            td.arm_rise_fall_delta = max(pre_angles) - min(post_angles)

    # Wrist elevation features
    if wrist_elevations:
        td.wrist_above_shoulder_count = sum(1 for e in wrist_elevations if e > 0)
        if len(wrist_elevations) >= 2:
            td.wrist_elevation_range = max(wrist_elevations) - min(wrist_elevations)

    # Torso features
    if torso_angles:
        td.peak_torso_angle = max(torso_angles)
        if len(torso_angles) >= 2:
            td.torso_angle_range = max(torso_angles) - min(torso_angles)


def _extract_temporal_pose_data(
    video: Video,
    cf: ContactFrame,
    pose_model: object,
    window: int,
    h: int,
    w: int,
) -> TemporalPoseData:
    """Extract pose across ±window frames and compute temporal features."""
    td = TemporalPoseData(
        category=cf.category,
        action=cf.action,
        rally_id=cf.rally_id,
        frame=cf.frame,
    )

    # Compute crop bbox from contact frame's nearest player (once)
    nearest = _find_nearest_player(cf)
    if nearest is None:
        return td
    crop_bbox = _compute_crop_bbox(nearest, h, w)
    if crop_bbox is None:
        return td

    # Read ±window frames and extract pose using same crop region
    frame_poses: list[PoseData | None] = []
    for offset in range(-window, window + 1):
        abs_frame = cf.abs_frame + offset
        if abs_frame < 0:
            frame_poses.append(None)
            continue

        raw_frame = video.read_frame(abs_frame)
        if raw_frame is None:
            frame_poses.append(None)
            continue

        frame_poses.append(_extract_pose_for_crop(raw_frame, crop_bbox, pose_model))

    _compute_temporal_features(td, frame_poses, window)
    return td


def _print_temporal_stats(temporal_data: list[TemporalPoseData]) -> None:
    """Print aggregate temporal pose statistics comparing TP vs FP."""
    tp = [t for t in temporal_data if t.category == "TP"]
    fp = [t for t in temporal_data if t.category == "FP"]

    console.print("\n[bold underline]Temporal Pose Analysis[/bold underline]\n")
    console.print(f"  TP: {len(tp)} contacts, FP: {len(fp)} contacts\n")

    # Valid frame counts
    tp_valid = [t.num_valid_frames for t in tp if t.num_valid_frames > 0]
    fp_valid = [t.num_valid_frames for t in fp if t.num_valid_frames > 0]
    if tp_valid and fp_valid:
        console.print(
            f"  Valid frames per window: "
            f"TP median={statistics.median(tp_valid):.0f}, "
            f"FP median={statistics.median(fp_valid):.0f}\n"
        )

    def _dist(
        label: str,
        tp_vals: list[float],
        fp_vals: list[float],
    ) -> None:
        if len(tp_vals) < 3 or len(fp_vals) < 3:
            console.print(f"  [dim]{label}: insufficient data (TP={len(tp_vals)}, FP={len(fp_vals)})[/dim]")
            return
        tp_med = statistics.median(tp_vals)
        fp_med = statistics.median(fp_vals)
        tp_sorted = sorted(tp_vals)
        fp_sorted = sorted(fp_vals)
        tp_p25 = tp_sorted[len(tp_vals) // 4]
        fp_p25 = fp_sorted[len(fp_vals) // 4]
        tp_p75 = tp_sorted[3 * len(tp_vals) // 4]
        fp_p75 = fp_sorted[3 * len(fp_vals) // 4]

        # Check IQR overlap for separation signal
        overlap = tp_p25 <= fp_p75 and fp_p25 <= tp_p75
        if tp_med > fp_med * 1.3:
            sep = "[green]>>>[/green]" if not overlap else ">>>"
        elif fp_med > tp_med * 1.3:
            sep = "[red]<<<[/red]" if not overlap else "<<<"
        else:
            sep = " ~ "

        console.print(f"  [bold]{label}[/bold]  ({sep})")
        console.print(
            f"    TP (n={len(tp_vals):3d}): "
            f"median={tp_med:7.1f}  p25={tp_p25:7.1f}  p75={tp_p75:7.1f}  "
            f"range=[{min(tp_vals):.1f}, {max(tp_vals):.1f}]"
        )
        console.print(
            f"    FP (n={len(fp_vals):3d}): "
            f"median={fp_med:7.1f}  p25={fp_p25:7.1f}  p75={fp_p75:7.1f}  "
            f"range=[{min(fp_vals):.1f}, {max(fp_vals):.1f}]"
        )
        console.print()

    console.print("[bold underline]Temporal Feature Distributions (TP vs FP)[/bold underline]\n")

    _dist(
        "Arm angle range (deg, higher=more transition)",
        [t.arm_angle_range for t in tp if t.arm_angle_range is not None],
        [t.arm_angle_range for t in fp if t.arm_angle_range is not None],
    )
    _dist(
        "Arm rise/fall delta (deg, positive=wind-up then drop)",
        [t.arm_rise_fall_delta for t in tp if t.arm_rise_fall_delta is not None],
        [t.arm_rise_fall_delta for t in fp if t.arm_rise_fall_delta is not None],
    )
    _dist(
        "Max arm angle rate (deg/frame, higher=faster motion)",
        [t.max_arm_angle_rate for t in tp if t.max_arm_angle_rate is not None],
        [t.max_arm_angle_rate for t in fp if t.max_arm_angle_rate is not None],
    )
    _dist(
        "Peak arm angle (deg, higher=more raised)",
        [t.peak_arm_angle for t in tp if t.peak_arm_angle is not None],
        [t.peak_arm_angle for t in fp if t.peak_arm_angle is not None],
    )
    _dist(
        "Arm angle std (deg, higher=more variation)",
        [t.arm_angle_std for t in tp if t.arm_angle_std is not None],
        [t.arm_angle_std for t in fp if t.arm_angle_std is not None],
    )
    _dist(
        "Wrist above shoulder count (frames)",
        [float(t.wrist_above_shoulder_count) for t in tp if t.num_valid_frames > 0],
        [float(t.wrist_above_shoulder_count) for t in fp if t.num_valid_frames > 0],
    )
    _dist(
        "Wrist elevation range (px)",
        [t.wrist_elevation_range for t in tp if t.wrist_elevation_range is not None],
        [t.wrist_elevation_range for t in fp if t.wrist_elevation_range is not None],
    )
    _dist(
        "Peak torso angle (deg from vertical)",
        [t.peak_torso_angle for t in tp if t.peak_torso_angle is not None],
        [t.peak_torso_angle for t in fp if t.peak_torso_angle is not None],
    )
    _dist(
        "Torso angle range (deg, higher=more lunging)",
        [t.torso_angle_range for t in tp if t.torso_angle_range is not None],
        [t.torso_angle_range for t in fp if t.torso_angle_range is not None],
    )
    _dist(
        "Peak arm angle offset (frames, negative=before contact)",
        [float(t.peak_arm_angle_offset) for t in tp if t.peak_arm_angle_offset is not None],
        [float(t.peak_arm_angle_offset) for t in fp if t.peak_arm_angle_offset is not None],
    )

    # Per-action breakdown for key temporal features
    console.print("[bold underline]Arm Angle Range by Action (TP only)[/bold underline]\n")
    action_table = Table()
    action_table.add_column("Action", style="bold")
    action_table.add_column("N", justify="right")
    action_table.add_column("Median", justify="right")
    action_table.add_column("P25", justify="right")
    action_table.add_column("P75", justify="right")
    action_table.add_column("Range", justify="right")

    actions_seen: dict[str, list[float]] = {}
    for t in tp:
        if t.arm_angle_range is not None:
            actions_seen.setdefault(t.action, []).append(t.arm_angle_range)
    for action in sorted(actions_seen.keys()):
        vals = sorted(actions_seen[action])
        if len(vals) < 2:
            continue
        action_table.add_row(
            action,
            str(len(vals)),
            f"{statistics.median(vals):.1f}",
            f"{vals[len(vals)//4]:.1f}",
            f"{vals[3*len(vals)//4]:.1f}",
            f"[{vals[0]:.1f}, {vals[-1]:.1f}]",
        )
    console.print(action_table)

    # FP breakdown
    console.print("\n[bold underline]Arm Angle Range by Action (FP only)[/bold underline]\n")
    fp_table = Table()
    fp_table.add_column("Pred Action", style="bold")
    fp_table.add_column("N", justify="right")
    fp_table.add_column("Median", justify="right")
    fp_table.add_column("P25", justify="right")
    fp_table.add_column("P75", justify="right")

    fp_actions: dict[str, list[float]] = {}
    for t in fp:
        if t.arm_angle_range is not None:
            fp_actions.setdefault(t.action, []).append(t.arm_angle_range)
    for action in sorted(fp_actions.keys()):
        vals = sorted(fp_actions[action])
        if len(vals) < 2:
            continue
        fp_table.add_row(
            action,
            str(len(vals)),
            f"{statistics.median(vals):.1f}",
            f"{vals[len(vals)//4]:.1f}",
            f"{vals[3*len(vals)//4]:.1f}",
        )
    console.print(fp_table)

    # Threshold analysis: for each action, show FP removed vs TP lost at various cutoffs
    console.print("\n[bold underline]Threshold Analysis (suppress contact if feature < threshold)[/bold underline]")

    for feature_name, getter in [
        ("arm_angle_range", lambda t: t.arm_angle_range),
        ("max_arm_angle_rate", lambda t: t.max_arm_angle_rate),
        ("arm_angle_std", lambda t: t.arm_angle_std),
    ]:
        console.print(f"\n  [bold]{feature_name}[/bold]\n")

        # Per-action and all-actions
        for action_filter in [None, "attack", "serve", "set"]:
            label = action_filter or "all"
            tp_vals = [getter(t) for t in tp if getter(t) is not None
                       and (action_filter is None or t.action == action_filter)]
            fp_vals = [getter(t) for t in fp if getter(t) is not None
                       and (action_filter is None or t.action == action_filter)]

            if not tp_vals or not fp_vals:
                continue

            thresh_table = Table(title=f"{label} (TP={len(tp_vals)}, FP={len(fp_vals)})")
            thresh_table.add_column("Threshold", justify="right")
            thresh_table.add_column("FP removed", justify="right")
            thresh_table.add_column("TP lost", justify="right")
            thresh_table.add_column("Net gain", justify="right")
            thresh_table.add_column("Precision", justify="right")

            thresholds = [5, 8, 10, 15, 20, 30]
            for t in thresholds:
                fp_removed = sum(1 for v in fp_vals if v < t)
                tp_lost = sum(1 for v in tp_vals if v < t)
                net = fp_removed - tp_lost
                remaining_tp = len(tp_vals) - tp_lost
                remaining_fp = len(fp_vals) - fp_removed
                prec = remaining_tp / max(1, remaining_tp + remaining_fp)
                style = "green" if net > 0 and tp_lost == 0 else (
                    "yellow" if net > 0 else "dim"
                )
                thresh_table.add_row(
                    f"< {t}",
                    f"{fp_removed}/{len(fp_vals)}",
                    f"{tp_lost}/{len(tp_vals)}",
                    f"[{style}]{net:+d}[/{style}]",
                    f"{prec:.0%}",
                )
            console.print(thresh_table)


def _print_pose_stats(pose_data: list[PoseData]) -> None:
    """Print aggregate pose statistics comparing TP vs FP."""
    tp = [p for p in pose_data if p.category == "TP"]
    fp = [p for p in pose_data if p.category == "FP"]
    fn = [p for p in pose_data if p.category == "FN"]

    # Detection rate
    console.print("\n[bold underline]Pose Detection Rate[/bold underline]\n")
    for label, items in [("TP", tp), ("FP", fp), ("FN", fn)]:
        detected = sum(1 for p in items if p.pose_detected)
        total = len(items)
        rate = detected / max(1, total)
        avg_kps = (
            statistics.mean(p.num_keypoints for p in items if p.pose_detected)
            if any(p.pose_detected for p in items) else 0
        )
        console.print(
            f"  {label}: {detected}/{total} ({rate:.0%}) detected, "
            f"avg {avg_kps:.1f} keypoints"
        )

    # Feature distributions
    console.print("\n[bold underline]Pose Feature Distributions (TP vs FP)[/bold underline]\n")

    def _dist(
        label: str,
        tp_vals: list[float],
        fp_vals: list[float],
    ) -> None:
        if len(tp_vals) < 3 or len(fp_vals) < 3:
            console.print(f"  [dim]{label}: insufficient data (TP={len(tp_vals)}, FP={len(fp_vals)})[/dim]")
            return
        tp_med = statistics.median(tp_vals)
        fp_med = statistics.median(fp_vals)
        tp_sorted = sorted(tp_vals)
        fp_sorted = sorted(fp_vals)
        tp_p25 = tp_sorted[len(tp_vals) // 4]
        fp_p25 = fp_sorted[len(fp_vals) // 4]
        tp_p75 = tp_sorted[3 * len(tp_vals) // 4]
        fp_p75 = fp_sorted[3 * len(fp_vals) // 4]
        sep = ">>>" if tp_med > fp_med * 1.3 else ("<<<" if fp_med > tp_med * 1.3 else " ~ ")
        console.print(f"  [bold]{label}[/bold]  ({sep})")
        console.print(
            f"    TP (n={len(tp_vals):3d}): "
            f"median={tp_med:7.1f}  p25={tp_p25:7.1f}  p75={tp_p75:7.1f}  "
            f"range=[{min(tp_vals):.1f}, {max(tp_vals):.1f}]"
        )
        console.print(
            f"    FP (n={len(fp_vals):3d}): "
            f"median={fp_med:7.1f}  p25={fp_p25:7.1f}  p75={fp_p75:7.1f}  "
            f"range=[{min(fp_vals):.1f}, {max(fp_vals):.1f}]"
        )
        console.print()

    _dist(
        "Max arm angle (deg, higher=raised)",
        [p.max_arm_angle for p in tp if p.max_arm_angle is not None],
        [p.max_arm_angle for p in fp if p.max_arm_angle is not None],
    )
    _dist(
        "Left arm angle (deg)",
        [p.left_arm_angle for p in tp if p.left_arm_angle is not None],
        [p.left_arm_angle for p in fp if p.left_arm_angle is not None],
    )
    _dist(
        "Right arm angle (deg)",
        [p.right_arm_angle for p in tp if p.right_arm_angle is not None],
        [p.right_arm_angle for p in fp if p.right_arm_angle is not None],
    )
    _dist(
        "Max wrist elevation (px, higher=above shoulder)",
        [p.max_wrist_elevation for p in tp if p.max_wrist_elevation is not None],
        [p.max_wrist_elevation for p in fp if p.max_wrist_elevation is not None],
    )
    _dist(
        "Torso angle (deg from vertical, higher=leaning)",
        [p.torso_angle for p in tp if p.torso_angle is not None],
        [p.torso_angle for p in fp if p.torso_angle is not None],
    )
    _dist(
        "Player bbox area (proxy for distance)",
        [p.player_area for p in tp if p.pose_detected],
        [p.player_area for p in fp if p.pose_detected],
    )

    # Per-action breakdown
    console.print("[bold underline]Max Arm Angle by Action (TP only)[/bold underline]\n")
    action_table = Table()
    action_table.add_column("Action", style="bold")
    action_table.add_column("N", justify="right")
    action_table.add_column("Median", justify="right")
    action_table.add_column("P25", justify="right")
    action_table.add_column("P75", justify="right")
    action_table.add_column("Range", justify="right")

    actions_seen: dict[str, list[float]] = {}
    for p in tp:
        if p.max_arm_angle is not None:
            actions_seen.setdefault(p.action, []).append(p.max_arm_angle)
    for action in sorted(actions_seen.keys()):
        vals = sorted(actions_seen[action])
        if len(vals) < 2:
            continue
        action_table.add_row(
            action,
            str(len(vals)),
            f"{statistics.median(vals):.1f}",
            f"{vals[len(vals)//4]:.1f}",
            f"{vals[3*len(vals)//4]:.1f}",
            f"[{vals[0]:.1f}, {vals[-1]:.1f}]",
        )
    console.print(action_table)

    # FP breakdown by predicted action
    console.print("\n[bold underline]Max Arm Angle by Action (FP only)[/bold underline]\n")
    fp_action_table = Table()
    fp_action_table.add_column("Pred Action", style="bold")
    fp_action_table.add_column("N", justify="right")
    fp_action_table.add_column("Median", justify="right")
    fp_action_table.add_column("P25", justify="right")
    fp_action_table.add_column("P75", justify="right")

    fp_actions: dict[str, list[float]] = {}
    for p in fp:
        if p.max_arm_angle is not None:
            fp_actions.setdefault(p.action, []).append(p.max_arm_angle)
    for action in sorted(fp_actions.keys()):
        vals = sorted(fp_actions[action])
        if len(vals) < 2:
            continue
        fp_action_table.add_row(
            action,
            str(len(vals)),
            f"{statistics.median(vals):.1f}",
            f"{vals[len(vals)//4]:.1f}",
            f"{vals[3*len(vals)//4]:.1f}",
        )
    console.print(fp_action_table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize contact poses (Phase 0)")
    parser.add_argument("--rally", type=str, help="Specific rally ID")
    parser.add_argument("--pose", action="store_true", help="Overlay YOLOv8-pose keypoints + print stats")
    parser.add_argument("--stats", action="store_true", help="Pose stats only (no image saving)")
    parser.add_argument(
        "--window", type=int, default=0,
        help="Temporal window size: extract pose across +/-N frames per contact",
    )
    args = parser.parse_args()

    # --stats implies --window 5 by default (unless explicitly set)
    window = args.window
    if args.stats and window == 0:
        window = 5

    use_temporal = window > 0
    use_pose = args.pose or args.stats or use_temporal
    save_images = not args.stats

    rallies = load_rallies_with_action_gt(rally_id=args.rally)
    if not rallies:
        console.print("[red]No rallies with action GT found.[/red]")
        return

    console.print(f"[bold]Building contact frames from {len(rallies)} rallies...[/bold]")
    contact_frames = _build_contact_frames(rallies)

    tp_count = sum(1 for cf in contact_frames if cf.category == "TP")
    fp_count = sum(1 for cf in contact_frames if cf.category == "FP")
    fn_count = sum(1 for cf in contact_frames if cf.category == "FN")
    console.print(f"  {tp_count} TP, {fp_count} FP, {fn_count} FN")
    if use_temporal:
        total_frames = len(contact_frames) * (2 * window + 1)
        console.print(f"  Temporal window: +/-{window} frames ({total_frames} total frames)")
    console.print()

    if not contact_frames:
        console.print("[yellow]No contact frames to visualize.[/yellow]")
        return

    if save_images:
        for cat in ("TP", "FP", "FN"):
            (OUTPUT_DIR / cat).mkdir(parents=True, exist_ok=True)

    # Load pose model if needed
    pose_model = None
    if use_pose:
        from ultralytics import YOLO

        console.print("[dim]Loading YOLOv8-pose model...[/dim]")
        pose_model = YOLO("yolov8n-pose.pt")

    # Group frames by video_id to minimize video reopens
    by_video: dict[str, list[ContactFrame]] = {}
    for cf in contact_frames:
        by_video.setdefault(cf.video_id, []).append(cf)

    total_saved = 0
    all_pose_data: list[PoseData] = []
    all_temporal_data: list[TemporalPoseData] = []

    with Progress(console=console) as progress:
        label = (
            "Extracting temporal poses..." if use_temporal
            else "Extracting frames..."
        )
        task = progress.add_task(label, total=len(contact_frames))

        for video_id, cfs in by_video.items():
            video_path = get_video_path(video_id)
            if video_path is None:
                console.print(f"[yellow]Video {video_id[:8]} not found, skipping {len(cfs)} frames[/yellow]")
                progress.advance(task, len(cfs))
                continue

            video = Video(video_path)

            # Get video dimensions from first frame
            probe_frame = video.read_frame(0)
            if probe_frame is None:
                console.print(f"[yellow]Video {video_id[:8]} unreadable, skipping[/yellow]")
                progress.advance(task, len(cfs))
                del video
                continue
            vid_h, vid_w = probe_frame.shape[:2]

            # Sort by absolute frame for sequential-ish access
            cfs.sort(key=lambda cf: cf.abs_frame)

            for cf in cfs:
                if use_temporal and pose_model is not None:
                    # Temporal mode: extract across ±window frames
                    td = _extract_temporal_pose_data(video, cf, pose_model, window, vid_h, vid_w)
                    all_temporal_data.append(td)
                else:
                    # Single-frame mode
                    raw_frame = video.read_frame(cf.abs_frame)
                    if raw_frame is None:
                        progress.advance(task)
                        continue

                    if pose_model is not None:
                        pd = _extract_pose_data(raw_frame, cf, pose_model)
                        all_pose_data.append(pd)

                    if save_images:
                        annotated = _annotate_frame(raw_frame, cf)
                        if pose_model is not None:
                            annotated = _overlay_pose(annotated, cf, pose_model)

                        fname = f"{cf.rally_id[:8]}_f{cf.frame:04d}_{cf.action}.jpg"
                        out_path = OUTPUT_DIR / cf.category / fname
                        cv2.imwrite(str(out_path), annotated)
                        total_saved += 1

                progress.advance(task)

            del video

    if save_images and total_saved > 0:
        console.print(f"\n[bold green]Saved {total_saved} frames to {OUTPUT_DIR}/[/bold green]")
        console.print(f"  TP: {OUTPUT_DIR / 'TP'}")
        console.print(f"  FP: {OUTPUT_DIR / 'FP'}")
        console.print(f"  FN: {OUTPUT_DIR / 'FN'}")

    if all_temporal_data:
        _print_temporal_stats(all_temporal_data)
    elif all_pose_data:
        _print_pose_stats(all_pose_data)


if __name__ == "__main__":
    main()
