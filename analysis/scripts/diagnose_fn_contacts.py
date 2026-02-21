"""Diagnose false-negative (missed) contacts in contact detection.

For each GT contact that went undetected, categorizes the root cause:
  - no_ball_data:      No ball position within ±10 frames of GT
  - no_candidate:      Ball data exists but no candidate was generated
  - rejected_candidate: A candidate existed but failed validation

Usage:
    cd analysis
    uv run python scripts/diagnose_fn_contacts.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# Add analysis root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np  # noqa: E402

from rallycut.tracking.action_classifier import classify_rally_actions  # noqa: E402
from rallycut.tracking.ball_tracker import BallPosition as BallPos  # noqa: E402
from rallycut.tracking.contact_detector import (
    _CONFIDENCE_THRESHOLD,
    ContactDetectionConfig,
    _compute_direction_change,
    _compute_velocities,
    _filter_noise_spikes,
    _find_inflection_candidates,
    _find_net_crossing_candidates,
    _find_parabolic_breakpoints,
    _find_velocity_reversal_candidates,
    _merge_candidates,
    detect_contacts,
    estimate_net_position,
)
from rallycut.tracking.player_tracker import PlayerPosition as PlayerPos
from scripts.eval_action_detection import (
    GtLabel,
    load_rallies_with_action_gt,
    match_contacts,
)

BALL_WINDOW = 10          # ±frames to look for ball data near a GT contact
TOLERANCE_MS = 167        # Default eval tolerance


def build_ball_index(ball_positions: list[BallPos]) -> dict[int, BallPos]:
    """Build frame -> BallPos dict for confident detections only."""
    return {
        bp.frame_number: bp
        for bp in ball_positions
        if bp.confidence >= _CONFIDENCE_THRESHOLD
    }


def nearest_ball(frame: int, ball_by_frame: dict[int, BallPos], window: int = 10) -> tuple[BallPos | None, int]:
    """Find the closest ball position within ±window frames.
    Returns (ball_position, offset) or (None, ∞)."""
    best_bp = None
    best_off = window + 1
    for offset in range(0, window + 1):
        for sign in ([0] if offset == 0 else [-offset, offset]):
            f = frame + sign
            if f in ball_by_frame:
                if abs(sign) < best_off:
                    best_off = abs(sign)
                    best_bp = ball_by_frame[f]
    return best_bp, best_off


def compute_local_velocity(
    frame: int,
    ball_by_frame: dict[int, BallPos],
    window: int = 5,
) -> float | None:
    """Compute approximate ball speed (norm units/frame) near GT frame."""
    frames_sorted = sorted(ball_by_frame.keys())
    # Find frames within window
    local = [f for f in frames_sorted if abs(f - frame) <= window]
    if len(local) < 2:
        return None
    speeds = []
    for i in range(1, len(local)):
        f_prev, f_curr = local[i - 1], local[i]
        gap = f_curr - f_prev
        if gap <= 0 or gap > 5:
            continue
        bp_p = ball_by_frame[f_prev]
        bp_c = ball_by_frame[f_curr]
        dx = (bp_c.x - bp_p.x) / gap
        dy = (bp_c.y - bp_p.y) / gap
        speeds.append(math.sqrt(dx * dx + dy * dy))
    return float(np.mean(speeds)) if speeds else None


def get_candidates_for_rally(
    ball_positions: list[BallPos],
    cfg: ContactDetectionConfig,
    net_y: float | None = None,
) -> list[int]:
    """Reproduce the candidate generation step from detect_contacts (no validation)."""
    from scipy.signal import find_peaks as sp_find_peaks

    if cfg.enable_noise_filter:
        ball_positions = _filter_noise_spikes(ball_positions, cfg.noise_spike_max_jump)

    velocities = _compute_velocities(ball_positions)
    if not velocities:
        return []

    frames = sorted(velocities.keys())
    if len(frames) < 3:
        return []

    speeds = [velocities[f][0] for f in frames]

    # Smooth
    window = cfg.smoothing_window
    half_w = window // 2
    smoothed = []
    for i in range(len(speeds)):
        start = max(0, i - half_w)
        end = min(len(speeds), i + half_w + 1)
        smoothed.append(sum(speeds[start:end]) / (end - start))

    peak_indices, _ = sp_find_peaks(
        smoothed,
        height=cfg.min_peak_velocity,
        prominence=cfg.min_peak_prominence,
        distance=cfg.min_peak_distance_frames,
    )
    velocity_peak_frames = [frames[idx] for idx in peak_indices]

    ball_by_frame: dict[int, BallPos] = {
        bp.frame_number: bp
        for bp in ball_positions
        if bp.confidence >= _CONFIDENCE_THRESHOLD
    }
    confident_frames = sorted(ball_by_frame.keys())

    inflection_frames: list[int] = []
    if cfg.enable_inflection_detection:
        inflection_frames = _find_inflection_candidates(
            ball_by_frame,
            confident_frames,
            min_angle_deg=cfg.min_inflection_angle_deg,
            check_frames=cfg.inflection_check_frames,
            min_distance_frames=cfg.min_peak_distance_frames,
        )

    reversal_frames = _find_velocity_reversal_candidates(
        velocities, frames, cfg.min_peak_distance_frames
    )

    # Parabolic breakpoints
    parabolic_frames: list[int] = []
    if cfg.enable_parabolic_detection:
        parabolic_frames, _ = _find_parabolic_breakpoints(
            ball_by_frame,
            confident_frames,
            window_frames=cfg.parabolic_window_frames,
            stride=cfg.parabolic_stride,
            min_residual=cfg.parabolic_min_residual,
            min_prominence=cfg.parabolic_min_prominence,
            min_distance_frames=cfg.min_peak_distance_frames,
        )

    # Net-crossing candidates
    estimated_net_y = net_y if net_y is not None else estimate_net_position(ball_positions)
    net_crossing_frames = _find_net_crossing_candidates(
        ball_by_frame, confident_frames, estimated_net_y, cfg.min_peak_distance_frames
    )

    inflection_and_reversal = _merge_candidates(
        inflection_frames, reversal_frames, cfg.min_peak_distance_frames
    )
    traditional = _merge_candidates(
        velocity_peak_frames, inflection_and_reversal, cfg.min_peak_distance_frames
    )
    with_parabolic = _merge_candidates(
        traditional, parabolic_frames, cfg.min_peak_distance_frames
    )
    candidate_frames = _merge_candidates(
        with_parabolic, net_crossing_frames, cfg.min_peak_distance_frames
    )
    return candidate_frames


def nearest_pred_contact(gt_frame: int, pred_actions: list[dict]) -> tuple[int | None, int]:
    """Find nearest predicted contact frame. Returns (frame, distance)."""
    best_frame = None
    best_dist = 10**9
    for p in pred_actions:
        pf = p.get("frame", 0)
        d = abs(gt_frame - pf)
        if d < best_dist:
            best_dist = d
            best_frame = pf
    return best_frame, best_dist


def nearest_player(gt_frame: int, player_positions: list[PlayerPos], ball_x: float, ball_y: float) -> tuple[int, float]:
    """Find nearest player (track_id, distance) to the ball position."""
    best_id = -1
    best_dist = float("inf")
    for p in player_positions:
        if abs(p.frame_number - gt_frame) > 5:
            continue
        px = p.x
        py = p.y - p.height * 0.25
        d = math.sqrt((ball_x - px) ** 2 + (ball_y - py) ** 2)
        if d < best_dist:
            best_dist = d
            best_id = p.track_id
    return best_id, best_dist


def categorize_miss(
    gt: GtLabel,
    ball_by_frame: dict[int, BallPos],
    candidate_frames: list[int],
    cfg: ContactDetectionConfig,
) -> tuple[str, dict]:
    """Return (category, details_dict)."""
    frame = gt.frame

    # 1. Check ball data
    near_bp, ball_offset = nearest_ball(frame, ball_by_frame, BALL_WINDOW)
    if near_bp is None:
        return "no_ball_data", {"ball_offset": None, "velocity": None, "dir_change": None}

    # Compute velocity and direction change at or near the GT frame
    velocity = compute_local_velocity(frame, ball_by_frame, window=5)
    dir_change = _compute_direction_change(ball_by_frame, frame, check_frames=5)

    # 2. Check if a candidate was generated near this frame
    nearby_candidates = [
        c for c in candidate_frames
        if abs(c - frame) <= cfg.min_peak_distance_frames
    ]
    if not nearby_candidates:
        return "no_candidate", {
            "ball_offset": ball_offset,
            "velocity": velocity,
            "dir_change": dir_change,
        }

    # 3. Candidate existed — it was rejected by validation
    best_cand = min(nearby_candidates, key=lambda c: abs(c - frame))
    return "rejected_candidate", {
        "ball_offset": ball_offset,
        "candidate_frame": best_cand,
        "candidate_offset": abs(best_cand - frame),
        "velocity": velocity,
        "dir_change": dir_change,
    }


def main() -> None:
    print("Loading rallies with action GT...")
    rallies = load_rallies_with_action_gt()
    if not rallies:
        print("ERROR: No rallies found with action ground truth.")
        return

    print(f"Found {len(rallies)} rallies.\n")

    cfg = ContactDetectionConfig()
    tolerance_ms = TOLERANCE_MS

    all_misses: list[dict] = []

    for rally in rallies:
        fps = rally.fps or 30.0
        tolerance_frames = max(1, round(fps * tolerance_ms / 1000))

        # Build ball positions
        if not rally.ball_positions_json:
            # All GT contacts are missed due to no ball data
            for gt in rally.gt_labels:
                all_misses.append({
                    "rally_id": rally.rally_id[:8],
                    "gt_frame": gt.frame,
                    "gt_action": gt.action,
                    "category": "no_ball_data",
                    "velocity": None,
                    "dir_change": None,
                    "ball_offset": None,
                    "nearest_pred_frame": None,
                    "nearest_pred_dist": None,
                    "nearest_player_dist": None,
                    "candidate_frame": None,
                    "detail": "no ball_positions_json",
                })
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

        # Get predicted contacts (re-detect from scratch)
        contacts = detect_contacts(
            ball_positions=ball_positions,
            player_positions=player_positions,
            config=cfg,
            net_y=rally.court_split_y,
            frame_count=rally.frame_count or None,
        )
        rally_actions = classify_rally_actions(contacts, rally.rally_id)
        pred_action_dicts = [a.to_dict() for a in rally_actions.actions]

        matches, unmatched = match_contacts(
            rally.gt_labels,
            pred_action_dicts,
            tolerance=tolerance_frames,
        )

        # Get candidates (no validation) for this rally
        candidate_frames = get_candidates_for_rally(ball_positions, cfg, net_y=rally.court_split_y)

        # Build ball index (confident positions)
        if cfg.enable_noise_filter:
            filtered_bp = _filter_noise_spikes(ball_positions, cfg.noise_spike_max_jump)
        else:
            filtered_bp = ball_positions
        ball_by_frame = build_ball_index(filtered_bp)

        # Process misses
        for m in matches:
            if m.pred_frame is not None:
                continue  # Matched — not a miss

            gt = GtLabel(
                frame=m.gt_frame,
                action=m.gt_action,
                player_track_id=-1,
            )

            category, details = categorize_miss(gt, ball_by_frame, candidate_frames, cfg)

            # Get nearest pred contact (for miss distance)
            near_pred_f, near_pred_d = nearest_pred_contact(m.gt_frame, pred_action_dicts)

            # Get nearest player distance at GT frame (using ball position if known)
            near_bp_for_player, _ = nearest_ball(m.gt_frame, ball_by_frame, BALL_WINDOW)
            if near_bp_for_player is not None and player_positions:
                _, player_dist = nearest_player(
                    m.gt_frame, player_positions,
                    near_bp_for_player.x, near_bp_for_player.y,
                )
            else:
                player_dist = None

            all_misses.append({
                "rally_id": rally.rally_id[:8],
                "gt_frame": m.gt_frame,
                "gt_action": m.gt_action,
                "category": category,
                "velocity": details.get("velocity"),
                "dir_change": details.get("dir_change"),
                "ball_offset": details.get("ball_offset"),
                "nearest_pred_frame": near_pred_f,
                "nearest_pred_dist": near_pred_d if near_pred_f is not None else None,
                "nearest_player_dist": round(player_dist, 3) if player_dist is not None else None,
                "candidate_frame": details.get("candidate_frame"),
                "candidate_offset": details.get("candidate_offset"),
            })

    # ── Print summary table ──────────────────────────────────────────────────
    print(f"Total missed GT contacts: {len(all_misses)}\n")

    # Category counts
    from collections import Counter
    cat_counts = Counter(m["category"] for m in all_misses)
    print("Category breakdown:")
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {cnt}")
    print()

    # Full table
    header = (
        f"{'#':<3} {'Rally':^10} {'Frame':>6} {'Action':^12} "
        f"{'Category':^20} {'Vel':>7} {'DirChg':>8} "
        f"{'BallOff':>8} {'NearPred':>9} {'NearPlayer':>11}"
    )
    print(header)
    print("-" * len(header))

    for i, m in enumerate(all_misses, 1):
        vel_str = f"{m['velocity']:.4f}" if m["velocity"] is not None else "  N/A  "
        dir_str = f"{m['dir_change']:.1f}°" if m["dir_change"] is not None else "  N/A  "
        ball_off_str = str(m["ball_offset"]) if m["ball_offset"] is not None else "N/A"
        pred_str = (
            f"+{m['nearest_pred_dist']}f"
            if m["nearest_pred_dist"] is not None
            else "none"
        )
        player_str = (
            f"{m['nearest_player_dist']:.3f}"
            if m["nearest_player_dist"] is not None
            else "N/A"
        )
        cand_info = ""
        if m.get("candidate_frame") is not None:
            cand_info = f" [cand@{m['candidate_offset']}f away]"

        print(
            f"{i:<3} {m['rally_id']:^10} {m['gt_frame']:>6} {m['gt_action']:^12} "
            f"{m['category'] + cand_info:^20} {vel_str:>7} {dir_str:>8} "
            f"{ball_off_str:>8} {pred_str:>9} {player_str:>11}"
        )

    # ── Additional groupings ─────────────────────────────────────────────────
    print()
    print("Misses by action type:")
    by_action = Counter(m["gt_action"] for m in all_misses)
    for action, cnt in sorted(by_action.items(), key=lambda x: -x[1]):
        print(f"  {action}: {cnt}")

    print()
    print("Misses by rally:")
    by_rally = Counter(m["rally_id"] for m in all_misses)
    for rally_id, cnt in sorted(by_rally.items(), key=lambda x: -x[1]):
        print(f"  {rally_id}: {cnt}")

    # ── Rejected candidate detail ────────────────────────────────────────────
    rejected = [m for m in all_misses if m["category"] == "rejected_candidate"]
    if rejected:
        print()
        print(f"Rejected candidate details ({len(rejected)} misses):")
        print(
            f"  {'Rally':^10} {'GTFrame':>8} {'Action':^12} "
            f"{'CandOff':>8} {'Vel':>8} {'DirChg':>9} {'NearPlayer':>11}"
        )
        print("  " + "-" * 65)
        for m in rejected:
            vel_str = f"{m['velocity']:.4f}" if m["velocity"] is not None else "N/A"
            dir_str = f"{m['dir_change']:.1f}°" if m["dir_change"] is not None else "N/A"
            player_str = f"{m['nearest_player_dist']:.3f}" if m["nearest_player_dist"] is not None else "N/A"
            coff = m.get("candidate_offset")
            coff_str = f"{coff}f" if coff is not None else "?"
            print(
                f"  {m['rally_id']:^10} {m['gt_frame']:>8} {m['gt_action']:^12} "
                f"{coff_str:>8} {vel_str:>8} {dir_str:>9} {player_str:>11}"
            )


if __name__ == "__main__":
    main()
