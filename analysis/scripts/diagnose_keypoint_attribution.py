"""Diagnose whether hand/wrist keypoint distance beats bbox centroid for attribution.

For each matched contact, compares:
  1. Baseline: nearest player by bbox upper-quarter centroid to ball
  2. Wrist: nearest player by min(left_wrist, right_wrist) distance to ball
  3. Upper-body: nearest player by min(shoulders, elbows, wrists) distance to ball

All evaluated under independent per-rally oracle permutations (Hungarian assignment).

Usage:
    cd analysis
    uv run python scripts/diagnose_keypoint_attribution.py
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

from rich.console import Console
from rich.table import Table

from scripts.eval_action_detection import (
    _build_player_positions,
    load_rallies_with_action_gt,
    match_contacts,
)
from scripts.production_eval import MatchResult, _rally_permutation_oracle
from rallycut.tracking.action_classifier import classify_rally_actions
from rallycut.tracking.ball_tracker import BallPosition
from rallycut.tracking.contact_detector import ContactDetectionConfig, detect_contacts
from rallycut.tracking.player_tracker import PlayerPosition

console = Console()

# COCO keypoint indices (upper body)
KPT_LEFT_SHOULDER = 5
KPT_RIGHT_SHOULDER = 6
KPT_LEFT_ELBOW = 7
KPT_RIGHT_ELBOW = 8
KPT_LEFT_WRIST = 9
KPT_RIGHT_WRIST = 10

UPPER_BODY_KPTS = [
    KPT_LEFT_SHOULDER, KPT_RIGHT_SHOULDER,
    KPT_LEFT_ELBOW, KPT_RIGHT_ELBOW,
    KPT_LEFT_WRIST, KPT_RIGHT_WRIST,
]
WRIST_KPTS = [KPT_LEFT_WRIST, KPT_RIGHT_WRIST]

MIN_KPT_CONF = 0.3  # minimum keypoint confidence to use
HIGH_CONF_THRESHOLD = 0.5  # threshold for high-confidence split
ACTION_ORDER = ["serve", "receive", "set", "attack", "dig", "block"]
SEARCH_FRAMES = 5


def _build_frame_index(
    player_positions: list[PlayerPosition],
) -> dict[int, list[PlayerPosition]]:
    idx: dict[int, list[PlayerPosition]] = defaultdict(list)
    for p in player_positions:
        idx[p.frame_number].append(p)
    return idx


def _bbox_dist(px: float, py: float, ph: float, ball_x: float, ball_y: float) -> float:
    """Euclidean distance from upper-quarter bbox point to ball."""
    uq_y = py - ph * 0.25
    return math.sqrt((ball_x - px) ** 2 + (ball_y - uq_y) ** 2)


def _keypoint_dist(
    player: PlayerPosition,
    ball_x: float,
    ball_y: float,
    kpt_indices: list[int],
) -> tuple[float, float]:
    """Min Euclidean distance from specified keypoints to ball.

    Returns (distance, mean_confidence). Distance is inf if no keypoint
    has sufficient confidence.
    """
    if player.keypoints is None:
        return float("inf"), 0.0

    best_dist = float("inf")
    conf_sum = 0.0
    conf_count = 0

    for idx in kpt_indices:
        if idx >= len(player.keypoints):
            continue
        kx, ky, kc = player.keypoints[idx]
        conf_sum += kc
        conf_count += 1
        if kc < MIN_KPT_CONF:
            continue
        dist = math.sqrt((ball_x - kx) ** 2 + (ball_y - ky) ** 2)
        if dist < best_dist:
            best_dist = dist

    mean_conf = conf_sum / conf_count if conf_count > 0 else 0.0
    return best_dist, mean_conf


def _nearest_by_method(
    frame: int,
    ball_x: float,
    ball_y: float,
    frame_index: dict[int, list[PlayerPosition]],
    method: str,
) -> tuple[int, bool, float]:
    """Find nearest player using specified method.

    Returns (track_id, had_pose, mean_keypoint_conf).
    had_pose is True if the winning player had usable keypoints.
    """
    # Collect per-track best distance across search window
    track_best: dict[int, float] = {}
    track_pose: dict[int, bool] = {}
    track_conf: dict[int, float] = {}

    for offset in range(SEARCH_FRAMES + 1):
        for sign in ([0] if offset == 0 else [-1, 1]):
            f = frame + sign * offset
            for p in frame_index.get(f, []):
                if method == "bbox":
                    dist = _bbox_dist(p.x, p.y, p.height, ball_x, ball_y)
                    had_pose = False
                    mean_conf = 0.0
                elif method == "wrist":
                    dist, mean_conf = _keypoint_dist(p, ball_x, ball_y, WRIST_KPTS)
                    had_pose = dist < float("inf")
                    # Fall back to bbox if no wrist keypoints
                    if not had_pose:
                        dist = _bbox_dist(p.x, p.y, p.height, ball_x, ball_y)
                elif method == "upper_body":
                    dist, mean_conf = _keypoint_dist(p, ball_x, ball_y, UPPER_BODY_KPTS)
                    had_pose = dist < float("inf")
                    if not had_pose:
                        dist = _bbox_dist(p.x, p.y, p.height, ball_x, ball_y)
                else:
                    raise ValueError(f"Unknown method: {method}")

                tid = p.track_id
                if tid not in track_best or dist < track_best[tid]:
                    track_best[tid] = dist
                    track_pose[tid] = had_pose
                    track_conf[tid] = mean_conf

    if not track_best:
        return -1, False, 0.0

    best_tid = min(track_best, key=track_best.get)  # type: ignore[arg-type]
    return best_tid, track_pose[best_tid], track_conf[best_tid]


@dataclass
class ContactResult:
    gt_tid: int
    gt_action: str
    gt_frame: int
    pred_frame: int
    bbox_tid: int
    wrist_tid: int
    upper_body_tid: int
    ball_x: float
    ball_y: float
    had_pose: bool  # whether any player had pose at this contact
    wrist_conf: float
    upper_conf: float


@dataclass
class Accumulators:
    correct: int = 0
    total: int = 0
    # Error-set (relative to baseline)
    rescues: int = 0
    regressions: int = 0
    # Per-action
    per_action: dict[str, dict[str, int]] = field(
        default_factory=lambda: defaultdict(
            lambda: {"correct": 0, "total": 0}
        )
    )
    # High-confidence subset
    hc_correct: int = 0
    hc_total: int = 0


def main() -> None:
    rallies = load_rallies_with_action_gt()
    if not rallies:
        console.print("[red]No rallies found with action ground truth.[/red]")
        return

    console.print(f"[bold]Keypoint Attribution Diagnostic — {len(rallies)} rallies[/bold]\n")

    methods = ["bbox", "wrist", "upper_body"]
    acc: dict[str, Accumulators] = {m: Accumulators() for m in methods}

    # Coverage stats
    contacts_with_pose = 0
    contacts_without_pose = 0
    total_evaluable = 0

    # Baseline error tracking for rescue analysis
    baseline_errors_total = 0

    skipped = 0
    processed = 0

    for idx, rally in enumerate(rallies, 1):
        if not rally.ball_positions_json or not rally.positions_json:
            skipped += 1
            continue

        ball_positions = [
            BallPosition(
                frame_number=bp["frameNumber"],
                x=bp["x"],
                y=bp["y"],
                confidence=bp.get("confidence", 1.0),
            )
            for bp in rally.ball_positions_json
            if bp.get("x", 0) > 0 or bp.get("y", 0) > 0
        ]

        # inject_pose=True to get keypoints from pose cache
        player_positions = _build_player_positions(
            rally.positions_json, rally.rally_id, inject_pose=True
        )
        frame_index = _build_frame_index(player_positions)

        contacts = detect_contacts(
            ball_positions=ball_positions,
            player_positions=player_positions,
            config=ContactDetectionConfig(),
            net_y=rally.court_split_y,
            frame_count=rally.frame_count or None,
        )

        rally_actions = classify_rally_actions(contacts, rally.rally_id)
        pred_actions = [a.to_dict() for a in rally_actions.actions]
        real_pred = [a for a in pred_actions if not a.get("isSynthetic")]

        tolerance = max(1, round(rally.fps * 150 / 1000))
        matches, _ = match_contacts(rally.gt_labels, real_pred, tolerance=tolerance)

        pred_by_frame: dict[int, dict] = {}
        for a in real_pred:
            f = a.get("frame")
            if f is not None:
                pred_by_frame[f] = a

        # Build MatchResult lists per method
        method_matches: dict[str, list[MatchResult]] = {m: [] for m in methods}
        contact_results: list[ContactResult] = []

        for m in matches:
            if m.pred_frame is None:
                continue

            gt_label = next(
                (g for g in rally.gt_labels if g.frame == m.gt_frame), None
            )
            if gt_label is None or gt_label.player_track_id < 0:
                continue

            pred = pred_by_frame.get(m.pred_frame)
            if pred is None:
                continue

            gt_tid = gt_label.player_track_id
            ball_x = pred.get("ballX", 0.0)
            ball_y = pred.get("ballY", 0.0)
            if ball_x == 0.0 and ball_y == 0.0:
                continue

            # Compute attribution for each method
            bbox_tid, _, _ = _nearest_by_method(
                m.pred_frame, ball_x, ball_y, frame_index, "bbox"
            )
            wrist_tid, wrist_pose, wrist_conf = _nearest_by_method(
                m.pred_frame, ball_x, ball_y, frame_index, "wrist"
            )
            ub_tid, ub_pose, ub_conf = _nearest_by_method(
                m.pred_frame, ball_x, ball_y, frame_index, "upper_body"
            )

            if bbox_tid < 0:
                continue

            had_pose = wrist_pose or ub_pose
            if had_pose:
                contacts_with_pose += 1
            else:
                contacts_without_pose += 1
            total_evaluable += 1

            tids = {"bbox": bbox_tid, "wrist": wrist_tid, "upper_body": ub_tid}
            for method_name in methods:
                mr = MatchResult(
                    gt_frame=m.gt_frame,
                    gt_action=m.gt_action,
                    pred_frame=m.pred_frame,
                    pred_action=m.pred_action,
                    player_evaluable=True,
                )
                mr._gt_tid = gt_tid  # type: ignore[attr-defined]
                mr._pred_tid = tids[method_name]  # type: ignore[attr-defined]
                method_matches[method_name].append(mr)

            contact_results.append(ContactResult(
                gt_tid=gt_tid,
                gt_action=m.gt_action,
                gt_frame=m.gt_frame,
                pred_frame=m.pred_frame,
                bbox_tid=bbox_tid,
                wrist_tid=wrist_tid,
                upper_body_tid=ub_tid,
                ball_x=ball_x,
                ball_y=ball_y,
                had_pose=had_pose,
                wrist_conf=wrist_conf,
                upper_conf=ub_conf,
            ))

        # Run oracle independently for each method
        perms: dict[str, dict[int, int]] = {}
        for method_name in methods:
            c, n, perm = _rally_permutation_oracle(method_matches[method_name])
            acc[method_name].correct += c
            acc[method_name].total += n
            perms[method_name] = perm

        # Per-contact analysis under each method's own permutation
        for cr in contact_results:
            if cr.gt_action == "block":
                continue

            b_ok = perms["bbox"].get(cr.gt_tid) == cr.bbox_tid
            w_ok = perms["wrist"].get(cr.gt_tid) == cr.wrist_tid
            u_ok = perms["upper_body"].get(cr.gt_tid) == cr.upper_body_tid

            action = cr.gt_action

            # Bbox (baseline)
            acc["bbox"].per_action[action]["total"] += 1
            if b_ok:
                acc["bbox"].per_action[action]["correct"] += 1

            # Wrist
            acc["wrist"].per_action[action]["total"] += 1
            if w_ok:
                acc["wrist"].per_action[action]["correct"] += 1

            # Upper body
            acc["upper_body"].per_action[action]["total"] += 1
            if u_ok:
                acc["upper_body"].per_action[action]["correct"] += 1

            # Error-set rescue analysis (relative to bbox baseline)
            if not b_ok:
                baseline_errors_total += 1
                if w_ok:
                    acc["wrist"].rescues += 1
                if u_ok:
                    acc["upper_body"].rescues += 1
            else:
                if not w_ok:
                    acc["wrist"].regressions += 1
                if not u_ok:
                    acc["upper_body"].regressions += 1

            # High-confidence subset (pose conf >= threshold)
            if cr.wrist_conf >= HIGH_CONF_THRESHOLD:
                acc["wrist"].hc_total += 1
                if w_ok:
                    acc["wrist"].hc_correct += 1
            if cr.upper_conf >= HIGH_CONF_THRESHOLD:
                acc["upper_body"].hc_total += 1
                if u_ok:
                    acc["upper_body"].hc_correct += 1
            # Baseline at high-conf contacts (use upper_conf as proxy for "had good pose")
            if cr.upper_conf >= HIGH_CONF_THRESHOLD:
                acc["bbox"].hc_total += 1
                if b_ok:
                    acc["bbox"].hc_correct += 1

        processed += 1
        if processed % 50 == 0:
            console.print(f"  [{processed}/{len(rallies)}] processed...")

    # === Results ===
    console.print(
        f"\n[bold]Results[/bold] ({processed} rallies, {skipped} skipped)\n"
    )

    # Table 1: Coverage
    console.print("[bold]Pose Coverage[/bold]")
    console.print(f"  Contacts with pose keypoints: {contacts_with_pose}/{total_evaluable} "
                  f"({contacts_with_pose / max(1, total_evaluable) * 100:.1f}%)")
    console.print(f"  Contacts without pose:        {contacts_without_pose}/{total_evaluable}\n")

    # Table 2: Overall oracle accuracy
    console.print("[bold]Overall Oracle Accuracy[/bold]")
    t1 = Table()
    t1.add_column("Method", style="bold")
    t1.add_column("Correct", justify="right")
    t1.add_column("Total", justify="right")
    t1.add_column("Accuracy", justify="right")
    t1.add_column("Delta vs bbox", justify="right")

    b_pct = acc["bbox"].correct / max(1, acc["bbox"].total) * 100
    for method_name, label in [
        ("bbox", "Baseline (bbox centroid)"),
        ("wrist", "Wrist keypoints"),
        ("upper_body", "Upper-body keypoints"),
    ]:
        a = acc[method_name]
        pct = a.correct / max(1, a.total) * 100
        delta = pct - b_pct
        t1.add_row(
            label,
            str(a.correct), str(a.total),
            f"{pct:.1f}%",
            "—" if method_name == "bbox" else f"{delta:+.1f}pp",
        )
    console.print(t1)

    # Table 3: Error-set rescue
    console.print(f"\n[bold]Error-Set Analysis[/bold] (baseline errors: {baseline_errors_total})")
    t2 = Table()
    t2.add_column("Method", style="bold")
    t2.add_column("Rescues", justify="right")
    t2.add_column("Rescue %", justify="right")
    t2.add_column("Regressions", justify="right")
    t2.add_column("Net", justify="right")

    for method_name, label in [
        ("wrist", "Wrist keypoints"),
        ("upper_body", "Upper-body keypoints"),
    ]:
        a = acc[method_name]
        rescue_pct = a.rescues / max(1, baseline_errors_total) * 100
        net = a.rescues - a.regressions
        t2.add_row(
            label,
            str(a.rescues), f"{rescue_pct:.1f}%",
            str(a.regressions),
            f"{'+' if net >= 0 else ''}{net}",
        )
    console.print(t2)

    # Table 4: Per-action breakdown
    console.print(f"\n[bold]Per-Action Breakdown[/bold]")
    t3 = Table()
    t3.add_column("Action", style="bold")
    t3.add_column("Bbox", justify="right")
    t3.add_column("Wrist", justify="right")
    t3.add_column("W Δ", justify="right")
    t3.add_column("Upper", justify="right")
    t3.add_column("U Δ", justify="right")
    t3.add_column("N", justify="right")

    for action in ACTION_ORDER:
        pa_b = acc["bbox"].per_action.get(action)
        if not pa_b or pa_b["total"] == 0:
            continue
        n = pa_b["total"]
        b_a = pa_b["correct"] / n * 100

        pa_w = acc["wrist"].per_action.get(action, {"correct": 0, "total": 0})
        w_a = pa_w["correct"] / max(1, pa_w["total"]) * 100

        pa_u = acc["upper_body"].per_action.get(action, {"correct": 0, "total": 0})
        u_a = pa_u["correct"] / max(1, pa_u["total"]) * 100

        t3.add_row(
            action,
            f"{b_a:.1f}%",
            f"{w_a:.1f}%", f"{w_a - b_a:+.1f}pp",
            f"{u_a:.1f}%", f"{u_a - b_a:+.1f}pp",
            str(n),
        )
    console.print(t3)

    # Table 5: High-confidence pose subset
    console.print(f"\n[bold]High-Confidence Pose Subset (conf ≥ {HIGH_CONF_THRESHOLD})[/bold]")
    t4 = Table()
    t4.add_column("Method", style="bold")
    t4.add_column("Correct", justify="right")
    t4.add_column("Total", justify="right")
    t4.add_column("Accuracy", justify="right")
    t4.add_column("Delta vs bbox", justify="right")

    hc_b_pct = acc["bbox"].hc_correct / max(1, acc["bbox"].hc_total) * 100
    for method_name, label in [
        ("bbox", "Baseline (bbox centroid)"),
        ("wrist", "Wrist keypoints"),
        ("upper_body", "Upper-body keypoints"),
    ]:
        a = acc[method_name]
        pct = a.hc_correct / max(1, a.hc_total) * 100
        delta = pct - hc_b_pct
        t4.add_row(
            label,
            str(a.hc_correct), str(a.hc_total),
            f"{pct:.1f}%",
            "—" if method_name == "bbox" else f"{delta:+.1f}pp",
        )
    console.print(t4)

    # Verdict
    w_delta = acc["wrist"].correct / max(1, acc["wrist"].total) * 100 - b_pct
    u_delta = acc["upper_body"].correct / max(1, acc["upper_body"].total) * 100 - b_pct
    best_delta = max(w_delta, u_delta)
    best_name = "Wrist" if w_delta >= u_delta else "Upper-body"

    console.print(f"\n[bold]Verdict[/bold]")
    if best_delta > 1.0:
        console.print(
            f"  [green]{best_name} keypoints are +{best_delta:.1f}pp over bbox baseline. "
            f"Worth pursuing in production attribution.[/green]"
        )
    elif best_delta > -1.0:
        console.print(
            f"  [yellow]{best_name} keypoints are {best_delta:+.1f}pp vs bbox baseline. "
            f"Marginal — probably not worth the complexity.[/yellow]"
        )
    else:
        console.print(
            f"  [red]{best_name} keypoints are {best_delta:+.1f}pp vs bbox baseline. "
            f"Worse. Do not pursue.[/red]"
        )


if __name__ == "__main__":
    main()
