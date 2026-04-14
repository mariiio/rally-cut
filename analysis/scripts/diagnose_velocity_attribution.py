"""Diagnose whether closing velocity is a better attribution signal than nearest-player.

For each matched contact, compares:
  1. Baseline: nearest player to ball at contact frame (current production method)
  2. Velocity: player with highest closing velocity toward ball in 0.5s before contact

Both evaluated under independent per-rally oracle permutations (Hungarian assignment).

Usage:
    cd analysis
    uv run python scripts/diagnose_velocity_attribution.py
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from rich.console import Console
from rich.table import Table

from scripts.eval_action_detection import (
    GtLabel,
    MatchResult,
    _build_player_positions,
    load_rallies_with_action_gt,
    match_contacts,
)
from scripts.production_eval import _rally_permutation_oracle
from rallycut.tracking.action_classifier import classify_rally_actions
from rallycut.tracking.ball_tracker import BallPosition
from rallycut.tracking.contact_detector import ContactDetectionConfig, detect_contacts
from rallycut.tracking.player_tracker import PlayerPosition

console = Console()

VELOCITY_WINDOW = 15  # frames (~0.5s at 30fps)
ACTION_ORDER = ["serve", "receive", "set", "attack", "dig", "block"]


def _build_frame_index(
    player_positions: list[PlayerPosition],
) -> dict[int, list[PlayerPosition]]:
    """Index player positions by frame number for O(1) per-frame lookup."""
    idx: dict[int, list[PlayerPosition]] = defaultdict(list)
    for p in player_positions:
        idx[p.frame_number].append(p)
    return idx


def _player_dist(px: float, py: float, ph: float, ball_x: float, ball_y: float) -> float:
    """Image-space Euclidean distance from upper-quarter point to ball."""
    uq_y = py - ph * 0.25
    return math.sqrt((ball_x - px) ** 2 + (ball_y - uq_y) ** 2)


def compute_velocity_attribution(
    contact_frame: int,
    ball_x: float,
    ball_y: float,
    frame_index: dict[int, list[PlayerPosition]],
) -> int:
    """Pick the player with highest closing velocity toward the ball.

    For each track visible in [contact_frame - VELOCITY_WINDOW, contact_frame]:
      - Compute image-space distance from upper-quarter point to ball at each frame
      - closing_velocity = (dist_earliest - dist_latest) / frames_apart
      - Pick highest positive closing velocity
      - Fall back to nearest player at contact frame if none positive
    """
    start_frame = contact_frame - VELOCITY_WINDOW

    # track_id -> list of (frame, distance_to_ball)
    track_measurements: dict[int, list[tuple[int, float]]] = defaultdict(list)

    for f in range(start_frame, contact_frame + 1):
        for p in frame_index.get(f, []):
            dist = _player_dist(p.x, p.y, p.height, ball_x, ball_y)
            track_measurements[p.track_id].append((f, dist))

    best_tid = -1
    best_velocity = 0.0  # must be positive to count

    for tid, measurements in track_measurements.items():
        if len(measurements) < 2:
            continue
        measurements.sort(key=lambda x: x[0])
        earliest_dist = measurements[0][1]
        latest_dist = measurements[-1][1]
        frames_apart = measurements[-1][0] - measurements[0][0]
        if frames_apart <= 0:
            continue
        closing_vel = (earliest_dist - latest_dist) / frames_apart
        if closing_vel > best_velocity:
            best_velocity = closing_vel
            best_tid = tid

    # Fallback: nearest player at contact frame (± search window)
    if best_tid < 0:
        best_dist = float("inf")
        for offset in range(6):
            for sign in ([0] if offset == 0 else [-1, 1]):
                f = contact_frame + sign * offset
                for p in frame_index.get(f, []):
                    dist = _player_dist(p.x, p.y, p.height, ball_x, ball_y)
                    if dist < best_dist:
                        best_dist = dist
                        best_tid = p.track_id
            if best_tid >= 0:
                break

    return best_tid


def _nearest_player_at_frame(
    frame: int,
    ball_x: float,
    ball_y: float,
    frame_index: dict[int, list[PlayerPosition]],
    search_frames: int = 5,
) -> int:
    """Find nearest player to ball at given frame (baseline method)."""
    best_tid = -1
    best_dist = float("inf")
    # Per-track best: take closest frame observation
    candidates: dict[int, float] = {}
    for offset in range(search_frames + 1):
        for sign in ([0] if offset == 0 else [-1, 1]):
            f = frame + sign * offset
            for p in frame_index.get(f, []):
                dist = _player_dist(p.x, p.y, p.height, ball_x, ball_y)
                if p.track_id not in candidates or dist < candidates[p.track_id]:
                    candidates[p.track_id] = dist

    if candidates:
        best_tid = min(candidates, key=candidates.get)  # type: ignore[arg-type]
    return best_tid


@dataclass
class ContactResult:
    gt_tid: int
    gt_action: str
    gt_frame: int
    pred_frame: int
    baseline_tid: int
    velocity_tid: int
    ball_x: float
    ball_y: float


def main() -> None:
    rallies = load_rallies_with_action_gt()
    if not rallies:
        console.print("[red]No rallies found with action ground truth.[/red]")
        return

    console.print(f"[bold]Velocity Attribution Diagnostic — {len(rallies)} rallies[/bold]\n")

    # Accumulators
    baseline_oracle_correct = 0
    baseline_oracle_total = 0
    velocity_oracle_correct = 0
    velocity_oracle_total = 0

    # Error-set rescue
    baseline_errors = 0
    velocity_rescues = 0
    velocity_regressions = 0  # baseline right, velocity wrong

    # Per-action
    per_action: dict[str, dict[str, int]] = defaultdict(
        lambda: {"b_correct": 0, "b_total": 0, "v_correct": 0, "v_total": 0}
    )

    # Disagreement examples
    disagreements: list[dict] = []

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

        player_positions = _build_player_positions(rally.positions_json, rally.rally_id)
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

        # Build pred lookup by frame
        pred_by_frame: dict[int, dict] = {}
        for a in real_pred:
            f = a.get("frame")
            if f is not None:
                pred_by_frame[f] = a

        # Build MatchResult lists with side-channel tids for each method
        baseline_matches: list[MatchResult] = []
        velocity_matches: list[MatchResult] = []
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

            # Baseline: nearest player (recompute from raw positions for consistency)
            baseline_tid = _nearest_player_at_frame(
                m.pred_frame, ball_x, ball_y, frame_index
            )

            # Velocity: closing velocity in pre-contact window
            velocity_tid = compute_velocity_attribution(
                m.pred_frame, ball_x, ball_y, frame_index
            )

            if baseline_tid < 0 or velocity_tid < 0:
                continue

            # MatchResult for baseline oracle
            bm = MatchResult(
                gt_frame=m.gt_frame,
                gt_action=m.gt_action,
                pred_frame=m.pred_frame,
                pred_action=m.pred_action,
                player_evaluable=True,
            )
            bm._gt_tid = gt_tid  # type: ignore[attr-defined]
            bm._pred_tid = baseline_tid  # type: ignore[attr-defined]
            baseline_matches.append(bm)

            # MatchResult for velocity oracle
            vm = MatchResult(
                gt_frame=m.gt_frame,
                gt_action=m.gt_action,
                pred_frame=m.pred_frame,
                pred_action=m.pred_action,
                player_evaluable=True,
            )
            vm._gt_tid = gt_tid  # type: ignore[attr-defined]
            vm._pred_tid = velocity_tid  # type: ignore[attr-defined]
            velocity_matches.append(vm)

            contact_results.append(ContactResult(
                gt_tid=gt_tid,
                gt_action=m.gt_action,
                gt_frame=m.gt_frame,
                pred_frame=m.pred_frame,
                baseline_tid=baseline_tid,
                velocity_tid=velocity_tid,
                ball_x=ball_x,
                ball_y=ball_y,
            ))

        # Run oracle independently for each method
        b_correct, b_total, b_perm = _rally_permutation_oracle(baseline_matches)
        v_correct, v_total, v_perm = _rally_permutation_oracle(velocity_matches)

        baseline_oracle_correct += b_correct
        baseline_oracle_total += b_total
        velocity_oracle_correct += v_correct
        velocity_oracle_total += v_total

        # Per-contact analysis under each method's own permutation
        for cr, bm_i, vm_i in zip(contact_results, baseline_matches, velocity_matches):
            if cr.gt_action == "block":
                continue  # oracle excludes blocks

            b_ok = b_perm.get(cr.gt_tid) == cr.baseline_tid
            v_ok = v_perm.get(cr.gt_tid) == cr.velocity_tid

            action = cr.gt_action
            per_action[action]["b_total"] += 1
            per_action[action]["v_total"] += 1
            if b_ok:
                per_action[action]["b_correct"] += 1
            if v_ok:
                per_action[action]["v_correct"] += 1

            if not b_ok:
                baseline_errors += 1
                if v_ok:
                    velocity_rescues += 1

            if b_ok and not v_ok:
                velocity_regressions += 1

            # Track disagreements
            if cr.baseline_tid != cr.velocity_tid and len(disagreements) < 30:
                disagreements.append({
                    "rally": rally.rally_id[:8],
                    "frame": cr.pred_frame,
                    "action": cr.gt_action,
                    "gt_tid": cr.gt_tid,
                    "baseline_tid": cr.baseline_tid,
                    "velocity_tid": cr.velocity_tid,
                    "b_ok": b_ok,
                    "v_ok": v_ok,
                })

        processed += 1
        if processed % 50 == 0:
            console.print(f"  [{processed}/{len(rallies)}] processed...")

    # === Results ===
    console.print(f"\n[bold]Results[/bold] ({processed} rallies processed, {skipped} skipped)\n")

    # Table 1: Overall comparison
    b_pct = baseline_oracle_correct / max(1, baseline_oracle_total) * 100
    v_pct = velocity_oracle_correct / max(1, velocity_oracle_total) * 100
    delta = v_pct - b_pct

    console.print("[bold]Overall Oracle Accuracy[/bold]")
    t1 = Table()
    t1.add_column("Method", style="bold")
    t1.add_column("Correct", justify="right")
    t1.add_column("Total", justify="right")
    t1.add_column("Accuracy", justify="right")
    t1.add_column("Delta", justify="right")
    t1.add_row(
        "Baseline (nearest)",
        str(baseline_oracle_correct), str(baseline_oracle_total),
        f"{b_pct:.1f}%", "—",
    )
    t1.add_row(
        "Velocity (closing)",
        str(velocity_oracle_correct), str(velocity_oracle_total),
        f"{v_pct:.1f}%", f"{delta:+.1f}pp",
    )
    console.print(t1)

    # Table 2: Error-set rescue
    console.print(f"\n[bold]Error-Set Analysis[/bold]")
    console.print(f"  Baseline errors (oracle):         {baseline_errors}")
    rescue_pct = velocity_rescues / max(1, baseline_errors) * 100
    console.print(f"  Velocity rescues:                 {velocity_rescues}/{baseline_errors} ({rescue_pct:.1f}%)")
    console.print(f"  Velocity regressions (b ok, v not): {velocity_regressions}")
    net = velocity_rescues - velocity_regressions
    console.print(f"  [bold]Net: {'+' if net >= 0 else ''}{net} contacts[/bold]")

    # Table 3: Per-action breakdown
    console.print(f"\n[bold]Per-Action Breakdown[/bold]")
    t3 = Table()
    t3.add_column("Action", style="bold")
    t3.add_column("Baseline", justify="right")
    t3.add_column("Velocity", justify="right")
    t3.add_column("Delta", justify="right")
    t3.add_column("Total", justify="right")

    for action in ACTION_ORDER:
        pa = per_action.get(action)
        if not pa or pa["b_total"] == 0:
            continue
        ba = pa["b_correct"] / pa["b_total"] * 100
        va = pa["v_correct"] / pa["v_total"] * 100
        d = va - ba
        t3.add_row(
            action,
            f"{pa['b_correct']}/{pa['b_total']} ({ba:.1f}%)",
            f"{pa['v_correct']}/{pa['v_total']} ({va:.1f}%)",
            f"{d:+.1f}pp",
            str(pa["b_total"]),
        )
    console.print(t3)

    # Table 4: Disagreement examples
    if disagreements:
        console.print(f"\n[bold]Disagreement Examples (first {len(disagreements)})[/bold]")
        t4 = Table()
        t4.add_column("Rally", max_width=8)
        t4.add_column("Frame", justify="right")
        t4.add_column("Action")
        t4.add_column("GT", justify="right")
        t4.add_column("Baseline", justify="right")
        t4.add_column("Velocity", justify="right")
        t4.add_column("B ok?")
        t4.add_column("V ok?")

        for d in disagreements[:20]:
            t4.add_row(
                d["rally"],
                str(d["frame"]),
                d["action"],
                str(d["gt_tid"]),
                str(d["baseline_tid"]),
                str(d["velocity_tid"]),
                "[green]Y[/green]" if d["b_ok"] else "[red]N[/red]",
                "[green]Y[/green]" if d["v_ok"] else "[red]N[/red]",
            )
        console.print(t4)

    # Summary verdict
    console.print(f"\n[bold]Verdict[/bold]")
    if delta > 1.0:
        console.print(f"  [green]Velocity attribution is +{delta:.1f}pp over baseline. Worth pursuing.[/green]")
    elif delta > -1.0:
        console.print(f"  [yellow]Velocity attribution is {delta:+.1f}pp vs baseline. Marginal — probably not worth the complexity.[/yellow]")
    else:
        console.print(f"  [red]Velocity attribution is {delta:+.1f}pp vs baseline. Worse. Do not pursue.[/red]")


if __name__ == "__main__":
    main()
