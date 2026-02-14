"""Evaluate WASB+VballNet ensemble for ball tracking.

Combines WASB (high precision, lower recall) with VballNet (high recall, lower
precision) to get the best of both models.

Strategies:
  1. wasb-primary: Use WASB where available, VballNet as fallback
  2. wasb-only: WASB raw (baseline comparison)
  3. vballnet-only: VballNet filtered (baseline comparison)

Usage:
    cd analysis
    uv run python scripts/eval_ensemble.py
    uv run python scripts/eval_ensemble.py --device mps          # Use Apple GPU
    uv run python scripts/eval_ensemble.py --wasb-threshold 0.3  # Lower WASB threshold
"""

from __future__ import annotations

import argparse

from eval_wasb import run_wasb_inference
from rich.console import Console
from rich.table import Table

from rallycut.evaluation.tracking.ball_grid_search import (
    BallRawCache,
    apply_ball_filter_config,
)
from rallycut.evaluation.tracking.ball_metrics import (
    evaluate_ball_tracking,
    find_optimal_frame_offset,
)
from rallycut.evaluation.tracking.db import (
    get_video_path,
    load_labeled_rallies,
)
from rallycut.tracking.ball_filter import BallFilterConfig
from rallycut.tracking.ball_tracker import BallPosition
from rallycut.tracking.wasb_model import load_wasb_model

console = Console()


def merge_wasb_primary(
    wasb_positions: list[BallPosition],
    vballnet_positions: list[BallPosition],
) -> list[BallPosition]:
    """Merge WASB and VballNet: WASB takes priority, VballNet fills gaps.

    Args:
        wasb_positions: WASB raw detections (high precision).
        vballnet_positions: VballNet detections (high recall, may be filtered).

    Returns:
        Merged positions with source info encoded in motion_energy field:
        motion_energy >= 1.0 = WASB source, < 1.0 = VballNet source.
    """
    # Index WASB by frame (already highest-confidence per frame from inference)
    wasb_by_frame: dict[int, BallPosition] = {}
    for p in wasb_positions:
        if p.confidence > 0:
            existing = wasb_by_frame.get(p.frame_number)
            if existing is None or p.confidence > existing.confidence:
                wasb_by_frame[p.frame_number] = p

    # Index VballNet by frame
    vnet_by_frame: dict[int, BallPosition] = {}
    for p in vballnet_positions:
        if p.confidence > 0:
            existing = vnet_by_frame.get(p.frame_number)
            if existing is None or p.confidence > existing.confidence:
                vnet_by_frame[p.frame_number] = p

    # Merge: WASB primary, VballNet fallback
    merged: dict[int, BallPosition] = {}
    all_frames = set(wasb_by_frame.keys()) | set(vnet_by_frame.keys())

    for frame in all_frames:
        wasb_det = wasb_by_frame.get(frame)
        vnet_det = vnet_by_frame.get(frame)

        if wasb_det is not None:
            # Use WASB (mark source with motion_energy=1.0)
            merged[frame] = BallPosition(
                frame_number=frame,
                x=wasb_det.x,
                y=wasb_det.y,
                confidence=wasb_det.confidence,
                motion_energy=1.0,  # Tag: WASB source
            )
        elif vnet_det is not None:
            # Fallback to VballNet (keep original motion_energy < 1.0)
            merged[frame] = BallPosition(
                frame_number=frame,
                x=vnet_det.x,
                y=vnet_det.y,
                confidence=vnet_det.confidence,
                motion_energy=min(vnet_det.motion_energy, 0.99),  # Tag: VballNet source
            )

    return [merged[f] for f in sorted(merged.keys())]


def apply_light_filter(
    positions: list[BallPosition],
) -> list[BallPosition]:
    """Apply a lighter filter tuned for ensemble output.

    WASB positions are already high-precision, so we use:
    - Shorter min_segment_frames (WASB has shorter accurate segments)
    - No motion energy filter (WASB doesn't produce stationary FPs)
    - Keep interpolation (fills gaps between WASB/VballNet detections)
    - Keep segment pruning with relaxed thresholds
    """
    config = BallFilterConfig(
        # Lighter segment pruning (WASB has shorter but accurate segments)
        enable_segment_pruning=True,
        segment_jump_threshold=0.20,
        min_segment_frames=5,  # Was 15 — WASB segments can be short but accurate
        # Skip motion energy filter (WASB doesn't have this issue)
        enable_motion_energy_filter=False,
        # Keep exit ghost removal
        enable_exit_ghost_removal=True,
        # Lighter oscillation pruning
        enable_oscillation_pruning=True,
        min_oscillation_frames=12,
        # Keep outlier/blip removal
        enable_outlier_removal=True,
        enable_blip_removal=True,
        # Interpolation to fill gaps
        enable_interpolation=True,
        max_interpolation_gap=10,
        # Output threshold
        min_output_confidence=0.05,
    )
    return apply_ball_filter_config(positions, config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate WASB+VballNet ensemble")
    parser.add_argument("--device", default="mps", help="Device (cpu, cuda, mps)")
    parser.add_argument(
        "--wasb-threshold", type=float, default=0.3, help="WASB heatmap threshold"
    )
    parser.add_argument("--rally", default=None, help="Evaluate specific rally ID (prefix)")
    parser.add_argument(
        "--no-filter", action="store_true", help="Skip filter pipeline on ensemble output"
    )
    args = parser.parse_args()

    # Load models
    try:
        wasb_model = load_wasb_model(device=args.device)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        return

    console.print(f"WASB threshold: {args.wasb_threshold}, Device: {args.device}")
    console.print(f"Ensemble filter: {'disabled' if args.no_filter else 'light (tuned for ensemble)'}")
    print()

    # Load rallies
    rallies = load_labeled_rallies()
    if not rallies:
        console.print("[red]No rallies with ball GT found[/red]")
        return

    if args.rally:
        rallies = [r for r in rallies if r.rally_id.startswith(args.rally)]
        if not rallies:
            console.print(f"[red]No rally matching '{args.rally}'[/red]")
            return

    console.print(f"Evaluating {len(rallies)} rallies\n")

    raw_cache = BallRawCache()
    vnet_filter_config = BallFilterConfig()  # Standard VballNet filter

    # Results per strategy: rally_id -> metrics dict
    results: dict[str, list[tuple[str, str, dict[str, float]]]] = {
        "vballnet": [],
        "wasb": [],
        "ensemble": [],
    }

    for rally in rallies:
        rally_label = rally.rally_id[:8]
        console.print(f"[bold]Rally {rally_label}[/bold] (video {rally.video_id[:8]})")

        # --- VballNet baseline (filtered) ---
        cached = raw_cache.get(rally.rally_id)
        if cached is not None:
            vnet_raw = cached.raw_ball_positions
            vnet_filtered = apply_ball_filter_config(vnet_raw, vnet_filter_config)
        elif rally.predictions is not None and rally.predictions.ball_positions:
            vnet_raw = rally.predictions.ball_positions
            vnet_filtered = vnet_raw  # Already filtered from DB
        else:
            console.print("  [yellow]No VballNet predictions, skipping[/yellow]")
            continue

        vnet_metrics = evaluate_ball_tracking(
            ground_truth=rally.ground_truth.positions,
            predictions=vnet_filtered,
            video_width=rally.video_width,
            video_height=rally.video_height,
            video_fps=rally.video_fps,
        )
        results["vballnet"].append((rally.rally_id, rally.video_id, {
            "detection": vnet_metrics.detection_rate,
            "match": vnet_metrics.match_rate,
            "mean_err": vnet_metrics.mean_error_px,
            "gt_frames": vnet_metrics.num_gt_frames,
        }))

        # --- WASB inference ---
        video_path = get_video_path(rally.video_id)
        if video_path is None:
            console.print("  [yellow]Video not available, skipping[/yellow]")
            for key in ["wasb", "ensemble"]:
                results[key].append((rally.rally_id, rally.video_id, {
                    "detection": 0.0, "match": 0.0, "mean_err": 0.0, "gt_frames": 0,
                }))
            continue

        wasb_raw = run_wasb_inference(
            wasb_model, video_path, rally.start_ms, rally.end_ms,
            device=args.device, threshold=args.wasb_threshold,
        )

        # Find optimal offset for WASB
        wasb_offset, _ = find_optimal_frame_offset(
            rally.ground_truth.positions, wasb_raw, rally.video_width, rally.video_height,
        )

        # Apply offset to WASB
        if wasb_offset > 0:
            wasb_shifted = [
                BallPosition(
                    frame_number=p.frame_number - wasb_offset,
                    x=p.x, y=p.y,
                    confidence=p.confidence,
                    motion_energy=p.motion_energy,
                )
                for p in wasb_raw
            ]
        else:
            wasb_shifted = wasb_raw

        # Evaluate WASB alone
        wasb_metrics = evaluate_ball_tracking(
            ground_truth=rally.ground_truth.positions,
            predictions=wasb_shifted,
            video_width=rally.video_width,
            video_height=rally.video_height,
            video_fps=rally.video_fps,
        )
        results["wasb"].append((rally.rally_id, rally.video_id, {
            "detection": wasb_metrics.detection_rate,
            "match": wasb_metrics.match_rate,
            "mean_err": wasb_metrics.mean_error_px,
            "gt_frames": wasb_metrics.num_gt_frames,
        }))

        # --- Ensemble: WASB primary + VballNet fallback ---
        ensemble_merged = merge_wasb_primary(wasb_shifted, vnet_filtered)

        # Apply light filter (or no filter)
        if args.no_filter:
            ensemble_final = ensemble_merged
        else:
            ensemble_final = apply_light_filter(ensemble_merged)

        # Find optimal offset for ensemble (may differ slightly)
        ens_offset, _ = find_optimal_frame_offset(
            rally.ground_truth.positions, ensemble_final,
            rally.video_width, rally.video_height,
        )
        if ens_offset > 0:
            ensemble_eval = [
                BallPosition(
                    frame_number=p.frame_number - ens_offset,
                    x=p.x, y=p.y,
                    confidence=p.confidence,
                    motion_energy=p.motion_energy,
                )
                for p in ensemble_final
            ]
        else:
            ensemble_eval = ensemble_final

        ens_metrics = evaluate_ball_tracking(
            ground_truth=rally.ground_truth.positions,
            predictions=ensemble_eval,
            video_width=rally.video_width,
            video_height=rally.video_height,
            video_fps=rally.video_fps,
        )
        results["ensemble"].append((rally.rally_id, rally.video_id, {
            "detection": ens_metrics.detection_rate,
            "match": ens_metrics.match_rate,
            "mean_err": ens_metrics.mean_error_px,
            "gt_frames": ens_metrics.num_gt_frames,
        }))

        # Count sources in ensemble
        wasb_frames = sum(1 for p in ensemble_merged if p.motion_energy >= 1.0)
        vnet_frames = sum(1 for p in ensemble_merged if p.motion_energy < 1.0)

        # Print per-rally
        console.print(
            f"  VballNet:  det={vnet_metrics.detection_rate:.1%}  "
            f"match={vnet_metrics.match_rate:.1%}  err={vnet_metrics.mean_error_px:.1f}px"
        )
        console.print(
            f"  WASB:      det={wasb_metrics.detection_rate:.1%}  "
            f"match={wasb_metrics.match_rate:.1%}  err={wasb_metrics.mean_error_px:.1f}px  "
            f"(offset={wasb_offset})"
        )
        console.print(
            f"  Ensemble:  det={ens_metrics.detection_rate:.1%}  "
            f"match=[bold]{ens_metrics.match_rate:.1%}[/bold]  "
            f"err={ens_metrics.mean_error_px:.1f}px  "
            f"({wasb_frames}W+{vnet_frames}V={wasb_frames + vnet_frames} frames, "
            f"offset={ens_offset})"
        )

        # Show best and delta
        best_match = max(vnet_metrics.match_rate, wasb_metrics.match_rate, ens_metrics.match_rate)
        if ens_metrics.match_rate >= best_match - 0.005:
            console.print("  [green]  Ensemble is best or tied[/green]")
        elif ens_metrics.match_rate > vnet_metrics.match_rate:
            console.print(
                f"  [yellow]  Ensemble beats VballNet "
                f"(+{(ens_metrics.match_rate - vnet_metrics.match_rate):.1%}) "
                f"but not WASB alone[/yellow]"
            )
        else:
            console.print("  [red]  Ensemble underperforms[/red]")
        print()

    # --- Summary table ---
    if not results["vballnet"]:
        return

    console.print("[bold]== Summary: VballNet vs WASB vs Ensemble ==[/bold]")
    table = Table(show_header=True, header_style="bold")
    table.add_column("Rally")
    table.add_column("VNet Match%", justify="right")
    table.add_column("WASB Match%", justify="right")
    table.add_column("Ens Match%", justify="right")
    table.add_column("VNet Err", justify="right")
    table.add_column("WASB Err", justify="right")
    table.add_column("Ens Err", justify="right")
    table.add_column("Best")

    totals = {k: {"match": 0.0, "err": 0.0, "det": 0.0} for k in results}
    count = 0

    for i in range(len(results["vballnet"])):
        rid = results["vballnet"][i][0][:8]
        vr = results["vballnet"][i][2]
        wr = results["wasb"][i][2]
        er = results["ensemble"][i][2]

        matches = {"VballNet": vr["match"], "WASB": wr["match"], "Ensemble": er["match"]}
        best_name = max(matches, key=matches.get)  # type: ignore[arg-type]
        if best_name == "Ensemble":
            best_label = "[green bold]Ensemble[/green bold]"
        elif best_name == "WASB":
            best_label = "[cyan]WASB[/cyan]"
        else:
            best_label = "[blue]VballNet[/blue]"

        # Bold the best match% in each row
        vm = f"{vr['match']:.1%}"
        wm = f"{wr['match']:.1%}"
        em = f"{er['match']:.1%}"
        if best_name == "VballNet":
            vm = f"[bold]{vm}[/bold]"
        elif best_name == "WASB":
            wm = f"[bold]{wm}[/bold]"
        else:
            em = f"[bold]{em}[/bold]"

        table.add_row(
            rid, vm, wm, em,
            f"{vr['mean_err']:.1f}px",
            f"{wr['mean_err']:.1f}px" if wr["mean_err"] > 0 else "N/A",
            f"{er['mean_err']:.1f}px",
            best_label,
        )

        for k, r in [("vballnet", vr), ("wasb", wr), ("ensemble", er)]:
            totals[k]["match"] += r["match"]
            totals[k]["err"] += r["mean_err"]
            totals[k]["det"] += r["detection"]
        count += 1

    if count > 0:
        # Average row
        avg_matches = {
            "VballNet": totals["vballnet"]["match"] / count,
            "WASB": totals["wasb"]["match"] / count,
            "Ensemble": totals["ensemble"]["match"] / count,
        }
        best_avg = max(avg_matches, key=avg_matches.get)  # type: ignore[arg-type]
        if best_avg == "Ensemble":
            avg_label = "[green bold]Ensemble[/green bold]"
        elif best_avg == "WASB":
            avg_label = "[cyan]WASB[/cyan]"
        else:
            avg_label = "[blue]VballNet[/blue]"

        table.add_row(
            "[bold]Average",
            f"[bold]{totals['vballnet']['match'] / count:.1%}",
            f"[bold]{totals['wasb']['match'] / count:.1%}",
            f"[bold]{totals['ensemble']['match'] / count:.1%}",
            f"[bold]{totals['vballnet']['err'] / count:.1f}px",
            f"[bold]{totals['wasb']['err'] / count:.1f}px",
            f"[bold]{totals['ensemble']['err'] / count:.1f}px",
            avg_label,
        )

    console.print(table)

    # Final stats
    if count > 0:
        print()
        for name, key in [("VballNet", "vballnet"), ("WASB", "wasb"), ("Ensemble", "ensemble")]:
            avg_m = totals[key]["match"] / count
            avg_e = totals[key]["err"] / count
            avg_d = totals[key]["det"] / count
            console.print(f"  {name:10s}: det={avg_d:.1%}  match={avg_m:.1%}  err={avg_e:.1f}px")

        ens_gain = (totals["ensemble"]["match"] - totals["vballnet"]["match"]) / count
        console.print(f"\n  Ensemble vs VballNet: {ens_gain:+.1%} match rate")
        ens_vs_wasb = (totals["ensemble"]["match"] - totals["wasb"]["match"]) / count
        console.print(f"  Ensemble vs WASB:     {ens_vs_wasb:+.1%} match rate")


if __name__ == "__main__":
    main()
