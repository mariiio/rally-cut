"""Evaluate post-occlusion identity repair against labeled rallies.

Tests the repair algorithm on the 16 GT-labeled rallies, checking if it
correctly identifies and fixes the 7 known identity switches without
introducing false repairs.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, cast

from rich.console import Console
from rich.table import Table

from rallycut.evaluation.db import get_connection
from rallycut.evaluation.tracking.db import (
    get_video_path,
    load_labeled_rallies,
)
from rallycut.evaluation.tracking.metrics import (
    IdentityMetrics,
    compute_identity_metrics,
    smart_interpolate_gt,
)
from rallycut.tracking.identity_repair import (
    apply_repairs,
    repair_rally_identities,
)
from rallycut.tracking.player_features import PlayerAppearanceProfile
from rallycut.tracking.player_tracker import PlayerPosition

console = Console()


def _load_match_analysis(
    video_id: str,
) -> tuple[dict[int, PlayerAppearanceProfile], dict[str, dict[int, int]]] | None:
    """Load profiles and per-rally track-to-player mappings from DB (single query)."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT match_analysis_json FROM videos WHERE id = %s",
                [video_id],
            )
            row = cur.fetchone()

    if not row or not row[0]:
        return None

    analysis = cast(dict[str, Any], row[0])
    profiles_data = analysis.get("playerProfiles", {})
    if not profiles_data:
        return None

    profiles: dict[int, PlayerAppearanceProfile] = {}
    for pid_str, pdata in profiles_data.items():
        profiles[int(pid_str)] = PlayerAppearanceProfile.from_dict(pdata)

    # Build rally_id → track_to_player mapping
    ttp_by_rally: dict[str, dict[int, int]] = {}
    for rally_entry in analysis.get("rallies", []):
        rid = rally_entry.get("rallyId") or rally_entry.get("rally_id", "")
        ttp = rally_entry.get("trackToPlayer") or rally_entry.get(
            "track_to_player", {}
        )
        if ttp:
            ttp_by_rally[rid] = {int(k): int(v) for k, v in ttp.items()}

    return profiles, ttp_by_rally


def _compute_identity_for_rally(rally: Any) -> IdentityMetrics:
    """Compute identity metrics for a rally."""
    from rallycut.cli.commands.compare_tracking import _match_detections

    gt = smart_interpolate_gt(
        rally.ground_truth, rally.predictions, rally.predictions.frame_count,
    )

    gt_by_frame: dict[int, list] = defaultdict(list)
    pred_by_frame: dict[int, list[tuple[int, float, float, float, float]]] = defaultdict(list)

    for p in gt.player_positions:
        gt_by_frame[p.frame_number].append(
            (p.track_id, p.x, p.y, p.width, p.height)
        )
    for p in rally.predictions.positions:
        pred_by_frame[p.frame_number].append(
            (p.track_id, p.x, p.y, p.width, p.height)
        )

    matches_by_frame: dict[int, list[tuple[int, int]]] = {}
    all_frames = sorted(set(list(gt_by_frame.keys()) + list(pred_by_frame.keys())))
    for frame in all_frames:
        gt_boxes = gt_by_frame.get(frame, [])
        pred_boxes = pred_by_frame.get(frame, [])
        if gt_boxes and pred_boxes:
            m, _, _ = _match_detections(gt_boxes, pred_boxes, 0.5)
            matches_by_frame[frame] = m

    return compute_identity_metrics(pred_by_frame, matches_by_frame)


def main() -> None:
    rallies = load_labeled_rallies()
    console.print(f"Loaded {len(rallies)} labeled rallies\n")

    # Group by video for profile loading
    by_video: dict[str, list] = defaultdict(list)
    for r in rallies:
        by_video[r.video_id].append(r)

    table = Table(title="Identity Repair Evaluation")
    table.add_column("Rally", style="cyan")
    table.add_column("Video", style="dim")
    table.add_column("Cands", justify="right")
    table.add_column("Repairs", justify="right")
    table.add_column("Before IDsw", justify="right")
    table.add_column("After IDsw", justify="right")
    table.add_column("Details")

    total_repairs = 0
    total_before_sw = 0
    total_after_sw = 0
    skipped_no_profiles = 0

    for video_id, video_rallies in by_video.items():
        analysis_data = _load_match_analysis(video_id)
        video_path = get_video_path(video_id)

        if not analysis_data or not video_path:
            for r in video_rallies:
                table.add_row(
                    r.rally_id[:8], video_id[:8],
                    "-", "-", "-", "-",
                    "[yellow]no profiles/video[/yellow]",
                )
                skipped_no_profiles += 1
            continue

        profiles, ttp_by_rally = analysis_data

        for rally in video_rallies:
            ttp = ttp_by_rally.get(rally.rally_id)
            if not ttp:
                table.add_row(
                    rally.rally_id[:8], video_id[:8],
                    "-", "-", "-", "-",
                    "[yellow]no trackToPlayer[/yellow]",
                )
                continue

            # Compute identity metrics BEFORE repair
            before_metrics = _compute_identity_for_rally(rally)

            # Run repair (dry-run — work on a copy of positions)
            positions_copy = [
                PlayerPosition(
                    frame_number=p.frame_number,
                    track_id=p.track_id,
                    x=p.x, y=p.y,
                    width=p.width, height=p.height,
                    confidence=p.confidence,
                )
                for p in rally.predictions.positions
            ]

            repair_result = repair_rally_identities(
                positions=positions_copy,
                track_to_player=ttp,
                player_profiles=profiles,
                video_path=video_path,
                start_ms=rally.start_ms,
                rally_id=rally.rally_id,
            )

            if repair_result.skipped:
                table.add_row(
                    rally.rally_id[:8], video_id[:8],
                    str(repair_result.num_candidates), "-",
                    str(before_metrics.num_switches), "-",
                    f"[dim]{repair_result.skip_reason}[/dim]",
                )
                total_before_sw += before_metrics.num_switches
                total_after_sw += before_metrics.num_switches
                continue

            # Apply repairs to copy
            if repair_result.num_repairs > 0:
                apply_repairs(positions_copy, repair_result.decisions)

            # Compute identity metrics AFTER repair
            # Replace predictions positions temporarily
            orig_positions = rally.predictions.positions
            rally.predictions.positions = positions_copy
            after_metrics = _compute_identity_for_rally(rally)
            rally.predictions.positions = orig_positions

            # Build details string
            details_parts = []
            for cand in repair_result.candidates:
                details_parts.append(
                    f"  shift: track {cand.track_id} (p{cand.player_id}) "
                    f"prof {cand.best_profile_first}→{cand.best_profile_second} "
                    f"~frame {cand.approximate_switch_frame} "
                    f"cost {cand.cost_first_current:.3f}→{cand.cost_second_current:.3f}"
                )
            for dec in repair_result.decisions:
                status = "[green]SWAP[/green]" if dec.accepted else "[dim]skip[/dim]"
                ct = "cross" if dec.is_cross_team else "same"
                direction = "←" if dec.is_backward else "→"
                details_parts.append(
                    f"{status} p{dec.player_a}↔p{dec.player_b} "
                    f"{direction}frame {dec.swap_frame} "
                    f"Δ={dec.improvement:+.3f}/{dec.threshold:.2f} ({ct})"
                )

            sw_change = after_metrics.num_switches - before_metrics.num_switches
            sw_style = (
                "[green]" if sw_change < 0
                else "[red]" if sw_change > 0
                else ""
            )
            sw_suffix = f" ({sw_change:+d})" if sw_change != 0 else ""

            table.add_row(
                rally.rally_id[:8],
                video_id[:8],
                str(repair_result.num_candidates),
                str(repair_result.num_repairs),
                str(before_metrics.num_switches),
                f"{sw_style}{after_metrics.num_switches}{sw_suffix}",
                "\n".join(details_parts) if details_parts else "[dim]no shifts[/dim]",
            )

            total_repairs += repair_result.num_repairs
            total_before_sw += before_metrics.num_switches
            total_after_sw += after_metrics.num_switches

    console.print(table)
    console.print("\n[bold]Summary:[/bold]")
    console.print(f"  Total repairs applied: {total_repairs}")
    console.print(f"  Identity switches: {total_before_sw} → {total_after_sw} ({total_after_sw - total_before_sw:+d})")
    if skipped_no_profiles:
        console.print(f"  Skipped (no profiles/video): {skipped_no_profiles}")


if __name__ == "__main__":
    main()
