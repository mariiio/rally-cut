"""Compute the rescued-contact stratum's per-contact court_side and
player_attribution accuracy vs the baseline-stratum, plus check whether
any GT blocks have rescued contacts within ±5 frames.

Read-only. ~70s on CPU. Mirrors diagnose_sequence_recovery_regressions.py
but adds the stratum split + GT-block proximity check that the existing
diagnostic doesn't expose.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import production_eval  # noqa: E402
from eval_action_detection import (  # noqa: E402
    _load_match_team_assignments,
    load_rallies_with_action_gt,
    match_contacts,
)

from rallycut.tracking.player_tracker import PlayerPosition  # noqa: E402

console = Console()


def _build_team_map(
    rallies: list[Any],
) -> tuple[dict[str, dict[int, int]], dict[str, Any]]:
    rally_pos_lookup: dict[str, list[PlayerPosition]] = {}
    for r in rallies:
        if r.positions_json:
            rally_pos_lookup[r.rally_id] = [
                PlayerPosition(
                    frame_number=pp["frameNumber"],
                    track_id=pp["trackId"],
                    x=pp["x"], y=pp["y"],
                    width=pp["width"], height=pp["height"],
                    confidence=pp.get("confidence", 1.0),
                    keypoints=pp.get("keypoints"),
                )
                for pp in r.positions_json
            ]
    video_ids = {r.video_id for r in rallies if r.video_id}
    team_map = _load_match_team_assignments(video_ids, rally_positions=rally_pos_lookup)
    calibrators = production_eval._build_calibrators(video_ids)
    return team_map, calibrators


def _run(
    rallies: list[Any],
    team_map: dict[str, dict[int, int]],
    calibrators: dict[str, Any],
    ctx: Any,
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for rally in rallies:
        if not rally.ball_positions_json or not rally.positions_json:
            continue
        if not rally.frame_count or rally.frame_count < 10:
            continue
        try:
            preds = production_eval._run_rally(
                rally, team_map.get(rally.rally_id),
                calibrators.get(rally.video_id), ctx,
            )
        except Exception:
            continue
        out[rally.rally_id] = [a for a in preds if not a.get("isSynthetic")]
    return out


def main() -> None:
    console.print("[bold]Loading rallies + team map...[/bold]")
    rallies = load_rallies_with_action_gt()
    console.print(f"  {len(rallies)} rallies")
    team_map, calibrators = _build_team_map(rallies)

    console.print("\n[bold]Run baseline (rescue off)[/bold]")
    base_preds = _run(
        rallies, team_map, calibrators,
        production_eval.PipelineContext(skip_sequence_recovery=True),
    )
    console.print(f"  {len(base_preds)} rallies")

    console.print("\n[bold]Run rescue on[/bold]")
    resc_preds = _run(
        rallies, team_map, calibrators,
        production_eval.PipelineContext(),
    )
    console.print(f"  {len(resc_preds)} rallies")

    rally_lookup = {r.rally_id: r for r in rallies}

    base_stratum_cs = [0, 0]      # [correct, total]
    base_stratum_pa = [0, 0]
    rescued_stratum_cs = [0, 0]
    rescued_stratum_pa = [0, 0]

    rescued_action_dist: Counter[str] = Counter()
    block_gt_near_miss = 0
    block_gt_total = 0
    rescued_block_tp = 0
    rescued_block_fp = 0

    for rid, base_list in base_preds.items():
        if rid not in resc_preds:
            continue
        rally = rally_lookup[rid]
        tol = max(1, round(rally.fps * 167 / 1000))

        ta = team_map.get(rally.rally_id) or {}
        # match both
        b_matches, _ = match_contacts(rally.gt_labels, base_list, tolerance=tol, team_assignments=ta)
        r_matches, _ = match_contacts(rally.gt_labels, resc_preds[rid], tolerance=tol, team_assignments=ta)

        base_frames: set[int] = {
            p["frame"] for p in base_list if p.get("frame") is not None
        }
        resc_frame_set: set[int] = {
            p["frame"] for p in resc_preds[rid] if p.get("frame") is not None
        }

        # Index resc preds by frame for quick lookup
        resc_by_frame: dict[int, dict[str, Any]] = {
            p["frame"]: p for p in resc_preds[rid] if p.get("frame") is not None
        }

        # Walk r_matches: classify each matched pred frame as baseline-stratum
        # (existed in base) or rescued-stratum (new in resc).
        for m in r_matches:
            if m.pred_frame is None:
                continue
            is_rescued = m.pred_frame not in base_frames
            if m.court_side_correct is not None:
                bucket = rescued_stratum_cs if is_rescued else base_stratum_cs
                bucket[1] += 1
                if m.court_side_correct:
                    bucket[0] += 1
            if m.player_correct is not None:
                bucket = rescued_stratum_pa if is_rescued else base_stratum_pa
                bucket[1] += 1
                if m.player_correct:
                    bucket[0] += 1

        # Rescued action distribution + block accounting
        for frame in resc_frame_set - base_frames:
            p = resc_by_frame[frame]
            act = p.get("action") or "?"
            rescued_action_dist[str(act)] += 1

        # GT-block near-miss check
        for gt in rally.gt_labels:
            if gt.action != "block":
                continue
            block_gt_total += 1
            # Was this GT block matched in baseline? Skip if so.
            base_block_match = next(
                (m for m in b_matches
                 if m.gt_frame == gt.frame and m.pred_frame is not None),
                None,
            )
            if base_block_match is not None:
                continue
            # Did the rescue add any pred (any action) within ±5 frames?
            for f in resc_frame_set - base_frames:
                if abs(f - gt.frame) <= 5:
                    block_gt_near_miss += 1
                    break

        # Rescued-block TP/FP accounting
        for m in r_matches:
            if m.pred_action == "block" and m.pred_frame is not None and m.pred_frame not in base_frames:
                if m.gt_action == "block":
                    rescued_block_tp += 1
                else:
                    rescued_block_fp += 1

    def pct(b: list[int]) -> str:
        return f"{(b[0]/b[1]*100) if b[1] else 0:.1f}% ({b[0]}/{b[1]})"

    tbl = Table(title="Per-contact stratum accuracy")
    tbl.add_column("stratum")
    tbl.add_column("court_side")
    tbl.add_column("player_attribution")
    tbl.add_row("baseline (existed pre-rescue)", pct(base_stratum_cs), pct(base_stratum_pa))
    tbl.add_row("rescued (new in rescue run)", pct(rescued_stratum_cs), pct(rescued_stratum_pa))
    console.print(tbl)

    cs_gap = (
        (base_stratum_cs[0]/base_stratum_cs[1] if base_stratum_cs[1] else 0)
        - (rescued_stratum_cs[0]/rescued_stratum_cs[1] if rescued_stratum_cs[1] else 0)
    ) * 100
    pa_gap = (
        (base_stratum_pa[0]/base_stratum_pa[1] if base_stratum_pa[1] else 0)
        - (rescued_stratum_pa[0]/rescued_stratum_pa[1] if rescued_stratum_pa[1] else 0)
    ) * 100
    console.print("\n[bold]Stratum gaps (baseline − rescued):[/bold]")
    console.print(f"  court_side gap:        {cs_gap:+.1f}pp")
    console.print(f"  player_attribution gap:{pa_gap:+.1f}pp")

    console.print("\n[bold]GT-block proximity (rescue did not directly TP):[/bold]")
    console.print(f"  GT blocks that gained a ±5f rescued candidate: {block_gt_near_miss}/{block_gt_total}")

    console.print("\n[bold]Rescued-contact block accounting:[/bold]")
    console.print(f"  rescued contacts labeled 'block': "
                  f"TP={rescued_block_tp}  FP={rescued_block_fp}")

    console.print(f"\n[bold]Rescued action distribution:[/bold] {dict(rescued_action_dist.most_common())}")


if __name__ == "__main__":
    main()
