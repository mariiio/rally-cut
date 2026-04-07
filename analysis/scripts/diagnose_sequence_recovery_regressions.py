"""Diagnose the court_side and block-F1 regressions from sequence recovery.

Per-rally diff between baseline (rescue off) and rescue on. Reports the
rallies where court_side_accuracy or block F1 drop the most, and dumps the
rescued contacts on those rallies so we can see what was added.

Read-only. ~50s on CPU.

Usage:
    cd analysis
    uv run python scripts/diagnose_sequence_recovery_regressions.py \
        --tau 0.80 --floor 0.20
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

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

from rallycut.tracking import sequence_action_runtime  # noqa: E402
from rallycut.tracking.player_tracker import PlayerPosition  # noqa: E402

console = Console()


@dataclass
class RallyMetrics:
    contact_count: int
    tp: int
    fp: int
    fn: int
    cs_correct: int
    cs_total: int
    block_tp: int
    block_fp: int
    block_fn: int
    pred_dicts: list[dict]


def _evaluate_rally(rally, team_map, calibrators, ctx) -> RallyMetrics | None:
    try:
        pred_actions = production_eval._run_rally(
            rally, team_map.get(rally.rally_id),
            calibrators.get(rally.video_id), ctx,
        )
    except Exception:
        return None

    real_pred = [a for a in pred_actions if not a.get("isSynthetic")]
    tol = max(1, round(rally.fps * 167 / 1000))
    matches, unmatched = match_contacts(rally.gt_labels, real_pred, tolerance=tol)

    tp = sum(1 for m in matches if m.pred_frame is not None)
    fn = sum(1 for m in matches if m.pred_frame is None)
    fp = len(unmatched)

    cs_correct = sum(
        1 for m in matches
        if m.pred_frame is not None and m.court_side_correct is True
    )
    cs_total = sum(
        1 for m in matches
        if m.pred_frame is not None and m.court_side_correct is not None
    )

    block_tp = sum(
        1 for m in matches
        if m.gt_action == "block" and m.pred_action == "block"
    )
    block_fp_m = sum(
        1 for m in matches
        if m.pred_action == "block" and m.gt_action != "block" and m.pred_frame is not None
    )
    block_fp_u = sum(1 for p in unmatched if p.get("action") == "block")
    block_fn = sum(
        1 for m in matches
        if m.gt_action == "block" and (m.pred_frame is None or m.pred_action != "block")
    )

    return RallyMetrics(
        contact_count=len(real_pred),
        tp=tp, fp=fp, fn=fn,
        cs_correct=cs_correct, cs_total=cs_total,
        block_tp=block_tp, block_fp=block_fp_m + block_fp_u, block_fn=block_fn,
        pred_dicts=real_pred,
    )


def _run_all(rallies, team_map, calibrators, ctx) -> dict[str, RallyMetrics]:
    out: dict[str, RallyMetrics] = {}
    for rally in rallies:
        if not rally.ball_positions_json or not rally.positions_json:
            continue
        if not rally.frame_count or rally.frame_count < 10:
            continue
        rm = _evaluate_rally(rally, team_map, calibrators, ctx)
        if rm is not None:
            out[rally.rally_id] = rm
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tau", type=float, default=0.80)
    parser.add_argument("--floor", type=float, default=0.20)
    parser.add_argument("--top-n", type=int, default=15)
    args = parser.parse_args()

    console.print("[bold]Loading rallies + team map...[/bold]")
    rallies = load_rallies_with_action_gt()
    console.print(f"  {len(rallies)} rallies")

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

    sequence_action_runtime.SEQ_RECOVERY_TAU = args.tau
    sequence_action_runtime.SEQ_RECOVERY_CLF_FLOOR = args.floor

    console.print("\n[bold]Run 1: baseline (skip_sequence_recovery=True)[/bold]")
    base = _run_all(
        rallies, team_map, calibrators,
        production_eval.PipelineContext(skip_sequence_recovery=True),
    )
    console.print(f"  {len(base)} rallies")

    console.print(f"\n[bold]Run 2: rescue on (tau={args.tau}, floor={args.floor})[/bold]")
    resc = _run_all(
        rallies, team_map, calibrators,
        production_eval.PipelineContext(),
    )
    console.print(f"  {len(resc)} rallies")

    # Per-rally delta.
    deltas = []
    for rid in base:
        if rid not in resc:
            continue
        b = base[rid]
        r = resc[rid]
        b_cs = (b.cs_correct / b.cs_total) if b.cs_total > 0 else 0.0
        r_cs = (r.cs_correct / r.cs_total) if r.cs_total > 0 else 0.0
        cs_delta = r_cs - b_cs
        block_tp_delta = r.block_tp - b.block_tp
        block_fp_delta = r.block_fp - b.block_fp
        rescued = r.contact_count - b.contact_count
        deltas.append({
            "rid": rid,
            "cs_delta": cs_delta,
            "b_cs": b_cs,
            "r_cs": r_cs,
            "b_contacts": b.contact_count,
            "r_contacts": r.contact_count,
            "rescued": rescued,
            "block_tp_delta": block_tp_delta,
            "block_fp_delta": block_fp_delta,
            "b_obj": b,
            "r_obj": r,
        })

    # Court-side regressions.
    cs_regress = sorted([d for d in deltas if d["cs_delta"] < -0.01], key=lambda d: d["cs_delta"])
    console.print(f"\n[bold]Court-side regressions (Δ < -1pp): {len(cs_regress)}[/bold]")
    tbl = Table()
    tbl.add_column("rally")
    tbl.add_column("Δcs", justify="right")
    tbl.add_column("base%", justify="right")
    tbl.add_column("resc%", justify="right")
    tbl.add_column("+rescued", justify="right")
    tbl.add_column("Δblock_tp", justify="right")
    for d in cs_regress[: args.top_n]:
        tbl.add_row(
            d["rid"][:8], f"{d['cs_delta']*100:+.0f}",
            f"{d['b_cs']*100:.0f}", f"{d['r_cs']*100:.0f}",
            str(d["rescued"]), f"{d['block_tp_delta']:+d}",
        )
    console.print(tbl)

    # Dump rescued contacts for top 3 worst court-side regressors.
    console.print("\n[bold]Rescued contacts on worst 3 court_side regressions:[/bold]")
    for d in cs_regress[:3]:
        rid = d["rid"]
        b_frames = {p.get("frame"): p for p in d["b_obj"].pred_dicts}
        r_frames = {p.get("frame"): p for p in d["r_obj"].pred_dicts}
        rescued_preds = [p for f, p in r_frames.items() if f not in b_frames]
        lost_preds = [p for f, p in b_frames.items() if f not in r_frames]
        console.print(
            f"\n[cyan]Rally {rid[:8]}  Δcs={d['cs_delta']*100:+.0f}pp "
            f"base={d['b_contacts']} resc={d['r_contacts']}[/cyan]"
        )
        console.print(f"  Rescued ({len(rescued_preds)}):")
        for p in rescued_preds:
            console.print(
                f"    frame={p.get('frame')} "
                f"action={p.get('actionType') or p.get('action_type')} "
                f"court={p.get('courtSide') or p.get('court_side')} "
                f"track={p.get('playerTrackId') or p.get('player_track_id')} "
                f"conf={p.get('confidence', 0):.2f}"
            )
        if lost_preds:
            console.print(f"  [red]Lost ({len(lost_preds)}):[/red]")
            for p in lost_preds:
                console.print(
                    f"    frame={p.get('frame')} "
                    f"action={p.get('actionType') or p.get('action_type')} "
                    f"court={p.get('courtSide') or p.get('court_side')}"
                )

    # Block regressions.
    console.print("\n[bold]Block regressions (Δblock_tp < 0 OR Δblock_fp > 0):[/bold]")
    blk_tbl = Table()
    blk_tbl.add_column("rally")
    blk_tbl.add_column("Δtp", justify="right")
    blk_tbl.add_column("Δfp", justify="right")
    blk_tbl.add_column("+rescued", justify="right")
    blk_regress = [
        d for d in deltas
        if d["block_tp_delta"] < 0 or d["block_fp_delta"] > 0
    ]
    blk_regress.sort(key=lambda d: (d["block_tp_delta"], -d["block_fp_delta"]))
    for d in blk_regress[: args.top_n]:
        blk_tbl.add_row(
            d["rid"][:8],
            f"{d['block_tp_delta']:+d}",
            f"{d['block_fp_delta']:+d}",
            str(d["rescued"]),
        )
    console.print(blk_tbl)

    # Aggregate stats on rescued contacts.
    total_rescued = 0
    by_action: Counter[str] = Counter()
    by_court: Counter[str] = Counter()
    by_track_neg = 0
    confs: list[float] = []
    for rid in base:
        if rid not in resc:
            continue
        b_frames = {p.get("frame") for p in base[rid].pred_dicts}
        for p in resc[rid].pred_dicts:
            if p.get("frame") in b_frames:
                continue
            total_rescued += 1
            act = p.get("action") or "?"
            cs = p.get("courtSide") or p.get("court_side") or "?"
            tid = p.get("playerTrackId") or p.get("player_track_id") or -1
            by_action[str(act)] += 1
            by_court[str(cs)] += 1
            if tid is None or (isinstance(tid, int) and tid < 0):
                by_track_neg += 1
            confs.append(float(p.get("confidence", 0.0)))

    console.print(f"\n[bold]Rescued total: {total_rescued}[/bold]")
    console.print(f"  by action: {dict(by_action.most_common())}")
    console.print(f"  by court:  {dict(by_court.most_common())}")
    console.print(f"  unattributed (tid < 0): {by_track_neg}")
    if confs:
        confs_sorted = sorted(confs)
        console.print(
            f"  classifier confidence: median={confs_sorted[len(confs_sorted)//2]:.2f} "
            f"min={confs_sorted[0]:.2f} max={confs_sorted[-1]:.2f}"
        )


if __name__ == "__main__":
    main()
