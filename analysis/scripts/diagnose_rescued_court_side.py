"""Phase A diagnostic for Session 4 (Simpson's paradox polish).

Measure whether the sequence-rescue contacts that gained court_side errors
cluster near the net (|ball_y - net_y| < 0.05). Read-only.

Per rescued contact (present in rescue-on run, absent from rescue-off run):
  - pull ball_y from pred dict
  - estimate net_y as the median of the rally's ball y-positions (cheap
    dynamic proxy used elsewhere in the codebase)
  - compute |ball_y - net_y| and bucket: near_net (<0.05), mid (0.05-0.15),
    far (>=0.15)
  - evaluate court_side correctness via match_contacts + team_assignments

Compare bucket accuracies to the baseline (non-rescued) stratum.

Go/no-go: if >=60% of rescued court_side ERRORS are in near_net bucket,
design a team-based tiebreaker for near-net low-confidence contacts.

Usage:
    cd analysis
    uv run python scripts/diagnose_rescued_court_side.py
"""

from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import dataclass
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

from rallycut.tracking import sequence_action_runtime  # noqa: E402
from rallycut.tracking.player_tracker import PlayerPosition  # noqa: E402

console = Console()


@dataclass
class ContactRow:
    rally_id: str
    frame: int
    ball_y: float
    dist_to_net: float
    bucket: str
    court_side: str
    cs_correct: bool | None
    action: str
    confidence: float
    is_rescued: bool


def _bucket(d: float) -> str:
    if d < 0.05:
        return "near_net"
    if d < 0.15:
        return "mid"
    return "far"


def _estimate_net_y(rally: Any) -> float | None:
    """Median ball y over the rally — cheap dynamic net_y proxy."""
    bp = rally.ball_positions_json or []
    ys = [p["y"] for p in bp if p.get("y") is not None]
    if len(ys) < 10:
        return None
    return float(statistics.median(ys))


def _run_and_collect(
    rallies: list[Any],
    team_map: dict[str, dict[int, int]],
    calibrators: dict[str, Any],
    ctx: Any,
    label: str,
) -> dict[str, tuple[list[dict], float]]:
    """Return {rally_id: (pred_dicts, net_y)} for all rallies that ran."""
    out: dict[str, tuple[list[dict], float]] = {}
    for i, rally in enumerate(rallies):
        if not rally.ball_positions_json or not rally.positions_json:
            continue
        if not rally.frame_count or rally.frame_count < 10:
            continue
        net_y = _estimate_net_y(rally)
        if net_y is None:
            continue
        try:
            pred_actions = production_eval._run_rally(
                rally, team_map.get(rally.rally_id),
                calibrators.get(rally.video_id), ctx,
            )
        except Exception as e:
            console.print(f"  [red]{rally.rally_id[:8]} failed: {e}[/red]")
            continue
        real_pred = [a for a in pred_actions if not a.get("isSynthetic")]
        out[rally.rally_id] = (real_pred, net_y)
        if (i + 1) % 25 == 0:
            console.print(f"  [{label}] {i + 1}/{len(rallies)}")
    return out


def _collect_contact_rows(
    rallies: list[Any],
    base_map: dict[str, tuple[list[dict], float]],
    resc_map: dict[str, tuple[list[dict], float]],
    team_map: dict[str, dict[int, int]],
) -> list[ContactRow]:
    rows: list[ContactRow] = []
    rally_by_id = {r.rally_id: r for r in rallies}

    for rid, (resc_preds, net_y) in resc_map.items():
        rally = rally_by_id.get(rid)
        if rally is None:
            continue
        base_preds, _ = base_map.get(rid, ([], net_y))
        base_frames = {p.get("frame") for p in base_preds}

        tol = max(1, round(rally.fps * 167 / 1000))
        team_asgs = team_map.get(rid)
        matches, _ = match_contacts(
            rally.gt_labels, resc_preds,
            tolerance=tol, team_assignments=team_asgs,
        )
        # Map pred_frame -> cs_correct (only for matched preds).
        pred_cs = {
            m.pred_frame: m.court_side_correct
            for m in matches
            if m.pred_frame is not None
        }

        for p in resc_preds:
            f = p.get("frame")
            if f is None:
                continue
            by = p.get("ballY")
            if by is None:
                continue
            d = abs(by - net_y)
            rows.append(ContactRow(
                rally_id=rid,
                frame=f,
                ball_y=by,
                dist_to_net=d,
                bucket=_bucket(d),
                court_side=str(p.get("courtSide") or "unknown"),
                cs_correct=pred_cs.get(f),
                action=str(p.get("action") or "?"),
                confidence=float(p.get("confidence", 0.0)),
                is_rescued=(f not in base_frames),
            ))
    return rows


def _bucket_stats(rows: list[ContactRow], rescued: bool) -> dict[str, dict]:
    buckets: dict[str, dict] = {"near_net": {}, "mid": {}, "far": {}}
    for b in buckets:
        sel = [r for r in rows if r.is_rescued == rescued and r.bucket == b]
        evaluable = [r for r in sel if r.cs_correct is not None]
        correct = sum(1 for r in evaluable if r.cs_correct)
        buckets[b] = {
            "n": len(sel),
            "eval": len(evaluable),
            "correct": correct,
            "acc": (correct / len(evaluable)) if evaluable else None,
        }
    return buckets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tau", type=float, default=0.80)
    parser.add_argument("--floor", type=float, default=0.20)
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
    team_map = _load_match_team_assignments(
        video_ids, rally_positions=rally_pos_lookup
    )
    calibrators = production_eval._build_calibrators(video_ids)

    sequence_action_runtime.SEQ_RECOVERY_TAU = args.tau
    sequence_action_runtime.SEQ_RECOVERY_CLF_FLOOR = args.floor

    console.print("\n[bold]Run 1: baseline (rescue off)[/bold]")
    base = _run_and_collect(
        rallies, team_map, calibrators,
        production_eval.PipelineContext(skip_sequence_recovery=True),
        "base",
    )
    console.print(f"  {len(base)} rallies")

    console.print(f"\n[bold]Run 2: rescue on (tau={args.tau}, floor={args.floor})[/bold]")
    resc = _run_and_collect(
        rallies, team_map, calibrators,
        production_eval.PipelineContext(),
        "resc",
    )
    console.print(f"  {len(resc)} rallies")

    rows = _collect_contact_rows(rallies, base, resc, team_map)
    rescued_rows = [r for r in rows if r.is_rescued]
    non_rescued = [r for r in rows if not r.is_rescued]
    console.print(
        f"\n[bold]Collected {len(rows)} contacts "
        f"({len(rescued_rows)} rescued, {len(non_rescued)} pre-existing)[/bold]"
    )

    # ---- Bucket table: rescued vs non-rescued ----
    b_resc = _bucket_stats(rows, rescued=True)
    b_base = _bucket_stats(rows, rescued=False)

    tbl = Table(title="Court-side accuracy by |ball_y - net_y| bucket")
    tbl.add_column("bucket")
    tbl.add_column("rescued n", justify="right")
    tbl.add_column("rescued cs%", justify="right")
    tbl.add_column("baseline n", justify="right")
    tbl.add_column("baseline cs%", justify="right")
    tbl.add_column("gap (pp)", justify="right")
    for b in ("near_net", "mid", "far"):
        rs = b_resc[b]
        ns = b_base[b]
        r_acc = rs["acc"]
        n_acc = ns["acc"]
        gap = (
            (r_acc - n_acc) * 100
            if (r_acc is not None and n_acc is not None)
            else None
        )
        tbl.add_row(
            b,
            str(rs["n"]),
            f"{r_acc*100:.1f}" if r_acc is not None else "-",
            str(ns["n"]),
            f"{n_acc*100:.1f}" if n_acc is not None else "-",
            f"{gap:+.1f}" if gap is not None else "-",
        )
    console.print(tbl)

    # ---- Error localization ----
    resc_errors = [r for r in rescued_rows if r.cs_correct is False]
    total_err = len(resc_errors)
    near_err = sum(1 for r in resc_errors if r.bucket == "near_net")
    mid_err = sum(1 for r in resc_errors if r.bucket == "mid")
    far_err = sum(1 for r in resc_errors if r.bucket == "far")

    console.print(
        f"\n[bold]Rescued court_side errors: {total_err} "
        f"(eval denom = {sum(1 for r in rescued_rows if r.cs_correct is not None)})[/bold]"
    )
    if total_err > 0:
        console.print(
            f"  near_net: {near_err} ({near_err/total_err*100:.0f}%)"
        )
        console.print(
            f"  mid:      {mid_err} ({mid_err/total_err*100:.0f}%)"
        )
        console.print(
            f"  far:      {far_err} ({far_err/total_err*100:.0f}%)"
        )

        pct_near = near_err / total_err * 100
        console.print()
        if pct_near >= 60:
            console.print(
                f"[green]>= 60% ({pct_near:.0f}%) of errors are near-net. "
                "Go: design team-based tiebreaker (Phase B1).[/green]"
            )
        else:
            console.print(
                f"[yellow]< 60% ({pct_near:.0f}%) near-net. "
                "No-go: accept as compositional noise (Phase B2).[/yellow]"
            )

    # ---- Action breakdown of rescued errors ----
    from collections import Counter
    act_ctr: Counter[str] = Counter()
    for row in resc_errors:
        act_ctr[row.action] += 1
    console.print(f"\n[bold]Rescued error actions:[/bold] {dict(act_ctr.most_common())}")

    # ---- Confidence distribution of rescued errors ----
    if resc_errors:
        confs = sorted(r.confidence for r in resc_errors)
        mid = confs[len(confs) // 2]
        console.print(
            f"[bold]Rescued error confidence:[/bold] "
            f"median={mid:.2f} min={confs[0]:.2f} max={confs[-1]:.2f}"
        )


if __name__ == "__main__":
    main()
