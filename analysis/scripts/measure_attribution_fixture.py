"""End-to-end attribution-accuracy harness against trusted GT.

Used by Phase 1.5+ of the crop-guided attribution research plan. Workflow
per variant under test:

1. (Optional) invoke ``rallycut reattribute-actions <video-id> <flags>``
   for each fixture video. This MUTATES the DB. Skip with ``--no-rerun``
   to just measure the current DB state.
2. Read the current ``player_tracks.actions_json`` for each rally and
   the ``videos.match_analysis_json.rallies[].trackToPlayer`` map.
3. For each trusted-GT action, find the matched prediction (±3 frames),
   normalize predicted ``playerTrackId`` through ``trackToPlayer`` to
   canonical 1-4, compare to ``trustedCanonicalPid``.
4. Print a per-fixture and combined report. Optionally write JSON.

Important: because each run mutates the DB, run baseline measurements
first (with ``--reattribute-flags ""``), then run the variant being
tested. The harness does not snapshot/restore.

Usage:
    cd analysis
    # Baseline (current DB state, no rerun):
    uv run python scripts/measure_attribution_fixture.py \\
        --trusted-gt ../reports/trusted_gt_combined_2026_04_19.json \\
        --no-rerun \\
        --label "baseline_current_db"

    # Re-run baseline reattribute-actions then measure:
    uv run python scripts/measure_attribution_fixture.py \\
        --trusted-gt ../reports/trusted_gt_combined_2026_04_19.json \\
        --reattribute-flags "" \\
        --label "baseline_freshrun"

    # VideoMAE attribution variant:
    uv run python scripts/measure_attribution_fixture.py \\
        --trusted-gt ../reports/trusted_gt_combined_2026_04_19.json \\
        --reattribute-flags "--visual" \\
        --label "videomae"
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from rallycut.evaluation.db import get_connection
from rallycut.training.action_gt_query import load_for_videos

console = Console()

FRAME_TOLERANCE = 3  # default; overridable via --frame-tolerance


@dataclass
class PredAction:
    rally_id: str
    frame: int
    raw_player_track_id: int


def load_predictions_for_videos(
    video_ids: list[str],
) -> tuple[dict[str, list[PredAction]], dict[str, dict[int, int]]]:
    """Return (rally_id → predictions list, rally_id → trackToPlayer)."""
    placeholders = ", ".join(["%s"] * len(video_ids))
    pred_query = f"""
        SELECT r.id, r.video_id, pt.actions_json
        FROM rallies r
        JOIN player_tracks pt ON pt.rally_id = r.id
        WHERE r.video_id IN ({placeholders})
          AND pt.actions_json IS NOT NULL
    """
    ma_query = f"""
        SELECT id, match_analysis_json
        FROM videos
        WHERE id IN ({placeholders})
          AND match_analysis_json IS NOT NULL
    """
    preds_by_rally: dict[str, list[PredAction]] = defaultdict(list)
    ttp_by_rally: dict[str, dict[int, int]] = {}

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(pred_query, video_ids)
            for rid, _vid, actions_json in cur.fetchall():
                rid = str(rid)
                raw_actions = (
                    actions_json.get("actions", [])
                    if isinstance(actions_json, dict) else []
                )
                if isinstance(raw_actions, dict):
                    raw_actions = raw_actions.get("actions", [])
                for a in raw_actions:
                    if not isinstance(a, dict):
                        continue
                    preds_by_rally[rid].append(PredAction(
                        rally_id=rid,
                        frame=int(a.get("frame", 0)),
                        raw_player_track_id=int(a.get("playerTrackId", -1)),
                    ))
            cur.execute(ma_query, video_ids)
            for _vid, ma_json in cur.fetchall():
                if not isinstance(ma_json, dict):
                    continue
                for entry in ma_json.get("rallies", []):
                    rid = entry.get("rallyId") or entry.get("rally_id", "")
                    ttp = entry.get("trackToPlayer") or entry.get("track_to_player", {})
                    if rid and ttp:
                        ttp_by_rally[rid] = {int(k): int(v) for k, v in ttp.items()}
    return preds_by_rally, ttp_by_rally


def match_pred_for_action(
    gt_frame: int,
    preds: list[PredAction],
    tolerance: int = FRAME_TOLERANCE,
) -> PredAction | None:
    best = None
    best_dist = tolerance + 1
    for p in preds:
        d = abs(p.frame - gt_frame)
        if d < best_dist:
            best_dist = d
            best = p
    return best


def measure(
    trusted_actions: list[dict[str, Any]],
    preds_by_rally: dict[str, list[PredAction]],
    ttp_by_rally: dict[str, dict[int, int]],
    tolerance: int = FRAME_TOLERANCE,
) -> dict[str, Any]:
    """Compare trusted-GT against predictions in DB. Return metrics dict."""
    per_video: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "total": 0, "correct": 0, "wrong": 0, "no_pred": 0,
        "by_action": defaultdict(lambda: [0, 0]),
        "no_trusted": 0,
    })
    for a in trusted_actions:
        vid = a["videoId"]
        rid = a["rallyId"]
        gt_frame = int(a["frame"])
        trusted = a.get("trustedCanonicalPid")
        action_name = str(a.get("action", ""))
        v_stats = per_video[vid]
        v_stats["total"] += 1
        if trusted is None:
            v_stats["no_trusted"] += 1
            continue
        match = match_pred_for_action(gt_frame, preds_by_rally.get(rid, []), tolerance)
        if match is None:
            v_stats["no_pred"] += 1
            v_stats["by_action"][action_name][1] += 1
            continue
        ttp = ttp_by_rally.get(rid, {})
        pred_canonical = ttp.get(match.raw_player_track_id)
        v_stats["by_action"][action_name][1] += 1
        if pred_canonical is None:
            v_stats["no_pred"] += 1
            continue
        if pred_canonical == trusted:
            v_stats["correct"] += 1
            v_stats["by_action"][action_name][0] += 1
        else:
            v_stats["wrong"] += 1

    out: dict[str, Any] = {"per_video": {}, "combined": {}}
    total = correct = wrong = no_pred = no_trusted = 0
    by_action_combined: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for vid, s in per_video.items():
        out["per_video"][vid] = {
            "total": s["total"], "correct": s["correct"],
            "wrong": s["wrong"], "no_pred": s["no_pred"],
            "no_trusted": s["no_trusted"],
            "acc_full": s["correct"] / max(1, s["total"] - s["no_trusted"]),
            "acc_predicted": s["correct"] / max(1, s["correct"] + s["wrong"]),
            "by_action": {
                k: {"correct": v[0], "total": v[1],
                    "acc": v[0] / max(1, v[1])}
                for k, v in s["by_action"].items()
            },
        }
        total += s["total"]
        correct += s["correct"]
        wrong += s["wrong"]
        no_pred += s["no_pred"]
        no_trusted += s["no_trusted"]
        for k, v in s["by_action"].items():
            by_action_combined[k][0] += v[0]
            by_action_combined[k][1] += v[1]
    eval_total = total - no_trusted
    out["combined"] = {
        "total": total,
        "no_trusted": no_trusted,
        "evaluable": eval_total,
        "correct": correct,
        "wrong": wrong,
        "no_pred": no_pred,
        "acc_full": correct / max(1, eval_total),
        "acc_predicted": correct / max(1, correct + wrong),
        "by_action": {
            k: {"correct": v[0], "total": v[1], "acc": v[0] / max(1, v[1])}
            for k, v in by_action_combined.items()
        },
    }
    return out


def run_reattribute(video_id: str, flags: str) -> tuple[float, str]:
    """Invoke ``rallycut reattribute-actions <video-id> <flags>`` synchronously."""
    cmd = ["uv", "run", "rallycut", "reattribute-actions", video_id]
    if flags:
        cmd += shlex.split(flags)
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    elapsed = time.time() - t0
    return elapsed, proc.stdout + proc.stderr


FIXTURE_VIDEO_IDS = [
    "0a383519-ecaa-411a-8e5e-e0aadc835725",
    "7d77980f-3006-40e0-adc0-db491a5bb659",
    "2e984c43-cef6-4215-8d8e-50d892b510b9",
]


def load_db_trusted_actions(video_ids: list[str]) -> list[dict[str, Any]]:
    """Read each rally's GT from rally_action_ground_truth + match_analysis_json's
    `trackToPlayer` for the given videos. Emit a flat list of actions with
    `trustedCanonicalPid` derived as ttp[gt.playerTrackId].

    Only actions where the GT's playerTrackId maps to a canonical pid are
    emitted with `trustedCanonicalPid`. Others get `None` (counted as
    "no_trusted" in measure()).
    """
    placeholders = ", ".join(["%s"] * len(video_ids))
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT id, match_analysis_json FROM videos WHERE id IN ({placeholders})",
                video_ids,
            )
            ma_rows = cur.fetchall()
            cur.execute(
                f"""
                SELECT DISTINCT r.id, r.video_id
                FROM rallies r JOIN rally_action_ground_truth gt ON gt.rally_id = r.id
                WHERE r.video_id IN ({placeholders})
                """,
                video_ids,
            )
            rally_rows = cur.fetchall()

        gt_by_rally = load_for_videos(conn, video_ids)

    ttp_by_rally: dict[str, dict[int, int]] = {}
    vid_by_rally: dict[str, str] = {str(r[0]): str(r[1]) for r in rally_rows}
    for _vid, ma in ma_rows:
        if not isinstance(ma, dict):
            continue
        for entry in ma.get("rallies", []):
            rid = entry.get("rallyId") or entry.get("rally_id", "")
            ttp = entry.get("trackToPlayer") or entry.get("track_to_player", {})
            if rid and ttp:
                ttp_by_rally[rid] = {int(k): int(v) for k, v in ttp.items()}
    actions: list[dict[str, Any]] = []
    for rid, gt_labels in gt_by_rally.items():
        vid = vid_by_rally.get(rid, "")
        ttp = ttp_by_rally.get(rid, {})
        for item in gt_labels:
            if not isinstance(item, dict):
                continue
            raw_tid = int(item.get("playerTrackId", -1))
            trusted_pid = ttp.get(raw_tid)
            actions.append({
                "videoId": vid,
                "rallyId": rid,
                "frame": int(item.get("frame", 0)),
                "action": str(item.get("action", "")),
                "rawPlayerTrackId": raw_tid,
                "trustedCanonicalPid": trusted_pid,
            })
    return actions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trusted-gt", type=Path, default=None,
        help="Optional path to a trusted-GT JSON file. If omitted, reads DB's "
             "rally_action_ground_truth directly for the three fixture videos.",
    )
    parser.add_argument("--label", required=True,
                        help="Short label for this measurement run.")
    parser.add_argument(
        "--reattribute-flags", default=None,
        help="If set, invoke `rallycut reattribute-actions <video-id> <flags>` "
             "before measurement. Empty string means re-run baseline. "
             "Omit (or pass --no-rerun) to skip and measure current DB state.",
    )
    parser.add_argument(
        "--no-rerun", action="store_true",
        help="Skip reattribute-actions; measure current DB state only.",
    )
    parser.add_argument(
        "--loo-fold", type=int, default=None,
        help="If set, filter trusted actions to rallies in this fold of "
             "reports/session3/loo_folds.json (evaluate-on-fold protocol). "
             "Report per-fold metric. Combine folds externally for median.",
    )
    parser.add_argument(
        "--loo-folds-path", type=Path,
        default=Path("../reports/session3/loo_folds.json"),
        help="Path to the LOO folds JSON (default: reports/session3/loo_folds.json).",
    )
    parser.add_argument(
        "--frame-tolerance", type=int, default=FRAME_TOLERANCE,
        help=f"Match GT frame to prediction within ±N frames (default: {FRAME_TOLERANCE}).",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.trusted_gt is not None:
        trusted = json.loads(args.trusted_gt.read_text())
        actions = trusted["actions"]
    else:
        console.print("[dim]Reading trusted GT from DB (rally_action_ground_truth).[/dim]")
        actions = load_db_trusted_actions(FIXTURE_VIDEO_IDS)
        console.print(f"[dim]  loaded {len(actions)} actions from DB[/dim]")

    if args.loo_fold is not None:
        if not args.loo_folds_path.is_file():
            console.print(f"[red]LOO folds file missing: {args.loo_folds_path}[/red]")
            return 2
        folds_doc = json.loads(args.loo_folds_path.read_text())
        folds_by_video = folds_doc["folds_by_video"]
        # Action is in the evaluated fold iff its rallyId is in that fold's list
        # for its video. Actions from videos absent from the folds file are
        # dropped (shouldn't happen with a matched fixture).
        def in_fold(action: dict[str, Any]) -> bool:
            video_folds = folds_by_video.get(action["videoId"])
            if not video_folds:
                return False
            fold_rallies = set(video_folds.get(str(args.loo_fold), []))
            return action["rallyId"] in fold_rallies

        actions = [a for a in actions if in_fold(a)]
        console.print(
            f"[dim]LOO fold {args.loo_fold}: "
            f"filtered to {len(actions)}/{len(trusted['actions'])} actions.[/dim]"
        )

    video_ids = sorted({a["videoId"] for a in actions})
    console.print(
        f"[bold]Measurement label:[/bold] {args.label}\n"
        f"  trusted GT: {args.trusted_gt}\n"
        f"  videos: {len(video_ids)}\n"
        f"  actions: {len(actions)}\n"
    )

    if not args.no_rerun and args.reattribute_flags is not None:
        console.print(
            f"[yellow]Running reattribute-actions {args.reattribute_flags!r} "
            f"on {len(video_ids)} videos (will MUTATE DB)...[/yellow]"
        )
        for vid in video_ids:
            elapsed, log = run_reattribute(vid, args.reattribute_flags)
            console.print(f"  [{vid[:8]}] reattribute-actions done in {elapsed:.1f}s")
            tail = "\n".join(log.strip().splitlines()[-5:])
            if tail:
                console.print(f"    {tail}")
    elif args.no_rerun:
        console.print("[dim]--no-rerun: measuring current DB state.[/dim]")
    else:
        console.print(
            "[dim]No --reattribute-flags provided: measuring current DB state.[/dim]"
        )

    preds, ttp = load_predictions_for_videos(video_ids)
    metrics = measure(actions, preds, ttp, args.frame_tolerance)

    table = Table(title=f"Attribution accuracy — {args.label}")
    table.add_column("Video")
    table.add_column("Total", justify="right")
    table.add_column("Correct", justify="right")
    table.add_column("Wrong", justify="right")
    table.add_column("No-pred", justify="right")
    table.add_column("Acc full", justify="right")
    table.add_column("Acc pred", justify="right")
    for vid, s in metrics["per_video"].items():
        table.add_row(
            vid[:8],
            str(s["total"]), str(s["correct"]),
            str(s["wrong"]), str(s["no_pred"]),
            f"{s['acc_full']:.1%}", f"{s['acc_predicted']:.1%}",
        )
    c = metrics["combined"]
    table.add_row(
        "[bold]COMBINED[/bold]",
        f"[bold]{c['total']}[/bold]", f"[bold]{c['correct']}[/bold]",
        f"[bold]{c['wrong']}[/bold]", f"[bold]{c['no_pred']}[/bold]",
        f"[bold]{c['acc_full']:.1%}[/bold]",
        f"[bold]{c['acc_predicted']:.1%}[/bold]",
    )
    console.print(table)

    # Per-action breakdown.
    pa = Table(title="Per-action accuracy (combined)")
    pa.add_column("Action")
    pa.add_column("Correct/Total", justify="right")
    pa.add_column("Acc", justify="right")
    for act, s in sorted(c["by_action"].items()):
        pa.add_row(act, f"{s['correct']}/{s['total']}", f"{s['acc']:.1%}")
    console.print(pa)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "label": args.label,
            "trusted_gt": str(args.trusted_gt),
            "reattribute_flags": args.reattribute_flags,
            "no_rerun": args.no_rerun,
            "metrics": metrics,
        }, indent=2))
        console.print(f"[green]wrote {args.output}[/green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
