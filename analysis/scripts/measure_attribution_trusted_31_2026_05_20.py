#!/usr/bin/env python3
"""Measure attribution accuracy on the trusted-31 corpus from current DB state.

For each rally_action_ground_truth row across the 32 trusted videos:
  1. Find the pipeline action whose frame is closest to the GT frame within
     ±5 frames AND whose action_type matches (preferred) — fall back to
     closest frame within ±5 if no type match.
  2. Compare action.playerTrackId vs gt.resolved_track_id.

Reports per-action and per-video accuracy. Designed to be run twice (once
with scorer OFF, once with scorer ON) to produce an A/B; the script itself
just reads DB state — toggle the scorer flag via redetect_all_actions.

Frozen snapshot for Sub-lever 1 A/B (v12 scorer chain-context fallback vs v11
baseline). Corpus: trusted-29 + koko + kuku + haha = 32 videos.
Do NOT modify this script's corpus; create a new script for future expansions.

Usage:
    cd analysis
    uv run python scripts/measure_attribution_trusted_31_2026_05_20.py
    uv run python scripts/measure_attribution_trusted_31_2026_05_20.py --label scorer_off
    uv run python scripts/measure_attribution_trusted_31_2026_05_20.py --label scorer_on --compare-to scorer_off
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import psycopg

DB_DSN = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5436/rallycut",
)
TRUSTED_31 = (
    "titi", "toto", "lulu", "wawa", "caco", "cece", "cici", "cuco",
    "gaga", "gigi", "kaka", "kiki", "keke", "koko", "kuku",
    "juju", "yeye", "gugu", "mame", "meme", "mimi", "moma", "mumu",
    "papa", "pepe", "pipi", "popo", "pupu", "veve", "vivi", "vovo",
    "haha",
)
ACTION_TYPES = ("SERVE", "RECEIVE", "SET", "ATTACK", "DIG", "BLOCK")

OUT_DIR = Path("reports/attribution_trusted_31_2026_05_20")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def match_action(actions: list[dict], gt_frame: int, gt_action: str) -> dict | None:
    """Closest action of matching type within ±5 frames; else closest within ±5."""
    best = None
    best_delta = 6
    for a in actions:
        if (a.get("action") or "").upper() != gt_action.upper():
            continue
        delta = abs(int(a.get("frame", -10**9)) - gt_frame)
        if delta < best_delta:
            best_delta = delta
            best = a
    if best is not None:
        return best
    best_delta = 6
    for a in actions:
        delta = abs(int(a.get("frame", -10**9)) - gt_frame)
        if delta < best_delta:
            best_delta = delta
            best = a
    return best


def measure() -> dict:
    by_action_total: dict[str, int] = defaultdict(int)
    by_action_correct: dict[str, int] = defaultdict(int)
    by_action_unmatched: dict[str, int] = defaultdict(int)
    by_video_total: dict[str, int] = defaultdict(int)
    by_video_correct: dict[str, int] = defaultdict(int)
    by_video_action_total: dict[tuple[str, str], int] = defaultdict(int)
    by_video_action_correct: dict[tuple[str, str], int] = defaultdict(int)

    with psycopg.connect(DB_DSN) as conn:
        cur = conn.execute(
            """
            SELECT v.name, rg.action::text, rg.frame, rg.resolved_track_id,
                   pt.actions_json
            FROM rally_action_ground_truth rg
            JOIN rallies r ON rg.rally_id = r.id
            JOIN videos v ON r.video_id = v.id
            JOIN player_tracks pt ON pt.rally_id = r.id
            WHERE v.name = ANY(%s)
              AND rg.resolved_track_id IS NOT NULL
            """,
            [list(TRUSTED_31)],
        )
        rows = cur.fetchall()

    for video, gt_action_raw, gt_frame, gt_tid, actions_json in rows:
        gt_action = gt_action_raw.upper()
        aj = (
            actions_json if isinstance(actions_json, dict)
            else (json.loads(actions_json) if isinstance(actions_json, str) else {})
        )
        actions = aj.get("actions") or []
        a = match_action(actions, gt_frame, gt_action)
        by_action_total[gt_action] += 1
        by_video_total[video] += 1
        by_video_action_total[(video, gt_action)] += 1
        if a is None:
            by_action_unmatched[gt_action] += 1
            continue
        pick = int(a.get("playerTrackId", -1))
        if pick == gt_tid:
            by_action_correct[gt_action] += 1
            by_video_correct[video] += 1
            by_video_action_correct[(video, gt_action)] += 1

    totals = {
        "total": sum(by_action_total.values()),
        "correct": sum(by_action_correct.values()),
        "unmatched": sum(by_action_unmatched.values()),
    }
    return {
        "totals": totals,
        "by_action": {
            a: {
                "total": by_action_total[a],
                "correct": by_action_correct[a],
                "unmatched": by_action_unmatched[a],
                "accuracy": (
                    by_action_correct[a] / by_action_total[a] * 100
                    if by_action_total[a] else 0.0
                ),
            }
            for a in ACTION_TYPES
        },
        "by_video": {
            v: {
                "total": by_video_total[v],
                "correct": by_video_correct[v],
                "accuracy": (
                    by_video_correct[v] / by_video_total[v] * 100
                    if by_video_total[v] else 0.0
                ),
            }
            for v in TRUSTED_31
        },
        "by_video_action": {
            f"{v}|{a}": {
                "total": by_video_action_total.get((v, a), 0),
                "correct": by_video_action_correct.get((v, a), 0),
            }
            for v in TRUSTED_31 for a in ACTION_TYPES
            if by_video_action_total.get((v, a), 0) > 0
        },
    }


def fmt_pct(num: int, denom: int) -> str:
    return f"{num}/{denom} ({num / max(1, denom) * 100:.1f}%)"


def print_report(label: str, result: dict) -> None:
    print(f"\n=== {label} ===", flush=True)
    t = result["totals"]
    print(f"  Total: {fmt_pct(t['correct'], t['total'])}, unmatched: {t['unmatched']}", flush=True)
    print("  Per action:", flush=True)
    for a in ACTION_TYPES:
        ba = result["by_action"][a]
        print(f"    {a:8s} {fmt_pct(ba['correct'], ba['total']):>16s}  unmatched={ba['unmatched']}", flush=True)
    print("  Per video:", flush=True)
    for v in TRUSTED_31:
        bv = result["by_video"][v]
        print(f"    {v:6s} {fmt_pct(bv['correct'], bv['total']):>16s}", flush=True)


def print_ab(baseline: dict, candidate: dict) -> None:
    print("\n=== A/B BASELINE vs CANDIDATE ===", flush=True)
    bt, ct = baseline["totals"], candidate["totals"]
    bp = bt["correct"] / max(1, bt["total"]) * 100
    cp = ct["correct"] / max(1, ct["total"]) * 100
    print(f"  Total accuracy: {bp:.1f}% → {cp:.1f}% (Δ {cp - bp:+.2f}pp, +{ct['correct'] - bt['correct']} correct)", flush=True)
    print("  Per action:", flush=True)
    for a in ACTION_TYPES:
        bp = baseline["by_action"][a]["accuracy"]
        cp = candidate["by_action"][a]["accuracy"]
        d = cp - bp
        marker = "▲" if d > 0.1 else ("▼" if d < -0.1 else " ")
        print(f"    {a:8s} {bp:5.1f}% → {cp:5.1f}% ({d:+5.1f}pp) {marker}", flush=True)
    print("  Per video:", flush=True)
    for v in TRUSTED_31:
        bp = baseline["by_video"][v]["accuracy"]
        cp = candidate["by_video"][v]["accuracy"]
        d = cp - bp
        marker = "▲" if d > 0.1 else ("▼" if d < -0.1 else " ")
        print(f"    {v:6s} {bp:5.1f}% → {cp:5.1f}% ({d:+5.1f}pp) {marker}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--label", type=str, default="current",
                   help="Snapshot label (e.g. scorer_off, scorer_on)")
    p.add_argument("--compare-to", type=str, default=None,
                   help="Label of a prior snapshot to A/B against")
    args = p.parse_args()

    result = measure()
    out_path = OUT_DIR / f"{args.label}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print_report(args.label, result)
    print(f"\nWrote {out_path}", flush=True)

    if args.compare_to:
        cmp_path = OUT_DIR / f"{args.compare_to}.json"
        if not cmp_path.exists():
            print(f"\nNo baseline snapshot at {cmp_path}; skipping A/B", flush=True)
        else:
            baseline = json.loads(cmp_path.read_text())
            print_ab(baseline, result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
