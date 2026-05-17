#!/usr/bin/env python3
"""Measure trusted-29 attribution accuracy at varying GT-match windows.

Pure measurement probe — no pipeline changes. Tells us the empirical ceiling
of widening the ±5f match window to ±N. If the lift from ±5→±10 is large and
the attributions are mostly correct in the recovered NEAR cases, then a snap-to-
onset code change (shifting contact frames ~5-7f earlier) would deliver real
production lift. If the lift is small or attributions are wrong in recovered
cases, snap-to-onset isn't worth the regression risk.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import psycopg

DB_DSN = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5436/rallycut",
)
TRUSTED_29 = (
    "titi", "toto", "lulu", "wawa", "caco", "cece", "cici", "cuco",
    "gaga", "kaka", "kiki", "juju", "yeye", "keke",
    "gigi", "gugu", "mame", "meme", "mimi", "moma", "mumu",
    "papa", "pepe", "pipi", "popo", "pupu", "veve", "vivi", "vovo",
)
ACTION_TYPES = ("SERVE", "RECEIVE", "SET", "ATTACK", "DIG", "BLOCK")
WINDOWS = (5, 7, 10, 12, 15)


def match_action(actions: list[dict], gt_frame: int, gt_action: str, window: int) -> dict | None:
    """Closest action of matching type within ±window frames; else closest within ±window."""
    best = None
    best_delta = window + 1
    for a in actions:
        if (a.get("action") or "").upper() != gt_action.upper():
            continue
        delta = abs(int(a.get("frame", -10**9)) - gt_frame)
        if delta < best_delta:
            best_delta = delta
            best = a
    if best is not None:
        return best
    best_delta = window + 1
    for a in actions:
        delta = abs(int(a.get("frame", -10**9)) - gt_frame)
        if delta < best_delta:
            best_delta = delta
            best = a
    return best


def measure(window: int, gt_rows: list) -> dict:
    by_action_total: dict[str, int] = defaultdict(int)
    by_action_correct: dict[str, int] = defaultdict(int)
    by_action_unmatched: dict[str, int] = defaultdict(int)

    for _video, gt_action_raw, gt_frame, gt_tid, actions_json in gt_rows:
        gt_action = gt_action_raw.upper()
        aj = (
            actions_json if isinstance(actions_json, dict)
            else (json.loads(actions_json) if isinstance(actions_json, str) else {})
        )
        actions = aj.get("actions") or []
        a = match_action(actions, gt_frame, gt_action, window)
        by_action_total[gt_action] += 1
        if a is None:
            by_action_unmatched[gt_action] += 1
            continue
        pick = int(a.get("playerTrackId", -1))
        if pick == gt_tid:
            by_action_correct[gt_action] += 1

    total = sum(by_action_total.values())
    correct = sum(by_action_correct.values())
    unmatched = sum(by_action_unmatched.values())
    return {
        "window": window,
        "total": total,
        "correct": correct,
        "unmatched": unmatched,
        "accuracy_pct": correct / max(1, total) * 100,
        "by_action": {
            a: {
                "total": by_action_total[a],
                "correct": by_action_correct[a],
                "unmatched": by_action_unmatched[a],
                "accuracy_pct": by_action_correct[a] / max(1, by_action_total[a]) * 100,
            }
            for a in ACTION_TYPES
        },
    }


def main() -> int:
    print("Fetching trusted-29 GT rows from DB...", flush=True)
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
            [list(TRUSTED_29)],
        )
        rows = cur.fetchall()
    print(f"Got {len(rows)} GT rows. Measuring across windows {WINDOWS}...\n", flush=True)

    results = [measure(w, rows) for w in WINDOWS]

    # Header
    print(f"{'Window':>8s}  {'Total':>10s}  {'Correct':>10s}  {'Unmatched':>10s}  {'Acc%':>7s}  {'Δvs±5':>7s}")
    base_acc = results[0]["accuracy_pct"]
    base_correct = results[0]["correct"]
    for r in results:
        delta = r["accuracy_pct"] - base_acc
        delta_n = r["correct"] - base_correct
        print(f"  ±{r['window']:>3d}f  "
              f"{r['total']:>10d}  {r['correct']:>10d}  {r['unmatched']:>10d}  "
              f"{r['accuracy_pct']:>6.1f}%  {delta:+5.2f}pp ({delta_n:+d})")

    print(f"\n{'Action':<10s}", end="")
    for r in results:
        print(f"  ±{r['window']:>2d}f", end="")
    print()
    for act in ACTION_TYPES:
        print(f"{act:<10s}", end="")
        for r in results:
            ba = r["by_action"][act]
            print(f"  {ba['accuracy_pct']:>4.1f}%", end="")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
