"""Measure contact-detection recall + precision proxy on the 409-rally
action-GT corpus.

Reads pipeline actions from player_tracks.actions_json (which depends on
contact detection upstream) and compares against rally_action_ground_truth.

For an A/B run with a relaxation flag ON, this script must be run AFTER
re-running detect_contacts + action_classifier on the affected rallies
with the flag set, so actions_json reflects the new contacts.

Run from analysis/:
    uv run python -u scripts/measure_contact_recall_full.py
    uv run python -u scripts/measure_contact_recall_full.py --label baseline
    uv run python -u scripts/measure_contact_recall_full.py --skip-coherence
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, cast

from rallycut.evaluation.db import get_connection
from rallycut.tracking.coherence_invariants import run_all as run_coherence_audit

MATCH_TOLERANCE_FRAMES = 10
DEFAULT_OUT = (
    Path(__file__).resolve().parent.parent
    / "reports" / "contact_detection_fn" / "measurement_baseline_2026_05_12.json"
)


def load_rallies() -> tuple[list[dict[str, Any]], list[str]]:
    rallies: list[dict[str, Any]] = []
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT r.id::text, r.video_id::text,
                       pt.actions_json
                FROM rallies r
                JOIN player_tracks pt ON pt.rally_id = r.id
                WHERE EXISTS (
                    SELECT 1 FROM rally_action_ground_truth gt
                    WHERE gt.rally_id = r.id
                )
                """
            )
            rows = cur.fetchall()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT rally_id::text, frame, action
                FROM rally_action_ground_truth ORDER BY rally_id, frame
                """
            )
            gt_rows = cur.fetchall()
    gt_by_rally: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rid_raw, frame_raw, action_raw in gt_rows:
        rid = cast(str, rid_raw)
        gt_by_rally[rid].append({
            "frame": cast(int, frame_raw),
            "action": str(action_raw).lower() if action_raw is not None else None,
        })
    video_ids: set[str] = set()
    for rid_raw, vid_raw, actions_json in rows:
        rid = cast(str, rid_raw)
        vid = cast(str, vid_raw)
        video_ids.add(vid)
        actions: list[dict[str, Any]] = []
        if isinstance(actions_json, dict):
            actions = actions_json.get("actions", []) or []
        rallies.append({
            "rally_id": rid,
            "video_id": vid,
            "actions": actions,
            "gt": gt_by_rally.get(rid, []),
        })
    return rallies, sorted(video_ids)


def score_rally(rally: dict[str, Any]) -> dict[str, Any]:
    gt = rally["gt"]
    actions = rally["actions"]
    used_pl: set[int] = set()
    n_recalled = 0
    n_recalled_per_type: Counter[str] = Counter()
    n_gt_per_type: Counter[str] = Counter()
    for g in gt:
        action_type = g["action"] or "?"
        n_gt_per_type[action_type] += 1
        best_idx, best_dist = None, None
        for i, a in enumerate(actions):
            if i in used_pl:
                continue
            d = abs(a["frame"] - g["frame"])
            if d <= MATCH_TOLERANCE_FRAMES and (best_dist is None or d < best_dist):
                best_dist, best_idx = d, i
        if best_idx is not None:
            used_pl.add(best_idx)
            n_recalled += 1
            n_recalled_per_type[action_type] += 1
    return {
        "n_gt": len(gt),
        "n_recalled": n_recalled,
        "n_pipeline": len(actions),
        "n_pipeline_matched": len(used_pl),
        "per_type_n_gt": dict(n_gt_per_type),
        "per_type_n_recalled": dict(n_recalled_per_type),
    }


def count_coherence_violations(video_ids: list[str]) -> dict[str, int]:
    """Aggregate C-1/C-2/C-3 violation counts across all videos by calling
    the audit's Python API per video (faster + simpler than the CLI)."""
    counts = {"C1": 0, "C2": 0, "C3": 0}
    for i, vid in enumerate(video_ids, 1):
        try:
            violations = run_coherence_audit(video_id=vid)
        except Exception as e:
            print(f"  [{i}/{len(video_ids)}] {vid}: audit error {e}", flush=True)
            continue
        for v in violations:
            inv = v.invariant.replace("-", "")  # "C-1" -> "C1"
            if inv in counts:
                counts[inv] += 1
        if i % 20 == 0:
            print(
                f"  [coherence {i}/{len(video_ids)}] running totals: {counts}",
                flush=True,
            )
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--label",
        type=str,
        default="baseline",
        help="Label for this measurement (e.g., 'baseline', 'dir_change_on')",
    )
    parser.add_argument(
        "--skip-coherence",
        action="store_true",
        help="Skip the per-video coherence audit (faster smoke run)",
    )
    args = parser.parse_args(argv)

    rallies, video_ids = load_rallies()
    print(
        f"[measure] {len(rallies)} rallies / {len(video_ids)} videos loaded",
        flush=True,
    )

    totals: Counter[str] = Counter()
    per_type_n_gt: Counter[str] = Counter()
    per_type_n_recalled: Counter[str] = Counter()
    for i, r in enumerate(rallies, 1):
        s = score_rally(r)
        totals["n_gt"] += s["n_gt"]
        totals["n_recalled"] += s["n_recalled"]
        totals["n_pipeline"] += s["n_pipeline"]
        totals["n_pipeline_matched"] += s["n_pipeline_matched"]
        for t, n in s["per_type_n_gt"].items():
            per_type_n_gt[t] += n
        for t, n in s["per_type_n_recalled"].items():
            per_type_n_recalled[t] += n
        if i % 50 == 0:
            print(
                f"  [{i}/{len(rallies)}] cumulative recall: "
                f"{totals['n_recalled']}/{totals['n_gt']} = "
                f"{totals['n_recalled'] / max(totals['n_gt'], 1):.1%}",
                flush=True,
            )

    recall = totals["n_recalled"] / max(totals["n_gt"], 1)
    precision = totals["n_pipeline_matched"] / max(totals["n_pipeline"], 1)

    coherence: dict[str, int] = {}
    if not args.skip_coherence:
        print(
            f"[measure] running coherence audit on {len(video_ids)} videos...",
            flush=True,
        )
        coherence = count_coherence_violations(video_ids)

    summary: dict[str, Any] = {
        "label": args.label,
        "n_rallies": len(rallies),
        "n_videos": len(video_ids),
        "totals": dict(totals),
        "recall": recall,
        "precision_proxy": precision,
        "per_type_recall": {
            t: per_type_n_recalled.get(t, 0) / per_type_n_gt[t]
            for t in per_type_n_gt
        },
        "per_type_n_gt": dict(per_type_n_gt),
        "per_type_n_recalled": dict(per_type_n_recalled),
        "coherence_violations": coherence,
    }

    print()
    print("=" * 60)
    print(f"MEASUREMENT - label={args.label}")
    print("=" * 60)
    print(
        f"recall:           {recall:.4f}  "
        f"({totals['n_recalled']}/{totals['n_gt']})"
    )
    print(
        f"precision proxy:  {precision:.4f}  "
        f"({totals['n_pipeline_matched']}/{totals['n_pipeline']})"
    )
    for t, r_t in sorted(summary["per_type_recall"].items()):
        n = per_type_n_gt[t]
        print(
            f"  {t:<8}  recall={r_t:.4f}  "
            f"({per_type_n_recalled.get(t, 0)}/{n})"
        )
    if coherence:
        print(
            f"coherence:  C1={coherence['C1']}  "
            f"C2={coherence['C2']}  C3={coherence['C3']}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
