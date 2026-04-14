"""D2 — per-miss classification for the literal vs oracle player_attribution gap.

Runs production_eval._run_once on the eval set, then for each MatchResult
with player_correct=False classifies the miss into one of:

  (a) real_attribution_error   — oracle permutation does NOT recover this
  (b) canonical_drift          — oracle permutation DOES recover this
                                 (pred is the right physical player; convention
                                 differs)
  (c) fn_contact               — pred_frame is None (no predicted contact)
  (d) unmapped_raw_id          — pred_tid is not in trackToPlayer; literal
                                 compare fell back to raw int
  (e) other                    — uncategorized

Stratifies by:
  - contact_index (0=first contact of rally) for D4 serve-asymmetry check
  - canonicalLocked status of the rally
  - per-action breakdown (serve / receive / set / attack / dig / block)

Outputs:
  outputs/trackid_stability/miss_classification.json
  outputs/trackid_stability/miss_classification.md (human-readable summary)

Usage:
    cd analysis
    uv run python scripts/classify_player_attr_misses.py
    uv run python scripts/classify_player_attr_misses.py --limit 50
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import psycopg
from rich.console import Console
from rich.table import Table

from scripts.eval_action_detection import (
    _load_track_to_player_maps,
    load_rallies_with_action_gt,
)
from scripts.production_eval import (
    PipelineContext,
    _build_calibrators,
    _build_camera_heights,
    _load_formation_semantic_flips_from_gt,
    _load_match_team_assignments,
    _load_team_templates_by_video,
    _parse_positions,
    _rally_permutation_oracle,
    _run_once,
)

console = Console()


def _load_canonical_locked(video_ids: set[str]) -> dict[str, bool]:
    """Return rally_id -> canonicalLocked from match_analysis_json."""
    if not video_ids:
        return {}
    out: dict[str, bool] = {}
    sql = (
        "SELECT id, match_analysis_json FROM videos "
        "WHERE id = ANY(%s) AND match_analysis_json IS NOT NULL"
    )
    with psycopg.connect(
        "host=localhost port=5436 user=postgres password=postgres dbname=rallycut"
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, [list(video_ids)])
            for _video_id, ma_json in cur.fetchall():
                if not isinstance(ma_json, dict):
                    continue
                for r in ma_json.get("rallies", []):
                    rid = r.get("rallyId") or r.get("rally_id")
                    if not rid:
                        continue
                    locked = r.get("canonicalLocked")
                    if locked is None:
                        locked = r.get("canonical_locked", False)
                    out[rid] = bool(locked)
    return out


@dataclass
class MissRecord:
    rally_id: str
    video_id: str
    locked: bool
    gt_frame: int
    gt_action: str
    gt_tid: int | None
    pred_frame: int | None
    pred_action: str | None
    pred_tid_normalized: int | None  # post-t2p
    contact_index: int  # 0 = first GT contact in rally
    bucket: str  # one of a/b/c/d/e


@dataclass
class Aggregate:
    by_bucket: dict[str, int] = field(default_factory=dict)
    by_bucket_action: dict[str, dict[str, int]] = field(default_factory=dict)
    by_bucket_locked: dict[str, dict[str, int]] = field(default_factory=dict)
    by_bucket_contact_index: dict[str, dict[int, int]] = field(default_factory=dict)


def classify_misses(
    matches_per_rally: dict[str, list[Any]],
    locked_by_rally: dict[str, bool],
    rally_video: dict[str, str],
) -> list[MissRecord]:
    misses: list[MissRecord] = []
    for rally_id, matches in matches_per_rally.items():
        # Recompute the rally's oracle permutation so we can ask
        # "would oracle recover this?".
        _, _, permutation = _rally_permutation_oracle(matches)

        # Index GT contacts by frame for contact_index ordering.
        gt_sorted_frames = sorted(
            {m.gt_frame for m in matches if m.gt_frame is not None}
        )
        frame_to_index = {f: i for i, f in enumerate(gt_sorted_frames)}

        for m in matches:
            if m.player_correct or not m.player_evaluable:
                continue
            gt_tid = getattr(m, "_gt_tid", None)
            pred_tid = getattr(m, "_pred_tid", None)

            # Bucket logic — order matters.
            if m.pred_frame is None:
                bucket = "c_fn_contact"
            elif gt_tid is None or pred_tid is None:
                # No usable side-channel — most likely synthetic-serve match
                # or an unlabeled GT (already filtered, but be safe).
                bucket = "e_other"
            elif permutation.get(gt_tid) == pred_tid:
                # Oracle's per-rally Hungarian assignment maps gt_tid->pred_tid.
                bucket = "b_canonical_drift"
            elif pred_tid > 4 or pred_tid < 1:
                # Outside canonical 1..4 range — pred_tid is a raw YOLO int
                # that t2p didn't normalize. Could overlap with (a) but
                # specifically points at the unmapped-raw-id failure mode.
                bucket = "d_unmapped_raw_id"
            else:
                # Both gt and pred are canonical 1..4, but oracle's optimal
                # within-rally permutation also says pred is wrong.
                bucket = "a_real_attribution_error"

            misses.append(MissRecord(
                rally_id=rally_id,
                video_id=rally_video.get(rally_id, ""),
                locked=locked_by_rally.get(rally_id, False),
                gt_frame=m.gt_frame,
                gt_action=m.gt_action,
                gt_tid=gt_tid,
                pred_frame=m.pred_frame,
                pred_action=m.pred_action,
                pred_tid_normalized=pred_tid,
                contact_index=frame_to_index.get(m.gt_frame, -1),
                bucket=bucket,
            ))
    return misses


def aggregate(misses: list[MissRecord]) -> Aggregate:
    agg = Aggregate()
    for m in misses:
        agg.by_bucket[m.bucket] = agg.by_bucket.get(m.bucket, 0) + 1
        agg.by_bucket_action.setdefault(m.bucket, {})
        agg.by_bucket_action[m.bucket][m.gt_action] = (
            agg.by_bucket_action[m.bucket].get(m.gt_action, 0) + 1
        )
        lock_key = "locked" if m.locked else "unlocked"
        agg.by_bucket_locked.setdefault(m.bucket, {})
        agg.by_bucket_locked[m.bucket][lock_key] = (
            agg.by_bucket_locked[m.bucket].get(lock_key, 0) + 1
        )
        agg.by_bucket_contact_index.setdefault(m.bucket, {})
        agg.by_bucket_contact_index[m.bucket][m.contact_index] = (
            agg.by_bucket_contact_index[m.bucket].get(m.contact_index, 0) + 1
        )
    return agg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N rallies (debug).")
    ap.add_argument("--out-json", type=str,
                    default="outputs/trackid_stability/miss_classification.json")
    ap.add_argument("--out-md", type=str,
                    default="outputs/trackid_stability/miss_classification.md")
    args = ap.parse_args()

    console.print("[bold]Loading rallies with action GT...[/bold]")
    rallies = load_rallies_with_action_gt()
    if args.limit:
        rallies = rallies[: args.limit]
    console.print(f"  {len(rallies)} rallies")

    rally_pos_lookup: dict[str, list] = {}
    for r in rallies:
        if r.positions_json:
            rally_pos_lookup[r.rally_id] = _parse_positions(r.positions_json)
    video_ids = {r.video_id for r in rallies}
    rally_video = {r.rally_id: r.video_id for r in rallies}

    console.print(f"[bold]Loading match assignments for {len(video_ids)} videos...[/bold]")
    team_map = _load_match_team_assignments(video_ids, rally_positions=rally_pos_lookup)
    t2p_by_rally = _load_track_to_player_maps(video_ids)
    formation_flip_by_rally = _load_formation_semantic_flips_from_gt(video_ids)
    team_templates_by_video = _load_team_templates_by_video(video_ids)
    calibrators = _build_calibrators(video_ids)
    camera_heights = _build_camera_heights(video_ids, calibrators)
    locked_by_rally = _load_canonical_locked(video_ids)
    n_locked = sum(1 for v in locked_by_rally.values() if v)
    console.print(
        f"  t2p={len(t2p_by_rally)} cal={len(calibrators)} "
        f"locked={n_locked}/{len(locked_by_rally)}"
    )

    ctx = PipelineContext()
    console.print("[bold]Running _run_once (single pass)...[/bold]")
    matches, unmatched, rejections, pred_by_video, gt_lookup, oracle_counts, _ = _run_once(
        rallies, team_map, calibrators, ctx, t2p_by_rally,
        formation_flip_by_rally,
        camera_heights=camera_heights,
        team_templates_by_video=team_templates_by_video,
        print_progress=True,
    )
    oracle_correct, oracle_total = oracle_counts
    console.print(
        f"  matches={len(matches)} oracle={oracle_correct}/{oracle_total} "
        f"({oracle_correct / max(oracle_total, 1) * 100:.2f}%) rejections={len(rejections)}"
    )

    # Group MatchResult by rally for permutation recompute.
    matches_per_rally: dict[str, list] = defaultdict(list)
    # The MatchResult records don't carry rally_id; we reconstruct via
    # gt_frame. Re-iterate per rally and call match_contacts? Cheaper: just
    # re-run the per-rally grouping via gt_labels.
    # production_eval emits all_matches in rally order, so we can split by
    # the rally each gt_frame belongs to. But MatchResult lacks rally_id —
    # the only way is to re-run per rally. Easier: re-run a thin classifier
    # by walking rallies + their gt_labels and matching them in lockstep
    # with the matches list (length = sum of gt_labels per rally, in order).
    cursor = 0
    for rally in rallies:
        n = len(rally.gt_labels)
        matches_per_rally[rally.rally_id] = matches[cursor : cursor + n]
        cursor += n
    if cursor != len(matches):
        console.print(
            f"[red]match cursor mismatch: cursor={cursor} matches={len(matches)}[/red]"
        )

    # Classify
    console.print("[bold]Classifying misses...[/bold]")
    misses = classify_misses(matches_per_rally, locked_by_rally, rally_video)
    n_total_evaluable = sum(
        1 for m in matches if m.player_evaluable
    )
    n_correct = sum(
        1 for m in matches if m.player_evaluable and m.player_correct
    )
    n_misses = len(misses)
    console.print(
        f"  evaluable={n_total_evaluable} correct={n_correct} misses={n_misses} "
        f"literal_acc={n_correct / max(n_total_evaluable, 1) * 100:.2f}%"
    )

    # Aggregate
    agg = aggregate(misses)

    # Print summary table
    table = Table(title="Miss bucket distribution", show_header=True)
    table.add_column("Bucket")
    table.add_column("Count", justify="right")
    table.add_column("Pct of misses", justify="right")
    table.add_column("Pct of evaluable", justify="right")
    for b in sorted(agg.by_bucket.keys()):
        c = agg.by_bucket[b]
        table.add_row(
            b, str(c),
            f"{c / max(n_misses, 1) * 100:.1f}%",
            f"{c / max(n_total_evaluable, 1) * 100:.2f}%",
        )
    console.print(table)

    # Per-action × bucket
    actions = sorted({m.gt_action for m in misses})
    action_table = Table(title="Bucket × GT action (counts)", show_header=True)
    action_table.add_column("Bucket")
    for act in actions:
        action_table.add_column(act, justify="right")
    for b in sorted(agg.by_bucket.keys()):
        row = [b]
        for act in actions:
            row.append(str(agg.by_bucket_action.get(b, {}).get(act, 0)))
        action_table.add_row(*row)
    console.print(action_table)

    # Per-locked × bucket
    lock_table = Table(title="Bucket × Lock status (counts)", show_header=True)
    lock_table.add_column("Bucket")
    lock_table.add_column("locked", justify="right")
    lock_table.add_column("unlocked", justify="right")
    lock_table.add_column("locked %", justify="right")
    for b in sorted(agg.by_bucket.keys()):
        loc = agg.by_bucket_locked.get(b, {})
        n_loc = loc.get("locked", 0)
        n_unl = loc.get("unlocked", 0)
        tot = n_loc + n_unl
        pct = (n_loc / tot * 100) if tot else 0.0
        lock_table.add_row(b, str(n_loc), str(n_unl), f"{pct:.1f}%")
    console.print(lock_table)

    # Per-contact-index × bucket (for D4 serve asymmetry)
    ci_table = Table(title="Bucket × contact_index (D4 serve check)", show_header=True)
    ci_table.add_column("Bucket")
    for ci in range(0, 6):
        ci_table.add_column(f"i={ci}", justify="right")
    ci_table.add_column("i>=6", justify="right")
    for b in sorted(agg.by_bucket.keys()):
        ci_dict = agg.by_bucket_contact_index.get(b, {})
        row = [b]
        for ci in range(0, 6):
            row.append(str(ci_dict.get(ci, 0)))
        row.append(str(sum(v for k, v in ci_dict.items() if k >= 6)))
        ci_table.add_row(*row)
    console.print(ci_table)

    # Write JSON
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "n_rallies": len(rallies),
        "n_evaluable": n_total_evaluable,
        "n_correct": n_correct,
        "n_misses": n_misses,
        "literal_accuracy": n_correct / max(n_total_evaluable, 1),
        "oracle_accuracy": oracle_correct / max(oracle_total, 1),
        "buckets": agg.by_bucket,
        "bucket_action": agg.by_bucket_action,
        "bucket_locked": agg.by_bucket_locked,
        "bucket_contact_index": {
            k: {str(ci): v for ci, v in d.items()}
            for k, d in agg.by_bucket_contact_index.items()
        },
        "misses": [asdict(m) for m in misses],
    }
    with out_json.open("w") as f:
        json.dump(payload, f, indent=2)
    console.print(f"\n[dim]Wrote {out_json}[/dim]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
