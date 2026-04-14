"""F3b repair migration — restore positionsJson canonical IDs on locked rallies.

Context: `analysis/outputs/trackid_stability/diagnostic_report.md` identifies
442 `canonical_drift` misses on 148 canonicalLocked rallies — pred is on
the right physical player but canonical IDs disagree because historical
retracks corrupted positionsJson.trackId via `applyRemapToRally`'s
collision shift. F3b (commit 1436811 in `api/`) prevents the drift from
recurring. This script repairs the existing drift.

Mechanism: for each locked rally with observed drift, compute the within-
rally Hungarian permutation over (pred_tid, gt_tid) pairs — the same
permutation the `player_attribution_oracle` metric already uses. Rewrite:

    - player_tracks.positions_json[].trackId
    - player_tracks.contacts_json.contacts[].playerTrackId (+ playerCandidates)
    - player_tracks.actions_json.actions[].playerTrackId
    - player_tracks.actions_json.teamAssignments (keys are trackIds)
    - player_tracks.primary_track_ids
    - videos.match_analysis_json.rallies[i].trackToPlayer (values only)

`action_ground_truth_json` is NOT rewritten — it's the alignment target.

Safeguards (all mandatory):
    - Dry-run by default. Pass --apply to mutate.
    - **Confidence gate**: best permutation score must beat second-best by
      ≥1 contact. Ambiguous rallies SKIPPED and listed for manual review.
    - **Per-rally JSON backup**: written to `outputs/trackid_stability/
      backups/<rally_id>.json` BEFORE any mutation. Rollback = restore.
    - --limit N processes only the first N affected rallies (for
      spot-check on 5 before bulk apply).
    - --rally-id X operates on a single rally (for debugging).

Usage:
    cd analysis
    uv run python scripts/repair_canonical_drift.py                    # dry-run all
    uv run python scripts/repair_canonical_drift.py --limit 5 --apply  # bulk apply 5
    uv run python scripts/repair_canonical_drift.py --apply            # bulk apply all
    uv run python scripts/repair_canonical_drift.py --rally-id abc --apply
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import psycopg
from psycopg.rows import dict_row
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
    _run_once,
)

console = Console()

DB_CONN_STR = "host=localhost port=5436 user=postgres password=postgres dbname=rallycut"
BACKUP_DIR = Path("outputs/trackid_stability/backups")
POSE_CACHE_BACKUP_DIR = Path("outputs/trackid_stability/backups/pose_cache")
REPORT_PATH = Path("outputs/trackid_stability/repair_report.json")
POSE_CACHE_DIR = Path("training_data/pose_cache")


@dataclass
class RepairDecision:
    rally_id: str
    video_id: str
    locked: bool
    n_evaluable: int
    n_correct_before: int
    n_correct_best_perm: int
    n_correct_second_perm: int
    confidence_margin: int  # best - second
    permutation: dict[int, int] = field(default_factory=dict)  # pred_canonical -> gt_canonical
    decision: str = ""  # APPLIED | SKIPPED_AMBIGUOUS | SKIPPED_NO_DRIFT | SKIPPED_NO_DATA | ERROR
    reason: str = ""
    n_positions_rewritten: int = 0
    n_contacts_rewritten: int = 0
    n_actions_rewritten: int = 0


def _load_canonical_locked(video_ids: set[str]) -> dict[str, bool]:
    if not video_ids:
        return {}
    out: dict[str, bool] = {}
    with psycopg.connect(DB_CONN_STR) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, match_analysis_json FROM videos "
            "WHERE id = ANY(%s) AND match_analysis_json IS NOT NULL",
            [list(video_ids)],
        )
        for _vid, ma in cur.fetchall():
            if not isinstance(ma, dict):
                continue
            for r in ma.get("rallies", []):
                rid = r.get("rallyId") or r.get("rally_id")
                if not rid:
                    continue
                locked = r.get("canonicalLocked")
                if locked is None:
                    locked = r.get("canonical_locked", False)
                out[rid] = bool(locked)
    return out


def _score_permutation(
    perm: dict[int, int],
    pairs: list[tuple[int, int]],
) -> int:
    """Count (gt_tid, pred_tid) pairs where perm[pred_tid] == gt_tid."""
    return sum(1 for gt, pred in pairs if perm.get(pred) == gt)


def _best_two_permutations(
    pairs: list[tuple[int, int]],
    canonical_domain: tuple[int, ...] = (1, 2, 3, 4),
) -> tuple[dict[int, int], int, dict[int, int], int]:
    """Enumerate all permutations over canonical_domain (size ≤ 4, so ≤24
    combinations) and return (best_perm, best_score, second_perm, second_score).

    We always work in the full canonical 1..4 space so the resulting
    permutation is a complete bijection applicable to all trackIds in the
    rally, not just the ones observed in contacts.
    """
    best_perm: dict[int, int] = {}
    best_score = -1
    second_perm: dict[int, int] = {}
    second_score = -1
    for perm_tuple in itertools.permutations(canonical_domain):
        perm = dict(zip(canonical_domain, perm_tuple))
        score = _score_permutation(perm, pairs)
        if score > best_score:
            second_perm = best_perm
            second_score = best_score
            best_perm = perm
            best_score = score
        elif score > second_score:
            second_perm = perm
            second_score = score
    return best_perm, best_score, second_perm, second_score


def _apply_rewrite_to_player_track(
    cur: psycopg.Cursor,
    rally_id: str,
    perm: dict[int, int],
) -> tuple[int, int, int]:
    """Apply perm (pred_canonical → gt_canonical) to player_tracks row.

    Returns (n_positions, n_contacts, n_actions) mutated.
    """
    cur.execute(
        "SELECT positions_json, contacts_json, actions_json, primary_track_ids, "
        "action_ground_truth_json "
        "FROM player_tracks WHERE rally_id = %s",
        [rally_id],
    )
    row = cur.fetchone()
    if not row:
        return (0, 0, 0)
    positions, contacts_data, actions_data, primary_ids, _action_gt = row

    def remap(tid: Any) -> Any:
        if isinstance(tid, int) and tid in perm:
            return perm[tid]
        return tid

    n_pos = 0
    if isinstance(positions, list):
        for p in positions:
            old = p.get("trackId")
            if isinstance(old, int) and old in perm and perm[old] != old:
                p["trackId"] = perm[old]
                n_pos += 1

    n_con = 0
    if isinstance(contacts_data, dict):
        contacts = contacts_data.get("contacts", []) or []
        for c in contacts:
            old = c.get("playerTrackId")
            if isinstance(old, int) and old in perm and perm[old] != old:
                c["playerTrackId"] = perm[old]
                n_con += 1
            cands = c.get("playerCandidates") or []
            for cand in cands:
                if isinstance(cand, list) and len(cand) >= 1:
                    co = cand[0]
                    if isinstance(co, int) and co in perm:
                        cand[0] = perm[co]

    n_act = 0
    if isinstance(actions_data, dict):
        actions = actions_data.get("actions", []) or []
        for a in actions:
            old = a.get("playerTrackId")
            if isinstance(old, int) and old in perm and perm[old] != old:
                a["playerTrackId"] = perm[old]
                n_act += 1
        old_ta = actions_data.get("teamAssignments") or {}
        if old_ta:
            new_ta: dict[str, Any] = {}
            for k_str, v in old_ta.items():
                try:
                    k = int(k_str)
                except (TypeError, ValueError):
                    new_ta[k_str] = v
                    continue
                new_k = perm.get(k, k)
                new_ta[str(new_k)] = v
            actions_data["teamAssignments"] = new_ta

    new_primary: list[int] = []
    if isinstance(primary_ids, list):
        for pid in primary_ids:
            if isinstance(pid, int):
                new_primary.append(perm.get(pid, pid))
            else:
                new_primary.append(pid)

    cur.execute(
        "UPDATE player_tracks SET positions_json = %s, contacts_json = %s, "
        "actions_json = %s, primary_track_ids = %s WHERE rally_id = %s",
        [
            json.dumps(positions),
            json.dumps(contacts_data),
            json.dumps(actions_data),
            json.dumps(new_primary),
            rally_id,
        ],
    )
    return (n_pos, n_con, n_act)


def _apply_rewrite_to_match_analysis(
    cur: psycopg.Cursor,
    video_id: str,
    rally_id: str,
    perm: dict[int, int],
) -> None:
    """Rewrite trackToPlayer values in match_analysis_json for this rally.

    The MAP is raw_tid → canonical. We permute the canonical values in
    place so lock-preserved mapping stays consistent with the rewritten
    positionsJson. Raw keys are untouched.
    """
    cur.execute(
        "SELECT match_analysis_json FROM videos WHERE id = %s",
        [video_id],
    )
    row = cur.fetchone()
    if not row or not isinstance(row[0], dict):
        return
    ma = row[0]
    rallies = ma.get("rallies", [])
    changed = False
    for r in rallies:
        rid = r.get("rallyId") or r.get("rally_id")
        if rid != rally_id:
            continue
        for key in ("trackToPlayer", "track_to_player"):
            t2p = r.get(key)
            if not isinstance(t2p, dict):
                continue
            new_t2p: dict[str, int] = {}
            for raw_k, canonical_v in t2p.items():
                if isinstance(canonical_v, int):
                    new_t2p[raw_k] = perm.get(canonical_v, canonical_v)
                else:
                    new_t2p[raw_k] = canonical_v
            r[key] = new_t2p
            changed = True
    if changed:
        cur.execute(
            "UPDATE videos SET match_analysis_json = %s WHERE id = %s",
            [json.dumps(ma), video_id],
        )


def _backup_rally(rally_id: str, video_id: str) -> Path:
    """Snapshot every mutable row for a rally before rewriting.

    Returns path to DB backup JSON. Also snapshots the pose_cache .npz
    (if present) to POSE_CACHE_BACKUP_DIR. Rollback: read backup + restore
    pose_cache from the snapshot.
    """
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    out_path = BACKUP_DIR / f"{rally_id}.json"
    with psycopg.connect(DB_CONN_STR) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT rally_id, positions_json, contacts_json, actions_json, "
                "primary_track_ids, action_ground_truth_json "
                "FROM player_tracks WHERE rally_id = %s",
                [rally_id],
            )
            pt = cur.fetchone()
            cur.execute(
                "SELECT match_analysis_json FROM videos WHERE id = %s",
                [video_id],
            )
            ma = cur.fetchone()
    payload = {
        "rally_id": rally_id,
        "video_id": video_id,
        "player_track": pt,
        "match_analysis_json": ma["match_analysis_json"] if ma else None,
    }
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2, default=str)

    # Pose cache snapshot (the track_ids array is what gets rewritten).
    pose_path = POSE_CACHE_DIR / f"{rally_id}.npz"
    if pose_path.exists():
        POSE_CACHE_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        pose_backup = POSE_CACHE_BACKUP_DIR / f"{rally_id}.npz"
        # Only back up if not already backed up (avoid overwriting the
        # original on a second repair pass).
        if not pose_backup.exists():
            import shutil  # noqa: PLC0415
            shutil.copyfile(pose_path, pose_backup)

    return out_path


def _apply_rewrite_to_pose_cache(
    rally_id: str,
    perm: dict[int, int],
) -> int:
    """Permute pose_cache.npz track_ids so they align with rewritten
    positionsJson.trackId. Critical — eval_action_detection._build_player_positions
    uses (frame, trackId) as the cache key; mismatched keys attach the wrong
    keypoints to the wrong player and poison downstream contact/action
    classification (see outputs/trackid_stability/diagnostic_report.md).

    Returns the number of track_id entries mutated.
    """
    pose_path = POSE_CACHE_DIR / f"{rally_id}.npz"
    if not pose_path.exists():
        return 0
    data = dict(np.load(pose_path))
    if "track_ids" not in data:
        return 0
    track_ids = data["track_ids"]
    new_track_ids = track_ids.copy()
    n_mutated = 0
    for i, tid in enumerate(track_ids):
        tid_int = int(tid)
        if tid_int in perm and perm[tid_int] != tid_int:
            new_track_ids[i] = perm[tid_int]
            n_mutated += 1
    if n_mutated == 0:
        return 0
    data["track_ids"] = new_track_ids
    # np.savez_compressed auto-appends .npz to filenames, so we write to a
    # neighbor path with a _tmp suffix (no extension), let numpy add .npz,
    # then rename. This avoids the ".npz.tmp" → ".npz.tmp.npz" pitfall.
    tmp_stem = pose_path.parent / (pose_path.stem + "_tmp")
    np.savez_compressed(tmp_stem, **data)
    tmp_written = tmp_stem.with_suffix(".npz")
    tmp_written.replace(pose_path)
    return n_mutated


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="Write mutations. Default: dry-run.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only first N rallies with observable drift.")
    ap.add_argument("--rally-id", type=str, default=None,
                    help="Single rally for debugging.")
    ap.add_argument("--locked-only", action="store_true", default=True,
                    help="Skip unlocked rallies (default True — F3a scope).")
    ap.add_argument("--include-unlocked", action="store_true",
                    help="Also repair unlocked rallies (off by default).")
    args = ap.parse_args()

    if args.include_unlocked:
        args.locked_only = False

    console.print(f"[bold]Mode:[/bold] {'APPLY' if args.apply else 'DRY-RUN'}")
    console.print(f"[bold]Locked-only:[/bold] {args.locked_only}")

    console.print("\n[bold]Loading rallies + eval state...[/bold]")
    rallies = load_rallies_with_action_gt(rally_id=args.rally_id)
    console.print(f"  {len(rallies)} rallies loaded")

    rally_pos_lookup: dict[str, list] = {}
    for r in rallies:
        if r.positions_json:
            rally_pos_lookup[r.rally_id] = _parse_positions(r.positions_json)
    video_ids = {r.video_id for r in rallies}

    team_map = _load_match_team_assignments(video_ids, rally_positions=rally_pos_lookup)
    t2p_by_rally = _load_track_to_player_maps(video_ids)
    formation_flip_by_rally = _load_formation_semantic_flips_from_gt(video_ids)
    team_templates_by_video = _load_team_templates_by_video(video_ids)
    calibrators = _build_calibrators(video_ids)
    camera_heights = _build_camera_heights(video_ids, calibrators)
    locked_by_rally = _load_canonical_locked(video_ids)

    ctx = PipelineContext()
    console.print("[bold]Running _run_once to capture per-rally match side-channels...[/bold]")
    matches, *_ = _run_once(
        rallies, team_map, calibrators, ctx, t2p_by_rally,
        formation_flip_by_rally,
        camera_heights=camera_heights,
        team_templates_by_video=team_templates_by_video,
        print_progress=True,
    )

    # Group matches by rally (match_contacts returns in gt_labels order per rally,
    # production_eval concatenates in rally order).
    matches_per_rally: dict[str, list] = {}
    cursor = 0
    for rally in rallies:
        n = len(rally.gt_labels)
        matches_per_rally[rally.rally_id] = matches[cursor : cursor + n]
        cursor += n
    if cursor != len(matches):
        console.print(
            f"[red]match cursor mismatch: cursor={cursor} matches={len(matches)}[/red]"
        )

    # Compute per-rally permutation decision
    console.print("\n[bold]Scoring permutations...[/bold]")
    decisions: list[RepairDecision] = []

    for rally in rallies:
        rid = rally.rally_id
        locked = locked_by_rally.get(rid, False)
        if args.locked_only and not locked:
            continue
        rally_matches = matches_per_rally.get(rid, [])
        # Only evaluate on evaluable, non-FN, non-block contacts with both side
        # channels set — exactly the same filter _rally_permutation_oracle uses.
        pairs: list[tuple[int, int]] = []
        for m in rally_matches:
            if not m.player_evaluable or m.gt_action == "block":
                continue
            if m.pred_frame is None:
                continue
            gt_tid = getattr(m, "_gt_tid", None)
            pred_tid = getattr(m, "_pred_tid", None)
            if gt_tid is None or pred_tid is None:
                continue
            if not (1 <= gt_tid <= 4) or not (1 <= pred_tid <= 4):
                # Skip non-canonical pairs — permutation is over 1..4.
                continue
            pairs.append((gt_tid, pred_tid))

        if not pairs:
            decisions.append(RepairDecision(
                rally_id=rid, video_id=rally.video_id, locked=locked,
                n_evaluable=0, n_correct_before=0, n_correct_best_perm=0,
                n_correct_second_perm=0, confidence_margin=0,
                decision="SKIPPED_NO_DATA", reason="no evaluable pairs",
            ))
            continue

        identity_perm = {i: i for i in (1, 2, 3, 4)}
        correct_before = _score_permutation(identity_perm, pairs)
        best_perm, best_score, second_perm, second_score = _best_two_permutations(pairs)
        margin = best_score - second_score

        # No drift: identity is already optimal.
        if best_perm == identity_perm:
            decisions.append(RepairDecision(
                rally_id=rid, video_id=rally.video_id, locked=locked,
                n_evaluable=len(pairs), n_correct_before=correct_before,
                n_correct_best_perm=best_score, n_correct_second_perm=second_score,
                confidence_margin=margin, permutation=best_perm,
                decision="SKIPPED_NO_DRIFT", reason="identity is optimal",
            ))
            continue

        if margin < 1:
            decisions.append(RepairDecision(
                rally_id=rid, video_id=rally.video_id, locked=locked,
                n_evaluable=len(pairs), n_correct_before=correct_before,
                n_correct_best_perm=best_score, n_correct_second_perm=second_score,
                confidence_margin=margin, permutation=best_perm,
                decision="SKIPPED_AMBIGUOUS",
                reason=f"margin {margin} < 1; best=second={best_score}",
            ))
            continue

        decisions.append(RepairDecision(
            rally_id=rid, video_id=rally.video_id, locked=locked,
            n_evaluable=len(pairs), n_correct_before=correct_before,
            n_correct_best_perm=best_score, n_correct_second_perm=second_score,
            confidence_margin=margin, permutation=best_perm,
            decision="PENDING_APPLY" if args.apply else "PENDING_DRY",
            reason="passes confidence gate",
        ))

    # Summary table (pre-application)
    tbl = Table(title="Repair plan (per-rally decisions)", show_lines=False)
    tbl.add_column("Decision")
    tbl.add_column("Count", justify="right")
    counts = Counter(d.decision for d in decisions)
    for k in ("PENDING_APPLY", "PENDING_DRY", "SKIPPED_AMBIGUOUS", "SKIPPED_NO_DRIFT", "SKIPPED_NO_DATA"):
        tbl.add_row(k, str(counts.get(k, 0)))
    console.print(tbl)

    affected = [d for d in decisions if d.decision.startswith("PENDING_")]
    if args.limit is not None:
        affected = affected[: args.limit]
        console.print(f"[yellow]--limit active: processing {len(affected)} of {counts.get('PENDING_APPLY', 0) + counts.get('PENDING_DRY', 0)} affected rallies[/yellow]")

    # Projected uplift: change in correct count across affected rallies
    total_pairs = sum(d.n_evaluable for d in affected)
    total_before = sum(d.n_correct_before for d in affected)
    total_after = sum(d.n_correct_best_perm for d in affected)
    console.print(
        f"\n[bold]Affected-subset projection:[/bold] "
        f"{total_before}/{total_pairs} → {total_after}/{total_pairs} "
        f"(+{total_after - total_before} contacts on this subset)"
    )

    # Per-rally table (first 20)
    rally_tbl = Table(title="Per-rally plan (first 20 affected)", show_lines=False)
    rally_tbl.add_column("Rally")
    rally_tbl.add_column("Video", style="dim")
    rally_tbl.add_column("Locked", justify="center")
    rally_tbl.add_column("Pairs", justify="right")
    rally_tbl.add_column("Before", justify="right")
    rally_tbl.add_column("After", justify="right")
    rally_tbl.add_column("Margin", justify="right")
    rally_tbl.add_column("Perm", style="dim")
    for d in affected[:20]:
        perm_str = ",".join(f"{k}->{v}" for k, v in sorted(d.permutation.items()) if k != v)
        rally_tbl.add_row(
            d.rally_id[:8], d.video_id[:8],
            "✓" if d.locked else "-",
            str(d.n_evaluable), str(d.n_correct_before),
            str(d.n_correct_best_perm), str(d.confidence_margin),
            perm_str,
        )
    console.print(rally_tbl)

    # Apply mutations (or dry-run)
    if args.apply:
        console.print(f"\n[bold red]Applying mutations to {len(affected)} rallies...[/bold red]")
        with psycopg.connect(DB_CONN_STR) as conn:
            for i, d in enumerate(affected):
                try:
                    backup_path = _backup_rally(d.rally_id, d.video_id)
                    with conn.cursor() as cur:
                        n_pos, n_con, n_act = _apply_rewrite_to_player_track(
                            cur, d.rally_id, d.permutation,
                        )
                        _apply_rewrite_to_match_analysis(
                            cur, d.video_id, d.rally_id, d.permutation,
                        )
                    conn.commit()
                    # Pose cache rewrite is on the filesystem (not the DB
                    # transaction) and must happen AFTER the commit so rollback
                    # on DB failure doesn't leave an out-of-sync .npz.
                    n_pose = _apply_rewrite_to_pose_cache(
                        d.rally_id, d.permutation,
                    )
                    d.n_positions_rewritten = n_pos
                    d.n_contacts_rewritten = n_con
                    d.n_actions_rewritten = n_act
                    d.decision = "APPLIED"
                    d.reason = f"backup={backup_path.name}"
                    print(f"  [{i+1}/{len(affected)}] {d.rally_id[:8]} "
                          f"pos={n_pos} con={n_con} act={n_act} pose={n_pose} "
                          f"(+{d.n_correct_best_perm - d.n_correct_before} contacts)", flush=True)
                except Exception as exc:  # noqa: BLE001 — log and move on
                    conn.rollback()
                    d.decision = "ERROR"
                    d.reason = f"{type(exc).__name__}: {exc}"
                    print(f"  [{i+1}/{len(affected)}] {d.rally_id[:8]} ERROR: {exc}", flush=True)

    # Persist report
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "mode": "APPLY" if args.apply else "DRY-RUN",
        "locked_only": args.locked_only,
        "limit": args.limit,
        "total_rallies_scored": len(decisions),
        "counts": dict(counts),
        "projected_uplift_on_subset": {
            "n_pairs": total_pairs,
            "correct_before": total_before,
            "correct_after": total_after,
            "delta": total_after - total_before,
        },
        "decisions": [asdict(d) for d in decisions],
    }
    with REPORT_PATH.open("w") as f:
        json.dump(report, f, indent=2)
    console.print(f"\n[dim]Wrote {REPORT_PATH}[/dim]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
