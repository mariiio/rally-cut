"""Rollback script for the canonical-drift repair.

Restores player_tracks rows and videos.match_analysis_json rally entries
from per-rally backups in outputs/trackid_stability/backups/. Also
restores the pose_cache .npz files from outputs/trackid_stability/
backups/pose_cache/.

Why: the initial repair_canonical_drift.py didn't permute the pose_cache
alongside positionsJson, producing a -1.26pp oracle regression. This
script reverts those 39 rallies to the pre-repair state so the corrected
repair can run on clean input.

Usage:
    cd analysis
    uv run python scripts/restore_canonical_drift_backups.py           # dry-run
    uv run python scripts/restore_canonical_drift_backups.py --apply
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import psycopg
from rich.console import Console

console = Console()

DB_CONN_STR = "host=localhost port=5436 user=postgres password=postgres dbname=rallycut"
BACKUP_DIR = Path("outputs/trackid_stability/backups")
POSE_CACHE_BACKUP_DIR = Path("outputs/trackid_stability/backups/pose_cache")
POSE_CACHE_DIR = Path("training_data/pose_cache")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    if not BACKUP_DIR.exists():
        console.print(f"[red]No backup dir: {BACKUP_DIR}[/red]")
        return 1

    backup_files = sorted(BACKUP_DIR.glob("*.json"))
    console.print(f"Found {len(backup_files)} DB backup files.")

    restored_db = 0
    restored_pose = 0
    errors: list[str] = []

    with psycopg.connect(DB_CONN_STR) as conn:
        for bpath in backup_files:
            try:
                data = json.loads(bpath.read_text())
                rally_id = data["rally_id"]
                video_id = data["video_id"]
                pt = data.get("player_track") or {}
                ma = data.get("match_analysis_json")

                if args.apply:
                    with conn.cursor() as cur:
                        cur.execute(
                            "UPDATE player_tracks SET "
                            "positions_json = %s, "
                            "contacts_json = %s, "
                            "actions_json = %s, "
                            "primary_track_ids = %s, "
                            "action_ground_truth_json = %s "
                            "WHERE rally_id = %s",
                            [
                                json.dumps(pt.get("positions_json")),
                                json.dumps(pt.get("contacts_json")),
                                json.dumps(pt.get("actions_json")),
                                json.dumps(pt.get("primary_track_ids")),
                                json.dumps(pt.get("action_ground_truth_json")),
                                rally_id,
                            ],
                        )
                        if ma is not None:
                            cur.execute(
                                "UPDATE videos SET match_analysis_json = %s WHERE id = %s",
                                [json.dumps(ma), video_id],
                            )
                    conn.commit()
                    restored_db += 1

                # Restore pose cache from snapshot (if present)
                pose_backup = POSE_CACHE_BACKUP_DIR / f"{rally_id}.npz"
                pose_live = POSE_CACHE_DIR / f"{rally_id}.npz"
                if pose_backup.exists():
                    if args.apply:
                        shutil.copyfile(pose_backup, pose_live)
                    restored_pose += 1

                print(
                    f"  [{restored_db}/{len(backup_files)}] {rally_id[:8]} "
                    f"db_restored={args.apply} pose_restore={'yes' if pose_backup.exists() else 'n/a'}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{bpath.name}: {exc}")
                if args.apply:
                    try:
                        conn.rollback()
                    except Exception:  # noqa: BLE001
                        pass

    console.print(
        f"\nRestored DB rows: {restored_db} | pose caches: {restored_pose}"
    )
    if errors:
        console.print("[red]Errors:[/red]")
        for e in errors:
            console.print(f"  {e}")
    if not args.apply:
        console.print("[yellow]DRY-RUN — re-run with --apply[/yellow]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
