#!/usr/bin/env python3
"""One-off housekeeping-retrack driver for the 340 production_eval rallies.

Refreshes each rally's `player_tracks.ball_positions_json` (and player/pose
data as a side effect) via `POST /v1/rallies/:id/track-players`, so the
next `production_eval.py` run measures the Session 3 ball-filter change
(commit 8697970) end-to-end.

Not committed. Scoped strictly to the housekeeping retrack described in
`memory/session_playbook.md` "Housekeeping retracks".

Properties:
- Concurrency 2 (empirically validated: ~1.97x speedup, N=3 regresses).
- Resumable: skips rallies whose `completed_at` is after the snapshot
  timestamp recorded in the targets file.
- Bounded retries on transient HTTP failures (3 attempts, exp backoff).
- Streams progress to a log file; prints per-rally results to stdout.
- No output truncation (per CLAUDE.md "Running Diagnostics & Long Processes").

Usage:
    cd analysis
    uv run python scripts/retrack_eval_rallies.py \
        --targets outputs/housekeeping_retrack/retrack_targets_<ts>.tsv \
        --snapshot-ts "2026-04-09 10:45:11" \
        --log outputs/housekeeping_retrack/retrack_<ts>.log
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import requests

API_BASE = "http://localhost:3001"
VISITOR_ID = "405a071b-4800-4337-b0c2-0d5b59bda16f"
HEADERS = {"X-Visitor-Id": VISITOR_ID, "Content-Type": "application/json"}

PG_ENV = {"PGPASSWORD": "postgres", "PATH": "/opt/homebrew/opt/postgresql@15/bin:/usr/bin:/bin"}
PSQL = "/opt/homebrew/opt/postgresql@15/bin/psql"
PSQL_CMD = [PSQL, "-h", "localhost", "-p", "5436", "-U", "postgres", "-d", "rallycut", "-t", "-A"]


def parse_targets(path: Path) -> list[tuple[str, str, str, str]]:
    """Parse TSV: rally_id\tvideo_id\tcompleted_at\tstatus."""
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        rows.append((parts[0], parts[1], parts[2], parts[3]))
    return rows


def get_completed_epoch(rally_id: str) -> float | None:
    """Return completed_at as unix epoch (UTC). None if not set."""
    result = subprocess.run(
        [*PSQL_CMD, "-c",
         f"SELECT EXTRACT(EPOCH FROM completed_at)::bigint "
         f"FROM player_tracks WHERE rally_id='{rally_id}';"],
        capture_output=True, text=True, env={**PG_ENV},
    )
    out = result.stdout.strip()
    if not out:
        return None
    try:
        return float(out)
    except ValueError:
        return None


def is_already_fresh(rally_id: str, snapshot_epoch: float) -> bool:
    """Skip rally if its completed_at is already newer than the snapshot."""
    ep = get_completed_epoch(rally_id)
    if ep is None:
        return False
    return ep > snapshot_epoch


def retrack_one(rally_id: str, attempts: int = 3) -> tuple[str, bool, str]:
    """Retrack a single rally via per-rally endpoint. Returns (rally_id, ok, msg)."""
    url = f"{API_BASE}/v1/rallies/{rally_id}/track-players"
    for attempt in range(1, attempts + 1):
        t0 = time.time()
        try:
            resp = requests.post(url, headers=HEADERS, json={}, timeout=800)
            dt = time.time() - t0
            if resp.status_code == 200:
                data = resp.json()
                n_ball = len(data.get("ballPositions", []))
                return rally_id, True, f"ok ball_pts={n_ball} dt={dt:.0f}s"
            msg = f"HTTP {resp.status_code}: {resp.text[:120]}"
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
        if attempt < attempts:
            time.sleep(2 ** attempt)
    return rally_id, False, f"failed after {attempts}: {msg}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True, type=Path)
    ap.add_argument("--snapshot-epoch", required=True, type=float,
                    help="Unix epoch of the snapshot; rallies with completed_at > this are skipped")
    ap.add_argument("--log", required=True, type=Path)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None, help="Only process first N targets (for testing)")
    args = ap.parse_args()

    snapshot_epoch = args.snapshot_epoch
    targets = parse_targets(args.targets)
    if args.limit:
        targets = targets[: args.limit]
    total = len(targets)

    args.log.parent.mkdir(parents=True, exist_ok=True)
    log_file = args.log.open("a", buffering=1)

    def log(msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        log_file.write(line + "\n")

    log(f"START targets={total} concurrency={args.concurrency} snapshot_epoch={snapshot_epoch}")

    # Filter out already-fresh rallies
    todo: list[tuple[str, str]] = []
    skipped = 0
    for rid, vid, _completed, _status in targets:
        if is_already_fresh(rid, snapshot_epoch):
            skipped += 1
            log(f"SKIP {rid[:8]} video={vid[:8]} (already fresh)")
            continue
        todo.append((rid, vid))
    log(f"After resume-filter: todo={len(todo)} skipped={skipped}")

    if not todo:
        log("Nothing to do. Exiting.")
        return 0

    start_wall = time.time()
    ok_count = 0
    fail_count = 0
    failures: list[tuple[str, str]] = []

    idx = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(retrack_one, rid): (rid, vid) for rid, vid in todo}
        for fut in as_completed(futures):
            rid, vid = futures[fut]
            idx += 1
            try:
                _, ok, msg = fut.result()
            except Exception as e:  # noqa: BLE001
                ok, msg = False, f"driver exception: {type(e).__name__}: {e}"
            if ok:
                ok_count += 1
                log(f"[{idx}/{len(todo)}] {rid[:8]} {msg}")
            else:
                fail_count += 1
                failures.append((rid, msg))
                log(f"[{idx}/{len(todo)}] {rid[:8]} FAIL {msg}")
            # ETA
            if idx % 5 == 0:
                elapsed = time.time() - start_wall
                rate = idx / elapsed
                eta_s = (len(todo) - idx) / rate if rate > 0 else 0
                log(f"  progress: {idx}/{len(todo)} ok={ok_count} fail={fail_count} "
                    f"rate={rate*60:.1f}/min ETA={eta_s/3600:.2f}h")

    wall = time.time() - start_wall
    log(f"DONE wall={wall/3600:.2f}h ok={ok_count} fail={fail_count} skipped_fresh={skipped}")
    if failures:
        log("FAILURES:")
        for rid, msg in failures:
            log(f"  {rid}: {msg}")
    log_file.close()
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
