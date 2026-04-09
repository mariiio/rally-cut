#!/usr/bin/env python3
"""Housekeeping retrack recovery driver.

After the per-rally retrack run (2026-04-09), ~40% of retracked rallies had
broken track↔player ID mappings because `remapSingleRally` uses
`existing_profiles_as_frozen_anchors=true`, which leaves fresh YOLO raw track
IDs (like T22, T41) unmapped when they don't confidently match an existing
profile. Production_eval regressed catastrophically on court_side /
player_attribution / serve_attr.

This script runs the full batch-tracking match-analysis pipeline on every
affected video, which rebuilds `videos.match_analysis_json` from scratch
(no frozen anchors) and then remaps + reattributes all rallies in the video:

    match-players → remap-track-ids → reattribute-actions

It is the same sequence the batch tracking webhook invokes on completion.
Sequential execution, per-video progress, bounded retries.

Usage:
    cd analysis
    uv run python scripts/recover_match_state.py \
        --videos outputs/housekeeping_retrack/recovery_video_ids_<ts>.txt \
        --log outputs/housekeeping_retrack/recovery_<ts>.log
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def run_cli(args: list[str], log_file) -> tuple[int, str]:
    """Run a rallycut CLI command, return (exit_code, captured_output)."""
    start = time.time()
    result = subprocess.run(
        ["uv", "run", "rallycut", *args],
        capture_output=True, text=True,
    )
    dt = time.time() - start
    log_file.write(f"  $ rallycut {' '.join(args)} -> exit {result.returncode} dt={dt:.1f}s\n")
    # Write full stdout/stderr so we can audit failures
    if result.stdout:
        for line in result.stdout.splitlines():
            log_file.write(f"    out: {line}\n")
    if result.stderr and result.returncode != 0:
        for line in result.stderr.splitlines():
            log_file.write(f"    err: {line}\n")
    log_file.flush()
    return result.returncode, result.stdout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, type=Path)
    ap.add_argument("--log", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    video_ids = [
        line.strip() for line in args.videos.read_text().splitlines() if line.strip()
    ]
    if args.limit:
        video_ids = video_ids[: args.limit]

    args.log.parent.mkdir(parents=True, exist_ok=True)
    log_file = args.log.open("a", buffering=1)

    def log(msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    log(f"START videos={len(video_ids)}")
    start_wall = time.time()

    ok_count = 0
    fail_count = 0
    failures: list[tuple[str, str]] = []

    for i, vid in enumerate(video_ids, start=1):
        v_start = time.time()
        log(f"[{i}/{len(video_ids)}] {vid[:8]} — running pipeline")

        # Step 1: match-players (rebuild match_analysis_json video-wide)
        rc, _ = run_cli(["match-players", vid, "-q"], log_file)
        if rc != 0:
            fail_count += 1
            failures.append((vid, "match-players failed"))
            log(f"  FAIL match-players rc={rc}")
            continue

        # Step 2: remap-track-ids (apply new mapping to all rallies)
        rc, _ = run_cli(["remap-track-ids", vid], log_file)
        if rc != 0:
            fail_count += 1
            failures.append((vid, "remap-track-ids failed"))
            log(f"  FAIL remap-track-ids rc={rc}")
            continue

        # Step 3: reattribute-actions (re-attribute contacts using fresh teams)
        rc, _ = run_cli(["reattribute-actions", vid], log_file)
        if rc != 0:
            fail_count += 1
            failures.append((vid, "reattribute-actions failed"))
            log(f"  FAIL reattribute-actions rc={rc}")
            continue

        ok_count += 1
        dt = time.time() - v_start
        log(f"  OK dt={dt:.1f}s")

        if i % 5 == 0:
            elapsed = time.time() - start_wall
            rate = i / elapsed
            eta_s = (len(video_ids) - i) / rate if rate > 0 else 0
            log(f"  progress: {i}/{len(video_ids)} ok={ok_count} fail={fail_count} "
                f"ETA={eta_s/60:.1f}min")

    wall = time.time() - start_wall
    log(f"DONE wall={wall/60:.1f}min ok={ok_count} fail={fail_count}")
    if failures:
        log("FAILURES:")
        for vid, msg in failures:
            log(f"  {vid}: {msg}")
    log_file.close()
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
