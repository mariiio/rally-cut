r"""Validate A1 quality checks against negative + positive video fixtures.

Usage:
    uv run python analysis/scripts/validate_quality_checks.py \
        --videos ~/Desktop/rallies/Negative/bad\ angle.mp4 \
        --videos ~/Desktop/rallies/Matches/match.mp4

Prints per-video: metrics + firing status for each surviving preflight check.

Post-2026-04-15 the preflight surface is just metadata invariants + camera
geometry (court-keypoint confidence + behind-baseline heuristic). The
validation harness matches.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _print(msg: str) -> None:
    print(msg, flush=True)


def validate_one(video_path: str, sample_seconds: int) -> dict:
    # Imports inside the function so `--help` works without heavy deps.
    from rallycut.quality.camera_geometry import (
        MIN_COURT_CONFIDENCE,
        check_camera_geometry,
    )
    from rallycut.quality.metadata import MIN_FPS, MIN_WIDTH, check_metadata
    from rallycut.quality.runner import _load_video_inputs

    t0 = time.perf_counter()
    meta, corners = _load_video_inputs(video_path, sample_seconds=sample_seconds)
    t_load = time.perf_counter() - t0

    results = {
        "metadata": check_metadata(meta),
        "camera_geometry": check_camera_geometry(corners),
    }

    issues = []
    metrics: dict[str, float] = {}
    for check_name, r in results.items():
        metrics.update(r.metrics)
        for i in r.issues:
            issues.append({
                "check": check_name,
                "id": i.id,
                "tier": i.tier.value,
                "severity": round(i.severity, 3),
                "data": {k: round(v, 4) for k, v in i.data.items()},
            })

    thresholds = {
        "MIN_FPS": MIN_FPS,
        "MIN_WIDTH": MIN_WIDTH,
        "MIN_COURT_CONFIDENCE": MIN_COURT_CONFIDENCE,
    }

    return {
        "video": video_path,
        "load_time_s": round(t_load, 1),
        "metrics": {k: round(v, 4) for k, v in metrics.items()},
        "issues": issues,
        "thresholds": thresholds,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--videos",
        action="append",
        required=True,
        help="Path to a video file (repeatable).",
    )
    parser.add_argument("--sample-seconds", type=int, default=60)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON dump.")
    args = parser.parse_args()

    paths = [Path(v).expanduser() for v in args.videos]
    missing = [p for p in paths if not p.exists()]
    if missing:
        _print(f"ERROR: missing video(s): {missing}")
        return 1

    _print(f"Validating {len(paths)} video(s), sample_seconds={args.sample_seconds}")
    _print("")

    all_results = []
    for idx, p in enumerate(paths, 1):
        label = p.name
        _print(f"[{idx}/{len(paths)}] {label}")
        try:
            result = validate_one(str(p), sample_seconds=args.sample_seconds)
        except Exception as e:  # noqa: BLE001
            _print(f"  ERROR: {type(e).__name__}: {e}")
            all_results.append({"video": str(p), "error": f"{type(e).__name__}: {e}"})
            continue

        m = result["metrics"]
        _print(f"  load={result['load_time_s']}s")
        _print(
            f"  duration={m.get('durationS')}s "
            f"res={int(m.get('width', 0))}x{int(m.get('height', 0))} "
            f"fps={m.get('fps')}"
        )
        _print(
            f"  court:      confidence={m.get('courtConfidence')} "
            f"(block<{result['thresholds']['MIN_COURT_CONFIDENCE']})"
        )
        if result["issues"]:
            _print("  FIRED:")
            for i in result["issues"]:
                _print(
                    f"    - [{i['tier'].upper()}] {i['id']} "
                    f"severity={i['severity']} data={i['data']}"
                )
        else:
            _print("  FIRED: (none)")
        _print("")
        all_results.append(result)

    if args.output:
        args.output.write_text(json.dumps(all_results, indent=2))
        _print(f"Wrote {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
