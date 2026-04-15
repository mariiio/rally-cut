r"""Calibrate the beach-VB classifier threshold.

Runs open-clip on the 5 negative fixtures + positive examples, reports
per-video beach-VB probability, and selects a threshold where:

    threshold = max(lowest_positive_score - 0.10, 0.30)

Ships only if `lowest_positive - highest_negative >= 0.15`.

Usage:
    cd analysis
    uv run python scripts/calibrate_beach_vb.py \
        --negatives ~/Desktop/rallies/Negative/bad\ angle.mp4 \
        --negatives ~/Desktop/rallies/Negative/very\ bad\ angle.mp4 \
        --negatives ~/Desktop/rallies/Negative/indoor\ 1.mp4 \
        --negatives ~/Desktop/rallies/Negative/indoor\ 2.mp4 \
        --negatives ~/Desktop/rallies/Negative/not\ related.mp4 \
        --positives ~/Desktop/rallies/Matches/match.mp4 \
        --positives "~/Desktop/rallies/Matches/Newport Beach Games 10_12_25 Paul_Nate vs Colin_Tim - Nathan Gali (1080p, h264).mp4"

Writes analysis/reports/beach_vb_calibration_<date>.json.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from datetime import date
from pathlib import Path


def _extract_frames(video_path: Path, n_frames: int = 5) -> list[Path]:
    """Sample n_frames evenly across the video via ffmpeg."""
    out_dir = Path(tempfile.mkdtemp(prefix="beach_vb_calib_"))
    # ffprobe to get duration
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True, text=True, check=True,
    )
    duration = float(probe.stdout.strip())
    for i in range(n_frames):
        ts = ((i + 0.5) / n_frames) * duration
        out = out_dir / f"frame_{i:02d}.jpg"
        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error",
                "-ss", f"{ts:.2f}",
                "-i", str(video_path),
                "-frames:v", "1",
                "-q:v", "2",
                str(out),
            ],
            check=True,
        )
    frames = sorted(out_dir.glob("*.jpg"))
    if not frames:
        raise RuntimeError(f"ffmpeg produced no frames for {video_path}")
    return frames


def _score_video(video_path: Path) -> float:
    """Return the average beach-VB prob for video_path."""
    from PIL import Image

    from rallycut.quality.beach_vb_classifier import embed_and_score_frames

    frames = _extract_frames(video_path, n_frames=5)
    images = [Image.open(p).convert("RGB") for p in frames]
    probs = embed_and_score_frames(images)
    return sum(probs) / len(probs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--negatives", action="append", required=True, type=Path)
    parser.add_argument("--positives", action="append", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports") / f"beach_vb_calibration_{date.today().isoformat()}.json",
    )
    args = parser.parse_args()

    print(f"[calibrate] scoring {len(args.negatives)} negatives, {len(args.positives)} positives", flush=True)
    scores: dict[str, float] = {}
    for v in args.negatives + args.positives:
        print(f"  scoring {v.name} ...", flush=True)
        scores[str(v)] = _score_video(v)
        print(f"    = {scores[str(v)]:.3f}", flush=True)

    neg_scores = [scores[str(v)] for v in args.negatives]
    pos_scores = [scores[str(v)] for v in args.positives]
    highest_neg = max(neg_scores)
    lowest_pos = min(pos_scores)
    gap = lowest_pos - highest_neg
    threshold = max(lowest_pos - 0.10, 0.30)

    ships = gap >= 0.15
    report = {
        "date": date.today().isoformat(),
        "per_video": scores,
        "highest_negative": highest_neg,
        "lowest_positive": lowest_pos,
        "gap": gap,
        "chosen_threshold": threshold,
        "ship_gate_passed": ships,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"[calibrate] wrote {args.output}", flush=True)
    print(f"  highest_neg={highest_neg:.3f}  lowest_pos={lowest_pos:.3f}  gap={gap:.3f}", flush=True)
    print(f"  chosen_threshold={threshold:.3f}  SHIPS={ships}", flush=True)

    return 0 if ships else 1


if __name__ == "__main__":
    raise SystemExit(main())
