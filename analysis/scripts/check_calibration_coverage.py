"""V1: calibration coverage across labelled tracking GT rallies.

Counts how many rallies with tracking ground truth have court calibration
available — the court-plane velocity gate needs calibration to apply.
Rallies without calibration fall back to the image-plane path.

Usage:
    uv run python scripts/check_calibration_coverage.py
"""

from __future__ import annotations

import logging
from collections import Counter

from rallycut.evaluation.tracking.db import load_labeled_rallies

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("coverage")


def main() -> None:
    rallies = load_labeled_rallies()
    total = len(rallies)
    by_video: dict[str, list[bool]] = {}
    for r in rallies:
        cal = r.court_calibration_json
        has = bool(cal and isinstance(cal, list) and len(cal) == 4)
        by_video.setdefault(r.video_id, []).append(has)

    cal_count = sum(1 for r in rallies if r.court_calibration_json
                    and isinstance(r.court_calibration_json, list)
                    and len(r.court_calibration_json) == 4)
    pct = 100.0 * cal_count / total if total else 0.0
    logger.info(f"Total labelled rallies: {total}")
    logger.info(f"Calibrated: {cal_count} ({pct:.1f}%)")
    logger.info(f"Uncalibrated: {total - cal_count} ({100.0 - pct:.1f}%)")

    video_cal: Counter[str] = Counter()
    for vid, flags in by_video.items():
        video_cal["cal" if all(flags) else ("mixed" if any(flags) else "none")] += 1
    logger.info(f"Per-video: {dict(video_cal)}")

    logger.info("\nPer-video detail (video_id, n_rallies, n_calibrated):")
    for vid in sorted(by_video.keys()):
        flags = by_video[vid]
        n_cal = sum(flags)
        logger.info(f"  {vid[:8]}  {len(flags):2d} rallies  {n_cal:2d} calibrated")


if __name__ == "__main__":
    main()
