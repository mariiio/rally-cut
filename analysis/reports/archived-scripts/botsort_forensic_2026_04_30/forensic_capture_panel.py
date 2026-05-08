"""Forensic capture: run BoxMOT BoT-SORT on the 9-error / 4-control panel
with read-only instrumentation enabled.

For each rally in the panel, sets the env vars that activate
`InstrumentedBotSort` (subclass that captures per-frame BoxMOT internal
state to a JSONL sidecar) and re-runs the production track_video pathway.
The sidecar lands at:

  analysis/reports/botsort_forensic_2026_04_30/<short_id>/r<NN>.jsonl

No DB writes. No production behavior change. The instrumentation subclass
replicates BotSort's body verbatim and only adds JSONL writes; tracker
outputs are byte-identical to the parent class.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

_ANALYSIS_DIR = Path(__file__).resolve().parents[1]
if str(_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS_DIR))

from rallycut.evaluation.tracking.db import get_connection, get_video_path  # noqa: E402

# Panel mirrors analyze_panel_tracker_origin.py:53. rally_idx is 0-indexed
# (rally_idx=1 corresponds to "r02" in the verdict UI).
PANEL = [
    ("7d77980f-3006-40e0-adc0-db491a5bb659", 1, True, "p1 starts on p4's track (occluded), settles mid"),
    ("7d77980f-3006-40e0-adc0-db491a5bb659", 12, True, "p2 ↔ p4 swap (white vs dark)"),
    ("7d77980f-3006-40e0-adc0-db491a5bb659", 18, True, "p2 ↔ p4 swap (same shape as r13)"),
    ("7d77980f-3006-40e0-adc0-db491a5bb659", 0, False, "control"),
    ("b5fb0594-d64f-4a0d-bad9-de8fc36414d0", 9, True, "full 4-cycle"),
    ("b5fb0594-d64f-4a0d-bad9-de8fc36414d0", 3, True, "p3 ↔ p4 within team B"),
    ("b5fb0594-d64f-4a0d-bad9-de8fc36414d0", 5, True, "p3 ↔ p4 distinguishable"),
    ("b5fb0594-d64f-4a0d-bad9-de8fc36414d0", 0, False, "control"),
    ("5c756c41-1cc1-4486-a95c-97398912cfbe", 6, True, "p1 ↔ p4 cross-team (wawa)"),
    ("5c756c41-1cc1-4486-a95c-97398912cfbe", 2, True, "p2 ↔ p4 start swap (wawa)"),
    ("5c756c41-1cc1-4486-a95c-97398912cfbe", 0, False, "control"),
    ("854bb250-3e91-47d2-944d-f62413e3cf45", 0, True, "p2 ↔ p4 NOT visually similar"),
    ("854bb250-3e91-47d2-944d-f62413e3cf45", 1, False, "control"),
]

OUT_DIR = _ANALYSIS_DIR / "reports" / "botsort_forensic_2026_04_30"


@dataclass
class RallyInfo:
    rally_id: str
    video_id: str
    rally_idx_0based: int
    short_id: str
    rally_tag: str
    is_error: bool
    desc: str
    start_ms: int
    end_ms: int
    calibration_json: list[dict[str, float]] | None
    ball_positions_json: list[dict[str, Any]] | None


def resolve_panel_to_rally_ids() -> list[RallyInfo]:
    """Resolve each (video_id, rally_idx_0based) panel entry to a concrete
    rally_id + start/end ms + calibration + ball positions, by querying
    rallies ordered by start_ms (matching analyze_panel_tracker_origin.py).
    """
    rallies: list[RallyInfo] = []
    with get_connection() as conn, conn.cursor() as cur:
        for video_id, rally_idx, is_error, desc in PANEL:
            cur.execute(
                """
                SELECT r.id::text, r.start_ms, r.end_ms,
                       v.court_calibration_json, pt.ball_positions_json
                FROM rallies r
                JOIN player_tracks pt ON pt.rally_id = r.id
                JOIN videos v ON v.id = r.video_id
                WHERE r.video_id = %s
                  AND pt.positions_json IS NOT NULL
                ORDER BY r.start_ms
                """,
                [video_id],
            )
            rows = cur.fetchall()
            if rally_idx >= len(rows):
                print(
                    f"  ! {video_id[:8]} idx={rally_idx} "
                    f"out of range ({len(rows)} rallies); skipping",
                    flush=True,
                )
                continue
            rally_id, start_ms, end_ms, cal_json, ball_json = rows[rally_idx]
            short_id = video_id[:8]
            rally_tag = f"{short_id}/r{rally_idx + 1:02d}"
            rallies.append(RallyInfo(
                rally_id=cast(str, rally_id),
                video_id=video_id,
                rally_idx_0based=rally_idx,
                short_id=short_id,
                rally_tag=rally_tag,
                is_error=is_error,
                desc=desc,
                start_ms=int(cast(Any, start_ms)),
                end_ms=int(cast(Any, end_ms)),
                calibration_json=cast(
                    "list[dict[str, float]] | None", cal_json,
                ),
                ball_positions_json=cast(
                    "list[dict[str, Any]] | None", ball_json,
                ),
            ))
    return rallies


def _build_court_calibrator(
    cal_json: list[dict[str, float]] | None,
) -> Any:
    """Construct CourtCalibrator from rally's court_calibration_json (4 corners)."""
    if not cal_json or not isinstance(cal_json, list) or len(cal_json) != 4:
        return None
    try:
        from rallycut.court.calibration import CourtCalibrator
        cal = CourtCalibrator()
        cal.calibrate([(c["x"], c["y"]) for c in cal_json])
        return cal if cal.is_calibrated else None
    except Exception:
        return None


def _build_ball_positions(
    ball_json: list[dict[str, Any]] | None,
) -> list[Any] | None:
    """Convert DB ball positions JSON into BallPosition list (for filter)."""
    if not ball_json:
        return None
    from rallycut.tracking.ball_tracker import BallPosition
    out: list[Any] = []
    for b in ball_json:
        try:
            out.append(BallPosition(
                frame_number=int(b["frameNumber"]),
                x=float(b["x"]),
                y=float(b["y"]),
                confidence=float(b.get("confidence", 1.0)),
            ))
        except Exception:
            continue
    return out or None


def run_one_rally(rally: RallyInfo) -> tuple[bool, float, str]:
    """Run forensic-instrumented retrack on one rally.

    Returns (success, elapsed_seconds, message).
    """
    sidecar_path = OUT_DIR / f"{rally.rally_tag}.jsonl"
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    if sidecar_path.exists():
        # Forensic runs are deterministic given same weights; if a sidecar
        # already exists for this rally tag we re-write it to keep the run
        # idempotent.
        sidecar_path.unlink()

    os.environ["BOTSORT_FORENSIC_LOG_DIR"] = str(OUT_DIR)
    os.environ["BOTSORT_FORENSIC_RALLY_TAG"] = rally.rally_tag

    video_path = get_video_path(rally.video_id)
    if not video_path or not video_path.exists():
        return False, 0.0, f"video unavailable for {rally.video_id[:8]}"

    from rallycut.tracking.player_filter import PlayerFilterConfig
    from rallycut.tracking.player_tracker import (
        DEFAULT_TRACKER,
        PlayerTracker,
    )

    tracker = PlayerTracker(tracker=DEFAULT_TRACKER)
    court_cal = _build_court_calibrator(rally.calibration_json)
    ball_positions = _build_ball_positions(rally.ball_positions_json)
    filter_config = PlayerFilterConfig()

    t0 = time.time()
    try:
        tracker.track_video(
            video_path,
            start_ms=rally.start_ms,
            end_ms=rally.end_ms,
            filter_enabled=True,
            filter_config=filter_config,
            ball_positions=ball_positions,
            court_calibrator=court_cal,
        )
    except Exception as exc:
        elapsed = time.time() - t0
        return False, elapsed, f"track_video failed: {exc}"
    elapsed = time.time() - t0

    # Validate sidecar exists and is non-trivial.
    if not sidecar_path.exists():
        return False, elapsed, "sidecar not written"
    size_kb = sidecar_path.stat().st_size / 1024.0
    return True, elapsed, f"ok ({size_kb:.0f} KB)"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Forensic capture output dir: {OUT_DIR}", flush=True)
    print("Resolving panel rallies...", flush=True)
    rallies = resolve_panel_to_rally_ids()
    print(f"Resolved {len(rallies)} of {len(PANEL)} panel entries", flush=True)
    print(flush=True)

    results: list[tuple[RallyInfo, bool, float, str]] = []
    for i, rally in enumerate(rallies, start=1):
        kind = "ERROR" if rally.is_error else "CONTROL"
        print(
            f"[{i}/{len(rallies)}] {rally.rally_tag} ({kind}) {rally.desc}",
            flush=True,
        )
        ok, elapsed, msg = run_one_rally(rally)
        status = "OK" if ok else "FAIL"
        print(f"  → {status} ({elapsed:.1f}s) — {msg}", flush=True)
        results.append((rally, ok, elapsed, msg))
        print(flush=True)

    print("=" * 70, flush=True)
    n_ok = sum(1 for _, ok, _, _ in results if ok)
    print(f"Done: {n_ok}/{len(results)} rallies captured", flush=True)
    for rally, ok, elapsed, msg in results:
        kind = "E" if rally.is_error else "C"
        marker = "✓" if ok else "✗"
        print(
            f"  {marker} [{kind}] {rally.rally_tag:<14}  {elapsed:>5.1f}s  {msg}",
            flush=True,
        )


if __name__ == "__main__":
    main()
