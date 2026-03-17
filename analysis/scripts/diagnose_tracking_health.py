#!/usr/bin/env python3
"""Diagnose player tracking health across all processed rallies.

Audits stored tracking data for systemic issues: missing players (<4),
non-player distractors, low coverage, and quality report flags.

Usage:
    uv run python scripts/diagnose_tracking_health.py --all
    uv run python scripts/diagnose_tracking_health.py --video <video-id>
    uv run python scripts/diagnose_tracking_health.py --all --severity critical
    uv run python scripts/diagnose_tracking_health.py --all --patterns-only
"""

import argparse
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rallycut.evaluation.tracking.db import get_connection

# --- Severity levels ---

CRITICAL = "CRITICAL"
WARNING = "WARNING"
INFO = "INFO"
OK = "OK"

SEVERITY_ORDER = {CRITICAL: 0, WARNING: 1, INFO: 2, OK: 3}


@dataclass
class TrackDiagnostic:
    """Diagnostic info for a single track within a rally."""

    track_id: int
    frame_count: int
    total_frames: int
    coverage: float  # frames present / total frames
    mean_area: float
    mean_height: float
    mean_x: float
    position_spread: float
    issues: list[str] = field(default_factory=list)


@dataclass
class RallyDiagnostic:
    """Diagnostic result for a single rally."""

    rally_id: str
    video_id: str
    filename: str
    start_ms: int
    end_ms: int
    severity: str
    issues: list[str] = field(default_factory=list)
    player_count: int = 0
    trackability_score: float | None = None
    primary_track_count: int | None = None
    id_switch_count: int | None = None
    unique_raw_track_count: int | None = None
    frame_count: int = 0
    track_diagnostics: list[TrackDiagnostic] = field(default_factory=list)


def load_tracked_rallies(video_id: str | None = None) -> list[dict[str, Any]]:
    """Load rallies with tracking data from DB."""
    query = """
        SELECT
            r.id, r.video_id, v.filename, r.start_ms, r.end_ms,
            pt.positions_json, pt.raw_positions_json, pt.primary_track_ids,
            pt.quality_report_json, pt.fps, pt.frame_count, pt.ball_positions_json
        FROM rallies r
        JOIN player_tracks pt ON pt.rally_id = r.id
        JOIN videos v ON v.id = r.video_id
        WHERE pt.status = 'COMPLETED' AND pt.positions_json IS NOT NULL
          AND r.rejection_reason IS NULL
    """
    params: list[Any] = []
    if video_id:
        query += " AND r.video_id = %s"
        params.append(video_id)
    query += " ORDER BY v.filename, r.start_ms"

    results = []
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            cols = [
                "rally_id", "video_id", "filename", "start_ms", "end_ms",
                "positions_json", "raw_positions_json", "primary_track_ids",
                "quality_report_json", "fps", "frame_count", "ball_positions_json",
            ]
            for row in cur.fetchall():
                results.append(dict(zip(cols, row)))
    return results


def analyze_tracks(
    positions: list[dict[str, Any]], total_frames: int
) -> list[TrackDiagnostic]:
    """Analyze individual tracks for suspicious patterns."""
    # Group by track_id
    tracks: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for p in positions:
        tracks[p["trackId"]].append(p)

    diagnostics = []
    for track_id, track_positions in tracks.items():
        frames_present = len({p["frameNumber"] for p in track_positions})
        coverage = frames_present / total_frames if total_frames > 0 else 0.0

        areas = [p["width"] * p["height"] for p in track_positions]
        heights = [p["height"] for p in track_positions]
        xs = [p["x"] for p in track_positions]
        ys = [p["y"] for p in track_positions]

        mean_area = float(np.mean(areas))
        mean_height = float(np.mean(heights))
        mean_x = float(np.mean(xs))
        x_spread = float(np.std(xs))
        y_spread = float(np.std(ys))
        position_spread = float(np.sqrt(x_spread**2 + y_spread**2))

        issues: list[str] = []
        if mean_area < 0.003:
            issues.append(f"tiny_bbox(area={mean_area:.4f})")
        if mean_height < 0.08:
            issues.append(f"short(h={mean_height:.3f})")
        if position_spread < 0.010:
            issues.append(f"stationary(spread={position_spread:.4f})")
        if mean_x < 0.08 or mean_x > 0.92:
            issues.append(f"edge(x={mean_x:.3f})")
        if coverage < 0.20:
            issues.append(f"short_lived(cov={coverage:.1%})")

        diagnostics.append(TrackDiagnostic(
            track_id=track_id,
            frame_count=frames_present,
            total_frames=total_frames,
            coverage=coverage,
            mean_area=mean_area,
            mean_height=mean_height,
            mean_x=mean_x,
            position_spread=position_spread,
            issues=issues,
        ))
    return diagnostics


def diagnose_rally(rally: dict[str, Any]) -> RallyDiagnostic:
    """Run full diagnosis on a single rally."""
    positions = rally["positions_json"] or []
    raw_positions = rally["raw_positions_json"]
    quality_report = rally["quality_report_json"]
    frame_count = rally["frame_count"] or 0

    # Compute total frames from positions if not stored
    if frame_count == 0 and positions:
        frames = {p["frameNumber"] for p in positions}
        frame_count = max(frames) - min(frames) + 1 if frames else 1

    issues: list[str] = []

    # 1. Player count from primary_track_ids (authoritative, not inflated by ID switches)
    primary_ids = rally["primary_track_ids"] or []
    player_count = len(set(primary_ids))
    # Also track unique IDs in positions (may differ due to ID switches)
    unique_position_tracks = len({p["trackId"] for p in positions})

    if player_count < 3:
        issues.append(f"PLAYER_COUNT={player_count} (<3)")
    elif player_count == 3:
        issues.append("PLAYER_COUNT=3 (missing 1)")
    elif player_count > 4:
        issues.append(f"PLAYER_COUNT={player_count} (>4)")

    if unique_position_tracks > player_count:
        extra = unique_position_tracks - player_count
        issues.append(f"ID_SWITCH_FRAGMENTS={extra} extra trackIds in positions")

    # 2. Per-player frame coverage
    track_frame_counts: dict[int, int] = Counter()
    for p in positions:
        track_frame_counts[p["trackId"]] += 1
    # Deduplicate: count unique frames per track
    track_frames: dict[int, set[int]] = defaultdict(set)
    for p in positions:
        track_frames[p["trackId"]].add(p["frameNumber"])

    for tid, frames_set in track_frames.items():
        cov = len(frames_set) / frame_count if frame_count > 0 else 0
        if cov < 0.30:
            issues.append(f"LOW_COVERAGE track {tid}: {cov:.0%}")
        elif cov < 0.50:
            issues.append(f"MODERATE_COVERAGE track {tid}: {cov:.0%}")

    # 3. Per-frame player count histogram
    if positions:
        frame_player_counts: dict[int, int] = Counter()
        for p in positions:
            frame_player_counts[p["frameNumber"]] += 1
        counts = list(frame_player_counts.values())
        total_counted_frames = len(counts)
        under_4 = sum(1 for c in counts if c < 4)
        pct_under_4 = under_4 / total_counted_frames if total_counted_frames else 0
        if pct_under_4 > 0.30:
            issues.append(f"FRAME_DROPOUT {pct_under_4:.0%} frames <4 players")

    # 4. Suspicious tracks from raw positions
    analysis_positions = raw_positions if raw_positions else positions
    track_diags = analyze_tracks(analysis_positions, frame_count)
    suspicious = [t for t in track_diags if t.issues]
    for t in suspicious:
        issues.append(f"SUSPICIOUS track {t.track_id}: {', '.join(t.issues)}")

    # 5. Quality report flags
    trackability_score = None
    primary_track_count_stored = None
    id_switch_count = None
    unique_raw_track_count = None

    if quality_report:
        qr = quality_report if isinstance(quality_report, dict) else {}
        trackability_score = qr.get("trackabilityScore")
        primary_track_count_stored = qr.get("primaryTrackCount")
        id_switch_count = qr.get("idSwitchCount")
        unique_raw_track_count = qr.get("uniqueRawTrackCount")

        if trackability_score is not None and trackability_score < 0.5:
            issues.append(f"TRACKABILITY_CRITICAL={trackability_score:.2f}")
        elif trackability_score is not None and trackability_score < 0.7:
            issues.append(f"TRACKABILITY_LOW={trackability_score:.2f}")

        if primary_track_count_stored is not None and primary_track_count_stored != 4:
            issues.append(f"PRIMARY_COUNT={primary_track_count_stored}")

        if id_switch_count is not None and id_switch_count > 3:
            issues.append(f"ID_SWITCHES={id_switch_count}")

        if unique_raw_track_count is not None and unique_raw_track_count > 12:
            issues.append(f"MANY_RAW_TRACKS={unique_raw_track_count}")

    # Determine severity
    severity = OK
    if player_count < 3 or (trackability_score is not None and trackability_score < 0.5):
        severity = CRITICAL
    elif (
        player_count == 3
        or any("LOW_COVERAGE" in i for i in issues)
        or (trackability_score is not None and trackability_score < 0.7)
    ):
        severity = WARNING
    elif issues:
        severity = INFO

    return RallyDiagnostic(
        rally_id=rally["rally_id"],
        video_id=rally["video_id"],
        filename=rally["filename"] or "?",
        start_ms=rally["start_ms"],
        end_ms=rally["end_ms"],
        severity=severity,
        issues=issues,
        player_count=player_count,
        trackability_score=trackability_score,
        primary_track_count=primary_track_count_stored,
        id_switch_count=id_switch_count,
        unique_raw_track_count=unique_raw_track_count,
        frame_count=frame_count,
        track_diagnostics=track_diags,
    )


def format_timestamp(ms: int) -> str:
    """Format milliseconds as MM:SS."""
    s = ms // 1000
    return f"{s // 60}:{s % 60:02d}"


def print_video_summary(diagnostics: list[RallyDiagnostic]) -> None:
    """Print per-video summary table."""
    # Group by video
    videos: dict[str, list[RallyDiagnostic]] = defaultdict(list)
    for d in diagnostics:
        videos[d.filename].append(d)

    print("\n=== Per-Video Summary ===\n")
    print(
        f"{'Video':<28} {'Rallies':>7} {'CRIT':>5} {'WARN':>5} "
        f"{'INFO':>5} {'OK':>5} {'Trackability':>12}"
    )
    print("-" * 78)

    for filename in sorted(videos.keys()):
        rallies = videos[filename]
        counts = Counter(d.severity for d in rallies)
        scores = [d.trackability_score for d in rallies if d.trackability_score is not None]
        mean_track = f"{np.mean(scores):.2f}" if scores else "n/a"
        print(
            f"{filename[:28]:<28} {len(rallies):>7} "
            f"{counts.get(CRITICAL, 0):>5} {counts.get(WARNING, 0):>5} "
            f"{counts.get(INFO, 0):>5} {counts.get(OK, 0):>5} "
            f"{mean_track:>12}"
        )

    print()


def print_detailed_issues(
    diagnostics: list[RallyDiagnostic], severity_filter: str | None = None
) -> None:
    """Print detailed issues sorted by severity."""
    filtered = diagnostics
    if severity_filter:
        sev = severity_filter.upper()
        filtered = [d for d in diagnostics if d.severity == sev]

    # Sort by severity then filename
    filtered.sort(key=lambda d: (SEVERITY_ORDER.get(d.severity, 99), d.filename, d.start_ms))

    # Skip OK unless explicitly filtered
    if not severity_filter:
        filtered = [d for d in filtered if d.severity != OK]

    if not filtered:
        print("\nNo issues found matching filter.\n")
        return

    print(f"\n=== Detailed Issues ({len(filtered)} rallies) ===\n")

    current_severity = None
    for d in filtered:
        if d.severity != current_severity:
            current_severity = d.severity
            print(f"\n--- {current_severity} ---\n")

        time_range = f"{format_timestamp(d.start_ms)}-{format_timestamp(d.end_ms)}"
        track_str = f"trk={d.trackability_score:.2f}" if d.trackability_score is not None else ""
        print(
            f"  {d.filename[:25]:<25} {time_range:<12} "
            f"players={d.player_count} {track_str}"
        )
        print(f"  rally={d.rally_id}")
        for issue in d.issues:
            print(f"    - {issue}")
        print()


def print_pattern_analysis(diagnostics: list[RallyDiagnostic]) -> None:
    """Print aggregate pattern analysis."""
    print("\n=== Pattern Analysis ===\n")

    total = len(diagnostics)
    if total == 0:
        print("No rallies to analyze.\n")
        return

    # Severity breakdown
    sev_counts = Counter(d.severity for d in diagnostics)
    print(f"Total rallies: {total}")
    for sev in [CRITICAL, WARNING, INFO, OK]:
        count = sev_counts.get(sev, 0)
        print(f"  {sev:>10}: {count:>4} ({count * 100 / total:.1f}%)")
    print()

    # Issue category breakdown
    issue_categories: Counter[str] = Counter()
    for d in diagnostics:
        for issue in d.issues:
            # Extract category (first word before = or space)
            cat = issue.split("=")[0].split(" ")[0]
            issue_categories[cat] += 1

    if issue_categories:
        print("Issue categories:")
        for cat, count in issue_categories.most_common(20):
            print(f"  {cat:<30} {count:>4} ({count * 100 / total:.1f}% of rallies)")
        print()

    # Player count distribution
    player_counts = Counter(d.player_count for d in diagnostics)
    print("Player count distribution:")
    for pc in sorted(player_counts.keys()):
        count = player_counts[pc]
        print(f"  {pc} players: {count:>4} ({count * 100 / total:.1f}%)")
    print()

    # Trackability score distribution
    scores = [d.trackability_score for d in diagnostics if d.trackability_score is not None]
    if scores:
        print(f"Trackability scores (n={len(scores)}):")
        print(f"  mean={np.mean(scores):.3f}  median={np.median(scores):.3f}  "
              f"min={min(scores):.3f}  max={max(scores):.3f}")
        buckets = [
            ("<0.5", sum(1 for s in scores if s < 0.5)),
            ("0.5-0.7", sum(1 for s in scores if 0.5 <= s < 0.7)),
            ("0.7-0.9", sum(1 for s in scores if 0.7 <= s < 0.9)),
            (">=0.9", sum(1 for s in scores if s >= 0.9)),
        ]
        for label, count in buckets:
            print(f"  {label:>6}: {count:>4} ({count * 100 / len(scores):.1f}%)")
        print()

    # Rally duration vs issues
    durations_ok = [(d.end_ms - d.start_ms) / 1000 for d in diagnostics if d.severity == OK]
    durations_bad = [
        (d.end_ms - d.start_ms) / 1000
        for d in diagnostics
        if d.severity in (CRITICAL, WARNING)
    ]
    if durations_ok and durations_bad:
        print(f"Duration (seconds) — OK rallies: mean={np.mean(durations_ok):.1f}s, "
              f"problem rallies: mean={np.mean(durations_bad):.1f}s")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose player tracking health")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Analyze all tracked rallies")
    group.add_argument("--video", type=str, help="Analyze specific video by ID")
    parser.add_argument(
        "--severity", type=str, choices=["critical", "warning", "info", "ok"],
        help="Filter to specific severity level",
    )
    parser.add_argument(
        "--patterns-only", action="store_true",
        help="Only show pattern analysis (no per-rally details)",
    )
    args = parser.parse_args()

    video_id = args.video
    print("Loading tracked rallies from DB...")
    rallies = load_tracked_rallies(video_id=video_id)
    print(f"Found {len(rallies)} tracked rallies\n")

    if not rallies:
        print("No rallies found.")
        sys.exit(0)

    # Run diagnosis
    diagnostics = []
    for i, rally in enumerate(rallies):
        if (i + 1) % 50 == 0:
            print(f"  Analyzing {i + 1}/{len(rallies)}...")
        diagnostics.append(diagnose_rally(rally))

    # Output
    print_video_summary(diagnostics)

    if not args.patterns_only:
        print_detailed_issues(diagnostics, severity_filter=args.severity)

    print_pattern_analysis(diagnostics)


if __name__ == "__main__":
    main()
