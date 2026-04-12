"""Phase 0: Formation predictor failure analysis.

Categorizes every formation-based serving_team error into actionable buckets:

  1. ABSTENTION  — formation had some signal but below margin (ratio in [1.0, margin))
  2. TEAM_MAP    — formation picked the correct SIDE but A/B label is inverted
                   (flipping the prediction matches GT)
  3. TRUE_ERROR  — formation picked the wrong side entirely
  4. DATA_QUAL   — not enough tracked players or no court split

Also reports per-video breakdown and aggregate statistics to determine
Phase 1 priorities (see score-tracking-investigation-design.md).

Read-only. No DB writes. No production changes.

Usage:
    cd analysis
    uv run python scripts/diagnose_formation_errors.py
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rallycut.evaluation.tracking.db import get_connection  # noqa: E402
from rallycut.tracking.action_classifier import (  # noqa: E402
    _compute_auto_split_y,
    _find_serving_team_by_formation,
)
from rallycut.tracking.player_tracker import PlayerPosition  # noqa: E402

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class RallyRecord:
    rally_id: str
    video_id: str
    start_ms: int
    gt_serving_team: str  # "A" or "B"
    positions: list[PlayerPosition]
    ball_positions: list[dict]
    court_split_y: float | None
    fps: float
    rally_index: int = 0
    side_flipped: bool = False
    # Filled during analysis:
    team_assignments: dict[int, int] | None = None
    track_to_player: dict[int, int] | None = None
    semantic_flip: bool = False


@dataclass
class FormationDiag:
    """Intermediate formation predictor values for one rally."""
    rally_id: str
    video_id: str
    gt_serving_team: str
    predicted_team: str | None
    confidence: float
    # Internals
    n_tracks: int = 0
    near_sep: float = 0.0
    far_sep: float = 0.0
    ratio: float = 0.0  # max(near,far) / max(min(near,far), 1e-6)
    n_near: int = 0
    n_far: int = 0
    used_auto_split: bool = False
    effective_split_y: float = 0.0
    # Error categorization
    bucket: str = ""  # CORRECT, ABSTENTION, TEAM_MAP, TRUE_ERROR, DATA_QUAL
    flip_would_fix: bool = False


# ---------------------------------------------------------------------------
# Data loading (mirrors production_eval patterns)
# ---------------------------------------------------------------------------

def _load_rallies() -> dict[str, list[RallyRecord]]:
    """Load all rallies with gt_serving_team, positions, and side-switch info."""
    # Load side switches per video
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT id, player_matching_gt_json FROM videos
            WHERE id IN (SELECT DISTINCT video_id FROM rallies WHERE gt_serving_team IS NOT NULL)
        """)
        video_switches: dict[str, set[int]] = {}
        for row in cur.fetchall():
            vid_str = str(row[0])
            gt = row[1]
            sw = list(gt.get("sideSwitches", [])) if isinstance(gt, dict) else []
            video_switches[vid_str] = set(sw)

    # Load rally data
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT r.id, r.video_id, r.start_ms, r.gt_serving_team,
                   pt.positions_json, pt.ball_positions_json, pt.court_split_y, pt.fps
            FROM rallies r LEFT JOIN player_tracks pt ON pt.rally_id = r.id
            WHERE r.video_id IN (SELECT DISTINCT video_id FROM rallies WHERE gt_serving_team IS NOT NULL)
              AND r.gt_serving_team IS NOT NULL
            ORDER BY r.video_id, r.start_ms
        """)
        raw: dict[str, list[tuple]] = defaultdict(list)
        for row in cur.fetchall():
            video_id_val = str(row[1])
            raw[video_id_val].append(row)

    out: dict[str, list[RallyRecord]] = {}
    for vid, rows in raw.items():
        rows.sort(key=lambda r: r[2])
        switches = video_switches.get(vid, set())
        flipped = False
        vid_out: list[RallyRecord] = []
        for idx, (rid, _, sms, gt, pj, bpj, split_y, fps) in enumerate(rows):
            if idx in switches:
                flipped = not flipped
            positions = _parse_positions(pj or [])
            vid_out.append(RallyRecord(
                rally_id=rid,
                video_id=vid,
                start_ms=sms or 0,
                gt_serving_team=gt,
                positions=positions,
                ball_positions=bpj or [],
                court_split_y=split_y,
                fps=fps or 30.0,
                rally_index=idx,
                side_flipped=flipped,
            ))
        out[vid] = vid_out
    return out


def _parse_positions(raw: list[dict]) -> list[PlayerPosition]:
    return [
        PlayerPosition(
            frame_number=pp["frameNumber"],
            track_id=pp["trackId"],
            x=pp["x"],
            y=pp["y"],
            width=pp.get("width", 0.05),
            height=pp.get("height", 0.10),
            confidence=pp.get("confidence", 1.0),
            keypoints=pp.get("keypoints"),
        )
        for pp in raw
    ]


def _load_formation_semantic_flips(video_ids: set[str]) -> dict[str, bool]:
    """Per-rally semantic flip from match_analysis_json."""
    if not video_ids:
        return {}
    placeholders = ", ".join(["%s"] * len(video_ids))
    query = f"""
        SELECT id, match_analysis_json FROM videos
        WHERE id IN ({placeholders}) AND match_analysis_json IS NOT NULL
    """
    result: dict[str, bool] = {}
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(query, list(video_ids))
        rows = cur.fetchall()
    for _vid, ma_json in rows:
        if not isinstance(ma_json, dict):
            continue
        rally_entries = ma_json.get("rallies") or []
        if not isinstance(rally_entries, list):
            continue
        count = 0
        for entry in rally_entries:
            rid = entry.get("rallyId") or entry.get("rally_id")
            if rid:
                result[rid] = (count % 2 == 1)
            if entry.get("sideSwitchDetected") or entry.get("side_switch_detected"):
                count += 1
    return result


def _load_match_team_assignments(
    video_ids: set[str],
    rally_positions: dict[str, list[PlayerPosition]] | None = None,
) -> dict[str, dict[int, int]]:
    """Load match-level team assignments."""
    from rallycut.tracking.match_tracker import build_match_team_assignments

    if not video_ids:
        return {}
    placeholders = ", ".join(["%s"] * len(video_ids))
    query = f"""
        SELECT id, match_analysis_json FROM videos
        WHERE id IN ({placeholders}) AND match_analysis_json IS NOT NULL
    """
    result: dict[str, dict[int, int]] = {}
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(query, list(video_ids))
        rows = cur.fetchall()
    for _vid, ma_json in rows:
        if not isinstance(ma_json, dict):
            continue
        result.update(build_match_team_assignments(
            ma_json, 0.70, rally_positions=rally_positions,
        ))
    return result


def _load_track_to_player_maps(video_ids: set[str]) -> dict[str, dict[int, int]]:
    """Load track_to_player maps from match_analysis_json."""
    if not video_ids:
        return {}
    placeholders = ", ".join(["%s"] * len(video_ids))
    query = f"""
        SELECT id, match_analysis_json FROM videos
        WHERE id IN ({placeholders}) AND match_analysis_json IS NOT NULL
    """
    result: dict[str, dict[int, int]] = {}
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(query, list(video_ids))
        rows = cur.fetchall()
    for _vid, ma_json in rows:
        if not isinstance(ma_json, dict):
            continue
        for entry in ma_json.get("rallies", []):
            rid = entry.get("rallyId") or entry.get("rally_id", "")
            t2p = entry.get("trackToPlayer") or entry.get("track_to_player", {})
            if rid and t2p:
                result[rid] = {int(k): int(v) for k, v in t2p.items()}
    return result


# ---------------------------------------------------------------------------
# Instrumented formation analysis
# ---------------------------------------------------------------------------

def _analyze_formation(
    rally: RallyRecord,
    margin: float = 1.15,
    window_frames: int = 120,
) -> FormationDiag:
    """Run formation predictor and capture intermediate diagnostics."""
    diag = FormationDiag(
        rally_id=rally.rally_id,
        video_id=rally.video_id,
        gt_serving_team=rally.gt_serving_team,
        predicted_team=None,
        confidence=0.0,
    )

    positions = rally.positions
    if not positions:
        diag.bucket = "DATA_QUAL"
        return diag

    # Reproduce formation internals
    start_frame = 0
    end_frame = start_frame + window_frames
    by_track: dict[int, list[float]] = defaultdict(list)
    for p in positions:
        if p.track_id < 0:
            continue
        if start_frame <= p.frame_number < end_frame:
            by_track[p.track_id].append(p.y + p.height / 2.0)

    diag.n_tracks = len(by_track)
    if len(by_track) < 2:
        diag.bucket = "DATA_QUAL"
        return diag

    # Determine effective split
    net_y = rally.court_split_y or 0.5
    effective_split = net_y
    track_medians = {tid: sum(ys) / len(ys) for tid, ys in by_track.items()}
    near_count = sum(1 for y in track_medians.values() if y > effective_split)
    far_count = len(track_medians) - near_count

    if near_count == 0 or far_count == 0:
        auto_split = _compute_auto_split_y(positions)
        if auto_split is None:
            diag.bucket = "DATA_QUAL"
            return diag
        effective_split = auto_split
        diag.used_auto_split = True

    diag.effective_split_y = effective_split

    # Split tracks
    near_tids: list[int] = []
    far_tids: list[int] = []
    for tid, med_y in track_medians.items():
        if med_y > effective_split:
            near_tids.append(tid)
        else:
            far_tids.append(tid)

    diag.n_near = len(near_tids)
    diag.n_far = len(far_tids)

    if not near_tids or not far_tids:
        diag.bucket = "DATA_QUAL"
        return diag

    # Compute separations
    def _sep(tids: list[int]) -> float:
        if len(tids) >= 2:
            ys = [track_medians[t] for t in tids]
            return max(ys) - min(ys)
        return abs(track_medians[tids[0]] - effective_split) * 0.5

    diag.near_sep = _sep(near_tids)
    diag.far_sep = _sep(far_tids)
    diag.ratio = max(diag.near_sep, diag.far_sep) / max(min(diag.near_sep, diag.far_sep), 1e-6)

    # Now call the actual function to get the production prediction
    pred_team, pred_conf = _find_serving_team_by_formation(
        positions,
        start_frame=0,
        net_y=net_y,
        team_assignments=rally.team_assignments,
        track_to_player=rally.track_to_player,
        semantic_flip=rally.semantic_flip,
        window_frames=window_frames,
        margin=margin,
    )
    diag.predicted_team = pred_team
    diag.confidence = pred_conf

    # Categorize
    if pred_team is None:
        # Check if it's separation-based abstention or data issue
        if diag.ratio >= 1.0:
            diag.bucket = "ABSTENTION"
        else:
            diag.bucket = "DATA_QUAL"
    elif pred_team == rally.gt_serving_team:
        diag.bucket = "CORRECT"
    else:
        # Wrong prediction — is it a team mapping error (flip would fix)?
        flipped_team = "B" if pred_team == "A" else "A"
        if flipped_team == rally.gt_serving_team:
            diag.flip_would_fix = True
            diag.bucket = "TEAM_MAP"
        else:
            diag.bucket = "TRUE_ERROR"

    return diag


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 70)
    print("Phase 0: Formation Predictor Failure Analysis")
    print("=" * 70)

    print("\nLoading rallies with gt_serving_team...")
    video_rallies = _load_rallies()
    total_rallies = sum(len(v) for v in video_rallies.values())
    print(f"  {len(video_rallies)} videos, {total_rallies} rallies")

    video_ids = set(video_rallies.keys())

    print("Loading team assignments...")
    rally_pos_lookup: dict[str, list[PlayerPosition]] = {}
    for rallies in video_rallies.values():
        for r in rallies:
            if r.positions:
                rally_pos_lookup[r.rally_id] = r.positions
    team_map = _load_match_team_assignments(video_ids, rally_positions=rally_pos_lookup)

    print("Loading track-to-player maps...")
    t2p_map = _load_track_to_player_maps(video_ids)

    print("Loading formation semantic flips...")
    flip_map = _load_formation_semantic_flips(video_ids)

    # Enrich rally records with team data
    for rallies in video_rallies.values():
        for r in rallies:
            r.team_assignments = team_map.get(r.rally_id)
            r.track_to_player = t2p_map.get(r.rally_id)
            r.semantic_flip = flip_map.get(r.rally_id, False)

    # Run instrumented analysis
    print("\nAnalyzing formation predictions...\n")
    all_diags: list[FormationDiag] = []
    for rallies in video_rallies.values():
        for r in rallies:
            diag = _analyze_formation(r)
            all_diags.append(diag)

    # --------------- Summary table ---------------
    bucket_counts = Counter(d.bucket for d in all_diags)
    n_correct = bucket_counts.get("CORRECT", 0)
    n_errors = total_rallies - n_correct

    print("=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    print(f"  Total rallies:  {total_rallies}")
    print(f"  Correct:        {n_correct} ({n_correct / total_rallies * 100:.1f}%)")
    print(f"  Errors:         {n_errors} ({n_errors / total_rallies * 100:.1f}%)")
    print()

    print("ERROR BREAKDOWN:")
    for bucket in ["ABSTENTION", "TEAM_MAP", "TRUE_ERROR", "DATA_QUAL"]:
        count = bucket_counts.get(bucket, 0)
        pct = count / total_rallies * 100 if total_rallies else 0
        of_errors = count / n_errors * 100 if n_errors else 0
        print(f"  {bucket:12s}  {count:4d} ({pct:5.1f}% of total, {of_errors:5.1f}% of errors)")

    # --------------- Abstention detail ---------------
    abstentions = [d for d in all_diags if d.bucket == "ABSTENTION"]
    if abstentions:
        print(f"\nABSTENTION DETAIL ({len(abstentions)} rallies):")
        ratios = [d.ratio for d in abstentions]
        ratios.sort()
        print(f"  Separation ratio range: {min(ratios):.3f} - {max(ratios):.3f}")
        print(f"  Median ratio: {ratios[len(ratios) // 2]:.3f}")
        # How many would be rescued at lower margins?
        for threshold in [1.10, 1.05, 1.00]:
            rescued = sum(1 for d in abstentions if d.ratio >= threshold)
            print(f"  Rescued at margin={threshold:.2f}: {rescued}/{len(abstentions)}")

    # --------------- Team mapping detail ---------------
    team_maps = [d for d in all_diags if d.bucket == "TEAM_MAP"]
    if team_maps:
        print(f"\nTEAM MAPPING ERRORS ({len(team_maps)} rallies):")
        # Check if they cluster by video
        tm_by_video = Counter(d.video_id for d in team_maps)
        print("  Per-video distribution:")
        for vid, cnt in tm_by_video.most_common():
            vid_total = len(video_rallies[vid])
            print(f"    {vid[:8]}: {cnt}/{vid_total} ({cnt / vid_total * 100:.0f}%)")

    # --------------- True error detail ---------------
    true_errs = [d for d in all_diags if d.bucket == "TRUE_ERROR"]
    if true_errs:
        print(f"\nTRUE FORMATION ERRORS ({len(true_errs)} rallies):")
        ratios = [d.ratio for d in true_errs]
        print(f"  Separation ratio range: {min(ratios):.3f} - {max(ratios):.3f}")
        print(f"  Median ratio: {ratios[len(ratios) // 2]:.3f}")
        # Sub-categorize
        high_ratio = sum(1 for d in true_errs if d.ratio > 1.5)
        low_ratio = sum(1 for d in true_errs if d.ratio <= 1.5)
        print(f"  High confidence wrong (ratio > 1.5): {high_ratio}")
        print(f"  Low confidence wrong (ratio <= 1.5): {low_ratio}")

    # --------------- Per-video breakdown ---------------
    print(f"\n{'=' * 70}")
    print("PER-VIDEO BREAKDOWN (sorted by error rate)")
    print(f"{'=' * 70}")
    print(f"  {'video':>10s}  {'total':>5s}  {'correct':>7s}  {'abstain':>7s}  "
          f"{'tm_map':>6s}  {'true_err':>8s}  {'data_q':>6s}  {'acc':>6s}")
    print(f"  {'-' * 10}  {'-' * 5}  {'-' * 7}  {'-' * 7}  "
          f"{'-' * 6}  {'-' * 8}  {'-' * 6}  {'-' * 6}")

    video_stats: list[tuple[str, int, dict[str, int]]] = []
    for vid, rallies in video_rallies.items():
        vid_diags = [d for d in all_diags if d.video_id == vid]
        vid_buckets = Counter(d.bucket for d in vid_diags)
        video_stats.append((vid, len(rallies), vid_buckets))

    # Sort by accuracy (ascending = worst first)
    video_stats.sort(key=lambda x: x[2].get("CORRECT", 0) / max(x[1], 1))

    for vid, n, buckets in video_stats:
        correct = buckets.get("CORRECT", 0)
        acc = correct / n * 100 if n else 0
        print(f"  {vid[:10]}  {n:5d}  {correct:7d}  "
              f"{buckets.get('ABSTENTION', 0):7d}  "
              f"{buckets.get('TEAM_MAP', 0):6d}  "
              f"{buckets.get('TRUE_ERROR', 0):8d}  "
              f"{buckets.get('DATA_QUAL', 0):6d}  "
              f"{acc:5.1f}%")

    # --------------- Decision matrix ---------------
    print(f"\n{'=' * 70}")
    print("PHASE 1 PRIORITY RECOMMENDATION")
    print(f"{'=' * 70}")

    dominant = max(
        ["ABSTENTION", "TEAM_MAP", "TRUE_ERROR", "DATA_QUAL"],
        key=lambda b: bucket_counts.get(b, 0),
    )
    print(f"\n  Dominant error bucket: {dominant} "
          f"({bucket_counts.get(dominant, 0)}/{n_errors} = "
          f"{bucket_counts.get(dominant, 0) / max(n_errors, 1) * 100:.0f}% of errors)")

    if dominant == "ABSTENTION":
        print("  → Priority: graduated confidence + fallback signals (Phase 1a, 1b)")
        print("    Many rallies have formation signal but below margin threshold.")
        print("    Converting these from abstention to soft predictions will directly")
        print("    improve coverage and feed better data into the Viterbi.")
    elif dominant == "TEAM_MAP":
        print("  → Priority: fix semantic_flip / team assignment logic (Phase 1d)")
        print("    The formation correctly identifies the serving SIDE but the A/B")
        print("    label mapping is wrong. Fix the side-switch tracking or anchor")
        print("    from the first high-confidence prediction per video.")
    elif dominant == "TRUE_ERROR":
        print("  → Priority: new independent signals (Phase 1b, 1c)")
        print("    The formation signal itself is wrong — new signals like ball")
        print("    direction and server position are needed for disambiguation.")
    else:
        print("  → Priority: upstream tracking / court detection improvements")
        print("    Data quality issues (missing players, no court split) dominate.")

    # Also report multi-bucket recommendation
    top2 = sorted(
        ["ABSTENTION", "TEAM_MAP", "TRUE_ERROR", "DATA_QUAL"],
        key=lambda b: bucket_counts.get(b, 0),
        reverse=True,
    )[:2]
    if bucket_counts.get(top2[1], 0) > 0.3 * n_errors:
        print(f"\n  Secondary focus: {top2[1]} "
              f"({bucket_counts.get(top2[1], 0)}/{n_errors} = "
              f"{bucket_counts.get(top2[1], 0) / max(n_errors, 1) * 100:.0f}% of errors)")

    # --------------- Per-rally detail for debugging ---------------
    print(f"\n{'=' * 70}")
    print("SAMPLE ERRORS (first 10 of each type)")
    print(f"{'=' * 70}")
    for bucket in ["ABSTENTION", "TEAM_MAP", "TRUE_ERROR", "DATA_QUAL"]:
        errors = [d for d in all_diags if d.bucket == bucket][:10]
        if errors:
            print(f"\n  {bucket}:")
            for d in errors:
                print(f"    {d.rally_id[:8]} vid={d.video_id[:8]} "
                      f"gt={d.gt_serving_team} pred={d.predicted_team} "
                      f"near_sep={d.near_sep:.4f} far_sep={d.far_sep:.4f} "
                      f"ratio={d.ratio:.2f} n_tracks={d.n_tracks} "
                      f"n_near={d.n_near} n_far={d.n_far} "
                      f"auto_split={d.used_auto_split}")

    # --------------- Oracle comparison: GT flips vs automated flips ---------------
    print(f"\n{'=' * 70}")
    print("ORACLE COMPARISON: GT side_flipped vs automated semantic_flip")
    print(f"{'=' * 70}")

    # Re-run with GT side_flipped instead of automated semantic_flip
    gt_correct = 0
    auto_correct = 0
    flip_disagreements = 0
    for rallies in video_rallies.values():
        for r in rallies:
            if not r.positions or not r.team_assignments:
                continue
            net_y = r.court_split_y or 0.5
            # Automated flip (current production)
            pred_auto, _ = _find_serving_team_by_formation(
                r.positions, start_frame=0, net_y=net_y,
                team_assignments=r.team_assignments,
                track_to_player=r.track_to_player,
                semantic_flip=r.semantic_flip,
            )
            # GT flip
            pred_gt, _ = _find_serving_team_by_formation(
                r.positions, start_frame=0, net_y=net_y,
                team_assignments=r.team_assignments,
                track_to_player=r.track_to_player,
                semantic_flip=r.side_flipped,
            )
            if pred_auto == r.gt_serving_team:
                auto_correct += 1
            if pred_gt == r.gt_serving_team:
                gt_correct += 1
            if r.semantic_flip != r.side_flipped:
                flip_disagreements += 1

    predicted = sum(
        1 for rallies in video_rallies.values()
        for r in rallies if r.team_assignments
    )
    print(f"  Rallies with team_assignments: {predicted}")
    print(f"  Flip disagreements (auto vs GT): {flip_disagreements}")
    print(f"  Automated semantic_flip accuracy: {auto_correct}/{predicted} "
          f"= {auto_correct / max(predicted, 1) * 100:.1f}%")
    print(f"  GT side_flipped accuracy:        {gt_correct}/{predicted} "
          f"= {gt_correct / max(predicted, 1) * 100:.1f}%")
    print(f"  Delta (GT - auto):               +{gt_correct - auto_correct} rallies "
          f"(+{(gt_correct - auto_correct) / max(predicted, 1) * 100:.1f}pp)")

    # Per-video: check if entire video is inverted (all team_map errors would be fixed
    # by flipping the whole video)
    print(f"\n{'=' * 70}")
    print("PER-VIDEO FLIP ORACLE: would flipping entire video fix team_map errors?")
    print(f"{'=' * 70}")
    for vid, rallies in sorted(video_rallies.items()):
        vid_diags = [d for d in all_diags if d.video_id == vid]
        n_team_map = sum(1 for d in vid_diags if d.bucket == "TEAM_MAP")
        if n_team_map == 0:
            continue
        # Count: how many predicted rallies would be correct if we globally flipped A/B?
        n_pred = sum(1 for d in vid_diags if d.predicted_team is not None)
        n_correct_now = sum(1 for d in vid_diags if d.bucket == "CORRECT")
        n_correct_flipped = sum(
            1 for d in vid_diags
            if d.predicted_team is not None
            and ("B" if d.predicted_team == "A" else "A") == d.gt_serving_team
        )
        print(f"  {vid[:10]}: team_map={n_team_map:2d}  "
              f"correct_now={n_correct_now:2d}/{n_pred}  "
              f"correct_if_flipped={n_correct_flipped:2d}/{n_pred}  "
              f"{'← FLIP HELPS' if n_correct_flipped > n_correct_now else ''}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
