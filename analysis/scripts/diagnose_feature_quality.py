"""Diagnose cross-rally player matching feature quality.

Checks fundamentals: crop sizes, histogram fill rates, within-team vs cross-team
discriminative gaps, and whether features can distinguish "clearly different" players.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from rallycut.evaluation.tracking.db import (
    get_connection,
    get_video_path,
    load_rallies_for_video,
)
from rallycut.tracking.player_features import (
    HS_BINS,
    MIN_HIST_PIXELS,
    TrackAppearanceStats,
    _build_clothing_mask,
    _extract_hs_histogram,
    compute_track_similarity,
    extract_appearance_features,
)
from rallycut.tracking.match_tracker import extract_rally_appearances


def diagnose_video(video_id: str):
    """Run full feature quality diagnosis for a video."""
    # Get video info
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT width, height, s3_key FROM videos WHERE id = %s", [video_id]
            )
            row = cur.fetchone()
            if not row:
                print(f"Video {video_id} not found")
                return
            db_w, db_h, s3_key = row

    print(f"=== Video {video_id[:8]} ===")
    print(f"DB resolution: {db_w}x{db_h}")
    print(f"S3 key: {s3_key}")

    # Download video
    video_path = get_video_path(video_id)
    if not video_path:
        print("Could not download video")
        return

    # Check ACTUAL video resolution
    cap = cv2.VideoCapture(str(video_path))
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    print(f"Actual file resolution: {actual_w}x{actual_h} @ {fps:.1f}fps")
    if db_w and (actual_w != db_w or actual_h != db_h):
        print(f"  *** RESOLUTION MISMATCH: DB says {db_w}x{db_h}, file is {actual_w}x{actual_h} ***")

    # Load rallies
    rallies = load_rallies_for_video(video_id)
    print(f"Rallies: {len(rallies)}")

    if not rallies:
        return

    # Extract features for first few rallies and analyze
    all_track_stats: list[tuple[str, int, TrackAppearanceStats]] = []  # (rally_id, team, stats)

    for rally in rallies[:8]:
        track_stats = extract_rally_appearances(
            video_path=video_path,
            positions=rally.positions,
            primary_track_ids=rally.primary_track_ids,
            start_ms=rally.start_ms,
            end_ms=rally.end_ms,
            num_samples=12,
        )

        if not track_stats:
            continue

        print(f"\n--- Rally {rally.rally_id[:8]} ---")

        for tid, stats in track_stats.items():
            team = rally.team_assignments.get(tid, -1)
            all_track_stats.append((rally.rally_id, team, stats))

            n_features = len(stats.features)
            n_upper = sum(1 for f in stats.features if f.upper_body_hist is not None)
            n_lower = sum(1 for f in stats.features if f.lower_body_hist is not None)
            n_skin = sum(1 for f in stats.features if f.skin_tone_hsv is not None)

            # Get crop pixel sizes
            positions = [p for p in rally.positions if p.track_id == tid]
            if positions:
                avg_w = np.mean([p.width for p in positions]) * actual_w
                avg_h = np.mean([p.height for p in positions]) * actual_h
                upper_crop_h = avg_h * 0.35  # 20-55%
                lower_crop_h = avg_h * 0.28  # 50-78%
                center_w = avg_w * 0.5
            else:
                avg_w = avg_h = upper_crop_h = lower_crop_h = center_w = 0

            print(
                f"  Track {tid} (team {team}): {n_features} frames, "
                f"upper_hist={n_upper}/{n_features}, lower_hist={n_lower}/{n_features}, "
                f"skin={n_skin}/{n_features}, "
                f"bbox={avg_w:.0f}x{avg_h:.0f}px, "
                f"clothing_crop={center_w:.0f}x{upper_crop_h:.0f}px"
            )

            # Check histogram sparsity (how many bins are populated)
            if stats.avg_upper_hist is not None:
                nonzero_upper = np.count_nonzero(stats.avg_upper_hist)
                total_bins = stats.avg_upper_hist.size
                max_bin = stats.avg_upper_hist.max()
                print(f"    Upper hist: {nonzero_upper}/{total_bins} bins active, max_bin={max_bin:.3f}")

            if stats.avg_lower_hist is not None:
                nonzero_lower = np.count_nonzero(stats.avg_lower_hist)
                total_bins = stats.avg_lower_hist.size
                max_bin = stats.avg_lower_hist.max()
                print(f"    Lower hist: {nonzero_lower}/{total_bins} bins active, max_bin={max_bin:.3f}")

            if stats.avg_skin_tone_hsv:
                h, s, v = stats.avg_skin_tone_hsv
                print(f"    Skin HSV: H={h:.1f} S={s:.1f} V={v:.1f}")

            if stats.avg_dominant_color_hsv:
                h, s, v = stats.avg_dominant_color_hsv
                print(f"    Dominant color HSV: H={h:.1f} S={s:.1f} V={v:.1f}")

    # Compute ALL pairwise similarities
    print(f"\n=== Pairwise similarity analysis ({len(all_track_stats)} tracks) ===")

    same_team_costs = []
    diff_team_costs = []
    same_rally_same_team = []

    for i, (rid_a, team_a, stats_a) in enumerate(all_track_stats):
        for j, (rid_b, team_b, stats_b) in enumerate(all_track_stats):
            if j <= i:
                continue
            cost = compute_track_similarity(stats_a, stats_b)

            if team_a == team_b:
                same_team_costs.append(cost)
                if rid_a == rid_b:
                    same_rally_same_team.append((rid_a[:8], stats_a.track_id, stats_b.track_id, cost))
            else:
                diff_team_costs.append(cost)

    if same_team_costs:
        print(f"\nSame-team pairs: n={len(same_team_costs)}")
        print(f"  mean cost={np.mean(same_team_costs):.3f}, median={np.median(same_team_costs):.3f}, "
              f"min={np.min(same_team_costs):.3f}, max={np.max(same_team_costs):.3f}")

    if diff_team_costs:
        print(f"\nCross-team pairs: n={len(diff_team_costs)}")
        print(f"  mean cost={np.mean(diff_team_costs):.3f}, median={np.median(diff_team_costs):.3f}, "
              f"min={np.min(diff_team_costs):.3f}, max={np.max(diff_team_costs):.3f}")

    if same_team_costs and diff_team_costs:
        gap = np.mean(diff_team_costs) - np.mean(same_team_costs)
        print(f"\nDiscriminative gap (cross - same): {gap:.3f}")
        if gap < 0.05:
            print("  *** VERY LOW GAP: features cannot distinguish teams! ***")
        elif gap < 0.10:
            print("  *** LOW GAP: features barely distinguish teams ***")

    # Show same-rally same-team pairs (teammates within a rally)
    if same_rally_same_team:
        print(f"\nWithin-rally teammate similarities (hardest case):")
        for rid, t1, t2, cost in sorted(same_rally_same_team, key=lambda x: x[3])[:10]:
            print(f"  Rally {rid} tracks {t1} vs {t2}: cost={cost:.3f} (lower=more similar)")

    # Show cross-rally same-track similarities (should be very low cost = same person)
    # Group tracks by team within rallies
    print(f"\n=== Per-component similarity breakdown ===")
    # Take two random tracks from different teams
    if len(all_track_stats) >= 4:
        team0 = [s for _, t, s in all_track_stats if t == 0][:4]
        team1 = [s for _, t, s in all_track_stats if t == 1][:4]

        if team0 and team1:
            from rallycut.tracking.player_features import _histogram_similarity, _hsv_similarity

            def _component_str(sa, sb):  # type: ignore[no-untyped-def]
                lower = _histogram_similarity(sa.avg_lower_hist, sb.avg_lower_hist)
                upper = _histogram_similarity(sa.avg_upper_hist, sb.avg_upper_hist)
                lower_v = _histogram_similarity(sa.avg_lower_v_hist, sb.avg_lower_v_hist)
                skin = _hsv_similarity(sa.avg_skin_tone_hsv, sb.avg_skin_tone_hsv) if sa.avg_skin_tone_hsv and sb.avg_skin_tone_hsv else None
                dc = _hsv_similarity(sa.avg_dominant_color_hsv, sb.avg_dominant_color_hsv) if sa.avg_dominant_color_hsv and sb.avg_dominant_color_hsv else None

                def _fmt(v: float | None) -> str:
                    return f"{v:.3f}" if v is not None else "N/A"

                return (f"lower={_fmt(lower)}, upper={_fmt(upper)}, "
                        f"lower_v={_fmt(lower_v)}, skin={_fmt(skin)}, "
                        f"dc={_fmt(dc)}")

            print("\nSame-team (team 0, first 4 tracks):")
            for i, sa in enumerate(team0):
                for j, sb in enumerate(team0):
                    if j <= i:
                        continue
                    print(f"  t{sa.track_id} vs t{sb.track_id}: {_component_str(sa, sb)}")

            print("\nCross-team (team 0[0] vs team 1 tracks):")
            sa = team0[0]
            for sb in team1[:4]:
                print(f"  t{sa.track_id} vs t{sb.track_id}: {_component_str(sa, sb)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Pick first video with player matching GT
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id FROM videos
                    WHERE player_matching_gt_json IS NOT NULL
                    AND width IS NOT NULL
                    ORDER BY id LIMIT 1
                """)
                row = cur.fetchone()
                if row:
                    video_id = row[0]
                else:
                    print("No videos with player matching GT found")
                    sys.exit(1)
    else:
        video_id = sys.argv[1]

    diagnose_video(video_id)
