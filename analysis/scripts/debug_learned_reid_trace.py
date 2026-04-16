"""Debug: trace a single-rally learned-ReID retrack to see whether the cost
path actually activates the learned term.

Usage:
    WEIGHT_LEARNED_REID=0.20 uv run python scripts/debug_learned_reid_trace.py \\
        --rally-id fad29c31-6e2a-4a8d-86f1-9064b2f1f425
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import numpy as np  # noqa: F401  (torch-adjacent)

ANALYSIS = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rally-id", required=True)
    args = parser.parse_args()

    # Must be set BEFORE importing global_identity
    weight = os.environ.get("WEIGHT_LEARNED_REID", "0.0")
    print(f"[TRACE] WEIGHT_LEARNED_REID env={weight}")

    import rallycut.tracking.global_identity as gi
    from rallycut.cli.commands.evaluate_tracking import _compute_tracker_config_hash

    print(f"[TRACE] gi.WEIGHT_LEARNED_REID module constant = {gi.WEIGHT_LEARNED_REID}")

    h = _compute_tracker_config_hash()
    print(f"[TRACE] config_hash = {h}")

    prefix = hashlib.sha256(f"{args.rally_id}:{h}".encode()).hexdigest()[:16]
    cache_dir = Path.home() / ".cache" / "rallycut" / "retrack_cache"
    meta = cache_dir / f"{prefix}_meta.json"
    if not meta.exists():
        print(f"[ERROR] cache miss: {meta}")
        return 1
    print(f"[TRACE] cache hit: {meta}")

    # Load cache via the production path
    from rallycut.evaluation.tracking.retrack_cache import RetrackCache
    rc = RetrackCache()
    entry = rc.get(args.rally_id, h)
    if entry is None:
        print("[ERROR] RetrackCache.get returned None")
        return 2
    cached_data, color_store, appearance_store, learned_store = entry
    print(
        f"[TRACE] color_store.has_data={color_store.has_data()} "
        f"appearance.has_data={appearance_store.has_data()} "
        f"learned_store is None: {learned_store is None} "
        f"learned_store.has_data={'-' if learned_store is None else learned_store.has_data()}"
    )
    if learned_store is not None:
        print(
            f"[TRACE] learned_store.track_ids() = {sorted(learned_store.track_ids())[:15]} "
            f"... ({len(learned_store.track_ids())} total)"
        )

    # Monkey-patch global_identity functions to trace activation
    orig_cost = gi._compute_assignment_cost
    orig_mean = gi._get_segment_mean_embedding
    cost_stats = {
        "n_calls": 0,
        "n_learned_active": 0,
        "learned_contributions": [],
    }
    mean_stats = {"valid": 0, "none_empty": 0, "none_no_store": 0}

    def traced_cost(seg, prof, *a,
                    seg_learned_emb=None, profile_learned_emb=None, **kw):
        cost_stats["n_calls"] += 1
        c = orig_cost(seg, prof, *a,
                      seg_learned_emb=seg_learned_emb,
                      profile_learned_emb=profile_learned_emb, **kw)
        if seg_learned_emb is not None and profile_learned_emb is not None:
            cost_stats["n_learned_active"] += 1
            import numpy as _np
            dist = 1.0 - float(_np.dot(seg_learned_emb, profile_learned_emb))
            cost_stats["learned_contributions"].append(
                gi.WEIGHT_LEARNED_REID * dist
            )
        return c

    def traced_mean(seg, store):
        if store is None or not store.has_data():
            mean_stats["none_no_store"] += 1
            return None
        v = orig_mean(seg, store)
        if v is None:
            mean_stats["none_empty"] += 1
        else:
            mean_stats["valid"] += 1
        return v

    gi._compute_assignment_cost = traced_cost
    gi._get_segment_mean_embedding = traced_mean

    # Replay the full post-processing on cached data
    from rallycut.tracking.player_filter import PlayerFilterConfig
    from rallycut.tracking.player_tracker import PlayerTracker
    filter_config = PlayerFilterConfig().scaled_for_fps(cached_data.video_fps)
    tracking_result = PlayerTracker.apply_post_processing(
        positions=cached_data.positions,
        raw_positions=list(cached_data.positions),
        color_store=color_store,
        appearance_store=appearance_store,
        ball_positions=cached_data.ball_positions,
        video_fps=cached_data.video_fps,
        video_width=cached_data.video_width,
        video_height=cached_data.video_height,
        frame_count=cached_data.frame_count,
        start_frame=0,
        filter_enabled=True,
        filter_config=filter_config,
        learned_store=learned_store,
    )

    print(f"[TRACE] _compute_assignment_cost calls: {cost_stats['n_calls']}")
    print(f"[TRACE] with learned embeddings active: {cost_stats['n_learned_active']}")
    if cost_stats["learned_contributions"]:
        import statistics
        c = cost_stats["learned_contributions"]
        print(
            f"[TRACE] learned cost contribution — "
            f"mean={statistics.mean(c):.4f} "
            f"min={min(c):.4f} max={max(c):.4f} "
            f"std={statistics.stdev(c) if len(c)>1 else 0:.4f}"
        )
    print(f"[TRACE] _get_segment_mean_embedding stats: {mean_stats}")
    print(f"[TRACE] tracking_result positions: {len(tracking_result.positions)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
