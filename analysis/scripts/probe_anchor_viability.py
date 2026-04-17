"""Probe: within-team cosine separation across 43 GT rallies.

For each rally, computes the median learned-ReID embedding per primary
track, then cosine similarity between same-team pairs. Assesses whether
"anchor-based identity propagation" is viable: if enough rallies have
clear within-team separation, the best rally per video can anchor
identity across the match.

Usage:
    uv run python scripts/probe_anchor_viability.py
"""

from __future__ import annotations

import copy as _copy
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Ensure WEIGHT_LEARNED_REID > 0 so config_hash matches the cache built
# during Sessions 4-8 (which stored learned embeddings).
os.environ.setdefault("WEIGHT_LEARNED_REID", "0.05")
# Disable optional passes that might not be relevant here
os.environ.setdefault("ENABLE_OCCLUSION_RESOLVER", "0")

ANALYSIS_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ANALYSIS_ROOT / "reports" / "merge_veto"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
logger = logging.getLogger("probe_anchor_viability")


@dataclass
class RallyRow:
    rally_id: str
    video_id: str
    team0_pair: tuple[int, int]
    team1_pair: tuple[int, int]
    team0_cos: float | None
    team1_cos: float | None


def _load_post_processed_state(rally_id: str):
    """Load retrack cache + run apply_post_processing.

    Uses the monkey-patch trick from eval_occlusion_resolver.py so the
    learned_store is mutated in-place and its IDs match the post-processed
    primary track IDs.

    Returns:
        (result, learned_store) or None if not cached.
    """
    from rallycut.cli.commands.evaluate_tracking import _compute_tracker_config_hash
    from rallycut.evaluation.tracking.retrack_cache import RetrackCache
    from rallycut.tracking.player_filter import PlayerFilterConfig
    from rallycut.tracking.player_tracker import PlayerTracker

    rc = RetrackCache()
    entry = rc.get(rally_id, _compute_tracker_config_hash())
    if entry is None:
        return None

    cached_data, color, app, learned = entry
    if learned is None:
        return None

    filter_config = PlayerFilterConfig().scaled_for_fps(cached_data.video_fps)

    # Monkey-patch deepcopy so apply_post_processing's internal "deep copy"
    # of LearnedEmbeddingStore/ColorHistogramStore is a no-op. This means
    # the remap_ids calls inside post-processing mutate our local `learned`
    # reference directly, giving us the post-processed IDs after the call.
    original_deepcopy = _copy.deepcopy

    def _shallow_when_store(obj, memo=None):  # type: ignore[no-untyped-def]
        from rallycut.tracking.appearance_descriptor import AppearanceDescriptorStore
        from rallycut.tracking.color_repair import ColorHistogramStore, LearnedEmbeddingStore
        if isinstance(obj, (ColorHistogramStore, AppearanceDescriptorStore, LearnedEmbeddingStore)):
            return obj
        return original_deepcopy(obj, memo) if memo is not None else original_deepcopy(obj)

    _copy.deepcopy = _shallow_when_store  # type: ignore[assignment]
    try:
        result = PlayerTracker.apply_post_processing(
            positions=cached_data.positions,
            raw_positions=list(cached_data.positions),
            color_store=color,
            appearance_store=app,
            ball_positions=cached_data.ball_positions,
            video_fps=cached_data.video_fps,
            video_width=cached_data.video_width,
            video_height=cached_data.video_height,
            frame_count=cached_data.frame_count,
            start_frame=0,
            filter_enabled=True,
            filter_config=filter_config,
            learned_store=learned,
        )
    finally:
        _copy.deepcopy = original_deepcopy  # type: ignore[assignment]

    return result, learned


def _frames_for_track(result, track_id: int) -> set[int]:
    """Collect all frame numbers where track_id appears in result.positions."""
    frames: set[int] = set()
    for fp in result.positions:
        if fp.track_id == track_id:
            frames.add(fp.frame_number)
    return frames


def _cosine_pair(
    learned_store,
    tid_a: int,
    frames_a: set[int],
    tid_b: int,
    frames_b: set[int],
) -> float | None:
    """Return cosine similarity between two tracks' median embeddings, or None."""
    from rallycut.tracking.merge_veto import _segment_median_embedding

    emb_a = _segment_median_embedding(learned_store, tid_a, frames_a)
    emb_b = _segment_median_embedding(learned_store, tid_b, frames_b)
    if emb_a is None or emb_b is None:
        return None
    return float(np.dot(emb_a, emb_b))


def _process_rally(rally_id: str, video_id: str, idx: int, total: int) -> RallyRow | None:
    """Process a single rally: load, post-process, compute same-team cosines."""
    state = _load_post_processed_state(rally_id)
    if state is None:
        print(f"[{idx}/{total}] {rally_id[:8]}: SKIP (no cache or no learned store)")
        return None

    result, learned = state

    primary = result.primary_track_ids or []
    teams = result.team_assignments or {}

    if len(primary) < 4:
        print(f"[{idx}/{total}] {rally_id[:8]}: SKIP (only {len(primary)} primary tracks)")
        return None

    # Group primary tracks by team
    team0 = [t for t in primary if teams.get(t) == 0]
    team1 = [t for t in primary if teams.get(t) == 1]

    if len(team0) < 2 or len(team1) < 2:
        print(
            f"[{idx}/{total}] {rally_id[:8]}: SKIP "
            f"(team0={len(team0)} team1={len(team1)} primaries)"
        )
        return None

    # Use first two tracks per team (typically exactly 2 after filtering)
    t0a, t0b = team0[0], team0[1]
    t1a, t1b = team1[0], team1[1]

    frames_t0a = _frames_for_track(result, t0a)
    frames_t0b = _frames_for_track(result, t0b)
    frames_t1a = _frames_for_track(result, t1a)
    frames_t1b = _frames_for_track(result, t1b)

    cos0 = _cosine_pair(learned, t0a, frames_t0a, t0b, frames_t0b)
    cos1 = _cosine_pair(learned, t1a, frames_t1a, t1b, frames_t1b)

    cos0_str = f"{cos0:.3f}" if cos0 is not None else "N/A"
    cos1_str = f"{cos1:.3f}" if cos1 is not None else "N/A"
    print(f"[{idx}/{total}] {rally_id[:8]}: team0 cos={cos0_str}, team1 cos={cos1_str}")

    return RallyRow(
        rally_id=rally_id,
        video_id=video_id,
        team0_pair=(t0a, t0b),
        team1_pair=(t1a, t1b),
        team0_cos=cos0,
        team1_cos=cos1,
    )


def _render_report(rows: list[RallyRow], out_path: Path) -> None:
    # Collect all non-None cosine values (one per team-pair)
    cosines: list[float] = []
    for row in rows:
        if row.team0_cos is not None:
            cosines.append(row.team0_cos)
        if row.team1_cos is not None:
            cosines.append(row.team1_cos)

    n_pairs = len(cosines)
    if n_pairs == 0:
        out_path.write_text("# Anchor Viability Probe\n\nNo valid pairs found.\n")
        return

    arr = np.array(cosines, dtype=np.float32)
    mean_cos = float(np.mean(arr))
    median_cos = float(np.median(arr))
    p25_cos = float(np.percentile(arr, 25))
    p75_cos = float(np.percentile(arr, 75))

    # Per-rally stats: a rally has "clear separation" if ANY pair < 0.5
    # and "ambiguous" if ALL pairs > 0.7
    clear_sep_rallies = 0
    ambiguous_rallies = 0
    for row in rows:
        pair_cosines = [c for c in [row.team0_cos, row.team1_cos] if c is not None]
        if not pair_cosines:
            continue
        if any(c < 0.5 for c in pair_cosines):
            clear_sep_rallies += 1
        if all(c > 0.7 for c in pair_cosines):
            ambiguous_rallies += 1

    n_rallies = len(rows)

    # Per-VIDEO best anchor: for each video, lowest within-team cosine across all rallies
    by_video: dict[str, list[RallyRow]] = defaultdict(list)
    for row in rows:
        by_video[row.video_id].append(row)

    video_best: dict[str, float | None] = {}
    for vid, vrows in by_video.items():
        mins: list[float] = []
        for row in vrows:
            pair_cosines = [c for c in [row.team0_cos, row.team1_cos] if c is not None]
            if pair_cosines:
                mins.append(min(pair_cosines))
        video_best[vid] = min(mins) if mins else None

    n_videos = len(by_video)
    n_viable_videos = sum(1 for v in video_best.values() if v is not None and v < 0.5)
    pct_viable = (n_viable_videos / n_videos * 100) if n_videos > 0 else 0.0

    if pct_viable >= 70:
        verdict = "**VIABLE** — ≥70% of videos have a usable anchor rally (cosine < 0.5)."
    elif pct_viable >= 30:
        verdict = "**MARGINAL** — 30-70% of videos have a usable anchor rally (cosine < 0.5)."
    else:
        verdict = "**NOT VIABLE** — <30% of videos have a usable anchor rally (cosine < 0.5)."

    lines = [
        "# Anchor Viability Probe",
        "",
        "Measures within-team cosine similarity of Session-3 learned-ReID embeddings",
        "across 43 GT rallies. A low cosine between teammates means the head can distinguish",
        "them — a prerequisite for anchor-based identity propagation.",
        "",
        "## Distribution of within-team cosine similarities",
        "",
        f"- **N pairs**: {n_pairs} (from {n_rallies} rallies)",
        f"- **Mean**: {mean_cos:.3f}",
        f"- **Median**: {median_cos:.3f}",
        f"- **P25**: {p25_cos:.3f}",
        f"- **P75**: {p75_cos:.3f}",
        "",
        "## Separation counts",
        "",
        f"- **Clear separation** (≥1 pair with cosine < 0.5): **{clear_sep_rallies} / {n_rallies} rallies** "
        f"({clear_sep_rallies / n_rallies * 100:.0f}%)",
        f"- **Ambiguous** (all pairs with cosine > 0.7): **{ambiguous_rallies} / {n_rallies} rallies** "
        f"({ambiguous_rallies / n_rallies * 100:.0f}%)",
        "",
        "## Per-video best anchor",
        "",
        f"For each of {n_videos} videos, the best (lowest cosine) rally across all rallies in that video.",
        "",
        "| Video (short) | Best cosine | Viable? |",
        "|:--------------|:------------|:-------:|",
    ]

    for vid, best in sorted(video_best.items(), key=lambda kv: (kv[1] is None, kv[1] or 1.0)):
        if best is None:
            lines.append(f"| `{vid[:8]}` | N/A | — |")
        else:
            viable = "yes" if best < 0.5 else "no"
            lines.append(f"| `{vid[:8]}` | {best:.3f} | {viable} |")

    lines += [
        "",
        f"**Videos with viable anchor (best cosine < 0.5): {n_viable_videos} / {n_videos} ({pct_viable:.0f}%)**",
        "",
        "## Verdict",
        "",
        verdict,
        "",
        "## Per-rally detail",
        "",
        "| Rally (short) | Video (short) | Team-0 cos | Team-1 cos |",
        "|:--------------|:--------------|:----------:|:----------:|",
    ]

    for row in rows:
        cos0 = f"{row.team0_cos:.3f}" if row.team0_cos is not None else "N/A"
        cos1 = f"{row.team1_cos:.3f}" if row.team1_cos is not None else "N/A"
        lines.append(f"| `{row.rally_id[:8]}` | `{row.video_id[:8]}` | {cos0} | {cos1} |")

    out_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    from rallycut.evaluation.tracking.db import load_labeled_rallies

    rallies = load_labeled_rallies()
    total = len(rallies)
    print(f"Loaded {total} GT rallies")

    rows: list[RallyRow] = []
    for i, rally in enumerate(rallies, 1):
        row = _process_rally(rally.rally_id, rally.video_id, i, total)
        if row is not None:
            rows.append(row)

    print(f"\nProcessed {len(rows)} / {total} rallies successfully")

    # Summary stats
    cosines: list[float] = []
    for row in rows:
        if row.team0_cos is not None:
            cosines.append(row.team0_cos)
        if row.team1_cos is not None:
            cosines.append(row.team1_cos)

    if cosines:
        arr = np.array(cosines)
        print(f"\n--- Distribution ---")
        print(f"  N pairs:  {len(cosines)}")
        print(f"  Mean:     {np.mean(arr):.3f}")
        print(f"  Median:   {np.median(arr):.3f}")
        print(f"  P25:      {np.percentile(arr, 25):.3f}")
        print(f"  P75:      {np.percentile(arr, 75):.3f}")
        print(f"  Min:      {np.min(arr):.3f}")
        print(f"  Max:      {np.max(arr):.3f}")

        n_clear = sum(
            1 for row in rows
            if any(c < 0.5 for c in [row.team0_cos, row.team1_cos] if c is not None)
        )
        n_ambig = sum(
            1 for row in rows
            if all(c > 0.7 for c in [row.team0_cos, row.team1_cos] if c is not None)
        )
        by_video: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            for c in [row.team0_cos, row.team1_cos]:
                if c is not None:
                    by_video[row.video_id].append(c)
        video_bests = {v: min(cs) for v, cs in by_video.items()}
        n_viable = sum(1 for b in video_bests.values() if b < 0.5)
        n_vids = len(video_bests)
        pct = n_viable / n_vids * 100 if n_vids > 0 else 0.0

        print(f"\n--- Separation ---")
        print(f"  Clear (any pair cos<0.5): {n_clear} / {len(rows)} rallies")
        print(f"  Ambiguous (all pairs cos>0.7): {n_ambig} / {len(rows)} rallies")
        print(f"\n--- Per-video anchor coverage ---")
        print(f"  Viable videos (best cos<0.5): {n_viable} / {n_vids} ({pct:.0f}%)")
        if pct >= 70:
            print("  VERDICT: VIABLE")
        elif pct >= 30:
            print("  VERDICT: MARGINAL")
        else:
            print("  VERDICT: NOT VIABLE")

    out_path = OUT_DIR / "anchor_viability_probe.md"
    _render_report(rows, out_path)
    print(f"\nReport written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
