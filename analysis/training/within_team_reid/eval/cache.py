"""Build/load DINOv2 feature cache for held-out + cross-rally evaluation.

The cache is built ONCE per training process (or shared across processes via
file). It stores per-crop L2-norm DINOv2 ViT-S/14 features so per-epoch
validation only needs to run head forward + mean-pool + L2-norm — no DINOv2,
no video reads.

**Held-out roles (mirrors probe naming exactly)**:
- query         = pred_new POST-swap crops
- anchor_correct = pred_old PRE-swap crops (same physical human as query)
- anchor_wrong   = pred_new PRE-swap crops (teammate)

**Cross-rally gallery**: per (video, rally, canonical) entry, the crops sampled
by `build_cross_rally_gallery` (probe's MAX_CROPS_PER_WINDOW=10 evenly-spread
across the rally's primary tracks, filtered by `_is_quality_crop`).
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

# Ensure analysis/scripts/ is importable so we can reuse probe helpers verbatim.
_SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from probe_reid_models_on_swaps import (  # type: ignore[import-not-found]  # noqa: E402
    MAX_CROPS_PER_WINDOW,
    MIN_VALID_CROPS_PER_ANCHOR,
    WINDOW_FRAMES,
    DINOv2Backbone,
    GalleryEntry,
    SwapEvent,
    _collect_crops_for_pred,
    _fetch_rally_context,
    _is_quality_crop,
    _load_events_from_audit,
    _read_rally_frames,
)

from rallycut.evaluation.db import get_connection  # noqa: E402
from rallycut.evaluation.tracking.db import (  # noqa: E402
    get_video_path,
    load_rallies_for_video,
)
from rallycut.tracking.player_features import extract_bbox_crop  # noqa: E402

logger = logging.getLogger("within_team_reid.eval.cache")

# ---------------------------------------------------------------------------
# Schema dataclasses
# ---------------------------------------------------------------------------


@dataclass
class HeldOutEventMeta:
    event_idx: int
    rally_id: str
    video_id: str
    swap_frame: int
    pred_old: int
    pred_new: int
    correct_player_id: int | None
    wrong_player_id: int | None
    n_correct: int
    n_wrong: int
    n_query: int
    abstain_reason: str | None  # None if event has full data; else why we couldn't probe it

    def is_scorable(self) -> bool:
        return self.abstain_reason is None


@dataclass
class CrossRallyEntryMeta:
    entry_idx: int
    video_id: str
    rally_id: str
    canonical_id: int
    n_crops: int


@dataclass
class EvalCache:
    """In-memory representation of the eval cache.

    Features arrays are float32 (N_crops, 384), L2-normalized DINOv2 features.
    """

    held_out_events: list[HeldOutEventMeta]
    held_out_features: dict[str, NDArray[np.float32]]   # key = f"e{event_idx}_{role}"
    cross_rally_entries: list[CrossRallyEntryMeta]
    cross_rally_features: dict[str, NDArray[np.float32]]  # key = f"g{entry_idx}"
    config: dict[str, Any]


# ---------------------------------------------------------------------------
# Held-out event collection
# ---------------------------------------------------------------------------


def _load_all_sorted_events(reid_debug_dir: Path) -> list[SwapEvent]:
    """Load every swap event across audit JSONs, sorted by (rally_id, swap_frame)."""
    audit_dir = reid_debug_dir.parent
    rally_ids: list[str] = []
    for p in sorted(reid_debug_dir.glob("*.json")):
        name = p.name
        if name.startswith("_") or "sota_probe" in name:
            continue
        rally_ids.append(name.removesuffix(".json"))

    events: list[SwapEvent] = []
    for rid in rally_ids:
        audit_path = audit_dir / f"{rid}.json"
        if not audit_path.exists():
            logger.warning("audit missing for rally %s — skipping", rid[:8])
            continue
        events.extend(_load_events_from_audit(audit_path))
    events.sort(key=lambda e: (e.rally_id, e.swap_frame))
    return events


def _build_held_out_event(
    event_idx: int,
    event: SwapEvent,
    backbone: DINOv2Backbone,
) -> tuple[HeldOutEventMeta, dict[str, NDArray[np.float32]]]:
    """Extract crops + DINOv2 features for one held-out event.

    Returns metadata + features dict (keys: 'e{idx}_correct/wrong/query').
    Abstains gracefully if rally context or crops are missing.
    """
    ctx = _fetch_rally_context(event.rally_id)
    if ctx is None:
        meta = HeldOutEventMeta(
            event_idx=event_idx, rally_id=event.rally_id, video_id=event.video_id,
            swap_frame=event.swap_frame, pred_old=event.pred_old, pred_new=event.pred_new,
            correct_player_id=event.correct_player_id, wrong_player_id=event.wrong_player_id,
            n_correct=0, n_wrong=0, n_query=0,
            abstain_reason="no_rally_context",
        )
        return meta, {}

    pre_range = range(max(0, event.swap_frame - WINDOW_FRAMES), event.swap_frame)
    post_range = range(event.swap_frame, event.swap_frame + WINDOW_FRAMES)

    primary_set = set(ctx.primary_track_ids) or {event.pred_old, event.pred_new}
    primary_set = primary_set | {event.pred_old, event.pred_new}

    needed_frames: set[int] = set(pre_range) | set(post_range)
    frames = _read_rally_frames(ctx.video_path, ctx.start_ms, ctx.video_fps, needed_frames)

    correct_crops = _collect_crops_for_pred(
        event.pred_old, pre_range, ctx.predictions, primary_set, frames,
        ctx.frame_width, ctx.frame_height,
    )
    wrong_crops = _collect_crops_for_pred(
        event.pred_new, pre_range, ctx.predictions, primary_set, frames,
        ctx.frame_width, ctx.frame_height,
    )
    query_crops = _collect_crops_for_pred(
        event.pred_new, post_range, ctx.predictions, primary_set, frames,
        ctx.frame_width, ctx.frame_height,
    )

    n_c, n_w, n_q = len(correct_crops), len(wrong_crops), len(query_crops)

    abstain: str | None = None
    if n_q < MIN_VALID_CROPS_PER_ANCHOR:
        abstain = f"query<{MIN_VALID_CROPS_PER_ANCHOR}"
    elif n_c < MIN_VALID_CROPS_PER_ANCHOR:
        abstain = f"correct<{MIN_VALID_CROPS_PER_ANCHOR}"
    elif n_w < MIN_VALID_CROPS_PER_ANCHOR:
        abstain = f"wrong<{MIN_VALID_CROPS_PER_ANCHOR}"

    feats: dict[str, NDArray[np.float32]] = {}
    if abstain is None:
        feats[f"e{event_idx}_correct"] = backbone.embed(correct_crops).astype(np.float32)
        feats[f"e{event_idx}_wrong"] = backbone.embed(wrong_crops).astype(np.float32)
        feats[f"e{event_idx}_query"] = backbone.embed(query_crops).astype(np.float32)

    meta = HeldOutEventMeta(
        event_idx=event_idx, rally_id=event.rally_id, video_id=event.video_id,
        swap_frame=event.swap_frame, pred_old=event.pred_old, pred_new=event.pred_new,
        correct_player_id=event.correct_player_id, wrong_player_id=event.wrong_player_id,
        n_correct=n_c, n_wrong=n_w, n_query=n_q,
        abstain_reason=abstain,
    )
    return meta, feats


# ---------------------------------------------------------------------------
# Cross-rally gallery (per-crop features mirroring build_cross_rally_gallery)
# ---------------------------------------------------------------------------


def _collect_cross_rally_entries(
    video_ids: list[str],
    backbone: DINOv2Backbone,
) -> tuple[list[CrossRallyEntryMeta], dict[str, NDArray[np.float32]]]:
    """Mirror probe build_cross_rally_gallery but store per-crop DINOv2 features.

    Walks each video, loads match_analysis + per-rally tracks, for each rally
    samples MAX_CROPS_PER_WINDOW evenly-spread frames per primary track, runs
    _is_quality_crop, extracts BGR crops, embeds via DINOv2, saves per-crop
    features keyed by entry_idx.
    """
    from rallycut.tracking.swap_reid_probe import get_rally_track_to_player

    entries: list[CrossRallyEntryMeta] = []
    features: dict[str, NDArray[np.float32]] = {}
    entry_idx = 0

    for vid in video_ids:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT match_analysis_json, width, height, fps FROM videos WHERE id = %s",
                    [vid],
                )
                row = cur.fetchone()
        if not row or not row[0]:
            continue
        match_analysis = cast(dict[str, Any], row[0])
        frame_w = int(cast(Any, row[1]) or 1920)
        frame_h = int(cast(Any, row[2]) or 1080)
        default_fps = float(cast(Any, row[3]) or 30.0)

        rally_tracks = load_rallies_for_video(vid)
        if len(rally_tracks) < 2:
            continue
        video_path = get_video_path(vid)
        if video_path is None:
            continue

        for r in rally_tracks:
            t2p = get_rally_track_to_player(match_analysis, r.rally_id)
            if not t2p:
                continue
            primary_set = set(r.primary_track_ids) or set(t2p.keys())
            canonical_to_track: dict[int, int] = {}
            for track_id, canonical in t2p.items():
                if track_id in primary_set:
                    canonical_to_track[canonical] = track_id
            if not r.positions:
                continue

            frames_per_track: dict[int, list[int]] = defaultdict(list)
            for p in r.positions:
                frames_per_track[p.track_id].append(p.frame_number)
            for fs in frames_per_track.values():
                fs.sort()

            track_frames: dict[int, list[int]] = {}
            for canonical, track_id in canonical_to_track.items():
                fs = frames_per_track.get(track_id, [])
                if len(fs) < MIN_VALID_CROPS_PER_ANCHOR:
                    continue
                idxs = np.linspace(0, len(fs) - 1, MAX_CROPS_PER_WINDOW).astype(int)
                track_frames[track_id] = [fs[i] for i in idxs]

            all_needed: set[int] = set()
            for fs in track_frames.values():
                all_needed.update(fs)
            frames_by_rf = _read_rally_frames(video_path, r.start_ms, default_fps, all_needed)

            from rallycut.tracking.player_tracker import PlayerPosition
            pos_by_key: dict[tuple[int, int], PlayerPosition] = {
                (p.track_id, p.frame_number): p for p in r.positions
            }
            primary_by_frame: dict[int, list[PlayerPosition]] = defaultdict(list)
            for p in r.positions:
                if p.track_id in primary_set:
                    primary_by_frame[p.frame_number].append(p)

            for canonical, track_id in canonical_to_track.items():
                if track_id not in track_frames:
                    continue
                selected: list[NDArray[np.uint8]] = []
                for f in track_frames[track_id]:
                    pos = pos_by_key.get((track_id, f))
                    frame = frames_by_rf.get(f)
                    if pos is None or frame is None:
                        continue
                    if not _is_quality_crop(pos, primary_by_frame.get(f, [])):
                        continue
                    crop = extract_bbox_crop(
                        frame, (pos.x, pos.y, pos.width, pos.height), frame_w, frame_h,
                    )
                    if crop is not None:
                        selected.append(crop)

                if len(selected) < MIN_VALID_CROPS_PER_ANCHOR:
                    continue

                feats = backbone.embed(selected).astype(np.float32)
                key = f"g{entry_idx}"
                features[key] = feats
                entries.append(CrossRallyEntryMeta(
                    entry_idx=entry_idx,
                    video_id=vid,
                    rally_id=r.rally_id,
                    canonical_id=canonical,
                    n_crops=len(selected),
                ))
                entry_idx += 1
            logger.info(
                "  cross-rally %s/%s: %d entries (running total %d)",
                vid[:8], r.rally_id[:8],
                sum(1 for e in entries if e.video_id == vid and e.rally_id == r.rally_id),
                len(entries),
            )

    return entries, features


# ---------------------------------------------------------------------------
# Build + persist
# ---------------------------------------------------------------------------


def build_cache(
    cache_npz: Path,
    cache_meta: Path,
    reid_debug_dir: Path,
    held_out_start: int = 34,
    held_out_end: int = 58,
    cross_rally_entries_json: Path | None = None,
) -> EvalCache:
    """Build the eval cache and write to disk.

    Args:
        cache_npz: Output .npz path for stacked features.
        cache_meta: Output JSON path for event/entry metadata.
        reid_debug_dir: Directory containing per-rally audit JSONs.
        held_out_start: 0-indexed start of held-out slice (default 34 = events 35-58).
        held_out_end: 0-indexed exclusive end (default 58).
        cross_rally_entries_json: Optional sota_probe_cross_rally.json — if given,
            cross-rally videos are taken from that file's entries; else inferred
            from held-out events' video_ids (suboptimal — gallery would be small).
    """
    cache_npz.parent.mkdir(parents=True, exist_ok=True)
    cache_meta.parent.mkdir(parents=True, exist_ok=True)

    backbone = DINOv2Backbone("dinov2_vits14")
    if not backbone.is_available():
        raise RuntimeError("DINOv2 ViT-S/14 backbone unavailable; check torch.hub cache")
    logger.info("DINOv2 ViT-S/14 status: %s", backbone.status)

    # ----- Held-out events -----
    all_events = _load_all_sorted_events(reid_debug_dir)
    logger.info("Loaded %d total swap events from %s", len(all_events), reid_debug_dir)
    held_out = all_events[held_out_start:held_out_end]
    logger.info(
        "Slicing held-out [%d:%d] = %d events", held_out_start, held_out_end, len(held_out),
    )

    held_out_metas: list[HeldOutEventMeta] = []
    held_out_feats: dict[str, NDArray[np.float32]] = {}
    for i, ev in enumerate(held_out):
        logger.info(
            "[heldout %d/%d] rally=%s frame=%d pred_old=%d pred_new=%d",
            i + 1, len(held_out), ev.rally_id[:8], ev.swap_frame,
            ev.pred_old, ev.pred_new,
        )
        meta, feats = _build_held_out_event(i, ev, backbone)
        held_out_metas.append(meta)
        held_out_feats.update(feats)
        if meta.abstain_reason:
            logger.info(
                "    abstain: %s (correct=%d wrong=%d query=%d)",
                meta.abstain_reason, meta.n_correct, meta.n_wrong, meta.n_query,
            )

    # ----- Cross-rally gallery -----
    if cross_rally_entries_json is not None and cross_rally_entries_json.exists():
        ref = json.loads(cross_rally_entries_json.read_text())
        # Take video IDs from any model entry (they share the same video set).
        any_entries: list[dict[str, Any]] = []
        for model_block in ref.values():
            if isinstance(model_block, dict) and "entries" in model_block:
                any_entries = model_block["entries"]
                break
        video_ids = sorted({e["video_id"] for e in any_entries})
    else:
        video_ids = sorted({m.video_id for m in held_out_metas if m.video_id})

    logger.info("Cross-rally: building gallery over %d videos", len(video_ids))
    cross_metas, cross_feats = _collect_cross_rally_entries(video_ids, backbone)
    logger.info("Cross-rally: %d gallery entries", len(cross_metas))

    # ----- Persist -----
    all_arrays = {**held_out_feats, **cross_feats}
    np.savez_compressed(cache_npz, **all_arrays)
    logger.info("Wrote %d arrays → %s", len(all_arrays), cache_npz)

    meta_payload: dict[str, Any] = {
        "schema_version": 1,
        "config": {
            "WINDOW_FRAMES": WINDOW_FRAMES,
            "MAX_CROPS_PER_WINDOW": MAX_CROPS_PER_WINDOW,
            "MIN_VALID_CROPS_PER_ANCHOR": MIN_VALID_CROPS_PER_ANCHOR,
            "held_out_slice": [held_out_start, held_out_end],
            "n_total_events": len(all_events),
            "n_held_out_events": len(held_out_metas),
            "n_held_out_scorable": sum(1 for m in held_out_metas if m.is_scorable()),
            "n_cross_rally_entries": len(cross_metas),
        },
        "held_out_events": [
            {
                "event_idx": m.event_idx,
                "rally_id": m.rally_id,
                "video_id": m.video_id,
                "swap_frame": m.swap_frame,
                "pred_old": m.pred_old,
                "pred_new": m.pred_new,
                "correct_player_id": m.correct_player_id,
                "wrong_player_id": m.wrong_player_id,
                "n_correct": m.n_correct,
                "n_wrong": m.n_wrong,
                "n_query": m.n_query,
                "abstain_reason": m.abstain_reason,
            }
            for m in held_out_metas
        ],
        "cross_rally_entries": [
            {
                "entry_idx": e.entry_idx,
                "video_id": e.video_id,
                "rally_id": e.rally_id,
                "canonical_id": e.canonical_id,
                "n_crops": e.n_crops,
            }
            for e in cross_metas
        ],
    }
    cache_meta.write_text(json.dumps(meta_payload, indent=2))
    logger.info("Wrote metadata → %s", cache_meta)

    return EvalCache(
        held_out_events=held_out_metas,
        held_out_features={k: v for k, v in held_out_feats.items()},
        cross_rally_entries=cross_metas,
        cross_rally_features={k: v for k, v in cross_feats.items()},
        config=meta_payload["config"],
    )


def load_cache(cache_npz: Path, cache_meta: Path) -> EvalCache:
    """Load a previously-built eval cache from disk."""
    if not cache_npz.exists() or not cache_meta.exists():
        raise FileNotFoundError(
            f"Eval cache not found at {cache_npz} / {cache_meta} — run build-eval-cache first"
        )

    meta = json.loads(cache_meta.read_text())
    npz = np.load(cache_npz)

    held_out_events = [
        HeldOutEventMeta(**{k: e[k] for k in (
            "event_idx", "rally_id", "video_id", "swap_frame", "pred_old", "pred_new",
            "correct_player_id", "wrong_player_id",
            "n_correct", "n_wrong", "n_query", "abstain_reason",
        )})
        for e in meta["held_out_events"]
    ]
    cross_rally_entries = [
        CrossRallyEntryMeta(**{k: e[k] for k in (
            "entry_idx", "video_id", "rally_id", "canonical_id", "n_crops",
        )})
        for e in meta["cross_rally_entries"]
    ]

    held_out_features: dict[str, NDArray[np.float32]] = {}
    cross_rally_features: dict[str, NDArray[np.float32]] = {}
    for key in npz.files:
        arr = np.asarray(npz[key], dtype=np.float32)
        if key.startswith("e"):
            held_out_features[key] = arr
        elif key.startswith("g"):
            cross_rally_features[key] = arr

    return EvalCache(
        held_out_events=held_out_events,
        held_out_features=held_out_features,
        cross_rally_entries=cross_rally_entries,
        cross_rally_features=cross_rally_features,
        config=meta["config"],
    )
