"""Harvest within-team ReID training pairs (Session 2).

Produces a verified training corpus for Session 3's DINOv2-S head:
- positives     (same track, different temporal windows)      ~5000
- easy negs     (different teams, same rally)                 ~2000
- hard mid negs (same team, different canonical IDs, convergence frames) ≥2000
- gold negs     (labelled pred-exchange swap events 1-34)     ~34

Four subcommands, each re-runnable and idempotent:
  mine    → candidate_pairs.jsonl
  extract → crops/ + embeddings/ (DINOv2 multi-crop medians)
  render  → reports/within_team_reid/contact_sheet.html
  summary → harvest_summary.md + rejection-rate gate

Design notes:
- The harvest filter relaxes the probe's IoU>0.30 hard-reject to IoU>0.50
  because Session 1 showed the strict filter killed 53% of ranking events.
  Multi-crop median aggregation (N=20, drop bottom 25%) handles noisy crops.
- Video-level train/val split (seed=42) baked into the manifest — no leakage.
- Known-bad rallies (fad29c31*, 2dff5eeb*, c48eeb7d*) excluded from training.
- Gold tier DRAWS ONLY from ranking events 1-34; events 35-58 are Session 4
  acceptance-only and must not leak into training.

Usage:
    cd analysis
    uv run python scripts/harvest_within_team_pairs.py mine \\
        --output-dir training_data/within_team_reid
    uv run python scripts/harvest_within_team_pairs.py extract \\
        --input-dir training_data/within_team_reid
    uv run python scripts/harvest_within_team_pairs.py render \\
        --input-dir training_data/within_team_reid
    uv run python scripts/harvest_within_team_pairs.py summary \\
        --input-dir training_data/within_team_reid \\
        --rejected training_data/within_team_reid/rejected_pairs.json
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import logging
import random
import sys
import time
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from rallycut.evaluation.db import get_connection
from rallycut.evaluation.tracking.db import get_video_path
from rallycut.tracking.player_features import extract_bbox_crop
from rallycut.tracking.player_tracker import PlayerPosition
from rallycut.tracking.reid_embeddings import extract_backbone_features
from rallycut.tracking.swap_reid_probe import _iterate_rally_frames

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("harvest_within_team")

# ---------------------------------------------------------------------------
# Constants — locked by the plan
# ---------------------------------------------------------------------------

RANKING_SIZE = 34  # gold tier draws only from events 1..RANKING_SIZE
AUDIT_DIR = Path("reports/tracking_audit")
REID_DEBUG_DIR = AUDIT_DIR / "reid_debug"

# Known-bad rally prefixes (quarantined per memory)
KNOWN_BAD_PREFIXES = ("fad29c31", "2dff5eeb", "c48eeb7d")

# Crop filtering (harvest is lenient — probe used stricter values)
CROP_MIN_HEIGHT_FRAC = 0.05      # reject tiny boxes (< 5% frame height)
CROP_EDGE_MARGIN_FRAC = 0.02     # reject if within 2% of any edge
CROP_OCCLUSION_IOU_HARD = 0.50   # harvest: only reject near-total occlusion
CROP_MIN_CONFIDENCE = 0.30       # detection confidence floor

# Multi-crop median aggregation
N_CROPS_PER_WINDOW = 20           # sample count per pred window
QUALITY_DROP_FRAC = 0.25          # drop bottom 25% by quality
MIN_CROPS_FOR_EMBEDDING = 8       # skip window if < N survive post-drop

# Window sizing
GOLD_WINDOW_FRAMES = 30           # pre-swap window: [swap_frame-30, swap_frame)
MID_WINDOW_FRAMES = 30            # ±30 frames around convergence
POS_WINDOW_FRAMES = 60            # positive same-track windows
EASY_NEG_WINDOW_FRAMES = 60       # easy-neg cross-team windows

# Same-team convergence mining
CONVERGENCE_NORM_DIST = 0.15      # Euclidean normalized distance threshold
CONVERGENCE_GAP_FRAMES = 5        # cluster contiguous convergence frames
NEAR_NET_DELTA = 0.05             # |y - court_split_y| ≤ delta is "near net"
MIN_TRACK_FRAMES = 100            # track must have ≥ 100 frames to qualify
MAX_MID_PER_PAIR = 3              # cap to avoid rally dominance
MAX_POS_PER_RALLY = 8
MAX_EASY_NEG_PER_RALLY = 4

# Hit-quota early-stop ceilings (oversample by ~25% for rejection headroom)
TARGET_GOLD = 34                  # all ranking events
TARGET_MID = 2500
TARGET_POS = 6000
TARGET_EASY_NEG = 2500

# Contact-sheet render
SHEET_N_PER_TIER = 250
SHEET_MID_HARD_FRAC = 0.33        # 1/3 of mid slots for "hard" oversample
THUMB_SIZE = 224                  # DINOv2 native input

# Video-level split
SPLIT_SEED = 42
TRAIN_FRAC = 0.80

# Rejection gate
GOLD_REJECT_GATE = 0.05
MID_REJECT_GATE = 0.15

# JPEG quality for saved crops
JPEG_QUALITY = 92


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TrackWindow:
    """One track's slice of a rally used as a pair anchor."""

    track_id: int
    canonical_id: int | None      # match_players canonical (1-4), None if unknown
    team: int                     # 0 = near (players 1-2 base), 1 = far (players 3-4 base)
    window_frames: list[int]       # actual rally-frame integers in this window

    def window_key(self) -> str:
        """Stable key for embedding cache file naming."""
        lo, hi = min(self.window_frames), max(self.window_frames)
        return f"t{self.track_id}__f{lo:06d}-{hi:06d}"


@dataclass
class PairRecord:
    """One candidate training pair (positive / easy-neg / mid / gold)."""

    pair_id: str
    tier: str                     # "gold" | "mid" | "positive" | "easy_neg"
    label: str                    # "same" | "different"
    split: str                    # "train" | "val"
    rally_id: str
    video_id: str
    video_fps: float
    video_width: int
    video_height: int
    rally_start_ms: int
    track_a: TrackWindow
    track_b: TrackWindow
    convergence_frame: int | None # center frame for gold/mid
    distance_at_convergence: float | None
    near_net: bool                # mid tier: convergence was in near-net band
    difficulty_score: float       # higher = harder; shown on contact sheet


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _pair_id(prefix: str, rally_id: str, suffix: str) -> str:
    """Deterministic pair ID: <tier>_<rally8>_<suffix>."""
    return f"{prefix}_{rally_id[:8]}_{suffix}"


_split_cache: dict[str, dict[str, str]] = {}


def _split_for_video(video_id: str, all_video_ids: list[str] | None = None) -> str:
    """Deterministic 80/20 video-level split with exact quota.

    First call must pass the full video list so the split is computed once
    (sorted deterministically, first TRAIN_FRAC → train, rest → val). Cached
    for subsequent lookups. This beats hash-based split for small corpora
    where hash variance produces skewed splits (e.g., 63/5 instead of 54/14).
    """
    cache = _split_cache.setdefault("video", {})
    if not cache and all_video_ids:
        # Deterministic shuffle: sort by stable blake2b hash of the ID (seeded),
        # then split at exact quota.
        def key(vid: str) -> bytes:
            return hashlib.blake2b(
                f"{SPLIT_SEED}:{vid}".encode(), digest_size=8,
            ).digest()
        ordered = sorted(set(all_video_ids), key=key)
        cutoff = int(round(len(ordered) * TRAIN_FRAC))
        for i, vid in enumerate(ordered):
            cache[vid] = "train" if i < cutoff else "val"
    return cache.get(video_id, "train")


def _is_known_bad(rally_id: str) -> bool:
    return any(rally_id.startswith(p) for p in KNOWN_BAD_PREFIXES)


def _cos_sim(a: NDArray[np.floating], b: NDArray[np.floating]) -> float:
    denom = (float(np.linalg.norm(a)) * float(np.linalg.norm(b))) or 1.0
    return float(np.dot(a, b) / denom)


def _bbox_iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """IoU on normalized (cx, cy, w, h) bboxes."""
    ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2
    ax2, ay2 = a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2
    bx2, by2 = b[0] + b[2] / 2, b[1] + b[3] / 2
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _normalized_distance(a: PlayerPosition, b: PlayerPosition) -> float:
    return float(np.hypot(a.x - b.x, a.y - b.y))


def _mid_difficulty(
    distance: float, near_net: bool, min_track_frames: int,
) -> float:
    """Higher = harder. Combines convergence closeness, near-net, track stability."""
    closeness = max(0.0, 1.0 - distance / CONVERGENCE_NORM_DIST)
    net_w = 1.0 if near_net else 0.5
    stability = min(min_track_frames / 200.0, 1.0)
    return closeness * net_w * stability


# ---------------------------------------------------------------------------
# Gold event loader (duplicated from probe_reid_models_on_swaps.py to avoid
# fragile cross-script import; kept in strict lock-step with that script's
# `_load_events_from_audit` so the sorted (rally_id, swap_frame) order
# matches the probe's 34/24 split.)
# ---------------------------------------------------------------------------


@dataclass
class SwapEvent:
    rally_id: str
    video_id: str
    swap_frame: int
    gt_track_id: int
    pred_old: int
    pred_new: int
    correct_player_id: int | None
    wrong_player_id: int | None


def _player_id_from_gt_label(label: str) -> int | None:
    if not label or not label.startswith("player_"):
        return None
    try:
        pid = int(label.split("_", 1)[1])
        return pid if 1 <= pid <= 4 else None
    except ValueError:
        return None


def _load_events_from_audit(audit_path: Path) -> list[SwapEvent]:
    """Re-derive pred-exchange swap events from a rally audit JSON."""
    audit = json.loads(audit_path.read_text())
    gt_label_by_id: dict[int, str] = {
        int(g["gtTrackId"]): g["gtLabel"] for g in audit.get("perGt", [])
    }

    pred_history: dict[int, list[tuple[int, int, int]]] = {}
    for g in audit.get("perGt", []):
        for s, e, pid in g.get("predIdSpans", []):
            pred_history.setdefault(pid, []).append((g["gtTrackId"], s, e))
    for h in pred_history.values():
        h.sort(key=lambda t: t[1])

    def prior_gt_of(pid: int, before: int) -> int | None:
        last_gt: int | None = None
        for gt_id, s, _e in pred_history.get(pid, []):
            if s >= before:
                break
            last_gt = gt_id
        return last_gt

    out: list[SwapEvent] = []
    for g in audit.get("perGt", []):
        spans = g.get("predIdSpans", [])
        for prev, cur in zip(spans, spans[1:]):
            _, _prev_end, prev_pred = prev
            cur_start, _, cur_pred = cur
            if prev_pred == cur_pred or prev_pred < 0 or cur_pred < 0:
                continue
            incoming = prior_gt_of(cur_pred, cur_start)
            if incoming is None or incoming == g["gtTrackId"]:
                continue
            correct = _player_id_from_gt_label(gt_label_by_id.get(incoming, ""))
            wrong = _player_id_from_gt_label(g["gtLabel"])
            out.append(SwapEvent(
                rally_id=audit["rallyId"],
                video_id=audit["videoId"],
                swap_frame=cur_start,
                gt_track_id=g["gtTrackId"],
                pred_old=prev_pred,
                pred_new=cur_pred,
                correct_player_id=correct,
                wrong_player_id=wrong,
            ))
    return out


def _load_ranking_gold_events() -> list[SwapEvent]:
    """Load gold events: the first RANKING_SIZE events across all audit rallies."""
    if not REID_DEBUG_DIR.exists():
        logger.warning("  reid_debug dir %s missing — no gold events", REID_DEBUG_DIR)
        return []

    rally_ids: list[str] = []
    for p in sorted(REID_DEBUG_DIR.glob("*.json")):
        name = p.name
        if name.startswith("_") or "sota_probe" in name:
            continue
        rally_ids.append(name.removesuffix(".json"))

    events: list[SwapEvent] = []
    for rid in rally_ids:
        audit_path = AUDIT_DIR / f"{rid}.json"
        if not audit_path.exists():
            continue
        events.extend(_load_events_from_audit(audit_path))

    events.sort(key=lambda e: (e.rally_id, e.swap_frame))
    return events[:RANKING_SIZE]


# ---------------------------------------------------------------------------
# DB loader — one sweep over videos × rallies × player_tracks
# ---------------------------------------------------------------------------


@dataclass
class HarvestRally:
    rally_id: str
    video_id: str
    start_ms: int
    end_ms: int
    video_fps: float
    video_width: int
    video_height: int
    positions: list[PlayerPosition]
    primary_track_ids: list[int]
    court_split_y: float | None
    team_assignments: dict[int, int]      # track_id → 0|1 (derived)
    canonical_by_track: dict[int, int]    # track_id → canonical_id (1-4)


def _team_from_canonical(canonical_id: int, side_switch_count: int) -> int:
    """Match-level canonical_id (1-4) → team, with cumulative side switches."""
    base_team = 0 if canonical_id <= 2 else 1
    return base_team if side_switch_count % 2 == 0 else 1 - base_team


def _teams_from_match_analysis(
    match_analysis: dict[str, Any], rally_id: str,
) -> tuple[dict[int, int], dict[int, int]]:
    """Derive (track_id→team, track_id→canonical) for one rally.

    Accounts for cumulative side switches by walking all prior rallies in
    match_analysis.rallies order. Side-switch-aware, matches
    build_match_team_assignments.
    """
    rallies = match_analysis.get("rallies", []) or []
    side_switch_count = 0
    teams: dict[int, int] = {}
    canonical: dict[int, int] = {}
    for entry in rallies:
        if entry.get("sideSwitchDetected") or entry.get("side_switch_detected"):
            side_switch_count += 1
        rid = entry.get("rallyId") or entry.get("rally_id", "")
        if rid != rally_id:
            continue
        t2p_raw = entry.get("trackToPlayer") or entry.get("track_to_player", {}) or {}
        for tid_str, pid in t2p_raw.items():
            tid = int(tid_str)
            cid = int(pid)
            canonical[tid] = cid
            teams[tid] = _team_from_canonical(cid, side_switch_count)
        return teams, canonical
    return teams, canonical


def iter_harvest_rallies() -> Iterator[HarvestRally]:
    """Yield every rally that has tracking + match_analysis populated.

    Streams results so callers can hit-quota early without loading everything.
    """
    query = """
        SELECT v.id, v.match_analysis_json, v.width, v.height, v.fps,
               r.id, r.start_ms, r.end_ms,
               pt.positions_json, pt.primary_track_ids, pt.court_split_y
        FROM videos v
        JOIN rallies r ON r.video_id = v.id
        JOIN player_tracks pt ON pt.rally_id = r.id
        WHERE v.match_analysis_json IS NOT NULL
          AND pt.positions_json IS NOT NULL
        ORDER BY v.id, r.start_ms
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            for row in cur.fetchall():
                (
                    video_id, match_analysis, width, height, fps,
                    rally_id, start_ms, end_ms,
                    positions_json, primary_track_ids, court_split_y,
                ) = row

                video_id = cast(str, video_id)
                rally_id = cast(str, rally_id)
                if _is_known_bad(rally_id):
                    continue

                match_analysis = cast(dict[str, Any], match_analysis)
                pos_json = cast(list[dict[str, Any]] | None, positions_json)
                if not pos_json:
                    continue

                positions = [
                    PlayerPosition(
                        frame_number=p["frameNumber"],
                        track_id=p["trackId"],
                        x=p["x"],
                        y=p["y"],
                        width=p["width"],
                        height=p["height"],
                        confidence=p["confidence"],
                    )
                    for p in pos_json
                ]

                teams, canonical = _teams_from_match_analysis(
                    match_analysis, rally_id,
                )
                if not teams or not canonical:
                    continue

                yield HarvestRally(
                    rally_id=rally_id,
                    video_id=video_id,
                    start_ms=cast(int, start_ms),
                    end_ms=cast(int, end_ms),
                    video_fps=float(cast(Any, fps) or 30.0),
                    video_width=int(cast(Any, width) or 1920),
                    video_height=int(cast(Any, height) or 1080),
                    positions=positions,
                    primary_track_ids=list(cast(list[int], primary_track_ids) or []),
                    court_split_y=(
                        float(cast(float, court_split_y))
                        if court_split_y is not None else None
                    ),
                    team_assignments=teams,
                    canonical_by_track=canonical,
                )


# ---------------------------------------------------------------------------
# Pair mining
# ---------------------------------------------------------------------------


def _positions_by_frame(
    positions: list[PlayerPosition], track_ids: set[int],
) -> dict[int, dict[int, PlayerPosition]]:
    """{frame: {track_id: PlayerPosition}} restricted to requested tracks."""
    out: dict[int, dict[int, PlayerPosition]] = defaultdict(dict)
    for p in positions:
        if p.track_id in track_ids:
            out[p.frame_number][p.track_id] = p
    return out


def _track_frames(positions: list[PlayerPosition], track_id: int) -> list[int]:
    return sorted({p.frame_number for p in positions if p.track_id == track_id})


def _build_rally_frame_maps(
    rally: HarvestRally,
) -> tuple[dict[tuple[int, int], PlayerPosition], dict[int, list[PlayerPosition]]]:
    """Maps used for mine-time hard-reject filtering (same shape as extract)."""
    pos_by_key: dict[tuple[int, int], PlayerPosition] = {
        (p.frame_number, p.track_id): p for p in rally.positions
    }
    primary_set = set(rally.primary_track_ids)
    primary_at_frame: dict[int, list[PlayerPosition]] = defaultdict(list)
    for p in rally.positions:
        if p.track_id in primary_set:
            primary_at_frame[p.frame_number].append(p)
    return pos_by_key, primary_at_frame


def _filter_extractable_frames(
    frames: list[int],
    track_id: int,
    pos_by_key: dict[tuple[int, int], PlayerPosition],
    primary_at_frame: dict[int, list[PlayerPosition]],
) -> list[int]:
    """Subset of frames whose crop would survive the extract-time hard filter.

    Mine-time analog of extract's `_reject_crop_hard` — runs the same rules
    (bbox size, edge margin, confidence, occlusion IoU ≤ 0.5) before a pair
    is committed to candidate_pairs.jsonl, so no pair enters the corpus with
    zero usable crops.
    """
    out: list[int] = []
    for f in frames:
        pos = pos_by_key.get((f, track_id))
        if pos is None:
            continue
        if _reject_crop_hard(pos, primary_at_frame.get(f, [])):
            continue
        out.append(f)
    return out


def _mine_mid_for_rally(rally: HarvestRally) -> list[PairRecord]:
    """Convergence-frame mining for within-team, different-canonical pairs."""
    if rally.court_split_y is None:
        return []

    primary = [tid for tid in rally.primary_track_ids if tid in rally.team_assignments]
    track_frame_counts = {
        tid: len(_track_frames(rally.positions, tid))
        for tid in primary
    }
    eligible = [tid for tid in primary if track_frame_counts[tid] >= MIN_TRACK_FRAMES]

    pos_by_frame = _positions_by_frame(rally.positions, set(eligible))
    pos_by_key, primary_at_frame = _build_rally_frame_maps(rally)
    pairs: list[PairRecord] = []
    split = _split_for_video(rally.video_id)

    for i, tid_a in enumerate(eligible):
        for tid_b in eligible[i + 1:]:
            team_a = rally.team_assignments.get(tid_a)
            team_b = rally.team_assignments.get(tid_b)
            if team_a is None or team_b is None or team_a != team_b:
                continue
            can_a = rally.canonical_by_track.get(tid_a)
            can_b = rally.canonical_by_track.get(tid_b)
            if can_a is None or can_b is None or can_a == can_b:
                continue

            convergence_frames: list[tuple[int, float, bool]] = []
            for frame, per_track in pos_by_frame.items():
                if tid_a not in per_track or tid_b not in per_track:
                    continue
                d = _normalized_distance(per_track[tid_a], per_track[tid_b])
                if d > CONVERGENCE_NORM_DIST:
                    continue
                y_mean = (per_track[tid_a].y + per_track[tid_b].y) / 2.0
                near_net = abs(y_mean - (rally.court_split_y or 0.5)) <= NEAR_NET_DELTA
                convergence_frames.append((frame, d, near_net))

            if not convergence_frames:
                continue
            convergence_frames.sort(key=lambda t: t[0])

            clusters: list[list[tuple[int, float, bool]]] = [[convergence_frames[0]]]
            for f in convergence_frames[1:]:
                if f[0] - clusters[-1][-1][0] <= CONVERGENCE_GAP_FRAMES:
                    clusters[-1].append(f)
                else:
                    clusters.append([f])

            clusters.sort(
                key=lambda c: _mid_difficulty(
                    distance=min(e[1] for e in c),
                    near_net=any(e[2] for e in c),
                    min_track_frames=min(
                        track_frame_counts[tid_a], track_frame_counts[tid_b],
                    ),
                ),
                reverse=True,
            )

            for cluster in clusters[:MAX_MID_PER_PAIR]:
                midpoint_idx = len(cluster) // 2
                convergence_frame, distance, near_net = cluster[midpoint_idx]
                window_lo = max(0, convergence_frame - MID_WINDOW_FRAMES)
                window_hi = convergence_frame + MID_WINDOW_FRAMES

                raw_a = [
                    f for f in _track_frames(rally.positions, tid_a)
                    if window_lo <= f <= window_hi
                ]
                raw_b = [
                    f for f in _track_frames(rally.positions, tid_b)
                    if window_lo <= f <= window_hi
                ]
                frames_a = _filter_extractable_frames(
                    raw_a, tid_a, pos_by_key, primary_at_frame,
                )
                frames_b = _filter_extractable_frames(
                    raw_b, tid_b, pos_by_key, primary_at_frame,
                )
                if (len(frames_a) < MIN_CROPS_FOR_EMBEDDING
                        or len(frames_b) < MIN_CROPS_FOR_EMBEDDING):
                    continue

                difficulty = _mid_difficulty(
                    distance, near_net,
                    min(track_frame_counts[tid_a], track_frame_counts[tid_b]),
                )

                pair = PairRecord(
                    pair_id=_pair_id("mid", rally.rally_id, f"{tid_a}-{tid_b}-f{convergence_frame}"),
                    tier="mid",
                    label="different",
                    split=split,
                    rally_id=rally.rally_id,
                    video_id=rally.video_id,
                    video_fps=rally.video_fps,
                    video_width=rally.video_width,
                    video_height=rally.video_height,
                    rally_start_ms=rally.start_ms,
                    track_a=TrackWindow(
                        track_id=tid_a,
                        canonical_id=can_a,
                        team=team_a,
                        window_frames=frames_a,
                    ),
                    track_b=TrackWindow(
                        track_id=tid_b,
                        canonical_id=can_b,
                        team=team_b,
                        window_frames=frames_b,
                    ),
                    convergence_frame=convergence_frame,
                    distance_at_convergence=distance,
                    near_net=near_net,
                    difficulty_score=difficulty,
                )
                pairs.append(pair)
    return pairs


def _mine_positives_for_rally(rally: HarvestRally) -> list[PairRecord]:
    """Same-track, two disjoint temporal windows → positive pair."""
    primary = [tid for tid in rally.primary_track_ids if tid in rally.team_assignments]
    pos_by_key, primary_at_frame = _build_rally_frame_maps(rally)
    pairs: list[PairRecord] = []
    split = _split_for_video(rally.video_id)

    emitted = 0
    for tid in primary:
        frames = _track_frames(rally.positions, tid)
        if len(frames) < MIN_TRACK_FRAMES:
            continue
        team = rally.team_assignments[tid]
        canonical = rally.canonical_by_track.get(tid)

        # Early third: frames[10 .. 10+POS_WINDOW]
        # Late third:  frames[-70 .. -10]
        raw_early = [f for f in frames if frames[10] <= f < frames[10] + POS_WINDOW_FRAMES]
        late_start_idx = max(10, len(frames) - (POS_WINDOW_FRAMES + 10))
        raw_late = [f for f in frames[late_start_idx:] if f <= frames[-10]]
        early = _filter_extractable_frames(raw_early, tid, pos_by_key, primary_at_frame)
        late = _filter_extractable_frames(raw_late, tid, pos_by_key, primary_at_frame)
        if len(early) < MIN_CROPS_FOR_EMBEDDING or len(late) < MIN_CROPS_FOR_EMBEDDING:
            continue
        if max(early) >= min(late):
            continue  # windows overlapping — skip

        pair = PairRecord(
            pair_id=_pair_id("pos", rally.rally_id, f"t{tid}"),
            tier="positive",
            label="same",
            split=split,
            rally_id=rally.rally_id,
            video_id=rally.video_id,
            video_fps=rally.video_fps,
            video_width=rally.video_width,
            video_height=rally.video_height,
            rally_start_ms=rally.start_ms,
            track_a=TrackWindow(
                track_id=tid, canonical_id=canonical, team=team,
                window_frames=early,
            ),
            track_b=TrackWindow(
                track_id=tid, canonical_id=canonical, team=team,
                window_frames=late,
            ),
            convergence_frame=None,
            distance_at_convergence=None,
            near_net=False,
            difficulty_score=float(len(early) + len(late)) / 200.0,
        )
        pairs.append(pair)
        emitted += 1
        if emitted >= MAX_POS_PER_RALLY:
            break
    return pairs


def _mine_easy_negs_for_rally(
    rally: HarvestRally, rng: random.Random,
) -> list[PairRecord]:
    """Cross-team same rally — easy negatives."""
    primary = [tid for tid in rally.primary_track_ids if tid in rally.team_assignments]
    pos_by_key, primary_at_frame = _build_rally_frame_maps(rally)
    pairs: list[PairRecord] = []
    split = _split_for_video(rally.video_id)

    by_team: dict[int, list[int]] = defaultdict(list)
    for tid in primary:
        if len(_track_frames(rally.positions, tid)) >= MIN_TRACK_FRAMES:
            by_team[rally.team_assignments[tid]].append(tid)

    if 0 not in by_team or 1 not in by_team:
        return []

    combos: list[tuple[int, int]] = [
        (a, b) for a in by_team[0] for b in by_team[1]
    ]
    rng.shuffle(combos)
    combos = combos[:MAX_EASY_NEG_PER_RALLY]

    for tid_a, tid_b in combos:
        frames_a = _track_frames(rally.positions, tid_a)
        frames_b = _track_frames(rally.positions, tid_b)
        mid_a = len(frames_a) // 2
        mid_b = len(frames_b) // 2
        raw_a = frames_a[max(0, mid_a - EASY_NEG_WINDOW_FRAMES // 2):
                         mid_a + EASY_NEG_WINDOW_FRAMES // 2]
        raw_b = frames_b[max(0, mid_b - EASY_NEG_WINDOW_FRAMES // 2):
                         mid_b + EASY_NEG_WINDOW_FRAMES // 2]
        w_a = _filter_extractable_frames(raw_a, tid_a, pos_by_key, primary_at_frame)
        w_b = _filter_extractable_frames(raw_b, tid_b, pos_by_key, primary_at_frame)
        if len(w_a) < MIN_CROPS_FOR_EMBEDDING or len(w_b) < MIN_CROPS_FOR_EMBEDDING:
            continue

        pair = PairRecord(
            pair_id=_pair_id("neg", rally.rally_id, f"t{tid_a}-t{tid_b}"),
            tier="easy_neg",
            label="different",
            split=split,
            rally_id=rally.rally_id,
            video_id=rally.video_id,
            video_fps=rally.video_fps,
            video_width=rally.video_width,
            video_height=rally.video_height,
            rally_start_ms=rally.start_ms,
            track_a=TrackWindow(
                track_id=tid_a,
                canonical_id=rally.canonical_by_track.get(tid_a),
                team=rally.team_assignments[tid_a],
                window_frames=w_a,
            ),
            track_b=TrackWindow(
                track_id=tid_b,
                canonical_id=rally.canonical_by_track.get(tid_b),
                team=rally.team_assignments[tid_b],
                window_frames=w_b,
            ),
            convergence_frame=None,
            distance_at_convergence=None,
            near_net=False,
            difficulty_score=0.0,
        )
        pairs.append(pair)
    return pairs


def _mine_gold_pairs(
    rally_lookup: dict[str, HarvestRally],
) -> list[PairRecord]:
    """Gold tier from labelled pred-exchange ranking events (1-RANKING_SIZE)."""
    gold_events = _load_ranking_gold_events()
    pairs: list[PairRecord] = []
    dropped_no_rally = 0
    dropped_cross_team = 0
    dropped_same_canonical = 0
    dropped_insufficient_frames = 0

    for ev in gold_events:
        rally = rally_lookup.get(ev.rally_id)
        if rally is None:
            dropped_no_rally += 1
            continue
        # Same-team determined from the GT-derived physical player IDs, NOT
        # from match_analysis team_assignments. match_analysis can mis-assign
        # pred_old/pred_new to canonicals that straddle the 1-2 / 3-4 boundary
        # when the tracker swap propagates through post-processing; the GT
        # labels (correct_player_id = priorGtOfNew, wrong_player_id = gtTrackId)
        # are the ground-truth identities we're trying to teach the head to
        # separate.
        if ev.correct_player_id is None or ev.wrong_player_id is None:
            dropped_same_canonical += 1  # no GT player_id — can't verify
            continue
        same_team = (ev.correct_player_id <= 2) == (ev.wrong_player_id <= 2)
        if not same_team:
            dropped_cross_team += 1
            continue
        if ev.correct_player_id == ev.wrong_player_id:
            dropped_same_canonical += 1
            continue
        # Team label for the record is GT-derived; canonical falls back to
        # match_analysis if present (useful for debugging / display only).
        team_old = 0 if ev.correct_player_id <= 2 else 1
        team_new = 0 if ev.wrong_player_id <= 2 else 1
        can_old = rally.canonical_by_track.get(ev.pred_old) or ev.correct_player_id
        can_new = rally.canonical_by_track.get(ev.pred_new) or ev.wrong_player_id

        lo = max(0, ev.swap_frame - GOLD_WINDOW_FRAMES)
        hi = ev.swap_frame
        pos_by_key, primary_at_frame = _build_rally_frame_maps(rally)
        raw_old = [f for f in _track_frames(rally.positions, ev.pred_old)
                   if lo <= f < hi]
        raw_new = [f for f in _track_frames(rally.positions, ev.pred_new)
                   if lo <= f < hi]
        frames_old = _filter_extractable_frames(
            raw_old, ev.pred_old, pos_by_key, primary_at_frame,
        )
        frames_new = _filter_extractable_frames(
            raw_new, ev.pred_new, pos_by_key, primary_at_frame,
        )
        if (len(frames_old) < MIN_CROPS_FOR_EMBEDDING
                or len(frames_new) < MIN_CROPS_FOR_EMBEDDING):
            dropped_insufficient_frames += 1
            continue

        split = _split_for_video(rally.video_id)
        pair = PairRecord(
            pair_id=_pair_id("gold", rally.rally_id, f"f{ev.swap_frame}"),
            tier="gold",
            label="different",
            split=split,
            rally_id=rally.rally_id,
            video_id=rally.video_id,
            video_fps=rally.video_fps,
            video_width=rally.video_width,
            video_height=rally.video_height,
            rally_start_ms=rally.start_ms,
            track_a=TrackWindow(
                track_id=ev.pred_old, canonical_id=can_old, team=team_old,
                window_frames=frames_old,
            ),
            track_b=TrackWindow(
                track_id=ev.pred_new, canonical_id=can_new, team=team_new,
                window_frames=frames_new,
            ),
            convergence_frame=ev.swap_frame,
            distance_at_convergence=None,
            near_net=False,
            difficulty_score=1.0,
        )
        pairs.append(pair)

    logger.info(
        "  gold: %d kept, %d dropped (no_rally=%d, cross_team=%d, "
        "same_canonical=%d, insufficient_frames=%d)",
        len(pairs), dropped_no_rally + dropped_cross_team + dropped_same_canonical
        + dropped_insufficient_frames, dropped_no_rally, dropped_cross_team,
        dropped_same_canonical, dropped_insufficient_frames,
    )
    return pairs


# ---------------------------------------------------------------------------
# Mine command
# ---------------------------------------------------------------------------


def cmd_mine(output_dir: Path, limit_rallies: int = 0) -> dict[str, Any]:
    """Enumerate DB rallies and emit candidate_pairs.jsonl + manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SPLIT_SEED)

    # Stream rallies and accumulate.
    rally_lookup: dict[str, HarvestRally] = {}
    mid_pairs: list[PairRecord] = []
    pos_pairs: list[PairRecord] = []
    easy_pairs: list[PairRecord] = []

    # Collect all rallies into a list first so we can shuffle deterministically.
    all_rallies = list(iter_harvest_rallies())
    logger.info("Loaded %d eligible rallies from DB", len(all_rallies))
    # Seed the deterministic video-level split with the full video set.
    _split_cache.clear()
    _split_for_video("__prime__", [r.video_id for r in all_rallies])
    rng.shuffle(all_rallies)
    if limit_rallies > 0:
        all_rallies = all_rallies[:limit_rallies]
        logger.info("  limited to %d rallies", limit_rallies)

    for idx, rally in enumerate(all_rallies, start=1):
        rally_lookup[rally.rally_id] = rally
        mid_pairs.extend(_mine_mid_for_rally(rally))
        pos_pairs.extend(_mine_positives_for_rally(rally))
        easy_pairs.extend(_mine_easy_negs_for_rally(rally, rng))

        if idx % 25 == 0 or idx == len(all_rallies):
            logger.info(
                "  [%d/%d] rally %s — mid=%d pos=%d easy=%d (cumulative)",
                idx, len(all_rallies), rally.rally_id[:8],
                len(mid_pairs), len(pos_pairs), len(easy_pairs),
            )

        hit_mid = len(mid_pairs) >= TARGET_MID
        hit_pos = len(pos_pairs) >= TARGET_POS
        hit_easy = len(easy_pairs) >= TARGET_EASY_NEG
        if hit_mid and hit_pos and hit_easy:
            logger.info(
                "  hit-quota early-stop at rally %d/%d (all targets met)",
                idx, len(all_rallies),
            )
            break

    gold_pairs = _mine_gold_pairs(rally_lookup)

    # Stratified sampling: if over-quota, randomly downsample each tier.
    rng.shuffle(mid_pairs)
    rng.shuffle(pos_pairs)
    rng.shuffle(easy_pairs)
    mid_pairs = mid_pairs[:TARGET_MID]
    pos_pairs = pos_pairs[:TARGET_POS]
    easy_pairs = easy_pairs[:TARGET_EASY_NEG]

    all_pairs = gold_pairs + mid_pairs + pos_pairs + easy_pairs
    logger.info(
        "Final tiers: gold=%d mid=%d positive=%d easy_neg=%d (total=%d)",
        len(gold_pairs), len(mid_pairs), len(pos_pairs), len(easy_pairs),
        len(all_pairs),
    )

    # Split summary
    train_videos = {p.video_id for p in all_pairs if p.split == "train"}
    val_videos = {p.video_id for p in all_pairs if p.split == "val"}

    # Write candidate_pairs.jsonl
    pairs_path = output_dir / "candidate_pairs.jsonl"
    with pairs_path.open("w") as fh:
        for p in all_pairs:
            fh.write(json.dumps(_pair_to_dict(p)) + "\n")
    logger.info("Wrote %s (%d pairs)", pairs_path, len(all_pairs))

    # Write manifest.json (index + split summary)
    manifest = {
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tier_counts": {
            "gold": len(gold_pairs),
            "mid": len(mid_pairs),
            "positive": len(pos_pairs),
            "easy_neg": len(easy_pairs),
        },
        "split_counts": {
            "train_videos": len(train_videos),
            "val_videos": len(val_videos),
            "train_pairs": sum(1 for p in all_pairs if p.split == "train"),
            "val_pairs": sum(1 for p in all_pairs if p.split == "val"),
        },
        "harvest_config": {
            "crop_min_height_frac": CROP_MIN_HEIGHT_FRAC,
            "crop_occlusion_iou_hard": CROP_OCCLUSION_IOU_HARD,
            "n_crops_per_window": N_CROPS_PER_WINDOW,
            "quality_drop_frac": QUALITY_DROP_FRAC,
            "min_crops_for_embedding": MIN_CROPS_FOR_EMBEDDING,
            "mid_window_frames": MID_WINDOW_FRAMES,
            "convergence_norm_dist": CONVERGENCE_NORM_DIST,
            "min_track_frames": MIN_TRACK_FRAMES,
            "split_seed": SPLIT_SEED,
            "train_frac": TRAIN_FRAC,
        },
        "known_bad_prefixes": list(KNOWN_BAD_PREFIXES),
        "ranking_event_cap": RANKING_SIZE,
        "mine_sanity_check": {
            "min_gold_for_continue": 15,
            "min_mid_for_continue": 500,
            "gold_below_target_warning": len(gold_pairs) < 150,
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Wrote %s", manifest_path)

    # Sanity check: if gold < 15 or mid < 500, flag the run as broken.
    sanity_pass = len(gold_pairs) >= 15 and len(mid_pairs) >= 500
    if limit_rallies > 0 and limit_rallies <= 5:
        # Smoke test doesn't need to hit full-scale targets.
        sanity_pass = True
    if not sanity_pass:
        logger.warning(
            "SANITY CHECK FAILED: gold=%d (need ≥15), mid=%d (need ≥500). "
            "Investigate before running extract.",
            len(gold_pairs), len(mid_pairs),
        )

    return manifest


def _pair_to_dict(p: PairRecord) -> dict[str, Any]:
    return {
        "pair_id": p.pair_id,
        "tier": p.tier,
        "label": p.label,
        "split": p.split,
        "rally_id": p.rally_id,
        "video_id": p.video_id,
        "video_fps": p.video_fps,
        "video_width": p.video_width,
        "video_height": p.video_height,
        "rally_start_ms": p.rally_start_ms,
        "track_a": asdict(p.track_a),
        "track_b": asdict(p.track_b),
        "convergence_frame": p.convergence_frame,
        "distance_at_convergence": p.distance_at_convergence,
        "near_net": p.near_net,
        "difficulty_score": p.difficulty_score,
    }


def _pair_from_dict(d: dict[str, Any]) -> PairRecord:
    return PairRecord(
        pair_id=d["pair_id"],
        tier=d["tier"],
        label=d["label"],
        split=d["split"],
        rally_id=d["rally_id"],
        video_id=d["video_id"],
        video_fps=d["video_fps"],
        video_width=d["video_width"],
        video_height=d["video_height"],
        rally_start_ms=d["rally_start_ms"],
        track_a=TrackWindow(**d["track_a"]),
        track_b=TrackWindow(**d["track_b"]),
        convergence_frame=d["convergence_frame"],
        distance_at_convergence=d["distance_at_convergence"],
        near_net=d["near_net"],
        difficulty_score=d["difficulty_score"],
    )


def _load_candidate_pairs(input_dir: Path) -> list[PairRecord]:
    pairs_path = input_dir / "candidate_pairs.jsonl"
    if not pairs_path.exists():
        raise FileNotFoundError(f"{pairs_path} not found — run `mine` first")
    pairs: list[PairRecord] = []
    with pairs_path.open() as fh:
        for line in fh:
            if line.strip():
                pairs.append(_pair_from_dict(json.loads(line)))
    return pairs


# ---------------------------------------------------------------------------
# Extract command — crop harvest + multi-crop median DINOv2 embeddings
# ---------------------------------------------------------------------------


def _crop_path(input_dir: Path, rally_id: str, track_id: int, frame: int) -> Path:
    return input_dir / "crops" / rally_id / f"t{track_id}" / f"{frame:06d}.jpg"


def _emb_path(input_dir: Path, rally_id: str, window: TrackWindow) -> Path:
    return input_dir / "embeddings" / f"{rally_id}__{window.window_key()}.npy"


def _quality_score(
    pos: PlayerPosition,
    primary_at_frame: list[PlayerPosition],
    crop: NDArray[np.uint8],
) -> float:
    """Higher = better crop. Used to drop bottom 25% before median."""
    height_score = min(pos.height / 0.20, 1.0)
    confidence_score = min(pos.confidence, 1.0)

    left = pos.x - pos.width / 2
    right = pos.x + pos.width / 2
    top = pos.y - pos.height / 2
    bottom = pos.y + pos.height / 2
    edge_margin = min(left, 1 - right, top, 1 - bottom, 0.5)
    edge_score = max(0.0, min(edge_margin / 0.05, 1.0))

    occlusion_penalty = 0.0
    for other in primary_at_frame:
        if other.track_id == pos.track_id:
            continue
        iou = _bbox_iou(
            (pos.x, pos.y, pos.width, pos.height),
            (other.x, other.y, other.width, other.height),
        )
        occlusion_penalty = max(occlusion_penalty, iou)
    occlusion_score = max(0.0, 1.0 - occlusion_penalty / CROP_OCCLUSION_IOU_HARD)

    # Laplacian variance as blur proxy (on the grayscale crop)
    try:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        blur_score = min(lap_var / 200.0, 1.0)
    except cv2.error:
        blur_score = 0.0

    return float(
        height_score * confidence_score * edge_score * occlusion_score * blur_score,
    )


def _reject_crop_hard(pos: PlayerPosition, primary_at_frame: list[PlayerPosition]) -> bool:
    """Only reject on the lenient HARVEST gate: tiny box, edge clip, total occlusion."""
    if pos.height < CROP_MIN_HEIGHT_FRAC:
        return True
    if pos.confidence < CROP_MIN_CONFIDENCE:
        return True
    left = pos.x - pos.width / 2
    right = pos.x + pos.width / 2
    top = pos.y - pos.height / 2
    bottom = pos.y + pos.height / 2
    if (left < CROP_EDGE_MARGIN_FRAC or right > 1 - CROP_EDGE_MARGIN_FRAC
            or top < CROP_EDGE_MARGIN_FRAC or bottom > 1 - CROP_EDGE_MARGIN_FRAC):
        return True
    for other in primary_at_frame:
        if other.track_id == pos.track_id:
            continue
        iou = _bbox_iou(
            (pos.x, pos.y, pos.width, pos.height),
            (other.x, other.y, other.width, other.height),
        )
        if iou > CROP_OCCLUSION_IOU_HARD:
            return True
    return False


def _select_window_frames(all_frames: list[int], n: int) -> list[int]:
    """Subsample up to n evenly-spaced frames (deterministic)."""
    if len(all_frames) <= n:
        return list(all_frames)
    idxs = np.linspace(0, len(all_frames) - 1, n).astype(int)
    return [all_frames[i] for i in idxs]


def cmd_extract(input_dir: Path, dry_run: bool = False) -> dict[str, Any]:
    """Read frames per video, dump crops, compute multi-crop median DINOv2 embeddings."""
    pairs = _load_candidate_pairs(input_dir)
    (input_dir / "crops").mkdir(parents=True, exist_ok=True)
    (input_dir / "embeddings").mkdir(parents=True, exist_ok=True)

    # Collect all (rally, track, window) combos we need, dedup'd.
    # A window is a set of frames; we index by rally + track_id + window hash.
    windows_by_rally: dict[str, dict[tuple[int, str], TrackWindow]] = defaultdict(dict)
    pair_windows: dict[str, tuple[tuple[int, str], tuple[int, str]]] = {}
    rally_meta: dict[str, dict[str, Any]] = {}
    video_groups: dict[str, set[str]] = defaultdict(set)

    for p in pairs:
        rally_meta[p.rally_id] = {
            "video_id": p.video_id,
            "video_fps": p.video_fps,
            "video_width": p.video_width,
            "video_height": p.video_height,
            "rally_start_ms": p.rally_start_ms,
        }
        video_groups[p.video_id].add(p.rally_id)
        for tw in (p.track_a, p.track_b):
            key = (tw.track_id, tw.window_key())
            windows_by_rally[p.rally_id][key] = tw
        pair_windows[p.pair_id] = (
            (p.track_a.track_id, p.track_a.window_key()),
            (p.track_b.track_id, p.track_b.window_key()),
        )

    n_videos = len(video_groups)
    logger.info(
        "Extract plan: %d pairs → %d videos, %d rallies, %d unique track-windows",
        len(pairs), n_videos, len(windows_by_rally),
        sum(len(v) for v in windows_by_rally.values()),
    )
    if dry_run:
        return {"n_pairs": len(pairs), "n_videos": n_videos, "dry_run": True}

    stats = {
        "n_videos_processed": 0,
        "n_videos_missing": 0,
        "n_rallies_processed": 0,
        "n_crops_saved": 0,
        "n_crops_skipped_reject": 0,
        "n_embeddings_computed": 0,
        "n_embeddings_skipped_insufficient": 0,
    }

    for vidx, (video_id, rally_ids) in enumerate(
        sorted(video_groups.items()), start=1,
    ):
        video_path = get_video_path(video_id)
        if video_path is None or not video_path.exists():
            logger.warning(
                "  [%d/%d] video %s missing — skipping %d rallies",
                vidx, n_videos, video_id[:8], len(rally_ids),
            )
            stats["n_videos_missing"] += 1
            continue
        logger.info(
            "  [%d/%d] video %s → %s (%d rallies)",
            vidx, n_videos, video_id[:8], video_path.name, len(rally_ids),
        )

        for rally_id in sorted(rally_ids):
            meta = rally_meta[rally_id]
            windows = windows_by_rally[rally_id]
            all_frames_needed: set[int] = set()
            for tw in windows.values():
                all_frames_needed.update(tw.window_frames)

            # Build a lookup for PlayerPosition per (frame, track) from DB
            # (we need to re-query positions — faster than pickling them in jsonl).
            positions, primary_ids = _load_rally_positions(rally_id)
            if not positions:
                logger.warning(
                    "    rally %s has no stored positions — skip", rally_id[:8],
                )
                continue
            pos_by_key: dict[tuple[int, int], PlayerPosition] = {
                (p.frame_number, p.track_id): p for p in positions
            }
            # Include ALL primary tracks in the rally for the occlusion check,
            # not just the tracks we're currently harvesting — bbox IoU must
            # see every player to correctly spot near-total overlap.
            primary_set = set(primary_ids) | {tw.track_id for tw in windows.values()}
            primary_at_frame: dict[int, list[PlayerPosition]] = defaultdict(list)
            for pos in positions:
                if pos.track_id in primary_set:
                    primary_at_frame[pos.frame_number].append(pos)

            # Check which crops are already on disk; only request frames we need.
            missing_frames: set[int] = set()
            for tw in windows.values():
                for f in tw.window_frames:
                    if not _crop_path(input_dir, rally_id, tw.track_id, f).exists():
                        missing_frames.add(f)

            frames_by_rally_frame: dict[int, NDArray[np.uint8]] = {}
            if missing_frames:
                frames_by_rally_frame = _iterate_rally_frames(
                    video_path=video_path,
                    rally_start_ms=float(meta["rally_start_ms"]),
                    video_fps=meta["video_fps"],
                    rally_frames=missing_frames,
                )

            # Save crops for each window's frames.
            rally_crops_saved = 0
            rally_crops_rejected = 0
            for tw in windows.values():
                for f in tw.window_frames:
                    crop_file = _crop_path(input_dir, rally_id, tw.track_id, f)
                    if crop_file.exists():
                        continue
                    ppos = pos_by_key.get((f, tw.track_id))
                    if ppos is None:
                        continue
                    if _reject_crop_hard(ppos, primary_at_frame.get(f, [])):
                        rally_crops_rejected += 1
                        continue
                    frame = frames_by_rally_frame.get(f)
                    if frame is None:
                        continue
                    crop = extract_bbox_crop(
                        frame,
                        (ppos.x, ppos.y, ppos.width, ppos.height),
                        meta["video_width"], meta["video_height"],
                    )
                    if crop is None:
                        continue
                    crop_file.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(
                        str(crop_file), crop,
                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
                    )
                    rally_crops_saved += 1

            stats["n_crops_saved"] += rally_crops_saved
            stats["n_crops_skipped_reject"] += rally_crops_rejected

            # Compute multi-crop median embedding per window.
            for tw in windows.values():
                emb_file = _emb_path(input_dir, rally_id, tw)
                if emb_file.exists():
                    stats["n_embeddings_computed"] += 1
                    continue
                emb = _compute_median_embedding(
                    input_dir, rally_id, tw, pos_by_key, primary_at_frame,
                )
                if emb is None:
                    stats["n_embeddings_skipped_insufficient"] += 1
                    continue
                emb_file.parent.mkdir(parents=True, exist_ok=True)
                np.save(emb_file, emb)
                stats["n_embeddings_computed"] += 1

            stats["n_rallies_processed"] += 1
            logger.info(
                "    rally %s — crops saved=%d rejected=%d, windows=%d",
                rally_id[:8], rally_crops_saved, rally_crops_rejected, len(windows),
            )

        stats["n_videos_processed"] += 1

    stats_path = input_dir / "extract_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    logger.info("Extract stats: %s", stats)
    return stats


def _load_rally_positions(
    rally_id: str,
) -> tuple[list[PlayerPosition], list[int]]:
    """Load positions + primary_track_ids for the rally."""
    query = """
        SELECT positions_json, primary_track_ids
        FROM player_tracks WHERE rally_id = %s
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, [rally_id])
            row = cur.fetchone()
            if row is None or not row[0]:
                return [], []
            pos_json = cast(list[dict[str, Any]], row[0])
            primary = list(cast(list[int], row[1]) or [])
    positions = [
        PlayerPosition(
            frame_number=p["frameNumber"],
            track_id=p["trackId"],
            x=p["x"],
            y=p["y"],
            width=p["width"],
            height=p["height"],
            confidence=p["confidence"],
        )
        for p in pos_json
    ]
    return positions, primary


def _compute_median_embedding(
    input_dir: Path,
    rally_id: str,
    window: TrackWindow,
    pos_by_key: dict[tuple[int, int], PlayerPosition],
    primary_at_frame: dict[int, list[PlayerPosition]],
) -> NDArray[np.floating] | None:
    """Load saved crops, quality-score, drop bottom 25%, embed, median."""
    selected_frames = _select_window_frames(window.window_frames, N_CROPS_PER_WINDOW)
    loaded: list[tuple[int, NDArray[np.uint8], float]] = []
    for f in selected_frames:
        crop_file = _crop_path(input_dir, rally_id, window.track_id, f)
        if not crop_file.exists():
            continue
        crop_raw = cv2.imread(str(crop_file), cv2.IMREAD_COLOR)
        if crop_raw is None:
            continue
        crop = np.asarray(crop_raw, dtype=np.uint8)
        pos = pos_by_key.get((f, window.track_id))
        if pos is None:
            continue
        quality = _quality_score(pos, primary_at_frame.get(f, []), crop)
        loaded.append((f, crop, quality))

    if len(loaded) < MIN_CROPS_FOR_EMBEDDING:
        return None

    loaded.sort(key=lambda t: t[2])
    drop_n = int(len(loaded) * QUALITY_DROP_FRAC)
    survivors = loaded[drop_n:]
    if len(survivors) < MIN_CROPS_FOR_EMBEDDING:
        survivors = loaded  # fall back to all crops if drop left us too thin

    crops_only = [c for _, c, _ in survivors]
    embeddings = extract_backbone_features(crops_only)
    median = np.median(embeddings, axis=0)
    norm = float(np.linalg.norm(median))
    if norm < 1e-8:
        return None
    result: NDArray[np.floating] = (median / norm).astype(np.float32)
    return result


# ---------------------------------------------------------------------------
# Render command — contact sheet with 224×224 thumbnails
# ---------------------------------------------------------------------------


CONTACT_SHEET_CSS = """
body { font-family: -apple-system, system-ui, sans-serif; margin: 16px; background: #0f1218; color: #e8ecf4; }
h1 { margin-top: 0; }
nav { position: sticky; top: 0; background: #0f1218; padding: 8px 0 12px; border-bottom: 1px solid #2a3040; margin-bottom: 16px; z-index: 10; }
nav .row { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-bottom: 6px; }
nav a { color: #6aa9ff; text-decoration: none; font-weight: 600; }
nav a:hover { text-decoration: underline; }
.badge { display: inline-block; background: #2a3040; color: #b8c2d8; padding: 2px 8px; border-radius: 8px; font-size: 12px; margin-left: 4px; }
.badge.reject { background: #4b1d22; color: #ff8a93; }
nav .controls { margin-left: auto; display: flex; gap: 8px; align-items: center; }
nav button { background: #2a3040; color: #e8ecf4; border: 1px solid #3a4256; padding: 6px 12px; border-radius: 6px; cursor: pointer; font-weight: 600; font-size: 12px; }
nav button:hover { background: #3a4256; }
nav button.primary { background: #6aa9ff; color: #0f1218; border-color: #6aa9ff; }
nav button.primary:hover { background: #84bbff; }
.stats { font-size: 12px; color: #b8c2d8; }
.stats strong { color: #e8ecf4; }
details.instructions { background: #1a2030; border: 1px solid #2a3040; border-radius: 8px; padding: 8px 18px; margin-bottom: 24px; font-size: 14px; line-height: 1.6; }
details.instructions[open] { padding: 14px 18px; }
details.instructions > summary { cursor: pointer; padding: 6px 0; user-select: none; color: #e8ecf4; list-style: none; }
details.instructions > summary::-webkit-details-marker { display: none; }
details.instructions > summary::before { content: "▶ "; color: #6aa9ff; display: inline-block; width: 1em; transition: transform 0.15s; }
details.instructions[open] > summary::before { transform: rotate(90deg); }
details.instructions > summary:hover { color: #6aa9ff; }
details.instructions h3 { margin: 12px 0 4px; font-size: 13px; color: #6aa9ff; text-transform: uppercase; letter-spacing: 0.5px; }
details.instructions code { background: #0f1218; padding: 2px 6px; border-radius: 4px; font-family: ui-monospace, SF Mono, monospace; font-size: 12px; }
section { margin-bottom: 40px; }
section h2 { border-bottom: 2px solid #2a3040; padding-bottom: 4px; }
section h2 .tier-help { font-weight: 400; font-size: 13px; color: #8492b0; margin-left: 10px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(540px, 1fr)); gap: 12px; }
.card { background: #161a24; border: 1px solid #2a3040; border-radius: 8px; padding: 10px; position: relative; transition: opacity 0.15s, border-color 0.15s; }
.card.rejected { border-color: #f87171; opacity: 0.55; }
.card.rejected::after { content: "REJECTED"; position: absolute; top: 8px; right: 8px; background: #f87171; color: #0f1218; padding: 3px 8px; border-radius: 4px; font-weight: 700; font-size: 11px; letter-spacing: 1px; }
.card .crops { display: flex; gap: 8px; align-items: flex-start; }
.card img { width: 224px; height: 224px; object-fit: cover; border: 1px solid #2a3040; border-radius: 4px; }
.card .meta { font-size: 12px; color: #b8c2d8; margin-top: 8px; font-family: ui-monospace, SF Mono, monospace; }
.card .meta .label { color: #6aa9ff; }
.card .oversample { background: #ffcc00; color: #161a24; padding: 1px 6px; border-radius: 4px; font-weight: 700; font-size: 11px; margin-left: 6px; }
.card .same { color: #4ade80; }
.card .diff { color: #f87171; }
.card .actions { margin-top: 10px; display: flex; gap: 8px; }
.card button.reject-btn { background: #2a3040; color: #e8ecf4; border: 1px solid #3a4256; padding: 6px 14px; border-radius: 6px; cursor: pointer; font-weight: 600; font-size: 12px; flex: 1; }
.card button.reject-btn:hover { background: #4b1d22; border-color: #f87171; color: #ff8a93; }
.card.rejected button.reject-btn { background: #4b1d22; color: #ff8a93; border-color: #f87171; }
.card.rejected button.reject-btn:hover { background: #2a3040; border-color: #3a4256; color: #e8ecf4; }
.filter-hidden { display: none !important; }
footer { margin-top: 40px; padding-top: 16px; border-top: 1px solid #2a3040; font-size: 13px; color: #8492b0; }
"""


def _sample_for_tier(
    pairs: list[PairRecord], tier: str, n: int, rng: random.Random,
) -> list[tuple[PairRecord, bool]]:
    """Return [(pair, is_oversampled)] where is_oversampled marks hard-convergence mid cases."""
    tier_pairs = [p for p in pairs if p.tier == tier]
    if tier == "mid":
        n_hard = int(n * SHEET_MID_HARD_FRAC)
        n_standard = n - n_hard
        sorted_by_difficulty = sorted(
            tier_pairs, key=lambda p: p.difficulty_score, reverse=True,
        )
        top_third_cutoff = max(1, len(sorted_by_difficulty) // 3)
        hard_pool = sorted_by_difficulty[:top_third_cutoff]
        hard_ids = {p.pair_id for p in hard_pool}
        standard_pool = [p for p in tier_pairs if p.pair_id not in hard_ids]
        hard_sample = rng.sample(hard_pool, min(n_hard, len(hard_pool)))
        standard_sample = rng.sample(standard_pool, min(n_standard, len(standard_pool)))
        return [(p, True) for p in hard_sample] + [(p, False) for p in standard_sample]
    if tier == "gold":
        sample = tier_pairs[:n]  # gold is small; take all in order
        return [(p, False) for p in sample]
    rng.shuffle(tier_pairs)
    return [(p, False) for p in tier_pairs[:n]]


def _pick_display_frame(
    input_dir: Path,
    rally_id: str,
    track: TrackWindow,
    preferred: int | None,
) -> int | None:
    """Pick a frame whose crop is actually on disk.

    Preference order:
    1. The requested `preferred` frame if its crop file exists.
    2. The on-disk frame closest to `preferred` (or to the window midpoint
       if no preference).
    3. None if no crop file exists anywhere in the window.
    """
    frames = track.window_frames
    if not frames:
        return None
    # Which frames have a crop on disk?
    available = [
        f for f in frames
        if _crop_path(input_dir, rally_id, track.track_id, f).exists()
    ]
    if not available:
        return None
    anchor = preferred if preferred is not None else frames[len(frames) // 2]
    if (preferred is not None
            and _crop_path(input_dir, rally_id, track.track_id, preferred).exists()):
        return preferred
    return min(available, key=lambda f: abs(f - anchor))


def _render_pair_card(
    input_dir: Path,
    thumbs_dir: Path,
    reports_dir: Path,
    pair: PairRecord,
    is_oversampled: bool,
) -> str:
    """Generate HTML for one pair card. Writes thumbnail JPEGs."""
    # Pick canonical display frame per tier — fall back to any on-disk crop
    # if the preferred frame was rejected at extract time.
    conv = pair.convergence_frame
    if pair.tier in ("gold", "mid") and conv is not None:
        pref_a = min(pair.track_a.window_frames, key=lambda f: abs(f - conv))
        pref_b = min(pair.track_b.window_frames, key=lambda f: abs(f - conv))
    else:
        pref_a = pair.track_a.window_frames[len(pair.track_a.window_frames) // 2]
        pref_b = pair.track_b.window_frames[len(pair.track_b.window_frames) // 2]
    frame_a_opt = _pick_display_frame(input_dir, pair.rally_id, pair.track_a, pref_a)
    frame_b_opt = _pick_display_frame(input_dir, pair.rally_id, pair.track_b, pref_b)
    frame_a = frame_a_opt if frame_a_opt is not None else pref_a
    frame_b = frame_b_opt if frame_b_opt is not None else pref_b
    crop_missing = frame_a_opt is None or frame_b_opt is None

    thumb_a = thumbs_dir / f"{pair.pair_id}_a.jpg"
    thumb_b = thumbs_dir / f"{pair.pair_id}_b.jpg"
    _write_thumb(input_dir, pair.rally_id, pair.track_a.track_id, frame_a, thumb_a)
    _write_thumb(input_dir, pair.rally_id, pair.track_b.track_id, frame_b, thumb_b)

    # Cosine sim between median embeddings.
    emb_a_path = _emb_path(input_dir, pair.rally_id, pair.track_a)
    emb_b_path = _emb_path(input_dir, pair.rally_id, pair.track_b)
    cos_sim_str = "?"
    if emb_a_path.exists() and emb_b_path.exists():
        emb_a = np.load(emb_a_path)
        emb_b = np.load(emb_b_path)
        cos_sim_str = f"{_cos_sim(emb_a, emb_b):+.3f}"

    label_class = "same" if pair.label == "same" else "diff"
    oversample_badge = '<span class="oversample">HARD</span>' if is_oversampled else ""
    missing_badge = (
        '<span class="oversample" style="background:#f87171;color:#fff">NO CROPS</span>'
        if crop_missing else ""
    )

    thumb_a_rel = thumb_a.relative_to(reports_dir).as_posix()
    thumb_b_rel = thumb_b.relative_to(reports_dir).as_posix()

    safe_pid = html.escape(pair.pair_id)
    return f"""
<div class="card" id="{safe_pid}" data-pair="{safe_pid}" data-tier="{pair.tier}">
  <div class="crops">
    <img src="{thumb_a_rel}" alt="track_a">
    <img src="{thumb_b_rel}" alt="track_b">
  </div>
  <div class="meta">
    <div><span class="label">pair_id:</span> {safe_pid}{oversample_badge}{missing_badge}</div>
    <div><span class="label">tier:</span> {pair.tier} · <span class="{label_class}">{pair.label}</span> · split={pair.split}</div>
    <div><span class="label">rally:</span> {pair.rally_id[:8]} · video={pair.video_id[:8]}</div>
    <div><span class="label">frames:</span> a={frame_a} (t{pair.track_a.track_id} can={pair.track_a.canonical_id} team={pair.track_a.team})</div>
    <div><span class="label">       </span> b={frame_b} (t{pair.track_b.track_id} can={pair.track_b.canonical_id} team={pair.track_b.team})</div>
    <div><span class="label">conv_frame:</span> {pair.convergence_frame} · dist={pair.distance_at_convergence} · near_net={pair.near_net}</div>
    <div><span class="label">difficulty:</span> {pair.difficulty_score:.3f} · <span class="label">cos_sim(median):</span> {cos_sim_str}</div>
  </div>
  <div class="actions">
    <button class="reject-btn" onclick="toggleReject('{safe_pid}')" type="button">Reject</button>
  </div>
</div>
"""


def _write_thumb(
    input_dir: Path,
    rally_id: str,
    track_id: int,
    frame: int,
    thumb_path: Path,
) -> None:
    """Always regenerate the thumb so post-mine fixes to the display-frame
    picker are reflected on re-render. Cost is negligible (~1500 thumbs × ~1ms)."""
    thumb_path.parent.mkdir(parents=True, exist_ok=True)
    crop_path = _crop_path(input_dir, rally_id, track_id, frame)
    if not crop_path.exists():
        # Write a placeholder (blank 224x224) so the sheet doesn't 404.
        placeholder = np.zeros((THUMB_SIZE, THUMB_SIZE, 3), dtype=np.uint8)
        cv2.putText(
            placeholder, "missing", (30, 120),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2,
        )
        cv2.imwrite(str(thumb_path), placeholder)
        return
    crop = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
    if crop is None:
        return
    resized = cv2.resize(crop, (THUMB_SIZE, THUMB_SIZE))
    cv2.imwrite(str(thumb_path), resized, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])


def _self_check_gold(pairs: list[PairRecord], n: int = 10) -> list[str]:
    """Sanity-check: sampled gold pairs must be same-team different-canonical."""
    gold = [p for p in pairs if p.tier == "gold"]
    if not gold:
        return ["SELF-CHECK: no gold pairs found."]
    sample = random.Random(SPLIT_SEED).sample(gold, min(n, len(gold)))
    issues: list[str] = []
    for p in sample:
        if p.track_a.team != p.track_b.team:
            issues.append(
                f"{p.pair_id}: cross-team ({p.track_a.team} vs {p.track_b.team})",
            )
        if p.track_a.canonical_id == p.track_b.canonical_id:
            issues.append(
                f"{p.pair_id}: same canonical ({p.track_a.canonical_id})",
            )
    return issues


def cmd_render(
    input_dir: Path,
    reports_dir: Path,
    n_per_tier: int = SHEET_N_PER_TIER,
) -> dict[str, Any]:
    pairs = _load_candidate_pairs(input_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    thumbs_dir = reports_dir / "thumbs"

    # Automated self-check on gold.
    issues = _self_check_gold(pairs)
    if issues:
        logger.error("Gold self-check found %d issues:", len(issues))
        for msg in issues:
            logger.error("  - %s", msg)
        logger.error("Fix harvest logic before proceeding to user review.")
    else:
        logger.info("Gold self-check: PASS (10 random gold pairs are same-team different-canonical)")

    rng = random.Random(SPLIT_SEED)
    tier_order = ["gold", "mid", "positive", "easy_neg"]
    tier_samples: dict[str, list[tuple[PairRecord, bool]]] = {}
    for tier in tier_order:
        tier_samples[tier] = _sample_for_tier(pairs, tier, n_per_tier, rng)
        logger.info(
            "  sampled %d %s pairs (available=%d)",
            len(tier_samples[tier]), tier,
            sum(1 for p in pairs if p.tier == tier),
        )

    # Per-tier "what to look for" hints (shown under each tier heading).
    tier_hints = {
        "gold": (
            "Two DIFFERENT teammates caught mid-swap. Look for similar kit + "
            "different faces/builds. REJECT if: same person, clearly different "
            "teams (different kits), or crops unrecognizable."
        ),
        "mid": (
            "Two teammates converging near each other (e.g., at the net during "
            "a block). Look for similar kit + different bodies. REJECT if: same "
            "person on both sides, off-court spectator, or visually different "
            "teams (means team assignment is broken)."
        ),
        "positive": (
            "The SAME player at two moments in the rally. Look for obvious "
            "identity match (same kit, build, face). REJECT if: tracker drifted "
            "to a different person (the two crops are two different people)."
        ),
        "easy_neg": (
            "Players from OPPOSING teams. Look for visibly different kits. "
            "REJECT if: same kit color on both (team assignment is broken) or "
            "same person."
        ),
    }

    n_total = sum(len(v) for v in tier_samples.values())
    rendered_ids_by_tier = {
        t: [p.pair_id for p, _ in tier_samples[t]] for t in tier_order
    }

    # Build HTML
    parts: list[str] = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'>",
        "<title>Within-Team ReID Contact Sheet</title>",
        f"<style>{CONTACT_SHEET_CSS}</style>",
        "</head><body>",
        "<h1>Within-Team ReID — Contact Sheet</h1>",
        f"<p>{n_total} pairs across 4 tiers. Crops shown at native 224×224 "
        "(DINOv2 input resolution). Click <strong>Reject</strong> on any "
        "card that's incorrect; state persists across page reloads. When done, "
        "click <strong>Export rejections</strong> (top-right) to save "
        "<code>rejected_pairs.json</code> into the training-data folder.</p>",
        "<nav>",
        '  <div class="row">',
    ]
    for tier in tier_order:
        parts.append(
            f'    <a href="#{tier}">{tier}</a>'
            f'<span class="badge" id="badge-{tier}">{len(tier_samples[tier])}</span>'
            f'<span class="badge reject" id="rej-{tier}" style="display:none">0 rej</span>',
        )
    parts.append("  </div>")
    parts.append('  <div class="row">')
    parts.append(
        '    <span class="stats" id="stats">'
        '<strong id="stat-reviewed">0</strong> reviewed · '
        '<strong id="stat-rejected">0</strong> rejected · '
        '<strong id="stat-pending">' + str(n_total) + '</strong> pending'
        '</span>'
    )
    parts.append('    <div class="controls">')
    parts.append('      <button type="button" onclick="setFilter(\'all\')">All</button>')
    parts.append('      <button type="button" onclick="setFilter(\'rejected\')">Rejected</button>')
    parts.append('      <button type="button" onclick="setFilter(\'unreviewed\')">Unreviewed</button>')
    parts.append('      <button type="button" onclick="importRejections()">Import…</button>')
    parts.append('      <button type="button" class="primary" onclick="exportRejections()">Export rejections</button>')
    parts.append('    </div>')
    parts.append('  </div>')
    parts.append("</nav>")

    # How-to-review panel (collapsible, collapsed by default)
    parts.append('<details class="instructions">')
    parts.append('<summary><strong>How to review</strong> (click to expand)</summary>')
    parts.append(
        '<p>You\'re verifying that auto-harvested training pairs actually match '
        'their tier. Focus on gold + mid first — those have strict rejection '
        'gates that decide Session 3 readiness.</p>'
    )
    parts.append('<h3>What to look for, per tier</h3>')
    for tier in tier_order:
        parts.append(
            f'<p><strong>{tier}</strong>: {html.escape(tier_hints[tier])}</p>',
        )
    parts.append('<h3>Acceptance gate (Session 3 readiness)</h3>')
    parts.append(
        '<p>Gold rejection < 5 % AND mid rejection < 15 % → Session 3 greenlit. '
        'Anything higher means the harvest rules need fixing before training. '
        'Positive and easy-neg tiers don\'t gate Session 3, but rejections there '
        'still improve training quality.</p>'
    )
    parts.append(
        '<h3>Flow</h3>'
        '<p>1. Scroll through the pairs. '
        '2. Click <strong>Reject</strong> on any card that fails the tier '
        'criterion. 3. When done, click <strong>Export rejections</strong> to '
        'download <code>rejected_pairs.json</code>. 4. Save it to '
        f'<code>{input_dir}/rejected_pairs.json</code> (the browser will prompt). '
        '5. Run <code>uv run python scripts/harvest_within_team_pairs.py summary '
        f'--input-dir {input_dir}</code> to compute rejection rates.</p>'
    )
    parts.append('</details>')

    if issues:
        parts.append('<section><h2 style="color:#f87171">⚠ Self-check issues</h2>')
        parts.append('<ul>')
        for msg in issues:
            parts.append(f'<li>{html.escape(msg)}</li>')
        parts.append('</ul></section>')

    for tier in tier_order:
        parts.append(f'<section id="{tier}">')
        parts.append(
            f'<h2>{tier.replace("_", " ")} '
            f'<span class="badge">{len(tier_samples[tier])}</span>'
            f'<span class="tier-help">{html.escape(tier_hints[tier])}</span>'
            '</h2>',
        )
        parts.append('<div class="grid">')
        for p, is_oversampled in tier_samples[tier]:
            parts.append(_render_pair_card(
                input_dir, thumbs_dir, reports_dir, p, is_oversampled,
            ))
        parts.append('</div></section>')

    parts.append('<footer>Rejection-state is kept in your browser '
                 '(<code>localStorage</code>) so reloading the page doesn\'t '
                 'lose your work. Export + save as '
                 f'<code>{input_dir}/rejected_pairs.json</code>, then run '
                 '<code>summary</code> to compute rejection rates.</footer>')

    # Client-side JS: rejection toggle + localStorage + filter + import/export
    rendered_js = (
        "const RENDERED = "
        + json.dumps(rendered_ids_by_tier)
        + ";\n"
        "const TIERS = " + json.dumps(tier_order) + ";\n"
        "const STORAGE_KEY = 'within_team_reid_rejections_v1';\n"
        "const ALL_IDS = TIERS.flatMap(t => RENDERED[t]);\n"
        "function loadRej() {\n"
        "  try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}'); }\n"
        "  catch (e) { return {}; }\n"
        "}\n"
        "function saveRej(r) { localStorage.setItem(STORAGE_KEY, JSON.stringify(r)); }\n"
        "function toggleReject(pairId) {\n"
        "  const r = loadRej();\n"
        "  if (r[pairId]) delete r[pairId]; else r[pairId] = 1;\n"
        "  saveRej(r);\n"
        "  applyState();\n"
        "}\n"
        "function applyState() {\n"
        "  const r = loadRej();\n"
        "  document.querySelectorAll('.card').forEach(c => {\n"
        "    const pid = c.dataset.pair;\n"
        "    const isRej = !!r[pid];\n"
        "    c.classList.toggle('rejected', isRej);\n"
        "    const btn = c.querySelector('.reject-btn');\n"
        "    if (btn) btn.textContent = isRej ? 'Rejected — click to undo' : 'Reject';\n"
        "  });\n"
        "  let rejCount = 0;\n"
        "  TIERS.forEach(t => {\n"
        "    const n = RENDERED[t].filter(p => r[p]).length;\n"
        "    rejCount += n;\n"
        "    const badge = document.getElementById('rej-' + t);\n"
        "    if (badge) {\n"
        "      badge.textContent = n + ' rej';\n"
        "      badge.style.display = n > 0 ? 'inline-block' : 'none';\n"
        "    }\n"
        "  });\n"
        "  const reviewed = Object.keys(r).length;\n"
        "  const pending = ALL_IDS.length - reviewed;\n"
        "  document.getElementById('stat-reviewed').textContent = reviewed;\n"
        "  document.getElementById('stat-rejected').textContent = rejCount;\n"
        "  document.getElementById('stat-pending').textContent = pending;\n"
        "}\n"
        "function setFilter(mode) {\n"
        "  const r = loadRej();\n"
        "  document.querySelectorAll('.card').forEach(c => {\n"
        "    const pid = c.dataset.pair;\n"
        "    const isRej = !!r[pid];\n"
        "    let show = true;\n"
        "    if (mode === 'rejected') show = isRej;\n"
        "    else if (mode === 'unreviewed') show = !isRej;\n"
        "    c.classList.toggle('filter-hidden', !show);\n"
        "  });\n"
        "}\n"
        "async function exportRejections() {\n"
        "  const r = loadRej();\n"
        "  const ids = Object.keys(r).sort();\n"
        "  const served = window.location.protocol === 'http:' "
        "|| window.location.protocol === 'https:';\n"
        "  if (served) {\n"
        "    try {\n"
        "      const resp = await fetch('/save_rejections', {\n"
        "        method: 'POST',\n"
        "        headers: {'Content-Type': 'application/json'},\n"
        "        body: JSON.stringify(ids),\n"
        "      });\n"
        "      if (!resp.ok) throw new Error('HTTP ' + resp.status);\n"
        "      const j = await resp.json();\n"
        "      const gate = j.gate || {};\n"
        "      alert('Saved ' + j.saved_count + ' rejections to\\n' + j.saved_to"
        " + '\\n\\nSession 3 green light: ' "
        "+ (gate.session3_ok ? 'YES' : 'NO') + '\\n\\n"
        "See reports/within_team_reid/harvest_summary.md for details.');\n"
        "      return;\n"
        "    } catch (err) {\n"
        "      alert('Server save failed: ' + err.message + "
        "'\\nFalling back to download — save manually to training_data/"
        "within_team_reid/rejected_pairs.json.');\n"
        "    }\n"
        "  }\n"
        "  const blob = new Blob([JSON.stringify(ids, null, 2)], "
        "{type: 'application/json'});\n"
        "  const url = URL.createObjectURL(blob);\n"
        "  const a = document.createElement('a');\n"
        "  a.href = url; a.download = 'rejected_pairs.json';\n"
        "  a.click();\n"
        "  URL.revokeObjectURL(url);\n"
        "}\n"
        "function importRejections() {\n"
        "  const inp = document.createElement('input');\n"
        "  inp.type = 'file'; inp.accept = 'application/json';\n"
        "  inp.onchange = async (e) => {\n"
        "    const f = e.target.files[0]; if (!f) return;\n"
        "    const text = await f.text();\n"
        "    try {\n"
        "      const ids = JSON.parse(text);\n"
        "      const r = {};\n"
        "      if (Array.isArray(ids)) ids.forEach(pid => r[pid] = 1);\n"
        "      saveRej(r); applyState();\n"
        "      alert('Imported ' + Object.keys(r).length + ' rejections.');\n"
        "    } catch (err) { alert('Import failed: ' + err.message); }\n"
        "  };\n"
        "  inp.click();\n"
        "}\n"
        "applyState();\n"
    )
    parts.append(f'<script>{rendered_js}</script>')
    parts.append("</body></html>")

    sheet_path = reports_dir / "contact_sheet.html"
    sheet_path.write_text("\n".join(parts))
    logger.info("Wrote %s", sheet_path)

    return {
        "sheet_path": str(sheet_path),
        "n_rendered": {t: len(tier_samples[t]) for t in tier_order},
        "self_check_issues": issues,
    }


# ---------------------------------------------------------------------------
# Summary command — rejection rates + gate verdict
# ---------------------------------------------------------------------------


def cmd_summary(
    input_dir: Path, reports_dir: Path, rejected_path: Path | None,
) -> dict[str, Any]:
    pairs = _load_candidate_pairs(input_dir)
    rejected_ids: set[str] = set()
    if rejected_path and rejected_path.exists():
        data = json.loads(rejected_path.read_text())
        if isinstance(data, list):
            rejected_ids = set(data)

    counts: dict[str, int] = defaultdict(int)
    rejected_counts: dict[str, int] = defaultdict(int)
    for p in pairs:
        counts[p.tier] += 1
        if p.pair_id in rejected_ids:
            rejected_counts[p.tier] += 1

    rates = {
        tier: (rejected_counts[tier] / counts[tier]) if counts[tier] else 0.0
        for tier in counts
    }

    gold_pass = rates.get("gold", 0) < GOLD_REJECT_GATE
    mid_pass = rates.get("mid", 0) < MID_REJECT_GATE
    mid_borderline = MID_REJECT_GATE <= rates.get("mid", 0) < 0.25

    gold_count = counts.get("gold", 0)
    gold_volume_pass = gold_count >= 150
    session3_ok = gold_pass and mid_pass and gold_volume_pass
    gold_volume_line = "PASS" if gold_volume_pass else f"FAIL ({gold_count} actual)"
    mid_gate_result = "PASS" if mid_pass else ("BORDERLINE" if mid_borderline else "FAIL")

    lines = [
        "# Session 2 Harvest Summary",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        "",
        "## Per-tier counts",
        "",
        "| Tier | Harvested | Rejected | Rate | Gate |",
        "|---|---:|---:|---:|---|",
    ]
    for tier in ("gold", "mid", "positive", "easy_neg"):
        n = counts.get(tier, 0)
        rej = rejected_counts.get(tier, 0)
        rate = rates.get(tier, 0.0)
        if tier == "gold":
            gate = "PASS" if rate < GOLD_REJECT_GATE else "FAIL"
            gate_desc = f"<{GOLD_REJECT_GATE:.0%}"
        elif tier == "mid":
            gate = "PASS" if rate < MID_REJECT_GATE else (
                "BORDERLINE" if mid_borderline else "FAIL"
            )
            gate_desc = f"<{MID_REJECT_GATE:.0%}"
        else:
            gate = "-"
            gate_desc = ""
        lines.append(f"| {tier} | {n} | {rej} | {rate:.1%} | {gate} ({gate_desc}) |")

    split_train = sum(1 for p in pairs if p.split == "train")
    split_val = sum(1 for p in pairs if p.split == "val")
    train_videos = len({p.video_id for p in pairs if p.split == "train"})
    val_videos = len({p.video_id for p in pairs if p.split == "val"})

    lines += [
        "",
        "## Train/val split (video-level)",
        "",
        f"- Train: {split_train} pairs across {train_videos} videos",
        f"- Val: {split_val} pairs across {val_videos} videos",
        f"- Split seed: {SPLIT_SEED}, train fraction: {TRAIN_FRAC}",
        "",
        "## Session 3 readiness",
        "",
        f"- Gold tier rejection gate (<{GOLD_REJECT_GATE:.0%}): {'PASS' if gold_pass else 'FAIL'}",
        f"- Mid tier rejection gate (<{MID_REJECT_GATE:.0%}): {mid_gate_result}",
        f"- Gold volume ≥ 150: {gold_volume_line}",
        f"- Overall Session 3 green light: {'YES' if session3_ok else 'NO'}",
        "",
    ]

    if not gold_volume_pass:
        lines += [
            "### Action needed",
            "",
            "Gold tier is below the 150-pair Session 3 prerequisite. "
            "Schedule a Session 2b labeller-in-the-loop round: 10 rallies with "
            "dense near-net convergence × 10–15 hard pairs each. Defer Session 3 "
            "head-training until gold volume ≥ 150.",
            "",
        ]

    summary_path = reports_dir / "harvest_summary.md"
    summary_path.write_text("\n".join(lines))
    logger.info("Wrote %s", summary_path)

    return {
        "counts": dict(counts),
        "rejected_counts": dict(rejected_counts),
        "rates": rates,
        "gold_pass": gold_pass,
        "mid_pass": mid_pass,
        "gold_volume_pass": gold_volume_pass,
        "session3_ok": session3_ok,
    }


# ---------------------------------------------------------------------------
# Serve command — local HTTP server that saves rejections to the right path
# ---------------------------------------------------------------------------


def cmd_serve(
    input_dir: Path, reports_dir: Path, port: int = 8765,
) -> None:
    """Launch a tiny localhost HTTP server so the contact sheet's Export
    button POSTs rejections directly to
    `<input_dir>/rejected_pairs.json`. Running `summary` inline is automatic:
    the button's alert tells the user the gate verdict.
    """
    import http.server
    import socketserver
    from urllib.parse import urlparse

    reports_dir = reports_dir.resolve()
    input_dir = input_dir.resolve()
    rejected_path = input_dir / "rejected_pairs.json"

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(reports_dir), **kwargs)

        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: N802
            logger.info("  http: " + fmt, *args)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self.send_response(302)
                self.send_header("Location", "/contact_sheet.html")
                self.end_headers()
                return
            if parsed.path == "/load_rejections":
                if rejected_path.exists():
                    body = rejected_path.read_bytes()
                else:
                    body = b"[]"
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            super().do_GET()

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != "/save_rejections":
                self.send_response(404)
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            try:
                data = json.loads(raw or b"[]")
            except json.JSONDecodeError as exc:
                self._send_err(400, f"Bad JSON: {exc}")
                return
            if not isinstance(data, list) or not all(
                isinstance(x, str) for x in data
            ):
                self._send_err(400, "Expected a JSON array of pair_id strings.")
                return

            ids = sorted(set(data))
            rejected_path.write_text(json.dumps(ids, indent=2))
            gate = cmd_summary(input_dir, reports_dir, rejected_path)

            resp = json.dumps({
                "saved_count": len(ids),
                "saved_to": str(rejected_path),
                "gate": gate,
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        def _send_err(self, code: int, msg: str) -> None:
            body = msg.encode()
            self.send_response(code)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    with socketserver.TCPServer(("127.0.0.1", port), Handler) as httpd:
        url = f"http://127.0.0.1:{port}/contact_sheet.html"
        logger.info("Contact sheet + rejection-save server running at:")
        logger.info("  %s", url)
        logger.info("Rejections will be written to: %s", rejected_path)
        logger.info("Press Ctrl-C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            logger.info("Server stopped.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "cmd",
        choices=["mine", "extract", "render", "summary", "serve", "all"],
        help="Subcommand to run",
    )
    p.add_argument(
        "--output-dir", "--input-dir",
        dest="input_dir",
        type=Path,
        default=Path("training_data/within_team_reid"),
        help="Training data directory (output for `mine`, input for the rest)",
    )
    p.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports/within_team_reid"),
        help="Where contact_sheet.html + harvest_summary.md land",
    )
    p.add_argument(
        "--limit-rallies", type=int, default=0,
        help="Smoke-test: process only first N rallies in `mine`.",
    )
    p.add_argument(
        "--rejected", type=Path, default=None,
        help="Optional path to rejected_pairs.json (for `summary`).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="`extract`: report plan without running video I/O.",
    )
    p.add_argument(
        "--n-per-tier", type=int, default=SHEET_N_PER_TIER,
        help="`render`: pairs shown per tier (default 250).",
    )
    p.add_argument(
        "--port", type=int, default=8765,
        help="`serve`: port for the local contact-sheet + rejection-save server.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    t0 = time.time()

    if args.cmd in ("mine", "all"):
        cmd_mine(args.input_dir, limit_rallies=args.limit_rallies)
    if args.cmd in ("extract", "all"):
        cmd_extract(args.input_dir, dry_run=args.dry_run)
    if args.cmd in ("render", "all"):
        cmd_render(args.input_dir, args.reports_dir, n_per_tier=args.n_per_tier)
    if args.cmd == "summary":
        rejected = args.rejected or (args.input_dir / "rejected_pairs.json")
        cmd_summary(args.input_dir, args.reports_dir, rejected)
    if args.cmd == "serve":
        cmd_serve(args.input_dir, args.reports_dir, port=args.port)

    wall = time.time() - t0
    logger.info("Done. Wall: %.1f min", wall / 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
