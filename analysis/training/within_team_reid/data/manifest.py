"""Pair manifest loader for within-team ReID training.

Reads `candidate_pairs.jsonl`, drops `rejected_pairs.json`, splits by the
manifest's baked-in train/val field, and asserts the load-bearing invariants
that the training loop and losses depend on.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("within_team_reid.data.manifest")


@dataclass(frozen=True)
class TrackInfo:
    """One side of a pair (track_a or track_b)."""

    track_id: int
    canonical_id: int
    team: int
    window_frames: tuple[int, ...]


@dataclass(frozen=True)
class Pair:
    """One labelled pair from the harvest manifest.

    Identity label for SupCon = (video_id, track_a.canonical_id) or (video_id, track_b.canonical_id).
    Team label for teammate-margin = (rally_id, team).
    """

    pair_id: str
    tier: str            # "gold" | "mid" | "positive" | "easy_neg"
    label: str           # "same" | "different"
    split: str           # "train" | "val"
    rally_id: str
    video_id: str
    video_fps: float
    video_width: int
    video_height: int
    rally_start_ms: int
    track_a: TrackInfo
    track_b: TrackInfo
    convergence_frame: int | None
    near_net: bool
    difficulty_score: float


@dataclass
class ManifestData:
    train_pairs: list[Pair]
    val_pairs: list[Pair]
    rejected_ids: set[str]
    known_bad_prefixes: tuple[str, ...]
    config: dict


def _track_info_from_dict(d: dict) -> TrackInfo:
    return TrackInfo(
        track_id=int(d["track_id"]),
        canonical_id=int(d["canonical_id"]),
        team=int(d["team"]),
        window_frames=tuple(int(f) for f in d["window_frames"]),
    )


def _pair_from_dict(d: dict) -> Pair:
    return Pair(
        pair_id=d["pair_id"],
        tier=d["tier"],
        label=d["label"],
        split=d["split"],
        rally_id=d["rally_id"],
        video_id=d["video_id"],
        video_fps=float(d.get("video_fps") or 30.0),
        video_width=int(d.get("video_width") or 1920),
        video_height=int(d.get("video_height") or 1080),
        rally_start_ms=int(d.get("rally_start_ms") or 0),
        track_a=_track_info_from_dict(d["track_a"]),
        track_b=_track_info_from_dict(d["track_b"]),
        convergence_frame=int(d["convergence_frame"]) if d.get("convergence_frame") is not None else None,
        near_net=bool(d.get("near_net", False)),
        difficulty_score=float(d.get("difficulty_score") or 0.0),
    )


def _assert_pair_invariants(pair: Pair) -> None:
    """Assertions that the losses depend on. Failure = harvest bug.

    - For label=='same' (positive): canonical_a must equal canonical_b.
    - For tier in {'mid','gold'}: same team, different canonicals (load-bearing for teammate-margin).
    - For tier=='easy_neg': different teams (load-bearing for SupCon negative semantics).
    - track_id != track_id within same pair (harvest already enforces this for cross-track tiers).
      Note: positive pairs CAN have track_a.track_id == track_b.track_id (different time windows
      of the same track), which is intentional for temporal invariance — assertion skipped there.
    """
    if pair.label == "same":
        assert pair.track_a.canonical_id == pair.track_b.canonical_id, (
            f"positive pair {pair.pair_id} has different canonicals"
            f" ({pair.track_a.canonical_id} vs {pair.track_b.canonical_id})"
        )
    if pair.tier in {"mid", "gold"}:
        assert pair.track_a.team == pair.track_b.team, (
            f"hard-neg pair {pair.pair_id} crosses teams"
            f" ({pair.track_a.team} vs {pair.track_b.team})"
        )
        assert pair.track_a.canonical_id != pair.track_b.canonical_id, (
            f"hard-neg pair {pair.pair_id} has equal canonicals"
            f" ({pair.track_a.canonical_id} == {pair.track_b.canonical_id})"
        )
    if pair.tier == "easy_neg":
        assert pair.track_a.team != pair.track_b.team, (
            f"easy-neg pair {pair.pair_id} shares team"
            f" ({pair.track_a.team} == {pair.track_b.team})"
        )


def load_manifest(corpus_root: Path) -> ManifestData:
    """Load and validate the pair manifest, applying rejection + known-bad filters."""
    manifest_path = corpus_root / "manifest.json"
    pairs_path = corpus_root / "candidate_pairs.jsonl"
    rejected_path = corpus_root / "rejected_pairs.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json missing at {manifest_path}")
    if not pairs_path.exists():
        raise FileNotFoundError(f"candidate_pairs.jsonl missing at {pairs_path}")

    config = json.loads(manifest_path.read_text())
    known_bad_prefixes = tuple(config.get("known_bad_prefixes", []))

    rejected_ids: set[str] = set()
    if rejected_path.exists():
        rejected_ids = set(json.loads(rejected_path.read_text()))
    logger.info("Rejected pair count from rejected_pairs.json: %d", len(rejected_ids))

    train_pairs: list[Pair] = []
    val_pairs: list[Pair] = []
    n_dropped_rejected = 0
    n_dropped_known_bad = 0
    n_total = 0

    with pairs_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_total += 1
            d = json.loads(line)
            pair = _pair_from_dict(d)

            if pair.pair_id in rejected_ids:
                n_dropped_rejected += 1
                continue
            if any(pair.rally_id.startswith(p) for p in known_bad_prefixes):
                n_dropped_known_bad += 1
                continue

            _assert_pair_invariants(pair)
            if pair.split == "train":
                train_pairs.append(pair)
            elif pair.split == "val":
                val_pairs.append(pair)

    logger.info(
        "Manifest loaded: %d total pairs, %d train, %d val (dropped: %d rejected, %d known-bad)",
        n_total, len(train_pairs), len(val_pairs), n_dropped_rejected, n_dropped_known_bad,
    )
    _log_tier_breakdown("train", train_pairs)
    _log_tier_breakdown("val", val_pairs)

    return ManifestData(
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        rejected_ids=rejected_ids,
        known_bad_prefixes=known_bad_prefixes,
        config=config,
    )


def _log_tier_breakdown(label: str, pairs: list[Pair]) -> None:
    counts: dict[str, int] = {}
    for p in pairs:
        counts[p.tier] = counts.get(p.tier, 0) + 1
    logger.info("  %s tier breakdown: %s", label, dict(sorted(counts.items())))


def partition_by_tier(pairs: list[Pair]) -> dict[str, list[Pair]]:
    """Group pairs by tier for sampler use."""
    out: dict[str, list[Pair]] = {"gold": [], "mid": [], "positive": [], "easy_neg": []}
    for p in pairs:
        out.setdefault(p.tier, []).append(p)
    return out
