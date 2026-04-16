"""ReID diagnostic probe for gallery-detected swap events.

Given a pred-exchange swap (pred_new took over a GT that pred_old was tracking),
this module extracts pred_new's appearance around the swap frame and compares
it to every canonical player profile stored in match_analysis_json.
It tells us whether the ReID features had the signal to keep the correct
assignment — or not.

Three classifications:

  reid_had_signal        Δcost(correct − wrong) ≥ 0.08 in pred_new's favour
                         → structural fix: rebalance cost weights at this situation
  reid_blind             |Δcost| < 0.05
                         → needs pose/role features (ReID itself can't distinguish)
  reid_wrong_preference  Δcost ≥ 0.05 in the wrong direction
                         → ReID actively mis-identified (jersey change, lighting,
                           pose-dependent crop; needs data/model investigation)

The same PlayerAppearanceProfile / compute_appearance_similarity used by
match-players / repair-identities are used here — no new model, no new signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from rallycut.tracking.player_features import (
    PlayerAppearanceProfile,
    TrackAppearanceStats,
    compute_appearance_similarity,
    extract_appearance_features,
)
from rallycut.tracking.player_tracker import PlayerPosition

CLASS_REID_HAD_SIGNAL = "reid_had_signal"
CLASS_REID_BLIND = "reid_blind"
CLASS_REID_WRONG_PREFERENCE = "reid_wrong_preference"
CLASS_INSUFFICIENT_DATA = "insufficient_data"

DELTA_HAS_SIGNAL = 0.08  # difference in cost below which ReID has clear signal
DELTA_BLIND = 0.05       # difference below which ReID can't distinguish


@dataclass
class SwapProbeResult:
    """Per-swap-event ReID diagnostic."""

    rally_id: str
    swap_frame: int
    gt_track_id: int
    pred_old: int
    pred_new: int
    prior_gt_of_new: int | None
    # player_id → appearance cost (0-1; lower is better match)
    player_costs_pre_swap: dict[int, float]   # pred_new's appearance before swap
    player_costs_post_swap: dict[int, float]  # pred_new's appearance after swap
    # Spatial context
    spatial_distance_to_each_gt: dict[int, float]  # centroid distance at swap frame
    classification: str
    reasoning: str
    samples_used_pre: int
    samples_used_post: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "rallyId": self.rally_id,
            "swapFrame": self.swap_frame,
            "gtTrackId": self.gt_track_id,
            "predOld": self.pred_old,
            "predNew": self.pred_new,
            "priorGtOfNew": self.prior_gt_of_new,
            "playerCostsPreSwap": {str(k): v for k, v in self.player_costs_pre_swap.items()},
            "playerCostsPostSwap": {str(k): v for k, v in self.player_costs_post_swap.items()},
            "spatialDistanceToEachGt": {str(k): v for k, v in self.spatial_distance_to_each_gt.items()},
            "classification": self.classification,
            "reasoning": self.reasoning,
            "samplesUsedPre": self.samples_used_pre,
            "samplesUsedPost": self.samples_used_post,
        }


def load_player_profiles_from_match_analysis(
    match_analysis_json: dict,
) -> dict[int, PlayerAppearanceProfile]:
    """Parse match_analysis_json.playerProfiles into PlayerAppearanceProfile objects.

    Keys in JSON are string player_ids ("1", "2", ...); returns int-keyed dict.
    """
    raw = match_analysis_json.get("playerProfiles", {}) or {}
    result: dict[int, PlayerAppearanceProfile] = {}
    for pid_str, profile_dict in raw.items():
        pid = int(pid_str)
        result[pid] = PlayerAppearanceProfile.from_dict(profile_dict)
    return result


def get_rally_track_to_player(
    match_analysis_json: dict,
    rally_id: str,
) -> dict[int, int]:
    """Extract the pred_id → canonical_player_id mapping for a specific rally."""
    for rally_entry in match_analysis_json.get("rallies", []) or []:
        if rally_entry.get("rallyId") == rally_id:
            raw = rally_entry.get("trackToPlayer", {}) or {}
            return {int(k): int(v) for k, v in raw.items()}
    return {}


def _iterate_rally_frames(
    video_path: Path,
    rally_start_ms: float,
    video_fps: float,
    rally_frames: set[int],
) -> dict[int, np.ndarray]:
    """Read only the rally frames we need from the source video.

    `rally_frames` are rally-relative (0-indexed). The corresponding absolute
    video frame is `rally_start_frame + rally_frame`. Returns rally_frame →
    BGR ndarray.
    """
    rally_start_frame = int(round(rally_start_ms / 1000.0 * video_fps))
    abs_needed = sorted(rally_start_frame + f for f in rally_frames)
    if not abs_needed:
        return {}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {}

    result: dict[int, np.ndarray] = {}
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, abs_needed[0])
        cur = abs_needed[0]
        target_idx = 0
        while target_idx < len(abs_needed):
            target = abs_needed[target_idx]
            while cur < target:
                cap.read()  # skip
                cur += 1
                if cur > target + 10_000:  # safety
                    break
            ok, frame = cap.read()
            cur += 1
            if not ok:
                break
            if target == abs_needed[target_idx]:
                rally_frame = target - rally_start_frame
                result[rally_frame] = frame
                target_idx += 1
    finally:
        cap.release()
    return result


def _collect_stats_for_pred(
    pred_id: int,
    predictions: list[PlayerPosition],
    frame_range: range,
    frames_by_rally_frame: dict[int, np.ndarray],
) -> tuple[TrackAppearanceStats, int]:
    """Build a TrackAppearanceStats from pred positions in a frame window."""
    stats = TrackAppearanceStats(track_id=pred_id)
    pred_by_frame: dict[int, PlayerPosition] = {
        p.frame_number: p for p in predictions if p.track_id == pred_id
    }
    samples = 0
    for f in frame_range:
        pos = pred_by_frame.get(f)
        frame = frames_by_rally_frame.get(f)
        if pos is None or frame is None:
            continue
        h, w = frame.shape[:2]
        feats = extract_appearance_features(
            frame=frame,
            track_id=pred_id,
            frame_number=f,
            bbox=(pos.x, pos.y, pos.width, pos.height),
            frame_width=w,
            frame_height=h,
        )
        stats.features.append(feats)
        samples += 1
    stats.compute_averages()
    return stats, samples


def _spatial_distances(
    pred_new: int,
    swap_frame: int,
    predictions: list[PlayerPosition],
) -> dict[int, float]:
    """Euclidean distance from pred_new to every other primary pred at swap_frame."""
    preds_here = [p for p in predictions if p.frame_number == swap_frame]
    new_here = next((p for p in preds_here if p.track_id == pred_new), None)
    if new_here is None:
        return {}
    out: dict[int, float] = {}
    for p in preds_here:
        if p.track_id == pred_new:
            continue
        d = ((p.x - new_here.x) ** 2 + (p.y - new_here.y) ** 2) ** 0.5
        out[p.track_id] = d
    return out


def _classify(
    player_costs_post_swap: dict[int, float],
    correct_player: int | None,
    wrong_player: int | None,
) -> tuple[str, str]:
    """Produce a classification + human-readable reasoning string."""
    if correct_player is None or wrong_player is None:
        return (
            CLASS_INSUFFICIENT_DATA,
            f"canonical player mapping missing (correct={correct_player}, wrong={wrong_player}) — "
            f"likely a non-1..4 GT track_id (LS artifact) or no prior GT for pred_new",
        )
    missing: list[str] = []
    if correct_player not in player_costs_post_swap:
        missing.append(f"P{correct_player}")
    if wrong_player not in player_costs_post_swap:
        missing.append(f"P{wrong_player}")
    if missing:
        return (
            CLASS_INSUFFICIENT_DATA,
            f"match_analysis.playerProfiles is missing "
            f"{' and '.join(missing)} — match-players never built a profile for these players "
            f"(insufficient samples at cross-rally matching time)",
        )
    if not player_costs_post_swap:
        return (
            CLASS_INSUFFICIENT_DATA,
            "no post-swap appearance samples — pred_new doesn't appear in the post-swap window",
        )
    cost_correct = player_costs_post_swap[correct_player]
    cost_wrong = player_costs_post_swap[wrong_player]
    delta = cost_wrong - cost_correct  # +ve → correct is cheaper (ReID had signal)

    if delta >= DELTA_HAS_SIGNAL:
        return (
            CLASS_REID_HAD_SIGNAL,
            f"cost(correct P{correct_player})={cost_correct:.3f} vs cost(wrong P{wrong_player})={cost_wrong:.3f} "
            f"(Δ=+{delta:.3f}); ReID preferred correct — swap caused by non-ReID factor (spatial / gap)",
        )
    if delta <= -DELTA_BLIND:
        return (
            CLASS_REID_WRONG_PREFERENCE,
            f"cost(correct P{correct_player})={cost_correct:.3f} vs cost(wrong P{wrong_player})={cost_wrong:.3f} "
            f"(Δ={delta:.3f}); ReID actively favoured wrong player — appearance drift / pose / crop quality",
        )
    return (
        CLASS_REID_BLIND,
        f"cost(correct P{correct_player})={cost_correct:.3f} vs cost(wrong P{wrong_player})={cost_wrong:.3f} "
        f"(|Δ|={abs(delta):.3f} within ±{DELTA_BLIND}); ReID cannot distinguish — needs pose/role features",
    )


def probe_swap(
    rally_id: str,
    swap_frame: int,
    gt_track_id: int,
    pred_old: int,
    pred_new: int,
    prior_gt_of_new: int | None,
    video_path: Path,
    rally_start_ms: float,
    video_fps: float,
    player_profiles: dict[int, PlayerAppearanceProfile],
    correct_player_id: int | None,
    wrong_player_id: int | None,
    predictions: list[PlayerPosition],
    window: int = 15,
) -> SwapProbeResult:
    """Main probe entry. See module docstring for classification semantics.

    correct_player_id = canonical player_id of the GT pred_new was tracking BEFORE
                        the swap (the one ReID should have stuck to).
    wrong_player_id   = canonical player_id of the GT pred_new is tracking AFTER
                        the swap (the hijacked GT's canonical player).
    """
    correct_player = correct_player_id
    wrong_player = wrong_player_id

    # Collect pred_new's features in pre-swap and post-swap windows.
    pre_range = range(max(0, swap_frame - window), swap_frame)
    post_range = range(swap_frame, swap_frame + window)
    needed = set(pre_range) | set(post_range)
    frames = _iterate_rally_frames(video_path, rally_start_ms, video_fps, needed)

    stats_pre, samples_pre = _collect_stats_for_pred(pred_new, predictions, pre_range, frames)
    stats_post, samples_post = _collect_stats_for_pred(pred_new, predictions, post_range, frames)

    # Per-player costs (lower = more similar to that player's canonical profile).
    def costs(stats: TrackAppearanceStats) -> dict[int, float]:
        out: dict[int, float] = {}
        for pid, profile in player_profiles.items():
            out[pid] = compute_appearance_similarity(profile, stats)
        return out

    costs_pre = costs(stats_pre) if samples_pre > 0 else {}
    costs_post = costs(stats_post) if samples_post > 0 else {}

    spatial = _spatial_distances(pred_new, swap_frame, predictions)

    classification, reasoning = _classify(
        costs_post, correct_player=correct_player, wrong_player=wrong_player
    )

    return SwapProbeResult(
        rally_id=rally_id,
        swap_frame=swap_frame,
        gt_track_id=gt_track_id,
        pred_old=pred_old,
        pred_new=pred_new,
        prior_gt_of_new=prior_gt_of_new,
        player_costs_pre_swap=costs_pre,
        player_costs_post_swap=costs_post,
        spatial_distance_to_each_gt=spatial,
        classification=classification,
        reasoning=reasoning,
        samples_used_pre=samples_pre,
        samples_used_post=samples_post,
    )
