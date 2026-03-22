#!/usr/bin/env python3
"""Feasibility test: ReID embeddings for player attribution.

At each matched contact from the action GT evaluation, compares candidate
player crops against user-selected reference crop embeddings. Tests whether
ReID models can identify the correct player better than proximity.

Models tested:
- dinov2, clip, osnet: Nearest-prototype (baseline, no training)
- finetuned: Few-shot fine-tuned linear classifier on DINOv2 features

GO threshold: >80% accuracy on fixable errors (excluding missing_player).

Usage:
    cd analysis
    uv run python scripts/feasibility_reid_attribution.py
    uv run python scripts/feasibility_reid_attribution.py --model finetuned
    uv run python scripts/feasibility_reid_attribution.py --model dinov2
"""

from __future__ import annotations

import argparse
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from rich.console import Console
from rich.table import Table

from rallycut.court.calibration import CourtCalibrator
from rallycut.evaluation.db import get_connection
from rallycut.evaluation.tracking.db import get_video_path, load_court_calibration
from rallycut.tracking.action_classifier import classify_rally_actions
from rallycut.tracking.contact_detector import (
    Contact,
    ContactDetectionConfig,
    detect_contacts,
)

from scripts.eval_action_detection import (
    RallyData,
    _load_match_team_assignments,
    load_rallies_with_action_gt,
    match_contacts,
)

logging.basicConfig(level=logging.WARNING, format="%(message)s")
console = Console()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ReferenceCropInfo:
    """Reference crop metadata from DB."""

    player_id: int
    frame_ms: int
    bbox_x: float
    bbox_y: float
    bbox_w: float
    bbox_h: float


@dataclass
class ContactEvalResult:
    """Result of evaluating ReID on a single contact."""

    rally_id: str
    frame: int
    gt_track_id: int
    gt_player_id: int  # 1-4 from match analysis
    pred_track_id: int
    error_type: str  # correct, wrong_nearest, wrong_team, missing_player

    # ReID result
    reid_player_id: int = -1  # Player ID with closest reference embedding
    reid_correct: bool = False
    reid_team_correct: bool = False  # Correct team (2-way)
    reid_margin: float = 0.0  # Gap between best and second-best distance
    reid_distances: dict[int, float] = field(default_factory=dict)  # player_id -> dist

    # Crop diagnostics
    gt_crop_pixels: int = 0  # Pixels in GT player's crop
    num_candidates: int = 0


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


_model_cache: dict[str, tuple] = {}


def _get_dinov2_model(device: str) -> tuple:
    """Load DINOv2 ViT-S/14 model."""
    if "dinov2" not in _model_cache:
        console.print("  Loading DINOv2 ViT-S/14...")
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        model = model.to(device).eval()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        _model_cache["dinov2"] = (model, mean, std, 224)
    return _model_cache["dinov2"]


def _get_clip_model(device: str) -> tuple:
    """Load CLIP ViT-B/32 visual encoder."""
    if "clip" not in _model_cache:
        console.print("  Loading CLIP ViT-B/32...")
        from transformers import CLIPModel, CLIPProcessor

        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model = model.to(device).eval()
        _model_cache["clip"] = (model, processor, 224)
    return _model_cache["clip"]


def extract_embeddings_dinov2(
    crops: list[np.ndarray], device: str
) -> np.ndarray:
    """Extract DINOv2 CLS embeddings from BGR crops.

    Args:
        crops: List of BGR images (any size).
        device: Torch device string.

    Returns:
        (N, embed_dim) L2-normalized embeddings.
    """
    model, mean, std, size = _get_dinov2_model(device)

    batch = []
    for crop in crops:
        img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size))
        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        batch.append(tensor)

    batch_tensor = torch.stack(batch).to(device)
    batch_tensor = (batch_tensor - mean) / std

    with torch.inference_mode():
        features = model(batch_tensor)

    features = F.normalize(features, dim=1)
    return features.cpu().numpy()


def extract_embeddings_clip(
    crops: list[np.ndarray], device: str
) -> np.ndarray:
    """Extract CLIP visual embeddings from BGR crops.

    Args:
        crops: List of BGR images (any size).
        device: Torch device string.

    Returns:
        (N, embed_dim) L2-normalized embeddings.
    """
    model, processor, _size = _get_clip_model(device)

    # Convert BGR to RGB PIL-like arrays
    images = []
    for crop in crops:
        img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        images.append(img)

    inputs = processor(images=images, return_tensors="pt").to(device)

    with torch.inference_mode():
        features = model.get_image_features(**inputs)

    features = F.normalize(features, dim=1)
    return features.cpu().numpy()


def _get_osnet_model(device: str) -> tuple:
    """Load OSNet-x0.25 ReID model."""
    if "osnet" not in _model_cache:
        console.print("  Loading OSNet-x0.25...")
        import torchreid

        extractor = torchreid.utils.FeatureExtractor(
            model_name="osnet_x0_25",
            model_path="",  # ImageNet pretrained
            device=device,
        )
        _model_cache["osnet"] = (extractor,)
    return _model_cache["osnet"]


def extract_embeddings_osnet(
    crops: list[np.ndarray], device: str
) -> np.ndarray:
    """Extract OSNet embeddings from BGR crops.

    Args:
        crops: List of BGR images (any size).
        device: Torch device string.

    Returns:
        (N, embed_dim) L2-normalized embeddings.
    """
    (extractor,) = _get_osnet_model(device)

    # OSNet expects PIL-like RGB images or paths, but FeatureExtractor
    # also accepts list of numpy arrays (RGB, HWC)
    images = []
    for crop in crops:
        img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        # Resize to standard ReID size (256x128)
        img = cv2.resize(img, (128, 256))
        images.append(img)

    # FeatureExtractor accepts numpy arrays
    features = extractor(images)  # (N, embed_dim) tensor
    features = F.normalize(features, dim=1)
    return features.cpu().numpy()


EXTRACT_FN = {
    "dinov2": extract_embeddings_dinov2,
    "clip": extract_embeddings_clip,
    "osnet": extract_embeddings_osnet,
}


# ---------------------------------------------------------------------------
# DB queries
# ---------------------------------------------------------------------------


def load_reference_crops(video_ids: set[str]) -> dict[str, list[ReferenceCropInfo]]:
    """Load reference crops from DB grouped by video_id."""
    if not video_ids:
        return {}

    placeholders = ", ".join(["%s"] * len(video_ids))
    query = f"""
        SELECT video_id, player_id, frame_ms, bbox_x, bbox_y, bbox_w, bbox_h
        FROM player_reference_crops
        WHERE video_id IN ({placeholders})
        ORDER BY video_id, player_id, created_at
    """

    result: dict[str, list[ReferenceCropInfo]] = defaultdict(list)
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, list(video_ids))
            for row in cur.fetchall():
                vid_val, pid, fms, bx, by, bw, bh = row
                result[str(vid_val)].append(ReferenceCropInfo(
                    player_id=pid, frame_ms=fms,  # type: ignore[arg-type]
                    bbox_x=bx, bbox_y=by, bbox_w=bw, bbox_h=bh,  # type: ignore[arg-type]
                ))
    return dict(result)


def load_match_analysis(video_ids: set[str]) -> dict[str, dict]:
    """Load match_analysis_json for each video."""
    if not video_ids:
        return {}

    placeholders = ", ".join(["%s"] * len(video_ids))
    query = f"""
        SELECT id, match_analysis_json
        FROM videos
        WHERE id IN ({placeholders})
          AND match_analysis_json IS NOT NULL
    """

    result: dict[str, dict] = {}
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, list(video_ids))
            for row in cur.fetchall():
                vid_val = str(row[0])
                ma_json = row[1]
                if isinstance(ma_json, dict):
                    result[vid_val] = ma_json
    return result


def build_rally_track_to_player(
    match_analysis: dict,
) -> dict[str, dict[int, int]]:
    """Build rally_id -> {track_id: player_id} from match analysis."""
    rallies = match_analysis.get("rallies", [])
    if not isinstance(rallies, list):
        return {}

    result: dict[str, dict[int, int]] = {}
    for rally_entry in rallies:
        rid = rally_entry.get("rallyId") or rally_entry.get("rally_id", "")
        t2p = rally_entry.get("trackToPlayer") or rally_entry.get("track_to_player", {})
        if rid and t2p:
            result[rid] = {int(k): int(v) for k, v in t2p.items()}
    return result


# ---------------------------------------------------------------------------
# Crop extraction from video
# ---------------------------------------------------------------------------


def extract_crop_from_frame(
    frame: np.ndarray,
    bbox_x: float,
    bbox_y: float,
    bbox_w: float,
    bbox_h: float,
) -> np.ndarray | None:
    """Extract a crop from a BGR frame using normalized bbox coordinates."""
    h, w = frame.shape[:2]
    x1 = max(0, int((bbox_x - bbox_w / 2) * w))
    y1 = max(0, int((bbox_y - bbox_h / 2) * h))
    x2 = min(w, int((bbox_x + bbox_w / 2) * w))
    y2 = min(h, int((bbox_y + bbox_h / 2) * h))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    return crop


def extract_reference_embeddings(
    video_path: Path,
    crops: list[ReferenceCropInfo],
    model_name: str,
    device: str,
) -> dict[int, np.ndarray]:
    """Extract embeddings for reference crops, averaged per player_id.

    Returns:
        dict of player_id -> L2-normalized embedding.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {}

    extract_fn = EXTRACT_FN[model_name]

    # Group crops by player_id
    by_player: dict[int, list[np.ndarray]] = defaultdict(list)

    # Sort by frame_ms for forward seeking
    sorted_crops = sorted(crops, key=lambda c: c.frame_ms)

    for crop_info in sorted_crops:
        cap.set(cv2.CAP_PROP_POS_MSEC, crop_info.frame_ms)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        crop = extract_crop_from_frame(
            frame, crop_info.bbox_x, crop_info.bbox_y,
            crop_info.bbox_w, crop_info.bbox_h,
        )
        if crop is not None and crop.shape[0] >= 16 and crop.shape[1] >= 8:
            by_player[crop_info.player_id].append(crop)

    cap.release()

    # Compute embeddings per player (average all crops)
    result: dict[int, np.ndarray] = {}
    for pid, player_crops in by_player.items():
        embeddings = extract_fn(player_crops, device)
        mean_emb = embeddings.mean(axis=0)
        mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)
        result[pid] = mean_emb

    return result


# ---------------------------------------------------------------------------
# Contact evaluation
# ---------------------------------------------------------------------------


def evaluate_contacts_for_video(
    video_path: Path,
    rallies: list[RallyData],
    ref_embeddings: dict[int, np.ndarray],
    rally_track_to_player: dict[str, dict[int, int]],
    match_teams_by_rally: dict[str, dict[int, int]],
    calibrators: dict[str, CourtCalibrator | None],
    model_name: str,
    device: str,
    tolerance_ms: int = 167,
) -> list[ContactEvalResult]:
    """Evaluate ReID attribution on all contacts for one video's rallies."""
    from rallycut.tracking.ball_tracker import BallPosition as BallPos
    from rallycut.tracking.player_tracker import PlayerPosition as PlayerPos

    extract_fn = EXTRACT_FN[model_name]
    results: list[ContactEvalResult] = []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return results

    fps_cap = cap.get(cv2.CAP_PROP_FPS) or 30.0

    for rally in rallies:
        if not rally.ball_positions_json or not rally.positions_json:
            continue

        t2p = rally_track_to_player.get(rally.rally_id, {})
        if not t2p:
            continue

        # Run contact detection (same as diagnose_attribution.py)
        ball_positions = [
            BallPos(
                frame_number=bp["frameNumber"],
                x=bp["x"], y=bp["y"],
                confidence=bp.get("confidence", 1.0),
            )
            for bp in rally.ball_positions_json
            if bp.get("x", 0) > 0 or bp.get("y", 0) > 0
        ]

        player_positions = [
            PlayerPos(
                frame_number=pp["frameNumber"],
                track_id=pp["trackId"],
                x=pp["x"], y=pp["y"],
                width=pp["width"], height=pp["height"],
                confidence=pp.get("confidence", 1.0),
            )
            for pp in rally.positions_json
        ]

        cal = calibrators.get(rally.video_id)
        contact_seq = detect_contacts(
            ball_positions=ball_positions,
            player_positions=player_positions,
            config=ContactDetectionConfig(),
            net_y=rally.court_split_y,
            frame_count=rally.frame_count or None,
            court_calibrator=cal,
        )

        teams = match_teams_by_rally.get(rally.rally_id)
        rally_actions = classify_rally_actions(
            contact_seq, rally.rally_id,
            match_team_assignments=teams,
        )
        pred_actions = [a.to_dict() for a in rally_actions.actions]
        real_pred = [a for a in pred_actions if not a.get("isSynthetic")]

        # Match GT to predictions
        avail_tids: set[int] = {pp["trackId"] for pp in rally.positions_json}
        fps = rally.fps or 30.0
        tol = max(1, round(fps * tolerance_ms / 1000))
        matches, _ = match_contacts(
            rally.gt_labels, real_pred,
            tolerance=tol, available_track_ids=avail_tids,
        )

        # Build position lookup: (frame_number, track_id) -> position
        pos_by_frame_track: dict[tuple[int, int], dict] = {}
        for pp in rally.positions_json:
            key = (pp["frameNumber"], pp["trackId"])
            pos_by_frame_track[key] = pp

        # Build contact lookup
        contact_by_frame: dict[int, Contact] = {c.frame: c for c in contact_seq.contacts}

        for m in matches:
            if m.pred_frame is None:
                continue

            gt_tid = next(
                (gt.player_track_id for gt in rally.gt_labels if gt.frame == m.gt_frame),
                -1,
            )
            pred_tid = next(
                (a.get("playerTrackId", -1) for a in real_pred if a.get("frame") == m.pred_frame),
                -1,
            )

            # Classify error type
            contact = contact_by_frame.get(m.pred_frame)
            if gt_tid < 0 or gt_tid not in avail_tids:
                continue  # Skip stale/unknown GT

            gt_player_id = t2p.get(gt_tid, -1)
            if gt_player_id < 0:
                continue  # GT track not in match analysis

            if gt_tid == pred_tid:
                error_type = "correct"
            elif contact and contact.player_candidates:
                candidate_tids = [tid for tid, _ in contact.player_candidates]
                if gt_tid in candidate_tids:
                    # Check if it's cross-team or within-team
                    if teams:
                        pred_team = teams.get(pred_tid, -1)
                        gt_team = teams.get(gt_tid, -2)
                        error_type = "wrong_team" if pred_team != gt_team else "wrong_nearest"
                    else:
                        error_type = "wrong_nearest"
                else:
                    error_type = "missing_player"
            else:
                error_type = "missing_player"

            # Get candidate bboxes at contact frame
            if not contact or not contact.player_candidates:
                continue

            # Collect candidate positions for crop extraction
            candidate_positions: dict[int, dict] = {}
            candidate_player_ids: dict[int, int] = {}

            for cand_tid, cand_dist in contact.player_candidates:
                cand_pid = t2p.get(cand_tid, -1)
                if cand_pid < 0:
                    continue

                # Find position within ±5 frames of contact
                best_pos = None
                for delta in range(6):
                    for fn in [m.pred_frame + delta, m.pred_frame - delta]:
                        if fn < 0:
                            continue
                        pos = pos_by_frame_track.get((fn, cand_tid))
                        if pos is not None:
                            best_pos = pos
                            break
                    if best_pos is not None:
                        break

                if best_pos is None:
                    continue

                candidate_player_ids[cand_tid] = cand_pid

                candidate_positions[cand_tid] = best_pos

            if not candidate_positions or gt_player_id not in ref_embeddings:
                continue

            # Extract crops from video frames
            actual_crops: dict[int, np.ndarray] = {}
            rally_start_frame = _get_rally_start_frame(rally.rally_id, fps_cap)

            for cand_tid, pos in candidate_positions.items():
                abs_fn = rally_start_frame + pos["frameNumber"]
                cap.set(cv2.CAP_PROP_POS_FRAMES, abs_fn)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                crop = extract_crop_from_frame(
                    frame, pos["x"], pos["y"], pos["width"], pos["height"],
                )
                if crop is not None and crop.shape[0] >= 16 and crop.shape[1] >= 8:
                    actual_crops[cand_tid] = crop

            if not actual_crops:
                continue

            # Compute embeddings for all candidate crops
            crop_list = list(actual_crops.values())
            tid_list = list(actual_crops.keys())
            embeddings = extract_fn(crop_list, device)

            # Compare each candidate to each reference player
            reid_distances: dict[int, float] = {}  # player_id -> min distance
            reid_by_tid: dict[int, dict[int, float]] = {}  # tid -> {pid -> dist}

            for i, tid in enumerate(tid_list):
                cand_emb = embeddings[i]
                tid_dists: dict[int, float] = {}
                for pid, ref_emb in ref_embeddings.items():
                    dist = 1.0 - float(np.dot(cand_emb, ref_emb))
                    tid_dists[pid] = dist
                reid_by_tid[tid] = tid_dists

            # For each candidate, find the reference player it's closest to
            # Then pick the candidate-player pair with the overall best assignment
            # Simple approach: for each reference player, which candidate is closest?
            # Better: for the GT player's reference, which candidate is closest?
            best_tid_for_gt = -1
            best_dist_for_gt = float("inf")
            all_dists_to_gt_ref: list[tuple[int, float]] = []

            for tid in tid_list:
                if gt_player_id in reid_by_tid.get(tid, {}):
                    dist = reid_by_tid[tid][gt_player_id]
                    all_dists_to_gt_ref.append((tid, dist))
                    if dist < best_dist_for_gt:
                        best_dist_for_gt = dist
                        best_tid_for_gt = tid

            # Also find overall closest: for each candidate, its best-match reference
            cand_best_pid: dict[int, tuple[int, float]] = {}
            for tid in tid_list:
                best_pid = -1
                best_d = float("inf")
                for pid, d in reid_by_tid.get(tid, {}).items():
                    if d < best_d:
                        best_d = d
                        best_pid = pid
                cand_best_pid[tid] = (best_pid, best_d)

            # Global assignment: find which candidate has the lowest distance
            # to the GT player's reference embedding
            reid_player_id = candidate_player_ids.get(best_tid_for_gt, -1)
            reid_correct = (best_tid_for_gt == gt_tid)

            # Team-level correctness (P1-P2 = team 0, P3-P4 = team 1)
            gt_team = 0 if gt_player_id <= 2 else 1
            reid_team = 0 if reid_player_id <= 2 else 1
            reid_team_correct = (gt_team == reid_team) if reid_player_id > 0 else False

            # Compute margin
            sorted_dists = sorted(all_dists_to_gt_ref, key=lambda x: x[1])
            if len(sorted_dists) >= 2:
                margin = sorted_dists[1][1] - sorted_dists[0][1]
            else:
                margin = 0.0

            # Collect distances for reporting
            for pid, ref_emb in ref_embeddings.items():
                if gt_tid in reid_by_tid:
                    reid_distances[pid] = reid_by_tid[gt_tid].get(pid, float("inf"))

            # Crop size diagnostic
            gt_crop_pixels = 0
            if gt_tid in actual_crops:
                c = actual_crops[gt_tid]
                gt_crop_pixels = c.shape[0] * c.shape[1]

            results.append(ContactEvalResult(
                rally_id=rally.rally_id,
                frame=m.gt_frame,
                gt_track_id=gt_tid,
                gt_player_id=gt_player_id,
                pred_track_id=pred_tid,
                error_type=error_type,
                reid_player_id=reid_player_id,
                reid_correct=reid_correct,
                reid_team_correct=reid_team_correct,
                reid_margin=margin,
                reid_distances=reid_distances,
                gt_crop_pixels=gt_crop_pixels,
                num_candidates=len(actual_crops),
            ))

    cap.release()
    return results


_rally_start_cache: dict[str, int] = {}


def _get_rally_start_frame(rally_id: str, video_fps: float) -> int:
    """Get absolute start frame for a rally (cached)."""
    key = f"{rally_id}:{video_fps:.1f}"
    if key not in _rally_start_cache:
        query = "SELECT start_ms FROM rallies WHERE id = %s"
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, [rally_id])
                row = cur.fetchone()
                start_ms: int = row[0] if row and row[0] is not None else 0  # type: ignore[assignment]
                _rally_start_cache[key] = int(start_ms / 1000.0 * video_fps)
    return _rally_start_cache[key]


# ---------------------------------------------------------------------------
# Fine-tuned classifier evaluation
# ---------------------------------------------------------------------------


def _evaluate_finetuned_for_video(
    video_path: Path,
    ref_crop_infos: list[ReferenceCropInfo],
    rallies: list[RallyData],
    rally_track_to_player: dict[str, dict[int, int]],
    match_teams_by_rally: dict[str, dict[int, int]],
    calibrators: dict[str, CourtCalibrator | None],
    device: str,
    tolerance_ms: int = 167,
) -> list[ContactEvalResult]:
    """Evaluate few-shot fine-tuned classifier on contacts for one video."""
    from rallycut.tracking.ball_tracker import BallPosition as BallPos
    from rallycut.tracking.player_tracker import PlayerPosition as PlayerPos
    from rallycut.tracking.reid_embeddings import (
        PlayerReIDClassifier,
        extract_crops_from_video,
    )

    # Extract reference crops from video
    crop_dicts = [
        {
            "player_id": c.player_id, "frame_ms": c.frame_ms,
            "bbox_x": c.bbox_x, "bbox_y": c.bbox_y,
            "bbox_w": c.bbox_w, "bbox_h": c.bbox_h,
        }
        for c in ref_crop_infos
    ]
    crops_by_player = extract_crops_from_video(video_path, crop_dicts)

    if len(crops_by_player) < 2:
        console.print(f"  [yellow]{str(video_path.name)}: <2 players with crops, skipping[/yellow]")
        return []

    # Train classifier
    classifier = PlayerReIDClassifier(device=device)
    stats = classifier.train(crops_by_player, verbose=False)
    console.print(
        f"  {video_path.stem[:8]}: trained on "
        f"{sum(len(c) for c in crops_by_player.values())} crops "
        f"({len(crops_by_player)} players), "
        f"train_acc={stats['train_acc']:.0%}"
    )

    results: list[ContactEvalResult] = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return results

    fps_cap = cap.get(cv2.CAP_PROP_FPS) or 30.0

    for rally in rallies:
        if not rally.ball_positions_json or not rally.positions_json:
            continue

        t2p = rally_track_to_player.get(rally.rally_id, {})
        if not t2p:
            continue

        # Run contact detection
        ball_positions = [
            BallPos(
                frame_number=bp["frameNumber"],
                x=bp["x"], y=bp["y"],
                confidence=bp.get("confidence", 1.0),
            )
            for bp in rally.ball_positions_json
            if bp.get("x", 0) > 0 or bp.get("y", 0) > 0
        ]

        player_positions = [
            PlayerPos(
                frame_number=pp["frameNumber"],
                track_id=pp["trackId"],
                x=pp["x"], y=pp["y"],
                width=pp["width"], height=pp["height"],
                confidence=pp.get("confidence", 1.0),
            )
            for pp in rally.positions_json
        ]

        cal = calibrators.get(rally.video_id)
        contact_seq = detect_contacts(
            ball_positions=ball_positions,
            player_positions=player_positions,
            config=ContactDetectionConfig(),
            net_y=rally.court_split_y,
            frame_count=rally.frame_count or None,
            court_calibrator=cal,
        )

        teams = match_teams_by_rally.get(rally.rally_id)
        rally_actions = classify_rally_actions(
            contact_seq, rally.rally_id,
            match_team_assignments=teams,
        )
        pred_actions = [a.to_dict() for a in rally_actions.actions]
        real_pred = [a for a in pred_actions if not a.get("isSynthetic")]

        avail_tids: set[int] = {pp["trackId"] for pp in rally.positions_json}
        fps = rally.fps or 30.0
        tol = max(1, round(fps * tolerance_ms / 1000))
        matches, _ = match_contacts(
            rally.gt_labels, real_pred,
            tolerance=tol, available_track_ids=avail_tids,
        )

        pos_by_frame_track: dict[tuple[int, int], dict] = {}
        for pp in rally.positions_json:
            pos_by_frame_track[(pp["frameNumber"], pp["trackId"])] = pp

        contact_by_frame: dict[int, Contact] = {
            c.frame: c for c in contact_seq.contacts
        }

        rally_start_frame = _get_rally_start_frame(rally.rally_id, fps_cap)

        for m in matches:
            if m.pred_frame is None:
                continue

            gt_tid = next(
                (gt.player_track_id for gt in rally.gt_labels if gt.frame == m.gt_frame),
                -1,
            )
            pred_tid = next(
                (a.get("playerTrackId", -1) for a in real_pred if a.get("frame") == m.pred_frame),
                -1,
            )

            if gt_tid < 0 or gt_tid not in avail_tids:
                continue
            gt_player_id = t2p.get(gt_tid, -1)
            if gt_player_id < 0:
                continue

            contact = contact_by_frame.get(m.pred_frame)
            if gt_tid == pred_tid:
                error_type = "correct"
            elif contact and contact.player_candidates:
                candidate_tids = [tid for tid, _ in contact.player_candidates]
                if gt_tid in candidate_tids:
                    if teams:
                        pred_team = teams.get(pred_tid, -1)
                        gt_team_val = teams.get(gt_tid, -2)
                        error_type = "wrong_team" if pred_team != gt_team_val else "wrong_nearest"
                    else:
                        error_type = "wrong_nearest"
                else:
                    error_type = "missing_player"
            else:
                error_type = "missing_player"

            if not contact or not contact.player_candidates:
                continue

            # Extract candidate crops and classify
            candidate_crops: list[tuple[int, int, np.ndarray]] = []  # (tid, pid, crop)

            for cand_tid, _cand_dist in contact.player_candidates:
                cand_pid = t2p.get(cand_tid, -1)
                if cand_pid < 0:
                    continue

                best_pos = None
                for delta in range(6):
                    for fn in [m.pred_frame + delta, m.pred_frame - delta]:
                        if fn < 0:
                            continue
                        pos = pos_by_frame_track.get((fn, cand_tid))
                        if pos is not None:
                            best_pos = pos
                            break
                    if best_pos is not None:
                        break

                if best_pos is None:
                    continue

                abs_fn = rally_start_frame + best_pos["frameNumber"]
                cap.set(cv2.CAP_PROP_POS_FRAMES, abs_fn)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                crop = extract_crop_from_frame(
                    frame, best_pos["x"], best_pos["y"],
                    best_pos["width"], best_pos["height"],
                )
                if crop is not None and crop.shape[0] >= 16 and crop.shape[1] >= 8:
                    candidate_crops.append((cand_tid, cand_pid, crop))

            if not candidate_crops:
                continue

            # Classify all candidates
            crops_list = [c[2] for c in candidate_crops]
            probs_list = classifier.predict(crops_list)

            # Find candidate with highest probability for GT player
            best_tid = -1
            best_prob = -1.0
            best_pid = -1
            all_gt_probs: list[tuple[int, float]] = []

            for idx, (cand_tid, cand_pid, _crop) in enumerate(candidate_crops):
                prob_for_gt = probs_list[idx].get(gt_player_id, 0.0)
                all_gt_probs.append((cand_tid, prob_for_gt))
                if prob_for_gt > best_prob:
                    best_prob = prob_for_gt
                    best_tid = cand_tid
                    best_pid = cand_pid

            reid_correct = (best_tid == gt_tid)
            gt_team = 0 if gt_player_id <= 2 else 1
            reid_team = 0 if best_pid <= 2 else 1
            reid_team_correct = (gt_team == reid_team) if best_pid > 0 else False

            # Margin: gap between top-2 candidates for GT player's probability
            sorted_gt_probs = sorted(all_gt_probs, key=lambda x: x[1], reverse=True)
            margin = (sorted_gt_probs[0][1] - sorted_gt_probs[1][1]) if len(sorted_gt_probs) >= 2 else 0.0

            gt_crop_pixels = 0
            for cand_tid_check, _cand_pid_check, crop_check in candidate_crops:
                if cand_tid_check == gt_tid:
                    gt_crop_pixels = crop_check.shape[0] * crop_check.shape[1]
                    break

            results.append(ContactEvalResult(
                rally_id=rally.rally_id,
                frame=m.gt_frame,
                gt_track_id=gt_tid,
                gt_player_id=gt_player_id,
                pred_track_id=pred_tid,
                error_type=error_type,
                reid_player_id=best_pid,
                reid_correct=reid_correct,
                reid_team_correct=reid_team_correct,
                reid_margin=margin,
                reid_distances={},
                gt_crop_pixels=gt_crop_pixels,
                num_candidates=len(candidate_crops),
            ))

    cap.release()
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Feasibility test: ReID embeddings for player attribution"
    )
    parser.add_argument(
        "--model", choices=["finetuned", "dinov2", "clip", "osnet", "all"], default="all",
        help="Model to test (default: all = finetuned + dinov2 + osnet)",
    )
    parser.add_argument(
        "--tolerance-ms", type=int, default=167,
        help="Matching tolerance in ms (default: 167)",
    )
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    console.print(f"\n[bold]ReID Attribution Feasibility Test[/bold]")
    console.print(f"  Device: {device}")

    # Load rallies with action GT
    rallies = load_rallies_with_action_gt()
    if not rallies:
        console.print("[red]No rallies with action GT found.[/red]")
        return

    video_ids = {r.video_id for r in rallies}
    console.print(f"  Rallies: {len(rallies)} across {len(video_ids)} videos")

    # Load reference crops
    ref_crops_by_video = load_reference_crops(video_ids)
    videos_with_refs = set(ref_crops_by_video.keys())
    console.print(f"  Videos with reference crops: {len(videos_with_refs)}")

    if not videos_with_refs:
        console.print("[red]No videos have reference crops. Upload crops via the web UI first.[/red]")
        return

    # Filter rallies to videos with reference crops
    rallies = [r for r in rallies if r.video_id in videos_with_refs]
    video_ids = {r.video_id for r in rallies}
    console.print(f"  Evaluable rallies: {len(rallies)} across {len(video_ids)} videos")

    # Load match analysis for track→player mapping
    match_analyses = load_match_analysis(video_ids)
    rally_t2p_by_video: dict[str, dict[str, dict[int, int]]] = {}
    for vid, ma in match_analyses.items():
        rally_t2p_by_video[vid] = build_rally_track_to_player(ma)

    # Load calibrations and team assignments
    calibrators: dict[str, CourtCalibrator | None] = {}
    for vid in video_ids:
        corners = load_court_calibration(vid)
        if corners and len(corners) == 4:
            cal = CourtCalibrator()
            cal.calibrate([(c["x"], c["y"]) for c in corners])
            calibrators[vid] = cal
        else:
            calibrators[vid] = None

    match_teams_by_rally = _load_match_team_assignments(video_ids, min_confidence=0.70)

    # Determine models to test
    if args.model == "all":
        models = ["finetuned", "dinov2", "osnet"]
    else:
        models = [args.model]

    for model_name in models:
        console.print(f"\n[bold]═══ Model: {model_name.upper()} ═══[/bold]\n")
        t0 = time.monotonic()

        all_results: list[ContactEvalResult] = []

        for vid in sorted(video_ids):
            video_path = get_video_path(vid)
            if video_path is None or not video_path.exists():
                console.print(f"  [yellow]Skipping {vid[:8]}: video not available[/yellow]")
                continue

            ref_crops = ref_crops_by_video.get(vid, [])
            if not ref_crops:
                continue

            vid_rallies = [r for r in rallies if r.video_id == vid]
            rally_t2p = rally_t2p_by_video.get(vid, {})

            if model_name == "finetuned":
                # Few-shot fine-tuned classifier
                results = _evaluate_finetuned_for_video(
                    video_path=video_path,
                    ref_crop_infos=ref_crops,
                    rallies=vid_rallies,
                    rally_track_to_player=rally_t2p,
                    match_teams_by_rally=match_teams_by_rally,
                    calibrators=calibrators,
                    device=device,
                    tolerance_ms=args.tolerance_ms,
                )
            else:
                # Embedding-based nearest prototype
                ref_embs = extract_reference_embeddings(
                    video_path, ref_crops, model_name, device,
                )
                if not ref_embs:
                    console.print(f"  [yellow]{vid[:8]}: no reference embeddings extracted[/yellow]")
                    continue

                ref_players = sorted(ref_embs.keys())
                console.print(
                    f"  {vid[:8]}: {len(ref_crops)} ref crops → "
                    f"embeddings for players {ref_players}"
                )

                results = evaluate_contacts_for_video(
                    video_path=video_path,
                    rallies=vid_rallies,
                    ref_embeddings=ref_embs,
                    rally_track_to_player=rally_t2p,
                    match_teams_by_rally=match_teams_by_rally,
                    calibrators=calibrators,
                    model_name=model_name,
                    device=device,
                    tolerance_ms=args.tolerance_ms,
                )

            all_results.extend(results)
            console.print(f"    → {len(results)} evaluable contacts")

        elapsed = time.monotonic() - t0

        if not all_results:
            console.print("[red]No evaluable contacts.[/red]")
            continue

        # === Report ===
        _print_report(all_results, model_name, elapsed)


def _print_report(results: list[ContactEvalResult], model_name: str, elapsed: float) -> None:
    """Print detailed results report."""
    total = len(results)
    correct_baseline = sum(1 for r in results if r.error_type == "correct")
    correct_reid = sum(1 for r in results if r.reid_correct)

    # By error type
    by_type: dict[str, list[ContactEvalResult]] = defaultdict(list)
    for r in results:
        by_type[r.error_type].append(r)

    console.print(f"\n[bold]Results: {model_name.upper()} ({elapsed:.1f}s)[/bold]")
    console.print(f"  Total evaluable contacts: {total}")
    console.print(
        f"  Baseline (proximity): {correct_baseline}/{total} "
        f"= {correct_baseline / total:.1%}"
    )
    console.print(
        f"  ReID attribution:     {correct_reid}/{total} "
        f"= {correct_reid / total:.1%}"
    )
    correct_team_reid = sum(1 for r in results if r.reid_team_correct)
    console.print(
        f"  ReID team (2-way):    {correct_team_reid}/{total} "
        f"= {correct_team_reid / total:.1%}"
    )

    # Crop size diagnostics
    crop_pixels = [r.gt_crop_pixels for r in results if r.gt_crop_pixels > 0]
    if crop_pixels:
        console.print(f"\n  Crop size (GT player):")
        console.print(f"    Mean:   {np.mean(crop_pixels):.0f} px")
        console.print(f"    Median: {np.median(crop_pixels):.0f} px")
        console.print(f"    Min:    {np.min(crop_pixels):.0f} px")
        console.print(f"    Max:    {np.max(crop_pixels):.0f} px")
    avg_cands = np.mean([r.num_candidates for r in results])
    console.print(f"    Avg candidates/contact: {avg_cands:.1f}")

    # Breakdown table
    table = Table(title=f"ReID Attribution by Error Type ({model_name})")
    table.add_column("Error Type", style="bold")
    table.add_column("Count", justify="right")
    table.add_column("ReID Correct", justify="right")
    table.add_column("ReID Acc", justify="right")
    table.add_column("Avg Margin", justify="right")

    for etype in ["correct", "wrong_team", "wrong_nearest", "missing_player"]:
        items = by_type.get(etype, [])
        if not items:
            continue
        n = len(items)
        n_correct = sum(1 for r in items if r.reid_correct)
        avg_margin = np.mean([r.reid_margin for r in items]) if items else 0
        table.add_row(
            etype,
            str(n),
            str(n_correct),
            f"{n_correct / n:.1%}",
            f"{avg_margin:.4f}",
        )

    # Totals row
    fixable = [r for r in results if r.error_type != "missing_player"]
    fixable_correct = sum(1 for r in fixable if r.reid_correct)
    fixable_margin = np.mean([r.reid_margin for r in fixable]) if fixable else 0
    table.add_row(
        "[bold]TOTAL (fixable)[/bold]",
        f"[bold]{len(fixable)}[/bold]",
        f"[bold]{fixable_correct}[/bold]",
        f"[bold]{fixable_correct / max(1, len(fixable)):.1%}[/bold]",
        f"[bold]{fixable_margin:.4f}[/bold]",
    )
    console.print(table)

    # Margin distribution
    margins = [r.reid_margin for r in results if r.reid_correct]
    if margins:
        console.print(f"\n  [bold]Margin stats (correct only):[/bold]")
        console.print(f"    Mean:   {np.mean(margins):.4f}")
        console.print(f"    Median: {np.median(margins):.4f}")
        console.print(f"    Min:    {np.min(margins):.4f}")
        console.print(f"    P25:    {np.percentile(margins, 25):.4f}")
        console.print(f"    P75:    {np.percentile(margins, 75):.4f}")

    # Wrong cases analysis
    wrong = [r for r in results if not r.reid_correct and r.error_type != "missing_player"]
    if wrong:
        console.print(f"\n  [bold]ReID failures ({len(wrong)}):[/bold]")
        for r in wrong[:10]:
            console.print(
                f"    rally={r.rally_id[:8]} frame={r.frame} "
                f"GT=P{r.gt_player_id} ReID=P{r.reid_player_id} "
                f"type={r.error_type} margin={r.reid_margin:.4f}"
            )
        if len(wrong) > 10:
            console.print(f"    ... and {len(wrong) - 10} more")

    # GO/NO-GO decision
    fixable_acc = fixable_correct / max(1, len(fixable))
    console.print()
    if fixable_acc > 0.80:
        console.print(
            f"  [green bold]GO: {fixable_acc:.1%} accuracy on fixable errors "
            f"(>{80}% threshold)[/green bold]"
        )
    elif fixable_acc > 0.70:
        console.print(
            f"  [yellow bold]MARGINAL: {fixable_acc:.1%} accuracy on fixable errors. "
            f"May help as ensemble.[/yellow bold]"
        )
    else:
        console.print(
            f"  [red bold]NO-GO: {fixable_acc:.1%} accuracy on fixable errors "
            f"(<70% threshold)[/red bold]"
        )


if __name__ == "__main__":
    main()
