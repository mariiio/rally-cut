"""Diagnostic probe: can DINOv2 features from player crops classify action types?

Extracts DINOv2 ViT-S/14 (384-dim) features from the attributed player's
bounding box crop at each GT contact frame. Trains a logistic regression
with leave-one-video-out CV to measure how much action-type signal exists
in the visual appearance alone — especially for the dig/set confusion.

Usage:
    cd analysis
    uv run python scripts/probe_visual_action_features.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from rich.console import Console
from rich.table import Table
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from rallycut.evaluation.tracking.db import get_video_path
from rallycut.tracking.ball_tracker import BallPosition as BallPos
from rallycut.tracking.contact_detector import ContactDetectionConfig, detect_contacts
from rallycut.tracking.player_features import extract_bbox_crop
from rallycut.tracking.player_tracker import PlayerPosition as PlayerPos
from rallycut.tracking.reid_embeddings import extract_backbone_features
from scripts.eval_action_detection import (
    RallyData,
    load_rallies_with_action_gt,
    match_contacts,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
console = Console()

ACTION_CLASSES = ["serve", "receive", "set", "attack", "dig"]


def _find_player_bbox_at_frame(
    positions_json: list[dict],
    track_id: int,
    frame: int,
    search_radius: int = 3,
) -> tuple[float, float, float, float] | None:
    """Find the bbox for a track_id at or near a frame.

    Returns (cx, cy, w, h) normalized or None.
    """
    best = None
    best_dist = search_radius + 1
    for pp in positions_json:
        if pp["trackId"] != track_id:
            continue
        dist = abs(pp["frameNumber"] - frame)
        if dist < best_dist:
            best_dist = dist
            best = (pp["x"], pp["y"], pp["width"], pp["height"])
    return best


def extract_crops_for_rally(
    rally: RallyData,
    cap: cv2.VideoCapture,
    tolerance: int = 5,
) -> list[tuple[np.ndarray, str, str]]:
    """Extract player crops at matched contact frames.

    Returns list of (crop_bgr, gt_action, video_id).
    """
    if not rally.ball_positions_json or not rally.positions_json:
        return []

    ball_positions = [
        BallPos(
            frame_number=bp["frameNumber"],
            x=bp["x"],
            y=bp["y"],
            confidence=bp.get("confidence", 1.0),
        )
        for bp in rally.ball_positions_json
        if bp.get("x", 0) > 0 or bp.get("y", 0) > 0
    ]
    if not ball_positions:
        return []

    player_positions = [
        PlayerPos(
            frame_number=pp["frameNumber"],
            track_id=pp["trackId"],
            x=pp["x"],
            y=pp["y"],
            width=pp["width"],
            height=pp["height"],
            confidence=pp.get("confidence", 1.0),
        )
        for pp in rally.positions_json
    ]

    contact_seq = detect_contacts(
        ball_positions=ball_positions,
        player_positions=player_positions,
        config=ContactDetectionConfig(),
        net_y=rally.court_split_y,
        frame_count=rally.frame_count or None,
    )
    if not contact_seq.contacts:
        return []

    pred_actions = [
        {"frame": c.frame, "action": "unknown", "playerTrackId": c.player_track_id}
        for c in contact_seq.contacts
    ]
    matches, _ = match_contacts(rally.gt_labels, pred_actions, tolerance=tolerance)

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Rally start offset
    rally_start_frame = 0
    if rally.start_ms > 0:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        rally_start_frame = int(rally.start_ms / 1000.0 * fps)

    results = []
    for m in matches:
        if m.pred_frame is None:
            continue
        gt_action = m.gt_action
        if gt_action not in ACTION_CLASSES:
            continue

        # Find the contact to get the attributed player
        contact = None
        for c in contact_seq.contacts:
            if c.frame == m.pred_frame:
                contact = c
                break
        if contact is None or contact.player_track_id < 0:
            continue

        # Find player bbox at contact frame
        bbox = _find_player_bbox_at_frame(
            rally.positions_json, contact.player_track_id, contact.frame,
        )
        if bbox is None:
            continue

        # Seek to the absolute frame in the video
        abs_frame = rally_start_frame + contact.frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, abs_frame)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        crop = extract_bbox_crop(
            np.asarray(frame, dtype=np.uint8), bbox, frame_w, frame_h,
        )
        if crop is None:
            continue

        results.append((crop, gt_action, rally.video_id))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="DINOv2 visual action probe")
    parser.add_argument("--pca-dims", type=int, default=16,
                        help="PCA dimensions (default: 16)")
    parser.add_argument("--device", default=None,
                        help="Torch device (auto-detected if omitted)")
    args = parser.parse_args()

    console.print("[bold]Loading GT rallies...[/bold]")
    rallies = load_rallies_with_action_gt()
    console.print(f"  {len(rallies)} rallies loaded")

    # Group rallies by video
    by_video: dict[str, list[RallyData]] = defaultdict(list)
    for r in rallies:
        by_video[r.video_id].append(r)
    console.print(f"  {len(by_video)} videos")

    # Extract crops per video
    all_crops: list[np.ndarray] = []
    all_labels: list[str] = []
    all_video_ids: list[str] = []

    for vi, (video_id, video_rallies) in enumerate(sorted(by_video.items())):
        video_path = get_video_path(video_id)
        if video_path is None:
            console.print(f"  [yellow]Skip {video_id[:8]} — video not found[/yellow]")
            continue

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            console.print(f"  [yellow]Skip {video_id[:8]} — cannot open[/yellow]")
            continue

        n_before = len(all_crops)
        for rally in video_rallies:
            crops_labels = extract_crops_for_rally(rally, cap)
            for crop, action, vid in crops_labels:
                all_crops.append(crop)
                all_labels.append(action)
                all_video_ids.append(vid)
        cap.release()

        n_new = len(all_crops) - n_before
        console.print(
            f"  [{vi+1}/{len(by_video)}] {video_id[:8]}: {n_new} crops "
            f"(total: {len(all_crops)})"
        )

    if len(all_crops) < 10:
        console.print("[red]Too few crops extracted. Aborting.[/red]")
        sys.exit(1)

    console.print(f"\n[bold]Extracting DINOv2 features for {len(all_crops)} crops...[/bold]")

    # Extract features in batches to manage memory
    batch_size = 64
    all_features = []
    for i in range(0, len(all_crops), batch_size):
        batch = all_crops[i:i + batch_size]
        feats = extract_backbone_features(batch, device=args.device)
        all_features.append(feats)
        if (i + batch_size) % 256 == 0 or i + batch_size >= len(all_crops):
            console.print(f"  {min(i + batch_size, len(all_crops))}/{len(all_crops)}")
    X = np.vstack(all_features)  # (N, 384)
    y = np.array(all_labels)
    video_ids = np.array(all_video_ids)

    # Class distribution
    console.print("\n[bold]Class distribution:[/bold]")
    for cls in ACTION_CLASSES:
        n = np.sum(y == cls)
        console.print(f"  {cls:10s}: {n}")

    # PCA
    console.print(f"\n[bold]PCA: 384 → {args.pca_dims} dims[/bold]")
    pca = PCA(n_components=args.pca_dims, random_state=42)
    X_pca = pca.fit_transform(X)
    var_explained = np.sum(pca.explained_variance_ratio_) * 100
    console.print(f"  Variance explained: {var_explained:.1f}%")

    # Leave-one-video-out CV
    console.print("\n[bold]Leave-one-video-out CV (LogisticRegression)...[/bold]")
    unique_videos = sorted(set(all_video_ids))
    all_preds = np.empty_like(y)

    for vid in unique_videos:
        mask_test = video_ids == vid
        mask_train = ~mask_test

        X_train, y_train = X_pca[mask_train], y[mask_train]
        X_test = X_pca[mask_test]

        if len(X_test) == 0:
            continue

        clf = LogisticRegression(
            C=0.1, max_iter=1000, random_state=42, solver="lbfgs",
        )
        clf.fit(X_train, y_train)
        all_preds[mask_test] = clf.predict(X_test)

    # Overall accuracy
    acc = np.mean(all_preds == y) * 100
    console.print(f"\n[bold green]Overall LOO-CV accuracy: {acc:.1f}%[/bold green]")

    # Per-class report
    console.print("\n[bold]Classification report:[/bold]")
    print(classification_report(y, all_preds, labels=ACTION_CLASSES, digits=3))

    # Confusion matrix
    console.print("[bold]Confusion matrix (rows=true, cols=predicted):[/bold]")
    cm = confusion_matrix(y, all_preds, labels=ACTION_CLASSES)
    table = Table(title="", show_header=True)
    table.add_column("True\\Pred", style="bold")
    for cls in ACTION_CLASSES:
        table.add_column(cls, justify="right")
    for i, cls in enumerate(ACTION_CLASSES):
        row = [str(v) for v in cm[i]]
        table.add_row(cls, *row)
    console.print(table)

    # Dig vs Set focus
    dig_mask = y == "dig"
    set_mask = y == "set"
    ds_mask = dig_mask | set_mask
    if ds_mask.sum() > 0:
        ds_acc = np.mean(all_preds[ds_mask] == y[ds_mask]) * 100
        console.print(f"\n[bold]Dig vs Set subset accuracy: {ds_acc:.1f}%[/bold]")

        # Within dig/set, what does the model predict?
        n_dig = dig_mask.sum()
        n_set = set_mask.sum()
        dig_correct = np.sum((all_preds == "dig") & dig_mask)
        set_correct = np.sum((all_preds == "set") & set_mask)
        dig_as_set = np.sum((all_preds == "set") & dig_mask)
        set_as_dig = np.sum((all_preds == "dig") & set_mask)
        console.print(f"  Dig correct: {dig_correct}/{n_dig} ({100*dig_correct/n_dig:.1f}%)")
        console.print(f"  Dig→Set: {dig_as_set}/{n_dig} ({100*dig_as_set/n_dig:.1f}%)")
        console.print(f"  Set correct: {set_correct}/{n_set} ({100*set_correct/n_set:.1f}%)")
        console.print(f"  Set→Dig: {set_as_dig}/{n_set} ({100*set_as_dig/n_set:.1f}%)")

    # Also try raw features (no PCA) for comparison
    console.print("\n[bold]--- Raw 384-dim features (no PCA) ---[/bold]")
    all_preds_raw = np.empty_like(y)
    for vid in unique_videos:
        mask_test = video_ids == vid
        mask_train = ~mask_test
        X_train, y_train = X[mask_train], y[mask_train]
        X_test = X[mask_test]
        if len(X_test) == 0:
            continue
        clf = LogisticRegression(
            C=0.01, max_iter=1000, random_state=42, solver="lbfgs",
        )
        clf.fit(X_train, y_train)
        all_preds_raw[mask_test] = clf.predict(X_test)

    acc_raw = np.mean(all_preds_raw == y) * 100
    console.print(f"Overall LOO-CV accuracy (raw 384d): {acc_raw:.1f}%")
    if ds_mask.sum() > 0:
        ds_acc_raw = np.mean(all_preds_raw[ds_mask] == y[ds_mask]) * 100
        console.print(f"Dig vs Set subset accuracy (raw 384d): {ds_acc_raw:.1f}%")


if __name__ == "__main__":
    main()
