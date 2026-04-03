"""Diagnostic probe: can VideoMAE features from 16-frame player clips classify actions?

Extracts VideoMAE CLS tokens (768-dim) from 16-frame player clips around
each GT contact. Trains a logistic regression with leave-one-video-out CV.

This captures approach motion (dig=low/platform, set=overhead) which is
the strongest visual discriminator for dig vs set.

Usage:
    cd analysis
    uv run python scripts/probe_videomae_action_features.py
    uv run python scripts/probe_videomae_action_features.py --pca-dims 32
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict

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
from rallycut.tracking.player_tracker import PlayerPosition as PlayerPos
from rallycut.tracking.visual_attribution import (
    extract_player_clip,
    compute_geom_features,
    PlayerClip,
    VisualAttributionClassifier,
)
from scripts.eval_action_detection import (
    RallyData,
    load_rallies_with_action_gt,
    match_contacts,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
console = Console()

ACTION_CLASSES = ["serve", "receive", "set", "attack", "dig"]


def _build_positions_by_frame(
    positions_json: list[dict],
    track_id: int,
) -> dict[int, dict[str, float]]:
    """Build frame-indexed lookup for a single track."""
    result: dict[int, dict[str, float]] = {}
    for pp in positions_json:
        if pp["trackId"] == track_id:
            result[pp["frameNumber"]] = {
                "x": pp["x"],
                "y": pp["y"],
                "width": pp["width"],
                "height": pp["height"],
            }
    return result


def extract_clips_for_rally(
    rally: RallyData,
    cap: cv2.VideoCapture,
    tolerance: int = 5,
) -> list[tuple[PlayerClip, str, str]]:
    """Extract 16-frame player clips at matched contact frames.

    Returns list of (PlayerClip, gt_action, video_id).
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

        contact = None
        for c in contact_seq.contacts:
            if c.frame == m.pred_frame:
                contact = c
                break
        if contact is None or contact.player_track_id < 0:
            continue

        positions_by_frame = _build_positions_by_frame(
            rally.positions_json, contact.player_track_id,
        )
        if not positions_by_frame:
            continue

        clip_frames = extract_player_clip(
            cap, positions_by_frame, contact.frame,
            rally_start_frame, frame_w, frame_h,
        )
        if clip_frames is None:
            continue

        geom = compute_geom_features(positions_by_frame, contact.frame)
        clip = PlayerClip(
            track_id=contact.player_track_id,
            frames=clip_frames,
            geom_features=geom,
        )
        results.append((clip, gt_action, rally.video_id))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="VideoMAE visual action probe")
    parser.add_argument("--pca-dims", type=int, default=32,
                        help="PCA dimensions (default: 32)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="VideoMAE batch size (default: 16)")
    args = parser.parse_args()

    console.print("[bold]Loading GT rallies...[/bold]")
    rallies = load_rallies_with_action_gt()
    console.print(f"  {len(rallies)} rallies loaded")

    by_video: dict[str, list[RallyData]] = defaultdict(list)
    for r in rallies:
        by_video[r.video_id].append(r)
    console.print(f"  {len(by_video)} videos")

    # Extract clips per video
    all_clips: list[PlayerClip] = []
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

        n_before = len(all_clips)
        for rally in video_rallies:
            clips_labels = extract_clips_for_rally(rally, cap)
            for clip, action, vid in clips_labels:
                all_clips.append(clip)
                all_labels.append(action)
                all_video_ids.append(vid)
        cap.release()

        n_new = len(all_clips) - n_before
        print(
            f"  [{vi+1}/{len(by_video)}] {video_id[:8]}: {n_new} clips "
            f"(total: {len(all_clips)})",
            flush=True,
        )

    if len(all_clips) < 10:
        console.print("[red]Too few clips extracted. Aborting.[/red]")
        sys.exit(1)

    console.print(f"\n[bold]Extracting VideoMAE features for {len(all_clips)} clips...[/bold]")

    # Extract VideoMAE features in batches
    classifier = VisualAttributionClassifier()
    all_features = []
    for i in range(0, len(all_clips), args.batch_size):
        batch = all_clips[i:i + args.batch_size]
        feats = classifier.extract_features(batch)  # (batch, 768+4)
        all_features.append(feats)
        done = min(i + args.batch_size, len(all_clips))
        console.print(f"  {done}/{len(all_clips)}")

    X_full = np.vstack(all_features)  # (N, 772)
    X_visual = X_full[:, :768]  # VideoMAE CLS only
    y = np.array(all_labels)
    video_ids = np.array(all_video_ids)

    console.print(f"  Feature shape: {X_full.shape}")

    # Class distribution
    console.print("\n[bold]Class distribution:[/bold]")
    for cls in ACTION_CLASSES:
        console.print(f"  {cls:10s}: {np.sum(y == cls)}")

    # ------- Evaluate: VideoMAE CLS only (768d → PCA) -------
    console.print(f"\n[bold]PCA: 768 → {args.pca_dims} dims (VideoMAE CLS)[/bold]")
    pca = PCA(n_components=args.pca_dims, random_state=42)
    X_pca = pca.fit_transform(X_visual)
    var_explained = np.sum(pca.explained_variance_ratio_) * 100
    console.print(f"  Variance explained: {var_explained:.1f}%")

    console.print("\n[bold]Leave-one-video-out CV (LogReg, PCA VideoMAE CLS)...[/bold]")
    unique_videos = sorted(set(all_video_ids))
    preds_pca = np.empty_like(y)

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
        preds_pca[mask_test] = clf.predict(X_test)

    acc = np.mean(preds_pca == y) * 100
    console.print(f"\n[bold green]Overall LOO-CV accuracy (PCA {args.pca_dims}d): {acc:.1f}%[/bold green]")
    print(classification_report(y, preds_pca, labels=ACTION_CLASSES, digits=3))

    # Confusion matrix
    console.print("[bold]Confusion matrix:[/bold]")
    cm = confusion_matrix(y, preds_pca, labels=ACTION_CLASSES)
    table = Table(show_header=True)
    table.add_column("True\\Pred", style="bold")
    for cls in ACTION_CLASSES:
        table.add_column(cls, justify="right")
    for i, cls in enumerate(ACTION_CLASSES):
        table.add_row(cls, *[str(v) for v in cm[i]])
    console.print(table)

    # Dig vs Set focus
    ds_mask = (y == "dig") | (y == "set")
    if ds_mask.sum() > 0:
        ds_acc = np.mean(preds_pca[ds_mask] == y[ds_mask]) * 100
        console.print(f"\n[bold]Dig vs Set accuracy (PCA {args.pca_dims}d): {ds_acc:.1f}%[/bold]")

    # ------- Evaluate: Full features (768 + 4 geom) -------
    console.print(f"\n[bold]--- Full features (768+4 geom, PCA → {args.pca_dims}d) ---[/bold]")
    pca_full = PCA(n_components=args.pca_dims, random_state=42)
    X_pca_full = pca_full.fit_transform(X_full)

    preds_full = np.empty_like(y)
    for vid in unique_videos:
        mask_test = video_ids == vid
        mask_train = ~mask_test
        X_train, y_train = X_pca_full[mask_train], y[mask_train]
        X_test = X_pca_full[mask_test]
        if len(X_test) == 0:
            continue
        clf = LogisticRegression(
            C=0.1, max_iter=1000, random_state=42, solver="lbfgs",
        )
        clf.fit(X_train, y_train)
        preds_full[mask_test] = clf.predict(X_test)

    acc_full = np.mean(preds_full == y) * 100
    console.print(f"Overall LOO-CV accuracy (full 772d → PCA): {acc_full:.1f}%")
    if ds_mask.sum() > 0:
        ds_acc_full = np.mean(preds_full[ds_mask] == y[ds_mask]) * 100
        console.print(f"Dig vs Set accuracy (full 772d → PCA): {ds_acc_full:.1f}%")

    # ------- Summary -------
    console.print("\n[bold]Summary:[/bold]")
    console.print(f"  VideoMAE CLS (PCA {args.pca_dims}d): {acc:.1f}% overall")
    console.print(f"  VideoMAE + geom (PCA {args.pca_dims}d): {acc_full:.1f}% overall")
    console.print(f"  Trajectory-only GBM baseline: 83.1% (from memory)")


if __name__ == "__main__":
    main()
