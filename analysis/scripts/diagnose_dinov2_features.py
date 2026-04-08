#!/usr/bin/env python3
"""DINOv2 feature diagnostic for cross-rally player matching.

Extracts DINOv2 CLS embeddings from player crops of GT-labeled videos
and measures within-team vs cross-team discriminability gap.

GO threshold: gap > 0.10 (OSNet was 0.088 — not discriminative enough).

Usage:
    cd analysis
    uv run python scripts/diagnose_dinov2_features.py
    uv run python scripts/diagnose_dinov2_features.py --model dinov2_vitb14  # larger model
    uv run python scripts/diagnose_dinov2_features.py --num-samples 20       # more frames
"""

from __future__ import annotations

import argparse
import logging
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as functional
from rich.console import Console

from rallycut.evaluation.db import get_connection
from rallycut.evaluation.gt_loader import load_all_from_db
from rallycut.evaluation.tracking.db import get_video_path, load_rallies_for_video

logging.basicConfig(level=logging.WARNING, format="%(message)s")
console = Console()


def load_gt_videos() -> list[dict]:
    """Load videos that have player matching ground truth."""
    results = []
    with get_connection() as conn:
        with conn.cursor() as cur:
            db_rows = load_all_from_db(cur)
            # Also fetch name via a second query keyed by id.
            cur.execute(
                "SELECT id, name FROM videos WHERE player_matching_gt_json IS NOT NULL"
            )
            name_by_id = {str(r[0]): r[1] for r in cur.fetchall()}
    for db_row in db_rows:
        vid = db_row.video_id
        video_path = get_video_path(vid)
        if video_path:
            results.append({
                "video_id": vid,
                "title": name_by_id.get(vid) or vid[:8],
                "local_path": str(video_path),
                # Shape matches what analyze_discriminability expects:
                # {"rallies": {rally_id: {track_id_str: player_id_int}}}
                "gt": {"rallies": db_row.gt.rallies},
            })
    return results


def extract_player_crops(
    video_path: str,
    rallies: list,
    num_samples: int = 12,
) -> dict[str, list[np.ndarray]]:
    """Extract player crops keyed by 'rally_id:track_id'.

    Returns dict mapping key -> list of BGR crops (224x224).
    """
    crops: dict[str, list[np.ndarray]] = defaultdict(list)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        console.print(f"[red]Cannot open {video_path}[/red]")
        return crops

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    for rally in rallies:
        rid = rally.rally_id
        start_frame = int(rally.start_ms / 1000 * fps)
        positions = rally.positions

        # Group by track
        by_track: dict[int, list] = defaultdict(list)
        for p in positions:
            if p.track_id in rally.primary_track_ids:
                by_track[p.track_id].append(p)

        # Sample frames per track
        frame_requests: dict[int, list[tuple[int, object]]] = {}
        for tid, pos_list in by_track.items():
            pos_list.sort(key=lambda p: p.frame_number)
            n = len(pos_list)
            if n <= num_samples:
                indices = list(range(n))
            else:
                indices = [int(i * (n - 1) / (num_samples - 1)) for i in range(num_samples)]

            for idx in indices:
                p = pos_list[idx]
                fn = p.frame_number
                if fn not in frame_requests:
                    frame_requests[fn] = []
                frame_requests[fn].append((tid, p))

        for fn in sorted(frame_requests.keys()):
            abs_frame = start_frame + fn
            cap.set(cv2.CAP_PROP_POS_FRAMES, abs_frame)
            ret, frame = cap.read()
            if not ret:
                continue

            for tid, p in frame_requests[fn]:
                # Extract crop
                cx, cy, w, h = p.x, p.y, p.width, p.height
                x1 = max(0, int((cx - w / 2) * frame_w))
                y1 = max(0, int((cy - h / 2) * frame_h))
                x2 = min(frame_w, int((cx + w / 2) * frame_w))
                y2 = min(frame_h, int((cy + h / 2) * frame_h))

                if x2 <= x1 or y2 <= y1:
                    continue

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                # Resize to 224x224 for DINOv2
                crop_resized = cv2.resize(crop, (224, 224))
                key = f"{rid}:{tid}"
                crops[key].append(crop_resized)

    cap.release()
    return crops


def extract_dinov2_embeddings(
    crops_by_key: dict[str, list[np.ndarray]],
    model_name: str = "dinov2_vits14",
) -> dict[str, np.ndarray]:
    """Extract DINOv2 CLS embeddings from crops.

    Returns dict mapping key -> mean embedding (L2-normalized).
    """
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    console.print(f"  Loading DINOv2 ({model_name}) on {device}...")

    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model = model.to(device)
    model.eval()

    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    embeddings: dict[str, np.ndarray] = {}

    with torch.inference_mode():
        for key, crop_list in crops_by_key.items():
            if not crop_list:
                continue

            # Stack crops into batch
            batch = []
            for crop in crop_list:
                # BGR -> RGB, HWC -> CHW, normalize
                img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                batch.append(tensor)

            batch_tensor = torch.stack(batch).to(device)
            batch_tensor = (batch_tensor - mean) / std

            # Extract CLS token embeddings
            features = model(batch_tensor)  # (N, embed_dim)

            # Average and L2-normalize
            mean_feat = features.mean(dim=0)
            mean_feat = functional.normalize(mean_feat, dim=0)
            embeddings[key] = mean_feat.cpu().numpy()

    return embeddings


def analyze_discriminability(
    embeddings: dict[str, np.ndarray],
    gt_data: dict,
    rallies: list,
) -> dict[str, float]:
    """Compute within-team vs cross-team cosine distance statistics.

    Returns dict with gap and detailed stats.
    """
    # Build rally_id -> {track_id: player_id} from GT
    # GT structure: {"rallies": {rally_id: {track_id_str: player_id}}}
    gt_rallies: dict[str, dict[int, int]] = {}
    gt_rallies_raw = gt_data.get("rallies", {})
    for rid, t2p in gt_rallies_raw.items():
        if rid and t2p:
            gt_rallies[rid] = {int(k): int(v) for k, v in t2p.items()}

    # Compute all pairwise distances within and across teams
    within_team_dists: list[float] = []
    cross_team_dists: list[float] = []
    within_player_dists: list[float] = []  # Same player, different rally

    keys = list(embeddings.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            k1, k2 = keys[i], keys[j]
            r1, t1 = k1.rsplit(":", 1)
            r2, t2 = k2.rsplit(":", 1)
            t1_int, t2_int = int(t1), int(t2)

            # Get player IDs from GT
            if r1 not in gt_rallies or r2 not in gt_rallies:
                continue
            if t1_int not in gt_rallies[r1] or t2_int not in gt_rallies[r2]:
                continue

            p1 = gt_rallies[r1][t1_int]
            p2 = gt_rallies[r2][t2_int]

            # Cosine distance (1 - cosine_similarity)
            cos_dist = 1.0 - float(np.dot(embeddings[k1], embeddings[k2]))

            team1 = 0 if p1 <= 2 else 1
            team2 = 0 if p2 <= 2 else 1

            if p1 == p2:
                within_player_dists.append(cos_dist)
                within_team_dists.append(cos_dist)
            elif team1 == team2:
                within_team_dists.append(cos_dist)
            else:
                cross_team_dists.append(cos_dist)

    results: dict[str, float] = {}
    if within_team_dists:
        results["within_team_mean"] = float(np.mean(within_team_dists))
        results["within_team_median"] = float(np.median(within_team_dists))
    if cross_team_dists:
        results["cross_team_mean"] = float(np.mean(cross_team_dists))
        results["cross_team_median"] = float(np.median(cross_team_dists))
    if within_player_dists:
        results["within_player_mean"] = float(np.mean(within_player_dists))

    if within_team_dists and cross_team_dists:
        gap = results["cross_team_mean"] - results["within_team_mean"]
        results["discriminability_gap"] = gap

    results["n_within_team"] = float(len(within_team_dists))
    results["n_cross_team"] = float(len(cross_team_dists))
    results["n_within_player"] = float(len(within_player_dists))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="DINOv2 feature diagnostic for player matching")
    parser.add_argument("--model", default="dinov2_vits14", help="DINOv2 model variant (default: dinov2_vits14)")
    parser.add_argument("--num-samples", type=int, default=12, help="Frames per track (default: 12)")
    parser.add_argument("--max-videos", type=int, default=5, help="Max videos to process (default: 5)")
    args = parser.parse_args()

    videos = load_gt_videos()
    if not videos:
        console.print("[red]No videos with player matching GT found[/red]")
        return

    videos = videos[:args.max_videos]
    console.print("\n[bold]DINOv2 Feature Diagnostic[/bold]")
    console.print(f"  Model: {args.model}")
    console.print(f"  Videos: {len(videos)}")
    console.print(f"  Samples/track: {args.num_samples}\n")

    all_results: list[dict[str, float]] = []

    for video in videos:
        vid = video["video_id"]
        title = video["title"]
        local_path = video["local_path"]

        if not Path(local_path).exists():
            console.print(f"  [yellow]Skipping {title}: file not found[/yellow]")
            continue

        console.print(f"  Processing {title}...")
        t0 = time.monotonic()

        rallies = load_rallies_for_video(vid)
        if not rallies:
            console.print(f"  [yellow]No rallies for {title}[/yellow]")
            continue

        # Limit to first 10 rallies for speed
        rallies = rallies[:10]

        crops = extract_player_crops(local_path, rallies, num_samples=args.num_samples)
        if not crops:
            console.print("  [yellow]No crops extracted[/yellow]")
            continue

        embeddings = extract_dinov2_embeddings(crops, model_name=args.model)

        results = analyze_discriminability(embeddings, video["gt"], rallies)
        results["video"] = 0.0  # placeholder
        all_results.append(results)

        elapsed = time.monotonic() - t0
        gap = results.get("discriminability_gap", 0)
        console.print(
            f"    Gap: {gap:.4f} "
            f"(within-team: {results.get('within_team_mean', 0):.4f}, "
            f"cross-team: {results.get('cross_team_mean', 0):.4f}) "
            f"[{elapsed:.1f}s]"
        )

    if not all_results:
        console.print("[red]No results[/red]")
        return

    # Summary
    console.print("\n[bold]Summary[/bold]")
    avg_gap = np.mean([r.get("discriminability_gap", 0) for r in all_results])
    avg_within = np.mean([r.get("within_team_mean", 0) for r in all_results])
    avg_cross = np.mean([r.get("cross_team_mean", 0) for r in all_results])
    avg_within_player = np.mean([r.get("within_player_mean", 0) for r in all_results if "within_player_mean" in r])

    console.print(f"  Avg within-team distance:   {avg_within:.4f}")
    console.print(f"  Avg cross-team distance:    {avg_cross:.4f}")
    console.print(f"  Avg within-player distance: {avg_within_player:.4f}")
    console.print(f"  [bold]Discriminability gap:        {avg_gap:.4f}[/bold]")

    if avg_gap > 0.10:
        console.print(f"\n  [green]GO: gap {avg_gap:.4f} > 0.10 threshold. DINOv2 features are discriminative![/green]")
    elif avg_gap > 0.05:
        console.print(f"\n  [yellow]MARGINAL: gap {avg_gap:.4f}. May help as part of ensemble.[/yellow]")
    else:
        console.print(f"\n  [red]NO-GO: gap {avg_gap:.4f} <= 0.05. DINOv2 not discriminative enough.[/red]")


if __name__ == "__main__":
    main()
