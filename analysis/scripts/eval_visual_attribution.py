"""Evaluate visual attribution with leave-one-video-out cross-validation.

Extracts VideoMAE features from player crops at contact frames, trains a
binary acting/not-acting classifier, and evaluates attribution accuracy
via LOO-CV. Compares to the 67.8% proximity baseline.

Usage:
    cd analysis
    uv run python scripts/eval_visual_attribution.py
    uv run python scripts/eval_visual_attribution.py --cache features.npz
    uv run python scripts/eval_visual_attribution.py --train  # Train + save production model
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from rich.console import Console
from rich.table import Table

from rallycut.evaluation.db import get_connection
from rallycut.tracking.match_tracker import build_match_team_assignments
from rallycut.tracking.visual_attribution import (
    FRAME_WINDOW,
    VisualAttributionClassifier,
    build_positions_by_frame,
    compute_geom_features,
    extract_player_clip,
    get_same_side_track_ids,
)

console = Console()


@dataclass
class ContactSample:
    """A single (contact, candidate_player) sample."""
    video_id: str
    rally_id: str
    contact_frame: int
    action: str
    track_id: int
    is_gt: bool
    features: np.ndarray | None  # (768,) or (772,) if geom included
    geom_features: list[float]  # [vert_disp, ar_change, h_change, horiz_disp]
    contact_group: int  # groups samples from same contact


def load_gt_contacts_with_positions() -> list[dict[str, Any]]:
    """Load GT contacts with all required data from DB."""
    query = """
        SELECT
            r.id AS rally_id,
            r.video_id,
            pt.action_ground_truth_json,
            pt.positions_json,
            pt.court_split_y,
            r.start_ms,
            pt.fps
        FROM rallies r
        JOIN player_tracks pt ON pt.rally_id = r.id
        WHERE pt.action_ground_truth_json IS NOT NULL
          AND pt.positions_json IS NOT NULL
        ORDER BY r.video_id, r.start_ms
    """
    results: list[dict[str, Any]] = []
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            for row in cur.fetchall():
                rally_id, video_id, gt_json, pos_json, split_y, start_ms, fps = row
                if not gt_json or not pos_json:
                    continue
                for label in gt_json:
                    tid = label.get("playerTrackId", -1)
                    if tid < 0:
                        continue
                    results.append({
                        "rally_id": rally_id,
                        "video_id": video_id,
                        "frame": label["frame"],
                        "action": label["action"],
                        "player_track_id": tid,
                        "positions_json": pos_json,
                        "court_split_y": split_y,
                        "start_ms": start_ms or 0,
                        "fps": fps or 30.0,
                    })
    return results


def load_team_assignments() -> dict[str, dict[int, int]]:
    """Load match-level team assignments."""
    query = """
        SELECT id, match_analysis_json
        FROM videos WHERE match_analysis_json IS NOT NULL
    """
    result: dict[str, dict[int, int]] = {}
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            for _vid, ma_json in cur.fetchall():
                if isinstance(ma_json, dict):
                    result.update(build_match_team_assignments(ma_json, 0.0))
    return result


def extract_all_samples(
    contacts: list[dict[str, Any]],
    all_teams: dict[str, dict[int, int]],
    cache_path: Path | None = None,
) -> list[ContactSample]:
    """Extract VideoMAE features for all contacts. Optionally cache to disk."""
    from lib.volleyball_ml.video_mae import GameStateClassifier
    from rallycut.evaluation.tracking.db import get_video_path

    # Check cache
    if cache_path and cache_path.exists():
        console.print(f"  Loading cached features from {cache_path}")
        return _load_cache(cache_path)

    model = GameStateClassifier()
    samples: list[ContactSample] = []
    contact_group = 0

    # Group by video for efficient access
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in contacts:
        by_video[c["video_id"]].append(c)

    video_ids = sorted(by_video.keys())
    console.print(f"  {len(contacts)} contacts across {len(video_ids)} videos")

    for vid_num, video_id in enumerate(video_ids):
        vid_contacts = by_video[video_id]
        video_path = get_video_path(video_id)
        if video_path is None:
            continue

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        try:
            for contact in vid_contacts:
                teams = all_teams.get(contact["rally_id"])
                gt_tid = contact["player_track_id"]
                contact_frame = contact["frame"]
                positions_json = contact["positions_json"]

                same_side = get_same_side_track_ids(
                    positions_json, gt_tid, contact_frame,
                    teams, contact["court_split_y"],
                )
                if not same_side:
                    continue

                rally_start_frame = int(contact["start_ms"] / 1000.0 * fps)
                all_tids = [gt_tid] + same_side

                # Extract clips for all candidates
                clips_data: list[tuple[int, bool, list[np.ndarray], list[float]]] = []
                for tid in all_tids:
                    is_gt = (tid == gt_tid)
                    pos_by_frame = build_positions_by_frame(positions_json, tid)
                    if not pos_by_frame:
                        continue

                    clip_frames = extract_player_clip(
                        cap, pos_by_frame, contact_frame,
                        rally_start_frame, frame_w, frame_h,
                    )
                    if clip_frames is None:
                        continue

                    geom = compute_geom_features(pos_by_frame, contact_frame)
                    clips_data.append((tid, is_gt, clip_frames, geom))

                if len(clips_data) < 2:
                    continue

                # Extract VideoMAE features in batch
                batch = [frames for _, _, frames, _ in clips_data]
                try:
                    feats = model.get_encoder_features_batch(batch, pooling="cls")
                except Exception:
                    continue

                for i, (tid, is_gt, _, geom) in enumerate(clips_data):
                    samples.append(ContactSample(
                        video_id=video_id,
                        rally_id=contact["rally_id"],
                        contact_frame=contact_frame,
                        action=contact["action"],
                        track_id=tid,
                        is_gt=is_gt,
                        features=feats[i],
                        geom_features=geom,
                        contact_group=contact_group,
                    ))

                contact_group += 1
        finally:
            cap.release()

        console.print(
            f"  [{vid_num + 1}/{len(video_ids)}] {video_id[:8]}: "
            f"{contact_group} contacts, {len(samples)} samples",
        )

    # Save cache
    if cache_path and samples:
        _save_cache(samples, cache_path)
        console.print(f"  Cached features to {cache_path}")

    return samples


def _save_cache(samples: list[ContactSample], path: Path) -> None:
    """Save extracted features to disk."""
    import json
    path.parent.mkdir(parents=True, exist_ok=True)

    features = np.array([s.features for s in samples])
    geom = np.array([s.geom_features for s in samples])
    meta = [
        {
            "video_id": s.video_id,
            "rally_id": s.rally_id,
            "contact_frame": s.contact_frame,
            "action": s.action,
            "track_id": s.track_id,
            "is_gt": s.is_gt,
            "contact_group": s.contact_group,
        }
        for s in samples
    ]
    np.savez_compressed(
        path,
        features=features,
        geom=geom,
        meta=json.dumps(meta),
    )


def _load_cache(path: Path) -> list[ContactSample]:
    """Load cached features from disk."""
    import json

    data = np.load(path, allow_pickle=False)
    features = data["features"]
    geom = data["geom"]
    meta = json.loads(str(data["meta"]))

    samples: list[ContactSample] = []
    for i, m in enumerate(meta):
        samples.append(ContactSample(
            video_id=m["video_id"],
            rally_id=m["rally_id"],
            contact_frame=m["contact_frame"],
            action=m["action"],
            track_id=m["track_id"],
            is_gt=m["is_gt"],
            features=features[i],
            geom_features=geom[i].tolist(),
            contact_group=m["contact_group"],
        ))
    return samples


def run_loocv(
    samples: list[ContactSample],
) -> None:
    """Run leave-one-video-out cross-validation."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    video_ids = sorted(set(s.video_id for s in samples))
    n_contacts = len(set(s.contact_group for s in samples))

    console.print(f"\n[bold]LOO-CV: {n_contacts} contacts, {len(samples)} samples, "
                  f"{len(video_ids)} videos[/bold]\n")

    configs = [
        ("VideoMAE + Geom", True, True),
        ("VideoMAE only", True, False),
        ("Geom only", False, True),
    ]

    for config_name, use_visual, use_geom in configs:
        correct = 0
        total = 0
        per_action_correct: dict[str, int] = defaultdict(int)
        per_action_total: dict[str, int] = defaultdict(int)

        for held_out_vid in video_ids:
            train = [s for s in samples if s.video_id != held_out_vid]
            test = [s for s in samples if s.video_id == held_out_vid]
            if not train or not test:
                continue

            def _build_x(sample_list: list[ContactSample]) -> np.ndarray:
                parts = []
                for s in sample_list:
                    fp: list[np.ndarray] = []
                    if use_visual and s.features is not None:
                        fp.append(s.features)
                    if use_geom:
                        fp.append(np.array(s.geom_features))
                    parts.append(np.concatenate(fp))
                return np.array(parts)

            X_train = _build_x(train)
            y_train = np.array([1 if s.is_gt else 0 for s in train])
            X_test = _build_x(test)

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
            clf.fit(X_train_s, y_train)
            probs = clf.predict_proba(X_test_s)[:, 1]

            # Pick highest-scoring candidate per contact group
            by_group: dict[int, list[tuple[float, bool, str]]] = defaultdict(list)
            for s, p in zip(test, probs):
                by_group[s.contact_group].append((p, s.is_gt, s.action))

            for group_candidates in by_group.values():
                group_candidates.sort(key=lambda x: x[0], reverse=True)
                picked_is_gt = group_candidates[0][1]
                action = group_candidates[0][2]
                total += 1
                per_action_total[action] += 1
                if picked_is_gt:
                    correct += 1
                    per_action_correct[action] += 1

        rate = 100 * correct / total if total > 0 else 0
        console.print(f"  [bold]{config_name}[/bold]: {correct}/{total} = "
                      f"[bold green]{rate:.1f}%[/bold green]")

        table = Table(title=f"{config_name} per Action")
        table.add_column("Action", style="cyan")
        table.add_column("Correct", justify="right")
        table.add_column("Total", justify="right")
        table.add_column("Rate", justify="right", style="bold")

        for act in ["serve", "receive", "set", "attack", "dig", "block"]:
            n = per_action_total.get(act, 0)
            c = per_action_correct.get(act, 0)
            if n == 0:
                continue
            table.add_row(act, str(c), str(n), f"{100 * c / n:.1f}%")

        console.print(table)

    console.print(f"\n  [dim]Proximity baseline: 67.8%[/dim]")


def train_production_model(samples: list[ContactSample]) -> None:
    """Train on all data and save production model."""
    console.print("\n[bold]Training production model on all data...[/bold]")

    # Build feature matrix
    X_parts = []
    y_parts = []
    for s in samples:
        if s.features is None:
            continue
        feat = np.concatenate([s.features, np.array(s.geom_features)])
        X_parts.append(feat)
        y_parts.append(1 if s.is_gt else 0)

    X = np.array(X_parts)
    y = np.array(y_parts)

    classifier = VisualAttributionClassifier()
    # We need to set the scaler and model directly since we have pre-extracted features
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    classifier._scaler = StandardScaler()
    X_scaled = classifier._scaler.fit_transform(X)
    classifier._model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    classifier._model.fit(X_scaled, y)
    classifier._use_geom = True

    train_probs = classifier._model.predict_proba(X_scaled)[:, 1]
    train_preds = (train_probs >= 0.5).astype(int)
    acc = float(np.mean(train_preds == y))

    console.print(f"  Train accuracy: {acc:.1%}")
    console.print(f"  Samples: {len(y)} ({int(y.sum())} positive, {int((1 - y).sum())} negative)")

    classifier.save()
    console.print(f"  [green]Saved to {classifier.save.__defaults__}[/green]")  # type: ignore[union-attr]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=str, default=None,
                        help="Path to cache extracted features (e.g., features.npz)")
    parser.add_argument("--train", action="store_true",
                        help="Train and save production model after LOO-CV")
    args = parser.parse_args()

    cache_path = Path(args.cache) if args.cache else None

    console.print("[bold]Loading GT contacts...[/bold]")
    contacts = load_gt_contacts_with_positions()
    console.print(f"  {len(contacts)} GT contacts")

    console.print("[bold]Loading team assignments...[/bold]")
    all_teams = load_team_assignments()
    console.print(f"  {len(all_teams)} rallies with teams")

    console.print("[bold]Extracting VideoMAE features...[/bold]")
    t0 = time.time()
    samples = extract_all_samples(contacts, all_teams, cache_path)
    elapsed = time.time() - t0
    console.print(f"  {len(samples)} samples in {elapsed:.0f}s")

    if not samples:
        console.print("[red]No samples extracted. Check video access.[/red]")
        return

    run_loocv(samples)

    if args.train:
        train_production_model(samples)


if __name__ == "__main__":
    main()
