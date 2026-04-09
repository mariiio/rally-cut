"""Diagnose whether player bbox motion at contact time discriminates the hitting player.

Hypothesis: the hitting player's bbox changes measurably at contact — arm swing
widens bbox, jump shifts Y, dive changes aspect ratio. If true, this is a free
feature for player attribution (no model needed).

For each GT contact in the 126 labeled rallies, we compare bbox deltas in ±5 frames
for the GT player vs all other visible players, then report effect sizes and AUCs.

Usage:
    cd analysis
    uv run python scripts/diagnose_bbox_motion.py
"""

from __future__ import annotations

import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from rich.console import Console
from rich.table import Table
from sklearn.metrics import roc_auc_score

# Reuse the GT loader from eval_action_detection
sys.path.insert(0, "scripts")
from eval_action_detection import GtLabel, RallyData, load_rallies_with_action_gt

from rallycut.tracking.player_tracker import PlayerPosition

console = Console()

WINDOW = 5  # ±5 frames around contact
MIN_FRAMES_IN_WINDOW = 3  # skip tracks with fewer observations

METRICS = ["d_width", "d_height", "d_y", "d_aspect", "d_area",
           "max_d_width", "max_d_height", "max_d_y", "max_d_aspect", "max_d_area"]


@dataclass
class TrackWindow:
    """Bbox observations for one track in a ±WINDOW frame range."""
    frames: dict[int, PlayerPosition] = field(default_factory=dict)

    def delta(self, center: int) -> dict[str, float] | None:
        """Compute bbox deltas across the window. Returns None if insufficient data."""
        if len(self.frames) < MIN_FRAMES_IN_WINDOW:
            return None

        sorted_frames = sorted(self.frames.keys())
        first = self.frames[sorted_frames[0]]
        last = self.frames[sorted_frames[-1]]

        def aspect(p: PlayerPosition) -> float:
            return p.width / p.height if p.height > 1e-6 else 0.0

        # Endpoint deltas
        d_width = abs(last.width - first.width)
        d_height = abs(last.height - first.height)
        d_y = abs(last.y - first.y)
        d_aspect = abs(aspect(last) - aspect(first))
        d_area = abs(last.width * last.height - first.width * first.height)

        # Max abs frame-to-frame delta across window
        positions = [self.frames[f] for f in sorted_frames]
        max_d_width = max(abs(positions[i+1].width - positions[i].width) for i in range(len(positions)-1))
        max_d_height = max(abs(positions[i+1].height - positions[i].height) for i in range(len(positions)-1))
        max_d_y = max(abs(positions[i+1].y - positions[i].y) for i in range(len(positions)-1))
        max_d_aspect = max(abs(aspect(positions[i+1]) - aspect(positions[i])) for i in range(len(positions)-1))
        max_d_area = max(
            abs(positions[i+1].width * positions[i+1].height - positions[i].width * positions[i].height)
            for i in range(len(positions)-1)
        )

        return {
            "d_width": d_width, "d_height": d_height, "d_y": d_y,
            "d_aspect": d_aspect, "d_area": d_area,
            "max_d_width": max_d_width, "max_d_height": max_d_height,
            "max_d_y": max_d_y, "max_d_aspect": max_d_aspect, "max_d_area": max_d_area,
        }


def parse_positions(positions_json: list[dict]) -> list[PlayerPosition]:
    return [
        PlayerPosition(
            frame_number=p["frameNumber"],
            track_id=p["trackId"],
            x=p["x"], y=p["y"],
            width=p["width"], height=p["height"],
            confidence=p.get("confidence", 1.0),
        )
        for p in positions_json
    ]


def build_track_windows(
    positions: list[PlayerPosition], contact_frame: int,
) -> dict[int, TrackWindow]:
    """Build per-track bbox windows around a contact frame."""
    windows: dict[int, TrackWindow] = defaultdict(TrackWindow)
    for p in positions:
        if abs(p.frame_number - contact_frame) <= WINDOW:
            windows[p.track_id].frames[p.frame_number] = p
    return dict(windows)


def cohens_d(a: list[float], b: list[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    na, nb = len(a), len(b)
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled < 1e-12:
        return 0.0
    return (np.mean(a) - np.mean(b)) / pooled


def main() -> None:
    console.print("[bold]Loading rallies with action GT...[/bold]")
    rallies = load_rallies_with_action_gt()
    console.print(f"Loaded {len(rallies)} rallies")

    # Collect per-metric values for GT player and others
    gt_values: dict[str, list[float]] = defaultdict(list)
    other_values: dict[str, list[float]] = defaultdict(list)
    # Per-action breakdown
    action_gt: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    action_other: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    # Track how often GT player ranks #1 per metric
    gt_rank1: dict[str, int] = defaultdict(int)
    gt_evaluable: dict[str, int] = defaultdict(int)

    total_contacts = 0
    skipped_no_track = 0
    skipped_no_positions = 0
    skipped_sparse = 0
    evaluated = 0

    contact_idx = 0
    for rally in rallies:
        if not rally.positions_json:
            skipped_no_positions += len([g for g in rally.gt_labels if g.player_track_id != -1])
            total_contacts += len(rally.gt_labels)
            continue

        positions = parse_positions(rally.positions_json)

        for gt in rally.gt_labels:
            total_contacts += 1
            if gt.player_track_id == -1:
                skipped_no_track += 1
                continue

            windows = build_track_windows(positions, gt.frame)

            # GT player must have sufficient frames
            gt_window = windows.get(gt.player_track_id)
            if gt_window is None or len(gt_window.frames) < MIN_FRAMES_IN_WINDOW:
                skipped_sparse += 1
                continue

            gt_deltas = gt_window.delta(gt.frame)
            if gt_deltas is None:
                skipped_sparse += 1
                continue

            # Collect deltas for all other tracks
            other_deltas_list: list[dict[str, float]] = []
            for tid, tw in windows.items():
                if tid == gt.player_track_id:
                    continue
                d = tw.delta(gt.frame)
                if d is not None:
                    other_deltas_list.append(d)

            if not other_deltas_list:
                skipped_sparse += 1
                continue

            evaluated += 1
            contact_idx += 1

            # Store values
            for m in METRICS:
                gt_val = gt_deltas[m]
                gt_values[m].append(gt_val)
                action_gt[gt.action][m].append(gt_val)

                for od in other_deltas_list:
                    other_values[m].append(od[m])
                    action_other[gt.action][m].append(od[m])

                # Is GT player rank #1 for this metric?
                all_vals = [(gt.player_track_id, gt_val)]
                for od_idx, od in enumerate(other_deltas_list):
                    all_vals.append((od_idx, od[m]))
                all_vals.sort(key=lambda x: x[1], reverse=True)
                gt_evaluable[m] += 1
                if all_vals[0][0] == gt.player_track_id:
                    gt_rank1[m] += 1

            if contact_idx % 50 == 0 or contact_idx <= 5:
                console.print(
                    f"  [{contact_idx}] rally={rally.rally_id[:8]} frame={gt.frame} "
                    f"gt_track={gt.player_track_id} action={gt.action} "
                    f"Δwidth_gt={gt_deltas['d_width']:.4f} "
                    f"Δwidth_other_avg={np.mean([od['d_width'] for od in other_deltas_list]):.4f}"
                )

    # --- Summary ---
    console.print()
    console.print(f"[bold]Contacts: {total_contacts} total, {evaluated} evaluated[/bold]")
    console.print(f"  Skipped: {skipped_no_track} no GT track, {skipped_no_positions} no positions, {skipped_sparse} sparse")
    console.print()

    # Overall table
    table = Table(title="Bbox Motion Discrimination (GT player vs Others)")
    table.add_column("Metric", style="cyan")
    table.add_column("GT mean", justify="right")
    table.add_column("Other mean", justify="right")
    table.add_column("Cohen's d", justify="right")
    table.add_column("AUC", justify="right")
    table.add_column("GT rank #1", justify="right")

    for m in METRICS:
        g = gt_values[m]
        o = other_values[m]
        d = cohens_d(g, o)

        # AUC: 1=GT, 0=other
        labels = [1] * len(g) + [0] * len(o)
        scores = g + o
        try:
            auc = roc_auc_score(labels, scores)
        except ValueError:
            auc = 0.5

        rank1_pct = gt_rank1[m] / gt_evaluable[m] * 100 if gt_evaluable[m] > 0 else 0

        table.add_row(
            m,
            f"{np.mean(g):.5f}",
            f"{np.mean(o):.5f}",
            f"{d:.3f}",
            f"{auc:.3f}",
            f"{rank1_pct:.1f}%",
        )

    console.print(table)

    # Per-action breakdown
    actions_seen = sorted(set(action_gt.keys()))
    for m in METRICS:
        at = Table(title=f"Per-Action: {m}")
        at.add_column("Action", style="cyan")
        at.add_column("N", justify="right")
        at.add_column("GT mean", justify="right")
        at.add_column("Other mean", justify="right")
        at.add_column("Cohen's d", justify="right")
        at.add_column("AUC", justify="right")

        for action in actions_seen:
            g = action_gt[action][m]
            o = action_other[action][m]
            if len(g) < 5:
                continue
            d = cohens_d(g, o)
            labels = [1] * len(g) + [0] * len(o)
            scores = g + o
            try:
                auc = roc_auc_score(labels, scores)
            except ValueError:
                auc = 0.5
            at.add_row(action, str(len(g)), f"{np.mean(g):.5f}", f"{np.mean(o):.5f}",
                        f"{d:.3f}", f"{auc:.3f}")

        console.print(at)


if __name__ == "__main__":
    main()
