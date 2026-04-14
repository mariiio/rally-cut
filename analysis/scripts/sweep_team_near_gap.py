"""Sweep min_y_gap threshold for localize_team_near.

Tests whether raising the 3% threshold increases accuracy (with lower
return rate). Also tests majority-vote across a video to correct outliers.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rich.console import Console
from rich.table import Table

from eval_action_detection import _load_track_to_player_maps  # noqa: E402
from eval_score_tracking import load_score_gt  # noqa: E402
from production_eval import _load_team_templates_by_video  # noqa: E402
from rallycut.tracking.action_classifier import (  # noqa: E402
    _find_serving_side_by_formation,
)
from rallycut.tracking.player_tracker import PlayerPosition  # noqa: E402
from rallycut.tracking.team_identity import localize_team_near  # noqa: E402

console = Console()


def _parse_positions(pos_json: list[dict]) -> list[PlayerPosition]:
    return [
        PlayerPosition(
            track_id=p.get("trackId", p.get("track_id", -1)),
            frame_number=p.get("frameNumber", p.get("frame_number", 0)),
            x=p["x"], y=p["y"],
            width=p["width"], height=p["height"],
            confidence=p.get("confidence", 1.0),
        )
        for p in pos_json
    ]


def main() -> int:
    console.print("[bold]Loading score GT...[/bold]")
    video_rallies = load_score_gt()
    video_ids = set(video_rallies.keys())
    templates_by_vid = _load_team_templates_by_video(video_ids)
    t2p_by_rally = _load_track_to_player_maps(video_ids)
    console.print(f"  {len(templates_by_vid)}/{len(video_ids)} videos with templates\n")

    # Gather all GT rallies with templates and t2p
    samples = []  # (vid, rally_id, positions, t2p, templates, expected_tn)

    # First pass: compute label_a per video
    vid_label_a: dict[str, str] = {}
    for vid, rallies in sorted(video_rallies.items()):
        templates = templates_by_vid.get(vid)
        if templates is None:
            continue
        t0, t1 = templates
        votes: dict[str, int] = {t0.team_label: 0, t1.team_label: 0}
        for rally in rallies:
            if rally.gt_serving_team is None:
                continue
            t2p = t2p_by_rally.get(rally.rally_id)
            if not t2p:
                continue
            positions = _parse_positions(rally.positions)
            net_y = rally.court_split_y if rally.court_split_y else 0.5
            formation_side, _ = _find_serving_side_by_formation(
                positions, net_y=net_y, start_frame=0,
            )
            if formation_side is None:
                continue
            tn = localize_team_near(positions, t2p, templates, min_y_gap=0.0)
            if tn is None:
                continue
            serving_label = tn if formation_side == "near" else (
                t0.team_label if tn == t1.team_label else t1.team_label
            )
            if rally.gt_serving_team == "A":
                votes[serving_label] += 1
            else:
                votes[serving_label] -= 1
        if votes:
            vid_label_a[vid] = max(votes, key=lambda k: votes[k])

    # Second pass: gather samples
    for vid, rallies in sorted(video_rallies.items()):
        templates = templates_by_vid.get(vid)
        if templates is None or vid not in vid_label_a:
            continue
        t0, t1 = templates
        label_a = vid_label_a[vid]
        label_b = t0.team_label if label_a == t1.team_label else t1.team_label

        for rally in rallies:
            if rally.gt_serving_team is None:
                continue
            t2p = t2p_by_rally.get(rally.rally_id)
            if not t2p:
                continue
            positions = _parse_positions(rally.positions)
            net_y = rally.court_split_y if rally.court_split_y else 0.5
            formation_side, _ = _find_serving_side_by_formation(
                positions, net_y=net_y, start_frame=0,
            )
            if formation_side is None:
                continue

            if formation_side == "near":
                expected_tn = label_a if rally.gt_serving_team == "A" else label_b
            else:
                expected_tn = label_b if rally.gt_serving_team == "A" else label_a

            samples.append((
                vid, rally.rally_id, positions, t2p, templates, expected_tn,
            ))

    console.print(f"Total gated samples: {len(samples)}\n")

    # Sweep min_y_gap
    table = Table(title="min_y_gap sweep (full-rally)")
    table.add_column("gap")
    table.add_column("Returns", justify="right")
    table.add_column("Return%", justify="right")
    table.add_column("Correct", justify="right")
    table.add_column("Accuracy", justify="right")

    for gap in [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15]:
        returns = 0
        correct = 0
        for _, _, positions, t2p, templates, expected in samples:
            tn = localize_team_near(positions, t2p, templates, min_y_gap=gap)
            if tn is not None:
                returns += 1
                if tn == expected:
                    correct += 1
        ret_pct = returns / len(samples) if samples else 0
        acc = correct / returns if returns else 0
        table.add_row(
            f"{gap:.2f}", str(returns), f"{ret_pct:.0%}",
            str(correct), f"{acc:.1%}",
        )
    console.print(table)

    # Majority vote per video: for each rally where team_near != majority, flip it
    # Test at gap=0.03
    console.print("\n[bold]Per-video majority vote (gap=0.03)[/bold]")

    # Group samples by video
    by_vid: dict[str, list] = {}
    for s in samples:
        by_vid.setdefault(s[0], []).append(s)

    # For each video, compute full-rally team_near for each rally
    # Then majority-vote: if team X is majority near, override outliers UNLESS
    # we detect a side switch (need to segment).
    # Simple version: assume no switch within a video for now.
    maj_returns = 0
    maj_correct = 0
    for _, rally_samples in by_vid.items():
        # Get raw team_near for each rally
        raw_tns = []
        expecteds = []
        for _, _, positions, t2p, templates, expected in rally_samples:
            tn = localize_team_near(positions, t2p, templates, min_y_gap=0.03)
            raw_tns.append(tn)
            expecteds.append(expected)

        # Majority vote (ignoring None)
        non_none = [tn for tn in raw_tns if tn is not None]
        if not non_none:
            continue
        majority = max(set(non_none), key=non_none.count)
        maj_count = non_none.count(majority)
        # Use majority if ≥70% of non-None agree AND ≥3 observations
        use_majority = len(non_none) >= 3 and maj_count / len(non_none) >= 0.7

        for tn, expected in zip(raw_tns, expecteds):
            # Fill None with majority if confident
            if tn is None:
                if use_majority:
                    tn = majority
                else:
                    continue
            maj_returns += 1
            if tn == expected:
                maj_correct += 1

    maj_ret_pct = maj_returns / len(samples) if samples else 0
    maj_acc = maj_correct / maj_returns if maj_returns else 0
    console.print(f"  Returns: {maj_returns} ({maj_ret_pct:.0%})")
    console.print(f"  Accuracy: {maj_correct}/{maj_returns} = {maj_acc:.1%}")

    # Also test: always use majority (force fill None and override outliers)
    console.print("\n[bold]Per-video majority override (flip outliers)[/bold]")

    mo_returns = 0
    mo_correct = 0
    for _, rally_samples in by_vid.items():
        raw_tns = []
        expecteds = []
        for _, _, positions, t2p, templates, expected in rally_samples:
            tn = localize_team_near(positions, t2p, templates, min_y_gap=0.03)
            raw_tns.append(tn)
            expecteds.append(expected)
        non_none = [tn for tn in raw_tns if tn is not None]
        if len(non_none) < 3:
            continue
        majority = max(set(non_none), key=non_none.count)
        maj_count = non_none.count(majority)
        if maj_count / len(non_none) < 0.7:
            continue

        for tn, expected in zip(raw_tns, expecteds):
            # Always override to majority
            final_tn = majority
            mo_returns += 1
            if final_tn == expected:
                mo_correct += 1

    mo_ret = mo_returns / len(samples) if samples else 0
    mo_acc = mo_correct / mo_returns if mo_returns else 0
    console.print(f"  Returns: {mo_returns} ({mo_ret:.0%})")
    console.print(f"  Accuracy: {mo_correct}/{mo_returns} = {mo_acc:.1%}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
