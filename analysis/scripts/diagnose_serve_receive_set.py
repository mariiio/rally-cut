"""Diagnose serve→receive and receive→set confusions.

These are both heuristic errors (not classifier):
- serve→receive: serve detection fails, real serve classified as receive
- receive→set: touch counting wrong, receive counted as 2nd touch (set)

Usage:
    cd analysis
    uv run python scripts/diagnose_serve_receive_set.py
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from rich.console import Console
from rich.table import Table

from rallycut.court.calibration import CourtCalibrator
from rallycut.evaluation.tracking.db import load_court_calibration
from rallycut.tracking.action_classifier import classify_rally_actions
from rallycut.tracking.ball_tracker import BallPosition as BallPos
from rallycut.tracking.contact_detector import (
    ContactDetectionConfig,
    detect_contacts,
)
from rallycut.tracking.player_tracker import PlayerPosition as PlayerPos
from scripts.eval_action_detection import (
    _load_match_team_assignments,
    load_rallies_with_action_gt,
    match_contacts,
)

console = Console()


@dataclass
class MisclassCase:
    rally_id: str
    gt_frame: int
    pred_frame: int
    gt_action: str
    pred_action: str
    contact_index: int
    total_contacts: int
    pred_court_side: str
    pred_contact_count_on_side: int
    is_synthetic_serve: bool  # was a synthetic serve inserted?
    ball_y: float
    net_y: float
    has_teams: bool


def analyze_rally(rally, team_assignments, calibrator, tolerance_frames):
    if not rally.ball_positions_json:
        return []

    ball_positions = [
        BallPos(
            frame_number=bp["frameNumber"],
            x=bp["x"], y=bp["y"],
            confidence=bp.get("confidence", 1.0),
        )
        for bp in rally.ball_positions_json
        if bp.get("x", 0) > 0 or bp.get("y", 0) > 0
    ]

    player_positions: list[PlayerPos] = []
    if rally.positions_json:
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

    contact_seq = detect_contacts(
        ball_positions=ball_positions,
        player_positions=player_positions,
        config=ContactDetectionConfig(),
        net_y=rally.court_split_y,
        frame_count=rally.frame_count or None,
        team_assignments=team_assignments,
        court_calibrator=calibrator,
    )

    rally_actions = classify_rally_actions(
        contact_seq, rally.rally_id,
        match_team_assignments=team_assignments,
    )
    pred_dicts = [a.to_dict() for a in rally_actions.actions]
    real_pred = [a for a in pred_dicts if not a.get("isSynthetic")]

    avail_tids: set[int] | None = None
    if rally.positions_json:
        avail_tids = {pp["trackId"] for pp in rally.positions_json}

    matches, _ = match_contacts(
        rally.gt_labels, real_pred,
        tolerance=tolerance_frames,
        available_track_ids=avail_tids,
    )

    # Check for synthetic serves
    has_synthetic = any(a.get("isSynthetic") for a in pred_dicts)

    # Build action lookup
    action_by_frame: dict[int, dict] = {a.get("frame", -1): a for a in pred_dicts}

    net_y = rally.court_split_y or 0.5

    cases: list[MisclassCase] = []
    for m in matches:
        if m.pred_frame is None or m.pred_action is None:
            continue
        if m.gt_action == m.pred_action:
            continue
        # Only serve→receive and receive→set
        pair = (m.gt_action, m.pred_action)
        if pair not in (("serve", "receive"), ("receive", "set")):
            continue

        pred_dict = action_by_frame.get(m.pred_frame, {})

        # Find contact index
        contact_idx = -1
        for i, c in enumerate(contact_seq.contacts):
            if c.frame == m.pred_frame:
                contact_idx = i
                break

        cases.append(MisclassCase(
            rally_id=rally.rally_id,
            gt_frame=m.gt_frame,
            pred_frame=m.pred_frame,
            gt_action=m.gt_action,
            pred_action=m.pred_action,
            contact_index=contact_idx,
            total_contacts=len(contact_seq.contacts),
            pred_court_side=pred_dict.get("courtSide", "?"),
            pred_contact_count_on_side=pred_dict.get("contactCountOnSide", -1),
            is_synthetic_serve=has_synthetic,
            ball_y=pred_dict.get("ballY", -1),
            net_y=net_y,
            has_teams=team_assignments is not None,
        ))

    return cases


def main() -> None:
    rallies = load_rallies_with_action_gt()
    if not rallies:
        console.print("[red]No rallies found.[/red]")
        return

    calibrators: dict[str, CourtCalibrator | None] = {}
    video_ids = {r.video_id for r in rallies}
    for vid in video_ids:
        corners = load_court_calibration(vid)
        if corners and len(corners) == 4:
            cal = CourtCalibrator()
            cal.calibrate([(c["x"], c["y"]) for c in corners])
            calibrators[vid] = cal
        else:
            calibrators[vid] = None

    match_teams_by_rally = _load_match_team_assignments(video_ids, min_confidence=0.70)

    console.print(f"\n[bold]Serve/Receive/Set Confusion Diagnosis ({len(rallies)} rallies)[/bold]\n")

    serve_to_receive: list[MisclassCase] = []
    receive_to_set: list[MisclassCase] = []

    for i, rally in enumerate(rallies):
        fps = rally.fps or 30.0
        tol = max(1, round(fps * 167 / 1000))
        teams = match_teams_by_rally.get(rally.rally_id)
        cal = calibrators.get(rally.video_id)
        cases = analyze_rally(rally, teams, cal, tol)
        for c in cases:
            if c.gt_action == "serve":
                serve_to_receive.append(c)
            else:
                receive_to_set.append(c)
        if (i + 1) % 20 == 0 or i == len(rallies) - 1:
            print(f"  [{i + 1}/{len(rallies)}]")

    # ==========================================
    # SERVE → RECEIVE
    # ==========================================
    console.print(f"\n{'=' * 60}")
    console.print(f"[bold]SERVE → RECEIVE ({len(serve_to_receive)} cases)[/bold]")
    console.print(f"{'=' * 60}")

    if serve_to_receive:
        # Contact index distribution
        console.print("\n[bold]Contact index of misclassified serve:[/bold]")
        idx_counts = Counter(c.contact_index for c in serve_to_receive)
        for idx, count in sorted(idx_counts.items()):
            console.print(f"  contact #{idx}: {count}")

        # Court side
        console.print("\n[bold]Predicted court_side:[/bold]")
        for side, count in Counter(c.pred_court_side for c in serve_to_receive).most_common():
            console.print(f"  {side}: {count}")

        # Ball Y relative to net
        console.print("\n[bold]Ball position relative to net:[/bold]")
        near_count = sum(1 for c in serve_to_receive if c.ball_y > c.net_y)
        far_count = sum(1 for c in serve_to_receive if c.ball_y <= c.net_y)
        console.print(f"  Ball on near side (y > net_y): {near_count}")
        console.print(f"  Ball on far side (y <= net_y): {far_count}")

        # Has synthetic serve?
        synth_count = sum(1 for c in serve_to_receive if c.is_synthetic_serve)
        console.print(f"\n[bold]Synthetic serve in same rally:[/bold] {synth_count}/{len(serve_to_receive)}")

        # Has team assignments?
        teams_count = sum(1 for c in serve_to_receive if c.has_teams)
        console.print(f"[bold]Has team assignments:[/bold] {teams_count}/{len(serve_to_receive)}")

        # touch count
        console.print("\n[bold]contact_count_on_side for misclassified serves:[/bold]")
        for cnt, n in Counter(c.pred_contact_count_on_side for c in serve_to_receive).most_common():
            console.print(f"  count={cnt}: {n}")

        # Detail table
        console.print()
        t = Table(title="Serve → Receive Details")
        t.add_column("Rally", style="dim", max_width=8)
        t.add_column("GT Frm", justify="right")
        t.add_column("Pred Frm", justify="right")
        t.add_column("Idx", justify="right")
        t.add_column("Tot", justify="right")
        t.add_column("Side")
        t.add_column("SideCt", justify="right")
        t.add_column("BallY", justify="right")
        t.add_column("NetY", justify="right")
        t.add_column("Synth")
        t.add_column("Teams")

        for c in sorted(serve_to_receive, key=lambda x: (x.rally_id, x.gt_frame)):
            t.add_row(
                c.rally_id[:8],
                str(c.gt_frame), str(c.pred_frame),
                str(c.contact_index), str(c.total_contacts),
                c.pred_court_side,
                str(c.pred_contact_count_on_side),
                f"{c.ball_y:.3f}", f"{c.net_y:.3f}",
                "Y" if c.is_synthetic_serve else "N",
                "Y" if c.has_teams else "N",
            )
        console.print(t)

    # ==========================================
    # RECEIVE → SET
    # ==========================================
    console.print(f"\n{'=' * 60}")
    console.print(f"[bold]RECEIVE → SET ({len(receive_to_set)} cases)[/bold]")
    console.print(f"{'=' * 60}")

    if receive_to_set:
        # Contact index
        console.print("\n[bold]Contact index of misclassified receive:[/bold]")
        idx_counts = Counter(c.contact_index for c in receive_to_set)
        for idx, count in sorted(idx_counts.items()):
            console.print(f"  contact #{idx}: {count}")

        # Touch count — this is the key: should be 1, is it 2?
        console.print("\n[bold]contact_count_on_side (should be 1 for receive):[/bold]")
        for cnt, n in Counter(c.pred_contact_count_on_side for c in receive_to_set).most_common():
            console.print(f"  count={cnt}: {n}")

        # Court side
        console.print("\n[bold]Predicted court_side:[/bold]")
        for side, count in Counter(c.pred_court_side for c in receive_to_set).most_common():
            console.print(f"  {side}: {count}")

        # Has synthetic serve?
        synth_count = sum(1 for c in receive_to_set if c.is_synthetic_serve)
        console.print(f"\n[bold]Synthetic serve in same rally:[/bold] {synth_count}/{len(receive_to_set)}")

        teams_count = sum(1 for c in receive_to_set if c.has_teams)
        console.print(f"[bold]Has team assignments:[/bold] {teams_count}/{len(receive_to_set)}")

        # Detail table
        console.print()
        t = Table(title="Receive → Set Details")
        t.add_column("Rally", style="dim", max_width=8)
        t.add_column("GT Frm", justify="right")
        t.add_column("Pred Frm", justify="right")
        t.add_column("Idx", justify="right")
        t.add_column("Tot", justify="right")
        t.add_column("Side")
        t.add_column("SideCt", justify="right")
        t.add_column("BallY", justify="right")
        t.add_column("NetY", justify="right")
        t.add_column("Synth")
        t.add_column("Teams")

        for c in sorted(receive_to_set, key=lambda x: (x.rally_id, x.gt_frame)):
            t.add_row(
                c.rally_id[:8],
                str(c.gt_frame), str(c.pred_frame),
                str(c.contact_index), str(c.total_contacts),
                c.pred_court_side,
                str(c.pred_contact_count_on_side),
                f"{c.ball_y:.3f}", f"{c.net_y:.3f}",
                "Y" if c.is_synthetic_serve else "N",
                "Y" if c.has_teams else "N",
            )
        console.print(t)


if __name__ == "__main__":
    main()
