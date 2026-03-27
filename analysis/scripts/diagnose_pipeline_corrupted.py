"""Diagnose the 31 pipeline_corrupted attribution errors.

For each case where detect_contacts got the right player but the final
pipeline output changed it to the wrong one, shows:
- Team assignments and expected teams
- Whether the serve chain expectation is wrong
- Action sequence context

Usage:
    cd analysis
    uv run python scripts/diagnose_pipeline_corrupted.py
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
    Contact,
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
class CorruptedCase:
    rally_id: str
    gt_frame: int
    gt_action: str
    gt_track_id: int
    detect_track_id: int  # correct (from detect_contacts)
    final_track_id: int  # wrong (after pipeline)
    detect_team: int | None
    final_team: int | None
    gt_team: int | None
    court_side: str
    pred_action: str
    contact_index: int
    total_contacts: int
    candidates: list[tuple[int, float]]
    # Analysis
    serve_tid: int  # track_id of the server (contact 0)
    serve_team: int | None  # team of server


def analyze_rally(
    rally, team_assignments, calibrator, tolerance_frames,
) -> list[CorruptedCase]:
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
    contacts = contact_seq.contacts

    # Snapshot detect_contacts attribution
    detect_attr: dict[int, int] = {c.frame: c.player_track_id for c in contacts}
    contact_by_frame: dict[int, Contact] = {c.frame: c for c in contacts}

    # Run full pipeline
    rally_actions = classify_rally_actions(
        contact_seq, rally.rally_id,
        match_team_assignments=team_assignments,
    )
    pred_actions = [a.to_dict() for a in rally_actions.actions]
    real_pred = [a for a in pred_actions if not a.get("isSynthetic")]

    # Match GT
    avail_tids: set[int] | None = None
    if rally.positions_json:
        avail_tids = {pp["trackId"] for pp in rally.positions_json}

    matches, _ = match_contacts(
        rally.gt_labels, real_pred,
        tolerance=tolerance_frames,
        available_track_ids=avail_tids,
    )

    # Find server
    serve_tid = -1
    serve_team = None
    for a in real_pred:
        if a.get("action") == "serve":
            serve_tid = a.get("playerTrackId", -1)
            if team_assignments:
                serve_team = team_assignments.get(serve_tid)
            break

    cases: list[CorruptedCase] = []
    for m in matches:
        if m.pred_frame is None:
            continue

        gt_tid = next(
            (gt.player_track_id for gt in rally.gt_labels if gt.frame == m.gt_frame),
            -1,
        )

        det_tid = detect_attr.get(m.pred_frame, -1)
        final_tid = next(
            (a.get("playerTrackId", -1) for a in real_pred if a.get("frame") == m.pred_frame),
            -1,
        )

        if gt_tid < 0 or det_tid != gt_tid or final_tid == gt_tid:
            continue
        if avail_tids is not None and gt_tid not in avail_tids:
            continue

        contact = contact_by_frame.get(m.pred_frame)
        candidates = contact.player_candidates if contact else []
        court_side = next(
            (a.get("courtSide", "?") for a in real_pred if a.get("frame") == m.pred_frame),
            "?",
        )
        contact_idx = next(
            (i for i, c in enumerate(contacts) if c.frame == m.pred_frame), -1,
        )

        cases.append(CorruptedCase(
            rally_id=rally.rally_id,
            gt_frame=m.gt_frame,
            gt_action=m.gt_action,
            gt_track_id=gt_tid,
            detect_track_id=det_tid,
            final_track_id=final_tid,
            detect_team=team_assignments.get(det_tid) if team_assignments else None,
            final_team=team_assignments.get(final_tid) if team_assignments else None,
            gt_team=team_assignments.get(gt_tid) if team_assignments else None,
            court_side=court_side,
            pred_action=m.pred_action or "?",
            contact_index=contact_idx,
            total_contacts=len(contacts),
            candidates=[(tid, d) for tid, d in candidates],
            serve_tid=serve_tid,
            serve_team=serve_team,
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

    console.print(f"\n[bold]Pipeline Corrupted Diagnosis ({len(rallies)} rallies)[/bold]\n")

    all_cases: list[CorruptedCase] = []
    for i, rally in enumerate(rallies):
        fps = rally.fps or 30.0
        tol = max(1, round(fps * 167 / 1000))
        teams = match_teams_by_rally.get(rally.rally_id)
        cal = calibrators.get(rally.video_id)
        cases = analyze_rally(rally, teams, cal, tol)
        all_cases.extend(cases)
        if (i + 1) % 20 == 0 or i == len(rallies) - 1:
            print(f"  [{i + 1}/{len(rallies)}]")

    console.print(f"\n[bold]{len(all_cases)} pipeline_corrupted cases[/bold]\n")
    if not all_cases:
        return

    # === By GT action type ===
    console.print("[bold]By GT action type:[/bold]")
    for action, count in Counter(c.gt_action for c in all_cases).most_common():
        console.print(f"  {action:12s} {count}")

    # === Team analysis ===
    console.print("\n[bold]Team analysis:[/bold]")
    has_teams = [c for c in all_cases if c.gt_team is not None]
    no_teams = len(all_cases) - len(has_teams)
    console.print(f"  Has team assignments: {len(has_teams)}")
    console.print(f"  No team assignments:  {no_teams}")

    if has_teams:
        # Was GT player on a different team than the final choice?
        same_team_swap = sum(1 for c in has_teams if c.gt_team == c.final_team)
        diff_team_swap = sum(1 for c in has_teams if c.gt_team != c.final_team)
        console.print(f"\n  GT and final on SAME team:   {same_team_swap} (swapped within team)")
        console.print(f"  GT and final on DIFF team:   {diff_team_swap} (swapped across teams)")

        # For diff-team swaps: is the problem that the expected team was wrong?
        console.print("\n[bold]Cross-team swaps — why did reattribute pick the wrong team?[/bold]")
        for c in sorted(
            [c for c in has_teams if c.gt_team != c.final_team],
            key=lambda x: (x.rally_id, x.gt_frame),
        ):
            # The GT player was on team X, but reattribute moved to team Y
            console.print(
                f"  {c.rally_id[:8]} frm={c.gt_frame:4d} "
                f"gt={c.gt_action:8s} idx={c.contact_index}/{c.total_contacts} "
                f"gt_team={c.gt_team} final_team={c.final_team} "
                f"side={c.court_side} "
                f"serve_team={c.serve_team}"
            )

    # === Contact index distribution ===
    console.print("\n[bold]Contact index distribution (which position in rally):[/bold]")
    idx_counts = Counter(c.contact_index for c in all_cases)
    for idx in sorted(idx_counts):
        console.print(f"  contact #{idx}: {idx_counts[idx]}")

    # === Detailed case table ===
    console.print()
    detail_table = Table(title="All Pipeline Corrupted Cases")
    detail_table.add_column("Rally", style="dim", max_width=8)
    detail_table.add_column("Frame", justify="right")
    detail_table.add_column("GT Act", style="bold")
    detail_table.add_column("Pred", max_width=7)
    detail_table.add_column("Idx", justify="right")
    detail_table.add_column("GT→Final TID", justify="center")
    detail_table.add_column("GT Tm", justify="right")
    detail_table.add_column("Fn Tm", justify="right")
    detail_table.add_column("Side")
    detail_table.add_column("Sv Tm", justify="right")
    detail_table.add_column("Candidates", style="dim", max_width=35)

    for c in sorted(all_cases, key=lambda x: (x.rally_id, x.gt_frame)):
        cands_str = " ".join(
            f"{'>' if tid == c.gt_track_id else ''}{tid}({d:.3f})"
            for tid, d in c.candidates[:4]
        )
        detail_table.add_row(
            c.rally_id[:8],
            str(c.gt_frame),
            c.gt_action,
            c.pred_action,
            str(c.contact_index),
            f"{c.gt_track_id} -> {c.final_track_id}",
            str(c.gt_team) if c.gt_team is not None else "?",
            str(c.final_team) if c.final_team is not None else "?",
            c.court_side,
            str(c.serve_team) if c.serve_team is not None else "?",
            cands_str,
        )
    console.print(detail_table)

    # === Rally concentration ===
    console.print("\n[bold]Rally concentration:[/bold]")
    rally_counts = Counter(c.rally_id for c in all_cases)
    for rid, count in rally_counts.most_common(10):
        console.print(f"  {rid[:8]}: {count} corrupted")


if __name__ == "__main__":
    main()
