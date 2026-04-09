"""Deep-dive diagnostic for 5 rallies where tracking looks perfect but
player_attribution is 0%. Dumps GT vs pred vs trackToPlayer sidecar so we
can see which layer is dropping the ball.

Read-only. No DB writes.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from rich.console import Console  # noqa: E402

from eval_action_detection import (  # noqa: E402
    load_rallies_with_action_gt,
    match_contacts,
)
from production_eval import (  # noqa: E402
    PipelineContext,
    _build_calibrators,
    _load_match_team_assignments,
    _load_track_to_player_maps,
    _parse_positions,
    _run_rally,
    _tolerance_frames,
)

console = Console()

TARGET_RALLIES = {
    # worst
    "753a4ec7": "moma.mp4",
    "169292de": "wowo.mp4",
    "fad29c31": "mech.mp4",
    "2d3cb54b": "yeye.mp4",
    "cc50cf7e": "pupu.mp4",
    # best control
    "173a4a61": "lolo.mp4",
    "721bb968": "yeye.mp4 (good)",
}


def main() -> int:
    rallies = load_rallies_with_action_gt()
    targets = [r for r in rallies if r.rally_id[:8] in TARGET_RALLIES]
    console.print(f"matched {len(targets)}/{len(TARGET_RALLIES)} target rallies")

    video_ids = {r.video_id for r in targets}
    rally_pos_lookup = {r.rally_id: _parse_positions(r.positions_json) for r in targets}
    team_map = _load_match_team_assignments(video_ids, rally_positions=rally_pos_lookup)
    t2p_by_rally = _load_track_to_player_maps(video_ids)
    calibrators = _build_calibrators(video_ids)
    ctx = PipelineContext()

    for rally in targets:
        tag = TARGET_RALLIES[rally.rally_id[:8]]
        console.print(f"\n[bold cyan]━━━ {tag} · {rally.rally_id[:8]} ━━━[/bold cyan]")

        raw_tids_in_pos = sorted({pp["trackId"] for pp in rally.positions_json})
        console.print(f"trackIds in positions_json: {raw_tids_in_pos}")

        t2p = t2p_by_rally.get(rally.rally_id)
        if t2p:
            console.print(f"trackToPlayer sidecar: {dict(sorted(t2p.items()))}")
        else:
            console.print("[red]NO trackToPlayer sidecar for this rally[/red]")

        teams = team_map.get(rally.rally_id)
        console.print(f"team_assignments: {dict(sorted(teams.items())) if teams else None}")

        gt_tids = sorted({gl.player_track_id for gl in rally.gt_labels if gl.player_track_id >= 0})
        console.print(f"GT contact player_track_ids (unique): {gt_tids}")
        console.print(f"GT contact count: {len(rally.gt_labels)}")

        # Run the production pipeline
        pred_actions, _ = _run_rally(
            rally,
            team_map.get(rally.rally_id),
            calibrators.get(rally.video_id),
            ctx,
        )
        real_pred = [a for a in pred_actions if not a.get("isSynthetic")]
        pred_tids = sorted({a.get("playerTrackId") for a in real_pred if a.get("playerTrackId") is not None})
        console.print(f"Pred contact playerTrackIds (unique): {pred_tids}")
        console.print(f"Pred contact count: {len(real_pred)}")

        # Normalize + match
        raw_avail = {pp["trackId"] for pp in rally.positions_json}
        rally_t2p = t2p
        if rally_t2p:
            avail = {rally_t2p.get(tid, tid) for tid in raw_avail}
        else:
            avail = raw_avail

        matches, _ = match_contacts(
            rally.gt_labels,
            real_pred,
            tolerance=_tolerance_frames(rally.fps),
            available_track_ids=avail,
            team_assignments=teams,
            track_id_map=rally_t2p,
        )

        # Build per-frame player lookup from positions_json for spatial test
        by_frame: dict[int, list[tuple[int, float, float]]] = {}
        for pp in rally.positions_json:
            by_frame.setdefault(pp["frameNumber"], []).append(
                (pp["trackId"], pp["x"], pp["y"])
            )

        def closest_track(frame: int, bx: float, by: float) -> int | None:
            # search nearest frame with positions (tolerance ±5)
            best_frame = None
            for df in range(0, 6):
                for f in (frame - df, frame + df):
                    if f in by_frame:
                        best_frame = f
                        break
                if best_frame is not None:
                    break
            if best_frame is None:
                return None
            rows = by_frame[best_frame]
            return min(rows, key=lambda r: (r[1] - bx) ** 2 + (r[2] - by) ** 2)[0]

        n_gt_with_ball = sum(1 for gl in rally.gt_labels if gl.ball_x is not None)
        console.print(f"GT labels with ball coords: {n_gt_with_ball}/{len(rally.gt_labels)}")

        # Per-contact dump
        console.print("per-contact match detail:")
        spatial_correct = 0
        spatial_evaluable = 0
        for i, m in enumerate(matches, 1):
            # Find corresponding gt label and pred action
            gt_tid = None
            for gl in rally.gt_labels:
                if gl.frame == m.gt_frame and gl.action == m.gt_action:
                    gt_tid = gl.player_track_id
                    break
            pred_tid = None
            if m.pred_frame is not None:
                for a in real_pred:
                    if a.get("frame") == m.pred_frame and a.get("action") == m.pred_action:
                        pred_tid = a.get("playerTrackId")
                        break
            pred_tid_norm = rally_t2p.get(pred_tid, pred_tid) if rally_t2p and pred_tid else pred_tid

            # Spatial attribution check — find gt ball coords and closest pred track
            gt_ball = None
            for gl in rally.gt_labels:
                if gl.frame == m.gt_frame and gl.action == m.gt_action:
                    if gl.ball_x is not None and gl.ball_y is not None:
                        gt_ball = (gl.ball_x, gl.ball_y)
                    break

            sp_closest: int | None = None
            if gt_ball is not None:
                sp_closest = closest_track(m.gt_frame, *gt_ball)
            sp_closest_norm: int | None = None
            if sp_closest is not None and rally_t2p:
                sp_closest_norm = rally_t2p.get(sp_closest, sp_closest)
            elif sp_closest is not None:
                sp_closest_norm = sp_closest

            # Spatial-correct: pred picked the player who was physically closest to ball
            if pred_tid is not None and sp_closest is not None:
                spatial_evaluable += 1
                if pred_tid == sp_closest:
                    spatial_correct += 1

            mark = "[green]✓[/green]" if m.player_correct else ("[red]✗[/red]" if m.player_evaluable else "[dim]n/a[/dim]")
            sp_mark = ""
            if sp_closest is not None:
                sp_mark = f" ball→track{sp_closest}"
                if sp_closest_norm != sp_closest:
                    sp_mark += f"→p{sp_closest_norm}"
            console.print(
                f"  {mark} #{i} f={m.gt_frame} gt={m.gt_action}(tid={gt_tid}) "
                f"pred={m.pred_action}(tid={pred_tid}→{pred_tid_norm}){sp_mark} "
                f"correct={m.player_correct}"
            )

        if spatial_evaluable:
            console.print(
                f"  [magenta]spatial self-consistency:[/magenta] "
                f"{spatial_correct}/{spatial_evaluable} = {100*spatial_correct/spatial_evaluable:.0f}% "
                f"(does pred's chosen player match the pred track closest to GT ball?)"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
