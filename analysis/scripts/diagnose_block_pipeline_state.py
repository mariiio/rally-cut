"""Per-missed-block pipeline-state snapshot for block detection diagnosis.

For every GT block on the production_eval pool (355 rallies, `--skip-session
6f599a0e-...`) that is not matched as a block in the pipeline output, record
the full pipeline state at the GT frame: nearest Contact, dedup survival,
attribution, is_at_net, previous contact's action/side/frame gap.

Buckets missed blocks into a failure class so the next step can pick the
smallest block-gated lever. See `.claude/plans/logical-mixing-avalanche.md`
for the decision gate.

Usage:
    cd analysis
    uv run python scripts/diagnose_block_pipeline_state.py
    uv run python scripts/diagnose_block_pipeline_state.py --rally <id>    # Debug one
    uv run python scripts/diagnose_block_pipeline_state.py --no-skip-session  # Include all
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from eval_action_detection import (  # noqa: E402
    RallyData,
    _build_player_positions,
    _load_match_team_assignments,
    _load_track_to_player_maps,
    load_rallies_with_action_gt,
)
from production_eval import (  # noqa: E402
    PipelineContext,
    _build_calibrators,
    _build_camera_heights,
    _load_formation_semantic_flips_from_gt,
    _load_team_templates_by_video,
    _parse_ball,
)

from rallycut.evaluation.tracking.db import get_connection  # noqa: E402
from rallycut.tracking import contact_detector as _cd_mod  # noqa: E402
from rallycut.tracking.action_classifier import classify_rally_actions  # noqa: E402
from rallycut.tracking.contact_detector import (  # noqa: E402
    Contact,
    ContactSequence,
    detect_contacts,
)
from rallycut.tracking.match_tracker import verify_team_assignments  # noqa: E402
from rallycut.tracking.player_tracker import PlayerPosition  # noqa: E402
from rallycut.tracking.sequence_action_runtime import (  # noqa: E402
    apply_sequence_override,
    get_sequence_probs,
)

console = Console()

POOR_SESSION_ID = "6f599a0e-b8ea-4bf0-a331-ce7d9ef88164"

# --------------------------------------------------------------------------- #
# Dedup spy                                                                   #
# --------------------------------------------------------------------------- #

# _deduplicate_contacts is called multiple times per rally (inside
# detect_contacts). We capture (input, output, min_distance, adaptive) per
# call. The FINAL call is the production-output dedup pass.
_DEDUP_CALLS: list[tuple[list[Contact], list[Contact], int, bool]] = []
_ORIG_DEDUP = _cd_mod._deduplicate_contacts


def _spy_dedup(
    contacts: list[Contact],
    min_distance: int,
    adaptive: bool = False,
) -> list[Contact]:
    inp = list(contacts)
    out = _ORIG_DEDUP(contacts, min_distance, adaptive=adaptive)
    _DEDUP_CALLS.append((inp, list(out), min_distance, adaptive))
    return out


def install_spy() -> None:
    _cd_mod._deduplicate_contacts = _spy_dedup


def reset_spy() -> None:
    _DEDUP_CALLS.clear()


def last_pre_dedup() -> list[Contact]:
    """Final pre-dedup contact list (the pool hitting the output dedup pass)."""
    return list(_DEDUP_CALLS[-1][0]) if _DEDUP_CALLS else []


def last_dedup_call() -> tuple[list[Contact], list[Contact], int, bool] | None:
    """Final (input, output, min_distance, adaptive) tuple, or None."""
    return _DEDUP_CALLS[-1] if _DEDUP_CALLS else None


# --------------------------------------------------------------------------- #
# Dedup-decision replay                                                       #
# --------------------------------------------------------------------------- #


_CROSS_SIDE_MIN_DISTANCE = 4  # mirrors contact_detector._CROSS_SIDE_MIN_DISTANCE


def _effective_min(
    a: Contact, b: Contact, base_min: int, adaptive: bool,
) -> int:
    if not adaptive:
        return base_min
    sides_known = (
        a.court_side in ("near", "far") and b.court_side in ("near", "far")
    )
    if sides_known and a.court_side != b.court_side:
        return _CROSS_SIDE_MIN_DISTANCE
    return base_min


@dataclass
class KillRecord:
    """Why a pre-dedup candidate didn't survive the dedup pass."""

    killed_frame: int
    killed_court_side: str
    killed_conf: float
    killed_is_at_net: bool
    killed_player_tid: int
    blocker_frame: int
    blocker_court_side: str
    blocker_conf: float
    blocker_is_at_net: bool
    blocker_player_tid: int
    effective_min: int
    gap: int
    cross_side: bool
    side_known_both: bool
    cause: str  # 'gap_lt_4_cross_side', 'gap_lt_12_unknown_side', 'gap_lt_12_same_side'


def replay_kills(
    inp: list[Contact],
    out: list[Contact],
    min_distance: int,
    adaptive: bool,
    near_frame: int,
    window: int = 10,
) -> list[KillRecord]:
    """Re-run the dedup decision, recording why each killed candidate died.

    Mirrors `_deduplicate_contacts` exactly: sort by confidence desc, accept
    when no already-accepted contact is within effective_min frames, otherwise
    record the (blocker, cause) tuple.
    """
    out_frames = {c.frame for c in out}
    sorted_cs = sorted(inp, key=lambda c: c.confidence, reverse=True)
    accepted: list[Contact] = []
    kills: list[KillRecord] = []
    for c in sorted_cs:
        was_killed = c.frame not in out_frames
        if not was_killed:
            accepted.append(c)
            continue
        # Find the blocker — the first already-accepted contact within effective_min.
        blocker: Contact | None = None
        block_eff: int = -1
        for existing in accepted:
            eff = _effective_min(c, existing, min_distance, adaptive)
            gap = abs(c.frame - existing.frame)
            if gap < eff:
                blocker = existing
                block_eff = eff
                break
        if blocker is None:
            # Killed but no blocker found — duplicate frame? (rare). Skip.
            continue

        if abs(c.frame - near_frame) > window:
            continue

        sides_known = (
            c.court_side in ("near", "far")
            and blocker.court_side in ("near", "far")
        )
        cross_side = sides_known and c.court_side != blocker.court_side

        if cross_side and block_eff == _CROSS_SIDE_MIN_DISTANCE:
            cause = "gap_lt_4_cross_side"
        elif not sides_known and block_eff == min_distance:
            cause = "gap_lt_12_unknown_side"
        elif sides_known and not cross_side:
            cause = "gap_lt_12_same_side"
        else:
            cause = f"gap_lt_{block_eff}_other"

        kills.append(KillRecord(
            killed_frame=c.frame,
            killed_court_side=c.court_side,
            killed_conf=c.confidence,
            killed_is_at_net=c.is_at_net,
            killed_player_tid=c.player_track_id,
            blocker_frame=blocker.frame,
            blocker_court_side=blocker.court_side,
            blocker_conf=blocker.confidence,
            blocker_is_at_net=blocker.is_at_net,
            blocker_player_tid=blocker.player_track_id,
            effective_min=block_eff,
            gap=abs(c.frame - blocker.frame),
            cross_side=cross_side,
            side_known_both=sides_known,
            cause=cause,
        ))

    return kills


# --------------------------------------------------------------------------- #
# Pool loading                                                                #
# --------------------------------------------------------------------------- #


def load_pool(
    skip_session_id: str | None = POOR_SESSION_ID,
    rally_id: str | None = None,
) -> list[RallyData]:
    """Load production_eval pool, optionally filtering one session."""
    rallies = load_rallies_with_action_gt(rally_id=rally_id)
    if not skip_session_id:
        return rallies
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT video_id FROM session_videos WHERE session_id = %s",
            [skip_session_id],
        )
        excluded_videos = {row[0] for row in cur.fetchall()}
    return [r for r in rallies if r.video_id not in excluded_videos]


# --------------------------------------------------------------------------- #
# Per-rally production-mirrored pipeline — returns ContactSequence too        #
# --------------------------------------------------------------------------- #


def run_rally_with_contacts(
    rally: RallyData,
    match_teams: dict[int, int] | None,
    calibrator: Any,
    ctx: PipelineContext,
    track_to_player: dict[int, int] | None = None,
    formation_semantic_flip: bool = False,
    camera_height: float = 0.0,
) -> tuple[list[dict], Any, ContactSequence]:
    """Mirror of `production_eval._run_rally`, but also returns ContactSequence."""
    ball_positions = _parse_ball(rally.ball_positions_json or [])
    player_positions: list[PlayerPosition] = _build_player_positions(
        rally.positions_json or [], rally_id=rally.rally_id, inject_pose=True,
    )

    teams: dict[int, int] | None = dict(match_teams) if match_teams else None
    if teams is not None and not ctx.skip_verify_teams:
        teams = verify_team_assignments(teams, player_positions)

    sequence_probs: np.ndarray | None = get_sequence_probs(
        ball_positions=ball_positions,
        player_positions=player_positions,
        court_split_y=rally.court_split_y,
        frame_count=rally.frame_count,
        team_assignments=teams,
        calibrator=calibrator,
    )

    contact_sequence = detect_contacts(
        ball_positions=ball_positions,
        player_positions=player_positions,
        config=None,
        use_classifier=True,
        frame_count=rally.frame_count or None,
        team_assignments=teams,
        court_calibrator=calibrator,
        sequence_probs=sequence_probs,
    )

    rally_actions = classify_rally_actions(
        contact_sequence,
        rally.rally_id,
        team_assignments=teams,
        match_team_assignments=teams,
        calibrator=calibrator,
        track_to_player=track_to_player,
        formation_semantic_flip=formation_semantic_flip,
        camera_height=camera_height,
    )

    if sequence_probs is not None:
        apply_sequence_override(rally_actions, sequence_probs)

    return (
        [a.to_dict() for a in rally_actions.actions],
        rally_actions,
        contact_sequence,
    )


# --------------------------------------------------------------------------- #
# Missed-block snapshot                                                       #
# --------------------------------------------------------------------------- #


@dataclass
class BlockSnapshot:
    """Per missed GT block: pipeline state + derived bucket."""

    rally_id: str
    video_id: str
    gt_frame: int
    gt_player_tid: int
    fps: float

    # A — pipeline state at GT
    nearest_contact_offset: int | None = None  # signed frame delta (contact - gt)
    nearest_contact_frame: int | None = None
    nearest_contact_is_at_net: bool | None = None
    nearest_contact_court_side: str | None = None
    nearest_contact_player_tid: int | None = None
    pred_action_at_nearest: str | None = None  # 'dig', 'set', 'attack', 'dropped', None
    opposite_side_in_candidates: bool | None = None  # A4
    pre_dedup_candidate_exists: bool = False  # A6

    # B — block-rule preconditions at nearest Contact
    prev_action_type: str | None = None
    prev_court_side: str | None = None
    frame_gap_to_prev: int | None = None
    cross_side_vs_prev: bool | None = None
    is_at_net_of_contact: bool | None = None

    # Derived bucket
    bucket: str = "other"

    # Pre-dedup candidates near GT (for dedup-decision tracing)
    pre_dedup_near: list[dict] = field(default_factory=list)
    kill_records: list[dict] = field(default_factory=list)


def _find_nearest(contacts: list[Contact], frame: int) -> tuple[Contact | None, int | None]:
    best_c = None
    best_off: int | None = None
    for c in contacts:
        off = c.frame - frame
        if best_off is None or abs(off) < abs(best_off):
            best_c = c
            best_off = off
    return best_c, best_off


def _action_at_frame(actions_json: list[dict], frame: int) -> str | None:
    """Return emitted action dict's 'action' at exactly this frame, else None."""
    for a in actions_json:
        if a.get("frame") == frame and not a.get("isSynthetic"):
            return a.get("action")
    return None


def _prev_real_action(actions_json: list[dict], frame: int) -> dict | None:
    """Most recent non-synthetic action with frame < given frame."""
    before = [
        a for a in actions_json
        if a.get("frame", 10**9) < frame and not a.get("isSynthetic")
    ]
    if not before:
        return None
    return max(before, key=lambda a: a.get("frame", -1))


def _build_snapshot(
    rally: RallyData,
    gt_frame: int,
    gt_player_tid: int,
    contacts: list[Contact],
    pre_dedup_contacts: list[Contact],
    actions_json: list[dict],
    teams: dict[int, int] | None,
    tolerance_frames: int,
    kill_records: list[KillRecord] | None = None,
) -> BlockSnapshot:
    snap = BlockSnapshot(
        rally_id=rally.rally_id,
        video_id=rally.video_id,
        gt_frame=gt_frame,
        gt_player_tid=gt_player_tid,
        fps=rally.fps,
    )

    near_c, near_off = _find_nearest(contacts, gt_frame)
    snap.nearest_contact_offset = near_off
    if near_c is not None:
        snap.nearest_contact_frame = near_c.frame
        snap.nearest_contact_is_at_net = near_c.is_at_net
        snap.nearest_contact_court_side = near_c.court_side
        snap.nearest_contact_player_tid = near_c.player_track_id

        # A5 — what action did the pipeline emit at this Contact's frame?
        act = _action_at_frame(actions_json, near_c.frame)
        snap.pred_action_at_nearest = act if act is not None else "dropped"

        # A4 — is there an opposite-team player in the candidate list?
        if teams and near_c.player_candidates:
            own_team = teams.get(near_c.player_track_id)
            opp = False
            for cand_tid, _dist in near_c.player_candidates[:4]:
                cand_team = teams.get(cand_tid)
                if (
                    cand_team is not None
                    and own_team is not None
                    and cand_team != own_team
                ):
                    opp = True
                    break
            snap.opposite_side_in_candidates = opp
        else:
            snap.opposite_side_in_candidates = None

    # A6 — any pre-dedup candidate within ±5 of GT that's NOT in the
    # post-dedup contact list? (Proxy for "dedup killed it".)
    window = 5
    post_frames = {c.frame for c in contacts}
    pre_nearby = [
        pc for pc in pre_dedup_contacts
        if abs(pc.frame - gt_frame) <= window and pc.frame not in post_frames
    ]
    snap.pre_dedup_candidate_exists = len(pre_nearby) > 0

    # B — precondition state
    if near_c is not None:
        snap.is_at_net_of_contact = near_c.is_at_net
        # Find previous Contact in the ContactSequence
        prev_c: Contact | None = None
        for c in contacts:
            if c.frame < near_c.frame:
                if prev_c is None or c.frame > prev_c.frame:
                    prev_c = c
        if prev_c is not None:
            snap.frame_gap_to_prev = near_c.frame - prev_c.frame
            snap.prev_court_side = prev_c.court_side
            snap.cross_side_vs_prev = (
                near_c.court_side in ("near", "far")
                and prev_c.court_side in ("near", "far")
                and near_c.court_side != prev_c.court_side
            )
        prev_act = _prev_real_action(actions_json, near_c.frame)
        if prev_act is not None:
            snap.prev_action_type = prev_act.get("action")

    # Pre-dedup candidates within ±10f of GT — context for the kill records.
    snap.pre_dedup_near = [
        {
            "frame": pc.frame,
            "court_side": pc.court_side,
            "player_tid": pc.player_track_id,
            "confidence": pc.confidence,
            "is_at_net": pc.is_at_net,
            "in_post": pc.frame in {c.frame for c in contacts},
        }
        for pc in pre_dedup_contacts
        if abs(pc.frame - gt_frame) <= 10
    ]
    if kill_records:
        snap.kill_records = [
            {
                "killed_frame": k.killed_frame,
                "killed_court_side": k.killed_court_side,
                "killed_conf": k.killed_conf,
                "killed_is_at_net": k.killed_is_at_net,
                "killed_player_tid": k.killed_player_tid,
                "blocker_frame": k.blocker_frame,
                "blocker_court_side": k.blocker_court_side,
                "blocker_conf": k.blocker_conf,
                "blocker_is_at_net": k.blocker_is_at_net,
                "blocker_player_tid": k.blocker_player_tid,
                "effective_min": k.effective_min,
                "gap": k.gap,
                "cross_side": k.cross_side,
                "side_known_both": k.side_known_both,
                "cause": k.cause,
            }
            for k in kill_records
        ]

    # Bucket the failure.
    snap.bucket = _bucket(snap, tolerance_frames)
    return snap


def _bucket(snap: BlockSnapshot, tolerance_frames: int) -> str:
    """Derive the failure bucket per the plan's taxonomy."""
    # No Contact anywhere near GT
    if (
        snap.nearest_contact_offset is None
        or abs(snap.nearest_contact_offset) > max(5, tolerance_frames)
    ):
        # Distinguish dedup-killed from no_candidate
        if snap.pre_dedup_candidate_exists:
            return "dedup_killed"
        return "no_candidate"

    # Contact exists — which precondition failed?
    # First: was the Contact emitted as a non-BLOCK action, or dropped?
    if snap.pred_action_at_nearest == "block":
        # Tolerance mismatch — pipeline says block but our match used ±tolerance.
        # Shouldn't happen since caller filters to missed blocks.
        return "precondition_met_but_not_emitted"

    # Check B-preconditions in the order they'd cause the heuristic to skip.
    # Rule: is_at_net AND last_action==ATTACK AND gap<=8 AND cross-side.
    if snap.prev_action_type is None:
        return "no_prev_action"
    if snap.prev_action_type != "attack":
        return "wrong_prev_action"
    if snap.is_at_net_of_contact is False:
        return "not_at_net"
    if snap.frame_gap_to_prev is not None and snap.frame_gap_to_prev > 8:
        return "frame_gap_exceeded"
    if snap.cross_side_vs_prev is False:
        # Same-side attribution — is it attacker-same-side (fixable) or genuine ceiling?
        if snap.opposite_side_in_candidates is True:
            return "wrong_attribution_same_side"
        return "wrong_attribution_no_opposite"

    # All preconditions nominally pass but BLOCK not emitted — unexpected.
    return "precondition_met_but_not_emitted"


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-session", default=POOR_SESSION_ID,
        help="Session ID whose videos to exclude (default: poor-tracking session)",
    )
    parser.add_argument("--no-skip-session", action="store_true",
                        help="Include all rallies (override --skip-session)")
    parser.add_argument("--rally", default=None, help="Debug one rally by ID")
    parser.add_argument(
        "--output", default="analysis/outputs/block_diagnostic_2026_04_15.json",
        help="Path to write JSON report",
    )
    args = parser.parse_args()

    skip_id = None if args.no_skip_session else args.skip_session

    install_spy()

    rallies = load_pool(skip_session_id=skip_id, rally_id=args.rally)
    if not rallies:
        console.print("[red]No rallies loaded — check DB connection and args[/red]")
        return
    # Only consider rallies that HAVE a GT block — most rallies don't.
    rallies_with_block = [
        r for r in rallies
        if any(g.action == "block" for g in r.gt_labels)
    ]

    total_gt_blocks = sum(
        1 for r in rallies_with_block for g in r.gt_labels if g.action == "block"
    )
    console.print(
        f"[bold]Pool:[/bold] {len(rallies)} rallies in the production_eval "
        f"pool; {len(rallies_with_block)} have a GT block ({total_gt_blocks} "
        f"GT blocks total)."
    )

    # Production deps.
    video_ids = {r.video_id for r in rallies_with_block}
    calibrators = _build_calibrators(video_ids)
    camera_heights = _build_camera_heights(video_ids, calibrators)

    # match_teams needs rally positions for verify_team_assignments.
    rally_pos_lookup: dict[str, list[PlayerPosition]] = {}
    for r in rallies_with_block:
        if r.positions_json:
            rally_pos_lookup[r.rally_id] = [
                PlayerPosition(
                    frame_number=pp["frameNumber"], track_id=pp["trackId"],
                    x=pp["x"], y=pp["y"],
                    width=pp["width"], height=pp["height"],
                    confidence=pp.get("confidence", 1.0),
                )
                for pp in r.positions_json
            ]

    match_teams_by_rally = _load_match_team_assignments(
        video_ids, min_confidence=0.70, rally_positions=rally_pos_lookup,
    )
    t2p_by_rally = _load_track_to_player_maps(video_ids)
    formation_flip_by_rally = _load_formation_semantic_flips_from_gt(video_ids)
    _team_templates_by_video = _load_team_templates_by_video(video_ids)  # noqa: F841

    ctx = PipelineContext()

    all_snaps: list[BlockSnapshot] = []
    n_block_tp = 0
    n_block_fn = 0

    for idx, rally in enumerate(rallies_with_block, start=1):
        if not rally.ball_positions_json or not rally.positions_json:
            console.print(
                f"  [{idx}/{len(rallies_with_block)}] {rally.rally_id[:8]} "
                f"[yellow]SKIP — missing tracking data[/yellow]"
            )
            continue
        if not rally.frame_count or rally.frame_count < 10:
            continue

        reset_spy()
        try:
            pred_actions, _rally_actions, contact_seq = run_rally_with_contacts(
                rally,
                match_teams_by_rally.get(rally.rally_id),
                calibrators.get(rally.video_id),
                ctx,
                track_to_player=t2p_by_rally.get(rally.rally_id),
                formation_semantic_flip=formation_flip_by_rally.get(
                    rally.rally_id, False,
                ),
                camera_height=camera_heights.get(rally.video_id, 0.0),
            )
        except Exception as exc:  # noqa: BLE001
            console.print(
                f"  [{idx}/{len(rallies_with_block)}] {rally.rally_id[:8]} "
                f"[red]REJECTED — {type(exc).__name__}: {exc}[/red]"
            )
            continue

        pre_dedup = last_pre_dedup()
        last_call = last_dedup_call()
        contacts = list(contact_seq.contacts)
        tolerance_frames = max(1, round(rally.fps * 167 / 1000))
        teams = match_teams_by_rally.get(rally.rally_id)

        # Find missed GT blocks (GT block with no BLOCK pred within tolerance).
        block_pred_frames = [
            a.get("frame") for a in pred_actions
            if a.get("action") == "block" and not a.get("isSynthetic")
        ]

        rally_misses = 0
        rally_tps = 0
        for gt in rally.gt_labels:
            if gt.action != "block":
                continue
            matched_block = any(
                abs((pf or -10**9) - gt.frame) <= tolerance_frames
                for pf in block_pred_frames
            )
            if matched_block:
                rally_tps += 1
                continue
            rally_misses += 1
            kills_for_gt: list[KillRecord] = []
            if last_call is not None:
                inp_, out_, mind_, adap_ = last_call
                kills_for_gt = replay_kills(
                    inp_, out_, mind_, adap_, gt.frame, window=10,
                )
            snap = _build_snapshot(
                rally, gt.frame, gt.player_track_id,
                contacts, pre_dedup, pred_actions, teams, tolerance_frames,
                kill_records=kills_for_gt,
            )
            all_snaps.append(snap)

        n_block_tp += rally_tps
        n_block_fn += rally_misses

        if rally_misses > 0:
            console.print(
                f"  [{idx}/{len(rallies_with_block)}] {rally.rally_id[:8]} "
                f"{len([g for g in rally.gt_labels if g.action == 'block'])} "
                f"GT blocks, {rally_tps} TP, {rally_misses} miss"
            )
        elif rally_tps > 0:
            console.print(
                f"  [{idx}/{len(rallies_with_block)}] {rally.rally_id[:8]} "
                f"[green]all {rally_tps} blocks matched[/green]"
            )

    # Summary bucket table
    bucket_counts = Counter(s.bucket for s in all_snaps)
    console.print()
    console.print(
        f"[bold]Block summary:[/bold] {n_block_tp} TP, {n_block_fn} FN "
        f"(of {total_gt_blocks} GT blocks)"
    )
    console.print()
    summary = Table(title="Missed-block failure buckets")
    summary.add_column("Bucket")
    summary.add_column("Count", justify="right")
    summary.add_column("% of misses", justify="right")
    for bucket, count in bucket_counts.most_common():
        pct = count / max(1, n_block_fn) * 100.0
        summary.add_row(bucket, str(count), f"{pct:.1f}%")
    console.print(summary)

    # Per-miss detail table
    detail = Table(title="Per-missed-block detail", show_lines=False)
    detail.add_column("Rally", max_width=10)
    detail.add_column("GTf", justify="right")
    detail.add_column("ΔC", justify="right")  # nearest contact offset
    detail.add_column("PredA", max_width=7)  # pred action at nearest
    detail.add_column("prev", max_width=7)
    detail.add_column("Δf", justify="right")  # frame gap to prev
    detail.add_column("xSide", justify="center")
    detail.add_column("@net", justify="center")
    detail.add_column("oppCand", justify="center")
    detail.add_column("predDedup", justify="center")
    detail.add_column("Bucket")
    for s in all_snaps:
        detail.add_row(
            s.rally_id[:8],
            str(s.gt_frame),
            str(s.nearest_contact_offset) if s.nearest_contact_offset is not None else "—",
            s.pred_action_at_nearest or "—",
            s.prev_action_type or "—",
            str(s.frame_gap_to_prev) if s.frame_gap_to_prev is not None else "—",
            "Y" if s.cross_side_vs_prev else ("N" if s.cross_side_vs_prev is False else "?"),
            "Y" if s.is_at_net_of_contact else ("N" if s.is_at_net_of_contact is False else "?"),
            "Y" if s.opposite_side_in_candidates else ("N" if s.opposite_side_in_candidates is False else "?"),
            "Y" if s.pre_dedup_candidate_exists else "N",
            s.bucket,
        )
    console.print(detail)

    # Dedup-kill cause summary across all near-GT killed candidates.
    all_kills = [k for s in all_snaps for k in s.kill_records]
    if all_kills:
        cause_counts: Counter[str] = Counter(k["cause"] for k in all_kills)
        kill_summary = Table(title="Dedup-kill causes (all candidates within ±10f of any GT block)")
        kill_summary.add_column("Cause")
        kill_summary.add_column("Count", justify="right")
        for cause, count in cause_counts.most_common():
            kill_summary.add_row(cause, str(count))
        console.print()
        console.print(kill_summary)

        # Per-rally kill detail — only for missed-block rallies that have ≥1 kill.
        kill_detail = Table(title="Per-missed-block dedup kills near GT")
        kill_detail.add_column("Rally", max_width=10)
        kill_detail.add_column("GTf", justify="right")
        kill_detail.add_column("Kf", justify="right")
        kill_detail.add_column("KSide", max_width=5)
        kill_detail.add_column("KConf", justify="right")
        kill_detail.add_column("K@net", justify="center")
        kill_detail.add_column("Bf", justify="right")
        kill_detail.add_column("BSide", max_width=5)
        kill_detail.add_column("BConf", justify="right")
        kill_detail.add_column("B@net", justify="center")
        kill_detail.add_column("eMin", justify="right")
        kill_detail.add_column("Gap", justify="right")
        kill_detail.add_column("Cause")
        for s in all_snaps:
            for k in s.kill_records:
                kill_detail.add_row(
                    s.rally_id[:8],
                    str(s.gt_frame),
                    str(k["killed_frame"]),
                    k["killed_court_side"][:5],
                    f"{k['killed_conf']:.2f}",
                    "Y" if k["killed_is_at_net"] else "N",
                    str(k["blocker_frame"]),
                    k["blocker_court_side"][:5],
                    f"{k['blocker_conf']:.2f}",
                    "Y" if k["blocker_is_at_net"] else "N",
                    str(k["effective_min"]),
                    str(k["gap"]),
                    k["cause"],
                )
        console.print(kill_detail)

        # Crisp signal: among kills, what fraction are at-net cross-side
        # candidates that lost on confidence (not on the cross-side gate)?
        cross_at_net_lost = sum(
            1 for k in all_kills
            if k["cross_side"] and k["killed_is_at_net"]
        )
        unknown_side_killed = sum(
            1 for k in all_kills if not k["side_known_both"]
        )
        console.print()
        console.print(
            f"[bold]Carve-out signal:[/bold] "
            f"{cross_at_net_lost} kills are at-net cross-side (the pattern "
            f"adaptive_dedup *should* preserve), "
            f"{unknown_side_killed} kills had unknown-side fallback to min={(_DEDUP_CALLS[-1][2] if _DEDUP_CALLS else 12)}."
        )

    # Write JSON report
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "pool_size": len(rallies),
        "rallies_with_gt_block": len(rallies_with_block),
        "total_gt_blocks": total_gt_blocks,
        "n_block_tp": n_block_tp,
        "n_block_fn": n_block_fn,
        "bucket_counts": dict(bucket_counts),
        "missed_blocks": [
            {
                "rally_id": s.rally_id,
                "video_id": s.video_id,
                "gt_frame": s.gt_frame,
                "gt_player_tid": s.gt_player_tid,
                "fps": s.fps,
                "nearest_contact_offset": s.nearest_contact_offset,
                "nearest_contact_frame": s.nearest_contact_frame,
                "nearest_contact_is_at_net": s.nearest_contact_is_at_net,
                "nearest_contact_court_side": s.nearest_contact_court_side,
                "nearest_contact_player_tid": s.nearest_contact_player_tid,
                "pred_action_at_nearest": s.pred_action_at_nearest,
                "opposite_side_in_candidates": s.opposite_side_in_candidates,
                "pre_dedup_candidate_exists": s.pre_dedup_candidate_exists,
                "prev_action_type": s.prev_action_type,
                "prev_court_side": s.prev_court_side,
                "frame_gap_to_prev": s.frame_gap_to_prev,
                "cross_side_vs_prev": s.cross_side_vs_prev,
                "is_at_net_of_contact": s.is_at_net_of_contact,
                "bucket": s.bucket,
                "pre_dedup_near": s.pre_dedup_near,
                "kill_records": s.kill_records,
            }
            for s in all_snaps
        ],
    }
    out_path.write_text(json.dumps(report, indent=2))
    console.print(f"\n[dim]Report written to {out_path}[/dim]")


if __name__ == "__main__":
    main()
