"""Measure candidate serve-side signals on the full dataset.

For each signal, computes precision/recall broken down by whether the
formation predictor is correct or wrong.  This tells us which signals
are safe to integrate and how to gate them.

Signals:
  A — Server-enters-frame: track starts late (>frame 15) at frame edge.
      Gated to cases where one side has 0-1 tracks initially.
  B — First contact attribution: first contact's player determines side.

Usage:
    cd analysis
    uv run python scripts/measure_serve_side_signals.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rallycut.evaluation.tracking.db import get_connection  # noqa: E402
from rallycut.tracking.action_classifier import (  # noqa: E402
    _compute_auto_split_y,
    _find_serving_side_by_formation,
    _serving_side_from_contact,
)
from rallycut.tracking.player_tracker import PlayerPosition  # noqa: E402

EXCLUDED_VIDEOS = {
    "0a383519",  # yoyo / IMG_2313 — low camera angle
    "627c1add",  # caca — tilted camera
}


def _parse_positions(raw: list[dict]) -> list[PlayerPosition]:
    return [
        PlayerPosition(
            frame_number=p["frameNumber"],
            track_id=p["trackId"],
            x=p["x"],
            y=p["y"],
            width=p.get("width", 0.05),
            height=p.get("height", 0.10),
            confidence=p.get("confidence", 1.0),
            keypoints=p.get("keypoints"),
        )
        for p in raw
    ]


def _gt_physical_side(
    gt_serving_team: str, side_flipped: bool, initial_near_is_a: bool,
) -> str:
    near_is_a = initial_near_is_a != side_flipped
    if gt_serving_team == "A":
        return "near" if near_is_a else "far"
    else:
        return "far" if near_is_a else "near"


# ── Signal A: Server-Enters-Frame ────────────────────────────────────────


def signal_a_enters_frame(
    positions: list[PlayerPosition],
    net_y: float,
    window_frames: int = 120,
    entry_frame_threshold: int = 15,
    edge_x_threshold: float = 0.05,
    edge_y_threshold: float = 0.85,
) -> tuple[str | None, float, str]:
    """Detect serving side by a track entering the frame late at an edge.

    Returns (predicted_side, confidence, reason).
    The late-entering player's side = the serving side.
    """
    by_track: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    for p in positions:
        if p.track_id < 0:
            continue
        if p.frame_number < window_frames:
            foot_y = p.y + p.height / 2.0
            by_track[p.track_id].append((p.frame_number, p.x, foot_y))

    if len(by_track) < 2:
        return None, 0.0, "too_few_tracks"

    # Compute per-track: first frame, first position, median foot_y
    track_info: dict[int, dict] = {}
    for tid, pts in by_track.items():
        pts.sort(key=lambda t: t[0])
        first_frame = pts[0][0]
        first_x = pts[0][1]
        first_foot_y = pts[0][2]
        median_foot_y = sorted([p[2] for p in pts])[len(pts) // 2]
        track_info[tid] = {
            "first_frame": first_frame,
            "first_x": first_x,
            "first_foot_y": first_foot_y,
            "median_foot_y": median_foot_y,
        }

    # Determine effective split (same logic as formation predictor)
    track_medians = {tid: info["median_foot_y"] for tid, info in track_info.items()}
    effective_split = net_y
    n_near = sum(1 for y in track_medians.values() if y > effective_split)
    n_far = len(track_medians) - n_near
    if n_near == 0 or n_far == 0:
        windowed = [p for p in positions if p.track_id >= 0 and p.frame_number < window_frames]
        auto = _compute_auto_split_y(windowed)
        if auto is not None:
            effective_split = auto
            n_near = sum(1 for y in track_medians.values() if y > effective_split)
            n_far = len(track_medians) - n_near

    # Find late-entry tracks at frame edges
    late_entries: list[tuple[int, str]] = []  # (track_id, side)
    for tid, info in track_info.items():
        if info["first_frame"] <= entry_frame_threshold:
            continue
        fx = info["first_x"]
        ffy = info["first_foot_y"]
        at_edge = (fx < edge_x_threshold or fx > 1.0 - edge_x_threshold
                   or ffy > edge_y_threshold)
        if not at_edge:
            continue
        side = "near" if info["median_foot_y"] > effective_split else "far"
        late_entries.append((tid, side))

    if not late_entries:
        return None, 0.0, "no_late_entry"

    # Count late entries per side
    near_late = sum(1 for _, s in late_entries if s == "near")
    far_late = sum(1 for _, s in late_entries if s == "far")

    if near_late > 0 and far_late == 0:
        max_ff = max(track_info[tid]["first_frame"] for tid, s in late_entries if s == "near")
        conf = min(1.0, max_ff / 60.0)
        return "near", conf, f"near_late={near_late}"
    elif far_late > 0 and near_late == 0:
        max_ff = max(track_info[tid]["first_frame"] for tid, s in late_entries if s == "far")
        conf = min(1.0, max_ff / 60.0)
        return "far", conf, f"far_late={far_late}"
    else:
        return None, 0.0, "both_sides_late"


# ── Signal B: First Contact Attribution ──────────────────────────────────


def signal_b_first_contact(
    positions: list[PlayerPosition],
    net_y: float,
    contacts_json: dict | None,
) -> tuple[str | None, float, str]:
    """Predict serving side from who made the first contact.

    Returns (predicted_side, confidence, reason).
    """
    if not contacts_json or not isinstance(contacts_json, dict):
        return None, 0.0, "no_contacts"

    contacts = contacts_json.get("contacts", [])
    if not contacts:
        return None, 0.0, "empty_contacts"

    first = contacts[0]
    player_tid = first.get("playerTrackId")
    if player_tid is None or player_tid < 0:
        return None, 0.0, "no_player_tid"

    action = first.get("action", "")
    ball_y = first.get("ballY")
    frame = first.get("frame", 0)

    # Find this player's side from their foot position near the contact frame
    foot_ys: list[float] = []
    for p in positions:
        if p.track_id == player_tid and abs(p.frame_number - frame) <= 5:
            foot_ys.append(p.y + p.height / 2.0)

    if not foot_ys:
        return None, 0.0, "player_not_found"

    median_foot_y = sorted(foot_ys)[len(foot_ys) // 2]
    player_side = "near" if median_foot_y > net_y else "far"

    # If action is classified as "serve", player's side is serving side
    # If "receive", opposite side is serving
    if action == "serve":
        return player_side, 0.8, f"serve_action_{player_side}"
    elif action == "receive":
        opp = "far" if player_side == "near" else "near"
        return opp, 0.7, f"receive_action_{player_side}"

    # No action label — use ball position heuristic
    if ball_y is not None:
        ball_side = "near" if ball_y > net_y else "far"
        if ball_side == player_side:
            # Ball on same side as contact player → likely serve
            return player_side, 0.6, f"ball_same_side_{player_side}"
        else:
            # Ball crossed net → likely receive
            opp = "far" if player_side == "near" else "near"
            return opp, 0.5, f"ball_crossed_{player_side}"

    # Fallback: first contact player is usually the server
    return player_side, 0.4, f"fallback_{player_side}"


# ── Main ─────────────────────────────────────────────────────────────────


@dataclass
class RallySignals:
    rally_id: str
    video_id: str
    video_name: str
    gt_side: str
    formation_side: str | None
    formation_conf: float
    formation_correct: bool | None
    n_tracks: int
    n_near: int
    n_far: int
    signal_a_side: str | None = None
    signal_a_conf: float = 0.0
    signal_a_reason: str = ""
    signal_b_side: str | None = None
    signal_b_conf: float = 0.0
    signal_b_reason: str = ""


def main() -> int:
    from scripts.eval_score_tracking import load_score_gt

    video_rallies = load_score_gt()
    video_rallies = {
        vid: rallies for vid, rallies in video_rallies.items()
        if not any(vid.startswith(ex) for ex in EXCLUDED_VIDEOS)
    }
    total = sum(len(v) for v in video_rallies.values())
    print(f"Loaded {total} rallies from {len(video_rallies)} videos")

    # Load video names
    video_names: dict[str, str] = {}
    with get_connection() as conn, conn.cursor() as cur:
        vids = list(video_rallies.keys())
        ph = ", ".join(["%s"] * len(vids))
        cur.execute(f"SELECT id, s3_key FROM videos WHERE id IN ({ph})", vids)
        for vid, s3_key in cur.fetchall():
            video_names[vid] = Path(s3_key).stem if s3_key else vid[:8]

    # Load contacts_json for all rallies
    all_rally_ids = [r.rally_id for rallies in video_rallies.values() for r in rallies]
    contacts_by_rally: dict[str, dict | None] = {}
    with get_connection() as conn, conn.cursor() as cur:
        ph = ", ".join(["%s"] * len(all_rally_ids))
        cur.execute(
            f"SELECT rally_id, contacts_json FROM player_tracks WHERE rally_id IN ({ph})",
            all_rally_ids,
        )
        for rid, cj in cur.fetchall():
            contacts_by_rally[rid] = cj

    # Per-video convention calibration
    initial_near_is_a: dict[str, bool] = {}
    for vid, rallies in video_rallies.items():
        votes_true = 0
        votes_false = 0
        for r in rallies:
            positions = _parse_positions(r.positions)
            ny = r.court_split_y or 0.5
            pred_side, _ = _find_serving_side_by_formation(positions, net_y=ny, start_frame=0)
            if pred_side is None:
                continue
            gt_if_true = _gt_physical_side(r.gt_serving_team, r.side_flipped, True)
            gt_if_false = _gt_physical_side(r.gt_serving_team, r.side_flipped, False)
            if pred_side == gt_if_true:
                votes_true += 1
            if pred_side == gt_if_false:
                votes_false += 1
        initial_near_is_a[vid] = votes_true >= votes_false

    # Evaluate all signals on all rallies
    results: list[RallySignals] = []
    for vid, rallies in sorted(video_rallies.items(), key=lambda x: video_names.get(x[0], "")):
        near_is_a = initial_near_is_a[vid]
        for r in rallies:
            positions = _parse_positions(r.positions)
            net_y = r.court_split_y or 0.5
            gt_phys = _gt_physical_side(r.gt_serving_team, r.side_flipped, near_is_a)

            # Formation prediction
            pred_side, conf = _find_serving_side_by_formation(
                positions, net_y=net_y, start_frame=0,
            )
            formation_correct = pred_side == gt_phys if pred_side is not None else None

            # Track counts
            by_track: dict[int, list[float]] = defaultdict(list)
            for p in positions:
                if p.track_id >= 0 and p.frame_number < 120:
                    by_track[p.track_id].append(p.y + p.height / 2.0)
            track_medians = {tid: sum(ys) / len(ys) for tid, ys in by_track.items()}
            eff_split = net_y
            n_near = sum(1 for y in track_medians.values() if y > eff_split)
            n_far = len(track_medians) - n_near
            if n_near == 0 or n_far == 0:
                auto = _compute_auto_split_y(
                    [p for p in positions if p.track_id >= 0 and p.frame_number < 120]
                )
                if auto is not None:
                    eff_split = auto
                    n_near = sum(1 for y in track_medians.values() if y > eff_split)
                    n_far = len(track_medians) - n_near

            # Signal A
            a_side, a_conf, a_reason = signal_a_enters_frame(positions, net_y)

            # Signal B
            contacts = contacts_by_rally.get(r.rally_id)
            b_side, b_conf, b_reason = signal_b_first_contact(positions, net_y, contacts)

            results.append(RallySignals(
                rally_id=r.rally_id,
                video_id=vid,
                video_name=video_names.get(vid, vid[:8]),
                gt_side=gt_phys,
                formation_side=pred_side,
                formation_conf=conf,
                formation_correct=formation_correct,
                n_tracks=len(track_medians),
                n_near=n_near,
                n_far=n_far,
                signal_a_side=a_side,
                signal_a_conf=a_conf,
                signal_a_reason=a_reason,
                signal_b_side=b_side,
                signal_b_conf=b_conf,
                signal_b_reason=b_reason,
            ))

    # ── Report ───────────────────────────────────────────────────────────
    formation_correct = [r for r in results if r.formation_correct is True]
    formation_wrong = [r for r in results if r.formation_correct is False]
    formation_abstain = [r for r in results if r.formation_correct is None]
    formation_error = formation_wrong + formation_abstain

    print(f"\n{'='*70}")
    print(f"Formation: {len(formation_correct)} correct, "
          f"{len(formation_wrong)} wrong, {len(formation_abstain)} abstain "
          f"({len(formation_correct)}/{len(results)} = "
          f"{len(formation_correct)/len(results)*100:.1f}%)")
    print(f"{'='*70}")

    for signal_name, get_side in [
        ("A (enters-frame)", lambda r: r.signal_a_side),
        ("B (first-contact)", lambda r: r.signal_b_side),
    ]:
        print(f"\n--- Signal {signal_name} ---")
        for category_name, category_rallies in [
            ("ALL", results),
            ("FORMATION_CORRECT", formation_correct),
            ("FORMATION_WRONG", formation_wrong),
            ("FORMATION_ABSTAIN", formation_abstain),
            ("FORMATION_ERROR (wrong+abstain)", formation_error),
        ]:
            fires = [r for r in category_rallies if get_side(r) is not None]
            correct = [r for r in fires if get_side(r) == r.gt_side]
            wrong = [r for r in fires if get_side(r) != r.gt_side]
            prec = len(correct) / len(fires) * 100 if fires else 0.0

            # Regression: signal disagrees with correct formation
            if category_name == "FORMATION_CORRECT":
                disagree = [r for r in fires if get_side(r) != r.formation_side]
                regr_str = f"  regr={len(disagree)}/{len(category_rallies)}"
            else:
                regr_str = ""

            print(f"  {category_name:<35s}  fires={len(fires):>3d}/{len(category_rallies):<3d}  "
                  f"correct={len(correct):>3d}  wrong={len(wrong):>3d}  "
                  f"prec={prec:>5.1f}%{regr_str}")

    # Signal A detail: show which error rallies it fires on
    print(f"\n--- Signal A detail on formation errors ---")
    print(f"{'rally':>10s}  {'video':<20s}  {'n/f':>5s}  {'gt':>5s}  "
          f"{'A_pred':>6s}  {'A_ok':>4s}  {'reason':<25s}")
    for r in sorted(formation_error, key=lambda x: (x.video_name, x.rally_id)):
        a_ok = "Y" if r.signal_a_side == r.gt_side else ("N" if r.signal_a_side else "-")
        print(f"{r.rally_id[:10]:>10s}  {r.video_name:<20s}  "
              f"{r.n_near}v{r.n_far:>1d}  {r.gt_side:>5s}  "
              f"{(r.signal_a_side or '-'):>6s}  {a_ok:>4s}  {r.signal_a_reason:<25s}")

    # Signal B detail: show which error rallies it fires on
    print(f"\n--- Signal B detail on formation errors ---")
    print(f"{'rally':>10s}  {'video':<20s}  {'n/f':>5s}  {'gt':>5s}  "
          f"{'B_pred':>6s}  {'B_ok':>4s}  {'reason':<25s}")
    for r in sorted(formation_error, key=lambda x: (x.video_name, x.rally_id)):
        b_ok = "Y" if r.signal_b_side == r.gt_side else ("N" if r.signal_b_side else "-")
        print(f"{r.rally_id[:10]:>10s}  {r.video_name:<20s}  "
              f"{r.n_near}v{r.n_far:>1d}  {r.gt_side:>5s}  "
              f"{(r.signal_b_side or '-'):>6s}  {b_ok:>4s}  {r.signal_b_reason:<25s}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
