"""Measure serve-side signals with smart gating strategies.

Tests how signals perform when ONLY applied to low-confidence formation
predictions. The key question: can we safely override the formation model
when it's uncertain AND a secondary signal fires?

Usage:
    cd analysis
    uv run python scripts/measure_gated_signals.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rallycut.evaluation.tracking.db import get_connection  # noqa: E402
from rallycut.tracking.action_classifier import (  # noqa: E402
    _compute_auto_split_y,
    _find_serving_side_by_formation,
)
from rallycut.tracking.player_tracker import PlayerPosition  # noqa: E402

EXCLUDED_VIDEOS = {"0a383519", "627c1add"}


def _parse_positions(raw: list[dict]) -> list[PlayerPosition]:
    return [
        PlayerPosition(
            frame_number=p["frameNumber"], track_id=p["trackId"],
            x=p["x"], y=p["y"], width=p.get("width", 0.05),
            height=p.get("height", 0.10), confidence=p.get("confidence", 1.0),
            keypoints=p.get("keypoints"),
        )
        for p in raw
    ]


def _gt_physical_side(gt_team: str, flipped: bool, near_is_a: bool) -> str:
    nia = near_is_a != flipped
    return ("near" if nia else "far") if gt_team == "A" else ("far" if nia else "near")


# ── Signal: Late-entry track ────────────────────────────────────────────


def signal_late_entry(
    positions: list[PlayerPosition],
    net_y: float,
    window: int = 120,
    min_first_frame: int = 15,
    edge_x: float = 0.08,
    edge_y: float = 0.88,
) -> tuple[str | None, float, str]:
    """Server enters frame late at an edge."""
    by_track: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    for p in positions:
        if p.track_id >= 0 and p.frame_number < window:
            by_track[p.track_id].append((p.frame_number, p.x, p.y + p.height / 2.0))

    if len(by_track) < 2:
        return None, 0.0, "few_tracks"

    track_info: dict[int, dict] = {}
    for tid, pts in by_track.items():
        pts.sort()
        ff = pts[0][0]
        track_info[tid] = {
            "first_frame": ff,
            "first_x": pts[0][1],
            "first_foot_y": pts[0][2],
            "median_foot_y": sorted(p[2] for p in pts)[len(pts) // 2],
        }

    # Effective split
    medians = {tid: info["median_foot_y"] for tid, info in track_info.items()}
    eff = net_y
    nn = sum(1 for y in medians.values() if y > eff)
    nf = len(medians) - nn
    if nn == 0 or nf == 0:
        auto = _compute_auto_split_y(
            [p for p in positions if p.track_id >= 0 and p.frame_number < window]
        )
        if auto:
            eff = auto

    late: list[tuple[int, str]] = []
    for tid, info in track_info.items():
        if info["first_frame"] <= min_first_frame:
            continue
        fx, fy = info["first_x"], info["first_foot_y"]
        at_edge = fx < edge_x or fx > 1.0 - edge_x or fy > edge_y
        if not at_edge:
            continue
        side = "near" if info["median_foot_y"] > eff else "far"
        late.append((tid, side))

    if not late:
        return None, 0.0, "no_late"

    near_late = [t for t in late if t[1] == "near"]
    far_late = [t for t in late if t[1] == "far"]

    if near_late and not far_late:
        max_ff = max(track_info[tid]["first_frame"] for tid, _ in near_late)
        return "near", min(1.0, max_ff / 60.0), f"near_late={len(near_late)}_f={max_ff}"
    elif far_late and not near_late:
        max_ff = max(track_info[tid]["first_frame"] for tid, _ in far_late)
        return "far", min(1.0, max_ff / 60.0), f"far_late={len(far_late)}_f={max_ff}"
    return None, 0.0, "both_sides"


# ── Signal: Closest player at first contact ─────────────────────────────


def signal_contact_proximity(
    positions: list[PlayerPosition],
    net_y: float,
    contacts_json: dict | None,
) -> tuple[str | None, float, str]:
    """The player closest to the ball at first contact is the server."""
    if not contacts_json or not isinstance(contacts_json, dict):
        return None, 0.0, "no_contacts"
    contacts = contacts_json.get("contacts", [])
    if not contacts:
        return None, 0.0, "empty"

    first = contacts[0]
    frame = first.get("frame", 0)
    ball_x = first.get("ballX")
    ball_y = first.get("ballY")
    if ball_x is None or ball_y is None:
        return None, 0.0, "no_ball_pos"

    # Find all players near the contact frame
    player_dists: list[tuple[int, float, float]] = []  # (tid, dist, foot_y)
    for p in positions:
        if p.track_id < 0 or abs(p.frame_number - frame) > 3:
            continue
        foot_y = p.y + p.height / 2.0
        # Wrist approximation: top 30% of bbox
        wrist_y = p.y + p.height * 0.3
        wrist_x = p.x
        dist = ((wrist_x - ball_x) ** 2 + (wrist_y - ball_y) ** 2) ** 0.5
        player_dists.append((p.track_id, dist, foot_y))

    if not player_dists:
        return None, 0.0, "no_players_near"

    # Closest player
    closest = min(player_dists, key=lambda x: x[1])
    tid, dist, foot_y = closest
    side = "near" if foot_y > net_y else "far"

    # Confidence from distance (closer = more confident)
    conf = max(0.0, min(1.0, 1.0 - dist / 0.15))
    return side, conf, f"T{tid}_d={dist:.3f}_{side}"


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> int:
    from scripts.eval_score_tracking import load_score_gt

    video_rallies = load_score_gt()
    video_rallies = {
        vid: rallies for vid, rallies in video_rallies.items()
        if not any(vid.startswith(ex) for ex in EXCLUDED_VIDEOS)
    }
    total = sum(len(v) for v in video_rallies.values())
    print(f"Loaded {total} rallies from {len(video_rallies)} videos")

    # Load contacts
    all_ids = [r.rally_id for rallies in video_rallies.values() for r in rallies]
    contacts_by_rally: dict[str, dict | None] = {}
    with get_connection() as conn, conn.cursor() as cur:
        ph = ", ".join(["%s"] * len(all_ids))
        cur.execute(
            f"SELECT rally_id, contacts_json FROM player_tracks WHERE rally_id IN ({ph})",
            all_ids,
        )
        for rid, cj in cur.fetchall():
            contacts_by_rally[rid] = cj

    # Convention calibration
    initial_near_is_a: dict[str, bool] = {}
    for vid, rallies in video_rallies.items():
        vt, vf = 0, 0
        for r in rallies:
            pos = _parse_positions(r.positions)
            ny = r.court_split_y or 0.5
            ps, _ = _find_serving_side_by_formation(pos, net_y=ny, start_frame=0)
            if ps is None:
                continue
            if ps == _gt_physical_side(r.gt_serving_team, r.side_flipped, True):
                vt += 1
            if ps == _gt_physical_side(r.gt_serving_team, r.side_flipped, False):
                vf += 1
        initial_near_is_a[vid] = vt >= vf

    # Evaluate gated strategies
    @dataclass
    class Rally:
        gt_side: str
        form_side: str | None
        form_conf: float
        late_side: str | None
        late_conf: float
        late_reason: str
        contact_side: str | None
        contact_conf: float
        contact_reason: str

    rallies_data: list[Rally] = []
    for vid, rallies in video_rallies.items():
        nia = initial_near_is_a[vid]
        for r in rallies:
            pos = _parse_positions(r.positions)
            ny = r.court_split_y or 0.5
            gt = _gt_physical_side(r.gt_serving_team, r.side_flipped, nia)
            fs, fc = _find_serving_side_by_formation(pos, net_y=ny, start_frame=0)
            ls, lc, lr = signal_late_entry(pos, ny)
            cs, cc, cr = signal_contact_proximity(pos, ny, contacts_by_rally.get(r.rally_id))
            rallies_data.append(Rally(gt, fs, fc, ls, lc, lr, cs, cc, cr))

    # ── Test gating strategies ───────────────────────────────────────────

    def evaluate_strategy(name: str, predict_fn):
        """Run a prediction strategy and report metrics."""
        correct = 0
        wrong = 0
        overrides = 0
        override_correct = 0
        regressions = 0  # was correct, now wrong
        for rd in rallies_data:
            pred = predict_fn(rd)
            if pred == rd.gt_side:
                correct += 1
            else:
                wrong += 1
            # Track overrides vs baseline
            if pred != rd.form_side:
                overrides += 1
                if pred == rd.gt_side and rd.form_side != rd.gt_side:
                    override_correct += 1  # fixed an error
                elif pred != rd.gt_side and rd.form_side == rd.gt_side:
                    regressions += 1  # broke a correct one

        n = len(rallies_data)
        acc = correct / n * 100
        return (f"  {name:<45s}  acc={acc:>5.1f}%  "
                f"overrides={overrides:>3d}  fixed={override_correct:>3d}  "
                f"regressed={regressions:>3d}  net={override_correct - regressions:>+3d}")

    baseline_correct = sum(1 for r in rallies_data if r.form_side == r.gt_side)
    print(f"\nBaseline formation: {baseline_correct}/{len(rallies_data)} "
          f"= {baseline_correct / len(rallies_data) * 100:.1f}%\n")

    print("=== Strategy: Late-entry override ===")
    for gate in [0.05, 0.10, 0.15, 0.20, 0.30, 1.0]:
        def make_fn(g):
            def fn(rd):
                if rd.form_conf < g and rd.late_side is not None:
                    return rd.late_side
                return rd.form_side
            return fn
        print(evaluate_strategy(f"late_entry gate<{gate:.2f}", make_fn(gate)))

    print("\n=== Strategy: Contact-proximity override ===")
    for gate in [0.05, 0.10, 0.15, 0.20, 0.30, 1.0]:
        def make_fn(g):
            def fn(rd):
                if rd.form_conf < g and rd.contact_side is not None:
                    return rd.contact_side
                return rd.form_side
            return fn
        print(evaluate_strategy(f"contact_prox gate<{gate:.2f}", make_fn(gate)))

    print("\n=== Strategy: Late-entry only when abstain/None ===")
    def fn_late_abstain(rd):
        if rd.form_side is None and rd.late_side is not None:
            return rd.late_side
        return rd.form_side
    print(evaluate_strategy("late_entry (abstain only)", fn_late_abstain))

    print("\n=== Strategy: Contact-proximity only when abstain/None ===")
    def fn_contact_abstain(rd):
        if rd.form_side is None and rd.contact_side is not None:
            return rd.contact_side
        return rd.form_side
    print(evaluate_strategy("contact_prox (abstain only)", fn_contact_abstain))

    print("\n=== Strategy: Both signals agree → override ===")
    for gate in [0.10, 0.15, 0.20, 0.30]:
        def make_fn(g):
            def fn(rd):
                if rd.form_conf < g and rd.late_side is not None and rd.contact_side is not None:
                    if rd.late_side == rd.contact_side:
                        return rd.late_side
                return rd.form_side
            return fn
        print(evaluate_strategy(f"both_agree gate<{gate:.2f}", make_fn(gate)))

    print("\n=== Strategy: Either signal when abstain, both when low-conf ===")
    for gate in [0.10, 0.15, 0.20]:
        def make_fn(g):
            def fn(rd):
                if rd.form_side is None:
                    # Abstain: use any available signal
                    if rd.late_side is not None:
                        return rd.late_side
                    if rd.contact_side is not None:
                        return rd.contact_side
                elif rd.form_conf < g:
                    # Low conf: require both to agree
                    if (rd.late_side is not None and rd.contact_side is not None
                            and rd.late_side == rd.contact_side):
                        return rd.late_side
                return rd.form_side
            return fn
        print(evaluate_strategy(f"hybrid gate<{gate:.2f}", make_fn(gate)))

    # Detail: show what each strategy does to current errors
    print("\n=== Detail: contact_prox on formation errors (gate<0.20) ===")
    print(f"{'gt':>5s}  {'form':>5s}  {'fconf':>5s}  {'contact':>8s}  {'cconf':>5s}  {'result':>6s}  {'reason'}")
    for rd in rallies_data:
        if rd.form_side == rd.gt_side:
            continue
        would_override = rd.form_conf < 0.20 and rd.contact_side is not None
        new_pred = rd.contact_side if would_override else rd.form_side
        result = "FIX" if new_pred == rd.gt_side and rd.form_side != rd.gt_side else (
            "REGR" if new_pred != rd.gt_side and rd.form_side == rd.gt_side else "same")
        if would_override:
            print(f"{rd.gt_side:>5s}  {(rd.form_side or '-'):>5s}  {rd.form_conf:>5.2f}  "
                  f"{(rd.contact_side or '-'):>8s}  {rd.contact_conf:>5.2f}  "
                  f"{result:>6s}  {rd.contact_reason}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
