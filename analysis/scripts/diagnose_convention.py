"""Diagnose convention determination failures in the Viterbi scorer.

For each video with a convention gap (dual-hypothesis vs GT-calibrated),
shows per-rally:
  - Formation side + confidence
  - Player IDs per court side (from track_to_player)
  - GT vs automated side-switch boundaries
  - Convention vote (near=A or near=B)

Usage:
    cd analysis
    uv run python scripts/diagnose_convention.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rallycut.evaluation.tracking.db import get_connection  # noqa: E402
from rallycut.scoring.cross_rally_viterbi import (  # noqa: E402
    RallyObservation,
    calibrate_initial_side,
    decode_video,
    decode_video_dual_hypothesis,
)
from rallycut.tracking.action_classifier import (  # noqa: E402
    _find_serving_side_by_formation,
)
from rallycut.tracking.player_tracker import PlayerPosition  # noqa: E402


@dataclass
class DiagRally:
    rally_id: str
    video_id: str
    start_ms: int
    rally_index: int
    gt_serving_team: str
    positions: list[PlayerPosition]
    court_split_y: float | None
    side_flipped: bool  # GT cumulative flip
    prod_semantic_flip: bool  # automated cumulative flip
    track_to_player: dict[int, int]  # track_id -> player_id


def _parse_positions(raw: list[dict[str, Any]]) -> list[PlayerPosition]:
    return [
        PlayerPosition(
            frame_number=pp["frameNumber"],
            track_id=pp["trackId"],
            x=pp["x"],
            y=pp["y"],
            width=pp.get("width", 0.05),
            height=pp.get("height", 0.10),
            confidence=pp.get("confidence", 1.0),
            keypoints=pp.get("keypoints"),
        )
        for pp in raw
    ]


def _load_data() -> dict[str, list[DiagRally]]:
    """Load rally data with track_to_player for diagnostic."""

    # GT side switches
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT id, player_matching_gt_json FROM videos
            WHERE id IN (SELECT DISTINCT video_id FROM rallies
                         WHERE gt_serving_team IS NOT NULL)
        """)
        video_gt_switches: dict[str, set[int]] = {}
        for row in cur.fetchall():
            vid = str(row[0])
            gt = row[1]
            sw = list(gt.get("sideSwitches", [])) if isinstance(gt, dict) else []
            video_gt_switches[vid] = set(sw)

    # Match analysis: track_to_player + automated side switches
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT id, match_analysis_json FROM videos
            WHERE id IN (SELECT DISTINCT video_id FROM rallies
                         WHERE gt_serving_team IS NOT NULL)
              AND match_analysis_json IS NOT NULL
        """)
        # rally_id -> track_to_player
        rally_t2p: dict[str, dict[int, int]] = {}
        # rally_id -> prod_semantic_flip
        rally_prod_flip: dict[str, bool] = {}
        for row in cur.fetchall():
            ma = row[1]
            if not isinstance(ma, dict):
                continue
            count = 0
            for entry in ma.get("rallies") or []:
                rid = entry.get("rallyId") or entry.get("rally_id")
                if not rid:
                    continue
                # Track to player
                t2p = entry.get("trackToPlayer") or entry.get("track_to_player") or {}
                rally_t2p[rid] = {int(k): int(v) for k, v in t2p.items()}
                # Semantic flip
                rally_prod_flip[rid] = (count % 2 == 1)
                if entry.get("sideSwitchDetected") or entry.get("side_switch_detected"):
                    count += 1

    # Rally data
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT r.id, r.video_id, r.start_ms, r.gt_serving_team,
                   pt.positions_json, pt.court_split_y
            FROM rallies r LEFT JOIN player_tracks pt ON pt.rally_id = r.id
            WHERE r.video_id IN (SELECT DISTINCT video_id FROM rallies
                                 WHERE gt_serving_team IS NOT NULL)
              AND r.gt_serving_team IS NOT NULL
            ORDER BY r.video_id, r.start_ms
        """)
        raw: dict[str, list[tuple[Any, ...]]] = defaultdict(list)
        for row in cur.fetchall():
            raw[str(row[1])].append(row)

    out: dict[str, list[DiagRally]] = {}
    for vid, rows in raw.items():
        rows.sort(key=lambda r: r[2])
        gt_switches = video_gt_switches.get(vid, set())
        gt_flipped = False
        vid_out: list[DiagRally] = []
        for idx, (rid, _, sms, gt, pj, split_y) in enumerate(rows):
            rid_str = str(rid)
            if idx in gt_switches:
                gt_flipped = not gt_flipped
            vid_out.append(DiagRally(
                rally_id=rid_str,
                video_id=vid,
                start_ms=sms or 0,
                rally_index=idx,
                gt_serving_team=gt,
                positions=_parse_positions(pj or []),
                court_split_y=split_y,
                side_flipped=gt_flipped,
                prod_semantic_flip=rally_prod_flip.get(rid_str, False),
                track_to_player=rally_t2p.get(rid_str, {}),
            ))
        out[vid] = vid_out
    return out


def _classify_tracks_per_side(
    positions: list[PlayerPosition],
    net_y: float,
    window_frames: int = 120,
) -> tuple[list[int], list[int]]:
    """Classify track_ids into near (y > net_y) and far (y < net_y) sides."""
    from collections import defaultdict as dd
    track_ys: dict[int, list[float]] = dd(list)
    for p in positions:
        if p.frame_number > window_frames:
            break
        track_ys[p.track_id].append(p.y)

    near_tids: list[int] = []
    far_tids: list[int] = []
    for tid, ys in track_ys.items():
        avg_y = sum(ys) / len(ys)
        if avg_y > net_y:
            near_tids.append(tid)
        else:
            far_tids.append(tid)
    return near_tids, far_tids


def _team_from_pids(pids: list[int]) -> str | None:
    """Determine team from player IDs. {1,2} -> A, {3,4} -> B."""
    a_count = sum(1 for p in pids if p <= 2)
    b_count = sum(1 for p in pids if p >= 3)
    if a_count > b_count:
        return "A"
    if b_count > a_count:
        return "B"
    return None


def diagnose_video(vid: str, rallies: list[DiagRally]) -> None:
    """Print per-rally diagnostic for a single video."""
    print(f"\n{'=' * 80}")
    print(f"VIDEO: {vid}")
    print(f"  Rallies: {len(rallies)}")
    print(f"{'=' * 80}")

    # Compute formation observations and GT/dual results
    observations: list[RallyObservation] = []
    gt_teams: list[str | None] = []
    for r in rallies:
        net_y = r.court_split_y or 0.5
        fs, fc = _find_serving_side_by_formation(
            r.positions, net_y=net_y, start_frame=0,
        )
        observations.append(RallyObservation(
            rally_id=r.rally_id,
            formation_side=fs,
            formation_confidence=fc,
            gt_serving_team=r.gt_serving_team,
        ))
        gt_teams.append(r.gt_serving_team)

    # GT-calibrated decode
    gt_switch_indices: set[int] = set()
    for i in range(1, len(rallies)):
        if rallies[i].side_flipped != rallies[i - 1].side_flipped:
            gt_switch_indices.add(i)
    initial_near_is_a = calibrate_initial_side(
        observations, gt_teams, gt_switch_indices,
    )

    gt_decoded = decode_video(
        observations, initial_near_is_a=initial_near_is_a,
        side_switch_rallies=gt_switch_indices,
    )

    # Dual-hypothesis decode
    prod_switch_indices: set[int] = set()
    for i in range(1, len(rallies)):
        if rallies[i].prod_semantic_flip != rallies[i - 1].prod_semantic_flip:
            prod_switch_indices.add(i)

    dual_decoded = decode_video_dual_hypothesis(
        observations, side_switch_rallies=prod_switch_indices,
    )

    print(f"\n  GT initial_near_is_a: {initial_near_is_a}")
    print(f"  GT switches at indices: {sorted(gt_switch_indices)}")
    print(f"  Prod switches at indices: {sorted(prod_switch_indices)}")

    # Detect what convention dual-hyp picked
    # If dual picks near=A at rally 0, initial_near_is_a=True
    if dual_decoded:
        dual_r0 = dual_decoded[0]
        dual_near_is_a = (dual_r0.serving_team == "A" and dual_r0.serving_side == "near") or \
                         (dual_r0.serving_team == "B" and dual_r0.serving_side == "far")
        print(f"  Dual initial_near_is_a: {dual_near_is_a}")
        print(f"  Convention {'MATCHES' if dual_near_is_a == initial_near_is_a else 'INVERTED'}")

    # Per-rally analysis
    print(f"\n  {'#':>3s}  {'GT':>2s}  {'form':>5s}  {'conf':>5s}  "
          f"{'near_pids':>12s}  {'far_pids':>12s}  {'near_tm':>7s}  "
          f"{'GT_sw':>5s}  {'P_sw':>5s}  {'gt_dec':>6s}  {'du_dec':>6s}  {'match':>5s}")
    print(f"  {'---':>3s}  {'--':>2s}  {'-----':>5s}  {'-----':>5s}  "
          f"{'------------':>12s}  {'------------':>12s}  {'-------':>7s}  "
          f"{'-----':>5s}  {'-----':>5s}  {'------':>6s}  {'------':>6s}  {'-----':>5s}")

    # Identity votes
    votes_near_a = 0
    votes_near_b = 0
    cumulative_gt_switches = 0

    for i, r in enumerate(rallies):
        net_y = r.court_split_y or 0.5
        obs = observations[i]

        # Classify tracks per side
        near_tids, far_tids = _classify_tracks_per_side(r.positions, net_y)

        # Map to player IDs
        near_pids = sorted(r.track_to_player.get(t, -1) for t in near_tids
                           if r.track_to_player.get(t) is not None)
        far_pids = sorted(r.track_to_player.get(t, -1) for t in far_tids
                          if r.track_to_player.get(t) is not None)

        # Determine near team
        near_team = _team_from_pids(near_pids)

        # Track GT switches
        gt_sw = "SW" if i in gt_switch_indices else ""
        prod_sw = "SW" if i in prod_switch_indices else ""

        if i in gt_switch_indices:
            cumulative_gt_switches += 1
        gt_flipped = (cumulative_gt_switches % 2 == 1)

        # Identity vote (accounting for GT switches)
        if near_team:
            effective_team = near_team
            if gt_flipped:
                effective_team = "B" if near_team == "A" else "A"
            if effective_team == "A":
                votes_near_a += 1
            else:
                votes_near_b += 1

        gt_dec = gt_decoded[i].serving_team if i < len(gt_decoded) else "?"
        du_dec = dual_decoded[i].serving_team if i < len(dual_decoded) else "?"
        match = "OK" if gt_dec == r.gt_serving_team and du_dec == r.gt_serving_team else \
                "gt" if gt_dec == r.gt_serving_team else \
                "du" if du_dec == r.gt_serving_team else "MISS"

        print(f"  {i:3d}  {r.gt_serving_team:>2s}  "
              f"{obs.formation_side or 'None':>5s}  {obs.formation_confidence:5.2f}  "
              f"{str(near_pids):>12s}  {str(far_pids):>12s}  {near_team or '?':>7s}  "
              f"{gt_sw:>5s}  {prod_sw:>5s}  {gt_dec:>6s}  {du_dec:>6s}  {match:>5s}")

    gt_correct = sum(1 for r, d in zip(rallies, gt_decoded)
                     if d.serving_team == r.gt_serving_team)
    dual_correct = sum(1 for r, d in zip(rallies, dual_decoded)
                       if d.serving_team == r.gt_serving_team)

    print(f"\n  Identity votes: near_is_A={votes_near_a}, near_is_B={votes_near_b}")
    if votes_near_a + votes_near_b > 0:
        identity_convention = votes_near_a >= votes_near_b
        print(f"  Identity convention: near_is_A={identity_convention} "
              f"(confidence: {abs(votes_near_a - votes_near_b) / (votes_near_a + votes_near_b):.2f})")
        print(f"  GT convention: near_is_A={initial_near_is_a}")
        print(f"  {'CORRECT' if identity_convention == initial_near_is_a else 'WRONG'}")

    print(f"\n  GT-cal accuracy: {gt_correct}/{len(rallies)} = {gt_correct/len(rallies)*100:.1f}%")
    print(f"  Dual accuracy:   {dual_correct}/{len(rallies)} = {dual_correct/len(rallies)*100:.1f}%")
    print(f"  Gap: {(gt_correct - dual_correct)/len(rallies)*100:+.1f}pp")


def main() -> int:
    print("Convention Determination Diagnostic")
    print("=" * 80)

    data = _load_data()
    total = sum(len(v) for v in data.values())
    print(f"Loaded {len(data)} videos, {total} rallies")

    # Focus on videos with convention gap
    # Compute per-video dual vs GT-cal accuracy
    gaps: list[tuple[str, float, int]] = []
    for vid, rallies in sorted(data.items()):
        observations: list[RallyObservation] = []
        gt_teams: list[str | None] = []
        for r in rallies:
            net_y = r.court_split_y or 0.5
            fs, fc = _find_serving_side_by_formation(r.positions, net_y=net_y, start_frame=0)
            observations.append(RallyObservation(
                rally_id=r.rally_id, formation_side=fs,
                formation_confidence=fc, gt_serving_team=r.gt_serving_team,
            ))
            gt_teams.append(r.gt_serving_team)

        gt_sw: set[int] = set()
        for i in range(1, len(rallies)):
            if rallies[i].side_flipped != rallies[i - 1].side_flipped:
                gt_sw.add(i)
        nia = calibrate_initial_side(observations, gt_teams, gt_sw)
        gt_dec = decode_video(observations, initial_near_is_a=nia, side_switch_rallies=gt_sw)

        prod_sw: set[int] = set()
        for i in range(1, len(rallies)):
            if rallies[i].prod_semantic_flip != rallies[i - 1].prod_semantic_flip:
                prod_sw.add(i)
        dual_dec = decode_video_dual_hypothesis(observations, side_switch_rallies=prod_sw)

        gt_c = sum(1 for r, d in zip(rallies, gt_dec) if d.serving_team == r.gt_serving_team)
        du_c = sum(1 for r, d in zip(rallies, dual_dec) if d.serving_team == r.gt_serving_team)
        gap = (gt_c - du_c) / len(rallies) * 100
        gaps.append((vid, gap, len(rallies)))

    # Show overview
    print(f"\n{'Video':>12s}  {'n':>3s}  {'Gap':>7s}")
    print(f"{'-'*12}  {'-'*3}  {'-'*7}")
    for vid, gap, n in sorted(gaps, key=lambda x: -x[1]):
        marker = " ***" if gap > 5 else ""
        print(f"  {vid[:10]}  {n:3d}  {gap:+6.1f}pp{marker}")

    # Diagnose videos with gap > 5pp
    for vid, gap, _ in sorted(gaps, key=lambda x: -x[1]):
        if gap > 5:
            diagnose_video(vid, data[vid])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
