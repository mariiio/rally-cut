"""Phase 1.7 probe — does `enable_player_motion_candidates` surface missed BLOCKs?

Hypothesis (from contact_detection_fn_v1_2026_05_12 memo):
  Re-enabling `enable_player_motion_candidates` in `contact_detector.py` would
  surface the 10 GT BLOCKs that the pipeline mistyped as `attack` in the 12
  trusted-GT corpus (titi/toto/lulu/wawa/caco/cece/cici/cuco/gaga/kaka/juju/yeye).

Method:
  1. Find every GT BLOCK row whose nearest pipeline action (±5 frames) is NOT
     `block` (i.e. mis-typed as attack, or no pipeline action at all).
  2. For each such case, re-run candidate generation with
     `enable_player_motion_candidates=True` and check whether a candidate
     fires within ±5 frames of the GT block frame.
  3. Also count total player_motion candidates fired fleet-wide on those 12
     videos, broken down by:
       - already covered by an existing pipeline contact (deduped)
       - new candidate not near any pipeline contact:
           a) near a GT row → useful new candidate
           b) near no GT row → likely FP

Output: reports/phase_1_7_probe_2026_05_14/results.md + .json

Usage:
    cd analysis
    uv run python scripts/probe_phase_1_7_player_motion_blocks_2026_05_14.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from rallycut.evaluation.tracking.db import get_connection  # noqa: E402
from rallycut.tracking.ball_tracker import BallPosition  # noqa: E402
from rallycut.tracking.contact_detector import (  # noqa: E402
    ContactDetectionConfig,
    _prepare_candidates,
)
from rallycut.tracking.player_tracker import PlayerPosition  # noqa: E402

VIDEO_NAMES = [
    "titi", "toto", "lulu", "wawa", "caco", "cece",
    "cici", "cuco", "gaga", "kaka", "juju", "yeye",
]
FRAME_TOL = 5

REPORT_DIR = HERE.parent / "reports" / "phase_1_7_probe_2026_05_14"
JSON_PATH = REPORT_DIR / "results.json"
MD_PATH = REPORT_DIR / "results.md"


@dataclass
class RallyBundle:
    rally_id: str
    video_name: str
    actions: list[dict[str, Any]]  # pipeline actions (with frame, action, playerTrackId, ...)
    pipeline_contact_frames: list[int]
    ball_positions: list[BallPosition]
    player_positions: list[PlayerPosition]


@dataclass
class GTRow:
    rally_id: str
    frame: int
    action: str
    resolved_tid: int | None


def _hydrate_ball(rows: list[dict[str, Any]]) -> list[BallPosition]:
    out: list[BallPosition] = []
    for bp in rows or []:
        try:
            out.append(BallPosition(
                frame_number=int(bp["frameNumber"]),
                x=float(bp.get("x", 0.0)),
                y=float(bp.get("y", 0.0)),
                confidence=float(bp.get("confidence", 0.0)),
            ))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _hydrate_players(rows: list[dict[str, Any]]) -> list[PlayerPosition]:
    out: list[PlayerPosition] = []
    for p in rows or []:
        try:
            out.append(PlayerPosition(
                frame_number=int(p["frameNumber"]),
                track_id=int(p["trackId"]),
                x=float(p["x"]),
                y=float(p["y"]),
                width=float(p["width"]),
                height=float(p["height"]),
                confidence=float(p.get("confidence", 0.0)),
            ))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def fetch_rallies() -> tuple[list[RallyBundle], list[GTRow]]:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, name FROM videos WHERE name = ANY(%s)",
            [VIDEO_NAMES],
        )
        video_id_to_name = {str(r[0]): str(r[1]) for r in cur.fetchall()}
        video_ids = list(video_id_to_name.keys())

        cur.execute(
            """
            SELECT r.id, r.video_id, pt.actions_json, pt.contacts_json,
                   pt.positions_json, pt.ball_positions_json
            FROM rallies r
            JOIN player_tracks pt ON pt.rally_id = r.id
            WHERE r.video_id = ANY(%s)
              AND (r.status = 'CONFIRMED' OR r.status IS NULL)
            ORDER BY r.start_ms
            """,
            [video_ids],
        )
        prows = cur.fetchall()

        cur.execute(
            """
            SELECT g.rally_id, g.frame, g.action, g.resolved_track_id
            FROM rally_action_ground_truth g
            JOIN rallies r ON r.id = g.rally_id
            WHERE r.video_id = ANY(%s)
            """,
            [video_ids],
        )
        gt_db_rows = cur.fetchall()

    bundles: list[RallyBundle] = []
    for rrow in prows:
        rally_id = str(rrow[0])
        video_id = str(rrow[1])
        aj = rrow[2] or {}
        cj = rrow[3] or {}
        positions = rrow[4] or []
        ball_positions = rrow[5] or []

        actions = aj.get("actions", []) if isinstance(aj, dict) else []
        contacts = cj.get("contacts", []) if isinstance(cj, dict) else []
        contact_frames = sorted({
            int(c["frame"]) for c in contacts if "frame" in c
        })

        bundles.append(RallyBundle(
            rally_id=rally_id,
            video_name=video_id_to_name.get(video_id, video_id),
            actions=sorted(actions, key=lambda a: int(a.get("frame", 0))),
            pipeline_contact_frames=contact_frames,
            ball_positions=_hydrate_ball(ball_positions),
            player_positions=_hydrate_players(positions),
        ))

    gts: list[GTRow] = []
    for g in gt_db_rows:
        gts.append(GTRow(
            rally_id=str(g[0]),
            frame=int(g[1]),
            action=str(g[2]),
            resolved_tid=int(g[3]) if g[3] is not None else None,
        ))

    return bundles, gts


def main() -> int:
    print("=== Phase 1.7 probe: player_motion candidates → missed BLOCKs ===")
    print(f"Videos: {VIDEO_NAMES}")
    print(f"FRAME_TOL = ±{FRAME_TOL}")
    print()

    print("Fetching rally data...")
    bundles, gt_rows = fetch_rallies()
    bundles_by_id = {b.rally_id: b for b in bundles}
    print(f"Loaded {len(bundles)} rallies, {len(gt_rows)} GT rows")

    # ----- Step 2: identify missed BLOCK cases -----
    # For each GT BLOCK row, find nearest pipeline action of ANY type ±5 frames;
    # case is "missed" iff nearest is NOT `block` (i.e. mis-typed or absent).
    missed_block_cases: list[dict[str, Any]] = []
    block_match_pipeline_block = 0
    for g in gt_rows:
        if g.action != "BLOCK":
            continue
        b = bundles_by_id.get(g.rally_id)
        if b is None:
            continue
        # Find nearest pipeline action of any type within ±5
        nearest = None
        nearest_dt = FRAME_TOL + 1
        for a in b.actions:
            af = int(a.get("frame", -1))
            dt = abs(af - g.frame)
            if dt <= FRAME_TOL and dt < nearest_dt:
                nearest = a
                nearest_dt = dt
        nearest_action = (
            str(nearest.get("action", "")).lower() if nearest else None
        )
        nearest_pid = (
            int(nearest["playerTrackId"])
            if nearest and nearest.get("playerTrackId") is not None
            else None
        )
        if nearest_action == "block":
            block_match_pipeline_block += 1
            continue  # NOT a missed case
        missed_block_cases.append({
            "rally_id": g.rally_id,
            "video": b.video_name,
            "gt_frame": g.frame,
            "gt_pid": g.resolved_tid,
            "nearest_pipeline_action": nearest_action,  # may be None
            "nearest_pipeline_pid": nearest_pid,
            "nearest_pipeline_dt": nearest_dt if nearest else None,
        })

    print(f"\nGT BLOCK rows: {sum(1 for g in gt_rows if g.action == 'BLOCK')}")
    print(f"  → matched to pipeline BLOCK (no work): {block_match_pipeline_block}")
    print(f"  → MISSED (probe targets): {len(missed_block_cases)}")
    print()

    # ----- Step 3 & 4: re-run candidate gen with flag=True per rally -----
    # Use ContactDetectionConfig defaults BUT enable_player_motion_candidates=True.
    # We only call _prepare_candidates (pre-classifier) — that's where the
    # generator fires. We don't run classifier validation, so this is a pure
    # raw-candidate probe (matches the spec).
    cfg_on = ContactDetectionConfig(enable_player_motion_candidates=True)
    cfg_off = ContactDetectionConfig(enable_player_motion_candidates=False)

    rally_motion_frames_on: dict[str, list[int]] = {}
    rally_motion_frames_off: dict[str, list[int]] = {}
    rally_candidates_on: dict[str, list[int]] = {}
    rally_candidates_off: dict[str, list[int]] = {}

    # GT frames per rally for FP/TP classification
    gt_frames_by_rally: dict[str, list[int]] = defaultdict(list)
    for g in gt_rows:
        gt_frames_by_rally[g.rally_id].append(g.frame)

    print(f"Running candidate generation on {len(bundles)} rallies (flag ON vs OFF)...")
    for i, b in enumerate(bundles, 1):
        # Skip rallies with no ball/player data
        if not b.ball_positions or not b.player_positions:
            rally_motion_frames_on[b.rally_id] = []
            rally_motion_frames_off[b.rally_id] = []
            rally_candidates_on[b.rally_id] = []
            rally_candidates_off[b.rally_id] = []
            continue

        prep_off = _prepare_candidates(
            list(b.ball_positions), list(b.player_positions), cfg_off,
        )
        prep_on = _prepare_candidates(
            list(b.ball_positions), list(b.player_positions), cfg_on,
        )

        cands_off = list(prep_off.candidate_frames or [])
        cands_on = list(prep_on.candidate_frames or [])

        # player_motion frames = candidates in ON but not in OFF.
        # This is the cleanest way to isolate generator output — `n_player_motion`
        # only gives a count, not the frame list, post-merge.
        set_off = set(cands_off)
        motion_frames_on = sorted([f for f in cands_on if f not in set_off])

        rally_motion_frames_on[b.rally_id] = motion_frames_on
        rally_motion_frames_off[b.rally_id] = []
        rally_candidates_on[b.rally_id] = cands_on
        rally_candidates_off[b.rally_id] = cands_off

        if i % 20 == 0 or i == len(bundles):
            print(
                f"  [{i}/{len(bundles)}] {b.video_name} {b.rally_id[:8]} "
                f"motion+={len(motion_frames_on)} cands_on={len(cands_on)} "
                f"cands_off={len(cands_off)}"
            )

    # ----- For each missed BLOCK case: did motion candidate fire near gt_frame? -----
    print("\nStep 3: probe results per missed-BLOCK case")
    print()
    probe_results: list[dict[str, Any]] = []
    motion_fires = 0
    for case in missed_block_cases:
        rid = case["rally_id"]
        gt_f = case["gt_frame"]
        motion_frames = rally_motion_frames_on.get(rid, [])
        fired_near = [f for f in motion_frames if abs(f - gt_f) <= FRAME_TOL]
        # Cardinality of TOTAL motion candidates in this rally
        total_in_rally = len(motion_frames)
        fired_flag = bool(fired_near)
        if fired_flag:
            motion_fires += 1
        # Note: attribution match would require running attribution downstream;
        # the raw generator returns frame numbers only, not pids. We mark
        # attribution as "n/a (raw generator)" but record nearest tid from
        # bbox-proximity at the candidate frame — see comment in
        # `_find_player_motion_candidates`. For this probe we keep it simple
        # and just report whether the candidate fires; downstream attribution
        # would still need to run to get the pid.
        probe_results.append({
            **case,
            "motion_fired_near_gt": fired_flag,
            "motion_fired_frames_in_window": fired_near,
            "total_motion_candidates_in_rally": total_in_rally,
        })

    block_recall_lift = motion_fires
    block_recall_pct = (
        100.0 * motion_fires / len(missed_block_cases)
        if missed_block_cases else 0.0
    )
    print(
        f"Block recall lift: {block_recall_lift}/{len(missed_block_cases)} "
        f"= {block_recall_pct:.1f}%"
    )

    # ----- Step 4: overall FP/TP volume on raw generator output -----
    print("\nStep 4: overall player_motion candidate FP/TP volume")
    total_motion_candidates = 0
    near_existing_pipeline_contact = 0   # would be deduped (not new)
    new_near_gt = 0                      # genuinely new + near a GT row → useful
    new_near_no_gt = 0                   # genuinely new + far from GT → likely FP
    # Per-video breakdown
    per_video_total: dict[str, int] = defaultdict(int)
    per_video_dedup: dict[str, int] = defaultdict(int)
    per_video_new_gt: dict[str, int] = defaultdict(int)
    per_video_new_no_gt: dict[str, int] = defaultdict(int)

    for b in bundles:
        motion_frames = rally_motion_frames_on.get(b.rally_id, [])
        if not motion_frames:
            continue
        pipeline_contacts = b.pipeline_contact_frames
        gt_frames = gt_frames_by_rally.get(b.rally_id, [])
        for f in motion_frames:
            total_motion_candidates += 1
            per_video_total[b.video_name] += 1
            near_existing = any(abs(f - c) <= FRAME_TOL for c in pipeline_contacts)
            if near_existing:
                near_existing_pipeline_contact += 1
                per_video_dedup[b.video_name] += 1
                continue
            near_gt = any(abs(f - g) <= FRAME_TOL for g in gt_frames)
            if near_gt:
                new_near_gt += 1
                per_video_new_gt[b.video_name] += 1
            else:
                new_near_no_gt += 1
                per_video_new_no_gt[b.video_name] += 1

    new_total = new_near_gt + new_near_no_gt
    fp_rate = (
        100.0 * new_near_no_gt / new_total if new_total else 0.0
    )
    print(f"Total player_motion candidates: {total_motion_candidates}")
    print(f"  Near existing pipeline contact (dedup): {near_existing_pipeline_contact}")
    print(f"  Genuinely new: {new_total}")
    print(f"    - near GT row: {new_near_gt}")
    print(f"    - near no GT row (likely FP): {new_near_no_gt}")
    print(f"  FP rate on new candidates: {fp_rate:.1f}%")

    # ----- Write report -----
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # JSON
    out = {
        "videos": VIDEO_NAMES,
        "n_rallies": len(bundles),
        "n_gt_rows": len(gt_rows),
        "n_gt_blocks": sum(1 for g in gt_rows if g.action == "BLOCK"),
        "n_pipeline_block_match": block_match_pipeline_block,
        "missed_block_cases": probe_results,
        "block_recall": {
            "n_missed_cases": len(missed_block_cases),
            "motion_fired_count": motion_fires,
            "pct": block_recall_pct,
        },
        "fp_volume": {
            "total_motion_candidates": total_motion_candidates,
            "near_existing_pipeline_contact": near_existing_pipeline_contact,
            "new_total": new_total,
            "new_near_gt": new_near_gt,
            "new_near_no_gt": new_near_no_gt,
            "fp_rate_on_new_pct": fp_rate,
            "per_video": {
                v: {
                    "total": per_video_total.get(v, 0),
                    "dedup": per_video_dedup.get(v, 0),
                    "new_near_gt": per_video_new_gt.get(v, 0),
                    "new_near_no_gt": per_video_new_no_gt.get(v, 0),
                }
                for v in VIDEO_NAMES
            },
        },
    }
    JSON_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nWrote JSON: {JSON_PATH}")

    # Markdown
    lines: list[str] = []
    lines.append("# Phase 1.7 probe: does player_motion surface missed blocks?")
    lines.append("")
    lines.append(f"Corpus: 12 trusted-GT videos ({', '.join(VIDEO_NAMES)})")
    lines.append(f"Frame tolerance: ±{FRAME_TOL}")
    lines.append("")
    lines.append("## Step 1 — flag mechanics")
    lines.append("")
    lines.append("- Flag: `ContactDetectionConfig.enable_player_motion_candidates` "
                 "(default `False`, defined at `analysis/rallycut/tracking/contact_detector.py:231`).")
    lines.append("- Gated function: `_find_player_motion_candidates` "
                 "(at `analysis/rallycut/tracking/contact_detector.py:824`); "
                 "fired from `_prepare_candidates` step 5f at line 2259.")
    lines.append("- Signal: for each ball frame, finds players within bbox-proximity "
                 "(≤ `player_motion_max_ball_distance=0.20`), then checks peak "
                 "Δy or Δheight over a ±5-frame window against "
                 "`player_motion_min_d_y=0.015` / `player_motion_min_d_height=0.015`. "
                 "Skips frames already within `min_peak_distance_frames=12` of an "
                 "existing candidate. Returns frame numbers only — no pid attribution.")
    lines.append("- Comment in code: \"adds 265 candidates but only 9 TPs (3.4% hit "
                 "rate), hurting classifier LOO CV.\"")
    lines.append("")
    lines.append("## Step 2 — missed BLOCK cases identified")
    lines.append("")
    lines.append(f"- Total GT BLOCK rows: {sum(1 for g in gt_rows if g.action == 'BLOCK')}")
    lines.append(f"- Of those, nearest pipeline action (±5) IS `block`: "
                 f"{block_match_pipeline_block}")
    lines.append(f"- Missed cases (probe targets): {len(missed_block_cases)}")
    lines.append("")
    lines.append("| # | video | rally (8) | gt_frame | gt_pid | "
                 "nearest pipe action | nearest pipe pid | Δframes |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for i, c in enumerate(missed_block_cases, 1):
        dt = c["nearest_pipeline_dt"]
        lines.append(
            f"| {i} | {c['video']} | {c['rally_id'][:8]} | {c['gt_frame']} | "
            f"{c['gt_pid']} | {c['nearest_pipeline_action']} | "
            f"{c['nearest_pipeline_pid']} | {dt if dt is not None else '—'} |"
        )
    lines.append("")
    lines.append("## Step 3 — probe results per case")
    lines.append("")
    lines.append("| # | video | rally (8) | gt_frame | motion fired ±5? | "
                 "fired frames (in window) | total motion cands in rally |")
    lines.append("|---|---|---|---|---|---|---|")
    for i, r in enumerate(probe_results, 1):
        lines.append(
            f"| {i} | {r['video']} | {r['rally_id'][:8]} | {r['gt_frame']} | "
            f"{'YES' if r['motion_fired_near_gt'] else 'no'} | "
            f"{r['motion_fired_frames_in_window']} | "
            f"{r['total_motion_candidates_in_rally']} |"
        )
    lines.append("")
    lines.append("Summary:")
    lines.append(f"- **Block recall lift: {block_recall_lift}/{len(missed_block_cases)} "
                 f"= {block_recall_pct:.1f}%** (cases where a player_motion "
                 f"candidate fires within ±{FRAME_TOL} of GT block frame)")
    lines.append("- Block attribution lift: n/a — the raw generator returns frame "
                 "numbers only. Attribution would be applied downstream by the "
                 "standard pose/temporal attribution path. Not measured in this "
                 "probe (per spec — Step 3 asks about candidate fire, attribution "
                 "is the second-order question).")
    lines.append("")
    lines.append("## Step 4 — overall FP volume")
    lines.append("")
    lines.append(f"- Total player_motion candidates (12 videos): {total_motion_candidates}")
    lines.append(f"- Already covered by existing pipeline contact (within ±{FRAME_TOL}, "
                 f"would be deduped): {near_existing_pipeline_contact}")
    lines.append(f"- Genuinely new candidates: {new_total}")
    lines.append(f"  - Of which, near a GT row (useful): {new_near_gt}")
    lines.append(f"  - Of which, near no GT row (likely FP): {new_near_no_gt}")
    lines.append("")
    lines.append(f"**FP rate on new candidates: {new_near_no_gt}/{new_total} = {fp_rate:.1f}%**")
    lines.append("")
    lines.append("Per video:")
    lines.append("")
    lines.append("| video | total motion cands | near pipe contact (dedup) | "
                 "new near GT | new near no GT |")
    lines.append("|---|---|---|---|---|")
    for v in VIDEO_NAMES:
        lines.append(
            f"| {v} | {per_video_total.get(v, 0)} | {per_video_dedup.get(v, 0)} | "
            f"{per_video_new_gt.get(v, 0)} | {per_video_new_no_gt.get(v, 0)} |"
        )
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    # Apply the spec's decision rules
    if block_recall_lift >= 5 and fp_rate < 50.0:
        verdict = "**SHIP-1.7** — generator surfaces ≥50% of missed blocks and FP rate <50%"
    elif block_recall_lift >= 5 and fp_rate >= 75.0:
        verdict = ("**NEEDS-CO-CONDITIONS** — generator finds blocks, but precision "
                   "floor too low; co-conditions (near-net + in-air + ball-from-far-side) "
                   "mandatory before ship")
    elif block_recall_lift >= 5 and fp_rate >= 50.0:
        verdict = ("**NEEDS-CO-CONDITIONS** — generator finds blocks; FP rate sits "
                   "in the gray zone. Co-conditions strongly recommended before ship")
    elif block_recall_lift < 5:
        verdict = ("**NO-SHIP** — generator does not see the specific missed blocks "
                   "(<50% recall lift on raw candidates). Phase 1.7 needs a different "
                   "approach (e.g. direct net-area motion detection)")
    else:
        verdict = "**NO-SHIP** — fallback"
    lines.append(verdict)
    lines.append("")
    lines.append("Decision rules (from spec):")
    lines.append("- block recall ≥5/10 AND FP rate <50% → SHIP-1.7")
    lines.append("- block recall ≥5/10 AND FP rate ≥75% → NEEDS-CO-CONDITIONS")
    lines.append("- block recall <5/10 → NO-SHIP (generator can't see these blocks)")
    lines.append("")

    MD_PATH.write_text("\n".join(lines))
    print(f"Wrote markdown: {MD_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
