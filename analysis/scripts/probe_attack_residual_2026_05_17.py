#!/usr/bin/env python3
"""ATTACK residual failure-mode catalog (2026-05-17).

For each GT ATTACK row on trusted-21, run the v2 scorer in-process and
classify the residual errors. Output:

  - Per-error CSV with fields (video, rally_id, gt_frame, gt_tid,
    pipeline_pick, prob_pick, prob_gt, gap, category, ...)
  - Summary table by category (count + share)

Categories (mutually exclusive, checked in order):
  GT_MISSING           — GT resolved_tid not in candidate set
  CROSS_TEAM           — pick and GT on different teams (failure of
                         team-chain / ratio-cap relaxation)
  POSE_BLIND_SPOT      — same team, both contestants arms_raised=1,
                         |wrist_to_ball_pick - wrist_to_ball_gt| < 0.04
  DEPTH_OVERWEIGHT     — same team, pick has smaller bbox_dist (upper-
                         quarter) but GT has smaller raw bbox-center
                         distance (depth-correction inverting the order)
  TIGHT_CONTEST        — same team, |prob_pick - prob_gt| < 0.05
  OTHER                — anything left over

The probe does NOT mutate DB. Reads positions_json + contacts_json +
GT + team_assignments and runs the scorer in-process.
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import psycopg

# Force scorer ON so get_scorer() returns the singleton
os.environ.setdefault("USE_DYNAMIC_ATTRIBUTION_SCORER", "1")

# Add analysis package to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rallycut.tracking.dynamic_attribution_scorer import (  # noqa: E402
    DynamicAttributionScorer, extract_features, position_from_dict,
)

DB_DSN = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5436/rallycut",
)
TRUSTED_21 = (
    "titi", "toto", "lulu", "wawa", "caco", "cece", "cici", "cuco",
    "gaga", "kaka", "kiki", "juju", "yeye", "keke",
    "gigi", "gugu", "mame", "meme", "mimi", "moma", "mumu",
)

OUT_DIR = Path("reports/attack_residual_2026_05_17")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUT_DIR / "errors.csv"
SUMMARY_PATH = OUT_DIR / "summary.md"


def raw_bbox_dist(pos, ball_x: float, ball_y: float) -> float:
    """Raw bbox-center distance (no upper-quarter offset)."""
    cx = pos.x + pos.width / 2
    cy = pos.y + pos.height / 2
    return math.hypot(cx - ball_x, cy - ball_y)


def find_pipeline_contact(contacts: list[dict], gt_frame: int) -> dict | None:
    """Closest pipeline contact (any team) to the GT frame, ±15 frames."""
    best = None
    best_delta = 16
    for c in contacts:
        f = int(c.get("frame", -10_000))
        d = abs(f - gt_frame)
        if d < best_delta:
            best_delta = d
            best = c
    return best


def team_for_tid(team_assignments: dict, tid: int) -> str | None:
    """Map a track_id to its team label ('A'|'B') if known."""
    if not team_assignments:
        return None
    s = str(tid)
    if s in team_assignments:
        return str(team_assignments[s])
    if tid in team_assignments:
        return str(team_assignments[tid])
    return None


def classify(
    *,
    gt_tid: int,
    pipeline_pick: int,
    candidates_features: dict,   # tid → CandidateFeatures
    candidates_raw: dict,        # tid → raw_bbox_dist
    candidates_in_set: set[int],
    team_assignments: dict,
    probs_by_tid: dict[int, float],
    pose_arms_raised: dict[int, float],
    pose_wrist_to_ball: dict[int, float],
) -> str:
    if gt_tid not in candidates_in_set:
        return "GT_MISSING"
    t_pick = team_for_tid(team_assignments, pipeline_pick)
    t_gt = team_for_tid(team_assignments, gt_tid)
    if t_pick is not None and t_gt is not None and t_pick != t_gt:
        return "CROSS_TEAM"

    # same team (or unknown team — treat as same)
    arm_pick = pose_arms_raised.get(pipeline_pick, 0.0)
    arm_gt = pose_arms_raised.get(gt_tid, 0.0)
    wb_pick = pose_wrist_to_ball.get(pipeline_pick, 1.0)
    wb_gt = pose_wrist_to_ball.get(gt_tid, 1.0)
    if arm_pick >= 0.5 and arm_gt >= 0.5 and abs(wb_pick - wb_gt) < 0.04:
        return "POSE_BLIND_SPOT"

    cf_pick = candidates_features.get(pipeline_pick)
    cf_gt = candidates_features.get(gt_tid)
    raw_pick = candidates_raw.get(pipeline_pick, math.inf)
    raw_gt = candidates_raw.get(gt_tid, math.inf)
    if (cf_pick is not None and cf_gt is not None
            and cf_pick.bbox_dist < cf_gt.bbox_dist
            and raw_gt < raw_pick):
        return "DEPTH_OVERWEIGHT"

    p_pick = probs_by_tid.get(pipeline_pick, 0.0)
    p_gt = probs_by_tid.get(gt_tid, 0.0)
    if abs(p_pick - p_gt) < 0.05:
        return "TIGHT_CONTEST"

    return "OTHER"


def main() -> int:
    scorer = DynamicAttributionScorer()
    if not scorer.is_available:
        print(f"No scorer manifest at {scorer.models_dir}", file=sys.stderr)
        return 1

    rows_written: list[dict] = []
    counter: Counter[str] = Counter()
    total = 0
    correct = 0
    skipped = 0
    no_attack_contact = 0

    with psycopg.connect(DB_DSN) as conn:
        # Pull GT + rally + video + tracking data in one query
        cur = conn.execute("""
            SELECT v.name, r.id, r."order", rg.frame, rg.resolved_track_id,
                   pt.positions_json, pt.ball_positions_json,
                   pt.contacts_json, pt.actions_json
            FROM rally_action_ground_truth rg
            JOIN rallies r ON rg.rally_id = r.id
            JOIN videos v ON r.video_id = v.id
            JOIN player_tracks pt ON pt.rally_id = r.id
            WHERE v.name = ANY(%s)
              AND rg.action = 'ATTACK'
              AND rg.resolved_track_id IS NOT NULL
            ORDER BY v.name, r."order", rg.frame
        """, [list(TRUSTED_21)])
        gt_rows = cur.fetchall()

    print(f"Loaded {len(gt_rows)} GT ATTACK rows", flush=True)

    for (video, rally_id, rally_order, gt_frame, gt_tid,
         positions_json, ball_positions_json, contacts_json, actions_json) in gt_rows:
        total += 1
        if not positions_json or not ball_positions_json:
            skipped += 1
            continue
        positions = [position_from_dict(d) for d in positions_json]

        # Find the matching pipeline contact (closest frame, prefer ATTACK type)
        contacts_dict = contacts_json if isinstance(contacts_json, dict) else (
            json.loads(contacts_json) if isinstance(contacts_json, str) else {}
        )
        all_contacts = contacts_dict.get("contacts") or []
        attack_contacts = [c for c in all_contacts if c.get("actionType") == "ATTACK"]
        # Prefer ATTACK-typed contacts within ±15 frames; else fall back to any
        contact = find_pipeline_contact(attack_contacts, gt_frame)
        if contact is None or abs(int(contact["frame"]) - gt_frame) > 15:
            contact = find_pipeline_contact(all_contacts, gt_frame)
        if contact is None:
            no_attack_contact += 1
            continue

        cf = int(contact["frame"])
        ball_x = float(contact.get("ballX", 0.0))
        ball_y = float(contact.get("ballY", 0.0))
        # playerCandidates is a list of [track_id, distance] pairs
        cand_tids = [int(pc[0]) for pc in contact.get("playerCandidates", [])]
        if not cand_tids:
            skipped += 1
            continue

        # post-ball for wrist alignment
        post_bx = post_by = None
        for bp in ball_positions_json:
            if int(bp.get("frameNumber", -1)) == cf + 5:
                post_bx = float(bp.get("x", ball_x))
                post_by = float(bp.get("y", ball_y))
                break

        # Look up prev action's player_track_id (for same_as_prev)
        actions_dict = actions_json if isinstance(actions_json, dict) else (
            json.loads(actions_json) if isinstance(actions_json, str) else {}
        )
        actions = actions_dict.get("actions") or []
        prev_tid = -1
        for a in actions:
            if int(a.get("frame", 10**9)) < cf:
                prev_tid = int(a.get("playerTrackId", -1))
            else:
                break

        # Build candidate features
        candidates_features = {}
        candidates_raw = {}
        for tid in cand_tids:
            feats = extract_features(
                positions, tid, cf, ball_x, ball_y,
                prev_action_tid=prev_tid,
                post_ball_x=post_bx, post_ball_y=post_by,
            )
            if feats is not None:
                candidates_features[tid] = feats
                p_at = next((p for p in positions if p.track_id == tid
                             and abs(p.frame_number - cf) <= 5), None)
                if p_at is not None:
                    candidates_raw[tid] = raw_bbox_dist(p_at, ball_x, ball_y)

        if not candidates_features:
            skipped += 1
            continue

        ordered = list(candidates_features.values())
        result = scorer.pick_with_probs("ATTACK", ordered)
        if result is None:
            skipped += 1
            continue
        pipeline_pick, probs = result
        probs_by_tid = {ordered[i].track_id: probs[i] for i in range(len(ordered))}

        if pipeline_pick == gt_tid:
            correct += 1
            continue

        # Wrong → classify
        # team_assignments lookup
        team_assignments = actions_dict.get("teamAssignments") or {}
        pose_arms = {tid: cf_.arms_raised for tid, cf_ in candidates_features.items()}
        pose_wb = {tid: cf_.wrist_to_ball_min for tid, cf_ in candidates_features.items()}

        category = classify(
            gt_tid=gt_tid,
            pipeline_pick=pipeline_pick,
            candidates_features=candidates_features,
            candidates_raw=candidates_raw,
            candidates_in_set=set(candidates_features.keys()),
            team_assignments=team_assignments,
            probs_by_tid=probs_by_tid,
            pose_arms_raised=pose_arms,
            pose_wrist_to_ball=pose_wb,
        )
        counter[category] += 1

        cf_pick = candidates_features.get(pipeline_pick)
        cf_gt = candidates_features.get(gt_tid)
        rows_written.append({
            "video": video,
            "rally_order": rally_order,
            "rally_id": rally_id,
            "gt_frame": gt_frame,
            "pipeline_frame": cf,
            "gt_tid": gt_tid,
            "pick_tid": pipeline_pick,
            "prob_pick": round(probs_by_tid.get(pipeline_pick, 0.0), 4),
            "prob_gt": round(probs_by_tid.get(gt_tid, 0.0), 4),
            "prob_gap": round(probs_by_tid.get(pipeline_pick, 0.0)
                              - probs_by_tid.get(gt_tid, 0.0), 4),
            "pick_team": team_for_tid(team_assignments, pipeline_pick) or "?",
            "gt_team": team_for_tid(team_assignments, gt_tid) or "?",
            "pick_bbox_dist": round(cf_pick.bbox_dist, 4) if cf_pick else "",
            "gt_bbox_dist": round(cf_gt.bbox_dist, 4) if cf_gt else "",
            "pick_raw_dist": round(candidates_raw.get(pipeline_pick, 0.0), 4),
            "gt_raw_dist": round(candidates_raw.get(gt_tid, 0.0), 4),
            "pick_arms_raised": pose_arms.get(pipeline_pick, 0.0),
            "gt_arms_raised": pose_arms.get(gt_tid, 0.0),
            "pick_wrist_to_ball": round(pose_wb.get(pipeline_pick, 1.0), 4),
            "gt_wrist_to_ball": round(pose_wb.get(gt_tid, 1.0), 4),
            "category": category,
        })

    # Write CSV
    if rows_written:
        fieldnames = list(rows_written[0].keys())
        with CSV_PATH.open("w") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows_written)

    errors = total - correct - skipped - no_attack_contact
    print(flush=True)
    print(f"TOTAL GT ATTACK rows: {total}", flush=True)
    print(f"  correct          : {correct}", flush=True)
    print(f"  wrong            : {errors}", flush=True)
    print(f"  skipped (no data): {skipped}", flush=True)
    print(f"  no matching contact within ±15f: {no_attack_contact}", flush=True)
    print(f"  attribution accy : {correct / max(1, correct + errors) * 100:.1f}%", flush=True)
    print(flush=True)
    print("Failure-mode breakdown (of wrong picks):", flush=True)
    for cat, n in counter.most_common():
        share = n / max(1, errors) * 100
        print(f"  {cat:20s} {n:4d}  ({share:5.1f}%)", flush=True)
    print(flush=True)
    print(f"Per-error CSV: {CSV_PATH}", flush=True)

    # Markdown summary
    lines = [
        "# ATTACK residual failure-mode catalog (trusted-21, v2 scorer)",
        "",
        f"- Total GT ATTACK rows: {total}",
        f"- Correct (scorer picked GT): {correct}",
        f"- Wrong: {errors}",
        f"- Skipped (no positions / no candidates / no contact within ±15f): {skipped + no_attack_contact}",
        f"- Attribution accuracy on classified rows: "
        f"**{correct / max(1, correct + errors) * 100:.1f}%**",
        "",
        "## Failure-mode breakdown (wrong picks)",
        "",
        "| Category | Count | Share |",
        "|---|---:|---:|",
    ]
    for cat, n in counter.most_common():
        share = n / max(1, errors) * 100
        lines.append(f"| {cat} | {n} | {share:.1f}% |")
    lines += [
        "",
        "## Category definitions",
        "",
        "- **GT_MISSING** — GT resolved_track_id not present in `contact.player_candidates`."
        " Upstream tracker / candidate-collection bug; no per-action scorer can fix.",
        "- **CROSS_TEAM** — pick and GT on different teams (team_assignments). Failure"
        " of team-chain repair / ratio-cap relaxation.",
        "- **POSE_BLIND_SPOT** — same team, both contestants `arms_raised=1` AND"
        " `|wrist_to_ball_pick - wrist_to_ball_gt| < 0.04`. Pose features can't"
        " disambiguate; needs visual scene reasoning (DINOv2/V-JEPA/VLM).",
        "- **DEPTH_OVERWEIGHT** — same team, pick has smaller `bbox_dist` (upper-quarter)"
        " but GT has smaller raw bbox-center distance. Depth-correction overweights"
        " near-side blocker. Try: cap depth-scale, or use raw distance as a feature.",
        "- **TIGHT_CONTEST** — same team, `|prob_pick - prob_gt| < 0.05`. Scorer is"
        " genuinely uncertain; more GT or a new signal class needed.",
        "- **OTHER** — everything else (rare; investigate individually).",
        "",
        f"- Per-error CSV: `{CSV_PATH.relative_to(Path('.'))}`",
    ]
    SUMMARY_PATH.write_text("\n".join(lines) + "\n")
    print(f"Summary: {SUMMARY_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
