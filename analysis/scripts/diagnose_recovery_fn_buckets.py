"""Per-FN root-cause analysis for coherence-driven contact recovery.

For each GT FN in the panel, classify why the recovery missed it. Used
to decide whether to invest in (a) widening gap windows, (b) loosening
candidate-generator thresholds, or (c) accepting the v1 limit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from dataclasses import dataclass

from rallycut.evaluation.tracking.db import get_connection
from rallycut.tracking.contact_recovery import (
    GATE_BALL_CONF_MIN,
    GATE_DUPLICATE_FRAME_WINDOW,
    GATE_GBM_MIN,
    GATE_SEQ_TAU,
    Gap,
    audit_violation_count_for_actions,
    derive_gaps_from_actions,
    load_rally_inputs,
    _seq_peak_nonbg,
)
from rallycut.tracking.contact_detector import (
    ContactDetectionConfig,
    detect_contacts,
)
from rallycut.tracking.sequence_action_runtime import get_sequence_probs
import numpy as np

PANEL_IDS = [
    "b5fb0594-d64f-4a0d-bad9-de8fc36414d0",
    "7d77980f-3006-40e0-adc0-db491a5bb659",
    "854bb250-3e91-47d2-944d-f62413e3cf45",
    "5c756c41-1cc1-4486-a95c-97398912cfbe",
    "073cb11b-c7ba-4fac-8cc9-b032b3152ad6",
]
GT_PATH = Path("training_datasets/beach_v11/action_ground_truth.json")
FRAME_TOLERANCE = 15


@dataclass
class FNDiagnosis:
    """Per-FN classification result."""
    video_name: str
    rally_id: str
    fn_frame: int
    gt_action: str
    gt_player: int
    bucket: str
    detail: str


def _match_actions(
    gt: list[dict[str, Any]], pred: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (matched_pairs, fn_list) by greedy frame-tolerance matching.

    Args:
        gt: Ground truth actions.
        pred: Predicted actions.

    Returns:
        Tuple of (matched_gt_actions, fn_actions) where matched_gt_actions are
        those matched to predictions, and fn_actions are unmatched GTs.
    """
    used: set[int] = set()
    matched_gt = []
    fn_list = []

    for g in gt:
        gf = int(g.get("frame", 0))
        best_i = -1
        best_d = FRAME_TOLERANCE + 1
        for i, p in enumerate(pred):
            if i in used:
                continue
            d = abs(int(p.get("frame", 0)) - gf)
            if d < best_d:
                best_d = d
                best_i = i
        if best_i >= 0 and best_d <= FRAME_TOLERANCE:
            used.add(best_i)
            matched_gt.append(g)
        else:
            fn_list.append(g)

    return matched_gt, fn_list


def classify_fn(
    *,
    fn_frame: int,
    fn_player: int,
    fn_action: str,
    gaps: list[Gap],
    contacts: list[Any],
    seq_probs: np.ndarray,
    team_assignments_str: dict[str, str],
    ball_conf_by_frame: dict[int, float],
    existing_action_frames: list[int],
) -> tuple[str, str]:
    """Classify why the FN wasn't recovered.

    Returns:
        (bucket, detail) tuple.
    """
    # 1. Gap check
    gap = next((g for g in gaps if g.lo <= fn_frame <= g.hi), None)
    if gap is None:
        return "out_of_gap", f"no gap covers frame {fn_frame}"

    # 2. Find nearby candidates within ±FRAME_TOLERANCE of FN
    nearby = [c for c in contacts if abs(c.frame - fn_frame) <= FRAME_TOLERANCE]
    if not nearby:
        return "no_candidate", f"no contact near frame {fn_frame}"

    # Pick nearest
    cand = min(nearby, key=lambda c: abs(c.frame - fn_frame))
    cand_frame = cand.frame
    dist_to_fn = abs(cand_frame - fn_frame)

    # 3. Duplicate check
    if any(abs(cand_frame - f) <= GATE_DUPLICATE_FRAME_WINDOW for f in existing_action_frames):
        return "duplicate", f"cand@{cand_frame} within ±{GATE_DUPLICATE_FRAME_WINDOW} of existing action"

    # 4. Team check
    if gap.expected_team is not None:
        tlabel = team_assignments_str.get(str(cand.player_track_id))
        if tlabel != gap.expected_team:
            return "team_mismatch", f"cand@{cand_frame} is team {tlabel or 'unknown'}, expected {gap.expected_team}"

    # 5. Ball confidence
    bconf = ball_conf_by_frame.get(cand_frame, 0.0)
    if bconf < GATE_BALL_CONF_MIN:
        return "gate_ball_conf_low", f"cand@{cand_frame} ball conf {bconf:.2f} < {GATE_BALL_CONF_MIN}"

    # 6. GBM confidence
    if cand.confidence < GATE_GBM_MIN:
        return "gate_gbm_low", f"cand@{cand_frame} gbm {cand.confidence:.3f} < {GATE_GBM_MIN}"

    # 7. Sequence peak
    seq = _seq_peak_nonbg(seq_probs, cand_frame)
    if seq < GATE_SEQ_TAU:
        return "gate_seq_low", f"cand@{cand_frame} seq peak {seq:.3f} < {GATE_SEQ_TAU}"

    # 8. All gates passed — but does insertion actually reduce violations?
    # For this diagnostic, we assume it would if all gates pass (the audit-after
    # check is expensive and most FNs die at earlier gates).
    return "recovered", f"cand@{cand_frame} would pass all gates; gbm={cand.confidence:.3f} seq={seq:.3f}"


def main() -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, content_hash, name FROM videos WHERE id = ANY(%s)",
                [PANEL_IDS],
            )
            meta = {r[0]: (r[1], r[2]) for r in cur.fetchall()}

    gt_data = json.loads(GT_PATH.read_text())
    hash_to_id = {h: vid for vid, (h, _) in meta.items()}
    panel_gt = [
        (hash_to_id[r["video_content_hash"]], r)
        for r in gt_data["rallies"]
        if r["video_content_hash"] in hash_to_id
    ]

    # Print header
    print(
        f"{'video':<6} {'rally':<8} {'FN_frame':>8} {'GT_action':<10} "
        f"{'bucket':<20} {'detail':<60}"
    )
    print("-" * 120)

    diagnoses: list[FNDiagnosis] = []

    with get_connection() as conn:
        for vid, gt_rally in panel_gt:
            name = meta[vid][1]
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT r.id FROM rallies r
                       WHERE r.video_id = %s AND r.start_ms = %s""",
                    [vid, gt_rally["rally_start_ms"]],
                )
                row = cur.fetchone()
            if row is None:
                continue
            rid = str(row[0])

            try:
                inputs = load_rally_inputs(rid)
            except ValueError:
                continue

            base_actions = list(inputs.actions_json.get("actions") or [])
            gt_actions = gt_rally["action_ground_truth_json"]

            # Match actions to find FNs
            _, fn_actions = _match_actions(gt_actions, base_actions)

            if not fn_actions:
                continue

            # Load inputs for recovery diagnostics
            gaps = derive_gaps_from_actions(
                actions=base_actions,
                team_assignments=inputs.team_assignments_str,
                rally_start_frame=inputs.rally_start_frame,
            )

            # Get sequence probs
            seq_probs = get_sequence_probs(
                inputs.ball_positions, inputs.player_positions,
                inputs.court_split_y, inputs.frame_count,
                inputs.team_assignments_int, calibrator=None,
            )
            if seq_probs is None:
                # Can't diagnose without seq probs
                for fn in fn_actions:
                    diagnoses.append(FNDiagnosis(
                        video_name=name, rally_id=rid,
                        fn_frame=int(fn.get("frame", 0)),
                        gt_action=str(fn.get("action", "unknown")),
                        gt_player=int(fn.get("playerTrackId", -1)),
                        bucket="seq_probs_unavailable",
                        detail="MS-TCN++ probs not available",
                    ))
                continue

            # Run contact detection with rescue enabled
            contact_seq = detect_contacts(
                ball_positions=inputs.ball_positions,
                player_positions=inputs.player_positions,
                config=ContactDetectionConfig(classifier_threshold_override=GATE_GBM_MIN),
                net_y=inputs.court_split_y,
                frame_count=inputs.frame_count or None,
                team_assignments=inputs.team_assignments_int,
                court_calibrator=None,
                sequence_probs=seq_probs,
                enable_rescue=True,
                primary_track_ids=list(inputs.primary_track_ids) or None,
            )

            ball_conf_by_frame = {
                bp.frame_number: bp.confidence for bp in inputs.ball_positions
            }
            existing_action_frames = [int(a.get("frame", 0)) for a in base_actions]

            # Classify each FN
            for fn in fn_actions:
                fn_frame = int(fn.get("frame", 0))
                fn_action = str(fn.get("action", "unknown"))
                fn_player = int(fn.get("playerTrackId", -1))

                bucket, detail = classify_fn(
                    fn_frame=fn_frame,
                    fn_player=fn_player,
                    fn_action=fn_action,
                    gaps=gaps,
                    contacts=contact_seq.contacts,
                    seq_probs=seq_probs,
                    team_assignments_str=inputs.team_assignments_str,
                    ball_conf_by_frame=ball_conf_by_frame,
                    existing_action_frames=existing_action_frames,
                )

                diagnoses.append(FNDiagnosis(
                    video_name=name, rally_id=rid,
                    fn_frame=fn_frame,
                    gt_action=fn_action,
                    gt_player=fn_player,
                    bucket=bucket,
                    detail=detail,
                ))

                print(
                    f"{name:<6} {rid[:8]:<8} {fn_frame:>8} {fn_action:<10} "
                    f"{bucket:<20} {detail:<60}"
                )

    # Summary
    print("\n" + "=" * 120)
    print(f"SUMMARY ({len(diagnoses)} FNs):\n")

    bucket_counts: dict[str, int] = {}
    for d in diagnoses:
        bucket_counts[d.bucket] = bucket_counts.get(d.bucket, 0) + 1

    for bucket in sorted(bucket_counts.keys()):
        count = bucket_counts[bucket]
        pct = 100.0 * count / len(diagnoses) if diagnoses else 0.0
        print(f"  {bucket:<25} : {count:>3} ({pct:>5.1f}%)")

    print(f"\nTotal FNs: {len(diagnoses)}")

    # Report any "recovered" cases
    recovered = [d for d in diagnoses if d.bucket == "recovered"]
    if recovered:
        print(f"\n** RECOVERED CASES ({len(recovered)}) **")
        for d in recovered:
            print(f"  {d.video_name} / {d.rally_id[:8]} @ frame {d.fn_frame}")

    # Report per-gap analysis for out_of_gap FNs
    out_of_gap = [d for d in diagnoses if d.bucket == "out_of_gap"]
    if out_of_gap:
        print(f"\n** OUT_OF_GAP ANALYSIS ({len(out_of_gap)}) **")
        # Re-load for detailed per-rally analysis
        with get_connection() as conn:
            for vid, gt_rally in panel_gt:
                name = meta[vid][1]
                with conn.cursor() as cur:
                    cur.execute(
                        """SELECT r.id FROM rallies r
                           WHERE r.video_id = %s AND r.start_ms = %s""",
                        [vid, gt_rally["rally_start_ms"]],
                    )
                    row = cur.fetchone()
                if row is None:
                    continue
                rid = str(row[0])

                rally_fn = [d for d in out_of_gap if d.rally_id == rid]
                if not rally_fn:
                    continue

                try:
                    inputs = load_rally_inputs(rid)
                except ValueError:
                    continue

                base_actions = list(inputs.actions_json.get("actions") or [])
                gaps = derive_gaps_from_actions(
                    actions=base_actions,
                    team_assignments=inputs.team_assignments_str,
                    rally_start_frame=inputs.rally_start_frame,
                )

                pred_action_frames = sorted([int(a.get("frame", 0)) for a in base_actions])

                print(f"\n  {name} / {rid[:8]}:")
                print(f"    Gaps: {[f'({g.rule} {g.lo}-{g.hi})' for g in gaps]}")
                print(f"    Pred frames: {pred_action_frames}")

                for fn_d in rally_fn:
                    print(f"    FN@{fn_d.fn_frame} ({fn_d.gt_action}): {fn_d.detail}")
                    # Suggest gap adjustment
                    if pred_action_frames:
                        before = [f for f in pred_action_frames if f <= fn_d.fn_frame]
                        after = [f for f in pred_action_frames if f > fn_d.fn_frame]
                        if not before:
                            lo = inputs.rally_start_frame
                        else:
                            lo = max(before)
                        if not after:
                            hi = inputs.frame_count - 1
                        else:
                            hi = min(after)
                        print(f"      → Could extend gap to [{lo}, {hi}] to include FN@{fn_d.fn_frame}")


if __name__ == "__main__":
    main()
