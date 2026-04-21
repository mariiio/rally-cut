"""Dedup-winner trace: for each Category 7 FN, identify which candidate won dedup
and its downstream fate. Distinguishes sub-mechanisms H7a-e from Phase 3.

Input:
  analysis/outputs/phase4_category_assignments.jsonl (84 Cat 7 FNs)
  DB: rally data (ball/player positions, GT labels)

Strategy:
- Re-run the production contact_detector on each affected rally with an
  instrumented dedup helper that logs (eliminated_candidate, winner) pairs.
- Join the eliminated candidates against the Cat 7 FN frames to identify which
  winner caused each elimination.
- Cross-reference the winner with final rally_actions to determine if it
  survived to final pred, ended up within 7f of a different GT (matching-steal),
  or was dropped downstream (pre-serve FP / phantom rejection).

Usage:
    cd analysis && uv run python scripts/trace_dedup_winners.py
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rallycut.tracking.action_classifier import classify_rally_actions
from rallycut.tracking.candidate_decoder import TransitionMatrix
from rallycut.tracking.contact_detector import ContactDetectionConfig, detect_contacts
from rallycut.tracking.decoder_runtime import run_decoder_over_rally
from scripts.eval_action_detection import (
    load_rallies_with_action_gt,
)
from scripts.eval_loo_video import (
    RallyPrecomputed,
    _inject_action_classifier,
    _precompute,
    _reset_action_classifier_cache,
    _train_fold,
)

logging.getLogger("rallycut.tracking.action_classifier").setLevel(logging.ERROR)

REPO = Path(__file__).resolve().parent.parent
CAT_PATH = REPO / "outputs" / "phase4_category_assignments.jsonl"
OUT_PATH = REPO / "outputs" / "dedup_winner_trace_2026_04_21.jsonl"

MATCHER_TOL_FRAMES = 7  # 233ms @ 30fps
DEDUP_WINDOW_FRAMES = 16  # > min_peak_distance_frames=12, covers cross-side 4 too


@dataclass
class WinnerTrace:
    rally_id: str
    gt_frame: int
    gt_action: str
    eliminated_frame: int | None
    eliminated_conf: float | None
    eliminated_court_side: str | None
    eliminated_player_tid: int | None
    winner_frame: int | None
    winner_conf: float | None
    winner_court_side: str | None
    winner_player_tid: int | None
    frame_gap: int | None
    same_side: bool | None
    sides_known: bool | None
    effective_min: int | None  # dedup distance that fired (4 vs 12)
    winner_in_final_pred: bool
    winner_in_final_within_gt_tol: bool  # within 7f of THIS GT
    winner_in_final_within_other_gt_tol: bool  # within 7f of ANOTHER GT in rally
    winner_action_in_final: str | None
    notes: str


def _instrument_dedup(cfg: ContactDetectionConfig):
    """Monkey-patch _deduplicate_contacts to log elimination pairs."""
    import rallycut.tracking.contact_detector as cd
    cross_side_min = cd._CROSS_SIDE_MIN_DISTANCE
    both_conf_floor = cd._BOTH_CONFIDENT_FLOOR
    original_dedup = cd._deduplicate_contacts
    eliminations: list[dict] = []

    def instrumented(contacts, min_distance, adaptive=False):
        if not contacts:
            return contacts
        sorted_c = sorted(contacts, key=lambda c: c.confidence, reverse=True)
        result = []
        for contact in sorted_c:
            killer = None
            for existing in result:
                frame_gap = abs(contact.frame - existing.frame)
                if adaptive:
                    sides_known = (
                        contact.court_side in ("near", "far")
                        and existing.court_side in ("near", "far")
                    )
                    if sides_known and contact.court_side != existing.court_side:
                        effective_min = cross_side_min
                    elif (
                        contact.confidence >= both_conf_floor
                        and existing.confidence >= both_conf_floor
                    ):
                        effective_min = cross_side_min
                    else:
                        effective_min = min_distance
                else:
                    sides_known = None
                    effective_min = min_distance
                if frame_gap < effective_min:
                    killer = (existing, frame_gap, sides_known, effective_min)
                    break
            if killer is None:
                result.append(contact)
            else:
                existing, frame_gap, sides_known, effective_min = killer
                eliminations.append({
                    "eliminated_frame": contact.frame,
                    "eliminated_conf": contact.confidence,
                    "eliminated_court_side": contact.court_side,
                    "eliminated_player_tid": contact.player_track_id,
                    "winner_frame": existing.frame,
                    "winner_conf": existing.confidence,
                    "winner_court_side": existing.court_side,
                    "winner_player_tid": existing.player_track_id,
                    "frame_gap": frame_gap,
                    "same_side": (
                        None if sides_known is None or not sides_known
                        else contact.court_side == existing.court_side
                    ),
                    "sides_known": sides_known,
                    "effective_min": effective_min,
                })
        return sorted(result, key=lambda c: c.frame)

    cd._deduplicate_contacts = instrumented
    return eliminations, original_dedup


def _restore_dedup(original):
    import rallycut.tracking.contact_detector as cd
    cd._deduplicate_contacts = original


def main() -> None:
    # Load Cat 7 FNs (primary 7-dedup_kill and 7+4-dedup_kill_with_occlusion)
    cat7_fns = []
    for line in CAT_PATH.open():
        a = json.loads(line)
        if a["primary_category"] in ("7-dedup_kill", "7+4-dedup_kill_with_occlusion"):
            cat7_fns.append(a)
    print(f"Loaded {len(cat7_fns)} Cat 7 FNs to trace.")

    # Group by rally_id for efficient reprocessing
    from collections import defaultdict
    fns_by_rally: dict[str, list[dict]] = defaultdict(list)
    for fn in cat7_fns:
        fns_by_rally[fn["rally_id"]].append(fn)
    print(f"Spanning {len(fns_by_rally)} rallies.")

    # Load GT rallies + precompute
    cfg = ContactDetectionConfig()
    transitions = TransitionMatrix.default()
    print("Loading + precomputing rallies...")
    all_rallies = load_rallies_with_action_gt()
    target_rally_ids = set(fns_by_rally.keys())
    precomputed = {}
    for i, r in enumerate(all_rallies):
        if r.rally_id not in target_rally_ids:
            continue
        pre = _precompute(r, cfg)
        if pre is not None:
            precomputed[r.rally_id] = pre
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(all_rallies)}]")
    print(f"Precomputed {len(precomputed)} target rallies.")

    # Group target rallies by video for LOO fold training
    video_ids = sorted({p.rally.video_id for p in precomputed.values()})
    print(f"Target videos: {len(video_ids)}")

    # For each video, train fold and process each rally
    traces: list[WinnerTrace] = []
    all_precomputed = []
    for r in all_rallies:
        pre = _precompute(r, cfg)
        if pre is not None:
            all_precomputed.append(pre)
    print(f"Total precomputed for training: {len(all_precomputed)}")

    by_video: dict[str, list[RallyPrecomputed]] = defaultdict(list)
    for p in all_precomputed:
        by_video[p.rally.video_id].append(p)

    for v_idx, vid in enumerate(video_ids, 1):
        held = [p for p in precomputed.values() if p.rally.video_id == vid]
        train = [p for v, rs in by_video.items() if v != vid for p in rs]
        print(f"[{v_idx}/{len(video_ids)}] video {vid[:8]} ({len(held)} target rallies)")
        contact_clf, action_clf = _train_fold(train, threshold=0.30)
        _inject_action_classifier(action_clf if action_clf.is_trained else None)
        try:
            for pre in held:
                rally = pre.rally
                rally_fns = fns_by_rally.get(rally.rally_id, [])
                if not rally_fns:
                    continue

                # Instrument dedup and run detect_contacts
                eliminations, original_dedup = _instrument_dedup(cfg)
                try:
                    contact_seq = detect_contacts(
                        ball_positions=pre.ball_positions,
                        player_positions=pre.player_positions,
                        config=cfg,
                        net_y=rally.court_split_y,
                        frame_count=rally.frame_count or None,
                        classifier=contact_clf,
                        use_classifier=True,
                        sequence_probs=pre.sequence_probs,
                    )

                    decoder_contacts = run_decoder_over_rally(
                        ball_positions=pre.ball_positions,
                        player_positions=pre.player_positions,
                        sequence_probs=pre.sequence_probs,
                        classifier=contact_clf,
                        contact_config=cfg,
                        gt_frames=[gt.frame for gt in rally.gt_labels],
                        transitions=transitions,
                        skip_penalty=1.0,
                    )

                    rally_actions = classify_rally_actions(
                        contact_seq,
                        rally_id=rally.rally_id,
                        use_classifier=action_clf.is_trained,
                        sequence_probs=pre.sequence_probs,
                        decoder_contacts=decoder_contacts,
                    )
                finally:
                    _restore_dedup(original_dedup)

                pred = [a.to_dict() for a in rally_actions.actions]
                real_pred = [a for a in pred if not a.get("isSynthetic")]
                pred_frames = {a.get("frame") for a in real_pred}

                # For each Cat 7 FN in this rally, find the elimination whose
                # eliminated_frame is closest to the FN's GT frame
                for fn in rally_fns:
                    gt_frame = fn["gt_frame"]
                    # Best candidate: elimination within DEDUP_WINDOW_FRAMES of GT
                    best = None
                    best_dist = DEDUP_WINDOW_FRAMES + 1
                    for e in eliminations:
                        d = abs(e["eliminated_frame"] - gt_frame)
                        if d < best_dist:
                            best_dist = d
                            best = e
                    if best is None:
                        traces.append(WinnerTrace(
                            rally_id=rally.rally_id,
                            gt_frame=gt_frame,
                            gt_action=fn["gt_action"],
                            eliminated_frame=None, eliminated_conf=None,
                            eliminated_court_side=None, eliminated_player_tid=None,
                            winner_frame=None, winner_conf=None,
                            winner_court_side=None, winner_player_tid=None,
                            frame_gap=None, same_side=None, sides_known=None,
                            effective_min=None,
                            winner_in_final_pred=False,
                            winner_in_final_within_gt_tol=False,
                            winner_in_final_within_other_gt_tol=False,
                            winner_action_in_final=None,
                            notes="no elimination found near GT in instrumented dedup",
                        ))
                        continue

                    winner_frame = best["winner_frame"]
                    # Winner fate: is its frame in the final pred?
                    winner_in_final = winner_frame in pred_frames
                    winner_in_gt_tol = any(
                        abs(winner_frame - gt_frame) <= MATCHER_TOL_FRAMES
                        and abs(a.get("frame") - winner_frame) <= 0
                        for a in real_pred
                    )
                    # Is winner within tolerance of ANOTHER GT (matching-steal)?
                    other_gt_frames = [
                        gt.frame for gt in rally.gt_labels
                        if gt.frame != gt_frame
                    ]
                    winner_in_other_gt = any(
                        abs(winner_frame - ogf) <= MATCHER_TOL_FRAMES
                        for ogf in other_gt_frames
                    )
                    winner_action = None
                    for a in real_pred:
                        if a.get("frame") == winner_frame:
                            winner_action = a.get("action")
                            break

                    notes_parts = []
                    if not winner_in_final:
                        notes_parts.append("winner dropped downstream (stage 8+)")
                    if winner_in_other_gt and not winner_in_gt_tol:
                        notes_parts.append("winner matches another GT")
                    if best.get("sides_known") is False or best.get("sides_known") is None:
                        notes_parts.append("court_side unknown on >=1 contact")

                    traces.append(WinnerTrace(
                        rally_id=rally.rally_id,
                        gt_frame=gt_frame,
                        gt_action=fn["gt_action"],
                        eliminated_frame=best["eliminated_frame"],
                        eliminated_conf=best["eliminated_conf"],
                        eliminated_court_side=best["eliminated_court_side"],
                        eliminated_player_tid=best["eliminated_player_tid"],
                        winner_frame=winner_frame,
                        winner_conf=best["winner_conf"],
                        winner_court_side=best["winner_court_side"],
                        winner_player_tid=best["winner_player_tid"],
                        frame_gap=best["frame_gap"],
                        same_side=best["same_side"],
                        sides_known=best["sides_known"],
                        effective_min=best["effective_min"],
                        winner_in_final_pred=winner_in_final,
                        winner_in_final_within_gt_tol=winner_in_gt_tol,
                        winner_in_final_within_other_gt_tol=winner_in_other_gt,
                        winner_action_in_final=winner_action,
                        notes=" | ".join(notes_parts) if notes_parts else "",
                    ))
                print(f"    rally {rally.rally_id[:8]}: {len(rally_fns)} FNs, "
                      f"{len(eliminations)} total eliminations in rally")
        finally:
            _reset_action_classifier_cache()

    # Write output
    with OUT_PATH.open("w") as f:
        for t in traces:
            f.write(json.dumps(asdict(t), default=str) + "\n")

    print(f"\nWrote {len(traces)} traces to {OUT_PATH.relative_to(REPO)}")

    # Quick summary
    no_elim = sum(1 for t in traces if t.eliminated_frame is None)
    same_side_true = sum(1 for t in traces if t.same_side is True)
    same_side_false = sum(1 for t in traces if t.same_side is False)
    sides_unknown = sum(1 for t in traces if t.sides_known is False)
    winner_dropped = sum(1 for t in traces if t.eliminated_frame is not None and not t.winner_in_final_pred)
    winner_matches_other_gt = sum(1 for t in traces if t.winner_in_final_within_other_gt_tol and not t.winner_in_final_within_gt_tol)
    eff_min_12 = sum(1 for t in traces if t.effective_min == 12)
    eff_min_4 = sum(1 for t in traces if t.effective_min == 4)

    print(f"\n=== Summary (n={len(traces)}) ===")
    print(f"No elimination found near GT:     {no_elim}")
    print(f"Same-side dedup (effective_min=12): {eff_min_12}")
    print(f"Cross-side dedup (effective_min=4): {eff_min_4}")
    print(f"  among those, same_side=True:     {same_side_true}")
    print(f"  among those, same_side=False:    {same_side_false}")
    print(f"  sides NOT both known:            {sides_unknown}")
    print(f"Winner dropped later (not in final pred): {winner_dropped}")
    print(f"Winner matches a DIFFERENT GT (matching-steal pattern): {winner_matches_other_gt}")


if __name__ == "__main__":
    main()
