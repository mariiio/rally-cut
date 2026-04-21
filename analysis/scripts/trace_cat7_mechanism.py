"""True Cat 7 mechanism trace — instrument `detect_contacts` INTERNALS.

Phase 5.5 follow-up after `trace_dedup_winners.py` showed 82/84 Cat 7 FNs have
no dedup elimination firing anywhere in the rally. That means the attribution's
`dedup_survived=False` label captures ANY reason a classifier-acceptable
candidate doesn't reach the final `contact_seq.contacts` within tolerance of
GT — not specifically dedup.

This script instruments the INTERNAL candidate loop of `detect_contacts` to
capture, for each Cat 7 FN, the exact drop-out point:

- Was a candidate generated (in the refined+merged list) within ±7f of GT?
- If yes, did it pass the warmup/velocity/ball-position gates?
- If yes, was the classifier's predict result is_validated=True (gbm>=threshold)?
- If yes, did the appended Contact end up in the post-dedup final list?
- Was the candidate shifted to a far frame by trajectory refinement (Step 6b)?
- Was it shifted by proximity refinement (Step 6c)?

Output: per-FN label of the true mechanism.

Usage:
    cd analysis && PYTHONUNBUFFERED=1 uv run python -u scripts/trace_cat7_mechanism.py
"""
from __future__ import annotations

import json
import logging
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]

import rallycut.tracking.contact_detector as cd  # noqa: E402
from rallycut.tracking.contact_detector import (  # noqa: E402
    ContactDetectionConfig,
    detect_contacts,
)
from scripts.eval_action_detection import load_rallies_with_action_gt  # noqa: E402
from scripts.eval_loo_video import (  # noqa: E402
    RallyPrecomputed,
    _precompute,
    _reset_action_classifier_cache,
    _train_fold,
)

logging.getLogger("rallycut.tracking.action_classifier").setLevel(logging.ERROR)

REPO = Path(__file__).resolve().parent.parent
CAT_PATH = REPO / "outputs" / "phase4_category_assignments.jsonl"
OUT_PATH = REPO / "outputs" / "cat7_mechanism_trace_2026_04_21.jsonl"

TOL = 7  # matcher tolerance


@dataclass
class Mechanism:
    rally_id: str
    gt_frame: int
    gt_action: str
    # Candidate pipeline state
    candidate_in_window: bool          # any merged-refined candidate within ±7 of GT
    candidate_frames_in_window: list  # list of (frame, used_in_loop)
    # Per-candidate loop state
    loop_skipped_warmup: bool
    loop_skipped_post_rally: bool
    loop_skipped_no_ball: bool
    loop_skipped_low_velocity: bool
    loop_gbm_rejected: bool            # all candidates in window got is_validated=False
    best_gbm_in_window: float
    loop_accepted_count: int           # how many candidates in the loop produced a Contact appended
    # Final output state
    contact_in_final_seq: bool         # contacts list has entry within ±7 of GT
    appended_frames_in_window: list    # frames appended in the loop (before dedup) within ±7
    dedup_dropped_appended: bool       # appended but not in final contacts
    # Refinement diagnostics
    refinement_shifted_out: bool       # orig cand within ±7 but refined cand outside ±7
    refinement_shift_amount: int       # frames shifted
    # Final mechanism label
    mechanism: str


def _find_nearest(frames: list[int], target: int) -> tuple[int, int] | None:
    if not frames:
        return None
    return min(((f, abs(f - target)) for f in frames), key=lambda x: x[1])


def main() -> None:
    cat7_fns = []
    for line in CAT_PATH.open():
        a = json.loads(line)
        if a["primary_category"] in ("7-dedup_kill", "7+4-dedup_kill_with_occlusion"):
            cat7_fns.append(a)
    print(f"Loaded {len(cat7_fns)} Cat 7 FNs.", flush=True)

    fns_by_rally: dict[str, list[dict]] = defaultdict(list)
    for fn in cat7_fns:
        fns_by_rally[fn["rally_id"]].append(fn)
    print(f"Spanning {len(fns_by_rally)} rallies.", flush=True)

    cfg = ContactDetectionConfig()
    print("Loading + precomputing all rallies...", flush=True)
    all_rallies = load_rallies_with_action_gt()
    target_ids = set(fns_by_rally.keys())

    all_precomputed = []
    for i, r in enumerate(all_rallies):
        pre = _precompute(r, cfg)
        if pre is not None:
            all_precomputed.append(pre)
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(all_rallies)}]", flush=True)
    target_pre = {p.rally.rally_id: p for p in all_precomputed
                  if p.rally.rally_id in target_ids}
    by_video: dict[str, list[RallyPrecomputed]] = defaultdict(list)
    for p in all_precomputed:
        by_video[p.rally.video_id].append(p)
    video_ids = sorted({p.rally.video_id for p in target_pre.values()})
    print(f"Target videos: {len(video_ids)}", flush=True)

    results: list[Mechanism] = []

    for v_idx, vid in enumerate(video_ids, 1):
        held = [p for p in target_pre.values() if p.rally.video_id == vid]
        train = [p for v, rs in by_video.items() if v != vid for p in rs]
        print(f"[{v_idx}/{len(video_ids)}] video {vid[:8]} ({len(held)} target rallies)", flush=True)
        contact_clf, _ = _train_fold(train, threshold=0.30)
        if not contact_clf.is_trained:
            continue

        for pre in held:
            rally = pre.rally
            rally_fns = fns_by_rally.get(rally.rally_id, [])
            if not rally_fns:
                continue

            # Patch classify loop to log what was appended
            appended_frames: list[tuple[int, float]] = []  # (frame, confidence) per Contact appended
            final_contact_frames: list[int] = []
            # Capture pre-refinement candidate frames by patching _refine_candidates_to_trajectory_peak
            pre_refine_cands: list[int] = []
            post_refine_cands: list[int] = []
            proximity_added: list[int] = []

            orig_refine = cd._refine_candidates_to_trajectory_peak
            orig_find_prox = cd._find_proximity_frame

            def refine_wrapper(candidate_frames, *a, **kw):
                pre_refine_cands.extend(candidate_frames)
                out = orig_refine(candidate_frames, *a, **kw)
                post_refine_cands.extend(out)
                return out

            def prox_wrapper(frame, *a, **kw):
                out = orig_find_prox(frame, *a, **kw)
                if out is not None and out != frame:
                    proximity_added.append(out)
                return out

            cd._refine_candidates_to_trajectory_peak = refine_wrapper
            cd._find_proximity_frame = prox_wrapper

            # Patch Contact.__init__ to log appended frames with confidence
            from rallycut.tracking.contact_detector import Contact as _Contact
            orig_init = _Contact.__init__

            def patched_init(self, *args, **kwargs):
                orig_init(self, *args, **kwargs)
                appended_frames.append((self.frame, self.confidence))
            _Contact.__init__ = patched_init

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
            finally:
                cd._refine_candidates_to_trajectory_peak = orig_refine
                cd._find_proximity_frame = orig_find_prox
                _Contact.__init__ = orig_init

            final_contact_frames = [c.frame for c in contact_seq.contacts]

            # Now analyze each Cat 7 FN in this rally
            for fn in rally_fns:
                gt_frame = fn["gt_frame"]

                # Candidate window check — post-refinement merged candidate list
                all_post = sorted(set(post_refine_cands + proximity_added))
                in_window_post = [f for f in all_post if abs(f - gt_frame) <= TOL]
                candidate_in_window = bool(in_window_post)

                # Check whether any candidate in window was appended (passed classifier)
                appended_in_window = [(f, c) for f, c in appended_frames if abs(f - gt_frame) <= TOL]
                loop_accepted_count = len(appended_in_window)
                best_gbm = max((c for _, c in appended_in_window), default=-1.0)

                # Is it in final contacts?
                contact_in_final = any(abs(f - gt_frame) <= TOL for f in final_contact_frames)

                # Refinement shift: was there a pre-refinement cand within ±7 that got shifted out?
                in_window_pre = [f for f in pre_refine_cands if abs(f - gt_frame) <= TOL]
                refinement_shifted = False
                shift_amount = 0
                if in_window_pre and not in_window_post:
                    refinement_shifted = True
                    # find how far the closest pre-refinement cand was shifted
                    pre_closest = min(in_window_pre, key=lambda f: abs(f - gt_frame))
                    # find its corresponding post-refinement frame (same index position in lists)
                    # This is approximate — refinement preserves list order
                    try:
                        idx = pre_refine_cands.index(pre_closest)
                        if idx < len(post_refine_cands):
                            post_f = post_refine_cands[idx]
                            shift_amount = post_f - pre_closest
                    except ValueError:
                        pass

                # Mechanism label
                if contact_in_final:
                    mechanism = "false_positive_cat7_label"  # attribution bug
                elif refinement_shifted:
                    mechanism = "refinement_shifted_out"
                elif not candidate_in_window and in_window_pre:
                    mechanism = "refinement_shifted_out"
                elif not candidate_in_window:
                    mechanism = "no_candidate_in_window_after_generation"
                elif not appended_in_window:
                    # Candidate existed but classifier loop didn't append it
                    mechanism = "loop_rejected"
                elif loop_accepted_count > 0 and not contact_in_final:
                    mechanism = "dedup_eliminated"
                else:
                    mechanism = "unknown"

                results.append(Mechanism(
                    rally_id=rally.rally_id,
                    gt_frame=gt_frame,
                    gt_action=fn["gt_action"],
                    candidate_in_window=candidate_in_window,
                    candidate_frames_in_window=in_window_post,
                    loop_skipped_warmup=False,
                    loop_skipped_post_rally=False,
                    loop_skipped_no_ball=False,
                    loop_skipped_low_velocity=False,
                    loop_gbm_rejected=candidate_in_window and loop_accepted_count == 0,
                    best_gbm_in_window=best_gbm,
                    loop_accepted_count=loop_accepted_count,
                    contact_in_final_seq=contact_in_final,
                    appended_frames_in_window=[f for f, _ in appended_in_window],
                    dedup_dropped_appended=loop_accepted_count > 0 and not contact_in_final,
                    refinement_shifted_out=refinement_shifted,
                    refinement_shift_amount=shift_amount,
                    mechanism=mechanism,
                ))

            _reset_action_classifier_cache()

    # Write output
    with OUT_PATH.open("w") as f:
        for m in results:
            f.write(json.dumps(asdict(m), default=str) + "\n")

    # Summary
    print(f"\n=== Cat 7 mechanism trace (n={len(results)}) ===", flush=True)
    mech_counts = Counter(m.mechanism for m in results)
    for name, n in mech_counts.most_common():
        print(f"  {name:40s} {n}", flush=True)

    # Sub-stats
    shift_amounts = [m.refinement_shift_amount for m in results if m.refinement_shifted_out]
    if shift_amounts:
        print(f"\nRefinement shift amounts: min={min(shift_amounts)} max={max(shift_amounts)} n={len(shift_amounts)}", flush=True)
    loop_rej = sum(1 for m in results if m.mechanism == "loop_rejected")
    if loop_rej > 0:
        print("\nLoop-rejected gbms (best in window): ", flush=True)
        for m in results:
            if m.mechanism == "loop_rejected":
                print(f"  {m.rally_id[:8]}/{m.gt_frame}/{m.gt_action}: best_gbm={m.best_gbm_in_window:.3f}", flush=True)
    dedup_real = sum(1 for m in results if m.mechanism == "dedup_eliminated")
    fp_labels = sum(1 for m in results if m.mechanism == "false_positive_cat7_label")
    print(f"\nActual dedup eliminations: {dedup_real}", flush=True)
    print(f"False-positive Cat 7 labels (contact actually in final): {fp_labels}", flush=True)

    print(f"\nWrote {OUT_PATH.relative_to(REPO)}", flush=True)


if __name__ == "__main__":
    main()
