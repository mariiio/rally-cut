"""Phase 2 sampler: stratified 20-case FN sample for visual inspection.

Brief: docs/superpowers/briefs/2026-04-21-contact-detection-full-review.md

Strategy:
- Source of truth for stage label: outputs/fn_stage_attribution.jsonl
  (266 rows; corpus_eval_reconciled.jsonl has 313 FNs but 47 serves lack
   stage attribution — we skip those for stage-stratified sampling).
- Join to outputs/action_errors/corpus_eval_reconciled.jsonl to pick up the
  per-FN diagnostic fields we want in the case log.
- Per action class, sample proportional to non-block stage-attribution
  shares for that class. Deterministic seed.
- For serves: additionally force 1 near / 1 far via gt_player_track_id
  convention (tracks 1-2 = near team 0, 3-4 = far team 1).

Writes:
  analysis/outputs/phase2_sample_2026_04_21.jsonl (machine-readable)
  analysis/reports/contact_fn_visual_log_2026_04_21.md (the pre-registered log)
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ATTR_PATH = REPO / "outputs" / "fn_stage_attribution.jsonl"
CORPUS_PATH = REPO / "outputs" / "action_errors" / "corpus_eval_reconciled.jsonl"
CLIPS_DIR = REPO / "outputs" / "action_errors" / "clips"
OUT_JSONL = REPO / "outputs" / "phase2_sample_2026_04_21.jsonl"
OUT_MD = REPO / "reports" / "contact_fn_visual_log_2026_04_21.md"

SEED = 42

# Per-action stratum size (brief: 5 dig, 5 receive, 5 set, 3 attack, 2 serve)
STRATUM_SIZES = {"dig": 5, "receive": 5, "set": 5, "attack": 3, "serve": 2}


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.open()]


def team_from_track(tid: int) -> str:
    if tid in (1, 2):
        return "near"
    if tid in (3, 4):
        return "far"
    return "unknown"


def stratified_sample_within_class(
    rows: list[dict], n: int, rng: random.Random,
) -> list[dict]:
    """Sample n rows proportional to lost_at_stage shares."""
    by_stage: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_stage[r["lost_at_stage"]].append(r)

    total = len(rows)
    # Largest-remainder method for integer quota allocation
    quotas_float = {s: (len(rs) / total) * n for s, rs in by_stage.items()}
    quotas_int = {s: int(q) for s, q in quotas_float.items()}
    leftover = n - sum(quotas_int.values())
    # Distribute leftover to stages with largest fractional remainders
    remainders = sorted(
        ((s, quotas_float[s] - quotas_int[s]) for s in quotas_int),
        key=lambda x: -x[1],
    )
    for i in range(leftover):
        quotas_int[remainders[i % len(remainders)][0]] += 1

    picked: list[dict] = []
    for stage, rs in by_stage.items():
        k = min(quotas_int.get(stage, 0), len(rs))
        if k == 0:
            continue
        rng.shuffle(rs)  # In-place; deterministic under seeded rng.
        picked.extend(rs[:k])
    return picked


def draw_sample() -> list[dict]:
    attr = load_jsonl(ATTR_PATH)
    corpus_rows = load_jsonl(CORPUS_PATH)
    corpus_idx = {
        (r["rally_id"], r["gt_frame"]): r
        for r in corpus_rows if r["error_class"] == "FN_contact"
    }
    # Enrich attribution rows with corpus diagnostic fields when available.
    non_block = []
    for r in attr:
        if r["gt_action"] == "block":
            continue
        joined = dict(r)
        corpus_row = corpus_idx.get((r["rally_id"], r["gt_frame"]))
        if corpus_row:
            # Copy fields we want to display in the log.
            for k in (
                "classifier_conf",
                "nearest_cand_dist",
                "ball_gap_frames",
                "velocity",
                "direction_change_deg",
                "player_distance",
                "seq_peak_nonbg_within_5f",
                "seq_peak_action",
                "seq_peak_action_prob",
                "gt_player_track_id",
                "fn_subcategory",
                "fps",
                "start_ms",
            ):
                joined.setdefault(k, corpus_row.get(k))
        non_block.append(joined)

    rng = random.Random(SEED)

    sampled: list[dict] = []
    for action, n in STRATUM_SIZES.items():
        class_rows = [r for r in non_block if r["gt_action"] == action]
        if action == "serve":
            # Need 1 near, 1 far via gt_player_track_id.
            near = [r for r in class_rows
                    if team_from_track(r.get("gt_player_track_id", -1)) == "near"]
            far = [r for r in class_rows
                   if team_from_track(r.get("gt_player_track_id", -1)) == "far"]
            rng.shuffle(near)
            rng.shuffle(far)
            picks = []
            if near:
                picks.append(near[0])
            if far:
                picks.append(far[0])
            # Backfill if one side absent (e.g., off-screen server with -1 tid)
            while len(picks) < n:
                remaining = [r for r in class_rows if r not in picks]
                if not remaining:
                    break
                rng.shuffle(remaining)
                picks.append(remaining[0])
            sampled.extend(picks[:n])
            continue
        sampled.extend(stratified_sample_within_class(class_rows, n, rng))

    # Stable sort by (gt_action, lost_at_stage, rally_id) for presentation.
    sampled.sort(key=lambda r: (r["gt_action"], r["lost_at_stage"], r["rally_id"]))

    # Attach clip path.
    for i, r in enumerate(sampled, start=1):
        clip_name = f"{r['rally_id']}_{r['gt_frame']}.mp4"
        clip_path = CLIPS_DIR / clip_name
        r["_case_id"] = f"C{i:02d}"
        r["_clip_name"] = clip_name
        r["_clip_path"] = str(clip_path)
        r["_clip_exists"] = clip_path.exists()

    return sampled


def write_jsonl(sample: list[dict]) -> None:
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSONL.open("w") as f:
        for r in sample:
            f.write(json.dumps(r, default=str) + "\n")


def write_markdown_log(sample: list[dict]) -> None:
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Contact FN visual inspection log — 2026-04-21")
    lines.append("")
    lines.append("Phase 2 deliverable of the contact-detection full review "
                 "(brief: `docs/superpowers/briefs/2026-04-21-contact-detection-full-review.md`).")
    lines.append("")
    lines.append("**Discipline (from brief § Phase 2):** answer the 5 questions "
                 "for each case BEFORE opening the diagnostic data. Watch the clip first.")
    lines.append("")
    lines.append("**Sample provenance:** seed=42 stratified draw from "
                 "`analysis/outputs/fn_stage_attribution.jsonl` (238 non-block FNs), "
                 "proportional to `lost_at_stage` shares within each action class. "
                 "Serves stratified 1 near / 1 far via gt_player_track_id. "
                 "Generator: `analysis/scripts/sample_phase2_fns.py`. "
                 "Machine copy: `analysis/outputs/phase2_sample_2026_04_21.jsonl`.")
    lines.append("")
    lines.append("## Case index")
    lines.append("")
    lines.append("| Case | Action | Stage lost | Rally | GT frame | "
                 "GT trk | Clip |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in sample:
        rally = r["rally_id"]
        lines.append(
            f"| {r['_case_id']} | {r['gt_action']} | {r['lost_at_stage']} | "
            f"`{rally[:8]}` | {r['gt_frame']} | "
            f"{r.get('gt_player_track_id', '—')} | "
            f"`{r['_clip_name']}`{' ✓' if r['_clip_exists'] else ' ✗'} |"
        )
    lines.append("")
    lines.append("## Per-case log")
    lines.append("")
    lines.append("Each case has two sections:")
    lines.append("")
    lines.append(
        "- **A. Visual observations (fill first, before opening diagnostics).** "
        "Answer the 5 brief questions.")
    lines.append(
        "- **B. Pipeline state (fill after A).** Diagnostic fields from the "
        "corpus + attribution. Note agreements/disagreements with A.")
    lines.append("")
    for r in sample:
        rally = r["rally_id"]
        lines.append(f"### {r['_case_id']} — {r['gt_action']} @ frame "
                     f"{r['gt_frame']} (rally `{rally[:8]}…`)")
        lines.append("")
        lines.append(f"- **Clip:** `outputs/action_errors/clips/{r['_clip_name']}`")
        lines.append(f"- **Video id:** `{r['video_id']}`")
        lines.append(f"- **Lost at stage:** `{r['lost_at_stage']}`")
        lines.append(f"- **GT player track id:** {r.get('gt_player_track_id', '—')} "
                     f"({team_from_track(r.get('gt_player_track_id', -1))})")
        lines.append("")
        lines.append("#### A. Visual observations")
        lines.append("")
        lines.append("1. **Is the ball visibly deflected at the contact frame?** "
                     "(yes / subtle / no-visible-change)")
        lines.append("   - _TBD_")
        lines.append("")
        lines.append("2. **Is the ball visible (not occluded) throughout ±5 frames "
                     "around contact?** (yes / partial / occluded)")
        lines.append("   - _TBD_")
        lines.append("")
        lines.append("3. **Is a player visibly in contact position (hands/arms "
                     "positioned to touch the ball)?** (yes / ambiguous / no)")
        lines.append("   - _TBD_")
        lines.append("")
        lines.append("4. **Are there any other visible contacts within ±10 frames?** "
                     "(none / one-adjacent / multiple)")
        lines.append("   - _TBD_")
        lines.append("")
        lines.append("5. **Compared to ACCEPTED contacts nearby in the same rally, "
                     "what's visually different?** (free-text)")
        lines.append("   - _TBD_")
        lines.append("")
        lines.append("#### B. Pipeline state (fill after A)")
        lines.append("")

        def fmt(v, p=3):
            if v is None:
                return "—"
            if isinstance(v, float):
                return f"{v:.{p}f}"
            return str(v)

        lines.append(f"- `ball_tracked`: {r.get('ball_tracked', '—')}  "
                     f"`player_tracked`: {r.get('player_tracked', '—')}")
        lines.append(f"- `candidate_generated`: {r.get('candidate_generated', '—')}  "
                     f"`classifier_accepted`: {r.get('classifier_accepted', '—')}  "
                     f"`dedup_survived`: {r.get('dedup_survived', '—')}  "
                     f"`action_labeled`: {r.get('action_labeled', '—')}")
        lines.append(f"- `nearest_candidate_distance`: "
                     f"{fmt(r.get('nearest_candidate_distance'))} frames  "
                     f"`nearest_candidate_gbm`: "
                     f"{fmt(r.get('nearest_candidate_gbm'))}")
        lines.append(f"- corpus `classifier_conf`: "
                     f"{fmt(r.get('classifier_conf'))}  "
                     f"`fn_subcategory`: {r.get('fn_subcategory', '—')}")
        lines.append(f"- `velocity`: {fmt(r.get('velocity'), 4)}  "
                     f"`direction_change_deg`: "
                     f"{fmt(r.get('direction_change_deg'), 1)}°  "
                     f"`player_distance`: "
                     f"{fmt(r.get('player_distance'), 3)}")
        lines.append(f"- `ball_gap_frames` (corpus): "
                     f"{fmt(r.get('ball_gap_frames'))}")
        lines.append(f"- `seq_peak_nonbg_within_5f` (corpus): "
                     f"{fmt(r.get('seq_peak_nonbg_within_5f'), 3)}  "
                     f"attribution `seq_peak_nonbg`: "
                     f"{fmt(r.get('seq_peak_nonbg'), 3)}  "
                     f"`seq_peak_action`: {r.get('seq_peak_action', '—')}")
        lines.append(f"- `adjacent_gt_took_it_frame`: "
                     f"{r.get('adjacent_gt_took_it_frame', '—')}  "
                     f"`adjacent_gt_took_it_action`: "
                     f"{r.get('adjacent_gt_took_it_action', '—')}")
        lines.append(f"- `detected_contact_frames_in_window`: "
                     f"{r.get('detected_contact_frames_in_window', '—')}  "
                     f"`rally_actions_in_window`: "
                     f"{r.get('rally_actions_in_window', '—')}")
        lines.append(f"- `matched_to_gt`: {r.get('matched_to_gt', '—')}  "
                     f"`accepted_in_window_nearest_gbm`: "
                     f"{fmt(r.get('accepted_in_window_nearest_gbm'))}")
        lines.append("")
        lines.append("**Agreements / disagreements with Visual observations:**")
        lines.append("- _TBD_")
        lines.append("")
        lines.append("---")
        lines.append("")

    OUT_MD.write_text("\n".join(lines))


def main() -> None:
    sample = draw_sample()
    write_jsonl(sample)
    write_markdown_log(sample)

    # Stdout summary.
    from collections import Counter
    print(f"Drew {len(sample)} cases. Seed={SEED}.")
    by_action = Counter(r["gt_action"] for r in sample)
    by_stage = Counter(r["lost_at_stage"] for r in sample)
    print("By action:", dict(by_action))
    print("By stage :", dict(by_stage))
    missing = [r for r in sample if not r["_clip_exists"]]
    print(f"Clips missing: {len(missing)}")
    if missing:
        for r in missing:
            print(f"  MISSING {r['_case_id']} {r['_clip_path']}")
    print(f"\nWrote: {OUT_JSONL.relative_to(REPO)}")
    print(f"Wrote: {OUT_MD.relative_to(REPO)}")


if __name__ == "__main__":
    main()
