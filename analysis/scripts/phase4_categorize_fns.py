"""Phase 4 categorization: corpus-wide assignment of non-block FNs to Phase 3 categories.

Brief: docs/superpowers/briefs/2026-04-21-contact-detection-full-review.md
Categories: analysis/reports/contact_fn_phase3_categories_2026_04_21.md

Inputs:
  analysis/outputs/fn_stage_attribution.jsonl        (266 rows, stage labels)
  analysis/outputs/action_errors/corpus_eval_reconciled.jsonl  (313 FN rows, diag fields)

Strategy:
- Join by (rally_id, gt_frame). Attribution is authoritative for stage; corpus carries
  per-FN numeric features the attribution lacks.
- For each non-block FN, run all 7 category detectors; assign ≤1 primary category by priority,
  plus Category 7 as a cross-cutting tag if applicable.
- Output: counts table + per-category examples + jsonl with full assignment per FN.

Priority order (primary category):
  stage ball_tracked=False                                 → 5 (serve) or 6 (other)
  stage candidate_generated=False AND ball_tracked=True    → 1 (interp-erases) — ball gap 1-5
  stage classifier_accepted=False                          → 4 (dual-occlusion) | 3 (kin-max) | 2 (kin-under) | 99 (other)
  stage dedup_survived=False                               → Category 7 (cross-cutting); primary still set per features
Cross-cutting tag:
  Category 7 if classifier_accepted=True AND dedup_survived=False AND nearest_candidate_gbm >= 0.30.

Run:
  cd analysis && uv run python scripts/phase4_categorize_fns.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rallycut.evaluation.corpus_freshness import iter_errors, verify_corpus_fresh

REPO = Path(__file__).resolve().parent.parent
ATTR_PATH = REPO / "outputs" / "fn_stage_attribution.jsonl"
# v2 carries the freshness _meta header (2026-04-21 rebuild). The pre-rebuild
# `corpus_eval_reconciled.jsonl` was stale by ~12% of FN records — see
# feedback_corpus_freshness.md. Point at v2 until the naming is consolidated.
CORPUS_PATH = REPO / "outputs" / "action_errors" / "corpus_eval_reconciled_v2.jsonl"
OUT_JSONL = REPO / "outputs" / "phase4_category_assignments.jsonl"
OUT_MD = REPO / "reports" / "contact_fn_phase4_counts_2026_04_21.md"

# Detector thresholds — chosen to match the Phase 3 observable signatures exactly.
TOL_MATCHER = 7       # Hungarian matcher tolerance (frames, at 30fps)
MIN_PEAK_DIST = 12    # contact_detector.py ContactDetectionConfig
GBM_THRESHOLD = 0.30  # production threshold
MAX_INTERP_GAP = 5    # ball_filter.py max_interpolation_gap


def load_jsonl(p: Path) -> list[dict]:
    """Load jsonl, skipping any `_meta` header line for corpus freshness support."""
    return list(iter_errors(p))


def safe(d: dict, key: str, default=None):
    v = d.get(key, default)
    if v is None:
        return default
    return v


def join_rows() -> list[dict]:
    """Return enriched non-block FN rows: every attribution row + carried corpus fields."""
    attr = load_jsonl(ATTR_PATH)
    corpus = load_jsonl(CORPUS_PATH)
    corpus_idx = {
        (r["rally_id"], r["gt_frame"]): r
        for r in corpus if r["error_class"] == "FN_contact"
    }
    out: list[dict] = []
    for r in attr:
        if r["gt_action"] == "block":
            continue
        c = corpus_idx.get((r["rally_id"], r["gt_frame"])) or {}
        merged = {**r}
        for k in (
            "classifier_conf", "nearest_cand_dist", "ball_gap_frames",
            "velocity", "direction_change_deg", "player_distance",
            "seq_peak_nonbg_within_5f", "seq_peak_action", "seq_peak_action_prob",
            "gt_player_track_id", "fn_subcategory", "fps", "start_ms", "video_id",
        ):
            if k not in merged or merged[k] is None:
                merged[k] = c.get(k)
        out.append(merged)
    return out


def detect_category_7(r: dict) -> bool:
    """Cross-cutting: Confident-accept-killed-by-dedup."""
    return (
        r.get("classifier_accepted") is True
        and r.get("dedup_survived") is False
        and safe(r, "accepted_in_window_nearest_gbm", -1.0) >= GBM_THRESHOLD
    )


def detect_category_1(r: dict) -> bool:
    """Interp-erases-deflection: small ball gap, interp filled, no candidate fired."""
    gap = safe(r, "ball_gap_frames", 0)
    # Use attribution's nearest_candidate_distance if corpus lacks it
    nearest = safe(r, "nearest_cand_dist", 9999)
    return (
        r.get("ball_tracked") is True
        and r.get("candidate_generated") is False
        and 1 <= gap <= MAX_INTERP_GAP
        and nearest > MIN_PEAK_DIST
    )


def detect_category_2(r: dict) -> bool:
    """Kinematic-underreports-visual.

    Loosened 2026-04-21 (2 rounds):
    - r1: dir_change<=10 → <=30; player_d<=0.10 → <=0.15 (matches `player_contact_radius`)
    - r2: ball_gap=0 → <=3 (absorbs the Mode ζ small-interp-gap sub-mode into Mode α since they
      share the same classifier-rejection mechanism at the classifier stage; Cat 1 remains the
      candidate-generator-stage expression of Mode ζ where no candidate fires at all).
    """
    if r.get("classifier_accepted") is not False:
        return False
    gap = safe(r, "ball_gap_frames", 0)
    if gap is None or gap > 3:
        return False
    dir_ch = safe(r, "direction_change_deg", -1.0)
    player_d = safe(r, "player_distance", float("inf"))
    seq = safe(r, "seq_peak_nonbg_within_5f", 0.0)
    gbm = safe(r, "nearest_candidate_gbm", 0.0)
    if player_d is None or (isinstance(player_d, float) and player_d != player_d):
        return False
    # dir_ch == -1.0 means corpus didn't compute it (small-gap degenerate); treat as
    # "kinematic signal unavailable" which is still consistent with Cat 2's mechanism.
    dir_ok = dir_ch is None or dir_ch <= 30.0 or dir_ch == -1.0
    return (
        dir_ok
        and player_d != float("inf") and player_d <= 0.15
        and seq >= 0.85
        and 0.0 <= gbm < GBM_THRESHOLD
    )


def detect_category_2b(r: dict) -> bool:
    """Kinematic-moderate-GBM-rejects-deeply.

    Loosened 2026-04-21 (r2): gbm<=0.15 → <=0.30 (covers the full sub-threshold range).
    Rejected candidates where dir_change is MODEST (30-170°) — not flat like Cat 2, not maximal
    like Cat 3. Real kinematic signal exists but GBM doesn't credit.
    """
    if r.get("classifier_accepted") is not False:
        return False
    if safe(r, "ball_gap_frames", 0) != 0:
        return False
    dir_ch = safe(r, "direction_change_deg", -1.0)
    gbm = safe(r, "nearest_candidate_gbm", 1.0)
    if dir_ch is None or gbm is None:
        return False
    return 30.0 < dir_ch < 170.0 and 0.0 <= gbm < GBM_THRESHOLD


def detect_category_3(r: dict) -> bool:
    """Kinematic-maximal-GBM-rejects-deeply."""
    if r.get("classifier_accepted") is not False:
        return False
    dir_ch = safe(r, "direction_change_deg", 0.0)
    gbm = safe(r, "nearest_candidate_gbm", 1.0)
    return dir_ch is not None and dir_ch >= 170.0 and 0.0 <= gbm <= 0.05


def detect_category_4(r: dict) -> bool:
    """Heavy-dual-occlusion: player lost AND ball gap present."""
    player_d = safe(r, "player_distance", 0.0)
    gap = safe(r, "ball_gap_frames", 0)
    return player_d == float("inf") and gap is not None and gap >= 1


def detect_category_5(r: dict) -> bool:
    """Serve-ball-dropout-shifted-candidate."""
    if r["gt_action"] != "serve":
        return False
    if r.get("ball_tracked") is not False:
        return False
    gap = safe(r, "ball_gap_frames", 0)
    nearest = safe(r, "nearest_candidate_distance", 9999)  # from attribution
    return gap is not None and gap >= 5 and TOL_MATCHER < nearest < 9999


def detect_category_6(r: dict) -> bool:
    """Ball-gap-exceeds-interp.

    Loosened 2026-04-21 from gap>5 to gap>=4 after first-pass showed 17 uncategorized
    ball_tracker cases had gap 4-6. Linear interp formally fills gap<=5, but in practice
    gap=4-5 cases behave identically to gap>5 (cascade failures) because the interp is
    too aggressive/wrong. Treat gap>=4 as the operational boundary.

    Also relaxed nearest>=10 check since 12/17 u-ball_tracker cases were serves with
    no candidate generated anywhere (-1 value).
    """
    if r.get("ball_tracked") is not False:
        return False
    gap = safe(r, "ball_gap_frames", 0)
    return gap is not None and gap >= 4


def detect_category_5b(r: dict) -> bool:
    """Serve-cand-gen-other (new in Phase 4 loosening).

    Serves lost at candidate_generated stage without a large ball gap — WASB had positions
    but no kinematic generator fired. Consistent with brief §closed "Serve candidate generator
    NO-GO 2026-04-20": 53/59 Mode C serves are a WASB emergence-recall issue + generator
    thresholds miscalibrated for serve kinematics.
    """
    if r.get("gt_action") != "serve":
        return False
    if r.get("lost_at_stage") != "candidate_generated":
        return False
    gap = safe(r, "ball_gap_frames", 0)
    return gap is not None and gap <= 3


def assign_primary(r: dict) -> str:
    """Assign primary category label. Returns category id or 'uncategorized'."""
    # Priority by stage-lost first
    stage = r.get("lost_at_stage")

    if stage == "ball_tracked":
        if detect_category_5(r):
            return "5-serve_ball_dropout"
        if detect_category_6(r):
            return "6-ball_gap_exceeds_interp"
        return "u-ball_tracker_other"

    if stage == "candidate_generated":
        if detect_category_5b(r):
            return "5b-serve_cand_gen_other"
        if detect_category_1(r):
            return "1-interp_erases_deflection"
        return "u-candidate_gen_other"

    if stage == "classifier_accepted":
        # Check dual-occlusion first (most specific feature-based signature)
        if detect_category_4(r):
            return "4-dual_occlusion"
        if detect_category_3(r):
            return "3-kin_max_gbm_rejects"
        if detect_category_2(r):
            return "2-kin_underreports"
        if detect_category_2b(r):
            return "2b-kin_moderate_gbm_rejects"
        return "u-classifier_other"

    if stage == "dedup_survived":
        # Category 7 is cross-cutting, but dedup losses are primarily 7
        if detect_category_7(r):
            # Sub-label by co-occurring feature-level category
            if detect_category_4(r):
                return "7+4-dedup_kill_with_occlusion"
            return "7-dedup_kill"
        return "u-dedup_other"

    if stage == "matched_to_gt":
        return "10-matcher_steal"

    if stage == "action_labeled":
        return "8-action_labeling"

    if stage == "seq_signal":
        return "seq-signal-only"

    return f"stage-{stage}"


def main() -> None:
    # Freshness pre-flight: reproduce the canary fold and verify the corpus's
    # stored fingerprint matches current code. Catches the "stale corpus"
    # bug class documented in feedback_corpus_freshness.md. ~1 min overhead.
    import argparse as _ap
    parser = _ap.ArgumentParser()
    parser.add_argument("--skip-freshness-check", action="store_true",
                        help="Skip the canary-fold reproduction. Use only if you know why.")
    args = parser.parse_args()
    if not args.skip_freshness_check:
        from scripts.build_eval_reconciled_corpus import reproduce_single_fold
        verify_corpus_fresh(
            CORPUS_PATH,
            reproduce_canary_fn=reproduce_single_fold,
            abort_on_stale=True,
            abort_on_legacy=False,
        )

    rows = join_rows()
    print(f"Loaded {len(rows)} non-block FNs from attribution ∪ corpus.")

    # Assign primary + cross-cutting
    assignments: list[dict] = []
    primary_counter: Counter = Counter()
    secondary_cat7: set = set()
    examples_by_primary: dict[str, list[tuple[str, int, str]]] = defaultdict(list)

    for r in rows:
        primary = assign_primary(r)
        is_cat7 = detect_category_7(r)
        # 1/2/3/4/5/6 detectors (flat, for overlap analysis)
        tags = {
            "cat1": detect_category_1(r),
            "cat2": detect_category_2(r),
            "cat2b": detect_category_2b(r),
            "cat3": detect_category_3(r),
            "cat4": detect_category_4(r),
            "cat5": detect_category_5(r),
            "cat5b": detect_category_5b(r),
            "cat6": detect_category_6(r),
            "cat7": is_cat7,
        }
        assignments.append({
            "rally_id": r["rally_id"],
            "gt_frame": r["gt_frame"],
            "gt_action": r["gt_action"],
            "lost_at_stage": r.get("lost_at_stage"),
            "primary_category": primary,
            "tag_cat7": is_cat7,
            **tags,
            # Copy key diagnostic fields for at-a-glance review
            "ball_gap_frames": r.get("ball_gap_frames"),
            "direction_change_deg": r.get("direction_change_deg"),
            "player_distance": r.get("player_distance"),
            "nearest_candidate_gbm": r.get("nearest_candidate_gbm"),
            "accepted_in_window_nearest_gbm": r.get("accepted_in_window_nearest_gbm"),
            "seq_peak_nonbg_within_5f": r.get("seq_peak_nonbg_within_5f"),
            "seq_peak_action": r.get("seq_peak_action"),
        })
        primary_counter[primary] += 1
        if is_cat7:
            secondary_cat7.add((r["rally_id"], r["gt_frame"]))
        if len(examples_by_primary[primary]) < 5:
            examples_by_primary[primary].append((
                r["rally_id"][:8], r["gt_frame"], r["gt_action"],
            ))

    # Individual category counts (flat, possibly overlapping)
    flat_counts = {
        "cat1": sum(1 for a in assignments if a["cat1"]),
        "cat2": sum(1 for a in assignments if a["cat2"]),
        "cat2b": sum(1 for a in assignments if a["cat2b"]),
        "cat3": sum(1 for a in assignments if a["cat3"]),
        "cat4": sum(1 for a in assignments if a["cat4"]),
        "cat5": sum(1 for a in assignments if a["cat5"]),
        "cat5b": sum(1 for a in assignments if a["cat5b"]),
        "cat6": sum(1 for a in assignments if a["cat6"]),
        "cat7": sum(1 for a in assignments if a["cat7"]),
    }
    print("Flat category counts (may overlap):", flat_counts)
    print("Primary category counts:")
    for cat, n in primary_counter.most_common():
        print(f"  {cat:35s} {n}")

    # Write jsonl
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSONL.open("w") as f:
        for a in assignments:
            f.write(json.dumps(a, default=str) + "\n")

    # Per-category per-stage (sanity: does each primary cluster actually concentrate in the expected stage?)
    stage_by_primary: dict[str, Counter] = defaultdict(Counter)
    action_by_primary: dict[str, Counter] = defaultdict(Counter)
    for a in assignments:
        stage_by_primary[a["primary_category"]][a["lost_at_stage"] or "?"] += 1
        action_by_primary[a["primary_category"]][a["gt_action"]] += 1

    # Write markdown report
    lines: list[str] = []
    lines.append("# Contact FN Phase 4 — Corpus-wide category counts")
    lines.append("")
    lines.append("Counts over 238 non-block FNs. Source: "
                 "`outputs/fn_stage_attribution.jsonl` joined with "
                 "`outputs/action_errors/corpus_eval_reconciled.jsonl`. "
                 "Generator: `scripts/phase4_categorize_fns.py` (SEED=n/a, deterministic).")
    lines.append("")
    lines.append(f"**Total non-block FNs analyzed: {len(rows)}**")
    lines.append("")
    lines.append("## Primary category counts")
    lines.append("")
    lines.append(
        "Each FN is assigned to exactly one primary category. "
        "Priority: stage-lost → most-specific feature signature. "
        "Uncategorized-within-stage buckets use prefix `u-`.")
    lines.append("")
    lines.append("| Category | Count | % of 238 | Expected from sample | Examples (rally[:8]/frame/action) |")
    lines.append("|---|---|---|---|---|")
    sample_expected = {
        "1-interp_erases_deflection": "~12",
        "2-kin_underreports": "~60",
        "3-kin_max_gbm_rejects": "~24",
        "4-dual_occlusion": "~24",
        "5-serve_ball_dropout": "~12",
        "6-ball_gap_exceeds_interp": "~12",
        "7-dedup_kill": "~107 (but capped by 84 dedup-lost)",
        "7+4-dedup_kill_with_occlusion": "subset of 7",
    }
    for cat, n in sorted(primary_counter.items(), key=lambda x: -x[1]):
        pct = f"{100 * n / len(rows):.1f}%"
        expected = sample_expected.get(cat, "—")
        exs = ", ".join(f"`{r}`/{f}/{a}" for r, f, a in examples_by_primary[cat][:3])
        lines.append(f"| `{cat}` | {n} | {pct} | {expected} | {exs} |")
    lines.append("")

    lines.append("## Flat category counts (include cross-cutting overlaps)")
    lines.append("")
    lines.append("| Category | Flat count | % of 238 |")
    lines.append("|---|---|---|")
    cat_descs = [
        ("cat1", "1-interp_erases_deflection"),
        ("cat2", "2-kin_underreports (dir<=30)"),
        ("cat2b", "2b-kin_moderate_gbm_rejects (30<dir<170 AND gbm<=0.15)"),
        ("cat3", "3-kin_max_gbm_rejects (dir>=170)"),
        ("cat4", "4-dual_occlusion"),
        ("cat5", "5-serve_ball_dropout"),
        ("cat5b", "5b-serve_cand_gen_other"),
        ("cat6", "6-ball_gap_exceeds_interp (gap>=4)"),
        ("cat7", "7-dedup_kill"),
    ]
    for key, name in cat_descs:
        n = flat_counts[key]
        pct = f"{100 * n / len(rows):.1f}%"
        lines.append(f"| `{name}` | {n} | {pct} |")
    lines.append("")

    lines.append("## Per-primary stage sanity-check")
    lines.append("")
    lines.append("Each row: primary category → how its FNs distribute across `lost_at_stage`.")
    lines.append("")
    lines.append("| Primary | top 3 stages |")
    lines.append("|---|---|")
    for cat, _ in sorted(primary_counter.items(), key=lambda x: -x[1]):
        top3 = stage_by_primary[cat].most_common(3)
        top3_str = ", ".join(f"{s}:{n}" for s, n in top3)
        lines.append(f"| `{cat}` | {top3_str} |")
    lines.append("")

    lines.append("## Per-primary action distribution")
    lines.append("")
    lines.append("| Primary | action distribution (top 3) |")
    lines.append("|---|---|")
    for cat, _ in sorted(primary_counter.items(), key=lambda x: -x[1]):
        top3 = action_by_primary[cat].most_common(3)
        top3_str = ", ".join(f"{a}:{n}" for a, n in top3)
        lines.append(f"| `{cat}` | {top3_str} |")
    lines.append("")

    lines.append("## Cross-cutting: Category 7 overlap with primary categories")
    lines.append("")
    lines.append("Category 7 is the `classifier_accepted=True AND dedup_survived=False` signature. "
                 "Primary assignment is dedup_survived-stage-based, but Cat 7 can tag "
                 "classifier-accepted cases outside dedup-stage if they exist.")
    lines.append("")
    cat7_by_primary: Counter = Counter()
    for a in assignments:
        if a["tag_cat7"]:
            cat7_by_primary[a["primary_category"]] += 1
    lines.append("| Primary category | Count with Cat7 tag |")
    lines.append("|---|---|")
    for cat, n in cat7_by_primary.most_common():
        lines.append(f"| `{cat}` | {n} |")
    lines.append("")

    lines.append("## Output artifacts")
    lines.append("")
    lines.append(f"- `{OUT_JSONL.relative_to(REPO)}` — per-FN assignment (one line per non-block FN)")
    lines.append(f"- `{OUT_MD.relative_to(REPO)}` — this file")
    lines.append("")
    lines.append("Interpret via `analysis/reports/contact_fn_phase3_categories_2026_04_21.md` for signature + mechanism hypotheses.")
    lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines))
    print(f"\nWrote {OUT_JSONL.relative_to(REPO)}")
    print(f"Wrote {OUT_MD.relative_to(REPO)}")


if __name__ == "__main__":
    main()
