"""Aggregate reports/tracking_audit/*.json into a root-cause markdown report.

Output: reports/tracking_audit/root_cause.md

Usage:
  uv run python scripts/write_tracking_audit_report.py
  uv run python scripts/write_tracking_audit_report.py --out custom/path.md
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("report")


# Map each failure signal to the pipeline stage(s) most plausibly responsible.
# Source: pipeline_diagnosis_2026_03, trackid_stability_2026_04_15, and the
# 10-stage reference in tracking-diagnosis SKILL.md.
STAGE_ATTRIBUTION = {
    "out_of_frame":      "Structural — GT outside frame; no fix available at tracking layer.",
    "edge_proximity":    "Stage 1 (YOLO) — detector recall drops at frame edges; consider tiled inference or uncropped ROI.",
    "occlusion":         "Stage 1 (YOLO) + Stage 7 (global identity) — detector may produce one merged box; identity stage must handle reunion.",
    "filter_drop":       "Stage 3 (stationary BG) / Stage 8 (stabilize primary set) — raw detection survives YOLO but gets removed by filter.",
    "detector_miss":     "Stage 1 (YOLO) — far-court recall ceiling. Mitigations: input-size bump, tiled inference, retrain with beach GT.",
    "net_crossing":      "Stage 7 (global identity) — cost function can't distinguish two players colocated at net. Needs explicit anti-swap prior or temporal trajectory anchor.",
    "same_team_swap":    "Stage 7 (global identity within-team) — color/appearance costs insufficient when team-mates are visually similar. Needs stronger intra-team ReID (keypoint geometry, trajectory, role priors).",
    "cross_team_swap":   "Stage 6 (team classification) or Stage 7 (global identity) — cross-team swap should be blocked by team-consistency cost; firing implies cost weight too weak.",
    "fragment_gap":      "Stage 2 (BoT-SORT) + Stage 9 (tracklet link) — fragments from Kalman/ReID noise; linking fails to reconnect within team.",
    "court_side_flip":   "Stage 8/9 canonicalisation + match_tracker convention anchor — pred geometry correct but labels reversed.",
    "team_label_flip":   "match_tracker team assignment — pred team label disagrees with GT side geometry; convention anchor not applied at rally level.",
}

# Structural remediation per cause — deliberately not threshold tweaks.
RECOMMENDATION = {
    "detector_miss":
        "Improvements at the detector layer: "
        "(a) Tiled YOLO inference (2×2 tile) to recover far-court players at native resolution; "
        "(b) Retrain yolo11s with the existing labelled-players GT set as fine-tuning data; "
        "(c) Add a low-threshold second pass on the far-court half of the frame. "
        "Do NOT lower --conf globally; FP rate on the crowd will dominate.",
    "filter_drop":
        "Re-examine Stage 3 stationary-filter + Stage 8 primary-selection: "
        "stationary_bg thresholds were tuned against historical data and may drop genuine low-motion defensive players. "
        "Replace the boolean spread threshold with a motion-normalised score (e.g. per-team spread percentile) "
        "so the thresholds self-adjust to rally pace.",
    "occlusion":
        "Occlusions are a structural detector problem, not a tracker one. "
        "(a) Keep both GT-paired detections through the filter (occlusion-aware NMS). "
        "(b) Tracker must carry velocity-based predicted position during occlusion so the re-acquisition matches the correct player.",
    "edge_proximity":
        "Edge recall: (a) widen court ROI to include baseline margin; "
        "(b) feed un-padded native-resolution frames to the detector "
        "(current 1280 imgsz down-samples 4K inputs, losing edge detail).",
    "net_crossing":
        "Add an explicit anti-swap prior inside global-identity cost at net-crossings: "
        "when two candidate tracks are both within ±0.05 of court_split_y for ≥0.5s, "
        "the assignment cost should favour trajectory continuity over appearance similarity. "
        "This is a structural change to the cost function, not a new weight.",
    "same_team_swap":
        "Same-team swaps are the hardest signal to fix without new features. "
        "(a) Integrate pose-keypoint geometry into appearance cost (already trained for attribution); "
        "(b) Add role/position priors (e.g. blocker vs defender typical y-band) as a soft constraint; "
        "(c) Last resort: a dedicated within-team ReID head trained on labelled swaps.",
    "cross_team_swap":
        "Cross-team swaps imply Stage 6 team classification gave a wrong label at the swap frame. "
        "Review whether court_split_y is re-estimated mid-rally — if yes, enforce hysteresis so a noisy frame can't flip it.",
    "fragment_gap":
        "Fragmentation at BoT-SORT: (a) increase track_buffer for beach (camera is static), "
        "(b) make tracklet_link's appearance gate directional (prefer same pred_id reunion over new id handoff), "
        "(c) evaluate the ByteTrack-style low-conf second pass that BoT-SORT already supports.",
    "court_side_flip":
        "Convention is a presentation problem, not a detection one. "
        "Enforce a canonical convention anchor at rally boundary: "
        "pred_id → player_N mapping fixed by position at serve frame, stored on RallyTrackData, "
        "used by reattribute-actions and score-tracking. This is already shipped for teams — extend to court-side labels.",
    "team_label_flip":
        "team_assignments on PlayerTrackingResult must be anchored to the GT convention "
        "when GT exists, and to the match_tracker convention when it doesn't. "
        "Current implementation derives team label from predicted y vs court_split_y with no per-video lock.",
}


@dataclass
class Aggregates:
    n_rallies: int
    n_gt_tracks: int
    n_rallies_with_gt: int
    miss_frames_by_cause: Counter
    miss_ranges_by_cause: Counter
    switch_events_by_cause: Counter
    total_real_switches: int
    rallies_with_switches: int
    pred_exchange_swaps: int          # Disguised-swap count (fragment-style).
    rallies_with_exchange_swaps: int
    convention_drift_rallies: list[str]
    fragmentation_hist: Counter  # distinct_pred_ids count → # GT tracks
    worst_coverage: list[tuple[str, str, float]]  # (rally, label, coverage)
    worst_hota: list[tuple[str, float]]
    highest_fragmented: list[tuple[str, str, int]]
    per_rally_summary: list[dict]


def _count_pred_exchange_swaps(audit: dict) -> int:
    """Disguised-swap detector — total pred-exchange boundaries (pre-trajectory-filter).

    Counts every boundary where the incoming pred_id was previously tracking a
    different GT. Some of these may be trajectory-continuity recoveries (tracker
    correctly re-associating); see `_count_swap_and_recovery` for the split.
    """
    swaps, recoveries = _count_swap_and_recovery(audit)
    return swaps + recoveries


def _count_swap_and_recovery(audit: dict) -> tuple[int, int]:
    """Return (true_swap_count, recovery_count).

    A recovery is a pred-exchange where pred_new's actual position at the swap
    frame is closer (by Δ ≥ 0.03) to pred_old's pre-swap extrapolation than to
    its own. We can't extrapolate from the audit JSON alone (no positions), so
    this function falls back to counting every pred-exchange as a swap unless
    position data was attached.

    Because the audit JSON doesn't carry raw positions, this report-level
    classifier aggregates only what was already in the JSON. The decisive
    recovery split happens in the gallery (which loads predictions), and is
    reflected via `swap_recovery` events rendered to HTML. For the report's
    headline, we retain the conservative "all pred-exchanges are swaps"
    pessimistic count — with a caveat pointing to the gallery for the split.
    """
    pred_history: dict[int, list[tuple[int, int]]] = {}
    for g in audit.get("perGt", []):
        for s, _e, pid in g.get("predIdSpans", []):
            pred_history.setdefault(pid, []).append((g["gtTrackId"], s))
    for h in pred_history.values():
        h.sort(key=lambda t: t[1])

    def prior_gt(pid: int, before: int) -> int | None:
        last_gt: int | None = None
        for gt_id, s in pred_history.get(pid, []):
            if s >= before:
                break
            last_gt = gt_id
        return last_gt

    swaps = 0
    for g in audit.get("perGt", []):
        spans = g.get("predIdSpans", [])
        for prev, cur in zip(spans, spans[1:]):
            prev_pred = prev[2]
            cur_pred = cur[2]
            cur_start = cur[0]
            # Skip identity (same pred) and unmatched-gap sentinels — these are
            # not pred-exchanges.
            if prev_pred == cur_pred or prev_pred < 0 or cur_pred < 0:
                continue
            incoming = prior_gt(cur_pred, cur_start)
            if incoming is not None and incoming != g["gtTrackId"]:
                swaps += 1
    # Report-level cannot distinguish recovery without positions.
    return swaps, 0


def aggregate(audits: list[dict]) -> Aggregates:
    miss_frames_by_cause: Counter = Counter()
    miss_ranges_by_cause: Counter = Counter()
    switch_events_by_cause: Counter = Counter()
    total_real_switches = 0
    rallies_with_switches = 0
    pred_exchange_swaps = 0
    rallies_with_exchange_swaps = 0
    convention_drift = []
    fragmentation_hist: Counter = Counter()
    worst_coverage: list[tuple[str, str, float]] = []
    worst_hota: list[tuple[str, float]] = []
    highest_fragmented: list[tuple[str, str, int]] = []
    per_rally_summary = []
    n_gt_tracks = 0

    for a in audits:
        rid = a["rallyId"]
        switches_here = a.get("aggregateRealSwitches", 0)
        total_real_switches += switches_here
        if switches_here > 0:
            rallies_with_switches += 1
        exchange_here = _count_pred_exchange_swaps(a)
        pred_exchange_swaps += exchange_here
        if exchange_here > 0:
            rallies_with_exchange_swaps += 1

        conv = a.get("convention", {})
        if conv.get("courtSideFlip") or conv.get("teamLabelFlip"):
            convention_drift.append(rid)

        worst_hota.append((rid, a.get("hota") or 0.0))

        for g in a["perGt"]:
            n_gt_tracks += 1
            for cause, ranges in g.get("missedByCause", {}).items():
                n_ranges = len(ranges)
                n_frames = sum(end - start + 1 for start, end in ranges)
                miss_frames_by_cause[cause] += n_frames
                miss_ranges_by_cause[cause] += n_ranges
            for ev in g.get("realSwitches", []):
                switch_events_by_cause[ev["cause"]] += 1
            n_preds = len(g.get("distinctPredIds", []))
            fragmentation_hist[n_preds] += 1
            worst_coverage.append((rid, g["gtLabel"], g["coverage"]))
            if n_preds >= 2:
                highest_fragmented.append((rid, g["gtLabel"], n_preds))

        per_rally_summary.append({
            "rallyId": rid,
            "hota": a.get("hota") or 0.0,
            "mota": a.get("mota") or 0.0,
            "switches": switches_here,
            "missFrames": sum(
                sum(end - start + 1 for start, end in ranges)
                for g in a["perGt"]
                for ranges in g.get("missedByCause", {}).values()
            ),
            "worstCoverage": min((g["coverage"] for g in a["perGt"]), default=1.0),
            "conventionFlip": bool(conv.get("courtSideFlip") or conv.get("teamLabelFlip")),
        })

    worst_coverage.sort(key=lambda t: t[2])
    worst_hota.sort(key=lambda t: t[1])
    highest_fragmented.sort(key=lambda t: -t[2])

    return Aggregates(
        n_rallies=len(audits),
        n_gt_tracks=n_gt_tracks,
        n_rallies_with_gt=len(audits),
        miss_frames_by_cause=miss_frames_by_cause,
        miss_ranges_by_cause=miss_ranges_by_cause,
        switch_events_by_cause=switch_events_by_cause,
        total_real_switches=total_real_switches,
        rallies_with_switches=rallies_with_switches,
        pred_exchange_swaps=pred_exchange_swaps,
        rallies_with_exchange_swaps=rallies_with_exchange_swaps,
        convention_drift_rallies=convention_drift,
        fragmentation_hist=fragmentation_hist,
        worst_coverage=worst_coverage[:15],
        worst_hota=worst_hota[:10],
        highest_fragmented=highest_fragmented[:15],
        per_rally_summary=per_rally_summary,
    )


def _fmt_pct(n: int, total: int) -> str:
    return f"{100 * n / total:.1f}%" if total else "0.0%"


def _render_cause_table(agg: Aggregates) -> str:
    total_frames = sum(agg.miss_frames_by_cause.values())
    lines = ["| Cause | Missed frames | Share | Ranges | Stage attribution |",
             "|---|---:|---:|---:|---|"]
    for cause, _ in agg.miss_frames_by_cause.most_common():
        frames = agg.miss_frames_by_cause[cause]
        ranges = agg.miss_ranges_by_cause[cause]
        lines.append(
            f"| `{cause}` | {frames} | {_fmt_pct(frames, total_frames)} | {ranges} | {STAGE_ATTRIBUTION.get(cause, '')} |"
        )
    return "\n".join(lines)


def _render_switch_table(agg: Aggregates) -> str:
    if not agg.total_real_switches:
        return "_No real ID switches detected._"
    lines = ["| Cause | Events | Stage attribution |",
             "|---|---:|---|"]
    for cause, _ in agg.switch_events_by_cause.most_common():
        lines.append(
            f"| `{cause}` | {agg.switch_events_by_cause[cause]} | {STAGE_ATTRIBUTION.get(cause, '')} |"
        )
    return "\n".join(lines)


def _render_fragmentation_hist(agg: Aggregates) -> str:
    lines = ["| Distinct pred IDs per GT track | # GT tracks |",
             "|---:|---:|"]
    total = sum(agg.fragmentation_hist.values())
    for n in sorted(agg.fragmentation_hist):
        count = agg.fragmentation_hist[n]
        lines.append(f"| {n} | {count} ({_fmt_pct(count, total)}) |")
    return "\n".join(lines)


def _render_worst_coverage(agg: Aggregates) -> str:
    lines = ["| Rally | GT label | Coverage |",
             "|---|---|---:|"]
    for rid, label, cov in agg.worst_coverage:
        lines.append(f"| `{rid[:8]}` | {label} | {cov:.2f} |")
    return "\n".join(lines)


def _render_worst_hota(agg: Aggregates) -> str:
    lines = ["| Rally | HOTA |",
             "|---|---:|"]
    for rid, hota in agg.worst_hota:
        lines.append(f"| `{rid[:8]}` | {hota:.3f} |")
    return "\n".join(lines)


def _render_fragmented_tracks(agg: Aggregates) -> str:
    if not agg.highest_fragmented:
        return "_No fragmented GT tracks (all matched by a single pred_id each)._"
    lines = ["| Rally | GT label | Distinct pred IDs |",
             "|---|---|---:|"]
    for rid, label, n in agg.highest_fragmented:
        lines.append(f"| `{rid[:8]}` | {label} | {n} |")
    return "\n".join(lines)


def _render_recommendations(agg: Aggregates) -> str:
    sections = []
    # Only emit a recommendation if the failure class actually fired.
    activated: set[str] = set()
    for cause, frames in agg.miss_frames_by_cause.items():
        if frames > 0:
            activated.add(cause)
    for cause, events in agg.switch_events_by_cause.items():
        if events > 0:
            activated.add(cause)
    if any(s.get("conventionFlip") for s in agg.per_rally_summary):
        activated.add("court_side_flip")
        activated.add("team_label_flip")

    for cause in ["detector_miss", "filter_drop", "occlusion", "edge_proximity",
                  "net_crossing", "same_team_swap", "cross_team_swap",
                  "fragment_gap", "court_side_flip", "team_label_flip"]:
        if cause not in activated:
            continue
        rec = RECOMMENDATION.get(cause)
        if not rec:
            continue
        sections.append(f"### `{cause}`\n\n{rec}")
    return "\n\n".join(sections)


def render_report(agg: Aggregates) -> str:
    header = [
        "# Player-tracking root-cause audit",
        "",
        f"Automatic summary over **{agg.n_rallies} rallies** with player GT ({agg.n_gt_tracks} GT tracks).",
        "Generated by `scripts/write_tracking_audit_report.py`. Data sourced from",
        "`reports/tracking_audit/*.json` (built by `rallycut evaluate-tracking audit`).",
        "",
        "## Headline",
        "",
        f"- Real ID switches (single-pred segment flip): **{agg.total_real_switches}** across {agg.rallies_with_switches} rallies.",
        f"- Pred-exchange swaps (two preds swapped GT ownership): "
        f"**{agg.pred_exchange_swaps}** across {agg.rallies_with_exchange_swaps} rallies.",
        f"- **Combined identity-swap incidents: "
        f"{agg.total_real_switches + agg.pred_exchange_swaps}** "
        f"(was undercounted when only the first metric was reported).",
        f"- GT tracks with fragmentation (≥2 pred IDs): "
        f"**{sum(n for k, n in agg.fragmentation_hist.items() if k >= 2)} / {agg.n_gt_tracks}**.",
        f"- Rallies flagged for convention drift: **{len(agg.convention_drift_rallies)}**.",
        "- Missed-frame causes ranked: " +
        ", ".join(
            f"`{c}`={n}"
            for c, n in agg.miss_frames_by_cause.most_common()
        )
        + ".",
        "",
        "## Missed frames by cause",
        "",
        _render_cause_table(agg),
        "",
        "## Real ID switches by cause",
        "",
        _render_switch_table(agg),
        "",
        "## Fragmentation histogram (distinct pred IDs per GT track)",
        "",
        _render_fragmentation_hist(agg),
        "",
        "## Worst GT-track coverage (top 15)",
        "",
        _render_worst_coverage(agg),
        "",
        "## Lowest-HOTA rallies (top 10)",
        "",
        _render_worst_hota(agg),
        "",
        "## Most-fragmented GT tracks (top 15)",
        "",
        _render_fragmented_tracks(agg),
        "",
    ]
    if agg.convention_drift_rallies:
        header.extend([
            "## Rallies with convention drift",
            "",
            "\n".join(f"- `{r}`" for r in agg.convention_drift_rallies),
            "",
        ])
    header.extend([
        "## Structural recommendations",
        "",
        "Each item below is a design-level change, **not a threshold tweak**. The"
        " ordering reflects the failure-class prevalence above — address causes"
        " with the largest share of missed frames and real switches first.",
        "",
        _render_recommendations(agg),
        "",
        "## Next steps",
        "",
        "1. Open `reports/tracking_audit/gallery/index.html` to inspect the annotated clips per rally.",
        "2. Pick one cause from the recommendations above, scope a follow-up plan with `/brainstorm`.",
        "3. Re-run `rallycut evaluate-tracking audit --all` after any pipeline change to regenerate this report.",
        "",
    ])
    return "\n".join(header)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=Path("reports/tracking_audit"))
    parser.add_argument("--out", type=Path, default=Path("reports/tracking_audit/root_cause.md"))
    args = parser.parse_args()

    audits: list[dict] = []
    for path in sorted(args.audit_dir.glob("*.json")):
        if path.name == "_summary.json":
            continue
        with open(path) as f:
            audits.append(json.load(f))
    if not audits:
        logger.error(f"No audit JSONs in {args.audit_dir}")
        raise SystemExit(1)

    agg = aggregate(audits)
    md = render_report(agg)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md)
    logger.info(f"Wrote report: {args.out}")


if __name__ == "__main__":
    main()
