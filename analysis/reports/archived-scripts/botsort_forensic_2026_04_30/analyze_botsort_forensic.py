"""Analyze BoxMOT forensic sidecars to produce one mechanism statement per
panel error rally.

Reads JSONL sidecars from `analysis/reports/botsort_forensic_2026_04_30/`
written by `InstrumentedBotSort`, plus each rally's `primary_track_ids` and
`positions_json` from the DB. For each frame, applies five falsifiable rules
to flag suspicious BoxMOT assignment decisions. Emits a Markdown report
pairing each panel error with one mechanism statement (or honest "no
observable mechanism" when no rule fires) and a null-hypothesis check on
the controls.

The five rules (defined operationally so they're falsifiable):

  F-IoU-cliff:     a primary track inherited a det whose IoU vs that
                   track's prediction was < IOU_CLIFF_LOW (0.10) but the
                   same det had IoU > IOU_CLIFF_HIGH (0.30) with at least
                   one OTHER primary track's prediction. The Hungarian
                   "stole" a detection that geometrically belonged to a
                   different player.

  F-Embedding-only: a primary track inherited a det whose IoU cost was
                   > IOU_FAILED (0.90, i.e. effectively no IoU support)
                   but the gated embedding distance was < EMB_LOW (0.25),
                   so the fused min(IoU, emb) match was forced by ReID
                   alone with no spatial agreement.

  F-Lost-reactivate: a primary track was re-activated this frame after
                   being in `lost_pre` for ≥ MIN_LOST_FRAMES (5), AND the
                   matched det is INSIDE another primary's predicted bbox
                   (>OVERLAP_REACTIVATE 0.30 IoU).

  F-Second-pass:   a primary track was matched in the second-pass IoU-only
                   association (where there is no embedding gate) against a
                   det that overlaps another primary's prediction by
                   >SECOND_PASS_OVERLAP (0.30) IoU.

  F-Greedy:        Hungarian picked an assignment for a primary track that
                   had a strictly better-scoring partner detection
                   (delta > GREEDY_DELTA, 0.10) which got claimed by another
                   higher-priority track.

A rule is "candidate-shippable" if it fires on ≥1 error and 0 controls.
For any rule that fires on a control, ALL error claims based on that rule
are downgraded to "no observable mechanism."

Usage:
    uv run python -u scripts/analyze_botsort_forensic.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

_ANALYSIS_DIR = Path(__file__).resolve().parents[1]
if str(_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS_DIR))

from rallycut.evaluation.tracking.db import get_connection  # noqa: E402
from scripts.forensic_capture_panel import (  # noqa: E402
    OUT_DIR,
    PANEL,
    resolve_panel_to_rally_ids,
)

REPORT_PATH = OUT_DIR / "REPORT.md"

# Rule thresholds (locked by definition, not tunable knobs).
IOU_CLIFF_LOW = 0.10  # max IoU vs own prediction below which match is suspect
IOU_CLIFF_HIGH = 0.30  # min IoU vs another's prediction above which det was "stolen"
IOU_FAILED = 0.90  # IoU cost above this counts as "no IoU support"
EMB_LOW = 0.25  # gated embedding cost below this = "forced by ReID alone"
MIN_LOST_FRAMES = 5  # min lost duration before reactivate is suspect
OVERLAP_REACTIVATE = 0.30  # det overlap with peer required for F-Lost-reactivate
SECOND_PASS_OVERLAP = 0.30  # det overlap with peer required for F-Second-pass
GREEDY_DELTA = 0.10  # cost gap that defines "strictly better partner"


# ---------- helpers ----------


def _bbox_iou(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a_area + b_area - inter
    if union <= 0:
        return 0.0
    return inter / union


# ---------- data ----------


@dataclass
class FrameRecord:
    f: int
    n_dets_first: int
    n_strack_pool: int
    active_pre: list[dict[str, Any]]
    lost_pre: list[dict[str, Any]]
    active_post: list[dict[str, Any]]
    strack_pool_predicted: list[dict[str, Any]]
    dets_first: list[dict[str, Any]]
    dets_second: list[dict[str, Any]]
    first_assoc: dict[str, Any]
    second_assoc: dict[str, Any]
    matched_via: dict[int, str]
    new_track_ids: list[int]
    dropped_track_ids: list[int]


@dataclass
class SidecarData:
    rally_tag: str
    config: dict[str, Any]
    frames: list[FrameRecord]


@dataclass
class FlagEvent:
    rule: str
    frame: int
    track_id: int
    det_ind: int
    detail: str
    candidate_intervention: str


@dataclass
class RallyAnalysis:
    rally_tag: str
    is_error: bool
    desc: str
    primary_track_ids: list[int]
    sidecar_path: Path
    flags: list[FlagEvent] = field(default_factory=list)
    n_frames: int = 0
    n_first_pass_matches: int = 0
    n_second_pass_matches: int = 0
    n_reactivations: int = 0
    n_marked_lost: int = 0
    n_new_tracks: int = 0
    error: str | None = None


# ---------- sidecar I/O ----------


def load_sidecar(path: Path) -> SidecarData:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"empty sidecar: {path}")
    meta = json.loads(lines[0])
    if meta.get("type") != "meta":
        raise ValueError(f"first line is not meta: {path}")
    frames: list[FrameRecord] = []
    for line in lines[1:]:
        if not line.strip():
            continue
        f = json.loads(line)
        if f.get("type") != "frame":
            continue
        # JSON keys in matched_via are strings; convert back to int.
        mv_raw = f.get("matched_via", {}) or {}
        matched_via = {int(k): v for k, v in mv_raw.items()}
        frames.append(FrameRecord(
            f=int(f.get("f", -1)),
            n_dets_first=int(f.get("n_dets_first", 0)),
            n_strack_pool=int(f.get("n_strack_pool", 0)),
            active_pre=f.get("active_pre", []) or [],
            lost_pre=f.get("lost_pre", []) or [],
            active_post=f.get("active_post", []) or [],
            strack_pool_predicted=f.get("strack_pool_predicted", []) or [],
            dets_first=f.get("dets_first", []) or [],
            dets_second=f.get("dets_second", []) or [],
            first_assoc=f.get("first_assoc", {}) or {},
            second_assoc=f.get("second_assoc", {}) or {},
            matched_via=matched_via,
            new_track_ids=f.get("new_track_ids", []) or [],
            dropped_track_ids=f.get("dropped_track_ids", []) or [],
        ))
    return SidecarData(
        rally_tag=str(meta.get("rally_tag", "")),
        config=meta.get("config", {}) or {},
        frames=frames,
    )


def fetch_primary_track_ids(rally_id: str) -> list[int]:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT primary_track_ids FROM player_tracks WHERE rally_id = %s",
            [rally_id],
        )
        row = cur.fetchone()
        if not row or row[0] is None:
            return []
        return [int(t) for t in cast("list[Any]", row[0])]


# ---------- rule application ----------


def _detection_lookup(frame: FrameRecord) -> dict[int, list[float]]:
    """Map det_ind → xyxy bbox for both first-pass and second-pass dets."""
    out: dict[int, list[float]] = {}
    for d in frame.dets_first:
        out[int(d["det_ind"])] = [
            float(d["x1"]), float(d["y1"]), float(d["x2"]), float(d["y2"]),
        ]
    for d in frame.dets_second:
        out[int(d["det_ind"])] = [
            float(d["x1"]), float(d["y1"]), float(d["x2"]), float(d["y2"]),
        ]
    return out


def _predicted_bbox_lookup(frame: FrameRecord) -> dict[int, list[float]]:
    """Map track_id → predicted xyxy from strack_pool_predicted."""
    out: dict[int, list[float]] = {}
    for entry in frame.strack_pool_predicted:
        out[int(entry["id"])] = [float(x) for x in entry.get("xyxy_pred", [])]
    return out


@dataclass
class RallyActivity:
    n_first_pass_matches: int = 0
    n_second_pass_matches: int = 0
    n_reactivations: int = 0
    n_marked_lost: int = 0
    n_new_tracks: int = 0


def apply_rules(
    sidecar: SidecarData,
    primary_track_ids: list[int],
    activity: RallyActivity,
) -> list[FlagEvent]:
    """Apply F-* rules to every frame; return all flag events for primary tracks.

    The analyzer flags only events touching `primary_track_ids` so noise on
    spectators / non-primary tracks doesn't dominate. Also accumulates
    per-rally activity counters so 'no observable mechanism' cards can show
    what activity DID happen on primary tracks.
    """
    flags: list[FlagEvent] = []
    primary_set = set(primary_track_ids)
    activity.n_new_tracks = 0
    activity.n_marked_lost = 0
    activity.n_reactivations = 0
    activity.n_first_pass_matches = 0
    activity.n_second_pass_matches = 0
    for frame in sidecar.frames:
        # Tally activity touching primary tracks.
        for tid_str, via in frame.matched_via.items():
            tid = int(tid_str)
            if tid not in primary_set:
                continue
            if via.startswith("first"):
                activity.n_first_pass_matches += 1
            elif via.startswith("second"):
                activity.n_second_pass_matches += 1
            if via.endswith("reactivated"):
                activity.n_reactivations += 1
        for new_tid in frame.new_track_ids:
            if int(new_tid) in primary_set:
                activity.n_new_tracks += 1
        for lost_tid in (frame.second_assoc.get("marked_lost") or []):
            if int(lost_tid) in primary_set:
                activity.n_marked_lost += 1
        first = frame.first_assoc
        second = frame.second_assoc
        track_pool_ids = [int(x) for x in first.get("track_pool_ids", [])]
        ious_dists = first.get("ious_dists", [])  # (T, D)
        emb_gated = first.get("emb_dists_gated", [])  # (T, D)
        first_matches = first.get("matches", [])  # [(t_pool_idx, det_idx)]
        det_ind_lookup = _detection_lookup(frame)
        pred_lookup = _predicted_bbox_lookup(frame)

        # Map track id → its row index in the cost matrices (for first pass).
        # Map detection idx in the matching matrix to det_ind.
        first_dets_meta = frame.dets_first
        det_idx_to_det_ind: dict[int, int] = {
            i: int(first_dets_meta[i]["det_ind"])
            for i in range(len(first_dets_meta))
        }

        # First-pass per-match analysis.
        for it_idx, idet_idx in first_matches:
            it_idx = int(it_idx)
            idet_idx = int(idet_idx)
            if it_idx >= len(track_pool_ids) or idet_idx >= len(first_dets_meta):
                continue
            track_id = track_pool_ids[it_idx]
            if track_id not in primary_set:
                continue
            det_ind = det_idx_to_det_ind[idet_idx]
            iou_cost = float(ious_dists[it_idx][idet_idx]) if ious_dists else 1.0
            emb_cost = (
                float(emb_gated[it_idx][idet_idx]) if emb_gated else 1.0
            )
            det_bbox = det_ind_lookup.get(det_ind, [])

            # F-IoU-cliff: weak IoU vs own prediction, strong IoU vs peer.
            own_iou = 1.0 - iou_cost  # ious_dists is 1 - iou
            if own_iou < IOU_CLIFF_LOW:
                peer_overlap_max = 0.0
                peer_id_max = -1
                for peer_id in primary_set - {track_id}:
                    peer_box = pred_lookup.get(peer_id, [])
                    if not peer_box or not det_bbox:
                        continue
                    iou = _bbox_iou(peer_box, det_bbox)
                    if iou > peer_overlap_max:
                        peer_overlap_max = iou
                        peer_id_max = peer_id
                if peer_overlap_max > IOU_CLIFF_HIGH:
                    flags.append(FlagEvent(
                        rule="F-IoU-cliff",
                        frame=frame.f,
                        track_id=track_id,
                        det_ind=det_ind,
                        detail=(
                            f"own IoU={own_iou:.3f}; peer t{peer_id_max} "
                            f"IoU={peer_overlap_max:.3f}; emb_gated={emb_cost:.3f}"
                        ),
                        candidate_intervention=(
                            "Gate the fused cost in BotSort._first_association "
                            "by requiring own-track IoU > " f"{IOU_CLIFF_LOW:.2f} "
                            "before allowing the embedding to win the match."
                        ),
                    ))

            # F-Embedding-only: no IoU support, but ReID forced match.
            if iou_cost > IOU_FAILED and emb_cost < EMB_LOW:
                flags.append(FlagEvent(
                    rule="F-Embedding-only",
                    frame=frame.f,
                    track_id=track_id,
                    det_ind=det_ind,
                    detail=(
                        f"iou_cost={iou_cost:.3f} > {IOU_FAILED:.2f}; "
                        f"emb_gated={emb_cost:.3f} < {EMB_LOW:.2f}"
                    ),
                    candidate_intervention=(
                        "Lower BotSort.appearance_thresh (currently 0.30) so "
                        "low-IoU embedding-only matches don't pass the gate."
                    ),
                ))

            # F-Lost-reactivate: track was lost ≥ MIN_LOST_FRAMES + match overlaps peer.
            matched_via_label = frame.matched_via.get(track_id, "")
            if matched_via_label == "first_reactivated":
                lost_entry = next(
                    (e for e in frame.lost_pre if int(e["id"]) == track_id),
                    None,
                )
                if lost_entry is not None:
                    last_seen = int(lost_entry.get("frame_id", -1))
                    lost_dur = max(0, frame.f - last_seen)
                    if lost_dur >= MIN_LOST_FRAMES:
                        peer_overlap_max = 0.0
                        peer_id_max = -1
                        for peer_id in primary_set - {track_id}:
                            peer_box = pred_lookup.get(peer_id, [])
                            if not peer_box or not det_bbox:
                                continue
                            iou = _bbox_iou(peer_box, det_bbox)
                            if iou > peer_overlap_max:
                                peer_overlap_max = iou
                                peer_id_max = peer_id
                        if peer_overlap_max > OVERLAP_REACTIVATE:
                            flags.append(FlagEvent(
                                rule="F-Lost-reactivate",
                                frame=frame.f,
                                track_id=track_id,
                                det_ind=det_ind,
                                detail=(
                                    f"lost {lost_dur} frames; reactivated against "
                                    f"det overlapping peer t{peer_id_max} by "
                                    f"{peer_overlap_max:.3f}"
                                ),
                                candidate_intervention=(
                                    "When reactivating a lost track against a "
                                    "detection that overlaps another active "
                                    "primary's predicted bbox by > "
                                    f"{OVERLAP_REACTIVATE:.2f}, defer "
                                    "reactivation by one frame to let the "
                                    "active primary win the second-pass IoU."
                                ),
                            ))

            # F-Greedy: a strictly better partner existed AND is unclaimed.
            # Hungarian is globally optimal, so a flag here means the
            # Hungarian's tradeoff was severe — track was matched to det X
            # while det Y had Δ > GREEDY_DELTA lower cost AND was left in
            # u_detection (no track claimed it).
            if ious_dists and emb_gated:
                fused_row = first.get("fused", [])
                if fused_row and it_idx < len(fused_row):
                    row = fused_row[it_idx]
                    matched_cost = float(row[idet_idx])
                    matched_det_idxs = {int(m[1]) for m in first_matches}
                    unmatched_det_idxs = (
                        set(range(len(row))) - matched_det_idxs
                    )
                    if unmatched_det_idxs:
                        best_unmatched_idx = min(
                            unmatched_det_idxs, key=lambda i: row[i],
                        )
                        best_cost = float(row[best_unmatched_idx])
                        if matched_cost - best_cost > GREEDY_DELTA:
                            best_det_ind = det_idx_to_det_ind.get(
                                best_unmatched_idx, -1,
                            )
                            flags.append(FlagEvent(
                                rule="F-Greedy",
                                frame=frame.f,
                                track_id=track_id,
                                det_ind=det_ind,
                                detail=(
                                    f"matched det_ind={det_ind} cost={matched_cost:.3f}; "
                                    f"unclaimed det_ind={best_det_ind} cost={best_cost:.3f} "
                                    f"(Δ={matched_cost - best_cost:.3f}) was available"
                                ),
                                candidate_intervention=(
                                    "Add a post-Hungarian re-assignment pass: "
                                    "for any primary track whose matched fused "
                                    "cost exceeds the best UNCLAIMED det's "
                                    f"cost by > {GREEDY_DELTA:.2f}, swap."
                                ),
                            ))

        # F-Second-pass: primary matched in the second pass + det overlaps peer.
        sec_matches = second.get("matches", [])
        sec_track_ids = [int(x) for x in second.get("r_tracked_ids", [])]
        sec_dets_meta = frame.dets_second
        sec_det_idx_to_ind: dict[int, int] = {
            i: int(sec_dets_meta[i]["det_ind"])
            for i in range(len(sec_dets_meta))
        }
        for it_idx, idet_idx in sec_matches:
            it_idx = int(it_idx)
            idet_idx = int(idet_idx)
            if it_idx >= len(sec_track_ids) or idet_idx >= len(sec_dets_meta):
                continue
            track_id = sec_track_ids[it_idx]
            if track_id not in primary_set:
                continue
            det_ind = sec_det_idx_to_ind[idet_idx]
            det_bbox = det_ind_lookup.get(det_ind, [])
            peer_overlap_max = 0.0
            peer_id_max = -1
            for peer_id in primary_set - {track_id}:
                peer_box = pred_lookup.get(peer_id, [])
                if not peer_box or not det_bbox:
                    continue
                iou = _bbox_iou(peer_box, det_bbox)
                if iou > peer_overlap_max:
                    peer_overlap_max = iou
                    peer_id_max = peer_id
            if peer_overlap_max > SECOND_PASS_OVERLAP:
                flags.append(FlagEvent(
                    rule="F-Second-pass",
                    frame=frame.f,
                    track_id=track_id,
                    det_ind=det_ind,
                    detail=(
                        f"second-pass IoU-only match; det overlaps peer "
                        f"t{peer_id_max} by {peer_overlap_max:.3f}"
                    ),
                    candidate_intervention=(
                        "In BotSort._second_association, mask out detections "
                        "whose IoU with another active primary's predicted "
                        f"bbox > {SECOND_PASS_OVERLAP:.2f} before running the "
                        "IoU-only Hungarian."
                    ),
                ))

    return flags


# ---------- reporting ----------


def write_report(analyses: list[RallyAnalysis]) -> None:
    # Tally per-rule fires across errors vs controls.
    rule_fires: dict[str, dict[str, int]] = defaultdict(
        lambda: {"error_rallies": 0, "control_rallies": 0},
    )
    for a in analyses:
        rules_in_rally = {f.rule for f in a.flags}
        for rule in rules_in_rally:
            if a.is_error:
                rule_fires[rule]["error_rallies"] += 1
            else:
                rule_fires[rule]["control_rallies"] += 1

    # A rule is candidate-shippable if it fires on ≥1 error + 0 controls.
    rule_status: dict[str, str] = {}
    for rule, counts in rule_fires.items():
        if counts["error_rallies"] >= 1 and counts["control_rallies"] == 0:
            rule_status[rule] = "CANDIDATE"
        elif counts["control_rallies"] >= 1:
            rule_status[rule] = "REJECTED (fires on control)"
        else:
            rule_status[rule] = "INERT (fires on neither)"

    lines: list[str] = []
    lines.append("# BoT-SORT forensic mechanism report")
    lines.append("")
    lines.append(
        f"Generated from sidecars in `{OUT_DIR.relative_to(_ANALYSIS_DIR)}`. "
        "Each error rally gets one falsifiable mechanism statement (or honest "
        "\"no observable mechanism\") with the supporting frame/track/det "
        "evidence. Controls run the same rules as a null-hypothesis check."
    )
    lines.append("")

    # High-level bucket counts.
    errors = [a for a in analyses if a.is_error and not a.error]
    controls = [a for a in analyses if not a.is_error and not a.error]
    skipped = [a for a in analyses if a.error]
    errors_with_mechanism = [a for a in errors if a.flags]
    errors_no_mechanism = [a for a in errors if not a.flags]
    controls_clean = [a for a in controls if not a.flags]
    controls_fired = [a for a in controls if a.flags]

    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Errors with observable mechanism: **{len(errors_with_mechanism)} / {len(errors)}**")
    for a in errors_with_mechanism:
        rules = sorted({f.rule for f in a.flags})
        lines.append(f"  - `{a.rally_tag}` — {', '.join(rules)}")
    lines.append(f"- Errors with NO observable mechanism: **{len(errors_no_mechanism)} / {len(errors)}**")
    for a in errors_no_mechanism:
        lines.append(f"  - `{a.rally_tag}` — origin downstream of BoT-SORT")
    lines.append(f"- Controls clean (null hypothesis OK): **{len(controls_clean)} / {len(controls)}**")
    if controls_fired:
        lines.append(f"- Controls fired (rule INVALIDATED): **{len(controls_fired)} / {len(controls)}**")
        for a in controls_fired:
            rules = sorted({f.rule for f in a.flags})
            lines.append(f"  - `{a.rally_tag}` — {', '.join(rules)}")
    if skipped:
        lines.append(f"- Skipped: **{len(skipped)}**")
        for a in skipped:
            lines.append(f"  - `{a.rally_tag}` — {a.error}")
    lines.append("")

    # Rule status table.
    lines.append("## Rule status across panel")
    lines.append("")
    lines.append("| Rule | Errors fired | Controls fired | Status |")
    lines.append("|---|---|---|---|")
    for rule, counts in sorted(rule_fires.items()):
        lines.append(
            f"| {rule} | {counts['error_rallies']} / "
            f"{sum(1 for a in analyses if a.is_error)} | "
            f"{counts['control_rallies']} / "
            f"{sum(1 for a in analyses if not a.is_error)} | "
            f"{rule_status[rule]} |"
        )
    if not rule_fires:
        lines.append("| (no rules fired) | 0 | 0 | — |")
    lines.append("")

    # Per-rally cards.
    lines.append("## Per-rally cards")
    lines.append("")
    for a in analyses:
        kind = "ERROR" if a.is_error else "CONTROL"
        lines.append(f"### {a.rally_tag}  [{kind}]  — {a.desc}")
        lines.append("")
        if a.error:
            lines.append(f"**SKIPPED:** {a.error}")
            lines.append("")
            continue
        lines.append(
            f"- frames: {a.n_frames}, primary track ids: "
            f"{', '.join(f't{t}' for t in a.primary_track_ids) or '—'}"
        )
        lines.append(
            f"- activity on primary tracks: "
            f"{a.n_first_pass_matches} first-pass matches, "
            f"{a.n_second_pass_matches} second-pass matches, "
            f"{a.n_reactivations} reactivations, "
            f"{a.n_marked_lost} marked-lost events, "
            f"{a.n_new_tracks} new-track events"
        )
        lines.append("")
        if not a.flags:
            if a.is_error:
                lines.append(
                    "**Mechanism:** NO observable mechanism. Across all "
                    f"{a.n_frames} frames, none of "
                    "F-IoU-cliff / F-Embedding-only / F-Lost-reactivate / "
                    "F-Second-pass / F-Greedy fired on any of the rally's "
                    f"{len(a.primary_track_ids)} primary tracks. The "
                    "BoT-SORT decisions appear internally consistent: each "
                    "primary's matches stayed within IoU/embedding gates "
                    "and no peer-stealing pattern is visible. Identity "
                    "error therefore originates downstream of BoT-SORT "
                    "(in cross-rally matching, primary-track filtering, "
                    "or post-processing) or at a granularity below the "
                    "per-frame state BoxMOT exposes (e.g. inside the "
                    "Kalman gating, or in the earlier YOLO detection step)."
                )
            else:
                lines.append("Null hypothesis OK: no rule fired.")
        else:
            # Group by rule for readability; use earliest fire of each rule.
            by_rule: dict[str, list[FlagEvent]] = defaultdict(list)
            for f in a.flags:
                by_rule[f.rule].append(f)
            primary_flag = min(a.flags, key=lambda f: (f.frame, f.rule))
            primary_status = rule_status.get(primary_flag.rule, "")
            if a.is_error:
                lines.append(
                    f"**Mechanism:** **{primary_flag.rule}** at "
                    f"f={primary_flag.frame}, track t{primary_flag.track_id} "
                    f"inherited det_ind={primary_flag.det_ind}. "
                    f"{primary_flag.detail}."
                )
                lines.append("")
                lines.append(
                    f"**Candidate intervention:** "
                    f"{primary_flag.candidate_intervention}"
                )
                lines.append("")
                lines.append(
                    f"**Rule status:** {primary_status}"
                    + (
                        " — claim downgraded to 'no observable mechanism'."
                        if primary_status.startswith("REJECTED")
                        else ""
                    )
                )
            else:
                lines.append(
                    f"**Control fired** (rule unsafe): {primary_flag.rule} "
                    f"at f={primary_flag.frame}, track t{primary_flag.track_id}. "
                    f"{primary_flag.detail}."
                )
            lines.append("")
            lines.append(
                "<details><summary>All flags this rally "
                f"({len(a.flags)} total)</summary>"
            )
            lines.append("")
            lines.append("| Rule | Frame | Track | Det | Detail |")
            lines.append("|---|---|---|---|---|")
            for f in sorted(a.flags, key=lambda f: (f.frame, f.rule)):
                lines.append(
                    f"| {f.rule} | {f.frame} | t{f.track_id} | "
                    f"{f.det_ind} | {f.detail} |"
                )
            lines.append("")
            lines.append("</details>")
        lines.append("")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT_PATH} ({len(lines)} lines)", flush=True)


# ---------- driver ----------


def main() -> None:
    rallies = resolve_panel_to_rally_ids()
    print(f"Resolved {len(rallies)} of {len(PANEL)} panel entries", flush=True)

    analyses: list[RallyAnalysis] = []
    for rally in rallies:
        sidecar_path = OUT_DIR / f"{rally.rally_tag}.jsonl"
        analysis = RallyAnalysis(
            rally_tag=rally.rally_tag,
            is_error=rally.is_error,
            desc=rally.desc,
            primary_track_ids=fetch_primary_track_ids(rally.rally_id),
            sidecar_path=sidecar_path,
        )
        if not sidecar_path.exists():
            analysis.error = f"sidecar missing: {sidecar_path}"
            analyses.append(analysis)
            print(f"  ! {rally.rally_tag} sidecar missing", flush=True)
            continue
        try:
            sidecar = load_sidecar(sidecar_path)
        except Exception as exc:
            analysis.error = f"failed to load sidecar: {exc}"
            analyses.append(analysis)
            print(f"  ! {rally.rally_tag} load failed: {exc}", flush=True)
            continue
        analysis.n_frames = len(sidecar.frames)
        activity = RallyActivity()
        analysis.flags = apply_rules(
            sidecar, analysis.primary_track_ids, activity,
        )
        analysis.n_first_pass_matches = activity.n_first_pass_matches
        analysis.n_second_pass_matches = activity.n_second_pass_matches
        analysis.n_reactivations = activity.n_reactivations
        analysis.n_marked_lost = activity.n_marked_lost
        analysis.n_new_tracks = activity.n_new_tracks
        kind = "E" if rally.is_error else "C"
        n_unique_rules = len({f.rule for f in analysis.flags})
        print(
            f"  [{kind}] {rally.rally_tag:<14}  frames={analysis.n_frames}  "
            f"flags={len(analysis.flags)}  rules_fired={n_unique_rules}",
            flush=True,
        )
        analyses.append(analysis)

    write_report(analyses)


if __name__ == "__main__":
    main()
