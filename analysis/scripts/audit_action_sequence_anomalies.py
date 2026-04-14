"""Audit action-sequence anomalies across GT rallies.

Diagnostic script for the 2026-04-14 action-detection review. Loads every
rally with an action_ground_truth_json, classifies each predicted sequence
against a volleyball-grammar checklist, tags each anomaly with the stage
most likely responsible, and separates low-quality inputs (bad angle /
crowded scene / ball-tracking dropout) from genuine pipeline errors.

No pipeline re-run. We read the stored `actions_json` and
`action_ground_truth_json` columns and compare them. The goal is to
quantify *how often* each failure mode fires and *which stage* produced
it — not to re-measure F1. For F1-style evaluation use
`scripts/eval_action_detection.py`.

Usage:

    cd analysis
    uv run python scripts/audit_action_sequence_anomalies.py \\
        --output outputs/action_anomaly_audit_2026_04_14.md
    uv run python scripts/audit_action_sequence_anomalies.py \\
        --video 211e2a4c-c9a3-4438-9b0c-bea4e7555ad0
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.table import Table

from rallycut.evaluation.db import get_connection

console = Console()

# Anomaly buckets — each rally is tagged with zero or more of these.
BUCKETS = [
    "double_serve",
    "opposite_side_double",
    "same_side_double",
    "over_three_same_side",
    "unknown_rewritten",
    "missed_contact_cluster",
    "court_side_flip",
    "action_type_mismatch",
]


@dataclass
class RallyRow:
    rally_id: str
    video_id: str
    rally_order: int
    start_ms: int
    end_ms: int
    pred_actions: list[dict]
    gt_actions: list[dict]
    # Per-rally quality proxies.
    detection_rate: float | None
    avg_player_count: float | None
    avg_ball_conf: float | None
    frame_count: int
    fps: float
    court_split_y: float | None
    # Video-level flag for low-quality videos (set after histogramming).
    low_quality: bool = False
    # Per-rally anomaly tags.
    flags: dict[str, list[str]] = field(default_factory=dict)

    @property
    def has_any_flag(self) -> bool:
        return any(v for v in self.flags.values())


def _load_rallies(video_id: str | None = None) -> list[RallyRow]:
    """Fetch every rally that has action GT plus its quality proxies.

    Joins player_tracks to pull per-rally quality indicators in one trip.
    Ball confidence is computed Python-side from the ball_positions_json
    blob (median of per-frame confidences) because Postgres jsonb_path
    access would balloon the query.
    """
    where = ["pt.action_ground_truth_json IS NOT NULL"]
    params: list[str] = []
    if video_id:
        where.append("r.video_id = %s")
        params.append(video_id)

    query = f"""
        SELECT
            r.id, r.video_id, r."order", r.start_ms, r.end_ms,
            pt.actions_json, pt.action_ground_truth_json,
            pt.detection_rate, pt.avg_player_count,
            pt.ball_positions_json,
            pt.frame_count, pt.fps, pt.court_split_y
        FROM rallies r
        JOIN player_tracks pt ON pt.rally_id = r.id
        WHERE {" AND ".join(where)}
        ORDER BY r.video_id, r."order"
    """

    out: list[RallyRow] = []
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(query, params)
        for row in cur.fetchall():
            (rid, vid, order, start_ms, end_ms, actions_json, gt_json,
             det_rate, avg_pc, ball_json, frame_count, fps, court_split_y) = row

            if isinstance(actions_json, str):
                actions_json = json.loads(actions_json)
            if isinstance(gt_json, str):
                gt_json = json.loads(gt_json)
            if isinstance(ball_json, str):
                ball_json = json.loads(ball_json)

            pred = (actions_json or {}).get("actions") or []
            gt = gt_json or []

            ball_conf = _median_ball_conf(ball_json)

            out.append(RallyRow(
                rally_id=rid,
                video_id=vid,
                rally_order=order or 0,
                start_ms=start_ms or 0,
                end_ms=end_ms or 0,
                pred_actions=pred,
                gt_actions=gt,
                detection_rate=det_rate,
                avg_player_count=avg_pc,
                avg_ball_conf=ball_conf,
                frame_count=frame_count or 0,
                fps=fps or 30.0,
                court_split_y=court_split_y,
            ))
    return out


def _median_ball_conf(ball_json: list[dict] | None) -> float | None:
    """Median ball detection confidence. None if no ball positions stored."""
    if not ball_json:
        return None
    confs = [
        float(b["confidence"]) for b in ball_json
        if isinstance(b, dict) and b.get("confidence") is not None
    ]
    if not confs:
        return None
    return statistics.median(confs)


def _tag_low_quality(rallies: list[RallyRow]) -> None:
    """Flag bottom-10% rallies on any quality proxy as low_quality.

    We deliberately flag the *rally* (not the video) so that one bad
    rally inside an otherwise-clean video is still excluded from the
    clean bucket. Separately we log which videos have the densest
    low-quality share so the user can eyeball them.
    """
    if not rallies:
        return

    det_rates = sorted(r.detection_rate for r in rallies if r.detection_rate is not None)
    player_counts = sorted(
        r.avg_player_count for r in rallies if r.avg_player_count is not None
    )
    ball_confs = sorted(r.avg_ball_conf for r in rallies if r.avg_ball_conf is not None)

    def _p10(xs: list[float]) -> float | None:
        if len(xs) < 10:
            return None
        return xs[len(xs) // 10]

    det_threshold = _p10(det_rates)
    player_threshold = _p10(player_counts)
    ball_threshold = _p10(ball_confs)

    for r in rallies:
        reasons = []
        if det_threshold is not None and r.detection_rate is not None and r.detection_rate < det_threshold:
            reasons.append(f"detection_rate={r.detection_rate:.3f}<p10={det_threshold:.3f}")
        if player_threshold is not None and r.avg_player_count is not None and r.avg_player_count < player_threshold:
            reasons.append(
                f"avg_player_count={r.avg_player_count:.2f}<p10={player_threshold:.2f}"
            )
        if ball_threshold is not None and r.avg_ball_conf is not None and r.avg_ball_conf < ball_threshold:
            reasons.append(
                f"ball_conf={r.avg_ball_conf:.2f}<p10={ball_threshold:.2f}"
            )
        if reasons:
            r.low_quality = True
            r.flags.setdefault("quality_reason", []).extend(reasons)


# --------------------------------------------------------------------------- #
# Anomaly detectors                                                           #
# --------------------------------------------------------------------------- #


def _match_pred_to_gt(
    pred: list[dict], gt: list[dict], tol_frames: int = 8,
) -> list[tuple[int | None, int | None]]:
    """Pair predicted contacts to GT contacts by nearest-frame within tol.

    Greedy nearest-neighbor match. Returns a list of (pred_idx, gt_idx)
    where either side may be None (unmatched). Unmatched preds indicate
    FPs; unmatched GTs indicate FNs.
    """
    used_gt: set[int] = set()
    pairs: list[tuple[int | None, int | None]] = []
    for pi, p in enumerate(pred):
        pf = p.get("frame", -1)
        best_gi = None
        best_d = tol_frames + 1
        for gi, g in enumerate(gt):
            if gi in used_gt:
                continue
            gf = g.get("frame", -1)
            d = abs(pf - gf)
            if d <= tol_frames and d < best_d:
                best_d = d
                best_gi = gi
        if best_gi is not None:
            used_gt.add(best_gi)
            pairs.append((pi, best_gi))
        else:
            pairs.append((pi, None))
    for gi in range(len(gt)):
        if gi not in used_gt:
            pairs.append((None, gi))
    return pairs


def _flag_double_serve(r: RallyRow) -> list[str]:
    """Every predicted `serve` past the first one is illegal."""
    serves = [a for a in r.pred_actions if a.get("action") == "serve"]
    if len(serves) <= 1:
        return []
    frames = [str(a.get("frame")) for a in serves]
    return [f"{len(serves)} serves at frames {','.join(frames)}"]


def _flag_consecutive(
    r: RallyRow, opposite_sides: bool,
) -> list[str]:
    """Flag adjacent predictions with the same action_type.

    - opposite_sides=True  → illegal (can't dig→dig across the net)
    - opposite_sides=False → legal in beach (double-touch dig/set) but
      still worth tracking as a sanity signal
    """
    msgs: list[str] = []
    for i in range(1, len(r.pred_actions)):
        prev, cur = r.pred_actions[i - 1], r.pred_actions[i]
        if prev.get("action") != cur.get("action"):
            continue
        if prev.get("action") == "serve":  # handled by double_serve
            continue
        same_side = prev.get("courtSide") == cur.get("courtSide")
        if same_side and not opposite_sides:
            msgs.append(
                f"{prev.get('action')} at frames "
                f"{prev.get('frame')}→{cur.get('frame')} same-side "
                f"({prev.get('courtSide')})"
            )
        elif not same_side and opposite_sides:
            msgs.append(
                f"{prev.get('action')} at frames "
                f"{prev.get('frame')}→{cur.get('frame')} opposite-sides "
                f"({prev.get('courtSide')}→{cur.get('courtSide')})"
            )
    return msgs


def _flag_over_three_same_side(r: RallyRow) -> list[str]:
    """Beach volleyball: > 3 consecutive actions on one side = missed cross."""
    msgs: list[str] = []
    run_side: str | None = None
    run = 0
    run_start = -1
    run_end = -1

    def _flush() -> None:
        if run > 3 and run_side is not None:
            msgs.append(
                f"run={run} on {run_side} frames {run_start}→{run_end}"
            )

    for a in r.pred_actions:
        side = a.get("courtSide")
        if side not in ("near", "far"):
            continue
        if a.get("action") == "block":
            continue
        if side != run_side:
            _flush()
            run_side = side
            run = 1
            run_start = a.get("frame", -1)
            run_end = run_start
        else:
            run += 1
            run_end = a.get("frame", -1)
    _flush()
    return msgs


def _flag_fp_preserve_contact(r: RallyRow) -> list[str]:
    """Predicted serve before the first GT contact with low confidence.

    Attributable to the sequence-recovery rescue (CLF_FLOOR=0.20) +
    the override rewriting UNKNOWN to serve. See the plan.
    """
    if not r.gt_actions:
        return []
    first_gt_frame = min(a.get("frame", 10**9) for a in r.gt_actions)
    msgs = []
    for a in r.pred_actions:
        if a.get("action") != "serve":
            continue
        frame = a.get("frame", -1)
        conf = float(a.get("confidence", 1.0))
        if frame < first_gt_frame - 4 and conf < 0.40:
            msgs.append(
                f"pre-GT serve at frame {frame} conf={conf:.2f} "
                f"(first GT frame {first_gt_frame})"
            )
    return msgs


def _flag_unknown_rewritten(r: RallyRow) -> list[str]:
    """Heuristic for override_rewrote_unknown.

    We can't directly read MS-TCN++ argmax from the stored JSON, but a
    strong signal is a `serve` with very low classifier confidence
    (< 0.30) paired with a second `serve` at high confidence. That's
    the UNKNOWN→serve rewrite pattern seen on rally 2.
    """
    serves = [
        a for a in r.pred_actions if a.get("action") == "serve"
    ]
    if len(serves) < 2:
        return []
    confs = [float(a.get("confidence", 1.0)) for a in serves]
    if min(confs) < 0.30 and max(confs) > 0.70:
        low = next(a for a in serves if float(a.get("confidence", 1)) < 0.30)
        high = next(a for a in serves if float(a.get("confidence", 1)) > 0.70)
        return [
            f"low-conf serve f{low.get('frame')} ({confs[0]:.2f}) + "
            f"high-conf serve f{high.get('frame')} ({max(confs):.2f})"
        ]
    return []


def _flag_missed_contact_cluster(r: RallyRow) -> list[str]:
    """≥2 consecutive GT contacts with no pred match within ±8 frames.

    This is the ball-dropout signature: WASB loses the ball for 1-3s
    and multiple GT contacts disappear together.
    """
    if not r.gt_actions:
        return []
    pairs = _match_pred_to_gt(r.pred_actions, r.gt_actions)
    matched_gt = {gi for (pi, gi) in pairs if pi is not None and gi is not None}
    unmatched_gt_indices = [
        gi for gi in range(len(r.gt_actions)) if gi not in matched_gt
    ]
    if len(unmatched_gt_indices) < 2:
        return []
    # Cluster consecutive unmatched GT indices.
    runs: list[list[int]] = []
    cur: list[int] = []
    for gi in unmatched_gt_indices:
        if cur and gi == cur[-1] + 1:
            cur.append(gi)
        else:
            if len(cur) >= 2:
                runs.append(cur)
            cur = [gi]
    if len(cur) >= 2:
        runs.append(cur)

    return [
        f"missed {len(run)} consecutive GT contacts at frames "
        f"{[r.gt_actions[gi].get('frame') for gi in run]}"
        for run in runs
    ]


def _flag_action_type_mismatch(r: RallyRow) -> list[str]:
    """Pred/GT contacts aligned within ±5 frames but action_type differs."""
    if not r.gt_actions:
        return []
    pairs = _match_pred_to_gt(r.pred_actions, r.gt_actions, tol_frames=5)
    msgs = []
    for pi, gi in pairs:
        if pi is None or gi is None:
            continue
        p = r.pred_actions[pi]
        g = r.gt_actions[gi]
        if p.get("action") != g.get("action"):
            msgs.append(
                f"f{p.get('frame')} pred={p.get('action')} vs "
                f"GT f{g.get('frame')}={g.get('action')}"
            )
    return msgs


def _flag_court_side_flip(
    r: RallyRow, team_to_side: dict[str, str] | None = None,
) -> list[str]:
    """Predicted action's courtSide disagrees with the rally's own modal
    team→side convention.

    The team↔side binding is video-specific: in one match team A may sit
    on "near", in another on "far". To avoid hardcoding, we infer the
    convention from the rally's own predictions — the modal side each
    team appears on — then flag actions where a player's team is on the
    opposite side. Persistent in-rally flips (≥ 2) are the signal; a
    single flip can be a legitimate cross (e.g. a blocker touching the
    net line).

    If a video-level convention is provided via ``team_to_side`` it
    takes precedence.
    """
    # Build modal convention from the rally's predictions.
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for a in r.pred_actions:
        side = a.get("courtSide")
        team = a.get("team")
        if side in ("near", "far") and team in ("A", "B"):
            counts[(team, side)] += 1

    if team_to_side is None:
        team_to_side = {}
        for team in ("A", "B"):
            sides = {s: counts.get((team, s), 0) for s in ("near", "far")}
            if sides["near"] == 0 and sides["far"] == 0:
                continue
            team_to_side[team] = "near" if sides["near"] >= sides["far"] else "far"

    # If we couldn't infer both teams, skip.
    if len(team_to_side) < 2:
        return []
    # Sanity: teams should be on opposite sides.
    if team_to_side.get("A") == team_to_side.get("B"):
        return []

    mismatches = []
    for a in r.pred_actions:
        side = a.get("courtSide")
        team = a.get("team")
        if side not in ("near", "far") or team not in team_to_side:
            continue
        if side != team_to_side[team]:
            mismatches.append(
                f"f{a.get('frame')} team={team} on {side} "
                f"(expected {team_to_side[team]}, {a.get('action')})"
            )
    if len(mismatches) >= 2:
        return mismatches
    return []


DETECTORS = {
    "double_serve": _flag_double_serve,
    "opposite_side_double": lambda r: _flag_consecutive(r, opposite_sides=True),
    "same_side_double": lambda r: _flag_consecutive(r, opposite_sides=False),
    "over_three_same_side": _flag_over_three_same_side,
    "fp_preserve_contact": _flag_fp_preserve_contact,
    "unknown_rewritten": _flag_unknown_rewritten,
    "missed_contact_cluster": _flag_missed_contact_cluster,
    "action_type_mismatch": _flag_action_type_mismatch,
    "court_side_flip": _flag_court_side_flip,  # Replaced in _run_detectors with video-aware closure
}


def _infer_video_conventions(
    rallies: list[RallyRow],
) -> dict[str, dict[str, str]]:
    """Per-video modal team→side convention.

    Computed across every predicted action in the video, not just one
    rally. Lets us catch whole-rally team/side swaps (where within the
    rally the convention *looks* consistent, but it's the inversion of
    the rest of the match).
    """
    counts: dict[str, dict[tuple[str, str], int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for r in rallies:
        for a in r.pred_actions:
            side = a.get("courtSide")
            team = a.get("team")
            if side in ("near", "far") and team in ("A", "B"):
                counts[r.video_id][(team, side)] += 1

    out: dict[str, dict[str, str]] = {}
    for vid, cnt in counts.items():
        conv: dict[str, str] = {}
        for team in ("A", "B"):
            n = cnt.get((team, "near"), 0)
            f = cnt.get((team, "far"), 0)
            if n == 0 and f == 0:
                continue
            conv[team] = "near" if n >= f else "far"
        # Only keep if teams end up on opposite sides (sanity).
        if conv.get("A") and conv.get("B") and conv["A"] != conv["B"]:
            out[vid] = conv
    return out


def _run_detectors(rallies: list[RallyRow]) -> None:
    conventions = _infer_video_conventions(rallies)
    for r in rallies:
        for name, fn in DETECTORS.items():
            if name == "court_side_flip":
                msgs = _flag_court_side_flip(
                    r, team_to_side=conventions.get(r.video_id),
                )
            else:
                msgs = fn(r)
            if msgs:
                r.flags[name] = msgs


# --------------------------------------------------------------------------- #
# Reporting                                                                   #
# --------------------------------------------------------------------------- #


def _print_histogram(rallies: list[RallyRow]) -> tuple[dict[str, int], dict[str, int]]:
    """Print two histograms: clean pool vs low-quality pool."""
    clean = [r for r in rallies if not r.low_quality]
    low = [r for r in rallies if r.low_quality]

    def _tally(pool: list[RallyRow]) -> dict[str, int]:
        counter: Counter[str] = Counter()
        for r in pool:
            for name in DETECTORS:
                if r.flags.get(name):
                    counter[name] += 1
        return dict(counter)

    clean_hist = _tally(clean)
    low_hist = _tally(low)

    table = Table(title="Anomaly histogram (clean vs low-quality)")
    table.add_column("bucket")
    table.add_column(f"clean (n={len(clean)})", justify="right")
    table.add_column(f"low_quality (n={len(low)})", justify="right")
    table.add_column("clean %", justify="right")
    for name in DETECTORS:
        c = clean_hist.get(name, 0)
        lq = low_hist.get(name, 0)
        pct = (c / len(clean) * 100.0) if clean else 0.0
        table.add_row(name, str(c), str(lq), f"{pct:.1f}%")
    console.print(table)
    return clean_hist, low_hist


def _write_markdown(
    rallies: list[RallyRow],
    clean_hist: dict[str, int],
    low_hist: dict[str, int],
    path: Path,
) -> None:
    clean = [r for r in rallies if not r.low_quality]
    low = [r for r in rallies if r.low_quality]

    lines: list[str] = []
    lines.append("# Action-sequence anomaly audit — 2026-04-14")
    lines.append("")
    lines.append(f"- Total rallies with action GT: **{len(rallies)}**")
    lines.append(f"- Clean bucket: **{len(clean)}**  ·  Low-quality bucket: **{len(low)}**")
    lines.append("")
    lines.append("## Histogram")
    lines.append("")
    lines.append("| Bucket | Clean n | Clean % | Low-quality n | Low-quality % |")
    lines.append("|---|---:|---:|---:|---:|")
    for name in DETECTORS:
        c = clean_hist.get(name, 0)
        lq = low_hist.get(name, 0)
        cpct = (c / len(clean) * 100.0) if clean else 0.0
        lpct = (lq / len(low) * 100.0) if low else 0.0
        lines.append(f"| `{name}` | {c} | {cpct:.1f}% | {lq} | {lpct:.1f}% |")
    lines.append("")
    lines.append("## Per-rally flags (non-empty only)")
    lines.append("")
    lines.append(
        "| Video | Rally # | Rally ID | LQ | Flags |"
    )
    lines.append("|---|---:|---|:---:|---|")
    for r in sorted(
        rallies, key=lambda x: (not x.has_any_flag, x.video_id, x.rally_order)
    ):
        if not r.has_any_flag:
            continue
        flag_names = [
            name for name in DETECTORS
            if r.flags.get(name)
        ]
        # Shorten: "name: first message" per flag.
        flag_summary = "; ".join(
            f"**{name}** ({len(r.flags[name])})" for name in flag_names
        )
        lq_mark = "✓" if r.low_quality else ""
        lines.append(
            f"| {r.video_id[:8]} | {r.rally_order} | `{r.rally_id[:8]}` | {lq_mark} | {flag_summary} |"
        )
    lines.append("")
    lines.append("## Full detail")
    lines.append("")
    for r in sorted(rallies, key=lambda x: (x.video_id, x.rally_order)):
        if not r.has_any_flag:
            continue
        lines.append(
            f"### {r.video_id[:8]} · rally #{r.rally_order} "
            f"(`{r.rally_id}`)"
        )
        if r.low_quality:
            reasons = r.flags.get("quality_reason", [])
            lines.append(f"- LOW QUALITY: {', '.join(reasons)}")
        for name in DETECTORS:
            msgs = r.flags.get(name)
            if not msgs:
                continue
            lines.append(f"- **{name}**")
            for m in msgs:
                lines.append(f"  - {m}")
        lines.append("")

    path.write_text("\n".join(lines))
    console.print(f"Wrote {path}")


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", help="Restrict to a single video id")
    parser.add_argument(
        "--output", type=Path,
        default=Path("outputs/action_anomaly_audit_2026_04_14.md"),
        help="Markdown output path",
    )
    args = parser.parse_args()

    console.print("[bold]Loading rallies with action GT…[/bold]")
    rallies = _load_rallies(args.video)
    console.print(f"  loaded {len(rallies)} rallies "
                  f"({len({r.video_id for r in rallies})} videos)")

    if not rallies:
        console.print("[yellow]No rallies with action GT found.[/yellow]")
        return

    _tag_low_quality(rallies)
    _run_detectors(rallies)

    # Per-video summary printed to stderr for quick eyeball sanity.
    by_video: dict[str, list[RallyRow]] = defaultdict(list)
    for r in rallies:
        by_video[r.video_id].append(r)
    console.print(f"[dim]Per-video rally counts:[/dim]")
    for vid, rs in sorted(by_video.items(), key=lambda kv: -len(kv[1])):
        flagged = sum(1 for r in rs if r.has_any_flag)
        lq = sum(1 for r in rs if r.low_quality)
        console.print(
            f"  {vid[:8]}  n={len(rs):3d}  "
            f"flagged={flagged:3d}  low_quality={lq:3d}"
        )

    clean_hist, low_hist = _print_histogram(rallies)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_markdown(rallies, clean_hist, low_hist, args.output)


if __name__ == "__main__":
    main()
