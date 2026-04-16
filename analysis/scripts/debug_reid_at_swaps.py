"""Diagnose ReID at pred-exchange swap events detected by the audit.

For each `swap` event across a rally (or set of rallies), recompute pred_new's
appearance in a window around the swap frame and compare it to every canonical
player profile from `match_analysis_json.playerProfiles`. Classify each swap:

    reid_had_signal        ReID preferred correct — swap caused by spatial/gap
    reid_blind             ReID couldn't distinguish — needs pose/role features
    reid_wrong_preference  ReID actively chose wrong — data/model issue

Outputs:
    reports/tracking_audit/reid_debug/<rally>.html
    reports/tracking_audit/reid_debug/_summary.md

Usage:
    uv run python scripts/debug_reid_at_swaps.py                        # default trio
    uv run python scripts/debug_reid_at_swaps.py --rally <id>           # one rally
    uv run python scripts/debug_reid_at_swaps.py --all-swap-rallies     # all 18
"""

from __future__ import annotations

import argparse
import html
import json
import logging
from collections import Counter
from pathlib import Path

from rallycut.evaluation.db import get_connection
from rallycut.evaluation.tracking.db import (
    get_video_path,
    load_labeled_rallies,
)
from rallycut.tracking.swap_reid_probe import (
    CLASS_INSUFFICIENT_DATA,
    CLASS_REID_BLIND,
    CLASS_REID_HAD_SIGNAL,
    CLASS_REID_WRONG_PREFERENCE,
    SwapProbeResult,
    get_rally_track_to_player,
    load_player_profiles_from_match_analysis,
    probe_swap,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("reid-debug")

DEFAULT_RALLIES = [
    "fad29c31-6e2a-4a8d-86f1-9064b2f1f425",
    "209be896-b680-44dc-bf31-693f4e287149",
    "d724bbf0-bd0c-44e8-93d5-135aa07df5a1",
]


def _player_id_from_gt_label(label: str) -> int | None:
    """Extract canonical player_id from a GT label like 'player_3'."""
    if not label or not label.startswith("player_"):
        return None
    try:
        pid = int(label.split("_", 1)[1])
        return pid if 1 <= pid <= 4 else None
    except ValueError:
        return None


def _load_swap_events_from_audit(audit_path: Path) -> list[dict]:
    """Extract swap-style events from a rally audit JSON.

    We re-derive the same pred-exchange swap events the gallery generator
    detects, but structured for the probe.
    """
    audit = json.loads(audit_path.read_text())

    # Map GT track_id → canonical player_id via label ("player_3" → 3).
    gt_label_by_id: dict[int, str] = {
        int(g["gtTrackId"]): g["gtLabel"] for g in audit.get("perGt", [])
    }

    # Build a reverse index: pred_id → ordered (gt_track_id, start_frame, end_frame)
    pred_history: dict[int, list[tuple[int, int, int]]] = {}
    for g in audit.get("perGt", []):
        for s, e, pid in g.get("predIdSpans", []):
            pred_history.setdefault(pid, []).append((g["gtTrackId"], s, e))
    for h in pred_history.values():
        h.sort(key=lambda t: t[1])

    def prior_gt_of(pid: int, before: int) -> int | None:
        last_gt: int | None = None
        for gt_id, s, _e in pred_history.get(pid, []):
            if s >= before:
                break
            last_gt = gt_id
        return last_gt

    events: list[dict] = []
    for g in audit.get("perGt", []):
        spans = g.get("predIdSpans", [])
        for prev, cur in zip(spans, spans[1:]):
            _, prev_end, prev_pred = prev
            cur_start, _, cur_pred = cur
            if prev_pred == cur_pred:
                continue
            # Skip unmatched-gap sentinels — not real swaps.
            if prev_pred < 0 or cur_pred < 0:
                continue
            incoming_prior_gt = prior_gt_of(cur_pred, cur_start)
            if incoming_prior_gt is None or incoming_prior_gt == g["gtTrackId"]:
                continue  # true fragment, not a swap
            correct_pid = _player_id_from_gt_label(gt_label_by_id.get(incoming_prior_gt, ""))
            wrong_pid = _player_id_from_gt_label(g["gtLabel"])
            events.append({
                "rally_id": audit["rallyId"],
                "video_id": audit["videoId"],
                "swap_frame": cur_start,
                "gt_track_id": g["gtTrackId"],
                "gt_label": g["gtLabel"],
                "pred_old": prev_pred,
                "pred_new": cur_pred,
                "prior_gt_of_new": incoming_prior_gt,
                "correct_player_id": correct_pid,
                "wrong_player_id": wrong_pid,
            })
    return events


def _load_match_analysis(video_id: str) -> dict | None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT match_analysis_json FROM videos WHERE id = %s",
                [video_id],
            )
            row = cur.fetchone()
    return row[0] if row and row[0] else None


def _rally_start_ms_and_fps(rally_id: str) -> tuple[float, float]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT r.start_ms, v.fps
                FROM rallies r JOIN videos v ON v.id = r.video_id
                WHERE r.id = %s
                """,
                [rally_id],
            )
            row = cur.fetchone()
    if not row:
        return 0.0, 30.0
    return float(row[0] or 0), float(row[1] or 30.0)


def run_for_rally(rally_id: str, audit_dir: Path) -> list[SwapProbeResult]:
    audit_path = audit_dir / f"{rally_id}.json"
    if not audit_path.exists():
        logger.warning(f"  no audit JSON for {rally_id}, skipping")
        return []
    events = _load_swap_events_from_audit(audit_path)
    if not events:
        logger.info(f"  no swap events for {rally_id}")
        return []
    video_id = events[0]["video_id"]

    match_analysis = _load_match_analysis(video_id)
    if not match_analysis:
        logger.warning(f"  no match_analysis_json for video {video_id}")
        return []
    profiles = load_player_profiles_from_match_analysis(match_analysis)
    if not profiles:
        logger.warning(f"  no player profiles for video {video_id}")
        return []

    rallies = load_labeled_rallies(rally_id=rally_id)
    if not rallies:
        logger.warning(f"  rally {rally_id} not loadable")
        return []
    rally = rallies[0]
    if rally.predictions is None:
        logger.warning(f"  no predictions for {rally_id}")
        return []

    video_path = get_video_path(video_id)
    if video_path is None:
        logger.warning(f"  can't fetch video {video_id}")
        return []

    rally_start_ms, video_fps = _rally_start_ms_and_fps(rally_id)
    results: list[SwapProbeResult] = []
    for idx, ev in enumerate(events, start=1):
        result = probe_swap(
            rally_id=rally_id,
            swap_frame=ev["swap_frame"],
            gt_track_id=ev["gt_track_id"],
            pred_old=ev["pred_old"],
            pred_new=ev["pred_new"],
            prior_gt_of_new=ev["prior_gt_of_new"],
            video_path=video_path,
            rally_start_ms=rally_start_ms,
            video_fps=video_fps,
            player_profiles=profiles,
            correct_player_id=ev["correct_player_id"],
            wrong_player_id=ev["wrong_player_id"],
            predictions=rally.predictions.positions,
        )
        logger.info(
            f"  [{idx}/{len(events)}] swap@{ev['swap_frame']} "
            f"pred {ev['pred_old']}→{ev['pred_new']} on {ev['gt_label']}: "
            f"{result.classification}  ({result.samples_used_pre}pre/{result.samples_used_post}post)"
        )
        results.append(result)
    return results


def render_rally_html(rally_id: str, results: list[SwapProbeResult], track_to_player: dict[int, int]) -> str:
    def _costs_row(costs: dict[int, float]) -> str:
        if not costs:
            return "<em>no samples</em>"
        return " · ".join(
            f"<span>P{pid}=<code>{v:.3f}</code></span>"
            for pid, v in sorted(costs.items())
        )

    cards = []
    for r in results:
        correct_player = track_to_player.get(r.pred_new)
        wrong_player = track_to_player.get(r.pred_old)
        cls_color = {
            CLASS_REID_HAD_SIGNAL: "#090",
            CLASS_REID_BLIND: "#c90",
            CLASS_REID_WRONG_PREFERENCE: "#c00",
            CLASS_INSUFFICIENT_DATA: "#666",
        }.get(r.classification, "#333")

        spatial = " · ".join(
            f"pred#{pid}=<code>{d:.3f}</code>"
            for pid, d in sorted(r.spatial_distance_to_each_gt.items(), key=lambda t: t[1])
        ) or "—"
        cards.append(f"""
        <div class='card'>
            <div class='hdr'>
                <span class='k' style='background:{cls_color}22; color:{cls_color}'>
                    {html.escape(r.classification)}
                </span>
                swap @ frame <code>{r.swap_frame}</code> ·
                pred <code>{r.pred_old}</code>→<code>{r.pred_new}</code> on GT <code>{r.gt_track_id}</code>
            </div>
            <div class='body'>
                <div><strong>Canonical mapping</strong>: pred_new=<code>{r.pred_new}</code>→P<code>{correct_player}</code> (ReID-correct),
                pred_old=<code>{r.pred_old}</code>→P<code>{wrong_player}</code> (now hijacked)</div>
                <div><strong>Pre-swap costs</strong> ({r.samples_used_pre} samples): {_costs_row(r.player_costs_pre_swap)}</div>
                <div><strong>Post-swap costs</strong> ({r.samples_used_post} samples): {_costs_row(r.player_costs_post_swap)}</div>
                <div><strong>Spatial dist @ swap</strong>: {spatial}</div>
                <div class='reasoning'><em>{html.escape(r.reasoning)}</em></div>
            </div>
        </div>
        """)

    summary = Counter(r.classification for r in results)
    summary_line = " · ".join(f"{k}={v}" for k, v in summary.most_common())

    return f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'><title>ReID debug — {rally_id[:8]}</title>
<style>
body {{ font-family: -apple-system, system-ui, sans-serif; margin: 2em; color: #222; }}
h1 {{ margin-bottom: 0.2em; }}
.sub {{ color: #666; margin-bottom: 1.5em; }}
.card {{ border: 1px solid #ddd; border-radius: 6px; margin-bottom: 1em; overflow: hidden; }}
.card .hdr {{ padding: 8px 12px; background: #fafafa; border-bottom: 1px solid #eee; }}
.card .hdr .k {{ display:inline-block; padding:2px 8px; border-radius:3px; font-size:12px; font-weight:600; margin-right:8px; }}
.card .body {{ padding: 10px 14px; font-size: 13px; }}
.card .body > div {{ margin: 3px 0; }}
.reasoning {{ color: #555; margin-top: 6px !important; }}
code {{ background:#f0f0f0; padding:1px 4px; border-radius:3px; font-family: ui-monospace, SFMono-Regular, monospace; font-size: 12px; }}
</style></head>
<body>
<h1>ReID diagnostic — rally <code>{rally_id[:8]}</code></h1>
<div class='sub'>{len(results)} swap event(s). Summary: {summary_line}</div>
{''.join(cards)}
</body></html>
"""


def render_markdown_summary(all_results: dict[str, list[SwapProbeResult]]) -> str:
    total = sum(len(v) for v in all_results.values())
    summary = Counter()
    for results in all_results.values():
        for r in results:
            summary[r.classification] += 1

    lines = [
        "# ReID Debug Summary",
        "",
        f"Probed **{total}** swap events across {len(all_results)} rally(s).",
        "",
        "## Classification counts",
        "",
        "| Class | Count | Share | What it means |",
        "|---|---:|---:|---|",
    ]
    totals_by_class_label = {
        CLASS_REID_HAD_SIGNAL: "ReID had the correct signal; swap caused by non-ReID cost (spatial, gap).",
        CLASS_REID_BLIND: "ReID cannot distinguish correct vs wrong; needs pose/role/trajectory features.",
        CLASS_REID_WRONG_PREFERENCE: "ReID actively favoured the wrong player; data/model issue (lighting, pose, crop).",
        CLASS_INSUFFICIENT_DATA: "Missing canonical mapping or insufficient samples to decide.",
    }
    for cls, desc in totals_by_class_label.items():
        n = summary.get(cls, 0)
        pct = f"{100 * n / total:.1f}%" if total else "0.0%"
        lines.append(f"| `{cls}` | {n} | {pct} | {desc} |")
    lines.extend([
        "",
        "## Interpretation",
        "",
    ])
    dominant = max(summary, key=lambda k: summary[k]) if summary else None

    if dominant == CLASS_REID_HAD_SIGNAL:
        lines.append(
            "- **Dominant class is `reid_had_signal`** → the appearance feature has the signal, "
            "the issue is in the cost function's weighting. Rebalance global-identity weights "
            "(lower SPATIAL, raise APPEARANCE) with an eval gate to find the new optimum without regression."
        )
    elif dominant == CLASS_REID_BLIND:
        lines.append(
            "- **Dominant class is `reid_blind`** → teammates are indistinguishable by HSV histograms. "
            "Structural fix: add pose-keypoint geometry (shoulder/hip ratios, gait), role priors "
            "(blocker vs defender y-band), or a dedicated within-team ReID head."
        )
    elif dominant == CLASS_REID_WRONG_PREFERENCE:
        lines.append(
            "- **Dominant class is `reid_wrong_preference`** → ReID is actively making the wrong call, "
            "likely driven by jersey/lighting changes during the rally or poor crop quality at occlusion. "
            "Inspect specific frames and consider multi-crop averaging, or blacklist crops with low "
            "appearance-crop quality (blurriness, extreme angles)."
        )
    elif dominant == CLASS_INSUFFICIENT_DATA:
        lines.append(
            "- **Dominant class is `insufficient_data`** → the probe lacks canonical mappings or samples. "
            "Check `match_analysis_json.rallies[i].trackToPlayer` and ensure the probe window is wide enough."
        )

    lines.append("")
    lines.append("## Per-rally details")
    lines.append("")
    for rally_id, results in all_results.items():
        cls_counter = Counter(r.classification for r in results)
        cls_str = " · ".join(f"{k}={v}" for k, v in cls_counter.most_common())
        lines.append(
            f"- [`{rally_id[:8]}`]({rally_id}.html) — {len(results)} swap(s): {cls_str}"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=Path("reports/tracking_audit"))
    parser.add_argument(
        "--out-dir", type=Path, default=Path("reports/tracking_audit/reid_debug"),
    )
    parser.add_argument("--rally", type=str, default=None)
    parser.add_argument("--all-swap-rallies", action="store_true")
    args = parser.parse_args()

    # Decide rally set
    rally_ids: list[str]
    if args.rally:
        rally_ids = [args.rally]
    elif args.all_swap_rallies:
        # Every rally whose audit contains at least one swap event
        rally_ids = []
        for p in sorted(args.audit_dir.glob("*.json")):
            if p.name == "_summary.json":
                continue
            events = _load_swap_events_from_audit(p)
            if events:
                rally_ids.append(events[0]["rally_id"])
    else:
        rally_ids = DEFAULT_RALLIES

    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, list[SwapProbeResult]] = {}
    for idx, rid in enumerate(rally_ids, start=1):
        logger.info(f"[{idx}/{len(rally_ids)}] {rid[:8]}")
        results = run_for_rally(rid, audit_dir=args.audit_dir)
        if not results:
            continue
        # For HTML rendering we need the rally's trackToPlayer too.
        audit = json.loads((args.audit_dir / f"{rid}.json").read_text())
        match_analysis = _load_match_analysis(audit["videoId"])
        track_to_player = get_rally_track_to_player(match_analysis or {}, rid)
        html_str = render_rally_html(rid, results, track_to_player)
        (args.out_dir / f"{rid}.html").write_text(html_str)

        # Also dump structured JSON per rally
        (args.out_dir / f"{rid}.json").write_text(
            json.dumps([r.to_dict() for r in results], indent=2)
        )
        all_results[rid] = results

    if all_results:
        md = render_markdown_summary(all_results)
        (args.out_dir / "_summary.md").write_text(md)
        logger.info(f"\nSummary: {args.out_dir / '_summary.md'}")


if __name__ == "__main__":
    main()
