"""Aggregate per-rally failure budgets into an HTML dashboard.

Reads analysis/outputs/ball_gap_report/{rally_id}/failure_budget.json for
every rally processed by diagnose_ball_gaps.py and emits:

- outputs/ball_gap_report/index.html  : sortable per-rally table linking to
  each rally's overlay.mp4 + stacked-bar failure budget.
- outputs/ball_gap_report/aggregate.json : cross-rally totals and
  per-bucket breakdown so the root-cause writeup can cite numbers.
"""

from __future__ import annotations

import argparse
import json
import logging
from html import escape
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _load_budget(rally_dir: Path) -> dict[str, Any] | None:
    budget_path = rally_dir / "failure_budget.json"
    if not budget_path.exists():
        return None
    try:
        with budget_path.open() as f:
            data: dict[str, Any] = json.load(f)
            return data
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"{rally_dir.name}: failed to load budget.json ({e})")
        return None


def _load_events(rally_dir: Path) -> dict[str, Any] | None:
    events_path = rally_dir / "events.json"
    if not events_path.exists():
        return None
    try:
        with events_path.open() as f:
            data: dict[str, Any] = json.load(f)
            return data
    except (json.JSONDecodeError, OSError):
        return None


def aggregate(output_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for rally_dir in sorted(output_dir.iterdir()):
        if not rally_dir.is_dir():
            continue
        budget = _load_budget(rally_dir)
        if budget is None:
            continue
        events = _load_events(rally_dir) or {}
        gt = budget["total_gt_frames"] or 1
        match_pct = (
            (budget["matched"] + budget["interpolated_correct"]) / gt * 100.0
        )
        rows.append(
            {
                "rally_id": budget["rally_id"],
                "rally_short": budget["rally_id"][:8],
                "total_gt_frames": budget["total_gt_frames"],
                "matched": budget["matched"],
                "missed_no_raw": budget["missed_no_raw"],
                "missed_filter_killed": budget["missed_filter_killed"],
                "wrong_object": budget["wrong_object"],
                "interpolated_correct": budget["interpolated_correct"],
                "interpolated_wrong": budget["interpolated_wrong"],
                "teleport_count": budget["teleport_count"],
                "stationary_cluster_count": budget["stationary_cluster_count"],
                "stationary_cluster_frames": budget["stationary_cluster_frames"],
                "revisit_cluster_count": budget.get("revisit_cluster_count", 0),
                "revisit_cluster_visits": budget.get("revisit_cluster_visits", 0),
                "two_ball_count": budget["two_ball_count"],
                "frame_offset": budget["frame_offset"],
                "smoothness_px_p90": budget.get("smoothness_px_p90", 0.0),
                "smoothness_px_median": budget.get("smoothness_px_median", 0.0),
                "gt_mode": budget.get("gt_mode", "interpolated"),
                "raw_available": budget.get("raw_available", False),
                "prediction_source": budget.get("prediction_source", "db"),
                "match_pct": match_pct,
                "relative_mp4": (
                    f"{budget['rally_id']}/overlay.mp4"
                    if (rally_dir / "overlay.mp4").exists()
                    else None
                ),
                "relative_events": f"{budget['rally_id']}/events.json"
                if events
                else None,
            }
        )

    # Cross-rally totals
    totals = {
        "rallies": len(rows),
        "rallies_with_raw": sum(1 for r in rows if r["raw_available"]),
        "total_gt_frames": sum(r["total_gt_frames"] for r in rows),
        "matched": sum(r["matched"] for r in rows),
        "missed_no_raw": sum(r["missed_no_raw"] for r in rows),
        "missed_filter_killed": sum(r["missed_filter_killed"] for r in rows),
        "wrong_object": sum(r["wrong_object"] for r in rows),
        "interpolated_correct": sum(r["interpolated_correct"] for r in rows),
        "interpolated_wrong": sum(r["interpolated_wrong"] for r in rows),
        "teleport_count": sum(r["teleport_count"] for r in rows),
        "stationary_cluster_count": sum(r["stationary_cluster_count"] for r in rows),
        "stationary_cluster_frames": sum(r["stationary_cluster_frames"] for r in rows),
        "revisit_cluster_count": sum(r["revisit_cluster_count"] for r in rows),
        "revisit_cluster_visits": sum(r["revisit_cluster_visits"] for r in rows),
        "two_ball_count": sum(r["two_ball_count"] for r in rows),
    }
    denom = totals["total_gt_frames"] or 1
    totals["pct_matched"] = (totals["matched"] + totals["interpolated_correct"]) / denom * 100
    totals["pct_missed_no_raw"] = totals["missed_no_raw"] / denom * 100
    totals["pct_missed_filter_killed"] = totals["missed_filter_killed"] / denom * 100
    totals["pct_wrong_object"] = totals["wrong_object"] / denom * 100
    totals["pct_interpolated_wrong"] = totals["interpolated_wrong"] / denom * 100
    smooth_vals = [r["smoothness_px_p90"] for r in rows if r["smoothness_px_p90"] > 0]
    totals["smoothness_px_p90_median"] = (
        sorted(smooth_vals)[len(smooth_vals) // 2] if smooth_vals else 0.0
    )
    totals["smoothness_px_p90_max"] = max(smooth_vals) if smooth_vals else 0.0

    # Default sort: revisit clusters (persistent distractors) desc, teleport
    # count (transient jumps) desc, smoothness (jitter) desc, match% asc.
    # Column headers are clickable so users can resort in the browser.
    rows.sort(
        key=lambda r: (
            -r["revisit_cluster_count"],
            -r["teleport_count"],
            -r["smoothness_px_p90"],
            r["match_pct"],
        )
    )
    return {"totals": totals, "rallies": rows}


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------


_CSS = """
body { font-family: -apple-system, Segoe UI, Helvetica, sans-serif; margin: 20px; background:#f6f7f9; color:#222; }
h1 { margin-top:0; }
.totals { display:flex; gap:16px; flex-wrap:wrap; margin-bottom:20px; }
.card { background:#fff; border-radius:8px; padding:12px 18px; box-shadow: 0 1px 3px rgba(0,0,0,.06); }
.card .label { font-size:11px; color:#666; text-transform:uppercase; letter-spacing:.08em; }
.card .value { font-size:22px; font-weight:600; }
.card .sub   { font-size:11px; color:#888; }
table { width:100%; border-collapse: collapse; background:#fff; border-radius:8px; overflow:hidden; }
th, td { padding: 8px 10px; text-align:left; font-size:13px; border-bottom:1px solid #eee; }
th { background:#f1f3f5; cursor:pointer; user-select:none; position:sticky; top:0; }
th.sorted-asc::after  { content:" ▲"; font-size:10px; }
th.sorted-desc::after { content:" ▼"; font-size:10px; }
tr:hover { background:#fbfbfb; }
.bar { display:flex; height:14px; border-radius:4px; overflow:hidden; background:#eee; min-width:160px; }
.bar > div { height:100%; }
.b-match   { background:#4cb069; }
.b-interp  { background:#8fd6a0; }
.b-miss    { background:#d9a35c; }
.b-killed  { background:#d04040; }
.b-wrong   { background:#d96fbc; }
.b-iwrong  { background:#a0439e; }
.legend span { display:inline-flex; align-items:center; gap:4px; margin-right:14px; font-size:12px; color:#444; }
.legend span::before { content:""; display:inline-block; width:10px; height:10px; border-radius:2px; }
.legend .b-match::before    { background:#4cb069; }
.legend .b-interp::before   { background:#8fd6a0; }
.legend .b-miss::before     { background:#d9a35c; }
.legend .b-killed::before   { background:#d04040; }
.legend .b-wrong::before    { background:#d96fbc; }
.legend .b-iwrong::before   { background:#a0439e; }
.noraw { color:#aaa; font-size:11px; }
a { color:#2563eb; text-decoration:none; }
a:hover { text-decoration:underline; }
.event-count { display:inline-block; min-width:22px; text-align:center; padding:1px 6px; border-radius:10px; background:#eee; font-size:11px; }
.event-count.warn { background:#fde68a; color:#6b3a00; }
.event-count.crit { background:#fca5a5; color:#7f1d1d; }
footer { margin-top:24px; font-size:12px; color:#888; }
"""

_JS = """
document.querySelectorAll('th[data-col]').forEach(th => {
  th.addEventListener('click', () => {
    const table = th.closest('table');
    const tbody = table.tBodies[0];
    const col = th.dataset.col;
    const type = th.dataset.type || 'string';
    const currentDir = th.dataset.dir === 'asc' ? 'desc' : 'asc';
    document.querySelectorAll('th').forEach(t => {
      t.classList.remove('sorted-asc', 'sorted-desc');
      delete t.dataset.dir;
    });
    th.dataset.dir = currentDir;
    th.classList.add(currentDir === 'asc' ? 'sorted-asc' : 'sorted-desc');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    rows.sort((a, b) => {
      const av = a.querySelector(`td[data-col="${col}"]`).dataset.value;
      const bv = b.querySelector(`td[data-col="${col}"]`).dataset.value;
      if (type === 'number') return (+av - +bv) * (currentDir === 'asc' ? 1 : -1);
      return av.localeCompare(bv) * (currentDir === 'asc' ? 1 : -1);
    });
    rows.forEach(r => tbody.appendChild(r));
  });
});
"""


def _bar_segment(fraction: float, css_class: str, title: str) -> str:
    pct = fraction * 100
    if pct <= 0:
        return ""
    return (
        f'<div class="{css_class}" style="width:{pct:.2f}%" '
        f'title="{escape(title)}"></div>'
    )


def _render_bar(row: dict[str, Any]) -> str:
    total = row["total_gt_frames"] or 1
    parts = [
        _bar_segment(row["matched"] / total, "b-match", f'matched: {row["matched"]}'),
        _bar_segment(
            row["interpolated_correct"] / total,
            "b-interp",
            f'interp correct: {row["interpolated_correct"]}',
        ),
        _bar_segment(
            row["missed_no_raw"] / total,
            "b-miss",
            f'missed (no raw): {row["missed_no_raw"]}',
        ),
        _bar_segment(
            row["missed_filter_killed"] / total,
            "b-killed",
            f'missed (filter killed): {row["missed_filter_killed"]}',
        ),
        _bar_segment(
            row["wrong_object"] / total,
            "b-wrong",
            f'wrong object: {row["wrong_object"]}',
        ),
        _bar_segment(
            row["interpolated_wrong"] / total,
            "b-iwrong",
            f'interp wrong: {row["interpolated_wrong"]}',
        ),
    ]
    return '<div class="bar">' + "".join(parts) + "</div>"


def _event_chip(value: int, warn: int = 3, crit: int = 10) -> str:
    cls = ""
    if value >= crit:
        cls = "crit"
    elif value >= warn:
        cls = "warn"
    return f'<span class="event-count {cls}">{value}</span>'


def render_html(aggregate_data: dict[str, Any], output_path: Path) -> None:
    totals = aggregate_data["totals"]
    rows = aggregate_data["rallies"]

    cards = [
        (
            "Rallies",
            f"{totals['rallies']}",
            f"{totals['rallies_with_raw']} with raw cache",
        ),
        (
            "Match %",
            f"{totals['pct_matched']:.1f}%",
            f"{totals['matched']} / {totals['total_gt_frames']} GT frames",
        ),
        (
            "Missed (no raw)",
            f"{totals['pct_missed_no_raw']:.1f}%",
            f"{totals['missed_no_raw']} frames — detector miss",
        ),
        (
            "Missed (filter killed)",
            f"{totals['pct_missed_filter_killed']:.1f}%",
            f"{totals['missed_filter_killed']} frames — filter dropped a real ball",
        ),
        (
            "Wrong object",
            f"{totals['pct_wrong_object']:.1f}%",
            f"{totals['wrong_object']} frames — pred > 100 px from GT",
        ),
        (
            "Teleports",
            f"{totals['teleport_count']}",
            "> 120 px/frame jumps",
        ),
        (
            "Stationary clusters",
            f"{totals['stationary_cluster_count']}",
            f"spanning {totals['stationary_cluster_frames']} raw frames",
        ),
        (
            "Jitter p90 (median)",
            f"{totals['smoothness_px_p90_median']:.1f}",
            f"max rally: {totals['smoothness_px_p90_max']:.1f} px/frame²",
        ),
        (
            "Revisit clusters",
            f"{totals['revisit_cluster_count']}",
            f"{totals['revisit_cluster_visits']} total visits (non-contiguous)",
        ),
    ]

    legend_html = (
        '<div class="legend">'
        '<span class="b-match">matched</span>'
        '<span class="b-interp">interp correct</span>'
        '<span class="b-miss">missed (no raw)</span>'
        '<span class="b-killed">missed (filter killed)</span>'
        '<span class="b-wrong">wrong object</span>'
        '<span class="b-iwrong">interp wrong</span>'
        "</div>"
    )

    card_html = "".join(
        f'<div class="card"><div class="label">{escape(label)}</div>'
        f'<div class="value">{escape(value)}</div>'
        f'<div class="sub">{escape(sub)}</div></div>'
        for label, value, sub in cards
    )

    rows_html = []
    for r in rows:
        mp4_cell = (
            f'<a href="{escape(r["relative_mp4"])}">overlay.mp4</a>'
            if r["relative_mp4"]
            else '<span class="noraw">—</span>'
        )
        events_cell = (
            f'<a href="{escape(r["relative_events"])}">events</a>'
            if r["relative_events"]
            else ""
        )
        raw_cell = (
            "✓"
            if r["raw_available"]
            else '<span class="noraw">no raw</span>'
        )
        rows_html.append(
            "<tr>"
            f'<td data-col="rally" data-value="{escape(r["rally_short"])}">'
            f'<code>{escape(r["rally_short"])}</code></td>'
            f'<td data-col="match" data-value="{r["match_pct"]:.2f}">'
            f'{r["match_pct"]:.1f}%</td>'
            f'<td data-col="total" data-value="{r["total_gt_frames"]}">{r["total_gt_frames"]}</td>'
            f'<td data-col="bar" data-value="{r["match_pct"]:.2f}">{_render_bar(r)}</td>'
            f'<td data-col="noraw" data-value="{r["missed_no_raw"]}">{r["missed_no_raw"]}</td>'
            f'<td data-col="killed" data-value="{r["missed_filter_killed"]}">{r["missed_filter_killed"]}</td>'
            f'<td data-col="wrong" data-value="{r["wrong_object"]}">{r["wrong_object"]}</td>'
            f'<td data-col="tele" data-value="{r["teleport_count"]}">'
            f'{_event_chip(r["teleport_count"])}</td>'
            f'<td data-col="smooth" data-value="{r["smoothness_px_p90"]:.2f}">'
            f'{r["smoothness_px_p90"]:.1f}</td>'
            f'<td data-col="revisit" data-value="{r["revisit_cluster_count"]}">'
            f'{_event_chip(r["revisit_cluster_count"])}'
            + (f' <span class="noraw">({r["revisit_cluster_visits"]})</span>'
               if r["revisit_cluster_count"] > 0 else "")
            + '</td>'
            f'<td data-col="static" data-value="{r["stationary_cluster_count"]}">'
            f'{_event_chip(r["stationary_cluster_count"])}</td>'
            f'<td data-col="twob" data-value="{r["two_ball_count"]}">'
            f'{_event_chip(r["two_ball_count"])}</td>'
            f'<td data-col="raw"  data-value="{int(r["raw_available"])}">{raw_cell}</td>'
            f'<td data-col="src"  data-value="{escape(r["prediction_source"])}">'
            f'{escape(r["prediction_source"])}</td>'
            f'<td data-col="gt" data-value="{escape(r["gt_mode"])}">'
            f'{escape(r["gt_mode"])}</td>'
            f'<td data-col="mp4"  data-value="{1 if r["relative_mp4"] else 0}">'
            f"{mp4_cell}</td>"
            f'<td data-col="evt"  data-value="{1 if r["relative_events"] else 0}">'
            f"{events_cell}</td>"
            "</tr>"
        )

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Ball Tracking Gap Report</title>
<style>{_CSS}</style>
</head>
<body>
<h1>Ball Tracking Gap Report</h1>
<p>Generated from <code>failure_budget.json</code> files under
<code>analysis/outputs/ball_gap_report/</code>. Click column headers to sort.
Bars stack the per-rally failure budget by root-cause bucket. Click any
<code>overlay.mp4</code> link to open the annotated video in a new tab.</p>

<div class="totals">{card_html}</div>
{legend_html}

<table>
<thead>
<tr>
<th data-col="rally" data-type="string">Rally</th>
<th data-col="match" data-type="number">Match %</th>
<th data-col="total" data-type="number">GT frames</th>
<th data-col="bar">Budget</th>
<th data-col="noraw" data-type="number" title="Frames where GT had a ball but raw WASB produced no near detection">Miss (no raw)</th>
<th data-col="killed" data-type="number" title="Frames where raw had a GT-close detection but the filter dropped it">Miss (killed)</th>
<th data-col="wrong" data-type="number" title="Frames where pred exists but is >100 px from GT">Wrong</th>
<th data-col="tele" data-type="number" title="Consecutive predictions separated by >120 px/frame">Teleports</th>
<th data-col="smooth" data-type="number" title="p90 of per-frame prediction acceleration magnitude (px/frame²). Smooth parabolic ball motion sits near 10; jittery tracking spikes higher.">Jitter p90</th>
<th data-col="revisit" data-type="number" title="Spatial hot-spots the tracker returns to across the rally (flags, logos). Not temporally sustained so the static detector misses them. (visits shown in parens)">Revisit</th>
<th data-col="static" data-type="number" title="Clusters of raw detections with spread <0.5% for >=20 frames">Static</th>
<th data-col="twob" data-type="number" title="Sustained simultaneous high-conf detections >200 px apart">Two-ball</th>
<th data-col="raw"  data-type="number" title="Whether the BallRawCache had this rally">Raw</th>
<th data-col="src"  data-type="string" title="Prediction source: db (stored) or refilter (current filter config re-applied to raw)">Source</th>
<th data-col="gt"   data-type="string" title="GT evaluation mode: keyframes (honest) or interpolated (per-frame)">GT mode</th>
<th data-col="mp4"  data-type="number">Video</th>
<th data-col="evt"  data-type="number">Events</th>
</tr>
</thead>
<tbody>
{''.join(rows_html)}
</tbody>
</table>

<footer>Built by <code>analysis/scripts/build_ball_failure_report.py</code>.</footer>
<script>{_JS}</script>
</body>
</html>
"""

    output_path.write_text(html)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="outputs/ball_gap_report",
        help="Directory containing per-rally subdirs (relative to analysis/)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    output_dir = Path(args.output_dir).resolve()
    if not output_dir.exists():
        logger.error(f"Output dir not found: {output_dir}")
        return 2

    data = aggregate(output_dir)
    (output_dir / "aggregate.json").write_text(json.dumps(data, indent=2))
    logger.info(
        f"Aggregated {len(data['rallies'])} rallies "
        f"({data['totals']['rallies_with_raw']} with raw cache)"
    )
    index_path = output_dir / "index.html"
    render_html(data, index_path)
    logger.info(f"Wrote {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
