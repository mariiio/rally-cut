"""Session 5 — render the per-event labeller HTML from events.json.

Reads ``reports/occlusion_resolver/events.json`` and produces
``reports/occlusion_resolver/labeller.html`` — a single static page with
one card per convergence event. Forks the Session-2 contact-sheet UI
pattern (dark theme, sticky nav, localStorage persistence) but:

- Each card shows 4 crop strips (pre_a, pre_b, post_a, post_b) instead of
  2 thumbnails.
- Keyboard shortcuts: 1=no-swap, 2=swap, 3=fragment-only, 4=unclear,
  space=advance-to-next-unlabelled.
- Exports to ``training_data/occlusion_resolver/labels.json``.

Usage:
    uv run python scripts/render_occlusion_labeller.py
    open analysis/reports/occlusion_resolver/labeller.html
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import sys
from pathlib import Path

ANALYSIS_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ANALYSIS_ROOT / "reports" / "occlusion_resolver"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("render_occlusion_labeller")


CSS = """
body { font-family: -apple-system, system-ui, sans-serif; margin: 16px; background: #0f1218; color: #e8ecf4; }
h1 { margin-top: 0; }
nav { position: sticky; top: 0; background: #0f1218; padding: 8px 0 12px; border-bottom: 1px solid #2a3040; margin-bottom: 16px; z-index: 10; }
nav .row { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-bottom: 6px; }
nav .controls { margin-left: auto; display: flex; gap: 8px; align-items: center; }
nav button { background: #2a3040; color: #e8ecf4; border: 1px solid #3a4256; padding: 6px 12px; border-radius: 6px; cursor: pointer; font-weight: 600; font-size: 12px; }
nav button:hover { background: #3a4256; }
nav button.primary { background: #6aa9ff; color: #0f1218; border-color: #6aa9ff; }
nav button.primary:hover { background: #84bbff; }
.badge { display: inline-block; background: #2a3040; color: #b8c2d8; padding: 2px 8px; border-radius: 8px; font-size: 12px; margin-left: 4px; }
.badge.positive { background: #1e3a2a; color: #4ade80; }
.stats { font-size: 12px; color: #b8c2d8; }
.stats strong { color: #e8ecf4; }
.progress { background: #2a3040; border-radius: 4px; height: 6px; overflow: hidden; margin-top: 6px; }
.progress-bar { height: 100%; background: #6aa9ff; transition: width 0.2s; }

details.instructions { background: #1a2030; border: 1px solid #2a3040; border-radius: 8px; padding: 8px 18px; margin-bottom: 24px; font-size: 14px; line-height: 1.6; }
details.instructions[open] { padding: 14px 18px; }
details.instructions > summary { cursor: pointer; padding: 6px 0; user-select: none; color: #e8ecf4; list-style: none; }
details.instructions > summary::-webkit-details-marker { display: none; }
details.instructions > summary::before { content: "▶ "; color: #6aa9ff; display: inline-block; width: 1em; transition: transform 0.15s; }
details.instructions[open] > summary::before { transform: rotate(90deg); }
details.instructions h3 { margin: 12px 0 4px; font-size: 13px; color: #6aa9ff; text-transform: uppercase; letter-spacing: 0.5px; }
details.instructions kbd { background: #0f1218; border: 1px solid #2a3040; padding: 2px 6px; border-radius: 4px; font-family: ui-monospace, monospace; font-size: 12px; }

.events { display: flex; flex-direction: column; gap: 16px; }
.card { background: #161a24; border: 1px solid #2a3040; border-radius: 10px; padding: 14px; position: relative; scroll-margin-top: 120px; }
.card.active { border-color: #6aa9ff; box-shadow: 0 0 0 2px #6aa9ff44; }
.card[data-label="swap"] { border-left: 4px solid #f97316; }
.card[data-label="no-swap"] { border-left: 4px solid #4ade80; }
.card[data-label="fragment-only"] { border-left: 4px solid #a78bfa; }
.card[data-label="unclear"] { border-left: 4px solid #64748b; }

.card .header { display: flex; gap: 12px; align-items: baseline; flex-wrap: wrap; margin-bottom: 10px; font-size: 13px; color: #b8c2d8; font-family: ui-monospace, monospace; }
.card .header .id { color: #e8ecf4; font-weight: 600; }
.card .header .meta-item { }
.card .header .meta-item .label { color: #6aa9ff; }
.card .header .hint { background: #ffcc00; color: #161a24; padding: 2px 8px; border-radius: 4px; font-weight: 700; font-size: 11px; }

.strips { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 10px; }
.strip { background: #0f1218; border: 1px solid #2a3040; border-radius: 6px; padding: 8px; }
.strip.pre_a, .strip.post_a { border-left-color: #6aa9ff; border-left-width: 3px; }
.strip.pre_b, .strip.post_b { border-left-color: #fb923c; border-left-width: 3px; }
.strip .title { font-size: 11px; color: #8492b0; font-family: ui-monospace, monospace; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px; }
.strip .title .track-a-dot { color: #6aa9ff; }
.strip .title .track-b-dot { color: #fb923c; }
.strip .thumbs { display: flex; gap: 4px; overflow-x: auto; }
.strip img { width: 100px; height: 100px; object-fit: cover; border: 1px solid #2a3040; border-radius: 3px; flex-shrink: 0; }

.actions { display: flex; gap: 8px; margin-top: 8px; }
.actions button { flex: 1; padding: 8px 10px; border-radius: 6px; border: 1px solid #3a4256; background: #2a3040; color: #e8ecf4; cursor: pointer; font-weight: 600; font-size: 13px; }
.actions button.no-swap { border-color: #4ade8044; }
.actions button.no-swap:hover, .card[data-label="no-swap"] .actions button.no-swap { background: #14532d; border-color: #4ade80; color: #4ade80; }
.actions button.swap { border-color: #f9731644; }
.actions button.swap:hover, .card[data-label="swap"] .actions button.swap { background: #7c2d12; border-color: #f97316; color: #fdba74; }
.actions button.fragment-only { border-color: #a78bfa44; }
.actions button.fragment-only:hover, .card[data-label="fragment-only"] .actions button.fragment-only { background: #3b2568; border-color: #a78bfa; color: #c4b5fd; }
.actions button.unclear { border-color: #64748b44; }
.actions button.unclear:hover, .card[data-label="unclear"] .actions button.unclear { background: #1e293b; border-color: #64748b; color: #94a3b8; }
.actions .hint-key { display: inline-block; background: #0f1218; padding: 1px 6px; border-radius: 3px; margin-left: 6px; font-family: ui-monospace, monospace; font-size: 11px; opacity: 0.7; }

footer { margin-top: 40px; padding-top: 16px; border-top: 1px solid #2a3040; font-size: 13px; color: #8492b0; }
"""


def _render_card(ev: dict) -> str:
    pre_a = ev.get("crops", {}).get("pre_a", [])
    pre_b = ev.get("crops", {}).get("pre_b", [])
    post_a = ev.get("crops", {}).get("post_a", [])
    post_b = ev.get("crops", {}).get("post_b", [])

    def _strip(label: str, css_class: str, paths: list[str], track_class: str) -> str:
        if not paths:
            return (
                f'<div class="strip {css_class}"><div class="title">'
                f'<span class="{track_class}-dot">●</span> {html.escape(label)} '
                f'<span style="color:#f87171">(no crops — abstain)</span>'
                f'</div></div>'
            )
        imgs = "".join(
            f'<img src="{html.escape(p)}" alt="{html.escape(label)}">' for p in paths
        )
        count = f" <span style='color:#e8ecf4'>({len(paths)})</span>"
        return (
            f'<div class="strip {css_class}">'
            f'  <div class="title"><span class="{track_class}-dot">●</span> '
            f'{html.escape(label)}{count}</div>'
            f'  <div class="thumbs">{imgs}</div>'
            f'</div>'
        )

    header_bits = [
        f'<span class="id">{html.escape(ev["event_id"])}</span>',
        (
            f'<span class="meta-item"><span class="label">rally:</span> '
            f'{html.escape(ev["rally_id"][:8])}</span>'
        ),
        (
            f'<span class="meta-item"><span class="label">tracks:</span> '
            f'<span style="color:#6aa9ff">T{ev["track_a"]}</span>/'
            f'<span style="color:#fb923c">T{ev["track_b"]}</span></span>'
        ),
        (
            f'<span class="meta-item"><span class="label">team:</span> '
            f'{ev["team"]}</span>'
        ),
        (
            f'<span class="meta-item"><span class="label">frames:</span> '
            f'{ev["start_frame"]}–{ev["end_frame"]} '
            f'(dur {ev["duration_frames"]})</span>'
        ),
        (
            f'<span class="meta-item"><span class="label">source:</span> '
            f'{html.escape(ev["source"])}</span>'
        ),
    ]
    if ev.get("crossed_switch"):
        header_bits.append('<span class="hint">audit-hint: crossed_switch</span>')

    strips = "".join([
        _strip("pre · track A", "pre_a", pre_a, "track-a"),
        _strip("pre · track B", "pre_b", pre_b, "track-b"),
        _strip("post · track A", "post_a", post_a, "track-a"),
        _strip("post · track B", "post_b", post_b, "track-b"),
    ])

    actions = (
        '<div class="actions">'
        '  <button type="button" class="no-swap" data-label="no-swap">'
        'No swap <span class="hint-key">1</span></button>'
        '  <button type="button" class="swap" data-label="swap">'
        'Swap <span class="hint-key">2</span></button>'
        '  <button type="button" class="fragment-only" data-label="fragment-only">'
        'Fragment-only <span class="hint-key">3</span></button>'
        '  <button type="button" class="unclear" data-label="unclear">'
        'Unclear <span class="hint-key">4</span></button>'
        '</div>'
    )

    return (
        f'<div class="card" data-event-id="{html.escape(ev["event_id"])}" '
        f'data-rally="{html.escape(ev["rally_id"])}" '
        f'data-frame="{ev["start_frame"]}">'
        f'  <div class="header">{" · ".join(header_bits)}</div>'
        f'  <div class="strips">{strips}</div>'
        f'  {actions}'
        f'</div>'
    )


SCRIPT = """
const STORAGE_KEY = 'rallycut_session5_labels_v1';
const state = loadState();
const events = Array.from(document.querySelectorAll('.card'));
const totalEvents = events.length;
let activeIdx = 0;

function loadState() {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}'); }
  catch { return {}; }
}
function saveState() { localStorage.setItem(STORAGE_KEY, JSON.stringify(state)); }

function refreshCard(card) {
  const id = card.dataset.eventId;
  const label = state[id] || null;
  if (label) card.setAttribute('data-label', label);
  else card.removeAttribute('data-label');
}

function updateStats() {
  const labels = Object.values(state);
  const by = { 'no-swap': 0, 'swap': 0, 'fragment-only': 0, 'unclear': 0 };
  labels.forEach(l => { if (by[l] !== undefined) by[l]++; });
  const labelled = labels.length;
  const pct = totalEvents ? Math.round(100 * labelled / totalEvents) : 0;
  document.getElementById('stat-total').textContent = totalEvents;
  document.getElementById('stat-labelled').textContent = labelled;
  document.getElementById('stat-pending').textContent = totalEvents - labelled;
  document.getElementById('stat-no-swap').textContent = by['no-swap'];
  document.getElementById('stat-swap').textContent = by['swap'];
  document.getElementById('stat-fragment').textContent = by['fragment-only'];
  document.getElementById('stat-unclear').textContent = by['unclear'];
  document.getElementById('progress-bar').style.width = pct + '%';
}

function setLabel(card, label) {
  const id = card.dataset.eventId;
  state[id] = label;
  saveState();
  refreshCard(card);
  updateStats();
}

function setActive(idx) {
  events.forEach(c => c.classList.remove('active'));
  activeIdx = Math.max(0, Math.min(events.length - 1, idx));
  if (events[activeIdx]) {
    events[activeIdx].classList.add('active');
    events[activeIdx].scrollIntoView({ behavior: 'smooth', block: 'center' });
  }
}

function advanceToNextUnlabelled() {
  for (let off = 1; off <= events.length; off++) {
    const i = (activeIdx + off) % events.length;
    const id = events[i].dataset.eventId;
    if (!state[id]) { setActive(i); return; }
  }
  // All labelled — just move next
  setActive(activeIdx + 1);
}

events.forEach((card, idx) => {
  refreshCard(card);
  card.querySelectorAll('.actions button').forEach(btn => {
    btn.addEventListener('click', () => {
      setLabel(card, btn.dataset.label);
      setActive(idx);
      setTimeout(advanceToNextUnlabelled, 150);
    });
  });
  card.addEventListener('click', (e) => {
    if (!e.target.closest('button')) setActive(idx);
  });
});

document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  const card = events[activeIdx];
  if (!card) return;
  switch (e.key) {
    case '1': setLabel(card, 'no-swap'); setTimeout(advanceToNextUnlabelled, 100); break;
    case '2': setLabel(card, 'swap'); setTimeout(advanceToNextUnlabelled, 100); break;
    case '3': setLabel(card, 'fragment-only'); setTimeout(advanceToNextUnlabelled, 100); break;
    case '4': setLabel(card, 'unclear'); setTimeout(advanceToNextUnlabelled, 100); break;
    case ' ': e.preventDefault(); advanceToNextUnlabelled(); break;
    case 'ArrowDown': case 'j': setActive(activeIdx + 1); break;
    case 'ArrowUp': case 'k': setActive(activeIdx - 1); break;
  }
});

window.exportLabels = function () {
  const eventMeta = window.__EVENT_META__;
  const payload = {
    version: 1,
    schema: 'rallycut-session5-occlusion-labels-v1',
    ts: new Date().toISOString(),
    labels: Object.entries(state).map(([event_id, label]) => {
      const meta = eventMeta[event_id] || {};
      return { event_id, label, ...meta };
    }),
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'labels.json';
  document.body.appendChild(a); a.click(); a.remove();
};

window.importLabels = function () {
  const inp = document.createElement('input');
  inp.type = 'file'; inp.accept = '.json';
  inp.onchange = (ev) => {
    const file = ev.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const data = JSON.parse(reader.result);
        const labels = data.labels || [];
        labels.forEach(l => { if (l.event_id && l.label) state[l.event_id] = l.label; });
        saveState();
        events.forEach(refreshCard);
        updateStats();
      } catch (e) { alert('Import failed: ' + e); }
    };
    reader.readAsText(file);
  };
  document.body.appendChild(inp); inp.click(); inp.remove();
};

window.clearAll = function () {
  if (!confirm('Delete all labels? This cannot be undone.')) return;
  Object.keys(state).forEach(k => delete state[k]);
  saveState();
  events.forEach(refreshCard);
  updateStats();
};

setActive(0);
updateStats();
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--events", type=Path, default=OUT_DIR / "events.json",
    )
    parser.add_argument(
        "--out", type=Path, default=OUT_DIR / "labeller.html",
    )
    args = parser.parse_args()

    if not args.events.exists():
        logger.error("events JSON missing: %s", args.events)
        return 1
    data = json.loads(args.events.read_text())
    events = data.get("events", [])
    logger.info("rendering %d events -> %s", len(events), args.out)

    # Per-event metadata dict (for export)
    event_meta = {
        e["event_id"]: {
            k: e.get(k)
            for k in (
                "rally_id", "video_id", "track_a", "track_b", "team",
                "start_frame", "end_frame", "duration_frames",
                "crossed_switch", "source",
            )
        }
        for e in events
    }

    cards_html = "\n".join(_render_card(e) for e in events)

    html_doc = f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'>
<title>Occlusion Labeller — RallyCut Session 5</title>
<style>{CSS}</style>
</head><body>
<h1>Occlusion Labeller <span class="badge">{len(events)} events</span></h1>
<p>Convergences where two same-team tracks overlap or where a known swap
event was detected in the Session-4 retrack audit. Label each event with
one of the four options below to produce the Session-5 resolver's training
+ evaluation set.</p>

<nav>
  <div class="row">
    <span class="stats">
      <strong id="stat-total">{len(events)}</strong> total ·
      <strong id="stat-labelled">0</strong> labelled ·
      <strong id="stat-pending">{len(events)}</strong> pending
    </span>
    <span class="stats" style="margin-left:16px">
      no-swap: <strong id="stat-no-swap">0</strong> ·
      swap: <strong id="stat-swap">0</strong> ·
      fragment: <strong id="stat-fragment">0</strong> ·
      unclear: <strong id="stat-unclear">0</strong>
    </span>
    <div class="controls">
      <button type="button" onclick="window.importLabels()">Import…</button>
      <button type="button" class="primary" onclick="window.exportLabels()">Export labels</button>
      <button type="button" onclick="window.clearAll()" style="background:#4b1d22;color:#ff8a93;border-color:#f87171">Clear all</button>
    </div>
  </div>
  <div class="progress"><div class="progress-bar" id="progress-bar" style="width:0%"></div></div>
</nav>

<details class="instructions" open>
<summary><strong>How to label</strong> (click to collapse)</summary>
<p>You'll see each convergence event as 4 strips of crops:</p>
<ul>
  <li><strong style="color:#6aa9ff">Track A</strong> — pre-convergence (before) and post-convergence (after) for the FIRST track.</li>
  <li><strong style="color:#fb923c">Track B</strong> — pre and post for the SECOND track.</li>
</ul>
<p>Your question: <strong>does each track's post-convergence identity match its pre-convergence identity?</strong></p>
<ul>
  <li><kbd>1</kbd> <strong style="color:#4ade80">No swap</strong>: Post A matches Pre A, Post B matches Pre B. Tracks stayed correct.</li>
  <li><kbd>2</kbd> <strong style="color:#fdba74">Swap</strong>: Post A looks like Pre B, Post B looks like Pre A. Identities crossed.</li>
  <li><kbd>3</kbd> <strong style="color:#c4b5fd">Fragment-only</strong>: One track dies during the convergence. No swap to detect — it's just a track loss.</li>
  <li><kbd>4</kbd> <strong style="color:#94a3b8">Unclear</strong>: Too occluded / insufficient evidence to decide. Abstain.</li>
</ul>
<p><kbd>space</kbd> advances to the next unlabelled event. <kbd>↑/↓</kbd> or <kbd>j/k</kbd> navigates.</p>
<p>When done, click <strong>Export labels</strong> to save
<code>labels.json</code>. Move it to
<code>training_data/occlusion_resolver/labels.json</code>.</p>
</details>

<div class="events">
{cards_html}
</div>

<footer>
Session 5 labeller · {len(events)} convergence events · crops native resolution.
Data from <code>{html.escape(str(args.events.name))}</code>.
</footer>

<script>window.__EVENT_META__ = {json.dumps(event_meta)};</script>
<script>{SCRIPT}</script>
</body></html>
"""
    args.out.write_text(html_doc)
    logger.info("wrote %s (%d events)", args.out, len(events))
    return 0


if __name__ == "__main__":
    sys.exit(main())
