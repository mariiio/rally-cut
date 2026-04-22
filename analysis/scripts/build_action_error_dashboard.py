#!/usr/bin/env python3
"""Build a friendly HTML dashboard for reviewing action detection errors.

Groups errors into clear categories, shows video clips inline, and lets
the reviewer quickly tag each case with one-click feedback.

Usage:
    uv run python scripts/build_action_error_dashboard.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "action_errors"


def load_corpus(path: Path) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            # Skip the canary _meta header that newer corpora prepend.
            if isinstance(rec, dict) and rec.get("_meta") is True:
                continue
            errors.append(rec)
    return errors


def load_tags(path: Path) -> dict[str, dict[str, str]]:
    tags: dict[str, dict[str, str]] = {}
    if not path.exists():
        return tags
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = f"{row['rally_id']}_{row['gt_frame']}"
            tags[key] = dict(row)
    return tags


def friendly_category(e: dict) -> str:
    ec = e.get("error_class", "")
    if ec == "FN_contact":
        return "missed_contact"
    elif ec == "wrong_action":
        return "wrong_action_type"
    elif ec == "wrong_player":
        return "wrong_player"
    return "other"


def friendly_reason(e: dict) -> str:
    ec = e.get("error_class", "")
    fn_cat = e.get("fn_subcategory", "")
    seq_peak = e.get("seq_peak_nonbg_within_5f", 0.0)
    seq_act = e.get("seq_peak_action", "")
    seq_prob = e.get("seq_peak_action_prob", 0.0)
    seq_suffix = ""
    if seq_peak:
        seq_suffix = f" · seq={seq_act}({seq_prob:.2f})"

    if ec == "FN_contact":
        if fn_cat == "ball_dropout":
            return f"Ball lost for {e.get('ball_gap_frames', '?')} frames around this contact{seq_suffix}"
        if fn_cat == "rejected_by_classifier":
            conf = e.get("classifier_conf", 0)
            return (
                f"Candidate scored {conf:.2f} (threshold 0.40, rescue floor 0.20)"
                f"{seq_suffix}"
            )
        if fn_cat == "rejected_by_gates":
            return f"Candidate rejected by validation rules{seq_suffix}"
        if fn_cat == "no_player_nearby":
            return f"Candidate fired but no player in 0.15 radius{seq_suffix}"
        if fn_cat == "deduplicated":
            return f"Contact merged with a nearby stronger contact{seq_suffix}"
        if fn_cat == "no_candidate":
            return f"Ball trajectory didn't trigger any candidate generator{seq_suffix}"
        if fn_cat == "no_ball_data":
            return "No ball tracking data for this rally"
        return fn_cat or "Unknown reason"
    elif ec == "wrong_action":
        return (
            f"Detected as '{e.get('pred_action', '?')}' instead of "
            f"'{e.get('gt_action', '?')}'{seq_suffix}"
        )
    elif ec == "wrong_player":
        return (
            f"Attributed to T{e.get('pred_player_track_id', '?')} instead of "
            f"T{e.get('gt_player_track_id', '?')}{seq_suffix}"
        )
    return ""


def decoder_rescuable(e: dict) -> bool:
    """Matches the seq argmax == gt_action AND seq_prob ≥ 0.80 signature.

    Only meaningful on FN_contact + wrong_action — the decoder fuses
    emissions for the contact-accept / action-label decision. It does not
    touch player attribution, so wrong_player records are always marked
    non-rescuable here regardless of seq signal.
    """
    if e.get("error_class") == "wrong_player":
        return False
    if e.get("seq_peak_action") != e.get("gt_action"):
        return False
    return float(e.get("seq_peak_action_prob", 0.0)) >= 0.80


def seq_disagreement(e: dict) -> bool:
    """`rejected_by_classifier` FN where GBM ≈ 0 but seq ≈ 1 — the §2.1
    max-margin disagreement audit bucket."""
    if e.get("error_class") != "FN_contact":
        return False
    if e.get("fn_subcategory") != "rejected_by_classifier":
        return False
    if float(e.get("classifier_conf", 0.0)) >= 0.05:
        return False
    return float(e.get("seq_peak_nonbg_within_5f", 0.0)) >= 0.95


def generate_html(errors: list[dict[str, Any]], tags: dict[str, dict[str, str]]) -> str:
    for e in errors:
        key = f"{e['rally_id']}_{e['gt_frame']}"
        t = tags.get(key, {})
        e["_cat"] = friendly_category(e)
        e["_reason"] = friendly_reason(e)
        e["_tag"] = t.get("tag", "")
        e["_notes"] = t.get("notes", "")
        e["_decoder_rescuable"] = decoder_rescuable(e)
        e["_seq_disagreement"] = seq_disagreement(e)

    errors_json = json.dumps(errors, default=str)
    tags_json = json.dumps(tags, default=str)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Action Detection Review</title>
<style>{CSS}</style>
</head>
<body>
<div id="app"></div>
<script>
const ERRORS = {errors_json};
const TAGS = {tags_json};
{JS}
</script>
</body>
</html>"""


CSS = """
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg: #f5f6fa; --card: #fff; --border: #e2e5ec;
  --text: #1a1a2e; --muted: #6b7280; --accent: #4f46e5;
  --miss-bg: #fef2f2; --miss-border: #fca5a5; --miss-text: #991b1b;
  --wrong-bg: #fffbeb; --wrong-border: #fcd34d; --wrong-text: #92400e;
  --player-bg: #eff6ff; --player-border: #93c5fd; --player-text: #1e40af;
  --mono: "SF Mono", "Fira Code", "Cascadia Code", monospace;
}
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: var(--bg); color: var(--text); }

/* Header */
.header { background: var(--text); color: #fff; padding: 20px 32px; position: sticky; top: 0; z-index: 100; }
.header h1 { font-size: 22px; font-weight: 700; margin-bottom: 16px; }
.tabs { display: flex; gap: 4px; }
.tab {
  padding: 10px 20px; border-radius: 8px 8px 0 0; font-size: 14px; font-weight: 600;
  cursor: pointer; background: rgba(255,255,255,0.08); color: rgba(255,255,255,0.7);
  border: none; transition: all 0.15s;
}
.tab:hover { background: rgba(255,255,255,0.15); color: #fff; }
.tab.active { background: var(--bg); color: var(--text); }
.tab .count { margin-left: 6px; opacity: 0.6; font-family: var(--mono); font-size: 12px; }

/* Content area */
.content { max-width: 1200px; margin: 0 auto; padding: 24px; }

/* Summary cards */
.summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px; }
.stat-card {
  background: var(--card); border-radius: 12px; padding: 20px; border: 1px solid var(--border);
  text-align: center;
}
.stat-card .number { font-size: 36px; font-weight: 800; font-family: var(--mono); }
.stat-card .label { font-size: 13px; color: var(--muted); margin-top: 4px; text-transform: uppercase; letter-spacing: 0.5px; }
.stat-card.miss .number { color: var(--miss-text); }
.stat-card.wrong .number { color: var(--wrong-text); }
.stat-card.player .number { color: var(--player-text); }

/* Error list */
.error-list { display: flex; flex-direction: column; gap: 12px; }

/* Error card */
.error-card {
  background: var(--card); border-radius: 12px; border: 1px solid var(--border);
  overflow: hidden; transition: box-shadow 0.2s;
}
.error-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.08); }
.error-card.cat-missed_contact { border-left: 4px solid var(--miss-border); }
.error-card.cat-wrong_action_type { border-left: 4px solid var(--wrong-border); }
.error-card.cat-wrong_player { border-left: 4px solid var(--player-border); }

.card-header { display: flex; align-items: center; gap: 12px; padding: 14px 20px; cursor: pointer; }
.card-header:hover { background: rgba(0,0,0,0.02); }

.badge {
  padding: 4px 10px; border-radius: 6px; font-size: 11px; font-weight: 700;
  text-transform: uppercase; letter-spacing: 0.3px; white-space: nowrap;
}
.badge-miss { background: var(--miss-bg); color: var(--miss-text); border: 1px solid var(--miss-border); }
.badge-wrong { background: var(--wrong-bg); color: var(--wrong-text); border: 1px solid var(--wrong-border); }
.badge-player { background: var(--player-bg); color: var(--player-text); border: 1px solid var(--player-border); }

.card-info { flex: 1; min-width: 0; }
.card-title { font-size: 15px; font-weight: 600; }
.card-title .action-gt { color: #059669; font-weight: 700; }
.card-title .action-pred { color: #dc2626; font-weight: 700; }
.card-title .arrow { color: var(--muted); margin: 0 6px; }
.card-reason { font-size: 13px; color: var(--muted); margin-top: 2px; }
.card-meta { font-size: 11px; color: var(--muted); font-family: var(--mono); white-space: nowrap; }
.card-tag-indicator {
  padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: 600;
  background: #e0e7ff; color: #3730a3;
}
.card-flag {
  padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: 700;
  white-space: nowrap;
}
.card-flag.decoder { background: #dcfce7; color: #14532d; border: 1px solid #86efac; }
.card-flag.seq-dis { background: #fee2e2; color: #7f1d1d; border: 1px solid #fca5a5; }
.card-chevron { color: var(--muted); font-size: 16px; transition: transform 0.2s; }
.error-card.open .card-chevron { transform: rotate(90deg); }

/* Expanded detail */
.card-detail { display: none; border-top: 1px solid var(--border); }
.error-card.open .card-detail { display: block; }

.clip-area { background: #111; padding: 8px; text-align: center; }
.clip-area video { max-width: 100%; border-radius: 8px; max-height: 400px; }
.clip-fallback { padding: 40px; color: #666; font-size: 14px; }

.detail-content { padding: 16px 20px; }

/* GT vs Prediction comparison */
.comparison { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 16px; }
.comp-box {
  padding: 14px; border-radius: 8px; border: 1px solid var(--border);
}
.comp-box.gt { background: #f0fdf4; border-color: #86efac; }
.comp-box.pred { background: #fef2f2; border-color: #fca5a5; }
.comp-label { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px; }
.comp-box.gt .comp-label { color: #166534; }
.comp-box.pred .comp-label { color: #991b1b; }
.comp-row { display: flex; justify-content: space-between; padding: 4px 0; font-size: 13px; }
.comp-key { color: var(--muted); }
.comp-val { font-family: var(--mono); font-weight: 600; }

/* Quick context */
.context-bar {
  display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px;
  padding: 10px 14px; background: #f8fafc; border-radius: 8px; border: 1px solid var(--border);
  font-size: 12px; color: var(--muted);
}
.ctx-item { display: flex; gap: 4px; align-items: center; }
.ctx-item strong { color: var(--text); font-family: var(--mono); }

/* Feedback section */
.feedback {
  padding: 16px; background: #fafbff; border-radius: 10px; border: 1px solid #c7d2fe;
}
.feedback-title { font-size: 13px; font-weight: 700; color: #4338ca; margin-bottom: 12px; }

.tag-buttons { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 12px; }
.tag-btn {
  padding: 8px 14px; border-radius: 8px; font-size: 12px; font-weight: 600;
  cursor: pointer; border: 2px solid var(--border); background: var(--card);
  color: var(--text); transition: all 0.15s;
}
.tag-btn:hover { border-color: var(--accent); background: #eef2ff; }
.tag-btn.selected { border-color: var(--accent); background: var(--accent); color: #fff; }

.notes-input {
  width: 100%; padding: 10px 14px; border: 1px solid var(--border); border-radius: 8px;
  font-size: 13px; font-family: inherit; resize: vertical; min-height: 38px;
}
.notes-input:focus { outline: none; border-color: var(--accent); box-shadow: 0 0 0 3px rgba(79,70,229,0.1); }

/* Export bar */
.export-bar {
  position: fixed; bottom: 0; left: 0; right: 0; background: var(--card);
  border-top: 1px solid var(--border); padding: 10px 32px;
  display: flex; gap: 12px; align-items: center; justify-content: center; z-index: 100;
}
.export-bar button {
  padding: 8px 20px; border-radius: 8px; font-size: 13px; font-weight: 600;
  cursor: pointer; border: 1px solid var(--border); background: var(--card);
}
.export-bar button:hover { background: #f0f0f0; }
.export-bar button.primary { background: var(--accent); color: #fff; border-color: var(--accent); }
.export-bar button.primary:hover { background: #4338ca; }
.export-bar .status { font-size: 12px; color: var(--muted); }

body { padding-bottom: 60px; }
"""


JS = r"""
const CATEGORIES = {
  missed_contact: { label: "Contact Not Detected", badge: "badge-miss", icon: "🔴" },
  wrong_action_type: { label: "Wrong Action Type", badge: "badge-wrong", icon: "🟡" },
  wrong_player: { label: "Wrong Player", badge: "badge-player", icon: "🔵" },
};

const FEEDBACK_OPTIONS = {
  missed_contact: [
    { value: "ball_occluded_fixable", label: "Ball hidden by player — trajectory still clear", desc: "We can detect this from the trajectory change" },
    { value: "serve_off_frame", label: "Serve happened off screen", desc: "Synthetic serve detection should handle this" },
    { value: "ball_truly_lost", label: "Ball genuinely lost", desc: "No ball data at all, tracking failed" },
    { value: "soft_contact", label: "Very soft/subtle contact", desc: "Tip, deflection, or light touch — hard to detect" },
    { value: "gt_wrong", label: "GT label seems wrong", desc: "I don't think this contact actually happened" },
    { value: "looks_fixable", label: "Looks fixable (other)", desc: "There's enough signal to detect this" },
    { value: "genuinely_hard", label: "Genuinely difficult case", desc: "Even a human would struggle" },
  ],
  wrong_action_type: [
    { value: "obvious_mistake", label: "Obvious misclassification", desc: "Clearly wrong, should be easy to fix" },
    { value: "ambiguous_actions", label: "Actions look similar here", desc: "The two actions are genuinely hard to tell apart in this context" },
    { value: "court_position_issue", label: "Court position would clarify", desc: "Knowing where on court would disambiguate" },
    { value: "sequence_would_help", label: "Sequence context would help", desc: "Looking at previous/next contacts would clarify" },
    { value: "gt_wrong", label: "GT label seems wrong", desc: "I think the prediction is actually correct" },
  ],
  wrong_player: [
    { value: "clear_attribution_mistake", label: "Clear attribution mistake", desc: "Obviously wrong player — should be fixable" },
    { value: "clearly_correct_pred", label: "Prediction looks correct actually", desc: "The predicted player IS closest — GT may be wrong" },
    { value: "players_very_close", label: "Two players very close together", desc: "Genuinely hard to tell who touched it" },
    { value: "tracking_id_issue", label: "Player IDs seem swapped/wrong", desc: "Track IDs don't match what I see" },
    { value: "gt_wrong", label: "GT label seems wrong", desc: "The GT player doesn't look right" },
    { value: "attribution_correct_but_mapped_wrong", label: "Looks correct — mapping issue?", desc: "Visual proximity is right but IDs don't match" },
  ],
};

let currentTab = 'all';
let openCards = new Set();

function formatTimestamp(e) {
  const startMs = e.start_ms || 0;
  const fps = e.fps || 30;
  const frameMs = (e.gt_frame || 0) / fps * 1000;
  const totalSec = Math.round((startMs + frameMs) / 1000);
  const min = Math.floor(totalSec / 60);
  const sec = totalSec % 60;
  return min + ':' + String(sec).padStart(2, '0');
}

function lsKey(e) { return 'aed_' + e.rally_id + '_' + e.gt_frame; }
function getUserData(e) { try { return JSON.parse(localStorage.getItem(lsKey(e)) || 'null'); } catch { return null; } }
function setUserData(e, d) { const x = getUserData(e) || {}; localStorage.setItem(lsKey(e), JSON.stringify({...x, ...d})); }
function getTag(e) { return getUserData(e)?.tag || e._tag || ''; }
function getNotes(e) { return getUserData(e)?.notes ?? e._notes ?? ''; }

function init() {
  renderApp();
}

function renderApp() {
  const cats = { missed_contact: 0, wrong_action_type: 0, wrong_player: 0 };
  const tagged = { total: 0 };
  let decoderCount = 0;
  let seqDisCount = 0;
  ERRORS.forEach(e => {
    cats[e._cat] = (cats[e._cat]||0) + 1;
    if (getTag(e)) tagged.total++;
    if (e._decoder_rescuable) decoderCount++;
    if (e._seq_disagreement) seqDisCount++;
  });

  let filtered = ERRORS;
  if (currentTab === 'decoder_rescuable') {
    filtered = ERRORS.filter(e => e._decoder_rescuable);
  } else if (currentTab === 'seq_disagreement') {
    filtered = ERRORS.filter(e => e._seq_disagreement);
  } else if (currentTab !== 'all') {
    filtered = ERRORS.filter(e => e._cat === currentTab);
  }

  document.getElementById('app').innerHTML = `
    <div class="header">
      <h1>Action Detection Review</h1>
      <div class="tabs">
        <button class="tab ${currentTab==='all'?'active':''}" onclick="setTab('all')">All Errors<span class="count">${ERRORS.length}</span></button>
        <button class="tab ${currentTab==='missed_contact'?'active':''}" onclick="setTab('missed_contact')">${CATEGORIES.missed_contact.icon} Not Detected<span class="count">${cats.missed_contact}</span></button>
        <button class="tab ${currentTab==='wrong_action_type'?'active':''}" onclick="setTab('wrong_action_type')">${CATEGORIES.wrong_action_type.icon} Wrong Type<span class="count">${cats.wrong_action_type}</span></button>
        <button class="tab ${currentTab==='wrong_player'?'active':''}" onclick="setTab('wrong_player')">${CATEGORIES.wrong_player.icon} Wrong Player<span class="count">${cats.wrong_player}</span></button>
        <button class="tab ${currentTab==='decoder_rescuable'?'active':''}" onclick="setTab('decoder_rescuable')">🟢 Decoder-rescuable<span class="count">${decoderCount}</span></button>
        <button class="tab ${currentTab==='seq_disagreement'?'active':''}" onclick="setTab('seq_disagreement')">⚠️ Max GBM↔seq disagree<span class="count">${seqDisCount}</span></button>
      </div>
    </div>
    <div class="content">
      <div class="summary">
        <div class="stat-card miss"><div class="number">${cats.missed_contact}</div><div class="label">Contacts Missed</div></div>
        <div class="stat-card wrong"><div class="number">${cats.wrong_action_type}</div><div class="label">Wrong Action</div></div>
        <div class="stat-card player"><div class="number">${cats.wrong_player}</div><div class="label">Wrong Player</div></div>
        <div class="stat-card"><div class="number">${tagged.total}/${ERRORS.length}</div><div class="label">Reviewed</div></div>
      </div>
      <div class="error-list">
        ${filtered.map((e, i) => renderCard(e, ERRORS.indexOf(e))).join('')}
      </div>
    </div>
    <div class="export-bar">
      <span class="status">${tagged.total} of ${ERRORS.length} reviewed</span>
      <button onclick="exportCSV()">Export Feedback CSV</button>
      <button onclick="document.getElementById('imp').click()">Import CSV</button>
      <input type="file" id="imp" accept=".csv" style="display:none" onchange="importCSV(event)">
    </div>
  `;
}

function setTab(t) { currentTab = t; renderApp(); }

function renderCard(e, idx) {
  const cat = CATEGORIES[e._cat] || CATEGORIES.missed_contact;
  const isOpen = openCards.has(idx);
  const tag = getTag(e);
  const predAction = e.pred_action || 'NOT DETECTED';
  const rally8 = (e.rally_id || '').substring(0, 8);

  let titleHtml;
  if (e._cat === 'missed_contact') {
    titleHtml = `<span class="action-gt">${e.gt_action}</span> <span style="color:var(--muted)">was not detected</span>`;
  } else if (e._cat === 'wrong_action_type') {
    titleHtml = `Expected <span class="action-gt">${e.gt_action}</span><span class="arrow">→</span>Got <span class="action-pred">${predAction}</span>`;
  } else {
    titleHtml = `<span class="action-gt">${e.gt_action}</span> — wrong player attributed`;
  }

  const flagsHtml = [
    e._decoder_rescuable ? '<span class="card-flag decoder" title="seq argmax == gt_action AND prob ≥ 0.80 — Viterbi decoder would likely fix">decoder-fix</span>' : '',
    e._seq_disagreement ? '<span class="card-flag seq-dis" title="GBM conf < 0.05 but seq ≥ 0.95 — max-margin emission disagreement">seq↔gbm disagree</span>' : '',
  ].filter(Boolean).join(' ');

  return `
    <div class="error-card cat-${e._cat} ${isOpen?'open':''}" id="card-${idx}">
      <div class="card-header" onclick="toggle(${idx})">
        <span class="badge ${cat.badge}">${cat.label}</span>
        <div class="card-info">
          <div class="card-title">${titleHtml}</div>
          <div class="card-reason">${e._reason}</div>
        </div>
        ${flagsHtml}
        ${tag ? `<span class="card-tag-indicator">${friendlyTag(tag)}</span>` : ''}
        <span class="card-meta">${rally8} f:${e.gt_frame}</span>
        <span class="card-chevron">▶</span>
      </div>
      <div class="card-detail">
        ${isOpen ? renderDetail(e, idx) : ''}
      </div>
    </div>
  `;
}

function friendlyTag(tag) {
  const map = {
    ball_occluded_fixable: "Fixable (occluded)",
    serve_off_frame: "Off-frame serve",
    ball_truly_lost: "Ball lost",
    soft_contact: "Soft contact",
    gt_wrong: "GT wrong",
    looks_fixable: "Fixable",
    genuinely_hard: "Hard case",
    obvious_mistake: "Obvious fix",
    ambiguous_actions: "Ambiguous",
    court_position_issue: "Needs court pos",
    sequence_would_help: "Needs sequence",
    clearly_correct_pred: "Pred correct",
    players_very_close: "Close players",
    tracking_id_issue: "ID issue",
    attribution_correct_but_mapped_wrong: "Mapping issue",
  };
  return map[tag] || tag.replace(/_/g, ' ');
}

function toggle(idx) {
  if (openCards.has(idx)) openCards.delete(idx); else openCards.add(idx);
  const scrollY = window.scrollY;
  renderApp();
  window.scrollTo(0, scrollY);
  if (openCards.has(idx)) {
    setTimeout(() => {
      const el = document.getElementById('card-'+idx);
      if (el) {
        const rect = el.getBoundingClientRect();
        if (rect.top < 0 || rect.top > window.innerHeight * 0.5) {
          el.scrollIntoView({behavior:'smooth', block:'nearest'});
        }
      }
    }, 50);
  }
}

function renderDetail(e, idx) {
  const clipSrc = 'clips/' + e.rally_id + '_' + e.gt_frame + '.mp4';
  const tag = getTag(e);
  const notes = getNotes(e).replace(/"/g, '&quot;').replace(/</g, '&lt;');
  const options = FEEDBACK_OPTIONS[e._cat] || [];

  return `
    <div class="clip-area">
      <video src="${clipSrc}" controls loop muted playsinline preload="metadata"
        onerror="this.outerHTML='<div class=clip-fallback>Clip not rendered yet. Run: uv run python scripts/render_action_error_strips.py</div>'"
      ></video>
    </div>
    <div class="detail-content">
      ${renderComparison(e)}
      ${renderContext(e)}
      <div class="feedback">
        <div class="feedback-title">What do you see?</div>
        <div class="tag-buttons">
          ${options.map(o => `
            <button class="tag-btn ${tag===o.value?'selected':''}"
              onclick="setTag(${idx},'${o.value}')" title="${o.desc}">
              ${o.label}
            </button>
          `).join('')}
        </div>
        <input class="notes-input" type="text" placeholder="Add notes about what you see..."
          value="${notes}" onchange="setNotes(${idx}, this.value)">
      </div>
    </div>
  `;
}

function renderComparison(e) {
  const seqRow = (e.seq_peak_nonbg_within_5f !== undefined && e.seq_peak_nonbg_within_5f > 0)
    ? `<div class="comp-row"><span class="comp-key">Seq (MS-TCN++) ±5f</span><span class="comp-val">${e.seq_peak_action || '?'} ${(e.seq_peak_action_prob||0).toFixed(2)} (peak ${(e.seq_peak_nonbg_within_5f||0).toFixed(2)})</span></div>`
    : '';
  if (e._cat === 'missed_contact') {
    return `
      <div class="comparison">
        <div class="comp-box gt">
          <div class="comp-label">Expected (Ground Truth)</div>
          <div class="comp-row"><span class="comp-key">Action</span><span class="comp-val">${e.gt_action}</span></div>
          <div class="comp-row"><span class="comp-key">Frame</span><span class="comp-val">${e.gt_frame}</span></div>
          <div class="comp-row"><span class="comp-key">Player</span><span class="comp-val">T${e.gt_player_track_id} (green box)</span></div>
          ${seqRow}
        </div>
        <div class="comp-box pred">
          <div class="comp-label">What Happened</div>
          <div class="comp-row"><span class="comp-key">Detection</span><span class="comp-val">Not detected</span></div>
          <div class="comp-row"><span class="comp-key">Failure</span><span class="comp-val">${e.fn_subcategory || '?'}</span></div>
          <div class="comp-row"><span class="comp-key">GBM conf</span><span class="comp-val">${(e.classifier_conf||0).toFixed(3)}</span></div>
          <div class="comp-row"><span class="comp-key">Nearest cand</span><span class="comp-val">${e.nearest_cand_dist !== undefined ? e.nearest_cand_dist + 'f' : '–'}</span></div>
          <div class="comp-row"><span class="comp-key">Player dist</span><span class="comp-val">${e.player_distance !== undefined && isFinite(e.player_distance) ? e.player_distance.toFixed(3) : '∞'}</span></div>
        </div>
      </div>
    `;
  }
  if (e._cat === 'wrong_action_type') {
    return `
      <div class="comparison">
        <div class="comp-box gt">
          <div class="comp-label">Expected (Ground Truth)</div>
          <div class="comp-row"><span class="comp-key">Action</span><span class="comp-val">${e.gt_action}</span></div>
          <div class="comp-row"><span class="comp-key">Frame</span><span class="comp-val">${e.gt_frame}</span></div>
          <div class="comp-row"><span class="comp-key">Player</span><span class="comp-val">T${e.gt_player_track_id}</span></div>
          ${seqRow}
        </div>
        <div class="comp-box pred">
          <div class="comp-label">Prediction</div>
          <div class="comp-row"><span class="comp-key">Action</span><span class="comp-val">${e.pred_action}</span></div>
          <div class="comp-row"><span class="comp-key">Frame</span><span class="comp-val">${e.pred_frame || '—'}</span></div>
          <div class="comp-row"><span class="comp-key">Player</span><span class="comp-val">T${e.pred_player_track_id || '?'}</span></div>
          <div class="comp-row"><span class="comp-key">GBM conf</span><span class="comp-val">${(e.classifier_conf||0).toFixed(3)}</span></div>
        </div>
      </div>
    `;
  }
  return `
    <div class="comparison">
      <div class="comp-box gt">
        <div class="comp-label">Expected (Ground Truth)</div>
        <div class="comp-row"><span class="comp-key">Action</span><span class="comp-val">${e.gt_action}</span></div>
        <div class="comp-row"><span class="comp-key">Player</span><span class="comp-val">T${e.gt_player_track_id} (green box)</span></div>
        <div class="comp-row"><span class="comp-key">Frame</span><span class="comp-val">${e.gt_frame}</span></div>
        ${seqRow}
      </div>
      <div class="comp-box pred">
        <div class="comp-label">Prediction</div>
        <div class="comp-row"><span class="comp-key">Action</span><span class="comp-val">${e.pred_action}</span></div>
        <div class="comp-row"><span class="comp-key">Player</span><span class="comp-val">T${e.pred_player_track_id || '?'} (red box)</span></div>
        <div class="comp-row"><span class="comp-key">Frame</span><span class="comp-val">${e.pred_frame || '—'}</span></div>
      </div>
    </div>
  `;
}

function renderContext(e) {
  const rq = e.rally_quality || {};
  const seqStr = (e.seq_peak_nonbg_within_5f !== undefined && e.seq_peak_nonbg_within_5f > 0)
    ? `${e.seq_peak_action || '?'} ${(e.seq_peak_action_prob||0).toFixed(2)}`
    : '—';
  return `
    <div class="context-bar">
      <div class="ctx-item">Ball coverage: <strong>${rq.ball_coverage_pct || '?'}%</strong></div>
      <div class="ctx-item">Ball max gap: <strong>${rq.ball_max_gap_frames || '?'}f</strong></div>
      <div class="ctx-item">Players tracked: <strong>${rq.player_track_count || '?'}</strong></div>
      <div class="ctx-item">Seq argmax ±5f: <strong>${seqStr}</strong></div>
      <div class="ctx-item">Rally: <strong>${(e.rally_id||'').substring(0,8)}</strong></div>
      <div class="ctx-item">Video: <strong>${(e.video_id||'').substring(0,8)}</strong> ${e.video_name ? '(' + e.video_name + ')' : ''}</div>
      <div class="ctx-item">Time: <strong>${formatTimestamp(e)}</strong></div>
      <div class="ctx-item">FPS: <strong>${e.fps}</strong></div>
    </div>
  `;
}

function setTag(idx, val) {
  const e = ERRORS[idx];
  const cur = getTag(e);
  const newVal = cur === val ? '' : val;
  e._tag = newVal;
  setUserData(e, { tag: newVal });
  renderApp();
}
function setNotes(idx, val) {
  setUserData(ERRORS[idx], { notes: val });
  ERRORS[idx]._notes = val;
}

function exportCSV() {
  let csv = 'rally_id,gt_frame,error_class,gt_action,tag,notes\n';
  for (const e of ERRORS) {
    const tag = getTag(e);
    if (!tag) continue;
    const notes = getNotes(e).replace(/"/g, '""');
    csv += `${e.rally_id},${e.gt_frame},${e.error_class},${e.gt_action},${tag},"${notes}"\n`;
  }
  const blob = new Blob([csv], {type:'text/csv'});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
  a.download = 'review_feedback.csv'; a.click();
}
function importCSV(ev) {
  const f = ev.target.files[0]; if (!f) return;
  const r = new FileReader();
  r.onload = function(e) {
    const lines = e.target.result.split('\n');
    let n = 0;
    for (let i = 1; i < lines.length; i++) {
      if (!lines[i].trim()) continue;
      const m = lines[i].match(/^([^,]+),(\d+),([^,]*),([^,]*),([^,]*)(?:,"(.*)")?/);
      if (m) {
        const err = ERRORS.find(e => e.rally_id === m[1] && String(e.gt_frame) === m[2]);
        if (err) { setUserData(err, {tag: m[5], notes: m[6]||''}); err._tag = m[5]; n++; }
      }
    }
    alert('Imported ' + n + ' reviews'); renderApp();
  };
  r.readAsText(f);
}

init();
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build action error review dashboard")
    parser.add_argument("--corpus", type=Path, default=OUTPUT_DIR / "corpus_annotated.jsonl")
    parser.add_argument("--tags", type=Path, default=OUTPUT_DIR / "tags.csv")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "dashboard.html")
    args = parser.parse_args()

    errors = load_corpus(args.corpus)
    print(f"Loaded {len(errors)} errors from {args.corpus}")
    tags = load_tags(args.tags)
    if tags:
        print(f"Loaded {len(tags)} tags from {args.tags}")

    html = generate_html(errors, tags)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html)
    print(f"Dashboard: {args.output} ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
