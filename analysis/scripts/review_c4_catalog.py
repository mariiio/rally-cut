"""Friendly C-4 catalog review helper.

Two-step workflow:
  1. `--generate` reads the catalog CSV, auto-classifies the obvious buckets,
     samples ~N stratified cards from what's left, and writes a markdown
     file where each card has a blank `root_cause:` line you fill in.
  2. `--finalize` parses the filled markdown, merges user-fills with the
     auto-class for unfilled cards, writes a final review markdown with
     pattern distribution + placeholder-vs-truth agreement.

High-precision auto-classification: prefers leaving a row `unclassified`
over guessing. The card sample is then *stratified* across (prev, curr)
action cells so dominant patterns get representative coverage.

Spec: docs/superpowers/specs/2026-05-13-sub-2b-coherence-repair-design.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT_CAUSE_VOCAB = (
    "synthetic_serve_cascade",
    "attribution_swap_prev",
    "attribution_swap_curr",
    "genuine_double_touch",
    "ghost_contact_prev",
    "ghost_contact_curr",
    "wrong_action_type",
    "block_exception_miss",
    "other",
    "unclassified",
)

# NOTE: There is NO beach-2v2 exception for same-player consecutive contacts.
# The volleyball rule is strict: consecutive contacts must be by different
# players. The only exception is when the previous action was a `block`
# (the block exception is enforced in the C-4 detector itself, so those
# pairs never reach this catalog).
#
# The auto-classifier therefore does NOT auto-tag any pair as
# `genuine_double_touch`. That label remains available for hand-
# classification when a human review concludes the same player legitimately
# played twice for a non-block reason (e.g., a deflection off a block that
# was mis-classified as something else, or a ghost contact that the human
# spot-check decides to dismiss).


def _f(row: dict[str, str], key: str) -> float:
    """Parse a float from CSV cell. NaN on empty/invalid."""
    raw = row.get(key, "")
    if not raw:
        return math.nan
    try:
        return float(raw)
    except ValueError:
        return math.nan


def _bool(row: dict[str, str], key: str) -> bool:
    return row.get(key, "").strip().lower() in ("true", "1")


def auto_classify(row: dict[str, str]) -> tuple[str, str]:
    """Return (root_cause, reason). Prefers 'unclassified' when uncertain.

    Rules are intentionally conservative — better to leave the row blank
    than mislabel it. The hand-review fills in the unclassified cases.
    """
    if _bool(row, "co_c3_fires"):
        return "synthetic_serve_cascade", "co_c3_fires=true"

    geom_prev = row.get("signal_team_geometry_prev", "")
    geom_curr = row.get("signal_team_geometry_curr", "")
    alt_prev = _f(row, "prev_best_same_team_alt_ratio")
    alt_curr = _f(row, "curr_best_same_team_alt_ratio")
    type_fit_prev = row.get("signal_type_fit_prev", "")
    type_fit_curr = row.get("signal_type_fit_curr", "")
    conf_prev = _f(row, "conf_prev")
    conf_curr = _f(row, "conf_curr")

    # Strong attribution_swap_curr: geometry says wrong-team-picked AND a
    # same-team alt is meaningfully closer AND type-fit doesn't rule curr out.
    if (
        geom_curr == "violates"
        and math.isfinite(alt_curr) and alt_curr < 0.5
        and type_fit_curr != "wrong"
    ):
        return (
            "attribution_swap_curr",
            f"geom_curr=violates, alt_curr={alt_curr:.2f}<0.5",
        )

    if (
        geom_prev == "violates"
        and math.isfinite(alt_prev) and alt_prev < 0.5
        and type_fit_prev != "wrong"
    ):
        return (
            "attribution_swap_prev",
            f"geom_prev=violates, alt_prev={alt_prev:.2f}<0.5",
        )

    # Ghost contact: very low confidence on one side AND a double-action-type
    # pair (curr.action == prev.action), which suggests a spurious second
    # contact-detection fire on the same physical event.
    pair = (row.get("action_prev_type", ""), row.get("action_curr_type", ""))
    if pair[0] == pair[1]:
        if math.isfinite(conf_prev) and conf_prev < 0.4:
            return ("ghost_contact_prev", f"same-type pair {pair}, conf_prev={conf_prev:.2f}<0.4")
        if math.isfinite(conf_curr) and conf_curr < 0.4:
            return ("ghost_contact_curr", f"same-type pair {pair}, conf_curr={conf_curr:.2f}<0.4")

    return "unclassified", "no rule matched"


def _stratified_sample(
    rows: list[dict[str, str]],
    target_n: int,
    seed: int = 0,
) -> list[dict[str, str]]:
    """Sample target_n rows stratified by (prev_action, curr_action) cell.

    Proportional to cell frequency, but capped so no cell dominates.
    """
    if len(rows) <= target_n:
        return list(rows)
    rng = random.Random(seed)
    by_cell: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        cell = (r.get("action_prev_type", ""), r.get("action_curr_type", ""))
        by_cell[cell].append(r)

    # Quota per cell: proportional to size, but at least 1, capped at 5.
    total = len(rows)
    sampled: list[dict[str, str]] = []
    for cell, cell_rows in sorted(by_cell.items(), key=lambda kv: -len(kv[1])):
        quota = max(1, round(target_n * len(cell_rows) / total))
        quota = min(quota, 5, len(cell_rows))
        sampled.extend(rng.sample(cell_rows, quota))
        if len(sampled) >= target_n:
            break
    return sampled[:target_n]


def _fmt_candidates(raw: str) -> str:
    """Render a JSON candidates list as a compact human-readable string."""
    if not raw:
        return "(none)"
    try:
        items = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw
    if not items:
        return "(none)"
    parts = []
    for entry in items:
        if isinstance(entry, list) and len(entry) >= 3:
            tid, dist, team = entry[0], entry[1], entry[2]
            parts.append(f"tid={tid} dist={float(dist):.3f} team={team}")
    return " | ".join(parts) if parts else "(empty)"


def _render_card(idx: int, total: int, row: dict[str, str]) -> str:
    """One markdown card per violation."""
    auto, reason = auto_classify(row)
    pair = f"`{row['action_prev_type']}` → `{row['action_curr_type']}`"
    conf_prev = _f(row, "conf_prev")
    conf_curr = _f(row, "conf_curr")
    alt_prev = _f(row, "prev_best_same_team_alt_ratio")
    alt_curr = _f(row, "curr_best_same_team_alt_ratio")
    co_tags: list[str] = []
    for k in ("co_c1_fires", "co_c2_fires", "co_c3_fires"):
        if _bool(row, k):
            co_tags.append(k[3:].upper())
    pid = row.get("co_pid_invariant_fires", "")
    if pid:
        co_tags.append(f"PID={pid}")
    co_str = ", ".join(co_tags) if co_tags else "(none)"

    return (
        f"## Card {idx}/{total} — pair {pair} player {row['player_id']} (team {row['team_label']})\n"
        f"\n"
        f"- **Rally:** `{row['rally_id']}` (video `{row['video_id']}`)\n"
        f"- **Frames:** prev={row['frame_prev']} ({row['action_prev_type']}, "
        f"conf={conf_prev:.2f}) → curr={row['frame_curr']} ({row['action_curr_type']}, "
        f"conf={conf_curr:.2f})\n"
        f"- **Distances:** prev_player_dist={row.get('prev_player_dist', '?')}, "
        f"curr_player_dist={row.get('curr_player_dist', '?')}\n"
        f"- **Top-3 prev candidates:** {_fmt_candidates(row.get('prev_top3_candidates', ''))}\n"
        f"- **Top-3 curr candidates:** {_fmt_candidates(row.get('curr_top3_candidates', ''))}\n"
        f"- **Signals:** type_fit prev=`{row.get('signal_type_fit_prev', '?')}`, "
        f"curr=`{row.get('signal_type_fit_curr', '?')}` "
        f"| team_geom prev=`{row.get('signal_team_geometry_prev', '?')}`, "
        f"curr=`{row.get('signal_team_geometry_curr', '?')}` "
        f"| alt_ratio prev={alt_prev:.2f}, curr={alt_curr:.2f}\n"
        f"- **Co-violations:** {co_str}\n"
        f"- **Auto-class:** `{auto}` _(reason: {reason})_\n"
        f"- **Placeholder rec:** `{row.get('repair_recommendation', '?')}`\n"
        f"\n"
        f"**root_cause:** \n"
        f"<!-- leave blank to accept auto-class; or replace with one of: "
        f"synthetic_serve_cascade | attribution_swap_prev | attribution_swap_curr | "
        f"genuine_double_touch | ghost_contact_prev | ghost_contact_curr | "
        f"wrong_action_type | block_exception_miss | other -->\n"
        f"\n---\n\n"
    )


def cmd_generate(args: argparse.Namespace) -> int:
    in_path: Path = args.input
    out_path: Path = args.cards
    target_n: int = args.sample
    seed: int = args.seed

    rows = list(csv.DictReader(in_path.open()))
    print(f"[review] loaded {len(rows)} catalog rows from {in_path}", flush=True)

    classified: list[tuple[dict[str, str], str, str]] = []
    auto_counter: Counter[str] = Counter()
    for r in rows:
        cls, reason = auto_classify(r)
        classified.append((r, cls, reason))
        auto_counter[cls] += 1

    print("[review] auto-class distribution:", flush=True)
    for cls, count in auto_counter.most_common():
        pct = 100 * count / len(rows) if rows else 0
        print(f"  {cls:32s} {count:4d} ({pct:5.1f}%)", flush=True)

    unclassified = [r for r, c, _ in classified if c == "unclassified"]
    sampled = _stratified_sample(unclassified, target_n=target_n, seed=seed)
    sampled_ids = {(r["rally_id"], r["pair_idx"]) for r in sampled}
    print(
        f"[review] sampled {len(sampled)} of {len(unclassified)} unclassified rows "
        f"(stratified by (prev,curr) cell, seed={seed})",
        flush=True,
    )

    # Persist the full auto-class map so finalize can merge it back.
    autoclass_sidecar = out_path.with_suffix(".autoclass.json")
    autoclass_payload: list[dict[str, str]] = [
        {
            "rally_id": r["rally_id"],
            "pair_idx": r["pair_idx"],
            "auto_class": c,
            "auto_reason": reason,
            "sampled": str((r["rally_id"], r["pair_idx"]) in sampled_ids),
        }
        for r, c, reason in classified
    ]
    autoclass_sidecar.write_text(json.dumps(autoclass_payload, indent=2))
    print(f"[review] wrote auto-class sidecar to {autoclass_sidecar}", flush=True)

    # Render cards markdown.
    lines: list[str] = []
    lines.append("# C-4 Hand-Review Cards (sampled subset)\n")
    lines.append(
        f"Generated from `{in_path.name}` with seed={seed}, sample target "
        f"N={target_n}. {len(rows)} total catalog rows; "
        f"{auto_counter['unclassified']} unclassified by auto-rules; this "
        f"file presents **{len(sampled)} stratified samples** for hand review.\n\n"
    )
    lines.append("## Auto-class distribution (full catalog)\n\n")
    lines.append("| class | count | % |\n")
    lines.append("|---|---:|---:|\n")
    for cls, count in auto_counter.most_common():
        pct = 100 * count / len(rows) if rows else 0
        lines.append(f"| `{cls}` | {count} | {pct:.1f}% |\n")
    lines.append("\n")
    lines.append(
        f"For each sampled card below, the **root_cause:** line is blank. "
        f"Fill it with one of the vocabulary terms OR leave it blank to "
        f"accept the auto-class (which is `unclassified` here, so you "
        f"actually do need to pick something — or accept that this card "
        f"will remain `unclassified` in the final review). When done, run:\n\n"
        f"```\nuv run python -u scripts/review_c4_catalog.py finalize \\\n"
        f"    --cards {out_path} \\\n"
        f"    --input {in_path} \\\n"
        f"    --review {out_path.with_name(out_path.stem.replace('_cards', '') + '_review.md')}\n```\n\n"
    )
    lines.append("---\n\n")
    for i, r in enumerate(sampled, start=1):
        lines.append(_render_card(i, len(sampled), r))

    out_path.write_text("".join(lines))
    print(f"[review] wrote {len(sampled)} cards to {out_path}", flush=True)
    return 0


# Match a card's leading H2 line and capture rally + pair_idx from the URL-safe parts.
_CARD_HEADER_RE = re.compile(r"^## Card \d+/\d+ — pair `[^`]+` → `[^`]+` player ")
_RALLY_LINE_RE = re.compile(r"^- \*\*Rally:\*\* `([^`]+)` \(video `[^`]+`\)")
_ROOT_CAUSE_LINE_RE = re.compile(r"^\*\*root_cause:\*\* *(.*)$")
_FRAME_LINE_RE = re.compile(r"^- \*\*Frames:\*\* prev=(\d+) ")


def _parse_cards(md_path: Path) -> dict[tuple[str, str], str]:
    """Return {(rally_id, pair_idx): user_root_cause} from a filled cards file.

    Empty root_cause values mean "accept auto-class"; we still record them
    as "" so finalize can distinguish "saw but didn't fill" from "missed".
    """
    text = md_path.read_text()
    out: dict[tuple[str, str], str] = {}

    # Split on the H2 boundary; iterate each card block.
    blocks = re.split(r"(?=^## Card )", text, flags=re.MULTILINE)
    for block in blocks:
        if not _CARD_HEADER_RE.match(block):
            continue
        rally_id: str | None = None
        curr_frame: str | None = None
        root_cause = ""
        for line in block.splitlines():
            m = _RALLY_LINE_RE.match(line)
            if m and rally_id is None:
                rally_id = m.group(1)
            m = _FRAME_LINE_RE.match(line)
            if m and curr_frame is None:
                # Use frame_prev as a fallback identifier; we'll match on
                # rally_id + frame to recover pair_idx during merge.
                curr_frame = m.group(1)
            m = _ROOT_CAUSE_LINE_RE.match(line)
            if m:
                root_cause = m.group(1).strip()
        if rally_id is None:
            continue
        # Use (rally_id, frame_prev) as the natural key. pair_idx is derivable
        # from the catalog row but frame_prev is in the card text directly.
        out[(rally_id, curr_frame or "")] = root_cause
    return out


def cmd_finalize(args: argparse.Namespace) -> int:
    cards_path: Path = args.cards
    in_path: Path = args.input
    out_path: Path = args.review

    rows = list(csv.DictReader(in_path.open()))
    autoclass_sidecar = cards_path.with_suffix(".autoclass.json")
    if not autoclass_sidecar.exists():
        print(
            f"[review] missing auto-class sidecar at {autoclass_sidecar}; "
            f"did you run --generate first?",
            file=sys.stderr,
        )
        return 1
    autoclass_payload = json.loads(autoclass_sidecar.read_text())
    autoclass_by_key: dict[tuple[str, str], dict[str, str]] = {
        (e["rally_id"], e["pair_idx"]): e for e in autoclass_payload
    }

    # Parse the cards file for any user-filled root_cause lines.
    user_fills = _parse_cards(cards_path)

    final_labels: list[tuple[dict[str, str], str, str]] = []  # row, label, source
    for r in rows:
        key = (r["rally_id"], r["pair_idx"])
        auto = autoclass_by_key.get(key, {})
        auto_class = auto.get("auto_class", "unclassified")
        # Match by (rally_id, frame_prev) since that's what the card carried.
        card_key = (r["rally_id"], str(r["frame_prev"]))
        user_value = user_fills.get(card_key, None)
        if user_value:
            label = user_value
            source = "user"
        else:
            label = auto_class
            source = "auto"
        final_labels.append((r, label, source))

    # Compute pattern distribution + placeholder agreement.
    label_counter: Counter[str] = Counter(lab for _, lab, _ in final_labels)
    source_counter: Counter[str] = Counter(src for _, _, src in final_labels)
    # Placeholder-vs-truth: for each row whose label != 'unclassified', does
    # placeholder agree with the implied direction?
    agreement = {"agree": 0, "disagree": 0, "n_a": 0}
    for r, label, _ in final_labels:
        placeholder = r.get("repair_recommendation", "")
        if label in ("unclassified", "wrong_action_type", "other", "block_exception_miss"):
            agreement["n_a"] += 1
            continue
        # Map labels to expected placeholder verdicts.
        expected = {
            "synthetic_serve_cascade": "skip",  # let upstream fix surface
            "attribution_swap_prev": "repair_prev",
            "attribution_swap_curr": "repair_curr",
            "genuine_double_touch": "skip",
            "ghost_contact_prev": "skip",
            "ghost_contact_curr": "skip",
        }.get(label, None)
        if expected is None:
            agreement["n_a"] += 1
        elif placeholder == expected:
            agreement["agree"] += 1
        else:
            agreement["disagree"] += 1

    # Write the review markdown.
    lines: list[str] = []
    lines.append("# C-4 Phase 1 → Phase 2 Gated Review\n")
    lines.append(f"Generated from {len(rows)} catalog rows in `{in_path.name}`.\n\n")
    lines.append("## Label sources\n\n")
    lines.append("| source | count |\n|---|---:|")
    for src, count in source_counter.most_common():
        lines.append(f"| {src} | {count} |")
    lines.append("\n## Final label distribution\n\n")
    lines.append("| root_cause | count | % |\n|---|---:|---:|")
    for label, count in label_counter.most_common():
        pct = 100 * count / len(rows) if rows else 0
        lines.append(f"| `{label}` | {count} | {pct:.1f}% |")
    total_agree = agreement["agree"] + agreement["disagree"]
    agree_pct = 100 * agreement["agree"] / total_agree if total_agree else 0
    lines.append("\n## Placeholder-vs-truth agreement\n\n")
    lines.append(
        f"- **Agree:** {agreement['agree']} ({agree_pct:.1f}% of decidable)\n"
        f"- **Disagree:** {agreement['disagree']}\n"
        f"- **N/A:** {agreement['n_a']} (label has no canonical placeholder mapping)\n"
    )
    lines.append("\n## Phase 2 viability call\n\n")
    swaps = label_counter["attribution_swap_prev"] + label_counter["attribution_swap_curr"]
    swap_pct = 100 * swaps / len(rows) if rows else 0
    legit = label_counter["genuine_double_touch"]
    legit_pct = 100 * legit / len(rows) if rows else 0
    unclassified = label_counter["unclassified"]
    upstream = label_counter["synthetic_serve_cascade"]
    lines.append(
        f"Of {len(rows)} C-4 violations:\n"
        f"- **{swaps} ({swap_pct:.1f}%) are attribution swaps** — Phase 2 repair target.\n"
        f"- **{legit} ({legit_pct:.1f}%) are genuine beach-2v2 double-touches** — Phase 2 must NOT repair these.\n"
        f"- **{upstream} are upstream synthetic-serve cascades** — fix upstream, not in Phase 2.\n"
        f"- **{unclassified} remain unclassified** — auto-rules + sample didn't reach them.\n\n"
        f"**Recommendation:** "
    )
    if swaps >= 20:
        lines.append(
            "Phase 2 has clear signal. Design the multi-signal repair rule "
            "from the attribution_swap_prev/curr rows; gate by skip-when-"
            "genuine-double-touch-pattern. A/B on the 22-rally panel before "
            "default-ON.\n"
        )
    elif swaps >= 5:
        lines.append(
            "Phase 2 has modest signal. Phase 2 viable but expect <5pp panel "
            "lift. Consider whether the ROI justifies the build cost.\n"
        )
    else:
        lines.append(
            "Phase 2 has weak signal. Most violations are legitimate play or "
            "upstream-driven. Park Phase 2 and revisit after upstream fixes "
            "(synthetic-serve placement, contact-detection recall, GT expansion).\n"
        )

    out_path.write_text("".join(line + "\n" for line in lines))
    print(f"[review] wrote review to {out_path}", flush=True)

    # Also rewrite the CSV with root_cause filled in.
    csv_out = in_path.with_name(in_path.stem + "_labeled" + in_path.suffix)
    fieldnames = list(rows[0].keys()) if rows else []
    if "label_source" not in fieldnames:
        fieldnames.append("label_source")
    with csv_out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r, label, source in final_labels:
            r["root_cause"] = label
            r["label_source"] = source
            w.writerow(r)
    print(f"[review] wrote labeled CSV to {csv_out}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)

    gen = subs.add_parser("generate", help="Write cards markdown.")
    gen.add_argument("--input", type=Path, required=True,
                     help="Path to catalog CSV (from catalog_c4_violations.py).")
    gen.add_argument("--cards", type=Path, required=True,
                     help="Path to write the cards markdown.")
    gen.add_argument("--sample", type=int, default=30,
                     help="Number of cards to sample for hand review (default 30).")
    gen.add_argument("--seed", type=int, default=0,
                     help="Random seed for stratified sample.")
    gen.set_defaults(func=cmd_generate)

    fin = subs.add_parser("finalize", help="Parse cards + write review.")
    fin.add_argument("--cards", type=Path, required=True,
                     help="Path to the (filled) cards markdown.")
    fin.add_argument("--input", type=Path, required=True,
                     help="Path to catalog CSV (same as generate's --input).")
    fin.add_argument("--review", type=Path, required=True,
                     help="Path to write the final review markdown.")
    fin.set_defaults(func=cmd_finalize)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
