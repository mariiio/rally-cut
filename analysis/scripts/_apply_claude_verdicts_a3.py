"""Patch Claude's first-pass vision verdicts into the A3 probe v2 results.

This is a one-off helper: takes the existing JSON output of
``probe_a3_block_reclassification.py`` and merges Claude's per-case
verdicts into each case, recomputes the aggregate, and regenerates
the HTML + markdown so the page reflects the first-pass call.

Usage:
    uv run python scripts/_apply_claude_verdicts_a3.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

# Reuse the probe module's HTML template + write helpers.
from scripts.probe_a3_block_reclassification import (  # noqa: E402
    _HTML_TEMPLATE,
    HTML_PATH,
    JSON_PATH,
    MD_PATH,
)

# Per-case Claude vision verdicts from Phase 4 manual inspection (v2 probe).
# Keyed by idx (1..10).
CLAUDE_VERDICTS: dict[int, tuple[str, str]] = {
    1: ("block",
        "F5 canonical (loose-only, strong). Suspect p2(A) at the net with "
        "arms reaching up; ball directly above his hands. Team B player "
        "(p1 blue) adjacent. Classic net duel — wrist at net height. "
        "(a)′ pass head-near-ball; pose detected wrist above net."),
    2: ("block",
        "STRICT, strong. p1(A) jumping high at the left side of the net "
        "with arms above the net plane; ball at wrist level. Team B "
        "player ducking right at the net opposite him. Textbook block."),
    3: ("attack",
        "STRICT, moderate (b unknown). p3(B) is a small distant figure on "
        "back court holding ball at face level. NOT at net — the net is "
        "far in the background. dc=11° but no jump, no above-net arms. "
        "False positive — not a block."),
    4: ("block",
        "STRICT, strong. p1(A) at right side of net jumping with arms "
        "raised above the net plane; ball at hand at top of net. Team B "
        "red player adjacent at net opposite. Classic block, ball "
        "deflected (dc=50.5°)."),
    5: ("block",
        "STRICT, strong. p2(A) jumping at the net with both arms raised "
        "high above net; ball directly above hands. Teammate p1(A) "
        "jumping with him (double block). dc=7° tight deflection."),
    6: ("block",
        "STRICT, strong. p1(A) jumping HIGH at the net with arms straight "
        "up, ball at hands at top of jump. Classic block pose. dc=82° is "
        "near the 90° boundary but consistent with a hard-hit block where "
        "the ball reverses sharply at the net. prev=set(B) means team B's "
        "setter set their teammate; team A's blocker intercepted at net."),
    7: ("attack",
        "LOOSE-only, moderate (b unknown). p1(A) is small figure at far "
        "back-left near net, bent over with arms forward — receive/dig "
        "posture, not block. dc=0.1° means ball did NOT deflect (block "
        "would deflect). Mis-typed attack; not a block."),
    8: ("block",
        "LOOSE-only, strong. p1(A) jumping high at center net with arms "
        "raised; ball at wrist at top of net. Three team B players "
        "stacked at net below — solo blocker against multiple opponents. "
        "Classic block."),
    9: ("attack",
        "LOOSE-only, moderate (b unknown). p2(B) small distant figure at "
        "back, ball is well ABOVE him (not contacting). dc=1.2° — no "
        "deflection. Likely no real contact at all; mis-classified attack. "
        "Not a block."),
    10: ("block",
         "LOOSE-only, strong. p3(A) at right side of net jumping with arm "
         "extended upward; wrist at the top of the net. Ball at hand "
         "level. Two team B players (red) on opposing side of net in "
         "front of him. Single blocker. dc=13.4° clean deflection."),
}


def main() -> int:
    data = json.loads(JSON_PATH.read_text())
    cases = data["cases"]
    if len(cases) != len(CLAUDE_VERDICTS):
        print(f"WARNING: cases={len(cases)} but verdicts={len(CLAUDE_VERDICTS)}")

    for c in cases:
        idx = c["idx"]
        if idx in CLAUDE_VERDICTS:
            verdict, note = CLAUDE_VERDICTS[idx]
            c["claude_verdict"] = verdict
            c["claude_verdict_note"] = note

    # Recompute aggregate including per-variant + per-confidence tally.
    claude_tally = {"block": 0, "attack": 0, "ambig": 0, "unset": 0}
    by_variant = {"strict": {"n": 0, "block": 0, "attack": 0, "ambig": 0, "unset": 0},
                  "loose-only": {"n": 0, "block": 0, "attack": 0, "ambig": 0, "unset": 0}}
    by_conf = {"strong": {"n": 0, "block": 0, "attack": 0, "ambig": 0, "unset": 0},
               "moderate": {"n": 0, "block": 0, "attack": 0, "ambig": 0, "unset": 0},
               "weak": {"n": 0, "block": 0, "attack": 0, "ambig": 0, "unset": 0},
               "none": {"n": 0, "block": 0, "attack": 0, "ambig": 0, "unset": 0}}
    for c in cases:
        v = c.get("claude_verdict") if c.get("claude_verdict") in claude_tally else "unset"
        claude_tally[v] += 1
        var = c.get("variant")
        if var in by_variant:
            by_variant[var]["n"] += 1
            by_variant[var][v] += 1
        ct = c.get("confidence_tier")
        if ct in by_conf:
            by_conf[ct]["n"] += 1
            by_conf[ct][v] += 1

    agg = data["agg"]
    agg["claude_tally"] = claude_tally
    agg["claude_block_count"] = claude_tally["block"]
    agg["by_variant"] = by_variant
    agg["by_confidence_tier"] = by_conf
    agg["verdict"] = (
        "SHIP A3" if claude_tally["block"] >= 7 else "NO-SHIP A3"
    )

    JSON_PATH.write_text(json.dumps(data, indent=2))
    print(f"Updated {JSON_PATH}")
    print(f"Claude tally: {claude_tally}")
    print(f"Block count: {claude_tally['block']} / {len(cases)} "
          f"(threshold ≥ 7 → {agg['verdict']})")
    print(f"By variant: {by_variant}")
    print(f"By confidence: {by_conf}")

    # Regenerate HTML.
    html = _HTML_TEMPLATE.replace("__DATA_JSON__", json.dumps(data, indent=2))
    HTML_PATH.write_text(html)
    print(f"Regenerated HTML: {HTML_PATH}")

    # Regenerate markdown — append a "Claude verdicts" section.
    md_lines = MD_PATH.read_text().splitlines()
    # Strip any prior Claude section.
    for i, line in enumerate(md_lines):
        if line.startswith("## Claude vision first-pass"):
            md_lines = md_lines[:i]
            break

    md_lines.append("")
    md_lines.append("## Claude vision first-pass verdicts (Phase 4)")
    md_lines.append("")
    md_lines.append(f"- block: **{claude_tally['block']}**")
    md_lines.append(f"- attack: **{claude_tally['attack']}**")
    md_lines.append(f"- ambiguous: **{claude_tally['ambig']}**")
    md_lines.append("")
    md_lines.append(f"**Threshold ≥ 7 blocks → {agg['verdict']}**")
    md_lines.append("")
    md_lines.append("### Per-variant breakdown")
    md_lines.append("")
    md_lines.append("| variant | n | 🟦 block | 🟧 attack | ⚠️ ambig |")
    md_lines.append("|---------|---|---------|----------|---------|")
    for var, st in by_variant.items():
        md_lines.append(f"| {var} | {st['n']} | {st['block']} | {st['attack']} | {st['ambig']} |")
    md_lines.append("")
    md_lines.append("### Per-confidence breakdown")
    md_lines.append("")
    md_lines.append("| confidence | n | 🟦 block | 🟧 attack | ⚠️ ambig |")
    md_lines.append("|------------|---|---------|----------|---------|")
    for ct, st in by_conf.items():
        if st["n"] > 0:
            md_lines.append(f"| {ct} | {st['n']} | {st['block']} | {st['attack']} | {st['ambig']} |")
    md_lines.append("")
    md_lines.append("### Per-case verdicts")
    md_lines.append("")
    md_lines.append("| # | video / rally | frame | variant | conf-tier | verdict | note |")
    md_lines.append("|---|---------------|-------|---------|-----------|---------|------|")
    icon = {"block": "🟦", "attack": "🟧", "ambig": "⚠️"}
    for c in cases:
        idx = c["idx"]
        v = c.get("claude_verdict", "")
        ic = icon.get(v, "·")
        note = (c.get("claude_verdict_note") or "").replace("|", "\\|")
        md_lines.append(
            f"| {idx} | {c['video']}/{c['rally_short']} | {c['pl_frame']} | "
            f"{c.get('variant','?')} | {c.get('confidence_tier','?')} | "
            f"{ic} {v} | {note} |"
        )

    MD_PATH.write_text("\n".join(md_lines))
    print(f"Updated markdown: {MD_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
