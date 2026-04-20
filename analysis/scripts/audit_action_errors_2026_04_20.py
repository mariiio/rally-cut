"""Audit remaining contact FNs + action misclassifications — fresh taxonomy.

Consumes `outputs/action_errors/corpus.jsonl` (freshly rebuilt with seq
metadata per FN/wrong-action/wrong-player record) and produces:

1. Per-subcategory FN breakdown with seq-signal distribution. The headline
   slice: within `rejected_by_classifier`, how many FNs have GBM conf below
   the production SEQ_RECOVERY_CLF_FLOOR=0.20 *despite* strong
   (≥0.80) MS-TCN++ agreement? Those are the "would-be-rescuable" cases
   that the current floor blocks.

2. `no_player_nearby` bucket re-check. With seq metadata we can separate
   "candidate fired + seq endorses but no player in radius" (likely
   attribution/tracking gap) from "no seq either" (likely GT mislabel or
   real physical no-contact).

3. `wrong_action` confusion matrix + decoder-rescuable signature: when
   `seq_peak_action == gt_action` the Viterbi decoder would most likely
   correct the label. This is the per-error answer to "would flipping the
   decoder flag help?" without re-running the whole LOO decoder eval.

4. `wrong_player` (647, largest error class, never taxonomized) — size
   distribution by gt_action, seq signal, decoder-compatibility.

Read-only. No retraining. Runtime: < 10s on 1k records.

Usage (cd analysis):
    uv run python scripts/audit_action_errors_2026_04_20.py
    uv run python scripts/audit_action_errors_2026_04_20.py \
        --corpus outputs/action_errors/corpus.jsonl \
        --out reports/action_detection_audit_2026_04_20.md
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()

SEQ_RECOVERY_TAU = 0.80    # production rescue gate in contact_detector.py
SEQ_RECOVERY_CLF_FLOOR = 0.20


def _prob_bucket(x: float) -> str:
    """Coarse bucketing that matches the rescue-gate intuition."""
    if x < 0.05:
        return "<0.05"
    if x < 0.10:
        return "[0.05, 0.10)"
    if x < 0.20:
        return "[0.10, 0.20)"  # below SEQ_RECOVERY_CLF_FLOOR
    if x < 0.30:
        return "[0.20, 0.30)"
    if x < 0.40:
        return "[0.30, 0.40)"  # below ContactClassifier.threshold
    return ">=0.40"


def _seq_bucket(x: float) -> str:
    if x < 0.50:
        return "<0.50"
    if x < 0.80:
        return "[0.50, 0.80)"  # below SEQ_RECOVERY_TAU
    if x < 0.95:
        return "[0.80, 0.95)"
    return ">=0.95"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="outputs/action_errors/corpus.jsonl")
    ap.add_argument("--out", default="reports/action_detection_audit_2026_04_20.md")
    args = ap.parse_args()

    rows = [json.loads(line) for line in Path(args.corpus).read_text().splitlines()]
    if not rows:
        console.print("[red]Empty corpus[/red]")
        return

    by_class = defaultdict(list)
    for r in rows:
        by_class[r["error_class"]].append(r)

    # ================================================================ Summary
    console.print(f"\n[bold]Corpus: {len(rows)} errors[/bold]")
    for k, v in sorted(by_class.items(), key=lambda kv: -len(kv[1])):
        console.print(f"  {k:20s} {len(v):4d}")

    fns = by_class.get("FN_contact", [])
    wa = by_class.get("wrong_action", [])
    wp = by_class.get("wrong_player", [])

    # =========================================================== FN subcat
    console.print("\n[bold]FN subcategory breakdown[/bold]")
    subcat = Counter(r.get("fn_subcategory") for r in fns)
    t = Table()
    for c in ("Subcategory", "Count", "%"):
        t.add_column(c, justify="right" if c != "Subcategory" else "left")
    for sc, n in subcat.most_common():
        t.add_row(str(sc), str(n), f"{n / len(fns) * 100:.1f}%")
    console.print(t)

    # ================================= rejected_by_classifier × seq signal
    rejected = [r for r in fns if r.get("fn_subcategory") == "rejected_by_classifier"]
    if rejected:
        console.print(
            f"\n[bold]rejected_by_classifier ({len(rejected)}): GBM conf × seq peak[/bold]"
        )
        crosstab: dict[tuple[str, str], int] = defaultdict(int)
        for r in rejected:
            gb = _prob_bucket(r.get("classifier_conf", 0.0))
            sb = _seq_bucket(r.get("seq_peak_nonbg_within_5f", 0.0))
            crosstab[(gb, sb)] += 1

        prob_order = ["<0.05", "[0.05, 0.10)", "[0.10, 0.20)",
                      "[0.20, 0.30)", "[0.30, 0.40)", ">=0.40"]
        seq_order = ["<0.50", "[0.50, 0.80)", "[0.80, 0.95)", ">=0.95"]

        t = Table(title="rejected_by_classifier  (rows=GBM conf, cols=seq peak ±5f)")
        t.add_column("GBM conf \\ seq", style="bold")
        for sb in seq_order:
            t.add_column(sb, justify="right")
        t.add_column("total", justify="right")
        col_totals = defaultdict(int)
        for gb in prob_order:
            row = [gb]
            total = 0
            for sb in seq_order:
                n = crosstab.get((gb, sb), 0)
                total += n
                col_totals[sb] += n
                row.append(str(n) if n else "-")
            row.append(f"[bold]{total}[/bold]")
            t.add_row(*row)
        t.add_row(
            "[bold]total[/bold]",
            *[f"[bold]{col_totals[sb]}[/bold]" for sb in seq_order],
            f"[bold]{len(rejected)}[/bold]",
        )
        console.print(t)

        # Sweet spots
        would_rescue_if_floor_0_10 = sum(
            1 for r in rejected
            if r.get("classifier_conf", 0.0) >= 0.10
            and r.get("seq_peak_nonbg_within_5f", 0.0) >= SEQ_RECOVERY_TAU
        )
        would_rescue_if_floor_0_05 = sum(
            1 for r in rejected
            if r.get("classifier_conf", 0.0) >= 0.05
            and r.get("seq_peak_nonbg_within_5f", 0.0) >= SEQ_RECOVERY_TAU
        )
        no_seq_signal = sum(
            1 for r in rejected if r.get("seq_peak_nonbg_within_5f", 0.0) < 0.50
        )
        console.print(
            f"\n  Lower CLF_FLOOR 0.20→0.10 would allow rescue of "
            f"{would_rescue_if_floor_0_10}/{len(rejected)} "
            f"(seq ≥ {SEQ_RECOVERY_TAU} AND conf ≥ 0.10)"
        )
        console.print(
            f"  Lower CLF_FLOOR 0.20→0.05 would allow rescue of "
            f"{would_rescue_if_floor_0_05}/{len(rejected)}"
        )
        console.print(
            f"  No seq signal (<0.50): {no_seq_signal}/{len(rejected)} "
            f"— genuinely hard, not rescuable by any seq-based gate"
        )

        # Per-action
        per_action = defaultdict(lambda: {"total": 0, "rescuable_0_10": 0,
                                          "rescuable_0_05": 0, "no_seq": 0})
        for r in rejected:
            a = r.get("gt_action", "?")
            per_action[a]["total"] += 1
            if (r.get("classifier_conf", 0.0) >= 0.10
                    and r.get("seq_peak_nonbg_within_5f", 0.0) >= SEQ_RECOVERY_TAU):
                per_action[a]["rescuable_0_10"] += 1
            if (r.get("classifier_conf", 0.0) >= 0.05
                    and r.get("seq_peak_nonbg_within_5f", 0.0) >= SEQ_RECOVERY_TAU):
                per_action[a]["rescuable_0_05"] += 1
            if r.get("seq_peak_nonbg_within_5f", 0.0) < 0.50:
                per_action[a]["no_seq"] += 1

        t = Table(title="rejected_by_classifier per action")
        for c in ("action", "total", "rescuable @ floor=0.10",
                  "rescuable @ floor=0.05", "no_seq"):
            t.add_column(c, justify="right" if c != "action" else "left")
        for a, d in sorted(per_action.items(), key=lambda kv: -kv[1]["total"]):
            t.add_row(a, str(d["total"]),
                      str(d["rescuable_0_10"]), str(d["rescuable_0_05"]),
                      str(d["no_seq"]))
        console.print(t)

    # =============================================== no_player_nearby × seq
    no_plr = [r for r in fns if r.get("fn_subcategory") == "no_player_nearby"]
    if no_plr:
        console.print(
            f"\n[bold]no_player_nearby ({len(no_plr)}): seq-endorsement sub-split[/bold]"
        )
        # When seq endorses → tracking/distance gate; when seq doesn't → possibly
        # legitimately no-contact or GT mislabel.
        seq_endorsed = [r for r in no_plr
                        if r.get("seq_peak_nonbg_within_5f", 0.0) >= SEQ_RECOVERY_TAU]
        seq_medium = [r for r in no_plr
                      if 0.50 <= r.get("seq_peak_nonbg_within_5f", 0.0) < SEQ_RECOVERY_TAU]
        seq_none = [r for r in no_plr
                    if r.get("seq_peak_nonbg_within_5f", 0.0) < 0.50]

        console.print(
            f"  seq ≥ 0.80 (endorsed, likely distance-gate/tracking): "
            f"{len(seq_endorsed)}"
        )
        console.print(f"  seq ∈ [0.50, 0.80): {len(seq_medium)}")
        console.print(
            f"  seq < 0.50 (no endorsement, likely GT issue or no contact): "
            f"{len(seq_none)}"
        )

        # Per-action
        pa = Counter(r["gt_action"] for r in seq_endorsed)
        if pa:
            console.print(f"  seq-endorsed per action: {dict(pa)}")

    # ========================================= wrong_action confusion matrix
    if wa:
        console.print(f"\n[bold]wrong_action ({len(wa)}): confusion + decoder signature[/bold]")
        conf = Counter((r["gt_action"], r["pred_action"]) for r in wa)
        # Count how many have seq_peak_action == gt_action (decoder-rescuable signature)
        rescuable = sum(
            1 for r in wa
            if r.get("seq_peak_action") == r.get("gt_action")
            and r.get("seq_peak_action_prob", 0.0) >= SEQ_RECOVERY_TAU
        )
        console.print(
            f"  Decoder-rescuable signature (seq argmax matches GT, prob ≥ 0.80): "
            f"{rescuable}/{len(wa)}"
        )

        t = Table(title="wrong_action pairs (top 20)")
        for c in ("gt → pred", "count", "seq=gt ≥0.80", "seq=pred"):
            t.add_column(c, justify="right" if c != "gt → pred" else "left")
        # Per-pair rescuability
        pair_rescue = defaultdict(int)
        pair_seq_pred = defaultdict(int)
        for r in wa:
            key = (r["gt_action"], r["pred_action"])
            if (r.get("seq_peak_action") == r.get("gt_action")
                    and r.get("seq_peak_action_prob", 0.0) >= SEQ_RECOVERY_TAU):
                pair_rescue[key] += 1
            if r.get("seq_peak_action") == r.get("pred_action"):
                pair_seq_pred[key] += 1
        for (g, p), n in conf.most_common(20):
            t.add_row(f"{g} → {p}", str(n),
                      str(pair_rescue.get((g, p), 0)),
                      str(pair_seq_pred.get((g, p), 0)))
        console.print(t)

    # ============================================= wrong_player taxonomy
    if wp:
        console.print(f"\n[bold]wrong_player ({len(wp)}): per-action + seq check[/bold]")
        t = Table()
        for c in ("gt_action", "total", "seq_matches_gt ≥0.80",
                  "seq_matches_pred_action"):
            t.add_column(c, justify="right" if c != "gt_action" else "left")

        per_action = defaultdict(
            lambda: {"total": 0, "seq_gt": 0, "seq_pred": 0})
        for r in wp:
            a = r["gt_action"]
            per_action[a]["total"] += 1
            if (r.get("seq_peak_action") == a
                    and r.get("seq_peak_action_prob", 0.0) >= SEQ_RECOVERY_TAU):
                per_action[a]["seq_gt"] += 1
            if r.get("seq_peak_action") == r.get("pred_action"):
                per_action[a]["seq_pred"] += 1
        for a, d in sorted(per_action.items(), key=lambda kv: -kv[1]["total"]):
            t.add_row(a, str(d["total"]), str(d["seq_gt"]), str(d["seq_pred"]))
        console.print(t)

    # ============================================================ Report
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Action Detection Error Audit — 2026-04-20",
        "",
        f"Source: `{args.corpus}`",
        "",
        "## Corpus size",
        "",
        f"- Total errors: **{len(rows)}**",
    ]
    for k, v in sorted(by_class.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"- `{k}`: {len(v)}")

    lines.append("")
    lines.append("## FN subcategory breakdown")
    lines.append("")
    lines.append("| Subcategory | Count | Share |")
    lines.append("|---|---:|---:|")
    for sc, n in subcat.most_common():
        lines.append(f"| {sc} | {n} | {n / max(1, len(fns)) * 100:.1f}% |")

    if rejected:
        lines.append("")
        lines.append(f"## rejected_by_classifier ({len(rejected)}) × seq signal")
        lines.append("")
        lines.append("| GBM conf \\ seq peak | " + " | ".join(seq_order) + " | total |")
        lines.append("|---|" + "|".join("---:" for _ in seq_order) + "|---:|")
        for gb in prob_order:
            cells = [gb]
            total = 0
            for sb in seq_order:
                n = crosstab.get((gb, sb), 0)
                total += n
                cells.append(str(n) if n else "–")
            cells.append(f"**{total}**")
            lines.append("| " + " | ".join(cells) + " |")
        col_totals_row = ["**total**"] + [
            f"**{col_totals[sb]}**" for sb in seq_order
        ] + [f"**{len(rejected)}**"]
        lines.append("| " + " | ".join(col_totals_row) + " |")

        lines.append("")
        lines.append(
            f"- **Lower `SEQ_RECOVERY_CLF_FLOOR` 0.20→0.10** would admit "
            f"{would_rescue_if_floor_0_10}/{len(rejected)} FNs (have seq ≥ 0.80 AND conf ≥ 0.10)."
        )
        lines.append(
            f"- **Lower to 0.05** would admit {would_rescue_if_floor_0_05}/{len(rejected)}."
        )
        lines.append(
            f"- **No seq signal** (<0.50) on {no_seq_signal}/{len(rejected)} FNs — "
            f"unrescuable by any seq-based gate."
        )

    if no_plr:
        lines.append("")
        lines.append(f"## no_player_nearby ({len(no_plr)})")
        lines.append("")
        lines.append(f"- seq ≥ 0.80: {len(seq_endorsed)} — likely distance/tracking gap")
        lines.append(f"- seq ∈ [0.50, 0.80): {len(seq_medium)}")
        lines.append(f"- seq < 0.50: {len(seq_none)} — likely GT/physical")

    if wa:
        lines.append("")
        lines.append(f"## wrong_action ({len(wa)}) confusion + decoder signature")
        lines.append("")
        lines.append(
            f"- Decoder-rescuable signature (seq argmax == GT, prob ≥ 0.80): "
            f"**{rescuable}/{len(wa)}**"
        )
        lines.append("")
        lines.append("| gt → pred | count | seq=gt ≥0.80 (decoder-fix) | seq=pred |")
        lines.append("|---|---:|---:|---:|")
        for (g, p), n in conf.most_common(20):
            lines.append(
                f"| {g} → {p} | {n} | {pair_rescue.get((g, p), 0)} "
                f"| {pair_seq_pred.get((g, p), 0)} |"
            )

    if wp:
        lines.append("")
        lines.append(f"## wrong_player ({len(wp)}) per-action")
        lines.append("")
        lines.append(
            "| gt_action | total | seq_matches_gt ≥0.80 | seq_matches_pred |"
        )
        lines.append("|---|---:|---:|---:|")
        for a, d in sorted(per_action.items(), key=lambda kv: -kv[1]["total"]):
            lines.append(
                f"| {a} | {d['total']} | {d['seq_gt']} | {d['seq_pred']} |"
            )

    out.write_text("\n".join(lines))
    console.print(f"\n[green]Report: {out}[/green]")


if __name__ == "__main__":
    main()
