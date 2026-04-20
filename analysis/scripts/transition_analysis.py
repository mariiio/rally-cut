"""Phase CRF-0: transition matrix discovery + feasibility probe.

Before writing any decoder, answer: is ``P(action_j | action_i, gap, cross_team)``
informative? Specifically:

1. Do known-valid transitions (set→attack same-team, serve→receive cross-team,
   attack→dig cross-team) dominate their respective slots?
2. Are known-invalid transitions (attack→attack same-team within 5f, etc.)
   near zero?
3. For the 219 rescuable FN contacts, what transition would rescue them — and
   do those transitions have supportive prior in training data?

If transitions are flat → STOP the CRF line, go to candidate-generator work.
If transitions are strongly structured → PROCEED with Phase CRF-1.

Data sources:
- GT rally sequences (load_rallies_with_action_gt — 2095 contacts / 364 rallies)
- Oracle events cache (orthogonality_events_2026_04_19.json) for the
  rescuable-FN list.

Writes ``reports/transition_analysis_2026_04_20.md`` + transition_matrix.json.
"""
from __future__ import annotations

import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.eval_action_detection import load_rallies_with_action_gt

logging.getLogger("rallycut.tracking.action_classifier").setLevel(logging.ERROR)
console = Console()

ACTIONS = ["serve", "receive", "set", "attack", "dig", "block"]

# Gap buckets in frames (at rally.fps). Chosen per textbook volleyball timings:
# - ≤5: rapid same-rally events (attack→block, block→dig)
# - 6-15: tight transitions (block→dig, in-rally recoveries)
# - 16-40: standard same-side sequence (receive→set, set→attack)
# - 41-120: slow transitions (serve→receive arrival time, long rallies)
# - >120: rally-gap (shouldn't appear within one rally)
GAP_BUCKETS = [(0, 5), (6, 15), (16, 40), (41, 120), (121, 10_000)]
GAP_BUCKET_LABELS = [f"{lo}-{hi if hi < 1000 else '+'}" for lo, hi in GAP_BUCKETS]


def _bucket(gap: int) -> int:
    for idx, (lo, hi) in enumerate(GAP_BUCKETS):
        if lo <= gap <= hi:
            return idx
    return len(GAP_BUCKETS) - 1


def _team(player_id: int) -> int:
    """Convention: player_id 1-2 = team 0 (near), 3-4 = team 1 (far)."""
    if player_id < 1 or player_id > 4:
        return -1
    return (player_id - 1) // 2


def main() -> None:
    rallies = load_rallies_with_action_gt()
    console.print(f"[bold]Loaded {len(rallies)} rallies, "
                  f"{sum(len(r.gt_labels) for r in rallies)} contacts[/bold]\n")

    # 1. Build all observed (action_i, action_j, gap_bucket, cross_team) tuples
    #    from consecutive GT contacts within each rally.
    transitions: Counter[tuple[str, str, int, str]] = Counter()
    context: Counter[tuple[str, int, str]] = Counter()  # (action_i, gap_bucket, cross_team)
    per_action: Counter[str] = Counter()

    # For analyzing where the FN rescues fall
    all_gap_raw: list[tuple[str, str, int, str]] = []

    for rally in rallies:
        sorted_gt = sorted(rally.gt_labels, key=lambda x: x.frame)
        for i in range(len(sorted_gt) - 1):
            a = sorted_gt[i]
            b = sorted_gt[i + 1]
            if a.action not in ACTIONS or b.action not in ACTIONS:
                continue
            gap = b.frame - a.frame
            bucket_idx = _bucket(gap)
            team_a = _team(a.player_track_id)
            team_b = _team(b.player_track_id)
            cross = "unknown"
            if team_a in (0, 1) and team_b in (0, 1):
                cross = "cross" if team_a != team_b else "same"
            transitions[(a.action, b.action, bucket_idx, cross)] += 1
            context[(a.action, bucket_idx, cross)] += 1
            all_gap_raw.append((a.action, b.action, gap, cross))
        for gt in sorted_gt:
            if gt.action in ACTIONS:
                per_action[gt.action] += 1

    console.print(f"Observed action counts: {dict(per_action)}\n")

    # 2. Compute conditional P(action_j | action_i, gap_bucket, cross_team).
    #    With Laplace smoothing (alpha=0.5) to handle unseen transitions.
    alpha = 0.5
    vocab_size = len(ACTIONS)
    prob: dict[tuple[str, int, str], dict[str, float]] = {}
    for key, count_a in context.items():
        a_i, bucket_idx, cross = key
        denom = count_a + alpha * vocab_size
        row = {}
        for a_j in ACTIONS:
            row[a_j] = (transitions[(a_i, a_j, bucket_idx, cross)] + alpha) / denom
        prob[key] = row

    # 3. Inspect: do known-valid transitions dominate their bucket?
    canonical_checks = [
        # (name, action_i, action_j, bucket_hint, cross_hint, expect)
        ("serve→receive cross 41-120", "serve", "receive", 3, "cross", "HIGH"),
        ("receive→set same 6-15", "receive", "set", 1, "same", "HIGH"),
        ("set→attack same 16-40", "set", "attack", 2, "same", "HIGH"),
        ("attack→dig cross 0-5", "attack", "dig", 0, "cross", "HIGH"),
        ("attack→block cross 0-5", "attack", "block", 0, "cross", "HIGH"),
        ("block→dig cross 0-5", "block", "dig", 0, "cross", "HIGH"),
        ("attack→attack same 0-5", "attack", "attack", 0, "same", "LOW"),
        ("serve→attack any", "serve", "attack", 0, "same", "LOW"),
        ("receive→receive same 0-5", "receive", "receive", 0, "same", "LOW"),
        ("dig→dig same 0-5", "dig", "dig", 0, "same", "LOW"),
    ]
    check_rows: list[dict] = []
    for name, a, b, bkt, cross, expect in canonical_checks:
        key = (a, bkt, cross)
        if key in prob:
            p = prob[key].get(b, 0.0)
            n = transitions[(a, b, bkt, cross)]
            denom = context[key]
            row = {
                "name": name, "a": a, "b": b, "bucket": GAP_BUCKET_LABELS[bkt],
                "cross": cross, "expect": expect,
                "prob": p, "count": n, "denom": denom,
            }
        else:
            row = {
                "name": name, "a": a, "b": b, "bucket": GAP_BUCKET_LABELS[bkt],
                "cross": cross, "expect": expect,
                "prob": 0.0, "count": 0, "denom": 0,
            }
        check_rows.append(row)

    tbl = Table(title="Canonical transition checks", show_header=True, header_style="bold")
    for col in ("Transition", "Bucket", "Cross", "Expect", "P(b|context)", "Count", "Denom"):
        tbl.add_column(col, justify="right" if col in ("P(b|context)", "Count", "Denom") else "left")
    for r in check_rows:
        style = ""
        if r["expect"] == "HIGH" and r["prob"] >= 0.40:
            style = "green"
        elif r["expect"] == "HIGH" and r["prob"] < 0.20:
            style = "red"
        elif r["expect"] == "LOW" and r["prob"] <= 0.05:
            style = "green"
        elif r["expect"] == "LOW" and r["prob"] > 0.10:
            style = "red"
        tbl.add_row(
            r["name"], r["bucket"], r["cross"], r["expect"],
            f"{r['prob']:.3f}", str(r["count"]), str(r["denom"]),
            style=style,
        )
    console.print(tbl)

    # 4. Full (action_i, action_j) marginal (ignoring gap+cross)
    marg: dict[tuple[str, str], int] = Counter()
    marg_denom: Counter[str] = Counter()
    for key, n in transitions.items():
        a, b, _bkt, _cross = key
        marg[(a, b)] += n
    for key, n in context.items():
        a, _bkt, _cross = key
        marg_denom[a] += n

    tbl2 = Table(title="Marginal P(b | a) across all gaps/cross", show_header=True, header_style="bold")
    tbl2.add_column("action_i", style="bold")
    for b in ACTIONS:
        tbl2.add_column(b, justify="right")
    for a in ACTIONS:
        denom = marg_denom.get(a, 0)
        row = [a]
        for b in ACTIONS:
            n = marg[(a, b)]
            p = n / denom if denom else 0.0
            row.append(f"{p:.2f}")
        tbl2.add_row(*row)
    console.print(tbl2)

    # 5. Informative-ness: KL divergence of P(b|a,gap,cross) from uniform.
    #    High KL → transitions carry real signal.
    kls: list[float] = []
    for key, row in prob.items():
        q = np.array([row[b] for b in ACTIONS])
        u = np.ones(vocab_size) / vocab_size
        kl = float(np.sum(q * np.log(q / u + 1e-12)))
        kls.append(kl)
    kl_mean = float(np.mean(kls)) if kls else 0.0
    kl_median = float(np.median(kls)) if kls else 0.0
    console.print(
        f"\n[bold]Transition informativeness (KL vs uniform):[/bold] "
        f"mean={kl_mean:.3f}, median={kl_median:.3f}"
    )
    # Reference: uniform has KL=0. log(7) ≈ 1.95 is max.

    # 6. Rescue-candidate analysis: for each GBM-MISS contact, what transition
    #    would rescue it, and does that transition have prior support?
    events_path = Path("reports/orthogonality_events_2026_04_19.json")
    rescue_analysis = None
    if events_path.exists():
        events = json.loads(events_path.read_text())
        by_rally: dict[str, list] = defaultdict(list)
        for e in events:
            by_rally[e["rally_id"]].append(e)

        # Map rally → sorted GTs
        rallies_by_id = {r.rally_id: r for r in rallies}

        miss_rescue_stats = {"total_miss": 0, "analyzed": 0, "supported": 0,
                             "ood": 0, "supported_hi": 0}
        rescue_examples: list[dict] = []

        for rid, rlist in by_rally.items():
            rally = rallies_by_id.get(rid)
            if rally is None:
                continue
            sorted_gts = sorted(rally.gt_labels, key=lambda x: x.frame)
            for idx, gt in enumerate(sorted_gts):
                # Find matching event
                ev = next((e for e in rlist if e["frame"] == gt.frame
                           and e["action"] == gt.action), None)
                if ev is None or ev["gbm_hit"]:
                    continue
                miss_rescue_stats["total_miss"] += 1
                if idx == 0:
                    continue  # can't evaluate transition for first contact
                prev_gt = sorted_gts[idx - 1]
                if prev_gt.action not in ACTIONS or gt.action not in ACTIONS:
                    continue
                gap = gt.frame - prev_gt.frame
                bucket_idx = _bucket(gap)
                team_a = _team(prev_gt.player_track_id)
                team_b = _team(gt.player_track_id)
                cross = (
                    "cross" if (team_a != team_b and team_a in (0, 1) and team_b in (0, 1))
                    else "same" if team_a == team_b
                    else "unknown"
                )
                key = (prev_gt.action, bucket_idx, cross)
                p = prob.get(key, {}).get(gt.action, 0.0)
                support = transitions[(prev_gt.action, gt.action, bucket_idx, cross)]
                miss_rescue_stats["analyzed"] += 1
                if support >= 10:
                    miss_rescue_stats["supported"] += 1
                if p >= 0.30:
                    miss_rescue_stats["supported_hi"] += 1
                if support == 0:
                    miss_rescue_stats["ood"] += 1
                if len(rescue_examples) < 15:
                    rescue_examples.append({
                        "rally": rid[:8], "frame": gt.frame, "action": gt.action,
                        "prev_action": prev_gt.action, "gap": gap,
                        "bucket": GAP_BUCKET_LABELS[bucket_idx], "cross": cross,
                        "p_rescue": p, "support": support,
                    })

        rescue_analysis = miss_rescue_stats
        console.print("\n[bold]Rescue-candidate transition analysis[/bold]")
        console.print(f"  Total GBM-MISSes in cache: {miss_rescue_stats['total_miss']}")
        console.print(f"  Analyzable (had prior GT): {miss_rescue_stats['analyzed']}")
        console.print(f"  Transition support ≥ 10 events: "
                      f"{miss_rescue_stats['supported']} ({miss_rescue_stats['supported']/max(1,miss_rescue_stats['analyzed']):.1%})")
        console.print(f"  P(rescue-transition) ≥ 0.30: "
                      f"{miss_rescue_stats['supported_hi']} ({miss_rescue_stats['supported_hi']/max(1,miss_rescue_stats['analyzed']):.1%})")
        console.print(f"  Out-of-distribution (0 support): "
                      f"{miss_rescue_stats['ood']} ({miss_rescue_stats['ood']/max(1,miss_rescue_stats['analyzed']):.1%})")

        if rescue_examples:
            etbl = Table(title="Sample rescue-candidate transitions", show_header=True)
            for col in ("rally", "frame", "gt_action", "prev", "gap", "bucket",
                        "cross", "p(rescue)", "support"):
                etbl.add_column(col, justify="right" if col in ("p(rescue)", "support",
                                "gap", "frame") else "left")
            for ex in rescue_examples:
                etbl.add_row(
                    ex["rally"], str(ex["frame"]), ex["action"], ex["prev_action"],
                    str(ex["gap"]), ex["bucket"], ex["cross"],
                    f"{ex['p_rescue']:.3f}", str(ex["support"]),
                )
            console.print(etbl)

    # 7. Verdict
    verdict_hi_green = sum(1 for r in check_rows
                           if r["expect"] == "HIGH" and r["prob"] >= 0.40)
    verdict_hi_total = sum(1 for r in check_rows if r["expect"] == "HIGH")
    verdict_lo_green = sum(1 for r in check_rows
                           if r["expect"] == "LOW" and r["prob"] <= 0.05)
    verdict_lo_total = sum(1 for r in check_rows if r["expect"] == "LOW")

    verdict = (
        "STRONG STRUCTURE — PROCEED"
        if (verdict_hi_green >= verdict_hi_total - 1
            and verdict_lo_green >= verdict_lo_total - 1)
        else "PARTIAL STRUCTURE — PROCEED WITH CAUTION"
        if verdict_hi_green >= verdict_hi_total // 2 else
        "FLAT/NOISY TRANSITIONS — STOP"
    )
    color = "green" if "STRONG" in verdict else "yellow" if "PARTIAL" in verdict else "red"
    console.print(f"\n[bold]Verdict: [{color}]{verdict}[/{color}][/bold]")
    console.print(
        f"  HIGH-expect transitions passing: {verdict_hi_green}/{verdict_hi_total}"
    )
    console.print(
        f"  LOW-expect transitions passing: {verdict_lo_green}/{verdict_lo_total}"
    )

    # 8. Write report
    out = Path("reports/transition_analysis_2026_04_20.md")
    lines: list[str] = []
    lines.append("# CRF-0 Transition Matrix Analysis")
    lines.append("")
    lines.append(f"- Rallies analysed: {len(rallies)}")
    lines.append(f"- Consecutive-contact transitions observed: {sum(transitions.values())}")
    lines.append(f"- Laplace smoothing α={alpha}")
    lines.append(f"- Gap buckets: {GAP_BUCKET_LABELS}")
    lines.append("")
    lines.append("## Canonical transition checks")
    lines.append("")
    lines.append("| Transition | Bucket | Cross | Expect | P(b|a,bkt,cross) | Count | Denom |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in check_rows:
        lines.append(
            f"| {r['name']} | {r['bucket']} | {r['cross']} | {r['expect']} "
            f"| {r['prob']:.3f} | {r['count']} | {r['denom']} |"
        )
    lines.append("")
    lines.append("## Marginal transition matrix P(b | a) (all gaps/cross)")
    lines.append("")
    header = "| a \\ b | " + " | ".join(ACTIONS) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(ACTIONS) + 1))
    for a in ACTIONS:
        denom = marg_denom.get(a, 0)
        row_cells = [a]
        for b in ACTIONS:
            n = marg[(a, b)]
            p = n / denom if denom else 0.0
            row_cells.append(f"{p:.2f}")
        lines.append("| " + " | ".join(row_cells) + " |")
    lines.append("")
    lines.append("## Transition informativeness")
    lines.append("")
    lines.append(f"- Mean KL(P || Uniform): {kl_mean:.3f}")
    lines.append(f"- Median KL: {kl_median:.3f}")
    lines.append("- Max possible: log(7) ≈ 1.945  (higher = more structure)")
    lines.append("")

    if rescue_analysis:
        lines.append("## Rescue-candidate transition analysis")
        lines.append("")
        lines.append(f"- Total GBM-MISSes analysed: {rescue_analysis['analyzed']}")
        lines.append(f"- With transition support ≥ 10 events: "
                     f"{rescue_analysis['supported']} ({rescue_analysis['supported']/max(1,rescue_analysis['analyzed']):.1%})")
        lines.append(f"- With P(rescue transition) ≥ 0.30: "
                     f"{rescue_analysis['supported_hi']} ({rescue_analysis['supported_hi']/max(1,rescue_analysis['analyzed']):.1%})")
        lines.append(f"- Out-of-distribution (0 observations): "
                     f"{rescue_analysis['ood']} ({rescue_analysis['ood']/max(1,rescue_analysis['analyzed']):.1%})")
        lines.append("")

    lines.append(f"## Verdict: {verdict}")
    lines.append("")
    lines.append(f"- HIGH-expect transitions passing (P≥0.40): {verdict_hi_green}/{verdict_hi_total}")
    lines.append(f"- LOW-expect transitions passing (P≤0.05): {verdict_lo_green}/{verdict_lo_total}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    console.print(f"\n[green]Report: {out}[/green]")

    # Also dump the full transition matrix
    matrix_path = Path("reports/transition_matrix_2026_04_20.json")
    matrix_data = {
        "gap_buckets": GAP_BUCKET_LABELS,
        "actions": ACTIONS,
        "laplace_alpha": alpha,
        "counts": {
            f"{a}|{b}|{bkt}|{cross}": n
            for (a, b, bkt, cross), n in transitions.items()
        },
        "probs": {
            f"{a}|{bkt}|{cross}": row
            for (a, bkt, cross), row in prob.items()
        },
        "marginals": {
            f"{a}|{b}": marg[(a, b)] / max(1, marg_denom[a])
            for a in ACTIONS for b in ACTIONS
        },
        "kl_mean": kl_mean,
        "kl_median": kl_median,
        "verdict": verdict,
    }
    matrix_path.write_text(json.dumps(matrix_data, indent=2))
    console.print(f"[green]Matrix: {matrix_path}[/green]")


if __name__ == "__main__":
    main()
