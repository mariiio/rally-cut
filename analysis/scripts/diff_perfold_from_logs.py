"""Diff per-fold F1 from two corpus-builder logs.

Parses lines of the form:
    [I/N] <video_id>[:8] (Nr): gt=G tp=T fn=F wrong_action=W fp=P (Ss)

Computes per-fold contact F1 = 2*TP / (2*TP + FN + FP) and reports the
delta between logs. Flags folds where new_F1 - old_F1 < --fail-delta-pp
(default -0.8pp, the Phase C retrain pre-registered gate).

Usage:
    cd analysis
    uv run python scripts/diff_perfold_from_logs.py \\
        --old outputs/action_errors/corpus_v2_rerun.log \\
        --new outputs/action_errors/corpus_v5_rebuild.log
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()

LINE_RE = re.compile(
    r"\[(\d+)/(\d+)\]\s+(?P<vid>[0-9a-f]{8})\s+\((?P<n>\d+)r\):\s+"
    r"gt=(?P<gt>\d+)\s+tp=(?P<tp>\d+)\s+fn=(?P<fn>\d+)\s+"
    r"wrong_action=(?P<wa>\d+)\s+fp=(?P<fp>\d+)"
)


def parse_log(path: Path) -> dict[str, dict[str, int]]:
    """Return {video_prefix_8: {gt, tp, fn, wa, fp}} from a corpus-build log."""
    out: dict[str, dict[str, int]] = {}
    for line in path.read_text().splitlines():
        m = LINE_RE.search(line)
        if not m:
            continue
        out[m.group("vid")] = {
            "gt": int(m.group("gt")),
            "tp": int(m.group("tp")),
            "fn": int(m.group("fn")),
            "wa": int(m.group("wa")),
            "fp": int(m.group("fp")),
        }
    return out


def f1(tp: int, fn: int, fp: int) -> float:
    den = 2 * tp + fn + fp
    return 2 * tp / den if den > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", type=Path, required=True)
    ap.add_argument("--new", type=Path, required=True)
    ap.add_argument("--fail-delta-pp", type=float, default=-0.8)
    args = ap.parse_args()

    old = parse_log(args.old)
    new = parse_log(args.new)
    console.print(
        f"[bold]Parsed[/bold] old={len(old)} folds from {args.old.name}  "
        f"new={len(new)} folds from {args.new.name}"
    )

    vids = sorted(set(old) | set(new))
    rows: list[tuple] = []
    for v in vids:
        o = old.get(v)
        n = new.get(v)
        if o is None or n is None:
            continue
        f_o = f1(o["tp"], o["fn"], o["fp"])
        f_n = f1(n["tp"], n["fn"], n["fp"])
        rows.append(
            (
                v, o["gt"], f_o, f_n, (f_n - f_o) * 100,
                o["tp"], o["fn"], o["fp"], o["wa"],
                n["tp"], n["fn"], n["fp"], n["wa"],
            )
        )

    rows.sort(key=lambda r: r[4])  # regressions first

    table = Table(title=f"Per-fold F1 diff: {args.new.name} vs {args.old.name}")
    table.add_column("video")
    table.add_column("GT", justify="right")
    table.add_column("F1 old", justify="right")
    table.add_column("F1 new", justify="right")
    table.add_column("Δpp", justify="right")
    table.add_column("tp old→new", justify="right")
    table.add_column("fn old→new", justify="right")
    table.add_column("fp old→new", justify="right")
    table.add_column("wa old→new", justify="right")

    regressions: list[str] = []
    for v, gt, fo, fn_, d, tpo, fno, fpo, wao, tpn, fnn, fpn, wan in rows:
        style = (
            "[red]" if d < args.fail_delta_pp
            else "[yellow]" if d < 0
            else "[green]"
        )
        table.add_row(
            v,
            str(gt),
            f"{fo * 100:.2f}",
            f"{fn_ * 100:.2f}",
            f"{style}{d:+.2f}[/]",
            f"{tpo}→{tpn}",
            f"{fno}→{fnn}",
            f"{fpo}→{fpn}",
            f"{wao}→{wan}",
        )
        if d < args.fail_delta_pp:
            regressions.append(
                f"  {v}: gt={gt} F1 {fo*100:.2f}→{fn_*100:.2f} ({d:+.2f}pp)  "
                f"tp {tpo}→{tpn}  fn {fno}→{fnn}  fp {fpo}→{fpn}  wa {wao}→{wan}"
            )
    console.print(table)

    agg = Table(title="Aggregate (sum over folds)")
    agg.add_column("")
    agg.add_column("old")
    agg.add_column("new")
    agg.add_column("Δ")
    for k in ("gt", "tp", "fn", "wa", "fp"):
        o_sum = sum(old[v][k] for v in old)
        n_sum = sum(new[v][k] for v in new)
        agg.add_row(k, str(o_sum), str(n_sum), f"{n_sum - o_sum:+d}")
    o_f1 = f1(
        sum(old[v]["tp"] for v in old),
        sum(old[v]["fn"] for v in old),
        sum(old[v]["fp"] for v in old),
    )
    n_f1 = f1(
        sum(new[v]["tp"] for v in new),
        sum(new[v]["fn"] for v in new),
        sum(new[v]["fp"] for v in new),
    )
    agg.add_row(
        "F1",
        f"{o_f1 * 100:.4f}%",
        f"{n_f1 * 100:.4f}%",
        f"{(n_f1 - o_f1) * 100:+.4f}pp",
    )
    console.print(agg)

    if regressions:
        console.print(
            f"\n[red bold]HARD FAIL: {len(regressions)} fold(s) regressed by "
            f"more than {-args.fail_delta_pp:.2f}pp F1:[/red bold]"
        )
        for r in regressions:
            console.print(r)
        return 1

    console.print(
        f"\n[green bold]PASS: no fold regressed by more than "
        f"{-args.fail_delta_pp:.2f}pp F1.[/green bold]"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
