#!/usr/bin/env python3
"""Coherence-invariant audit summary across trusted-29.

Runs C-1..C-5 invariants on each video's current DB state and reports:
  - Total violations by invariant
  - Per-video violation counts
  - Total errors vs warnings

Designed for OFF vs scorer-ON A/B: run twice (once per DB state), save
JSON snapshots, diff.

Usage:
    cd analysis
    uv run python scripts/audit_coherence_trusted_29_2026_05_17.py --label v3_1_on
    uv run python scripts/audit_coherence_trusted_29_2026_05_17.py --label scorer_off --compare-to v3_1_on
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rallycut.tracking.coherence_invariants import run_all  # noqa: E402

DB_DSN = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5436/rallycut",
)
TRUSTED_29 = (
    "titi", "toto", "lulu", "wawa", "caco", "cece", "cici", "cuco",
    "gaga", "kaka", "kiki", "juju", "yeye", "keke",
    "gigi", "gugu", "mame", "meme", "mimi", "moma", "mumu",
    "papa", "pepe", "pipi", "popo", "pupu", "veve", "vivi", "vovo",
)

OUT_DIR = Path("reports/coherence_trusted_29_2026_05_17")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def video_ids() -> list[tuple[str, str]]:
    with psycopg.connect(DB_DSN) as conn:
        cur = conn.execute(
            "SELECT id, name FROM videos WHERE name = ANY(%s) ORDER BY name",
            [list(TRUSTED_29)],
        )
        return [(str(r[0]), str(r[1])) for r in cur.fetchall()]


def audit_all() -> dict:
    results: dict[str, dict] = {}
    invariant_totals: Counter = Counter()
    severity_totals: Counter = Counter()
    grand_violations = 0
    grand_errors = 0
    for vid, name in video_ids():
        try:
            violations, _stale = run_all(video_id=vid)
        except Exception as exc:
            print(f"  {name}: ERROR {exc}", flush=True)
            continue
        per_invariant: Counter = Counter()
        per_severity: Counter = Counter()
        for v in violations:
            per_invariant[v.invariant] += 1
            per_severity[v.severity] += 1
            invariant_totals[v.invariant] += 1
            severity_totals[v.severity] += 1
        n_total = len(violations)
        n_errors = per_severity.get("error", 0)
        grand_violations += n_total
        grand_errors += n_errors
        results[name] = {
            "video_id": vid,
            "total": n_total,
            "errors": n_errors,
            "by_invariant": dict(per_invariant),
        }
        print(f"  {name:6s} total={n_total:4d}  err={n_errors:3d}  "
              + " ".join(f"{k}={v}" for k, v in sorted(per_invariant.items())),
              flush=True)
    return {
        "totals": {
            "violations": grand_violations,
            "errors": grand_errors,
            "by_invariant": dict(invariant_totals),
            "by_severity": dict(severity_totals),
        },
        "by_video": results,
    }


def diff(baseline: dict, candidate: dict) -> None:
    print()
    print("=== A/B DIFF ===", flush=True)
    bt, ct = baseline["totals"], candidate["totals"]
    print(f"  Total violations: {bt['violations']} → {ct['violations']} "
          f"(Δ {ct['violations'] - bt['violations']:+d})", flush=True)
    print(f"  Errors only:      {bt['errors']} → {ct['errors']} "
          f"(Δ {ct['errors'] - bt['errors']:+d})", flush=True)
    print("  Per invariant:", flush=True)
    all_invariants = sorted(set(bt["by_invariant"]) | set(ct["by_invariant"]))
    for inv in all_invariants:
        bv = bt["by_invariant"].get(inv, 0)
        cv = ct["by_invariant"].get(inv, 0)
        marker = "▲" if cv > bv else ("▼" if cv < bv else " ")
        print(f"    {inv:8s} {bv:4d} → {cv:4d} (Δ {cv - bv:+4d}) {marker}",
              flush=True)
    print("  Per video deltas (only changes):", flush=True)
    by_video_baseline = baseline.get("by_video", {})
    by_video_candidate = candidate.get("by_video", {})
    all_v = sorted(set(by_video_baseline) | set(by_video_candidate))
    diffs = []
    for v in all_v:
        b = by_video_baseline.get(v, {}).get("total", 0)
        c = by_video_candidate.get(v, {}).get("total", 0)
        if b != c:
            diffs.append((v, b, c, c - b))
    if not diffs:
        print("    (no per-video changes)", flush=True)
    else:
        for v, b, c, d in sorted(diffs, key=lambda x: -abs(x[3])):
            marker = "▲" if d > 0 else "▼"
            print(f"    {v:6s} {b:3d} → {c:3d} (Δ {d:+d}) {marker}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--label", type=str, default="current",
                   help="Snapshot label (e.g. scorer_off, v3_1_on)")
    p.add_argument("--compare-to", type=str, default=None,
                   help="Label of a prior snapshot to diff against")
    args = p.parse_args()

    print(f"Auditing {len(TRUSTED_29)} trusted videos…", flush=True)
    result = audit_all()
    out = OUT_DIR / f"{args.label}.json"
    out.write_text(json.dumps(result, indent=2))
    print()
    t = result["totals"]
    print(f"Total: {t['violations']} violations ({t['errors']} errors)", flush=True)
    print("By invariant:", flush=True)
    for inv, n in sorted(t["by_invariant"].items()):
        print(f"  {inv:8s} {n}", flush=True)
    print(f"Wrote {out}", flush=True)

    if args.compare_to:
        cmp_path = OUT_DIR / f"{args.compare_to}.json"
        if not cmp_path.exists():
            print(f"\nNo baseline snapshot at {cmp_path}", flush=True)
        else:
            baseline = json.loads(cmp_path.read_text())
            diff(baseline, result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
