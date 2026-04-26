"""lock_baseline_v2.py — re-lock the 9-fixture baseline at HEAD code.

Why v2: the existing baseline (`baseline_2026_04_24.json`) was scored under
a now-stale GT-resolution convention (legacy match-analysis trackToPlayer,
pre action-GT trackId-anchor schema fix). With the schema fix landed and
the Surface A circular-GT-resolution fix in `measure_relabel_lift.py`,
the locked baseline needs to be regenerated at HEAD with crops in place
so the relabel-lift comparison has an apples-to-apples reference.

Per-fixture cycle:
  1. Run `match-players <video>` (with crops in DB).
  2. Run `reattribute-actions <video>`.
  3. Score Surface A — gt resolution uses the JUST-WRITTEN match_analysis ttp
     (the v2 reference convention).

Output: `analysis/reports/attribution_rebuild/baseline_2026_04_24_v2.json`
with the same shape as v1 plus a `reference_ttp_by_rally` block per fixture
so downstream tools can pin against the same convention.

Usage:
    cd analysis
    uv run python scripts/lock_baseline_v2.py
    uv run python scripts/lock_baseline_v2.py --fixtures cece tata
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_ANALYSIS_DIR = Path(__file__).resolve().parents[1]
if str(_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS_DIR))

from rallycut.evaluation.attribution_bench import (  # noqa: E402
    MATCH_TOLERANCE_FRAMES,
    WRONG_CATEGORIES,
    aggregate,
    score_rally,
)
from rallycut.evaluation.db import get_connection  # noqa: E402

# Reuse the Surface A scoring helpers from measure_relabel_lift so v2 baseline
# uses the same code path the relabel-lift harness scores against.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_relabel_lift import (  # noqa: E402
    _team_templates_pid_to_team,
    _ttp_by_rally_id,
    load_rallies_for_surface_a,
)

REPO_ROOT = _ANALYSIS_DIR.parent
ATTR_REPORTS = _ANALYSIS_DIR / "reports" / "attribution_rebuild"
FIXTURE_REGISTRY = ATTR_REPORTS / "fixture_video_ids_2026_04_24.json"
OUT_PATH = ATTR_REPORTS / "baseline_2026_04_24_v2.json"


def _run_cli(args: list[str], log_path: Path, label: str) -> int:
    cmd = ["uv", "run", "rallycut", *args]
    t0 = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as fh:
        fh.write(f"$ {' '.join(shlex.quote(c) for c in cmd)}\n\n")
        proc = subprocess.run(
            cmd, cwd=_ANALYSIS_DIR, capture_output=True, text=True
        )
        fh.write(proc.stdout)
        if proc.stderr:
            fh.write("\n--- stderr ---\n")
            fh.write(proc.stderr)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-15:])
        print(f"      [ERROR] {label} exit={proc.returncode} in {elapsed:.1f}s")
        print(f"      log tail:\n{tail}")
    else:
        print(f"      [{label}] OK in {elapsed:.1f}s")
    return proc.returncode


def _has_ref_crops(video_id: str) -> int:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM player_reference_crops WHERE video_id = %s",
            [video_id],
        )
        row = cur.fetchone()
        return int(row[0]) if row else 0


def lock_fixture(
    fixture: str, video_id: str, log_dir: Path
) -> dict[str, Any]:
    """Run match-players + reattribute-actions then score Surface A.
    Returns a per-fixture entry for the v2 baseline JSON."""
    print(f"\n=== {fixture} ({video_id[:8]}) ===")
    log_dir.mkdir(parents=True, exist_ok=True)

    n_crops = _has_ref_crops(video_id)
    print(f"  ref_crops in DB: {n_crops}")
    if n_crops == 0:
        print("  WARN: no reference crops — match-players will run blind, "
              "v2 baseline will reflect blind ttp for this fixture")

    print("  match-players...")
    rc = _run_cli(
        ["match-players", video_id, "--quiet"],
        log_dir / "match_players.log", "match-players",
    )
    if rc != 0:
        return {"fixture": fixture, "video_id": video_id, "error": "match-players-failed"}

    print("  reattribute-actions...")
    rc = _run_cli(
        ["reattribute-actions", video_id],
        log_dir / "reattribute.log", "reattribute-actions",
    )
    if rc != 0:
        return {"fixture": fixture, "video_id": video_id, "error": "reattribute-failed"}

    # Score Surface A using the (just-written) DB ttp as the reference.
    rallies = load_rallies_for_surface_a(video_id, fixture)
    agg = aggregate(rallies)
    fxs = list(agg["per_fixture"].values())
    if not fxs:
        print("  [score] no GT rallies → empty entry")
        per_fx = {"counts": {"n_gt_actions": 0}, "rates": {}}
    else:
        per_fx = fxs[0]

    counts = per_fx["counts"]
    rates = per_fx.get("rates", {})
    n = counts.get("n_gt_actions", 0)
    correct = counts.get("correct", 0)
    rate = rates.get("correct_rate", 0.0)
    print(f"  Surface A: n={n} correct={correct} ({rate:.1%}) "
          f"missing={counts.get('missing', 0)}")

    # Capture the reference ttp keyed by rally_id (for later use).
    reference_ttp = _ttp_by_rally_id(video_id)

    return {
        "fixture": fixture,
        "video_id": video_id,
        "n_ref_crops": n_crops,
        "counts": counts,
        "rates": rates,
        "rallies": rallies,  # full scored records for diff/transition analysis
        "reference_ttp_by_rally": {
            rid: {str(k): v for k, v in m.items()}
            for rid, m in reference_ttp.items()
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--fixtures", nargs="*", default=None,
        help="Fixture names (default: all 9 baseline fixtures).",
    )
    ap.add_argument(
        "--out", type=Path, default=OUT_PATH,
        help=f"Output path (default: {OUT_PATH}).",
    )
    args = ap.parse_args()

    fixture_map = json.loads(FIXTURE_REGISTRY.read_text())["fixtures"]
    if args.fixtures:
        unknown = [f for f in args.fixtures if f not in fixture_map]
        if unknown:
            print(f"Unknown fixture(s): {unknown}", file=sys.stderr)
            print(f"Known: {sorted(fixture_map)}", file=sys.stderr)
            return 2
        targets = [(f, fixture_map[f]["video_id"]) for f in args.fixtures]
    else:
        targets = [(f, fixture_map[f]["video_id"]) for f in fixture_map]

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    log_root = ATTR_REPORTS / "lock_baseline_v2" / ts
    log_root.mkdir(parents=True, exist_ok=True)
    print(f"Logs: {log_root}")
    print(f"Targets: {[t[0] for t in targets]}")

    fixture_entries: list[dict[str, Any]] = []
    for i, (fx, vid) in enumerate(targets, 1):
        print(f"\n[{i}/{len(targets)}] {fx}...")
        entry = lock_fixture(fx, vid, log_root / fx)
        fixture_entries.append(entry)

    # Aggregate across fixtures (mirroring v1 shape).
    fixtures_counts: dict[str, dict[str, Any]] = {}
    fixtures_rates: dict[str, dict[str, Any]] = {}
    all_rallies: list[dict[str, Any]] = []
    reference_ttp_by_fixture: dict[str, dict[str, dict[str, int]]] = {}

    for entry in fixture_entries:
        if entry.get("error"):
            print(f"\n[skip-aggregate] {entry['fixture']}: {entry['error']}")
            continue
        fx = entry["fixture"]
        fixtures_counts[fx] = entry["counts"]
        fixtures_rates[fx] = entry["rates"]
        all_rallies.extend(entry.get("rallies") or [])
        reference_ttp_by_fixture[fx] = entry.get("reference_ttp_by_rally", {})

    agg = aggregate(all_rallies) if all_rallies else {"combined": {"counts": {}, "rates": {}}}
    combined = agg["combined"]

    payload = {
        "generated_at": ts,
        "match_tolerance_frames": MATCH_TOLERANCE_FRAMES,
        "source": (
            "match-players + reattribute-actions on each baseline fixture at "
            "HEAD code (with reference crops in DB). Surface A scored via "
            "measure_relabel_lift.load_rallies_for_surface_a — gt resolution "
            "uses the JUST-WRITTEN ttp, which becomes the v2 reference "
            "convention. Used as the apples-to-apples baseline by "
            "measure_relabel_lift.py."
        ),
        "fixtures": fixtures_counts,
        "fixture_rates": fixtures_rates,
        "aggregate": combined,
        "reference_ttp_by_fixture": reference_ttp_by_fixture,
        "fixture_errors": [
            {"fixture": e["fixture"], "error": e["error"]}
            for e in fixture_entries if e.get("error")
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, default=str))

    # Per-fixture report.
    print("\n" + "=" * 90)
    print("BASELINE v2 — Surface A per fixture")
    print("=" * 90)
    print(f"{'fixture':8s} {'n_gt':>4s}  {'correct':>7s}  {'rate':>6s}  "
          f"{'missing':>7s}  {'wrong':>5s}  {'crops':>5s}")
    n_total = correct_total = missing_total = wrong_total = 0
    for entry in fixture_entries:
        if entry.get("error"):
            print(f"{entry['fixture']:8s}   ERROR: {entry['error']}")
            continue
        c = entry["counts"]
        r = entry["rates"]
        n = c.get("n_gt_actions", 0)
        correct = c.get("correct", 0)
        missing = c.get("missing", 0)
        wrong = sum(c.get(k, 0) for k in WRONG_CATEGORIES)
        rate = r.get("correct_rate", 0.0)
        n_crops = entry.get("n_ref_crops", 0)
        print(f"{entry['fixture']:8s} {n:>4d}  {correct:>7d}  {rate:>6.1%}  "
              f"{missing:>7d}  {wrong:>5d}  {n_crops:>5d}")
        n_total += n
        correct_total += correct
        missing_total += missing
        wrong_total += wrong

    rate_total = correct_total / n_total if n_total else 0.0
    print("-" * 90)
    print(f"{'AGG':8s} {n_total:>4d}  {correct_total:>7d}  {rate_total:>6.1%}  "
          f"{missing_total:>7d}  {wrong_total:>5d}")

    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
