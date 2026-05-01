"""Analyze Phase-1 H-Profile-Drift probe outputs.

Reads the three verdict files (baseline, dropema, baseline_restore) plus all
probe sidecars under analysis/reports/profile_drift_probe/, and emits a
markdown summary suitable for the PHASE1_FINDINGS memo.

Usage:
    uv run python scripts/analyze_phase1_drift.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

OUTDIR = Path(__file__).resolve().parents[1] / "reports" / "profile_drift_probe"

PANEL_BAD_RALLIES = {
    ("b5fb0594", "r10"),
    ("5c756c41", "r03"),
    ("5c756c41", "r07"),
    ("7d77980f", "r02"),
}


def parse_verdict(path: Path) -> dict[str, dict[str, str]]:
    """Parse panel_verdict_per_frame.py output -> {rally_tag: {field: value}}."""
    if not path.exists():
        return {}
    out: dict[str, dict[str, str]] = {}
    in_table = False
    for line in path.read_text().splitlines():
        if line.startswith("rally") and "expect" in line:
            in_table = True
            continue
        if in_table and line.startswith("---"):
            continue
        if in_table and line.strip() and not line.startswith("AGREES"):
            parts = line.split(maxsplit=8)
            if len(parts) < 6:
                continue
            rally_tag = parts[0]
            row = {
                "kind": parts[1],
                "expect": parts[2],
                "actual": parts[3],
                "agree": parts[4],
                "pids": parts[5] if len(parts) > 5 else "",
                "derived_shape": parts[8] if len(parts) > 8 else "",
            }
            out[rally_tag] = row
        elif line.startswith("AGREES:"):
            in_table = False
    return out


def agrees_count(verdict: dict[str, dict[str, str]]) -> int:
    return sum(1 for r in verdict.values() if r.get("agree") == "YES")


def compare_verdicts(
    baseline: dict[str, dict[str, str]],
    dropema: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    """Per-rally comparison of baseline vs dropema."""
    rows: list[dict[str, str]] = []
    rallies = sorted(set(baseline.keys()) | set(dropema.keys()))
    for rally in rallies:
        b = baseline.get(rally, {})
        d = dropema.get(rally, {})
        rows.append({
            "rally": rally,
            "kind": b.get("kind", d.get("kind", "?")),
            "expected": b.get("expect", d.get("expect", "?")),
            "baseline_actual": b.get("actual", "?"),
            "baseline_agree": b.get("agree", "?"),
            "baseline_shape": b.get("derived_shape", ""),
            "dropema_actual": d.get("actual", "?"),
            "dropema_agree": d.get("agree", "?"),
            "dropema_shape": d.get("derived_shape", ""),
        })
    return rows


def panel_bad_outcome(rows: list[dict[str, str]]) -> dict[str, dict[str, bool]]:
    """For each panel BAD rally, did dropema flip it to GOOD?"""
    out: dict[str, dict[str, bool]] = {}
    for row in rows:
        rally = row["rally"]
        # rally tag like "b5fb0594/r10"
        parts = rally.split("/")
        if len(parts) != 2:
            continue
        if (parts[0], parts[1]) not in PANEL_BAD_RALLIES:
            continue
        baseline_bad = row["baseline_actual"] == "BAD"
        dropema_good = row["dropema_actual"] == "GOOD"
        out[rally] = {
            "baseline_bad": baseline_bad,
            "dropema_good": dropema_good,
            "flipped_good": baseline_bad and dropema_good,
        }
    return out


def good_or_control_regressed(
    rows: list[dict[str, str]],
) -> list[str]:
    """Rallies that were GOOD/CTRL-GOOD in baseline but BAD in dropema."""
    regressed: list[str] = []
    for row in rows:
        if row["baseline_actual"] == "GOOD" and row["dropema_actual"] == "BAD":
            regressed.append(row["rally"])
    return regressed


def load_sidecars(outdir: Path) -> dict[str, list[Path]]:
    """Group sidecars by (video_short, mode)."""
    pattern = re.compile(r"^([0-9a-f]{8})_(baseline|dropema|baseline_restore)_\d{8}T\d{6}Z\.json$")
    grouped: dict[str, list[Path]] = {}
    for path in sorted(outdir.glob("*.json")):
        m = pattern.match(path.name)
        if not m:
            continue
        key = f"{m.group(1)}/{m.group(2)}"
        grouped.setdefault(key, []).append(path)
    return grouped


def drift_summary(sidecar_path: Path) -> dict[str, str]:
    """Compute drift magnitude for a single sidecar."""
    with sidecar_path.open() as f:
        p = json.load(f)
    update_records = p["update_records"]
    if not update_records:
        return {}
    # For each PID, compute lower_hist_l2 range across rallies
    pid_l2: dict[str, list[float]] = {}
    for u in update_records:
        for pid_str, snap in (u.get("after") or {}).items():
            v = snap.get("lower_hist_l2")
            if v is None:
                continue
            pid_l2.setdefault(pid_str, []).append(float(v))
    summary = {
        "video": p["video_id"][:8],
        "drop_ema": p["EXPERIMENTAL_DROP_PROFILE_EMA"],
        "iter_records": str(len(p["iter_records"])),
        "update_records": str(len(update_records)),
    }
    for pid, l2s in sorted(pid_l2.items()):
        if not l2s:
            continue
        rng = max(l2s) - min(l2s)
        summary[f"pid{pid}_lower_l2_range"] = f"{rng:.4f}"
    return summary


def main() -> None:
    baseline = parse_verdict(OUTDIR / "verdict_baseline.txt")
    dropema = parse_verdict(OUTDIR / "verdict_dropema.txt")
    restore = parse_verdict(OUTDIR / "verdict_baseline_restore.txt")

    print("# Phase-1 verdict counts")
    print()
    print(f"- baseline:         AGREES {agrees_count(baseline)}/13")
    print(f"- dropema:          AGREES {agrees_count(dropema)}/13")
    print(f"- baseline_restore: AGREES {agrees_count(restore)}/13"
          " (sanity: should match baseline)")
    print()

    if not baseline or not dropema:
        print("**Missing verdict files; aborting comparison.**")
        return

    rows = compare_verdicts(baseline, dropema)

    print("# Per-rally comparison (baseline vs dropema)")
    print()
    print("| rally | kind | expected | baseline | dropema | flipped? |")
    print("|---|---|---|---|---|---|")
    for row in rows:
        flip = "→GOOD" if (
            row["baseline_actual"] == "BAD" and row["dropema_actual"] == "GOOD"
        ) else (
            "→BAD" if (
                row["baseline_actual"] == "GOOD" and row["dropema_actual"] == "BAD"
            ) else "—"
        )
        print(
            f"| {row['rally']} | {row['kind']} | {row['expected']} "
            f"| {row['baseline_actual']} ({row['baseline_shape']}) "
            f"| {row['dropema_actual']} ({row['dropema_shape']}) | {flip} |"
        )
    print()

    print("# Panel BAD rallies — flip status")
    print()
    bad_outcomes = panel_bad_outcome(rows)
    flipped_count = sum(1 for v in bad_outcomes.values() if v["flipped_good"])
    print(f"- BAD→GOOD flips: {flipped_count} / 4 panel-BAD rallies")
    for rally, info in sorted(bad_outcomes.items()):
        print(f"  - {rally}: baseline_bad={info['baseline_bad']}, "
              f"dropema_good={info['dropema_good']}, "
              f"flipped={info['flipped_good']}")
    print()

    regressions = good_or_control_regressed(rows)
    print(f"- GOOD→BAD regressions in dropema: {len(regressions)}")
    for r in regressions:
        print(f"  - {r}")
    print()

    print("# Phase 1 verdict gate")
    print()
    if flipped_count >= 3 and len(regressions) == 0:
        print("**CASCADE CONFIRMED** — proceed to Phase 2.")
    elif flipped_count == 0 or agrees_count(dropema) < agrees_count(baseline):
        print("**CASCADE FALSIFIED** — drop EMA hypothesis, write NO-GO memo.")
    else:
        print("**CASCADE PARTIAL** — stop, redesign, write findings.")
    print()

    print("# Probe sidecar drift summary")
    print()
    sidecars = load_sidecars(OUTDIR)
    for key, paths in sorted(sidecars.items()):
        if not paths:
            continue
        latest = paths[-1]
        print(f"## {key} (sidecar: {latest.name})")
        s = drift_summary(latest)
        for k, v in s.items():
            print(f"- {k}: {v}")
        print()


if __name__ == "__main__":
    main()
