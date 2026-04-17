"""Analyze the Pattern A rescue-gate sweep and measure Pattern C's marginal impact.

Reads cell snapshots from `outputs/action_errors/sweep/*/` and compares to two
baselines (Arm B off + Pattern C off, Arm B off + Pattern C on) to isolate
each mechanism's contribution. Emits the full `pattern_a_rescue_sweep_2026_04_17.md`
report and a `pattern_c_serve_anchor_2026_04_17.md` decision note.

Runs two small corpus rebuilds (the two baselines) if they are missing. The
baseline cells live at:
  outputs/action_errors/sweep/baseline_no_armb_no_anchor/
  outputs/action_errors/sweep/baseline_no_armb_anchor_only/
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ANALYSIS_DIR = Path(__file__).parent.parent
OUTPUT_DIR = ANALYSIS_DIR / "outputs" / "action_errors"
SWEEP_DIR = OUTPUT_DIR / "sweep"
REPORT_A = ANALYSIS_DIR / "reports" / "pattern_a_rescue_sweep_2026_04_17.md"
REPORT_C = ANALYSIS_DIR / "reports" / "pattern_c_serve_anchor_2026_04_17.md"

GRID: list[tuple[float, int]] = [
    (0.05, 2), (0.05, 3),
    (0.08, 2), (0.08, 3),
    (0.10, 2), (0.10, 3),
    (0.12, 2), (0.12, 3),
    (0.15, 2), (0.15, 3),
]

USER_CONFIRMED_RESCUES: list[tuple[str, int]] = [
    ("fb7f9c23-3544-48bd-910d-10a8f12fd594", 230),
    ("99a01ce4-4a00-4e54-b015-80f843ff21fc", 371),
    ("99a01ce4-4a00-4e54-b015-80f843ff21fc", 813),
    ("71c5d769-581e-4302-ad79-87cfdcb71b80", 234),
]

LATE_TRACK_START_FNS: list[tuple[str, int]] = [
    ("c3b31af2", 42), ("55e78d3a", 45), ("0d84f858", 46),
    ("618002a8", 49), ("30ffb876", 54), ("5c792f3f", 77),
    ("a8ef3948", 82), ("7c7f0ba3", 85), ("39e866fd", 87),
    ("21a9b203", 128), ("9e4f0c7b", 194),  # gt_frame per corpus audit
]


def _cell_dir(floor: float, min_gens: int) -> Path:
    return SWEEP_DIR / f"floor{floor:.2f}_gens{min_gens}"


def _run_baseline(
    name: str,
    arm_b_off: bool,
    anchor_off: bool,
) -> Path:
    cell_dir = SWEEP_DIR / name
    cell_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "import sys",
        "sys.argv = ['build_action_error_corpus.py']",
        "import rallycut.tracking.sequence_action_runtime as _sar",
    ]
    if arm_b_off:
        lines.append("_sar.SEQ_RECOVERY_MIN_GENERATORS = 999")
    if anchor_off:
        lines.append("_sar.SERVE_ANCHOR_TAU = 1.1  # unreachable → anchor disabled")
    lines += [
        "from scripts.build_action_error_corpus import main",
        "main()",
    ]
    code = "\n".join(lines) + "\n"

    print(f"\n=== {name} ===")
    t0 = time.monotonic()
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ANALYSIS_DIR,
        capture_output=True,
        text=True,
    )
    elapsed = time.monotonic() - t0
    (cell_dir / "stdout.log").write_text(proc.stdout)
    (cell_dir / "stderr.log").write_text(proc.stderr)
    if proc.returncode != 0:
        print(f"  FAILED in {elapsed:.1f}s")
        raise SystemExit(1)
    for fn in ("corpus.jsonl", "rally_quality.json", "corpus_annotated.jsonl"):
        src = OUTPUT_DIR / fn
        if src.exists():
            shutil.copy2(src, cell_dir / fn)
    print(f"  ok in {elapsed:.1f}s")
    return cell_dir


def _load_cell(cell_dir: Path) -> dict:
    corpus_file = cell_dir / "corpus.jsonl"
    quality_file = cell_dir / "rally_quality.json"
    if not corpus_file.exists() or not quality_file.exists():
        raise FileNotFoundError(f"Missing snapshot in {cell_dir}")

    rows = [
        json.loads(line)
        for line in corpus_file.read_text().splitlines()
        if line
    ]
    quality = json.loads(quality_file.read_text())

    rally_stats: dict[str, dict] = {
        rid: {"tp": 0, "fn": 0, "wrong_action": 0, "wrong_player": 0}
        for rid in quality
    }
    for r in rows:
        rid = r["rally_id"]
        if rid not in rally_stats:
            rally_stats[rid] = {"tp": 0, "fn": 0, "wrong_action": 0, "wrong_player": 0}
        ec = r["error_class"]
        if ec == "FN_contact":
            rally_stats[rid]["fn"] += 1
        elif ec == "wrong_action":
            rally_stats[rid]["wrong_action"] += 1
        elif ec == "wrong_player":
            rally_stats[rid]["wrong_player"] += 1

    for rid, st in rally_stats.items():
        gt = quality.get(rid, {}).get("gt_contact_count", 0)
        st["tp"] = max(0, gt - st["fn"] - st["wrong_action"] - st["wrong_player"])
        st["extra_pred"] = quality.get(rid, {}).get("extra_predictions", 0)

    totals = {k: sum(s[k] for s in rally_stats.values())
              for k in ("tp", "fn", "wrong_action", "wrong_player", "extra_pred")}

    user_rescues = {}
    for rid, frame in USER_CONFIRMED_RESCUES:
        fn_at = any(
            r["rally_id"] == rid
            and r["gt_frame"] == frame
            and r["error_class"] == "FN_contact"
            for r in rows
        )
        user_rescues[f"{rid[:8]}:{frame}"] = not fn_at

    late_track_rescues = {}
    for rid_prefix, frame in LATE_TRACK_START_FNS:
        fn_at = any(
            r["rally_id"].startswith(rid_prefix)
            and r["gt_frame"] == frame
            and r["error_class"] == "FN_contact"
            for r in rows
        )
        late_track_rescues[f"{rid_prefix}:{frame}"] = not fn_at

    return {
        "totals": totals,
        "rally_stats": rally_stats,
        "user_rescues": user_rescues,
        "late_track_rescues": late_track_rescues,
    }


def _compare(baseline: dict, cell: dict) -> dict:
    b, c = baseline["totals"], cell["totals"]
    delta = {k: c[k] - b[k] for k in b}
    regressed = []
    for rid, b_st in baseline["rally_stats"].items():
        c_st = cell["rally_stats"].get(rid)
        if c_st is None:
            continue
        b_score = b_st["tp"] + b_st["wrong_player"]
        c_score = c_st["tp"] + c_st["wrong_player"]
        extra_delta = c_st["extra_pred"] - b_st["extra_pred"]
        if c_score - b_score <= -1 or extra_delta >= 2:
            regressed.append(rid)
    return {"delta": delta, "regressed": regressed}


def _write_report_a(baseline_c: dict, cells: dict[str, dict]) -> None:
    REPORT_A.parent.mkdir(parents=True, exist_ok=True)
    b = baseline_c["totals"]

    lines = [
        "# Pattern A rescue-gate sweep — 2026-04-17\n",
        "## Setup\n",
        "Two-arm rescue gate in `rallycut/tracking/contact_detector.py`. "
        "Arm A unchanged (`SEQ_RECOVERY_CLF_FLOOR=0.20`). Arm B (new) "
        "rescues when `conf >= FLOOR_MULTIGEN AND n_generators >= MIN_GENERATORS "
        "AND player_distance <= 0.15 AND seq_peak >= 0.80`. Baseline disables "
        "Arm B (`SEQ_RECOVERY_MIN_GENERATORS=999`) with Pattern C (serve anchor) "
        "enabled so Arm B's marginal impact is isolated.\n",
        "## Baseline (Arm A + Pattern C, Arm B disabled)\n",
        f"TP={b['tp']}  FN={b['fn']}  wrong_action={b['wrong_action']}  "
        f"wrong_player={b['wrong_player']}  extra_pred={b['extra_pred']}\n",
        "## Grid (Δ vs baseline)\n",
        "| FLOOR | MIN_GENS | ΔTP | ΔFN | Δwrong_action | Δwrong_player | "
        "Δextra_pred | n_regressed | user_rescues |",
        "|------:|---------:|----:|----:|--------------:|--------------:|"
        "------------:|------------:|:-------------|",
    ]
    for floor, min_gens in GRID:
        key = f"floor{floor:.2f}_gens{min_gens}"
        if key not in cells:
            continue
        cmp = _compare(baseline_c, cells[key])
        d = cmp["delta"]
        n_reg = len(cmp["regressed"])
        confirmed = sum(1 for v in cells[key]["user_rescues"].values() if v)
        lines.append(
            f"| {floor:.2f} | {min_gens} | "
            f"{d['tp']:+d} | {d['fn']:+d} | {d['wrong_action']:+d} | "
            f"{d['wrong_player']:+d} | {d['extra_pred']:+d} | "
            f"{n_reg} | {confirmed}/4 |"
        )
    lines += [
        "",
        "## Threshold selection",
        "",
        "Rule: maximize ΔTP subject to Δextra_pred / ΔTP ≤ 0.5, n_regressed ≤ 5, "
        "ΔFN ≤ 0, and user_rescues = 4/4. Tiebreak toward higher FLOOR then "
        "higher MIN_GENS.",
        "",
    ]
    # Compute best cell.
    best_key = None
    best_tp = -1
    for floor, min_gens in GRID:
        key = f"floor{floor:.2f}_gens{min_gens}"
        if key not in cells:
            continue
        cmp = _compare(baseline_c, cells[key])
        d = cmp["delta"]
        confirmed = sum(1 for v in cells[key]["user_rescues"].values() if v)
        if (
            d["fn"] <= 0
            and confirmed == 4
            and len(cmp["regressed"]) <= 5
            and (d["extra_pred"] <= 0 or d["tp"] > 0 and d["extra_pred"] / max(1, d["tp"]) <= 0.5)
            and d["tp"] > best_tp
        ):
            best_tp = d["tp"]
            best_key = key
    if best_key:
        lines.append(f"**Recommended cell: `{best_key}`** (ΔTP=+{best_tp}).\n")
    else:
        lines.append("**No cell cleared the decision rule.** See per-rally regressions "
                     "or fallback to higher MIN_GENS.\n")

    REPORT_A.write_text("\n".join(lines) + "\n")
    print(f"Pattern A report: {REPORT_A}")
    return best_key


def _write_report_c(baseline_none: dict, baseline_c: dict) -> None:
    REPORT_C.parent.mkdir(parents=True, exist_ok=True)
    cmp = _compare(baseline_none, baseline_c)
    d = cmp["delta"]
    late_rescued = sum(1 for v in baseline_c["late_track_rescues"].values() if v)
    late_total = len(baseline_c["late_track_rescues"])

    ship = (
        late_rescued >= 6
        and d["extra_pred"] <= 3
        and len(cmp["regressed"]) == 0
    )
    verdict = "SHIP" if ship else "NO SHIP"

    lines = [
        "# Pattern C (rally-start serve anchor) validation — 2026-04-17\n",
        "## Setup\n",
        "Synthetic-serve injection in `rallycut/tracking/contact_detector.py` via "
        "`_maybe_anchor_rally_start_serve`. Fires when no existing contact lives "
        "in the first `SERVE_ANCHOR_MAX_FRAME=90` frames AND MS-TCN++ serve-class "
        "probability peaks above `SERVE_ANCHOR_TAU=0.85` in that window. Baseline "
        "runs with `SERVE_ANCHOR_TAU=1.1` (unreachable → anchor disabled).\n",
        "## Δ vs anchor-disabled baseline\n",
        f"ΔTP={d['tp']:+d}  ΔFN={d['fn']:+d}  Δwrong_action={d['wrong_action']:+d}  "
        f"Δwrong_player={d['wrong_player']:+d}  Δextra_pred={d['extra_pred']:+d}  "
        f"n_regressed={len(cmp['regressed'])}  late-track rescues: {late_rescued}/{late_total}\n",
        "## Decision rule\n",
        "- Ship if ≥6 of the 11 late-track-start FNs are rescued AND "
        "Δextra_pred ≤ 3 AND no rally regresses.\n",
        f"\n## Verdict: **{verdict}**\n",
    ]
    REPORT_C.write_text("\n".join(lines) + "\n")
    print(f"Pattern C report: {REPORT_C}")
    return ship


def main() -> None:
    # Load sweep cells.
    cells: dict[str, dict] = {}
    for floor, min_gens in GRID:
        cell_dir = _cell_dir(floor, min_gens)
        if cell_dir.exists():
            cells[cell_dir.name] = _load_cell(cell_dir)

    # Ensure baselines exist; run them otherwise.
    bl_none_dir = SWEEP_DIR / "baseline_no_armb_no_anchor"
    bl_anchor_dir = SWEEP_DIR / "baseline_no_armb_anchor_only"

    if not (bl_none_dir / "corpus.jsonl").exists():
        _run_baseline("baseline_no_armb_no_anchor", arm_b_off=True, anchor_off=True)
    if not (bl_anchor_dir / "corpus.jsonl").exists():
        _run_baseline("baseline_no_armb_anchor_only", arm_b_off=True, anchor_off=False)

    baseline_none = _load_cell(bl_none_dir)
    baseline_c = _load_cell(bl_anchor_dir)

    print("\nBaseline (no Arm B, no anchor):", baseline_none["totals"])
    print("Baseline (no Arm B, anchor on):", baseline_c["totals"])
    for key, cell in cells.items():
        print(f"{key}:", cell["totals"])

    _write_report_a(baseline_c, cells)
    _write_report_c(baseline_none, baseline_c)


if __name__ == "__main__":
    main()
