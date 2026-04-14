"""Diagnose localize_team_near failure modes.

Measures:
  1. Return None rate (narrow-angle cameras)
  2. When non-None, accuracy vs GT-derived ground truth team_near
  3. Whether restricting to serve window (frames 0-120) improves accuracy
  4. **None-return bucket distribution**: which of the 4 in-code paths
     triggered each None return (empty_input / no_pid_mapped /
     one_team_unmapped / y_gap_below_threshold). See team_identity.py
     lines 151-178 for the exact branches being mirrored.
  5. **Wrong-return Y-gap histogram**: whether wrong returns cluster near
     the min_y_gap=0.03 boundary (→ signal strength is the issue) or are
     uniformly distributed (→ track_to_player phantom flips dominate).

The key question: is team_near errors the bottleneck preventing production
score_accuracy from reaching the formation accuracy ceiling?

Usage:
    cd analysis
    uv run python scripts/diagnose_team_near.py
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rich.console import Console  # noqa: E402,I001
from rich.table import Table  # noqa: E402

from eval_action_detection import _load_track_to_player_maps  # noqa: E402
from eval_score_tracking import load_score_gt  # noqa: E402
from production_eval import _load_team_templates_by_video  # noqa: E402
from rallycut.tracking.action_classifier import (  # noqa: E402
    _find_serving_side_by_formation,
)
from rallycut.tracking.player_tracker import PlayerPosition  # noqa: E402
from rallycut.tracking.team_identity import localize_team_near  # noqa: E402

console = Console()


def _parse_positions(pos_json: list[dict]) -> list[PlayerPosition]:
    return [
        PlayerPosition(
            track_id=p.get("trackId", p.get("track_id", -1)),
            frame_number=p.get("frameNumber", p.get("frame_number", 0)),
            x=p["x"], y=p["y"],
            width=p["width"], height=p["height"],
            confidence=p.get("confidence", 1.0),
        )
        for p in pos_json
    ]


# In-code branch reasons (mirrors team_identity.py:151-178).
REASON_EMPTY_INPUT = "empty_input"
REASON_NO_PID_MAPPED = "no_pid_mapped"
REASON_ONE_TEAM_UNMAPPED = "one_team_unmapped"
REASON_Y_GAP_BELOW = "y_gap_below_threshold"
REASON_OK = "ok"
NONE_REASONS = (
    REASON_EMPTY_INPUT, REASON_NO_PID_MAPPED,
    REASON_ONE_TEAM_UNMAPPED, REASON_Y_GAP_BELOW,
)


def _localize_team_near_debug(
    positions: list[PlayerPosition],
    track_to_player: dict[int, int],
    templates: tuple[Any, Any],
    min_y_gap: float = 0.03,
) -> dict[str, Any]:
    """Mirror of localize_team_near() that reports which branch fired.

    Exact branch structure tracks team_identity.py:151-178. Returns the
    same result (team label or None) plus a `reason` tag and the
    intermediate quantities needed for sub-bucket analysis.
    """
    n_positions = len(positions)
    n_t2p = len(track_to_player)
    base: dict[str, Any] = {
        "result": None, "reason": REASON_EMPTY_INPUT,
        "y_gap": None, "t0_n": 0, "t1_n": 0, "n_pid_mapped": 0,
        "n_positions": n_positions, "n_t2p": n_t2p,
    }

    if not positions or not track_to_player:
        return base

    t0, t1 = templates
    t0_pids = set(t0.player_ids)
    t1_pids = set(t1.player_ids)

    pid_ys: dict[int, list[float]] = {}
    for p in positions:
        pid = track_to_player.get(p.track_id)
        if pid is not None:
            pid_ys.setdefault(pid, []).append(p.y + p.height / 2.0)

    n_pid_mapped = len(pid_ys)
    base["n_pid_mapped"] = n_pid_mapped

    if not pid_ys:
        base["reason"] = REASON_NO_PID_MAPPED
        return base

    t0_ys = [float(np.mean(pid_ys[pid])) for pid in t0_pids if pid in pid_ys]
    t1_ys = [float(np.mean(pid_ys[pid])) for pid in t1_pids if pid in pid_ys]
    base["t0_n"] = len(t0_ys)
    base["t1_n"] = len(t1_ys)

    if not t0_ys or not t1_ys:
        base["reason"] = REASON_ONE_TEAM_UNMAPPED
        return base

    mean_t0 = float(np.mean(t0_ys))
    mean_t1 = float(np.mean(t1_ys))
    y_gap = abs(mean_t0 - mean_t1)
    base["y_gap"] = y_gap

    if y_gap < min_y_gap:
        base["reason"] = REASON_Y_GAP_BELOW
        return base

    base["reason"] = REASON_OK
    base["result"] = t0.team_label if mean_t0 > mean_t1 else t1.team_label
    return base


def main() -> int:
    console.print("[bold]Loading score GT...[/bold]")
    video_rallies = load_score_gt()
    video_ids = set(video_rallies.keys())
    templates_by_vid = _load_team_templates_by_video(video_ids)
    t2p_by_rally = _load_track_to_player_maps(video_ids)
    console.print(f"  {sum(len(r) for r in video_rallies.values())} rallies, "
                  f"{len(templates_by_vid)} videos with templates, "
                  f"{len(t2p_by_rally)} rallies with track_to_player\n")

    # Per-video stats
    vid_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "total": 0, "has_template": 0, "has_t2p": 0,
            "full_returns": 0, "full_none": 0, "full_correct": 0, "full_wrong": 0,
            "early_returns": 0, "early_none": 0, "early_correct": 0, "early_wrong": 0,
        }
    )

    # For each GT-labeled rally, compute:
    # 1. Full-window team_near (default)
    # 2. Early-window team_near (frames 0-120 only)
    # 3. GT team_near: the team label whose gt_serving_team = formation_side-matches

    total_rallies_with_gt = 0
    skipped_no_template = 0
    skipped_no_t2p = 0

    for vid, rallies in sorted(video_rallies.items()):
        stats = vid_stats[vid[:10]]
        templates = templates_by_vid.get(vid)
        if templates is None:
            skipped_no_template += len(rallies)
            continue
        stats["has_template"] = len(rallies)

        for rally in rallies:
            if rally.gt_serving_team is None:
                continue
            total_rallies_with_gt += 1
            stats["total"] += 1

            positions = _parse_positions(rally.positions)
            t2p = t2p_by_rally.get(rally.rally_id)
            if not t2p:
                skipped_no_t2p += 1
                continue
            stats["has_t2p"] += 1

            # Full-rally team_near
            full_tn = localize_team_near(positions, t2p, templates)

            # Early-window team_near (frames 0-120 = serve formation)
            early_positions = [p for p in positions if p.frame_number < 120]
            early_tn = localize_team_near(early_positions, t2p, templates)

            # Ground truth for team_near: derive from gt_serving_team and formation_side
            # If formation says "near" served and gt says team X served, then team X is near.
            # We use formation as the gating: only evaluate team_near when formation is correct.
            net_y = rally.court_split_y if rally.court_split_y else 0.5
            formation_side, _ = _find_serving_side_by_formation(
                positions, net_y=net_y, start_frame=0,
            )
            if formation_side is None:
                continue

            # GT-derived physical side (from team + flips)
            # Compute per-rally near_is_a
            # Use side_flipped directly: if flipped, gt_team maps to opposite side
            # We need a reference: assume initial_near_is_a=True, then invert per side_flipped
            # Simpler: use majority vote over all GT rallies in this video
            # But we don't have that here — use side_flipped flag from rally
            # Actually, what we want is: GT says team X served. If formation says "near",
            # then team X is the near team (correct team_near prediction = team X's label).

            t0, t1 = templates
            # Map gt_serving_team ("A"/"B") to template label
            # We need another source for this — use per-video majority-vote calibration
            # For simplicity, assume team_label "0" = A and "1" = B, adjusting if wrong
            # Actually, we don't need to know that — we just need to know which template
            # label matches the expected near team given formation_side and gt_team.

            # Skip: we need per-video label_a to do this properly.
            # Alternative: measure consistency — does full agree with early?
            if full_tn is not None:
                stats["full_returns"] += 1
            else:
                stats["full_none"] += 1

            if early_tn is not None:
                stats["early_returns"] += 1
            else:
                stats["early_none"] += 1

    # Second pass: use per-video calibration to measure accuracy
    # label_a = which template label corresponds to GT "A"
    # Then for a GT-labeled rally where formation is correct,
    # expected team_near = label_a if (gt=A and formation="near") or (gt=B and formation="far")

    vid_label_a: dict[str, str] = {}
    for vid, rallies in sorted(video_rallies.items()):
        templates = templates_by_vid.get(vid)
        if templates is None:
            continue
        t0, t1 = templates
        votes: dict[str, int] = {t0.team_label: 0, t1.team_label: 0}

        for rally in rallies:
            if rally.gt_serving_team is None:
                continue
            positions = _parse_positions(rally.positions)
            t2p = t2p_by_rally.get(rally.rally_id)
            if not t2p:
                continue
            net_y = rally.court_split_y if rally.court_split_y else 0.5
            formation_side, _ = _find_serving_side_by_formation(
                positions, net_y=net_y, start_frame=0,
            )
            if formation_side is None:
                continue
            tn = localize_team_near(positions, t2p, templates)
            if tn is None:
                continue
            # If formation="near", serving team's label = tn; if "far", opposite
            if formation_side == "near":
                serving_label = tn
            else:
                serving_label = t0.team_label if tn == t1.team_label else t1.team_label
            # Vote for which label = A
            if rally.gt_serving_team == "A":
                votes[serving_label] = votes.get(serving_label, 0) + 1
            else:
                votes[serving_label] = votes.get(serving_label, 0) - 1

        if votes:
            vid_label_a[vid] = max(votes, key=lambda k: votes[k])

    # Third pass: measure team_near accuracy using label_a
    for vid, rallies in sorted(video_rallies.items()):
        templates = templates_by_vid.get(vid)
        if templates is None or vid not in vid_label_a:
            continue
        stats = vid_stats[vid[:10]]
        label_a = vid_label_a[vid]
        t0, t1 = templates
        label_b = t0.team_label if label_a == t1.team_label else t1.team_label

        for rally in rallies:
            if rally.gt_serving_team is None:
                continue
            positions = _parse_positions(rally.positions)
            t2p = t2p_by_rally.get(rally.rally_id)
            if not t2p:
                continue
            net_y = rally.court_split_y if rally.court_split_y else 0.5
            formation_side, _ = _find_serving_side_by_formation(
                positions, net_y=net_y, start_frame=0,
            )
            if formation_side is None:
                continue

            # Expected team_near: the team that's near based on formation + gt
            # If formation="near" and gt="A" → near team has label_a
            # If formation="near" and gt="B" → near team has label_b
            # If formation="far" and gt="A" → near team has label_b (A is far)
            # If formation="far" and gt="B" → near team has label_a (B is far)
            if formation_side == "near":
                expected_tn = label_a if rally.gt_serving_team == "A" else label_b
            else:
                expected_tn = label_b if rally.gt_serving_team == "A" else label_a

            full_tn = localize_team_near(positions, t2p, templates)
            early_positions = [p for p in positions if p.frame_number < 120]
            early_tn = localize_team_near(early_positions, t2p, templates)

            if full_tn is not None:
                if full_tn == expected_tn:
                    stats["full_correct"] += 1
                else:
                    stats["full_wrong"] += 1

            if early_tn is not None:
                if early_tn == expected_tn:
                    stats["early_correct"] += 1
                else:
                    stats["early_wrong"] += 1

    # Print summary
    console.print(f"Total GT rallies: {total_rallies_with_gt}")
    console.print(f"  Skipped (no template): {skipped_no_template}")
    console.print(f"  Skipped (no t2p): {skipped_no_t2p}")
    console.print()

    table = Table(title="localize_team_near: Full vs Early window (frames 0-120)")
    table.add_column("Video", style="cyan")
    table.add_column("N", justify="right")
    table.add_column("Full: ret%", justify="right")
    table.add_column("Full: acc", justify="right")
    table.add_column("Early: ret%", justify="right")
    table.add_column("Early: acc", justify="right")
    table.add_column("Δ acc", justify="right")

    total_full_correct = total_full_wrong = 0
    total_early_correct = total_early_wrong = 0
    total_full_ret = total_early_ret = 0
    total_n = 0

    for vid in sorted(vid_stats.keys()):
        s = vid_stats[vid]
        n = s["total"]
        if n == 0:
            continue
        full_n = s["full_correct"] + s["full_wrong"]
        early_n = s["early_correct"] + s["early_wrong"]
        full_ret = full_n / n if n else 0
        full_acc = s["full_correct"] / full_n if full_n else 0
        early_ret = early_n / n if n else 0
        early_acc = s["early_correct"] / early_n if early_n else 0
        delta = early_acc - full_acc
        delta_str = f"{delta:+.0%}" if abs(delta) > 0.005 else "0%"
        style = "red" if full_acc < 0.9 else ""
        table.add_row(
            vid, str(n),
            f"{full_ret:.0%}", f"{full_acc:.0%}",
            f"{early_ret:.0%}", f"{early_acc:.0%}",
            delta_str, style=style,
        )
        total_n += n
        total_full_correct += s["full_correct"]
        total_full_wrong += s["full_wrong"]
        total_early_correct += s["early_correct"]
        total_early_wrong += s["early_wrong"]
        total_full_ret += full_n
        total_early_ret += early_n

    full_acc_tot = total_full_correct / max(1, total_full_correct + total_full_wrong)
    early_acc_tot = total_early_correct / max(1, total_early_correct + total_early_wrong)
    table.add_row(
        "TOTAL", str(total_n),
        f"{total_full_ret/max(1,total_n):.1%}",
        f"{full_acc_tot:.1%}",
        f"{total_early_ret/max(1,total_n):.1%}",
        f"{early_acc_tot:.1%}",
        f"{early_acc_tot - full_acc_tot:+.1%}",
        style="bold",
    )
    console.print(table)

    console.print("\n[bold]Summary[/bold]")
    console.print(f"  Full-window accuracy:  {full_acc_tot:.1%} "
                  f"({total_full_correct}/{total_full_correct + total_full_wrong})")
    console.print(f"  Early-window accuracy: {early_acc_tot:.1%} "
                  f"({total_early_correct}/{total_early_correct + total_early_wrong})")
    console.print(f"  Delta: {(early_acc_tot - full_acc_tot):+.1%}")

    # ------------------------------------------------------------------
    # Pass 4: In-code branch bucketing + wrong-return Y-gap histogram.
    # Mirror the 4 None-return paths in team_identity.py:151-178 and
    # classify every GT rally.
    # ------------------------------------------------------------------
    console.print(
        "\n[bold]Pass 4: None-return branch bucketing + "
        "wrong-return Y-gap histogram[/bold]",
    )

    records: list[dict[str, Any]] = []
    for p4_vid, p4_rallies in sorted(video_rallies.items()):
        p4_templates = templates_by_vid.get(p4_vid)
        if p4_templates is None:
            continue  # Function can't be called without templates.
        p4_label_a: str | None = vid_label_a.get(p4_vid)
        p4_label_b: str | None = None
        if p4_label_a is not None:
            p4_t0, p4_t1 = p4_templates
            p4_label_b = (
                p4_t0.team_label
                if p4_label_a == p4_t1.team_label
                else p4_t1.team_label
            )

        for p4_rally in p4_rallies:
            if p4_rally.gt_serving_team is None:
                continue
            p4_positions = _parse_positions(p4_rally.positions)
            p4_t2p = t2p_by_rally.get(p4_rally.rally_id, {})

            dbg = _localize_team_near_debug(
                p4_positions, p4_t2p, p4_templates,
            )

            p4_expected_tn: str | None = None
            p4_formation_side: str | None = None
            if p4_label_a is not None and p4_label_b is not None:
                net_y = (
                    p4_rally.court_split_y if p4_rally.court_split_y else 0.5
                )
                p4_formation_side, _ = _find_serving_side_by_formation(
                    p4_positions, net_y=net_y, start_frame=0,
                )
                if p4_formation_side is not None:
                    if p4_formation_side == "near":
                        p4_expected_tn = (
                            p4_label_a
                            if p4_rally.gt_serving_team == "A"
                            else p4_label_b
                        )
                    else:
                        p4_expected_tn = (
                            p4_label_b
                            if p4_rally.gt_serving_team == "A"
                            else p4_label_a
                        )

            records.append({
                "vid": p4_vid, "rally_id": p4_rally.rally_id,
                "reason": dbg["reason"], "result": dbg["result"],
                "y_gap": dbg["y_gap"],
                "t0_n": dbg["t0_n"], "t1_n": dbg["t1_n"],
                "n_pid_mapped": dbg["n_pid_mapped"],
                "n_positions": dbg["n_positions"], "n_t2p": dbg["n_t2p"],
                "expected_tn": p4_expected_tn,
                "formation_side": p4_formation_side,
            })

    total_records = len(records)
    reason_counts: Counter = Counter(r["reason"] for r in records)
    none_total = sum(reason_counts[r] for r in NONE_REASONS)

    console.print(f"\nTotal GT rallies with templates: {total_records}")
    console.print(
        f"  ok (returned):       {reason_counts[REASON_OK]} "
        f"({reason_counts[REASON_OK] / max(1, total_records):.1%})",
    )
    console.print(
        f"  None (any reason):   {none_total} "
        f"({none_total / max(1, total_records):.1%})",
    )

    # Table A: global None bucket distribution.
    table_a = Table(title="Table A: None-return bucket distribution (global)")
    table_a.add_column("Reason", style="cyan")
    table_a.add_column("Count", justify="right")
    table_a.add_column("% of None", justify="right")
    table_a.add_column("% of total", justify="right")
    for reason in NONE_REASONS:
        c = reason_counts[reason]
        table_a.add_row(
            reason, str(c),
            f"{c / max(1, none_total):.1%}",
            f"{c / max(1, total_records):.1%}",
        )
    table_a.add_row(
        "ALL None", str(none_total), "100%",
        f"{none_total / max(1, total_records):.1%}",
        style="bold",
    )
    console.print(table_a)

    # Table B: wrong-return Y-gap histogram.
    bin_edges = [0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, float("inf")]

    def _bin_label(lo: float, hi: float) -> str:
        hi_str = "inf" if hi == float("inf") else f"{hi:.3f}"
        return f"[{lo:.3f}, {hi_str})"

    scored = [r for r in records
              if r["result"] is not None and r["expected_tn"] is not None]
    scored_wrong = [r for r in scored if r["result"] != r["expected_tn"]]
    scored_correct = [r for r in scored if r["result"] == r["expected_tn"]]

    console.print(f"\nScored (result + expected_tn available): {len(scored)}")
    console.print(
        f"  Correct: {len(scored_correct)} "
        f"({len(scored_correct) / max(1, len(scored)):.1%})",
    )
    console.print(
        f"  Wrong:   {len(scored_wrong)} "
        f"({len(scored_wrong) / max(1, len(scored)):.1%})",
    )

    table_b = Table(title="Table B: Wrong-return Y-gap histogram")
    table_b.add_column("Y-gap bin", style="cyan")
    table_b.add_column("Correct", justify="right")
    table_b.add_column("Wrong", justify="right")
    table_b.add_column("Wrong-rate", justify="right")
    table_b.add_column("% of wrong", justify="right")
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        c = sum(1 for r in scored_correct
                if r["y_gap"] is not None and lo <= r["y_gap"] < hi)
        w = sum(1 for r in scored_wrong
                if r["y_gap"] is not None and lo <= r["y_gap"] < hi)
        tot = c + w
        wr = w / tot if tot else 0.0
        w_share = w / max(1, len(scored_wrong))
        table_b.add_row(
            _bin_label(lo, hi), str(c), str(w),
            f"{wr:.1%}" if tot else "-",
            f"{w_share:.1%}",
        )
    console.print(table_b)

    # Decision signal: boundary vs tail wrong-rate.
    bd_c = sum(1 for r in scored_correct
               if r["y_gap"] is not None and 0.03 <= r["y_gap"] < 0.05)
    bd_w = sum(1 for r in scored_wrong
               if r["y_gap"] is not None and 0.03 <= r["y_gap"] < 0.05)
    tl_c = sum(1 for r in scored_correct
               if r["y_gap"] is not None and r["y_gap"] >= 0.10)
    tl_w = sum(1 for r in scored_wrong
               if r["y_gap"] is not None and r["y_gap"] >= 0.10)
    bd_rate = bd_w / max(1, bd_c + bd_w)
    tl_rate = tl_w / max(1, tl_c + tl_w)
    console.print(
        f"\n  Wrong-rate at boundary [0.03, 0.05): {bd_rate:.1%} "
        f"({bd_w}/{bd_c + bd_w})",
    )
    console.print(
        f"  Wrong-rate at tail     [0.10, inf):  {tl_rate:.1%} "
        f"({tl_w}/{tl_c + tl_w})",
    )
    ratio_str = f"{bd_rate / tl_rate:.2f}x" if tl_rate > 0 else "inf"
    console.print(f"  Ratio (boundary / tail): {ratio_str}")

    # Table C: dominant None bucket sub-split.
    dominant = max(NONE_REASONS, key=lambda r: reason_counts[r])
    dominant_records = [r for r in records if r["reason"] == dominant]
    dom_count = len(dominant_records)
    console.print(
        f"\n[bold]Table C: Sub-split of dominant None bucket "
        f"[cyan]{dominant}[/cyan] (n={dom_count})[/bold]",
    )

    if dominant == REASON_Y_GAP_BELOW:
        sub_edges = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03]
        table_c = Table(title="Table C: y_gap_below_threshold Y-gap histogram")
        table_c.add_column("Y-gap bin", style="cyan")
        table_c.add_column("Count", justify="right")
        table_c.add_column("% of dominant", justify="right")
        for i in range(len(sub_edges) - 1):
            lo, hi = sub_edges[i], sub_edges[i + 1]
            c = sum(1 for r in dominant_records
                    if r["y_gap"] is not None and lo <= r["y_gap"] < hi)
            table_c.add_row(
                f"[{lo:.3f}, {hi:.3f})", str(c),
                f"{c / max(1, dom_count):.1%}",
            )
        console.print(table_c)
    elif dominant == REASON_ONE_TEAM_UNMAPPED:
        pair_counts: Counter = Counter(
            (r["t0_n"], r["t1_n"]) for r in dominant_records
        )
        table_c = Table(title="Table C: one_team_unmapped (t0_n, t1_n)")
        table_c.add_column("(t0_n, t1_n)", style="cyan")
        table_c.add_column("Count", justify="right")
        table_c.add_column("%", justify="right")
        for (a, b), c in pair_counts.most_common():
            table_c.add_row(
                f"({a}, {b})", str(c), f"{c / max(1, dom_count):.1%}",
            )
        console.print(table_c)
        t2p_counts: Counter = Counter(r["n_t2p"] for r in dominant_records)
        console.print(
            f"  n_t2p distribution: {dict(sorted(t2p_counts.items()))}",
        )
        n_pid_counts: Counter = Counter(
            r["n_pid_mapped"] for r in dominant_records
        )
        console.print(
            "  n_pid_mapped distribution: "
            f"{dict(sorted(n_pid_counts.items()))}",
        )
    elif dominant == REASON_NO_PID_MAPPED:
        t2p_counts = Counter(r["n_t2p"] for r in dominant_records)
        pos_counts: Counter = Counter(
            r["n_positions"] for r in dominant_records
        )
        table_c = Table(title="Table C: no_pid_mapped n_t2p distribution")
        table_c.add_column("n_t2p", style="cyan")
        table_c.add_column("Count", justify="right")
        for k, c in sorted(t2p_counts.items()):
            table_c.add_row(str(k), str(c))
        console.print(table_c)
        if pos_counts:
            console.print(
                f"  n_positions: min={min(pos_counts)} "
                f"max={max(pos_counts)} zeros={pos_counts.get(0, 0)}",
            )
    elif dominant == REASON_EMPTY_INPUT:
        empty_pos = sum(1 for r in dominant_records if r["n_positions"] == 0)
        empty_t2p = sum(1 for r in dominant_records if r["n_t2p"] == 0)
        both_empty = sum(
            1 for r in dominant_records
            if r["n_positions"] == 0 and r["n_t2p"] == 0
        )
        console.print(
            f"  positions empty: {empty_pos} "
            f"({empty_pos / max(1, dom_count):.1%})",
        )
        console.print(
            f"  t2p empty:       {empty_t2p} "
            f"({empty_t2p / max(1, dom_count):.1%})",
        )
        console.print(
            f"  both empty:      {both_empty} "
            f"({both_empty / max(1, dom_count):.1%})",
        )

    # Table D: per-video None-rate + dominant None bucket.
    by_vid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_vid[r["vid"]].append(r)

    def _none_rate(vid_records: list[dict[str, Any]]) -> float:
        if not vid_records:
            return 0.0
        n_none = sum(1 for r in vid_records if r["reason"] in NONE_REASONS)
        return n_none / len(vid_records)

    table_d = Table(title="Table D: Per-video None-rate + dominant None bucket")
    table_d.add_column("Video", style="cyan")
    table_d.add_column("N", justify="right")
    table_d.add_column("None%", justify="right")
    table_d.add_column("Top None bucket", style="yellow")
    table_d.add_column("Top %", justify="right")

    for vid, vid_records in sorted(
        by_vid.items(), key=lambda kv: -_none_rate(kv[1]),
    ):
        n = len(vid_records)
        vid_counts = Counter(r["reason"] for r in vid_records)
        vid_none = sum(vid_counts[r] for r in NONE_REASONS)
        if vid_none > 0:
            top_reason = max(
                NONE_REASONS, key=lambda k: vid_counts[k],
            )
            top_share = vid_counts[top_reason] / vid_none
            top_share_str = f"{top_share:.0%}"
        else:
            top_reason, top_share_str = "-", "-"
        table_d.add_row(
            vid[:10], str(n),
            f"{vid_none / max(1, n):.0%}",
            top_reason, top_share_str,
        )
    console.print(table_d)

    # Invariants + cross-checks.
    assert sum(reason_counts.values()) == total_records, (
        f"Reason counts {sum(reason_counts.values())} != total {total_records}"
    )
    marg_wrong = sum(
        1 for r in scored_wrong
        if r["y_gap"] is not None and r["y_gap"] < 0.05
    )
    console.print("\n[bold]Cross-checks[/bold]")
    console.print(
        f"  Invariant OK: sum(reason_counts) = total_records = {total_records}",
    )
    console.print(
        f"  Wrong-return y_gap<0.05: {marg_wrong} / {len(scored_wrong)} "
        f"= {marg_wrong / max(1, len(scored_wrong)):.0%} "
        f"(diagnose_phantom_flips.py reported 23% marginal_y_gap)",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
