"""Dump baseline + MS-TCN-off action confusion matrices.

Mirrors `production_eval.py` setup, runs `_run_once` twice with the same rally
list/team_map/calibrators, then aggregates `MatchResult.gt_action ×
pred_action` into 6x6 confusion matrices.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))

from eval_action_detection import (  # noqa: E402
    _load_match_team_assignments,
    load_rallies_with_action_gt,
)
from production_eval import (  # noqa: E402
    PipelineContext,
    _build_calibrators,
    _parse_positions,
    _run_once,
)

CLASSES = ["serve", "receive", "set", "attack", "block", "dig"]


def confusion(matches):
    """rows=GT class, cols=pred class (or 'unmatched'/'other'). Returns Counter."""
    c: Counter = Counter()
    for m in matches:
        if m.gt_action not in CLASSES:
            continue
        pred = m.pred_action if m.pred_action in CLASSES else ("unmatched" if m.pred_action is None else "other")
        c[(m.gt_action, pred)] += 1
    return c


def render(c: Counter, label: str) -> str:
    cols = CLASSES + ["unmatched", "other"]
    lines = [f"\n=== {label} ===", f"{'gt\\pred':>10}  " + "  ".join(f"{x:>9}" for x in cols) + "  " + f"{'total':>6}"]
    for gt in CLASSES:
        row = [c[(gt, p)] for p in cols]
        total = sum(row)
        lines.append(f"{gt:>10}  " + "  ".join(f"{v:>9}" for v in row) + "  " + f"{total:>6}")
    return "\n".join(lines)


def per_class_f1(c: Counter) -> dict[str, float]:
    out = {}
    for cls in CLASSES:
        tp = c[(cls, cls)]
        fn = sum(c[(cls, p)] for p in (CLASSES + ["unmatched", "other"]) if p != cls)
        fp = sum(c[(g, cls)] for g in CLASSES if g != cls)
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        out[cls] = 2 * prec * rec / max(1e-9, prec + rec)
    return out


def main() -> int:
    print("Loading rallies with action GT...")
    rallies = load_rallies_with_action_gt()
    print(f"  {len(rallies)} rallies")

    rally_pos_lookup = {
        r.rally_id: _parse_positions(r.positions_json)
        for r in rallies if r.positions_json
    }
    video_ids = {r.video_id for r in rallies}
    team_map = _load_match_team_assignments(video_ids, rally_positions=rally_pos_lookup)
    calibrators = _build_calibrators(video_ids)
    print(f"  team_map={len(team_map)} calibrators={len(calibrators)}/{len(video_ids)}")

    print("\nRun A: baseline (MS-TCN++ override ON)")
    matches_a, _, rej_a = _run_once(rallies, team_map, calibrators, PipelineContext(), print_progress=False)
    print(f"  matches={len(matches_a)} rejections={len(rej_a)}")

    print("\nRun B: ablation (MS-TCN++ override OFF)")
    ctx_b = PipelineContext(skip_sequence_override=True)
    matches_b, _, rej_b = _run_once(rallies, team_map, calibrators, ctx_b, print_progress=False)
    print(f"  matches={len(matches_b)} rejections={len(rej_b)}")

    cm_a = confusion(matches_a)
    cm_b = confusion(matches_b)

    print(render(cm_a, "BASELINE (MS-TCN++ ON)"))
    print(render(cm_b, "ABLATION (MS-TCN++ OFF)"))

    f1_a = per_class_f1(cm_a)
    f1_b = per_class_f1(cm_b)
    print("\n=== per-class F1 ===")
    print(f"{'class':>10}  {'baseline':>10}  {'ablation':>10}  {'delta_pp':>10}")
    for cls in CLASSES:
        d = (f1_b[cls] - f1_a[cls]) * 100
        print(f"{cls:>10}  {f1_a[cls]*100:>9.1f}%  {f1_b[cls]*100:>9.1f}%  {d:>+9.1f}")

    # Dig row diff
    print("\n=== dig row delta (ablation - baseline) ===")
    for p in CLASSES + ["unmatched", "other"]:
        d = cm_b[("dig", p)] - cm_a[("dig", p)]
        if d != 0:
            print(f"  dig -> {p:>10}: {cm_a[('dig', p)]:>4} -> {cm_b[('dig', p)]:>4}  ({d:+d})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
