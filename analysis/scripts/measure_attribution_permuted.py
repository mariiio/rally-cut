"""Permutation-invariant player-attribution measurement.

For each Phase-0 fixture:
  1. Compute the optimal matcher_pid → gt_pid permutation per video using
     the tracking GT (same approach as measure_pid_accuracy.py).
  2. Apply that permutation to ALL pred actions' player_track_id in that
     video's rallies.
  3. Compare permuted pred actions to action-GT (which uses gt trackId).
  4. Report correct / wrong_cross_team / wrong_same_team / missing.

This is the honest attribution number — permutation-invariant, so canonical
label shifts don't contaminate it.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

from rallycut.evaluation.db import get_connection

FIXTURE_MAP_PATH = (
    Path(__file__).resolve().parent.parent
    / "reports" / "attribution_rebuild" / "fixture_video_ids_2026_04_24.json"
)
MATCH_TOLERANCE_FRAMES = 10


def _compute_permutation(
    video_id: str,
) -> dict[int, int] | None:
    """Compute optimal matcher_pid → gt_pid permutation for a video by
    delegating to measure_pid_accuracy.py and parsing its output.
    """
    import re
    import subprocess
    import os
    result = subprocess.run(
        ["python", "-u", "scripts/measure_pid_accuracy.py", video_id],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        cwd=Path(__file__).resolve().parent.parent,
    )
    if result.returncode != 0:
        return None
    # Parse "Optimal permutation matcher_pid → gt_pid: {1: 1, 2: 4, 3: 3, 4: 2}"
    m = re.search(r"Optimal permutation [^:]+:\s*(\{[^}]+\})", result.stdout)
    if not m:
        return None
    raw = m.group(1)
    # Eval as dict literal — safe enough since we control the source
    try:
        perm = {int(k): int(v) for k, v in
                (item.split(":") for item in raw.strip("{}").split(","))}
    except Exception:
        return None
    return perm or None
    matcher_pids = sorted({m for m, _ in samples})
    gt_pids = sorted({g for _, g in samples})
    if not matcher_pids or not gt_pids:
        return None
    best_perm: dict[int, int] = {}
    best_score = -1
    for perm in permutations(gt_pids, len(matcher_pids)):
        mapping = dict(zip(matcher_pids, perm))
        score = sum(1 for m, g in samples if mapping.get(m) == g)
        if score > best_score:
            best_score = score
            best_perm = mapping
    return best_perm


def main() -> None:
    fixture_map = json.loads(FIXTURE_MAP_PATH.read_text())["fixtures"]
    print(f"{'fixture':<8} {'n_gt':>5} {'correct':>9} {'wrong_x':>9} "
          f"{'wrong_s':>9} {'missing':>9} {'unknown':>9} {'perm':<22}")
    print("-" * 100)

    totals = defaultdict(int)
    rows_out: list[dict[str, Any]] = []

    with get_connection() as conn:
        for fixture_name, finfo in fixture_map.items():
            video_id = finfo["video_id"]
            perm = _compute_permutation(video_id)
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT pt.action_ground_truth_json, pt.actions_json
                       FROM rallies r JOIN player_tracks pt ON pt.rally_id = r.id
                       WHERE r.video_id = %s
                         AND pt.action_ground_truth_json IS NOT NULL
                         AND jsonb_array_length(pt.action_ground_truth_json::jsonb) > 0
                       ORDER BY r.start_ms""",
                    (video_id,),
                )
                rally_rows = cur.fetchall()
            fix_counts = defaultdict(int)
            for gt_actions, aj in rally_rows:
                pred_actions = (aj or {}).get("actions") or []
                team_assignments = (aj or {}).get("teamAssignments") or {}
                # Map each GT action to nearest pred within ±MATCH_TOLERANCE.
                used: set[int] = set()
                for g in gt_actions or []:
                    gf = int(g.get("frame", 0))
                    gt_pid = g.get("trackId")
                    fix_counts["n_gt"] += 1
                    if gt_pid is None:
                        fix_counts["missing"] += 1
                        continue
                    candidates = [
                        (i, p) for i, p in enumerate(pred_actions)
                        if i not in used
                        and abs(int(p.get("frame", 0)) - gf) <= MATCH_TOLERANCE_FRAMES
                    ]
                    if not candidates:
                        fix_counts["missing"] += 1
                        continue
                    candidates.sort(key=lambda c: (
                        0 if c[1].get("action") == g.get("action") else 1,
                        abs(int(c[1].get("frame", 0)) - gf),
                    ))
                    used.add(candidates[0][0])
                    pred = candidates[0][1]
                    pl_pid_raw = pred.get("playerTrackId")
                    if pl_pid_raw is None:
                        fix_counts["unknown"] += 1
                        continue
                    pl_pid = int(pl_pid_raw)
                    # Apply permutation
                    permuted = perm.get(pl_pid, pl_pid) if perm else pl_pid
                    if int(permuted) == int(gt_pid):
                        fix_counts["correct"] += 1
                    else:
                        gt_team = team_assignments.get(str(gt_pid))
                        pl_team = team_assignments.get(str(pl_pid))
                        if gt_team is None or pl_team is None:
                            fix_counts["wrong_unknown"] += 1
                        elif gt_team == pl_team:
                            fix_counts["wrong_same"] += 1
                        else:
                            fix_counts["wrong_cross"] += 1

            n_gt = fix_counts["n_gt"]
            perm_str = "no tracking GT" if perm is None else str(perm)
            print(f"{fixture_name:<8} {n_gt:>5} "
                  f"{fix_counts['correct']:>3} ({fix_counts['correct']/max(1,n_gt):>5.1%}) "
                  f"{fix_counts['wrong_cross']:>3} ({fix_counts['wrong_cross']/max(1,n_gt):>5.1%}) "
                  f"{fix_counts['wrong_same']:>3} ({fix_counts['wrong_same']/max(1,n_gt):>5.1%}) "
                  f"{fix_counts['missing']:>3} ({fix_counts['missing']/max(1,n_gt):>5.1%}) "
                  f"{fix_counts['wrong_unknown']:>3}       "
                  f"{perm_str[:22]:<22}")
            for k, v in fix_counts.items():
                totals[k] += v
            rows_out.append({"fixture": fixture_name, **fix_counts, "perm": perm})

    print("-" * 100)
    n = totals["n_gt"]
    print(f"{'TOTAL':<8} {n:>5} "
          f"{totals['correct']:>3} ({totals['correct']/max(1,n):>5.1%}) "
          f"{totals['wrong_cross']:>3} ({totals['wrong_cross']/max(1,n):>5.1%}) "
          f"{totals['wrong_same']:>3} ({totals['wrong_same']/max(1,n):>5.1%}) "
          f"{totals['missing']:>3} ({totals['missing']/max(1,n):>5.1%}) "
          f"{totals['wrong_unknown']:>3}")
    print()
    print(f"=== AGGREGATE PERMUTED ATTRIBUTION (n={n}) ===")
    print(f"  correct:       {totals['correct']:>4} ({totals['correct']/max(1,n):>5.1%})")
    print(f"  wrong_cross_team: {totals['wrong_cross']:>4} ({totals['wrong_cross']/max(1,n):>5.1%})")
    print(f"  wrong_same_team:  {totals['wrong_same']:>4} ({totals['wrong_same']/max(1,n):>5.1%})")
    print(f"  missing:       {totals['missing']:>4} ({totals['missing']/max(1,n):>5.1%})")
    print(f"  unknown_team:  {totals['wrong_unknown']:>4} ({totals['wrong_unknown']/max(1,n):>5.1%})")
    print()
    print("Reference: 2026-04-15 baseline player_attr = 57.66% (DIRECT, no permutation).")
    print("Reference: 2026-04-24 attribution_bench baseline correct = 43.8% (DIRECT).")


if __name__ == "__main__":
    main()
