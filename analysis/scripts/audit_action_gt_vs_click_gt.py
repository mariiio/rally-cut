"""Cross-check action-GT vs click-GT on overlapping (rally, frame) pairs.

Both GT sources label the same physical contacts but were collected
independently. action-GT lives in rally_action_ground_truth and its playerTrackId
reflects whatever canonical convention was live in the editor at label time.
click-GT lives in reports/session3/verdicts/ and its actor_pid reflects the
user clicking on a physical player position.

If the canonical convention has drifted (e.g. via a `match-players` re-run
that changed the ttp permutation without remapping the GT), the two sources
disagree on the same physical event. This script measures that disagreement
across the 5 fixtures that have both, breaking it down per rally.

A consistent per-rally permutation in the disagreement (e.g. rally X always
maps action_gt.pid 1 → click_gt.pid 3) confirms drift, and the recovered
permutation is the patch the editor or a remap script needs to apply.

Usage:
    cd analysis
    uv run python scripts/audit_action_gt_vs_click_gt.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_ANALYSIS_DIR = Path(__file__).resolve().parents[1]
if str(_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS_DIR))

from rallycut.evaluation.db import get_connection  # noqa: E402
from rallycut.training.action_gt_query import load_for_videos  # noqa: E402

REPO_ROOT = _ANALYSIS_DIR.parent
VERDICTS_DIR = REPO_ROOT / "reports" / "session3" / "verdicts"
TOL_FRAMES = 10

# 5 fixtures present in both 9-fixture baseline AND day-4 click-GT.
FIXTURES = {
    "tata": "7d77980f-3006-40e0-adc0-db491a5bb659",
    "rere": "808a5618-a5dc-4d36-93a8-c502c3eb53c5",
    "cece": "950fbe5d-fdad-4862-b05d-8b374bdd5ec6",
    "yeye": "eb693a6f-30df-4265-9291-eea46b6a1426",
    "lulu": "4f2bd66a-61a1-49ac-8137-fd2576e0e851",
}
VSHORT_BY_FIXTURE = {
    "tata": "7d77980f", "rere": "808a5618", "cece": "950fbe5d",
    "yeye": "eb693a6f", "lulu": "4f2bd66a",
}


def load_click_gt(verdicts_path: Path) -> dict[str, dict[int, int]]:
    raw = json.loads(verdicts_path.read_text())
    out: dict[str, dict[int, int]] = defaultdict(dict)
    for key, v in raw.get("verdicts", {}).items():
        rally_short, _, frame_str = key.rpartition("-")
        if not frame_str.isdigit():
            continue
        actor_pid = v.get("actor_pid")
        if actor_pid is None:
            continue
        out[rally_short][int(frame_str)] = int(actor_pid)
    return dict(out)


def load_action_gt(video_id: str) -> dict[str, list[dict[str, Any]]]:
    with get_connection() as conn:
        gt_by_rally = load_for_videos(conn, [video_id], include_unresolved=True)
    # Key by 8-char prefix to match click-GT short IDs.
    out: dict[str, list[dict[str, Any]]] = {}
    for rid, labels in gt_by_rally.items():
        out[rid[:8]] = labels
    return out


def find_verdicts(vshort: str) -> Path | None:
    matches = sorted(VERDICTS_DIR.glob(f"verdicts_{vshort}_*.json"))
    return matches[-1] if matches else None


def audit_fixture(fixture: str, video_id: str) -> dict[str, Any]:
    print(f"\n=== {fixture} ===", flush=True)
    vshort = VSHORT_BY_FIXTURE[fixture]
    verdicts_path = find_verdicts(vshort)
    if verdicts_path is None:
        print("  no click-GT — skip")
        return {"fixture": fixture, "skipped": "no_click_gt"}
    click = load_click_gt(verdicts_path)
    action = load_action_gt(video_id)

    n_overlap = 0
    n_agree = 0
    n_disagree = 0
    per_rally_perm: dict[str, dict[tuple[int, int], int]] = defaultdict(Counter)

    for rally_short, click_frames in click.items():
        a_list = action.get(rally_short, [])
        if not a_list:
            continue
        for click_frame, click_pid in click_frames.items():
            best = None
            best_d = TOL_FRAMES + 1
            for a in a_list:
                af = int(a.get("frame", -9999))
                d = abs(af - click_frame)
                if d < best_d:
                    best_d = d
                    best = a
            if best is None:
                continue
            a_pid = best.get("playerTrackId")
            if a_pid is None:
                continue
            n_overlap += 1
            if int(a_pid) == int(click_pid):
                n_agree += 1
            else:
                n_disagree += 1
            per_rally_perm[rally_short][(int(a_pid), int(click_pid))] += 1

    agree_rate = n_agree / n_overlap if n_overlap else 0.0
    print(f"  overlap={n_overlap}  agree={n_agree} ({agree_rate:.1%})  "
          f"disagree={n_disagree}")
    print(f"  per-rally action_gt → click_gt mapping (only differing pairs):")
    consistent_perm = True
    fixture_perm: dict[int, set[int]] = defaultdict(set)
    for rally_short, mapping in per_rally_perm.items():
        diffs = {k: v for k, v in mapping.items() if k[0] != k[1]}
        if not diffs:
            continue
        # Build per-rally permutation: action_gt.pid → click_gt.pid
        rally_perm: dict[int, set[int]] = defaultdict(set)
        for (a_pid, c_pid), n in mapping.items():
            rally_perm[a_pid].add(c_pid)
            fixture_perm[a_pid].add(c_pid)
        ambig = {k: v for k, v in rally_perm.items() if len(v) > 1}
        marker = "" if not ambig else "  [AMBIGUOUS]"
        perm_str = ",".join(
            f"{a}→{sorted(s)[0] if len(s)==1 else sorted(s)}"
            for a, s in sorted(rally_perm.items())
        )
        print(f"    {rally_short}: {perm_str}{marker}")
    fixture_ambig = {k: sorted(v) for k, v in fixture_perm.items() if len(v) > 1}
    if fixture_ambig:
        consistent_perm = False
        print(f"  fixture-level perm AMBIGUOUS: {fixture_ambig}")
    else:
        unambig = {k: sorted(v)[0] for k, v in fixture_perm.items() if len(v) == 1}
        if unambig:
            print(f"  fixture-level recovered perm: {unambig}")

    return {
        "fixture": fixture,
        "n_overlap": n_overlap,
        "n_agree": n_agree,
        "n_disagree": n_disagree,
        "agree_rate": agree_rate,
        "consistent_per_fixture_perm": consistent_perm,
        "fixture_perm": {
            str(k): sorted(v) for k, v in fixture_perm.items()
        },
    }


def main() -> int:
    summaries = []
    for fx, vid in FIXTURES.items():
        try:
            summaries.append(audit_fixture(fx, vid))
        except Exception as e:
            import traceback
            traceback.print_exc()
            summaries.append({"fixture": fx, "error": str(e)})

    print("\n" + "=" * 70)
    print("SUMMARY — action-GT vs click-GT agreement on overlapping events")
    print("=" * 70)
    print(f"{'fixture':8s}  {'overlap':>7s}  {'agree':>7s}  {'rate':>7s}  perm-consistent")
    n_t = a_t = 0
    for s in summaries:
        if "skipped" in s or "error" in s:
            print(f"{s['fixture']:8s}  {s.get('skipped', s.get('error'))}")
            continue
        n_t += s["n_overlap"]
        a_t += s["n_agree"]
        marker = "yes" if s.get("consistent_per_fixture_perm") else "NO"
        print(f"{s['fixture']:8s}  {s['n_overlap']:>7d}  {s['n_agree']:>7d}  "
              f"{s['agree_rate']:>6.1%}  {marker}")
    if n_t:
        print(f"{'AGG':8s}  {n_t:>7d}  {a_t:>7d}  {a_t / n_t:>6.1%}")

    out = _ANALYSIS_DIR / "reports" / "attribution_rebuild" / "audit_action_gt_vs_click_gt.json"
    out.write_text(json.dumps({"fixtures": summaries}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
