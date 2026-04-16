"""Probe: does the existing tracklet_link velocity gate cover our swap events?

`tracklet_link._would_create_velocity_anomaly` is already called in
`_greedy_merge` and `_swap_optimize` (the appearance-linking paths). We want to
know whether, if the same gate were applied to the pred_old / pred_new pair at
every pred-exchange swap we've identified, it would have REJECTED the junction
or ALLOWED it.

  gate_would_reject   endpoint or window pair displacement > MAX_MERGE_VELOCITY
                      → these two tracks already look kinematically separate;
                        a velocity check on whichever stage produced this swap
                        would have prevented it. Fix: add the gate to the
                        missing call site (likely `global_identity` or
                        `stabilize_track_ids`, neither of which consults it).
  gate_would_allow    junction displacement is small (< threshold)
                      → velocity alone is insufficient at the convergence
                        events where swaps actually happen. Fix: need a
                        complementary signal (appearance+velocity, pose, role).
  no_data             insufficient positions for pred_old or pred_new in-rally.

Usage:
    uv run python scripts/check_velocity_gate_on_swaps.py
    uv run python scripts/check_velocity_gate_on_swaps.py --all-swap-rallies
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

from rallycut.evaluation.tracking.db import load_labeled_rallies
from rallycut.tracking.tracklet_link import (
    DEFAULT_MAX_MERGE_VELOCITY,
    DEFAULT_MERGE_VELOCITY_WINDOW,
    _would_create_velocity_anomaly,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("velocity-gate-probe")

DEFAULT_RALLIES = [
    "fad29c31-6e2a-4a8d-86f1-9064b2f1f425",
    "209be896-b680-44dc-bf31-693f4e287149",
    "d724bbf0-bd0c-44e8-93d5-135aa07df5a1",
]


def _load_swap_events(audit_path: Path) -> list[dict]:
    audit = json.loads(audit_path.read_text())
    pred_history: dict[int, list[tuple[int, int, int]]] = {}
    for g in audit.get("perGt", []):
        for s, e, pid in g.get("predIdSpans", []):
            pred_history.setdefault(pid, []).append((g["gtTrackId"], s, e))
    for h in pred_history.values():
        h.sort(key=lambda t: t[1])

    def prior_gt_of(pid: int, before: int) -> int | None:
        last_gt: int | None = None
        for gt_id, s, _e in pred_history.get(pid, []):
            if s >= before:
                break
            last_gt = gt_id
        return last_gt

    events = []
    for g in audit.get("perGt", []):
        spans = g.get("predIdSpans", [])
        for prev, cur in zip(spans, spans[1:]):
            _, prev_end, prev_pred = prev
            cur_start, _, cur_pred = cur
            if prev_pred == cur_pred or prev_pred < 0 or cur_pred < 0:
                continue
            incoming = prior_gt_of(cur_pred, cur_start)
            if incoming is None or incoming == g["gtTrackId"]:
                continue
            events.append({
                "rally_id": audit["rallyId"],
                "swap_frame": cur_start,
                "gt_track_id": g["gtTrackId"],
                "pred_old": prev_pred,
                "pred_new": cur_pred,
            })
    return events


def run_for_rally(rally_id: str, audit_dir: Path) -> list[dict]:
    audit_path = audit_dir / f"{rally_id}.json"
    if not audit_path.exists():
        return []
    events = _load_swap_events(audit_path)
    if not events:
        return []
    rallies = load_labeled_rallies(rally_id=rally_id)
    if not rallies or rallies[0].predictions is None:
        return []
    predictions = rallies[0].predictions.positions

    results = []
    for ev in events:
        # Both pred tracks must have at least one position in the rally for the
        # gate function to do anything meaningful.
        has_old = any(p.track_id == ev["pred_old"] for p in predictions)
        has_new = any(p.track_id == ev["pred_new"] for p in predictions)
        if not (has_old and has_new):
            verdict = "no_data"
        else:
            anomaly = _would_create_velocity_anomaly(
                positions=predictions,
                tid_a=ev["pred_old"],
                tid_b=ev["pred_new"],
                max_displacement=DEFAULT_MAX_MERGE_VELOCITY,
                window=DEFAULT_MERGE_VELOCITY_WINDOW,
            )
            verdict = "gate_would_reject" if anomaly else "gate_would_allow"

        results.append({
            "rally_id": rally_id,
            "swap_frame": ev["swap_frame"],
            "pred_old": ev["pred_old"],
            "pred_new": ev["pred_new"],
            "gt_track_id": ev["gt_track_id"],
            "verdict": verdict,
        })
        logger.info(
            f"  swap@{ev['swap_frame']} pred {ev['pred_old']}→{ev['pred_new']} on GT {ev['gt_track_id']}: {verdict}"
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=Path("reports/tracking_audit"))
    parser.add_argument("--out", type=Path, default=Path("reports/tracking_audit/reid_debug/velocity_gate_coverage.md"))
    parser.add_argument("--rally", type=str, default=None)
    parser.add_argument("--all-swap-rallies", action="store_true")
    args = parser.parse_args()

    if args.rally:
        rally_ids = [args.rally]
    elif args.all_swap_rallies:
        rally_ids = []
        for p in sorted(args.audit_dir.glob("*.json")):
            if p.name == "_summary.json":
                continue
            if _load_swap_events(p):
                rally_ids.append(json.loads(p.read_text())["rallyId"])
    else:
        rally_ids = DEFAULT_RALLIES

    all_results: list[dict] = []
    for idx, rid in enumerate(rally_ids, start=1):
        logger.info(f"[{idx}/{len(rally_ids)}] {rid[:8]}")
        all_results.extend(run_for_rally(rid, audit_dir=args.audit_dir))

    counts = Counter(r["verdict"] for r in all_results)
    total = len(all_results)

    lines = [
        "# Velocity-gate coverage at swap events",
        "",
        f"Probed **{total}** pred-exchange swap events across {len(rally_ids)} rally(s) "
        f"using `_would_create_velocity_anomaly` with threshold="
        f"{DEFAULT_MAX_MERGE_VELOCITY} and window={DEFAULT_MERGE_VELOCITY_WINDOW}.",
        "",
        "## Verdict counts",
        "",
        "| Verdict | Count | Share | What it tells us |",
        "|---|---:|---:|---|",
    ]
    descriptions = {
        "gate_would_reject":
            "Fix = add the existing gate to the missing call site (likely "
            "`global_identity` or `stabilize_track_ids`, neither currently consults it).",
        "gate_would_allow":
            "Fix = need a different signal; velocity alone is insufficient at the "
            "convergence events where swaps occur. Options: combine with appearance, "
            "use trajectory extrapolation (not endpoint displacement), or a learned "
            "within-team ReID head.",
        "no_data":
            "pred_old or pred_new has no positions in the rally — probably a mid-rally "
            "track not linked to a primary. Excluded from the actionable tally.",
    }
    for v, desc in descriptions.items():
        n = counts.get(v, 0)
        pct = f"{100 * n / total:.1f}%" if total else "0.0%"
        lines.append(f"| `{v}` | {n} | {pct} | {desc} |")

    lines.extend(["", "## Interpretation", ""])
    reject = counts.get("gate_would_reject", 0)
    allow = counts.get("gate_would_allow", 0)
    if reject > allow and reject >= 5:
        lines.append(
            f"- **Dominant: `gate_would_reject` ({reject}/{total}).** "
            f"The existing velocity gate has the signal. Fix = extend its coverage "
            f"to the missing pipeline stage. Next step: identify the stage that "
            f"introduces the swap (stabilize_track_ids or global_identity reassignment)."
        )
    elif allow > reject and allow >= 5:
        lines.append(
            f"- **Dominant: `gate_would_allow` ({allow}/{total}).** "
            f"Velocity alone is insufficient. Swaps happen at convergence where "
            f"endpoint displacement is small — the gate is too permissive by "
            f"construction. Structural next step: learned within-team ReID head, "
            f"or combine trajectory+appearance+pose into a joint cost."
        )
    else:
        lines.append(
            f"- **Mixed: {reject} reject / {allow} allow.** Inspect per-event detail below."
        )

    lines.extend(["", "## Per-event detail", "",
                  "| Rally | Swap frame | pred_old→pred_new | GT | Verdict |",
                  "|---|---:|---|---:|---|"])
    for r in all_results:
        lines.append(
            f"| `{r['rally_id'][:8]}` | {r['swap_frame']} | "
            f"{r['pred_old']}→{r['pred_new']} | {r['gt_track_id']} | `{r['verdict']}` |"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    logger.info(f"\nReport: {args.out}")


if __name__ == "__main__":
    main()
