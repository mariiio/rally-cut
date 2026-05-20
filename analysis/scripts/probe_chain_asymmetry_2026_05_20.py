#!/usr/bin/env python3
"""Chain-asymmetry diagnostic probe (2026-05-20).

For each of the 94 chain-disagreement contacts from L6, collect evidence
for H1/H2/H3 hypotheses about why pipeline systematically over-assigns
team A (63 pipe=A/gt=B vs 31 pipe=B/gt=A, binomial p≈0.001).

H1 (serving-team detection bias): first-serve attribution wrong → chain
    seeded with wrong team.
H2 (team_assignments labeling skew): disagreement rallies have more
    A-PIDs than agreeing rallies / fleet baseline.
H3 (chain-walker init/transition bug): synthetic-serve seed, UNKNOWN
    actions, or chain-integrity False at contact.

Output:
  reports/chain_asymmetry_2026_05_20/per_disagreement.csv
  reports/chain_asymmetry_2026_05_20/per_hypothesis.json
  reports/chain_asymmetry_2026_05_20/summary.md
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import psycopg

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ANALYSIS_DIR / "scripts"))

from _upstream_probe_common import (  # noqa: E402
    fetch_rally_state,
    load_wrong_attribution_corpus,
)

DB_DSN = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5436/rallycut",
)
OUT_DIR = ANALYSIS_DIR / "reports" / "chain_asymmetry_2026_05_20"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def pipeline_team_at_frame(actions: list[dict], frame: int, teams: dict[str, str]) -> str | None:
    """Pipeline team at this frame from actions_json."""
    for a in actions:
        if int(a.get("frame", -1)) == frame:
            tid = a.get("playerTrackId")
            if tid is None or tid == -1:
                return None
            t = teams.get(str(tid))
            return t if t in ("A", "B") else None
    return None


def gt_team_at_frame(
    rally_id: str, frame: int, teams: dict[str, str],
) -> str | None:
    """GT team at nearest GT frame within ±5 of `frame`."""
    with psycopg.connect(DB_DSN) as conn:
        cur = conn.execute(
            """
            SELECT frame, resolved_track_id FROM rally_action_ground_truth
            WHERE rally_id = %s AND resolved_track_id IS NOT NULL
              AND ABS(frame - %s) <= 5
            ORDER BY ABS(frame - %s)
            LIMIT 1
            """,
            [rally_id, frame, frame],
        )
        row = cur.fetchone()
    if not row:
        return None
    _, tid = row
    t = teams.get(str(tid))
    return t if t in ("A", "B") else None


def first_serve_in_rally(actions: list[dict]) -> dict | None:
    """Return first action with action='serve' (chronological), or None."""
    serves = sorted(
        [a for a in actions if (a.get("action") or "").lower() == "serve"],
        key=lambda a: int(a.get("frame", 10**9)),
    )
    return serves[0] if serves else None


def gt_first_serve(rally_id: str) -> tuple[int, int] | None:
    """Return (gt_frame, gt_resolved_tid) for the first GT serve in rally,
    or None if no GT serve."""
    with psycopg.connect(DB_DSN) as conn:
        cur = conn.execute(
            """
            SELECT frame, resolved_track_id FROM rally_action_ground_truth
            WHERE rally_id = %s AND action::text = 'serve'
              AND resolved_track_id IS NOT NULL
            ORDER BY frame
            LIMIT 1
            """,
            [rally_id],
        )
        row = cur.fetchone()
    if not row:
        return None
    return int(row[0]), int(row[1])


def count_unknown_before(actions: list[dict], frame: int) -> int:
    """How many UNKNOWN actions appear in `actions` before `frame`?"""
    return sum(
        1 for a in actions
        if (a.get("action") or "").upper() == "UNKNOWN"
        and int(a.get("frame", 10**9)) < frame
    )


def chain_integrity_at_contact(
    actions: list[dict], frame: int,
) -> tuple[bool, bool]:
    """Replicates _chain_integrity logic; returns (chain_integrity_at_contact,
    first_serve_is_synthetic).

    Chain seed = first SERVE with playerTrackId >= 0. Integrity broken by
    UNKNOWN or non-seed synthetic actions between seed and contact.
    """
    sorted_actions = sorted(actions, key=lambda a: int(a.get("frame", 10**9)))
    seen_seed = False
    broken = False
    first_serve_is_synthetic = False
    integrity = False
    for a in sorted_actions:
        a_frame = int(a.get("frame", -1))
        a_action = (a.get("action") or "").lower()
        a_tid = int(a.get("playerTrackId", -1))
        a_is_synth = bool(a.get("isSynthetic", False))
        if not seen_seed:
            if a_action == "serve" and a_tid >= 0:
                seen_seed = True
                first_serve_is_synthetic = a_is_synth
                if a_frame == frame:
                    integrity = True
                    break
            continue
        if a_action == "unknown" or (a_is_synth and a_action != "serve"):
            broken = True
        if a_frame == frame:
            integrity = not broken
            break
    return integrity, first_serve_is_synthetic


def team_distribution(teams: dict[str, str]) -> tuple[int, int]:
    """Return (n_team_a, n_team_b) in teamAssignments."""
    na = sum(1 for t in teams.values() if t == "A")
    nb = sum(1 for t in teams.values() if t == "B")
    return na, nb


def main() -> int:
    print("Loading wrong-attribution corpus...", flush=True)
    rows = load_wrong_attribution_corpus()
    print(f"  {len(rows)} wrong-attribution contacts", flush=True)

    disagreement_rows: list[dict[str, Any]] = []
    n_processed = 0
    agreeing_rally_ids: set[str] = set()
    disagreement_rally_ids: set[str] = set()

    for i, row in enumerate(rows):
        rally = fetch_rally_state(row.rally_id)
        if rally is None:
            continue
        teams = rally["teams"]
        pipe_team = pipeline_team_at_frame(rally["actions"], row.action_frame, teams)
        gt_team = gt_team_at_frame(row.rally_id, row.action_frame, teams)
        if pipe_team is None or gt_team is None:
            continue
        n_processed += 1
        if pipe_team == gt_team:
            agreeing_rally_ids.add(row.rally_id)
            continue
        disagreement_rally_ids.add(row.rally_id)

        # H1: first serve attribution check
        pipe_first_serve = first_serve_in_rally(rally["actions"])
        gt_first = gt_first_serve(row.rally_id)
        pipe_first_serve_tid = -1
        pipe_first_serve_team: str | None = None
        if pipe_first_serve is not None:
            pipe_first_serve_tid = int(pipe_first_serve.get("playerTrackId", -1))
            pipe_first_serve_team = teams.get(str(pipe_first_serve_tid))
        gt_first_serve_tid = -1
        gt_first_serve_team: str | None = None
        if gt_first is not None:
            gt_first_serve_tid = gt_first[1]
            gt_first_serve_team = teams.get(str(gt_first_serve_tid))
        h1_first_serve_wrong = (
            pipe_first_serve_tid >= 0 and gt_first_serve_tid >= 0
            and pipe_first_serve_tid != gt_first_serve_tid
        )

        # H2: team_assignments A/B distribution
        n_a, n_b = team_distribution(teams)

        # H3: chain-walker conditions at this contact
        integrity, first_serve_synthetic = chain_integrity_at_contact(
            rally["actions"], row.action_frame,
        )
        unknown_before = count_unknown_before(rally["actions"], row.action_frame)

        disagreement_rows.append({
            "rally_id": row.rally_id,
            "video": row.video,
            "action_frame": row.action_frame,
            "action_type": row.action_type,
            "pipeline_pid": row.pipeline_pid,
            "gt_pid": row.gt_pid,
            "pipe_team": pipe_team,
            "gt_team": gt_team,
            "h1_first_serve_wrong": h1_first_serve_wrong,
            "h1_pipe_first_serve_team": pipe_first_serve_team or "",
            "h1_gt_first_serve_team": gt_first_serve_team or "",
            "h2_team_a_count": n_a,
            "h2_team_b_count": n_b,
            "h3_chain_integrity": integrity,
            "h3_first_serve_synthetic": first_serve_synthetic,
            "h3_unknown_before": unknown_before,
        })

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(rows)}] processed", flush=True)

    print(f"\nProcessed {n_processed} contacts; "
          f"{len(disagreement_rows)} chain disagreements", flush=True)

    # Aggregate per-hypothesis evidence
    n_dis = len(disagreement_rows)

    # H1
    h1_wrong = sum(1 for r in disagreement_rows if r["h1_first_serve_wrong"])
    h1_wrong_a_seeded = sum(
        1 for r in disagreement_rows
        if r["h1_first_serve_wrong"] and r["h1_pipe_first_serve_team"] == "A"
    )
    h1_wrong_b_seeded = sum(
        1 for r in disagreement_rows
        if r["h1_first_serve_wrong"] and r["h1_pipe_first_serve_team"] == "B"
    )

    # H2: A-skew comparison
    h2_disagreement_avg_a = (
        sum(r["h2_team_a_count"] for r in disagreement_rows) / max(n_dis, 1)
    )
    h2_disagreement_avg_b = (
        sum(r["h2_team_b_count"] for r in disagreement_rows) / max(n_dis, 1)
    )
    # Fleet baseline: sample from agreeing rallies (those with chain == gt)
    # via fresh DB queries — best-effort approximation
    fleet_a_counts: list[int] = []
    fleet_b_counts: list[int] = []
    seen_rallies: set[str] = set()
    for r in disagreement_rows:
        if r["rally_id"] in seen_rallies:
            continue
        seen_rallies.add(r["rally_id"])
        fleet_a_counts.append(r["h2_team_a_count"])
        fleet_b_counts.append(r["h2_team_b_count"])
    h2_disagreement_rally_avg_a = (
        sum(fleet_a_counts) / max(len(fleet_a_counts), 1)
    )
    h2_disagreement_rally_avg_b = (
        sum(fleet_b_counts) / max(len(fleet_b_counts), 1)
    )

    # H3
    h3_synthetic_seed = sum(
        1 for r in disagreement_rows if r["h3_first_serve_synthetic"]
    )
    h3_chain_broken = sum(
        1 for r in disagreement_rows if not r["h3_chain_integrity"]
    )
    h3_with_unknowns = sum(
        1 for r in disagreement_rows if r["h3_unknown_before"] > 0
    )

    # Direction-asymmetry breakdown per hypothesis (does H1/H3 explain the A-skew?)
    pipe_a_gt_b = [r for r in disagreement_rows
                   if r["pipe_team"] == "A" and r["gt_team"] == "B"]
    pipe_b_gt_a = [r for r in disagreement_rows
                   if r["pipe_team"] == "B" and r["gt_team"] == "A"]
    h1_in_a_dir = sum(1 for r in pipe_a_gt_b if r["h1_first_serve_wrong"])
    h1_in_b_dir = sum(1 for r in pipe_b_gt_a if r["h1_first_serve_wrong"])

    # Primary cause assignment (heuristic: H1 > H3 > H2 if present)
    primary_cause: Counter = Counter()
    for r in disagreement_rows:
        if r["h1_first_serve_wrong"]:
            primary_cause["H1"] += 1
        elif not r["h3_chain_integrity"] or r["h3_first_serve_synthetic"]:
            primary_cause["H3"] += 1
        else:
            primary_cause["unexplained"] += 1
    # H2 is a rally-level statistic, not per-contact — reported separately

    per_hypothesis = {
        "n_disagreements": n_dis,
        "direction_split": {
            "pipe_A_gt_B": len(pipe_a_gt_b),
            "pipe_B_gt_A": len(pipe_b_gt_a),
        },
        "H1": {
            "first_serve_wrong_count": h1_wrong,
            "first_serve_wrong_pct": 100 * h1_wrong / max(n_dis, 1),
            "wrong_seeded_to_A": h1_wrong_a_seeded,
            "wrong_seeded_to_B": h1_wrong_b_seeded,
            "h1_explains_pipe_A_gt_B": h1_in_a_dir,
            "h1_explains_pipe_B_gt_A": h1_in_b_dir,
        },
        "H2": {
            "disagreement_contact_avg_A": h2_disagreement_avg_a,
            "disagreement_contact_avg_B": h2_disagreement_avg_b,
            "disagreement_rally_avg_A": h2_disagreement_rally_avg_a,
            "disagreement_rally_avg_B": h2_disagreement_rally_avg_b,
            "n_unique_rallies": len(fleet_a_counts),
        },
        "H3": {
            "synthetic_seed_count": h3_synthetic_seed,
            "synthetic_seed_pct": 100 * h3_synthetic_seed / max(n_dis, 1),
            "chain_broken_count": h3_chain_broken,
            "chain_broken_pct": 100 * h3_chain_broken / max(n_dis, 1),
            "with_unknowns_before_count": h3_with_unknowns,
            "with_unknowns_before_pct": 100 * h3_with_unknowns / max(n_dis, 1),
        },
        "primary_cause": dict(primary_cause),
    }

    # Write outputs
    (OUT_DIR / "per_hypothesis.json").write_text(
        json.dumps(per_hypothesis, indent=2),
    )
    csv_path = OUT_DIR / "per_disagreement.csv"
    if disagreement_rows:
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(disagreement_rows[0].keys()))
            w.writeheader()
            w.writerows(disagreement_rows)

    # Build summary.md
    md = ["# Chain Asymmetry Diagnostic — Summary (2026-05-20)", ""]
    md.append(f"Substrate: {n_dis} chain-disagreement contacts "
              f"(from {n_processed} processed).")
    md.append(f"Direction: {len(pipe_a_gt_b)} pipe=A/gt=B vs "
              f"{len(pipe_b_gt_a)} pipe=B/gt=A.")
    md.append("")
    md.append("## H1 — Serving-team detection bias")
    md.append("")
    h1_wrong_pct = 100 * h1_wrong / max(n_dis, 1)
    md.append(f"- First-serve attribution WRONG: **{h1_wrong}/{n_dis} "
              f"({h1_wrong_pct:.1f}%)**")
    md.append(f"- Of wrong first-serves, seeded to A: {h1_wrong_a_seeded}, "
              f"to B: {h1_wrong_b_seeded}")
    md.append(f"- H1 explains pipe=A errors: {h1_in_a_dir}/{len(pipe_a_gt_b)}")
    md.append(f"- H1 explains pipe=B errors: {h1_in_b_dir}/{len(pipe_b_gt_a)}")
    h1_dominates = h1_wrong / max(n_dis, 1) >= 0.6
    md.append(f"- **H1 dominant (≥60%):** {'YES' if h1_dominates else 'NO'}")
    md.append("")
    md.append("## H2 — team_assignments labeling skew")
    md.append("")
    md.append(f"- Disagreement-rally avg A: {h2_disagreement_rally_avg_a:.2f}, "
              f"avg B: {h2_disagreement_rally_avg_b:.2f}")
    h2_dominates = h2_disagreement_rally_avg_a > h2_disagreement_rally_avg_b * 1.5
    md.append(f"- Significant A-skew (avg_A > 1.5x avg_B): "
              f"{'YES' if h2_dominates else 'NO'}")
    md.append("")
    md.append("## H3 — Chain-walker init/transition bug")
    md.append("")
    h3_synthetic_seed_pct = 100 * h3_synthetic_seed / max(n_dis, 1)
    h3_chain_broken_pct = 100 * h3_chain_broken / max(n_dis, 1)
    h3_with_unknowns_pct = 100 * h3_with_unknowns / max(n_dis, 1)
    md.append(f"- Synthetic-seed: **{h3_synthetic_seed}/{n_dis} "
              f"({h3_synthetic_seed_pct:.1f}%)**")
    md.append(f"- Chain integrity False at contact: **{h3_chain_broken}/{n_dis} "
              f"({h3_chain_broken_pct:.1f}%)**")
    md.append(f"- Has UNKNOWN actions before contact: {h3_with_unknowns}/{n_dis} "
              f"({h3_with_unknowns_pct:.1f}%)")
    h3_dominates = (h3_synthetic_seed + h3_chain_broken) / max(n_dis, 1) >= 0.6
    md.append(f"- **H3 dominant (≥60%):** {'YES' if h3_dominates else 'NO'}")
    md.append("")
    md.append("## Primary-cause assignment")
    md.append("")
    for k, n in primary_cause.most_common():
        md.append(f"- {k}: {n}/{n_dis} ({100*n/max(n_dis, 1):.1f}%)")
    md.append("")
    md.append("## Verdict")
    md.append("")
    if h1_dominates:
        md.append("- **H1 dominates.** Fix: spec a serving-team detection "
                  "improvement / first-serve attribution guardrail.")
    elif h3_dominates:
        md.append("- **H3 dominates.** Fix: spec a chain-walker code fix "
                  "(synthetic-serve handling / chain-integrity logic).")
    elif h2_dominates:
        md.append("- **H2 dominates.** Investigate team_assignments "
                  "labeling pipeline.")
    else:
        md.append("- **No single hypothesis dominates.** Escalate to full "
                  "chain-quality rewrite (separate brainstorm cycle).")
    md.append("")

    (OUT_DIR / "summary.md").write_text("\n".join(md))
    print(f"\nWrote {OUT_DIR/'summary.md'}", flush=True)
    print(f"Wrote {OUT_DIR/'per_hypothesis.json'}", flush=True)
    print(f"Wrote {csv_path}", flush=True)
    print()
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
