#!/usr/bin/env python3
"""Extract empirical (prev_action_type → this_action_type) transition
probabilities from trusted-29 GT.

Used by the joint Viterbi to replace hand-coded transition priors with
data-derived ones — the probe for whether learned transitions (which
is what a CRF would do) would close the gap between naive Viterbi
(84.8%) and the v3.2 cascade (91.6%).

Outputs: JSON file at reports/empirical_transitions_2026_05_17.json
mapping prev_type → this_type → probability.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import psycopg

DB = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5436/rallycut",
)
TRUSTED_29 = (
    "titi", "toto", "lulu", "wawa", "caco", "cece", "cici", "cuco",
    "gaga", "kaka", "kiki", "juju", "yeye", "keke",
    "gigi", "gugu", "mame", "meme", "mimi", "moma", "mumu",
    "papa", "pepe", "pipi", "popo", "pupu", "veve", "vivi", "vovo",
)

OUT_DIR = Path("reports")
OUT_DIR.mkdir(exist_ok=True)
OUT_PATH = OUT_DIR / "empirical_transitions_2026_05_17.json"


def main() -> int:
    rally_actions: dict[str, list[tuple[int, str, int]]] = defaultdict(list)
    with psycopg.connect(DB) as conn:
        cur = conn.execute(
            """
            SELECT r.id, rg.frame, rg.action::text, rg.resolved_track_id
            FROM rally_action_ground_truth rg
            JOIN rallies r ON rg.rally_id = r.id
            JOIN videos v ON r.video_id = v.id
            WHERE v.name = ANY(%s) AND rg.resolved_track_id IS NOT NULL
            """,
            [list(TRUSTED_29)],
        )
        for rid, frame, action, tid in cur.fetchall():
            rally_actions[rid].append((frame, action.upper(), tid))

    print(f"Processed {len(rally_actions)} rallies with GT")

    # Count transitions
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    n_pairs = 0
    for actions in rally_actions.values():
        actions.sort(key=lambda x: x[0])
        for i in range(1, len(actions)):
            prev = actions[i - 1][1]
            this = actions[i][1]
            counts[prev][this] += 1
            n_pairs += 1

    print(f"Total transition pairs: {n_pairs}")
    print()
    print("Empirical transitions (prev → this | P, count):")

    transitions: dict[str, dict[str, float]] = {}
    for prev in sorted(counts):
        total = sum(counts[prev].values())
        transitions[prev] = {
            this: count / total for this, count in counts[prev].items()
        }
        print(f"  {prev} (n={total}):")
        for this in sorted(counts[prev].keys(), key=lambda k: -counts[prev][k]):
            n = counts[prev][this]
            p = transitions[prev][this]
            print(f"    → {this:10s}  {p:.3f}  (n={n})")
        print()

    OUT_PATH.write_text(json.dumps(transitions, indent=2))
    print(f"Wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
