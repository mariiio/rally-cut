"""Probe X-B (asymmetric): quantify FP risk of the new asymmetric is_at_net.

Old rule:  abs(ball_y - net_y) < 0.08
New rule:  -0.15 <= (ball_y - net_y) <= 0.08
           (image-y down → ball.y < net_y means ball is ABOVE net top)
           Strict superset of old: same below-net threshold, widened above.

For every contact stored on trusted-31 rallies:
  - classify under OLD and NEW rules
  - count flips: False→True (newly at-net), True→False (no longer at-net)
  - cross-tabulate flips by stored action_type so we can see whether the
    asymmetric widening lights up legitimate blocks vs mid-court sets/digs

Sample 20 random flips for visual inspection (printed as
`<video> <rally> <frame> <action>` lines — open at /tmp/net_verify or
re-render with probe_X_e_visual_annotate.py).
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict

import psycopg

DB_DSN = "postgresql://postgres:postgres@localhost:5436/rallycut"

TRUSTED_31 = [
    "titi", "toto", "lulu", "wawa", "caco", "cece", "cici", "cuco",
    "gaga", "gigi", "kaka", "kiki", "keke", "koko", "kuku", "juju",
    "yeye", "gugu", "mame", "meme", "mimi", "moma", "mumu", "papa",
    "pepe", "pipi", "popo", "pupu", "veve", "vivi", "vovo",
]

# OLD: symmetric ±0.08. NEW: asymmetric (-0.15 above, +0.08 below).
OLD_NET_ZONE = 0.08
NEW_ABOVE = 0.15
NEW_BELOW = 0.08


def _old_at_net(ball_y: float, net_y: float) -> bool:
    return abs(ball_y - net_y) < OLD_NET_ZONE


def _new_at_net(ball_y: float, net_y: float) -> bool:
    delta = ball_y - net_y
    return -NEW_ABOVE <= delta <= NEW_BELOW


def main() -> int:
    random.seed(42)
    with psycopg.connect(DB_DSN) as conn:
        placeholders = ",".join(["%s"] * len(TRUSTED_31))
        cur = conn.execute(
            f"""
            SELECT v.name, r.id::text, pt.contacts_json, pt.actions_json
            FROM rallies r
            JOIN videos v ON r.video_id = v.id
            JOIN player_tracks pt ON pt.rally_id = r.id
            WHERE v.name IN ({placeholders})
              AND pt.contacts_json IS NOT NULL
            ORDER BY v.name, r.start_ms
            """,
            TRUSTED_31,
        )
        rows = cur.fetchall()

    total = 0
    counts = Counter()  # (old, new) -> count
    flip_to_true: list[tuple[str, str, int, str, float]] = []
    flip_to_false: list[tuple[str, str, int, str, float]] = []

    # action-type lookup by (rally_id, frame)
    flips_by_action: dict[str, Counter] = defaultdict(Counter)

    for vname, rally_id, cj_raw, aj_raw in rows:
        cj = cj_raw if isinstance(cj_raw, dict) else json.loads(cj_raw or '{}')
        aj = aj_raw if isinstance(aj_raw, dict) else json.loads(aj_raw or '{}')
        net_y = cj.get("netY", 0.5)
        actions_by_frame = {a.get("frame"): a.get("action", "?") for a in aj.get("actions", [])}

        for c in cj.get("contacts", []):
            total += 1
            ball_y = c.get("ballY", 0.5)
            frame = c.get("frame", -1)
            old = _old_at_net(ball_y, net_y)
            new = _new_at_net(ball_y, net_y)
            counts[(old, new)] += 1
            action = actions_by_frame.get(frame, "unknown")
            delta = ball_y - net_y

            if (not old) and new:
                flip_to_true.append((vname, rally_id[:8], frame, action, delta))
                flips_by_action["to_true"][action] += 1
            elif old and (not new):
                flip_to_false.append((vname, rally_id[:8], frame, action, delta))
                flips_by_action["to_false"][action] += 1

    print(f"Trusted-31 contacts: total={total}", flush=True)
    print(f"Old at-net (sym 0.08):     {counts[(True, True)] + counts[(True, False)]:>5}", flush=True)
    print(f"New at-net (asym 0.15/0.08): {counts[(True, True)] + counts[(False, True)]:>5}", flush=True)
    print()
    print("Transition table (rows=OLD, cols=NEW):")
    print(f"{'':>10} {'NEW False':>10} {'NEW True':>10}")
    print(f"{'OLD False':>10} {counts[(False, False)]:>10} {counts[(False, True)]:>10}")
    print(f"{'OLD True':>10} {counts[(True, False)]:>10} {counts[(True, True)]:>10}")
    print()
    print(f"FLIP to True  (gained at-net): {len(flip_to_true)}", flush=True)
    print(f"FLIP to False (lost at-net):   {len(flip_to_false)}", flush=True)
    print()

    print("Flips-to-True by action type (these are the new at-net contacts):")
    for action, n in flips_by_action["to_true"].most_common():
        print(f"  {action:<10} {n:>4}")
    print()

    print("Flips-to-False by action type (these are removed at-net contacts):")
    for action, n in flips_by_action["to_false"].most_common():
        print(f"  {action:<10} {n:>4}")
    print()

    # Sample 20 random flips-to-True for visual inspection
    sample = random.sample(flip_to_true, min(20, len(flip_to_true)))
    print("Random 20 flip-to-True samples (video rally frame action ball.y-net.y):")
    for vname, rally, frame, action, delta in sample:
        print(f"  {vname:<6} {rally:<10} f={frame:>4}  action={action:<8}  delta={delta:+.3f}")
    print()

    # Acceptance check from plan: "must not flip more than 2× the recovered
    # block contacts as new FPs". Expected recovery ≤ 6 blocks → ≤ 12 FPs.
    # FP proxy = flips-to-True that are NOT block. (Sets/digs at the net are
    # ambiguous; visual sampling tells us if they're real near-net contacts.)
    expected_block_recovery = 6
    non_block_flips_to_true = sum(
        n for a, n in flips_by_action["to_true"].items() if a != "block"
    )
    ratio = non_block_flips_to_true / max(1, expected_block_recovery)
    print(
        f"FP-guard ratio (non-block flips-to-True per block recovered): "
        f"{non_block_flips_to_true}/{expected_block_recovery} = {ratio:.1f}x",
        flush=True,
    )
    print(
        f"Gate from plan: <= 2.0x. Verdict: "
        f"{'PASS' if ratio <= 2.0 else 'FAIL (sample to confirm)'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
