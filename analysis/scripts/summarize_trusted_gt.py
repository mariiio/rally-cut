"""Summarize Phase-0 trusted-GT audit.

Reports:
* How many of the 45 GT actions were re-labeled vs left unchanged
* How well the *predicted-by-current-pipeline* canonical PID agrees with
  the trusted human label — i.e. the new baseline the rest of the
  research plan measures against
* Convention-drift evidence (rows where original GT and trusted differ
  by a fixed permutation may indicate a wholesale renumbering)
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    args = parser.parse_args()

    data = json.loads(args.input.read_text())
    actions = data["actions"]
    n_total = len(actions)

    n_relabeled = sum(1 for a in actions if a["wasRelabeled"])
    n_cant_tell = sum(1 for a in actions if a["reviewedAs"] == "can't tell")
    n_no_actor = sum(1 for a in actions if a["reviewedAs"] == "no actor visible")
    n_evaluable = sum(1 for a in actions if a["trustedCanonicalPid"] is not None)

    pred_correct = sum(
        1 for a in actions
        if a["trustedCanonicalPid"] is not None
        and a["predictedCanonicalPid"] is not None
        and a["trustedCanonicalPid"] == a["predictedCanonicalPid"]
    )
    pred_wrong = sum(
        1 for a in actions
        if a["trustedCanonicalPid"] is not None
        and a["predictedCanonicalPid"] is not None
        and a["trustedCanonicalPid"] != a["predictedCanonicalPid"]
    )
    pred_missing = sum(
        1 for a in actions
        if a["trustedCanonicalPid"] is not None and a["predictedCanonicalPid"] is None
    )

    orig_correct = sum(
        1 for a in actions
        if a["trustedCanonicalPid"] is not None
        and a["originalCanonicalPid"] == a["trustedCanonicalPid"]
    )

    print(f"Total GT actions: {n_total}")
    print(f"  Re-labeled by reviewer: {n_relabeled} ({n_relabeled / n_total:.1%})")
    print(f"  'Can't tell':           {n_cant_tell}")
    print(f"  'No actor visible':     {n_no_actor}")
    print(f"  Evaluable (trusted PID): {n_evaluable}")
    print()
    print("Original DB GT vs trusted:")
    print(f"  matched: {orig_correct} / {n_evaluable} = "
          f"{orig_correct / max(1, n_evaluable):.1%}")
    print()
    print("Pipeline prediction vs trusted GT (the real baseline):")
    print(f"  correct:        {pred_correct}")
    print(f"  wrong:          {pred_wrong}")
    print(f"  no prediction:  {pred_missing}")
    n_pred = pred_correct + pred_wrong + pred_missing
    print(f"  attribution acc (correct / evaluable): "
          f"{pred_correct / max(1, n_pred):.1%} ({pred_correct}/{n_pred})")
    print(f"  attribution acc (correct / where pipeline made a prediction): "
          f"{pred_correct / max(1, pred_correct + pred_wrong):.1%} "
          f"({pred_correct}/{pred_correct + pred_wrong})")
    print()
    # Convention-drift check: tally orig→trusted permutation across relabeled rows.
    perm = Counter()
    for a in actions:
        if a["wasRelabeled"] and a["trustedCanonicalPid"] is not None:
            perm[(a["originalCanonicalPid"], a["trustedCanonicalPid"])] += 1
    print("Re-labeling permutation (orig_pid → trusted_pid : count):")
    for (o, t), c in sorted(perm.items(), key=lambda x: -x[1]):
        print(f"  P{o} → P{t}: {c}")
    print()
    # Per-action breakdown of prediction quality.
    by_action: dict[str, list[int]] = {}
    for a in actions:
        if a["trustedCanonicalPid"] is None:
            continue
        if a["predictedCanonicalPid"] is None:
            continue
        rec = by_action.setdefault(a["action"], [0, 0])  # [correct, total]
        rec[1] += 1
        if a["predictedCanonicalPid"] == a["trustedCanonicalPid"]:
            rec[0] += 1
    print("Per-action attribution accuracy (where pipeline made a prediction):")
    for act, (c, t) in sorted(by_action.items()):
        print(f"  {act:10s}: {c}/{t} = {c / t:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
