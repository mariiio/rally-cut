"""Pre-work diagnostic for the W3 blocker rule-based heuristic.

Question the diagnostic answers: is there enough signal in `(is_at_net,
contact_count_on_side, prev_action, court_side_flip, frame_gap)` to move
per-class block F1 from 17.8% toward ≥30% via a purely rule-based override,
without blowing up FP on non-block contacts?

For each GT block across all labeled rallies, runs the production pipeline
(stages 9–14 mirrored from `production_eval.py::_run_rally`) and records:

  - whether any pred contact exists within ±5 frames of the GT block frame
  - the current pred action_type at that contact
  - the Contact fields that feed the existing block rule: is_at_net,
    court_side, plus the previous contact's action/type/frame
  - which specific condition(s) of the current heuristic fail

Then scans every pred contact in every rally and counts, for each draft
rule, (a) how many additional GT blocks would flip to BLOCK, and (b) how
many non-block contacts would incorrectly flip — the FP budget.

Usage
-----
    cd analysis
    uv run python scripts/diagnose_block_heuristic.py
    uv run python scripts/diagnose_block_heuristic.py --limit 50

No DB or code mutations. Prints a report to stdout and writes a JSON dump
to ``analysis/outputs/block_heuristic_diagnostic.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# Reuse production_eval helpers — same pipeline, same loaders.
from eval_action_detection import (  # noqa: E402
    _build_player_positions,
    _load_match_team_assignments,
    load_rallies_with_action_gt,
)
from production_eval import (  # noqa: E402
    PipelineContext,
    _build_calibrators,
    _parse_ball,
    _parse_positions,
)

from rallycut.tracking.action_classifier import (  # noqa: E402
    ActionType,
    classify_rally_actions,
)
from rallycut.tracking.contact_detector import (  # noqa: E402
    Contact,
    detect_contacts,
)
from rallycut.tracking.match_tracker import verify_team_assignments  # noqa: E402
from rallycut.tracking.sequence_action_runtime import (  # noqa: E402
    apply_sequence_override,
    get_sequence_probs,
)

console = Console()

TOL_FRAMES = 5  # ±5 frame window for matching GT contact to pred contact
GAP_MAX = 8  # current ActionClassifierConfig.block_max_frame_gap default


def _nearest_contact(contacts: list[Contact], frame: int) -> Contact | None:
    best: Contact | None = None
    best_d = TOL_FRAMES + 1
    for c in contacts:
        d = abs(c.frame - frame)
        if d <= TOL_FRAMES and d < best_d:
            best = c
            best_d = d
    return best


def _action_at_frame(actions: list[dict], frame: int) -> dict | None:
    best: dict | None = None
    best_d = TOL_FRAMES + 1
    for a in actions:
        d = abs(a.get("frame", -10_000) - frame)
        if d <= TOL_FRAMES and d < best_d:
            best = a
            best_d = d
    return best


def _run_rally_with_contacts(
    rally: Any, match_teams: dict[int, int] | None, calibrator: Any, ctx: PipelineContext
) -> tuple[list[dict], list[Contact]]:
    """Run the prod pipeline for one rally AND return the contact sequence.

    Mirrors production_eval._run_rally exactly except we hold onto the
    contact list (which _run_rally discards).
    """
    ball_positions = _parse_ball(rally.ball_positions_json or [])
    player_positions = _build_player_positions(
        rally.positions_json or [], rally_id=rally.rally_id, inject_pose=True
    )

    teams = dict(match_teams) if match_teams else None
    if teams is not None and not ctx.skip_verify_teams:
        teams = verify_team_assignments(teams, player_positions)

    sequence_probs = get_sequence_probs(
        ball_positions=ball_positions,
        player_positions=player_positions,
        court_split_y=rally.court_split_y,
        frame_count=rally.frame_count,
        team_assignments=teams,
        calibrator=calibrator,
    )

    contact_sequence = detect_contacts(
        ball_positions=ball_positions,
        player_positions=player_positions,
        config=None,
        use_classifier=True,
        frame_count=rally.frame_count or None,
        team_assignments=teams,
        court_calibrator=calibrator,
        sequence_probs=sequence_probs,
    )

    rally_actions = classify_rally_actions(
        contact_sequence,
        rally.rally_id,
        team_assignments=teams,
        match_team_assignments=teams,
        calibrator=calibrator,
    )
    if sequence_probs is not None:
        apply_sequence_override(rally_actions, sequence_probs)

    pred_dicts = [a.to_dict() for a in rally_actions.actions if not a.is_synthetic]
    return pred_dicts, list(contact_sequence.contacts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    console.print("[bold]Loading rallies with action GT...[/bold]")
    rallies = load_rallies_with_action_gt()
    if args.limit:
        rallies = rallies[: args.limit]
    console.print(f"  {len(rallies)} rallies")

    rally_pos_lookup: dict[str, list[Any]] = {
        r.rally_id: _parse_positions(r.positions_json) for r in rallies if r.positions_json
    }
    video_ids = {r.video_id for r in rallies}
    team_map = _load_match_team_assignments(video_ids, rally_positions=rally_pos_lookup)
    calibrators = _build_calibrators(video_ids)
    ctx = PipelineContext()

    # Aggregate counters.
    n_gt_blocks = 0
    n_gt_blocks_with_pred_contact = 0
    pred_action_for_gt_block: Counter[str] = Counter()

    # Condition failure analysis on GT blocks (only for GT blocks WITH a
    # matched pred contact but NOT currently labeled block).
    cond_fail: Counter[str] = Counter()

    # Per-rule TP gain on GT blocks (how many currently-non-block GT blocks
    # would become block) and FP on non-block contacts.
    rules = ["R1_current", "R2_atnet_any_prev_side_flip", "R3_atnet_count3", "R4_atnet_prev_attack_or_set"]
    rule_gt_flip: dict[str, int] = dict.fromkeys(rules, 0)
    rule_fp: dict[str, int] = dict.fromkeys(rules, 0)

    per_rally_dump: list[dict] = []
    rejections = 0

    for idx, rally in enumerate(rallies, start=1):
        if not rally.ball_positions_json or not rally.positions_json:
            rejections += 1
            continue
        if not rally.frame_count or rally.frame_count < 10:
            rejections += 1
            continue

        gt_blocks = [gl for gl in rally.gt_labels if gl.action == "block"]
        # We still need to run the pipeline even if no GT blocks, to collect
        # FP candidates — but that doubles the runtime. For a diagnostic the
        # FP signal from rallies-without-blocks is valuable; keep it.

        try:
            pred_actions, contacts = _run_rally_with_contacts(
                rally,
                team_map.get(rally.rally_id),
                calibrators.get(rally.video_id),
                ctx,
            )
        except Exception as exc:  # noqa: BLE001
            rejections += 1
            console.print(f"  [red][{idx}/{len(rallies)}][/red] {rally.rally_id}: {type(exc).__name__}: {exc}")
            continue

        # Map pred action by frame for quick lookup.
        pred_action_by_frame = {a.get("frame"): a.get("action") for a in pred_actions}

        # ---------------- GT block analysis ----------------
        for gb in gt_blocks:
            n_gt_blocks += 1
            c = _nearest_contact(contacts, gb.frame)
            pa = _action_at_frame(pred_actions, gb.frame)
            pred_label = (pa.get("action") if pa else "_no_contact_")
            pred_action_for_gt_block[pred_label] += 1

            if c is None:
                continue
            n_gt_blocks_with_pred_contact += 1

            # Condition fail breakdown (only if currently NOT labeled block).
            if pred_label != "block":
                # Find prev contact for rule analysis.
                c_idx = contacts.index(c)
                prev_c = contacts[c_idx - 1] if c_idx > 0 else None
                prev_a = (
                    pred_action_by_frame.get(prev_c.frame) if prev_c is not None else None
                )
                if not c.is_at_net:
                    cond_fail["not_at_net"] += 1
                if prev_a != "attack":
                    cond_fail[f"prev_not_attack(is={prev_a})"] += 1
                if prev_c is None or (c.frame - prev_c.frame) > GAP_MAX:
                    cond_fail["gap_too_large"] += 1
                if prev_c is None or prev_c.court_side == c.court_side:
                    cond_fail["same_court_side"] += 1

        # ---------------- Draft rule evaluation ----------------
        # Scan every pred contact, decide if each rule would label it block.
        # GT match: is this contact's frame within ±TOL_FRAMES of a GT block?
        gt_block_frames = {gb.frame for gb in gt_blocks}

        for c_idx, c in enumerate(contacts):
            prev_c = contacts[c_idx - 1] if c_idx > 0 else None
            prev_a = pred_action_by_frame.get(prev_c.frame) if prev_c is not None else None
            current_label = pred_action_by_frame.get(c.frame, "unknown")
            if current_label == "block":
                # already block; no rule "flip" counted
                continue

            is_near_gt_block = any(abs(c.frame - gf) <= TOL_FRAMES for gf in gt_block_frames)

            # contact_count_on_side: count contacts on same side back to last flip.
            count_on_side = 1
            for j in range(c_idx - 1, -1, -1):
                if contacts[j].court_side != c.court_side:
                    break
                count_on_side += 1

            gap = (c.frame - prev_c.frame) if prev_c is not None else 10**9
            side_flip = (prev_c is not None and prev_c.court_side != c.court_side)

            # R1 — current heuristic (baseline; should never flip because
            # classify_rally already applied it)
            r1 = (
                c.is_at_net
                and prev_a == "attack"
                and gap <= GAP_MAX
                and side_flip
            )
            # R2 — is_at_net + side_flip, any prev action
            r2 = c.is_at_net and side_flip and gap <= GAP_MAX
            # R3 — is_at_net + count_on_side ∈ {1} (first touch after side flip)
            r3 = c.is_at_net and count_on_side == 1 and side_flip
            # R4 — is_at_net + prev ∈ {attack, set} + side flip
            r4 = (
                c.is_at_net
                and prev_a in ("attack", "set")
                and gap <= GAP_MAX
                and side_flip
            )

            for name, hit in (
                ("R1_current", r1),
                ("R2_atnet_any_prev_side_flip", r2),
                ("R3_atnet_count3", r3),
                ("R4_atnet_prev_attack_or_set", r4),
            ):
                if not hit:
                    continue
                if is_near_gt_block:
                    rule_gt_flip[name] += 1
                else:
                    rule_fp[name] += 1

        if idx % 20 == 0 or idx == len(rallies):
            console.print(f"  [{idx}/{len(rallies)}] processed")

    # ---------------- Report ----------------
    console.print()
    console.print("[bold]GT block coverage[/bold]")
    console.print(f"  total GT blocks: {n_gt_blocks}")
    console.print(
        f"  GT blocks with a pred contact within ±{TOL_FRAMES}f: "
        f"{n_gt_blocks_with_pred_contact} "
        f"({100.0 * n_gt_blocks_with_pred_contact / max(1, n_gt_blocks):.1f}%)"
    )

    tbl = Table(title="pred action_type on GT blocks")
    tbl.add_column("pred")
    tbl.add_column("count", justify="right")
    for k, v in pred_action_for_gt_block.most_common():
        tbl.add_row(k, str(v))
    console.print(tbl)

    tbl2 = Table(title="which heuristic condition fails on non-block GT blocks")
    tbl2.add_column("condition")
    tbl2.add_column("count", justify="right")
    for k, v in cond_fail.most_common():
        tbl2.add_row(k, str(v))
    console.print(tbl2)

    tbl3 = Table(title="draft rule TP gain / FP cost (across all rallies)")
    tbl3.add_column("rule")
    tbl3.add_column("GT-block flips (TP gain)", justify="right")
    tbl3.add_column("non-GT-block flips (FP)", justify="right")
    tbl3.add_column("TP / (TP+FP)", justify="right")
    for name in rules:
        tp = rule_gt_flip[name]
        fp = rule_fp[name]
        prec = (tp / (tp + fp)) if (tp + fp) else 0.0
        tbl3.add_row(name, str(tp), str(fp), f"{prec:.2f}")
    console.print(tbl3)

    # Ship-gate interpretation.
    console.print()
    console.print(f"  rejections: {rejections}")
    console.print()
    console.print("[bold]Gate evaluation[/bold]")
    current_block_tp = pred_action_for_gt_block.get("block", 0)
    console.print(f"  currently correct GT-block predictions: {current_block_tp}")
    console.print(
        "  gate: a draft rule is viable if (current_tp + tp_gain) / n_gt_blocks ≥ 0.30 "
        "AND fp ≤ tp_gain"
    )
    for name in rules:
        tp_gain = rule_gt_flip[name]
        fp = rule_fp[name]
        new_recall = (current_block_tp + tp_gain) / max(1, n_gt_blocks)
        gate_ok = new_recall >= 0.30 and fp <= tp_gain
        mark = "[green]OK[/green]" if gate_ok else "[red]NO[/red]"
        console.print(
            f"  {mark} {name}: new block recall ≈ {100*new_recall:.1f}%, "
            f"tp_gain={tp_gain}, fp={fp}"
        )

    # Dump JSON for traceability.
    out = {
        "n_rallies": len(rallies),
        "rejections": rejections,
        "n_gt_blocks": n_gt_blocks,
        "n_gt_blocks_with_pred_contact": n_gt_blocks_with_pred_contact,
        "pred_action_for_gt_block": dict(pred_action_for_gt_block),
        "cond_fail": dict(cond_fail),
        "rule_gt_flip": rule_gt_flip,
        "rule_fp": rule_fp,
        "current_block_tp": current_block_tp,
    }
    out_dir = Path(__file__).resolve().parent.parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "block_heuristic_diagnostic.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    console.print(f"[green]wrote[/green] {out_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)
