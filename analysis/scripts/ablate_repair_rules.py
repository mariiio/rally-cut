"""Per-rule ablation of repair_action_sequence().

Tests each repair rule independently using leave-one-video-out CV.
Reuses the LOO-CV infrastructure from eval_pipeline_loocv.py.

Reports: rule name, # times triggered, action accuracy delta, F1 delta.

Usage:
    cd analysis
    uv run python scripts/ablate_repair_rules.py
    uv run python scripts/ablate_repair_rules.py --only-video abc  # debug single fold
"""

from __future__ import annotations

import argparse
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.table import Table

from rallycut.court.calibration import CourtCalibrator
from rallycut.evaluation.tracking.db import load_court_calibration
from rallycut.tracking.action_classifier import (
    ActionClassifier,
    ActionClassifierConfig,
    assign_court_side_from_teams,
    repair_action_sequence,
    validate_action_sequence,
    viterbi_decode_actions,
)
from rallycut.tracking.action_type_classifier import ActionTypeClassifier
from rallycut.tracking.ball_tracker import BallPosition as BallPos
from rallycut.tracking.contact_detector import (
    ContactDetectionConfig,
    detect_contacts,
)
from rallycut.tracking.player_tracker import PlayerPosition as PlayerPos
from scripts.eval_action_detection import (
    MatchResult,
    RallyData,
    _load_match_team_assignments,
    _match_synthetic_serves,
    compute_metrics,
    load_rallies_with_action_gt,
    match_contacts,
)
from scripts.eval_pipeline_loocv import (
    preextract_all_features,
    train_action_gbm_from_cache,
)

# Suppress verbose logging
logging.getLogger("rallycut.tracking.contact_detector").setLevel(logging.WARNING)
logging.getLogger("rallycut.tracking.action_classifier").setLevel(logging.WARNING)

console = Console()

ALL_RULES = {0, 1, 2, 3, 4, 5, 6}

RULE_NAMES = {
    0: "wrong-side serve → receive + synth",
    1: "consecutive recv/dig → set",
    2: "recv/dig→attack → set",
    3: "duplicate serves → receive",
    4: "duplicate receives → set",
    5: "set→set → attack",
    6: "attack→attack → set",
}


@dataclass
class AblationConfig:
    """Which repair rules are enabled for this config."""
    name: str
    enabled_rules: set[int]  # Rules that are ON

    @property
    def disabled_rules(self) -> set[int]:
        return ALL_RULES - self.enabled_rules


def build_configs(combos: bool = False) -> list[AblationConfig]:
    """Build ablation configs.

    Default: no_repair, each rule solo, all_rules.
    With combos=True: baseline + promising combinations only.
    """
    if combos:
        return [
            AblationConfig("no_repair", set()),
            AblationConfig("rule_1", {1}),
            AblationConfig("rule_1+4", {1, 4}),
            AblationConfig("rule_1+3", {1, 3}),
            AblationConfig("rule_1+3+4", {1, 3, 4}),
            AblationConfig("rule_1+4+0", {1, 4, 0}),
            AblationConfig("rule_1+3+4+0", {1, 3, 4, 0}),
            AblationConfig("all_minus_2", ALL_RULES - {2}),
            AblationConfig("all_rules", ALL_RULES.copy()),
        ]
    configs = [
        AblationConfig("no_repair", set()),
        AblationConfig("all_rules", ALL_RULES.copy()),
    ]
    for r in sorted(ALL_RULES):
        configs.append(AblationConfig(f"only_rule_{r}", {r}))
    return configs


# ---------------------------------------------------------------------------
# Pipeline with per-rule control
# ---------------------------------------------------------------------------

def run_pipeline_with_rules(
    contact_sequence: Any,
    rally_id: str,
    action_classifier: ActionTypeClassifier | None,
    match_team_assignments: dict[int, int] | None,
    disabled_rules: set[int],
) -> tuple[Any, dict[int, int]]:
    """Run classify → repair (with rule control) → viterbi → validate.

    Returns (RallyActions, trigger_counts).
    """
    ac = ActionClassifier(ActionClassifierConfig())

    result = ac.classify_rally(
        contact_sequence, rally_id,
        classifier=action_classifier,
        match_team_assignments=match_team_assignments,
    )

    # Extract server track ID for repair
    serve_tid = -1
    for a in result.actions:
        if a.action_type.value == "serve":
            serve_tid = a.player_track_id
            break

    # Run repair with per-rule control (always get trigger counts)
    repaired, triggers = repair_action_sequence(
        result.actions,
        net_y=contact_sequence.net_y,
        ball_positions=contact_sequence.ball_positions,
        rally_start_frame=contact_sequence.rally_start_frame,
        server_track_id=serve_tid,
        disabled_rules=disabled_rules,
    )
    result.actions = repaired

    result.actions = viterbi_decode_actions(result.actions)
    result.actions = validate_action_sequence(result.actions, rally_id)

    if match_team_assignments:
        assign_court_side_from_teams(result.actions, match_team_assignments)

    return result, triggers


# ---------------------------------------------------------------------------
# Fold evaluation
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    video_id: str
    n_rallies: int = 0
    n_gt: int = 0
    n_tp: int = 0
    n_fp: int = 0
    n_fn: int = 0
    action_correct: int = 0
    triggers: dict[int, int] = field(default_factory=lambda: {r: 0 for r in range(7)})
    per_class_correct: dict[str, int] = field(default_factory=dict)
    per_class_total: dict[str, int] = field(default_factory=dict)


def evaluate_fold(
    rallies: list[RallyData],
    action_classifier: ActionTypeClassifier | None,
    match_teams_by_rally: dict[str, dict[int, int]],
    calibrators: dict[str, CourtCalibrator | None],
    disabled_rules: set[int],
    tolerance_ms: int = 167,
) -> FoldResult:
    result = FoldResult(
        video_id=rallies[0].video_id if rallies else "",
        n_rallies=len(rallies),
    )

    all_matches: list[MatchResult] = []
    all_unmatched: list[dict] = []

    contact_config = ContactDetectionConfig()

    for rally in rallies:
        if not rally.ball_positions_json:
            continue

        ball_positions = [
            BallPos(
                frame_number=bp["frameNumber"],
                x=bp["x"], y=bp["y"],
                confidence=bp.get("confidence", 1.0),
            )
            for bp in rally.ball_positions_json
            if bp.get("x", 0) > 0 or bp.get("y", 0) > 0
        ]

        player_positions: list[PlayerPos] = []
        if rally.positions_json:
            player_positions = [
                PlayerPos(
                    frame_number=pp["frameNumber"],
                    track_id=pp["trackId"],
                    x=pp["x"], y=pp["y"],
                    width=pp["width"], height=pp["height"],
                    confidence=pp.get("confidence", 1.0),
                )
                for pp in rally.positions_json
            ]

        match_teams = match_teams_by_rally.get(rally.rally_id)

        contacts = detect_contacts(
            ball_positions=ball_positions,
            player_positions=player_positions,
            config=contact_config,
            net_y=rally.court_split_y,
            frame_count=rally.frame_count or None,
            team_assignments=match_teams,
            court_calibrator=calibrators.get(rally.video_id),
        )

        rally_actions, triggers = run_pipeline_with_rules(
            contacts, rally.rally_id,
            action_classifier=action_classifier,
            match_team_assignments=match_teams,
            disabled_rules=disabled_rules,
        )

        # Accumulate triggers
        for rule_id, count in triggers.items():
            result.triggers[rule_id] = result.triggers.get(rule_id, 0) + count

        pred_actions = [a.to_dict() for a in rally_actions.actions]
        real_pred = [a for a in pred_actions if not a.get("isSynthetic")]
        synth_serves = [
            a for a in pred_actions
            if a.get("isSynthetic") and a.get("action") == "serve"
        ]

        tolerance_frames = max(1, round(rally.fps * tolerance_ms / 1000))
        avail_tids: set[int] | None = None
        if rally.positions_json:
            avail_tids = {pp["trackId"] for pp in rally.positions_json}

        matches, unmatched = match_contacts(
            rally.gt_labels, real_pred,
            tolerance=tolerance_frames,
            available_track_ids=avail_tids,
            team_assignments=match_teams,
        )

        if synth_serves:
            synth_tol = max(tolerance_frames, round(rally.fps * 1.0))
            _match_synthetic_serves(
                matches, synth_serves, rally.gt_labels,
                synth_tol, avail_tids, match_teams,
            )

        all_matches.extend(matches)
        all_unmatched.extend(unmatched)

    metrics = compute_metrics(all_matches, all_unmatched)
    result.n_gt = metrics["total_gt"]
    result.n_tp = metrics["tp"]
    result.n_fp = metrics["fp"]
    result.n_fn = metrics["fn"]

    matched = [m for m in all_matches if m.pred_frame is not None]
    result.action_correct = sum(1 for m in matched if m.gt_action == m.pred_action)

    # Per-class accuracy
    for m in matched:
        gt = m.gt_action
        result.per_class_total[gt] = result.per_class_total.get(gt, 0) + 1
        if m.gt_action == m.pred_action:
            result.per_class_correct[gt] = result.per_class_correct.get(gt, 0) + 1

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-rule ablation of repair_action_sequence (LOO-CV)"
    )
    parser.add_argument("--only-video", type=str, help="Only run one fold (debug)")
    parser.add_argument("--tolerance-ms", type=int, default=167)
    parser.add_argument("--combos", action="store_true",
                        help="Test promising rule combinations instead of solo rules")
    args = parser.parse_args()

    t_total = time.monotonic()

    # --- Load data ---
    console.print("[bold]Loading rallies with action GT...[/bold]")
    rallies = load_rallies_with_action_gt()

    # Need temporal format for preextract (even though we don't use temporal model here)
    from scripts.train_temporal_attribution import (
        load_rallies_with_action_gt as load_rallies_temporal_format,
    )
    rallies_temporal = load_rallies_temporal_format()

    console.print(f"  {len(rallies)} rallies")

    # Group by video
    video_ids = sorted({r.video_id for r in rallies})
    rallies_by_video: dict[str, list[RallyData]] = defaultdict(list)
    for r in rallies:
        rallies_by_video[r.video_id].append(r)

    console.print(f"  {len(video_ids)} unique videos")

    if args.only_video:
        video_ids = [v for v in video_ids if v.startswith(args.only_video)]
        if not video_ids:
            console.print(f"[red]No video matching '{args.only_video}'[/red]")
            return
        console.print(f"  Filtered to {len(video_ids)} video(s)")

    # Load court calibrations
    calibrators: dict[str, CourtCalibrator | None] = {}
    all_video_ids = {r.video_id for r in rallies}
    for vid in all_video_ids:
        corners = load_court_calibration(vid)
        if corners and len(corners) == 4:
            cal = CourtCalibrator()
            cal.calibrate([(c["x"], c["y"]) for c in corners])
            calibrators[vid] = cal
        else:
            calibrators[vid] = None

    # Load match-level team assignments
    match_teams_by_rally = _load_match_team_assignments(all_video_ids, min_confidence=0.70)
    n_with_match = sum(1 for r in rallies if r.rally_id in match_teams_by_rally)
    console.print(f"  Match teams: {n_with_match}/{len(rallies)} rallies")

    # --- Pre-extract features ---
    cached_features = preextract_all_features(
        rallies, rallies_temporal, match_teams_by_rally,
    )

    # --- Build configs ---
    configs = build_configs(combos=args.combos)
    console.print(
        f"\n[bold]Running {len(configs)} configs × {len(video_ids)} folds[/bold]\n"
    )

    # --- LOO-CV ---
    results: dict[str, list[FoldResult]] = {c.name: [] for c in configs}

    for fold_idx, held_out_video in enumerate(video_ids):
        t_fold = time.monotonic()
        held_out_rallies = rallies_by_video[held_out_video]

        print(
            f"[{fold_idx + 1}/{len(video_ids)}] "
            f"{held_out_video[:8]} ({len(held_out_rallies)} rallies)",
            end="",
            flush=True,
        )

        # Train action GBM (same for all configs)
        action_clf = train_action_gbm_from_cache(cached_features, held_out_video)

        for cfg in configs:
            fold_result = evaluate_fold(
                held_out_rallies,
                action_classifier=action_clf,
                match_teams_by_rally=match_teams_by_rally,
                calibrators=calibrators,
                disabled_rules=cfg.disabled_rules,
                tolerance_ms=args.tolerance_ms,
            )
            fold_result.video_id = held_out_video
            results[cfg.name].append(fold_result)

        # Print progress (first config = no_repair baseline)
        fr = results["no_repair"][-1]
        action_acc = fr.action_correct / max(1, fr.n_tp)
        elapsed = time.monotonic() - t_fold
        print(f" → baseline_acc={action_acc:.0%} [{elapsed:.1f}s]", flush=True)

    # --- Aggregate ---
    print(flush=True)

    # Compute baseline (no_repair) totals
    baseline_folds = results["no_repair"]
    baseline_tp = sum(fr.n_tp for fr in baseline_folds)
    baseline_action_correct = sum(fr.action_correct for fr in baseline_folds)
    baseline_action_acc = baseline_action_correct / max(1, baseline_tp)

    # Summary table
    table = Table(title="Repair Rule Ablation (LOO-CV)")
    table.add_column("Config", style="cyan")
    table.add_column("Rules ON", justify="right")
    table.add_column("Triggers", justify="right")
    table.add_column("Contact F1", justify="right")
    table.add_column("Action Acc", justify="right")
    table.add_column("Δ Action", justify="right")

    for cfg in configs:
        folds = results[cfg.name]
        total_tp = sum(fr.n_tp for fr in folds)
        total_fp = sum(fr.n_fp for fr in folds)
        total_fn = sum(fr.n_fn for fr in folds)
        total_action_correct = sum(fr.action_correct for fr in folds)

        p = total_tp / max(1, total_tp + total_fp)
        r = total_tp / max(1, total_tp + total_fn)
        f1 = 2 * p * r / max(1e-9, p + r)
        action_acc = total_action_correct / max(1, total_tp)
        delta = action_acc - baseline_action_acc

        total_triggers = sum(
            sum(fr.triggers.values()) for fr in folds
        )

        rules_str = ",".join(str(r) for r in sorted(cfg.enabled_rules)) or "none"
        delta_str = f"{delta:+.1%}" if cfg.name != "no_repair" else "—"

        style = ""
        if cfg.name != "no_repair":
            style = "green" if delta > 0.005 else "red" if delta < -0.005 else "dim"

        table.add_row(
            cfg.name,
            rules_str,
            str(total_triggers),
            f"{f1:.1%}",
            f"{action_acc:.1%}",
            delta_str,
            style=style,
        )

    console.print(table)

    # Per-rule trigger detail table
    trigger_table = Table(title="Per-Rule Trigger Counts (all_rules config)")
    trigger_table.add_column("Rule", style="cyan")
    trigger_table.add_column("Description")
    trigger_table.add_column("Triggers", justify="right")

    all_rules_folds = results["all_rules"]
    for rule_id in sorted(ALL_RULES):
        total = sum(fr.triggers[rule_id] for fr in all_rules_folds)
        trigger_table.add_row(
            str(rule_id),
            RULE_NAMES[rule_id],
            str(total),
        )
    console.print(trigger_table)

    # Per-class accuracy comparison: no_repair vs all_rules
    class_table = Table(title="Per-Class Action Accuracy: no_repair vs all_rules")
    class_table.add_column("Action", style="cyan")
    class_table.add_column("no_repair", justify="right")
    class_table.add_column("all_rules", justify="right")
    class_table.add_column("Δ", justify="right")

    actions_list = ["serve", "receive", "set", "attack", "dig"]
    for action in actions_list:
        nr_correct = sum(fr.per_class_correct.get(action, 0) for fr in baseline_folds)
        nr_total = sum(fr.per_class_total.get(action, 0) for fr in baseline_folds)
        ar_correct = sum(
            fr.per_class_correct.get(action, 0) for fr in all_rules_folds
        )
        ar_total = sum(
            fr.per_class_total.get(action, 0) for fr in all_rules_folds
        )

        nr_acc = nr_correct / max(1, nr_total)
        ar_acc = ar_correct / max(1, ar_total)
        delta = ar_acc - nr_acc

        delta_str = f"{delta:+.1%}"
        style = "green" if delta > 0.01 else "red" if delta < -0.01 else "dim"
        class_table.add_row(
            action,
            f"{nr_acc:.1%} ({nr_correct}/{nr_total})",
            f"{ar_acc:.1%} ({ar_correct}/{ar_total})",
            delta_str,
            style=style,
        )
    console.print(class_table)

    elapsed = time.monotonic() - t_total
    console.print(f"\n[bold]Total time: {elapsed:.0f}s[/bold]")
    console.print(f"Baseline (no repair): {baseline_action_acc:.1%} action accuracy")


if __name__ == "__main__":
    main()
