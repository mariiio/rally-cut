#!/usr/bin/env python3
"""Experiment 3: End-to-end action classification pipeline.

Runs ball tracking → contact detection → action classification on rallies
with stored ball tracking data and prints the action sequences.

Usage:
    # Run on all rallies with ball GT
    uv run python scripts/run_action_classification.py

    # Run on specific rally
    uv run python scripts/run_action_classification.py --rally-id bd77efd1

    # Save results
    uv run python scripts/run_action_classification.py -o action_results.json

    # Verbose output with contact details
    uv run python scripts/run_action_classification.py --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Add analysis root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline(
    rallies: list[Any],
    verbose: bool = False,
) -> dict[str, Any]:
    """Run contact detection + action classification on all rallies.

    Args:
        rallies: List of TrackingEvaluationRally from database.
        verbose: Print detailed contact/action info.

    Returns:
        Dict with per-rally and aggregate results.
    """
    from rallycut.tracking.action_classifier import (
        ActionClassifier,
        ActionType,
        RallyActions,
    )
    from rallycut.tracking.contact_detector import ContactDetectionConfig, detect_contacts

    config = ContactDetectionConfig()
    classifier = ActionClassifier()

    per_rally: dict[str, dict[str, Any]] = {}
    all_actions: list[RallyActions] = []

    # Action type counters
    action_counts: dict[str, int] = {t.value: 0 for t in ActionType}

    for i, rally in enumerate(rallies):
        rally_id_short = rally.rally_id[:8]

        # Get ball positions from predictions
        ball_positions = (
            rally.predictions.ball_positions
            if rally.predictions and rally.predictions.ball_positions
            else []
        )

        # Get player positions from predictions
        player_positions = (
            rally.predictions.positions
            if rally.predictions and rally.predictions.positions
            else []
        )

        if not ball_positions:
            print(f"  [{i+1}/{len(rallies)}] Rally {rally_id_short}: no ball positions, skipping")
            continue

        # Step 1: Contact detection
        contact_seq = detect_contacts(
            ball_positions=ball_positions,
            player_positions=player_positions if player_positions else None,
            config=config,
        )

        # Step 2: Action classification
        rally_actions = classifier.classify_rally(
            contact_sequence=contact_seq,
            rally_id=rally.rally_id,
        )
        all_actions.append(rally_actions)

        # Print summary
        action_seq = [a.action_type.value for a in rally_actions.actions]
        serve = rally_actions.serve

        print(
            f"  [{i+1}/{len(rallies)}] Rally {rally_id_short}: "
            f"{contact_seq.num_contacts} contacts → "
            f"{len(rally_actions.actions)} actions"
        )
        if action_seq:
            print(f"    Sequence: {' → '.join(action_seq)}")
        if serve:
            print(
                f"    Serve: player={serve.player_track_id} "
                f"side={serve.court_side} "
                f"vel={serve.velocity:.4f}"
            )

        # Count actions
        for action in rally_actions.actions:
            action_counts[action.action_type.value] += 1

        # Verbose: print each contact/action
        if verbose:
            print(f"    Net Y: {contact_seq.net_y:.3f}")
            for j, (contact, action) in enumerate(
                zip(contact_seq.contacts, rally_actions.actions)
            ):
                print(
                    f"    [{j}] frame={contact.frame:4d} "
                    f"ball=({contact.ball_x:.3f},{contact.ball_y:.3f}) "
                    f"vel={contact.velocity:.4f} "
                    f"dir_change={contact.direction_change_deg:.1f}° "
                    f"player={contact.player_track_id} "
                    f"dist={contact.player_distance:.3f} "
                    f"side={contact.court_side} "
                    f"→ {action.action_type.value} "
                    f"(conf={action.confidence:.1f})"
                )
            # If more contacts than actions (shouldn't happen), show extra contacts
            if contact_seq.num_contacts > len(rally_actions.actions):
                for j in range(len(rally_actions.actions), contact_seq.num_contacts):
                    contact = contact_seq.contacts[j]
                    print(
                        f"    [{j}] frame={contact.frame:4d} "
                        f"(unclassified contact)"
                    )

        per_rally[rally.rally_id] = {
            "rally_id": rally.rally_id,
            "video_id": rally.video_id,
            "num_ball_positions": len(ball_positions),
            "num_player_positions": len(player_positions),
            "num_contacts": contact_seq.num_contacts,
            "net_y": contact_seq.net_y,
            "num_actions": len(rally_actions.actions),
            "action_sequence": action_seq,
            "has_serve": rally_actions.serve is not None,
            "contacts": contact_seq.to_dict(),
            "actions": rally_actions.to_dict(),
        }

    # Aggregate stats
    total_rallies = len(per_rally)
    total_contacts = sum(r["num_contacts"] for r in per_rally.values())
    total_actions = sum(r["num_actions"] for r in per_rally.values())
    rallies_with_serve = sum(1 for r in per_rally.values() if r["has_serve"])

    aggregate = {
        "total_rallies": total_rallies,
        "total_contacts": total_contacts,
        "total_actions": total_actions,
        "rallies_with_serve": rallies_with_serve,
        "avg_contacts_per_rally": total_contacts / total_rallies if total_rallies > 0 else 0,
        "avg_actions_per_rally": total_actions / total_rallies if total_rallies > 0 else 0,
        "action_counts": action_counts,
    }

    return {
        "experiment": "exp-action-rules",
        "aggregate": aggregate,
        "per_rally": per_rally,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exp 3: End-to-end action classification pipeline"
    )
    parser.add_argument("-o", "--output", type=Path, help="Output JSON file")
    parser.add_argument("--rally-id", type=str, help="Test single rally (prefix match)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show contact details")
    args = parser.parse_args()

    from rallycut.evaluation.tracking.db import load_labeled_rallies

    print("Loading labeled rallies from database...")
    rallies = load_labeled_rallies()

    if args.rally_id:
        rallies = [r for r in rallies if r.rally_id.startswith(args.rally_id)]

    print(f"  Found {len(rallies)} rally(s) with ball tracking data\n")

    if not rallies:
        print("No rallies found!")
        return

    results = run_pipeline(rallies, verbose=args.verbose)

    # Print summary
    agg = results["aggregate"]
    print(f"\n{'='*60}")
    print("AGGREGATE RESULTS")
    print(f"{'='*60}")
    print(f"  Rallies analyzed:     {agg['total_rallies']}")
    print(f"  Total contacts:       {agg['total_contacts']}")
    print(f"  Total actions:        {agg['total_actions']}")
    print(f"  Rallies with serve:   {agg['rallies_with_serve']}")
    print(f"  Avg contacts/rally:   {agg['avg_contacts_per_rally']:.1f}")
    print(f"  Avg actions/rally:    {agg['avg_actions_per_rally']:.1f}")
    print("\n  Action distribution:")
    for action_type, count in sorted(
        agg["action_counts"].items(), key=lambda x: -x[1]
    ):
        if count > 0:
            print(f"    {action_type:<12} {count:4d}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
