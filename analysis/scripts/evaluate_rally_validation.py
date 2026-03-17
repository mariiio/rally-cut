"""
Dry-run evaluation of post-tracking rally validation (ball-pass FP filter).

Queries all tracked videos, applies the same validation logic as the API,
and reports which rallies would be demoted — without modifying the DB.

Usage:
    uv run python analysis/scripts/evaluate_rally_validation.py [--video-id ID]
"""

from __future__ import annotations

import argparse

from psycopg.rows import dict_row

from rallycut.evaluation.db import get_connection


def validate_rally(rally: dict) -> tuple[int, list[str]]:
    """Compute disqualification score for a rally. Returns (score, reasons)."""
    reasons: list[str] = []

    # Extract signals
    contacts_data = rally.get("contacts_json") or {}
    contacts = contacts_data.get("contacts", [])
    validated = [c for c in contacts if c.get("isValidated") is not False]
    contact_count = len(validated)

    actions_data = rally.get("actions_json") or {}
    action_seq = actions_data.get("actionSequence", [])
    actions_list = actions_data.get("actions", [])
    has_serve = any(a == "serve" for a in action_seq) or any(
        a.get("action") == "serve" for a in actions_list
    )

    duration_s = (rally["end_ms"] - rally["start_ms"]) / 1000.0

    score = 0

    if contact_count == 0:
        score += 3
        reasons.append("0 contacts (+3)")
    elif contact_count == 1:
        score += 1
        reasons.append("1 contact (+1)")

    has_receive = any(a == "receive" for a in action_seq) or any(
        a.get("action") == "receive" for a in actions_list
    )

    if not has_serve:
        score += 2
        reasons.append("no serve (+2)")

    if not has_receive:
        score += 1
        reasons.append("no receive (+1)")

    if duration_s < 6 and contact_count < 2:
        score += 1
        reasons.append(f"short+sparse ({duration_s:.1f}s, {contact_count} contacts) (+1)")

    return score, reasons


def main() -> None:
    parser = argparse.ArgumentParser(description="Dry-run rally validation")
    parser.add_argument("--video-id", help="Evaluate a single video")
    parser.add_argument("--threshold", type=int, default=4, help="Demotion threshold (default: 4)")
    args = parser.parse_args()

    with get_connection() as conn:
        cur = conn.cursor(row_factory=dict_row)

        # Build query — where_clause is a controlled literal, not user input
        where_clause = ""
        params: list = []
        if args.video_id:
            where_clause = "AND r.video_id = %s"
            params = [args.video_id]

        cur.execute(
            f"""
            SELECT r.id as rally_id, r.video_id, r.start_ms, r.end_ms, r.status,
                   r.rejection_reason, r.score_a, r.score_b, r.notes, r.serving_team,
                   pt.contacts_json, pt.actions_json, pt.quality_report_json,
                   pt.detection_rate
            FROM rallies r
            JOIN player_tracks pt ON pt.rally_id = r.id AND pt.status = 'COMPLETED'
            WHERE r.status = 'CONFIRMED' {where_clause}
            ORDER BY r.video_id, r.start_ms
            """,
            params,
        )
        rallies = cur.fetchall()

    if not rallies:
        print("No tracked CONFIRMED rallies found.")
        return

    # Group by video
    videos: dict[str, list] = {}
    for r in rallies:
        videos.setdefault(r["video_id"], []).append(r)

    total_rallies = 0
    total_demoted = 0
    total_skipped = 0
    demoted_details = []

    for vid, vid_rallies in sorted(videos.items()):
        demoted_in_video = 0

        for r in vid_rallies:
            total_rallies += 1

            # Safety gate: user-modified
            if (
                r["score_a"] is not None
                or r["score_b"] is not None
                or r["notes"] is not None
                or r["serving_team"] is not None
            ):
                total_skipped += 1
                continue

            # Safety gate: low detection rate
            qr = r.get("quality_report_json") or {}
            det_rate = qr.get("detectionRate") or r.get("detection_rate")
            if det_rate is not None and det_rate < 0.15:
                total_skipped += 1
                continue

            score, reasons = validate_rally(r)
            duration_s = (r["end_ms"] - r["start_ms"]) / 1000.0

            if score >= args.threshold:
                total_demoted += 1
                demoted_in_video += 1
                contacts_data = r.get("contacts_json") or {}
                contact_count = len(
                    [c for c in contacts_data.get("contacts", []) if c.get("isValidated") is not False]
                )
                demoted_details.append({
                    "video_id": vid[:8],
                    "rally_id": r["rally_id"][:8],
                    "score": score,
                    "duration_s": duration_s,
                    "contacts": contact_count,
                    "reasons": ", ".join(reasons),
                })

        if demoted_in_video > 0:
            print(f"Video {vid[:8]}: {demoted_in_video}/{len(vid_rallies)} would be demoted")

    print(f"\n{'='*70}")
    print(f"Summary: {total_demoted}/{total_rallies} rallies would be demoted (threshold={args.threshold})")
    print(f"  Skipped (user-modified or low detection): {total_skipped}")
    print(f"  Videos evaluated: {len(videos)}")

    if demoted_details:
        print("\nDemoted rallies:")
        print(f"{'Video':>10} {'Rally':>10} {'Score':>6} {'Dur(s)':>7} {'Contacts':>9}  Reasons")
        print(f"{'-'*10} {'-'*10} {'-'*6} {'-'*7} {'-'*9}  {'-'*30}")
        for d in demoted_details:
            print(
                f"{d['video_id']:>10} {d['rally_id']:>10} {d['score']:>6} "
                f"{d['duration_s']:>7.1f} {d['contacts']:>9}  {d['reasons']}"
            )


if __name__ == "__main__":
    main()
