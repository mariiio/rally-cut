"""Side-by-side smoke test of legacy vs parallel-decoder action paths.

Runs both `detect_contacts + classify_rally_actions` (legacy GBM path)
and `detect_contacts_via_decoder + build_rally_actions_from_decoder`
(parallel decoder path) on the SAME rallies, compares action sequences
frame-by-frame, and emits a side-by-side report. Watches dig specifically
per the code-review caveat.

Selection: 10 stable rallies covering diverse content (different
videos, different action mixes). Each rally is run with the SAME
ball/player/sequence_probs — only the contact/action emission changes.

Usage:
    cd analysis
    uv run python scripts/decoder_smoke_2026_04_24.py
    uv run python scripts/decoder_smoke_2026_04_24.py --n-rallies 20
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_rally(rally_id: str):  # type: ignore[no-untyped-def]
    """Load ball+player+seq_probs for a rally. Returns None if unavailable."""
    from rallycut.tracking.ball_tracker import BallPosition
    from rallycut.tracking.sequence_action_runtime import get_sequence_probs
    from scripts.eval_action_detection import (
        _build_player_positions,
        load_rallies_with_action_gt,
    )

    rallies = load_rallies_with_action_gt(rally_id=rally_id)
    if not rallies:
        return None
    rally = rallies[0]
    if not rally.ball_positions_json or not rally.positions_json:
        return None

    players = _build_player_positions(
        rally.positions_json, rally_id=rally.rally_id, inject_pose=True,
    )
    balls = [
        BallPosition(
            frame_number=bp["frameNumber"],
            x=bp["x"], y=bp["y"],
            confidence=bp.get("confidence", 1.0),
        )
        for bp in rally.ball_positions_json
    ]
    seq_probs = get_sequence_probs(
        balls, players, rally.court_split_y, rally.frame_count or 0, None,
    )
    return rally, balls, players, seq_probs


def _select_rally_ids(n: int) -> list[str]:
    """Pick `n` rallies stratified across videos."""
    from rallycut.evaluation.tracking.db import get_connection
    query = """
        SELECT DISTINCT ON (r.video_id) r.id::text
        FROM rallies r
        JOIN player_tracks pt ON pt.rally_id = r.id
        WHERE pt.action_ground_truth_json IS NOT NULL
          AND jsonb_array_length(pt.action_ground_truth_json) >= 4
        ORDER BY r.video_id, r.id::text
        LIMIT %s
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, [n])
            return [row[0] for row in cur.fetchall()]


def _run_legacy(rally, balls, players, seq_probs):  # type: ignore[no-untyped-def]
    """detect_contacts + classify_rally_actions — legacy GBM path."""
    from rallycut.tracking.action_classifier import classify_rally_actions
    from rallycut.tracking.contact_detector import (
        ContactDetectionConfig, detect_contacts,
    )
    from rallycut.tracking.decoder_runtime import run_decoder_for_production

    cfg = ContactDetectionConfig()
    contact_seq = detect_contacts(
        ball_positions=balls,
        player_positions=players,
        config=cfg,
        net_y=rally.court_split_y,
        frame_count=rally.frame_count or None,
        sequence_probs=seq_probs,
    )
    decoder_contacts = run_decoder_for_production(
        ball_positions=balls,
        player_positions=players,
        sequence_probs=seq_probs,
        contact_config=cfg,
    )
    rally_actions = classify_rally_actions(
        contact_seq, rally_id=rally.rally_id,
        sequence_probs=seq_probs,
        decoder_contacts=decoder_contacts,
    )
    return rally_actions


def _run_decoder(rally, balls, players, seq_probs):  # type: ignore[no-untyped-def]
    """detect_contacts_via_decoder + build_rally_actions_from_decoder."""
    from rallycut.tracking.contact_detector import (
        ContactDetectionConfig, detect_contacts_via_decoder,
    )
    from rallycut.tracking.decoder_actions import (
        build_rally_actions_from_decoder,
    )

    cfg = ContactDetectionConfig()
    contact_seq = detect_contacts_via_decoder(
        ball_positions=balls,
        player_positions=players,
        config=cfg,
        frame_count=rally.frame_count or None,
        sequence_probs=seq_probs,
    )
    return build_rally_actions_from_decoder(contact_seq)


def _summarize_actions(actions) -> dict:  # type: ignore[no-untyped-def]
    """Build a compact summary for side-by-side comparison."""
    by_action: Counter = Counter()
    by_player: Counter = Counter()
    synth_count = 0
    for a in actions:
        by_action[a.action_type.value] += 1
        if a.player_track_id >= 0:
            by_player[a.player_track_id] += 1
        if a.is_synthetic:
            synth_count += 1
    return {
        "n": len(actions),
        "sequence": [a.action_type.value for a in actions],
        "frames": [a.frame for a in actions],
        "tracks": [a.player_track_id for a in actions],
        "by_action": dict(by_action),
        "by_player": dict(by_player),
        "synth": synth_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-rallies", type=int, default=10)
    parser.add_argument("--out", type=str,
                        default="reports/decoder_smoke_2026_04_24.md")
    args = parser.parse_args()

    print(f"[smoke] Picking {args.n_rallies} stratified rallies...", flush=True)
    rally_ids = _select_rally_ids(args.n_rallies)
    print(f"[smoke] Got {len(rally_ids)} rallies", flush=True)

    rows: list[dict] = []
    legacy_action_totals: Counter = Counter()
    decoder_action_totals: Counter = Counter()

    for i, rid in enumerate(rally_ids, 1):
        loaded = _load_rally(rid)
        if loaded is None:
            print(f"[smoke] [{i}/{len(rally_ids)}] {rid}: skipped (no data)",
                  flush=True)
            continue
        rally, balls, players, seq_probs = loaded
        legacy = _run_legacy(rally, balls, players, seq_probs)
        decoder = _run_decoder(rally, balls, players, seq_probs)

        L = _summarize_actions(legacy.actions)
        D = _summarize_actions(decoder.actions)

        legacy_action_totals.update(L["by_action"])
        decoder_action_totals.update(D["by_action"])

        # Compare GT for ground truth
        gt_actions = [(g.frame, g.action) for g in sorted(rally.gt_labels, key=lambda g: g.frame)]

        rows.append({
            "rally_id": rid,
            "video_id": rally.video_id,
            "n_gt": len(gt_actions),
            "gt_seq": " → ".join(a for _, a in gt_actions),
            "legacy_seq": " → ".join(L["sequence"]),
            "decoder_seq": " → ".join(D["sequence"]),
            "legacy_n": L["n"],
            "decoder_n": D["n"],
            "legacy_dig": L["by_action"].get("dig", 0),
            "decoder_dig": D["by_action"].get("dig", 0),
            "legacy_serve": L["by_action"].get("serve", 0),
            "decoder_serve": D["by_action"].get("serve", 0),
            "decoder_synth": D["synth"],
        })
        print(f"[smoke] [{i}/{len(rally_ids)}] {rid}", flush=True)
        print(f"          GT     ({len(gt_actions):>2}): "
              f"{' → '.join(a for _, a in gt_actions)}", flush=True)
        print(f"          legacy ({L['n']:>2}): {' → '.join(L['sequence'])}",
              flush=True)
        print(f"          decode ({D['n']:>2}): {' → '.join(D['sequence'])}",
              flush=True)

    # Aggregate report
    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = ["# Parallel decoder smoke test — side-by-side vs legacy"]
    lines.append("")
    lines.append(f"- Rallies: {len(rows)}")
    lines.append("- Both paths share ball/player positions + MS-TCN++ probs")
    lines.append("- Legacy: `detect_contacts` + `run_decoder_for_production` overlay + `classify_rally_actions`")
    lines.append("- Decoder: `detect_contacts_via_decoder` + `build_rally_actions_from_decoder`")
    lines.append("")

    lines.append("## Aggregate action counts")
    lines.append("")
    lines.append("| Class | Legacy | Decoder | Δ |")
    lines.append("|---|---:|---:|---:|")
    for cls in ("serve", "receive", "set", "attack", "dig", "block", "unknown"):
        l_n = legacy_action_totals.get(cls, 0)
        d_n = decoder_action_totals.get(cls, 0)
        delta = d_n - l_n
        lines.append(f"| {cls} | {l_n} | {d_n} | {delta:+d} |")
    lines.append("")

    lines.append("## Per-rally side-by-side")
    lines.append("")
    for row in rows:
        lines.append(f"### `{row['rally_id'][:8]}…` ({row['video_id'][:8]}…) "
                     f"— {row['n_gt']} GT contacts")
        lines.append("")
        lines.append(f"- **GT**: {row['gt_seq']}")
        lines.append(f"- **Legacy** ({row['legacy_n']}): {row['legacy_seq']}")
        lines.append(f"- **Decoder** ({row['decoder_n']}): {row['decoder_seq']}")
        lines.append("")
        lines.append(
            f"- serve: legacy {row['legacy_serve']} vs decoder {row['decoder_serve']}; "
            f"dig: legacy {row['legacy_dig']} vs decoder {row['decoder_dig']}; "
            f"decoder synth-serves: {row['decoder_synth']}"
        )
        lines.append("")

    out_path.write_text("\n".join(lines))
    print(f"[smoke] Wrote {out_path}", flush=True)

    # Compact stdout verdict
    print()
    print("=" * 72)
    print("SMOKE VERDICT")
    print("=" * 72)
    print(f"{'Class':<10} {'Legacy':>8} {'Decoder':>8} {'Δ':>6}")
    for cls in ("serve", "receive", "set", "attack", "dig", "block"):
        l_n = legacy_action_totals.get(cls, 0)
        d_n = decoder_action_totals.get(cls, 0)
        delta = d_n - l_n
        marker = ""
        if cls == "dig" and delta < -2:
            marker = "  ⚠ dig regression watch"
        elif cls in ("serve", "receive") and delta > 0:
            marker = "  ✓ expected lift"
        print(f"{cls:<10} {l_n:>8} {d_n:>8} {delta:>+6d}{marker}")


if __name__ == "__main__":
    main()
