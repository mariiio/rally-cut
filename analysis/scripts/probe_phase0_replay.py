"""Phase 0 fixture-level integration check for scratchpad replay.

Calls the production `match_players_across_rallies` flow, then takes the
scratchpad it produced and replays Pass 2 stages 1+2 with the same final
profiles via `replay_refine_from_scratchpad`. Asserts byte-identical
trackToPlayer per rally.

This validates the end-to-end Phase 0 wiring on real video fixtures —
the unit tests exercise the same code path on synthetic data.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

_ANALYSIS_DIR = Path(__file__).resolve().parents[1]
if str(_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS_DIR))


FIXTURE_REGISTRY = Path(
    "/Users/mario/Personal/Projects/RallyCut/analysis/reports/"
    "attribution_rebuild/fixture_video_ids_2026_04_24.json"
)


def load_fixture_ids(names: list[str]) -> list[tuple[str, str]]:
    data = json.loads(FIXTURE_REGISTRY.read_text())
    fixtures = data["fixtures"]
    out: list[tuple[str, str]] = []
    for nm in names:
        if nm not in fixtures:
            raise SystemExit(
                f"Unknown fixture {nm!r}. Known: {sorted(fixtures)}"
            )
        out.append((nm, fixtures[nm]["video_id"]))
    return out


def probe_fixture(fixture_name: str, video_id: str) -> dict[str, Any]:
    print(f"\n=== Fixture: {fixture_name} (video_id={video_id[:8]}...) ===",
          flush=True)

    from rallycut.court.calibration import CourtCalibrator
    from rallycut.evaluation.db import get_connection
    from rallycut.evaluation.tracking.db import (
        get_video_path,
        load_rallies_for_video,
    )
    from rallycut.tracking.match_tracker import (
        match_players_across_rallies,
        replay_refine_from_scratchpad,
    )

    rallies = load_rallies_for_video(video_id)
    if not rallies:
        raise SystemExit(f"No rallies for {fixture_name}")
    print(f"  loaded {len(rallies)} rallies", flush=True)

    video_path = get_video_path(video_id)
    if video_path is None:
        raise SystemExit(f"no video path for {fixture_name}")
    print(f"  video: {video_path.name}", flush=True)

    # Court calibration (mirrors the match_players CLI path).
    court_calibrator = None
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT court_calibration_json FROM videos WHERE id = %s",
                [video_id],
            )
            cal_row = cur.fetchone()
    if (
        cal_row
        and cal_row[0]
        and isinstance(cal_row[0], list)
        and len(cal_row[0]) == 4
    ):
        court_calibrator = CourtCalibrator()
        court_calibrator.calibrate([(c["x"], c["y"]) for c in cal_row[0]])
        if not court_calibrator.is_calibrated:
            court_calibrator = None

    # Live path: the new wiring populates result.scratchpad.
    t0 = time.time()
    match_result = match_players_across_rallies(
        video_path=video_path,
        rallies=rallies,
        num_samples=12,
        reference_profiles=None,
        reid_model=None,
        calibrator=court_calibrator,
    )
    elapsed = time.time() - t0
    n = len(match_result.rally_results)
    print(f"  live run: {elapsed:.1f}s, {n} rallies, "
          f"sideSwitches={match_result.scratchpad.get('sideSwitches')}",
          flush=True)

    # Replay path: same scratchpad, same profiles.
    t1 = time.time()
    replay_results = replay_refine_from_scratchpad(
        scratchpad=match_result.scratchpad,
        player_profiles=match_result.player_profiles,
        initial_results=match_result.rally_results,
    )
    replay_elapsed = time.time() - t1
    print(f"  replay: {replay_elapsed:.2f}s", flush=True)

    # Compare per-rally trackToPlayer.
    identical = 0
    diffs: list[dict[str, Any]] = []
    for i, (live, replay) in enumerate(zip(match_result.rally_results, replay_results)):
        if live.track_to_player == replay.track_to_player:
            identical += 1
            continue
        all_tids = set(live.track_to_player) | set(replay.track_to_player)
        diffs.append({
            "rally_index": i,
            "changes": [
                {
                    "track_id": tid,
                    "live_pid": live.track_to_player.get(tid),
                    "replay_pid": replay.track_to_player.get(tid),
                }
                for tid in sorted(all_tids)
                if live.track_to_player.get(tid) != replay.track_to_player.get(tid)
            ],
        })

    print(
        f"  identical: {identical}/{n} "
        f"({100.0 * identical / n:.1f}%)",
        flush=True,
    )
    for d in diffs:
        print(
            f"    rally {d['rally_index']}: {len(d['changes'])} track diff(s): "
            f"{d['changes']}",
            flush=True,
        )

    return {
        "fixture": fixture_name,
        "video_id": video_id,
        "n_rallies": n,
        "identical": identical,
        "diverging": len(diffs),
        "live_elapsed_s": elapsed,
        "replay_elapsed_s": replay_elapsed,
        "side_switches": match_result.scratchpad.get("sideSwitches", []),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fixtures",
        nargs="+",
        default=["tata", "cuco", "rere"],
        help="Fixture names from fixture_video_ids_2026_04_24.json",
    )
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.WARNING if not args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("rallycut.tracking.reid_general").setLevel(logging.ERROR)
    logging.getLogger("rallycut.tracking.reid_embeddings").setLevel(logging.ERROR)

    pairs = load_fixture_ids(args.fixtures)
    print(f"Probing {len(pairs)} fixture(s) via Phase 0 wiring: "
          f"{[p[0] for p in pairs]}",
          flush=True)

    summaries = []
    for nm, vid in pairs:
        try:
            summaries.append(probe_fixture(nm, vid))
        except Exception as e:
            import traceback
            traceback.print_exc()
            summaries.append({"fixture": nm, "video_id": vid, "error": str(e)})

    print("\n=== VERDICT ===", flush=True)
    total_rallies = 0
    total_identical = 0
    for s in summaries:
        if "error" in s:
            print(f"  {s['fixture']}: ERROR — {s['error']}", flush=True)
            continue
        total_rallies += s["n_rallies"]
        total_identical += s["identical"]
        pct = 100.0 * s["identical"] / s["n_rallies"] if s["n_rallies"] else 0
        marker = "PASS" if s["identical"] == s["n_rallies"] else "DIVERGE"
        print(
            f"  {s['fixture']}: {s['identical']}/{s['n_rallies']} "
            f"({pct:.1f}%) [{marker}] "
            f"sideSwitches={s['side_switches']} "
            f"replay={s['replay_elapsed_s']:.2f}s",
            flush=True,
        )
    if total_rallies:
        agg_pct = 100.0 * total_identical / total_rallies
        print(
            f"\n  AGGREGATE: {total_identical}/{total_rallies} "
            f"({agg_pct:.2f}%) rallies identical",
            flush=True,
        )


if __name__ == "__main__":
    main()
