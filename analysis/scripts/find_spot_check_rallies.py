"""Find worst-attribution rallies for visual spot-check of positions_json drift.

Reproduces production_eval's per-rally loop (baseline ctx) and prints the
worst-attribution rallies with video filename + start/end timestamps so the
user can open the video, seek to the rally, and eyeball the tracking overlay.

Read-only. No DB writes. Diagnostic script only.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from eval_action_detection import (  # noqa: E402
    load_rallies_with_action_gt,
    match_contacts,
)
from production_eval import (  # noqa: E402
    PipelineContext,
    _build_calibrators,
    _load_match_team_assignments,
    _load_track_to_player_maps,
    _parse_positions,
    _run_rally,
    _tolerance_frames,
)
from rallycut.evaluation.tracking.db import get_connection  # noqa: E402

console = Console()


def _fmt_ms(ms: int | None) -> str:
    if ms is None:
        return "--:--"
    s = ms // 1000
    return f"{s // 60:02d}:{s % 60:02d}"


def _load_video_filenames(video_ids: set[str]) -> dict[str, str]:
    if not video_ids:
        return {}
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, filename FROM videos WHERE id = ANY(%s)",
            (list(video_ids),),
        )
        return {vid: fn for vid, fn in cur.fetchall()}


def main() -> int:
    console.print("[cyan]loading eval rallies[/cyan]")
    rallies = load_rallies_with_action_gt()
    console.print(f"  loaded {len(rallies)} rallies")

    video_ids = {r.video_id for r in rallies}
    console.print(f"  across {len(video_ids)} videos")

    console.print("[cyan]building calibrators + team maps + t2p[/cyan]")
    rally_pos_lookup = {
        r.rally_id: _parse_positions(r.positions_json) for r in rallies if r.positions_json
    }
    team_map = _load_match_team_assignments(video_ids, rally_positions=rally_pos_lookup)
    t2p_by_rally = _load_track_to_player_maps(video_ids)
    calibrators = _build_calibrators(video_ids)

    ctx = PipelineContext()  # baseline — all stages active

    per_rally: list[dict] = []
    console.print("[cyan]running production pipeline per rally[/cyan]")
    for idx, rally in enumerate(rallies, start=1):
        if not rally.ball_positions_json or not rally.positions_json:
            continue
        if not rally.frame_count or rally.frame_count < 10:
            continue
        try:
            pred_actions, _rally_actions = _run_rally(
                rally,
                team_map.get(rally.rally_id),
                calibrators.get(rally.video_id),
                ctx,
            )
        except Exception as exc:  # noqa: BLE001
            console.print(f"  [red][{idx}/{len(rallies)}][/red] {rally.rally_id}: {type(exc).__name__}: {exc}")
            continue

        real_pred = [a for a in pred_actions if not a.get("isSynthetic")]
        raw_avail = {pp["trackId"] for pp in rally.positions_json}
        rally_t2p = t2p_by_rally.get(rally.rally_id) or None
        if rally_t2p:
            avail = {rally_t2p.get(tid, tid) for tid in raw_avail}
        else:
            avail = raw_avail

        matches, _ = match_contacts(
            rally.gt_labels,
            real_pred,
            tolerance=_tolerance_frames(rally.fps),
            available_track_ids=avail,
            team_assignments=team_map.get(rally.rally_id),
            track_id_map=rally_t2p,
        )

        attr_pool = [m for m in matches if m.pred_frame is not None and m.player_evaluable]
        if not attr_pool:
            continue
        correct = sum(1 for m in attr_pool if m.player_correct)
        n_contacts = len(attr_pool)
        per_rally.append({
            "rally_id": rally.rally_id,
            "video_id": rally.video_id,
            "start_ms": rally.start_ms,
            "end_ms": rally.start_ms + int((rally.frame_count / rally.fps) * 1000) if rally.fps else None,
            "n_contacts": n_contacts,
            "correct": correct,
            "attr_acc": correct / n_contacts,
        })

        if idx % 40 == 0 or idx == len(rallies):
            console.print(f"  [{idx}/{len(rallies)}] processed")

    console.print(f"\n  {len(per_rally)} rallies with evaluable attribution matches")

    filenames = _load_video_filenames({r["video_id"] for r in per_rally})

    # Pick spot-check candidates:
    # - 5 worst: attribution accuracy low, ≥4 contacts so it's not single-contact noise
    # - 3 best: attribution 100%, ≥4 contacts (control group)
    substantive = [r for r in per_rally if r["n_contacts"] >= 4]
    substantive.sort(key=lambda r: (r["attr_acc"], -r["n_contacts"]))
    worst = substantive[:5]
    best = [r for r in reversed(substantive) if r["attr_acc"] >= 0.99][:3]

    def _render(title: str, rows: list[dict]) -> None:
        t = Table(title=title, show_lines=False)
        t.add_column("#", justify="right")
        t.add_column("video filename")
        t.add_column("rally_id (short)")
        t.add_column("start", justify="right")
        t.add_column("end", justify="right")
        t.add_column("contacts", justify="right")
        t.add_column("correct", justify="right")
        t.add_column("attr_acc", justify="right")
        for i, r in enumerate(rows, start=1):
            fn = filenames.get(r["video_id"], r["video_id"])
            t.add_row(
                str(i),
                fn,
                r["rally_id"][:8],
                _fmt_ms(r["start_ms"]),
                _fmt_ms(r["end_ms"]),
                str(r["n_contacts"]),
                str(r["correct"]),
                f"{r['attr_acc']*100:.0f}%",
            )
        console.print(t)

    _render("WORST player_attribution rallies (≥4 contacts) — spot check for drift", worst)
    _render("BEST player_attribution rallies (control group) — should look clean", best)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
