"""Derive per-panel-rally ground truth `(track_id → player_id)` mapping.

Reads the production-stored `matchAnalysisJson.trackToPlayer` from the videos
table and inverts it according to each panel rally's documented error shape.

Usage (as a library):
    from forensic_panel_ground_truth import build_panel_gt
    gt = build_panel_gt()  # dict: rally_tag → {track_id: player_id}

Run as a script to print an inversion sanity table for user verification.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

_ANALYSIS_DIR = Path(__file__).resolve().parents[1]
if str(_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS_DIR))

from rallycut.evaluation.tracking.db import get_connection  # noqa: E402

# Panel mirror — kept here so the GT script is independent of the BoxMOT
# capture script. (video_id, rally_idx_0based, is_error, gt_shape, args).
# `gt_shape` is one of:
#   "clean":   production trackToPlayer IS the GT (controls + uncorrected errors).
#   "swap":    swap the two pids in `args` for the two tracks production
#              currently assigns to those pids.
#   "cycle":   apply an explicit 4-cycle. `args` is dict {production_pid → gt_pid}
#              meaning "the track that production assigned to production_pid
#              should actually be gt_pid".
PANEL: list[tuple[str, int, bool, str, dict[str, Any]]] = [
    ("7d77980f-3006-40e0-adc0-db491a5bb659", 1, True, "swap", {"a": 1, "b": 4}),  # p1 starts on p4 — out-of-scope; treated as swap for completeness
    ("7d77980f-3006-40e0-adc0-db491a5bb659", 12, True, "swap", {"a": 2, "b": 4}),
    ("7d77980f-3006-40e0-adc0-db491a5bb659", 18, True, "swap", {"a": 2, "b": 4}),
    ("7d77980f-3006-40e0-adc0-db491a5bb659", 0, False, "clean", {}),
    # b5fb0594 r10: "full 4-cycle (p4→1, p3→2, p2→4, p1→3)".
    # Production produces (track_x → 1, track_y → 2, track_z → 4, track_w → 3) for
    # the four primary tracks. The user's description spells:
    #   "the track production calls p1 actually is p3"
    #   "the track production calls p2 actually is p4"
    #   "the track production calls p3 actually is p2"
    #   "the track production calls p4 actually is p1"
    ("b5fb0594-d64f-4a0d-bad9-de8fc36414d0", 9, True, "cycle",
     {"prod_to_gt": {1: 3, 2: 4, 3: 2, 4: 1}}),
    ("b5fb0594-d64f-4a0d-bad9-de8fc36414d0", 3, True, "swap", {"a": 3, "b": 4}),
    ("b5fb0594-d64f-4a0d-bad9-de8fc36414d0", 5, True, "swap", {"a": 3, "b": 4}),
    ("b5fb0594-d64f-4a0d-bad9-de8fc36414d0", 0, False, "clean", {}),
    ("5c756c41-1cc1-4486-a95c-97398912cfbe", 6, True, "swap", {"a": 1, "b": 4}),
    ("5c756c41-1cc1-4486-a95c-97398912cfbe", 2, True, "swap", {"a": 2, "b": 4}),
    ("5c756c41-1cc1-4486-a95c-97398912cfbe", 0, False, "clean", {}),
    ("854bb250-3e91-47d2-944d-f62413e3cf45", 0, True, "swap", {"a": 2, "b": 4}),
    ("854bb250-3e91-47d2-944d-f62413e3cf45", 1, False, "clean", {}),
]


@dataclass
class RallyGT:
    rally_tag: str
    rally_id: str
    short_id: str
    is_error: bool
    gt_shape: str
    production_track_to_player: dict[int, int]  # what DB stores
    gt_track_to_player: dict[int, int]  # what we infer is correct
    production_source: str  # "trackToPlayer" / "appliedFullMapping" / etc.


def _resolve_rally_id(
    cur: Any, video_id: str, rally_idx: int
) -> tuple[str, int, int] | None:
    """Resolve (rally_id, start_ms, end_ms) for the rally at the given idx.

    Same ordering as forensic_capture_panel.py's `resolve_panel_to_rally_ids`.
    """
    cur.execute(
        """
        SELECT r.id::text, r.start_ms, r.end_ms
        FROM rallies r
        JOIN player_tracks pt ON pt.rally_id = r.id
        WHERE r.video_id = %s
          AND pt.positions_json IS NOT NULL
        ORDER BY r.start_ms
        """,
        [video_id],
    )
    rows = cur.fetchall()
    if rally_idx >= len(rows):
        return None
    rid, start_ms, end_ms = rows[rally_idx]
    return cast(str, rid), int(start_ms), int(end_ms)


def _load_track_to_player(
    cur: Any, video_id: str, rally_id: str
) -> tuple[dict[int, int], str]:
    """Read production's canonical track→pid mapping for the rally.

    Prefers `trackToPlayer` (cross-rally Hungarian output, the stage-9b
    target) when it covers all 4 primary tracks. When trackToPlayer is
    incomplete (Hungarian declined to assign some tracks: cost > MAX),
    falls back to `appliedFullMapping` (post-remap-track-ids state, what
    the user sees in the visual editor; stored `{pid_str: track_id}`).
    Restricts to primary_track_ids and pids 1..4. Returns (mapping, source).
    """
    cur.execute(
        "SELECT match_analysis_json FROM videos WHERE id = %s", [video_id],
    )
    row = cur.fetchone()
    if not row or not row[0]:
        return {}, "none"
    payload = row[0] if isinstance(row[0], dict) else json.loads(row[0])
    rallies = payload.get("rallies", []) or []

    cur.execute(
        "SELECT primary_track_ids FROM player_tracks WHERE rally_id = %s",
        [rally_id],
    )
    pt_row = cur.fetchone()
    primary_set: set[int] | None = None
    if pt_row and pt_row[0] is not None:
        ptids = pt_row[0] if isinstance(pt_row[0], list) else json.loads(pt_row[0])
        primary_set = {int(t) for t in ptids}

    def _filter(raw: dict[int, int]) -> dict[int, int]:
        out: dict[int, int] = {}
        for tid, pid in raw.items():
            if pid < 1 or pid > 4:
                continue
            if primary_set is not None and tid not in primary_set:
                continue
            out[tid] = pid
        return out

    for entry in rallies:
        rid = entry.get("rallyId") or entry.get("rally_id")
        if rid != rally_id:
            continue
        ttp_raw = entry.get("trackToPlayer", {}) or {}
        ttp = _filter({int(k): int(v) for k, v in ttp_raw.items()})
        n_primary = len(primary_set) if primary_set else 4
        if len(ttp) >= min(4, n_primary):
            return ttp, "trackToPlayer"
        # Hungarian was incomplete; fall back to appliedFullMapping.
        applied = entry.get("appliedFullMapping") or {}
        if applied:
            inv = {int(tid): int(pid) for pid, tid in applied.items()}
            inv_filtered = _filter(inv)
            return inv_filtered, "appliedFullMapping"
        return ttp, "trackToPlayer-partial"
    return {}, "none"


def apply_swap(
    production: dict[int, int], pid_a: int, pid_b: int
) -> dict[int, int]:
    """Swap the two tracks production assigned to pid_a and pid_b. The
    correct GT swaps the pids back."""
    inv = {pid: tid for tid, pid in production.items()}
    track_a = inv.get(pid_a)
    track_b = inv.get(pid_b)
    if track_a is None or track_b is None:
        return dict(production)
    out = dict(production)
    out[track_a] = pid_b
    out[track_b] = pid_a
    return out


def apply_cycle(
    production: dict[int, int], prod_to_gt: dict[int, int]
) -> dict[int, int]:
    """Apply an explicit production-pid → gt-pid relabel. The track
    production assigned to `production_pid` actually should be
    `gt_pid` per user verdict."""
    out: dict[int, int] = {}
    for tid, prod_pid in production.items():
        out[tid] = prod_to_gt.get(prod_pid, prod_pid)
    return out


def derive_gt_from_production(
    production: dict[int, int], gt_shape: str, args: dict[str, Any]
) -> dict[int, int]:
    """Apply panel gt_shape inversion to a given production track→pid map.

    Used by the analyzer to derive GT in retrack-ID space (production map
    comes from the sidecar's `9b_synthetic_first_rally` record) without
    going through DB.
    """
    if gt_shape == "swap":
        return apply_swap(production, int(args["a"]), int(args["b"]))
    if gt_shape == "cycle":
        prod_to_gt = {int(k): int(v) for k, v in args["prod_to_gt"].items()}
        return apply_cycle(production, prod_to_gt)
    return dict(production)


def panel_descriptions() -> list[dict[str, Any]]:
    """Panel description metadata, decoupled from DB lookup. Used by the
    analyzer to map rally_tag → (is_error, gt_shape, args)."""
    out: list[dict[str, Any]] = []
    for video_id, rally_idx, is_error, gt_shape, args in PANEL:
        short_id = video_id[:8]
        rally_tag = f"{short_id}/r{rally_idx + 1:02d}"
        out.append({
            "rally_tag": rally_tag,
            "video_id": video_id,
            "rally_idx": rally_idx,
            "is_error": is_error,
            "gt_shape": gt_shape,
            "args": args,
        })
    return out


def build_panel_gt() -> list[RallyGT]:
    """Build per-rally GT for the panel by reading production trackToPlayer
    + applying the per-rally inversion shape."""
    out: list[RallyGT] = []
    with get_connection() as conn, conn.cursor() as cur:
        for video_id, rally_idx, is_error, gt_shape, args in PANEL:
            short_id = video_id[:8]
            rally_tag = f"{short_id}/r{rally_idx + 1:02d}"
            resolved = _resolve_rally_id(cur, video_id, rally_idx)
            if resolved is None:
                continue
            rally_id, _start_ms, _end_ms = resolved
            production, source = _load_track_to_player(cur, video_id, rally_id)

            if gt_shape == "clean":
                gt = dict(production)
            elif gt_shape == "swap":
                gt = apply_swap(production, args["a"], args["b"])
            elif gt_shape == "cycle":
                gt = apply_cycle(production, args["prod_to_gt"])
            else:
                gt = dict(production)

            out.append(RallyGT(
                rally_tag=rally_tag,
                rally_id=rally_id,
                short_id=short_id,
                is_error=is_error,
                gt_shape=gt_shape,
                production_track_to_player=production,
                gt_track_to_player=gt,
                production_source=source,
            ))
    return out


def main() -> None:
    """Print inversion sanity table for user verification."""
    panel_gt = build_panel_gt()
    print(f"Resolved {len(panel_gt)} of {len(PANEL)} panel entries")
    print()
    print(f"{'Rally':<14} {'Class':<8} {'Shape':<8} {'Src':<22} Production → GT")
    print("-" * 130)
    for entry in panel_gt:
        kind = "ERROR" if entry.is_error else "CONTROL"
        prod_str = "{" + ", ".join(
            f"t{tid}→p{pid}"
            for tid, pid in sorted(entry.production_track_to_player.items())
        ) + "}"
        gt_str = "{" + ", ".join(
            f"t{tid}→p{pid}"
            for tid, pid in sorted(entry.gt_track_to_player.items())
        ) + "}"
        diff = (
            sorted(
                tid
                for tid in entry.production_track_to_player
                if entry.production_track_to_player[tid]
                != entry.gt_track_to_player.get(tid)
            )
        )
        diff_str = f" Δ={diff}" if diff else ""
        print(f"{entry.rally_tag:<14} {kind:<8} {entry.gt_shape:<8} "
              f"{entry.production_source:<22} "
              f"{prod_str} → {gt_str}{diff_str}")


if __name__ == "__main__":
    main()
