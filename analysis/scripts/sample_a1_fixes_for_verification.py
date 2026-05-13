"""Sample 10 A1 fixes for human visual verification.

A1 "fix" = a C-4 violation pair that exists in the baseline catalog
(USE_VOLLEYBALL_RULE_ATTRIBUTION OFF) but does NOT exist in the A1-on
catalog (flag ON). Restricted to the spec-ship cohort
`curr_best_same_team_alt_ratio in [2.0, 50.0]`.

Stratified sampling:
  - At most 2 fixes per source video.
  - Mix of action-type pairs (target: cover attack/dig, set/attack,
    receive/set, dig/set when available).
  - Mix of alt_ratios across bands [2,3), [3,5), [5,10), [10,50).
  - Deterministic via fixed random seed.

For each sampled fix:
  - Look up source-video metadata (name, filename, fps, rally start_ms)
    in the local Postgres DB.
  - Recompute A1's choice in-process by re-running
    `reattribute_players` with USE_VOLLEYBALL_RULE_ATTRIBUTION=1 on the
    rally's stored actions_json + contacts_json.
  - Emit a markdown table + per-fix details into
    analysis/reports/a1_volleyball_rule/visual_verify_sample_2026_05_13.md.

Usage:
    cd analysis
    uv run python scripts/sample_a1_fixes_for_verification.py
"""
from __future__ import annotations

import csv
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# Add scripts dir + analysis to import path for reusable helpers.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

# Force A1 ON before importing reattribute_players so it's active when
# we call it. (The function reads the env at call-time, so we could
# also set it just before the call; doing it up-front is safer.)
os.environ.setdefault("USE_VOLLEYBALL_RULE_ATTRIBUTION", "1")

from rallycut.evaluation.tracking.db import get_connection  # noqa: E402
from rallycut.tracking.action_classifier import reattribute_players  # noqa: E402

from catalog_c4_violations_a1_inprocess import (  # noqa: E402
    _reconstruct_actions,
    _reconstruct_contacts,
    _team_assignments_int,
)

BASELINE_CSV = (
    HERE.parent
    / "reports/coherence_c4_catalog/2026-05-13_baseline_inprocess.csv"
)
A1_ON_CSV = HERE.parent / "reports/coherence_c4_catalog/2026-05-13_a1_on.csv"
OUTPUT_MD = (
    HERE.parent
    / "reports/a1_volleyball_rule/visual_verify_sample_2026_05_13.md"
)

ALT_RATIO_BANDS: list[tuple[float, float, str]] = [
    (2.0, 3.0, "[2,3)"),
    (3.0, 5.0, "[3,5)"),
    (5.0, 10.0, "[5,10)"),
    (10.0, 50.0, "[10,50]"),
]
TARGET_PER_BAND = 2  # 4 bands x 2 = 8; remaining 2 free-stratify.
TOTAL_TARGET = 10
MAX_PER_VIDEO = 2
SEED = 20260513


# ---------------------------------------------------------------------------
# Catalog loading
# ---------------------------------------------------------------------------

def _read_catalog(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def _pair_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return (
        row["rally_id"],
        int(row["frame_prev"]),
        int(row["frame_curr"]),
    )


def _parse_alt_ratio(val: str) -> float | None:
    if val is None or val == "" or val.lower() == "nan":
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _alt_ratio_band(r: float) -> str | None:
    for lo, hi, name in ALT_RATIO_BANDS:
        if lo <= r < hi:
            return name
    if 10.0 <= r <= 50.0:
        return "[10,50]"
    return None


def _stratified_sample(
    fixes: list[dict[str, Any]],
    *,
    seed: int = SEED,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)

    fixes_by_band: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fix in fixes:
        r = _parse_alt_ratio(fix["curr_best_same_team_alt_ratio"])
        if r is None:
            continue
        band = _alt_ratio_band(r)
        if band is None:
            continue
        fixes_by_band[band].append(fix)

    print("[sample] cohort distribution per band:", flush=True)
    for _lo, _hi, name in ALT_RATIO_BANDS:
        print(f"  band {name:9s}: {len(fixes_by_band.get(name, []))} fixes",
              flush=True)

    selected: list[dict[str, Any]] = []
    per_video: dict[str, int] = defaultdict(int)
    action_pair_counts: dict[tuple[str, str], int] = defaultdict(int)

    def _key(fix: dict[str, Any]) -> tuple[int, int, float]:
        """Sort key: prefer videos not yet sampled + new action-type pair.

        Lower tuple = pick first. video_count first, then action-pair
        rarity, then random jitter for determinism via rng-injection.
        """
        ap = (fix["action_prev_type"], fix["action_curr_type"])
        return (
            per_video[fix["video_id"]],
            action_pair_counts[ap],
            fix["_jitter"],
        )

    # Assign per-fix jitter once so sort is deterministic.
    for fix in fixes:
        fix["_jitter"] = rng.random()

    # Pass 1: take TARGET_PER_BAND from each band, respecting per-video cap.
    for _lo, _hi, name in ALT_RATIO_BANDS:
        pool = list(fixes_by_band.get(name, []))
        # Resort each iteration so updated counts steer the choice.
        taken_this_band = 0
        while taken_this_band < TARGET_PER_BAND and pool:
            pool.sort(key=_key)
            picked = None
            for cand in pool:
                if per_video[cand["video_id"]] >= MAX_PER_VIDEO:
                    continue
                picked = cand
                break
            if picked is None:
                break
            pool.remove(picked)
            selected.append(picked)
            per_video[picked["video_id"]] += 1
            action_pair_counts[(picked["action_prev_type"],
                                picked["action_curr_type"])] += 1
            taken_this_band += 1

    # Pass 2: fill to TOTAL_TARGET from any band, still respecting caps.
    leftover_pool: list[dict[str, Any]] = []
    for _lo, _hi, name in ALT_RATIO_BANDS:
        for fix in fixes_by_band.get(name, []):
            if fix in selected:
                continue
            leftover_pool.append(fix)
    while len(selected) < TOTAL_TARGET and leftover_pool:
        leftover_pool.sort(key=_key)
        picked = None
        for cand in leftover_pool:
            if per_video[cand["video_id"]] >= MAX_PER_VIDEO:
                continue
            picked = cand
            break
        if picked is None:
            break
        leftover_pool.remove(picked)
        selected.append(picked)
        per_video[picked["video_id"]] += 1
        action_pair_counts[(picked["action_prev_type"],
                            picked["action_curr_type"])] += 1

    # Pass 3 (relax per-video cap if still under target).
    if len(selected) < TOTAL_TARGET:
        leftover_pool = [
            fix for band in fixes_by_band.values() for fix in band
            if fix not in selected
        ]
        leftover_pool.sort(key=lambda f: f["_jitter"])
        for cand in leftover_pool:
            if len(selected) >= TOTAL_TARGET:
                break
            selected.append(cand)
            per_video[cand["video_id"]] += 1

    return selected


# ---------------------------------------------------------------------------
# DB enrichment + A1 simulation
# ---------------------------------------------------------------------------

def _load_video_meta(video_ids: list[str]) -> dict[str, dict[str, Any]]:
    if not video_ids:
        return {}
    placeholders = ",".join(["%s"] * len(video_ids))
    sql = (
        f"SELECT id, name, filename, fps FROM videos WHERE id IN ({placeholders})"
    )
    out: dict[str, dict[str, Any]] = {}
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, video_ids)
            for vid, name, filename, fps in cur.fetchall():
                out[str(vid)] = {
                    "name": name,
                    "filename": filename,
                    "fps": float(fps) if fps is not None else 30.0,
                }
    return out


def _load_rally_meta(rally_ids: list[str]) -> dict[str, dict[str, Any]]:
    if not rally_ids:
        return {}
    placeholders = ",".join(["%s"] * len(rally_ids))
    sql = (
        f"SELECT r.id, r.start_ms, r.end_ms, pt.actions_json, pt.contacts_json "
        f"FROM rallies r JOIN player_tracks pt ON pt.rally_id = r.id "
        f"WHERE r.id IN ({placeholders})"
    )
    out: dict[str, dict[str, Any]] = {}
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, rally_ids)
            for rid, start_ms, end_ms, aj, cj in cur.fetchall():
                out[str(rid)] = {
                    "start_ms": int(start_ms),
                    "end_ms": int(end_ms),
                    "actions_json": aj,
                    "contacts_json": cj,
                }
    return out


def _simulate_a1(
    *,
    actions_json: dict[str, Any],
    contacts_json: dict[str, Any] | list[Any] | None,
) -> tuple[dict[int, int], list[dict[str, Any]]]:
    """Run reattribute_players (A1 ON via env) and return new pid map.

    Returns (new_pid_by_frame, new_action_dicts).
    """
    actions_raw = actions_json.get("actions") or []
    team_assignments_raw = actions_json.get("teamAssignments") or {}
    contacts_raw: list[dict[str, Any]] = []
    if isinstance(contacts_json, dict):
        contacts_raw = contacts_json.get("contacts") or []
    elif isinstance(contacts_json, list):
        contacts_raw = contacts_json

    actions = _reconstruct_actions(actions_raw)
    contacts = _reconstruct_contacts(contacts_raw)
    team_ints = _team_assignments_int(team_assignments_raw)
    if team_ints and actions:
        reattribute_players(actions, contacts, team_ints)
    new_dicts = [a.to_dict() for a in actions]
    new_pid_by_frame: dict[int, int] = {}
    for a in actions:
        new_pid_by_frame[a.frame] = a.player_track_id
    return new_pid_by_frame, new_dicts


# ---------------------------------------------------------------------------
# Markdown emission
# ---------------------------------------------------------------------------

def _format_time(ms: int) -> str:
    total_s = ms / 1000.0
    minutes = int(total_s // 60)
    seconds = total_s - minutes * 60
    return f"{minutes:02d}:{seconds:06.3f}"


def _format_top_candidates(raw: str, max_n: int = 5) -> str:
    """`prev_top3_candidates` / `curr_top3_candidates` are JSON strings."""
    import json
    try:
        parsed = json.loads(raw) if raw else []
    except (json.JSONDecodeError, TypeError):
        return raw or ""
    out_parts = []
    for entry in parsed[:max_n]:
        if isinstance(entry, (list, tuple)) and len(entry) >= 3:
            tid, dist, team = entry[0], entry[1], entry[2]
            out_parts.append(f"p{tid}({team})@{dist:.4f}")
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
            tid, dist = entry[0], entry[1]
            out_parts.append(f"p{tid}@{dist:.4f}")
    return ", ".join(out_parts)


def _parse_action_team(actions_json: dict[str, Any], frame: int, pid: int) -> str:
    """Look up team label string ('A'/'B') for a (pid, frame) action."""
    for a in actions_json.get("actions", []):
        try:
            if int(a.get("frame", -1)) == frame:
                return str(a.get("team", "?"))
        except (TypeError, ValueError):
            continue
    # Fallback: look up the pid in teamAssignments.
    ta = actions_json.get("teamAssignments") or {}
    return str(ta.get(str(pid), "?"))


def _emit_markdown(
    sampled: list[dict[str, Any]],
    rally_meta: dict[str, dict[str, Any]],
    video_meta: dict[str, dict[str, Any]],
) -> str:
    lines: list[str] = []
    lines.append("# A1 Visual Verification Sample — 2026-05-13")
    lines.append("")
    lines.append(
        "User: scrub each contact frame in the source video. Mark each row "
        "✅ (A1 picked the real toucher) / ❌ (A1 made it worse) / "
        "⚠️ (ambiguous). Ship threshold: ≥ 8 / 10 ✅."
    )
    lines.append("")
    lines.append(
        "Conventions: `frame` is the frame index inside the rally clip "
        "(0-based, at the source-video fps). "
        "`Source-video time` = rally `start_ms` + `frame / fps * 1000`. "
        "Player IDs (`p1..p4`) are the post-`remap_track_ids` PIDs; teams "
        "are A/B as labeled in `actions_json.teamAssignments`."
    )
    lines.append("")
    lines.append(
        "| # | Video | Rally (short) | Source-video time | Frame | "
        "Prev action (frame, player, team) | Curr action | "
        "A1 before → after | alt_ratio | Verdict |"
    )
    lines.append(
        "|--:|:------|:--------------|:-----------------:|:-----:|"
        ":-----------------------------------|:------------|"
        ":------------------|:---------:|:-------:|"
    )

    details_blocks: list[str] = []

    for i, fix in enumerate(sampled, start=1):
        rid = fix["rally_id"]
        vid = fix["video_id"]
        rm = rally_meta.get(rid, {})
        vm = video_meta.get(vid, {})
        fps = vm.get("fps", 30.0) or 30.0
        start_ms = rm.get("start_ms", 0)
        frame_curr = int(fix["frame_curr"])
        frame_prev = int(fix["frame_prev"])
        curr_time_ms = start_ms + frame_curr * 1000.0 / fps
        prev_time_ms = start_ms + frame_prev * 1000.0 / fps

        pid_before = int(fix["player_id"])
        team_label = fix["team_label"]

        actions_json = rm.get("actions_json") or {}
        team_curr = _parse_action_team(actions_json, frame_curr, pid_before)
        team_prev = _parse_action_team(actions_json, frame_prev, pid_before)

        # Compute A1's post-attribution choice in process. Either side
        # of the pair (prev or curr) may have flipped — whichever makes
        # the pair no longer a C-4 violation.
        new_pid_by_frame = fix["_a1_pid_by_frame"]
        new_pid_curr = new_pid_by_frame.get(frame_curr, pid_before)
        new_pid_prev = new_pid_by_frame.get(frame_prev, pid_before)
        if new_pid_curr != pid_before:
            change = f"curr p{pid_before} → p{new_pid_curr}"
        elif new_pid_prev != pid_before:
            change = f"prev p{pid_before} → p{new_pid_prev}"
        else:
            change = f"p{pid_before} → ??? (uncertain/abstain)"
        pid_after = new_pid_curr

        alt_ratio = _parse_alt_ratio(fix["curr_best_same_team_alt_ratio"])
        alt_ratio_str = f"{alt_ratio:.1f}x" if alt_ratio else "—"

        rally_short = rid[:8]
        video_name = vm.get("name", "?")

        prev_cell = (
            f"{fix['action_prev_type']}@{frame_prev} p{pid_before}({team_prev})"
        )
        curr_cell = fix["action_curr_type"]

        lines.append(
            f"| {i} | {video_name} | {rally_short} | "
            f"{_format_time(int(curr_time_ms))} | {frame_curr} | "
            f"{prev_cell} | {curr_cell} | {change} | "
            f"{alt_ratio_str} |  |"
        )

        # Per-fix details block.
        prev_cands = _format_top_candidates(
            fix["prev_top3_candidates"], max_n=5
        )
        curr_cands = _format_top_candidates(
            fix["curr_top3_candidates"], max_n=5
        )
        block = []
        block.append(f"### {i}. {video_name} / rally {rally_short} / "
                     f"pair_idx {fix['pair_idx']}")
        block.append("")
        block.append(f"- **video_id**: `{vid}`")
        block.append(f"- **rally_id**: `{rid}`")
        block.append(f"- **filename**: `{vm.get('filename', '?')}`")
        block.append(f"- **fps**: {fps:g}, rally start_ms: {start_ms}")
        block.append(
            f"- **prev action**: `{fix['action_prev_type']}` "
            f"frame={frame_prev} (t={_format_time(int(prev_time_ms))}) "
            f"player=p{pid_before}({team_prev}) → **A1→ p{new_pid_prev}** "
            f"conf={float(fix['conf_prev']):.3f} "
            f"player_dist={float(fix['prev_player_dist']):.4f}"
        )
        block.append(
            f"- **curr action**: `{fix['action_curr_type']}` "
            f"frame={frame_curr} (t={_format_time(int(curr_time_ms))}) "
            f"player=p{pid_before}({team_curr}) → **A1→ p{new_pid_curr}** "
            f"conf={float(fix['conf_curr']):.3f} "
            f"player_dist={float(fix['curr_player_dist']):.4f}"
        )
        if alt_ratio:
            block.append(
                f"- **curr_best_same_team_alt_ratio**: {alt_ratio:.2f}x"
            )
        block.append(f"- **prev candidates (top 5)**: {prev_cands}")
        block.append(f"- **curr candidates (top 5)**: {curr_cands}")
        block.append("")
        details_blocks.append("\n".join(block))

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Per-fix details")
    lines.append("")
    lines.extend(details_blocks)

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print(f"[sample] reading baseline catalog: {BASELINE_CSV}", flush=True)
    baseline_rows = _read_catalog(BASELINE_CSV)
    print(f"[sample] reading a1-on catalog:    {A1_ON_CSV}", flush=True)
    a1_on_rows = _read_catalog(A1_ON_CSV)
    print(f"[sample] baseline n={len(baseline_rows)}, a1_on n={len(a1_on_rows)}",
          flush=True)

    # Filter to spec cohort.
    def _in_cohort(row: dict[str, Any]) -> bool:
        r = _parse_alt_ratio(row["curr_best_same_team_alt_ratio"])
        if r is None:
            return False
        return 2.0 <= r <= 50.0

    baseline_cohort = [r for r in baseline_rows if _in_cohort(r)]
    a1_on_cohort = [r for r in a1_on_rows if _in_cohort(r)]
    a1_on_keys = {_pair_key(r) for r in a1_on_cohort}
    print(f"[sample] baseline cohort (2x-50x): {len(baseline_cohort)}",
          flush=True)
    print(f"[sample] a1_on cohort   (2x-50x): {len(a1_on_cohort)}",
          flush=True)

    # Fixes = baseline cohort rows that are NOT in a1-on cohort.
    fixes = [r for r in baseline_cohort if _pair_key(r) not in a1_on_keys]
    print(f"[sample] fixes (baseline ∖ a1_on, cohort): {len(fixes)}",
          flush=True)

    sampled = _stratified_sample(fixes)
    print(f"[sample] sampled {len(sampled)} fixes", flush=True)
    for i, s in enumerate(sampled, start=1):
        r = _parse_alt_ratio(s["curr_best_same_team_alt_ratio"]) or 0.0
        print(
            f"  [{i}/{len(sampled)}] rally={s['rally_id'][:8]} "
            f"video={s['video_id'][:8]} "
            f"frames={s['frame_prev']}->{s['frame_curr']} "
            f"types={s['action_prev_type']}->{s['action_curr_type']} "
            f"alt_ratio={r:.2f}x",
            flush=True,
        )

    # Enrich.
    video_ids = sorted({s["video_id"] for s in sampled})
    rally_ids = sorted({s["rally_id"] for s in sampled})
    video_meta = _load_video_meta(video_ids)
    rally_meta = _load_rally_meta(rally_ids)

    # Simulate A1 per sampled rally (cache per rally so we don't re-run).
    a1_cache: dict[str, dict[int, int]] = {}
    for s in sampled:
        rid = s["rally_id"]
        rm = rally_meta.get(rid)
        if rm is None:
            a1_cache[rid] = {}
            continue
        if rid not in a1_cache:
            new_pid_by_frame, _ = _simulate_a1(
                actions_json=rm.get("actions_json") or {},
                contacts_json=rm.get("contacts_json"),
            )
            a1_cache[rid] = new_pid_by_frame
        s["_a1_pid_by_frame"] = a1_cache[rid]

    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    md = _emit_markdown(sampled, rally_meta, video_meta)
    OUTPUT_MD.write_text(md)
    print(f"[sample] wrote markdown -> {OUTPUT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
