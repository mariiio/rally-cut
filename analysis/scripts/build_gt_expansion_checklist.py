"""Build the T2 GT-expansion labeling checklist.

Produces a stratified pick list of ~150 contacts across ~30 rallies spanning
the failure patterns surfaced by the 2026-05-13 session:

  A. Cascade-shape (same-player back-to-back, prev != block) — from C-4 catalog.
  B. Mid-rally cross-team (F3/F5 shape) — from fresh C-5 audit over the fleet.
  C. At-net attacks — from probe-B s4_fleet_candidates (attack-after-attack /
     attack-after-set buckets; the at-net BLOCK-vs-ATTACK confusion zone).
  D. Random / control — high-confidence single-rank-1-margin contacts.
  E. Confidence percentiles — 5 low, 5 mid, 10 high.

Stratification constraints:
  - Max 5 picks per rally
  - Max 8 picks per video, target ~25-30 distinct videos
  - Skip rallies with >5 existing GT rows (already heavily labeled)
  - Prefer rallies with actions_pipeline_version = 'v2' (current)
  - Anchor: a0881d82 (titi) f128 + f225 must be in bucket A

Outputs:
  reports/gt_expansion_2026_05_14/labeling_checklist.md
  reports/gt_expansion_2026_05_14/picks.json

Usage:
    cd analysis
    uv run python scripts/build_gt_expansion_checklist.py
"""
from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, cast

from rallycut.evaluation.tracking.db import get_connection
from rallycut.tracking.action_classifier import ACTION_PIPELINE_VERSION

HERE = Path(__file__).resolve().parent
REPO = HERE.parent  # analysis/
OUTPUT_DIR = REPO / "reports" / "gt_expansion_2026_05_14"
CHECKLIST_MD = OUTPUT_DIR / "labeling_checklist.md"
PICKS_JSON = OUTPUT_DIR / "picks.json"

C4_CSV = REPO / "reports" / "coherence_c4_catalog" / "2026-05-13_post_redetect.csv"
S4_FLEET_JSON = (
    REPO / "reports" / "probe_b_sequence_aware"
    / "2026_05_14" / "s4_fleet_candidates.json"
)

# Targets per bucket. Total ~150.
TARGETS = {
    "A_cascade": 40,
    "B_cross_team": 30,
    "C_at_net_attack": 30,
    "D_random_control": 30,
    "E_confidence": 20,
}
# Sub-targets for E.
E_LOW = 5   # confidence < 0.3
E_MID = 5   # 0.3 <= confidence < 0.6
E_HIGH = 10  # confidence >= 0.7

MAX_PICKS_PER_RALLY = 5
MAX_PICKS_PER_VIDEO = 8
MAX_EXISTING_GT_ROWS = 5
SEED = 42
# Densify: prefer adding more picks to rallies we've already touched, so the
# user spends fewer rally-context-switches (target ~30 distinct rallies for
# ~150 contacts → ~5 contacts/rally).
TARGET_DISTINCT_RALLIES = 30


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Pick:
    bucket: str
    video_id: str
    video_name: str
    rally_id: str
    rally_order: int
    frame: int
    source_time_ms: int
    fps: float
    pipeline_pid: int | None
    pipeline_action: str | None
    prev_frame: int | None
    prev_pid: int | None
    prev_action: str | None
    conf: float | None
    notes: str = ""


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def load_video_metadata() -> tuple[dict[str, str], dict[str, float]]:
    """Return (video_id -> name, video_id -> fps)."""
    name_by_id: dict[str, str] = {}
    fps_by_id: dict[str, float] = {}
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT id, name, fps FROM videos")
        for vid, name, fps in cur.fetchall():
            name_by_id[str(vid)] = str(name)
            fps_by_id[str(vid)] = float(fps) if fps is not None else 30.0
    return name_by_id, fps_by_id


def load_rally_metadata() -> dict[str, dict[str, Any]]:
    """Return rally_id -> {video_id, start_ms, order, actions_json,
    actions_pipeline_version}."""
    out: dict[str, dict[str, Any]] = {}
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT r.id, r.video_id, r.start_ms,
                   ROW_NUMBER() OVER (PARTITION BY r.video_id ORDER BY r.start_ms)
                       AS rally_order,
                   pt.actions_json,
                   pt.actions_pipeline_version
              FROM rallies r
              LEFT JOIN player_tracks pt ON pt.rally_id = r.id
             WHERE (r.status = 'CONFIRMED' OR r.status IS NULL)
            """
        )
        for rid, vid, start_ms, order, aj, apv in cur.fetchall():
            out[str(rid)] = {
                "video_id": str(vid),
                "start_ms": int(start_ms) if start_ms is not None else 0,
                "rally_order": int(order),
                "actions_json": aj if isinstance(aj, dict) else None,
                "actions_pipeline_version": apv,
            }
    return out


def load_existing_gt_counts() -> dict[str, int]:
    """rally_id -> count of existing rally_action_ground_truth rows."""
    out: dict[str, int] = {}
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT rally_id, COUNT(*)
              FROM rally_action_ground_truth
             GROUP BY rally_id
            """
        )
        for rid, n in cur.fetchall():
            out[str(rid)] = int(n)
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def source_time_ms(rally_start_ms: int, frame: int, fps: float) -> int:
    return int(rally_start_ms + (frame * 1000.0) / max(fps, 1e-6))


def fmt_time(ms: int) -> str:
    total_s = ms / 1000.0
    m = int(total_s // 60)
    s = total_s - m * 60
    return f"{m}:{s:06.3f}"


def action_at_frame(
    actions: list[dict[str, Any]] | None, frame: int,
) -> dict[str, Any] | None:
    """Find the action whose 'frame' matches `frame` exactly."""
    if not actions:
        return None
    for a in actions:
        if int(a.get("frame", -1)) == frame:
            return a
    return None


def get_prev_action(
    actions: list[dict[str, Any]] | None, frame: int,
) -> dict[str, Any] | None:
    """Return the action immediately preceding `frame` (sorted by frame)."""
    if not actions:
        return None
    sorted_acts = sorted(actions, key=lambda a: int(a.get("frame", 0)))
    prev = None
    for a in sorted_acts:
        f = int(a.get("frame", -1))
        if f >= frame:
            break
        prev = a
    return prev


# ---------------------------------------------------------------------------
# Bucket A: Cascade-shape (C-4 catalog)
# ---------------------------------------------------------------------------

def bucket_a_cascade(
    rally_meta: dict[str, dict[str, Any]],
    video_names: dict[str, str],
    video_fps: dict[str, float],
    existing_gt: dict[str, int],
    per_rally_counts: dict[str, int],
    per_video_counts: dict[str, int],
    seen: set[tuple[str, int]],
    rng: random.Random,
) -> list[Pick]:
    """C-4 catalog rows are pairs (prev_action, curr_action) with same-player
    back-to-back. We want CONTACT-level picks — so each row contributes BOTH
    frames (prev + curr). We also stratify by alt_ratio in [1.5, 50]."""
    if not C4_CSV.exists():
        print(f"  WARN: C-4 catalog not found at {C4_CSV}")
        return []

    rows: list[dict[str, str]] = []
    with C4_CSV.open() as f:
        for r in csv.DictReader(f):
            rows.append(r)
    print(f"  Loaded {len(rows)} C-4 catalog rows")

    # Parse + filter rows
    candidates: list[dict[str, Any]] = []
    for r in rows:
        rid = r["rally_id"]
        meta = rally_meta.get(rid)
        if not meta:
            continue
        # Use the curr alt_ratio for stratification (the cascade pick).
        try:
            alt_ratio = float(r["curr_best_same_team_alt_ratio"])
        except (KeyError, ValueError):
            alt_ratio = float("nan")
        if alt_ratio != alt_ratio:  # NaN
            continue
        if not (1.5 <= alt_ratio <= 50.0):
            continue
        candidates.append({
            "row": r,
            "rally_id": rid,
            "alt_ratio": alt_ratio,
            "frame_prev": int(r["frame_prev"]),
            "frame_curr": int(r["frame_curr"]),
            "action_prev": r["action_prev_type"],
            "action_curr": r["action_curr_type"],
            "player_id": int(r["player_id"]),
            "meta": meta,
        })
    print(f"  After alt_ratio filter ({len(candidates)} candidates)")
    # Sort by alt_ratio desc — strongest signals first.
    candidates.sort(key=lambda c: -c["alt_ratio"])

    picks: list[Pick] = []

    # 1) Force anchors: titi/a0881d82 f128 + f225
    anchor_rally = "a0881d82"
    anchor_frames = {128, 225}
    for c in candidates:
        if not c["rally_id"].startswith(anchor_rally):
            continue
        for fno in (c["frame_prev"], c["frame_curr"]):
            if fno not in anchor_frames:
                continue
            key = (c["rally_id"], fno)
            if key in seen:
                continue
            pick = _build_pick_from_c4(
                c, fno, video_names, video_fps,
                bucket="A_cascade",
                notes="anchor titi a0881d82 cascade",
            )
            if pick is None:
                continue
            picks.append(pick)
            seen.add(key)
            per_rally_counts[c["rally_id"]] += 1
            per_video_counts[pick.video_id] += 1
    print(f"  Anchors locked in: {len(picks)} (target frames 128, 225)")

    # 2) Group candidates by rally so we can densify (max contacts/rally).
    #    Rally rank = max alt_ratio of its rows.
    by_rally: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in candidates:
        rid = c["rally_id"]
        # Skip heavily labeled / non-current-pipeline.
        if existing_gt.get(rid, 0) > MAX_EXISTING_GT_ROWS:
            continue
        if c["meta"].get("actions_pipeline_version") != ACTION_PIPELINE_VERSION:
            continue
        by_rally[rid].append(c)
    rally_rank = sorted(
        by_rally.items(),
        key=lambda kv: -max(c["alt_ratio"] for c in kv[1]),
    )
    target = TARGETS["A_cascade"]
    # Rank rallies: more cascade pairs first, then by max alt_ratio.
    rally_rank = sorted(
        by_rally.items(),
        key=lambda kv: (
            -len(kv[1]),
            -max(c["alt_ratio"] for c in kv[1]),
        ),
    )
    for rid, rally_candidates in rally_rank:
        if len(picks) >= target:
            break
        if per_rally_counts.get(rid, 0) >= MAX_PICKS_PER_RALLY:
            continue
        meta = rally_candidates[0]["meta"]
        vid = meta["video_id"]
        if per_video_counts.get(vid, 0) >= MAX_PICKS_PER_VIDEO:
            continue
        # Collect all unique frames in this rally (sorted by descending
        # alt_ratio so most-decisive cascade pairs get priority within rally).
        rally_candidates.sort(key=lambda c: -c["alt_ratio"])
        for c in rally_candidates:
            for fno in (c["frame_prev"], c["frame_curr"]):
                if len(picks) >= target:
                    break
                key = (rid, fno)
                if key in seen:
                    continue
                if per_rally_counts.get(rid, 0) >= MAX_PICKS_PER_RALLY:
                    break
                if per_video_counts.get(vid, 0) >= MAX_PICKS_PER_VIDEO:
                    break
                pick = _build_pick_from_c4(
                    c, fno, video_names, video_fps,
                    bucket="A_cascade",
                    notes=f"alt_ratio={c['alt_ratio']:.2f}",
                )
                if pick is None:
                    continue
                picks.append(pick)
                seen.add(key)
                per_rally_counts[rid] = per_rally_counts.get(rid, 0) + 1
                per_video_counts[vid] = per_video_counts.get(vid, 0) + 1
            if per_rally_counts.get(rid, 0) >= MAX_PICKS_PER_RALLY:
                break
        # Backfill: pull other actions in same rally up to MAX_PICKS_PER_RALLY,
        # tagged as A_cascade_neighbor. The user is already in this rally so
        # neighbor labels are cheap and useful for "is the whole rally clean?"
        if (
            per_rally_counts.get(rid, 0) < MAX_PICKS_PER_RALLY
            and per_video_counts.get(vid, 0) < MAX_PICKS_PER_VIDEO
            and len(picks) < target
        ):
            actions = (meta.get("actions_json") or {}).get("actions") or []
            for a in actions:
                if len(picks) >= target:
                    break
                if per_rally_counts.get(rid, 0) >= MAX_PICKS_PER_RALLY:
                    break
                if per_video_counts.get(vid, 0) >= MAX_PICKS_PER_VIDEO:
                    break
                pid = a.get("playerTrackId")
                if pid is None or pid == -1:
                    continue
                frame = int(a.get("frame", -1))
                key = (rid, frame)
                if key in seen:
                    continue
                vname = video_names.get(vid, vid[:8])
                fps = video_fps.get(vid, 30.0)
                prev_act = get_prev_action(actions, frame)
                pick = Pick(
                    bucket="A_cascade",
                    video_id=vid,
                    video_name=vname,
                    rally_id=rid,
                    rally_order=meta["rally_order"],
                    frame=frame,
                    source_time_ms=source_time_ms(meta["start_ms"], frame, fps),
                    fps=fps,
                    pipeline_pid=int(pid),
                    pipeline_action=str(a.get("action", "")),
                    prev_frame=(int(prev_act["frame"]) if prev_act else None),
                    prev_pid=(
                        int(prev_act["playerTrackId"])
                        if prev_act and prev_act.get("playerTrackId") is not None else None
                    ),
                    prev_action=(prev_act.get("action") if prev_act else None),
                    conf=(
                        float(a["confidence"])
                        if a.get("confidence") is not None else None
                    ),
                    notes="cascade-rally neighbor (densified)",
                )
                picks.append(pick)
                seen.add(key)
                per_rally_counts[rid] = per_rally_counts.get(rid, 0) + 1
                per_video_counts[vid] = per_video_counts.get(vid, 0) + 1
    print(f"  Bucket A total: {len(picks)} picks")
    return picks


def _build_pick_from_c4(
    c: dict[str, Any], frame: int,
    video_names: dict[str, str], video_fps: dict[str, float],
    *, bucket: str, notes: str,
) -> Pick | None:
    meta = c["meta"]
    vid = meta["video_id"]
    rid = c["rally_id"]
    vname = video_names.get(vid, vid[:8])
    fps = video_fps.get(vid, 30.0)
    actions = (meta.get("actions_json") or {}).get("actions") or []
    cur_act = action_at_frame(actions, frame)
    prev_act = get_prev_action(actions, frame)
    return Pick(
        bucket=bucket,
        video_id=vid,
        video_name=vname,
        rally_id=rid,
        rally_order=meta["rally_order"],
        frame=frame,
        source_time_ms=source_time_ms(meta["start_ms"], frame, fps),
        fps=fps,
        pipeline_pid=(
            int(cur_act["playerTrackId"]) if cur_act and cur_act.get("playerTrackId") is not None else None
        ),
        pipeline_action=cur_act.get("action") if cur_act else None,
        prev_frame=(int(prev_act["frame"]) if prev_act else None),
        prev_pid=(
            int(prev_act["playerTrackId"]) if prev_act and prev_act.get("playerTrackId") is not None else None
        ),
        prev_action=prev_act.get("action") if prev_act else None,
        conf=(float(cur_act["confidence"]) if cur_act and cur_act.get("confidence") is not None else None),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Bucket B: Mid-rally cross-team (C-5)
# ---------------------------------------------------------------------------

_C5_POSSESSION_TRANSFER = frozenset({"attack", "serve", "block"})


def bucket_b_cross_team(
    rally_meta: dict[str, dict[str, Any]],
    video_names: dict[str, str],
    video_fps: dict[str, float],
    existing_gt: dict[str, int],
    per_rally_counts: dict[str, int],
    per_video_counts: dict[str, int],
    seen: set[tuple[str, int]],
    rng: random.Random,
) -> list[Pick]:
    """Find mid-rally cross-team transitions where prev was NOT a possession-
    transfer action (attack/serve/block). Scan all v2 rallies."""
    candidates: list[dict[str, Any]] = []
    for rid, meta in rally_meta.items():
        if meta.get("actions_pipeline_version") != ACTION_PIPELINE_VERSION:
            continue
        if existing_gt.get(rid, 0) > MAX_EXISTING_GT_ROWS:
            continue
        actions_json = meta.get("actions_json") or {}
        actions = actions_json.get("actions") or []
        team_assignments = actions_json.get("teamAssignments") or {}
        if not actions or not team_assignments:
            continue
        sorted_acts = sorted(actions, key=lambda a: int(a.get("frame", 0)))
        for i in range(1, len(sorted_acts)):
            prev = sorted_acts[i - 1]
            curr = sorted_acts[i]
            prev_pid = prev.get("playerTrackId")
            curr_pid = curr.get("playerTrackId")
            if prev_pid is None or curr_pid is None:
                continue
            if prev_pid == -1 or curr_pid == -1:
                continue
            prev_team = team_assignments.get(str(prev_pid))
            curr_team = team_assignments.get(str(curr_pid))
            if not prev_team or not curr_team:
                continue
            if prev_team == curr_team:
                continue
            prev_act_type = str(prev.get("action", "")).lower()
            if prev_act_type in _C5_POSSESSION_TRANSFER:
                continue
            # Found a C-5 violation. The "curr" action is the one to label —
            # it's the one with suspicious cross-team attribution.
            candidates.append({
                "rally_id": rid,
                "meta": meta,
                "prev": prev,
                "curr": curr,
                "prev_team": prev_team,
                "curr_team": curr_team,
            })
    print(f"  Found {len(candidates)} C-5 candidate pairs")
    # Group by rally for densification. Rally rank: more violations = higher rank.
    by_rally: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in candidates:
        by_rally[c["rally_id"]].append(c)
    # Shuffle ordering within each rally; rank rallies by violation count desc.
    for rid, lst in by_rally.items():
        rng.shuffle(lst)
    rally_rank = sorted(by_rally.items(), key=lambda kv: -len(kv[1]))

    picks: list[Pick] = []
    target = TARGETS["B_cross_team"]
    for rid, rally_candidates in rally_rank:
        if len(picks) >= target:
            break
        if per_rally_counts.get(rid, 0) >= MAX_PICKS_PER_RALLY:
            continue
        meta = rally_candidates[0]["meta"]
        vid = meta["video_id"]
        if per_video_counts.get(vid, 0) >= MAX_PICKS_PER_VIDEO:
            continue
        for c in rally_candidates:
            if len(picks) >= target:
                break
            if per_rally_counts.get(rid, 0) >= MAX_PICKS_PER_RALLY:
                break
            if per_video_counts.get(vid, 0) >= MAX_PICKS_PER_VIDEO:
                break
            curr = c["curr"]
            frame = int(curr.get("frame", 0))
            key = (rid, frame)
            if key in seen:
                continue
            prev = c["prev"]
            vname = video_names.get(vid, vid[:8])
            fps = video_fps.get(vid, 30.0)
            pick = Pick(
                bucket="B_cross_team",
                video_id=vid,
                video_name=vname,
                rally_id=rid,
                rally_order=meta["rally_order"],
                frame=frame,
                source_time_ms=source_time_ms(meta["start_ms"], frame, fps),
                fps=fps,
                pipeline_pid=int(curr["playerTrackId"]),
                pipeline_action=str(curr.get("action", "")),
                prev_frame=int(prev.get("frame", 0)),
                prev_pid=int(prev["playerTrackId"]),
                prev_action=str(prev.get("action", "")),
                conf=(float(curr["confidence"]) if curr.get("confidence") is not None else None),
                notes=(
                    f"C-5 cross-team {c['prev_team']}->{c['curr_team']} "
                    f"prev={prev.get('action')} non-transfer"
                ),
            )
            picks.append(pick)
            seen.add(key)
            per_rally_counts[rid] = per_rally_counts.get(rid, 0) + 1
            per_video_counts[vid] = per_video_counts.get(vid, 0) + 1
        # Backfill same-rally neighbors after the cross-team picks.
        if (
            per_rally_counts.get(rid, 0) < MAX_PICKS_PER_RALLY
            and per_video_counts.get(vid, 0) < MAX_PICKS_PER_VIDEO
            and len(picks) < target
        ):
            actions = (meta.get("actions_json") or {}).get("actions") or []
            for a in actions:
                if len(picks) >= target:
                    break
                if per_rally_counts.get(rid, 0) >= MAX_PICKS_PER_RALLY:
                    break
                if per_video_counts.get(vid, 0) >= MAX_PICKS_PER_VIDEO:
                    break
                pid = a.get("playerTrackId")
                if pid is None or pid == -1:
                    continue
                frame = int(a.get("frame", -1))
                key = (rid, frame)
                if key in seen:
                    continue
                vname = video_names.get(vid, vid[:8])
                fps = video_fps.get(vid, 30.0)
                prev_act = get_prev_action(actions, frame)
                pick = Pick(
                    bucket="B_cross_team",
                    video_id=vid,
                    video_name=vname,
                    rally_id=rid,
                    rally_order=meta["rally_order"],
                    frame=frame,
                    source_time_ms=source_time_ms(meta["start_ms"], frame, fps),
                    fps=fps,
                    pipeline_pid=int(pid),
                    pipeline_action=str(a.get("action", "")),
                    prev_frame=(int(prev_act["frame"]) if prev_act else None),
                    prev_pid=(
                        int(prev_act["playerTrackId"])
                        if prev_act and prev_act.get("playerTrackId") is not None else None
                    ),
                    prev_action=(prev_act.get("action") if prev_act else None),
                    conf=(
                        float(a["confidence"])
                        if a.get("confidence") is not None else None
                    ),
                    notes="cross-team rally neighbor (densified)",
                )
                picks.append(pick)
                seen.add(key)
                per_rally_counts[rid] = per_rally_counts.get(rid, 0) + 1
                per_video_counts[vid] = per_video_counts.get(vid, 0) + 1
    print(f"  Bucket B total: {len(picks)} picks")
    return picks


# ---------------------------------------------------------------------------
# Bucket C: At-net attacks (probe-B s4_fleet_candidates: ATTACK rows)
# ---------------------------------------------------------------------------

def bucket_c_at_net_attack(
    rally_meta: dict[str, dict[str, Any]],
    video_names: dict[str, str],
    video_fps: dict[str, float],
    existing_gt: dict[str, int],
    per_rally_counts: dict[str, int],
    per_video_counts: dict[str, int],
    seen: set[tuple[str, int]],
    rng: random.Random,
) -> list[Pick]:
    """Pull ATTACK actions from s4_fleet_candidates.json. These are the
    BLOCK-vs-ATTACK at-net confusion zone — many are mis-classified as ATTACK
    when they're actually BLOCKs (or vice versa), and player attribution is
    also noisy at the net."""
    if not S4_FLEET_JSON.exists():
        print(f"  WARN: s4 fleet candidates not found at {S4_FLEET_JSON}")
        return []
    data = json.loads(S4_FLEET_JSON.read_text())
    flips = data.get("flips", [])
    # ATTACK action_type rows (the C bucket: at-net attacks, block candidates).
    attack_flips = [f for f in flips if f.get("action_type") == "ATTACK"]
    print(f"  s4 fleet has {len(attack_flips)} ATTACK flip candidates")

    # Group by rally for densification.
    by_rally: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for f in attack_flips:
        by_rally[f["rally_id"]].append(f)
    rally_rank = sorted(by_rally.items(), key=lambda kv: -len(kv[1]))

    picks: list[Pick] = []
    target = TARGETS["C_at_net_attack"]
    for rid, rally_flips in rally_rank:
        if len(picks) >= target:
            break
        meta = rally_meta.get(rid)
        if not meta:
            continue
        if meta.get("actions_pipeline_version") != ACTION_PIPELINE_VERSION:
            continue
        if existing_gt.get(rid, 0) > MAX_EXISTING_GT_ROWS:
            continue
        vid = meta["video_id"]
        if per_rally_counts.get(rid, 0) >= MAX_PICKS_PER_RALLY:
            continue
        if per_video_counts.get(vid, 0) >= MAX_PICKS_PER_VIDEO:
            continue
        vname = video_names.get(vid, vid[:8])
        fps = video_fps.get(vid, 30.0)
        actions = (meta.get("actions_json") or {}).get("actions") or []

        # First, take all the s4-flip ATTACK frames in this rally.
        added_in_rally = 0
        for f in rally_flips:
            if len(picks) >= target:
                break
            if per_rally_counts.get(rid, 0) >= MAX_PICKS_PER_RALLY:
                break
            if per_video_counts.get(vid, 0) >= MAX_PICKS_PER_VIDEO:
                break
            frame = int(f["pl_frame"])
            key = (rid, frame)
            if key in seen:
                continue
            cur_act = action_at_frame(actions, frame)
            conf = (
                float(cur_act["confidence"]) if cur_act and cur_act.get("confidence") is not None
                else None
            )
            pick = Pick(
                bucket="C_at_net_attack",
                video_id=vid,
                video_name=vname,
                rally_id=rid,
                rally_order=meta["rally_order"],
                frame=frame,
                source_time_ms=source_time_ms(meta["start_ms"], frame, fps),
                fps=fps,
                pipeline_pid=int(f["pipeline_pid"]),
                pipeline_action="ATTACK",
                prev_frame=(int(f["prev_action_frame"]) if f.get("prev_action_frame") is not None else None),
                prev_pid=(int(f["prev_toucher_pid"]) if f.get("prev_toucher_pid") is not None else None),
                prev_action=f.get("prev_action_type"),
                conf=conf,
                notes=(
                    f"s4 disagrees: pipeline=p{f['pipeline_pid']} "
                    f"vs s4=p{f['s4_pid']}"
                ),
            )
            picks.append(pick)
            seen.add(key)
            per_rally_counts[rid] = per_rally_counts.get(rid, 0) + 1
            per_video_counts[vid] = per_video_counts.get(vid, 0) + 1
            added_in_rally += 1

        # Then densify with other contacts in same rally (since the user is
        # already in this rally — labeling neighbors is cheap). Prefer at-net
        # actions (attack/block) first, then other contacts.
        actions_sorted = sorted(
            actions,
            key=lambda a: (
                0 if str(a.get("action", "")).lower() in ("attack", "block") else 1,
                int(a.get("frame", 0)),
            ),
        )
        for a in actions_sorted:
            if len(picks) >= target:
                break
            if per_rally_counts.get(rid, 0) >= MAX_PICKS_PER_RALLY:
                break
            if per_video_counts.get(vid, 0) >= MAX_PICKS_PER_VIDEO:
                break
            pid = a.get("playerTrackId")
            if pid is None or pid == -1:
                continue
            frame = int(a.get("frame", -1))
            key = (rid, frame)
            if key in seen:
                continue
            prev_act = get_prev_action(actions, frame)
            pick = Pick(
                bucket="C_at_net_attack",
                video_id=vid,
                video_name=vname,
                rally_id=rid,
                rally_order=meta["rally_order"],
                frame=frame,
                source_time_ms=source_time_ms(meta["start_ms"], frame, fps),
                fps=fps,
                pipeline_pid=int(pid),
                pipeline_action=str(a.get("action", "")),
                prev_frame=(int(prev_act["frame"]) if prev_act else None),
                prev_pid=(
                    int(prev_act["playerTrackId"])
                    if prev_act and prev_act.get("playerTrackId") is not None else None
                ),
                prev_action=(prev_act.get("action") if prev_act else None),
                conf=(
                    float(a["confidence"]) if a.get("confidence") is not None else None
                ),
                notes="at-net rally neighbor (densified)",
            )
            picks.append(pick)
            seen.add(key)
            per_rally_counts[rid] = per_rally_counts.get(rid, 0) + 1
            per_video_counts[vid] = per_video_counts.get(vid, 0) + 1
    print(f"  Bucket C total: {len(picks)} picks")
    return picks


# ---------------------------------------------------------------------------
# Bucket D + E: Random control + confidence percentiles
# ---------------------------------------------------------------------------

def _gather_all_actions(
    rally_meta: dict[str, dict[str, Any]],
    existing_gt: dict[str, int],
    skip_rallies: set[str],
) -> list[dict[str, Any]]:
    """Flatten all actions across all rallies (v2 only, low existing GT)."""
    out: list[dict[str, Any]] = []
    for rid, meta in rally_meta.items():
        if meta.get("actions_pipeline_version") != ACTION_PIPELINE_VERSION:
            continue
        if existing_gt.get(rid, 0) > MAX_EXISTING_GT_ROWS:
            continue
        if rid in skip_rallies:
            # Bucket D should AVOID rallies already used for buckets A/B/C —
            # we want unrelated rallies to verify pipeline isn't introducing
            # regressions on "easy" cases. (Bucket E may overlap; handled
            # separately.)
            continue
        actions = (meta.get("actions_json") or {}).get("actions") or []
        for a in actions:
            pid = a.get("playerTrackId")
            if pid is None or pid == -1:
                continue
            conf = a.get("confidence")
            if conf is None:
                continue
            out.append({
                "rally_id": rid,
                "meta": meta,
                "action": a,
            })
    return out


def bucket_d_random_control(
    rally_meta: dict[str, dict[str, Any]],
    video_names: dict[str, str],
    video_fps: dict[str, float],
    existing_gt: dict[str, int],
    per_rally_counts: dict[str, int],
    per_video_counts: dict[str, int],
    seen: set[tuple[str, int]],
    rng: random.Random,
    rallies_used_by_abc: set[str],
) -> list[Pick]:
    """Random sample of confident (conf >= 0.7) contacts from rallies not yet
    touched by A/B/C. Validates pipeline is well-behaved on the easy cases."""
    pool = _gather_all_actions(rally_meta, existing_gt, rallies_used_by_abc)
    pool = [p for p in pool if float(p["action"]["confidence"]) >= 0.7]
    print(f"  Bucket D pool: {len(pool)} high-conf contacts in unused rallies")

    # Group by rally; pick rallies that have at least 3 high-conf contacts
    # so we get density. Then shuffle rally ordering deterministically.
    by_rally: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in pool:
        by_rally[c["rally_id"]].append(c)
    eligible_rallies = [
        rid for rid, lst in by_rally.items() if len(lst) >= 3
    ]
    rng.shuffle(eligible_rallies)
    print(
        f"  {len(eligible_rallies)} rallies have >=3 high-conf contacts "
        f"(densify pool)"
    )

    picks: list[Pick] = []
    target = TARGETS["D_random_control"]
    for rid in eligible_rallies:
        if len(picks) >= target:
            break
        rally_contacts = by_rally[rid]
        meta = rally_contacts[0]["meta"]
        vid = meta["video_id"]
        if per_rally_counts.get(rid, 0) >= MAX_PICKS_PER_RALLY:
            continue
        if per_video_counts.get(vid, 0) >= MAX_PICKS_PER_VIDEO:
            continue
        rng.shuffle(rally_contacts)
        for c in rally_contacts:
            if len(picks) >= target:
                break
            if per_rally_counts.get(rid, 0) >= MAX_PICKS_PER_RALLY:
                break
            if per_video_counts.get(vid, 0) >= MAX_PICKS_PER_VIDEO:
                break
            a = c["action"]
            frame = int(a["frame"])
            key = (rid, frame)
            if key in seen:
                continue
            vname = video_names.get(vid, vid[:8])
            fps = video_fps.get(vid, 30.0)
            actions = (meta.get("actions_json") or {}).get("actions") or []
            prev_act = get_prev_action(actions, frame)
            pick = Pick(
                bucket="D_random_control",
                video_id=vid,
                video_name=vname,
                rally_id=rid,
                rally_order=meta["rally_order"],
                frame=frame,
                source_time_ms=source_time_ms(meta["start_ms"], frame, fps),
                fps=fps,
                pipeline_pid=int(a["playerTrackId"]),
                pipeline_action=str(a.get("action", "")),
                prev_frame=(int(prev_act["frame"]) if prev_act else None),
                prev_pid=(
                    int(prev_act["playerTrackId"])
                    if prev_act and prev_act.get("playerTrackId") is not None else None
                ),
                prev_action=(prev_act.get("action") if prev_act else None),
                conf=float(a["confidence"]),
                notes=f"high-conf control (conf={float(a['confidence']):.2f})",
            )
            picks.append(pick)
            seen.add(key)
            per_rally_counts[rid] = per_rally_counts.get(rid, 0) + 1
            per_video_counts[vid] = per_video_counts.get(vid, 0) + 1
    print(f"  Bucket D total: {len(picks)} picks")
    return picks


def bucket_e_confidence(
    rally_meta: dict[str, dict[str, Any]],
    video_names: dict[str, str],
    video_fps: dict[str, float],
    existing_gt: dict[str, int],
    per_rally_counts: dict[str, int],
    per_video_counts: dict[str, int],
    seen: set[tuple[str, int]],
    rng: random.Random,
) -> list[Pick]:
    """Stratified sample across confidence percentiles: 5 low / 5 mid / 10 high.

    These may overlap with A/B/C/D rallies (but not duplicate contact keys)
    since the goal is to calibrate the confidence-precision curve broadly."""
    # Build pools.
    all_actions: list[dict[str, Any]] = []
    for rid, meta in rally_meta.items():
        if meta.get("actions_pipeline_version") != ACTION_PIPELINE_VERSION:
            continue
        if existing_gt.get(rid, 0) > MAX_EXISTING_GT_ROWS:
            continue
        actions = (meta.get("actions_json") or {}).get("actions") or []
        for a in actions:
            pid = a.get("playerTrackId")
            if pid is None or pid == -1:
                continue
            conf = a.get("confidence")
            if conf is None:
                continue
            all_actions.append({
                "rally_id": rid,
                "meta": meta,
                "action": a,
                "conf": float(conf),
            })
    low = [c for c in all_actions if c["conf"] < 0.3]
    mid = [c for c in all_actions if 0.3 <= c["conf"] < 0.6]
    high = [c for c in all_actions if c["conf"] >= 0.7]
    print(
        f"  Bucket E pool: low={len(low)} mid={len(mid)} high={len(high)}"
    )
    # Sort each tier by "is rally already touched by other buckets" desc, then
    # random — this densifies E onto existing rallies where possible.
    def _densify_key(c: dict[str, Any]) -> tuple[int, float]:
        rid = c["rally_id"]
        already = per_rally_counts.get(rid, 0)
        # Higher already-count → higher priority (negative for asc sort).
        return (-already, rng.random())
    low.sort(key=_densify_key)
    mid.sort(key=_densify_key)
    high.sort(key=_densify_key)

    picks: list[Pick] = []
    for tier, pool, n in (("low", low, E_LOW), ("mid", mid, E_MID), ("high", high, E_HIGH)):
        added = 0
        for c in pool:
            if added >= n:
                break
            rid = c["rally_id"]
            meta = c["meta"]
            vid = meta["video_id"]
            if per_rally_counts.get(rid, 0) >= MAX_PICKS_PER_RALLY:
                continue
            if per_video_counts.get(vid, 0) >= MAX_PICKS_PER_VIDEO:
                continue
            a = c["action"]
            frame = int(a["frame"])
            key = (rid, frame)
            if key in seen:
                continue
            vname = video_names.get(vid, vid[:8])
            fps = video_fps.get(vid, 30.0)
            actions = (meta.get("actions_json") or {}).get("actions") or []
            prev_act = get_prev_action(actions, frame)
            pick = Pick(
                bucket=f"E_confidence_{tier}",
                video_id=vid,
                video_name=vname,
                rally_id=rid,
                rally_order=meta["rally_order"],
                frame=frame,
                source_time_ms=source_time_ms(meta["start_ms"], frame, fps),
                fps=fps,
                pipeline_pid=int(a["playerTrackId"]),
                pipeline_action=str(a.get("action", "")),
                prev_frame=(int(prev_act["frame"]) if prev_act else None),
                prev_pid=(
                    int(prev_act["playerTrackId"])
                    if prev_act and prev_act.get("playerTrackId") is not None else None
                ),
                prev_action=(prev_act.get("action") if prev_act else None),
                conf=float(a["confidence"]),
                notes=f"{tier}-conf tier (conf={c['conf']:.2f})",
            )
            picks.append(pick)
            seen.add(key)
            per_rally_counts[rid] = per_rally_counts.get(rid, 0) + 1
            per_video_counts[vid] = per_video_counts.get(vid, 0) + 1
            added += 1
        print(f"    Bucket E.{tier}: {added}/{n}")
    print(f"  Bucket E total: {len(picks)} picks")
    return picks


# ---------------------------------------------------------------------------
# Render markdown checklist
# ---------------------------------------------------------------------------

BUCKET_HEADERS = {
    "A_cascade": "Bucket A — Cascade-shape (same-player back-to-back)",
    "B_cross_team": "Bucket B — Mid-rally cross-team (F3/F5 shape)",
    "C_at_net_attack": "Bucket C — At-net attacks (block candidates / FP candidates)",
    "D_random_control": "Bucket D — Random / control (high-confidence)",
    "E_confidence_low": "Bucket E.low — Confidence percentile (conf < 0.3)",
    "E_confidence_mid": "Bucket E.mid — Confidence percentile (0.3 <= conf < 0.6)",
    "E_confidence_high": "Bucket E.high — Confidence percentile (conf >= 0.7)",
}
BUCKET_ORDER = [
    "A_cascade", "B_cross_team", "C_at_net_attack", "D_random_control",
    "E_confidence_low", "E_confidence_mid", "E_confidence_high",
]


def render_markdown(picks: list[Pick]) -> str:
    distinct_rallies = len({p.rally_id for p in picks})
    distinct_videos = len({p.video_id for p in picks})
    by_bucket: dict[str, list[Pick]] = defaultdict(list)
    for p in picks:
        by_bucket[p.bucket].append(p)

    lines: list[str] = []
    lines.append("# T2 GT-expansion labeling checklist")
    lines.append("")
    lines.append(
        "**Goal**: ~150 contacts labeled across ~30 rallies. Unlocks A1.v2 "
        "precision measurement, PGM Phase B per-action-type weights, "
        "contact-detector at-net FP retrain."
    )
    lines.append("")
    lines.append(
        f"**Sample**: {len(picks)} contacts across {distinct_rallies} rallies "
        f"and {distinct_videos} videos."
    )
    lines.append("")
    lines.append("**Time estimate**: ~60-90 s per contact = ~2.5-3.5 hours total.")
    lines.append("")
    lines.append("**How to label**: open each rally in the rally editor at the listed "
                 "source-video time, look at the contact, and either:")
    lines.append("- Confirm pipeline pick (do nothing, GT row created as-is)")
    lines.append("- Correct attribution (change player in editor → resolver updates GT row)")
    lines.append("- Correct action type (change type)")
    lines.append("- Delete (if the contact is a false positive / non-event)")
    lines.append("")
    lines.append("**Counts per bucket:**")
    lines.append("")
    for b in BUCKET_ORDER:
        n = len(by_bucket.get(b, []))
        lines.append(f"- `{b}`: {n}")
    lines.append("")

    idx = 0
    for bucket in BUCKET_ORDER:
        bucket_picks = by_bucket.get(bucket, [])
        if not bucket_picks:
            continue
        lines.append(f"## {BUCKET_HEADERS[bucket]}")
        lines.append("")
        lines.append(
            "| # | Video | Rally (#order, uuid) | Frame | Source time | "
            "Pipeline pick (pid / action) | Prev (frame / pid / action) | "
            "Conf | Notes |"
        )
        lines.append(
            "|--:|:------|:--------------------|------:|:------------|:-----------------------------|:----------------------------|----:|:------|"
        )
        for p in bucket_picks:
            idx += 1
            rally_short = p.rally_id.split("-")[0]
            pick_pid_str = (
                f"p{p.pipeline_pid}" if p.pipeline_pid is not None else "?"
            )
            pipeline_str = f"{pick_pid_str} / {p.pipeline_action or '?'}"
            prev_str = (
                f"f{p.prev_frame} / p{p.prev_pid} / {p.prev_action}"
                if p.prev_frame is not None else "—"
            )
            conf_str = f"{p.conf:.2f}" if p.conf is not None else "—"
            lines.append(
                f"| {idx} | {p.video_name} | #{p.rally_order} `{rally_short}` | "
                f"{p.frame} | **{fmt_time(p.source_time_ms)}** | {pipeline_str} | "
                f"{prev_str} | {conf_str} | {p.notes} |"
            )
        lines.append("")

    lines.append("## After labeling")
    lines.append("")
    lines.append("Tell Claude \"done with T2 labeling\" and I will:")
    lines.append("")
    lines.append("1. Re-run A1.v2 + S4 + role-attribution probes against the new GT.")
    lines.append("2. Compute per-bucket precision.")
    lines.append("3. Measure contact-detector at-net FP rate.")
    lines.append("4. Surface a precision-vs-confidence curve.")
    lines.append("5. Decide which workstream(s) ship next based on cleaned data.")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    print(f"[1/6] Loading video metadata...")
    video_names, video_fps = load_video_metadata()
    print(f"      {len(video_names)} videos loaded.")

    print(f"[2/6] Loading rally metadata (with actions_json)...")
    rally_meta = load_rally_metadata()
    print(f"      {len(rally_meta)} rallies loaded.")
    v2_count = sum(
        1 for m in rally_meta.values()
        if m.get("actions_pipeline_version") == ACTION_PIPELINE_VERSION
    )
    print(f"      {v2_count} rallies at current pipeline version "
          f"({ACTION_PIPELINE_VERSION}).")

    print(f"[3/6] Loading existing GT counts...")
    existing_gt = load_existing_gt_counts()
    overlabeled = sum(1 for n in existing_gt.values() if n > MAX_EXISTING_GT_ROWS)
    print(f"      {len(existing_gt)} rallies have GT; {overlabeled} are "
          f"over the {MAX_EXISTING_GT_ROWS}-row threshold (will be skipped).")

    # Tracking state shared across buckets.
    per_rally_counts: dict[str, int] = defaultdict(int)
    per_video_counts: dict[str, int] = defaultdict(int)
    seen: set[tuple[str, int]] = set()
    all_picks: list[Pick] = []

    print(f"[4/6] Bucket A — Cascade-shape (target {TARGETS['A_cascade']})...")
    picks_a = bucket_a_cascade(
        rally_meta, video_names, video_fps, existing_gt,
        per_rally_counts, per_video_counts, seen, rng,
    )
    all_picks.extend(picks_a)

    print(f"[4/6] Bucket B — Mid-rally cross-team (target {TARGETS['B_cross_team']})...")
    picks_b = bucket_b_cross_team(
        rally_meta, video_names, video_fps, existing_gt,
        per_rally_counts, per_video_counts, seen, rng,
    )
    all_picks.extend(picks_b)

    print(f"[4/6] Bucket C — At-net attacks (target {TARGETS['C_at_net_attack']})...")
    picks_c = bucket_c_at_net_attack(
        rally_meta, video_names, video_fps, existing_gt,
        per_rally_counts, per_video_counts, seen, rng,
    )
    all_picks.extend(picks_c)

    # Set of rallies already touched by A/B/C — Bucket D avoids these.
    rallies_used_by_abc = {p.rally_id for p in all_picks}

    print(f"[5/6] Bucket D — Random control (target {TARGETS['D_random_control']})...")
    picks_d = bucket_d_random_control(
        rally_meta, video_names, video_fps, existing_gt,
        per_rally_counts, per_video_counts, seen, rng, rallies_used_by_abc,
    )
    all_picks.extend(picks_d)

    print(f"[5/6] Bucket E — Confidence percentiles (target {TARGETS['E_confidence']})...")
    picks_e = bucket_e_confidence(
        rally_meta, video_names, video_fps, existing_gt,
        per_rally_counts, per_video_counts, seen, rng,
    )
    all_picks.extend(picks_e)

    # Write outputs.
    print(f"[6/6] Writing outputs...")
    distinct_rallies = len({p.rally_id for p in all_picks})
    distinct_videos = len({p.video_id for p in all_picks})
    print(f"      Total picks: {len(all_picks)}")
    print(f"      Distinct rallies: {distinct_rallies}")
    print(f"      Distinct videos: {distinct_videos}")
    by_bucket: dict[str, int] = defaultdict(int)
    for p in all_picks:
        by_bucket[p.bucket] += 1
    for b in BUCKET_ORDER:
        print(f"      {b}: {by_bucket.get(b, 0)}")

    md = render_markdown(all_picks)
    CHECKLIST_MD.write_text(md)
    print(f"      Wrote {CHECKLIST_MD}")

    picks_list_with_idx: list[dict[str, Any]] = []
    for i, p in enumerate(all_picks, 1):
        d = asdict(p)
        d["idx"] = i
        picks_list_with_idx.append(d)
    PICKS_JSON.write_text(json.dumps(picks_list_with_idx, indent=2, default=str))
    print(f"      Wrote {PICKS_JSON}")


if __name__ == "__main__":
    main()
