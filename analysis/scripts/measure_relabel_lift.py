"""measure_relabel_lift.py — Phase 1 validation gate for relabel-with-crops.

For each baseline fixture:
  1. Snapshot DB state (player_reference_crops + per-rally player_tracks columns
     + match_analysis_json) to disk.
  2. DELETE its rows from player_reference_crops.
  3. Run `match-players <video>` (blind — no crops in DB).
  4. Run `reattribute-actions <video>` (so actions_json picks up new ttp).
  5. Score Surface A (bench_attribution against action_ground_truth_json) and,
     if the fixture has a click-GT verdicts file, Surface B.
  6. RESTORE player_reference_crops rows from the snapshot.
  7. Run `relabel-with-crops <video>` (replays Pass 2 with frozen crops).
  8. Run `reattribute-actions <video>`.
  9. Score Surface A + Surface B again → AFTER.
 10. Report per-fixture before/after delta, then aggregate gate verdicts.

The script is restartable per-fixture: each fixture's snapshot lives in its own
directory under reports/attribution_rebuild/relabel_lift/<run_ts>/<fixture>/.
On any subprocess failure for a fixture, the snapshot is restored and the
fixture is marked as errored (other fixtures continue).

Plan §6 gates:
  - Surface A: AFTER ≥ 58% aggregate (≥ +15pp vs locked baseline 43.8%);
    no fixture regresses > 1pp vs the locked baseline; lulu and cuco show ≥ +5pp
    over BLIND (the worst baseline fixtures should benefit most from crops).
  - Surface B: AFTER ≥ 90% aggregate click-GT direct accuracy; ≥ 3 fixtures at
    100%; no fixture below 75%.

Usage:
    cd analysis
    uv run python scripts/measure_relabel_lift.py
    uv run python scripts/measure_relabel_lift.py --fixtures cece tata
    uv run python scripts/measure_relabel_lift.py --skip-blind  # debug only
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_ANALYSIS_DIR = Path(__file__).resolve().parents[1]
if str(_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS_DIR))

from rallycut.evaluation.attribution_bench import (  # noqa: E402
    WRONG_CATEGORIES,
    aggregate,
    score_rally,
)
from rallycut.evaluation.db import get_connection  # noqa: E402

REPO_ROOT = _ANALYSIS_DIR.parent
ANALYSIS_REPORTS = _ANALYSIS_DIR / "reports"
ATTR_REPORTS = ANALYSIS_REPORTS / "attribution_rebuild"
LOCKED_BASELINE_PATH = ATTR_REPORTS / "baseline_2026_04_24.json"
FIXTURE_REGISTRY = ATTR_REPORTS / "fixture_video_ids_2026_04_24.json"
VERDICTS_DIR = REPO_ROOT / "reports" / "session3" / "verdicts"

# Fixture short-name → 8-char video-id prefix used in click-GT verdicts.
# Only the intersection with baseline fixtures matters for Surface B.
CLICK_GT_VSHORTS_BY_FIXTURE = {
    "tata": "7d77980f",
    "rere": "808a5618",
    "cece": "950fbe5d",
    "yeye": "eb693a6f",
    "lulu": "4f2bd66a",
}

CLICK_GT_TOLERANCE_FRAMES = 10


# ---------------------------------------------------------------------------
# DB snapshot / restore
# ---------------------------------------------------------------------------


@dataclass
class FixtureSnapshot:
    """Per-fixture pre-mutation snapshot saved to disk for safe restore."""

    fixture: str
    video_id: str
    snapshot_dir: Path
    reference_crops: list[dict[str, Any]] = field(default_factory=list)
    match_analysis_json: Any = None
    rallies: list[dict[str, Any]] = field(default_factory=list)
    # rallies entries: {rally_id, actions_json, contacts_json, primary_track_ids}

    def save(self) -> None:
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "fixture": self.fixture,
            "video_id": self.video_id,
            "captured_at": datetime.now(UTC).isoformat(),
            "reference_crops": self.reference_crops,
            "match_analysis_json": self.match_analysis_json,
            "rallies": self.rallies,
        }
        (self.snapshot_dir / "snapshot.json").write_text(
            json.dumps(payload, indent=2, default=str)
        )


def capture_snapshot(
    fixture: str, video_id: str, snapshot_dir: Path
) -> FixtureSnapshot:
    snap = FixtureSnapshot(fixture=fixture, video_id=video_id, snapshot_dir=snapshot_dir)
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT id, player_id, frame_ms, bbox_x, bbox_y, bbox_w, bbox_h,
                      s3_key, created_at
               FROM player_reference_crops
               WHERE video_id = %s
               ORDER BY player_id, created_at""",
            [video_id],
        )
        for row in cur.fetchall():
            snap.reference_crops.append({
                "id": str(row[0]),
                "player_id": int(row[1]),
                "frame_ms": int(row[2]),
                "bbox_x": float(row[3]),
                "bbox_y": float(row[4]),
                "bbox_w": float(row[5]),
                "bbox_h": float(row[6]),
                "s3_key": row[7],
                "created_at": row[8].isoformat() if row[8] else None,
            })

        cur.execute(
            "SELECT match_analysis_json FROM videos WHERE id = %s",
            [video_id],
        )
        row = cur.fetchone()
        snap.match_analysis_json = row[0] if row else None

        cur.execute(
            """SELECT r.id::text, pt.actions_json, pt.contacts_json,
                      pt.primary_track_ids
               FROM rallies r JOIN player_tracks pt ON pt.rally_id = r.id
               WHERE r.video_id = %s""",
            [video_id],
        )
        for rid, actions, contacts, ptids in cur.fetchall():
            snap.rallies.append({
                "rally_id": rid,
                "actions_json": actions,
                "contacts_json": contacts,
                "primary_track_ids": ptids,
            })

    snap.save()
    return snap


def restore_snapshot(snap: FixtureSnapshot) -> None:
    """Restore the fixture to its pre-cycle state. Idempotent."""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "DELETE FROM player_reference_crops WHERE video_id = %s",
            [snap.video_id],
        )
        for c in snap.reference_crops:
            cur.execute(
                """INSERT INTO player_reference_crops
                       (id, video_id, player_id, frame_ms,
                        bbox_x, bbox_y, bbox_w, bbox_h, s3_key, created_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                [
                    c["id"], snap.video_id, c["player_id"], c["frame_ms"],
                    c["bbox_x"], c["bbox_y"], c["bbox_w"], c["bbox_h"],
                    c["s3_key"], c["created_at"],
                ],
            )
        cur.execute(
            "UPDATE videos SET match_analysis_json = %s WHERE id = %s",
            [
                json.dumps(snap.match_analysis_json) if snap.match_analysis_json
                else None,
                snap.video_id,
            ],
        )
        for r in snap.rallies:
            cur.execute(
                """UPDATE player_tracks
                   SET actions_json = %s,
                       contacts_json = %s,
                       primary_track_ids = %s
                   WHERE rally_id = %s""",
                [
                    json.dumps(r["actions_json"]) if r["actions_json"] else None,
                    json.dumps(r["contacts_json"]) if r["contacts_json"] else None,
                    json.dumps(r["primary_track_ids"])
                    if r["primary_track_ids"] is not None else None,
                    r["rally_id"],
                ],
            )
        conn.commit()


def delete_reference_crops(video_id: str) -> int:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "DELETE FROM player_reference_crops WHERE video_id = %s",
            [video_id],
        )
        n = cur.rowcount
        conn.commit()
    return n


# ---------------------------------------------------------------------------
# Subprocess CLI invocations
# ---------------------------------------------------------------------------


def _run_cli(args: list[str], log_path: Path, label: str) -> tuple[float, int]:
    cmd = ["uv", "run", "rallycut", *args]
    t0 = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as fh:
        fh.write(f"$ {' '.join(shlex.quote(c) for c in cmd)}\n\n")
        proc = subprocess.run(
            cmd, cwd=_ANALYSIS_DIR, capture_output=True, text=True
        )
        fh.write(proc.stdout)
        if proc.stderr:
            fh.write("\n--- stderr ---\n")
            fh.write(proc.stderr)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-15:])
        print(f"      [ERROR] {label} exit={proc.returncode} in {elapsed:.1f}s")
        print(f"      log tail:\n{tail}")
    else:
        print(f"      [{label}] OK in {elapsed:.1f}s")
    return elapsed, proc.returncode


def run_match_players(video_id: str, log_path: Path) -> int:
    return _run_cli(
        ["match-players", video_id, "--quiet"], log_path, "match-players"
    )[1]


def run_remap_track_ids(video_id: str, log_path: Path) -> int:
    return _run_cli(
        ["remap-track-ids", video_id, "--quiet"], log_path, "remap-track-ids"
    )[1]


def run_reattribute_actions(video_id: str, log_path: Path) -> int:
    return _run_cli(
        ["reattribute-actions", video_id], log_path, "reattribute-actions"
    )[1]


def run_relabel_with_crops(video_id: str, log_path: Path) -> int:
    return _run_cli(
        ["relabel-with-crops", video_id, "--quiet"], log_path, "relabel-with-crops"
    )[1]


# ---------------------------------------------------------------------------
# Surface A — action-GT scoring via bench_attribution primitives
# ---------------------------------------------------------------------------


def _ttp_by_rally_id(video_id: str) -> dict[str, dict[int, int]]:
    """Per-rally raw-trackId → canonical-pid (1-4).

    Source priority — mirrors `eval_action_detection._load_track_to_player_maps`:
        1. `videos.canonical_pid_map_json[rallies][rid]` — ref-crop-anchored,
           deterministic across re-runs. Source of truth when populated.
        2. `videos.match_analysis_json[].trackToPlayer` — legacy Hungarian
           output; used when canonical map absent for the rally (e.g. video
           lacks a full 4-pid ref-crop set).
    """
    out: dict[str, dict[int, int]] = {}
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT match_analysis_json, canonical_pid_map_json "
            "FROM videos WHERE id = %s",
            [video_id],
        )
        row = cur.fetchone()
    ma = row[0] if row and row[0] else {}
    cmap = row[1] if row and row[1] else {}
    # Legacy first; canonical wins per-rally when present (full 1:1 replacement).
    for entry in ma.get("rallies", []) if isinstance(ma, dict) else []:
        rid = entry.get("rallyId") or entry.get("rally_id", "")
        ttp = entry.get("trackToPlayer") or entry.get("track_to_player", {})
        if rid and ttp:
            out[rid] = {int(k): int(v) for k, v in ttp.items()}
    if isinstance(cmap, dict):
        for rid, rally_map in (cmap.get("rallies") or {}).items():
            if rid and rally_map:
                out[rid] = {int(k): int(v) for k, v in rally_map.items()}
    return out


def _team_templates_pid_to_team(video_id: str) -> dict[int, str]:
    """canonical_pid → 'A'|'B' from match_analysis.teamTemplates."""
    out: dict[int, str] = {}
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT match_analysis_json FROM videos WHERE id = %s",
            [video_id],
        )
        row = cur.fetchone()
    ma = row[0] if row and row[0] else {}
    tt = ma.get("teamTemplates") or {}
    for team_label, entry in tt.items():
        team_letter = "A" if str(team_label) == "0" else "B"
        for pid in entry.get("playerIds", []):
            out[int(pid)] = team_letter
    return out


def load_rallies_for_surface_a(
    video_id: str, fixture: str
) -> list[dict[str, Any]]:
    """Load rallies + resolve GT.trackId → display pid.

    Post-schema-fix (commits 3cf67c1/fa05965/737bcf0/66dddb1):
      - action_ground_truth_json[i] stores `trackId` (raw BoT-SORT, stable
        across re-tracks). Resolve via canonical_pid_map_json (preferred)
        or legacy trackToPlayer to get the display pid that bench_attribution
        compares against.
      - actions_json[i].playerTrackId is already the display pid post
        reattribute-actions + `_align_canonical_to_legacy`. Compare directly.
      - team_assignments is rebuilt from teamTemplates so it's keyed by
        display pid (1-4), which is what bench_attribution.score_rally
        expects. Legacy actions_json.teamAssignments is raw-tid-keyed and
        not usable here.
    """
    pid_to_team = _team_templates_pid_to_team(video_id)
    ttp_by_rally = _ttp_by_rally_id(video_id)

    rallies: list[dict[str, Any]] = []
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT r.id, r.start_ms, r.end_ms,
                      pt.action_ground_truth_json,
                      pt.actions_json,
                      pt.contacts_json,
                      pt.primary_track_ids
               FROM rallies r JOIN player_tracks pt ON pt.rally_id = r.id
               WHERE r.video_id = %s
                 AND pt.action_ground_truth_json IS NOT NULL
                 AND jsonb_array_length(pt.action_ground_truth_json::jsonb) > 0
               ORDER BY r.start_ms""",
            [video_id],
        )
        for rid, start_ms, end_ms, gt, actions, contacts, ptids in cur.fetchall():
            rid_str = str(rid)
            ttp = ttp_by_rally.get(rid_str, {})

            # Resolve GT.trackId → display pid for score_rally compatibility.
            # Prefer the new `trackId` field; tolerate legacy `playerTrackId`
            # rows during the migration window.
            gt_resolved: list[dict[str, Any]] = []
            for g in gt or []:
                if not isinstance(g, dict):
                    continue
                g_copy = dict(g)
                raw_tid = g_copy.get("trackId")
                if raw_tid is None:
                    raw_tid = g_copy.get("playerTrackId")
                display = ttp.get(int(raw_tid)) if raw_tid is not None else None
                # Score with display pid; bench_attribution reads playerTrackId.
                g_copy["playerTrackId"] = display if display is not None else -1
                gt_resolved.append(g_copy)

            # Pipeline actions already carry display pid post-reattribute.
            raw_actions = (actions or {}).get("actions", []) if actions else []
            pipeline_actions = [dict(a) for a in raw_actions if isinstance(a, dict)]

            team_assignments = {
                str(pid): team for pid, team in pid_to_team.items()
            }
            serving_team = (actions or {}).get("servingTeam") if actions else None
            pipeline_contacts = (contacts or {}).get("contacts", []) if contacts else []
            rally = {
                "rally_id": rid_str,
                "video_id": video_id,
                "fixture": fixture,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "primary_track_ids": ptids,
                "team_assignments": team_assignments,
                "serving_team": serving_team,
                "gt_actions": gt_resolved,
                "pipeline_actions": pipeline_actions,
                "pipeline_contacts": pipeline_contacts,
            }
            scored = score_rally(rally)
            rally["matches"] = scored["matches"]
            rally["rally_totals"] = scored["rally_totals"]
            rallies.append(rally)
    return rallies


# ---------------------------------------------------------------------------
# Surface B — click-GT direct accuracy (Day-4 protocol)
# ---------------------------------------------------------------------------


def find_verdicts_path(vshort: str) -> Path | None:
    matches = sorted(VERDICTS_DIR.glob(f"verdicts_{vshort}_*.json"))
    return matches[-1] if matches else None


def load_click_gt(verdicts_path: Path) -> dict[str, dict[int, dict[str, Any]]]:
    """Return {rally_short: {frame_int: {actor_pid, actor_tid}}}."""
    raw = json.loads(verdicts_path.read_text())
    out: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for key, v in raw.get("verdicts", {}).items():
        rally_short, _, frame_str = key.rpartition("-")
        if not frame_str.isdigit():
            continue
        out[rally_short][int(frame_str)] = {
            "actor_tid": v.get("actor_tid"),
            "actor_pid": v.get("actor_pid"),
        }
    return dict(out)


def load_pipeline_actions_by_rally_short(
    video_id: str,
) -> dict[str, list[dict[str, Any]]]:
    """Per-rally-short list of pipeline actions. playerTrackId is the
    display pid post-reattribute + `_align_canonical_to_legacy` — no
    additional ttp resolution needed."""
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT r.id::text, pt.actions_json
               FROM rallies r JOIN player_tracks pt ON pt.rally_id = r.id
               WHERE r.video_id = %s""",
            [video_id],
        )
        for rid, actions_json in cur.fetchall():
            short = rid[:8]
            actions = (actions_json or {}).get("actions", []) if actions_json else []
            out[short] = [
                dict(a) for a in (actions if isinstance(actions, list) else [])
                if isinstance(a, dict)
            ]
    return dict(out)


def measure_click_gt(video_id: str, vshort: str) -> dict[str, Any] | None:
    """Surface B measurement. Returns None if no click-GT for this fixture.

    Pid labels are arbitrary slot identifiers — in production the user binds
    names to slots ("this is Alice"), so any consistent permutation of pid
    labels is equally correct. The verdicts file's `actor_pid` was captured
    under whatever editor convention was live at click time; today's
    pipeline produces pids under the current convention. To compare them
    fairly, we collect (gt_pid, pl_pid) pairs and search the 24 permutations
    of {1,2,3,4} for the one that maximizes matches. acc_full reports the
    BEST permutation; acc_identity reports the literal-match score for
    diagnostic purposes (low acc_identity + high acc_full = pure convention
    drift; low both = real attribution failure).
    """
    import itertools

    verdicts_path = find_verdicts_path(vshort)
    if verdicts_path is None:
        return None
    click_gt = load_click_gt(verdicts_path)
    if not click_gt:
        return None

    pipeline_by_short = load_pipeline_actions_by_rally_short(video_id)

    # Map verdict's 8-char rally_short to full rally_id, then look up ttp.
    short_to_full: dict[str, str] = {}
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id::text FROM rallies WHERE video_id = %s", [video_id]
        )
        for row in cur.fetchall():
            short_to_full[row[0][:8]] = row[0]
    ttp_by_rally = _ttp_by_rally_id(video_id)

    n_no_actor_pid = 0
    n_no_pipeline = 0
    # (rally_short, gt_pid, pl_pid) tuples — collect first, score after.
    pairs: list[tuple[str, int, int]] = []

    for rally_short, frames in click_gt.items():
        prod_actions = pipeline_by_short.get(rally_short, [])
        full_rid = short_to_full.get(rally_short)
        ttp = ttp_by_rally.get(full_rid, {}) if full_rid else {}
        for frame, v in frames.items():
            actor_tid = v.get("actor_tid")
            if actor_tid is not None:
                resolved = ttp.get(int(actor_tid))
                actor_pid = resolved if resolved is not None else v.get("actor_pid")
            else:
                actor_pid = v.get("actor_pid")
            if actor_pid is None:
                n_no_actor_pid += 1
                continue

            best = None
            best_d = CLICK_GT_TOLERANCE_FRAMES + 1
            for pa in prod_actions:
                f = int(pa.get("frame", -9999))
                d = abs(f - frame)
                if d < best_d:
                    best_d = d
                    best = pa
            if best is None:
                n_no_pipeline += 1
                continue
            pl_pid = best.get("playerTrackId")
            if pl_pid is None:
                n_no_pipeline += 1
                continue
            pairs.append((rally_short, int(actor_pid), int(pl_pid)))

    n_eval = len(pairs)
    if n_eval == 0:
        return {
            "verdicts_path": str(verdicts_path),
            "n_eval": 0,
            "n_no_actor_pid": n_no_actor_pid,
            "n_no_pipeline": n_no_pipeline,
            "correct": 0,
            "wrong": 0,
            "acc_full": 0.0,
            "acc_identity": 0.0,
            "best_perm": None,
            "per_rally": {},
        }

    # Score the identity perm (literal match) for diagnostic comparison.
    correct_identity = sum(1 for _, gt, pl in pairs if gt == pl)

    # Search 24 perms of {1,2,3,4} → pick the one that maximizes correct.
    pids = [1, 2, 3, 4]
    best_perm: dict[int, int] = {}
    best_correct = 0
    for perm in itertools.permutations(pids):
        # perm maps original pid (1,2,3,4) to permuted pid (perm[0..3]).
        mapping = dict(zip(pids, perm))
        c = sum(1 for _, gt, pl in pairs if gt == mapping.get(pl, pl))
        if c > best_correct:
            best_correct = c
            best_perm = mapping

    # Per-rally breakdown under the best perm.
    per_rally: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n": 0, "correct": 0, "wrong": 0, "no_pred": 0}
    )
    for rally_short, gt, pl in pairs:
        per_rally[rally_short]["n"] += 1
        if gt == best_perm.get(pl, pl):
            per_rally[rally_short]["correct"] += 1
        else:
            per_rally[rally_short]["wrong"] += 1

    return {
        "verdicts_path": str(verdicts_path),
        "n_eval": n_eval,
        "n_no_actor_pid": n_no_actor_pid,
        "n_no_pipeline": n_no_pipeline,
        "correct": best_correct,
        "wrong": n_eval - best_correct,
        "acc_full": best_correct / n_eval,
        "acc_identity": correct_identity / n_eval,
        "best_perm": {str(k): v for k, v in best_perm.items()},
        "per_rally": dict(per_rally),
    }


# ---------------------------------------------------------------------------
# Per-fixture cycle
# ---------------------------------------------------------------------------


@dataclass
class FixtureResult:
    fixture: str
    video_id: str
    error: str | None = None
    n_ref_crops: int = 0
    blind_surface_a: dict[str, Any] | None = None
    after_surface_a: dict[str, Any] | None = None
    blind_surface_b: dict[str, Any] | None = None
    after_surface_b: dict[str, Any] | None = None
    timings: dict[str, float] = field(default_factory=dict)


def _surface_a_for_fixture(rallies: list[dict[str, Any]]) -> dict[str, Any]:
    agg = aggregate(rallies)
    fxs = list(agg["per_fixture"].values())
    if not fxs:
        return {"counts": {"n_gt_actions": 0}, "rates": {}}
    return fxs[0]


def run_fixture(
    fixture: str,
    video_id: str,
    out_dir: Path,
    skip_blind: bool = False,
) -> FixtureResult:
    print(f"\n=== {fixture} ({video_id[:8]}) ===")
    snapshot_dir = out_dir / fixture
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    result = FixtureResult(fixture=fixture, video_id=video_id)

    # 1. Snapshot.
    print("  [snapshot] capturing DB state...")
    snap = capture_snapshot(fixture, video_id, snapshot_dir)
    result.n_ref_crops = len(snap.reference_crops)
    print(f"  [snapshot] {len(snap.reference_crops)} crops, "
          f"{len(snap.rallies)} rallies, "
          f"match_analysis={'present' if snap.match_analysis_json else 'absent'}")

    if result.n_ref_crops == 0:
        result.error = "fixture has 0 ref crops in DB — relabel cycle is no-op"
        print(f"  [skip] {result.error}")
        return result

    vshort = CLICK_GT_VSHORTS_BY_FIXTURE.get(fixture)

    try:
        # 2. Delete crops.
        if not skip_blind:
            print("  [blind] deleting reference crops...")
            n_del = delete_reference_crops(video_id)
            print(f"  [blind] deleted {n_del} crop rows")

            # 3. Blind match-players.
            print("  [blind] running match-players (blind, no crops)...")
            t0 = time.time()
            rc = run_match_players(video_id, snapshot_dir / "blind_match_players.log")
            result.timings["blind_match_players"] = time.time() - t0
            if rc != 0:
                raise RuntimeError(
                    f"match-players failed (rc={rc}); see "
                    f"{snapshot_dir / 'blind_match_players.log'}"
                )

            # NOTE: skip remap-track-ids here. If we remap, positions get
            # canonicalized and the rallyScratchpad's TrackAppearanceStats
            # (keyed by RAW tracker ids) becomes orphaned — relabel-with-crops
            # then writes a ttp in raw-id space that the now-canonical
            # positions can't honor, making AFTER == BLIND. Score in-memory
            # via the trackToPlayer mapping instead.

            # 4. Reattribute actions.
            print("  [blind] running reattribute-actions...")
            t0 = time.time()
            rc = run_reattribute_actions(
                video_id, snapshot_dir / "blind_reattribute.log"
            )
            result.timings["blind_reattribute"] = time.time() - t0
            if rc != 0:
                raise RuntimeError(
                    f"reattribute-actions failed (rc={rc}); see "
                    f"{snapshot_dir / 'blind_reattribute.log'}"
                )

            # 5. Score BLIND.
            print("  [blind] scoring Surface A + Surface B...")
            blind_rallies = load_rallies_for_surface_a(video_id, fixture)
            result.blind_surface_a = _surface_a_for_fixture(blind_rallies)
            if vshort is not None:
                result.blind_surface_b = measure_click_gt(video_id, vshort)
            counts = result.blind_surface_a["counts"]
            n = counts["n_gt_actions"]
            corr_rate = result.blind_surface_a["rates"].get("correct_rate", 0.0)
            print(f"    BLIND Surface A: n={n} correct={counts['correct']} "
                  f"({corr_rate:.1%})")
            if result.blind_surface_b:
                b = result.blind_surface_b
                identity_pct = b.get("acc_identity", 0.0) * 100
                print(f"    BLIND Surface B: n={b['n_eval']} "
                      f"correct={b['correct']} ({b['acc_full']:.1%} best-perm, "
                      f"{identity_pct:.1f}% identity-perm)")

        # 6. Restore crops only (NOT match_analysis — the blind run wrote a
        # fresh scratchpad we want relabel to consume).
        print("  [after] restoring reference crops...")
        with get_connection() as conn, conn.cursor() as cur:
            cur.execute(
                "DELETE FROM player_reference_crops WHERE video_id = %s",
                [video_id],
            )
            for c in snap.reference_crops:
                cur.execute(
                    """INSERT INTO player_reference_crops
                           (id, video_id, player_id, frame_ms,
                            bbox_x, bbox_y, bbox_w, bbox_h, s3_key, created_at)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                    [
                        c["id"], video_id, c["player_id"], c["frame_ms"],
                        c["bbox_x"], c["bbox_y"], c["bbox_w"], c["bbox_h"],
                        c["s3_key"], c["created_at"],
                    ],
                )
            conn.commit()
        print(f"  [after] restored {len(snap.reference_crops)} crops")

        # 7. relabel-with-crops.
        print("  [after] running relabel-with-crops...")
        t0 = time.time()
        rc = run_relabel_with_crops(
            video_id, snapshot_dir / "after_relabel.log"
        )
        result.timings["after_relabel"] = time.time() - t0
        if rc != 0:
            raise RuntimeError(
                f"relabel-with-crops failed (rc={rc}); see "
                f"{snapshot_dir / 'after_relabel.log'}"
            )

        # NOTE: skip remap-track-ids here too — same reason. Positions
        # remain in the raw blind tracker space; scoring applies the relabel
        # ttp in-memory.

        # 8. reattribute-actions.
        print("  [after] running reattribute-actions...")
        t0 = time.time()
        rc = run_reattribute_actions(
            video_id, snapshot_dir / "after_reattribute.log"
        )
        result.timings["after_reattribute"] = time.time() - t0
        if rc != 0:
            raise RuntimeError(
                f"reattribute-actions failed (rc={rc}); see "
                f"{snapshot_dir / 'after_reattribute.log'}"
            )

        # 9. Score AFTER.
        print("  [after] scoring Surface A + Surface B...")
        after_rallies = load_rallies_for_surface_a(video_id, fixture)
        result.after_surface_a = _surface_a_for_fixture(after_rallies)
        if vshort is not None:
            result.after_surface_b = measure_click_gt(video_id, vshort)
        counts = result.after_surface_a["counts"]
        n = counts["n_gt_actions"]
        corr_rate = result.after_surface_a["rates"].get("correct_rate", 0.0)
        print(f"    AFTER Surface A: n={n} correct={counts['correct']} "
              f"({corr_rate:.1%})")
        if result.after_surface_b:
            b = result.after_surface_b
            identity_pct = b.get("acc_identity", 0.0) * 100
            best_perm = b.get("best_perm")
            print(f"    AFTER Surface B: n={b['n_eval']} "
                  f"correct={b['correct']} ({b['acc_full']:.1%} best-perm, "
                  f"{identity_pct:.1f}% identity-perm, perm={best_perm})")

    except Exception as e:
        result.error = repr(e)
        print(f"  [ERROR] {e!r} — restoring snapshot")
        try:
            restore_snapshot(snap)
            print("  [restore] OK")
        except Exception as restore_err:
            print(f"  [RESTORE FAILED] {restore_err!r} — manual recovery from "
                  f"{snapshot_dir / 'snapshot.json'} required")
        return result

    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _format_pp(v: float) -> str:
    sign = "+" if v >= 0 else ""
    return f"{sign}{v * 100:.1f}pp"


def _locked_baseline_per_fixture() -> dict[str, dict[str, Any]]:
    payload = json.loads(LOCKED_BASELINE_PATH.read_text())
    out: dict[str, dict[str, Any]] = {}
    for fx, counts in payload.get("fixtures", {}).items():
        rates = payload.get("fixture_rates", {}).get(fx, {})
        out[fx] = {"counts": counts, "rates": rates}
    return out


def render_report(results: list[FixtureResult]) -> dict[str, Any]:
    locked = _locked_baseline_per_fixture()

    print("\n" + "=" * 100)
    print("RELABEL LIFT — Surface A (action-GT) per fixture")
    print("=" * 100)
    print(f"{'fixture':8s} {'n_gt':>4s}  {'locked':>8s}  {'BLIND':>8s}  "
          f"{'AFTER':>8s}  {'Δ vs blind':>10s}  {'Δ vs locked':>11s}  "
          f"{'gate':>15s}")

    blind_total_n = blind_total_correct = 0
    after_total_n = after_total_correct = 0
    locked_total_n = locked_total_correct = 0
    fixtures_regress_locked: list[str] = []
    worst_two_lift: dict[str, float] = {}

    for r in results:
        if r.error and r.after_surface_a is None:
            print(f"{r.fixture:8s}   ERROR: {r.error}")
            continue
        a_after = r.after_surface_a or {}
        a_blind = r.blind_surface_a or {}
        c_after = a_after.get("counts", {})
        c_blind = a_blind.get("counts", {})
        n_gt = c_after.get("n_gt_actions") or c_blind.get("n_gt_actions") or 0
        rate_after = a_after.get("rates", {}).get("correct_rate", 0.0) if a_after else 0.0
        rate_blind = a_blind.get("rates", {}).get("correct_rate", 0.0) if a_blind else 0.0

        locked_fx = locked.get(r.fixture, {})
        locked_counts = locked_fx.get("counts", {})
        locked_n = locked_counts.get("n_gt_actions", n_gt)
        locked_correct = locked_counts.get("correct", 0)
        locked_rate = locked_fx.get("rates", {}).get(
            "correct_rate", locked_correct / locked_n if locked_n else 0.0
        )

        delta_blind = rate_after - rate_blind
        delta_locked = rate_after - locked_rate
        regressed = delta_locked < -0.01  # 1pp drop
        gate = "OK" if not regressed else "REGRESS>1pp"
        if r.fixture in ("lulu", "cuco"):
            worst_two_lift[r.fixture] = delta_blind
            if delta_blind < 0.05:
                gate = f"{gate}/lift<5pp" if regressed else "lift<5pp"

        print(f"{r.fixture:8s} {n_gt:>4d}  {locked_rate:>7.1%}  "
              f"{rate_blind:>7.1%}  {rate_after:>7.1%}  "
              f"{_format_pp(delta_blind):>10s}  "
              f"{_format_pp(delta_locked):>11s}  {gate:>15s}")

        if a_after:
            after_total_n += c_after.get("n_gt_actions", 0)
            after_total_correct += c_after.get("correct", 0)
        if a_blind:
            blind_total_n += c_blind.get("n_gt_actions", 0)
            blind_total_correct += c_blind.get("correct", 0)
        locked_total_n += locked_n
        locked_total_correct += locked_correct
        if regressed:
            fixtures_regress_locked.append(r.fixture)

    blind_rate = blind_total_correct / blind_total_n if blind_total_n else 0.0
    after_rate = after_total_correct / after_total_n if after_total_n else 0.0
    locked_rate_agg = locked_total_correct / locked_total_n if locked_total_n else 0.0

    print("-" * 100)
    print(f"{'AGG':8s} {after_total_n:>4d}  {locked_rate_agg:>7.1%}  "
          f"{blind_rate:>7.1%}  {after_rate:>7.1%}  "
          f"{_format_pp(after_rate - blind_rate):>10s}  "
          f"{_format_pp(after_rate - locked_rate_agg):>11s}")

    print("\n" + "=" * 100)
    print("RELABEL LIFT — Surface B (Day-4 click-GT) per fixture")
    print("=" * 100)
    print(f"{'fixture':8s} {'n_eval':>6s}  {'BLIND':>8s}  {'AFTER':>8s}  "
          f"{'Δ vs blind':>10s}  {'gate':>10s}")

    sb_total_n = sb_total_correct_after = sb_total_correct_blind = 0
    sb_fixtures_at_100 = 0
    sb_fixtures_below_75 = 0
    for r in results:
        if r.after_surface_b is None and r.blind_surface_b is None:
            continue
        b_after = r.after_surface_b or {}
        b_blind = r.blind_surface_b or {}
        n = b_after.get("n_eval") or b_blind.get("n_eval", 0)
        rate_after = b_after.get("acc_full", 0.0)
        rate_blind = b_blind.get("acc_full", 0.0)
        delta = rate_after - rate_blind

        gate_parts = []
        if rate_after >= 0.999 and n > 0:
            sb_fixtures_at_100 += 1
            gate_parts.append("100%")
        if rate_after < 0.75 and n > 0:
            sb_fixtures_below_75 += 1
            gate_parts.append("<75%")
        gate = "/".join(gate_parts) if gate_parts else "OK"

        print(f"{r.fixture:8s} {n:>6d}  {rate_blind:>7.1%}  {rate_after:>7.1%}  "
              f"{_format_pp(delta):>10s}  {gate:>10s}")

        if b_after:
            sb_total_n += b_after.get("n_eval", 0)
            sb_total_correct_after += b_after.get("correct", 0)
        if b_blind:
            sb_total_correct_blind += b_blind.get("correct", 0)

    sb_after_rate = sb_total_correct_after / sb_total_n if sb_total_n else 0.0
    sb_blind_rate = sb_total_correct_blind / sb_total_n if sb_total_n else 0.0
    print("-" * 100)
    print(f"{'AGG':8s} {sb_total_n:>6d}  {sb_blind_rate:>7.1%}  "
          f"{sb_after_rate:>7.1%}  "
          f"{_format_pp(sb_after_rate - sb_blind_rate):>10s}")

    # Gate verdicts.
    print("\n" + "=" * 100)
    print("PLAN §6 GATES")
    print("=" * 100)
    surface_a_lift_pp = (after_rate - locked_rate_agg) * 100
    surface_a_pass = after_rate >= 0.58 and not fixtures_regress_locked
    print(f"Surface A — aggregate AFTER {after_rate:.1%} "
          f"(target ≥ 58%, vs locked {locked_rate_agg:.1%}): "
          f"{'PASS' if after_rate >= 0.58 else 'FAIL'}")
    print(f"Surface A — Δ vs locked {_format_pp(after_rate - locked_rate_agg)} "
          f"(target ≥ +15pp): "
          f"{'PASS' if surface_a_lift_pp >= 15.0 else 'FAIL'}")
    print(f"Surface A — fixtures regressing > 1pp vs locked: "
          f"{fixtures_regress_locked or 'none'} "
          f"({'PASS' if not fixtures_regress_locked else 'FAIL'})")
    for fx, lift in worst_two_lift.items():
        verdict = "PASS" if lift >= 0.05 else "FAIL"
        print(f"Surface A — {fx} lift over BLIND "
              f"{_format_pp(lift)} (target ≥ +5pp): {verdict}")

    surface_b_pass = (
        sb_after_rate >= 0.90
        and sb_fixtures_at_100 >= 3
        and sb_fixtures_below_75 == 0
    )
    print(f"Surface B — aggregate {sb_after_rate:.1%} (target ≥ 90%): "
          f"{'PASS' if sb_after_rate >= 0.90 else 'FAIL'}")
    print(f"Surface B — fixtures at 100%: {sb_fixtures_at_100} (target ≥ 3): "
          f"{'PASS' if sb_fixtures_at_100 >= 3 else 'FAIL'}")
    print(f"Surface B — fixtures below 75%: {sb_fixtures_below_75} (target = 0): "
          f"{'PASS' if sb_fixtures_below_75 == 0 else 'FAIL'}")

    print(f"\nOverall: Surface A {'PASS' if surface_a_pass else 'FAIL'}, "
          f"Surface B {'PASS' if surface_b_pass else 'FAIL'}")

    return {
        "surface_a": {
            "blind_aggregate": {"n": blind_total_n, "correct": blind_total_correct,
                                "rate": blind_rate},
            "after_aggregate": {"n": after_total_n, "correct": after_total_correct,
                                "rate": after_rate},
            "locked_aggregate": {"n": locked_total_n, "correct": locked_total_correct,
                                 "rate": locked_rate_agg},
            "fixtures_regress_locked": fixtures_regress_locked,
            "worst_two_lift": worst_two_lift,
            "pass": surface_a_pass and surface_a_lift_pp >= 15.0,
        },
        "surface_b": {
            "blind_aggregate": {"n": sb_total_n,
                                "correct": sb_total_correct_blind,
                                "rate": sb_blind_rate},
            "after_aggregate": {"n": sb_total_n,
                                "correct": sb_total_correct_after,
                                "rate": sb_after_rate},
            "fixtures_at_100": sb_fixtures_at_100,
            "fixtures_below_75": sb_fixtures_below_75,
            "pass": surface_b_pass,
        },
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--fixtures", nargs="*", default=None,
        help="Fixture names (default: all 9 baseline fixtures).",
    )
    ap.add_argument(
        "--skip-blind", action="store_true",
        help="Skip blind match-players step (only run AFTER side; for debugging).",
    )
    ap.add_argument(
        "--out-dir", type=Path, default=None,
        help="Per-run output directory. Defaults to "
             "reports/attribution_rebuild/relabel_lift/<timestamp>/.",
    )
    args = ap.parse_args()

    fixture_map = json.loads(FIXTURE_REGISTRY.read_text())["fixtures"]
    if args.fixtures:
        unknown = [f for f in args.fixtures if f not in fixture_map]
        if unknown:
            print(f"Unknown fixture(s): {unknown}", file=sys.stderr)
            print(f"Known: {sorted(fixture_map)}", file=sys.stderr)
            return 2
        targets = [(f, fixture_map[f]["video_id"]) for f in args.fixtures]
    else:
        targets = [(f, fixture_map[f]["video_id"]) for f in fixture_map]

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out_dir or (
        ATTR_REPORTS / "relabel_lift" / ts
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")
    print(f"Targets: {[t[0] for t in targets]}")

    results: list[FixtureResult] = []
    for i, (fx, vid) in enumerate(targets, 1):
        print(f"\n[{i}/{len(targets)}] running {fx}...")
        res = run_fixture(fx, vid, out_dir, skip_blind=args.skip_blind)
        results.append(res)

    summary = render_report(results)

    # Persist results (results dataclass needs serialization).
    results_json = []
    for r in results:
        results_json.append({
            "fixture": r.fixture,
            "video_id": r.video_id,
            "error": r.error,
            "n_ref_crops": r.n_ref_crops,
            "blind_surface_a": r.blind_surface_a,
            "after_surface_a": r.after_surface_a,
            "blind_surface_b": r.blind_surface_b,
            "after_surface_b": r.after_surface_b,
            "timings": r.timings,
        })
    (out_dir / "results.json").write_text(
        json.dumps({
            "generated_at": ts,
            "fixtures": results_json,
            "summary": summary,
        }, indent=2, default=str)
    )
    print(f"\nwrote {out_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
