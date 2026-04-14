"""Step 1 diagnostic — pose-attributor confidence-floor lever check.

Read-only. Session 4 follow-up (see memory/fn_sequence_signal_2026_04.md):
the sequence rescue drops player_attribution + court_side because pose
attributor predictions on rescued (low-GBM-confidence) contacts are rejected
by the fixed floor `cfg.pose_attribution_min_confidence = 0.5` and fall back
to nearest-player, which is wrong more often than the pose model's pick.

This script measures whether relaxing the floor for rescued contacts is a
real lever. It monkey-patches `pose_attribution.inference.PoseAttributionInference.predict`
and `pose_attribution.features.extract_candidate_features` to capture every
per-candidate pose prediction during a rescue-on run, then pairs the log
with emitted pred contacts by rally+frame.

For each rescued contact (present rescue-on, absent rescue-off) the script
records: gbm_conf (from pred dict), chosen_tid (what shipped), pose_pred_tid,
pose_pred_conf, gt_player_tid (via match_contacts, Session-11 normalized).

Output tables (only rescued contacts):
  1. overall chosen-correct vs pose-correct
  2. pose_conf bin table — chosen_ok%, pose_ok%, lever Δ, flip_rate
  3. (gbm_conf × pose_conf) cross-tab restricted to pose_conf<0.5
     (the band where pose is currently rejected and chosen == pre-override).

Decision gate: a contiguous pose_conf band below 0.5 with n ≥ 40 and
lever Δ (pose_ok − chosen_ok) ≥ +10pp → GO.

Usage:
    cd analysis
    uv run python scripts/diagnose_pose_floor_by_confidence.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import production_eval  # noqa: E402
from eval_action_detection import (  # noqa: E402
    _load_match_team_assignments,
    _load_track_to_player_maps,
    load_rallies_with_action_gt,
    match_contacts,
)

from rallycut.tracking.player_tracker import PlayerPosition  # noqa: E402
from rallycut.tracking.pose_attribution import features as pa_features  # noqa: E402
from rallycut.tracking.pose_attribution import inference as pa_inference  # noqa: E402

console = Console()


# --------------------------------------------------------------------------- #
# Monkey-patch capture                                                        #
# --------------------------------------------------------------------------- #

_last_frame: dict[str, int] = {"frame": -1}
_pose_log: list[dict[str, Any]] = []

_orig_extract = pa_features.extract_candidate_features
_orig_predict = pa_inference.PoseAttributionInference.predict


def _patched_extract(*args: Any, **kwargs: Any) -> Any:
    frame = kwargs.get("contact_frame")
    if frame is None and args:
        # positional: contact_frame is first arg in current signature
        frame = args[0]
    _last_frame["frame"] = int(frame) if frame is not None else -1
    return _orig_extract(*args, **kwargs)


def _patched_predict(
    self: Any,
    candidate_features: list[Any],
    candidate_track_ids: list[int],
) -> tuple[int, float]:
    pred_tid, pred_conf = _orig_predict(self, candidate_features, candidate_track_ids)
    _pose_log.append({
        "frame": _last_frame["frame"],
        "cand_tids": list(candidate_track_ids),
        "pred_tid": int(pred_tid),
        "pred_conf": float(pred_conf),
    })
    return pred_tid, pred_conf


def _install_patches() -> None:
    pa_features.extract_candidate_features = _patched_extract
    pa_inference.PoseAttributionInference.predict = _patched_predict  # type: ignore[method-assign]


# --------------------------------------------------------------------------- #
# Binning                                                                     #
# --------------------------------------------------------------------------- #

POSE_BINS: list[tuple[float, float]] = [
    (0.00, 0.30),
    (0.30, 0.40),
    (0.40, 0.50),
    (0.50, 0.70),
    (0.70, 1.01),
]
GBM_BINS: list[tuple[float, float]] = [
    (0.20, 0.30),
    (0.30, 0.40),
    (0.40, 0.50),
    (0.50, 1.01),
]


# --------------------------------------------------------------------------- #
# Per-rally runner                                                            #
# --------------------------------------------------------------------------- #


def _run_and_capture(
    rallies: list[Any],
    team_map: dict[str, dict[int, int]],
    calibrators: dict[str, Any],
    ctx: production_eval.PipelineContext,
    label: str,
) -> dict[str, tuple[list[dict], list[dict]]]:
    """Return {rally_id: (real_pred_dicts, pose_log_entries)}."""
    out: dict[str, tuple[list[dict], list[dict]]] = {}
    for i, rally in enumerate(rallies):
        if (
            not rally.ball_positions_json
            or not rally.positions_json
            or not rally.frame_count
            or rally.frame_count < 10
        ):
            continue
        _pose_log.clear()
        _last_frame["frame"] = -1
        try:
            pred_actions, _ = production_eval._run_rally(
                rally,
                team_map.get(rally.rally_id),
                calibrators.get(rally.video_id),
                ctx,
            )
        except Exception as exc:  # noqa: BLE001
            console.print(f"  [red]{rally.rally_id[:8]} {label}: {exc}[/red]")
            continue
        real_pred = [a for a in pred_actions if not a.get("isSynthetic")]
        out[rally.rally_id] = (real_pred, list(_pose_log))
        if (i + 1) % 50 == 0:
            console.print(f"  [{label}] {i + 1}/{len(rallies)}")
    return out


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


def main() -> None:
    _install_patches()

    console.print("[bold]Loading rallies + team map + calibrators + t2p maps...[/bold]")
    rallies = load_rallies_with_action_gt()
    console.print(f"  {len(rallies)} rallies")

    video_ids = {r.video_id for r in rallies if r.video_id}

    rally_pos_lookup: dict[str, list[PlayerPosition]] = {}
    for r in rallies:
        if r.positions_json:
            rally_pos_lookup[r.rally_id] = [
                PlayerPosition(
                    frame_number=pp["frameNumber"],
                    track_id=pp["trackId"],
                    x=pp["x"],
                    y=pp["y"],
                    width=pp["width"],
                    height=pp["height"],
                    confidence=pp.get("confidence", 1.0),
                    keypoints=pp.get("keypoints"),
                )
                for pp in r.positions_json
            ]
    team_map = _load_match_team_assignments(
        video_ids, rally_positions=rally_pos_lookup
    )
    calibrators = production_eval._build_calibrators(video_ids)
    t2p_by_rally = _load_track_to_player_maps(video_ids) or {}

    # --- Run 1: rescue off (identify rescued frames) --- #
    console.print("\n[bold]Run 1: rescue off[/bold]")
    off_result = _run_and_capture(
        rallies,
        team_map,
        calibrators,
        production_eval.PipelineContext(skip_sequence_recovery=True),
        "off",
    )
    off_frame_sets: dict[str, set[int]] = {}
    for rid, (preds, _log) in off_result.items():
        off_frame_sets[rid] = {
            p.get("frame") for p in preds if p.get("frame") is not None
        }
    console.print(f"  captured {len(off_result)} rallies")

    # --- Run 2: rescue on (capture pose log) --- #
    console.print("\n[bold]Run 2: rescue on (pose log capture)[/bold]")
    on_result = _run_and_capture(
        rallies,
        team_map,
        calibrators,
        production_eval.PipelineContext(),
        "on",
    )
    console.print(f"  captured {len(on_result)} rallies")

    # --- Join with GT, build per-contact records --- #
    rally_by_id = {r.rally_id: r for r in rallies}
    records: list[dict[str, Any]] = []
    for rid, (real_pred, pose_log) in on_result.items():
        rally = rally_by_id.get(rid)
        if rally is None:
            continue
        off_frames = off_frame_sets.get(rid, set())
        rally_t2p = t2p_by_rally.get(rid) or None
        match_teams = team_map.get(rid)
        tol = max(1, round(rally.fps * 167 / 1000))
        matches, _unmatched = match_contacts(
            rally.gt_labels,
            real_pred,
            tolerance=tol,
            team_assignments=match_teams,
            track_id_map=rally_t2p,
        )
        gt_sorted = sorted(rally.gt_labels, key=lambda g: g.frame)
        frame_to_gt: dict[int, tuple[int, str]] = {}
        for gi, m in enumerate(matches):
            if m.pred_frame is None:
                continue
            if gi >= len(gt_sorted):
                continue
            gt = gt_sorted[gi]
            frame_to_gt[m.pred_frame] = (gt.player_track_id, gt.action)

        pose_by_frame: dict[int, dict] = {}
        for entry in pose_log:
            # Last-write-wins; same-frame collisions are rare.
            pose_by_frame[entry["frame"]] = entry

        def _norm(tid: int) -> int:
            if rally_t2p and tid >= 0:
                return rally_t2p.get(tid, tid)
            return tid

        for p in real_pred:
            f = p.get("frame")
            if f is None:
                continue
            gt_entry = frame_to_gt.get(f)
            if gt_entry is None:
                continue  # unmatched — can't evaluate correctness
            gt_tid, gt_action = gt_entry
            if gt_tid < 0:
                continue  # GT lacks player_track_id — not evaluable
            rescued = f not in off_frames
            chosen_tid = int(p.get("playerTrackId", -1))
            pose_entry = pose_by_frame.get(f)
            pose_pred_tid = pose_entry["pred_tid"] if pose_entry else -1
            pose_pred_conf = pose_entry["pred_conf"] if pose_entry else -1.0
            records.append({
                "rally_id": rid,
                "frame": f,
                "rescued": rescued,
                "gbm_conf": float(p.get("confidence", 0.0)),
                "pose_conf": float(pose_pred_conf),
                "chosen_tid_n": _norm(chosen_tid),
                "pose_pred_tid_n": _norm(pose_pred_tid),
                "gt_tid": gt_tid,
                "action": p.get("action") or "?",
                "has_pose_call": pose_entry is not None,
            })

    # --- Summary --- #
    rescued = [r for r in records if r["rescued"]]
    rescued_with_pose = [r for r in rescued if r["has_pose_call"]]
    console.print(
        f"\n[bold]Records: total={len(records)}  "
        f"rescued={len(rescued)}  rescued_with_pose_call={len(rescued_with_pose)}[/bold]"
    )

    if not rescued_with_pose:
        console.print("[red]No rescued contacts with pose calls — aborting analysis.[/red]")
        return

    def _acc(sel: list[dict], key: str) -> float:
        if not sel:
            return 0.0
        return 100.0 * sum(1 for r in sel if r[key] == r["gt_tid"]) / len(sel)

    overall_chosen = _acc(rescued, "chosen_tid_n")
    overall_pose = _acc(rescued_with_pose, "pose_pred_tid_n")
    console.print(
        f"\n  chosen correct (all rescued):             "
        f"{overall_chosen:.1f}%  (n={len(rescued)})"
    )
    console.print(
        f"  pose correct (rescued with pose call):    "
        f"{overall_pose:.1f}%  (n={len(rescued_with_pose)})"
    )

    # --- Pose-conf bin table --- #
    tbl = Table(title="Rescued contacts — pose_pred vs chosen, by pose_conf bin")
    tbl.add_column("pose_conf bin")
    tbl.add_column("n", justify="right")
    tbl.add_column("chosen_ok%", justify="right")
    tbl.add_column("pose_ok%", justify="right")
    tbl.add_column("Δ (pose−chosen) pp", justify="right")
    tbl.add_column("flip_rate%", justify="right")

    for lo, hi in POSE_BINS:
        sel = [r for r in rescued_with_pose if lo <= r["pose_conf"] < hi]
        n = len(sel)
        if n == 0:
            tbl.add_row(f"[{lo:.2f},{hi:.2f})", "0", "-", "-", "-", "-")
            continue
        ch_ok = _acc(sel, "chosen_tid_n")
        po_ok = _acc(sel, "pose_pred_tid_n")
        flip = 100.0 * sum(
            1 for r in sel if r["pose_pred_tid_n"] != r["chosen_tid_n"]
        ) / n
        tbl.add_row(
            f"[{lo:.2f},{hi:.2f})",
            str(n),
            f"{ch_ok:.1f}",
            f"{po_ok:.1f}",
            f"{po_ok - ch_ok:+.1f}",
            f"{flip:.1f}",
        )
    console.print(tbl)

    # --- Below-floor aggregate --- #
    below = [r for r in rescued_with_pose if r["pose_conf"] < 0.5]
    console.print(
        f"\n[bold]Rescued + pose_conf<0.5 (pose currently rejected): "
        f"n={len(below)}[/bold]"
    )
    if below:
        ch_ok = _acc(below, "chosen_tid_n")
        po_ok = _acc(below, "pose_pred_tid_n")
        console.print(f"  chosen (current ship):  {ch_ok:.1f}%")
        console.print(f"  pose_pred (floor→0):    {po_ok:.1f}%")
        console.print(f"  lever Δ:                {po_ok - ch_ok:+.1f}pp")

    # --- (gbm_conf × pose_conf) cross-tab, below-floor only --- #
    sub_pose_bins = [b for b in POSE_BINS if b[1] <= 0.5]
    tbl2 = Table(
        title="Lever Δ (pose_ok − chosen_ok, pp) by gbm_conf × pose_conf — rescued, pose_conf<0.5"
    )
    tbl2.add_column("gbm_conf")
    for lo, hi in sub_pose_bins:
        tbl2.add_column(f"[{lo:.2f},{hi:.2f})  n / Δpp", justify="right")
    for glo, ghi in GBM_BINS:
        row = [f"[{glo:.2f},{ghi:.2f})"]
        for plo, phi in sub_pose_bins:
            sel = [
                r for r in rescued_with_pose
                if glo <= r["gbm_conf"] < ghi and plo <= r["pose_conf"] < phi
            ]
            if not sel:
                row.append("0")
                continue
            ch_ok = _acc(sel, "chosen_tid_n")
            po_ok = _acc(sel, "pose_pred_tid_n")
            row.append(f"{len(sel)} / {po_ok - ch_ok:+.1f}")
        tbl2.add_row(*row)
    console.print(tbl2)

    # --- Decision gate --- #
    console.print("\n[bold]Decision gate[/bold]")
    candidate_bands: list[tuple[float, float, int, float]] = []
    for lo, hi in sub_pose_bins:
        sel = [r for r in rescued_with_pose if lo <= r["pose_conf"] < hi]
        if len(sel) < 40:
            continue
        ch_ok = _acc(sel, "chosen_tid_n")
        po_ok = _acc(sel, "pose_pred_tid_n")
        delta = po_ok - ch_ok
        if delta >= 10.0:
            candidate_bands.append((lo, hi, len(sel), delta))
    if candidate_bands:
        console.print("[green]GO — candidate band(s) meet n≥40 and Δ≥+10pp:[/green]")
        for lo, hi, n, d in candidate_bands:
            console.print(f"  pose_conf [{lo:.2f},{hi:.2f})  n={n}  Δ={d:+.1f}pp")
    else:
        console.print(
            "[yellow]NO-GO — no sub-floor band meets n≥40 and Δ≥+10pp. "
            "Write the NO-GO memo and close.[/yellow]"
        )


if __name__ == "__main__":
    main()
