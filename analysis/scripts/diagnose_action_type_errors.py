"""Diagnose where action-type classification errors originate.

For every contact-level action-type error (predicted action_type != GT,
both in {dig, set, attack}) on the canonical GT pool, bucket the error
by origin:

    A  pre-override = GT, post-override wrong              -> override regression
    B  pre-override wrong, post-override same wrong        -> both agree wrong
    C  pre-override wrong, post-override different wrong   -> override wrong->wrong
    D  pre-override wrong, override_argmax = GT blocked    -> guard rejected fix
    E  sequence_probs None                                 -> override didn't run

Captures per-contact:
    * action confidence at pre-override action (GBM top-1 proxy).
    * MS-TCN++ non-bg probability vector at the contact frame.
    * MS-TCN++ raw argmax (serve-excluded) at the contact frame.
    * Whether the dig-guard (DIG_GUARD_RATIO=2.5) rejected an override.
    * Pose availability at the contact frame (n poses + mean conf).
    * Session id of the rally's video (so the "poor" session can be
      separated from the clean pool).

No production code is modified. No training runs. Read-only.

Usage:
    cd analysis
    uv run python scripts/diagnose_action_type_errors.py                   # all 339
    uv run python scripts/diagnose_action_type_errors.py --rally <id>      # single rally
    uv run python scripts/diagnose_action_type_errors.py \\
        --skip-session 6f599a0e-b8ea-4bf0-a331-ce7d9ef88164                # exclude poor
    uv run python scripts/diagnose_action_type_errors.py --output out.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from eval_action_detection import (  # noqa: E402
    RallyData,
    _build_player_positions,
    _load_match_team_assignments,
    _load_track_to_player_maps,
    load_rallies_with_action_gt,
    match_contacts,
)
from production_eval import (  # noqa: E402
    _build_calibrators,
    _build_camera_heights,
    _load_formation_semantic_flips_from_gt,
    _parse_ball,
)

from rallycut.actions.trajectory_features import ACTION_TYPES  # noqa: E402
from rallycut.evaluation.tracking.db import get_connection  # noqa: E402
from rallycut.tracking.action_classifier import (  # noqa: E402
    ActionType,
    classify_rally_actions,
)
from rallycut.tracking.contact_detector import detect_contacts  # noqa: E402
from rallycut.tracking.match_tracker import verify_team_assignments  # noqa: E402
from rallycut.tracking.player_tracker import PlayerPosition  # noqa: E402
from rallycut.tracking.pose_attribution.pose_cache import load_pose_cache  # noqa: E402
from rallycut.tracking.sequence_action_runtime import (  # noqa: E402
    DIG_GUARD_RATIO,
    get_sequence_probs,
)

console = Console()

TARGET_CLASSES = {"dig", "set", "attack"}
POOR_SESSION_ID = "6f599a0e-b8ea-4bf0-a331-ce7d9ef88164"


# --------------------------------------------------------------------------- #
# Data containers                                                             #
# --------------------------------------------------------------------------- #


@dataclass
class ContactProbe:
    """Per-contact record for a GT-matched action in the diagnostic set."""

    rally_id: str
    video_id: str
    session_id: str | None
    contact_frame: int
    gt_frame: int
    gt_action: str
    pre_override_type: str        # ActionType.value after classify_rally_actions
    pre_override_confidence: float
    post_override_type: str       # ActionType.value after apply_sequence_override
    override_argmax: str          # MS-TCN++ raw argmax (serve-excluded), or "" if N/A
    override_would_apply: bool    # False if override skipped (synthetic, serve, OOB)
    guard_fired: bool             # dig->set guard refused the override
    seq_present: bool             # MS-TCN++ probs were available
    seq_probs_nonbg: list[float]  # per-class probs at contact frame [serve..block]
    seq_peak_prob: float          # max non-bg prob at contact frame
    pose_present: bool            # any player keypoints at this frame
    pose_count: int               # number of players with pose entries at frame
    pose_mean_conf: float         # mean keypoint confidence across detected poses

    @property
    def is_error(self) -> bool:
        """True iff post_override != GT and both in {dig, set, attack}."""
        return (
            self.gt_action in TARGET_CLASSES
            and self.post_override_type in TARGET_CLASSES
            and self.post_override_type != self.gt_action
        )

    @property
    def bucket(self) -> str:
        """Return error bucket A/B/C/D/E/'correct'/'out-of-scope'."""
        if self.gt_action not in TARGET_CLASSES:
            return "out-of-scope"
        if self.post_override_type == self.gt_action:
            return "correct"
        if self.post_override_type not in TARGET_CLASSES:
            return "out-of-scope"
        if not self.seq_present:
            return "E"
        if self.pre_override_type == self.gt_action:
            # override took a right answer and made it wrong
            return "A"
        # pre_override != GT, post_override != GT (both wrong)
        if (
            self.pre_override_type == self.post_override_type
            and self.guard_fired
            and self.override_argmax == self.gt_action
        ):
            # guard kept pre wrong; override would have flipped to GT
            return "D"
        if self.pre_override_type == self.post_override_type:
            # override didn't change it (either no-op, guard, or argmax==pre)
            return "B"
        # pre != post, both wrong, different classes
        return "C"


# --------------------------------------------------------------------------- #
# Session lookup (videos -> session_id)                                       #
# --------------------------------------------------------------------------- #


def _load_video_sessions(video_ids: set[str]) -> dict[str, str]:
    """Map video_id -> session_id via the session_videos table."""
    if not video_ids:
        return {}
    placeholders = ", ".join(["%s"] * len(video_ids))
    query = f"""
        SELECT video_id, session_id
        FROM session_videos
        WHERE video_id IN ({placeholders})
    """
    out: dict[str, str] = {}
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(query, list(video_ids))
        for vid, sid in cur.fetchall():
            out[str(vid)] = str(sid)
    return out


# --------------------------------------------------------------------------- #
# Pose availability                                                           #
# --------------------------------------------------------------------------- #


def _pose_at_frame(
    pose_cache: dict[str, Any] | None,
    frame: int,
) -> tuple[bool, int, float]:
    """Return (present, count, mean_keypoint_confidence) for pose at `frame`."""
    if pose_cache is None or len(pose_cache.get("frames", [])) == 0:
        return (False, 0, 0.0)
    frames = pose_cache["frames"]
    keypoints = pose_cache["keypoints"]
    mask = frames == frame
    n = int(mask.sum())
    if n == 0:
        return (False, 0, 0.0)
    # Mean confidence across the nth column of COCO keypoints (index 2).
    confs = keypoints[mask, :, 2]
    return (True, n, float(confs.mean()))


# --------------------------------------------------------------------------- #
# Per-rally probe                                                             #
# --------------------------------------------------------------------------- #


def _probe_rally(
    rally: RallyData,
    match_teams: dict[int, int] | None,
    calibrator: Any,
    track_to_player: dict[int, int] | None,
    formation_semantic_flip: bool,
    camera_height: float,
    video_session: str | None,
    tolerance_frames: int,
) -> list[ContactProbe]:
    """Mirror production_eval `_run_rally` stages 9-14, instrumented.

    Instead of calling ``apply_sequence_override``, we reimplement the exact
    per-action decision it makes so we can capture the argmax, the guard
    status, and the probability vector at the contact frame. Final
    ``action.action_type`` is set exactly as the production call would set
    it, so post-override predictions match production bit-for-bit.
    """
    ball_positions = _parse_ball(rally.ball_positions_json or [])
    player_positions: list[PlayerPosition] = _build_player_positions(
        rally.positions_json or [], rally_id=rally.rally_id, inject_pose=True,
    )
    if not ball_positions or not rally.frame_count:
        return []

    teams: dict[int, int] | None = dict(match_teams) if match_teams else None
    if teams is not None:
        teams = verify_team_assignments(teams, player_positions)

    sequence_probs: np.ndarray | None = get_sequence_probs(
        ball_positions=ball_positions,
        player_positions=player_positions,
        court_split_y=rally.court_split_y,
        frame_count=rally.frame_count,
        team_assignments=teams,
        calibrator=calibrator,
    )

    contact_sequence = detect_contacts(
        ball_positions=ball_positions,
        player_positions=player_positions,
        frame_count=rally.frame_count or None,
        team_assignments=teams,
        court_calibrator=calibrator,
        sequence_probs=sequence_probs,
    )

    rally_actions = classify_rally_actions(
        contact_sequence,
        rally.rally_id,
        team_assignments=teams,
        match_team_assignments=teams,
        calibrator=calibrator,
        track_to_player=track_to_player,
        formation_semantic_flip=formation_semantic_flip,
        camera_height=camera_height,
    )

    # Snapshot pre-override state: action_type + confidence indexed by frame.
    pre_types: dict[int, str] = {
        a.frame: a.action_type.value for a in rally_actions.actions
    }
    pre_confs: dict[int, float] = {
        a.frame: float(a.confidence) for a in rally_actions.actions
    }

    # Override state to capture per frame (only populated when override runs).
    override_info: dict[int, dict[str, Any]] = {}

    if sequence_probs is not None:
        set_idx = ACTION_TYPES.index("set")
        dig_idx = ACTION_TYPES.index("dig")

        for action in rally_actions.actions:
            if action.is_synthetic or action.action_type == ActionType.SERVE:
                continue
            frame = action.frame
            if not (0 <= frame < sequence_probs.shape[1]):
                continue
            per_frame = sequence_probs[1:, frame]  # length = len(ACTION_TYPES)
            cls = int(np.argmax(per_frame))
            new_type = ActionType(ACTION_TYPES[cls])

            argmax_label = ACTION_TYPES[cls]
            guard_fired = False
            would_apply = True

            if new_type == ActionType.SERVE:
                # Production: override never manufactures serves.
                would_apply = False

            if (
                would_apply
                and action.action_type == ActionType.DIG
                and new_type == ActionType.SET
            ):
                seq_set = float(per_frame[set_idx])
                seq_dig = float(per_frame[dig_idx])
                if seq_set < DIG_GUARD_RATIO * seq_dig:
                    guard_fired = True
                    would_apply = False

            override_info[frame] = {
                "argmax": argmax_label,
                "would_apply": would_apply,
                "guard_fired": guard_fired,
                "probs": [float(p) for p in per_frame],
            }

            if would_apply:
                action.action_type = new_type

    # Match GT to the final (post-override) predicted actions.
    pred_actions = [a.to_dict() for a in rally_actions.actions]
    matches, _ = match_contacts(
        rally.gt_labels, pred_actions, tolerance=tolerance_frames,
    )

    # Build a frame -> pred dict lookup for contact_frame and pose frame.
    pred_by_frame: dict[int, dict[str, Any]] = {
        p["frame"]: p for p in pred_actions
    }

    # Load pose cache once per rally (we query per frame).
    pose_cache = load_pose_cache(rally.rally_id)

    probes: list[ContactProbe] = []
    for m in matches:
        if m.pred_frame is None or m.pred_action is None:
            continue  # FN — not an action-type error
        pred = pred_by_frame.get(m.pred_frame)
        if pred is None:
            continue
        post_type = pred["action"]
        pre_type = pre_types.get(m.pred_frame, post_type)
        pre_conf = pre_confs.get(m.pred_frame, 0.0)
        oi = override_info.get(m.pred_frame, {})
        pose_present, pose_count, pose_conf = _pose_at_frame(pose_cache, m.pred_frame)

        if oi:
            probs_list = oi["probs"]
            seq_peak = float(max(probs_list)) if probs_list else 0.0
        else:
            probs_list = []
            seq_peak = 0.0

        probes.append(ContactProbe(
            rally_id=rally.rally_id,
            video_id=rally.video_id,
            session_id=video_session,
            contact_frame=m.pred_frame,
            gt_frame=m.gt_frame,
            gt_action=m.gt_action,
            pre_override_type=pre_type,
            pre_override_confidence=pre_conf,
            post_override_type=post_type,
            override_argmax=oi.get("argmax", ""),
            override_would_apply=bool(oi.get("would_apply", False)),
            guard_fired=bool(oi.get("guard_fired", False)),
            seq_present=sequence_probs is not None,
            seq_probs_nonbg=probs_list,
            seq_peak_prob=seq_peak,
            pose_present=pose_present,
            pose_count=pose_count,
            pose_mean_conf=pose_conf,
        ))

    return probes


# --------------------------------------------------------------------------- #
# Reporting                                                                   #
# --------------------------------------------------------------------------- #


def _bucket_counts(probes: list[ContactProbe]) -> dict[str, int]:
    c: Counter[str] = Counter()
    for p in probes:
        c[p.bucket] += 1
    return dict(c)


def _confusion_3x3(
    probes: list[ContactProbe],
) -> dict[str, dict[str, int]]:
    mat: dict[str, dict[str, int]] = {
        g: {p: 0 for p in TARGET_CLASSES} for g in TARGET_CLASSES
    }
    for p in probes:
        if p.gt_action in TARGET_CLASSES and p.post_override_type in TARGET_CLASSES:
            mat[p.gt_action][p.post_override_type] += 1
    return mat


def _summarize_bucket(
    probes: list[ContactProbe],
    bucket: str,
) -> dict[str, Any]:
    sub = [p for p in probes if p.bucket == bucket]
    if not sub:
        return {"n": 0}

    # Confusion pairs (gt -> post).
    pairs: Counter[tuple[str, str]] = Counter()
    for p in sub:
        pairs[(p.gt_action, p.post_override_type)] += 1

    # pre_override confidence distribution.
    confs = [p.pre_override_confidence for p in sub]
    # MS-TCN++ peak distribution.
    peaks = [p.seq_peak_prob for p in sub if p.seq_present]
    # Pose availability.
    pose_rate = sum(1 for p in sub if p.pose_present) / len(sub)
    # Sessions.
    sessions: Counter[str] = Counter()
    for p in sub:
        sessions[p.session_id or "unknown"] += 1

    # Top-5 rally_ids by count in this bucket.
    rally_counts: Counter[str] = Counter()
    for p in sub:
        rally_counts[p.rally_id] += 1

    return {
        "n": len(sub),
        "pairs": dict(pairs),
        "conf_mean": float(np.mean(confs)) if confs else 0.0,
        "conf_median": float(np.median(confs)) if confs else 0.0,
        "peak_mean": float(np.mean(peaks)) if peaks else 0.0,
        "peak_median": float(np.median(peaks)) if peaks else 0.0,
        "pose_rate": float(pose_rate),
        "sessions": dict(sessions),
        "top_rallies": rally_counts.most_common(5),
    }


def _print_console_report(probes: list[ContactProbe], *, title: str) -> None:
    total = len(probes)
    errors = [p for p in probes if p.is_error]
    correct = [p for p in probes if p.bucket == "correct"]

    console.print(f"\n[bold]{title}[/bold]")
    console.print(
        f"  matched contacts: {total}   action-type errors: {len(errors)}   "
        f"correct {{dig,set,attack}}: {len(correct)}"
    )

    tbl = Table(title="Error bucket counts")
    tbl.add_column("Bucket")
    tbl.add_column("Count", justify="right")
    tbl.add_column("Meaning")
    meaning = {
        "A": "GBM right, override flipped wrong",
        "B": "GBM wrong, override kept wrong (same class)",
        "C": "GBM wrong, override wrong different class",
        "D": "GBM wrong, guard blocked correct override",
        "E": "override didn't run (sequence_probs None)",
    }
    for b in ["A", "B", "C", "D", "E"]:
        n = sum(1 for p in probes if p.bucket == b)
        tbl.add_row(b, str(n), meaning[b])
    console.print(tbl)

    # Per-bucket confusion pairs.
    for b in ["A", "B", "C", "D", "E"]:
        summary = _summarize_bucket(probes, b)
        if summary["n"] == 0:
            continue
        sub_tbl = Table(title=f"Bucket {b} (n={summary['n']}) — confusion pairs")
        sub_tbl.add_column("gt -> post", justify="left")
        sub_tbl.add_column("count", justify="right")
        for (g, p), c in sorted(
            summary["pairs"].items(), key=lambda x: -x[1]
        ):
            sub_tbl.add_row(f"{g} -> {p}", str(c))
        console.print(sub_tbl)


def _write_markdown_report(
    probes_all: list[ContactProbe],
    probes_clean: list[ContactProbe],
    output_path: Path,
    n_rallies_loaded: int,
    n_rallies_evaluated: int,
    n_rallies_clean: int,
) -> None:
    """Write the per-plan markdown deliverable."""

    def fmt_pairs(pairs: dict[tuple[str, str], int]) -> str:
        return ", ".join(
            f"{g}->{p}:{c}"
            for (g, p), c in sorted(pairs.items(), key=lambda x: -x[1])
        )

    bucket_meaning = {
        "A": "GBM right, override flipped to wrong (override regression)",
        "B": "GBM wrong, override kept it wrong (same class)",
        "C": "GBM wrong, override flipped to a different wrong class",
        "D": "GBM wrong, override would fix it, dig-guard blocked it",
        "E": "Override didn't run (sequence_probs None)",
    }

    lines: list[str] = []
    lines.append("# Action-Type Error Origin Diagnostic\n")
    lines.append(
        f"Run date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    )
    lines.append(
        f"Canonical pool: {n_rallies_loaded} rallies loaded, "
        f"{n_rallies_evaluated} evaluated, {n_rallies_clean} clean "
        f"(poor session `{POOR_SESSION_ID[:8]}...` excluded).\n"
    )
    lines.append("## Bucket summary\n")
    lines.append("| Bucket | Full pool | Clean pool | Meaning |")
    lines.append("|---|---:|---:|---|")
    full_counts = _bucket_counts(probes_all)
    clean_counts = _bucket_counts(probes_clean)
    for b in ["A", "B", "C", "D", "E"]:
        lines.append(
            f"| {b} | {full_counts.get(b, 0)} | {clean_counts.get(b, 0)} "
            f"| {bucket_meaning[b]} |"
        )
    total_err_full = sum(full_counts.get(b, 0) for b in "ABCDE")
    total_err_clean = sum(clean_counts.get(b, 0) for b in "ABCDE")
    lines.append(
        f"| **Total errors** | **{total_err_full}** | **{total_err_clean}** | |"
    )
    n_correct_full = full_counts.get("correct", 0)
    n_correct_clean = clean_counts.get("correct", 0)
    lines.append(
        f"| Correct {{dig,set,attack}} | {n_correct_full} | {n_correct_clean} | |"
    )
    lines.append("")

    # Overall confusion matrix (clean pool).
    lines.append("## Confusion matrix (clean pool, post-override)\n")
    lines.append("| gt \\\\ post | dig | set | attack |")
    lines.append("|---|---:|---:|---:|")
    cm = _confusion_3x3(probes_clean)
    for g in ["dig", "set", "attack"]:
        row = cm.get(g, {})
        lines.append(
            f"| **{g}** | {row.get('dig', 0)} | {row.get('set', 0)} | {row.get('attack', 0)} |"
        )
    lines.append("")

    # Per-bucket detail (clean pool).
    lines.append("## Per-bucket detail (clean pool)\n")
    for b in ["A", "B", "C", "D", "E"]:
        summary = _summarize_bucket(probes_clean, b)
        if summary["n"] == 0:
            continue
        lines.append(f"### Bucket {b} ({bucket_meaning[b]})\n")
        lines.append(f"- n = **{summary['n']}**")
        lines.append(f"- confusion pairs: {fmt_pairs(summary['pairs'])}")
        lines.append(
            f"- pre-override confidence: mean={summary['conf_mean']:.3f}, "
            f"median={summary['conf_median']:.3f}"
        )
        lines.append(
            f"- MS-TCN++ peak prob at contact: mean={summary['peak_mean']:.3f}, "
            f"median={summary['peak_median']:.3f}"
        )
        lines.append(
            f"- pose availability at contact: {summary['pose_rate']*100:.1f}%"
        )
        session_line = ", ".join(
            f"{sid[:8] if sid != 'unknown' else sid}: {n}"
            for sid, n in sorted(
                summary["sessions"].items(), key=lambda x: -x[1]
            )
        )
        lines.append(f"- sessions: {session_line}")
        rally_line = ", ".join(
            f"{rid[:8]} ({n})" for rid, n in summary["top_rallies"]
        )
        lines.append(f"- top rallies: {rally_line}")
        lines.append("")

    # Probability margin distribution across buckets vs correct.
    lines.append("## Probability margins (clean pool)\n")
    lines.append("Pre-override confidence and MS-TCN++ peak at the contact frame, ")
    lines.append("comparing correct predictions to each error bucket.\n")
    lines.append("| Group | n | pre_conf med | pre_conf mean | seq_peak med | seq_peak mean |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for group_name, group_probes in [
        ("correct", [p for p in probes_clean if p.bucket == "correct"]),
        ("A", [p for p in probes_clean if p.bucket == "A"]),
        ("B", [p for p in probes_clean if p.bucket == "B"]),
        ("C", [p for p in probes_clean if p.bucket == "C"]),
        ("D", [p for p in probes_clean if p.bucket == "D"]),
        ("E", [p for p in probes_clean if p.bucket == "E"]),
    ]:
        if not group_probes:
            continue
        confs = [p.pre_override_confidence for p in group_probes]
        peaks = [
            p.seq_peak_prob for p in group_probes if p.seq_present
        ]
        lines.append(
            f"| {group_name} | {len(group_probes)} "
            f"| {np.median(confs):.3f} | {np.mean(confs):.3f} "
            f"| {np.median(peaks) if peaks else 0.0:.3f} "
            f"| {np.mean(peaks) if peaks else 0.0:.3f} |"
        )
    lines.append("")

    # Recommendation (generated from the numbers — leave blank for the human
    # reviewer to write the final paragraph; but populate a data-driven hint).
    lines.append("## Recommendation\n")
    ranked = sorted(
        [(b, clean_counts.get(b, 0)) for b in "ABCDE"],
        key=lambda x: -x[1],
    )
    top_bucket, top_n = ranked[0]
    lines.append(
        f"The largest error bucket on the clean pool is **{top_bucket}** "
        f"({top_n} errors — {bucket_meaning[top_bucket]}). "
        "See per-bucket lever notes:\n"
    )
    levers = {
        "A": (
            "The MS-TCN++ override takes correct GBM predictions and flips them wrong. "
            "Add pair-specific guards (like `DIG_GUARD_RATIO` but for the dominant "
            "confusion pair observed in A) to prevent the regression. Fast to ship."
        ),
        "B": (
            "Both GBM and MS-TCN++ agree on the wrong class. The current features + "
            "models cannot distinguish these. Candidate next experiments: new feature "
            "families (post-contact ball trajectory, opponent pose, longer-window "
            "wrist velocity curves), MS-TCN++ retrain with added input channels, or "
            "external-dataset pretrain."
        ),
        "C": (
            "GBM and MS-TCN++ disagree but both land on wrong classes. This suggests "
            "an ensembling/calibration problem — a probability-weighted combination "
            "might beat the current hard-override scheme. Cheap to prototype: "
            "sweep alternative ensemble rules offline."
        ),
        "D": (
            "The `DIG_GUARD_RATIO=2.5` guard is blocking correct overrides. Tune "
            "the threshold or make it asymmetric by confusion pair. Very cheap: "
            "a single sweep over the guard value on the existing canonical set."
        ),
        "E": (
            "MS-TCN++ is not running on these rallies. Investigate why (rally too "
            "short? features failed to extract?). Likely a small edge-case fix."
        ),
    }
    for b, _n in ranked:
        lines.append(f"- **{b}**: {levers[b]}\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rally", type=str, help="Restrict to a single rally id")
    parser.add_argument("--tolerance-ms", type=int, default=167,
                        help="GT-match tolerance (~5 frames @30fps)")
    parser.add_argument("--skip-session", type=str, default=POOR_SESSION_ID,
                        help="Session id to treat as 'poor' and exclude from "
                             "the clean-pool summary (default: the 2026-04-14 "
                             "'poor' session).")
    parser.add_argument("--output", type=str, default=None,
                        help="Markdown report path. Defaults to "
                             "analysis/outputs/action_error_origin_<date>.md")
    parser.add_argument("--probes-json", type=str, default=None,
                        help="Optional: dump all ContactProbes as JSON for "
                             "downstream analysis.")
    args = parser.parse_args()

    rallies = load_rallies_with_action_gt(rally_id=args.rally)
    if not rallies:
        console.print("[red]No rallies with action GT found.[/red]")
        return
    console.print(f"[bold]Loaded {len(rallies)} rallies with action GT.[/bold]")

    video_ids = {r.video_id for r in rallies if r.video_id}

    # Build the same production-eval context: teams, t2p, calibrators, flips,
    # camera heights. Match production path parity.
    rally_pos_lookup: dict[str, list[PlayerPosition]] = {}
    for r in rallies:
        if r.positions_json:
            rally_pos_lookup[r.rally_id] = [
                PlayerPosition(
                    frame_number=pp["frameNumber"],
                    track_id=pp["trackId"],
                    x=pp["x"], y=pp["y"],
                    width=pp["width"], height=pp["height"],
                    confidence=pp.get("confidence", 1.0),
                    keypoints=pp.get("keypoints"),
                )
                for pp in r.positions_json
            ]
    match_team_map = _load_match_team_assignments(
        video_ids, rally_positions=rally_pos_lookup,
    )
    t2p_map = _load_track_to_player_maps(video_ids)
    calibrators = _build_calibrators(video_ids)
    camera_heights = _build_camera_heights(video_ids, calibrators)
    flips = _load_formation_semantic_flips_from_gt(video_ids)
    video_sessions = _load_video_sessions(video_ids)

    probes_all: list[ContactProbe] = []
    n_rejected = 0
    n_evaluated = 0

    for i, rally in enumerate(rallies, start=1):
        if not rally.ball_positions_json or not rally.positions_json:
            n_rejected += 1
            console.print(
                f"  [{i}/{len(rallies)}] {rally.rally_id[:8]} [yellow]skipped "
                f"(missing positions/ball)[/yellow]"
            )
            continue
        if not rally.frame_count or rally.frame_count < 10:
            n_rejected += 1
            console.print(
                f"  [{i}/{len(rallies)}] {rally.rally_id[:8]} [yellow]skipped "
                f"(frame_count={rally.frame_count})[/yellow]"
            )
            continue

        tol = max(1, round(rally.fps * args.tolerance_ms / 1000))
        try:
            probes = _probe_rally(
                rally=rally,
                match_teams=match_team_map.get(rally.rally_id),
                calibrator=calibrators.get(rally.video_id),
                track_to_player=t2p_map.get(rally.rally_id),
                formation_semantic_flip=flips.get(rally.rally_id, False),
                camera_height=camera_heights.get(rally.video_id, 0.0),
                video_session=video_sessions.get(rally.video_id),
                tolerance_frames=tol,
            )
        except Exception as e:  # noqa: BLE001
            n_rejected += 1
            console.print(
                f"  [{i}/{len(rallies)}] {rally.rally_id[:8]} [red]ERROR: {e}[/red]"
            )
            continue

        n_evaluated += 1
        probes_all.extend(probes)

        n_err = sum(1 for p in probes if p.is_error)
        n_cor = sum(1 for p in probes if p.bucket == "correct")
        console.print(
            f"  [{i}/{len(rallies)}] {rally.rally_id[:8]} "
            f"matched={len(probes)} correct={n_cor} errors={n_err}"
        )

    console.print()
    if n_rejected:
        console.print(f"[yellow]Skipped {n_rejected} rallies.[/yellow]")

    probes_clean: list[ContactProbe] = [
        p for p in probes_all if p.session_id != args.skip_session
    ]
    n_rallies_clean = len({p.rally_id for p in probes_clean})

    console.print(
        f"[bold]Collected {len(probes_all)} matched contacts across "
        f"{n_evaluated} rallies ({len(probes_clean)} in clean pool).[/bold]"
    )

    _print_console_report(probes_all, title="== FULL POOL ==")
    _print_console_report(probes_clean, title="== CLEAN POOL (poor session excluded) ==")

    # Write markdown report.
    if args.output:
        output_path = Path(args.output)
    else:
        date_str = datetime.now().strftime("%Y-%m-%d")
        output_path = Path("outputs") / f"action_error_origin_{date_str}.md"
    _write_markdown_report(
        probes_all, probes_clean, output_path,
        n_rallies_loaded=len(rallies),
        n_rallies_evaluated=n_evaluated,
        n_rallies_clean=n_rallies_clean,
    )
    console.print(f"\n[green]Report written to {output_path}[/green]")

    if args.probes_json:
        probes_path = Path(args.probes_json)
        probes_path.parent.mkdir(parents=True, exist_ok=True)
        dumped = [
            {
                "rally_id": p.rally_id,
                "video_id": p.video_id,
                "session_id": p.session_id,
                "contact_frame": p.contact_frame,
                "gt_frame": p.gt_frame,
                "gt_action": p.gt_action,
                "pre_override_type": p.pre_override_type,
                "pre_override_confidence": p.pre_override_confidence,
                "post_override_type": p.post_override_type,
                "override_argmax": p.override_argmax,
                "override_would_apply": p.override_would_apply,
                "guard_fired": p.guard_fired,
                "seq_present": p.seq_present,
                "seq_probs_nonbg": p.seq_probs_nonbg,
                "seq_peak_prob": p.seq_peak_prob,
                "pose_present": p.pose_present,
                "pose_count": p.pose_count,
                "pose_mean_conf": p.pose_mean_conf,
                "is_error": p.is_error,
                "bucket": p.bucket,
            }
            for p in probes_all
        ]
        probes_path.write_text(json.dumps(dumped, indent=2), encoding="utf-8")
        console.print(f"[green]Probes dumped to {probes_path}[/green]")


if __name__ == "__main__":
    main()
