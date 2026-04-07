"""Diagnose whether MS-TCN++ sees the contacts the trajectory/pose classifier misses.

For each FN contact, measure the sequence model's peak non-background
probability within ±W frames of the GT frame. Compare against a control set
of random non-contact frames (>=15 frames from any GT or any prediction),
sampled in the same rallies. The go/no-go question for Option A
(sequence-model contact recovery) is whether the peak distribution on FNs
separates from the peak distribution on the control.

Design notes:
- Runs the production pipeline stages 9-10 (verify_team_assignments,
  get_sequence_probs) and stages 11-12 (detect_contacts) the same way
  `production_eval.py:_run_rally` does, so the FN pool matches the
  dashboard-canonical definition.
- FN categorisation reuses `diagnose_fn_contacts.diagnose_rally_fns`.
- Reports CDFs at τ ∈ {0.20, 0.30, ..., 0.80}, per-category and
  per-GT-action breakdowns, localisation (|peak - gt| histogram), and an
  ROC-style table matching sensitivity on classifier-rejected FNs to
  specificity on control frames.
- Read-only: never writes model weights or metrics files. Safe to rerun.

Usage:
    cd analysis
    uv run python scripts/diagnose_fn_sequence_signal.py
    uv run python scripts/diagnose_fn_sequence_signal.py --window 5 --control 20 --rally <id>
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from diagnose_fn_contacts import (  # noqa: E402
    FNDiagnostic,
    _reconstruct_ball_player_data,
    diagnose_rally_fns,
)
from eval_action_detection import (  # noqa: E402
    GtLabel,
    RallyData,
    _build_player_positions,
    _load_match_team_assignments,
    load_rallies_with_action_gt,
    match_contacts,
)

from rallycut.tracking.contact_classifier import load_contact_classifier  # noqa: E402
from rallycut.tracking.contact_detector import detect_contacts  # noqa: E402
from rallycut.tracking.match_tracker import verify_team_assignments  # noqa: E402
from rallycut.tracking.sequence_action_runtime import get_sequence_probs  # noqa: E402

console = Console()

# Class order inside sequence_probs (row 0 is background):
#   0 background | 1 serve | 2 receive | 3 set | 4 attack | 5 dig | 6 block
SEQ_CLASSES = ["bg", "serve", "receive", "set", "attack", "dig", "block"]

THRESHOLDS = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]


@dataclass
class SequenceSignal:
    """Peak non-background sequence-model signal around a frame."""

    rally_id: str
    frame: int
    is_fn: bool
    category: str  # FN category, or "control" for control samples
    gt_action: str  # "" for control
    peak_prob: float  # max(sequence_probs[1:, f±W])
    peak_class: str  # argmax class at peak frame
    peak_offset: int  # signed distance from f to the peak frame


# --------------------------------------------------------------------------- #
# Core extraction                                                             #
# --------------------------------------------------------------------------- #


def _peak_in_window(
    sequence_probs: np.ndarray,
    frame: int,
    window: int,
) -> tuple[float, str, int]:
    """Return (max_nonbg_prob, class_name, signed_offset) in [frame-W, frame+W].

    `sequence_probs` is (7, T). Row 0 is background; rows 1-6 are actions.
    If the window falls outside the model's range the function clips; if
    nothing overlaps it returns zero.
    """
    _, T = sequence_probs.shape
    lo = max(0, frame - window)
    hi = min(T - 1, frame + window)
    if hi < lo:
        return (0.0, "bg", 0)

    sub = sequence_probs[1:, lo : hi + 1]  # (6, W')
    flat_idx = int(np.argmax(sub))
    cls_idx, col = divmod(flat_idx, sub.shape[1])
    peak_frame = lo + col
    return (float(sub[cls_idx, col]), SEQ_CLASSES[cls_idx + 1], peak_frame - frame)


def _sample_control_frames(
    rally: RallyData,
    predicted_frames: set[int],
    k: int,
    exclusion: int,
    rng: random.Random,
) -> list[int]:
    """Pick `k` random frames >= `exclusion` from any GT or any predicted contact."""
    if not rally.frame_count or rally.frame_count <= 2 * exclusion:
        return []
    forbidden: set[int] = set()
    for gt in rally.gt_labels:
        for d in range(-exclusion, exclusion + 1):
            forbidden.add(gt.frame + d)
    for pf in predicted_frames:
        for d in range(-exclusion, exclusion + 1):
            forbidden.add(pf + d)

    candidates = [f for f in range(exclusion, rally.frame_count - exclusion) if f not in forbidden]
    if not candidates:
        return []
    return rng.sample(candidates, min(k, len(candidates)))


def _process_rally(
    rally: RallyData,
    match_teams: dict[int, int] | None,
    classifier,
    window: int,
    n_control: int,
    exclusion: int,
    tolerance_frames: int,
    rng: random.Random,
) -> list[SequenceSignal]:
    """Run production stages 9-12, extract FN sequence signal + control samples."""
    ball_positions, _ = _reconstruct_ball_player_data(rally)
    player_positions = _build_player_positions(
        rally.positions_json or [], rally_id=rally.rally_id, inject_pose=True,
    )

    if not ball_positions or not rally.frame_count:
        return []

    # Stage 9.
    teams: dict[int, int] | None = dict(match_teams) if match_teams else None
    if teams is not None:
        teams = verify_team_assignments(teams, player_positions)

    # Stage 10.
    sequence_probs = get_sequence_probs(
        ball_positions=ball_positions,
        player_positions=player_positions,
        court_split_y=rally.court_split_y,
        frame_count=rally.frame_count,
        team_assignments=teams,
        calibrator=None,
    )
    if sequence_probs is None:
        return []

    # Stages 11-12 (default config — production defaults).
    contact_sequence = detect_contacts(
        ball_positions=ball_positions,
        player_positions=player_positions,
        frame_count=rally.frame_count,
        team_assignments=teams,
        sequence_probs=sequence_probs,
    )
    pred_dicts = [c.to_dict() for c in contact_sequence.contacts]
    pred_frames = {c.frame for c in contact_sequence.contacts}

    # Match GT to preds; extract FN frames.
    matches, _ = match_contacts(rally.gt_labels, pred_dicts, tolerance=tolerance_frames)
    fn_labels = [
        GtLabel(
            frame=m.gt_frame,
            action=m.gt_action,
            player_track_id=-1,
            ball_x=next((gt.ball_x for gt in rally.gt_labels if gt.frame == m.gt_frame), None),
            ball_y=next((gt.ball_y for gt in rally.gt_labels if gt.frame == m.gt_frame), None),
        )
        for m in matches if m.pred_frame is None
    ]
    if not fn_labels:
        fn_diags: list[FNDiagnostic] = []
    else:
        fn_diags = diagnose_rally_fns(rally, fn_labels, classifier, tolerance_frames)

    signals: list[SequenceSignal] = []

    # FN frames.
    for d in fn_diags:
        peak_prob, peak_class, peak_offset = _peak_in_window(sequence_probs, d.gt_frame, window)
        signals.append(SequenceSignal(
            rally_id=rally.rally_id,
            frame=d.gt_frame,
            is_fn=True,
            category=d.category,
            gt_action=d.gt_action,
            peak_prob=peak_prob,
            peak_class=peak_class,
            peak_offset=peak_offset,
        ))

    # Control frames.
    for f in _sample_control_frames(rally, pred_frames, n_control, exclusion, rng):
        peak_prob, peak_class, peak_offset = _peak_in_window(sequence_probs, f, window)
        signals.append(SequenceSignal(
            rally_id=rally.rally_id,
            frame=f,
            is_fn=False,
            category="control",
            gt_action="",
            peak_prob=peak_prob,
            peak_class=peak_class,
            peak_offset=peak_offset,
        ))

    return signals


# --------------------------------------------------------------------------- #
# Reporting                                                                   #
# --------------------------------------------------------------------------- #


def _report_cdf(signals: list[SequenceSignal]) -> None:
    """Print the cumulative fraction >= τ for each FN category and control."""
    by_cat: dict[str, list[float]] = defaultdict(list)
    for s in signals:
        by_cat[s.category].append(s.peak_prob)

    # Order: classifier first (main question), then others, then control last.
    order = [
        "rejected_by_classifier",
        "rejected_by_gates",
        "no_candidate",
        "ball_dropout",
        "deduplicated",
        "no_player_nearby",
        "low_conf_ball",
        "control",
    ]
    cats = [c for c in order if c in by_cat]

    tbl = Table(title="Fraction of frames with max(seq_probs[1:, f±W]) >= τ")
    tbl.add_column("Category")
    tbl.add_column("n", justify="right")
    for t in THRESHOLDS:
        tbl.add_column(f">={t:.2f}", justify="right")

    for cat in cats:
        probs = by_cat[cat]
        n = len(probs)
        row = [cat, str(n)]
        for t in THRESHOLDS:
            frac = sum(1 for p in probs if p >= t) / max(1, n)
            row.append(f"{frac * 100:5.1f}%")
        tbl.add_row(*row)
    console.print(tbl)


def _report_per_action(signals: list[SequenceSignal]) -> None:
    """rejected_by_classifier subset, broken down by GT action type."""
    subset = [s for s in signals if s.category == "rejected_by_classifier"]
    if not subset:
        return

    by_act: dict[str, list[float]] = defaultdict(list)
    for s in subset:
        by_act[s.gt_action].append(s.peak_prob)

    tbl = Table(title="rejected_by_classifier FNs — peak prob CDF by GT action")
    tbl.add_column("GT action")
    tbl.add_column("n", justify="right")
    for t in THRESHOLDS:
        tbl.add_column(f">={t:.2f}", justify="right")

    for act in sorted(by_act.keys()):
        probs = by_act[act]
        n = len(probs)
        row = [act, str(n)]
        for t in THRESHOLDS:
            frac = sum(1 for p in probs if p >= t) / max(1, n)
            row.append(f"{frac * 100:5.1f}%")
        tbl.add_row(*row)
    console.print(tbl)


def _report_roc(signals: list[SequenceSignal]) -> None:
    """ROC-style: at each τ, sensitivity on classifier FNs vs specificity on control.

    Expected recall lift = (1 - FPR_per_rally * rally_length_cost) * TPR.
    We just print TPR and FPR; the reader decides.
    """
    fn_probs = [s.peak_prob for s in signals if s.category == "rejected_by_classifier"]
    ctrl_probs = [s.peak_prob for s in signals if s.category == "control"]
    if not fn_probs or not ctrl_probs:
        return

    tbl = Table(title="ROC: sensitivity on rejected_by_classifier FNs vs FPR on control")
    tbl.add_column("τ", justify="right")
    tbl.add_column("TPR (recover rate)", justify="right")
    tbl.add_column("FPR (control trigger rate)", justify="right")
    tbl.add_column("Ratio", justify="right")

    nfn = len(fn_probs)
    nc = len(ctrl_probs)
    for t in THRESHOLDS:
        tpr = sum(1 for p in fn_probs if p >= t) / nfn
        fpr = sum(1 for p in ctrl_probs if p >= t) / nc
        ratio = tpr / fpr if fpr > 0 else float("inf")
        tbl.add_row(
            f"{t:.2f}", f"{tpr * 100:5.1f}%", f"{fpr * 100:5.1f}%",
            "∞" if fpr == 0 else f"{ratio:5.2f}x",
        )
    console.print(tbl)


def _report_peak_class(signals: list[SequenceSignal]) -> None:
    """For classifier-rejected FNs with peak >= 0.5, does peak class match GT?"""
    subset = [
        s for s in signals
        if s.category == "rejected_by_classifier" and s.peak_prob >= 0.5
    ]
    if not subset:
        console.print(
            "\n[dim]No rejected_by_classifier FNs with peak_prob >= 0.5 — "
            "sequence model does not localise these events confidently.[/dim]"
        )
        return

    match_counts: Counter[str] = Counter()
    for s in subset:
        match_counts["match" if s.peak_class == s.gt_action else f"{s.gt_action}->{s.peak_class}"] += 1

    tbl = Table(title=f"Peak class vs GT for rejected_by_classifier FNs with peak>=0.50 (n={len(subset)})")
    tbl.add_column("Case")
    tbl.add_column("Count", justify="right")
    for case, cnt in match_counts.most_common():
        tbl.add_row(case, str(cnt))
    console.print(tbl)


def _report_localization(signals: list[SequenceSignal]) -> None:
    """Peak-offset histogram for classifier-rejected FNs with peak >= 0.5."""
    subset = [
        s for s in signals
        if s.category == "rejected_by_classifier" and s.peak_prob >= 0.5
    ]
    if not subset:
        return

    offsets = [abs(s.peak_offset) for s in subset]
    buckets = Counter()
    for o in offsets:
        if o == 0:
            buckets["0"] += 1
        elif o <= 2:
            buckets["1-2"] += 1
        elif o <= 5:
            buckets["3-5"] += 1
        else:
            buckets[">5"] += 1

    tbl = Table(title="Peak localisation |offset| from GT frame (classifier-rejected, peak>=0.5)")
    tbl.add_column("|offset|")
    tbl.add_column("Count", justify="right")
    for key in ["0", "1-2", "3-5", ">5"]:
        tbl.add_row(key, str(buckets.get(key, 0)))
    console.print(tbl)


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rally", type=str, help="Restrict to single rally id")
    parser.add_argument("--window", type=int, default=5,
                        help="Half-window for peak search around FN frame (default 5)")
    parser.add_argument("--control", type=int, default=20,
                        help="Control (non-contact) samples per rally (default 20)")
    parser.add_argument("--exclusion", type=int, default=15,
                        help="Min frame distance from any GT or pred to be a valid control (default 15)")
    parser.add_argument("--tolerance-ms", type=int, default=167,
                        help="GT-match tolerance in ms (default 167, ~5 frames @ 30fps)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    classifier = load_contact_classifier()
    if classifier is None:
        console.print("[red]No contact classifier loaded — aborting (production parity required).[/red]")
        return
    console.print(
        f"[dim]Loaded contact classifier (threshold={classifier.threshold:.2f})[/dim]"
    )

    rallies = load_rallies_with_action_gt(rally_id=args.rally)
    if not rallies:
        console.print("[red]No rallies with action GT found.[/red]")
        return

    # Build verified match teams exactly as production_eval does: parse
    # PlayerPosition objects (not raw dicts) so verify_team_assignments can run.
    from rallycut.tracking.player_tracker import PlayerPosition  # noqa: PLC0415

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
    video_ids = {r.video_id for r in rallies if r.video_id}
    match_team_map = _load_match_team_assignments(
        video_ids, rally_positions=rally_pos_lookup,
    )

    console.print(
        f"\n[bold]Diagnosing sequence signal on FN contacts across {len(rallies)} rallies"
        f" (window=±{args.window}, control={args.control}/rally)[/bold]\n"
    )

    all_signals: list[SequenceSignal] = []
    n_skipped = 0

    for i, rally in enumerate(rallies):
        tol = max(1, round(rally.fps * args.tolerance_ms / 1000))
        try:
            match_teams = match_team_map.get(rally.rally_id)
            signals = _process_rally(
                rally, match_teams, classifier,
                window=args.window, n_control=args.control, exclusion=args.exclusion,
                tolerance_frames=tol, rng=rng,
            )
        except Exception as e:  # noqa: BLE001
            n_skipped += 1
            console.print(f"  [{i + 1}/{len(rallies)}] {rally.rally_id[:8]} [red]SKIPPED: {e}[/red]")
            continue

        all_signals.extend(signals)

        # Per-rally progress.
        n_fn = sum(1 for s in signals if s.is_fn)
        n_clf = sum(
            1 for s in signals
            if s.category == "rejected_by_classifier" and s.peak_prob >= 0.5
        )
        console.print(
            f"  [{i + 1}/{len(rallies)}] {rally.rally_id[:8]} "
            f"FN={n_fn} clf_recoverable@0.5={n_clf}"
        )

    console.print()
    if n_skipped:
        console.print(f"[yellow]Skipped {n_skipped} rallies due to errors.[/yellow]\n")

    n_fn_total = sum(1 for s in all_signals if s.is_fn)
    n_ctrl = sum(1 for s in all_signals if not s.is_fn)
    console.print(
        f"[bold]Collected {len(all_signals)} samples: {n_fn_total} FN + {n_ctrl} control[/bold]\n"
    )

    if not all_signals:
        return

    _report_cdf(all_signals)
    console.print()
    _report_per_action(all_signals)
    console.print()
    _report_roc(all_signals)
    console.print()
    _report_peak_class(all_signals)
    console.print()
    _report_localization(all_signals)


if __name__ == "__main__":
    main()
