"""Diagnose serve-side formation detection: decompose errors and measure features.

Separates formation_side (physical near/far) accuracy from team mapping (A/B).
Measures 7 candidate position features and their discriminative power.

Usage:
    cd analysis
    uv run python scripts/diagnose_formation_serve.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rallycut.court.calibration import CourtCalibrator  # noqa: E402
from rallycut.evaluation.tracking.db import load_court_calibration  # noqa: E402
from rallycut.tracking.action_classifier import (  # noqa: E402
    _compute_auto_split_y,
    _find_serving_side_by_formation,
)
from rallycut.tracking.player_tracker import PlayerPosition  # noqa: E402
from scripts.eval_score_tracking import RallyData, load_score_gt  # noqa: E402

console = Console()

# Court-space constants (meters)
_COURT_NET_Y = 8.0
_COURT_LENGTH = 16.0


# ── Data helpers ────────────────────────────────────────────────────────


def _parse_positions(positions_json: list[dict]) -> list[PlayerPosition]:
    out: list[PlayerPosition] = []
    for pp in positions_json:
        out.append(PlayerPosition(
            frame_number=pp["frameNumber"],
            track_id=pp["trackId"],
            x=pp["x"],
            y=pp["y"],
            width=pp.get("width", 0.05),
            height=pp.get("height", 0.10),
            confidence=pp.get("confidence", 1.0),
            keypoints=pp.get("keypoints"),
        ))
    return out


def _gt_physical_side(
    gt_serving_team: str, side_flipped: bool, initial_near_is_a: bool = True,
) -> str:
    """Derive GT physical side from semantic team + flip state.

    The A/B label convention is per-video (GT labeler chose which team
    is "A"). ``initial_near_is_a`` tells us whether, at video start,
    team A is on the near side.

    After accounting for side switches (``side_flipped``):
    - near_is_a = initial_near_is_a XOR side_flipped
    """
    near_is_a = initial_near_is_a != side_flipped  # XOR
    if gt_serving_team == "A":
        return "near" if near_is_a else "far"
    else:
        return "far" if near_is_a else "near"


def _calibrate_initial_near_is_a(
    rallies: list[RallyData],
    positions_by_rally: dict[str, list[PlayerPosition]],
    net_ys: dict[str, float],
) -> bool:
    """Determine whether team A starts on near side for a video.

    Runs formation predictor on all rallies, counts agreement with both
    conventions, returns the one with more matches. This is label
    alignment — requires only 1 correct formation prediction to work.
    """
    votes_true = 0  # near_is_a = True
    votes_false = 0  # near_is_a = False

    for rally in rallies:
        positions = positions_by_rally.get(rally.rally_id, [])
        net_y = net_ys.get(rally.rally_id, 0.5)
        pred_side, _ = _find_serving_side_by_formation(
            positions, net_y=net_y, start_frame=0,
        )
        if pred_side is None:
            continue

        gt_if_true = _gt_physical_side(
            rally.gt_serving_team, rally.side_flipped, initial_near_is_a=True,
        )
        gt_if_false = _gt_physical_side(
            rally.gt_serving_team, rally.side_flipped, initial_near_is_a=False,
        )

        if pred_side == gt_if_true:
            votes_true += 1
        if pred_side == gt_if_false:
            votes_false += 1

    return votes_true >= votes_false


# ── Formation internals (reproduce to capture intermediate values) ──────


@dataclass
class FormationDiag:
    rally_id: str
    video_id: str
    gt_physical: str  # "near" / "far"
    gt_serving_team: str
    side_flipped: bool

    # Formation predictor output
    pred_side: str | None  # "near" / "far" / None
    confidence: float

    # Internals
    n_tracks: int
    n_near: int
    n_far: int
    near_sep: float
    far_sep: float
    ratio: float
    used_auto_split: bool
    effective_split_y: float
    court_split_y: float | None

    # Result
    correct: bool | None  # None if abstained

    # Features (computed later)
    features: dict[str, float | None] = field(default_factory=dict)


def _run_formation_with_internals(
    positions: list[PlayerPosition],
    net_y: float,
    start_frame: int = 0,
    window_frames: int = 120,
) -> dict:
    """Reproduce formation internals for diagnostic capture."""
    result: dict = {
        "pred_side": None, "confidence": 0.0,
        "n_tracks": 0, "n_near": 0, "n_far": 0,
        "near_sep": 0.0, "far_sep": 0.0, "ratio": 0.0,
        "used_auto_split": False, "effective_split_y": net_y,
    }

    if not positions:
        return result

    end_frame = start_frame + window_frames
    by_track: dict[int, list[float]] = defaultdict(list)
    for p in positions:
        if p.track_id < 0:
            continue
        if start_frame <= p.frame_number < end_frame:
            by_track[p.track_id].append(p.y + p.height / 2.0)

    result["n_tracks"] = len(by_track)
    if len(by_track) < 2:
        return result

    effective_split = net_y
    track_medians = {tid: sum(ys) / len(ys) for tid, ys in by_track.items()}
    near_count = sum(1 for y in track_medians.values() if y > effective_split)
    far_count = len(track_medians) - near_count

    if near_count == 0 or far_count == 0:
        auto_split = _compute_auto_split_y(positions)
        if auto_split is None:
            return result
        effective_split = auto_split
        result["used_auto_split"] = True

    result["effective_split_y"] = effective_split

    near_tids = [t for t, y in track_medians.items() if y > effective_split]
    far_tids = [t for t, y in track_medians.items() if y <= effective_split]
    result["n_near"] = len(near_tids)
    result["n_far"] = len(far_tids)

    if not near_tids or not far_tids:
        return result

    def _sep(tids: list[int]) -> float:
        if len(tids) >= 2:
            ys = [track_medians[t] for t in tids]
            return max(ys) - min(ys)
        return abs(track_medians[tids[0]] - effective_split) * 0.5

    near_sep = _sep(near_tids)
    far_sep = _sep(far_tids)
    ratio = max(near_sep, far_sep) / max(min(near_sep, far_sep), 1e-6)

    result["near_sep"] = near_sep
    result["far_sep"] = far_sep
    result["ratio"] = ratio

    if ratio < 1.0 + 1e-4:
        return result

    if near_sep >= far_sep:
        result["pred_side"] = "near"
    else:
        result["pred_side"] = "far"
    result["confidence"] = min(1.0, ratio - 1.0)

    return result


# ── Candidate features ──────────────────────────────────────────────────


def _compute_features(
    positions: list[PlayerPosition],
    ball_positions: list[dict],
    net_y: float,
    effective_split: float,
    calibrator: CourtCalibrator | None,
    window_frames: int = 120,
) -> dict[str, float | None]:
    """Compute 7 candidate features for serve-side prediction.

    All features use positive = near more likely serving convention.
    """
    features: dict[str, float | None] = {}

    # Parse positions in window
    by_track: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for p in positions:
        if p.track_id < 0:
            continue
        if 0 <= p.frame_number < window_frames:
            by_track[p.track_id].append((p.x, p.y + p.height / 2.0))

    if len(by_track) < 2:
        return {f"f{i}": None for i in range(1, 8)}

    track_pos = {
        tid: (
            float(np.mean([xy[0] for xy in xys])),
            float(np.mean([xy[1] for xy in xys])),
        )
        for tid, xys in by_track.items()
    }

    near_tids = [t for t, (_, y) in track_pos.items() if y > effective_split]
    far_tids = [t for t, (_, y) in track_pos.items() if y <= effective_split]

    if not near_tids or not far_tids:
        return {f"f{i}": None for i in range(1, 8)}

    # Feature 1: Current separation ratio (near_sep - far_sep)
    def _sep(tids: list[int]) -> float:
        if len(tids) >= 2:
            ys = [track_pos[t][1] for t in tids]
            return max(ys) - min(ys)
        return abs(track_pos[tids[0]][1] - effective_split) * 0.5
    features["f1_separation"] = _sep(near_tids) - _sep(far_tids)

    # Feature 2: Server isolation (most isolated player)
    all_tids = list(track_pos.keys())
    isolation: dict[int, float] = {}
    for tid in all_tids:
        px, py = track_pos[tid]
        min_dist = float("inf")
        for other in all_tids:
            if other == tid:
                continue
            ox, oy = track_pos[other]
            d = ((px - ox) ** 2 + (py - oy) ** 2) ** 0.5
            min_dist = min(min_dist, d)
        isolation[tid] = min_dist

    near_max_iso = max(isolation[t] for t in near_tids)
    far_max_iso = max(isolation[t] for t in far_tids)
    features["f2_isolation"] = near_max_iso - far_max_iso

    # Feature 3: Baseline proximity (image-space)
    # Near baseline = bottom of frame (y→1), far baseline = top (y→0)
    near_max_y = max(track_pos[t][1] for t in near_tids)
    far_min_y = min(track_pos[t][1] for t in far_tids)
    # How close is the farthest-from-net player to their baseline?
    near_bl_prox = near_max_y  # Higher = closer to near baseline (bottom)
    far_bl_prox = 1.0 - far_min_y  # Higher = closer to far baseline (top)
    features["f3_baseline_img"] = near_bl_prox - far_bl_prox

    # Feature 4: Baseline proximity (court-space)
    if calibrator is not None and calibrator.is_calibrated:
        try:
            court_ys: dict[int, float] = {}
            for tid, (fx, fy) in track_pos.items():
                _, cy = calibrator.image_to_court((fx, fy), 1, 1)
                court_ys[tid] = cy

            # Near baseline at cy≈0, far baseline at cy≈16
            near_bl_dists = [court_ys[t] for t in near_tids if t in court_ys]
            far_bl_dists = [_COURT_LENGTH - court_ys[t]
                            for t in far_tids if t in court_ys]
            if near_bl_dists and far_bl_dists:
                near_closest_bl = min(near_bl_dists)
                far_closest_bl = min(far_bl_dists)
                # Lower distance = closer to baseline = more likely serving
                features["f4_baseline_court"] = far_closest_bl - near_closest_bl
            else:
                features["f4_baseline_court"] = None
        except Exception:
            features["f4_baseline_court"] = None
    else:
        features["f4_baseline_court"] = None

    # Feature 5: Max net distance
    near_max_net_dist = max(abs(track_pos[t][1] - net_y) for t in near_tids)
    far_max_net_dist = max(abs(track_pos[t][1] - net_y) for t in far_tids)
    features["f5_net_dist"] = near_max_net_dist - far_max_net_dist

    # Feature 6: Ball position at rally start
    ball_ys = [
        bp["y"] for bp in ball_positions
        if bp.get("frameNumber", bp.get("frame", 999)) < window_frames
        and "y" in bp
    ]
    if ball_ys:
        med_ball_y = float(np.median(ball_ys))
        features["f6_ball_pos"] = med_ball_y - net_y  # positive = near side
    else:
        features["f6_ball_pos"] = None

    # Feature 7: Player count asymmetry
    features["f7_count_asym"] = float(len(near_tids) - len(far_tids))

    return features


# ── Feature evaluation ──────────────────────────────────────────────────


def _compute_auc(scores: list[float], labels: list[int]) -> float:
    """AUC-ROC via sklearn. Returns max(auc, 1-auc) for direction-agnostic."""
    from sklearn.metrics import roc_auc_score
    if len(set(labels)) < 2:
        return 0.5
    auc = roc_auc_score(labels, scores)
    return max(auc, 1.0 - auc)


def _best_threshold_acc(scores: list[float], labels: list[int]) -> float:
    """Best accuracy achievable by a single threshold on the score."""
    if not scores:
        return 0.0
    sorted_scores = sorted(set(scores))
    best = 0.0
    for i in range(len(sorted_scores)):
        if i == 0:
            thresh = sorted_scores[0] - 1.0
        else:
            thresh = (sorted_scores[i - 1] + sorted_scores[i]) / 2.0
        preds = [1 if s > thresh else 0 for s in scores]
        acc = sum(1 for p, l in zip(preds, labels) if p == l) / len(labels)
        best = max(best, acc, 1.0 - acc)
    return best


def _loo_video_combined(
    diags: list[FormationDiag],
    feature_names: list[str],
) -> dict[str, float]:
    """LOO-video CV with logistic regression and decision tree."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier

    videos = sorted(set(d.video_id for d in diags))
    results: dict[str, list[tuple[int, int]]] = {
        "logistic": [], "tree": [],
    }

    for held_out in videos:
        train = [d for d in diags if d.video_id != held_out]
        test = [d for d in diags if d.video_id == held_out]

        # Build feature matrices (skip rallies with any None feature)
        def _build_xy(ds: list[FormationDiag]) -> tuple[np.ndarray, np.ndarray]:
            X, y = [], []
            for d in ds:
                vals = [d.features.get(fn) for fn in feature_names]
                if any(v is None for v in vals):
                    continue
                X.append(vals)
                y.append(1 if d.gt_physical == "near" else 0)
            return np.array(X, dtype=float), np.array(y, dtype=int)

        X_train, y_train = _build_xy(train)
        X_test, y_test = _build_xy(test)

        if len(X_train) < 5 or len(X_test) == 0 or len(set(y_train)) < 2:
            continue

        for name, clf in [
            ("logistic", LogisticRegression(max_iter=1000, C=1.0)),
            ("tree", DecisionTreeClassifier(max_depth=3)),
        ]:
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            for p, l in zip(preds, y_test):
                results[name].append((int(p), int(l)))

    out = {}
    for name, pairs in results.items():
        if pairs:
            correct = sum(1 for p, l in pairs if p == l)
            out[name] = correct / len(pairs)
        else:
            out[name] = 0.0
    return out


# ── Main ────────────────────────────────────────────────────────────────


def main() -> int:
    console.print("[bold]Loading score GT...[/bold]")
    video_rallies = load_score_gt()
    total = sum(len(v) for v in video_rallies.values())
    console.print(f"Loaded {total} rallies across {len(video_rallies)} videos\n")

    # Load court calibrations
    calibrators: dict[str, CourtCalibrator] = {}
    for vid in video_rallies:
        corners = load_court_calibration(vid)
        if corners and len(corners) == 4:
            cal = CourtCalibrator()
            cal.calibrate([(c["x"], c["y"]) for c in corners])
            if cal.is_calibrated:
                calibrators[vid] = cal
    console.print(f"Court calibrations: {len(calibrators)}/{len(video_rallies)} videos\n")

    # ── Per-video convention calibration ────────────────────────────────

    # Pre-parse positions and net_y for all rallies
    positions_by_rally: dict[str, list[PlayerPosition]] = {}
    net_ys: dict[str, float] = {}
    for vid, rallies in video_rallies.items():
        for rally in rallies:
            positions_by_rally[rally.rally_id] = _parse_positions(rally.positions)
            net_ys[rally.rally_id] = rally.court_split_y if rally.court_split_y else 0.5

    # Calibrate per-video: does team A start near or far?
    initial_near_is_a: dict[str, bool] = {}
    console.print("[bold]Per-video convention calibration:[/bold]")
    for vid, rallies in sorted(video_rallies.items()):
        is_a = _calibrate_initial_near_is_a(rallies, positions_by_rally, net_ys)
        initial_near_is_a[vid] = is_a
        console.print(f"  {vid[:10]}: near={'A' if is_a else 'B'} at start")
    console.print()

    # ── Process all rallies ──────────────────────────────────────────────

    diags: list[FormationDiag] = []

    for vid, rallies in sorted(video_rallies.items()):
        cal = calibrators.get(vid)
        near_a = initial_near_is_a[vid]
        for rally in rallies:
            positions = positions_by_rally[rally.rally_id]
            net_y = net_ys[rally.rally_id]

            # Formation predictor (black-box call for correctness check)
            pred_side_bb, conf_bb = _find_serving_side_by_formation(
                positions, net_y=net_y, start_frame=0,
            )

            # Formation internals
            internals = _run_formation_with_internals(positions, net_y)

            # Note: _run_formation_with_internals uses the old separation-ratio
            # model, not the current multi-feature logistic. Skip assertion —
            # the black-box call is the production path.
            # Use black-box prediction as the authoritative pred_side.
            internals["pred_side"] = pred_side_bb
            internals["confidence"] = conf_bb

            gt_phys = _gt_physical_side(
                rally.gt_serving_team, rally.side_flipped, near_a,
            )

            correct: bool | None = None
            if internals["pred_side"] is not None:
                correct = internals["pred_side"] == gt_phys

            # Compute features
            features = _compute_features(
                positions, rally.ball_positions, net_y,
                internals["effective_split_y"], cal,
            )

            diags.append(FormationDiag(
                rally_id=rally.rally_id,
                video_id=vid,
                gt_physical=gt_phys,
                gt_serving_team=rally.gt_serving_team,
                side_flipped=rally.side_flipped,
                pred_side=internals["pred_side"],
                confidence=internals["confidence"],
                n_tracks=internals["n_tracks"],
                n_near=internals["n_near"],
                n_far=internals["n_far"],
                near_sep=internals["near_sep"],
                far_sep=internals["far_sep"],
                ratio=internals["ratio"],
                used_auto_split=internals["used_auto_split"],
                effective_split_y=internals["effective_split_y"],
                court_split_y=rally.court_split_y,
                correct=correct,
                features=features,
            ))

    # ── Section 1: Error Decomposition ───────────────────────────────────

    console.print("[bold]═══ Section 1: Error Decomposition ═══[/bold]\n")

    predicted = [d for d in diags if d.pred_side is not None]
    abstained = [d for d in diags if d.pred_side is None]
    phys_correct = [d for d in predicted if d.correct]
    phys_wrong = [d for d in predicted if not d.correct]

    table = Table(title="Formation Side Detection")
    table.add_column("Stage")
    table.add_column("Correct", justify="right")
    table.add_column("Wrong", justify="right")
    table.add_column("Abstain", justify="right")
    table.add_column("Accuracy", justify="right")
    table.add_column("Coverage", justify="right")

    phys_acc = len(phys_correct) / len(predicted) if predicted else 0
    table.add_row(
        "formation_side (physical)",
        str(len(phys_correct)),
        str(len(phys_wrong)),
        str(len(abstained)),
        f"{phys_acc:.1%}",
        f"{len(predicted) / len(diags):.1%}",
    )

    console.print(table)

    console.print(f"\n  Phase 0 claim ('0% formation errors'): "
                  f"{'CONFIRMED' if len(phys_wrong) == 0 else 'REFUTED'} "
                  f"— {len(phys_wrong)} physical side errors\n")

    # ── Section 2: Formation Error Classification ────────────────────────

    console.print("[bold]═══ Section 2: Formation Error Classification ═══[/bold]\n")

    if phys_wrong:
        # Confidence buckets
        high = [d for d in phys_wrong if d.confidence > 0.5]
        marginal = [d for d in phys_wrong if 0.1 <= d.confidence <= 0.5]
        barely = [d for d in phys_wrong if d.confidence < 0.1]
        console.print(f"  Confidence buckets (wrong predictions):")
        console.print(f"    High (>0.5):    {len(high)}")
        console.print(f"    Marginal:       {len(marginal)}")
        console.print(f"    Barely (<0.1):  {len(barely)}")

        # Player count patterns
        patterns: dict[str, int] = defaultdict(int)
        for d in phys_wrong:
            patterns[f"{d.n_near}v{d.n_far}"] += 1
        console.print(f"\n  Player count patterns (wrong):")
        for pat, cnt in sorted(patterns.items(), key=lambda x: -x[1]):
            console.print(f"    {pat}: {cnt}")

        # Net Y quality
        auto_split_wrong = sum(1 for d in phys_wrong if d.used_auto_split)
        no_split = sum(1 for d in phys_wrong if d.court_split_y is None)
        console.print(f"\n  Net Y quality (wrong): "
                      f"auto_split={auto_split_wrong}, no_court_split_y={no_split}")
    else:
        console.print("  No formation errors — all physical side predictions correct!\n")

    # Abstention analysis
    if abstained:
        console.print(f"\n  Abstention analysis ({len(abstained)} rallies):")
        few_tracks = sum(1 for d in abstained if d.n_tracks < 2)
        one_side = sum(1 for d in abstained
                       if d.n_tracks >= 2 and (d.n_near == 0 or d.n_far == 0))
        equal_sep = sum(1 for d in abstained
                        if d.n_near > 0 and d.n_far > 0 and d.ratio < 1.0001)
        console.print(f"    <2 tracks:       {few_tracks}")
        console.print(f"    All one side:    {one_side}")
        console.print(f"    Equal seps:      {equal_sep}")
        console.print(f"    Other:           {len(abstained) - few_tracks - one_side - equal_sep}")

    # Per-video breakdown
    console.print()
    vid_table = Table(title="Per-Video Formation Side Accuracy")
    vid_table.add_column("Video")
    vid_table.add_column("Rallies", justify="right")
    vid_table.add_column("Correct", justify="right")
    vid_table.add_column("Wrong", justify="right")
    vid_table.add_column("Abstain", justify="right")
    vid_table.add_column("Accuracy", justify="right")
    vid_table.add_column("Has Calibration")

    for vid in sorted(video_rallies.keys()):
        vd = [d for d in diags if d.video_id == vid]
        vp = [d for d in vd if d.pred_side is not None]
        vc = [d for d in vp if d.correct]
        vw = [d for d in vp if not d.correct]
        va = [d for d in vd if d.pred_side is None]
        acc = len(vc) / len(vp) if vp else 0
        vid_table.add_row(
            vid[:10],
            str(len(vd)),
            str(len(vc)),
            str(len(vw)),
            str(len(va)),
            f"{acc:.0%}",
            "Y" if vid in calibrators else "",
        )

    console.print(vid_table)

    # Per-rally error details
    if phys_wrong:
        console.print()
        err_table = Table(title=f"Formation Errors ({len(phys_wrong)} rallies)")
        err_table.add_column("Rally", style="dim")
        err_table.add_column("Video", style="dim")
        err_table.add_column("GT")
        err_table.add_column("Pred", style="red")
        err_table.add_column("Conf", justify="right")
        err_table.add_column("Tracks")
        err_table.add_column("Near Sep", justify="right")
        err_table.add_column("Far Sep", justify="right")
        err_table.add_column("Ratio", justify="right")
        err_table.add_column("Auto Split")
        err_table.add_column("Split Y", justify="right")

        for d in sorted(phys_wrong, key=lambda x: -x.confidence):
            err_table.add_row(
                d.rally_id[:8],
                d.video_id[:10],
                d.gt_physical,
                d.pred_side or "-",
                f"{d.confidence:.3f}",
                f"{d.n_near}v{d.n_far}",
                f"{d.near_sep:.4f}",
                f"{d.far_sep:.4f}",
                f"{d.ratio:.2f}",
                "Y" if d.used_auto_split else "",
                f"{d.effective_split_y:.3f}",
            )
        console.print(err_table)

    # ── Section 3: Candidate Feature Discriminative Power ────────────────

    console.print("\n[bold]═══ Section 3: Feature Discriminative Power ═══[/bold]\n")

    feature_names = [
        "f1_separation", "f2_isolation", "f3_baseline_img",
        "f4_baseline_court", "f5_net_dist", "f6_ball_pos", "f7_count_asym",
    ]
    feature_labels = [
        "Vertical separation",
        "Server isolation",
        "Baseline prox (image)",
        "Baseline prox (court)",
        "Max net distance",
        "Ball position",
        "Player count asym",
    ]

    feat_table = Table(title="Feature Discriminative Power")
    feat_table.add_column("Feature")
    feat_table.add_column("AUC-ROC", justify="right")
    feat_table.add_column("Best Thresh Acc", justify="right")
    feat_table.add_column("Coverage", justify="right")
    feat_table.add_column("Corr w/ f1", justify="right")

    # Collect f1 scores for correlation
    f1_scores = []
    f1_labels = []
    for d in diags:
        v = d.features.get("f1_separation")
        if v is not None:
            f1_scores.append(v)
            f1_labels.append(1 if d.gt_physical == "near" else 0)

    for fn, fl in zip(feature_names, feature_labels):
        scores = []
        labels = []
        for d in diags:
            v = d.features.get(fn)
            if v is not None:
                scores.append(v)
                labels.append(1 if d.gt_physical == "near" else 0)

        coverage = len(scores) / len(diags) if diags else 0
        if len(scores) < 10 or len(set(labels)) < 2:
            feat_table.add_row(fl, "-", "-", f"{coverage:.0%}", "-")
            continue

        auc = _compute_auc(scores, labels)
        best_acc = _best_threshold_acc(scores, labels)

        # Correlation with f1
        corr_str = "-"
        if fn != "f1_separation" and len(scores) > 10:
            # Build aligned arrays
            f1_aligned = []
            fn_aligned = []
            for d in diags:
                v1 = d.features.get("f1_separation")
                vn = d.features.get(fn)
                if v1 is not None and vn is not None:
                    f1_aligned.append(v1)
                    fn_aligned.append(vn)
            if len(f1_aligned) > 10:
                corr = float(np.corrcoef(f1_aligned, fn_aligned)[0, 1])
                corr_str = f"{corr:.3f}"

        feat_table.add_row(
            fl,
            f"{auc:.3f}",
            f"{best_acc:.1%}",
            f"{coverage:.0%}",
            "1.000" if fn == "f1_separation" else corr_str,
        )

    console.print(feat_table)

    # Confidence vs accuracy for current separation
    console.print("\n[bold]Separation confidence vs accuracy:[/bold]")
    conf_buckets = [(0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 1.0), (1.0, float("inf"))]
    for lo, hi in conf_buckets:
        bucket = [d for d in predicted if lo <= d.confidence < hi]
        if bucket:
            bc = sum(1 for d in bucket if d.correct)
            console.print(f"  conf [{lo:.1f}, {hi:.1f}): "
                          f"{bc}/{len(bucket)} = {bc / len(bucket):.0%}")

    # ── Section 4: Combined Model (LOO-video) ───────────────────────────

    console.print("\n[bold]═══ Section 4: Combined Model (LOO-video CV) ═══[/bold]\n")

    # All features
    combined = _loo_video_combined(diags, feature_names)
    console.print(f"  All 7 features:")
    for name, acc in combined.items():
        console.print(f"    {name}: {acc:.1%}")

    # Without court-space (higher coverage)
    no_court = [fn for fn in feature_names if fn != "f4_baseline_court"]
    combined_nc = _loo_video_combined(diags, no_court)
    console.print(f"\n  Without court-space feature (6 features):")
    for name, acc in combined_nc.items():
        console.print(f"    {name}: {acc:.1%}")

    # Top features only (separation + isolation + net_dist)
    top3 = ["f1_separation", "f2_isolation", "f5_net_dist"]
    combined_top = _loo_video_combined(diags, top3)
    console.print(f"\n  Top 3 features (separation + isolation + net_dist):")
    for name, acc in combined_top.items():
        console.print(f"    {name}: {acc:.1%}")

    # ── Section 5: Train Final Model and Extract Weights ────────────────

    console.print("\n[bold]═══ Section 5: Final Model Weights ═══[/bold]\n")

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    # Use all 7 features (all videos have calibration)
    all_feature_names = feature_names  # all 7
    X_all, y_all = [], []
    for d in diags:
        vals = [d.features.get(fn) for fn in all_feature_names]
        if any(v is None for v in vals):
            continue
        X_all.append(vals)
        y_all.append(1 if d.gt_physical == "near" else 0)

    X_arr = np.array(X_all, dtype=float)
    y_arr = np.array(y_all, dtype=int)
    console.print(f"  Training samples: {len(X_arr)} (skipped {len(diags) - len(X_arr)} "
                  f"with missing features)")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_arr)

    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_scaled, y_arr)

    train_acc = clf.score(X_scaled, y_arr)
    console.print(f"  Train accuracy: {train_acc:.1%}")

    # Print weights
    console.print(f"\n  Scaler means:  {list(np.round(scaler.mean_, 6))}")
    console.print(f"  Scaler scales: {list(np.round(scaler.scale_, 6))}")
    console.print(f"  Model coefs:   {list(np.round(clf.coef_[0], 6))}")
    console.print(f"  Model intercept: {float(clf.intercept_[0]):.6f}")

    # Feature importance (abs coefficient in standardized space)
    console.print(f"\n  Feature importance (|coef| in standardized space):")
    for fn, fl, c in sorted(
        zip(all_feature_names, feature_labels, clf.coef_[0]),
        key=lambda x: -abs(x[2]),
    ):
        console.print(f"    {fl:25s}: {c:+.4f}")

    # Convert to original-space weights (ready to paste into action_classifier.py)
    # w_orig[i] = w_std[i] / scale[i]
    # intercept_orig = intercept_std - sum(mean[i] * w_orig[i])
    _WEIGHT_KEY_MAP = {
        "f1_separation": "separation", "f2_isolation": "isolation",
        "f3_baseline_img": "baseline_img", "f4_baseline_court": "baseline_court",
        "f5_net_dist": "net_dist", "f6_ball_pos": "ball_pos",
        "f7_count_asym": "count_asym",
    }
    coefs_orig = clf.coef_[0] / scaler.scale_
    intercept_orig = float(clf.intercept_[0]) - float(np.sum(scaler.mean_ * coefs_orig))
    console.print("\n  [bold green]Original-space weights (copy to action_classifier.py):[/bold green]")
    console.print('  _FORMATION_WEIGHTS_7 = {')
    console.print(f'      "intercept": {intercept_orig:.8f},')
    for fn, w in zip(all_feature_names, coefs_orig):
        key = _WEIGHT_KEY_MAP[fn]
        console.print(f'      "{key}": {w:.8f},')
    console.print('  }')

    # LOO-video accuracy of this model (for comparison)
    console.print(f"\n  LOO-video CV accuracy:")
    loo_correct, loo_total = 0, 0
    videos = sorted(set(d.video_id for d in diags))
    for held_out in videos:
        train_d = [d for d in diags if d.video_id != held_out]
        test_d = [d for d in diags if d.video_id == held_out]

        X_tr, y_tr = [], []
        for d in train_d:
            vals = [d.features.get(fn) for fn in all_feature_names]
            if any(v is None for v in vals):
                continue
            X_tr.append(vals)
            y_tr.append(1 if d.gt_physical == "near" else 0)

        X_te, y_te = [], []
        for d in test_d:
            vals = [d.features.get(fn) for fn in all_feature_names]
            if any(v is None for v in vals):
                continue
            X_te.append(vals)
            y_te.append(1 if d.gt_physical == "near" else 0)

        if len(X_tr) < 5 or not X_te or len(set(y_tr)) < 2:
            continue

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(np.array(X_tr))
        X_te_s = sc.transform(np.array(X_te))
        m = LogisticRegression(max_iter=1000, C=1.0)
        m.fit(X_tr_s, y_tr)
        preds = m.predict(X_te_s)
        loo_correct += sum(int(p) == int(l) for p, l in zip(preds, y_te))
        loo_total += len(y_te)

    console.print(f"    {loo_correct}/{loo_total} = {loo_correct / loo_total:.1%}")

    # Also try without ball_pos (higher coverage)
    no_ball = [fn for fn in all_feature_names if fn != "f6_ball_pos"]
    X_nb, y_nb = [], []
    for d in diags:
        vals = [d.features.get(fn) for fn in no_ball]
        if any(v is None for v in vals):
            continue
        X_nb.append(vals)
        y_nb.append(1 if d.gt_physical == "near" else 0)

    console.print(f"\n  Without ball_pos (6 features, {len(X_nb)} samples):")
    sc_nb = StandardScaler()
    X_nb_s = sc_nb.fit_transform(np.array(X_nb))
    m_nb = LogisticRegression(max_iter=1000, C=1.0)
    m_nb.fit(X_nb_s, y_nb)
    console.print(f"    Train accuracy: {m_nb.score(X_nb_s, y_nb):.1%}")

    # LOO-video for no-ball
    loo_c2, loo_t2 = 0, 0
    for held_out in videos:
        train_d = [d for d in diags if d.video_id != held_out]
        test_d = [d for d in diags if d.video_id == held_out]

        X_tr, y_tr = [], []
        for d in train_d:
            vals = [d.features.get(fn) for fn in no_ball]
            if any(v is None for v in vals):
                continue
            X_tr.append(vals)
            y_tr.append(1 if d.gt_physical == "near" else 0)

        X_te, y_te = [], []
        for d in test_d:
            vals = [d.features.get(fn) for fn in no_ball]
            if any(v is None for v in vals):
                continue
            X_te.append(vals)
            y_te.append(1 if d.gt_physical == "near" else 0)

        if len(X_tr) < 5 or not X_te or len(set(y_tr)) < 2:
            continue

        sc2 = StandardScaler()
        X_tr_s = sc2.fit_transform(np.array(X_tr))
        X_te_s = sc2.transform(np.array(X_te))
        m2 = LogisticRegression(max_iter=1000, C=1.0)
        m2.fit(X_tr_s, y_tr)
        preds = m2.predict(X_te_s)
        loo_c2 += sum(int(p) == int(l) for p, l in zip(preds, y_te))
        loo_t2 += len(y_te)

    console.print(f"    LOO-video: {loo_c2}/{loo_t2} = {loo_c2 / loo_t2:.1%}")

    no_ball_labels = [
        "Vertical separation", "Server isolation", "Baseline prox (image)",
        "Baseline prox (court)", "Max net distance", "Player count asym",
    ]
    console.print(f"\n    Scaler means:  {list(np.round(sc_nb.mean_, 6))}")
    console.print(f"    Scaler scales: {list(np.round(sc_nb.scale_, 6))}")
    console.print(f"    Model coefs:   {list(np.round(m_nb.coef_[0], 6))}")
    console.print(f"    Model intercept: {float(m_nb.intercept_[0]):.6f}")
    console.print(f"\n    Feature importance:")
    for fl, c in sorted(
        zip(no_ball_labels, m_nb.coef_[0]),
        key=lambda x: -abs(x[1]),
    ):
        console.print(f"      {fl:25s}: {c:+.4f}")

    # Original-space weights for 6-feature model
    coefs_nb_orig = m_nb.coef_[0] / sc_nb.scale_
    intercept_nb_orig = float(m_nb.intercept_[0]) - float(
        np.sum(sc_nb.mean_ * coefs_nb_orig)
    )
    console.print("\n    [bold green]Original-space weights (copy to action_classifier.py):[/bold green]")
    console.print('    _FORMATION_WEIGHTS_6 = {')
    console.print(f'        "intercept": {intercept_nb_orig:.8f},')
    for fn, w in zip(no_ball, coefs_nb_orig):
        key = _WEIGHT_KEY_MAP[fn]
        console.print(f'        "{key}": {w:.8f},')
    console.print('    }')

    # ── Section 6: GT Physical Side Distribution ─────────────────────────

    console.print("\n[bold]═══ Section 6: GT Distribution ═══[/bold]\n")
    n_near = sum(1 for d in diags if d.gt_physical == "near")
    n_far = len(diags) - n_near
    console.print(f"  GT physical side: near={n_near}, far={n_far} "
                  f"(majority class = {max(n_near, n_far) / len(diags):.1%})")

    # Side-flipped distribution
    n_flipped = sum(1 for d in diags if d.side_flipped)
    console.print(f"  Side-flipped rallies: {n_flipped}/{len(diags)} "
                  f"({n_flipped / len(diags):.0%})")

    # ── Summary ──────────────────────────────────────────────────────────

    console.print("\n[bold]═══ Summary ═══[/bold]\n")

    if len(phys_wrong) == 0 and len(abstained) == 0:
        console.print("  Formation side is PERFECT (100% accuracy, 100% coverage).")
        console.print("  All score_accuracy errors come from team mapping (A/B assignment).")
    elif len(phys_wrong) == 0:
        console.print(f"  Formation side is perfect when it predicts "
                      f"({len(predicted)}/{len(diags)} coverage).")
        console.print(f"  {len(abstained)} abstentions need fallback improvement.")
    else:
        console.print(f"  Formation side accuracy: {phys_acc:.1%} "
                      f"({len(phys_wrong)} errors on {len(predicted)} predictions)")
        if len(phys_wrong) <= 5:
            console.print("  Very few errors — likely camera/tracking edge cases, "
                          "not a systematic heuristic failure.")
        else:
            best_feat = max(
                feature_names,
                key=lambda fn: _compute_auc(
                    [d.features[fn] for d in diags if d.features.get(fn) is not None],
                    [1 if d.gt_physical == "near" else 0
                     for d in diags if d.features.get(fn) is not None],
                ) if sum(1 for d in diags if d.features.get(fn) is not None) > 10
                else 0.5,
            )
            best_auc = _compute_auc(
                [d.features[best_feat] for d in diags
                 if d.features.get(best_feat) is not None],
                [1 if d.gt_physical == "near" else 0
                 for d in diags if d.features.get(best_feat) is not None],
            )
            console.print(f"  Best single feature: {best_feat} (AUC={best_auc:.3f})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
