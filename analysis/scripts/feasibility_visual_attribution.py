"""Feasibility test: can we visually distinguish the acting player from teammates?

Tests whether simple bbox geometry (vertical displacement, aspect ratio change,
height change, horizontal movement) discriminates the GT player from same-side
teammates at contact frames. This is the gate before building a full VideoMAE-based
per-player action classifier.

Usage:
    cd analysis
    uv run python scripts/feasibility_visual_attribution.py
    uv run python scripts/feasibility_visual_attribution.py --action attack
    uv run python scripts/feasibility_visual_attribution.py --verbose
    uv run python scripts/feasibility_visual_attribution.py --loocv
    uv run python scripts/feasibility_visual_attribution.py --visual   # VideoMAE features (GPU)
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.table import Table

from rallycut.evaluation.db import get_connection
from rallycut.tracking.match_tracker import build_match_team_assignments

console = Console()


# ---------------------------------------------------------------------------
# Data loading (reuses pattern from eval_action_detection.py)
# ---------------------------------------------------------------------------


@dataclass
class GtContact:
    rally_id: str
    video_id: str
    frame: int
    action: str
    player_track_id: int
    positions_json: list[dict[str, Any]]
    court_split_y: float | None
    start_ms: int


def load_gt_contacts() -> list[GtContact]:
    """Load GT contacts with player positions from the database."""
    query = """
        SELECT
            r.id AS rally_id,
            r.video_id,
            pt.action_ground_truth_json,
            pt.positions_json,
            pt.court_split_y,
            r.start_ms
        FROM rallies r
        JOIN player_tracks pt ON pt.rally_id = r.id
        WHERE pt.action_ground_truth_json IS NOT NULL
          AND pt.positions_json IS NOT NULL
        ORDER BY r.video_id, r.start_ms
    """

    contacts: list[GtContact] = []
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            for row in cur.fetchall():
                rally_id, video_id, gt_json, pos_json, split_y, start_ms = row
                if not gt_json or not pos_json:
                    continue
                for label in gt_json:
                    tid = label.get("playerTrackId", -1)
                    if tid < 0:
                        continue
                    contacts.append(GtContact(
                        rally_id=rally_id,
                        video_id=video_id,
                        frame=label["frame"],
                        action=label["action"],
                        player_track_id=tid,
                        positions_json=pos_json,
                        court_split_y=split_y,
                        start_ms=start_ms or 0,
                    ))
    return contacts


def load_team_assignments() -> dict[str, dict[int, int]]:
    """Load match-level team assignments (rally_id -> {track_id: team})."""
    query = """
        SELECT id, match_analysis_json
        FROM videos
        WHERE match_analysis_json IS NOT NULL
    """
    result: dict[str, dict[int, int]] = {}
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            for _vid, ma_json in cur.fetchall():
                if isinstance(ma_json, dict):
                    result.update(build_match_team_assignments(ma_json, 0.0))
    return result


# ---------------------------------------------------------------------------
# Position helpers
# ---------------------------------------------------------------------------


@dataclass
class PlayerPos:
    frame: int
    x: float  # center, normalized 0-1
    y: float
    width: float
    height: float


def build_pos_index(
    positions_json: list[dict[str, Any]],
) -> dict[int, dict[int, PlayerPos]]:
    """Build frame -> {track_id: PlayerPos} index."""
    index: dict[int, dict[int, PlayerPos]] = defaultdict(dict)
    for p in positions_json:
        fn = p.get("frameNumber")
        tid = p.get("trackId")
        if fn is None or tid is None:
            continue
        index[fn][tid] = PlayerPos(
            frame=fn,
            x=p.get("x", 0.0),
            y=p.get("y", 0.0),
            width=p.get("width", 0.0),
            height=p.get("height", 0.0),
        )
    return index


def get_same_side_tracks(
    pos_index: dict[int, dict[int, PlayerPos]],
    gt_track_id: int,
    contact_frame: int,
    team_assignments: dict[int, int] | None,
    court_split_y: float | None,
) -> list[int]:
    """Get track IDs of same-side teammates at the contact frame.

    Uses team assignments if available, else falls back to court_split_y.
    """
    frame_tracks = pos_index.get(contact_frame, {})
    if gt_track_id not in frame_tracks:
        return []

    gt_team = None
    if team_assignments and gt_track_id in team_assignments:
        gt_team = team_assignments[gt_track_id]

    same_side: list[int] = []
    for tid, pos in frame_tracks.items():
        if tid == gt_track_id:
            continue

        if gt_team is not None and team_assignments:
            if team_assignments.get(tid) == gt_team:
                same_side.append(tid)
        elif court_split_y is not None:
            gt_pos = frame_tracks[gt_track_id]
            if (gt_pos.y < court_split_y) == (pos.y < court_split_y):
                same_side.append(tid)

    return same_side


# ---------------------------------------------------------------------------
# Geometric features
# ---------------------------------------------------------------------------


@dataclass
class GeomFeatures:
    """Geometric features for a single player at a contact frame."""
    track_id: int
    is_gt: bool
    vertical_disp: float | None = None  # y(F) - y(F-10), negative = up
    aspect_ratio_change: float | None = None  # (h/w at F) / (h/w at F-10)
    height_change: float | None = None  # height(F) / height(F-10)
    horiz_disp: float | None = None  # |x(F) - x(F-10)|


def compute_geom_features(
    pos_index: dict[int, dict[int, PlayerPos]],
    track_id: int,
    contact_frame: int,
    is_gt: bool,
    lookback: int = 10,
) -> GeomFeatures:
    """Compute geometric features for a player relative to a contact frame."""
    feats = GeomFeatures(track_id=track_id, is_gt=is_gt)

    # Find position at contact frame
    pos_now = pos_index.get(contact_frame, {}).get(track_id)
    if pos_now is None:
        return feats

    # Find position at lookback frame (search ±2 frames for tolerance)
    ref_frame = contact_frame - lookback
    pos_ref = None
    for f in range(ref_frame - 2, ref_frame + 3):
        pos_ref = pos_index.get(f, {}).get(track_id)
        if pos_ref is not None:
            break

    if pos_ref is None:
        return feats

    # Vertical displacement (negative = upward in image coords)
    feats.vertical_disp = pos_now.y - pos_ref.y

    # Aspect ratio change
    ar_now = pos_now.height / max(pos_now.width, 1e-6)
    ar_ref = pos_ref.height / max(pos_ref.width, 1e-6)
    feats.aspect_ratio_change = ar_now / max(ar_ref, 1e-6)

    # Height change
    feats.height_change = pos_now.height / max(pos_ref.height, 1e-6)

    # Horizontal displacement
    feats.horiz_disp = abs(pos_now.x - pos_ref.x)

    return feats


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


@dataclass
class ContactResult:
    rally_id: str
    video_id: str
    frame: int
    action: str
    gt_track_id: int
    n_candidates: int
    # Per-feature: did GT player rank #1?
    vert_correct: bool | None = None
    aspect_correct: bool | None = None
    height_correct: bool | None = None
    horiz_correct: bool | None = None
    combined_correct: bool | None = None  # any-feature vote


def analyze_contact(
    contact: GtContact,
    team_assignments: dict[int, int] | None,
) -> ContactResult | None:
    """Analyze a single contact: compute features for GT + teammates, check ranking."""
    pos_index = build_pos_index(contact.positions_json)

    same_side = get_same_side_tracks(
        pos_index, contact.player_track_id, contact.frame,
        team_assignments, contact.court_split_y,
    )

    if not same_side:
        return None  # No same-side teammate = proximity always correct

    result = ContactResult(
        rally_id=contact.rally_id,
        video_id=contact.video_id,
        frame=contact.frame,
        action=contact.action,
        gt_track_id=contact.player_track_id,
        n_candidates=len(same_side) + 1,
    )

    # Compute features for GT player and all same-side teammates
    gt_feats = compute_geom_features(
        pos_index, contact.player_track_id, contact.frame, is_gt=True,
    )
    teammate_feats = [
        compute_geom_features(pos_index, tid, contact.frame, is_gt=False)
        for tid in same_side
    ]

    # For each feature, check if GT player has the most extreme value
    # (most movement = most likely acting)
    all_feats = [gt_feats] + teammate_feats

    def _rank_by(
        feats_list: list[GeomFeatures],
        key_fn: Any,
        reverse: bool = True,
    ) -> bool | None:
        """Check if GT player ranks #1 by key_fn. Returns None if data missing."""
        vals = []
        for f in feats_list:
            v = key_fn(f)
            if v is None:
                return None
            vals.append((v, f.is_gt))
        if not vals:
            return None
        vals.sort(key=lambda x: x[0], reverse=reverse)
        return vals[0][1]  # Is the top-ranked player the GT?

    # Vertical displacement: most negative = biggest jump upward
    result.vert_correct = _rank_by(
        all_feats, lambda f: f.vertical_disp, reverse=False,
    )

    # Aspect ratio change: highest = most change (jumping stretches bbox)
    result.aspect_correct = _rank_by(
        all_feats, lambda f: f.aspect_ratio_change, reverse=True,
    )

    # Height change: highest = most growth
    result.height_correct = _rank_by(
        all_feats, lambda f: f.height_change, reverse=True,
    )

    # Horizontal displacement: most horizontal movement
    result.horiz_correct = _rank_by(
        all_feats, lambda f: f.horiz_disp, reverse=True,
    )

    # Combined: GT ranks #1 on ANY feature
    checks = [result.vert_correct, result.aspect_correct,
              result.height_correct, result.horiz_correct]
    valid_checks = [c for c in checks if c is not None]
    result.combined_correct = any(valid_checks) if valid_checks else None

    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def report_results(results: list[ContactResult], verbose: bool = False) -> None:
    """Print summary tables."""
    if not results:
        console.print("[red]No evaluable contacts found.[/red]")
        return

    console.print(f"\n[bold]Feasibility: Geometric Feature Discrimination[/bold]")
    console.print(f"Evaluable contacts (with same-side teammate): {len(results)}\n")

    # Overall accuracy per feature
    features = [
        ("Vertical disp (jump)", "vert_correct"),
        ("Aspect ratio change", "aspect_correct"),
        ("Height change", "height_correct"),
        ("Horizontal disp", "horiz_correct"),
        ("Any feature (OR)", "combined_correct"),
    ]

    overall_table = Table(title="Overall Discrimination Rate")
    overall_table.add_column("Feature", style="cyan")
    overall_table.add_column("Correct", justify="right")
    overall_table.add_column("Evaluable", justify="right")
    overall_table.add_column("Rate", justify="right", style="bold")

    for name, attr in features:
        vals = [getattr(r, attr) for r in results if getattr(r, attr) is not None]
        n_correct = sum(vals)
        n_total = len(vals)
        rate = f"{100 * n_correct / n_total:.1f}%" if n_total > 0 else "N/A"
        overall_table.add_row(name, str(n_correct), str(n_total), rate)

    console.print(overall_table)

    # Per-action breakdown
    action_types = ["serve", "receive", "set", "attack", "dig", "block"]
    by_action: dict[str, list[ContactResult]] = defaultdict(list)
    for r in results:
        by_action[r.action].append(r)

    for feat_name, feat_attr in features:
        table = Table(title=f"Per-Action: {feat_name}")
        table.add_column("Action", style="cyan")
        table.add_column("Correct", justify="right")
        table.add_column("Total", justify="right")
        table.add_column("Rate", justify="right", style="bold")

        for act in action_types:
            act_results = by_action.get(act, [])
            vals = [getattr(r, feat_attr) for r in act_results
                    if getattr(r, feat_attr) is not None]
            if not vals:
                continue
            n_correct = sum(vals)
            n_total = len(vals)
            rate = f"{100 * n_correct / n_total:.1f}%"
            table.add_row(act, str(n_correct), str(n_total), rate)

        console.print(table)

    # Proximity baseline comparison
    console.print("\n[bold]Context:[/bold]")
    console.print("  Proximity baseline (current): 67.8% overall attribution")
    console.print("  This test: among contacts WITH a same-side teammate,")
    console.print("  does the GT player have the most extreme geometric signal?")
    console.print("  (Contacts without teammates are trivially correct.)\n")

    if verbose:
        # Per-contact details for wrong cases
        wrong = [r for r in results if r.combined_correct is False]
        if wrong:
            console.print(f"\n[yellow]Contacts where NO feature discriminates "
                          f"GT player ({len(wrong)}):[/yellow]")
            det_table = Table()
            det_table.add_column("Rally")
            det_table.add_column("Frame")
            det_table.add_column("Action")
            det_table.add_column("GT Track")
            det_table.add_column("Candidates")
            for r in wrong[:30]:
                det_table.add_row(
                    r.rally_id[:8], str(r.frame), r.action,
                    str(r.gt_track_id), str(r.n_candidates),
                )
            console.print(det_table)
            if len(wrong) > 30:
                console.print(f"  ... and {len(wrong) - 30} more")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_loocv_classifier(contacts: list[GtContact],
                         all_teams: dict[str, dict[int, int]]) -> None:
    """Train LOO-CV logistic regression on geometric features.

    For each contact, creates one sample per candidate (GT + same-side teammates).
    Uses LOO-CV by video to measure attribution accuracy.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    # Build dataset: one row per (contact, candidate) pair
    @dataclass
    class Sample:
        video_id: str
        contact_idx: int  # groups samples from same contact
        features: list[float]  # [vert_disp, aspect_change, height_change, horiz_disp]
        is_gt: bool

    samples: list[Sample] = []
    contact_idx = 0

    for contact in contacts:
        teams = all_teams.get(contact.rally_id)
        pos_index = build_pos_index(contact.positions_json)

        same_side = get_same_side_tracks(
            pos_index, contact.player_track_id, contact.frame,
            teams, contact.court_split_y,
        )
        if not same_side:
            continue

        all_track_ids = [contact.player_track_id] + same_side

        # Check all candidates have features
        candidate_feats = []
        all_valid = True
        for tid in all_track_ids:
            is_gt = (tid == contact.player_track_id)
            gf = compute_geom_features(pos_index, tid, contact.frame, is_gt)
            if gf.vertical_disp is None:
                all_valid = False
                break
            candidate_feats.append((tid, is_gt, gf))

        if not all_valid or len(candidate_feats) < 2:
            continue

        for tid, is_gt, gf in candidate_feats:
            assert gf.vertical_disp is not None
            assert gf.aspect_ratio_change is not None
            assert gf.height_change is not None
            assert gf.horiz_disp is not None
            samples.append(Sample(
                video_id=contact.video_id,
                contact_idx=contact_idx,
                features=[
                    gf.vertical_disp,
                    gf.aspect_ratio_change,
                    gf.height_change,
                    gf.horiz_disp,
                ],
                is_gt=is_gt,
            ))

        contact_idx += 1

    if not samples:
        console.print("[red]No samples for LOO-CV[/red]")
        return

    # Group by video for LOO-CV
    video_ids = sorted(set(s.video_id for s in samples))
    console.print(f"\n[bold]LOO-CV Logistic Regression on {contact_idx} contacts "
                  f"({len(samples)} samples, {len(video_ids)} videos)[/bold]")

    # LOO-CV by video
    correct = 0
    total = 0
    per_action_correct: dict[str, int] = defaultdict(int)
    per_action_total: dict[str, int] = defaultdict(int)

    # We need action info per contact_idx - rebuild quickly
    contact_actions: dict[int, str] = {}
    cidx = 0
    for contact in contacts:
        teams = all_teams.get(contact.rally_id)
        pos_index = build_pos_index(contact.positions_json)
        same_side = get_same_side_tracks(
            pos_index, contact.player_track_id, contact.frame,
            teams, contact.court_split_y,
        )
        if not same_side:
            continue
        # Check validity (same logic as above)
        all_valid = True
        for tid in [contact.player_track_id] + same_side:
            gf = compute_geom_features(pos_index, tid, contact.frame, True)
            if gf.vertical_disp is None:
                all_valid = False
                break
        if not all_valid or len(same_side) + 1 < 2:
            continue
        contact_actions[cidx] = contact.action
        cidx += 1

    for held_out_vid in video_ids:
        train_samples = [s for s in samples if s.video_id != held_out_vid]
        test_samples = [s for s in samples if s.video_id == held_out_vid]

        if not train_samples or not test_samples:
            continue

        X_train = np.array([s.features for s in train_samples])
        y_train = np.array([1 if s.is_gt else 0 for s in train_samples])

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        clf.fit(X_train_scaled, y_train)

        X_test = np.array([s.features for s in test_samples])
        X_test_scaled = scaler.transform(X_test)
        probs = clf.predict_proba(X_test_scaled)[:, 1]

        # Group test samples by contact_idx, pick candidate with highest prob
        test_by_contact: dict[int, list[tuple[float, bool]]] = defaultdict(list)
        for s, p in zip(test_samples, probs):
            test_by_contact[s.contact_idx].append((p, s.is_gt))

        for cidx_val, candidates in test_by_contact.items():
            candidates.sort(key=lambda x: x[0], reverse=True)
            picked_is_gt = candidates[0][1]
            total += 1
            action = contact_actions.get(cidx_val, "unknown")
            per_action_total[action] += 1
            if picked_is_gt:
                correct += 1
                per_action_correct[action] += 1

    rate = 100 * correct / total if total > 0 else 0
    console.print(f"\n  [bold green]LOO-CV Attribution Accuracy: {correct}/{total} "
                  f"= {rate:.1f}%[/bold green]")
    console.print(f"  (vs 67.8% proximity baseline)\n")

    # Per-action breakdown
    table = Table(title="LOO-CV per Action Type")
    table.add_column("Action", style="cyan")
    table.add_column("Correct", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Rate", justify="right", style="bold")

    for act in ["serve", "receive", "set", "attack", "dig", "block"]:
        n = per_action_total.get(act, 0)
        c = per_action_correct.get(act, 0)
        if n == 0:
            continue
        table.add_row(act, str(c), str(n), f"{100 * c / n:.1f}%")

    console.print(table)


def run_visual_loocv(contacts: list[GtContact],
                     all_teams: dict[str, dict[int, int]]) -> None:
    """Extract VideoMAE features from player crops and run LOO-CV.

    For each contact, extracts 16-frame clips of each same-side candidate,
    runs through frozen VideoMAE encoder for 768-dim features, and trains
    LOO-CV logistic regression for binary acting/not-acting classification.
    """
    import cv2
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from rallycut.evaluation.tracking.db import get_video_path

    @dataclass
    class VisualSample:
        video_id: str
        contact_idx: int
        features: np.ndarray | None  # (768,) VideoMAE CLS features
        geom_features: list[float]  # [vert_disp, ar_change, h_change, horiz_disp]
        is_gt: bool
        action: str

    # Temporal window: 16 frames ending 1 frame after contact
    WINDOW_BEFORE = 14
    WINDOW_AFTER = 1
    CROP_PAD = 0.2  # 20% padding around bbox

    # Group contacts by video for efficient video access
    contacts_by_video: dict[str, list[GtContact]] = defaultdict(list)
    for c in contacts:
        contacts_by_video[c.video_id].append(c)

    console.print(f"\n[bold]VideoMAE Feature Extraction[/bold]")
    console.print(f"  {len(contacts)} contacts across {len(contacts_by_video)} videos")

    # Load VideoMAE model
    from lib.volleyball_ml.video_mae import GameStateClassifier
    model = GameStateClassifier()

    samples: list[VisualSample] = []
    contact_idx = 0
    n_skipped_no_teammate = 0
    n_skipped_no_video = 0
    n_skipped_missing_frames = 0

    video_ids_sorted = sorted(contacts_by_video.keys())
    for vid_num, video_id in enumerate(video_ids_sorted):
        vid_contacts = contacts_by_video[video_id]

        video_path = get_video_path(video_id)
        if video_path is None:
            n_skipped_no_video += len(vid_contacts)
            continue

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            n_skipped_no_video += len(vid_contacts)
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        try:
            for contact in vid_contacts:
                teams = all_teams.get(contact.rally_id)
                pos_index = build_pos_index(contact.positions_json)

                same_side = get_same_side_tracks(
                    pos_index, contact.player_track_id, contact.frame,
                    teams, contact.court_split_y,
                )
                if not same_side:
                    n_skipped_no_teammate += 1
                    continue

                all_track_ids = [contact.player_track_id] + same_side
                rally_start_frame = int(contact.start_ms / 1000.0 * fps)

                # For each candidate, extract 16-frame crop clip
                candidate_clips: list[tuple[int, bool, list[np.ndarray], list[float]]] = []

                for tid in all_track_ids:
                    is_gt = (tid == contact.player_track_id)
                    frames_clip: list[np.ndarray] = []

                    # Compute geometric features too
                    gf = compute_geom_features(pos_index, tid, contact.frame, is_gt)
                    geom = [
                        gf.vertical_disp or 0.0,
                        gf.aspect_ratio_change or 1.0,
                        gf.height_change or 1.0,
                        gf.horiz_disp or 0.0,
                    ]

                    for offset in range(-WINDOW_BEFORE, WINDOW_AFTER + 1):
                        rel_frame = contact.frame + offset

                        # Find player position (search ±1 frame tolerance)
                        pos = None
                        for f_search in [rel_frame, rel_frame - 1, rel_frame + 1]:
                            pos = pos_index.get(f_search, {}).get(tid)
                            if pos is not None:
                                break

                        if pos is None:
                            # Interpolate from nearest known positions
                            # Find closest before and after
                            before_pos = None
                            after_pos = None
                            for delta in range(1, 10):
                                if before_pos is None:
                                    before_pos = pos_index.get(
                                        rel_frame - delta, {},
                                    ).get(tid)
                                if after_pos is None:
                                    after_pos = pos_index.get(
                                        rel_frame + delta, {},
                                    ).get(tid)
                                if before_pos and after_pos:
                                    break

                            if before_pos is not None:
                                pos = before_pos
                            elif after_pos is not None:
                                pos = after_pos
                            else:
                                break  # Can't find this player at all

                        # Read video frame and extract crop
                        abs_frame = rally_start_frame + rel_frame
                        cap.set(cv2.CAP_PROP_POS_FRAMES, abs_frame)
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            break

                        # Crop with padding
                        cx, cy = pos.x * frame_w, pos.y * frame_h
                        bw, bh = pos.width * frame_w, pos.height * frame_h
                        pad_w, pad_h = bw * CROP_PAD, bh * CROP_PAD

                        x1 = max(0, int(cx - bw / 2 - pad_w))
                        y1 = max(0, int(cy - bh / 2 - pad_h))
                        x2 = min(frame_w, int(cx + bw / 2 + pad_w))
                        y2 = min(frame_h, int(cy + bh / 2 + pad_h))

                        if x2 <= x1 or y2 <= y1:
                            break

                        crop = frame[y1:y2, x1:x2]
                        # Resize to 224x224 for VideoMAE
                        crop_resized = cv2.resize(crop, (224, 224))
                        frames_clip.append(crop_resized)

                    if len(frames_clip) == 16:
                        candidate_clips.append((tid, is_gt, frames_clip, geom))

                if len(candidate_clips) < 2:
                    n_skipped_missing_frames += 1
                    continue

                # Extract VideoMAE features for all candidate clips in batch
                batch_frames = [clip for _, _, clip, _ in candidate_clips]
                try:
                    features_batch = model.get_encoder_features_batch(
                        batch_frames, pooling="cls",
                    )
                except Exception as e:
                    console.print(f"  [red]VideoMAE error: {e}[/red]")
                    n_skipped_missing_frames += 1
                    continue

                for i, (tid, is_gt, _clip, geom) in enumerate(candidate_clips):
                    samples.append(VisualSample(
                        video_id=video_id,
                        contact_idx=contact_idx,
                        features=features_batch[i],
                        geom_features=geom,
                        is_gt=is_gt,
                        action=contact.action,
                    ))

                contact_idx += 1

        finally:
            cap.release()

        console.print(
            f"  [{vid_num + 1}/{len(video_ids_sorted)}] {video_id[:8]}: "
            f"{contact_idx} contacts extracted so far",
        )

    console.print(f"\n  Total: {contact_idx} contacts, {len(samples)} samples")
    console.print(f"  Skipped: no teammate={n_skipped_no_teammate}, "
                  f"no video={n_skipped_no_video}, "
                  f"missing frames={n_skipped_missing_frames}")

    if contact_idx == 0:
        console.print("[red]No contacts extracted. Cannot run LOO-CV.[/red]")
        return

    import numpy as np

    # LOO-CV by video
    video_ids_list = sorted(set(s.video_id for s in samples))
    console.print(f"\n[bold]LOO-CV: VideoMAE features ({len(video_ids_list)} videos)[/bold]")

    configs = [
        ("VideoMAE only", True, False),
        ("Geom only", False, True),
        ("VideoMAE + Geom", True, True),
    ]

    for config_name, use_visual, use_geom in configs:
        correct = 0
        total = 0
        per_action_correct: dict[str, int] = defaultdict(int)
        per_action_total: dict[str, int] = defaultdict(int)

        for held_out_vid in video_ids_list:
            train = [s for s in samples if s.video_id != held_out_vid]
            test = [s for s in samples if s.video_id == held_out_vid]
            if not train or not test:
                continue

            # Build feature vectors
            def _build_features(sample_list: list[VisualSample]) -> np.ndarray:
                parts = []
                for s in sample_list:
                    feat_parts: list[np.ndarray] = []
                    if use_visual and s.features is not None:
                        feat_parts.append(s.features)
                    if use_geom:
                        feat_parts.append(np.array(s.geom_features))
                    parts.append(np.concatenate(feat_parts))
                return np.array(parts)

            X_train = _build_features(train)
            y_train = np.array([1 if s.is_gt else 0 for s in train])
            X_test = _build_features(test)

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
            clf.fit(X_train_s, y_train)
            probs = clf.predict_proba(X_test_s)[:, 1]

            # Pick highest-scoring candidate per contact
            test_by_contact: dict[int, list[tuple[float, bool, str]]] = defaultdict(list)
            for s, p in zip(test, probs):
                test_by_contact[s.contact_idx].append((p, s.is_gt, s.action))

            for cidx_val, candidates in test_by_contact.items():
                candidates.sort(key=lambda x: x[0], reverse=True)
                picked_is_gt = candidates[0][1]
                action = candidates[0][2]
                total += 1
                per_action_total[action] += 1
                if picked_is_gt:
                    correct += 1
                    per_action_correct[action] += 1

        rate = 100 * correct / total if total > 0 else 0
        console.print(f"\n  [bold]{config_name}[/bold]: {correct}/{total} = "
                      f"[bold green]{rate:.1f}%[/bold green]")

        table = Table(title=f"{config_name} per Action")
        table.add_column("Action", style="cyan")
        table.add_column("Correct", justify="right")
        table.add_column("Total", justify="right")
        table.add_column("Rate", justify="right", style="bold")

        for act in ["serve", "receive", "set", "attack", "dig", "block"]:
            n = per_action_total.get(act, 0)
            c = per_action_correct.get(act, 0)
            if n == 0:
                continue
            table.add_row(act, str(c), str(n), f"{100 * c / n:.1f}%")

        console.print(table)

    console.print(f"\n  [dim]Proximity baseline: 67.8%[/dim]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", help="Filter to specific action type")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show per-contact details for failures")
    parser.add_argument("--loocv", action="store_true",
                        help="Run LOO-CV logistic regression on geometric features")
    parser.add_argument("--visual", action="store_true",
                        help="Run VideoMAE feature extraction + LOO-CV (requires GPU/video)")
    args = parser.parse_args()

    console.print("[bold]Loading GT contacts from database...[/bold]")
    contacts = load_gt_contacts()
    console.print(f"  Loaded {len(contacts)} GT contacts")

    if args.action:
        contacts = [c for c in contacts if c.action == args.action]
        console.print(f"  Filtered to {len(contacts)} '{args.action}' contacts")

    console.print("[bold]Loading team assignments...[/bold]")
    all_teams = load_team_assignments()
    console.print(f"  Loaded teams for {len(all_teams)} rallies")

    console.print("[bold]Analyzing geometric features...[/bold]")
    results: list[ContactResult] = []
    skipped_no_teammate = 0
    skipped_no_data = 0

    for i, contact in enumerate(contacts):
        teams = all_teams.get(contact.rally_id)
        result = analyze_contact(contact, teams)
        if result is None:
            skipped_no_teammate += 1
        else:
            results.append(result)

        if (i + 1) % 100 == 0:
            console.print(f"  [{i + 1}/{len(contacts)}] processed")

    console.print(f"  Evaluable: {len(results)}, "
                  f"skipped (no teammate): {skipped_no_teammate}")

    report_results(results, verbose=args.verbose)

    if args.loocv:
        run_loocv_classifier(contacts, all_teams)

    if args.visual:
        run_visual_loocv(contacts, all_teams)


if __name__ == "__main__":
    main()
