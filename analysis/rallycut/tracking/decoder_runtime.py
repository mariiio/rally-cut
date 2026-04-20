"""Production wrapper around the Viterbi candidate decoder.

Runs the decoder over a single rally's candidates and returns the
accepted DecodedContact list. Pure function — no DB access, no video
reads. Callers assemble the inputs (ball_positions, player_positions,
sequence_probs, classifier) and consume the decoded list.

Shape mirrors the bypass logic in scripts/eval_candidate_decoder.py
so production and eval paths use the same code.
"""
from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    from scripts.eval_action_detection import RallyData

from rallycut.tracking.ball_tracker import BallPosition as BallPos
from rallycut.tracking.candidate_decoder import (
    ACTIONS,
    DecodedContact,
    TransitionMatrix,
    decode_rally,
    infer_team_from_player_track,
)
from rallycut.tracking.candidate_decoder import (
    CandidateFeatures as DecoderCandidateFeatures,
)
from rallycut.tracking.contact_classifier import ContactClassifier
from rallycut.tracking.contact_detector import ContactDetectionConfig
from rallycut.tracking.player_tracker import PlayerPosition as PlayerPos


def _build_candidate_features(
    ball_by_frame: dict[int, BallPos],
    players_by_frame: dict[int, list[PlayerPos]],
    candidate_frames: list[int],
    gbm_probs: np.ndarray,
    sequence_probs: np.ndarray | None,
) -> list[DecoderCandidateFeatures]:
    results: list[DecoderCandidateFeatures] = []
    for i, frame in enumerate(candidate_frames):
        team = -1
        ball = ball_by_frame.get(frame)
        if ball is not None:
            best = None
            best_d = 1e9
            for pp in players_by_frame.get(frame, []):
                d = (pp.x - ball.x) ** 2 + (pp.y - ball.y) ** 2
                if d < best_d:
                    best_d = d
                    best = pp
            if best is not None and 1 <= best.track_id <= 4:
                team = infer_team_from_player_track(best.track_id)

        if (
            sequence_probs is not None
            and sequence_probs.ndim == 2
            and sequence_probs.size > 0
        ):
            f = max(0, min(sequence_probs.shape[1] - 1, frame))
            p = sequence_probs[:, f]
            pos = p[1:]
            s = float(pos.sum())
            if s > 1e-6:
                action_probs = pos / s
            else:
                action_probs = np.ones(len(ACTIONS)) / len(ACTIONS)
        else:
            action_probs = np.ones(len(ACTIONS)) / len(ACTIONS)

        results.append(DecoderCandidateFeatures(
            frame=frame,
            gbm_contact_prob=float(gbm_probs[i]),
            action_probs=action_probs,
            team=team,
        ))
    return results


def run_decoder_over_rally(
    ball_positions: list[BallPos],
    player_positions: list[PlayerPos],
    sequence_probs: np.ndarray | None,
    classifier: ContactClassifier,
    contact_config: ContactDetectionConfig,
    gt_frames: list[int] | None = None,
    transitions: TransitionMatrix | None = None,
    skip_penalty: float = 1.0,
    min_accept_prob: float = 0.0,
) -> list[DecodedContact]:
    """Run candidate extraction → GBM scoring → Viterbi decode.

    Args:
        gt_frames: pass when available (training/eval) so frames_since_last
            uses GT-matched semantics. None in production.
        transitions: defaults to TransitionMatrix.default() (shipped JSON).
        skip_penalty: 1.0 is the decoder ship config from
            candidate_decoder_sweep_2026_04_20.md.
        min_accept_prob: minimum posterior for a candidate to be emitted
            by the decoder; 0.0 keeps the shipped behaviour.

    Returns:
        [] if extraction produces no candidates or classifier is untrained.
    """
    # Local imports to avoid circular deps at module load. Note: callers
    # must have CWD=analysis/ (or scripts/ on sys.path) — scripts/ is not
    # a packaged module. Task 5 CLI wiring must respect this.
    from rallycut.tracking.contact_detector import _RallyDataShim
    from scripts.train_contact_classifier import extract_candidate_features

    if classifier.model is None:
        return []
    if transitions is None:
        transitions = TransitionMatrix.default()

    rally_shim = _RallyDataShim.from_positions(
        ball_positions=ball_positions,
        player_positions=player_positions,
    )
    feats_list, cand_frames = extract_candidate_features(
        cast("RallyData", rally_shim),
        config=contact_config,
        gt_frames=gt_frames,
        sequence_probs=sequence_probs,
    )
    if not feats_list:
        return []
    x_mat = np.array([f.to_array() for f in feats_list], dtype=np.float64)
    expected = classifier.model.n_features_in_
    if x_mat.shape[1] > expected:
        x_mat = x_mat[:, :expected]
    elif x_mat.shape[1] < expected:
        pad = np.zeros((x_mat.shape[0], expected - x_mat.shape[1]))
        x_mat = np.hstack([x_mat, pad])
    gbm_probs = classifier.model.predict_proba(x_mat)[:, 1]

    ball_by_frame: dict[int, BallPos] = {}
    for bp in ball_positions:
        if bp.x > 0 or bp.y > 0:
            ball_by_frame[bp.frame_number] = bp
    players_by_frame: dict[int, list[PlayerPos]] = defaultdict(list)
    for pp in player_positions:
        players_by_frame[pp.frame_number].append(pp)

    cand_features = _build_candidate_features(
        ball_by_frame, players_by_frame, cand_frames, gbm_probs, sequence_probs,
    )
    return decode_rally(
        cand_features, transitions,
        skip_penalty=skip_penalty,
        min_accept_prob=min_accept_prob,
    )
