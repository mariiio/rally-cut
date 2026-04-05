"""Contact-anchored sequence decoder for action detection.

Merges high-precision trajectory contact proposals (90% precision) with
high-recall MS-TCN++ sequence model predictions to produce discrete
action events. Uses constrained Viterbi decoding to enforce volleyball
rules on the action sequence.

Pipeline:
    1. MS-TCN++ → per-frame class probabilities P(class, t)
    2. Contact score S(t) = 1 - P(background, t)
    3. Trajectory proposals (from detect_contacts) boost S(t) at their frames
    4. Peaks in S(t) → merged contact proposals
    5. NMS on merged proposals
    6. Action type = argmax P(1:6, t*) at each proposal
    7. Constrained Viterbi decoding enforces volleyball transition rules
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from rallycut.actions.trajectory_features import ACTION_TYPES, NUM_CLASSES
from rallycut.tracking.action_classifier import ActionType, ClassifiedAction
from rallycut.tracking.contact_detector import Contact, ContactSequence

# Transition probabilities for constrained Viterbi (same structure as action_classifier)
# Maps (prev_action, curr_action) → probability
_TRANSITIONS: dict[tuple[str, str], float] = {
    # After serve: receive or dig on opposite side
    ("serve", "receive"): 0.80,
    ("serve", "dig"): 0.12,
    ("serve", "block"): 0.05,
    ("serve", "attack"): 0.03,
    # After receive: set or attack (same side)
    ("receive", "set"): 0.78,
    ("receive", "attack"): 0.15,
    ("receive", "dig"): 0.05,
    ("receive", "receive"): 0.02,
    # After set: attack (same side)
    ("set", "attack"): 0.88,
    ("set", "set"): 0.05,
    ("set", "dig"): 0.05,
    ("set", "receive"): 0.02,
    # After attack: dig, block, or set on opposite side
    ("attack", "dig"): 0.45,
    ("attack", "block"): 0.20,
    ("attack", "set"): 0.12,
    ("attack", "attack"): 0.10,
    ("attack", "receive"): 0.08,
    ("attack", "serve"): 0.05,
    # After block: dig/set/attack (ball could go anywhere)
    ("block", "dig"): 0.40,
    ("block", "set"): 0.20,
    ("block", "attack"): 0.20,
    ("block", "block"): 0.05,
    ("block", "receive"): 0.10,
    ("block", "serve"): 0.05,
    # After dig: set or attack (same side)
    ("dig", "set"): 0.60,
    ("dig", "attack"): 0.25,
    ("dig", "dig"): 0.10,
    ("dig", "receive"): 0.05,
}
_MIN_TRANSITION_PROB = 0.001


@dataclass
class DecoderConfig:
    """Configuration for contact-anchored decoder."""

    # Contact score parameters
    anchor_bonus: float = 0.3  # Bonus added to S(t) at trajectory contact frames
    sequence_peak_threshold: float = 0.5  # Min S(t) for sequence-only proposals
    smoothing_sigma: float = 2.0  # Gaussian smoothing sigma for contact score

    # NMS parameters
    min_peak_distance: int = 12  # Min frames between contacts (~0.4s @ 30fps)

    # Viterbi parameters
    viterbi_transition_weight: float = 1.0  # Weight for transition log-probs
    viterbi_emission_weight: float = 1.0  # Weight for emission log-probs

    # Serve detection
    serve_window_frames: int = 90  # First N frames to search for serve (3s @ 30fps)
    serve_min_confidence: float = 0.3  # Min confidence for serve detection


@dataclass
class DecodedAction:
    """A decoded action event from the contact-anchored decoder."""

    frame: int
    action: str  # serve/receive/set/attack/dig/block
    confidence: float  # Sequence model confidence for this action
    contact_score: float  # Merged contact score S(t)
    has_trajectory_anchor: bool  # Whether a trajectory contact was at this frame
    player_track_id: int = -1  # Player attribution (filled later)
    player_candidates: list[tuple[int, float]] = field(default_factory=list)
    ball_x: float = 0.0
    ball_y: float = 0.0
    velocity: float = 0.0
    court_side: str = "unknown"

    def to_classified_action(self) -> ClassifiedAction:
        """Convert to ClassifiedAction for compatibility with existing pipeline."""
        action_type = ActionType(self.action) if self.action != "unknown" else ActionType.UNKNOWN
        return ClassifiedAction(
            action_type=action_type,
            frame=self.frame,
            ball_x=self.ball_x,
            ball_y=self.ball_y,
            velocity=self.velocity,
            player_track_id=self.player_track_id,
            court_side=self.court_side,
            confidence=self.confidence,
            is_synthetic=False,
        )


def decode_actions(
    frame_probs: np.ndarray,
    contact_sequence: ContactSequence | None = None,
    *,
    config: DecoderConfig | None = None,
    net_y: float = 0.5,
    ball_positions_by_frame: dict[int, tuple[float, float]] | None = None,
) -> list[DecodedAction]:
    """Decode discrete action events from sequence model probabilities.

    Args:
        frame_probs: (NUM_CLASSES, T) per-frame class probabilities from MS-TCN++.
            Row 0 = background, rows 1-6 = action classes.
        contact_sequence: Optional trajectory contact proposals for anchoring.
        config: Decoder configuration.
        net_y: Net Y position for court side estimation.
        ball_positions_by_frame: Optional {frame: (x, y)} for ball position lookup.

    Returns:
        List of decoded action events, sorted by frame.
    """
    if config is None:
        config = DecoderConfig()

    num_classes, n_frames = frame_probs.shape
    assert num_classes == NUM_CLASSES, f"Expected {NUM_CLASSES} classes, got {num_classes}"

    # Step 1: Compute contact score S(t) = 1 - P(background, t)
    contact_score = 1.0 - frame_probs[0]  # (n_frames,)

    # Step 2: Build trajectory anchor set
    anchor_frames: set[int] = set()
    anchor_contacts: dict[int, Contact] = {}
    if contact_sequence is not None:
        for c in contact_sequence.contacts:
            if 0 <= c.frame < n_frames:
                anchor_frames.add(c.frame)
                anchor_contacts[c.frame] = c
                # Boost contact score at anchor frames
                contact_score[c.frame] = min(1.0, contact_score[c.frame] + config.anchor_bonus)

    # Step 3: Smooth contact score and find peaks
    smoothed_score = gaussian_filter1d(contact_score, sigma=config.smoothing_sigma)

    # Find peaks in smoothed score
    peaks, properties = find_peaks(
        smoothed_score,
        height=config.sequence_peak_threshold,
        distance=config.min_peak_distance,
    )

    # Also include all trajectory anchors that didn't make it as peaks
    all_proposal_frames = set(peaks.tolist())
    for af in anchor_frames:
        all_proposal_frames.add(af)

    # Step 4: NMS — keep highest-scoring proposal within min_peak_distance
    proposals = sorted(all_proposal_frames)
    if proposals:
        nms_proposals: list[int] = [proposals[0]]
        for frame in proposals[1:]:
            if frame - nms_proposals[-1] >= config.min_peak_distance:
                nms_proposals.append(frame)
            else:
                # Keep the one with higher contact score
                if smoothed_score[frame] > smoothed_score[nms_proposals[-1]]:
                    nms_proposals[-1] = frame
        proposals = nms_proposals

    # Step 5: Assign action type from sequence model at each proposal
    raw_actions: list[DecodedAction] = []
    for frame in proposals:
        # argmax over action classes (skip background at index 0)
        action_probs = frame_probs[1:, frame]  # (6,)
        best_cls = int(np.argmax(action_probs))
        action_name = ACTION_TYPES[best_cls]
        confidence = float(action_probs[best_cls])

        # Get ball position from anchor contact or lookup
        ball_x, ball_y, velocity = 0.0, 0.0, 0.0
        player_track_id = -1
        player_candidates: list[tuple[int, float]] = []
        court_side = "unknown"

        anchor = anchor_contacts.get(frame)
        if anchor is not None:
            ball_x = anchor.ball_x
            ball_y = anchor.ball_y
            velocity = anchor.velocity
            player_track_id = anchor.player_track_id
            player_candidates = list(anchor.player_candidates)
            court_side = anchor.court_side
        elif ball_positions_by_frame is not None:
            bp = ball_positions_by_frame.get(frame)
            if bp is not None:
                ball_x, ball_y = bp

        # Estimate court side from ball position if unknown
        if court_side == "unknown" and ball_y > 0:
            court_side = "near" if ball_y > net_y else "far"

        raw_actions.append(DecodedAction(
            frame=frame,
            action=action_name,
            confidence=confidence,
            contact_score=float(smoothed_score[frame]),
            has_trajectory_anchor=frame in anchor_frames,
            player_track_id=player_track_id,
            player_candidates=player_candidates,
            ball_x=ball_x,
            ball_y=ball_y,
            velocity=velocity,
            court_side=court_side,
        ))

    # Step 6: Constrained Viterbi decoding
    if len(raw_actions) >= 2:
        raw_actions = _viterbi_constrained(raw_actions, frame_probs, config)

    return raw_actions


def _viterbi_constrained(
    actions: list[DecodedAction],
    frame_probs: np.ndarray,
    config: DecoderConfig,
) -> list[DecodedAction]:
    """Apply constrained Viterbi decoding to enforce volleyball rules.

    Finds the most likely sequence of action types given transition
    probabilities and per-frame emission probabilities from the model.

    The first action is constrained to be 'serve' if we detect one in the
    serve window; otherwise all types are considered.
    """
    n_actions = len(actions)
    states = ACTION_TYPES  # 6 action types
    n_states = len(states)

    # Emission log-probs: log P(action_type | frame) from sequence model
    emissions = np.full((n_actions, n_states), -20.0)
    for i, act in enumerate(actions):
        frame = act.frame
        if 0 <= frame < frame_probs.shape[1]:
            for s in range(n_states):
                p = float(frame_probs[s + 1, frame])  # +1 to skip background
                emissions[i, s] = math.log(max(p, 1e-8))

    # Transition log-probs
    trans = np.full((n_states, n_states), math.log(_MIN_TRANSITION_PROB))
    for (prev_name, curr_name), prob in _TRANSITIONS.items():
        pi = states.index(prev_name)
        ci = states.index(curr_name)
        trans[pi, ci] = math.log(prob)

    # Viterbi forward pass
    w_t = config.viterbi_transition_weight
    w_e = config.viterbi_emission_weight

    # Initialize: first position
    dp = np.full((n_actions, n_states), -np.inf)
    backptr = np.zeros((n_actions, n_states), dtype=int)

    # First position: serve gets a bonus if within serve window
    for s in range(n_states):
        dp[0, s] = w_e * emissions[0, s]
        if actions[0].frame <= config.serve_window_frames:
            if states[s] == "serve":
                dp[0, s] += 2.0  # Strong prior for serve being first

    # Forward pass
    for i in range(1, n_actions):
        for s in range(n_states):
            best_score = -np.inf
            best_prev = 0
            for p in range(n_states):
                score = dp[i - 1, p] + w_t * trans[p, s] + w_e * emissions[i, s]
                if score > best_score:
                    best_score = score
                    best_prev = p
            dp[i, s] = best_score
            backptr[i, s] = best_prev

    # Backtrack
    best_path = [0] * n_actions
    best_path[-1] = int(np.argmax(dp[-1]))
    for i in range(n_actions - 2, -1, -1):
        best_path[i] = int(backptr[i + 1, best_path[i + 1]])

    # Update actions with Viterbi-decoded types
    for i, s_idx in enumerate(best_path):
        actions[i].action = states[s_idx]
        # Update confidence to reflect Viterbi probability
        actions[i].confidence = float(
            frame_probs[s_idx + 1, actions[i].frame]
        ) if 0 <= actions[i].frame < frame_probs.shape[1] else actions[i].confidence

    return actions


def attribute_with_team_hint(
    actions: list[DecodedAction],
    player_positions: list[tuple[int, int, float, float]] | None = None,
    team_assignments: dict[int, int] | None = None,
    net_y: float = 0.5,
) -> list[DecodedAction]:
    """Re-attribute players using team hints from decoded action types.

    The action type implies which team should be acting:
    - Serve/set/attack = possessing team
    - Receive/dig = defending team
    - Block = defending team (blocking the attack)

    By filtering candidates to the expected team, we approximate the
    oracle team ceiling (82.0%) without identity discrimination.

    Args:
        actions: Decoded actions with player_candidates populated.
        player_positions: List of (frame, track_id, x, y) for candidate lookup.
        team_assignments: track_id → team (0=near, 1=far) mapping.
        net_y: Net Y position for court side estimation.

    Returns:
        Actions with updated player_track_id.
    """
    if not team_assignments or not actions:
        return actions

    # Determine serving team from first action
    serving_team: int | None = None
    for act in actions:
        if act.action == "serve":
            # Server's team from court side
            if act.court_side == "near":
                serving_team = 0
            elif act.court_side == "far":
                serving_team = 1
            break

    if serving_team is None:
        return actions  # Can't determine possession without serve

    # Track possession: starts with serving team, flips on net-crossing actions
    possession_team = serving_team
    for act in actions:
        # Determine expected team for this action
        if act.action == "serve":
            expected_team = serving_team
        elif act.action in ("receive", "dig", "block"):
            # Defending team = opposite of possession
            expected_team = 1 - possession_team
        else:
            # Set, attack = possessing team
            expected_team = possession_team

        # Filter candidates to expected team
        if act.player_candidates:
            team_filtered = [
                (tid, dist) for tid, dist in act.player_candidates
                if team_assignments.get(tid) == expected_team
            ]
            if team_filtered:
                act.player_track_id = team_filtered[0][0]
                # Keep all candidates but reorder: team-filtered first
                other = [
                    (tid, dist) for tid, dist in act.player_candidates
                    if team_assignments.get(tid) != expected_team
                ]
                act.player_candidates = team_filtered + other

        # Update possession: attack crosses net, block stays same side
        if act.action == "attack":
            possession_team = 1 - possession_team
        elif act.action == "serve":
            possession_team = 1 - serving_team  # After serve, other team receives

    return actions
