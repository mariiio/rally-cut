"""Global cross-rally identity solver (Approach B, Day 1).

Replaces forward Pass-1 Hungarian + Pass-2 stages 1-2 with a single
coordinate-descent solve over all rallies. See
``docs/superpowers/plans/2026-04-27-global-cross-rally-identity.md`` and
``~/.claude/plans/ultrathink-the-cross-rally-player-identi-robust-porcupine.md``.

Day 1 scope: appearance-only K-medoid-style iteration with a Y-sort seed.
Day 2+ adds hard 2v2 team-pair constraint, side classification, position
continuity, orientation co-solve, spectral initialization, multi-restart,
confidence floor + fallback.

Pose feature slot: planned hook in ``_track_rep_cost`` for an additive
pose-distance term; left as ``# TODO(pose-probe)`` in code, no schema
change to ``StoredRallyData`` (per brief §5).
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import linear_sum_assignment

from rallycut.tracking import _profile_drift_probe as _probe
from rallycut.tracking.player_features import compute_track_similarity

if TYPE_CHECKING:
    from rallycut.tracking.match_tracker import StoredRallyData

logger = logging.getLogger(__name__)


# Number of clusters (player identities). Beach volleyball: 4.
NUM_CLUSTERS = 4

# Canonical within-team pair partition. By convention enforced upstream
# (`_initialize_first_rally` Y-sort), pids 1+2 form one team, 3+4 the
# other. The hard team-pair constraint forbids assignments that violate
# this partition when a rally's high-confidence 2v2 split is available.
TEAM_LO_PAIR = (1, 2)
TEAM_HI_PAIR = (3, 4)

# Neutral cost for assignments to clusters with no current members. Sits
# in the middle of the [0, 1] cost range so Hungarian will fill empty
# clusters when no other constraint dominates, but won't pull tracks
# away from clusters that already have evidence.
EMPTY_CLUSTER_COST = 0.5

# Hard team-pair constraint cost (matches HARD_TEAM_PAIR_COST in
# match_tracker.py). Applied to cells where assigning a track to a
# cluster would violate the {1,2}/{3,4} within-team partition under
# the current rally's pair-to-side mapping. The chosen value is far
# above any soft cost (~1.5 max) so it structurally forbids those
# assignments via linear_sum_assignment.
HARD_TEAM_PAIR_COST = 100.0

# Cross-rally position continuity weight. Cost matrix is blended as
# `appearance * (1 - POSITION_WEIGHT) + position * POSITION_WEIGHT`,
# matching the constant of the same name in match_tracker.py. A 0.45
# probe (2026-04-27) regressed cece/2d105b7b/jojo and did not fix wawa
# rally-10 — reverted. Higher weights help one fixture at the cost of
# others, so within-team disambiguation needs an orthogonal signal
# (per-frame rescue or pose), not a knob bump.
#
# Override via `MATCH_SOLVER_POSITION_WEIGHT` env var when diagnosing
# fixtures where appearance discrimination is borderline (low-margin
# Hungarian) and position cost is amplifying noise.
POSITION_WEIGHT = float(os.environ.get("MATCH_SOLVER_POSITION_WEIGHT", "0.30"))

# Normalization scale for the position distance term. Distances above
# this value clamp to 1.0 so a single far-away track doesn't dominate
# the cost matrix. Matches today's match_tracker `min(d/0.3, 1.0)`.
POSITION_NORM = 0.30

# Neutral position cost when the cluster has no prior member with a
# known late position (e.g., first rally a cluster appears, or members
# without `late_positions` entries). Mid-range so the appearance term
# governs without a free-variable bias.
EMPTY_POSITION_COST = 0.5

# Convergence: number of consecutive iterations with zero per-rally
# assignment changes to declare convergence.
CONVERGENCE_PASSES = 2

# Max coordinate-descent iterations before forcing termination.
MAX_ITERATIONS = 10


class MatchSolver:
    """Coordinate-descent solver for cross-rally cluster identity.

    Variables: ``cluster_label[(rally_i, top_track_t)] ∈ {1, 2, 3, 4}``
    (internal cluster ids; final canonical player_id mapping is applied
    by Stage C — ``_select_seed_rally`` + ``_within_team_permutation_from_seed``
    in match_tracker).

    Day 1 objective: minimize sum of within-cluster pairwise track-track
    appearance cost (HSV + ReID via ``compute_track_similarity``).

    Hard constraints (Day 1): within-rally distinct cluster labels are
    enforced implicitly by per-rally Hungarian (bijective).
    """

    def __init__(
        self,
        reid_blend: float = 0.5,
        max_iterations: int = MAX_ITERATIONS,
        convergence_passes: int = CONVERGENCE_PASSES,
    ) -> None:
        self.reid_blend = reid_blend
        self.max_iterations = max_iterations
        self.convergence_passes = convergence_passes

    def solve(
        self,
        stored_rally_data: list[StoredRallyData],
        *,
        pinned_assignments: dict[int, dict[int, int]] | None = None,
    ) -> list[dict[int, int]]:
        """Solve cross-rally identity.

        Returns: list of per-rally ``track_to_player`` mappings, one per
        input rally, in the same order. Empty dict for rallies with no
        ``top_tracks``.

        ``pinned_assignments`` (rally_idx → {track_id: cluster_id}) lets a
        caller anchor specific rallies' assignments. Pinned rallies skip
        ``_assign_rally`` but still contribute their members to
        ``members_by_cluster`` so other rallies' Hungarians see them as
        cluster anchors. This decouples each rally's MatchSolver decision
        from cross-rally input changes (e.g., re-tracking a different
        rally) — the canonical fix for the per-rally cascade falsified
        in Phase 1 (commit 3926cb5). Caller is responsible for validating
        the anchor still fits the current rally state (track_ids match).
        """
        if not stored_rally_data:
            return []

        n_rallies = len(stored_rally_data)
        pinned: dict[int, dict[int, int]] = (
            {i: dict(a) for i, a in pinned_assignments.items()}
            if pinned_assignments else {}
        )

        # Per-rally track_id -> cluster_id (1..NUM_CLUSTERS).
        assignments: list[dict[int, int]] = [{} for _ in range(n_rallies)]
        for idx, anchor in pinned.items():
            if 0 <= idx < n_rallies:
                assignments[idx] = dict(anchor)

        if not pinned:
            # Seed: Y-sort the rally with the most top_tracks. Matches today's
            # `_initialize_first_rally` convention so the canonical relabel
            # downstream (Stage C) sees a layout it can permute. Skipped when
            # any rally is pinned — pins provide ground-truth members for
            # subsequent iterations.
            seed_idx = self._select_seed_index(stored_rally_data)
            assignments[seed_idx] = self._init_from_seed(stored_rally_data[seed_idx])
            logger.info(
                "MatchSolver seed: rally %d (%d top_tracks) → %s",
                seed_idx,
                len(stored_rally_data[seed_idx].top_tracks),
                assignments[seed_idx],
            )
        else:
            logger.info(
                "MatchSolver pinned: %d/%d rallies anchored from prior solve",
                len(pinned), n_rallies,
            )

        # All rallies pinned → no iteration needed. Pure cache hit.
        if len(pinned) == n_rallies:
            return assignments

        # Coordinate descent. Pinned rallies skip _assign_rally but their
        # members still contribute to other rallies' cost matrices.
        converged_passes = 0
        for iteration in range(self.max_iterations):
            members_by_cluster = self._collect_members(assignments)

            changes = 0
            new_assignments: list[dict[int, int]] = []
            for i in range(n_rallies):
                if i in pinned:
                    new_assignments.append(assignments[i])
                    continue
                new_a = self._assign_rally(
                    rally_idx=i,
                    stored=stored_rally_data[i],
                    members_by_cluster=members_by_cluster,
                    all_rallies=stored_rally_data,
                    iteration=iteration,
                    prev_assignment=assignments[i],
                )
                if new_a != assignments[i]:
                    changes += 1
                new_assignments.append(new_a)

            assignments = new_assignments

            if changes == 0:
                converged_passes += 1
                if converged_passes >= self.convergence_passes:
                    logger.info(
                        "MatchSolver converged after %d iterations "
                        "(%d clean passes)",
                        iteration + 1, converged_passes,
                    )
                    return assignments
            else:
                converged_passes = 0
                logger.info(
                    "MatchSolver iter %d: %d/%d rallies reassigned",
                    iteration + 1, changes, n_rallies,
                )

        logger.info(
            "MatchSolver hit max_iterations=%d without 2-pass convergence",
            self.max_iterations,
        )
        return assignments

    def _select_seed_index(
        self,
        rallies: list[StoredRallyData],
    ) -> int:
        """Pick the rally with the most top_tracks for initialization.

        Counts only REAL track ids (>= 0). Synthetic sub-track ids
        (negative, from ``blind_track_split``) are excluded so the seed
        choice is invariant under whether/where the rescue fires —
        without this, splits cascade into a match-wide pid permutation.
        Prefers a rally with NO synthetic tids (clean anchor); falls back
        to first rally with ≥4 real tids; falls back to highest real-tid
        count.
        """
        # Pass 1: first rally with ≥ NUM_CLUSTERS real tids AND zero synth.
        for i, r in enumerate(rallies):
            real = sum(1 for t in r.top_tracks if t >= 0)
            synth = sum(1 for t in r.top_tracks if t < 0)
            if real >= NUM_CLUSTERS and synth == 0:
                return i
        # Pass 2: first rally with ≥ NUM_CLUSTERS real tids (may include synths).
        for i, r in enumerate(rallies):
            if sum(1 for t in r.top_tracks if t >= 0) >= NUM_CLUSTERS:
                return i
        # Pass 3: highest real-tid count, ties broken by lowest index.
        best_idx = 0
        best_count = -1
        for i, r in enumerate(rallies):
            n = sum(1 for t in r.top_tracks if t >= 0)
            if n > best_count:
                best_count = n
                best_idx = i
        return best_idx

    def _init_from_seed(
        self,
        rally: StoredRallyData,
    ) -> dict[int, int]:
        """Y-sort the seed rally's REAL top_tracks into cluster ids 1..N
        respecting the canonical {team_near → 1, 2} {team_far → 3, 4}
        partition.

        Without the team-partition step (the prior bug), a simple
        whole-court Y-sort could place near-court and far-court players
        in interleaved cluster ids when their bbox y-coordinates
        overlapped. The Pass-2 re-anchoring permutation
        (`_within_team_permutation_from_seed` in match_tracker.py) only
        swaps WITHIN team, so a cross-team mis-partition at the seed
        was never recoverable downstream — the entire match would be
        labeled with PIDs 1↔3, 2↔4 swapped (verified 2026-05-02 against
        `videos.player_matching_gt_json` ground truth on 5c756c41:
        matcher 10% accuracy collapsing to a deterministic global
        cross-team swap).

        Synthetic sub-track ids in the seed rally (negative) are skipped
        so the seed pid layout matches the baseline (no-split) convention
        verbatim. Synth ids in the seed get assigned by the iterative
        Hungarian in subsequent passes once cluster members are populated.

        Falls back to whole-court Y-sort when court sides are not
        classified.
        """
        real_tracks = [t for t in rally.top_tracks if t >= 0]
        if not real_tracks:
            return {}

        if not rally.early_positions:
            # No positions → fall back to track_id order.
            return {tid: i + 1 for i, tid in enumerate(sorted(real_tracks))}

        sides = rally.track_court_sides or {}
        near = [t for t in real_tracks if sides.get(t) == 0]
        far = [t for t in real_tracks if sides.get(t) == 1]
        unclassified = [t for t in real_tracks if t not in near and t not in far]

        # If side classification is degenerate (not exactly 2 per side),
        # fall through to whole-court Y-sort.
        if len(near) != 2 or len(far) != 2 or unclassified:
            sorted_tracks = sorted(
                real_tracks,
                key=lambda t: rally.early_positions.get(t, (0.0, 0.0))[1],
            )
            return {tid: i + 1 for i, tid in enumerate(sorted_tracks)}

        # Y-sort within each team. Lower y = first cluster of the team.
        near_sorted = sorted(near, key=lambda t: rally.early_positions[t][1])
        far_sorted = sorted(far, key=lambda t: rally.early_positions[t][1])
        return {
            near_sorted[0]: 1, near_sorted[1]: 2,
            far_sorted[0]: 3, far_sorted[1]: 4,
        }

    def _collect_members(
        self,
        assignments: list[dict[int, int]],
    ) -> dict[int, list[tuple[int, int]]]:
        """Invert per-rally assignments into cluster -> [(rally_idx, track_id)]."""
        members: dict[int, list[tuple[int, int]]] = {
            c: [] for c in range(1, NUM_CLUSTERS + 1)
        }
        for rally_idx, a in enumerate(assignments):
            for tid, cid in a.items():
                members.setdefault(cid, []).append((rally_idx, tid))
        return members

    def _assign_rally(
        self,
        rally_idx: int,
        stored: StoredRallyData,
        members_by_cluster: dict[int, list[tuple[int, int]]],
        all_rallies: list[StoredRallyData],
        *,
        iteration: int = -1,
        prev_assignment: dict[int, int] | None = None,
    ) -> dict[int, int]:
        """Hungarian-assign top_tracks to clusters with a team-pair guard.

        Base cost: mean ``compute_track_similarity`` against other-rally
        members of each cluster. Self-membership (same rally) is excluded
        so a track is never measured against itself.

        Hard team-pair constraint (Phase-1 step 3 in solver form): when
        the rally's y-side and bbox-side classifiers AGREE on a clean 2v2
        partition AND we have 4 top_tracks against 4 clusters, the
        assignment must respect the canonical within-team partition
        {1,2} vs {3,4}. Because the pair-to-side mapping for this rally
        is unknown a priori (a side switch may have flipped it), we run
        two constrained Hungarians (sides=0 → {1,2} OR sides=0 → {3,4})
        and keep the lower-cost result.
        """
        top = list(stored.top_tracks)
        if not top:
            return {}

        cluster_ids = sorted(members_by_cluster.keys())
        n_tracks = len(top)
        n_clusters = len(cluster_ids)

        appearance_cost = self._build_appearance_cost(
            top=top,
            stored=stored,
            cluster_ids=cluster_ids,
            members_by_cluster=members_by_cluster,
            all_rallies=all_rallies,
            rally_idx=rally_idx,
        )
        position_cost = self._build_position_cost(
            top=top,
            stored=stored,
            cluster_ids=cluster_ids,
            members_by_cluster=members_by_cluster,
            all_rallies=all_rallies,
            rally_idx=rally_idx,
        )
        cost = (
            appearance_cost * (1.0 - POSITION_WEIGHT)
            + position_cost * POSITION_WEIGHT
        )

        # Multi-signal team-pair determination (v3, 2026-05-03).
        # Every available position signal (y-side, bbox-side, upstream
        # team_assignments) proposes its own 2v2 partition over `top`.
        # `_assign_with_team_pair` then runs constrained Hungarian under
        # each candidate (and both orientations) and keeps the lowest-
        # cost result — appearance data resolves signal disagreement.
        # When no signal yields a clean 2v2 over `top`, fall back to
        # unconstrained Hungarian.
        candidate_partitions: list[frozenset[frozenset[int]]] = []
        if (
            n_tracks == NUM_CLUSTERS
            and n_clusters == NUM_CLUSTERS
            and set(cluster_ids) == set(TEAM_LO_PAIR + TEAM_HI_PAIR)
        ):
            candidate_partitions = _propose_team_partitions(
                top,
                stored.track_court_sides,
                stored.sides_by_bbox,
                _team_assignments_to_sides(
                    getattr(stored, "team_assignments", None),
                ),
            )
        if candidate_partitions:
            assignment, _team_pair_cost = self._assign_with_team_pair(
                top, cluster_ids, cost, candidate_partitions,
            )
        else:
            row_ind, col_ind = linear_sum_assignment(cost)
            assignment = {
                top[r]: cluster_ids[c] for r, c in zip(row_ind, col_ind)
            }

        _probe.record_solver_iteration(
            iteration=iteration,
            rally_idx=rally_idx,
            top_tracks=top,
            cluster_ids=cluster_ids,
            cost_matrix=cost,
            assignment=assignment,
            prev_assignment=prev_assignment,
        )
        return assignment

    def _build_appearance_cost(
        self,
        *,
        top: list[int],
        stored: StoredRallyData,
        cluster_ids: list[int],
        members_by_cluster: dict[int, list[tuple[int, int]]],
        all_rallies: list[StoredRallyData],
        rally_idx: int,
    ) -> np.ndarray:
        """Mean-cost-to-other-members appearance matrix [n_tracks × n_clusters]."""
        n_tracks = len(top)
        n_clusters = len(cluster_ids)
        cost = np.full((n_tracks, n_clusters), EMPTY_CLUSTER_COST, dtype=float)

        for ti, tid in enumerate(top):
            track_stat = stored.track_stats.get(tid)
            if track_stat is None:
                continue
            for ci, cid in enumerate(cluster_ids):
                members = members_by_cluster.get(cid, [])
                other_members = [
                    (mr, mt) for (mr, mt) in members if mr != rally_idx
                ]
                if not other_members:
                    continue
                sims: list[float] = []
                for (mr, mt) in other_members:
                    m_stat = all_rallies[mr].track_stats.get(mt)
                    if m_stat is None:
                        continue
                    sims.append(
                        compute_track_similarity(
                            track_stat, m_stat, self.reid_blend,
                        )
                    )
                if sims:
                    cost[ti, ci] = float(np.mean(sims))

                # TODO(pose-probe): add a pose-distance term here when
                # the per-track pose feature is wired (Q5 escalation).

        return cost

    def _build_position_cost(
        self,
        *,
        top: list[int],
        stored: StoredRallyData,
        cluster_ids: list[int],
        members_by_cluster: dict[int, list[tuple[int, int]]],
        all_rallies: list[StoredRallyData],
        rally_idx: int,
    ) -> np.ndarray:
        """Cross-rally position-continuity matrix [n_tracks × n_clusters].

        For each cluster, finds the most-recent prior rally where the
        cluster had a member with a known late position. The position
        cost for a candidate track t is the distance from that prior
        late position to t's early position in the current rally,
        clamped to [0, 1] via ``POSITION_NORM``.

        When a cluster has no prior member (first rally it appears) or
        positions are missing, a neutral cost is used so the appearance
        term governs without a free-variable bias.
        """
        n_tracks = len(top)
        n_clusters = len(cluster_ids)
        cost = np.full(
            (n_tracks, n_clusters), EMPTY_POSITION_COST, dtype=float,
        )

        # Find each cluster's most-recent late position (rally < rally_idx).
        cluster_prev_late: dict[int, tuple[float, float]] = {}
        for ci, cid in enumerate(cluster_ids):
            members = members_by_cluster.get(cid, [])
            best_idx = -1
            best_pos: tuple[float, float] | None = None
            for (mr, mt) in members:
                if mr >= rally_idx:
                    continue
                late = all_rallies[mr].late_positions.get(mt)
                if late is None:
                    continue
                if mr > best_idx:
                    best_idx = mr
                    best_pos = late
            if best_pos is not None:
                cluster_prev_late[cid] = best_pos

        for ti, tid in enumerate(top):
            curr_early = stored.early_positions.get(tid)
            if curr_early is None:
                continue
            for ci, cid in enumerate(cluster_ids):
                prev_late = cluster_prev_late.get(cid)
                if prev_late is None:
                    continue
                d = float(
                    ((curr_early[0] - prev_late[0]) ** 2
                     + (curr_early[1] - prev_late[1]) ** 2) ** 0.5
                )
                cost[ti, ci] = min(d / POSITION_NORM, 1.0)

        return cost

    def _assign_with_team_pair(
        self,
        top: list[int],
        cluster_ids: list[int],
        base_cost: np.ndarray,
        candidate_partitions: list[frozenset[frozenset[int]]],
    ) -> tuple[dict[int, int], float]:
        """Constrained Hungarian over multiple candidate 2v2 partitions.

        Each candidate partition is a frozenset of two frozensets of two
        track ids — one of the 3 possible ways to split 4 tracks into
        two teams. For every (partition, sides_orientation) combination,
        applies HARD_TEAM_PAIR_COST to cells that would violate the
        partition under that orientation, runs Hungarian, and keeps the
        global lowest-cost result.

        Returns (assignment, cost). Cost is the unpenalized total under
        the winning assignment (HARD penalties are stripped so a winner
        with all valid cells reflects true appearance cost).
        """
        cluster_idx_by_pid = {pid: i for i, pid in enumerate(cluster_ids)}
        lo_idx = [cluster_idx_by_pid[p] for p in TEAM_LO_PAIR]
        hi_idx = [cluster_idx_by_pid[p] for p in TEAM_HI_PAIR]

        best_cost = float("inf")
        best_assignment: dict[int, int] = {}

        for partition in candidate_partitions:
            # Each partition has exactly two pairs; assign one pair to
            # "side 0" and the other to "side 1". The orientation
            # choice maps each side to either {pid 1, 2} or {pid 3, 4}.
            pairs = list(partition)
            if len(pairs) != 2:
                continue
            for orientation in (0, 1):
                pair0 = pairs[orientation]
                pair1 = pairs[1 - orientation]
                # pair0 → low cluster pids (P1, P2), pair1 → high (P3, P4)
                # AND the mirror: pair0 → high, pair1 → low.
                for pair0_target, pair1_target in (
                    (lo_idx, hi_idx),
                    (hi_idx, lo_idx),
                ):
                    penalized = base_cost.copy()
                    for ti, tid in enumerate(top):
                        if tid in pair0:
                            forbidden = pair1_target
                        elif tid in pair1:
                            forbidden = pair0_target
                        else:
                            # Track not in this partition — only happens
                            # when partition's track set differs from
                            # `top`. Skip the constraint for it.
                            continue
                        for ci in forbidden:
                            penalized[ti, ci] = HARD_TEAM_PAIR_COST

                    row_ind, col_ind = linear_sum_assignment(penalized)
                    total = float(penalized[row_ind, col_ind].sum())
                    if total < best_cost:
                        best_cost = total
                        best_assignment = {
                            top[r]: cluster_ids[c]
                            for r, c in zip(row_ind, col_ind)
                        }

        return best_assignment, best_cost


def _signal_proposes_partition(
    signal: dict[int, int] | None, tracks: set[int],
) -> frozenset[frozenset[int]] | None:
    """If signal classifies the 4 `tracks` into a clean 2v2, return the
    partition as a frozenset of two frozensets. Else None.

    Robustness rule: ignores signal entries for track ids outside `tracks`
    (lets the same signal serve other rallies' top_tracks). Requires
    EXACTLY 2 + 2 within the requested track set; 3v1 / 4v0 / fewer-than-4
    classifications return None ("signal is degenerate, no proposal").
    """
    if not signal:
        return None
    near: set[int] = set()
    far: set[int] = set()
    for tid in tracks:
        side = signal.get(tid)
        if side == 0:
            near.add(tid)
        elif side == 1:
            far.add(tid)
    if len(near) != 2 or len(far) != 2:
        return None
    return frozenset({frozenset(near), frozenset(far)})


def _propose_team_partitions(
    tracks: list[int],
    *signals: dict[int, int] | None,
) -> list[frozenset[frozenset[int]]]:
    """Collect distinct 2v2 partition candidates from available signals.

    Each signal contributes at most one candidate (its 2v2 classification
    of the 4 `tracks`, or nothing if the signal is degenerate over those
    tracks). Duplicates collapse to one candidate. Returned list has
    1, 2, or 3 entries (or 0 when no signal proposes a clean 2v2).

    The cost-decided team-pair Hungarian (`_assign_with_team_pair`) tries
    every (partition, orientation) combination and picks the lowest-cost
    result, so signals don't need to agree — disagreement is resolved by
    the appearance data. This is the robust replacement for the legacy
    "require unanimity" gate that disengaged the team-pair constraint
    whenever any single track's two side classifiers disagreed.

    Robustness for our context (occlusion, off-screen, players crossing
    sides): bbox-height misclassifies crouching/diving players; y-position
    misclassifies players who cross sides or have small position samples;
    upstream team_assignments may be missing. Each signal contributes
    when valid; the union of candidates covers the cases where any one
    signal alone would have driven a wrong choice.
    """
    if len(tracks) != NUM_CLUSTERS:
        return []
    track_set = set(tracks)
    seen: set[frozenset[frozenset[int]]] = set()
    candidates: list[frozenset[frozenset[int]]] = []
    for sig in signals:
        prop = _signal_proposes_partition(sig, track_set)
        if prop is None or prop in seen:
            continue
        seen.add(prop)
        candidates.append(prop)
    return candidates


def _team_assignments_to_sides(
    team_assignments: dict[int, int] | None,
) -> dict[int, int] | None:
    """Convert upstream team_assignments {tid: 0/1} into a side-shaped
    dict the partition propose helper can consume.

    `team_assignments` from the tracking pipeline already uses 0=team A,
    1=team B (mapped from "near"/"far" in `_classify_track_sides`). The
    semantic match is exact, so this is currently a passthrough — kept
    as a named function so the caller's intent is explicit and so future
    encoding changes are localized.
    """
    return team_assignments
