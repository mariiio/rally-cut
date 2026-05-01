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
POSITION_WEIGHT = 0.30

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
        """Y-sort the seed rally's REAL top_tracks into cluster ids 1..N.

        Synthetic sub-track ids in the seed rally (negative) are skipped
        so the seed pid layout matches the baseline (no-split) convention
        verbatim. Synth ids in the seed get assigned by the iterative
        Hungarian in subsequent passes once cluster members are populated.

        Falls back to track_id order when ``early_positions`` is missing.
        Cluster ids start at 1 to match the existing player_id convention
        used downstream (Stage C, ``_apply_within_team_permutation``).
        """
        real_tracks = [t for t in rally.top_tracks if t >= 0]
        if not real_tracks:
            return {}

        if rally.early_positions:
            sorted_tracks = sorted(
                real_tracks,
                key=lambda t: rally.early_positions.get(t, (0.0, 0.0))[1],
            )
        else:
            sorted_tracks = sorted(real_tracks)

        return {tid: i + 1 for i, tid in enumerate(sorted_tracks)}

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

        high_conf = _high_confidence_sides_for_team_pair(
            stored.track_court_sides, stored.sides_by_bbox,
        )
        if (
            high_conf
            and n_tracks == NUM_CLUSTERS
            and n_clusters == NUM_CLUSTERS
            and set(cluster_ids) == set(TEAM_LO_PAIR + TEAM_HI_PAIR)
            and all(t in high_conf for t in top)
        ):
            assignment = self._assign_with_team_pair(
                top, cluster_ids, cost, high_conf,
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
        high_conf: dict[int, int],
    ) -> dict[int, int]:
        """Constrained Hungarian respecting the {1,2}/{3,4} partition.

        Tries both pair-to-side mappings and returns the lower-cost
        assignment. Cells that violate the chosen mapping are stamped
        with ``HARD_TEAM_PAIR_COST`` to make them structurally infeasible
        for ``linear_sum_assignment``.
        """
        cluster_idx_by_pid = {pid: i for i, pid in enumerate(cluster_ids)}
        lo_idx = [cluster_idx_by_pid[p] for p in TEAM_LO_PAIR]
        hi_idx = [cluster_idx_by_pid[p] for p in TEAM_HI_PAIR]

        best_cost = float("inf")
        best_assignment: dict[int, int] = {}

        for side_zero_pair, side_one_pair in (
            (lo_idx, hi_idx),
            (hi_idx, lo_idx),
        ):
            penalized = base_cost.copy()
            for ti, tid in enumerate(top):
                track_side = high_conf[tid]
                forbidden = side_one_pair if track_side == 0 else side_zero_pair
                for ci in forbidden:
                    penalized[ti, ci] = HARD_TEAM_PAIR_COST

            row_ind, col_ind = linear_sum_assignment(penalized)
            total = float(penalized[row_ind, col_ind].sum())
            if total < best_cost:
                best_cost = total
                best_assignment = {
                    top[r]: cluster_ids[c] for r, c in zip(row_ind, col_ind)
                }

        return best_assignment


def _high_confidence_sides_for_team_pair(
    track_court_sides: dict[int, int],
    sides_by_bbox: dict[int, int],
) -> dict[int, int]:
    """Local copy of ``MatchPlayerTracker._high_confidence_sides_for_team_pair``.

    Returns the agreement set ``{track_id: side}`` only when the y-side
    and bbox-side classifiers agree on a clean 2v2 split. Empty otherwise
    so the solver falls back to an unconstrained Hungarian.
    """
    if not sides_by_bbox or not track_court_sides:
        return {}
    agreed: dict[int, int] = {}
    for tid, y_side in track_court_sides.items():
        bb_side = sides_by_bbox.get(tid)
        if bb_side is not None and bb_side == y_side:
            agreed[tid] = y_side
    near = sum(1 for s in agreed.values() if s == 0)
    far = sum(1 for s in agreed.values() if s == 1)
    if near != 2 or far != 2:
        return {}
    return agreed
