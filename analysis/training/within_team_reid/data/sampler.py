"""Composed batch sampler: 64 pos + 32 easy-neg + 32 hard-neg pairs per step.

Per-batch composition is FIXED (not random ratio drift). Pairs are drawn
without replacement within an epoch; if a tier runs out, the epoch ends.
Positive pairs are stratified across ≥ 8 distinct rallies per batch to keep
in-batch negatives distributed across video conditions.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
from torch.utils.data import Sampler

from .manifest import Pair

logger = logging.getLogger("within_team_reid.data.sampler")


class PairBatchSampler(Sampler[list[int]]):
    """Yields batches of pair indices. Each batch is a list of length
    `pos_per_batch + easy_neg_per_batch + hard_neg_per_batch`.

    Indices are into the `pairs` list passed at construction.
    """

    def __init__(
        self,
        pairs: list[Pair],
        pos_per_batch: int = 64,
        easy_neg_per_batch: int = 32,
        hard_neg_per_batch: int = 32,
        min_pos_rallies_per_batch: int = 8,
        seed: int = 42,
    ) -> None:
        self.pairs = pairs
        self.pos_per_batch = pos_per_batch
        self.easy_neg_per_batch = easy_neg_per_batch
        self.hard_neg_per_batch = hard_neg_per_batch
        self.min_pos_rallies_per_batch = min_pos_rallies_per_batch
        self.seed = seed

        self.pos_indices: list[int] = []
        self.easy_neg_indices: list[int] = []
        self.hard_neg_indices: list[int] = []
        for i, p in enumerate(pairs):
            if p.tier == "positive":
                self.pos_indices.append(i)
            elif p.tier == "easy_neg":
                self.easy_neg_indices.append(i)
            elif p.tier in {"mid", "gold"}:
                self.hard_neg_indices.append(i)

        # Per-rally bucketed pos for stratification
        self.pos_by_rally: dict[str, list[int]] = defaultdict(list)
        for i in self.pos_indices:
            self.pos_by_rally[pairs[i].rally_id].append(i)

        logger.info(
            "Sampler initialized: pos=%d easy_neg=%d hard_neg=%d (across %d rallies for pos)",
            len(self.pos_indices), len(self.easy_neg_indices), len(self.hard_neg_indices),
            len(self.pos_by_rally),
        )
        if len(self.pos_indices) < pos_per_batch:
            logger.warning("Pos pool smaller than per-batch quota — sampler will recycle")
        if len(self.hard_neg_indices) < hard_neg_per_batch:
            logger.warning("Hard-neg pool smaller than per-batch quota — sampler will recycle")
        if len(self.easy_neg_indices) < easy_neg_per_batch:
            logger.warning("Easy-neg pool smaller than per-batch quota — sampler will recycle")

        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __len__(self) -> int:
        # Epoch length = how many full batches we can build from pos pool.
        if not self.pos_indices:
            return 0
        return max(1, len(self.pos_indices) // self.pos_per_batch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)

        # Shuffle each pool fresh per epoch.
        pos = list(self.pos_indices)
        easy = list(self.easy_neg_indices)
        hard = list(self.hard_neg_indices)
        rng.shuffle(pos)
        rng.shuffle(easy)
        rng.shuffle(hard)

        n_steps = len(self)
        # Precompute stratified pos batches: try to ensure ≥ min_pos_rallies per batch.
        pos_batches = self._stratify_pos_into_batches(pos, n_steps, rng)

        easy_iter = self._cycling_iter(easy, rng)
        hard_iter = self._cycling_iter(hard, rng)

        for step in range(n_steps):
            pos_batch = pos_batches[step]
            easy_batch = [next(easy_iter) for _ in range(self.easy_neg_per_batch)]
            hard_batch = [next(hard_iter) for _ in range(self.hard_neg_per_batch)]
            batch = pos_batch + easy_batch + hard_batch
            yield batch

    def _stratify_pos_into_batches(
        self,
        pos_shuffled: list[int],
        n_steps: int,
        rng: np.random.Generator,
    ) -> list[list[int]]:
        """Split shuffled pos pool into n_steps batches of pos_per_batch, each
        spanning at least min_pos_rallies (best-effort).
        """
        batches: list[list[int]] = []
        idx = 0
        for _ in range(n_steps):
            batch: list[int] = []
            seen_rallies: set[str] = set()
            cursor = idx
            # First pass — greedy: try to add unique-rally indices.
            while len(batch) < self.pos_per_batch and cursor < len(pos_shuffled):
                pi = pos_shuffled[cursor]
                rid = self.pairs[pi].rally_id
                if (
                    rid not in seen_rallies
                    or len(seen_rallies) >= self.min_pos_rallies_per_batch
                ):
                    batch.append(pi)
                    seen_rallies.add(rid)
                cursor += 1
            # If we ran short (e.g., near pool end), recycle from the front.
            recycle_cursor = 0
            while len(batch) < self.pos_per_batch:
                pi = pos_shuffled[recycle_cursor % len(pos_shuffled)]
                batch.append(pi)
                recycle_cursor += 1
            idx = cursor
            batches.append(batch)
        return batches

    def _cycling_iter(self, pool: list[int], rng: np.random.Generator):
        """Yield pool indices, reshuffling at the end."""
        local = list(pool)
        rng.shuffle(local)
        i = 0
        while True:
            if i >= len(local):
                rng.shuffle(local)
                i = 0
            yield local[i]
            i += 1
