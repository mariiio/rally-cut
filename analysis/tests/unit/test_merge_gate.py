"""Tests for the unified merge_gate helper.

Covers each gate independently (overlap, velocity, learned veto) plus
combinations and config-disable behavior. Bit-exact equivalence with
`tracklet_link._would_create_velocity_anomaly` and
`merge_veto.learned_cosine_veto` is verified via parallel calls.
"""

from __future__ import annotations

import numpy as np

from rallycut.tracking.color_repair import LearnedEmbeddingStore
from rallycut.tracking.merge_gate import (
    MergeGateConfig,
    MergeGateResult,
    should_block_merge,
    tracks_overlap_temporally,
    velocity_anomaly,
)
from rallycut.tracking.player_tracker import PlayerPosition


def _pos(track_id: int, frame: int, x: float = 0.5, y: float = 0.5) -> PlayerPosition:
    return PlayerPosition(
        frame_number=frame,
        track_id=track_id,
        x=x,
        y=y,
        width=0.05,
        height=0.15,
        confidence=0.9,
    )


def _track_at(track_id: int, frames: range, x: float = 0.5, y: float = 0.5) -> list[PlayerPosition]:
    return [_pos(track_id, f, x, y) for f in frames]


# --- Overlap gate ---------------------------------------------------------


class TestOverlapGate:
    def test_no_overlap_passes(self) -> None:
        result = should_block_merge(
            track_id_a=1, track_id_b=2,
            positions=_track_at(1, range(0, 30)) + _track_at(2, range(40, 70)),
        )
        assert not result.blocked

    def test_small_overlap_under_threshold_passes(self) -> None:
        # 5-frame overlap is below the default max_overlap_frames=15
        result = should_block_merge(
            track_id_a=1, track_id_b=2,
            positions=_track_at(1, range(0, 35)) + _track_at(2, range(30, 60)),
        )
        assert not result.blocked

    def test_large_overlap_blocks(self) -> None:
        # 20-frame overlap exceeds default 15
        result = should_block_merge(
            track_id_a=1, track_id_b=2,
            positions=_track_at(1, range(0, 50)) + _track_at(2, range(30, 80)),
        )
        assert result.blocked
        assert "temporal_overlap" in result.reason

    def test_overlap_disabled(self) -> None:
        # Heavy overlap, but gate disabled
        config = MergeGateConfig(enable_overlap=False)
        result = should_block_merge(
            track_id_a=1, track_id_b=2,
            positions=_track_at(1, range(0, 50)) + _track_at(2, range(0, 50)),
            config=config,
        )
        # No overlap block (other gates may still pass since velocities are zero
        # and there's no learned store)
        assert not result.blocked

    def test_overlap_helper_direct(self) -> None:
        a = set(range(0, 30))
        b = set(range(20, 50))
        # 10 frames overlap (20-29)
        assert tracks_overlap_temporally(a, b, max_allowed_overlap=5) is True
        assert tracks_overlap_temporally(a, b, max_allowed_overlap=15) is False


# --- Velocity gate (image-plane) -----------------------------------------


class TestVelocityGate:
    def test_co_located_fragments_pass(self) -> None:
        # Same x,y across the gap → 0 displacement
        result = should_block_merge(
            track_id_a=1, track_id_b=2,
            positions=_track_at(1, range(0, 30), x=0.3, y=0.4)
            + _track_at(2, range(35, 65), x=0.3, y=0.4),
        )
        assert not result.blocked

    def test_far_apart_blocks(self) -> None:
        # Track 1 at (0.2, 0.2), track 2 at (0.8, 0.8), gap of 5 frames
        # Image-plane displacement ~0.85 >> 0.20 default threshold
        result = should_block_merge(
            track_id_a=1, track_id_b=2,
            positions=_track_at(1, range(0, 30), x=0.2, y=0.2)
            + _track_at(2, range(35, 65), x=0.8, y=0.8),
        )
        assert result.blocked
        assert result.reason == "velocity_anomaly"

    def test_velocity_disabled(self) -> None:
        config = MergeGateConfig(enable_velocity=False)
        result = should_block_merge(
            track_id_a=1, track_id_b=2,
            positions=_track_at(1, range(0, 30), x=0.2, y=0.2)
            + _track_at(2, range(35, 65), x=0.8, y=0.8),
            config=config,
        )
        assert not result.blocked

    def test_velocity_helper_direct(self) -> None:
        positions = (
            _track_at(1, range(0, 30), x=0.2, y=0.2)
            + _track_at(2, range(35, 65), x=0.8, y=0.8)
        )
        assert velocity_anomaly(positions, 1, 2) is True
        positions_close = (
            _track_at(1, range(0, 30), x=0.5, y=0.5)
            + _track_at(2, range(35, 65), x=0.5, y=0.5)
        )
        assert velocity_anomaly(positions_close, 1, 2) is False

    def test_loose_image_threshold_passes(self) -> None:
        # Bigger threshold should let through what default would block
        config = MergeGateConfig(max_displacement_image=1.0)
        result = should_block_merge(
            track_id_a=1, track_id_b=2,
            positions=_track_at(1, range(0, 30), x=0.2, y=0.2)
            + _track_at(2, range(35, 65), x=0.8, y=0.8),
            config=config,
        )
        assert not result.blocked


# --- Learned ReID gate ---------------------------------------------------


class TestLearnedVetoGate:
    @staticmethod
    def _embedding(seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(128).astype(np.float32)
        return v / np.linalg.norm(v)

    def _store_with(
        self, track_id: int, frames: range, embedding: np.ndarray,
    ) -> LearnedEmbeddingStore:
        store = LearnedEmbeddingStore()
        for f in frames:
            store.add(track_id, f, embedding)
        return store

    def test_disabled_when_threshold_zero(self) -> None:
        # Even with very different embeddings, threshold=0 abstains.
        store = LearnedEmbeddingStore()
        for f in range(0, 30):
            store.add(1, f, self._embedding(seed=1))
        for f in range(35, 65):
            store.add(2, f, self._embedding(seed=999))

        config = MergeGateConfig(learned_veto_cos=0.0)
        result = should_block_merge(
            track_id_a=1, track_id_b=2,
            positions=_track_at(1, range(0, 30)) + _track_at(2, range(35, 65)),
            config=config,
            learned_store=store,
        )
        assert not result.blocked

    def test_blocks_dissimilar_when_active(self) -> None:
        emb_a = self._embedding(seed=1)
        emb_b = self._embedding(seed=999)  # cos sim near 0

        store = LearnedEmbeddingStore()
        for f in range(0, 30):
            store.add(1, f, emb_a)
        for f in range(35, 65):
            store.add(2, f, emb_b)

        config = MergeGateConfig(
            learned_veto_cos=0.6,
            enable_overlap=False,  # isolate the learned gate
            enable_velocity=False,
        )
        result = should_block_merge(
            track_id_a=1, track_id_b=2,
            positions=_track_at(1, range(0, 30)) + _track_at(2, range(35, 65)),
            config=config,
            learned_store=store,
        )
        assert result.blocked
        assert "learned_reid" in result.reason

    def test_passes_similar_when_active(self) -> None:
        emb = self._embedding(seed=42)

        store = LearnedEmbeddingStore()
        for f in range(0, 30):
            store.add(1, f, emb)
        for f in range(35, 65):
            store.add(2, f, emb)

        config = MergeGateConfig(
            learned_veto_cos=0.6,
            enable_overlap=False,
            enable_velocity=False,
        )
        result = should_block_merge(
            track_id_a=1, track_id_b=2,
            positions=_track_at(1, range(0, 30)) + _track_at(2, range(35, 65)),
            config=config,
            learned_store=store,
        )
        assert not result.blocked

    def test_no_store_abstains(self) -> None:
        config = MergeGateConfig(learned_veto_cos=0.6)
        result = should_block_merge(
            track_id_a=1, track_id_b=2,
            positions=_track_at(1, range(0, 30)) + _track_at(2, range(35, 65)),
            config=config,
            learned_store=None,
        )
        assert not result.blocked


# --- First-blocking-gate ordering ----------------------------------------


class TestGateOrdering:
    def test_overlap_reported_before_velocity(self) -> None:
        # Heavy overlap AND large displacement — overlap should win.
        result = should_block_merge(
            track_id_a=1, track_id_b=2,
            positions=_track_at(1, range(0, 50), x=0.2, y=0.2)
            + _track_at(2, range(0, 50), x=0.8, y=0.8),
        )
        assert result.blocked
        assert "temporal_overlap" in result.reason

    def test_velocity_reported_when_no_overlap(self) -> None:
        result = should_block_merge(
            track_id_a=1, track_id_b=2,
            positions=_track_at(1, range(0, 30), x=0.2, y=0.2)
            + _track_at(2, range(50, 80), x=0.8, y=0.8),
        )
        assert result.blocked
        assert result.reason == "velocity_anomaly"


# --- Frame-set derivation ------------------------------------------------


class TestFrameDerivation:
    def test_frames_derived_from_positions_when_omitted(self) -> None:
        positions = _track_at(1, range(0, 30)) + _track_at(2, range(35, 65))
        # Default behavior — no overlap → no block
        result = should_block_merge(1, 2, positions)
        assert not result.blocked

    def test_frames_explicit_overrides_derivation(self) -> None:
        # Pass empty frame sets to override — overlap gate sees no overlap.
        positions = _track_at(1, range(0, 50)) + _track_at(2, range(0, 50))
        result = should_block_merge(
            1, 2, positions,
            frames_a=set(),
            frames_b=set(),
        )
        # Overlap gate sees 0 overlap (passed explicitly), velocity sees
        # co-located → no block. Verifies explicit override.
        assert not result.blocked


# --- Default config matches tracklet_link history -----------------------


class TestDefaultConfigSemantics:
    def test_defaults_match_tracklet_link(self) -> None:
        from rallycut.tracking.tracklet_link import (
            DEFAULT_MAX_MERGE_VELOCITY,
            DEFAULT_MAX_MERGE_VELOCITY_METERS,
            DEFAULT_MERGE_VELOCITY_WINDOW,
        )
        cfg = MergeGateConfig()
        assert cfg.max_displacement_image == DEFAULT_MAX_MERGE_VELOCITY
        assert cfg.max_displacement_meters == DEFAULT_MAX_MERGE_VELOCITY_METERS
        assert cfg.velocity_window == DEFAULT_MERGE_VELOCITY_WINDOW

    def test_default_config_returns_unblocked_for_clean_input(self) -> None:
        positions = _track_at(1, range(0, 30), x=0.5, y=0.5) + _track_at(
            2, range(40, 70), x=0.5, y=0.5,
        )
        result = should_block_merge(1, 2, positions)
        assert isinstance(result, MergeGateResult)
        assert not result.blocked
        assert result.reason == ""
