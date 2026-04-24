"""Unit tests for relabel-with-crops helpers (Phase 1.1).

These helpers convert match_analysis_json on-disk shape into the typed
objects refine_assignments needs, run replay_refine_from_scratchpad,
and write the new assignments back. The CLI (relabel_with_crops.py)
is orchestration on top of these.
"""

from __future__ import annotations

import numpy as np

from rallycut.tracking.match_tracker import RallyTrackingResult
from rallycut.tracking.player_features import HS_BINS, PlayerAppearanceProfile


class TestReconstructInitialResults:
    """`reconstruct_initial_results` rebuilds RallyTrackingResult list from
    the match_analysis_json `rallies` array."""

    def test_handles_typical_rally_entry(self) -> None:
        from rallycut.tracking.relabel import reconstruct_initial_results

        entries = [
            {
                "rallyId": "abc-1",
                "rallyIndex": 0,
                "startMs": 1000,
                "endMs": 5000,
                "trackToPlayer": {"100": 1, "101": 2, "200": 3, "201": 4},
                "assignmentConfidence": 0.85,
                "sideSwitchDetected": False,
                "serverPlayerId": 2,
            },
        ]

        results = reconstruct_initial_results(entries)

        assert len(results) == 1
        r = results[0]
        assert r.rally_index == 0
        assert r.track_to_player == {100: 1, 101: 2, 200: 3, 201: 4}
        assert r.assignment_confidence == 0.85
        assert r.side_switch_detected is False
        assert r.server_player_id == 2

    def test_handles_missing_optional_fields(self) -> None:
        from rallycut.tracking.relabel import reconstruct_initial_results

        entries = [
            {
                "rallyId": "abc-2",
                "rallyIndex": 1,
                "trackToPlayer": {"5": 1},
                # No assignmentConfidence, sideSwitchDetected, serverPlayerId.
            },
        ]

        results = reconstruct_initial_results(entries)

        assert len(results) == 1
        assert results[0].assignment_confidence == 0.0
        assert results[0].side_switch_detected is False
        assert results[0].server_player_id is None

    def test_handles_empty_list(self) -> None:
        from rallycut.tracking.relabel import reconstruct_initial_results

        assert reconstruct_initial_results([]) == []


class TestReconstructProfiles:
    """`reconstruct_profiles` rebuilds the pid → PlayerAppearanceProfile dict
    from match_analysis_json `playerProfiles`."""

    def test_roundtrip(self) -> None:
        from rallycut.tracking.relabel import reconstruct_profiles

        original = {
            1: PlayerAppearanceProfile(player_id=1, rally_count=10),
            2: PlayerAppearanceProfile(player_id=2, rally_count=8),
        }
        original[1].avg_skin_tone_hsv = (15.0, 180.0, 120.0)
        original[1].avg_lower_hist = np.linspace(
            0, 1, int(np.prod(HS_BINS)), dtype=np.float32
        ).reshape(HS_BINS)
        original[1].lower_hist_count = 5

        serialized = {str(pid): prof.to_dict() for pid, prof in original.items()}

        restored = reconstruct_profiles(serialized)

        assert sorted(restored.keys()) == [1, 2]
        assert restored[1].player_id == 1
        assert restored[1].avg_skin_tone_hsv == (15.0, 180.0, 120.0)
        assert restored[1].rally_count == 10
        assert restored[1].lower_hist_count == 5
        assert np.array_equal(restored[1].avg_lower_hist, original[1].avg_lower_hist)

    def test_handles_empty_dict(self) -> None:
        from rallycut.tracking.relabel import reconstruct_profiles

        assert reconstruct_profiles({}) == {}


class TestApplyRelabelToRallyEntries:
    """`apply_relabel_to_rally_entries` produces the updated rally_entries
    list to write back to match_analysis_json after replay."""

    def test_updates_track_to_player_confidence_and_switch(self) -> None:
        from rallycut.tracking.relabel import apply_relabel_to_rally_entries

        original = [
            {
                "rallyId": "abc-1",
                "rallyIndex": 0,
                "startMs": 1000,
                "endMs": 5000,
                "trackToPlayer": {"100": 1},
                "assignmentConfidence": 0.5,
                "sideSwitchDetected": False,
                "serverPlayerId": 2,
            },
            {
                "rallyId": "abc-2",
                "rallyIndex": 1,
                "startMs": 6000,
                "endMs": 10_000,
                "trackToPlayer": {"200": 3},
                "assignmentConfidence": 0.6,
                "sideSwitchDetected": False,
                "serverPlayerId": 3,
            },
        ]
        refined = [
            RallyTrackingResult(
                rally_index=0,
                track_to_player={100: 2},  # Relabeled 1 → 2
                server_player_id=2,
                side_switch_detected=False,
                assignment_confidence=0.95,
            ),
            RallyTrackingResult(
                rally_index=1,
                track_to_player={200: 4},
                server_player_id=3,
                side_switch_detected=True,  # Newly detected
                assignment_confidence=0.88,
            ),
        ]

        new_entries = apply_relabel_to_rally_entries(original, refined)

        assert len(new_entries) == 2
        assert new_entries[0]["trackToPlayer"] == {"100": 2}
        assert new_entries[0]["assignmentConfidence"] == 0.95
        assert new_entries[0]["sideSwitchDetected"] is False
        # Untouched fields preserved.
        assert new_entries[0]["rallyId"] == "abc-1"
        assert new_entries[0]["startMs"] == 1000
        assert new_entries[0]["serverPlayerId"] == 2

        assert new_entries[1]["trackToPlayer"] == {"200": 4}
        assert new_entries[1]["sideSwitchDetected"] is True

    def test_does_not_mutate_input(self) -> None:
        from rallycut.tracking.relabel import apply_relabel_to_rally_entries

        original = [{
            "rallyId": "x",
            "rallyIndex": 0,
            "trackToPlayer": {"5": 1},
            "assignmentConfidence": 0.1,
            "sideSwitchDetected": False,
            "serverPlayerId": None,
        }]
        refined = [
            RallyTrackingResult(
                rally_index=0,
                track_to_player={5: 4},
                server_player_id=None,
                side_switch_detected=False,
                assignment_confidence=0.99,
            ),
        ]

        apply_relabel_to_rally_entries(original, refined)

        # Input must be unchanged.
        assert original[0]["trackToPlayer"] == {"5": 1}
        assert original[0]["assignmentConfidence"] == 0.1

    def test_length_mismatch_raises(self) -> None:
        from rallycut.tracking.relabel import apply_relabel_to_rally_entries

        try:
            apply_relabel_to_rally_entries([{"rallyId": "x", "rallyIndex": 0,
                                             "trackToPlayer": {}}], [])
        except ValueError as e:
            assert "length" in str(e).lower() or "mismatch" in str(e).lower()
        else:
            raise AssertionError("expected ValueError on length mismatch")


class TestDetectAnomalousCrops:
    """Intra-player crop anomaly check (Phase 1.2 / plan §5 Q1 refinement).

    When the user provides ≥2 crops for the same pid, we want to surface a
    warning if one crop is visually distant from the others — it might be a
    miss-click on a different player. Threshold: a crop's median pairwise
    cosine distance to its cohort exceeds (cohort_median + 1.5σ).

    With <3 crops there's no cohort to compare against, so we return no
    flags rather than guessing.
    """

    def _make_normed(self, vec: list[float]) -> np.ndarray:
        arr = np.array(vec, dtype=np.float32)
        n = float(np.linalg.norm(arr))
        return arr / n if n > 0 else arr

    def test_uniform_cohort_flags_nothing(self) -> None:
        from rallycut.tracking.relabel import detect_anomalous_crops

        # 4 nearly identical embeddings.
        rng = np.random.default_rng(seed=42)
        base = self._make_normed([1.0, 0.0, 0.0, 0.0])
        embeddings = np.stack([
            base + 0.01 * rng.standard_normal(4).astype(np.float32)
            for _ in range(4)
        ])
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        flags = detect_anomalous_crops({1: embeddings})

        assert flags[1] == []

    def test_single_outlier_flagged(self) -> None:
        from rallycut.tracking.relabel import detect_anomalous_crops

        # Three similar embeddings + one wildly different.
        good = self._make_normed([1.0, 0.0, 0.0, 0.0])
        outlier = self._make_normed([0.0, 0.0, 0.0, 1.0])  # cos ≈ 0 with `good`
        embeddings = np.stack([good, good, good, outlier])

        flags = detect_anomalous_crops({1: embeddings})

        assert flags[1] == [3]

    def test_two_crops_returns_empty(self) -> None:
        """With only 2 crops there's only 1 pairwise distance — no cohort."""
        from rallycut.tracking.relabel import detect_anomalous_crops

        a = self._make_normed([1.0, 0.0, 0.0, 0.0])
        b = self._make_normed([0.0, 1.0, 0.0, 0.0])

        flags = detect_anomalous_crops({1: np.stack([a, b])})

        assert flags[1] == []

    def test_one_crop_returns_empty(self) -> None:
        from rallycut.tracking.relabel import detect_anomalous_crops

        a = self._make_normed([1.0, 0.0, 0.0, 0.0])

        flags = detect_anomalous_crops({1: a.reshape(1, 4)})

        assert flags[1] == []

    def test_empty_input_returns_empty(self) -> None:
        from rallycut.tracking.relabel import detect_anomalous_crops

        assert detect_anomalous_crops({}) == {}

    def test_threshold_is_tunable(self) -> None:
        """Setting a stricter threshold should flag more crops."""
        from rallycut.tracking.relabel import detect_anomalous_crops

        # Mild outlier — flagged at strict threshold, not at lenient.
        good = self._make_normed([1.0, 0.0, 0.0, 0.0])
        mild = self._make_normed([0.7, 0.7, 0.0, 0.0])  # ~45° away
        embeddings = np.stack([good, good, good, mild])

        strict_flags = detect_anomalous_crops(
            {1: embeddings}, threshold_sigma=0.5
        )
        lenient_flags = detect_anomalous_crops(
            {1: embeddings}, threshold_sigma=10.0
        )

        assert 3 in strict_flags[1]
        assert lenient_flags[1] == []

    def test_handles_multiple_pids_independently(self) -> None:
        from rallycut.tracking.relabel import detect_anomalous_crops

        good = self._make_normed([1.0, 0.0, 0.0, 0.0])
        outlier = self._make_normed([0.0, 0.0, 0.0, 1.0])

        # pid 1 has an outlier; pid 2 is uniform.
        flags = detect_anomalous_crops({
            1: np.stack([good, good, good, outlier]),
            2: np.stack([good, good, good, good]),
        })

        assert flags[1] == [3]
        assert flags[2] == []
