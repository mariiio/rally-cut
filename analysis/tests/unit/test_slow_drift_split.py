"""Slow-drift bisect splitter — detection signal + safety gates.

The video-frame extraction path is not unit-tested here (requires a real
video); end-to-end verification lives in `panel_verdict_per_frame.py`
runs against the 4-fixture panel.
"""
from __future__ import annotations

from dataclasses import dataclass

from rallycut.tracking import _slow_drift_split as sds


@dataclass
class FakePos:
    track_id: int
    frame_number: int
    x: float
    y: float
    width: float = 0.05
    height: float = 0.10
    confidence: float = 0.9


def _build_drifted_rally(
    drift_pid: int = 4,
    drift_track_id: int = 7,
    n_frames: int = 200,
) -> tuple[list[FakePos], dict[int, int]]:
    """Synthesize a rally where one PID's track drifts mid-rally.

    PID4 (track 7) starts at x=0.3 (left half) and drifts to x=0.9 (right
    half) — clear half-shift > 0.20. Other PIDs occupy fixed positions
    that overlap PID4's x-range to trigger the second gate.
    """
    positions: list[FakePos] = []
    half = n_frames // 2
    for f in range(n_frames):
        # PID4 (drift_track_id=7) — drifts from x=0.3 to x=0.9 across halves
        positions.append(FakePos(
            track_id=drift_track_id, frame_number=f,
            x=0.3 if f < half else 0.9,
            y=0.5,
        ))
        # PID1 (track 1) — small range near x=0.2
        positions.append(FakePos(
            track_id=1, frame_number=f, x=0.18 + 0.04 * (f / n_frames), y=0.4,
        ))
        # PID2 (track 2) — small range near x=0.5
        positions.append(FakePos(
            track_id=2, frame_number=f, x=0.48 + 0.04 * (f / n_frames), y=0.4,
        ))
        # PID3 (track 3) — broad range from 0.6 to 0.95, overlapping PID4's
        # full x-range so the pairwise-overlap gate fires.
        positions.append(FakePos(
            track_id=3, frame_number=f, x=0.6 + 0.35 * (f / n_frames), y=0.6,
        ))

    track_to_player = {1: 1, 2: 2, 3: 3, drift_track_id: drift_pid}
    return positions, track_to_player


class TestSlowDriftDetection:
    def test_detects_drift_on_synthetic_rally(self) -> None:
        positions, t2p = _build_drifted_rally()
        result = sds._detect_slow_drift_pid(positions, t2p)  # type: ignore[arg-type]
        assert result is not None
        drift_pid, parent_tid, shift = result
        assert drift_pid == 4
        assert parent_tid == 7
        # Drift was 0.3 → 0.9 horizontal, no vertical → shift = 0.6
        assert 0.55 < shift < 0.65

    def test_no_drift_on_stationary_rally(self) -> None:
        """All PIDs at fixed positions → no half-shift."""
        positions: list[FakePos] = []
        t2p = {1: 1, 2: 2, 3: 3, 4: 4}
        for f in range(100):
            positions.append(FakePos(track_id=1, frame_number=f, x=0.1, y=0.4))
            positions.append(FakePos(track_id=2, frame_number=f, x=0.4, y=0.4))
            positions.append(FakePos(track_id=3, frame_number=f, x=0.6, y=0.6))
            positions.append(FakePos(track_id=4, frame_number=f, x=0.9, y=0.6))
        assert sds._detect_slow_drift_pid(positions, t2p) is None  # type: ignore[arg-type]

    def test_no_drift_when_xrange_overlap_low(self) -> None:
        """One PID drifts but no other PID overlaps its x-range → gate
        rejects."""
        positions: list[FakePos] = []
        t2p = {1: 1, 2: 2, 3: 3, 4: 4}
        # PID4 drifts but stays in its own narrow corner (no overlap with others)
        for f in range(200):
            positions.append(FakePos(track_id=1, frame_number=f, x=0.05, y=0.4))
            positions.append(FakePos(track_id=2, frame_number=f, x=0.10, y=0.4))
            positions.append(FakePos(track_id=3, frame_number=f, x=0.15, y=0.6))
            positions.append(FakePos(
                track_id=4, frame_number=f,
                x=0.85 if f < 100 else 0.95,
                y=0.5,
            ))
        # Half-shift fires on PID4 (0.10) but x-overlap < threshold.
        result = sds._detect_slow_drift_pid(positions, t2p)  # type: ignore[arg-type]
        # Shift here is 0.10 < HALF_SHIFT_THRESHOLD=0.20 — fails first gate
        assert result is None


class TestEnabledFlag:
    def test_default_off(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        monkeypatch.delenv("ENABLE_SLOW_DRIFT_SPLIT", raising=False)
        assert sds.is_enabled() is False

    def test_enabled_when_set(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        monkeypatch.setenv("ENABLE_SLOW_DRIFT_SPLIT", "1")
        assert sds.is_enabled() is True
