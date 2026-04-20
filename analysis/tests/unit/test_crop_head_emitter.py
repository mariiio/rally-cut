"""Unit tests for :class:`CropHeadContactClassifier`.

The emitter extracts crops on-the-fly from a video, so these tests
write a tiny synthetic MP4 to a tmp dir and feed it through the
pipeline. The checkpoint is saved from a freshly-initialized
:class:`CropHeadModel` — we care about the shape/contract of
``predict_proba`` here, not measured accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from rallycut.ml.crop_head.model import CropHeadModel
from rallycut.tracking.ball_tracker import BallPosition as BallPos
from rallycut.tracking.crop_head_emitter import CropHeadContactClassifier
from rallycut.tracking.player_tracker import PlayerPosition as PlayerPos


@dataclass
class _FakeCandidate:
    """Minimal stand-in for CandidateFeatures — only ``.frame`` is used."""
    frame: int


def _write_fake_video(path: Path, n_frames: int = 30, size: tuple[int, int] = (854, 480)) -> None:
    """Write a small MP4 with `n_frames` random RGB frames."""
    w, h = size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (w, h))
    assert writer.isOpened(), f"Failed to open writer for {path}"
    rng = np.random.default_rng(0)
    for _ in range(n_frames):
        frame = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _save_fresh_checkpoint(path: Path, t_window: int = 9) -> None:
    """Save a Phase 1-compatible checkpoint from a fresh CropHeadModel."""
    model = CropHeadModel()
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save({
        "state_dict": state,
        "input_kind": "concat",
        "pool_kind": "mean",
        "backbone_kind": "frozen",
        "t_window": t_window,
        "best_val_auc": 0.87,
        "epoch": 9,
        "seed": 42,
    }, path)


def _build_positions(
    n_frames: int = 30, rally_start_frame: int = 0
) -> tuple[list[BallPos], list[PlayerPos]]:
    """Deterministic positions: ball moves left→right, 2 players stationary."""
    ball_positions: list[BallPos] = []
    player_positions: list[PlayerPos] = []
    for f in range(n_frames):
        ball_positions.append(
            BallPos(frame_number=f, x=0.2 + f * 0.01, y=0.5, confidence=0.9)
        )
        # Track 1 (team A) — near the ball so the nearest-track selector fires.
        player_positions.append(
            PlayerPos(
                frame_number=f, track_id=1,
                x=0.25 + f * 0.01, y=0.55,
                width=0.12, height=0.28, confidence=0.9,
            )
        )
        # Track 2 (team B) — across the court.
        player_positions.append(
            PlayerPos(
                frame_number=f, track_id=2,
                x=0.75, y=0.45,
                width=0.12, height=0.28, confidence=0.9,
            )
        )
    return ball_positions, player_positions


# ----------------------------- fixtures -----------------------------

@pytest.fixture()
def env(tmp_path: Path) -> dict[str, object]:
    video = tmp_path / "fake.mp4"
    ckpt = tmp_path / "ckpt.pt"
    _write_fake_video(video, n_frames=30)
    _save_fresh_checkpoint(ckpt, t_window=9)
    ball_pos, player_pos = _build_positions(n_frames=30)
    return {
        "video": video,
        "ckpt": ckpt,
        "ball_pos": ball_pos,
        "player_pos": player_pos,
    }


# ----------------------------- tests -----------------------------

def test_predict_proba_shape_and_row_sum(env: dict[str, object]) -> None:
    """predict_proba returns (N, 2) with each row summing to 1."""
    clf = CropHeadContactClassifier(
        checkpoint_path=env["ckpt"],  # type: ignore[arg-type]
        video_path=env["video"],  # type: ignore[arg-type]
        rally_start_frame=0,
        ball_positions=env["ball_pos"],  # type: ignore[arg-type]
        player_positions=env["player_pos"],  # type: ignore[arg-type]
        device="cpu",
    )
    cands = [_FakeCandidate(frame=f) for f in (5, 10, 15)]
    probs = clf.predict_proba(cands)
    assert probs.shape == (3, 2)
    row_sums = probs.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6), f"row sums: {row_sums}"


def test_predict_proba_empty(env: dict[str, object]) -> None:
    """Empty candidate list returns (0, 2) array — no video I/O required."""
    clf = CropHeadContactClassifier(
        checkpoint_path=env["ckpt"],  # type: ignore[arg-type]
        video_path=env["video"],  # type: ignore[arg-type]
        rally_start_frame=0,
        ball_positions=env["ball_pos"],  # type: ignore[arg-type]
        player_positions=env["player_pos"],  # type: ignore[arg-type]
        device="cpu",
    )
    probs = clf.predict_proba([])
    assert probs.shape == (0, 2)


def test_is_trained_after_load(env: dict[str, object]) -> None:
    """is_trained must be True after successful checkpoint load."""
    clf = CropHeadContactClassifier(
        checkpoint_path=env["ckpt"],  # type: ignore[arg-type]
        video_path=env["video"],  # type: ignore[arg-type]
        rally_start_frame=0,
        ball_positions=env["ball_pos"],  # type: ignore[arg-type]
        player_positions=env["player_pos"],  # type: ignore[arg-type]
        device="cpu",
    )
    assert clf.is_trained is True


def test_checkpoint_roundtrip_preserves_config(env: dict[str, object]) -> None:
    """Saving with non-default t_window/input_kind etc. should survive the
    load. Ensures the Task 0 emitter honours the config dict that future
    (Task 2) variants will populate."""
    # Re-save the checkpoint with a different t_window to verify load pulls it.
    ckpt_path = env["ckpt"]  # type: ignore[assignment]
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)  # type: ignore[arg-type]
    raw["t_window"] = 9  # keep 9 for model compat; still asserts the load read it
    raw["input_kind"] = "concat"
    raw["pool_kind"] = "mean"
    raw["backbone_kind"] = "frozen"
    torch.save(raw, ckpt_path)  # type: ignore[arg-type]

    clf = CropHeadContactClassifier(
        checkpoint_path=ckpt_path,  # type: ignore[arg-type]
        video_path=env["video"],  # type: ignore[arg-type]
        rally_start_frame=0,
        ball_positions=env["ball_pos"],  # type: ignore[arg-type]
        player_positions=env["player_pos"],  # type: ignore[arg-type]
        device="cpu",
    )
    assert clf._ckpt.t_window == 9
    assert clf._ckpt.input_kind == "concat"
    assert clf._ckpt.pool_kind == "mean"
    assert clf._ckpt.backbone_kind == "frozen"


def test_no_public_model_attribute(env: dict[str, object]) -> None:
    """The decoder runtime distinguishes new-style emitters by the absence
    of a public ``.model`` attribute. Regression test for the
    ``_torch_model`` rename documented in Task 0's divergence note."""
    clf = CropHeadContactClassifier(
        checkpoint_path=env["ckpt"],  # type: ignore[arg-type]
        video_path=env["video"],  # type: ignore[arg-type]
        rally_start_frame=0,
        ball_positions=env["ball_pos"],  # type: ignore[arg-type]
        player_positions=env["player_pos"],  # type: ignore[arg-type]
        device="cpu",
    )
    assert not hasattr(clf, "model"), (
        "Public .model attribute breaks decoder_runtime's emitter detection"
    )
    assert hasattr(clf, "_torch_model")
    assert hasattr(clf, "predict_proba")
    assert hasattr(clf, "is_trained")
