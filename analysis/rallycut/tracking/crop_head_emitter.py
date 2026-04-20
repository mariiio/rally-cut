"""Inference-time crop-head emitter.

Adapts :class:`rallycut.ml.crop_head.model.CropHeadModel` to the
``predict_proba(candidates) -> (N, 2)`` protocol that
:func:`rallycut.tracking.decoder_runtime.run_decoder_over_rally`
consumes in its generalized emitter path (Task 0 of the Phase 2 plan).

At inference time, crops are extracted on-the-fly from the source video
so the training and inference distributions align. The bbox-selection
logic mirrors ``scripts/extract_crop_dataset.py`` —
``_nearest_track_id_at_frame`` at the candidate frame with a ±2f
nearest-neighbor fallback.

The torch model is exposed as ``self._torch_model`` (not ``self.model``)
so the decoder runtime can disambiguate new-style emitters from legacy
:class:`ContactClassifier` instances via ``hasattr(clf, "model")``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import torch

from rallycut.ml.crop_head.model import CropHeadModel
from rallycut.tracking.ball_tracker import BallPosition as BallPos
from rallycut.tracking.player_tracker import PlayerPosition as PlayerPos

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    pass


# Must match scripts/extract_crop_dataset.py constants — the checkpoint
# was trained with these exact pixel dimensions and padding fractions.
PLAYER_CROP_SIZE = 64
BALL_PATCH_SIZE = 32
BALL_PATCH_HALF_NORM = 0.04
PLAYER_PAD_FRAC = 0.10


@dataclass
class _Checkpoint:
    state_dict: dict
    input_kind: str
    pool_kind: str
    backbone_kind: str
    t_window: int


class CropHeadContactClassifier:
    """Per-rally emitter that scores contact candidates via a crop-head.

    ``predict_proba(candidates)`` returns an ``(N, 2)`` array where column
    0 is ``P(background)`` and column 1 is ``P(contact)`` — matching
    sklearn's binary ``predict_proba`` contract so the decoder runtime
    can slice ``[:, 1]`` uniformly across emitters.

    The emitter requires ``video_path`` + ``rally_start_frame`` +
    position lookups at construction so it can extract a T-frame
    window centered on each candidate.

    Note: the torch model is intentionally stored as ``self._torch_model``
    (not ``self.model``) so ``run_decoder_over_rally`` can distinguish
    new-style emitters (no ``.model`` attribute) from legacy
    :class:`ContactClassifier` instances (sklearn GBM at ``.model``).
    """

    # Exposed so the decoder runtime can skip padding logic when feeding
    # us candidates directly (we don't consume the 26-dim feature matrix).
    is_trained: bool

    def __init__(
        self,
        checkpoint_path: Path,
        video_path: Path,
        rally_start_frame: int,
        ball_positions: list[BallPos],
        player_positions: list[PlayerPos],
        device: str = "auto",
    ):
        self.video_path = Path(video_path)
        self.rally_start_frame = int(rally_start_frame)
        self.ball_positions = list(ball_positions)
        self.player_positions = list(player_positions)
        self.device = self._resolve_device(device)
        self._ckpt = self._load_checkpoint(Path(checkpoint_path))
        self._torch_model = CropHeadModel(
            input_kind=self._ckpt.input_kind,
            pool_kind=self._ckpt.pool_kind,
            backbone_kind=self._ckpt.backbone_kind,
        ).to(self.device)
        self._torch_model.load_state_dict(self._ckpt.state_dict)
        self._torch_model.eval()
        self.is_trained = True

    # ---------------------- construction helpers ----------------------

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return device

    @staticmethod
    def _load_checkpoint(path: Path) -> _Checkpoint:
        raw = torch.load(path, map_location="cpu", weights_only=False)
        return _Checkpoint(
            state_dict=raw["state_dict"],
            input_kind=raw.get("input_kind", "concat"),
            pool_kind=raw.get("pool_kind", "mean"),
            backbone_kind=raw.get("backbone_kind", "frozen"),
            t_window=int(raw.get("t_window", 9)),
        )

    # ---------------------- emission protocol ----------------------

    def predict_proba(self, candidates: list[Any]) -> np.ndarray:
        """Score ``candidates`` via crop-head. Returns ``(N, 2)`` probs.

        Column 0 is ``P(background)``, column 1 is ``P(contact)``.
        ``candidates`` is a list of ``CandidateFeatures`` objects; only
        their ``frame`` attribute is used (the 26-dim feature vector is
        ignored — we extract crops directly from the source video).
        """
        if not candidates:
            return np.zeros((0, 2), dtype=np.float64)

        # Build per-frame ball lookup (matches extract_crop_dataset.py)
        ball_by_frame: dict[int, tuple[float, float]] = {}
        for ball_obs in self.ball_positions:
            if ball_obs.x > 0 or ball_obs.y > 0:
                ball_by_frame[ball_obs.frame_number] = (ball_obs.x, ball_obs.y)

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video {self.video_path}")
        try:
            crops_pc: list[np.ndarray] = []
            crops_bp: list[np.ndarray] = []
            for c in candidates:
                p_crop, b_patch = self._extract_for_frame(
                    cap, int(c.frame), ball_by_frame
                )
                crops_pc.append(p_crop)
                crops_bp.append(b_patch)
        finally:
            cap.release()

        pc_tensor = torch.tensor(
            np.stack(crops_pc), dtype=torch.float32, device=self.device
        )
        bp_tensor = torch.tensor(
            np.stack(crops_bp), dtype=torch.float32, device=self.device
        )

        with torch.no_grad():
            logits = self._torch_model(pc_tensor, bp_tensor)
            probs = torch.sigmoid(logits).cpu().numpy().astype(np.float64)

        out = np.zeros((len(candidates), 2), dtype=np.float64)
        out[:, 0] = 1.0 - probs
        out[:, 1] = probs
        return out

    # ---------------------- crop extraction ----------------------

    def _extract_for_frame(
        self,
        cap: cv2.VideoCapture,
        candidate_rally_frame: int,
        ball_by_frame: dict[int, tuple[float, float]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract a ``(T, 3, 64, 64)`` player window and ``(T, 3, 32, 32)``
        ball window for one candidate. Mirrors
        ``scripts/extract_crop_dataset.py::_extract_sample``.

        Edge padding: frames before rally start are left as zeros — the
        training distribution had the same behavior for candidates within
        ``T//2`` of the rally start (and is bounded by VideoCapture read
        failures near the rally end).
        """
        t_window = self._ckpt.t_window
        half = t_window // 2

        # Nearest-to-ball track at the candidate frame (±2f fallback).
        track_id = self._nearest_track_id_at_frame(
            candidate_rally_frame, ball_by_frame
        )
        # Precompute bbox lookup for that track (rally-local frame → bbox).
        bbox_by_frame: dict[int, tuple[float, float, float, float]] = {}
        if track_id is not None:
            for p in self.player_positions:
                if p.track_id == track_id:
                    bbox_by_frame[p.frame_number] = (p.x, p.y, p.width, p.height)

        player_crops = np.zeros(
            (t_window, 3, PLAYER_CROP_SIZE, PLAYER_CROP_SIZE), dtype=np.float32
        )
        ball_patches = np.zeros(
            (t_window, 3, BALL_PATCH_SIZE, BALL_PATCH_SIZE), dtype=np.float32
        )

        for i, offset in enumerate(range(-half, half + 1)):
            rally_f = candidate_rally_frame + offset
            if rally_f < 0:
                continue
            abs_f = self.rally_start_frame + rally_f
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(abs_f))
            ok, img = cap.read()
            if not ok or img is None:
                continue

            # Player bbox at rally_f with ±2f nearest-neighbor fallback.
            bbox: tuple[float, float, float, float] | None = bbox_by_frame.get(rally_f)
            if bbox is None:
                for d in (1, -1, 2, -2):
                    bbox = bbox_by_frame.get(rally_f + d)
                    if bbox is not None:
                        break
            if bbox is not None:
                crop_hwc = _crop_player(img, bbox)
                player_crops[i] = crop_hwc.transpose(2, 0, 1)

            # Ball position at rally_f with ±2f nearest-neighbor fallback.
            ball: tuple[float, float] | None = ball_by_frame.get(rally_f)
            if ball is None:
                for d in (1, -1, 2, -2):
                    ball = ball_by_frame.get(rally_f + d)
                    if ball is not None:
                        break
            if ball is not None:
                patch_hwc = _crop_ball(img, ball)
                ball_patches[i] = patch_hwc.transpose(2, 0, 1)

        return player_crops, ball_patches

    def _nearest_track_id_at_frame(
        self,
        frame: int,
        ball_by_frame: dict[int, tuple[float, float]],
        window: int = 2,
    ) -> int | None:
        """Match extract_crop_dataset.py::_nearest_track_id_at_frame."""
        # Prefer exact frame ball; otherwise nearest within ±window.
        ball_xy: tuple[float, float] | None = ball_by_frame.get(frame)
        if ball_xy is None:
            for d in (1, -1, 2, -2):
                ball_xy = ball_by_frame.get(frame + d)
                if ball_xy is not None:
                    break
        if ball_xy is None:
            return None
        candidates: list[tuple[float, int]] = []
        for p in self.player_positions:
            if abs(p.frame_number - frame) <= window:
                dx = p.x - ball_xy[0]
                dy = p.y - ball_xy[1]
                d = (dx * dx + dy * dy) ** 0.5
                candidates.append((d, p.track_id))
        if not candidates:
            return None
        candidates.sort()
        return candidates[0][1]


# ---------------------- crop helpers (bgr numpy in, float HWC out) ----------------------


def _crop_player(
    img: np.ndarray, bbox_xywh_norm: tuple[float, float, float, float]
) -> np.ndarray:
    h, w = img.shape[:2]
    cx, cy, bw, bh = bbox_xywh_norm
    pad_w = bw * PLAYER_PAD_FRAC
    pad_h = bh * PLAYER_PAD_FRAC
    x0 = max(0, int((cx - bw / 2 - pad_w) * w))
    x1 = min(w, int((cx + bw / 2 + pad_w) * w))
    y0 = max(0, int((cy - bh / 2 - pad_h) * h))
    y1 = min(h, int((cy + bh / 2 + pad_h) * h))
    if x1 <= x0 or y1 <= y0:
        return np.zeros((PLAYER_CROP_SIZE, PLAYER_CROP_SIZE, 3), dtype=np.float32)
    crop = img[y0:y1, x0:x1]
    crop = cv2.resize(crop, (PLAYER_CROP_SIZE, PLAYER_CROP_SIZE))
    return crop.astype(np.float32) / 255.0


def _crop_ball(img: np.ndarray, ball_xy_norm: tuple[float, float]) -> np.ndarray:
    h, w = img.shape[:2]
    bx, by = ball_xy_norm
    x0 = max(0, int((bx - BALL_PATCH_HALF_NORM) * w))
    x1 = min(w, int((bx + BALL_PATCH_HALF_NORM) * w))
    y0 = max(0, int((by - BALL_PATCH_HALF_NORM) * h))
    y1 = min(h, int((by + BALL_PATCH_HALF_NORM) * h))
    if x1 <= x0 or y1 <= y0:
        return np.zeros((BALL_PATCH_SIZE, BALL_PATCH_SIZE, 3), dtype=np.float32)
    patch = img[y0:y1, x0:x1]
    patch = cv2.resize(patch, (BALL_PATCH_SIZE, BALL_PATCH_SIZE))
    return patch.astype(np.float32) / 255.0
