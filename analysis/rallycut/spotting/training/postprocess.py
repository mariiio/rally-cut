"""Postprocessing for E2E-Spot predictions: Soft-NMS and event extraction."""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as nnf

from rallycut.spotting.config import IDX_TO_ACTION, PostprocessConfig


def soft_nms_1d(
    scores: np.ndarray,
    classes: np.ndarray,
    sigma: float = 1.0,
    window: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply Soft-NMS along the temporal axis.

    For each peak, suppress nearby scores using Gaussian decay.

    Args:
        scores: (T,) confidence scores.
        classes: (T,) predicted class indices.
        sigma: Gaussian decay parameter.
        window: Half-window size for suppression.

    Returns:
        Tuple of (suppressed_scores, classes) arrays.
    """
    scores = scores.copy()
    t = len(scores)

    # Process in order of decreasing score
    order = np.argsort(-scores)
    for idx in order:
        if scores[idx] <= 0:
            continue
        # Suppress neighbors
        lo = max(0, idx - window)
        hi = min(t, idx + window + 1)
        for j in range(lo, hi):
            if j == idx:
                continue
            dist_sq = (idx - j) ** 2
            decay = np.exp(-dist_sq / (2 * sigma ** 2))
            scores[j] *= (1.0 - decay)

    return scores, classes


def extract_events(
    logits: torch.Tensor,
    offsets: torch.Tensor,
    config: PostprocessConfig,
) -> list[dict]:
    """Convert per-frame model output to discrete event detections.

    Args:
        logits: (T, num_classes) per-frame logits.
        offsets: (T, 1) per-frame temporal offsets.
        config: Postprocessing configuration.

    Returns:
        List of {"frame": int, "action": str, "confidence": float} dicts.
    """
    probs = nnf.softmax(logits, dim=-1).cpu().numpy()  # (T, C)
    offsets_np = offsets.squeeze(-1).cpu().numpy()      # (T,)

    # Get max non-background class and its probability
    fg_probs = probs[:, 1:]  # (T, 6)
    max_fg_idx = fg_probs.argmax(axis=1)  # (T,) indices into fg classes
    max_fg_prob = fg_probs.max(axis=1)     # (T,) max fg probability

    # Class indices (1-indexed to match ACTION_TO_IDX)
    pred_classes = max_fg_idx + 1  # (T,)

    # Threshold
    mask = max_fg_prob >= config.confidence_threshold
    scores = max_fg_prob * mask

    # Soft-NMS
    scores, pred_classes_nms = soft_nms_1d(
        scores, pred_classes,
        sigma=config.nms_sigma,
        window=config.nms_window,
    )

    # Extract peaks (local maxima above threshold after NMS)
    events: list[dict] = []
    for i in range(len(scores)):
        if scores[i] < config.confidence_threshold:
            continue
        cls = int(pred_classes_nms[i])
        if cls == 0:
            continue

        # Check if this is a local maximum
        is_peak = True
        for j in range(max(0, i - 1), min(len(scores), i + 2)):
            if j != i and scores[j] > scores[i]:
                is_peak = False
                break

        if not is_peak:
            continue

        # Apply offset refinement
        refined_frame = i + offsets_np[i]
        frame = max(0, round(refined_frame))

        action = IDX_TO_ACTION.get(cls)
        if action is None:
            continue

        events.append({
            "frame": frame,
            "action": action,
            "confidence": float(scores[i]),
        })

    # Sort by frame
    events.sort(key=lambda e: e["frame"])
    return events
