"""Binary zero-shot "is this beach volleyball" classifier using open-clip.

Scores each frame against two prompts and returns the softmax prob of the
beach-VB prompt. The runtime path (`embed_and_score_frames`) imports open-clip
lazily so unit tests can pass precomputed probabilities without loading the
model.

Calibration principle (see spec §Guiding Principle): threshold is set below
the lowest-scoring positive, not at the midpoint between positives and
negatives. Favor false-accept over false-reject.
"""
from __future__ import annotations

import statistics
from typing import Any

from rallycut.quality.types import CheckResult, Issue, Tier

# Calibrated against 5 negatives + positives (see
# analysis/reports/beach_vb_calibration_<date>.json). Post-Task 10 this value
# may be refined; keep the constant as the single source of truth.
BEACH_VB_BLOCK_THRESHOLD = 0.886

# Multi-class prompts with one positive (index 0) and several negatives.
# Binary prompts ("a video that is not beach volleyball" as the second class)
# failed calibration — CLIP assigned ~98% mass to the broad "not beach
# volleyball" prompt even for real matches. Replacing with specific
# alternatives forces the softmax to compete on concrete scene content.
# See analysis/reports/beach_vb_calibration_<date>.json for the sweep.
PROMPTS = (
    "a beach volleyball match played on sand",
    "an indoor volleyball match played on a wooden court",
    "a soccer or football match on grass",
    "a group of people on a beach",
    "an arbitrary video that is not about sports",
)


def classify_is_beach_vb(per_frame_beach_vb_probs: list[float]) -> CheckResult:
    """Classify a video from its per-frame beach-VB probabilities.

    Args:
        per_frame_beach_vb_probs: softmax prob of PROMPTS[0] per frame, in [0,1].

    Returns a CheckResult. Empty input is a no-op (no issues, no metrics).
    """
    if not per_frame_beach_vb_probs:
        return CheckResult(issues=[], metrics={})

    avg = statistics.mean(per_frame_beach_vb_probs)
    metrics = {"avgBeachVbProb": avg}
    issues: list[Issue] = []

    if avg < BEACH_VB_BLOCK_THRESHOLD:
        issues.append(Issue(
            id="not_beach_volleyball",
            tier=Tier.BLOCK,
            severity=1.0 - avg,
            message="This doesn't look like a beach volleyball match. RallyCut is tuned for beach volleyball only.",
            source="preview",
            data={"avgBeachVbProb": avg},
        ))
    return CheckResult(issues=issues, metrics=metrics)


def embed_and_score_frames(frames: list[Any]) -> list[float]:
    """Run open-clip ViT-B/32 on each frame and return PROMPTS[0] softmax probs.

    `frames` is a list of PIL.Image objects. Integration-tested via
    `rallycut preview-check`, not unit-tested.
    """
    import open_clip
    import torch

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()

    text_tokens = tokenizer(list(PROMPTS))
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        image_tensors = torch.stack([preprocess(f) for f in frames])
        image_features = model.encode_image(image_tensors)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        logits = (image_features @ text_features.T) * 100.0
        probs = logits.softmax(dim=-1).cpu().numpy()

    # Column 0 is PROMPTS[0] (beach VB). Return per-frame beach-VB prob.
    return [float(row[0]) for row in probs]
