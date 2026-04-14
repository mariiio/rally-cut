"""Zero-shot "is this beach volleyball" classifier using open-clip ViT-B/32.

The actual model inference is wrapped behind `embed_and_score_frames()` so
unit tests can pass precomputed probabilities without loading the model.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass

from rallycut.quality.types import CheckResult, Issue, Tier

BEACH_VB_BLOCK_THRESHOLD = 0.20  # average beach_vb prob below this = block

PROMPTS = {
    "beach_vb": "a beach volleyball match on sand",
    "indoor_vb": "an indoor volleyball match",
    "other": "a video that is not volleyball",
}


@dataclass(frozen=True)
class BeachVBProbabilities:
    beach_vb: float
    indoor_vb: float
    other: float


def classify_is_beach_vb(per_frame_probs: list[BeachVBProbabilities]) -> CheckResult:
    if not per_frame_probs:
        return CheckResult(issues=[], metrics={})

    avg_beach = statistics.mean(p.beach_vb for p in per_frame_probs)
    metrics = {"avgBeachVbProb": avg_beach}
    issues: list[Issue] = []
    if avg_beach < BEACH_VB_BLOCK_THRESHOLD:
        issues.append(Issue(
            id="wrong_angle_or_not_volleyball",
            tier=Tier.BLOCK,
            severity=1.0 - avg_beach,
            message="This doesn't look like a beach volleyball match. RallyCut is tuned for beach volleyball filmed from behind the baseline.",
            source="preflight",
            data={"avgBeachVbProb": avg_beach},
        ))
    return CheckResult(issues=issues, metrics=metrics)


def embed_and_score_frames(frames) -> list[BeachVBProbabilities]:
    """Run open-clip on each frame, return softmax over the three prompts.

    Kept out of unit tests — integration-tested via `rallycut preflight`.
    """
    import open_clip  # local import: heavy
    import torch

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()

    labels = list(PROMPTS.keys())
    text_tokens = tokenizer([PROMPTS[k] for k in labels])
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        image_tensors = torch.stack([preprocess(f) for f in frames])
        image_features = model.encode_image(image_tensors)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        logits = (image_features @ text_features.T) * 100.0
        probs = logits.softmax(dim=-1).cpu().numpy()

    out: list[BeachVBProbabilities] = []
    for row in probs:
        kv = dict(zip(labels, row))
        out.append(BeachVBProbabilities(
            beach_vb=float(kv["beach_vb"]),
            indoor_vb=float(kv["indoor_vb"]),
            other=float(kv["other"]),
        ))
    return out
