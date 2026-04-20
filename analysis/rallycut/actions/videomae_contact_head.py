"""MSTCN-based 7-class contact-spotting head on VideoMAE features.

Phase 3 of the VideoMAE contact validation plan. Reuses the MS-TCN++ model
from ``rallycut.temporal.ms_tcn.model`` with a small reconfig:

- feature_dim = 768 (VideoMAE CLS)
- num_classes = 7 (background + serve, receive, set, attack, dig, block)
- hidden_dim = 64 (small — 68 LOO folds, ~2k positives per class)
- 2 refinement stages × 8 dilated layers per stage (receptive field ~2048
  effective-fps frames = ~70s at stride=4).

The training script (``scripts/train_videomae_contact_head.py``) drives
LOO-per-video training with focal-like class weights + soft Gaussian
targets, then decodes per-frame class probs into discrete contact events
via per-class peak-NMS.
"""

from __future__ import annotations

from rallycut.temporal.ms_tcn.model import MSTCN, MSTCNConfig

# Canonical 7-class ordering for this head. Index 0 = background, the
# remaining 6 match rallycut.actions.trajectory_features.ACTION_TO_IDX
# ordering so downstream scoring via eval_action_detection.compute_metrics
# works without translation.
CONTACT_CLASSES = ["background", "serve", "receive", "set", "attack", "dig", "block"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CONTACT_CLASSES)}
NUM_CLASSES = len(CONTACT_CLASSES)


def build_contact_head(
    feature_dim: int = 768,
    num_stages: int = 2,
    num_layers: int = 8,
    hidden_dim: int = 64,
    dropout: float = 0.3,
) -> MSTCN:
    """Create a fresh MSTCN configured for 7-class contact spotting.

    Defaults target ~200K params — small enough to not overfit the 68-video
    LOO corpus, large enough to capture the temporal dilation receptive
    field needed to span a rally.

    Args:
        feature_dim: Input feature dimension (768 for VideoMAE v1 CLS).
        num_stages: Number of MSTCN refinement stages.
        num_layers: Dilated layers per stage. Effective receptive field is
            ~2^num_layers frames, so 8 = ~256 frames context at stride=4.
        hidden_dim: Internal channel count.
        dropout: Per-layer dropout.

    Returns:
        An MSTCN ready to train; output shape ``(batch, 7, T)`` logits.
    """
    config = MSTCNConfig(
        feature_dim=feature_dim,
        num_stages=num_stages,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        num_classes=NUM_CLASSES,
        dropout=dropout,
        ball_feature_dim=0,
    )
    return MSTCN(config)
