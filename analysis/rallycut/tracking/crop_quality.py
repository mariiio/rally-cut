"""Shared crop-quality gate for ReID embedding extraction.

Authoritative source of the Session-3 lock-in predicate:
bbox height ≥ 5% of frame, edges ≥ 2% from frame boundary, IoU with any
other primary track ≤ 30%. Used by the production tracker during learned-
ReID embedding extraction and by ``probe_reid_models_on_swaps`` when
scoring the held-out adversarial events.
"""

from __future__ import annotations

from rallycut.tracking.player_tracker import PlayerPosition

CROP_MIN_HEIGHT_FRAC = 0.05
CROP_EDGE_MARGIN_FRAC = 0.02
CROP_OCCLUSION_IOU = 0.30


def bbox_iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """IoU on normalized (cx, cy, w, h) bboxes."""
    ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2
    ax2, ay2 = a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2
    bx2, by2 = b[0] + b[2] / 2, b[1] + b[3] / 2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def is_quality_crop(
    pos: PlayerPosition,
    primary_positions_at_frame: list[PlayerPosition],
) -> bool:
    """Apply the crop-quality filter: size, edge margin, occlusion.

    ``primary_positions_at_frame`` is the list of other tracks visible at the
    same frame — any IoU > CROP_OCCLUSION_IOU against those rejects the crop.
    """
    if pos.height < CROP_MIN_HEIGHT_FRAC:
        return False
    left = pos.x - pos.width / 2
    right = pos.x + pos.width / 2
    top = pos.y - pos.height / 2
    bottom = pos.y + pos.height / 2
    if (left < CROP_EDGE_MARGIN_FRAC or right > 1.0 - CROP_EDGE_MARGIN_FRAC
            or top < CROP_EDGE_MARGIN_FRAC or bottom > 1.0 - CROP_EDGE_MARGIN_FRAC):
        return False
    for other in primary_positions_at_frame:
        if other.track_id == pos.track_id:
            continue
        iou = bbox_iou(
            (pos.x, pos.y, pos.width, pos.height),
            (other.x, other.y, other.width, other.height),
        )
        if iou > CROP_OCCLUSION_IOU:
            return False
    return True
