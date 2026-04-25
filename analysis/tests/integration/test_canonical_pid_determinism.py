"""Determinism test for `compute_canonical_pid_map`.

Pins the plan-doc invariant for the ref-crop canonical-identity workstream
(``docs/superpowers/plans/2026-04-25-ref-crop-canonical-identity.md``):

    Run match-players twice on a fixture with full crops, assert
    canonicalPidMapJson is bit-identical.

Without this guarantee the entire premise (canonicalPidMapJson is the single
source of truth for downstream pid display) collapses — re-runs would
silently shuffle pids again, defeating the column's reason to exist.

Fixtures: ``tata`` + ``lulu``. Both labeled at 100% on click-GT per
``memory/player_attribution_day4_2026_04_23.md``, so any drift surfaces
immediately rather than getting lost in measurement noise. Skipped when the
DB has no rallies for the fixture (CI without a populated DB).
"""
from __future__ import annotations

from typing import Any

import pytest

# Mark slow: each fixture runs DINOv2 inference per primary track per rally.
pytestmark = pytest.mark.slow


FIXTURE_VIDEO_IDS = {
    "tata": "7d77980f-3006-40e0-adc0-db491a5bb659",
    "lulu": "4f2bd66a-61a1-49ac-8137-fd2576e0e851",
}


def _load_anchors_and_rallies(video_id: str) -> tuple[Any, Any, Any]:
    """Load IdentityAnchors + rallies for a fixture from the live DB.

    Returns (anchors, rallies, video_path) or raises pytest.skip if any
    prerequisite (DB connection, rallies, full ref-crop set, video file)
    is missing.
    """
    try:
        from rallycut.evaluation.db import get_connection
        from rallycut.evaluation.tracking.db import (
            get_video_path,
            load_rallies_for_video,
        )
        from rallycut.tracking.crop_guided_identity import build_anchors_from_crops
    except ImportError as exc:
        pytest.skip(f"missing import: {exc}")

    import cv2
    import numpy as np

    try:
        conn = get_connection()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"DB unavailable: {exc}")

    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT player_id, frame_ms, bbox_x, bbox_y, bbox_w, bbox_h
                   FROM player_reference_crops WHERE video_id = %s
                   ORDER BY player_id, created_at""",
                [video_id],
            )
            crop_rows = cur.fetchall()

    if not crop_rows or {int(r[0]) for r in crop_rows} != {1, 2, 3, 4}:
        pytest.skip(f"video {video_id[:8]} lacks full 4-pid crop set")

    rallies = load_rallies_for_video(video_id)
    if not rallies:
        pytest.skip(f"video {video_id[:8]} has no tracked rallies")

    video_path = get_video_path(video_id)
    if video_path is None or not video_path.exists():
        pytest.skip(f"video {video_id[:8]} file not resolvable")

    from rallycut.tracking.player_features import extract_bbox_crop

    cap = cv2.VideoCapture(str(video_path))
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    bgr_by_pid: dict[int, list[Any]] = {}
    for r in crop_rows:
        pid, fms = int(r[0]), int(r[1])
        bx, by, bw, bh = float(r[2]), float(r[3]), float(r[4]), float(r[5])
        cap.set(cv2.CAP_PROP_POS_MSEC, float(fms))
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        # Bboxes are normalized center-format per extract_bbox_crop's contract;
        # mirror the CLI loader so the prototypes are byte-identical.
        crop = extract_bbox_crop(
            np.asarray(frame, dtype=np.uint8), (bx, by, bw, bh), fw, fh,
        )
        if crop is not None:
            bgr_by_pid.setdefault(pid, []).append(crop)
    cap.release()

    if set(bgr_by_pid.keys()) != {1, 2, 3, 4}:
        pytest.skip(f"video {video_id[:8]} crop extraction incomplete")

    anchors = build_anchors_from_crops(bgr_by_pid, source="user")
    if len(anchors.prototypes) != 4:
        pytest.skip(f"video {video_id[:8]} anchor build dropped a pid")

    return anchors, rallies, video_path


@pytest.mark.parametrize("fixture_name", list(FIXTURE_VIDEO_IDS.keys()))
def test_compute_canonical_pid_map_is_bit_identical(fixture_name: str) -> None:
    from rallycut.tracking.match_tracker import compute_canonical_pid_map

    video_id = FIXTURE_VIDEO_IDS[fixture_name]
    anchors, rallies, video_path = _load_anchors_and_rallies(video_id)

    map_a = compute_canonical_pid_map(video_path, rallies, anchors)
    map_b = compute_canonical_pid_map(video_path, rallies, anchors)

    assert map_a == map_b, (
        f"fixture {fixture_name}: canonical pid map differs across runs. "
        f"This breaks the ref-crop canonical-identity contract — see "
        f"docs/superpowers/plans/2026-04-25-ref-crop-canonical-identity.md."
    )
    assert map_a, f"fixture {fixture_name}: canonical pid map is empty"
