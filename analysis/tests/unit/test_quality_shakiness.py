import numpy as np

from rallycut.quality.shakiness import check_shakiness
from rallycut.quality.types import Tier


def test_static_footage_passes():
    # Identical frames → zero flow → no issue
    frames = [np.full((120, 160, 3), 128, dtype=np.uint8) for _ in range(10)]
    result = check_shakiness(frames)
    assert result.issues == []


def test_high_jitter_footage_gates():
    # Alternate black ↔ white frames → residual ≈ 1.0 (far above the 0.20 gate)
    black = np.zeros((120, 160, 3), dtype=np.uint8)
    white = np.full((120, 160, 3), 255, dtype=np.uint8)
    frames = [black if i % 2 == 0 else white for i in range(10)]
    result = check_shakiness(frames)
    issue = next(i for i in result.issues if i.id == "shaky_camera")
    assert issue.tier == Tier.GATE
