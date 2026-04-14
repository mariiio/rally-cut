import numpy as np

from rallycut.quality.shakiness import check_shakiness
from rallycut.quality.types import Tier


def test_static_footage_passes():
    # Identical frames → zero flow → no issue
    frames = [np.full((120, 160, 3), 128, dtype=np.uint8) for _ in range(10)]
    result = check_shakiness(frames)
    assert result.issues == []


def test_high_jitter_footage_gates():
    rng = np.random.default_rng(42)
    # Random noise per frame → huge frame-to-frame residual
    frames = [rng.integers(0, 255, size=(120, 160, 3), dtype=np.uint8) for _ in range(10)]
    result = check_shakiness(frames)
    issue = next(i for i in result.issues if i.id == "shaky_camera")
    assert issue.tier == Tier.GATE
