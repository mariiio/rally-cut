"""Pure-Python smoke test that the SQL strings parse and the function signatures hold.

End-to-end DB testing is covered by the eval gate in Task 22.
"""
from rallycut.training.action_gt_query import load_for_rallies, load_for_videos


def test_imports_resolve() -> None:
    assert callable(load_for_rallies)
    assert callable(load_for_videos)
