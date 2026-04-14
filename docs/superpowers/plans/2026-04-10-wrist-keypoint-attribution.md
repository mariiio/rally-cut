# Wrist Keypoint Attribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace bbox centroid with wrist keypoint distance for player attribution, gaining +2.5pp attribution accuracy.

**Architecture:** Extract a shared `_player_to_ball_dist()` helper that computes wrist-keypoint distance when pose data is available, falling back to the existing bbox upper-quarter distance. Both `_find_nearest_player()` and `_find_nearest_players()` call this helper. The distance value itself changes (wrist vs bbox), which is the correct thing to do since it better reflects actual contact physics.

**Tech Stack:** Python, COCO 17-keypoint pose data (indices 9/10 = left/right wrist), existing PlayerPosition.keypoints field.

---

### Task 1: Add `_player_to_ball_dist()` helper and update `_find_nearest_player()`

**Files:**
- Modify: `analysis/rallycut/tracking/contact_detector.py:445-482`
- Test: `analysis/tests/unit/test_contact_detector.py`

- [ ] **Step 1: Write tests for wrist-based distance**

Add to `analysis/tests/unit/test_contact_detector.py`:

```python
from rallycut.tracking.contact_detector import _player_to_ball_dist


def _pp_with_pose(
    frame: int, track_id: int, x: float, y: float,
    keypoints: list[list[float]] | None = None,
) -> PlayerPosition:
    """Helper to create a PlayerPosition with optional pose keypoints."""
    return PlayerPosition(
        frame_number=frame, track_id=track_id,
        x=x, y=y, width=0.05, height=0.15, confidence=0.9,
        keypoints=keypoints,
    )


def _make_coco_keypoints(
    left_wrist: tuple[float, float, float] = (0.0, 0.0, 0.0),
    right_wrist: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> list[list[float]]:
    """Build a 17-keypoint COCO array with only wrists specified."""
    kpts = [[0.0, 0.0, 0.0]] * 17
    kpts = [list(k) for k in kpts]  # make mutable copies
    kpts[9] = list(left_wrist)
    kpts[10] = list(right_wrist)
    return kpts


class TestPlayerToBallDist:
    """Tests for _player_to_ball_dist helper."""

    def test_uses_wrist_when_confident(self) -> None:
        """With high-confidence wrist keypoints, distance uses wrist position."""
        kpts = _make_coco_keypoints(
            left_wrist=(0.5, 0.5, 0.9),   # exactly at ball
            right_wrist=(0.3, 0.3, 0.9),  # farther
        )
        p = _pp_with_pose(10, 1, 0.6, 0.7, keypoints=kpts)
        dist = _player_to_ball_dist(p, 0.5, 0.5)
        # Left wrist is at (0.5, 0.5) = ball position, dist should be ~0
        assert dist < 0.01

    def test_falls_back_to_bbox_without_keypoints(self) -> None:
        """Without keypoints, uses bbox upper-quarter."""
        p = _pp_with_pose(10, 1, 0.5, 0.55, keypoints=None)
        dist = _player_to_ball_dist(p, 0.5, 0.5)
        # bbox upper-quarter: y = 0.55 - 0.15*0.25 = 0.5125
        # dist = sqrt(0 + 0.0125^2) ≈ 0.0125
        assert 0.01 < dist < 0.02

    def test_falls_back_to_bbox_with_low_confidence(self) -> None:
        """Low-confidence wrists fall back to bbox."""
        kpts = _make_coco_keypoints(
            left_wrist=(0.5, 0.5, 0.1),   # low conf
            right_wrist=(0.5, 0.5, 0.1),  # low conf
        )
        p = _pp_with_pose(10, 1, 0.5, 0.55, keypoints=kpts)
        dist_with_low_conf = _player_to_ball_dist(p, 0.5, 0.5)
        p_no_kpts = _pp_with_pose(10, 1, 0.5, 0.55, keypoints=None)
        dist_no_kpts = _player_to_ball_dist(p_no_kpts, 0.5, 0.5)
        assert abs(dist_with_low_conf - dist_no_kpts) < 1e-9

    def test_picks_closer_wrist(self) -> None:
        """Uses the wrist closer to the ball."""
        kpts = _make_coco_keypoints(
            left_wrist=(0.1, 0.1, 0.9),   # far from ball at (0.5, 0.5)
            right_wrist=(0.48, 0.48, 0.9),  # near ball
        )
        p = _pp_with_pose(10, 1, 0.3, 0.3, keypoints=kpts)
        dist = _player_to_ball_dist(p, 0.5, 0.5)
        # Should use right wrist: sqrt((0.5-0.48)^2 + (0.5-0.48)^2) ≈ 0.028
        assert dist < 0.04


class TestFindNearestPlayerWithPose:
    """Tests that _find_nearest_player uses wrist keypoints when available."""

    def test_wrist_beats_bbox_centroid(self) -> None:
        """Player whose bbox is farther but wrist is closer should win."""
        # Player 1: bbox far, but wrist right at ball
        kpts1 = _make_coco_keypoints(
            left_wrist=(0.5, 0.5, 0.9),
            right_wrist=(0.4, 0.4, 0.9),
        )
        p1 = _pp_with_pose(10, 1, 0.7, 0.7, keypoints=kpts1)  # bbox center far

        # Player 2: bbox close, no keypoints
        p2 = _pp_with_pose(10, 2, 0.51, 0.55, keypoints=None)  # bbox near ball

        track_id, dist, _ = _find_nearest_player(10, 0.5, 0.5, [p1, p2])
        assert track_id == 1  # wrist proximity wins over bbox proximity
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd analysis && uv run pytest tests/unit/test_contact_detector.py::TestPlayerToBallDist -v && uv run pytest tests/unit/test_contact_detector.py::TestFindNearestPlayerWithPose -v`
Expected: FAIL — `_player_to_ball_dist` not defined, and `test_wrist_beats_bbox_centroid` would pick player 2.

- [ ] **Step 3: Implement `_player_to_ball_dist()` and update `_find_nearest_player()`**

In `analysis/rallycut/tracking/contact_detector.py`, add the helper before `_find_nearest_player()` (around line 445):

```python
# COCO keypoint indices for wrists
_KPT_LEFT_WRIST = 9
_KPT_RIGHT_WRIST = 10
_MIN_WRIST_CONF = 0.3  # Minimum keypoint confidence to use wrist position


def _player_to_ball_dist(
    player: PlayerPosition,
    ball_x: float,
    ball_y: float,
) -> float:
    """Image-space distance from player to ball.

    Uses the closer wrist keypoint when pose data is available with
    sufficient confidence. Falls back to bbox upper-quarter (torso/arms)
    when keypoints are absent or low-confidence.

    Wrist distance is a +2.5pp improvement over bbox centroid for
    attribution (diagnostic: scripts/diagnose_keypoint_attribution.py).
    Volleyball contacts happen with hands/arms, so wrist position is a
    better proxy for who is touching the ball.
    """
    # Try wrist keypoints first
    if player.keypoints is not None and len(player.keypoints) > _KPT_RIGHT_WRIST:
        best_wrist_dist = float("inf")
        for kpt_idx in (_KPT_LEFT_WRIST, _KPT_RIGHT_WRIST):
            kx, ky, kc = player.keypoints[kpt_idx]
            if kc >= _MIN_WRIST_CONF:
                d = math.sqrt((ball_x - kx) ** 2 + (ball_y - ky) ** 2)
                if d < best_wrist_dist:
                    best_wrist_dist = d
        if best_wrist_dist < float("inf"):
            return best_wrist_dist

    # Fallback: bbox upper-quarter
    player_x = player.x
    player_y = player.y - player.height * 0.25
    return math.sqrt((ball_x - player_x) ** 2 + (ball_y - player_y) ** 2)
```

Then update `_find_nearest_player()` to use the helper:

```python
def _find_nearest_player(
    frame: int,
    ball_x: float,
    ball_y: float,
    player_positions: list[PlayerPosition],
    search_frames: int = 5,
) -> tuple[int, float, float]:
    """Find nearest player to ball at given frame.

    Uses wrist keypoint distance when pose data is available, falling
    back to bbox upper-quarter distance. See _player_to_ball_dist().

    Returns:
        (track_id, distance, player_center_y). track_id=-1 if no player found.
        player_center_y is the bbox center Y (for court side determination).
    """
    best_track_id = -1
    best_dist = float("inf")
    best_player_y = 0.5

    for p in player_positions:
        if abs(p.frame_number - frame) > search_frames:
            continue

        dist = _player_to_ball_dist(p, ball_x, ball_y)

        if dist < best_dist:
            best_dist = dist
            best_track_id = p.track_id
            best_player_y = p.y  # bbox center Y for court side

    return best_track_id, best_dist, best_player_y
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd analysis && uv run pytest tests/unit/test_contact_detector.py -v`
Expected: All pass including existing tests (`test_finds_closest_player`, `test_no_players_returns_default`, `test_respects_frame_window`).

- [ ] **Step 5: Commit**

```bash
git add analysis/rallycut/tracking/contact_detector.py analysis/tests/unit/test_contact_detector.py
git commit -m "contact_detector: use wrist keypoints for nearest-player attribution

Replace bbox upper-quarter centroid with wrist keypoint distance when
pose data is available (conf >= 0.3). Falls back to bbox when keypoints
are absent. Diagnostic showed +2.5pp attribution accuracy (dig +6.7pp,
attack +3.5pp)."
```

---

### Task 2: Update `_find_nearest_players()` to use wrist distance

**Files:**
- Modify: `analysis/rallycut/tracking/contact_detector.py:518-568`

- [ ] **Step 1: Update `_find_nearest_players()` to use `_player_to_ball_dist()`**

```python
def _find_nearest_players(
    frame: int,
    ball_x: float,
    ball_y: float,
    player_positions: list[PlayerPosition],
    search_frames: int = 15,
    max_candidates: int = 4,
    court_calibrator: CourtCalibrator | None = None,
) -> list[tuple[int, float, float]]:
    """Find nearest players to ball, ranked by perspective-corrected distance.

    Uses wrist keypoint distance when pose data is available, falling
    back to bbox upper-quarter distance. See _player_to_ball_dist().

    Ranks candidates by depth-scaled distance: wrist/bbox distance
    multiplied by a perspective correction factor derived from the court
    corners. Far-court distances are scaled up (they appear artificially
    small due to perspective compression).

    Returns:
        List of (track_id, distance, player_center_y), sorted by
        depth-corrected distance. Up to max_candidates entries.
    """
    # Best distances per track (a track may appear in multiple frames)
    # track_id -> (rank_dist, img_dist, center_y)
    best_per_track: dict[int, tuple[float, float, float]] = {}

    for p in player_positions:
        if abs(p.frame_number - frame) > search_frames:
            continue

        img_dist = _player_to_ball_dist(p, ball_x, ball_y)

        # Rank by depth-corrected distance: scale by perspective at player Y.
        # Use bbox upper-quarter Y for perspective scaling (stable regardless
        # of whether wrist or bbox was used for distance).
        scale_y = p.y - p.height * 0.25
        scale = _depth_scale_at_y(scale_y, court_calibrator)
        rank_dist = img_dist * scale

        if p.track_id not in best_per_track or rank_dist < best_per_track[p.track_id][0]:
            best_per_track[p.track_id] = (rank_dist, img_dist, p.y)

    ranked = sorted(best_per_track.items(), key=lambda x: x[1][0])
    return [
        (tid, img_dist, center_y)
        for tid, (_rank_dist, img_dist, center_y) in ranked[:max_candidates]
    ]
```

Note: perspective scaling still uses `p.y - p.height * 0.25` (bbox position) rather than wrist Y, because the depth scale is about the player's position in the scene, not the contact point.

- [ ] **Step 2: Run existing tests**

Run: `cd analysis && uv run pytest tests/unit/test_contact_detector.py -v`
Expected: All pass.

- [ ] **Step 3: Commit**

```bash
git add analysis/rallycut/tracking/contact_detector.py
git commit -m "_find_nearest_players: use wrist keypoint distance for candidate ranking"
```

---

### Task 3: Export `_player_to_ball_dist` and run production_eval

**Files:**
- Modify: `analysis/tests/unit/test_contact_detector.py` (import already added in Task 1)

- [ ] **Step 1: Run type checks**

Run: `cd analysis && uv run mypy rallycut/tracking/contact_detector.py`
Expected: No new errors.

- [ ] **Step 2: Run full test suite**

Run: `cd analysis && uv run pytest tests/unit/ -v`
Expected: All pass.

- [ ] **Step 3: Run production_eval to measure end-to-end delta**

Run: `cd analysis && uv run python scripts/production_eval.py`
Expected output: Full metric table. Compare `player_attribution_oracle` and `contact_f1` against baseline (`run_2026-04-09-181807.json` or latest canonical run).

Key metrics to watch:
- `player_attribution_oracle`: should improve (baseline 83.6%)
- `serve_attr_oracle`: should stay stable or improve (baseline 94.7%)
- `contact_f1`: must not regress (baseline 85.0%) — this is the GBM risk
- `action_accuracy`: should stay stable (baseline 92.4%)

- [ ] **Step 4: Evaluate results and decide**

If `player_attribution_oracle` improves and `contact_f1` doesn't regress: ship it.

If `contact_f1` regresses: the GBM contact classifier was trained on bbox distances. The distance feature distribution has shifted. Options:
1. Retrain the contact classifier GBM with the new distance feature
2. Keep wrist for ranking but bbox for the `Contact.player_distance` feature (fallback)

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "wrist keypoint attribution: production_eval results [TBD based on numbers]"
```
