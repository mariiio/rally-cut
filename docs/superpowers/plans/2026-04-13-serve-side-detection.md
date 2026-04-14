# Serve-Side Detection: Contact Classifier + Formation Fusion

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve serve-side detection from 86% to near-perfect by adding a serve contact classifier that uses ball/player attributes at the first detected contact, fused with the existing formation predictor.

**Architecture:** Three independent components: (1) fix auto_split windowing bug, (2) build a serve/receive contact classifier from existing contact attributes, (3) fuse formation + contact predictions with confidence-weighted voting. Each is testable independently.

**Tech Stack:** Python, existing `Contact` dataclass attributes (ballY, isAtNet, playerDistance, playerTrackId), existing player position data.

---

### Task 1: Fix auto_split windowing bug

**Files:**
- Modify: `analysis/rallycut/tracking/action_classifier.py:919-923`
- Test: `analysis/tests/unit/test_action_classifier.py`

The bug: `_compute_auto_split_y(player_positions)` on line 920 passes the full unwindowed position list. When adaptive window is active (e.g., frames 45-105), the auto_split computes medians from ALL frames, giving a different split than the windowed data expects.

- [ ] **Step 1: Write failing test**

Add to `TestFindServingSideByFormation` class in `analysis/tests/unit/test_action_classifier.py`:

```python
def test_auto_split_uses_windowed_positions(self) -> None:
    """Auto-split should use positions from the formation window, not full rally.

    Scenario: In the formation window (frames 0-60), 4 players are clearly
    2 near + 2 far. But in later frames (100+), players move and the
    full-rally auto_split gives a wrong split.
    """
    positions: list[PlayerPosition] = []
    # Formation window (frames 0-60): clear 2+2 split
    for f in range(60):
        positions.append(_player_pos(f, 1, y=0.70))  # near, foot=0.775
        positions.append(_player_pos(f, 2, y=0.60))  # near, foot=0.675
        positions.append(_player_pos(f, 3, y=0.30))  # far, foot=0.375
        positions.append(_player_pos(f, 4, y=0.20))  # far, foot=0.275
    # Later frames (100-200): all players move to near side
    for f in range(100, 200):
        positions.append(_player_pos(f, 1, y=0.80))
        positions.append(_player_pos(f, 2, y=0.70))
        positions.append(_player_pos(f, 3, y=0.65))
        positions.append(_player_pos(f, 4, y=0.60))

    # With a deliberately wrong net_y that puts all windowed players
    # on one side, auto_split should use windowed positions and find
    # the correct 2+2 split (not the full-rally split).
    side, conf = _find_serving_side_by_formation(
        positions, net_y=0.2, start_frame=0, window_frames=60,
    )
    # Should predict (near side has larger separation: 0.775-0.675=0.1
    # vs far side 0.375-0.275=0.1 — equal, but the model should still
    # produce a prediction rather than abstaining due to Nv0 failure)
    assert side is not None, "Should not abstain — 4 players clearly 2+2 in window"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd analysis && uv run pytest tests/unit/test_action_classifier.py::TestFindServingSideByFormation::test_auto_split_uses_windowed_positions -v`

Expected: FAIL — auto_split uses full rally positions, the later frames pull the split up, and the function may abstain or misclassify.

- [ ] **Step 3: Fix the auto_split call**

In `analysis/rallycut/tracking/action_classifier.py`, find line ~920 inside `_find_serving_side_by_formation`:

```python
    if near_count == 0 or far_count == 0:
        auto_split = _compute_auto_split_y(player_positions)
```

Replace with:

```python
    if near_count == 0 or far_count == 0:
        # Use only positions from the formation window (not full rally)
        # to match the tracks already computed in by_track.
        windowed_positions = [
            p for p in player_positions
            if p.track_id >= 0 and start_frame <= p.frame_number < end_frame
        ]
        auto_split = _compute_auto_split_y(windowed_positions)
```

Apply the same fix at line ~616 inside `_find_serving_team_by_formation` (the team-mapping auto_split call):

```python
    if near_count == 0 or near_count == len(track_medians):
        auto_split = _compute_auto_split_y(player_positions)
```

Replace with:

```python
    if near_count == 0 or near_count == len(track_medians):
        windowed_positions = [
            p for p in player_positions
            if p.track_id >= 0 and start_frame <= p.frame_number < end_frame
        ]
        auto_split = _compute_auto_split_y(windowed_positions)
```

- [ ] **Step 4: Run tests**

Run: `cd analysis && uv run pytest tests/unit/test_action_classifier.py -x -q`

Expected: 151 passed (150 existing + 1 new).

- [ ] **Step 5: Run ruff + mypy**

Run: `cd analysis && uv run ruff check rallycut/tracking/action_classifier.py && uv run mypy rallycut/tracking/action_classifier.py --no-error-summary`

Expected: clean.

- [ ] **Step 6: Commit**

```bash
cd /Users/mario/Personal/Projects/RallyCut
git add analysis/rallycut/tracking/action_classifier.py analysis/tests/unit/test_action_classifier.py
git commit -m "fix: auto_split uses windowed positions instead of full rally"
```

---

### Task 2: Add serve contact classifier

**Files:**
- Modify: `analysis/rallycut/tracking/action_classifier.py` (add new function after `_compute_adaptive_window`)
- Test: `analysis/tests/unit/test_action_classifier.py`

- [ ] **Step 1: Write failing tests**

Add a new test class in `analysis/tests/unit/test_action_classifier.py`:

```python
class TestClassifyServeContact:
    """Tests for _classify_serve_contact."""

    def test_serve_ball_high_not_at_net(self) -> None:
        """Ball above net (low Y from toss), not at net, player reaching → serve."""
        contact = Contact(
            frame=50, ball_x=0.5, ball_y=0.3,  # ball high (above net_y=0.5)
            velocity=0.01, direction_change_deg=150.0,
            player_track_id=1, player_distance=0.06,
            is_at_net=False, court_side="far",
        )
        result = _classify_serve_contact(contact, net_y=0.5)
        assert result is True

    def test_receive_ball_below_net(self) -> None:
        """Ball below net (high Y, near side) → receive."""
        contact = Contact(
            frame=80, ball_x=0.5, ball_y=0.7,  # ball below net_y=0.5
            velocity=0.02, direction_change_deg=130.0,
            player_track_id=2, player_distance=0.02,
            is_at_net=False, court_side="near",
        )
        result = _classify_serve_contact(contact, net_y=0.5)
        assert result is False

    def test_at_net_not_serve(self) -> None:
        """Contact at net with ball high → uncertain (not serve)."""
        contact = Contact(
            frame=60, ball_x=0.5, ball_y=0.4,
            velocity=0.015, direction_change_deg=140.0,
            player_track_id=3, player_distance=0.02,
            is_at_net=True, court_side="near",
        )
        result = _classify_serve_contact(contact, net_y=0.5)
        assert result is None  # at net → uncertain

    def test_player_too_close_uncertain(self) -> None:
        """Ball high but player very close to ball → uncertain."""
        contact = Contact(
            frame=50, ball_x=0.5, ball_y=0.3,
            velocity=0.01, direction_change_deg=150.0,
            player_track_id=1, player_distance=0.01,  # very close
            is_at_net=False, court_side="far",
        )
        result = _classify_serve_contact(contact, net_y=0.5)
        assert result is None  # player too close for serve
```

Add the import at the top of the test file alongside existing imports from `action_classifier`:

```python
from rallycut.tracking.action_classifier import _classify_serve_contact
```

Also add the `Contact` import:

```python
from rallycut.tracking.contact_detector import Contact
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd analysis && uv run pytest tests/unit/test_action_classifier.py::TestClassifyServeContact -v`

Expected: FAIL — `_classify_serve_contact` does not exist yet.

- [ ] **Step 3: Implement the classifier**

Add to `analysis/rallycut/tracking/action_classifier.py`, after the `_compute_adaptive_window` function (around line 840):

```python
def _classify_serve_contact(
    contact: Contact,
    net_y: float = 0.5,
    player_distance_min: float = 0.03,
) -> bool | None:
    """Classify whether a detected contact is a serve or receive.

    Uses ball position, net proximity, and player distance — attributes
    already computed by the contact detector. Validated on 322 GT-labeled
    contacts: ballY < net_y is 99% indicative of serve, ballY >= net_y
    is 95% indicative of receive.

    Args:
        contact: The detected contact with ball/player attributes.
        net_y: Court split Y (net position in image space).
        player_distance_min: Minimum player-to-ball distance to qualify
            as a serve (server reaches for toss, so distance is larger
            than a close-range receive). Default 0.03 from training data
            (serve median=0.059, receive median=0.023).

    Returns:
        True if contact is a serve, False if receive, None if uncertain.
    """
    # Ball below net_y (near side, high Y) → receive (ball has crossed net)
    if contact.ball_y >= net_y:
        return False

    # Ball above net_y (far side, low Y from toss) + not at net + player
    # reaching (distance > threshold) → serve
    if (
        not contact.is_at_net
        and contact.player_distance >= player_distance_min
    ):
        return True

    # Uncertain: ball high but at net or player too close
    return None
```

- [ ] **Step 4: Run tests**

Run: `cd analysis && uv run pytest tests/unit/test_action_classifier.py -x -q`

Expected: 155 passed (151 + 4 new).

- [ ] **Step 5: Run ruff + mypy**

Run: `cd analysis && uv run ruff check rallycut/tracking/action_classifier.py && uv run mypy rallycut/tracking/action_classifier.py --no-error-summary`

Expected: clean.

- [ ] **Step 6: Commit**

```bash
cd /Users/mario/Personal/Projects/RallyCut
git add analysis/rallycut/tracking/action_classifier.py analysis/tests/unit/test_action_classifier.py
git commit -m "feat: add serve contact classifier (serve vs receive from ball/player attributes)"
```

---

### Task 3: Add serving side from contact

**Files:**
- Modify: `analysis/rallycut/tracking/action_classifier.py` (add function after `_classify_serve_contact`)
- Test: `analysis/tests/unit/test_action_classifier.py`

This function takes the contact classifier output and the contact player's position to determine the serving SIDE ("near"/"far").

- [ ] **Step 1: Write failing tests**

Add to `TestClassifyServeContact` class:

```python
def test_serving_side_from_serve_contact(self) -> None:
    """Serve contact → player's side is serving side."""
    contact = Contact(
        frame=50, ball_x=0.5, ball_y=0.3,
        velocity=0.01, direction_change_deg=150.0,
        player_track_id=1, player_distance=0.06,
        is_at_net=False, court_side="far",
    )
    # Player 1 is on near side (foot Y > net_y)
    positions = [_player_pos(f, 1, y=0.70) for f in range(60)]
    positions += [_player_pos(f, 2, y=0.60) for f in range(60)]
    positions += [_player_pos(f, 3, y=0.30) for f in range(60)]

    side, conf = _serving_side_from_contact(contact, positions, net_y=0.5)
    assert side == "near"  # player 1 is on near side → near serves
    assert conf > 0

def test_serving_side_from_receive_contact(self) -> None:
    """Receive contact → opposite of player's side."""
    contact = Contact(
        frame=80, ball_x=0.5, ball_y=0.7,
        velocity=0.02, direction_change_deg=130.0,
        player_track_id=3, player_distance=0.02,
        is_at_net=False, court_side="near",
    )
    # Player 3 is on far side (foot Y < net_y)
    positions = [_player_pos(f, 1, y=0.70) for f in range(90)]
    positions += [_player_pos(f, 3, y=0.30) for f in range(90)]

    side, conf = _serving_side_from_contact(contact, positions, net_y=0.5)
    assert side == "near"  # player 3 on far received → near served
    assert conf > 0

def test_serving_side_no_player_positions(self) -> None:
    """No positions for the contact player → None."""
    contact = Contact(
        frame=50, ball_x=0.5, ball_y=0.3,
        velocity=0.01, direction_change_deg=150.0,
        player_track_id=99, player_distance=0.06,
        is_at_net=False, court_side="far",
    )
    side, conf = _serving_side_from_contact(contact, [], net_y=0.5)
    assert side is None
```

Add the import:

```python
from rallycut.tracking.action_classifier import _serving_side_from_contact
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd analysis && uv run pytest tests/unit/test_action_classifier.py::TestClassifyServeContact::test_serving_side_from_serve_contact -v`

Expected: FAIL — `_serving_side_from_contact` not defined.

- [ ] **Step 3: Implement**

Add to `analysis/rallycut/tracking/action_classifier.py`, after `_classify_serve_contact`:

```python
def _serving_side_from_contact(
    contact: Contact,
    player_positions: list[PlayerPosition],
    net_y: float = 0.5,
) -> tuple[str | None, float]:
    """Determine serving side from a contact's classification and player position.

    Classifies the contact as serve/receive, then uses the contact player's
    court position to determine which side served:
    - Serve contact → player's side is serving side
    - Receive contact → opposite side is serving side

    Args:
        contact: First detected contact in the rally.
        player_positions: Player positions for locating the contact player.
        net_y: Court split Y.

    Returns:
        (side, confidence) where side is "near", "far", or None.
    """
    is_serve = _classify_serve_contact(contact, net_y)
    if is_serve is None:
        return None, 0.0

    # Find contact player's court side from their foot position near contact frame
    tid = contact.player_track_id
    if tid < 0:
        return None, 0.0

    player_ys = [
        p.y + p.height / 2.0
        for p in player_positions
        if p.track_id == tid and abs(p.frame_number - contact.frame) < 30
    ]
    if not player_ys:
        return None, 0.0

    player_y = sum(player_ys) / len(player_ys)
    player_side = "near" if player_y > net_y else "far"

    # Confidence from how clearly the contact matches serve/receive
    ball_dist_from_net = abs(contact.ball_y - net_y)
    confidence = min(1.0, ball_dist_from_net / 0.15)

    if is_serve:
        return player_side, confidence
    # Receive → opposite side served
    return ("far" if player_side == "near" else "near"), confidence
```

- [ ] **Step 4: Run tests**

Run: `cd analysis && uv run pytest tests/unit/test_action_classifier.py -x -q`

Expected: 158 passed.

- [ ] **Step 5: Commit**

```bash
cd /Users/mario/Personal/Projects/RallyCut
git add analysis/rallycut/tracking/action_classifier.py analysis/tests/unit/test_action_classifier.py
git commit -m "feat: serving side derivation from contact classification"
```

---

### Task 4: Wire fusion into formation predictor

**Files:**
- Modify: `analysis/rallycut/tracking/action_classifier.py` — update `_find_serving_side_by_formation` to accept contacts and fuse
- Modify: `analysis/rallycut/tracking/action_classifier.py` — update `classify_rally_actions` to pass contacts
- Modify: `analysis/scripts/production_eval.py` — pass contacts to formation predictor in `_apply_viterbi_scoring`

- [ ] **Step 1: Add `first_contact` parameter to `_find_serving_side_by_formation`**

Add a new optional parameter to `_find_serving_side_by_formation`:

```python
def _find_serving_side_by_formation(
    player_positions: list[PlayerPosition],
    net_y: float,
    start_frame: int = 0,
    window_frames: int = 120,
    ball_positions: list[BallPosition] | None = None,
    calibrator: CourtCalibrator | None = None,
    first_contact_frame: int | None = None,
    adaptive_window: bool = False,
    first_contact: Contact | None = None,  # NEW
) -> tuple[str | None, float]:
```

At the end of the function, before the final return, add the fusion logic:

```python
    # Fuse with contact classifier when available
    if first_contact is not None:
        side_contact, conf_contact = _serving_side_from_contact(
            first_contact, player_positions, net_y,
        )
        if side_contact is not None:
            if score_side is None:
                # Formation abstained — use contact
                return side_contact, conf_contact
            if side_contact == score_side:
                # Agreement — boost confidence
                return score_side, max(confidence, conf_contact)
            # Disagreement — pick higher confidence
            if conf_contact > confidence:
                return side_contact, conf_contact

    # ... existing return logic
```

Note: `score_side` refers to the formation prediction computed earlier in the function. The variable holding this is the result computed from the logistic model. You will need to capture it before the final return. Rename the existing final prediction to `formation_side` and `formation_conf` and use those in the fusion logic.

- [ ] **Step 2: Thread `first_contact` through `_find_serving_team_by_formation`**

Add `first_contact: Contact | None = None` parameter to `_find_serving_team_by_formation` and pass it through to `_find_serving_side_by_formation`.

- [ ] **Step 3: Thread `first_contact` through `classify_rally_actions`**

In `classify_rally_actions`, the formation call already has `first_contact_frame_val`. Add the actual `Contact` object:

```python
        first_contact_obj: Contact | None = None
        if contact_sequence.contacts:
            first_contact_obj = contact_sequence.contacts[0]

        formation_team, _ = _find_serving_team_by_formation(
            ...,
            first_contact=first_contact_obj,
        )
```

- [ ] **Step 4: Thread contacts in `production_eval.py` `_apply_viterbi_scoring`**

In `_apply_viterbi_scoring`, the formation call needs the first contact. The contact data is available from `rd.contacts_json`:

```python
                # Build first Contact for fusion
                first_contact_obj = None
                if rd and rd.contacts_json and isinstance(rd.contacts_json, dict):
                    contacts_list = rd.contacts_json.get("contacts", [])
                    if contacts_list:
                        c = contacts_list[0]
                        first_contact_obj = Contact(
                            frame=c.get("frame", 0),
                            ball_x=c.get("ballX", 0.5),
                            ball_y=c.get("ballY", 0.5),
                            velocity=c.get("velocity", 0.0),
                            direction_change_deg=c.get("directionChangeDeg", 0.0),
                            player_track_id=c.get("playerTrackId", -1),
                            player_distance=c.get("playerDistance", float("inf")),
                            is_at_net=c.get("isAtNet", False),
                            court_side=c.get("courtSide", "unknown"),
                        )
                formation_side, formation_conf = _find_serving_side_by_formation(
                    ...,
                    first_contact=first_contact_obj,
                )
```

Add `Contact` to the imports in `production_eval.py`:

```python
from rallycut.tracking.contact_detector import Contact
```

- [ ] **Step 5: Run tests + lint**

Run: `cd analysis && uv run pytest tests/unit/test_action_classifier.py -x -q && uv run ruff check rallycut/tracking/action_classifier.py scripts/production_eval.py && uv run mypy rallycut/tracking/action_classifier.py --no-error-summary`

Expected: all pass, all clean.

- [ ] **Step 6: Commit**

```bash
cd /Users/mario/Personal/Projects/RallyCut
git add analysis/rallycut/tracking/action_classifier.py analysis/scripts/production_eval.py
git commit -m "feat: fuse serve contact classifier with formation predictor"
```

---

### Task 5: Measure and verify

**Files:**
- Run: `analysis/scripts/production_eval.py`
- Run: diagnostic measurement script (inline)

- [ ] **Step 1: Run production_eval**

Run: `cd analysis && uv run python scripts/production_eval.py --reruns 1`

Check output for `score_accuracy`. Expected: improvement over 74.6% baseline.

- [ ] **Step 2: Measure formation side accuracy on target footage**

Run inline measurement comparing formation-only vs fusion on the 448-rally dataset to understand per-category impact (Nv0 fix, contact classifier, fusion).

- [ ] **Step 3: Generate debug clips for any remaining errors**

If errors remain, generate clips using `scripts/extract_serve_debug_clips.py` to visually confirm they're genuinely ambiguous.

- [ ] **Step 4: Final verification**

Run: `cd analysis && uv run pytest tests/unit/test_action_classifier.py -x -q && uv run ruff check rallycut/tracking/action_classifier.py && uv run mypy rallycut/tracking/action_classifier.py --no-error-summary`

Expected: all tests pass, lint clean, types clean.
