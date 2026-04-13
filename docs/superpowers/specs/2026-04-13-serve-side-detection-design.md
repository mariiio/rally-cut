# Serve-Side Detection: Contact Classifier + Formation Fusion

## Context

Score tracking requires knowing which side served each rally. The current formation predictor (multi-feature logistic on player positions + adaptive window) achieves 86.2% formation side accuracy on target footage (74.6% end-to-end score_accuracy).

An extensive diagnostic session (2026-04-13) identified the remaining 51 errors in 3 categories:
- **Nv0 split failure (11, 22%)**: auto_split bug uses wrong frame range
- **Server off-frame (12, 23%)**: near-side server starts off-screen
- **2v2 ambiguous (28, 55%)**: both sides look identical from positions alone

Visual debugging of 10 error clips revealed: in 7/10, the first contact correctly identifies the server (closest wrist to ball). In 3/10, ball origin from off-screen indicates the serving side. The contact detector already captures `playerTrackId`, `ballY`, `isAtNet`, `playerDistance`, and `directionChangeDeg` — sufficient to classify serve vs receive.

Analysis of 322 labeled contacts confirmed serve contacts are clearly distinguishable: ball is on far side (high from toss) in 99% of serves vs 64% of receives, and the contact player is farther from the ball (0.059 vs 0.023) due to the toss/reach motion.

## Design

### Component 1: Auto-split windowing fix

**Problem**: `_compute_auto_split_y` uses full-rally positions instead of the formation window, causing all 4 players to be classified on one side when court_split_y is wrong.

**Fix**: Pass only the windowed positions (same frames as formation analysis) to auto_split. No new logic — just correct the frame range passed to the existing function.

**Expected impact**: Fixes up to 11 Nv0 errors.

### Component 2: Serve contact classifier

A rule-based binary classifier on the first detected contact that determines: is this a serve or a receive?

**Features** (from existing contact data, no new detection needed):
- `ballY` relative to `net_y`: ball high in air (low Y, from toss) = serve (99% of serves)
- `isAtNet`: serves are never at the net (4% vs 32%)
- `playerDistance`: server is farther from ball at contact (toss/reach)

**Classification rules**:
```
is_serve_contact(contact, net_y):
    if contact.ballY < net_y AND NOT contact.isAtNet AND contact.playerDistance > 0.03:
        return True   # serve: ball high + not at net + player reaching
    if contact.ballY >= net_y:
        return False  # receive: ball on near/low side
    return None       # uncertain
```

**Serving side derivation**:
- `is_serve = True`: `playerTrackId`'s court side = serving side
- `is_serve = False`: opposite of `playerTrackId`'s court side = serving side
- `is_serve = None`: no prediction (fall back to formation)

The player's court side is determined by their foot Y position relative to net_y at the contact frame, using the same near/far classification as the formation predictor.

**Expected impact**: Addresses the 28 core 2v2 ambiguous errors where formation can't distinguish sides but contact attribution can.

### Component 3: Formation + contact fusion

Two independent signals combined:
1. **Formation predictor** (existing): static spatial signal from player positions at rally start. Strong when formation is clear (server at baseline, receivers mid-court).
2. **Contact classifier** (new): dynamic event signal from who hit the ball and how. Strong when formation is ambiguous but the serve contact is detected.

**Fusion logic**:
```
predict_serving_side(rally):
    side_f, conf_f = formation_predictor(...)
    side_c, conf_c = contact_classifier(first_contact, net_y, positions)
    
    if side_f and side_c:
        if side_f == side_c:
            return side_f, max(conf_f, conf_c)   # agreement = high confidence
        # disagreement: pick higher confidence
        return (side_f, conf_f) if conf_f > conf_c else (side_c, conf_c)
    
    # only one has prediction
    return (side_f, conf_f) if side_f else (side_c, conf_c)
```

The confidence for the contact classifier is based on how clearly the contact matches serve characteristics (e.g., how far ballY is below net_y, how large playerDistance is).

## Files to modify

- `analysis/rallycut/tracking/action_classifier.py`:
  - Fix `_compute_auto_split_y` call to use windowed positions
  - Add `_classify_serve_contact()` function
  - Add `_predict_serving_side_fused()` or integrate into existing `_find_serving_side_by_formation`
- `analysis/scripts/production_eval.py`: thread contact data to fusion predictor
- `analysis/tests/unit/test_action_classifier.py`: add tests for serve contact classifier

## Verification

1. **Measure serve contact classifier accuracy** on 322 labeled contacts (LOO-video CV)
2. **Measure fusion accuracy** on 448-rally dataset (LOO-video CV) — compare formation-only vs contact-only vs fusion
3. **Run production_eval** to measure end-to-end score_accuracy delta
4. **Generate debug clips** for any remaining errors to confirm they're genuinely ambiguous
5. **All existing tests pass** (150 unit tests + mypy + ruff)
