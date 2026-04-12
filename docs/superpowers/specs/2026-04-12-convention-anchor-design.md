# Convention Anchor: Closing the 10pp Score Tracking Gap

## Context

The dual-hypothesis Viterbi decoder (commit `b31db4c`) achieves 67-68.5% score_accuracy
without user input, vs 76-78% with oracle calibration — a ~10pp gap. The physical serving
side detection (formation) is near-perfect (97% coverage, 0% true errors). The bottleneck
is the A/B team convention determination: mapping physical near/far to semantic team A/B.

**Two distinct failure modes identified:**

| Category | Videos | Gap | Root cause |
|----------|--------|-----|------------|
| Convention INVERTED | lulu (4f2bd66a), lili (b026dc6c) | Full inversion | Narrow angle, 22-29/29 formation abstentions, plausibility coin flip |
| Convention PARTIAL | 84e66e74, 635dcba2, ff175026 | 10-14pp each | Side-switch boundaries wrong, initial convention correct but drifts |

**Goal:** Close ~8-10pp of the gap without user input.

## Phase 1: Position-based side-switch detection

### Problem

Current side-switch detection comes from `match_analysis.sideSwitchDetected`, which is
derived from appearance-based cross-rally matching in `match_tracker.py`. For 3
normal-formation videos (84e66e74, 635dcba2, ff175026), incorrect switch boundaries cause
the correct initial convention to drift mid-video, losing 10-14pp each.

### Signal

In beach volleyball, side switches are physical — all 4 players swap court sides
simultaneously between rallies. This is directly observable from player positions.

### Algorithm

```
detect_side_switches_from_positions(
    observations: list[RallyObservation],
    player_positions: list[list[PlayerPosition]],  # per-rally
    track_to_player: list[dict[int, int]],          # per-rally
    net_y_values: list[float],                      # per-rally
) -> set[int]:
    
    For each rally i:
      1. Classify each track as near/far:
         - Primary: use net_y (court_split_y) to split tracks by foot Y
         - Fallback: if net_y unreliable (broken court_split_y), cluster
           tracks by bottom-bbox Y using k=2 clustering
      2. Map track_ids -> player_ids via track_to_player
      3. Compute near_team_signal:
         - Count player_ids in {1,2} vs {3,4} on the near side
         - near_team = A if majority {1,2}, B if majority {3,4}, None if tied/empty
    
    Detect transitions with hysteresis:
      1. Walk through near_team sequence: [A, A, A, B, B, B, ...]
      2. Candidate switch at rally i where near_team[i] != near_team[i-1]
      3. Confirm only if new team persists for >= 2 consecutive rallies
      4. Reject isolated 1-rally flips as tracking noise
    
    Return set of confirmed switch rally indices.
```

### Score milestone soft prior (production enhancement)

Side switches correlate with cumulative score milestones (every 7 points in sets 1-2,
every 5 in set 3). When multiple candidate switch points exist within a window, prefer
the one closest to a milestone boundary. NOT a hard constraint because:
- Players sometimes forget to switch or points get replayed
- GT videos are mid-match subsets (not from rally 1)
- Too strict breaks eval on partial-match recordings

### Integration

New function in `cross_rally_viterbi.py`. Replaces the `switch_indices` computed from
`prod_semantic_flip` differences in `eval_score_viterbi.py:_eval_dual_hypothesis()` and
`production_eval.py:_apply_viterbi_scoring()`.

### Files to modify

- `analysis/rallycut/scoring/cross_rally_viterbi.py` — add `detect_side_switches_from_positions()`
- `analysis/scripts/eval_score_viterbi.py` — add eval config using position-based switches
- `analysis/scripts/production_eval.py` — wire position-based switches into `_apply_viterbi_scoring()`

## Phase 2: Team-identity convention anchor

### Problem

The plausibility heuristic (`_score_plausibility()`) picks the A/B convention using only
sequence statistics (mean run length ~2.06, score balance, long-run penalty). This is a
weak signal — for 2 narrow-angle videos (lulu, lili) it picks the wrong convention
entirely, and it provides no confidence margin for borderline cases.

### Signal

For each rally where we know the physical serving side, we can identify players on that
side and look up their team via `track_to_player`. The serving-side partner at the net is
almost always visible (even when the server is off-screen near or occluded far). Their
player_id directly indicates which team is on that side.

### Algorithm

```
calibrate_from_player_identity(
    observations: list[RallyObservation],
    player_positions: list[list[PlayerPosition]],
    track_to_player: list[dict[int, int]],
    net_y_values: list[float],
    side_switch_rallies: set[int],  # from Phase 1
) -> tuple[bool, float]:  # (initial_near_is_a, confidence)

    votes_near_is_a = 0.0
    votes_near_is_b = 0.0
    cumulative_switches = 0
    
    For each rally i:
      if i in side_switch_rallies:
        cumulative_switches += 1
      flipped = (cumulative_switches % 2 == 1)
      
      # Get tracks on the serving side (if formation signal available)
      if observations[i].formation_side is not None:
        serving_side = observations[i].formation_side  # "near" or "far"
        tracks_on_serving_side = [t for t in rally_tracks if on_side(t, serving_side, net_y)]
      else:
        # No formation: just look at which team is near (regardless of serving)
        tracks_on_serving_side = None  # use all near-side tracks for team ID
        
      # Map to player_ids and determine team
      for track in relevant_tracks:
        pid = track_to_player[i].get(track.track_id)
        if pid is None: continue
        team = "A" if pid <= 2 else "B"
        
        # Account for side switches
        if flipped: team = "B" if team == "A" else "A"
        
        # Vote
        weight = observations[i].formation_confidence or 0.5
        if team == "A": votes_near_is_a += weight
        else: votes_near_is_b += weight
    
    total = votes_near_is_a + votes_near_is_b
    if total == 0:
      return True, 0.0  # no signal, default
    
    near_is_a = votes_near_is_a >= votes_near_is_b
    confidence = abs(votes_near_is_a - votes_near_is_b) / total
    return near_is_a, confidence
```

### For narrow-angle videos

When formation abstains on most rallies (lulu: 22/23, lili: 29/29), the algorithm still
works because it falls back to identifying which team's players are on the near side
regardless of formation. Even without knowing which side is serving, knowing which team
is near → convention.

### Fallback

If `calibrate_from_player_identity()` returns confidence < 0.1 (too few votes or near
tie), fall back to the existing `decode_video_dual_hypothesis()` with plausibility scoring.

### Integration

New function in `cross_rally_viterbi.py`. Called before Viterbi decoding. If confident,
feeds `initial_near_is_a` to a single `decode_video()` call. If not confident, falls back
to dual-hypothesis.

### Files to modify

- `analysis/rallycut/scoring/cross_rally_viterbi.py` — add `calibrate_from_player_identity()`
- `analysis/scripts/eval_score_viterbi.py` — add eval configs for identity calibration
- `analysis/scripts/production_eval.py` — wire into `_apply_viterbi_scoring()`

## Combined flow

```
1. Load per-rally: observations, player_positions, track_to_player, net_y
2. Phase 1: detect_side_switches_from_positions() -> switch_indices
3. Phase 2: calibrate_from_player_identity(switch_indices) -> initial_near_is_a, confidence
4. If confidence >= 0.1:
     decode_video(initial_near_is_a, switch_indices)
   Else:
     decode_video_dual_hypothesis(switch_indices)  # plausibility fallback
```

## Verification

### Step 1: Diagnostic script

New `analysis/scripts/diagnose_convention.py`:
- For each of the 5 problem videos: show per-rally formation_side, player_ids per side,
  current vs position-based switch boundaries, identity votes
- Compare convention picked vs GT convention
- Identify exactly where and why each video fails

### Step 2: Eval harness integration

Add configs to `eval_score_viterbi.py`:

| Config | Description |
|--------|-------------|
| `position_switches` | Viterbi + position-based switch detection (Phase 1 only) |
| `identity_cal` | Viterbi + identity convention anchor (Phase 2 only, original switches) |
| `combined` | Both Phase 1 + Phase 2 |

### Step 3: Gate criteria

- **Aggregate:** Combined accuracy >= 76% on 304 rallies (+8pp over dual-hypothesis baseline)
- **Per-video:** Each of the 5 problem videos improves >= 5pp
- **No regressions:** Videos that dual-hypothesis gets right should not regress > 1pp
- **Run >= 2 times** to confirm stability (score_accuracy std ~0.0045)

### Step 4: Production integration

Wire into `_apply_viterbi_scoring()` in `production_eval.py`:
- Replace `prod_semantic_flip`-based switch detection with position-based
- Add identity calibration as convention picker
- Verify with full production_eval run

## Existing code to reuse

| Function | File | Purpose |
|----------|------|---------|
| `_find_serving_side_by_formation()` | `action_classifier.py:734` | Physical side detection (already used by Viterbi) |
| `_compute_auto_split_y()` | `action_classifier.py` | Fallback net_y when court_split_y broken |
| `decode_video()` | `cross_rally_viterbi.py:55` | Single-hypothesis Viterbi decoder |
| `decode_video_dual_hypothesis()` | `cross_rally_viterbi.py:344` | Dual-hypothesis fallback |
| `calibrate_from_noisy_predictions()` | `cross_rally_viterbi.py:326` | Existing calibration (to be replaced/augmented) |
| `verify_team_assignments()` | `match_tracker.py:88` | Validates physical near/far alignment |
| `_load_eval_data()` | `eval_score_viterbi.py:78` | Loads rally data + production signals |

## Risk assessment

- **Phase 1 risk: Low.** Position-based switch detection uses a strong binary signal
  (players physically swap sides). Main risk: tracking gaps at switch transitions.
- **Phase 2 risk: Medium.** Depends on track_to_player quality. Per-rally signal ~70%
  correct, but majority vote across 20+ rallies should be >95% for the convention decision.
  Fallback to plausibility if identity signal is weak.
- **Combined risk: Low.** Each phase independently improves different failure modes.
  Regressions unlikely because plausibility fallback preserved.
