# Team Identity System — Design Spec

## Context

Score tracking accuracy was 68.5% (production_eval). The pipeline correctly detects
which **side** serves (formation: 97% coverage) and tracks **individuals** across
rallies (DINOv2: 94.9%). The entire gap was a **label alignment problem**: the system
defaults "near = team A" but GT may define team A as the far-side team. This inverts
all labels for affected videos.

## Key Finding

**The convention problem is a presentation concern, not a detection problem.** The
system correctly identifies which *players* serve in each rally. The A/B label is
arbitrary — in production, the user sees player thumbnails and knows who's who.

For eval, standard label alignment (1 GT label per video aligns the convention) brings
accuracy to **76.1%** — confirming the pipeline works. Remaining errors are from
formation abstentions and switch boundary noise.

## Goal

Make team identity a first-class concept stored in the DB:
1. Define a team as a pair of individually-identified players with stored appearance signatures
2. Store team templates in match_analysis_json for downstream features (team-specific stats, heatmaps)
3. For eval: label alignment via `calibrate_initial_side()` (standard eval methodology)

## Architecture

### Data Model

```python
@dataclass
class TeamTemplate:
    team_label: str                    # "A" or "B"
    player_ids: list[int]              # [1, 2] or [3, 4]
    reid_embeddings: list[np.ndarray]  # Individual DINOv2/OSNet embeddings (per player)
    lower_body_hists: list[np.ndarray] # Individual shorts HSV histograms
    upper_body_hists: list[np.ndarray] # Individual jersey HSV histograms
    confidence: float                  # Inter-team discriminability score
```

Two templates per video, built from match_tracker's `PlayerAppearanceProfile` after Pass 2.

### Components

#### 1. TeamTemplateBuilder (`tracking/team_identity.py`)

**Input**: match_tracker's 4 `PlayerAppearanceProfile` objects after Pass 2.

**Output**: 2 `TeamTemplate` objects.

**Logic**:
- Group profiles by match_tracker's team assignment: team 0 = {profile[1], profile[2]}, team 1 = {profile[3], profile[4]}
- Extract per-player ReID embeddings and HSV histograms from profiles
- Compute inter-team discriminability: mean pairwise Bhattacharyya distance between teams' lower-body histograms
- Set `confidence` based on discriminability (high when teams look different, low when similar)

#### 2. TeamLocalizer (`tracking/team_identity.py`)

**Input**: Per-rally `TrackAppearanceStats` for 4 tracks + 2 `TeamTemplate` objects.

**Output**: `dict[str, str]` mapping team label to court side ("near"/"far").

**Logic**:
- For each track (typically 2-4 per rally), compute similarity to each of the 4 template players using existing `compute_track_similarity()` (Bhattacharyya + ReID cosine blend)
- Build an NxM cost matrix (N tracks x 4 template players) and solve with Hungarian matching (or greedy assignment when N < 4)
- Group matched tracks by team template
- Determine which team's tracks have higher Y (near) vs lower Y (far)
- Return team-to-side mapping
- Require at least 1 track matched per team to produce a result; otherwise return None

**Key**: This is individual identity matching (94.9% accuracy), not team-color clustering. Robust even when both teams wear similar colors.

**Fallback**: If matching confidence is below threshold (e.g., best-vs-second margin < 0.05), return None (caller uses position-based fallback).

#### 3. ConventionResolver (`scoring/convention.py`)

**Input**: Team templates + first-rally formation data.

**Output**: A/B label assignment (which template = team A).

**Automatic mode** (production):
- Rally 1: formation says "near serves"
- TeamLocalizer says "template X is near"
- Convention: template X = team A (the first-serving team)
- Deterministic: anchored to rally 1's formation + appearance

**GT calibration mode** (eval):
- From one GT-labeled rally where `gt_serving_team` is known:
  - Formation: "near serves"
  - TeamLocalizer: "template X is near"
  - GT: "team A serves"
  - → Template X = team A
- Majority-vote across multiple GT rallies for robustness
- Overrides automatic mode when GT is available

#### 4. Viterbi Integration (`scoring/cross_rally_viterbi.py`)

**Replace** `_score_plausibility()` with `_score_team_consistency()`:

- Input: per-rally team localization results from TeamLocalizer
- For each hypothesis (near=A vs near=B):
  - Check how many rallies have consistent team assignment (TeamLocalizer's team-to-side matches the hypothesis's expectation)
  - Sum consistency scores weighted by per-rally confidence
- Return hypothesis with higher consistency

**Extend** `RallyObservation` with optional team localization data:
```python
@dataclass
class RallyObservation:
    formation_side: str | None        # "near" / "far" (existing)
    formation_confidence: float       # (existing)
    team_near: str | None = None      # "A" or "B" from TeamLocalizer
    team_localization_conf: float = 0  # TeamLocalizer confidence
```

### Data Flow

```
match_tracker.process_all_rallies()
    → 4 PlayerAppearanceProfiles (post Pass 2)
    → TeamTemplateBuilder.build(profiles)
        → TeamTemplate A (player_ids=[1,2], embeddings, hists)
        → TeamTemplate B (player_ids=[3,4], embeddings, hists)
    → stored in match_analysis_json["teamTemplates"]

reattribute_actions:
    load TeamTemplates from match_analysis_json
    ConventionResolver.resolve(templates, rally_1_formation)
        → template_A, template_B with A/B labels

    per rally:
        TeamLocalizer.localize(track_stats, [template_A, template_B])
            → {"A": "near", "B": "far"} (or None if low confidence)
        formation → "near serves"
        → RallyObservation(formation_side="near", team_near="A")

    Viterbi decoder:
        _score_team_consistency(observations)
            → hypothesis with higher appearance consistency wins
        decode → score sequence with team-labeled serving
```

### DB Storage

Team templates stored in `match_analysis_json`:

```json
{
  "teamTemplates": {
    "A": {
      "playerIds": [1, 2],
      "confidence": 0.82
    },
    "B": {
      "playerIds": [3, 4],
      "confidence": 0.82
    }
  }
}
```

Note: Full appearance data (embeddings, histograms) lives in the player profiles already stored in match_analysis_json. Team templates reference players by ID — no need to duplicate the feature data.

## Edge Cases

| Case | Behavior |
|------|----------|
| Similar team colors | Individual ReID matching still works (body shape, hair, skin). HSV is secondary signal. |
| Rally-1 misassignment | Templates built from post-Pass-2 profiles (EMA across all rallies). Single-rally noise absorbed. |
| Side switch mid-video | Not a problem. TeamLocalizer determines team-to-side per rally from appearance, independent of position history. |
| TeamLocalizer low confidence | Fall back to position-based convention (near=A). No worse than today. |
| < 2 rallies | match_tracker requires ≥2. Fall back to position convention. |
| No match_analysis (lili/lulu) | Cannot build templates. Fall back to position convention. Same as today. |
| Re-processing same video | Deterministic given same model weights. Convention anchored to rally 1 formation + appearance. |

## Files to Modify

| File | Change |
|------|--------|
| **New: `analysis/rallycut/tracking/team_identity.py`** | `TeamTemplate`, `TeamTemplateBuilder`, `TeamLocalizer` |
| **New: `analysis/rallycut/scoring/convention.py`** | `ConventionResolver` (automatic + GT calibration) |
| `analysis/rallycut/tracking/match_tracker.py` | After Pass 2, call `TeamTemplateBuilder`. Export templates in match_analysis output. |
| `analysis/rallycut/scoring/cross_rally_viterbi.py` | Replace `_score_plausibility()` with `_score_team_consistency()`. Extend `RallyObservation` with team localization. |
| `analysis/rallycut/cli/commands/reattribute_actions.py` | Load team templates. Run `TeamLocalizer` per rally. Feed team-labeled observations to Viterbi. |
| `analysis/scripts/production_eval.py` | Add GT calibration path via `ConventionResolver`. |
| `api/src/services/matchAnalysisService.ts` | TypeScript types for team templates in match_analysis_json. |

### Existing Code to Reuse

- `compute_track_similarity()` (`player_features.py`) — pairwise cost for TeamLocalizer
- `PlayerAppearanceProfile` — source data for TeamTemplate
- `extract_appearance_features()` — already extracted per rally in match_players
- `decode_video_dual_hypothesis()` — extend with team consistency scorer
- `calibrate_from_noisy_predictions()` (`cross_rally_viterbi.py`) — pattern for GT calibration
- `linear_sum_assignment` (scipy) — Hungarian matching in TeamLocalizer

## Verification

1. **Unit tests**:
   - `TeamTemplateBuilder` produces correct templates from synthetic profiles
   - `TeamLocalizer` correctly assigns teams when swapping near/far
   - `ConventionResolver` GT calibration flips convention when GT disagrees with automatic
   - `_score_team_consistency()` breaks ties that `_score_plausibility()` cannot

2. **Integration test**:
   - Run `production_eval.py` on full 11-video eval set
   - Expect ≥ 75% score_accuracy (no regression)
   - Any video where team A starts far AND has match_analysis should now be correct
   - Run ≥ 2 full eval passes to confirm determinism (std ≈ 0.0045)

3. **Manual spot-check**:
   - Pick a video with known side switches
   - Verify TeamLocalizer correctly tracks team-to-side across switches
   - Verify convention matches GT when GT calibration is applied
