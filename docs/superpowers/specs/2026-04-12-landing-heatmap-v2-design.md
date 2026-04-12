# Landing Heatmap v2 — Per-Team, All Attacks, Per-Rally Debug

**Date:** 2026-04-12
**Status:** Design approved, pending implementation

## Context

Landing heatmap v1 shipped with serve targets (receiver's feet) and terminal attack landings (ball-stopped). Three gaps identified:

1. **No per-team grouping** — all landings shown on one court with no team separation
2. **Mid-rally attacks ignored** — only terminal attacks captured (~0.5/match), missing the 3-5x more attacks where a defender touches the ball
3. **No per-rally debug view** — no way to visually verify individual rally detections

## Design Decisions

- **Per-team by default**, architecture ready for per-player drill-down later (blocked on user-selected player crops / higher identity confidence)
- **Normalize to half-court** — all points rotated to team's canonical perspective (own half bottom, opponent half top) regardless of side switches
- **Normalize at aggregation time** (Approach B) — detection stores raw court coords + court_side; aggregation flips when building per-team grids
- **All attacks included** — mid-rally attacks use next-contact defender's feet (Z=0, exact homography); terminal attacks use ball-stopped or trajectory fallback
- **Per-rally debug** via toggle in PlayerTrackingToolbar, mini court overlay on video player

## 1. Landing Detection Changes

### File: `analysis/rallycut/statistics/landing_detector.py`

**Extended `LandingPoint` dataclass:**
```python
@dataclass
class LandingPoint:
    # ... existing fields ...
    court_side: str  # "near" or "far" — acting team's court side this rally
```

**Extended `detect_rally_landings()` — mid-rally attack detection:**

For every ATTACK action in the rally (not just terminal):
- If a next contact exists (dig, block, receive, etc.): use **next-contact player's feet** as landing point. Reuses the exact pattern from `play_annotations.py` lines 364-377 — iterate `actions_sorted[i+1:]`, skip UNKNOWN, take first real contact's player feet via `_player_feet_court_xy()`.
- If no next contact (terminal): existing ball-stopped detection or trajectory fallback.

The `court_side` field is populated from the rally's `team_assignments` and the action's `team` — if the acting team maps to team 0 (near), `court_side="near"`; team 1 (far), `court_side="far"`.

**Side-switch awareness:** The acting team's physical court side is already encoded in `RallyActions.team_assignments` after `reattribute-actions` runs with semantic flips. No additional side-switch logic needed at detection time.

## 2. Per-Team Aggregation with Normalization

### File: `analysis/rallycut/statistics/landing_detector.py` — `compute_landing_heatmaps()`

**Normalization rule:**
- Canonical view: team's own half at bottom (Y=8-16m), opponent's half at top (Y=0-8m)
- When acting team was on "near" side (Y=8-16m in physical coords), their serve/attack targets land on far side (Y=0-8m) — already in canonical position, no flip
- When acting team was on "far" side (Y=0-8m), their targets land on near side (Y=8-16m) — flip: `y = 16 - y`, `x = 8 - x`
- After normalization, all points for a team's serves/attacks sit in Y=0-8m (opponent's half of the canonical view)

**Output structure:**
```python
{
    "teamA": {
        "serve": {"grid": [[float]*8]*4, "count": int},  # 4 rows x 8 cols (2m x 1m cells)
        "attack": {"grid": [[float]*8]*4, "count": int},
        "all": {"grid": [[float]*8]*4, "count": int},
        "points": [LandingPoint.to_dict(), ...]  # normalized coords
    },
    "teamB": { ... },
    "perRally": {
        "<rallyId>": {
            "points": [LandingPoint.to_dict(), ...],  # raw court coords
            "servingTeam": "A" | "B"
        }
    }
}
```

Grids are **4 rows x 8 cols** (each cell = 2m tall x 1m wide on one 8x8m half-court). Normalized (sum=1.0).

**Per-player readiness:** Each point carries `playerTrackId`. When per-player filtering is added, the frontend filters `points` by player ID — no backend changes needed.

## 3. Match Stats Integration

### File: `analysis/rallycut/statistics/match_stats.py`

The `compute_match_stats()` call to `detect_rally_landings()` remains the same. The change is in `compute_landing_heatmaps()` which now receives the team context needed for normalization.

**Side-switch info source:** `RallyActions.team_assignments` already reflects side switches (via `reattribute-actions` with semantic flips). The `court_side` on each `LandingPoint` is derived from this at detection time.

### File: `analysis/rallycut/cli/commands/compute_match_stats.py`

No changes beyond what v1 already provides. `positions_raw_map` is already threaded through.

## 4. Web — Per-Team Heatmaps

### File: `web/src/services/api.ts`

Updated `landingHeatmaps` type:
```typescript
landingHeatmaps?: {
    teamA: {
        serve: { grid: number[][]; count: number };
        attack: { grid: number[][]; count: number };
        all: { grid: number[][]; count: number };
        points: LandingPointData[];
    };
    teamB: { ... };
    perRally: Record<string, {
        points: LandingPointData[];
        servingTeam: string;
    }>;
};
```

### File: `web/src/components/CourtHeatmap.tsx`

- Support half-court rendering: SVG viewBox `0 0 800 800` (8m x 8m)
- Grid cells: 8 cols x 4 rows
- Net line at bottom, opponent baseline at top
- Same color scale and styling as v1

### File: `web/src/components/MatchStatsPanel.tsx`

Side-by-side Team A / Team B heatmaps:
```
Landing Zones
[All] [Serve] [Attack]

Team A              Team B
┌──────────┐        ┌──────────┐
│ opponent  │        │ opponent │
│  ●● ●    │        │   ● ●●  │
╠══════════╣        ╠══════════╣
│ (own)    │        │ (own)    │
└──────────┘        └──────────┘
23 serves, 12 atk    21 serves, 14 atk
```

## 5. Per-Rally Debug Overlay

### File: `web/src/stores/playerTrackingStore.ts`

Add `showLandingZones: boolean` to store state + toggle action.

### File: `web/src/components/PlayerTrackingToolbar.tsx`

Add "Landing Zones" toggle button alongside existing Players/Ball/Court toggles.

### File: `web/src/components/RallyLandingOverlay.tsx` (NEW)

- Small (120x240px) semi-transparent court diagram in bottom-right of video player
- **Full court view** (not half-court) — shows raw positions for debugging
- Points color-coded: coral for serves, teal for attacks
- Team A = filled circles, Team B = outlined circles
- Data from `matchStatsJson.landingHeatmaps.perRally[currentRallyId]`
- Hidden when: toggle off, no rally selected, or no landing data for rally

## Files Changed

| File | Action |
|------|--------|
| `analysis/rallycut/statistics/landing_detector.py` | EDIT — mid-rally attacks, court_side field, per-team normalization, restructured output |
| `analysis/rallycut/statistics/match_stats.py` | EDIT — minor: pass context for normalization |
| `web/src/services/api.ts` | EDIT — updated landingHeatmaps type |
| `web/src/components/CourtHeatmap.tsx` | EDIT — half-court support |
| `web/src/components/MatchStatsPanel.tsx` | EDIT — side-by-side team heatmaps |
| `web/src/components/RallyLandingOverlay.tsx` | NEW — per-rally debug overlay |
| `web/src/stores/playerTrackingStore.ts` | EDIT — add showLandingZones state |
| `web/src/components/PlayerTrackingToolbar.tsx` | EDIT — add toggle button |

No new API endpoints. No DB schema changes. No new dependencies.

## Verification

1. **Accuracy**: Re-run on a video with known side switches, verify serve targets flip correctly to canonical half-court view
2. **Mid-rally attacks**: Check a rally with `serve → receive → set → attack → dig → ...` — both attacks should appear
3. **Debug overlay**: Select rallies one by one, verify overlay matches expected landing positions
4. **Backfill**: Re-run compute-match-stats + DB update for all 60 videos
5. **Type check**: mypy + tsc pass
6. **Visual**: Start dev server, open match stats for a calibrated video, verify side-by-side team heatmaps render
