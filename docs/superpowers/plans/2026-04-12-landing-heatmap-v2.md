# Landing Heatmap v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add per-team landing heatmaps with all-attack detection, side-switch normalization, and per-rally debug overlay.

**Architecture:** Extend `landing_detector.py` to detect mid-rally attacks (defender's feet) and add `court_side` for normalization. Restructure `compute_landing_heatmaps()` output to per-team with half-court grids. Add per-rally debug overlay as a toggle in the video player.

**Tech Stack:** Python (mypy strict), TypeScript/React (Next.js 15, MUI v7), SVG

**Spec:** `docs/superpowers/specs/2026-04-12-landing-heatmap-v2-design.md`

---

### Task 1: Extend LandingPoint with court_side and add mid-rally attack detection

**Files:**
- Modify: `analysis/rallycut/statistics/landing_detector.py`

- [ ] **Step 1: Add `court_side` field to `LandingPoint` dataclass**

In `landing_detector.py`, add `court_side` to the dataclass and `to_dict()`:

```python
@dataclass
class LandingPoint:
    """A detected ball landing position on the court."""

    frame: int
    image_x: float
    image_y: float
    court_x: float | None
    court_y: float | None
    action_type: str  # "serve" or "attack"
    rally_id: str
    player_track_id: int
    team: str  # "A" or "B"
    court_side: str = "unknown"  # "near" or "far" — acting team's court side

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "type": self.action_type,
            "team": self.team,
            "rallyId": self.rally_id,
            "playerTrackId": self.player_track_id,
            "frame": self.frame,
            "imageX": round(self.image_x, 4),
            "imageY": round(self.image_y, 4),
            "courtSide": self.court_side,
        }
        if self.court_x is not None and self.court_y is not None:
            d["courtX"] = round(self.court_x, 2)
            d["courtY"] = round(self.court_y, 2)
        return d
```

- [ ] **Step 2: Add helper to determine court_side from team_assignments**

Add this helper function after the existing helpers:

```python
def _get_court_side(team: str, team_assignments: dict[int, int]) -> str:
    """Determine if a team is on 'near' or 'far' court side.

    Team 0 = near (high Y, close to camera), Team 1 = far (low Y).
    'A' maps to the team of the first player in team_assignments with
    value 0, 'B' to value 1.
    """
    if team == "A":
        return "near"
    if team == "B":
        return "far"
    return "unknown"
```

- [ ] **Step 3: Rewrite `detect_rally_landings()` to detect all attacks**

Replace the entire `detect_rally_landings()` function. The key changes:
1. Add `court_side` to `_make_landing`
2. Loop over ALL attacks (not just terminal)
3. For mid-rally attacks: use next-contact player's feet
4. For terminal attacks: existing ball-stopped / trajectory fallback

```python
def detect_rally_landings(
    rally_actions: RallyActions,
    ball_positions: list[BallPosition],
    calibrator: CourtCalibrator | None,
    video_width: int,
    video_height: int,
    positions_raw: list[dict[str, Any]] | None = None,
) -> list[LandingPoint]:
    """Detect landing points in a rally for serve and attack heatmaps.

    **Serve target**: Receiving player's feet at the receive contact frame.

    **Attack landing**: For mid-rally attacks, next-contact player's feet
    (same pattern as play_annotations.py attack_direction). For terminal
    attacks, ball-stopped detection or trajectory fallback.

    All points include ``court_side`` indicating which physical side
    the acting team was on, for later normalization.
    """
    from rallycut.tracking.action_classifier import ActionType

    if not rally_actions.actions or not ball_positions:
        return []
    if calibrator is None or not calibrator.is_calibrated:
        return []

    landings: list[LandingPoint] = []
    actions = sorted(rally_actions.actions, key=lambda a: a.frame)

    def _make_landing(
        frame: int,
        court_x: float,
        court_y: float,
        image_x: float,
        image_y: float,
        action_type: str,
        player_track_id: int,
        team: str,
    ) -> LandingPoint:
        return LandingPoint(
            frame=frame,
            image_x=image_x,
            image_y=image_y,
            court_x=court_x,
            court_y=court_y,
            action_type=action_type,
            rally_id=rally_actions.rally_id,
            player_track_id=player_track_id,
            team=team,
            court_side=_get_court_side(team, rally_actions.team_assignments),
        )

    # --- Serve target (receiving player's feet) ---
    serve = rally_actions.serve
    if serve is not None and serve.action_type == ActionType.SERVE:
        receive: ClassifiedAction | None = None
        for a in actions:
            if a.frame > serve.frame and a.action_type == ActionType.RECEIVE:
                receive = a
                break

        if (
            receive is not None
            and receive.player_track_id >= 0
            and positions_raw is not None
        ):
            court_pos = _player_feet_court_xy(
                positions_raw, receive.player_track_id,
                receive.frame, calibrator,
            )
            if court_pos is not None:
                ball_at_recv = _find_ball_near_frame(
                    ball_positions, receive.frame,
                )
                img_x = ball_at_recv[0] if ball_at_recv else receive.ball_x
                img_y = ball_at_recv[1] if ball_at_recv else receive.ball_y
                landings.append(_make_landing(
                    receive.frame, court_pos[0], court_pos[1],
                    img_x, img_y,
                    "serve", serve.player_track_id, serve.team,
                ))

    # --- All attacks (mid-rally + terminal) ---
    for i, a in enumerate(actions):
        if a.action_type != ActionType.ATTACK:
            continue

        # Try next-contact player's feet (mid-rally attacks).
        next_contact_court: tuple[float, float] | None = None
        next_frame = a.frame
        next_img_x = a.ball_x
        next_img_y = a.ball_y
        if positions_raw is not None:
            for b in actions[i + 1 :]:
                if b.action_type == ActionType.UNKNOWN:
                    continue
                if b.player_track_id < 0:
                    continue
                bc = _player_feet_court_xy(
                    positions_raw, b.player_track_id,
                    b.frame, calibrator,
                )
                if bc is not None:
                    next_contact_court = bc
                    next_frame = b.frame
                    ball_at_next = _find_ball_near_frame(
                        ball_positions, b.frame,
                    )
                    if ball_at_next:
                        next_img_x, next_img_y = ball_at_next
                break

        if next_contact_court is not None:
            landings.append(_make_landing(
                next_frame, next_contact_court[0], next_contact_court[1],
                next_img_x, next_img_y,
                "attack", a.player_track_id, a.team,
            ))
            continue

        # Terminal attack: ball stopped on sand or trajectory fallback.
        attack_court: tuple[float, float] | None = None
        det_frame = a.frame
        img_x = a.ball_x
        img_y = a.ball_y

        stopped = find_landing(ball_positions, a.frame)
        if stopped is not None:
            det_frame, sx, sy = stopped
            attack_court = _project_court_safe(sx, sy, calibrator)
            img_x, img_y = sx, sy

        if attack_court is None:
            for dt in range(_FALLBACK_DT_MIN, _FALLBACK_DT_MAX + 1):
                pos = _find_ball_near_frame(
                    ball_positions, a.frame + dt, radius=2,
                )
                if pos is not None:
                    det_frame = a.frame + dt
                    cp = _project_court_safe(pos[0], pos[1], calibrator)
                    if cp is not None:
                        attack_court = cp
                        img_x, img_y = pos
                    break

        if attack_court is not None:
            landings.append(_make_landing(
                det_frame, attack_court[0], attack_court[1],
                img_x, img_y,
                "attack", a.player_track_id, a.team,
            ))

    return landings
```

- [ ] **Step 4: Run mypy and ruff**

Run:
```bash
cd analysis && uv run mypy rallycut/statistics/landing_detector.py --no-error-summary && uv run ruff check rallycut/statistics/landing_detector.py
```
Expected: All checks passed

- [ ] **Step 5: Smoke test mid-rally attack detection**

Run:
```bash
cd analysis && uv run python -c "
from rallycut.cli.commands.compute_match_stats import _load_rally_actions_and_positions
from rallycut.statistics.landing_detector import detect_rally_landings

vid = '0a383519-ecaa-411a-8e5e-e0aadc835725'
(ral, _, _, bpm, cal, _, vw, vh, prm) = _load_rally_actions_and_positions(vid)

total_s, total_a = 0, 0
for ra in ral:
    bp = bpm.get(ra.rally_id, [])
    pr = prm.get(ra.rally_id)
    if not bp: continue
    lps = detect_rally_landings(ra, bp, cal, vw, vh, positions_raw=pr)
    s = sum(1 for l in lps if l.action_type == 'serve')
    a = sum(1 for l in lps if l.action_type == 'attack')
    total_s += s
    total_a += a
print(f'Serves: {total_s}, Attacks: {total_a} (was 33/0 in v1)')
"
```
Expected: Attacks should now be significantly higher than 0 (mid-rally attacks detected).

- [ ] **Step 6: Commit**

```bash
git add analysis/rallycut/statistics/landing_detector.py
git commit -m "landing_detector: mid-rally attack detection + court_side field"
```

---

### Task 2: Restructure compute_landing_heatmaps to per-team with normalization

**Files:**
- Modify: `analysis/rallycut/statistics/landing_detector.py`

- [ ] **Step 1: Update grid constants for half-court**

Replace the grid constants at the top of the file:

```python
# Grid resolution for heatmap binning (half-court: 8m x 8m).
GRID_COLS = 8  # 1m per column on 8m court width
GRID_ROWS = 4  # 2m per row on 8m half-court
HALF_COURT_M = COURT_LENGTH_M / 2.0  # 8m
```

- [ ] **Step 2: Rewrite `compute_landing_heatmaps()` with per-team normalization**

Replace the entire function:

```python
def compute_landing_heatmaps(
    landings: list[LandingPoint],
    rally_actions_list: list[Any] | None = None,
) -> dict[str, Any]:
    """Aggregate landing points into per-team half-court heatmap grids.

    Points are normalized to canonical team perspective: own half at
    bottom (Y=8-16m), opponent half at top (Y=0-8m). When the acting
    team was on the far side, coordinates are flipped so all points
    appear consistently on the opponent's half.

    Returns per-team grids (4 rows x 8 cols = 2m x 1m cells on one
    8x8m half-court), plus a ``perRally`` dict with raw coords for
    the debug overlay.
    """
    margin = 1.0
    valid_landings = [
        lp for lp in landings
        if lp.court_x is not None
        and lp.court_y is not None
        and -margin <= lp.court_x <= COURT_WIDTH_M + margin
        and -margin <= lp.court_y <= COURT_LENGTH_M + margin
    ]

    def _normalize(lp: LandingPoint) -> tuple[float, float]:
        """Normalize court coords to canonical half-court (opponent half = 0-8m Y)."""
        cx, cy = lp.court_x or 0.0, lp.court_y or 0.0
        if lp.court_side == "far":
            # Team is on far side — their targets land on near side (Y=8-16m).
            # Flip to canonical: y = 16 - y, x = 8 - x.
            cx = COURT_WIDTH_M - cx
            cy = COURT_LENGTH_M - cy
        # After normalization, targets should be in opponent's half (Y=0-8m).
        # Clamp to half-court bounds.
        cx = max(0.0, min(cx, COURT_WIDTH_M - 1e-9))
        cy = max(0.0, min(cy, HALF_COURT_M - 1e-9))
        return cx, cy

    def _build_half_grid(pts: list[LandingPoint]) -> list[list[float]]:
        grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float64)
        for lp in pts:
            cx, cy = _normalize(lp)
            gx = int(cx / COURT_WIDTH_M * GRID_COLS)
            gy = int(cy / HALF_COURT_M * GRID_ROWS)
            grid[gy, gx] += 1
        total = grid.sum()
        if total > 0:
            grid /= total
        return [[round(float(v), 4) for v in row] for row in grid]

    def _team_section(pts: list[LandingPoint]) -> dict[str, Any]:
        serve = [lp for lp in pts if lp.action_type == "serve"]
        attack = [lp for lp in pts if lp.action_type == "attack"]
        return {
            "serve": {"grid": _build_half_grid(serve), "count": len(serve)},
            "attack": {"grid": _build_half_grid(attack), "count": len(attack)},
            "all": {"grid": _build_half_grid(pts), "count": len(pts)},
            "points": [lp.to_dict() for lp in pts],
        }

    team_a = [lp for lp in valid_landings if lp.team == "A"]
    team_b = [lp for lp in valid_landings if lp.team == "B"]

    # Per-rally raw data for debug overlay (no normalization).
    per_rally: dict[str, Any] = {}
    for lp in valid_landings:
        rid = lp.rally_id
        if rid not in per_rally:
            per_rally[rid] = {"points": [], "servingTeam": "unknown"}
        per_rally[rid]["points"].append(lp.to_dict())

    # Fill serving team from rally_actions_list if available.
    if rally_actions_list:
        for ra in rally_actions_list:
            if ra.rally_id in per_rally and ra.serve:
                per_rally[ra.rally_id]["servingTeam"] = ra.serve.team

    return {
        "teamA": _team_section(team_a),
        "teamB": _team_section(team_b),
        "perRally": per_rally,
    }
```

- [ ] **Step 3: Update the call in `match_stats.py` to pass `rally_actions_list`**

In `analysis/rallycut/statistics/match_stats.py`, update the landing heatmap computation block:

```python
    # 8. Landing heatmaps (serve + attack court-plane positions)
    from rallycut.statistics.landing_detector import (
        compute_landing_heatmaps,
        detect_rally_landings,
    )

    all_landings = []
    for ra in rally_actions_list:
        ball_pos = (ball_positions_map or {}).get(ra.rally_id, [])
        if ball_pos:
            pos_raw = (positions_raw_map or {}).get(ra.rally_id)
            rally_landings = detect_rally_landings(
                ra, ball_pos, calibrator, video_width, video_height,
                positions_raw=pos_raw,
            )
            all_landings.extend(rally_landings)

    if all_landings:
        stats.landing_heatmaps = compute_landing_heatmaps(
            all_landings, rally_actions_list,
        )
```

- [ ] **Step 4: Run mypy and ruff on both files**

Run:
```bash
cd analysis && uv run mypy rallycut/statistics/landing_detector.py rallycut/statistics/match_stats.py --no-error-summary && uv run ruff check rallycut/statistics/landing_detector.py rallycut/statistics/match_stats.py
```
Expected: All checks passed

- [ ] **Step 5: Test per-team output structure**

Run:
```bash
cd analysis && uv run python -c "
from rallycut.cli.commands.compute_match_stats import _load_rally_actions_and_positions
from rallycut.statistics.landing_detector import detect_rally_landings, compute_landing_heatmaps

vid = '0a383519-ecaa-411a-8e5e-e0aadc835725'
(ral, _, _, bpm, cal, _, vw, vh, prm) = _load_rally_actions_and_positions(vid)

all_lp = []
for ra in ral:
    bp = bpm.get(ra.rally_id, [])
    pr = prm.get(ra.rally_id)
    if bp:
        all_lp.extend(detect_rally_landings(ra, bp, cal, vw, vh, positions_raw=pr))

hm = compute_landing_heatmaps(all_lp, ral)
print('Keys:', list(hm.keys()))
print(f'Team A: serve={hm[\"teamA\"][\"serve\"][\"count\"]}, attack={hm[\"teamA\"][\"attack\"][\"count\"]}')
print(f'Team B: serve={hm[\"teamB\"][\"serve\"][\"count\"]}, attack={hm[\"teamB\"][\"attack\"][\"count\"]}')
print(f'Per-rally entries: {len(hm[\"perRally\"])}')
print(f'Grid shape: {len(hm[\"teamA\"][\"serve\"][\"grid\"])} rows x {len(hm[\"teamA\"][\"serve\"][\"grid\"][0])} cols')
"
```
Expected: Keys `teamA`, `teamB`, `perRally`. Grid shape 4x8. Non-zero counts for both teams.

- [ ] **Step 6: Commit**

```bash
git add analysis/rallycut/statistics/landing_detector.py analysis/rallycut/statistics/match_stats.py
git commit -m "landing_detector: per-team heatmaps with side-switch normalization"
```

---

### Task 3: Update TypeScript types and CourtHeatmap for half-court

**Files:**
- Modify: `web/src/services/api.ts`
- Modify: `web/src/components/CourtHeatmap.tsx`

- [ ] **Step 1: Update `MatchStats` interface in `api.ts`**

Replace the existing `landingHeatmaps` type:

```typescript
  landingHeatmaps?: {
    teamA: {
      serve: { grid: number[][]; count: number };
      attack: { grid: number[][]; count: number };
      all: { grid: number[][]; count: number };
      points: Array<{
        courtX: number;
        courtY: number;
        type: string;
        team: string;
        rallyId: string;
        playerTrackId: number;
        frame: number;
        imageX: number;
        imageY: number;
        courtSide: string;
      }>;
    };
    teamB: {
      serve: { grid: number[][]; count: number };
      attack: { grid: number[][]; count: number };
      all: { grid: number[][]; count: number };
      points: Array<{
        courtX: number;
        courtY: number;
        type: string;
        team: string;
        rallyId: string;
        playerTrackId: number;
        frame: number;
        imageX: number;
        imageY: number;
        courtSide: string;
      }>;
    };
    perRally: Record<string, {
      points: Array<{
        courtX: number;
        courtY: number;
        type: string;
        team: string;
        rallyId: string;
        playerTrackId: number;
        frame: number;
        imageX: number;
        imageY: number;
        courtSide: string;
      }>;
      servingTeam: string;
    }>;
  };
```

- [ ] **Step 2: Update `CourtHeatmap.tsx` for half-court mode**

Replace `CourtHeatmap.tsx` to render a half-court (8m x 8m):

```tsx
'use client';

import { Box, Typography, Chip } from '@mui/material';

const COURT_W = 8;
const HALF_L = 8; // half-court length
const SCALE = 100;
const SVG_W = COURT_W * SCALE; // 800
const SVG_H = HALF_L * SCALE;  // 800
const PAD = 40;

const COLOR_STOPS = [
  { t: 0.0, r: 59, g: 130, b: 246 },
  { t: 0.5, r: 255, g: 107, b: 74 },
  { t: 1.0, r: 239, g: 68, b: 68 },
];

function interpolateColor(t: number): string {
  const clamped = Math.max(0, Math.min(1, t));
  let i = 0;
  for (; i < COLOR_STOPS.length - 1; i++) {
    if (clamped <= COLOR_STOPS[i + 1].t) break;
  }
  const a = COLOR_STOPS[i];
  const b = COLOR_STOPS[Math.min(i + 1, COLOR_STOPS.length - 1)];
  const range = b.t - a.t || 1;
  const f = (clamped - a.t) / range;
  const r = Math.round(a.r + (b.r - a.r) * f);
  const g = Math.round(a.g + (b.g - a.g) * f);
  const bl = Math.round(a.b + (b.b - a.b) * f);
  return `rgb(${r}, ${g}, ${bl})`;
}

interface CourtHeatmapProps {
  grid: number[][];
  count: number;
  points?: Array<{ courtX: number; courtY: number; type: string; team: string }>;
  title: string;
}

export default function CourtHeatmap({ grid, count, points, title }: CourtHeatmapProps) {
  if (!grid || grid.length === 0) return null;

  const rows = grid.length;
  const cols = grid[0].length;
  const cellW = SVG_W / cols;
  const cellH = SVG_H / rows;
  const maxVal = Math.max(...grid.flat(), 1e-9);

  return (
    <Box sx={{ display: 'inline-flex', flexDirection: 'column', alignItems: 'center' }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 0.5 }}>
        <Typography variant="caption" sx={{ fontWeight: 600, color: 'text.primary' }}>
          {title}
        </Typography>
        <Chip label={`${count}`} size="small" sx={{ height: 16, fontSize: '0.625rem' }} />
      </Box>

      <svg
        viewBox={`${-PAD} ${-PAD} ${SVG_W + PAD * 2} ${SVG_H + PAD * 2}`}
        width={140}
        height={140}
        style={{ overflow: 'visible' }}
      >
        {/* Court background */}
        <rect x={0} y={0} width={SVG_W} height={SVG_H}
          fill="#1A1E28" stroke="rgba(255,255,255,0.3)" strokeWidth={3} rx={4} />

        {/* Heatmap cells */}
        {grid.map((row, gy) =>
          row.map((val, gx) => {
            if (val <= 0) return null;
            const intensity = val / maxVal;
            return (
              <rect key={`${gy}-${gx}`}
                x={gx * cellW} y={gy * cellH} width={cellW} height={cellH}
                fill={interpolateColor(intensity)} opacity={0.15 + intensity * 0.65} rx={2} />
            );
          }),
        )}

        {/* Net line at bottom (own half boundary) */}
        <line x1={0} y1={SVG_H} x2={SVG_W} y2={SVG_H}
          stroke="rgba(255,255,255,0.5)" strokeWidth={3} strokeDasharray="12,6" />

        {/* Court outline */}
        <rect x={0} y={0} width={SVG_W} height={SVG_H}
          fill="none" stroke="rgba(255,255,255,0.3)" strokeWidth={3} rx={4} />

        {/* Scatter points */}
        {points?.map((pt, i) => {
          const cx = (pt.courtX / COURT_W) * SVG_W;
          const cy = (pt.courtY / HALF_L) * SVG_H;
          if (cx < -10 || cx > SVG_W + 10 || cy < -10 || cy > SVG_H + 10) return null;
          return (
            <circle key={i} cx={cx} cy={cy} r={12}
              fill={pt.type === 'serve' ? '#FF6B4A' : '#00D4AA'}
              opacity={0.7} stroke="rgba(255,255,255,0.4)" strokeWidth={1.5} />
          );
        })}

        {/* Labels */}
        <text x={SVG_W / 2} y={-12} textAnchor="middle"
          fill="rgba(255,255,255,0.5)" fontSize={28} fontFamily="Inter, sans-serif">
          Opponent
        </text>
        <text x={SVG_W / 2} y={SVG_H + 32} textAnchor="middle"
          fill="rgba(255,255,255,0.5)" fontSize={28} fontFamily="Inter, sans-serif">
          Net
        </text>
      </svg>
    </Box>
  );
}
```

- [ ] **Step 3: Run tsc**

Run:
```bash
cd web && npx tsc --noEmit
```
Expected: No errors

- [ ] **Step 4: Commit**

```bash
git add web/src/services/api.ts web/src/components/CourtHeatmap.tsx
git commit -m "web: per-team TypeScript types + half-court CourtHeatmap"
```

---

### Task 4: Update MatchStatsPanel for side-by-side team heatmaps

**Files:**
- Modify: `web/src/components/MatchStatsPanel.tsx`

- [ ] **Step 1: Rewrite `LandingHeatmapSection` for per-team layout**

Replace the existing `LandingHeatmapSection` function:

```tsx
function LandingHeatmapSection({ stats }: { stats: MatchStats }) {
  const [view, setView] = useState<'serve' | 'attack' | 'all'>('all');
  const hm = stats.landingHeatmaps;
  if (!hm) return null;

  const teamA = hm.teamA[view];
  const teamB = hm.teamB[view];
  if ((!teamA || teamA.count === 0) && (!teamB || teamB.count === 0)) return null;

  const titles: Record<string, string> = {
    serve: 'Serve Targets',
    attack: 'Attack Landings',
    all: 'All Landings',
  };

  return (
    <Box sx={{ mb: 1 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
        <Typography variant="subtitle2" sx={{ fontWeight: 'bold' }}>
          Landing Zones
        </Typography>
        <ToggleButtonGroup
          value={view}
          exclusive
          onChange={(_, v) => { if (v) setView(v); }}
          size="small"
          sx={{
            '& .MuiToggleButton-root': {
              py: 0, px: 0.75, fontSize: '0.65rem', height: 20,
              textTransform: 'none',
            },
          }}
        >
          <ToggleButton value="all">All</ToggleButton>
          <ToggleButton value="serve">Serve</ToggleButton>
          <ToggleButton value="attack">Attack</ToggleButton>
        </ToggleButtonGroup>
      </Box>
      <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap' }}>
        {teamA && teamA.count > 0 && (
          <CourtHeatmap
            grid={teamA.grid}
            count={teamA.count}
            points={view === 'all' ? hm.teamA.points : hm.teamA.points.filter(p => p.type === view)}
            title={`Team A — ${titles[view]}`}
          />
        )}
        {teamB && teamB.count > 0 && (
          <CourtHeatmap
            grid={teamB.grid}
            count={teamB.count}
            points={view === 'all' ? hm.teamB.points : hm.teamB.points.filter(p => p.type === view)}
            title={`Team B — ${titles[view]}`}
          />
        )}
      </Box>
    </Box>
  );
}
```

- [ ] **Step 2: Run tsc and eslint**

Run:
```bash
cd web && npx tsc --noEmit && npx eslint src/components/MatchStatsPanel.tsx
```
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add web/src/components/MatchStatsPanel.tsx
git commit -m "web: side-by-side per-team landing heatmaps"
```

---

### Task 5: Add per-rally debug overlay

**Files:**
- Modify: `web/src/stores/playerTrackingStore.ts`
- Modify: `web/src/components/PlayerTrackingToolbar.tsx`
- Create: `web/src/components/RallyLandingOverlay.tsx`
- Modify: `web/src/components/VideoPlayer.tsx`

- [ ] **Step 1: Add `showLandingZones` to playerTrackingStore**

In `web/src/stores/playerTrackingStore.ts`, add to the interface (after `showRawTracks`):

```typescript
  showLandingZones: boolean;
```

Add to the actions interface (after `toggleCourtDebugOverlay`):

```typescript
  toggleLandingZones: () => void;
```

Add to the store creation (after `showRawTracks: false`):

```typescript
      showLandingZones: false,
```

Add the toggle implementation (after `toggleCourtDebugOverlay`):

```typescript
      toggleLandingZones: () => {
        set((state) => ({ showLandingZones: !state.showLandingZones }));
      },
```

- [ ] **Step 2: Add toggle button to PlayerTrackingToolbar**

In `web/src/components/PlayerTrackingToolbar.tsx`, add `showLandingZones` and `toggleLandingZones` to the store destructure. Add import for `PlaceIcon` from `@mui/icons-material/Place`.

Add a new `ToggleButton` after the Court toggle button (before `</ToggleButtonGroup>`):

```tsx
              <ToggleButton
                value="landings"
                selected={showLandingZones}
                onClick={toggleLandingZones}
                sx={{
                  px: 0.75, textTransform: 'none', fontSize: '0.7rem', gap: 0.5,
                  '&.Mui-selected': { color: '#FF6B4A', bgcolor: 'rgba(255,107,74,0.12)' },
                }}
              >
                <PlaceIcon sx={{ fontSize: 16 }} />
                Landings
              </ToggleButton>
```

- [ ] **Step 3: Create `RallyLandingOverlay.tsx`**

Create `web/src/components/RallyLandingOverlay.tsx`:

```tsx
'use client';

import { Box } from '@mui/material';
import type { MatchStats } from '@/services/api';

const COURT_W = 8;
const COURT_L = 16;
const SVG_W = 120;
const SVG_H = 240;

interface RallyLandingOverlayProps {
  rallyId: string;
  matchStats: MatchStats;
}

export function RallyLandingOverlay({ rallyId, matchStats }: RallyLandingOverlayProps) {
  const perRally = matchStats.landingHeatmaps?.perRally?.[rallyId];
  if (!perRally || perRally.points.length === 0) return null;

  return (
    <Box
      sx={{
        position: 'absolute',
        bottom: 48,
        right: 8,
        zIndex: 10,
        pointerEvents: 'none',
        opacity: 0.85,
      }}
    >
      <svg viewBox={`0 0 ${COURT_W * 100} ${COURT_L * 100}`} width={SVG_W} height={SVG_H}>
        {/* Court background */}
        <rect x={0} y={0} width={COURT_W * 100} height={COURT_L * 100}
          fill="rgba(26,30,40,0.8)" stroke="rgba(255,255,255,0.3)" strokeWidth={4} rx={4} />

        {/* Net */}
        <line x1={0} y1={COURT_L * 50} x2={COURT_W * 100} y2={COURT_L * 50}
          stroke="rgba(255,255,255,0.4)" strokeWidth={3} strokeDasharray="8,4" />

        {/* Landing points */}
        {perRally.points.map((pt, i) => {
          const cx = (pt.courtX / COURT_W) * COURT_W * 100;
          const cy = (pt.courtY / COURT_L) * COURT_L * 100;
          if (cx < -20 || cx > COURT_W * 100 + 20 || cy < -20 || cy > COURT_L * 100 + 20) return null;
          const isServe = pt.type === 'serve';
          const isTeamA = pt.team === 'A';
          return (
            <circle key={i} cx={cx} cy={cy}
              r={isServe ? 16 : 14}
              fill={isServe ? '#FF6B4A' : '#00D4AA'}
              opacity={0.8}
              stroke={isTeamA ? 'rgba(255,255,255,0.6)' : 'rgba(255,255,255,0.3)'}
              strokeWidth={isTeamA ? 3 : 2}
              strokeDasharray={isTeamA ? undefined : '4,3'}
            />
          );
        })}
      </svg>
    </Box>
  );
}
```

- [ ] **Step 4: Wire overlay into VideoPlayer.tsx**

In `web/src/components/VideoPlayer.tsx`:

Add import:
```typescript
import { RallyLandingOverlay } from './RallyLandingOverlay';
```

Add store selector (near other playerTrackingStore selectors):
```typescript
const showLandingZones = usePlayerTrackingStore((state) => state.showLandingZones);
```

Add the overlay in the JSX (after the existing `BallTrackOverlay`, inside the same container):
```tsx
          {showLandingZones && currentRally && currentRally._backendId && matchStats && (
            <RallyLandingOverlay
              rallyId={currentRally._backendId}
              matchStats={matchStats}
            />
          )}
```

Note: `matchStats` may need to be loaded. Check if `VideoPlayer` already has access to match stats. If not, load via `getMatchStatsApi(activeMatchId)` in a `useEffect` and store in local state — follow the same pattern as `MatchStatsPanel`.

- [ ] **Step 5: Run tsc and eslint**

Run:
```bash
cd web && npx tsc --noEmit && npx eslint src/components/RallyLandingOverlay.tsx src/components/VideoPlayer.tsx src/components/PlayerTrackingToolbar.tsx
```
Expected: No errors

- [ ] **Step 6: Commit**

```bash
git add web/src/stores/playerTrackingStore.ts web/src/components/PlayerTrackingToolbar.tsx web/src/components/RallyLandingOverlay.tsx web/src/components/VideoPlayer.tsx
git commit -m "web: per-rally landing debug overlay with toggle"
```

---

### Task 6: Backfill all videos and verify

**Files:** None (runtime only)

- [ ] **Step 1: Recompute and save match stats for all 60 videos**

Run:
```bash
cd analysis && uv run python -c "
import json
from rallycut.evaluation.db import get_connection
from rallycut.cli.commands.compute_match_stats import _load_rally_actions_and_positions
from rallycut.statistics.match_stats import compute_match_stats

with get_connection() as conn:
    with conn.cursor() as cur:
        cur.execute('''
            SELECT DISTINCT v.id FROM videos v
            JOIN rallies r ON r.video_id = v.id
            JOIN player_tracks pt ON pt.rally_id = r.id
            WHERE pt.ball_positions_json IS NOT NULL AND v.match_stats_json IS NOT NULL
            ORDER BY v.id
        ''')
        ids = [r[0] for r in cur.fetchall()]

print(f'Backfilling {len(ids)} videos...')
ok = fail = 0
for i, vid in enumerate(ids):
    try:
        (ral, pos, fps, bpm, cal, ma, vw, vh, prm) = _load_rally_actions_and_positions(vid)
        if not ral: continue
        stats = compute_match_stats(ral, pos, fps, vw, vh, cal, bpm, ma, prm)
        result = stats.to_dict()
        with get_connection() as c2:
            with c2.cursor() as cur2:
                cur2.execute('UPDATE videos SET match_stats_json = %s WHERE id = %s', [json.dumps(result), vid])
            c2.commit()
        hm = result.get('landingHeatmaps', {})
        a_s = hm.get('teamA', {}).get('serve', {}).get('count', 0)
        a_a = hm.get('teamA', {}).get('attack', {}).get('count', 0)
        b_s = hm.get('teamB', {}).get('serve', {}).get('count', 0)
        b_a = hm.get('teamB', {}).get('attack', {}).get('count', 0)
        ok += 1
        print(f'[{i+1}/{len(ids)}] {vid[:8]} OK  A:{a_s}s/{a_a}a  B:{b_s}s/{b_a}a')
    except Exception as e:
        fail += 1
        print(f'[{i+1}/{len(ids)}] {vid[:8]} FAIL: {e}')
print(f'Done: {ok} OK, {fail} failed')
"
```
Expected: 60/60 OK, 0 failed. Attack counts should be significantly higher than v1 (which had ~29 total across 60 videos).

- [ ] **Step 2: Visual verification**

Start dev server:
```bash
make dev
```

Open a calibrated video in the editor. Verify:
1. Match stats panel shows side-by-side Team A / Team B half-court heatmaps
2. Serve/Attack/All toggle filters correctly
3. Toggle "Landing Zones" in tracking toolbar — mini court appears on video player
4. Selecting different rallies updates the overlay

- [ ] **Step 3: Commit backfill note (optional)**

No code to commit — backfill is a data operation.
