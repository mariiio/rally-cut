# RallyCut Web

Rally editor web application for volleyball video editing.

## Stack

- Next.js 15 (App Router), React 19, TypeScript
- MUI v7, Emotion for styling
- Zustand for state management
- @dnd-kit for drag-and-drop

## Commands

```bash
npm run dev          # Development server
npm run build        # Production build
npm run lint         # ESLint
npx tsc --noEmit     # Type check
```

## Structure

```
src/
├── app/              # Next.js pages
│   ├── (landing)/    # Public landing page
│   ├── (app)/        # Protected editor
│   │   ├── sessions/[id]/  # Multi-video session editor
│   │   └── videos/[id]/    # Single video editor
├── components/
│   ├── EditorLayout  # Main 3-panel layout
│   ├── VideoPlayer   # HTML5 video with camera transforms
│   ├── Timeline      # Rally timeline with drag handles
│   ├── RallyList     # Rally list sidebar
│   ├── HighlightsPanel
│   ├── CameraPanel   # Camera edit controls
│   ├── CameraOverlay # Visual preview overlay
│   ├── PlayerTrackingToolbar # Player/ball tracking controls
│   ├── PlayerOverlay # Player bounding box visualization
│   ├── BallTrackOverlay # Ball trajectory visualization
│   └── OriginalQualityBanner # FREE tier upgrade prompt (7-day quality warning)
├── stores/           # Zustand stores (see below)
├── services/
│   ├── api.ts        # REST client
│   └── syncService   # Backend sync (5s debounce)
└── utils/
    ├── videoExport.ts
    └── cameraInterpolation.ts
```

## Zustand Stores

| Store | Purpose |
|-------|---------|
| `editorStore` | Rallies, highlights, undo/redo (50 entries), localStorage + backend sync |
| `playerStore` | Playback state, highlight playlist, buffered ranges |
| `cameraStore` | Per-rally camera edits, keyframes, aspect ratio, handheld motion |
| `uploadStore` | Upload progress, multipart handling, local blob URLs |
| `exportStore` | Export job tracking, download progress |
| `tierStore` | Subscription tier, usage limits |
| `playerTrackingStore` | Player/ball tracking data, court calibration, overlay visibility |

## Camera System

Instagram-style zoom/pan effects per rally:

- **Aspect ratios**: ORIGINAL (16:9) or VERTICAL (9:16)
- **Keyframes**: timeOffset (0-1), positionX/Y (0-1), zoom (1-3x), easing
- **Handheld motion**: OFF/LIGHT/MEDIUM/HEAVY presets (simulates camera shake)
- **Interpolation**: Keyframes blended with easing (LINEAR, EASE_IN, EASE_OUT, EASE_IN_OUT)

Camera edits stored per aspect ratio (switching preserves both).

## Player Tracking

Debug visualization for player and ball detection:

- **PlayerTrackingToolbar**: Controls for tracking, calibration, overlay toggles, track swapping, and Label Studio integration
- **PlayerOverlay**: Renders player bounding boxes on video with interpolation for smooth display
- **BallTrackOverlay**: Renders ball trajectory trail overlay

Features:
- **Court calibration**: 4-corner court mapping for position projection, persisted to DB per video (localStorage as fast cache, synced with API)
- **Primary track filtering**: Only shows the 4 identified players (excludes referees/spectators)
- **Position interpolation**: Fills gaps up to 1.5s and shows last position for 1s after detection ends
- **Track swapping**: "Swap Tracks" dialog fixes ID switches by swapping two track IDs from current frame onward
- **Label Studio integration**: "Label" exports fresh predictions to Label Studio, "Edit" resumes existing annotations, "Save GT" imports corrections

### Ground Truth Labeling (Label Studio)

Workflow for correcting tracking predictions:
1. Track a rally using "Track Players" button
2. Click "Label" to open in Label Studio with pre-filled predictions
3. Correct bounding boxes in Label Studio's interpolation mode
4. Click "Save GT" to import corrections as ground truth

**Labels**: player_1 (green), player_2 (blue), player_3 (orange), player_4 (purple), ball (red)
**Note**: Requires Label Studio running locally at `http://localhost:8082` with `LABEL_STUDIO_API_KEY` configured in API.

## Video Loading Priority

For fast editing of large videos:

1. **localBlobUrl** - Instant playback after upload (from uploadStore)
2. **proxyUrl** - 720p proxy after processing (~30-60s)
3. **videoUrl** - Full quality (fallback)

Also uses: `preload="metadata"`, 1280px poster thumbnail.

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| Space | Play/pause (restart if at end) |
| Arrow Left/Right | Seek ±1s (±0.3s in camera edit mode) |
| Shift + Arrow | Shrink/expand rally end ±0.5s |
| Cmd/Ctrl + Arrow | Jump to prev/next rally |
| Cmd/Ctrl + Enter | Create new rally at cursor |
| Delete/Backspace | Delete keyframe (priority) or rally (double-press) |
| Enter | Toggle rally in highlight |
| M | Mark rally start/end (recording mode) |
| Escape | Cancel deletion / deselect / cancel recording |
| Cmd/Ctrl + Z | Undo |
| Cmd/Ctrl + Shift + Z | Redo |
| [ / ] | Collapse left/right panel |
| Cmd/Ctrl + Shift + C | Open camera panel |

## State Persistence

- **localStorage**: Immediate save on every mutation
- **Backend sync**: 5s debounce after `markDirty()`, paid tiers only (Pro/Elite)
- **Undo/redo**: 50 entries, stored in `past[]`/`future[]` arrays

## Session/Match Architecture

Sessions contain multiple matches (videos), each with their own rallies:

```
Session
├── Match 1 (Video 1)
│   ├── Rally 1 (match_1_rally_0)
│   ├── Rally 2 (match_1_rally_1)
│   └── ...
├── Match 2 (Video 2)
│   ├── Rally 1 (match_2_rally_0)
│   └── ...
└── Highlights (cross-match, reference rallies by ID)
```

- **activeMatchId**: Currently selected match/video in the editor
- **Switching matches**: Saves current rallies to session, loads new match's rallies
- **Rally IDs**: `{matchId}_rally_{n}` format enables match lookup from rally ID
- **Highlights**: Can include rallies from any match in the session

## API Integration

- `NEXT_PUBLIC_API_URL` for backend
- Rally IDs: frontend uses `{matchId}_rally_{n}`, backend uses UUIDs
- `syncService.markDirty()` schedules state sync

## Export

- `exportServerSide()` triggers Lambda export via API
- Polls for progress, downloads when complete
- FREE = 720p + watermark (browser export), PRO/ELITE = original + camera effects (server export)

## Caveats

- **Rally ID format**: Frontend `{matchId}_rally_{n}` ↔ Backend UUIDs (mapped during sync)
- **Confirmation lock**: After rally confirmation, all rally edits locked for that video
- **FREE tier**: No server sync, localStorage only
- **Camera edits on boundary**: Start/end adjustments cannot exclude first/last keyframe
- **Highlight playback**: Auto-switches videos via `pendingMatchSwitch` in playerStore
- **Upload blob cleanup**: `localVideoUrls` cleared on page refresh, then uses proxy
