# RallyCut Web

Rally editor web application for beach volleyball video editing.

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
│   └── CameraOverlay # Visual preview overlay
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

## Camera System

Instagram-style zoom/pan effects per rally:

- **Aspect ratios**: ORIGINAL (16:9) or VERTICAL (9:16)
- **Keyframes**: timeOffset (0-1), positionX/Y (0-1), zoom (1-3x), easing
- **Handheld motion**: OFF/LIGHT/MEDIUM/HEAVY presets (simulates camera shake)
- **Interpolation**: Keyframes blended with easing (LINEAR, EASE_IN, EASE_OUT, EASE_IN_OUT)

Camera edits stored per aspect ratio (switching preserves both).

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
| Escape | Cancel deletion / deselect |
| Cmd/Ctrl + Z | Undo |
| Cmd/Ctrl + Shift + Z | Redo |
| [ / ] | Collapse left/right panel |
| Cmd/Ctrl + Shift + C | Open camera panel |

## State Persistence

- **localStorage**: Immediate save on every mutation
- **Backend sync**: 5s debounce after `markDirty()`, PREMIUM only
- **Undo/redo**: 50 entries, stored in `past[]`/`future[]` arrays

## API Integration

- `NEXT_PUBLIC_API_URL` for backend
- Rally IDs: frontend uses `{matchId}_rally_{n}`, backend uses UUIDs
- `syncService.markDirty()` schedules state sync

## Export

- `exportServerSide()` triggers Lambda export via API
- Polls for progress, downloads when complete
- FREE = 720p + watermark, PREMIUM = original + camera effects

## Caveats

- **Rally ID format**: Frontend `{matchId}_rally_{n}` ↔ Backend UUIDs (mapped during sync)
- **Confirmation lock**: After rally confirmation, all rally edits locked for that video
- **FREE tier**: No server sync, localStorage only
- **Camera edits on boundary**: Start/end adjustments cannot exclude first/last keyframe
- **Highlight playback**: Auto-switches videos via `pendingMatchSwitch` in playerStore
- **Upload blob cleanup**: `localVideoUrls` cleared on page refresh, then uses proxy
