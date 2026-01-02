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
├── app/              # Next.js pages (page.tsx, editor/[id]/)
├── components/       # React components
│   ├── EditorLayout  # Main 3-panel layout
│   ├── VideoPlayer   # HTML5 video with controls
│   ├── Timeline      # Rally timeline with drag handles
│   ├── RalliesPanel  # Rally list sidebar
│   └── HighlightsPanel # Highlight management
├── stores/           # Zustand stores
│   ├── editorStore   # Rallies, highlights, undo/redo
│   └── playerStore   # Video playback state
├── services/
│   ├── api.ts        # REST client
│   └── syncService   # Backend sync (debounced)
├── utils/
│   └── videoExport.ts # Server-side export via API
└── types/rally.ts    # TypeScript types
```

## Key Patterns

### State with History
- `editorStore` maintains `past[]` and `future[]` arrays for undo/redo
- Every mutation calls `pushHistory()` first, then `syncService.markDirty()`
- State persisted to localStorage immediately, synced to backend after 5s debounce

### Timeline Drag
- Uses @dnd-kit for drag handles on rally boundaries
- `adjustRallyStart/End` methods validate against overlap and video duration
- Timeline zoom controlled by wheel events

### Video Playback
- `playerStore` separate from editor for performance
- Keyboard shortcuts: Space (play/pause), J/K/L (speed), arrows (seek)
- Highlight playback switches videos automatically

## Code Style

- TypeScript strict mode
- ESLint with Next.js config
- Functional components with hooks
- Memoization for expensive computations (useMemo, useCallback)

## API Integration

- `NEXT_PUBLIC_API_URL` environment variable for backend
- `fetchSession()` loads session with videos, rallies, highlights
- `syncService.markDirty()` schedules state sync to backend
- Rally IDs: frontend uses `{videoId}_rally_{n}`, backend uses UUIDs

### Video Export
- `exportServerSide()` in `utils/videoExport.ts` triggers server-side export
- Creates export job via API, polls for progress, downloads when complete
- **FREE tier**: 720p + watermark, **PREMIUM tier**: original quality
- Falls back to local FFmpeg.wasm if server export unavailable
