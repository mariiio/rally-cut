# RallyCut Web

Rally editor web application for beach volleyball video editing.

## Stack

- **Framework**: Next.js 15 (App Router)
- **UI**: React 19, MUI v7, Emotion
- **State**: Zustand with localStorage persistence
- **Drag & Drop**: @dnd-kit/core
- **Video**: HTML5 video with custom controls

## Getting Started

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Type check
npx tsc --noEmit

# Lint
npm run lint
```

Open [http://localhost:3000](http://localhost:3000) to view the app.

## Environment Variables

Create `.env.local`:

```bash
NEXT_PUBLIC_API_URL=http://localhost:3001  # Backend API URL
```

## Project Structure

```
src/
├── app/                    # Next.js App Router
│   ├── (landing)/          # Public landing page
│   ├── (app)/              # Protected editor routes
│   │   ├── sessions/[id]/  # Multi-video session editor
│   │   └── videos/[id]/    # Single video editor
│   └── theme.ts            # MUI theme + design tokens
├── components/
│   ├── EditorLayout.tsx    # Main editor layout
│   ├── EditorHeader.tsx    # Toolbar with undo/redo/export
│   ├── VideoPlayer.tsx     # Video player with keyboard controls
│   ├── Timeline.tsx        # Rally timeline with drag handles
│   ├── RalliesPanel.tsx    # Rally list sidebar
│   ├── HighlightsPanel.tsx # Highlights management
│   ├── SyncStatus.tsx      # Cloud sync indicator
│   └── UploadProgress.tsx  # Video upload progress
├── stores/
│   ├── editorStore.ts      # Main state (rallies, highlights, history)
│   ├── playerStore.ts      # Video playback state
│   └── uploadStore.ts      # Upload progress state
├── services/
│   ├── api.ts              # REST API client
│   └── syncService.ts      # State sync to backend
├── types/
│   └── rally.ts            # TypeScript types
└── utils/
    ├── fileHandlers.ts     # JSON import/export
    └── videoExport.ts      # FFmpeg video export
```

## Key Features

### Timeline Editor
- Drag rally boundaries to adjust start/end times
- Click to seek, double-click to play rally
- Zoom in/out with scroll wheel
- Keyboard shortcuts (J/K/L for playback, arrow keys for frame stepping)

### Undo/Redo
- Full history with Cmd+Z / Cmd+Shift+Z
- Persisted to localStorage
- Reset to original detection results

### Highlights
- Create highlight reels from selected rallies
- Drag rallies between highlights
- Reorder rallies within highlights
- Color-coded for visual distinction

### Sync
- Auto-sync to backend every 5 seconds after changes
- localStorage for immediate persistence
- Visual indicator for sync status

## Code Patterns

### State Management
```typescript
// Zustand store with history
const useEditorStore = create<EditorState>((set, get) => ({
  rallies: [],
  past: [],      // Undo stack
  future: [],    // Redo stack

  updateRally: (id, updates) => {
    get().pushHistory();  // Save current state first
    set({ rallies: ... });
    syncService.markDirty();  // Schedule backend sync
  },
}));
```

### Video Player
```typescript
// Keyboard shortcuts
useEffect(() => {
  const handleKeyDown = (e: KeyboardEvent) => {
    if (e.key === ' ') togglePlay();
    if (e.key === 'ArrowLeft') seek(currentTime - 5);
    if (e.key === 'ArrowRight') seek(currentTime + 5);
  };
  window.addEventListener('keydown', handleKeyDown);
  return () => window.removeEventListener('keydown', handleKeyDown);
}, []);
```

## Testing

```bash
# Type check
npx tsc --noEmit

# Lint
npm run lint

# Build
npm run build
```
