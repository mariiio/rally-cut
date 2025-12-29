# Quickstart: Video Segment Editor

## Prerequisites

- Node.js 18+ (LTS recommended)
- pnpm (recommended) or npm

## Setup

```bash
# Navigate to web directory
cd /Users/mario/Personal/Projects/RallyCut/web

# Install dependencies
pnpm install

# Start development server
pnpm dev
```

Open http://localhost:3000

## Project Commands

```bash
pnpm dev      # Start development server
pnpm build    # Build for production
pnpm start    # Start production server
pnpm lint     # Run ESLint
```

## Usage

1. **Load Video**: Click "Upload Video" and select a .mp4, .mov, or .webm file
2. **Load Segments**: Click "Load JSON" and select a RallyCut segments.json file
3. **Review**: Click segments in the list to jump to that point in the video
4. **Edit**:
   - Drag segment edges on timeline to trim
   - Click segment, then "Edit" to modify times directly
   - Click "Add Segment" to create new segments
   - Click trash icon to delete segments
5. **Export**: Click "Export JSON" to download the edited segments file

## Sample Data

A sample segments.json file is available at:
```
/Users/mario/Desktop/segments.json
```

## Key Files

| File | Purpose |
|------|---------|
| `src/app/page.tsx` | Main editor page |
| `src/stores/editorStore.ts` | Segment state and CRUD operations |
| `src/stores/playerStore.ts` | Video playback state |
| `src/components/VideoPlayer.tsx` | react-player wrapper |
| `src/components/Timeline.tsx` | Timeline editor |
| `src/types/segment.ts` | TypeScript interfaces |

## Troubleshooting

**Video won't play**: Ensure the video format is supported (.mp4, .mov, .webm)

**JSON won't load**: Verify the JSON matches RallyCut format with `version: "1.0"`

**Timeline not syncing**: Check browser console for errors; try refreshing
