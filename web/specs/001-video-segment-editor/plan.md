# Implementation Plan: Video Segment Editor

**Branch**: `001-video-segment-editor` | **Date**: 2025-12-29 | **Spec**: [spec.md](./spec.md)

## Summary

Build a browser-based video segment editor that allows users to load volleyball videos and RallyCut JSON segment files, review detected rallies, edit segment boundaries (trim/add/remove/reorder), and export updated JSON. Uses react-player for video, @xzdarcy/react-timeline-editor for timeline, Material UI for components, and Zustand for state management.

## Technical Context

**Language/Version**: TypeScript 5.x with strict mode
**Framework**: Next.js 14+ with App Router
**Primary Dependencies**:
- react-player (video playback)
- @xzdarcy/react-timeline-editor (timeline UI)
- @mui/material v6 (UI components)
- zustand (state management)

**Storage**: Browser only (File API, localStorage for preferences)
**Testing**: Manual testing for MVP (no unit tests per constitution)
**Target Platform**: Modern browsers (Chrome, Firefox, Safari, Edge)
**Project Type**: Web frontend (single-page application)
**Performance Goals**: Video loads within 5 seconds, UI interactions < 100ms
**Constraints**: Browser-only, no backend API, must work offline after initial load
**Scale/Scope**: Single user, local files only, ~30-50 segments typical

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Library-First | PASS | Using react-player, react-timeline-editor, MUI, Zustand |
| II. Browser-Only | PASS | No backend API calls, File API for video/JSON |
| III. Format Compatibility | PASS | TypeScript interfaces match RallyCut JSON exactly |
| IV. TypeScript Strict | PASS | tsconfig.json will enable strict mode |
| V. Simplicity Over Features | PASS | MVP scope only - no undo, thumbnails, or keyboard shortcuts |

## Project Structure

### Documentation (this feature)

```
specs/001-video-segment-editor/
├── spec.md              # Feature specification
├── plan.md              # This file
├── research.md          # Library evaluation
├── data-model.md        # TypeScript interfaces
├── quickstart.md        # Developer setup guide
└── tasks.md             # Implementation tasks
```

### Source Code (repository root)

```
web/
├── src/
│   ├── app/
│   │   ├── layout.tsx           # MUI ThemeProvider, CssBaseline
│   │   ├── page.tsx             # Main editor page
│   │   └── globals.css          # Global styles
│   ├── components/
│   │   ├── VideoPlayer.tsx      # react-player wrapper
│   │   ├── PlayerControls.tsx   # Play/pause, seek, time display
│   │   ├── Timeline.tsx         # react-timeline-editor wrapper
│   │   ├── SegmentList.tsx      # Segment list with edit/delete
│   │   ├── SegmentForm.tsx      # Add/edit segment dialog
│   │   ├── FileControls.tsx     # Upload video, load/export JSON
│   │   └── EditorLayout.tsx     # Main grid layout
│   ├── stores/
│   │   ├── editorStore.ts       # Segments, video metadata, CRUD
│   │   └── playerStore.ts       # Playback state, currentTime
│   ├── types/
│   │   └── segment.ts           # TypeScript interfaces
│   └── utils/
│       ├── fileHandlers.ts      # File upload/download utilities
│       └── timeFormat.ts        # MM:SS.ms formatting
├── package.json
├── tsconfig.json
├── next.config.ts
└── tailwind.config.ts
```

**Structure Decision**: Single web frontend project. No backend directory needed for MVP (browser-only per constitution).

## Complexity Tracking

No violations - all choices align with constitution principles.
