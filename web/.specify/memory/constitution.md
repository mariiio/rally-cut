<!--
Sync Impact Report
- Version change: 0.0.0 → 1.0.0
- Added sections: All (initial constitution)
- Templates requiring updates: None (initial setup)
- Follow-up TODOs: None
-->

# RallyCut Web Video Segment Editor Constitution

## Core Principles

### I. Library-First Development
MUST prefer existing, well-maintained libraries over custom implementations.
- Video playback: Use react-player (not custom HTML5 video wrapper)
- Timeline editing: Use @xzdarcy/react-timeline-editor (not custom canvas/SVG)
- State management: Use Zustand (not custom context)
- UI components: Use Material UI (not custom styled components)

Rationale: The primary goal is shipping quickly. Custom implementations are only permitted when no suitable library exists or existing libraries have critical bugs.

### II. Browser-Only Architecture
MUST NOT require backend API calls for core functionality in MVP.
- Video loading: Use browser File API and URL.createObjectURL()
- JSON import/export: Handle entirely in browser with FileReader and download links
- State persistence: Use browser localStorage only (no server database)

Rationale: Simplifies deployment, reduces complexity, and allows offline usage. Backend integration is a future enhancement.

### III. Format Compatibility
MUST maintain exact compatibility with RallyCut analysis output JSON format.
- Preserve all fields: version, video metadata, rallies array, stats object
- Use snake_case for JSON properties (matching Python output)
- Recalculate derived fields (duration, stats) on export

Rationale: Users must be able to round-trip JSON files between the Python CLI and web editor without data loss or format conversion.

### IV. TypeScript Strict Mode
MUST use TypeScript with strict mode enabled.
- No `any` types except for external library integration when types unavailable
- All component props explicitly typed
- All store state and actions typed

Rationale: Type safety prevents runtime errors and improves developer experience with autocomplete and refactoring.

### V. Simplicity Over Features
MUST ship minimal viable features before adding enhancements.
- MVP scope: Load video, load JSON, edit segments (trim/add/remove/reorder), export JSON
- NO undo/redo in MVP
- NO thumbnail generation in MVP
- NO keyboard shortcuts in MVP (add as enhancement)

Rationale: Ship working software first, iterate based on actual usage feedback.

## Technology Stack

- **Framework**: Next.js 14+ with App Router
- **UI Library**: Material UI v6 with Emotion
- **Video Player**: react-player
- **Timeline**: @xzdarcy/react-timeline-editor
- **State Management**: Zustand
- **Language**: TypeScript (strict mode)

## Development Workflow

- Start with types and interfaces
- Build stores with CRUD operations
- Create components bottom-up (small components first)
- Test each feature manually before moving to next
- No unit tests in MVP (add as enhancement)

## Governance

This constitution governs all development decisions for the RallyCut Web Video Segment Editor.
- Amendments require explicit user approval
- Deviations from principles must be documented with rationale
- When in doubt, choose the simpler option

**Version**: 1.0.0 | **Ratified**: 2025-12-29 | **Last Amended**: 2025-12-29
