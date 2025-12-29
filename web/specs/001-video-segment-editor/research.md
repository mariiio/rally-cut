# Research: Video Segment Editor

**Date**: 2025-12-29
**Branch**: 001-video-segment-editor

## Video Player Library

**Decision**: react-player

**Rationale**:
- 2M+ weekly npm downloads, actively maintained
- Supports local file URLs via `URL.createObjectURL()`
- Provides `seekTo(seconds)` method for programmatic seeking
- Callbacks: `onProgress`, `onDuration`, `onReady` for sync with timeline
- Lightweight compared to Video.js

**Alternatives Considered**:
- **Video.js**: More features but heavier, known seeking issues
- **Plyr**: Good but less React-specific integration
- **Native HTML5 video**: Would require building all controls manually

## Timeline Library

**Decision**: @xzdarcy/react-timeline-editor

**Rationale**:
- Purpose-built for video timeline editing use case
- Supports drag-to-resize segments via `onChange` callback
- Clean API: `editorData` (rows with actions), `effects` (styling)
- Has working CodeSandbox examples
- Small footprint, integrates quickly

**Alternatives Considered**:
- **video-editing-timeline-react**: Newer, less documentation
- **Remotion**: Designed for video generation, overkill for segment editing
- **Custom canvas/SVG**: Would take too long (violates constitution principle I)

## State Management

**Decision**: Zustand

**Rationale**:
- Minimal boilerplate compared to Redux
- No providers needed, works anywhere in component tree
- TypeScript support out of the box
- Perfect for small-to-medium apps

**Alternatives Considered**:
- **Redux Toolkit**: More boilerplate than needed for this scope
- **React Context**: Fine for simple cases but less ergonomic for complex state
- **Jotai**: Good alternative but Zustand more widely adopted

## UI Components

**Decision**: Material UI v6 (per user requirement)

**Rationale**:
- User explicitly requested Material UI
- Comprehensive component library
- Good TypeScript support
- Theming support for future customization

## File Handling

**Decision**: Browser File API + URL.createObjectURL()

**Rationale**:
- Standard browser API, no dependencies needed
- `URL.createObjectURL()` creates blob URL that react-player can load
- FileReader API for JSON parsing
- Download via anchor tag with `download` attribute

**Implementation Notes**:
- Video files are not read into memory; blob URL streams from disk
- JSON files are small enough to parse entirely
- Use `URL.revokeObjectURL()` when video is replaced to free memory
