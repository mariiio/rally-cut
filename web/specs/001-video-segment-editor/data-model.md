# Data Model: Video Segment Editor

**Date**: 2025-12-29
**Branch**: 001-video-segment-editor

## Core Types

### VideoMetadata

Metadata about the loaded video file, matching RallyCut JSON `video` field.

```typescript
interface VideoMetadata {
  path: string;           // Original file path (preserved from input JSON)
  duration: number;       // Total duration in seconds
  fps: number;            // Frames per second
  width: number;          // Video width in pixels
  height: number;         // Video height in pixels
  frame_count: number;    // Total frame count
}
```

### Rally (Segment)

A single segment representing a rally in the video.

```typescript
interface Rally {
  id: string;             // Unique identifier, e.g., "rally_1"
  start_time: number;     // Start time in seconds
  end_time: number;       // End time in seconds
  start_frame: number;    // Start frame number
  end_frame: number;      // End frame number
  duration: number;       // Duration in seconds (derived: end_time - start_time)
  type: "rally";          // Segment type (always "rally" for now)
  thumbnail_time: number; // Time for thumbnail preview (midpoint of segment)
}
```

**Derived Field Calculations**:
- `duration = end_time - start_time`
- `start_frame = Math.round(start_time * fps)`
- `end_frame = Math.round(end_time * fps)`
- `thumbnail_time = start_time + duration / 2`

### SegmentStats

Aggregate statistics about all segments.

```typescript
interface SegmentStats {
  original_duration: number;    // Total video duration
  kept_duration: number;        // Sum of all segment durations
  removed_duration: number;     // original_duration - kept_duration
  kept_percentage: number;      // (kept_duration / original_duration) * 100
  removed_percentage: number;   // (removed_duration / original_duration) * 100
  segment_count: number;        // Number of segments
}
```

### SegmentFile

Complete JSON file structure matching RallyCut output.

```typescript
interface SegmentFile {
  version: "1.0";
  video: VideoMetadata;
  rallies: Rally[];
  stats: SegmentStats;
}
```

## Store State

### EditorStore

Main application state for segment editing.

```typescript
interface EditorState {
  // Video state
  videoFile: File | null;
  videoUrl: string | null;          // Blob URL from createObjectURL
  videoMetadata: VideoMetadata | null;

  // Segment state
  segments: Rally[];
  selectedSegmentId: string | null;
  hasUnsavedChanges: boolean;

  // Original JSON (for preserving non-edited fields)
  originalJson: SegmentFile | null;

  // Actions
  setVideoFile: (file: File) => void;
  loadSegmentsFromJson: (json: SegmentFile) => void;
  addSegment: (segment: Omit<Rally, 'id'>) => void;
  updateSegment: (id: string, updates: Partial<Rally>) => void;
  removeSegment: (id: string) => void;
  reorderSegments: (fromIndex: number, toIndex: number) => void;
  selectSegment: (id: string | null) => void;
  exportToJson: () => SegmentFile;
  clearAll: () => void;
}
```

### PlayerStore

Video player synchronization state.

```typescript
interface PlayerState {
  isPlaying: boolean;
  currentTime: number;
  duration: number;
  isReady: boolean;

  // Actions
  play: () => void;
  pause: () => void;
  togglePlay: () => void;
  seek: (time: number) => void;
  setCurrentTime: (time: number) => void;
  setDuration: (duration: number) => void;
  setReady: (ready: boolean) => void;
}
```

## Validation Rules

### Segment Validation

1. `start_time >= 0`
2. `end_time > start_time`
3. `end_time <= video.duration` (warning only, not blocking)
4. `id` must be unique across all segments

### JSON Import Validation

1. `version` must be "1.0"
2. `video` object must exist with required fields
3. `rallies` must be an array
4. Each rally must have required fields: `id`, `start_time`, `end_time`

## State Transitions

### Segment Lifecycle

```
[No Segments] --loadJson--> [Segments Loaded]
[Segments Loaded] --edit/add/remove--> [Has Unsaved Changes]
[Has Unsaved Changes] --export--> [Saved] (hasUnsavedChanges = false)
[Has Unsaved Changes] --navigate away--> [Show Warning Dialog]
```

### Video Lifecycle

```
[No Video] --upload--> [Video Loading]
[Video Loading] --onReady--> [Video Ready]
[Video Ready] --replace--> [Revoke Old URL] --> [Video Loading]
```
