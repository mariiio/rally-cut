import { create } from 'zustand';
import {
  Rally,
  VideoMetadata,
  SegmentFile,
  SegmentStats,
  createRally,
  recalculateRally,
  calculateStats,
} from '@/types/segment';

interface EditorState {
  // Video state
  videoFile: File | null;
  videoUrl: string | null;
  videoMetadata: VideoMetadata | null;

  // Segment state
  segments: Rally[];
  selectedSegmentId: string | null;
  hasUnsavedChanges: boolean;

  // Original JSON (for preserving non-edited fields)
  originalJson: SegmentFile | null;

  // Actions
  setVideoFile: (file: File) => void;
  setVideoUrl: (url: string) => void;
  clearVideo: () => void;
  loadSegmentsFromJson: (json: SegmentFile) => void;
  addSegment: (startTime: number, endTime: number) => void;
  updateSegment: (id: string, updates: Partial<Rally>) => void;
  adjustSegmentStart: (id: string, delta: number) => boolean;
  adjustSegmentEnd: (id: string, delta: number) => boolean;
  createSegmentAtTime: (time: number, duration?: number) => void;
  removeSegment: (id: string) => void;
  reorderSegments: (fromIndex: number, toIndex: number) => void;
  selectSegment: (id: string | null) => void;
  exportToJson: () => SegmentFile | null;
  clearAll: () => void;
}

export const useEditorStore = create<EditorState>((set, get) => ({
  // Initial state
  videoFile: null,
  videoUrl: null,
  videoMetadata: null,
  segments: [],
  selectedSegmentId: null,
  hasUnsavedChanges: false,
  originalJson: null,

  setVideoFile: (file: File) => {
    const state = get();
    // Revoke old URL to free memory
    if (state.videoUrl) {
      URL.revokeObjectURL(state.videoUrl);
    }
    const url = URL.createObjectURL(file);
    set({
      videoFile: file,
      videoUrl: url,
    });
  },

  setVideoUrl: (url: string) => {
    const state = get();
    // Revoke old blob URL if exists
    if (state.videoUrl && state.videoUrl.startsWith('blob:')) {
      URL.revokeObjectURL(state.videoUrl);
    }
    set({
      videoFile: null,
      videoUrl: url,
    });
  },

  clearVideo: () => {
    const state = get();
    if (state.videoUrl) {
      URL.revokeObjectURL(state.videoUrl);
    }
    set({
      videoFile: null,
      videoUrl: null,
      videoMetadata: null,
    });
  },

  loadSegmentsFromJson: (json: SegmentFile) => {
    set({
      segments: json.rallies,
      videoMetadata: json.video,
      originalJson: json,
      hasUnsavedChanges: false,
      selectedSegmentId: null,
    });
  },

  addSegment: (startTime: number, endTime: number) => {
    const state = get();
    const fps = state.videoMetadata?.fps ?? 30;

    // Generate unique ID
    const existingIds = state.segments.map((s) => s.id);
    let counter = state.segments.length + 1;
    let newId = `rally_${counter}`;
    while (existingIds.includes(newId)) {
      counter++;
      newId = `rally_${counter}`;
    }

    const newRally = createRally(newId, startTime, endTime, fps);

    set({
      segments: [...state.segments, newRally],
      hasUnsavedChanges: true,
    });
  },

  updateSegment: (id: string, updates: Partial<Rally>) => {
    const state = get();
    const fps = state.videoMetadata?.fps ?? 30;

    set({
      segments: state.segments.map((segment) => {
        if (segment.id !== id) return segment;
        const updated = { ...segment, ...updates };
        // Recalculate derived fields if times changed
        if ('start_time' in updates || 'end_time' in updates) {
          return recalculateRally(updated, fps);
        }
        return updated;
      }),
      hasUnsavedChanges: true,
    });
  },

  adjustSegmentStart: (id: string, delta: number) => {
    const state = get();
    const segment = state.segments.find((s) => s.id === id);
    if (!segment) return false;

    const newStart = segment.start_time + delta;

    // Boundary checks
    if (newStart < 0) return false;
    if (newStart >= segment.end_time - 0.5) return false; // Min 0.5s duration

    // Overlap check (only for expanding, i.e., negative delta)
    if (delta < 0) {
      const prevSegment = state.segments
        .filter((s) => s.end_time <= segment.start_time)
        .sort((a, b) => b.end_time - a.end_time)[0];
      if (prevSegment && newStart < prevSegment.end_time) return false;
    }

    state.updateSegment(id, { start_time: newStart });
    return true;
  },

  adjustSegmentEnd: (id: string, delta: number) => {
    const state = get();
    const segment = state.segments.find((s) => s.id === id);
    if (!segment) return false;

    const newEnd = segment.end_time + delta;
    const videoDuration = state.videoMetadata?.duration ?? Infinity;

    // Boundary checks
    if (newEnd > videoDuration) return false;
    if (newEnd <= segment.start_time + 0.5) return false; // Min 0.5s duration

    // Overlap check (only for expanding, i.e., positive delta)
    if (delta > 0) {
      const nextSegment = state.segments
        .filter((s) => s.start_time >= segment.end_time)
        .sort((a, b) => a.start_time - b.start_time)[0];
      if (nextSegment && newEnd > nextSegment.start_time) return false;
    }

    state.updateSegment(id, { end_time: newEnd });
    return true;
  },

  createSegmentAtTime: (time: number, duration: number = 5) => {
    const state = get();
    const videoDuration = state.videoMetadata?.duration ?? Infinity;

    // Find next segment to avoid overlap
    const nextSegment = state.segments
      .filter((s) => s.start_time > time)
      .sort((a, b) => a.start_time - b.start_time)[0];

    // Find previous segment to avoid overlap
    const prevSegment = state.segments
      .filter((s) => s.end_time <= time)
      .sort((a, b) => b.end_time - a.end_time)[0];

    // Check if we're inside an existing segment
    const insideSegment = state.segments.find(
      (s) => time >= s.start_time && time <= s.end_time
    );
    if (insideSegment) return; // Can't create inside existing segment

    // Calculate start and end with constraints
    let startTime = time;
    let endTime = time + duration;

    // Adjust if overlapping with next segment
    if (nextSegment && endTime > nextSegment.start_time) {
      endTime = nextSegment.start_time;
    }

    // Ensure we don't exceed video duration
    if (endTime > videoDuration) {
      endTime = videoDuration;
    }

    // Ensure minimum duration
    if (endTime - startTime < 0.5) return;

    state.addSegment(startTime, endTime);

    // Select the new segment
    const newSegments = get().segments;
    const newSegment = newSegments.find(
      (s) => s.start_time === startTime && s.end_time === endTime
    );
    if (newSegment) {
      set({ selectedSegmentId: newSegment.id });
    }
  },

  removeSegment: (id: string) => {
    const state = get();
    set({
      segments: state.segments.filter((s) => s.id !== id),
      selectedSegmentId:
        state.selectedSegmentId === id ? null : state.selectedSegmentId,
      hasUnsavedChanges: true,
    });
  },

  reorderSegments: (fromIndex: number, toIndex: number) => {
    const state = get();
    const segments = [...state.segments];
    const [removed] = segments.splice(fromIndex, 1);
    segments.splice(toIndex, 0, removed);
    set({
      segments,
      hasUnsavedChanges: true,
    });
  },

  selectSegment: (id: string | null) => {
    set({ selectedSegmentId: id });
  },

  exportToJson: () => {
    const state = get();
    if (!state.videoMetadata) return null;

    const fps = state.videoMetadata.fps;
    // Recalculate all segments before export
    const recalculatedSegments = state.segments.map((s) =>
      recalculateRally(s, fps)
    );

    const stats = calculateStats(
      recalculatedSegments,
      state.videoMetadata.duration
    );

    const result: SegmentFile = {
      version: '1.0',
      video: state.videoMetadata,
      rallies: recalculatedSegments,
      stats,
    };

    set({ hasUnsavedChanges: false });
    return result;
  },

  clearAll: () => {
    const state = get();
    if (state.videoUrl) {
      URL.revokeObjectURL(state.videoUrl);
    }
    set({
      videoFile: null,
      videoUrl: null,
      videoMetadata: null,
      segments: [],
      selectedSegmentId: null,
      hasUnsavedChanges: false,
      originalJson: null,
    });
  },
}));
