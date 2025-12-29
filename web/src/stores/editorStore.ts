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
  clearVideo: () => void;
  loadSegmentsFromJson: (json: SegmentFile) => void;
  addSegment: (startTime: number, endTime: number) => void;
  updateSegment: (id: string, updates: Partial<Rally>) => void;
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
