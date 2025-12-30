import { create } from 'zustand';
import {
  Rally,
  VideoMetadata,
  SegmentFile,
  SegmentStats,
  Highlight,
  HIGHLIGHT_COLORS,
  createRally,
  recalculateRally,
  calculateStats,
} from '@/types/segment';

// History management types
interface HistoryEntry {
  segments: Rally[];
  highlights: Highlight[];
  timestamp: number;
}

interface PersistedHistory {
  videoPath: string | null;
  segments: Rally[];
  highlights: Highlight[];
  past: HistoryEntry[];
  future: HistoryEntry[];
  originalSegments: Rally[];
  originalHighlights: Highlight[];
  savedAt: number;
}

const STORAGE_KEY = 'rallycut_editor_history_v1';
const MAX_HISTORY_SIZE = 50;

// Debounce helper
let saveTimeout: ReturnType<typeof setTimeout> | null = null;
const debouncedSave = (fn: () => void) => {
  if (saveTimeout) clearTimeout(saveTimeout);
  saveTimeout = setTimeout(fn, 500);
};

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

  // History state
  past: HistoryEntry[];
  future: HistoryEntry[];
  originalSegments: Rally[];
  originalHighlights: Highlight[];

  // Highlights state
  highlights: Highlight[];
  selectedHighlightId: string | null;

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

  // History actions
  pushHistory: () => void;
  undo: () => void;
  redo: () => void;
  resetToOriginal: () => void;
  saveToStorage: () => void;
  loadFromStorage: () => boolean;
  clearHistory: () => void;

  // Highlights actions
  createHighlight: (name?: string) => string;
  deleteHighlight: (id: string) => void;
  renameHighlight: (id: string, name: string) => void;
  addSegmentToHighlight: (segmentId: string, highlightId: string) => void;
  removeSegmentFromHighlight: (segmentId: string, highlightId: string) => void;
  selectHighlight: (id: string | null) => void;

  // Computed helpers (called as functions since Zustand doesn't have computed)
  canUndo: () => boolean;
  canRedo: () => boolean;
  hasChangesFromOriginal: () => boolean;
  canCreateHighlight: () => boolean;
  getNextHighlightColor: () => string;
  getHighlightsForSegment: (segmentId: string) => Highlight[];
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

  // History state
  past: [],
  future: [],
  originalSegments: [],
  originalHighlights: [],

  // Highlights state
  highlights: [],
  selectedHighlightId: null,

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
    // Store original data for reset functionality
    const originalSegments = [...json.rallies];
    const originalHighlights = json.highlights ? [...json.highlights] : [];

    // Try to load persisted state for this video
    let segments = json.rallies;
    let highlights = json.highlights || [];
    let past: HistoryEntry[] = [];
    let future: HistoryEntry[] = [];

    if (typeof window !== 'undefined') {
      try {
        const stored = localStorage.getItem(STORAGE_KEY);
        if (stored) {
          const persisted: PersistedHistory = JSON.parse(stored);
          // Only restore if it's for the same video
          if (persisted.videoPath === json.video.path) {
            segments = persisted.segments;
            highlights = persisted.highlights || [];
            past = persisted.past;
            future = persisted.future;
          }
        }
      } catch {
        // localStorage unavailable or corrupted, continue with fresh state
      }
    }

    set({
      segments,
      highlights,
      videoMetadata: json.video,
      originalJson: json,
      originalSegments,
      originalHighlights,
      past,
      future,
      hasUnsavedChanges: past.length > 0,
      selectedSegmentId: null,
      selectedHighlightId: null,
    });
  },

  addSegment: (startTime: number, endTime: number) => {
    const state = get();
    state.pushHistory();

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

    debouncedSave(() => get().saveToStorage());
  },

  updateSegment: (id: string, updates: Partial<Rally>) => {
    const state = get();
    state.pushHistory();

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

    debouncedSave(() => get().saveToStorage());
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
    state.pushHistory();

    // Also remove from all highlights
    const updatedHighlights = state.highlights.map((h) => ({
      ...h,
      segmentIds: h.segmentIds.filter((sid) => sid !== id),
    }));

    set({
      segments: state.segments.filter((s) => s.id !== id),
      highlights: updatedHighlights,
      selectedSegmentId:
        state.selectedSegmentId === id ? null : state.selectedSegmentId,
      hasUnsavedChanges: true,
    });

    debouncedSave(() => get().saveToStorage());
  },

  reorderSegments: (fromIndex: number, toIndex: number) => {
    const state = get();
    state.pushHistory();

    const segments = [...state.segments];
    const [removed] = segments.splice(fromIndex, 1);
    segments.splice(toIndex, 0, removed);
    set({
      segments,
      hasUnsavedChanges: true,
    });

    debouncedSave(() => get().saveToStorage());
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
      highlights: state.highlights.length > 0 ? state.highlights : undefined,
    };

    set({ hasUnsavedChanges: false });
    return result;
  },

  clearAll: () => {
    const state = get();
    if (state.videoUrl) {
      URL.revokeObjectURL(state.videoUrl);
    }

    // Clear localStorage
    if (typeof window !== 'undefined') {
      try {
        localStorage.removeItem(STORAGE_KEY);
      } catch {
        // Ignore localStorage errors
      }
    }

    set({
      videoFile: null,
      videoUrl: null,
      videoMetadata: null,
      segments: [],
      highlights: [],
      selectedSegmentId: null,
      selectedHighlightId: null,
      hasUnsavedChanges: false,
      originalJson: null,
      past: [],
      future: [],
      originalSegments: [],
      originalHighlights: [],
    });
  },

  // History management actions
  pushHistory: () => {
    const state = get();
    const entry: HistoryEntry = {
      segments: [...state.segments],
      highlights: [...state.highlights],
      timestamp: Date.now(),
    };

    // Limit history size
    const past = [...state.past, entry].slice(-MAX_HISTORY_SIZE);

    set({
      past,
      future: [], // Clear redo stack on new action
    });
  },

  undo: () => {
    const state = get();
    if (state.past.length === 0) return;

    const past = [...state.past];
    const entry = past.pop()!;

    const futureEntry: HistoryEntry = {
      segments: [...state.segments],
      highlights: [...state.highlights],
      timestamp: Date.now(),
    };

    set({
      segments: entry.segments,
      highlights: entry.highlights,
      past,
      future: [...state.future, futureEntry],
      hasUnsavedChanges: past.length > 0 || state.future.length > 0,
    });

    debouncedSave(() => get().saveToStorage());
  },

  redo: () => {
    const state = get();
    if (state.future.length === 0) return;

    const future = [...state.future];
    const entry = future.pop()!;

    const pastEntry: HistoryEntry = {
      segments: [...state.segments],
      highlights: [...state.highlights],
      timestamp: Date.now(),
    };

    set({
      segments: entry.segments,
      highlights: entry.highlights,
      past: [...state.past, pastEntry],
      future,
      hasUnsavedChanges: true,
    });

    debouncedSave(() => get().saveToStorage());
  },

  resetToOriginal: () => {
    const state = get();
    if (state.originalSegments.length === 0) return;

    // Push current state to history before resetting
    state.pushHistory();

    set({
      segments: [...state.originalSegments],
      highlights: [...state.originalHighlights],
      future: [], // Clear redo stack
      hasUnsavedChanges: false,
    });

    // Clear localStorage since we're back to original
    if (typeof window !== 'undefined') {
      try {
        localStorage.removeItem(STORAGE_KEY);
      } catch {
        // Ignore localStorage errors
      }
    }
  },

  saveToStorage: () => {
    const state = get();
    if (!state.videoMetadata) return;

    if (typeof window !== 'undefined') {
      try {
        const data: PersistedHistory = {
          videoPath: state.videoMetadata.path,
          segments: state.segments,
          highlights: state.highlights,
          past: state.past,
          future: state.future,
          originalSegments: state.originalSegments,
          originalHighlights: state.originalHighlights,
          savedAt: Date.now(),
        };
        localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
      } catch {
        // localStorage unavailable or quota exceeded
        // Could trim history here if needed
      }
    }
  },

  loadFromStorage: () => {
    if (typeof window === 'undefined') return false;

    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (!stored) return false;

      const persisted: PersistedHistory = JSON.parse(stored);
      set({
        segments: persisted.segments,
        highlights: persisted.highlights || [],
        past: persisted.past,
        future: persisted.future,
        originalSegments: persisted.originalSegments,
        originalHighlights: persisted.originalHighlights || [],
        hasUnsavedChanges: persisted.past.length > 0,
      });
      return true;
    } catch {
      return false;
    }
  },

  clearHistory: () => {
    set({
      past: [],
      future: [],
    });

    if (typeof window !== 'undefined') {
      try {
        localStorage.removeItem(STORAGE_KEY);
      } catch {
        // Ignore localStorage errors
      }
    }
  },

  // Highlights actions
  createHighlight: (name?: string) => {
    const state = get();
    state.pushHistory();

    // Generate unique ID
    const existingIds = state.highlights.map((h) => h.id);
    let counter = state.highlights.length + 1;
    let newId = `highlight_${counter}`;
    while (existingIds.includes(newId)) {
      counter++;
      newId = `highlight_${counter}`;
    }

    // Get next available color
    const color = state.getNextHighlightColor();

    const newHighlight: Highlight = {
      id: newId,
      name: name || `Highlight ${counter}`,
      color,
      segmentIds: [],
      createdAt: Date.now(),
    };

    set({
      highlights: [...state.highlights, newHighlight],
      hasUnsavedChanges: true,
    });

    debouncedSave(() => get().saveToStorage());
    return newId;
  },

  deleteHighlight: (id: string) => {
    const state = get();
    state.pushHistory();

    set({
      highlights: state.highlights.filter((h) => h.id !== id),
      selectedHighlightId:
        state.selectedHighlightId === id ? null : state.selectedHighlightId,
      hasUnsavedChanges: true,
    });

    debouncedSave(() => get().saveToStorage());
  },

  renameHighlight: (id: string, name: string) => {
    const state = get();
    state.pushHistory();

    set({
      highlights: state.highlights.map((h) =>
        h.id === id ? { ...h, name } : h
      ),
      hasUnsavedChanges: true,
    });

    debouncedSave(() => get().saveToStorage());
  },

  addSegmentToHighlight: (segmentId: string, highlightId: string) => {
    const state = get();
    const highlight = state.highlights.find((h) => h.id === highlightId);
    if (!highlight) return;

    // Don't add if already in highlight
    if (highlight.segmentIds.includes(segmentId)) return;

    state.pushHistory();

    set({
      highlights: state.highlights.map((h) =>
        h.id === highlightId
          ? { ...h, segmentIds: [...h.segmentIds, segmentId] }
          : h
      ),
      hasUnsavedChanges: true,
    });

    debouncedSave(() => get().saveToStorage());
  },

  removeSegmentFromHighlight: (segmentId: string, highlightId: string) => {
    const state = get();
    state.pushHistory();

    set({
      highlights: state.highlights.map((h) =>
        h.id === highlightId
          ? { ...h, segmentIds: h.segmentIds.filter((id) => id !== segmentId) }
          : h
      ),
      hasUnsavedChanges: true,
    });

    debouncedSave(() => get().saveToStorage());
  },

  selectHighlight: (id: string | null) => {
    set({ selectedHighlightId: id });
  },

  // Computed helpers
  canUndo: () => get().past.length > 0,
  canRedo: () => get().future.length > 0,
  hasChangesFromOriginal: () => {
    const state = get();
    if (state.originalSegments.length === 0) return false;
    if (state.segments.length !== state.originalSegments.length) return true;
    return JSON.stringify(state.segments) !== JSON.stringify(state.originalSegments);
  },

  canCreateHighlight: () => {
    const state = get();
    // Can create if no highlights OR all existing highlights have at least one segment
    if (state.highlights.length === 0) return true;
    return state.highlights.every((h) => h.segmentIds.length > 0);
  },

  getNextHighlightColor: () => {
    const state = get();
    const usedColors = state.highlights.map((h) => h.color);
    // Find first unused color
    const available = HIGHLIGHT_COLORS.find((c) => !usedColors.includes(c));
    // If all used, cycle through
    return available || HIGHLIGHT_COLORS[state.highlights.length % HIGHLIGHT_COLORS.length];
  },

  getHighlightsForSegment: (segmentId: string) => {
    const state = get();
    return state.highlights.filter((h) => h.segmentIds.includes(segmentId));
  },
}));
