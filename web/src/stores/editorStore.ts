import { create } from 'zustand';
import {
  Rally,
  VideoMetadata,
  RallyFile,
  RallyStats,
  Highlight,
  HIGHLIGHT_COLORS,
  createRally,
  recalculateRally,
  calculateStats,
} from '@/types/rally';

// History management types
interface HistoryEntry {
  rallies: Rally[];
  highlights: Highlight[];
  timestamp: number;
}

interface PersistedHistory {
  videoPath: string | null;
  rallies: Rally[];
  highlights: Highlight[];
  past: HistoryEntry[];
  future: HistoryEntry[];
  originalRallies: Rally[];
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

  // Rally state
  rallies: Rally[];
  selectedRallyId: string | null;
  hasUnsavedChanges: boolean;

  // Original JSON (for preserving non-edited fields)
  originalJson: RallyFile | null;

  // History state
  past: HistoryEntry[];
  future: HistoryEntry[];
  originalRallies: Rally[];
  originalHighlights: Highlight[];

  // Highlights state
  highlights: Highlight[];
  selectedHighlightId: string | null;

  // Actions
  setVideoFile: (file: File) => void;
  setVideoUrl: (url: string) => void;
  clearVideo: () => void;
  loadRalliesFromJson: (json: RallyFile) => void;
  addRally: (startTime: number, endTime: number) => void;
  updateRally: (id: string, updates: Partial<Rally>) => void;
  adjustRallyStart: (id: string, delta: number) => boolean;
  adjustRallyEnd: (id: string, delta: number) => boolean;
  createRallyAtTime: (time: number, duration?: number) => void;
  removeRally: (id: string) => void;
  reorderRallies: (fromIndex: number, toIndex: number) => void;
  selectRally: (id: string | null) => void;
  exportToJson: () => RallyFile | null;
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
  addRallyToHighlight: (rallyId: string, highlightId: string) => void;
  removeRallyFromHighlight: (rallyId: string, highlightId: string) => void;
  selectHighlight: (id: string | null) => void;

  // Computed helpers (called as functions since Zustand doesn't have computed)
  canUndo: () => boolean;
  canRedo: () => boolean;
  hasChangesFromOriginal: () => boolean;
  canCreateHighlight: () => boolean;
  getNextHighlightColor: () => string;
  getHighlightsForRally: (rallyId: string) => Highlight[];
}

export const useEditorStore = create<EditorState>((set, get) => ({
  // Initial state
  videoFile: null,
  videoUrl: null,
  videoMetadata: null,
  rallies: [],
  selectedRallyId: null,
  hasUnsavedChanges: false,
  originalJson: null,

  // History state
  past: [],
  future: [],
  originalRallies: [],
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

  loadRalliesFromJson: (json: RallyFile) => {
    // Store original data for reset functionality
    const originalRallies = [...json.rallies];
    const originalHighlights = json.highlights ? [...json.highlights] : [];

    // Try to load persisted state for this video
    let rallies = json.rallies;
    let highlights = json.highlights || [];
    let past: HistoryEntry[] = [];
    let future: HistoryEntry[] = [];

    if (typeof window !== 'undefined') {
      try {
        const stored = localStorage.getItem(STORAGE_KEY);
        if (stored) {
          const persisted: PersistedHistory = JSON.parse(stored);
          // Only restore if it's for the same video
          if (persisted.videoPath === json.video.path && persisted.rallies) {
            rallies = persisted.rallies;
            highlights = persisted.highlights || [];
            past = persisted.past || [];
            future = persisted.future || [];
          }
        }
      } catch {
        // localStorage unavailable or corrupted, continue with fresh state
      }
    }

    set({
      rallies,
      highlights,
      videoMetadata: json.video,
      originalJson: json,
      originalRallies,
      originalHighlights,
      past,
      future,
      hasUnsavedChanges: past.length > 0,
      selectedRallyId: null,
      selectedHighlightId: null,
    });
  },

  addRally: (startTime: number, endTime: number) => {
    const state = get();
    state.pushHistory();

    const fps = state.videoMetadata?.fps ?? 30;

    // Generate unique ID
    const existingIds = state.rallies.map((s) => s.id);
    let counter = state.rallies.length + 1;
    let newId = `rally_${counter}`;
    while (existingIds.includes(newId)) {
      counter++;
      newId = `rally_${counter}`;
    }

    const newRally = createRally(newId, startTime, endTime, fps);

    set({
      rallies: [...state.rallies, newRally],
      hasUnsavedChanges: true,
    });

    debouncedSave(() => get().saveToStorage());
  },

  updateRally: (id: string, updates: Partial<Rally>) => {
    const state = get();
    state.pushHistory();

    const fps = state.videoMetadata?.fps ?? 30;

    set({
      rallies: state.rallies.map((rally) => {
        if (rally.id !== id) return rally;
        const updated = { ...rally, ...updates };
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

  adjustRallyStart: (id: string, delta: number) => {
    const state = get();
    const rally = state.rallies.find((s) => s.id === id);
    if (!rally) return false;

    const newStart = rally.start_time + delta;

    // Boundary checks
    if (newStart < 0) return false;
    if (newStart >= rally.end_time - 0.5) return false; // Min 0.5s duration

    // Overlap check (only for expanding, i.e., negative delta)
    if (delta < 0) {
      const prevRally = state.rallies
        .filter((s) => s.end_time <= rally.start_time)
        .sort((a, b) => b.end_time - a.end_time)[0];
      if (prevRally && newStart < prevRally.end_time) return false;
    }

    state.updateRally(id, { start_time: newStart });
    return true;
  },

  adjustRallyEnd: (id: string, delta: number) => {
    const state = get();
    const rally = state.rallies.find((s) => s.id === id);
    if (!rally) return false;

    const newEnd = rally.end_time + delta;
    const videoDuration = state.videoMetadata?.duration ?? Infinity;

    // Boundary checks
    if (newEnd > videoDuration) return false;
    if (newEnd <= rally.start_time + 0.5) return false; // Min 0.5s duration

    // Overlap check (only for expanding, i.e., positive delta)
    if (delta > 0) {
      const nextRally = state.rallies
        .filter((s) => s.start_time >= rally.end_time)
        .sort((a, b) => a.start_time - b.start_time)[0];
      if (nextRally && newEnd > nextRally.start_time) return false;
    }

    state.updateRally(id, { end_time: newEnd });
    return true;
  },

  createRallyAtTime: (time: number, duration: number = 5) => {
    const state = get();
    const videoDuration = state.videoMetadata?.duration ?? Infinity;

    // Find next rally to avoid overlap
    const nextRally = state.rallies
      .filter((s) => s.start_time > time)
      .sort((a, b) => a.start_time - b.start_time)[0];

    // Find previous rally to avoid overlap
    const prevRally = state.rallies
      .filter((s) => s.end_time <= time)
      .sort((a, b) => b.end_time - a.end_time)[0];

    // Check if we're inside an existing rally
    const insideRally = state.rallies.find(
      (s) => time >= s.start_time && time <= s.end_time
    );
    if (insideRally) return; // Can't create inside existing rally

    // Calculate start and end with constraints
    let startTime = time;
    let endTime = time + duration;

    // Adjust if overlapping with next rally
    if (nextRally && endTime > nextRally.start_time) {
      endTime = nextRally.start_time;
    }

    // Ensure we don't exceed video duration
    if (endTime > videoDuration) {
      endTime = videoDuration;
    }

    // Ensure minimum duration
    if (endTime - startTime < 0.5) return;

    state.addRally(startTime, endTime);

    // Select the new rally
    const newRallies = get().rallies;
    const newRally = newRallies.find(
      (s) => s.start_time === startTime && s.end_time === endTime
    );
    if (newRally) {
      set({ selectedRallyId: newRally.id });
    }
  },

  removeRally: (id: string) => {
    const state = get();
    state.pushHistory();

    // Also remove from all highlights
    const updatedHighlights = state.highlights.map((h) => ({
      ...h,
      rallyIds: h.rallyIds.filter((rid) => rid !== id),
    }));

    set({
      rallies: state.rallies.filter((s) => s.id !== id),
      highlights: updatedHighlights,
      selectedRallyId:
        state.selectedRallyId === id ? null : state.selectedRallyId,
      hasUnsavedChanges: true,
    });

    debouncedSave(() => get().saveToStorage());
  },

  reorderRallies: (fromIndex: number, toIndex: number) => {
    const state = get();
    state.pushHistory();

    const rallies = [...state.rallies];
    const [removed] = rallies.splice(fromIndex, 1);
    rallies.splice(toIndex, 0, removed);
    set({
      rallies,
      hasUnsavedChanges: true,
    });

    debouncedSave(() => get().saveToStorage());
  },

  selectRally: (id: string | null) => {
    set({ selectedRallyId: id });
  },

  exportToJson: () => {
    const state = get();
    if (!state.videoMetadata) return null;

    const fps = state.videoMetadata.fps;
    // Recalculate all rallies before export
    const recalculatedRallies = state.rallies.map((s) =>
      recalculateRally(s, fps)
    );

    const stats = calculateStats(
      recalculatedRallies,
      state.videoMetadata.duration
    );

    const result: RallyFile = {
      version: '1.0',
      video: state.videoMetadata,
      rallies: recalculatedRallies,
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
      rallies: [],
      highlights: [],
      selectedRallyId: null,
      selectedHighlightId: null,
      hasUnsavedChanges: false,
      originalJson: null,
      past: [],
      future: [],
      originalRallies: [],
      originalHighlights: [],
    });
  },

  // History management actions
  pushHistory: () => {
    const state = get();
    const entry: HistoryEntry = {
      rallies: [...state.rallies],
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
      rallies: [...state.rallies],
      highlights: [...state.highlights],
      timestamp: Date.now(),
    };

    set({
      rallies: entry.rallies,
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
      rallies: [...state.rallies],
      highlights: [...state.highlights],
      timestamp: Date.now(),
    };

    set({
      rallies: entry.rallies,
      highlights: entry.highlights,
      past: [...state.past, pastEntry],
      future,
      hasUnsavedChanges: true,
    });

    debouncedSave(() => get().saveToStorage());
  },

  resetToOriginal: () => {
    const state = get();
    if (state.originalRallies.length === 0) return;

    // Push current state to history before resetting
    state.pushHistory();

    set({
      rallies: [...state.originalRallies],
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
          rallies: state.rallies,
          highlights: state.highlights,
          past: state.past,
          future: state.future,
          originalRallies: state.originalRallies,
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
      if (!persisted.rallies) return false;

      set({
        rallies: persisted.rallies,
        highlights: persisted.highlights || [],
        past: persisted.past || [],
        future: persisted.future || [],
        originalRallies: persisted.originalRallies || [],
        originalHighlights: persisted.originalHighlights || [],
        hasUnsavedChanges: (persisted.past?.length ?? 0) > 0,
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
      rallyIds: [],
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

  addRallyToHighlight: (rallyId: string, highlightId: string) => {
    const state = get();
    const highlight = state.highlights.find((h) => h.id === highlightId);
    if (!highlight) return;

    // Don't add if already in highlight
    if (highlight.rallyIds.includes(rallyId)) return;

    state.pushHistory();

    set({
      highlights: state.highlights.map((h) =>
        h.id === highlightId
          ? { ...h, rallyIds: [...h.rallyIds, rallyId] }
          : h
      ),
      hasUnsavedChanges: true,
    });

    debouncedSave(() => get().saveToStorage());
  },

  removeRallyFromHighlight: (rallyId: string, highlightId: string) => {
    const state = get();
    state.pushHistory();

    set({
      highlights: state.highlights.map((h) =>
        h.id === highlightId
          ? { ...h, rallyIds: h.rallyIds.filter((id) => id !== rallyId) }
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
  canUndo: () => (get().past?.length ?? 0) > 0,
  canRedo: () => (get().future?.length ?? 0) > 0,
  hasChangesFromOriginal: () => {
    const state = get();
    if (!state.originalRallies || state.originalRallies.length === 0) return false;
    if (!state.rallies) return false;
    if (state.rallies.length !== state.originalRallies.length) return true;
    return JSON.stringify(state.rallies) !== JSON.stringify(state.originalRallies);
  },

  canCreateHighlight: () => {
    const state = get();
    // Can create if no highlights OR all existing highlights have at least one rally
    if (!state.highlights || state.highlights.length === 0) return true;
    return state.highlights.every((h) => h.rallyIds?.length > 0);
  },

  getNextHighlightColor: () => {
    const state = get();
    const highlights = state.highlights ?? [];
    const usedColors = highlights.map((h) => h.color);
    // Find first unused color
    const available = HIGHLIGHT_COLORS.find((c) => !usedColors.includes(c));
    // If all used, cycle through
    return available || HIGHLIGHT_COLORS[highlights.length % HIGHLIGHT_COLORS.length];
  },

  getHighlightsForRally: (rallyId: string) => {
    const state = get();
    return (state.highlights ?? []).filter((h) => h.rallyIds?.includes(rallyId));
  },
}));
