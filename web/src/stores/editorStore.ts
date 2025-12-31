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
  Session,
  Match,
  SessionManifest,
} from '@/types/rally';
import { usePlayerStore } from './playerStore';

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

// Session-aware storage format
interface PersistedSession {
  sessionId: string;
  matchRallies: Record<string, Rally[]>; // matchId -> rallies
  highlights: Highlight[];
  savedAt: number;
}

const STORAGE_KEY_PREFIX = 'rallycut_session_';
const MAX_HISTORY_SIZE = 50;
const DEFAULT_RALLY_DURATION = 7;

const getStorageKey = (sessionId: string) => `${STORAGE_KEY_PREFIX}${sessionId}_v1`;

// Debounce helper
let saveTimeout: ReturnType<typeof setTimeout> | null = null;
const debouncedSave = (fn: () => void) => {
  if (saveTimeout) clearTimeout(saveTimeout);
  saveTimeout = setTimeout(fn, 500);
};

interface EditorState {
  // Session state
  session: Session | null;
  activeMatchId: string | null;

  // Video state (derived from active match when in session mode)
  videoFile: File | null;
  videoUrl: string | null;
  videoMetadata: VideoMetadata | null;

  // Rally state (derived from active match when in session mode)
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
  originalRalliesPerMatch: Record<string, Rally[]>;

  // Highlights state
  highlights: Highlight[];
  selectedHighlightId: string | null;

  // Session actions
  loadSession: (sessionId: string) => Promise<void>;
  setActiveMatch: (matchId: string) => void;
  getActiveMatch: () => Match | null;
  getAllRallies: () => Rally[];
  getRallyMatch: (rallyId: string) => Match | null;

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

  // Highlights actions
  createHighlight: (name?: string) => string;
  deleteHighlight: (id: string) => void;
  renameHighlight: (id: string, name: string) => void;
  addRallyToHighlight: (rallyId: string, highlightId: string) => void;
  removeRallyFromHighlight: (rallyId: string, highlightId: string) => void;
  reorderHighlightRallies: (highlightId: string, fromIndex: number, toIndex: number) => void;
  moveRallyBetweenHighlights: (rallyId: string, fromHighlightId: string, toHighlightId: string, toIndex: number) => void;
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
  session: null,
  activeMatchId: null,
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
  originalRalliesPerMatch: {},

  // Highlights state
  highlights: [],
  selectedHighlightId: null,

  // Session actions
  loadSession: async (sessionId: string) => {
    try {
      // Fetch session manifest
      const manifestRes = await fetch(`/samples/session_${sessionId}/session.json`);
      if (!manifestRes.ok) {
        console.error('Failed to load session manifest');
        return;
      }
      const manifest: SessionManifest = await manifestRes.json();

      // Check localStorage for saved edits
      let savedData: PersistedSession | null = null;
      if (typeof window !== 'undefined') {
        try {
          const stored = localStorage.getItem(getStorageKey(sessionId));
          if (stored) {
            savedData = JSON.parse(stored);
          }
        } catch {
          // Ignore localStorage errors
        }
      }

      // Load all match data in parallel
      // Store original rallies per match for reset functionality
      const originalRalliesPerMatch: Record<string, Rally[]> = {};

      const matches: Match[] = await Promise.all(
        manifest.session.matches.map(async (matchRef) => {
          const dataRes = await fetch(`/samples/session_${sessionId}/${matchRef.dataFile}`);
          const matchData: RallyFile = await dataRes.json();

          // Rally IDs should already be prefixed in the JSON files
          // But if they're not, prefix them here for safety
          const originalRallies = matchData.rallies.map((rally) => ({
            ...rally,
            id: rally.id.startsWith(matchRef.id) ? rally.id : `${matchRef.id}_${rally.id}`,
          }));

          // Store the original rallies before applying localStorage edits
          originalRalliesPerMatch[matchRef.id] = originalRallies;

          // Apply saved edits from localStorage if available
          let rallies = originalRallies;
          if (savedData?.matchRallies?.[matchRef.id]) {
            rallies = savedData.matchRallies[matchRef.id];
          }

          return {
            id: matchRef.id,
            name: matchRef.name,
            videoUrl: `/samples/session_${sessionId}/${matchRef.videoFile}`,
            video: matchData.video,
            rallies,
          };
        })
      );

      // Get all original rallies flattened for the active match comparison
      const firstMatchOriginalRallies = originalRalliesPerMatch[matches[0]?.id] || [];

      // Build session object with restored highlights
      // Use the URL sessionId for consistency (not manifest.session.id which may differ)
      const session: Session = {
        id: sessionId,
        name: manifest.session.name,
        matches,
        highlights: savedData?.highlights || [],
      };

      // Set first match as active
      const firstMatch = matches[0];

      set({
        session,
        activeMatchId: firstMatch?.id || null,
        videoUrl: firstMatch?.videoUrl || null,
        videoMetadata: firstMatch?.video || null,
        rallies: firstMatch?.rallies || [],
        highlights: session.highlights,
        selectedRallyId: null,
        selectedHighlightId: null,
        past: [],
        future: [],
        originalRallies: firstMatchOriginalRallies,
        originalHighlights: [],
        originalRalliesPerMatch,
        hasUnsavedChanges: savedData !== null,
      });
    } catch (error) {
      console.error('Error loading session:', error);
    }
  },

  setActiveMatch: (matchId: string) => {
    const state = get();
    if (!state.session) return;

    const match = state.session.matches.find((m) => m.id === matchId);
    if (!match) return;

    // Sync current rallies back to the session before switching
    const updatedMatches = state.session.matches.map((m) => {
      if (m.id === state.activeMatchId) {
        return { ...m, rallies: state.rallies };
      }
      return m;
    });

    // Only reset player state if NOT in the middle of highlight playback
    // During highlight playback, VideoPlayer handles seeking after video loads
    const playerState = usePlayerStore.getState();
    if (!playerState.playingHighlightId) {
      playerState.pause();
      playerState.seek(0);
    }

    set({
      session: { ...state.session, matches: updatedMatches },
      activeMatchId: matchId,
      videoUrl: match.videoUrl,
      videoMetadata: match.video,
      rallies: match.rallies,
      originalRallies: state.originalRalliesPerMatch[matchId] || [],
      selectedRallyId: null,
    });
  },

  getActiveMatch: () => {
    const state = get();
    if (!state.session || !state.activeMatchId) return null;
    return state.session.matches.find((m) => m.id === state.activeMatchId) || null;
  },

  getAllRallies: () => {
    const state = get();
    if (!state.session) return state.rallies;
    // Return rallies from all matches, with current active match's edited rallies
    return state.session.matches.flatMap((m) => {
      if (m.id === state.activeMatchId) {
        return state.rallies; // Use current edited rallies for active match
      }
      return m.rallies;
    });
  },

  getRallyMatch: (rallyId: string) => {
    const state = get();
    if (!state.session) return null;
    // Extract match ID from rally ID prefix (e.g., "match_1_rally_5" -> "match_1")
    const matchId = rallyId.replace(/_rally_\d+$/, '');
    return state.session.matches.find((m) => m.id === matchId) || null;
  },

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

    set({
      rallies: json.rallies,
      highlights: json.highlights || [],
      videoMetadata: json.video,
      originalJson: json,
      originalRallies,
      originalHighlights,
      past: [],
      future: [],
      hasUnsavedChanges: false,
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

  createRallyAtTime: (time: number, duration: number = DEFAULT_RALLY_DURATION) => {
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

    // Clear localStorage for current session
    if (typeof window !== 'undefined' && state.session) {
      try {
        localStorage.removeItem(getStorageKey(state.session.id));
      } catch {
        // Ignore localStorage errors
      }
    }

    set({
      session: null,
      activeMatchId: null,
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
      originalRalliesPerMatch: {},
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

    // In session mode, also reset all matches to their original rallies
    let updatedSession = state.session;
    if (state.session && Object.keys(state.originalRalliesPerMatch).length > 0) {
      const updatedMatches = state.session.matches.map((m) => ({
        ...m,
        rallies: state.originalRalliesPerMatch[m.id] || m.rallies,
      }));
      updatedSession = { ...state.session, matches: updatedMatches, highlights: [] };
    }

    set({
      session: updatedSession,
      rallies: [...state.originalRallies],
      highlights: [...state.originalHighlights],
      future: [], // Clear redo stack
      hasUnsavedChanges: false,
    });

    // Clear localStorage since we're back to original
    if (typeof window !== 'undefined' && state.session) {
      try {
        localStorage.removeItem(getStorageKey(state.session.id));
      } catch {
        // Ignore localStorage errors
      }
    }
  },

  saveToStorage: () => {
    const state = get();
    if (!state.session) return;

    if (typeof window !== 'undefined') {
      try {
        // Build matchRallies from session, with current active match's rallies updated
        const matchRallies: Record<string, Rally[]> = {};
        for (const match of state.session.matches) {
          if (match.id === state.activeMatchId) {
            // Use the current edited rallies for active match
            matchRallies[match.id] = state.rallies;
          } else {
            // Use the session's stored rallies for other matches
            matchRallies[match.id] = match.rallies;
          }
        }

        const data: PersistedSession = {
          sessionId: state.session.id,
          matchRallies,
          highlights: state.highlights,
          savedAt: Date.now(),
        };
        localStorage.setItem(getStorageKey(state.session.id), JSON.stringify(data));
      } catch {
        // Ignore localStorage errors
      }
    }
  },

  loadFromStorage: () => {
    const state = get();
    if (typeof window === 'undefined' || !state.session) return false;

    try {
      const stored = localStorage.getItem(getStorageKey(state.session.id));
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

  reorderHighlightRallies: (highlightId: string, fromIndex: number, toIndex: number) => {
    const state = get();
    const highlight = state.highlights.find((h) => h.id === highlightId);
    if (!highlight) return;

    state.pushHistory();

    const rallyIds = [...highlight.rallyIds];
    const [removed] = rallyIds.splice(fromIndex, 1);
    rallyIds.splice(toIndex, 0, removed);

    set({
      highlights: state.highlights.map((h) =>
        h.id === highlightId ? { ...h, rallyIds } : h
      ),
      hasUnsavedChanges: true,
    });

    debouncedSave(() => get().saveToStorage());
  },

  moveRallyBetweenHighlights: (rallyId: string, fromHighlightId: string, toHighlightId: string, toIndex: number) => {
    const state = get();
    const fromHighlight = state.highlights.find((h) => h.id === fromHighlightId);
    const toHighlight = state.highlights.find((h) => h.id === toHighlightId);
    if (!fromHighlight || !toHighlight) return;
    if (!fromHighlight.rallyIds.includes(rallyId)) return;

    state.pushHistory();

    set({
      highlights: state.highlights.map((h) => {
        if (h.id === fromHighlightId) {
          // Remove from source highlight
          return { ...h, rallyIds: h.rallyIds.filter((id) => id !== rallyId) };
        }
        if (h.id === toHighlightId) {
          // Add to target highlight at specified index
          const newRallyIds = [...h.rallyIds];
          // Don't add if already exists (shouldn't happen but safety check)
          if (!newRallyIds.includes(rallyId)) {
            newRallyIds.splice(toIndex, 0, rallyId);
          }
          return { ...h, rallyIds: newRallyIds };
        }
        return h;
      }),
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
