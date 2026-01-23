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
import { fetchSession as fetchSessionFromApi, fetchVideoForEditor, getCurrentUser, getConfirmationStatus, AccessDeniedError, type CameraEditMap, type GlobalCameraSettingsMap } from '@/services/api';
import { useCameraStore } from './cameraStore';
import type { RallyCameraEdit } from '@/types/camera';
import { syncService } from '@/services/syncService';

// History management types
interface HistoryEntry {
  rallies: Rally[];
  highlights: Highlight[];
  cameraEdits: Record<string, RallyCameraEdit>;
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
  cameraEdits?: Record<string, RallyCameraEdit>; // rallyId -> camera edit
  savedAt: number;
}

const STORAGE_KEY_PREFIX = 'rallycut_session_';
const MAX_HISTORY_SIZE = 50;
const DEFAULT_RALLY_DURATION = 7;
const SESSION_CACHE_TTL = 60 * 1000; // 1 minute

const getStorageKey = (sessionId: string) => `${STORAGE_KEY_PREFIX}${sessionId}_v1`;

// Debounce helper
let saveTimeout: ReturnType<typeof setTimeout> | null = null;
const debouncedSave = (fn: () => void) => {
  if (saveTimeout) clearTimeout(saveTimeout);
  saveTimeout = setTimeout(fn, 500);
};

// Sync subscription cleanup
let syncUnsubscribe: (() => void) | null = null;

// Sync status type
interface SyncStatus {
  isSyncing: boolean;
  pendingCount: number;
  error: string | null;
  lastSyncAt: number;
}

// Confirmation status type (per video/match)
export interface ConfirmationStatus {
  id: string;
  status: 'PENDING' | 'PROCESSING' | 'CONFIRMED' | 'FAILED';
  progress: number;
  error?: string | null;
  confirmedAt?: string | null;
  originalDurationMs: number;
  trimmedDurationMs?: number | null;
}

// Session error state for access denied scenarios
interface SessionError {
  type: 'not_found' | 'access_denied' | 'unknown';
  sessionName?: string;
  ownerName?: string | null;
  hasPendingRequest?: boolean;
}

interface EditorState {
  // Session loading state
  isLoadingSession: boolean;
  sessionLoadStep: string;
  sessionLoadProgress: number;
  sessionError: SessionError | null;
  clearSessionError: () => void;

  // Session state
  session: Session | null;
  activeMatchId: string | null;
  userRole: 'owner' | 'member' | null;
  currentUserId: string | null;
  currentUserName: string | null;
  currentUserEmail: string | null;

  // Video state (derived from active match when in session mode)
  videoFile: File | null;
  videoUrl: string | null;
  posterUrl: string | null;
  proxyUrl: string | null;
  videoMetadata: VideoMetadata | null;

  // Rally state (derived from active match when in session mode)
  rallies: Rally[];
  selectedRallyId: string | null;
  hasUnsavedChanges: boolean;

  // Sync state
  syncStatus: SyncStatus;
  localStorageError: string | null;

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

  // Confirmation state (per match/video)
  confirmationStatus: Record<string, ConfirmationStatus | null>;
  isConfirming: boolean;

  // Camera edit mode state
  isCameraTabActive: boolean;
  setIsCameraTabActive: (active: boolean) => void;

  // Add video modal state
  showAddVideoModal: boolean;
  setShowAddVideoModal: (show: boolean) => void;

  // Left panel tab state
  leftPanelTab: 'rallies' | 'highlights';
  setLeftPanelTab: (tab: 'rallies' | 'highlights') => void;

  // Expanded highlights in panel
  expandedHighlightIds: Set<string>;
  expandHighlight: (id: string) => void;
  collapseHighlight: (id: string) => void;
  toggleHighlightExpanded: (id: string) => void;

  // Single video mode state
  singleVideoMode: boolean;
  singleVideoId: string | null;

  // Session cache (for back navigation)
  sessionCache: Record<string, { session: Session; cameraEdits: CameraEditMap; globalCameraSettings: GlobalCameraSettingsMap; timestamp: number }>;
  invalidateSessionCache: (sessionId?: string) => void;

  // Session actions
  loadSession: (sessionId: string, initialVideoId?: string) => Promise<void>;
  loadVideo: (videoId: string) => Promise<void>;
  reloadSession: () => Promise<void>;
  reloadCurrentMatch: () => Promise<{ ralliesCount: number } | null>;
  setActiveMatch: (matchId: string) => void;
  renameMatch: (matchId: string, name: string) => void;
  renameSession: (name: string) => void;
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
  mergeRallies: (firstId: string, secondId: string) => void;
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

  // Sync actions
  syncNow: () => Promise<boolean>;
  updateSyncStatus: (status: SyncStatus) => void;

  // Highlights actions
  createHighlight: (name?: string) => string;
  deleteHighlight: (id: string) => void;
  renameHighlight: (id: string, name: string) => void;
  addRallyToHighlight: (rallyId: string, highlightId: string) => void;
  removeRallyFromHighlight: (rallyId: string, highlightId: string) => void;
  reorderHighlightRallies: (highlightId: string, fromIndex: number, toIndex: number) => void;
  moveRallyBetweenHighlights: (rallyId: string, fromHighlightId: string, toHighlightId: string, toIndex: number) => void;
  selectHighlight: (id: string | null) => void;

  // User actions
  setCurrentUser: (userId: string, userName: string | null) => void;

  // Confirmation actions
  setConfirmationStatus: (matchId: string, status: ConfirmationStatus | null) => void;
  setIsConfirming: (isConfirming: boolean) => void;
  updateConfirmationProgress: (matchId: string, progress: number) => void;

  // Computed helpers (called as functions since Zustand doesn't have computed)
  canUndo: () => boolean;
  canRedo: () => boolean;
  hasChangesFromOriginal: () => boolean;
  canCreateHighlight: () => boolean;
  canEditHighlight: (highlightId: string) => boolean;
  getNextHighlightColor: () => string;
  getHighlightsForRally: (rallyId: string) => Highlight[];
  isRallyEditingLocked: () => boolean;

  // Rally recording state (M key hotkey)
  isRecordingRally: boolean;
  rallyRecordingStartTime: number | null;
  startRallyRecording: () => void;
  stopRallyRecording: (endTime: number) => void;
  cancelRallyRecording: () => void;
}

export const useEditorStore = create<EditorState>((set, get) => ({
  // Initial state
  isLoadingSession: false,
  sessionLoadStep: '',
  sessionLoadProgress: 0,
  sessionError: null,
  clearSessionError: () => set({ sessionError: null }),

  session: null,
  activeMatchId: null,
  userRole: null,
  currentUserId: null,
  currentUserName: null,
  currentUserEmail: null,
  videoFile: null,
  videoUrl: null,
  posterUrl: null,
  proxyUrl: null,
  videoMetadata: null,
  rallies: [],
  selectedRallyId: null,
  hasUnsavedChanges: false,
  originalJson: null,

  // Sync state
  syncStatus: {
    isSyncing: false,
    pendingCount: 0,
    error: null,
    lastSyncAt: 0,
  },
  localStorageError: null,

  // History state
  past: [],
  future: [],
  originalRallies: [],
  originalHighlights: [],
  originalRalliesPerMatch: {},

  // Highlights state
  highlights: [],
  selectedHighlightId: null,

  // Confirmation state
  confirmationStatus: {},
  isConfirming: false,

  // Camera edit mode state
  isCameraTabActive: false,
  setIsCameraTabActive: (active: boolean) => set({ isCameraTabActive: active }),

  // Rally recording state (M key hotkey)
  isRecordingRally: false,
  rallyRecordingStartTime: null,

  // Add video modal state
  showAddVideoModal: false,
  setShowAddVideoModal: (show: boolean) => set({ showAddVideoModal: show }),

  // Left panel tab state
  leftPanelTab: 'rallies',
  setLeftPanelTab: (tab: 'rallies' | 'highlights') => set({ leftPanelTab: tab }),

  // Expanded highlights in panel
  expandedHighlightIds: new Set<string>(),
  expandHighlight: (id: string) => set((state) => ({
    expandedHighlightIds: new Set(state.expandedHighlightIds).add(id),
  })),
  collapseHighlight: (id: string) => set((state) => {
    const next = new Set(state.expandedHighlightIds);
    next.delete(id);
    return { expandedHighlightIds: next };
  }),
  toggleHighlightExpanded: (id: string) => set((state) => {
    const next = new Set(state.expandedHighlightIds);
    if (next.has(id)) {
      next.delete(id);
    } else {
      next.add(id);
    }
    return { expandedHighlightIds: next };
  }),

  // Single video mode state
  singleVideoMode: false,
  singleVideoId: null,

  // Session cache (for back navigation - 1 minute TTL)
  sessionCache: {},
  invalidateSessionCache: (sessionId?: string) => {
    if (sessionId) {
      set((state) => {
        const newCache = { ...state.sessionCache };
        delete newCache[sessionId];
        return { sessionCache: newCache };
      });
    } else {
      set({ sessionCache: {} });
    }
  },

  // Session actions
  loadSession: async (sessionId: string, initialVideoId?: string) => {
    // Start loading and clear any previous error
    set({
      isLoadingSession: true,
      sessionLoadStep: 'Loading session...',
      sessionLoadProgress: 10,
      sessionError: null,
    });

    try {
      let session: Session;
      const originalRalliesPerMatch: Record<string, Rally[]> = {};

      // Try to load from API first
      const useApi = process.env.NEXT_PUBLIC_API_URL && !sessionId.startsWith('local_');

      if (useApi) {
        // Check session cache first (for back navigation)
        const cached = get().sessionCache[sessionId];
        let result: { session: Session; cameraEdits: CameraEditMap; globalCameraSettings: GlobalCameraSettingsMap };

        if (cached && Date.now() - cached.timestamp < SESSION_CACHE_TTL) {
          // Use cached data
          result = { session: cached.session, cameraEdits: cached.cameraEdits, globalCameraSettings: cached.globalCameraSettings };
        } else {
          // Fetch from backend API
          result = await fetchSessionFromApi(sessionId);

          // Store in cache for future back-navigation
          set((state) => ({
            sessionCache: {
              ...state.sessionCache,
              [sessionId]: {
                session: result.session,
                cameraEdits: result.cameraEdits,
                globalCameraSettings: result.globalCameraSettings,
                timestamp: Date.now(),
              },
            },
          }));
        }

        session = result.session;

        // Load camera edits and global settings into camera store (migration handled by loadCameraEdits)
        // Pass raw data - loadCameraEdits will migrate old format if needed
        useCameraStore.getState().loadCameraEdits(result.cameraEdits, result.globalCameraSettings);

        // Update progress
        set({
          sessionLoadStep: 'Preparing video...',
          sessionLoadProgress: 50,
        });

        // Fetch current user info
        try {
          const user = await getCurrentUser();
          set({ currentUserId: user.id, currentUserName: user.name, currentUserEmail: user.email });
        } catch (e) {
          console.warn('Failed to load current user:', e);
        }

        // Initialize sync service
        syncService.init(sessionId);

        // Store original rallies per match
        for (const match of session.matches) {
          originalRalliesPerMatch[match.id] = [...match.rallies];
        }

        // Check localStorage for unsaved local edits
        let savedData: PersistedSession | null = null;
        if (typeof window !== 'undefined') {
          try {
            const stored = localStorage.getItem(getStorageKey(sessionId));
            if (stored) {
              savedData = JSON.parse(stored);
            }
          } catch (e) {
            console.warn('Failed to load from localStorage:', e);
          }
        }

        // Compare timestamps to decide whether to apply local edits
        const apiUpdatedAt = session.updatedAt ? new Date(session.updatedAt).getTime() : 0;
        const localSavedAt = savedData?.savedAt;

        // Apply local edits if:
        // 1. We have local rallies AND
        // 2. Either local data has no timestamp (legacy format - be conservative) OR local is newer than backend
        const shouldApplyLocalEdits = savedData?.matchRallies && (
          localSavedAt === undefined || // Legacy localStorage without timestamp - apply to be safe
          localSavedAt > apiUpdatedAt   // Local edits are newer
        );

        if (shouldApplyLocalEdits && savedData) {
          // Local edits should be applied
          session = {
            ...session,
            matches: session.matches.map((match) => {
              const localRallies = savedData.matchRallies?.[match.id];
              if (localRallies && localRallies.length > 0) {
                return { ...match, rallies: localRallies };
              }
              return match;
            }),
            highlights: savedData.highlights || session.highlights,
          };

          // Load camera edits from localStorage (overrides API data)
          if (savedData.cameraEdits && Object.keys(savedData.cameraEdits).length > 0) {
            useCameraStore.getState().loadCameraEdits(savedData.cameraEdits);
          }
        } else if (savedData && localSavedAt !== undefined && localSavedAt <= apiUpdatedAt) {
          // Backend is newer and we have a timestamp to compare - discard stale local cache
          localStorage.removeItem(getStorageKey(sessionId));
          console.info('Discarded stale localStorage edits - backend has newer data');
        }

        // Set up state getter for sync service
        syncService.setStateGetter(() => ({
          session: get().session,
          rallies: get().rallies,
          highlights: get().highlights,
          activeMatchId: get().activeMatchId,
        }));

        // Subscribe to sync status updates (store unsubscribe for cleanup)
        if (syncUnsubscribe) {
          syncUnsubscribe();
        }
        syncUnsubscribe = syncService.subscribe((status) => {
          get().updateSyncStatus(status);
        });
      } else {
        // Fallback to static files (for local development without backend)
        set({
          sessionLoadStep: 'Loading from local files...',
          sessionLoadProgress: 20,
        });

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
          } catch (e) {
            console.warn('Failed to load from localStorage:', e);
          }
        }

        // Update progress
        set({
          sessionLoadStep: 'Loading video data...',
          sessionLoadProgress: 40,
        });

        // Load all match data in parallel
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

        // Build session object with restored highlights
        session = {
          id: sessionId,
          name: manifest.session.name,
          matches,
          highlights: savedData?.highlights || [],
        };

        // Load camera edits from localStorage
        if (savedData?.cameraEdits && Object.keys(savedData.cameraEdits).length > 0) {
          useCameraStore.getState().loadCameraEdits(savedData.cameraEdits);
        }
      }

      // Determine which match to set as active - use initialVideoId if provided and valid
      const initialMatch = initialVideoId
        ? session.matches.find(m => m.id === initialVideoId)
        : null;
      const activeMatch = initialMatch || session.matches[0];

      // Get all original rallies flattened for the active match comparison
      const activeMatchOriginalRallies = originalRalliesPerMatch[activeMatch?.id] || [];

      // Final progress update before setting session
      set({
        sessionLoadStep: 'Ready',
        sessionLoadProgress: 100,
      });

      set({
        isLoadingSession: false,
        sessionLoadStep: '',
        sessionLoadProgress: 0,
        session,
        activeMatchId: activeMatch?.id || null,
        userRole: session.userRole || 'owner', // Default to owner if not specified
        videoUrl: activeMatch?.videoUrl || null,
        posterUrl: activeMatch?.posterUrl || null,
        proxyUrl: activeMatch?.proxyUrl || null,
        videoMetadata: activeMatch?.video || null,
        rallies: activeMatch?.rallies || [],
        highlights: session.highlights,
        selectedRallyId: null,
        selectedHighlightId: null,
        past: [],
        future: [],
        originalRallies: activeMatchOriginalRallies,
        originalHighlights: [],
        originalRalliesPerMatch,
        hasUnsavedChanges: false,
      });

      // Enable "skip dead time" by default for guests with rallies (if no explicit preference)
      const hasRallies = session.matches.some(m => m.rallies.length > 0);
      const isGuest = session.userRole === 'member';
      const hasExplicitPreference = typeof window !== 'undefined' &&
        localStorage.getItem('rallycut-play-only-rallies') !== null;

      if (isGuest && hasRallies && !hasExplicitPreference) {
        usePlayerStore.getState().setPlayOnlyRallies(true);
      }

      // Load confirmation status for all matches
      for (const match of session.matches) {
        try {
          const confirmResult = await getConfirmationStatus(match.id);
          if (confirmResult.confirmation) {
            get().setConfirmationStatus(match.id, {
              id: confirmResult.confirmation.id,
              status: confirmResult.confirmation.status,
              progress: confirmResult.confirmation.progress,
              error: confirmResult.confirmation.error,
              confirmedAt: confirmResult.confirmation.confirmedAt,
              originalDurationMs: confirmResult.confirmation.originalDurationMs,
              trimmedDurationMs: confirmResult.confirmation.trimmedDurationMs,
            });
          }
        } catch {
          // Ignore - no confirmation exists for this video
        }
      }
    } catch (error) {
      // Handle AccessDeniedError specially
      if (error instanceof AccessDeniedError) {
        set({
          isLoadingSession: false,
          sessionLoadStep: '',
          sessionLoadProgress: 0,
          sessionError: {
            type: 'access_denied',
            sessionName: error.sessionName,
            ownerName: error.ownerName,
            hasPendingRequest: error.hasPendingRequest,
          },
        });
        return;
      }

      // Handle other errors
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      const isNotFound = errorMessage.includes('404') || errorMessage.toLowerCase().includes('not found');

      set({
        isLoadingSession: false,
        sessionLoadStep: '',
        sessionLoadProgress: 0,
        sessionError: {
          type: isNotFound ? 'not_found' : 'unknown',
        },
      });
      console.error('Error loading session:', error);
    }
  },

  // Load a single video for editing (uses ALL_VIDEOS session for sync)
  loadVideo: async (videoId: string) => {
    // Start loading
    set({
      isLoadingSession: true,
      sessionLoadStep: 'Loading video...',
      sessionLoadProgress: 10,
      singleVideoMode: true,
      singleVideoId: videoId,
    });

    try {
      // Fetch video data from API
      const result = await fetchVideoForEditor(videoId);

      // Update progress
      set({
        sessionLoadStep: 'Preparing video...',
        sessionLoadProgress: 50,
      });

      // Load camera edits and global settings into camera store
      useCameraStore.getState().loadCameraEdits(result.cameraEdits, result.globalCameraSettings);

      // Fetch current user info
      try {
        const user = await getCurrentUser();
        set({ currentUserId: user.id, currentUserName: user.name, currentUserEmail: user.email });
      } catch (e) {
        console.warn('Failed to load current user:', e);
      }

      // Initialize sync service with the ALL_VIDEOS session ID
      syncService.init(result.allVideosSessionId);

      // Build a virtual session with single match for the editor
      const session: Session = {
        id: result.allVideosSessionId,
        name: result.video.name,
        matches: [result.match],
        highlights: result.highlights,
        userRole: 'owner',
      };

      // Store original rallies
      const originalRalliesPerMatch: Record<string, Rally[]> = {
        [videoId]: [...result.match.rallies],
      };

      // Set up state getter for sync service
      syncService.setStateGetter(() => ({
        session: get().session,
        rallies: get().rallies,
        highlights: get().highlights,
        activeMatchId: get().activeMatchId,
      }));

      // Subscribe to sync status updates
      if (syncUnsubscribe) {
        syncUnsubscribe();
      }
      syncUnsubscribe = syncService.subscribe((status) => {
        get().updateSyncStatus(status);
      });

      // Final progress update
      set({
        sessionLoadStep: 'Ready',
        sessionLoadProgress: 100,
      });

      set({
        isLoadingSession: false,
        sessionLoadStep: '',
        sessionLoadProgress: 0,
        session,
        activeMatchId: videoId,
        userRole: 'owner',
        videoUrl: result.match.videoUrl,
        posterUrl: result.match.posterUrl || null,
        proxyUrl: result.match.proxyUrl || null,
        videoMetadata: result.match.video,
        rallies: result.match.rallies,
        highlights: result.highlights,
        selectedRallyId: null,
        selectedHighlightId: null,
        past: [],
        future: [],
        originalRallies: result.match.rallies,
        originalHighlights: [],
        originalRalliesPerMatch,
        hasUnsavedChanges: false,
      });

      // Load confirmation status for this video
      try {
        const confirmResult = await getConfirmationStatus(videoId);
        if (confirmResult.confirmation) {
          get().setConfirmationStatus(videoId, {
            id: confirmResult.confirmation.id,
            status: confirmResult.confirmation.status,
            progress: confirmResult.confirmation.progress,
            error: confirmResult.confirmation.error,
            confirmedAt: confirmResult.confirmation.confirmedAt,
            originalDurationMs: confirmResult.confirmation.originalDurationMs,
            trimmedDurationMs: confirmResult.confirmation.trimmedDurationMs,
          });
        }
      } catch {
        // Ignore - no confirmation exists for this video
      }
    } catch (error) {
      // Clear loading state on error
      set({
        isLoadingSession: false,
        sessionLoadStep: '',
        sessionLoadProgress: 0,
        singleVideoMode: false,
        singleVideoId: null,
      });
      console.error('Error loading video:', error);
      throw error;
    }
  },

  reloadSession: async () => {
    const state = get();
    if (!state.session) return;

    try {
      // Fetch fresh session data from API
      const result = await fetchSessionFromApi(state.session.id);
      const freshSession = result.session;

      // Load camera edits and global settings into camera store (migration handled by loadCameraEdits)
      useCameraStore.getState().loadCameraEdits(result.cameraEdits, result.globalCameraSettings);

      // Preserve activeMatchId if the match still exists, otherwise use first match
      const matchStillExists = freshSession.matches.some((m: { id: string }) => m.id === state.activeMatchId);
      const activeMatchId = matchStillExists ? state.activeMatchId : freshSession.matches[0]?.id || null;
      const activeMatch = freshSession.matches.find((m: { id: string }) => m.id === activeMatchId);

      // Update original rallies for all matches
      const originalRalliesPerMatch: Record<string, Rally[]> = {};
      for (const match of freshSession.matches) {
        originalRalliesPerMatch[match.id] = [...match.rallies];
      }

      set({
        session: freshSession,
        activeMatchId,
        userRole: freshSession.userRole || 'owner',
        videoUrl: activeMatch?.videoUrl || null,
        posterUrl: activeMatch?.posterUrl || null,
        proxyUrl: activeMatch?.proxyUrl || null,
        videoMetadata: activeMatch?.video || null,
        rallies: activeMatch?.rallies || [],
        highlights: freshSession.highlights,
        originalRallies: activeMatch?.rallies || [],
        originalRalliesPerMatch,
        past: [],
        future: [],
        hasUnsavedChanges: false,
      });
    } catch (error) {
      console.error('Error reloading session:', error);
    }
  },

  reloadCurrentMatch: async () => {
    const state = get();
    if (!state.activeMatchId) return null;

    // Track previous rally count to detect when detection adds rallies
    const previousRalliesCount = state.rallies.length;

    try {
      // Handle single video mode differently - fetch video directly
      if (state.singleVideoMode && state.singleVideoId) {
        const result = await fetchVideoForEditor(state.singleVideoId);

        // Load camera edits and global settings into camera store
        useCameraStore.getState().loadCameraEdits(result.cameraEdits, result.globalCameraSettings);

        // Build updated session with fresh match data
        const updatedSession = state.session ? {
          ...state.session,
          matches: [result.match],
        } : null;

        // Update state with fresh video data
        set({
          session: updatedSession,
          videoUrl: result.match.videoUrl,
          proxyUrl: result.match.proxyUrl || null,
          posterUrl: result.match.posterUrl || null,
          videoMetadata: result.match.video,
          rallies: result.match.rallies,
          originalRallies: result.match.rallies,
          past: [],
          future: [],
          hasUnsavedChanges: false,
        });

        // Enable "skip dead time" when detection adds rallies (if no explicit preference)
        const ralliesAdded = previousRalliesCount === 0 && result.match.rallies.length > 0;
        const hasExplicitPreference = typeof window !== 'undefined' &&
          localStorage.getItem('rallycut-play-only-rallies') !== null;

        if (ralliesAdded && !hasExplicitPreference) {
          usePlayerStore.getState().setPlayOnlyRallies(true);
        }

        return { ralliesCount: result.match.rallies.length };
      }

      // Session mode - fetch full session
      if (!state.session) return null;

      // Fetch fresh session data from API
      const result = await fetchSessionFromApi(state.session.id);
      const freshSession = result.session;
      const freshMatch = freshSession.matches.find((m: { id: string }) => m.id === state.activeMatchId);

      // Load camera edits and global settings into camera store (migration handled by loadCameraEdits)
      useCameraStore.getState().loadCameraEdits(result.cameraEdits, result.globalCameraSettings);

      if (!freshMatch) return null;

      // Update the session with fresh match data
      const updatedMatches = state.session.matches.map((m) =>
        m.id === state.activeMatchId ? freshMatch : m
      );

      // Update original rallies for this match
      const newOriginalRalliesPerMatch = {
        ...state.originalRalliesPerMatch,
        [state.activeMatchId]: [...freshMatch.rallies],
      };

      set({
        session: { ...state.session, matches: updatedMatches },
        // Update video URLs and metadata (important after confirmation changes proxy/duration)
        videoUrl: freshMatch.videoUrl,
        proxyUrl: freshMatch.proxyUrl || null,
        posterUrl: freshMatch.posterUrl || null,
        videoMetadata: freshMatch.video,
        rallies: freshMatch.rallies,
        originalRallies: freshMatch.rallies,
        originalRalliesPerMatch: newOriginalRalliesPerMatch,
        past: [],
        future: [],
        hasUnsavedChanges: false,
      });

      // Enable "skip dead time" when detection adds rallies (if no explicit preference)
      const ralliesAdded = previousRalliesCount === 0 && freshMatch.rallies.length > 0;
      const hasExplicitPreference = typeof window !== 'undefined' &&
        localStorage.getItem('rallycut-play-only-rallies') !== null;

      if (ralliesAdded && !hasExplicitPreference) {
        usePlayerStore.getState().setPlayOnlyRallies(true);
      }

      return { ralliesCount: freshMatch.rallies.length };
    } catch (error) {
      console.error('Error reloading match:', error);
      return null;
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
      posterUrl: match.posterUrl || null,
      proxyUrl: match.proxyUrl || null,
      videoMetadata: match.video,
      rallies: match.rallies,
      originalRallies: state.originalRalliesPerMatch[matchId] || [],
      selectedRallyId: null,
      // Cancel any in-progress rally recording when switching videos
      isRecordingRally: false,
      rallyRecordingStartTime: null,
    });
  },

  renameMatch: (matchId: string, name: string) => {
    const state = get();
    if (!state.session) return;

    // Update local state
    set({
      session: {
        ...state.session,
        matches: state.session.matches.map((m) =>
          m.id === matchId ? { ...m, name } : m
        ),
      },
    });
  },

  renameSession: (name: string) => {
    const state = get();
    if (!state.session) return;

    set({
      session: {
        ...state.session,
        name,
      },
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
    // Find match by checking each match's rallies array
    // This avoids regex issues when match ID contains "_rally_" (e.g., "game_rally_1_rally_5")
    return state.session.matches.find((m) => m.rallies.some((r) => r.id === rallyId)) || null;
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
      posterUrl: null,
      proxyUrl: null,
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

    // Generate unique ID with match prefix for session mode
    const matchPrefix = state.activeMatchId ? `${state.activeMatchId}_` : '';
    const existingIds = state.rallies.map((s) => s.id);
    let counter = state.rallies.length + 1;
    let newId = `${matchPrefix}rally_${counter}`;
    while (existingIds.includes(newId)) {
      counter++;
      newId = `${matchPrefix}rally_${counter}`;
    }

    const newRally = createRally(newId, startTime, endTime, fps);

    set({
      rallies: [...state.rallies, newRally],
      hasUnsavedChanges: true,
    });

    // Mark dirty for backend sync
    syncService.markDirty();

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

    // Mark dirty for backend sync
    syncService.markDirty();

    debouncedSave(() => get().saveToStorage());
  },

  adjustRallyStart: (id: string, delta: number) => {
    const state = get();
    const rally = state.rallies.find((s) => s.id === id);
    if (!rally) return false;

    const newStart = rally.start_time + delta;
    const rallyDuration = rally.end_time - rally.start_time;

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

    // Keyframe boundary check - don't allow dragging past the first keyframe
    // When collapsing start (delta > 0), ensure we don't exclude keyframes
    const cameraEdit = useCameraStore.getState().cameraEdits[id];
    if (cameraEdit && delta > 0) {
      const allKeyframes = [
        ...(cameraEdit.keyframes.ORIGINAL ?? []),
        ...(cameraEdit.keyframes.VERTICAL ?? []),
      ];
      if (allKeyframes.length > 0) {
        // Find the earliest keyframe (smallest timeOffset)
        const firstKeyframeOffset = Math.min(...allKeyframes.map((kf) => kf.timeOffset));
        // Convert to absolute time
        const firstKeyframeAbsTime = rally.start_time + firstKeyframeOffset * rallyDuration;
        // Don't allow start to go past the first keyframe
        if (newStart > firstKeyframeAbsTime) return false;
      }
    }

    // Update the rally start time
    state.updateRally(id, { start_time: newStart });

    // Recalculate keyframe timeOffsets to keep them at the same absolute video time
    if (cameraEdit) {
      const newDuration = rally.end_time - newStart;
      const updateKeyframesForRatio = (keyframes: typeof cameraEdit.keyframes.ORIGINAL) => {
        return keyframes.map((kf) => {
          // Convert old timeOffset to absolute time
          const absTime = rally.start_time + kf.timeOffset * rallyDuration;
          // Convert back to new timeOffset
          const newTimeOffset = (absTime - newStart) / newDuration;
          return { ...kf, timeOffset: Math.max(0, Math.min(1, newTimeOffset)) };
        });
      };

      const updatedEdit = {
        ...cameraEdit,
        keyframes: {
          ORIGINAL: updateKeyframesForRatio(cameraEdit.keyframes.ORIGINAL ?? []),
          VERTICAL: updateKeyframesForRatio(cameraEdit.keyframes.VERTICAL ?? []),
        },
      };
      useCameraStore.getState().setCameraEdit(id, updatedEdit);
    }

    return true;
  },

  adjustRallyEnd: (id: string, delta: number) => {
    const state = get();
    const rally = state.rallies.find((s) => s.id === id);
    if (!rally) return false;

    const newEnd = rally.end_time + delta;
    const rallyDuration = rally.end_time - rally.start_time;
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

    // Keyframe boundary check - don't allow dragging past the last keyframe
    // When collapsing end (delta < 0), ensure we don't exclude keyframes
    const cameraEdit = useCameraStore.getState().cameraEdits[id];
    if (cameraEdit && delta < 0) {
      const allKeyframes = [
        ...(cameraEdit.keyframes.ORIGINAL ?? []),
        ...(cameraEdit.keyframes.VERTICAL ?? []),
      ];
      if (allKeyframes.length > 0) {
        // Find the latest keyframe (largest timeOffset)
        const lastKeyframeOffset = Math.max(...allKeyframes.map((kf) => kf.timeOffset));
        // Convert to absolute time
        const lastKeyframeAbsTime = rally.start_time + lastKeyframeOffset * rallyDuration;
        // Don't allow end to go before the last keyframe
        if (newEnd < lastKeyframeAbsTime) return false;
      }
    }

    // Update the rally end time
    state.updateRally(id, { end_time: newEnd });

    // Recalculate keyframe timeOffsets to keep them at the same absolute video time
    if (cameraEdit) {
      const newDuration = newEnd - rally.start_time;
      const updateKeyframesForRatio = (keyframes: typeof cameraEdit.keyframes.ORIGINAL) => {
        return keyframes.map((kf) => {
          // Convert old timeOffset to absolute time
          const absTime = rally.start_time + kf.timeOffset * rallyDuration;
          // Convert back to new timeOffset
          const newTimeOffset = (absTime - rally.start_time) / newDuration;
          return { ...kf, timeOffset: Math.max(0, Math.min(1, newTimeOffset)) };
        });
      };

      const updatedEdit = {
        ...cameraEdit,
        keyframes: {
          ORIGINAL: updateKeyframesForRatio(cameraEdit.keyframes.ORIGINAL ?? []),
          VERTICAL: updateKeyframesForRatio(cameraEdit.keyframes.VERTICAL ?? []),
        },
      };
      useCameraStore.getState().setCameraEdit(id, updatedEdit);
    }

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
    const startTime = time;
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

    // Remove camera edits for the deleted rally
    useCameraStore.getState().removeCameraEdit(id);

    // Mark dirty for backend sync
    syncService.markDirty();

    debouncedSave(() => get().saveToStorage());
  },

  mergeRallies: (firstId: string, secondId: string) => {
    const state = get();
    const first = state.rallies.find((r) => r.id === firstId);
    const second = state.rallies.find((r) => r.id === secondId);
    if (!first || !second) return;

    // Ensure first rally comes before second
    const [earlier, later] = first.start_time < second.start_time
      ? [first, second]
      : [second, first];
    const earlierId = earlier.id;
    const laterId = later.id;

    state.pushHistory();

    const fps = state.videoMetadata?.fps ?? 30;
    const merged = recalculateRally({
      ...earlier,
      end_time: later.end_time,
    }, fps);

    // Merge camera edits from both rallies
    const cameraStore = useCameraStore.getState();
    const earlierEdit = cameraStore.cameraEdits[earlierId];
    const laterEdit = cameraStore.cameraEdits[laterId];

    if (earlierEdit || laterEdit) {
      const earlierDuration = earlier.end_time - earlier.start_time;
      const laterDuration = later.end_time - later.start_time;
      const mergedDuration = later.end_time - earlier.start_time;

      // Determine aspect ratio: prefer VERTICAL if either has it
      const useVertical =
        earlierEdit?.aspectRatio === 'VERTICAL' ||
        laterEdit?.aspectRatio === 'VERTICAL';
      const mergedAspectRatio = useVertical ? 'VERTICAL' : 'ORIGINAL';

      // Helper to convert keyframe timeOffset to new merged rally coordinates
      const convertEarlierKeyframe = (kf: { id: string; timeOffset: number; positionX: number; positionY: number; zoom: number; rotation: number; easing: 'LINEAR' | 'EASE_IN' | 'EASE_OUT' | 'EASE_IN_OUT' }) => ({
        ...kf,
        // Earlier rally starts at 0, so just scale down the offset
        timeOffset: (kf.timeOffset * earlierDuration) / mergedDuration,
      });

      const convertLaterKeyframe = (kf: { id: string; timeOffset: number; positionX: number; positionY: number; zoom: number; rotation: number; easing: 'LINEAR' | 'EASE_IN' | 'EASE_OUT' | 'EASE_IN_OUT' }) => ({
        ...kf,
        // Later rally starts after the gap, calculate absolute time then convert to offset
        timeOffset: ((later.start_time - earlier.start_time) + kf.timeOffset * laterDuration) / mergedDuration,
      });

      // Merge keyframes for each aspect ratio
      const mergeKeyframesForRatio = (ratio: 'ORIGINAL' | 'VERTICAL') => {
        const earlierKfs = earlierEdit?.keyframes[ratio] ?? [];
        const laterKfs = laterEdit?.keyframes[ratio] ?? [];
        return [
          ...earlierKfs.map(convertEarlierKeyframe),
          ...laterKfs.map(convertLaterKeyframe),
        ].sort((a, b) => a.timeOffset - b.timeOffset);
      };

      const mergedCameraEdit: RallyCameraEdit = {
        enabled: earlierEdit?.enabled || laterEdit?.enabled || false,
        aspectRatio: mergedAspectRatio,
        keyframes: {
          ORIGINAL: mergeKeyframesForRatio('ORIGINAL'),
          VERTICAL: mergeKeyframesForRatio('VERTICAL'),
        },
      };

      // Set merged camera edit and remove later rally's edit
      cameraStore.setCameraEdit(earlierId, mergedCameraEdit);
      cameraStore.removeCameraEdit(laterId);
    }

    // Transfer highlights from later rally to earlier (deduplicate)
    const updatedHighlights = state.highlights.map((h) => ({
      ...h,
      rallyIds: h.rallyIds.includes(laterId)
        ? [...new Set([...h.rallyIds.filter((id) => id !== laterId), earlierId])]
        : h.rallyIds,
    }));

    set({
      rallies: state.rallies
        .filter((r) => r.id !== laterId)
        .map((r) => (r.id === earlierId ? merged : r)),
      highlights: updatedHighlights,
      selectedRallyId: earlierId,
      hasUnsavedChanges: true,
    });

    // Mark dirty for backend sync
    syncService.markDirty();

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
      } catch (e) {
        console.warn('Failed to clear localStorage:', e);
      }
    }

    // Clean up sync subscription and reset service
    if (syncUnsubscribe) {
      syncUnsubscribe();
      syncUnsubscribe = null;
    }
    syncService.reset();

    set({
      isLoadingSession: false,
      sessionLoadStep: '',
      sessionLoadProgress: 0,
      session: null,
      activeMatchId: null,
      userRole: null,
      currentUserId: null,
      currentUserName: null,
      currentUserEmail: null,
      videoFile: null,
      videoUrl: null,
      posterUrl: null,
      proxyUrl: null,
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
      syncStatus: {
        isSyncing: false,
        pendingCount: 0,
        error: null,
        lastSyncAt: 0,
      },
      confirmationStatus: {},
      isConfirming: false,
      isCameraTabActive: false,
    });
  },

  // History management actions
  pushHistory: () => {
    const state = get();
    const cameraEdits = useCameraStore.getState().cameraEdits;
    const entry: HistoryEntry = {
      rallies: [...state.rallies],
      highlights: [...state.highlights],
      cameraEdits: { ...cameraEdits },
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

    const currentCameraEdits = useCameraStore.getState().cameraEdits;
    const futureEntry: HistoryEntry = {
      rallies: [...state.rallies],
      highlights: [...state.highlights],
      cameraEdits: { ...currentCameraEdits },
      timestamp: Date.now(),
    };

    set({
      rallies: entry.rallies,
      highlights: entry.highlights,
      past,
      future: [...state.future, futureEntry],
      hasUnsavedChanges: past.length > 0 || state.future.length > 0,
    });

    // Restore camera edits if present in history entry
    if (entry.cameraEdits) {
      useCameraStore.getState().loadCameraEdits(entry.cameraEdits);
    }

    // Mark dirty for backend sync
    syncService.markDirty();

    debouncedSave(() => get().saveToStorage());
  },

  redo: () => {
    const state = get();
    if (state.future.length === 0) return;

    const future = [...state.future];
    const entry = future.pop()!;

    const currentCameraEdits = useCameraStore.getState().cameraEdits;
    const pastEntry: HistoryEntry = {
      rallies: [...state.rallies],
      highlights: [...state.highlights],
      cameraEdits: { ...currentCameraEdits },
      timestamp: Date.now(),
    };

    set({
      rallies: entry.rallies,
      highlights: entry.highlights,
      past: [...state.past, pastEntry],
      future,
      hasUnsavedChanges: true,
    });

    // Restore camera edits if present in history entry
    if (entry.cameraEdits) {
      useCameraStore.getState().loadCameraEdits(entry.cameraEdits);
    }

    // Mark dirty for backend sync
    syncService.markDirty();

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

    // Mark dirty for backend sync (to sync the reset state)
    syncService.markDirty();

    // Clear localStorage since we're back to original
    if (typeof window !== 'undefined' && state.session) {
      try {
        localStorage.removeItem(getStorageKey(state.session.id));
      } catch (e) {
        console.warn('Failed to clear localStorage on reset:', e);
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

        // Get camera edits from camera store
        const cameraEdits = useCameraStore.getState().cameraEdits;

        const data: PersistedSession = {
          sessionId: state.session.id,
          matchRallies,
          highlights: state.highlights,
          cameraEdits,
          savedAt: Date.now(),
        };

        const serialized = JSON.stringify(data);

        // Check size before saving (~5MB limit for localStorage)
        const sizeBytes = new Blob([serialized]).size;
        const maxBytes = 4 * 1024 * 1024; // 4MB safety margin (browser limit is ~5MB)
        if (sizeBytes > maxBytes) {
          console.error(
            `[localStorage] Session data too large (${(sizeBytes / 1024 / 1024).toFixed(2)}MB). ` +
            `Consider reducing rally count or upgrading to a paid tier for server sync.`
          );
          set({ localStorageError: 'Session too large for local storage. Upgrade for cloud sync.' });
          return;
        }

        localStorage.setItem(getStorageKey(state.session.id), serialized);
        // Clear any previous error on successful save
        if (state.localStorageError) {
          set({ localStorageError: null });
        }
      } catch (e) {
        // Handle QuotaExceededError specifically
        if (e instanceof DOMException && (e.name === 'QuotaExceededError' || e.code === 22)) {
          console.error('[localStorage] Quota exceeded. Session data not saved.');
          set({ localStorageError: 'Storage quota exceeded. Upgrade for cloud sync.' });
        } else {
          console.warn('Failed to save to localStorage:', e);
        }
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
    } catch (e) {
      console.warn('Failed to load from localStorage:', e);
      return false;
    }
  },

  // Sync actions
  syncNow: async () => {
    return syncService.syncNow();
  },

  updateSyncStatus: (status: SyncStatus) => {
    set({ syncStatus: status });
  },

  // User actions
  setCurrentUser: (userId: string, userName: string | null) => {
    set({ currentUserId: userId, currentUserName: userName });
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
      createdByUserId: state.currentUserId,
    };

    set({
      highlights: [...state.highlights, newHighlight],
      hasUnsavedChanges: true,
    });

    // Mark dirty for backend sync
    syncService.markDirty();

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

    // Mark dirty for backend sync
    syncService.markDirty();

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

    // Mark dirty for backend sync
    syncService.markDirty();

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

    // Mark dirty for backend sync
    syncService.markDirty();

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

    // Mark dirty for backend sync
    syncService.markDirty();

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

    // Mark dirty for backend sync
    syncService.markDirty();

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

    // Mark dirty for backend sync
    syncService.markDirty();

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

  canEditHighlight: (highlightId: string) => {
    const state = get();
    const highlight = state.highlights?.find((h) => h.id === highlightId);
    if (!highlight) return false;
    // If no creator ID set, anyone can edit (legacy highlights - log for tracking)
    if (!highlight.createdByUserId) {
      console.warn(`[canEditHighlight] Legacy highlight ${highlightId} has no createdByUserId`);
      return true;
    }
    // Only creator can edit
    return highlight.createdByUserId === state.currentUserId;
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

  // Confirmation actions
  setConfirmationStatus: (matchId: string, status: ConfirmationStatus | null) => {
    set((state) => ({
      confirmationStatus: {
        ...state.confirmationStatus,
        [matchId]: status,
      },
    }));
  },

  setIsConfirming: (isConfirming: boolean) => {
    set({ isConfirming });
  },

  updateConfirmationProgress: (matchId: string, progress: number) => {
    set((state) => {
      const existing = state.confirmationStatus[matchId];
      if (!existing) return state;
      return {
        confirmationStatus: {
          ...state.confirmationStatus,
          [matchId]: { ...existing, progress },
        },
      };
    });
  },

  // Check if rally editing is locked (video confirmed or user is guest)
  isRallyEditingLocked: () => {
    const state = get();
    // Guest users (members) cannot edit rallies
    if (state.userRole === 'member') return true;
    if (!state.activeMatchId) return false;
    const confirmation = state.confirmationStatus[state.activeMatchId];
    return confirmation?.status === 'CONFIRMED';
  },

  // Rally recording actions (M key hotkey)
  startRallyRecording: () => {
    const state = get();
    const playerState = usePlayerStore.getState();
    const currentTime = playerState.currentTime;

    // Don't allow recording if editing is locked
    if (state.isRallyEditingLocked()) return;

    // Check if inside existing rally - don't allow recording inside a rally
    const insideRally = state.rallies.some(
      (r) => currentTime >= r.start_time && currentTime <= r.end_time
    );
    if (insideRally) return;

    set({
      isRecordingRally: true,
      rallyRecordingStartTime: currentTime,
    });
  },

  stopRallyRecording: (endTime: number) => {
    const state = get();
    if (!state.isRecordingRally || state.rallyRecordingStartTime === null) return;

    let startTime = state.rallyRecordingStartTime;
    let finalEndTime = endTime;

    // Handle case where user rewound (swap times)
    if (startTime > finalEndTime) {
      [startTime, finalEndTime] = [finalEndTime, startTime];
    }

    // Enforce minimum duration (4s for recording mode)
    if (finalEndTime - startTime < 4) {
      finalEndTime = startTime + 4;
    }

    // Clamp to video duration
    const videoDuration = state.videoMetadata?.duration ?? Infinity;
    if (finalEndTime > videoDuration) {
      finalEndTime = videoDuration;
    }

    // Check for overlap with existing rallies
    const overlapping = state.rallies.filter(
      (r) => startTime < r.end_time && finalEndTime > r.start_time
    );

    if (overlapping.length > 0) {
      // Cancel recording if would overlap
      set({ isRecordingRally: false, rallyRecordingStartTime: null });
      return;
    }

    // Clear recording state first
    set({ isRecordingRally: false, rallyRecordingStartTime: null });

    // Create the rally using existing addRally action
    state.addRally(startTime, finalEndTime);
  },

  cancelRallyRecording: () => {
    set({
      isRecordingRally: false,
      rallyRecordingStartTime: null,
    });
  },
}));

// Subscribe to cameraStore changes to trigger localStorage persistence
// This ensures camera edits are saved alongside rally/highlight edits
useCameraStore.subscribe(
  (state) => state.cameraEdits,
  () => {
    // Only save if editorStore has a session loaded
    if (useEditorStore.getState().session) {
      debouncedSave(() => useEditorStore.getState().saveToStorage());
    }
  }
);
