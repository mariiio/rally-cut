import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { trackPlayers, getPlayerTrack, type TrackPlayersResponse, type GetPlayerTrackResponse, type PlayerPosition as ApiPlayerPosition, type BallPosition, type ContactsData, type ActionsData } from '@/services/api';

// Types for player tracking data (store format)
export interface PlayerPosition {
  frame: number;
  x: number;
  y: number;
  w: number;
  h: number;
  confidence: number;
  courtX?: number;
  courtY?: number;
}

export interface PlayerTrackData {
  trackId: number;
  positions: PlayerPosition[];
}

export interface TracksJson {
  fps: number;
  frameCount: number;
  tracks: PlayerTrackData[];
  // Ball positions for trajectory overlay
  ballPositions?: BallPosition[];
  // Contact detection and action classification
  contacts?: ContactsData;
  actions?: ActionsData;
}

export interface PlayerTrack {
  id: string;
  rallyId: string;
  status: 'PENDING' | 'PROCESSING' | 'COMPLETED' | 'FAILED';
  tracksJson: TracksJson | null;
  playerCount: number | null;
  progress: number | null;
  progressMessage: string | null;
}

// Court calibration types
export interface Corner {
  x: number;
  y: number;
}

export interface CourtCalibration {
  videoId: string;
  corners: Corner[];
  savedAt: number;
}

interface PlayerTrackingState {
  // Player tracking state per rally
  playerTracks: Record<string, PlayerTrack>;
  isTracking: Record<string, boolean>; // keyed by rallyId
  isLoadingTrack: Record<string, boolean>; // keyed by rallyId
  trackingErrors: Record<string, string>; // keyed by rallyId

  // Overlay visibility
  showPlayerOverlay: boolean;
  showBallOverlay: boolean;

  // Selected track for highlighting
  selectedTrackId: number | null;

  // Calibration state
  isCalibrating: boolean;
  calibrations: Record<string, CourtCalibration>; // keyed by videoId

  // Actions
  togglePlayerOverlay: () => void;
  toggleBallOverlay: () => void;
  setSelectedTrack: (trackId: number | null) => void;
  setIsCalibrating: (value: boolean) => void;
  saveCalibration: (videoId: string, corners: Corner[]) => void;
  getCalibration: (videoId: string) => CourtCalibration | null;
  trackPlayersForRally: (rallyId: string, videoId: string, fallbackFps?: number) => Promise<void>;
  loadPlayerTrack: (rallyId: string, fallbackFps?: number) => Promise<boolean>;
}

/**
 * Convert flat API positions array to grouped tracks format for overlay rendering.
 */
function positionsToTracks(positions: ApiPlayerPosition[]): PlayerTrackData[] {
  // Group positions by trackId
  const trackMap = new Map<number, PlayerPosition[]>();

  for (const p of positions) {
    const trackId = p.trackId;
    if (!trackMap.has(trackId)) {
      trackMap.set(trackId, []);
    }
    trackMap.get(trackId)!.push({
      frame: p.frameNumber,
      x: p.x,
      y: p.y,
      w: p.width,
      h: p.height,
      confidence: p.confidence,
    });
  }

  // Convert to array and sort positions within each track
  const tracks: PlayerTrackData[] = [];
  for (const [trackId, trackPositions] of trackMap) {
    tracks.push({
      trackId,
      positions: trackPositions.sort((a, b) => a.frame - b.frame),
    });
  }

  // Sort tracks by ID for consistent rendering
  return tracks.sort((a, b) => a.trackId - b.trackId);
}

// Convert API response to store format
function apiResponseToPlayerTrack(
  rallyId: string,
  response: TrackPlayersResponse | GetPlayerTrackResponse,
  fallbackFps: number = 30
): PlayerTrack {
  if (response.status !== 'completed') {
    return {
      id: `track-${rallyId}`,
      rallyId,
      status: 'FAILED',
      tracksJson: null,
      playerCount: null,
      progress: null,
      progressMessage: response.error || 'Tracking failed',
    };
  }

  // Convert positions to tracks format
  let tracks = response.positions ? positionsToTracks(response.positions) : [];

  // Filter to only primary tracks if available (excludes referees, spectators)
  if (response.primaryTrackIds && response.primaryTrackIds.length > 0) {
    const primarySet = new Set(response.primaryTrackIds);
    tracks = tracks.filter(t => primarySet.has(t.trackId));
  }

  // Use fps from response (actual tracked video fps), fallback to parameter
  const fps = response.fps ?? fallbackFps;

  return {
    id: `track-${rallyId}`,
    rallyId,
    status: 'COMPLETED',
    tracksJson: {
      fps,
      frameCount: response.frameCount || 0,
      tracks,
      ballPositions: response.ballPositions,
      contacts: response.contacts,
      actions: response.actions,
    },
    playerCount: response.uniqueTrackCount || 0,
    progress: 100,
    progressMessage: null,
  };
}

export const usePlayerTrackingStore = create<PlayerTrackingState>()(
  persist(
    (set, get) => ({
      playerTracks: {},
      isTracking: {},
      isLoadingTrack: {},
      trackingErrors: {},
      showPlayerOverlay: false,
      showBallOverlay: false,
      selectedTrackId: null,
      isCalibrating: false,
      calibrations: {},

      togglePlayerOverlay: () => {
        set((state) => ({ showPlayerOverlay: !state.showPlayerOverlay }));
      },

      toggleBallOverlay: () => {
        set((state) => ({ showBallOverlay: !state.showBallOverlay }));
      },

      setSelectedTrack: (trackId: number | null) => {
        set({ selectedTrackId: trackId });
      },

      setIsCalibrating: (value: boolean) => {
        set({ isCalibrating: value });
      },

      saveCalibration: (videoId: string, corners: Corner[]) => {
        set((state) => ({
          calibrations: {
            ...state.calibrations,
            [videoId]: {
              videoId,
              corners,
              savedAt: Date.now(),
            },
          },
          isCalibrating: false,
        }));
      },

      getCalibration: (videoId: string) => {
        return get().calibrations[videoId] || null;
      },

      /**
       * Load existing player tracking data for a rally.
       * Returns true if data was found and loaded.
       */
      loadPlayerTrack: async (rallyId: string, fallbackFps: number = 30): Promise<boolean> => {
        // Skip if already loaded or currently loading
        const state = get();
        if (state.playerTracks[rallyId]?.status === 'COMPLETED') {
          return true;
        }
        if (state.isLoadingTrack[rallyId]) {
          return false;
        }

        // Set loading state
        set((state) => ({
          isLoadingTrack: { ...state.isLoadingTrack, [rallyId]: true },
        }));

        try {
          const response = await getPlayerTrack(rallyId);

          if (response.status === 'not_found') {
            set((state) => ({
              isLoadingTrack: { ...state.isLoadingTrack, [rallyId]: false },
            }));
            return false;
          }

          const playerTrack = apiResponseToPlayerTrack(rallyId, response, fallbackFps);

          set((state) => ({
            playerTracks: { ...state.playerTracks, [rallyId]: playerTrack },
            isLoadingTrack: { ...state.isLoadingTrack, [rallyId]: false },
          }));

          return response.status === 'completed';
        } catch (error) {
          console.error('[PlayerTrackingStore] Failed to load player track:', error);
          set((state) => ({
            isLoadingTrack: { ...state.isLoadingTrack, [rallyId]: false },
          }));
          return false;
        }
      },

      trackPlayersForRally: async (rallyId: string, videoId: string, fallbackFps: number = 30) => {
        // Set loading state
        set((state) => ({
          isTracking: { ...state.isTracking, [rallyId]: true },
          trackingErrors: { ...state.trackingErrors, [rallyId]: '' },
        }));

        try {
          // Get calibration for this video if available
          const calibration = get().calibrations[videoId];
          const calibrationCorners = calibration?.corners;

          const response = await trackPlayers(rallyId, calibrationCorners);
          const playerTrack = apiResponseToPlayerTrack(rallyId, response, fallbackFps);

          set((state) => ({
            playerTracks: { ...state.playerTracks, [rallyId]: playerTrack },
            isTracking: { ...state.isTracking, [rallyId]: false },
            showPlayerOverlay: true, // Auto-show overlay after tracking
            showBallOverlay: !!playerTrack.tracksJson?.ballPositions?.length, // Auto-show ball overlay if positions available
          }));
        } catch (error) {
          set((state) => ({
            isTracking: { ...state.isTracking, [rallyId]: false },
            trackingErrors: {
              ...state.trackingErrors,
              [rallyId]: error instanceof Error ? error.message : 'Tracking failed',
            },
          }));
        }
      },
    }),
    {
      name: 'player-tracking-storage',
      partialize: (state) => ({
        // Only persist calibrations, not transient tracking data
        calibrations: state.calibrations,
      }),
    }
  )
);
