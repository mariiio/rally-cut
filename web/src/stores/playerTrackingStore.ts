import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { trackPlayers, getPlayerTrack, swapPlayerTracks, promoteRawTrack, saveCourtCalibration, deleteCourtCalibration, getActionGroundTruth, saveActionGroundTruth as apiSaveActionGroundTruth, trackAllRallies as apiTrackAllRallies, getBatchTrackingStatus as apiGetBatchTrackingStatus, getPlayerReferenceCrops, uploadPlayerReferenceCrop, deletePlayerReferenceCrop, type TrackPlayersResponse, type GetPlayerTrackResponse, type PlayerPosition as ApiPlayerPosition, type BallPosition, type ContactsData, type ActionsData, type ActionGroundTruthLabel, type QualityReport, type BatchTrackingStatus, type PlayerReferenceCrop } from '@/services/api';

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
  // Raw (non-primary) tracks for overlay & promote-to-primary
  rawTracks?: PlayerTrackData[];
  // Ball positions for trajectory overlay
  ballPositions?: BallPosition[];
  // Contact detection and action classification
  contacts?: ContactsData;
  actions?: ActionsData;
  qualityReport?: QualityReport;
  // Court split Y for team zone visualization
  courtSplitY?: number;
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
  showCourtDebugOverlay: boolean;
  showRawTracks: boolean;

  // Selected track for highlighting
  selectedTrackId: number | null;

  // Calibration state
  isCalibrating: boolean;
  calibrations: Record<string, CourtCalibration>; // keyed by videoId

  // Batch tracking state
  batchTracking: Record<string, BatchTrackingStatus>; // keyed by videoId
  batchTrackingPolling: Record<string, boolean>; // keyed by videoId

  // Action labeling state
  isLabelingActions: boolean;
  actionGroundTruth: Record<string, ActionGroundTruthLabel[]>; // keyed by rallyId
  actionGtDirty: Record<string, boolean>; // keyed by rallyId
  actionGtSaving: Record<string, boolean>; // keyed by rallyId

  // Reference crops state
  referenceCrops: PlayerReferenceCrop[];
  referenceCropsLoading: boolean;

  // Actions
  togglePlayerOverlay: () => void;
  toggleBallOverlay: () => void;
  toggleCourtDebugOverlay: () => void;
  toggleRawTracks: () => void;
  setSelectedTrack: (trackId: number | null) => void;
  setIsCalibrating: (value: boolean) => void;
  saveCalibration: (videoId: string, corners: Corner[]) => void;
  clearCalibration: (videoId: string) => void;
  clearLocalCalibration: (videoId: string) => void;
  getCalibration: (videoId: string) => CourtCalibration | null;
  hydrateCalibration: (videoId: string, corners: Corner[]) => void;
  hydrateFromAutoSave: (videoId: string, corners: Corner[]) => void;
  trackPlayersForRally: (rallyId: string, videoId: string, fallbackFps?: number) => Promise<void>;
  loadPlayerTrack: (rallyId: string, fallbackFps?: number, forceRefresh?: boolean) => Promise<boolean>;
  reindexTrack: (rallyId: string, oldStartTime: number, newStartTime: number, newEndTime: number) => void;
  swapTracks: (rallyId: string, trackA: number, trackB: number, fromFrame: number, fallbackFps?: number) => Promise<void>;
  promoteTracks: (rallyId: string, demoteTrackId: number, promoteTrackId: number, fromFrame: number, fallbackFps?: number) => Promise<void>;

  // Batch tracking actions
  trackAllRalliesForVideo: (videoId: string) => Promise<void>;
  pollBatchTrackingStatus: (videoId: string, fallbackFps?: number) => void;
  stopPollingBatchTracking: (videoId: string) => void;

  // Reference crop actions
  loadReferenceCrops: (videoId: string) => Promise<void>;
  addReferenceCrop: (videoId: string, playerId: number, frameMs: number, bbox: { x: number; y: number; w: number; h: number }, imageData: string) => Promise<PlayerReferenceCrop>;
  removeReferenceCrop: (videoId: string, cropId: string) => Promise<void>;

  // Action labeling actions
  setIsLabelingActions: (value: boolean) => void;
  addActionLabel: (rallyId: string, label: ActionGroundTruthLabel) => void;
  removeActionLabel: (rallyId: string, frame: number) => void;
  updateActionLabel: (rallyId: string, frame: number, action: ActionGroundTruthLabel['action']) => void;
  updateActionLabelPlayer: (rallyId: string, frame: number, playerTrackId: number) => void;
  loadActionGroundTruth: (rallyId: string) => Promise<void>;
  saveActionGroundTruth: (rallyId: string) => Promise<void>;
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

  // Convert raw (non-primary) positions to tracks format
  const rawTracks = response.rawPositions ? positionsToTracks(response.rawPositions) : undefined;

  return {
    id: `track-${rallyId}`,
    rallyId,
    status: 'COMPLETED',
    tracksJson: {
      fps,
      frameCount: response.frameCount || 0,
      tracks,
      rawTracks: rawTracks && rawTracks.length > 0 ? rawTracks : undefined,
      ballPositions: response.ballPositions,
      contacts: response.contacts,
      actions: response.actions,
      qualityReport: response.qualityReport,
      courtSplitY: response.courtSplitY,
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
      showCourtDebugOverlay: false,
      showRawTracks: false,
      selectedTrackId: null,
      isCalibrating: false,
      calibrations: {},
      batchTracking: {},
      batchTrackingPolling: {},
      isLabelingActions: false,
      actionGroundTruth: {},
      actionGtDirty: {},
      actionGtSaving: {},
      referenceCrops: [],
      referenceCropsLoading: false,

      togglePlayerOverlay: () => {
        set((state) => ({ showPlayerOverlay: !state.showPlayerOverlay }));
      },

      toggleBallOverlay: () => {
        set((state) => ({ showBallOverlay: !state.showBallOverlay }));
      },

      toggleCourtDebugOverlay: () => {
        set((state) => ({ showCourtDebugOverlay: !state.showCourtDebugOverlay }));
      },

      toggleRawTracks: () => {
        set((state) => ({ showRawTracks: !state.showRawTracks }));
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
        // Fire-and-forget: persist to backend
        saveCourtCalibration(videoId, corners).catch((err) => {
          console.error('[PlayerTrackingStore] Failed to save calibration to API:', err);
        });
      },

      clearCalibration: (videoId: string) => {
        set((state) => {
          const { [videoId]: _, ...rest } = state.calibrations;
          return { calibrations: rest };
        });
        // Fire-and-forget: clear from backend
        deleteCourtCalibration(videoId).catch((err) => {
          console.error('[PlayerTrackingStore] Failed to delete calibration from API:', err);
        });
      },

      clearLocalCalibration: (videoId: string) => {
        // Clear stale localStorage cache only — does NOT delete from backend.
        // Used when DB returns null calibration to evict outdated local entries.
        if (!get().calibrations[videoId]) return;
        set((state) => {
          const { [videoId]: _, ...rest } = state.calibrations;
          return { calibrations: rest };
        });
      },

      getCalibration: (videoId: string) => {
        return get().calibrations[videoId] || null;
      },

      hydrateCalibration: (videoId: string, corners: Corner[]) => {
        // Only hydrate if not already in local store (local is authoritative cache)
        if (get().calibrations[videoId]) return;
        set((state) => ({
          calibrations: {
            ...state.calibrations,
            [videoId]: {
              videoId,
              corners,
              savedAt: Date.now(),
            },
          },
        }));
      },

      hydrateFromAutoSave: (videoId: string, corners: Corner[]) => {
        // Always overwrite — auto-saved calibration from quality check
        set((state) => ({
          calibrations: {
            ...state.calibrations,
            [videoId]: {
              videoId,
              corners,
              savedAt: Date.now(),
            },
          },
        }));
      },

      /**
       * Load existing player tracking data for a rally.
       * Returns true if data was found and loaded.
       */
      loadPlayerTrack: async (rallyId: string, fallbackFps: number = 30, forceRefresh: boolean = false): Promise<boolean> => {
        // Skip if already loaded or currently loading
        const state = get();
        if (!forceRefresh && state.playerTracks[rallyId]?.status === 'COMPLETED') {
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
            set((state) => {
              const { [rallyId]: _, ...restTracks } = state.playerTracks;
              return {
                playerTracks: restTracks,
                isLoadingTrack: { ...state.isLoadingTrack, [rallyId]: false },
              };
            });
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

      // Shift frame numbers in cached tracking data so overlays stay aligned
      // when rally boundaries change. Times are in seconds (matching Rally.start_time).
      reindexTrack: (rallyId: string, oldStartTime: number, newStartTime: number, newEndTime: number) => {
        const track = get().playerTracks[rallyId];
        if (!track?.tracksJson) return;

        const { tracksJson } = track;
        const fps = tracksJson.fps || 30;
        const deltaFrames = Math.round((newStartTime - oldStartTime) * fps);
        const newFrameCount = Math.round((newEndTime - newStartTime) * fps);

        if (deltaFrames === 0 && newFrameCount === tracksJson.frameCount) return;

        const shiftPlayerPositions = (positions: PlayerPosition[]) =>
          positions
            .map((p) => ({ ...p, frame: p.frame - deltaFrames }))
            .filter((p) => p.frame >= 0 && p.frame < newFrameCount);

        const reindexedTracks = tracksJson.tracks.map((t) => ({
          ...t,
          positions: shiftPlayerPositions(t.positions),
        }));

        const reindexedRawTracks = tracksJson.rawTracks?.map((t) => ({
          ...t,
          positions: shiftPlayerPositions(t.positions),
        }));

        const reindexedBallPositions = tracksJson.ballPositions
          ?.map((p) => ({ ...p, frameNumber: p.frameNumber - deltaFrames }))
          .filter((p) => p.frameNumber >= 0 && p.frameNumber < newFrameCount);

        let reindexedContacts = tracksJson.contacts;
        if (reindexedContacts) {
          const shiftedContacts = reindexedContacts.contacts
            .map((c) => ({ ...c, frame: c.frame - deltaFrames }))
            .filter((c) => c.frame >= 0 && c.frame < newFrameCount);
          reindexedContacts = {
            ...reindexedContacts,
            rallyStartFrame: reindexedContacts.rallyStartFrame - deltaFrames,
            numContacts: shiftedContacts.length,
            contacts: shiftedContacts,
          };
        }

        let reindexedActions = tracksJson.actions;
        if (reindexedActions) {
          const shiftedActions = reindexedActions.actions
            .map((a) => ({ ...a, frame: a.frame - deltaFrames }))
            .filter((a) => a.frame >= 0 && a.frame < newFrameCount);
          reindexedActions = {
            ...reindexedActions,
            numContacts: shiftedActions.length,
            actionSequence: shiftedActions.map((a) => a.action),
            actions: shiftedActions,
          };
        }

        set((state) => ({
          playerTracks: {
            ...state.playerTracks,
            [rallyId]: {
              ...track,
              tracksJson: {
                ...tracksJson,
                frameCount: newFrameCount,
                tracks: reindexedTracks,
                rawTracks: reindexedRawTracks,
                ballPositions: reindexedBallPositions,
                contacts: reindexedContacts,
                actions: reindexedActions,
              },
            },
          },
        }));
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

      swapTracks: async (rallyId: string, trackA: number, trackB: number, fromFrame: number, fallbackFps: number = 30) => {
        try {
          await swapPlayerTracks(rallyId, trackA, trackB, fromFrame);
          // Re-fetch from server (old data stays visible until new data arrives)
          await get().loadPlayerTrack(rallyId, fallbackFps, true);
        } catch (error) {
          console.error('[PlayerTrackingStore] Failed to swap tracks:', error);
          throw error;
        }
      },

      promoteTracks: async (rallyId: string, demoteTrackId: number, promoteTrackId: number, fromFrame: number, fallbackFps: number = 30) => {
        try {
          await promoteRawTrack(rallyId, demoteTrackId, promoteTrackId, fromFrame);
          // Re-fetch from server to refresh both primary and raw tracks
          await get().loadPlayerTrack(rallyId, fallbackFps, true);
        } catch (error) {
          console.error('[PlayerTrackingStore] Failed to promote raw track:', error);
          throw error;
        }
      },

      // Batch tracking
      trackAllRalliesForVideo: async (videoId: string) => {
        try {
          const result = await apiTrackAllRallies(videoId);
          set((state) => ({
            batchTracking: {
              ...state.batchTracking,
              [videoId]: {
                status: 'pending',
                jobId: result.jobId,
                totalRallies: result.totalRallies,
                completedRallies: 0,
                failedRallies: 0,
              },
            },
          }));
          // Auto-start polling
          get().pollBatchTrackingStatus(videoId);
        } catch (error) {
          console.error('[PlayerTrackingStore] Failed to start batch tracking:', error);
          set((state) => ({
            batchTracking: {
              ...state.batchTracking,
              [videoId]: {
                status: 'failed',
                error: error instanceof Error ? error.message : 'Failed to start',
              },
            },
          }));
        }
      },

      pollBatchTrackingStatus: (videoId: string, fallbackFps: number = 30) => {
        // Don't start duplicate polls
        if (get().batchTrackingPolling[videoId]) return;

        set((state) => ({
          batchTrackingPolling: { ...state.batchTrackingPolling, [videoId]: true },
        }));

        const poll = async () => {
          if (!get().batchTrackingPolling[videoId]) return;

          try {
            const status = await apiGetBatchTrackingStatus(videoId);
            const prevStatus = get().batchTracking[videoId];

            set((state) => ({
              batchTracking: { ...state.batchTracking, [videoId]: status },
            }));

            // Auto-load completed rally tracks
            if (status.rallyStatuses) {
              const prevCompleted = new Set(
                prevStatus?.rallyStatuses
                  ?.filter(r => r.status === 'COMPLETED')
                  .map(r => r.rallyId) ?? []
              );

              for (const rs of status.rallyStatuses) {
                if (rs.status === 'COMPLETED' && !prevCompleted.has(rs.rallyId)) {
                  // New completion — load the track data
                  get().loadPlayerTrack(rs.rallyId, fallbackFps, true);
                }
              }
            }

            // Continue polling if still active
            if (status.status === 'pending' || status.status === 'processing') {
              setTimeout(poll, 2000);
            } else {
              // Done — stop polling, auto-show overlays
              set((state) => ({
                batchTrackingPolling: { ...state.batchTrackingPolling, [videoId]: false },
                showPlayerOverlay: true,
                showBallOverlay: true,
              }));
            }
          } catch (error) {
            console.error('[PlayerTrackingStore] Batch status poll failed:', error);
            if (!get().batchTrackingPolling[videoId]) return;
            setTimeout(poll, 5000); // Retry after longer delay
          }
        };

        poll();
      },

      stopPollingBatchTracking: (videoId: string) => {
        set((state) => ({
          batchTrackingPolling: { ...state.batchTrackingPolling, [videoId]: false },
        }));
      },

      // Action labeling
      setIsLabelingActions: (value: boolean) => {
        set({ isLabelingActions: value });
      },

      addActionLabel: (rallyId: string, label: ActionGroundTruthLabel) => {
        set((state) => {
          const existing = state.actionGroundTruth[rallyId] ?? [];
          // Replace if same frame exists, otherwise add
          const filtered = existing.filter(l => l.frame !== label.frame);
          const updated = [...filtered, label].sort((a, b) => a.frame - b.frame);
          return {
            actionGroundTruth: { ...state.actionGroundTruth, [rallyId]: updated },
            actionGtDirty: { ...state.actionGtDirty, [rallyId]: true },
          };
        });
      },

      removeActionLabel: (rallyId: string, frame: number) => {
        set((state) => {
          const existing = state.actionGroundTruth[rallyId] ?? [];
          const updated = existing.filter(l => l.frame !== frame);
          return {
            actionGroundTruth: { ...state.actionGroundTruth, [rallyId]: updated },
            actionGtDirty: { ...state.actionGtDirty, [rallyId]: true },
          };
        });
      },

      updateActionLabel: (rallyId: string, frame: number, action: ActionGroundTruthLabel['action']) => {
        set((state) => {
          const existing = state.actionGroundTruth[rallyId] ?? [];
          const updated = existing.map(l => l.frame === frame ? { ...l, action } : l);
          return {
            actionGroundTruth: { ...state.actionGroundTruth, [rallyId]: updated },
            actionGtDirty: { ...state.actionGtDirty, [rallyId]: true },
          };
        });
      },

      updateActionLabelPlayer: (rallyId: string, frame: number, playerTrackId: number) => {
        set((state) => {
          const existing = state.actionGroundTruth[rallyId] ?? [];
          const updated = existing.map(l => l.frame === frame ? { ...l, playerTrackId } : l);
          return {
            actionGroundTruth: { ...state.actionGroundTruth, [rallyId]: updated },
            actionGtDirty: { ...state.actionGtDirty, [rallyId]: true },
          };
        });
      },

      loadActionGroundTruth: async (rallyId: string) => {
        try {
          const result = await getActionGroundTruth(rallyId);
          set((state) => ({
            actionGroundTruth: { ...state.actionGroundTruth, [rallyId]: result.labels },
            actionGtDirty: { ...state.actionGtDirty, [rallyId]: false },
          }));
        } catch (error) {
          console.error('[PlayerTrackingStore] Failed to load action GT:', error);
        }
      },

      saveActionGroundTruth: async (rallyId: string) => {
        const state = get();
        const labels = state.actionGroundTruth[rallyId] ?? [];

        set((s) => ({ actionGtSaving: { ...s.actionGtSaving, [rallyId]: true } }));
        try {
          await apiSaveActionGroundTruth(rallyId, labels);
          set((s) => ({
            actionGtDirty: { ...s.actionGtDirty, [rallyId]: false },
            actionGtSaving: { ...s.actionGtSaving, [rallyId]: false },
          }));
        } catch (error) {
          console.error('[PlayerTrackingStore] Failed to save action GT:', error);
          set((s) => ({ actionGtSaving: { ...s.actionGtSaving, [rallyId]: false } }));
          throw error;
        }
      },

      loadReferenceCrops: async (videoId: string) => {
        set({ referenceCropsLoading: true });
        try {
          const crops = await getPlayerReferenceCrops(videoId);
          set({ referenceCrops: crops, referenceCropsLoading: false });
        } catch (error) {
          console.error('[PlayerTrackingStore] Failed to load reference crops:', error);
          set({ referenceCropsLoading: false });
        }
      },

      addReferenceCrop: async (videoId, playerId, frameMs, bbox, imageData) => {
        const crop = await uploadPlayerReferenceCrop(videoId, { playerId, frameMs, bbox, imageData });
        set((state) => ({
          referenceCrops: [...state.referenceCrops, crop],
        }));
        return crop;
      },

      removeReferenceCrop: async (videoId, cropId) => {
        await deletePlayerReferenceCrop(videoId, cropId);
        set((state) => ({
          referenceCrops: state.referenceCrops.filter((c) => c.id !== cropId),
        }));
      },
    }),
    {
      name: 'player-tracking-storage',
      partialize: (state) => ({
        // Only persist calibrations and overlay preferences, not transient tracking data
        calibrations: state.calibrations,
        showRawTracks: state.showRawTracks,
      }),
    }
  )
);
