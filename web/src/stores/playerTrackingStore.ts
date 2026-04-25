import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { trackPlayers, getPlayerTrack, swapPlayerTracks, promoteRawTrack, saveCourtCalibration, deleteCourtCalibration, getActionGroundTruth, saveActionGroundTruth as apiSaveActionGroundTruth, trackAllRallies as apiTrackAllRallies, getBatchTrackingStatus as apiGetBatchTrackingStatus, getPlayerReferenceCrops, uploadPlayerReferenceCrop, deletePlayerReferenceCrop, getVideoScoreGt, saveRallyScoreGt, getMatchAnalysis, type TrackPlayersResponse, type GetPlayerTrackResponse, type PlayerPosition as ApiPlayerPosition, type BallPosition, type ContactsData, type ActionsData, type ActionGroundTruthLabel, type QualityReport, type BatchTrackingStatus, type PlayerReferenceCrop, type ScoreGtEntry, type ScoreTeam, type MatchAnalysis } from '@/services/api';

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
  showLandingZones: boolean;

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

  // Per-video match analysis (cached so the editor can resolve display pid
  // from raw trackId via rallies[].appliedFullMapping).
  matchAnalysis: Record<string, MatchAnalysis>; // keyed by videoId
  matchAnalysisLoading: Record<string, boolean>; // keyed by videoId
  /** In-flight promise per videoId so concurrent callers await the same
   *  fetch instead of racing. Not persisted; lives only in memory. */
  _matchAnalysisPending?: Record<string, Promise<MatchAnalysis | null>>;

  // Reference crops state
  referenceCrops: PlayerReferenceCrop[];
  referenceCropsLoading: boolean;

  // Score ground truth (Session 5) — keyed by videoId
  scoreGt: Record<string, ScoreGtEntry[]>;
  scoreGtLoading: Record<string, boolean>;
  fetchScoreGt: (videoId: string) => Promise<void>;
  /**
   * Save the serving-team and point-winner GT for a single rally (optimistic).
   * Does **not** touch `gtSideSwitch` — use `setRallySideSwitch` for that.
   */
  saveScoreGt: (
    videoId: string,
    rallyId: string,
    body: { gtServingTeam: ScoreTeam; gtPointWinner: ScoreTeam },
  ) => Promise<void>;
  /**
   * Set (or clear) the manual side-switch override for a rally. Refetches
   * the whole video's score GT after the PUT so every downstream
   * `nearSideTeam` recomputes server-side.
   */
  setRallySideSwitch: (
    videoId: string,
    rallyId: string,
    gtSideSwitch: boolean | null,
  ) => Promise<void>;

  // Actions
  togglePlayerOverlay: () => void;
  toggleBallOverlay: () => void;
  toggleCourtDebugOverlay: () => void;
  toggleLandingZones: () => void;
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
  /** Update the player anchor for an existing GT label. `trackId` is the raw
   *  BoT-SORT id (stable anchor); storing as canonical pid is deprecated. */
  updateActionLabelPlayer: (rallyId: string, frame: number, trackId: number) => void;
  loadActionGroundTruth: (rallyId: string) => Promise<void>;
  saveActionGroundTruth: (rallyId: string) => Promise<void>;

  // Match analysis loader (backs appliedFullMapping / trackToPlayer lookups).
  loadMatchAnalysis: (videoId: string, forceRefresh?: boolean) => Promise<MatchAnalysis | null>;
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
      showLandingZones: false,
      selectedTrackId: null,
      isCalibrating: false,
      calibrations: {},
      batchTracking: {},
      batchTrackingPolling: {},
      isLabelingActions: false,
      actionGroundTruth: {},
      actionGtDirty: {},
      actionGtSaving: {},
      matchAnalysis: {},
      matchAnalysisLoading: {},
      referenceCrops: [],
      referenceCropsLoading: false,
      scoreGt: {},
      scoreGtLoading: {},

      fetchScoreGt: async (videoId: string) => {
        if (get().scoreGtLoading[videoId]) return;
        set((state) => ({
          scoreGtLoading: { ...state.scoreGtLoading, [videoId]: true },
        }));
        try {
          const result = await getVideoScoreGt(videoId);
          const sorted = [...result.rallies].sort(
            (a, b) => a.order - b.order || a.startMs - b.startMs,
          );
          set((state) => ({
            scoreGt: { ...state.scoreGt, [videoId]: sorted },
            scoreGtLoading: { ...state.scoreGtLoading, [videoId]: false },
          }));
        } catch (error) {
          console.error('[PlayerTrackingStore] Failed to load score GT:', error);
          set((state) => ({
            scoreGtLoading: { ...state.scoreGtLoading, [videoId]: false },
          }));
        }
      },

      setRallySideSwitch: async (videoId, rallyId, gtSideSwitch) => {
        const state = get();
        const prevList = state.scoreGt[videoId] ?? [];
        const prevEntry = prevList.find((r) => r.rallyId === rallyId);
        if (!prevEntry) return;
        // Optimistic update on just this rally; near-side recompute happens
        // after the refetch (server is the source of truth for the chain).
        const optimistic = prevList.map((r) =>
          r.rallyId === rallyId ? { ...r, gtSideSwitch } : r,
        );
        set((s) => ({ scoreGt: { ...s.scoreGt, [videoId]: optimistic } }));
        try {
          await saveRallyScoreGt(rallyId, {
            gtServingTeam: prevEntry.gtServingTeam,
            gtPointWinner: prevEntry.gtPointWinner,
            gtSideSwitch,
          });
          // Refetch so downstream nearSideTeam updates for every rally.
          const refreshed = await getVideoScoreGt(videoId);
          const sorted = [...refreshed.rallies].sort(
            (a, b) => a.order - b.order || a.startMs - b.startMs,
          );
          set((s) => ({ scoreGt: { ...s.scoreGt, [videoId]: sorted } }));
        } catch (error) {
          // Revert
          set((s) => {
            const list = s.scoreGt[videoId] ?? [];
            return {
              scoreGt: {
                ...s.scoreGt,
                [videoId]: list.map((r) => (r.rallyId === rallyId ? prevEntry : r)),
              },
            };
          });
          console.error('[PlayerTrackingStore] Failed to set side switch:', error);
          throw error;
        }
      },

      saveScoreGt: async (videoId, rallyId, body) => {
        const state = get();
        const prevList = state.scoreGt[videoId] ?? [];
        const prevEntry = prevList.find((r) => r.rallyId === rallyId);
        // Optimistic update
        const optimistic = prevList.map((r) =>
          r.rallyId === rallyId
            ? { ...r, gtServingTeam: body.gtServingTeam, gtPointWinner: body.gtPointWinner }
            : r,
        );
        set((s) => ({ scoreGt: { ...s.scoreGt, [videoId]: optimistic } }));
        try {
          const updated = await saveRallyScoreGt(rallyId, body);
          set((s) => {
            const list = s.scoreGt[videoId] ?? [];
            return {
              scoreGt: {
                ...s.scoreGt,
                [videoId]: list.map((r) => (r.rallyId === rallyId ? { ...r, ...updated } : r)),
              },
            };
          });
        } catch (error) {
          // Revert only if we had a prior entry to restore.
          if (prevEntry) {
            set((s) => {
              const list = s.scoreGt[videoId] ?? [];
              return {
                scoreGt: {
                  ...s.scoreGt,
                  [videoId]: list.map((r) => (r.rallyId === rallyId ? prevEntry : r)),
                },
              };
            });
          }
          console.error('[PlayerTrackingStore] Failed to save score GT:', error);
          throw error;
        }
      },

      togglePlayerOverlay: () => {
        set((state) => ({ showPlayerOverlay: !state.showPlayerOverlay }));
      },

      toggleBallOverlay: () => {
        set((state) => ({ showBallOverlay: !state.showBallOverlay }));
      },

      toggleCourtDebugOverlay: () => {
        set((state) => ({ showCourtDebugOverlay: !state.showCourtDebugOverlay }));
      },

      toggleLandingZones: () => {
        set((state) => ({ showLandingZones: !state.showLandingZones }));
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

      updateActionLabelPlayer: (rallyId: string, frame: number, trackId: number) => {
        set((state) => {
          const existing = state.actionGroundTruth[rallyId] ?? [];
          const updated = existing.map(l => {
            if (l.frame !== frame) return l;
            // Write the new stable anchor and drop the legacy field so older
            // canonical-pid values (from pre-migration rows) don't shadow it.
            const rest = { ...l };
            delete rest.playerTrackId;
            return { ...rest, trackId };
          });
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

      loadMatchAnalysis: async (videoId: string, forceRefresh: boolean = false) => {
        const state = get();
        if (!forceRefresh && state.matchAnalysis[videoId]) {
          return state.matchAnalysis[videoId];
        }
        // Dedupe concurrent callers: they all await the same in-flight promise,
        // so the second effect in the same tick doesn't re-issue the request
        // or fall into the sort-index fallback on first render.
        const pending = state._matchAnalysisPending?.[videoId];
        if (pending && !forceRefresh) return pending;

        const fetchP = (async () => {
          set((s) => ({
            matchAnalysisLoading: { ...s.matchAnalysisLoading, [videoId]: true },
          }));
          try {
            const analysis = await getMatchAnalysis(videoId);
            set((s) => ({
              matchAnalysis: analysis
                ? { ...s.matchAnalysis, [videoId]: analysis }
                : s.matchAnalysis,
              matchAnalysisLoading: { ...s.matchAnalysisLoading, [videoId]: false },
            }));
            return analysis;
          } catch (error) {
            console.error('[PlayerTrackingStore] Failed to load match analysis:', error);
            set((s) => ({
              matchAnalysisLoading: { ...s.matchAnalysisLoading, [videoId]: false },
            }));
            return null;
          } finally {
            set((s) => {
              const next = { ...(s._matchAnalysisPending ?? {}) };
              delete next[videoId];
              return { _matchAnalysisPending: next };
            });
          }
        })();
        set((s) => ({
          _matchAnalysisPending: {
            ...(s._matchAnalysisPending ?? {}),
            [videoId]: fetchP,
          },
        }));
        return fetchP;
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
        // The API endpoint already nulled canonicalPidMapJson and queued a
        // 'refCrop' pending edit; this nudges analysisStore so the existing
        // match-analysis debounce/rebuild path actually fires. Without this,
        // the queued edit sits until the user happens to also edit a rally.
        const { useAnalysisStore } = await import('@/stores/analysisStore');
        useAnalysisStore.getState().notifyRefCropEdited(videoId);
        return crop;
      },

      removeReferenceCrop: async (videoId, cropId) => {
        await deletePlayerReferenceCrop(videoId, cropId);
        set((state) => ({
          referenceCrops: state.referenceCrops.filter((c) => c.id !== cropId),
        }));
        const { useAnalysisStore } = await import('@/stores/analysisStore');
        useAnalysisStore.getState().notifyRefCropEdited(videoId);
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
