'use client';

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import {
  assessQuality,
  getAnalysisPipelineStatus,
  triggerRallyDetection,
  getDetectionStatus,
  trackAllRallies,
  getBatchTrackingStatus,
  getMatchStatsApi,
  type QualityAssessmentResult,
} from '@/services/api';

export type AnalysisPhase =
  | 'idle'
  | 'quality_check'
  | 'quality_warning'
  | 'detecting'
  | 'tracking'
  | 'completing'
  | 'done'
  | 'error';

export interface AnalysisPipeline {
  phase: AnalysisPhase;
  progress: number;
  stepMessage: string;
  qualityResult?: QualityAssessmentResult;
  ralliesFound?: number;
  trackingProgress?: { completed: number; total: number };
  playerCount?: number;
  error?: string;
  startedAt?: number;
}

const DEFAULT_PIPELINE: AnalysisPipeline = {
  phase: 'idle',
  progress: 0,
  stepMessage: '',
};

interface AnalysisState {
  pipelines: Record<string, AnalysisPipeline>;

  // Show player naming dialog
  showPlayerNaming: string | null; // videoId or null

  // Actions
  getPipeline: (videoId: string) => AnalysisPipeline;
  startAnalysis: (videoId: string) => Promise<void>;
  dismissWarnings: (videoId: string) => void;
  cancelAnalysis: (videoId: string) => void;
  resumeIfNeeded: (videoId: string) => Promise<void>;
  setShowPlayerNaming: (videoId: string | null) => void;
}

// Polling intervals stored outside React state.
// On HMR re-evaluation, the previous module's timers become orphaned.
// We store a reference on globalThis so we can clean up stale timers.
const HMR_KEY = '__rallycut_analysis_poll_timers__' as const;

// Clean up any orphaned timers from previous HMR module evaluation
const staleTimers = (globalThis as Record<string, unknown>)[HMR_KEY] as Record<string, ReturnType<typeof setInterval>> | undefined;
if (staleTimers) {
  Object.values(staleTimers).forEach(clearInterval);
}

const pollTimers: Record<string, ReturnType<typeof setInterval>> = {};
(globalThis as Record<string, unknown>)[HMR_KEY] = pollTimers;

// Guard against concurrent completeAnalysis calls
const completingLock = new Set<string>();

// Stale-poll escape hatch: if N consecutive polls return an unrecognized status,
// assume the backend job is gone and surface an error instead of looping forever.
const stalePollCounts = new Map<string, number>();
const MAX_STALE_POLLS = 6; // 6 × 10s = 60s
const STALE_TIMEOUT_MS = 30 * 60 * 1000; // 30 min absolute timeout

function clearPollTimer(videoId: string) {
  if (pollTimers[videoId]) {
    clearInterval(pollTimers[videoId]);
    delete pollTimers[videoId];
  }
  stalePollCounts.delete(videoId);
}

export const useAnalysisStore = create<AnalysisState>()(
  persist(
    (set, get) => ({
      pipelines: {},
      showPlayerNaming: null,

      getPipeline: (videoId: string) => {
        return get().pipelines[videoId] ?? DEFAULT_PIPELINE;
      },

      startAnalysis: async (videoId: string) => {
        // Prevent re-entrance: skip if already running for this video
        const existing = get().pipelines[videoId];
        if (existing && existing.phase !== 'idle' && existing.phase !== 'done' && existing.phase !== 'error') {
          return;
        }

        const update = (patch: Partial<AnalysisPipeline>) => {
          set((state) => ({
            pipelines: {
              ...state.pipelines,
              [videoId]: { ...DEFAULT_PIPELINE, ...patch },
            },
          }));
        };

        // Initialize pipeline
        update({
          phase: 'quality_check',
          progress: 2,
          stepMessage: 'Checking your video...',
          startedAt: Date.now(),
          error: undefined,
        });

        // Step 1: Quality check
        try {
          const qualityResult = await assessQuality(videoId);
          const hasWarnings = qualityResult.quality.warnings.length > 0;

          if (hasWarnings) {
            update({
              phase: 'quality_warning',
              progress: 5,
              stepMessage: 'We found some potential issues',
              qualityResult,
            });
            // Wait for user to dismiss warnings (via dismissWarnings action)
            return;
          }

          // No warnings — proceed directly to detection
          update({
            phase: 'detecting',
            progress: 8,
            stepMessage: 'Looking for rallies...',
            qualityResult,
          });
          await startDetection(videoId, set, get);
        } catch (err) {
          // Quality check failed — proceed to detection anyway (non-critical)
          console.warn('[ANALYSIS] Quality check failed, continuing:', err);
          update({
            phase: 'detecting',
            progress: 8,
            stepMessage: 'Looking for rallies...',
          });
          await startDetection(videoId, set, get);
        }
      },

      dismissWarnings: (videoId: string) => {
        const pipeline = get().pipelines[videoId];
        if (!pipeline || pipeline.phase !== 'quality_warning') return;

        set((state) => ({
          pipelines: {
            ...state.pipelines,
            [videoId]: {
              ...pipeline,
              phase: 'detecting',
              progress: 8,
              stepMessage: 'Looking for rallies...',
            },
          },
        }));

        startDetection(videoId, set, get);
      },

      cancelAnalysis: (videoId: string) => {
        clearPollTimer(videoId);
        set((state) => ({
          pipelines: {
            ...state.pipelines,
            [videoId]: DEFAULT_PIPELINE,
          },
        }));
      },

      resumeIfNeeded: async (videoId: string) => {
        const pipeline = get().pipelines[videoId];
        if (!pipeline) return;

        const updatePipeline = (patch: Partial<AnalysisPipeline>) => {
          set((state) => ({
            pipelines: {
              ...state.pipelines,
              [videoId]: { ...(state.pipelines[videoId] ?? DEFAULT_PIPELINE), ...patch },
            },
          }));
        };

        // Absolute staleness check: if pipeline started > 30 min ago, error out
        if (pipeline.startedAt && Date.now() - pipeline.startedAt > STALE_TIMEOUT_MS) {
          clearPollTimer(videoId);
          updatePipeline({
            phase: 'error',
            error: 'Analysis timed out. Please try again.',
            stepMessage: 'Analysis timed out',
          });
          return;
        }

        if (pipeline.phase === 'detecting' && !pollTimers[videoId]) {
          // Validate server state before blindly polling
          try {
            const status = await getDetectionStatus(videoId);
            if (status.job?.status === 'COMPLETED') {
              // Detection already finished — advance to tracking
              const pipelineStatus = await getAnalysisPipelineStatus(videoId);
              const ralliesFound = pipelineStatus.detection.ralliesFound;
              if (ralliesFound === 0) {
                updatePipeline({ phase: 'done', progress: 100, stepMessage: 'No rallies found in this video', ralliesFound: 0 });
              } else {
                updatePipeline({ phase: 'tracking', progress: 45, stepMessage: `Found ${ralliesFound} rallies! Starting player tracking...`, ralliesFound });
                await startTracking(videoId, set, get);
              }
              return;
            } else if (status.job?.status === 'FAILED') {
              updatePipeline({ phase: 'error', error: status.job.errorMessage || 'Detection failed', stepMessage: 'Detection failed' });
              return;
            } else if (status.job?.status === 'RUNNING') {
              // Legitimate in-progress job — resume polling
              pollDetection(videoId, set, get);
            } else {
              // No job or unrecognized status — detection is not running
              stalePollCounts.delete(videoId);
              updatePipeline({ phase: 'error', error: 'Detection is no longer running. Please try again.', stepMessage: 'Detection not found' });
            }
          } catch {
            // API call failed — start polling, stale counter will catch permanent failures
            pollDetection(videoId, set, get);
          }
        } else if (pipeline.phase === 'tracking' && !pollTimers[videoId]) {
          try {
            const status = await getBatchTrackingStatus(videoId);
            if (status.status === 'completed') {
              updatePipeline({ phase: 'completing', progress: 90, stepMessage: 'Generating match stats...' });
              await completeAnalysis(videoId, set);
              return;
            } else if (status.status === 'failed') {
              updatePipeline({ phase: 'error', error: status.error || 'Tracking failed', stepMessage: 'Tracking failed' });
              return;
            } else if (status.status === 'processing' || status.status === 'pending') {
              pollTracking(videoId, set, get);
            } else {
              stalePollCounts.delete(videoId);
              updatePipeline({ phase: 'error', error: 'Tracking is no longer running. Please try again.', stepMessage: 'Tracking not found' });
            }
          } catch {
            pollTracking(videoId, set, get);
          }
        } else if (pipeline.phase === 'completing') {
          await completeAnalysis(videoId, set);
        }
      },

      setShowPlayerNaming: (videoId: string | null) => {
        set({ showPlayerNaming: videoId });
      },
    }),
    {
      name: 'rallycut-analysis',
      version: 1,
      partialize: (state) => ({
        // Only persist pipeline state, not ephemeral UI
        pipelines: state.pipelines,
      }),
    },
  ),
);

// ============================================================================
// Pipeline step helpers (not part of store, but operate on it)
// ============================================================================

async function startDetection(
  videoId: string,
  set: (fn: (state: AnalysisState) => Partial<AnalysisState>) => void,
  get: () => AnalysisState,
) {
  const updatePipeline = (patch: Partial<AnalysisPipeline>) => {
    set((state) => ({
      pipelines: {
        ...state.pipelines,
        [videoId]: { ...(state.pipelines[videoId] ?? DEFAULT_PIPELINE), ...patch },
      },
    }));
  };

  try {
    // First check if detection is already complete
    const pipelineStatus = await getAnalysisPipelineStatus(videoId);
    if (pipelineStatus.detection.status === 'completed' && pipelineStatus.detection.ralliesFound > 0) {
      // Skip detection — already done
      updatePipeline({
        phase: 'tracking',
        progress: 45,
        stepMessage: `Found ${pipelineStatus.detection.ralliesFound} rallies! Starting player tracking...`,
        ralliesFound: pipelineStatus.detection.ralliesFound,
      });
      await startTracking(videoId, set, get);
      return;
    }

    // Trigger detection
    await triggerRallyDetection(videoId, 'beach');
    updatePipeline({
      progress: 10,
      stepMessage: 'Analyzing video for rallies...',
    });

    // Start polling
    pollDetection(videoId, set, get);
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Detection failed';
    // If it's a conflict (already detecting), start polling
    if (message.includes('must be UPLOADED') || message.includes('DETECTING')) {
      pollDetection(videoId, set, get);
      return;
    }
    updatePipeline({
      phase: 'error',
      error: message,
      stepMessage: 'Detection failed',
    });
  }
}

function pollDetection(
  videoId: string,
  set: (fn: (state: AnalysisState) => Partial<AnalysisState>) => void,
  get: () => AnalysisState,
) {
  clearPollTimer(videoId);

  const updatePipeline = (patch: Partial<AnalysisPipeline>) => {
    set((state) => ({
      pipelines: {
        ...state.pipelines,
        [videoId]: { ...(state.pipelines[videoId] ?? DEFAULT_PIPELINE), ...patch },
      },
    }));
  };

  pollTimers[videoId] = setInterval(async () => {
    try {
      const pipeline = get().pipelines[videoId];
      if (!pipeline || pipeline.phase !== 'detecting') {
        clearPollTimer(videoId);
        return;
      }

      // Absolute timeout: if pipeline started > 30 min ago, give up
      if (pipeline.startedAt && Date.now() - pipeline.startedAt > STALE_TIMEOUT_MS) {
        clearPollTimer(videoId);
        updatePipeline({
          phase: 'error',
          error: 'Detection timed out. Please try again.',
          stepMessage: 'Detection timed out',
        });
        return;
      }

      const status = await getDetectionStatus(videoId);

      if (status.job?.status === 'COMPLETED') {
        clearPollTimer(videoId);

        // Fetch rallies found count
        const pipelineStatus = await getAnalysisPipelineStatus(videoId);
        const ralliesFound = pipelineStatus.detection.ralliesFound;

        if (ralliesFound === 0) {
          updatePipeline({
            phase: 'done',
            progress: 100,
            stepMessage: 'No rallies found in this video',
            ralliesFound: 0,
          });
          return;
        }

        updatePipeline({
          phase: 'tracking',
          progress: 45,
          stepMessage: `Found ${ralliesFound} rallies! Starting player tracking...`,
          ralliesFound,
        });

        await startTracking(videoId, set, get);
      } else if (status.job?.status === 'FAILED') {
        clearPollTimer(videoId);
        updatePipeline({
          phase: 'error',
          error: status.job.errorMessage || 'Detection failed',
          stepMessage: 'Detection failed',
        });
      } else if (status.job?.status === 'RUNNING') {
        stalePollCounts.delete(videoId);
        // Map detection progress (0-100) to pipeline progress (10-45)
        const detProgress = status.job.progress ?? 0;
        const progress = 10 + (detProgress / 100) * 35;
        updatePipeline({
          progress,
          stepMessage: status.job.progressMessage || 'Analyzing video for rallies...',
        });
      } else {
        // Unrecognized status (e.g. UPLOADED with no job) — count as stale
        const count = (stalePollCounts.get(videoId) ?? 0) + 1;
        stalePollCounts.set(videoId, count);
        if (count >= MAX_STALE_POLLS) {
          clearPollTimer(videoId);
          updatePipeline({
            phase: 'error',
            error: 'Detection does not appear to be running. Please try again.',
            stepMessage: 'Detection not found',
          });
        }
      }
    } catch {
      // Ignore poll errors, retry on next interval
    }
  }, 10000);
}

async function startTracking(
  videoId: string,
  set: (fn: (state: AnalysisState) => Partial<AnalysisState>) => void,
  get: () => AnalysisState,
) {
  const updatePipeline = (patch: Partial<AnalysisPipeline>) => {
    set((state) => ({
      pipelines: {
        ...state.pipelines,
        [videoId]: { ...(state.pipelines[videoId] ?? DEFAULT_PIPELINE), ...patch },
      },
    }));
  };

  try {
    // Check if tracking is already complete
    const status = await getBatchTrackingStatus(videoId);
    if (status.status === 'completed') {
      updatePipeline({
        phase: 'completing',
        progress: 90,
        stepMessage: 'Generating match stats...',
        trackingProgress: { completed: status.completedRallies ?? 0, total: status.totalRallies ?? 0 },
      });
      await completeAnalysis(videoId, set);
      return;
    }

    if (status.status !== 'processing' && status.status !== 'pending') {
      // Start batch tracking
      const result = await trackAllRallies(videoId);
      updatePipeline({
        progress: 48,
        stepMessage: `Tracking players (0 of ${result.totalRallies})...`,
        trackingProgress: { completed: 0, total: result.totalRallies },
      });
    }

    pollTracking(videoId, set, get);
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Tracking failed';
    updatePipeline({
      phase: 'error',
      error: message,
      stepMessage: 'Tracking failed',
    });
  }
}

function pollTracking(
  videoId: string,
  set: (fn: (state: AnalysisState) => Partial<AnalysisState>) => void,
  get: () => AnalysisState,
) {
  clearPollTimer(videoId);

  const updatePipeline = (patch: Partial<AnalysisPipeline>) => {
    set((state) => ({
      pipelines: {
        ...state.pipelines,
        [videoId]: { ...(state.pipelines[videoId] ?? DEFAULT_PIPELINE), ...patch },
      },
    }));
  };

  pollTimers[videoId] = setInterval(async () => {
    try {
      const pipeline = get().pipelines[videoId];
      if (!pipeline || pipeline.phase !== 'tracking') {
        clearPollTimer(videoId);
        return;
      }

      // Absolute timeout: if pipeline started > 30 min ago, give up
      if (pipeline.startedAt && Date.now() - pipeline.startedAt > STALE_TIMEOUT_MS) {
        clearPollTimer(videoId);
        updatePipeline({
          phase: 'error',
          error: 'Tracking timed out. Please try again.',
          stepMessage: 'Tracking timed out',
        });
        return;
      }

      const status = await getBatchTrackingStatus(videoId);

      if (status.status === 'completed') {
        clearPollTimer(videoId);

        updatePipeline({
          phase: 'completing',
          progress: 90,
          stepMessage: 'Generating match stats...',
          trackingProgress: {
            completed: status.completedRallies ?? 0,
            total: status.totalRallies ?? 0,
          },
        });

        await completeAnalysis(videoId, set);
      } else if (status.status === 'failed') {
        clearPollTimer(videoId);
        updatePipeline({
          phase: 'error',
          error: status.error || 'Tracking failed',
          stepMessage: 'Tracking failed',
        });
      } else if (status.status === 'processing' || status.status === 'pending') {
        stalePollCounts.delete(videoId);
        const completed = status.completedRallies ?? 0;
        const total = status.totalRallies ?? 1;
        // Map tracking progress (0-total) to pipeline progress (50-90)
        const progress = 50 + (completed / total) * 40;
        updatePipeline({
          progress,
          stepMessage: `Tracking players (rally ${completed + 1} of ${total})...`,
          trackingProgress: { completed, total },
        });
      } else {
        // Unrecognized status — count as stale
        const count = (stalePollCounts.get(videoId) ?? 0) + 1;
        stalePollCounts.set(videoId, count);
        if (count >= MAX_STALE_POLLS) {
          clearPollTimer(videoId);
          updatePipeline({
            phase: 'error',
            error: 'Tracking does not appear to be running. Please try again.',
            stepMessage: 'Tracking not found',
          });
        }
      }
    } catch {
      // Ignore poll errors
    }
  }, 10000);
}

async function completeAnalysis(
  videoId: string,
  set: (fn: (state: AnalysisState) => Partial<AnalysisState>) => void,
) {
  // Prevent concurrent calls (resumeIfNeeded + pollTracking can race)
  if (completingLock.has(videoId)) return;
  completingLock.add(videoId);

  const updatePipeline = (patch: Partial<AnalysisPipeline>) => {
    set((state) => ({
      pipelines: {
        ...state.pipelines,
        [videoId]: { ...(state.pipelines[videoId] ?? DEFAULT_PIPELINE), ...patch },
      },
    }));
  };

  try {
    // Wait a moment for match analysis to complete (auto-triggered by batch tracking)
    await new Promise((r) => setTimeout(r, 2000));

    const status = await getAnalysisPipelineStatus(videoId);
    const stats = await getMatchStatsApi(videoId);
    const playerCount = stats?.playerStats?.length ?? 0;
    const ralliesFound = status.detection.ralliesFound;

    updatePipeline({
      phase: 'done',
      progress: 100,
      stepMessage: `Analysis complete! ${ralliesFound} rallies, ${playerCount} players`,
      ralliesFound,
      playerCount,
    });

    // Show player naming dialog
    if (playerCount > 0) {
      set(() => ({ showPlayerNaming: videoId }));
    }
  } catch {
    // Stats failed but tracking is done — still mark as done
    updatePipeline({
      phase: 'done',
      progress: 100,
      stepMessage: 'Analysis complete!',
    });
  } finally {
    completingLock.delete(videoId);
  }
}
