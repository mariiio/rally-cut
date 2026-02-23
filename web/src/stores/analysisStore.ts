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
  resumeIfNeeded: (videoId: string) => void;
  setShowPlayerNaming: (videoId: string | null) => void;
}

// Polling intervals stored outside React state
const pollTimers: Record<string, ReturnType<typeof setInterval>> = {};
// Guard against concurrent completeAnalysis calls
const completingLock = new Set<string>();

function clearPollTimer(videoId: string) {
  if (pollTimers[videoId]) {
    clearInterval(pollTimers[videoId]);
    delete pollTimers[videoId];
  }
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

      resumeIfNeeded: (videoId: string) => {
        const pipeline = get().pipelines[videoId];
        if (!pipeline) return;

        // Resume polling based on current phase
        if (pipeline.phase === 'detecting' && !pollTimers[videoId]) {
          pollDetection(videoId, set, get);
        } else if (pipeline.phase === 'tracking' && !pollTimers[videoId]) {
          pollTracking(videoId, set, get);
        } else if (pipeline.phase === 'completing') {
          completeAnalysis(videoId, set);
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
        // Map detection progress (0-100) to pipeline progress (10-45)
        const detProgress = status.job.progress ?? 0;
        const progress = 10 + (detProgress / 100) * 35;
        updatePipeline({
          progress,
          stepMessage: status.job.progressMessage || 'Analyzing video for rallies...',
        });
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
      } else if (status.status === 'processing') {
        const completed = status.completedRallies ?? 0;
        const total = status.totalRallies ?? 1;
        // Map tracking progress (0-total) to pipeline progress (50-90)
        const progress = 50 + (completed / total) * 40;
        updatePipeline({
          progress,
          stepMessage: `Tracking players (rally ${completed + 1} of ${total})...`,
          trackingProgress: { completed, total },
        });
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
