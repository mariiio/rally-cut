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

// ============================================================================
// Shared types & helpers for pipeline step functions
// ============================================================================

type SetFn = (fn: (state: AnalysisState) => Partial<AnalysisState>) => void;
type GetFn = () => AnalysisState;
type PipelineUpdater = (patch: Partial<AnalysisPipeline>) => void;

function makePipelineUpdater(videoId: string, set: SetFn): PipelineUpdater {
  return (patch) =>
    set((state) => ({
      pipelines: {
        ...state.pipelines,
        [videoId]: { ...(state.pipelines[videoId] ?? DEFAULT_PIPELINE), ...patch },
      },
    }));
}

/**
 * Advance the pipeline after detection is known to be complete.
 * Fetches rally count and transitions to tracking or done.
 * Returns true if advanced to tracking (caller should follow up with startTracking).
 */
async function advanceAfterDetection(
  videoId: string,
  updatePipeline: PipelineUpdater,
): Promise<boolean> {
  const status = await getAnalysisPipelineStatus(videoId);
  const ralliesFound = status.detection.ralliesFound;
  if (ralliesFound === 0) {
    updatePipeline({ phase: 'done', progress: 100, stepMessage: 'No rallies found in this video', ralliesFound: 0 });
    return false;
  }
  updatePipeline({ phase: 'tracking', progress: 45, stepMessage: `Found ${ralliesFound} rallies! Starting player tracking...`, ralliesFound });
  return true;
}

// ============================================================================
// Polling infrastructure
// ============================================================================

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
const POLL_INTERVAL_MS = 10_000;

function clearPollTimer(videoId: string) {
  if (pollTimers[videoId]) {
    clearInterval(pollTimers[videoId]);
    delete pollTimers[videoId];
  }
  stalePollCounts.delete(videoId);
}

/** Returns true (and errors the pipeline) if the pipeline started > 30 min ago. */
function checkAbsoluteTimeout(videoId: string, pipeline: AnalysisPipeline, updatePipeline: PipelineUpdater, label: string): boolean {
  if (pipeline.startedAt && Date.now() - pipeline.startedAt > STALE_TIMEOUT_MS) {
    clearPollTimer(videoId);
    updatePipeline({ phase: 'error', error: `${label} timed out. Please try again.`, stepMessage: `${label} timed out` });
    return true;
  }
  return false;
}

/** Increment stale counter; error the pipeline if threshold reached. Returns true if errored. */
function countStaleAndMaybeError(videoId: string, updatePipeline: PipelineUpdater, label: string): boolean {
  const count = (stalePollCounts.get(videoId) ?? 0) + 1;
  stalePollCounts.set(videoId, count);
  if (count >= MAX_STALE_POLLS) {
    clearPollTimer(videoId);
    updatePipeline({ phase: 'error', error: `${label} does not appear to be running. Please try again.`, stepMessage: `${label} not found` });
    return true;
  }
  return false;
}

// ============================================================================
// Store
// ============================================================================

export const useAnalysisStore = create<AnalysisState>()(
  persist(
    (set, get) => ({
      pipelines: {},
      showPlayerNaming: null,

      getPipeline: (videoId: string) => {
        return get().pipelines[videoId] ?? DEFAULT_PIPELINE;
      },

      startAnalysis: async (videoId: string) => {
        // Cancel any stuck pipeline before restarting
        const existing = get().pipelines[videoId];
        if (existing && existing.phase !== 'idle' && existing.phase !== 'done' && existing.phase !== 'error') {
          clearPollTimer(videoId);
        }
        completingLock.delete(videoId);

        // Capture generation marker — all updates in this run include it so
        // concurrent/stale async callbacks can detect restarts.
        const startedAt = Date.now();

        const update = (patch: Partial<AnalysisPipeline>) => {
          set((state) => ({
            pipelines: {
              ...state.pipelines,
              [videoId]: { ...DEFAULT_PIPELINE, ...patch, startedAt },
            },
          }));
        };

        // Initialize pipeline
        update({
          phase: 'quality_check',
          progress: 2,
          stepMessage: 'Checking your video...',
        });

        // Step 1: Quality check
        try {
          const qualityResult = await assessQuality(videoId);

          // Hydrate court calibration if auto-saved during quality check.
          if (qualityResult.courtDetection.autoSaved && qualityResult.courtDetection.corners) {
            const { usePlayerTrackingStore } = await import('@/stores/playerTrackingStore');
            usePlayerTrackingStore.getState().hydrateFromAutoSave(videoId, qualityResult.courtDetection.corners);
          }

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

        // Capture generation marker — if startAnalysis resets the pipeline while
        // we're awaiting an API call, startedAt will change and we must bail out.
        const generation = pipeline.startedAt;

        const isStale = () => {
          const current = get().pipelines[videoId];
          return !current || current.startedAt !== generation;
        };

        const baseUpdate = makePipelineUpdater(videoId, set);
        const updatePipeline: PipelineUpdater = (patch) => {
          if (!isStale()) baseUpdate(patch);
        };

        if (checkAbsoluteTimeout(videoId, pipeline, updatePipeline, 'Analysis')) return;

        if (pipeline.phase === 'detecting' && !pollTimers[videoId]) {
          await resumeDetecting(videoId, set, get, updatePipeline, isStale);
        } else if (pipeline.phase === 'tracking' && !pollTimers[videoId]) {
          await resumeTracking(videoId, set, get, updatePipeline, isStale);
        } else if (pipeline.phase === 'completing') {
          if (!isStale()) await completeAnalysis(videoId, set, get);
        }
      },

      setShowPlayerNaming: (videoId: string | null) => {
        set({ showPlayerNaming: videoId });
      },
    }),
    {
      name: 'rallycut-analysis',
      version: 2,
      migrate: () => ({ pipelines: {} }),
      partialize: (state) => ({
        // Only persist resumable in-progress pipelines. Terminal states
        // (idle/done/error) are transient feedback. Quality phases are quick
        // local API calls that can't be resumed — just restart on reload.
        pipelines: Object.fromEntries(
          Object.entries(state.pipelines).filter(
            ([, p]) =>
              p.phase !== 'idle' &&
              p.phase !== 'done' &&
              p.phase !== 'error' &&
              p.phase !== 'quality_check' &&
              p.phase !== 'quality_warning',
          ),
        ),
      }),
    },
  ),
);

// ============================================================================
// Pipeline step helpers (not part of store, but operate on it)
// ============================================================================

async function startDetection(videoId: string, set: SetFn, get: GetFn) {
  const updatePipeline = makePipelineUpdater(videoId, set);

  try {
    // First check if detection is already complete
    const pipelineStatus = await getAnalysisPipelineStatus(videoId);
    if (pipelineStatus.detection.status === 'completed' && pipelineStatus.detection.ralliesFound > 0) {
      if (await advanceAfterDetection(videoId, updatePipeline)) {
        await startTracking(videoId, set, get);
      }
      return;
    }

    // Trigger detection
    const triggerResult = await triggerRallyDetection(videoId, 'beach');

    // If detection was resolved instantly (e.g. content-hash cache hit), skip polling
    if (triggerResult.status === 'completed') {
      if (await advanceAfterDetection(videoId, updatePipeline)) {
        await startTracking(videoId, set, get);
      }
      return;
    }

    updatePipeline({ progress: 10, stepMessage: 'Analyzing video for rallies...' });
    pollDetection(videoId, set, get);
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Detection failed';
    // If it's a conflict (already detecting), start polling
    if (message.includes('must be UPLOADED') || message.includes('DETECTING')) {
      pollDetection(videoId, set, get);
      return;
    }
    updatePipeline({ phase: 'error', error: message, stepMessage: 'Detection failed' });
  }
}

function pollDetection(videoId: string, set: SetFn, get: GetFn) {
  clearPollTimer(videoId);
  const updatePipeline = makePipelineUpdater(videoId, set);

  pollTimers[videoId] = setInterval(async () => {
    try {
      const pipeline = get().pipelines[videoId];
      if (!pipeline || pipeline.phase !== 'detecting') {
        clearPollTimer(videoId);
        return;
      }

      if (checkAbsoluteTimeout(videoId, pipeline, updatePipeline, 'Detection')) return;

      const status = await getDetectionStatus(videoId);

      if (status.job?.status === 'COMPLETED' || (!status.job && status.status === 'DETECTED')) {
        // Detection done — either job completed or resolved via content-hash cache
        clearPollTimer(videoId);
        if (await advanceAfterDetection(videoId, updatePipeline)) {
          await startTracking(videoId, set, get);
        }
      } else if (status.job?.status === 'FAILED') {
        clearPollTimer(videoId);
        updatePipeline({ phase: 'error', error: status.job.errorMessage || 'Detection failed', stepMessage: 'Detection failed' });
      } else if (status.job?.status === 'RUNNING' || status.job?.status === 'PENDING') {
        stalePollCounts.delete(videoId);
        const detProgress = status.job.progress ?? 0;
        const progress = 10 + (detProgress / 100) * 35;
        const fallbackMessage = status.job.status === 'PENDING' ? 'Waiting for detection to start...' : 'Analyzing video for rallies...';
        updatePipeline({ progress, stepMessage: status.job.progressMessage || fallbackMessage });
      } else {
        countStaleAndMaybeError(videoId, updatePipeline, 'Detection');
      }
    } catch {
      // Ignore poll errors, retry on next interval
    }
  }, POLL_INTERVAL_MS);
}

async function startTracking(videoId: string, set: SetFn, get: GetFn) {
  const updatePipeline = makePipelineUpdater(videoId, set);

  try {
    // Check if tracking is currently in progress (e.g. resumed after navigation)
    const status = await getBatchTrackingStatus(videoId);

    if (status.status === 'processing' || status.status === 'pending') {
      pollTracking(videoId, set, get);
      return;
    }

    // Start a new batch tracking job
    const result = await trackAllRallies(videoId);
    updatePipeline({
      progress: 48,
      stepMessage: `Tracking players (0 of ${result.totalRallies})...`,
      trackingProgress: { completed: 0, total: result.totalRallies },
    });

    pollTracking(videoId, set, get);
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Tracking failed';
    updatePipeline({ phase: 'error', error: message, stepMessage: 'Tracking failed' });
  }
}

function pollTracking(videoId: string, set: SetFn, get: GetFn) {
  clearPollTimer(videoId);
  const updatePipeline = makePipelineUpdater(videoId, set);

  pollTimers[videoId] = setInterval(async () => {
    try {
      const pipeline = get().pipelines[videoId];
      if (!pipeline || pipeline.phase !== 'tracking') {
        clearPollTimer(videoId);
        return;
      }

      if (checkAbsoluteTimeout(videoId, pipeline, updatePipeline, 'Tracking')) return;

      const status = await getBatchTrackingStatus(videoId);

      if (status.status === 'completed') {
        clearPollTimer(videoId);
        updatePipeline({
          phase: 'completing',
          progress: 90,
          stepMessage: 'Generating match stats...',
          trackingProgress: { completed: status.completedRallies ?? 0, total: status.totalRallies ?? 0 },
        });
        await completeAnalysis(videoId, set, get);
      } else if (status.status === 'failed') {
        clearPollTimer(videoId);
        updatePipeline({ phase: 'error', error: status.error || 'Tracking failed', stepMessage: 'Tracking failed' });
      } else if (status.status === 'processing' || status.status === 'pending') {
        stalePollCounts.delete(videoId);
        const completed = status.completedRallies ?? 0;
        const total = status.totalRallies ?? 1;
        const progress = 50 + (completed / total) * 40;
        updatePipeline({
          progress,
          stepMessage: `Tracking players (rally ${completed + 1} of ${total})...`,
          trackingProgress: { completed, total },
        });
      } else {
        countStaleAndMaybeError(videoId, updatePipeline, 'Tracking');
      }
    } catch {
      // Ignore poll errors
    }
  }, POLL_INTERVAL_MS);
}

async function completeAnalysis(videoId: string, set: SetFn, get: GetFn) {
  // Prevent concurrent calls (resumeIfNeeded + pollTracking can race)
  if (completingLock.has(videoId)) return;
  completingLock.add(videoId);

  // Capture generation to detect pipeline restarts during async work
  const generation = get().pipelines[videoId]?.startedAt;
  const isStale = () => {
    const current = get().pipelines[videoId];
    return !current || current.startedAt !== generation;
  };
  const baseUpdate = makePipelineUpdater(videoId, set);
  const updatePipeline: PipelineUpdater = (patch) => {
    if (!isStale()) baseUpdate(patch);
  };

  try {
    // Wait a moment for match analysis to complete (auto-triggered by batch tracking)
    await new Promise((r) => setTimeout(r, 2000));
    if (isStale()) return;

    const status = await getAnalysisPipelineStatus(videoId);
    const stats = await getMatchStatsApi(videoId);
    if (isStale()) return;

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
    if (playerCount > 0 && !isStale()) {
      set(() => ({ showPlayerNaming: videoId }));
    }
  } catch {
    // Stats failed but tracking is done — still mark as done
    updatePipeline({ phase: 'done', progress: 100, stepMessage: 'Analysis complete!' });
  } finally {
    completingLock.delete(videoId);
  }
}

// ============================================================================
// resumeIfNeeded sub-handlers (keep the main action readable)
// ============================================================================

async function resumeDetecting(
  videoId: string,
  set: SetFn,
  get: GetFn,
  updatePipeline: PipelineUpdater,
  isStale: () => boolean,
) {
  try {
    const status = await getDetectionStatus(videoId);
    if (isStale()) return;

    if (status.job?.status === 'COMPLETED' || (!status.job && status.status === 'DETECTED')) {
      if (await advanceAfterDetection(videoId, updatePipeline)) {
        if (!isStale()) await startTracking(videoId, set, get);
      }
    } else if (status.job?.status === 'FAILED') {
      updatePipeline({ phase: 'error', error: status.job.errorMessage || 'Detection failed', stepMessage: 'Detection failed' });
    } else if (status.job?.status === 'RUNNING' || status.job?.status === 'PENDING') {
      if (!isStale()) pollDetection(videoId, set, get);
    } else {
      updatePipeline({ phase: 'error', error: 'Detection is no longer running. Please try again.', stepMessage: 'Detection not found' });
    }
  } catch {
    // API call failed — start polling, stale counter will catch permanent failures
    if (!isStale()) pollDetection(videoId, set, get);
  }
}

async function resumeTracking(
  videoId: string,
  set: SetFn,
  get: GetFn,
  updatePipeline: PipelineUpdater,
  isStale: () => boolean,
) {
  try {
    const status = await getBatchTrackingStatus(videoId);
    if (isStale()) return;

    if (status.status === 'completed') {
      updatePipeline({ phase: 'completing', progress: 90, stepMessage: 'Generating match stats...' });
      if (!isStale()) await completeAnalysis(videoId, set, get);
    } else if (status.status === 'failed') {
      updatePipeline({ phase: 'error', error: status.error || 'Tracking failed', stepMessage: 'Tracking failed' });
    } else if (status.status === 'processing' || status.status === 'pending') {
      if (!isStale()) pollTracking(videoId, set, get);
    } else {
      updatePipeline({ phase: 'error', error: 'Tracking is no longer running. Please try again.', stepMessage: 'Tracking not found' });
    }
  } catch {
    if (!isStale()) pollTracking(videoId, set, get);
  }
}
