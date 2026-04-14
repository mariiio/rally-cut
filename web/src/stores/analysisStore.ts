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
  | 'preflight'
  | 'preflight_gate'
  | 'detecting'
  | 'ready_tracking'
  | 'match_analyzing'
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
  /** Set to true once BatchTrackingJob.status === COMPLETED. Used to gate the
   *  match-analysis debounce: the debounce only fires when both this flag is
   *  true AND a 5-second quiet window has elapsed since the last rally CRUD. */
  batchTrackingComplete?: boolean;
}

const DEFAULT_PIPELINE: AnalysisPipeline = {
  phase: 'idle',
  progress: 0,
  stepMessage: '',
};

interface AnalysisState {
  pipelines: Record<string, AnalysisPipeline>;

  // Actions
  getPipeline: (videoId: string) => AnalysisPipeline;
  startAnalysis: (videoId: string) => Promise<void>;
  dismissWarnings: (videoId: string) => void;
  cancelAnalysis: (videoId: string) => void;
  resumeIfNeeded: (videoId: string) => Promise<void>;

  /** Called by editorStore after any rally CRUD for this video. Resets the
   *  5-second debounce that eventually fires match-analysis — but only if the
   *  pipeline is currently in `ready_tracking` AND batchTrackingComplete=true.
   *  Edits during active tracking (before the batch finishes) are held until
   *  the batch completes and arms the initial debounce. */
  notifyRallyEdited: (videoId: string) => void;
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
  updatePipeline({ phase: 'ready_tracking', progress: 45, stepMessage: `Found ${ralliesFound} rallies! Starting player tracking...`, ralliesFound });
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

// ============================================================================
// Match-analysis debounce timers (keyed per videoId)
// ============================================================================

const MATCH_ANALYSIS_DEBOUNCE_MS = 5000;

const HMR_DEBOUNCE_KEY = '__rallycut_match_analysis_debounce_timers__' as const;

const staleDebounceTimers = (globalThis as Record<string, unknown>)[HMR_DEBOUNCE_KEY] as
  | Record<string, ReturnType<typeof setTimeout>>
  | undefined;
if (staleDebounceTimers) {
  Object.values(staleDebounceTimers).forEach(clearTimeout);
}

const matchAnalysisDebounceTimers: Record<string, ReturnType<typeof setTimeout>> = {};
(globalThis as Record<string, unknown>)[HMR_DEBOUNCE_KEY] = matchAnalysisDebounceTimers;

function clearMatchAnalysisDebounce(videoId: string) {
  if (matchAnalysisDebounceTimers[videoId]) {
    clearTimeout(matchAnalysisDebounceTimers[videoId]);
    delete matchAnalysisDebounceTimers[videoId];
  }
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

function armMatchAnalysisDebounce(videoId: string, set: SetFn, get: GetFn) {
  clearMatchAnalysisDebounce(videoId);
  const updatePipeline = makePipelineUpdater(videoId, set);
  matchAnalysisDebounceTimers[videoId] = setTimeout(async () => {
    delete matchAnalysisDebounceTimers[videoId];
    const current = get().pipelines[videoId];
    if (!current || current.phase !== 'ready_tracking' || !current.batchTrackingComplete) return;

    updatePipeline({
      phase: 'match_analyzing',
      progress: 92,
      stepMessage: 'Generating match stats...',
    });
    try {
      const { triggerMatchAnalysis } = await import('@/services/api');
      await triggerMatchAnalysis(videoId);
      await completeAnalysis(videoId, set, get);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Match analysis failed';
      updatePipeline({ phase: 'error', error: message, stepMessage: 'Match analysis failed' });
    }
  }, MATCH_ANALYSIS_DEBOUNCE_MS);
}

export const useAnalysisStore = create<AnalysisState>()(
  persist(
    (set, get) => ({
      pipelines: {},

      getPipeline: (videoId: string) => {
        return get().pipelines[videoId] ?? DEFAULT_PIPELINE;
      },

      startAnalysis: async (videoId: string) => {
        // Cancel any stuck pipeline before restarting
        const existing = get().pipelines[videoId];
        if (existing && existing.phase !== 'idle' && existing.phase !== 'done' && existing.phase !== 'error') {
          clearPollTimer(videoId);
        }
        clearMatchAnalysisDebounce(videoId);
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
          phase: 'preflight',
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
              phase: 'preflight_gate',
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
        if (!pipeline || pipeline.phase !== 'preflight_gate') return;

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
        clearMatchAnalysisDebounce(videoId);
        set((state) => ({
          pipelines: {
            ...state.pipelines,
            [videoId]: DEFAULT_PIPELINE,
          },
        }));
      },

      notifyRallyEdited: (videoId: string) => {
        const pipeline = get().pipelines[videoId];
        if (!pipeline || pipeline.phase !== 'ready_tracking') return;
        if (!pipeline.batchTrackingComplete) return;
        armMatchAnalysisDebounce(videoId, set, get);
      },

      resumeIfNeeded: async (videoId: string) => {
        const pipeline = get().pipelines[videoId];
        const phase = pipeline?.phase ?? 'idle';

        // Don't interfere with active polling
        if (pollTimers[videoId]) return;

        // For persisted in-progress phases, resume from local state
        if (pipeline && phase !== 'idle') {
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

          if (phase === 'detecting') {
            await resumeDetecting(videoId, set, get, updatePipeline, isStale);
          } else if (phase === 'ready_tracking') {
            await resumeTracking(videoId, set, get, updatePipeline, isStale);
          } else if (phase === 'match_analyzing') {
            if (!isStale()) await completeAnalysis(videoId, set, get);
          }
          return;
        }

        // For idle/missing pipeline, check backend for active processing.
        // Catches page reloads during non-persisted phases (quality_check)
        // where the backend is still running.
        try {
          const status = await getAnalysisPipelineStatus(videoId);
          // Re-check — startAnalysis may have fired while we were awaiting
          const current = get().pipelines[videoId];
          if (current && current.phase !== 'idle') return;

          const updatePipeline = makePipelineUpdater(videoId, set);

          if (status.detection.status === 'processing') {
            updatePipeline({
              phase: 'detecting',
              progress: 15,
              stepMessage: 'Analyzing video for rallies...',
              startedAt: Date.now(),
            });
            pollDetection(videoId, set, get);
          } else if (status.tracking.status === 'processing' || status.tracking.status === 'pending') {
            const completed = status.tracking.completed ?? 0;
            const total = Math.max(status.tracking.total ?? 1, 1);
            updatePipeline({
              phase: 'ready_tracking',
              progress: 50 + (completed / total) * 40,
              stepMessage: `Tracking players (rally ${completed + 1} of ${total})...`,
              trackingProgress: { completed, total },
              startedAt: Date.now(),
            });
            pollTracking(videoId, set, get);
          }
        } catch {
          // Non-critical — if backend is unreachable, stay idle
        }
      },

    }),
    {
      name: 'rallycut-analysis',
      version: 3,
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
              p.phase !== 'preflight' &&
              p.phase !== 'preflight_gate',
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
    // If detection already complete with rallies, skip to tracking
    const pipelineStatus = await getAnalysisPipelineStatus(videoId);
    if (pipelineStatus.detection.status === 'completed' && pipelineStatus.detection.ralliesFound > 0) {
      if (await advanceAfterDetection(videoId, updatePipeline)) {
        await startTracking(videoId, set, get);
      }
      return;
    }
    // If detection "complete" but 0 rallies (user deleted them), fall through to re-trigger

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
    // Always call trackAllRallies — it handles dedup internally
    // (returns existing job if rally count unchanged, creates new one otherwise)
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
      if (!pipeline || pipeline.phase !== 'ready_tracking') {
        clearPollTimer(videoId);
        return;
      }

      if (checkAbsoluteTimeout(videoId, pipeline, updatePipeline, 'Tracking')) return;

      const status = await getBatchTrackingStatus(videoId);

      if (status.status === 'completed') {
        clearPollTimer(videoId);
        updatePipeline({
          progress: 90,
          stepMessage: 'Tracking complete — waiting for edits to settle...',
          trackingProgress: { completed: status.completedRallies ?? 0, total: status.totalRallies ?? 0 },
          batchTrackingComplete: true,
        });
        armMatchAnalysisDebounce(videoId, set, get);
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
    // Match analysis is now fire-and-forget on the server (Task 3 removed the
    // synchronous webhook call). Poll until stats appear or we time out.
    const MATCH_ANALYSIS_TIMEOUT_MS = 60_000;
    const POLL_INTERVAL_MS = 2_000;
    const start = Date.now();

    while (Date.now() - start < MATCH_ANALYSIS_TIMEOUT_MS) {
      await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
      if (isStale()) return;

      const stats = await getMatchStatsApi(videoId);
      if (isStale()) return;

      const playerCount = stats?.playerStats?.length ?? 0;
      if (playerCount > 0) {
        const status = await getAnalysisPipelineStatus(videoId);
        if (isStale()) return;
        const ralliesFound = status.detection.ralliesFound;
        updatePipeline({
          phase: 'done',
          progress: 100,
          stepMessage: `Analysis complete! ${ralliesFound} rallies, ${playerCount} players`,
          ralliesFound,
          playerCount,
        });
        return;
      }
    }

    // Timed out — stats never materialized. Fall back to "done" with whatever we have.
    const status = await getAnalysisPipelineStatus(videoId);
    if (isStale()) return;
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
      updatePipeline({
        progress: 90,
        stepMessage: 'Tracking complete — waiting for edits to settle...',
        batchTrackingComplete: true,
      });
      if (!isStale()) armMatchAnalysisDebounce(videoId, set, get);
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
