import {
  expireStaleBatchTrackingJobs,
  expireStaleDetectionJobs,
} from '../services/staleJobRecovery.js';

const SWEEP_INTERVAL_MS = 5 * 60 * 1000;

let sweeperHandle: ReturnType<typeof setInterval> | null = null;

async function sweep(): Promise<void> {
  try {
    const [batchExpired, detExpired] = await Promise.all([
      expireStaleBatchTrackingJobs(),
      expireStaleDetectionJobs(),
    ]);
    if (batchExpired + detExpired > 0) {
      console.log(
        `[STALE_JOB_SWEEPER] expired ${batchExpired} batch + ${detExpired} detection job(s)`,
      );
    }
  } catch (err) {
    console.error('[STALE_JOB_SWEEPER] sweep failed:', err);
  }
}

/**
 * Start the 5-minute stale-job sweeper. Single process only — if the API
 * is scaled horizontally (not today), this will fire N times per interval,
 * but the updateMany queries are race-safe so the only cost is extra
 * console logging. A2b+ can replace with a distributed lock (Redis, PgAdvisoryLock)
 * if / when that matters.
 *
 * The sweep runs once immediately on startup, then every SWEEP_INTERVAL_MS.
 * Call stopStaleJobSweeper() on graceful shutdown to clear the timer.
 */
export function startStaleJobSweeper(): void {
  if (sweeperHandle !== null) return; // idempotent
  console.log(`[STALE_JOB_SWEEPER] starting, interval ${SWEEP_INTERVAL_MS / 1000}s`);
  void sweep(); // run immediately
  sweeperHandle = setInterval(() => void sweep(), SWEEP_INTERVAL_MS);
  // Don't keep the event loop alive just for the sweeper — if the API is
  // shutting down, don't block on the next tick.
  if (typeof sweeperHandle.unref === 'function') {
    sweeperHandle.unref();
  }
}

export function stopStaleJobSweeper(): void {
  if (sweeperHandle !== null) {
    clearInterval(sweeperHandle);
    sweeperHandle = null;
    console.log('[STALE_JOB_SWEEPER] stopped');
  }
}
