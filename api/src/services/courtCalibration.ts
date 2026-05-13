/**
 * Court auto-calibration helpers shared between the preflight pipeline
 * (which runs the keypoint detector standalone) and the player-tracking
 * pipeline (which runs `auto_detect_court` per rally).
 *
 * Two write semantics live here because the two callers have different
 * needs:
 *
 *   - `refreshCourtAutoCalibration` (preflight) is authoritative. Every
 *     preflight run reflects the current state of the video, so saving
 *     when confident AND clearing when not is the right shape.
 *
 *   - `backfillCourtCalibration` (tracker) is a fill-only fallback. The
 *     tracker fires per-rally and per-rally detection can be noisy; it
 *     must never overwrite an existing calibration (manual OR auto) and
 *     must never clear. Its only job is to populate a missing snapshot.
 *
 * Manual calibrations are protected by both writers.
 */
import { Prisma } from '@prisma/client';
import { prisma } from '../lib/prisma.js';
import type { CourtDetection } from './qualityReport.js';

export type { CourtDetection };

// Confidence floor for writing courtCalibrationJson. Matches the threshold
// used by the pre-refactor assessVideoQuality path (commit 3bbb7a48).
export const AUTO_CALIBRATION_MIN_CONFIDENCE = 0.7;
// Per-axis off-screen budget. Beach footage with low camera angles legitimately
// places corners ~0.15 outside the frame; anything beyond 0.3 is degenerate.
export const AUTO_CALIBRATION_MAX_OFFSCREEN_MARGIN = 0.3;

export function areCornersReasonable(corners: Array<{ x: number; y: number }>): boolean {
  if (corners.length !== 4) return false;
  for (const c of corners) {
    if (
      c.x < -AUTO_CALIBRATION_MAX_OFFSCREEN_MARGIN ||
      c.x > 1 + AUTO_CALIBRATION_MAX_OFFSCREEN_MARGIN ||
      c.y < -AUTO_CALIBRATION_MAX_OFFSCREEN_MARGIN ||
      c.y > 1 + AUTO_CALIBRATION_MAX_OFFSCREEN_MARGIN
    ) {
      return false;
    }
  }
  return true;
}

/**
 * Internal: does this detection clear the save threshold?
 *
 * Exported for testability. Pure — no DB.
 */
export function detectionPassesQualityBar(detection: CourtDetection | null): boolean {
  return (
    detection !== null
    && areCornersReasonable(detection.corners)
    && detection.confidence >= AUTO_CALIBRATION_MIN_CONFIDENCE
  );
}

/**
 * Preflight write semantics — refresh.
 *
 * Saves a fresh court detection into `Video.courtCalibrationJson` when it
 * clears the confidence + on-screen bars, and clears a stale auto-calibration
 * when it doesn't. Manual calibrations are never overwritten.
 *
 * Wrapped in a transaction so concurrent runs can't TOCTOU a manual save.
 */
export async function refreshCourtAutoCalibration(
  videoId: string,
  detection: CourtDetection | null,
): Promise<void> {
  const passesQualityBar = detectionPassesQualityBar(detection);

  await prisma.$transaction(async (tx) => {
    const current = await tx.video.findUnique({
      where: { id: videoId },
      select: { courtCalibrationJson: true, courtCalibrationSource: true },
    });
    if (!current) return;
    if (current.courtCalibrationSource === 'manual') {
      if (detection !== null) {
        console.log(
          `[COURT_CAL] Skipped refresh for video ${videoId} — manual calibration exists`,
        );
      }
      return;
    }

    if (passesQualityBar && detection !== null) {
      await tx.video.update({
        where: { id: videoId },
        data: {
          courtCalibrationJson:
            detection.corners as unknown as Prisma.InputJsonValue,
          courtCalibrationSource: 'auto',
        },
      });
      console.log(
        `[COURT_CAL] Refreshed auto-calibration for video ${videoId} ` +
          `(confidence: ${detection.confidence.toFixed(2)})`,
      );
      return;
    }

    // Below the bar — clear any prior auto-calibration so the editor falls
    // back to defaults instead of a known-bad snapshot.
    if (current.courtCalibrationJson !== null) {
      await tx.video.update({
        where: { id: videoId },
        data: {
          courtCalibrationJson: Prisma.DbNull,
          courtCalibrationSource: null,
        },
      });
      const reason = detection !== null
        ? `confidence ${detection.confidence.toFixed(2)} < ${AUTO_CALIBRATION_MIN_CONFIDENCE}` +
          (areCornersReasonable(detection.corners) ? '' : ' or corners off-screen')
        : 'no detection';
      console.log(
        `[COURT_CAL] Cleared stale auto-calibration for video ${videoId} (${reason})`,
      );
    }
  });
}

/**
 * Tracker write semantics — fill-only.
 *
 * Saves the detection ONLY when `courtCalibrationJson` is null and the
 * detection clears the confidence + on-screen bars. Never overwrites, never
 * clears. Designed for the player-tracker path where:
 *
 *   - The tracker fires per-rally, so a single noisy rally must not clobber
 *     a stable preflight save.
 *   - Manual calibrations must remain inviolate.
 *   - A previous auto-save (from preflight or a prior rally) is left alone
 *     since the algorithm is the same and refreshing per-rally would thrash.
 *
 * Same TOCTOU-safe transaction shape as the refresh path.
 */
export async function backfillCourtCalibration(
  videoId: string,
  detection: CourtDetection | null,
): Promise<void> {
  if (!detectionPassesQualityBar(detection) || detection === null) return;

  await prisma.$transaction(async (tx) => {
    const current = await tx.video.findUnique({
      where: { id: videoId },
      select: { courtCalibrationJson: true, courtCalibrationSource: true },
    });
    if (!current) return;
    // Never touch anything that already has a value — manual OR auto.
    if (current.courtCalibrationJson !== null) return;
    if (current.courtCalibrationSource === 'manual') return;

    await tx.video.update({
      where: { id: videoId },
      data: {
        courtCalibrationJson:
          detection.corners as unknown as Prisma.InputJsonValue,
        courtCalibrationSource: 'auto',
      },
    });
    console.log(
      `[COURT_CAL] Backfilled auto-calibration from tracker for video ${videoId} ` +
        `(confidence: ${detection.confidence.toFixed(2)})`,
    );
  });
}
