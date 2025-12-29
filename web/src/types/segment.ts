/**
 * Video segment types matching RallyCut JSON output format.
 * @see /Users/mario/Desktop/segments.json for example
 */

/** Metadata about the loaded video file */
export interface VideoMetadata {
  path: string;
  duration: number;
  fps: number;
  width: number;
  height: number;
  frame_count: number;
}

/** A single segment representing a rally in the video */
export interface Rally {
  id: string;
  start_time: number;
  end_time: number;
  start_frame: number;
  end_frame: number;
  duration: number;
  type: 'rally';
  thumbnail_time: number;
}

/** Aggregate statistics about all segments */
export interface SegmentStats {
  original_duration: number;
  kept_duration: number;
  removed_duration: number;
  kept_percentage: number;
  removed_percentage: number;
  segment_count: number;
}

/** Complete JSON file structure matching RallyCut output */
export interface SegmentFile {
  version: '1.0';
  video: VideoMetadata;
  rallies: Rally[];
  stats: SegmentStats;
}

/** Helper to create a new Rally with auto-generated fields */
export function createRally(
  id: string,
  startTime: number,
  endTime: number,
  fps: number
): Rally {
  const duration = endTime - startTime;
  return {
    id,
    start_time: startTime,
    end_time: endTime,
    start_frame: Math.round(startTime * fps),
    end_frame: Math.round(endTime * fps),
    duration,
    type: 'rally',
    thumbnail_time: startTime + duration / 2,
  };
}

/** Recalculate derived fields for a rally */
export function recalculateRally(rally: Rally, fps: number): Rally {
  const duration = rally.end_time - rally.start_time;
  return {
    ...rally,
    start_frame: Math.round(rally.start_time * fps),
    end_frame: Math.round(rally.end_time * fps),
    duration,
    thumbnail_time: rally.start_time + duration / 2,
  };
}

/** Calculate stats from rallies array */
export function calculateStats(
  rallies: Rally[],
  originalDuration: number
): SegmentStats {
  const keptDuration = rallies.reduce((sum, r) => sum + r.duration, 0);
  const removedDuration = originalDuration - keptDuration;
  return {
    original_duration: originalDuration,
    kept_duration: keptDuration,
    removed_duration: removedDuration,
    kept_percentage: (keptDuration / originalDuration) * 100,
    removed_percentage: (removedDuration / originalDuration) * 100,
    segment_count: rallies.length,
  };
}
