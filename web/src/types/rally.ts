/**
 * Video rally types matching RallyCut JSON output format.
 * @see /Users/mario/Desktop/rallies.json for example
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

/** A single rally in the video */
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

/** Aggregate statistics about all rallies */
export interface RallyStats {
  original_duration: number;
  kept_duration: number;
  removed_duration: number;
  kept_percentage: number;
  removed_percentage: number;
  rally_count: number;
}

/** A highlight collection of rally IDs */
export interface Highlight {
  id: string;           // Unique identifier (e.g., "highlight_1")
  name: string;         // User-editable name (e.g., "Best Rallies")
  color: string;        // Hex color from palette (e.g., "#FF6B6B")
  rallyIds: string[];   // Array of rally IDs belonging to this highlight
  createdAt: number;    // Timestamp for ordering
}

/** Color palette for highlights - designed for dark theme */
export const HIGHLIGHT_COLORS = [
  '#FF6B6B', // Coral Red
  '#4ECDC4', // Teal
  '#FFE66D', // Yellow
  '#95E1D3', // Mint
  '#F38181', // Salmon
  '#AA96DA', // Lavender
  '#FCBAD3', // Pink
  '#A8D8EA', // Sky Blue
  '#FF9F43', // Orange
  '#6A89CC', // Periwinkle
] as const;

/** Complete JSON file structure matching RallyCut output */
export interface RallyFile {
  version: '1.0';
  video: VideoMetadata;
  rallies: Rally[];
  stats: RallyStats;
  highlights?: Highlight[]; // Optional for backwards compatibility
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
): RallyStats {
  const keptDuration = rallies.reduce((sum, r) => sum + r.duration, 0);
  const removedDuration = originalDuration - keptDuration;
  return {
    original_duration: originalDuration,
    kept_duration: keptDuration,
    removed_duration: removedDuration,
    kept_percentage: (keptDuration / originalDuration) * 100,
    removed_percentage: (removedDuration / originalDuration) * 100,
    rally_count: rallies.length,
  };
}
