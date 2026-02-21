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
  /** Backend UUID, populated after sync with server */
  _backendId?: string;
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
  createdByUserId?: string | null;  // User who created this highlight
  createdByUserName?: string | null; // Name of user who created this highlight
  /** Backend UUID, populated after sync with server */
  _backendId?: string;
}

/** A single match (video) within a session */
export interface Match {
  id: string;           // API video UUID
  name: string;         // Display name, e.g., "Match 1"
  s3Key?: string;       // S3 key for server-side operations (export, etc.) - only for API-loaded videos
  videoUrl: string;     // Runtime URL (full quality, public path or blob URL)
  proxyUrl?: string;    // Optional 720p proxy URL for faster editing
  posterUrl?: string;   // Optional poster image URL for instant display
  video: VideoMetadata; // Video metadata (fps, duration, dimensions)
  rallies: Rally[];     // Rallies for this match
  createdAt?: string;   // ISO timestamp when video was uploaded
  qualityDowngradedAt?: string | null; // ISO timestamp when original quality was removed (FREE tier)
  status?: 'PENDING' | 'UPLOADED' | 'DETECTING' | 'DETECTED' | 'ERROR'; // Video detection status
  characteristicsJson?: VideoCharacteristics | null; // Auto-detected video quality characteristics
}

/** Auto-detected video characteristics for quality insights and stratified evaluation */
export interface VideoCharacteristics {
  brightness?: { mean: number; category: string };
  cameraDistance?: { avgBboxHeight: number; category: string };
  sceneComplexity?: { avgPeople: number; category: string };
  courtDetection?: {
    detected: boolean;
    confidence: number;
    linesFound: number;
    cameraHeight: string;
    lineVisibility: string;
    recordingTips: string[];
  };
  expectedQuality?: number;
  uploadWarnings?: string[];
  version: number;
}

/** A session containing multiple matches */
export interface Session {
  id: string;           // e.g., "session_1"
  name: string;         // Display name
  matches: Match[];     // Collection of matches (videos)
  highlights: Highlight[]; // Cross-match highlights
  userRole?: 'owner' | 'VIEWER' | 'EDITOR' | 'ADMIN' | null; // User's role in this session
  updatedAt?: string;   // ISO timestamp of last server update
}

/** Session manifest file structure (used for loading from static files) */
export interface SessionManifest {
  version: '2.0';
  session: {
    id: string;
    name: string;
    matches: {
      id: string;
      name: string;
      dataFile: string;
      videoFile: string;
    }[];
  };
}

/** Color palette for highlights - beach volleyball theme
 * Note: Avoid #FF6B4A, #FF8A6F (selected rally colors) and #60A5FA, #3B82F6 (unselected rally colors)
 */
export const HIGHLIGHT_COLORS = [
  '#FFE66D', // Sand Gold
  '#F06292', // Pink
  '#00D4AA', // Teal (Secondary)
  '#FFCC80', // Light Orange
  '#4DDFBF', // Aqua Mint
  '#F5BC3C', // Warm Gold
  '#4DD0E1', // Cyan
  '#FB7185', // Rose
  '#A78BFA', // Lavender
  '#34D399', // Emerald
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
