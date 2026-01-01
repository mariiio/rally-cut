/**
 * API client for RallyCut backend
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001';

// API response types (from backend)
interface ApiSession {
  id: string;
  name: string;
  createdAt: string;
  updatedAt: string;
  videos: ApiVideo[];
  highlights: ApiHighlight[];
}

interface ApiVideo {
  id: string;
  sessionId: string;
  name: string;
  filename: string;
  s3Key: string;
  contentHash: string;
  status: 'PENDING' | 'UPLOADED' | 'DETECTING' | 'DETECTED' | 'ERROR';
  durationMs: number | null;
  width: number | null;
  height: number | null;
  fileSizeBytes: string | null;
  order: number;
  rallies: ApiRally[];
}

interface ApiRally {
  id: string;
  videoId: string;
  startMs: number;
  endMs: number;
  confidence: number | null;
  scoreA: number | null;
  scoreB: number | null;
  servingTeam: 'A' | 'B' | null;
  notes: string | null;
  order: number;
}

interface ApiHighlight {
  id: string;
  sessionId: string;
  name: string;
  color: string;
  highlightRallies: ApiHighlightRally[];
}

interface ApiHighlightRally {
  id: string;
  highlightId: string;
  rallyId: string;
  order: number;
  rally?: ApiRally;
}

// Frontend types (from @/types/rally)
import type { Session, Match, Rally, Highlight, VideoMetadata } from '@/types/rally';

// Extended types with backend IDs for sync
export interface RallyWithBackendId extends Rally {
  _backendId: string;
}

export interface HighlightWithBackendId extends Highlight {
  _backendId: string;
  _rallyBackendIds: Record<string, string>; // frontendRallyId -> highlightRallyId
}

// Transform API response to frontend format
function apiRallyToFrontend(apiRally: ApiRally, videoId: string, fps: number): RallyWithBackendId {
  const startTime = apiRally.startMs / 1000;
  const endTime = apiRally.endMs / 1000;
  const duration = endTime - startTime;

  return {
    id: `${videoId}_rally_${apiRally.order + 1}`,
    _backendId: apiRally.id,
    start_time: startTime,
    end_time: endTime,
    start_frame: Math.round(startTime * fps),
    end_frame: Math.round(endTime * fps),
    duration,
    type: 'rally',
    thumbnail_time: startTime + duration / 2,
  };
}

function apiVideoToMatch(apiVideo: ApiVideo, cloudfrontDomain?: string): Match {
  // Default FPS if not known (will be updated when video loads)
  const fps = 30;
  const duration = apiVideo.durationMs ? apiVideo.durationMs / 1000 : 0;

  // Build video URL - use CloudFront if available, otherwise S3 key for local dev
  // s3Key already includes the full path (e.g., "videos/sessionId/videoId/filename.mp4")
  let videoUrl = `/${apiVideo.s3Key}`;
  if (cloudfrontDomain) {
    videoUrl = `https://${cloudfrontDomain}/${apiVideo.s3Key}`;
  }

  const videoMetadata: VideoMetadata = {
    path: apiVideo.filename,
    duration,
    fps,
    width: apiVideo.width || 1920,
    height: apiVideo.height || 1080,
    frame_count: Math.round(duration * fps),
  };

  const rallies = apiVideo.rallies.map(r => apiRallyToFrontend(r, apiVideo.id, fps));

  return {
    id: apiVideo.id,
    name: apiVideo.name,
    videoUrl,
    video: videoMetadata,
    rallies,
  };
}

function apiHighlightToFrontend(apiHighlight: ApiHighlight): HighlightWithBackendId {
  const rallyBackendIds: Record<string, string> = {};
  const rallyIds = apiHighlight.highlightRallies
    .sort((a, b) => a.order - b.order)
    .map(hr => {
      // Map API rally IDs to frontend format
      // We need the video ID to construct the frontend rally ID
      if (hr.rally) {
        const frontendId = `${hr.rally.videoId}_rally_${hr.rally.order + 1}`;
        rallyBackendIds[frontendId] = hr.id; // Store highlightRally ID
        return frontendId;
      }
      return hr.rallyId;
    });

  return {
    id: apiHighlight.id,
    _backendId: apiHighlight.id,
    _rallyBackendIds: rallyBackendIds,
    name: apiHighlight.name,
    color: apiHighlight.color,
    rallyIds,
    createdAt: Date.now(),
  };
}

function apiSessionToFrontend(apiSession: ApiSession, cloudfrontDomain?: string): Session {
  const matches = apiSession.videos
    .sort((a, b) => a.order - b.order)
    .map(v => apiVideoToMatch(v, cloudfrontDomain));

  const highlights = apiSession.highlights.map(h => apiHighlightToFrontend(h));

  return {
    id: apiSession.id,
    name: apiSession.name,
    matches,
    highlights,
  };
}

// API client functions
export async function fetchSession(sessionId: string): Promise<Session> {
  const response = await fetch(`${API_BASE_URL}/v1/sessions/${sessionId}`, {
    credentials: 'include', // Include cookies for CloudFront signed cookies
  });

  if (!response.ok) {
    throw new Error(`Failed to fetch session: ${response.status}`);
  }

  const apiSession: ApiSession = await response.json();

  // Get CloudFront domain from env if available
  const cloudfrontDomain = process.env.NEXT_PUBLIC_CLOUDFRONT_DOMAIN;

  return apiSessionToFrontend(apiSession, cloudfrontDomain);
}

export async function createSession(name: string): Promise<{ id: string; name: string }> {
  const response = await fetch(`${API_BASE_URL}/v1/sessions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name }),
  });

  if (!response.ok) {
    throw new Error(`Failed to create session: ${response.status}`);
  }

  return response.json();
}

export interface ListSessionsResponse {
  data: Array<{
    id: string;
    name: string;
    createdAt: string;
    updatedAt: string;
    _count: { videos: number; highlights: number };
  }>;
  pagination: {
    page: number;
    limit: number;
    total: number;
    totalPages: number;
  };
}

export async function listSessions(page = 1, limit = 20): Promise<ListSessionsResponse> {
  const response = await fetch(`${API_BASE_URL}/v1/sessions?page=${page}&limit=${limit}`);

  if (!response.ok) {
    throw new Error(`Failed to list sessions: ${response.status}`);
  }

  return response.json();
}

// Batch update types
type BatchOperation =
  | { type: 'create'; entity: 'rally'; tempId: string; data: { videoId: string; startMs: number; endMs: number } }
  | { type: 'create'; entity: 'highlight'; tempId: string; data: { name: string; color: string } }
  | { type: 'update'; entity: 'rally'; id: string; data: { startMs?: number; endMs?: number } }
  | { type: 'update'; entity: 'highlight'; id: string; data: { name?: string; color?: string } }
  | { type: 'delete'; entity: 'rally' | 'highlight' | 'highlightRally'; id: string }
  | { type: 'reorder'; entity: 'rally' | 'highlightRally'; parentId: string; order: string[] }
  | { type: 'addRallyToHighlight'; highlightId: string; rallyId: string; tempId?: string };

interface BatchResponse {
  success: boolean;
  created: Record<string, string>;
  updatedAt: string;
}

export async function batchUpdate(sessionId: string, operations: BatchOperation[]): Promise<BatchResponse> {
  const response = await fetch(`${API_BASE_URL}/v1/sessions/${sessionId}/batch`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ operations }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error?.message || `Batch update failed: ${response.status}`);
  }

  return response.json();
}

// Helper to convert frontend rally to API format for batch operations
export function frontendRallyToApi(rally: Rally): { startMs: number; endMs: number } {
  return {
    startMs: Math.round(rally.start_time * 1000),
    endMs: Math.round(rally.end_time * 1000),
  };
}

// Upload URL request
export async function requestUploadUrl(
  sessionId: string,
  filename: string,
  contentHash: string,
  fileSize: number,
  durationMs?: number
): Promise<{ uploadUrl: string; videoId: string; s3Key: string }> {
  const response = await fetch(`${API_BASE_URL}/v1/sessions/${sessionId}/videos/upload-url`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ filename, contentHash, fileSize, durationMs }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error?.message || `Failed to get upload URL: ${response.status}`);
  }

  return response.json();
}

// Confirm upload complete
export async function confirmUpload(
  sessionId: string,
  videoId: string,
  durationMs: number,
  width?: number,
  height?: number
): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/v1/sessions/${sessionId}/videos`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ videoId, durationMs, width, height }),
  });

  if (!response.ok) {
    throw new Error(`Failed to confirm upload: ${response.status}`);
  }
}

// Trigger rally detection
export async function triggerRallyDetection(videoId: string): Promise<{ jobId: string; status: string }> {
  const response = await fetch(`${API_BASE_URL}/v1/videos/${videoId}/detect-rallies`, {
    method: 'POST',
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error?.message || `Failed to trigger detection: ${response.status}`);
  }

  return response.json();
}

// Get detection status
export async function getDetectionStatus(videoId: string): Promise<{
  videoId: string;
  status: string;
  job: {
    id: string;
    status: string;
    progress?: number;
    progressMessage?: string;
    startedAt?: string;
    completedAt?: string;
    errorMessage?: string;
  } | null;
}> {
  const response = await fetch(`${API_BASE_URL}/v1/videos/${videoId}/detection-status`);

  if (!response.ok) {
    throw new Error(`Failed to get detection status: ${response.status}`);
  }

  return response.json();
}

// Delete session
export async function deleteSession(sessionId: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/v1/sessions/${sessionId}`, {
    method: 'DELETE',
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error?.message || `Failed to delete session: ${response.status}`);
  }
}

// Delete video
export async function deleteVideo(videoId: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/v1/videos/${videoId}`, {
    method: 'DELETE',
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error?.message || `Failed to delete video: ${response.status}`);
  }
}

// Rename video
export async function renameVideo(videoId: string, name: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/v1/videos/${videoId}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error?.message || `Failed to rename video: ${response.status}`);
  }
}

export { API_BASE_URL };
