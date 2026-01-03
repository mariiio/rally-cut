/**
 * API client for RallyCut backend
 */

import { getVisitorId } from '@/utils/visitorId';

export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001';

/**
 * Get default headers including X-Visitor-Id for user identification.
 */
function getHeaders(contentType?: string): HeadersInit {
  const headers: HeadersInit = {};

  const visitorId = getVisitorId();
  if (visitorId) {
    headers['X-Visitor-Id'] = visitorId;
  }

  if (contentType) {
    headers['Content-Type'] = contentType;
  }

  return headers;
}

/**
 * Get video stream URL from S3 key.
 * Uses CloudFront if configured, otherwise falls back to local proxy.
 */
export function getVideoStreamUrl(s3Key: string): string {
  const cloudfrontDomain = process.env.NEXT_PUBLIC_CLOUDFRONT_DOMAIN;
  return cloudfrontDomain
    ? `https://${cloudfrontDomain}/${s3Key}`
    : `/${s3Key}`;
}

// Video processing status (for upload optimization)
export type ProcessingStatus = 'PENDING' | 'QUEUED' | 'PROCESSING' | 'COMPLETED' | 'FAILED' | 'SKIPPED';

// API response types (from backend)
interface ApiSession {
  id: string;
  name: string;
  createdAt: string;
  updatedAt: string;
  videos: ApiVideo[];
  highlights: ApiHighlight[];
  userRole?: 'owner' | 'member' | null;
}

interface ApiVideo {
  id: string;
  sessionId: string;
  name: string;
  filename: string;
  s3Key: string;
  contentHash: string;
  status: 'PENDING' | 'UPLOADED' | 'DETECTING' | 'DETECTED' | 'ERROR';
  processingStatus?: ProcessingStatus;
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
  createdByUserId?: string | null;
  createdByUser?: { id: string; name: string | null } | null;
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

// Extended types with backend IDs for sync (internal use only)
interface RallyWithBackendId extends Rally {
  _backendId: string;
}

interface HighlightWithBackendId extends Highlight {
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

  // Build video URL
  // In production: use CloudFront for global CDN delivery
  // In development: always use relative URL (proxied to backend) to avoid CORS issues with fetch()
  const isProd = process.env.NODE_ENV === 'production';
  let videoUrl = `/${apiVideo.s3Key}`;
  if (isProd && cloudfrontDomain) {
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
    createdByUserId: apiHighlight.createdByUserId ?? null,
    createdByUserName: apiHighlight.createdByUser?.name ?? null,
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
    userRole: apiSession.userRole,
  };
}

// API client functions
export async function fetchSession(sessionId: string): Promise<Session> {
  const response = await fetch(`${API_BASE_URL}/v1/sessions/${sessionId}`, {
    headers: getHeaders(),
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
    headers: getHeaders('application/json'),
    body: JSON.stringify({ name }),
  });

  if (!response.ok) {
    throw new Error(`Failed to create session: ${response.status}`);
  }

  return response.json();
}

export type SessionType = 'REGULAR' | 'ALL_VIDEOS';

export interface ListSessionsResponse {
  data: Array<{
    id: string;
    name: string;
    type: SessionType;
    createdAt: string;
    updatedAt: string;
    videoCount: number;
    highlightCount: number;
  }>;
  pagination: {
    page: number;
    limit: number;
    total: number;
    totalPages: number;
  };
}

export async function listSessions(page = 1, limit = 20): Promise<ListSessionsResponse> {
  const response = await fetch(`${API_BASE_URL}/v1/sessions?page=${page}&limit=${limit}`, {
    headers: getHeaders(),
  });

  if (!response.ok) {
    throw new Error(`Failed to list sessions: ${response.status}`);
  }

  return response.json();
}

// Check if user has any content (for returning user banner)
export async function checkUserHasContent(): Promise<{ hasContent: boolean; sessionCount?: number }> {
  const response = await fetch(`${API_BASE_URL}/v1/sessions/has-content`, {
    headers: getHeaders(),
  });

  if (!response.ok) {
    return { hasContent: false };
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
    headers: getHeaders('application/json'),
    body: JSON.stringify({ operations }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error?.message || `Batch update failed: ${response.status}`);
  }

  return response.json();
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
    headers: getHeaders('application/json'),
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
    headers: getHeaders('application/json'),
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
    headers: getHeaders(),
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
  const response = await fetch(`${API_BASE_URL}/v1/videos/${videoId}/detection-status`, {
    headers: getHeaders(),
  });

  if (!response.ok) {
    throw new Error(`Failed to get detection status: ${response.status}`);
  }

  return response.json();
}

// Delete session
export async function deleteSession(sessionId: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/v1/sessions/${sessionId}`, {
    method: 'DELETE',
    headers: getHeaders(),
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
    headers: getHeaders(),
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
    headers: getHeaders('application/json'),
    body: JSON.stringify({ name }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error?.message || `Failed to rename video: ${response.status}`);
  }
}

// ============================================================================
// Video Library API (user-scoped videos)
// ============================================================================

export interface VideoSession {
  id: string;
  name: string;
  type: SessionType;
}

export interface VideoListItem {
  id: string;
  name: string;
  filename: string;
  s3Key: string;
  status: 'PENDING' | 'UPLOADED' | 'DETECTING' | 'DETECTED' | 'ERROR';
  processingStatus?: ProcessingStatus;
  durationMs: number | null;
  width: number | null;
  height: number | null;
  fileSizeBytes: string | null;
  createdAt: string;
  sessionCount: number;
  sessions?: VideoSession[];
}

export interface ListVideosResponse {
  data: VideoListItem[];
  pagination: {
    page: number;
    limit: number;
    total: number;
    totalPages: number;
  };
}

// List all videos for current user
export async function listVideos(
  page = 1,
  limit = 20,
  search?: string
): Promise<ListVideosResponse> {
  const params = new URLSearchParams({ page: String(page), limit: String(limit) });
  if (search) params.append('search', search);

  const response = await fetch(`${API_BASE_URL}/v1/videos?${params}`, {
    headers: getHeaders(),
  });

  if (!response.ok) {
    throw new Error(`Failed to list videos: ${response.status}`);
  }

  return response.json();
}

// Request upload URL for a new video (not linked to a session)
export async function requestVideoUploadUrl(params: {
  filename: string;
  contentHash: string;
  fileSize: number;
  durationMs?: number;
}): Promise<{ uploadUrl: string; videoId: string; s3Key: string }> {
  const response = await fetch(`${API_BASE_URL}/v1/videos/upload-url`, {
    method: 'POST',
    headers: getHeaders('application/json'),
    body: JSON.stringify(params),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error?.message || `Failed to get upload URL: ${response.status}`);
  }

  return response.json();
}

// Confirm video upload (for videos uploaded without a session)
export async function confirmVideoUpload(
  videoId: string,
  params: { durationMs?: number; width?: number; height?: number }
): Promise<VideoListItem> {
  const response = await fetch(`${API_BASE_URL}/v1/videos/${videoId}/confirm`, {
    method: 'POST',
    headers: getHeaders('application/json'),
    body: JSON.stringify(params),
  });

  if (!response.ok) {
    throw new Error(`Failed to confirm upload: ${response.status}`);
  }

  return response.json();
}

// Add video to session (creates junction)
export async function addVideoToSession(
  sessionId: string,
  videoId: string,
  order?: number
): Promise<void> {
  const response = await fetch(
    `${API_BASE_URL}/v1/sessions/${sessionId}/videos/${videoId}`,
    {
      method: 'POST',
      headers: getHeaders('application/json'),
      body: JSON.stringify({ order }),
    }
  );

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error?.message || `Failed to add video to session: ${response.status}`);
  }
}

// Remove video from session (removes junction only)
export async function removeVideoFromSession(
  sessionId: string,
  videoId: string
): Promise<void> {
  const response = await fetch(
    `${API_BASE_URL}/v1/sessions/${sessionId}/videos/${videoId}`,
    {
      method: 'DELETE',
      headers: getHeaders(),
    }
  );

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error?.message || `Failed to remove video from session: ${response.status}`);
  }
}

// Restore a soft-deleted video
export async function restoreVideo(videoId: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/v1/videos/${videoId}/restore`, {
    method: 'POST',
    headers: getHeaders(),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error?.message || `Failed to restore video: ${response.status}`);
  }
}

// ============================================================================
// User API
// ============================================================================

export type UserTier = 'FREE' | 'PREMIUM';

export interface TierLimits {
  detectionsPerMonth: number;
  maxVideoDurationMs: number;
  maxFileSizeBytes: number;
  monthlyUploadCount: number | null;
  exportQuality: '720p' | 'original';
  exportWatermark: boolean;
  lambdaExportEnabled: boolean;
  retentionDays: number | null;
  serverSyncEnabled: boolean;
  highlightsEnabled: boolean;
}

export interface UsageQuota {
  detectionsUsed: number;
  detectionsLimit: number;
  detectionsRemaining: number;
  uploadsThisMonth: number;
  uploadsLimit: number | null;
  uploadsRemaining: number | null;
  periodStart: string;
}

export interface UserResponse {
  id: string;
  email: string | null;
  name: string | null;
  avatarUrl: string | null;
  tier: UserTier;
  tierLimits: TierLimits;
  usage: UsageQuota;
  createdAt: string;
  convertedAt: string | null;
  videoCount: number;
  sessionCount: number;
}

// Get current user
export async function getCurrentUser(): Promise<UserResponse> {
  const response = await fetch(`${API_BASE_URL}/v1/me`, {
    headers: getHeaders(),
  });

  if (!response.ok) {
    throw new Error(`Failed to get user: ${response.status}`);
  }

  return response.json();
}

// Update current user (e.g., name)
export async function updateCurrentUser(data: { name?: string }): Promise<UserResponse> {
  const response = await fetch(`${API_BASE_URL}/v1/me`, {
    method: 'PATCH',
    headers: getHeaders('application/json'),
    body: JSON.stringify(data),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error?.message || `Failed to update user: ${response.status}`);
  }

  return response.json();
}

// ============================================================================
// Session Sharing API
// ============================================================================

export interface ShareInfo {
  token: string;
  createdAt: string;
  members: Array<{
    userId: string;
    name: string | null;
    email: string | null;
    avatarUrl: string | null;
    joinedAt: string;
  }>;
}

export interface SharePreview {
  sessionId: string;
  sessionName: string;
  ownerName: string | null;
}

export interface SharedSession {
  id: string;
  name: string;
  type: SessionType;
  createdAt: string;
  updatedAt: string;
  videoCount: number;
  highlightCount: number;
  ownerName: string | null;
  joinedAt: string;
}

// Create or get share link for a session (owner only)
export async function createShare(sessionId: string): Promise<{ token: string; createdAt: string }> {
  const response = await fetch(`${API_BASE_URL}/v1/sessions/${sessionId}/share`, {
    method: 'POST',
    headers: getHeaders(),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error?.message || `Failed to create share: ${response.status}`);
  }

  return response.json();
}

// Get share info including members (owner only)
export async function getShare(sessionId: string): Promise<ShareInfo | null> {
  const response = await fetch(`${API_BASE_URL}/v1/sessions/${sessionId}/share`, {
    headers: getHeaders(),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error?.message || `Failed to get share: ${response.status}`);
  }

  return response.json();
}

// Delete share (revokes all access)
export async function deleteShare(sessionId: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/v1/sessions/${sessionId}/share`, {
    method: 'DELETE',
    headers: getHeaders(),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error?.message || `Failed to delete share: ${response.status}`);
  }
}

// Remove a member from shared session
export async function removeShareMember(sessionId: string, userId: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/v1/sessions/${sessionId}/share/members/${userId}`, {
    method: 'DELETE',
    headers: getHeaders(),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error?.message || `Failed to remove member: ${response.status}`);
  }
}

// Get share preview (public - for accept page)
export async function getSharePreview(token: string): Promise<SharePreview> {
  const response = await fetch(`${API_BASE_URL}/v1/share/${token}`, {
    headers: getHeaders(),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error?.message || `Failed to get share preview: ${response.status}`);
  }

  return response.json();
}

// Accept a share invite
export async function acceptShare(token: string): Promise<{ sessionId: string; alreadyOwner?: boolean; alreadyMember?: boolean }> {
  const response = await fetch(`${API_BASE_URL}/v1/share/${token}/accept`, {
    method: 'POST',
    headers: getHeaders(),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error?.message || `Failed to accept share: ${response.status}`);
  }

  return response.json();
}

// List sessions shared with current user
export async function listSharedSessions(): Promise<{ data: SharedSession[] }> {
  const response = await fetch(`${API_BASE_URL}/v1/sessions/shared`, {
    headers: getHeaders(),
  });

  if (!response.ok) {
    throw new Error(`Failed to list shared sessions: ${response.status}`);
  }

  return response.json();
}

// ============================================================================
// Export Jobs API
// ============================================================================

export type ExportTier = 'FREE' | 'PREMIUM';
export type ExportStatus = 'PENDING' | 'PROCESSING' | 'COMPLETED' | 'FAILED';

export interface ExportJobResponse {
  id: string;
  status: ExportStatus;
  tier: ExportTier;
  progress: number;
  error?: string;
  createdAt: string;
  updatedAt: string;
}

export interface CreateExportJobRequest {
  sessionId: string;
  tier?: ExportTier;
  config?: {
    format?: 'mp4' | 'webm';
  };
  rallies: Array<{
    videoId: string;
    videoS3Key: string;
    startMs: number;
    endMs: number;
  }>;
}

// Create a server-side export job
export async function createExportJob(
  request: CreateExportJobRequest
): Promise<ExportJobResponse> {
  const response = await fetch(`${API_BASE_URL}/v1/export-jobs`, {
    method: 'POST',
    headers: getHeaders('application/json'),
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error?.message || `Failed to create export job: ${response.status}`);
  }

  return response.json();
}

// Get export job status
export async function getExportJobStatus(jobId: string): Promise<ExportJobResponse> {
  const response = await fetch(`${API_BASE_URL}/v1/export-jobs/${jobId}`, {
    headers: getHeaders(),
  });

  if (!response.ok) {
    throw new Error(`Failed to get export job status: ${response.status}`);
  }

  return response.json();
}

// Get export download URL
export async function getExportDownloadUrl(
  jobId: string
): Promise<{ id: string; status: ExportStatus; downloadUrl: string | null }> {
  const response = await fetch(`${API_BASE_URL}/v1/export-jobs/${jobId}/download`, {
    headers: getHeaders(),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error?.message || `Failed to get download URL: ${response.status}`);
  }

  return response.json();
}

// ============================================================================
// Rally Confirmation API (Trimmed Video Generation)
// ============================================================================

export type ConfirmationStatusType = 'PENDING' | 'PROCESSING' | 'CONFIRMED' | 'FAILED';

export interface ConfirmationStatusResponse {
  videoId: string;
  confirmation: {
    id: string;
    status: ConfirmationStatusType;
    progress: number;
    error?: string | null;
    confirmedAt?: string | null;
    originalDurationMs: number;
    trimmedDurationMs?: number | null;
    timestampMappings?: Array<{
      rallyId: string;
      originalStartMs: number;
      originalEndMs: number;
      trimmedStartMs: number;
      trimmedEndMs: number;
    }>;
  } | null;
}

export interface ConfirmationResult {
  confirmationId: string;
  status: ConfirmationStatusType;
  progress: number;
  createdAt: string;
}

// Initiate rally confirmation (generates trimmed video)
export async function confirmRallies(videoId: string): Promise<ConfirmationResult> {
  const response = await fetch(`${API_BASE_URL}/v1/videos/${videoId}/confirm-rallies`, {
    method: 'POST',
    headers: getHeaders(),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error?.message || `Failed to confirm rallies: ${response.status}`);
  }

  return response.json();
}

// Get confirmation status for a video
export async function getConfirmationStatus(videoId: string): Promise<ConfirmationStatusResponse> {
  const response = await fetch(`${API_BASE_URL}/v1/videos/${videoId}/confirmation-status`, {
    headers: getHeaders(),
  });

  if (!response.ok) {
    throw new Error(`Failed to get confirmation status: ${response.status}`);
  }

  return response.json();
}

// Restore original video (delete trimmed version)
export async function restoreOriginalVideo(videoId: string): Promise<{ success: boolean }> {
  const response = await fetch(`${API_BASE_URL}/v1/videos/${videoId}/restore-original`, {
    method: 'POST',
    headers: getHeaders(),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error?.message || `Failed to restore original: ${response.status}`);
  }

  return response.json();
}

// ============================================================================
// Feedback API
// ============================================================================

export type FeedbackType = 'BUG' | 'FEATURE' | 'FEEDBACK';

export interface SubmitFeedbackRequest {
  type: FeedbackType;
  message: string;
  email?: string;
  pageUrl?: string;
}

export interface FeedbackResponse {
  id: string;
  type: FeedbackType;
  message: string;
  createdAt: string;
}

// Submit user feedback
export async function submitFeedback(feedback: SubmitFeedbackRequest): Promise<FeedbackResponse> {
  const response = await fetch(`${API_BASE_URL}/v1/feedback`, {
    method: 'POST',
    headers: getHeaders('application/json'),
    body: JSON.stringify(feedback),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error?.message || `Failed to submit feedback: ${response.status}`);
  }

  return response.json();
}
