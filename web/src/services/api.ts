/**
 * API client for RallyCut backend
 */

import { getVisitorId } from '@/utils/visitorId';
import { getAuthToken } from '@/services/authToken';
import type { RallyCameraEdit } from '@/types/camera';
import { createRallyId } from '@/utils/rallyId';
import { mapApiKeyframes } from '@/utils/cameraKeyframe';

export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001';

/**
 * Get default headers including auth token and/or X-Visitor-Id.
 * JWT Authorization header takes precedence on the server side.
 * X-Visitor-Id is still sent for backward compatibility and anonymous fallback.
 */
export function getHeaders(contentType?: string): HeadersInit {
  const headers: HeadersInit = {};

  // Add JWT token if authenticated
  const authToken = getAuthToken();
  if (authToken) {
    headers['Authorization'] = `Bearer ${authToken}`;
  }

  // Always send visitor ID for backward compat / anonymous fallback
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
 * Custom error for ACCESS_DENIED responses (403 with accessRequestable flag)
 */
export class AccessDeniedError extends Error {
  public readonly sessionName: string;
  public readonly ownerName: string | null;
  public readonly hasPendingRequest: boolean;

  constructor(sessionName: string, ownerName: string | null, hasPendingRequest: boolean) {
    super("You don't have access to this session");
    this.name = 'AccessDeniedError';
    this.sessionName = sessionName;
    this.ownerName = ownerName;
    this.hasPendingRequest = hasPendingRequest;
  }
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
  userRole?: 'owner' | 'VIEWER' | 'EDITOR' | 'ADMIN' | null;
}

interface ApiVideoCameraSettings {
  id: string;
  videoId: string;
  zoom: number;
  positionX: number;
  positionY: number;
  rotation: number;
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
  posterS3Key?: string | null;
  proxyS3Key?: string | null;
  order: number;
  rallies: ApiRally[];
  cameraSettings?: ApiVideoCameraSettings | null;
  createdAt: string;
  qualityDowngradedAt?: string | null;
  characteristicsJson?: VideoCharacteristics | null;
}

interface ApiCameraKeyframe {
  id: string;
  timeOffset: number;
  positionX: number;
  positionY: number;
  zoom: number;
  rotation?: number;  // degrees, defaults to 0
  easing: 'LINEAR' | 'EASE_IN' | 'EASE_OUT' | 'EASE_IN_OUT';
}

interface ApiCameraEdit {
  id: string;
  enabled: boolean;
  aspectRatio: 'ORIGINAL' | 'VERTICAL';
  keyframes: ApiCameraKeyframe[];
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
  cameraEdit?: ApiCameraEdit | null;
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
import type { Session, Match, Rally, Highlight, VideoMetadata, VideoCharacteristics } from '@/types/rally';

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
    id: createRallyId(videoId, apiRally.order + 1),
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

  const buildUrl = (s3Key: string): string => {
    if (isProd && cloudfrontDomain) {
      return `https://${cloudfrontDomain}/${s3Key}`;
    }
    return `/${s3Key}`;
  };

  const videoUrl = buildUrl(apiVideo.s3Key);
  const posterUrl = apiVideo.posterS3Key ? buildUrl(apiVideo.posterS3Key) : undefined;
  const proxyUrl = apiVideo.proxyS3Key ? buildUrl(apiVideo.proxyS3Key) : undefined;

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
    s3Key: apiVideo.s3Key,
    videoUrl,
    posterUrl,
    proxyUrl,
    video: videoMetadata,
    rallies,
    createdAt: apiVideo.createdAt,
    qualityDowngradedAt: apiVideo.qualityDowngradedAt,
    status: apiVideo.status,
    characteristicsJson: apiVideo.characteristicsJson ?? null,
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
        const frontendId = createRallyId(hr.rally.videoId, hr.rally.order + 1);
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
    updatedAt: apiSession.updatedAt,
  };
}

// Export camera edit types for external use
export type { ApiCameraKeyframe, ApiCameraEdit };

// Camera edit extraction result (uses RallyCameraEdit which has keyframes per aspect ratio)
export interface CameraEditMap {
  [rallyId: string]: RallyCameraEdit;
}

// Extract camera edits from API session response
function extractCameraEdits(apiSession: ApiSession): CameraEditMap {
  const result: CameraEditMap = {};

  for (const video of apiSession.videos) {
    for (const rally of video.rallies) {
      if (rally.cameraEdit && rally.cameraEdit.enabled) {
        const frontendRallyId = createRallyId(video.id, rally.order + 1);
        const aspectRatio = rally.cameraEdit.aspectRatio;
        const keyframes = mapApiKeyframes(rally.cameraEdit.keyframes);
        // Put keyframes under the correct aspect ratio key (new format)
        result[frontendRallyId] = {
          enabled: rally.cameraEdit.enabled,
          aspectRatio,
          keyframes: {
            ORIGINAL: aspectRatio === 'ORIGINAL' ? keyframes : [],
            VERTICAL: aspectRatio === 'VERTICAL' ? keyframes : [],
          },
        };
      }
    }
  }

  return result;
}

// Global camera settings map (videoId -> settings)
export interface GlobalCameraSettingsMap {
  [videoId: string]: {
    zoom: number;
    positionX: number;
    positionY: number;
    rotation: number;
  };
}

// Extract global camera settings from API session response
function extractGlobalCameraSettings(apiSession: ApiSession): GlobalCameraSettingsMap {
  const result: GlobalCameraSettingsMap = {};

  for (const video of apiSession.videos) {
    if (video.cameraSettings) {
      result[video.id] = {
        zoom: video.cameraSettings.zoom,
        positionX: video.cameraSettings.positionX,
        positionY: video.cameraSettings.positionY,
        rotation: video.cameraSettings.rotation,
      };
    }
  }

  return result;
}

// Session fetch result including camera edits
export interface FetchSessionResult {
  session: Session;
  cameraEdits: CameraEditMap;
  globalCameraSettings: GlobalCameraSettingsMap;
}

// API client functions
export async function fetchSession(sessionId: string): Promise<FetchSessionResult> {
  const response = await fetch(`${API_BASE_URL}/v1/sessions/${sessionId}`, {
    headers: getHeaders(),
    credentials: 'include', // Include cookies for CloudFront signed cookies
  });

  if (!response.ok) {
    // Handle 403 ACCESS_DENIED specially
    if (response.status === 403) {
      const errorData = await response.json().catch(() => null);
      if (errorData?.error?.code === 'ACCESS_DENIED' && errorData?.error?.details?.accessRequestable) {
        throw new AccessDeniedError(
          errorData.error.details.sessionName,
          errorData.error.details.ownerName,
          errorData.error.details.hasPendingRequest
        );
      }
    }
    throw new Error(`Failed to fetch session: ${response.status}`);
  }

  const apiSession: ApiSession = await response.json();

  // Get CloudFront domain from env if available
  const cloudfrontDomain = process.env.NEXT_PUBLIC_CLOUDFRONT_DOMAIN;

  // Extract camera edits and global settings before transforming
  const cameraEdits = extractCameraEdits(apiSession);
  const globalCameraSettings = extractGlobalCameraSettings(apiSession);

  return {
    session: apiSessionToFrontend(apiSession, cloudfrontDomain),
    cameraEdits,
    globalCameraSettings,
  };
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
    videoPreviews: Array<{ id: string; posterS3Key: string | null }>;
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
export async function triggerRallyDetection(
  videoId: string,
  model?: 'indoor' | 'beach'
): Promise<{ jobId: string; status: string }> {
  const response = await fetch(`${API_BASE_URL}/v1/videos/${videoId}/detect-rallies`, {
    method: 'POST',
    headers: getHeaders(),
    body: model ? JSON.stringify({ model }) : undefined,
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

// Update session
export async function updateSession(
  sessionId: string,
  data: { name?: string }
): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/v1/sessions/${sessionId}`, {
    method: 'PATCH',
    headers: getHeaders('application/json'),
    body: JSON.stringify(data),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(
      error.error?.message || `Failed to update session: ${response.status}`
    );
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
  posterS3Key?: string | null;
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

// ============================================================================
// Single Video Editor
// ============================================================================

// API response for single video editor
interface ApiVideoEditorResponse {
  video: {
    id: string;
    name: string;
    filename: string;
    s3Key: string;
    posterS3Key?: string | null;
    proxyS3Key?: string | null;
    processedS3Key?: string | null;
    durationMs: number | null;
    width: number | null;
    height: number | null;
    rallies: ApiRally[];
    status: 'PENDING' | 'UPLOADED' | 'DETECTING' | 'DETECTED' | 'ERROR';
    cameraSettings?: ApiVideoCameraSettings | null;
    courtCalibrationJson?: Array<{ x: number; y: number }> | null;
    characteristicsJson?: VideoCharacteristics | null;
  };
  allVideosSessionId: string;
  highlights: ApiHighlight[];
}

// Re-export for convenience
export type { VideoCharacteristics } from '@/types/rally';

// Result type for fetchVideoForEditor
export interface FetchVideoEditorResult {
  video: {
    id: string;
    name: string;
    filename: string;
    s3Key: string;
    posterS3Key?: string | null;
    proxyS3Key?: string | null;
    processedS3Key?: string | null;
    durationMs: number | null;
    width: number | null;
    height: number | null;
  };
  match: Match;
  highlights: Highlight[];
  allVideosSessionId: string;
  cameraEdits: CameraEditMap;
  globalCameraSettings: GlobalCameraSettingsMap;
  courtCalibration: Array<{ x: number; y: number }> | null;
  characteristicsJson: VideoCharacteristics | null;
}

/**
 * Fetch video data for single-video editor mode.
 * Transforms API response to frontend format with a single Match.
 */
export async function fetchVideoForEditor(videoId: string): Promise<FetchVideoEditorResult> {
  const response = await fetch(`${API_BASE_URL}/v1/videos/${videoId}/editor`, {
    headers: getHeaders(),
    credentials: 'include',
  });

  if (!response.ok) {
    throw new Error(`Failed to fetch video for editor: ${response.status}`);
  }

  const data: ApiVideoEditorResponse = await response.json();
  const cloudfrontDomain = process.env.NEXT_PUBLIC_CLOUDFRONT_DOMAIN;

  // Build video URLs
  const getUrl = (s3Key: string | null | undefined): string | undefined => {
    if (!s3Key) return undefined;
    return cloudfrontDomain ? `https://${cloudfrontDomain}/${s3Key}` : `/${s3Key}`;
  };

  // Default FPS to 30 (not available from API)
  const fps = 30;

  // Transform rallies to frontend format
  const rallies: Rally[] = data.video.rallies.map((apiRally) => {
    const startTime = apiRally.startMs / 1000;
    const endTime = apiRally.endMs / 1000;
    const duration = endTime - startTime;

    return {
      id: createRallyId(videoId, apiRally.order + 1),
      _backendId: apiRally.id,
      start_time: startTime,
      end_time: endTime,
      start_frame: Math.round(startTime * fps),
      end_frame: Math.round(endTime * fps),
      duration,
      type: 'rally' as const,
      thumbnail_time: startTime + duration / 2,
    };
  });

  // Extract camera edits
  const cameraEdits: CameraEditMap = {};
  for (const rally of data.video.rallies) {
    if (rally.cameraEdit && rally.cameraEdit.enabled) {
      const frontendRallyId = createRallyId(videoId, rally.order + 1);
      const aspectRatio = rally.cameraEdit.aspectRatio;
      const keyframes = mapApiKeyframes(rally.cameraEdit.keyframes);
      // Put keyframes under the correct aspect ratio key (new format)
      cameraEdits[frontendRallyId] = {
        enabled: rally.cameraEdit.enabled,
        aspectRatio,
        keyframes: {
          ORIGINAL: aspectRatio === 'ORIGINAL' ? keyframes : [],
          VERTICAL: aspectRatio === 'VERTICAL' ? keyframes : [],
        },
      };
    }
  }

  // Build Match object
  const duration = (data.video.durationMs ?? 0) / 1000;
  const match: Match = {
    id: videoId,
    name: data.video.name,
    s3Key: data.video.s3Key,
    videoUrl: getUrl(data.video.processedS3Key) || getUrl(data.video.s3Key) || '',
    proxyUrl: getUrl(data.video.proxyS3Key),
    posterUrl: getUrl(data.video.posterS3Key),
    video: {
      path: data.video.s3Key,
      fps,
      duration,
      width: data.video.width ?? 1920,
      height: data.video.height ?? 1080,
      frame_count: Math.round(duration * fps),
    },
    rallies,
    status: data.video.status,
    characteristicsJson: data.video.characteristicsJson ?? null,
  };

  // Transform highlights to frontend format
  // Build a map of backend rally IDs to frontend rally IDs for this video
  const backendToFrontendRallyId: Record<string, string> = {};
  for (const rally of data.video.rallies) {
    backendToFrontendRallyId[rally.id] = createRallyId(videoId, rally.order + 1);
  }

  const highlights: Highlight[] = data.highlights.map((apiHighlight) => {
    const rallyBackendIds: Record<string, string> = {};
    const rallyIds: string[] = [];

    for (const hr of apiHighlight.highlightRallies) {
      const frontendRallyId = backendToFrontendRallyId[hr.rallyId];
      if (frontendRallyId) {
        rallyIds.push(frontendRallyId);
        rallyBackendIds[frontendRallyId] = hr.id;
      }
    }

    return {
      id: `highlight_${apiHighlight.id}`,
      _backendId: apiHighlight.id,
      _rallyBackendIds: rallyBackendIds,
      name: apiHighlight.name,
      color: apiHighlight.color,
      rallyIds,
      createdAt: Date.now(),
      createdByUserId: apiHighlight.createdByUserId ?? null,
      createdByUserName: apiHighlight.createdByUser?.name ?? null,
    };
  });

  // Extract global camera settings for this video
  const globalCameraSettings: GlobalCameraSettingsMap = {};
  if (data.video.cameraSettings) {
    globalCameraSettings[videoId] = {
      zoom: data.video.cameraSettings.zoom,
      positionX: data.video.cameraSettings.positionX,
      positionY: data.video.cameraSettings.positionY,
      rotation: data.video.cameraSettings.rotation,
    };
  }

  return {
    video: {
      id: data.video.id,
      name: data.video.name,
      filename: data.video.filename,
      s3Key: data.video.s3Key,
      posterS3Key: data.video.posterS3Key,
      proxyS3Key: data.video.proxyS3Key,
      processedS3Key: data.video.processedS3Key,
      durationMs: data.video.durationMs,
      width: data.video.width,
      height: data.video.height,
    },
    match,
    highlights,
    allVideosSessionId: data.allVideosSessionId,
    cameraEdits,
    globalCameraSettings,
    courtCalibration: data.video.courtCalibrationJson ?? null,
    characteristicsJson: data.video.characteristicsJson ?? null,
  };
}

// Request upload URL for a new video (not linked to a session)
export async function requestVideoUploadUrl(params: {
  filename: string;
  contentHash: string;
  fileSize: number;
  durationMs?: number;
}): Promise<{ uploadUrl: string | null; videoId: string; s3Key: string; alreadyExists?: boolean }> {
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

// ============================================================================
// Multipart Upload (for large files)
// ============================================================================

export interface InitiateMultipartResponse {
  videoId: string;
  s3Key: string;
  uploadId: string | null;
  partSize: number;
  partUrls: string[];
  alreadyExists?: boolean;
}

// Initiate multipart upload - returns presigned URLs for all parts
export async function initiateMultipartUpload(params: {
  filename: string;
  contentHash: string;
  fileSize: number;
  durationMs?: number;
}): Promise<InitiateMultipartResponse> {
  const response = await fetch(`${API_BASE_URL}/v1/videos/multipart/init`, {
    method: 'POST',
    headers: getHeaders('application/json'),
    body: JSON.stringify(params),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error?.message || `Failed to initiate multipart upload: ${response.status}`);
  }

  return response.json();
}

// Complete multipart upload by combining all parts
export async function completeMultipartUpload(
  videoId: string,
  uploadId: string,
  parts: { partNumber: number; etag: string }[]
): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/v1/videos/${videoId}/multipart/complete`, {
    method: 'POST',
    headers: getHeaders('application/json'),
    body: JSON.stringify({ uploadId, parts }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error?.message || `Failed to complete multipart upload: ${response.status}`);
  }
}

// Abort multipart upload and cleanup parts
export async function abortMultipartUpload(
  videoId: string,
  uploadId: string
): Promise<void> {
  // Best effort - don't throw on error
  try {
    await fetch(`${API_BASE_URL}/v1/videos/${videoId}/multipart/abort`, {
      method: 'POST',
      headers: getHeaders('application/json'),
      body: JSON.stringify({ uploadId }),
    });
  } catch {
    // Ignore errors on abort - cleanup will happen via S3 lifecycle
  }
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

export type UserTier = 'FREE' | 'PRO' | 'ELITE';

export interface TierLimits {
  detectionsPerMonth: number;
  maxVideoDurationMs: number;
  maxFileSizeBytes: number;
  monthlyUploadCount: number;
  storageCapBytes: number;
  exportQuality: '720p' | 'original';
  exportWatermark: boolean;
  lambdaExportEnabled: boolean;
  originalQualityDays: number | null;
  inactivityDeleteDays: number | null;
  serverSyncEnabled: boolean;
  highlightsEnabled: boolean;
}

export interface UsageQuota {
  detectionsUsed: number;
  detectionsLimit: number;
  detectionsRemaining: number;
  uploadsThisMonth: number;
  uploadsLimit: number;
  uploadsRemaining: number;
  storageUsedBytes: number;
  storageLimitBytes: number;
  storageRemainingBytes: number;
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

// User cache for getCurrentUser (5 minute TTL)
let cachedUser: UserResponse | null = null;
let userCacheTimestamp = 0;
const USER_CACHE_TTL = 5 * 60 * 1000; // 5 minutes

// Invalidate user cache (call after user updates)
export function invalidateUserCache(): void {
  cachedUser = null;
  userCacheTimestamp = 0;
}

// Get current user (cached)
export async function getCurrentUser(): Promise<UserResponse> {
  // Return cached user if still valid
  if (cachedUser && Date.now() - userCacheTimestamp < USER_CACHE_TTL) {
    return cachedUser;
  }

  const response = await fetch(`${API_BASE_URL}/v1/me`, {
    headers: getHeaders(),
  });

  if (!response.ok) {
    throw new Error(`Failed to get user: ${response.status}`);
  }

  const user = await response.json();
  cachedUser = user;
  userCacheTimestamp = Date.now();
  return user;
}

// Update current user (e.g., name)
export async function updateCurrentUser(data: { name?: string; avatarUrl?: string | null }): Promise<UserResponse> {
  const response = await fetch(`${API_BASE_URL}/v1/me`, {
    method: 'PATCH',
    headers: getHeaders('application/json'),
    body: JSON.stringify(data),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error?.message || `Failed to update user: ${response.status}`);
  }

  const user = await response.json();
  // Update cache with new user data
  cachedUser = user;
  userCacheTimestamp = Date.now();
  return user;
}

// ============================================================================
// Session Sharing API
// ============================================================================

export type MemberRole = 'VIEWER' | 'EDITOR' | 'ADMIN';

export interface ShareLink {
  token: string;
  role: MemberRole;
  createdAt: string;
}

export interface ShareInfo {
  shares: ShareLink[];
  members: Array<{
    userId: string;
    name: string | null;
    email: string | null;
    avatarUrl: string | null;
    role: MemberRole;
    joinedAt: string;
  }>;
}

export interface SharePreview {
  type: 'session' | 'video';
  sessionId?: string;
  sessionName?: string;
  videoId?: string;
  videoName?: string;
  ownerName: string | null;
  role: MemberRole;
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
  role: MemberRole;
}

// Create share links for a session (owner or admin) - creates all 3 role links
export async function createShare(sessionId: string): Promise<{ shares: ShareLink[] }> {
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

// Get share info including members (owner or admin)
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

// Update a member's role
export async function updateMemberRole(
  sessionId: string,
  userId: string,
  role: MemberRole
): Promise<{ role: MemberRole }> {
  const response = await fetch(`${API_BASE_URL}/v1/sessions/${sessionId}/share/members/${userId}/role`, {
    method: 'PATCH',
    headers: getHeaders('application/json'),
    body: JSON.stringify({ role }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error?.message || `Failed to update member role: ${response.status}`);
  }

  return response.json();
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

// Accept a share invite (optionally with a display name)
export async function acceptShare(token: string, name?: string): Promise<{ type?: 'session' | 'video'; sessionId?: string; videoId?: string; alreadyOwner?: boolean; alreadyMember?: boolean; role?: MemberRole }> {
  const response = await fetch(`${API_BASE_URL}/v1/share/${token}/accept`, {
    method: 'POST',
    headers: {
      ...getHeaders(),
      'Content-Type': 'application/json',
    },
    body: name ? JSON.stringify({ name }) : undefined,
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
// Video Sharing API
// ============================================================================

export interface VideoShareInfo {
  shares: ShareLink[];
  members: Array<{
    userId: string;
    name: string | null;
    email: string | null;
    avatarUrl: string | null;
    role: MemberRole;
    joinedAt: string;
  }>;
}

// Create share links for a video (owner or admin) - creates all 3 role links
export async function createVideoShare(videoId: string): Promise<{ shares: ShareLink[] }> {
  const response = await fetch(`${API_BASE_URL}/v1/videos/${videoId}/share`, {
    method: 'POST',
    headers: getHeaders(),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error?.message || `Failed to create share: ${response.status}`);
  }

  return response.json();
}

// Get video share info including members (owner or admin)
export async function getVideoShare(videoId: string): Promise<VideoShareInfo | null> {
  const response = await fetch(`${API_BASE_URL}/v1/videos/${videoId}/share`, {
    headers: getHeaders(),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error?.message || `Failed to get share: ${response.status}`);
  }

  return response.json();
}

// Delete video share (revokes all access)
export async function deleteVideoShare(videoId: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/v1/videos/${videoId}/share`, {
    method: 'DELETE',
    headers: getHeaders(),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error?.message || `Failed to delete share: ${response.status}`);
  }
}

// Remove a member from shared video
export async function removeVideoShareMember(videoId: string, userId: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/v1/videos/${videoId}/share/members/${userId}`, {
    method: 'DELETE',
    headers: getHeaders(),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error?.message || `Failed to remove member: ${response.status}`);
  }
}

// Update a video member's role
export async function updateVideoMemberRole(
  videoId: string,
  userId: string,
  role: MemberRole
): Promise<{ role: MemberRole }> {
  const response = await fetch(`${API_BASE_URL}/v1/videos/${videoId}/share/members/${userId}/role`, {
    method: 'PATCH',
    headers: getHeaders('application/json'),
    body: JSON.stringify({ role }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error?.message || `Failed to update member role: ${response.status}`);
  }

  return response.json();
}

// ============================================================================
// Access Requests API
// ============================================================================

export interface AccessRequest {
  id: string;
  userId: string;
  userName: string | null;
  userEmail: string | null;
  userAvatarUrl: string | null;
  message: string | null;
  requestedAt: string;
}

// Create an access request for a session
export async function requestAccess(
  sessionId: string,
  message?: string
): Promise<{ id: string; status: string; requestedAt: string }> {
  const response = await fetch(`${API_BASE_URL}/v1/sessions/${sessionId}/access-requests`, {
    method: 'POST',
    headers: getHeaders('application/json'),
    body: JSON.stringify({ message }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error?.message || `Failed to request access: ${response.status}`);
  }

  return response.json();
}

// Get pending access requests for a session (owner or admin)
export async function getAccessRequests(sessionId: string): Promise<{ requests: AccessRequest[] }> {
  const response = await fetch(`${API_BASE_URL}/v1/sessions/${sessionId}/access-requests`, {
    headers: getHeaders(),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error?.message || `Failed to get access requests: ${response.status}`);
  }

  return response.json();
}

// Get count of pending access requests (owner or admin)
export async function getAccessRequestsCount(sessionId: string): Promise<{ pending: number }> {
  const response = await fetch(`${API_BASE_URL}/v1/sessions/${sessionId}/access-requests/count`, {
    headers: getHeaders(),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error?.message || `Failed to get access request count: ${response.status}`);
  }

  return response.json();
}

// Accept an access request (owner or admin)
export async function acceptAccessRequest(
  sessionId: string,
  requestId: string
): Promise<{ success: boolean }> {
  const response = await fetch(`${API_BASE_URL}/v1/sessions/${sessionId}/access-requests/${requestId}/accept`, {
    method: 'POST',
    headers: getHeaders(),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error?.message || `Failed to accept access request: ${response.status}`);
  }

  return response.json();
}

// Reject an access request (owner or admin)
export async function rejectAccessRequest(
  sessionId: string,
  requestId: string
): Promise<{ success: boolean }> {
  const response = await fetch(`${API_BASE_URL}/v1/sessions/${sessionId}/access-requests/${requestId}/reject`, {
    method: 'POST',
    headers: getHeaders(),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error?.message || `Failed to reject access request: ${response.status}`);
  }

  return response.json();
}

// ============================================================================
// Export Jobs API
// ============================================================================

export type ExportTier = 'FREE' | 'PRO' | 'ELITE';
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

// Camera keyframe for export (matches backend schema)
export interface ExportCameraKeyframe {
  timeOffset: number;
  positionX: number;
  positionY: number;
  zoom: number;
  rotation: number;
  easing: 'LINEAR' | 'EASE_IN' | 'EASE_OUT' | 'EASE_IN_OUT';
}

// Camera edit configuration for export
export interface ExportCameraEdit {
  aspectRatio: 'ORIGINAL' | 'VERTICAL';
  keyframes: ExportCameraKeyframe[];
}

export interface CreateExportJobRequest {
  sessionId: string;
  tier?: ExportTier;
  config?: {
    format?: 'mp4' | 'webm';
    quality?: 'original' | '720p';
  };
  rallies: Array<{
    videoId: string;
    videoS3Key: string;
    startMs: number;
    endMs: number;
    camera?: ExportCameraEdit;
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

// ============================================================================
// Court Calibration
// ============================================================================

/**
 * Save court calibration corners for a video.
 */
export async function saveCourtCalibration(
  videoId: string,
  corners: Array<{ x: number; y: number }>
): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/v1/videos/${videoId}/court-calibration`, {
    method: 'PUT',
    headers: getHeaders('application/json'),
    body: JSON.stringify({ corners }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error?.message || `Failed to save court calibration: ${response.status}`);
  }
}

/**
 * Delete court calibration for a video.
 */
export async function deleteCourtCalibration(videoId: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/v1/videos/${videoId}/court-calibration`, {
    method: 'DELETE',
    headers: getHeaders(),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error?.message || `Failed to delete court calibration: ${response.status}`);
  }
}

// ============================================================================
// Player Tracking
// ============================================================================

// Ball position from player tracking
export interface BallPosition {
  frameNumber: number;
  x: number;
  y: number;
  confidence: number;
}

export interface PlayerDetection {
  trackId: number;
  x: number;
  y: number;
  width: number;
  height: number;
  confidence: number;
}

export interface PlayerFrameData {
  frameNumber: number;
  players: PlayerDetection[];
}

// Player position from tracking API
export interface PlayerPosition {
  frameNumber: number;
  trackId: number;
  x: number;
  y: number;
  width: number;
  height: number;
  confidence: number;
}

// Contact detection from ball trajectory inflection points
export interface ContactInfo {
  frame: number;
  ballX: number;
  ballY: number;
  velocity: number;
  directionChangeDeg: number;
  playerTrackId: number;
  playerDistance: number;
  courtSide: string;
  isAtNet: boolean;
  isValidated: boolean;
}

export interface ContactsData {
  numContacts: number;
  netY: number;
  rallyStartFrame: number;
  contacts: ContactInfo[];
}

// Action ground truth label
export interface ActionGroundTruthLabel {
  frame: number;
  action: 'serve' | 'receive' | 'set' | 'attack' | 'block' | 'dig';
  playerTrackId: number;
  ballX?: number;
  ballY?: number;
}

// Action classification from contact sequence
export interface ActionInfo {
  action: string;  // "serve", "receive", "set", "attack", "block", "dig", "unknown"
  frame: number;
  ballX: number;
  ballY: number;
  velocity: number;
  playerTrackId: number;
  courtSide: string;
  confidence: number;
}

export interface ActionsData {
  rallyId: string;
  numContacts: number;
  actionSequence: string[];
  actions: ActionInfo[];
}

export interface QualityReport {
  ballDetectionRate: number;
  ballTrajectorySpread: number;
  avgDetectionsPerFrame: number;
  primaryTrackCount: number;
  trackCreationRate: number;
  trackDestructionRate: number;
  avgTrackLifespanFrames: number;
  idSwitchCount: number;
  colorSplitCount: number;
  swapFixCount: number;
  appearanceLinkCount: number;
  uniqueRawTrackCount: number;
  calibrationRecommended: boolean;
  courtDetected: boolean;
  courtConfidence: number;
  trackabilityScore: number;
  suggestions: string[];
}

export interface TrackPlayersResponse {
  status: 'completed' | 'failed';
  frameCount?: number;
  fps?: number;  // Actual fps from tracked video
  detectionRate?: number;
  avgConfidence?: number;
  avgPlayerCount?: number;
  uniqueTrackCount?: number;
  processingTimeMs?: number;
  error?: string;
  courtSplitY?: number;
  primaryTrackIds?: number[];
  // Positions included for immediate display
  positions?: PlayerPosition[];
  // Ball positions for trajectory overlay
  ballPositions?: BallPosition[];
  // Contact detection and action classification
  contacts?: ContactsData;
  actions?: ActionsData;
  qualityReport?: QualityReport;
}

export interface GetPlayerTrackResponse {
  status: 'completed' | 'failed' | 'not_found';
  frameCount?: number;
  fps?: number;
  detectionRate?: number;
  avgConfidence?: number;
  avgPlayerCount?: number;
  uniqueTrackCount?: number;
  courtSplitY?: number;
  primaryTrackIds?: number[];
  positions?: PlayerPosition[];
  ballPositions?: BallPosition[];
  contacts?: ContactsData;
  actions?: ActionsData;
  qualityReport?: QualityReport;
  error?: string;
}

// Calibration corners for court projection
export interface CalibrationCorner {
  x: number;
  y: number;
}

/**
 * Track players in a rally using YOLOv8 + ByteTrack.
 *
 * @param rallyId - Backend rally ID (UUID)
 * @param calibrationCorners - Optional court calibration corners (4 points)
 * @returns Player tracking result with per-frame detections
 */
export async function trackPlayers(
  rallyId: string,
  calibrationCorners?: CalibrationCorner[]
): Promise<TrackPlayersResponse> {
  const response = await fetch(`${API_BASE_URL}/v1/rallies/${rallyId}/track-players`, {
    method: 'POST',
    headers: getHeaders('application/json'),
    body: JSON.stringify({ calibrationCorners }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error?.message || `Player tracking failed: ${response.status}`);
  }

  return response.json();
}

/**
 * Get existing player tracking data for a rally.
 *
 * @param rallyId - Backend rally ID (UUID)
 * @returns Player tracking result if exists
 */
export async function getPlayerTrack(rallyId: string): Promise<GetPlayerTrackResponse> {
  const response = await fetch(`${API_BASE_URL}/v1/rallies/${rallyId}/player-track`, {
    method: 'GET',
    headers: getHeaders(),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error?.message || `Failed to get player track: ${response.status}`);
  }

  return response.json();
}

/**
 * Swap two player tracks from a given frame onward.
 */
export async function swapPlayerTracks(
  rallyId: string,
  trackA: number,
  trackB: number,
  fromFrame: number,
): Promise<{ swappedCount: number }> {
  const response = await fetch(`${API_BASE_URL}/v1/rallies/${rallyId}/player-track/swap`, {
    method: 'POST',
    headers: getHeaders('application/json'),
    body: JSON.stringify({ trackA, trackB, fromFrame }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error?.message || `Failed to swap player tracks: ${response.status}`);
  }

  return response.json();
}

// ============================================================================
// Label Studio Integration (Ground Truth Labeling)
// ============================================================================

export interface LabelStudioStatus {
  hasTrackingData: boolean;
  hasGroundTruth: boolean;
  taskId?: number;
  syncedAt?: string;
}

export interface LabelStudioExportResult {
  success: boolean;
  taskId?: number;
  projectId?: number;
  taskUrl?: string;
  error?: string;
}

export interface LabelStudioImportResult {
  success: boolean;
  playerCount?: number;
  ballCount?: number;
  frameCount?: number;
  error?: string;
}

/**
 * Get Label Studio integration status for a rally.
 *
 * @param rallyId - Backend rally ID (UUID)
 * @returns Status including whether tracking/ground truth exists
 */
export async function getLabelStudioStatus(rallyId: string): Promise<LabelStudioStatus> {
  const response = await fetch(`${API_BASE_URL}/v1/rallies/${rallyId}/label-studio`, {
    headers: getHeaders(),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error?.message || `Failed to get Label Studio status: ${response.status}`);
  }

  return response.json();
}

/**
 * Export tracking predictions to Label Studio for labeling.
 * Creates a task with pre-filled annotations from current tracking.
 *
 * @param rallyId - Backend rally ID (UUID)
 * @param videoUrl - Public URL of the video for Label Studio to access
 * @param options - Optional configuration including Label Studio API settings and forceRegenerate flag
 * @returns Export result with task URL
 */
export async function exportToLabelStudio(
  rallyId: string,
  videoUrl: string,
  options?: { apiKey?: string; apiUrl?: string; forceRegenerate?: boolean }
): Promise<LabelStudioExportResult> {
  const response = await fetch(`${API_BASE_URL}/v1/rallies/${rallyId}/label-studio/export`, {
    method: 'POST',
    headers: getHeaders('application/json'),
    body: JSON.stringify({ videoUrl, ...options }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error?.message || `Failed to export to Label Studio: ${response.status}`);
  }

  return response.json();
}

/**
 * Import corrected annotations from Label Studio as ground truth.
 *
 * @param rallyId - Backend rally ID (UUID)
 * @param taskId - Label Studio task ID to import from
 * @param config - Optional Label Studio API configuration
 * @returns Import result with counts
 */
export async function importFromLabelStudio(
  rallyId: string,
  taskId: number,
  config?: { apiKey?: string; apiUrl?: string }
): Promise<LabelStudioImportResult> {
  const response = await fetch(`${API_BASE_URL}/v1/rallies/${rallyId}/label-studio/import`, {
    method: 'POST',
    headers: getHeaders('application/json'),
    body: JSON.stringify({ taskId, ...config }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error?.message || `Failed to import from Label Studio: ${response.status}`);
  }

  return response.json();
}

// ============================================================================
// Action Ground Truth
// ============================================================================

/**
 * Get action ground truth labels for a rally.
 */
export async function getActionGroundTruth(
  rallyId: string,
): Promise<{ labels: ActionGroundTruthLabel[] }> {
  const response = await fetch(`${API_BASE_URL}/v1/rallies/${rallyId}/action-ground-truth`, {
    headers: getHeaders(),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error?.message || `Failed to get action ground truth: ${response.status}`);
  }

  return response.json();
}

/**
 * Save action ground truth labels for a rally.
 */
export async function saveActionGroundTruth(
  rallyId: string,
  labels: ActionGroundTruthLabel[],
): Promise<{ savedCount: number }> {
  const response = await fetch(`${API_BASE_URL}/v1/rallies/${rallyId}/action-ground-truth`, {
    method: 'PUT',
    headers: getHeaders('application/json'),
    body: JSON.stringify({ labels }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error?.message || `Failed to save action ground truth: ${response.status}`);
  }

  return response.json();
}
