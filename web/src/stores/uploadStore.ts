import { create } from 'zustand';
import { hashFile, getVideoDuration } from '@/utils/fileHandlers';
import {
  requestUploadUrl,
  confirmUpload,
  requestVideoUploadUrl,
  confirmVideoUpload,
  initiateMultipartUpload,
  completeMultipartUpload,
  abortMultipartUpload,
} from '@/services/api';
import { useTierStore } from './tierStore';

interface UploadResult {
  success: boolean;
  videoId?: string;
}

interface UploadState {
  // Upload status
  isUploading: boolean;
  progress: number;
  currentStep: string;
  error: string | null;

  // Abort controller for cancellation
  abortController: AbortController | null;

  // Local video URLs for instant playback before processing completes
  // Maps videoId → blob URL
  localVideoUrls: Map<string, string>;

  // Actions
  uploadVideo: (sessionId: string, file: File) => Promise<boolean>;
  uploadVideoToLibrary: (file: File) => Promise<UploadResult>;
  cancel: () => void;
  clearError: () => void;
  reset: () => void;
  getLocalVideoUrl: (videoId: string) => string | undefined;
  clearLocalVideoUrl: (videoId: string) => void;
}

// Progress constants
const PROGRESS_ANALYZE = 10;
const PROGRESS_UPLOAD_START = 15;
const PROGRESS_UPLOAD_RANGE = 75; // 15-90%
const PROGRESS_FINALIZE = 95;

// Multipart upload constants
const MULTIPART_THRESHOLD = 100 * 1024 * 1024; // 100MB - use multipart above this
const MAX_CONCURRENT_PARTS = 4; // Upload 4 parts in parallel

/**
 * Format file size for display (e.g., "1.2 GB", "350 MB")
 */
function formatFileSize(bytes: number): string {
  if (bytes >= 1024 * 1024 * 1024) {
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
  }
  return `${Math.round(bytes / (1024 * 1024))} MB`;
}

/**
 * Get a friendly upload message based on progress
 */
function getUploadMessage(progress: number, fileName: string, isMultipart: boolean, partsComplete?: number, totalParts?: number): string {
  const shortName = fileName.length > 25 ? fileName.slice(0, 22) + '...' : fileName;

  if (isMultipart && partsComplete !== undefined && totalParts !== undefined) {
    // Multipart: show parts progress
    if (partsComplete === 0) {
      return `Starting upload for "${shortName}"`;
    } else if (partsComplete < totalParts) {
      return `Sending "${shortName}" (${partsComplete} of ${totalParts} parts)`;
    } else {
      return `Assembling "${shortName}"`;
    }
  }

  // Single upload: vary message by progress
  if (progress < 30) {
    return `Starting upload for "${shortName}"`;
  } else if (progress < 60) {
    return `Sending "${shortName}"`;
  } else if (progress < 85) {
    return `Almost there with "${shortName}"`;
  } else {
    return `Finishing "${shortName}"`;
  }
}

/**
 * Validate upload against tier limits.
 * Returns error message if validation fails, null if valid.
 */
function validateUploadLimits(fileSize: number, durationMs: number): string | null {
  const tierStore = useTierStore.getState();
  const { limits, usage, tier } = tierStore;

  // Check upload quota
  if (usage.uploadsRemaining !== null && usage.uploadsRemaining <= 0) {
    return `Monthly upload limit reached (${usage.uploadsThisMonth}/${usage.uploadsLimit}). Upgrade to Premium for unlimited uploads, or wait until next month.`;
  }

  // Check file size
  if (fileSize > limits.maxFileSizeBytes) {
    const fileSizeMB = Math.round(fileSize / (1024 * 1024));
    const limitMB = Math.round(limits.maxFileSizeBytes / (1024 * 1024));
    const limitStr = limitMB >= 1024 ? `${limitMB / 1024} GB` : `${limitMB} MB`;
    return `File size (${fileSizeMB} MB) exceeds ${limitStr} limit for ${tier} tier.${tier === 'FREE' ? ' Upgrade to Premium for 2 GB uploads.' : ''}`;
  }

  // Check duration
  if (durationMs > limits.maxVideoDurationMs) {
    const durationMin = Math.round(durationMs / 60000);
    const limitMin = Math.round(limits.maxVideoDurationMs / 60000);
    return `Video duration (${durationMin} min) exceeds ${limitMin} minute limit for ${tier} tier.${tier === 'FREE' ? ' Upgrade to Premium for 25 minute videos.' : ''}`;
  }

  return null;
}

/**
 * Upload a file to S3 with progress tracking.
 * Returns a promise that resolves when upload is complete.
 */
async function uploadToS3(
  file: File,
  uploadUrl: string,
  abortSignal: AbortSignal,
  onProgress: (percent: number) => void
): Promise<void> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();

    xhr.upload.addEventListener('progress', (event) => {
      if (event.lengthComputable) {
        const uploadProgress = Math.round((event.loaded / event.total) * PROGRESS_UPLOAD_RANGE);
        onProgress(PROGRESS_UPLOAD_START + uploadProgress);
      }
    });

    xhr.addEventListener('load', () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve();
      } else {
        reject(new Error(`Upload failed with status ${xhr.status}`));
      }
    });

    xhr.addEventListener('error', () => {
      reject(new Error('Network error during upload'));
    });

    xhr.addEventListener('abort', () => {
      reject(new Error('Upload cancelled'));
    });

    abortSignal.addEventListener('abort', () => {
      xhr.abort();
    });

    xhr.open('PUT', uploadUrl, true);
    xhr.setRequestHeader('Content-Type', file.type || 'video/mp4');
    xhr.send(file);
  });
}

/**
 * Upload a file using multipart upload (for large files).
 * Uploads parts in parallel for better performance.
 */
async function uploadToS3Multipart(
  file: File,
  partUrls: string[],
  partSize: number,
  abortSignal: AbortSignal,
  onProgress: (percent: number, partsComplete: number, totalParts: number) => void
): Promise<{ partNumber: number; etag: string }[]> {
  const parts: { partNumber: number; etag: string }[] = [];
  const totalParts = partUrls.length;
  let completedParts = 0;

  // Upload a single part
  const uploadPart = async (partNumber: number): Promise<void> => {
    if (abortSignal.aborted) throw new Error('Upload cancelled');

    const start = (partNumber - 1) * partSize;
    const end = Math.min(start + partSize, file.size);
    const chunk = file.slice(start, end);

    const response = await fetch(partUrls[partNumber - 1], {
      method: 'PUT',
      body: chunk,
      signal: abortSignal,
    });

    if (!response.ok) {
      throw new Error(`Part ${partNumber} upload failed with status ${response.status}`);
    }

    // Get ETag from response (required for completing multipart upload)
    const etag = response.headers.get('ETag')?.replace(/"/g, '') || '';
    parts.push({ partNumber, etag });

    completedParts++;
    const uploadProgress = Math.round((completedParts / totalParts) * PROGRESS_UPLOAD_RANGE);
    onProgress(PROGRESS_UPLOAD_START + uploadProgress, completedParts, totalParts);
  };

  // Process parts with concurrency limit
  const queue = Array.from({ length: totalParts }, (_, i) => i + 1);
  const workers = Array(MAX_CONCURRENT_PARTS)
    .fill(null)
    .map(async () => {
      while (queue.length > 0) {
        const partNumber = queue.shift()!;
        await uploadPart(partNumber);
      }
    });

  await Promise.all(workers);

  // Sort by part number (required by S3)
  return parts.sort((a, b) => a.partNumber - b.partNumber);
}

export const useUploadStore = create<UploadState>((set, get) => ({
  isUploading: false,
  progress: 0,
  currentStep: '',
  error: null,
  abortController: null,
  localVideoUrls: new Map(),

  uploadVideo: async (sessionId: string, file: File) => {
    if (get().isUploading) {
      return false;
    }

    const abortController = new AbortController();
    set({
      isUploading: true,
      progress: 0,
      currentStep: 'Analyzing video...',
      error: null,
      abortController,
    });

    try {
      // Get file hash and duration in parallel
      const [contentHash, durationMs] = await Promise.all([
        hashFile(file),
        getVideoDuration(file),
      ]);

      if (abortController.signal.aborted) throw new Error('Upload cancelled');

      // Pre-validate against tier limits
      const validationError = validateUploadLimits(file.size, durationMs);
      if (validationError) {
        throw new Error(validationError);
      }

      // Request presigned upload URL
      set({ progress: PROGRESS_ANALYZE, currentStep: 'Preparing upload...' });
      const { uploadUrl, videoId } = await requestUploadUrl(
        sessionId,
        file.name,
        contentHash,
        file.size,
        durationMs
      );

      if (abortController.signal.aborted) throw new Error('Upload cancelled');

      // Upload to S3 with progress tracking
      set({ progress: PROGRESS_UPLOAD_START, currentStep: 'Uploading video...' });
      await uploadToS3(file, uploadUrl, abortController.signal, (progress) => {
        set({ progress });
      });

      if (abortController.signal.aborted) throw new Error('Upload cancelled');

      // Confirm upload with API
      set({ progress: PROGRESS_FINALIZE, currentStep: 'Finalizing...' });
      await confirmUpload(sessionId, videoId, durationMs);

      // Store local blob URL for instant playback
      const blobUrl = URL.createObjectURL(file);
      const { localVideoUrls } = get();
      localVideoUrls.set(videoId, blobUrl);
      set({ localVideoUrls: new Map(localVideoUrls) });

      set({
        isUploading: false,
        progress: 100,
        currentStep: 'Upload complete',
        abortController: null,
      });

      return true;
    } catch (err) {
      const isCancelled = err instanceof Error && err.message === 'Upload cancelled';
      if (!isCancelled) {
        console.error('Upload error:', err);
      }
      set({
        isUploading: false,
        progress: 0,
        currentStep: '',
        error: isCancelled ? null : (err instanceof Error ? err.message : 'Upload failed'),
        abortController: null,
      });
      return false;
    }
  },

  uploadVideoToLibrary: async (file: File): Promise<UploadResult> => {
    if (get().isUploading) {
      return { success: false };
    }

    const abortController = new AbortController();
    const shortName = file.name.length > 25 ? file.name.slice(0, 22) + '...' : file.name;
    const fileSize = formatFileSize(file.size);

    set({
      isUploading: true,
      progress: 0,
      currentStep: `Reading "${shortName}" (${fileSize})`,
      error: null,
      abortController,
    });

    try {
      // Get file hash and duration in parallel
      const [contentHash, durationMs] = await Promise.all([
        hashFile(file),
        getVideoDuration(file),
      ]);

      if (abortController.signal.aborted) throw new Error('Upload cancelled');

      // Pre-validate against tier limits
      const validationError = validateUploadLimits(file.size, durationMs);
      if (validationError) {
        throw new Error(validationError);
      }

      let videoId: string;

      // Choose upload method based on file size
      if (file.size > MULTIPART_THRESHOLD) {
        // Multipart upload for large files (100MB+)
        set({ progress: PROGRESS_ANALYZE, currentStep: `Preparing "${shortName}" for upload` });
        const { videoId: vid, uploadId, partSize, partUrls } = await initiateMultipartUpload({
          filename: file.name,
          contentHash,
          fileSize: file.size,
          durationMs,
        });
        videoId = vid;

        if (abortController.signal.aborted) throw new Error('Upload cancelled');

        // Upload parts in parallel
        const totalParts = partUrls.length;
        set({
          progress: PROGRESS_UPLOAD_START,
          currentStep: getUploadMessage(PROGRESS_UPLOAD_START, file.name, true, 0, totalParts),
        });
        try {
          const parts = await uploadToS3Multipart(
            file,
            partUrls,
            partSize,
            abortController.signal,
            (progress, partsComplete, total) => set({
              progress,
              currentStep: getUploadMessage(progress, file.name, true, partsComplete, total),
            })
          );

          if (abortController.signal.aborted) throw new Error('Upload cancelled');

          // Complete the multipart upload
          set({ progress: PROGRESS_FINALIZE, currentStep: `Assembling "${shortName}"` });
          await completeMultipartUpload(videoId, uploadId, parts);
        } catch (err) {
          // Abort multipart upload on any error (cleanup S3 parts)
          await abortMultipartUpload(videoId, uploadId);
          throw err;
        }
      } else {
        // Single PUT upload for smaller files
        set({ progress: PROGRESS_ANALYZE, currentStep: `Preparing "${shortName}" for upload` });
        const { uploadUrl, videoId: vid } = await requestVideoUploadUrl({
          filename: file.name,
          contentHash,
          fileSize: file.size,
          durationMs,
        });
        videoId = vid;

        if (abortController.signal.aborted) throw new Error('Upload cancelled');

        // Upload to S3 with progress tracking
        set({
          progress: PROGRESS_UPLOAD_START,
          currentStep: getUploadMessage(PROGRESS_UPLOAD_START, file.name, false),
        });
        await uploadToS3(file, uploadUrl, abortController.signal, (progress) => {
          set({
            progress,
            currentStep: getUploadMessage(progress, file.name, false),
          });
        });

        if (abortController.signal.aborted) throw new Error('Upload cancelled');
        set({ progress: PROGRESS_FINALIZE, currentStep: `Saving "${shortName}"` });
      }

      // Confirm upload with API (user-scoped)
      await confirmVideoUpload(videoId, { durationMs });

      // Store local blob URL for instant playback
      const blobUrl = URL.createObjectURL(file);
      const { localVideoUrls } = get();
      localVideoUrls.set(videoId, blobUrl);
      set({ localVideoUrls: new Map(localVideoUrls) });

      set({
        isUploading: false,
        progress: 100,
        currentStep: `"${shortName}" uploaded successfully`,
        abortController: null,
      });

      return { success: true, videoId };
    } catch (err) {
      const isCancelled = err instanceof Error && err.message === 'Upload cancelled';
      if (!isCancelled) {
        console.error('Upload error:', err);
      }
      set({
        isUploading: false,
        progress: 0,
        currentStep: '',
        error: isCancelled ? null : (err instanceof Error ? err.message : 'Upload failed'),
        abortController: null,
      });
      return { success: false };
    }
  },

  cancel: () => {
    const { abortController } = get();
    if (abortController) {
      abortController.abort();
    }
    set({
      isUploading: false,
      progress: 0,
      currentStep: '',
      error: null,
      abortController: null,
    });
  },

  clearError: () => {
    set({ error: null });
  },

  reset: () => {
    set({
      isUploading: false,
      progress: 0,
      currentStep: '',
      error: null,
      abortController: null,
    });
  },

  getLocalVideoUrl: (videoId: string) => {
    return get().localVideoUrls.get(videoId);
  },

  clearLocalVideoUrl: (videoId: string) => {
    const { localVideoUrls } = get();
    const blobUrl = localVideoUrls.get(videoId);
    if (blobUrl) {
      URL.revokeObjectURL(blobUrl);
      localVideoUrls.delete(videoId);
      set({ localVideoUrls: new Map(localVideoUrls) });
    }
  },
}));
