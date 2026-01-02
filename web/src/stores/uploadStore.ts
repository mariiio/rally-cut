import { create } from 'zustand';
import { hashFile, getVideoDuration } from '@/utils/fileHandlers';
import {
  requestUploadUrl,
  confirmUpload,
  requestVideoUploadUrl,
  confirmVideoUpload,
} from '@/services/api';

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

  // Actions
  uploadVideo: (sessionId: string, file: File) => Promise<boolean>;
  uploadVideoToLibrary: (file: File) => Promise<UploadResult>;
  cancel: () => void;
  clearError: () => void;
  reset: () => void;
}

// Progress constants
const PROGRESS_ANALYZE = 10;
const PROGRESS_UPLOAD_START = 15;
const PROGRESS_UPLOAD_RANGE = 75; // 15-90%
const PROGRESS_FINALIZE = 95;

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

export const useUploadStore = create<UploadState>((set, get) => ({
  isUploading: false,
  progress: 0,
  currentStep: '',
  error: null,
  abortController: null,

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

      // Request presigned upload URL (user-scoped, no session)
      set({ progress: PROGRESS_ANALYZE, currentStep: 'Preparing upload...' });
      const { uploadUrl, videoId } = await requestVideoUploadUrl({
        filename: file.name,
        contentHash,
        fileSize: file.size,
        durationMs,
      });

      if (abortController.signal.aborted) throw new Error('Upload cancelled');

      // Upload to S3 with progress tracking
      set({ progress: PROGRESS_UPLOAD_START, currentStep: 'Uploading video...' });
      await uploadToS3(file, uploadUrl, abortController.signal, (progress) => {
        set({ progress });
      });

      if (abortController.signal.aborted) throw new Error('Upload cancelled');

      // Confirm upload with API (user-scoped)
      set({ progress: PROGRESS_FINALIZE, currentStep: 'Finalizing...' });
      await confirmVideoUpload(videoId, { durationMs });

      set({
        isUploading: false,
        progress: 100,
        currentStep: 'Upload complete',
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
}));
