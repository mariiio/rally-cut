import { create } from 'zustand';
import { hashFile, getVideoDuration } from '@/utils/fileHandlers';
import { requestUploadUrl, confirmUpload } from '@/services/api';

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
  cancel: () => void;
  clearError: () => void;
  reset: () => void;
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

      // Check if cancelled
      if (abortController.signal.aborted) {
        throw new Error('Upload cancelled');
      }

      // Request presigned upload URL
      set({ progress: 10, currentStep: 'Preparing upload...' });
      const { uploadUrl, videoId } = await requestUploadUrl(
        sessionId,
        file.name,
        contentHash,
        file.size,
        durationMs
      );

      // Check if cancelled
      if (abortController.signal.aborted) {
        throw new Error('Upload cancelled');
      }

      // Upload to S3 with progress tracking using XMLHttpRequest
      set({ progress: 15, currentStep: 'Uploading video...' });

      await new Promise<void>((resolve, reject) => {
        const xhr = new XMLHttpRequest();

        xhr.upload.addEventListener('progress', (event) => {
          if (event.lengthComputable) {
            // Map 15-90% of progress bar to actual upload progress
            const uploadProgress = Math.round((event.loaded / event.total) * 75);
            set({ progress: 15 + uploadProgress });
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

        // Store abort handler
        abortController.signal.addEventListener('abort', () => {
          xhr.abort();
        });

        xhr.open('PUT', uploadUrl, true);
        xhr.setRequestHeader('Content-Type', file.type || 'video/mp4');
        xhr.send(file);
      });

      // Check if cancelled
      if (abortController.signal.aborted) {
        throw new Error('Upload cancelled');
      }

      // Confirm upload with API
      set({ progress: 95, currentStep: 'Finalizing...' });
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
        set({
          isUploading: false,
          progress: 0,
          currentStep: '',
          error: err instanceof Error ? err.message : 'Upload failed',
          abortController: null,
        });
      } else {
        set({
          isUploading: false,
          progress: 0,
          currentStep: '',
          error: null,
          abortController: null,
        });
      }

      return false;
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
