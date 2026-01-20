import { create } from 'zustand';
import { Rally } from '@/types/rally';
import {
  isFFmpegSupported,
  exportSingleRally,
  exportConcatenated,
  exportMultiSourceConcatenated,
  downloadBlob,
  downloadFromUrl,
  getVideoName,
  cancelExport,
  VideoSource,
  RallyWithSource,
} from '@/utils/videoExport';
import {
  createExportJob,
  getExportJobStatus,
  getExportDownloadUrl,
} from '@/services/api';
import { useCameraStore } from './cameraStore';

// Export options from dialog
export interface ExportOptions {
  quality: 'original' | '720p';
  applyCameraEdits: boolean;
  withFade: boolean;
}

const BROWSER_NOT_SUPPORTED_ERROR =
  'Your browser does not support video export. Please use a modern browser like Chrome, Firefox, or Edge.';

const AUTO_RESET_DELAY_MS = 2000;

// Video info needed for server-side export (distinct from rally.ts VideoMetadata)
interface ExportVideoInfo {
  id: string;
  s3Key: string;
  name: string;
}

interface ExportState {
  // Export status
  isExporting: boolean;
  progress: number;
  currentStep: string;
  error: string | null;

  // Track which item is being exported
  exportingRallyId: string | null;
  exportingHighlightId: string | null;
  exportingAll: boolean;

  // Browser-based export (legacy, for non-confirmed videos)
  downloadRally: (videoSource: VideoSource, rally: Rally, withWatermark?: boolean) => Promise<void>;
  downloadAllRallies: (videoSource: VideoSource, rallies: Rally[], withFade: boolean, withWatermark?: boolean) => Promise<void>;
  downloadHighlight: (
    ralliesWithSource: RallyWithSource[],
    highlightId: string,
    highlightName: string,
    withFade: boolean,
    withWatermark?: boolean
  ) => Promise<void>;

  // Server-side export (handles confirmed videos with reverse timestamp mapping)
  downloadRallyServerSide: (
    sessionId: string,
    video: ExportVideoInfo,
    rally: Rally,
    options?: ExportOptions
  ) => Promise<void>;
  downloadAllRalliesServerSide: (
    sessionId: string,
    video: ExportVideoInfo,
    rallies: Rally[],
    options?: ExportOptions
  ) => Promise<void>;

  cancel: () => void;
  clearError: () => void;
  reset: () => void;
}

// Cancellation flag for server-side export polling
let serverExportCancelled = false;

export const useExportStore = create<ExportState>((set, get) => ({
  isExporting: false,
  progress: 0,
  currentStep: '',
  error: null,
  exportingRallyId: null,
  exportingHighlightId: null,
  exportingAll: false,

  downloadRally: async (videoSource: VideoSource, rally: Rally, withWatermark = true) => {
    if (!isFFmpegSupported()) {
      set({ error: BROWSER_NOT_SUPPORTED_ERROR });
      return;
    }

    if (get().isExporting) {
      return;
    }

    set({
      isExporting: true,
      progress: 0,
      currentStep: 'Starting...',
      error: null,
      exportingRallyId: rally.id,
      exportingHighlightId: null,
      exportingAll: false,
    });

    try {
      const blob = await exportSingleRally(videoSource, rally, withWatermark, (progress, step) => {
        set({ progress, currentStep: step });
      });

      const baseName = getVideoName(videoSource);
      const filename = `${baseName}_${rally.id}.mp4`;
      downloadBlob(blob, filename);

      set({
        isExporting: false,
        progress: 100,
        currentStep: 'Download complete',
        exportingRallyId: null,
      });

      // Auto-reset after success
      setTimeout(() => {
        set({ progress: 0, currentStep: '' });
      }, AUTO_RESET_DELAY_MS);
    } catch (err) {
      console.error('Export error:', err);
      let errorMessage = 'Export failed';
      if (err instanceof Error) {
        errorMessage = err.message;
        // Check for common issues
        if (err.message.includes('memory') || err.message.includes('OOM')) {
          errorMessage = 'Video is too large to process in browser. Try a smaller video file.';
        }
      }
      set({
        isExporting: false,
        error: errorMessage,
        exportingRallyId: null,
      });
    }
  },

  downloadAllRallies: async (videoSource: VideoSource, rallies: Rally[], withFade: boolean, withWatermark = true) => {
    if (!isFFmpegSupported()) {
      set({ error: BROWSER_NOT_SUPPORTED_ERROR });
      return;
    }

    if (get().isExporting) {
      return;
    }

    if (rallies.length === 0) {
      set({ error: 'No rallies to export' });
      return;
    }

    set({
      isExporting: true,
      progress: 0,
      currentStep: 'Starting...',
      error: null,
      exportingRallyId: null,
      exportingHighlightId: null,
      exportingAll: true,
    });

    try {
      const blob = await exportConcatenated(videoSource, rallies, withFade, withWatermark, (progress, step) => {
        set({ progress, currentStep: step });
      });

      const baseName = getVideoName(videoSource);
      const fadeLabel = withFade ? '_fade' : '';
      const filename = `${baseName}_all_${rallies.length}rallies${fadeLabel}.mp4`;
      downloadBlob(blob, filename);

      set({
        isExporting: false,
        progress: 100,
        currentStep: 'Download complete',
        exportingAll: false,
      });

      setTimeout(() => {
        set({ progress: 0, currentStep: '' });
      }, AUTO_RESET_DELAY_MS);
    } catch (err) {
      set({
        isExporting: false,
        error: err instanceof Error ? err.message : 'Export failed',
        exportingAll: false,
      });
    }
  },

  downloadHighlight: async (
    ralliesWithSource: RallyWithSource[],
    highlightId: string,
    highlightName: string,
    withFade: boolean,
    withWatermark = true
  ) => {
    if (!isFFmpegSupported()) {
      set({ error: BROWSER_NOT_SUPPORTED_ERROR });
      return;
    }

    if (get().isExporting) {
      return;
    }

    if (ralliesWithSource.length === 0) {
      set({ error: 'No rallies in this highlight' });
      return;
    }

    set({
      isExporting: true,
      progress: 0,
      currentStep: 'Starting...',
      error: null,
      exportingRallyId: null,
      exportingHighlightId: highlightId,
      exportingAll: false,
    });

    try {
      const blob = await exportMultiSourceConcatenated(ralliesWithSource, withFade, withWatermark, (progress, step) => {
        set({ progress, currentStep: step });
      });

      // Use the first rally's video source for the filename base
      const baseName = getVideoName(ralliesWithSource[0].videoSource);
      const safeName = highlightName.replace(/[^a-zA-Z0-9]/g, '_').toLowerCase();
      const fadeLabel = withFade ? '_fade' : '';
      const filename = `${baseName}_${safeName}${fadeLabel}.mp4`;
      downloadBlob(blob, filename);

      set({
        isExporting: false,
        progress: 100,
        currentStep: 'Download complete',
        exportingHighlightId: null,
      });

      setTimeout(() => {
        set({ progress: 0, currentStep: '' });
      }, AUTO_RESET_DELAY_MS);
    } catch (err) {
      set({
        isExporting: false,
        error: err instanceof Error ? err.message : 'Export failed',
        exportingHighlightId: null,
      });
    }
  },

  // Server-side export for single rally (handles confirmed videos)
  downloadRallyServerSide: async (sessionId: string, video: ExportVideoInfo, rally: Rally, options?: ExportOptions) => {
    if (get().isExporting) {
      return;
    }

    serverExportCancelled = false;

    set({
      isExporting: true,
      progress: 0,
      currentStep: 'Submitting export job...',
      error: null,
      exportingRallyId: rally.id,
      exportingHighlightId: null,
      exportingAll: false,
    });

    try {
      // Convert rally timestamps (seconds) to milliseconds
      const startMs = Math.round(rally.start_time * 1000);
      const endMs = Math.round(rally.end_time * 1000);

      // Get camera edits if applicable
      const cameraStore = useCameraStore.getState();
      let camera: { aspectRatio: 'ORIGINAL'; keyframes: Array<{ timeOffset: number; positionX: number; positionY: number; zoom: number; rotation: number; easing: 'LINEAR' | 'EASE_IN' | 'EASE_OUT' | 'EASE_IN_OUT' }> } | undefined;

      if (options?.applyCameraEdits) {
        const globalSettings = cameraStore.getGlobalSettings(video.id);
        const rallyCameraEdit = cameraStore.cameraEdits[rally.id];

        // Only use ORIGINAL aspect ratio keyframes (VERTICAL not supported for export)
        const keyframes = rallyCameraEdit?.keyframes?.ORIGINAL ?? [];

        // Combine global settings with rally keyframes
        if (keyframes.length > 0 || cameraStore.hasGlobalSettings(video.id)) {
          camera = {
            aspectRatio: 'ORIGINAL',
            keyframes: keyframes.map(kf => ({
              timeOffset: kf.timeOffset,
              // Combine global + rally positions
              positionX: Math.max(0, Math.min(1, globalSettings.positionX + (kf.positionX - 0.5))),
              positionY: Math.max(0, Math.min(1, globalSettings.positionY + (kf.positionY - 0.5))),
              zoom: globalSettings.zoom * kf.zoom,
              rotation: globalSettings.rotation + kf.rotation,
              easing: kf.easing,
            })),
          };
          // If only global settings and no keyframes, create a single static keyframe
          if (camera.keyframes.length === 0 && cameraStore.hasGlobalSettings(video.id)) {
            camera.keyframes = [{
              timeOffset: 0,
              positionX: globalSettings.positionX,
              positionY: globalSettings.positionY,
              zoom: globalSettings.zoom,
              rotation: globalSettings.rotation,
              easing: 'LINEAR',
            }];
          }
        }
      }

      // Create export job - backend handles reverse timestamp mapping for confirmed videos
      const exportRequest = {
        sessionId,
        config: {
          format: 'mp4' as const,
          quality: options?.quality,
        },
        rallies: [{
          videoId: video.id,
          videoS3Key: video.s3Key,
          startMs,
          endMs,
          camera,
        }],
      };
      const job = await createExportJob(exportRequest);

      set({ progress: 5, currentStep: 'Processing...' });

      // Poll for completion
      const result = await pollExportJob(job.id, (progress, step) => {
        if (!serverExportCancelled) {
          set({ progress, currentStep: step });
        }
      });

      if (serverExportCancelled) {
        return;
      }

      if (result.status === 'COMPLETED' && result.downloadUrl) {
        set({ progress: 95, currentStep: 'Downloading...' });

        // Download the file
        const filename = `${video.name}_rally_${rally.id}.mp4`;
        await downloadFromUrl(result.downloadUrl, filename);

        set({
          isExporting: false,
          progress: 100,
          currentStep: 'Download complete',
          exportingRallyId: null,
        });

        setTimeout(() => {
          set({ progress: 0, currentStep: '' });
        }, AUTO_RESET_DELAY_MS);
      } else {
        throw new Error(result.error || 'Export failed');
      }
    } catch (err) {
      if (!serverExportCancelled) {
        set({
          isExporting: false,
          error: err instanceof Error ? err.message : 'Export failed',
          exportingRallyId: null,
        });
      }
    }
  },

  // Server-side export for all rallies (handles confirmed videos)
  downloadAllRalliesServerSide: async (sessionId: string, video: ExportVideoInfo, rallies: Rally[], options?: ExportOptions) => {
    if (get().isExporting) {
      return;
    }

    if (rallies.length === 0) {
      set({ error: 'No rallies to export' });
      return;
    }

    serverExportCancelled = false;

    set({
      isExporting: true,
      progress: 0,
      currentStep: 'Submitting export job...',
      error: null,
      exportingRallyId: null,
      exportingHighlightId: null,
      exportingAll: true,
    });

    try {
      // Get camera edits if applicable
      const cameraStore = useCameraStore.getState();
      const globalSettings = cameraStore.getGlobalSettings(video.id);

      // Convert rally timestamps (seconds) to milliseconds and include camera edits
      const exportRallies = rallies.map(rally => {
        let camera: { aspectRatio: 'ORIGINAL'; keyframes: Array<{ timeOffset: number; positionX: number; positionY: number; zoom: number; rotation: number; easing: 'LINEAR' | 'EASE_IN' | 'EASE_OUT' | 'EASE_IN_OUT' }> } | undefined;

        if (options?.applyCameraEdits) {
          const rallyCameraEdit = cameraStore.cameraEdits[rally.id];

          // Only use ORIGINAL aspect ratio keyframes (VERTICAL not supported for export)
          const keyframes = rallyCameraEdit?.keyframes?.ORIGINAL ?? [];

          // Combine global settings with rally keyframes
          if (keyframes.length > 0 || cameraStore.hasGlobalSettings(video.id)) {
            camera = {
              aspectRatio: 'ORIGINAL',
              keyframes: keyframes.map(kf => ({
                timeOffset: kf.timeOffset,
                // Combine global + rally positions
                positionX: Math.max(0, Math.min(1, globalSettings.positionX + (kf.positionX - 0.5))),
                positionY: Math.max(0, Math.min(1, globalSettings.positionY + (kf.positionY - 0.5))),
                zoom: globalSettings.zoom * kf.zoom,
                rotation: globalSettings.rotation + kf.rotation,
                easing: kf.easing,
              })),
            };
            // If only global settings and no keyframes, create a single static keyframe
            if (camera.keyframes.length === 0 && cameraStore.hasGlobalSettings(video.id)) {
              camera.keyframes = [{
                timeOffset: 0,
                positionX: globalSettings.positionX,
                positionY: globalSettings.positionY,
                zoom: globalSettings.zoom,
                rotation: globalSettings.rotation,
                easing: 'LINEAR',
              }];
            }
          }
        }

        return {
          videoId: video.id,
          videoS3Key: video.s3Key,
          startMs: Math.round(rally.start_time * 1000),
          endMs: Math.round(rally.end_time * 1000),
          camera,
        };
      });

      // Create export job - backend handles reverse timestamp mapping for confirmed videos
      const exportRequest = {
        sessionId,
        config: {
          format: 'mp4' as const,
          quality: options?.quality,
          withFade: options?.withFade,
        },
        rallies: exportRallies,
      };
      const job = await createExportJob(exportRequest);

      set({ progress: 5, currentStep: 'Processing...' });

      // Poll for completion
      const result = await pollExportJob(job.id, (progress, step) => {
        if (!serverExportCancelled) {
          set({ progress, currentStep: step });
        }
      });

      if (serverExportCancelled) {
        return;
      }

      if (result.status === 'COMPLETED' && result.downloadUrl) {
        set({ progress: 95, currentStep: 'Downloading...' });

        // Download the file
        const filename = `${video.name}_all_${rallies.length}rallies.mp4`;
        await downloadFromUrl(result.downloadUrl, filename);

        set({
          isExporting: false,
          progress: 100,
          currentStep: 'Download complete',
          exportingAll: false,
        });

        setTimeout(() => {
          set({ progress: 0, currentStep: '' });
        }, AUTO_RESET_DELAY_MS);
      } else {
        throw new Error(result.error || 'Export failed');
      }
    } catch (err) {
      if (!serverExportCancelled) {
        set({
          isExporting: false,
          error: err instanceof Error ? err.message : 'Export failed',
          exportingAll: false,
        });
      }
    }
  },

  cancel: () => {
    cancelExport();
    serverExportCancelled = true;
    set({
      isExporting: false,
      progress: 0,
      currentStep: '',
      error: null,
      exportingRallyId: null,
      exportingHighlightId: null,
      exportingAll: false,
    });
  },

  clearError: () => {
    set({ error: null });
  },

  reset: () => {
    set({
      isExporting: false,
      progress: 0,
      currentStep: '',
      error: null,
      exportingRallyId: null,
      exportingHighlightId: null,
      exportingAll: false,
    });
  },
}));

/**
 * Poll an export job until completion or failure
 */
async function pollExportJob(
  jobId: string,
  onProgress: (progress: number, step: string) => void,
  maxAttempts = 300, // 10 minutes at 2s intervals
  interval = 2000
): Promise<{ status: string; downloadUrl?: string; error?: string }> {
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    if (serverExportCancelled) {
      throw new Error('Export cancelled');
    }

    const job = await getExportJobStatus(jobId);

    // Map server progress (0-100) to our range (5-90)
    const mappedProgress = 5 + Math.round((job.progress / 100) * 85);
    onProgress(mappedProgress, getServerProgressMessage(job.progress));

    if (job.status === 'COMPLETED') {
      // Get download URL
      const downloadResult = await getExportDownloadUrl(jobId);
      return {
        status: 'COMPLETED',
        downloadUrl: downloadResult.downloadUrl || undefined,
      };
    }

    if (job.status === 'FAILED') {
      return { status: 'FAILED', error: job.error };
    }

    // Wait before next poll
    await new Promise((resolve) => setTimeout(resolve, interval));
  }

  return { status: 'FAILED', error: 'Export timed out' };
}

/**
 * Get progress message for server export
 */
function getServerProgressMessage(progress: number): string {
  if (progress < 10) return 'Warming up on the server...';
  if (progress < 30) return 'Downloading video clips...';
  if (progress < 60) return 'Extracting rallies...';
  if (progress < 80) return 'Processing video...';
  if (progress < 95) return 'Uploading final video...';
  return 'Finishing up...';
}
