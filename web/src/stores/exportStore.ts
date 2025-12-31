import { create } from 'zustand';
import { Rally } from '@/types/rally';
import {
  isFFmpegSupported,
  exportSingleRally,
  exportConcatenated,
  exportMultiSourceConcatenated,
  downloadBlob,
  getVideoName,
  VideoSource,
  RallyWithSource,
} from '@/utils/videoExport';

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

  // Actions
  downloadRally: (videoSource: VideoSource, rally: Rally) => Promise<void>;
  downloadAllRallies: (videoSource: VideoSource, rallies: Rally[], withFade: boolean) => Promise<void>;
  downloadHighlight: (
    ralliesWithSource: RallyWithSource[],
    highlightId: string,
    highlightName: string,
    withFade: boolean
  ) => Promise<void>;
  clearError: () => void;
  reset: () => void;
}

export const useExportStore = create<ExportState>((set, get) => ({
  isExporting: false,
  progress: 0,
  currentStep: '',
  error: null,
  exportingRallyId: null,
  exportingHighlightId: null,
  exportingAll: false,

  downloadRally: async (videoSource: VideoSource, rally: Rally) => {
    if (!isFFmpegSupported()) {
      set({ error: 'Your browser does not support video export. Please use a modern browser like Chrome, Firefox, or Edge.' });
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
      const blob = await exportSingleRally(videoSource, rally, (progress, step) => {
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
      }, 2000);
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

  downloadAllRallies: async (videoSource: VideoSource, rallies: Rally[], withFade: boolean) => {
    if (!isFFmpegSupported()) {
      set({ error: 'Your browser does not support video export. Please use a modern browser like Chrome, Firefox, or Edge.' });
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
      const blob = await exportConcatenated(videoSource, rallies, withFade, (progress, step) => {
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
      }, 2000);
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
    withFade: boolean
  ) => {
    if (!isFFmpegSupported()) {
      set({ error: 'Your browser does not support video export. Please use a modern browser like Chrome, Firefox, or Edge.' });
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
      const blob = await exportMultiSourceConcatenated(ralliesWithSource, withFade, (progress, step) => {
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
      }, 2000);
    } catch (err) {
      set({
        isExporting: false,
        error: err instanceof Error ? err.message : 'Export failed',
        exportingHighlightId: null,
      });
    }
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
