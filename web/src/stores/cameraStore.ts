import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import {
  AspectRatio,
  CameraKeyframe,
  RallyCameraEdit,
  CameraState,
  DEFAULT_CAMERA_EDIT,
  DEFAULT_KEYFRAME,
  DEFAULT_CAMERA_STATE,
  migrateCameraEdit,
  HandheldPreset,
} from '@/types/camera';
import { interpolateCameraState } from '@/utils/cameraInterpolation';
import { syncService } from '@/services/syncService';

// Lazy reference to editor store to avoid circular dependency
// The store will be accessed at runtime when pushHistory is called
let editorStoreRef: { pushHistory: () => void } | null = null;

const pushEditorHistory = () => {
  // Lazy load editor store reference on first use
  if (!editorStoreRef) {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const { useEditorStore } = require('./editorStore');
    editorStoreRef = {
      pushHistory: () => useEditorStore.getState().pushHistory(),
    };
  }
  editorStoreRef.pushHistory();
};

// Use object instead of Map for better shallow comparison
type CameraEditsRecord = Record<string, RallyCameraEdit>;

// Legacy format from API - keyframes as flat array (will be migrated)
type LegacyCameraEdit = {
  enabled: boolean;
  aspectRatio: AspectRatio;
  keyframes: CameraKeyframe[];
};
type LegacyCameraEditsRecord = Record<string, LegacyCameraEdit | RallyCameraEdit>;

interface CameraStoreState {
  // Per-rally camera edits (rallyId -> edit)
  cameraEdits: CameraEditsRecord;

  // Handheld motion preset (global, applies to all rallies)
  handheldPreset: HandheldPreset;

  // UI state
  selectedKeyframeId: string | null;
  dragPosition: { x: number; y: number } | null; // Position during active drag

  // Actions
  setCameraEdit: (rallyId: string, edit: RallyCameraEdit) => void;
  setAspectRatio: (rallyId: string, ratio: AspectRatio) => void;

  // Keyframe actions
  addKeyframe: (rallyId: string, keyframe: Omit<CameraKeyframe, 'id'>) => string;
  updateKeyframe: (rallyId: string, keyframeId: string, updates: Partial<CameraKeyframe>) => void;
  removeKeyframe: (rallyId: string, keyframeId: string) => void;

  // UI actions
  selectKeyframe: (id: string | null) => void;
  setDragPosition: (pos: { x: number; y: number } | null) => void;

  // Handheld motion actions
  setHandheldPreset: (preset: HandheldPreset) => void;

  // Rally-level actions
  resetCamera: (rallyId: string) => void;
  removeCameraEdit: (rallyId: string) => void;
  // Batch operation for ball tracking (single history entry + single sync)
  applyBallTrackingKeyframes: (
    rallyId: string,
    aspectRatio: AspectRatio,
    keyframes: Array<Omit<CameraKeyframe, 'id'>>
  ) => void;

  // Getters (computed at call time, not stored)
  getCameraEdit: (rallyId: string) => RallyCameraEdit;
  getActiveKeyframes: (rallyId: string) => CameraKeyframe[];
  getCameraStateAtTime: (rallyId: string, timeOffset: number) => CameraState;
  hasKeyframes: (rallyId: string) => boolean;
  hasAnyKeyframes: (rallyId: string) => boolean;

  // Bulk operations (accepts both old and new format - migration is automatic)
  loadCameraEdits: (edits: CameraEditsRecord | LegacyCameraEditsRecord | Map<string, RallyCameraEdit>) => void;
  clearAll: () => void;

  // Dirty tracking for sync
  isDirty: boolean;
  markDirty: () => void;
  clearDirty: () => void;

  // Ball tracking debug visualization
  debugBallPositions: Array<{ frameNumber: number; x: number; y: number; confidence: number }> | null;
  debugFrameCount: number | null;
  debugRallyId: string | null;
  setDebugBallTracking: (rallyId: string, positions: Array<{ frameNumber: number; x: number; y: number; confidence: number }>, frameCount: number) => void;
  clearDebugBallTracking: () => void;
}

// Generate unique keyframe ID
let keyframeCounter = 0;
const generateKeyframeId = () => `kf_${Date.now()}_${++keyframeCounter}`;

export const useCameraStore = create<CameraStoreState>()(
  subscribeWithSelector((set, get) => ({
    // Initial state
    cameraEdits: {},
    handheldPreset: 'OFF' as HandheldPreset,
    selectedKeyframeId: null,
    dragPosition: null,
    isDirty: false,
    debugBallPositions: null,
    debugFrameCount: null,
    debugRallyId: null,

    // Get camera edit for a rally (returns default if not set)
    getCameraEdit: (rallyId: string) => {
      const edit = get().cameraEdits[rallyId];
      if (!edit) return { ...DEFAULT_CAMERA_EDIT };
      // Ensure proper format (migration handled in loadCameraEdits)
      return edit;
    },

    // Get keyframes for the active aspect ratio
    getActiveKeyframes: (rallyId: string) => {
      const edit = get().cameraEdits[rallyId];
      if (!edit) return [];
      return edit.keyframes[edit.aspectRatio] ?? [];
    },

    // Set camera edit for a rally
    setCameraEdit: (rallyId: string, edit: RallyCameraEdit) => {
      set((state) => ({
        cameraEdits: { ...state.cameraEdits, [rallyId]: edit },
        isDirty: true,
      }));
      syncService.markDirty();
    },

    // Set aspect ratio for a rally (also deselects keyframe since it belongs to other ratio)
    setAspectRatio: (rallyId: string, ratio: AspectRatio) => {
      const existing = get().cameraEdits[rallyId];
      // Only push history if aspect ratio is actually changing
      if (!existing || existing.aspectRatio !== ratio) {
        pushEditorHistory();
      }
      set((state) => {
        const edit = state.cameraEdits[rallyId] ?? { ...DEFAULT_CAMERA_EDIT };
        return {
          cameraEdits: {
            ...state.cameraEdits,
            [rallyId]: { ...edit, aspectRatio: ratio },
          },
          // Deselect keyframe when switching aspect ratio (it belongs to the other set)
          selectedKeyframeId: null,
          isDirty: true,
        };
      });
      syncService.markDirty();
    },

    // Add a keyframe at specified time offset (to the active aspect ratio)
    addKeyframe: (rallyId: string, keyframe: Omit<CameraKeyframe, 'id'>) => {
      pushEditorHistory();
      const id = generateKeyframeId();
      set((state) => {
        const existing = state.cameraEdits[rallyId] ?? { ...DEFAULT_CAMERA_EDIT };
        const activeRatio = existing.aspectRatio;
        const newKeyframe: CameraKeyframe = { id, ...keyframe };

        // Get current keyframes for active aspect ratio and insert in sorted order
        const currentKeyframes = existing.keyframes[activeRatio] ?? [];
        const newKeyframes = [...currentKeyframes, newKeyframe].sort(
          (a, b) => a.timeOffset - b.timeOffset
        );

        return {
          cameraEdits: {
            ...state.cameraEdits,
            [rallyId]: {
              ...existing,
              keyframes: {
                ...existing.keyframes,
                [activeRatio]: newKeyframes,
              },
              enabled: true,
            },
          },
          selectedKeyframeId: id,
          isDirty: true,
        };
      });
      syncService.markDirty();
      return id;
    },

    // Update an existing keyframe (in the active aspect ratio)
    updateKeyframe: (rallyId: string, keyframeId: string, updates: Partial<CameraKeyframe>) => {
      pushEditorHistory();
      set((state) => {
        const existing = state.cameraEdits[rallyId];
        if (!existing) return state;

        const activeRatio = existing.aspectRatio;
        const currentKeyframes = existing.keyframes[activeRatio] ?? [];

        let newKeyframes = currentKeyframes.map((kf) =>
          kf.id === keyframeId ? { ...kf, ...updates } : kf
        );

        // Re-sort if timeOffset changed
        if ('timeOffset' in updates) {
          newKeyframes = newKeyframes.sort((a, b) => a.timeOffset - b.timeOffset);
        }

        return {
          cameraEdits: {
            ...state.cameraEdits,
            [rallyId]: {
              ...existing,
              keyframes: {
                ...existing.keyframes,
                [activeRatio]: newKeyframes,
              },
            },
          },
          isDirty: true,
        };
      });
      syncService.markDirty();
    },

    // Remove a keyframe (from the active aspect ratio)
    removeKeyframe: (rallyId: string, keyframeId: string) => {
      pushEditorHistory();
      set((state) => {
        const existing = state.cameraEdits[rallyId];
        if (!existing) return state;

        const activeRatio = existing.aspectRatio;
        const currentKeyframes = existing.keyframes[activeRatio] ?? [];
        const newKeyframes = currentKeyframes.filter((kf) => kf.id !== keyframeId);
        const newSelectedId = state.selectedKeyframeId === keyframeId ? null : state.selectedKeyframeId;

        return {
          cameraEdits: {
            ...state.cameraEdits,
            [rallyId]: {
              ...existing,
              keyframes: {
                ...existing.keyframes,
                [activeRatio]: newKeyframes,
              },
            },
          },
          selectedKeyframeId: newSelectedId,
          isDirty: true,
        };
      });
      syncService.markDirty();
    },

    // UI state
    selectKeyframe: (id: string | null) => set({ selectedKeyframeId: id }),
    setDragPosition: (pos: { x: number; y: number } | null) => set({ dragPosition: pos }),

    // Handheld motion
    setHandheldPreset: (preset: HandheldPreset) => {
      set({ handheldPreset: preset });
      // Persist to localStorage
      if (typeof window !== 'undefined') {
        localStorage.setItem('rallycut-handheld-preset', preset);
      }
    },

    // Reset camera for a rally (remove all keyframes and settings)
    resetCamera: (rallyId: string) => {
      pushEditorHistory();
      set((state) => {
        const newEdits = { ...state.cameraEdits };
        delete newEdits[rallyId];
        return {
          cameraEdits: newEdits,
          selectedKeyframeId: null,
          isDirty: true,
        };
      });
      syncService.markDirty();
    },

    // Batch operation for ball tracking - single history entry + single sync
    applyBallTrackingKeyframes: (rallyId, aspectRatio, keyframes) => {
      pushEditorHistory();
      set((state) => {
        // Generate IDs for all keyframes
        const newKeyframes: CameraKeyframe[] = keyframes.map((kf) => ({
          id: generateKeyframeId(),
          ...kf,
        }));

        // Sort by timeOffset
        newKeyframes.sort((a, b) => a.timeOffset - b.timeOffset);

        // Create new camera edit with all keyframes at once
        const newEdit: RallyCameraEdit = {
          ...DEFAULT_CAMERA_EDIT,
          aspectRatio,
          keyframes: {
            ORIGINAL: aspectRatio === 'ORIGINAL' ? newKeyframes : [],
            VERTICAL: aspectRatio === 'VERTICAL' ? newKeyframes : [],
          },
        };

        return {
          cameraEdits: {
            ...state.cameraEdits,
            [rallyId]: newEdit,
          },
          selectedKeyframeId: null,
          isDirty: true,
        };
      });
      syncService.markDirty();
    },

    // Remove camera edit for a rally (used during merge, doesn't push history)
    removeCameraEdit: (rallyId: string) => {
      set((state) => {
        const newEdits = { ...state.cameraEdits };
        delete newEdits[rallyId];
        return {
          cameraEdits: newEdits,
          isDirty: true,
        };
      });
    },

    // Get interpolated camera state at a time offset within a rally (uses active aspect ratio)
    getCameraStateAtTime: (rallyId: string, timeOffset: number): CameraState => {
      const edit = get().cameraEdits[rallyId];
      if (!edit) return DEFAULT_CAMERA_STATE;

      const activeKeyframes = edit.keyframes[edit.aspectRatio] ?? [];
      if (activeKeyframes.length === 0) {
        return DEFAULT_CAMERA_STATE;
      }
      return interpolateCameraState(activeKeyframes, timeOffset);
    },

    // Check if rally has keyframes for the active aspect ratio
    hasKeyframes: (rallyId: string) => {
      const edit = get().cameraEdits[rallyId];
      if (!edit) return false;
      return (edit.keyframes[edit.aspectRatio]?.length ?? 0) > 0;
    },

    // Check if rally has keyframes for any aspect ratio
    hasAnyKeyframes: (rallyId: string) => {
      const edit = get().cameraEdits[rallyId];
      if (!edit) return false;
      return (edit.keyframes.ORIGINAL?.length ?? 0) > 0 ||
             (edit.keyframes.VERTICAL?.length ?? 0) > 0;
    },

    // Load camera edits from backend/storage (migrates old format if needed)
    loadCameraEdits: (edits: CameraEditsRecord | LegacyCameraEditsRecord | Map<string, RallyCameraEdit>) => {
      // Convert Map to object if needed
      const rawRecord = edits instanceof Map
        ? Object.fromEntries(edits)
        : edits;

      // Migrate any old format entries (keyframes as array instead of object)
      const record: CameraEditsRecord = {};
      for (const [rallyId, edit] of Object.entries(rawRecord)) {
        record[rallyId] = migrateCameraEdit(edit);
      }

      set({ cameraEdits: record, isDirty: false });
    },

    // Clear all camera edits
    clearAll: () => {
      set({
        cameraEdits: {},
        selectedKeyframeId: null,
        isDirty: false,
      });
    },

    // Dirty tracking for sync
    markDirty: () => set({ isDirty: true }),
    clearDirty: () => set({ isDirty: false }),

    // Ball tracking debug visualization
    setDebugBallTracking: (rallyId, positions, frameCount) => {
      set({ debugBallPositions: positions, debugFrameCount: frameCount, debugRallyId: rallyId });
    },
    clearDebugBallTracking: () => {
      set({ debugBallPositions: null, debugFrameCount: null, debugRallyId: null });
    },
  }))
);

// Helper to create a keyframe at a time offset
// If currentState is provided, uses those values instead of defaults
export function createDefaultKeyframe(
  timeOffset: number,
  currentState?: { positionX: number; positionY: number; zoom: number }
): Omit<CameraKeyframe, 'id'> {
  return {
    ...DEFAULT_KEYFRAME,
    timeOffset,
    positionX: currentState?.positionX ?? DEFAULT_KEYFRAME.positionX,
    positionY: currentState?.positionY ?? DEFAULT_KEYFRAME.positionY,
    zoom: currentState?.zoom ?? DEFAULT_KEYFRAME.zoom,
  };
}

// Selector helpers for optimized subscriptions
export const selectCameraEdit = (rallyId: string | null) => (state: CameraStoreState) =>
  rallyId ? state.cameraEdits[rallyId] : undefined;

export const selectSelectedKeyframeId = (state: CameraStoreState) => state.selectedKeyframeId;

export const selectDragPosition = (state: CameraStoreState) => state.dragPosition;

export const selectHandheldPreset = (state: CameraStoreState) => state.handheldPreset;

// Initialize handheld preset from localStorage on client side
if (typeof window !== 'undefined') {
  const saved = localStorage.getItem('rallycut-handheld-preset');
  if (saved && ['OFF', 'NATURAL'].includes(saved)) {
    useCameraStore.setState({ handheldPreset: saved as HandheldPreset });
  }
}
