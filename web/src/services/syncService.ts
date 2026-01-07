/**
 * Simplified sync service for persisting state to backend.
 *
 * Strategy:
 * - localStorage handles immediate persistence and undo/redo history (via editorStore)
 * - This service periodically syncs the current state to the backend
 * - No operation tracking - just pushes the final rally/highlight state
 *
 * Tier restrictions:
 * - FREE tier: localStorage only, no server sync
 * - PREMIUM tier: Full server sync enabled
 */

import { API_BASE_URL, getHeaders } from './api';
import type { Rally, Highlight } from '@/types/rally';
import { useTierStore } from '@/stores/tierStore';
import { useCameraStore } from '@/stores/cameraStore';

// Sync configuration
const SYNC_DEBOUNCE_MS = 5000; // Sync 5 seconds after last change
const STORAGE_KEY_PREFIX = 'rallycut_sync_meta_';

interface SyncMeta {
  lastSyncAt: number;
  isDirty: boolean;
}

interface SyncState {
  sessionId: string;
  isDirty: boolean;
  isSyncing: boolean;
  lastSyncAt: number;
  error: string | null;
}

type SyncStatusListener = (status: {
  isSyncing: boolean;
  pendingCount: number;
  error: string | null;
  lastSyncAt: number;
}) => void;

// State getter function - will be set by editorStore
type StateGetter = () => {
  session: { id: string; matches: Array<{ id: string; rallies: Rally[] }> } | null;
  rallies: Rally[];
  highlights: Highlight[];
  activeMatchId: string | null;
};

class SyncService {
  private state: SyncState | null = null;
  private syncTimeout: ReturnType<typeof setTimeout> | null = null;
  private listeners: Set<SyncStatusListener> = new Set();
  private getState: StateGetter | null = null;

  /**
   * Initialize sync service for a session.
   */
  init(sessionId: string) {
    const meta = this.loadMeta(sessionId);

    // Check tier - FREE users cannot sync to server
    const canSync = useTierStore.getState().canSyncToServer();

    // Restore dirty state from localStorage - if there were unsaved changes before reload
    // For FREE tier, always mark as not dirty since we can't sync anyway
    const wasDirty = canSync ? (meta?.isDirty ?? false) : false;

    this.state = {
      sessionId,
      isDirty: wasDirty,
      isSyncing: false,
      lastSyncAt: meta?.lastSyncAt || Date.now(),
      error: null,
    };

    this.notifyListeners();

    // If we had unsaved changes and can sync, schedule a sync
    if (wasDirty && canSync) {
      this.scheduleSyncDebounced();
    }
  }

  /**
   * Set the state getter function (called by editorStore).
   */
  setStateGetter(getter: StateGetter) {
    this.getState = getter;
  }

  /**
   * Mark state as dirty and schedule sync.
   * For FREE tier, only saves locally without attempting cloud sync.
   */
  markDirty() {
    if (!this.state) return;

    // Check tier early - don't even schedule sync for FREE tier
    const canSync = useTierStore.getState().canSyncToServer();
    if (!canSync) {
      // FREE tier: just mark as saved locally, don't schedule sync
      this.state.isDirty = false;
      this.state.error = null;
      this.notifyListeners();
      return;
    }

    this.state.isDirty = true;
    this.state.error = null;

    // Persist dirty state so it survives page reload
    this.saveMeta(this.state.sessionId, {
      lastSyncAt: this.state.lastSyncAt,
      isDirty: true,
    });

    this.notifyListeners();
    this.scheduleSyncDebounced();
  }

  /**
   * Force immediate sync.
   */
  async syncNow(): Promise<boolean> {
    return this.performSync();
  }

  /**
   * Subscribe to sync status changes.
   */
  subscribe(listener: SyncStatusListener): () => void {
    this.listeners.add(listener);
    if (this.state) {
      listener(this.getStatus());
    }
    return () => this.listeners.delete(listener);
  }

  /**
   * Get current sync status.
   */
  getStatus() {
    if (!this.state) {
      return { isSyncing: false, pendingCount: 0, error: null, lastSyncAt: 0 };
    }
    return {
      isSyncing: this.state.isSyncing,
      pendingCount: this.state.isDirty ? 1 : 0,
      error: this.state.error,
      lastSyncAt: this.state.lastSyncAt,
    };
  }

  /**
   * Reset sync service.
   */
  reset() {
    if (this.syncTimeout) {
      clearTimeout(this.syncTimeout);
      this.syncTimeout = null;
    }
    this.listeners.clear();
    this.state = null;
    this.getState = null;
  }

  // Private methods

  private scheduleSyncDebounced() {
    // Don't schedule sync for FREE tier
    if (!useTierStore.getState().canSyncToServer()) {
      return;
    }

    if (this.syncTimeout) {
      clearTimeout(this.syncTimeout);
    }
    this.syncTimeout = setTimeout(() => {
      this.performSync();
    }, SYNC_DEBOUNCE_MS);
  }

  private async performSync(): Promise<boolean> {
    if (!this.state || !this.state.isDirty || this.state.isSyncing || !this.getState) {
      return true;
    }

    // Check tier - FREE users cannot sync to server
    const canSync = useTierStore.getState().canSyncToServer();
    if (!canSync) {
      // Clear dirty flag since we can't sync anyway
      this.state.isDirty = false;
      this.state.error = 'Sync disabled for FREE tier';
      this.saveMeta(this.state.sessionId, {
        lastSyncAt: this.state.lastSyncAt,
        isDirty: false,
      });
      this.notifyListeners();
      return true;
    }

    this.state.isSyncing = true;
    this.state.error = null;
    this.notifyListeners();

    try {
      const currentState = this.getState();
      if (!currentState.session) {
        throw new Error('No session loaded');
      }

      // Build the state to sync
      // Collect rallies from all matches, using current active match's edited rallies
      const cameraEdits = useCameraStore.getState().cameraEdits;

      // Helper to map keyframes to sync format
      const mapKeyframes = (keyframes: Array<{
        timeOffset: number;
        positionX: number;
        positionY: number;
        zoom: number;
        easing: 'LINEAR' | 'EASE_IN' | 'EASE_OUT' | 'EASE_IN_OUT';
      }>) => keyframes.map(kf => ({
        timeOffset: kf.timeOffset,
        positionX: kf.positionX,
        positionY: kf.positionY,
        zoom: kf.zoom,
        easing: kf.easing,
      }));

      const ralliesPerVideo: Record<string, Array<{
        id?: string;
        startMs: number;
        endMs: number;
        cameraEdit?: {
          enabled: boolean;
          aspectRatio: 'ORIGINAL' | 'VERTICAL';
          // Backend expects a flat array of keyframes for the current aspect ratio
          keyframes: Array<{
            timeOffset: number;
            positionX: number;
            positionY: number;
            zoom: number;
            easing: 'LINEAR' | 'EASE_IN' | 'EASE_OUT' | 'EASE_IN_OUT';
          }>;
        };
      }>> = {};

      for (const match of currentState.session.matches) {
        const rallies = match.id === currentState.activeMatchId
          ? currentState.rallies
          : match.rallies;

        ralliesPerVideo[match.id] = rallies.map(r => {
          const cameraEdit = cameraEdits[r.id];
          // Only include camera edit if it has keyframes for the current aspect ratio
          const currentAspectRatio = cameraEdit?.aspectRatio ?? 'ORIGINAL';
          const keyframesForAspect = cameraEdit?.keyframes[currentAspectRatio] ?? [];
          const hasKeyframes = keyframesForAspect.length > 0;
          return {
            id: r._backendId,
            startMs: Math.round(r.start_time * 1000),
            endMs: Math.round(r.end_time * 1000),
            ...(hasKeyframes && {
              cameraEdit: {
                enabled: true, // Always enabled if we have keyframes
                aspectRatio: currentAspectRatio,
                // Backend expects a flat array of keyframes for the current aspect ratio
                keyframes: mapKeyframes(keyframesForAspect),
              },
            }),
          };
        });
      }

      // Build highlights
      const highlights = currentState.highlights.map(h => ({
        id: h._backendId,
        name: h.name,
        color: h.color,
        rallyIds: h.rallyIds,
      }));

      // Send to backend
      const response = await fetch(`${API_BASE_URL}/v1/sessions/${this.state.sessionId}/sync-state`, {
        method: 'POST',
        headers: getHeaders('application/json'),
        body: JSON.stringify({
          ralliesPerVideo,
          highlights,
        }),
      });

      if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.error?.message || `Sync failed: ${response.status}`);
      }

      // Success
      this.state.isDirty = false;
      this.state.lastSyncAt = Date.now();
      this.state.error = null;
      this.saveMeta(this.state.sessionId, {
        lastSyncAt: this.state.lastSyncAt,
        isDirty: false,
      });

      this.notifyListeners();
      return true;
    } catch (error) {
      console.error('Sync failed:', error);
      this.state.error = error instanceof Error ? error.message : 'Sync failed';
      this.notifyListeners();
      return false;
    } finally {
      if (this.state) {
        this.state.isSyncing = false;
        this.notifyListeners();
      }
    }
  }

  private getStorageKey(sessionId: string): string {
    return `${STORAGE_KEY_PREFIX}${sessionId}`;
  }

  private saveMeta(sessionId: string, meta: SyncMeta) {
    try {
      localStorage.setItem(this.getStorageKey(sessionId), JSON.stringify(meta));
    } catch {
      // Ignore localStorage errors
    }
  }

  private loadMeta(sessionId: string): SyncMeta | null {
    try {
      const stored = localStorage.getItem(this.getStorageKey(sessionId));
      return stored ? JSON.parse(stored) : null;
    } catch {
      return null;
    }
  }

  private notifyListeners() {
    const status = this.getStatus();
    for (const listener of this.listeners) {
      listener(status);
    }
  }
}

// Singleton instance
export const syncService = new SyncService();
