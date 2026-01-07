import { create } from 'zustand';
import type { UserTier, TierLimits, UsageQuota } from '@/services/api';
import { getCurrentUser } from '@/services/api';
import { syncService } from '@/services/syncService';

interface TierState {
  tier: UserTier;
  limits: TierLimits;
  usage: UsageQuota;
  isLoading: boolean;
  error: string | null;
  lastFetched: Date | null;
}

interface TierActions {
  fetchTier: (force?: boolean) => Promise<void>;
  setTier: (tier: UserTier, limits: TierLimits, usage: UsageQuota) => void;
  isPremium: () => boolean;
  canDetect: () => boolean;
  canUpload: () => boolean;
  getUploadsRemaining: () => number | null;
  canSyncToServer: () => boolean;
  shouldShowWatermark: () => boolean;
  canUseLambdaExport: () => boolean;
  shouldRefetch: () => boolean;
}

// Cache TTL for tier data (5 minutes)
const CACHE_TTL_MS = 5 * 60 * 1000;

// Default FREE tier limits (in case fetch fails)
const DEFAULT_FREE_LIMITS: TierLimits = {
  detectionsPerMonth: 1,
  maxVideoDurationMs: 15 * 60 * 1000,
  maxFileSizeBytes: 1 * 1024 * 1024 * 1024, // 1 GB
  monthlyUploadCount: 5,
  exportQuality: '720p',
  exportWatermark: true,
  lambdaExportEnabled: false,
  retentionDays: null, // Deprecated
  originalQualityDays: 3,
  inactivityDeleteDays: 60,
  serverSyncEnabled: false,
  highlightsEnabled: true,
};

const DEFAULT_FREE_USAGE: UsageQuota = {
  detectionsUsed: 0,
  detectionsLimit: 1,
  detectionsRemaining: 1,
  uploadsThisMonth: 0,
  uploadsLimit: 5,
  uploadsRemaining: 5,
  periodStart: new Date().toISOString(),
};

export const useTierStore = create<TierState & TierActions>((set, get) => ({
  tier: 'FREE',
  limits: DEFAULT_FREE_LIMITS,
  usage: DEFAULT_FREE_USAGE,
  isLoading: false,
  error: null,
  lastFetched: null,

  fetchTier: async (force = false) => {
    // Skip if cache is still valid (unless forced)
    if (!force && !get().shouldRefetch()) {
      return;
    }

    const previousTier = get().tier;
    set({ isLoading: true, error: null });

    try {
      const user = await getCurrentUser();
      set({
        tier: user.tier,
        limits: user.tierLimits,
        usage: user.usage,
        isLoading: false,
        lastFetched: new Date(),
      });

      // If upgraded from FREE to PREMIUM, trigger sync to push localStorage data to server
      // Use a short delay to ensure editor/syncService has initialized
      if (previousTier === 'FREE' && user.tier === 'PREMIUM') {
        console.log('[TierStore] Upgraded to PREMIUM - triggering sync for localStorage data');
        // Immediate attempt
        syncService.markDirty();
        // Retry after editor likely initialized (in case page is still loading)
        setTimeout(() => {
          console.log('[TierStore] Retry sync after delay');
          syncService.markDirty();
        }, 2000);
      }
    } catch (error) {
      console.error('Failed to fetch tier:', error);
      set({
        isLoading: false,
        error: error instanceof Error ? error.message : 'Failed to fetch tier',
      });
    }
  },

  setTier: (tier, limits, usage) => {
    set({ tier, limits, usage });
  },

  isPremium: () => get().tier === 'PREMIUM',

  canDetect: () => {
    const { usage } = get();
    return usage.detectionsRemaining > 0;
  },

  canUpload: () => {
    const { usage } = get();
    // null means unlimited
    if (usage.uploadsRemaining === null) return true;
    return usage.uploadsRemaining > 0;
  },

  getUploadsRemaining: () => get().usage.uploadsRemaining,

  canSyncToServer: () => get().limits.serverSyncEnabled,

  shouldShowWatermark: () => get().limits.exportWatermark,

  canUseLambdaExport: () => get().limits.lambdaExportEnabled,

  shouldRefetch: () => {
    const { lastFetched } = get();
    if (!lastFetched) return true;
    return Date.now() - lastFetched.getTime() > CACHE_TTL_MS;
  },
}));
