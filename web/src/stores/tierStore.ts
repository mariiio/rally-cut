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
  isPaidTier: () => boolean;
  canDetect: () => boolean;
  canUpload: () => boolean;
  getUploadsRemaining: () => number;
  canSyncToServer: () => boolean;
  shouldShowWatermark: () => boolean;
  canUseLambdaExport: () => boolean;
  shouldRefetch: () => boolean;
  getStorageUsedPercent: () => number;
}

// Cache TTL for tier data (5 minutes)
const CACHE_TTL_MS = 5 * 60 * 1000;

/**
 * Tier limits for upgrade prompts and display text.
 * IMPORTANT: Keep in sync with api/src/config/tiers.ts (source of truth).
 * These are ONLY used for static UI text before API data is fetched.
 * Actual enforcement happens server-side via tierService.
 */
export const TIER_LIMITS_DISPLAY = {
  FREE: {
    maxFileSizeMB: 500,
    maxVideoDurationMin: 15,
    storageCapGB: 1,
    detectionsPerMonth: 2,
    uploadsPerMonth: 3,
  },
  PRO: {
    maxFileSizeGB: 2,
    maxVideoDurationMin: 45,
    storageCapGB: 20,
    detectionsPerMonth: 15,
    uploadsPerMonth: 20,
  },
  ELITE: {
    maxFileSizeGB: 5,
    maxVideoDurationMin: 90,
    storageCapGB: 75,
    detectionsPerMonth: 50,
    uploadsPerMonth: 50,
  },
} as const;

// Default FREE tier limits (in case fetch fails)
const DEFAULT_FREE_LIMITS: TierLimits = {
  detectionsPerMonth: 2,
  maxVideoDurationMs: 15 * 60 * 1000,
  maxFileSizeBytes: 500 * 1024 * 1024, // 500 MB
  monthlyUploadCount: 3,
  storageCapBytes: 1 * 1024 * 1024 * 1024, // 1 GB
  exportQuality: '720p',
  exportWatermark: true,
  lambdaExportEnabled: false,
  originalQualityDays: 3,
  inactivityDeleteDays: 30,
  serverSyncEnabled: false,
  highlightsEnabled: true,
};

const DEFAULT_FREE_USAGE: UsageQuota = {
  detectionsUsed: 0,
  detectionsLimit: 2,
  detectionsRemaining: 2,
  uploadsThisMonth: 0,
  uploadsLimit: 3,
  uploadsRemaining: 3,
  storageUsedBytes: 0,
  storageLimitBytes: 1 * 1024 * 1024 * 1024,
  storageRemainingBytes: 1 * 1024 * 1024 * 1024,
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

      // If upgraded from FREE to a paid tier, trigger sync to push localStorage data to server
      // Use a short delay to ensure editor/syncService has initialized
      if (previousTier === 'FREE' && (user.tier === 'PRO' || user.tier === 'ELITE')) {
        // Immediate attempt
        syncService.markDirty();
        // Retry after editor likely initialized (in case page is still loading)
        setTimeout(() => {
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

  isPaidTier: () => {
    const tier = get().tier;
    return tier === 'PRO' || tier === 'ELITE';
  },

  canDetect: () => {
    const { usage } = get();
    return usage.detectionsRemaining > 0;
  },

  canUpload: () => {
    const { usage } = get();
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

  getStorageUsedPercent: () => {
    const { usage } = get();
    if (usage.storageLimitBytes === 0) return 0;
    return Math.round((usage.storageUsedBytes / usage.storageLimitBytes) * 100);
  },
}));
