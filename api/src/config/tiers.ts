/**
 * Centralized tier configuration.
 * Single source of truth for all tier limits across the application.
 *
 * To modify tier limits, update this file only.
 */

// Tier names as enum for type safety
export type UserTier = "FREE" | "PRO" | "ELITE";

// Export quality options
export type ExportQuality = "720p" | "original";

// Tier configuration interface
export interface TierConfig {
  // Detection & Upload Limits
  detectionsPerMonth: number;
  monthlyUploadCount: number;

  // Video Constraints
  maxVideoDurationMs: number;
  maxFileSizeBytes: number;
  storageCapBytes: number;

  // Export Settings
  exportQuality: ExportQuality;
  exportWatermark: boolean;
  lambdaExportEnabled: boolean;

  // Retention Policy
  originalQualityDays: number | null; // null = forever
  inactivityDeleteDays: number | null; // null = never

  // Features
  serverSyncEnabled: boolean;
  highlightsEnabled: boolean;
}

// Helper constants
const GB = 1024 * 1024 * 1024;
const MB = 1024 * 1024;
const MINUTES = 60 * 1000;

/**
 * Tier Configuration
 *
 * | Feature                  | BASIC (Free) | PRO ($9.99)  | ELITE ($24.99) |
 * |--------------------------|--------------|--------------|----------------|
 * | AI Detections/month      | 2            | 15           | 50             |
 * | Monthly Uploads          | 5            | 20           | 50             |
 * | Max Video Duration       | 30 min       | 60 min       | 90 min         |
 * | Max File Size            | 500 MB       | 3 GB         | 5 GB           |
 * | Storage Cap              | 2 GB         | 20 GB        | 75 GB          |
 * | Export Quality           | 720p         | Original     | Original       |
 * | Watermark                | Yes          | No           | No             |
 * | Server Export            | No           | Yes          | Yes            |
 * | Original Quality         | 7 days       | 14 days      | 60 days        |
 * | Video Retention          | 90 days      | 6 months     | 1 year         |
 * | Server Sync              | No           | Yes          | Yes            |
 */
export const TIER_CONFIG: Record<UserTier, TierConfig> = {
  FREE: {
    // Limits
    detectionsPerMonth: 2,
    monthlyUploadCount: 5,

    // Video Constraints
    maxVideoDurationMs: 30 * MINUTES,
    maxFileSizeBytes: 500 * MB,
    storageCapBytes: 2 * GB,

    // Export
    exportQuality: "720p",
    exportWatermark: true,
    lambdaExportEnabled: false,

    // Retention
    originalQualityDays: 7,
    inactivityDeleteDays: 90,

    // Features
    serverSyncEnabled: false,
    highlightsEnabled: true,
  },

  PRO: {
    // Limits
    detectionsPerMonth: 15,
    monthlyUploadCount: 20,

    // Video Constraints
    maxVideoDurationMs: 60 * MINUTES,
    maxFileSizeBytes: 3 * GB,
    storageCapBytes: 20 * GB,

    // Export
    exportQuality: "original",
    exportWatermark: false,
    lambdaExportEnabled: true,

    // Retention (6 months = 180 days)
    originalQualityDays: 14,
    inactivityDeleteDays: 180,

    // Features
    serverSyncEnabled: true,
    highlightsEnabled: true,
  },

  ELITE: {
    // Limits
    detectionsPerMonth: 50,
    monthlyUploadCount: 50,

    // Video Constraints
    maxVideoDurationMs: 90 * MINUTES,
    maxFileSizeBytes: 5 * GB,
    storageCapBytes: 75 * GB,

    // Export
    exportQuality: "original",
    exportWatermark: false,
    lambdaExportEnabled: true,

    // Retention (1 year = 365 days)
    originalQualityDays: 60,
    inactivityDeleteDays: 365,

    // Features
    serverSyncEnabled: true,
    highlightsEnabled: true,
  },
};

/**
 * Get tier configuration.
 */
export function getTierConfig(tier: UserTier): TierConfig {
  return TIER_CONFIG[tier];
}

/**
 * Pricing information (for display purposes).
 */
export const PRICING = {
  FREE: {
    monthly: 0,
    yearly: 0,
  },
  PRO: {
    monthly: 9.99,
    yearly: 95.9, // ~20% savings
  },
  ELITE: {
    monthly: 24.99,
    yearly: 239.9, // ~20% savings
  },
};

/**
 * Pay-per-match credit pricing.
 */
export const CREDIT_PRICE = 0.99;

/**
 * Helper to format bytes for display.
 */
export function formatBytes(bytes: number): string {
  if (bytes >= GB) {
    return `${(bytes / GB).toFixed(bytes % GB === 0 ? 0 : 1)} GB`;
  }
  if (bytes >= MB) {
    return `${(bytes / MB).toFixed(0)} MB`;
  }
  return `${bytes} bytes`;
}

/**
 * Helper to format duration for display.
 */
export function formatDuration(ms: number): string {
  const minutes = Math.floor(ms / MINUTES);
  return `${minutes} min`;
}
