/**
 * Enum constants matching Prisma schema enums.
 * Provides type-safe constants for frontend use without Prisma client dependency.
 */

export const VideoStatus = {
  PENDING: 'PENDING',
  UPLOADED: 'UPLOADED',
  DETECTING: 'DETECTING',
  DETECTED: 'DETECTED',
  ERROR: 'ERROR',
} as const;

export type VideoStatus = (typeof VideoStatus)[keyof typeof VideoStatus];

export const ConfirmationStatus = {
  PENDING: 'PENDING',
  PROCESSING: 'PROCESSING',
  CONFIRMED: 'CONFIRMED',
  FAILED: 'FAILED',
} as const;

export type ConfirmationStatus = (typeof ConfirmationStatus)[keyof typeof ConfirmationStatus];

export const AspectRatio = {
  ORIGINAL: 'ORIGINAL',
  VERTICAL: 'VERTICAL',
} as const;

export type AspectRatio = (typeof AspectRatio)[keyof typeof AspectRatio];

export const CameraEasing = {
  LINEAR: 'LINEAR',
  EASE_IN: 'EASE_IN',
  EASE_OUT: 'EASE_OUT',
  EASE_IN_OUT: 'EASE_IN_OUT',
} as const;

export type CameraEasing = (typeof CameraEasing)[keyof typeof CameraEasing];

export const SessionType = {
  REGULAR: 'REGULAR',
  ALL_VIDEOS: 'ALL_VIDEOS',
} as const;

export type SessionType = (typeof SessionType)[keyof typeof SessionType];

export const ExportStatus = {
  PENDING: 'PENDING',
  PROCESSING: 'PROCESSING',
  COMPLETED: 'COMPLETED',
  FAILED: 'FAILED',
} as const;

export type ExportStatus = (typeof ExportStatus)[keyof typeof ExportStatus];
