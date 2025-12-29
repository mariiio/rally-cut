/**
 * Format seconds to MM:SS.ms format
 */
export function formatTime(seconds: number): string {
  if (!isFinite(seconds) || seconds < 0) return '00:00.0';

  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  const ms = Math.floor((seconds % 1) * 10);

  return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}.${ms}`;
}

/**
 * Format seconds to MM:SS format (no milliseconds)
 */
export function formatTimeShort(seconds: number): string {
  if (!isFinite(seconds) || seconds < 0) return '00:00';

  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);

  return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

/**
 * Format duration in seconds to human-readable string
 */
export function formatDuration(seconds: number): string {
  if (!isFinite(seconds) || seconds < 0) return '0s';

  if (seconds < 60) {
    return `${seconds.toFixed(1)}s`;
  }

  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;

  if (secs === 0) {
    return `${mins}m`;
  }

  return `${mins}m ${secs.toFixed(0)}s`;
}

/**
 * Parse time string (MM:SS or MM:SS.ms) to seconds
 */
export function parseTime(timeStr: string): number | null {
  // Try MM:SS.ms format
  const fullMatch = timeStr.match(/^(\d+):(\d{2})\.(\d+)$/);
  if (fullMatch) {
    const mins = parseInt(fullMatch[1], 10);
    const secs = parseInt(fullMatch[2], 10);
    const ms = parseInt(fullMatch[3], 10) / Math.pow(10, fullMatch[3].length);
    return mins * 60 + secs + ms;
  }

  // Try MM:SS format
  const shortMatch = timeStr.match(/^(\d+):(\d{2})$/);
  if (shortMatch) {
    const mins = parseInt(shortMatch[1], 10);
    const secs = parseInt(shortMatch[2], 10);
    return mins * 60 + secs;
  }

  // Try seconds only
  const secsMatch = timeStr.match(/^(\d+\.?\d*)$/);
  if (secsMatch) {
    return parseFloat(secsMatch[1]);
  }

  return null;
}
