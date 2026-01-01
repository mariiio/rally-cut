import { RallyFile } from '@/types/rally';

/**
 * Parse a JSON file and validate it's a valid RallyFile
 */
export async function parseRallyJson(file: File): Promise<RallyFile> {
  const text = await file.text();
  const json = JSON.parse(text);

  // Basic validation
  if (!json.version || json.version !== '1.0') {
    throw new Error('Invalid JSON format: missing or incorrect version');
  }
  if (!json.video || typeof json.video.duration !== 'number') {
    throw new Error('Invalid JSON format: missing video metadata');
  }
  if (!Array.isArray(json.rallies)) {
    throw new Error('Invalid JSON format: rallies must be an array');
  }

  // Validate each rally has required fields
  for (const rally of json.rallies) {
    if (!rally.id || typeof rally.start_time !== 'number' || typeof rally.end_time !== 'number') {
      throw new Error(`Invalid rally: missing required fields in ${rally.id || 'unknown'}`);
    }
  }

  return json as RallyFile;
}

/**
 * Download a RallyFile as JSON
 */
export function downloadRallyJson(data: RallyFile, filename: string = 'rallies.json'): void {
  const json = JSON.stringify(data, null, 2);
  const blob = new Blob([json], { type: 'application/json' });
  const url = URL.createObjectURL(blob);

  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);

  URL.revokeObjectURL(url);
}

/**
 * Check if a file is a valid video format
 */
export function isValidVideoFile(file: File): boolean {
  const validTypes = ['video/mp4', 'video/quicktime', 'video/webm', 'video/x-m4v'];
  const validExtensions = ['.mp4', '.mov', '.webm', '.m4v'];

  if (validTypes.includes(file.type)) {
    return true;
  }

  const ext = file.name.toLowerCase().slice(file.name.lastIndexOf('.'));
  return validExtensions.includes(ext);
}

/**
 * Check if a file is a JSON file
 */
export function isJsonFile(file: File): boolean {
  return file.type === 'application/json' || file.name.toLowerCase().endsWith('.json');
}

/**
 * Generate a SHA-256 hash of a file (for deduplication)
 */
export async function hashFile(file: File): Promise<string> {
  const buffer = await file.arrayBuffer();
  const hashBuffer = await crypto.subtle.digest('SHA-256', buffer);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
}

/**
 * Get video duration using a hidden video element
 */
export function getVideoDuration(file: File): Promise<number> {
  return new Promise((resolve, reject) => {
    const video = document.createElement('video');
    video.preload = 'metadata';
    video.onloadedmetadata = () => {
      URL.revokeObjectURL(video.src);
      resolve(video.duration * 1000); // Return milliseconds
    };
    video.onerror = () => {
      URL.revokeObjectURL(video.src);
      reject(new Error('Failed to load video metadata'));
    };
    video.src = URL.createObjectURL(file);
  });
}
