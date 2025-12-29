import { SegmentFile } from '@/types/segment';

/**
 * Parse a JSON file and validate it's a valid SegmentFile
 */
export async function parseSegmentJson(file: File): Promise<SegmentFile> {
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

  return json as SegmentFile;
}

/**
 * Download a SegmentFile as JSON
 */
export function downloadSegmentJson(data: SegmentFile, filename: string = 'segments.json'): void {
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
