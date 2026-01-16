import { Rally } from '@/types/rally';
import type { FFmpegInstance } from '@/types/ffmpeg';
import { getVisitorId } from '@/utils/visitorId';

export type VideoSource = File | string;

// For multi-source exports (highlights with rallies from different matches)
export interface RallyWithSource {
  rally: Rally;
  videoSource: VideoSource;
}

// FFmpeg encoding settings - extracted to avoid duplication
const FFMPEG_VIDEO_CODEC = ['-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23'];
const FFMPEG_PIXEL_FORMAT = ['-pix_fmt', 'yuv420p'];
const FFMPEG_AUDIO_CODEC = ['-c:a', 'aac', '-b:a', '128k'];
const FFMPEG_ENCODE_SETTINGS = [...FFMPEG_VIDEO_CODEC, ...FFMPEG_PIXEL_FORMAT, ...FFMPEG_AUDIO_CODEC];

// Fade transition duration in seconds
export const FADE_DURATION = 0.5;

// Watermark settings
const WATERMARK_FILENAME = 'watermark.png';
let watermarkLoaded = false;

// Singleton FFmpeg instance
let ffmpeg: FFmpegInstance | null = null;
let loadPromise: Promise<void> | null = null;
let currentProgressCallback: ProgressCallback | null = null;
// Track minimum progress to prevent FFmpeg from resetting our stage-based progress
let minProgress = 0;
// Track the current FFmpeg operation's progress range for proper mapping
let ffmpegProgressRange = { min: 0, max: 100 };
// Cancellation flag
let isCancelled = false;

// Video cache to avoid re-downloading the same video for multiple rally exports
// - Small videos (< 200MB): Persist in IndexedDB across page refreshes
// - Large videos (>= 200MB): Memory cache only (cleared on refresh)
const memoryCache = new Map<string, Uint8Array>();
const MAX_INDEXEDDB_VIDEO_SIZE = 200 * 1024 * 1024; // Only persist videos < 200MB to IndexedDB
const MAX_INDEXEDDB_TOTAL_SIZE = 500 * 1024 * 1024; // 500MB total IndexedDB cache limit
const DB_NAME = 'rallycut-video-cache';
const DB_VERSION = 2; // Bumped to add timestamp index
const STORE_NAME = 'videos';

let dbPromise: Promise<IDBDatabase> | null = null;

/**
 * Open IndexedDB connection (lazy, singleton)
 */
function openDB(): Promise<IDBDatabase> {
  if (dbPromise) return dbPromise;

  dbPromise = new Promise((resolve, reject) => {
    if (typeof indexedDB === 'undefined') {
      reject(new Error('IndexedDB not available'));
      return;
    }

    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);

    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result;
      const tx = (event.target as IDBOpenDBRequest).transaction!;

      if (!db.objectStoreNames.contains(STORE_NAME)) {
        // Fresh install
        const store = db.createObjectStore(STORE_NAME, { keyPath: 'key' });
        store.createIndex('timestamp', 'timestamp', { unique: false });
      } else {
        // Upgrading from v1 - add timestamp index and clear old data
        const store = tx.objectStore(STORE_NAME);
        if (!store.indexNames.contains('timestamp')) {
          // Clear old entries (they don't have size/timestamp fields)
          store.clear();
          store.createIndex('timestamp', 'timestamp', { unique: false });
        }
      }
    };
  });

  return dbPromise;
}

interface CacheEntry {
  key: string;
  data: Uint8Array;
  size: number;
  timestamp: number;
}

/**
 * Get cached video data from IndexedDB or memory
 */
async function getCachedVideo(key: string): Promise<Uint8Array | null> {
  // Check memory cache first (faster)
  const memCached = memoryCache.get(key);
  if (memCached) return memCached;

  // Try IndexedDB
  try {
    const db = await openDB();
    return new Promise((resolve) => {
      const tx = db.transaction(STORE_NAME, 'readonly');
      const store = tx.objectStore(STORE_NAME);
      const request = store.get(key);

      request.onsuccess = () => {
        const result = request.result as CacheEntry | undefined;
        if (result?.data) {
          // Also store in memory for faster subsequent access
          memoryCache.set(key, result.data);
          resolve(result.data);
        } else {
          resolve(null);
        }
      };
      request.onerror = () => resolve(null);
    });
  } catch {
    return null;
  }
}

/**
 * Get total size of all cached entries
 */
async function getCacheTotalSize(db: IDBDatabase): Promise<number> {
  return new Promise((resolve) => {
    const tx = db.transaction(STORE_NAME, 'readonly');
    const store = tx.objectStore(STORE_NAME);
    const request = store.getAll();

    request.onsuccess = () => {
      const entries = request.result as CacheEntry[];
      const totalSize = entries.reduce((sum, entry) => sum + (entry.size || 0), 0);
      resolve(totalSize);
    };
    request.onerror = () => resolve(0);
  });
}

/**
 * Evict oldest entries until we're under the size limit
 */
async function evictIfNeeded(db: IDBDatabase, newEntrySize: number): Promise<void> {
  const currentSize = await getCacheTotalSize(db);
  let sizeToFree = (currentSize + newEntrySize) - MAX_INDEXEDDB_TOTAL_SIZE;

  if (sizeToFree <= 0) return;

  // Get all entries sorted by timestamp (oldest first)
  return new Promise((resolve) => {
    const tx = db.transaction(STORE_NAME, 'readwrite');
    const store = tx.objectStore(STORE_NAME);
    const index = store.index('timestamp');
    const request = index.openCursor();

    request.onsuccess = (event) => {
      const cursor = (event.target as IDBRequest<IDBCursorWithValue>).result;
      if (cursor && sizeToFree > 0) {
        const entry = cursor.value as CacheEntry;
        sizeToFree -= entry.size || 0;
        cursor.delete();
        cursor.continue();
      } else {
        resolve();
      }
    };
    request.onerror = () => resolve();
  });
}

/**
 * Cache video data in memory, and IndexedDB for small videos
 */
async function cacheVideo(key: string, data: Uint8Array): Promise<void> {
  const sizeMB = (data.length / 1024 / 1024).toFixed(1);

  // Always store in memory for current session
  memoryCache.set(key, data);

  // Only persist small videos to IndexedDB (large videos would fill up storage)
  if (data.length >= MAX_INDEXEDDB_VIDEO_SIZE) {
    console.log(`[VideoExport] Cached in memory only (too large for IndexedDB): ${key.substring(0, 50)}... (${sizeMB} MB)`);
    return;
  }

  // Store in IndexedDB for persistence across refreshes
  try {
    const db = await openDB();

    // Evict old entries if adding this would exceed the limit
    await evictIfNeeded(db, data.length);

    // Store the new entry
    const tx = db.transaction(STORE_NAME, 'readwrite');
    const store = tx.objectStore(STORE_NAME);
    const entry: CacheEntry = {
      key,
      data,
      size: data.length,
      timestamp: Date.now(),
    };
    store.put(entry);

    console.log(`[VideoExport] Cached in IndexedDB: ${key.substring(0, 50)}... (${sizeMB} MB)`);
  } catch (err) {
    console.warn('[VideoExport] Failed to cache in IndexedDB:', err);
  }
}

/**
 * Clear video cache (IndexedDB and memory)
 */
export async function clearVideoCache(): Promise<void> {
  memoryCache.clear();

  try {
    const db = await openDB();
    const tx = db.transaction(STORE_NAME, 'readwrite');
    const store = tx.objectStore(STORE_NAME);
    store.clear();
    console.log('[VideoExport] Video cache cleared');
  } catch {
    // Ignore errors
  }
}

export type ProgressCallback = (progress: number, step: string) => void;

/**
 * Cancel the current export operation
 */
export function cancelExport(): void {
  isCancelled = true;
}

/**
 * Check if export was cancelled and throw if so
 */
function checkCancelled(): void {
  if (isCancelled) {
    throw new Error('Export cancelled');
  }
}

/**
 * Report progress and update minimum to prevent FFmpeg from going backwards
 */
function reportProgress(progress: number, step: string) {
  // Clamp progress to valid range
  const clampedProgress = Math.max(0, Math.min(100, Math.round(progress)));
  minProgress = Math.max(minProgress, clampedProgress);
  currentProgressCallback?.(minProgress, step);
}

/**
 * Beach volleyball-themed progress messages
 */
const rallyVerbs = [
  'Spiking',
  'Setting',
  'Bumping',
  'Serving',
  'Blocking',
  'Diving for',
  'Smashing',
  'Poking',
];

const messages = {
  loadingFFmpeg: 'Warming up...',
  ffmpegLoaded: 'Ready to play',
  loadingVideo: 'Setting up the court...',
  loadingVideos: (current: number, total: number) => `Loading match ${current}/${total}...`,
  extractingClip: (current: number, total: number) => `Digging rally ${current} of ${total}...`,
  encodingClip: (current: number, total: number) => {
    const verb = rallyVerbs[current % rallyVerbs.length];
    return `${verb} rally ${current} of ${total}...`;
  },
  joiningClips: 'Running the play...',
  applyingFade: 'Adding smooth transitions...',
  finalizing: 'Match point...',
  complete: 'Game, set, match!',
  processing: 'In play...',
};

/**
 * Reset progress tracking for a new export
 */
function resetProgress() {
  minProgress = 0;
  ffmpegProgressRange = { min: 0, max: 100 };
  isCancelled = false;
}

/**
 * Set the progress range for the current FFmpeg operation
 * FFmpeg's 0-100% will be mapped to this range
 */
function setFFmpegProgressRange(min: number, max: number) {
  ffmpegProgressRange = { min, max };
}

/**
 * Generate watermark PNG using canvas
 * Creates a "created with RallyCut" watermark with volleyball icon
 */
async function generateWatermarkPng(): Promise<Uint8Array> {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d')!;

  // Watermark dimensions
  const width = 280;
  const height = 40;
  canvas.width = width;
  canvas.height = height;

  // Clear with transparency
  ctx.clearRect(0, 0, width, height);

  // Draw volleyball icon (simplified circle with lines)
  const iconSize = 24;
  const iconX = 14;
  const iconY = height / 2;

  ctx.save();
  ctx.globalAlpha = 0.7;

  // Volleyball circle
  ctx.beginPath();
  ctx.arc(iconX, iconY, iconSize / 2, 0, Math.PI * 2);
  ctx.strokeStyle = 'white';
  ctx.lineWidth = 2;
  ctx.stroke();

  // Volleyball lines (simplified)
  ctx.beginPath();
  ctx.moveTo(iconX - iconSize / 2, iconY);
  ctx.lineTo(iconX + iconSize / 2, iconY);
  ctx.stroke();

  ctx.beginPath();
  ctx.arc(iconX, iconY - iconSize / 2, iconSize / 2, 0.3, Math.PI - 0.3);
  ctx.stroke();

  ctx.beginPath();
  ctx.arc(iconX, iconY + iconSize / 2, iconSize / 2, Math.PI + 0.3, -0.3);
  ctx.stroke();

  // Draw text
  ctx.font = '600 16px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
  ctx.fillStyle = 'white';
  ctx.textBaseline = 'middle';
  ctx.shadowColor = 'rgba(0, 0, 0, 0.5)';
  ctx.shadowBlur = 4;
  ctx.shadowOffsetX = 1;
  ctx.shadowOffsetY = 1;
  ctx.fillText('created with RallyCut', iconX + iconSize / 2 + 10, height / 2);

  ctx.restore();

  // Convert canvas to PNG blob
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (!blob) {
        reject(new Error('Failed to create watermark'));
        return;
      }
      blob.arrayBuffer().then((buffer) => {
        resolve(new Uint8Array(buffer));
      }).catch(reject);
    }, 'image/png');
  });
}

/**
 * Load watermark into FFmpeg virtual filesystem
 */
async function loadWatermark(ff: FFmpegInstance): Promise<void> {
  if (watermarkLoaded) {
    return;
  }

  const watermarkData = await generateWatermarkPng();
  await ff.writeFile(WATERMARK_FILENAME, watermarkData);
  watermarkLoaded = true;
}

// Local FFmpeg core files (served from public folder to avoid CORS issues with COOP/COEP)
const FFMPEG_CORE_URL = '/ffmpeg/ffmpeg-core.js';
const FFMPEG_WASM_URL = '/ffmpeg/ffmpeg-core.wasm';

/**
 * Check if SharedArrayBuffer is available (required for FFmpeg.wasm)
 */
export function isFFmpegSupported(): boolean {
  const supported = typeof SharedArrayBuffer !== 'undefined';
  if (!supported) {
    console.warn('SharedArrayBuffer not available. Make sure COOP/COEP headers are set and restart the dev server.');
  }
  return supported;
}

/**
 * Load FFmpeg script from local files (avoids bundler and CORS issues)
 */
async function loadFFmpegScript(): Promise<void> {
  // Check if already loaded
  if (window.FFmpegWASM) {
    return;
  }

  return new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = '/ffmpeg/ffmpeg.js';
    script.onload = () => resolve();
    script.onerror = () => reject(new Error('Failed to load FFmpeg script'));
    document.head.appendChild(script);
  });
}

/**
 * Initialize FFmpeg.wasm (lazy load on first use)
 */
async function getFFmpeg(onProgress?: ProgressCallback): Promise<FFmpegInstance> {
  // Always update the current progress callback so new exports use the right callback
  currentProgressCallback = onProgress || null;

  if (ffmpeg && ffmpeg.loaded) {
    return ffmpeg;
  }

  if (loadPromise) {
    await loadPromise;
    return ffmpeg!;
  }

  loadPromise = (async () => {
    onProgress?.(0, messages.loadingFFmpeg);

    try {
      // Load FFmpeg script via script tag
      await loadFFmpegScript();

      if (!window.FFmpegWASM?.FFmpeg) {
        throw new Error('FFmpeg not available after script load');
      }

      ffmpeg = new window.FFmpegWASM.FFmpeg();

      // Use indirect callback so we can update it for each export
      // Map FFmpeg's 0-1 progress to the current operation's range
      ffmpeg.on('progress', ({ progress }: { progress: number }) => {
        // FFmpeg progress should be between 0 and 1, but can sometimes be invalid
        if (typeof progress !== 'number' || !isFinite(progress) || progress < 0 || progress > 1) {
          return;
        }
        // Map FFmpeg's 0-100% to the current operation's range
        const { min, max } = ffmpegProgressRange;
        const mappedProgress = Math.round(min + progress * (max - min));
        if (mappedProgress > minProgress && mappedProgress <= 100) {
          minProgress = mappedProgress;
          currentProgressCallback?.(mappedProgress, messages.processing);
        }
      });


      // Load FFmpeg core using local files
      await ffmpeg.load({
        coreURL: FFMPEG_CORE_URL,
        wasmURL: FFMPEG_WASM_URL,
      });

      onProgress?.(5, messages.ffmpegLoaded);
    } catch (err) {
      console.error('Failed to load FFmpeg:', err);
      loadPromise = null;
      ffmpeg = null;

      // Provide more specific error messages
      const errorMsg = err instanceof Error ? err.message : String(err);
      if (errorMsg.includes('SharedArrayBuffer')) {
        throw new Error('SharedArrayBuffer not available. Please restart the dev server to apply COOP/COEP headers.');
      } else if (errorMsg.includes('CORS') || errorMsg.includes('cross-origin')) {
        throw new Error('Failed to load FFmpeg due to CORS restrictions.');
      } else {
        throw new Error(`Failed to load FFmpeg: ${errorMsg}`);
      }
    }
  })();

  await loadPromise;
  // ffmpeg is guaranteed to be set after loadPromise resolves successfully
  // (if it failed, loadPromise would have thrown)
  return ffmpeg!;
}

/**
 * Get a presigned download URL for a video stored in S3.
 * This bypasses CORS issues when fetching videos for export.
 */
async function getPresignedDownloadUrl(s3Key: string): Promise<string> {
  const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001';
  const visitorId = getVisitorId();

  const headers: HeadersInit = {};
  if (visitorId) {
    headers['X-Visitor-Id'] = visitorId;
  }

  const response = await fetch(
    `${apiUrl}/v1/videos/download-url?s3Key=${encodeURIComponent(s3Key)}`,
    { headers }
  );

  if (!response.ok) {
    throw new Error(`Failed to get download URL: ${response.status}`);
  }

  const data = await response.json();
  return data.downloadUrl;
}

/**
 * Fetch file utility - works with File objects and URLs
 * For relative video paths (/videos/...), uses presigned S3 URLs to bypass CORS
 * Shows download progress for large files
 * Uses in-memory cache to avoid re-downloading the same video
 */
async function fetchFileData(source: VideoSource, onProgress?: (loaded: number, total: number) => void): Promise<Uint8Array> {
  if (source instanceof File) {
    // For File objects, use the file name as cache key
    const cacheKey = `file:${source.name}:${source.size}`;
    const cached = await getCachedVideo(cacheKey);
    if (cached) {
      console.log(`[VideoExport] Cache hit for file ${source.name} (${(cached.length / 1024 / 1024).toFixed(1)} MB)`);
      onProgress?.(cached.length, cached.length);
      // Return a copy (.slice()) to avoid ArrayBuffer detachment issues with FFmpeg worker
      return cached.slice();
    }
    console.log(`[VideoExport] Cache miss for file ${source.name}, reading...`);

    const buffer = await source.arrayBuffer();
    const data = new Uint8Array(buffer);
    await cacheVideo(cacheKey, data);
    // Return a copy to avoid ArrayBuffer detachment issues
    return data.slice();
  } else {
    // Check cache first using the source URL as key
    const cacheKey = source;
    const cached = await getCachedVideo(cacheKey);
    if (cached) {
      console.log(`[VideoExport] Cache hit for ${cacheKey.substring(0, 50)}... (${(cached.length / 1024 / 1024).toFixed(1)} MB)`);
      onProgress?.(cached.length, cached.length);
      // Return a copy (.slice()) to avoid ArrayBuffer detachment issues with FFmpeg worker
      return cached.slice();
    }
    console.log(`[VideoExport] Cache miss for ${cacheKey.substring(0, 50)}..., downloading...`);

    let url = source;

    // If it's a relative video path, get a presigned S3 URL
    if (source.startsWith('/videos/')) {
      // Extract s3Key from the path (remove leading /)
      const s3Key = source.slice(1);
      url = await getPresignedDownloadUrl(s3Key);
    }

    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to fetch video: ${response.status}`);
    }

    // If we have content-length and a body stream, track progress
    const contentLength = response.headers.get('content-length');
    if (contentLength && response.body && onProgress) {
      const total = parseInt(contentLength, 10);
      const reader = response.body.getReader();
      const chunks: Uint8Array[] = [];
      let loaded = 0;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        chunks.push(value);
        loaded += value.length;
        onProgress(loaded, total);
      }

      // Combine chunks into single Uint8Array
      const result = new Uint8Array(loaded);
      let offset = 0;
      for (const chunk of chunks) {
        result.set(chunk, offset);
        offset += chunk.length;
      }

      // Cache the result and return a copy
      await cacheVideo(cacheKey, result);
      return result.slice();
    }

    // Fallback: no progress tracking
    const buffer = await response.arrayBuffer();
    const data = new Uint8Array(buffer);
    await cacheVideo(cacheKey, data);
    // Return a copy to avoid ArrayBuffer detachment issues
    return data.slice();
  }
}

/**
 * Get video file extension from File object or URL
 */
function getVideoExtension(source: VideoSource): string {
  if (typeof source === 'string') {
    // Extract extension from URL path
    const path = source.split('?')[0];
    const ext = path.split('.').pop()?.toLowerCase();
    return ext || 'mp4';
  }
  const ext = source.name.split('.').pop()?.toLowerCase();
  return ext || 'mp4';
}

/**
 * Get video name from source for filename generation
 */
function getVideoName(source: VideoSource): string {
  if (typeof source === 'string') {
    const path = source.split('?')[0];
    const filename = path.split('/').pop() || 'video';
    return filename.replace(/\.[^/.]+$/, '');
  }
  return source.name.replace(/\.[^/.]+$/, '');
}

/**
 * Export a single rally from the video
 * @param withWatermark - Whether to add "created with RallyCut" watermark (default: true)
 */
export async function exportSingleRally(
  videoSource: VideoSource,
  rally: Rally,
  withWatermark: boolean = true,
  onProgress?: ProgressCallback
): Promise<Blob> {
  if (!isFFmpegSupported()) {
    throw new Error('Your browser does not support video export. Please use Chrome, Firefox, or Edge.');
  }

  resetProgress();
  const ff = await getFFmpeg(onProgress);
  const ext = getVideoExtension(videoSource);
  const inputName = `input.${ext}`;
  const outputName = 'output.mp4'; // Always output mp4 when re-encoding

  checkCancelled();
  reportProgress(10, messages.loadingVideo);

  // Write input file to FFmpeg virtual filesystem
  try {
    const videoData = await fetchFileData(videoSource, (loaded, total) => {
      // Map download progress from 10% to 25%
      const downloadProgress = 10 + Math.round((loaded / total) * 15);
      const mb = (loaded / 1024 / 1024).toFixed(1);
      const totalMb = (total / 1024 / 1024).toFixed(1);
      reportProgress(downloadProgress, `Downloading ${mb}/${totalMb} MB...`);
    });
    checkCancelled();
    reportProgress(26, 'Loading video into memory...');
    await ff.writeFile(inputName, videoData);
  } catch (err) {
    console.error('Failed to load video:', err);
    throw new Error('Failed to load video. The file may be too large for browser processing (max ~500MB recommended).');
  }

  checkCancelled();
  reportProgress(30, messages.extractingClip(1, 1));

  // Set progress range for FFmpeg encoding (30% to 90%)
  setFFmpegProgressRange(30, 90);

  if (withWatermark) {
    // Load watermark for overlay
    await loadWatermark(ff);

    // Re-encode with watermark overlay (bottom-right corner with padding)
    await ff.exec([
      '-ss', rally.start_time.toString(),
      '-i', inputName,
      '-i', WATERMARK_FILENAME,
      '-t', rally.duration.toString(),
      '-filter_complex', '[0:v][1:v]overlay=W-w-20:H-h-20:format=auto[v]',
      '-map', '[v]',
      '-map', '0:a?',
      ...FFMPEG_ENCODE_SETTINGS,
      '-avoid_negative_ts', 'make_zero',
      '-y',
      outputName,
    ]);
  } else {
    // Fast path: stream copy without watermark
    const noWatermarkOutput = `output.${ext}`;
    await ff.exec([
      '-ss', rally.start_time.toString(),
      '-i', inputName,
      '-t', rally.duration.toString(),
      '-c', 'copy',
      '-avoid_negative_ts', 'make_zero',
      noWatermarkOutput,
    ]);

    checkCancelled();
    reportProgress(90, messages.finalizing);

    const data = await ff.readFile(noWatermarkOutput);
    await ff.deleteFile(inputName);
    await ff.deleteFile(noWatermarkOutput);

    reportProgress(100, messages.complete);
    const uint8Data = data instanceof Uint8Array ? new Uint8Array(data) : data;
    return new Blob([uint8Data], { type: `video/${ext}` });
  }

  checkCancelled();
  reportProgress(90, messages.finalizing);

  // Read output file
  const data = await ff.readFile(outputName);

  // Cleanup
  await ff.deleteFile(inputName);
  await ff.deleteFile(outputName);

  reportProgress(100, messages.complete);

  // Convert to regular Uint8Array to satisfy TypeScript (FFmpeg returns Uint8Array with SharedArrayBuffer)
  const uint8Data = data instanceof Uint8Array ? new Uint8Array(data) : data;
  return new Blob([uint8Data], { type: 'video/mp4' });
}

/**
 * Export multiple rallies concatenated into a single video
 * @param withWatermark - Whether to add "created with RallyCut" watermark (default: true)
 */
export async function exportConcatenated(
  videoSource: VideoSource,
  rallies: Rally[],
  withFade: boolean,
  withWatermark: boolean = true,
  onProgress?: ProgressCallback
): Promise<Blob> {
  if (!isFFmpegSupported()) {
    throw new Error('Your browser does not support video export. Please use Chrome, Firefox, or Edge.');
  }

  if (rallies.length === 0) {
    throw new Error('No rallies to export');
  }

  // For single rally, use the simpler method
  if (rallies.length === 1) {
    return exportSingleRally(videoSource, rallies[0], withWatermark, onProgress);
  }

  resetProgress();
  const ff = await getFFmpeg(onProgress);
  const ext = getVideoExtension(videoSource);
  const inputName = `input.${ext}`;
  // Always output mp4 when using watermark (requires re-encoding)
  const outputName = withWatermark ? 'output.mp4' : `output.${ext}`;

  reportProgress(5, messages.loadingVideo);

  // Write input file
  try {
    const videoData = await fetchFileData(videoSource, (loaded, total) => {
      // Map download progress from 5% to 12%
      const downloadProgress = 5 + Math.round((loaded / total) * 7);
      const mb = (loaded / 1024 / 1024).toFixed(1);
      const totalMb = (total / 1024 / 1024).toFixed(1);
      reportProgress(downloadProgress, `Downloading ${mb}/${totalMb} MB...`);
    });
    reportProgress(13, 'Loading video into memory...');
    await ff.writeFile(inputName, videoData);
  } catch (err) {
    console.error('Failed to load video:', err);
    throw new Error('Failed to load video. The file may be too large for browser processing (max ~500MB recommended).');
  }

  // Load watermark if needed
  if (withWatermark) {
    await loadWatermark(ff);
  }

  if (withFade) {
    // With fade transitions - requires re-encoding
    await exportWithFade(ff, inputName, outputName, rallies, withWatermark);
  } else {
    // Without transitions - use concat demuxer (fast if no watermark)
    await exportWithConcat(ff, inputName, outputName, ext, rallies, withWatermark);
  }

  reportProgress(95, messages.finalizing);

  // Read output file
  const data = await ff.readFile(outputName);

  // Cleanup input and output
  await ff.deleteFile(inputName);
  await ff.deleteFile(outputName);

  reportProgress(100, messages.complete);

  // Convert to regular Uint8Array to satisfy TypeScript (FFmpeg returns Uint8Array with SharedArrayBuffer)
  const uint8Data = data instanceof Uint8Array ? new Uint8Array(data) : data;
  return new Blob([uint8Data], { type: withWatermark ? 'video/mp4' : `video/${ext}` });
}

/**
 * Export rallies from multiple video sources concatenated into a single video
 * @param withWatermark - Whether to add "created with RallyCut" watermark (default: true)
 */
export async function exportMultiSourceConcatenated(
  ralliesWithSource: RallyWithSource[],
  withFade: boolean,
  withWatermark: boolean = true,
  onProgress?: ProgressCallback
): Promise<Blob> {
  if (!isFFmpegSupported()) {
    throw new Error('Your browser does not support video export. Please use Chrome, Firefox, or Edge.');
  }

  if (ralliesWithSource.length === 0) {
    throw new Error('No rallies to export');
  }

  // For single rally, use simpler method
  if (ralliesWithSource.length === 1) {
    const { rally, videoSource } = ralliesWithSource[0];
    return exportSingleRally(videoSource, rally, withWatermark, onProgress);
  }

  // Check if all rallies are from the same source
  const firstSource = ralliesWithSource[0].videoSource;
  const allSameSource = ralliesWithSource.every(
    (r) => r.videoSource === firstSource
  );

  if (allSameSource) {
    // Use optimized single-source export
    return exportConcatenated(
      firstSource,
      ralliesWithSource.map((r) => r.rally),
      withFade,
      withWatermark,
      onProgress
    );
  }

  // Multi-source export: need to load each video and extract clips
  resetProgress();
  const ff = await getFFmpeg(onProgress);

  // Load watermark if needed
  if (withWatermark) {
    await loadWatermark(ff);
  }

  reportProgress(5, messages.loadingVideo);

  // Group rallies by their video source for efficient loading
  const sourceToRallies = new Map<VideoSource, { rally: Rally; index: number }[]>();
  ralliesWithSource.forEach(({ rally, videoSource }, index) => {
    const existing = sourceToRallies.get(videoSource) || [];
    existing.push({ rally, index });
    sourceToRallies.set(videoSource, existing);
  });

  // Load all unique video sources and extract clips
  const clipNames: string[] = new Array(ralliesWithSource.length);
  let processedCount = 0;
  const totalRallies = ralliesWithSource.length;

  for (const [videoSource, rallies] of sourceToRallies) {
    const ext = getVideoExtension(videoSource);
    const inputName = `input_${processedCount}.${ext}`;

    // Load this video source
    const sourceIndex = Array.from(sourceToRallies.keys()).indexOf(videoSource) + 1;
    const baseProgress = Math.round(5 + (processedCount / totalRallies) * 10);
    reportProgress(baseProgress, messages.loadingVideos(sourceIndex, sourceToRallies.size));

    try {
      const videoData = await fetchFileData(videoSource, (loaded, total) => {
        const mb = (loaded / 1024 / 1024).toFixed(1);
        const totalMb = (total / 1024 / 1024).toFixed(1);
        reportProgress(baseProgress, `Downloading video ${sourceIndex}/${sourceToRallies.size} (${mb}/${totalMb} MB)...`);
      });
      await ff.writeFile(inputName, videoData);
    } catch (err) {
      console.error('Failed to load video:', err);
      throw new Error('Failed to load video. The file may be too large for browser processing.');
    }

    // Extract clips from this video
    // Always re-encode for multi-source to ensure consistent format and keyframe alignment
    for (const { rally, index } of rallies) {
      const clipName = `clip_${index}.mp4`;
      clipNames[index] = clipName;

      const progressBase = 15 + ((processedCount + rallies.indexOf({ rally, index })) / totalRallies) * 50;
      reportProgress(Math.round(progressBase), messages.encodingClip(processedCount + 1, totalRallies));

      // Re-encode to ensure consistent format across different source videos
      // and proper keyframe alignment to avoid flickering
      await ff.exec([
        '-ss', rally.start_time.toString(),
        '-i', inputName,
        '-t', rally.duration.toString(),
        ...FFMPEG_ENCODE_SETTINGS,
        '-avoid_negative_ts', 'make_zero',
        '-y',
        clipName,
      ]);
      processedCount++;
    }

    // Cleanup this input file to save memory
    await ff.deleteFile(inputName);
  }

  // Now concatenate all clips
  const outputName = 'output.mp4';

  if (withFade) {
    reportProgress(70, messages.applyingFade);
    await exportMultiSourceWithFade(ff, clipNames, outputName, ralliesWithSource.map((r) => r.rally), withWatermark);
  } else {
    reportProgress(70, messages.joiningClips);
    // Use concat demuxer - stream copy if no watermark, re-encode with overlay if watermark
    const concatList = clipNames.map((name) => `file '${name}'`).join('\n');
    await ff.writeFile('concat.txt', concatList);

    if (withWatermark) {
      // Concatenate with watermark overlay
      const tempConcat = 'temp_concat.mp4';
      await ff.exec([
        '-f', 'concat',
        '-safe', '0',
        '-i', 'concat.txt',
        '-c', 'copy',
        tempConcat,
      ]);

      // Apply watermark overlay
      await ff.exec([
        '-i', tempConcat,
        '-i', WATERMARK_FILENAME,
        '-filter_complex', '[0:v][1:v]overlay=W-w-20:H-h-20:format=auto[v]',
        '-map', '[v]',
        '-map', '0:a?',
        ...FFMPEG_ENCODE_SETTINGS,
        '-y',
        outputName,
      ]);

      await ff.deleteFile(tempConcat);
    } else {
      await ff.exec([
        '-f', 'concat',
        '-safe', '0',
        '-i', 'concat.txt',
        '-c', 'copy',
        outputName,
      ]);
    }

    await ff.deleteFile('concat.txt');
  }

  reportProgress(95, messages.finalizing);

  // Read output
  const data = await ff.readFile(outputName);

  // Cleanup clips
  for (const clipName of clipNames) {
    await ff.deleteFile(clipName);
  }
  await ff.deleteFile(outputName);

  reportProgress(100, messages.complete);

  const uint8Data = data instanceof Uint8Array ? new Uint8Array(data) : data;
  return new Blob([uint8Data], { type: 'video/mp4' });
}

/**
 * Apply crossfade transitions for multi-source export (clips already extracted)
 * Uses xfade filter for smooth blending between clips
 */
async function exportMultiSourceWithFade(
  ff: FFmpegInstance,
  clipNames: string[],
  outputName: string,
  rallies: Rally[],
  withWatermark: boolean
): Promise<void> {
  const fadeDuration = FADE_DURATION;
  // If watermark, output to temp file first, then apply watermark
  const fadeOutputName = withWatermark ? 'temp_multi_faded.mp4' : outputName;

  // For 2 clips, use simple xfade
  if (clipNames.length === 2) {
    const offset = Math.max(0, rallies[0].duration - fadeDuration);

    await ff.exec([
      '-i', clipNames[0],
      '-i', clipNames[1],
      '-filter_complex',
      `[0:v][1:v]xfade=transition=fade:duration=${fadeDuration}:offset=${offset},format=yuv420p[v];[0:a][1:a]acrossfade=d=${fadeDuration}[a]`,
      '-map', '[v]',
      '-map', '[a]',
      ...FFMPEG_VIDEO_CODEC,
      '-c:a', 'aac',
      '-movflags', '+faststart',
      '-y',
      fadeOutputName,
    ]);
  } else {
    // For 3+ clips, chain xfade operations iteratively (process pairs)
    let currentInput = clipNames[0];
    let currentDuration = rallies[0].duration;

    for (let i = 1; i < clipNames.length; i++) {
      const nextClip = clipNames[i];
      const isLast = i === clipNames.length - 1;
      const tempOutput = isLast ? fadeOutputName : `xfade_temp_${i}.mp4`;
      const offset = Math.max(0, currentDuration - fadeDuration);

      await ff.exec([
        '-i', currentInput,
        '-i', nextClip,
        '-filter_complex',
        `[0:v][1:v]xfade=transition=fade:duration=${fadeDuration}:offset=${offset},format=yuv420p[v];[0:a][1:a]acrossfade=d=${fadeDuration}[a]`,
        '-map', '[v]',
        '-map', '[a]',
        ...FFMPEG_VIDEO_CODEC,
        '-c:a', 'aac',
        '-movflags', '+faststart',
        '-y',
        tempOutput,
      ]);

      // Clean up previous temp file (not the original clips)
      if (i > 1) {
        await ff.deleteFile(currentInput);
      }

      currentInput = tempOutput;
      // New duration = previous duration + next clip duration - fade overlap
      currentDuration = currentDuration + rallies[i].duration - fadeDuration;
    }
  }

  // Apply watermark if enabled
  if (withWatermark) {
    await ff.exec([
      '-i', fadeOutputName,
      '-i', WATERMARK_FILENAME,
      '-filter_complex', '[0:v][1:v]overlay=W-w-20:H-h-20:format=auto[v]',
      '-map', '[v]',
      '-map', '0:a?',
      ...FFMPEG_ENCODE_SETTINGS,
      '-y',
      outputName,
    ]);
    await ff.deleteFile(fadeOutputName);
  }
}

/**
 * Export with concat demuxer (fast if no watermark, re-encode if watermark)
 */
async function exportWithConcat(
  ff: FFmpegInstance,
  inputName: string,
  outputName: string,
  ext: string,
  rallies: Rally[],
  withWatermark: boolean
): Promise<void> {
  // First, extract each clip
  const clipNames: string[] = [];

  for (let i = 0; i < rallies.length; i++) {
    const rally = rallies[i];
    const clipName = `clip_${i}.${ext}`;
    clipNames.push(clipName);

    const progressBase = 10 + (i / rallies.length) * 60;
    reportProgress(Math.round(progressBase), messages.extractingClip(i + 1, rallies.length));

    await ff.exec([
      '-ss', rally.start_time.toString(),
      '-i', inputName,
      '-t', rally.duration.toString(),
      '-c', 'copy',
      '-avoid_negative_ts', 'make_zero',
      clipName,
    ]);
  }

  reportProgress(75, messages.joiningClips);

  // Create concat list file
  const concatList = clipNames.map(name => `file '${name}'`).join('\n');
  await ff.writeFile('concat.txt', concatList);

  if (withWatermark) {
    // Concatenate first, then apply watermark overlay
    const tempConcat = 'temp_concat.mp4';
    await ff.exec([
      '-f', 'concat',
      '-safe', '0',
      '-i', 'concat.txt',
      '-c', 'copy',
      tempConcat,
    ]);

    // Apply watermark overlay
    await ff.exec([
      '-i', tempConcat,
      '-i', WATERMARK_FILENAME,
      '-filter_complex', '[0:v][1:v]overlay=W-w-20:H-h-20:format=auto[v]',
      '-map', '[v]',
      '-map', '0:a?',
      ...FFMPEG_ENCODE_SETTINGS,
      '-y',
      outputName,
    ]);

    await ff.deleteFile(tempConcat);
  } else {
    // Fast path: just concatenate without re-encoding
    await ff.exec([
      '-f', 'concat',
      '-safe', '0',
      '-i', 'concat.txt',
      '-c', 'copy',
      outputName,
    ]);
  }

  // Cleanup temp files
  await ff.deleteFile('concat.txt');
  for (const clipName of clipNames) {
    await ff.deleteFile(clipName);
  }
}

/**
 * Export with crossfade transitions (requires re-encoding)
 * Uses xfade filter for smooth blending between clips
 */
async function exportWithFade(
  ff: FFmpegInstance,
  inputName: string,
  outputName: string,
  rallies: Rally[],
  withWatermark: boolean
): Promise<void> {
  const fadeDuration = FADE_DURATION;
  // If watermark, output to temp file first, then apply watermark
  const fadeOutputName = withWatermark ? 'temp_faded.mp4' : outputName;

  // First extract all clips with consistent format
  const clipNames: string[] = [];

  for (let i = 0; i < rallies.length; i++) {
    const rally = rallies[i];
    const clipName = `clip_${i}.mp4`;
    clipNames.push(clipName);

    const progressBase = 10 + (i / rallies.length) * 40;
    reportProgress(Math.round(progressBase), messages.encodingClip(i + 1, rallies.length));

    await ff.exec([
      '-ss', rally.start_time.toString(),
      '-i', inputName,
      '-t', rally.duration.toString(),
      ...FFMPEG_ENCODE_SETTINGS,
      '-avoid_negative_ts', 'make_zero',
      '-y',
      clipName,
    ]);
  }

  reportProgress(55, messages.applyingFade);

  // For 2 clips, use simple xfade
  if (clipNames.length === 2) {
    const offset = Math.max(0, rallies[0].duration - fadeDuration);

    await ff.exec([
      '-i', clipNames[0],
      '-i', clipNames[1],
      '-filter_complex',
      `[0:v][1:v]xfade=transition=fade:duration=${fadeDuration}:offset=${offset},format=yuv420p[v];[0:a][1:a]acrossfade=d=${fadeDuration}[a]`,
      '-map', '[v]',
      '-map', '[a]',
      ...FFMPEG_VIDEO_CODEC,
      '-c:a', 'aac',
      '-movflags', '+faststart',
      '-y',
      fadeOutputName,
    ]);
  } else {
    // For 3+ clips, chain xfade operations iteratively
    let currentInput = clipNames[0];
    let currentDuration = rallies[0].duration;

    for (let i = 1; i < clipNames.length; i++) {
      const nextClip = clipNames[i];
      const isLast = i === clipNames.length - 1;
      const tempOutput = isLast ? fadeOutputName : `xfade_temp_${i}.mp4`;
      const offset = Math.max(0, currentDuration - fadeDuration);

      reportProgress(55 + Math.round((i / clipNames.length) * 35), messages.applyingFade);

      await ff.exec([
        '-i', currentInput,
        '-i', nextClip,
        '-filter_complex',
        `[0:v][1:v]xfade=transition=fade:duration=${fadeDuration}:offset=${offset},format=yuv420p[v];[0:a][1:a]acrossfade=d=${fadeDuration}[a]`,
        '-map', '[v]',
        '-map', '[a]',
        ...FFMPEG_VIDEO_CODEC,
        '-c:a', 'aac',
        '-movflags', '+faststart',
        '-y',
        tempOutput,
      ]);

      // Clean up previous temp file (not original clips yet)
      if (i > 1) {
        await ff.deleteFile(currentInput);
      }

      currentInput = tempOutput;
      currentDuration = currentDuration + rallies[i].duration - fadeDuration;
    }
  }

  // Apply watermark if enabled
  if (withWatermark) {
    await ff.exec([
      '-i', fadeOutputName,
      '-i', WATERMARK_FILENAME,
      '-filter_complex', '[0:v][1:v]overlay=W-w-20:H-h-20:format=auto[v]',
      '-map', '[v]',
      '-map', '0:a?',
      ...FFMPEG_ENCODE_SETTINGS,
      '-y',
      outputName,
    ]);
    await ff.deleteFile(fadeOutputName);
  }

  // Cleanup original clips
  for (const clipName of clipNames) {
    try {
      await ff.deleteFile(clipName);
    } catch {
      // Ignore if already deleted
    }
  }
}

/**
 * Trigger browser download of a blob
 */
export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// ============================================================================
// Server-Side Export (for large videos or premium quality)
// ============================================================================

import {
  createExportJob,
  getExportJobStatus,
  getExportDownloadUrl,
  type ExportTier,
  type ExportStatus,
} from '@/services/api';

export interface ServerExportOptions {
  sessionId: string;
  tier?: ExportTier;
  format?: 'mp4' | 'webm';
}

export interface ServerExportResult {
  jobId: string;
  status: ExportStatus;
  downloadUrl?: string;
  error?: string;
}

/**
 * Export rallies using server-side processing (Lambda).
 * Use this for large videos or when premium quality is needed.
 *
 * @param rallies - Array of rallies with their video metadata
 * @param options - Export options including sessionId and tier
 * @param onProgress - Progress callback (0-100)
 * @returns Export result with download URL when complete
 */
export async function exportServerSide(
  rallies: Array<{
    videoId: string;
    videoS3Key: string;
    startMs: number;
    endMs: number;
  }>,
  options: ServerExportOptions,
  onProgress?: ProgressCallback
): Promise<ServerExportResult> {
  if (rallies.length === 0) {
    throw new Error('No rallies to export');
  }

  onProgress?.(0, 'Submitting export job...');

  // Create the export job
  const job = await createExportJob({
    sessionId: options.sessionId,
    tier: options.tier || 'FREE',
    config: { format: options.format || 'mp4' },
    rallies,
  });

  onProgress?.(5, 'Export job started...');

  // Poll for completion
  const result = await pollExportJob(job.id, onProgress);

  if (result.status === 'COMPLETED') {
    // Get download URL
    const downloadResult = await getExportDownloadUrl(job.id);
    return {
      jobId: job.id,
      status: 'COMPLETED',
      downloadUrl: downloadResult.downloadUrl || undefined,
    };
  }

  if (result.status === 'FAILED') {
    return {
      jobId: job.id,
      status: 'FAILED',
      error: result.error || 'Export failed',
    };
  }

  return {
    jobId: job.id,
    status: result.status,
  };
}

/**
 * Poll an export job until completion or failure
 */
async function pollExportJob(
  jobId: string,
  onProgress?: ProgressCallback,
  maxAttempts = 300, // 10 minutes at 2s intervals
  interval = 2000
): Promise<{ status: ExportStatus; error?: string }> {
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    // Check cancellation
    if (isCancelled) {
      throw new Error('Export cancelled');
    }

    const job = await getExportJobStatus(jobId);

    // Map server progress (0-100) to our range (5-95)
    const mappedProgress = 5 + Math.round((job.progress / 100) * 90);
    onProgress?.(mappedProgress, getServerProgressMessage(job.progress));

    if (job.status === 'COMPLETED') {
      onProgress?.(100, 'Export complete!');
      return { status: 'COMPLETED' };
    }

    if (job.status === 'FAILED') {
      return { status: 'FAILED', error: job.error };
    }

    // Wait before next poll
    await new Promise((resolve) => setTimeout(resolve, interval));
  }

  return { status: 'FAILED', error: 'Export timed out' };
}

/**
 * Get a beach volleyball-themed progress message for server export
 */
function getServerProgressMessage(progress: number): string {
  if (progress < 10) return 'Warming up on the server...';
  if (progress < 30) return 'Downloading video clips...';
  if (progress < 60) return 'Extracting rallies...';
  if (progress < 80) return 'Processing video...';
  if (progress < 95) return 'Uploading final video...';
  return 'Finishing up...';
}

/**
 * Download a file from a URL
 */
export async function downloadFromUrl(
  url: string,
  filename: string,
  onProgress?: (loaded: number, total: number) => void
): Promise<void> {
  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(`Download failed: ${response.status}`);
  }

  const contentLength = response.headers.get('content-length');
  const total = contentLength ? parseInt(contentLength, 10) : 0;

  if (response.body && total > 0 && onProgress) {
    const reader = response.body.getReader();
    const chunks: Uint8Array[] = [];
    let loaded = 0;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      chunks.push(value);
      loaded += value.length;
      onProgress(loaded, total);
    }

    // Combine chunks
    const data = new Uint8Array(loaded);
    let offset = 0;
    for (const chunk of chunks) {
      data.set(chunk, offset);
      offset += chunk.length;
    }

    const blob = new Blob([data], { type: 'video/mp4' });
    downloadBlob(blob, filename);
  } else {
    // No progress tracking
    const blob = await response.blob();
    downloadBlob(blob, filename);
  }
}

export { getVideoName };
