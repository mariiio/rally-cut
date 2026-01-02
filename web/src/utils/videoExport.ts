import { Rally } from '@/types/rally';
import type { FFmpegInstance } from '@/types/ffmpeg';

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
// Cancellation flag
let isCancelled = false;

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
  isCancelled = false;
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
      // Only report FFmpeg progress if it's valid and higher than our current minimum
      ffmpeg.on('progress', ({ progress }: { progress: number }) => {
        // FFmpeg progress should be between 0 and 1, but can sometimes be invalid
        if (typeof progress !== 'number' || !isFinite(progress) || progress < 0 || progress > 1) {
          return;
        }
        const ffmpegProgress = Math.round(progress * 100);
        if (ffmpegProgress > minProgress && ffmpegProgress <= 100) {
          currentProgressCallback?.(ffmpegProgress, messages.processing);
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
 * Fetch file utility - works with File objects and URLs
 */
async function fetchFileData(source: VideoSource): Promise<Uint8Array> {
  if (source instanceof File) {
    const buffer = await source.arrayBuffer();
    return new Uint8Array(buffer);
  } else {
    const response = await fetch(source);
    const buffer = await response.arrayBuffer();
    return new Uint8Array(buffer);
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
    const videoData = await fetchFileData(videoSource);
    checkCancelled();
    await ff.writeFile(inputName, videoData);
  } catch (err) {
    console.error('Failed to load video:', err);
    throw new Error('Failed to load video. The file may be too large for browser processing (max ~500MB recommended).');
  }

  checkCancelled();
  reportProgress(30, messages.extractingClip(1, 1));

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
    const videoData = await fetchFileData(videoSource);
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
    reportProgress(
      Math.round(5 + (processedCount / totalRallies) * 10),
      messages.loadingVideos(sourceIndex, sourceToRallies.size)
    );

    try {
      const videoData = await fetchFileData(videoSource);
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

export { getVideoName };
