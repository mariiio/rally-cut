import { Rally } from '@/types/rally';

export type VideoSource = File | string;

// Singleton FFmpeg instance
let ffmpeg: any = null;
let loadPromise: Promise<void> | null = null;
let currentProgressCallback: ProgressCallback | null = null;
// Track minimum progress to prevent FFmpeg from resetting our stage-based progress
let minProgress = 0;

export type ProgressCallback = (progress: number, step: string) => void;

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
 * Reset progress tracking for a new export
 */
function resetProgress() {
  minProgress = 0;
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
  if ((window as any).FFmpegWASM) {
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
async function getFFmpeg(onProgress?: ProgressCallback): Promise<any> {
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
    onProgress?.(0, 'Loading FFmpeg...');

    try {
      // Load FFmpeg script via script tag
      await loadFFmpegScript();

      const FFmpegWASM = (window as any).FFmpegWASM;
      if (!FFmpegWASM || !FFmpegWASM.FFmpeg) {
        throw new Error('FFmpeg not available after script load');
      }

      ffmpeg = new FFmpegWASM.FFmpeg();

      // Use indirect callback so we can update it for each export
      // Only report FFmpeg progress if it's valid and higher than our current minimum
      ffmpeg.on('progress', ({ progress }: { progress: number }) => {
        // FFmpeg progress should be between 0 and 1, but can sometimes be invalid
        if (typeof progress !== 'number' || !isFinite(progress) || progress < 0 || progress > 1) {
          return;
        }
        const ffmpegProgress = Math.round(progress * 100);
        if (ffmpegProgress > minProgress && ffmpegProgress <= 100) {
          currentProgressCallback?.(ffmpegProgress, 'Processing...');
        }
      });

      // Log FFmpeg messages for debugging
      ffmpeg.on('log', ({ message }: { message: string }) => {
        console.log('[FFmpeg]', message);
      });

      // Load FFmpeg core using local files
      await ffmpeg.load({
        coreURL: FFMPEG_CORE_URL,
        wasmURL: FFMPEG_WASM_URL,
      });

      onProgress?.(5, 'FFmpeg loaded');
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
  return ffmpeg;
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
 */
export async function exportSingleRally(
  videoSource: VideoSource,
  rally: Rally,
  onProgress?: ProgressCallback
): Promise<Blob> {
  if (!isFFmpegSupported()) {
    throw new Error('Your browser does not support video export. Please use Chrome, Firefox, or Edge.');
  }

  resetProgress();
  const ff = await getFFmpeg(onProgress);
  const ext = getVideoExtension(videoSource);
  const inputName = `input.${ext}`;
  const outputName = `output.${ext}`;

  reportProgress(10, 'Loading video...');

  // Write input file to FFmpeg virtual filesystem
  try {
    const videoData = await fetchFileData(videoSource);
    await ff.writeFile(inputName, videoData);
  } catch (err) {
    console.error('Failed to load video:', err);
    throw new Error('Failed to load video. The file may be too large for browser processing (max ~500MB recommended).');
  }

  reportProgress(30, 'Extracting clip...');

  // Extract the clip using stream copy (fast, no re-encoding)
  await ff.exec([
    '-ss', rally.start_time.toString(),
    '-i', inputName,
    '-t', rally.duration.toString(),
    '-c', 'copy',
    '-avoid_negative_ts', 'make_zero',
    outputName,
  ]);

  reportProgress(90, 'Finalizing...');

  // Read output file
  const data = await ff.readFile(outputName);

  // Cleanup
  await ff.deleteFile(inputName);
  await ff.deleteFile(outputName);

  reportProgress(100, 'Complete');

  // Convert to regular Uint8Array to satisfy TypeScript (FFmpeg returns Uint8Array with SharedArrayBuffer)
  const uint8Data = data instanceof Uint8Array ? new Uint8Array(data) : data;
  return new Blob([uint8Data], { type: `video/${ext}` });
}

/**
 * Export multiple rallies concatenated into a single video
 */
export async function exportConcatenated(
  videoSource: VideoSource,
  rallies: Rally[],
  withFade: boolean,
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
    return exportSingleRally(videoSource, rallies[0], onProgress);
  }

  resetProgress();
  const ff = await getFFmpeg(onProgress);
  const ext = getVideoExtension(videoSource);
  const inputName = `input.${ext}`;
  const outputName = `output.${ext}`;

  reportProgress(5, 'Loading video...');

  // Write input file
  try {
    const videoData = await fetchFileData(videoSource);
    await ff.writeFile(inputName, videoData);
  } catch (err) {
    console.error('Failed to load video:', err);
    throw new Error('Failed to load video. The file may be too large for browser processing (max ~500MB recommended).');
  }

  if (withFade) {
    // With fade transitions - requires re-encoding
    await exportWithFade(ff, inputName, outputName, rallies);
  } else {
    // Without transitions - use concat demuxer (fast, no re-encode)
    await exportWithConcat(ff, inputName, outputName, ext, rallies);
  }

  reportProgress(95, 'Finalizing...');

  // Read output file
  const data = await ff.readFile(outputName);

  // Cleanup input and output
  await ff.deleteFile(inputName);
  await ff.deleteFile(outputName);

  reportProgress(100, 'Complete');

  // Convert to regular Uint8Array to satisfy TypeScript (FFmpeg returns Uint8Array with SharedArrayBuffer)
  const uint8Data = data instanceof Uint8Array ? new Uint8Array(data) : data;
  return new Blob([uint8Data], { type: `video/${ext}` });
}

/**
 * Export with concat demuxer (fast, no re-encoding)
 */
async function exportWithConcat(
  ff: any,
  inputName: string,
  outputName: string,
  ext: string,
  rallies: Rally[]
): Promise<void> {
  // First, extract each clip
  const clipNames: string[] = [];

  for (let i = 0; i < rallies.length; i++) {
    const rally = rallies[i];
    const clipName = `clip_${i}.${ext}`;
    clipNames.push(clipName);

    const progressBase = 10 + (i / rallies.length) * 60;
    reportProgress(Math.round(progressBase), `Extracting clip ${i + 1}/${rallies.length}...`);

    await ff.exec([
      '-ss', rally.start_time.toString(),
      '-i', inputName,
      '-t', rally.duration.toString(),
      '-c', 'copy',
      '-avoid_negative_ts', 'make_zero',
      clipName,
    ]);
  }

  reportProgress(75, 'Joining clips...');

  // Create concat list file
  const concatList = clipNames.map(name => `file '${name}'`).join('\n');
  await ff.writeFile('concat.txt', concatList);

  // Concatenate using concat demuxer
  await ff.exec([
    '-f', 'concat',
    '-safe', '0',
    '-i', 'concat.txt',
    '-c', 'copy',
    outputName,
  ]);

  // Cleanup temp files
  await ff.deleteFile('concat.txt');
  for (const clipName of clipNames) {
    await ff.deleteFile(clipName);
  }
}

/**
 * Export with fade transitions (requires re-encoding)
 */
async function exportWithFade(
  ff: any,
  inputName: string,
  outputName: string,
  rallies: Rally[]
): Promise<void> {
  const fadeDuration = 0.5;

  console.log('[exportWithFade] Starting fade export with', rallies.length, 'rallies');

  // For fade transitions, we need to build a complex filter graph
  // First extract all clips with re-encoding to ensure consistent format
  const clipNames: string[] = [];

  for (let i = 0; i < rallies.length; i++) {
    const rally = rallies[i];
    const clipName = `clip_${i}.mp4`;
    clipNames.push(clipName);

    const progressBase = 10 + (i / rallies.length) * 40;
    reportProgress(Math.round(progressBase), `Encoding clip ${i + 1}/${rallies.length}...`);
    console.log(`[exportWithFade] Encoding clip ${i + 1}/${rallies.length}, start: ${rally.start_time}, duration: ${rally.duration}`);

    // Re-encode to ensure consistent format for xfade
    // Use ultrafast preset for better browser performance
    const ret = await ff.exec([
      '-ss', rally.start_time.toString(),
      '-i', inputName,
      '-t', rally.duration.toString(),
      '-c:v', 'libx264',
      '-preset', 'ultrafast',
      '-crf', '28',
      '-c:a', 'aac',
      '-b:a', '128k',
      '-avoid_negative_ts', 'make_zero',
      '-y',
      clipName,
    ]);
    console.log(`[exportWithFade] Clip ${i + 1} encoding result:`, ret);
  }

  reportProgress(55, 'Applying fade transitions...');
  console.log('[exportWithFade] Applying xfade transitions...');

  // Build xfade filter chain
  // For N clips, we need N-1 xfade operations
  if (clipNames.length === 2) {
    // Simple case: just two clips
    const clip0Duration = rallies[0].duration;
    const offset = Math.max(0, clip0Duration - fadeDuration);
    const filterComplex = `[0:v][1:v]xfade=transition=fade:duration=${fadeDuration}:offset=${offset}[v];[0:a][1:a]acrossfade=d=${fadeDuration}[a]`;

    console.log('[exportWithFade] 2-clip filter:', filterComplex);

    const ret = await ff.exec([
      '-i', clipNames[0],
      '-i', clipNames[1],
      '-filter_complex', filterComplex,
      '-map', '[v]',
      '-map', '[a]',
      '-c:v', 'libx264',
      '-preset', 'ultrafast',
      '-crf', '28',
      '-c:a', 'aac',
      '-y',
      outputName,
    ]);
    console.log('[exportWithFade] 2-clip xfade result:', ret);
  } else {
    // Multiple clips: chain xfade operations
    // Build input arguments
    const inputs: string[] = [];
    for (const clipName of clipNames) {
      inputs.push('-i', clipName);
    }

    // Build filter complex for chained xfade
    let filterComplex = '';
    let videoLabel = '0:v';
    let audioLabel = '0:a';
    let cumulativeDuration = rallies[0].duration;

    for (let i = 1; i < clipNames.length; i++) {
      const offset = Math.max(0, cumulativeDuration - fadeDuration);
      const outVideoLabel = i === clipNames.length - 1 ? '[v]' : `[v${i}]`;
      const outAudioLabel = i === clipNames.length - 1 ? '[a]' : `[a${i}]`;

      filterComplex += `[${videoLabel}][${i}:v]xfade=transition=fade:duration=${fadeDuration}:offset=${offset}${outVideoLabel};`;
      filterComplex += `[${audioLabel}][${i}:a]acrossfade=d=${fadeDuration}${outAudioLabel};`;

      videoLabel = outVideoLabel.replace('[', '').replace(']', '');
      audioLabel = outAudioLabel.replace('[', '').replace(']', '');
      cumulativeDuration += rallies[i].duration - fadeDuration;
    }

    // Remove trailing semicolon
    filterComplex = filterComplex.slice(0, -1);
    console.log('[exportWithFade] Multi-clip filter:', filterComplex);

    reportProgress(60, 'Applying fade transitions...');

    const ret = await ff.exec([
      ...inputs,
      '-filter_complex', filterComplex,
      '-map', '[v]',
      '-map', '[a]',
      '-c:v', 'libx264',
      '-preset', 'ultrafast',
      '-crf', '28',
      '-c:a', 'aac',
      '-y',
      outputName,
    ]);
    console.log('[exportWithFade] Multi-clip xfade result:', ret);
  }

  // Cleanup temp clips
  for (const clipName of clipNames) {
    await ff.deleteFile(clipName);
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

/**
 * Format a filename for download
 */
export function formatExportFilename(baseName: string, rallies: Rally[]): string {
  const date = new Date().toISOString().split('T')[0];
  if (rallies.length === 1) {
    return `${baseName}_rally_${rallies[0].id}_${date}.mp4`;
  }
  return `${baseName}_${rallies.length}rallies_${date}.mp4`;
}

export { getVideoName };
