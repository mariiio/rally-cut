import { Rally } from '@/types/rally';

export type VideoSource = File | string;

// For multi-source exports (highlights with rallies from different matches)
export interface RallyWithSource {
  rally: Rally;
  videoSource: VideoSource;
}

// Singleton FFmpeg instance
let ffmpeg: any = null;
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
    onProgress?.(0, messages.loadingFFmpeg);

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

  // Extract the clip using stream copy (fast, no re-encoding)
  await ff.exec([
    '-ss', rally.start_time.toString(),
    '-i', inputName,
    '-t', rally.duration.toString(),
    '-c', 'copy',
    '-avoid_negative_ts', 'make_zero',
    outputName,
  ]);

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

  reportProgress(5, messages.loadingVideo);

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

  reportProgress(95, messages.finalizing);

  // Read output file
  const data = await ff.readFile(outputName);

  // Cleanup input and output
  await ff.deleteFile(inputName);
  await ff.deleteFile(outputName);

  reportProgress(100, messages.complete);

  // Convert to regular Uint8Array to satisfy TypeScript (FFmpeg returns Uint8Array with SharedArrayBuffer)
  const uint8Data = data instanceof Uint8Array ? new Uint8Array(data) : data;
  return new Blob([uint8Data], { type: `video/${ext}` });
}

/**
 * Export rallies from multiple video sources concatenated into a single video
 */
export async function exportMultiSourceConcatenated(
  ralliesWithSource: RallyWithSource[],
  withFade: boolean,
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
    return exportSingleRally(videoSource, rally, onProgress);
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
      onProgress
    );
  }

  // Multi-source export: need to load each video and extract clips
  resetProgress();
  const ff = await getFFmpeg(onProgress);

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
      // yuv420p is required for consistent pixel format across clips
      await ff.exec([
        '-ss', rally.start_time.toString(),
        '-i', inputName,
        '-t', rally.duration.toString(),
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-crf', '23',
        '-pix_fmt', 'yuv420p',
        '-c:a', 'aac',
        '-b:a', '128k',
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
    await exportMultiSourceWithFade(ff, clipNames, outputName, ralliesWithSource.map((r) => r.rally));
  } else {
    reportProgress(70, messages.joiningClips);
    // Use concat demuxer with stream copy since all clips are now in consistent format
    const concatList = clipNames.map((name) => `file '${name}'`).join('\n');
    await ff.writeFile('concat.txt', concatList);

    await ff.exec([
      '-f', 'concat',
      '-safe', '0',
      '-i', 'concat.txt',
      '-c', 'copy',
      outputName,
    ]);

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
  ff: any,
  clipNames: string[],
  outputName: string,
  rallies: Rally[]
): Promise<void> {
  const fadeDuration = 0.5;

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
      '-c:v', 'libx264',
      '-preset', 'ultrafast',
      '-crf', '23',
      '-c:a', 'aac',
      '-movflags', '+faststart',
      '-y',
      outputName,
    ]);
    return;
  }

  // For 3+ clips, chain xfade operations iteratively (process pairs)
  let currentInput = clipNames[0];
  let currentDuration = rallies[0].duration;

  for (let i = 1; i < clipNames.length; i++) {
    const nextClip = clipNames[i];
    const isLast = i === clipNames.length - 1;
    const tempOutput = isLast ? outputName : `xfade_temp_${i}.mp4`;
    const offset = Math.max(0, currentDuration - fadeDuration);

    await ff.exec([
      '-i', currentInput,
      '-i', nextClip,
      '-filter_complex',
      `[0:v][1:v]xfade=transition=fade:duration=${fadeDuration}:offset=${offset},format=yuv420p[v];[0:a][1:a]acrossfade=d=${fadeDuration}[a]`,
      '-map', '[v]',
      '-map', '[a]',
      '-c:v', 'libx264',
      '-preset', 'ultrafast',
      '-crf', '23',
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
 * Export with crossfade transitions (requires re-encoding)
 * Uses xfade filter for smooth blending between clips
 */
async function exportWithFade(
  ff: any,
  inputName: string,
  outputName: string,
  rallies: Rally[]
): Promise<void> {
  const fadeDuration = 0.5;

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
      '-c:v', 'libx264',
      '-preset', 'ultrafast',
      '-crf', '23',
      '-pix_fmt', 'yuv420p',
      '-c:a', 'aac',
      '-b:a', '128k',
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
      '-c:v', 'libx264',
      '-preset', 'ultrafast',
      '-crf', '23',
      '-c:a', 'aac',
      '-movflags', '+faststart',
      '-y',
      outputName,
    ]);
  } else {
    // For 3+ clips, chain xfade operations iteratively
    let currentInput = clipNames[0];
    let currentDuration = rallies[0].duration;

    for (let i = 1; i < clipNames.length; i++) {
      const nextClip = clipNames[i];
      const isLast = i === clipNames.length - 1;
      const tempOutput = isLast ? outputName : `xfade_temp_${i}.mp4`;
      const offset = Math.max(0, currentDuration - fadeDuration);

      reportProgress(55 + Math.round((i / clipNames.length) * 35), messages.applyingFade);

      await ff.exec([
        '-i', currentInput,
        '-i', nextClip,
        '-filter_complex',
        `[0:v][1:v]xfade=transition=fade:duration=${fadeDuration}:offset=${offset},format=yuv420p[v];[0:a][1:a]acrossfade=d=${fadeDuration}[a]`,
        '-map', '[v]',
        '-map', '[a]',
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-crf', '23',
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
