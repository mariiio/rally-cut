/**
 * FFmpeg.wasm type definitions
 * These types cover the subset of FFmpeg.wasm API used in this project.
 */

export interface FFmpegProgressEvent {
  progress: number; // 0-1 range
  time?: number;
}

export interface FFmpegLoadConfig {
  coreURL: string;
  wasmURL: string;
}

/**
 * FFmpeg instance interface for browser-based video processing.
 * Based on @ffmpeg/ffmpeg but typed for our usage.
 */
export interface FFmpegInstance {
  /** Whether FFmpeg core has been loaded */
  loaded: boolean;

  /** Load FFmpeg core with specified config */
  load(config: FFmpegLoadConfig): Promise<void>;

  /** Execute FFmpeg command with arguments */
  exec(args: string[]): Promise<void>;

  /** Write file to FFmpeg virtual filesystem */
  writeFile(path: string, data: Uint8Array | string): Promise<void>;

  /** Read file from FFmpeg virtual filesystem */
  readFile(path: string): Promise<Uint8Array>;

  /** Delete file from FFmpeg virtual filesystem */
  deleteFile(path: string): Promise<void>;

  /** Register event handler */
  on(event: 'progress', callback: (event: FFmpegProgressEvent) => void): void;
  on(event: 'log', callback: (event: { message: string }) => void): void;

  /** Terminate FFmpeg instance */
  terminate(): void;
}

/**
 * Global FFmpeg WASM module interface (loaded via script tag)
 */
export interface FFmpegWASMModule {
  FFmpeg: new () => FFmpegInstance;
}

declare global {
  interface Window {
    FFmpegWASM?: FFmpegWASMModule;
  }
}
