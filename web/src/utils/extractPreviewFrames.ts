/**
 * Extract N evenly-spaced JPEG thumbnails from a local video File.
 * No external deps: uses <video> + <canvas>. ~100-300ms for a typical clip.
 */
export async function extractPreviewFrames(file: File, count = 5, maxWidth = 640): Promise<Blob[]> {
  const url = URL.createObjectURL(file);
  try {
    const video = document.createElement('video');
    video.muted = true;
    video.playsInline = true;
    video.src = url;
    await new Promise<void>((resolve, reject) => {
      video.onloadedmetadata = () => resolve();
      video.onerror = () => reject(new Error('Failed to load video metadata'));
    });
    const duration = video.duration || 0;
    if (duration < 1) throw new Error('Video too short to extract preview frames');

    const scale = Math.min(1, maxWidth / video.videoWidth);
    const w = Math.round(video.videoWidth * scale);
    const h = Math.round(video.videoHeight * scale);
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Canvas context unavailable');

    const blobs: Blob[] = [];
    for (let i = 0; i < count; i++) {
      const t = ((i + 0.5) / count) * duration;
      await new Promise<void>((resolve, reject) => {
        const onSeeked = () => {
          video.removeEventListener('seeked', onSeeked);
          resolve();
        };
        video.addEventListener('seeked', onSeeked, { once: true });
        video.addEventListener('error', () => reject(new Error('Video seek failed')), { once: true });
        video.currentTime = t;
      });
      ctx.drawImage(video, 0, 0, w, h);
      const blob: Blob = await new Promise((resolve, reject) => {
        canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('toBlob failed'))), 'image/jpeg', 0.85);
      });
      blobs.push(blob);
    }
    return blobs;
  } finally {
    URL.revokeObjectURL(url);
  }
}

export interface PreviewMetadata {
  width: number;
  height: number;
  durationS: number;
}

export async function readVideoMetadata(file: File): Promise<PreviewMetadata> {
  const url = URL.createObjectURL(file);
  try {
    const video = document.createElement('video');
    video.muted = true;
    video.playsInline = true;
    video.src = url;
    await new Promise<void>((resolve, reject) => {
      video.onloadedmetadata = () => resolve();
      video.onerror = () => reject(new Error('Failed to read metadata'));
    });
    return { width: video.videoWidth, height: video.videoHeight, durationS: video.duration };
  } finally {
    URL.revokeObjectURL(url);
  }
}
