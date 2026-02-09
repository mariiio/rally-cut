'use client';

import { useMemo, useEffect, useRef, useCallback } from 'react';
import { usePlayerTrackingStore, PlayerPosition } from '@/stores/playerTrackingStore';

interface PlayerOverlayProps {
  rallyId: string;
  rallyStartTime: number;
  videoRef: React.RefObject<HTMLVideoElement | null>;
  containerRef: React.RefObject<HTMLDivElement | null>;
  fps?: number;
}

// Default colors for tracks before player assignment
const TRACK_COLORS = [
  '#FF6B6B', // Red
  '#4ECDC4', // Teal
  '#45B7D1', // Blue
  '#96CEB4', // Green
];

// Linear interpolation helper
function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

// Maximum tracks we'll render
const MAX_TRACKS = 8;

export function PlayerOverlay({
  rallyId,
  rallyStartTime,
  videoRef,
  containerRef,
  fps: propFps = 30,
}: PlayerOverlayProps) {
  const overlayRef = useRef<HTMLDivElement>(null);
  const trackElementsRef = useRef<Map<number, HTMLDivElement>>(new Map());
  const rafIdRef = useRef<number | null>(null);
  const dimensionsRef = useRef({ width: 0, height: 0, offsetX: 0, offsetY: 0 });

  // Get track data from store
  const { playerTracks, showPlayerOverlay, selectedTrackId } = usePlayerTrackingStore();
  const track = playerTracks[rallyId];
  const fps = track?.tracksJson?.fps ?? propFps;

  // Build per-track sorted positions for interpolation
  const trackPositions = useMemo(() => {
    if (!track?.tracksJson?.tracks) {
      return new Map<number, PlayerPosition[]>();
    }

    const positions = new Map<number, PlayerPosition[]>();
    for (const playerTrack of track.tracksJson.tracks) {
      const sortedPositions = [...playerTrack.positions].sort((a, b) => a.frame - b.frame);
      positions.set(playerTrack.trackId, sortedPositions);
    }
    return positions;
  }, [track]);

  // Get interpolated position for a track at a given frame
  const getInterpolatedPosition = useCallback((positions: PlayerPosition[], currentFrame: number) => {
    if (positions.length === 0) return null;

    const maxDistanceFromDetection = 10;
    const firstFrame = positions[0].frame;
    const lastFrame = positions[positions.length - 1].frame;

    if (currentFrame < firstFrame - maxDistanceFromDetection) return null;
    if (currentFrame > lastFrame + maxDistanceFromDetection) return null;

    // Before first frame
    if (currentFrame <= firstFrame) {
      const pos = positions[0];
      return { x: pos.x, y: pos.y, w: pos.w, h: pos.h, courtX: pos.courtX, courtY: pos.courtY };
    }

    // After last frame
    if (currentFrame >= lastFrame) {
      const pos = positions[positions.length - 1];
      return { x: pos.x, y: pos.y, w: pos.w, h: pos.h, courtX: pos.courtX, courtY: pos.courtY };
    }

    // Binary search for position before current frame
    let lo = 0;
    let hi = positions.length - 1;
    while (lo < hi) {
      const mid = Math.ceil((lo + hi + 1) / 2);
      if (positions[mid].frame <= currentFrame) {
        lo = mid;
      } else {
        hi = mid - 1;
      }
    }

    const before = positions[lo];
    const after = positions[lo + 1];
    const gap = after.frame - before.frame;

    // Handle large gaps
    const maxGap = 15;
    if (gap > maxGap) {
      const distToBefore = currentFrame - before.frame;
      const distToAfter = after.frame - currentFrame;
      if (distToBefore <= maxDistanceFromDetection) {
        return { x: before.x, y: before.y, w: before.w, h: before.h, courtX: before.courtX, courtY: before.courtY };
      } else if (distToAfter <= maxDistanceFromDetection) {
        return { x: after.x, y: after.y, w: after.w, h: after.h, courtX: after.courtX, courtY: after.courtY };
      }
      return null;
    }

    // Interpolate
    const t = (currentFrame - before.frame) / gap;
    return {
      x: lerp(before.x, after.x, t),
      y: lerp(before.y, after.y, t),
      w: lerp(before.w, after.w, t),
      h: lerp(before.h, after.h, t),
      courtX: before.courtX !== undefined && after.courtX !== undefined
        ? lerp(before.courtX, after.courtX, t) : before.courtX ?? after.courtX,
      courtY: before.courtY !== undefined && after.courtY !== undefined
        ? lerp(before.courtY, after.courtY, t) : before.courtY ?? after.courtY,
    };
  }, []);

  // Create/update track elements when tracks change
  useEffect(() => {
    const overlay = overlayRef.current;
    if (!overlay) return;

    const currentTrackIds = new Set(trackPositions.keys());
    const existingElements = trackElementsRef.current;

    // Remove elements for tracks that no longer exist
    for (const [trackId, element] of existingElements) {
      if (!currentTrackIds.has(trackId)) {
        element.remove();
        existingElements.delete(trackId);
      }
    }

    // Create elements for new tracks
    let trackIndex = 0;
    for (const trackId of currentTrackIds) {
      if (trackIndex >= MAX_TRACKS) break;

      if (!existingElements.has(trackId)) {
        const color = TRACK_COLORS[(trackId - 1) % TRACK_COLORS.length];

        const trackEl = document.createElement('div');
        trackEl.className = 'player-track';
        trackEl.dataset.trackId = String(trackId);
        trackEl.style.cssText = `
          position: absolute;
          left: 0;
          top: 0;
          border: 2px solid ${color};
          border-radius: 4px;
          pointer-events: none;
          will-change: transform, width, height;
          display: none;
        `;

        // Track label
        const label = document.createElement('div');
        label.className = 'track-label';
        label.textContent = `Track ${trackId}`;
        label.style.cssText = `
          position: absolute;
          top: -24px;
          left: 0;
          background-color: ${color};
          color: white;
          padding: 2px 6px;
          border-radius: 4px;
          font-size: 11px;
          font-weight: 600;
          white-space: nowrap;
        `;
        trackEl.appendChild(label);

        // Court position label
        const courtLabel = document.createElement('div');
        courtLabel.className = 'court-label';
        courtLabel.style.cssText = `
          position: absolute;
          bottom: -18px;
          left: 0;
          background-color: rgba(0,0,0,0.7);
          color: white;
          padding: 1px 4px;
          border-radius: 2px;
          font-size: 9px;
          font-family: monospace;
          display: none;
        `;
        trackEl.appendChild(courtLabel);

        overlay.appendChild(trackEl);
        existingElements.set(trackId, trackEl);
      }
      trackIndex++;
    }
  }, [trackPositions]);

  // Update dimensions on resize
  useEffect(() => {
    const video = videoRef.current;
    const container = containerRef.current;
    if (!video || !container) return;

    const updateDimensions = () => {
      const containerRect = container.getBoundingClientRect();
      const videoAspect = video.videoWidth / video.videoHeight || 16/9;
      const containerAspect = containerRect.width / containerRect.height;

      let displayWidth: number;
      let displayHeight: number;

      if (containerAspect > videoAspect) {
        displayHeight = containerRect.height;
        displayWidth = displayHeight * videoAspect;
      } else {
        displayWidth = containerRect.width;
        displayHeight = displayWidth / videoAspect;
      }

      dimensionsRef.current = {
        width: displayWidth,
        height: displayHeight,
        offsetX: (containerRect.width - displayWidth) / 2,
        offsetY: (containerRect.height - displayHeight) / 2,
      };
    };

    updateDimensions();

    const resizeObserver = new ResizeObserver(updateDimensions);
    resizeObserver.observe(container);
    video.addEventListener('loadedmetadata', updateDimensions);

    return () => {
      resizeObserver.disconnect();
      video.removeEventListener('loadedmetadata', updateDimensions);
    };
  }, [videoRef, containerRef]);

  // Animation loop - direct DOM manipulation
  useEffect(() => {
    if (!showPlayerOverlay) {
      // Hide all elements when overlay is disabled
      for (const element of trackElementsRef.current.values()) {
        element.style.display = 'none';
      }
      return;
    }

    const video = videoRef.current;
    if (!video) return;

    const updatePositions = () => {
      const { width, height, offsetX, offsetY } = dimensionsRef.current;
      if (width === 0) {
        rafIdRef.current = requestAnimationFrame(updatePositions);
        return;
      }

      const videoTime = video.currentTime;
      const currentFrame = (videoTime - rallyStartTime) * fps;

      for (const [trackId, positions] of trackPositions) {
        const element = trackElementsRef.current.get(trackId);
        if (!element) continue;

        const pos = getInterpolatedPosition(positions, currentFrame);

        if (!pos) {
          element.style.display = 'none';
          continue;
        }

        const x = offsetX + (pos.x - pos.w / 2) * width;
        const y = offsetY + (pos.y - pos.h / 2) * height;
        const boxWidth = pos.w * width;
        const boxHeight = pos.h * height;

        element.style.display = 'block';
        element.style.transform = `translate3d(${x}px, ${y}px, 0)`;
        element.style.width = `${boxWidth}px`;
        element.style.height = `${boxHeight}px`;

        // Update selection styling
        const isSelected = selectedTrackId === trackId;
        const color = TRACK_COLORS[(trackId - 1) % TRACK_COLORS.length];
        element.style.boxShadow = isSelected
          ? `0 0 0 2px white, 0 0 10px ${color}`
          : '0 2px 4px rgba(0,0,0,0.3)';

        // Update court label
        const courtLabel = element.querySelector('.court-label') as HTMLDivElement;
        if (courtLabel) {
          if (pos.courtX !== undefined && pos.courtY !== undefined) {
            courtLabel.style.display = 'block';
            courtLabel.textContent = `${pos.courtX.toFixed(1)}m, ${pos.courtY.toFixed(1)}m`;
          } else {
            courtLabel.style.display = 'none';
          }
        }
      }

      rafIdRef.current = requestAnimationFrame(updatePositions);
    };

    rafIdRef.current = requestAnimationFrame(updatePositions);

    return () => {
      if (rafIdRef.current !== null) {
        cancelAnimationFrame(rafIdRef.current);
      }
    };
  }, [videoRef, rallyStartTime, fps, trackPositions, showPlayerOverlay, selectedTrackId, getInterpolatedPosition]);

  if (!track?.tracksJson) {
    return null;
  }

  return (
    <div
      ref={overlayRef}
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none',
        zIndex: 5,
      }}
    />
  );
}
