'use client';

import { useEffect, useRef, RefObject, useMemo } from 'react';
import type { BallPosition, BallPhase } from '@/services/api';

// Phase colors matching volleyball semantics
const PHASE_COLORS: Record<string, string> = {
  serve: '#4CAF50',     // Green - start of play
  receive: '#2196F3',   // Blue - first touch (pass)
  set: '#FFC107',       // Amber - setter touch
  attack: '#f44336',    // Red - spike/hit
  dig: '#9C27B0',       // Purple - defensive save
  defense: '#2196F3',   // Blue (legacy alias for receive)
  transition: '#FFC107', // Amber (legacy alias for set)
  unknown: '#ed6c02',   // Orange - default
};

interface BallTrackOverlayProps {
  positions: BallPosition[];
  frameCount: number;
  rallyStartTime: number;
  rallyEndTime: number;
  videoRef: RefObject<HTMLVideoElement | null>;
  ballPhases?: BallPhase[];  // Optional ball phases for color coding
  showPhaseColors?: boolean; // Whether to use phase-based coloring
}

const TRAIL_DURATION = 0.5; // seconds
const MAX_TRAIL_DOTS = 15; // Maximum dots to render

export function BallTrackOverlay({
  positions,
  frameCount,
  rallyStartTime,
  rallyEndTime,
  videoRef,
  ballPhases = [],
  showPhaseColors = true,
}: BallTrackOverlayProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const dotsRef = useRef<HTMLDivElement[]>([]);
  const rafIdRef = useRef<number | undefined>(undefined);

  // Pre-calculate the absolute time and phase for each position (only when data changes)
  const positionsWithTime = useMemo(() => {
    // frameNumber is SEGMENT-RELATIVE (0 to frameCount-1) because tracking
    // runs on an extracted video segment, not the full video.
    // Map segment frame to absolute video time using rally boundaries.
    const rallyDuration = rallyEndTime - rallyStartTime;
    const maxFrame = Math.max(1, frameCount - 1);

    // Build frame-to-phase lookup
    const frameToPhase: Record<number, string> = {};
    if (showPhaseColors && ballPhases.length > 0) {
      for (const phase of ballPhases) {
        for (let f = phase.frameStart; f <= phase.frameEnd; f++) {
          frameToPhase[f] = phase.phase;
        }
      }
    }

    return positions.map((p) => ({
      ...p,
      // Map segment frame (0 to maxFrame) to video time (rallyStart to rallyEnd)
      absoluteTime: rallyStartTime + (p.frameNumber / maxFrame) * rallyDuration,
      phase: frameToPhase[p.frameNumber] || 'unknown',
    }));
  }, [positions, frameCount, rallyStartTime, rallyEndTime, ballPhases, showPhaseColors]);

  // Create dot elements once
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    // Clear existing dots
    container.innerHTML = '';
    dotsRef.current = [];

    // Create pool of dot elements
    for (let i = 0; i < MAX_TRAIL_DOTS; i++) {
      const dot = document.createElement('div');
      dot.style.cssText = `
        position: absolute;
        border-radius: 50%;
        transform: translate(-50%, -50%);
        pointer-events: none;
        will-change: transform, opacity, width, height;
        display: none;
      `;
      container.appendChild(dot);
      dotsRef.current.push(dot);
    }
  }, []);

  // Animation loop - direct DOM manipulation, no React re-renders
  useEffect(() => {
    const updatePositions = () => {
      const video = videoRef.current;
      if (!video || dotsRef.current.length === 0) {
        rafIdRef.current = requestAnimationFrame(updatePositions);
        return;
      }

      const videoTime = video.currentTime;

      // Find visible positions (within trail duration)
      const visible: typeof positionsWithTime = [];
      for (const p of positionsWithTime) {
        const timeDiff = videoTime - p.absoluteTime;
        if (timeDiff >= 0 && timeDiff < TRAIL_DURATION) {
          visible.push(p);
        }
      }

      // Find current position (closest to video time)
      let currentFrameNumber = -1;
      if (visible.length > 0) {
        let closest = visible[0];
        for (const p of visible) {
          if (Math.abs(p.absoluteTime - videoTime) < Math.abs(closest.absoluteTime - videoTime)) {
            closest = p;
          }
        }
        currentFrameNumber = closest.frameNumber;
      }

      // Update dot elements directly
      const dots = dotsRef.current;
      for (let i = 0; i < dots.length; i++) {
        const dot = dots[i];
        const pos = visible[i];

        if (!pos) {
          dot.style.display = 'none';
          continue;
        }

        const age = videoTime - pos.absoluteTime;
        const isCurrent = pos.frameNumber === currentFrameNumber;
        // Higher minimum opacity for better visibility
        const opacity = Math.max(0.4, 1 - age / TRAIL_DURATION) * Math.max(0.6, pos.confidence);
        // Larger dots for better visibility (current: 20px, trail: 12px min)
        const size = isCurrent ? 20 : 12 + (TRAIL_DURATION - age) * 12;

        // Get color based on phase (with type assertion for extended position)
        const posWithPhase = pos as { phase?: string };
        const phaseColor = PHASE_COLORS[posWithPhase.phase || 'unknown'] || PHASE_COLORS.unknown;

        dot.style.display = 'block';
        dot.style.left = `${pos.x * 100}%`;
        dot.style.top = `${pos.y * 100}%`;
        dot.style.width = `${size}px`;
        dot.style.height = `${size}px`;
        dot.style.opacity = String(opacity);
        dot.style.backgroundColor = phaseColor;
        // Stronger glow effect for visibility
        dot.style.boxShadow = isCurrent
          ? `0 0 15px ${phaseColor}, 0 0 30px ${phaseColor}`
          : `0 0 8px ${phaseColor}`;
        dot.style.border = isCurrent ? '3px solid white' : '1px solid rgba(255,255,255,0.5)';
      }

      rafIdRef.current = requestAnimationFrame(updatePositions);
    };

    rafIdRef.current = requestAnimationFrame(updatePositions);

    return () => {
      if (rafIdRef.current) {
        cancelAnimationFrame(rafIdRef.current);
      }
    };
  }, [videoRef, positionsWithTime]);

  return (
    <div
      ref={containerRef}
      style={{
        position: 'absolute',
        inset: 0,
        pointerEvents: 'none',
        zIndex: 15,
      }}
    />
  );
}
