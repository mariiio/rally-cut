'use client';

import { useEffect, useRef, useMemo, RefObject } from 'react';
import type { BallPosition } from '@/services/api';
import type { Corner } from '@/stores/playerTrackingStore';

// Team colors
const TEAM_A_COLOR = 'rgba(244, 67, 54, 0.85)'; // Red (near court)
const TEAM_B_COLOR = 'rgba(33, 150, 243, 0.85)'; // Blue (far court)
const TEAM_A_ZONE = 'rgba(244, 67, 54, 0.12)';
const TEAM_B_ZONE = 'rgba(33, 150, 243, 0.12)';
const SPLIT_LINE_COLOR = '#00BCD4'; // Cyan

interface CourtDebugOverlayProps {
  corners?: Corner[];
  courtSplitY?: number;
  ballPositions?: BallPosition[];
  fps: number;
  rallyStartTime: number;
  videoRef: RefObject<HTMLVideoElement | null>;
}

export function CourtDebugOverlay({
  corners,
  courtSplitY,
  ballPositions,
  fps,
  rallyStartTime,
  videoRef,
}: CourtDebugOverlayProps) {
  const badgeRef = useRef<HTMLDivElement>(null);

  // Pre-calculate ball absolute times
  const ballWithTime = useMemo(() => {
    if (!ballPositions) return [];
    return ballPositions.map((p) => ({
      ...p,
      absoluteTime: rallyStartTime + p.frameNumber / fps,
    }));
  }, [ballPositions, fps, rallyStartTime]);

  // RVFC loop for ball side badge
  useEffect(() => {
    const video = videoRef.current;
    const badge = badgeRef.current;
    if (!video || !badge || !courtSplitY || ballWithTime.length === 0) {
      if (badge) badge.style.display = 'none';
      return;
    }

    let rvfcId: number | undefined;

    const render = (videoTime: number) => {
      // Binary search for closest ball position (ballWithTime sorted by absoluteTime)
      let lo = 0;
      let hi = ballWithTime.length - 1;
      while (lo < hi) {
        const mid = (lo + hi) >>> 1;
        if (ballWithTime[mid].absoluteTime < videoTime) {
          lo = mid + 1;
        } else {
          hi = mid;
        }
      }
      // lo is the first position >= videoTime; check lo and lo-1 for closest
      let closest = ballWithTime[lo];
      if (lo > 0) {
        const prev = ballWithTime[lo - 1];
        if (Math.abs(prev.absoluteTime - videoTime) < Math.abs(closest.absoluteTime - videoTime)) {
          closest = prev;
        }
      }

      // Only show badge if within 0.5s of a detection
      if (Math.abs(closest.absoluteTime - videoTime) > 0.5) {
        badge.style.display = 'none';
        return;
      }

      const isNear = closest.y > courtSplitY;
      const side = isNear ? 'NEAR' : 'FAR';
      const team = isNear ? 'A' : 'B';
      const color = isNear ? TEAM_A_COLOR : TEAM_B_COLOR;

      badge.style.display = 'flex';
      badge.style.backgroundColor = color;
      badge.textContent = `${side} (${team})`;
    };

    const onFrame = (_now: DOMHighResTimeStamp, metadata: VideoFrameCallbackMetadata) => {
      render(metadata.mediaTime);
      rvfcId = video.requestVideoFrameCallback(onFrame);
    };

    render(video.currentTime);
    rvfcId = video.requestVideoFrameCallback(onFrame);

    return () => {
      if (rvfcId !== undefined) video.cancelVideoFrameCallback(rvfcId);
    };
  }, [videoRef, courtSplitY, ballWithTime]);

  const hasCorners = corners && corners.length === 4;
  const hasSplitY = courtSplitY !== undefined && courtSplitY > 0;

  if (!hasCorners && !hasSplitY) {
    return null;
  }

  // Build SVG polygon points from corners (normalized 0-1 → viewBox 0-100)
  const polygonPoints = hasCorners
    ? corners.map((c) => `${c.x * 100},${c.y * 100}`).join(' ')
    : '';

  const splitYPct = hasSplitY ? courtSplitY * 100 : 0;

  return (
    <>
      {/* Static SVG layer */}
      <svg
        viewBox="0 0 100 100"
        preserveAspectRatio="none"
        style={{
          position: 'absolute',
          inset: 0,
          width: '100%',
          height: '100%',
          pointerEvents: 'none',
          zIndex: 12,
        }}
      >
        {/* Team zone tints */}
        {hasSplitY && (
          <>
            <rect
              x="0"
              y="0"
              width="100"
              height={splitYPct}
              fill={TEAM_B_ZONE}
            />
            <rect
              x="0"
              y={splitYPct}
              width="100"
              height={100 - splitYPct}
              fill={TEAM_A_ZONE}
            />
          </>
        )}

        {/* Court polygon */}
        {hasCorners && (
          <polygon
            points={polygonPoints}
            fill="rgba(255, 255, 255, 0.06)"
            stroke="rgba(255, 255, 255, 0.7)"
            strokeWidth="0.4"
            strokeDasharray="1.5,1"
          />
        )}

        {/* Split-Y line */}
        {hasSplitY && (
          <>
            <line
              x1="0"
              y1={splitYPct}
              x2="100"
              y2={splitYPct}
              stroke={SPLIT_LINE_COLOR}
              strokeWidth="0.4"
              strokeDasharray="2,1"
            />
            {/* Team labels with background for readability */}
            <rect
              x="0.5"
              y={splitYPct - 4.5}
              width="13"
              height="3.5"
              rx="0.5"
              fill="rgba(0, 0, 0, 0.5)"
            />
            <text
              x="1.5"
              y={splitYPct - 1.8}
              fill={SPLIT_LINE_COLOR}
              fontSize="2.8"
              fontFamily="monospace"
              fontWeight="bold"
            >
              Far (B)
            </text>
            <rect
              x="0.5"
              y={splitYPct + 0.8}
              width="14"
              height="3.5"
              rx="0.5"
              fill="rgba(0, 0, 0, 0.5)"
            />
            <text
              x="1.5"
              y={splitYPct + 3.5}
              fill={SPLIT_LINE_COLOR}
              fontSize="2.8"
              fontFamily="monospace"
              fontWeight="bold"
            >
              Near (A)
            </text>
          </>
        )}
      </svg>

      {/* Dynamic ball side badge */}
      <div
        ref={badgeRef}
        style={{
          position: 'absolute',
          top: 10,
          right: 10,
          display: 'none',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '5px 12px',
          borderRadius: 6,
          color: 'white',
          fontSize: 13,
          fontWeight: 700,
          fontFamily: 'monospace',
          pointerEvents: 'none',
          zIndex: 12,
          letterSpacing: '0.5px',
          boxShadow: '0 2px 8px rgba(0,0,0,0.4)',
          border: '1px solid rgba(255,255,255,0.25)',
        }}
      />
    </>
  );
}
