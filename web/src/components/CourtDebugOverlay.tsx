'use client';

import { useEffect, useRef, useMemo, RefObject } from 'react';
import type { ActionInfo, BallPosition, ContactInfo } from '@/services/api';
import type { Corner } from '@/stores/playerTrackingStore';

// Team colors
const TEAM_A_COLOR = 'rgba(244, 67, 54, 0.85)'; // Red (near court)
const TEAM_B_COLOR = 'rgba(33, 150, 243, 0.85)'; // Blue (far court)
const TEAM_A_ZONE = 'rgba(244, 67, 54, 0.12)';
const TEAM_B_ZONE = 'rgba(33, 150, 243, 0.12)';
const SPLIT_LINE_COLOR = '#00BCD4'; // Cyan

/** Cross product of vectors (bx-ax, by-ay) and (px-ax, py-ay). Positive = left of AB. */
function crossProduct(
  ax: number, ay: number, bx: number, by: number, px: number, py: number,
): number {
  return (bx - ax) * (py - ay) - (by - ay) * (px - ax);
}

/** Line-line intersection using homogeneous coordinates. Returns null if parallel. */
function lineIntersection(
  p1: Corner, p2: Corner, p3: Corner, p4: Corner,
): Corner | null {
  // Lines in homogeneous form: l = p1 × p2
  const l1x = p1.y - p2.y;
  const l1y = p2.x - p1.x;
  const l1w = p1.x * p2.y - p2.x * p1.y;

  const l2x = p3.y - p4.y;
  const l2y = p4.x - p3.x;
  const l2w = p3.x * p4.y - p4.x * p3.y;

  // Intersection = l1 × l2
  const ix = l1y * l2w - l1w * l2y;
  const iy = l1w * l2x - l1x * l2w;
  const iw = l1x * l2y - l1y * l2x;

  if (Math.abs(iw) < 1e-10) return null;
  return { x: ix / iw, y: iy / iw };
}

/**
 * Compute the perspective-correct net line from 4 court corners.
 *
 * The court quad is a perspective projection of a rectangle. The net bisects
 * the court at the physical midpoint (8m on a 16m court). In image space,
 * this is NOT the Euclidean midpoint of the sidelines — perspective compresses
 * the far half.
 *
 * Projective geometry: the diagonals of a quad (perspective image of a
 * rectangle) intersect at the image of the rectangle's center. The net line
 * passes through this center, parallel to the baselines in the projective
 * sense (i.e., it also passes through the baseline vanishing point).
 */
function computePerspectiveNetLine(corners: Corner[]): { left: Corner; right: Corner } | null {
  if (corners.length !== 4) return null;
  const [nearLeft, nearRight, farRight, farLeft] = corners;

  // Step 1: Diagonal intersection = projective center of the court
  // Diagonal 1: near-left ↔ far-right, Diagonal 2: near-right ↔ far-left
  const center = lineIntersection(nearLeft, farRight, nearRight, farLeft);
  if (!center) return null;

  // Step 2: Baseline vanishing point (where far and near baselines converge)
  // Far baseline: far-left → far-right, Near baseline: near-left → near-right
  const baselineVP = lineIntersection(farLeft, farRight, nearLeft, nearRight);

  // Step 3: Net line = line through center and baseline VP
  // Find where it intersects each sideline
  if (baselineVP) {
    const left = lineIntersection(center, baselineVP, farLeft, nearLeft);
    const right = lineIntersection(center, baselineVP, farRight, nearRight);
    if (left && right) return { left, right };
  }

  // Fallback: baselines nearly parallel → VP at infinity → net is horizontal
  // through center. Intersect horizontal line through center with sidelines.
  const horizFar = { x: center.x - 1, y: center.y };
  const horizNear = { x: center.x + 1, y: center.y };
  const left = lineIntersection(horizFar, horizNear, farLeft, nearLeft);
  const right = lineIntersection(horizFar, horizNear, farRight, nearRight);
  if (left && right) return { left, right };

  return null;
}

interface CourtDebugOverlayProps {
  corners?: Corner[];
  courtSplitY?: number;
  ballPositions?: BallPosition[];
  contacts?: ContactInfo[];
  actions?: ActionInfo[];
  fps: number;
  rallyStartTime: number;
  videoRef: RefObject<HTMLVideoElement | null>;
}

export function CourtDebugOverlay({
  corners,
  courtSplitY,
  ballPositions,
  contacts,
  actions,
  fps,
  rallyStartTime,
  videoRef,
}: CourtDebugOverlayProps) {
  const badgeRef = useRef<HTMLDivElement>(null);

  // Compute perspective-correct net line from calibration corners
  const netLine = useMemo(() => {
    if (!corners || corners.length !== 4) return null;
    return computePerspectiveNetLine(corners);
  }, [corners]);

  // Pre-calculate court side events with absolute times (sorted by frame).
  // Prefer actions (have propagated court_side from volleyball rules) over raw contacts.
  const courtSideEvents = useMemo(() => {
    if (actions && actions.length > 0) {
      return actions
        .map((a) => ({
          frame: a.frame,
          courtSide: a.courtSide,
          absoluteTime: rallyStartTime + a.frame / fps,
        }))
        .sort((a, b) => a.absoluteTime - b.absoluteTime);
    }
    if (!contacts || contacts.length === 0) return [];
    return contacts
      .filter((c) => c.isValidated)
      .map((c) => ({
        frame: c.frame,
        courtSide: c.courtSide,
        absoluteTime: rallyStartTime + c.frame / fps,
      }))
      .sort((a, b) => a.absoluteTime - b.absoluteTime);
  }, [actions, contacts, fps, rallyStartTime]);

  // RVFC loop for ball side badge — shows court side from last action/contact
  useEffect(() => {
    const video = videoRef.current;
    const badge = badgeRef.current;
    if (!video || !badge || courtSideEvents.length === 0) {
      if (badge) badge.style.display = 'none';
      return;
    }

    let rvfcId: number | undefined;

    const render = (videoTime: number) => {
      // Binary search for the rightmost event <= videoTime
      let lo = 0;
      let hi = courtSideEvents.length - 1;
      let lastEvent: (typeof courtSideEvents)[0] | null = null;

      while (lo <= hi) {
        const mid = (lo + hi) >>> 1;
        if (courtSideEvents[mid].absoluteTime <= videoTime) {
          lastEvent = courtSideEvents[mid];
          lo = mid + 1;
        } else {
          hi = mid - 1;
        }
      }

      // Hide before first event or more than 3s after last event
      if (!lastEvent || videoTime - lastEvent.absoluteTime > 3.0) {
        badge.style.display = 'none';
        return;
      }

      const isNear = lastEvent.courtSide === 'near';
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
  }, [videoRef, courtSideEvents]);

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

  // Perspective-correct net line endpoints scaled to viewBox (0-100)
  const perspectiveNet = hasCorners && netLine ? {
    leftX: netLine.left.x * 100,
    leftY: netLine.left.y * 100,
    rightX: netLine.right.x * 100,
    rightY: netLine.right.y * 100,
  } : null;

  // Near zone: near-left → near-right → netRight → netLeft
  const nearZonePoints = hasCorners && perspectiveNet
    ? `${corners[0].x * 100},${corners[0].y * 100} ${corners[1].x * 100},${corners[1].y * 100} ${perspectiveNet.rightX},${perspectiveNet.rightY} ${perspectiveNet.leftX},${perspectiveNet.leftY}`
    : '';

  // Far zone: netLeft → netRight → far-right → far-left
  const farZonePoints = hasCorners && perspectiveNet
    ? `${perspectiveNet.leftX},${perspectiveNet.leftY} ${perspectiveNet.rightX},${perspectiveNet.rightY} ${corners[2].x * 100},${corners[2].y * 100} ${corners[3].x * 100},${corners[3].y * 100}`
    : '';

  // Net line label Y position (midpoint of the net line)
  const netLabelY = perspectiveNet
    ? (perspectiveNet.leftY + perspectiveNet.rightY) / 2
    : splitYPct;

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
        {hasCorners && perspectiveNet ? (
          <>
            {/* Perspective-correct polygon zones following court shape */}
            <polygon points={farZonePoints} fill={TEAM_B_ZONE} />
            <polygon points={nearZonePoints} fill={TEAM_A_ZONE} />
          </>
        ) : hasSplitY ? (
          <>
            {/* Horizontal fallback when no calibration corners */}
            <rect x="0" y="0" width="100" height={splitYPct} fill={TEAM_B_ZONE} />
            <rect x="0" y={splitYPct} width="100" height={100 - splitYPct} fill={TEAM_A_ZONE} />
          </>
        ) : null}

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

        {/* Net line: perspective-aware when corners available, horizontal fallback */}
        {perspectiveNet ? (
          <>
            <line
              x1={perspectiveNet.leftX}
              y1={perspectiveNet.leftY}
              x2={perspectiveNet.rightX}
              y2={perspectiveNet.rightY}
              stroke={SPLIT_LINE_COLOR}
              strokeWidth="0.4"
              strokeDasharray="2,1"
            />
            {/* Team labels */}
            <rect
              x="0.5"
              y={netLabelY - 4.5}
              width="13"
              height="3.5"
              rx="0.5"
              fill="rgba(0, 0, 0, 0.5)"
            />
            <text
              x="1.5"
              y={netLabelY - 1.8}
              fill={SPLIT_LINE_COLOR}
              fontSize="2.8"
              fontFamily="monospace"
              fontWeight="bold"
            >
              Far (B)
            </text>
            <rect
              x="0.5"
              y={netLabelY + 0.8}
              width="14"
              height="3.5"
              rx="0.5"
              fill="rgba(0, 0, 0, 0.5)"
            />
            <text
              x="1.5"
              y={netLabelY + 3.5}
              fill={SPLIT_LINE_COLOR}
              fontSize="2.8"
              fontFamily="monospace"
              fontWeight="bold"
            >
              Near (A)
            </text>
          </>
        ) : hasSplitY ? (
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
        ) : null}
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
