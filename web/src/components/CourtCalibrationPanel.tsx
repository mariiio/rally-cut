'use client';

import { useState, useCallback, useEffect, useRef } from 'react';
import { Box, Button, Typography, Stack } from '@mui/material';
import CheckIcon from '@mui/icons-material/Check';
import CloseIcon from '@mui/icons-material/Close';
import { usePlayerTrackingStore } from '@/stores/playerTrackingStore';

interface CourtCalibrationPanelProps {
  videoId: string;
  videoWidth: number;
  videoHeight: number;
  containerRef: React.RefObject<HTMLDivElement | null>;
  /**
   * Optional seed for the net-top handle when the calibration has not
   * stored one yet. Caller passes the per-rally ball-trajectory estimate
   * (`contacts.netY`) — used only to position the handle on first open
   * so the user has a starting point near the ball-trajectory midpoint
   * rather than at a generic default. Once the user drags + saves, the
   * stored calibration value takes over.
   */
  netYSeed?: number;
}

// Corner labels for beach volleyball court
const CORNER_LABELS = ['Bottom-Left', 'Bottom-Right', 'Top-Right', 'Top-Left'];
const CORNER_COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'];

// Net top handle styling
const NET_Y_COLOR = '#FF9800';
const NET_HANDLE_RADIUS_PX = 9;

// Sentinel indices for the two independent net-top handles. Distinct from
// the corner indices (0..3); separate from each other so each handle has
// its own y-state and the line they connect can express tilt.
const NET_LEFT_HANDLE_INDEX = -1;
const NET_RIGHT_HANDLE_INDEX = -2;
// Normalized x positions of the two net-top handles. Match the convention
// from the labeling spec (analysis/docs/labeling/net_top_endpoints.md).
const NET_LEFT_X = 0.08;
const NET_RIGHT_X = 0.92;

// Default corner positions (normalized 0-1)
const DEFAULT_CORNERS = [
  { x: 0.15, y: 0.85 }, // bottom-left
  { x: 0.85, y: 0.85 }, // bottom-right
  { x: 0.75, y: 0.35 }, // top-right
  { x: 0.25, y: 0.35 }, // top-left
];

export function CourtCalibrationPanel({
  videoId,
  videoWidth,
  videoHeight,
  containerRef,
  netYSeed,
}: CourtCalibrationPanelProps) {
  const { setIsCalibrating, saveCalibration, calibrations } = usePlayerTrackingStore();

  // Load existing calibration or use defaults
  const existingCalibration = calibrations[videoId];
  const [corners, setCorners] = useState(
    existingCalibration?.corners || DEFAULT_CORNERS
  );
  // v9: two independent net-top y values, one per post. Priority on
  // initial mount:
  //   1. stored v9 endpoints (netTopLeftY, netTopRightY)
  //   2. stored legacy scalar (netTopY) — both handles seed to it,
  //      so an existing v7/v8 horizontal label converts cleanly
  //   3. caller-provided ball-trajectory seed (netYSeed)
  //   4. 0.45 sensible default
  const initialNetY = (() => {
    if (existingCalibration?.netTopY !== undefined) return existingCalibration.netTopY;
    if (netYSeed !== undefined && netYSeed > 0) return netYSeed;
    return 0.45;
  })();
  const [netTopLeftY, setNetTopLeftY] = useState<number>(
    existingCalibration?.netTopLeftY ?? initialNetY
  );
  const [netTopRightY, setNetTopRightY] = useState<number>(
    existingCalibration?.netTopRightY ?? initialNetY
  );
  // Per-endpoint visibility flags (2=visible, 1=extrapolated, 0=skip).
  const [leftVisibility, setLeftVisibility] = useState<0 | 1 | 2>(
    (existingCalibration?.netTopEndpoints?.leftVisibility as 0 | 1 | 2) ?? 2
  );
  const [rightVisibility, setRightVisibility] = useState<0 | 1 | 2>(
    (existingCalibration?.netTopEndpoints?.rightVisibility as 0 | 1 | 2) ?? 2
  );
  const [draggingIndex, setDraggingIndex] = useState<number | null>(null);
  const draggingIndexRef = useRef<number | null>(null);

  const handleCancel = useCallback(() => {
    setIsCalibrating(false);
  }, [setIsCalibrating]);

  const handleSave = useCallback(() => {
    saveCalibration(videoId, corners, {
      leftY: netTopLeftY,
      rightY: netTopRightY,
      endpoints: { leftVisibility, rightVisibility },
    });
  }, [
    videoId, corners, netTopLeftY, netTopRightY,
    leftVisibility, rightVisibility, saveCalibration,
  ]);

  // Cycle visibility 2 → 1 → 0 → 2 on right-click of a handle.
  const cycleVisibility = (v: 0 | 1 | 2): 0 | 1 | 2 =>
    v === 2 ? 1 : v === 1 ? 0 : 2;

  const handleMouseDown = useCallback((index: number) => (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDraggingIndex(index);
    draggingIndexRef.current = index;
  }, []);

  // Use document-level mouse events to prevent drag getting stuck
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      const idx = draggingIndexRef.current;
      if (idx === null || !containerRef.current) return;

      const rect = containerRef.current.getBoundingClientRect();
      const y = (e.clientY - rect.top) / rect.height;

      if (idx === NET_LEFT_HANDLE_INDEX) {
        setNetTopLeftY(Math.max(0, Math.min(1, y)));
        return;
      }
      if (idx === NET_RIGHT_HANDLE_INDEX) {
        setNetTopRightY(Math.max(0, Math.min(1, y)));
        return;
      }

      // Allow corners to go outside the video (no clamping)
      const x = (e.clientX - rect.left) / rect.width;
      setCorners(prev => {
        const updated = [...prev];
        updated[idx] = { x, y };
        return updated;
      });
    };

    const handleMouseUp = () => {
      setDraggingIndex(null);
      draggingIndexRef.current = null;
    };

    // Add listeners to document for reliable drag tracking
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [containerRef]);

  // Calculate center line with proper perspective projection
  // For a rectangle in perspective, we need to:
  // 1. Find the intersection of diagonals (perspective center)
  // 2. Find the vanishing point of the baselines
  // 3. Draw line through center toward vanishing point, intersecting sidelines

  // Helper: find intersection of two lines (p1-p2) and (p3-p4)
  const lineIntersection = (
    p1: { x: number; y: number },
    p2: { x: number; y: number },
    p3: { x: number; y: number },
    p4: { x: number; y: number }
  ) => {
    const denom = (p1.x - p2.x) * (p3.y - p4.y) - (p1.y - p2.y) * (p3.x - p4.x);
    if (Math.abs(denom) < 0.0001) {
      // Lines are parallel, return midpoint fallback
      return { x: (p1.x + p2.x) / 2, y: (p1.y + p2.y) / 2 };
    }
    const t = ((p1.x - p3.x) * (p3.y - p4.y) - (p1.y - p3.y) * (p3.x - p4.x)) / denom;
    return {
      x: p1.x + t * (p2.x - p1.x),
      y: p1.y + t * (p2.y - p1.y),
    };
  };

  // Perspective center: intersection of diagonals
  // Diagonal 1: bottom-left (0) to top-right (2)
  // Diagonal 2: bottom-right (1) to top-left (3)
  const perspectiveCenter = lineIntersection(corners[0], corners[2], corners[1], corners[3]);

  // Vanishing point: where bottom edge (0-1) and top edge (3-2) would meet
  const vanishingPoint = lineIntersection(corners[0], corners[1], corners[3], corners[2]);

  // Center line: passes through perspectiveCenter toward vanishingPoint
  // Find where this line intersects left sideline (0-3) and right sideline (1-2)
  const leftCenterPoint = lineIntersection(
    perspectiveCenter,
    vanishingPoint,
    corners[0],
    corners[3]
  );
  const rightCenterPoint = lineIntersection(
    perspectiveCenter,
    vanishingPoint,
    corners[1],
    corners[2]
  );

  return (
    <Box
      sx={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        zIndex: 10,
        cursor: draggingIndex !== null ? 'grabbing' : 'default',
        overflow: 'visible',
      }}
    >
      {/* Corner markers - transparent rings with crosshairs using HTML/CSS */}
      {corners.map((corner, index) => {
        const color = CORNER_COLORS[index];
        return (
          <Box
            key={`marker-${index}`}
            sx={{
              position: 'absolute',
              left: `${corner.x * 100}%`,
              top: `${corner.y * 100}%`,
              transform: 'translate(-50%, -50%)',
              width: 32,
              height: 32,
              borderRadius: '50%',
              border: `3px solid ${color}`,
              boxShadow: `0 0 0 1px rgba(255,255,255,0.5)`,
              pointerEvents: 'none',
              '&::before': {
                content: '""',
                position: 'absolute',
                top: '50%',
                left: -4,
                right: -4,
                height: 2,
                backgroundColor: color,
                transform: 'translateY(-50%)',
              },
              '&::after': {
                content: '""',
                position: 'absolute',
                left: '50%',
                top: -4,
                bottom: -4,
                width: 2,
                backgroundColor: color,
                transform: 'translateX(-50%)',
              },
            }}
          />
        );
      })}

      {/* Number labels for corners */}
      {corners.map((corner, index) => {
        const color = CORNER_COLORS[index];
        return (
          <Box
            key={`label-${index}`}
            sx={{
              position: 'absolute',
              left: `${corner.x * 100}%`,
              top: `${corner.y * 100}%`,
              transform: 'translate(16px, -24px)',
              color: color,
              fontSize: 14,
              fontWeight: 'bold',
              textShadow: '1px 1px 2px black, -1px -1px 2px black',
              pointerEvents: 'none',
            }}
          >
            {index + 1}
          </Box>
        );
      })}

      {/* SVG overlay for lines - use viewBox for percentage-like coordinates */}
      <svg
        viewBox="0 0 100 100"
        preserveAspectRatio="none"
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          overflow: 'visible',
          pointerEvents: 'none',
        }}
      >
        {/* Semi-transparent court fill */}
        <polygon
          points={corners.map(c => `${c.x * 100},${c.y * 100}`).join(' ')}
          fill="rgba(255,215,0,0.1)"
          stroke="none"
        />

        {/* Court boundary lines */}
        {corners.map((corner, index) => {
          const nextCorner = corners[(index + 1) % corners.length];
          return (
            <line
              key={index}
              x1={corner.x * 100}
              y1={corner.y * 100}
              x2={nextCorner.x * 100}
              y2={nextCorner.y * 100}
              stroke="white"
              strokeWidth="0.3"
              strokeDasharray="1,0.5"
              opacity="0.8"
              vectorEffect="non-scaling-stroke"
            />
          );
        })}

        {/* Center line (net projection on floor) */}
        <line
          x1={leftCenterPoint.x * 100}
          y1={leftCenterPoint.y * 100}
          x2={rightCenterPoint.x * 100}
          y2={rightCenterPoint.y * 100}
          stroke="#FFD700"
          strokeWidth="0.3"
          strokeDasharray="0.8,0.4"
          opacity="0.9"
          vectorEffect="non-scaling-stroke"
        />

        {/* User-set net top — TWO independent draggable endpoints (v9).
            Camera is fixed across a match so net top is one pair per
            video. Left handle = x=NET_LEFT_X; right handle = x=NET_RIGHT_X.
            The line between them expresses tilt (left.y ≠ right.y).
            Distinct from the gold center-line above (perspective net
            midline ON the ground); this is the visible NET TOP TAPE
            in image space. Labeling: docs/labeling/net_top_endpoints.md. */}
        <line
          x1={NET_LEFT_X * 100}
          y1={netTopLeftY * 100}
          x2={NET_RIGHT_X * 100}
          y2={netTopRightY * 100}
          stroke={NET_Y_COLOR}
          strokeWidth="1"
          opacity="0.9"
          vectorEffect="non-scaling-stroke"
        />
        <text
          x={NET_LEFT_X * 100 + 0.6}
          y={netTopLeftY * 100 - 0.6}
          fill={NET_Y_COLOR}
          fontSize="2.1"
          fontFamily="monospace"
          fontWeight="bold"
          style={{ paintOrder: 'stroke', stroke: 'rgba(0,0,0,0.65)', strokeWidth: 0.5 }}
        >
          net top L
        </text>
        <text
          x={NET_RIGHT_X * 100 - 8}
          y={netTopRightY * 100 - 0.6}
          fill={NET_Y_COLOR}
          fontSize="2.1"
          fontFamily="monospace"
          fontWeight="bold"
          style={{ paintOrder: 'stroke', stroke: 'rgba(0,0,0,0.65)', strokeWidth: 0.5 }}
        >
          net top R
        </text>
      </svg>

      {/* Invisible drag handles over corners */}
      {corners.map((corner, index) => (
        <Box
          key={index}
          onMouseDown={handleMouseDown(index)}
          sx={{
            position: 'absolute',
            left: `${corner.x * 100}%`,
            top: `${corner.y * 100}%`,
            transform: 'translate(-50%, -50%)',
            width: 40,
            height: 40,
            borderRadius: '50%',
            cursor: 'grab',
            '&:active': {
              cursor: 'grabbing',
            },
          }}
        />
      ))}

      {/* Visible net-top handles: two independent circles, one per
          post. Each drags its own y. Right-click cycles visibility
          flag 2→1→0→2 (2=clearly visible, 1=extrapolated, 0=skip). */}
      {[
        { side: 'left' as const, leftPct: NET_LEFT_X * 100, y: netTopLeftY,
          visibility: leftVisibility, handleIdx: NET_LEFT_HANDLE_INDEX,
          setVisibility: setLeftVisibility },
        { side: 'right' as const, leftPct: NET_RIGHT_X * 100, y: netTopRightY,
          visibility: rightVisibility, handleIdx: NET_RIGHT_HANDLE_INDEX,
          setVisibility: setRightVisibility },
      ].map(({ side, leftPct, y, visibility, handleIdx, setVisibility }) => (
        <Box
          key={`net-handle-${side}`}
          onMouseDown={handleMouseDown(handleIdx)}
          onContextMenu={(e) => {
            e.preventDefault();
            setVisibility(cycleVisibility(visibility));
          }}
          title={`net top ${side} — visibility=${visibility} (right-click to cycle)`}
          sx={{
            position: 'absolute',
            left: `${leftPct}%`,
            top: `${y * 100}%`,
            transform: 'translate(-50%, -50%)',
            width: NET_HANDLE_RADIUS_PX * 2,
            height: NET_HANDLE_RADIUS_PX * 2,
            borderRadius: '50%',
            border: `2px solid ${NET_Y_COLOR}`,
            backgroundColor: visibility === 2
              ? 'rgba(0,0,0,0.55)'
              : visibility === 1
                ? 'rgba(255,193,7,0.55)'    // amber for extrapolated
                : 'rgba(244,67,54,0.55)',   // red for skip
            boxShadow: '0 0 0 1px rgba(255,255,255,0.4)',
            cursor: 'ns-resize',
            '&:active': {
              cursor: 'ns-resize',
              backgroundColor: visibility === 2
                ? 'rgba(0,0,0,0.75)'
                : visibility === 1
                  ? 'rgba(255,193,7,0.75)'
                  : 'rgba(244,67,54,0.75)',
            },
          }}
        />
      ))}

      {/* Control buttons */}
      <Stack
        direction="row"
        spacing={1}
        sx={{
          position: 'absolute',
          bottom: 16,
          left: '50%',
          transform: 'translateX(-50%)',
          backgroundColor: 'rgba(0,0,0,0.8)',
          borderRadius: 2,
          p: 1,
        }}
      >
        <Button
          size="small"
          variant="outlined"
          color="error"
          startIcon={<CloseIcon />}
          onClick={handleCancel}
        >
          Cancel
        </Button>
        <Button
          size="small"
          variant="contained"
          color="success"
          startIcon={<CheckIcon />}
          onClick={handleSave}
        >
          Save Calibration
        </Button>
      </Stack>

      {/* Instructions */}
      <Box
        sx={{
          position: 'absolute',
          top: 16,
          left: '50%',
          transform: 'translateX(-50%)',
          backgroundColor: 'rgba(0,0,0,0.8)',
          borderRadius: 2,
          px: 2,
          py: 1,
        }}
      >
        <Typography variant="body2" sx={{ color: 'white', textAlign: 'center' }}>
          Drag the 4 corners to match the court boundaries.
          Drag the left & right orange handles INDEPENDENTLY to the
          center of the white tape at each post.
          Right-click a handle to mark partial-occlusion / skip
          (amber → red).
        </Typography>
        <Stack direction="row" spacing={2} sx={{ mt: 1, justifyContent: 'center' }}>
          {CORNER_LABELS.map((label, index) => (
            <Box key={index} sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
              <Box
                sx={{
                  width: 12,
                  height: 12,
                  borderRadius: '50%',
                  backgroundColor: CORNER_COLORS[index],
                }}
              />
              <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                {label}
              </Typography>
            </Box>
          ))}
        </Stack>
      </Box>
    </Box>
  );
}
