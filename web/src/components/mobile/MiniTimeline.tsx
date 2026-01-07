'use client';

import { useState, useRef, useCallback, useMemo, useEffect } from 'react';
import { Box, Typography } from '@mui/material';
import { Rally } from '@/types/rally';
import { designTokens } from '@/app/theme';
import { formatTime } from '@/utils/timeFormat';

interface MiniTimelineProps {
  rally: Rally;
  videoDuration: number;
  currentTime: number;
  onStartChange: (newStart: number) => void;
  onEndChange: (newEnd: number) => void;
  onSeek: (time: number) => void;
  adjacentRallies?: {
    prev?: { end_time: number };
    next?: { start_time: number };
  };
}

const TIMELINE_HEIGHT = 80;
const HANDLE_WIDTH = 44; // Touch-friendly handle size
const RULER_HEIGHT = 20;
const MIN_DURATION = 0.5;

export function MiniTimeline({
  rally,
  videoDuration,
  currentTime,
  onStartChange,
  onEndChange,
  onSeek,
  adjacentRallies,
}: MiniTimelineProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const startHandleRef = useRef<HTMLDivElement>(null);
  const endHandleRef = useRef<HTMLDivElement>(null);
  const [containerWidth, setContainerWidth] = useState(300);
  const [dragging, setDragging] = useState<'start' | 'end' | null>(null);
  const [localStart, setLocalStart] = useState<number | null>(null);
  const [localEnd, setLocalEnd] = useState<number | null>(null);
  const dragStartRef = useRef<{ x: number; time: number } | null>(null);
  const draggingRef = useRef<'start' | 'end' | null>(null);

  // Calculate 30-second window centered on rally
  const window = useMemo(() => {
    const windowDuration = designTokens.mobile.miniTimeline.windowDuration;
    const rallyCenter = (rally.start_time + rally.end_time) / 2;
    const halfWindow = windowDuration / 2;

    let windowStart = Math.max(0, rallyCenter - halfWindow);
    const windowEnd = Math.min(videoDuration, windowStart + windowDuration);

    // Adjust if we hit the end
    if (windowEnd === videoDuration) {
      windowStart = Math.max(0, windowEnd - windowDuration);
    }

    return { windowStart, windowEnd, duration: windowEnd - windowStart };
  }, [rally.start_time, rally.end_time, videoDuration]);

  // Update container width on resize
  useEffect(() => {
    if (!containerRef.current) return;
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setContainerWidth(entry.contentRect.width);
      }
    });
    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  // Convert time to pixel position
  const timeToPixel = useCallback(
    (time: number) => {
      const ratio = (time - window.windowStart) / window.duration;
      return ratio * containerWidth;
    },
    [window, containerWidth]
  );

  // Convert pixel position to time
  const pixelToTime = useCallback(
    (pixel: number) => {
      const ratio = pixel / containerWidth;
      return window.windowStart + ratio * window.duration;
    },
    [window, containerWidth]
  );

  // Calculate positions
  const effectiveStart = localStart ?? rally.start_time;
  const effectiveEnd = localEnd ?? rally.end_time;
  const rallyStartPixel = timeToPixel(effectiveStart);
  const rallyEndPixel = timeToPixel(effectiveEnd);
  const rallyWidth = rallyEndPixel - rallyStartPixel;
  const playheadPixel = timeToPixel(currentTime);

  // Calculate boundaries
  const minStart = adjacentRallies?.prev?.end_time ?? 0;
  const maxEnd = adjacentRallies?.next?.start_time ?? videoDuration;

  // Generate time markers (every 5 seconds)
  const timeMarkers = useMemo(() => {
    const markers: number[] = [];
    const startMarker = Math.ceil(window.windowStart / 5) * 5;
    for (let t = startMarker; t <= window.windowEnd; t += 5) {
      markers.push(t);
    }
    return markers;
  }, [window]);

  // Store refs for values needed in native event handlers
  const stateRef = useRef({
    containerWidth,
    windowDuration: window.duration,
    minStart,
    maxEnd,
    effectiveStart,
    effectiveEnd,
    localStart,
    localEnd,
    onStartChange,
    onEndChange,
    rallyStartTime: rally.start_time,
    rallyEndTime: rally.end_time,
  });

  // Update refs when values change
  useEffect(() => {
    stateRef.current = {
      containerWidth,
      windowDuration: window.duration,
      minStart,
      maxEnd,
      effectiveStart,
      effectiveEnd,
      localStart,
      localEnd,
      onStartChange,
      onEndChange,
      rallyStartTime: rally.start_time,
      rallyEndTime: rally.end_time,
    };
  }, [containerWidth, window.duration, minStart, maxEnd, effectiveStart, effectiveEnd, localStart, localEnd, onStartChange, onEndChange, rally.start_time, rally.end_time]);

  // Native touch event handlers (to use passive: false)
  // Only set up once on mount - use refs for all dynamic values
  useEffect(() => {
    const startHandle = startHandleRef.current;
    const endHandle = endHandleRef.current;
    const container = containerRef.current;
    if (!startHandle || !endHandle || !container) return;

    const handleTouchStart = (edge: 'start' | 'end') => (e: TouchEvent) => {
      e.preventDefault();
      e.stopPropagation();
      const touch = e.touches[0];
      const state = stateRef.current;
      dragStartRef.current = {
        x: touch.clientX,
        time: edge === 'start' ? state.rallyStartTime : state.rallyEndTime,
      };
      draggingRef.current = edge;
      setDragging(edge);

      // Haptic feedback
      if (navigator.vibrate) {
        navigator.vibrate(10);
      }
    };

    const handleTouchMove = (e: TouchEvent) => {
      if (!draggingRef.current || !dragStartRef.current) return;
      e.preventDefault();

      const touch = e.touches[0];
      const state = stateRef.current;
      const deltaX = touch.clientX - dragStartRef.current.x;
      const deltaTime = (deltaX / state.containerWidth) * state.windowDuration;
      let newTime = dragStartRef.current.time + deltaTime;

      if (draggingRef.current === 'start') {
        // Clamp start time
        newTime = Math.max(state.minStart, newTime);
        newTime = Math.min(state.effectiveEnd - MIN_DURATION, newTime);
        setLocalStart(newTime);
      } else {
        // Clamp end time
        newTime = Math.min(state.maxEnd, newTime);
        newTime = Math.max(state.effectiveStart + MIN_DURATION, newTime);
        setLocalEnd(newTime);
      }
    };

    const handleTouchEnd = () => {
      if (!draggingRef.current) return;

      const state = stateRef.current;

      // Commit the change
      if (draggingRef.current === 'start' && state.localStart !== null) {
        state.onStartChange(state.localStart);
      } else if (draggingRef.current === 'end' && state.localEnd !== null) {
        state.onEndChange(state.localEnd);
      }

      draggingRef.current = null;
      setDragging(null);
      setLocalStart(null);
      setLocalEnd(null);
      dragStartRef.current = null;

      // Haptic feedback
      if (navigator.vibrate) {
        navigator.vibrate(5);
      }
    };

    const startTouchStart = handleTouchStart('start');
    const endTouchStart = handleTouchStart('end');

    // Add listeners with passive: false to allow preventDefault
    startHandle.addEventListener('touchstart', startTouchStart, { passive: false });
    endHandle.addEventListener('touchstart', endTouchStart, { passive: false });
    container.addEventListener('touchmove', handleTouchMove, { passive: false });
    container.addEventListener('touchend', handleTouchEnd, { passive: false });
    container.addEventListener('touchcancel', handleTouchEnd, { passive: false });

    return () => {
      startHandle.removeEventListener('touchstart', startTouchStart);
      endHandle.removeEventListener('touchstart', endTouchStart);
      container.removeEventListener('touchmove', handleTouchMove);
      container.removeEventListener('touchend', handleTouchEnd);
      container.removeEventListener('touchcancel', handleTouchEnd);
    };
  }, []); // Empty deps - only run once, use refs for dynamic values

  // Handle tap to seek
  const handleTimelineTap = useCallback(
    (e: React.MouseEvent) => {
      if (!containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const time = pixelToTime(x);
      onSeek(Math.max(0, Math.min(videoDuration, time)));
    },
    [pixelToTime, onSeek, videoDuration]
  );

  return (
    <Box
      ref={containerRef}
      sx={{
        height: TIMELINE_HEIGHT + RULER_HEIGHT,
        bgcolor: designTokens.colors.timeline.background,
        borderRadius: 1,
        overflow: 'hidden',
        position: 'relative',
        touchAction: 'none', // Prevent scroll while dragging
      }}
    >
      {/* Time ruler */}
      <Box
        sx={{
          height: RULER_HEIGHT,
          display: 'flex',
          alignItems: 'flex-end',
          px: 1,
          bgcolor: 'rgba(0,0,0,0.3)',
        }}
      >
        {timeMarkers.map((time) => (
          <Typography
            key={time}
            variant="caption"
            sx={{
              position: 'absolute',
              left: timeToPixel(time),
              transform: 'translateX(-50%)',
              fontSize: '0.625rem',
              color: 'text.secondary',
            }}
          >
            {formatTime(time)}
          </Typography>
        ))}
      </Box>

      {/* Timeline track */}
      <Box
        sx={{
          height: TIMELINE_HEIGHT,
          position: 'relative',
          cursor: 'pointer',
        }}
        onClick={handleTimelineTap}
      >
        {/* Adjacent rally zones (greyed out) */}
        {adjacentRallies?.prev && (
          <Box
            sx={{
              position: 'absolute',
              left: 0,
              width: timeToPixel(adjacentRallies.prev.end_time),
              height: '100%',
              bgcolor: 'rgba(255,255,255,0.05)',
              borderRight: '1px solid rgba(255,255,255,0.1)',
            }}
          />
        )}
        {adjacentRallies?.next && (
          <Box
            sx={{
              position: 'absolute',
              left: timeToPixel(adjacentRallies.next.start_time),
              right: 0,
              height: '100%',
              bgcolor: 'rgba(255,255,255,0.05)',
              borderLeft: '1px solid rgba(255,255,255,0.1)',
            }}
          />
        )}

        {/* Rally block */}
        <Box
          sx={{
            position: 'absolute',
            left: rallyStartPixel,
            width: Math.max(0, rallyWidth),
            height: '100%',
            background: designTokens.colors.timeline.rallySelected,
            borderRadius: 1,
            transition: dragging ? 'none' : 'left 0.05s ease, width 0.05s ease',
          }}
        />

        {/* Start handle */}
        <Box
          ref={startHandleRef}
          sx={{
            position: 'absolute',
            left: rallyStartPixel - HANDLE_WIDTH / 2,
            width: HANDLE_WIDTH,
            height: '100%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            cursor: 'ew-resize',
            zIndex: 2,
            '&:active': {
              '& > div': {
                transform: 'scaleX(1.5)',
                bgcolor: 'primary.main',
              },
            },
          }}
        >
          <Box
            sx={{
              width: 4,
              height: '60%',
              bgcolor: dragging === 'start' ? 'primary.main' : 'rgba(255,255,255,0.8)',
              borderRadius: 2,
              transition: 'transform 0.1s ease, background-color 0.1s ease',
            }}
          />
        </Box>

        {/* End handle */}
        <Box
          ref={endHandleRef}
          sx={{
            position: 'absolute',
            left: rallyEndPixel - HANDLE_WIDTH / 2,
            width: HANDLE_WIDTH,
            height: '100%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            cursor: 'ew-resize',
            zIndex: 2,
            '&:active': {
              '& > div': {
                transform: 'scaleX(1.5)',
                bgcolor: 'primary.main',
              },
            },
          }}
        >
          <Box
            sx={{
              width: 4,
              height: '60%',
              bgcolor: dragging === 'end' ? 'primary.main' : 'rgba(255,255,255,0.8)',
              borderRadius: 2,
              transition: 'transform 0.1s ease, background-color 0.1s ease',
            }}
          />
        </Box>

        {/* Playhead */}
        {currentTime >= window.windowStart && currentTime <= window.windowEnd && (
          <Box
            sx={{
              position: 'absolute',
              left: playheadPixel,
              top: 0,
              bottom: 0,
              width: 2,
              bgcolor: designTokens.colors.timeline.cursor,
              boxShadow: designTokens.colors.timeline.cursorGlow,
              transform: 'translateX(-1px)',
              pointerEvents: 'none',
            }}
          />
        )}
      </Box>

      {/* Time labels */}
      <Box
        sx={{
          position: 'absolute',
          bottom: 4,
          left: 8,
          right: 8,
          display: 'flex',
          justifyContent: 'space-between',
          pointerEvents: 'none',
        }}
      >
        <Typography
          variant="caption"
          sx={{
            bgcolor: 'rgba(0,0,0,0.6)',
            px: 0.5,
            borderRadius: 0.5,
            fontSize: '0.6875rem',
            fontWeight: 600,
          }}
        >
          {formatTime(effectiveStart)}
        </Typography>
        <Typography
          variant="caption"
          sx={{
            bgcolor: 'rgba(0,0,0,0.6)',
            px: 0.5,
            borderRadius: 0.5,
            fontSize: '0.6875rem',
            fontWeight: 600,
          }}
        >
          {formatTime(effectiveEnd)}
        </Typography>
      </Box>
    </Box>
  );
}
