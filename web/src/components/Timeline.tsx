'use client';

import { useMemo, useCallback, useState, useEffect, useRef } from 'react';
import { Box, Typography, IconButton, Stack, Tooltip, Switch, FormControlLabel } from '@mui/material';
import ZoomInIcon from '@mui/icons-material/ZoomIn';
import ZoomOutIcon from '@mui/icons-material/ZoomOut';
import FitScreenIcon from '@mui/icons-material/FitScreen';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import PauseIcon from '@mui/icons-material/Pause';
import SkipPreviousIcon from '@mui/icons-material/SkipPrevious';
import SkipNextIcon from '@mui/icons-material/SkipNext';
import FastForwardIcon from '@mui/icons-material/FastForward';
import AddIcon from '@mui/icons-material/Add';
import DeleteIcon from '@mui/icons-material/Delete';
import CheckIcon from '@mui/icons-material/Check';
import CloseIcon from '@mui/icons-material/Close';
import ChevronLeftIcon from '@mui/icons-material/ChevronLeft';
import ChevronRightIcon from '@mui/icons-material/ChevronRight';
import {
  Timeline as TimelineEditor,
  TimelineRow,
  TimelineAction,
  TimelineEffect,
  TimelineState,
} from '@xzdarcy/react-timeline-editor';
import { useEditorStore } from '@/stores/editorStore';
import { usePlayerStore } from '@/stores/playerStore';
import { formatTimeShort, formatTime } from '@/utils/timeFormat';

// Custom effect for rally segments
const effects: Record<string, TimelineEffect> = {
  rally: {
    id: 'rally',
    name: 'Rally',
  },
};

const SCALE_WIDTH = 160; // pixels per scale unit

// Zoom limits - will be calculated based on video duration
const MIN_SCALE = 5; // 5 seconds per marker (very zoomed in)

export function Timeline() {
  const { segments, updateSegment, selectSegment, selectedSegmentId, adjustSegmentStart, adjustSegmentEnd, createSegmentAtTime, removeSegment, videoMetadata } =
    useEditorStore();
  const { currentTime, duration, seek, isPlaying, pause, togglePlay, playOnlyRallies, togglePlayOnlyRallies } = usePlayerStore();
  const timelineRef = useRef<TimelineState>(null);
  const timelineContainerRef = useRef<HTMLDivElement>(null);
  const [scrollLeft, setScrollLeft] = useState(0);
  const [isDraggingCursor, setIsDraggingCursor] = useState(false);
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);

  // Clear delete confirmation when selection changes
  useEffect(() => {
    setDeleteConfirmId(null);
  }, [selectedSegmentId]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      if (target.matches('input, textarea')) return;

      switch (e.code) {
        case 'Space':
          e.preventDefault();
          togglePlay();
          break;
        case 'ArrowLeft':
          e.preventDefault();
          seek(Math.max(0, currentTime - 5));
          break;
        case 'ArrowRight':
          e.preventDefault();
          seek(Math.min(duration, currentTime + 5));
          break;
        case 'Delete':
        case 'Backspace':
          if (selectedSegmentId) {
            e.preventDefault();
            if (deleteConfirmId === selectedSegmentId) {
              // Confirm delete on second press
              removeSegment(selectedSegmentId);
              setDeleteConfirmId(null);
              selectSegment(null);
            } else {
              // Show confirmation on first press
              setDeleteConfirmId(selectedSegmentId);
            }
          }
          break;
        case 'Escape':
          if (deleteConfirmId) {
            setDeleteConfirmId(null);
          } else if (selectedSegmentId) {
            selectSegment(null);
          }
          break;
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [togglePlay, seek, currentTime, duration, selectedSegmentId, deleteConfirmId, removeSegment, selectSegment]);

  // Jump to previous/next segment
  const jumpToPrevSegment = useCallback(() => {
    const sorted = [...segments].sort((a, b) => a.start_time - b.start_time);
    const prev = sorted.reverse().find(s => s.start_time < currentTime - 0.5);
    if (prev) seek(prev.start_time);
  }, [segments, currentTime, seek]);

  const jumpToNextSegment = useCallback(() => {
    const sorted = [...segments].sort((a, b) => a.start_time - b.start_time);
    const next = sorted.find(s => s.start_time > currentTime + 0.5);
    if (next) seek(next.start_time);
  }, [segments, currentTime, seek]);

  // Skip dead time - when playOnlyRallies is enabled, jump to next segment if in dead time
  useEffect(() => {
    if (!isPlaying || !playOnlyRallies || segments.length === 0) return;

    const sorted = [...segments].sort((a, b) => a.start_time - b.start_time);

    // Check if current time is within any segment
    const inSegment = sorted.some(s => currentTime >= s.start_time && currentTime <= s.end_time);

    if (!inSegment) {
      // Find next segment to jump to
      const nextSegment = sorted.find(s => s.start_time > currentTime);
      if (nextSegment) {
        seek(nextSegment.start_time);
      }
    }
  }, [currentTime, isPlaying, playOnlyRallies, segments, seek]);

  // Stop playback at end of selected segment
  useEffect(() => {
    if (!isPlaying || !selectedSegmentId) return;

    const selectedSegment = segments.find(s => s.id === selectedSegmentId);
    if (!selectedSegment) return;

    // Pause when reaching the end of the selected segment (with small tolerance)
    if (currentTime >= selectedSegment.end_time - 0.05) {
      pause();
    }
  }, [currentTime, isPlaying, selectedSegmentId, segments, pause]);

  // Calculate min/max scale based on video duration
  const minScale = MIN_SCALE; // 5 seconds per marker (zoomed in)

  // Estimate visible markers based on wide screen
  // Container ~2500px wide, scaleWidth=160px = ~16 markers visible
  const estimatedVisibleMarkers = 16;

  const maxScale = useMemo(() => {
    if (duration > 0) {
      // Max: video fills the visible area exactly
      return Math.max(minScale * 2, Math.ceil(duration / estimatedVisibleMarkers));
    }
    return 300;
  }, [duration, minScale]);

  // Calculate optimal scale - video should fill visible area with minimal overflow
  const getAutoScale = useCallback(() => {
    if (duration > 0) {
      // Fit entire video to fill visible markers
      return Math.max(minScale, Math.ceil(duration / estimatedVisibleMarkers));
    }
    return 30;
  }, [duration, minScale]);

  const [scale, setScale] = useState(() => 30);

  // Auto-fit scale when duration changes
  useEffect(() => {
    if (duration > 0) {
      setScale(getAutoScale());
    }
  }, [duration, getAutoScale]);

  // Sync timeline cursor with video playback position
  useEffect(() => {
    if (timelineRef.current) {
      timelineRef.current.setTime(currentTime);
    }
  }, [currentTime]);

  // Track scroll position and cursor dragging for button positioning
  useEffect(() => {
    const container = timelineContainerRef.current;
    if (!container) return;

    const editArea = container.querySelector('.timeline-editor-edit-area');
    if (!editArea) return;

    const handleScroll = () => {
      setScrollLeft(editArea.scrollLeft);
    };

    // Track cursor dragging
    const cursorTop = container.querySelector('.timeline-editor-cursor-top');
    const handleMouseDown = () => setIsDraggingCursor(true);
    const handleMouseUp = () => setIsDraggingCursor(false);

    editArea.addEventListener('scroll', handleScroll);
    cursorTop?.addEventListener('mousedown', handleMouseDown);
    window.addEventListener('mouseup', handleMouseUp);

    return () => {
      editArea.removeEventListener('scroll', handleScroll);
      cursorTop?.removeEventListener('mousedown', handleMouseDown);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [segments.length]); // Re-attach when segments load

  // Calculate cursor pixel position
  const cursorPixelPosition = useMemo(() => {
    const pixelsPerSecond = SCALE_WIDTH / scale;
    const startLeft = 10; // matches startLeft prop
    return startLeft + (currentTime * pixelsPerSecond) - scrollLeft;
  }, [currentTime, scale, scrollLeft]);

  // Calculate selected segment position for delete button overlay
  const selectedSegmentPosition = useMemo(() => {
    if (!selectedSegmentId) return null;
    const segment = segments.find(s => s.id === selectedSegmentId);
    if (!segment) return null;

    const pixelsPerSecond = SCALE_WIDTH / scale;
    const startLeft = 10;
    const left = startLeft + (segment.start_time * pixelsPerSecond) - scrollLeft;
    const width = (segment.end_time - segment.start_time) * pixelsPerSecond;
    const center = left + (width / 2);

    return { left, width, center };
  }, [selectedSegmentId, segments, scale, scrollLeft]);

  // Convert Rally[] to TimelineRow[] format
  const editorData: TimelineRow[] = useMemo(() => {
    return [
      {
        id: 'segments',
        actions: segments.map((seg) => ({
          id: seg.id,
          start: seg.start_time,
          end: seg.end_time,
          effectId: 'rally',
          selected: seg.id === selectedSegmentId,
          flexible: false,
          movable: false,
        })),
      },
    ];
  }, [segments, selectedSegmentId]);

  // Check if a move/resize would cause overlap
  const checkOverlap = useCallback(
    (actionId: string, newStart: number, newEnd: number) => {
      return segments.some(
        (seg) =>
          seg.id !== actionId &&
          newStart < seg.end_time &&
          newEnd > seg.start_time
      );
    },
    [segments]
  );

  // Handle segment changes (resize/move)
  const handleChange = useCallback(
    (data: TimelineRow[]) => {
      const actions = data[0]?.actions || [];
      actions.forEach((action: TimelineAction) => {
        const segment = segments.find((s) => s.id === action.id);
        if (segment) {
          if (
            segment.start_time !== action.start ||
            segment.end_time !== action.end
          ) {
            updateSegment(action.id, {
              start_time: action.start,
              end_time: action.end,
            });
          }
        }
      });
    },
    [segments, updateSegment]
  );

  // Prevent overlapping during move
  const handleActionMoving = useCallback(
    (params: { action: TimelineAction; start: number; end: number }) => {
      return !checkOverlap(params.action.id, params.start, params.end);
    },
    [checkOverlap]
  );

  // Prevent overlapping during resize
  const handleActionResizing = useCallback(
    (params: { action: TimelineAction; start: number; end: number }) => {
      return !checkOverlap(params.action.id, params.start, params.end);
    },
    [checkOverlap]
  );

  // Persist changes after resize ends
  const handleActionResizeEnd = useCallback(
    (params: { action: TimelineAction; start: number; end: number }) => {
      updateSegment(params.action.id, {
        start_time: params.start,
        end_time: params.end,
      });
      return true;
    },
    [updateSegment]
  );

  // Persist changes after move ends
  const handleActionMoveEnd = useCallback(
    (params: { action: TimelineAction; start: number; end: number }) => {
      updateSegment(params.action.id, {
        start_time: params.start,
        end_time: params.end,
      });
      return true;
    },
    [updateSegment]
  );

  // Handle clicking on timeline to seek and deselect
  const handleClickTimeArea = useCallback(
    (time: number) => {
      seek(time);
      selectSegment(null);
      return true;
    },
    [seek, selectSegment]
  );

  // Handle clicking on an action to select it
  const handleClickAction = useCallback(
    (_e: React.MouseEvent, action: { action: TimelineAction }) => {
      selectSegment(action.action.id);
      seek(action.action.start);
    },
    [selectSegment, seek]
  );

  // Custom scale rendering
  const getScaleRender = useCallback((scaleValue: number) => {
    return <span>{formatTimeShort(scaleValue)}</span>;
  }, []);

  // Create segment at current playhead position
  const handleCreateSegment = useCallback(() => {
    createSegmentAtTime(currentTime);
  }, [createSegmentAtTime, currentTime]);

  // Check if we can create a segment at current time (for toolbar button)
  const canCreateSegmentToolbar = useMemo(() => {
    if (!videoMetadata) return false;
    const insideSegment = segments.some(
      (s) => currentTime >= s.start_time && currentTime <= s.end_time
    );
    return !insideSegment;
  }, [segments, currentTime, videoMetadata]);

  // Check if we should show the floating + button on cursor
  const showFloatingAddButton = useMemo(() => {
    if (!videoMetadata) return false;
    if (isDraggingCursor) return false;
    if (selectedSegmentId) return false; // Hide when a segment is selected
    // Check if we're inside an existing segment
    const insideSegment = segments.some(
      (s) => currentTime >= s.start_time && currentTime <= s.end_time
    );
    return !insideSegment;
  }, [segments, currentTime, videoMetadata, isDraggingCursor, selectedSegmentId]);

  // Scroll to current time position
  const scrollToCurrentTime = useCallback(() => {
    if (timelineRef.current && currentTime > 0) {
      // Calculate scroll position based on current time and scale
      const pixelsPerSecond = SCALE_WIDTH / scale;
      const scrollLeft = Math.max(0, currentTime * pixelsPerSecond - 200); // 200px offset to center
      timelineRef.current.setScrollLeft(scrollLeft);
    }
  }, [currentTime, scale]);

  // Zoom handlers with sensible limits based on video duration
  const handleZoomIn = () => {
    setScale((s) => {
      const newScale = Math.round(s / 1.5);
      return Math.max(minScale, newScale);
    });
    setTimeout(scrollToCurrentTime, 50);
  };

  const handleZoomOut = () => {
    setScale((s) => {
      const newScale = Math.round(s * 1.5);
      return Math.min(maxScale, newScale);
    });
    setTimeout(scrollToCurrentTime, 50);
  };

  const handleFitToScreen = () => {
    setScale(getAutoScale());
    if (timelineRef.current) {
      timelineRef.current.setScrollLeft(0);
    }
  };

  // Check if zoom limits reached
  const canZoomIn = scale > minScale;
  const canZoomOut = scale < maxScale;

  if (segments.length === 0) {
    return (
      <Box
        sx={{
          height: 150,
          bgcolor: 'background.paper',
          borderRadius: 1,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: 'text.secondary',
        }}
      >
        <Typography variant="body2">
          Load a JSON file to see the timeline
        </Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ bgcolor: 'background.paper', borderRadius: 1, overflow: 'hidden' }}>
      {/* Toolbar with playback controls */}
      <Stack
        direction="row"
        alignItems="center"
        justifyContent="space-between"
        sx={{ px: 1, py: 0.5, borderBottom: 1, borderColor: 'divider' }}
      >
        {/* Playback controls */}
        <Stack direction="row" alignItems="center" spacing={0.5}>
          <IconButton size="small" onClick={jumpToPrevSegment} title="Previous segment">
            <SkipPreviousIcon fontSize="small" />
          </IconButton>
          <IconButton size="small" onClick={togglePlay} title="Play/Pause (Space)">
            {isPlaying ? <PauseIcon /> : <PlayArrowIcon />}
          </IconButton>
          <IconButton size="small" onClick={jumpToNextSegment} title="Next segment">
            <SkipNextIcon fontSize="small" />
          </IconButton>
          <Typography variant="body2" sx={{ ml: 1, fontFamily: 'monospace', minWidth: 130 }}>
            {formatTime(currentTime)} / {formatTime(duration)}
          </Typography>
          <Tooltip title={canCreateSegmentToolbar ? "Create segment at playhead" : "Cannot create segment here (inside existing segment)"} arrow>
            <span>
              <IconButton
                size="small"
                onClick={handleCreateSegment}
                disabled={!canCreateSegmentToolbar}
                color="primary"
                sx={{ ml: 1 }}
              >
                <AddIcon fontSize="small" />
              </IconButton>
            </span>
          </Tooltip>
          <Tooltip title="Skip dead time between rallies" arrow>
            <Stack direction="row" alignItems="center" sx={{ ml: 1 }}>
              <IconButton
                size="small"
                onClick={togglePlayOnlyRallies}
                color={playOnlyRallies ? 'primary' : 'default'}
                title="Play rallies only"
              >
                <FastForwardIcon fontSize="small" />
              </IconButton>
              <Typography
                variant="caption"
                sx={{
                  color: playOnlyRallies ? 'primary.main' : 'text.secondary',
                  cursor: 'pointer',
                  ml: 0.5,
                }}
                onClick={togglePlayOnlyRallies}
              >
                Rallies only
              </Typography>
            </Stack>
          </Tooltip>
        </Stack>

        {/* Info and zoom controls */}
        <Stack direction="row" alignItems="center" spacing={1}>
          <Typography variant="caption" sx={{ color: 'text.secondary' }}>
            {segments.length} segments
          </Typography>
          <Stack direction="row" spacing={0.5}>
            <IconButton
              size="small"
              onClick={handleZoomOut}
              disabled={!canZoomOut}
              title="Zoom out (show more time)"
            >
              <ZoomOutIcon fontSize="small" />
            </IconButton>
            <IconButton
              size="small"
              onClick={handleZoomIn}
              disabled={!canZoomIn}
              title="Zoom in (show less time)"
            >
              <ZoomInIcon fontSize="small" />
            </IconButton>
            <IconButton size="small" onClick={handleFitToScreen} title="Fit to screen">
              <FitScreenIcon fontSize="small" />
            </IconButton>
          </Stack>
        </Stack>
      </Stack>

      {/* Timeline editor */}
      <Box
        ref={timelineContainerRef}
        onClick={(e) => {
          // Deselect if clicking on empty area (not on a segment)
          const target = e.target as HTMLElement;
          if (!target.closest('.timeline-editor-action')) {
            selectSegment(null);
          }
        }}
        sx={{
          height: 180,
          width: '100%',
          overflow: 'visible',
          position: 'relative',
          pt: 4, // Add padding top for delete button
          '& .timeline-editor': {
            background: 'transparent !important',
            width: '100% !important',
          },
          '& .timeline-editor-edit-area': {
            width: '100% !important',
          },
          '& .timeline-editor-time-area': {
            background: '#1a1a1a !important',
          },
          '& .timeline-editor-action': {
            background: '#1976d2 !important',
            borderRadius: '4px !important',
            cursor: 'pointer',
            overflow: 'visible !important',
            zIndex: '10 !important',
            transition: 'all 0.2s ease !important',
            // Dim non-selected when there's a selection
            ...(selectedSegmentId && {
              opacity: '0.4 !important',
              filter: 'saturate(0.5) !important',
            }),
            '&:hover': {
              background: '#1565c0 !important',
              opacity: '0.7 !important',
            },
          },
          // Allow resize cursor on edges
          '& .timeline-editor-action-left-stretch, & .timeline-editor-action-right-stretch': {
            cursor: 'ew-resize !important',
          },
          '& .timeline-editor-action-selected': {
            background: '#2196f3 !important',
            boxShadow: '0 0 0 2px rgba(255,255,255,0.8), 0 4px 12px rgba(33,150,243,0.4) !important',
            zIndex: '20 !important',
            opacity: '1 !important',
            filter: 'saturate(1) !important',
            transform: 'scale(1.02)',
            '&:hover': {
              background: '#42a5f5 !important',
              opacity: '1 !important',
              boxShadow: '0 0 0 2px rgba(255,255,255,0.9), 0 6px 16px rgba(33,150,243,0.5) !important',
            },
          },
          // Hide the default cursor completely
          '& .timeline-editor-cursor': {
            display: 'none !important',
          },
          '& .timeline-editor-edit-row': {
            background: 'transparent !important',
            overflow: 'visible !important',
          },
          '& .timeline-editor-action-row': {
            overflow: 'visible !important',
          },
        }}
      >
        <TimelineEditor
          ref={timelineRef}
          editorData={editorData}
          effects={effects}
          onChange={handleChange}
          onClickTimeArea={handleClickTimeArea}
          onClickAction={handleClickAction}
          onActionMoving={handleActionMoving}
          onActionMoveEnd={handleActionMoveEnd}
          onActionResizing={handleActionResizing}
          onActionResizeEnd={handleActionResizeEnd}
          scale={scale}
          scaleWidth={SCALE_WIDTH}
          startLeft={10}
          rowHeight={100}
          gridSnap={true}
          dragLine={true}
          autoScroll={true}
          autoReRender={true}
          getScaleRender={getScaleRender}
          getActionRender={(action) => {
            const segment = segments.find((s) => s.id === action.id);
            const isSelected = selectedSegmentId === action.id;

            const edgeButtonStyle = {
              p: 0,
              minWidth: 20,
              minHeight: 20,
              width: 20,
              height: 20,
              color: 'white',
              bgcolor: 'primary.dark',
              border: '1px solid rgba(255,255,255,0.3)',
              borderRadius: '4px',
              boxShadow: '0 1px 3px rgba(0,0,0,0.3)',
              '&:hover': {
                bgcolor: 'primary.main',
                border: '1px solid rgba(255,255,255,0.5)',
              },
            };

            const handleAdjustStart = (delta: number) => {
              if (adjustSegmentStart(action.id, delta) && segment) {
                seek(segment.start_time + delta);
              }
            };

            const handleAdjustEnd = (delta: number) => {
              if (adjustSegmentEnd(action.id, delta) && segment) {
                seek(segment.end_time + delta);
              }
            };

            return (
              <Box
                sx={{
                  width: '100%',
                  height: '100%',
                  position: 'relative',
                  overflow: 'visible',
                }}
              >
                {/* Center content */}
                <Box
                  sx={{
                    width: '100%',
                    height: '100%',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: 11,
                    fontWeight: 500,
                    color: 'white',
                    userSelect: 'none',
                  }}
                >
                  <Typography
                    variant="caption"
                    sx={{
                      whiteSpace: 'nowrap',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      fontWeight: 500,
                      px: 1,
                    }}
                  >
                    {segment?.id}
                  </Typography>
                </Box>

                {/* Left edge controls - only show when selected */}
                {isSelected && (
                  <Stack
                    direction="column"
                    spacing={0.5}
                    sx={{
                      position: 'absolute',
                      left: -24,
                      top: '50%',
                      transform: 'translateY(-50%)',
                      zIndex: 100,
                    }}
                  >
                    <IconButton
                      size="small"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleAdjustStart(-0.5);
                      }}
                      sx={edgeButtonStyle}
                    >
                      <ChevronLeftIcon sx={{ fontSize: 14 }} />
                    </IconButton>
                    <IconButton
                      size="small"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleAdjustStart(0.5);
                      }}
                      sx={edgeButtonStyle}
                    >
                      <ChevronRightIcon sx={{ fontSize: 14 }} />
                    </IconButton>
                  </Stack>
                )}

                {/* Right edge controls - only show when selected */}
                {isSelected && (
                  <Stack
                    direction="column"
                    spacing={0.5}
                    sx={{
                      position: 'absolute',
                      right: -24,
                      top: '50%',
                      transform: 'translateY(-50%)',
                      zIndex: 100,
                    }}
                  >
                    <IconButton
                      size="small"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleAdjustEnd(0.5);
                      }}
                      sx={edgeButtonStyle}
                    >
                      <ChevronRightIcon sx={{ fontSize: 14 }} />
                    </IconButton>
                    <IconButton
                      size="small"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleAdjustEnd(-0.5);
                      }}
                      sx={edgeButtonStyle}
                    >
                      <ChevronLeftIcon sx={{ fontSize: 14 }} />
                    </IconButton>
                  </Stack>
                )}

              </Box>
            );
          }}
        />

        {/* Custom cursor - line, head, and optional add button */}
        {cursorPixelPosition > 0 && cursorPixelPosition < (timelineContainerRef.current?.clientWidth || 9999) && (
          <Box
            sx={{
              position: 'absolute',
              left: cursorPixelPosition,
              top: 0,
              bottom: 0,
              pointerEvents: 'none',
              zIndex: 30,
            }}
          >
            {/* Cursor line */}
            <Box
              sx={{
                position: 'absolute',
                left: -1,
                top: 28,
                bottom: 0,
                width: 2,
                bgcolor: '#ef5350',
              }}
            />
            {/* Cursor head */}
            <Box
              sx={{
                position: 'absolute',
                left: '50%',
                top: 4,
                transform: 'translateX(-50%)',
                width: 14,
                height: 22,
                bgcolor: '#ef5350',
                borderRadius: '3px 3px 0 0',
                boxShadow: '0 2px 4px rgba(0,0,0,0.3)',
                cursor: 'ew-resize',
                pointerEvents: 'auto',
                '&:hover': {
                  bgcolor: '#f44336',
                },
                '&::after': {
                  content: '""',
                  position: 'absolute',
                  bottom: -6,
                  left: '50%',
                  transform: 'translateX(-50%)',
                  width: 0,
                  height: 0,
                  borderLeft: '5px solid transparent',
                  borderRight: '5px solid transparent',
                  borderTop: '6px solid #ef5350',
                },
              }}
              onMouseDown={(e) => {
                e.preventDefault();
                setIsDraggingCursor(true);
                const startX = e.clientX;
                const startTime = currentTime;
                const pixelsPerSecond = SCALE_WIDTH / scale;

                const handleMouseMove = (moveEvent: MouseEvent) => {
                  const deltaX = moveEvent.clientX - startX;
                  const deltaTime = deltaX / pixelsPerSecond;
                  const newTime = Math.max(0, Math.min(duration, startTime + deltaTime));
                  seek(newTime);
                };

                const handleMouseUp = () => {
                  setIsDraggingCursor(false);
                  window.removeEventListener('mousemove', handleMouseMove);
                  window.removeEventListener('mouseup', handleMouseUp);
                };

                window.addEventListener('mousemove', handleMouseMove);
                window.addEventListener('mouseup', handleMouseUp);
              }}
            />
            {/* Add button - only when in non-segment area */}
            {showFloatingAddButton && (
              <IconButton
                onClick={(e) => {
                  e.stopPropagation();
                  handleCreateSegment();
                }}
                size="small"
                sx={{
                  position: 'absolute',
                  left: '50%',
                  top: 'calc(50% + 20px)',
                  transform: 'translate(-50%, -50%)',
                  pointerEvents: 'auto',
                  width: 28,
                  height: 28,
                  bgcolor: '#ef5350',
                  color: 'white',
                  border: '2px solid white',
                  boxShadow: '0 2px 6px rgba(0,0,0,0.4)',
                  '&:hover': {
                    bgcolor: '#f44336',
                    transform: 'translate(-50%, -50%) scale(1.1)',
                  },
                  transition: 'transform 0.15s, background-color 0.15s',
                }}
              >
                <AddIcon sx={{ fontSize: 16 }} />
              </IconButton>
            )}
          </Box>
        )}

        {/* Delete button overlay for selected segment */}
        {selectedSegmentId && selectedSegmentPosition && selectedSegmentPosition.center > 0 && (
          <Box
            sx={{
              position: 'absolute',
              left: selectedSegmentPosition.center,
              top: 36,
              transform: 'translateX(-50%)',
              zIndex: 100,
              pointerEvents: 'auto',
            }}
          >
            {deleteConfirmId === selectedSegmentId ? (
              // Confirmation UI
              <Stack
                direction="row"
                spacing={0.5}
                alignItems="center"
                sx={{
                  bgcolor: 'rgba(20,20,20,0.95)',
                  borderRadius: '8px',
                  py: 0.5,
                  px: 1,
                  boxShadow: '0 4px 12px rgba(0,0,0,0.5)',
                  border: '1px solid rgba(255,255,255,0.15)',
                }}
              >
                <Typography
                  variant="caption"
                  sx={{
                    color: 'white',
                    fontSize: 12,
                    fontWeight: 500,
                    mr: 0.5,
                  }}
                >
                  Delete?
                </Typography>
                <IconButton
                  size="small"
                  onClick={(e) => {
                    e.stopPropagation();
                    removeSegment(selectedSegmentId);
                    setDeleteConfirmId(null);
                    selectSegment(null);
                  }}
                  sx={{
                    width: 26,
                    height: 26,
                    bgcolor: '#d32f2f',
                    color: 'white',
                    '&:hover': { bgcolor: '#f44336' },
                  }}
                >
                  <CheckIcon sx={{ fontSize: 16 }} />
                </IconButton>
                <IconButton
                  size="small"
                  onClick={(e) => {
                    e.stopPropagation();
                    setDeleteConfirmId(null);
                  }}
                  sx={{
                    width: 26,
                    height: 26,
                    bgcolor: 'rgba(255,255,255,0.1)',
                    color: 'white',
                    '&:hover': { bgcolor: 'rgba(255,255,255,0.2)' },
                  }}
                >
                  <CloseIcon sx={{ fontSize: 16 }} />
                </IconButton>
              </Stack>
            ) : (
              // Delete button
              <IconButton
                size="small"
                onClick={(e) => {
                  e.stopPropagation();
                  setDeleteConfirmId(selectedSegmentId);
                }}
                sx={{
                  width: 28,
                  height: 28,
                  bgcolor: 'rgba(20,20,20,0.9)',
                  color: 'rgba(255,255,255,0.8)',
                  border: '1px solid rgba(255,255,255,0.15)',
                  boxShadow: '0 2px 8px rgba(0,0,0,0.4)',
                  '&:hover': {
                    bgcolor: '#d32f2f',
                    color: 'white',
                    border: '1px solid transparent',
                  },
                  transition: 'all 0.15s ease',
                }}
              >
                <DeleteIcon sx={{ fontSize: 16 }} />
              </IconButton>
            )}
          </Box>
        )}
      </Box>
    </Box>
  );
}
