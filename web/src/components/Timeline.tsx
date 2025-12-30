'use client';

import { useMemo, useCallback, useState, useEffect, useRef } from 'react';
import { Box, Typography, IconButton, Stack, Tooltip, Popover } from '@mui/material';
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
import KeyboardIcon from '@mui/icons-material/Keyboard';
import StarIcon from '@mui/icons-material/Star';
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

// Hotkey row component for the legend
function HotkeyRow({ keys, description, secondary }: { keys: string[]; description: string; secondary?: boolean }) {
  return (
    <Stack direction="row" alignItems="center" justifyContent="space-between" spacing={2}>
      <Stack direction="row" spacing={0.5}>
        {keys.map((key, i) => (
          <Box
            key={i}
            sx={{
              px: 0.75,
              py: 0.25,
              bgcolor: 'action.selected',
              borderRadius: 0.5,
              fontSize: 11,
              fontFamily: 'monospace',
              fontWeight: 500,
              minWidth: 24,
              textAlign: 'center',
            }}
          >
            {key}
          </Box>
        ))}
      </Stack>
      <Typography
        variant="caption"
        sx={{
          color: secondary ? 'text.disabled' : 'text.secondary',
          fontSize: 12,
          fontStyle: secondary ? 'italic' : 'normal',
        }}
      >
        {description}
      </Typography>
    </Stack>
  );
}

export function Timeline() {
  const {
    segments,
    updateSegment,
    selectSegment,
    selectedSegmentId,
    adjustSegmentStart,
    adjustSegmentEnd,
    createSegmentAtTime,
    removeSegment,
    videoMetadata,
    highlights,
    selectedHighlightId,
    createHighlight,
    addSegmentToHighlight,
    removeSegmentFromHighlight,
    selectHighlight,
    getHighlightsForSegment,
  } = useEditorStore();
  const {
    currentTime,
    duration,
    seek,
    isPlaying,
    pause,
    play,
    playOnlyRallies,
    togglePlayOnlyRallies,
    playingHighlightId,
    highlightSegmentIndex,
    advanceHighlightPlayback,
    stopHighlightPlayback,
  } = usePlayerStore();
  const timelineRef = useRef<TimelineState>(null);
  const timelineContainerRef = useRef<HTMLDivElement>(null);
  const [isDraggingCursor, setIsDraggingCursor] = useState(false);
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);
  const [hotkeysAnchorEl, setHotkeysAnchorEl] = useState<HTMLButtonElement | null>(null);

  // Track if we auto-paused at segment end (for restart behavior)
  const autoPausedAtEndRef = useRef<string | null>(null); // stores segment ID if auto-paused

  // Clear delete confirmation when selection changes
  useEffect(() => {
    setDeleteConfirmId(null);
  }, [selectedSegmentId]);

  // Helper to check if current time is inside the selected segment
  const isInsideSelectedSegment = useCallback(() => {
    if (!selectedSegmentId) return false;
    const segment = segments.find(s => s.id === selectedSegmentId);
    if (!segment) return false;
    return currentTime >= segment.start_time && currentTime <= segment.end_time;
  }, [selectedSegmentId, segments, currentTime]);

  // Get selected segment
  const getSelectedSegment = useCallback(() => {
    if (!selectedSegmentId) return null;
    return segments.find(s => s.id === selectedSegmentId) || null;
  }, [selectedSegmentId, segments]);

  // Navigate to previous segment and select it
  const goToPrevSegment = useCallback(() => {
    const sorted = [...segments].sort((a, b) => a.start_time - b.start_time);
    const prev = sorted.reverse().find(s => s.start_time < currentTime - 0.5);
    if (prev) {
      selectSegment(prev.id);
      seek(prev.start_time);
    }
  }, [segments, currentTime, seek, selectSegment]);

  // Navigate to next segment and select it
  const goToNextSegment = useCallback(() => {
    const sorted = [...segments].sort((a, b) => a.start_time - b.start_time);
    const next = sorted.find(s => s.start_time > currentTime + 0.5);
    if (next) {
      selectSegment(next.id);
      seek(next.start_time);
    }
  }, [segments, currentTime, seek, selectSegment]);

  // Check if segment is in selected highlight
  const isSegmentInSelectedHighlight = useMemo(() => {
    if (!selectedSegmentId || !selectedHighlightId) return false;
    const highlight = highlights.find(h => h.id === selectedHighlightId);
    return highlight?.segmentIds.includes(selectedSegmentId) ?? false;
  }, [selectedSegmentId, selectedHighlightId, highlights]);

  // Toggle segment in highlight (add if not present, remove if present)
  const handleToggleHighlight = useCallback(() => {
    if (!selectedSegmentId) return;

    if (selectedHighlightId) {
      // Check if segment is already in the selected highlight
      const highlight = highlights.find(h => h.id === selectedHighlightId);
      if (highlight?.segmentIds.includes(selectedSegmentId)) {
        // Remove from highlight
        removeSegmentFromHighlight(selectedSegmentId, selectedHighlightId);
      } else {
        // Add to highlight
        addSegmentToHighlight(selectedSegmentId, selectedHighlightId);
      }
    } else {
      // Create new highlight and add segment
      const newId = createHighlight();
      addSegmentToHighlight(selectedSegmentId, newId);
      selectHighlight(newId);
    }
  }, [selectedSegmentId, selectedHighlightId, highlights, addSegmentToHighlight, removeSegmentFromHighlight, createHighlight, selectHighlight]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      if (target.matches('input, textarea')) return;

      const isMod = e.metaKey || e.ctrlKey;

      switch (e.code) {
        case 'Space':
          e.preventDefault();

          if (isPlaying) {
            // Simply pause
            pause();
          } else {
            // Playing from paused state
            const selectedSegment = getSelectedSegment();

            if (autoPausedAtEndRef.current && selectedSegment && autoPausedAtEndRef.current === selectedSegment.id) {
              // Auto-paused at segment end: restart from segment start
              seek(selectedSegment.start_time);
              autoPausedAtEndRef.current = null;
              play();
            } else if (selectedSegmentId) {
              // Has a selected segment
              if (isInsideSelectedSegment()) {
                // Inside segment: play from current position
                play();
              } else {
                // Outside segment: deselect and play
                selectSegment(null);
                play();
              }
            } else {
              // No segment selected: just play
              play();
            }
          }
          break;

        case 'ArrowLeft':
          e.preventDefault();
          if (isMod) {
            // Cmd/Ctrl + Left: go to previous segment
            goToPrevSegment();
          } else if (selectedSegmentId) {
            // Left with segment selected: adjust segment
            const segLeft = getSelectedSegment();
            if (e.shiftKey) {
              // Shift + Left: shrink end (-0.5s)
              if (adjustSegmentEnd(selectedSegmentId, -0.5) && segLeft) {
                seek(segLeft.end_time - 0.5);
              }
            } else {
              // Left: expand start (-0.5s)
              if (adjustSegmentStart(selectedSegmentId, -0.5) && segLeft) {
                seek(segLeft.start_time - 0.5);
              }
            }
          } else {
            // Arrow Left: seek back 5 seconds
            seek(Math.max(0, currentTime - 5));
            // Clear auto-pause flag when manually seeking
            autoPausedAtEndRef.current = null;
          }
          break;

        case 'ArrowRight':
          e.preventDefault();
          if (isMod) {
            // Cmd/Ctrl + Right: go to next segment
            goToNextSegment();
          } else if (selectedSegmentId) {
            // Right with segment selected: adjust segment
            const segRight = getSelectedSegment();
            if (e.shiftKey) {
              // Shift + Right: expand end (+0.5s)
              if (adjustSegmentEnd(selectedSegmentId, 0.5) && segRight) {
                seek(segRight.end_time + 0.5);
              }
            } else {
              // Right: shrink start (+0.5s)
              if (adjustSegmentStart(selectedSegmentId, 0.5) && segRight) {
                seek(segRight.start_time + 0.5);
              }
            }
          } else {
            // Arrow Right: seek forward 5 seconds
            seek(Math.min(duration, currentTime + 5));
            // Clear auto-pause flag when manually seeking
            autoPausedAtEndRef.current = null;
          }
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

        case 'Enter':
          if (isMod) {
            // Cmd/Ctrl + Enter: Create new segment at cursor (if not inside existing segment)
            if (videoMetadata) {
              const insideSegment = segments.some(
                (s) => currentTime >= s.start_time && currentTime <= s.end_time
              );
              if (!insideSegment) {
                e.preventDefault();
                createSegmentAtTime(currentTime);
              }
            }
          } else if (selectedSegmentId) {
            // Enter (without modifier): Toggle segment in highlight
            e.preventDefault();
            handleToggleHighlight();
          }
          break;
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isPlaying, play, pause, seek, currentTime, duration, selectedSegmentId, deleteConfirmId, removeSegment, selectSegment, isInsideSelectedSegment, getSelectedSegment, goToPrevSegment, goToNextSegment, adjustSegmentStart, adjustSegmentEnd, videoMetadata, segments, createSegmentAtTime, handleToggleHighlight]);

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
      autoPausedAtEndRef.current = selectedSegmentId; // Mark that we auto-paused
      pause();
    }
  }, [currentTime, isPlaying, selectedSegmentId, segments, pause]);

  // Highlight playback - auto-advance through segments
  useEffect(() => {
    if (!isPlaying || !playingHighlightId) return;

    const highlight = highlights.find(h => h.id === playingHighlightId);
    if (!highlight || highlight.segmentIds.length === 0) {
      stopHighlightPlayback();
      return;
    }

    // Get sorted segments for this highlight
    const highlightSegments = segments
      .filter(s => highlight.segmentIds.includes(s.id))
      .sort((a, b) => a.start_time - b.start_time);

    if (highlightSegmentIndex >= highlightSegments.length) {
      // End of highlight - stop playback
      stopHighlightPlayback();
      return;
    }

    const currentSegment = highlightSegments[highlightSegmentIndex];
    if (!currentSegment) {
      stopHighlightPlayback();
      return;
    }

    // Check if we need to jump to segment start (if we're before it or in dead time)
    if (currentTime < currentSegment.start_time - 0.1) {
      seek(currentSegment.start_time);
      return;
    }

    // Check if we reached the end of current segment
    if (currentTime >= currentSegment.end_time - 0.05) {
      const nextIndex = highlightSegmentIndex + 1;
      if (nextIndex < highlightSegments.length) {
        // Advance to next segment
        advanceHighlightPlayback();
        seek(highlightSegments[nextIndex].start_time);
      } else {
        // End of highlight
        stopHighlightPlayback();
      }
    }
  }, [currentTime, isPlaying, playingHighlightId, highlightSegmentIndex, highlights, segments, seek, advanceHighlightPlayback, stopHighlightPlayback]);

  // Calculate optimal scale based on video duration
  // Estimate visible markers: Container ~2500px wide, scaleWidth=160px = ~16 markers visible
  const estimatedVisibleMarkers = 16;

  const getAutoScale = useCallback(() => {
    if (duration > 0) {
      // Fit entire video to fill visible markers
      return Math.max(MIN_SCALE, Math.ceil(duration / estimatedVisibleMarkers));
    }
    return 30;
  }, [duration]);

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

  // Track cursor dragging for button visibility
  useEffect(() => {
    const container = timelineContainerRef.current;
    if (!container) return;

    const cursorTop = container.querySelector('.timeline-editor-cursor-top');
    const handleMouseDown = () => setIsDraggingCursor(true);
    const handleMouseUp = () => setIsDraggingCursor(false);

    cursorTop?.addEventListener('mousedown', handleMouseDown);
    window.addEventListener('mouseup', handleMouseUp);

    return () => {
      cursorTop?.removeEventListener('mousedown', handleMouseDown);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [segments.length]); // Re-attach when segments load

  // Calculate cursor pixel position (no scrollLeft since scrolling is disabled)
  const cursorPixelPosition = useMemo(() => {
    const pixelsPerSecond = SCALE_WIDTH / scale;
    const startLeft = 10; // matches startLeft prop
    return startLeft + (currentTime * pixelsPerSecond);
  }, [currentTime, scale]);

  // Calculate selected segment position for delete button overlay (no scrollLeft since scrolling is disabled)
  const selectedSegmentPosition = useMemo(() => {
    if (!selectedSegmentId) return null;
    const segment = segments.find(s => s.id === selectedSegmentId);
    if (!segment) return null;

    const pixelsPerSecond = SCALE_WIDTH / scale;
    const startLeft = 10;
    const left = startLeft + (segment.start_time * pixelsPerSecond);
    const width = (segment.end_time - segment.start_time) * pixelsPerSecond;
    const center = left + (width / 2);

    return { left, width, center };
  }, [selectedSegmentId, segments, scale]);

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
        {/* Playback controls - Left */}
        <Stack direction="row" alignItems="center" spacing={0.5} sx={{ flex: 1 }}>
          <IconButton size="small" onClick={jumpToPrevSegment} title="Previous segment">
            <SkipPreviousIcon fontSize="small" />
          </IconButton>
          <IconButton size="small" onClick={() => isPlaying ? pause() : play()} title="Play/Pause (Space)">
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
        </Stack>

        {/* Rallies only toggle - Center */}
        <Tooltip title="Skip dead time between rallies" arrow>
          <Box
            onClick={togglePlayOnlyRallies}
            sx={{
              display: 'flex',
              alignItems: 'center',
              gap: 0.75,
              px: 1.5,
              py: 0.5,
              borderRadius: 2,
              cursor: 'pointer',
              bgcolor: playOnlyRallies ? 'primary.main' : 'action.hover',
              color: playOnlyRallies ? 'primary.contrastText' : 'text.secondary',
              transition: 'all 0.2s ease',
              '&:hover': {
                bgcolor: playOnlyRallies ? 'primary.dark' : 'action.selected',
              },
            }}
          >
            <FastForwardIcon sx={{ fontSize: 16 }} />
            <Typography
              variant="caption"
              sx={{
                fontWeight: 500,
                fontSize: 12,
                userSelect: 'none',
              }}
            >
              Rallies only
            </Typography>
          </Box>
        </Tooltip>

        {/* Info and controls - Right */}
        <Stack direction="row" alignItems="center" spacing={1} sx={{ flex: 1, justifyContent: 'flex-end' }}>
          <Typography variant="caption" sx={{ color: 'text.secondary' }}>
            {segments.length} segments
          </Typography>
          <Tooltip title="Keyboard shortcuts">
            <IconButton
              size="small"
              onClick={(e) => setHotkeysAnchorEl(e.currentTarget)}
              sx={{ color: 'text.secondary' }}
            >
              <KeyboardIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </Stack>
      </Stack>

      {/* Hotkeys Legend Popover */}
      <Popover
        open={Boolean(hotkeysAnchorEl)}
        anchorEl={hotkeysAnchorEl}
        onClose={() => setHotkeysAnchorEl(null)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
        transformOrigin={{ vertical: 'top', horizontal: 'right' }}
      >
        <Box sx={{ p: 2, minWidth: 280 }}>
          <Typography variant="subtitle2" sx={{ mb: 1.5, fontWeight: 600 }}>
            Keyboard Shortcuts
          </Typography>
          <Stack spacing={1}>
            <HotkeyRow keys={['Space']} description="Play / Pause" />
            <HotkeyRow keys={['←', '→']} description="Seek ±5s / Adjust start (selected)" />
            <HotkeyRow keys={['⇧', '←', '→']} description="Adjust end ±0.5s (selected)" />
            <HotkeyRow keys={['⌘/Ctrl', '←']} description="Previous segment" />
            <HotkeyRow keys={['⌘/Ctrl', '→']} description="Next segment" />
            <HotkeyRow keys={['↵']} description="Toggle highlight (selected)" />
            <HotkeyRow keys={['⌘/Ctrl', '↵']} description="New segment at cursor" />
            <HotkeyRow keys={['Delete']} description="Delete segment (press twice)" />
            <HotkeyRow keys={['⌘/Ctrl', 'Z']} description="Undo" />
            <HotkeyRow keys={['⌘/Ctrl', '⇧', 'Z']} description="Redo" />
            <HotkeyRow keys={['Esc']} description="Deselect segment" />
          </Stack>
        </Box>
      </Popover>

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
            overflowX: 'hidden !important',
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
          autoScroll={false}
          autoReRender={true}
          maxScaleCount={duration > 0 ? Math.ceil(duration / scale) + 1 : 100}
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

                {/* Highlight color dots */}
                {(() => {
                  const segmentHighlights = getHighlightsForSegment(action.id);
                  if (segmentHighlights.length === 0) return null;
                  return (
                    <Stack
                      direction="row"
                      spacing={0.25}
                      sx={{
                        position: 'absolute',
                        bottom: 4,
                        left: 4,
                        pointerEvents: 'none',
                      }}
                    >
                      {segmentHighlights.slice(0, 4).map((h) => (
                        <Box
                          key={h.id}
                          sx={{
                            width: 8,
                            height: 8,
                            borderRadius: '50%',
                            bgcolor: h.color,
                            border: '1px solid rgba(0,0,0,0.3)',
                            boxShadow: '0 1px 2px rgba(0,0,0,0.2)',
                          }}
                        />
                      ))}
                      {segmentHighlights.length > 4 && (
                        <Typography
                          sx={{
                            fontSize: 9,
                            fontWeight: 600,
                            color: 'white',
                            textShadow: '0 1px 2px rgba(0,0,0,0.5)',
                          }}
                        >
                          +{segmentHighlights.length - 4}
                        </Typography>
                      )}
                    </Stack>
                  );
                })()}

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
              // Toggle highlight and Delete buttons
              <Stack direction="row" spacing={0.5}>
                <Tooltip title={
                  isSegmentInSelectedHighlight
                    ? `Remove from ${highlights.find(h => h.id === selectedHighlightId)?.name}`
                    : selectedHighlightId
                      ? `Add to ${highlights.find(h => h.id === selectedHighlightId)?.name}`
                      : 'Add to new highlight'
                }>
                  <IconButton
                    size="small"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleToggleHighlight();
                    }}
                    sx={{
                      width: 28,
                      height: 28,
                      bgcolor: isSegmentInSelectedHighlight
                        ? highlights.find(h => h.id === selectedHighlightId)?.color || '#FFE66D'
                        : 'rgba(20,20,20,0.9)',
                      color: isSegmentInSelectedHighlight
                        ? 'rgba(0,0,0,0.8)'
                        : selectedHighlightId
                          ? highlights.find(h => h.id === selectedHighlightId)?.color || 'rgba(255,255,255,0.8)'
                          : 'rgba(255,255,255,0.8)',
                      border: '1px solid rgba(255,255,255,0.15)',
                      boxShadow: '0 2px 8px rgba(0,0,0,0.4)',
                      '&:hover': {
                        bgcolor: isSegmentInSelectedHighlight
                          ? 'rgba(20,20,20,0.9)'
                          : selectedHighlightId
                            ? highlights.find(h => h.id === selectedHighlightId)?.color || '#FFE66D'
                            : '#FFE66D',
                        color: isSegmentInSelectedHighlight
                          ? highlights.find(h => h.id === selectedHighlightId)?.color || 'rgba(255,255,255,0.8)'
                          : 'rgba(0,0,0,0.8)',
                        border: '1px solid transparent',
                      },
                      transition: 'all 0.15s ease',
                    }}
                  >
                    <StarIcon sx={{ fontSize: 16 }} />
                  </IconButton>
                </Tooltip>
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
              </Stack>
            )}
          </Box>
        )}
      </Box>
    </Box>
  );
}
