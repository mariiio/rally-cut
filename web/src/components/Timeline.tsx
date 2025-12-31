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
    rallies,
    updateRally,
    selectRally,
    selectedRallyId,
    adjustRallyStart,
    adjustRallyEnd,
    createRallyAtTime,
    removeRally,
    videoMetadata,
    highlights,
    selectedHighlightId,
    createHighlight,
    addRallyToHighlight,
    removeRallyFromHighlight,
    selectHighlight,
    getHighlightsForRally,
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
    highlightRallyIndex,
    advanceHighlightPlayback,
    stopHighlightPlayback,
  } = usePlayerStore();
  const timelineRef = useRef<TimelineState>(null);
  const timelineContainerRef = useRef<HTMLDivElement>(null);
  const [isDraggingCursor, setIsDraggingCursor] = useState(false);
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);
  const [hotkeysAnchorEl, setHotkeysAnchorEl] = useState<HTMLButtonElement | null>(null);
  const [scrollLeft, setScrollLeft] = useState(0);
  const [containerWidth, setContainerWidth] = useState(0);

  // Track if we auto-paused at segment end (for restart behavior)
  const autoPausedAtEndRef = useRef<string | null>(null); // stores segment ID if auto-paused

  // Clear delete confirmation when selection changes
  useEffect(() => {
    setDeleteConfirmId(null);
  }, [selectedRallyId]);

  // Helper to check if current time is inside the selected rally
  const isInsideSelectedRally = useCallback(() => {
    if (!selectedRallyId || !rallies) return false;
    const rally = rallies.find(s => s.id === selectedRallyId);
    if (!rally) return false;
    return currentTime >= rally.start_time && currentTime <= rally.end_time;
  }, [selectedRallyId, rallies, currentTime]);

  // Get selected rally
  const getSelectedRally = useCallback(() => {
    if (!selectedRallyId || !rallies) return null;
    return rallies.find(s => s.id === selectedRallyId) || null;
  }, [selectedRallyId, rallies]);

  // Navigate to previous rally and select it
  const goToPrevRally = useCallback(() => {
    if (!rallies) return;
    const sorted = [...rallies].sort((a, b) => a.start_time - b.start_time);
    const prev = sorted.reverse().find(s => s.start_time < currentTime - 0.5);
    if (prev) {
      selectRally(prev.id);
      seek(prev.start_time);
    }
  }, [rallies, currentTime, seek, selectRally]);

  // Navigate to next rally and select it
  const goToNextRally = useCallback(() => {
    if (!rallies) return;
    const sorted = [...rallies].sort((a, b) => a.start_time - b.start_time);
    const next = sorted.find(s => s.start_time > currentTime + 0.5);
    if (next) {
      selectRally(next.id);
      seek(next.start_time);
    }
  }, [rallies, currentTime, seek, selectRally]);

  // Check if rally is in selected highlight
  const isRallyInSelectedHighlight = useMemo(() => {
    if (!selectedRallyId || !selectedHighlightId) return false;
    const highlight = highlights.find(h => h.id === selectedHighlightId);
    return highlight?.rallyIds.includes(selectedRallyId) ?? false;
  }, [selectedRallyId, selectedHighlightId, highlights]);

  // Toggle rally in highlight (add if not present, remove if present)
  const handleToggleHighlight = useCallback(() => {
    if (!selectedRallyId) return;

    if (selectedHighlightId) {
      // Check if rally is already in the selected highlight
      const highlight = highlights.find(h => h.id === selectedHighlightId);
      if (highlight?.rallyIds.includes(selectedRallyId)) {
        // Remove from highlight
        removeRallyFromHighlight(selectedRallyId, selectedHighlightId);
      } else {
        // Add to highlight
        addRallyToHighlight(selectedRallyId, selectedHighlightId);
      }
    } else {
      // Create new highlight and add rally
      const newId = createHighlight();
      addRallyToHighlight(selectedRallyId, newId);
      selectHighlight(newId);
    }
  }, [selectedRallyId, selectedHighlightId, highlights, addRallyToHighlight, removeRallyFromHighlight, createHighlight, selectHighlight]);

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
            const selectedRally = getSelectedRally();

            if (autoPausedAtEndRef.current && selectedRally && autoPausedAtEndRef.current === selectedRally.id) {
              // Auto-paused at rally end: restart from rally start
              seek(selectedRally.start_time);
              autoPausedAtEndRef.current = null;
              play();
            } else if (selectedRallyId) {
              // Has a selected rally
              if (isInsideSelectedRally()) {
                // Inside rally: play from current position
                play();
              } else {
                // Outside rally: deselect and play
                selectRally(null);
                play();
              }
            } else {
              // No rally selected: just play
              play();
            }
          }
          break;

        case 'ArrowLeft':
          e.preventDefault();
          if (isMod) {
            // Cmd/Ctrl + Left: go to previous rally
            goToPrevRally();
          } else if (selectedRallyId) {
            // Left with rally selected: adjust rally
            const rallyLeft = getSelectedRally();
            if (e.shiftKey) {
              // Shift + Left: shrink end (-0.5s)
              if (adjustRallyEnd(selectedRallyId, -0.5) && rallyLeft) {
                seek(rallyLeft.end_time - 0.5);
              }
            } else {
              // Left: expand start (-0.5s)
              if (adjustRallyStart(selectedRallyId, -0.5) && rallyLeft) {
                seek(rallyLeft.start_time - 0.5);
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
            // Cmd/Ctrl + Right: go to next rally
            goToNextRally();
          } else if (selectedRallyId) {
            // Right with rally selected: adjust rally
            const rallyRight = getSelectedRally();
            if (e.shiftKey) {
              // Shift + Right: expand end (+0.5s)
              if (adjustRallyEnd(selectedRallyId, 0.5) && rallyRight) {
                seek(rallyRight.end_time + 0.5);
              }
            } else {
              // Right: shrink start (+0.5s)
              if (adjustRallyStart(selectedRallyId, 0.5) && rallyRight) {
                seek(rallyRight.start_time + 0.5);
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
          if (selectedRallyId) {
            e.preventDefault();
            if (deleteConfirmId === selectedRallyId) {
              // Confirm delete on second press
              removeRally(selectedRallyId);
              setDeleteConfirmId(null);
              selectRally(null);
            } else {
              // Show confirmation on first press
              setDeleteConfirmId(selectedRallyId);
            }
          }
          break;

        case 'Escape':
          if (deleteConfirmId) {
            setDeleteConfirmId(null);
          } else if (selectedRallyId) {
            selectRally(null);
          }
          break;

        case 'Enter':
          if (isMod) {
            // Cmd/Ctrl + Enter: Create new rally at cursor (if not inside existing rally)
            if (videoMetadata && rallies) {
              const insideRally = rallies.some(
                (s) => currentTime >= s.start_time && currentTime <= s.end_time
              );
              if (!insideRally) {
                e.preventDefault();
                createRallyAtTime(currentTime);
              }
            }
          } else if (selectedRallyId) {
            // Enter (without modifier): Toggle rally in highlight
            e.preventDefault();
            handleToggleHighlight();
          }
          break;
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isPlaying, play, pause, seek, currentTime, duration, selectedRallyId, deleteConfirmId, removeRally, selectRally, isInsideSelectedRally, getSelectedRally, goToPrevRally, goToNextRally, adjustRallyStart, adjustRallyEnd, videoMetadata, rallies, createRallyAtTime, handleToggleHighlight]);

  // Jump to previous/next rally
  const jumpToPrevRally = useCallback(() => {
    if (!rallies) return;
    const sorted = [...rallies].sort((a, b) => a.start_time - b.start_time);
    const prev = sorted.reverse().find(s => s.start_time < currentTime - 0.5);
    if (prev) seek(prev.start_time);
  }, [rallies, currentTime, seek]);

  const jumpToNextRally = useCallback(() => {
    if (!rallies) return;
    const sorted = [...rallies].sort((a, b) => a.start_time - b.start_time);
    const next = sorted.find(s => s.start_time > currentTime + 0.5);
    if (next) seek(next.start_time);
  }, [rallies, currentTime, seek]);

  // Skip dead time - when playOnlyRallies is enabled, jump to next rally if in dead time
  useEffect(() => {
    if (!isPlaying || !playOnlyRallies || !rallies || rallies.length === 0) return;

    const sorted = [...rallies].sort((a, b) => a.start_time - b.start_time);

    // Check if current time is within any rally
    const inRally = sorted.some(s => currentTime >= s.start_time && currentTime <= s.end_time);

    if (!inRally) {
      // Find next rally to jump to
      const nextRally = sorted.find(s => s.start_time > currentTime);
      if (nextRally) {
        seek(nextRally.start_time);
      }
    }
  }, [currentTime, isPlaying, playOnlyRallies, rallies, seek]);

  // Stop playback at end of selected rally
  useEffect(() => {
    if (!isPlaying || !selectedRallyId || !rallies) return;

    const selectedRally = rallies.find(s => s.id === selectedRallyId);
    if (!selectedRally) return;

    // Pause when reaching the end of the selected rally (with small tolerance)
    if (currentTime >= selectedRally.end_time - 0.05) {
      autoPausedAtEndRef.current = selectedRallyId; // Mark that we auto-paused
      pause();
    }
  }, [currentTime, isPlaying, selectedRallyId, rallies, pause]);

  // Highlight playback - auto-advance through rallies
  useEffect(() => {
    if (!isPlaying || !playingHighlightId) return;

    const highlight = highlights.find(h => h.id === playingHighlightId);
    if (!highlight || highlight.rallyIds.length === 0) {
      stopHighlightPlayback();
      return;
    }

    // Get sorted rallies for this highlight
    const highlightRallies = rallies
      .filter(s => highlight.rallyIds.includes(s.id))
      .sort((a, b) => a.start_time - b.start_time);

    if (highlightRallyIndex >= highlightRallies.length) {
      // End of highlight - stop playback
      stopHighlightPlayback();
      return;
    }

    const currentRally = highlightRallies[highlightRallyIndex];
    if (!currentRally) {
      stopHighlightPlayback();
      return;
    }

    // Check if we need to jump to rally start (if we're before it or in dead time)
    if (currentTime < currentRally.start_time - 0.1) {
      seek(currentRally.start_time);
      return;
    }

    // Check if we reached the end of current rally
    if (currentTime >= currentRally.end_time - 0.05) {
      const nextIndex = highlightRallyIndex + 1;
      if (nextIndex < highlightRallies.length) {
        // Advance to next rally
        advanceHighlightPlayback();
        seek(highlightRallies[nextIndex].start_time);
      } else {
        // End of highlight
        stopHighlightPlayback();
      }
    }
  }, [currentTime, isPlaying, playingHighlightId, highlightRallyIndex, highlights, rallies, seek, advanceHighlightPlayback, stopHighlightPlayback]);

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
  }, [rallies?.length]); // Re-attach when rallies load

  // Handle scroll events from the timeline library
  const handleScroll = useCallback((params: { scrollLeft: number }) => {
    setScrollLeft(params.scrollLeft);
  }, []);

  // Also listen directly to scroll events on the ReactVirtualized Grid elements
  useEffect(() => {
    const container = timelineContainerRef.current;
    if (!container) return;

    // The actual scrollable elements are the ReactVirtualized Grids in both edit area and time area
    const editGrid = container.querySelector('.timeline-editor-edit-area .ReactVirtualized__Grid');
    const timeGrid = container.querySelector('.timeline-editor-time-area .ReactVirtualized__Grid');

    const handleDirectScroll = (e: Event) => {
      const target = e.target as HTMLElement;
      setScrollLeft(target.scrollLeft);
    };

    // Set initial scroll position
    if (editGrid) {
      setScrollLeft(editGrid.scrollLeft);
    }

    editGrid?.addEventListener('scroll', handleDirectScroll);
    timeGrid?.addEventListener('scroll', handleDirectScroll);
    return () => {
      editGrid?.removeEventListener('scroll', handleDirectScroll);
      timeGrid?.removeEventListener('scroll', handleDirectScroll);
    };
  }, [rallies?.length]); // Re-attach when rallies load

  // Track container width for cursor visibility
  useEffect(() => {
    const container = timelineContainerRef.current;
    if (!container) return;

    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setContainerWidth(entry.contentRect.width);
      }
    });

    setContainerWidth(container.clientWidth);
    resizeObserver.observe(container);

    return () => resizeObserver.disconnect();
  }, []);

  // Calculate cursor pixel position (accounting for scroll offset)
  const cursorPixelPosition = useMemo(() => {
    const pixelsPerSecond = SCALE_WIDTH / scale;
    const startLeft = 10; // matches startLeft prop
    return startLeft + (currentTime * pixelsPerSecond) - scrollLeft;
  }, [currentTime, scale, scrollLeft]);

  // Calculate selected rally position for delete button overlay (accounting for scroll offset)
  const selectedRallyPosition = useMemo(() => {
    if (!selectedRallyId || !rallies) return null;
    const rally = rallies.find(s => s.id === selectedRallyId);
    if (!rally) return null;

    const pixelsPerSecond = SCALE_WIDTH / scale;
    const startLeft = 10;
    const left = startLeft + (rally.start_time * pixelsPerSecond) - scrollLeft;
    const width = (rally.end_time - rally.start_time) * pixelsPerSecond;
    const center = left + (width / 2);

    return { left, width, center };
  }, [selectedRallyId, rallies, scale, scrollLeft]);

  // Convert Rally[] to TimelineRow[] format
  const editorData: TimelineRow[] = useMemo(() => {
    return [
      {
        id: 'rallies',
        actions: (rallies ?? []).map((rally) => ({
          id: rally.id,
          start: rally.start_time,
          end: rally.end_time,
          effectId: 'rally',
          selected: rally.id === selectedRallyId,
          flexible: false,
          movable: false,
        })),
      },
    ];
  }, [rallies, selectedRallyId]);

  // Check if a move/resize would cause overlap
  const checkOverlap = useCallback(
    (actionId: string, newStart: number, newEnd: number) => {
      return (rallies ?? []).some(
        (rally) =>
          rally.id !== actionId &&
          newStart < rally.end_time &&
          newEnd > rally.start_time
      );
    },
    [rallies]
  );

  // Handle rally changes (resize/move)
  const handleChange = useCallback(
    (data: TimelineRow[]) => {
      if (!rallies) return;
      const actions = data[0]?.actions || [];
      actions.forEach((action: TimelineAction) => {
        const rally = rallies.find((s) => s.id === action.id);
        if (rally) {
          if (
            rally.start_time !== action.start ||
            rally.end_time !== action.end
          ) {
            updateRally(action.id, {
              start_time: action.start,
              end_time: action.end,
            });
          }
        }
      });
    },
    [rallies, updateRally]
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
      updateRally(params.action.id, {
        start_time: params.start,
        end_time: params.end,
      });
      return true;
    },
    [updateRally]
  );

  // Persist changes after move ends
  const handleActionMoveEnd = useCallback(
    (params: { action: TimelineAction; start: number; end: number }) => {
      updateRally(params.action.id, {
        start_time: params.start,
        end_time: params.end,
      });
      return true;
    },
    [updateRally]
  );

  // Handle clicking on timeline to seek and deselect
  const handleClickTimeArea = useCallback(
    (time: number) => {
      seek(time);
      selectRally(null);
      return true;
    },
    [seek, selectRally]
  );

  // Handle clicking on an action to select it
  const handleClickAction = useCallback(
    (_e: React.MouseEvent, action: { action: TimelineAction }) => {
      selectRally(action.action.id);
      seek(action.action.start);
    },
    [selectRally, seek]
  );

  // Custom scale rendering
  const getScaleRender = useCallback((scaleValue: number) => {
    return <span>{formatTimeShort(scaleValue)}</span>;
  }, []);

  // Create rally at current playhead position
  const handleCreateRally = useCallback(() => {
    createRallyAtTime(currentTime);
  }, [createRallyAtTime, currentTime]);

  // Check if we can create a rally at current time (for toolbar button)
  const canCreateRallyToolbar = useMemo(() => {
    if (!videoMetadata || !rallies) return false;
    const insideRally = rallies.some(
      (s) => currentTime >= s.start_time && currentTime <= s.end_time
    );
    return !insideRally;
  }, [rallies, currentTime, videoMetadata]);

  // Check if we should show the floating + button on cursor
  const showFloatingAddButton = useMemo(() => {
    if (!videoMetadata || !rallies) return false;
    if (isDraggingCursor) return false;
    if (selectedRallyId) return false; // Hide when a rally is selected
    // Check if we're inside an existing rally
    const insideRally = rallies.some(
      (s) => currentTime >= s.start_time && currentTime <= s.end_time
    );
    return !insideRally;
  }, [rallies, currentTime, videoMetadata, isDraggingCursor, selectedRallyId]);

  if (!rallies || rallies.length === 0) {
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
          <IconButton size="small" onClick={jumpToPrevRally} title="Previous rally">
            <SkipPreviousIcon fontSize="small" />
          </IconButton>
          <IconButton size="small" onClick={() => isPlaying ? pause() : play()} title="Play/Pause (Space)">
            {isPlaying ? <PauseIcon /> : <PlayArrowIcon />}
          </IconButton>
          <IconButton size="small" onClick={jumpToNextRally} title="Next rally">
            <SkipNextIcon fontSize="small" />
          </IconButton>
          <Typography variant="body2" sx={{ ml: 1, fontFamily: 'monospace', minWidth: 130 }}>
            {formatTime(currentTime)} / {formatTime(duration)}
          </Typography>
          <Tooltip title={canCreateRallyToolbar ? "Create rally at playhead" : "Cannot create rally here (inside existing rally)"} arrow>
            <span>
              <IconButton
                size="small"
                onClick={handleCreateRally}
                disabled={!canCreateRallyToolbar}
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
            {rallies?.length ?? 0} rallies
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
          // Deselect if clicking on empty area (not on a rally)
          const target = e.target as HTMLElement;
          if (!target.closest('.timeline-editor-action')) {
            selectRally(null);
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
            overflowX: 'auto !important',
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
            ...(selectedRallyId && {
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
          minScaleCount={1}
          maxScaleCount={Math.max(1, duration > 0 ? Math.ceil(duration / scale) + 1 : 100)}
          onScroll={handleScroll}
          getScaleRender={getScaleRender}
          getActionRender={(action) => {
            const rally = rallies?.find((s) => s.id === action.id);
            const isSelected = selectedRallyId === action.id;

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
              if (adjustRallyStart(action.id, delta) && rally) {
                seek(rally.start_time + delta);
              }
            };

            const handleAdjustEnd = (delta: number) => {
              if (adjustRallyEnd(action.id, delta) && rally) {
                seek(rally.end_time + delta);
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
                    {rally?.id}
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
                  const rallyHighlights = getHighlightsForRally(action.id);
                  if (rallyHighlights.length === 0) return null;
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
                      {rallyHighlights.slice(0, 4).map((h) => (
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
                      {rallyHighlights.length > 4 && (
                        <Typography
                          sx={{
                            fontSize: 9,
                            fontWeight: 600,
                            color: 'white',
                            textShadow: '0 1px 2px rgba(0,0,0,0.5)',
                          }}
                        >
                          +{rallyHighlights.length - 4}
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
        {cursorPixelPosition > 0 && cursorPixelPosition < (containerWidth || 9999) && (
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
            {/* Add button - only when in non-rally area */}
            {showFloatingAddButton && (
              <IconButton
                onClick={(e) => {
                  e.stopPropagation();
                  handleCreateRally();
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

        {/* Delete button overlay for selected rally */}
        {selectedRallyId && selectedRallyPosition && selectedRallyPosition.center > 0 && (
          <Box
            sx={{
              position: 'absolute',
              left: selectedRallyPosition.center,
              top: 36,
              transform: 'translateX(-50%)',
              zIndex: 100,
              pointerEvents: 'auto',
            }}
          >
            {deleteConfirmId === selectedRallyId ? (
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
                    removeRally(selectedRallyId);
                    setDeleteConfirmId(null);
                    selectRally(null);
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
                  isRallyInSelectedHighlight
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
                      bgcolor: isRallyInSelectedHighlight
                        ? highlights.find(h => h.id === selectedHighlightId)?.color || '#FFE66D'
                        : 'rgba(20,20,20,0.9)',
                      color: isRallyInSelectedHighlight
                        ? 'rgba(0,0,0,0.8)'
                        : selectedHighlightId
                          ? highlights.find(h => h.id === selectedHighlightId)?.color || 'rgba(255,255,255,0.8)'
                          : 'rgba(255,255,255,0.8)',
                      border: '1px solid rgba(255,255,255,0.15)',
                      boxShadow: '0 2px 8px rgba(0,0,0,0.4)',
                      '&:hover': {
                        bgcolor: isRallyInSelectedHighlight
                          ? 'rgba(20,20,20,0.9)'
                          : selectedHighlightId
                            ? highlights.find(h => h.id === selectedHighlightId)?.color || '#FFE66D'
                            : '#FFE66D',
                        color: isRallyInSelectedHighlight
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
                    setDeleteConfirmId(selectedRallyId);
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
