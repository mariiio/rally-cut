'use client';

import { useMemo, useCallback, useState, useEffect, useRef } from 'react';
import { Box, Typography, IconButton, Stack, Tooltip, Popover, Button, CircularProgress } from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import PauseIcon from '@mui/icons-material/Pause';
import SkipPreviousIcon from '@mui/icons-material/SkipPrevious';
import SkipNextIcon from '@mui/icons-material/SkipNext';
import FastForwardIcon from '@mui/icons-material/FastForward';
import AddIcon from '@mui/icons-material/Add';
import DeleteIcon from '@mui/icons-material/Delete';
import CheckIcon from '@mui/icons-material/Check';
import CloseIcon from '@mui/icons-material/Close';
import KeyboardIcon from '@mui/icons-material/Keyboard';
import StarIcon from '@mui/icons-material/Star';
import AutoFixHighIcon from '@mui/icons-material/AutoFixHigh';
import CameraAltIcon from '@mui/icons-material/CameraAlt';
import MergeTypeIcon from '@mui/icons-material/MergeType';
import {
  Timeline as TimelineEditor,
  TimelineRow,
  TimelineAction,
  TimelineEffect,
  TimelineState,
} from '@xzdarcy/react-timeline-editor';
import { useEditorStore } from '@/stores/editorStore';
import { usePlayerStore } from '@/stores/playerStore';
import { useCameraStore } from '@/stores/cameraStore';
import { formatTimeShort, formatTime } from '@/utils/timeFormat';
import { triggerRallyDetection, getDetectionStatus } from '@/services/api';

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
        {keys.map((key) => (
          <Box
            key={key}
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
    mergeRallies,
    videoMetadata,
    highlights,
    selectedHighlightId,
    createHighlight,
    addRallyToHighlight,
    removeRallyFromHighlight,
    selectHighlight,
    getHighlightsForRally,
    activeMatchId,
    reloadCurrentMatch,
    isRallyEditingLocked,
    isCameraTabActive,
    setIsCameraTabActive,
    setLeftPanelTab,
    expandHighlight,
  } = useEditorStore();

  // Check if rally editing is locked (after confirmation)
  const isLocked = isRallyEditingLocked();

  // Camera edit mode: rally selected AND camera tab is active
  const isInCameraEditMode = selectedRallyId !== null && isCameraTabActive;

  // Camera edits for keyframe markers
  const cameraEdits = useCameraStore((state) => state.cameraEdits);
  const selectKeyframe = useCameraStore((state) => state.selectKeyframe);
  const selectedKeyframeId = useCameraStore((state) => state.selectedKeyframeId);
  const removeKeyframe = useCameraStore((state) => state.removeKeyframe);
  const setCameraEdit = useCameraStore((state) => state.setCameraEdit);

  // Detection state
  const [isDetecting, setIsDetecting] = useState(false);
  const [detectionStatus, setDetectionStatus] = useState<string | null>(null);
  const [detectionError, setDetectionError] = useState<string | null>(null);
  const [detectionProgress, setDetectionProgress] = useState(0);
  const [elapsedTime, setElapsedTime] = useState(0);
  const [videoDetectionStatus, setVideoDetectionStatus] = useState<string | null>(null); // UPLOADED, DETECTING, DETECTED
  const [detectionResult, setDetectionResult] = useState<{ ralliesCount: number } | null>(null);
  const pollIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const elapsedIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const {
    currentTime,
    duration,
    seek,
    isPlaying,
    pause,
    play,
    playOnlyRallies,
    togglePlayOnlyRallies,
    bufferedRanges,
  } = usePlayerStore();
  const timelineRef = useRef<TimelineState>(null);
  const timelineContainerRef = useRef<HTMLDivElement>(null);
  const [isDraggingCursor, setIsDraggingCursor] = useState(false);
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);
  const [hotkeysAnchorEl, setHotkeysAnchorEl] = useState<HTMLButtonElement | null>(null);
  const [scrollLeft, setScrollLeft] = useState(0);
  const [containerWidth, setContainerWidth] = useState(0);
  const [keyframeBlockedRallyId, setKeyframeBlockedRallyId] = useState<string | null>(null);
  const keyframeBlockedTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Pending resize position - used to keep library in sync during drag
  // This overrides rally position in editorData during active resize
  const [pendingResize, setPendingResize] = useState<{
    rallyId: string;
    start: number;
    end: number;
  } | null>(null);

  // Track if we auto-paused at segment end (for restart behavior)
  const autoPausedAtEndRef = useRef<string | null>(null); // stores segment ID if auto-paused

  // Clear delete confirmation when selection changes
  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- intentional reset on selection change
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

  // Get the target highlight for toggle action (selected or last one)
  const targetHighlight = useMemo(() => {
    if (selectedHighlightId) {
      return highlights.find(h => h.id === selectedHighlightId) ?? null;
    }
    return highlights.length > 0 ? highlights[highlights.length - 1] : null;
  }, [selectedHighlightId, highlights]);

  // Check if rally is in the target highlight
  const isRallyInTargetHighlight = useMemo(() => {
    if (!selectedRallyId || !targetHighlight) return false;
    return targetHighlight.rallyIds.includes(selectedRallyId);
  }, [selectedRallyId, targetHighlight]);

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
        setLeftPanelTab('highlights');
        expandHighlight(selectedHighlightId);
      }
    } else if (highlights.length > 0) {
      // No highlight selected but highlights exist - add to the latest one
      const latestHighlight = highlights[highlights.length - 1];
      addRallyToHighlight(selectedRallyId, latestHighlight.id);
      selectHighlight(latestHighlight.id);
      setLeftPanelTab('highlights');
      expandHighlight(latestHighlight.id);
    } else {
      // No highlights exist - create new one
      const newId = createHighlight();
      addRallyToHighlight(selectedRallyId, newId);
      selectHighlight(newId);
      setLeftPanelTab('highlights');
      expandHighlight(newId);
    }
  }, [selectedRallyId, selectedHighlightId, highlights, addRallyToHighlight, removeRallyFromHighlight, createHighlight, selectHighlight, setLeftPanelTab, expandHighlight]);

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
          } else if (isInCameraEditMode) {
            // Camera edit mode: move playhead left 0.3s within rally bounds (finer control)
            const rally = getSelectedRally();
            if (rally) {
              const newTime = Math.max(rally.start_time, currentTime - 0.3);
              seek(newTime);
            }
          } else if (selectedRallyId && !isLocked) {
            // Left with rally selected (not in camera mode): adjust rally
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
            // Arrow Left: seek back 1 second
            seek(Math.max(0, currentTime - 1));
            // Clear auto-pause flag when manually seeking
            autoPausedAtEndRef.current = null;
          }
          break;

        case 'ArrowRight':
          e.preventDefault();
          if (isMod) {
            // Cmd/Ctrl + Right: go to next rally
            goToNextRally();
          } else if (isInCameraEditMode) {
            // Camera edit mode: move playhead right 0.3s within rally bounds (finer control)
            const rally = getSelectedRally();
            if (rally) {
              const newTime = Math.min(rally.end_time, currentTime + 0.3);
              seek(newTime);
            }
          } else if (selectedRallyId && !isLocked) {
            // Right with rally selected (not in camera mode): adjust rally
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
            // Arrow Right: seek forward 1 second
            seek(Math.min(duration, currentTime + 1));
            // Clear auto-pause flag when manually seeking
            autoPausedAtEndRef.current = null;
          }
          break;

        case 'Delete':
        case 'Backspace':
          // Priority 1: Delete keyframe in camera edit mode (single press)
          if (isInCameraEditMode && selectedKeyframeId && selectedRallyId) {
            e.preventDefault();
            removeKeyframe(selectedRallyId, selectedKeyframeId);
            selectKeyframe(null);
            break;
          }

          // Priority 2: Delete rally (when not in camera edit mode)
          if (selectedRallyId && !isLocked && !isInCameraEditMode) {
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
            // Cmd/Ctrl + Enter: Create new rally at cursor (disabled when locked)
            if (videoMetadata && rallies && !isLocked) {
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
  }, [isPlaying, play, pause, seek, currentTime, duration, selectedRallyId, deleteConfirmId, removeRally, selectRally, isInsideSelectedRally, getSelectedRally, goToPrevRally, goToNextRally, adjustRallyStart, adjustRallyEnd, videoMetadata, rallies, createRallyAtTime, handleToggleHighlight, isLocked, isInCameraEditMode, selectedKeyframeId, removeKeyframe, selectKeyframe]);

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

  // Note: Highlight playback (including cross-match) is now handled by VideoPlayer
  // using a stable playlist stored in playerStore

  // Calculate optimal scale based on video duration and container width
  // scale = seconds per marker, SCALE_WIDTH = pixels per marker
  // totalWidth = (duration / scale) * SCALE_WIDTH
  // We want totalWidth = containerWidth, so scale = duration * SCALE_WIDTH / containerWidth
  const getAutoScale = useCallback(() => {
    if (duration > 0 && containerWidth > 0) {
      // Calculate exact scale to fit entire duration in container width
      const exactScale = (duration * SCALE_WIDTH) / containerWidth;
      // Return as-is (no rounding) to maintain exact fit, but respect min
      return Math.max(MIN_SCALE, exactScale);
    }
    return 30;
  }, [duration, containerWidth]);

  const [scale, setScale] = useState(() => 30);

  // Auto-fit scale when duration or container width changes
  useEffect(() => {
    if (duration > 0 && containerWidth > 0) {
      const newScale = getAutoScale();
      // eslint-disable-next-line react-hooks/set-state-in-effect -- intentional auto-fit on resize
      setScale(newScale);
    }
  }, [duration, containerWidth, getAutoScale]);

  // Track previous camera edit mode state and store original scale/scroll
  const prevCameraEditModeRef = useRef<boolean>(false);
  const originalScaleRef = useRef<number | null>(null);
  const editedRallyIdRef = useRef<string | null>(null);

  // Zoom and center timeline on selected rally when entering camera edit mode
  // Restore original scale and center on edited rally when exiting camera edit mode
  useEffect(() => {
    const wasInCameraEditMode = prevCameraEditModeRef.current;
    prevCameraEditModeRef.current = isInCameraEditMode;

    // Exiting camera edit mode - restore original scale and center on edited rally
    if (wasInCameraEditMode && !isInCameraEditMode && originalScaleRef.current !== null) {
      const restoredScale = originalScaleRef.current;
      const editedRallyId = editedRallyIdRef.current;
      setScale(restoredScale);
      originalScaleRef.current = null;

      // Center on the rally that was being edited
      if (editedRallyId && containerWidth) {
        const rally = rallies?.find((r) => r.id === editedRallyId);
        if (rally) {
          setTimeout(() => {
            const container = timelineContainerRef.current;
            if (!container) return;

            const editGrid = container.querySelector('.timeline-editor-edit-area .ReactVirtualized__Grid');
            if (editGrid) {
              const pixelsPerSecond = SCALE_WIDTH / restoredScale;
              const rallyCenter = (rally.start_time + rally.end_time) / 2;
              const centerScrollLeft = Math.max(0, (rallyCenter * pixelsPerSecond) - (containerWidth / 2));
              editGrid.scrollLeft = centerScrollLeft;
            }
          }, 50);
        }
      }
      editedRallyIdRef.current = null;
      return;
    }

    // Entering camera edit mode - save scale and zoom to rally
    if (!wasInCameraEditMode && isInCameraEditMode) {
      if (!containerWidth || !rallies) return;
      const rally = rallies.find((r) => r.id === selectedRallyId);
      if (!rally) return;

      const rallyDuration = rally.end_time - rally.start_time;
      if (rallyDuration <= 0) return;

      // Save current scale and rally ID before zooming
      originalScaleRef.current = scale;
      editedRallyIdRef.current = selectedRallyId;

      // Target: rally takes ~60% of viewport width
      const targetWidth = containerWidth * 0.6;
      const optimalScale = (rallyDuration * SCALE_WIDTH) / targetWidth;
      const newScale = Math.max(MIN_SCALE, Math.min(optimalScale, 30));

      setScale(newScale);

      // Scroll to center the rally after a brief delay for scale to apply
      setTimeout(() => {
        const container = timelineContainerRef.current;
        if (!container) return;

        const editGrid = container.querySelector('.timeline-editor-edit-area .ReactVirtualized__Grid');
        if (editGrid) {
          const pixelsPerSecond = SCALE_WIDTH / newScale;
          const rallyCenter = (rally.start_time + rally.end_time) / 2;
          const centerScrollLeft = Math.max(0, (rallyCenter * pixelsPerSecond) - (containerWidth / 2));
          editGrid.scrollLeft = centerScrollLeft;
        }
      }, 50);
    }
  }, [isInCameraEditMode, selectedRallyId, rallies, containerWidth, scale]);

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

  // Calculate close rally pairs for merge buttons (gap <= 3 seconds)
  const closeRallyPairs = useMemo(() => {
    if (!rallies || rallies.length < 2 || isLocked) return [];
    const sorted = [...rallies].sort((a, b) => a.start_time - b.start_time);
    const pairs: { first: typeof sorted[0]; second: typeof sorted[0]; gap: number; position: number }[] = [];

    const pixelsPerSecond = SCALE_WIDTH / scale;
    const startLeft = 10;

    for (let i = 0; i < sorted.length - 1; i++) {
      const gap = sorted[i + 1].start_time - sorted[i].end_time;
      if (gap <= 3.0 && gap > 0) {
        const midpoint = sorted[i].end_time + gap / 2;
        const position = startLeft + (midpoint * pixelsPerSecond) - scrollLeft;
        pairs.push({ first: sorted[i], second: sorted[i + 1], gap, position });
      }
    }
    return pairs;
  }, [rallies, scale, scrollLeft, isLocked]);

  // Convert Rally[] to TimelineRow[] format
  // Use pendingResize position during active drag to keep library in sync
  const editorData: TimelineRow[] = useMemo(() => {
    return [
      {
        id: 'rallies',
        actions: (rallies ?? []).map((rally) => {
          // During resize, use pending position to prevent library desync
          const usePending = pendingResize && pendingResize.rallyId === rally.id;
          return {
            id: rally.id,
            start: usePending ? pendingResize.start : rally.start_time,
            end: usePending ? pendingResize.end : rally.end_time,
            effectId: 'rally',
            selected: rally.id === selectedRallyId,
            flexible: !isLocked && !isInCameraEditMode,
            movable: false,
          };
        }),
      },
    ];
  }, [rallies, selectedRallyId, isLocked, isInCameraEditMode, pendingResize]);

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

  // Handle rally changes - only used for visual updates during drag
  // Actual persistence happens in handleActionResizeEnd/handleActionMoveEnd
  const handleChange = useCallback(
    () => {
      // No-op: we persist changes in onActionResizeEnd/onActionMoveEnd
      // to avoid double history entries
    },
    []
  );

  // Prevent overlapping during move (also disable when locked)
  const handleActionMoving = useCallback(
    (params: { action: TimelineAction; start: number; end: number }) => {
      if (isLocked) return false;
      return !checkOverlap(params.action.id, params.start, params.end);
    },
    [checkOverlap, isLocked]
  );

  // Handle resize: always return true to avoid shift, clamp position via pendingResize
  // The visual offset is corrected via CSS transform in getActionRender
  const handleActionResizing = useCallback(
    (params: { action: TimelineAction; start: number; end: number }) => {
      if (isLocked) return false;

      const rally = rallies?.find((r) => r.id === params.action.id);
      if (!rally) return true;

      // Check overlap - keep current position if would overlap
      if (checkOverlap(params.action.id, params.start, params.end)) {
        // Don't update pendingResize, keep last valid
        return true;
      }

      // Calculate clamped position based on keyframe boundaries
      let clampedStart = params.start;
      let clampedEnd = params.end;
      let isBlocked = false;

      const cameraEdit = cameraEdits[params.action.id];
      if (cameraEdit) {
        const duration = rally.end_time - rally.start_time;
        const allKeyframes = [
          ...(cameraEdit.keyframes.ORIGINAL ?? []),
          ...(cameraEdit.keyframes.VERTICAL ?? []),
        ];

        for (const kf of allKeyframes) {
          const kfAbsTime = rally.start_time + kf.timeOffset * duration;
          // Clamp start - can't exclude keyframe
          if (kfAbsTime < clampedStart) {
            clampedStart = kfAbsTime;
            isBlocked = true;
          }
          // Clamp end - can't exclude keyframe
          if (kfAbsTime > clampedEnd) {
            clampedEnd = kfAbsTime;
            isBlocked = true;
          }
        }

        // Show keyframe indicators when blocked
        if (isBlocked) {
          if (keyframeBlockedRallyId !== params.action.id) {
            setKeyframeBlockedRallyId(params.action.id);
          }
          if (keyframeBlockedTimeoutRef.current) {
            clearTimeout(keyframeBlockedTimeoutRef.current);
            keyframeBlockedTimeoutRef.current = null;
          }
        }
      }

      // Update pendingResize with CLAMPED position (CSS will correct visual)
      setPendingResize({
        rallyId: params.action.id,
        start: clampedStart,
        end: clampedEnd,
      });

      // Update video preview to clamped position
      const startDelta = params.start - rally.start_time;
      const isResizingStart = Math.abs(startDelta) > 0.01;
      seek(isResizingStart ? clampedStart : clampedEnd);

      return true;
    },
    [checkOverlap, isLocked, cameraEdits, rallies, seek, keyframeBlockedRallyId]
  );

  // Hide keyframe indicators and clear pending resize when mouse is released
  useEffect(() => {
    const handleMouseUp = () => {
      // Clear pending resize (drag ended)
      setPendingResize(null);

      if (keyframeBlockedRallyId) {
        // Clear any existing timeout
        if (keyframeBlockedTimeoutRef.current) {
          clearTimeout(keyframeBlockedTimeoutRef.current);
        }
        // Hide after a delay so user can see why resize was blocked
        keyframeBlockedTimeoutRef.current = setTimeout(() => {
          setKeyframeBlockedRallyId(null);
        }, 800);
      }
    };
    window.addEventListener('mouseup', handleMouseUp);
    return () => window.removeEventListener('mouseup', handleMouseUp);
  }, [keyframeBlockedRallyId]);

  // Persist changes after resize ends
  // Use pendingResize values (clamped) instead of raw params (may have library offset)
  // Also recalculate keyframe offsets to keep them at the same absolute video time
  const handleActionResizeEnd = useCallback(
    (params: { action: TimelineAction; start: number; end: number }) => {
      const rally = rallies?.find((r) => r.id === params.action.id);
      const cameraEdit = cameraEdits[params.action.id];

      // Use pending position if available (it's clamped), otherwise use params
      const finalStart = pendingResize?.rallyId === params.action.id ? pendingResize.start : params.start;
      const finalEnd = pendingResize?.rallyId === params.action.id ? pendingResize.end : params.end;

      // Recalculate keyframe timeOffsets if rally has camera edits
      if (rally && cameraEdit) {
        const oldDuration = rally.end_time - rally.start_time;
        const newDuration = finalEnd - finalStart;

        const updateKeyframesForRatio = (keyframes: typeof cameraEdit.keyframes.ORIGINAL) => {
          return keyframes.map((kf) => {
            // Convert old timeOffset to absolute time
            const absTime = rally.start_time + kf.timeOffset * oldDuration;
            // Convert back to new timeOffset
            const newTimeOffset = (absTime - finalStart) / newDuration;
            return { ...kf, timeOffset: Math.max(0, Math.min(1, newTimeOffset)) };
          });
        };

        const updatedEdit = {
          ...cameraEdit,
          keyframes: {
            ORIGINAL: updateKeyframesForRatio(cameraEdit.keyframes.ORIGINAL ?? []),
            VERTICAL: updateKeyframesForRatio(cameraEdit.keyframes.VERTICAL ?? []),
          },
        };
        setCameraEdit(params.action.id, updatedEdit);
      }

      updateRally(params.action.id, {
        start_time: finalStart,
        end_time: finalEnd,
      });

      // Clear pending resize
      setPendingResize(null);
      return true;
    },
    [updateRally, rallies, cameraEdits, setCameraEdit, pendingResize]
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

  // Handle clicking on timeline to seek (don't deselect - user can press Escape)
  const handleClickTimeArea = useCallback(
    (time: number) => {
      // In camera edit mode, constrain to rally bounds
      if (isInCameraEditMode) {
        const rally = getSelectedRally();
        if (rally) {
          const constrainedTime = Math.max(rally.start_time, Math.min(rally.end_time, time));
          seek(constrainedTime);
          return true;
        }
      }
      seek(time);
      // Don't deselect rally - allow moving playhead while keeping rally selected
      return true;
    },
    [seek, isInCameraEditMode, getSelectedRally]
  );

  // Handle clicking on an action to select it
  const handleClickAction = useCallback(
    (_e: React.MouseEvent, action: { action: TimelineAction }) => {
      const clickedSameRally = action.action.id === selectedRallyId;
      selectRally(action.action.id);
      seek(action.action.start);
      // Only exit camera edit mode when clicking on a different rally
      if (!clickedSameRally) {
        setIsCameraTabActive(false);
      }
    },
    [selectRally, seek, setIsCameraTabActive, selectedRallyId]
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
    if (!videoMetadata || !rallies || isLocked) return false;
    const insideRally = rallies.some(
      (s) => currentTime >= s.start_time && currentTime <= s.end_time
    );
    return !insideRally;
  }, [rallies, currentTime, videoMetadata, isLocked]);

  // Check if we should show the floating + button on cursor
  const showFloatingAddButton = useMemo(() => {
    if (!videoMetadata || !rallies || isLocked) return false;
    if (isDraggingCursor) return false;
    if (selectedRallyId) return false; // Hide when a rally is selected
    // Check if we're inside an existing rally
    const insideRally = rallies.some(
      (s) => currentTime >= s.start_time && currentTime <= s.end_time
    );
    return !insideRally;
  }, [rallies, currentTime, videoMetadata, isDraggingCursor, selectedRallyId, isLocked]);

  // Format elapsed time as MM:SS
  const formatElapsed = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Stop polling - defined before startPolling since it's used there
  const stopPolling = useCallback(() => {
    if (elapsedIntervalRef.current) {
      clearInterval(elapsedIntervalRef.current);
      elapsedIntervalRef.current = null;
    }
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
      pollIntervalRef.current = null;
    }
  }, []);

  // Start polling for detection status
  const startPolling = useCallback((startTime?: number) => {
    if (!activeMatchId) return;

    // Set up elapsed time tracking
    const start = startTime || Date.now();
    setElapsedTime(Math.floor((Date.now() - start) / 1000));

    // Clear any existing intervals
    if (elapsedIntervalRef.current) clearInterval(elapsedIntervalRef.current);
    if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);

    // Update elapsed time every second
    elapsedIntervalRef.current = setInterval(() => {
      setElapsedTime(Math.floor((Date.now() - start) / 1000));
    }, 1000);

    // Poll for status every 5 seconds
    pollIntervalRef.current = setInterval(async () => {
      try {
        const status = await getDetectionStatus(activeMatchId);
        if (status.job?.status === 'COMPLETED') {
          stopPolling();
          setDetectionProgress(100);
          setDetectionStatus('Loading results...');
          // Reload match data from API instead of page reload
          const result = await reloadCurrentMatch();
          setIsDetecting(false);
          setVideoDetectionStatus('DETECTED');
          if (result) {
            setDetectionResult(result);
            if (result.ralliesCount > 0) {
              setDetectionStatus(`Found ${result.ralliesCount} rallies!`);
            } else {
              setDetectionStatus('No rallies detected');
              setDetectionError('The ML model did not find any rallies in this video. You can add them manually.');
            }
          } else {
            setDetectionStatus('Detection complete');
          }
          // Clear success message after 5 seconds
          setTimeout(() => {
            setDetectionStatus(null);
            setDetectionResult(null);
          }, 5000);
        } else if (status.job?.status === 'FAILED') {
          stopPolling();
          setIsDetecting(false);
          setDetectionError(status.job.errorMessage || 'Detection failed');
          setDetectionStatus(null);
        } else {
          // Update progress from server
          const progress = status.job?.progress ?? 0;
          const message = status.job?.progressMessage || (status.job?.status === 'RUNNING' ? 'Processing video' : 'Preparing');
          setDetectionProgress(progress);
          setDetectionStatus(message);
        }
      } catch {
        // Ignore polling errors, keep trying
      }
    }, 5000);
  }, [activeMatchId, stopPolling, reloadCurrentMatch]);

  // Check detection status on mount
  useEffect(() => {
    if (!activeMatchId) return;

    const checkInitialStatus = async () => {
      try {
        const status = await getDetectionStatus(activeMatchId);
        setVideoDetectionStatus(status.status);
        if (status.status === 'DETECTING' && status.job?.status === 'RUNNING') {
          // Detection is already in progress - resume polling
          setIsDetecting(true);
          setDetectionProgress(status.job.progress ?? 0);
          setDetectionStatus(status.job.progressMessage || 'Processing video');
          setDetectionError(null);
          // Use job start time if available
          const startTime = status.job.startedAt ? new Date(status.job.startedAt).getTime() : Date.now();
          startPolling(startTime);
        }
      } catch {
        // Ignore errors on initial check
      }
    };

    checkInitialStatus();

    // Cleanup on unmount
    return () => stopPolling();
  }, [activeMatchId, startPolling, stopPolling]);

  // Handle starting rally detection
  const handleStartDetection = async () => {
    if (!activeMatchId) return;

    setIsDetecting(true);
    setDetectionStatus('Starting analysis...');
    setDetectionProgress(0);
    setDetectionError(null);
    setElapsedTime(0);

    try {
      await triggerRallyDetection(activeMatchId);
      setDetectionStatus('Processing video');
      startPolling(Date.now());
    } catch (err) {
      setIsDetecting(false);
      setDetectionError(err instanceof Error ? err.message : 'Failed to start detection');
      setDetectionStatus(null);
    }
  };

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
          <IconButton
            size="small"
            onClick={() => isPlaying ? pause() : play()}
            title="Play/Pause (Space)"
            sx={{
              bgcolor: 'primary.main',
              color: 'primary.contrastText',
              '&:hover': { bgcolor: 'primary.dark' },
            }}
          >
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

        {/* Rallies only toggle - Center (hidden when confirmed since video already contains only rallies) */}
        {!isLocked && (
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
        )}

        {/* Info and controls - Right */}
        <Stack direction="row" alignItems="center" spacing={1} sx={{ flex: 1, justifyContent: 'flex-end' }}>
          {/* Detection status or button */}
          {isDetecting ? (
            <Stack direction="row" alignItems="center" spacing={1} sx={{
              bgcolor: 'action.hover',
              px: 1.5,
              py: 0.5,
              borderRadius: 2,
            }}>
              <CircularProgress
                size={18}
                variant={detectionProgress > 0 ? "determinate" : "indeterminate"}
                value={detectionProgress}
                color="primary"
              />
              <Stack spacing={0}>
                <Typography variant="caption" sx={{ color: 'text.primary', fontWeight: 500, lineHeight: 1.2 }}>
                  {detectionStatus || 'Analyzing...'}
                </Typography>
                <Typography variant="caption" sx={{
                  color: 'text.secondary',
                  fontFamily: 'monospace',
                  fontSize: 10,
                  lineHeight: 1.2,
                }}>
                  {formatElapsed(elapsedTime)}
                </Typography>
              </Stack>
            </Stack>
          ) : detectionResult ? (
            // Show success/result message
            <Stack direction="row" alignItems="center" spacing={1} sx={{
              bgcolor: detectionResult.ralliesCount > 0 ? 'success.main' : 'warning.main',
              color: 'white',
              px: 1.5,
              py: 0.5,
              borderRadius: 2,
            }}>
              <Typography variant="caption" sx={{ fontWeight: 500 }}>
                {detectionStatus}
              </Typography>
            </Stack>
          ) : detectionError ? (
            <Stack direction="row" alignItems="center" spacing={0.5}>
              <Typography variant="caption" sx={{ color: 'warning.main', maxWidth: 200 }}>
                {detectionError}
              </Typography>
              <Button size="small" onClick={() => setDetectionError(null)} sx={{ minWidth: 'auto', px: 1 }}>
                OK
              </Button>
            </Stack>
          ) : videoDetectionStatus === 'DETECTED' || isLocked ? (
            // Video already detected or confirmed - don't show detect button
            null
          ) : (
            <Tooltip title="Use ML to automatically detect rallies in this video">
              <Button
                size="small"
                variant="outlined"
                startIcon={<AutoFixHighIcon sx={{ fontSize: 16 }} />}
                onClick={handleStartDetection}
                disabled={!activeMatchId}
                sx={{
                  fontSize: 12,
                  py: 0.25,
                  textTransform: 'none',
                }}
              >
                Detect Rallies
              </Button>
            </Tooltip>
          )}
          <Typography variant="caption" sx={{ color: 'text.secondary', ml: 1 }}>
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
            background: '#0F1116 !important',
          },
          // Action container is INVISIBLE - we render our own styled Box in getActionRender
          // This allows us to control visual position during drag (clamping to keyframes)
          '& .timeline-editor-action': {
            background: 'transparent !important',
            border: 'none !important',
            boxShadow: 'none !important',
            cursor: 'pointer',
            overflow: 'visible !important',
            zIndex: '10 !important',
          },
          // Resize handles on edges (disabled when locked or in camera edit mode)
          '& .timeline-editor-action .timeline-editor-action-left-stretch, & .timeline-editor-action .timeline-editor-action-right-stretch': {
            cursor: (isLocked || isInCameraEditMode) ? 'default !important' : 'ew-resize !important',
            pointerEvents: (isLocked || isInCameraEditMode) ? 'none !important' : 'auto !important',
            width: '10px !important',
            overflow: 'visible !important',
            background: 'transparent !important',
            zIndex: '50 !important',
          },
          '& .timeline-editor-action-selected': {
            background: 'transparent !important',
            border: 'none !important',
            boxShadow: 'none !important',
            zIndex: '20 !important',
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
        {/* Buffer progress bar - shows which parts of video are loaded */}
        {duration > 0 && (
          <Box
            sx={{
              position: 'absolute',
              left: 10, // matches startLeft
              right: 10,
              top: 32, // below the time markers
              height: 3,
              bgcolor: 'rgba(255,255,255,0.08)',
              borderRadius: 1,
              overflow: 'hidden',
              zIndex: 5,
            }}
          >
            {bufferedRanges.map((range, index) => {
              const startPercent = (range.start / duration) * 100;
              const widthPercent = ((range.end - range.start) / duration) * 100;
              return (
                <Box
                  key={index}
                  sx={{
                    position: 'absolute',
                    left: `${startPercent}%`,
                    width: `${widthPercent}%`,
                    height: '100%',
                    bgcolor: 'rgba(255,255,255,0.25)',
                    borderRadius: 1,
                    transition: 'width 0.3s ease-out',
                  }}
                />
              );
            })}
            {/* Played progress overlay */}
            <Box
              sx={{
                position: 'absolute',
                left: 0,
                width: `${(currentTime / duration) * 100}%`,
                height: '100%',
                bgcolor: 'primary.main',
                opacity: 0.6,
                borderRadius: 1,
              }}
            />
          </Box>
        )}

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
          minScaleCount={Math.max(1, duration > 0 ? Math.ceil(duration / scale) + 2 : 1)}
          maxScaleCount={Math.max(100, duration > 0 ? Math.ceil(duration / scale) + 10 : 100)}
          onScroll={handleScroll}
          getScaleRender={getScaleRender}
          getActionRender={(action) => {
            const rally = rallies?.find((s) => s.id === action.id);

            // Get keyframes for this rally
            const cameraEdit = cameraEdits[action.id];
            const activeKeyframes = cameraEdit
              ? cameraEdit.keyframes[cameraEdit.aspectRatio] ?? []
              : [];
            // Get ALL keyframes (both aspect ratios) for resize blocking display
            const allKeyframes = cameraEdit
              ? [...(cameraEdit.keyframes.ORIGINAL ?? []), ...(cameraEdit.keyframes.VERTICAL ?? [])]
              : [];

            // Check if this rally has any camera edits
            const hasCameraEdits = cameraEdit && (
              cameraEdit.aspectRatio === 'VERTICAL' ||
              allKeyframes.length > 0
            );

            // Is THIS specific rally being camera-edited?
            const isThisRallyInCameraEditMode = selectedRallyId === action.id && isCameraTabActive;

            // Show keyframes when resize is blocked by them (to explain why)
            const isResizeBlockedByKeyframes = keyframeBlockedRallyId === action.id;

            // Determine what to render
            // In camera edit mode: show only active aspect ratio keyframes
            // When resize is blocked: show ALL keyframes (any could be blocking)
            const keyframesToShow = isResizeBlockedByKeyframes ? allKeyframes : activeKeyframes;
            const showKeyframeDots = (isThisRallyInCameraEditMode || isResizeBlockedByKeyframes) && keyframesToShow.length > 0;
            const showCameraIndicator = hasCameraEdits && !isThisRallyInCameraEditMode && !isResizeBlockedByKeyframes;

            // Calculate original rally duration for keyframe positioning
            const originalDuration = rally ? rally.end_time - rally.start_time : 1;
            const currentDuration = action.end - action.start;

            // Calculate position offset: library renders at action.start, we want pendingResize.start
            const pending = pendingResize?.rallyId === action.id ? pendingResize : null;
            const pixelsPerSecond = SCALE_WIDTH / scale;
            const offsetX = pending ? (pending.start - action.start) * pixelsPerSecond : 0;
            const visualWidth = pending
              ? (pending.end - pending.start) * pixelsPerSecond
              : currentDuration * pixelsPerSecond;

            // Determine if this action is selected
            const isSelected = action.id === selectedRallyId;
            const hasSelection = selectedRallyId !== null;

            return (
              <Box
                sx={{
                  width: '100%',
                  height: '100%',
                  position: 'relative',
                  overflow: 'visible',
                }}
              >
                {/* Visual segment - positioned at clamped position */}
                <Box
                  sx={{
                    position: 'absolute',
                    left: offsetX,
                    top: 0,
                    width: visualWidth,
                    height: '100%',
                    borderRadius: '4px',
                    cursor: 'pointer',
                    transition: pending ? 'none' : 'all 0.2s ease', // No transition during drag
                    // Styling based on selected state
                    background: isSelected
                      ? 'linear-gradient(180deg, #FF6B4A 0%, #E55235 100%)'
                      : 'linear-gradient(180deg, #3B82F6 0%, #2563EB 100%)',
                    border: isSelected
                      ? '1px solid rgba(255,255,255,0.3)'
                      : '1px solid rgba(255,255,255,0.1)',
                    boxShadow: isSelected
                      ? '0 0 0 2px rgba(255,107,74,0.5), 0 4px 12px rgba(255,107,74,0.4)'
                      : '0 2px 4px rgba(0,0,0,0.2)',
                    // Dim non-selected when there's a selection
                    ...(!isSelected && hasSelection && {
                      opacity: 0.4,
                      filter: 'saturate(0.5)',
                    }),
                    ...(isSelected && {
                      zIndex: 10,
                    }),
                    '&:hover': isSelected ? {
                      background: 'linear-gradient(180deg, #FF8A6F 0%, #FF6B4A 100%)',
                      boxShadow: '0 0 0 2px rgba(255,107,74,0.6), 0 6px 16px rgba(255,107,74,0.5)',
                    } : {
                      background: 'linear-gradient(180deg, #60A5FA 0%, #3B82F6 100%)',
                      opacity: 0.7,
                      boxShadow: '0 4px 8px rgba(59, 130, 246, 0.3)',
                    },
                  }}
                >
                {/* Keyframe tick marks */}
                {showKeyframeDots && keyframesToShow.map((kf) => {
                  // Calculate absolute time of keyframe based on original rally bounds
                  const kfAbsTime = rally
                    ? rally.start_time + kf.timeOffset * originalDuration
                    : action.start + kf.timeOffset * currentDuration;
                  // Calculate position relative to the VISUAL segment bounds (pending or rally)
                  const visualStart = pending?.start ?? rally?.start_time ?? action.start;
                  const visualEnd = pending?.end ?? rally?.end_time ?? action.end;
                  const visualDuration = visualEnd - visualStart;
                  const compensatedLeft = ((kfAbsTime - visualStart) / visualDuration) * 100;

                  return (
                  <Box
                    key={kf.id}
                    onClick={(e) => {
                      e.stopPropagation();
                      // Select the rally and keyframe
                      selectRally(action.id);
                      selectKeyframe(kf.id);
                      // Switch to camera edit mode
                      setIsCameraTabActive(true);
                      // Seek to keyframe position
                      if (rally) {
                        const keyframeTime = rally.start_time + kf.timeOffset * (rally.end_time - rally.start_time);
                        seek(keyframeTime);
                      }
                    }}
                    sx={{
                      position: 'absolute',
                      left: `${compensatedLeft}%`,
                      top: 0,
                      bottom: 0,
                      width: 14,
                      transform: 'translateX(-50%)',
                      cursor: 'pointer',
                      zIndex: 10,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      '&:hover .kf-dot': {
                        transform: 'scale(1.4)',
                        bgcolor: '#FFD700',
                      },
                      '&:hover .kf-line': {
                        bgcolor: 'rgba(255,255,255,0.6)',
                      },
                    }}
                  >
                    {/* Vertical line */}
                    <Box
                      className="kf-line"
                      sx={{
                        position: 'absolute',
                        top: 0,
                        bottom: 0,
                        left: '50%',
                        width: 1,
                        transform: 'translateX(-50%)',
                        bgcolor: 'rgba(255,255,255,0.35)',
                        transition: 'background-color 0.15s',
                      }}
                    />
                    {/* Centered clickable dot */}
                    <Box
                      className="kf-dot"
                      sx={{
                        position: 'relative',
                        width: 10,
                        height: 10,
                        borderRadius: '50%',
                        bgcolor: selectedKeyframeId === kf.id ? '#FFD700' : '#FFA500',
                        border: '2px solid',
                        borderColor: selectedKeyframeId === kf.id ? '#FFF' : 'rgba(0,0,0,0.4)',
                        boxShadow: '0 1px 3px rgba(0,0,0,0.4)',
                        transition: 'transform 0.15s, background-color 0.15s',
                      }}
                    />
                  </Box>
                  );
                })}

                {/* Camera indicator badge */}
                {showCameraIndicator && (
                  <Box
                    sx={{
                      position: 'absolute',
                      top: 4,
                      right: 4,
                      zIndex: 15,
                      bgcolor: 'rgba(0,0,0,0.5)',
                      borderRadius: '4px',
                      p: 0.25,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                    }}
                  >
                    <CameraAltIcon sx={{ fontSize: 12, color: 'rgba(255,255,255,0.85)' }} />
                  </Box>
                )}

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
                    position: 'relative',
                    zIndex: 1,
                  }}
                >
                  <Typography
                    variant="caption"
                    sx={{
                      fontWeight: 600,
                      fontSize: 12,
                      textShadow: '0 1px 2px rgba(0,0,0,0.5)',
                    }}
                  >
                    {(() => {
                      const sortedRallies = [...(rallies ?? [])].sort((a, b) => a.start_time - b.start_time);
                      const index = sortedRallies.findIndex(r => r.id === action.id);
                      return index >= 0 ? index + 1 : '';
                    })()}
                  </Typography>
                </Box>


                {/* Highlight color bar on bottom edge */}
                {(() => {
                  const rallyHighlights = getHighlightsForRally(action.id);
                  if (rallyHighlights.length === 0) return null;

                  const colors = rallyHighlights.map(h => h.color);
                  // Create horizontal gradient with equal stops for each color
                  const gradientStops = colors.map((color, i) => {
                    const start = (i / colors.length) * 100;
                    const end = ((i + 1) / colors.length) * 100;
                    return `${color} ${start}%, ${color} ${end}%`;
                  }).join(', ');

                  return (
                    <Box
                      sx={{
                        position: 'absolute',
                        left: -1,
                        right: -1,
                        bottom: -1,
                        height: 5,
                        background: colors.length === 1
                          ? colors[0]
                          : `linear-gradient(90deg, ${gradientStops})`,
                        borderRadius: '0 0 4px 4px',
                        border: '1px solid rgba(255,255,255,0.1)',
                        borderTop: 'none',
                      }}
                    />
                  );
                })()}
                </Box>
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
                bgcolor: '#FF6B4A',
                boxShadow: '0 0 8px rgba(255, 107, 74, 0.6)',
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
                bgcolor: '#FF6B4A',
                borderRadius: '3px 3px 0 0',
                boxShadow: '0 2px 6px rgba(255, 107, 74, 0.4)',
                cursor: 'ew-resize',
                pointerEvents: 'auto',
                '&:hover': {
                  bgcolor: '#FF8A6F',
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
                  borderTop: '6px solid #FF6B4A',
                },
              }}
              onMouseDown={(e) => {
                e.preventDefault();
                setIsDraggingCursor(true);
                const startX = e.clientX;
                const startTime = currentTime;
                const pixelsPerSecond = SCALE_WIDTH / scale;
                // Get rally bounds for camera edit mode constraint
                const rally = isInCameraEditMode ? getSelectedRally() : null;

                const handleMouseMove = (moveEvent: MouseEvent) => {
                  const deltaX = moveEvent.clientX - startX;
                  const deltaTime = deltaX / pixelsPerSecond;
                  // In camera edit mode, constrain to rally bounds
                  const minTime = rally ? rally.start_time : 0;
                  const maxTime = rally ? rally.end_time : duration;
                  const newTime = Math.max(minTime, Math.min(maxTime, startTime + deltaTime));
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
                  bgcolor: '#FF6B4A',
                  color: 'white',
                  border: '2px solid white',
                  boxShadow: '0 2px 8px rgba(255, 107, 74, 0.5)',
                  '&:hover': {
                    bgcolor: '#FF8A6F',
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

        {/* Delete button overlay for selected rally (hidden when locked or in camera edit mode) */}
        {selectedRallyId && selectedRallyPosition && selectedRallyPosition.center > 0 && !isLocked && !isInCameraEditMode && (
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
                  isRallyInTargetHighlight
                    ? `Remove from ${targetHighlight?.name}`
                    : targetHighlight
                      ? `Add to ${targetHighlight.name}`
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
                      bgcolor: isRallyInTargetHighlight
                        ? targetHighlight?.color || '#FFE66D'
                        : 'rgba(20,20,20,0.9)',
                      color: isRallyInTargetHighlight
                        ? 'rgba(0,0,0,0.8)'
                        : targetHighlight
                          ? targetHighlight.color
                          : 'rgba(255,255,255,0.8)',
                      border: '1px solid rgba(255,255,255,0.15)',
                      boxShadow: '0 2px 8px rgba(0,0,0,0.4)',
                      '&:hover': {
                        bgcolor: isRallyInTargetHighlight
                          ? 'rgba(20,20,20,0.9)'
                          : targetHighlight
                            ? targetHighlight.color
                            : '#FFE66D',
                        color: isRallyInTargetHighlight
                          ? targetHighlight?.color || 'rgba(255,255,255,0.8)'
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

        {/* Merge buttons between close rallies (gap <= 3s) */}
        {closeRallyPairs.map(({ first, second, position }) => (
          position > 0 && position < (containerWidth || 9999) && (
            <Tooltip key={`merge-${first.id}-${second.id}`} title="Merge rallies" arrow>
              <IconButton
                size="small"
                onClick={(e) => {
                  e.stopPropagation();
                  mergeRallies(first.id, second.id);
                }}
                sx={{
                  position: 'absolute',
                  left: position,
                  top: 'calc(50% + 20px)',
                  transform: 'translate(-50%, -50%)',
                  zIndex: 40,
                  width: 24,
                  height: 24,
                  bgcolor: 'warning.main',
                  color: 'warning.contrastText',
                  border: '1px solid rgba(255,255,255,0.2)',
                  boxShadow: '0 2px 6px rgba(0,0,0,0.3)',
                  '&:hover': {
                    bgcolor: 'warning.dark',
                    transform: 'translate(-50%, -50%) scale(1.1)',
                  },
                  transition: 'all 0.15s ease',
                }}
              >
                <MergeTypeIcon sx={{ fontSize: 14 }} />
              </IconButton>
            </Tooltip>
          )
        ))}
      </Box>
    </Box>
  );
}
