'use client';

import { useRef, useEffect, useCallback, useState, useMemo } from 'react';
import { Box, CircularProgress, Typography } from '@mui/material';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import { usePlayerStore } from '@/stores/playerStore';
import { useEditorStore } from '@/stores/editorStore';
import { useUploadStore } from '@/stores/uploadStore';
import { useCameraStore } from '@/stores/cameraStore';
import { calculateVideoTransform, getCameraStateWithHandheld, interpolateCameraState } from '@/utils/cameraInterpolation';
import { DEFAULT_CAMERA_STATE, DEFAULT_GLOBAL_CAMERA, GlobalCameraSettings, CameraState, CameraKeyframe } from '@/types/camera';
import { designTokens } from '@/app/theme';
import { CameraOverlay } from './CameraOverlay';
import { BallTrackingDebugOverlay } from './BallTrackingDebugOverlay';
import { RotationGridOverlay } from './RotationGridOverlay';
import { CropMaskOverlay } from './CropMaskOverlay';

export function VideoPlayer() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const videoContainerRef = useRef<HTMLDivElement>(null);
  const fullscreenContainerRef = useRef<HTMLDivElement>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isBuffering, setIsBuffering] = useState(false);
  const [bufferProgress, setBufferProgress] = useState(0);
  // Camera time for seeking (when paused, we may need to update camera position)
  const [, setCameraTime] = useState(0);

  // Ref for RAF loop - stores camera data to avoid effect restarts
  const rafDataRef = useRef<{
    rallyStart: number;
    rallyEnd: number;
    keyframes: CameraKeyframe[];
    aspectRatio: 'ORIGINAL' | 'VERTICAL';
    globalSettings: GlobalCameraSettings;
  } | null>(null);
  // Ref for transform wrapper
  const transformWrapperRef = useRef<HTMLDivElement>(null);

  const videoUrl = useEditorStore((state) => state.videoUrl);
  const posterUrl = useEditorStore((state) => state.posterUrl);
  const proxyUrl = useEditorStore((state) => state.proxyUrl);
  const activeMatchId = useEditorStore((state) => state.activeMatchId);
  const setActiveMatch = useEditorStore((state) => state.setActiveMatch);
  const isLoadingSession = useEditorStore((state) => state.isLoadingSession);

  // Get local blob URL if available (from recent upload)
  // Subscribe to localVideoUrls directly so component re-renders when it changes
  const localVideoUrls = useUploadStore((state) => state.localVideoUrls);
  const localVideoUrl = activeMatchId ? localVideoUrls.get(activeMatchId) : undefined;

  // Priority: local blob (instant) > proxy (fast) > full video
  const effectiveVideoUrl = localVideoUrl || proxyUrl || videoUrl;

  const isPlaying = usePlayerStore((state) => state.isPlaying);
  const seekTo = usePlayerStore((state) => state.seekTo);
  const clearSeek = usePlayerStore((state) => state.clearSeek);
  const setCurrentTime = usePlayerStore((state) => state.setCurrentTime);
  const setDuration = usePlayerStore((state) => state.setDuration);
  const setReady = usePlayerStore((state) => state.setReady);
  const setBufferedRanges = usePlayerStore((state) => state.setBufferedRanges);
  const applyCameraEdits = usePlayerStore((state) => state.applyCameraEdits);
  const currentTime = usePlayerStore((state) => state.currentTime);
  const playbackRate = usePlayerStore((state) => state.playbackRate);
  const setFullscreen = usePlayerStore((state) => state.setFullscreen);
  const registerToggleFullscreen = usePlayerStore((state) => state.registerToggleFullscreen);

  // Camera state
  const cameraEdits = useCameraStore((state) => state.cameraEdits);
  const globalCameraSettings = useCameraStore((state) => state.globalCameraSettings);
  const dragPosition = useCameraStore((state) => state.dragPosition);
  const debugBallPositions = useCameraStore((state) => state.debugBallPositions);
  const debugFrameCount = useCameraStore((state) => state.debugFrameCount);
  const debugRallyId = useCameraStore((state) => state.debugRallyId);
  const isAdjustingRotation = useCameraStore((state) => state.isAdjustingRotation);

  // Get rallies from editor store
  const rallies = useEditorStore((state) => state.rallies);
  const selectedRallyId = useEditorStore((state) => state.selectedRallyId);
  const isCameraTabActive = useEditorStore((state) => state.isCameraTabActive);
  const isRecordingRally = useEditorStore((state) => state.isRecordingRally);

  // Find current rally based on playhead position
  const currentRally = useMemo(() => {
    // If a rally is selected, use it for camera preview
    if (selectedRallyId) {
      return rallies.find((r) => r.id === selectedRallyId) ?? null;
    }
    // Otherwise find rally at current time
    return rallies.find((r) => currentTime >= r.start_time && currentTime <= r.end_time) ?? null;
  }, [rallies, currentTime, selectedRallyId]);

  // Get camera edit for current rally
  const currentCameraEdit = useMemo(() => {
    if (!currentRally) return null;
    return cameraEdits[currentRally.id] ?? null;
  }, [currentRally, cameraEdits]);

  // Get global settings for current video (use activeMatchId directly)
  const currentGlobalSettings = useMemo((): GlobalCameraSettings => {
    if (!activeMatchId) return DEFAULT_GLOBAL_CAMERA;
    return globalCameraSettings[activeMatchId] ?? DEFAULT_GLOBAL_CAMERA;
  }, [activeMatchId, globalCameraSettings]);

  // Helper to combine global settings with per-rally state
  // Rally-specific values OVERRIDE global values when explicitly set (not default)
  const combineWithGlobal = useCallback((rallyState: CameraState): CameraState => {
    const global = currentGlobalSettings;
    // If rally has non-default value, use it; otherwise use global
    const hasRallyZoom = rallyState.zoom !== 1.0;
    const hasRallyPositionX = rallyState.positionX !== 0.5;
    const hasRallyPositionY = rallyState.positionY !== 0.5;
    const hasRallyRotation = rallyState.rotation !== 0;

    return {
      zoom: hasRallyZoom ? rallyState.zoom : global.zoom,
      positionX: hasRallyPositionX ? rallyState.positionX : global.positionX,
      positionY: hasRallyPositionY ? rallyState.positionY : global.positionY,
      rotation: hasRallyRotation ? rallyState.rotation : global.rotation,
    };
  }, [currentGlobalSettings]);

  // Check if camera edit has keyframes for the active aspect ratio
  const hasCameraKeyframes = currentCameraEdit &&
    (currentCameraEdit.keyframes[currentCameraEdit.aspectRatio]?.length ?? 0) > 0;

  // Check if we have any camera adjustments (global or per-rally)
  const hasGlobalCameraSettings = currentGlobalSettings.zoom !== 1.0 ||
    currentGlobalSettings.positionX !== 0.5 ||
    currentGlobalSettings.positionY !== 0.5 ||
    currentGlobalSettings.rotation !== 0;

  // Determine if camera should be applied - only when preview toggle is ON
  const shouldApplyCamera = applyCameraEdits;

  // Track last camera state for when RAF is controlling
  const lastCameraStateRef = useRef<CameraState>(DEFAULT_CAMERA_STATE);

  // Calculate camera state and video transform style
  const { videoTransformStyle, cameraState: currentCameraState } = useMemo(() => {
    // During playback with keyframes, RAF controls the transform directly via DOM manipulation
    // Return empty styles so React doesn't interfere - RAF updates lastCameraStateRef
    if (isPlaying && hasCameraKeyframes) {
      return { videoTransformStyle: {}, cameraState: DEFAULT_CAMERA_STATE };
    }

    // If no camera preview and no global settings, return defaults
    if (!shouldApplyCamera) {
      return { videoTransformStyle: {}, cameraState: DEFAULT_CAMERA_STATE };
    }

    const aspectRatio = currentCameraEdit?.aspectRatio ?? 'ORIGINAL';

    // No current rally - apply global settings only (entire video)
    if (!currentRally) {
      if (!hasGlobalCameraSettings && !dragPosition) {
        return { videoTransformStyle: {}, cameraState: DEFAULT_CAMERA_STATE };
      }
      const globalState = combineWithGlobal(DEFAULT_CAMERA_STATE);
      // Apply drag position for live preview in global mode
      const effectiveState = dragPosition
        ? { ...globalState, positionX: dragPosition.x, positionY: dragPosition.y }
        : globalState;
      return {
        videoTransformStyle: calculateVideoTransform(effectiveState, 'ORIGINAL'),
        cameraState: effectiveState,
      };
    }

    // If no keyframes but we have global settings, apply global settings only
    if (!hasCameraKeyframes) {
      const baseState = combineWithGlobal(DEFAULT_CAMERA_STATE);
      return {
        videoTransformStyle: calculateVideoTransform(baseState, aspectRatio),
        cameraState: baseState,
      };
    }

    // Use currentTime from playerStore for camera position calculation
    const effectiveTime = currentTime;

    // Calculate time offset within the rally (0-1)
    const rallyDuration = currentRally.end_time - currentRally.start_time;
    const timeWithinRally = rallyDuration > 0
      ? (effectiveTime - currentRally.start_time) / rallyDuration
      : 0;
    const clampedOffset = Math.max(0, Math.min(1, timeWithinRally));

    // Get keyframes for the active aspect ratio
    const keyframes = currentCameraEdit?.keyframes[aspectRatio] ?? [];

    // Get interpolated camera state (handheld disabled for smooth transitions)
    const rawCameraState = getCameraStateWithHandheld(
      keyframes,
      clampedOffset,
      effectiveTime,
      currentRally.id,
      'OFF'
    );

    // Combine with global settings
    const cameraState = combineWithGlobal(rawCameraState);

    // If dragging, override position with drag position for live preview
    // Note: drag position is in combined space, so we apply it after global settings
    const effectiveState = dragPosition
      ? { ...cameraState, positionX: dragPosition.x, positionY: dragPosition.y }
      : cameraState;

    // Calculate CSS transform
    const transform = calculateVideoTransform(effectiveState, aspectRatio);

    return {
      videoTransformStyle: transform,
      cameraState: effectiveState,
    };
  }, [shouldApplyCamera, currentRally, hasCameraKeyframes, hasGlobalCameraSettings, currentCameraEdit, currentTime, dragPosition, combineWithGlobal, isPlaying]);

  // Get container aspect ratio - show aspect ratio even without keyframes when preview is on
  const containerAspectRatio = useMemo(() => {
    if (!shouldApplyCamera || !currentRally) {
      return '16/9';
    }
    const aspectRatio = currentCameraEdit?.aspectRatio ?? 'ORIGINAL';
    return aspectRatio === 'VERTICAL' ? '9/16' : '16/9';
  }, [shouldApplyCamera, currentRally, currentCameraEdit]);

  // Play/pause based on isPlaying state
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    if (isPlaying) {
      video.play().catch(() => {});
    } else {
      video.pause();
    }
  }, [isPlaying]);

  // Apply playback rate to video element
  useEffect(() => {
    if (videoRef.current) {
      videoRef.current.playbackRate = playbackRate;
    }
  }, [playbackRate]);

  // Update RAF data ref when camera data changes
  useEffect(() => {
    if (!currentRally || !currentCameraEdit) {
      rafDataRef.current = null;
      return;
    }
    const aspectRatio = currentCameraEdit.aspectRatio;
    rafDataRef.current = {
      rallyStart: currentRally.start_time,
      rallyEnd: currentRally.end_time,
      keyframes: currentCameraEdit.keyframes[aspectRatio] ?? [],
      aspectRatio,
      globalSettings: currentGlobalSettings,
    };
  }, [currentRally, currentCameraEdit, currentGlobalSettings]);

  // Video frame callback for smooth camera animation synced to video frames
  useEffect(() => {
    if (!isPlaying || !shouldApplyCamera || !hasCameraKeyframes) {
      return;
    }

    const video = videoRef.current;
    if (!video) return;

    let callbackId: number;
    const hasRVFC = 'requestVideoFrameCallback' in video;

    const updateTransform = (now: DOMHighResTimeStamp, metadata?: { mediaTime: number }) => {
      const data = rafDataRef.current;
      if (!data) {
        if (hasRVFC) {
          callbackId = (video as HTMLVideoElement & { requestVideoFrameCallback: (cb: (n: number, m: { mediaTime: number }) => void) => number }).requestVideoFrameCallback(updateTransform);
        } else {
          callbackId = requestAnimationFrame(() => updateTransform(performance.now()));
        }
        return;
      }

      const { rallyStart, rallyEnd, keyframes, aspectRatio, globalSettings } = data;
      const videoTime = metadata?.mediaTime ?? video.currentTime;

      const duration = rallyEnd - rallyStart;
      const offset = duration > 0 ? (videoTime - rallyStart) / duration : 0;
      const clampedOffset = Math.max(0, Math.min(1, offset));

      // Pure interpolation - keyframe easing provides the smoothness
      const raw = interpolateCameraState(keyframes, clampedOffset);

      // Combine with global settings
      const state: CameraState = {
        zoom: raw.zoom !== 1.0 ? raw.zoom : globalSettings.zoom,
        positionX: raw.positionX !== 0.5 ? raw.positionX : globalSettings.positionX,
        positionY: raw.positionY !== 0.5 ? raw.positionY : globalSettings.positionY,
        rotation: raw.rotation !== 0 ? raw.rotation : globalSettings.rotation,
      };

      const transform = calculateVideoTransform(state, aspectRatio);

      // Update the ref so React has the correct state when playback stops
      lastCameraStateRef.current = state;

      // Apply directly to transform wrapper
      const element = transformWrapperRef.current || video;
      element.style.transform = transform.transform as string;
      element.style.transformOrigin = transform.transformOrigin as string;

      // Schedule next update
      if (hasRVFC) {
        callbackId = (video as HTMLVideoElement & { requestVideoFrameCallback: (cb: (n: number, m: { mediaTime: number }) => void) => number }).requestVideoFrameCallback(updateTransform);
      } else {
        callbackId = requestAnimationFrame(() => updateTransform(performance.now()));
      }
    };

    // Start the animation loop
    if (hasRVFC) {
      callbackId = (video as HTMLVideoElement & { requestVideoFrameCallback: (cb: (n: number, m: { mediaTime: number }) => void) => number }).requestVideoFrameCallback(updateTransform);
    } else {
      callbackId = requestAnimationFrame(() => updateTransform(performance.now()));
    }

    return () => {
      if (hasRVFC && 'cancelVideoFrameCallback' in video) {
        (video as HTMLVideoElement & { cancelVideoFrameCallback: (id: number) => void }).cancelVideoFrameCallback(callbackId);
      } else {
        cancelAnimationFrame(callbackId);
      }
      // Sync currentTime when animation stops (e.g., user pauses)
      setCurrentTime(video.currentTime);
    };
  }, [isPlaying, shouldApplyCamera, hasCameraKeyframes, setCurrentTime]);

  // Handle manual seek requests
  useEffect(() => {
    if (seekTo !== null && videoRef.current) {
      videoRef.current.currentTime = seekTo;
      // Update cameraTime immediately for smooth preview when paused
      setCameraTime(seekTo); // eslint-disable-line react-hooks/set-state-in-effect -- intentional: syncing camera time with seek
      // Also update currentTime - ensures immediate sync since handleTimeUpdate skips updates during camera animation
      setCurrentTime(seekTo);
      clearSeek();
      // Resume playback if we're supposed to be playing
      if (isPlaying) {
        videoRef.current.play().catch(() => {});
      }
    }
  }, [seekTo, clearSeek, isPlaying, setCurrentTime]);

  // Handle seeked event - update camera time when scrubbing
  const handleSeeked = useCallback(() => {
    const video = videoRef.current;
    if (video && !isPlaying) {
      setCameraTime(video.currentTime);
    }
  }, [isPlaying]);

  // Track if we're in the middle of a match switch
  const switchingMatchRef = useRef(false);

  // Helper: check for highlight playback rally transitions
  const checkHighlightTransition = useCallback((video: HTMLVideoElement) => {
    if (switchingMatchRef.current) return;
    const playerState = usePlayerStore.getState();
    const editorState = useEditorStore.getState();
    if (!playerState.playingHighlightId) return;

    const currentRally = playerState.getCurrentPlaylistRally();
    if (currentRally && currentRally.matchId === editorState.activeMatchId && video.currentTime >= currentRally.end_time) {
      const nextRally = playerState.advanceHighlightPlayback();
      if (nextRally) {
        if (nextRally.matchId === editorState.activeMatchId) {
          video.currentTime = nextRally.start_time;
        } else {
          switchingMatchRef.current = true;
          setActiveMatch(nextRally.matchId);
        }
      } else {
        playerState.stopHighlightPlayback();
      }
    }
  }, [setActiveMatch]);

  // Get playOnlyRallies setting for skip dead time logic
  const playOnlyRallies = usePlayerStore((state) => state.playOnlyRallies);
  const seek = usePlayerStore((state) => state.seek);

  // Time update handler - check for rally end during highlight playback
  const handleTimeUpdate = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;

    const videoTime = video.currentTime;

    // During camera animation, skip React state updates to prevent jitter
    // Handle skip-dead-time logic here using video.currentTime directly
    if (isPlaying && hasCameraKeyframes) {
      // Check if we've exited the rally into dead time
      const inAnyRally = rallies.some(
        (r) => videoTime >= r.start_time && videoTime <= r.end_time
      );

      if (!inAnyRally) {
        if (playOnlyRallies) {
          // In dead time with "rallies only" enabled - jump to next rally
          const sorted = [...rallies].sort((a, b) => a.start_time - b.start_time);
          const nextRally = sorted.find((r) => r.start_time > videoTime);
          if (nextRally) {
            seek(nextRally.start_time);
          }
        } else {
          // Exited rally into dead time - update currentTime to allow
          // hasCameraKeyframes to recalculate (breaks circular dependency)
          setCurrentTime(videoTime);
        }
      }

      checkHighlightTransition(video);
      return;
    }

    setCurrentTime(videoTime);
    checkHighlightTransition(video);
  }, [setCurrentTime, isPlaying, hasCameraKeyframes, rallies, playOnlyRallies, seek, checkHighlightTransition]);

  // Handle video metadata loaded
  const handleLoadedMetadata = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;

    setDuration(video.duration);
    setReady(true);
    setIsLoading(false);

    // Read fresh values from stores to avoid stale closures
    const playerState = usePlayerStore.getState();
    const editorState = useEditorStore.getState();

    // If we just loaded a new video during highlight playback, seek to current rally
    if (switchingMatchRef.current && playerState.playingHighlightId) {
      const currentRally = playerState.getCurrentPlaylistRally();
      if (currentRally && currentRally.matchId === editorState.activeMatchId) {
        switchingMatchRef.current = false;
        video.currentTime = currentRally.start_time;
        video.play().catch(() => {});
      }
    } else {
      switchingMatchRef.current = false;
    }
  }, [setDuration, setReady]);

  const handleLoadStart = useCallback(() => {
    setIsLoading(true);
    setBufferProgress(0);
    setReady(false);
  }, [setReady]);

  // Track video buffering progress
  const handleProgress = useCallback(() => {
    const video = videoRef.current;
    if (!video || video.buffered.length === 0) return;

    const bufferedEnd = video.buffered.end(video.buffered.length - 1);
    const duration = video.duration;
    if (duration > 0) {
      setBufferProgress(Math.round((bufferedEnd / duration) * 100));
    }

    // Update buffered ranges in store for timeline visualization
    const ranges: { start: number; end: number }[] = [];
    for (let i = 0; i < video.buffered.length; i++) {
      ranges.push({
        start: video.buffered.start(i),
        end: video.buffered.end(i),
      });
    }
    setBufferedRanges(ranges);
  }, [setBufferedRanges]);

  // Handle buffering stalls during playback
  const handleWaiting = useCallback(() => {
    setIsBuffering(true);
  }, []);

  const handleCanPlay = useCallback(() => {
    setIsBuffering(false);
  }, []);

  // Fullscreen toggle function
  const toggleFullscreen = useCallback(() => {
    if (document.fullscreenElement) {
      document.exitFullscreen();
    } else {
      fullscreenContainerRef.current?.requestFullscreen();
    }
  }, []);

  // Register fullscreen toggle with store and sync fullscreen state
  useEffect(() => {
    registerToggleFullscreen(toggleFullscreen);

    const handleFullscreenChange = () => {
      setFullscreen(!!document.fullscreenElement);
    };

    document.addEventListener('fullscreenchange', handleFullscreenChange);
    return () => {
      document.removeEventListener('fullscreenchange', handleFullscreenChange);
    };
  }, [toggleFullscreen, registerToggleFullscreen, setFullscreen]);

  const handleEmptyStateClick = () => {
    // Trigger file upload via EditorHeader's exposed function
    const triggerFn = (window as unknown as { triggerVideoUpload?: () => void }).triggerVideoUpload;
    if (triggerFn) {
      triggerFn();
    }
  };

  // Show loading skeleton during session fetch
  if (isLoadingSession) {
    return (
      <Box
        sx={{
          width: '100%',
          aspectRatio: '16/9',
          bgcolor: '#000',
          borderRadius: 2,
          overflow: 'hidden',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          boxShadow: designTokens.colors.video.shadow,
        }}
      >
        <CircularProgress color="primary" size={48} />
      </Box>
    );
  }

  if (!effectiveVideoUrl) {
    return (
      <Box
        onClick={handleEmptyStateClick}
        sx={{
          width: '100%',
          aspectRatio: '16/9',
          bgcolor: designTokens.colors.surface[2],
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          borderRadius: 2,
          border: '2px dashed',
          borderColor: 'divider',
          color: 'text.secondary',
          gap: 2,
          cursor: 'pointer',
          transition: designTokens.transitions.normal,
          '&:hover': {
            borderColor: 'primary.main',
            bgcolor: designTokens.colors.surface[3],
          },
        }}
      >
        <Box
          sx={{
            width: 64,
            height: 64,
            borderRadius: '50%',
            bgcolor: 'rgba(255,255,255,0.05)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            mb: 1,
          }}
        >
          <UploadFileIcon sx={{ fontSize: 32, color: 'text.secondary' }} />
        </Box>
        <Typography variant="body1" sx={{ color: 'text.secondary', fontWeight: 500 }}>
          Click to upload a video
        </Typography>
        <Typography variant="caption" sx={{ color: 'text.disabled' }}>
          Supports MP4, MOV, and WebM
        </Typography>
      </Box>
    );
  }

  // Check if camera preview UI should be shown (must be in camera edit mode)
  const isCameraPreviewActive = isCameraTabActive && shouldApplyCamera && currentRally !== null;

  return (
    <Box
      ref={fullscreenContainerRef}
      sx={{
        width: '100%',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        bgcolor: '#000',
        borderRadius: 2,
        position: 'relative',
        boxShadow: designTokens.colors.video.shadow,
        // When in vertical mode, contain the vertical video within a 16:9 outer frame
        aspectRatio: '16/9',
        // Clip video content at player boundaries (important for 9:16 mode where video extends beyond inner container)
        overflow: 'hidden',
        // Premium border effect
        '&::before': {
          content: '""',
          position: 'absolute',
          inset: 0,
          borderRadius: 2,
          padding: '1px',
          background: 'linear-gradient(135deg, rgba(255,255,255,0.1) 0%, transparent 50%, rgba(255,255,255,0.05) 100%)',
          WebkitMask: 'linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0)',
          WebkitMaskComposite: 'xor',
          maskComposite: 'exclude',
          pointerEvents: 'none',
        },
        // Camera preview active indicator
        ...(isCameraPreviewActive && {
          '&::after': {
            content: '""',
            position: 'absolute',
            inset: -2,
            borderRadius: 2.5,
            border: '2px solid',
            borderColor: 'primary.main',
            opacity: 0.6,
            pointerEvents: 'none',
          },
        }),
      }}
    >
      {/* Loading overlay - initial load */}
      {isLoading && (
        <Box
          sx={{
            position: 'absolute',
            inset: 0,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            bgcolor: 'rgba(0, 0, 0, 0.7)',
            zIndex: 2,
            gap: 1.5,
          }}
        >
          <CircularProgress color="primary" />
          <Typography variant="body2" sx={{ color: 'text.secondary' }}>
            {bufferProgress > 0 && bufferProgress < 100
              ? `Loading video... ${bufferProgress}%`
              : 'Loading video...'}
          </Typography>
        </Box>
      )}

      {/* Buffering indicator - during playback stalls */}
      {isBuffering && !isLoading && (
        <Box
          sx={{
            position: 'absolute',
            top: 12,
            right: 12,
            display: 'flex',
            alignItems: 'center',
            gap: 1,
            bgcolor: 'rgba(0, 0, 0, 0.6)',
            borderRadius: 1,
            px: 1.5,
            py: 0.75,
            zIndex: 2,
          }}
        >
          <CircularProgress color="primary" size={16} thickness={4} />
          <Typography variant="caption" sx={{ color: 'text.secondary' }}>
            Buffering...
          </Typography>
        </Box>
      )}

      {/* Camera preview mode indicator */}
      {isCameraPreviewActive && (
        <Box
          sx={{
            position: 'absolute',
            top: 12,
            left: 12,
            display: 'flex',
            alignItems: 'center',
            gap: 0.75,
            bgcolor: 'rgba(0, 0, 0, 0.6)',
            borderRadius: 1,
            px: 1.5,
            py: 0.5,
            zIndex: 2,
          }}
        >
          <Box
            sx={{
              width: 8,
              height: 8,
              borderRadius: '50%',
              bgcolor: 'primary.main',
              animation: 'pulse 2s infinite',
              '@keyframes pulse': {
                '0%, 100%': { opacity: 1 },
                '50%': { opacity: 0.5 },
              },
            }}
          />
          <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 500 }}>
            {currentCameraEdit?.aspectRatio === 'VERTICAL' ? '9:16' : '16:9'} Preview
          </Typography>
        </Box>
      )}

      {/* Rally recording indicator */}
      {isRecordingRally && (
        <Box
          sx={{
            position: 'absolute',
            top: 12,
            right: isBuffering && !isLoading ? 100 : 12,
            display: 'flex',
            alignItems: 'center',
            gap: 0.75,
            bgcolor: '#d32f2f',
            borderRadius: 1,
            px: 1.5,
            py: 0.5,
            zIndex: 2,
            animation: 'recPulse 1s infinite',
            '@keyframes recPulse': {
              '0%, 100%': { opacity: 1 },
              '50%': { opacity: 0.7 },
            },
          }}
        >
          <Box
            sx={{
              width: 8,
              height: 8,
              borderRadius: '50%',
              bgcolor: 'white',
            }}
          />
          <Typography variant="caption" sx={{ color: 'white', fontWeight: 600 }}>
            REC
          </Typography>
        </Box>
      )}

      {/* Semi-transparent overlay showing excluded areas in 9:16 mode */}
      {containerAspectRatio === '9/16' && shouldApplyCamera && (
        <CropMaskOverlay aspectRatio="VERTICAL" />
      )}

      {/* Unified video container - single video element to prevent remounting on aspect ratio change */}
      <Box
        sx={{
          position: containerAspectRatio === '9/16' ? 'absolute' : 'relative',
          top: containerAspectRatio === '9/16' ? 0 : undefined,
          bottom: containerAspectRatio === '9/16' ? 0 : undefined,
          // For 9:16: Calculate width explicitly for 9:16 in a 16:9 parent
          // width = height * (9/16), but height = parentHeight
          // parentWidth = parentHeight * (16/9)
          // so width as % of parent = (9/16) / (16/9) = 31.64%
          width: containerAspectRatio === '9/16' ? `${(9/16) / (16/9) * 100}%` : '100%',
          height: '100%',
          left: containerAspectRatio === '9/16' ? '50%' : undefined,
          transform: containerAspectRatio === '9/16' ? 'translateX(-50%)' : undefined,
          // Allow video to extend beyond container in 9:16 mode so excluded areas are visible
          overflow: containerAspectRatio === '9/16' ? 'visible' : 'hidden',
        }}
      >
        <Box
          ref={videoContainerRef}
          sx={{
            position: containerAspectRatio === '9/16' ? 'absolute' : 'relative',
            inset: containerAspectRatio === '9/16' ? 0 : undefined,
            width: '100%',
            height: '100%',
            // Allow video to extend beyond container in 9:16 mode
            overflow: containerAspectRatio === '9/16' ? 'visible' : 'hidden',
          }}
        >
          <CameraOverlay containerRef={videoContainerRef} />
          <RotationGridOverlay isVisible={isAdjustingRotation} />
          {/* Ball tracking debug overlay */}
          {debugBallPositions && debugFrameCount && debugRallyId === selectedRallyId && currentRally && (
            <BallTrackingDebugOverlay
              positions={debugBallPositions}
              frameCount={debugFrameCount}
              rallyStartTime={currentRally.start_time}
              rallyEndTime={currentRally.end_time}
              videoRef={videoRef}
              aspectRatio={currentCameraEdit?.aspectRatio}
              cameraX={currentCameraState.positionX}
              cameraY={currentCameraState.positionY}
              zoom={currentCameraState.zoom}
            />
          )}
          {/* Transform wrapper - video frame callback updates, CSS transition smooths between frames */}
          <div
            ref={transformWrapperRef}
            style={containerAspectRatio === '9/16' ? {
              position: 'absolute',
              top: 0,
              left: '50%',
              width: 'auto',
              height: '100%',
              willChange: 'transform',
              // CSS transition smooths between video frame updates
              transition: isPlaying && hasCameraKeyframes ? 'transform 60ms ease-out' : 'none',
              ...(isPlaying && hasCameraKeyframes ? {} : videoTransformStyle),
            } : {
              width: '100%',
              height: '100%',
              willChange: isCameraPreviewActive ? 'transform' : 'auto',
              // CSS transition smooths between video frame updates
              transition: isPlaying && hasCameraKeyframes ? 'transform 60ms ease-out' : 'none',
              ...(isPlaying && hasCameraKeyframes ? {} : videoTransformStyle),
            }}
          >
            <video
              ref={videoRef}
              src={effectiveVideoUrl}
              poster={posterUrl || undefined}
              preload="metadata"
              crossOrigin={process.env.NODE_ENV === 'production' ? 'anonymous' : undefined}
              style={containerAspectRatio === '9/16' ? {
                width: 'auto',
                height: '100%',
              } : {
                width: '100%',
                height: '100%',
                objectFit: 'contain',
              }}
              onTimeUpdate={handleTimeUpdate}
              onSeeked={handleSeeked}
              onLoadedMetadata={handleLoadedMetadata}
              onLoadStart={handleLoadStart}
              onProgress={handleProgress}
              onWaiting={handleWaiting}
              onCanPlay={handleCanPlay}
              playsInline
            />
          </div>
        </Box>
      </Box>
    </Box>
  );
}
