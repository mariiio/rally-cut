'use client';

import { useRef, useEffect, useCallback, useState, useMemo } from 'react';
import { Box, CircularProgress, Typography } from '@mui/material';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import { usePlayerStore } from '@/stores/playerStore';
import { useEditorStore } from '@/stores/editorStore';
import { useUploadStore } from '@/stores/uploadStore';
import { useCameraStore, selectHandheldPreset } from '@/stores/cameraStore';
import { calculateVideoTransform, getCameraStateWithHandheld, resetHandheldState } from '@/utils/cameraInterpolation';
import { DEFAULT_CAMERA_STATE } from '@/types/camera';
import { designTokens } from '@/app/theme';
import { CameraOverlay } from './CameraOverlay';

export function VideoPlayer() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const videoContainerRef = useRef<HTMLDivElement>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isBuffering, setIsBuffering] = useState(false);
  const [bufferProgress, setBufferProgress] = useState(0);
  // Camera time updated via RAF for smooth camera panning during playback
  const [cameraTime, setCameraTime] = useState(0);

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

  // Camera state
  const cameraEdits = useCameraStore((state) => state.cameraEdits);
  const dragPosition = useCameraStore((state) => state.dragPosition);
  const selectedKeyframeId = useCameraStore((state) => state.selectedKeyframeId);
  const handheldPreset = useCameraStore(selectHandheldPreset);

  // Get rallies from editor store
  const rallies = useEditorStore((state) => state.rallies);
  const selectedRallyId = useEditorStore((state) => state.selectedRallyId);
  const isCameraTabActive = useEditorStore((state) => state.isCameraTabActive);

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

  // Check if camera edit has keyframes for the active aspect ratio
  const hasCameraKeyframes = currentCameraEdit &&
    (currentCameraEdit.keyframes[currentCameraEdit.aspectRatio]?.length ?? 0) > 0;

  // Determine if camera should be applied (toggle on OR actively editing a keyframe)
  const shouldApplyCamera = applyCameraEdits || selectedKeyframeId !== null;

  // Calculate video transform style based on camera state
  const videoTransformStyle = useMemo(() => {
    // Only apply when camera preview is on and we have a rally
    if (!shouldApplyCamera || !currentRally) {
      return {};
    }

    const aspectRatio = currentCameraEdit?.aspectRatio ?? 'ORIGINAL';

    // If no keyframes, show default centered position for the aspect ratio
    if (!hasCameraKeyframes) {
      return calculateVideoTransform(DEFAULT_CAMERA_STATE, aspectRatio);
    }

    // Use cameraTime for camera position - updated via RAF during playback
    // and via seeked event when scrubbing
    const effectiveTime = cameraTime;

    // Calculate time offset within the rally (0-1)
    const rallyDuration = currentRally.end_time - currentRally.start_time;
    const timeWithinRally = rallyDuration > 0
      ? (effectiveTime - currentRally.start_time) / rallyDuration
      : 0;
    const clampedOffset = Math.max(0, Math.min(1, timeWithinRally));

    // Get keyframes for the active aspect ratio
    const keyframes = currentCameraEdit?.keyframes[aspectRatio] ?? [];

    // Get interpolated camera state with handheld motion applied
    const cameraState = getCameraStateWithHandheld(
      keyframes,
      clampedOffset,
      effectiveTime,
      currentRally.id,
      handheldPreset
    );

    // If dragging, override position with drag position for live preview
    const effectiveState = dragPosition
      ? { ...cameraState, positionX: dragPosition.x, positionY: dragPosition.y }
      : cameraState;

    // Calculate CSS transform
    return calculateVideoTransform(effectiveState, aspectRatio);
  }, [shouldApplyCamera, currentRally, hasCameraKeyframes, currentCameraEdit, cameraTime, handheldPreset, dragPosition]);

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

  // RAF loop for smooth camera updates during playback
  // This runs at 60fps to sample video.currentTime for smooth panning
  useEffect(() => {
    if (!isPlaying || !shouldApplyCamera || !hasCameraKeyframes) {
      return;
    }

    let rafId: number;
    const updateCameraTime = () => {
      const video = videoRef.current;
      if (video) {
        setCameraTime(video.currentTime);
      }
      rafId = requestAnimationFrame(updateCameraTime);
    };

    rafId = requestAnimationFrame(updateCameraTime);

    return () => {
      cancelAnimationFrame(rafId);
      // Reset handheld state when playback stops to avoid stale spring physics
      resetHandheldState();
    };
  }, [isPlaying, shouldApplyCamera, hasCameraKeyframes]);

  // Handle manual seek requests
  useEffect(() => {
    if (seekTo !== null && videoRef.current) {
      videoRef.current.currentTime = seekTo;
      // Update cameraTime immediately for smooth preview when paused
      // eslint-disable-next-line react-hooks/set-state-in-effect -- intentional: syncing camera time with seek
      setCameraTime(seekTo);
      clearSeek();
      // Resume playback if we're supposed to be playing
      if (isPlaying) {
        videoRef.current.play().catch(() => {});
      }
    }
  }, [seekTo, clearSeek, isPlaying]);

  // Handle seeked event - update camera time when scrubbing
  const handleSeeked = useCallback(() => {
    const video = videoRef.current;
    if (video && !isPlaying) {
      setCameraTime(video.currentTime);
    }
  }, [isPlaying]);

  // Track if we're in the middle of a match switch
  const switchingMatchRef = useRef(false);

  // Time update handler - check for rally end during highlight playback
  const handleTimeUpdate = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;

    setCurrentTime(video.currentTime);

    // Skip if we're in the middle of switching matches
    if (switchingMatchRef.current) return;

    // Read fresh values from stores to avoid stale closures
    const playerState = usePlayerStore.getState();
    const editorState = useEditorStore.getState();

    // Handle highlight playback
    if (playerState.playingHighlightId) {
      const currentRally = playerState.getCurrentPlaylistRally();
      // Verify the current rally belongs to the active match
      if (currentRally && currentRally.matchId === editorState.activeMatchId && video.currentTime >= currentRally.end_time) {
        const nextRally = playerState.advanceHighlightPlayback();
        if (nextRally) {
          if (nextRally.matchId === editorState.activeMatchId) {
            // Same match, just seek
            video.currentTime = nextRally.start_time;
          } else {
            // Different match - switch to that match's video
            switchingMatchRef.current = true;
            setActiveMatch(nextRally.matchId);
          }
        } else {
          // No more rallies, stop playback
          playerState.stopHighlightPlayback();
        }
      }
    }
  }, [setCurrentTime, setActiveMatch]);

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
          overflow: 'hidden',
        }}
      >
        <Box
          ref={videoContainerRef}
          sx={{
            position: containerAspectRatio === '9/16' ? 'absolute' : 'relative',
            inset: containerAspectRatio === '9/16' ? 0 : undefined,
            width: '100%',
            height: '100%',
            overflow: 'hidden',
          }}
        >
          <CameraOverlay containerRef={videoContainerRef} />
          <video
            ref={videoRef}
            src={effectiveVideoUrl}
            poster={posterUrl || undefined}
            preload="metadata"
            crossOrigin={process.env.NODE_ENV === 'production' ? 'anonymous' : undefined}
            style={containerAspectRatio === '9/16' ? {
              // 9:16 mode: position video at center, transform handles panning
              position: 'absolute',
              top: 0,
              left: '50%',
              width: 'auto',
              height: '100%',
              willChange: 'transform',
              transition: dragPosition ? 'none' : 'transform 0.2s ease-out',
              ...videoTransformStyle,
            } : {
              // 16:9 mode: normal video display
              width: '100%',
              height: '100%',
              objectFit: 'contain',
              willChange: isCameraPreviewActive ? 'transform' : 'auto',
              transition: dragPosition ? 'none' : 'transform 0.2s ease-out',
              ...videoTransformStyle,
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
        </Box>
      </Box>
    </Box>
  );
}
