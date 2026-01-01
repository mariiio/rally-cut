'use client';

import { useRef, useEffect, useCallback, useState } from 'react';
import { Box, CircularProgress, Typography } from '@mui/material';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import { usePlayerStore } from '@/stores/playerStore';
import { useEditorStore } from '@/stores/editorStore';
import { designTokens } from '@/app/theme';

export function VideoPlayer() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isLoading, setIsLoading] = useState(false);

  const videoUrl = useEditorStore((state) => state.videoUrl);
  const activeMatchId = useEditorStore((state) => state.activeMatchId);
  const setActiveMatch = useEditorStore((state) => state.setActiveMatch);

  const isPlaying = usePlayerStore((state) => state.isPlaying);
  const seekTo = usePlayerStore((state) => state.seekTo);
  const clearSeek = usePlayerStore((state) => state.clearSeek);
  const setCurrentTime = usePlayerStore((state) => state.setCurrentTime);
  const setDuration = usePlayerStore((state) => state.setDuration);
  const setReady = usePlayerStore((state) => state.setReady);

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

  // Handle manual seek requests
  useEffect(() => {
    if (seekTo !== null && videoRef.current) {
      videoRef.current.currentTime = seekTo;
      clearSeek();
      // Resume playback if we're supposed to be playing
      if (isPlaying) {
        videoRef.current.play().catch(() => {});
      }
    }
  }, [seekTo, clearSeek, isPlaying]);

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
    setReady(false);
  }, [setReady]);

  const handleEmptyStateClick = () => {
    // Trigger file upload via EditorHeader's exposed function
    const triggerFn = (window as unknown as { triggerVideoUpload?: () => void }).triggerVideoUpload;
    if (triggerFn) {
      triggerFn();
    }
  };

  if (!videoUrl) {
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

  return (
    <Box
      sx={{
        width: '100%',
        aspectRatio: '16/9',
        bgcolor: '#000',
        borderRadius: 2,
        overflow: 'hidden',
        position: 'relative',
        boxShadow: designTokens.colors.video.shadow,
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
      }}
    >
      {/* Loading overlay */}
      {isLoading && (
        <Box
          sx={{
            position: 'absolute',
            inset: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            bgcolor: 'rgba(0, 0, 0, 0.7)',
            zIndex: 2,
          }}
        >
          <CircularProgress color="primary" />
        </Box>
      )}

      <video
        ref={videoRef}
        src={videoUrl}
        crossOrigin={process.env.NODE_ENV === 'production' ? 'anonymous' : undefined}
        style={{
          width: '100%',
          height: '100%',
          objectFit: 'contain',
        }}
        onTimeUpdate={handleTimeUpdate}
        onLoadedMetadata={handleLoadedMetadata}
        onLoadStart={handleLoadStart}
        playsInline
      />
    </Box>
  );
}
