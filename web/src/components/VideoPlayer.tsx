'use client';

import { useRef, useEffect, useCallback, useState } from 'react';
import { Box, CircularProgress } from '@mui/material';
import { usePlayerStore } from '@/stores/playerStore';
import { useEditorStore } from '@/stores/editorStore';

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
    }
  }, [seekTo, clearSeek]);

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

  if (!videoUrl) {
    return (
      <Box
        sx={{
          width: '100%',
          aspectRatio: '16/9',
          bgcolor: 'background.paper',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          borderRadius: 1,
          color: 'text.secondary',
        }}
      >
        Upload a video to get started
      </Box>
    );
  }

  return (
    <Box
      sx={{
        width: '100%',
        aspectRatio: '16/9',
        bgcolor: 'black',
        borderRadius: 1,
        overflow: 'hidden',
        position: 'relative',
      }}
    >
      {isLoading && (
        <Box
          sx={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            zIndex: 2,
          }}
        >
          <CircularProgress />
        </Box>
      )}

      <video
        ref={videoRef}
        src={videoUrl}
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
