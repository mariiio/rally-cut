'use client';

import { useRef, useEffect, useCallback, useState, useMemo } from 'react';
import { Box, CircularProgress } from '@mui/material';
import { usePlayerStore } from '@/stores/playerStore';
import { useEditorStore } from '@/stores/editorStore';

export function VideoPlayer() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isLoading, setIsLoading] = useState(false);
  const transitioningRef = useRef(false);

  const videoUrl = useEditorStore((state) => state.videoUrl);
  const highlights = useEditorStore((state) => state.highlights);
  const rallies = useEditorStore((state) => state.rallies);

  const isPlaying = usePlayerStore((state) => state.isPlaying);
  const seekTo = usePlayerStore((state) => state.seekTo);
  const clearSeek = usePlayerStore((state) => state.clearSeek);
  const setCurrentTime = usePlayerStore((state) => state.setCurrentTime);
  const setDuration = usePlayerStore((state) => state.setDuration);
  const setReady = usePlayerStore((state) => state.setReady);
  const playingHighlightId = usePlayerStore((state) => state.playingHighlightId);
  const highlightRallyIndex = usePlayerStore((state) => state.highlightRallyIndex);
  const advanceHighlightPlayback = usePlayerStore((state) => state.advanceHighlightPlayback);
  const stopHighlightPlayback = usePlayerStore((state) => state.stopHighlightPlayback);

  const highlightRallies = useMemo(() => {
    if (!playingHighlightId || !highlights || !rallies) return [];
    const highlight = highlights.find((h) => h.id === playingHighlightId);
    if (!highlight) return [];
    return rallies
      .filter((r) => highlight.rallyIds?.includes(r.id))
      .sort((a, b) => a.start_time - b.start_time);
  }, [playingHighlightId, highlights, rallies]);

  const currentRally = highlightRallies[highlightRallyIndex];
  const nextRally = highlightRallies[highlightRallyIndex + 1];

  // Play/pause
  useEffect(() => {
    const video = videoRef.current;
    if (!video || transitioningRef.current) return;
    if (isPlaying) {
      video.play().catch(() => {});
    } else {
      video.pause();
    }
  }, [isPlaying]);

  // Manual seek
  useEffect(() => {
    if (seekTo !== null && videoRef.current) {
      videoRef.current.currentTime = seekTo;
      clearSeek();
    }
  }, [seekTo, clearSeek]);

  // Start highlight playback
  useEffect(() => {
    if (playingHighlightId && highlightRallies.length > 0 && videoRef.current) {
      transitioningRef.current = false;
      videoRef.current.currentTime = highlightRallies[0].start_time;
      videoRef.current.play().catch(() => {});
    }
  }, [playingHighlightId, highlightRallies]);

  // Time update
  const handleTimeUpdate = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;

    setCurrentTime(video.currentTime);

    if (!playingHighlightId || !currentRally || transitioningRef.current) return;

    if (video.currentTime >= currentRally.end_time) {
      if (nextRally) {
        transitioningRef.current = true;
        advanceHighlightPlayback();
        video.currentTime = nextRally.start_time;
        transitioningRef.current = false;
      } else {
        stopHighlightPlayback();
      }
    }
  }, [playingHighlightId, currentRally, nextRally, setCurrentTime, advanceHighlightPlayback, stopHighlightPlayback]);

  const handleLoadedMetadata = useCallback(() => {
    if (videoRef.current) {
      setDuration(videoRef.current.duration);
      setReady(true);
      setIsLoading(false);
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
