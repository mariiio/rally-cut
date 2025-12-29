'use client';

import { useRef, useEffect, useCallback, useState } from 'react';
import { Box, CircularProgress } from '@mui/material';
import { usePlayerStore } from '@/stores/playerStore';
import { useEditorStore } from '@/stores/editorStore';

export function VideoPlayer() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isLoading, setIsLoading] = useState(false);
  const videoUrl = useEditorStore((state) => state.videoUrl);
  const {
    isPlaying,
    seekTo,
    clearSeek,
    setCurrentTime,
    setDuration,
    setReady,
  } = usePlayerStore();

  // Handle play/pause
  useEffect(() => {
    if (!videoRef.current) return;
    if (isPlaying) {
      videoRef.current.play().catch(() => {});
    } else {
      videoRef.current.pause();
    }
  }, [isPlaying]);

  // Handle seek requests from store
  useEffect(() => {
    if (seekTo !== null && videoRef.current) {
      videoRef.current.currentTime = seekTo;
      clearSeek();
    }
  }, [seekTo, clearSeek]);

  const handleTimeUpdate = useCallback(() => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime);
    }
  }, [setCurrentTime]);

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
            zIndex: 1,
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
