'use client';

import { Box, IconButton, Slider, Typography } from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import PauseIcon from '@mui/icons-material/Pause';
import SkipPreviousIcon from '@mui/icons-material/SkipPrevious';
import SkipNextIcon from '@mui/icons-material/SkipNext';
import { usePlayerStore } from '@/stores/playerStore';
import { formatTime } from '@/utils/timeFormat';

export function PlayerControls() {
  const { isPlaying, currentTime, duration, togglePlay, seek } =
    usePlayerStore();

  const handleSliderChange = (_: Event, value: number | number[]) => {
    seek(value as number);
  };

  const handleSkipBack = () => {
    seek(Math.max(0, currentTime - 5));
  };

  const handleSkipForward = () => {
    seek(Math.min(duration, currentTime + 5));
  };

  return (
    <Box sx={{ width: '100%', px: 2, py: 1 }}>
      {/* Time slider */}
      <Slider
        size="small"
        value={currentTime}
        max={duration || 100}
        onChange={handleSliderChange}
        sx={{
          color: 'primary.main',
          '& .MuiSlider-thumb': {
            width: 12,
            height: 12,
          },
        }}
      />

      {/* Controls row */}
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 1,
        }}
      >
        <IconButton onClick={handleSkipBack} size="small" color="inherit">
          <SkipPreviousIcon />
        </IconButton>

        <IconButton onClick={togglePlay} size="medium" color="primary">
          {isPlaying ? (
            <PauseIcon fontSize="large" />
          ) : (
            <PlayArrowIcon fontSize="large" />
          )}
        </IconButton>

        <IconButton onClick={handleSkipForward} size="small" color="inherit">
          <SkipNextIcon />
        </IconButton>

        <Typography variant="body2" sx={{ ml: 2, fontFamily: 'monospace' }}>
          {formatTime(currentTime)} / {formatTime(duration)}
        </Typography>
      </Box>
    </Box>
  );
}
