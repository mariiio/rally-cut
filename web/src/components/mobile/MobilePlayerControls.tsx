'use client';

import {
  Box,
  IconButton,
  Slider,
  Typography,
  Button,
  Stack,
  CircularProgress,
} from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import PauseIcon from '@mui/icons-material/Pause';
import SkipPreviousIcon from '@mui/icons-material/SkipPrevious';
import SkipNextIcon from '@mui/icons-material/SkipNext';
import AddIcon from '@mui/icons-material/Add';
import AutoFixHighIcon from '@mui/icons-material/AutoFixHigh';
import { usePlayerStore } from '@/stores/playerStore';
import { useEditorStore } from '@/stores/editorStore';
import { formatTime } from '@/utils/timeFormat';
import { designTokens } from '@/app/theme';
import { VideoStatus } from '@/constants/enums';
import { useAnalysisStore } from '@/stores/analysisStore';

export function MobilePlayerControls() {
  const { isPlaying, currentTime, duration, togglePlay, seek } = usePlayerStore();
  const {
    rallies,
    activeMatchId,
    createRallyAtTime,
    videoUrl,
    isRallyEditingLocked,
  } = useEditorStore();

  const isLocked = isRallyEditingLocked();

  // Read detection state from analysisStore (single source of truth)
  const analysisPipeline = useAnalysisStore(
    (s) => activeMatchId ? s.pipelines[activeMatchId] ?? null : null
  );
  const getActiveMatch = useEditorStore((s) => s.getActiveMatch);
  const activeMatch = getActiveMatch();

  const isDetecting = analysisPipeline?.phase === 'detecting' || analysisPipeline?.phase === 'tracking'
    || activeMatch?.status === 'DETECTING';
  const detectionProgress = analysisPipeline?.phase === 'detecting'
    ? Math.max(0, Math.min(100, Math.round(((analysisPipeline.progress - 10) / 35) * 100)))
    : 0;
  const videoDetectionStatus = activeMatch?.status ?? null;

  const startAnalysis = useAnalysisStore((s) => s.startAnalysis);

  const handleStartDetection = () => {
    if (!activeMatchId) return;
    startAnalysis(activeMatchId);
  };

  const handleSliderChange = (_: Event, value: number | number[]) => {
    seek(value as number);
  };

  const handleSkipBack = () => {
    seek(Math.max(0, currentTime - 5));
  };

  const handleSkipForward = () => {
    seek(Math.min(duration, currentTime + 5));
  };

  const handleCreateRally = () => {
    createRallyAtTime(currentTime);
  };

  // Check if current time is inside a rally (can't create overlapping)
  const isInsideRally = rallies?.some(
    (r) => currentTime >= r.start_time && currentTime <= r.end_time
  );

  const hasVideo = Boolean(videoUrl);
  const canDetect = hasVideo && !isDetecting && videoDetectionStatus !== VideoStatus.DETECTED;

  return (
    <Box
      sx={{
        px: 2,
        py: 1,
        bgcolor: 'background.paper',
        borderBottom: '1px solid',
        borderColor: 'divider',
      }}
    >
      {/* Time slider */}
      <Slider
        size="small"
        value={currentTime}
        max={duration || 100}
        onChange={handleSliderChange}
        disabled={!hasVideo}
        sx={{
          color: 'primary.main',
          py: 1,
          '& .MuiSlider-thumb': {
            width: 16,
            height: 16,
          },
          '& .MuiSlider-track': {
            height: 4,
          },
          '& .MuiSlider-rail': {
            height: 4,
          },
        }}
      />

      {/* Controls row */}
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}
      >
        {/* Playback controls */}
        <Stack direction="row" alignItems="center" spacing={0.5}>
          <IconButton
            onClick={handleSkipBack}
            size="small"
            disabled={!hasVideo}
            sx={{ minWidth: designTokens.mobile.touchTarget }}
          >
            <SkipPreviousIcon />
          </IconButton>

          <IconButton
            onClick={togglePlay}
            size="medium"
            color="primary"
            disabled={!hasVideo}
            sx={{
              minWidth: designTokens.mobile.touchTarget,
              minHeight: designTokens.mobile.touchTarget,
            }}
          >
            {isPlaying ? <PauseIcon fontSize="large" /> : <PlayArrowIcon fontSize="large" />}
          </IconButton>

          <IconButton
            onClick={handleSkipForward}
            size="small"
            disabled={!hasVideo}
            sx={{ minWidth: designTokens.mobile.touchTarget }}
          >
            <SkipNextIcon />
          </IconButton>

          <Typography
            variant="caption"
            sx={{ fontFamily: 'monospace', color: 'text.secondary', ml: 1 }}
          >
            {formatTime(currentTime)} / {formatTime(duration)}
          </Typography>
        </Stack>

        {/* Action buttons */}
        <Stack direction="row" alignItems="center" spacing={1}>
          {/* Create Rally */}
          <IconButton
            onClick={handleCreateRally}
            size="small"
            color="primary"
            disabled={!hasVideo || isInsideRally || isLocked}
            sx={{
              minWidth: designTokens.mobile.touchTarget,
              minHeight: designTokens.mobile.touchTarget,
              bgcolor: 'rgba(255, 107, 74, 0.1)',
              '&:hover': { bgcolor: 'rgba(255, 107, 74, 0.2)' },
              '&.Mui-disabled': { bgcolor: 'transparent' },
            }}
          >
            <AddIcon />
          </IconButton>

          {/* Detect Rallies */}
          {isDetecting ? (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
              <CircularProgress size={20} />
              <Typography variant="caption" color="text.secondary">
                {detectionProgress > 0 ? `${detectionProgress}%` : '...'}
              </Typography>
            </Box>
          ) : canDetect ? (
            <Button
              size="small"
              variant="outlined"
              startIcon={<AutoFixHighIcon sx={{ fontSize: 16 }} />}
              onClick={handleStartDetection}
              sx={{
                fontSize: '0.75rem',
                py: 0.5,
                px: 1,
                minHeight: 32,
                textTransform: 'none',
              }}
            >
              Detect
            </Button>
          ) : videoDetectionStatus === VideoStatus.DETECTED ? (
            <Typography variant="caption" color="success.main">
              {rallies?.length || 0} rallies
            </Typography>
          ) : null}
        </Stack>
      </Box>
    </Box>
  );
}
