'use client';

import { Box, LinearProgress, Typography } from '@mui/material';
import { useEditorStore } from '@/stores/editorStore';

export function SessionLoadingProgress() {
  const isLoadingSession = useEditorStore((state) => state.isLoadingSession);
  const sessionLoadStep = useEditorStore((state) => state.sessionLoadStep);
  const sessionLoadProgress = useEditorStore((state) => state.sessionLoadProgress);

  // Show nothing when not loading
  if (!isLoadingSession || sessionLoadProgress === 0) {
    return null;
  }

  return (
    <Box
      sx={{
        width: '100%',
        opacity: 1,
        transition: 'opacity 0.2s ease',
      }}
    >
      {/* Progress bar */}
      <Box sx={{ position: 'relative', height: 3 }}>
        <LinearProgress
          variant="determinate"
          value={sessionLoadProgress}
          sx={{
            height: 3,
            bgcolor: 'rgba(255, 255, 255, 0.06)',
            '& .MuiLinearProgress-bar': {
              bgcolor: 'primary.main',
              transition: 'transform 0.2s ease',
            },
          }}
        />
        {/* Glow sweep */}
        {sessionLoadProgress > 0 && sessionLoadProgress < 100 && (
          <Box
            sx={{
              position: 'absolute',
              top: 0,
              bottom: 0,
              left: 0,
              width: `${sessionLoadProgress}%`,
              overflow: 'hidden',
              pointerEvents: 'none',
            }}
          >
            <Box
              sx={{
                position: 'absolute',
                top: 0,
                bottom: 0,
                width: 40,
                background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.35), transparent)',
                animation: 'sweep 1.5s ease-in-out infinite',
                '@keyframes sweep': {
                  '0%': { left: '-40px' },
                  '100%': { left: '100%' },
                },
              }}
            />
          </Box>
        )}
      </Box>

      {/* Status text */}
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          px: 1,
          py: 0.5,
          bgcolor: 'rgba(0, 0, 0, 0.3)',
        }}
      >
        <Typography
          variant="caption"
          sx={{
            color: 'text.secondary',
            fontSize: 11,
          }}
        >
          {sessionLoadStep}
        </Typography>
        <Typography
          variant="caption"
          sx={{
            color: 'text.secondary',
            fontSize: 11,
            fontFamily: 'monospace',
          }}
        >
          {sessionLoadProgress}%
        </Typography>
      </Box>
    </Box>
  );
}
