import { Box, LinearProgress, Typography } from '@mui/material';
import CheckIcon from '@mui/icons-material/Check';
import { designTokens } from '@/app/designTokens';

interface ProgressBarProps {
  progress: number;
  isActive: boolean;
  isComplete?: boolean;
  stepText: string;
  completeText?: string;
  /** Right-side content (e.g. cancel button) shown when not complete */
  actions?: React.ReactNode;
}

export function ProgressBar({
  progress,
  isActive,
  isComplete = false,
  stepText,
  completeText = 'Complete',
  actions,
}: ProgressBarProps) {
  const showSweep = !isComplete && progress > 0;

  return (
    <Box
      sx={{
        width: '100%',
        opacity: isActive || isComplete ? 1 : 0,
        transition: 'opacity 0.2s ease',
      }}
    >
      {/* Progress bar */}
      <Box sx={{ position: 'relative', height: 3 }}>
        <LinearProgress
          variant="determinate"
          value={progress}
          sx={{
            height: 3,
            bgcolor: designTokens.alpha.white[6],
            '& .MuiLinearProgress-bar': {
              bgcolor: isComplete ? 'success.main' : 'primary.main',
              transition: 'transform 0.2s ease',
            },
          }}
        />
        {/* Glow sweep */}
        {showSweep && (
          <Box
            sx={{
              position: 'absolute',
              top: 0,
              bottom: 0,
              left: 0,
              width: `${progress}%`,
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
                background: `linear-gradient(90deg, transparent, ${designTokens.alpha.white[35]}, transparent)`,
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
          bgcolor: designTokens.alpha.black[30],
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75 }}>
          {isComplete && (
            <CheckIcon sx={{ fontSize: 12, color: 'success.main' }} />
          )}
          <Typography
            variant="caption"
            sx={{
              color: isComplete ? 'success.main' : 'text.secondary',
              fontSize: 11,
            }}
          >
            {isComplete ? completeText : stepText}
          </Typography>
        </Box>
        {!isComplete && (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Typography
              variant="caption"
              sx={{
                color: 'text.secondary',
                fontSize: 11,
                fontFamily: 'monospace',
              }}
            >
              {progress}%
            </Typography>
            {actions}
          </Box>
        )}
      </Box>
    </Box>
  );
}
