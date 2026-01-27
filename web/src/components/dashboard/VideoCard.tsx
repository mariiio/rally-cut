'use client';

import { Box, Card, CardContent, Chip, Typography, Skeleton } from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';
import HourglassEmptyIcon from '@mui/icons-material/HourglassEmpty';
import { designTokens } from '@/app/theme';
import { getVideoStreamUrl } from '@/services/api';
import { VideoStatus } from '@/constants/enums';

interface VideoCardProps {
  id: string;
  name: string;
  durationMs: number | null;
  posterS3Key?: string | null;
  status?: VideoStatus;
  onClick?: () => void;
  variant?: 'compact' | 'medium' | 'featured';
  sessionTag?: string;
}

function formatDuration(ms: number | null): string {
  if (!ms) return '--:--';
  const totalSeconds = Math.floor(ms / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, '0')}`;
}

export function VideoCard({
  name,
  durationMs,
  posterS3Key,
  status = VideoStatus.DETECTED,
  onClick,
  variant = 'medium',
  sessionTag,
}: VideoCardProps) {
  const isProcessing = status === VideoStatus.DETECTING || status === VideoStatus.PENDING || status === VideoStatus.UPLOADED;
  const isError = status === VideoStatus.ERROR;

  const aspectRatio = variant === 'featured' ? '21/9' : '16/9';
  const playIconSize = variant === 'compact' ? 32 : variant === 'featured' ? 56 : 44;

  return (
    <Card
      sx={{
        bgcolor: designTokens.colors.surface[2],
        borderRadius: variant === 'featured' ? 3 : 2,
        overflow: 'hidden',
        border: '1px solid',
        borderColor: variant === 'featured' ? 'primary.main' : 'transparent',
        transition: 'all 0.25s cubic-bezier(0.4, 0, 0.2, 1)',
        cursor: onClick ? 'pointer' : 'default',
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        ...(variant === 'featured' && {
          '&::before': {
            content: '""',
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            height: 3,
            background: designTokens.gradients.sunset,
            zIndex: 1,
          },
          position: 'relative',
        }),
        '&:hover': {
          transform: onClick ? 'translateY(-4px) scale(1.01)' : 'none',
          boxShadow: onClick ? designTokens.shadows.lg : 'none',
          borderColor: onClick ? 'primary.main' : 'transparent',
          '& .thumbnail-overlay': {
            opacity: 1,
          },
          '& .play-icon': {
            transform: 'translate(-50%, -50%) scale(1)',
            opacity: 1,
          },
        },
      }}
      onClick={onClick}
    >
      <Box sx={{ position: 'relative', aspectRatio, flexShrink: 0 }}>
        {posterS3Key ? (
          /* eslint-disable-next-line @next/next/no-img-element -- dynamic API URL */
          <img
            src={getVideoStreamUrl(posterS3Key)}
            alt={name}
            loading="lazy"
            style={{
              width: '100%',
              height: '100%',
              objectFit: 'cover',
              backgroundColor: designTokens.colors.surface[0],
              display: 'block',
            }}
          />
        ) : (
          <Box
            sx={{
              width: '100%',
              height: '100%',
              bgcolor: designTokens.colors.surface[0],
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            {isProcessing && (
              <HourglassEmptyIcon sx={{ fontSize: 32, color: 'text.disabled' }} />
            )}
            {isError && (
              <ErrorOutlineIcon sx={{ fontSize: 32, color: 'error.main' }} />
            )}
          </Box>
        )}

        {/* Gradient overlay */}
        <Box
          sx={{
            position: 'absolute',
            inset: 0,
            background:
              'linear-gradient(to top, rgba(0,0,0,0.7) 0%, rgba(0,0,0,0.2) 40%, transparent 100%)',
            pointerEvents: 'none',
          }}
        />

        {/* Hover overlay */}
        <Box
          className="thumbnail-overlay"
          sx={{
            position: 'absolute',
            inset: 0,
            background: designTokens.alpha.black[30],
            opacity: 0,
            transition: 'opacity 0.2s ease',
            pointerEvents: 'none',
          }}
        />

        {/* Play icon */}
        {status === VideoStatus.DETECTED && onClick && (
          <Box
            className="play-icon"
            sx={{
              position: 'absolute',
              top: '50%',
              left: '50%',
              transform: 'translate(-50%, -50%) scale(0.8)',
              opacity: 0,
              transition: 'all 0.25s cubic-bezier(0.4, 0, 0.2, 1)',
              width: playIconSize,
              height: playIconSize,
              borderRadius: '50%',
              bgcolor: 'rgba(255, 107, 74, 0.95)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: designTokens.shadows.glow.primary,
              pointerEvents: 'none',
            }}
          >
            <PlayArrowIcon
              sx={{ color: 'white', fontSize: playIconSize * 0.6 }}
            />
          </Box>
        )}

        {/* Top-left badges */}
        {(isProcessing || isError || sessionTag) && (
          <Box
            sx={{
              position: 'absolute',
              top: 8,
              left: 8,
              display: 'flex',
              gap: 0.5,
              maxWidth: 'calc(100% - 16px)',
            }}
          >
            {sessionTag && (
              <Chip
                label={sessionTag}
                size="small"
                sx={{
                  bgcolor: 'rgba(0, 212, 170, 0.9)',
                  color: 'white',
                  fontSize: '0.65rem',
                  fontWeight: 600,
                  height: 22,
                  backdropFilter: 'blur(4px)',
                  '& .MuiChip-label': {
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                  },
                }}
              />
            )}
            {isProcessing && (
              <Chip
                icon={<HourglassEmptyIcon sx={{ fontSize: 14 }} />}
                label="Processing"
                size="small"
                sx={{
                  bgcolor: 'rgba(59, 130, 246, 0.9)',
                  color: 'white',
                  fontSize: '0.7rem',
                  fontWeight: 600,
                  height: 24,
                  backdropFilter: 'blur(4px)',
                  '& .MuiChip-icon': {
                    color: 'white',
                  },
                }}
              />
            )}
            {isError && (
              <Chip
                icon={<ErrorOutlineIcon sx={{ fontSize: 14 }} />}
                label="Error"
                size="small"
                sx={{
                  bgcolor: 'rgba(239, 68, 68, 0.9)',
                  color: 'white',
                  fontSize: '0.7rem',
                  fontWeight: 600,
                  height: 24,
                  backdropFilter: 'blur(4px)',
                  '& .MuiChip-icon': {
                    color: 'white',
                  },
                }}
              />
            )}
          </Box>
        )}

        {/* Duration badge */}
        <Chip
          label={formatDuration(durationMs)}
          size="small"
          sx={{
            position: 'absolute',
            bottom: 8,
            right: 8,
            bgcolor: 'rgba(0, 0, 0, 0.8)',
            color: 'white',
            fontSize: '0.7rem',
            fontWeight: 600,
            height: 24,
            backdropFilter: 'blur(4px)',
          }}
        />
      </Box>

      <CardContent
        sx={{
          py: variant === 'compact' ? 1 : 1.5,
          px: variant === 'compact' ? 1 : 1.5,
          '&:last-child': { pb: variant === 'compact' ? 1 : 1.5 },
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
        }}
      >
        <Typography
          variant={variant === 'featured' ? 'subtitle1' : 'body2'}
          noWrap
          title={name}
          sx={{
            fontWeight: 500,
            color: 'text.primary',
          }}
        >
          {name}
        </Typography>
      </CardContent>
    </Card>
  );
}

export function VideoCardSkeleton({ variant = 'medium' }: { variant?: 'compact' | 'medium' | 'featured' }) {
  const aspectRatio = variant === 'featured' ? '21/9' : '16/9';

  return (
    <Card
      sx={{
        bgcolor: designTokens.colors.surface[2],
        borderRadius: variant === 'featured' ? 3 : 2,
        overflow: 'hidden',
        border: '1px solid',
        borderColor: 'transparent',
      }}
    >
      <Skeleton
        variant="rectangular"
        sx={{ aspectRatio, width: '100%' }}
      />
      <CardContent
        sx={{
          py: variant === 'compact' ? 1 : 1.5,
          px: variant === 'compact' ? 1 : 1.5,
          '&:last-child': { pb: variant === 'compact' ? 1 : 1.5 },
        }}
      >
        <Skeleton width="80%" />
      </CardContent>
    </Card>
  );
}
