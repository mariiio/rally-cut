'use client';

import { Box, Card, CardActionArea, CardContent, Chip, Stack, Typography, Skeleton } from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import ChevronRightIcon from '@mui/icons-material/ChevronRight';
import PeopleIcon from '@mui/icons-material/People';
import { designTokens } from '@/app/theme';
import { getVideoStreamUrl } from '@/services/api';

interface VideoThumbnail {
  id: string;
  posterS3Key?: string | null;
}

interface SessionCardProps {
  id: string;
  name: string;
  videoCount: number;
  updatedAt?: string;
  videos?: VideoThumbnail[];
  onClick?: () => void;
  onAddVideos?: () => void;
  variant?: 'default' | 'featured' | 'shared';
  sharedBy?: string;
  highlightCount?: number;
}

function formatRelativeTime(dateString: string): string {
  const date = new Date(dateString);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

  if (diffDays === 0) return 'Updated today';
  if (diffDays === 1) return 'Updated yesterday';
  if (diffDays < 7) return `Updated ${diffDays} days ago`;
  if (diffDays < 30) return `Updated ${Math.floor(diffDays / 7)} weeks ago`;
  return `Updated ${Math.floor(diffDays / 30)} months ago`;
}

function ThumbnailMosaic({ videos }: { videos: VideoThumbnail[] }) {
  const thumbnails = videos.slice(0, 4);
  const gridCols = thumbnails.length === 1 ? 1 : 2;

  return (
    <Box
      sx={{
        display: 'grid',
        gridTemplateColumns: `repeat(${gridCols}, 1fr)`,
        gap: 0.5,
        aspectRatio: '16/9',
        overflow: 'hidden',
        borderRadius: 1.5,
        bgcolor: designTokens.colors.surface[0],
      }}
    >
      {thumbnails.map((video, index) => (
        <Box
          key={video.id}
          sx={{
            position: 'relative',
            overflow: 'hidden',
            ...(thumbnails.length === 3 && index === 0 && { gridRow: 'span 2' }),
          }}
        >
          {video.posterS3Key ? (
            /* eslint-disable-next-line @next/next/no-img-element -- dynamic API URL */
            <img
              src={getVideoStreamUrl(video.posterS3Key)}
              alt=""
              loading="lazy"
              style={{
                width: '100%',
                height: '100%',
                objectFit: 'cover',
                display: 'block',
              }}
            />
          ) : (
            <Box
              sx={{
                width: '100%',
                height: '100%',
                bgcolor: designTokens.colors.surface[1],
              }}
            />
          )}
        </Box>
      ))}
    </Box>
  );
}

export function SessionCard({
  name,
  videoCount,
  updatedAt,
  videos = [],
  onClick,
  onAddVideos,
  variant = 'default',
  sharedBy,
  highlightCount,
}: SessionCardProps) {
  const isEmpty = videoCount === 0;
  const isFeatured = variant === 'featured';
  const isShared = variant === 'shared';

  if (isEmpty && onAddVideos) {
    return (
      <Card
        sx={{
          bgcolor: designTokens.colors.surface[1],
          borderRadius: 2,
          border: '2px dashed',
          borderColor: 'divider',
          transition: 'all 0.25s cubic-bezier(0.4, 0, 0.2, 1)',
          cursor: 'pointer',
          '&:hover': {
            borderColor: 'primary.main',
            bgcolor: designTokens.colors.surface[2],
            '& .add-icon': {
              transform: 'scale(1.1)',
              bgcolor: 'rgba(255, 107, 74, 0.2)',
            },
          },
        }}
        onClick={onAddVideos}
      >
        <CardContent sx={{ p: 3, textAlign: 'center' }}>
          <Box
            className="add-icon"
            sx={{
              width: 48,
              height: 48,
              borderRadius: 2,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              bgcolor: 'rgba(255, 107, 74, 0.1)',
              mx: 'auto',
              mb: 2,
              transition: 'all 0.2s ease',
            }}
          >
            <AddIcon sx={{ color: 'primary.main', fontSize: 24 }} />
          </Box>
          <Typography variant="subtitle2" fontWeight={600} sx={{ mb: 0.5 }}>
            {name}
          </Typography>
          <Typography variant="caption" color="text.secondary">
            Add videos to get started
          </Typography>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card
      sx={{
        bgcolor: designTokens.colors.surface[1],
        borderRadius: isFeatured ? 3 : 2,
        overflow: 'hidden',
        border: isFeatured ? '2px solid' : '1px solid',
        borderColor: isFeatured ? 'primary.main' : isShared ? 'secondary.dark' : 'divider',
        transition: 'all 0.25s cubic-bezier(0.4, 0, 0.2, 1)',
        position: 'relative',
        '&:hover': {
          transform: 'translateY(-4px)',
          boxShadow: designTokens.shadows.lg,
          borderColor: isShared ? 'secondary.main' : 'primary.main',
          '& .arrow-indicator': {
            transform: 'translateX(4px)',
            opacity: 1,
          },
        },
        ...(isFeatured && {
          '&::before': {
            content: '""',
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            height: 4,
            background: designTokens.gradients.sunset,
            zIndex: 1,
          },
        }),
        ...(isShared && {
          '&::before': {
            content: '""',
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            height: 3,
            background: designTokens.gradients.secondary,
            zIndex: 1,
          },
        }),
      }}
    >
      <CardActionArea onClick={onClick}>
        <Box sx={{ p: 2 }}>
          {/* Thumbnail mosaic - only show when we have actual thumbnails */}
          {videos.length > 0 && <ThumbnailMosaic videos={videos} />}

          {/* Content */}
          <Box sx={{ mt: videos.length > 0 ? 2 : 0 }}>
            {/* Shared indicator */}
            {isShared && sharedBy && (
              <Stack direction="row" alignItems="center" spacing={0.75} sx={{ mb: 1 }}>
                <Box
                  sx={{
                    width: 20,
                    height: 20,
                    borderRadius: '50%',
                    bgcolor: 'rgba(0, 212, 170, 0.2)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}
                >
                  <PeopleIcon sx={{ fontSize: 12, color: 'secondary.main' }} />
                </Box>
                <Typography variant="caption" color="secondary.main" fontWeight={500}>
                  Shared by {sharedBy}
                </Typography>
              </Stack>
            )}

            {/* Title row */}
            <Stack direction="row" alignItems="center" justifyContent="space-between">
              <Typography
                variant={isFeatured ? 'h6' : 'subtitle1'}
                fontWeight={600}
                noWrap
                sx={{ flex: 1 }}
              >
                {name}
              </Typography>
              <ChevronRightIcon
                className="arrow-indicator"
                sx={{
                  color: 'text.disabled',
                  fontSize: 20,
                  opacity: 0.5,
                  transition: 'all 0.2s ease',
                }}
              />
            </Stack>

            {/* Meta info */}
            <Stack direction="row" spacing={1} sx={{ mt: 1 }}>
              <Chip
                label={`${videoCount} video${videoCount !== 1 ? 's' : ''}`}
                size="small"
                sx={{
                  bgcolor: 'rgba(0, 212, 170, 0.15)',
                  color: 'secondary.main',
                  fontWeight: 500,
                  height: 22,
                  fontSize: '0.7rem',
                }}
              />
              {highlightCount !== undefined && highlightCount > 0 && (
                <Chip
                  label={`${highlightCount} highlight${highlightCount !== 1 ? 's' : ''}`}
                  size="small"
                  sx={{
                    bgcolor: designTokens.colors.surface[3],
                    fontWeight: 500,
                    height: 22,
                    fontSize: '0.7rem',
                  }}
                />
              )}
            </Stack>

            {/* Updated time */}
            {updatedAt && (
              <Typography variant="caption" color="text.disabled" sx={{ mt: 1.5, display: 'block' }}>
                {formatRelativeTime(updatedAt)}
              </Typography>
            )}
          </Box>
        </Box>
      </CardActionArea>
    </Card>
  );
}

export function SessionCardSkeleton() {
  return (
    <Card
      sx={{
        bgcolor: designTokens.colors.surface[1],
        borderRadius: 2,
        overflow: 'hidden',
        border: '1px solid',
        borderColor: 'divider',
      }}
    >
      <Box sx={{ p: 2 }}>
        <Skeleton variant="rectangular" sx={{ aspectRatio: '16/9', borderRadius: 1.5 }} />
        <Skeleton width="60%" sx={{ mt: 2 }} />
        <Skeleton width="40%" />
      </Box>
    </Card>
  );
}
