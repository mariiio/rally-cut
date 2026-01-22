'use client';

import { Box, Stack, Typography, Skeleton } from '@mui/material';
import VideoLibraryIcon from '@mui/icons-material/VideoLibrary';
import FolderIcon from '@mui/icons-material/Folder';
import StorageIcon from '@mui/icons-material/Storage';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import { designTokens } from '@/app/theme';

interface StatItemProps {
  icon: React.ReactNode;
  value: string | number;
  label: string;
  iconBg: string;
  loading?: boolean;
}

function StatItem({ icon, value, label, iconBg, loading }: StatItemProps) {
  return (
    <Box
      sx={{
        display: 'flex',
        alignItems: 'center',
        gap: 1.5,
        px: 2,
        py: 1.5,
        bgcolor: designTokens.colors.surface[1],
        borderRadius: 2,
        border: '1px solid',
        borderColor: 'divider',
        flex: 1,
        minWidth: 0,
        transition: 'all 0.2s ease',
        '&:hover': {
          borderColor: 'rgba(255, 255, 255, 0.12)',
          bgcolor: designTokens.colors.surface[2],
        },
      }}
    >
      <Box
        sx={{
          width: 36,
          height: 36,
          borderRadius: 1.5,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          bgcolor: iconBg,
          flexShrink: 0,
          '& svg': {
            fontSize: 20,
          },
        }}
      >
        {icon}
      </Box>
      <Box sx={{ minWidth: 0 }}>
        {loading ? (
          <>
            <Skeleton width={40} height={24} />
            <Skeleton width={60} height={16} />
          </>
        ) : (
          <>
            <Typography
              variant="h6"
              sx={{
                fontWeight: 700,
                lineHeight: 1.2,
                fontSize: '1.1rem',
              }}
            >
              {value}
            </Typography>
            <Typography
              variant="caption"
              color="text.secondary"
              sx={{
                display: 'block',
                lineHeight: 1.2,
                whiteSpace: 'nowrap',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
              }}
            >
              {label}
            </Typography>
          </>
        )}
      </Box>
    </Box>
  );
}

interface StatsBarProps {
  videoCount: number;
  sessionCount: number;
  storageUsed?: string;
  aiCredits?: number;
  loading?: boolean;
}

export function StatsBar({
  videoCount,
  sessionCount,
  storageUsed,
  aiCredits,
  loading,
}: StatsBarProps) {
  const stats = [
    {
      icon: <VideoLibraryIcon sx={{ color: designTokens.stats.iconColor.videos }} />,
      value: videoCount,
      label: 'Videos',
      iconBg: designTokens.stats.iconBg.videos,
    },
    {
      icon: <FolderIcon sx={{ color: designTokens.stats.iconColor.sessions }} />,
      value: sessionCount,
      label: 'Sessions',
      iconBg: designTokens.stats.iconBg.sessions,
    },
    ...(storageUsed !== undefined
      ? [
          {
            icon: <StorageIcon sx={{ color: designTokens.stats.iconColor.storage }} />,
            value: storageUsed,
            label: 'Storage',
            iconBg: designTokens.stats.iconBg.storage,
          },
        ]
      : []),
    ...(aiCredits !== undefined
      ? [
          {
            icon: <AutoAwesomeIcon sx={{ color: designTokens.stats.iconColor.credits }} />,
            value: aiCredits,
            label: 'AI Credits',
            iconBg: designTokens.stats.iconBg.credits,
          },
        ]
      : []),
  ];

  return (
    <Stack
      direction={{ xs: 'column', sm: 'row' }}
      spacing={1.5}
      sx={{
        mb: 4,
        display: 'grid',
        gridTemplateColumns: {
          xs: '1fr 1fr',
          sm: `repeat(${stats.length}, 1fr)`,
        },
        gap: 1.5,
      }}
    >
      {stats.map((stat, index) => (
        <StatItem key={index} {...stat} loading={loading} />
      ))}
    </Stack>
  );
}
