'use client';

import { Box, Typography, Skeleton } from '@mui/material';
import VideoLibraryIcon from '@mui/icons-material/VideoLibrary';
import FolderIcon from '@mui/icons-material/Folder';
import StorageIcon from '@mui/icons-material/Storage';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import { designTokens } from '@/app/theme';

interface StatItemProps {
  icon: React.ReactNode;
  value: string | number;
  label: string;
  color: string;
  glowColor: string;
  loading?: boolean;
}

function StatItem({ icon, value, label, color, glowColor, loading }: StatItemProps) {
  return (
    <Box
      sx={{
        position: 'relative',
        display: 'flex',
        alignItems: 'center',
        gap: 2,
        px: 2.5,
        py: 2,
        bgcolor: designTokens.colors.surface[1],
        borderRadius: 2.5,
        border: '1px solid',
        borderColor: 'rgba(255, 255, 255, 0.06)',
        overflow: 'hidden',
        transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
        cursor: 'default',
        '&::before': {
          content: '""',
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          height: '2px',
          background: `linear-gradient(90deg, ${color}, transparent)`,
          opacity: 0.6,
        },
        '&::after': {
          content: '""',
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          background: `radial-gradient(circle at 0% 0%, ${glowColor} 0%, transparent 60%)`,
          opacity: 0,
          transition: 'opacity 0.3s ease',
          pointerEvents: 'none',
        },
        '&:hover': {
          borderColor: 'rgba(255, 255, 255, 0.12)',
          transform: 'translateY(-2px)',
          boxShadow: `0 8px 24px rgba(0, 0, 0, 0.3), 0 0 0 1px rgba(255, 255, 255, 0.05)`,
          '&::after': {
            opacity: 1,
          },
          '& .stat-icon': {
            transform: 'scale(1.1)',
          },
        },
      }}
    >
      {/* Icon */}
      <Box
        className="stat-icon"
        sx={{
          width: 44,
          height: 44,
          borderRadius: 2,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          background: `linear-gradient(135deg, ${color}20 0%, ${color}10 100%)`,
          border: `1px solid ${color}30`,
          flexShrink: 0,
          transition: 'transform 0.3s ease',
          '& svg': {
            fontSize: 22,
            color: color,
            filter: `drop-shadow(0 2px 4px ${color}40)`,
          },
        }}
      >
        {icon}
      </Box>

      {/* Content */}
      <Box sx={{ minWidth: 0, position: 'relative', zIndex: 1 }}>
        {loading ? (
          <>
            <Skeleton width={50} height={32} sx={{ bgcolor: 'rgba(255,255,255,0.1)' }} />
            <Skeleton width={70} height={18} sx={{ bgcolor: 'rgba(255,255,255,0.05)' }} />
          </>
        ) : (
          <>
            <Typography
              sx={{
                fontWeight: 700,
                fontSize: '1.5rem',
                lineHeight: 1.1,
                letterSpacing: '-0.02em',
                color: 'text.primary',
              }}
            >
              {value}
            </Typography>
            <Typography
              sx={{
                fontSize: '0.75rem',
                fontWeight: 500,
                color: 'text.secondary',
                textTransform: 'uppercase',
                letterSpacing: '0.05em',
                mt: 0.25,
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
      icon: <VideoLibraryIcon />,
      value: videoCount,
      label: 'Videos',
      color: '#FF6B4A',
      glowColor: 'rgba(255, 107, 74, 0.15)',
    },
    {
      icon: <FolderIcon />,
      value: sessionCount,
      label: 'Sessions',
      color: '#00D4AA',
      glowColor: 'rgba(0, 212, 170, 0.15)',
    },
    ...(storageUsed !== undefined
      ? [
          {
            icon: <StorageIcon />,
            value: storageUsed,
            label: 'Storage',
            color: '#FFD166',
            glowColor: 'rgba(255, 209, 102, 0.15)',
          },
        ]
      : []),
    ...(aiCredits !== undefined
      ? [
          {
            icon: <AutoAwesomeIcon />,
            value: aiCredits,
            label: 'AI Credits',
            color: '#3B82F6',
            glowColor: 'rgba(59, 130, 246, 0.15)',
          },
        ]
      : []),
  ];

  return (
    <Box
      sx={{
        display: 'grid',
        gridTemplateColumns: {
          xs: '1fr 1fr',
          sm: `repeat(${stats.length}, 1fr)`,
        },
        gap: 2,
        mb: 4,
      }}
    >
      {stats.map((stat, index) => (
        <StatItem key={index} {...stat} loading={loading} />
      ))}
    </Box>
  );
}
