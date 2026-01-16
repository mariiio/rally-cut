'use client';

import { Box } from '@mui/material';

interface RotationGridOverlayProps {
  isVisible: boolean;
}

/**
 * Grid overlay shown while adjusting rotation.
 * Provides horizontal and vertical reference lines for alignment.
 */
export function RotationGridOverlay({ isVisible }: RotationGridOverlayProps) {
  if (!isVisible) return null;

  return (
    <Box
      sx={{
        position: 'absolute',
        inset: 0,
        pointerEvents: 'none',
        zIndex: 5,
      }}
    >
      {/* Horizontal lines */}
      <Box
        sx={{
          position: 'absolute',
          top: '33.33%',
          left: 0,
          right: 0,
          height: '1px',
          bgcolor: 'rgba(255, 255, 255, 0.4)',
        }}
      />
      <Box
        sx={{
          position: 'absolute',
          top: '50%',
          left: 0,
          right: 0,
          height: '1px',
          bgcolor: 'rgba(255, 255, 255, 0.6)',
        }}
      />
      <Box
        sx={{
          position: 'absolute',
          top: '66.67%',
          left: 0,
          right: 0,
          height: '1px',
          bgcolor: 'rgba(255, 255, 255, 0.4)',
        }}
      />
      {/* Vertical lines */}
      <Box
        sx={{
          position: 'absolute',
          left: '33.33%',
          top: 0,
          bottom: 0,
          width: '1px',
          bgcolor: 'rgba(255, 255, 255, 0.4)',
        }}
      />
      <Box
        sx={{
          position: 'absolute',
          left: '50%',
          top: 0,
          bottom: 0,
          width: '1px',
          bgcolor: 'rgba(255, 255, 255, 0.6)',
        }}
      />
      <Box
        sx={{
          position: 'absolute',
          left: '66.67%',
          top: 0,
          bottom: 0,
          width: '1px',
          bgcolor: 'rgba(255, 255, 255, 0.4)',
        }}
      />
    </Box>
  );
}
