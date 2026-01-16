'use client';

import { Box } from '@mui/material';

interface CropMaskOverlayProps {
  aspectRatio: 'ORIGINAL' | 'VERTICAL';
  opacity?: number; // Default 0.5 (50% darkening)
}

/**
 * Semi-transparent overlay that darkens the excluded areas when in 9:16 mode.
 * The 9:16 window is always centered - the video pans underneath it.
 * This overlay shows what parts of the video will be cropped in the final export.
 */
export function CropMaskOverlay({
  aspectRatio,
  opacity = 0.5,
}: CropMaskOverlayProps) {
  // Only show overlay in VERTICAL (9:16) mode
  if (aspectRatio !== 'VERTICAL') {
    return null;
  }

  // The 9:16 window is always centered in the 16:9 container
  // Window width: (9/16) / (16/9) = ~31.64% of container width
  const windowWidth = (9 / 16) / (16 / 9); // ~0.3164

  // Window is centered, so equal margins on both sides
  // Each side: (1 - 0.3164) / 2 = ~34.18%
  const sideWidth = (1 - windowWidth) / 2;

  // Convert to percentages for CSS
  const leftOverlayWidth = sideWidth * 100;
  const rightOverlayWidth = sideWidth * 100;
  const windowLeftPosition = sideWidth * 100;

  return (
    <>
      {/* Left semi-transparent overlay */}
      <Box
        sx={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: `${leftOverlayWidth}%`,
          height: '100%',
          bgcolor: `rgba(0, 0, 0, ${opacity})`,
          pointerEvents: 'none',
          zIndex: 15, // Above video (which extends beyond its container)
        }}
      />

      {/* Right semi-transparent overlay */}
      <Box
        sx={{
          position: 'absolute',
          top: 0,
          right: 0,
          width: `${rightOverlayWidth}%`,
          height: '100%',
          bgcolor: `rgba(0, 0, 0, ${opacity})`,
          pointerEvents: 'none',
          zIndex: 15, // Above video (which extends beyond its container)
        }}
      />

      {/* Thin border around the 9:16 window for clarity */}
      <Box
        sx={{
          position: 'absolute',
          top: 0,
          left: `${windowLeftPosition}%`,
          width: `${windowWidth * 100}%`,
          height: '100%',
          border: '1px solid rgba(255, 255, 255, 0.3)',
          pointerEvents: 'none',
          zIndex: 16,
          boxSizing: 'border-box',
        }}
      />
    </>
  );
}
