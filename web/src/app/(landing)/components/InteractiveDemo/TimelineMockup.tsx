'use client';

import { Box } from '@mui/material';
import { motion } from 'framer-motion';
import { designTokens } from '@/app/theme';

interface Rally {
  id: string;
  left: string;
  width: string;
  color: string;
  delay?: number;
}

interface TimelineMockupProps {
  rallies: Rally[];
  showScanEffect?: boolean;
  scanProgress?: number;
  highlightedRally?: string;
  animate?: boolean;
}

export function TimelineMockup({
  rallies,
  showScanEffect = false,
  scanProgress = 0,
  highlightedRally,
  animate = true,
}: TimelineMockupProps) {
  return (
    <Box
      sx={{
        height: 50,
        bgcolor: designTokens.colors.surface[0],
        borderRadius: 1.5,
        position: 'relative',
        overflow: 'hidden',
        border: '1px solid',
        borderColor: 'divider',
      }}
    >
      {/* Waveform background pattern */}
      <Box
        sx={{
          position: 'absolute',
          inset: 0,
          opacity: 0.15,
          backgroundImage: `repeating-linear-gradient(
            90deg,
            transparent,
            transparent 2px,
            rgba(255, 255, 255, 0.1) 2px,
            rgba(255, 255, 255, 0.1) 4px
          )`,
        }}
      />

      {/* AI Scanning effect */}
      {showScanEffect && (
        <motion.div
          initial={{ x: '-100%' }}
          animate={{ x: `${scanProgress * 100 - 100}%` }}
          transition={{ duration: 0.1, ease: 'linear' }}
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            background: 'linear-gradient(90deg, transparent 0%, rgba(255, 107, 74, 0.3) 50%, transparent 100%)',
            pointerEvents: 'none',
          }}
        />
      )}

      {/* Rally markers */}
      {rallies.map((rally) => (
        <motion.div
          key={rally.id}
          initial={animate ? { scaleX: 0, opacity: 0 } : false}
          animate={{ scaleX: 1, opacity: 1 }}
          transition={{
            delay: rally.delay || 0,
            duration: 0.3,
            ease: [0.22, 1, 0.36, 1],
          }}
          style={{
            position: 'absolute',
            left: rally.left,
            top: '15%',
            width: rally.width,
            height: '70%',
            background: rally.color,
            borderRadius: 4,
            transformOrigin: 'left center',
            boxShadow:
              highlightedRally === rally.id
                ? `0 0 12px ${rally.color}80`
                : 'none',
            border:
              highlightedRally === rally.id
                ? '2px solid white'
                : 'none',
          }}
        />
      ))}

      {/* Playhead */}
      <motion.div
        animate={{
          left: ['2%', '95%'],
        }}
        transition={{
          duration: 8,
          repeat: Infinity,
          ease: 'linear',
        }}
        style={{
          position: 'absolute',
          top: 0,
          width: 2,
          height: '100%',
          background: 'white',
          boxShadow: '0 0 8px rgba(255, 255, 255, 0.5)',
        }}
      />
    </Box>
  );
}
