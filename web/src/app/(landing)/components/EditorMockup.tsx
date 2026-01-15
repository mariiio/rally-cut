'use client';

import { Box, Typography } from '@mui/material';
import { motion, useReducedMotion } from 'framer-motion';
import { designTokens } from '@/app/theme';

// Rally marker in the timeline
function RallyMarker({
  left,
  width,
  delay,
  color = '#FF6B4A',
}: {
  left: string;
  width: string;
  delay: number;
  color?: string;
}) {
  return (
    <motion.div
      initial={{ scaleX: 0, opacity: 0 }}
      animate={{ scaleX: 1, opacity: 1 }}
      transition={{
        delay,
        duration: 0.4,
        ease: [0.22, 1, 0.36, 1],
      }}
      style={{
        position: 'absolute',
        left,
        top: '50%',
        transform: 'translateY(-50%)',
        width,
        height: '60%',
        background: color,
        borderRadius: 4,
        transformOrigin: 'left center',
      }}
    />
  );
}

// Animated cursor selecting a rally
function AnimatedCursor() {
  return (
    <motion.div
      initial={{ opacity: 0, x: '15%', y: '40%' }}
      animate={{
        opacity: [0, 1, 1, 1, 0],
        x: ['15%', '22%', '22%', '22%', '22%'],
        y: ['40%', '88%', '88%', '88%', '88%'],
      }}
      transition={{
        duration: 4,
        delay: 2,
        repeat: Infinity,
        repeatDelay: 3,
        times: [0, 0.3, 0.4, 0.9, 1],
      }}
      style={{
        position: 'absolute',
        zIndex: 10,
        pointerEvents: 'none',
      }}
    >
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
        <path
          d="M5.5 3.21V20.8c0 .45.54.67.85.35l4.86-4.86a.5.5 0 0 1 .35-.15h6.87a.5.5 0 0 0 .35-.85L6.35 2.86a.5.5 0 0 0-.85.35z"
          fill="white"
          stroke="black"
          strokeWidth="1"
        />
      </svg>
    </motion.div>
  );
}

// Selection highlight effect - highlights the second rally marker
function SelectionHighlight() {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{
        opacity: [0, 0, 1, 1, 0],
        scale: [0.95, 0.95, 1, 1, 1],
      }}
      transition={{
        duration: 4,
        delay: 2,
        repeat: Infinity,
        repeatDelay: 3,
        times: [0, 0.35, 0.45, 0.9, 1],
      }}
      style={{
        position: 'absolute',
        left: '19.5%',
        bottom: '8%',
        width: '9%',
        height: '6%',
        border: '2px solid #00D4AA',
        borderRadius: 4,
        background: 'rgba(0, 212, 170, 0.15)',
        pointerEvents: 'none',
      }}
    />
  );
}

export function EditorMockup() {
  const shouldReduceMotion = useReducedMotion();

  const rallies = [
    { left: '5%', width: '12%', delay: 0.5, color: '#FF6B4A' },
    { left: '20%', width: '8%', delay: 0.7, color: '#FF6B4A' },
    { left: '32%', width: '15%', delay: 0.9, color: '#00D4AA' },
    { left: '52%', width: '10%', delay: 1.1, color: '#FF6B4A' },
    { left: '66%', width: '7%', delay: 1.3, color: '#FFD166' },
    { left: '78%', width: '12%', delay: 1.5, color: '#FF6B4A' },
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 30, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{
        duration: 0.8,
        delay: 0.3,
        ease: [0.22, 1, 0.36, 1],
      }}
      style={{ width: '100%' }}
    >
      <Box
        sx={{
          position: 'relative',
          width: '100%',
          aspectRatio: '16/10',
          borderRadius: 3,
          overflow: 'hidden',
          bgcolor: designTokens.colors.surface[1],
          border: '1px solid',
          borderColor: 'divider',
          boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.5)',
        }}
      >
        {/* Window chrome */}
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            gap: 1,
            px: 2,
            py: 1.5,
            bgcolor: designTokens.colors.surface[2],
            borderBottom: '1px solid',
            borderColor: 'divider',
          }}
        >
          {['#FF5F56', '#FFBD2E', '#27C93F'].map((color) => (
            <Box
              key={color}
              sx={{
                width: 10,
                height: 10,
                borderRadius: '50%',
                bgcolor: color,
              }}
            />
          ))}
          <Typography
            variant="caption"
            sx={{
              ml: 2,
              color: 'text.secondary',
              fontSize: '0.7rem',
            }}
          >
            RallyCut Editor
          </Typography>
        </Box>

        {/* Fake video area */}
        <Box
          sx={{
            height: '60%',
            bgcolor: designTokens.colors.surface[0],
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            position: 'relative',
            overflow: 'hidden',
            mx: 2,
            mt: 1.5,
            borderRadius: 1,
          }}
        >
          {/* Volleyball court placeholder */}
          <Box
            sx={{
              width: '80%',
              height: '70%',
              bgcolor: '#D4A574',
              borderRadius: 1,
              position: 'relative',
              opacity: 0.3,
            }}
          >
            {/* Court lines */}
            <Box
              sx={{
                position: 'absolute',
                top: '50%',
                left: 0,
                right: 0,
                height: 2,
                bgcolor: 'white',
              }}
            />
            <Box
              sx={{
                position: 'absolute',
                top: 0,
                bottom: 0,
                left: '50%',
                width: 2,
                bgcolor: 'white',
              }}
            />
          </Box>

          {/* Play button */}
          <Box
            sx={{
              position: 'absolute',
              width: 56,
              height: 56,
              borderRadius: '50%',
              bgcolor: 'rgba(255, 255, 255, 0.9)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: '0 4px 20px rgba(0, 0, 0, 0.3)',
            }}
          >
            <Box
              sx={{
                width: 0,
                height: 0,
                borderTop: '12px solid transparent',
                borderBottom: '12px solid transparent',
                borderLeft: '18px solid #0D0E12',
                ml: 0.5,
              }}
            />
          </Box>
        </Box>

        {/* Timeline area */}
        <Box
          sx={{
            height: 'calc(40% - 40px)',
            bgcolor: designTokens.colors.surface[2],
            px: 3,
            py: 2,
            position: 'relative',
          }}
        >
          {/* Timeline label */}
          <Typography
            variant="caption"
            sx={{
              color: 'text.secondary',
              fontSize: '0.65rem',
              mb: 1,
              display: 'block',
            }}
          >
            Timeline • 6 rallies detected
          </Typography>

          {/* Timeline track */}
          <Box
            sx={{
              height: 48,
              bgcolor: designTokens.colors.surface[0],
              borderRadius: 1,
              position: 'relative',
              overflow: 'hidden',
            }}
          >
            {/* Rally markers */}
            {!shouldReduceMotion &&
              rallies.map((rally, index) => <RallyMarker key={index} {...rally} />)}

            {/* Static fallback for reduced motion */}
            {shouldReduceMotion &&
              rallies.map((rally, index) => (
                <Box
                  key={index}
                  sx={{
                    position: 'absolute',
                    left: rally.left,
                    top: '50%',
                    transform: 'translateY(-50%)',
                    width: rally.width,
                    height: '60%',
                    bgcolor: rally.color,
                    borderRadius: 0.5,
                  }}
                />
              ))}
          </Box>

          {/* Time markers */}
          <Box
            sx={{
              display: 'flex',
              justifyContent: 'space-between',
              mt: 0.5,
              px: 0.5,
            }}
          >
            {['0:00', '5:00', '10:00', '15:00', '20:00'].map((time) => (
              <Typography key={time} variant="caption" sx={{ color: 'text.disabled', fontSize: '0.6rem' }}>
                {time}
              </Typography>
            ))}
          </Box>
        </Box>

        {/* Animated cursor and selection */}
        {!shouldReduceMotion && (
          <>
            <AnimatedCursor />
            <SelectionHighlight />
          </>
        )}
      </Box>
    </motion.div>
  );
}
