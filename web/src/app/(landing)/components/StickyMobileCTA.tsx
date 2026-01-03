'use client';

import { Box, Button } from '@mui/material';
import { motion, useScroll, useMotionValueEvent, useReducedMotion } from 'framer-motion';
import { useState } from 'react';
import Link from 'next/link';

export function StickyMobileCTA() {
  const { scrollY } = useScroll();
  const [isVisible, setIsVisible] = useState(false);
  const shouldReduceMotion = useReducedMotion();

  useMotionValueEvent(scrollY, 'change', (latest) => {
    // Show after scrolling past hero (roughly 500px)
    setIsVisible(latest > 500);
  });

  if (shouldReduceMotion) {
    if (!isVisible) return null;
    return (
      <Box
        sx={{
          position: 'fixed',
          bottom: 0,
          left: 0,
          right: 0,
          zIndex: 1000,
          display: { xs: 'block', md: 'none' },
          p: 2,
          pb: 'calc(16px + env(safe-area-inset-bottom))',
          bgcolor: 'rgba(13, 14, 18, 0.95)',
          backdropFilter: 'blur(12px)',
          borderTop: '1px solid',
          borderColor: 'divider',
        }}
      >
        <Button
          component={Link}
          href="/editor"
          variant="contained"
          fullWidth
          size="large"
          sx={{
            py: 1.5,
            fontWeight: 600,
            fontSize: '1rem',
          }}
        >
          Start Editing Free
        </Button>
      </Box>
    );
  }

  return (
    <motion.div
      initial={{ y: 100 }}
      animate={{ y: isVisible ? 0 : 100 }}
      transition={{ type: 'spring', stiffness: 300, damping: 30 }}
      style={{
        position: 'fixed',
        bottom: 0,
        left: 0,
        right: 0,
        zIndex: 1000,
      }}
    >
      <Box
        sx={{
          display: { xs: 'block', md: 'none' },
          p: 2,
          pb: 'calc(16px + env(safe-area-inset-bottom))',
          bgcolor: 'rgba(13, 14, 18, 0.95)',
          backdropFilter: 'blur(12px)',
          borderTop: '1px solid',
          borderColor: 'divider',
        }}
      >
        <Button
          component={Link}
          href="/editor"
          variant="contained"
          fullWidth
          size="large"
          sx={{
            py: 1.5,
            fontWeight: 600,
            fontSize: '1rem',
            boxShadow: '0 4px 20px rgba(255, 107, 74, 0.3)',
          }}
        >
          Start Editing Free
        </Button>
      </Box>
    </motion.div>
  );
}
