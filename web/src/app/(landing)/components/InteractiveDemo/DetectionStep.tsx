'use client';

import { Box, Typography } from '@mui/material';
import { motion } from 'framer-motion';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import { useState, useEffect } from 'react';
import { TimelineMockup } from './TimelineMockup';
import { designTokens } from '@/app/theme';

const RALLIES = [
  { id: '1', left: '3%', width: '10%', color: '#FF6B4A', delay: 1.2 },
  { id: '2', left: '16%', width: '7%', color: '#FF6B4A', delay: 1.6 },
  { id: '3', left: '26%', width: '12%', color: '#00D4AA', delay: 2.0 },
  { id: '4', left: '42%', width: '8%', color: '#FF6B4A', delay: 2.4 },
  { id: '5', left: '54%', width: '6%', color: '#FFD166', delay: 2.8 },
  { id: '6', left: '63%', width: '11%', color: '#FF6B4A', delay: 3.2 },
  { id: '7', left: '78%', width: '9%', color: '#FF6B4A', delay: 3.6 },
  { id: '8', left: '90%', width: '7%', color: '#00D4AA', delay: 4.0 },
];

export function DetectionStep() {
  const [scanProgress, setScanProgress] = useState(0);
  const [ralliesFound, setRalliesFound] = useState(0);
  const [isComplete, setIsComplete] = useState(false);

  useEffect(() => {
    // Scanning animation
    const scanDuration = 3500;
    const startTime = Date.now();

    const scanInterval = setInterval(() => {
      const elapsed = Date.now() - startTime;
      const progress = Math.min(elapsed / scanDuration, 1);
      setScanProgress(progress);

      // Update rally count based on scan progress
      const foundCount = RALLIES.filter((r) => (r.delay || 0) < elapsed / 1000 + 0.5).length;
      setRalliesFound(foundCount);

      if (progress >= 1) {
        clearInterval(scanInterval);
        setIsComplete(true);
      }
    }, 50);

    return () => clearInterval(scanInterval);
  }, []);

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        minHeight: 280,
      }}
    >
      {/* Video preview area */}
      <Box
        sx={{
          flex: 1,
          bgcolor: designTokens.colors.surface[0],
          borderRadius: 2,
          position: 'relative',
          overflow: 'hidden',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          mb: 3,
        }}
      >
        {/* Fake video frame - volleyball court */}
        <Box
          sx={{
            width: '70%',
            height: '60%',
            bgcolor: '#C4A46B',
            borderRadius: 1,
            position: 'relative',
            opacity: 0.4,
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

        {/* AI Scanning overlay */}
        {!isComplete && (
          <motion.div
            animate={{
              opacity: [0.3, 0.6, 0.3],
            }}
            transition={{
              duration: 1.5,
              repeat: Infinity,
            }}
            style={{
              position: 'absolute',
              inset: 0,
              background: `linear-gradient(180deg, rgba(255, 107, 74, 0.1) 0%, transparent 50%, rgba(255, 107, 74, 0.1) 100%)`,
            }}
          />
        )}

        {/* AI badge */}
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.3 }}
          style={{
            position: 'absolute',
            top: 16,
            left: 16,
          }}
        >
          <Box
            sx={{
              display: 'flex',
              alignItems: 'center',
              gap: 0.75,
              px: 1.5,
              py: 0.75,
              bgcolor: 'rgba(255, 107, 74, 0.15)',
              borderRadius: 2,
              border: '1px solid rgba(255, 107, 74, 0.3)',
            }}
          >
            <motion.div
              animate={!isComplete ? { rotate: 360 } : {}}
              transition={{
                duration: 2,
                repeat: isComplete ? 0 : Infinity,
                ease: 'linear',
              }}
            >
              <AutoAwesomeIcon sx={{ fontSize: 16, color: 'primary.main' }} />
            </motion.div>
            <Typography variant="caption" sx={{ color: 'primary.light', fontWeight: 600 }}>
              {isComplete ? 'Analysis Complete' : 'AI Analyzing...'}
            </Typography>
          </Box>
        </motion.div>

        {/* Rally counter */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.8 }}
          style={{
            position: 'absolute',
            bottom: 16,
            right: 16,
          }}
        >
          <Box
            sx={{
              px: 2,
              py: 1,
              bgcolor: designTokens.colors.surface[2],
              borderRadius: 2,
              border: '1px solid',
              borderColor: 'divider',
            }}
          >
            <Typography variant="h5" sx={{ fontWeight: 700, color: 'text.primary' }}>
              <motion.span
                key={ralliesFound}
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
              >
                {ralliesFound}
              </motion.span>
            </Typography>
            <Typography variant="caption" sx={{ color: 'text.secondary' }}>
              rallies detected
            </Typography>
          </Box>
        </motion.div>
      </Box>

      {/* Timeline with detected rallies */}
      <TimelineMockup
        rallies={RALLIES}
        showScanEffect={!isComplete}
        scanProgress={scanProgress}
        animate={true}
      />

      {/* Progress text */}
      <Box sx={{ mt: 2, textAlign: 'center' }}>
        <Typography variant="caption" sx={{ color: 'text.secondary' }}>
          {isComplete
            ? '8 rallies found • 32:15 of action identified'
            : `Scanning video... ${Math.round(scanProgress * 100)}%`}
        </Typography>
      </Box>
    </Box>
  );
}
