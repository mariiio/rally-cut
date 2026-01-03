'use client';

import { Box, Typography, Stack } from '@mui/material';
import { motion } from 'framer-motion';
import MovieIcon from '@mui/icons-material/Movie';
import DownloadIcon from '@mui/icons-material/Download';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import { useState, useEffect } from 'react';
import { designTokens } from '@/app/theme';

const SELECTED_RALLIES = [
  { id: '1', left: '3%', width: '10%', color: '#00D4AA' },
  { id: '3', left: '26%', width: '12%', color: '#00D4AA' },
  { id: '6', left: '63%', width: '11%', color: '#00D4AA' },
  { id: '8', left: '90%', width: '7%', color: '#00D4AA' },
];

export function ExportStep() {
  const [exportProgress, setExportProgress] = useState(0);
  const [isComplete, setIsComplete] = useState(false);

  useEffect(() => {
    const startDelay = 800;
    const exportDuration = 2500;

    const timeout = setTimeout(() => {
      const startTime = Date.now();

      const progressInterval = setInterval(() => {
        const elapsed = Date.now() - startTime;
        const progress = Math.min(elapsed / exportDuration, 1);
        setExportProgress(progress);

        if (progress >= 1) {
          clearInterval(progressInterval);
          setIsComplete(true);
        }
      }, 50);

      return () => clearInterval(progressInterval);
    }, startDelay);

    return () => clearTimeout(timeout);
  }, []);

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100%',
        minHeight: 280,
      }}
    >
      {/* Selected rallies preview */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        style={{ width: '100%', maxWidth: 450 }}
      >
        <Box
          sx={{
            bgcolor: designTokens.colors.surface[1],
            borderRadius: 3,
            p: 3,
            border: '1px solid',
            borderColor: 'divider',
          }}
        >
          {/* Header */}
          <Stack direction="row" alignItems="center" spacing={2} sx={{ mb: 3 }}>
            <Box
              sx={{
                width: 40,
                height: 40,
                borderRadius: 2,
                bgcolor: 'rgba(0, 212, 170, 0.15)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <MovieIcon sx={{ color: 'secondary.main' }} />
            </Box>
            <Box>
              <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                Highlight Reel
              </Typography>
              <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                4 rallies selected • 8:24 total
              </Typography>
            </Box>
          </Stack>

          {/* Mini timeline showing selected rallies */}
          <Box
            sx={{
              height: 36,
              bgcolor: designTokens.colors.surface[0],
              borderRadius: 1.5,
              position: 'relative',
              overflow: 'hidden',
              mb: 3,
            }}
          >
            {SELECTED_RALLIES.map((rally, index) => (
              <motion.div
                key={rally.id}
                initial={{ opacity: 0, scale: 0 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{
                  delay: 0.1 + index * 0.1,
                  type: 'spring',
                  stiffness: 200,
                }}
                style={{
                  position: 'absolute',
                  left: rally.left,
                  top: '15%',
                  width: rally.width,
                  height: '70%',
                  background: rally.color,
                  borderRadius: 4,
                  boxShadow: `0 0 8px ${rally.color}60`,
                }}
              />
            ))}
          </Box>

          {/* Export progress / completion */}
          {!isComplete ? (
            <Box>
              <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 1 }}>
                <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                  Exporting...
                </Typography>
                <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                  {Math.round(exportProgress * 100)}%
                </Typography>
              </Stack>
              <Box
                sx={{
                  height: 6,
                  bgcolor: designTokens.colors.surface[2],
                  borderRadius: 1,
                  overflow: 'hidden',
                }}
              >
                <motion.div
                  style={{
                    height: '100%',
                    width: `${exportProgress * 100}%`,
                    background: designTokens.gradients.secondary,
                    borderRadius: 4,
                  }}
                />
              </Box>
            </Box>
          ) : (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ type: 'spring', stiffness: 200 }}
            >
              <Box
                sx={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  p: 2,
                  bgcolor: 'rgba(0, 212, 170, 0.1)',
                  borderRadius: 2,
                  border: '1px solid rgba(0, 212, 170, 0.2)',
                }}
              >
                <Stack direction="row" alignItems="center" spacing={1.5}>
                  <motion.div
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ type: 'spring', stiffness: 300, delay: 0.1 }}
                  >
                    <CheckCircleIcon sx={{ color: 'secondary.main' }} />
                  </motion.div>
                  <Box>
                    <Typography variant="body2" sx={{ fontWeight: 600 }}>
                      highlights_reel.mp4
                    </Typography>
                    <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                      Ready to download • 128 MB
                    </Typography>
                  </Box>
                </Stack>

                <motion.div
                  animate={{
                    y: [0, -3, 0],
                  }}
                  transition={{
                    duration: 1,
                    repeat: Infinity,
                    ease: 'easeInOut',
                  }}
                >
                  <Box
                    sx={{
                      width: 36,
                      height: 36,
                      borderRadius: 2,
                      bgcolor: 'secondary.main',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      cursor: 'pointer',
                      '&:hover': {
                        bgcolor: 'secondary.dark',
                      },
                    }}
                  >
                    <DownloadIcon sx={{ color: 'white', fontSize: 20 }} />
                  </Box>
                </motion.div>
              </Box>
            </motion.div>
          )}
        </Box>
      </motion.div>

      {/* Share options hint */}
      {isComplete && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
        >
          <Typography
            variant="caption"
            sx={{
              color: 'text.secondary',
              mt: 2,
              display: 'block',
              textAlign: 'center',
            }}
          >
            Share directly to Instagram, TikTok, or YouTube
          </Typography>
        </motion.div>
      )}
    </Box>
  );
}
