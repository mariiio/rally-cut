'use client';

import { Box, Typography } from '@mui/material';
import { motion } from 'framer-motion';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import { designTokens } from '@/app/theme';

export function UploadStep() {
  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100%',
        minHeight: 280,
        position: 'relative',
      }}
    >
      {/* Drop zone */}
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.4 }}
        style={{ width: '100%', maxWidth: 400 }}
      >
        <Box
          sx={{
            border: '2px dashed',
            borderColor: 'primary.main',
            borderRadius: 3,
            p: 5,
            textAlign: 'center',
            bgcolor: 'rgba(255, 107, 74, 0.05)',
            position: 'relative',
            overflow: 'hidden',
          }}
        >
          {/* Animated file icon */}
          <motion.div
            initial={{ y: -60, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{
              delay: 0.3,
              duration: 0.6,
              type: 'spring',
              stiffness: 100,
            }}
          >
            <Box
              sx={{
                width: 64,
                height: 64,
                mx: 'auto',
                mb: 2,
                bgcolor: designTokens.colors.surface[2],
                borderRadius: 2,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                boxShadow: designTokens.shadows.lg,
              }}
            >
              <motion.div
                animate={{
                  y: [0, -5, 0],
                }}
                transition={{
                  duration: 2,
                  repeat: Infinity,
                  ease: 'easeInOut',
                }}
              >
                <CloudUploadIcon sx={{ fontSize: 32, color: 'primary.main' }} />
              </motion.div>
            </Box>
          </motion.div>

          {/* File name appearing */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.8 }}
          >
            <Typography
              variant="body2"
              sx={{
                color: 'text.primary',
                fontWeight: 500,
                mb: 1,
              }}
            >
              beach_volleyball_match.mp4
            </Typography>
            <Typography variant="caption" sx={{ color: 'text.secondary' }}>
              1.2 GB • 45:32
            </Typography>
          </motion.div>

          {/* Progress bar */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 1.2 }}
          >
            <Box
              sx={{
                mt: 3,
                height: 6,
                bgcolor: designTokens.colors.surface[2],
                borderRadius: 1,
                overflow: 'hidden',
              }}
            >
              <motion.div
                initial={{ width: '0%' }}
                animate={{ width: '100%' }}
                transition={{
                  delay: 1.4,
                  duration: 2,
                  ease: 'easeOut',
                }}
                style={{
                  height: '100%',
                  background: designTokens.gradients.primary,
                  borderRadius: 4,
                }}
              />
            </Box>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 3.4 }}
            >
              <Box
                sx={{
                  mt: 2,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: 1,
                }}
              >
                <motion.div
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{
                    delay: 3.5,
                    type: 'spring',
                    stiffness: 200,
                  }}
                >
                  <Box
                    sx={{
                      width: 20,
                      height: 20,
                      borderRadius: '50%',
                      bgcolor: 'secondary.main',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: 'white',
                      fontSize: '0.75rem',
                      fontWeight: 700,
                    }}
                  >
                    ✓
                  </Box>
                </motion.div>
                <Typography variant="caption" sx={{ color: 'secondary.main' }}>
                  Upload complete
                </Typography>
              </Box>
            </motion.div>
          </motion.div>
        </Box>
      </motion.div>
    </Box>
  );
}
