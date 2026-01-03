'use client';

import { Box, Typography } from '@mui/material';
import { motion, AnimatePresence, useReducedMotion } from 'framer-motion';
import { useState, useEffect } from 'react';
import SportsVolleyballIcon from '@mui/icons-material/SportsVolleyball';
import VideoLibraryIcon from '@mui/icons-material/VideoLibrary';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import ContentCutIcon from '@mui/icons-material/ContentCut';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';

interface VolleyballProgressProps {
  progress: number;
  size?: 'sm' | 'md' | 'lg';
  stepText?: string;
  showPercentage?: boolean;
}

const SIZES = { sm: 48, md: 80, lg: 120 } as const;

const ICONS = [
  { Icon: CloudUploadIcon, label: 'Uploading' },
  { Icon: SportsVolleyballIcon, label: 'Volleyball' },
  { Icon: AutoAwesomeIcon, label: 'AI' },
  { Icon: ContentCutIcon, label: 'Editing' },
  { Icon: VideoLibraryIcon, label: 'Video' },
];

export function VolleyballProgress({
  progress,
  size = 'md',
  stepText,
  showPercentage = true,
}: VolleyballProgressProps) {
  const shouldReduceMotion = useReducedMotion();
  const [iconIndex, setIconIndex] = useState(0);

  const dimension = SIZES[size];
  const strokeWidth = size === 'sm' ? 3 : size === 'md' ? 4 : 5;
  const radius = (dimension - strokeWidth * 2) / 2;
  const circumference = 2 * Math.PI * radius;
  const clampedProgress = Math.min(100, Math.max(0, progress));
  const strokeDashoffset = circumference - (clampedProgress / 100) * circumference;

  const fontSize = size === 'sm' ? 11 : size === 'md' ? 13 : 15;
  const iconSize = size === 'sm' ? 20 : size === 'md' ? 32 : 44;

  // Cycle through icons
  useEffect(() => {
    if (shouldReduceMotion) return;

    const interval = setInterval(() => {
      setIconIndex((prev) => (prev + 1) % ICONS.length);
    }, 2000);

    return () => clearInterval(interval);
  }, [shouldReduceMotion]);

  const CurrentIcon = ICONS[iconIndex].Icon;

  return (
    <Box
      role="progressbar"
      aria-valuenow={clampedProgress}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-label={stepText || 'Loading progress'}
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: 2,
      }}
    >
      {/* Progress ring with icon */}
      <Box sx={{ position: 'relative', width: dimension, height: dimension }}>
        {/* Background ring */}
        <svg
          width={dimension}
          height={dimension}
          style={{ position: 'absolute', top: 0, left: 0 }}
        >
          <circle
            cx={dimension / 2}
            cy={dimension / 2}
            r={radius}
            fill="none"
            stroke="rgba(255, 255, 255, 0.1)"
            strokeWidth={strokeWidth}
          />
        </svg>

        {/* Progress ring */}
        <svg
          width={dimension}
          height={dimension}
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            transform: 'rotate(-90deg)',
          }}
        >
          <motion.circle
            cx={dimension / 2}
            cy={dimension / 2}
            r={radius}
            fill="none"
            stroke="url(#progressGradient)"
            strokeWidth={strokeWidth}
            strokeLinecap="round"
            strokeDasharray={circumference}
            initial={{ strokeDashoffset: circumference }}
            animate={{ strokeDashoffset }}
            transition={shouldReduceMotion ? { duration: 0 } : { duration: 0.5, ease: 'easeOut' }}
          />
          <defs>
            <linearGradient id="progressGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#FF6B4A" />
              <stop offset="100%" stopColor="#FF8A6F" />
            </linearGradient>
          </defs>
        </svg>

        {/* Center icon */}
        <Box
          sx={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <AnimatePresence mode="wait">
            <motion.div
              key={iconIndex}
              initial={shouldReduceMotion ? {} : { opacity: 0, scale: 0.5, rotate: -30 }}
              animate={{ opacity: 1, scale: 1, rotate: 0 }}
              exit={shouldReduceMotion ? {} : { opacity: 0, scale: 0.5, rotate: 30 }}
              transition={{ duration: 0.3, ease: 'easeOut' }}
              style={{ display: 'flex' }}
            >
              <CurrentIcon
                sx={{
                  fontSize: iconSize,
                  color: 'primary.main',
                }}
              />
            </motion.div>
          </AnimatePresence>
        </Box>
      </Box>

      {/* Text */}
      {(stepText || showPercentage) && (
        <Box sx={{ textAlign: 'center' }}>
          {stepText && (
            <Typography
              sx={{
                color: 'text.secondary',
                fontSize,
                mb: showPercentage ? 0.5 : 0,
              }}
            >
              {stepText}
            </Typography>
          )}
          {showPercentage && (
            <Typography
              sx={{
                fontFamily: 'monospace',
                fontWeight: 600,
                color: 'text.primary',
                fontSize: fontSize + 2,
              }}
            >
              {clampedProgress}%
            </Typography>
          )}
        </Box>
      )}
    </Box>
  );
}
