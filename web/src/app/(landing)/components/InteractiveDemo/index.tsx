'use client';

import { Box } from '@mui/material';
import { AnimatePresence, motion } from 'framer-motion';
import { useState, useEffect, useCallback } from 'react';
import { DemoTabs } from './DemoTabs';
import { UploadStep } from './UploadStep';
import { DetectionStep } from './DetectionStep';
import { ExportStep } from './ExportStep';
import { designTokens } from '@/app/theme';

const STEP_DURATIONS = [4000, 5000, 4500]; // ms per step

export function InteractiveDemo() {
  const [activeStep, setActiveStep] = useState(0);
  const [isAutoPlaying, setIsAutoPlaying] = useState(true);
  const [key, setKey] = useState(0); // For resetting step animations

  const goToStep = useCallback((step: number) => {
    setActiveStep(step);
    setKey((k) => k + 1); // Reset animations
    setIsAutoPlaying(false); // Pause auto-play when user interacts
  }, []);

  // Auto-advance through steps
  useEffect(() => {
    if (!isAutoPlaying) return;

    const timer = setTimeout(() => {
      const nextStep = (activeStep + 1) % 3;
      setActiveStep(nextStep);
      setKey((k) => k + 1);

      // If we've cycled through all steps, pause for a moment then restart
      if (nextStep === 0) {
        setIsAutoPlaying(false);
        setTimeout(() => setIsAutoPlaying(true), 1500);
      }
    }, STEP_DURATIONS[activeStep]);

    return () => clearTimeout(timer);
  }, [activeStep, isAutoPlaying]);

  // Resume auto-play after inactivity
  useEffect(() => {
    if (isAutoPlaying) return;

    const resumeTimer = setTimeout(() => {
      setIsAutoPlaying(true);
    }, 8000); // Resume after 8 seconds of inactivity

    return () => clearTimeout(resumeTimer);
  }, [isAutoPlaying, activeStep]);

  return (
    <Box
      sx={{
        maxWidth: 600,
        mx: 'auto',
      }}
    >
      {/* Step tabs */}
      <DemoTabs
        activeStep={activeStep}
        onStepChange={goToStep}
        isAutoPlaying={isAutoPlaying}
      />

      {/* Step content */}
      <Box
        sx={{
          bgcolor: designTokens.colors.surface[1],
          borderRadius: 4,
          p: { xs: 2, sm: 4 },
          border: '1px solid',
          borderColor: 'divider',
          minHeight: 350,
          position: 'relative',
          overflow: 'hidden',
        }}
      >
        <AnimatePresence mode="wait">
          <motion.div
            key={`${activeStep}-${key}`}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            transition={{ duration: 0.3, ease: [0.22, 1, 0.36, 1] }}
            style={{ height: '100%' }}
          >
            {activeStep === 0 && <UploadStep />}
            {activeStep === 1 && <DetectionStep />}
            {activeStep === 2 && <ExportStep />}
          </motion.div>
        </AnimatePresence>
      </Box>

      {/* Pause/play hint */}
      <Box
        sx={{
          textAlign: 'center',
          mt: 2,
          opacity: 0.6,
          fontSize: '0.75rem',
          color: 'text.secondary',
        }}
      >
        {isAutoPlaying ? 'Auto-playing • Click a step to pause' : 'Click a step or wait to resume'}
      </Box>
    </Box>
  );
}
