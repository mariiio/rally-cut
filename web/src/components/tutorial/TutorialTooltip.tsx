'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import { Box, Typography, Button, Stack } from '@mui/material';
import { useTutorial } from './TutorialProvider';

const TOOLTIP_OFFSET = 16;
const TOOLTIP_WIDTH = 280;

interface TutorialTooltipProps {
  targetRect: {
    top: number;
    left: number;
    width: number;
    height: number;
  };
}

export function TutorialTooltip({ targetRect }: TutorialTooltipProps) {
  const { currentStepData, currentStep, visibleSteps, nextStep, prevStep, skipTutorial } =
    useTutorial();
  const tooltipRef = useRef<HTMLDivElement>(null);
  const [position, setPosition] = useState({ top: 0, left: 0 });

  const calculatePosition = useCallback(() => {
    if (!currentStepData || !tooltipRef.current) return;

    const tooltip = tooltipRef.current.getBoundingClientRect();
    const vw = window.innerWidth;
    const vh = window.innerHeight;

    let top = 0;
    let left = 0;

    switch (currentStepData.placement) {
      case 'top':
        top = targetRect.top - tooltip.height - TOOLTIP_OFFSET;
        left = targetRect.left + targetRect.width / 2 - tooltip.width / 2;
        break;
      case 'bottom':
        top = targetRect.top + targetRect.height + TOOLTIP_OFFSET;
        left = targetRect.left + targetRect.width / 2 - tooltip.width / 2;
        break;
      case 'left':
        top = targetRect.top + targetRect.height / 2 - tooltip.height / 2;
        left = targetRect.left - tooltip.width - TOOLTIP_OFFSET;
        break;
      case 'right':
        top = targetRect.top + targetRect.height / 2 - tooltip.height / 2;
        left = targetRect.left + targetRect.width + TOOLTIP_OFFSET;
        break;
    }

    // Clamp to viewport
    setPosition({
      top: Math.max(16, Math.min(top, vh - tooltip.height - 16)),
      left: Math.max(16, Math.min(left, vw - tooltip.width - 16)),
    });
  }, [targetRect, currentStepData]);

  useEffect(() => {
    calculatePosition();
  }, [calculatePosition]);

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).matches('input, textarea')) return;

      if (e.key === 'Enter' || e.key === 'ArrowRight') {
        e.preventDefault();
        nextStep();
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault();
        prevStep();
      } else if (e.key === 'Escape') {
        e.preventDefault();
        skipTutorial();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [nextStep, prevStep, skipTutorial]);

  if (!currentStepData) return null;

  const isFirst = currentStep === 0;
  const isLast = currentStep === visibleSteps.length - 1;

  return (
    <Box
      ref={tooltipRef}
      sx={{
        position: 'fixed',
        top: position.top,
        left: position.left,
        width: TOOLTIP_WIDTH,
        bgcolor: '#252B38',
        borderRadius: 2,
        border: '1px solid rgba(255, 107, 74, 0.3)',
        boxShadow: '0 8px 32px rgba(0, 0, 0, 0.5)',
        zIndex: 1302,
        p: 2,
        transition: 'top 0.3s ease-out, left 0.3s ease-out',
      }}
    >
      <Typography variant="subtitle1" sx={{ fontWeight: 600, color: 'text.primary', mb: 0.5 }}>
        {currentStepData.title}
      </Typography>

      <Typography variant="body2" sx={{ color: 'text.secondary', mb: 2, lineHeight: 1.5 }}>
        {currentStepData.description}
      </Typography>

      {/* Step indicator dots */}
      <Stack direction="row" spacing={0.75} sx={{ mb: 2, justifyContent: 'center' }}>
        {visibleSteps.map((step, idx) => (
          <Box
            key={step.id}
            sx={{
              width: 8,
              height: 8,
              borderRadius: '50%',
              bgcolor: idx === currentStep ? 'primary.main' : 'rgba(255, 255, 255, 0.3)',
              transition: 'background-color 0.2s ease',
            }}
          />
        ))}
      </Stack>

      {/* Navigation buttons */}
      <Stack direction="row" justifyContent="space-between" alignItems="center">
        <Button
          variant="text"
          size="small"
          onClick={skipTutorial}
          sx={{
            color: 'text.secondary',
            textTransform: 'none',
            '&:hover': { bgcolor: 'rgba(255, 255, 255, 0.08)', color: 'text.primary' },
          }}
        >
          Skip
        </Button>

        <Stack direction="row" spacing={1}>
          {!isFirst && (
            <Button
              variant="text"
              size="small"
              onClick={prevStep}
              sx={{
                color: 'text.secondary',
                textTransform: 'none',
                '&:hover': { bgcolor: 'rgba(255, 255, 255, 0.08)', color: 'text.primary' },
              }}
            >
              Back
            </Button>
          )}
          <Button variant="contained" size="small" onClick={nextStep} sx={{ textTransform: 'none' }}>
            {isLast ? 'Done' : 'Next'}
          </Button>
        </Stack>
      </Stack>
    </Box>
  );
}
