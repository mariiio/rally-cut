'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { Box, Portal } from '@mui/material';
import { useTutorial } from './TutorialProvider';
import { TutorialTooltip } from './TutorialTooltip';

const SPOTLIGHT_PADDING = 8;
const SPOTLIGHT_BORDER_RADIUS = 8;
const MAX_RETRIES = 5;
const RETRY_DELAY = 200;

interface TargetRect {
  top: number;
  left: number;
  width: number;
  height: number;
}

export function TutorialOverlay() {
  const { currentStepData, isActive, nextStep, markStepUnavailable } = useTutorial();
  const [targetRect, setTargetRect] = useState<TargetRect | null>(null);
  const retryCountRef = useRef(0);
  const currentStepIdRef = useRef<string | null>(null);

  const measureTarget = useCallback((): boolean => {
    if (!currentStepData) {
      setTargetRect(null);
      return false;
    }

    const target = document.querySelector(`[data-tutorial="${currentStepData.id}"]`);
    if (!target) {
      setTargetRect(null);
      return false;
    }

    const rect = target.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) {
      setTargetRect(null);
      return false;
    }

    setTargetRect({
      top: rect.top - SPOTLIGHT_PADDING,
      left: rect.left - SPOTLIGHT_PADDING,
      width: rect.width + SPOTLIGHT_PADDING * 2,
      height: rect.height + SPOTLIGHT_PADDING * 2,
    });
    return true;
  }, [currentStepData]);

  // Find target element with retries
  useEffect(() => {
    if (!currentStepData) return;

    // Reset retry count when step changes
    if (currentStepIdRef.current !== currentStepData.id) {
      currentStepIdRef.current = currentStepData.id;
      retryCountRef.current = 0;
    }

    let timeoutId: ReturnType<typeof setTimeout> | null = null;

    const tryFind = () => {
      const found = measureTarget();

      if (!found && retryCountRef.current < MAX_RETRIES) {
        retryCountRef.current++;
        timeoutId = setTimeout(tryFind, RETRY_DELAY);
      } else if (!found) {
        // Element not found after retries - mark unavailable and advance
        markStepUnavailable(currentStepData.id);
        nextStep();
      }
    };

    tryFind();

    return () => {
      if (timeoutId) clearTimeout(timeoutId);
    };
  }, [currentStepData, measureTarget, markStepUnavailable, nextStep]);

  // Update position on resize/scroll
  useEffect(() => {
    if (!targetRect) return;

    const handleUpdate = () => measureTarget();
    window.addEventListener('resize', handleUpdate);
    window.addEventListener('scroll', handleUpdate, true);

    return () => {
      window.removeEventListener('resize', handleUpdate);
      window.removeEventListener('scroll', handleUpdate, true);
    };
  }, [targetRect, measureTarget]);

  if (!isActive || !currentStepData || !targetRect) {
    return null;
  }

  return (
    <Portal>
      {/* Backdrop with spotlight cutout */}
      <Box
        sx={{
          position: 'fixed',
          inset: 0,
          bgcolor: 'rgba(0, 0, 0, 0.6)',
          zIndex: 1300,
          pointerEvents: 'none',
          transition: 'clip-path 0.3s ease-out',
          clipPath: `polygon(
            0% 0%, 0% 100%,
            ${targetRect.left}px 100%,
            ${targetRect.left}px ${targetRect.top}px,
            ${targetRect.left + targetRect.width}px ${targetRect.top}px,
            ${targetRect.left + targetRect.width}px ${targetRect.top + targetRect.height}px,
            ${targetRect.left}px ${targetRect.top + targetRect.height}px,
            ${targetRect.left}px 100%,
            100% 100%, 100% 0%
          )`,
        }}
      />

      {/* Spotlight border */}
      <Box
        sx={{
          position: 'fixed',
          top: targetRect.top,
          left: targetRect.left,
          width: targetRect.width,
          height: targetRect.height,
          borderRadius: `${SPOTLIGHT_BORDER_RADIUS}px`,
          border: '2px solid',
          borderColor: 'primary.main',
          boxShadow: '0 0 0 4px rgba(255, 107, 74, 0.2)',
          zIndex: 1301,
          pointerEvents: 'none',
          transition: 'all 0.3s ease-out',
        }}
      />

      <TutorialTooltip targetRect={targetRect} />
    </Portal>
  );
}
