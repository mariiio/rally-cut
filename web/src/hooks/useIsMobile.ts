'use client';

import { useState, useEffect } from 'react';
import { useMediaQuery } from '@mui/material';
import { designTokens } from '@/app/theme';

/**
 * Hook to detect if the current viewport is mobile (<640px)
 * Uses MUI's useMediaQuery with proper SSR handling to avoid hydration mismatch.
 *
 * On server and initial client render, returns false (desktop layout).
 * After hydration, returns the actual media query result.
 *
 * @returns boolean - true if viewport width is below mobile breakpoint
 */
export function useIsMobile(): boolean {
  const [mounted, setMounted] = useState(false);
  const isMobile = useMediaQuery(`(max-width: ${designTokens.mobile.breakpoint - 1}px)`);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- intentional: tracking mount state for SSR hydration
    setMounted(true);
  }, []);

  // Return false during SSR and initial hydration to match server render
  // After mount, return the actual media query result
  return mounted ? isMobile : false;
}
