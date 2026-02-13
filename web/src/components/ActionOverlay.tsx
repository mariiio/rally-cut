'use client';

import { useEffect, useRef, RefObject, useMemo } from 'react';
import type { ActionsData } from '@/services/api';

// Action colors matching volleyball semantics
const ACTION_COLORS: Record<string, string> = {
  serve: '#4CAF50',     // Green
  receive: '#2196F3',   // Blue
  set: '#FFC107',       // Amber
  spike: '#f44336',     // Red
  block: '#9C27B0',     // Purple
  dig: '#FF9800',       // Orange
  unknown: '#9e9e9e',   // Grey
};

interface ActionOverlayProps {
  actions: ActionsData;
  frameCount: number;
  rallyStartTime: number;
  rallyEndTime: number;
  videoRef: RefObject<HTMLVideoElement | null>;
}

// How long (seconds) to show the action label after its contact frame
const LABEL_SHOW_DURATION = 1.0;
// How long before the contact frame to start fading in
const LABEL_FADE_IN = 0.15;

export function ActionOverlay({
  actions,
  frameCount,
  rallyStartTime,
  rallyEndTime,
  videoRef,
}: ActionOverlayProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const labelsRef = useRef<HTMLDivElement[]>([]);
  const rafIdRef = useRef<number | undefined>(undefined);

  // Pre-calculate absolute time for each action
  const actionsWithTime = useMemo(() => {
    const rallyDuration = rallyEndTime - rallyStartTime;
    const maxFrame = Math.max(1, frameCount - 1);

    return actions.actions.map((a) => ({
      ...a,
      absoluteTime: rallyStartTime + (a.frame / maxFrame) * rallyDuration,
    }));
  }, [actions, frameCount, rallyStartTime, rallyEndTime]);

  // Create label elements
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    container.innerHTML = '';
    labelsRef.current = [];

    for (const action of actionsWithTime) {
      const label = document.createElement('div');
      const color = ACTION_COLORS[action.action] || ACTION_COLORS.unknown;

      label.style.cssText = `
        position: absolute;
        transform: translate(-50%, -100%);
        pointer-events: none;
        will-change: transform, opacity;
        display: none;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: white;
        background-color: ${color};
        box-shadow: 0 2px 8px rgba(0,0,0,0.4);
        white-space: nowrap;
        z-index: 20;
      `;
      label.textContent = action.action;

      // Add player track ID badge if available
      if (action.playerTrackId >= 0) {
        const badge = document.createElement('span');
        badge.style.cssText = `
          margin-left: 4px;
          font-size: 10px;
          opacity: 0.8;
        `;
        badge.textContent = `#${action.playerTrackId}`;
        label.appendChild(badge);
      }

      container.appendChild(label);
      labelsRef.current.push(label);
    }
  }, [actionsWithTime]);

  // Animation loop
  useEffect(() => {
    const updateLabels = () => {
      const video = videoRef.current;
      if (!video || labelsRef.current.length === 0) {
        rafIdRef.current = requestAnimationFrame(updateLabels);
        return;
      }

      const videoTime = video.currentTime;

      for (let i = 0; i < labelsRef.current.length; i++) {
        const label = labelsRef.current[i];
        const action = actionsWithTime[i];
        if (!label || !action) continue;

        const timeSinceAction = videoTime - action.absoluteTime;

        // Show label from LABEL_FADE_IN before contact to LABEL_SHOW_DURATION after
        if (timeSinceAction >= -LABEL_FADE_IN && timeSinceAction < LABEL_SHOW_DURATION) {
          // Fade in
          let opacity = 1.0;
          if (timeSinceAction < 0) {
            opacity = 1 - Math.abs(timeSinceAction) / LABEL_FADE_IN;
          }
          // Fade out in last 0.3s
          if (timeSinceAction > LABEL_SHOW_DURATION - 0.3) {
            opacity = Math.max(0, (LABEL_SHOW_DURATION - timeSinceAction) / 0.3);
          }

          // Position label above the ball position, offset upward
          const offsetY = -25; // pixels above ball
          label.style.display = 'block';
          label.style.left = `${action.ballX * 100}%`;
          label.style.top = `calc(${action.ballY * 100}% + ${offsetY}px)`;
          label.style.opacity = String(opacity);
        } else {
          label.style.display = 'none';
        }
      }

      rafIdRef.current = requestAnimationFrame(updateLabels);
    };

    rafIdRef.current = requestAnimationFrame(updateLabels);

    return () => {
      if (rafIdRef.current) {
        cancelAnimationFrame(rafIdRef.current);
      }
    };
  }, [videoRef, actionsWithTime]);

  return (
    <div
      ref={containerRef}
      style={{
        position: 'absolute',
        inset: 0,
        pointerEvents: 'none',
        zIndex: 20,
      }}
    />
  );
}
