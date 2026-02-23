'use client';

import { useEffect, useRef, RefObject, useMemo, useState, useCallback } from 'react';
import { Box, Popover, IconButton, Select, MenuItem, type SelectChangeEvent } from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import type { ActionsData, ActionGroundTruthLabel } from '@/services/api';

// Action colors matching volleyball semantics
const ACTION_COLORS: Record<string, string> = {
  serve: '#4CAF50',     // Green
  receive: '#2196F3',   // Blue
  set: '#FFC107',       // Amber
  attack: '#f44336',    // Red
  block: '#9C27B0',     // Purple
  dig: '#FF9800',       // Orange
  unknown: '#9e9e9e',   // Grey
};

const ACTION_TYPES = ['serve', 'receive', 'set', 'attack', 'block', 'dig'] as const;

interface ActionOverlayProps {
  actions: ActionsData;
  fps: number;
  rallyStartTime: number;
  videoRef: RefObject<HTMLVideoElement | null>;
  groundTruthLabels?: ActionGroundTruthLabel[];
  isLabelingMode?: boolean;
  onUpdateLabel?: (frame: number, action: ActionGroundTruthLabel['action']) => void;
  onDeleteLabel?: (frame: number) => void;
}

// How long (seconds) to show the action label after its contact frame
const LABEL_SHOW_DURATION = 1.0;
// How long before the contact frame to start fading in
const LABEL_FADE_IN = 0.15;

export function ActionOverlay({
  actions,
  fps,
  rallyStartTime,
  videoRef,
  groundTruthLabels,
  isLabelingMode,
  onUpdateLabel,
  onDeleteLabel,
}: ActionOverlayProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const labelsRef = useRef<HTMLDivElement[]>([]);
  const gtLabelsRef = useRef<HTMLDivElement[]>([]);
  const [editAnchor, setEditAnchor] = useState<HTMLDivElement | null>(null);
  const [editFrame, setEditFrame] = useState<number | null>(null);
  const [editAction, setEditAction] = useState<string>('');

  // Pre-calculate absolute time for each auto-detected action
  const actionsWithTime = useMemo(() => {
    return actions.actions.map((a) => ({
      ...a,
      absoluteTime: rallyStartTime + a.frame / fps,
    }));
  }, [actions, fps, rallyStartTime]);

  // Pre-calculate absolute time for GT labels
  const gtWithTime = useMemo(() => {
    if (!groundTruthLabels?.length) return [];

    return groundTruthLabels.map((l) => ({
      ...l,
      absoluteTime: rallyStartTime + l.frame / fps,
    }));
  }, [groundTruthLabels, fps, rallyStartTime]);

  const handleGtLabelClick = useCallback((e: MouseEvent) => {
    if (!isLabelingMode) return;
    const target = e.currentTarget as HTMLDivElement;
    const frame = parseInt(target.dataset.frame ?? '', 10);
    const action = target.dataset.action ?? '';
    if (!isNaN(frame)) {
      setEditAnchor(target);
      setEditFrame(frame);
      setEditAction(action);
    }
  }, [isLabelingMode]);

  // Create auto-detected label elements
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    // Clear all children
    container.innerHTML = '';
    labelsRef.current = [];
    gtLabelsRef.current = [];

    // Auto-detected actions
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

    // Ground truth labels
    const createdGtLabels: HTMLDivElement[] = [];
    for (const gt of gtWithTime) {
      const label = document.createElement('div');
      const color = ACTION_COLORS[gt.action] || ACTION_COLORS.unknown;

      label.style.cssText = `
        position: absolute;
        transform: translate(-50%, -100%);
        pointer-events: ${isLabelingMode ? 'auto' : 'none'};
        cursor: ${isLabelingMode ? 'pointer' : 'default'};
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
        border: 2px solid white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.5);
        white-space: nowrap;
        z-index: 21;
      `;

      // GT badge
      const gtBadge = document.createElement('span');
      gtBadge.style.cssText = `
        margin-left: 4px;
        font-size: 9px;
        background: rgba(255,255,255,0.3);
        padding: 0 3px;
        border-radius: 2px;
      `;
      gtBadge.textContent = 'GT';

      label.textContent = gt.action;
      label.appendChild(gtBadge);

      if (gt.playerTrackId >= 0) {
        const badge = document.createElement('span');
        badge.style.cssText = `
          margin-left: 4px;
          font-size: 10px;
          opacity: 0.8;
        `;
        badge.textContent = `#${gt.playerTrackId}`;
        label.appendChild(badge);
      }

      label.dataset.frame = String(gt.frame);
      label.dataset.action = gt.action;

      if (isLabelingMode) {
        label.addEventListener('click', handleGtLabelClick as EventListener);
      }

      container.appendChild(label);
      createdGtLabels.push(label);
    }
    gtLabelsRef.current = createdGtLabels;

    // Capture in closure for correct cleanup (gtLabelsRef.current is mutable)
    const clickHandler = handleGtLabelClick as EventListener;
    return () => {
      for (const label of createdGtLabels) {
        label.removeEventListener('click', clickHandler);
      }
    };
  }, [actionsWithTime, gtWithTime, isLabelingMode, handleGtLabelClick]);

  // Animation loop — uses requestVideoFrameCallback for frame-accurate sync
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    let rvfcId: number;

    const render = (videoTime: number) => {
      // Update auto-detected labels
      for (let i = 0; i < labelsRef.current.length; i++) {
        const label = labelsRef.current[i];
        const action = actionsWithTime[i];
        if (!label || !action) continue;

        const timeSinceAction = videoTime - action.absoluteTime;

        if (timeSinceAction >= -LABEL_FADE_IN && timeSinceAction < LABEL_SHOW_DURATION) {
          let opacity = 1.0;
          if (timeSinceAction < 0) {
            opacity = 1 - Math.abs(timeSinceAction) / LABEL_FADE_IN;
          }
          if (timeSinceAction > LABEL_SHOW_DURATION - 0.3) {
            opacity = Math.max(0, (LABEL_SHOW_DURATION - timeSinceAction) / 0.3);
          }

          const offsetY = -25;
          label.style.display = 'block';
          label.style.left = `${action.ballX * 100}%`;
          label.style.top = `calc(${action.ballY * 100}% + ${offsetY}px)`;
          label.style.opacity = String(opacity);
        } else {
          label.style.display = 'none';
        }
      }

      // Update GT labels — always visible in label mode, timed otherwise
      for (let i = 0; i < gtLabelsRef.current.length; i++) {
        const label = gtLabelsRef.current[i];
        const gt = gtWithTime[i];
        if (!label || !gt) continue;

        const timeSinceAction = videoTime - gt.absoluteTime;

        // In label mode, show GT labels with wider time window for easier editing
        const showDuration = isLabelingMode ? 2.0 : LABEL_SHOW_DURATION;
        const fadeIn = isLabelingMode ? 0.5 : LABEL_FADE_IN;

        if (timeSinceAction >= -fadeIn && timeSinceAction < showDuration) {
          let opacity = 1.0;
          if (timeSinceAction < 0) {
            opacity = 1 - Math.abs(timeSinceAction) / fadeIn;
          }
          if (timeSinceAction > showDuration - 0.3) {
            opacity = Math.max(0, (showDuration - timeSinceAction) / 0.3);
          }

          // Position GT labels above auto-detected ones
          const offsetY = -45;
          const x = gt.ballX ?? 0.5;
          const y = gt.ballY ?? 0.3;
          label.style.display = 'block';
          label.style.left = `${x * 100}%`;
          label.style.top = `calc(${y * 100}% + ${offsetY}px)`;
          label.style.opacity = String(opacity);
        } else {
          label.style.display = 'none';
        }
      }
    };

    const onFrame = (_now: DOMHighResTimeStamp, metadata: VideoFrameCallbackMetadata) => {
      render(metadata.mediaTime);
      rvfcId = video.requestVideoFrameCallback(onFrame);
    };

    // Initial render + start loop
    render(video.currentTime);
    rvfcId = video.requestVideoFrameCallback(onFrame);

    return () => {
      video.cancelVideoFrameCallback(rvfcId);
    };
  }, [videoRef, actionsWithTime, gtWithTime, isLabelingMode]);

  const handleEditClose = () => {
    setEditAnchor(null);
    setEditFrame(null);
    setEditAction('');
  };

  const handleActionChange = (e: SelectChangeEvent<string>) => {
    const newAction = e.target.value as ActionGroundTruthLabel['action'];
    if (editFrame !== null && onUpdateLabel) {
      onUpdateLabel(editFrame, newAction);
    }
    handleEditClose();
  };

  const handleDelete = () => {
    if (editFrame !== null && onDeleteLabel) {
      onDeleteLabel(editFrame);
    }
    handleEditClose();
  };

  return (
    <>
      <div
        ref={containerRef}
        style={{
          position: 'absolute',
          inset: 0,
          pointerEvents: 'none',
          zIndex: 20,
        }}
      />
      {/* Edit popover for GT labels */}
      <Popover
        open={!!editAnchor}
        anchorEl={editAnchor}
        onClose={handleEditClose}
        anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
        transformOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', p: 1, gap: 0.5 }}>
          <Select
            size="small"
            value={editAction}
            onChange={handleActionChange}
            sx={{ minWidth: 100, fontSize: '0.85rem' }}
          >
            {ACTION_TYPES.map(type => (
              <MenuItem key={type} value={type} sx={{ fontSize: '0.85rem' }}>
                {type.charAt(0).toUpperCase() + type.slice(1)}
              </MenuItem>
            ))}
          </Select>
          <IconButton size="small" onClick={handleDelete} color="error">
            <DeleteIcon fontSize="small" />
          </IconButton>
        </Box>
      </Popover>
    </>
  );
}
