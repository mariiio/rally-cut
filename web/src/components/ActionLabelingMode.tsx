'use client';

import { useEffect, useCallback, RefObject } from 'react';
import { usePlayerTrackingStore } from '@/stores/playerTrackingStore';
import { usePlayerStore } from '@/stores/playerStore';
import { useEditorStore } from '@/stores/editorStore';
import type { ActionGroundTruthLabel } from '@/services/api';

const ACTION_KEYS: Record<string, ActionGroundTruthLabel['action']> = {
  s: 'serve',
  r: 'receive',
  t: 'set',
  a: 'attack',
  b: 'block',
  d: 'dig',
};

interface ActionLabelingModeProps {
  videoRef: RefObject<HTMLVideoElement | null>;
  onLabelAdded?: (action: string, frame: number) => void;
}

export function ActionLabelingMode({ videoRef, onLabelAdded }: ActionLabelingModeProps) {
  const isLabelingActions = usePlayerTrackingStore((s) => s.isLabelingActions);
  const addActionLabel = usePlayerTrackingStore((s) => s.addActionLabel);
  const setIsLabelingActions = usePlayerTrackingStore((s) => s.setIsLabelingActions);
  const playerTracks = usePlayerTrackingStore((s) => s.playerTracks);
  const seek = usePlayerStore((s) => s.seek);

  const selectedRallyId = useEditorStore((s) => s.selectedRallyId);
  const rallies = useEditorStore((s) => s.rallies);

  const selectedRally = rallies.find((r) => r.id === selectedRallyId);
  const backendRallyId = selectedRally?._backendId ?? null;
  const trackData = backendRallyId ? playerTracks[backendRallyId]?.tracksJson : null;

  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if (!isLabelingActions || !backendRallyId || !trackData || !selectedRally) return;

    // Don't capture if user is typing in an input
    const target = e.target as HTMLElement;
    if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.tagName === 'SELECT') return;

    // Escape exits label mode
    if (e.key === 'Escape') {
      setIsLabelingActions(false);
      return;
    }

    // Frame stepping: , = -1 frame, . = +1 frame
    if (e.key === ',' || e.key === '.') {
      e.preventDefault();
      e.stopPropagation();
      const video = videoRef.current;
      if (!video) return;
      const frameDuration = 1 / trackData.fps;
      const direction = e.key === ',' ? -1 : 1;
      seek(video.currentTime + direction * frameDuration);
      return;
    }

    const action = ACTION_KEYS[e.key.toLowerCase()];
    if (!action) return;

    e.preventDefault();
    e.stopPropagation();

    const video = videoRef.current;
    if (!video) return;

    // Calculate rally-relative frame from current video time
    const rallyStart = selectedRally.start_time;
    const rallyEnd = selectedRally.end_time;
    const rallyDuration = rallyEnd - rallyStart;
    const fps = trackData.fps;
    const maxFrame = Math.max(1, trackData.frameCount - 1);

    const timeInRally = Math.max(0, video.currentTime - rallyStart);
    const frame = Math.round((timeInRally / rallyDuration) * maxFrame);

    // Find ball position at this frame (from ball positions or actions data)
    let ballX: number | undefined;
    let ballY: number | undefined;
    if (trackData.ballPositions) {
      const ballPos = trackData.ballPositions.find(bp => bp.frameNumber === frame);
      if (ballPos) {
        ballX = ballPos.x;
        ballY = ballPos.y;
      }
    }

    // Find nearest player to ball position at this frame
    let nearestTrackId = -1;
    if (ballX !== undefined && ballY !== undefined) {
      let minDist = Infinity;
      for (const track of trackData.tracks) {
        const pos = track.positions.find(p => p.frame === frame)
          ?? track.positions.find(p => Math.abs(p.frame - frame) <= 2);
        if (!pos) continue;

        // Use upper-quarter of bbox (torso/arms area) for distance
        const playerX = pos.x + pos.w / 2;
        const playerY = pos.y + pos.h * 0.25;
        const dist = Math.hypot(playerX - ballX, playerY - ballY);
        if (dist < minDist) {
          minDist = dist;
          nearestTrackId = track.trackId;
        }
      }
    }

    const label: ActionGroundTruthLabel = {
      frame,
      action,
      playerTrackId: nearestTrackId,
      ballX,
      ballY,
    };

    addActionLabel(backendRallyId, label);
    onLabelAdded?.(action, frame);
  }, [isLabelingActions, backendRallyId, trackData, selectedRally, videoRef, addActionLabel, setIsLabelingActions, onLabelAdded, seek]);

  useEffect(() => {
    if (!isLabelingActions) return;

    // Use capture phase so we get the event before other handlers (like space for play/pause)
    window.addEventListener('keydown', handleKeyDown, true);
    return () => {
      window.removeEventListener('keydown', handleKeyDown, true);
    };
  }, [isLabelingActions, handleKeyDown]);

  // Component renders nothing — purely handles keyboard events
  return null;
}
