'use client';

import { useEffect, useCallback, RefObject } from 'react';
import { usePlayerTrackingStore } from '@/stores/playerTrackingStore';
import { usePlayerStore } from '@/stores/playerStore';
import { useEditorStore } from '@/stores/editorStore';
import type { ActionGroundTruthLabel, ActionGroundTruthInput } from '@/services/api';

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
  const updateActionLabelPlayer = usePlayerTrackingStore((s) => s.updateActionLabelPlayer);
  const setIsLabelingActions = usePlayerTrackingStore((s) => s.setIsLabelingActions);
  const playerTracks = usePlayerTrackingStore((s) => s.playerTracks);
  const seek = usePlayerStore((s) => s.seek);

  const selectedRallyId = useEditorStore((s) => s.selectedRallyId);
  const rallies = useEditorStore((s) => s.rallies);

  const selectedRally = rallies.find((r) => r.id === selectedRallyId);
  const backendRallyId = selectedRally?._backendId ?? null;
  const trackData = backendRallyId ? playerTracks[backendRallyId]?.tracksJson : null;
  // Note: post-remap, trackData.tracks[].trackId IS the canonical pid 1-4.
  // The labeler writes canonical pid as the GT row's trackId; the server
  // looks it up in positionsJson (canonical) and stores resolved_track_id
  // canonical. This keeps the entire flow canonical end-to-end and removes
  // the half-canonical state where a freshly-saved row's raw value got
  // mis-read as canonical by gtLabelDisplay (pre-2026-05-14 bug).

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

    // Player assignment: 1-4 overrides trackId on current frame's GT label
    const num = parseInt(e.key, 10);
    if (num >= 1 && num <= 4) {
      e.preventDefault();
      e.stopPropagation();
      const video = videoRef.current;
      if (!video) return;

      const rallyStart = selectedRally.start_time;
      const fps = trackData.fps;
      const timeInRally = Math.max(0, video.currentTime - rallyStart);
      const frame = Math.round(timeInRally * fps);

      // Read GT labels fresh from store to avoid stale closure after action key + number key back-to-back
      const gtLabels = usePlayerTrackingStore.getState().actionGroundTruth[backendRallyId] ?? [];
      const labelAtFrame = gtLabels.find(l => l.frame === frame);
      if (!labelAtFrame) return;

      // Send the canonical pid (1-4) as the trackId. The server save path
      // looks up bbox in positionsJson which holds canonical pids post-remap,
      // and reresolveVideoGtAgainstCanonical also writes canonical. Keeping
      // the value canonical end-to-end means the display (gtLabelDisplay)
      // can read resolved_track_id directly without routing through
      // appliedFullMapping — and avoids the half-canonical state where a
      // freshly-saved row's raw value gets mis-read as canonical.
      updateActionLabelPlayer(backendRallyId, frame, num);
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
    const fps = trackData.fps;

    const timeInRally = Math.max(0, video.currentTime - rallyStart);
    const frame = Math.round(timeInRally * fps);

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

    // Auto-detected trackId from nearest-to-ball heuristic. trackData.tracks
    // is sourced from positionsJson which is canonical post-remap, so
    // nearestTrackId is already a canonical pid — no conversion needed.
    const label: ActionGroundTruthInput = {
      frame,
      action,
      trackId: nearestTrackId >= 0 ? nearestTrackId : undefined,
      ballX,
      ballY,
    };

    addActionLabel(backendRallyId, label);
    onLabelAdded?.(action, frame);
  }, [isLabelingActions, backendRallyId, trackData, selectedRally, videoRef, addActionLabel, updateActionLabelPlayer, setIsLabelingActions, onLabelAdded, seek]);

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
