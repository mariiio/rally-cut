'use client';

import { useEffect, useCallback, RefObject, useMemo } from 'react';
import { usePlayerTrackingStore } from '@/stores/playerTrackingStore';
import { usePlayerStore } from '@/stores/playerStore';
import { useEditorStore } from '@/stores/editorStore';
import type { ActionGroundTruthLabel } from '@/services/api';
import { canonicalRallyMapFor, pidToTrackId as resolveDisplayPidToTrackId } from '@/utils/canonicalPid';
import { rallyMatchEntry } from '@/utils/gtLabelDisplay';

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
  const matchAnalysis = usePlayerTrackingStore((s) => s.matchAnalysis);
  const loadMatchAnalysis = usePlayerTrackingStore((s) => s.loadMatchAnalysis);
  const seek = usePlayerStore((s) => s.seek);

  const selectedRallyId = useEditorStore((s) => s.selectedRallyId);
  const rallies = useEditorStore((s) => s.rallies);
  const activeMatchId = useEditorStore((s) => s.activeMatchId);

  const selectedRally = rallies.find((r) => r.id === selectedRallyId);
  const backendRallyId = selectedRally?._backendId ?? null;
  const trackData = backendRallyId ? playerTracks[backendRallyId]?.tracksJson : null;

  // Lazy-load match analysis when labeling begins — gives us the per-rally
  // canonicalPidMap (preferred) and appliedFullMapping (legacy fallback) we
  // need to anchor GT to raw BoT-SORT track ids.
  useEffect(() => {
    if (!isLabelingActions || !activeMatchId) return;
    if (matchAnalysis[activeMatchId]) return;
    void loadMatchAnalysis(activeMatchId);
  }, [isLabelingActions, activeMatchId, matchAnalysis, loadMatchAnalysis]);

  const currentAnalysis = activeMatchId ? matchAnalysis[activeMatchId] : undefined;
  const currentRallyEntry = useMemo(
    () => rallyMatchEntry(currentAnalysis, backendRallyId),
    [currentAnalysis, backendRallyId],
  );
  const currentCanonicalRallyMap = useMemo(
    () => canonicalRallyMapFor(currentAnalysis, backendRallyId),
    [currentAnalysis, backendRallyId],
  );

  const sortedTracks = useMemo(
    () => (trackData ? [...trackData.tracks].sort((a, b) => a.trackId - b.trackId) : []),
    [trackData],
  );
  const sortOrderMap = useMemo(() => {
    const m = new Map<number, number>();
    sortedTracks.forEach((t, idx) => m.set(t.trackId, idx + 1));
    return m;
  }, [sortedTracks]);

  /** Reverse a visible display pid (1-4) into the raw BoT-SORT id to anchor
   *  the GT row. Read-side priority is mirrored: canonical first, then
   *  legacy `appliedFullMapping`, then sort-order. */
  const resolveRawTrackIdForPid = useCallback(
    (displayPid: number): number | null =>
      resolveDisplayPidToTrackId(
        displayPid,
        currentCanonicalRallyMap,
        currentRallyEntry?.appliedFullMapping,
        sortOrderMap,
      ),
    [currentCanonicalRallyMap, currentRallyEntry, sortOrderMap],
  );

  /** Visible trackId could be either a canonical pid (post-remap rallies)
   *  or a raw BoT-SORT id (pre-remap). When it falls in {1..4} we invert
   *  through the same priority chain to recover the raw id; otherwise it's
   *  already raw. */
  const resolveRawTrackId = useCallback(
    (visibleTrackId: number): number => {
      if (visibleTrackId >= 1 && visibleTrackId <= 4) {
        const raw = resolveRawTrackIdForPid(visibleTrackId);
        if (raw !== null) return raw;
      }
      return visibleTrackId;
    },
    [resolveRawTrackIdForPid],
  );

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

      // Resolve "Player N" to a raw BoT-SORT track id. Priority:
      // canonicalPidMap (ref-crop sourced) → appliedFullMapping (legacy
      // Hungarian) → sort-order over visible tracks.
      const rawTrackId = resolveRawTrackIdForPid(num);
      if (rawTrackId === null) {
        console.warn(
          `[ActionLabelingMode] Could not resolve Player ${num} to a raw track id ` +
          'on this rally; skipping label.',
        );
        return;
      }
      updateActionLabelPlayer(backendRallyId, frame, rawTrackId);
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

    const label: ActionGroundTruthLabel = {
      frame,
      action,
      trackId: nearestTrackId >= 0 ? resolveRawTrackId(nearestTrackId) : undefined,
      ballX,
      ballY,
    };

    addActionLabel(backendRallyId, label);
    onLabelAdded?.(action, frame);
  }, [isLabelingActions, backendRallyId, trackData, selectedRally, videoRef, addActionLabel, updateActionLabelPlayer, setIsLabelingActions, onLabelAdded, seek, resolveRawTrackId, resolveRawTrackIdForPid]);

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
