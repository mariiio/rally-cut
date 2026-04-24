'use client';

import { useEffect, useCallback, RefObject, useMemo } from 'react';
import { usePlayerTrackingStore } from '@/stores/playerTrackingStore';
import { usePlayerStore } from '@/stores/playerStore';
import { useEditorStore } from '@/stores/editorStore';
import type { ActionGroundTruthLabel, MatchAnalysis } from '@/services/api';
import { rallyMatchEntry } from '@/utils/gtLabelDisplay';

/**
 * Invert a rally's raw-trackId→canonical-pid mapping into pid→trackId using
 * `appliedFullMapping` only.
 *
 * We deliberately do NOT fall back to `trackToPlayer` here. `trackToPlayer`
 * carries Hungarian's canonical-pid output, which doesn't match what the
 * editor displays before `remap-track-ids` has run: pre-remap the editor
 * shows sort-order-based player numbers (see `labelingPlayerNumbers` in
 * VideoPlayer.tsx / `playerNumberMap` in PlayerTrackingToolbar.tsx), and
 * `trackToPlayer`'s canonical pids can disagree with that sort order. If we
 * inverted `trackToPlayer` here, pressing "Player 2" would store the raw id
 * of Hungarian's pid-2, which might render as P4 on the sort-order-based
 * badge — the exact symptom we hit on a retracked rally after match-players
 * ran but remap-track-ids had not.
 *
 * When `appliedFullMapping` is absent, the caller falls back to sort-order
 * inversion — the same source the display uses. Read path
 * (`resolveGtDisplayPid`) mirrors this: appliedFullMapping → playerNumberMap
 * (sort-order), never trackToPlayer.
 */
function buildPidToTrackId(
  rallyEntry: MatchAnalysis['rallies'][number] | undefined,
): Record<number, number> {
  const out: Record<number, number> = {};
  if (!rallyEntry) return out;
  const source = rallyEntry.appliedFullMapping;
  if (!source) return out;
  for (const [rawTidStr, pid] of Object.entries(source)) {
    const rawTid = Number(rawTidStr);
    if (!Number.isFinite(rawTid) || !Number.isFinite(pid)) continue;
    // First-wins on collisions — consistent with how `measure_relabel_lift.py`
    // canonicalizes via ttp.
    if (!(pid in out)) out[pid as number] = rawTid;
  }
  return out;
}

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
  // appliedFullMapping we need to anchor GT to raw BoT-SORT track ids.
  useEffect(() => {
    if (!isLabelingActions || !activeMatchId) return;
    if (matchAnalysis[activeMatchId]) return;
    void loadMatchAnalysis(activeMatchId);
  }, [isLabelingActions, activeMatchId, matchAnalysis, loadMatchAnalysis]);

  const currentRallyEntry = useMemo(() => {
    const analysis = activeMatchId ? matchAnalysis[activeMatchId] : undefined;
    return rallyMatchEntry(analysis, backendRallyId);
  }, [activeMatchId, backendRallyId, matchAnalysis]);

  const pidToTrackId = useMemo(
    () => buildPidToTrackId(currentRallyEntry),
    [currentRallyEntry],
  );

  /** Reverse a visible track id (= canonical pid post-remap, or raw id when
   *  no mapping is available) into the raw BoT-SORT id we want to store. */
  const resolveRawTrackId = useCallback(
    (visibleTrackId: number): number => {
      const mapped = pidToTrackId[visibleTrackId];
      return mapped ?? visibleTrackId;
    },
    [pidToTrackId],
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

      // Resolve "Player N" to a raw BoT-SORT track id via appliedFullMapping
      // (pid→trackId inversion). Fallback: sort-order index over visible
      // tracks, matching the pre-mapping behavior.
      let rawTrackId = pidToTrackId[num];
      if (rawTrackId === undefined) {
        const sortedTracks = [...trackData.tracks].sort((a, b) => a.trackId - b.trackId);
        const playerIndex = num - 1;
        if (playerIndex >= sortedTracks.length) return;
        rawTrackId = sortedTracks[playerIndex].trackId;
        console.warn(
          '[ActionLabelingMode] No appliedFullMapping for rally — storing ' +
          `visible trackId ${rawTrackId} as raw anchor for Player ${num}. ` +
          'Run the match-analysis pipeline to stabilize GT across re-runs.',
        );
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
  }, [isLabelingActions, backendRallyId, trackData, selectedRally, videoRef, addActionLabel, updateActionLabelPlayer, setIsLabelingActions, onLabelAdded, seek, pidToTrackId, resolveRawTrackId]);

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
