'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import {
  Dialog,
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  Button,
  Box,
  Skeleton,
  Chip,
  Tooltip,
  Divider,
  CircularProgress,
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import SaveIcon from '@mui/icons-material/Save';
import DownloadIcon from '@mui/icons-material/Download';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import RemoveCircleOutlineIcon from '@mui/icons-material/RemoveCircleOutline';
import UndoIcon from '@mui/icons-material/Undo';
import {
  getMatchAnalysis,
  getPlayerMatchingGtApi,
  savePlayerMatchingGtApi,
  getPlayerTrack,
  type PlayerPosition as ApiPlayerPosition,
} from '@/services/api';
import { useEditorStore } from '@/stores/editorStore';
import { usePlayerStore } from '@/stores/playerStore';

// Player colors matching PlayerOverlay
const TRACK_COLORS = ['#4caf50', '#2196f3', '#ff9800', '#9c27b0'];
const PLAYER_LABELS = ['P1', 'P2', 'P3', 'P4'];

const CROP_WIDTH = 80;
const CROP_HEIGHT = 160;

interface PlayerMatchingDialogProps {
  open: boolean;
  videoId: string;
  onClose: () => void;
}

interface CellId {
  rallyId: string;
  pid: number;
}

function cellKey(rallyId: string, pid: number): string {
  return `${rallyId}_${pid}`;
}

// Match analysis JSON may use camelCase or snake_case keys depending on when it was written
interface RallyEntry {
  rallyId: string;
  rallyIndex: number;
  startMs: number;
  endMs: number;
  trackToPlayer: Record<string, number>;
  assignmentConfidence: number;
  sideSwitchDetected: boolean;
}

// IoU between two center-normalized bboxes (same convention as PlayerPosition).
function iou(
  a: { x: number; y: number; width: number; height: number },
  b: { x: number; y: number; width: number; height: number },
): number {
  const ax1 = a.x - a.width / 2, ay1 = a.y - a.height / 2;
  const ax2 = a.x + a.width / 2, ay2 = a.y + a.height / 2;
  const bx1 = b.x - b.width / 2, by1 = b.y - b.height / 2;
  const bx2 = b.x + b.width / 2, by2 = b.y + b.height / 2;
  const iw = Math.max(0, Math.min(ax2, bx2) - Math.max(ax1, bx1));
  const ih = Math.max(0, Math.min(ay2, by2) - Math.max(ay1, by1));
  const inter = iw * ih;
  if (inter <= 0) return 0;
  const area = a.width * a.height + b.width * b.height - inter;
  return area > 0 ? inter / area : 0;
}

const IOU_THRESHOLD = 0.3;
const ISOLATION_MAX_IOU = 0.1;

// Bbox-keyed GT label — mirror of gt_loader.py.
interface GtLabel {
  playerId: number;
  frame: number;
  cx: number;
  cy: number;
  w: number;
  h: number;
}

// Resolve GT labels to the {trackIdStr: playerId} runtime shape the
// dialog uses internally. Mirrors gt_loader._resolve_label: for each
// label, find the position at `frame` with the highest IoU against the
// label bbox; drop labels whose max IoU falls below the threshold.
function resolveLabelsToAssignment(
  labels: GtLabel[],
  positions: ApiPlayerPosition[],
): Record<string, number> {
  const out: Record<string, number> = {};
  for (const label of labels) {
    const labelBox = { x: label.cx, y: label.cy, width: label.w, height: label.h };
    let bestIou = 0;
    let bestTrack: number | null = null;
    for (const pos of positions) {
      if (pos.frameNumber !== label.frame) continue;
      const v = iou(labelBox, pos);
      if (v > bestIou) {
        bestIou = v;
        bestTrack = pos.trackId;
      }
    }
    if (bestTrack != null && bestIou >= IOU_THRESHOLD) {
      out[String(bestTrack)] = label.playerId;
    }
  }
  return out;
}

// Pick a frame for this track where its bbox is large AND has low IoU
// with every other track at that frame, so the label's bbox can be
// IoU-resolved unambiguously back to this track at load time.
function pickIsolatedFrame(
  positions: ApiPlayerPosition[],
  trackId: number,
): ApiPlayerPosition | null {
  const ownPositions = positions.filter((p) => p.trackId === trackId);
  if (ownPositions.length === 0) return null;

  const byFrame = new Map<number, ApiPlayerPosition[]>();
  for (const p of positions) {
    const list = byFrame.get(p.frameNumber);
    if (list) list.push(p);
    else byFrame.set(p.frameNumber, [p]);
  }

  let best: ApiPlayerPosition | null = null;
  let bestArea = -1;
  let fallback: ApiPlayerPosition | null = null;
  let fallbackArea = 0;

  for (const tpos of ownPositions) {
    const area = tpos.width * tpos.height;
    if (area > fallbackArea) {
      fallbackArea = area;
      fallback = tpos;
    }
    const peers = byFrame.get(tpos.frameNumber) ?? [];
    let maxPeerIou = 0;
    for (const peer of peers) {
      if (peer.trackId === trackId) continue;
      const v = iou(tpos, peer);
      if (v > maxPeerIou) maxPeerIou = v;
    }
    if (maxPeerIou >= ISOLATION_MAX_IOU) continue;
    if (area > bestArea) {
      bestArea = area;
      best = tpos;
    }
  }
  return best ?? fallback;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function normalizeRallyEntry(raw: any): RallyEntry {
  return {
    rallyId: raw.rallyId ?? raw.rally_id ?? '',
    rallyIndex: raw.rallyIndex ?? raw.rally_index ?? 0,
    startMs: raw.startMs ?? raw.start_ms ?? 0,
    endMs: raw.endMs ?? raw.end_ms ?? 0,
    trackToPlayer: raw.trackToPlayer ?? raw.track_to_player ?? {},
    assignmentConfidence: raw.assignmentConfidence ?? raw.assignment_confidence ?? 0,
    sideSwitchDetected: raw.sideSwitchDetected ?? raw.side_switch_detected ?? false,
  };
}

export function PlayerMatchingDialog({ open, videoId, onClose }: PlayerMatchingDialogProps) {
  const [normalizedRallies, setNormalizedRallies] = useState<RallyEntry[] | null>(null);
  const [assignments, setAssignments] = useState<Record<string, Record<string, number>>>({});
  const [crops, setCrops] = useState<Map<string, string>>(new Map());
  const [selectedCell, setSelectedCellState] = useState<CellId | null>(null);
  // Ref twin avoids stale closures in handleCellClick during async crop loading
  const selectedCellRef = useRef<CellId | null>(null);
  const setSelectedCell = useCallback((cell: CellId | null) => {
    selectedCellRef.current = cell;
    setSelectedCellState(cell);
  }, []);
  const [loadingProgress, setLoadingProgress] = useState(0);
  const [totalRallies, setTotalRallies] = useState(0);
  const [isDirty, setIsDirty] = useState(false);
  const [saving, setSaving] = useState(false);
  const [sideSwitches, setSideSwitches] = useState<number[]>([]);
  const [untrackedCount, setUntrackedCount] = useState(0);
  const [excludedRallies, setExcludedRallies] = useState<Set<string>>(new Set());
  const videoRef = useRef<HTMLVideoElement | null>(null);

  const rallies = useEditorStore((state) => state.rallies);
  const proxyUrl = useEditorStore((state) => state.proxyUrl);
  const videoUrl = useEditorStore((state) => state.videoUrl);
  const seek = usePlayerStore((state) => state.seek);

  const effectiveVideoUrl = proxyUrl || videoUrl;

  // Ref to hold initial assignments for crop extraction (avoids dependency cycle)
  const assignmentsRef = useRef<Record<string, Record<string, number>>>({});
  // Per-rally positions cache, populated by load (for v2 resolution) and by
  // crop extraction. Save path reads this to emit v2 labels with bbox anchors.
  const rallyPositionsRef = useRef<Map<string, ApiPlayerPosition[]>>(new Map());

  // Load match analysis + existing GT on open
  useEffect(() => {
    if (!open || !videoId) return;

    // Clear any positions cached from a previous dialog session — stale
    // entries would cause GT labels to resolve against the wrong video's
    // tracking data when the dialog is opened for a different video.
    rallyPositionsRef.current = new Map();

    let cancelled = false;

    async function load() {
      try {
        const [analysis, existingGt] = await Promise.all([
          getMatchAnalysis(videoId),
          getPlayerMatchingGtApi(videoId).catch(() => null),
        ]);

        if (cancelled) return;

        if (!analysis?.rallies) return;

        // Normalize rally entries to handle both camelCase and snake_case from DB
        const entries = analysis.rallies.map(normalizeRallyEntry);
        setNormalizedRallies(entries);

        // Build initial assignments from match analysis
        const initial: Record<string, Record<string, number>> = {};
        const switches: number[] = [];
        for (const entry of entries) {
          initial[entry.rallyId] = { ...entry.trackToPlayer };
          if (entry.sideSwitchDetected) {
            switches.push(entry.rallyIndex);
          }
        }

        // Override with existing GT if available. GT is bbox-keyed; we
        // fetch each rally's current positions in parallel and IoU-resolve
        // the labels to the current track ids the dialog uses internally.
        if (existingGt) {
          const entries = Object.entries(existingGt.rallies);
          await Promise.all(
            entries.map(async ([rid]) => {
              try {
                const track = await getPlayerTrack(rid);
                rallyPositionsRef.current.set(rid, track.positions ?? []);
              } catch (err) {
                console.warn(`[load] failed to fetch positions for GT rally ${rid}:`, err);
              }
            }),
          );
          if (cancelled) return;

          for (const [rid, entry] of entries) {
            if (!initial[rid]) continue;
            const positions = rallyPositionsRef.current.get(rid) ?? [];
            initial[rid] = resolveLabelsToAssignment(entry.labels, positions);
          }
          setSideSwitches(existingGt.sideSwitches);
          if (existingGt.excludedRallies) {
            setExcludedRallies(new Set(existingGt.excludedRallies));
          }
        } else {
          setSideSwitches(switches);
        }

        assignmentsRef.current = initial;
        setAssignments(initial);
        setTotalRallies(analysis.rallies.length);
      } catch (err) {
        console.error('Failed to load match analysis:', err);
      }
    }

    load();
    return () => { cancelled = true; };
  }, [open, videoId]);

  // Extract crops: fetch tracking data from API per rally, seek video, extract bbox crops
  useEffect(() => {
    if (!normalizedRallies || !effectiveVideoUrl || !open) return;

    let cancelled = false;
    const video = document.createElement('video');
    video.preload = 'auto';
    video.muted = true;
    videoRef.current = video;

    const cropMap = new Map<string, string>();
    let untracked = 0;

    async function extractCrops() {
      // Wait for video metadata — try without crossOrigin first (same-origin proxy)
      video.src = effectiveVideoUrl!;
      await new Promise<void>((resolve, reject) => {
        video.onloadedmetadata = () => resolve();
        video.onerror = () => reject(new Error('Video load failed'));
      });

      if (cancelled) return;

      const canvas = document.createElement('canvas');
      canvas.width = CROP_WIDTH;
      canvas.height = CROP_HEIGHT;
      const ctx = canvas.getContext('2d')!;
      const vw = video.videoWidth;
      const vh = video.videoHeight;

      for (let i = 0; i < normalizedRallies!.length; i++) {
        if (cancelled) break;
        const entry = normalizedRallies![i];

        // Always use backend startMs as seek origin — tracking frame numbers
        // are relative to the segment extracted at startMs, not the editor rally time
        const seekStart = entry.startMs / 1000;

        // Fetch tracking data from API
        let positions: ApiPlayerPosition[] = [];
        let rallyFps = 30;
        let primaryTrackIds: number[] | undefined;
        try {
          const trackResp = await getPlayerTrack(entry.rallyId);
          positions = trackResp.positions ?? [];
          // Cache positions so the save path can build v2 labels with bbox
          // anchors without re-fetching.
          rallyPositionsRef.current.set(entry.rallyId, positions);
          if (trackResp.fps) rallyFps = trackResp.fps;
          primaryTrackIds = trackResp.primaryTrackIds;
        } catch (err) {
          console.warn(`Failed to fetch tracking for rally ${entry.rallyId}:`, err);
        }

        if (cancelled) break;
        if (positions.length === 0) {
          untracked++;
          setUntrackedCount(untracked);
          setLoadingProgress(i + 1);
          continue;
        }

        const currentAssignments = assignmentsRef.current;
        const trackIds = new Set(positions.map((p) => p.trackId));
        const rallyAssignment = currentAssignments[entry.rallyId] ?? {};

        // Use primaryTrackIds to identify the 4 active players. These are
        // always reliable regardless of whether remap-track-ids has run.
        // The player ID assignment comes from match analysis or free-slot fallback.
        const usedPids = new Set<number>();
        const effectiveMapping: Record<number, number> = {};
        const primarySet = new Set(primaryTrackIds ?? []);

        // First pass: use match analysis mapping for primary tracks
        for (const [tidStr, pid] of Object.entries(rallyAssignment)) {
          const tid = Number(tidStr);
          if (primarySet.has(tid) && trackIds.has(tid) && !usedPids.has(pid)) {
            effectiveMapping[tid] = pid;
            usedPids.add(pid);
          }
        }
        // Second pass: assign remaining primary tracks to free player slots
        for (const tid of primarySet) {
          if (trackIds.has(tid) && !effectiveMapping[tid]) {
            for (let p = 1; p <= 4; p++) {
              if (!usedPids.has(p)) { effectiveMapping[tid] = p; usedPids.add(p); break; }
            }
          }
        }

        for (const trackId of primarySet) {
          if (!trackIds.has(trackId)) continue;
          const pid = effectiveMapping[trackId];
          if (!pid) continue;

          // Pick the frame where this player's bbox is largest (most visible).
          // For near-side players this is similar to midpoint; for far-side
          // players it finds the moment they're closest to the camera.
          const trackPositions = positions.filter((p) => p.trackId === trackId);
          let bestPos = trackPositions[0];
          let bestArea = 0;
          for (const pos of trackPositions) {
            const area = pos.width * pos.height;
            if (area > bestArea) {
              bestArea = area;
              bestPos = pos;
            }
          }

          if (!bestPos) continue;

          // Seek video to this track's best frame time
          const frameTimeSec = seekStart + bestPos.frameNumber / rallyFps;
          video.currentTime = frameTimeSec;
          await new Promise<void>((resolve) => {
            video.onseeked = () => resolve();
          });
          if (cancelled) break;

          // bbox coords are center-based normalized: x,y = center, width,height = full size
          // Ensure minimum crop size so far-side players are visible in context
          const MIN_CROP_W = 0.08; // ~100px on 1280w — enough to see a person
          const MIN_CROP_H = 0.28; // ~200px on 720h
          const bw = Math.max(bestPos.width * 1.15, MIN_CROP_W);
          const bh = Math.max(bestPos.height * 1.15, MIN_CROP_H);
          const sx = Math.max(0, (bestPos.x - bw / 2)) * vw;
          const sy = Math.max(0, (bestPos.y - bh / 2)) * vh;
          const sw = Math.min(bw * vw, vw - sx);
          const sh = Math.min(bh * vh, vh - sy);

          ctx.clearRect(0, 0, CROP_WIDTH, CROP_HEIGHT);
          ctx.drawImage(video, sx, sy, sw, sh, 0, 0, CROP_WIDTH, CROP_HEIGHT);

          const dataUrl = canvas.toDataURL('image/jpeg', 0.8);
          cropMap.set(cellKey(entry.rallyId, pid), dataUrl);
        }

        setLoadingProgress(i + 1);
        setCrops(new Map(cropMap));
      }
    }

    extractCrops().catch(console.error);

    return () => {
      cancelled = true;
      video.src = '';
      videoRef.current = null;
    };
  }, [normalizedRallies, effectiveVideoUrl, open]);

  const handleCellClick = useCallback((rallyId: string, pid: number) => {
    const sel = selectedCellRef.current;

    if (!sel || sel.rallyId !== rallyId) {
      setSelectedCell({ rallyId, pid });
      return;
    }

    if (sel.pid === pid) {
      setSelectedCell(null);
      return;
    }

    // Same row, different column: swap
    const currentAssignments = assignmentsRef.current;
    const asgn = { ...currentAssignments };
    const rowAssignment = { ...asgn[rallyId] };

    // Find which track IDs are assigned to these player IDs
    let trackA: string | null = null;
    let trackB: string | null = null;
    for (const [tid, p] of Object.entries(rowAssignment)) {
      if (p === sel.pid) trackA = tid;
      if (p === pid) trackB = tid;
    }

    if (trackA !== null || trackB !== null) {
      if (trackA !== null) rowAssignment[trackA] = pid;
      if (trackB !== null) rowAssignment[trackB] = sel.pid;
      asgn[rallyId] = rowAssignment;
      assignmentsRef.current = asgn;
      setAssignments(asgn);

      setCrops((prev) => {
        const newCrops = new Map(prev);
        const keyA = cellKey(rallyId, sel.pid);
        const keyB = cellKey(rallyId, pid);
        const cropA = newCrops.get(keyA);
        const cropB = newCrops.get(keyB);
        if (cropA) newCrops.set(keyB, cropA); else newCrops.delete(keyB);
        if (cropB) newCrops.set(keyA, cropB); else newCrops.delete(keyA);
        return newCrops;
      });

      setIsDirty(true);
    }

    setSelectedCell(null);
  }, [setSelectedCell]);

  // Toggle side switch
  const handleToggleSideSwitch = useCallback((rallyIndex: number) => {
    setSideSwitches((prev) => {
      const next = prev.includes(rallyIndex)
        ? prev.filter((i) => i !== rallyIndex)
        : [...prev, rallyIndex].sort((a, b) => a - b);
      setIsDirty(true);
      return next;
    });
  }, []);

  // Exclude/include rally from GT
  const handleToggleExclude = useCallback((rallyId: string) => {
    setExcludedRallies((prev) => {
      const next = new Set(prev);
      if (next.has(rallyId)) {
        next.delete(rallyId);
      } else {
        next.add(rallyId);
      }
      setIsDirty(true);
      return next;
    });
  }, []);

  // Seek to rally
  const handleSeekToRally = useCallback((rallyId: string, startMs: number, endMs: number) => {
    const rally = rallies.find((r) => r._backendId === rallyId);
    const midTime = rally
      ? (rally.start_time + rally.end_time) / 2
      : (startMs + endMs) / 2000;
    seek(midTime);
  }, [rallies, seek]);

  // Save GT — validates mappings before writing to prevent corrupt data
  const handleSave = useCallback(async () => {
    setSaving(true);
    try {
      const filteredAssignments = { ...assignments };
      for (const rid of excludedRallies) {
        delete filteredAssignments[rid];
      }

      // Validate: every rally must have exactly 4 entries with unique player IDs 1-4
      const warnings: string[] = [];
      for (const [rid, mapping] of Object.entries(filteredAssignments)) {
        const pids = Object.values(mapping);
        if (pids.length < 4) {
          warnings.push(`Rally ${rid.slice(0, 8)}: only ${pids.length}/4 players assigned`);
        }
        const uniquePids = new Set(pids);
        if (uniquePids.size !== pids.length) {
          warnings.push(`Rally ${rid.slice(0, 8)}: duplicate player IDs`);
        }
      }
      if (warnings.length > 0) {
        console.warn('[save] GT validation warnings:', warnings);
      }

      // Emit GT labels: each (trackId, playerId) assignment becomes a
      // bbox-keyed label anchored to an ISOLATED frame for that track —
      // one where the track's bbox has low IoU with every other track
      // present. This guarantees the label resolves unambiguously at
      // load time no matter how the tracking has been re-run since.
      const gtRallies: Record<string, { labels: GtLabel[] }> = {};
      const skipped: string[] = [];
      for (const [rid, mapping] of Object.entries(filteredAssignments)) {
        const positions = rallyPositionsRef.current.get(rid) ?? [];
        const labels: GtLabel[] = [];
        for (const [trackIdStr, playerId] of Object.entries(mapping)) {
          const trackId = Number(trackIdStr);
          const best = pickIsolatedFrame(positions, trackId);
          if (!best) {
            skipped.push(`${rid.slice(0, 8)}/track${trackId}`);
            continue;
          }
          labels.push({
            playerId,
            frame: best.frameNumber,
            cx: best.x,
            cy: best.y,
            w: best.width,
            h: best.height,
          });
        }
        if (labels.length > 0) {
          gtRallies[rid] = { labels };
        }
      }
      if (skipped.length > 0) {
        console.warn('[save] dropped labels with no resolvable bbox:', skipped);
      }

      await savePlayerMatchingGtApi(videoId, {
        rallies: gtRallies,
        sideSwitches,
        excludedRallies: [...excludedRallies],
      });
      setIsDirty(false);
    } catch (err) {
      console.error('Failed to save player matching GT:', err);
    } finally {
      setSaving(false);
    }
  }, [videoId, assignments, sideSwitches, excludedRallies]);

  // Export JSON
  const handleExport = useCallback(() => {
    const data = {
      video_id: videoId,
      notes: '',
      side_switches: sideSwitches,
      rallies: assignments,
    };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `match_gt_${videoId.slice(0, 8)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [videoId, assignments, sideSwitches]);

  if (!normalizedRallies) {
    return (
      <Dialog fullScreen open={open} onClose={onClose}>
        <AppBar sx={{ position: 'relative' }}>
          <Toolbar>
            <IconButton edge="start" color="inherit" onClick={onClose}><CloseIcon /></IconButton>
            <Typography sx={{ ml: 2, flex: 1 }} variant="h6">Label Player Matching</Typography>
          </Toolbar>
        </AppBar>
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
          <CircularProgress />
        </Box>
      </Dialog>
    );
  }

  const isLoading = loadingProgress < totalRallies;

  return (
    <Dialog fullScreen open={open} onClose={onClose} PaperProps={{ sx: { display: 'flex', flexDirection: 'column' } }}>
      <AppBar sx={{ position: 'sticky', top: 0, bgcolor: '#1a1a2e', zIndex: 1 }}>
        <Toolbar>
          <IconButton edge="start" color="inherit" onClick={onClose}><CloseIcon /></IconButton>
          <Typography sx={{ ml: 2, flex: 1 }} variant="h6">
            Label Player Matching
            {isLoading && (
              <Typography component="span" variant="body2" sx={{ ml: 2, opacity: 0.7 }}>
                Loading crops... {loadingProgress}/{totalRallies}
              </Typography>
            )}
          </Typography>
          <Tooltip title="Export JSON">
            <IconButton color="inherit" onClick={handleExport}><DownloadIcon /></IconButton>
          </Tooltip>
          <Button
            color="inherit"
            startIcon={saving ? <CircularProgress size={16} color="inherit" /> : <SaveIcon />}
            onClick={handleSave}
            disabled={saving || !isDirty}
            sx={{ ml: 1 }}
          >
            Save
          </Button>
        </Toolbar>
      </AppBar>

      <Box sx={{ p: 2, bgcolor: '#0f0f1a', flex: 1, overflow: 'auto' }}>
        {/* Instructions */}
        <Typography variant="body2" sx={{ color: 'grey.500', mb: 2 }}>
          Click a player crop to select it, then click another in the same row to swap assignments.
          Click the side-switch chip to toggle.
        </Typography>

        {/* Warning when tracking data is missing */}
        {!isLoading && untrackedCount > 0 && (
          <Box sx={{ mb: 2, p: 1.5, bgcolor: 'warning.dark', borderRadius: 1, opacity: 0.9 }}>
            <Typography variant="body2" sx={{ color: 'white' }}>
              {untrackedCount === totalRallies
                ? 'No rallies have tracking data. Run "Track All" first to generate player crops.'
                : `${untrackedCount}/${totalRallies} rallies have no tracking data (no crops).`}
            </Typography>
          </Box>
        )}

        {/* Column headers */}
        <Box sx={{ display: 'grid', gridTemplateColumns: `200px repeat(4, ${CROP_WIDTH + 16}px)`, gap: 1, mb: 1, alignItems: 'center' }}>
          <Box />
          {PLAYER_LABELS.map((label, i) => (
            <Box key={label} sx={{ textAlign: 'center' }}>
              <Chip
                label={label}
                size="small"
                sx={{
                  bgcolor: TRACK_COLORS[i],
                  color: 'white',
                  fontWeight: 'bold',
                }}
              />
            </Box>
          ))}
        </Box>

        {/* Rally rows */}
        {normalizedRallies.map((entry, rallyIndex) => {
          const startSec = entry.startMs / 1000;
          const endSec = entry.endMs / 1000;
          const rally = rallies.find((r) => r._backendId === entry.rallyId)
            ?? rallies.find((r) => {
              const overlap = Math.min(r.end_time, endSec) - Math.max(r.start_time, startSec);
              return overlap > 0.5;
            });
          const isSideSwitch = sideSwitches.includes(entry.rallyIndex);
          const rowAssignment = assignments[entry.rallyId] || {};
          const isExcluded = excludedRallies.has(entry.rallyId);

          return (
            <Box key={entry.rallyId}>
              {/* Side switch divider */}
              {isSideSwitch && (
                <Divider sx={{ my: 1, borderColor: '#ff9800' }}>
                  <Chip
                    label="Side Switch"
                    size="small"
                    sx={{ bgcolor: '#ff9800', color: 'white', cursor: 'pointer' }}
                    onClick={() => handleToggleSideSwitch(entry.rallyIndex)}
                  />
                </Divider>
              )}

              <Box
                sx={{
                  display: 'grid',
                  gridTemplateColumns: `200px repeat(4, ${CROP_WIDTH + 16}px)`,
                  gap: 1,
                  mb: 0.5,
                  alignItems: 'center',
                  py: 0.5,
                  opacity: isExcluded ? 0.3 : 1,
                  '&:hover': { bgcolor: 'rgba(255,255,255,0.03)' },
                }}
              >
                {/* Row label */}
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  <Tooltip title="Seek to rally">
                    <IconButton
                      size="small"
                      onClick={() => handleSeekToRally(entry.rallyId, entry.startMs, entry.endMs)}
                      sx={{ color: 'grey.500' }}
                    >
                      <PlayArrowIcon fontSize="small" />
                    </IconButton>
                  </Tooltip>
                  <Tooltip title={isExcluded ? 'Include in GT' : 'Exclude from GT'}>
                    <IconButton
                      size="small"
                      onClick={() => handleToggleExclude(entry.rallyId)}
                      sx={{ color: isExcluded ? 'warning.main' : 'grey.700' }}
                    >
                      {isExcluded ? <UndoIcon fontSize="small" /> : <RemoveCircleOutlineIcon fontSize="small" />}
                    </IconButton>
                  </Tooltip>
                  <Box>
                    <Typography variant="body2" sx={{ color: 'white', fontWeight: 500 }}>
                      R{rallyIndex + 1}
                      <Typography component="span" variant="caption" sx={{ color: 'grey.500', ml: 0.5 }}>
                        {entry.rallyId.slice(0, 8)}
                      </Typography>
                    </Typography>
                    <Typography variant="caption" sx={{ color: 'grey.600' }}>
                      {formatTime(rally?.start_time ?? entry.startMs / 1000)} - {formatTime(rally?.end_time ?? entry.endMs / 1000)}
                    </Typography>
                  </Box>
                  <Tooltip title="Toggle side switch before this rally">
                    <Chip
                      label={`${Math.round(entry.assignmentConfidence * 100)}%`}
                      size="small"
                      sx={{
                        ml: 'auto',
                        bgcolor: entry.assignmentConfidence > 0.8 ? 'success.dark' : entry.assignmentConfidence > 0.5 ? 'warning.dark' : 'error.dark',
                        color: 'white',
                        fontSize: '0.65rem',
                        height: 20,
                        cursor: 'pointer',
                      }}
                      onClick={() => handleToggleSideSwitch(entry.rallyIndex)}
                    />
                  </Tooltip>
                </Box>

                {/* Player cells (P1-P4) */}
                {[1, 2, 3, 4].map((pid) => {
                  const key = cellKey(entry.rallyId, pid);
                  const cropUrl = crops.get(key);
                  const isSelected = selectedCell?.rallyId === entry.rallyId && selectedCell?.pid === pid;
                  const hasAssignment = Object.values(rowAssignment).includes(pid);

                  return (
                    <Box
                      key={pid}
                      onClick={() => handleCellClick(entry.rallyId, pid)}
                      sx={{
                        width: CROP_WIDTH + 8,
                        height: CROP_HEIGHT + 8,
                        border: isSelected
                          ? `3px solid ${TRACK_COLORS[pid - 1]}`
                          : '2px solid transparent',
                        borderRadius: 1,
                        overflow: 'hidden',
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        bgcolor: 'rgba(255,255,255,0.05)',
                        transition: 'border-color 0.15s',
                        '&:hover': {
                          borderColor: isSelected ? TRACK_COLORS[pid - 1] : 'rgba(255,255,255,0.2)',
                        },
                      }}
                    >
                      {cropUrl ? (
                        /* eslint-disable-next-line @next/next/no-img-element -- data URLs from canvas */
                        <img
                          src={cropUrl}
                          alt={`R${rallyIndex + 1} P${pid}`}
                          draggable={false}
                          style={{ width: CROP_WIDTH, height: CROP_HEIGHT, objectFit: 'cover', pointerEvents: 'none' }}
                        />
                      ) : isLoading ? (
                        <Skeleton
                          variant="rectangular"
                          width={CROP_WIDTH}
                          height={CROP_HEIGHT}
                          sx={{ bgcolor: 'rgba(255,255,255,0.1)' }}
                        />
                      ) : hasAssignment ? (
                        <Typography variant="caption" sx={{ color: 'grey.600' }}>
                          No crop
                        </Typography>
                      ) : (
                        <Typography variant="caption" sx={{ color: 'grey.700' }}>
                          -
                        </Typography>
                      )}
                    </Box>
                  );
                })}
              </Box>
            </Box>
          );
        })}
      </Box>
    </Dialog>
  );
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}
