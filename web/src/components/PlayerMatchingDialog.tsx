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
import {
  getMatchAnalysis,
  getPlayerMatchingGtApi,
  savePlayerMatchingGtApi,
  getPlayerTrack,
  type PlayerPosition as ApiPlayerPosition,
} from '@/services/api';
import { useEditorStore } from '@/stores/editorStore';
import { usePlayerStore } from '@/stores/playerStore';

// Player colors matching PlayerNamingDialog + PlayerOverlay
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
  const [selectedCell, setSelectedCell] = useState<CellId | null>(null);
  const [loadingProgress, setLoadingProgress] = useState(0);
  const [totalRallies, setTotalRallies] = useState(0);
  const [isDirty, setIsDirty] = useState(false);
  const [saving, setSaving] = useState(false);
  const [sideSwitches, setSideSwitches] = useState<number[]>([]);
  const [untrackedCount, setUntrackedCount] = useState(0);
  const videoRef = useRef<HTMLVideoElement | null>(null);

  const rallies = useEditorStore((state) => state.rallies);
  const proxyUrl = useEditorStore((state) => state.proxyUrl);
  const videoUrl = useEditorStore((state) => state.videoUrl);
  const seek = usePlayerStore((state) => state.seek);

  const effectiveVideoUrl = proxyUrl || videoUrl;

  // Ref to hold initial assignments for crop extraction (avoids dependency cycle)
  const assignmentsRef = useRef<Record<string, Record<string, number>>>({});

  // Load match analysis + existing GT on open
  useEffect(() => {
    if (!open || !videoId) return;

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

        // Override with existing GT if available
        if (existingGt) {
          for (const [rid, mapping] of Object.entries(existingGt.rallies)) {
            if (initial[rid]) {
              initial[rid] = { ...mapping };
            }
          }
          setSideSwitches(existingGt.sideSwitches);
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
    video.crossOrigin = 'anonymous';
    video.preload = 'auto';
    video.muted = true;
    videoRef.current = video;

    const cropMap = new Map<string, string>();
    let untracked = 0;

    async function extractCrops() {
      // Wait for video metadata
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

        // Find matching editor rally: prefer exact ID match, fall back to time overlap
        const startSec = entry.startMs / 1000;
        const endSec = entry.endMs / 1000;
        const rally = rallies.find((r) => r._backendId === entry.rallyId)
          ?? rallies.find((r) => {
            // Match by time overlap (rally IDs may be stale after re-detection)
            const overlap = Math.min(r.end_time, endSec) - Math.max(r.start_time, startSec);
            return overlap > 0.5; // >0.5s overlap
          });

        // Use match analysis timing for seeking (always available), rally for backend ID
        const seekStart = rally?.start_time ?? startSec;
        const seekEnd = rally?.end_time ?? endSec;
        const backendRallyId = rally?._backendId;

        // Fetch tracking data from API (need a backend rally ID)
        let positions: ApiPlayerPosition[] = [];
        let rallyFps = 30;
        if (backendRallyId) {
          try {
            const trackResp = await getPlayerTrack(backendRallyId);
            positions = trackResp.positions ?? [];
            if (trackResp.fps) rallyFps = trackResp.fps;
          } catch {
            // No tracking data for this rally
          }
        }

        if (cancelled) break;
        if (positions.length === 0) {
          untracked++;
          setUntrackedCount(untracked);
          setLoadingProgress(i + 1);
          continue;
        }

        // Seek to mid-rally
        const midTime = (seekStart + seekEnd) / 2;
        video.currentTime = midTime;
        await new Promise<void>((resolve) => {
          video.onseeked = () => resolve();
        });

        if (cancelled) break;

        // Find frame closest to midTime (positions use rally-relative frameNumber)
        const midFrame = Math.round((midTime - seekStart) * rallyFps);

        // Group positions by trackId, find closest to midFrame
        const trackIds = new Set(positions.map((p) => p.trackId));
        const currentAssignments = assignmentsRef.current;

        for (const trackId of trackIds) {
          const trackPositions = positions.filter((p) => p.trackId === trackId);
          let bestPos = trackPositions[0];
          let bestDist = Infinity;
          for (const pos of trackPositions) {
            const dist = Math.abs(pos.frameNumber - midFrame);
            if (dist < bestDist) {
              bestDist = dist;
              bestPos = pos;
            }
          }

          if (!bestPos || bestDist > rallyFps) continue; // Skip if no position within ~1s

          // bbox coords are center-based normalized: x,y = center, width,height = full size
          const sx = (bestPos.x - bestPos.width / 2) * vw;
          const sy = (bestPos.y - bestPos.height / 2) * vh;
          const sw = bestPos.width * vw;
          const sh = bestPos.height * vh;

          ctx.clearRect(0, 0, CROP_WIDTH, CROP_HEIGHT);
          ctx.drawImage(video, sx, sy, sw, sh, 0, 0, CROP_WIDTH, CROP_HEIGHT);

          const dataUrl = canvas.toDataURL('image/jpeg', 0.8);

          // Key by rallyId + playerId (from assignment)
          const pid = currentAssignments[entry.rallyId]?.[String(trackId)];
          if (pid) {
            cropMap.set(cellKey(entry.rallyId, pid), dataUrl);
          }
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
  }, [normalizedRallies, effectiveVideoUrl, open, rallies]);

  // Handle cell click
  const handleCellClick = useCallback((rallyId: string, pid: number) => {
    if (!selectedCell) {
      // First click: select
      setSelectedCell({ rallyId, pid });
      return;
    }

    if (selectedCell.rallyId !== rallyId) {
      // Different row: reselect
      setSelectedCell({ rallyId, pid });
      return;
    }

    if (selectedCell.pid === pid) {
      // Same cell: deselect
      setSelectedCell(null);
      return;
    }

    // Same row, different column: swap
    const asgn = { ...assignments };
    const rowAssignment = { ...asgn[rallyId] };

    // Find which track IDs are assigned to these player IDs
    let trackA: string | null = null;
    let trackB: string | null = null;
    for (const [tid, p] of Object.entries(rowAssignment)) {
      if (p === selectedCell.pid) trackA = tid;
      if (p === pid) trackB = tid;
    }

    if (trackA !== null || trackB !== null) {
      // Swap: both assigned, or move one to the other column
      if (trackA !== null) rowAssignment[trackA] = pid;
      if (trackB !== null) rowAssignment[trackB] = selectedCell.pid;
      asgn[rallyId] = rowAssignment;
      assignmentsRef.current = asgn;
      setAssignments(asgn);

      // Swap crops too
      const newCrops = new Map(crops);
      const keyA = cellKey(rallyId, selectedCell.pid);
      const keyB = cellKey(rallyId, pid);
      const cropA = newCrops.get(keyA);
      const cropB = newCrops.get(keyB);
      if (cropA) newCrops.set(keyB, cropA); else newCrops.delete(keyB);
      if (cropB) newCrops.set(keyA, cropB); else newCrops.delete(keyA);
      setCrops(newCrops);

      setIsDirty(true);
    }

    setSelectedCell(null);
  }, [selectedCell, assignments, crops]);

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

  // Seek to rally
  const handleSeekToRally = useCallback((rallyId: string, startMs: number, endMs: number) => {
    const rally = rallies.find((r) => r._backendId === rallyId);
    const midTime = rally
      ? (rally.start_time + rally.end_time) / 2
      : (startMs + endMs) / 2000;
    seek(midTime);
  }, [rallies, seek]);

  // Save GT
  const handleSave = useCallback(async () => {
    setSaving(true);
    try {
      await savePlayerMatchingGtApi(videoId, {
        rallies: assignments,
        sideSwitches,
      });
      setIsDirty(false);
    } catch (err) {
      console.error('Failed to save player matching GT:', err);
    } finally {
      setSaving(false);
    }
  }, [videoId, assignments, sideSwitches]);

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
    <Dialog fullScreen open={open} onClose={onClose}>
      <AppBar sx={{ position: 'relative', bgcolor: '#1a1a2e' }}>
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

      <Box sx={{ p: 2, bgcolor: '#0f0f1a', minHeight: '100vh', overflow: 'auto' }}>
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
          const isSideSwitch = sideSwitches.includes(rallyIndex);
          const rowAssignment = assignments[entry.rallyId] || {};

          return (
            <Box key={entry.rallyId}>
              {/* Side switch divider */}
              {isSideSwitch && (
                <Divider sx={{ my: 1, borderColor: '#ff9800' }}>
                  <Chip
                    label="Side Switch"
                    size="small"
                    sx={{ bgcolor: '#ff9800', color: 'white', cursor: 'pointer' }}
                    onClick={() => handleToggleSideSwitch(rallyIndex)}
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
                  '&:hover': { bgcolor: 'rgba(255,255,255,0.03)' },
                }}
              >
                {/* Row label */}
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Tooltip title="Seek to rally">
                    <IconButton
                      size="small"
                      onClick={() => handleSeekToRally(entry.rallyId, entry.startMs, entry.endMs)}
                      sx={{ color: 'grey.500' }}
                    >
                      <PlayArrowIcon fontSize="small" />
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
                      onClick={() => handleToggleSideSwitch(rallyIndex)}
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
                          style={{ width: CROP_WIDTH, height: CROP_HEIGHT, objectFit: 'cover' }}
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
