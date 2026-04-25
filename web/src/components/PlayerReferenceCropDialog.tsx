'use client';

import { useState, useEffect, useCallback, useMemo } from 'react';
import {
  Dialog,
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  Button,
  Box,
  Chip,
  Tooltip,
  CircularProgress,
  Popover,
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import DeleteIcon from '@mui/icons-material/Delete';
import RefreshIcon from '@mui/icons-material/Refresh';
import { useEditorStore } from '@/stores/editorStore';
import { usePlayerTrackingStore } from '@/stores/playerTrackingStore';
import {
  getMatchAnalysis,
  getPlayerTrack,
  runMatchAnalysis,
  validateReferenceCrops,
  type PlayerPosition as ApiPlayerPosition,
  type ReferenceCropValidationResult,
} from '@/services/api';

const TRACK_COLORS = ['#4caf50', '#2196f3', '#ff9800', '#9c27b0'];
const PLAYER_LABELS = ['P1', 'P2', 'P3', 'P4'];
const CROP_WIDTH = 80;
const CROP_HEIGHT = 160;

// How many rallies to sample crops from (spread across the match)
const MAX_SAMPLE_RALLIES = 6;

interface PlayerReferenceCropDialogProps {
  open: boolean;
  videoId: string;
  onClose: () => void;
}

/** A candidate crop extracted from the video, not yet assigned. */
interface CandidateCrop {
  id: string; // unique key
  trackId: number;
  rallyId: string;
  rallyIndex: number;
  dataUrl: string; // base64 image
  frameMs: number;
  bbox: { x: number; y: number; w: number; h: number };
}

export function PlayerReferenceCropDialog({ open, videoId, onClose }: PlayerReferenceCropDialogProps) {
  const [candidates, setCandidates] = useState<CandidateCrop[]>([]);
  const [loadingProgress, setLoadingProgress] = useState(0);
  const [totalToLoad, setTotalToLoad] = useState(0);
  const [anchorEl, setAnchorEl] = useState<HTMLElement | null>(null);
  const [pendingCandidate, setPendingCandidate] = useState<CandidateCrop | null>(null);
  const [saving, setSaving] = useState(false);
  const [rerunning, setRerunning] = useState(false);
  const [rerunStep, setRerunStep] = useState('');
  const [validation, setValidation] = useState<ReferenceCropValidationResult | null>(null);
  const [validating, setValidating] = useState(false);

  const proxyUrl = useEditorStore((state) => state.proxyUrl);
  const videoUrl = useEditorStore((state) => state.videoUrl);
  const effectiveVideoUrl = proxyUrl || videoUrl;

  const referenceCrops = usePlayerTrackingStore((state) => state.referenceCrops);
  const referenceCropsLoading = usePlayerTrackingStore((state) => state.referenceCropsLoading);
  const loadReferenceCrops = usePlayerTrackingStore((state) => state.loadReferenceCrops);
  const addReferenceCrop = usePlayerTrackingStore((state) => state.addReferenceCrop);
  const removeReferenceCrop = usePlayerTrackingStore((state) => state.removeReferenceCrop);
  const loadMatchAnalysis = usePlayerTrackingStore((state) => state.loadMatchAnalysis);

  // Load reference crops on open
  useEffect(() => {
    if (open && videoId) {
      loadReferenceCrops(videoId);
    }
  }, [open, videoId, loadReferenceCrops]);

  // Extract candidate crops from tracked rallies
  useEffect(() => {
    if (!open || !effectiveVideoUrl || !videoId) return;

    let cancelled = false;

    async function extractCandidates() {
      // Step 1: Get match analysis to find tracked rallies
      let analysis;
      try {
        analysis = await getMatchAnalysis(videoId);
      } catch {
        return;
      }
      if (cancelled || !analysis?.rallies?.length) return;

      // Pick rallies spread across the match
      const allRallies = analysis.rallies;
      const step = Math.max(1, Math.floor(allRallies.length / MAX_SAMPLE_RALLIES));
      const sampleIndices: number[] = [];
      for (let i = 0; i < allRallies.length && sampleIndices.length < MAX_SAMPLE_RALLIES; i += step) {
        sampleIndices.push(i);
      }

      setTotalToLoad(sampleIndices.length);
      setLoadingProgress(0);

      // Step 2: Create hidden video for crop extraction
      const video = document.createElement('video');
      video.preload = 'auto';
      video.muted = true;
      video.src = effectiveVideoUrl!;

      await new Promise<void>((resolve, reject) => {
        video.onloadedmetadata = () => resolve();
        video.onerror = () => reject(new Error('Video load failed'));
      });
      if (cancelled) { video.src = ''; return; }

      const canvas = document.createElement('canvas');
      canvas.width = CROP_WIDTH;
      canvas.height = CROP_HEIGHT;
      const ctx = canvas.getContext('2d')!;
      const vw = video.videoWidth;
      const vh = video.videoHeight;

      const newCandidates: CandidateCrop[] = [];

      // Step 3: For each sampled rally, fetch tracking and extract crops
      for (let si = 0; si < sampleIndices.length; si++) {
        if (cancelled) break;
        const rallyIdx = sampleIndices[si];
        const entry = allRallies[rallyIdx];
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const entryAny = entry as any;
        const rallyId: string = entryAny.rallyId ?? entryAny.rally_id ?? '';
        const startMs: number = entryAny.startMs ?? entryAny.start_ms ?? 0;

        // Fetch tracking data
        let positions: ApiPlayerPosition[] = [];
        let rallyFps = 30;
        let primaryTrackIds: number[] | undefined;
        try {
          const trackResp = await getPlayerTrack(rallyId);
          positions = trackResp.positions ?? [];
          if (trackResp.fps) rallyFps = trackResp.fps;
          primaryTrackIds = trackResp.primaryTrackIds;
        } catch {
          setLoadingProgress(si + 1);
          continue;
        }
        if (cancelled) break;
        if (positions.length === 0) {
          setLoadingProgress(si + 1);
          continue;
        }

        // Show one crop per primary player track (the 4 active players).
        // primaryTrackIds is always reliable regardless of remap state.
        const posTrackIds = new Set(positions.map((p) => p.trackId));
        const trackIdsToShow: number[] = [];
        if (primaryTrackIds) {
          for (const tid of primaryTrackIds) {
            if (posTrackIds.has(tid)) trackIdsToShow.push(tid);
          }
        }
        if (trackIdsToShow.length === 0) {
          // No primary IDs — take up to 4 unique tracks
          trackIdsToShow.push(...[...posTrackIds].slice(0, 4));
        }

        for (const trackId of trackIdsToShow) {
          if (cancelled) break;
          const trackPositions = positions.filter((p) => p.trackId === trackId);

          // Pick the mid-rally frame for this track (most representative).
          // Sort by frame number and take the middle position.
          const sorted = [...trackPositions].sort((a, b) => a.frameNumber - b.frameNumber);
          const bestPos = sorted[Math.floor(sorted.length / 2)];
          if (!bestPos) continue;

          // Seek video to this frame
          const frameTimeSec = startMs / 1000 + bestPos.frameNumber / rallyFps;
          video.currentTime = frameTimeSec;
          await new Promise<void>((resolve) => {
            video.onseeked = () => resolve();
          });
          if (cancelled) break;

          // Tight crop around the player bbox with 20% padding.
          // No minimum size — we want the player to fill the crop,
          // not a large background area with a tiny player.
          const bw = bestPos.width * 1.2;
          const bh = bestPos.height * 1.2;
          const sx = Math.max(0, (bestPos.x - bw / 2)) * vw;
          const sy = Math.max(0, (bestPos.y - bh / 2)) * vh;
          const sw = Math.min(bw * vw, vw - sx);
          const sh = Math.min(bh * vh, vh - sy);

          ctx.clearRect(0, 0, CROP_WIDTH, CROP_HEIGHT);
          ctx.drawImage(video, sx, sy, sw, sh, 0, 0, CROP_WIDTH, CROP_HEIGHT);

          const dataUrl = canvas.toDataURL('image/jpeg', 0.8);

          newCandidates.push({
            id: `${rallyId}_${trackId}`,
            trackId,
            rallyId,
            rallyIndex: rallyIdx,
            dataUrl,
            frameMs: Math.round(frameTimeSec * 1000),
            bbox: { x: bestPos.x, y: bestPos.y, w: bestPos.width, h: bestPos.height },
          });
        }

        setLoadingProgress(si + 1);
        setCandidates([...newCandidates]);
      }

      video.src = '';
    }

    extractCandidates().catch(console.error);

    return () => { cancelled = true; };
  }, [open, effectiveVideoUrl, videoId]);

  // Handle clicking a candidate crop to assign
  const handleCandidateClick = useCallback((event: React.MouseEvent<HTMLElement>, candidate: CandidateCrop) => {
    setPendingCandidate(candidate);
    setAnchorEl(event.currentTarget);
  }, []);

  // Assign candidate to player
  const handleAssignPlayer = useCallback(async (playerId: number) => {
    if (!pendingCandidate) return;
    setAnchorEl(null);
    setSaving(true);

    try {
      // Strip data:image/jpeg;base64, prefix
      const imageData = pendingCandidate.dataUrl.split(',')[1];

      await addReferenceCrop(videoId, playerId, pendingCandidate.frameMs, pendingCandidate.bbox, imageData);
    } catch (error) {
      console.error('Failed to save reference crop:', error);
    } finally {
      setSaving(false);
      setPendingCandidate(null);
    }
  }, [pendingCandidate, videoId, addReferenceCrop]);

  // Handle delete crop
  const handleDeleteCrop = useCallback(async (cropId: string) => {
    try {
      await removeReferenceCrop(videoId, cropId);
    } catch (error) {
      console.error('Failed to delete reference crop:', error);
    }
  }, [videoId, removeReferenceCrop]);

  // Handle re-run matching
  const handleRerunMatching = useCallback(async () => {
    setRerunning(true);
    setRerunStep('Starting...');
    try {
      await runMatchAnalysis(videoId, (progress) => {
        setRerunStep(progress.step || `Step ${progress.index}/${progress.total}`);
      });
      // Force-refresh the cached MatchAnalysis snapshot in the store —
      // without this the editor keeps rendering from the pre-rerun cache
      // and shows stale `appliedFullMapping` / `canonicalPidMap`. That's
      // the exact failure mode that flipped GT badges across re-runs.
      await loadMatchAnalysis(videoId, true);
      // Let other components (rally list confidence badge, match-stats)
      // refresh without prop-drilling a callback.
      window.dispatchEvent(new CustomEvent('match-analysis-updated', { detail: { videoId } }));
      setRerunStep('');
      setRerunning(false);
    } catch (error) {
      console.error('Failed to re-run matching:', error);
      setRerunStep('');
      setRerunning(false);
    }
  }, [videoId, loadMatchAnalysis]);

  const cropsByPlayer = useMemo(() => {
    const grouped: Record<number, typeof referenceCrops> = { 1: [], 2: [], 3: [], 4: [] };
    for (const crop of referenceCrops) {
      if (crop.playerId >= 1 && crop.playerId <= 4) {
        grouped[crop.playerId].push(crop);
      }
    }
    return grouped;
  }, [referenceCrops]);

  // Debounced pre-flight validator. Runs ~600 ms after any change to the
  // user's crop selection. The validator short-circuits immediately if a
  // player is missing crops (cheap SQL read) — only does DINOv2 prototype
  // math when all 4 players have coverage, so the slow path only fires
  // once the selection is complete.
  useEffect(() => {
    if (!open || !videoId || referenceCropsLoading) return;
    if (referenceCrops.length === 0) {
      setValidation(null);
      return;
    }

    let cancelled = false;
    setValidating(true);
    const timeout = setTimeout(() => {
      validateReferenceCrops(videoId)
        .then((result) => {
          if (!cancelled) setValidation(result);
        })
        .catch((err) => {
          console.error('Reference-crop validation failed:', err);
          if (!cancelled) setValidation(null);
        })
        .finally(() => {
          if (!cancelled) setValidating(false);
        });
    }, 600);

    return () => {
      cancelled = true;
      clearTimeout(timeout);
    };
  }, [open, videoId, referenceCrops, referenceCropsLoading]);

  const validationPassed = validation?.pass === true;
  const blockingIssues = useMemo(
    () => (validation?.issues ?? []).filter(i => i.code !== 'few_crops'),
    [validation],
  );
  const warnings = useMemo(
    () => (validation?.issues ?? []).filter(i => i.code === 'few_crops'),
    [validation],
  );

  const hasAnyCrops = referenceCrops.length > 0;
  const isLoading = totalToLoad > 0 && loadingProgress < totalToLoad;

  return (
    <Dialog
      fullScreen
      open={open}
      onClose={onClose}
      PaperProps={{ sx: { display: 'flex', flexDirection: 'column', bgcolor: '#0f0f1a' } }}
    >
      <AppBar sx={{ position: 'sticky', top: 0, bgcolor: '#1a1a2e', zIndex: 1 }}>
        <Toolbar>
          <IconButton edge="start" color="inherit" onClick={onClose}>
            <CloseIcon />
          </IconButton>
          <Typography sx={{ ml: 2, flex: 1 }} variant="h6">
            Player Reference Crops
            {isLoading && (
              <Typography component="span" variant="body2" sx={{ ml: 2, opacity: 0.7 }}>
                Loading crops... {loadingProgress}/{totalToLoad}
              </Typography>
            )}
          </Typography>
          {hasAnyCrops && (
            <>
              {rerunning && rerunStep && (
                <Typography variant="body2" sx={{ color: 'grey.400', mr: 1 }}>
                  {rerunStep}
                </Typography>
              )}
              {validating && !rerunning && (
                <Typography variant="body2" sx={{ color: 'grey.400', mr: 1 }}>
                  Validating crops...
                </Typography>
              )}
              <Tooltip
                title={
                  !validationPassed && blockingIssues.length > 0
                    ? `Fix ${blockingIssues.length} issue${blockingIssues.length === 1 ? '' : 's'} below before re-running`
                    : ''
                }
              >
                <span>
                  <Button
                    color="inherit"
                    startIcon={rerunning ? <CircularProgress size={16} color="inherit" /> : <RefreshIcon />}
                    onClick={handleRerunMatching}
                    disabled={rerunning || validating || !validationPassed}
                  >
                    {rerunning ? 'Running...' : 'Re-run Matching'}
                  </Button>
                </span>
              </Tooltip>
            </>
          )}
        </Toolbar>
      </AppBar>

      <Box sx={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        {/* Left: Assigned crops per player */}
        <Box sx={{ width: 420, p: 2, overflowY: 'auto', borderRight: '1px solid rgba(255,255,255,0.1)' }}>
          <Typography variant="subtitle2" sx={{ color: 'grey.300', mb: 1 }}>
            Assigned Crops
          </Typography>
          <Typography variant="body2" sx={{ color: 'grey.500', mb: 2 }}>
            Click a candidate on the right to assign it to a player. Label
            as many or as few players as you like — single-player labeling
            is valid and will fix swaps involving that player. 2-4 diverse
            crops per labeled player is recommended.
          </Typography>

          {/* Pre-flight validation feedback */}
          {validation && hasAnyCrops && (
            <Box
              sx={{
                mb: 2,
                p: 1.5,
                borderRadius: 1,
                bgcolor: validationPassed
                  ? 'rgba(76, 175, 80, 0.08)'
                  : 'rgba(244, 67, 54, 0.08)',
                border: `1px solid ${
                  validationPassed
                    ? 'rgba(76, 175, 80, 0.3)'
                    : 'rgba(244, 67, 54, 0.3)'
                }`,
              }}
            >
              <Typography
                variant="body2"
                sx={{
                  fontWeight: 'bold',
                  color: validationPassed ? 'success.light' : 'error.light',
                  mb: blockingIssues.length > 0 || warnings.length > 0 ? 0.5 : 0,
                }}
              >
                {validationPassed
                  ? 'Crops look good — ready to re-run matching'
                  : `${blockingIssues.length} issue${blockingIssues.length === 1 ? '' : 's'} to fix before re-running`}
              </Typography>
              {blockingIssues.map((issue, idx) => (
                <Typography
                  key={`block-${idx}`}
                  variant="caption"
                  sx={{ color: 'error.light', display: 'block', mt: 0.5 }}
                >
                  • {issue.message}
                </Typography>
              ))}
              {warnings.map((issue, idx) => (
                <Typography
                  key={`warn-${idx}`}
                  variant="caption"
                  sx={{ color: 'warning.light', display: 'block', mt: 0.5 }}
                >
                  • {issue.message}
                </Typography>
              ))}
            </Box>
          )}

          {referenceCropsLoading ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
              <CircularProgress size={24} />
            </Box>
          ) : (
            <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 1 }}>
              {PLAYER_LABELS.map((label, i) => {
                const pid = i + 1;
                const crops = cropsByPlayer[pid];
                return (
                  <Box key={label}>
                    <Box sx={{ textAlign: 'center', mb: 1 }}>
                      <Chip
                        label={`${label} (${crops.length})`}
                        size="small"
                        sx={{ bgcolor: TRACK_COLORS[i], color: 'white', fontWeight: 'bold' }}
                      />
                    </Box>
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5, alignItems: 'center' }}>
                      {crops.map((crop) => (
                        <Box
                          key={crop.id}
                          sx={{
                            position: 'relative',
                            width: CROP_WIDTH,
                            height: CROP_HEIGHT,
                            border: `2px solid ${TRACK_COLORS[i]}`,
                            borderRadius: 1,
                            overflow: 'hidden',
                            '&:hover .delete-btn': { opacity: 1 },
                          }}
                        >
                          {/* eslint-disable-next-line @next/next/no-img-element */}
                          <img
                            src={crop.downloadUrl}
                            alt={`${label} crop`}
                            style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                          />
                          <IconButton
                            className="delete-btn"
                            size="small"
                            onClick={() => handleDeleteCrop(crop.id)}
                            sx={{
                              position: 'absolute',
                              top: 0,
                              right: 0,
                              opacity: 0,
                              transition: 'opacity 0.2s',
                              bgcolor: 'rgba(0,0,0,0.6)',
                              color: 'white',
                              p: 0.25,
                              '&:hover': { bgcolor: 'rgba(200,0,0,0.8)' },
                            }}
                          >
                            <DeleteIcon sx={{ fontSize: 14 }} />
                          </IconButton>
                        </Box>
                      ))}
                      {crops.length === 0 && (
                        <Box
                          sx={{
                            width: CROP_WIDTH,
                            height: CROP_HEIGHT,
                            border: '2px dashed rgba(255,255,255,0.2)',
                            borderRadius: 1,
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                          }}
                        >
                          <Typography variant="caption" sx={{ color: 'grey.600', textAlign: 'center', px: 0.5 }}>
                            No crops
                          </Typography>
                        </Box>
                      )}
                    </Box>
                  </Box>
                );
              })}
            </Box>
          )}
        </Box>

        {/* Right: Candidate crops from tracked rallies */}
        <Box sx={{ flex: 1, p: 2, overflowY: 'auto' }}>
          <Typography variant="subtitle2" sx={{ color: 'grey.300', mb: 1 }}>
            Candidates from Tracked Rallies
          </Typography>
          <Typography variant="body2" sx={{ color: 'grey.500', mb: 2 }}>
            Click a crop to assign it as a reference for P1, P2, P3, or P4.
          </Typography>

          {candidates.length === 0 && !isLoading && (
            <Box sx={{ py: 6, textAlign: 'center' }}>
              <Typography variant="body2" sx={{ color: 'grey.500' }}>
                No tracked rallies found. Run &quot;Track All&quot; first to generate player crops.
              </Typography>
            </Box>
          )}

          {isLoading && candidates.length === 0 && (
            <Box sx={{ display: 'flex', justifyContent: 'center', py: 6 }}>
              <CircularProgress size={32} />
            </Box>
          )}

          {/* Group candidates by rally */}
          {(() => {
            const byRally = new Map<number, CandidateCrop[]>();
            for (const c of candidates) {
              const list = byRally.get(c.rallyIndex) ?? [];
              list.push(c);
              byRally.set(c.rallyIndex, list);
            }

            return [...byRally.entries()].map(([rallyIdx, rallyCandidates]) => (
              <Box key={rallyIdx} sx={{ mb: 2 }}>
                <Typography variant="caption" sx={{ color: 'grey.500', mb: 0.5, display: 'block' }}>
                  Rally {rallyIdx + 1}
                </Typography>
                <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                  {rallyCandidates.map((candidate) => (
                    <Box
                      key={candidate.id}
                      onClick={(e) => handleCandidateClick(e, candidate)}
                      sx={{
                        width: CROP_WIDTH,
                        height: CROP_HEIGHT,
                        borderRadius: 1,
                        overflow: 'hidden',
                        cursor: 'pointer',
                        border: '2px solid rgba(255,255,255,0.15)',
                        transition: 'all 0.15s',
                        '&:hover': {
                          border: '2px solid rgba(255,255,255,0.6)',
                          transform: 'scale(1.05)',
                        },
                      }}
                    >
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={candidate.dataUrl}
                        alt={`Track ${candidate.trackId}`}
                        style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                      />
                    </Box>
                  ))}
                </Box>
              </Box>
            ));
          })()}
        </Box>
      </Box>

      {/* Saving overlay */}
      {saving && (
        <Box
          sx={{
            position: 'fixed',
            top: 64,
            right: 16,
            bgcolor: 'rgba(0,0,0,0.8)',
            borderRadius: 1,
            px: 2,
            py: 1,
            display: 'flex',
            alignItems: 'center',
            gap: 1,
            zIndex: 2,
          }}
        >
          <CircularProgress size={16} sx={{ color: 'white' }} />
          <Typography variant="body2" sx={{ color: 'white' }}>Saving...</Typography>
        </Box>
      )}

      {/* Player assignment popover */}
      <Popover
        open={Boolean(anchorEl)}
        anchorEl={anchorEl}
        onClose={() => { setAnchorEl(null); setPendingCandidate(null); }}
        anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
        transformOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Box sx={{ p: 1, display: 'flex', gap: 0.5 }}>
          <Typography variant="body2" sx={{ alignSelf: 'center', mr: 0.5, color: 'text.secondary' }}>
            Assign to:
          </Typography>
          {PLAYER_LABELS.map((label, i) => (
            <Tooltip key={label} title={`Assign to ${label}`}>
              <Button
                size="small"
                variant="contained"
                onClick={() => handleAssignPlayer(i + 1)}
                sx={{
                  bgcolor: TRACK_COLORS[i],
                  color: 'white',
                  minWidth: 48,
                  '&:hover': { bgcolor: TRACK_COLORS[i], filter: 'brightness(1.2)' },
                }}
              >
                {label}
              </Button>
            </Tooltip>
          ))}
        </Box>
      </Popover>
    </Dialog>
  );
}
