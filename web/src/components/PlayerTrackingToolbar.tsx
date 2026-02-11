'use client';

import { useEffect, useState } from 'react';
import { Box, Button, CircularProgress, Tooltip, Typography, Chip, Stack, Collapse, IconButton, Dialog, DialogTitle, DialogContent, DialogActions, TextField } from '@mui/material';
import CropFreeIcon from '@mui/icons-material/CropFree';
import PersonSearchIcon from '@mui/icons-material/PersonSearch';
import VisibilityIcon from '@mui/icons-material/Visibility';
import VisibilityOffIcon from '@mui/icons-material/VisibilityOff';
import SportsVolleyballIcon from '@mui/icons-material/SportsVolleyball';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import EditIcon from '@mui/icons-material/Edit';
import SaveIcon from '@mui/icons-material/Save';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import { usePlayerTrackingStore } from '@/stores/playerTrackingStore';
import { useEditorStore } from '@/stores/editorStore';
import { usePlayerStore } from '@/stores/playerStore';
import type { BallPhase } from '@/services/api';
import { getLabelStudioStatus, exportToLabelStudio, importFromLabelStudio, API_BASE_URL } from '@/services/api';

// Phase colors matching volleyball semantics
const PHASE_COLORS: Record<string, string> = {
  serve: '#4CAF50',     // Green - start of play
  receive: '#2196F3',   // Blue - first touch (pass)
  set: '#FFC107',       // Amber - setter touch
  attack: '#f44336',    // Red - spike/hit
  dig: '#9C27B0',       // Purple - defensive save
  defense: '#2196F3',   // Blue (legacy alias for receive)
  transition: '#FFC107', // Amber (legacy alias for set)
  unknown: '#9e9e9e',   // Grey
};

// Format frame number to timestamp (mm:ss.ms)
function frameToTime(frame: number, fps: number): string {
  const totalSeconds = frame / fps;
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toFixed(2).padStart(5, '0')}`;
}

// Format phase for display
function formatPhase(phase: string): string {
  return phase.charAt(0).toUpperCase() + phase.slice(1);
}

// Phase Timeline Component - Visual horizontal bar showing phases
interface PhaseTimelineProps {
  phases: BallPhase[];
  currentFrame: number;
  totalFrames: number;
  fps: number;
  rallyStartTime: number;
}

function PhaseTimeline({ phases, currentFrame, totalFrames, fps, rallyStartTime }: PhaseTimelineProps) {
  const seek = usePlayerStore((state) => state.seek);

  if (totalFrames === 0 || phases.length === 0) return null;

  const sortedPhases = [...phases].sort((a, b) => a.frameStart - b.frameStart);
  const rallyDuration = totalFrames / fps;

  const handleSeek = (frame: number) => {
    const time = rallyStartTime + (frame / totalFrames) * rallyDuration;
    seek(time);
  };

  return (
    <Box
      sx={{
        width: '100%',
        mt: 1,
        px: 0.5,
      }}
    >
      {/* Timeline bar */}
      <Box
        sx={{
          position: 'relative',
          height: 24,
          bgcolor: 'action.hover',
          borderRadius: 1,
          overflow: 'hidden',
          cursor: 'pointer',
        }}
      >
        {/* Phase segments */}
        {sortedPhases.map((phase, index) => {
          const startPercent = (phase.frameStart / totalFrames) * 100;
          const endPercent = (phase.frameEnd / totalFrames) * 100;
          const widthPercent = Math.max(1, endPercent - startPercent); // Minimum 1% width

          return (
            <Tooltip
              key={index}
              title={`${formatPhase(phase.phase)} (${frameToTime(phase.frameStart, fps)} - ${frameToTime(phase.frameEnd, fps)})`}
            >
              <Box
                onClick={() => handleSeek(phase.frameStart)}
                sx={{
                  position: 'absolute',
                  left: `${startPercent}%`,
                  width: `${widthPercent}%`,
                  height: '100%',
                  bgcolor: PHASE_COLORS[phase.phase] || PHASE_COLORS.unknown,
                  opacity: 0.85,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  '&:hover': {
                    opacity: 1,
                    transform: 'scaleY(1.1)',
                  },
                  transition: 'opacity 0.15s, transform 0.15s',
                }}
              >
                {widthPercent > 8 && (
                  <Typography
                    variant="caption"
                    sx={{
                      color: 'white',
                      fontSize: '0.65rem',
                      fontWeight: 'bold',
                      textShadow: '0 1px 2px rgba(0,0,0,0.5)',
                      textTransform: 'uppercase',
                    }}
                  >
                    {phase.phase.slice(0, 3)}
                  </Typography>
                )}
              </Box>
            </Tooltip>
          );
        })}

        {/* Playhead indicator */}
        {currentFrame >= 0 && (
          <Box
            sx={{
              position: 'absolute',
              left: `${(currentFrame / totalFrames) * 100}%`,
              width: 2,
              height: '100%',
              bgcolor: 'common.white',
              boxShadow: '0 0 4px rgba(0,0,0,0.5)',
              pointerEvents: 'none',
              zIndex: 10,
            }}
          />
        )}
      </Box>

      {/* Frame numbers */}
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'space-between',
          mt: 0.25,
          px: 0.5,
        }}
      >
        <Typography variant="caption" sx={{ color: 'text.secondary', fontSize: '0.6rem' }}>
          0:00
        </Typography>
        <Typography variant="caption" sx={{ color: 'text.secondary', fontSize: '0.6rem' }}>
          {frameToTime(totalFrames, fps)}
        </Typography>
      </Box>
    </Box>
  );
}

// Find the active ball phase for the current frame
function findActivePhase(ballPhases: BallPhase[], currentFrame: number): BallPhase | null {
  // Find phase where currentFrame is between frameStart and frameEnd
  for (const phase of ballPhases) {
    if (currentFrame >= phase.frameStart && currentFrame <= phase.frameEnd) {
      return phase;
    }
  }
  // If no active phase, find the most recent one
  const sortedPhases = [...ballPhases].sort((a, b) => b.frameStart - a.frameStart);
  for (const phase of sortedPhases) {
    if (currentFrame >= phase.frameStart) {
      return phase;
    }
  }
  return null;
}

export function PlayerTrackingToolbar() {
  const [showEventDetails, setShowEventDetails] = useState(false);
  const [labelStudioLoading, setLabelStudioLoading] = useState(false);
  const [hasGroundTruth, setHasGroundTruth] = useState(false);
  const [labelStudioTaskId, setLabelStudioTaskId] = useState<number | null>(null);
  const [showImportDialog, setShowImportDialog] = useState(false);
  const [importTaskId, setImportTaskId] = useState('');

  const activeMatchId = useEditorStore((state) => state.activeMatchId);
  const selectedRallyId = useEditorStore((state) => state.selectedRallyId);
  const rallies = useEditorStore((state) => state.rallies);
  const getActiveMatch = useEditorStore((state) => state.getActiveMatch);

  // Get current playback time for real-time phase highlighting
  const currentTime = usePlayerStore((state) => state.currentTime);

  const {
    isCalibrating,
    setIsCalibrating,
    calibrations,
    isTracking,
    isLoadingTrack,
    playerTracks,
    trackPlayersForRally,
    loadPlayerTrack,
    showPlayerOverlay,
    togglePlayerOverlay,
    showBallOverlay,
    toggleBallOverlay,
  } = usePlayerTrackingStore();

  // Get backend rally ID from selected rally
  const selectedRally = rallies.find((r) => r.id === selectedRallyId);
  const backendRallyId = selectedRally?._backendId ?? null;
  const activeMatch = getActiveMatch();
  const fps = activeMatch?.video?.fps || 30;

  // Compute track data variables (needed for hooks below)
  const hasCalibration = activeMatchId ? !!calibrations[activeMatchId] : false;
  const isTrackingRally = backendRallyId ? isTracking[backendRallyId] : false;
  const isLoadingTrackData = backendRallyId ? isLoadingTrack[backendRallyId] : false;
  const trackData = backendRallyId ? playerTracks[backendRallyId]?.tracksJson : null;
  const hasTrackingData = !!trackData?.tracks?.length;

  // Load existing tracking data when rally is selected
  useEffect(() => {
    if (backendRallyId) {
      loadPlayerTrack(backendRallyId, fps);
    }
  }, [backendRallyId, fps, loadPlayerTrack]);

  // Check Label Studio status when tracking data is available
  useEffect(() => {
    const checkLabelStudioStatus = async () => {
      if (!backendRallyId || !hasTrackingData) {
        setHasGroundTruth(false);
        setLabelStudioTaskId(null);
        return;
      }
      try {
        const status = await getLabelStudioStatus(backendRallyId);
        setHasGroundTruth(status.hasGroundTruth);
        setLabelStudioTaskId(status.taskId ?? null);
      } catch (error) {
        console.error('Failed to get Label Studio status:', error);
      }
    };
    checkLabelStudioStatus();
  }, [backendRallyId, hasTrackingData]);

  // Don't show if no video loaded
  if (!activeMatchId) {
    return null;
  }

  // Get ball phase data
  const ballPhases = trackData?.ballPhases || [];
  const serverInfo = trackData?.serverInfo;

  // Calculate current frame relative to the rally for real-time highlighting
  const rallyStartTime = selectedRally?.start_time ?? 0;
  const rallyEndTime = selectedRally?.end_time ?? 0;
  const isWithinRally = currentTime >= rallyStartTime && currentTime <= rallyEndTime;
  const relativeTime = isWithinRally ? currentTime - rallyStartTime : -1;
  const currentFrame = relativeTime >= 0 ? Math.floor(relativeTime * fps) : -1;

  // Find active ball phase
  const activePhase = ballPhases.length > 0 && currentFrame >= 0
    ? findActivePhase(ballPhases, currentFrame)
    : null;

  // Compute phase counts for summary
  const phaseCounts = ballPhases.reduce((acc, phase) => {
    acc[phase.phase] = (acc[phase.phase] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  const handleStartCalibration = () => {
    setIsCalibrating(true);
  };

  const handleTrackPlayers = async () => {
    if (!backendRallyId || !activeMatchId) return;
    await trackPlayersForRally(backendRallyId, activeMatchId, fps);
  };

  // Open tracking data in Label Studio for correction
  const handleOpenInLabelStudio = async () => {
    if (!backendRallyId || !activeMatch?.videoUrl) return;

    setLabelStudioLoading(true);
    try {
      // Convert relative URL to absolute URL for Label Studio (must use API server, not web server)
      const videoUrl = new URL(activeMatch.videoUrl, API_BASE_URL).href;
      // Always force regenerate to ensure fresh predictions with correct timing
      // This fixes issues where existing tasks have stale/incorrect frame numbers
      const result = await exportToLabelStudio(backendRallyId, videoUrl, { forceRegenerate: true });
      if (result.success && result.taskUrl) {
        setLabelStudioTaskId(result.taskId ?? null);
        // Open Label Studio in new tab
        window.open(result.taskUrl, '_blank');
      } else {
        console.error('Failed to export to Label Studio:', result.error);
        alert(result.error || 'Failed to export to Label Studio');
      }
    } catch (error) {
      console.error('Failed to open Label Studio:', error);
      alert(error instanceof Error ? error.message : 'Failed to open Label Studio');
    } finally {
      setLabelStudioLoading(false);
    }
  };

  // Import corrected annotations from Label Studio
  const handleSaveGroundTruth = async () => {
    if (!backendRallyId) return;

    // If we have a known task ID, use it directly
    if (labelStudioTaskId) {
      setLabelStudioLoading(true);
      try {
        const result = await importFromLabelStudio(backendRallyId, labelStudioTaskId);
        if (result.success) {
          setHasGroundTruth(true);
          alert(`Ground truth saved! ${result.playerCount} player annotations, ${result.ballCount} ball annotations across ${result.frameCount} frames.`);
        } else {
          console.error('Failed to import from Label Studio:', result.error);
          alert(result.error || 'Failed to save ground truth');
        }
      } catch (error) {
        console.error('Failed to save ground truth:', error);
        alert(error instanceof Error ? error.message : 'Failed to save ground truth');
      } finally {
        setLabelStudioLoading(false);
      }
    } else {
      // Show dialog to enter task ID manually
      setShowImportDialog(true);
    }
  };

  // Handle import dialog submission
  const handleImportSubmit = async () => {
    if (!backendRallyId || !importTaskId) return;

    const taskId = parseInt(importTaskId, 10);
    if (isNaN(taskId) || taskId <= 0) {
      alert('Please enter a valid task ID');
      return;
    }

    setShowImportDialog(false);
    setLabelStudioLoading(true);
    try {
      const result = await importFromLabelStudio(backendRallyId, taskId);
      if (result.success) {
        setHasGroundTruth(true);
        setLabelStudioTaskId(taskId);
        setImportTaskId('');
        alert(`Ground truth saved! ${result.playerCount} player annotations, ${result.ballCount} ball annotations across ${result.frameCount} frames.`);
      } else {
        console.error('Failed to import from Label Studio:', result.error);
        alert(result.error || 'Failed to save ground truth');
      }
    } catch (error) {
      console.error('Failed to save ground truth:', error);
      alert(error instanceof Error ? error.message : 'Failed to save ground truth');
    } finally {
      setLabelStudioLoading(false);
    }
  };

  const hasBallPositions = !!trackData?.ballPositions?.length;

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        py: 1,
        px: 0.5,
      }}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
      {/* Calibration button */}
      {!isCalibrating && (
        <Button
          size="small"
          variant="outlined"
          startIcon={<CropFreeIcon />}
          onClick={handleStartCalibration}
        >
          {hasCalibration ? 'Re-calibrate Court' : 'Calibrate Court'}
        </Button>
      )}

      {/* Track Players button */}
      {!isCalibrating && selectedRallyId && backendRallyId && (
        <Tooltip title={!selectedRallyId ? 'Select a rally first' : ''}>
          <span>
            <Button
              size="small"
              variant="outlined"
              startIcon={
                isTrackingRally || isLoadingTrackData ? (
                  <CircularProgress size={16} />
                ) : (
                  <PersonSearchIcon />
                )
              }
              onClick={handleTrackPlayers}
              disabled={isTrackingRally || isLoadingTrackData || !selectedRallyId}
            >
              {isTrackingRally
                ? 'Tracking...'
                : isLoadingTrackData
                  ? 'Loading...'
                  : hasTrackingData
                    ? 'Re-track Players'
                    : 'Track Players'}
            </Button>
          </span>
        </Tooltip>
      )}

      {/* Toggle player overlay visibility */}
      {hasTrackingData && !isCalibrating && (
        <Tooltip title={showPlayerOverlay ? 'Hide player overlay' : 'Show player overlay'}>
          <Button
            size="small"
            variant={showPlayerOverlay ? 'contained' : 'outlined'}
            onClick={togglePlayerOverlay}
            sx={{ minWidth: 'auto', px: 1 }}
          >
            {showPlayerOverlay ? <VisibilityIcon /> : <VisibilityOffIcon />}
          </Button>
        </Tooltip>
      )}

      {/* Toggle ball overlay visibility - only show if ball positions available */}
      {hasTrackingData && hasBallPositions && !isCalibrating && (
        <Tooltip title={showBallOverlay ? 'Hide ball track' : 'Show ball track'}>
          <Button
            size="small"
            variant={showBallOverlay ? 'contained' : 'outlined'}
            onClick={toggleBallOverlay}
            sx={{
              minWidth: 'auto',
              px: 1,
              bgcolor: showBallOverlay ? '#FFC107' : undefined,
              '&:hover': showBallOverlay ? { bgcolor: '#FFB300' } : undefined,
            }}
          >
            <SportsVolleyballIcon fontSize="small" />
          </Button>
        </Tooltip>
      )}

      {/* Label Studio Integration - Ground Truth Labeling */}
      {hasTrackingData && !isCalibrating && (
        <>
          <Tooltip title="Open in Label Studio to correct tracking">
            <span>
              <Button
                size="small"
                variant="outlined"
                startIcon={labelStudioLoading ? <CircularProgress size={14} /> : <EditIcon />}
                onClick={handleOpenInLabelStudio}
                disabled={labelStudioLoading}
                sx={{ ml: 1 }}
              >
                Label
              </Button>
            </span>
          </Tooltip>
          <Tooltip title={hasGroundTruth ? 'Ground truth saved' : 'Save corrected labels as ground truth'}>
            <span>
              <Button
                size="small"
                variant={hasGroundTruth ? 'contained' : 'outlined'}
                startIcon={hasGroundTruth ? <CheckCircleIcon /> : <SaveIcon />}
                onClick={handleSaveGroundTruth}
                disabled={labelStudioLoading}
                color={hasGroundTruth ? 'success' : 'primary'}
              >
                {hasGroundTruth ? 'GT Saved' : 'Save GT'}
              </Button>
            </span>
          </Tooltip>
        </>
      )}

      {/* Ball Phase Info */}
      {hasTrackingData && ballPhases.length > 0 && !isCalibrating && (
        <Box sx={{ ml: 2, display: 'flex', alignItems: 'center', gap: 1 }}>
          <SportsVolleyballIcon fontSize="small" sx={{ color: 'text.secondary' }} />
          <Stack direction="row" spacing={0.5}>
            {Object.entries(phaseCounts).map(([phase, count]) => (
              <Chip
                key={phase}
                label={`${phase}: ${count}`}
                size="small"
                sx={{
                  bgcolor: PHASE_COLORS[phase] || PHASE_COLORS.unknown,
                  color: 'white',
                  fontSize: '0.7rem',
                  height: 20,
                }}
              />
            ))}
          </Stack>
          {serverInfo && serverInfo.trackId >= 0 && (
            <Tooltip title={`Server confidence: ${Math.round(serverInfo.confidence * 100)}%`}>
              <Typography
                variant="caption"
                sx={{
                  ml: 1,
                  color: 'success.main',
                  fontWeight: 'bold',
                }}
              >
                Server: #{serverInfo.trackId}
              </Typography>
            </Tooltip>
          )}
          {/* Current Phase Indicator - prominent real-time display */}
          {activePhase && (
            <Chip
              label={formatPhase(activePhase.phase)}
              size="small"
              sx={{
                ml: 2,
                bgcolor: PHASE_COLORS[activePhase.phase] || PHASE_COLORS.unknown,
                color: 'white',
                fontWeight: 'bold',
                fontSize: '0.85rem',
                height: 28,
                px: 1,
                animation: 'pulse 1.5s ease-in-out infinite',
                boxShadow: `0 0 8px ${PHASE_COLORS[activePhase.phase] || PHASE_COLORS.unknown}`,
                '@keyframes pulse': {
                  '0%, 100%': { opacity: 1 },
                  '50%': { opacity: 0.7 },
                },
              }}
            />
          )}
          <Tooltip title={showEventDetails ? 'Hide event timeline' : 'Show event timeline'}>
            <IconButton
              size="small"
              onClick={() => setShowEventDetails(!showEventDetails)}
              sx={{ ml: 0.5 }}
            >
              {showEventDetails ? <ExpandLessIcon fontSize="small" /> : <ExpandMoreIcon fontSize="small" />}
            </IconButton>
          </Tooltip>
        </Box>
      )}
      </Box>

      {/* Phase Timeline - Visual horizontal bar */}
      {hasTrackingData && ballPhases.length > 0 && !isCalibrating && trackData && (
        <PhaseTimeline
          phases={ballPhases}
          currentFrame={currentFrame}
          totalFrames={trackData.frameCount}
          fps={fps}
          rallyStartTime={rallyStartTime}
        />
      )}

      {/* Event Timeline Detail */}
      {hasTrackingData && ballPhases.length > 0 && !isCalibrating && (
        <Collapse in={showEventDetails} sx={{ width: '100%', mt: 1 }}>
          <Box
            sx={{
              display: 'flex',
              flexWrap: 'wrap',
              gap: 0.5,
              p: 1,
              bgcolor: 'background.paper',
              borderRadius: 1,
              border: '1px solid',
              borderColor: 'divider',
            }}
          >
            {ballPhases
              .sort((a, b) => a.frameStart - b.frameStart)
              .map((event, index) => {
                const isActive = activePhase === event;
                return (
                  <Tooltip
                    key={index}
                    title={`Velocity: ${event.velocity.toFixed(3)} | Position: (${event.ballX.toFixed(2)}, ${event.ballY.toFixed(2)})`}
                  >
                    <Chip
                      label={`${frameToTime(event.frameStart, fps)} ${formatPhase(event.phase)}`}
                      size="small"
                      sx={{
                        bgcolor: PHASE_COLORS[event.phase] || PHASE_COLORS.unknown,
                        color: 'white',
                        fontSize: '0.7rem',
                        height: 22,
                        '& .MuiChip-label': {
                          px: 1,
                        },
                        // Highlight active event
                        ...(isActive && {
                          outline: '2px solid white',
                          outlineOffset: 1,
                          boxShadow: `0 0 8px ${PHASE_COLORS[event.phase] || PHASE_COLORS.unknown}`,
                          transform: 'scale(1.1)',
                        }),
                        transition: 'transform 0.2s, box-shadow 0.2s',
                      }}
                    />
                  </Tooltip>
                );
              })}
            {serverInfo && serverInfo.trackId >= 0 && (
              <Tooltip title={`Confidence: ${Math.round(serverInfo.confidence * 100)}% | Velocity: ${serverInfo.serveVelocity.toFixed(3)}`}>
                <Chip
                  label={`${frameToTime(serverInfo.serveFrame, fps)} Server #${serverInfo.trackId}`}
                  size="small"
                  variant="outlined"
                  sx={{
                    borderColor: PHASE_COLORS.serve,
                    color: PHASE_COLORS.serve,
                    fontSize: '0.7rem',
                    height: 22,
                    fontWeight: 'bold',
                  }}
                />
              </Tooltip>
            )}
          </Box>
        </Collapse>
      )}

      {/* Import Task ID Dialog */}
      <Dialog open={showImportDialog} onClose={() => setShowImportDialog(false)}>
        <DialogTitle>Import Ground Truth from Label Studio</DialogTitle>
        <DialogContent>
          <Typography variant="body2" sx={{ mb: 2 }}>
            Enter the Label Studio task ID to import corrected annotations.
          </Typography>
          <TextField
            autoFocus
            label="Task ID"
            type="number"
            fullWidth
            value={importTaskId}
            onChange={(e) => setImportTaskId(e.target.value)}
            helperText="Find the task ID in the Label Studio URL (e.g., .../task=123)"
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowImportDialog(false)}>Cancel</Button>
          <Button onClick={handleImportSubmit} variant="contained" disabled={!importTaskId}>
            Import
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
