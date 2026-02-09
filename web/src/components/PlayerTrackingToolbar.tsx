'use client';

import { useEffect } from 'react';
import { Box, Button, CircularProgress, Tooltip } from '@mui/material';
import CropFreeIcon from '@mui/icons-material/CropFree';
import PersonSearchIcon from '@mui/icons-material/PersonSearch';
import VisibilityIcon from '@mui/icons-material/Visibility';
import VisibilityOffIcon from '@mui/icons-material/VisibilityOff';
import { usePlayerTrackingStore } from '@/stores/playerTrackingStore';
import { useEditorStore } from '@/stores/editorStore';

export function PlayerTrackingToolbar() {
  const activeMatchId = useEditorStore((state) => state.activeMatchId);
  const selectedRallyId = useEditorStore((state) => state.selectedRallyId);
  const rallies = useEditorStore((state) => state.rallies);
  const getActiveMatch = useEditorStore((state) => state.getActiveMatch);

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
  } = usePlayerTrackingStore();

  // Get backend rally ID from selected rally
  const selectedRally = rallies.find((r) => r.id === selectedRallyId);
  const backendRallyId = selectedRally?._backendId ?? null;
  const activeMatch = getActiveMatch();
  const fps = activeMatch?.video?.fps || 30;

  // Load existing tracking data when rally is selected
  useEffect(() => {
    if (backendRallyId) {
      loadPlayerTrack(backendRallyId, fps);
    }
  }, [backendRallyId, fps, loadPlayerTrack]);

  // Don't show if no video loaded
  if (!activeMatchId) {
    return null;
  }

  const hasCalibration = !!calibrations[activeMatchId];
  const isTrackingRally = backendRallyId ? isTracking[backendRallyId] : false;
  const isLoadingTrackData = backendRallyId ? isLoadingTrack[backendRallyId] : false;
  const hasTrackingData = backendRallyId ? !!playerTracks[backendRallyId]?.tracksJson?.tracks?.length : false;

  const handleStartCalibration = () => {
    setIsCalibrating(true);
  };

  const handleTrackPlayers = async () => {
    if (!backendRallyId || !activeMatchId) return;
    await trackPlayersForRally(backendRallyId, activeMatchId, fps);
  };

  return (
    <Box
      sx={{
        display: 'flex',
        alignItems: 'center',
        gap: 1,
        py: 1,
        px: 0.5,
      }}
    >
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

      {/* Toggle overlay visibility */}
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
    </Box>
  );
}
