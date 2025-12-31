'use client';

import { useState } from 'react';
import {
  Box,
  Typography,
  Chip,
  Stack,
  IconButton,
  Popover,
  Switch,
  FormControlLabel,
  Button,
  Tooltip,
  CircularProgress,
} from '@mui/material';
import FileDownloadIcon from '@mui/icons-material/FileDownload';
import { useEditorStore } from '@/stores/editorStore';
import { usePlayerStore } from '@/stores/playerStore';
import { useExportStore } from '@/stores/exportStore';
import { formatTime, formatDuration } from '@/utils/timeFormat';

export function RallyList() {
  const { rallies, selectedRallyId, selectRally, getHighlightsForRally, videoFile, videoUrl } = useEditorStore();
  const { currentTime, seek } = usePlayerStore();
  const { isExporting, exportingRallyId, exportingAll, downloadRally, downloadAllRallies } = useExportStore();

  // Use File if available, otherwise use URL
  const videoSource = videoFile || videoUrl;

  // Popover state for Download All
  const [downloadAllAnchor, setDownloadAllAnchor] = useState<HTMLButtonElement | null>(null);
  const [withFade, setWithFade] = useState(false);

  const handleClick = (id: string, startTime: number) => {
    selectRally(id);
    seek(startTime);
  };

  const handleDownloadRally = (e: React.MouseEvent, rallyId: string) => {
    e.stopPropagation();
    if (!videoSource) return;
    const rally = rallies?.find((r) => r.id === rallyId);
    if (rally) {
      downloadRally(videoSource, rally);
    }
  };

  const handleDownloadAll = () => {
    if (!videoSource || !sortedRallies.length) return;
    downloadAllRallies(videoSource, sortedRallies, withFade);
    setDownloadAllAnchor(null);
  };

  // Sort rallies by start time for display
  const sortedRallies = [...(rallies ?? [])].sort((a, b) => a.start_time - b.start_time);

  // Find which rally contains the current time
  const activeRallyId = sortedRallies.find(
    (s) => currentTime >= s.start_time && currentTime <= s.end_time
  )?.id;

  // Calculate total duration
  const totalDuration = sortedRallies.reduce((sum, s) => sum + s.duration, 0);

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* Header */}
      <Box
        sx={{
          px: 1.5,
          py: 1,
          borderBottom: 1,
          borderColor: 'divider',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}
      >
        <Box>
          <Typography
            variant="overline"
            sx={{
              fontSize: 11,
              fontWeight: 600,
              letterSpacing: 1,
              color: 'text.secondary',
            }}
          >
            Rallies
          </Typography>
          {sortedRallies.length > 0 && (
            <Typography
              variant="caption"
              sx={{
                display: 'block',
                color: 'text.disabled',
                fontSize: 10,
                mt: -0.5,
              }}
            >
              Total: {formatDuration(totalDuration)}
            </Typography>
          )}
        </Box>
        {sortedRallies.length > 0 && (
          <Stack direction="row" spacing={0.5} alignItems="center">
            <Chip
              label={sortedRallies.length}
              size="small"
              sx={{
                height: 20,
                fontSize: 11,
                fontWeight: 600,
                bgcolor: 'action.selected',
              }}
            />
            <Tooltip title={!videoSource ? 'Load a video first' : 'Download all rallies'}>
              <span>
                <IconButton
                  size="small"
                  disabled={!videoSource || isExporting}
                  onClick={(e) => setDownloadAllAnchor(e.currentTarget)}
                  sx={{
                    p: 0.5,
                    color: exportingAll ? 'primary.main' : 'text.secondary',
                  }}
                >
                  {exportingAll ? (
                    <CircularProgress size={14} />
                  ) : (
                    <FileDownloadIcon sx={{ fontSize: 16 }} />
                  )}
                </IconButton>
              </span>
            </Tooltip>
          </Stack>
        )}
      </Box>

      {/* Download All Popover */}
      <Popover
        open={Boolean(downloadAllAnchor)}
        anchorEl={downloadAllAnchor}
        onClose={() => setDownloadAllAnchor(null)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
        transformOrigin={{ vertical: 'top', horizontal: 'right' }}
      >
        <Box sx={{ p: 2, minWidth: 180 }}>
          <FormControlLabel
            control={
              <Switch
                size="small"
                checked={withFade}
                onChange={(e) => setWithFade(e.target.checked)}
              />
            }
            label={
              <Typography variant="body2">Add fade (0.5s)</Typography>
            }
          />
          <Button
            fullWidth
            variant="contained"
            size="small"
            startIcon={<FileDownloadIcon />}
            onClick={handleDownloadAll}
            disabled={isExporting}
            sx={{ mt: 1.5 }}
          >
            Download
          </Button>
        </Box>
      </Popover>

      {/* Rally list */}
      <Box sx={{ flex: 1, overflow: 'auto' }}>
        {sortedRallies.length === 0 ? (
          <Box
            sx={{
              p: 2,
              textAlign: 'center',
              color: 'text.disabled',
            }}
          >
            <Typography variant="caption">
              No rallies loaded
            </Typography>
          </Box>
        ) : (
          <Box sx={{ py: 0.5 }}>
            {sortedRallies.map((rally, index) => {
              const isSelected = selectedRallyId === rally.id;
              const isActive = activeRallyId === rally.id;

              return (
                <Box
                  key={rally.id}
                  onClick={() => handleClick(rally.id, rally.start_time)}
                  sx={{
                    display: 'flex',
                    alignItems: 'center',
                    px: 1.5,
                    py: 0.75,
                    cursor: 'pointer',
                    borderLeft: '2px solid',
                    borderColor: isActive ? 'primary.main' : 'transparent',
                    bgcolor: isSelected
                      ? 'rgba(33, 150, 243, 0.12)'
                      : 'transparent',
                    transition: 'all 0.15s ease',
                    '&:hover': {
                      bgcolor: isSelected
                        ? 'rgba(33, 150, 243, 0.18)'
                        : 'rgba(255, 255, 255, 0.04)',
                    },
                  }}
                >
                  {/* Status indicator */}
                  <Box
                    sx={{
                      width: 16,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      mr: 0.5,
                      color: isActive
                        ? 'primary.main'
                        : isSelected
                        ? 'text.primary'
                        : 'transparent',
                      fontSize: 10,
                    }}
                  >
                    {isActive ? '●' : isSelected ? '▸' : ''}
                  </Box>

                  {/* Rally number */}
                  <Typography
                    sx={{
                      fontFamily: 'monospace',
                      fontSize: 11,
                      fontWeight: isSelected ? 600 : 400,
                      color: isSelected ? 'text.primary' : 'text.secondary',
                      width: 24,
                      flexShrink: 0,
                    }}
                  >
                    {String(index + 1).padStart(2, '0')}
                  </Typography>

                  {/* Time range */}
                  <Typography
                    sx={{
                      fontFamily: 'monospace',
                      fontSize: 11,
                      color: isSelected ? 'text.primary' : 'text.secondary',
                      flex: 1,
                      mx: 1,
                    }}
                  >
                    {formatTime(rally.start_time)}
                    <Box
                      component="span"
                      sx={{ color: 'text.disabled', mx: 0.5 }}
                    >
                      →
                    </Box>
                    {formatTime(rally.end_time)}
                  </Typography>

                  {/* Duration */}
                  <Typography
                    sx={{
                      fontFamily: 'monospace',
                      fontSize: 10,
                      color: 'text.disabled',
                      flexShrink: 0,
                    }}
                  >
                    {rally.duration.toFixed(1)}s
                  </Typography>

                  {/* Highlight color dots */}
                  {(() => {
                    const rallyHighlights = getHighlightsForRally(rally.id);
                    if (rallyHighlights.length === 0) return null;
                    return (
                      <Stack direction="row" spacing={0.25} sx={{ ml: 0.75 }}>
                        {rallyHighlights.slice(0, 3).map((h) => (
                          <Box
                            key={h.id}
                            sx={{
                              width: 6,
                              height: 6,
                              borderRadius: '50%',
                              bgcolor: h.color,
                            }}
                          />
                        ))}
                        {rallyHighlights.length > 3 && (
                          <Typography
                            sx={{ fontSize: 8, color: 'text.disabled', lineHeight: 1 }}
                          >
                            +{rallyHighlights.length - 3}
                          </Typography>
                        )}
                      </Stack>
                    );
                  })()}

                  {/* Download button */}
                  {videoSource && (
                    <Tooltip title="Download rally">
                      <IconButton
                        size="small"
                        onClick={(e) => handleDownloadRally(e, rally.id)}
                        disabled={isExporting}
                        sx={{
                          ml: 0.5,
                          p: 0.25,
                          opacity: isSelected || exportingRallyId === rally.id ? 1 : 0,
                          transition: 'opacity 0.15s',
                          '.MuiBox-root:hover &': { opacity: 1 },
                          color: exportingRallyId === rally.id ? 'primary.main' : 'text.secondary',
                        }}
                      >
                        {exportingRallyId === rally.id ? (
                          <CircularProgress size={12} />
                        ) : (
                          <FileDownloadIcon sx={{ fontSize: 14 }} />
                        )}
                      </IconButton>
                    </Tooltip>
                  )}
                </Box>
              );
            })}
          </Box>
        )}
      </Box>
    </Box>
  );
}
