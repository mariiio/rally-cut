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
  Collapse,
} from '@mui/material';
import FileDownloadIcon from '@mui/icons-material/FileDownload';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ChevronRightIcon from '@mui/icons-material/ChevronRight';
import { useEditorStore } from '@/stores/editorStore';
import { usePlayerStore } from '@/stores/playerStore';
import { useExportStore } from '@/stores/exportStore';
import { formatTime, formatDuration } from '@/utils/timeFormat';
import { Rally } from '@/types/rally';

export function RallyList() {
  const {
    session,
    activeMatchId,
    setActiveMatch,
    rallies,
    selectedRallyId,
    selectRally,
    getHighlightsForRally,
    videoFile,
    videoUrl,
  } = useEditorStore();
  const { currentTime, seek } = usePlayerStore();
  const { isExporting, exportingRallyId, exportingAll, downloadRally, downloadAllRallies } = useExportStore();

  // Use File if available, otherwise use URL
  const videoSource = videoFile || videoUrl;

  // Track which matches are expanded
  const [expandedMatches, setExpandedMatches] = useState<Set<string>>(
    new Set(activeMatchId ? [activeMatchId] : [])
  );

  // Popover state for Download All
  const [downloadAllAnchor, setDownloadAllAnchor] = useState<HTMLButtonElement | null>(null);
  const [withFade, setWithFade] = useState(false);

  const toggleMatchExpanded = (matchId: string) => {
    setExpandedMatches((prev) => {
      const next = new Set(prev);
      if (next.has(matchId)) {
        next.delete(matchId);
      } else {
        next.add(matchId);
      }
      return next;
    });
  };

  const handleRallyClick = (matchId: string, rally: Rally) => {
    // Switch match if needed
    if (matchId !== activeMatchId) {
      setActiveMatch(matchId);
      // Expand the match if collapsed
      setExpandedMatches((prev) => new Set(prev).add(matchId));
    }
    selectRally(rally.id);
    seek(rally.start_time);
  };

  const handleDownloadRally = (e: React.MouseEvent, rally: Rally) => {
    e.stopPropagation();
    if (!videoSource) return;
    downloadRally(videoSource, rally);
  };

  const handleDownloadAll = () => {
    if (!videoSource || !sortedRallies.length) return;
    downloadAllRallies(videoSource, sortedRallies, withFade);
    setDownloadAllAnchor(null);
  };

  // Sort active match rallies by start time
  const sortedRallies = [...(rallies ?? [])].sort((a, b) => a.start_time - b.start_time);

  // Find which rally contains the current time (for active match only)
  const activeRallyId = sortedRallies.find(
    (s) => currentTime >= s.start_time && currentTime <= s.end_time
  )?.id;

  // Calculate total duration for active match
  const totalDuration = sortedRallies.reduce((sum, s) => sum + s.duration, 0);

  // If no session, show the old flat list behavior
  if (!session) {
    return (
      <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
        <Box
          sx={{
            px: 1.5,
            py: 1,
            borderBottom: 1,
            borderColor: 'divider',
          }}
        >
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
          <Typography
            variant="caption"
            sx={{
              display: 'block',
              color: 'text.disabled',
              fontSize: 10,
            }}
          >
            No session loaded
          </Typography>
        </Box>
      </Box>
    );
  }

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
            Matches
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
              Active: {formatDuration(totalDuration)}
            </Typography>
          )}
        </Box>
        {sortedRallies.length > 0 && (
          <Stack direction="row" spacing={0.5} alignItems="center">
            <Chip
              label={session.matches.length}
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

      {/* Match list with collapsible rallies */}
      <Box sx={{ flex: 1, overflow: 'auto' }}>
        {session.matches.map((match) => {
          const isExpanded = expandedMatches.has(match.id);
          const isActiveMatch = activeMatchId === match.id;
          const matchRallies = [...match.rallies].sort((a, b) => a.start_time - b.start_time);
          const matchDuration = matchRallies.reduce((sum, r) => sum + r.duration, 0);

          return (
            <Box key={match.id}>
              {/* Match header */}
              <Box
                onClick={() => toggleMatchExpanded(match.id)}
                sx={{
                  display: 'flex',
                  alignItems: 'center',
                  px: 1,
                  py: 0.75,
                  cursor: 'pointer',
                  borderLeft: '3px solid',
                  borderColor: isActiveMatch ? 'primary.main' : 'transparent',
                  bgcolor: isActiveMatch ? 'rgba(33, 150, 243, 0.08)' : 'transparent',
                  '&:hover': {
                    bgcolor: isActiveMatch
                      ? 'rgba(33, 150, 243, 0.12)'
                      : 'rgba(255, 255, 255, 0.04)',
                  },
                }}
              >
                {/* Expand/collapse icon */}
                <Box sx={{ mr: 0.5, display: 'flex', alignItems: 'center' }}>
                  {isExpanded ? (
                    <ExpandMoreIcon sx={{ fontSize: 18, color: 'text.secondary' }} />
                  ) : (
                    <ChevronRightIcon sx={{ fontSize: 18, color: 'text.secondary' }} />
                  )}
                </Box>

                {/* Match name */}
                <Typography
                  sx={{
                    flex: 1,
                    fontSize: 13,
                    fontWeight: isActiveMatch ? 600 : 400,
                    color: isActiveMatch ? 'text.primary' : 'text.secondary',
                  }}
                >
                  {match.name}
                </Typography>

                {/* Rally count chip */}
                <Chip
                  label={matchRallies.length}
                  size="small"
                  sx={{
                    height: 18,
                    fontSize: 10,
                    fontWeight: 500,
                    bgcolor: isActiveMatch ? 'primary.main' : 'action.hover',
                    color: isActiveMatch ? 'primary.contrastText' : 'text.secondary',
                    '& .MuiChip-label': { px: 0.75 },
                  }}
                />

                {/* Duration */}
                <Typography
                  sx={{
                    ml: 1,
                    fontFamily: 'monospace',
                    fontSize: 10,
                    color: 'text.disabled',
                  }}
                >
                  {formatDuration(matchDuration)}
                </Typography>
              </Box>

              {/* Collapsible rally list */}
              <Collapse in={isExpanded}>
                <Box sx={{ pl: 1 }}>
                  {matchRallies.map((rally, index) => {
                    const isSelected = selectedRallyId === rally.id;
                    const isActive = isActiveMatch && activeRallyId === rally.id;

                    return (
                      <Box
                        key={rally.id}
                        onClick={() => handleRallyClick(match.id, rally)}
                        sx={{
                          display: 'flex',
                          alignItems: 'center',
                          px: 1.5,
                          py: 0.5,
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
                            width: 14,
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            mr: 0.5,
                            color: isActive
                              ? 'primary.main'
                              : isSelected
                              ? 'text.primary'
                              : 'transparent',
                            fontSize: 8,
                          }}
                        >
                          {isActive ? '●' : isSelected ? '▸' : ''}
                        </Box>

                        {/* Rally number */}
                        <Typography
                          sx={{
                            fontFamily: 'monospace',
                            fontSize: 10,
                            fontWeight: isSelected ? 600 : 400,
                            color: isSelected ? 'text.primary' : 'text.secondary',
                            width: 20,
                            flexShrink: 0,
                          }}
                        >
                          {String(index + 1).padStart(2, '0')}
                        </Typography>

                        {/* Time range */}
                        <Typography
                          sx={{
                            fontFamily: 'monospace',
                            fontSize: 10,
                            color: isSelected ? 'text.primary' : 'text.secondary',
                            flex: 1,
                            mx: 0.5,
                          }}
                        >
                          {formatTime(rally.start_time)}
                          <Box
                            component="span"
                            sx={{ color: 'text.disabled', mx: 0.25 }}
                          >
                            -
                          </Box>
                          {formatTime(rally.end_time)}
                        </Typography>

                        {/* Duration */}
                        <Typography
                          sx={{
                            fontFamily: 'monospace',
                            fontSize: 9,
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
                            <Stack direction="row" spacing={0.25} sx={{ ml: 0.5 }}>
                              {rallyHighlights.slice(0, 2).map((h) => (
                                <Box
                                  key={h.id}
                                  sx={{
                                    width: 5,
                                    height: 5,
                                    borderRadius: '50%',
                                    bgcolor: h.color,
                                  }}
                                />
                              ))}
                              {rallyHighlights.length > 2 && (
                                <Typography
                                  sx={{ fontSize: 7, color: 'text.disabled', lineHeight: 1 }}
                                >
                                  +{rallyHighlights.length - 2}
                                </Typography>
                              )}
                            </Stack>
                          );
                        })()}

                        {/* Download button */}
                        {videoSource && isActiveMatch && (
                          <Tooltip title="Download rally">
                            <IconButton
                              size="small"
                              onClick={(e) => handleDownloadRally(e, rally)}
                              disabled={isExporting}
                              sx={{
                                ml: 0.25,
                                p: 0.25,
                                opacity: isSelected || exportingRallyId === rally.id ? 1 : 0,
                                transition: 'opacity 0.15s',
                                '.MuiBox-root:hover &': { opacity: 1 },
                                color: exportingRallyId === rally.id ? 'primary.main' : 'text.secondary',
                              }}
                            >
                              {exportingRallyId === rally.id ? (
                                <CircularProgress size={10} />
                              ) : (
                                <FileDownloadIcon sx={{ fontSize: 12 }} />
                              )}
                            </IconButton>
                          </Tooltip>
                        )}
                      </Box>
                    );
                  })}
                </Box>
              </Collapse>
            </Box>
          );
        })}
      </Box>
    </Box>
  );
}
