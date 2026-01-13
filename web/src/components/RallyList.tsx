'use client';

import { useState, useMemo, useCallback, useEffect } from 'react';
import { useRouter, usePathname, useSearchParams } from 'next/navigation';
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
  TextField,
  Divider,
} from '@mui/material';
import FileDownloadIcon from '@mui/icons-material/FileDownload';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ChevronRightIcon from '@mui/icons-material/ChevronRight';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import DeleteIcon from '@mui/icons-material/Delete';
import EditIcon from '@mui/icons-material/Edit';
import MoreVertIcon from '@mui/icons-material/MoreVert';
import LockIcon from '@mui/icons-material/Lock';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import RestoreIcon from '@mui/icons-material/Restore';
import Menu from '@mui/material/Menu';
import MenuItem from '@mui/material/MenuItem';
import ListItemIcon from '@mui/material/ListItemIcon';
import ListItemText from '@mui/material/ListItemText';
import AddIcon from '@mui/icons-material/Add';
import { useEditorStore } from '@/stores/editorStore';
import { usePlayerStore } from '@/stores/playerStore';
import { useExportStore } from '@/stores/exportStore';
import { useTierStore } from '@/stores/tierStore';
import { formatTime, formatDuration } from '@/utils/timeFormat';
import { Rally, Match } from '@/types/rally';
import { designTokens } from '@/app/theme';
import { ConfirmDialog } from './ConfirmDialog';
import { removeVideoFromSession, renameVideo, confirmRallies, getConfirmationStatus, restoreOriginalVideo } from '@/services/api';
import { syncService } from '@/services/syncService';

export function RallyList() {
  const {
    session,
    activeMatchId,
    setActiveMatch,
    renameMatch,
    rallies,
    selectedRallyId,
    selectRally,
    getHighlightsForRally,
    reloadSession,
    isRallyEditingLocked,
    removeRally,
    setIsCameraTabActive,
    confirmationStatus,
    setConfirmationStatus,
    isConfirming,
    setIsConfirming,
    setShowAddVideoModal,
  } = useEditorStore();
  const isPaidTier = useTierStore((state) => state.isPaidTier());
  const { currentTime, seek } = usePlayerStore();
  const {
    isExporting,
    exportingRallyId,
    exportingAll,
    downloadRallyServerSide,
    downloadAllRalliesServerSide,
  } = useExportStore();
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  // Track which matches are expanded
  const [expandedMatches, setExpandedMatches] = useState<Set<string>>(
    new Set(activeMatchId ? [activeMatchId] : [])
  );

  // Popover state for Download All
  const [downloadAllAnchor, setDownloadAllAnchor] = useState<HTMLButtonElement | null>(null);
  const [withFade, setWithFade] = useState(false);

  // Delete video dialog state
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const [matchToDelete, setMatchToDelete] = useState<Match | null>(null);
  const [, setIsDeleting] = useState(false);

  // Rename video state
  const [editingMatchId, setEditingMatchId] = useState<string | null>(null);
  const [editName, setEditName] = useState('');
  const [isRenaming, setIsRenaming] = useState(false);

  // Rally action menu state
  const [rallyMenuAnchor, setRallyMenuAnchor] = useState<{
    el: HTMLElement;
    rally: Rally;
    matchId: string;
  } | null>(null);

  // Rally delete confirmation state
  const [rallyToDelete, setRallyToDelete] = useState<Rally | null>(null);

  // Video (match) action menu state
  const [videoMenuAnchor, setVideoMenuAnchor] = useState<{
    el: HTMLElement;
    match: Match;
  } | null>(null);

  // Confirm rallies dialog state
  const [showConfirmRalliesDialog, setShowConfirmRalliesDialog] = useState(false);
  const [matchToConfirm, setMatchToConfirm] = useState<Match | null>(null);

  // Restore original video dialog state
  const [showRestoreDialog, setShowRestoreDialog] = useState(false);
  const [matchToRestore, setMatchToRestore] = useState<Match | null>(null);
  const [isRestoring, setIsRestoring] = useState(false);

  // Poll for confirmation status when confirming
  useEffect(() => {
    if (!isConfirming || !activeMatchId) return;

    const status = confirmationStatus[activeMatchId];
    const isProcessing = status?.status === 'PENDING' || status?.status === 'PROCESSING';
    if (!isProcessing) return;

    const intervalId = setInterval(async () => {
      try {
        const result = await getConfirmationStatus(activeMatchId);
        if (result.confirmation) {
          setConfirmationStatus(activeMatchId, {
            id: result.confirmation.id,
            status: result.confirmation.status,
            progress: result.confirmation.progress,
            error: result.confirmation.error,
            confirmedAt: result.confirmation.confirmedAt,
            originalDurationMs: result.confirmation.originalDurationMs,
            trimmedDurationMs: result.confirmation.trimmedDurationMs,
          });

          if (result.confirmation.status === 'CONFIRMED' || result.confirmation.status === 'FAILED') {
            setIsConfirming(false);
            if (result.confirmation.status === 'CONFIRMED') {
              await reloadSession();
            }
          }
        }
      } catch (e) {
        console.error('Failed to poll confirmation status:', e);
      }
    }, 2000);

    return () => clearInterval(intervalId);
  }, [isConfirming, activeMatchId, confirmationStatus, setConfirmationStatus, setIsConfirming, reloadSession]);

  // Update URL param when switching videos
  const updateVideoUrlParam = useCallback((videoId: string) => {
    const params = new URLSearchParams(searchParams.toString());
    params.set('video', videoId);
    router.replace(`${pathname}?${params.toString()}`, { scroll: false });
  }, [pathname, searchParams, router]);

  const handleMatchClick = (match: Match) => {
    const isActiveMatch = activeMatchId === match.id;
    // Use rallies from store for active match (live updates), session for others
    const matchRallies = isActiveMatch ? rallies : match.rallies;
    const hasRallies = matchRallies.length > 0;

    if (hasRallies) {
      // Toggle expand/collapse for videos with rallies
      setExpandedMatches((prev) => {
        const next = new Set(prev);
        if (next.has(match.id)) {
          next.delete(match.id);
        } else {
          next.add(match.id);
        }
        return next;
      });
    } else {
      // Switch to video in player for videos with no rallies
      if (!isActiveMatch) {
        setActiveMatch(match.id);
        updateVideoUrlParam(match.id);
      }
    }
  };

  const handleRallyClick = useCallback((matchId: string, rally: Rally) => {
    // Switch match if needed
    if (matchId !== activeMatchId) {
      setActiveMatch(matchId);
      updateVideoUrlParam(matchId);
      // Expand the match if collapsed
      setExpandedMatches((prev) => new Set(prev).add(matchId));
    }
    selectRally(rally.id);
    seek(rally.start_time);
    // Exit camera edit mode when clicking on a rally
    setIsCameraTabActive(false);
  }, [activeMatchId, setActiveMatch, updateVideoUrlParam, selectRally, seek, setIsCameraTabActive]);

  // Get the active match from session
  const activeMatch = useMemo(
    () => session?.matches?.find(m => m.id === activeMatchId),
    [session, activeMatchId]
  );

  // Server-side export for single rally (handles confirmed videos correctly)
  const handleDownloadRally = useCallback((e: React.MouseEvent, rally: Rally, matchId: string) => {
    e.stopPropagation();
    const match = session?.matches?.find(m => m.id === matchId);
    if (!session || !match || !match.s3Key) return;

    downloadRallyServerSide(session.id, {
      id: match.id,
      s3Key: match.s3Key,
      name: match.name,
    }, rally);
  }, [session, downloadRallyServerSide]);

  // Sort active match rallies by start time - memoized
  const sortedRallies = useMemo(
    () => [...(rallies ?? [])].sort((a, b) => a.start_time - b.start_time),
    [rallies]
  );

  // Find which rally contains the current time (for active match only) - memoized
  const activeRallyId = useMemo(
    () => sortedRallies.find((s) => currentTime >= s.start_time && currentTime <= s.end_time)?.id,
    [sortedRallies, currentTime]
  );

  // Calculate total duration for active match - memoized
  const totalDuration = useMemo(
    () => sortedRallies.reduce((sum, s) => sum + s.duration, 0),
    [sortedRallies]
  );

  // Server-side export for all rallies (handles confirmed videos correctly)
  const handleDownloadAll = useCallback(() => {
    if (!session || !activeMatch || !activeMatch.s3Key || !sortedRallies.length) return;

    downloadAllRalliesServerSide(session.id, {
      id: activeMatch.id,
      s3Key: activeMatch.s3Key,
      name: activeMatch.name,
    }, sortedRallies);
    setDownloadAllAnchor(null);
  }, [session, activeMatch, sortedRallies, downloadAllRalliesServerSide]);

  const handleDeleteClick = useCallback((e: React.MouseEvent, match: Match) => {
    e.stopPropagation(); // Don't toggle expand/collapse
    setMatchToDelete(match);
    setShowDeleteDialog(true);
  }, []);

  const handleConfirmRemove = useCallback(async () => {
    if (!matchToDelete || !session) return;

    const isLastVideo = session.matches.length === 1;

    setIsDeleting(true);
    try {
      await removeVideoFromSession(session.id, matchToDelete.id);
      setShowDeleteDialog(false);
      setMatchToDelete(null);

      if (isLastVideo) {
        // Redirect to sessions page since the session will be empty
        router.push('/sessions');
      } else {
        await reloadSession();
      }
    } catch (error) {
      console.error('Failed to remove video from session:', error);
    } finally {
      setIsDeleting(false);
    }
  }, [matchToDelete, reloadSession, session, router]);

  const handleStartRename = useCallback((e: React.MouseEvent, match: Match) => {
    e.stopPropagation();
    setEditingMatchId(match.id);
    setEditName(match.name);
  }, []);

  const handleSaveRename = useCallback(async () => {
    if (!editingMatchId || !editName.trim()) {
      setEditingMatchId(null);
      return;
    }

    setIsRenaming(true);
    try {
      await renameVideo(editingMatchId, editName.trim());
      renameMatch(editingMatchId, editName.trim());
    } catch (error) {
      console.error('Failed to rename video:', error);
    } finally {
      setIsRenaming(false);
      setEditingMatchId(null);
    }
  }, [editingMatchId, editName, renameMatch]);

  const handleCancelRename = useCallback(() => {
    setEditingMatchId(null);
    setEditName('');
  }, []);

  // If no session, show empty state
  if (!session) {
    return (
      <Box
        sx={{
          flex: 1,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          p: 3,
        }}
      >
        <Typography variant="body2" sx={{ color: 'text.disabled', textAlign: 'center' }}>
          Load a JSON file to see rallies
        </Typography>
      </Box>
    );
  }

  // If session has no videos, show add video prompt
  if (!session.matches || session.matches.length === 0) {
    return (
      <Box
        sx={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          p: 3,
          gap: 2,
        }}
      >
        <Typography variant="body2" sx={{ color: 'text.secondary', textAlign: 'center' }}>
          No videos in this session.
        </Typography>
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={() => setShowAddVideoModal(true)}
          sx={{ textTransform: 'none' }}
        >
          Add Video
        </Button>
      </Box>
    );
  }

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* Summary bar */}
      {sortedRallies.length > 0 && (
        <Box
          sx={{
            px: 2,
            py: 1,
            borderBottom: '1px solid',
            borderColor: 'divider',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            bgcolor: designTokens.colors.surface[2],
          }}
        >
          <Typography variant="caption" sx={{ color: 'text.secondary' }}>
            Total: <strong>{formatDuration(totalDuration)}</strong>
          </Typography>
          <Tooltip title={!activeMatch ? 'Load a video first' : 'Download all rallies'}>
            <span>
              <IconButton
                size="small"
                disabled={!activeMatch || isExporting}
                onClick={(e) => setDownloadAllAnchor(e.currentTarget)}
                sx={{
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
        </Box>
      )}


      {/* Download All Popover */}
      <Popover
        open={Boolean(downloadAllAnchor)}
        anchorEl={downloadAllAnchor}
        onClose={() => setDownloadAllAnchor(null)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
        transformOrigin={{ vertical: 'top', horizontal: 'right' }}
      >
        <Box sx={{ p: 2, minWidth: 200 }}>
          <Typography variant="subtitle2" sx={{ mb: 1.5 }}>
            Export Options
          </Typography>
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
            sx={{ mt: 2 }}
          >
            Download All
          </Button>
        </Box>
      </Popover>

      {/* Match list with collapsible rallies */}
      <Box sx={{ flex: 1, overflow: 'auto' }}>
        {session.matches.map((match) => {
          const isExpanded = expandedMatches.has(match.id);
          const isActiveMatch = activeMatchId === match.id;
          // Use rallies from store for active match (live updates), session for others
          const matchRallies = [...(isActiveMatch ? rallies : match.rallies)].sort((a, b) => a.start_time - b.start_time);
          const matchDuration = matchRallies.reduce((sum, r) => sum + r.duration, 0);

          return (
            <Box key={match.id}>
              {/* Match header */}
              <Box
                onClick={() => handleMatchClick(match)}
                sx={{
                  display: 'flex',
                  alignItems: 'center',
                  px: 1.5,
                  py: 1,
                  cursor: 'pointer',
                  borderLeft: '3px solid',
                  borderColor: isActiveMatch ? 'primary.main' : 'transparent',
                  bgcolor: isActiveMatch ? 'action.selected' : 'transparent',
                  transition: designTokens.transitions.fast,
                  '&:hover': {
                    bgcolor: isActiveMatch
                      ? 'action.selected'
                      : 'action.hover',
                  },
                  '&:hover .video-menu-btn': { opacity: 1 },
                }}
              >
                {/* Expand/collapse icon */}
                <Box sx={{ mr: 1, display: 'flex', alignItems: 'center' }}>
                  {isExpanded ? (
                    <ExpandMoreIcon sx={{ fontSize: 20, color: 'text.secondary' }} />
                  ) : (
                    <ChevronRightIcon sx={{ fontSize: 20, color: 'text.secondary' }} />
                  )}
                </Box>

                {/* Match name - inline editing */}
                {editingMatchId === match.id ? (
                  <TextField
                    value={editName}
                    onChange={(e) => setEditName(e.target.value)}
                    onBlur={handleSaveRename}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') {
                        e.preventDefault();
                        handleSaveRename();
                      }
                      if (e.key === 'Escape') {
                        handleCancelRename();
                      }
                    }}
                    onClick={(e) => e.stopPropagation()}
                    autoFocus
                    size="small"
                    disabled={isRenaming}
                    sx={{
                      flex: 1,
                      '& .MuiInputBase-input': {
                        fontSize: '0.875rem',
                        fontWeight: isActiveMatch ? 600 : 400,
                        py: 0.25,
                        px: 0.5,
                      },
                      '& .MuiOutlinedInput-root': {
                        '& fieldset': {
                          borderColor: 'primary.main',
                        },
                      },
                    }}
                  />
                ) : (
                  <Typography
                    onDoubleClick={(e) => handleStartRename(e, match)}
                    sx={{
                      flex: 1,
                      fontSize: '0.875rem',
                      fontWeight: isActiveMatch ? 600 : 400,
                      color: isActiveMatch ? 'text.primary' : 'text.secondary',
                      cursor: 'pointer',
                    }}
                  >
                    {match.name}
                  </Typography>
                )}

                {/* Rally count chip */}
                <Chip
                  label={matchRallies.length}
                  size="small"
                  color={isActiveMatch ? 'primary' : 'default'}
                  sx={{
                    height: 20,
                    fontSize: '0.6875rem',
                    fontWeight: 600,
                  }}
                />

                {/* Duration */}
                <Typography
                  sx={{
                    ml: 1,
                    fontFamily: 'monospace',
                    fontSize: '0.6875rem',
                    color: 'text.disabled',
                  }}
                >
                  {formatDuration(matchDuration)}
                </Typography>

                {/* Locked indicator */}
                {confirmationStatus[match.id]?.status === 'CONFIRMED' && (
                  <Tooltip title="Rallies confirmed">
                    <LockIcon sx={{ fontSize: 14, color: 'success.main', ml: 0.5 }} />
                  </Tooltip>
                )}

                {/* Processing indicator */}
                {(confirmationStatus[match.id]?.status === 'PENDING' || confirmationStatus[match.id]?.status === 'PROCESSING') && (
                  <CircularProgress size={14} sx={{ ml: 0.5 }} />
                )}

                {/* Video menu button */}
                <IconButton
                  className="video-menu-btn"
                  size="small"
                  onClick={(e) => {
                    e.stopPropagation();
                    setVideoMenuAnchor({ el: e.currentTarget, match });
                  }}
                  sx={{
                    ml: 0.5,
                    p: 0.25,
                    opacity: 0,
                    transition: 'opacity 0.15s',
                    color: 'text.secondary',
                  }}
                >
                  <MoreVertIcon sx={{ fontSize: 16 }} />
                </IconButton>
              </Box>

              {/* Collapsible rally list */}
              <Collapse in={isExpanded}>
                <Box sx={{ pl: 1.5 }}>
                  {matchRallies.map((rally, index) => {
                    const isSelected = selectedRallyId === rally.id;
                    const isActive = isActiveMatch && activeRallyId === rally.id;
                    const rallyHighlights = getHighlightsForRally(rally.id);

                    return (
                      <Box
                        key={rally.id}
                        onClick={() => handleRallyClick(match.id, rally)}
                        sx={{
                          display: 'flex',
                          alignItems: 'center',
                          px: 1.5,
                          py: 0.75,
                          cursor: 'pointer',
                          borderLeft: '2px solid',
                          borderColor: isActive ? 'secondary.main' : 'transparent',
                          bgcolor: isSelected
                            ? 'action.selected'
                            : 'transparent',
                          transition: designTokens.transitions.fast,
                          '&:hover': {
                            bgcolor: isSelected
                              ? 'action.selected'
                              : 'action.hover',
                          },
                          '&:hover .more-btn': { opacity: 1 },
                        }}
                      >
                        {/* Playing indicator */}
                        <Box
                          sx={{
                            width: 16,
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            mr: 0.5,
                          }}
                        >
                          {isActive ? (
                            <PlayArrowIcon sx={{ fontSize: 12, color: 'secondary.main' }} />
                          ) : isSelected ? (
                            <ChevronRightIcon sx={{ fontSize: 12, color: 'primary.main' }} />
                          ) : null}
                        </Box>

                        {/* Rally number */}
                        <Typography
                          sx={{
                            fontFamily: 'monospace',
                            fontSize: '0.6875rem',
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
                            fontSize: '0.6875rem',
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
                            fontSize: '0.625rem',
                            color: 'text.disabled',
                            flexShrink: 0,
                          }}
                        >
                          {rally.duration.toFixed(1)}s
                        </Typography>

                        {/* Highlight color dots */}
                        {rallyHighlights.length > 0 && (
                          <Stack direction="row" spacing={0.25} sx={{ ml: 1 }}>
                            {rallyHighlights.slice(0, 3).map((h) => (
                              <Box
                                key={h.id}
                                sx={{
                                  width: 6,
                                  height: 6,
                                  borderRadius: '50%',
                                  bgcolor: h.color,
                                  boxShadow: `0 0 4px ${h.color}`,
                                }}
                              />
                            ))}
                            {rallyHighlights.length > 3 && (
                              <Typography
                                sx={{ fontSize: '0.5rem', color: 'text.disabled' }}
                              >
                                +{rallyHighlights.length - 3}
                              </Typography>
                            )}
                          </Stack>
                        )}

                        {/* More menu button */}
                        {match.s3Key && isActiveMatch && (
                          <IconButton
                            className="more-btn"
                            size="small"
                            onClick={(e) => {
                              e.stopPropagation();
                              setRallyMenuAnchor({ el: e.currentTarget, rally, matchId: match.id });
                            }}
                            sx={{
                              ml: 0.5,
                              p: 0.25,
                              opacity: isSelected || exportingRallyId === rally.id ? 1 : 0,
                              transition: 'opacity 0.15s',
                              color: exportingRallyId === rally.id ? 'primary.main' : 'text.secondary',
                            }}
                          >
                            {exportingRallyId === rally.id ? (
                              <CircularProgress size={12} />
                            ) : (
                              <MoreVertIcon sx={{ fontSize: 14 }} />
                            )}
                          </IconButton>
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

      {/* Rally action menu */}
      <Menu
        anchorEl={rallyMenuAnchor?.el}
        open={Boolean(rallyMenuAnchor)}
        onClose={() => setRallyMenuAnchor(null)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
        transformOrigin={{ vertical: 'top', horizontal: 'right' }}
      >
        <MenuItem
          onClick={(e) => {
            handleDownloadRally(e, rallyMenuAnchor!.rally, rallyMenuAnchor!.matchId);
            setRallyMenuAnchor(null);
          }}
          disabled={isExporting}
        >
          <ListItemIcon>
            {exportingRallyId === rallyMenuAnchor?.rally.id ? (
              <CircularProgress size={16} />
            ) : (
              <FileDownloadIcon fontSize="small" />
            )}
          </ListItemIcon>
          <ListItemText>Download</ListItemText>
        </MenuItem>
        {/* Hide delete option when rallies are confirmed */}
        {!isRallyEditingLocked() && (
          <MenuItem
            onClick={() => {
              setRallyToDelete(rallyMenuAnchor!.rally);
              setRallyMenuAnchor(null);
            }}
            sx={{ color: 'error.main' }}
          >
            <ListItemIcon>
              <DeleteIcon fontSize="small" sx={{ color: 'error.main' }} />
            </ListItemIcon>
            <ListItemText>Delete</ListItemText>
          </MenuItem>
        )}
      </Menu>

      {/* Video (match) action menu */}
      <Menu
        anchorEl={videoMenuAnchor?.el}
        open={Boolean(videoMenuAnchor)}
        onClose={() => setVideoMenuAnchor(null)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
        transformOrigin={{ vertical: 'top', horizontal: 'right' }}
      >
        <MenuItem
          onClick={() => {
            const matchToRename = videoMenuAnchor?.match;
            setVideoMenuAnchor(null);
            if (matchToRename) {
              // Delay to ensure menu is fully closed before starting rename
              requestAnimationFrame(() => {
                setEditingMatchId(matchToRename.id);
                setEditName(matchToRename.name);
              });
            }
          }}
        >
          <ListItemIcon>
            <EditIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText>Rename</ListItemText>
        </MenuItem>
        <MenuItem
          onClick={() => {
            if (videoMenuAnchor) {
              setMatchToDelete(videoMenuAnchor.match);
              setShowDeleteDialog(true);
            }
            setVideoMenuAnchor(null);
          }}
        >
          <ListItemIcon>
            <DeleteIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText>Remove from session</ListItemText>
        </MenuItem>
        <Divider />
        {confirmationStatus[videoMenuAnchor?.match.id ?? '']?.status === 'CONFIRMED' ? (
          <MenuItem
            onClick={() => {
              if (!videoMenuAnchor) return;
              setMatchToRestore(videoMenuAnchor.match);
              setVideoMenuAnchor(null);
              setShowRestoreDialog(true);
            }}
            disabled={isRestoring}
          >
            <ListItemIcon>
              <RestoreIcon fontSize="small" />
            </ListItemIcon>
            <ListItemText
              primary="Restore Original"
              secondary="Revert to full video"
            />
          </MenuItem>
        ) : (
          <MenuItem
            onClick={() => {
              if (!videoMenuAnchor || !isPaidTier) return;
              setMatchToConfirm(videoMenuAnchor.match);
              setVideoMenuAnchor(null);
              setShowConfirmRalliesDialog(true);
            }}
            disabled={
              !isPaidTier ||
              isConfirming ||
              (videoMenuAnchor?.match.rallies.length ?? 0) === 0 ||
              confirmationStatus[videoMenuAnchor?.match.id ?? '']?.status === 'PENDING' ||
              confirmationStatus[videoMenuAnchor?.match.id ?? '']?.status === 'PROCESSING'
            }
          >
            <ListItemIcon>
              {!isPaidTier ? (
                <LockIcon fontSize="small" />
              ) : (
                <CheckCircleIcon fontSize="small" />
              )}
            </ListItemIcon>
            <ListItemText
              primary="Confirm Rallies"
              secondary={
                !isPaidTier
                  ? 'Paid feature'
                  : (videoMenuAnchor?.match.rallies.length ?? 0) === 0
                  ? 'Add rallies first'
                  : 'Trim video to keep only rallies'
              }
            />
          </MenuItem>
        )}
      </Menu>

      {/* Remove video from session confirmation dialog */}
      <ConfirmDialog
        open={showDeleteDialog}
        title={session?.matches.length === 1 ? "Remove last video?" : "Remove video from session?"}
        message={
          session?.matches.length === 1
            ? `This is the last video in this session. Removing "${matchToDelete?.name}" will leave the session empty. The video itself will remain in your library.`
            : `Remove "${matchToDelete?.name}" from this session? The video will remain in your library. Rallies and highlights associated with this video will be removed from the session.`
        }
        confirmLabel="Remove"
        cancelLabel="Cancel"
        onConfirm={handleConfirmRemove}
        onCancel={() => {
          setShowDeleteDialog(false);
          setMatchToDelete(null);
        }}
      />

      {/* Delete rally confirmation dialog */}
      <ConfirmDialog
        open={Boolean(rallyToDelete)}
        title="Delete rally?"
        message="This will permanently delete this rally. This action cannot be undone."
        confirmLabel="Delete"
        cancelLabel="Cancel"
        onConfirm={() => {
          if (rallyToDelete) {
            removeRally(rallyToDelete.id);
            setRallyToDelete(null);
          }
        }}
        onCancel={() => setRallyToDelete(null)}
      />

      {/* Confirm rallies confirmation dialog */}
      <ConfirmDialog
        open={showConfirmRalliesDialog}
        title="Confirm rallies?"
        message="This will create a new trimmed video containing only the rally segments. Dead time between rallies will be removed, and rally editing will be locked."
        confirmLabel="Confirm"
        cancelLabel="Cancel"
        onConfirm={async () => {
          if (!matchToConfirm) return;
          setShowConfirmRalliesDialog(false);
          setIsConfirming(true);
          try {
            // Sync pending changes to ensure rallies are in the database
            await syncService.syncNow();
            const result = await confirmRallies(matchToConfirm.id);
            setConfirmationStatus(matchToConfirm.id, {
              id: result.confirmationId,
              status: result.status,
              progress: result.progress,
              error: null,
              confirmedAt: null,
              originalDurationMs: 0,
              trimmedDurationMs: null,
            });
          } catch (error) {
            console.error('Failed to confirm rallies:', error);
            setIsConfirming(false);
          }
          setMatchToConfirm(null);
        }}
        onCancel={() => {
          setShowConfirmRalliesDialog(false);
          setMatchToConfirm(null);
        }}
      />

      {/* Restore original video confirmation dialog */}
      <ConfirmDialog
        open={showRestoreDialog}
        title="Restore original video?"
        message="This will delete the trimmed video and restore rally timestamps to their original values. You'll need to confirm again to create a new trimmed video."
        confirmLabel="Restore"
        cancelLabel="Cancel"
        onConfirm={async () => {
          if (!matchToRestore) return;
          setShowRestoreDialog(false);
          setIsRestoring(true);
          try {
            await restoreOriginalVideo(matchToRestore.id);
            setConfirmationStatus(matchToRestore.id, null);
            await reloadSession();
          } catch (error) {
            console.error('Failed to restore original video:', error);
          } finally {
            setIsRestoring(false);
            setMatchToRestore(null);
          }
        }}
        onCancel={() => {
          setShowRestoreDialog(false);
          setMatchToRestore(null);
        }}
      />
    </Box>
  );
}
