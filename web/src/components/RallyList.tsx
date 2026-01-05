'use client';

import { useState, useMemo, useCallback } from 'react';
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
} from '@mui/material';
import FileDownloadIcon from '@mui/icons-material/FileDownload';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ChevronRightIcon from '@mui/icons-material/ChevronRight';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import DeleteIcon from '@mui/icons-material/Delete';
import EditIcon from '@mui/icons-material/Edit';
import MoreVertIcon from '@mui/icons-material/MoreVert';
import Menu from '@mui/material/Menu';
import MenuItem from '@mui/material/MenuItem';
import ListItemIcon from '@mui/material/ListItemIcon';
import ListItemText from '@mui/material/ListItemText';
import { useEditorStore } from '@/stores/editorStore';
import { usePlayerStore } from '@/stores/playerStore';
import { useExportStore } from '@/stores/exportStore';
import { useTierStore } from '@/stores/tierStore';
import { formatTime, formatDuration } from '@/utils/timeFormat';
import { Rally, Match } from '@/types/rally';
import { designTokens } from '@/app/theme';
import { ConfirmDialog } from './ConfirmDialog';
import { ConfirmRallies, LockedRalliesBanner } from './ConfirmRallies';
import { deleteVideo, renameVideo } from '@/services/api';

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
    videoFile,
    videoUrl,
    reloadSession,
    isRallyEditingLocked,
    removeRally,
    setIsCameraTabActive,
  } = useEditorStore();
  const isPremium = useTierStore((state) => state.isPremium());
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

  // Delete video dialog state
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const [matchToDelete, setMatchToDelete] = useState<Match | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);

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

  const handleRallyClick = useCallback((matchId: string, rally: Rally) => {
    // Switch match if needed
    if (matchId !== activeMatchId) {
      setActiveMatch(matchId);
      // Expand the match if collapsed
      setExpandedMatches((prev) => new Set(prev).add(matchId));
    }
    selectRally(rally.id);
    seek(rally.start_time);
    // Exit camera edit mode when clicking on a rally normally
    setIsCameraTabActive(false);
  }, [activeMatchId, setActiveMatch, selectRally, seek, setIsCameraTabActive]);

  const handleDownloadRally = useCallback((e: React.MouseEvent, rally: Rally) => {
    e.stopPropagation();
    if (!videoSource) return;
    downloadRally(videoSource, rally);
  }, [videoSource, downloadRally]);

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

  const handleDownloadAll = useCallback(() => {
    if (!videoSource || !sortedRallies.length) return;
    downloadAllRallies(videoSource, sortedRallies, withFade);
    setDownloadAllAnchor(null);
  }, [videoSource, sortedRallies, withFade, downloadAllRallies]);

  const handleDeleteClick = useCallback((e: React.MouseEvent, match: Match) => {
    e.stopPropagation(); // Don't toggle expand/collapse
    setMatchToDelete(match);
    setShowDeleteDialog(true);
  }, []);

  const handleConfirmDelete = useCallback(async () => {
    if (!matchToDelete) return;

    setIsDeleting(true);
    try {
      await deleteVideo(matchToDelete.id);
      await reloadSession();
      setShowDeleteDialog(false);
      setMatchToDelete(null);
    } catch (error) {
      console.error('Failed to delete video:', error);
    } finally {
      setIsDeleting(false);
    }
  }, [matchToDelete, reloadSession]);

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
          <Tooltip title={!videoSource ? 'Load a video first' : 'Download all rallies'}>
            <span>
              <IconButton
                size="small"
                disabled={!videoSource || isExporting}
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

      {/* Locked rallies banner */}
      <LockedRalliesBanner />

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
                  '&:hover .delete-btn, &:hover .edit-btn': { opacity: 1 },
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
                      cursor: 'text',
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

                {/* Edit button */}
                <Tooltip title="Rename video">
                  <IconButton
                    className="edit-btn"
                    size="small"
                    onClick={(e) => handleStartRename(e, match)}
                    disabled={isRenaming || editingMatchId === match.id}
                    sx={{
                      ml: 0.5,
                      p: 0.25,
                      opacity: 0,
                      transition: 'opacity 0.15s',
                      color: 'text.secondary',
                      '&:hover': {
                        color: 'primary.main',
                      },
                    }}
                  >
                    <EditIcon sx={{ fontSize: 16 }} />
                  </IconButton>
                </Tooltip>

                {/* Delete button */}
                {session.matches.length > 1 ? (
                  <Tooltip title="Delete video">
                    <IconButton
                      className="delete-btn"
                      size="small"
                      onClick={(e) => handleDeleteClick(e, match)}
                      disabled={isDeleting}
                      sx={{
                        ml: 0.5,
                        p: 0.25,
                        opacity: 0,
                        transition: 'opacity 0.15s',
                        color: 'text.secondary',
                        '&:hover': {
                          color: 'error.main',
                        },
                      }}
                    >
                      <DeleteIcon sx={{ fontSize: 16 }} />
                    </IconButton>
                  </Tooltip>
                ) : (
                  <Tooltip title="Cannot delete the only video">
                    <span>
                      <IconButton
                        className="delete-btn"
                        size="small"
                        disabled
                        sx={{
                          ml: 0.5,
                          p: 0.25,
                          opacity: 0,
                          transition: 'opacity 0.15s',
                        }}
                      >
                        <DeleteIcon sx={{ fontSize: 16 }} />
                      </IconButton>
                    </span>
                  </Tooltip>
                )}
              </Box>

              {/* Collapsible rally list */}
              <Collapse in={isExpanded}>
                {/* Confirm rallies action for active match */}
                {isActiveMatch && (
                  <Box sx={{ px: 1.5, py: 1, borderBottom: '1px solid', borderColor: 'divider' }}>
                    <ConfirmRallies matchId={match.id} isPremium={isPremium} />
                  </Box>
                )}
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
                        {videoSource && isActiveMatch && (
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
            handleDownloadRally(e, rallyMenuAnchor!.rally);
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
        <MenuItem
          onClick={() => {
            setRallyToDelete(rallyMenuAnchor!.rally);
            setRallyMenuAnchor(null);
          }}
          disabled={isRallyEditingLocked()}
          sx={{ color: 'error.main' }}
        >
          <ListItemIcon>
            <DeleteIcon fontSize="small" sx={{ color: 'error.main' }} />
          </ListItemIcon>
          <ListItemText>Delete</ListItemText>
        </MenuItem>
      </Menu>

      {/* Delete video confirmation dialog */}
      <ConfirmDialog
        open={showDeleteDialog}
        title="Delete video?"
        message={`This will permanently delete "${matchToDelete?.name}" and all its rallies. This action cannot be undone.`}
        confirmLabel="Delete"
        cancelLabel="Keep video"
        onConfirm={handleConfirmDelete}
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
    </Box>
  );
}
