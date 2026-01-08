'use client';

import { useState, useCallback, useMemo } from 'react';
import {
  Box,
  Typography,
  IconButton,
  Chip,
  Stack,
  Fab,
  Card,
  CardContent,
  CardActions,
  Collapse,
  TextField,
} from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import StopIcon from '@mui/icons-material/Stop';
import DeleteIcon from '@mui/icons-material/Delete';
import AddIcon from '@mui/icons-material/Add';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ChevronRightIcon from '@mui/icons-material/ChevronRight';
import CheckIcon from '@mui/icons-material/Check';
import CloseIcon from '@mui/icons-material/Close';
import { useEditorStore } from '@/stores/editorStore';
import { usePlayerStore } from '@/stores/playerStore';
import { formatDuration } from '@/utils/timeFormat';
import { designTokens } from '@/app/theme';
import { NamePromptModal } from '../NamePromptModal';

interface MobileHighlightsPanelProps {
  onRallyTap: (rallyId: string) => void;
}

export function MobileHighlightsPanel({ onRallyTap }: MobileHighlightsPanelProps) {
  const {
    session,
    activeMatchId,
    rallies,
    highlights,
    selectedHighlightId,
    selectHighlight,
    createHighlight,
    deleteHighlight,
    renameHighlight,
    removeRallyFromHighlight,
    canCreateHighlight,
    canEditHighlight,
    currentUserName,
  } = useEditorStore();
  const {
    playingHighlightId,
    startHighlightPlayback,
    stopHighlightPlayback,
  } = usePlayerStore();

  const [expandedHighlights, setExpandedHighlights] = useState<Set<string>>(new Set());
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editName, setEditName] = useState('');
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);
  const [showNamePrompt, setShowNamePrompt] = useState(false);
  const [pendingCreateHighlight, setPendingCreateHighlight] = useState(false);

  const toggleExpanded = (highlightId: string) => {
    setExpandedHighlights((prev) => {
      const next = new Set(prev);
      if (next.has(highlightId)) {
        next.delete(highlightId);
      } else {
        next.add(highlightId);
      }
      return next;
    });
  };

  // Get all rallies with their match info
  const sessionMatches = session?.matches;
  const allRallies = useMemo(() => {
    if (!sessionMatches) return {};
    const map: Record<string, { rally: typeof sessionMatches[0]['rallies'][0]; matchName: string }> = {};
    sessionMatches.forEach((match) => {
      // Use current rallies from store for active match
      const matchRallies = match.id === activeMatchId ? (rallies || []) : match.rallies;
      matchRallies.forEach((rally) => {
        map[rally.id] = { rally, matchName: match.name };
      });
    });
    return map;
  }, [sessionMatches, activeMatchId, rallies]);

  const handleCreateHighlight = () => {
    if (!currentUserName) {
      setPendingCreateHighlight(true);
      setShowNamePrompt(true);
      return;
    }
    const newId = createHighlight();
    selectHighlight(newId);
    setExpandedHighlights((prev) => new Set(prev).add(newId));
  };

  const handleNameSet = () => {
    if (pendingCreateHighlight) {
      setPendingCreateHighlight(false);
      const newId = createHighlight();
      selectHighlight(newId);
      setExpandedHighlights((prev) => new Set(prev).add(newId));
    }
  };

  const handlePlay = (highlightId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    const highlight = highlights.find((h) => h.id === highlightId);
    if (!highlight || highlight.rallyIds.length === 0) return;

    if (playingHighlightId === highlightId) {
      stopHighlightPlayback();
    } else {
      // Build playlist with match info
      const playlist = highlight.rallyIds
        .map((rallyId) => {
          const info = allRallies[rallyId];
          if (!info) return null;
          const matchId = rallyId.split('_rally_')[0];
          return {
            ...info.rally,
            matchId,
          };
        })
        .filter(Boolean);
      if (playlist.length > 0) {
        startHighlightPlayback(highlightId, playlist as { id: string; matchId: string; start_time: number; end_time: number }[]);
      }
    }
  };

  const handleDelete = (highlightId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (deleteConfirmId === highlightId) {
      deleteHighlight(highlightId);
      setDeleteConfirmId(null);
    } else {
      setDeleteConfirmId(highlightId);
    }
  };

  const handleCancelDelete = (e: React.MouseEvent) => {
    e.stopPropagation();
    setDeleteConfirmId(null);
  };

  const startEditing = (highlightId: string, currentName: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setEditingId(highlightId);
    setEditName(currentName);
  };

  const saveEdit = () => {
    if (editingId && editName.trim()) {
      renameHighlight(editingId, editName.trim());
    }
    setEditingId(null);
    setEditName('');
  };

  const cancelEdit = () => {
    setEditingId(null);
    setEditName('');
  };

  if (!highlights || highlights.length === 0) {
    return (
      <Box
        sx={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          height: '100%',
          p: 3,
          textAlign: 'center',
        }}
      >
        <Typography color="text.secondary" sx={{ mb: 2 }}>
          No highlights yet.
          <br />
          Create one to start collecting rallies.
        </Typography>
        <Fab
          color="primary"
          size="medium"
          onClick={handleCreateHighlight}
          disabled={!canCreateHighlight()}
        >
          <AddIcon />
        </Fab>

        <NamePromptModal
          open={showNamePrompt}
          onClose={() => {
            setShowNamePrompt(false);
            setPendingCreateHighlight(false);
          }}
          onNameSet={handleNameSet}
        />
      </Box>
    );
  }

  return (
    <Box sx={{ pb: 2 }}>
      {/* Summary */}
      <Box
        sx={{
          px: 2,
          py: 1.5,
          bgcolor: 'background.paper',
          borderBottom: '1px solid',
          borderColor: 'divider',
          position: 'sticky',
          top: 0,
          zIndex: 1,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}
      >
        <Typography variant="caption" color="text.secondary">
          {highlights.length} highlight{highlights.length !== 1 ? 's' : ''}
        </Typography>
        <IconButton
          size="small"
          color="primary"
          onClick={handleCreateHighlight}
          disabled={!canCreateHighlight()}
        >
          <AddIcon fontSize="small" />
        </IconButton>
      </Box>

      {/* Highlights List */}
      <Stack spacing={1} sx={{ p: 2 }}>
        {highlights.map((highlight) => {
          const isExpanded = expandedHighlights.has(highlight.id);
          const isPlaying = playingHighlightId === highlight.id;
          const isDeleting = deleteConfirmId === highlight.id;
          const isEditing = editingId === highlight.id;
          const canEdit = canEditHighlight(highlight.id);
          const totalDuration = highlight.rallyIds.reduce((sum, rallyId) => {
            const info = allRallies[rallyId];
            return sum + (info?.rally.duration || 0);
          }, 0);

          return (
            <Card
              key={highlight.id}
              sx={{
                bgcolor: 'background.paper',
                borderLeft: `4px solid ${highlight.color}`,
                boxShadow: isPlaying ? `0 0 12px ${highlight.color}40` : 'none',
                transition: 'box-shadow 0.2s ease',
              }}
            >
              <CardContent
                onClick={() => toggleExpanded(highlight.id)}
                sx={{
                  py: 1.5,
                  px: 2,
                  cursor: 'pointer',
                  '&:last-child': { pb: 1.5 },
                }}
              >
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <IconButton size="small" sx={{ p: 0.5 }}>
                    {isExpanded ? (
                      <ExpandMoreIcon fontSize="small" />
                    ) : (
                      <ChevronRightIcon fontSize="small" />
                    )}
                  </IconButton>

                  <Box sx={{ flex: 1, minWidth: 0 }}>
                    {isEditing ? (
                      <TextField
                        value={editName}
                        onChange={(e) => setEditName(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') saveEdit();
                          if (e.key === 'Escape') cancelEdit();
                        }}
                        onBlur={saveEdit}
                        onClick={(e) => e.stopPropagation()}
                        autoFocus
                        size="small"
                        fullWidth
                        sx={{ '& input': { py: 0.5 } }}
                      />
                    ) : (
                      <Typography
                        variant="body2"
                        fontWeight={600}
                        onDoubleClick={
                          canEdit
                            ? (e: React.MouseEvent) => startEditing(highlight.id, highlight.name, e)
                            : undefined
                        }
                        sx={{
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                        }}
                      >
                        {highlight.name}
                      </Typography>
                    )}
                    <Typography variant="caption" color="text.secondary">
                      {highlight.rallyIds.length} rallies
                      {totalDuration > 0 && ` \u2022 ${formatDuration(totalDuration)}`}
                    </Typography>
                  </Box>

                  <Chip
                    size="small"
                    label={highlight.rallyIds.length}
                    sx={{
                      bgcolor: `${highlight.color}30`,
                      color: highlight.color,
                      fontWeight: 600,
                    }}
                  />
                </Box>
              </CardContent>

              <Collapse in={isExpanded}>
                {/* Rally chips */}
                {highlight.rallyIds.length > 0 && (
                  <Box sx={{ px: 2, pb: 1 }}>
                    <Stack direction="row" flexWrap="wrap" gap={0.5}>
                      {highlight.rallyIds.map((rallyId) => {
                        const info = allRallies[rallyId];
                        if (!info) return null;
                        const rallyNumber = rallyId.split('_rally_')[1];
                        return (
                          <Chip
                            key={rallyId}
                            size="small"
                            label={`#${rallyNumber}`}
                            onClick={(e) => {
                              e.stopPropagation();
                              onRallyTap(rallyId);
                            }}
                            onDelete={
                              canEdit
                                ? (e) => {
                                    e.stopPropagation();
                                    removeRallyFromHighlight(rallyId, highlight.id);
                                  }
                                : undefined
                            }
                            sx={{
                              minHeight: 28,
                              '&:active': { transform: 'scale(0.95)' },
                            }}
                          />
                        );
                      })}
                    </Stack>
                  </Box>
                )}

                {highlight.rallyIds.length === 0 && (
                  <Typography
                    variant="caption"
                    color="text.secondary"
                    sx={{ px: 2, pb: 1, display: 'block' }}
                  >
                    No rallies yet. Tap a rally to add it.
                  </Typography>
                )}
              </Collapse>

              <CardActions sx={{ px: 2, py: 1, borderTop: '1px solid', borderColor: 'divider' }}>
                {/* Play/Stop */}
                <IconButton
                  size="small"
                  onClick={(e) => handlePlay(highlight.id, e)}
                  disabled={highlight.rallyIds.length === 0}
                  sx={{
                    color: isPlaying ? highlight.color : 'text.secondary',
                    minWidth: designTokens.mobile.touchTarget,
                    minHeight: designTokens.mobile.touchTarget,
                  }}
                >
                  {isPlaying ? <StopIcon /> : <PlayArrowIcon />}
                </IconButton>

                <Box sx={{ flex: 1 }} />

                {/* Delete with confirmation */}
                {canEdit && (
                  <>
                    {isDeleting ? (
                      <>
                        <IconButton
                          size="small"
                          onClick={handleCancelDelete}
                          sx={{ color: 'text.secondary' }}
                        >
                          <CloseIcon fontSize="small" />
                        </IconButton>
                        <IconButton
                          size="small"
                          onClick={(e) => handleDelete(highlight.id, e)}
                          sx={{ color: 'error.main' }}
                        >
                          <CheckIcon fontSize="small" />
                        </IconButton>
                      </>
                    ) : (
                      <IconButton
                        size="small"
                        onClick={(e) => handleDelete(highlight.id, e)}
                        sx={{
                          color: 'text.secondary',
                          minWidth: designTokens.mobile.touchTarget,
                          minHeight: designTokens.mobile.touchTarget,
                        }}
                      >
                        <DeleteIcon fontSize="small" />
                      </IconButton>
                    )}
                  </>
                )}
              </CardActions>
            </Card>
          );
        })}
      </Stack>

      <NamePromptModal
        open={showNamePrompt}
        onClose={() => {
          setShowNamePrompt(false);
          setPendingCreateHighlight(false);
        }}
        onNameSet={handleNameSet}
      />
    </Box>
  );
}
