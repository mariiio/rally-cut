'use client';

import { useState } from 'react';
import {
  Box,
  Typography,
  IconButton,
  Stack,
  Chip,
  TextField,
  Tooltip,
} from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import StopIcon from '@mui/icons-material/Stop';
import DeleteIcon from '@mui/icons-material/Delete';
import CheckIcon from '@mui/icons-material/Check';
import CloseIcon from '@mui/icons-material/Close';
import { useEditorStore } from '@/stores/editorStore';
import { usePlayerStore } from '@/stores/playerStore';

export function HighlightsPanel() {
  const {
    highlights,
    segments,
    selectedHighlightId,
    selectHighlight,
    createHighlight,
    deleteHighlight,
    renameHighlight,
    canCreateHighlight,
  } = useEditorStore();

  const {
    playingHighlightId,
    startHighlightPlayback,
    stopHighlightPlayback,
    seek,
  } = usePlayerStore();

  const [editingId, setEditingId] = useState<string | null>(null);
  const [editName, setEditName] = useState('');
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);

  const handleCreate = () => {
    const newId = createHighlight();
    selectHighlight(newId);
  };

  const handlePlay = (highlightId: string, e: React.MouseEvent) => {
    e.stopPropagation();

    const highlight = highlights.find((h) => h.id === highlightId);
    if (!highlight || highlight.segmentIds.length === 0) return;

    // Get first segment sorted by time
    const highlightSegments = segments
      .filter((s) => highlight.segmentIds.includes(s.id))
      .sort((a, b) => a.start_time - b.start_time);

    if (highlightSegments.length > 0) {
      seek(highlightSegments[0].start_time);
      startHighlightPlayback(highlightId);
    }
  };

  const handleStop = (e: React.MouseEvent) => {
    e.stopPropagation();
    stopHighlightPlayback();
  };

  const handleStartEdit = (highlight: { id: string; name: string }, e: React.MouseEvent) => {
    e.stopPropagation();
    setEditingId(highlight.id);
    setEditName(highlight.name);
  };

  const handleSaveEdit = () => {
    if (editingId && editName.trim()) {
      renameHighlight(editingId, editName.trim());
    }
    setEditingId(null);
  };

  const handleDeleteClick = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setDeleteConfirmId(id);
  };

  const handleConfirmDelete = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    deleteHighlight(id);
    setDeleteConfirmId(null);
  };

  const handleCancelDelete = (e: React.MouseEvent) => {
    e.stopPropagation();
    setDeleteConfirmId(null);
  };

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
        <Typography
          variant="overline"
          sx={{
            fontSize: 11,
            fontWeight: 600,
            letterSpacing: 1,
            color: 'text.secondary',
          }}
        >
          Highlights
        </Typography>
        <Tooltip
          title={
            canCreateHighlight()
              ? 'Create highlight'
              : 'All highlights must have segments first'
          }
        >
          <span>
            <IconButton
              size="small"
              onClick={handleCreate}
              disabled={!canCreateHighlight()}
              color="primary"
            >
              <AddIcon fontSize="small" />
            </IconButton>
          </span>
        </Tooltip>
      </Box>

      {/* Highlights list */}
      <Box sx={{ flex: 1, overflow: 'auto' }}>
        {highlights.length === 0 ? (
          <Box
            sx={{
              p: 2,
              textAlign: 'center',
              color: 'text.disabled',
            }}
          >
            <Typography variant="caption">No highlights yet</Typography>
            <Typography
              variant="caption"
              sx={{ display: 'block', mt: 0.5, fontSize: 10 }}
            >
              Select a segment and press Enter
            </Typography>
          </Box>
        ) : (
          <Box sx={{ py: 0.5 }}>
            {highlights.map((highlight) => {
              const isSelected = selectedHighlightId === highlight.id;
              const isPlaying = playingHighlightId === highlight.id;
              const segmentCount = highlight.segmentIds.length;

              return (
                <Box
                  key={highlight.id}
                  onClick={() =>
                    selectHighlight(isSelected ? null : highlight.id)
                  }
                  sx={{
                    display: 'flex',
                    alignItems: 'center',
                    px: 1.5,
                    py: 0.75,
                    cursor: 'pointer',
                    borderLeft: '3px solid',
                    borderColor: isSelected ? highlight.color : 'transparent',
                    bgcolor: isSelected
                      ? 'rgba(255, 255, 255, 0.08)'
                      : isPlaying
                      ? 'rgba(255, 255, 255, 0.04)'
                      : 'transparent',
                    transition: 'all 0.15s ease',
                    '&:hover': {
                      bgcolor: isSelected
                        ? 'rgba(255, 255, 255, 0.12)'
                        : 'rgba(255, 255, 255, 0.04)',
                    },
                  }}
                >
                  {/* Color dot */}
                  <Box
                    sx={{
                      width: 12,
                      height: 12,
                      borderRadius: '50%',
                      bgcolor: highlight.color,
                      mr: 1,
                      flexShrink: 0,
                      boxShadow: isPlaying
                        ? `0 0 8px ${highlight.color}`
                        : 'none',
                    }}
                  />

                  {/* Name (editable) */}
                  {editingId === highlight.id ? (
                    <TextField
                      value={editName}
                      onChange={(e) => setEditName(e.target.value)}
                      onBlur={handleSaveEdit}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') handleSaveEdit();
                        if (e.key === 'Escape') setEditingId(null);
                        e.stopPropagation();
                      }}
                      onClick={(e) => e.stopPropagation()}
                      size="small"
                      autoFocus
                      sx={{
                        flex: 1,
                        '& .MuiInputBase-input': {
                          fontSize: 13,
                          py: 0.25,
                        },
                      }}
                    />
                  ) : (
                    <Typography
                      onDoubleClick={(e) => handleStartEdit(highlight, e)}
                      sx={{
                        flex: 1,
                        fontSize: 13,
                        fontWeight: isSelected ? 500 : 400,
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                      }}
                    >
                      {highlight.name}
                    </Typography>
                  )}

                  {/* Segment count */}
                  <Chip
                    label={segmentCount}
                    size="small"
                    sx={{
                      height: 18,
                      fontSize: 10,
                      fontWeight: 600,
                      mx: 0.5,
                      bgcolor: 'action.selected',
                    }}
                  />

                  {/* Play/Stop button */}
                  {isPlaying ? (
                    <IconButton
                      size="small"
                      onClick={handleStop}
                      sx={{ color: highlight.color }}
                    >
                      <StopIcon sx={{ fontSize: 16 }} />
                    </IconButton>
                  ) : (
                    <Tooltip title={segmentCount === 0 ? 'No segments' : 'Play highlight'}>
                      <span>
                        <IconButton
                          size="small"
                          onClick={(e) => handlePlay(highlight.id, e)}
                          disabled={segmentCount === 0}
                          sx={{ color: 'text.secondary' }}
                        >
                          <PlayArrowIcon sx={{ fontSize: 16 }} />
                        </IconButton>
                      </span>
                    </Tooltip>
                  )}

                  {/* Delete button with confirmation */}
                  {deleteConfirmId === highlight.id ? (
                    <Stack direction="row" spacing={0.25}>
                      <Tooltip title="Confirm delete">
                        <IconButton
                          size="small"
                          onClick={(e) => handleConfirmDelete(highlight.id, e)}
                          sx={{
                            color: 'white',
                            bgcolor: '#d32f2f',
                            width: 22,
                            height: 22,
                            '&:hover': { bgcolor: '#f44336' },
                          }}
                        >
                          <CheckIcon sx={{ fontSize: 14 }} />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="Cancel">
                        <IconButton
                          size="small"
                          onClick={handleCancelDelete}
                          sx={{
                            color: 'text.secondary',
                            width: 22,
                            height: 22,
                            '&:hover': { bgcolor: 'action.hover' },
                          }}
                        >
                          <CloseIcon sx={{ fontSize: 14 }} />
                        </IconButton>
                      </Tooltip>
                    </Stack>
                  ) : (
                    <IconButton
                      size="small"
                      onClick={(e) => handleDeleteClick(highlight.id, e)}
                      sx={{ color: 'text.secondary' }}
                    >
                      <DeleteIcon sx={{ fontSize: 16 }} />
                    </IconButton>
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
