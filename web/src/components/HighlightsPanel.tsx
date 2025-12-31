'use client';

import { useState } from 'react';
import {
  Box,
  Typography,
  IconButton,
  Chip,
  TextField,
  Tooltip,
  Popover,
  Switch,
  FormControlLabel,
  Button,
  CircularProgress,
  Stack,
} from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import StopIcon from '@mui/icons-material/Stop';
import DeleteIcon from '@mui/icons-material/Delete';
import CheckIcon from '@mui/icons-material/Check';
import CloseIcon from '@mui/icons-material/Close';
import FileDownloadIcon from '@mui/icons-material/FileDownload';
import { useEditorStore } from '@/stores/editorStore';
import { usePlayerStore } from '@/stores/playerStore';
import { useExportStore } from '@/stores/exportStore';

export function HighlightsPanel() {
  const {
    highlights,
    rallies,
    selectedHighlightId,
    selectHighlight,
    createHighlight,
    deleteHighlight,
    renameHighlight,
    canCreateHighlight,
    videoFile,
    videoUrl,
  } = useEditorStore();

  // Use File if available, otherwise use URL
  const videoSource = videoFile || videoUrl;

  const {
    playingHighlightId,
    startHighlightPlayback,
    stopHighlightPlayback,
    seek,
  } = usePlayerStore();

  const {
    isExporting,
    exportingHighlightId,
    downloadHighlight,
  } = useExportStore();

  const [editingId, setEditingId] = useState<string | null>(null);
  const [editName, setEditName] = useState('');
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);
  const [downloadAnchor, setDownloadAnchor] = useState<{ el: HTMLButtonElement; id: string } | null>(null);
  const [withFade, setWithFade] = useState(false);

  const handleCreate = () => {
    const newId = createHighlight();
    selectHighlight(newId);
  };

  const handlePlay = (highlightId: string, e: React.MouseEvent) => {
    e.stopPropagation();

    const highlight = highlights?.find((h) => h.id === highlightId);
    if (!highlight || !highlight.rallyIds || highlight.rallyIds.length === 0) return;

    // Get first rally sorted by time
    const highlightRallies = (rallies ?? [])
      .filter((s) => highlight.rallyIds.includes(s.id))
      .sort((a, b) => a.start_time - b.start_time);

    if (highlightRallies.length > 0) {
      seek(highlightRallies[0].start_time);
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

  const handleDownloadClick = (highlightId: string, e: React.MouseEvent<HTMLButtonElement>) => {
    e.stopPropagation();
    setDownloadAnchor({ el: e.currentTarget, id: highlightId });
  };

  const handleDownload = () => {
    if (!videoSource || !downloadAnchor) return;

    const highlight = highlights?.find((h) => h.id === downloadAnchor.id);
    if (!highlight) return;

    const highlightRallies = (rallies ?? [])
      .filter((r) => highlight.rallyIds?.includes(r.id))
      .sort((a, b) => a.start_time - b.start_time);

    downloadHighlight(videoSource, highlightRallies, highlight.id, highlight.name, withFade);
    setDownloadAnchor(null);
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
        {!highlights || highlights.length === 0 ? (
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
              const rallyCount = highlight.rallyIds?.length ?? 0;

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

                  {/* Rally count */}
                  <Chip
                    label={rallyCount}
                    size="small"
                    sx={{
                      height: 18,
                      fontSize: 10,
                      fontWeight: 600,
                      mx: 0.5,
                      bgcolor: 'action.selected',
                    }}
                  />

                  {/* Download button */}
                  {videoSource && (
                    <Tooltip title={rallyCount === 0 ? 'No rallies' : 'Download highlight'}>
                      <span>
                        <IconButton
                          size="small"
                          onClick={(e) => handleDownloadClick(highlight.id, e)}
                          disabled={rallyCount === 0 || isExporting}
                          sx={{ color: exportingHighlightId === highlight.id ? 'primary.main' : 'text.secondary' }}
                        >
                          {exportingHighlightId === highlight.id ? (
                            <CircularProgress size={14} />
                          ) : (
                            <FileDownloadIcon sx={{ fontSize: 16 }} />
                          )}
                        </IconButton>
                      </span>
                    </Tooltip>
                  )}

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
                    <Tooltip title={rallyCount === 0 ? 'No rallies' : 'Play highlight'}>
                      <span>
                        <IconButton
                          size="small"
                          onClick={(e) => handlePlay(highlight.id, e)}
                          disabled={rallyCount === 0}
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

      {/* Download Popover */}
      <Popover
        open={Boolean(downloadAnchor)}
        anchorEl={downloadAnchor?.el}
        onClose={() => setDownloadAnchor(null)}
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
            onClick={handleDownload}
            disabled={isExporting}
            sx={{ mt: 1.5 }}
          >
            Download
          </Button>
        </Box>
      </Popover>
    </Box>
  );
}
