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
  Collapse,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
} from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import StopIcon from '@mui/icons-material/Stop';
import DeleteIcon from '@mui/icons-material/Delete';
import CheckIcon from '@mui/icons-material/Check';
import CloseIcon from '@mui/icons-material/Close';
import FileDownloadIcon from '@mui/icons-material/FileDownload';
import EditIcon from '@mui/icons-material/Edit';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ChevronRightIcon from '@mui/icons-material/ChevronRight';
import DragIndicatorIcon from '@mui/icons-material/DragIndicator';
import {
  DndContext,
  closestCenter,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors,
  DragEndEvent,
} from '@dnd-kit/core';
import {
  SortableContext,
  sortableKeyboardCoordinates,
  useSortable,
  verticalListSortingStrategy,
} from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import { useEditorStore } from '@/stores/editorStore';
import { usePlayerStore } from '@/stores/playerStore';
import { useExportStore } from '@/stores/exportStore';
import { Rally } from '@/types/rally';

// Helper to format time as MM:SS
function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

// Sortable rally item component
interface SortableRallyItemProps {
  rally: Rally;
  index: number;
  matchName: string;
  isActive: boolean;
  onClick: () => void;
  onRemove: () => void;
}

function SortableRallyItem({ rally, index, matchName, isActive, onClick, onRemove }: SortableRallyItemProps) {
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging,
  } = useSortable({ id: rally.id });

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.5 : 1,
  };

  return (
    <ListItem
      ref={setNodeRef}
      style={style}
      onClick={onClick}
      sx={{
        py: 0.25,
        px: 0.5,
        minHeight: 28,
        cursor: 'pointer',
        bgcolor: isActive ? 'rgba(255, 255, 255, 0.12)' : 'transparent',
        '&:hover': { bgcolor: 'rgba(255, 255, 255, 0.06)' },
        '&:hover .remove-btn': { opacity: 1 },
      }}
    >
      <ListItemIcon
        {...attributes}
        {...listeners}
        sx={{ minWidth: 20, cursor: 'grab' }}
      >
        <DragIndicatorIcon sx={{ fontSize: 14, color: 'text.disabled' }} />
      </ListItemIcon>

      <ListItemText
        primary={
          <Typography component="span" sx={{ fontSize: 11, fontFamily: 'monospace' }}>
            {String(index + 1).padStart(2, '0')} {formatTime(rally.start_time)}-{formatTime(rally.end_time)}
          </Typography>
        }
        secondary={
          <Typography component="span" sx={{ fontSize: 9, color: 'text.disabled' }}>
            {matchName} ({(rally.end_time - rally.start_time).toFixed(1)}s)
          </Typography>
        }
        sx={{ my: 0 }}
      />

      <IconButton
        className="remove-btn"
        size="small"
        onClick={(e) => { e.stopPropagation(); onRemove(); }}
        sx={{
          opacity: 0,
          p: 0.25,
          transition: 'opacity 0.15s',
        }}
      >
        <CloseIcon sx={{ fontSize: 12 }} />
      </IconButton>
    </ListItem>
  );
}

export function HighlightsPanel() {
  const {
    highlights,
    selectedHighlightId,
    selectHighlight,
    createHighlight,
    deleteHighlight,
    renameHighlight,
    canCreateHighlight,
    videoFile,
    videoUrl,
    getAllRallies,
    getRallyMatch,
    setActiveMatch,
    activeMatchId,
    reorderHighlightRallies,
    removeRallyFromHighlight,
  } = useEditorStore();

  // Get all rallies across all matches for cross-match highlights
  const allRallies = getAllRallies();

  // Use File if available, otherwise use URL
  const videoSource = videoFile || videoUrl;

  const {
    playingHighlightId,
    startHighlightPlayback,
    stopHighlightPlayback,
    seek,
    highlightRallyIndex,
    getCurrentPlaylistRally,
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
  const [expandedHighlights, setExpandedHighlights] = useState<Set<string>>(new Set());

  // DnD sensors
  const sensors = useSensors(
    useSensor(PointerSensor, {
      activationConstraint: { distance: 5 },
    }),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    })
  );

  const toggleExpanded = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setExpandedHighlights((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const handleDragEnd = (highlightId: string, event: DragEndEvent) => {
    const { active, over } = event;

    if (over && active.id !== over.id) {
      const highlight = highlights?.find((h) => h.id === highlightId);
      if (!highlight) return;

      const oldIndex = highlight.rallyIds.indexOf(active.id as string);
      const newIndex = highlight.rallyIds.indexOf(over.id as string);

      if (oldIndex !== -1 && newIndex !== -1) {
        reorderHighlightRallies(highlightId, oldIndex, newIndex);
      }
    }
  };

  const handleRallyClick = (rally: Rally) => {
    const match = getRallyMatch(rally.id);
    if (match && match.id !== activeMatchId) {
      setActiveMatch(match.id);
    }
    seek(rally.start_time);
  };

  const handleCreate = () => {
    const newId = createHighlight();
    selectHighlight(newId);
  };

  const handlePlay = (highlightId: string, e: React.MouseEvent) => {
    e.stopPropagation();

    const highlight = highlights?.find((h) => h.id === highlightId);
    if (!highlight || !highlight.rallyIds || highlight.rallyIds.length === 0) return;

    // Get rallies in custom order (rallyIds order)
    const highlightRallies = highlight.rallyIds
      .map((id) => allRallies.find((r) => r.id === id))
      .filter((r): r is Rally => r !== undefined);

    if (highlightRallies.length > 0) {
      // Build playlist with match IDs for the player store
      const playlist = highlightRallies.map((rally) => {
        const match = getRallyMatch(rally.id);
        return {
          id: rally.id,
          matchId: match?.id || '',
          start_time: rally.start_time,
          end_time: rally.end_time,
        };
      });

      const firstRally = playlist[0];
      // Switch to the match containing the first rally if needed
      if (firstRally.matchId) {
        setActiveMatch(firstRally.matchId);
      }
      seek(firstRally.start_time);
      startHighlightPlayback(highlightId, playlist);
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

    // Get rallies in custom order (rallyIds order)
    const highlightRallies = highlight.rallyIds
      .map((id) => allRallies.find((r) => r.id === id))
      .filter((r): r is Rally => r !== undefined);

    // Note: Cross-match export would need special handling in exportStore
    // For now, this will only export rallies from the current video source
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
              const isExpanded = expandedHighlights.has(highlight.id);

              // Get rallies in custom order (rallyIds order)
              const orderedRallies = highlight.rallyIds
                .map((id) => allRallies.find((r) => r.id === id))
                .filter((r): r is Rally => r !== undefined);

              // Get current playing rally for this highlight
              const currentPlaylistRally = getCurrentPlaylistRally();

              return (
                <Box key={highlight.id}>
                  {/* Highlight row */}
                  <Box
                    onClick={() =>
                      selectHighlight(isSelected ? null : highlight.id)
                    }
                    sx={{
                      display: 'flex',
                      alignItems: 'center',
                      px: 1,
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
                    {/* Expand/collapse icon */}
                    <IconButton
                      size="small"
                      onClick={(e) => toggleExpanded(highlight.id, e)}
                      disabled={rallyCount === 0}
                      sx={{ p: 0.25, mr: 0.5 }}
                    >
                      {isExpanded ? (
                        <ExpandMoreIcon sx={{ fontSize: 16, color: 'text.secondary' }} />
                      ) : (
                        <ChevronRightIcon sx={{ fontSize: 16, color: rallyCount === 0 ? 'text.disabled' : 'text.secondary' }} />
                      )}
                    </IconButton>

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

                    {/* Edit button */}
                    {editingId !== highlight.id && (
                      <Tooltip title="Rename">
                        <IconButton
                          size="small"
                          onClick={(e) => handleStartEdit(highlight, e)}
                          sx={{ color: 'text.secondary' }}
                        >
                          <EditIcon sx={{ fontSize: 16 }} />
                        </IconButton>
                      </Tooltip>
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

                  {/* Collapsible rally list */}
                  <Collapse in={isExpanded && rallyCount > 0}>
                    <Box sx={{ pl: 4, pr: 1, bgcolor: 'rgba(0, 0, 0, 0.15)' }}>
                      <DndContext
                        sensors={sensors}
                        collisionDetection={closestCenter}
                        onDragEnd={(event) => handleDragEnd(highlight.id, event)}
                      >
                        <SortableContext
                          items={orderedRallies.map((r) => r.id)}
                          strategy={verticalListSortingStrategy}
                        >
                          <List dense disablePadding>
                            {orderedRallies.map((rally, index) => (
                              <SortableRallyItem
                                key={rally.id}
                                rally={rally}
                                index={index}
                                matchName={getRallyMatch(rally.id)?.name ?? 'Unknown'}
                                isActive={isPlaying && currentPlaylistRally?.id === rally.id}
                                onClick={() => handleRallyClick(rally)}
                                onRemove={() => removeRallyFromHighlight(rally.id, highlight.id)}
                              />
                            ))}
                          </List>
                        </SortableContext>
                      </DndContext>
                    </Box>
                  </Collapse>
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
