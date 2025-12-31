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
import { RallyWithSource } from '@/utils/videoExport';
import { designTokens } from '@/app/theme';

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
        py: 0.5,
        px: 1,
        minHeight: 32,
        cursor: 'pointer',
        bgcolor: isActive ? 'action.selected' : 'transparent',
        borderRadius: 1,
        mb: 0.25,
        transition: designTokens.transitions.fast,
        '&:hover': { bgcolor: isActive ? 'action.selected' : 'action.hover' },
        '&:hover .remove-btn': { opacity: 1 },
      }}
    >
      <ListItemIcon
        {...attributes}
        {...listeners}
        sx={{ minWidth: 24, cursor: 'grab' }}
      >
        <DragIndicatorIcon sx={{ fontSize: 16, color: 'text.disabled' }} />
      </ListItemIcon>

      <ListItemText
        primary={
          <Typography component="span" sx={{ fontSize: '0.75rem', fontFamily: 'monospace' }}>
            {String(index + 1).padStart(2, '0')} {formatTime(rally.start_time)}→{formatTime(rally.end_time)}
          </Typography>
        }
        secondary={
          <Typography component="span" sx={{ fontSize: '0.625rem', color: 'text.disabled' }}>
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
          '&:hover': { color: 'error.main' },
        }}
      >
        <CloseIcon sx={{ fontSize: 14 }} />
      </IconButton>
    </ListItem>
  );
}

export function HighlightsPanel() {
  const {
    highlights,
    selectedHighlightId,
    selectHighlight,
    deleteHighlight,
    renameHighlight,
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
    if (!downloadAnchor) return;

    const highlight = highlights?.find((h) => h.id === downloadAnchor.id);
    if (!highlight) return;

    // Build rallies with their video sources from their respective matches
    const ralliesWithSource: RallyWithSource[] = highlight.rallyIds
      .map((id) => {
        const rally = allRallies.find((r) => r.id === id);
        if (!rally) return null;
        const match = getRallyMatch(rally.id);
        // Use match's video URL, or fall back to active video source
        const source = match?.videoUrl || videoSource;
        if (!source) return null;
        return { rally, videoSource: source };
      })
      .filter((r): r is RallyWithSource => r !== null);

    if (ralliesWithSource.length === 0) return;

    downloadHighlight(ralliesWithSource, highlight.id, highlight.name, withFade);
    setDownloadAnchor(null);
  };

  // Empty state
  if (!highlights || highlights.length === 0) {
    return (
      <Box
        sx={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          p: 3,
          textAlign: 'center',
        }}
      >
        <Typography variant="body2" sx={{ color: 'text.disabled', mb: 1 }}>
          No highlights yet
        </Typography>
        <Typography variant="caption" sx={{ color: 'text.disabled' }}>
          Select a rally and press Enter to add to a highlight
        </Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* Highlights list */}
      <Box sx={{ flex: 1, overflow: 'auto', py: 0.5 }}>
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
                  px: 1.5,
                  py: 1,
                  cursor: 'pointer',
                  borderLeft: '3px solid',
                  borderColor: isSelected ? highlight.color : 'transparent',
                  bgcolor: isSelected
                    ? 'action.selected'
                    : isPlaying
                    ? 'rgba(0, 212, 170, 0.08)'
                    : 'transparent',
                  transition: designTokens.transitions.fast,
                  '&:hover': {
                    bgcolor: isSelected
                      ? 'action.selected'
                      : 'action.hover',
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
                    <ExpandMoreIcon sx={{ fontSize: 18, color: 'text.secondary' }} />
                  ) : (
                    <ChevronRightIcon sx={{ fontSize: 18, color: rallyCount === 0 ? 'text.disabled' : 'text.secondary' }} />
                  )}
                </IconButton>

                {/* Color dot */}
                <Box
                  sx={{
                    width: 14,
                    height: 14,
                    borderRadius: '50%',
                    bgcolor: highlight.color,
                    mr: 1.5,
                    flexShrink: 0,
                    boxShadow: isPlaying
                      ? `0 0 12px ${highlight.color}`
                      : 'none',
                    transition: 'box-shadow 0.3s ease',
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
                        fontSize: '0.875rem',
                        py: 0.25,
                      },
                    }}
                  />
                ) : (
                  <Typography
                    onDoubleClick={(e) => handleStartEdit(highlight, e)}
                    sx={{
                      flex: 1,
                      fontSize: '0.875rem',
                      fontWeight: isSelected ? 600 : 400,
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
                      sx={{ opacity: isSelected ? 1 : 0.5, '&:hover': { opacity: 1 } }}
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
                    height: 20,
                    fontSize: '0.6875rem',
                    fontWeight: 600,
                    mx: 0.5,
                    bgcolor: `${highlight.color}20`,
                    color: highlight.color,
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
                        sx={{
                          opacity: isSelected || exportingHighlightId === highlight.id ? 1 : 0.5,
                          '&:hover': { opacity: 1 },
                        }}
                      >
                        {exportingHighlightId === highlight.id ? (
                          <CircularProgress size={16} />
                        ) : (
                          <FileDownloadIcon sx={{ fontSize: 18 }} />
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
                    <StopIcon sx={{ fontSize: 18 }} />
                  </IconButton>
                ) : (
                  <Tooltip title={rallyCount === 0 ? 'No rallies' : 'Play highlight'}>
                    <span>
                      <IconButton
                        size="small"
                        onClick={(e) => handlePlay(highlight.id, e)}
                        disabled={rallyCount === 0}
                        color="secondary"
                      >
                        <PlayArrowIcon sx={{ fontSize: 18 }} />
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
                          bgcolor: 'error.main',
                          width: 24,
                          height: 24,
                          '&:hover': { bgcolor: 'error.light' },
                        }}
                      >
                        <CheckIcon sx={{ fontSize: 14 }} />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Cancel">
                      <IconButton
                        size="small"
                        onClick={handleCancelDelete}
                        sx={{ width: 24, height: 24 }}
                      >
                        <CloseIcon sx={{ fontSize: 14 }} />
                      </IconButton>
                    </Tooltip>
                  </Stack>
                ) : (
                  <IconButton
                    size="small"
                    onClick={(e) => handleDeleteClick(highlight.id, e)}
                    sx={{
                      opacity: isSelected ? 1 : 0.5,
                      '&:hover': { opacity: 1, color: 'error.main' },
                    }}
                  >
                    <DeleteIcon sx={{ fontSize: 18 }} />
                  </IconButton>
                )}
              </Box>

              {/* Collapsible rally list */}
              <Collapse in={isExpanded && rallyCount > 0}>
                <Box sx={{ px: 1, py: 0.5, bgcolor: designTokens.colors.surface[2] }}>
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

      {/* Download Popover */}
      <Popover
        open={Boolean(downloadAnchor)}
        anchorEl={downloadAnchor?.el}
        onClose={() => setDownloadAnchor(null)}
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
            onClick={handleDownload}
            disabled={isExporting}
            sx={{ mt: 2 }}
          >
            Download
          </Button>
        </Box>
      </Popover>
    </Box>
  );
}
