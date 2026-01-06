'use client';

import { useState, useMemo, useCallback, memo } from 'react';
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
import MoreVertIcon from '@mui/icons-material/MoreVert';
import Menu from '@mui/material/Menu';
import MenuItem from '@mui/material/MenuItem';
import {
  DndContext,
  closestCenter,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors,
  DragEndEvent,
  DragStartEvent,
  DragOverEvent,
  useDroppable,
  DragOverlay,
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
import { RallyWithSource, FADE_DURATION } from '@/utils/videoExport';
import { designTokens } from '@/app/theme';
import { NamePromptModal } from './NamePromptModal';

// Helper to format time as MM:SS
function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

// Helper to create composite ID for drag and drop
function createDragId(highlightId: string, rallyId: string): string {
  return `${highlightId}::${rallyId}`;
}

// Helper to parse composite ID
function parseDragId(dragId: string): { highlightId: string; rallyId: string } | null {
  const parts = dragId.split('::');
  if (parts.length !== 2) return null;
  return { highlightId: parts[0], rallyId: parts[1] };
}

// Droppable wrapper for entire highlight (uses useDroppable hook)
interface DroppableHighlightWrapperProps {
  highlightId: string;
  highlightColor: string;
  isOverHighlight: boolean;
  children: React.ReactNode;
}

function DroppableHighlightWrapper({
  highlightId,
  highlightColor,
  isOverHighlight,
  children
}: DroppableHighlightWrapperProps) {
  const { setNodeRef } = useDroppable({
    id: `droppable::${highlightId}`,
  });

  return (
    <Box
      ref={setNodeRef}
      sx={{
        borderRadius: 1,
        border: isOverHighlight ? `2px dashed ${highlightColor}` : '2px dashed transparent',
        bgcolor: isOverHighlight ? `${highlightColor}10` : 'transparent',
        transition: 'all 0.15s ease',
        mb: 0.5,
      }}
    >
      {children}
    </Box>
  );
}

// Expanded rally list section (no droppable - parent handles it)
interface ExpandedRallySectionProps {
  isEmpty: boolean;
  highlightColor: string;
  isOverHighlight: boolean;
  children: React.ReactNode;
}

function ExpandedRallySection({ isEmpty, highlightColor, isOverHighlight, children }: ExpandedRallySectionProps) {
  return (
    <Box
      sx={{
        px: 1,
        py: 0.5,
        bgcolor: designTokens.colors.surface[2],
        minHeight: isEmpty ? 40 : 'auto',
        borderBottomLeftRadius: 4,
        borderBottomRightRadius: 4,
      }}
    >
      {isEmpty ? (
        <Typography
          variant="caption"
          sx={{
            color: isOverHighlight ? highlightColor : 'text.disabled',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            height: 32,
          }}
        >
          {isOverHighlight ? 'Drop here' : 'No rallies'}
        </Typography>
      ) : (
        children
      )}
    </Box>
  );
}

// Helper to extract rally number from rally ID (e.g., "match_1_rally_5" -> 5)
function getRallyNumber(rallyId: string): number {
  const match = rallyId.match(/_rally_(\d+)$/);
  return match ? parseInt(match[1], 10) : 0;
}

// Sortable rally item component
interface SortableRallyItemProps {
  rally: Rally;
  highlightId: string;
  matchName: string;
  isActive: boolean;
  isPreselected: boolean;
  onClick: () => void;
  onRemove: () => void;
}

const SortableRallyItem = memo(function SortableRallyItem({ rally, highlightId, matchName, isActive, isPreselected, onClick, onRemove }: SortableRallyItemProps) {
  const [confirmingRemove, setConfirmingRemove] = useState(false);
  const dragId = createDragId(highlightId, rally.id);
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging,
  } = useSortable({ id: dragId });

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.5 : 1,
  };

  const rallyNumber = getRallyNumber(rally.id);
  const duration = (rally.end_time - rally.start_time).toFixed(1);

  const showHighlight = isActive || isPreselected;

  const handleRemoveClick = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    setConfirmingRemove(true);
  }, []);

  const handleConfirmRemove = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    onRemove();
    setConfirmingRemove(false);
  }, [onRemove]);

  const handleCancelRemove = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    setConfirmingRemove(false);
  }, []);

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
        bgcolor: showHighlight ? 'action.selected' : 'transparent',
        borderRadius: 1,
        mb: 0.25,
        transition: designTokens.transitions.fast,
        '&:hover': { bgcolor: showHighlight ? 'action.selected' : 'action.hover' },
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
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75 }}>
            <Typography component="span" sx={{ fontSize: '0.75rem', color: 'text.secondary', fontWeight: 600 }}>
              {matchName}
            </Typography>
            <Typography component="span" sx={{ fontSize: '0.8rem' }}>
              Rally {rallyNumber}
            </Typography>
            <Typography component="span" sx={{ fontSize: '0.7rem', color: 'text.disabled', fontFamily: 'monospace' }}>
              {duration}s
            </Typography>
          </Box>
        }
        sx={{ my: 0 }}
      />

      {confirmingRemove ? (
        <Stack direction="row" spacing={0.25}>
          <Tooltip title="Confirm remove">
            <IconButton
              size="small"
              onClick={handleConfirmRemove}
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
              onClick={handleCancelRemove}
              sx={{ width: 24, height: 24 }}
            >
              <CloseIcon sx={{ fontSize: 14 }} />
            </IconButton>
          </Tooltip>
        </Stack>
      ) : (
        <IconButton
          className="remove-btn"
          size="small"
          onClick={handleRemoveClick}
          sx={{
            opacity: 0,
            p: 0.25,
            transition: 'opacity 0.15s',
            '&:hover': { color: 'error.main' },
          }}
        >
          <CloseIcon sx={{ fontSize: 14 }} />
        </IconButton>
      )}
    </ListItem>
  );
});

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
    moveRallyBetweenHighlights,
    removeRallyFromHighlight,
    canEditHighlight,
    currentUserName,
    currentUserId,
    expandedHighlightIds,
    expandHighlight,
    collapseHighlight,
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
    jumpToPlaylistRally,
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
  const [activeDragId, setActiveDragId] = useState<string | null>(null);
  const [overHighlightId, setOverHighlightId] = useState<string | null>(null);
  // Track preselected rally per highlight (highlightId -> rallyId)
  const [preselectedRallies, setPreselectedRallies] = useState<Record<string, string>>({});
  const [showNamePromptModal, setShowNamePromptModal] = useState(false);
  // Highlight action menu state
  const [highlightMenuAnchor, setHighlightMenuAnchor] = useState<{
    el: HTMLElement;
    highlight: { id: string; name: string; color: string; rallyIds: string[] };
  } | null>(null);

  // DnD sensors
  const sensors = useSensors(
    useSensor(PointerSensor, {
      activationConstraint: { distance: 5 },
    }),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    })
  );

  // Use closestCenter for reliable collision detection in vertical list
  const collisionDetection = closestCenter;

  const handleDragStart = (event: DragStartEvent) => {
    setActiveDragId(event.active.id as string);
    setOverHighlightId(null);
  };

  const handleDragOver = (event: DragOverEvent) => {
    const { active, over } = event;
    if (!over) {
      setOverHighlightId(null);
      return;
    }

    // Get the source highlight ID from the dragged item
    const activeData = parseDragId(active.id as string);
    const sourceHighlightId = activeData?.highlightId;

    const overId = over.id as string;
    let targetHighlightId: string | null = null;

    // Check if over a droppable highlight container
    if (overId.startsWith('droppable::')) {
      targetHighlightId = overId.replace('droppable::', '');
    } else {
      // Check if over a sortable rally item - extract the highlight ID
      const parsed = parseDragId(overId);
      if (parsed) {
        targetHighlightId = parsed.highlightId;
      }
    }

    // Only set overHighlightId if we're over a DIFFERENT highlight
    if (targetHighlightId && targetHighlightId !== sourceHighlightId) {
      setOverHighlightId(targetHighlightId);
    } else {
      setOverHighlightId(null);
    }
  };

  const handleDragEnd = (event: DragEndEvent) => {
    const { active, over } = event;
    setActiveDragId(null);
    setOverHighlightId(null);

    if (!over) return;

    const activeData = parseDragId(active.id as string);
    if (!activeData) return;

    const { highlightId: sourceHighlightId, rallyId } = activeData;

    // Check if dropped on a droppable zone (highlight container)
    const overId = over.id as string;

    // Handle drop on highlight container
    if (overId.startsWith('droppable::')) {
      const targetHighlightId = overId.replace('droppable::', '');
      if (targetHighlightId !== sourceHighlightId) {
        // Move to the end of the target highlight
        const targetHighlight = highlights?.find((h) => h.id === targetHighlightId);
        const targetIndex = targetHighlight?.rallyIds.length ?? 0;
        moveRallyBetweenHighlights(rallyId, sourceHighlightId, targetHighlightId, targetIndex);
        // Auto-expand the target highlight
        expandHighlight(targetHighlightId);
      }
      return;
    }

    // Check if dropped on another rally item
    const overData = parseDragId(overId);
    if (!overData) return;

    const { highlightId: targetHighlightId, rallyId: overRallyId } = overData;

    if (sourceHighlightId === targetHighlightId) {
      // Reordering within the same highlight
      const highlight = highlights?.find((h) => h.id === sourceHighlightId);
      if (!highlight) return;

      const oldIndex = highlight.rallyIds.indexOf(rallyId);
      const newIndex = highlight.rallyIds.indexOf(overRallyId);

      if (oldIndex !== -1 && newIndex !== -1 && oldIndex !== newIndex) {
        reorderHighlightRallies(sourceHighlightId, oldIndex, newIndex);
      }
    } else {
      // Moving between highlights
      const targetHighlight = highlights?.find((h) => h.id === targetHighlightId);
      if (!targetHighlight) return;

      const targetIndex = targetHighlight.rallyIds.indexOf(overRallyId);
      if (targetIndex !== -1) {
        moveRallyBetweenHighlights(rallyId, sourceHighlightId, targetHighlightId, targetIndex);
        // Auto-expand the target highlight
        expandHighlight(targetHighlightId);
      }
    }
  };

  // Get active drag item info for overlay - memoized
  const activeDragData = useMemo(
    () => (activeDragId ? parseDragId(activeDragId) : null),
    [activeDragId]
  );
  const activeDragRally = useMemo(
    () => (activeDragData ? allRallies.find((r) => r.id === activeDragData.rallyId) : null),
    [activeDragData, allRallies]
  );

  const handleRallyClick = useCallback((rally: Rally, highlightId: string) => {
    // If this highlight is currently playing, jump to the rally within the playlist
    if (playingHighlightId === highlightId) {
      const targetRally = jumpToPlaylistRally(rally.id);
      if (targetRally) {
        // Handle match switch if needed
        const match = getRallyMatch(rally.id);
        if (match && match.id !== activeMatchId) {
          setActiveMatch(match.id);
        }
        return;
      }
    }

    // Preselect this rally for when play is pressed
    setPreselectedRallies((prev) => ({ ...prev, [highlightId]: rally.id }));

    // Normal behavior: switch match if needed and seek
    const match = getRallyMatch(rally.id);
    if (match && match.id !== activeMatchId) {
      setActiveMatch(match.id);
    }
    seek(rally.start_time);
  }, [playingHighlightId, jumpToPlaylistRally, getRallyMatch, activeMatchId, setActiveMatch, seek]);

  const handlePlay = useCallback((highlightId: string, e: React.MouseEvent) => {
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

      // Find start index from preselected rally
      const preselectedId = preselectedRallies[highlightId];
      let startIndex = 0;
      if (preselectedId) {
        const idx = playlist.findIndex((r) => r.id === preselectedId);
        if (idx !== -1) startIndex = idx;
      }

      const startRally = playlist[startIndex];
      // Switch to the match containing the start rally if needed
      if (startRally.matchId) {
        setActiveMatch(startRally.matchId);
      }
      seek(startRally.start_time);
      startHighlightPlayback(highlightId, playlist, startIndex);
    }
  }, [highlights, allRallies, getRallyMatch, setActiveMatch, seek, startHighlightPlayback, preselectedRallies]);

  const handleStop = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    stopHighlightPlayback();
  }, [stopHighlightPlayback]);

  const handleStartEdit = useCallback((highlight: { id: string; name: string }, e: React.MouseEvent) => {
    e.stopPropagation();
    setEditingId(highlight.id);
    setEditName(highlight.name);
  }, []);

  const handleSaveEdit = useCallback(() => {
    if (editingId && editName.trim()) {
      renameHighlight(editingId, editName.trim());
    }
    setEditingId(null);
  }, [editingId, editName, renameHighlight]);

  const handleDeleteClick = useCallback((id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setDeleteConfirmId(id);
  }, []);

  const handleConfirmDelete = useCallback((id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    deleteHighlight(id);
    setDeleteConfirmId(null);
  }, [deleteHighlight]);

  const handleCancelDelete = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    setDeleteConfirmId(null);
  }, []);

  const handleDownloadClick = useCallback((highlightId: string, e: React.MouseEvent<HTMLButtonElement>) => {
    e.stopPropagation();
    setDownloadAnchor({ el: e.currentTarget, id: highlightId });
  }, []);

  const handleDownload = useCallback(() => {
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
  }, [downloadAnchor, highlights, allRallies, getRallyMatch, videoSource, withFade, downloadHighlight]);

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
      {/* Highlights list with shared DndContext for cross-highlight dragging */}
      <DndContext
        sensors={sensors}
        collisionDetection={collisionDetection}
        onDragStart={handleDragStart}
        onDragOver={handleDragOver}
        onDragEnd={handleDragEnd}
      >
        <Box sx={{ flex: 1, overflow: 'auto', py: 0.5 }}>
          {highlights.map((highlight) => {
            const isSelected = selectedHighlightId === highlight.id;
            const isPlaying = playingHighlightId === highlight.id;
            const rallyCount = highlight.rallyIds?.length ?? 0;
            const isExpanded = expandedHighlightIds.has(highlight.id);
            const canEdit = canEditHighlight(highlight.id);
            const creatorName = highlight.createdByUserName;
            const isOwnHighlight = highlight.createdByUserId === currentUserId;

            // Get rallies in custom order (rallyIds order)
            const orderedRallies = highlight.rallyIds
              .map((id) => allRallies.find((r) => r.id === id))
              .filter((r): r is Rally => r !== undefined);

            // Get current playing rally for this highlight
            const currentPlaylistRally = getCurrentPlaylistRally();

            // overHighlightId is only set when dragging over a DIFFERENT highlight
            const isOverThis = overHighlightId === highlight.id;

            return (
              <DroppableHighlightWrapper
                key={highlight.id}
                highlightId={highlight.id}
                highlightColor={highlight.color}
                isOverHighlight={isOverThis}
              >
                {/* Highlight row */}
                <Box
                  onClick={() => {
                    if (isSelected) {
                      // Deselect and collapse
                      selectHighlight(null);
                      collapseHighlight(highlight.id);
                    } else {
                      // Select and expand
                      selectHighlight(highlight.id);
                      expandHighlight(highlight.id);
                    }
                  }}
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
                  {/* Expand/collapse icon - visual indicator only, clicking row handles selection+expansion */}
                  <Box sx={{ p: 0.25, mr: 0.5, display: 'flex', alignItems: 'center' }}>
                    {isExpanded ? (
                      <ExpandMoreIcon sx={{ fontSize: 18, color: 'text.secondary' }} />
                    ) : (
                      <ChevronRightIcon sx={{ fontSize: 18, color: 'text.secondary' }} />
                    )}
                  </Box>

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
                      onDoubleClick={(e) => canEdit && handleStartEdit(highlight, e)}
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

                  {/* Creator badge - only show for other users' highlights */}
                  {creatorName && !isOwnHighlight && (
                    <Tooltip title={`Created by ${creatorName}`}>
                      <Chip
                        label={creatorName}
                        size="small"
                        sx={{
                          height: 18,
                          fontSize: '0.625rem',
                          ml: 0.5,
                          bgcolor: 'action.hover',
                          color: 'text.secondary',
                        }}
                      />
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

                  {/* More menu button */}
                  <IconButton
                    size="small"
                    onClick={(e) => {
                      e.stopPropagation();
                      setHighlightMenuAnchor({ el: e.currentTarget, highlight });
                    }}
                    sx={{
                      opacity: isSelected || exportingHighlightId === highlight.id ? 1 : 0.5,
                      '&:hover': { opacity: 1 },
                    }}
                  >
                    {exportingHighlightId === highlight.id ? (
                      <CircularProgress size={16} />
                    ) : (
                      <MoreVertIcon sx={{ fontSize: 18 }} />
                    )}
                  </IconButton>
                </Box>

                {/* Collapsible rally list */}
                <Collapse in={isExpanded}>
                  <ExpandedRallySection
                    isEmpty={rallyCount === 0}
                    highlightColor={highlight.color}
                    isOverHighlight={isOverThis}
                  >
                    <SortableContext
                      items={orderedRallies.map((r) => createDragId(highlight.id, r.id))}
                      strategy={verticalListSortingStrategy}
                    >
                      <List dense disablePadding>
                        {orderedRallies.map((rally) => (
                          <SortableRallyItem
                            key={rally.id}
                            rally={rally}
                            highlightId={highlight.id}
                            matchName={getRallyMatch(rally.id)?.name ?? 'Unknown'}
                            isActive={isPlaying && currentPlaylistRally?.id === rally.id}
                            isPreselected={!isPlaying && preselectedRallies[highlight.id] === rally.id}
                            onClick={() => handleRallyClick(rally, highlight.id)}
                            onRemove={() => removeRallyFromHighlight(rally.id, highlight.id)}
                          />
                        ))}
                      </List>
                    </SortableContext>
                  </ExpandedRallySection>
                </Collapse>
              </DroppableHighlightWrapper>
            );
          })}
        </Box>

        {/* Drag overlay for visual feedback */}
        <DragOverlay>
          {activeDragRally && activeDragData ? (() => {
            const matchName = getRallyMatch(activeDragRally.id)?.name ?? 'Unknown';
            const rallyNumber = getRallyNumber(activeDragRally.id);
            const duration = (activeDragRally.end_time - activeDragRally.start_time).toFixed(1);
            return (
              <Box
                sx={{
                  py: 0.5,
                  px: 1,
                  minHeight: 32,
                  bgcolor: 'background.paper',
                  borderRadius: 1,
                  boxShadow: 3,
                  display: 'flex',
                  alignItems: 'center',
                  gap: 0.75,
                }}
              >
                <DragIndicatorIcon sx={{ fontSize: 16, color: 'text.disabled' }} />
                <Typography component="span" sx={{ fontSize: '0.75rem', color: 'text.secondary', fontWeight: 600 }}>
                  {matchName}
                </Typography>
                <Typography component="span" sx={{ fontSize: '0.8rem' }}>
                  Rally {rallyNumber}
                </Typography>
                <Typography component="span" sx={{ fontSize: '0.7rem', color: 'text.disabled', fontFamily: 'monospace' }}>
                  {duration}s
                </Typography>
              </Box>
            );
          })() : null}
        </DragOverlay>
      </DndContext>

      {/* Highlight action menu */}
      <Menu
        anchorEl={highlightMenuAnchor?.el}
        open={Boolean(highlightMenuAnchor)}
        onClose={() => setHighlightMenuAnchor(null)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
        transformOrigin={{ vertical: 'top', horizontal: 'right' }}
      >
        {/* Rename - only if user can edit */}
        {highlightMenuAnchor && canEditHighlight(highlightMenuAnchor.highlight.id) && (
          <MenuItem
            onClick={(e) => {
              handleStartEdit(highlightMenuAnchor.highlight, e);
              setHighlightMenuAnchor(null);
            }}
          >
            <ListItemIcon>
              <EditIcon fontSize="small" />
            </ListItemIcon>
            <ListItemText>Rename</ListItemText>
          </MenuItem>
        )}

        {/* Download */}
        <MenuItem
          onClick={(e) => {
            if (highlightMenuAnchor) {
              setDownloadAnchor({ el: highlightMenuAnchor.el as HTMLButtonElement, id: highlightMenuAnchor.highlight.id });
            }
            setHighlightMenuAnchor(null);
          }}
          disabled={
            (highlightMenuAnchor?.highlight.rallyIds?.length ?? 0) === 0 ||
            isExporting ||
            !videoSource
          }
        >
          <ListItemIcon>
            <FileDownloadIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText>Download</ListItemText>
        </MenuItem>

        {/* Delete - only if user can edit */}
        {highlightMenuAnchor && canEditHighlight(highlightMenuAnchor.highlight.id) && (
          <MenuItem
            onClick={(e) => {
              handleDeleteClick(highlightMenuAnchor.highlight.id, e);
              setHighlightMenuAnchor(null);
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
              <Typography variant="body2">Add fade ({FADE_DURATION}s)</Typography>
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

      {/* Name Prompt Modal */}
      <NamePromptModal
        open={showNamePromptModal}
        onClose={() => setShowNamePromptModal(false)}
        onNameSet={() => {
          // Name was set, modal will close automatically
        }}
      />
    </Box>
  );
}
