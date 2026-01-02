'use client';

import { useState, useMemo, useCallback } from 'react';
import {
  Box,
  Typography,
  IconButton,
  Button,
  Chip,
  Stack,
  SwipeableDrawer,
  Divider,
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import PauseIcon from '@mui/icons-material/Pause';
import DeleteIcon from '@mui/icons-material/Delete';
import CheckIcon from '@mui/icons-material/Check';
import { MiniTimeline } from './MiniTimeline';
import { useEditorStore } from '@/stores/editorStore';
import { usePlayerStore } from '@/stores/playerStore';
import { formatTime, formatDuration } from '@/utils/timeFormat';
import { designTokens } from '@/app/theme';

interface MobileRallyEditorModalProps {
  open: boolean;
  onClose: () => void;
  rallyId: string | null;
}

export function MobileRallyEditorModal({
  open,
  onClose,
  rallyId,
}: MobileRallyEditorModalProps) {
  const {
    session,
    rallies,
    highlights,
    updateRally,
    removeRally,
    addRallyToHighlight,
    removeRallyFromHighlight,
    getHighlightsForRally,
    getActiveMatch,
  } = useEditorStore();
  const { currentTime, isPlaying, seek, play, pause } = usePlayerStore();

  const [deleteConfirm, setDeleteConfirm] = useState(false);

  // Find the rally and its match
  const rally = useMemo(() => {
    if (!rallyId || !rallies) return null;
    return rallies.find((r) => r.id === rallyId) || null;
  }, [rallyId, rallies]);

  const match = useMemo(() => {
    if (!rallyId || !session?.matches) return null;
    // Extract match ID from rally ID (format: {matchId}_rally_{n})
    const matchId = rallyId.split('_rally_')[0];
    return session.matches.find((m) => m.id === matchId) || null;
  }, [rallyId, session?.matches]);

  const activeMatch = getActiveMatch();
  const videoDuration = activeMatch?.video?.duration || 0;

  // Get rally index within match
  const rallyIndex = useMemo(() => {
    if (!match || !rally) return -1;
    return match.rallies.findIndex((r) => r.id === rally.id);
  }, [match, rally]);

  // Get adjacent rallies for boundary constraints
  const adjacentRallies = useMemo(() => {
    if (!match || !rally || rallyIndex < 0) return undefined;
    const prev = rallyIndex > 0 ? match.rallies[rallyIndex - 1] : undefined;
    const next =
      rallyIndex < match.rallies.length - 1
        ? match.rallies[rallyIndex + 1]
        : undefined;
    return { prev, next };
  }, [match, rally, rallyIndex]);

  // Get highlights this rally belongs to
  const rallyHighlights = useMemo(() => {
    if (!rallyId) return [];
    return getHighlightsForRally(rallyId);
  }, [rallyId, getHighlightsForRally]);

  // Handle start/end time changes
  const handleStartChange = useCallback(
    (newStart: number) => {
      if (!rally) return;
      updateRally(rally.id, { start_time: newStart });
    },
    [rally, updateRally]
  );

  const handleEndChange = useCallback(
    (newEnd: number) => {
      if (!rally) return;
      updateRally(rally.id, { end_time: newEnd });
    },
    [rally, updateRally]
  );

  // Handle seek
  const handleSeek = useCallback(
    (time: number) => {
      seek(time);
    },
    [seek]
  );

  // Toggle play/pause for this rally segment
  const handlePlayToggle = useCallback(() => {
    if (!rally) return;
    if (isPlaying) {
      pause();
    } else {
      seek(rally.start_time);
      play();
    }
  }, [rally, isPlaying, seek, play, pause]);

  // Handle highlight toggle
  const handleHighlightToggle = useCallback(
    (highlightId: string) => {
      if (!rallyId) return;
      const isInHighlight = rallyHighlights.some((h) => h.id === highlightId);
      if (isInHighlight) {
        removeRallyFromHighlight(rallyId, highlightId);
      } else {
        addRallyToHighlight(rallyId, highlightId);
      }
    },
    [rallyId, rallyHighlights, addRallyToHighlight, removeRallyFromHighlight]
  );

  // Handle delete
  const handleDelete = useCallback(() => {
    if (!rally) return;
    if (deleteConfirm) {
      removeRally(rally.id);
      setDeleteConfirm(false);
      onClose();
    } else {
      setDeleteConfirm(true);
    }
  }, [rally, deleteConfirm, removeRally, onClose]);

  const handleCancelDelete = useCallback(() => {
    setDeleteConfirm(false);
  }, []);

  // Reset delete confirm when modal closes
  const handleClose = useCallback(() => {
    setDeleteConfirm(false);
    onClose();
  }, [onClose]);

  if (!rally || !match) {
    return null;
  }

  return (
    <SwipeableDrawer
      anchor="bottom"
      open={open}
      onClose={handleClose}
      onOpen={() => {}}
      swipeAreaWidth={20}
      disableSwipeToOpen
      ModalProps={{ keepMounted: false }}
      PaperProps={{
        sx: {
          borderTopLeftRadius: 16,
          borderTopRightRadius: 16,
          maxHeight: '85vh',
          overflow: 'hidden',
        },
      }}
    >
      {/* Drawer handle */}
      <Box
        sx={{
          width: 40,
          height: 4,
          bgcolor: 'rgba(255,255,255,0.3)',
          borderRadius: 2,
          mx: 'auto',
          mt: 1,
          mb: 0.5,
        }}
      />

      {/* Header */}
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          px: 2,
          py: 1,
          borderBottom: '1px solid',
          borderColor: 'divider',
        }}
      >
        <Box sx={{ flex: 1 }}>
          <Typography variant="subtitle1" fontWeight={600}>
            Rally {rallyIndex + 1}
          </Typography>
          <Typography variant="caption" color="text.secondary">
            {match.name} \u2022 {formatDuration(rally.duration)}
          </Typography>
        </Box>
        <IconButton onClick={handleClose} size="small">
          <CloseIcon />
        </IconButton>
      </Box>

      {/* Content */}
      <Box sx={{ p: 2, overflow: 'auto' }}>
        {/* Mini Timeline */}
        <MiniTimeline
          rally={rally}
          videoDuration={videoDuration}
          currentTime={currentTime}
          onStartChange={handleStartChange}
          onEndChange={handleEndChange}
          onSeek={handleSeek}
          adjacentRallies={adjacentRallies}
        />

        {/* Time display */}
        <Box
          sx={{
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            gap: 2,
            mt: 2,
            mb: 1,
          }}
        >
          <Typography variant="body2" color="text.secondary">
            {formatTime(rally.start_time)}
          </Typography>
          <Typography variant="body2" fontWeight={600}>
            \u2192
          </Typography>
          <Typography variant="body2" color="text.secondary">
            {formatTime(rally.end_time)}
          </Typography>
        </Box>

        {/* Play button */}
        <Box sx={{ display: 'flex', justifyContent: 'center', mb: 2 }}>
          <Button
            variant="contained"
            startIcon={isPlaying ? <PauseIcon /> : <PlayArrowIcon />}
            onClick={handlePlayToggle}
            sx={{ minWidth: 140 }}
          >
            {isPlaying ? 'Pause' : 'Play Segment'}
          </Button>
        </Box>

        <Divider sx={{ my: 2 }} />

        {/* Highlights */}
        {highlights && highlights.length > 0 && (
          <Box>
            <Typography
              variant="subtitle2"
              color="text.secondary"
              sx={{ mb: 1 }}
            >
              Add to Highlight
            </Typography>
            <Stack direction="row" flexWrap="wrap" gap={1}>
              {highlights.map((highlight) => {
                const isInHighlight = rallyHighlights.some(
                  (h) => h.id === highlight.id
                );
                return (
                  <Chip
                    key={highlight.id}
                    label={highlight.name}
                    onClick={() => handleHighlightToggle(highlight.id)}
                    sx={{
                      minHeight: designTokens.mobile.touchTarget,
                      px: 1,
                      bgcolor: isInHighlight ? highlight.color : 'transparent',
                      color: isInHighlight ? 'rgba(0,0,0,0.87)' : highlight.color,
                      border: `2px solid ${highlight.color}`,
                      fontWeight: isInHighlight ? 600 : 400,
                      '&:active': {
                        transform: 'scale(0.95)',
                      },
                      transition: 'transform 0.1s ease',
                    }}
                  />
                );
              })}
            </Stack>
          </Box>
        )}

        {highlights && highlights.length === 0 && (
          <Typography variant="body2" color="text.secondary" textAlign="center">
            No highlights yet. Create one to add this rally.
          </Typography>
        )}

        <Divider sx={{ my: 2 }} />

        {/* Delete */}
        <Box sx={{ display: 'flex', justifyContent: 'center' }}>
          {deleteConfirm ? (
            <Stack direction="row" spacing={1} alignItems="center">
              <Typography variant="body2" color="error.main">
                Delete this rally?
              </Typography>
              <IconButton
                size="small"
                onClick={handleCancelDelete}
                sx={{ color: 'text.secondary' }}
              >
                <CloseIcon fontSize="small" />
              </IconButton>
              <IconButton
                size="small"
                onClick={handleDelete}
                sx={{ color: 'error.main' }}
              >
                <CheckIcon fontSize="small" />
              </IconButton>
            </Stack>
          ) : (
            <Button
              variant="text"
              color="error"
              startIcon={<DeleteIcon />}
              onClick={handleDelete}
              sx={{ minHeight: designTokens.mobile.touchTarget }}
            >
              Delete Rally
            </Button>
          )}
        </Box>
      </Box>

      {/* Safe area padding */}
      <Box
        sx={{
          height: designTokens.mobile.bottomNav.safeAreaPadding,
          minHeight: 16,
        }}
      />
    </SwipeableDrawer>
  );
}
