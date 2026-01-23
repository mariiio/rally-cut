'use client';

import { useState, useMemo, useCallback, useEffect, useRef } from 'react';
import {
  Box,
  Typography,
  List,
  ListItemButton,
  Stack,
  Chip,
  Collapse,
  IconButton,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ChevronRightIcon from '@mui/icons-material/ChevronRight';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import { useEditorStore } from '@/stores/editorStore';
import { usePlayerStore } from '@/stores/playerStore';
import { formatTime, formatDuration } from '@/utils/timeFormat';
import { Rally } from '@/types/rally';
import { designTokens } from '@/app/theme';

interface MobileRallyListProps {
  onRallyTap: (rallyId: string) => void;
}

export function MobileRallyList({ onRallyTap }: MobileRallyListProps) {
  const {
    session,
    activeMatchId,
    setActiveMatch,
    rallies,
    selectedRallyId,
    selectRally,
    getHighlightsForRally,
  } = useEditorStore();
  const { currentTime, seek, isPlaying } = usePlayerStore();

  // Track which matches are expanded
  const [expandedMatches, setExpandedMatches] = useState<Set<string>>(
    new Set(activeMatchId ? [activeMatchId] : [])
  );

  // Refs for scrolling to rallies
  const rallyRefs = useRef<Map<string, HTMLElement>>(new Map());
  const prevSelectedRef = useRef<string | null>(null);

  // Auto-expand and scroll to selected rally when it changes
  useEffect(() => {
    if (!selectedRallyId || selectedRallyId === prevSelectedRef.current) {
      prevSelectedRef.current = selectedRallyId;
      return;
    }
    prevSelectedRef.current = selectedRallyId;

    // Find which match contains this rally
    if (!session?.matches) return;

    for (const match of session.matches) {
      const matchRallies = match.id === activeMatchId ? (rallies || []) : match.rallies;
      const hasRally = matchRallies.some(r => r.id === selectedRallyId);

      if (hasRally) {
        // Expand the match
        // eslint-disable-next-line react-hooks/set-state-in-effect -- intentional: auto-expand to show selected rally
        setExpandedMatches(prev => {
          if (prev.has(match.id)) return prev;
          return new Set(prev).add(match.id);
        });

        // Scroll to the rally after a short delay (for collapse animation)
        setTimeout(() => {
          const element = rallyRefs.current.get(selectedRallyId);
          if (element) {
            element.scrollIntoView({ behavior: 'smooth', block: 'center' });
          }
        }, 150);
        break;
      }
    }
  }, [selectedRallyId, session?.matches, activeMatchId, rallies]);

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

  const handleRallyTap = useCallback(
    (matchId: string, rally: Rally) => {
      // Switch match if needed
      if (matchId !== activeMatchId) {
        setActiveMatch(matchId);
        setExpandedMatches((prev) => new Set(prev).add(matchId));
      }
      selectRally(rally.id);
      seek(rally.start_time);
      // Open the editor modal
      onRallyTap(rally.id);
    },
    [activeMatchId, setActiveMatch, selectRally, seek, onRallyTap]
  );

  // Check if a rally is currently playing
  const isRallyPlaying = useCallback(
    (matchId: string, rally: Rally) => {
      if (matchId !== activeMatchId) return false;
      return (
        isPlaying &&
        currentTime >= rally.start_time &&
        currentTime <= rally.end_time
      );
    },
    [activeMatchId, currentTime, isPlaying]
  );

  // Calculate total duration across all matches
  const sessionMatches = session?.matches;
  const totalDuration = useMemo(() => {
    if (!sessionMatches) return 0;
    return sessionMatches.reduce((sum, match) => {
      // Use current rallies from store for active match
      const matchRallies = match.id === activeMatchId ? (rallies || []) : match.rallies;
      return (
        sum + matchRallies.reduce((rallySum, rally) => rallySum + rally.duration, 0)
      );
    }, 0);
  }, [sessionMatches, activeMatchId, rallies]);

  if (!session?.matches || session.matches.length === 0) {
    return (
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          height: '100%',
          p: 3,
        }}
      >
        <Typography color="text.secondary" textAlign="center">
          No videos loaded yet.
          <br />
          Upload a video to get started.
        </Typography>
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
        }}
      >
        <Typography variant="caption" color="text.secondary">
          {rallies?.length || 0} rallies
          {totalDuration > 0 && ` \u2022 ${formatDuration(totalDuration)}`}
        </Typography>
      </Box>

      {/* Match List */}
      <List disablePadding>
        {session.matches.map((match) => {
          const isExpanded = expandedMatches.has(match.id);
          const isActive = match.id === activeMatchId;
          // Use current rallies from store for active match, otherwise use session data
          const matchRallies = isActive ? (rallies || []) : (match.rallies || []);
          const matchDuration = matchRallies.reduce(
            (sum, r) => sum + r.duration,
            0
          );

          return (
            <Box key={match.id}>
              {/* Match Header */}
              <ListItemButton
                onClick={() => toggleMatchExpanded(match.id)}
                sx={{
                  minHeight: designTokens.mobile.rallyItem.minHeight,
                  px: 2,
                  borderBottom: '1px solid',
                  borderColor: 'divider',
                  bgcolor: isActive ? 'action.selected' : 'transparent',
                }}
              >
                <IconButton
                  size="small"
                  sx={{ mr: 1, p: 0.5 }}
                  onClick={(e) => {
                    e.stopPropagation();
                    toggleMatchExpanded(match.id);
                  }}
                >
                  {isExpanded ? (
                    <ExpandMoreIcon fontSize="small" />
                  ) : (
                    <ChevronRightIcon fontSize="small" />
                  )}
                </IconButton>

                <Box sx={{ flex: 1, minWidth: 0 }}>
                  <Typography
                    variant="body2"
                    fontWeight={600}
                    sx={{
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                    }}
                  >
                    {match.name}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    {matchRallies.length} rallies \u2022 {formatDuration(matchDuration)}
                  </Typography>
                </Box>
              </ListItemButton>

              {/* Rally Items */}
              <Collapse in={isExpanded}>
                {matchRallies.map((rally, index) => {
                  const rallyHighlights = getHighlightsForRally(rally.id);
                  const isSelected = selectedRallyId === rally.id;
                  const isCurrentlyPlaying = isRallyPlaying(match.id, rally);

                  return (
                    <ListItemButton
                      key={rally.id}
                      ref={(el) => {
                        if (el) {
                          rallyRefs.current.set(rally.id, el);
                        } else {
                          rallyRefs.current.delete(rally.id);
                        }
                      }}
                      onClick={() => handleRallyTap(match.id, rally)}
                      sx={{
                        minHeight: designTokens.mobile.rallyItem.minHeight,
                        pl: 5,
                        pr: 2,
                        bgcolor: isSelected
                          ? 'rgba(255, 107, 74, 0.12)'
                          : 'transparent',
                        borderBottom: '1px solid',
                        borderColor: 'divider',
                        '&:active': {
                          bgcolor: 'action.selected',
                          transform: 'scale(0.98)',
                        },
                        transition: 'transform 0.1s ease',
                      }}
                    >
                      {/* Playing indicator */}
                      <Box
                        sx={{
                          width: 24,
                          mr: 1,
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                        }}
                      >
                        {isCurrentlyPlaying && (
                          <PlayArrowIcon
                            fontSize="small"
                            color="primary"
                          />
                        )}
                      </Box>

                      {/* Rally number */}
                      <Typography
                        variant="body2"
                        sx={{
                          width: 32,
                          fontWeight: 600,
                          color: isSelected ? 'primary.main' : 'text.primary',
                        }}
                      >
                        {String(index + 1).padStart(2, '0')}
                      </Typography>

                      {/* Time range and duration */}
                      <Box sx={{ flex: 1, minWidth: 0 }}>
                        <Typography
                          variant="body2"
                          color={isSelected ? 'primary.main' : 'text.primary'}
                        >
                          {formatTime(rally.start_time)} \u2192 {formatTime(rally.end_time)}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {formatDuration(rally.duration)}
                        </Typography>
                      </Box>

                      {/* Highlight dots */}
                      {rallyHighlights.length > 0 && (
                        <Stack direction="row" spacing={0.5} sx={{ ml: 1 }}>
                          {rallyHighlights.slice(0, 3).map((hl) => (
                            <Box
                              key={hl.id}
                              sx={{
                                width: 8,
                                height: 8,
                                borderRadius: '50%',
                                bgcolor: hl.color,
                              }}
                            />
                          ))}
                          {rallyHighlights.length > 3 && (
                            <Chip
                              size="small"
                              label={`+${rallyHighlights.length - 3}`}
                              sx={{
                                height: 16,
                                fontSize: '0.625rem',
                                '& .MuiChip-label': { px: 0.5 },
                              }}
                            />
                          )}
                        </Stack>
                      )}
                    </ListItemButton>
                  );
                })}
              </Collapse>
            </Box>
          );
        })}
      </List>
    </Box>
  );
}
