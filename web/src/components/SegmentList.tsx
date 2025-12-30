'use client';

import { Box, Typography, Chip, Stack } from '@mui/material';
import { useEditorStore } from '@/stores/editorStore';
import { usePlayerStore } from '@/stores/playerStore';
import { formatTime, formatDuration } from '@/utils/timeFormat';

export function SegmentList() {
  const { segments, selectedSegmentId, selectSegment, getHighlightsForSegment } = useEditorStore();
  const { currentTime, seek } = usePlayerStore();

  const handleClick = (id: string, startTime: number) => {
    selectSegment(id);
    seek(startTime);
  };

  // Sort segments by start time for display
  const sortedSegments = [...segments].sort((a, b) => a.start_time - b.start_time);

  // Find which segment contains the current time
  const activeSegmentId = sortedSegments.find(
    (s) => currentTime >= s.start_time && currentTime <= s.end_time
  )?.id;

  // Calculate total duration
  const totalDuration = sortedSegments.reduce((sum, s) => sum + s.duration, 0);

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
        <Box>
          <Typography
            variant="overline"
            sx={{
              fontSize: 11,
              fontWeight: 600,
              letterSpacing: 1,
              color: 'text.secondary',
            }}
          >
            Segments
          </Typography>
          {sortedSegments.length > 0 && (
            <Typography
              variant="caption"
              sx={{
                display: 'block',
                color: 'text.disabled',
                fontSize: 10,
                mt: -0.5,
              }}
            >
              Total: {formatDuration(totalDuration)}
            </Typography>
          )}
        </Box>
        {sortedSegments.length > 0 && (
          <Chip
            label={sortedSegments.length}
            size="small"
            sx={{
              height: 20,
              fontSize: 11,
              fontWeight: 600,
              bgcolor: 'action.selected',
            }}
          />
        )}
      </Box>

      {/* Segment list */}
      <Box sx={{ flex: 1, overflow: 'auto' }}>
        {sortedSegments.length === 0 ? (
          <Box
            sx={{
              p: 2,
              textAlign: 'center',
              color: 'text.disabled',
            }}
          >
            <Typography variant="caption">
              No segments loaded
            </Typography>
          </Box>
        ) : (
          <Box sx={{ py: 0.5 }}>
            {sortedSegments.map((segment, index) => {
              const isSelected = selectedSegmentId === segment.id;
              const isActive = activeSegmentId === segment.id;

              return (
                <Box
                  key={segment.id}
                  onClick={() => handleClick(segment.id, segment.start_time)}
                  sx={{
                    display: 'flex',
                    alignItems: 'center',
                    px: 1.5,
                    py: 0.75,
                    cursor: 'pointer',
                    borderLeft: '2px solid',
                    borderColor: isActive ? 'primary.main' : 'transparent',
                    bgcolor: isSelected
                      ? 'rgba(33, 150, 243, 0.12)'
                      : 'transparent',
                    transition: 'all 0.15s ease',
                    '&:hover': {
                      bgcolor: isSelected
                        ? 'rgba(33, 150, 243, 0.18)'
                        : 'rgba(255, 255, 255, 0.04)',
                    },
                  }}
                >
                  {/* Status indicator */}
                  <Box
                    sx={{
                      width: 16,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      mr: 0.5,
                      color: isActive
                        ? 'primary.main'
                        : isSelected
                        ? 'text.primary'
                        : 'transparent',
                      fontSize: 10,
                    }}
                  >
                    {isActive ? '●' : isSelected ? '▸' : ''}
                  </Box>

                  {/* Segment number */}
                  <Typography
                    sx={{
                      fontFamily: 'monospace',
                      fontSize: 11,
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
                      fontSize: 11,
                      color: isSelected ? 'text.primary' : 'text.secondary',
                      flex: 1,
                      mx: 1,
                    }}
                  >
                    {formatTime(segment.start_time)}
                    <Box
                      component="span"
                      sx={{ color: 'text.disabled', mx: 0.5 }}
                    >
                      →
                    </Box>
                    {formatTime(segment.end_time)}
                  </Typography>

                  {/* Duration */}
                  <Typography
                    sx={{
                      fontFamily: 'monospace',
                      fontSize: 10,
                      color: 'text.disabled',
                      flexShrink: 0,
                    }}
                  >
                    {segment.duration.toFixed(1)}s
                  </Typography>

                  {/* Highlight color dots */}
                  {(() => {
                    const segmentHighlights = getHighlightsForSegment(segment.id);
                    if (segmentHighlights.length === 0) return null;
                    return (
                      <Stack direction="row" spacing={0.25} sx={{ ml: 0.75 }}>
                        {segmentHighlights.slice(0, 3).map((h) => (
                          <Box
                            key={h.id}
                            sx={{
                              width: 6,
                              height: 6,
                              borderRadius: '50%',
                              bgcolor: h.color,
                            }}
                          />
                        ))}
                        {segmentHighlights.length > 3 && (
                          <Typography
                            sx={{ fontSize: 8, color: 'text.disabled', lineHeight: 1 }}
                          >
                            +{segmentHighlights.length - 3}
                          </Typography>
                        )}
                      </Stack>
                    );
                  })()}
                </Box>
              );
            })}
          </Box>
        )}
      </Box>
    </Box>
  );
}
