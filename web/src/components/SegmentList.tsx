'use client';

import {
  Box,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  IconButton,
  Typography,
  Chip,
} from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import { useEditorStore } from '@/stores/editorStore';
import { usePlayerStore } from '@/stores/playerStore';
import { formatTime, formatDuration } from '@/utils/timeFormat';

interface SegmentListProps {
  onDelete?: (id: string) => void;
}

export function SegmentList({ onDelete }: SegmentListProps) {
  const { segments, selectedSegmentId, selectSegment } = useEditorStore();
  const { currentTime, seek } = usePlayerStore();

  const handleClick = (id: string, startTime: number) => {
    selectSegment(id);
    seek(startTime);
  };

  // Find which segment contains the current time
  const activeSegmentId = segments.find(
    (s) => currentTime >= s.start_time && currentTime <= s.end_time
  )?.id;

  if (segments.length === 0) {
    return (
      <Box
        sx={{
          p: 2,
          textAlign: 'center',
          color: 'text.secondary',
        }}
      >
        <Typography variant="body2">
          No segments loaded. Load a JSON file to see segments.
        </Typography>
      </Box>
    );
  }

  return (
    <List dense sx={{ overflow: 'auto', maxHeight: '100%' }}>
      {segments.map((segment, index) => {
        const isSelected = selectedSegmentId === segment.id;
        const isActive = activeSegmentId === segment.id;

        return (
          <ListItem
            key={segment.id}
            disablePadding
            secondaryAction={
              onDelete && (
                <IconButton
                  edge="end"
                  size="small"
                  onClick={(e) => {
                    e.stopPropagation();
                    onDelete(segment.id);
                  }}
                >
                  <DeleteIcon fontSize="small" />
                </IconButton>
              )
            }
            sx={{
              borderLeft: isActive ? 3 : 0,
              borderColor: 'primary.main',
            }}
          >
            <ListItemButton
              selected={isSelected}
              onClick={() => handleClick(segment.id, segment.start_time)}
            >
              <ListItemText
                primary={
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Chip
                      label={`#${index + 1}`}
                      size="small"
                      color={isActive ? 'primary' : 'default'}
                      sx={{ minWidth: 40 }}
                    />
                    <Typography variant="body2" component="span">
                      {formatTime(segment.start_time)} -{' '}
                      {formatTime(segment.end_time)}
                    </Typography>
                  </Box>
                }
                secondary={
                  <Typography variant="caption" color="text.secondary">
                    Duration: {formatDuration(segment.duration)}
                  </Typography>
                }
              />
            </ListItemButton>
          </ListItem>
        );
      })}
    </List>
  );
}
