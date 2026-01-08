'use client';

import { useState, useEffect } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  Stack,
  Typography,
  IconButton,
} from '@mui/material';
import AccessTimeIcon from '@mui/icons-material/AccessTime';
import { useEditorStore } from '@/stores/editorStore';
import { usePlayerStore } from '@/stores/playerStore';
import { formatTime, parseTime } from '@/utils/timeFormat';

interface RallyFormProps {
  open: boolean;
  rallyId: string | null; // null = add mode, string = edit mode
  onClose: () => void;
}

export function RallyForm({ open, rallyId, onClose }: RallyFormProps) {
  const [startTimeStr, setStartTimeStr] = useState('');
  const [endTimeStr, setEndTimeStr] = useState('');
  const [error, setError] = useState<string | null>(null);

  const { rallies, addRally, updateRally, videoMetadata } = useEditorStore();
  const { currentTime } = usePlayerStore();

  const isEditing = rallyId !== null;
  const rally = isEditing ? rallies?.find((s) => s.id === rallyId) : null;

  // Initialize form when opened
  useEffect(() => {
    if (open) {
      if (isEditing && rally) {
        // eslint-disable-next-line react-hooks/set-state-in-effect -- intentional: initializing form state
        setStartTimeStr(formatTime(rally.start_time));
        setEndTimeStr(formatTime(rally.end_time));
      } else {
        // Default to current time for new rallies
        setStartTimeStr(formatTime(currentTime));
        setEndTimeStr(formatTime(currentTime + 10)); // Default 10 second rally
      }
      setError(null);
    }
  }, [open, isEditing, rally, currentTime]);

  const handleSetStartToCurrent = () => {
    setStartTimeStr(formatTime(currentTime));
  };

  const handleSetEndToCurrent = () => {
    setEndTimeStr(formatTime(currentTime));
  };

  const handleSubmit = () => {
    const startTime = parseTime(startTimeStr);
    const endTime = parseTime(endTimeStr);

    if (startTime === null) {
      setError('Invalid start time format. Use MM:SS or MM:SS.s');
      return;
    }
    if (endTime === null) {
      setError('Invalid end time format. Use MM:SS or MM:SS.s');
      return;
    }
    if (endTime <= startTime) {
      setError('End time must be after start time');
      return;
    }

    if (isEditing && rallyId) {
      updateRally(rallyId, {
        start_time: startTime,
        end_time: endTime,
      });
    } else {
      addRally(startTime, endTime);
    }

    onClose();
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="xs" fullWidth>
      <DialogTitle>{isEditing ? 'Edit Rally' : 'Add Rally'}</DialogTitle>
      <DialogContent>
        <Stack spacing={2} sx={{ mt: 1 }}>
          {error && (
            <Typography color="error" variant="body2">
              {error}
            </Typography>
          )}

          <Stack direction="row" spacing={1} alignItems="center">
            <TextField
              label="Start Time"
              value={startTimeStr}
              onChange={(e) => setStartTimeStr(e.target.value)}
              placeholder="MM:SS.s"
              size="small"
              fullWidth
            />
            <IconButton
              onClick={handleSetStartToCurrent}
              title="Set to current video time"
              size="small"
            >
              <AccessTimeIcon />
            </IconButton>
          </Stack>

          <Stack direction="row" spacing={1} alignItems="center">
            <TextField
              label="End Time"
              value={endTimeStr}
              onChange={(e) => setEndTimeStr(e.target.value)}
              placeholder="MM:SS.s"
              size="small"
              fullWidth
            />
            <IconButton
              onClick={handleSetEndToCurrent}
              title="Set to current video time"
              size="small"
            >
              <AccessTimeIcon />
            </IconButton>
          </Stack>

          {videoMetadata && (
            <Typography variant="caption" color="text.secondary">
              Video duration: {formatTime(videoMetadata.duration)}
            </Typography>
          )}
        </Stack>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button onClick={handleSubmit} variant="contained">
          {isEditing ? 'Save' : 'Add'}
        </Button>
      </DialogActions>
    </Dialog>
  );
}
