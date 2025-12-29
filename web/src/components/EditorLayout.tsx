'use client';

import { useState, useEffect } from 'react';
import {
  Box,
  AppBar,
  Toolbar,
  Typography,
  Paper,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
  Button,
} from '@mui/material';
import { VideoPlayer } from './VideoPlayer';
import { Timeline } from './Timeline';
import { SegmentList } from './SegmentList';
import { FileControls } from './FileControls';
import { useEditorStore } from '@/stores/editorStore';
import { formatDuration } from '@/utils/timeFormat';

export function EditorLayout() {
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);

  const { segments, removeSegment, setVideoUrl, loadSegmentsFromJson } = useEditorStore();

  // Auto-load sample data in development
  useEffect(() => {
    if (process.env.NODE_ENV === 'development' && segments.length === 0) {
      const loadSampleData = async () => {
        try {
          const response = await fetch('/samples/segments.json');
          if (response.ok) {
            const json = await response.json();
            loadSegmentsFromJson(json);
            setVideoUrl('/samples/video.mov');
          }
        } catch (e) {
          // Sample files not available, ignore
        }
      };
      loadSampleData();
    }
  }, []);

  const handleDeleteConfirm = () => {
    if (deleteConfirmId) {
      removeSegment(deleteConfirmId);
      setDeleteConfirmId(null);
    }
  };

  const totalKeptDuration = segments.reduce((sum, s) => sum + s.duration, 0);

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      {/* Header */}
      <AppBar position="static" color="default" elevation={1}>
        <Toolbar variant="dense">
          <Typography variant="h6" sx={{ flexGrow: 1 }}>
            RallyCut Editor
          </Typography>
          <FileControls />
        </Toolbar>
      </AppBar>

      {/* Main content */}
      <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', p: 2, gap: 2, overflow: 'hidden' }}>
        {/* Video section */}
        <Box sx={{ flex: 1, display: 'flex', gap: 2, minHeight: 0 }}>
          {/* Video player */}
          <Box sx={{ flex: 2, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
            <VideoPlayer />
          </Box>

          {/* Segment list */}
          <Paper
            sx={{
              flex: 1,
              display: 'flex',
              flexDirection: 'column',
              overflow: 'hidden',
              minWidth: 280,
              maxWidth: 350,
            }}
          >
            <Box
              sx={{
                p: 1,
                borderBottom: 1,
                borderColor: 'divider',
              }}
            >
              <Typography variant="subtitle2">
                Segments ({segments.length})
                {segments.length > 0 && (
                  <Typography
                    component="span"
                    variant="caption"
                    color="text.secondary"
                    sx={{ ml: 1 }}
                  >
                    {formatDuration(totalKeptDuration)}
                  </Typography>
                )}
              </Typography>
            </Box>
            <Box sx={{ flex: 1, overflow: 'auto' }}>
              <SegmentList
                onDelete={(id) => setDeleteConfirmId(id)}
              />
            </Box>
          </Paper>
        </Box>

        {/* Timeline */}
        <Paper sx={{ flexShrink: 0 }}>
          <Timeline />
        </Paper>
      </Box>

      {/* Delete Confirmation Dialog */}
      <Dialog
        open={deleteConfirmId !== null}
        onClose={() => setDeleteConfirmId(null)}
      >
        <DialogTitle>Delete Segment</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Are you sure you want to delete this segment? This action cannot be
            undone.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteConfirmId(null)}>Cancel</Button>
          <Button onClick={handleDeleteConfirm} color="error" variant="contained">
            Delete
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
