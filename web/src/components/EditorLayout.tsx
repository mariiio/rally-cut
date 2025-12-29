'use client';

import { useState } from 'react';
import {
  Box,
  AppBar,
  Toolbar,
  Typography,
  Paper,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
} from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import { VideoPlayer } from './VideoPlayer';
import { PlayerControls } from './PlayerControls';
import { Timeline } from './Timeline';
import { SegmentList } from './SegmentList';
import { FileControls } from './FileControls';
import { SegmentForm } from './SegmentForm';
import { useEditorStore } from '@/stores/editorStore';
import { formatDuration } from '@/utils/timeFormat';

export function EditorLayout() {
  const [isAddDialogOpen, setIsAddDialogOpen] = useState(false);
  const [editingSegmentId, setEditingSegmentId] = useState<string | null>(null);
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);

  const { segments, removeSegment, videoMetadata } = useEditorStore();

  const handleEdit = (id: string) => {
    setEditingSegmentId(id);
  };

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
        <Box sx={{ display: 'flex', gap: 2 }}>
          {/* Video player */}
          <Box sx={{ flex: 2 }}>
            <VideoPlayer />
            <PlayerControls />
          </Box>

          {/* Segment list */}
          <Paper
            sx={{
              flex: 1,
              display: 'flex',
              flexDirection: 'column',
              overflow: 'hidden',
              minWidth: 280,
            }}
          >
            <Box
              sx={{
                p: 1,
                borderBottom: 1,
                borderColor: 'divider',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
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
              <Button
                size="small"
                startIcon={<AddIcon />}
                onClick={() => setIsAddDialogOpen(true)}
                disabled={!videoMetadata}
              >
                Add
              </Button>
            </Box>
            <Box sx={{ flex: 1, overflow: 'auto' }}>
              <SegmentList
                onEdit={handleEdit}
                onDelete={(id) => setDeleteConfirmId(id)}
              />
            </Box>
          </Paper>
        </Box>

        {/* Timeline */}
        <Paper sx={{ p: 1 }}>
          <Timeline />
        </Paper>
      </Box>

      {/* Add/Edit Segment Dialog */}
      <SegmentForm
        open={isAddDialogOpen || editingSegmentId !== null}
        segmentId={editingSegmentId}
        onClose={() => {
          setIsAddDialogOpen(false);
          setEditingSegmentId(null);
        }}
      />

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
