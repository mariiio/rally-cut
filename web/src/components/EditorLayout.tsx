'use client';

import { useEffect } from 'react';
import {
  Box,
  AppBar,
  Toolbar,
  Typography,
  Paper,
} from '@mui/material';
import { VideoPlayer } from './VideoPlayer';
import { Timeline } from './Timeline';
import { SegmentList } from './SegmentList';
import { FileControls } from './FileControls';
import { useEditorStore } from '@/stores/editorStore';

export function EditorLayout() {
  const { segments, setVideoUrl, loadSegmentsFromJson } = useEditorStore();

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
          {/* Segment list - LEFT side */}
          <Paper
            sx={{
              width: 280,
              flexShrink: 0,
              display: 'flex',
              flexDirection: 'column',
              overflow: 'hidden',
            }}
          >
            <SegmentList />
          </Paper>

          {/* Video player - RIGHT side */}
          <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
            <VideoPlayer />
          </Box>
        </Box>

        {/* Timeline */}
        <Paper sx={{ flexShrink: 0 }}>
          <Timeline />
        </Paper>
      </Box>
    </Box>
  );
}
