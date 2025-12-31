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
import { RallyList } from './RallyList';
import { HighlightsPanel } from './HighlightsPanel';
import { FileControls } from './FileControls';
import { ExportProgress } from './ExportProgress';
import { useEditorStore } from '@/stores/editorStore';

interface EditorLayoutProps {
  sessionId?: string;
}

export function EditorLayout({ sessionId }: EditorLayoutProps) {
  const { loadSession, session, undo, redo, canUndo, canRedo } = useEditorStore();

  // Keyboard shortcuts for undo/redo
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Check for Cmd/Ctrl modifier
      const isMod = e.metaKey || e.ctrlKey;
      if (!isMod) return;

      // Undo: Cmd/Ctrl + Z (without Shift)
      if (e.key === 'z' && !e.shiftKey) {
        e.preventDefault();
        if (canUndo()) {
          undo();
        }
        return;
      }

      // Redo: Cmd/Ctrl + Shift + Z or Cmd/Ctrl + Y
      if ((e.key === 'z' && e.shiftKey) || e.key === 'y') {
        e.preventDefault();
        if (canRedo()) {
          redo();
        }
        return;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [undo, redo, canUndo, canRedo]);

  // Load session data
  useEffect(() => {
    if (sessionId && !session) {
      loadSession(sessionId);
    }
  }, [sessionId, session, loadSession]);

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
          {/* Rally list - LEFT side */}
          <Paper
            sx={{
              width: 280,
              flexShrink: 0,
              display: 'flex',
              flexDirection: 'column',
              overflow: 'hidden',
            }}
          >
            <RallyList />
          </Paper>

          {/* Video player - CENTER */}
          <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
            <VideoPlayer />
          </Box>

          {/* Highlights panel - RIGHT side */}
          <Paper
            sx={{
              width: 240,
              flexShrink: 0,
              display: 'flex',
              flexDirection: 'column',
              overflow: 'hidden',
            }}
          >
            <HighlightsPanel />
          </Paper>
        </Box>

        {/* Timeline */}
        <Paper sx={{ flexShrink: 0 }}>
          <Timeline />
        </Paper>
      </Box>

      {/* Export progress indicator */}
      <ExportProgress />
    </Box>
  );
}
