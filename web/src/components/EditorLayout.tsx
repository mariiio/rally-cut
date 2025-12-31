'use client';

import { useEffect, useState } from 'react';
import { Box, IconButton, Tooltip, Badge } from '@mui/material';
import ListAltIcon from '@mui/icons-material/ListAlt';
import StarIcon from '@mui/icons-material/Star';
import AddIcon from '@mui/icons-material/Add';
import { VideoPlayer } from './VideoPlayer';
import { Timeline } from './Timeline';
import { RallyList } from './RallyList';
import { HighlightsPanel } from './HighlightsPanel';
import { EditorHeader } from './EditorHeader';
import { CollapsiblePanel } from './CollapsiblePanel';
import { ExportProgress } from './ExportProgress';
import { useEditorStore } from '@/stores/editorStore';
import { designTokens } from '@/app/theme';

interface EditorLayoutProps {
  sessionId?: string;
}

export function EditorLayout({ sessionId }: EditorLayoutProps) {
  const {
    loadSession,
    session,
    undo,
    redo,
    canUndo,
    canRedo,
    rallies,
    highlights,
    createHighlight,
    canCreateHighlight,
    selectHighlight,
  } = useEditorStore();

  // Panel collapse state
  const [leftPanelCollapsed, setLeftPanelCollapsed] = useState(false);
  const [rightPanelCollapsed, setRightPanelCollapsed] = useState(false);

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

      // Panel toggle shortcuts
      if (e.key === '[') {
        e.preventDefault();
        setLeftPanelCollapsed((prev) => !prev);
        return;
      }
      if (e.key === ']') {
        e.preventDefault();
        setRightPanelCollapsed((prev) => !prev);
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

  // Responsive: collapse panels on smaller screens
  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth < 1024) {
        setLeftPanelCollapsed(true);
        setRightPanelCollapsed(true);
      }
    };

    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const handleCreateHighlight = () => {
    const newId = createHighlight();
    selectHighlight(newId);
  };

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        height: '100vh',
        bgcolor: 'background.default',
        overflow: 'hidden',
      }}
    >
      {/* Header */}
      <EditorHeader />

      {/* Main content */}
      <Box
        sx={{
          flex: 1,
          display: 'flex',
          minHeight: 0,
          overflow: 'hidden',
        }}
      >
        {/* Left Panel - Rally Navigator */}
        <CollapsiblePanel
          title="Rallies"
          count={rallies?.length}
          collapsed={leftPanelCollapsed}
          onToggle={() => setLeftPanelCollapsed((prev) => !prev)}
          position="left"
          collapsedIcon={
            <Tooltip title="Rallies" placement="right">
              <Badge
                badgeContent={rallies?.length || 0}
                color="primary"
                max={99}
                sx={{
                  '& .MuiBadge-badge': {
                    fontSize: '0.625rem',
                    height: 16,
                    minWidth: 16,
                  },
                }}
              >
                <IconButton
                  size="small"
                  onClick={() => setLeftPanelCollapsed(false)}
                  sx={{ color: 'text.secondary' }}
                >
                  <ListAltIcon fontSize="small" />
                </IconButton>
              </Badge>
            </Tooltip>
          }
        >
          <RallyList />
        </CollapsiblePanel>

        {/* Center - Video Area */}
        <Box
          sx={{
            flex: 1,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            bgcolor: designTokens.colors.video.background,
            p: 3,
            minWidth: 0,
            position: 'relative',
          }}
        >
          {/* Video Container */}
          <Box
            sx={{
              width: '100%',
              maxWidth: 1200,
              display: 'flex',
              flexDirection: 'column',
            }}
          >
            <VideoPlayer />
          </Box>
        </Box>

        {/* Right Panel - Highlights */}
        <CollapsiblePanel
          title="Highlights"
          count={highlights?.length}
          collapsed={rightPanelCollapsed}
          onToggle={() => setRightPanelCollapsed((prev) => !prev)}
          position="right"
          headerAction={
            <Tooltip
              title={
                canCreateHighlight()
                  ? 'Create highlight'
                  : 'All highlights must have segments first'
              }
            >
              <span>
                <IconButton
                  size="small"
                  onClick={handleCreateHighlight}
                  disabled={!canCreateHighlight()}
                  color="primary"
                >
                  <AddIcon fontSize="small" />
                </IconButton>
              </span>
            </Tooltip>
          }
          collapsedIcon={
            <Tooltip title="Highlights" placement="left">
              <Badge
                badgeContent={highlights?.length || 0}
                color="secondary"
                max={99}
                sx={{
                  '& .MuiBadge-badge': {
                    fontSize: '0.625rem',
                    height: 16,
                    minWidth: 16,
                    bgcolor: 'secondary.main',
                  },
                }}
              >
                <IconButton
                  size="small"
                  onClick={() => setRightPanelCollapsed(false)}
                  sx={{ color: 'text.secondary' }}
                >
                  <StarIcon fontSize="small" />
                </IconButton>
              </Badge>
            </Tooltip>
          }
        >
          <HighlightsPanel />
        </CollapsiblePanel>
      </Box>

      {/* Timeline */}
      <Box
        sx={{
          height: designTokens.spacing.timeline.normal,
          flexShrink: 0,
          bgcolor: 'background.paper',
          borderTop: '1px solid',
          borderColor: 'divider',
        }}
      >
        <Timeline />
      </Box>

      {/* Export progress indicator */}
      <ExportProgress />
    </Box>
  );
}
