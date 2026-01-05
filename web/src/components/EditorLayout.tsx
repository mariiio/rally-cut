'use client';

import { useEffect, useState } from 'react';
import { Box, IconButton, Tooltip, Badge, Skeleton, Typography, Stack, LinearProgress, Tabs, Tab } from '@mui/material';
import ListAltIcon from '@mui/icons-material/ListAlt';
import StarIcon from '@mui/icons-material/Star';
import AddIcon from '@mui/icons-material/Add';
import SportsVolleyballIcon from '@mui/icons-material/SportsVolleyball';
import VideocamIcon from '@mui/icons-material/Videocam';
import { VideoPlayer } from './VideoPlayer';
import { Timeline } from './Timeline';
import { RallyList } from './RallyList';
import { HighlightsPanel } from './HighlightsPanel';
import { CameraPanel } from './CameraPanel';
import { EditorHeader } from './EditorHeader';
import { CollapsiblePanel } from './CollapsiblePanel';
import { ExportProgress } from './ExportProgress';
import { UploadProgress } from './UploadProgress';
import { SessionLoadingProgress } from './SessionLoadingProgress';
import { NamePromptModal } from './NamePromptModal';
import { MobileEditorLayout } from './mobile';
import { useEditorStore } from '@/stores/editorStore';
import { useCameraStore } from '@/stores/cameraStore';
import { useIsMobile } from '@/hooks/useIsMobile';
import { designTokens } from '@/app/theme';

interface EditorLayoutProps {
  sessionId?: string;
}

export function EditorLayout({ sessionId }: EditorLayoutProps) {
  const isMobile = useIsMobile();
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
    currentUserName,
    setIsCameraTabActive,
    isCameraTabActive,
  } = useEditorStore();

  // Panel collapse state
  const [leftPanelCollapsed, setLeftPanelCollapsed] = useState(false);
  const [rightPanelCollapsed, setRightPanelCollapsed] = useState(false);
  const [showNamePromptModal, setShowNamePromptModal] = useState(false);
  const [pendingCreateHighlight, setPendingCreateHighlight] = useState(false);
  const [rightPanelTab, setRightPanelTab] = useState<'highlights' | 'camera'>('highlights');

  // Camera store - count rallies with camera edits (has keyframes in any aspect ratio)
  const cameraEdits = useCameraStore((state) => state.cameraEdits);
  const cameraEditCount = Object.values(cameraEdits).filter((e) =>
    (e.keyframes.ORIGINAL?.length ?? 0) > 0 || (e.keyframes.VERTICAL?.length ?? 0) > 0
  ).length;

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

      // Camera panel shortcut (Cmd/Ctrl + Shift + C)
      if (e.key === 'c' && e.shiftKey) {
        e.preventDefault();
        setRightPanelCollapsed(false);
        setRightPanelTab('camera');
        return;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [undo, redo, canUndo, canRedo]);

  // Sync camera tab state to store (for Timeline to access)
  useEffect(() => {
    setIsCameraTabActive(rightPanelTab === 'camera');
  }, [rightPanelTab, setIsCameraTabActive]);

  // Sync store camera tab state back to local tab (for Timeline keyframe clicks)
  useEffect(() => {
    if (isCameraTabActive && rightPanelTab !== 'camera') {
      setRightPanelTab('camera');
    }
  }, [isCameraTabActive, rightPanelTab]);

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
    // If user doesn't have a name yet, prompt them first
    if (!currentUserName) {
      setPendingCreateHighlight(true);
      setShowNamePromptModal(true);
      return;
    }
    const newId = createHighlight();
    selectHighlight(newId);
  };

  const handleNameSet = (name: string) => {
    // If there was a pending highlight creation, complete it now
    if (pendingCreateHighlight) {
      setPendingCreateHighlight(false);
      const newId = createHighlight();
      selectHighlight(newId);
    }
  };

  // Render mobile layout for phones
  if (isMobile) {
    return <MobileEditorLayout />;
  }

  // Desktop layout
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
            <SessionLoadingProgress />
            <UploadProgress />
            <ExportProgress />
            <VideoPlayer />
          </Box>
        </Box>

        {/* Right Panel - Highlights & Camera */}
        <CollapsiblePanel
          title={rightPanelTab === 'highlights' ? 'Highlights' : 'Camera'}
          count={rightPanelTab === 'highlights' ? highlights?.length : cameraEditCount}
          collapsed={rightPanelCollapsed}
          onToggle={() => setRightPanelCollapsed((prev) => !prev)}
          position="right"
          headerAction={
            rightPanelTab === 'highlights' ? (
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
            ) : null
          }
          collapsedIcon={
            <Stack spacing={1}>
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
                    onClick={() => {
                      setRightPanelCollapsed(false);
                      setRightPanelTab('highlights');
                    }}
                    sx={{ color: 'text.secondary' }}
                  >
                    <StarIcon fontSize="small" />
                  </IconButton>
                </Badge>
              </Tooltip>
              <Tooltip title="Camera" placement="left">
                <Badge
                  badgeContent={cameraEditCount || 0}
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
                    onClick={() => {
                      setRightPanelCollapsed(false);
                      setRightPanelTab('camera');
                    }}
                    sx={{ color: 'text.secondary' }}
                  >
                    <VideocamIcon fontSize="small" />
                  </IconButton>
                </Badge>
              </Tooltip>
            </Stack>
          }
        >
          {/* Tab switcher */}
          <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
            <Tabs
              value={rightPanelTab}
              onChange={(_, v) => setRightPanelTab(v)}
              variant="fullWidth"
              sx={{
                minHeight: 36,
                '& .MuiTab-root': {
                  minHeight: 36,
                  py: 0.5,
                  fontSize: '0.75rem',
                },
              }}
            >
              <Tab
                value="highlights"
                label="Highlights"
                icon={<StarIcon sx={{ fontSize: 16 }} />}
                iconPosition="start"
                sx={{ minHeight: 36 }}
              />
              <Tab
                value="camera"
                label="Camera"
                icon={<VideocamIcon sx={{ fontSize: 16 }} />}
                iconPosition="start"
                sx={{ minHeight: 36 }}
              />
            </Tabs>
          </Box>
          {/* Tab content */}
          <Box sx={{ flex: 1, overflow: 'hidden' }}>
            {rightPanelTab === 'highlights' ? <HighlightsPanel /> : <CameraPanel />}
          </Box>
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

      {/* Name Prompt Modal */}
      <NamePromptModal
        open={showNamePromptModal}
        onClose={() => {
          setShowNamePromptModal(false);
          setPendingCreateHighlight(false);
        }}
        onNameSet={handleNameSet}
      />
    </Box>
  );
}
