'use client';

import { useEffect, useState, useRef } from 'react';
import { Box, IconButton, Tooltip, Badge, Typography, Stack, Tabs, Tab } from '@mui/material';
import ListAltIcon from '@mui/icons-material/ListAlt';
import StarIcon from '@mui/icons-material/Star';
import AddIcon from '@mui/icons-material/Add';
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
import { OriginalQualityBanner } from './OriginalQualityBanner';
import { NamePromptModal } from './NamePromptModal';
import { MobileEditorLayout } from './mobile';
import { AccessRequestForm } from './AccessRequestForm';
import { TutorialProvider, TutorialContext } from './tutorial';
import { useEditorStore } from '@/stores/editorStore';
import { useIsMobile } from '@/hooks/useIsMobile';
import { designTokens } from '@/app/theme';

interface EditorLayoutProps {
  sessionId?: string;
  videoId?: string;
  initialVideoId?: string;
}

export function EditorLayout({ sessionId, videoId, initialVideoId }: EditorLayoutProps) {
  const isMobile = useIsMobile();
  const {
    loadSession,
    loadVideo,
    undo,
    redo,
    canUndo,
    canRedo,
    rallies,
    getAllRallies,
    highlights,
    createHighlight,
    canCreateHighlight,
    selectHighlight,
    currentUserName,
    setIsCameraTabActive,
    isCameraTabActive,
    leftPanelTab,
    setLeftPanelTab,
    getActiveMatch,
    sessionError,
    userRole,
  } = useEditorStore();

  // Get current match for quality banner
  const currentMatch = getActiveMatch();

  // Panel collapse state
  const [leftPanelCollapsed, setLeftPanelCollapsed] = useState(false);
  const [rightPanelCollapsed, setRightPanelCollapsed] = useState(false);
  const [showNamePromptModal, setShowNamePromptModal] = useState(false);
  const [pendingCreateHighlight, setPendingCreateHighlight] = useState(false);

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
        return;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [undo, redo, canUndo, canRedo]);

  // When store requests camera tab active (e.g., from Timeline keyframe click or camera button), expand right panel
  const prevIsCameraTabActive = useRef(isCameraTabActive);
  useEffect(() => {
    if (isCameraTabActive && !prevIsCameraTabActive.current) {
      // eslint-disable-next-line react-hooks/set-state-in-effect -- intentional: auto-expand panel when camera tab activated
      setRightPanelCollapsed(false);
    }
    prevIsCameraTabActive.current = isCameraTabActive;
  }, [isCameraTabActive]);

  // Exit camera edit mode when right panel is collapsed
  useEffect(() => {
    if (rightPanelCollapsed && isCameraTabActive) {
      setIsCameraTabActive(false);
    }
  }, [rightPanelCollapsed, isCameraTabActive, setIsCameraTabActive]);

  // Load session or video data on mount
  // Always fetch fresh data - localStorage edits are applied inside loadSession/loadVideo
  // This ensures deleted videos don't appear as ghosts while preserving user's local edits
  useEffect(() => {
    if (videoId) {
      loadVideo(videoId);
    } else if (sessionId) {
      loadSession(sessionId, initialVideoId);
    }
  }, [sessionId, videoId, initialVideoId, loadSession, loadVideo]);

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

  const handleNameSet = (_name: string) => {
    // If there was a pending highlight creation, complete it now
    if (pendingCreateHighlight) {
      setPendingCreateHighlight(false);
      const newId = createHighlight();
      selectHighlight(newId);
    }
  };

  // Handle session errors (access denied, not found)
  if (sessionError) {
    if (sessionError.type === 'access_denied' && sessionId) {
      return (
        <AccessRequestForm
          sessionId={sessionId}
          sessionName={sessionError.sessionName}
          ownerName={sessionError.ownerName}
          hasPendingRequest={sessionError.hasPendingRequest}
        />
      );
    }
    // For not_found or unknown errors, show a simple error page
    return (
      <Box
        sx={{
          minHeight: '100vh',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          bgcolor: 'grey.900',
        }}
      >
        <Typography color="text.secondary">
          {sessionError.type === 'not_found'
            ? 'Session not found'
            : 'Failed to load session'}
        </Typography>
      </Box>
    );
  }

  // Render mobile layout for phones
  if (isMobile) {
    return <MobileEditorLayout />;
  }

  // Tutorial context - show "Add Rallies" step for owners or when no rallies exist
  const tutorialContext: TutorialContext = {
    userRole: userRole ?? 'owner',
    hasRallies: (rallies?.length ?? 0) > 0,
  };

  // Desktop layout
  return (
    <TutorialProvider context={tutorialContext}>
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
        {/* Left Panel - Rallies & Highlights */}
        <CollapsiblePanel
          title=""
          collapsed={leftPanelCollapsed}
          onToggle={() => setLeftPanelCollapsed((prev) => !prev)}
          position="left"
          headerAction={
            leftPanelTab === 'highlights' ? (
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
              <Tooltip title="Rallies" placement="right">
                <Badge
                  badgeContent={getAllRallies().length}
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
                      setLeftPanelCollapsed(false);
                      setLeftPanelTab('rallies');
                    }}
                    sx={{ color: 'text.secondary' }}
                  >
                    <ListAltIcon fontSize="small" />
                  </IconButton>
                </Badge>
              </Tooltip>
              <Tooltip title="Highlights" placement="right">
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
                      setLeftPanelCollapsed(false);
                      setLeftPanelTab('highlights');
                    }}
                    sx={{ color: 'text.secondary' }}
                  >
                    <StarIcon fontSize="small" />
                  </IconButton>
                </Badge>
              </Tooltip>
            </Stack>
          }
        >
          {/* Tab switcher */}
          <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
            <Tabs
              value={leftPanelTab}
              onChange={(_, v) => setLeftPanelTab(v)}
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
                value="rallies"
                label={`Rallies (${getAllRallies().length})`}
                icon={<ListAltIcon sx={{ fontSize: 16 }} />}
                iconPosition="start"
                sx={{ minHeight: 36 }}
              />
              <Tab
                value="highlights"
                label={`Highlights (${highlights?.length ?? 0})`}
                icon={<StarIcon sx={{ fontSize: 16 }} />}
                iconPosition="start"
                sx={{ minHeight: 36 }}
              />
            </Tabs>
          </Box>
          {/* Tab content */}
          <Box sx={{ flex: 1, overflow: 'hidden' }}>
            {leftPanelTab === 'rallies' ? (
              <RallyList />
            ) : (
              <HighlightsPanel />
            )}
          </Box>
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
            overflow: 'hidden',
          }}
        >
          {/* Video Container */}
          <Box
            sx={{
              width: '100%',
              maxWidth: 1200,
              maxHeight: '100%',
              display: 'flex',
              flexDirection: 'column',
            }}
          >
            <SessionLoadingProgress />
            <UploadProgress />
            <ExportProgress />
            <OriginalQualityBanner currentMatch={currentMatch} />
            <VideoPlayer />
          </Box>
        </Box>

        {/* Right Panel - Camera */}
        <CollapsiblePanel
          title="Camera"
          collapsed={rightPanelCollapsed}
          onToggle={() => setRightPanelCollapsed((prev) => !prev)}
          position="right"
          collapsedIcon={
            <Tooltip title="Camera" placement="left">
              <IconButton
                size="small"
                onClick={() => setRightPanelCollapsed(false)}
                sx={{ color: 'text.secondary' }}
              >
                <VideocamIcon fontSize="small" />
              </IconButton>
            </Tooltip>
          }
        >
          <CameraPanel />
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
    </TutorialProvider>
  );
}
