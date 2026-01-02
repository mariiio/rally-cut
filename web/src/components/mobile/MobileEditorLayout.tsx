'use client';

import { useState, useRef, useEffect } from 'react';
import { Box } from '@mui/material';
import { VideoPlayer } from '../VideoPlayer';
import { MobileEditorHeader } from './MobileEditorHeader';
import { MobilePlayerControls } from './MobilePlayerControls';
import { MobileTabNavigation } from './MobileTabNavigation';
import { MobileRallyList } from './MobileRallyList';
import { MobileHighlightsPanel } from './MobileHighlightsPanel';
import { MobileRallyEditorModal } from './MobileRallyEditorModal';
import { UploadProgress } from '../UploadProgress';
import { ExportProgress } from '../ExportProgress';
import { useEditorStore } from '@/stores/editorStore';
import { designTokens } from '@/app/theme';

type TabValue = 'rallies' | 'highlights';

export function MobileEditorLayout() {
  const { rallies, highlights } = useEditorStore();
  const [activeTab, setActiveTab] = useState<TabValue>('rallies');
  const [editingRallyId, setEditingRallyId] = useState<string | null>(null);

  // Preserve scroll positions when switching tabs
  const scrollPositions = useRef<{ rallies: number; highlights: number }>({
    rallies: 0,
    highlights: 0,
  });
  const contentRef = useRef<HTMLDivElement>(null);

  const handleTabChange = (newTab: TabValue) => {
    // Save current scroll position
    if (contentRef.current) {
      scrollPositions.current[activeTab] = contentRef.current.scrollTop;
    }
    setActiveTab(newTab);
  };

  // Restore scroll position after tab switch
  useEffect(() => {
    if (contentRef.current) {
      contentRef.current.scrollTop = scrollPositions.current[activeTab];
    }
  }, [activeTab]);

  const handleRallyTap = (rallyId: string) => {
    setEditingRallyId(rallyId);
  };

  const handleCloseEditor = () => {
    setEditingRallyId(null);
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
      <MobileEditorHeader />

      {/* Video Player Area */}
      <Box
        sx={{
          flexShrink: 0,
          bgcolor: designTokens.colors.video.background,
          position: 'relative',
        }}
      >
        <UploadProgress />
        <ExportProgress />
        <VideoPlayer />
      </Box>

      {/* Player Controls */}
      <MobilePlayerControls />

      {/* Scrollable Content Panel */}
      <Box
        ref={contentRef}
        sx={{
          flex: 1,
          overflow: 'auto',
          pb: `${designTokens.mobile.bottomNav.height + 16}px`, // Space for bottom nav
          WebkitOverflowScrolling: 'touch', // Smooth scrolling on iOS
        }}
      >
        {activeTab === 'rallies' ? (
          <MobileRallyList onRallyTap={handleRallyTap} />
        ) : (
          <MobileHighlightsPanel onRallyTap={handleRallyTap} />
        )}
      </Box>

      {/* Bottom Tab Navigation */}
      <MobileTabNavigation
        activeTab={activeTab}
        onTabChange={handleTabChange}
        rallyCount={rallies?.length || 0}
        highlightCount={highlights?.length || 0}
      />

      {/* Rally Editor Modal */}
      <MobileRallyEditorModal
        open={editingRallyId !== null}
        onClose={handleCloseEditor}
        rallyId={editingRallyId}
      />
    </Box>
  );
}
