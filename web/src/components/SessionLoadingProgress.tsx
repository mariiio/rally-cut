'use client';

import { useEditorStore } from '@/stores/editorStore';
import { ProgressBar } from './ProgressBar';

export function SessionLoadingProgress() {
  const isLoadingSession = useEditorStore((state) => state.isLoadingSession);
  const sessionLoadStep = useEditorStore((state) => state.sessionLoadStep);
  const sessionLoadProgress = useEditorStore((state) => state.sessionLoadProgress);

  // Show nothing when not loading
  if (!isLoadingSession || sessionLoadProgress === 0) {
    return null;
  }

  return (
    <ProgressBar
      progress={sessionLoadProgress}
      isActive={isLoadingSession}
      stepText={sessionLoadStep}
    />
  );
}
