'use client';

import { Tooltip } from '@mui/material';
import CloudDoneIcon from '@mui/icons-material/CloudDone';
import CloudSyncIcon from '@mui/icons-material/CloudSync';
import CloudOffIcon from '@mui/icons-material/CloudOff';
import { useEditorStore } from '@/stores/editorStore';

/**
 * Subtle sync status indicator showing cloud sync state.
 */
export function SyncStatus() {
  const syncStatus = useEditorStore((state) => state.syncStatus);
  const syncNow = useEditorStore((state) => state.syncNow);

  if (!syncStatus) {
    return null;
  }

  const handleRetry = () => {
    syncNow();
  };

  // Error state - needs attention
  if (syncStatus.error) {
    return (
      <Tooltip title="Changes not saved to cloud. Click to retry.">
        <CloudOffIcon
          onClick={handleRetry}
          sx={{
            fontSize: 18,
            color: 'warning.main',
            cursor: 'pointer',
            opacity: 0.8,
            '&:hover': { opacity: 1 },
          }}
        />
      </Tooltip>
    );
  }

  // Syncing or pending changes
  if (syncStatus.isSyncing || syncStatus.pendingCount > 0) {
    return (
      <Tooltip title={syncStatus.isSyncing ? 'Syncing...' : 'Saving to cloud...'}>
        <CloudSyncIcon
          sx={{
            fontSize: 18,
            color: 'text.disabled',
            opacity: 0.6,
            animation: syncStatus.isSyncing ? 'pulse 1.5s infinite' : 'none',
            '@keyframes pulse': {
              '0%, 100%': { opacity: 0.6 },
              '50%': { opacity: 1 },
            },
          }}
        />
      </Tooltip>
    );
  }

  // All synced
  return (
    <Tooltip title="All changes saved">
      <CloudDoneIcon
        sx={{
          fontSize: 18,
          color: 'text.disabled',
          opacity: 0.5,
        }}
      />
    </Tooltip>
  );
}
