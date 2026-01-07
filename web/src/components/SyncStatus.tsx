'use client';

import { Tooltip } from '@mui/material';
import CloudDoneIcon from '@mui/icons-material/CloudDone';
import CloudSyncIcon from '@mui/icons-material/CloudSync';
import CloudOffIcon from '@mui/icons-material/CloudOff';
import SaveIcon from '@mui/icons-material/Save';
import { useEditorStore } from '@/stores/editorStore';
import { useTierStore } from '@/stores/tierStore';

/**
 * Subtle sync status indicator showing cloud sync state.
 * FREE tier: Shows "saved locally" icon
 * PREMIUM tier: Shows cloud sync status with retry option
 */
export function SyncStatus() {
  const syncStatus = useEditorStore((state) => state.syncStatus);
  const syncNow = useEditorStore((state) => state.syncNow);
  const canSyncToServer = useTierStore((state) => state.canSyncToServer);

  if (!syncStatus) {
    return null;
  }

  // FREE tier - always show "saved locally", never show cloud status
  // This takes priority over all other states (syncing, error, etc.)
  if (!canSyncToServer()) {
    return (
      <Tooltip title="Changes saved locally. Upgrade to Premium for cloud sync.">
        <SaveIcon
          sx={{
            fontSize: 18,
            color: 'text.disabled',
            opacity: 0.5,
          }}
        />
      </Tooltip>
    );
  }

  // PREMIUM tier states below

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
    <Tooltip title="All changes saved to cloud">
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
