'use client';

import { Alert, Box, Typography } from '@mui/material';
import { usePlayerTrackingStore } from '@/stores/playerTrackingStore';

/**
 * Persistent banner shown at the top of the video area while the user is
 * labeling action ground truth. Displays the keyboard legend and offers an
 * Esc / close control to exit labeling mode.
 */
export function LabelingModeBanner() {
  const isLabelingActions = usePlayerTrackingStore((s) => s.isLabelingActions);
  const setIsLabelingActions = usePlayerTrackingStore((s) => s.setIsLabelingActions);

  if (!isLabelingActions) return null;

  return (
    <Box
      sx={{
        position: 'absolute',
        top: 8,
        left: '50%',
        transform: 'translateX(-50%)',
        zIndex: 5,
        maxWidth: 'calc(100% - 24px)',
      }}
    >
      <Alert
        severity="warning"
        variant="filled"
        onClose={() => setIsLabelingActions(false)}
        sx={{ py: 0.25, '& .MuiAlert-message': { py: 0.25 } }}
      >
        <Typography variant="caption" sx={{ fontWeight: 600 }}>
          Labeling actions · S R T A B D · 1-4 · Esc to exit
        </Typography>
      </Alert>
    </Box>
  );
}
