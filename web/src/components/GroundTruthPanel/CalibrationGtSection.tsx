'use client';

import { ReactNode } from 'react';
import { Box, Button, Chip, Typography } from '@mui/material';
import CropFreeIcon from '@mui/icons-material/CropFree';
import { usePlayerTrackingStore } from '@/stores/playerTrackingStore';
import { useEditorStore } from '@/stores/editorStore';

interface Props {
  renderSection: (status: ReactNode, body: ReactNode) => ReactNode;
}

export function CalibrationGtSection({ renderSection }: Props) {
  const activeMatchId = useEditorStore((s) => s.activeMatchId);
  const calibrations = usePlayerTrackingStore((s) => s.calibrations);
  const isCalibrating = usePlayerTrackingStore((s) => s.isCalibrating);
  const setIsCalibrating = usePlayerTrackingStore((s) => s.setIsCalibrating);

  const hasCalibration = activeMatchId ? !!calibrations[activeMatchId] : false;

  const status = (
    <Chip
      label={hasCalibration ? 'Done' : 'Not set'}
      size="small"
      color={hasCalibration ? 'success' : 'default'}
      sx={{ height: 18, fontSize: '0.65rem' }}
    />
  );

  const body = (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
      <Typography variant="caption" color="text.secondary">
        {hasCalibration
          ? 'Four court corners are calibrated for this video.'
          : 'Calibrate the four court corners to enable real-world stats.'}
      </Typography>
      <Button
        size="small"
        variant={hasCalibration ? 'outlined' : 'contained'}
        startIcon={<CropFreeIcon />}
        onClick={() => setIsCalibrating(true)}
        disabled={isCalibrating || !activeMatchId}
        sx={{ alignSelf: 'flex-start' }}
      >
        {hasCalibration ? 'Edit corners' : 'Calibrate corners'}
      </Button>
    </Box>
  );

  return <>{renderSection(status, body)}</>;
}
