'use client';

import { useMemo, useState, useEffect } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  FormControlLabel,
  Checkbox,
  ToggleButton,
  ToggleButtonGroup,
  Typography,
  Box,
  Link,
  CircularProgress,
} from '@mui/material';
import LockIcon from '@mui/icons-material/Lock';
import FileDownloadIcon from '@mui/icons-material/FileDownload';
import { useTierStore } from '@/stores/tierStore';
import { useCameraStore } from '@/stores/cameraStore';
import { Rally } from '@/types/rally';

export type ExportQuality = 'original' | '720p';

export interface ExportOptions {
  quality: ExportQuality;
  applyCameraEdits: boolean;
}

interface ExportOptionsDialogProps {
  open: boolean;
  onClose: () => void;
  onExport: (options: ExportOptions) => void;
  title: string;
  rallies: Rally[];
  videoId?: string;
  isExporting?: boolean;
}

export function ExportOptionsDialog({
  open,
  onClose,
  onExport,
  title,
  rallies,
  videoId,
  isExporting = false,
}: ExportOptionsDialogProps) {
  const isPaidTier = useTierStore((state) => state.isPaidTier());
  const cameraStore = useCameraStore();

  // Check if any rally has camera edits
  const hasCameraEdits = useMemo(() => {
    // Check per-rally keyframes
    for (const rally of rallies) {
      if (cameraStore.hasAnyKeyframes(rally.id)) {
        return true;
      }
    }

    // Check global camera settings for the video
    if (videoId && cameraStore.hasGlobalSettings(videoId)) {
      return true;
    }

    return false;
  }, [rallies, videoId, cameraStore]);

  // State for options
  const [quality, setQuality] = useState<ExportQuality>(
    isPaidTier ? 'original' : '720p'
  );
  const [applyCameraEdits, setApplyCameraEdits] = useState(true);

  // Reset state when dialog opens - setState in effect is intentional for dialog reset pattern
  /* eslint-disable react-hooks/set-state-in-effect */
  useEffect(() => {
    if (open) {
      setQuality(isPaidTier ? 'original' : '720p');
      setApplyCameraEdits(true);
    }
  }, [open, isPaidTier]);
  /* eslint-enable react-hooks/set-state-in-effect */

  const handleExport = () => {
    onExport({
      quality,
      applyCameraEdits: hasCameraEdits && applyCameraEdits,
    });
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="xs" fullWidth>
      <DialogTitle>{title}</DialogTitle>
      <DialogContent dividers>
        {/* Quality Selection */}
        <Box sx={{ mb: 3 }}>
          <Typography variant="subtitle2" sx={{ mb: 1 }}>
            Quality
          </Typography>
          <ToggleButtonGroup
            value={quality}
            exclusive
            onChange={(_, value) => value && setQuality(value)}
            fullWidth
            size="small"
          >
            <ToggleButton
              value="original"
              disabled={!isPaidTier}
              sx={{ textTransform: 'none' }}
            >
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                {!isPaidTier && <LockIcon sx={{ fontSize: 14 }} />}
                Original
              </Box>
            </ToggleButton>
            <ToggleButton value="720p" sx={{ textTransform: 'none' }}>
              720p
            </ToggleButton>
          </ToggleButtonGroup>
          {!isPaidTier && (
            <Typography variant="caption" sx={{ color: 'text.secondary', mt: 0.5, display: 'block' }}>
              <Link href="/pricing" sx={{ color: 'primary.main' }}>
                Upgrade to Pro
              </Link>{' '}
              for original quality exports
            </Typography>
          )}
        </Box>

        {/* Options Section */}
        {hasCameraEdits && (
          <Box>
            <Typography variant="subtitle2" sx={{ mb: 1 }}>
              Options
            </Typography>

            {/* Camera Edits Toggle */}
            <FormControlLabel
              control={
                <Checkbox
                  checked={applyCameraEdits}
                  onChange={(e) => setApplyCameraEdits(e.target.checked)}
                  size="small"
                />
              }
              label="Apply camera effects"
            />
          </Box>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose} disabled={isExporting}>
          Cancel
        </Button>
        <Button
          onClick={handleExport}
          variant="contained"
          disabled={isExporting || rallies.length === 0}
          startIcon={isExporting ? <CircularProgress size={16} /> : <FileDownloadIcon />}
        >
          {isExporting ? 'Exporting...' : 'Download'}
        </Button>
      </DialogActions>
    </Dialog>
  );
}
