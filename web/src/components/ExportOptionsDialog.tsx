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
  Tooltip,
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
  withFade: boolean;
}

interface ExportOptionsDialogProps {
  open: boolean;
  onClose: () => void;
  onExport: (options: ExportOptions) => void;
  title: string;
  rallies: Rally[];
  videoId?: string;
  showFadeOption?: boolean;
  isExporting?: boolean;
}

export function ExportOptionsDialog({
  open,
  onClose,
  onExport,
  title,
  rallies,
  videoId,
  showFadeOption = false,
  isExporting = false,
}: ExportOptionsDialogProps) {
  const isPaidTier = useTierStore((state) => state.isPaidTier());
  const cameraStore = useCameraStore();

  // Check if any rally has camera edits
  const { hasCameraEdits, hasVerticalEdits } = useMemo(() => {
    let hasCameraEdits = false;
    let hasVerticalEdits = false;

    // Check per-rally keyframes
    for (const rally of rallies) {
      if (cameraStore.hasAnyKeyframes(rally.id)) {
        hasCameraEdits = true;
        // Check if this rally has vertical keyframes
        const edit = cameraStore.cameraEdits[rally.id];
        if (edit?.keyframes?.VERTICAL?.length > 0) {
          hasVerticalEdits = true;
        }
      }
    }

    // Check global camera settings for the video
    if (videoId && cameraStore.hasGlobalSettings(videoId)) {
      hasCameraEdits = true;
    }

    return { hasCameraEdits, hasVerticalEdits };
  }, [rallies, videoId, cameraStore]);

  // State for options
  const [quality, setQuality] = useState<ExportQuality>(
    isPaidTier ? 'original' : '720p'
  );
  const [applyCameraEdits, setApplyCameraEdits] = useState(true);
  const [withFade, setWithFade] = useState(false);

  // Reset state when dialog opens - setState in effect is intentional for dialog reset pattern
  /* eslint-disable react-hooks/set-state-in-effect */
  useEffect(() => {
    if (open) {
      setQuality(isPaidTier ? 'original' : '720p');
      setApplyCameraEdits(true);
      setWithFade(false);
    }
  }, [open, isPaidTier]);
  /* eslint-enable react-hooks/set-state-in-effect */

  // Camera edits disabled when any rally has vertical (9:16) edits
  const cameraEditsDisabled = hasVerticalEdits;
  const cameraEditsTooltip = hasVerticalEdits
    ? 'Camera effects cannot be applied when rallies have 9:16 edits'
    : '';

  const handleExport = () => {
    onExport({
      quality,
      applyCameraEdits: hasCameraEdits && applyCameraEdits && !cameraEditsDisabled,
      withFade: showFadeOption && withFade,
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
        {/* Fade only applies when downloading multiple rallies */}
        {(hasCameraEdits || (showFadeOption && rallies.length > 1)) && (
          <Box>
            <Typography variant="subtitle2" sx={{ mb: 1 }}>
              Options
            </Typography>

            {/* Camera Edits Toggle */}
            {hasCameraEdits && (
              <Tooltip title={cameraEditsTooltip} placement="top">
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={applyCameraEdits && !cameraEditsDisabled}
                      onChange={(e) => setApplyCameraEdits(e.target.checked)}
                      disabled={cameraEditsDisabled}
                      size="small"
                    />
                  }
                  label="Apply camera effects"
                  sx={{ mb: showFadeOption && rallies.length > 1 ? 0.5 : 0 }}
                />
              </Tooltip>
            )}

            {/* Fade Toggle - only for multiple rallies */}
            {showFadeOption && rallies.length > 1 && (
              <FormControlLabel
                control={
                  <Checkbox
                    checked={withFade}
                    onChange={(e) => setWithFade(e.target.checked)}
                    size="small"
                  />
                }
                label="Add fade between rallies (0.5s)"
              />
            )}
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
