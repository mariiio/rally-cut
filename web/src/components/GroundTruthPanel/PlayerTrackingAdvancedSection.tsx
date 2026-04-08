'use client';

import { ReactNode, useEffect, useState } from 'react';
import { Box, Button, Dialog, DialogActions, DialogContent, DialogTitle, Stack, TextField, Typography } from '@mui/material';
import EditIcon from '@mui/icons-material/Edit';
import OpenInNewIcon from '@mui/icons-material/OpenInNew';
import SaveIcon from '@mui/icons-material/Save';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import PersonSearchIcon from '@mui/icons-material/PersonSearch';
import PeopleIcon from '@mui/icons-material/People';
import { usePlayerTrackingStore } from '@/stores/playerTrackingStore';
import { useEditorStore } from '@/stores/editorStore';
import {
  API_BASE_URL,
  exportToLabelStudio,
  getLabelStudioStatus,
  importFromLabelStudio,
} from '@/services/api';

interface Props {
  renderSection: (body: ReactNode) => ReactNode;
  onOpenPlayerMatching?: () => void;
  onOpenReferenceCrops?: () => void;
}

export function PlayerTrackingAdvancedSection({
  renderSection,
  onOpenPlayerMatching,
  onOpenReferenceCrops,
}: Props) {
  const selectedRallyId = useEditorStore((s) => s.selectedRallyId);
  const rallies = useEditorStore((s) => s.rallies);
  const getActiveMatch = useEditorStore((s) => s.getActiveMatch);
  const activeMatch = getActiveMatch();
  const selectedRally = rallies.find((r) => r.id === selectedRallyId);
  const backendRallyId = selectedRally?._backendId ?? null;

  const playerTracks = usePlayerTrackingStore((s) => s.playerTracks);
  const hasTrackingData = backendRallyId
    ? !!playerTracks[backendRallyId]?.tracksJson?.tracks?.length
    : false;

  const [labelStudioLoading, setLabelStudioLoading] = useState(false);
  const [hasGroundTruth, setHasGroundTruth] = useState(false);
  const [labelStudioTaskId, setLabelStudioTaskId] = useState<number | null>(null);
  const [showImportDialog, setShowImportDialog] = useState(false);
  const [importTaskId, setImportTaskId] = useState('');

  useEffect(() => {
    const checkStatus = async () => {
      if (!backendRallyId || !hasTrackingData) {
        setHasGroundTruth(false);
        setLabelStudioTaskId(null);
        return;
      }
      try {
        const status = await getLabelStudioStatus(backendRallyId);
        setHasGroundTruth(status.hasGroundTruth);
        setLabelStudioTaskId(status.taskId ?? null);
      } catch (error) {
        console.error('Failed to get Label Studio status:', error);
      }
    };
    void checkStatus();
  }, [backendRallyId, hasTrackingData]);

  const handleOpenLabelStudio = async (forceRegenerate: boolean) => {
    if (!backendRallyId || !activeMatch?.videoUrl) return;
    setLabelStudioLoading(true);
    try {
      const videoUrl = new URL(activeMatch.videoUrl, API_BASE_URL).href;
      const opts = forceRegenerate ? { forceRegenerate: true as const } : undefined;
      const result = await exportToLabelStudio(backendRallyId, videoUrl, opts);
      if (result.success && result.taskUrl) {
        setLabelStudioTaskId(result.taskId ?? null);
        window.open(result.taskUrl, '_blank');
      } else {
        alert(result.error || 'Failed to open Label Studio');
      }
    } catch (error) {
      alert(error instanceof Error ? error.message : 'Failed to open Label Studio');
    } finally {
      setLabelStudioLoading(false);
    }
  };

  const handleSaveGroundTruth = async () => {
    if (!backendRallyId) return;
    if (labelStudioTaskId) {
      setLabelStudioLoading(true);
      try {
        const result = await importFromLabelStudio(backendRallyId, labelStudioTaskId);
        if (result.success) {
          setHasGroundTruth(true);
          alert(
            `Ground truth saved! ${result.playerCount} player annotations, ${result.ballCount} ball annotations across ${result.frameCount} frames.`,
          );
        } else {
          alert(result.error || 'Failed to save ground truth');
        }
      } catch (error) {
        alert(error instanceof Error ? error.message : 'Failed to save ground truth');
      } finally {
        setLabelStudioLoading(false);
      }
    } else {
      setShowImportDialog(true);
    }
  };

  const handleImportSubmit = async () => {
    if (!backendRallyId || !importTaskId) return;
    const taskId = parseInt(importTaskId, 10);
    if (isNaN(taskId) || taskId <= 0) {
      alert('Please enter a valid task ID');
      return;
    }
    setShowImportDialog(false);
    setLabelStudioLoading(true);
    try {
      const result = await importFromLabelStudio(backendRallyId, taskId);
      if (result.success) {
        setHasGroundTruth(true);
        setLabelStudioTaskId(taskId);
        setImportTaskId('');
        alert(
          `Ground truth saved! ${result.playerCount} player annotations, ${result.ballCount} ball annotations across ${result.frameCount} frames.`,
        );
      } else {
        alert(result.error || 'Failed to save ground truth');
      }
    } catch (error) {
      alert(error instanceof Error ? error.message : 'Failed to save ground truth');
    } finally {
      setLabelStudioLoading(false);
    }
  };

  const body = (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
      <Typography variant="caption" color="text.secondary">
        Advanced ground-truth tools for player tracking.
      </Typography>
      <Stack direction="column" spacing={0.75}>
        <Button
          size="small"
          variant="outlined"
          startIcon={<EditIcon />}
          onClick={() => handleOpenLabelStudio(true)}
          disabled={labelStudioLoading || !backendRallyId || !hasTrackingData}
        >
          Export to Label Studio
        </Button>
        <Button
          size="small"
          variant="outlined"
          startIcon={<OpenInNewIcon />}
          onClick={() => handleOpenLabelStudio(false)}
          disabled={labelStudioLoading || !labelStudioTaskId}
        >
          Resume Label Studio
        </Button>
        <Button
          size="small"
          variant="outlined"
          startIcon={
            hasGroundTruth ? <CheckCircleIcon color="success" /> : <SaveIcon />
          }
          onClick={handleSaveGroundTruth}
          disabled={labelStudioLoading || !backendRallyId}
        >
          {hasGroundTruth ? 'GT Saved' : 'Save GT from Label Studio'}
        </Button>
        {onOpenPlayerMatching && (
          <Button
            size="small"
            variant="outlined"
            startIcon={<PeopleIcon />}
            onClick={onOpenPlayerMatching}
          >
            Player matching…
          </Button>
        )}
        {onOpenReferenceCrops && (
          <Button
            size="small"
            variant="outlined"
            startIcon={<PersonSearchIcon />}
            onClick={onOpenReferenceCrops}
          >
            Reference crops…
          </Button>
        )}
      </Stack>

      <Dialog open={showImportDialog} onClose={() => setShowImportDialog(false)}>
        <DialogTitle>Import Ground Truth from Label Studio</DialogTitle>
        <DialogContent>
          <Typography variant="body2" sx={{ mb: 2 }}>
            Enter the Label Studio task ID to import corrected annotations.
          </Typography>
          <TextField
            autoFocus
            label="Task ID"
            type="number"
            fullWidth
            value={importTaskId}
            onChange={(e) => setImportTaskId(e.target.value)}
            helperText="Find the task ID in the Label Studio URL (e.g., .../task=123)"
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowImportDialog(false)}>Cancel</Button>
          <Button onClick={handleImportSubmit} variant="contained" disabled={!importTaskId}>
            Import
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );

  return <>{renderSection(body)}</>;
}
