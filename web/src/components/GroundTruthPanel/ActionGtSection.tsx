'use client';

import { ReactNode, useMemo } from 'react';
import { Box, Button, Chip, CircularProgress, Typography } from '@mui/material';
import LabelIcon from '@mui/icons-material/Label';
import SaveIcon from '@mui/icons-material/Save';
import { usePlayerTrackingStore } from '@/stores/playerTrackingStore';
import { useEditorStore } from '@/stores/editorStore';

interface Props {
  renderSection: (status: ReactNode, body: ReactNode) => ReactNode;
}

export function ActionGtSection({ renderSection }: Props) {
  const selectedRallyId = useEditorStore((s) => s.selectedRallyId);
  const rallies = useEditorStore((s) => s.rallies);
  const selectedRally = rallies.find((r) => r.id === selectedRallyId);
  const backendRallyId = selectedRally?._backendId ?? null;

  const {
    isLabelingActions,
    setIsLabelingActions,
    actionGroundTruth,
    actionGtDirty,
    actionGtSaving,
    saveActionGroundTruth,
  } = usePlayerTrackingStore();

  const gtLabels = backendRallyId ? actionGroundTruth[backendRallyId] : undefined;
  const gtCount = gtLabels?.length ?? 0;
  const isDirty = backendRallyId ? !!actionGtDirty[backendRallyId] : false;
  const isSaving = backendRallyId ? !!actionGtSaving[backendRallyId] : false;

  // Total rallies in the active match (approximate progress denominator).
  const total = useMemo(() => rallies.length, [rallies.length]);
  const labeledRallies = useMemo(() => {
    let n = 0;
    for (const r of rallies) {
      if (r._backendId && (actionGroundTruth[r._backendId]?.length ?? 0) > 0) n += 1;
    }
    return n;
  }, [rallies, actionGroundTruth]);

  const status = (
    <Chip
      label={`${labeledRallies} / ${total}`}
      size="small"
      sx={{ height: 18, fontSize: '0.65rem' }}
    />
  );

  const handleSave = async () => {
    if (!backendRallyId) return;
    try {
      await saveActionGroundTruth(backendRallyId);
    } catch (error) {
      console.error('Failed to save action GT:', error);
      alert(error instanceof Error ? error.message : 'Failed to save action ground truth');
    }
  };

  const body = (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
      {!selectedRallyId ? (
        <Typography variant="caption" color="text.secondary">
          Select a rally to label actions.
        </Typography>
      ) : (
        <>
          <Typography variant="caption" color="text.secondary">
            {gtCount > 0
              ? `${gtCount} label${gtCount === 1 ? '' : 's'} for this rally${isDirty ? ' (unsaved)' : ''}`
              : 'No action labels yet for this rally.'}
          </Typography>
          <Box sx={{ display: 'flex', gap: 0.75, flexWrap: 'wrap' }}>
            <Button
              size="small"
              variant={isLabelingActions ? 'contained' : 'outlined'}
              startIcon={<LabelIcon />}
              onClick={() => setIsLabelingActions(!isLabelingActions)}
              color={isLabelingActions ? 'warning' : 'primary'}
            >
              {isLabelingActions ? 'Labeling…' : 'Start labeling'}
            </Button>
            <Button
              size="small"
              variant={isDirty ? 'contained' : 'outlined'}
              startIcon={isSaving ? <CircularProgress size={14} /> : <SaveIcon />}
              onClick={handleSave}
              disabled={!isDirty || isSaving || !backendRallyId}
              color={isDirty ? 'primary' : 'success'}
            >
              {isSaving ? 'Saving…' : isDirty ? `Save GT (${gtCount})` : `Saved (${gtCount})`}
            </Button>
          </Box>
        </>
      )}
    </Box>
  );

  return <>{renderSection(status, body)}</>;
}
