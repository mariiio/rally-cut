'use client';

import { useRef, useState } from 'react';
import {
  Button,
  Stack,
  Snackbar,
  Alert,
  IconButton,
  Tooltip,
  Divider,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
} from '@mui/material';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import FileOpenIcon from '@mui/icons-material/FileOpen';
import DownloadIcon from '@mui/icons-material/Download';
import UndoIcon from '@mui/icons-material/Undo';
import RedoIcon from '@mui/icons-material/Redo';
import RestoreIcon from '@mui/icons-material/Restore';
import { useEditorStore } from '@/stores/editorStore';
import {
  parseSegmentJson,
  downloadSegmentJson,
  isValidVideoFile,
  isJsonFile,
} from '@/utils/fileHandlers';

export function FileControls() {
  const videoInputRef = useRef<HTMLInputElement>(null);
  const jsonInputRef = useRef<HTMLInputElement>(null);
  const [error, setError] = useState<string | null>(null);
  const [resetDialogOpen, setResetDialogOpen] = useState(false);

  const {
    setVideoFile,
    loadSegmentsFromJson,
    exportToJson,
    segments,
    undo,
    redo,
    resetToOriginal,
    canUndo,
    canRedo,
    hasChangesFromOriginal,
  } = useEditorStore();

  const handleVideoUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!isValidVideoFile(file)) {
      setError('Invalid video format. Please use MP4, MOV, or WebM.');
      return;
    }

    setVideoFile(file);
    // Reset input so same file can be selected again
    e.target.value = '';
  };

  const handleJsonUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!isJsonFile(file)) {
      setError('Please select a JSON file.');
      return;
    }

    try {
      const json = await parseSegmentJson(file);
      loadSegmentsFromJson(json);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to parse JSON');
    }

    // Reset input so same file can be selected again
    e.target.value = '';
  };

  const handleExport = () => {
    const data = exportToJson();
    if (data) {
      downloadSegmentJson(data, 'segments_edited.json');
    } else {
      setError('No data to export. Load a segments JSON first.');
    }
  };

  const handleResetConfirm = () => {
    resetToOriginal();
    setResetDialogOpen(false);
  };

  const isMac = typeof navigator !== 'undefined' && navigator.platform.includes('Mac');
  const modKey = isMac ? '⌘' : 'Ctrl';

  return (
    <>
      <Stack direction="row" spacing={1} alignItems="center">
        <input
          type="file"
          ref={videoInputRef}
          onChange={handleVideoUpload}
          accept="video/mp4,video/quicktime,video/webm,.mp4,.mov,.webm"
          style={{ display: 'none' }}
        />
        <Button
          variant="outlined"
          startIcon={<UploadFileIcon />}
          onClick={() => videoInputRef.current?.click()}
          size="small"
        >
          Upload Video
        </Button>

        <input
          type="file"
          ref={jsonInputRef}
          onChange={handleJsonUpload}
          accept="application/json,.json"
          style={{ display: 'none' }}
        />
        <Button
          variant="outlined"
          startIcon={<FileOpenIcon />}
          onClick={() => jsonInputRef.current?.click()}
          size="small"
        >
          Load JSON
        </Button>

        <Button
          variant="contained"
          startIcon={<DownloadIcon />}
          onClick={handleExport}
          size="small"
          disabled={segments.length === 0}
        >
          Export JSON
        </Button>

        <Divider orientation="vertical" flexItem sx={{ mx: 0.5 }} />

        <Tooltip title={`Undo (${modKey}+Z)`}>
          <span>
            <IconButton
              size="small"
              onClick={undo}
              disabled={!canUndo()}
              sx={{ color: 'text.secondary' }}
            >
              <UndoIcon fontSize="small" />
            </IconButton>
          </span>
        </Tooltip>

        <Tooltip title={`Redo (${modKey}+Shift+Z)`}>
          <span>
            <IconButton
              size="small"
              onClick={redo}
              disabled={!canRedo()}
              sx={{ color: 'text.secondary' }}
            >
              <RedoIcon fontSize="small" />
            </IconButton>
          </span>
        </Tooltip>

        <Tooltip title="Reset all changes">
          <span>
            <IconButton
              size="small"
              onClick={() => setResetDialogOpen(true)}
              disabled={!hasChangesFromOriginal()}
              sx={{ color: 'text.secondary' }}
            >
              <RestoreIcon fontSize="small" />
            </IconButton>
          </span>
        </Tooltip>
      </Stack>

      {/* Reset Confirmation Dialog */}
      <Dialog
        open={resetDialogOpen}
        onClose={() => setResetDialogOpen(false)}
        maxWidth="xs"
      >
        <DialogTitle>Reset All Changes?</DialogTitle>
        <DialogContent>
          <DialogContentText>
            This will restore all segments to their original state. This action
            cannot be undone.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setResetDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleResetConfirm} color="error" variant="contained">
            Reset
          </Button>
        </DialogActions>
      </Dialog>

      <Snackbar
        open={!!error}
        autoHideDuration={5000}
        onClose={() => setError(null)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert severity="error" onClose={() => setError(null)}>
          {error}
        </Alert>
      </Snackbar>
    </>
  );
}
