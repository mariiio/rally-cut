'use client';

import { useRef, useState } from 'react';
import {
  Box,
  Typography,
  IconButton,
  Button,
  Stack,
  Tooltip,
  Divider,
  Menu,
  MenuItem,
  ListItemIcon,
  ListItemText,
  Select,
  SelectChangeEvent,
  Snackbar,
  Alert,
} from '@mui/material';
import SportsVolleyballIcon from '@mui/icons-material/SportsVolleyball';
import UndoIcon from '@mui/icons-material/Undo';
import RedoIcon from '@mui/icons-material/Redo';
import DownloadIcon from '@mui/icons-material/Download';
import FileOpenIcon from '@mui/icons-material/FileOpen';
import RestoreIcon from '@mui/icons-material/Restore';
import KeyboardArrowDownIcon from '@mui/icons-material/KeyboardArrowDown';
import HomeIcon from '@mui/icons-material/Home';
import { useEditorStore } from '@/stores/editorStore';
import { useUploadStore } from '@/stores/uploadStore';
import {
  parseRallyJson,
  downloadRallyJson,
  isValidVideoFile,
  isJsonFile,
} from '@/utils/fileHandlers';
import { designTokens } from '@/app/theme';
import { ConfirmDialog } from './ConfirmDialog';
import { SyncStatus } from './SyncStatus';

export function EditorHeader() {
  const videoInputRef = useRef<HTMLInputElement>(null);
  const jsonInputRef = useRef<HTMLInputElement>(null);
  const [error, setError] = useState<string | null>(null);
  const [exportAnchorEl, setExportAnchorEl] = useState<null | HTMLElement>(null);
  const [showResetDialog, setShowResetDialog] = useState(false);

  const {
    session,
    activeMatchId,
    setActiveMatch,
    loadRalliesFromJson,
    exportToJson,
    rallies,
    undo,
    redo,
    resetToOriginal,
    canUndo,
    canRedo,
    hasChangesFromOriginal,
  } = useEditorStore();

  const { isUploading, uploadVideo } = useUploadStore();

  const handleVideoUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!isValidVideoFile(file)) {
      setError('Invalid video format. Please use MP4, MOV, or WebM.');
      return;
    }

    if (!session) {
      setError('No session loaded. Please create or open a session first.');
      return;
    }

    e.target.value = '';

    const success = await uploadVideo(session.id, file);
    if (success) {
      // Reload page to fetch updated session with new video
      window.location.reload();
    }
  };

  // Expose upload trigger for VideoPlayer empty state
  const triggerVideoUpload = () => {
    if (!isUploading) {
      videoInputRef.current?.click();
    }
  };

  // Store the trigger function globally so VideoPlayer can access it
  if (typeof window !== 'undefined') {
    (window as unknown as { triggerVideoUpload?: () => void }).triggerVideoUpload = triggerVideoUpload;
  }

  const handleJsonUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!isJsonFile(file)) {
      setError('Please select a JSON file.');
      return;
    }

    try {
      const json = await parseRallyJson(file);
      loadRalliesFromJson(json);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to parse JSON');
    }

    e.target.value = '';
  };

  const handleExport = () => {
    const data = exportToJson();
    if (data) {
      downloadRallyJson(data, 'rallies_edited.json');
    } else {
      setError('No data to export. Load a rallies JSON first.');
    }
    setExportAnchorEl(null);
  };

  const handleMatchChange = (event: SelectChangeEvent<string>) => {
    setActiveMatch(event.target.value);
  };

  const isMac = typeof navigator !== 'undefined' && navigator.platform.includes('Mac');
  const modKey = isMac ? '⌘' : 'Ctrl';

  return (
    <>
      <Box
        component="header"
        sx={{
          height: designTokens.spacing.header,
          display: 'flex',
          alignItems: 'center',
          px: 2,
          gap: 2,
          bgcolor: 'background.paper',
          borderBottom: '1px solid',
          borderColor: 'divider',
          backgroundImage: designTokens.gradients.toolbar,
          flexShrink: 0,
        }}
      >
        {/* Brand */}
        <Stack direction="row" alignItems="center" spacing={1} sx={{ minWidth: 'fit-content' }}>
          <SportsVolleyballIcon
            sx={{
              fontSize: 28,
              color: 'primary.main',
              filter: 'drop-shadow(0 2px 4px rgba(255, 107, 74, 0.3))',
            }}
          />
          <Typography
            variant="h6"
            sx={{
              fontWeight: 700,
              background: designTokens.gradients.primary,
              backgroundClip: 'text',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              letterSpacing: '-0.02em',
            }}
          >
            RallyCut
          </Typography>
        </Stack>

        <Divider orientation="vertical" flexItem sx={{ mx: 1 }} />

        {/* Session/Match Context */}
        <Stack direction="row" alignItems="center" spacing={2} sx={{ flex: 1, minWidth: 0 }}>
          {session ? (
            <>
              {session.matches.length > 1 && (
                <Select
                  value={activeMatchId || ''}
                  onChange={handleMatchChange}
                  size="small"
                  displayEmpty
                  sx={{
                    minWidth: 180,
                    '& .MuiSelect-select': {
                      py: 0.75,
                      fontSize: '0.875rem',
                    },
                  }}
                >
                  {session.matches.map((match) => (
                    <MenuItem key={match.id} value={match.id}>
                      {match.name}
                    </MenuItem>
                  ))}
                </Select>
              )}

              {session.matches.length === 1 && (
                <Typography
                  variant="body2"
                  sx={{
                    color: 'text.primary',
                    fontWeight: 500,
                  }}
                >
                  {session.matches[0].name}
                </Typography>
              )}

              {/* Sync Status Indicator */}
              <SyncStatus />
            </>
          ) : (
            <Typography variant="body2" sx={{ color: 'text.disabled' }}>
              No session loaded
            </Typography>
          )}
        </Stack>

        {/* History Controls */}
        <Stack direction="row" spacing={0.5} alignItems="center">
          <Tooltip title={`Undo (${modKey}+Z)`}>
            <span>
              <IconButton
                size="small"
                onClick={undo}
                disabled={!canUndo()}
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
              >
                <RedoIcon fontSize="small" />
              </IconButton>
            </span>
          </Tooltip>

          <Tooltip title="Reset all changes">
            <span>
              <IconButton
                size="small"
                onClick={() => setShowResetDialog(true)}
                disabled={!hasChangesFromOriginal()}
              >
                <RestoreIcon fontSize="small" />
              </IconButton>
            </span>
          </Tooltip>
        </Stack>

        <Divider orientation="vertical" flexItem sx={{ mx: 1 }} />

        {/* Actions */}
        <Stack direction="row" spacing={1} alignItems="center">
          {/* Export Button */}
          <Button
            variant="outlined"
            size="small"
            onClick={(e) => setExportAnchorEl(e.currentTarget)}
            disabled={!rallies || rallies.length === 0}
            startIcon={<DownloadIcon />}
            endIcon={<KeyboardArrowDownIcon />}
            sx={{ minWidth: 100 }}
          >
            Export
          </Button>
          <Menu
            anchorEl={exportAnchorEl}
            open={Boolean(exportAnchorEl)}
            onClose={() => setExportAnchorEl(null)}
            anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
            transformOrigin={{ vertical: 'top', horizontal: 'right' }}
          >
            <MenuItem onClick={handleExport}>
              <ListItemIcon>
                <FileOpenIcon fontSize="small" />
              </ListItemIcon>
              <ListItemText>Export JSON</ListItemText>
            </MenuItem>
          </Menu>

          {/* Home Button */}
          <Tooltip title="Back to sessions">
            <IconButton
              size="small"
              onClick={() => window.location.href = '/'}
              sx={{ color: 'text.secondary' }}
            >
              <HomeIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </Stack>
      </Box>

      {/* Hidden file inputs */}
      <input
        type="file"
        ref={videoInputRef}
        onChange={handleVideoUpload}
        accept="video/mp4,video/quicktime,video/webm,.mp4,.mov,.webm"
        style={{ display: 'none' }}
      />
      <input
        type="file"
        ref={jsonInputRef}
        onChange={handleJsonUpload}
        accept="application/json,.json"
        style={{ display: 'none' }}
      />

      {/* Error Snackbar */}
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

      {/* Reset confirmation dialog */}
      <ConfirmDialog
        open={showResetDialog}
        title="Reset all changes?"
        message="This will discard all your edits and restore the original rally data. This action cannot be undone."
        confirmLabel="Reset"
        cancelLabel="Keep editing"
        onConfirm={resetToOriginal}
        onCancel={() => setShowResetDialog(false)}
      />
    </>
  );
}
