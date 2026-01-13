'use client';

import { useEffect, useRef, useState } from 'react';
import { useRouter } from 'next/navigation';
import {
  Box,
  Typography,
  IconButton,
  Stack,
  Tooltip,
  Divider,
  Snackbar,
  Alert,
  Menu,
  MenuItem,
  ListItemIcon,
  ListItemText,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
} from '@mui/material';
import SportsVolleyballIcon from '@mui/icons-material/SportsVolleyball';
import UndoIcon from '@mui/icons-material/Undo';
import RedoIcon from '@mui/icons-material/Redo';
import RestoreIcon from '@mui/icons-material/Restore';
import AddIcon from '@mui/icons-material/Add';
import ShareIcon from '@mui/icons-material/Share';
import PeopleIcon from '@mui/icons-material/People';
import MoreVertIcon from '@mui/icons-material/MoreVert';
import DeleteIcon from '@mui/icons-material/Delete';
import EditIcon from '@mui/icons-material/Edit';
import ChatBubbleOutlineIcon from '@mui/icons-material/ChatBubbleOutline';
import DiamondIcon from '@mui/icons-material/Diamond';
import Chip from '@mui/material/Chip';
import { useEditorStore } from '@/stores/editorStore';
import { useUploadStore } from '@/stores/uploadStore';
import { useTierStore } from '@/stores/tierStore';
import { isValidVideoFile } from '@/utils/fileHandlers';
import { deleteSession, updateSession } from '@/services/api';
import { designTokens } from '@/app/theme';
import { ConfirmDialog } from './ConfirmDialog';
import { SyncStatus } from './SyncStatus';
import { AddVideoModal } from './AddVideoModal';
import { ShareModal } from './ShareModal';
import { FeedbackModal } from './FeedbackModal';
import { PendingAccessRequests } from './PendingAccessRequests';

export function EditorHeader() {
  const router = useRouter();
  const videoInputRef = useRef<HTMLInputElement>(null);
  const [error, setError] = useState<string | null>(null);
  const [showResetDialog, setShowResetDialog] = useState(false);
  const [showShareModal, setShowShareModal] = useState(false);
  const [showFeedbackModal, setShowFeedbackModal] = useState(false);
  const [menuAnchor, setMenuAnchor] = useState<HTMLElement | null>(null);
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [editingName, setEditingName] = useState(false);
  const [editValue, setEditValue] = useState('');
  const [isRenaming, setIsRenaming] = useState(false);

  const {
    session,
    userRole,
    singleVideoMode,
    undo,
    redo,
    resetToOriginal,
    canUndo,
    canRedo,
    hasChangesFromOriginal,
    reloadSession,
    showAddVideoModal,
    setShowAddVideoModal,
    renameSession,
  } = useEditorStore();

  const { isUploading, uploadVideo } = useUploadStore();
  const isPaidTier = useTierStore((state) => state.isPaidTier());
  const userTier = useTierStore((state) => state.tier);
  const fetchTier = useTierStore((state) => state.fetchTier);

  // Fetch tier on mount
  useEffect(() => {
    fetchTier();
  }, [fetchTier]);

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
      await reloadSession();
    }
  };

  const isMac = typeof navigator !== 'undefined' && navigator.platform.includes('Mac');
  const modKey = isMac ? '⌘' : 'Ctrl';

  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setMenuAnchor(event.currentTarget);
  };

  const handleMenuClose = () => {
    setMenuAnchor(null);
  };

  const handleDeleteClick = () => {
    handleMenuClose();
    setShowDeleteDialog(true);
  };

  const handleDeleteConfirm = async () => {
    if (!session) return;
    try {
      setDeleting(true);
      await deleteSession(session.id);
      setShowDeleteDialog(false);
      router.push('/sessions');
    } catch (err) {
      console.error('Failed to delete session:', err);
      setError('Failed to delete session');
      setDeleting(false);
    }
  };

  const handleDeleteCancel = () => {
    setShowDeleteDialog(false);
  };

  const handleStartEditName = () => {
    if (!session) return;
    setEditValue(session.name);
    setEditingName(true);
  };

  const handleSaveEditName = async () => {
    if (!session || !editValue.trim()) {
      setEditingName(false);
      return;
    }

    const newName = editValue.trim();
    if (newName === session.name) {
      setEditingName(false);
      return;
    }

    setIsRenaming(true);
    try {
      await updateSession(session.id, { name: newName });
      renameSession(newName);
    } catch (err) {
      console.error('Failed to rename session:', err);
      setError('Failed to rename session');
    } finally {
      setIsRenaming(false);
      setEditingName(false);
    }
  };

  const handleCancelEditName = () => {
    setEditingName(false);
    setEditValue('');
  };

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
        {/* Brand - Clickable to go home */}
        <Tooltip title={singleVideoMode ? 'Back to videos' : 'Back to home'}>
          <Stack
            direction="row"
            alignItems="center"
            spacing={1}
            onClick={() => router.push(singleVideoMode ? '/videos' : '/sessions')}
            sx={{
              minWidth: 'fit-content',
              cursor: 'pointer',
              borderRadius: 1,
              px: 1,
              py: 0.5,
              mx: -1,
              transition: 'background-color 0.2s',
              '&:hover': {
                bgcolor: 'action.hover',
              },
            }}
          >
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
        </Tooltip>

        <Divider orientation="vertical" flexItem sx={{ mx: 1 }} />

        {/* Session/Match Context */}
        <Stack direction="row" alignItems="center" spacing={2} sx={{ flex: 1, minWidth: 0 }}>
          {session ? (
            <>
              {/* Session name */}
              {editingName ? (
                <TextField
                  value={editValue}
                  onChange={(e) => setEditValue(e.target.value)}
                  onBlur={handleSaveEditName}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      e.preventDefault();
                      handleSaveEditName();
                    }
                    if (e.key === 'Escape') {
                      handleCancelEditName();
                    }
                  }}
                  autoFocus
                  size="small"
                  disabled={isRenaming}
                  sx={{
                    minWidth: 150,
                    '& .MuiInputBase-input': {
                      py: 0.5,
                      fontSize: '0.875rem',
                      fontWeight: 600,
                    },
                  }}
                />
              ) : (
                <Stack direction="row" alignItems="center">
                  <Typography
                    variant="body2"
                    sx={{
                      color: 'text.primary',
                      fontWeight: 600,
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                    }}
                  >
                    {session.name}
                  </Typography>
                  {userRole === 'owner' && (
                    <Tooltip title="Rename session">
                      <IconButton
                        size="small"
                        onClick={handleStartEditName}
                        sx={{
                          color: 'text.secondary',
                          p: 0.5,
                          ml: 1,
                          '&:hover': { color: 'text.primary' },
                        }}
                      >
                        <EditIcon sx={{ fontSize: 16 }} />
                      </IconButton>
                    </Tooltip>
                  )}
                </Stack>
              )}

              {/* Add Video Button - hide in single video mode */}
              {!singleVideoMode && (
                <Tooltip title="Add video">
                  <span>
                    <IconButton
                      size="small"
                      onClick={() => setShowAddVideoModal(true)}
                      disabled={isUploading}
                      sx={{ color: 'text.secondary' }}
                    >
                      <AddIcon fontSize="small" />
                    </IconButton>
                  </span>
                </Tooltip>
              )}

              {/* Sync Status Indicator */}
              <SyncStatus />

              {/* Tier badge */}
              {isPaidTier && (
                <Chip
                  icon={<DiamondIcon sx={{ fontSize: 14 }} />}
                  label={userTier === 'ELITE' ? 'Elite' : 'Pro'}
                  size="small"
                  variant="outlined"
                  sx={{
                    ml: 1,
                    height: 24,
                    borderColor: 'rgba(255, 209, 102, 0.5)',
                    color: designTokens.colors.tertiary.main,
                    fontWeight: 600,
                    fontSize: '0.75rem',
                    '& .MuiChip-icon': {
                      color: designTokens.colors.tertiary.main,
                    },
                  }}
                />
              )}

              {/* Shared badge for members */}
              {userRole === 'member' && (
                <Chip
                  icon={<PeopleIcon />}
                  label="Shared"
                  size="small"
                  variant="outlined"
                  sx={{ ml: 1 }}
                />
              )}
            </>
          ) : (
            <Typography variant="body2" sx={{ color: 'text.disabled' }}>
              No session loaded
            </Typography>
          )}
        </Stack>

        {/* Share Button (only for owners) */}
        {session && userRole === 'owner' && (
          <Tooltip title="Share session">
            <IconButton
              size="small"
              onClick={() => setShowShareModal(true)}
              sx={{ color: 'text.secondary' }}
            >
              <ShareIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        )}

        {/* Pending Access Requests (only for owners) */}
        {session && userRole === 'owner' && (
          <PendingAccessRequests sessionId={session.id} />
        )}

        {/* Feedback Button */}
        <Tooltip title="Send feedback">
          <IconButton
            size="small"
            onClick={() => setShowFeedbackModal(true)}
            sx={{ color: 'text.secondary' }}
          >
            <ChatBubbleOutlineIcon fontSize="small" />
          </IconButton>
        </Tooltip>

        <Divider orientation="vertical" flexItem sx={{ mx: 1 }} />

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

        {/* Options Menu */}
        <Tooltip title="Options">
          <IconButton
            size="small"
            onClick={handleMenuOpen}
            sx={{ color: 'text.secondary' }}
          >
            <MoreVertIcon fontSize="small" />
          </IconButton>
        </Tooltip>
      </Box>

      {/* Hidden file input */}
      <input
        type="file"
        ref={videoInputRef}
        onChange={handleVideoUpload}
        accept="video/mp4,video/quicktime,video/webm,.mp4,.mov,.webm"
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

      {/* Add Video Modal */}
      {session && (
        <AddVideoModal
          open={showAddVideoModal}
          onClose={() => setShowAddVideoModal(false)}
          sessionId={session.id}
          existingVideoIds={session.matches.map((m) => m.id)}
          onVideoAdded={() => reloadSession()}
        />
      )}

      {/* Share Modal */}
      {session && (
        <ShareModal
          open={showShareModal}
          onClose={() => setShowShareModal(false)}
          sessionId={session.id}
          sessionName={session.name}
        />
      )}

      {/* Feedback Modal */}
      <FeedbackModal
        open={showFeedbackModal}
        onClose={() => setShowFeedbackModal(false)}
      />

      {/* Options Menu */}
      <Menu
        anchorEl={menuAnchor}
        open={Boolean(menuAnchor)}
        onClose={handleMenuClose}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
        transformOrigin={{ vertical: 'top', horizontal: 'right' }}
        slotProps={{
          paper: {
            sx: {
              minWidth: 180,
              bgcolor: designTokens.colors.surface[3],
              border: '1px solid',
              borderColor: 'divider',
            },
          },
        }}
      >
        <MenuItem
          onClick={handleDeleteClick}
          sx={{
            color: 'error.main',
            '&:hover': {
              bgcolor: 'rgba(239, 68, 68, 0.1)',
            },
          }}
        >
          <ListItemIcon>
            <DeleteIcon fontSize="small" sx={{ color: 'error.main' }} />
          </ListItemIcon>
          <ListItemText>Delete session</ListItemText>
        </MenuItem>
      </Menu>

      {/* Delete Session Confirmation Dialog */}
      <Dialog
        open={showDeleteDialog}
        onClose={handleDeleteCancel}
        maxWidth="xs"
        fullWidth
      >
        <DialogTitle sx={{ pb: 1 }}>
          Delete session?
        </DialogTitle>
        <DialogContent>
          <Typography color="text.secondary">
            Are you sure you want to delete <strong>{session?.name}</strong>? This action cannot be undone.
          </Typography>
        </DialogContent>
        <DialogActions sx={{ px: 3, pb: 2 }}>
          <Button onClick={handleDeleteCancel} disabled={deleting}>
            Cancel
          </Button>
          <Button
            onClick={handleDeleteConfirm}
            variant="contained"
            color="error"
            disabled={deleting}
          >
            {deleting ? 'Deleting...' : 'Delete'}
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
}
