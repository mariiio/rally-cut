'use client';

import { useState, useEffect, useCallback } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Box,
  Typography,
  TextField,
  InputAdornment,
  IconButton,
  CircularProgress,
  List,
  ListItem,
  ListItemText,
  ListItemAvatar,
  ListItemSecondaryAction,
  Avatar,
  Alert,
  Divider,
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import LinkIcon from '@mui/icons-material/Link';
import PersonIcon from '@mui/icons-material/Person';
import DeleteIcon from '@mui/icons-material/Delete';
import {
  createShare,
  getShare,
  deleteShare,
  removeShareMember,
  type ShareInfo,
} from '@/services/api';

interface ShareModalProps {
  open: boolean;
  onClose: () => void;
  sessionId: string;
  sessionName: string;
}

export function ShareModal({ open, onClose, sessionId, sessionName }: ShareModalProps) {
  const [loading, setLoading] = useState(true);
  const [shareInfo, setShareInfo] = useState<ShareInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [removingMember, setRemovingMember] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);

  const shareUrl = shareInfo
    ? `${typeof window !== 'undefined' ? window.location.origin : ''}/share/${shareInfo.token}`
    : '';

  const loadOrCreateShare = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      // First try to get existing share
      let share = await getShare(sessionId);

      // If no share exists, create one
      if (!share) {
        const created = await createShare(sessionId);
        share = {
          token: created.token,
          createdAt: created.createdAt,
          members: [],
        };
      }

      setShareInfo(share);
    } catch (err) {
      console.error('Failed to load share:', err);
      setError(err instanceof Error ? err.message : 'Failed to load share');
    } finally {
      setLoading(false);
    }
  }, [sessionId]);

  useEffect(() => {
    if (open) {
      loadOrCreateShare();
    }
  }, [open, loadOrCreateShare]);

  useEffect(() => {
    if (!open) {
      setShareInfo(null);
      setError(null);
      setCopied(false);
    }
  }, [open]);

  const handleCopyLink = async () => {
    try {
      await navigator.clipboard.writeText(shareUrl);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      setError('Failed to copy link');
    }
  };

  const handleRemoveMember = async (userId: string) => {
    try {
      setRemovingMember(userId);
      await removeShareMember(sessionId, userId);
      // Refresh share info
      const updated = await getShare(sessionId);
      setShareInfo(updated);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to remove member');
    } finally {
      setRemovingMember(null);
    }
  };

  const handleDeleteShare = async () => {
    if (!confirm('This will revoke access for all members. Are you sure?')) {
      return;
    }

    try {
      setDeleting(true);
      await deleteShare(sessionId);
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete share');
      setDeleting(false);
    }
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        Share &ldquo;{sessionName}&rdquo;
        <IconButton onClick={onClose} size="small">
          <CloseIcon />
        </IconButton>
      </DialogTitle>

      <DialogContent>
        {error && (
          <Alert severity="error" onClose={() => setError(null)} sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
            <CircularProgress />
          </Box>
        ) : shareInfo ? (
          <>
            <Box sx={{ mb: 3 }}>
              <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                Share link
              </Typography>
              <TextField
                fullWidth
                value={shareUrl}
                slotProps={{
                  input: {
                    readOnly: true,
                    startAdornment: (
                      <InputAdornment position="start">
                        <LinkIcon sx={{ color: 'grey.500' }} />
                      </InputAdornment>
                    ),
                    endAdornment: (
                      <InputAdornment position="end">
                        <IconButton onClick={handleCopyLink} edge="end">
                          <ContentCopyIcon />
                        </IconButton>
                      </InputAdornment>
                    ),
                  }
                }}
                size="small"
                sx={{
                  '& .MuiOutlinedInput-root': {
                    bgcolor: 'grey.900',
                  },
                }}
              />
              {copied && (
                <Typography variant="caption" color="success.main" sx={{ mt: 0.5, display: 'block' }}>
                  Link copied to clipboard!
                </Typography>
              )}
              <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                Anyone with this link can view the session and create their own highlights.
              </Typography>
            </Box>

            <Divider sx={{ my: 2 }} />

            <Box>
              <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                Members ({shareInfo.members.length})
              </Typography>

              {shareInfo.members.length === 0 ? (
                <Typography color="text.secondary" sx={{ py: 2, textAlign: 'center' }}>
                  No one has joined yet
                </Typography>
              ) : (
                <List sx={{ bgcolor: 'grey.900', borderRadius: 1 }}>
                  {shareInfo.members.map((member, index) => (
                    <ListItem
                      key={member.userId}
                      divider={index < shareInfo.members.length - 1}
                    >
                      <ListItemAvatar>
                        <Avatar sx={{ bgcolor: 'primary.main' }}>
                          {member.name ? member.name[0].toUpperCase() : <PersonIcon />}
                        </Avatar>
                      </ListItemAvatar>
                      <ListItemText
                        primary={member.name || 'Anonymous'}
                        secondary={`Joined ${new Date(member.joinedAt).toLocaleDateString()}`}
                      />
                      <ListItemSecondaryAction>
                        <IconButton
                          edge="end"
                          onClick={() => handleRemoveMember(member.userId)}
                          disabled={removingMember === member.userId}
                          size="small"
                        >
                          {removingMember === member.userId ? (
                            <CircularProgress size={20} />
                          ) : (
                            <DeleteIcon />
                          )}
                        </IconButton>
                      </ListItemSecondaryAction>
                    </ListItem>
                  ))}
                </List>
              )}
            </Box>
          </>
        ) : null}
      </DialogContent>

      <DialogActions sx={{ px: 3, pb: 2, justifyContent: 'space-between' }}>
        <Button
          color="error"
          onClick={handleDeleteShare}
          disabled={deleting || loading}
          startIcon={deleting ? <CircularProgress size={16} /> : <DeleteIcon />}
        >
          Delete Share
        </Button>
        <Button onClick={onClose}>Close</Button>
      </DialogActions>
    </Dialog>
  );
}
