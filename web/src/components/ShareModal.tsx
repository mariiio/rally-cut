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
  Stack,
  Tooltip,
  Select,
  MenuItem,
  Chip,
  type SelectChangeEvent,
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import CheckIcon from '@mui/icons-material/Check';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import LinkIcon from '@mui/icons-material/Link';
import PersonIcon from '@mui/icons-material/Person';
import DeleteIcon from '@mui/icons-material/Delete';
import {
  createShare,
  getShare,
  deleteShare,
  removeShareMember,
  updateMemberRole,
  updateDefaultRole,
  type ShareInfo,
  type MemberRole,
} from '@/services/api';
import { useEditorStore } from '@/stores/editorStore';

const ROLE_LABELS: Record<MemberRole, string> = {
  VIEWER: 'Viewer',
  EDITOR: 'Editor',
  ADMIN: 'Admin',
};

const ROLE_DESCRIPTIONS: Record<MemberRole, string> = {
  VIEWER: 'Can view rallies and highlights',
  EDITOR: 'Can edit rallies and create highlights',
  ADMIN: 'Can manage members and share settings',
};

const ROLE_COLORS: Record<MemberRole, 'default' | 'info' | 'warning'> = {
  VIEWER: 'default',
  EDITOR: 'info',
  ADMIN: 'warning',
};

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
  const [confirmingRemove, setConfirmingRemove] = useState<string | null>(null);
  const [confirmingDelete, setConfirmingDelete] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [updatingRole, setUpdatingRole] = useState<string | null>(null);

  const userRole = useEditorStore((s) => s.userRole);
  const isOwner = userRole === 'owner';

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
          defaultRole: created.defaultRole,
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
      setRemovingMember(null);
      setConfirmingRemove(null);
      setConfirmingDelete(false);
      setDeleting(false);
      setUpdatingRole(null);
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

  const handleRemoveClick = (userId: string) => {
    if (confirmingRemove === userId) {
      // Second click - actually remove
      handleRemoveMember(userId);
    } else {
      // First click - show confirmation
      setConfirmingRemove(userId);
    }
  };

  const handleRemoveMember = async (userId: string) => {
    try {
      setConfirmingRemove(null);
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

  // Auto-cancel confirmations after 3 seconds
  useEffect(() => {
    if (!confirmingRemove && !confirmingDelete) return;
    const timeout = setTimeout(() => {
      setConfirmingRemove(null);
      setConfirmingDelete(false);
    }, 3000);
    return () => clearTimeout(timeout);
  }, [confirmingRemove, confirmingDelete]);

  const handleDeleteClick = () => {
    if (confirmingDelete) {
      handleDeleteShare();
    } else {
      setConfirmingDelete(true);
    }
  };

  const handleDeleteShare = async () => {
    try {
      setConfirmingDelete(false);
      setDeleting(true);
      await deleteShare(sessionId);
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete share');
      setDeleting(false);
    }
  };

  const handleDefaultRoleChange = async (event: SelectChangeEvent) => {
    const newRole = event.target.value as MemberRole;
    try {
      await updateDefaultRole(sessionId, newRole);
      setShareInfo((prev) => prev ? { ...prev, defaultRole: newRole } : prev);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update default role');
    }
  };

  const handleMemberRoleChange = async (userId: string, newRole: MemberRole) => {
    try {
      setUpdatingRole(userId);
      await updateMemberRole(sessionId, userId, newRole);
      setShareInfo((prev) => {
        if (!prev) return prev;
        return {
          ...prev,
          members: prev.members.map((m) =>
            m.userId === userId ? { ...m, role: newRole } : m
          ),
        };
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update member role');
    } finally {
      setUpdatingRole(null);
    }
  };

  const defaultRoleDescription = shareInfo
    ? `Anyone with this link will join as ${ROLE_LABELS[shareInfo.defaultRole].toLowerCase()}.`
    : '';

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

              {/* Default role selector */}
              <Stack direction="row" spacing={1} alignItems="center" sx={{ mt: 1.5 }}>
                <Typography variant="caption" color="text.secondary">
                  New members join as:
                </Typography>
                <Select
                  value={shareInfo.defaultRole}
                  onChange={handleDefaultRoleChange}
                  size="small"
                  variant="outlined"
                  sx={{ minWidth: 100, '& .MuiSelect-select': { py: 0.25, fontSize: '0.75rem' } }}
                >
                  <MenuItem value="VIEWER">{ROLE_LABELS.VIEWER}</MenuItem>
                  <MenuItem value="EDITOR">{ROLE_LABELS.EDITOR}</MenuItem>
                  <MenuItem value="ADMIN">{ROLE_LABELS.ADMIN}</MenuItem>
                </Select>
              </Stack>
              <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block' }}>
                {defaultRoleDescription}
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
                  {shareInfo.members.map((member, index) => {
                    // Admin actors can't change other admins
                    const canChangeRole = isOwner || (userRole === 'ADMIN' && member.role !== 'ADMIN');
                    // Only show admin option if actor is owner
                    const availableRoles: MemberRole[] = isOwner
                      ? ['VIEWER', 'EDITOR', 'ADMIN']
                      : ['VIEWER', 'EDITOR'];

                    return (
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
                          primary={
                            <Stack direction="row" spacing={1} alignItems="center">
                              <span>{member.name || 'Anonymous'}</span>
                              <Chip
                                label={ROLE_LABELS[member.role]}
                                size="small"
                                color={ROLE_COLORS[member.role]}
                                sx={{ height: 20, fontSize: '0.7rem' }}
                              />
                            </Stack>
                          }
                          secondary={`Joined ${new Date(member.joinedAt).toLocaleDateString()}`}
                        />
                        <ListItemSecondaryAction>
                          <Stack direction="row" spacing={0.5} alignItems="center">
                            {/* Role selector */}
                            {canChangeRole && (
                              <Select
                                value={member.role}
                                onChange={(e) => handleMemberRoleChange(member.userId, e.target.value as MemberRole)}
                                size="small"
                                variant="outlined"
                                disabled={updatingRole === member.userId}
                                sx={{ minWidth: 90, '& .MuiSelect-select': { py: 0.25, fontSize: '0.75rem' } }}
                              >
                                {availableRoles.map((role) => (
                                  <MenuItem key={role} value={role}>
                                    <Tooltip title={ROLE_DESCRIPTIONS[role]} placement="left">
                                      <span>{ROLE_LABELS[role]}</span>
                                    </Tooltip>
                                  </MenuItem>
                                ))}
                              </Select>
                            )}

                            {/* Remove button */}
                            {removingMember === member.userId ? (
                              <CircularProgress size={20} />
                            ) : confirmingRemove === member.userId ? (
                              <Stack direction="row" spacing={0.25}>
                                <Tooltip title="Confirm remove">
                                  <IconButton
                                    size="small"
                                    onClick={() => handleRemoveClick(member.userId)}
                                    sx={{
                                      color: 'white',
                                      bgcolor: 'error.main',
                                      width: 24,
                                      height: 24,
                                      '&:hover': { bgcolor: 'error.light' },
                                    }}
                                  >
                                    <CheckIcon sx={{ fontSize: 14 }} />
                                  </IconButton>
                                </Tooltip>
                                <Tooltip title="Cancel">
                                  <IconButton
                                    size="small"
                                    onClick={() => setConfirmingRemove(null)}
                                    sx={{ width: 24, height: 24 }}
                                  >
                                    <CloseIcon sx={{ fontSize: 14 }} />
                                  </IconButton>
                                </Tooltip>
                              </Stack>
                            ) : (
                              <IconButton
                                edge="end"
                                onClick={() => handleRemoveClick(member.userId)}
                                size="small"
                              >
                                <DeleteIcon />
                              </IconButton>
                            )}
                          </Stack>
                        </ListItemSecondaryAction>
                      </ListItem>
                    );
                  })}
                </List>
              )}
            </Box>
          </>
        ) : null}
      </DialogContent>

      <DialogActions sx={{ px: 3, pb: 2, justifyContent: 'space-between' }}>
        {/* Only owner can delete share */}
        {isOwner ? (
          deleting ? (
            <Button color="error" disabled startIcon={<CircularProgress size={16} />}>
              Deleting...
            </Button>
          ) : confirmingDelete ? (
            <Stack direction="row" spacing={1} alignItems="center">
              <Typography variant="body2" color="text.secondary">
                Revoke all access?
              </Typography>
              <Tooltip title="Confirm delete">
                <IconButton
                  size="small"
                  onClick={handleDeleteClick}
                  sx={{
                    color: 'white',
                    bgcolor: 'error.main',
                    width: 28,
                    height: 28,
                    '&:hover': { bgcolor: 'error.light' },
                  }}
                >
                  <CheckIcon sx={{ fontSize: 16 }} />
                </IconButton>
              </Tooltip>
              <Tooltip title="Cancel">
                <IconButton
                  size="small"
                  onClick={() => setConfirmingDelete(false)}
                  sx={{ width: 28, height: 28 }}
                >
                  <CloseIcon sx={{ fontSize: 16 }} />
                </IconButton>
              </Tooltip>
            </Stack>
          ) : (
            <Button
              color="error"
              onClick={handleDeleteClick}
              disabled={loading}
              startIcon={<DeleteIcon />}
            >
              Delete Share
            </Button>
          )
        ) : (
          <Box /> // spacer for non-owners
        )}
        <Button onClick={onClose}>Close</Button>
      </DialogActions>
    </Dialog>
  );
}
