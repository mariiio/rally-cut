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
  Paper,
  alpha,
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import CheckIcon from '@mui/icons-material/Check';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import PersonIcon from '@mui/icons-material/Person';
import DeleteIcon from '@mui/icons-material/Delete';
import VisibilityIcon from '@mui/icons-material/Visibility';
import EditIcon from '@mui/icons-material/Edit';
import AdminPanelSettingsIcon from '@mui/icons-material/AdminPanelSettings';
import {
  createShare,
  getShare,
  deleteShare,
  removeShareMember,
  updateMemberRole,
  type ShareInfo,
  type MemberRole,
} from '@/services/api';
import { useEditorStore } from '@/stores/editorStore';

const ROLE_CONFIG: Record<MemberRole, {
  label: string;
  permissions: string[];
  icon: React.ReactNode;
  color: string;
  chipColor: 'default' | 'info' | 'warning';
}> = {
  VIEWER: {
    label: 'Viewer',
    permissions: [
      'Watch videos and rallies',
      'View highlights',
      'Export highlights',
    ],
    icon: <VisibilityIcon />,
    color: '#6B7280',
    chipColor: 'default',
  },
  EDITOR: {
    label: 'Editor',
    permissions: [
      'Everything viewers can do',
      'Edit rally boundaries',
      'Create and manage highlights',
      'Add camera effects',
    ],
    icon: <EditIcon />,
    color: '#3B82F6',
    chipColor: 'info',
  },
  ADMIN: {
    label: 'Admin',
    permissions: [
      'Everything editors can do',
      'Invite and remove members',
      'Change member roles',
      'Delete shared access',
    ],
    icon: <AdminPanelSettingsIcon />,
    color: '#F59E0B',
    chipColor: 'warning',
  },
};

interface ShareModalProps {
  open: boolean;
  onClose: () => void;
  sessionId: string;
  sessionName: string;
}

interface RoleCardProps {
  role: MemberRole;
  selected: boolean;
  onClick: () => void;
}

function RoleCard({ role, selected, onClick }: RoleCardProps) {
  const config = ROLE_CONFIG[role];

  return (
    <Paper
      onClick={onClick}
      elevation={0}
      sx={{
        flex: 1,
        py: 1,
        px: 1.5,
        cursor: 'pointer',
        border: 1.5,
        borderColor: selected ? config.color : 'transparent',
        bgcolor: selected ? alpha(config.color, 0.08) : 'grey.900',
        borderRadius: 1.5,
        transition: 'all 0.15s ease',
        '&:hover': {
          bgcolor: selected ? alpha(config.color, 0.12) : 'grey.800',
          borderColor: selected ? config.color : 'grey.700',
        },
      }}
    >
      <Stack direction="row" spacing={1} alignItems="center" justifyContent="center">
        <Box
          sx={{
            color: selected ? config.color : 'grey.500',
            transition: 'color 0.15s ease',
            '& svg': { fontSize: 18 },
            display: 'flex',
          }}
        >
          {config.icon}
        </Box>
        <Typography
          variant="body2"
          fontWeight={600}
          sx={{ color: selected ? 'text.primary' : 'text.secondary' }}
        >
          {config.label}
        </Typography>
      </Stack>
    </Paper>
  );
}

export function ShareModal({ open, onClose, sessionId, sessionName }: ShareModalProps) {
  const [loading, setLoading] = useState(true);
  const [shareInfo, setShareInfo] = useState<ShareInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [selectedRole, setSelectedRole] = useState<MemberRole>('VIEWER');
  const [removingMember, setRemovingMember] = useState<string | null>(null);
  const [confirmingRemove, setConfirmingRemove] = useState<string | null>(null);
  const [confirmingDelete, setConfirmingDelete] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [updatingRole, setUpdatingRole] = useState<string | null>(null);

  const userRole = useEditorStore((s) => s.userRole);
  const isOwner = userRole === 'owner';

  const selectedShare = shareInfo?.shares.find((s) => s.role === selectedRole);
  const shareUrl = selectedShare
    ? `${typeof window !== 'undefined' ? window.location.origin : ''}/share/${selectedShare.token}`
    : '';

  const loadOrCreateShare = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      let share = await getShare(sessionId);

      if (!share || share.shares.length === 0) {
        const created = await createShare(sessionId);
        share = {
          shares: created.shares,
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
      setSelectedRole('VIEWER');
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

  const handleRoleChange = (role: MemberRole) => {
    setSelectedRole(role);
    setCopied(false);
  };

  const handleRemoveClick = (userId: string) => {
    if (confirmingRemove === userId) {
      handleRemoveMember(userId);
    } else {
      setConfirmingRemove(userId);
    }
  };

  const handleRemoveMember = async (userId: string) => {
    try {
      setConfirmingRemove(null);
      setRemovingMember(userId);
      await removeShareMember(sessionId, userId);
      const updated = await getShare(sessionId);
      setShareInfo(updated);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to remove member');
    } finally {
      setRemovingMember(null);
    }
  };

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

  const selectedConfig = ROLE_CONFIG[selectedRole];

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="sm"
      fullWidth
      PaperProps={{
        sx: {
          borderRadius: 3,
          bgcolor: 'grey.900',
        },
      }}
    >
      <DialogTitle sx={{ pb: 1 }}>
        <Stack direction="row" justifyContent="space-between" alignItems="center">
          <Box>
            <Typography variant="h6" fontWeight={600}>
              Share session
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {sessionName}
            </Typography>
          </Box>
          <IconButton onClick={onClose} size="small" sx={{ color: 'grey.500' }}>
            <CloseIcon />
          </IconButton>
        </Stack>
      </DialogTitle>

      <DialogContent sx={{ pt: 2 }}>
        {error && (
          <Alert severity="error" onClose={() => setError(null)} sx={{ mb: 2, borderRadius: 2 }}>
            {error}
          </Alert>
        )}

        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', py: 6 }}>
            <CircularProgress />
          </Box>
        ) : shareInfo ? (
          <>
            {/* Role Selection */}
            <Box sx={{ mb: 3 }}>
              <Typography variant="overline" color="text.secondary" sx={{ fontWeight: 600, letterSpacing: 1 }}>
                Access level
              </Typography>
              <Stack direction="row" spacing={1.5} sx={{ mt: 1 }}>
                {(['VIEWER', 'EDITOR', 'ADMIN'] as MemberRole[]).map((role) => (
                  <RoleCard
                    key={role}
                    role={role}
                    selected={selectedRole === role}
                    onClick={() => handleRoleChange(role)}
                  />
                ))}
              </Stack>

              {/* Permissions list */}
              <Box
                sx={{
                  mt: 1.5,
                  py: 1,
                  px: 1.5,
                  bgcolor: 'grey.800',
                  borderRadius: 1.5,
                  borderLeft: 2,
                  borderColor: selectedConfig.color,
                }}
              >
                <Stack spacing={0.25}>
                  {selectedConfig.permissions.map((permission, i) => (
                    <Stack key={i} direction="row" spacing={0.75} alignItems="center">
                      <CheckIcon sx={{ fontSize: 12, color: selectedConfig.color }} />
                      <Typography variant="caption" color="text.secondary">
                        {permission}
                      </Typography>
                    </Stack>
                  ))}
                </Stack>
              </Box>
            </Box>

            {/* Share Link */}
            <Box sx={{ mb: 3 }}>
              <Typography variant="overline" color="text.secondary" sx={{ fontWeight: 600, letterSpacing: 1 }}>
                Share link
              </Typography>
              <Paper
                elevation={0}
                sx={{
                  mt: 1,
                  p: 0.5,
                  pl: 2,
                  bgcolor: 'grey.800',
                  borderRadius: 2,
                  display: 'flex',
                  alignItems: 'center',
                  gap: 1,
                }}
              >
                <Typography
                  variant="body2"
                  sx={{
                    flex: 1,
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                    color: 'text.secondary',
                    fontFamily: 'monospace',
                    fontSize: '0.8rem',
                  }}
                >
                  {shareUrl}
                </Typography>
                <Button
                  variant={copied ? 'contained' : 'text'}
                  color={copied ? 'success' : 'primary'}
                  startIcon={copied ? <CheckIcon /> : <ContentCopyIcon />}
                  onClick={handleCopyLink}
                  sx={{
                    minWidth: 100,
                    borderRadius: 1.5,
                    textTransform: 'none',
                    fontWeight: 600,
                  }}
                >
                  {copied ? 'Copied!' : 'Copy link'}
                </Button>
              </Paper>
              <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                Anyone with this link will join as <strong style={{ color: selectedConfig.color }}>{selectedConfig.label.toLowerCase()}</strong>
              </Typography>
            </Box>

            <Divider sx={{ my: 2, borderColor: 'grey.800' }} />

            {/* Members List */}
            <Box>
              <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 1 }}>
                <Typography variant="overline" color="text.secondary" sx={{ fontWeight: 600, letterSpacing: 1 }}>
                  Members
                </Typography>
                <Chip
                  label={shareInfo.members.length}
                  size="small"
                  sx={{ height: 20, fontSize: '0.7rem', bgcolor: 'grey.800' }}
                />
              </Stack>

              {shareInfo.members.length === 0 ? (
                <Paper
                  elevation={0}
                  sx={{
                    py: 4,
                    textAlign: 'center',
                    bgcolor: 'grey.800',
                    borderRadius: 2,
                  }}
                >
                  <PersonIcon sx={{ fontSize: 40, color: 'grey.600', mb: 1 }} />
                  <Typography color="text.secondary" variant="body2">
                    No one has joined yet
                  </Typography>
                  <Typography color="text.secondary" variant="caption">
                    Share the link above to invite collaborators
                  </Typography>
                </Paper>
              ) : (
                <List sx={{ bgcolor: 'grey.800', borderRadius: 2, py: 0.5 }}>
                  {shareInfo.members.map((member, index) => {
                    const canChangeRole = isOwner || (userRole === 'ADMIN' && member.role !== 'ADMIN');
                    const availableRoles: MemberRole[] = isOwner
                      ? ['VIEWER', 'EDITOR', 'ADMIN']
                      : ['VIEWER', 'EDITOR'];
                    const memberConfig = ROLE_CONFIG[member.role];

                    return (
                      <ListItem
                        key={member.userId}
                        divider={index < shareInfo.members.length - 1}
                        sx={{ py: 1.5, '& .MuiListItemSecondaryAction-root': { right: 8 } }}
                      >
                        <ListItemAvatar>
                          <Avatar
                            sx={{
                              bgcolor: alpha(memberConfig.color, 0.2),
                              color: memberConfig.color,
                              width: 36,
                              height: 36,
                              fontSize: '0.9rem',
                              fontWeight: 600,
                            }}
                          >
                            {member.name ? member.name[0].toUpperCase() : <PersonIcon />}
                          </Avatar>
                        </ListItemAvatar>
                        <ListItemText
                          primary={
                            <Stack direction="row" spacing={1} alignItems="center">
                              <Typography variant="body2" fontWeight={500}>
                                {member.name || 'Anonymous'}
                              </Typography>
                              <Chip
                                label={memberConfig.label}
                                size="small"
                                color={memberConfig.chipColor}
                                sx={{ height: 18, fontSize: '0.65rem', fontWeight: 600 }}
                              />
                            </Stack>
                          }
                          secondary={
                            <Typography variant="caption" color="text.secondary">
                              Joined {new Date(member.joinedAt).toLocaleDateString()}
                            </Typography>
                          }
                        />
                        <ListItemSecondaryAction>
                          <Stack direction="row" spacing={0.5} alignItems="center">
                            {canChangeRole && (
                              <Select
                                value={member.role}
                                onChange={(e) => handleMemberRoleChange(member.userId, e.target.value as MemberRole)}
                                size="small"
                                variant="outlined"
                                disabled={updatingRole === member.userId}
                                sx={{
                                  minWidth: 85,
                                  '& .MuiSelect-select': { py: 0.5, fontSize: '0.75rem' },
                                  '& .MuiOutlinedInput-notchedOutline': { borderColor: 'grey.700' },
                                }}
                              >
                                {availableRoles.map((role) => (
                                  <MenuItem key={role} value={role} sx={{ fontSize: '0.8rem' }}>
                                    {ROLE_CONFIG[role].label}
                                  </MenuItem>
                                ))}
                              </Select>
                            )}

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
                                      '&:hover': { bgcolor: 'error.dark' },
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
                              <Tooltip title="Remove member">
                                <IconButton
                                  onClick={() => handleRemoveClick(member.userId)}
                                  size="small"
                                  sx={{ color: 'grey.500', '&:hover': { color: 'error.main' } }}
                                >
                                  <DeleteIcon fontSize="small" />
                                </IconButton>
                              </Tooltip>
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

      <DialogActions sx={{ px: 3, pb: 2.5, pt: 1, justifyContent: 'space-between' }}>
        {isOwner ? (
          deleting ? (
            <Button color="error" disabled startIcon={<CircularProgress size={16} />} sx={{ textTransform: 'none' }}>
              Deleting...
            </Button>
          ) : confirmingDelete ? (
            <Stack direction="row" spacing={1} alignItems="center">
              <Typography variant="body2" color="error.main" fontWeight={500}>
                Revoke all access?
              </Typography>
              <Tooltip title="Confirm">
                <IconButton
                  size="small"
                  onClick={handleDeleteClick}
                  sx={{
                    color: 'white',
                    bgcolor: 'error.main',
                    width: 28,
                    height: 28,
                    '&:hover': { bgcolor: 'error.dark' },
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
              sx={{ textTransform: 'none', fontWeight: 500 }}
            >
              Revoke access
            </Button>
          )
        ) : (
          <Box />
        )}
        <Button
          onClick={onClose}
          variant="contained"
          sx={{
            textTransform: 'none',
            fontWeight: 600,
            borderRadius: 2,
            px: 3,
          }}
        >
          Done
        </Button>
      </DialogActions>
    </Dialog>
  );
}
