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
  Chip,
  CircularProgress,
  Autocomplete,
  TextField,
  Divider,
} from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import CloseIcon from '@mui/icons-material/Close';
import {
  listSessions,
  listVideos,
  addVideoToSession,
  removeVideoFromSession,
  createSession,
  type SessionType,
} from '@/services/api';

interface SessionOption {
  id: string;
  name: string;
  type: SessionType;
}

interface VideoSession {
  id: string;
  name: string;
  type: SessionType;
}

interface ManageSessionsDialogProps {
  open: boolean;
  onClose: () => void;
  video: { id: string; name: string; sessions?: VideoSession[] } | null;
  onChanged: () => void;
}

export function ManageSessionsDialog({
  open,
  onClose,
  video,
  onChanged,
}: ManageSessionsDialogProps) {
  const [allSessions, setAllSessions] = useState<SessionOption[]>([]);
  const [videoSessions, setVideoSessions] = useState<VideoSession[]>([]);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Track pending changes
  const [sessionsToAdd, setSessionsToAdd] = useState<SessionOption[]>([]);
  const [sessionsToRemove, setSessionsToRemove] = useState<Set<string>>(new Set());

  // For inline session creation
  const [newSessionName, setNewSessionName] = useState('');
  const [isCreatingSession, setIsCreatingSession] = useState(false);

  // Load available sessions and video sessions
  const loadData = useCallback(async (videoId: string, providedSessions?: VideoSession[]) => {
    try {
      setLoading(true);

      // Load all available sessions
      const sessionsResult = await listSessions(1, 100);
      const regularSessions = sessionsResult.data.filter(s => s.type === 'REGULAR');
      setAllSessions(regularSessions);

      // If sessions provided, use them; otherwise fetch video's sessions
      if (providedSessions) {
        setVideoSessions(providedSessions.filter(s => s.type !== 'ALL_VIDEOS'));
      } else {
        // Fetch video to get its sessions
        const videosResult = await listVideos(1, 100);
        const foundVideo = videosResult.data.find(v => v.id === videoId);
        if (foundVideo?.sessions) {
          setVideoSessions(foundVideo.sessions.filter(s => s.type !== 'ALL_VIDEOS'));
        } else {
          setVideoSessions([]);
        }
      }
    } catch (err) {
      console.error('Failed to load data:', err);
      setError('Failed to load sessions');
    } finally {
      setLoading(false);
    }
  }, []);

  // Load data when dialog opens
  useEffect(() => {
    if (open && video) {
      loadData(video.id, video.sessions);
      // Reset pending changes
      setSessionsToAdd([]);
      setSessionsToRemove(new Set());
      setNewSessionName('');
      setError(null);
    }
  }, [open, video, loadData]);

  // Get current sessions (excluding ALL_VIDEOS and accounting for pending changes)
  const getCurrentSessions = useCallback(() => {
    // Filter out sessions marked for removal
    const afterRemoval = videoSessions.filter(s => !sessionsToRemove.has(s.id));
    // Add sessions marked for addition
    return [...afterRemoval, ...sessionsToAdd];
  }, [videoSessions, sessionsToRemove, sessionsToAdd]);

  // Get available sessions (not already in current)
  const getAvailableSessions = useCallback(() => {
    const currentIds = new Set(getCurrentSessions().map(s => s.id));
    return allSessions.filter(s => !currentIds.has(s.id));
  }, [allSessions, getCurrentSessions]);

  const handleRemoveSession = (sessionId: string) => {
    // Check if this is a pending addition
    const pendingAdd = sessionsToAdd.find(s => s.id === sessionId);
    if (pendingAdd) {
      // Just remove from pending additions
      setSessionsToAdd(prev => prev.filter(s => s.id !== sessionId));
    } else {
      // Mark for removal
      setSessionsToRemove(prev => new Set(prev).add(sessionId));
    }
  };

  const handleAddSession = (session: SessionOption | null) => {
    if (!session) return;
    // Check if it was previously marked for removal
    if (sessionsToRemove.has(session.id)) {
      // Undo the removal
      setSessionsToRemove(prev => {
        const next = new Set(prev);
        next.delete(session.id);
        return next;
      });
    } else {
      // Add to pending additions
      setSessionsToAdd(prev => [...prev, session]);
    }
  };

  const handleCreateSession = async () => {
    if (!newSessionName.trim()) return;

    try {
      setIsCreatingSession(true);
      const session = await createSession(newSessionName.trim());
      // Add to all sessions list and to pending additions
      const newSession: SessionOption = { id: session.id, name: session.name, type: 'REGULAR' };
      setAllSessions(prev => [...prev, newSession]);
      setSessionsToAdd(prev => [...prev, newSession]);
      setNewSessionName('');
    } catch (err) {
      console.error('Failed to create session:', err);
      setError('Failed to create session');
    } finally {
      setIsCreatingSession(false);
    }
  };

  const handleSave = async () => {
    if (!video) return;

    try {
      setSaving(true);
      setError(null);

      // Apply removals
      for (const sessionId of sessionsToRemove) {
        await removeVideoFromSession(sessionId, video.id);
      }

      // Apply additions
      for (const session of sessionsToAdd) {
        await addVideoToSession(session.id, video.id);
      }

      onChanged();
      onClose();
    } catch (err) {
      console.error('Failed to save session changes:', err);
      setError('Failed to save changes');
    } finally {
      setSaving(false);
    }
  };

  const hasChanges = sessionsToAdd.length > 0 || sessionsToRemove.size > 0;
  const currentSessions = getCurrentSessions();
  const availableSessions = getAvailableSessions();

  return (
    <Dialog open={open} onClose={onClose} maxWidth="xs" fullWidth>
      <DialogTitle>
        <Typography variant="h6" component="div">
          Manage Sessions
        </Typography>
        {video && (
          <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
            {video.name}
          </Typography>
        )}
      </DialogTitle>

      <DialogContent>
        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
            <CircularProgress />
          </Box>
        ) : (
          <>
            {error && (
              <Typography color="error" variant="body2" sx={{ mb: 2 }}>
                {error}
              </Typography>
            )}

            {/* Current sessions */}
            <Box sx={{ mb: 3 }}>
              <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 1 }}>
                Current Sessions
              </Typography>
              {currentSessions.length === 0 ? (
                <Typography variant="body2" color="text.disabled">
                  Not in any session
                </Typography>
              ) : (
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                  {currentSessions.map((session) => {
                    const isPendingAdd = sessionsToAdd.some(s => s.id === session.id);
                    return (
                      <Chip
                        key={session.id}
                        label={session.name}
                        onDelete={() => handleRemoveSession(session.id)}
                        deleteIcon={<CloseIcon />}
                        size="small"
                        variant={isPendingAdd ? 'filled' : 'outlined'}
                        color={isPendingAdd ? 'primary' : 'default'}
                        sx={{
                          '& .MuiChip-deleteIcon': {
                            fontSize: 16,
                          },
                        }}
                      />
                    );
                  })}
                </Box>
              )}
            </Box>

            <Divider sx={{ my: 2 }} />

            {/* Add to session */}
            <Box>
              <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 1 }}>
                Add to Session
              </Typography>
              <Autocomplete
                options={availableSessions}
                getOptionLabel={(option) => option.name}
                onChange={(_, value) => handleAddSession(value)}
                value={null}
                renderInput={(params) => (
                  <TextField
                    {...params}
                    placeholder="Select a session..."
                    size="small"
                  />
                )}
                size="small"
                sx={{ mb: 2 }}
              />

              {/* Create new session inline */}
              <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                <TextField
                  placeholder="New session name"
                  value={newSessionName}
                  onChange={(e) => setNewSessionName(e.target.value)}
                  size="small"
                  fullWidth
                  disabled={isCreatingSession}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && newSessionName.trim()) {
                      handleCreateSession();
                    }
                  }}
                />
                <Button
                  variant="outlined"
                  size="small"
                  onClick={handleCreateSession}
                  disabled={!newSessionName.trim() || isCreatingSession}
                  startIcon={isCreatingSession ? <CircularProgress size={14} /> : <AddIcon />}
                  sx={{ whiteSpace: 'nowrap', minWidth: 'auto', px: 1.5 }}
                >
                  {isCreatingSession ? '' : 'Create'}
                </Button>
              </Box>
            </Box>
          </>
        )}
      </DialogContent>

      <DialogActions sx={{ px: 3, pb: 2 }}>
        <Button onClick={onClose} disabled={saving}>
          Cancel
        </Button>
        <Button
          variant="contained"
          onClick={handleSave}
          disabled={!hasChanges || saving}
        >
          {saving ? 'Saving...' : 'Save'}
        </Button>
      </DialogActions>
    </Dialog>
  );
}
