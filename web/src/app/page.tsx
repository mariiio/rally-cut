'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import {
  Box,
  Container,
  Typography,
  Card,
  CardContent,
  CardActionArea,
  Grid,
  Button,
  TextField,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  CircularProgress,
  Chip,
  Stack,
  IconButton,
  Tooltip,
} from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import DeleteIcon from '@mui/icons-material/Delete';
import VideoLibraryIcon from '@mui/icons-material/VideoLibrary';
import { listSessions, createSession, deleteSession, type ListSessionsResponse } from '@/services/api';

export default function HomePage() {
  const router = useRouter();
  const [sessions, setSessions] = useState<ListSessionsResponse['data']>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [newSessionName, setNewSessionName] = useState('');
  const [creating, setCreating] = useState(false);
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);

  useEffect(() => {
    loadSessions();
  }, []);

  const loadSessions = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await listSessions();
      setSessions(response.data);
    } catch (err) {
      console.error('Failed to load sessions:', err);
      setError('Failed to load sessions. Make sure the API is running.');
    } finally {
      setLoading(false);
    }
  };

  const handleCreateSession = async () => {
    if (!newSessionName.trim()) return;

    try {
      setCreating(true);
      const session = await createSession(newSessionName.trim());
      setCreateDialogOpen(false);
      setNewSessionName('');
      router.push(`/sessions/${session.id}`);
    } catch (err) {
      console.error('Failed to create session:', err);
      setError('Failed to create session');
    } finally {
      setCreating(false);
    }
  };

  const handleDeleteSession = async (sessionId: string) => {
    try {
      setDeleting(true);
      await deleteSession(sessionId);
      setSessions(sessions.filter(s => s.id !== sessionId));
      setDeleteConfirmId(null);
    } catch (err) {
      console.error('Failed to delete session:', err);
      setError('Failed to delete session');
    } finally {
      setDeleting(false);
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  return (
    <Box
      sx={{
        minHeight: '100vh',
        bgcolor: 'grey.900',
        color: 'white',
        py: 4,
      }}
    >
      <Container maxWidth="lg">
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 4 }}>
          <Box>
            <Typography variant="h4" fontWeight="bold" gutterBottom>
              RallyCut
            </Typography>
            <Typography variant="body1" color="grey.400">
              Beach Volleyball Video Analysis
            </Typography>
          </Box>
          <Button
            variant="contained"
            startIcon={<AddIcon />}
            onClick={() => setCreateDialogOpen(true)}
            sx={{ bgcolor: 'primary.main' }}
          >
            New Session
          </Button>
        </Box>

        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', py: 8 }}>
            <CircularProgress />
          </Box>
        ) : error ? (
          <Box sx={{ textAlign: 'center', py: 8 }}>
            <Typography color="error" gutterBottom>
              {error}
            </Typography>
            <Button variant="outlined" onClick={loadSessions} sx={{ mt: 2 }}>
              Retry
            </Button>
          </Box>
        ) : sessions.length === 0 ? (
          <Box sx={{ textAlign: 'center', py: 8 }}>
            <VideoLibraryIcon sx={{ fontSize: 64, color: 'grey.600', mb: 2 }} />
            <Typography variant="h6" color="grey.400" gutterBottom>
              No sessions yet
            </Typography>
            <Typography color="grey.500" sx={{ mb: 3 }}>
              Create your first session to start analyzing volleyball videos
            </Typography>
            <Button
              variant="contained"
              startIcon={<AddIcon />}
              onClick={() => setCreateDialogOpen(true)}
            >
              Create Session
            </Button>
          </Box>
        ) : (
          <Grid container spacing={3}>
            {sessions.map((session) => (
              <Grid size={{ xs: 12, sm: 6, md: 4 }} key={session.id}>
                <Card
                  sx={{
                    bgcolor: 'grey.800',
                    '&:hover': { bgcolor: 'grey.750' },
                    transition: 'background-color 0.2s',
                    position: 'relative',
                  }}
                >
                  <CardActionArea onClick={() => router.push(`/sessions/${session.id}`)}>
                    <CardContent>
                      <Typography variant="h6" gutterBottom noWrap sx={{ pr: 4 }}>
                        {session.name}
                      </Typography>
                      <Stack direction="row" spacing={1} sx={{ mb: 2 }}>
                        <Chip
                          label={`${session._count.videos} videos`}
                          size="small"
                          sx={{ bgcolor: 'grey.700' }}
                        />
                        <Chip
                          label={`${session._count.highlights} highlights`}
                          size="small"
                          sx={{ bgcolor: 'grey.700' }}
                        />
                      </Stack>
                      <Typography variant="caption" color="grey.500">
                        Updated {formatDate(session.updatedAt)}
                      </Typography>
                    </CardContent>
                  </CardActionArea>
                  {deleteConfirmId === session.id ? (
                    <Stack
                      direction="row"
                      spacing={1}
                      sx={{
                        position: 'absolute',
                        top: 8,
                        right: 8,
                        bgcolor: 'rgba(0,0,0,0.9)',
                        borderRadius: 1,
                        p: 0.5,
                      }}
                    >
                      <Button
                        size="small"
                        variant="contained"
                        color="error"
                        onClick={() => handleDeleteSession(session.id)}
                        disabled={deleting}
                        sx={{ minWidth: 'auto', px: 1 }}
                      >
                        {deleting ? '...' : 'Delete'}
                      </Button>
                      <Button
                        size="small"
                        variant="outlined"
                        onClick={() => setDeleteConfirmId(null)}
                        sx={{ minWidth: 'auto', px: 1 }}
                      >
                        Cancel
                      </Button>
                    </Stack>
                  ) : (
                    <Tooltip title="Delete session">
                      <IconButton
                        size="small"
                        onClick={(e) => {
                          e.stopPropagation();
                          setDeleteConfirmId(session.id);
                        }}
                        sx={{
                          position: 'absolute',
                          top: 8,
                          right: 8,
                          color: 'grey.500',
                          '&:hover': { color: 'error.main', bgcolor: 'rgba(255,255,255,0.1)' },
                        }}
                      >
                        <DeleteIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  )}
                </Card>
              </Grid>
            ))}
          </Grid>
        )}

        <Dialog open={createDialogOpen} onClose={() => setCreateDialogOpen(false)}>
          <DialogTitle>Create New Session</DialogTitle>
          <DialogContent>
            <TextField
              autoFocus
              margin="dense"
              label="Session Name"
              fullWidth
              variant="outlined"
              value={newSessionName}
              onChange={(e) => setNewSessionName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') handleCreateSession();
              }}
              placeholder="e.g., Beach Tournament 2024"
            />
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setCreateDialogOpen(false)}>Cancel</Button>
            <Button
              onClick={handleCreateSession}
              variant="contained"
              disabled={!newSessionName.trim() || creating}
            >
              {creating ? 'Creating...' : 'Create'}
            </Button>
          </DialogActions>
        </Dialog>
      </Container>
    </Box>
  );
}
