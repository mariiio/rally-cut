'use client';

import { useEffect, useState } from 'react';
import { useParams, useRouter } from 'next/navigation';
import {
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Container,
  TextField,
  Typography,
} from '@mui/material';
import PersonIcon from '@mui/icons-material/Person';
import FolderSharedIcon from '@mui/icons-material/FolderShared';
import { getSharePreview, acceptShare, getCurrentUser, type SharePreview } from '@/services/api';

export default function AcceptSharePage() {
  const params = useParams();
  const router = useRouter();
  const token = params.token as string;

  const [loading, setLoading] = useState(true);
  const [accepting, setAccepting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [preview, setPreview] = useState<SharePreview | null>(null);
  const [userName, setUserName] = useState<string | null>(null);
  const [nameInput, setNameInput] = useState('');
  const [needsName, setNeedsName] = useState(false);

  useEffect(() => {
    async function loadData() {
      try {
        // Load share preview and current user in parallel
        const [shareData, userData] = await Promise.all([
          getSharePreview(token),
          getCurrentUser().catch(() => null),
        ]);
        setPreview(shareData);

        // Check if user already has a name
        if (userData?.name) {
          setUserName(userData.name);
          setNeedsName(false);
        } else {
          setNeedsName(true);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load share');
      } finally {
        setLoading(false);
      }
    }

    if (token) {
      loadData();
    }
  }, [token]);

  const handleAccept = async () => {
    // Validate name if needed
    if (needsName && !nameInput.trim()) {
      setError('Please enter your name');
      return;
    }

    setAccepting(true);
    setError(null);

    try {
      // Pass name only if user needs to set one
      const nameToSend = needsName ? nameInput.trim() : undefined;
      const result = await acceptShare(token, nameToSend);

      if (result.alreadyOwner) {
        // User is the owner, just redirect
        router.push(`/sessions/${result.sessionId}`);
        return;
      }

      if (result.alreadyMember) {
        // User is already a member, just redirect
        router.push(`/sessions/${result.sessionId}`);
        return;
      }

      // Successfully joined, redirect to session
      router.push(`/sessions/${result.sessionId}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to accept share');
      setAccepting(false);
    }
  };

  if (loading) {
    return (
      <Box
        sx={{
          minHeight: '100vh',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          bgcolor: 'grey.900',
        }}
      >
        <CircularProgress />
      </Box>
    );
  }

  if (error && !preview) {
    return (
      <Box
        sx={{
          minHeight: '100vh',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          bgcolor: 'grey.900',
        }}
      >
        <Container maxWidth="sm">
          <Card sx={{ bgcolor: 'grey.800', textAlign: 'center', p: 4 }}>
            <CardContent>
              <Typography variant="h5" gutterBottom color="error.main">
                Share Not Found
              </Typography>
              <Typography color="text.secondary" sx={{ mb: 3 }}>
                This share link may have expired or been removed.
              </Typography>
              <Button variant="contained" onClick={() => router.push('/sessions')}>
                Go Home
              </Button>
            </CardContent>
          </Card>
        </Container>
      </Box>
    );
  }

  return (
    <Box
      sx={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        bgcolor: 'grey.900',
      }}
    >
      <Container maxWidth="sm">
        <Card sx={{ bgcolor: 'grey.800', textAlign: 'center', p: 4 }}>
          <CardContent>
            <FolderSharedIcon sx={{ fontSize: 64, color: 'primary.main', mb: 2 }} />

            <Typography variant="h5" gutterBottom>
              You&apos;ve been invited to a session
            </Typography>

            <Box
              sx={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: 1,
                my: 3,
                p: 2,
                bgcolor: 'grey.900',
                borderRadius: 1,
              }}
            >
              <PersonIcon sx={{ color: 'text.secondary' }} />
              <Typography color="text.secondary">
                {preview?.ownerName || 'Someone'} wants to share
              </Typography>
            </Box>

            <Typography variant="h4" sx={{ mb: 3, fontWeight: 500 }}>
              &ldquo;{preview?.sessionName}&rdquo;
            </Typography>

            {preview?.defaultRole && (
              <Chip
                label={`You'll join as ${preview.defaultRole === 'ADMIN' ? 'Admin' : preview.defaultRole === 'EDITOR' ? 'Editor' : 'Viewer'}`}
                size="small"
                color={preview.defaultRole === 'ADMIN' ? 'warning' : preview.defaultRole === 'EDITOR' ? 'info' : 'default'}
                sx={{ mb: 2 }}
              />
            )}

            <Typography color="text.secondary" sx={{ mb: 3 }}>
              {preview?.defaultRole === 'EDITOR'
                ? 'Accept the invitation to edit rallies and create highlights in this session.'
                : preview?.defaultRole === 'ADMIN'
                  ? 'Accept the invitation to manage and collaborate on this session.'
                  : 'Accept the invitation to view rallies and highlights in this session.'}
            </Typography>

            {/* Name input for users without a name */}
            {needsName && (
              <TextField
                fullWidth
                label="Your name"
                value={nameInput}
                onChange={(e) => setNameInput(e.target.value)}
                placeholder="Enter your name to join"
                sx={{ mb: 3 }}
                inputProps={{ maxLength: 100 }}
                helperText="This name will be visible to other session members"
              />
            )}

            {/* Show existing name */}
            {userName && (
              <Typography color="text.secondary" sx={{ mb: 3 }}>
                Joining as <strong>{userName}</strong>
              </Typography>
            )}

            {error && (
              <Typography color="error.main" sx={{ mb: 2 }}>
                {error}
              </Typography>
            )}

            <Button
              variant="contained"
              size="large"
              onClick={handleAccept}
              disabled={accepting}
              sx={{ minWidth: 200 }}
            >
              {accepting ? <CircularProgress size={24} /> : 'Accept & Open'}
            </Button>
          </CardContent>
        </Card>
      </Container>
    </Box>
  );
}
