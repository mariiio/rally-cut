'use client';

import { useEffect, useState } from 'react';
import { useParams, useRouter } from 'next/navigation';
import {
  Box,
  Button,
  Card,
  CardContent,
  CircularProgress,
  Container,
  Typography,
} from '@mui/material';
import PersonIcon from '@mui/icons-material/Person';
import FolderSharedIcon from '@mui/icons-material/FolderShared';
import { getSharePreview, acceptShare, type SharePreview } from '@/services/api';

export default function AcceptSharePage() {
  const params = useParams();
  const router = useRouter();
  const token = params.token as string;

  const [loading, setLoading] = useState(true);
  const [accepting, setAccepting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [preview, setPreview] = useState<SharePreview | null>(null);

  useEffect(() => {
    async function loadPreview() {
      try {
        const data = await getSharePreview(token);
        setPreview(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load share');
      } finally {
        setLoading(false);
      }
    }

    if (token) {
      loadPreview();
    }
  }, [token]);

  const handleAccept = async () => {
    setAccepting(true);
    setError(null);

    try {
      const result = await acceptShare(token);

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
              <Button variant="contained" onClick={() => router.push('/')}>
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

            <Typography color="text.secondary" sx={{ mb: 4 }}>
              Accept the invitation to view and add highlights to this session.
            </Typography>

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
