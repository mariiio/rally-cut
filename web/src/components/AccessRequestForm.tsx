'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import {
  Box,
  Button,
  Card,
  CardContent,
  Container,
  TextField,
  Typography,
  CircularProgress,
  Alert,
} from '@mui/material';
import LockIcon from '@mui/icons-material/Lock';
import AccessTimeIcon from '@mui/icons-material/AccessTime';
import PersonIcon from '@mui/icons-material/Person';
import { requestAccess } from '@/services/api';

interface AccessRequestFormProps {
  sessionId: string;
  sessionName?: string;
  ownerName?: string | null;
  hasPendingRequest?: boolean;
}

export function AccessRequestForm({
  sessionId,
  sessionName,
  ownerName,
  hasPendingRequest = false,
}: AccessRequestFormProps) {
  const router = useRouter();
  const [message, setMessage] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  const handleSubmit = async () => {
    setSubmitting(true);
    setError(null);

    try {
      await requestAccess(sessionId, message.trim() || undefined);
      setSuccess(true);
    } catch (err) {
      console.error('Failed to request access:', err);
      setError(err instanceof Error ? err.message : 'Failed to send request. Please try again.');
    } finally {
      setSubmitting(false);
    }
  };

  // Show pending request state
  if (hasPendingRequest || success) {
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
              <AccessTimeIcon sx={{ fontSize: 64, color: 'warning.main', mb: 2 }} />

              <Typography variant="h5" gutterBottom>
                Access Request Pending
              </Typography>

              {sessionName && (
                <Typography variant="h6" sx={{ mb: 2, fontWeight: 500 }}>
                  &ldquo;{sessionName}&rdquo;
                </Typography>
              )}

              <Typography color="text.secondary" sx={{ mb: 3 }}>
                {success
                  ? 'Your request has been sent! The session owner will review it.'
                  : 'You have already requested access to this session. Please wait for the owner to review your request.'}
              </Typography>

              {ownerName && (
                <Box
                  sx={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: 1,
                    mb: 3,
                    p: 2,
                    bgcolor: 'grey.900',
                    borderRadius: 1,
                  }}
                >
                  <PersonIcon sx={{ color: 'text.secondary' }} />
                  <Typography color="text.secondary">
                    Waiting for {ownerName} to approve
                  </Typography>
                </Box>
              )}

              <Button variant="outlined" onClick={() => router.push('/sessions')}>
                Go to My Sessions
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
            <LockIcon sx={{ fontSize: 64, color: 'primary.main', mb: 2 }} />

            <Typography variant="h5" gutterBottom>
              Request Access
            </Typography>

            {sessionName && (
              <Typography variant="h6" sx={{ mb: 2, fontWeight: 500 }}>
                &ldquo;{sessionName}&rdquo;
              </Typography>
            )}

            <Typography color="text.secondary" sx={{ mb: 3 }}>
              You don&apos;t have access to this session.
              {ownerName
                ? ` Send a request to ${ownerName} to get access.`
                : ' Send a request to the owner to get access.'}
            </Typography>

            {error && (
              <Alert severity="error" sx={{ mb: 2, textAlign: 'left' }}>
                {error}
              </Alert>
            )}

            <TextField
              fullWidth
              multiline
              rows={3}
              label="Message (optional)"
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              placeholder="Add a message to the owner..."
              sx={{ mb: 3 }}
            />

            <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center' }}>
              <Button variant="outlined" onClick={() => router.push('/sessions')}>
                Cancel
              </Button>
              <Button
                variant="contained"
                onClick={handleSubmit}
                disabled={submitting}
                startIcon={submitting ? <CircularProgress size={20} /> : undefined}
              >
                {submitting ? 'Sending...' : 'Request Access'}
              </Button>
            </Box>
          </CardContent>
        </Card>
      </Container>
    </Box>
  );
}
