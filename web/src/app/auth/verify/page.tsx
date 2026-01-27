'use client';

import { Suspense, useEffect, useState } from 'react';
import { useSearchParams } from 'next/navigation';
import {
  Box,
  Typography,
  Stack,
  Paper,
  Button,
  CircularProgress,
  Alert,
} from '@mui/material';
import SportsVolleyballIcon from '@mui/icons-material/SportsVolleyball';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import Link from 'next/link';
import { API_BASE_URL } from '@/services/api';
import { designTokens } from '@/app/theme';

function VerifyEmailContent() {
  const searchParams = useSearchParams();
  const token = searchParams.get('token');

  const [status, setStatus] = useState<'loading' | 'success' | 'error'>('loading');
  const [errorMessage, setErrorMessage] = useState<string>('');

  useEffect(() => {
    if (!token) {
      setStatus('error');
      setErrorMessage('No verification token provided.');
      return;
    }

    const verify = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/v1/auth/verify-email`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ token }),
        });

        if (!response.ok) {
          const data = await response.json().catch(() => ({}));
          setStatus('error');
          setErrorMessage(data?.error?.message || 'Verification failed');
          return;
        }

        setStatus('success');
      } catch {
        setStatus('error');
        setErrorMessage('An unexpected error occurred');
      }
    };

    verify();
  }, [token]);

  return (
    <Stack spacing={3} alignItems="center">
      {/* Logo */}
      <Stack
        component={Link}
        href="/"
        direction="row"
        alignItems="center"
        spacing={1}
        sx={{ textDecoration: 'none' }}
      >
        <SportsVolleyballIcon sx={{ fontSize: 32, color: 'primary.main' }} />
        <Typography
          variant="h5"
          sx={{
            fontWeight: 700,
            background: designTokens.gradients.primary,
            backgroundClip: 'text',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
          }}
        >
          RallyCut
        </Typography>
      </Stack>

      {status === 'loading' && (
        <>
          <CircularProgress />
          <Typography color="text.secondary">
            Verifying your email...
          </Typography>
        </>
      )}

      {status === 'success' && (
        <>
          <CheckCircleIcon sx={{ fontSize: 48, color: 'success.main' }} />
          <Typography variant="h6" fontWeight={600}>
            Email verified
          </Typography>
          <Typography color="text.secondary">
            Your email has been verified. You can now sign in.
          </Typography>
          <Button
            component={Link}
            href="/auth/signin"
            variant="contained"
            sx={{ mt: 1 }}
          >
            Sign In
          </Button>
        </>
      )}

      {status === 'error' && (
        <>
          <Alert severity="error" sx={{ width: '100%' }}>
            {errorMessage}
          </Alert>
          <Button
            component={Link}
            href="/auth/signin"
            variant="outlined"
          >
            Go to Sign In
          </Button>
        </>
      )}
    </Stack>
  );
}

export default function VerifyEmailPage() {
  return (
    <Box
      sx={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        bgcolor: 'background.default',
        px: 2,
      }}
    >
      <Paper
        sx={{
          maxWidth: 420,
          width: '100%',
          p: 4,
          bgcolor: 'background.paper',
          border: '1px solid',
          borderColor: 'divider',
          borderRadius: 2,
          textAlign: 'center',
        }}
      >
        <Suspense fallback={<CircularProgress />}>
          <VerifyEmailContent />
        </Suspense>
      </Paper>
    </Box>
  );
}
