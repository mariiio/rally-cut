'use client';

import { Suspense } from 'react';
import { useSearchParams } from 'next/navigation';
import {
  Box,
  Typography,
  Stack,
  Paper,
  Button,
  Alert,
  CircularProgress,
} from '@mui/material';
import SportsVolleyballIcon from '@mui/icons-material/SportsVolleyball';
import Link from 'next/link';
import { designTokens } from '@/app/theme';

const errorMessages: Record<string, string> = {
  Configuration: 'There is a problem with the server configuration.',
  AccessDenied: 'Access denied. You do not have permission to sign in.',
  Verification: 'The verification link may have expired or already been used.',
  Default: 'An authentication error occurred.',
};

function AuthErrorContent() {
  const searchParams = useSearchParams();
  const errorType = searchParams.get('error') || 'Default';
  const message = errorMessages[errorType] || errorMessages.Default;

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

      <Typography variant="h6" fontWeight={600}>
        Authentication Error
      </Typography>
      <Alert severity="error" sx={{ width: '100%' }}>
        {message}
      </Alert>
      <Stack direction="row" spacing={2}>
        <Button
          component={Link}
          href="/auth/signin"
          variant="contained"
        >
          Try Again
        </Button>
        <Button
          component={Link}
          href="/"
          variant="outlined"
        >
          Go Home
        </Button>
      </Stack>
    </Stack>
  );
}

export default function AuthErrorPage() {
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
          <AuthErrorContent />
        </Suspense>
      </Paper>
    </Box>
  );
}
