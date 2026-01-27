'use client';

import { Suspense, useEffect, useState } from 'react';
import { getProviders, signIn } from 'next-auth/react';
import { useRouter, useSearchParams } from 'next/navigation';
import {
  Box,
  Button,
  TextField,
  Typography,
  Stack,
  Divider,
  Alert,
  Paper,
  CircularProgress,
  IconButton,
  InputAdornment,
} from '@mui/material';
import SportsVolleyballIcon from '@mui/icons-material/SportsVolleyball';
import GoogleIcon from '@mui/icons-material/Google';
import Visibility from '@mui/icons-material/Visibility';
import VisibilityOff from '@mui/icons-material/VisibilityOff';
import Link from 'next/link';
import { API_BASE_URL, getHeaders } from '@/services/api';
import { getVisitorId } from '@/utils/visitorId';
import { designTokens } from '@/app/theme';

function SignInContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const callbackUrl = searchParams.get('callbackUrl') || '/sessions';
  const reason = searchParams.get('reason');

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [errorCode, setErrorCode] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [hasGoogle, setHasGoogle] = useState(false);
  const [providersLoaded, setProvidersLoaded] = useState(false);
  const [resendLoading, setResendLoading] = useState(false);
  const [resendSuccess, setResendSuccess] = useState(false);

  useEffect(() => {
    getProviders().then((providers) => {
      setHasGoogle(!!providers?.google);
      setProvidersLoaded(true);
    });
  }, []);

  const handleCredentialsSignIn = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setErrorCode(null);
    setResendSuccess(false);
    setLoading(true);

    try {
      const result = await signIn('credentials', {
        email,
        password,
        redirect: false,
      });

      if (result?.error) {
        if (result.error === 'EMAIL_NOT_VERIFIED') {
          setErrorCode('EMAIL_NOT_VERIFIED');
          setError('Please verify your email before signing in. Check your inbox.');
        } else {
          setError('Invalid email or password');
        }
      } else {
        router.push(callbackUrl);
        router.refresh();
      }
    } catch {
      setError('An unexpected error occurred');
    } finally {
      setLoading(false);
    }
  };

  const handleResendVerification = async () => {
    setResendLoading(true);
    setResendSuccess(false);

    try {
      const response = await fetch(`${API_BASE_URL}/v1/auth/resend-verification`, {
        method: 'POST',
        headers: getHeaders('application/json'),
      });

      if (response.ok) {
        setResendSuccess(true);
      } else {
        const data = await response.json().catch(() => ({}));
        const code = data?.error?.code;
        if (code === 'RATE_LIMITED') {
          setError('Please wait before requesting another verification email.');
        } else if (code === 'NO_EMAIL') {
          setError('Could not find your account. Please try registering again.');
        } else {
          setError(data?.error?.message || 'Failed to resend verification email.');
        }
      }
    } catch {
      setError('Failed to resend verification email.');
    } finally {
      setResendLoading(false);
    }
  };

  const handleGoogleSignIn = () => {
    // Store visitorId for post-OAuth anonymous data linking
    const visitorId = getVisitorId();
    if (visitorId) {
      sessionStorage.setItem('rallycut_link_visitor_id', visitorId);
    }
    signIn('google', { callbackUrl });
  };

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
        <SportsVolleyballIcon
          sx={{ fontSize: 32, color: 'primary.main' }}
        />
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
        Sign in to your account
      </Typography>

      {reason && (
        <Alert severity="info" sx={{ width: '100%' }}>
          {reason}
        </Alert>
      )}

      {error && (
        <Alert severity="error" sx={{ width: '100%' }}>
          {error}
        </Alert>
      )}

      {errorCode === 'EMAIL_NOT_VERIFIED' && (
        <Stack spacing={1} sx={{ width: '100%' }}>
          {resendSuccess ? (
            <Alert severity="success" sx={{ width: '100%' }}>
              Verification email sent! Check your inbox.
            </Alert>
          ) : (
            <Button
              variant="outlined"
              size="small"
              onClick={handleResendVerification}
              disabled={resendLoading}
              fullWidth
            >
              {resendLoading ? 'Sending...' : 'Resend verification email'}
            </Button>
          )}
        </Stack>
      )}

      {/* Google Sign In */}
      {providersLoaded && hasGoogle && (
        <>
          <Button
            fullWidth
            variant="outlined"
            startIcon={<GoogleIcon />}
            onClick={handleGoogleSignIn}
            sx={{
              py: 1.25,
              borderColor: 'divider',
              color: 'text.primary',
              '&:hover': {
                borderColor: 'text.secondary',
                bgcolor: 'action.hover',
              },
            }}
          >
            Continue with Google
          </Button>

          <Divider sx={{ width: '100%' }}>
            <Typography variant="body2" color="text.secondary">
              or
            </Typography>
          </Divider>
        </>
      )}

      {/* Email/Password Form */}
      <Box component="form" onSubmit={handleCredentialsSignIn} sx={{ width: '100%' }}>
        <Stack spacing={2}>
          <TextField
            label="Email"
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
            fullWidth
            autoComplete="email"
          />
          <TextField
            label="Password"
            type={showPassword ? 'text' : 'password'}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            fullWidth
            autoComplete="current-password"
            slotProps={{
              input: {
                endAdornment: (
                  <InputAdornment position="end">
                    <IconButton
                      onClick={() => setShowPassword(!showPassword)}
                      edge="end"
                      size="small"
                    >
                      {showPassword ? <VisibilityOff /> : <Visibility />}
                    </IconButton>
                  </InputAdornment>
                ),
              },
            }}
          />
          <Button
            type="submit"
            variant="contained"
            fullWidth
            disabled={loading}
            sx={{ py: 1.25 }}
          >
            {loading ? 'Signing in...' : 'Sign In'}
          </Button>
        </Stack>
      </Box>

      <Typography variant="body2" color="text.secondary">
        Don&apos;t have an account?{' '}
        <Typography
          component={Link}
          href={`/auth/register${callbackUrl !== '/sessions' ? `?callbackUrl=${encodeURIComponent(callbackUrl)}` : ''}`}
          variant="body2"
          sx={{ color: 'primary.main', textDecoration: 'none', '&:hover': { textDecoration: 'underline' } }}
        >
          Create one
        </Typography>
      </Typography>
    </Stack>
  );
}

export default function SignInPage() {
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
        }}
      >
        <Suspense fallback={<CircularProgress />}>
          <SignInContent />
        </Suspense>
      </Paper>
    </Box>
  );
}
