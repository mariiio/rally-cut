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

function RegisterContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const callbackUrl = searchParams.get('callbackUrl') || '/sessions';

  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);
  const [hasGoogle, setHasGoogle] = useState(false);
  const [providersLoaded, setProvidersLoaded] = useState(false);

  useEffect(() => {
    getProviders().then((providers) => {
      setHasGoogle(!!providers?.google);
      setProvidersLoaded(true);
    });
  }, []);

  const handleRegister = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (password !== confirmPassword) {
      setError('Passwords do not match');
      return;
    }

    if (password.length < 8) {
      setError('Password must be at least 8 characters');
      return;
    }

    setLoading(true);

    try {
      // Register via Express API (upgrades anonymous user)
      const response = await fetch(`${API_BASE_URL}/v1/auth/register`, {
        method: 'POST',
        headers: getHeaders('application/json'),
        body: JSON.stringify({ email, password, name: name || undefined }),
      });

      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        const code = data?.error?.code;

        if (code === 'EMAIL_EXISTS') {
          setError('An account with this email already exists. Try signing in.');
        } else if (code === 'DISPOSABLE_EMAIL') {
          setError('Disposable email addresses are not allowed.');
        } else {
          setError(data?.error?.message || 'Registration failed');
        }
        return;
      }

      const data = await response.json();

      // In dev mode, email is auto-verified — sign in directly
      if (data.autoVerified) {
        const result = await signIn('credentials', {
          email,
          password,
          redirect: false,
        });

        if (result?.error) {
          // Fallback to manual sign-in if auto-sign-in fails
          setSuccess(true);
        } else {
          router.push(callbackUrl);
          router.refresh();
        }
      } else {
        setSuccess(true);
      }
    } catch {
      setError('An unexpected error occurred');
    } finally {
      setLoading(false);
    }
  };

  const handleGoogleSignIn = () => {
    // Store visitorId for post-OAuth linking
    const visitorId = getVisitorId();
    if (visitorId) {
      sessionStorage.setItem('rallycut_link_visitor_id', visitorId);
    }

    signIn('google', { callbackUrl });
  };

  const signInHref = `/auth/signin${callbackUrl !== '/sessions' ? `?callbackUrl=${encodeURIComponent(callbackUrl)}` : ''}`;

  const confirmPasswordError =
    confirmPassword.length > 0 && password !== confirmPassword;

  if (success) {
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
          Check your email
        </Typography>
        <Typography color="text.secondary">
          We sent a verification link to <strong>{email}</strong>.
          Please verify your email to complete registration.
        </Typography>
        <Button
          component={Link}
          href={signInHref}
          variant="contained"
          sx={{ mt: 2 }}
        >
          Go to Sign In
        </Button>
      </Stack>
    );
  }

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
        Create your account
      </Typography>

      {error && (
        <Alert severity="error" sx={{ width: '100%' }}>
          {error}
        </Alert>
      )}

      {/* Google Sign Up */}
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

      {/* Registration Form */}
      <Box component="form" onSubmit={handleRegister} sx={{ width: '100%' }}>
        <Stack spacing={2}>
          <TextField
            label="Name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            fullWidth
            autoComplete="name"
          />
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
            autoComplete="new-password"
            helperText="At least 8 characters"
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
          <TextField
            label="Confirm Password"
            type={showConfirmPassword ? 'text' : 'password'}
            value={confirmPassword}
            onChange={(e) => setConfirmPassword(e.target.value)}
            required
            fullWidth
            autoComplete="new-password"
            error={confirmPasswordError}
            helperText={confirmPasswordError ? 'Passwords do not match' : undefined}
            slotProps={{
              input: {
                endAdornment: (
                  <InputAdornment position="end">
                    <IconButton
                      onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                      edge="end"
                      size="small"
                    >
                      {showConfirmPassword ? <VisibilityOff /> : <Visibility />}
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
            {loading ? 'Creating account...' : 'Create Account'}
          </Button>
        </Stack>
      </Box>

      <Typography variant="body2" color="text.secondary">
        Already have an account?{' '}
        <Typography
          component={Link}
          href={signInHref}
          variant="body2"
          sx={{ color: 'primary.main', textDecoration: 'none', '&:hover': { textDecoration: 'underline' } }}
        >
          Sign in
        </Typography>
      </Typography>
    </Stack>
  );
}

export default function RegisterPage() {
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
          <RegisterContent />
        </Suspense>
      </Paper>
    </Box>
  );
}
