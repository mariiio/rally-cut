'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { signOut, useSession } from 'next-auth/react';
import {
  Box,
  Container,
  Typography,
  Stack,
  Paper,
  Grid,
  Button,
  TextField,
  Avatar,
  Chip,
  Divider,
  Alert,
  LinearProgress,
} from '@mui/material';
import PersonIcon from '@mui/icons-material/Person';
import VerifiedIcon from '@mui/icons-material/Verified';
import LogoutIcon from '@mui/icons-material/Logout';
import DiamondIcon from '@mui/icons-material/Diamond';
import { useTierStore } from '@/stores/tierStore';
import { updateCurrentUser } from '@/services/api';
import { clearAuthToken } from '@/services/authToken';
import { designTokens } from '@/app/theme';
import { AppHeader, PageHeader } from '@/components/dashboard';

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function getProgressColor(used: number, limit: number): 'primary' | 'warning' | 'error' {
  if (limit === 0) return 'primary';
  const ratio = used / limit;
  if (ratio >= 0.95) return 'error';
  if (ratio >= 0.8) return 'warning';
  return 'primary';
}

export default function ProfilePage() {
  const router = useRouter();
  const { data: session, status } = useSession();
  const tier = useTierStore((s) => s.tier);
  const usage = useTierStore((s) => s.usage);
  const limits = useTierStore((s) => s.limits);
  const fetchTier = useTierStore((s) => s.fetchTier);

  const [name, setName] = useState('');
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (status === 'unauthenticated') {
      router.push('/auth/signin');
    }
  }, [status, router]);

  useEffect(() => {
    if (session?.user?.name) {
      setName(session.user.name);
    }
    fetchTier();
  }, [session, fetchTier]);

  const handleSave = async () => {
    setSaving(true);
    setError(null);
    setSaved(false);

    try {
      await updateCurrentUser({ name: name.trim() || undefined });
      setSaved(true);
      setTimeout(() => setSaved(false), 3000);
    } catch {
      setError('Failed to save changes');
    } finally {
      setSaving(false);
    }
  };

  const handleSignOut = async () => {
    clearAuthToken();
    await signOut({ callbackUrl: '/' });
  };

  if (status === 'loading') {
    return (
      <Box sx={{ height: '100%', bgcolor: designTokens.colors.surface[0], p: 4 }}>
        <LinearProgress />
      </Box>
    );
  }

  if (!session?.user) return null;

  const user = session.user;

  return (
    <Box
      sx={{
        height: '100%',
        overflow: 'auto',
        bgcolor: designTokens.colors.surface[0],
        color: 'text.primary',
        position: 'relative',
        '&::before': {
          content: '""',
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          height: 400,
          background: 'radial-gradient(ellipse at 50% 0%, rgba(255, 107, 74, 0.08) 0%, transparent 70%)',
          pointerEvents: 'none',
        },
      }}
    >
      <AppHeader />
      <Container maxWidth="lg" sx={{ position: 'relative', py: 4 }}>
        <PageHeader
          icon={<PersonIcon />}
          title="Profile"
        />

        <Grid container spacing={3}>
          {/* Left column - User Info */}
          <Grid size={{ xs: 12, md: 5 }}>
            <Paper sx={{ p: 3 }}>
              <Stack spacing={3} alignItems="center">
                {/* Avatar */}
                <Avatar
                  src={user.image ?? undefined}
                  alt={user.name ?? 'User'}
                  sx={{ width: 80, height: 80, bgcolor: 'primary.main', fontSize: 32 }}
                >
                  {(user.name || user.email || 'U')[0].toUpperCase()}
                </Avatar>

                {/* Name & email */}
                <Stack alignItems="center" spacing={0.5}>
                  <Typography variant="h5" fontWeight={700}>
                    {user.name || 'Unnamed'}
                  </Typography>
                  <Stack direction="row" alignItems="center" spacing={0.5}>
                    <Typography variant="body2" color="text.secondary">
                      {user.email}
                    </Typography>
                    <VerifiedIcon sx={{ fontSize: 16, color: 'success.main' }} />
                  </Stack>
                  <Chip
                    icon={tier !== 'FREE' ? <DiamondIcon sx={{ fontSize: 14 }} /> : undefined}
                    label={tier === 'FREE' ? 'Free' : tier === 'PRO' ? 'Pro' : 'Elite'}
                    size="small"
                    variant="outlined"
                    sx={{
                      mt: 0.5,
                      ...(tier !== 'FREE' && {
                        borderColor: 'rgba(255, 209, 102, 0.5)',
                        color: designTokens.colors.tertiary.main,
                        '& .MuiChip-icon': {
                          color: designTokens.colors.tertiary.main,
                        },
                      }),
                    }}
                  />
                </Stack>

                <Divider flexItem />

                {/* Edit Name */}
                <Stack spacing={1} sx={{ width: '100%' }}>
                  <Typography variant="body2" color="text.secondary" fontWeight={500}>
                    Display Name
                  </Typography>
                  <Stack direction="row" spacing={1}>
                    <TextField
                      value={name}
                      onChange={(e) => setName(e.target.value)}
                      size="small"
                      fullWidth
                      placeholder="Your name"
                    />
                    <Button
                      variant="contained"
                      onClick={handleSave}
                      disabled={saving || name === (user.name || '')}
                      sx={{ minWidth: 80 }}
                    >
                      {saving ? 'Saving...' : 'Save'}
                    </Button>
                  </Stack>
                  {saved && (
                    <Alert severity="success" sx={{ py: 0 }}>
                      Changes saved
                    </Alert>
                  )}
                  {error && (
                    <Alert severity="error" sx={{ py: 0 }}>
                      {error}
                    </Alert>
                  )}
                </Stack>
              </Stack>
            </Paper>
          </Grid>

          {/* Right column */}
          <Grid size={{ xs: 12, md: 7 }}>
            <Stack spacing={3}>
              {/* Usage & Subscription */}
              <Paper sx={{ p: 3 }}>
                <Stack spacing={3}>
                  <Stack direction="row" alignItems="center" justifyContent="space-between">
                    <Typography variant="h6" fontWeight={600}>
                      Usage & Subscription
                    </Typography>
                    {tier === 'FREE' && (
                      <Button
                        variant="outlined"
                        size="small"
                        sx={{
                          borderColor: 'rgba(255, 209, 102, 0.5)',
                          color: designTokens.colors.tertiary.main,
                          '&:hover': {
                            borderColor: designTokens.colors.tertiary.main,
                            bgcolor: 'rgba(255, 209, 102, 0.08)',
                          },
                        }}
                      >
                        Upgrade
                      </Button>
                    )}
                  </Stack>

                  {/* AI Detections */}
                  <Stack spacing={1}>
                    <Stack direction="row" justifyContent="space-between">
                      <Typography variant="body2" color="text.secondary">
                        AI Detections
                      </Typography>
                      <Typography variant="body2" fontWeight={500}>
                        {usage.detectionsUsed} / {limits.detectionsPerMonth}
                      </Typography>
                    </Stack>
                    <LinearProgress
                      variant="determinate"
                      value={limits.detectionsPerMonth > 0 ? Math.min((usage.detectionsUsed / limits.detectionsPerMonth) * 100, 100) : 0}
                      color={getProgressColor(usage.detectionsUsed, limits.detectionsPerMonth)}
                      sx={{ height: 8, borderRadius: 1 }}
                    />
                  </Stack>

                  {/* Monthly Uploads */}
                  <Stack spacing={1}>
                    <Stack direction="row" justifyContent="space-between">
                      <Typography variant="body2" color="text.secondary">
                        Monthly Uploads
                      </Typography>
                      <Typography variant="body2" fontWeight={500}>
                        {usage.uploadsThisMonth} / {limits.monthlyUploadCount}
                      </Typography>
                    </Stack>
                    <LinearProgress
                      variant="determinate"
                      value={limits.monthlyUploadCount > 0 ? Math.min((usage.uploadsThisMonth / limits.monthlyUploadCount) * 100, 100) : 0}
                      color={getProgressColor(usage.uploadsThisMonth, limits.monthlyUploadCount)}
                      sx={{ height: 8, borderRadius: 1 }}
                    />
                  </Stack>

                  {/* Storage */}
                  <Stack spacing={1}>
                    <Stack direction="row" justifyContent="space-between">
                      <Typography variant="body2" color="text.secondary">
                        Storage
                      </Typography>
                      <Typography variant="body2" fontWeight={500}>
                        {formatBytes(usage.storageUsedBytes)} / {formatBytes(usage.storageLimitBytes)}
                      </Typography>
                    </Stack>
                    <LinearProgress
                      variant="determinate"
                      value={usage.storageLimitBytes > 0 ? Math.min((usage.storageUsedBytes / usage.storageLimitBytes) * 100, 100) : 0}
                      color={getProgressColor(usage.storageUsedBytes, usage.storageLimitBytes)}
                      sx={{ height: 8, borderRadius: 1 }}
                    />
                  </Stack>
                </Stack>
              </Paper>

              {/* Account */}
              <Paper sx={{ p: 3 }}>
                <Stack spacing={2}>
                  <Typography variant="h6" fontWeight={600}>
                    Account
                  </Typography>
                  <Button
                    variant="outlined"
                    color="error"
                    startIcon={<LogoutIcon />}
                    onClick={handleSignOut}
                    sx={{ alignSelf: 'flex-start' }}
                  >
                    Sign Out
                  </Button>
                </Stack>
              </Paper>
            </Stack>
          </Grid>
        </Grid>
      </Container>
    </Box>
  );
}
