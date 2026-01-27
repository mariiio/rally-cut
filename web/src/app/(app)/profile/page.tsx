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
  Alert,
  LinearProgress,
} from '@mui/material';
import VerifiedIcon from '@mui/icons-material/Verified';
import LogoutIcon from '@mui/icons-material/Logout';
import DiamondIcon from '@mui/icons-material/Diamond';
import VideocamOutlinedIcon from '@mui/icons-material/VideocamOutlined';
import FolderOutlinedIcon from '@mui/icons-material/FolderOutlined';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import CloudOutlinedIcon from '@mui/icons-material/CloudOutlined';
import CalendarTodayIcon from '@mui/icons-material/CalendarToday';
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutline';
import { useTierStore } from '@/stores/tierStore';
import { getCurrentUser, updateCurrentUser } from '@/services/api';
import type { UserResponse } from '@/services/api';
import { clearAuthToken } from '@/services/authToken';
import { designTokens } from '@/app/theme';
import { AppHeader } from '@/components/dashboard';

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function formatBytesShort(bytes: number): string {
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(0)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
}

const TIER_COLORS = {
  FREE: { accent: designTokens.colors.surface[3], glow: 'none', gradient: designTokens.gradients.toolbar, border: 'divider', chipBg: designTokens.colors.surface[3], chipColor: '#A1A7B4', radialOpacity: 0.05 },
  PRO: { accent: '#FF6B4A', glow: '0 0 16px rgba(255, 107, 74, 0.4)', gradient: designTokens.gradients.primary, border: 'rgba(255, 107, 74, 0.3)', chipBg: designTokens.gradients.primary, chipColor: '#fff', radialOpacity: 0.08 },
  ELITE: { accent: '#FFD166', glow: '0 0 16px rgba(255, 209, 102, 0.4)', gradient: designTokens.gradients.tertiary, border: 'rgba(255, 209, 102, 0.3)', chipBg: designTokens.gradients.tertiary, chipColor: '#0D0E12', radialOpacity: 0.08 },
} as const;

function getTierFeatures(limits: {
  maxVideoDurationMs: number;
  maxFileSizeBytes: number;
  exportQuality: string;
  exportWatermark: boolean;
  serverSyncEnabled: boolean;
  originalQualityDays: number | null;
  inactivityDeleteDays: number | null;
}) {
  const durationMin = Math.round(limits.maxVideoDurationMs / 60000);
  const fileSizeGB = limits.maxFileSizeBytes / (1024 * 1024 * 1024);
  const fileSize = fileSizeGB >= 1 ? `${fileSizeGB} GB` : `${Math.round(limits.maxFileSizeBytes / (1024 * 1024))} MB`;
  const inactivityDays = limits.inactivityDeleteDays ?? 0;
  const retentionLabel = inactivityDays >= 365 ? '1 year' : inactivityDays >= 30 ? `${Math.round(inactivityDays / 30)} months` : `${inactivityDays} days`;
  const qualityDays = limits.originalQualityDays;
  return [
    { label: `${durationMin} min per video` },
    { label: `${fileSize} max file size` },
    { label: `${limits.exportQuality === 'original' ? 'Original' : '720p'} export quality${limits.exportWatermark ? ' (watermark)' : ''}` },
    { label: `Cloud sync: ${limits.serverSyncEnabled ? 'Yes' : 'No'}` },
    { label: `Original quality: ${qualityDays != null ? `${qualityDays} days` : 'Forever'}` },
    { label: `Data retention: ${limits.inactivityDeleteDays != null ? `${retentionLabel} inactive` : 'Forever'}` },
  ];
}

function getProgressGradient(ratio: number) {
  if (ratio >= 0.95) return { gradient: 'linear-gradient(90deg, #EF4444 0%, #F87171 100%)', glow: '0 0 8px rgba(239, 68, 68, 0.5)' };
  if (ratio >= 0.8) return { gradient: 'linear-gradient(90deg, #F59E0B 0%, #FBBF24 100%)', glow: '0 0 8px rgba(245, 158, 11, 0.5)' };
  return { gradient: designTokens.gradients.primary, glow: '0 0 8px rgba(255, 107, 74, 0.4)' };
}

function getResetDate(): string {
  const now = new Date();
  const next = new Date(now.getFullYear(), now.getMonth() + 1, 1);
  return next.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
}

function formatMemberSince(dateStr: string): string {
  const d = new Date(dateStr);
  return d.toLocaleDateString('en-US', { month: 'short', year: 'numeric' });
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
  const [userData, setUserData] = useState<UserResponse | null>(null);

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
    getCurrentUser().then(setUserData).catch(console.error);
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
  const tc = TIER_COLORS[tier] ?? TIER_COLORS.FREE;
  const tierLabel = tier === 'FREE' ? 'Free' : tier === 'PRO' ? 'Pro' : 'Elite';
  const isPaid = tier === 'PRO' || tier === 'ELITE';

  const statsCards = [
    { icon: <VideocamOutlinedIcon />, color: '#00D4AA', label: 'Videos', value: userData?.videoCount ?? '—', context: 'total' },
    { icon: <FolderOutlinedIcon />, color: '#FF6B4A', label: 'Sessions', value: userData?.sessionCount ?? '—', context: 'total' },
    { icon: <AutoAwesomeIcon />, color: '#FFD166', label: 'AI Detections', value: usage.detectionsUsed, context: `of ${limits.detectionsPerMonth} this month` },
    { icon: <CloudOutlinedIcon />, color: '#3B82F6', label: 'Storage', value: formatBytesShort(usage.storageUsedBytes), context: `of ${formatBytesShort(usage.storageLimitBytes)}` },
  ];

  const usageItems = [
    { label: 'AI Detections', used: usage.detectionsUsed, limit: limits.detectionsPerMonth, remaining: usage.detectionsRemaining },
    { label: 'Monthly Uploads', used: usage.uploadsThisMonth, limit: limits.monthlyUploadCount, remaining: usage.uploadsRemaining },
    { label: 'Storage', used: usage.storageUsedBytes, limit: usage.storageLimitBytes, remaining: usage.storageRemainingBytes, isBytes: true },
  ];

  const features = getTierFeatures(limits);

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
          background: tier === 'ELITE'
            ? `radial-gradient(ellipse at 50% 0%, rgba(255, 209, 102, ${tc.radialOpacity}) 0%, transparent 70%)`
            : `radial-gradient(ellipse at 50% 0%, rgba(255, 107, 74, ${tc.radialOpacity}) 0%, transparent 70%)`,
          pointerEvents: 'none',
        },
      }}
    >
      <AppHeader />
      <Container maxWidth="lg" sx={{ position: 'relative', py: 4 }}>
        <Typography variant="h4" sx={{ fontWeight: 700, letterSpacing: '-0.02em', mb: 3, py: 1 }}>
          Profile
        </Typography>

        {/* ─── Hero Profile Card ─── */}
        <Paper
          sx={{
            p: 0,
            mb: 3,
            overflow: 'hidden',
            position: 'relative',
            borderColor: tc.border,
            '&::before': {
              content: '""',
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              height: 4,
              background: tc.gradient,
            },
            ...(isPaid && {
              '&::after': {
                content: '""',
                position: 'absolute',
                top: 4,
                left: '50%',
                transform: 'translateX(-50%)',
                width: '60%',
                height: 80,
                background: tier === 'ELITE'
                  ? 'radial-gradient(ellipse, rgba(255, 209, 102, 0.06) 0%, transparent 70%)'
                  : 'radial-gradient(ellipse, rgba(255, 107, 74, 0.06) 0%, transparent 70%)',
                pointerEvents: 'none',
              },
            }),
          }}
        >
          <Box
            sx={{
              p: { xs: 3, md: 4 },
              pt: { xs: 4, md: 5 },
              display: 'flex',
              flexDirection: { xs: 'column', md: 'row' },
              alignItems: { xs: 'center', md: 'flex-start' },
              gap: { xs: 3, md: 4 },
            }}
          >
            {/* Avatar */}
            <Avatar
              src={user.image ?? undefined}
              alt={user.name ?? 'User'}
              sx={{
                width: { xs: 72, md: 96 },
                height: { xs: 72, md: 96 },
                bgcolor: 'primary.main',
                fontSize: { xs: 28, md: 36 },
                border: '3px solid',
                borderColor: tc.accent,
                boxShadow: tc.glow,
                flexShrink: 0,
              }}
            >
              {(user.name || user.email || 'U')[0].toUpperCase()}
            </Avatar>

            {/* Identity */}
            <Stack
              spacing={0.5}
              sx={{
                alignItems: { xs: 'center', md: 'flex-start' },
                textAlign: { xs: 'center', md: 'left' },
                flex: 1,
                minWidth: 0,
              }}
            >
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
                icon={isPaid ? <DiamondIcon sx={{ fontSize: 14 }} /> : undefined}
                label={tierLabel}
                size="small"
                sx={{
                  mt: 0.5,
                  fontWeight: 600,
                  ...(isPaid
                    ? {
                        background: tc.chipBg,
                        color: tc.chipColor,
                        border: 'none',
                        '& .MuiChip-icon': { color: tc.chipColor },
                      }
                    : {
                        bgcolor: tc.chipBg,
                        color: tc.chipColor,
                        border: 'none',
                      }),
                }}
              />
              {userData?.createdAt && (
                <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5 }}>
                  Member since {formatMemberSince(userData.createdAt)}
                </Typography>
              )}
            </Stack>

            {/* Name editor */}
            <Stack
              spacing={1}
              sx={{
                width: { xs: '100%', md: 280 },
                flexShrink: 0,
                alignSelf: { xs: 'stretch', md: 'center' },
              }}
            >
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
          </Box>
        </Paper>

        {/* ─── Stats Row ─── */}
        <Grid container spacing={2} sx={{ mb: 3 }}>
          {statsCards.map((card) => (
            <Grid key={card.label} size={{ xs: 6, md: 3 }}>
              <Paper
                sx={{
                  p: 2.5,
                  transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
                  cursor: 'default',
                  '&:hover': {
                    transform: 'translateY(-2px)',
                    borderColor: 'rgba(255, 255, 255, 0.12)',
                    bgcolor: designTokens.colors.surface[2],
                  },
                }}
              >
                <Box
                  sx={{
                    width: 36,
                    height: 36,
                    borderRadius: 1.5,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    bgcolor: `${card.color}14`,
                    color: card.color,
                    mb: 1.5,
                    '& .MuiSvgIcon-root': { fontSize: 20 },
                  }}
                >
                  {card.icon}
                </Box>
                <Typography variant="caption" color="text.secondary">
                  {card.label}
                </Typography>
                <Typography variant="h5" fontWeight={700} sx={{ mt: 0.25 }}>
                  {card.value}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {card.context}
                </Typography>
              </Paper>
            </Grid>
          ))}
        </Grid>

        {/* ─── Two-Column Content ─── */}
        <Grid container spacing={3}>
          {/* Left: Usage This Month */}
          <Grid size={{ xs: 12, md: 7 }}>
            <Paper sx={{ p: 3 }}>
              <Stack spacing={3}>
                <Box>
                  <Typography variant="overline" color="text.secondary">
                    Plan & Usage
                  </Typography>
                  <Typography variant="h6" fontWeight={600}>
                    Usage This Month
                  </Typography>
                </Box>

                {usageItems.map((item) => {
                  const ratio = item.limit > 0 ? item.used / item.limit : 0;
                  const { gradient, glow } = getProgressGradient(ratio);
                  const usedDisplay = item.isBytes ? formatBytes(item.used) : item.used;
                  const limitDisplay = item.isBytes ? formatBytes(item.limit) : item.limit;
                  const remainingDisplay = item.isBytes ? formatBytes(item.remaining) : item.remaining;
                  return (
                    <Stack key={item.label} spacing={1}>
                      <Stack direction="row" justifyContent="space-between" alignItems="baseline">
                        <Stack>
                          <Typography variant="body2" fontWeight={600}>
                            {item.label}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            {remainingDisplay} remaining
                          </Typography>
                        </Stack>
                        <Typography variant="body2" fontWeight={700}>
                          {usedDisplay} <Box component="span" sx={{ color: 'text.secondary', fontWeight: 400 }}>/</Box> {limitDisplay}
                        </Typography>
                      </Stack>
                      {/* Custom gradient progress bar */}
                      <Box
                        sx={{
                          width: '100%',
                          height: 10,
                          borderRadius: 5,
                          bgcolor: 'rgba(255, 255, 255, 0.08)',
                          overflow: 'hidden',
                          position: 'relative',
                        }}
                      >
                        <Box
                          sx={{
                            height: '100%',
                            borderRadius: 5,
                            width: `${Math.min(ratio * 100, 100)}%`,
                            background: gradient,
                            boxShadow: ratio > 0 ? glow : 'none',
                            transition: 'width 0.6s cubic-bezier(0.4, 0, 0.2, 1)',
                          }}
                        />
                      </Box>
                    </Stack>
                  );
                })}

                <Stack direction="row" alignItems="center" spacing={0.75} sx={{ pt: 0.5 }}>
                  <CalendarTodayIcon sx={{ fontSize: 14, color: 'text.secondary' }} />
                  <Typography variant="caption" color="text.secondary">
                    Usage resets {getResetDate()}
                  </Typography>
                </Stack>
              </Stack>
            </Paper>
          </Grid>

          {/* Right: Plan Details + Account */}
          <Grid size={{ xs: 12, md: 5 }}>
            <Stack spacing={3}>
              {/* Plan Card */}
              <Paper sx={{ p: 3 }}>
                <Stack spacing={2}>
                  <Typography
                    variant="h5"
                    fontWeight={700}
                    sx={
                      isPaid
                        ? {
                            background: tc.chipBg,
                            backgroundClip: 'text',
                            WebkitBackgroundClip: 'text',
                            WebkitTextFillColor: 'transparent',
                          }
                        : undefined
                    }
                  >
                    {tierLabel} Plan
                  </Typography>

                  <Stack spacing={1.5}>
                    {features.map((f) => (
                      <Stack key={f.label} direction="row" alignItems="center" spacing={1.5}>
                        <CheckCircleOutlineIcon sx={{ fontSize: 18, color: '#00D4AA' }} />
                        <Typography variant="body2">
                          {f.label}
                        </Typography>
                      </Stack>
                    ))}
                  </Stack>

                  {tier === 'FREE' && (
                    <Box
                      sx={{
                        mt: 1,
                        p: 2.5,
                        borderRadius: 2,
                        background: 'linear-gradient(135deg, rgba(255, 107, 74, 0.12) 0%, rgba(255, 209, 102, 0.08) 100%)',
                        border: '1px solid rgba(255, 107, 74, 0.2)',
                      }}
                    >
                      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                        Unlock longer videos, more detections, original quality exports, and cloud sync.
                      </Typography>
                      <Button
                        variant="contained"
                        fullWidth
                        startIcon={<AutoAwesomeIcon />}
                        onClick={() => router.push('/upgrade')}
                      >
                        Upgrade to Pro
                      </Button>
                    </Box>
                  )}
                </Stack>
              </Paper>

              {/* Account Card */}
              <Paper sx={{ p: 3 }}>
                <Stack spacing={2}>
                  <Typography variant="overline" color="text.secondary">
                    Account
                  </Typography>
                  <Stack direction="row" alignItems="center" spacing={1}>
                    <Typography variant="body2" color="text.secondary" sx={{ flex: 1 }}>
                      {user.email}
                    </Typography>
                    <VerifiedIcon sx={{ fontSize: 16, color: 'success.main' }} />
                  </Stack>
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
