'use client';

import { useState } from 'react';
import {
  Box,
  Container,
  Typography,
  Grid,
  Paper,
  Stack,
  Button,
  Switch,
  Chip,
} from '@mui/material';
import CheckIcon from '@mui/icons-material/Check';
import CloseIcon from '@mui/icons-material/Close';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import BoltIcon from '@mui/icons-material/Bolt';
import { designTokens } from '@/app/theme';
import { useTierStore } from '@/stores/tierStore';
import { AppHeader } from '@/components/dashboard';

interface TierFeature {
  label: string;
  included: boolean;
}

interface TierData {
  name: string;
  key: 'FREE' | 'PRO' | 'ELITE';
  subtitle: string;
  monthlyPrice: number;
  yearlyPrice: number;
  badge?: string;
  highlighted?: boolean;
  features: TierFeature[];
}

const TIERS: TierData[] = [
  {
    name: 'Basic',
    key: 'FREE',
    subtitle: 'Free forever',
    monthlyPrice: 0,
    yearlyPrice: 0,
    features: [
      { label: '2 AI detections / month', included: true },
      { label: '5 uploads / month', included: true },
      { label: '30 min, 500 MB max video', included: true },
      { label: '2 GB storage', included: true },
      { label: '720p export with watermark', included: true },
      { label: 'Cloud export', included: false },
      { label: 'Cloud sync', included: false },
    ],
  },
  {
    name: 'Pro',
    key: 'PRO',
    subtitle: 'For serious players',
    monthlyPrice: 9.99,
    yearlyPrice: 95.90,
    badge: 'Most Popular',
    highlighted: true,
    features: [
      { label: '15 AI detections / month', included: true },
      { label: '20 uploads / month', included: true },
      { label: '60 min, 3 GB max video', included: true },
      { label: '20 GB storage', included: true },
      { label: 'Original quality export', included: true },
      { label: 'Cloud export', included: true },
      { label: 'Cloud sync', included: true },
    ],
  },
  {
    name: 'Elite',
    key: 'ELITE',
    subtitle: 'For coaches & teams',
    monthlyPrice: 24.99,
    yearlyPrice: 239.90,
    features: [
      { label: '40 AI detections / month', included: true },
      { label: '50 uploads / month', included: true },
      { label: '90 min, 5 GB max video', included: true },
      { label: '75 GB storage', included: true },
      { label: 'Original quality export', included: true },
      { label: 'Cloud export', included: true },
      { label: 'Cloud sync', included: true },
    ],
  },
];

export default function UpgradePage() {
  const [yearly, setYearly] = useState(false);
  const tier = useTierStore((s) => s.tier);

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
      <Container maxWidth="md" sx={{ position: 'relative', py: 4 }}>
        {/* Header */}
        <Box sx={{ textAlign: 'center', mb: 5 }}>
          <Typography variant="h3" sx={{ fontWeight: 700, letterSpacing: '-0.02em', mb: 1 }}>
            Upgrade your plan
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Get more detections, storage, and pro exports.
          </Typography>
        </Box>

        {/* Billing toggle */}
        <Stack direction="row" alignItems="center" justifyContent="center" spacing={1.5} sx={{ mb: 5 }}>
          <Typography
            variant="body2"
            sx={{ fontWeight: !yearly ? 600 : 400, color: !yearly ? 'text.primary' : 'text.secondary' }}
          >
            Monthly
          </Typography>
          <Switch
            checked={yearly}
            onChange={(e) => setYearly(e.target.checked)}
            size="small"
          />
          <Typography
            variant="body2"
            sx={{ fontWeight: yearly ? 600 : 400, color: yearly ? 'text.primary' : 'text.secondary' }}
          >
            Yearly
          </Typography>
          <Chip
            label="Save 20%"
            size="small"
            sx={{
              bgcolor: 'rgba(0, 212, 170, 0.15)',
              color: 'secondary.main',
              fontWeight: 600,
              fontSize: '0.7rem',
            }}
          />
        </Stack>

        {/* Tier cards */}
        <Grid container spacing={3} sx={{ mb: 5 }}>
          {TIERS.map((t) => {
            const isCurrent = tier === t.key;
            const price = yearly ? t.yearlyPrice : t.monthlyPrice;
            const period = yearly ? '/yr' : '/mo';

            return (
              <Grid size={{ xs: 12, md: 4 }} key={t.key}>
                <Paper
                  sx={{
                    p: 3,
                    position: 'relative',
                    height: '100%',
                    display: 'flex',
                    flexDirection: 'column',
                    ...(t.highlighted && {
                      border: '1px solid rgba(255, 107, 74, 0.4)',
                      bgcolor: 'rgba(255, 107, 74, 0.04)',
                    }),
                  }}
                >
                  {/* Badge */}
                  {t.badge && (
                    <Chip
                      label={t.badge}
                      size="small"
                      icon={<AutoAwesomeIcon sx={{ fontSize: 14 }} />}
                      sx={{
                        position: 'absolute',
                        top: -12,
                        left: '50%',
                        transform: 'translateX(-50%)',
                        background: designTokens.gradients.primary,
                        color: 'white',
                        fontWeight: 600,
                        fontSize: '0.7rem',
                        '& .MuiChip-icon': { color: 'white' },
                      }}
                    />
                  )}

                  {/* Tier name */}
                  <Typography variant="h5" sx={{ fontWeight: 700, mt: t.badge ? 1 : 0 }}>
                    {t.name}
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                    {t.subtitle}
                  </Typography>

                  {/* Price */}
                  <Stack direction="row" alignItems="baseline" spacing={0.5} sx={{ mb: 3 }}>
                    <Typography
                      variant="h3"
                      sx={{
                        fontWeight: 700,
                        ...(t.highlighted && {
                          background: designTokens.gradients.primary,
                          backgroundClip: 'text',
                          WebkitBackgroundClip: 'text',
                          WebkitTextFillColor: 'transparent',
                        }),
                      }}
                    >
                      {price === 0 ? '$0' : `$${price.toFixed(2)}`}
                    </Typography>
                    {price > 0 && (
                      <Typography variant="body2" color="text.secondary">
                        {period}
                      </Typography>
                    )}
                    {price === 0 && (
                      <Typography variant="body2" color="text.secondary">
                        forever
                      </Typography>
                    )}
                  </Stack>

                  {/* CTA button */}
                  {isCurrent ? (
                    <Button variant="outlined" disabled fullWidth sx={{ mb: 3 }}>
                      Current Plan
                    </Button>
                  ) : t.key === 'FREE' ? (
                    <Button variant="outlined" disabled fullWidth sx={{ mb: 3 }}>
                      Free Tier
                    </Button>
                  ) : (
                    <Button
                      variant="contained"
                      fullWidth
                      disabled
                      sx={{ mb: 3 }}
                    >
                      Coming Soon
                    </Button>
                  )}

                  {/* Features */}
                  <Stack spacing={1.5} sx={{ flex: 1 }}>
                    {t.features.map((f) => (
                      <Stack direction="row" alignItems="center" spacing={1} key={f.label}>
                        {f.included ? (
                          <CheckIcon sx={{ fontSize: 18, color: 'secondary.main' }} />
                        ) : (
                          <CloseIcon sx={{ fontSize: 18, color: 'text.disabled' }} />
                        )}
                        <Typography
                          variant="body2"
                          sx={{ color: f.included ? 'text.primary' : 'text.disabled' }}
                        >
                          {f.label}
                        </Typography>
                      </Stack>
                    ))}
                  </Stack>
                </Paper>
              </Grid>
            );
          })}
        </Grid>

        {/* Pay-per-match callout */}
        <Paper
          sx={{
            p: 3,
            textAlign: 'center',
            border: '1px solid',
            borderColor: 'divider',
          }}
        >
          <Stack direction="row" alignItems="center" justifyContent="center" spacing={1.5}>
            <BoltIcon sx={{ color: 'warning.main', fontSize: 24 }} />
            <Box>
              <Typography variant="body1" fontWeight={600}>
                Pay-per-match credits
              </Typography>
              <Typography variant="body2" color="text.secondary">
                $0.99 per match for occasional AI detection use — no subscription needed.
              </Typography>
            </Box>
          </Stack>
        </Paper>

        {/* Custom tier callout */}
        <Paper
          sx={{
            mt: 3,
            p: 3,
            textAlign: 'center',
            border: '1px solid',
            borderColor: 'divider',
          }}
        >
          <Typography variant="body1" fontWeight={600}>
            Need more for your club or academy?
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Higher limits, priority support, and custom features — let&apos;s build a plan that fits.{' '}
            <Box
              component="a"
              href="mailto:support@rallycut.com"
              sx={{ color: 'primary.main', fontWeight: 600, textDecoration: 'none', '&:hover': { textDecoration: 'underline' } }}
            >
              Contact us
            </Box>
          </Typography>
        </Paper>
      </Container>
    </Box>
  );
}
