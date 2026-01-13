'use client';

import { useState, useRef } from 'react';
import {
  Box,
  Container,
  Typography,
  Grid,
  Paper,
  Button,
  Stack,
  Chip,
  Switch,
  FormControlLabel,
  Divider,
} from '@mui/material';
import { motion, useInView, useReducedMotion } from 'framer-motion';
import CheckIcon from '@mui/icons-material/Check';
import Link from 'next/link';
import { designTokens } from '@/app/theme';
import { WaitlistModal } from './WaitlistModal';

const plans = [
  {
    name: 'Basic',
    subtitle: 'Perfect for trying out',
    price: { monthly: 0, yearly: 0 },
    features: [
      '2 AI detections/month',
      '3 uploads/month',
      'Up to 500 MB per video',
      'Up to 15 min per video',
      '1 GB storage cap',
      'Full quality exports for 3 days',
      '720p exports after 3 days',
      'Watermark on exports',
      'Browser-based export',
      'Local storage only',
    ],
    cta: 'Start Free',
    href: '/sessions',
    highlighted: false,
  },
  {
    name: 'Pro',
    subtitle: 'For regular players',
    price: { monthly: 9.99, yearly: 95.9 },
    features: [
      '15 AI detections/month',
      '20 uploads/month',
      'Up to 2 GB per video',
      'Up to 45 min per video',
      '20 GB storage cap',
      'Original quality for 14 days',
      'No watermark',
      'Fast server-side export',
      'Cross-device sync',
      '6 months video retention',
    ],
    cta: 'Join Waitlist',
    href: null,
    highlighted: true,
    badge: 'Most Popular',
  },
  {
    name: 'Elite',
    subtitle: 'For coaches & teams',
    price: { monthly: 24.99, yearly: 239.9 },
    features: [
      '50 AI detections/month',
      '50 uploads/month',
      'Up to 5 GB per video',
      'Up to 90 min per video',
      '75 GB storage cap',
      'Original quality for 60 days',
      'No watermark',
      'Fast server-side export',
      'Cross-device sync',
      '1 year video retention',
    ],
    cta: 'Join Waitlist',
    href: null,
    highlighted: false,
  },
];

export function Pricing() {
  const [yearly, setYearly] = useState(false);
  const [waitlistOpen, setWaitlistOpen] = useState(false);
  const [selectedTier, setSelectedTier] = useState('pro');
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: '-80px' });
  const shouldReduceMotion = useReducedMotion();

  const handleWaitlistOpen = (tier: string) => {
    setSelectedTier(tier);
    setWaitlistOpen(true);
  };

  return (
    <Box
      component="section"
      id="pricing"
      ref={ref}
      sx={{
        py: { xs: 8, md: 12 },
        position: 'relative',
      }}
    >
      <Container maxWidth="lg">
        <motion.div
          initial={shouldReduceMotion ? {} : { opacity: 0, y: 30 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
        >
          <Box sx={{ textAlign: 'center', mb: 6 }}>
          <Typography
            variant="h2"
            sx={{
              fontSize: { xs: '2rem', md: '2.5rem' },
              fontWeight: 700,
              mb: 2,
            }}
          >
            Simple Pricing for Every Player
          </Typography>
          <Typography
            variant="body1"
            color="text.secondary"
            sx={{ maxWidth: 600, mx: 'auto', mb: 4 }}
          >
            Start free, upgrade when you need more. No hidden fees.
          </Typography>

          {/* Yearly Toggle */}
          <FormControlLabel
            control={
              <Switch
                checked={yearly}
                onChange={(e) => setYearly(e.target.checked)}
                sx={{
                  '& .MuiSwitch-switchBase.Mui-checked': {
                    color: 'primary.main',
                  },
                  '& .MuiSwitch-switchBase.Mui-checked + .MuiSwitch-track': {
                    bgcolor: 'primary.main',
                  },
                }}
              />
            }
            label={
              <Stack direction="row" alignItems="center" spacing={1}>
                <Typography>Yearly billing</Typography>
                <Chip
                  label="Save 20%"
                  size="small"
                  sx={{
                    bgcolor: 'secondary.main',
                    color: 'black',
                    fontWeight: 600,
                    fontSize: '0.7rem',
                  }}
                />
              </Stack>
            }
          />
        </Box>
        </motion.div>

        <Grid container spacing={3} justifyContent="center">
          {plans.map((plan, index) => (
            <Grid size={{ xs: 12, sm: 6, md: 4 }} key={plan.name}>
              <motion.div
                initial={shouldReduceMotion ? {} : { opacity: 0, y: 40 }}
                animate={isInView ? { opacity: 1, y: 0 } : {}}
                transition={{
                  duration: 0.6,
                  delay: index * 0.15,
                  ease: [0.22, 1, 0.36, 1],
                }}
                style={{ height: '100%' }}
              >
              <Paper
                elevation={0}
                sx={{
                  p: 3,
                  height: '100%',
                  bgcolor: plan.highlighted
                    ? designTokens.colors.surface[2]
                    : designTokens.colors.surface[1],
                  border: '2px solid',
                  borderColor: plan.highlighted ? 'primary.main' : 'divider',
                  borderRadius: 3,
                  position: 'relative',
                  transition: 'all 0.3s ease',
                  '&:hover': {
                    transform: 'translateY(-4px)',
                    boxShadow: plan.highlighted
                      ? designTokens.shadows.glow.primary
                      : designTokens.shadows.lg,
                  },
                }}
              >
                {/* Badge */}
                {plan.badge && (
                  <Chip
                    label={plan.badge}
                    sx={{
                      position: 'absolute',
                      top: -12,
                      right: 24,
                      bgcolor: 'primary.main',
                      color: 'white',
                      fontWeight: 600,
                    }}
                  />
                )}

                <Stack spacing={2.5}>
                  {/* Plan Name */}
                  <Box>
                    <Typography variant="h5" fontWeight={700}>
                      {plan.name}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {plan.subtitle}
                    </Typography>
                  </Box>

                  {/* Price */}
                  <Box>
                    <Stack direction="row" alignItems="baseline" spacing={0.5}>
                      <Typography
                        variant="h3"
                        sx={{
                          fontWeight: 800,
                          fontSize: { xs: '2rem', md: '2.5rem' },
                          background: plan.highlighted
                            ? designTokens.gradients.primary
                            : 'inherit',
                          backgroundClip: plan.highlighted ? 'text' : 'inherit',
                          WebkitBackgroundClip: plan.highlighted ? 'text' : 'inherit',
                          WebkitTextFillColor: plan.highlighted ? 'transparent' : 'inherit',
                        }}
                      >
                        ${yearly ? plan.price.yearly : plan.price.monthly}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        {plan.price.monthly > 0 ? (yearly ? '/year' : '/mo') : ''}
                      </Typography>
                    </Stack>
                    {yearly && plan.price.monthly > 0 && (
                      <Typography variant="caption" color="text.disabled">
                        ${(plan.price.yearly / 12).toFixed(2)}/mo billed annually
                      </Typography>
                    )}
                  </Box>

                  <Divider />

                  {/* Features */}
                  <Stack spacing={1}>
                    {plan.features.map((feature) => (
                      <Stack key={feature} direction="row" spacing={1} alignItems="flex-start">
                        <CheckIcon
                          sx={{
                            fontSize: 18,
                            color: plan.highlighted ? 'primary.main' : 'secondary.main',
                            mt: 0.2,
                          }}
                        />
                        <Typography variant="body2" color="text.secondary" sx={{ fontSize: '0.85rem' }}>
                          {feature}
                        </Typography>
                      </Stack>
                    ))}
                  </Stack>

                  {/* CTA */}
                  {plan.href ? (
                    <Button
                      component={Link}
                      href={plan.href}
                      variant={plan.highlighted ? 'contained' : 'outlined'}
                      size="large"
                      fullWidth
                      sx={{
                        py: 1.5,
                        fontWeight: 600,
                        mt: 'auto',
                      }}
                    >
                      {plan.cta}
                    </Button>
                  ) : (
                    <Button
                      variant={plan.highlighted ? 'contained' : 'outlined'}
                      size="large"
                      fullWidth
                      onClick={() => handleWaitlistOpen(plan.name.toLowerCase())}
                      sx={{
                        py: 1.5,
                        fontWeight: 600,
                        mt: 'auto',
                      }}
                    >
                      {plan.cta}
                    </Button>
                  )}
                </Stack>
              </Paper>
              </motion.div>
            </Grid>
          ))}
        </Grid>

        {/* Pay per match callout */}
        <Box
          sx={{
            mt: 6,
            p: 3,
            borderRadius: 2,
            bgcolor: designTokens.colors.surface[1],
            border: '1px solid',
            borderColor: 'divider',
            textAlign: 'center',
          }}
        >
          <Typography variant="h6" sx={{ mb: 1 }}>
            Need just a few extra matches?
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Pay-per-match credits:{' '}
            <Box component="span" sx={{ color: 'primary.main', fontWeight: 600 }}>
              $0.99 per match
            </Box>{' '}
            (up to 30 minutes). Perfect for Basic tier users who need occasional AI detections.
          </Typography>
        </Box>
      </Container>

      <WaitlistModal
        open={waitlistOpen}
        onClose={() => setWaitlistOpen(false)}
        tier={selectedTier}
      />
    </Box>
  );
}
