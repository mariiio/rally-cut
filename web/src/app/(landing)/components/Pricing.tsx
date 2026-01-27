'use client';

import { useState, useRef } from 'react';
import {
  Box,
  Container,
  Typography,
  Button,
  Stack,
  Chip,
  Switch,
  FormControlLabel,
} from '@mui/material';
import { motion, useInView, useReducedMotion } from 'framer-motion';
import CheckIcon from '@mui/icons-material/Check';
import CloseIcon from '@mui/icons-material/Close';
import Link from 'next/link';
import { designTokens } from '@/app/theme';
import { WaitlistModal } from './WaitlistModal';

const tiers = [
  {
    name: 'Basic',
    subtitle: 'Free forever',
    price: { monthly: 0, yearly: 0 },
    cta: 'Start Free',
    href: '/sessions',
    highlighted: false,
  },
  {
    name: 'Pro',
    subtitle: 'For serious players',
    price: { monthly: 9.99, yearly: 95.9 },
    cta: 'Join Waitlist',
    href: null,
    highlighted: true,
    badge: 'Most Popular',
  },
  {
    name: 'Elite',
    subtitle: 'For coaches & teams',
    price: { monthly: 24.99, yearly: 239.9 },
    cta: 'Join Waitlist',
    href: null,
    highlighted: false,
  },
];

const features = [
  { name: 'AI Detections/month', basic: '2', pro: '15', elite: '50' },
  { name: 'Uploads/month', basic: '5', pro: '20', elite: '50' },
  { name: 'Max video', basic: '30 min (500 MB)', pro: '60 min (3 GB)', elite: '90 min (5 GB)' },
  { name: 'Storage cap', basic: '2 GB', pro: '20 GB', elite: '75 GB' },
  { name: 'Export quality', basic: '720p', pro: 'Original', elite: 'Original' },
  { name: 'No watermark', basic: false, pro: true, elite: true },
  { name: 'Server export', basic: false, pro: true, elite: true },
  { name: 'Cloud sync', basic: false, pro: true, elite: true },
  { name: 'Quality retention', basic: '7 days', pro: '14 days', elite: '60 days' },
  { name: 'Video retention', basic: '90 days', pro: '6 months', elite: '1 year' },
];

function FeatureValue({ value }: { value: string | boolean }) {
  if (typeof value === 'boolean') {
    return value ? (
      <CheckIcon sx={{ fontSize: 20, color: 'secondary.main' }} />
    ) : (
      <CloseIcon sx={{ fontSize: 20, color: 'text.disabled' }} />
    );
  }
  return (
    <Typography variant="body2" sx={{ fontWeight: 500 }}>
      {value}
    </Typography>
  );
}

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

        {/* Comparison Table */}
        <motion.div
          initial={shouldReduceMotion ? {} : { opacity: 0, y: 40 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.6, delay: 0.15, ease: [0.22, 1, 0.36, 1] }}
        >
          <Box
            sx={{
              overflowX: 'auto',
              borderRadius: 3,
              border: '1px solid',
              borderColor: 'divider',
              bgcolor: designTokens.colors.surface[1],
              width: 'fit-content',
              mx: 'auto',
            }}
          >
            <Box
              sx={{
                display: 'grid',
                gridTemplateColumns: { xs: '150px 110px 110px 110px', md: '200px 180px 180px 180px' },
              }}
            >
              {/* Header Row */}
              <Box
                sx={{
                  display: 'contents',
                  '& > *': {
                    bgcolor: designTokens.colors.surface[2],
                  },
                  '& > *:first-of-type': {
                    borderRadius: '12px 0 0 0',
                  },
                  '& > *:last-child': {
                    borderRadius: '0 12px 0 0',
                  },
                }}
              >
                <Box sx={{ py: 2, px: 2, borderBottom: '1px solid', borderColor: 'divider' }} />
                {tiers.map((tier) => (
                  <Box
                    key={tier.name}
                    sx={{
                      py: 1.5,
                      px: 2,
                      textAlign: 'center',
                      borderLeft: '1px solid',
                      borderBottom: '1px solid',
                      borderColor: 'divider',
                      bgcolor: tier.highlighted ? 'rgba(255, 107, 74, 0.08)' : 'transparent',
                      display: 'flex',
                      flexDirection: 'column',
                      justifyContent: 'center',
                    }}
                  >
                    {tier.badge ? (
                      <Chip
                        label={tier.badge}
                        size="small"
                        sx={{
                          alignSelf: 'center',
                          mb: 0.5,
                          bgcolor: 'primary.main',
                          color: 'white',
                          fontWeight: 600,
                          fontSize: '0.6rem',
                          height: 18,
                        }}
                      />
                    ) : (
                      <Box sx={{ height: 18, mb: 0.5 }} />
                    )}
                    <Typography variant="subtitle1" fontWeight={700} lineHeight={1.2}>
                      {tier.name}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {tier.subtitle}
                    </Typography>
                  </Box>
                ))}
              </Box>

              {/* Price Row */}
              <Box
                sx={{
                  display: 'contents',
                }}
              >
                <Box sx={{ p: 2, display: 'flex', alignItems: 'center', borderBottom: '1px solid', borderColor: 'divider' }}>
                  <Typography variant="body2" fontWeight={600}>
                    Price
                  </Typography>
                </Box>
                {tiers.map((tier) => (
                  <Box
                    key={tier.name}
                    sx={{
                      p: 2,
                      textAlign: 'center',
                      borderLeft: '1px solid',
                      borderBottom: '1px solid',
                      borderColor: 'divider',
                      bgcolor: tier.highlighted ? 'rgba(255, 107, 74, 0.08)' : 'transparent',
                    }}
                  >
                    <Stack direction="row" justifyContent="center" alignItems="baseline" spacing={0.5}>
                      <Typography
                        variant="h4"
                        sx={{
                          fontWeight: 800,
                          background: tier.highlighted ? designTokens.gradients.primary : 'inherit',
                          backgroundClip: tier.highlighted ? 'text' : 'inherit',
                          WebkitBackgroundClip: tier.highlighted ? 'text' : 'inherit',
                          WebkitTextFillColor: tier.highlighted ? 'transparent' : 'inherit',
                        }}
                      >
                        ${yearly ? tier.price.yearly : tier.price.monthly}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        {tier.price.monthly > 0 ? (yearly ? '/yr' : '/mo') : ''}
                      </Typography>
                    </Stack>
                    {yearly && tier.price.monthly > 0 && (
                      <Typography variant="caption" color="text.disabled">
                        ${(tier.price.yearly / 12).toFixed(2)}/mo
                      </Typography>
                    )}
                  </Box>
                ))}
              </Box>

              {/* Feature Rows */}
              {features.map((feature) => (
                <Box
                  key={feature.name}
                  sx={{
                    display: 'contents',
                  }}
                >
                  <Box sx={{ py: 1.5, px: 2, display: 'flex', alignItems: 'center', borderBottom: '1px solid', borderColor: 'divider' }}>
                    <Typography variant="body2" color="text.secondary" fontSize="0.85rem">
                      {feature.name}
                    </Typography>
                  </Box>
                  {(['basic', 'pro', 'elite'] as const).map((tierKey, tierIndex) => (
                    <Box
                      key={tierKey}
                      sx={{
                        py: 1.5,
                        px: 2,
                        display: 'flex',
                        justifyContent: 'center',
                        alignItems: 'center',
                        borderLeft: '1px solid',
                        borderBottom: '1px solid',
                        borderColor: 'divider',
                        bgcolor: tiers[tierIndex].highlighted ? 'rgba(255, 107, 74, 0.08)' : 'transparent',
                      }}
                    >
                      <FeatureValue value={feature[tierKey]} />
                    </Box>
                  ))}
                </Box>
              ))}

              {/* CTA Row */}
              <Box
                sx={{
                  display: 'contents',
                  '& > *': {
                    bgcolor: designTokens.colors.surface[2],
                  },
                }}
              >
                <Box sx={{ p: 2 }} />
                {tiers.map((tier) => (
                  <Box
                    key={tier.name}
                    sx={{
                      p: 2,
                      display: 'flex',
                      justifyContent: 'center',
                      borderLeft: '1px solid',
                      borderColor: 'divider',
                      bgcolor: tier.highlighted ? 'rgba(255, 107, 74, 0.08)' : 'transparent',
                    }}
                  >
                    {tier.href ? (
                      <Button
                        component={Link}
                        href={tier.href}
                        variant={tier.highlighted ? 'contained' : 'outlined'}
                        size="small"
                        sx={{ fontWeight: 600, fontSize: '0.8rem' }}
                      >
                        {tier.cta}
                      </Button>
                    ) : (
                      <Button
                        variant={tier.highlighted ? 'contained' : 'outlined'}
                        size="small"
                        onClick={() => handleWaitlistOpen(tier.name.toLowerCase())}
                        sx={{ fontWeight: 600, fontSize: '0.8rem' }}
                      >
                        {tier.cta}
                      </Button>
                    )}
                  </Box>
                ))}
              </Box>
            </Box>
          </Box>
        </motion.div>

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
            </Box>
            . No subscription needed — just pay for the matches you want analyzed.
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
