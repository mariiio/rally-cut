'use client';

import { Box, Container, Typography, Button, Stack } from '@mui/material';
import { motion, useReducedMotion } from 'framer-motion';
import Link from 'next/link';
import { designTokens } from '@/app/theme';
import { HeroBackground } from './HeroBackground';
import { PipelineAnimation } from './PipelineAnimation';
import { fadeInUp, staggerContainer } from '../utils/animations';

// Animated gradient text component
function GradientText({ children }: { children: React.ReactNode }) {
  return (
    <Box
      component="span"
      sx={{
        background: designTokens.gradients.primary,
        backgroundClip: 'text',
        WebkitBackgroundClip: 'text',
        WebkitTextFillColor: 'transparent',
        display: 'inline',
      }}
    >
      {children}
    </Box>
  );
}

// Trust badge component
function TrustBadge({ text }: { text: string }) {
  return (
    <Typography
      variant="body2"
      sx={{
        color: 'text.secondary',
        display: 'flex',
        alignItems: 'center',
        gap: 0.75,
        fontSize: '0.9rem',
        '&::before': {
          content: '"✓"',
          color: 'secondary.main',
          fontWeight: 700,
        },
      }}
    >
      {text}
    </Typography>
  );
}

export function Hero() {
  const shouldReduceMotion = useReducedMotion();

  const MotionBox = shouldReduceMotion ? Box : motion.div;

  return (
    <Box
      component="section"
      sx={{
        minHeight: { xs: 'calc(100vh - 56px)', sm: 'calc(100vh - 64px)' },
        display: 'flex',
        position: 'relative',
        overflow: 'hidden',
        py: { xs: 4, md: 5 },
      }}
    >
      {/* Animated background */}
      <HeroBackground />

      <Container maxWidth="lg" sx={{ position: 'relative', zIndex: 1 }}>
        {/* Centered content */}
        <MotionBox
          {...(!shouldReduceMotion && {
            initial: 'hidden',
            animate: 'visible',
            variants: staggerContainer,
          })}
        >
          <Box sx={{ maxWidth: 720, mx: 'auto', textAlign: 'center' }}>
            {/* Headline */}
            <MotionBox {...(!shouldReduceMotion && { variants: fadeInUp })}>
              <Typography
                variant="h1"
                sx={{
                  fontSize: { xs: '2.5rem', sm: '3.25rem', md: '3.75rem', lg: '4.25rem' },
                  fontWeight: 800,
                  lineHeight: 1.08,
                  letterSpacing: '-0.03em',
                  mb: 2,
                  color: 'text.primary',
                }}
              >
                Turn Hours of Footage Into{' '}
                <GradientText>Minutes of Action</GradientText>
              </Typography>
            </MotionBox>

            {/* Subheadline */}
            <MotionBox {...(!shouldReduceMotion && { variants: fadeInUp })}>
              <Typography
                variant="h5"
                sx={{
                  color: 'text.secondary',
                  fontWeight: 400,
                  mb: 2.5,
                  lineHeight: 1.6,
                  fontSize: { xs: '0.95rem', md: '1.2rem' },
                }}
              >
                Upload your volleyball match and let{' '}
                <Box
                  component="span"
                  sx={{
                    fontWeight: 700,
                    fontSize: '1.1em',
                    letterSpacing: '0.04em',
                    color: 'primary.light',
                    animation: 'aiBreathe 5s ease-in-out infinite',
                    '@keyframes aiBreathe': {
                      '0%, 100%': { opacity: 0.7 },
                      '50%': { opacity: 1 },
                    },
                  }}
                >
                  AI
                </Box>{' '}
                cut the dead time.
                <br />
                Get a clean highlight reel with every rally, ready to share.
              </Typography>
            </MotionBox>

            {/* CTA Section */}
            <MotionBox {...(!shouldReduceMotion && { variants: fadeInUp })}>
              <Button
                component={Link}
                href="/sessions"
                variant="contained"
                size="large"
                sx={{
                  px: 4,
                  py: 1.75,
                  mb: 2,
                  fontSize: '1.1rem',
                  fontWeight: 600,
                  width: { xs: '100%', sm: 'auto' },
                  transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                  boxShadow: '0 4px 20px rgba(255, 107, 74, 0.3)',
                  '&:hover': {
                    transform: 'translateY(-2px)',
                    boxShadow: '0 8px 30px rgba(255, 107, 74, 0.4)',
                  },
                }}
              >
                Start Editing Free
              </Button>
            </MotionBox>

            {/* Trust Badges */}
            <MotionBox {...(!shouldReduceMotion && { variants: fadeInUp })}>
              <Stack
                direction={{ xs: 'column', sm: 'row' }}
                spacing={{ xs: 0.75, sm: 3 }}
                alignItems="center"
                justifyContent="center"
              >
                <TrustBadge text="No editing skills needed" />
                <TrustBadge text="Works in your browser" />
                <TrustBadge text="Beach volleyball" />
              </Stack>
            </MotionBox>
          </Box>
        </MotionBox>

        {/* Pipeline Animation */}
        <MotionBox
          {...(!shouldReduceMotion && {
            initial: 'hidden',
            animate: 'visible',
            variants: fadeInUp,
          })}
        >
          <PipelineAnimation />
        </MotionBox>
      </Container>
    </Box>
  );
}
