'use client';

import { Box, Container, Typography, Button, Stack, Chip } from '@mui/material';
import { motion, useReducedMotion } from 'framer-motion';
import Link from 'next/link';
import { designTokens } from '@/app/theme';
import { HeroBackground } from './HeroBackground';
import { EditorMockup } from './EditorMockup';
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

  const scrollToDemo = () => {
    const element = document.querySelector('#demo');
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  const MotionBox = shouldReduceMotion ? Box : motion.div;

  return (
    <Box
      component="section"
      sx={{
        minHeight: 'calc(100vh - 64px)',
        display: 'flex',
        alignItems: 'center',
        position: 'relative',
        overflow: 'hidden',
        py: { xs: 6, md: 8 },
      }}
    >
      {/* Animated background */}
      <HeroBackground />

      <Container maxWidth="lg" sx={{ position: 'relative', zIndex: 1 }}>
        <Box
          sx={{
            display: 'flex',
            flexDirection: { xs: 'column', lg: 'row' },
            alignItems: 'center',
            gap: { xs: 6, lg: 8 },
          }}
        >
          {/* Left content */}
          <MotionBox
            {...(!shouldReduceMotion && {
              initial: 'hidden',
              animate: 'visible',
              variants: staggerContainer,
            })}
            style={{ flex: 1 }}
          >
            <Box sx={{ maxWidth: 580 }}>
              {/* Badge */}
              <MotionBox {...(!shouldReduceMotion && { variants: fadeInUp })}>
                <Chip
                  label="AI-Powered Video Analysis"
                  sx={{
                    mb: 3,
                    bgcolor: 'rgba(255, 107, 74, 0.12)',
                    color: 'primary.light',
                    fontWeight: 600,
                    fontSize: '0.85rem',
                    border: '1px solid rgba(255, 107, 74, 0.25)',
                    '& .MuiChip-label': {
                      px: 1.5,
                    },
                  }}
                />
              </MotionBox>

              {/* Headline */}
              <MotionBox {...(!shouldReduceMotion && { variants: fadeInUp })}>
                <Typography
                  variant="h1"
                  sx={{
                    fontSize: { xs: '2.5rem', sm: '3.25rem', md: '3.75rem', lg: '4.25rem' },
                    fontWeight: 800,
                    lineHeight: 1.08,
                    letterSpacing: '-0.03em',
                    mb: 3,
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
                    mb: 4,
                    lineHeight: 1.6,
                    fontSize: { xs: '1.05rem', md: '1.2rem' },
                  }}
                >
                  RallyCut uses AI to automatically detect rallies and remove dead time from your
                  beach volleyball videos. Create highlight reels in minutes, no editing skills
                  required.
                </Typography>
              </MotionBox>

              {/* CTA Section - Single focused CTA */}
              <MotionBox {...(!shouldReduceMotion && { variants: fadeInUp })}>
                <Stack spacing={2} sx={{ mb: 4 }}>
                  <Button
                    component={Link}
                    href="/sessions"
                    variant="contained"
                    size="large"
                    sx={{
                      px: 4,
                      py: 1.75,
                      fontSize: '1.1rem',
                      fontWeight: 600,
                      width: { xs: '100%', sm: 'auto' },
                      alignSelf: 'flex-start',
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
                  <Typography
                    component="button"
                    onClick={scrollToDemo}
                    sx={{
                      background: 'none',
                      border: 'none',
                      color: 'text.secondary',
                      fontSize: '0.95rem',
                      cursor: 'pointer',
                      textAlign: 'left',
                      p: 0,
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: 0.5,
                      width: 'fit-content',
                      transition: 'color 0.2s',
                      '&:hover': {
                        color: 'primary.main',
                      },
                    }}
                  >
                    See how it works
                    <Box
                      component="span"
                      sx={{
                        display: 'inline-block',
                        transition: 'transform 0.2s',
                        '&:hover': { transform: 'translateX(4px)' },
                      }}
                    >
                      →
                    </Box>
                  </Typography>
                </Stack>
              </MotionBox>

              {/* Trust Badges */}
              <MotionBox {...(!shouldReduceMotion && { variants: fadeInUp })}>
                <Stack
                  direction={{ xs: 'column', sm: 'row' }}
                  spacing={{ xs: 1.5, sm: 3 }}
                  sx={{ flexWrap: 'wrap' }}
                >
                  <TrustBadge text="No account required" />
                  <TrustBadge text="Works in your browser" />
                  <TrustBadge text="Free tier available" />
                </Stack>
              </MotionBox>
            </Box>
          </MotionBox>

          {/* Right mockup */}
          <Box
            sx={{
              flex: 1,
              display: 'flex',
              justifyContent: 'center',
              width: '100%',
              maxWidth: { xs: 450, lg: 'none' },
            }}
          >
            <EditorMockup />
          </Box>
        </Box>
      </Container>
    </Box>
  );
}
