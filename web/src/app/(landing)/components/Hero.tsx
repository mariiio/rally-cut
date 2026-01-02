'use client';

import { Box, Container, Typography, Button, Stack, Chip } from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import Link from 'next/link';
import { designTokens } from '@/app/theme';

export function Hero() {
  const scrollToDemo = () => {
    const element = document.querySelector('#demo');
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <Box
      component="section"
      sx={{
        minHeight: 'calc(100vh - 64px)',
        display: 'flex',
        alignItems: 'center',
        position: 'relative',
        overflow: 'hidden',
        py: { xs: 8, md: 12 },
        '&::before': {
          content: '""',
          position: 'absolute',
          top: 0,
          left: '50%',
          transform: 'translateX(-50%)',
          width: '200%',
          height: '100%',
          background: 'radial-gradient(ellipse at 50% 0%, rgba(255, 107, 74, 0.12) 0%, transparent 60%)',
          pointerEvents: 'none',
        },
        '&::after': {
          content: '""',
          position: 'absolute',
          bottom: 0,
          left: '50%',
          transform: 'translateX(-50%)',
          width: '200%',
          height: '50%',
          background: 'radial-gradient(ellipse at 50% 100%, rgba(0, 212, 170, 0.06) 0%, transparent 60%)',
          pointerEvents: 'none',
        },
      }}
    >
      <Container maxWidth="lg" sx={{ position: 'relative', zIndex: 1 }}>
        <Box sx={{ textAlign: 'center', maxWidth: 800, mx: 'auto' }}>
          {/* Badge */}
          <Chip
            label="AI-Powered Video Analysis"
            sx={{
              mb: 3,
              bgcolor: 'rgba(255, 107, 74, 0.15)',
              color: 'primary.light',
              fontWeight: 600,
              fontSize: '0.8rem',
              border: '1px solid rgba(255, 107, 74, 0.3)',
            }}
          />

          {/* Headline */}
          <Typography
            variant="h1"
            sx={{
              fontSize: { xs: '2.5rem', sm: '3.5rem', md: '4rem' },
              fontWeight: 800,
              lineHeight: 1.1,
              mb: 3,
              background: 'linear-gradient(135deg, #FFFFFF 0%, rgba(255, 255, 255, 0.8) 100%)',
              backgroundClip: 'text',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}
          >
            Turn Hours of Footage Into{' '}
            <Box
              component="span"
              sx={{
                background: designTokens.gradients.primary,
                backgroundClip: 'text',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
              }}
            >
              Minutes of Action
            </Box>
          </Typography>

          {/* Subheadline */}
          <Typography
            variant="h5"
            sx={{
              color: 'text.secondary',
              fontWeight: 400,
              mb: 5,
              lineHeight: 1.6,
              fontSize: { xs: '1.1rem', md: '1.25rem' },
            }}
          >
            RallyCut uses AI to automatically detect rallies and remove dead time from your
            beach volleyball videos. Create highlight reels in minutes, no editing skills required.
          </Typography>

          {/* CTAs */}
          <Stack
            direction={{ xs: 'column', sm: 'row' }}
            spacing={2}
            justifyContent="center"
            sx={{ mb: 5 }}
          >
            <Button
              component={Link}
              href="/editor"
              variant="contained"
              size="large"
              sx={{
                px: 4,
                py: 1.5,
                fontSize: '1.1rem',
                fontWeight: 600,
                transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                '&:hover': {
                  transform: 'translateY(-3px)',
                  boxShadow: '0 8px 30px rgba(255, 107, 74, 0.4)',
                },
              }}
            >
              Start Editing Free
            </Button>
            <Button
              variant="outlined"
              size="large"
              onClick={scrollToDemo}
              startIcon={<PlayArrowIcon />}
              sx={{
                px: 4,
                py: 1.5,
                fontSize: '1.1rem',
                fontWeight: 600,
                borderColor: 'rgba(255, 255, 255, 0.3)',
                color: 'text.primary',
                '&:hover': {
                  borderColor: 'primary.main',
                  bgcolor: 'rgba(255, 107, 74, 0.08)',
                },
              }}
            >
              See How It Works
            </Button>
          </Stack>

          {/* Trust Badges */}
          <Stack
            direction="row"
            spacing={{ xs: 2, sm: 4 }}
            justifyContent="center"
            flexWrap="wrap"
            sx={{ gap: 1 }}
          >
            {['No account required', 'Works in your browser', 'Free tier available'].map((badge) => (
              <Typography
                key={badge}
                variant="body2"
                sx={{
                  color: 'text.secondary',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 0.5,
                  '&::before': {
                    content: '"✓"',
                    color: 'secondary.main',
                    fontWeight: 700,
                  },
                }}
              >
                {badge}
              </Typography>
            ))}
          </Stack>
        </Box>
      </Container>
    </Box>
  );
}
