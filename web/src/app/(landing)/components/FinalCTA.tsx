'use client';

import { Box, Container, Typography, Button, Stack } from '@mui/material';
import Link from 'next/link';
import { designTokens } from '@/app/theme';

export function FinalCTA() {
  return (
    <Box
      component="section"
      sx={{
        py: { xs: 10, md: 14 },
        position: 'relative',
        overflow: 'hidden',
        '&::before': {
          content: '""',
          position: 'absolute',
          inset: 0,
          background: `linear-gradient(135deg, rgba(255, 107, 74, 0.15) 0%, rgba(255, 138, 76, 0.08) 50%, rgba(0, 212, 170, 0.1) 100%)`,
        },
      }}
    >
      <Container maxWidth="md" sx={{ position: 'relative', zIndex: 1 }}>
        <Stack spacing={4} alignItems="center" textAlign="center">
          <Typography
            variant="h2"
            sx={{
              fontSize: { xs: '2rem', md: '3rem' },
              fontWeight: 800,
              background: 'linear-gradient(135deg, #FFFFFF 0%, rgba(255, 255, 255, 0.9) 100%)',
              backgroundClip: 'text',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}
          >
            Ready to Save Hours of Editing?
          </Typography>

          <Typography
            variant="h6"
            sx={{
              color: 'text.secondary',
              fontWeight: 400,
              maxWidth: 500,
            }}
          >
            Join players who&apos;ve already made the switch to AI-powered video editing.
          </Typography>

          <Button
            component={Link}
            href="/editor"
            variant="contained"
            size="large"
            sx={{
              px: 5,
              py: 2,
              fontSize: '1.1rem',
              fontWeight: 600,
              borderRadius: 2,
              transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
              boxShadow: designTokens.shadows.glow.primary,
              '&:hover': {
                transform: 'translateY(-4px)',
                boxShadow: '0 12px 40px rgba(255, 107, 74, 0.5)',
              },
            }}
          >
            Start Editing Free
          </Button>

          <Typography variant="body2" color="text.disabled">
            No credit card required. No account needed.
          </Typography>
        </Stack>
      </Container>
    </Box>
  );
}
