'use client';

import { Box, Container, Typography } from '@mui/material';
import { designTokens } from '@/app/theme';
import { InteractiveDemo } from './InteractiveDemo';
import { AnimatedSection } from './AnimatedSection';

export function Demo() {
  return (
    <Box
      component="section"
      id="demo"
      sx={{
        py: { xs: 8, md: 12 },
        bgcolor: designTokens.colors.surface[1],
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      {/* Subtle gradient accent */}
      <Box
        sx={{
          position: 'absolute',
          top: 0,
          left: '50%',
          transform: 'translateX(-50%)',
          width: '100%',
          maxWidth: 800,
          height: 1,
          background: 'linear-gradient(90deg, transparent, rgba(255, 107, 74, 0.3), transparent)',
        }}
      />

      <Container maxWidth="lg">
        <AnimatedSection>
          <Box sx={{ textAlign: 'center', mb: 6 }}>
            <Typography
              variant="h2"
              sx={{
                fontSize: { xs: '2rem', md: '2.5rem' },
                fontWeight: 700,
                mb: 2,
              }}
            >
              How It Works
            </Typography>
            <Typography
              variant="body1"
              color="text.secondary"
              sx={{ maxWidth: 550, mx: 'auto', lineHeight: 1.7 }}
            >
              Upload your footage, let AI detect the action, and export your highlights in minutes.
            </Typography>
          </Box>
        </AnimatedSection>

        <AnimatedSection delay={0.15}>
          <InteractiveDemo />
        </AnimatedSection>
      </Container>
    </Box>
  );
}
