'use client';

import { Box, Container, Typography, Grid, Stack } from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import MovieIcon from '@mui/icons-material/Movie';
import { designTokens } from '@/app/theme';

const steps = [
  {
    number: '01',
    title: 'Upload Your Video',
    description: 'Drag and drop your beach volleyball recording. We support all common video formats.',
    icon: CloudUploadIcon,
  },
  {
    number: '02',
    title: 'Let AI Do the Work',
    description: 'Our AI analyzes your video and identifies every rally automatically. Sit back and relax.',
    icon: AutoAwesomeIcon,
  },
  {
    number: '03',
    title: 'Export Your Highlights',
    description: 'Select your best moments, create collections, and export. Ready to share in minutes!',
    icon: MovieIcon,
  },
];

export function HowItWorks() {
  return (
    <Box
      component="section"
      sx={{
        py: { xs: 8, md: 12 },
        bgcolor: designTokens.colors.surface[1],
      }}
    >
      <Container maxWidth="lg">
        <Box sx={{ textAlign: 'center', mb: 8 }}>
          <Typography
            variant="h2"
            sx={{
              fontSize: { xs: '2rem', md: '2.5rem' },
              fontWeight: 700,
              mb: 2,
            }}
          >
            From Raw Footage to Highlights in 3 Steps
          </Typography>
          <Typography
            variant="body1"
            color="text.secondary"
            sx={{ maxWidth: 600, mx: 'auto' }}
          >
            No editing experience needed. Just upload, let AI work its magic, and export.
          </Typography>
        </Box>

        <Grid container spacing={4} alignItems="stretch">
          {steps.map((step, index) => (
            <Grid size={{ xs: 12, md: 4 }} key={step.number}>
              <Stack
                spacing={3}
                sx={{
                  height: '100%',
                  p: 4,
                  borderRadius: 3,
                  bgcolor: designTokens.colors.surface[2],
                  border: '1px solid',
                  borderColor: 'divider',
                  position: 'relative',
                  transition: 'all 0.3s ease',
                  '&:hover': {
                    borderColor: 'primary.main',
                    transform: 'translateY(-4px)',
                  },
                }}
              >
                {/* Step Number */}
                <Typography
                  sx={{
                    fontSize: '4rem',
                    fontWeight: 800,
                    background: designTokens.gradients.primary,
                    backgroundClip: 'text',
                    WebkitBackgroundClip: 'text',
                    WebkitTextFillColor: 'transparent',
                    opacity: 0.3,
                    lineHeight: 1,
                  }}
                >
                  {step.number}
                </Typography>

                {/* Icon */}
                <Box
                  sx={{
                    width: 64,
                    height: 64,
                    borderRadius: 2,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    bgcolor: 'rgba(255, 107, 74, 0.1)',
                    border: '1px solid rgba(255, 107, 74, 0.2)',
                  }}
                >
                  <step.icon sx={{ fontSize: 32, color: 'primary.main' }} />
                </Box>

                {/* Content */}
                <Box>
                  <Typography variant="h5" fontWeight={600} sx={{ mb: 1.5 }}>
                    {step.title}
                  </Typography>
                  <Typography variant="body1" color="text.secondary" sx={{ lineHeight: 1.7 }}>
                    {step.description}
                  </Typography>
                </Box>

                {/* Connector Arrow (not on last item) */}
                {index < steps.length - 1 && (
                  <Box
                    sx={{
                      display: { xs: 'none', md: 'block' },
                      position: 'absolute',
                      right: -32,
                      top: '50%',
                      transform: 'translateY(-50%)',
                      color: 'text.disabled',
                      fontSize: '2rem',
                    }}
                  >
                    →
                  </Box>
                )}
              </Stack>
            </Grid>
          ))}
        </Grid>
      </Container>
    </Box>
  );
}
