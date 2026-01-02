'use client';

import { Box, Container, Typography, Grid, Paper, Stack } from '@mui/material';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import TimelineIcon from '@mui/icons-material/Timeline';
import CollectionsIcon from '@mui/icons-material/Collections';
import DownloadIcon from '@mui/icons-material/Download';
import { designTokens } from '@/app/theme';

const features = [
  {
    icon: AutoAwesomeIcon,
    title: 'AI Rally Detection',
    description:
      'Our ML model detects active play with 95%+ accuracy. No more scrubbing through hours of footage to find the action.',
    color: 'primary.main',
  },
  {
    icon: TimelineIcon,
    title: 'Timeline Editor',
    description:
      'Fine-tune rally boundaries with intuitive drag-and-drop controls. Add or remove segments in seconds.',
    color: 'secondary.main',
  },
  {
    icon: CollectionsIcon,
    title: 'Highlight Collections',
    description:
      'Group your best rallies into custom highlight reels. Color-code and organize as you like.',
    color: '#FFB547',
  },
  {
    icon: DownloadIcon,
    title: 'One-Click Export',
    description:
      'Export your highlights with smooth fade transitions. Ready for Instagram, TikTok, or YouTube.',
    color: '#A78BFA',
  },
];

export function Features() {
  return (
    <Box
      component="section"
      id="features"
      sx={{
        py: { xs: 8, md: 12 },
        position: 'relative',
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
            Everything You Need to Create Highlight Reels
          </Typography>
          <Typography
            variant="body1"
            color="text.secondary"
            sx={{ maxWidth: 600, mx: 'auto' }}
          >
            Powerful features that make video editing effortless, so you can focus
            on the game, not the software.
          </Typography>
        </Box>

        <Grid container spacing={4}>
          {features.map((feature) => (
            <Grid size={{ xs: 12, sm: 6, md: 3 }} key={feature.title}>
              <Paper
                elevation={0}
                sx={{
                  p: 4,
                  height: '100%',
                  bgcolor: designTokens.colors.surface[1],
                  border: '1px solid',
                  borderColor: 'divider',
                  borderRadius: 3,
                  transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                  '&:hover': {
                    transform: 'translateY(-8px)',
                    boxShadow: designTokens.shadows.xl,
                    borderColor: feature.color,
                    '& .feature-icon': {
                      transform: 'scale(1.1)',
                    },
                  },
                }}
              >
                <Stack spacing={2}>
                  <Box
                    className="feature-icon"
                    sx={{
                      width: 56,
                      height: 56,
                      borderRadius: 2,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      bgcolor: `${feature.color}15`,
                      transition: 'transform 0.3s ease',
                    }}
                  >
                    <feature.icon sx={{ fontSize: 28, color: feature.color }} />
                  </Box>
                  <Typography variant="h6" fontWeight={600}>
                    {feature.title}
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ lineHeight: 1.7 }}>
                    {feature.description}
                  </Typography>
                </Stack>
              </Paper>
            </Grid>
          ))}
        </Grid>
      </Container>
    </Box>
  );
}
