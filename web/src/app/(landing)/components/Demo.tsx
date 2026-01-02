'use client';

import { Box, Container, Typography, Paper } from '@mui/material';
import PlayCircleOutlineIcon from '@mui/icons-material/PlayCircleOutline';
import { designTokens } from '@/app/theme';

export function Demo() {
  return (
    <Box
      component="section"
      id="demo"
      sx={{
        py: { xs: 8, md: 12 },
        bgcolor: designTokens.colors.surface[1],
      }}
    >
      <Container maxWidth="lg">
        <Box sx={{ textAlign: 'center', mb: 6 }}>
          <Typography
            variant="h2"
            sx={{
              fontSize: { xs: '2rem', md: '2.5rem' },
              fontWeight: 700,
              mb: 2,
            }}
          >
            See RallyCut in Action
          </Typography>
          <Typography
            variant="body1"
            color="text.secondary"
            sx={{ maxWidth: 600, mx: 'auto' }}
          >
            Watch how our AI automatically identifies rallies and helps you create
            highlight reels in minutes.
          </Typography>
        </Box>

        {/* Video Placeholder */}
        <Paper
          elevation={0}
          sx={{
            position: 'relative',
            aspectRatio: '16/9',
            maxWidth: 900,
            mx: 'auto',
            borderRadius: 3,
            overflow: 'hidden',
            bgcolor: designTokens.colors.surface[2],
            border: '1px solid',
            borderColor: 'divider',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            cursor: 'pointer',
            transition: 'all 0.3s ease',
            '&:hover': {
              borderColor: 'primary.main',
              '& .play-icon': {
                transform: 'scale(1.1)',
                color: 'primary.main',
              },
            },
          }}
        >
          {/* Placeholder Content */}
          <Box sx={{ textAlign: 'center', p: 4 }}>
            <PlayCircleOutlineIcon
              className="play-icon"
              sx={{
                fontSize: 80,
                color: 'text.secondary',
                mb: 2,
                transition: 'all 0.3s ease',
              }}
            />
            <Typography variant="h6" color="text.secondary" sx={{ mb: 1 }}>
              Demo Video Coming Soon
            </Typography>
            <Typography variant="body2" color="text.disabled">
              In the meantime, try it yourself — it&apos;s free!
            </Typography>
          </Box>

          {/* Decorative Elements */}
          <Box
            sx={{
              position: 'absolute',
              top: 16,
              left: 16,
              display: 'flex',
              gap: 1,
            }}
          >
            {['#FF5F56', '#FFBD2E', '#27C93F'].map((color) => (
              <Box
                key={color}
                sx={{
                  width: 12,
                  height: 12,
                  borderRadius: '50%',
                  bgcolor: color,
                  opacity: 0.6,
                }}
              />
            ))}
          </Box>
        </Paper>
      </Container>
    </Box>
  );
}
