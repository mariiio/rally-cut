'use client';

import { Box, Container, Typography, Grid, Paper, Stack } from '@mui/material';
import { motion, useInView, useReducedMotion } from 'framer-motion';
import { useRef } from 'react';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import TimelineIcon from '@mui/icons-material/Timeline';
import CollectionsIcon from '@mui/icons-material/Collections';
import DownloadIcon from '@mui/icons-material/Download';
import { designTokens } from '@/app/theme';
import { fadeInUp, staggerContainerFast } from '../utils/animations';

const features = [
  {
    icon: AutoAwesomeIcon,
    title: 'AI Rally Detection',
    description:
      'Our ML model detects active play with 95%+ accuracy. No more scrubbing through hours of footage to find the action.',
    color: '#FF6B4A',
    gradient: 'linear-gradient(135deg, #FF6B4A 0%, #FF8A6F 100%)',
  },
  {
    icon: TimelineIcon,
    title: 'Timeline Editor',
    description:
      'Fine-tune rally boundaries with intuitive drag-and-drop controls. Add or remove segments in seconds.',
    color: '#00D4AA',
    gradient: 'linear-gradient(135deg, #00D4AA 0%, #4DDFBF 100%)',
  },
  {
    icon: CollectionsIcon,
    title: 'Highlight Collections',
    description:
      'Group your best rallies into custom highlight reels. Color-code and organize as you like.',
    color: '#FFD166',
    gradient: 'linear-gradient(135deg, #FFD166 0%, #FFDE8A 100%)',
  },
  {
    icon: DownloadIcon,
    title: 'One-Click Export',
    description:
      'Export your highlights with smooth fade transitions. Ready for Instagram, TikTok, or YouTube.',
    color: '#A78BFA',
    gradient: 'linear-gradient(135deg, #A78BFA 0%, #C4B5FD 100%)',
  },
];

interface FeatureCardProps {
  feature: (typeof features)[0];
  index: number;
}

function FeatureCard({ feature, index }: FeatureCardProps) {
  const shouldReduceMotion = useReducedMotion();

  const cardContent = (
    <Paper
      elevation={0}
      sx={{
        p: 4,
        height: '100%',
        bgcolor: designTokens.colors.surface[1],
        border: '1px solid',
        borderColor: 'divider',
        borderRadius: 3,
        position: 'relative',
        overflow: 'hidden',
        transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
        '&:hover': {
          transform: 'translateY(-8px)',
          boxShadow: `0 20px 40px rgba(0, 0, 0, 0.3)`,
          borderColor: feature.color,
          '& .feature-icon': {
            transform: 'scale(1.1)',
          },
          '& .feature-glow': {
            opacity: 1,
          },
        },
      }}
    >
      {/* Hover glow effect */}
      <Box
        className="feature-glow"
        sx={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          height: 100,
          background: `linear-gradient(180deg, ${feature.color}15 0%, transparent 100%)`,
          opacity: 0,
          transition: 'opacity 0.3s ease',
          pointerEvents: 'none',
        }}
      />

      <Stack spacing={2.5} sx={{ position: 'relative' }}>
        <Box
          className="feature-icon"
          sx={{
            width: 56,
            height: 56,
            borderRadius: 2,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: `${feature.color}15`,
            border: `1px solid ${feature.color}30`,
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
  );

  if (shouldReduceMotion) {
    return cardContent;
  }

  return (
    <motion.div variants={fadeInUp} custom={index} style={{ height: '100%' }}>
      {cardContent}
    </motion.div>
  );
}

export function Features() {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: '-50px' });
  const shouldReduceMotion = useReducedMotion();

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
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
        >
          <Box sx={{ textAlign: 'center', mb: 8 }}>
            <Typography
              variant="h2"
              sx={{
                fontSize: { xs: '2rem', md: '2.5rem' },
                fontWeight: 700,
                mb: 2,
              }}
            >
              Powerful Features, Zero Complexity
            </Typography>
            <Typography
              variant="body1"
              color="text.secondary"
              sx={{ maxWidth: 600, mx: 'auto' }}
            >
              Professional video editing tools designed for athletes, not editors.
            </Typography>
          </Box>
        </motion.div>

        <motion.div
          ref={ref}
          initial="hidden"
          animate={isInView ? 'visible' : 'hidden'}
          variants={shouldReduceMotion ? {} : staggerContainerFast}
        >
          <Grid container spacing={4}>
            {features.map((feature, index) => (
              <Grid size={{ xs: 12, sm: 6, md: 3 }} key={feature.title}>
                <FeatureCard feature={feature} index={index} />
              </Grid>
            ))}
          </Grid>
        </motion.div>
      </Container>
    </Box>
  );
}
