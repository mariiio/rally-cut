'use client';

import { Box, Typography, Stack } from '@mui/material';
import { motion } from 'framer-motion';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import MovieIcon from '@mui/icons-material/Movie';
import { designTokens } from '@/app/theme';

const STEPS = [
  { id: 0, label: 'Upload', icon: CloudUploadIcon },
  { id: 1, label: 'Analyze', icon: AutoAwesomeIcon },
  { id: 2, label: 'Export', icon: MovieIcon },
];

interface DemoTabsProps {
  activeStep: number;
  onStepChange: (step: number) => void;
  isAutoPlaying: boolean;
}

export function DemoTabs({ activeStep, onStepChange, isAutoPlaying }: DemoTabsProps) {
  return (
    <Stack
      direction="row"
      spacing={{ xs: 1, sm: 2 }}
      sx={{
        justifyContent: 'center',
        mb: 4,
      }}
    >
      {STEPS.map((step, index) => {
        const isActive = activeStep === step.id;
        const isCompleted = activeStep > step.id;
        const Icon = step.icon;

        return (
          <Box
            key={step.id}
            onClick={() => onStepChange(step.id)}
            sx={{
              display: 'flex',
              alignItems: 'center',
              gap: 1.5,
              px: { xs: 2, sm: 3 },
              py: 1.5,
              borderRadius: 3,
              cursor: 'pointer',
              position: 'relative',
              bgcolor: isActive
                ? 'rgba(255, 107, 74, 0.12)'
                : 'transparent',
              border: '1px solid',
              borderColor: isActive
                ? 'rgba(255, 107, 74, 0.3)'
                : isCompleted
                ? 'rgba(0, 212, 170, 0.2)'
                : 'transparent',
              transition: 'all 0.3s ease',
              '&:hover': {
                bgcolor: isActive
                  ? 'rgba(255, 107, 74, 0.15)'
                  : 'rgba(255, 255, 255, 0.03)',
              },
            }}
          >
            {/* Step number / icon */}
            <Box
              sx={{
                width: 32,
                height: 32,
                borderRadius: '50%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                bgcolor: isActive
                  ? 'primary.main'
                  : isCompleted
                  ? 'secondary.main'
                  : designTokens.colors.surface[2],
                transition: 'all 0.3s ease',
              }}
            >
              {isCompleted ? (
                <Typography
                  sx={{
                    color: 'white',
                    fontSize: '0.85rem',
                    fontWeight: 700,
                  }}
                >
                  ✓
                </Typography>
              ) : (
                <Icon
                  sx={{
                    fontSize: 16,
                    color: isActive ? 'white' : 'text.secondary',
                  }}
                />
              )}
            </Box>

            {/* Label */}
            <Typography
              variant="body2"
              sx={{
                fontWeight: isActive ? 600 : 500,
                color: isActive
                  ? 'primary.light'
                  : isCompleted
                  ? 'secondary.light'
                  : 'text.secondary',
                display: { xs: 'none', sm: 'block' },
              }}
            >
              {step.label}
            </Typography>

            {/* Auto-play progress indicator */}
            {isActive && isAutoPlaying && (
              <motion.div
                initial={{ scaleX: 0 }}
                animate={{ scaleX: 1 }}
                transition={{ duration: 5, ease: 'linear' }}
                style={{
                  position: 'absolute',
                  bottom: 0,
                  left: 0,
                  right: 0,
                  height: 2,
                  background: designTokens.gradients.primary,
                  transformOrigin: 'left',
                  borderRadius: 1,
                }}
              />
            )}

            {/* Connector line to next step */}
            {index < STEPS.length - 1 && (
              <Box
                sx={{
                  position: 'absolute',
                  right: { xs: -8, sm: -16 },
                  top: '50%',
                  transform: 'translateY(-50%)',
                  width: { xs: 8, sm: 16 },
                  height: 2,
                  bgcolor: isCompleted
                    ? 'secondary.main'
                    : designTokens.colors.surface[2],
                  transition: 'background-color 0.3s ease',
                  display: { xs: 'none', md: 'block' },
                }}
              />
            )}
          </Box>
        );
      })}
    </Stack>
  );
}
