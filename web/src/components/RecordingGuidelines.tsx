'use client';

import { useState } from 'react';
import { Box, Typography, Collapse, ButtonBase } from '@mui/material';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import { designTokens } from '@/app/theme';

const recommendations = [
  { label: 'Height', value: '2.5–3m (8–10 ft)' },
  { label: 'Quality', value: '1080p at 30fps' },
] as const;

export function RecordingGuidelines() {
  const [expanded, setExpanded] = useState(false);

  return (
    <Box
      sx={{
        mt: 2,
        borderRadius: 2,
        bgcolor: designTokens.colors.surface[1],
        border: '1px solid',
        borderColor: 'divider',
        overflow: 'hidden',
      }}
    >
      <ButtonBase
        onClick={() => setExpanded(!expanded)}
        sx={{
          width: '100%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          p: 1.5,
          gap: 1,
          '&:hover': {
            bgcolor: 'action.hover',
          },
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <InfoOutlinedIcon sx={{ fontSize: 18, color: 'primary.main', opacity: 0.8 }} />
          <Typography variant="body2" sx={{ fontWeight: 600, color: 'text.secondary' }}>
            Recording Tips
          </Typography>
        </Box>
        <ExpandMoreIcon
          sx={{
            fontSize: 20,
            color: 'text.disabled',
            transition: 'transform 0.2s',
            transform: expanded ? 'rotate(180deg)' : 'rotate(0deg)',
          }}
        />
      </ButtonBase>
      <Collapse in={expanded}>
        <Box sx={{ px: 1.5, pt: 0.5, pb: 1.5 }}>
          <Box
            sx={{
              display: 'grid',
              gridTemplateColumns: '72px 1fr',
              rowGap: 0.5,
            }}
          >
            <Typography
              variant="caption"
              sx={{ color: 'text.disabled', fontWeight: 600 }}
            >
              Position
            </Typography>
            <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 500 }}>
              Center of back baseline{' '}
              <Typography
                component="span"
                variant="caption"
                sx={{ color: 'warning.main', fontWeight: 600 }}
              >
                (required)
              </Typography>
            </Typography>
          </Box>
          <Typography
            variant="caption"
            sx={{ color: 'text.disabled', mt: 1, mb: 0.25, display: 'block' }}
          >
            For best AI detection results:
          </Typography>
          <Box
            sx={{
              display: 'grid',
              gridTemplateColumns: '72px 1fr',
              rowGap: 0.5,
            }}
          >
            {recommendations.map(({ label, value }) => (
              <Box key={label} sx={{ display: 'contents' }}>
                <Typography
                  variant="caption"
                  sx={{ color: 'text.disabled', fontWeight: 500 }}
                >
                  {label}
                </Typography>
                <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                  {value}
                </Typography>
              </Box>
            ))}
          </Box>
        </Box>
      </Collapse>
    </Box>
  );
}
