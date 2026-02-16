'use client';

import { useState } from 'react';
import { Box, Typography, Collapse, ButtonBase } from '@mui/material';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import VideocamOutlinedIcon from '@mui/icons-material/VideocamOutlined';
import TripOriginOutlinedIcon from '@mui/icons-material/TripOriginOutlined';
import { designTokens } from '@/app/theme';

interface Tip {
  label: string;
  value: string;
  note?: string;
}

const cameraSettings: Tip[] = [
  {
    label: 'Distance',
    value: 'Far enough to see the full court and ball flight',
    note: 'Best for ball tracking — closer angles lose the ball when it goes high',
  },
  {
    label: 'Position',
    value: 'Behind baseline, centered on the court',
    note: 'Standard broadcast angle works best for player identification',
  },
  {
    label: 'Height',
    value: 'Eye-level or slightly above (5–6 ft / 1.5–2m)',
  },
  {
    label: 'Stability',
    value: 'Use a tripod or rest the camera on a stable surface',
    note: 'Shaky footage reduces player tracking accuracy',
  },
];

const sceneTips: Tip[] = [
  {
    label: 'Background',
    value: 'Minimize people behind the court',
    note: 'Spectators and passers-by can confuse player detection',
  },
];

function TipSection({ icon, title, tips }: { icon: React.ReactNode; title: string; tips: Tip[] }) {
  return (
    <Box sx={{ bgcolor: 'action.hover', borderRadius: 1.5, p: 1.5 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1.5 }}>
        {icon}
        <Typography
          variant="caption"
          sx={{
            fontWeight: 700,
            color: 'text.primary',
            textTransform: 'uppercase',
            letterSpacing: '0.5px',
            fontSize: '0.65rem',
          }}
        >
          {title}
        </Typography>
      </Box>
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
        {tips.map(({ label, value, note }) => (
          <Box
            key={label}
            sx={{
              display: 'grid',
              gridTemplateColumns: '80px 1fr',
              alignItems: 'baseline',
            }}
          >
            <Typography
              variant="caption"
              sx={{ color: 'text.disabled', fontWeight: 600, fontSize: '0.7rem' }}
            >
              {label}
            </Typography>
            <Box>
              <Typography
                variant="caption"
                sx={{ color: 'text.secondary', fontWeight: 500 }}
              >
                {value}
              </Typography>
              {note && (
                <Typography
                  variant="caption"
                  sx={{
                    color: 'text.disabled',
                    display: 'block',
                    fontSize: '0.65rem',
                    fontStyle: 'italic',
                    mt: 0.25,
                  }}
                >
                  {note}
                </Typography>
              )}
            </Box>
          </Box>
        ))}
      </Box>
    </Box>
  );
}

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
        <Box sx={{ px: 2, pb: 2, display: 'flex', flexDirection: 'column', gap: 1.5 }}>
          <TipSection icon={<VideocamOutlinedIcon sx={{ fontSize: 16, color: 'primary.main' }} />} title="Camera Setup" tips={cameraSettings} />
          <TipSection icon={<TripOriginOutlinedIcon sx={{ fontSize: 16, color: 'primary.main' }} />} title="Scene" tips={sceneTips} />
        </Box>
      </Collapse>
    </Box>
  );
}
