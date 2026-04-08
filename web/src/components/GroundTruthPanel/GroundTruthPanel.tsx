'use client';

import { ReactNode, useState } from 'react';
import {
  Accordion,
  AccordionDetails,
  AccordionSummary,
  Box,
  Chip,
  Typography,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import { CalibrationGtSection } from './CalibrationGtSection';
import { ActionGtSection } from './ActionGtSection';
import { ScoreGtSection } from './ScoreGtSection';
import { PlayerTrackingAdvancedSection } from './PlayerTrackingAdvancedSection';

interface GroundTruthPanelProps {
  onOpenPlayerMatching?: () => void;
  onOpenReferenceCrops?: () => void;
}

type SectionId = 'calibration' | 'actions' | 'score' | 'advanced';

interface SectionProps {
  id: SectionId;
  title: string;
  status?: ReactNode;
  expanded: boolean;
  onToggle: (id: SectionId) => void;
  children: ReactNode;
}

function Section({ id, title, status, expanded, onToggle, children }: SectionProps) {
  return (
    <Accordion
      disableGutters
      expanded={expanded}
      onChange={() => onToggle(id)}
      sx={{
        bgcolor: 'transparent',
        '&:before': { display: 'none' },
        borderBottom: '1px solid',
        borderColor: 'divider',
      }}
    >
      <AccordionSummary
        expandIcon={<ExpandMoreIcon fontSize="small" />}
        sx={{ minHeight: 36, '& .MuiAccordionSummary-content': { my: 0.5 } }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flex: 1 }}>
          <Typography variant="body2" sx={{ fontWeight: 600, flex: 1 }}>
            {title}
          </Typography>
          {status}
        </Box>
      </AccordionSummary>
      <AccordionDetails sx={{ px: 1.5, py: 1 }}>{children}</AccordionDetails>
    </Accordion>
  );
}

export function GroundTruthPanel({
  onOpenPlayerMatching,
  onOpenReferenceCrops,
}: GroundTruthPanelProps) {
  const [expanded, setExpanded] = useState<Record<SectionId, boolean>>({
    calibration: true,
    actions: true,
    score: true,
    advanced: false,
  });

  const handleToggle = (id: SectionId) => {
    setExpanded((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%', overflowY: 'auto' }}>
      <CalibrationGtSection
        renderSection={(status, body) => (
          <Section
            id="calibration"
            title="Court calibration"
            status={status}
            expanded={expanded.calibration}
            onToggle={handleToggle}
          >
            {body}
          </Section>
        )}
      />
      <ActionGtSection
        renderSection={(status, body) => (
          <Section
            id="actions"
            title="Actions"
            status={status}
            expanded={expanded.actions}
            onToggle={handleToggle}
          >
            {body}
          </Section>
        )}
      />
      <ScoreGtSection
        renderSection={(status, body) => (
          <Section
            id="score"
            title="Score"
            status={status}
            expanded={expanded.score}
            onToggle={handleToggle}
          >
            {body}
          </Section>
        )}
      />
      <PlayerTrackingAdvancedSection
        onOpenPlayerMatching={onOpenPlayerMatching}
        onOpenReferenceCrops={onOpenReferenceCrops}
        renderSection={(body) => (
          <Section
            id="advanced"
            title="Player tracking"
            status={<Chip label="Advanced" size="small" sx={{ height: 18, fontSize: '0.65rem' }} />}
            expanded={expanded.advanced}
            onToggle={handleToggle}
          >
            {body}
          </Section>
        )}
      />
    </Box>
  );
}
