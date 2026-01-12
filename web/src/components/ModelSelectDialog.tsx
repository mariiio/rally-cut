'use client';

import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Box,
  Typography,
  Paper,
  alpha,
} from '@mui/material';
import SportsVolleyballIcon from '@mui/icons-material/SportsVolleyball';
import BeachAccessIcon from '@mui/icons-material/BeachAccess';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';

const DIALOG_FADE_DURATION_MS = 200;

export type ModelVariant = 'indoor' | 'beach';

interface ModelOption {
  id: ModelVariant;
  title: string;
  description: string;
  icon: React.ReactNode;
}

const MODEL_OPTIONS: ModelOption[] = [
  {
    id: 'indoor',
    title: 'Indoor Volleyball',
    description: 'Original model optimized for indoor courts with consistent lighting',
    icon: <SportsVolleyballIcon sx={{ fontSize: 32 }} />,
  },
  {
    id: 'beach',
    title: 'Beach Volleyball',
    description: 'Fine-tuned for outdoor beach courts with variable conditions',
    icon: <BeachAccessIcon sx={{ fontSize: 32 }} />,
  },
];

interface ModelSelectDialogProps {
  open: boolean;
  onClose: () => void;
  onSelect: (model: ModelVariant) => void;
}

export function ModelSelectDialog({
  open,
  onClose,
  onSelect,
}: ModelSelectDialogProps) {
  return (
    <Dialog
      open={open}
      onClose={onClose}
      transitionDuration={DIALOG_FADE_DURATION_MS}
      maxWidth="sm"
      fullWidth
      PaperProps={{
        sx: {
          borderRadius: 2,
          overflow: 'hidden',
        },
      }}
    >
      <DialogTitle
        sx={{
          pb: 1,
          textAlign: 'center',
          fontWeight: 600,
        }}
      >
        Choose Detection Model
      </DialogTitle>
      <DialogContent sx={{ pt: 1, pb: 2 }}>
        <Typography
          variant="body2"
          color="text.secondary"
          sx={{ mb: 2.5, textAlign: 'center' }}
        >
          Select the model that best matches your video type
        </Typography>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
          {MODEL_OPTIONS.map((option) => (
            <Paper
              key={option.id}
              onClick={() => onSelect(option.id)}
              elevation={0}
              sx={{
                p: 2,
                cursor: 'pointer',
                border: '2px solid',
                borderColor: 'divider',
                borderRadius: 2,
                display: 'flex',
                alignItems: 'center',
                gap: 2,
                transition: 'all 0.15s ease',
                '&:hover': {
                  borderColor: 'primary.main',
                  bgcolor: (theme) => alpha(theme.palette.primary.main, 0.04),
                  '& .model-icon': {
                    color: 'primary.main',
                  },
                },
                '&:active': {
                  transform: 'scale(0.98)',
                },
              }}
            >
              <Box
                className="model-icon"
                sx={{
                  color: 'text.secondary',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  width: 48,
                  height: 48,
                  borderRadius: 1.5,
                  bgcolor: 'action.hover',
                  transition: 'color 0.15s ease',
                }}
              >
                {option.icon}
              </Box>
              <Box sx={{ flex: 1 }}>
                <Typography
                  variant="subtitle1"
                  sx={{ fontWeight: 600, lineHeight: 1.3 }}
                >
                  {option.title}
                </Typography>
                <Typography
                  variant="body2"
                  color="text.secondary"
                  sx={{ lineHeight: 1.4 }}
                >
                  {option.description}
                </Typography>
              </Box>
              <CheckCircleIcon
                sx={{
                  fontSize: 24,
                  color: 'transparent',
                  transition: 'color 0.15s ease',
                  '.MuiPaper-root:hover &': {
                    color: 'primary.main',
                  },
                }}
              />
            </Paper>
          ))}
        </Box>
      </DialogContent>
      <DialogActions sx={{ px: 3, pb: 2.5, pt: 0.5 }}>
        <Button
          onClick={onClose}
          variant="text"
          sx={{
            color: 'text.secondary',
            '&:hover': {
              bgcolor: 'action.hover',
            },
          }}
        >
          Cancel
        </Button>
      </DialogActions>
    </Dialog>
  );
}
