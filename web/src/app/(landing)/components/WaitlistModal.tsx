'use client';

import { useState } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Button,
  Typography,
  Box,
  IconButton,
  Alert,
  CircularProgress,
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import { designTokens } from '@/app/theme';
import { trackWaitlistJoin } from '@/lib/analytics';

interface WaitlistModalProps {
  open: boolean;
  onClose: () => void;
  tier: string;
}

export function WaitlistModal({ open, onClose, tier }: WaitlistModalProps) {
  const [email, setEmail] = useState('');
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/waitlist', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, tier }),
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.message || 'Failed to join waitlist');
      }

      trackWaitlistJoin(tier);
      setSuccess(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong');
    } finally {
      setLoading(false);
    }
  };

  const handleClose = () => {
    setEmail('');
    setSuccess(false);
    setError(null);
    onClose();
  };

  return (
    <Dialog
      open={open}
      onClose={handleClose}
      maxWidth="xs"
      fullWidth
      PaperProps={{
        sx: {
          bgcolor: designTokens.colors.surface[2],
          borderRadius: 3,
        },
      }}
    >
      <DialogTitle sx={{ pr: 6 }}>
        <Typography variant="h5" fontWeight={600}>
          Join the Waitlist
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
          Be first to know when {tier.charAt(0).toUpperCase() + tier.slice(1)} is available
        </Typography>
        <IconButton
          onClick={handleClose}
          sx={{
            position: 'absolute',
            right: 12,
            top: 12,
            color: 'text.secondary',
          }}
        >
          <CloseIcon />
        </IconButton>
      </DialogTitle>

      {success ? (
        <DialogContent sx={{ textAlign: 'center', py: 4 }}>
          <CheckCircleIcon sx={{ fontSize: 64, color: 'secondary.main', mb: 2 }} />
          <Typography variant="h6" sx={{ mb: 1 }}>
            You&apos;re on the list!
          </Typography>
          <Typography variant="body2" color="text.secondary">
            We&apos;ll notify you at <strong>{email}</strong> when paid plans launch.
          </Typography>
        </DialogContent>
      ) : (
        <Box component="form" onSubmit={handleSubmit}>
          <DialogContent>
            {error && (
              <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
                {error}
              </Alert>
            )}
            <TextField
              autoFocus
              fullWidth
              type="email"
              label="Email address"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              disabled={loading}
              sx={{
                '& .MuiOutlinedInput-root': {
                  bgcolor: designTokens.colors.surface[1],
                },
              }}
            />
          </DialogContent>
          <DialogActions sx={{ px: 3, pb: 3 }}>
            <Button onClick={handleClose} disabled={loading}>
              Cancel
            </Button>
            <Button
              type="submit"
              variant="contained"
              disabled={loading || !email}
              sx={{ minWidth: 120 }}
            >
              {loading ? <CircularProgress size={20} /> : 'Join Waitlist'}
            </Button>
          </DialogActions>
        </Box>
      )}
    </Dialog>
  );
}
