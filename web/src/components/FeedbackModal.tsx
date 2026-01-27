'use client';

import { useState, useEffect } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  CircularProgress,
  Alert,
  Stack,
  Typography,
  IconButton,
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import { submitFeedback, type FeedbackType } from '@/services/api';
import { useEditorStore } from '@/stores/editorStore';
import { useAuthStore } from '@/stores/authStore';

interface FeedbackModalProps {
  open: boolean;
  onClose: () => void;
}

const FEEDBACK_TYPES: { value: FeedbackType; label: string }[] = [
  { value: 'BUG', label: 'Bug Report' },
  { value: 'FEATURE', label: 'Feature Request' },
  { value: 'FEEDBACK', label: 'General Feedback' },
];

export function FeedbackModal({ open, onClose }: FeedbackModalProps) {
  const editorEmail = useEditorStore((state) => state.currentUserEmail);
  const authEmail = useAuthStore((state) => state.email);
  const userEmail = editorEmail || authEmail;

  const [type, setType] = useState<FeedbackType>('FEEDBACK');
  const [message, setMessage] = useState('');
  const [email, setEmail] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  // Reset form when modal closes
  useEffect(() => {
    if (!open) {
      // Delay reset to avoid visual glitch during close animation
      const timer = setTimeout(() => {
        setType('FEEDBACK');
        setMessage('');
        setEmail('');
        setError(null);
        setSuccess(false);
      }, 200);
      return () => clearTimeout(timer);
    }
  }, [open]);

  const handleSubmit = async () => {
    if (!message.trim()) return;

    try {
      setSubmitting(true);
      setError(null);

      await submitFeedback({
        type,
        message: message.trim(),
        email: !userEmail && email.trim() ? email.trim() : undefined,
        pageUrl: typeof window !== 'undefined' ? window.location.href : undefined,
      });

      setSuccess(true);

      // Auto-close after success
      setTimeout(() => {
        onClose();
      }, 1500);
    } catch (err) {
      console.error('Failed to submit feedback:', err);
      setError(err instanceof Error ? err.message : 'Failed to submit feedback');
    } finally {
      setSubmitting(false);
    }
  };

  const isValid = message.trim().length > 0;
  const showEmailField = !userEmail;

  return (
    <Dialog
      open={open}
      onClose={submitting ? undefined : onClose}
      maxWidth="sm"
      fullWidth
      PaperProps={{
        sx: { bgcolor: 'background.paper' },
      }}
    >
      <DialogTitle sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <span>Send Feedback</span>
        <IconButton
          size="small"
          onClick={onClose}
          disabled={submitting}
          sx={{ color: 'text.secondary' }}
        >
          <CloseIcon fontSize="small" />
        </IconButton>
      </DialogTitle>

      <DialogContent>
        {success ? (
          <Stack spacing={2} alignItems="center" sx={{ py: 4 }}>
            <CheckCircleIcon sx={{ fontSize: 48, color: 'success.main' }} />
            <Typography variant="h6">Thanks for your feedback!</Typography>
            <Typography color="text.secondary" textAlign="center">
              We appreciate you taking the time to help us improve.
            </Typography>
          </Stack>
        ) : (
          <Stack spacing={2.5} sx={{ pt: 1 }}>
            {error && (
              <Alert severity="error" onClose={() => setError(null)}>
                {error}
              </Alert>
            )}

            <FormControl fullWidth size="small">
              <InputLabel id="feedback-type-label">Type</InputLabel>
              <Select
                labelId="feedback-type-label"
                value={type}
                label="Type"
                onChange={(e) => setType(e.target.value as FeedbackType)}
                disabled={submitting}
              >
                {FEEDBACK_TYPES.map((t) => (
                  <MenuItem key={t.value} value={t.value}>
                    {t.label}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            <TextField
              label="Message"
              placeholder={
                type === 'BUG'
                  ? "Describe what happened, what you expected, and steps to reproduce..."
                  : type === 'FEATURE'
                    ? "Describe the feature you'd like to see..."
                    : "Share your thoughts with us..."
              }
              multiline
              rows={4}
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              disabled={submitting}
              fullWidth
              required
            />

            {showEmailField && (
              <TextField
                label="Email (optional)"
                placeholder="your@email.com"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                disabled={submitting}
                fullWidth
                helperText="So we can follow up with you"
                size="small"
              />
            )}
          </Stack>
        )}
      </DialogContent>

      {!success && (
        <DialogActions sx={{ px: 3, pb: 2 }}>
          <Button onClick={onClose} disabled={submitting}>
            Cancel
          </Button>
          <Button
            variant="contained"
            onClick={handleSubmit}
            disabled={submitting || !isValid}
            startIcon={submitting ? <CircularProgress size={16} color="inherit" /> : undefined}
          >
            {submitting ? 'Sending...' : 'Send Feedback'}
          </Button>
        </DialogActions>
      )}
    </Dialog>
  );
}
