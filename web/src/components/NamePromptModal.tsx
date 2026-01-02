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
} from '@mui/material';
import { updateCurrentUser } from '@/services/api';
import { useEditorStore } from '@/stores/editorStore';

interface NamePromptModalProps {
  open: boolean;
  onClose: () => void;
  onNameSet: (name: string) => void;
}

export function NamePromptModal({ open, onClose, onNameSet }: NamePromptModalProps) {
  const [name, setName] = useState('');
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { setCurrentUser, currentUserId } = useEditorStore();

  const handleSubmit = async () => {
    if (!name.trim()) return;

    setSaving(true);
    setError(null);

    try {
      const user = await updateCurrentUser({ name: name.trim() });
      setCurrentUser(user.id, user.name);
      onNameSet(user.name || name.trim());
      onClose();
      setName('');
    } catch (err) {
      console.error('Failed to save name:', err);
      setError('Failed to save your name. Please try again.');
    } finally {
      setSaving(false);
    }
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="xs" fullWidth>
      <DialogTitle>What's your name?</DialogTitle>
      <DialogContent>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          Your name will be shown on highlights you create.
        </Typography>
        <TextField
          autoFocus
          fullWidth
          label="Your name"
          value={name}
          onChange={(e) => setName(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') handleSubmit();
          }}
          error={!!error}
          helperText={error}
          placeholder="e.g., John"
        />
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button
          onClick={handleSubmit}
          variant="contained"
          disabled={!name.trim() || saving}
        >
          {saving ? 'Saving...' : 'Save'}
        </Button>
      </DialogActions>
    </Dialog>
  );
}
