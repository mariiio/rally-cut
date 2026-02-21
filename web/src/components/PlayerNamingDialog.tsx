'use client';

import { useState, useCallback } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  Stack,
  Typography,
  Chip,
  Box,
} from '@mui/material';
import { savePlayerNamesApi, getMatchStatsApi, type MatchStats } from '@/services/api';
import { useEffect } from 'react';

// Track colors matching the player overlay colors
const TRACK_COLORS = ['#4caf50', '#2196f3', '#ff9800', '#9c27b0'];

interface PlayerNamingDialogProps {
  open: boolean;
  videoId: string;
  onClose: () => void;
}

export function PlayerNamingDialog({ open, videoId, onClose }: PlayerNamingDialogProps) {
  const [names, setNames] = useState<Record<string, string>>({});
  const [players, setPlayers] = useState<Array<{ trackId: number; team: string }>>([]);
  const [saving, setSaving] = useState(false);

  // Load player info from match stats
  useEffect(() => {
    if (!open || !videoId) return;

    getMatchStatsApi(videoId).then((stats: MatchStats | null) => {
      if (!stats?.playerStats) return;
      const activePlayers = stats.playerStats
        .filter((p) => p.totalActions > 0)
        .sort((a, b) => a.trackId - b.trackId);
      setPlayers(activePlayers.map((p) => ({ trackId: p.trackId, team: p.team })));
      // Initialize names with defaults
      const defaults: Record<string, string> = {};
      activePlayers.forEach((p) => {
        defaults[String(p.trackId)] = '';
      });
      setNames(defaults);
    }).catch(() => {
      // Fallback: 4 default players
      setPlayers([
        { trackId: 1, team: 'A' },
        { trackId: 2, team: 'A' },
        { trackId: 3, team: 'B' },
        { trackId: 4, team: 'B' },
      ]);
      setNames({ '1': '', '2': '', '3': '', '4': '' });
    });
  }, [open, videoId]);

  const handleSave = useCallback(async () => {
    setSaving(true);
    try {
      // Filter out empty names, use defaults for those
      const finalNames: Record<string, string> = {};
      for (const [trackId, name] of Object.entries(names)) {
        finalNames[trackId] = name.trim() || `Player ${trackId}`;
      }
      await savePlayerNamesApi(videoId, finalNames);
      onClose();
    } catch (err) {
      console.error('Failed to save player names:', err);
      onClose();
    } finally {
      setSaving(false);
    }
  }, [videoId, names, onClose]);

  const handleSkip = useCallback(() => {
    onClose();
  }, [onClose]);

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="xs"
      fullWidth
      PaperProps={{ sx: { bgcolor: 'background.paper' } }}
    >
      <DialogTitle sx={{ pb: 0.5 }}>
        <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
          Name the players
        </Typography>
        <Typography variant="caption" color="text.secondary">
          Optional — helps identify players in stats
        </Typography>
      </DialogTitle>
      <DialogContent sx={{ pt: 1 }}>
        <Stack spacing={1.5} sx={{ mt: 1 }}>
          {players.map((p, i) => (
            <Stack key={p.trackId} direction="row" alignItems="center" spacing={1.5}>
              <Box
                sx={{
                  width: 28,
                  height: 28,
                  borderRadius: '50%',
                  bgcolor: TRACK_COLORS[i % TRACK_COLORS.length],
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  flexShrink: 0,
                }}
              >
                <Typography variant="caption" sx={{ color: 'white', fontWeight: 600, fontSize: 11 }}>
                  {p.trackId}
                </Typography>
              </Box>
              <Chip
                label={`Team ${p.team}`}
                size="small"
                color={p.team === 'A' ? 'primary' : 'secondary'}
                sx={{ height: 20, fontSize: '0.65rem', flexShrink: 0 }}
              />
              <TextField
                size="small"
                placeholder={`Player ${p.trackId}`}
                value={names[String(p.trackId)] ?? ''}
                onChange={(e) =>
                  setNames((prev) => ({ ...prev, [String(p.trackId)]: e.target.value }))
                }
                inputProps={{ maxLength: 100 }}
                sx={{
                  flex: 1,
                  '& .MuiInputBase-input': { py: 0.75, fontSize: 13 },
                }}
              />
            </Stack>
          ))}
        </Stack>
      </DialogContent>
      <DialogActions sx={{ px: 3, pb: 2 }}>
        <Button onClick={handleSkip} sx={{ textTransform: 'none' }}>
          Skip
        </Button>
        <Button
          variant="contained"
          onClick={handleSave}
          disabled={saving}
          sx={{ textTransform: 'none' }}
        >
          {saving ? 'Saving...' : 'Save'}
        </Button>
      </DialogActions>
    </Dialog>
  );
}
