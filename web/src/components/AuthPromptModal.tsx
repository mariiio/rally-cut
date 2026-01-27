'use client';

import { useRouter } from 'next/navigation';
import {
  Box,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Typography,
  Stack,
} from '@mui/material';
import SportsVolleyballIcon from '@mui/icons-material/SportsVolleyball';
import { useAuthStore } from '@/stores/authStore';

export function AuthPromptModal() {
  const router = useRouter();
  const { showAuthModal, authModalReason, closeAuthPrompt } = useAuthStore();

  const handleSignIn = () => {
    closeAuthPrompt();
    const reason = authModalReason ? `?reason=${encodeURIComponent(authModalReason)}` : '';
    router.push(`/auth/signin${reason}`);
  };

  const handleRegister = () => {
    closeAuthPrompt();
    router.push('/auth/register');
  };

  return (
    <Dialog
      open={showAuthModal}
      onClose={closeAuthPrompt}
      maxWidth="xs"
      fullWidth
    >
      <DialogTitle sx={{ pb: 1 }}>
        <Stack direction="row" alignItems="center" spacing={1}>
          <SportsVolleyballIcon sx={{ color: 'primary.main' }} />
          <Typography variant="h6" fontWeight={600}>
            Sign in required
          </Typography>
        </Stack>
      </DialogTitle>
      <DialogContent>
        <Typography color="text.secondary">
          {authModalReason || 'Create an account or sign in to access this feature.'}
        </Typography>
      </DialogContent>
      <DialogActions sx={{ px: 3, pb: 2.5, gap: 1 }}>
        <Button onClick={closeAuthPrompt} color="inherit" sx={{ color: 'text.secondary' }}>
          Continue as Guest
        </Button>
        <Box sx={{ flexGrow: 1 }} />
        <Button onClick={handleSignIn} variant="outlined">
          Sign In
        </Button>
        <Button onClick={handleRegister} variant="contained">
          Create Account
        </Button>
      </DialogActions>
    </Dialog>
  );
}
