'use client';

import { useRouter } from 'next/navigation';
import {
  Box,
  Typography,
  IconButton,
  Button,
  Menu,
  MenuItem,
  Divider,
  Avatar,
  ListItemIcon,
  ListItemText,
  Tooltip,
} from '@mui/material';
import PersonIcon from '@mui/icons-material/Person';
import LogoutIcon from '@mui/icons-material/Logout';
import { signOut } from 'next-auth/react';
import { useAuthStore } from '@/stores/authStore';
import { clearAuthToken } from '@/services/authToken';
import { designTokens } from '@/app/designTokens';
import { TierBadge } from './TierBadge';
import { useMenuAnchor } from '@/hooks/useMenuAnchor';

interface UserMenuProps {
  /** When true, just shows an avatar linking to /profile without dropdown */
  compact?: boolean;
  tier?: 'FREE' | 'PRO' | 'ELITE';
}

export function UserMenu({ compact = false, tier }: UserMenuProps) {
  const router = useRouter();
  const { isAuthenticated, name, email, avatarUrl } = useAuthStore();
  const menu = useMenuAnchor();

  if (!isAuthenticated) {
    return (
      <Button
        size="small"
        onClick={() => router.push('/auth/signin')}
        startIcon={<PersonIcon sx={{ fontSize: 16 }} />}
        sx={{
          color: 'text.secondary',
          textTransform: 'none',
          fontWeight: 500,
          '&:hover': { color: 'text.primary' },
        }}
      >
        Sign In
      </Button>
    );
  }

  const avatarElement = (
    <Avatar
      src={avatarUrl ?? undefined}
      alt={name ?? 'User'}
      sx={{
        width: compact ? 28 : 32,
        height: compact ? 28 : 32,
        bgcolor: 'primary.main',
        fontSize: 14,
      }}
    >
      {(name || 'U')[0].toUpperCase()}
    </Avatar>
  );

  if (compact) {
    return (
      <Tooltip title="Profile">
        <IconButton
          size="small"
          onClick={() => router.push('/profile')}
          sx={{ p: 0.5 }}
        >
          {avatarElement}
        </IconButton>
      </Tooltip>
    );
  }

  const handleSignOut = async () => {
    menu.close();
    clearAuthToken();
    await signOut({ callbackUrl: '/' });
  };

  return (
    <>
      <IconButton onClick={menu.open} sx={{ p: 0.5 }}>
        {avatarElement}
      </IconButton>

      <Menu
        anchorEl={menu.anchor}
        open={menu.isOpen}
        onClose={menu.close}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
        transformOrigin={{ vertical: 'top', horizontal: 'right' }}
        slotProps={{
          paper: {
            sx: {
              minWidth: 220,
              bgcolor: designTokens.colors.surface[3],
              border: '1px solid',
              borderColor: 'divider',
              mt: 0.5,
            },
          },
        }}
      >
        {/* User info header */}
        <Box sx={{ px: 2, py: 1.5 }}>
          <Typography variant="body2" fontWeight={600}>
            {name || 'User'}
          </Typography>
          {email && (
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
              {email}
            </Typography>
          )}
          {tier && tier !== 'FREE' && (
            <TierBadge tier={tier} sx={{ mt: 0.75 }} />
          )}
        </Box>

        <Divider />

        <MenuItem
          onClick={() => {
            menu.close();
            router.push('/profile');
          }}
        >
          <ListItemIcon>
            <PersonIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText>Profile</ListItemText>
        </MenuItem>

        <Divider />

        <MenuItem
          onClick={handleSignOut}
          sx={{
            color: 'error.main',
            '&:hover': {
              bgcolor: designTokens.alpha.error[10],
            },
          }}
        >
          <ListItemIcon>
            <LogoutIcon fontSize="small" sx={{ color: 'error.main' }} />
          </ListItemIcon>
          <ListItemText>Sign Out</ListItemText>
        </MenuItem>
      </Menu>
    </>
  );
}
