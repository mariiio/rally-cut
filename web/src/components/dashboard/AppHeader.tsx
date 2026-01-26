'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { signOut } from 'next-auth/react';
import {
  Box,
  Container,
  Stack,
  Typography,
  IconButton,
  Button,
  Menu,
  MenuItem,
  Divider,
  Avatar,
  Chip,
  ListItemIcon,
  ListItemText,
} from '@mui/material';
import SportsVolleyballIcon from '@mui/icons-material/SportsVolleyball';
import PersonIcon from '@mui/icons-material/Person';
import LogoutIcon from '@mui/icons-material/Logout';
import DiamondIcon from '@mui/icons-material/Diamond';
import { useAuthStore } from '@/stores/authStore';
import { useTierStore } from '@/stores/tierStore';
import { clearAuthToken } from '@/services/authToken';
import { designTokens } from '@/app/theme';

export function AppHeader() {
  const router = useRouter();
  const [menuAnchor, setMenuAnchor] = useState<HTMLElement | null>(null);

  const { isAuthenticated, name, email, avatarUrl } = useAuthStore();
  const tier = useTierStore((s) => s.tier);
  const fetchTier = useTierStore((s) => s.fetchTier);

  useEffect(() => {
    fetchTier();
  }, [fetchTier]);

  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setMenuAnchor(event.currentTarget);
  };

  const handleMenuClose = () => {
    setMenuAnchor(null);
  };

  const handleSignOut = async () => {
    handleMenuClose();
    clearAuthToken();
    await signOut({ callbackUrl: '/' });
  };

  return (
    <Box
      component="header"
      sx={{
        position: 'sticky',
        top: 0,
        zIndex: 1100,
        height: designTokens.spacing.header,
        bgcolor: 'rgba(13, 14, 18, 0.8)',
        backdropFilter: 'blur(12px)',
        borderBottom: '1px solid',
        borderColor: 'divider',
        flexShrink: 0,
      }}
    >
      <Container
        maxWidth="lg"
        sx={{
          height: '100%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}
      >
        {/* Brand */}
        <Stack
          direction="row"
          alignItems="center"
          spacing={1}
          onClick={() => router.push('/sessions')}
          sx={{
            cursor: 'pointer',
            borderRadius: 1,
            px: 1,
            py: 0.5,
            mx: -1,
            transition: 'background-color 0.2s',
            '&:hover': {
              bgcolor: 'action.hover',
            },
          }}
        >
          <SportsVolleyballIcon
            sx={{
              fontSize: 28,
              color: 'primary.main',
              filter: 'drop-shadow(0 2px 4px rgba(255, 107, 74, 0.3))',
            }}
          />
          <Typography
            variant="h6"
            sx={{
              fontWeight: 700,
              background: designTokens.gradients.primary,
              backgroundClip: 'text',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              letterSpacing: '-0.02em',
            }}
          >
            RallyCut
          </Typography>
        </Stack>

        {/* Auth UI */}
        {isAuthenticated ? (
          <>
            <IconButton
              onClick={handleMenuOpen}
              sx={{ p: 0.5 }}
            >
              <Avatar
                src={avatarUrl ?? undefined}
                alt={name ?? 'User'}
                sx={{ width: 32, height: 32, bgcolor: 'primary.main', fontSize: 14 }}
              >
                {(name || 'U')[0].toUpperCase()}
              </Avatar>
            </IconButton>

            <Menu
              anchorEl={menuAnchor}
              open={Boolean(menuAnchor)}
              onClose={handleMenuClose}
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
                {tier !== 'FREE' && (
                  <Chip
                    icon={<DiamondIcon sx={{ fontSize: 12 }} />}
                    label={tier === 'ELITE' ? 'Elite' : 'Pro'}
                    size="small"
                    variant="outlined"
                    sx={{
                      mt: 0.75,
                      height: 22,
                      borderColor: 'rgba(255, 209, 102, 0.5)',
                      color: designTokens.colors.tertiary.main,
                      fontWeight: 600,
                      fontSize: '0.7rem',
                      '& .MuiChip-icon': {
                        color: designTokens.colors.tertiary.main,
                      },
                    }}
                  />
                )}
              </Box>

              <Divider />

              <MenuItem
                onClick={() => {
                  handleMenuClose();
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
                    bgcolor: 'rgba(239, 68, 68, 0.1)',
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
        ) : (
          <Stack direction="row" spacing={1} alignItems="center">
            <Button
              onClick={() => router.push('/auth/signin')}
              sx={{
                color: 'text.secondary',
                textTransform: 'none',
                fontWeight: 500,
                '&:hover': { color: 'text.primary' },
              }}
            >
              Sign In
            </Button>
            <Button
              variant="contained"
              onClick={() => router.push('/auth/register')}
              sx={{
                display: { xs: 'none', sm: 'inline-flex' },
              }}
            >
              Create Account
            </Button>
          </Stack>
        )}
      </Container>
    </Box>
  );
}
