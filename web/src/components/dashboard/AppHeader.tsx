'use client';

import { useState, useEffect } from 'react';
import { useRouter, usePathname } from 'next/navigation';
import {
  Box,
  Container,
  Stack,
  IconButton,
  Button,
  Tabs,
  Tab,
  Tooltip,
} from '@mui/material';
import ChatBubbleOutlineIcon from '@mui/icons-material/ChatBubbleOutline';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import { useAuthStore } from '@/stores/authStore';
import { useTierStore } from '@/stores/tierStore';
import { designTokens } from '@/app/designTokens';
import { FeedbackModal } from '@/components/FeedbackModal';
import { BrandLogo } from '@/components/BrandLogo';
import { UserMenu } from '@/components/UserMenu';

export function AppHeader() {
  const router = useRouter();
  const pathname = usePathname();
  const [showFeedbackModal, setShowFeedbackModal] = useState(false);

  const { isAuthenticated } = useAuthStore();
  const tier = useTierStore((s) => s.tier);
  const fetchTier = useTierStore((s) => s.fetchTier);

  useEffect(() => {
    fetchTier();
  }, [fetchTier]);

  // Derive active tab from pathname
  const activeTab = pathname.startsWith('/videos')
    ? '/videos'
    : pathname.startsWith('/sessions')
      ? '/sessions'
      : false;

  return (
    <>
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
          {/* Left zone: Brand + Nav Tabs */}
          <Stack direction="row" alignItems="center" spacing={1}>
            {/* Brand */}
            <BrandLogo onClick={() => router.push('/sessions')} />

            {/* Navigation Tabs */}
            <Tabs
              value={activeTab}
              onChange={(_, value) => router.push(value)}
              sx={{
                display: { xs: 'none', sm: 'flex' },
                ml: 2,
                minHeight: designTokens.spacing.header,
                '& .MuiTabs-indicator': {
                  height: 2,
                },
              }}
            >
              <Tab
                label="Sessions"
                value="/sessions"
                disableRipple
                sx={{
                  minHeight: designTokens.spacing.header,
                  textTransform: 'none',
                  fontWeight: 500,
                  '&.Mui-selected': {
                    fontWeight: 600,
                    color: 'text.primary',
                  },
                }}
              />
              <Tab
                label="Videos"
                value="/videos"
                disableRipple
                sx={{
                  minHeight: designTokens.spacing.header,
                  textTransform: 'none',
                  fontWeight: 500,
                  '&.Mui-selected': {
                    fontWeight: 600,
                    color: 'text.primary',
                  },
                }}
              />
            </Tabs>
          </Stack>

          {/* Right zone */}
          {isAuthenticated ? (
            <Stack direction="row" alignItems="center" spacing={1}>
              <Tooltip title="Send feedback">
                <IconButton
                  onClick={() => setShowFeedbackModal(true)}
                  sx={{ color: 'text.secondary' }}
                >
                  <ChatBubbleOutlineIcon sx={{ fontSize: 20 }} />
                </IconButton>
              </Tooltip>

              {tier === 'FREE' && (
                <Button
                  size="small"
                  variant="outlined"
                  startIcon={<AutoAwesomeIcon sx={{ fontSize: 16 }} />}
                  onClick={() => router.push('/upgrade')}
                  sx={{
                    color: designTokens.colors.tertiary.main,
                    borderColor: designTokens.alpha.tertiary[40],
                    textTransform: 'none',
                    fontWeight: 600,
                    '&:hover': {
                      borderColor: designTokens.colors.tertiary.main,
                      bgcolor: designTokens.alpha.tertiary[8],
                    },
                  }}
                >
                  Upgrade
                </Button>
              )}

              <UserMenu tier={tier} />
            </Stack>
          ) : (
            <Stack direction="row" spacing={1} alignItems="center">
              <Tooltip title="Send feedback">
                <IconButton
                  onClick={() => setShowFeedbackModal(true)}
                  sx={{ color: 'text.secondary' }}
                >
                  <ChatBubbleOutlineIcon sx={{ fontSize: 20 }} />
                </IconButton>
              </Tooltip>
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

      <FeedbackModal open={showFeedbackModal} onClose={() => setShowFeedbackModal(false)} />
    </>
  );
}
