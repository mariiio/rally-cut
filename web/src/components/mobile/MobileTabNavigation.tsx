'use client';

import { BottomNavigation, BottomNavigationAction, Badge, Box } from '@mui/material';
import ListAltIcon from '@mui/icons-material/ListAlt';
import StarIcon from '@mui/icons-material/Star';
import { designTokens } from '@/app/theme';

interface MobileTabNavigationProps {
  activeTab: 'rallies' | 'highlights';
  onTabChange: (tab: 'rallies' | 'highlights') => void;
  rallyCount: number;
  highlightCount: number;
}

export function MobileTabNavigation({
  activeTab,
  onTabChange,
  rallyCount,
  highlightCount,
}: MobileTabNavigationProps) {
  return (
    <Box
      sx={{
        position: 'fixed',
        bottom: 0,
        left: 0,
        right: 0,
        zIndex: 1100,
        // Safe area for iOS notch/home indicator
        paddingBottom: designTokens.mobile.bottomNav.safeAreaPadding,
        bgcolor: 'background.paper',
        borderTop: '1px solid',
        borderColor: 'divider',
      }}
    >
      <BottomNavigation
        value={activeTab}
        onChange={(_, newValue) => onTabChange(newValue)}
        sx={{
          height: designTokens.mobile.bottomNav.height,
          bgcolor: 'transparent',
          '& .MuiBottomNavigationAction-root': {
            minWidth: 'auto',
            py: 1,
            color: 'text.secondary',
            '&.Mui-selected': {
              color: 'primary.main',
            },
          },
          '& .MuiBottomNavigationAction-label': {
            fontSize: '0.75rem',
            '&.Mui-selected': {
              fontSize: '0.75rem',
            },
          },
        }}
      >
        <BottomNavigationAction
          label="Rallies"
          value="rallies"
          icon={
            <Badge
              badgeContent={rallyCount}
              color="primary"
              max={99}
              sx={{
                '& .MuiBadge-badge': {
                  fontSize: '0.625rem',
                  height: 16,
                  minWidth: 16,
                },
              }}
            >
              <ListAltIcon />
            </Badge>
          }
        />
        <BottomNavigationAction
          label="Highlights"
          value="highlights"
          icon={
            <Badge
              badgeContent={highlightCount}
              color="secondary"
              max={99}
              sx={{
                '& .MuiBadge-badge': {
                  fontSize: '0.625rem',
                  height: 16,
                  minWidth: 16,
                  bgcolor: 'secondary.main',
                },
              }}
            >
              <StarIcon />
            </Badge>
          }
        />
      </BottomNavigation>
    </Box>
  );
}
