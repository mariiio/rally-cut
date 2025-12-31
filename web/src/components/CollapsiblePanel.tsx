'use client';

import { ReactNode } from 'react';
import {
  Box,
  Typography,
  IconButton,
  Chip,
  Tooltip,
} from '@mui/material';
import ChevronLeftIcon from '@mui/icons-material/ChevronLeft';
import ChevronRightIcon from '@mui/icons-material/ChevronRight';
import { designTokens } from '@/app/theme';

interface CollapsiblePanelProps {
  title: string;
  count?: number;
  collapsed: boolean;
  onToggle: () => void;
  position: 'left' | 'right';
  collapsedIcon?: ReactNode;
  headerAction?: ReactNode;
  footer?: ReactNode;
  children: ReactNode;
}

export function CollapsiblePanel({
  title,
  count,
  collapsed,
  onToggle,
  position,
  collapsedIcon,
  headerAction,
  footer,
  children,
}: CollapsiblePanelProps) {
  const expandedWidth = position === 'left'
    ? designTokens.spacing.panel.expanded.left
    : designTokens.spacing.panel.expanded.right;

  const collapsedWidth = designTokens.spacing.panel.collapsed;

  const isLeft = position === 'left';

  return (
    <Box
      sx={{
        width: collapsed ? collapsedWidth : expandedWidth,
        transition: designTokens.transitions.normal,
        display: 'flex',
        flexDirection: 'column',
        bgcolor: 'background.paper',
        borderRight: isLeft ? '1px solid' : 'none',
        borderLeft: isLeft ? 'none' : '1px solid',
        borderColor: 'divider',
        overflow: 'hidden',
        flexShrink: 0,
      }}
    >
      {/* Header */}
      <Box
        sx={{
          height: 48,
          display: 'flex',
          alignItems: 'center',
          px: collapsed ? 0 : 1.5,
          justifyContent: collapsed ? 'center' : 'flex-start',
          borderBottom: '1px solid',
          borderColor: 'divider',
          flexShrink: 0,
        }}
      >
        {collapsed ? (
          <Tooltip title={`Expand ${title}`} placement={isLeft ? 'right' : 'left'}>
            <IconButton
              size="small"
              onClick={onToggle}
              sx={{
                color: 'text.secondary',
                '&:hover': { color: 'text.primary' },
              }}
            >
              {isLeft ? <ChevronRightIcon /> : <ChevronLeftIcon />}
            </IconButton>
          </Tooltip>
        ) : (
          <>
            <Typography
              variant="overline"
              sx={{
                flex: 1,
                color: 'text.secondary',
                fontWeight: 600,
              }}
            >
              {title}
            </Typography>

            {count !== undefined && (
              <Chip
                label={count}
                size="small"
                sx={{
                  mr: 1,
                  height: 20,
                  minWidth: 28,
                  '& .MuiChip-label': { px: 0.75 },
                }}
              />
            )}

            {headerAction}

            <Tooltip title={`Collapse ${title}`}>
              <IconButton
                size="small"
                onClick={onToggle}
                sx={{
                  ml: 0.5,
                  color: 'text.secondary',
                  '&:hover': { color: 'text.primary' },
                }}
              >
                {isLeft ? <ChevronLeftIcon fontSize="small" /> : <ChevronRightIcon fontSize="small" />}
              </IconButton>
            </Tooltip>
          </>
        )}
      </Box>

      {/* Content */}
      {collapsed ? (
        <Box
          sx={{
            flex: 1,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            pt: 1,
            gap: 0.5,
          }}
        >
          {collapsedIcon}
        </Box>
      ) : (
        <Box
          sx={{
            flex: 1,
            overflow: 'auto',
            display: 'flex',
            flexDirection: 'column',
          }}
        >
          {children}
        </Box>
      )}

      {/* Footer */}
      {footer && !collapsed && (
        <Box
          sx={{
            borderTop: '1px solid',
            borderColor: 'divider',
            p: 1.5,
            flexShrink: 0,
          }}
        >
          {footer}
        </Box>
      )}
    </Box>
  );
}
