'use client';

import { useState } from 'react';
import {
  Box,
  IconButton,
  Typography,
  Menu,
  MenuItem,
  ListItemIcon,
  ListItemText,
  Divider,
} from '@mui/material';
import UndoIcon from '@mui/icons-material/Undo';
import RedoIcon from '@mui/icons-material/Redo';
import MoreVertIcon from '@mui/icons-material/MoreVert';
import HomeIcon from '@mui/icons-material/Home';
import ShareIcon from '@mui/icons-material/Share';
import DownloadIcon from '@mui/icons-material/Download';
import { useRouter } from 'next/navigation';
import { useEditorStore } from '@/stores/editorStore';
import { designTokens } from '@/app/theme';

export function MobileEditorHeader() {
  const router = useRouter();
  const { session, singleVideoMode, undo, redo, canUndo, canRedo } = useEditorStore();
  const [menuAnchor, setMenuAnchor] = useState<null | HTMLElement>(null);

  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setMenuAnchor(event.currentTarget);
  };

  const handleMenuClose = () => {
    setMenuAnchor(null);
  };

  const handleHome = () => {
    handleMenuClose();
    router.push(singleVideoMode ? '/videos' : '/sessions');
  };

  return (
    <Box
      sx={{
        height: designTokens.mobile.header.height,
        display: 'flex',
        alignItems: 'center',
        px: 1,
        bgcolor: 'background.paper',
        borderBottom: '1px solid',
        borderColor: 'divider',
        flexShrink: 0,
      }}
    >
      {/* Session/Video Name */}
      <Typography
        variant="subtitle1"
        sx={{
          flex: 1,
          fontWeight: 600,
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
          px: 1,
        }}
      >
        {singleVideoMode
          ? session?.matches[0]?.name || 'Video'
          : session?.name || 'RallyCut'}
      </Typography>

      {/* Undo/Redo */}
      <IconButton
        size="small"
        onClick={undo}
        disabled={!canUndo()}
        sx={{ minWidth: designTokens.mobile.touchTarget }}
      >
        <UndoIcon fontSize="small" />
      </IconButton>
      <IconButton
        size="small"
        onClick={redo}
        disabled={!canRedo()}
        sx={{ minWidth: designTokens.mobile.touchTarget }}
      >
        <RedoIcon fontSize="small" />
      </IconButton>

      {/* Menu */}
      <IconButton
        size="small"
        onClick={handleMenuOpen}
        sx={{ minWidth: designTokens.mobile.touchTarget }}
      >
        <MoreVertIcon fontSize="small" />
      </IconButton>

      <Menu
        anchorEl={menuAnchor}
        open={Boolean(menuAnchor)}
        onClose={handleMenuClose}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
        transformOrigin={{ vertical: 'top', horizontal: 'right' }}
      >
        <MenuItem onClick={handleHome}>
          <ListItemIcon>
            <HomeIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText>Home</ListItemText>
        </MenuItem>
        {!singleVideoMode && <Divider />}
        {!singleVideoMode && (
          <MenuItem onClick={handleMenuClose} disabled>
            <ListItemIcon>
              <ShareIcon fontSize="small" />
            </ListItemIcon>
            <ListItemText>Share</ListItemText>
          </MenuItem>
        )}
        <MenuItem onClick={handleMenuClose} disabled>
          <ListItemIcon>
            <DownloadIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText>Export</ListItemText>
        </MenuItem>
      </Menu>
    </Box>
  );
}
