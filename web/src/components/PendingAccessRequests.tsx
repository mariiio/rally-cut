'use client';

import { useEffect, useState, useCallback, useRef } from 'react';
import {
  Badge,
  Box,
  CircularProgress,
  IconButton,
  List,
  ListItem,
  ListItemAvatar,
  ListItemText,
  Avatar,
  Popover,
  Typography,
  Divider,
  Tooltip,
} from '@mui/material';
import PersonAddIcon from '@mui/icons-material/PersonAdd';
import CheckIcon from '@mui/icons-material/Check';
import CloseIcon from '@mui/icons-material/Close';
import {
  getAccessRequests,
  getAccessRequestsCount,
  acceptAccessRequest,
  rejectAccessRequest,
  type AccessRequest,
} from '@/services/api';

interface PendingAccessRequestsProps {
  sessionId: string;
}

export function PendingAccessRequests({ sessionId }: PendingAccessRequestsProps) {
  const [anchorEl, setAnchorEl] = useState<HTMLElement | null>(null);
  const [pendingCount, setPendingCount] = useState(0);
  const [requests, setRequests] = useState<AccessRequest[]>([]);
  const [loading, setLoading] = useState(false);
  const [processingId, setProcessingId] = useState<string | null>(null);

  // Fetch pending count on mount and periodically
  const fetchCount = useCallback(async () => {
    try {
      const result = await getAccessRequestsCount(sessionId);
      setPendingCount(result.pending);
    } catch (err) {
      // Silently fail - not critical
      console.error('Failed to fetch access request count:', err);
    }
  }, [sessionId]);

  // Fetch on mount and set up polling
  const hasFetched = useRef(false);
  useEffect(() => {
    // Only fetch once on mount (avoid React Strict Mode double-fetch)
    if (!hasFetched.current) {
      hasFetched.current = true;
      fetchCount();
    }

    // Poll less frequently when no pending requests (2 min vs 30s)
    const pollInterval = pendingCount > 0 ? 30000 : 120000;
    const interval = setInterval(fetchCount, pollInterval);
    return () => clearInterval(interval);
  }, [fetchCount, pendingCount]);

  // Fetch full list when popover opens
  const handleOpen = async (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
    setLoading(true);

    try {
      const result = await getAccessRequests(sessionId);
      setRequests(result.requests);
    } catch (err) {
      console.error('Failed to fetch access requests:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

  const handleAccept = async (requestId: string) => {
    setProcessingId(requestId);
    try {
      await acceptAccessRequest(sessionId, requestId);
      // Remove from list and update count
      setRequests((prev) => prev.filter((r) => r.id !== requestId));
      setPendingCount((prev) => Math.max(0, prev - 1));
    } catch (err) {
      console.error('Failed to accept request:', err);
    } finally {
      setProcessingId(null);
    }
  };

  const handleReject = async (requestId: string) => {
    setProcessingId(requestId);
    try {
      await rejectAccessRequest(sessionId, requestId);
      // Remove from list and update count
      setRequests((prev) => prev.filter((r) => r.id !== requestId));
      setPendingCount((prev) => Math.max(0, prev - 1));
    } catch (err) {
      console.error('Failed to reject request:', err);
    } finally {
      setProcessingId(null);
    }
  };

  const open = Boolean(anchorEl);

  // Don't render anything if no pending requests
  if (pendingCount === 0 && !open) {
    return null;
  }

  return (
    <>
      <Tooltip title="Access Requests">
        <IconButton
          onClick={handleOpen}
          size="small"
          sx={{ ml: 1 }}
        >
          <Badge badgeContent={pendingCount} color="warning">
            <PersonAddIcon sx={{ fontSize: 20 }} />
          </Badge>
        </IconButton>
      </Tooltip>

      <Popover
        open={open}
        anchorEl={anchorEl}
        onClose={handleClose}
        anchorOrigin={{
          vertical: 'bottom',
          horizontal: 'right',
        }}
        transformOrigin={{
          vertical: 'top',
          horizontal: 'right',
        }}
        slotProps={{
          paper: {
            sx: {
              width: 320,
              maxHeight: 400,
              bgcolor: 'grey.800',
            },
          },
        }}
      >
        <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
          <Typography variant="subtitle1" fontWeight={600}>
            Access Requests
          </Typography>
          <Typography variant="body2" color="text.secondary">
            {pendingCount} pending {pendingCount === 1 ? 'request' : 'requests'}
          </Typography>
        </Box>

        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
            <CircularProgress size={24} />
          </Box>
        ) : requests.length === 0 ? (
          <Box sx={{ p: 3, textAlign: 'center' }}>
            <Typography color="text.secondary">
              No pending requests
            </Typography>
          </Box>
        ) : (
          <List dense sx={{ py: 0 }}>
            {requests.map((request, index) => (
              <Box key={request.id}>
                {index > 0 && <Divider />}
                <ListItem
                  sx={{
                    py: 1.5,
                    opacity: processingId === request.id ? 0.5 : 1,
                  }}
                  secondaryAction={
                    <Box sx={{ display: 'flex', gap: 0.5 }}>
                      <IconButton
                        size="small"
                        onClick={() => handleAccept(request.id)}
                        disabled={processingId !== null}
                        color="success"
                      >
                        <CheckIcon fontSize="small" />
                      </IconButton>
                      <IconButton
                        size="small"
                        onClick={() => handleReject(request.id)}
                        disabled={processingId !== null}
                        color="error"
                      >
                        <CloseIcon fontSize="small" />
                      </IconButton>
                    </Box>
                  }
                >
                  <ListItemAvatar>
                    <Avatar
                      src={request.userAvatarUrl || undefined}
                      sx={{ width: 32, height: 32 }}
                    >
                      {request.userName?.[0]?.toUpperCase() || '?'}
                    </Avatar>
                  </ListItemAvatar>
                  <ListItemText
                    primary={request.userName || 'Anonymous'}
                    secondary={
                      request.message || (
                        <Typography
                          component="span"
                          variant="body2"
                          color="text.disabled"
                          sx={{ fontStyle: 'italic' }}
                        >
                          No message
                        </Typography>
                      )
                    }
                    primaryTypographyProps={{ variant: 'body2', fontWeight: 500 }}
                    secondaryTypographyProps={{
                      variant: 'caption',
                      sx: {
                        display: '-webkit-box',
                        WebkitLineClamp: 2,
                        WebkitBoxOrient: 'vertical',
                        overflow: 'hidden',
                      },
                    }}
                  />
                </ListItem>
              </Box>
            ))}
          </List>
        )}
      </Popover>
    </>
  );
}
