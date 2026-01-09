'use client';

import { useEffect, useState, useMemo } from 'react';
import { useRouter } from 'next/navigation';
import {
  Box,
  Container,
  Typography,
  Card,
  CardContent,
  Grid,
  Chip,
  IconButton,
  Pagination,
  TextField,
  InputAdornment,
  Menu,
  MenuItem,
  ListItemIcon,
  ListItemText,
  Skeleton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
  Button,
  Divider,
} from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import FolderIcon from '@mui/icons-material/Folder';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import MoreVertIcon from '@mui/icons-material/MoreVert';
import EditIcon from '@mui/icons-material/Edit';
import DeleteIcon from '@mui/icons-material/Delete';
import ChevronRightIcon from '@mui/icons-material/ChevronRight';
import { designTokens } from '@/app/theme';
import {
  listVideos,
  listSessions,
  getVideoStreamUrl,
  deleteVideo,
  renameVideo,
  type VideoListItem,
  type SessionType,
} from '@/services/api';

function formatDuration(ms: number | null): string {
  if (!ms) return '--:--';
  const totalSeconds = Math.floor(ms / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, '0')}`;
}

interface VideoSession {
  id: string;
  name: string;
  type: SessionType;
}

export default function VideosPage() {
  const router = useRouter();
  const [videos, setVideos] = useState<VideoListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [search, setSearch] = useState('');
  const [searchInput, setSearchInput] = useState('');
  const [allVideosSessionId, setAllVideosSessionId] = useState<string | null>(null);
  const limit = 20;

  // Video menu state
  const [menuAnchor, setMenuAnchor] = useState<{
    element: HTMLElement;
    video: VideoListItem;
  } | null>(null);

  // Session submenu state
  const [sessionSubmenuAnchor, setSessionSubmenuAnchor] = useState<HTMLElement | null>(null);

  // Rename dialog state
  const [renameDialogOpen, setRenameDialogOpen] = useState(false);
  const [renameVideo_target, setRenameVideoTarget] = useState<VideoListItem | null>(null);
  const [newName, setNewName] = useState('');
  const [renaming, setRenaming] = useState(false);

  // Delete dialog state
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [deleteVideo_target, setDeleteVideoTarget] = useState<VideoListItem | null>(null);
  const [deleting, setDeleting] = useState(false);

  // Filter out PENDING videos
  const readyVideos = useMemo(
    () => videos.filter((v) => v.status !== 'PENDING'),
    [videos]
  );

  useEffect(() => {
    loadVideos();
  }, [page, search]);

  // Fetch ALL_VIDEOS session ID on mount
  useEffect(() => {
    const fetchAllVideosSession = async () => {
      try {
        const result = await listSessions();
        const allVideosSession = result.data.find(s => s.type === 'ALL_VIDEOS');
        if (allVideosSession) {
          setAllVideosSessionId(allVideosSession.id);
        }
      } catch (error) {
        console.error('Failed to fetch sessions:', error);
      }
    };
    fetchAllVideosSession();
  }, []);

  // Debounce search input
  useEffect(() => {
    const timer = setTimeout(() => {
      if (searchInput !== search) {
        setSearch(searchInput);
        setPage(1);
      }
    }, 300);
    return () => clearTimeout(timer);
  }, [searchInput, search]);

  const loadVideos = async () => {
    try {
      setLoading(true);
      const result = await listVideos(page, limit, search || undefined);
      setVideos(result.data);
      setTotalPages(result.pagination.totalPages);
    } catch (error) {
      console.error('Failed to load videos:', error);
    } finally {
      setLoading(false);
    }
  };

  const getRegularSessions = (video: VideoListItem): VideoSession[] => {
    return (video.sessions || []).filter((s) => s.type !== 'ALL_VIDEOS');
  };

  // Menu handlers
  const handleMenuOpen = (event: React.MouseEvent<HTMLButtonElement>, video: VideoListItem) => {
    event.stopPropagation();
    setMenuAnchor({ element: event.currentTarget, video });
  };

  const handleMenuClose = () => {
    setMenuAnchor(null);
    setSessionSubmenuAnchor(null);
  };

  const handleSessionSubmenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setSessionSubmenuAnchor(event.currentTarget);
  };

  const handleSessionSubmenuClose = () => {
    setSessionSubmenuAnchor(null);
  };

  const handleGoToSession = (sessionId: string) => {
    router.push(`/sessions/${sessionId}`);
    handleMenuClose();
  };

  // Rename handlers
  const handleRenameClick = () => {
    if (menuAnchor?.video) {
      setRenameVideoTarget(menuAnchor.video);
      setNewName(menuAnchor.video.name);
      setRenameDialogOpen(true);
    }
    handleMenuClose();
  };

  const handleRenameConfirm = async () => {
    if (!renameVideo_target || !newName.trim()) return;
    try {
      setRenaming(true);
      await renameVideo(renameVideo_target.id, newName.trim());
      setRenameDialogOpen(false);
      setRenameVideoTarget(null);
      setNewName('');
      await loadVideos();
    } catch (error) {
      console.error('Failed to rename video:', error);
    } finally {
      setRenaming(false);
    }
  };

  const handleRenameCancel = () => {
    setRenameDialogOpen(false);
    setRenameVideoTarget(null);
    setNewName('');
  };

  // Delete handlers
  const handleDeleteClick = () => {
    if (menuAnchor?.video) {
      setDeleteVideoTarget(menuAnchor.video);
      setDeleteDialogOpen(true);
    }
    handleMenuClose();
  };

  const handleDeleteConfirm = async () => {
    if (!deleteVideo_target) return;
    try {
      setDeleting(true);
      await deleteVideo(deleteVideo_target.id);
      setDeleteDialogOpen(false);
      setDeleteVideoTarget(null);
      await loadVideos();
    } catch (error) {
      console.error('Failed to delete video:', error);
    } finally {
      setDeleting(false);
    }
  };

  const handleDeleteCancel = () => {
    setDeleteDialogOpen(false);
    setDeleteVideoTarget(null);
  };

  return (
    <Box
      sx={{
        minHeight: '100vh',
        bgcolor: 'background.default',
        pb: 4,
      }}
    >
      {/* Header */}
      <Box
        sx={{
          borderBottom: '1px solid',
          borderColor: 'divider',
          bgcolor: designTokens.colors.surface[1],
        }}
      >
        <Container maxWidth="xl" sx={{ py: 3 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 3 }}>
            <IconButton
              onClick={() => router.push('/sessions')}
              sx={{ color: 'text.secondary' }}
            >
              <ArrowBackIcon />
            </IconButton>
            <Typography variant="h4" component="h1" fontWeight={700}>
              Videos
            </Typography>
          </Box>

          {/* Search */}
          <TextField
            placeholder="Search videos..."
            value={searchInput}
            onChange={(e) => setSearchInput(e.target.value)}
            size="small"
            sx={{
              maxWidth: 400,
              '& .MuiOutlinedInput-root': {
                bgcolor: designTokens.colors.surface[0],
              },
            }}
            slotProps={{
              input: {
                startAdornment: (
                  <InputAdornment position="start">
                    <SearchIcon sx={{ color: 'text.secondary' }} />
                  </InputAdornment>
                ),
              },
            }}
          />
        </Container>
      </Box>

      {/* Content */}
      <Container maxWidth="xl" sx={{ mt: 4 }}>
        {loading ? (
          <Grid container spacing={2}>
            {Array.from({ length: 8 }).map((_, i) => (
              <Grid size={{ xs: 6, sm: 4, md: 3, lg: 2 }} key={i}>
                <Skeleton
                  variant="rectangular"
                  sx={{ aspectRatio: '16/9', borderRadius: 2 }}
                />
                <Skeleton sx={{ mt: 1 }} />
              </Grid>
            ))}
          </Grid>
        ) : readyVideos.length === 0 ? (
          <Box sx={{ textAlign: 'center', py: 8 }}>
            <Typography variant="h6" color="text.secondary">
              {search ? 'No videos found matching your search' : 'No videos yet'}
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
              {!search && 'Upload videos from the Sessions page to get started'}
            </Typography>
          </Box>
        ) : (
          <>
            <Grid container spacing={2}>
              {readyVideos.map((video) => {
                const regularSessions = getRegularSessions(video);

                return (
                  <Grid size={{ xs: 6, sm: 4, md: 3, lg: 2 }} key={video.id}>
                    <Card
                      sx={{
                        bgcolor: designTokens.colors.surface[2],
                        borderRadius: 2,
                        overflow: 'hidden',
                        border: '1px solid',
                        borderColor: 'transparent',
                        transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
                        cursor: 'pointer',
                        '&:hover': {
                          transform: 'translateY(-4px) scale(1.02)',
                          boxShadow: designTokens.shadows.lg,
                          borderColor: 'primary.main',
                          '& .thumbnail-overlay': {
                            opacity: 1,
                          },
                          '& .play-icon': {
                            transform: 'translate(-50%, -50%) scale(1)',
                            opacity: 1,
                          },
                        },
                      }}
                      onClick={() => {
                        if (allVideosSessionId) {
                          router.push(`/sessions/${allVideosSessionId}?video=${video.id}`);
                        }
                      }}
                    >
                      <Box sx={{ position: 'relative', aspectRatio: '16/9' }}>
                        <video
                          src={getVideoStreamUrl(video.s3Key)}
                          poster={
                            video.posterS3Key
                              ? getVideoStreamUrl(video.posterS3Key)
                              : undefined
                          }
                          preload="metadata"
                          muted
                          style={{
                            width: '100%',
                            height: '100%',
                            objectFit: 'cover',
                            backgroundColor: designTokens.colors.surface[0],
                            display: 'block',
                          }}
                        />
                        {/* Gradient overlay */}
                        <Box
                          sx={{
                            position: 'absolute',
                            inset: 0,
                            background:
                              'linear-gradient(to top, rgba(0,0,0,0.6) 0%, rgba(0,0,0,0.1) 40%, transparent 100%)',
                            pointerEvents: 'none',
                          }}
                        />
                        {/* Hover overlay */}
                        <Box
                          className="thumbnail-overlay"
                          sx={{
                            position: 'absolute',
                            inset: 0,
                            background: 'rgba(0, 0, 0, 0.3)',
                            opacity: 0,
                            transition: 'opacity 0.2s ease',
                            pointerEvents: 'none',
                          }}
                        />
                        {/* Play icon */}
                        <Box
                          className="play-icon"
                          sx={{
                            position: 'absolute',
                            top: '50%',
                            left: '50%',
                            transform: 'translate(-50%, -50%) scale(0.8)',
                            opacity: 0,
                            transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
                            width: 40,
                            height: 40,
                            borderRadius: '50%',
                            bgcolor: 'rgba(255, 107, 74, 0.9)',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            boxShadow: designTokens.shadows.glow.primary,
                            pointerEvents: 'none',
                          }}
                        >
                          <PlayArrowIcon sx={{ color: 'white', fontSize: 24 }} />
                        </Box>
                        {/* Duration badge */}
                        <Chip
                          label={formatDuration(video.durationMs)}
                          size="small"
                          sx={{
                            position: 'absolute',
                            bottom: 6,
                            right: 6,
                            bgcolor: 'rgba(0, 0, 0, 0.75)',
                            color: 'white',
                            fontSize: '0.7rem',
                            fontWeight: 600,
                            height: 22,
                            backdropFilter: 'blur(4px)',
                          }}
                        />
                      </Box>
                      <CardContent
                        sx={{ py: 1.5, px: 1.5, '&:last-child': { pb: 1.5 } }}
                      >
                        <Box
                          sx={{
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'space-between',
                            gap: 1,
                          }}
                        >
                          <Typography
                            variant="body2"
                            noWrap
                            title={video.name}
                            sx={{ fontWeight: 500, color: 'text.primary', flex: 1 }}
                          >
                            {video.name}
                          </Typography>
                          <IconButton
                            size="small"
                            onClick={(e) => handleMenuOpen(e, video)}
                            sx={{
                              color: 'text.secondary',
                              '&:hover': {
                                color: 'text.primary',
                                bgcolor: 'action.hover',
                              },
                            }}
                          >
                            <MoreVertIcon fontSize="small" />
                          </IconButton>
                        </Box>
                      </CardContent>
                    </Card>
                  </Grid>
                );
              })}
            </Grid>

            {/* Pagination */}
            {totalPages > 1 && (
              <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}>
                <Pagination
                  count={totalPages}
                  page={page}
                  onChange={(_, newPage) => setPage(newPage)}
                  color="primary"
                  sx={{
                    '& .MuiPaginationItem-root': {
                      color: 'text.primary',
                    },
                  }}
                />
              </Box>
            )}
          </>
        )}
      </Container>

      {/* Video menu */}
      <Menu
        anchorEl={menuAnchor?.element}
        open={Boolean(menuAnchor)}
        onClose={handleMenuClose}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
        transformOrigin={{ vertical: 'top', horizontal: 'right' }}
      >
        {/* Go to Session - only show if video has sessions */}
        {menuAnchor?.video && getRegularSessions(menuAnchor.video).length > 0 && [
          getRegularSessions(menuAnchor.video).length === 1 ? (
            <MenuItem
              key="session"
              onClick={() => handleGoToSession(getRegularSessions(menuAnchor.video)[0].id)}
            >
              <ListItemIcon>
                <FolderIcon fontSize="small" />
              </ListItemIcon>
              <ListItemText primary={`Go to ${getRegularSessions(menuAnchor.video)[0].name}`} />
            </MenuItem>
          ) : (
            <MenuItem
              key="sessions"
              onClick={handleSessionSubmenuOpen}
              sx={{ display: 'flex', justifyContent: 'space-between' }}
            >
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <ListItemIcon>
                  <FolderIcon fontSize="small" />
                </ListItemIcon>
                <ListItemText primary="Go to session" />
              </Box>
              <ChevronRightIcon fontSize="small" sx={{ color: 'text.secondary' }} />
            </MenuItem>
          ),
          <Divider key="divider1" />,
        ]}
        <MenuItem onClick={handleRenameClick}>
          <ListItemIcon>
            <EditIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText primary="Rename" />
        </MenuItem>
        <MenuItem onClick={handleDeleteClick} sx={{ color: 'error.main' }}>
          <ListItemIcon>
            <DeleteIcon fontSize="small" sx={{ color: 'error.main' }} />
          </ListItemIcon>
          <ListItemText primary="Delete" />
        </MenuItem>
      </Menu>

      {/* Session submenu */}
      <Menu
        anchorEl={sessionSubmenuAnchor}
        open={Boolean(sessionSubmenuAnchor)}
        onClose={handleSessionSubmenuClose}
        anchorOrigin={{ vertical: 'top', horizontal: 'right' }}
        transformOrigin={{ vertical: 'top', horizontal: 'left' }}
      >
        {menuAnchor?.video &&
          getRegularSessions(menuAnchor.video).map((session) => (
            <MenuItem key={session.id} onClick={() => handleGoToSession(session.id)}>
              <ListItemIcon>
                <FolderIcon fontSize="small" />
              </ListItemIcon>
              <ListItemText primary={session.name} />
            </MenuItem>
          ))}
      </Menu>

      {/* Rename dialog */}
      <Dialog open={renameDialogOpen} onClose={handleRenameCancel} maxWidth="xs" fullWidth>
        <DialogTitle>Rename Video</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            fullWidth
            label="Video name"
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && newName.trim()) {
                handleRenameConfirm();
              }
            }}
            sx={{ mt: 1 }}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={handleRenameCancel} disabled={renaming}>
            Cancel
          </Button>
          <Button
            onClick={handleRenameConfirm}
            variant="contained"
            disabled={renaming || !newName.trim()}
          >
            {renaming ? 'Renaming...' : 'Rename'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Delete confirmation dialog */}
      <Dialog
        open={deleteDialogOpen}
        onClose={handleDeleteCancel}
        maxWidth="xs"
        fullWidth
        PaperProps={{
          sx: {
            borderRadius: 3,
            overflow: 'hidden',
          },
        }}
      >
        <DialogTitle sx={{ pb: 1 }}>Delete Video Permanently</DialogTitle>

        <DialogContent sx={{ pt: 3 }}>
          <Typography variant="body1" textAlign="center" sx={{ mb: 2 }}>
            Are you sure you want to permanently delete
          </Typography>
          <Typography
            variant="body1"
            fontWeight={600}
            textAlign="center"
            sx={{
              bgcolor: 'action.hover',
              py: 1,
              px: 2,
              borderRadius: 1,
              mb: 3,
              wordBreak: 'break-word',
            }}
          >
            {deleteVideo_target?.name}
          </Typography>

          <Box
            sx={{
              bgcolor: 'rgba(239, 68, 68, 0.08)',
              border: '1px solid',
              borderColor: 'error.dark',
              borderRadius: 2,
              p: 2,
              mb: 2,
            }}
          >
            <Typography variant="body2" color="text.secondary">
              This action cannot be undone. The video file and all associated data will be permanently deleted:
            </Typography>
            <Box component="ul" sx={{ m: 0, mt: 1, pl: 2.5 }}>
              <Typography component="li" variant="body2" color="text.secondary">
                Rally detections and edits
              </Typography>
              <Typography component="li" variant="body2" color="text.secondary">
                Camera movement edits
              </Typography>
              <Typography component="li" variant="body2" color="text.secondary">
                Rallies removed from any highlights
              </Typography>
            </Box>
          </Box>
        </DialogContent>

        <DialogActions sx={{ px: 3, pb: 3, gap: 1 }}>
          <Button
            onClick={handleDeleteCancel}
            disabled={deleting}
            variant="outlined"
            fullWidth
            sx={{ py: 1.25 }}
          >
            Cancel
          </Button>
          <Button
            onClick={handleDeleteConfirm}
            variant="contained"
            color="error"
            disabled={deleting}
            fullWidth
            sx={{ py: 1.25 }}
          >
            {deleting ? 'Deleting...' : 'Delete Permanently'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
