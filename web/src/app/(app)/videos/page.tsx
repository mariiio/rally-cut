'use client';

import { Suspense, useEffect, useState, useMemo, useCallback } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import {
  Box,
  Container,
  Typography,
  Grid,
  IconButton,
  Pagination,
  TextField,
  InputAdornment,
  Menu,
  MenuItem,
  ListItemIcon,
  ListItemText,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Divider,
  Stack,
} from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import ClearIcon from '@mui/icons-material/Clear';
import CollectionsIcon from '@mui/icons-material/Collections';
import EditIcon from '@mui/icons-material/Edit';
import DeleteIcon from '@mui/icons-material/Delete';
import MoreVertIcon from '@mui/icons-material/MoreVert';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import FolderIcon from '@mui/icons-material/Folder';
import ShareIcon from '@mui/icons-material/Share';
import { designTokens } from '@/app/theme';
import {
  listVideos,
  listSessions,
  deleteVideo,
  renameVideo,
  type VideoListItem,
  type SessionType,
} from '@/services/api';
import {
  AppHeader,
  VideoCard,
  VideoCardSkeleton,
  EmptyState,
} from '@/components/dashboard';
import { AddVideoModal } from '@/components/AddVideoModal';
import { ManageSessionsDialog } from '@/components/ManageSessionsDialog';
import { VideoShareModal } from '@/components/VideoShareModal';

interface VideoSession {
  id: string;
  name: string;
  type: SessionType;
}

function VideosPageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [videos, setVideos] = useState<VideoListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(() => {
    const p = Number(searchParams.get('page'));
    return p > 0 ? p : 1;
  });
  const [totalPages, setTotalPages] = useState(1);
  const [totalVideos, setTotalVideos] = useState(0);
  const [search, setSearch] = useState(() => searchParams.get('search') ?? '');
  const [searchInput, setSearchInput] = useState(() => searchParams.get('search') ?? '');
  const [allVideosSessionId, setAllVideosSessionId] = useState<string | null>(null);
  const limit = 24;

  // Sync state to URL
  useEffect(() => {
    const params = new URLSearchParams();
    if (page > 1) params.set('page', String(page));
    if (search) params.set('search', search);
    const qs = params.toString();
    const url = qs ? `/videos?${qs}` : '/videos';
    router.replace(url, { scroll: false });
  }, [page, search, router]);

  // Scroll to top on page change
  useEffect(() => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }, [page]);

  // Video menu state
  const [menuAnchor, setMenuAnchor] = useState<{
    element: HTMLElement;
    video: VideoListItem;
  } | null>(null);


  // Rename dialog state
  const [renameDialogOpen, setRenameDialogOpen] = useState(false);
  const [renameVideo_target, setRenameVideoTarget] = useState<VideoListItem | null>(null);
  const [newName, setNewName] = useState('');
  const [renaming, setRenaming] = useState(false);

  // Delete dialog state
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [deleteVideo_target, setDeleteVideoTarget] = useState<VideoListItem | null>(null);
  const [deleting, setDeleting] = useState(false);

  // Upload modal state
  const [uploadModalOpen, setUploadModalOpen] = useState(false);

  // Manage sessions dialog state
  const [manageSessionsOpen, setManageSessionsOpen] = useState(false);
  const [manageSessionsVideo, setManageSessionsVideo] = useState<VideoListItem | null>(null);

  // Share modal state
  const [shareModalOpen, setShareModalOpen] = useState(false);
  const [shareVideo, setShareVideo] = useState<VideoListItem | null>(null);

  // Filter out PENDING videos
  const readyVideos = useMemo(
    () => videos.filter((v) => v.status !== 'PENDING'),
    [videos]
  );

  const loadVideos = useCallback(async () => {
    try {
      setLoading(true);
      const result = await listVideos(page, limit, search || undefined);
      setVideos(result.data);
      setTotalPages(result.pagination.totalPages);
      setTotalVideos(result.pagination.total);
    } catch (error) {
      console.error('Failed to load videos:', error);
    } finally {
      setLoading(false);
    }
  }, [page, search]);

  useEffect(() => {
    loadVideos();
  }, [loadVideos]);

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
  };

  const handleOpenInLibrary = () => {
    if (menuAnchor?.video && allVideosSessionId) {
      router.push(`/sessions/${allVideosSessionId}?video=${menuAnchor.video.id}`);
    }
    handleMenuClose();
  };

  const handleManageSessions = () => {
    if (menuAnchor?.video) {
      setManageSessionsVideo(menuAnchor.video);
      setManageSessionsOpen(true);
    }
    handleMenuClose();
  };

  const handleShareClick = () => {
    if (menuAnchor?.video) {
      setShareVideo(menuAnchor.video);
      setShareModalOpen(true);
    }
    handleMenuClose();
  };

  const handleVideoClick = (video: VideoListItem) => {
    // Open single video directly in the editor
    router.push(`/video/${video.id}`);
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

  const handleClearSearch = () => {
    setSearchInput('');
    setSearch('');
    setPage(1);
  };

  // Calculate showing range
  const startItem = (page - 1) * limit + 1;
  const endItem = Math.min(page * limit, totalVideos);

  return (
    <Box
      sx={{
        height: '100%',
        overflow: 'auto',
        bgcolor: designTokens.colors.surface[0],
        color: 'text.primary',
        position: 'relative',
        '&::before': {
          content: '""',
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          height: 400,
          background: 'radial-gradient(ellipse at 50% 0%, rgba(255, 107, 74, 0.08) 0%, transparent 70%)',
          pointerEvents: 'none',
        },
      }}
    >
      <AppHeader />
      <Container maxWidth="lg" sx={{ position: 'relative', py: 4 }}>
        {/* Header */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 4, py: 1 }}>
          <Box>
            <Typography variant="h4" sx={{ fontWeight: 700, letterSpacing: '-0.02em' }}>
              Video Library
            </Typography>
            {totalVideos > 0 && (
              <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
                {totalVideos} video{totalVideos !== 1 ? 's' : ''} in your library
              </Typography>
            )}
          </Box>
          <Button variant="contained" startIcon={<CloudUploadIcon />} onClick={() => setUploadModalOpen(true)}>
            Upload
          </Button>
        </Box>

        {/* Search Bar */}
        <Box sx={{ mb: 4, maxWidth: 400 }}>
          <TextField
            placeholder="Search videos..."
            value={searchInput}
            onChange={(e) => setSearchInput(e.target.value)}
            fullWidth
            size="small"
            sx={{
              '& .MuiOutlinedInput-root': {
                bgcolor: designTokens.colors.surface[1],
                borderRadius: 2,
                transition: 'all 0.2s ease',
                '&:hover': {
                  '& fieldset': {
                    borderColor: 'rgba(255, 107, 74, 0.5)',
                  },
                },
                '&.Mui-focused': {
                  bgcolor: designTokens.colors.surface[2],
                },
              },
            }}
            slotProps={{
              input: {
                startAdornment: (
                  <InputAdornment position="start">
                    <SearchIcon sx={{ color: 'text.secondary', fontSize: 20 }} />
                  </InputAdornment>
                ),
                endAdornment: searchInput && (
                  <InputAdornment position="end">
                    <IconButton
                      size="small"
                      onClick={handleClearSearch}
                      sx={{ color: 'text.secondary' }}
                    >
                      <ClearIcon fontSize="small" />
                    </IconButton>
                  </InputAdornment>
                ),
              },
            }}
          />
        </Box>

        {/* Content */}
        {loading ? (
          <Grid container spacing={2.5}>
            {Array.from({ length: 8 }).map((_, i) => (
              <Grid size={{ xs: 6, sm: 4, md: 3 }} key={i}>
                <VideoCardSkeleton />
              </Grid>
            ))}
          </Grid>
        ) : readyVideos.length === 0 ? (
          search ? (
            <EmptyState
              variant="search-no-results"
              onAction={handleClearSearch}
              title="No videos found"
              description={`No videos match "${search}". Try a different search term.`}
              actionLabel="Clear Search"
            />
          ) : (
            <EmptyState
              variant="no-videos"
              onAction={() => router.push('/sessions')}
              title="No videos yet"
              description="Upload videos from the Sessions page to get started"
              actionLabel="Go to Sessions"
            />
          )
        ) : (
          <>
            {/* Results info */}
            {totalVideos > limit && (
              <Typography
                variant="body2"
                color="text.secondary"
                sx={{ mb: 2 }}
              >
                Showing {startItem}-{endItem} of {totalVideos} videos
              </Typography>
            )}

            {/* Video Grid */}
            <Grid container spacing={2.5}>
              {readyVideos.map((video) => {
                const regularSessions = getRegularSessions(video);
                const firstSession = regularSessions[0];

                return (
                  <Grid size={{ xs: 6, sm: 4, md: 3 }} key={video.id}>
                    <Box sx={{ position: 'relative' }}>
                      <VideoCard
                        id={video.id}
                        name={video.name}
                        durationMs={video.durationMs}
                        posterS3Key={video.posterS3Key}
                        status={video.status}
                        sessionTag={firstSession?.name}
                        onClick={() => handleVideoClick(video)}
                      />
                      <IconButton
                        size="small"
                        onClick={(e) => handleMenuOpen(e, video)}
                        sx={{
                          position: 'absolute',
                          top: 8,
                          right: 8,
                          bgcolor: 'rgba(0, 0, 0, 0.5)',
                          backdropFilter: 'blur(4px)',
                          color: 'white',
                          opacity: 0,
                          transition: 'opacity 0.2s ease',
                          '.MuiGrid-root:hover &': {
                            opacity: 1,
                          },
                          '&:hover': {
                            bgcolor: 'rgba(0, 0, 0, 0.7)',
                          },
                        }}
                      >
                        <MoreVertIcon fontSize="small" />
                      </IconButton>
                    </Box>
                  </Grid>
                );
              })}
            </Grid>

            {/* Pagination */}
            {totalPages > 1 && (
              <Stack
                direction={{ xs: 'column', sm: 'row' }}
                alignItems="center"
                justifyContent="center"
                spacing={2}
                sx={{ mt: 5 }}
              >
                <Typography variant="body2" color="text.secondary">
                  Page {page} of {totalPages}
                </Typography>
                <Pagination
                  count={totalPages}
                  page={page}
                  onChange={(_, newPage) => setPage(newPage)}
                  color="primary"
                  shape="rounded"
                  sx={{
                    '& .MuiPaginationItem-root': {
                      color: 'text.primary',
                      borderColor: 'divider',
                      '&.Mui-selected': {
                        bgcolor: 'primary.main',
                        color: 'white',
                        '&:hover': {
                          bgcolor: 'primary.dark',
                        },
                      },
                    },
                  }}
                />
              </Stack>
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
        slotProps={{
          paper: {
            sx: {
              minWidth: 200,
              bgcolor: designTokens.colors.surface[3],
              border: '1px solid',
              borderColor: 'divider',
            },
          },
        }}
      >
        {/* Open in Library option */}
        {allVideosSessionId && (
          <MenuItem
            onClick={handleOpenInLibrary}
            sx={{
              '&:hover': {
                bgcolor: 'rgba(255, 107, 74, 0.08)',
              },
            }}
          >
            <ListItemIcon>
              <CollectionsIcon fontSize="small" sx={{ color: 'primary.main' }} />
            </ListItemIcon>
            <ListItemText
              primary="Open in Library"
              secondary="View without session context"
              slotProps={{
                secondary: {
                  sx: { fontSize: '0.7rem', color: 'text.disabled' },
                },
              }}
            />
          </MenuItem>
        )}
        {allVideosSessionId && <Divider />}
        <MenuItem onClick={handleShareClick}>
          <ListItemIcon>
            <ShareIcon fontSize="small" sx={{ color: 'primary.main' }} />
          </ListItemIcon>
          <ListItemText primary="Share" />
        </MenuItem>
        <MenuItem onClick={handleManageSessions}>
          <ListItemIcon>
            <FolderIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText primary="Manage Sessions" />
        </MenuItem>
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
        <DialogActions sx={{ px: 3, pb: 2 }}>
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

      {/* Upload modal */}
      <AddVideoModal
        open={uploadModalOpen}
        onClose={() => setUploadModalOpen(false)}
        onVideoAdded={loadVideos}
      />

      {/* Manage sessions dialog */}
      <ManageSessionsDialog
        open={manageSessionsOpen}
        onClose={() => {
          setManageSessionsOpen(false);
          setManageSessionsVideo(null);
        }}
        video={manageSessionsVideo}
        onChanged={loadVideos}
      />

      {/* Share modal */}
      {shareVideo && (
        <VideoShareModal
          open={shareModalOpen}
          onClose={() => {
            setShareModalOpen(false);
            setShareVideo(null);
          }}
          videoId={shareVideo.id}
          videoName={shareVideo.name}
          isOwner={true}
        />
      )}
    </Box>
  );
}

export default function VideosPage() {
  return (
    <Suspense>
      <VideosPageContent />
    </Suspense>
  );
}
