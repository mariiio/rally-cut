'use client';

import { useEffect, useState, useRef, useMemo } from 'react';
import { useRouter } from 'next/navigation';
import {
  Box,
  Container,
  Typography,
  Card,
  CardContent,
  CardActionArea,
  Grid,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  CircularProgress,
  Chip,
  Stack,
  IconButton,
  Alert,
  LinearProgress,
  Radio,
  RadioGroup,
  FormControlLabel,
  Divider,
  Menu,
  MenuItem,
  ListItemIcon,
  ListItemText,
  TextField,
} from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import DeleteIcon from '@mui/icons-material/Delete';
import MoreVertIcon from '@mui/icons-material/MoreVert';
import PeopleIcon from '@mui/icons-material/People';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import CloseIcon from '@mui/icons-material/Close';
import CollectionsIcon from '@mui/icons-material/Collections';
import SportsVolleyballIcon from '@mui/icons-material/SportsVolleyball';
import ChevronRightIcon from '@mui/icons-material/ChevronRight';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import { designTokens } from '@/app/theme';
import {
  listSessions,
  createSession,
  deleteSession,
  listSharedSessions,
  addVideoToSession,
  listVideos,
  getVideoStreamUrl,
  type ListSessionsResponse,
  type SharedSession,
  type VideoListItem,
} from '@/services/api';
import { useUploadStore } from '@/stores/uploadStore';
import { isValidVideoFile } from '@/utils/fileHandlers';
import { VolleyballProgress } from '@/components/VolleyballProgress';

interface SessionGroup {
  session: {
    id: string;
    name: string;
  };
  videos: VideoListItem[];
}

function formatDuration(ms: number | null): string {
  if (!ms) return '--:--';
  const totalSeconds = Math.floor(ms / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, '0')}`;
}

const NEW_SESSION_VALUE = '__new__';

export default function HomePage() {
  const router = useRouter();
  const [sessions, setSessions] = useState<ListSessionsResponse['data']>([]);
  const [videos, setVideos] = useState<VideoListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [menuAnchor, setMenuAnchor] = useState<{ element: HTMLElement; sessionId: string; sessionName: string } | null>(null);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [sessionToDelete, setSessionToDelete] = useState<{ id: string; name: string } | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [sharedSessions, setSharedSessions] = useState<SharedSession[]>([]);
  const [loadingShared, setLoadingShared] = useState(true);
  const [uploadDialogOpen, setUploadDialogOpen] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [creatingSession, setCreatingSession] = useState(false);
  const [selectedSessionId, setSelectedSessionId] = useState<string>(NEW_SESSION_VALUE);
  const [newSessionName, setNewSessionName] = useState<string>('');
  const [uploadQueue, setUploadQueue] = useState<File[]>([]);
  const [currentUploadIndex, setCurrentUploadIndex] = useState(0);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const { isUploading, progress, currentStep, uploadVideoToLibrary } = useUploadStore();

  // Get ALL_VIDEOS session for navigation
  const allVideosSession = useMemo(
    () => sessions.find((s) => s.type === 'ALL_VIDEOS'),
    [sessions]
  );

  // Get regular sessions for selection
  const regularSessions = useMemo(
    () => sessions.filter((s) => s.type !== 'ALL_VIDEOS'),
    [sessions]
  );

  // Filter out incomplete uploads (PENDING status)
  const readyVideos = useMemo(
    () => videos.filter((v) => v.status !== 'PENDING'),
    [videos]
  );

  // Group videos by their sessions (excluding ALL_VIDEOS)
  const groupedVideos = useMemo(() => {
    const groups: Map<string, SessionGroup> = new Map();

    readyVideos.forEach((video) => {
      video.sessions?.forEach((session) => {
        if (session.type === 'ALL_VIDEOS') return;

        if (!groups.has(session.id)) {
          groups.set(session.id, {
            session: { id: session.id, name: session.name },
            videos: [],
          });
        }
        groups.get(session.id)!.videos.push(video);
      });
    });

    return Array.from(groups.values());
  }, [readyVideos]);

  // Check if user has any ready content
  const hasVideos = readyVideos.length > 0;

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      setLoading(true);
      setError(null);
      const [sessionsResponse, videosResponse, sharedResponse] = await Promise.all([
        listSessions(),
        listVideos(1, 100),
        listSharedSessions().catch(() => ({ data: [] })),
      ]);
      setSessions(sessionsResponse.data);
      setVideos(videosResponse.data);
      setSharedSessions(sharedResponse.data);
      setLoadingShared(false);
    } catch (err) {
      console.error('Failed to load data:', err);
      setError('Failed to load data. Make sure the API is running.');
    } finally {
      setLoading(false);
    }
  };

  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>, sessionId: string, sessionName: string) => {
    event.stopPropagation();
    setMenuAnchor({ element: event.currentTarget, sessionId, sessionName });
  };

  const handleMenuClose = () => {
    setMenuAnchor(null);
  };

  const handleDeleteClick = () => {
    if (menuAnchor) {
      setSessionToDelete({ id: menuAnchor.sessionId, name: menuAnchor.sessionName });
      setDeleteDialogOpen(true);
    }
    handleMenuClose();
  };

  const handleDeleteConfirm = async () => {
    if (!sessionToDelete) return;
    try {
      setDeleting(true);
      await deleteSession(sessionToDelete.id);
      setSessions(sessions.filter((s) => s.id !== sessionToDelete.id));
      setDeleteDialogOpen(false);
      setSessionToDelete(null);
      loadData();
    } catch (err) {
      console.error('Failed to delete session:', err);
      setError('Failed to delete session');
    } finally {
      setDeleting(false);
    }
  };

  const handleDeleteCancel = () => {
    setDeleteDialogOpen(false);
    setSessionToDelete(null);
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    });
  };

  const generateSessionName = () => {
    return `Session - ${new Date().toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
    })}`;
  };

  const handleUploadDialogOpen = () => {
    setUploadDialogOpen(true);
    setUploadError(null);
    // Default to "new session" when opening
    setSelectedSessionId(NEW_SESSION_VALUE);
    setNewSessionName(generateSessionName());
    setUploadQueue([]);
    setCurrentUploadIndex(0);
  };

  const handleUploadDialogClose = () => {
    if (!isUploading && !creatingSession) {
      setUploadDialogOpen(false);
      setUploadError(null);
    }
  };

  const processMultipleUploads = async (files: File[]) => {
    // Validate all files first
    const validFiles = files.filter(isValidVideoFile);
    if (validFiles.length === 0) {
      setUploadError('No valid video files. Please use MP4, MOV, or WebM.');
      return;
    }
    if (validFiles.length !== files.length) {
      // Don't show error, just skip invalid files
      console.log(`Skipped ${files.length - validFiles.length} invalid file(s)`);
    }

    // Check for duplicate session name before starting uploads
    const isNewSession = selectedSessionId === NEW_SESSION_VALUE;
    const sessionName = isNewSession ? (newSessionName.trim() || generateSessionName()) : '';
    if (isNewSession) {
      const existingSession = sessions.find(
        (s) => s.name.toLowerCase() === sessionName.toLowerCase() && s.type === 'REGULAR'
      );
      if (existingSession) {
        setUploadError(`A session named "${sessionName}" already exists. Please choose a different name.`);
        return;
      }
    }

    setUploadError(null);
    setUploadQueue(validFiles);
    setCurrentUploadIndex(0);

    // Upload videos first (before creating session)
    const uploadedVideoIds: string[] = [];
    let lastUploadError: string | null = null;
    for (let i = 0; i < validFiles.length; i++) {
      setCurrentUploadIndex(i);
      const file = validFiles[i];

      const result = await uploadVideoToLibrary(file);
      if (!result.success || !result.videoId) {
        // Get the error from the store after the failed upload
        const storeError = useUploadStore.getState().error;
        if (storeError) {
          lastUploadError = storeError;
        }
        console.error(`Failed to upload ${file.name}`);
        continue; // Continue with next file
      }
      uploadedVideoIds.push(result.videoId);
    }

    // If no uploads succeeded, show error and don't create session
    if (uploadedVideoIds.length === 0) {
      setUploadQueue([]);
      setUploadError(lastUploadError || 'All uploads failed. Please try again.');
      return;
    }

    // Now create session (only if uploads succeeded)
    let targetSessionId: string;
    if (isNewSession) {
      setCreatingSession(true);
      try {
        const session = await createSession(sessionName);
        targetSessionId = session.id;
      } catch (err) {
        console.error('Failed to create session:', err);
        setUploadError('Videos uploaded but failed to create session. Your videos are in your library.');
        setCreatingSession(false);
        setUploadQueue([]);
        loadData();
        return;
      }
      setCreatingSession(false);
    } else {
      targetSessionId = selectedSessionId;
    }

    // Add uploaded videos to session
    let addedCount = 0;
    let lastAddError: string | null = null;
    for (const videoId of uploadedVideoIds) {
      try {
        await addVideoToSession(targetSessionId, videoId);
        addedCount++;
      } catch (err) {
        console.error(`Failed to add video ${videoId} to session:`, err);
        lastAddError = err instanceof Error ? err.message : 'Failed to add video to session';
      }
    }

    // Check if any videos were successfully added
    if (addedCount === 0) {
      setUploadQueue([]);
      setUploadError(lastAddError || 'Failed to add videos to session. Please try again.');
      // Reload data to refresh the UI since videos are in library
      loadData();
      return;
    }

    // All done - redirect
    setUploadQueue([]);
    setUploadDialogOpen(false);
    router.push(`/sessions/${targetSessionId}`);
  };

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    if (files.length === 0) return;
    e.target.value = '';
    await processMultipleUploads(files);
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    const files = Array.from(e.dataTransfer.files);
    if (files.length === 0) return;
    await processMultipleUploads(files);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  return (
    <Box
      sx={{
        minHeight: '100vh',
        bgcolor: designTokens.colors.surface[0],
        color: 'text.primary',
        py: 4,
        position: 'relative',
        '&::before': {
          content: '""',
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          height: 300,
          background: 'radial-gradient(ellipse at 50% 0%, rgba(255, 107, 74, 0.06) 0%, transparent 70%)',
          pointerEvents: 'none',
        },
      }}
    >
      <Container maxWidth="lg" sx={{ position: 'relative' }}>
        {/* Header */}
        <Box
          component="header"
          sx={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            mb: 5,
            py: 1,
          }}
        >
          <Stack direction="row" alignItems="center" spacing={2}>
            <SportsVolleyballIcon
              sx={{
                fontSize: 40,
                color: 'primary.main',
                filter: 'drop-shadow(0 2px 8px rgba(255, 107, 74, 0.4))',
              }}
            />
            <Box>
              <Typography
                variant="h4"
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
              <Typography variant="body2" color="text.secondary" sx={{ letterSpacing: '0.01em' }}>
                Beach Volleyball Video Analysis
              </Typography>
            </Box>
          </Stack>
          {hasVideos && (
            <Button
              variant="contained"
              startIcon={<CloudUploadIcon />}
              onClick={handleUploadDialogOpen}
              sx={{
                px: 3,
                py: 1,
                transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
                '&:hover': {
                  transform: 'translateY(-2px)',
                  boxShadow: '0 6px 20px rgba(255, 107, 74, 0.4)',
                },
              }}
            >
              Upload Video
            </Button>
          )}
        </Box>

        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', py: 8 }}>
            <CircularProgress />
          </Box>
        ) : error ? (
          <Box sx={{ textAlign: 'center', py: 8 }}>
            <Typography color="error" gutterBottom>
              {error}
            </Typography>
            <Button variant="outlined" onClick={loadData} sx={{ mt: 2 }}>
              Retry
            </Button>
          </Box>
        ) : !hasVideos ? (
          /* Empty State - No videos */
          <>
            {!loadingShared && sharedSessions.length > 0 ? (
              /* Has shared sessions but no own videos */
              <>
                <Box sx={{ textAlign: 'center', py: 6, mb: 5 }}>
                  <Box
                    sx={{
                      width: 100,
                      height: 100,
                      mx: 'auto',
                      mb: 3,
                      borderRadius: '50%',
                      background: `linear-gradient(135deg, ${designTokens.colors.surface[2]} 0%, ${designTokens.colors.surface[1]} 100%)`,
                      border: '2px dashed',
                      borderColor: 'rgba(255, 107, 74, 0.3)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                    }}
                  >
                    <SportsVolleyballIcon
                      sx={{
                        fontSize: 48,
                        color: 'primary.main',
                        filter: 'drop-shadow(0 4px 12px rgba(255, 107, 74, 0.3))',
                      }}
                    />
                  </Box>
                  <Typography
                    variant="h6"
                    sx={{
                      mb: 1,
                      fontWeight: 600,
                      background: designTokens.gradients.primary,
                      backgroundClip: 'text',
                      WebkitBackgroundClip: 'text',
                      WebkitTextFillColor: 'transparent',
                    }}
                  >
                    Start your own session
                  </Typography>
                  <Typography color="text.secondary" sx={{ mb: 3, maxWidth: 360, mx: 'auto' }}>
                    Upload a volleyball video to create your first session
                  </Typography>
                  <Button
                    variant="contained"
                    size="large"
                    startIcon={<CloudUploadIcon />}
                    onClick={handleUploadDialogOpen}
                    sx={{
                      px: 4,
                      py: 1.5,
                      transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
                      '&:hover': {
                        transform: 'translateY(-2px)',
                        boxShadow: '0 8px 24px rgba(255, 107, 74, 0.4)',
                      },
                    }}
                  >
                    Upload Video
                  </Button>
                </Box>
                <Stack direction="row" alignItems="center" spacing={1.5} sx={{ mb: 2 }}>
                  <Box
                    sx={{
                      width: 28,
                      height: 28,
                      borderRadius: 1,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      bgcolor: 'rgba(0, 212, 170, 0.15)',
                    }}
                  >
                    <PeopleIcon sx={{ fontSize: 16, color: 'secondary.main' }} />
                  </Box>
                  <Typography variant="subtitle2" color="text.secondary" fontWeight={500}>
                    Shared with me
                  </Typography>
                  <Chip
                    label={sharedSessions.length}
                    size="small"
                    sx={{
                      bgcolor: 'rgba(0, 212, 170, 0.2)',
                      color: 'secondary.main',
                      fontWeight: 600,
                      height: 20,
                      minWidth: 24,
                    }}
                  />
                </Stack>
                <Grid container spacing={3}>
                  {sharedSessions.map((session) => (
                    <Grid size={{ xs: 12, sm: 6, md: 4 }} key={session.id}>
                      <Card
                        sx={{
                          bgcolor: designTokens.colors.surface[1],
                          border: '1px solid',
                          borderColor: 'divider',
                          borderRadius: 2,
                          overflow: 'hidden',
                          transition: 'all 0.25s cubic-bezier(0.4, 0, 0.2, 1)',
                          '&:hover': {
                            transform: 'translateY(-4px)',
                            boxShadow: designTokens.shadows.lg,
                            borderColor: 'secondary.main',
                          },
                        }}
                      >
                        <Box sx={{ height: 3, background: designTokens.gradients.secondary }} />
                        <CardActionArea onClick={() => router.push(`/sessions/${session.id}`)}>
                          <CardContent sx={{ p: 2.5 }}>
                            <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 1.5 }}>
                              <Box
                                sx={{
                                  width: 22,
                                  height: 22,
                                  borderRadius: '50%',
                                  bgcolor: 'rgba(0, 212, 170, 0.2)',
                                  display: 'flex',
                                  alignItems: 'center',
                                  justifyContent: 'center',
                                }}
                              >
                                <PeopleIcon sx={{ fontSize: 12, color: 'secondary.main' }} />
                              </Box>
                              <Typography variant="caption" color="secondary.main" fontWeight={500}>
                                Shared by {session.ownerName || 'Unknown'}
                              </Typography>
                            </Stack>
                            <Typography variant="h6" fontWeight={600} noWrap sx={{ mb: 1.5 }}>
                              {session.name}
                            </Typography>
                            <Stack direction="row" spacing={1} sx={{ mb: 2 }}>
                              <Chip
                                label={`${session.videoCount} videos`}
                                size="small"
                                sx={{ bgcolor: designTokens.colors.surface[3], fontWeight: 500 }}
                              />
                              <Chip
                                label={`${session.highlightCount} highlights`}
                                size="small"
                                sx={{ bgcolor: designTokens.colors.surface[3], fontWeight: 500 }}
                              />
                            </Stack>
                            <Typography variant="caption" color="text.disabled">
                              Joined {formatDate(session.joinedAt)}
                            </Typography>
                          </CardContent>
                        </CardActionArea>
                      </Card>
                    </Grid>
                  ))}
                </Grid>
              </>
            ) : (
              /* No videos, no shared sessions - Main empty state */
              <Box sx={{ textAlign: 'center', py: 12 }}>
                <Box
                  sx={{
                    width: 120,
                    height: 120,
                    mx: 'auto',
                    mb: 4,
                    borderRadius: '50%',
                    background: `linear-gradient(135deg, ${designTokens.colors.surface[2]} 0%, ${designTokens.colors.surface[1]} 100%)`,
                    border: '2px dashed',
                    borderColor: 'rgba(255, 107, 74, 0.3)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    position: 'relative',
                    animation: 'pulse 3s ease-in-out infinite',
                    '@keyframes pulse': {
                      '0%, 100%': {
                        borderColor: 'rgba(255, 107, 74, 0.3)',
                        transform: 'scale(1)',
                      },
                      '50%': {
                        borderColor: 'rgba(255, 107, 74, 0.6)',
                        transform: 'scale(1.02)',
                      },
                    },
                  }}
                >
                  <SportsVolleyballIcon
                    sx={{
                      fontSize: 56,
                      color: 'primary.main',
                      filter: 'drop-shadow(0 4px 12px rgba(255, 107, 74, 0.3))',
                    }}
                  />
                </Box>
                <Typography
                  variant="h5"
                  sx={{
                    mb: 1,
                    fontWeight: 600,
                    background: designTokens.gradients.primary,
                    backgroundClip: 'text',
                    WebkitBackgroundClip: 'text',
                    WebkitTextFillColor: 'transparent',
                  }}
                >
                  Get started with RallyCut
                </Typography>
                <Typography
                  variant="body1"
                  color="text.secondary"
                  sx={{ mb: 4, maxWidth: 400, mx: 'auto' }}
                >
                  Upload your beach volleyball video to begin analyzing rallies and creating highlights
                </Typography>
                <Button
                  variant="contained"
                  size="large"
                  startIcon={<CloudUploadIcon />}
                  onClick={handleUploadDialogOpen}
                  sx={{
                    px: 4,
                    py: 1.5,
                    fontSize: '1rem',
                    transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                    '&:hover': {
                      transform: 'translateY(-3px)',
                      boxShadow: '0 8px 24px rgba(255, 107, 74, 0.4)',
                    },
                  }}
                >
                  Upload Your First Video
                </Button>
              </Box>
            )}
          </>
        ) : (
          /* Has videos - show All Videos card and sessions */
          <>
            {/* Pinned All Videos Card */}
            {allVideosSession && (
              <Card
                sx={{
                  mb: 4,
                  position: 'relative',
                  overflow: 'hidden',
                  bgcolor: designTokens.colors.surface[2],
                  border: '1px solid',
                  borderColor: 'primary.main',
                  borderRadius: 3,
                  transition: 'all 0.25s cubic-bezier(0.4, 0, 0.2, 1)',
                  '&::before': {
                    content: '""',
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    right: 0,
                    height: 3,
                    background: designTokens.gradients.sunset,
                  },
                  '&:hover': {
                    transform: 'translateY(-4px)',
                    boxShadow: `${designTokens.shadows.xl}, ${designTokens.shadows.glow.primary}`,
                    borderColor: 'primary.light',
                  },
                }}
              >
                <CardActionArea onClick={() => router.push('/videos')}>
                  <CardContent sx={{ py: 3, px: 3 }}>
                    <Stack direction="row" alignItems="center" spacing={2}>
                      <Box
                        sx={{
                          width: 48,
                          height: 48,
                          borderRadius: 2,
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          background: 'rgba(255, 107, 74, 0.15)',
                          border: '1px solid rgba(255, 107, 74, 0.3)',
                        }}
                      >
                        <CollectionsIcon sx={{ color: 'primary.main', fontSize: 26 }} />
                      </Box>
                      <Box sx={{ flex: 1 }}>
                        <Typography variant="h5" fontWeight={600}>
                          All Videos
                        </Typography>
                        <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
                          Browse all your videos and create highlights
                        </Typography>
                      </Box>
                      <Chip
                        label={`${readyVideos.length} video${readyVideos.length !== 1 ? 's' : ''}`}
                        sx={{
                          bgcolor: 'rgba(255, 107, 74, 0.2)',
                          color: 'primary.light',
                          fontWeight: 600,
                          fontSize: '0.8rem',
                        }}
                      />
                      <ChevronRightIcon sx={{ color: 'text.secondary', fontSize: 28 }} />
                    </Stack>
                  </CardContent>
                </CardActionArea>
              </Card>
            )}

            {/* Sessions with Video Previews */}
            {groupedVideos.length > 0 && (
              <Box sx={{ mb: 4 }}>
                <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 2, fontWeight: 500 }}>
                  Your Sessions
                </Typography>
                {groupedVideos.map(({ session, videos: sessionVideos }) => (
                  <Card
                    key={session.id}
                    sx={{
                      mb: 3,
                      bgcolor: designTokens.colors.surface[1],
                      borderRadius: 3,
                      overflow: 'hidden',
                      border: '1px solid',
                      borderColor: 'divider',
                      transition: 'all 0.25s cubic-bezier(0.4, 0, 0.2, 1)',
                      '&:hover': {
                        borderColor: 'rgba(255, 107, 74, 0.3)',
                        boxShadow: designTokens.shadows.lg,
                      },
                    }}
                  >
                    {/* Session Header */}
                    <Box
                      sx={{
                        p: 2,
                        display: 'flex',
                        alignItems: 'center',
                        background: designTokens.gradients.toolbar,
                        borderBottom: '1px solid',
                        borderColor: 'divider',
                      }}
                    >
                      <Typography variant="subtitle1" fontWeight={600} sx={{ flex: 1 }}>
                        {session.name}
                      </Typography>
                      <Chip
                        label={`${sessionVideos.length} video${sessionVideos.length !== 1 ? 's' : ''}`}
                        size="small"
                        sx={{
                          bgcolor: 'rgba(0, 212, 170, 0.15)',
                          color: 'secondary.main',
                          fontWeight: 500,
                        }}
                      />
                      <Box sx={{ ml: 2, display: 'flex', gap: 1, alignItems: 'center' }}>
                        <Button
                          size="small"
                          onClick={() => router.push(`/sessions/${session.id}`)}
                          sx={{
                            color: 'primary.main',
                            fontWeight: 600,
                            '&:hover': {
                              backgroundColor: 'rgba(255, 107, 74, 0.1)',
                            },
                          }}
                        >
                          Open
                        </Button>
                        <IconButton
                          size="small"
                          onClick={(e) => handleMenuOpen(e, session.id, session.name)}
                          sx={{
                            color: 'text.secondary',
                            '&:hover': { color: 'text.primary', bgcolor: 'action.hover' },
                          }}
                        >
                          <MoreVertIcon fontSize="small" />
                        </IconButton>
                      </Box>
                    </Box>
                    {/* Video Grid */}
                    <Box sx={{ p: 2, bgcolor: designTokens.colors.surface[0] }}>
                      <Grid container spacing={2}>
                        {sessionVideos.map((video) => (
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
                              onClick={() => router.push(`/sessions/${session.id}`)}
                            >
                              <Box sx={{ position: 'relative', aspectRatio: '16/9' }}>
                                <video
                                  src={getVideoStreamUrl(video.s3Key)}
                                  poster={video.posterS3Key ? getVideoStreamUrl(video.posterS3Key) : undefined}
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
                                    background: 'linear-gradient(to top, rgba(0,0,0,0.6) 0%, rgba(0,0,0,0.1) 40%, transparent 100%)',
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
                              <CardContent sx={{ py: 1.5, px: 1.5, '&:last-child': { pb: 1.5 } }}>
                                <Typography
                                  variant="body2"
                                  noWrap
                                  title={video.name}
                                  sx={{ fontWeight: 500, color: 'text.primary' }}
                                >
                                  {video.name}
                                </Typography>
                              </CardContent>
                            </Card>
                          </Grid>
                        ))}
                      </Grid>
                    </Box>
                  </Card>
                ))}
              </Box>
            )}

            {/* Shared with me */}
            {!loadingShared && sharedSessions.length > 0 && (
              <Box sx={{ mt: 5 }}>
                <Stack direction="row" alignItems="center" spacing={1.5} sx={{ mb: 3 }}>
                  <Box
                    sx={{
                      width: 32,
                      height: 32,
                      borderRadius: 1,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      bgcolor: 'rgba(0, 212, 170, 0.15)',
                    }}
                  >
                    <PeopleIcon sx={{ fontSize: 18, color: 'secondary.main' }} />
                  </Box>
                  <Typography variant="subtitle1" fontWeight={600} color="text.primary">
                    Shared with me
                  </Typography>
                  <Chip
                    label={sharedSessions.length}
                    size="small"
                    sx={{
                      bgcolor: 'rgba(0, 212, 170, 0.2)',
                      color: 'secondary.main',
                      fontWeight: 600,
                      height: 22,
                      minWidth: 26,
                    }}
                  />
                </Stack>
                <Grid container spacing={3}>
                  {sharedSessions.map((session) => (
                    <Grid size={{ xs: 12, sm: 6, md: 4 }} key={session.id}>
                      <Card
                        sx={{
                          bgcolor: designTokens.colors.surface[1],
                          border: '1px solid',
                          borderColor: 'divider',
                          borderRadius: 2,
                          overflow: 'hidden',
                          transition: 'all 0.25s cubic-bezier(0.4, 0, 0.2, 1)',
                          '&:hover': {
                            transform: 'translateY(-4px)',
                            boxShadow: designTokens.shadows.lg,
                            borderColor: 'secondary.main',
                          },
                        }}
                      >
                        <Box sx={{ height: 3, background: designTokens.gradients.secondary }} />
                        <CardActionArea onClick={() => router.push(`/sessions/${session.id}`)}>
                          <CardContent sx={{ p: 2.5 }}>
                            <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 1.5 }}>
                              <Box
                                sx={{
                                  width: 24,
                                  height: 24,
                                  borderRadius: '50%',
                                  bgcolor: 'rgba(0, 212, 170, 0.2)',
                                  display: 'flex',
                                  alignItems: 'center',
                                  justifyContent: 'center',
                                }}
                              >
                                <PeopleIcon sx={{ fontSize: 14, color: 'secondary.main' }} />
                              </Box>
                              <Typography variant="caption" color="secondary.main" fontWeight={500}>
                                Shared by {session.ownerName || 'Unknown'}
                              </Typography>
                            </Stack>
                            <Typography variant="h6" fontWeight={600} noWrap sx={{ mb: 1.5 }}>
                              {session.name}
                            </Typography>
                            <Stack direction="row" spacing={1} sx={{ mb: 2 }}>
                              <Chip
                                label={`${session.videoCount} videos`}
                                size="small"
                                sx={{ bgcolor: designTokens.colors.surface[3], fontWeight: 500 }}
                              />
                              <Chip
                                label={`${session.highlightCount} highlights`}
                                size="small"
                                sx={{ bgcolor: designTokens.colors.surface[3], fontWeight: 500 }}
                              />
                            </Stack>
                            <Typography variant="caption" color="text.disabled">
                              Joined {formatDate(session.joinedAt)}
                            </Typography>
                          </CardContent>
                        </CardActionArea>
                      </Card>
                    </Grid>
                  ))}
                </Grid>
              </Box>
            )}
          </>
        )}

        {/* Upload Dialog */}
        <Dialog open={uploadDialogOpen} onClose={handleUploadDialogClose} maxWidth="sm" fullWidth>
          <DialogTitle
            sx={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              borderBottom: '1px solid',
              borderColor: 'divider',
              pb: 2,
            }}
          >
            <Stack direction="row" alignItems="center" spacing={1.5}>
              <CloudUploadIcon sx={{ color: 'primary.main' }} />
              <Typography variant="h6" fontWeight={600}>
                Upload Videos
              </Typography>
            </Stack>
            <IconButton
              onClick={handleUploadDialogClose}
              size="small"
              disabled={isUploading || creatingSession}
              sx={{ color: 'text.secondary' }}
            >
              <CloseIcon />
            </IconButton>
          </DialogTitle>
          <DialogContent sx={{ pt: 3 }}>
            {uploadError && (
              <Alert severity="error" onClose={() => setUploadError(null)} sx={{ mb: 2 }}>
                {uploadError}
              </Alert>
            )}

            {/* Session Selection - only show when user has existing sessions */}
            {regularSessions.length > 0 && !isUploading && !creatingSession && (
              <>
                <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 1.5, fontWeight: 500 }}>
                  Add to session
                </Typography>
                <RadioGroup
                  value={selectedSessionId}
                  onChange={(e) => setSelectedSessionId(e.target.value)}
                >
                  <FormControlLabel
                    value={NEW_SESSION_VALUE}
                    control={<Radio size="small" sx={{ color: 'primary.main' }} />}
                    label={
                      <Stack direction="row" alignItems="center" spacing={1}>
                        <AddIcon sx={{ fontSize: 18, color: 'primary.main' }} />
                        <Typography variant="body2" fontWeight={500}>
                          New session
                        </Typography>
                      </Stack>
                    }
                    sx={{ mb: 0.5 }}
                  />
                  {selectedSessionId === NEW_SESSION_VALUE && (
                    <TextField
                      value={newSessionName}
                      onChange={(e) => setNewSessionName(e.target.value)}
                      placeholder="Session name"
                      size="small"
                      fullWidth
                      sx={{ ml: 4, mb: 1, maxWidth: 280 }}
                    />
                  )}
                  {regularSessions.map((session) => (
                    <FormControlLabel
                      key={session.id}
                      value={session.id}
                      control={<Radio size="small" />}
                      label={
                        <Stack direction="row" alignItems="center" spacing={1}>
                          <Typography variant="body2">{session.name}</Typography>
                          <Chip
                            label={`${session.videoCount} videos`}
                            size="small"
                            sx={{
                              height: 18,
                              fontSize: '0.65rem',
                              bgcolor: designTokens.colors.surface[3],
                              fontWeight: 500,
                            }}
                          />
                        </Stack>
                      }
                      sx={{ mb: 0.5 }}
                    />
                  ))}
                </RadioGroup>
                <Divider sx={{ my: 2.5 }} />
              </>
            )}

            {/* Session Name - show when no existing sessions */}
            {regularSessions.length === 0 && !isUploading && !creatingSession && (
              <>
                <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 1.5, fontWeight: 500 }}>
                  Session name
                </Typography>
                <TextField
                  value={newSessionName}
                  onChange={(e) => setNewSessionName(e.target.value)}
                  placeholder="Session name"
                  size="small"
                  fullWidth
                  sx={{ mb: 2.5 }}
                />
              </>
            )}

            {/* File Drop Zone */}
            <Box
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onClick={() => !isUploading && !creatingSession && fileInputRef.current?.click()}
              sx={{
                border: '2px dashed',
                borderColor: 'rgba(255, 107, 74, 0.3)',
                borderRadius: 3,
                p: 5,
                textAlign: 'center',
                cursor: isUploading || creatingSession ? 'default' : 'pointer',
                transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
                background: designTokens.colors.surface[0],
                '&:hover': {
                  borderColor: isUploading || creatingSession ? 'rgba(255, 107, 74, 0.3)' : 'primary.main',
                  bgcolor: isUploading || creatingSession ? designTokens.colors.surface[0] : 'rgba(255, 107, 74, 0.05)',
                },
              }}
            >
              {isUploading || creatingSession ? (
                <VolleyballProgress
                  progress={creatingSession ? 0 : progress}
                  stepText={
                    creatingSession
                      ? 'Creating session...'
                      : uploadQueue.length > 1
                        ? `Uploading video ${currentUploadIndex + 1} of ${uploadQueue.length}...`
                        : currentStep
                  }
                  size="lg"
                  showPercentage={!creatingSession}
                />
              ) : (
                <>
                  <Box
                    sx={{
                      width: 64,
                      height: 64,
                      mx: 'auto',
                      mb: 2,
                      borderRadius: 2,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      background: 'rgba(255, 107, 74, 0.1)',
                    }}
                  >
                    <CloudUploadIcon sx={{ fontSize: 32, color: 'primary.main' }} />
                  </Box>
                  <Typography variant="h6" fontWeight={600} gutterBottom>
                    Drop videos here or click to browse
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    MP4, MOV, WebM (multiple files allowed)
                  </Typography>
                </>
              )}
            </Box>
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileSelect}
              accept="video/mp4,video/quicktime,video/webm,.mp4,.mov,.webm"
              multiple
              style={{ display: 'none' }}
            />
          </DialogContent>
        </Dialog>

        {/* Session Options Menu */}
        <Menu
          anchorEl={menuAnchor?.element}
          open={Boolean(menuAnchor)}
          onClose={handleMenuClose}
          anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
          transformOrigin={{ vertical: 'top', horizontal: 'right' }}
          slotProps={{
            paper: {
              sx: {
                minWidth: 160,
                bgcolor: designTokens.colors.surface[3],
                border: '1px solid',
                borderColor: 'divider',
              },
            },
          }}
        >
          <MenuItem
            onClick={handleDeleteClick}
            sx={{
              color: 'error.main',
              '&:hover': {
                bgcolor: 'rgba(239, 68, 68, 0.1)',
              },
            }}
          >
            <ListItemIcon>
              <DeleteIcon fontSize="small" sx={{ color: 'error.main' }} />
            </ListItemIcon>
            <ListItemText>Delete session</ListItemText>
          </MenuItem>
        </Menu>

        {/* Delete Confirmation Dialog */}
        <Dialog
          open={deleteDialogOpen}
          onClose={handleDeleteCancel}
          maxWidth="xs"
          fullWidth
        >
          <DialogTitle sx={{ pb: 1 }}>
            Delete session?
          </DialogTitle>
          <DialogContent>
            <Typography color="text.secondary">
              Are you sure you want to delete <strong>{sessionToDelete?.name}</strong>? This action cannot be undone.
            </Typography>
          </DialogContent>
          <DialogActions sx={{ px: 3, pb: 2 }}>
            <Button onClick={handleDeleteCancel} disabled={deleting}>
              Cancel
            </Button>
            <Button
              onClick={handleDeleteConfirm}
              variant="contained"
              color="error"
              disabled={deleting}
            >
              {deleting ? 'Deleting...' : 'Delete'}
            </Button>
          </DialogActions>
        </Dialog>
      </Container>
    </Box>
  );
}
