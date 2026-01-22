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
import FolderIcon from '@mui/icons-material/Folder';
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
import { AddVideoModal } from '@/components/AddVideoModal';
import {
  PageHeader,
  StatsBar,
  SessionCard,
  SectionHeader,
  EmptyState,
} from '@/components/dashboard';

interface SessionGroup {
  session: {
    id: string;
    name: string;
  };
  videos: VideoListItem[];
  updatedAt: string;
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
  const [addToEmptySession, setAddToEmptySession] = useState<{ id: string; name: string } | null>(null);

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
            updatedAt: video.createdAt,
          });
        }
        const group = groups.get(session.id)!;
        group.videos.push(video);
        // Track most recent video date
        if (new Date(video.createdAt) > new Date(group.updatedAt)) {
          group.updatedAt = video.createdAt;
        }
      });
    });

    // Sort by most recently updated
    return Array.from(groups.values()).sort(
      (a, b) => new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime()
    );
  }, [readyVideos]);

  // Find empty sessions (sessions with no videos)
  const emptySessions = useMemo(() => {
    const sessionIdsWithVideos = new Set(groupedVideos.map((g) => g.session.id));
    return regularSessions.filter((s) => !sessionIdsWithVideos.has(s.id));
  }, [regularSessions, groupedVideos]);

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

  const generateSessionName = () => {
    const baseName = `Session - ${new Date().toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
    })}`;

    // Check if base name already exists
    const existingNames = new Set(
      regularSessions.map((s) => s.name.toLowerCase())
    );

    if (!existingNames.has(baseName.toLowerCase())) {
      return baseName;
    }

    // Find the next available number suffix
    let counter = 2;
    while (existingNames.has(`${baseName} (${counter})`.toLowerCase())) {
      counter++;
    }
    return `${baseName} (${counter})`;
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
        const storeError = useUploadStore.getState().error;
        if (storeError) {
          lastUploadError = storeError;
        }
        console.error(`Failed to upload ${file.name}`);
        continue;
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
        overflow: 'auto',
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
          height: 400,
          background: 'radial-gradient(ellipse at 50% 0%, rgba(255, 107, 74, 0.08) 0%, transparent 70%)',
          pointerEvents: 'none',
        },
      }}
    >
      <Container maxWidth="lg" sx={{ position: 'relative' }}>
        {/* Header */}
        <PageHeader
          icon={<SportsVolleyballIcon />}
          title="RallyCut"
          subtitle="Beach Volleyball Video Analysis"
          action={
            hasVideos && (
              <Button
                variant="contained"
                startIcon={<CloudUploadIcon />}
                onClick={handleUploadDialogOpen}
                sx={{
                  px: 3,
                  py: 1.25,
                  transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
                  '&:hover': {
                    transform: 'translateY(-2px)',
                    boxShadow: '0 6px 20px rgba(255, 107, 74, 0.4)',
                  },
                }}
              >
                Upload Video
              </Button>
            )
          }
        />

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
                <EmptyState
                  variant="no-videos"
                  onAction={handleUploadDialogOpen}
                  title="Start your own session"
                  description="Upload a volleyball video to create your first session"
                  actionLabel="Upload Video"
                />

                <Box sx={{ mt: 6 }}>
                  <SectionHeader
                    icon={<PeopleIcon />}
                    title="Shared with me"
                    count={sharedSessions.length}
                    color="secondary"
                  />
                  <Grid container spacing={3}>
                    {sharedSessions.map((session) => (
                      <Grid size={{ xs: 12, sm: 6, md: 4 }} key={session.id}>
                        <SessionCard
                          id={session.id}
                          name={session.name}
                          videoCount={session.videoCount}
                          highlightCount={session.highlightCount}
                          variant="shared"
                          sharedBy={session.ownerName || 'Unknown'}
                          onClick={() => router.push(`/sessions/${session.id}`)}
                        />
                      </Grid>
                    ))}
                  </Grid>
                </Box>
              </>
            ) : (
              /* No videos, no shared sessions - Main empty state */
              <EmptyState
                variant="no-videos"
                onAction={handleUploadDialogOpen}
              />
            )}
          </>
        ) : (
          /* Has videos - show stats, All Videos card and sessions */
          <>
            {/* Stats Bar */}
            <StatsBar
              videoCount={readyVideos.length}
              sessionCount={groupedVideos.length + emptySessions.length}
              loading={loading}
            />

            {/* All Videos Hero Card */}
            {allVideosSession && (
              <Card
                sx={{
                  mb: 4,
                  position: 'relative',
                  overflow: 'hidden',
                  bgcolor: designTokens.colors.surface[1],
                  borderRadius: 3,
                  border: '1px solid',
                  borderColor: 'rgba(255, 107, 74, 0.2)',
                  transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                  '&:hover': {
                    transform: 'translateY(-4px)',
                    boxShadow: `0 20px 40px rgba(0, 0, 0, 0.4), ${designTokens.shadows.glow.primary}`,
                    borderColor: 'rgba(255, 107, 74, 0.4)',
                    '& .thumbnail-strip': {
                      transform: 'scale(1.05)',
                    },
                    '& .arrow-icon': {
                      transform: 'translateX(4px)',
                    },
                    '& .view-text': {
                      opacity: 1,
                      transform: 'translateX(0)',
                    },
                  },
                }}
              >
                <CardActionArea onClick={() => router.push('/videos')}>
                  {/* Thumbnail Strip Background */}
                  <Box
                    sx={{
                      position: 'relative',
                      height: { xs: 100, sm: 120 },
                      overflow: 'hidden',
                    }}
                  >
                    {/* Thumbnail Grid */}
                    <Box
                      className="thumbnail-strip"
                      sx={{
                        display: 'flex',
                        position: 'absolute',
                        inset: 0,
                        transition: 'transform 0.5s cubic-bezier(0.4, 0, 0.2, 1)',
                      }}
                    >
                      {readyVideos.slice(0, 6).map((video, index) => (
                        <Box
                          key={video.id}
                          sx={{
                            flex: 1,
                            minWidth: { xs: '33.33%', sm: '16.66%' },
                            height: '100%',
                            position: 'relative',
                            display: index >= 3 ? { xs: 'none', sm: 'block' } : 'block',
                          }}
                        >
                          {video.posterS3Key ? (
                            <img
                              src={getVideoStreamUrl(video.posterS3Key)}
                              alt=""
                              style={{
                                width: '100%',
                                height: '100%',
                                objectFit: 'cover',
                                display: 'block',
                              }}
                            />
                          ) : (
                            <Box
                              sx={{
                                width: '100%',
                                height: '100%',
                                bgcolor: designTokens.colors.surface[2],
                              }}
                            />
                          )}
                        </Box>
                      ))}
                      {/* Fill remaining slots if less than 6 videos */}
                      {readyVideos.length < 6 &&
                        Array.from({ length: 6 - readyVideos.length }).map((_, i) => (
                          <Box
                            key={`empty-${i}`}
                            sx={{
                              flex: 1,
                              minWidth: { xs: '33.33%', sm: '16.66%' },
                              height: '100%',
                              bgcolor: designTokens.colors.surface[2],
                              display: i + readyVideos.length >= 3 ? { xs: 'none', sm: 'block' } : 'block',
                            }}
                          />
                        ))}
                    </Box>

                    {/* Gradient Overlays */}
                    <Box
                      sx={{
                        position: 'absolute',
                        inset: 0,
                        background: 'linear-gradient(to bottom, rgba(13, 14, 18, 0.3) 0%, rgba(13, 14, 18, 0.95) 100%)',
                        pointerEvents: 'none',
                      }}
                    />
                    <Box
                      sx={{
                        position: 'absolute',
                        inset: 0,
                        background: 'linear-gradient(135deg, rgba(255, 107, 74, 0.15) 0%, transparent 50%)',
                        pointerEvents: 'none',
                      }}
                    />
                  </Box>

                  {/* Content */}
                  <CardContent
                    sx={{
                      position: 'relative',
                      py: 2.5,
                      px: 3,
                      mt: -4,
                    }}
                  >
                    <Stack
                      direction="row"
                      alignItems="center"
                      justifyContent="space-between"
                      spacing={2}
                    >
                      <Stack direction="row" alignItems="center" spacing={2}>
                        <Box
                          sx={{
                            width: 48,
                            height: 48,
                            borderRadius: 2,
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            background: designTokens.gradients.primary,
                            boxShadow: designTokens.shadows.glow.primary,
                          }}
                        >
                          <CollectionsIcon sx={{ color: 'white', fontSize: 24 }} />
                        </Box>
                        <Box>
                          <Typography
                            variant="h5"
                            fontWeight={700}
                            sx={{
                              background: 'linear-gradient(135deg, #FFFFFF 0%, #A1A7B4 100%)',
                              backgroundClip: 'text',
                              WebkitBackgroundClip: 'text',
                              WebkitTextFillColor: 'transparent',
                            }}
                          >
                            Video Library
                          </Typography>
                          <Stack direction="row" alignItems="center" spacing={1.5} sx={{ mt: 0.5 }}>
                            <Chip
                              label={`${readyVideos.length} video${readyVideos.length !== 1 ? 's' : ''}`}
                              size="small"
                              sx={{
                                bgcolor: 'rgba(255, 107, 74, 0.2)',
                                color: 'primary.light',
                                fontWeight: 600,
                                fontSize: '0.75rem',
                                height: 24,
                              }}
                            />
                            <Typography variant="body2" color="text.secondary">
                              Browse & create highlights
                            </Typography>
                          </Stack>
                        </Box>
                      </Stack>

                      <Stack direction="row" alignItems="center" spacing={1}>
                        <Typography
                          className="view-text"
                          variant="body2"
                          sx={{
                            color: 'primary.main',
                            fontWeight: 600,
                            opacity: 0,
                            transform: 'translateX(-8px)',
                            transition: 'all 0.2s ease',
                          }}
                        >
                          View all
                        </Typography>
                        <Box
                          sx={{
                            width: 36,
                            height: 36,
                            borderRadius: '50%',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            bgcolor: 'rgba(255, 107, 74, 0.1)',
                            border: '1px solid rgba(255, 107, 74, 0.3)',
                          }}
                        >
                          <ChevronRightIcon
                            className="arrow-icon"
                            sx={{
                              color: 'primary.main',
                              fontSize: 22,
                              transition: 'transform 0.2s ease',
                            }}
                          />
                        </Box>
                      </Stack>
                    </Stack>
                  </CardContent>
                </CardActionArea>
              </Card>
            )}

            {/* Your Sessions */}
            {groupedVideos.length > 0 && (
              <Box sx={{ mb: 5 }}>
                <SectionHeader
                  icon={<FolderIcon />}
                  title="Your Sessions"
                  count={groupedVideos.length}
                />
                <Grid container spacing={3}>
                  {groupedVideos.map(({ session, videos: sessionVideos, updatedAt }) => (
                    <Grid size={{ xs: 12, sm: 6, md: 4 }} key={session.id}>
                      <Box sx={{ position: 'relative' }}>
                        <SessionCard
                          id={session.id}
                          name={session.name}
                          videoCount={sessionVideos.length}
                          updatedAt={updatedAt}
                          videos={sessionVideos.slice(0, 4).map(v => ({
                            id: v.id,
                            posterS3Key: v.posterS3Key,
                          }))}
                          onClick={() => router.push(`/sessions/${session.id}`)}
                        />
                        <IconButton
                          size="small"
                          onClick={(e) => handleMenuOpen(e, session.id, session.name)}
                          sx={{
                            position: 'absolute',
                            top: 8,
                            right: 8,
                            bgcolor: 'rgba(0, 0, 0, 0.5)',
                            backdropFilter: 'blur(4px)',
                            color: 'white',
                            '&:hover': {
                              bgcolor: 'rgba(0, 0, 0, 0.7)',
                            },
                          }}
                        >
                          <MoreVertIcon fontSize="small" />
                        </IconButton>
                      </Box>
                    </Grid>
                  ))}
                </Grid>
              </Box>
            )}

            {/* Empty Sessions */}
            {emptySessions.length > 0 && (
              <Box sx={{ mb: 5 }}>
                <SectionHeader
                  icon={<FolderIcon />}
                  title="Empty Sessions"
                  count={emptySessions.length}
                />
                <Grid container spacing={3}>
                  {emptySessions.map((session) => (
                    <Grid size={{ xs: 12, sm: 6, md: 4 }} key={session.id}>
                      <Box sx={{ position: 'relative' }}>
                        <SessionCard
                          id={session.id}
                          name={session.name}
                          videoCount={0}
                          onAddVideos={() => setAddToEmptySession({ id: session.id, name: session.name })}
                        />
                        <IconButton
                          size="small"
                          onClick={(e) => handleMenuOpen(e, session.id, session.name)}
                          sx={{
                            position: 'absolute',
                            top: 8,
                            right: 8,
                            bgcolor: 'rgba(0, 0, 0, 0.3)',
                            color: 'text.secondary',
                            '&:hover': {
                              bgcolor: 'rgba(0, 0, 0, 0.5)',
                              color: 'white',
                            },
                          }}
                        >
                          <MoreVertIcon fontSize="small" />
                        </IconButton>
                      </Box>
                    </Grid>
                  ))}
                </Grid>
              </Box>
            )}

            {/* Shared with me */}
            {!loadingShared && sharedSessions.length > 0 && (
              <Box sx={{ mt: 5 }}>
                <SectionHeader
                  icon={<PeopleIcon />}
                  title="Shared with me"
                  count={sharedSessions.length}
                  color="secondary"
                />
                <Grid container spacing={3}>
                  {sharedSessions.map((session) => (
                    <Grid size={{ xs: 12, sm: 6, md: 4 }} key={session.id}>
                      <SessionCard
                        id={session.id}
                        name={session.name}
                        videoCount={session.videoCount}
                        highlightCount={session.highlightCount}
                        variant="shared"
                        sharedBy={session.ownerName || 'Unknown'}
                        onClick={() => router.push(`/sessions/${session.id}`)}
                      />
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

        {/* Add Video Modal for empty sessions */}
        {addToEmptySession && (
          <AddVideoModal
            open={true}
            onClose={() => setAddToEmptySession(null)}
            sessionId={addToEmptySession.id}
            existingVideoIds={[]}
            onVideoAdded={() => {
              setAddToEmptySession(null);
              loadData();
            }}
          />
        )}
      </Container>
    </Box>
  );
}
