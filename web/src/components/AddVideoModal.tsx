'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Box,
  Typography,
  Tabs,
  Tab,
  Checkbox,
  CircularProgress,
  Chip,
  TextField,
  InputAdornment,
  IconButton,
  Alert,
  FormControl,
  RadioGroup,
  FormControlLabel,
  Radio,
  Divider,
} from '@mui/material';
import { VolleyballProgress } from './VolleyballProgress';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import VideoLibraryIcon from '@mui/icons-material/VideoLibrary';
import SearchIcon from '@mui/icons-material/Search';
import CloseIcon from '@mui/icons-material/Close';
import AddIcon from '@mui/icons-material/Add';
import { RecordingGuidelines } from './RecordingGuidelines';
import {
  listVideos,
  listSessions,
  addVideoToSession,
  createSession,
  getVideoStreamUrl,
  type VideoListItem,
  type SessionType,
} from '@/services/api';
import { useUploadStore } from '@/stores/uploadStore';
import { isValidVideoFile } from '@/utils/fileHandlers';

interface SessionOption {
  id: string;
  name: string;
  type: SessionType;
}

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel({ children, value, index }: TabPanelProps) {
  return (
    <Box role="tabpanel" hidden={value !== index} sx={{ py: 2 }}>
      {value === index && children}
    </Box>
  );
}

function formatDuration(ms: number | null): string {
  if (!ms) return '--:--';
  const totalSeconds = Math.floor(ms / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, '0')}`;
}

interface AddVideoModalProps {
  open: boolean;
  onClose: () => void;
  sessionId?: string;
  existingVideoIds?: string[];
  onVideoAdded: () => void;
}

export function AddVideoModal({
  open,
  onClose,
  sessionId,
  existingVideoIds = [],
  onVideoAdded,
}: AddVideoModalProps) {
  const [tabIndex, setTabIndex] = useState(0);
  const [videos, setVideos] = useState<VideoListItem[]>([]);
  const [loadingVideos, setLoadingVideos] = useState(false);
  const [selectedVideoIds, setSelectedVideoIds] = useState<string[]>([]);
  const [addingVideos, setAddingVideos] = useState(false);
  const [search, setSearch] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Session selection state (when sessionId is not provided)
  const [sessions, setSessions] = useState<SessionOption[]>([]);
  const [loadingSessions, setLoadingSessions] = useState(false);
  const [selectedSessionId, setSelectedSessionId] = useState<string>('library'); // 'library', session ID, or 'new'
  const [newSessionName, setNewSessionName] = useState('');
  const [creatingSession, setCreatingSession] = useState(false);

  const { isUploading, progress, currentStep, uploadVideo, uploadVideoToLibrary } = useUploadStore();

  const loadVideos = useCallback(async (searchQuery?: string) => {
    try {
      setLoadingVideos(true);
      setError(null);
      const response = await listVideos(1, 50, searchQuery);
      // Filter out videos already in the session
      const availableVideos = response.data.filter(
        (v) => !existingVideoIds.includes(v.id)
      );
      setVideos(availableVideos);
    } catch (err) {
      console.error('Failed to load videos:', err);
      setError('Failed to load video library');
    } finally {
      setLoadingVideos(false);
    }
  }, [existingVideoIds]);

  // Load videos when tab changes to library
  useEffect(() => {
    if (open && tabIndex === 1) {
      loadVideos();
    }
  }, [open, tabIndex, loadVideos]);

  // Load available sessions when modal opens and no sessionId provided
  useEffect(() => {
    if (open && !sessionId) {
      const fetchSessions = async () => {
        try {
          setLoadingSessions(true);
          const result = await listSessions(1, 100);
          // Filter to only REGULAR sessions (not ALL_VIDEOS)
          const regularSessions = result.data.filter(s => s.type === 'REGULAR');
          setSessions(regularSessions);
        } catch (err) {
          console.error('Failed to load sessions:', err);
        } finally {
          setLoadingSessions(false);
        }
      };
      fetchSessions();
    }
  }, [open, sessionId]);

  // Debounced search
  useEffect(() => {
    if (tabIndex !== 1) return;
    const debounce = setTimeout(() => {
      loadVideos(search || undefined);
    }, 300);
    return () => clearTimeout(debounce);
  }, [search, tabIndex, loadVideos]);

  // Reset state when modal closes
  useEffect(() => {
    if (!open) {
      setTabIndex(0);
      setSelectedVideoIds([]);
      setSearch('');
      setError(null);
      // Reset session selection
      setSelectedSessionId('library');
      setNewSessionName('');
    }
  }, [open]);

  const handleTabChange = (_: React.SyntheticEvent, newValue: number) => {
    setTabIndex(newValue);
  };

  const handleVideoSelect = (videoId: string) => {
    setSelectedVideoIds((prev) =>
      prev.includes(videoId)
        ? prev.filter((id) => id !== videoId)
        : [...prev, videoId]
    );
  };

  const handleAddSelectedVideos = async () => {
    if (selectedVideoIds.length === 0 || !sessionId) return;

    try {
      setAddingVideos(true);
      setError(null);

      // Add all selected videos
      for (const videoId of selectedVideoIds) {
        await addVideoToSession(sessionId, videoId);
      }

      onVideoAdded();
      onClose();
    } catch (err) {
      console.error('Failed to add videos:', err);
      setError('Failed to add videos to session');
    } finally {
      setAddingVideos(false);
    }
  };

  // Shared upload logic for both file select and drag-drop
  const handleUpload = async (file: File) => {
    if (!isValidVideoFile(file)) {
      setError('Invalid video format. Please use MP4, MOV, or WebM.');
      return;
    }

    // If sessionId is provided, upload directly to that session
    if (sessionId) {
      const success = await uploadVideo(sessionId, file);
      if (success) {
        onVideoAdded();
        onClose();
      }
      return;
    }

    // Session selection mode
    try {
      let targetSessionId: string | null = null;

      // Create new session if needed
      if (selectedSessionId === 'new') {
        if (!newSessionName.trim()) {
          setError('Please enter a session name');
          return;
        }
        setCreatingSession(true);
        const session = await createSession(newSessionName.trim());
        targetSessionId = session.id;
        setCreatingSession(false);
      } else if (selectedSessionId !== 'library') {
        targetSessionId = selectedSessionId;
      }

      // Upload to library first
      const result = await uploadVideoToLibrary(file);
      if (!result.success || !result.videoId) {
        return; // Error handled by uploadStore
      }

      // Add to session if one was selected
      if (targetSessionId) {
        await addVideoToSession(targetSessionId, result.videoId);
      }

      onVideoAdded();
      onClose();
    } catch (err) {
      setCreatingSession(false);
      setError(err instanceof Error ? err.message : 'Upload failed');
    }
  };

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    e.target.value = '';
    await handleUpload(file);
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files[0];
    if (!file) return;
    await handleUpload(file);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleDragEnter = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    // Only set to false if leaving the drop zone entirely (not entering a child)
    if (e.currentTarget.contains(e.relatedTarget as Node)) return;
    setIsDragOver(false);
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        Add Video
        <IconButton onClick={onClose} size="small">
          <CloseIcon />
        </IconButton>
      </DialogTitle>

      <Box sx={{ borderBottom: 1, borderColor: 'divider', px: 2 }}>
        <Tabs value={tabIndex} onChange={handleTabChange}>
          <Tab
            icon={<CloudUploadIcon />}
            iconPosition="start"
            label="Upload"
          />
          <Tab
            icon={<VideoLibraryIcon />}
            iconPosition="start"
            label="Library"
          />
        </Tabs>
      </Box>

      <DialogContent sx={{ minHeight: 300 }}>
        {error && (
          <Alert severity="error" onClose={() => setError(null)} sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        <TabPanel value={tabIndex} index={0}>
          {/* Session selector - only show when sessionId is not provided */}
          {!sessionId && !isUploading && (
            <Box sx={{ mb: 3 }}>
              <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 1.5 }}>
                Add to session
              </Typography>
              {loadingSessions ? (
                <Box sx={{ display: 'flex', justifyContent: 'center', py: 2 }}>
                  <CircularProgress size={24} />
                </Box>
              ) : (
                <FormControl component="fieldset" fullWidth>
                  <RadioGroup
                    value={selectedSessionId}
                    onChange={(e) => setSelectedSessionId(e.target.value)}
                  >
                    <FormControlLabel
                      value="library"
                      control={<Radio size="small" />}
                      label={
                        <Typography variant="body2">
                          Library only
                          <Typography component="span" variant="caption" color="text.disabled" sx={{ ml: 1 }}>
                            (no session)
                          </Typography>
                        </Typography>
                      }
                      sx={{ mb: 0.5 }}
                    />
                    {sessions.map((session) => (
                      <FormControlLabel
                        key={session.id}
                        value={session.id}
                        control={<Radio size="small" />}
                        label={<Typography variant="body2">{session.name}</Typography>}
                        sx={{ mb: 0.5 }}
                      />
                    ))}
                    <Divider sx={{ my: 1 }} />
                    <FormControlLabel
                      value="new"
                      control={<Radio size="small" />}
                      label={
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                          <AddIcon sx={{ fontSize: 16 }} />
                          <Typography variant="body2">Create new session</Typography>
                        </Box>
                      }
                    />
                    {selectedSessionId === 'new' && (
                      <TextField
                        placeholder="Session name"
                        value={newSessionName}
                        onChange={(e) => setNewSessionName(e.target.value)}
                        size="small"
                        fullWidth
                        autoFocus
                        disabled={creatingSession}
                        sx={{ ml: 4, mt: 1, maxWidth: 250 }}
                      />
                    )}
                  </RadioGroup>
                </FormControl>
              )}
            </Box>
          )}

          <Box
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragEnter={handleDragEnter}
            onDragLeave={handleDragLeave}
            onClick={() => !isUploading && !creatingSession && fileInputRef.current?.click()}
            sx={{
              border: '2px dashed',
              borderColor: isDragOver && !isUploading ? 'primary.main' : 'grey.600',
              borderRadius: 2,
              p: 4,
              textAlign: 'center',
              cursor: isUploading || creatingSession ? 'default' : 'pointer',
              transition: 'all 0.2s',
              bgcolor: isDragOver && !isUploading ? 'action.hover' : 'transparent',
              '&:hover': {
                borderColor: isUploading || creatingSession ? 'grey.600' : 'primary.main',
                bgcolor: isUploading || creatingSession ? 'transparent' : 'action.hover',
              },
            }}
          >
            {isUploading || creatingSession ? (
              <VolleyballProgress
                progress={creatingSession ? 5 : progress}
                stepText={creatingSession ? 'Creating session...' : currentStep}
                size="md"
                showPercentage={!creatingSession}
              />
            ) : (
              <>
                <CloudUploadIcon sx={{ fontSize: 48, color: 'grey.500', mb: 2 }} />
                <Typography variant="h6" gutterBottom>
                  Drop video here or click to browse
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Supported formats: MP4, MOV, WebM
                </Typography>
              </>
            )}
          </Box>

          {/* Recording Guidelines - only show when not uploading */}
          {!isUploading && !creatingSession && <RecordingGuidelines />}

          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileSelect}
            accept="video/mp4,video/quicktime,video/webm,.mp4,.mov,.webm"
            style={{ display: 'none' }}
          />
        </TabPanel>

        <TabPanel value={tabIndex} index={1}>
          <TextField
            fullWidth
            placeholder="Search videos..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            size="small"
            slotProps={{
              input: {
                startAdornment: (
                  <InputAdornment position="start">
                    <SearchIcon sx={{ color: 'grey.500' }} />
                  </InputAdornment>
                ),
              },
            }}
            sx={{ mb: 2 }}
          />

          {loadingVideos ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
              <CircularProgress />
            </Box>
          ) : videos.length === 0 ? (
            <Box sx={{ textAlign: 'center', py: 4 }}>
              <VideoLibraryIcon sx={{ fontSize: 48, color: 'grey.500', mb: 2 }} />
              <Typography color="text.secondary">
                {search
                  ? 'No videos found matching your search'
                  : existingVideoIds.length > 0
                  ? 'All videos are already in this session'
                  : 'No videos in your library yet'}
              </Typography>
            </Box>
          ) : (
            <Box sx={{ maxHeight: 350, overflow: 'auto' }}>
              <Box
                sx={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(2, 1fr)',
                  gap: 1.5,
                }}
              >
                {videos.map((video) => {
                  const isSelected = selectedVideoIds.includes(video.id);
                  return (
                    <Box
                      key={video.id}
                      onClick={() => handleVideoSelect(video.id)}
                      sx={{
                        cursor: 'pointer',
                        borderRadius: 1.5,
                        overflow: 'hidden',
                        border: '2px solid',
                        borderColor: isSelected ? 'primary.main' : 'transparent',
                        bgcolor: isSelected ? 'action.selected' : 'grey.900',
                        transition: 'all 0.15s',
                        '&:hover': {
                          borderColor: isSelected ? 'primary.main' : 'grey.700',
                          bgcolor: isSelected ? 'action.selected' : 'grey.800',
                        },
                      }}
                    >
                      {/* Poster */}
                      <Box sx={{ position: 'relative', aspectRatio: '16/9', bgcolor: 'grey.800' }}>
                        {video.posterS3Key ? (
                          /* eslint-disable-next-line @next/next/no-img-element -- dynamic API URL */
                          <img
                            src={getVideoStreamUrl(video.posterS3Key)}
                            alt={video.name}
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
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center',
                            }}
                          >
                            <VideoLibraryIcon sx={{ fontSize: 32, color: 'grey.600' }} />
                          </Box>
                        )}
                        {/* Duration badge */}
                        <Chip
                          label={formatDuration(video.durationMs)}
                          size="small"
                          sx={{
                            position: 'absolute',
                            bottom: 4,
                            right: 4,
                            height: 20,
                            fontSize: '0.7rem',
                            bgcolor: 'rgba(0,0,0,0.7)',
                            color: 'white',
                          }}
                        />
                        {/* Selection checkbox */}
                        <Checkbox
                          checked={isSelected}
                          sx={{
                            position: 'absolute',
                            top: 2,
                            left: 2,
                            p: 0.5,
                            bgcolor: 'rgba(0,0,0,0.5)',
                            borderRadius: 1,
                            '&:hover': { bgcolor: 'rgba(0,0,0,0.7)' },
                          }}
                          size="small"
                        />
                      </Box>
                      {/* Video name */}
                      <Box sx={{ p: 1 }}>
                        <Typography
                          variant="body2"
                          noWrap
                          title={video.name}
                          sx={{ fontWeight: 500 }}
                        >
                          {video.name}
                        </Typography>
                      </Box>
                    </Box>
                  );
                })}
              </Box>
            </Box>
          )}
        </TabPanel>
      </DialogContent>

      <DialogActions sx={{ px: 3, pb: 2 }}>
        <Button onClick={onClose}>Cancel</Button>
        {tabIndex === 1 && (
          <Button
            variant="contained"
            onClick={handleAddSelectedVideos}
            disabled={selectedVideoIds.length === 0 || addingVideos}
          >
            {addingVideos
              ? 'Adding...'
              : `Add ${selectedVideoIds.length} Video${selectedVideoIds.length !== 1 ? 's' : ''}`}
          </Button>
        )}
      </DialogActions>
    </Dialog>
  );
}
