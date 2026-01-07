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
} from '@mui/material';
import { VolleyballProgress } from './VolleyballProgress';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import VideoLibraryIcon from '@mui/icons-material/VideoLibrary';
import SearchIcon from '@mui/icons-material/Search';
import CloseIcon from '@mui/icons-material/Close';
import {
  listVideos,
  addVideoToSession,
  getVideoStreamUrl,
  type VideoListItem,
} from '@/services/api';
import { useUploadStore } from '@/stores/uploadStore';
import { isValidVideoFile } from '@/utils/fileHandlers';

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
  sessionId: string;
  existingVideoIds: string[];
  onVideoAdded: () => void;
}

export function AddVideoModal({
  open,
  onClose,
  sessionId,
  existingVideoIds,
  onVideoAdded,
}: AddVideoModalProps) {
  const [tabIndex, setTabIndex] = useState(0);
  const [videos, setVideos] = useState<VideoListItem[]>([]);
  const [loadingVideos, setLoadingVideos] = useState(false);
  const [selectedVideoIds, setSelectedVideoIds] = useState<string[]>([]);
  const [addingVideos, setAddingVideos] = useState(false);
  const [search, setSearch] = useState('');
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const { isUploading, progress, currentStep, uploadVideo } = useUploadStore();

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
    if (selectedVideoIds.length === 0) return;

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

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!isValidVideoFile(file)) {
      setError('Invalid video format. Please use MP4, MOV, or WebM.');
      return;
    }

    e.target.value = '';

    const success = await uploadVideo(sessionId, file);
    if (success) {
      onVideoAdded();
      onClose();
    }
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (!file) return;

    if (!isValidVideoFile(file)) {
      setError('Invalid video format. Please use MP4, MOV, or WebM.');
      return;
    }

    const success = await uploadVideo(sessionId, file);
    if (success) {
      onVideoAdded();
      onClose();
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
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
          <Box
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onClick={() => !isUploading && fileInputRef.current?.click()}
            sx={{
              border: '2px dashed',
              borderColor: 'grey.600',
              borderRadius: 2,
              p: 4,
              textAlign: 'center',
              cursor: isUploading ? 'default' : 'pointer',
              transition: 'all 0.2s',
              '&:hover': {
                borderColor: isUploading ? 'grey.600' : 'primary.main',
                bgcolor: isUploading ? 'transparent' : 'action.hover',
              },
            }}
          >
            {isUploading ? (
              <VolleyballProgress
                progress={progress}
                stepText={currentStep}
                size="md"
                showPercentage
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
