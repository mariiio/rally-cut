import { create } from 'zustand';

// Minimal rally info needed for playback
interface PlaylistRally {
  id: string;
  matchId: string;
  start_time: number;
  end_time: number;
}

// Buffered time range
interface BufferedRange {
  start: number;
  end: number;
}

interface PlayerState {
  isPlaying: boolean;
  currentTime: number;
  duration: number;
  isReady: boolean;
  seekTo: number | null; // When set, player should seek to this time
  playOnlyRallies: boolean; // Skip dead time between rallies
  applyCameraEdits: boolean; // Apply camera edits during playback
  bufferedRanges: BufferedRange[]; // Which parts of the video are buffered
  playbackRate: number; // Playback speed (0.5, 1, 2)

  // Highlight playback state
  playingHighlightId: string | null;
  highlightPlaylist: PlaylistRally[]; // Stable list of rallies to play
  highlightRallyIndex: number;
  pendingMatchSwitch: string | null; // Match ID we're switching to

  // Actions
  play: () => void;
  pause: () => void;
  togglePlay: () => void;
  seek: (time: number) => void;
  clearSeek: () => void;
  setCurrentTime: (time: number) => void;
  setDuration: (duration: number) => void;
  setReady: (ready: boolean) => void;
  setBufferedRanges: (ranges: BufferedRange[]) => void;
  togglePlayOnlyRallies: () => void;
  toggleApplyCameraEdits: () => void;
  setApplyCameraEdits: (apply: boolean) => void;
  setPlaybackRate: (rate: number) => void;
  cyclePlaybackRate: (direction: 'faster' | 'slower') => void;

  // Highlight playback actions
  startHighlightPlayback: (highlightId: string, playlist: PlaylistRally[], startIndex?: number) => void;
  advanceHighlightPlayback: () => PlaylistRally | null; // Returns next rally or null if done
  jumpToPlaylistRally: (rallyId: string) => PlaylistRally | null; // Jump to specific rally in playlist
  clearPendingMatchSwitch: () => void;
  stopHighlightPlayback: () => void;
  getCurrentPlaylistRally: () => PlaylistRally | null;
}

export const usePlayerStore = create<PlayerState>((set, get) => ({
  isPlaying: false,
  currentTime: 0,
  duration: 0,
  isReady: false,
  seekTo: null,
  playOnlyRallies: false,
  applyCameraEdits: true,
  bufferedRanges: [],
  playbackRate: 1,

  // Highlight playback state
  playingHighlightId: null,
  highlightPlaylist: [],
  highlightRallyIndex: 0,
  pendingMatchSwitch: null,

  play: () => set({ isPlaying: true }),
  pause: () => set({ isPlaying: false }),
  togglePlay: () => set((state) => ({ isPlaying: !state.isPlaying })),

  seek: (time: number) => {
    const state = get();
    // Clamp to valid range
    const clampedTime = Math.max(0, Math.min(time, state.duration || time));
    set({ seekTo: clampedTime, currentTime: clampedTime });
  },

  clearSeek: () => set({ seekTo: null }),

  setCurrentTime: (time: number) => set({ currentTime: time }),
  setDuration: (duration: number) => set({ duration }),
  setReady: (ready: boolean) => set({ isReady: ready }),
  setBufferedRanges: (ranges) => set({ bufferedRanges: ranges }),
  togglePlayOnlyRallies: () => {
    const newValue = !get().playOnlyRallies;
    set({ playOnlyRallies: newValue });
    if (typeof window !== 'undefined') {
      localStorage.setItem('rallycut-play-only-rallies', String(newValue));
    }
  },
  toggleApplyCameraEdits: () => {
    const newValue = !get().applyCameraEdits;
    set({ applyCameraEdits: newValue });
    if (typeof window !== 'undefined') {
      localStorage.setItem('rallycut-apply-camera-edits', String(newValue));
    }
  },
  setApplyCameraEdits: (apply) => {
    set({ applyCameraEdits: apply });
    if (typeof window !== 'undefined') {
      localStorage.setItem('rallycut-apply-camera-edits', String(apply));
    }
  },

  setPlaybackRate: (rate) => {
    const validRates = [0.5, 1, 2];
    if (!validRates.includes(rate)) return;
    set({ playbackRate: rate });
    if (typeof window !== 'undefined') {
      localStorage.setItem('rallycut-playback-rate', String(rate));
    }
  },

  cyclePlaybackRate: (direction) => {
    const rates = [0.5, 1, 2];
    const idx = rates.indexOf(get().playbackRate);
    const newIdx = direction === 'faster'
      ? Math.min(rates.length - 1, idx + 1)
      : Math.max(0, idx - 1);
    get().setPlaybackRate(rates[newIdx]);
  },

  // Highlight playback actions
  startHighlightPlayback: (highlightId: string, playlist: PlaylistRally[], startIndex = 0) => {
    set({
      playingHighlightId: highlightId,
      highlightPlaylist: playlist,
      highlightRallyIndex: startIndex,
      pendingMatchSwitch: null,
      isPlaying: true,
    });
  },

  advanceHighlightPlayback: () => {
    const state = get();
    const nextIndex = state.highlightRallyIndex + 1;
    const nextRally = state.highlightPlaylist[nextIndex] || null;

    if (nextRally) {
      const currentRally = state.highlightPlaylist[state.highlightRallyIndex];
      const needsMatchSwitch = currentRally && nextRally.matchId !== currentRally.matchId;

      set({
        highlightRallyIndex: nextIndex,
        pendingMatchSwitch: needsMatchSwitch ? nextRally.matchId : null,
      });
    }

    return nextRally;
  },

  jumpToPlaylistRally: (rallyId: string) => {
    const state = get();
    const targetIndex = state.highlightPlaylist.findIndex((r) => r.id === rallyId);
    if (targetIndex === -1) return null;

    const targetRally = state.highlightPlaylist[targetIndex];
    const currentRally = state.highlightPlaylist[state.highlightRallyIndex];
    const needsMatchSwitch = currentRally && targetRally.matchId !== currentRally.matchId;

    set({
      highlightRallyIndex: targetIndex,
      pendingMatchSwitch: needsMatchSwitch ? targetRally.matchId : null,
      seekTo: targetRally.start_time,
      currentTime: targetRally.start_time,
      isPlaying: true,
    });

    return targetRally;
  },

  clearPendingMatchSwitch: () => set({ pendingMatchSwitch: null }),

  stopHighlightPlayback: () => {
    set({
      playingHighlightId: null,
      highlightPlaylist: [],
      highlightRallyIndex: 0,
      pendingMatchSwitch: null,
      isPlaying: false,
    });
  },

  getCurrentPlaylistRally: () => {
    const state = get();
    return state.highlightPlaylist[state.highlightRallyIndex] || null;
  },
}));

// Initialize settings from localStorage on client side
if (typeof window !== 'undefined') {
  const savedApplyCameraEdits = localStorage.getItem('rallycut-apply-camera-edits');
  if (savedApplyCameraEdits !== null) {
    usePlayerStore.setState({ applyCameraEdits: savedApplyCameraEdits === 'true' });
  }
  const savedPlayOnlyRallies = localStorage.getItem('rallycut-play-only-rallies');
  if (savedPlayOnlyRallies !== null) {
    usePlayerStore.setState({ playOnlyRallies: savedPlayOnlyRallies === 'true' });
  }
  const savedRate = localStorage.getItem('rallycut-playback-rate');
  if (savedRate) {
    const rate = parseFloat(savedRate);
    if ([0.5, 1, 2].includes(rate)) {
      usePlayerStore.setState({ playbackRate: rate });
    }
  }
}
