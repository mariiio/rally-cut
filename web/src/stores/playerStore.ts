import { create } from 'zustand';

// Minimal rally info needed for playback
interface PlaylistRally {
  id: string;
  matchId: string;
  start_time: number;
  end_time: number;
}

interface PlayerState {
  isPlaying: boolean;
  currentTime: number;
  duration: number;
  isReady: boolean;
  seekTo: number | null; // When set, player should seek to this time
  playOnlyRallies: boolean; // Skip dead time between rallies

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
  togglePlayOnlyRallies: () => void;

  // Highlight playback actions
  startHighlightPlayback: (highlightId: string, playlist: PlaylistRally[]) => void;
  advanceHighlightPlayback: () => PlaylistRally | null; // Returns next rally or null if done
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
  togglePlayOnlyRallies: () => set((state) => ({ playOnlyRallies: !state.playOnlyRallies })),

  // Highlight playback actions
  startHighlightPlayback: (highlightId: string, playlist: PlaylistRally[]) => {
    set({
      playingHighlightId: highlightId,
      highlightPlaylist: playlist,
      highlightRallyIndex: 0,
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
