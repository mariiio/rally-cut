import { create } from 'zustand';

interface PlayerState {
  isPlaying: boolean;
  currentTime: number;
  duration: number;
  isReady: boolean;
  seekTo: number | null; // When set, player should seek to this time
  playOnlyRallies: boolean; // Skip dead time between rallies

  // Highlight playback state
  playingHighlightId: string | null; // Currently playing highlight
  highlightRallyIndex: number; // Current rally index in highlight playback

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
  startHighlightPlayback: (highlightId: string) => void;
  advanceHighlightPlayback: () => void;
  stopHighlightPlayback: () => void;
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
  highlightRallyIndex: 0,

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
  startHighlightPlayback: (highlightId: string) => {
    set({
      playingHighlightId: highlightId,
      highlightRallyIndex: 0,
      isPlaying: true,
    });
  },

  advanceHighlightPlayback: () => {
    set((state) => ({
      highlightRallyIndex: state.highlightRallyIndex + 1,
    }));
  },

  stopHighlightPlayback: () => {
    set({
      playingHighlightId: null,
      highlightRallyIndex: 0,
      isPlaying: false,
    });
  },
}));
