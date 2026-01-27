import { create } from 'zustand';

interface AuthState {
  isAuthenticated: boolean;
  userId: string | null;
  email: string | null;
  name: string | null;
  avatarUrl: string | null;

  // Auth prompt modal
  showAuthModal: boolean;
  authModalReason: string | null;

  // Actions
  setUser: (user: {
    id: string;
    email?: string | null;
    name?: string | null;
    image?: string | null;
  }) => void;
  clearUser: () => void;
  promptSignIn: (reason: string) => void;
  closeAuthPrompt: () => void;
}

export const useAuthStore = create<AuthState>((set) => ({
  isAuthenticated: false,
  userId: null,
  email: null,
  name: null,
  avatarUrl: null,
  showAuthModal: false,
  authModalReason: null,

  setUser: (user) =>
    set({
      isAuthenticated: true,
      userId: user.id,
      email: user.email ?? null,
      name: user.name ?? null,
      avatarUrl: user.image ?? null,
    }),

  clearUser: () =>
    set({
      isAuthenticated: false,
      userId: null,
      email: null,
      name: null,
      avatarUrl: null,
    }),

  promptSignIn: (reason) =>
    set({
      showAuthModal: true,
      authModalReason: reason,
    }),

  closeAuthPrompt: () =>
    set({
      showAuthModal: false,
      authModalReason: null,
    }),
}));
