'use client';

import { SessionProvider } from 'next-auth/react';
import { AuthSync } from './AuthSync';
import { AuthPromptModal } from './AuthPromptModal';

export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <SessionProvider>
      <AuthSync />
      <AuthPromptModal />
      {children}
    </SessionProvider>
  );
}
