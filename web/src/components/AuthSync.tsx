'use client';

import { useEffect, useRef } from 'react';
import { useSession } from 'next-auth/react';
import { useAuthStore } from '@/stores/authStore';
import { refreshAuthToken, clearAuthToken } from '@/services/authToken';
import { API_BASE_URL, getHeaders, invalidateUserCache } from '@/services/api';

/**
 * Syncs Auth.js session state with the auth Zustand store,
 * manages the API JWT token cache, and handles post-OAuth
 * anonymous data linking.
 */
export function AuthSync() {
  const { data: session, status } = useSession();
  const { setUser, clearUser } = useAuthStore();
  const hasLinked = useRef(false);

  useEffect(() => {
    if (status === 'authenticated' && session?.user) {
      setUser({
        id: session.user.id!,
        email: session.user.email,
        name: session.user.name,
        image: session.user.image,
      });
      // Fetch JWT for Express API calls
      refreshAuthToken().then(() => {
        // After getting the JWT, try to link anonymous data
        if (!hasLinked.current) {
          hasLinked.current = true;
          linkAnonymousData();
        }
      });
    } else if (status === 'unauthenticated') {
      clearUser();
      clearAuthToken();
      hasLinked.current = false;
    }
  }, [session, status, setUser, clearUser]);

  // Refresh token periodically (every 4 minutes, cache TTL is 5)
  useEffect(() => {
    if (status !== 'authenticated') return;

    const interval = setInterval(() => {
      refreshAuthToken();
    }, 4 * 60 * 1000);

    return () => clearInterval(interval);
  }, [status]);

  return null;
}

/**
 * After OAuth sign-in, link anonymous user data to the authenticated user.
 * Uses the visitorId stored in sessionStorage by the register page.
 */
async function linkAnonymousData() {
  try {
    // Check for stored visitorId from pre-OAuth flow
    const visitorId = typeof window !== 'undefined'
      ? sessionStorage.getItem('rallycut_link_visitor_id')
      : null;

    if (!visitorId) return;

    // Remove it so we don't try again
    sessionStorage.removeItem('rallycut_link_visitor_id');

    const response = await fetch(`${API_BASE_URL}/v1/auth/link-anonymous`, {
      method: 'POST',
      headers: getHeaders('application/json'),
      body: JSON.stringify({ visitorId }),
    });

    if (response.ok) {
      // Invalidate user cache since data may have changed
      invalidateUserCache();
    }
  } catch {
    // Non-critical, don't throw
  }
}
