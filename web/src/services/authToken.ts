/**
 * JWT token management for Express API calls.
 * Caches the token in memory. Refreshed periodically by AuthSync.
 *
 * Auth readiness gate: on page load, API calls that depend on user identity
 * (e.g., fetchSession) must await waitForAuthReady() to avoid race conditions
 * where the request fires before the JWT is available, causing the backend to
 * resolve a different (anonymous) user.
 */

let cachedToken: string | null = null;
let fetchInFlight: Promise<string | null> | null = null;

// Auth readiness gate — resolves once we know the user's auth state
let authReadyResolve: (() => void) | null = null;
const authReadyPromise = new Promise<void>((resolve) => {
  authReadyResolve = resolve;
});

/**
 * Wait for auth initialization to complete.
 * Resolves once AuthSync has determined the user's auth state
 * (either JWT fetched for authenticated users, or confirmed unauthenticated).
 * Safe to call multiple times — resolves immediately after first signal.
 * Times out after 3s as a safety net (proceeds with visitor ID only).
 */
export function waitForAuthReady(): Promise<void> {
  if (typeof window === 'undefined') return Promise.resolve(); // SSR
  return Promise.race([
    authReadyPromise,
    new Promise<void>((resolve) => setTimeout(resolve, 3000)),
  ]);
}

/**
 * Signal that auth initialization is complete.
 * Called by AuthSync after JWT is fetched (authenticated) or
 * when confirmed unauthenticated.
 */
export function signalAuthReady(): void {
  authReadyResolve?.();
}

/**
 * Get the cached auth token (synchronous).
 * Returns null if not authenticated or token not yet fetched.
 */
export function getAuthToken(): string | null {
  return cachedToken;
}

/**
 * Fetch a fresh auth token from the Next.js API.
 * Deduplicates concurrent calls to avoid redundant fetches.
 */
export async function refreshAuthToken(): Promise<string | null> {
  if (fetchInFlight) return fetchInFlight;

  fetchInFlight = doRefresh();
  try {
    return await fetchInFlight;
  } finally {
    fetchInFlight = null;
  }
}

async function doRefresh(): Promise<string | null> {
  try {
    const response = await fetch('/api/auth/token');
    if (!response.ok) {
      cachedToken = null;
      return null;
    }

    const data = await response.json();
    cachedToken = data.token;
    return cachedToken;
  } catch {
    return null;
  }
}

/**
 * Clear the cached token (call on sign-out).
 */
export function clearAuthToken(): void {
  cachedToken = null;
}
