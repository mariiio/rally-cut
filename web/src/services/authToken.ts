/**
 * JWT token management for Express API calls.
 * Caches the token in memory. Refreshed periodically by AuthSync.
 */

let cachedToken: string | null = null;
let fetchInFlight: Promise<string | null> | null = null;

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
