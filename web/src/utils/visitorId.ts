/**
 * Visitor ID management for anonymous user tracking.
 * Stored in localStorage and sent with all API requests.
 */

const VISITOR_ID_KEY = 'rallycut_visitor_id';

/**
 * Get or create a visitor ID.
 * Creates a new UUID if one doesn't exist in localStorage.
 */
export function getVisitorId(): string {
  if (typeof window === 'undefined') {
    // Server-side rendering - return empty string
    return '';
  }

  try {
    let visitorId = localStorage.getItem(VISITOR_ID_KEY);

    if (!visitorId) {
      visitorId = crypto.randomUUID();
      localStorage.setItem(VISITOR_ID_KEY, visitorId);
    }

    return visitorId;
  } catch (e) {
    // localStorage might be unavailable (private browsing, quota exceeded, etc.)
    console.warn('Failed to access localStorage for visitor ID:', e);
    // Return a session-scoped ID as fallback
    return crypto.randomUUID();
  }
}

/**
 * Check if visitor ID exists (user has been initialized).
 */
export function hasVisitorId(): boolean {
  if (typeof window === 'undefined') {
    return false;
  }
  return localStorage.getItem(VISITOR_ID_KEY) !== null;
}

/**
 * Clear visitor ID (for testing/debugging).
 */
export function clearVisitorId(): void {
  if (typeof window === 'undefined') {
    return;
  }
  localStorage.removeItem(VISITOR_ID_KEY);
}
