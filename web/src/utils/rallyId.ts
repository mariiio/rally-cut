/**
 * Create a frontend rally ID from a match ID and order index.
 * Format: `{matchId}_rally_{order}`
 */
export function createRallyId(matchId: string, order: number): string {
  return `${matchId}_rally_${order}`;
}

/**
 * Parse a frontend rally ID back into its components.
 * Returns null if the ID doesn't match the expected format.
 */
export function parseRallyId(id: string): { matchId: string; order: number } | null {
  const idx = id.lastIndexOf('_rally_');
  if (idx === -1) return null;
  const matchId = id.slice(0, idx);
  const order = parseInt(id.slice(idx + 7), 10);
  if (isNaN(order)) return null;
  return { matchId, order };
}
