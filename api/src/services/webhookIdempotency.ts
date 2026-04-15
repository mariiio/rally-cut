import crypto from 'node:crypto';
import { Prisma } from '@prisma/client';
import { prisma } from '../lib/prisma.js';

/**
 * Atomically record a webhook delivery. Returns true if this is the first
 * time we've seen this deliveryId (caller should process the webhook).
 * Returns false if the deliveryId was already recorded (caller should
 * respond 200 with { deduplicated: true } and skip side effects).
 *
 * Unique-constraint violations on the insert are the success signal for
 * dedup — we don't need a prior SELECT. Other Prisma errors bubble.
 */
export async function tryRecordDelivery(
  deliveryId: string,
  webhookPath: string,
): Promise<boolean> {
  try {
    await prisma.webhookDelivery.create({
      data: { deliveryId, webhookPath },
    });
    return true;
  } catch (err) {
    if (
      err instanceof Prisma.PrismaClientKnownRequestError &&
      err.code === 'P2002'
    ) {
      // Unique constraint violated — we've seen this delivery before.
      return false;
    }
    throw err;
  }
}

function canonicalStringify(v: unknown): string {
  if (v === null || typeof v !== 'object') return JSON.stringify(v);
  if (Array.isArray(v)) return `[${v.map(canonicalStringify).join(',')}]`;
  const keys = Object.keys(v as object).sort();
  return `{${keys
    .map((k) => `${JSON.stringify(k)}:${canonicalStringify((v as Record<string, unknown>)[k])}`)
    .join(',')}}`;
}

/**
 * Derive a deterministic deliveryId from an arbitrary webhook payload
 * when the caller didn't supply one explicitly. Uses SHA-256 over the
 * path + canonical JSON of the body (keys sorted), so identical retries
 * dedup and legitimately distinct deliveries don't collide regardless of
 * key-insertion order in Modal's sender.
 */
export function fingerprintDelivery(
  webhookPath: string,
  body: unknown,
): string {
  const canonical = canonicalStringify(body);
  return crypto
    .createHash('sha256')
    .update(webhookPath)
    .update('\0')
    .update(canonical)
    .digest('hex');
}

const DELIVERY_RETENTION_MS = 7 * 24 * 60 * 60 * 1000;

/**
 * Drop WebhookDelivery rows older than 7 days. Called by the stale-job
 * sweeper so the idempotency table doesn't grow unbounded. The uniqueness
 * window that matters is minutes (Modal retries happen within seconds),
 * so 7 days is generously safe.
 */
export async function pruneOldWebhookDeliveries(): Promise<number> {
  const cutoff = new Date(Date.now() - DELIVERY_RETENTION_MS);
  const { count } = await prisma.webhookDelivery.deleteMany({
    where: { receivedAt: { lt: cutoff } },
  });
  return count;
}

/**
 * Resolve a deliveryId from a webhook payload. If the payload includes
 * a top-level `deliveryId` string, use it. Otherwise, fall back to a
 * content fingerprint. Both yield idempotency; the explicit ID is just
 * better at distinguishing "Modal sent the same result twice" from
 * "two consecutive progress updates that happen to be identical".
 */
export function resolveDeliveryId(
  webhookPath: string,
  body: { deliveryId?: unknown } & Record<string, unknown>,
): string {
  if (typeof body.deliveryId === 'string' && body.deliveryId.length > 0) {
    return body.deliveryId;
  }
  return fingerprintDelivery(webhookPath, body);
}
