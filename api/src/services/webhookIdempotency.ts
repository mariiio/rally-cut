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

/**
 * Derive a deterministic deliveryId from an arbitrary webhook payload
 * when the caller didn't supply one explicitly. Uses SHA-256 over the
 * path + canonical JSON of the body, so identical retries dedup and
 * legitimately distinct deliveries don't collide.
 */
export function fingerprintDelivery(
  webhookPath: string,
  body: unknown,
): string {
  const canonical = JSON.stringify(body);
  return crypto
    .createHash('sha256')
    .update(webhookPath)
    .update('\0')
    .update(canonical)
    .digest('hex');
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
