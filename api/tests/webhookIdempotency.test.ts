import 'dotenv/config';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { prisma } from '../src/lib/prisma';
import {
  fingerprintDelivery,
  resolveDeliveryId,
  tryRecordDelivery,
} from '../src/services/webhookIdempotency';

describe('webhookIdempotency helper', () => {
  const deliveryIdUsed: string[] = [];

  afterEach(async () => {
    if (deliveryIdUsed.length > 0) {
      await prisma.webhookDelivery.deleteMany({
        where: { deliveryId: { in: deliveryIdUsed } },
      });
      deliveryIdUsed.length = 0;
    }
  });

  it('tryRecordDelivery returns true on first call, false on duplicate', async () => {
    const id = `test-${crypto.randomUUID()}`;
    deliveryIdUsed.push(id);
    expect(await tryRecordDelivery(id, '/v1/webhooks/foo')).toBe(true);
    expect(await tryRecordDelivery(id, '/v1/webhooks/foo')).toBe(false);
  });

  it('fingerprintDelivery is deterministic for identical (path, body) pairs', () => {
    const a = fingerprintDelivery('/p', { x: 1, y: 'two' });
    const b = fingerprintDelivery('/p', { x: 1, y: 'two' });
    expect(a).toBe(b);
  });

  it('fingerprintDelivery diverges when body differs', () => {
    const a = fingerprintDelivery('/p', { x: 1 });
    const b = fingerprintDelivery('/p', { x: 2 });
    expect(a).not.toBe(b);
  });

  it('fingerprintDelivery diverges when webhookPath differs', () => {
    const body = { x: 1 };
    const a = fingerprintDelivery('/a', body);
    const b = fingerprintDelivery('/b', body);
    expect(a).not.toBe(b);
  });

  it('resolveDeliveryId prefers explicit deliveryId over fingerprint', () => {
    const explicit = resolveDeliveryId('/p', { deliveryId: 'explicit-id', x: 1 });
    expect(explicit).toBe('explicit-id');
  });

  it('resolveDeliveryId falls back to fingerprint when deliveryId is absent', () => {
    const body = { x: 1 };
    const id = resolveDeliveryId('/p', body);
    expect(id).toBe(fingerprintDelivery('/p', body));
  });
});
