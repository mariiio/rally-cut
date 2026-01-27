import { NextResponse } from 'next/server';
import { SignJWT } from 'jose';
import { auth } from '@/lib/auth';

/**
 * Returns a standard signed JWT for the authenticated user.
 * The frontend fetches this and sends it to the Express API
 * as Authorization: Bearer <token>.
 *
 * Uses jose (JWS) instead of Auth.js encode (JWE) so the
 * Express API can verify with standard jsonwebtoken library.
 */
export async function GET() {
  const session = await auth();

  if (!session?.user?.id) {
    return NextResponse.json({ error: 'Not authenticated' }, { status: 401 });
  }

  const secret = new TextEncoder().encode(process.env.AUTH_SECRET!);

  const token = await new SignJWT({
    sub: session.user.id,
    email: session.user.email ?? undefined,
    name: session.user.name ?? undefined,
  })
    .setProtectedHeader({ alg: 'HS256' })
    .setIssuedAt()
    .setExpirationTime('30d')
    .sign(secret);

  return NextResponse.json({ token });
}
