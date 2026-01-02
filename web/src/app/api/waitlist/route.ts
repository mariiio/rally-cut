import { NextRequest, NextResponse } from 'next/server';

// Simple in-memory store for development
// In production, this should be stored in a database
const waitlist: Array<{ email: string; tier: string; createdAt: Date }> = [];

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { email, tier } = body;

    // Validate email
    if (!email || typeof email !== 'string') {
      return NextResponse.json(
        { message: 'Email is required' },
        { status: 400 }
      );
    }

    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      return NextResponse.json(
        { message: 'Invalid email format' },
        { status: 400 }
      );
    }

    // Validate tier
    const validTiers = ['pro', 'unlimited'];
    if (!tier || !validTiers.includes(tier)) {
      return NextResponse.json(
        { message: 'Invalid tier' },
        { status: 400 }
      );
    }

    // Check for duplicate
    const existing = waitlist.find((entry) => entry.email.toLowerCase() === email.toLowerCase());
    if (existing) {
      // Update tier if different
      existing.tier = tier;
      return NextResponse.json(
        { success: true, message: "You're already on the list!" },
        { status: 200 }
      );
    }

    // Add to waitlist
    waitlist.push({
      email: email.toLowerCase(),
      tier,
      createdAt: new Date(),
    });

    console.log(`[Waitlist] New signup: ${email} for ${tier} tier`);
    console.log(`[Waitlist] Total signups: ${waitlist.length}`);

    return NextResponse.json(
      { success: true, message: "You're on the list!" },
      { status: 201 }
    );
  } catch (error) {
    console.error('[Waitlist] Error:', error);
    return NextResponse.json(
      { message: 'Something went wrong' },
      { status: 500 }
    );
  }
}

export async function GET() {
  // Simple endpoint to check waitlist count (for admin)
  return NextResponse.json({
    count: waitlist.length,
    entries: waitlist.map((e) => ({ email: e.email, tier: e.tier, createdAt: e.createdAt })),
  });
}
