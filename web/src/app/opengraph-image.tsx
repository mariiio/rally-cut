import { ImageResponse } from 'next/og';

export const runtime = 'edge';
export const alt = 'RallyCut - AI-Powered Beach Volleyball Video Analysis';
export const size = {
  width: 1200,
  height: 630,
};
export const contentType = 'image/png';

export default async function Image() {
  return new ImageResponse(
    (
      <div
        style={{
          height: '100%',
          width: '100%',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          backgroundColor: '#0D0E12',
          backgroundImage: 'radial-gradient(ellipse at 50% 0%, rgba(255, 107, 74, 0.15) 0%, transparent 60%)',
        }}
      >
        {/* Logo and Brand */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            marginBottom: 40,
          }}
        >
          <svg
            width="80"
            height="80"
            viewBox="0 0 24 24"
            fill="none"
            style={{ marginRight: 20 }}
          >
            <circle cx="12" cy="12" r="10" stroke="#FF6B4A" strokeWidth="2" fill="none" />
            <path
              d="M12 2C12 2 12 12 12 12M12 12C12 12 2 12 2 12M12 12C12 12 22 12 22 12M12 12C12 12 12 22 12 22"
              stroke="#FF6B4A"
              strokeWidth="1.5"
            />
          </svg>
          <span
            style={{
              fontSize: 72,
              fontWeight: 800,
              background: 'linear-gradient(135deg, #FF6B4A 0%, #FF8A6F 100%)',
              backgroundClip: 'text',
              color: 'transparent',
              letterSpacing: '-0.02em',
            }}
          >
            RallyCut
          </span>
        </div>

        {/* Tagline */}
        <div
          style={{
            fontSize: 36,
            color: '#F5F5F7',
            textAlign: 'center',
            maxWidth: 900,
            lineHeight: 1.3,
            marginBottom: 30,
          }}
        >
          Turn Hours of Footage Into
          <span style={{ color: '#FF6B4A' }}> Minutes of Action</span>
        </div>

        {/* Subtitle */}
        <div
          style={{
            fontSize: 24,
            color: '#A1A7B4',
            textAlign: 'center',
            maxWidth: 700,
          }}
        >
          AI-Powered Beach Volleyball Video Analysis
        </div>

        {/* Bottom accent */}
        <div
          style={{
            position: 'absolute',
            bottom: 0,
            left: 0,
            right: 0,
            height: 6,
            background: 'linear-gradient(90deg, #FF6B4A 0%, #FFD166 50%, #00D4AA 100%)',
          }}
        />
      </div>
    ),
    {
      ...size,
    }
  );
}
