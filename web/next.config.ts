import type { NextConfig } from "next";

const isProd = process.env.NODE_ENV === 'production';
const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001';

const nextConfig: NextConfig = {
  // Required for FFmpeg.wasm SharedArrayBuffer support
  // Using 'credentialless' for COEP allows loading cross-origin resources (CloudFront videos)
  // while still enabling SharedArrayBuffer
  async headers() {
    return [
      {
        source: '/:path*',
        headers: [
          {
            key: 'Cross-Origin-Opener-Policy',
            value: 'same-origin',
          },
          {
            key: 'Cross-Origin-Embedder-Policy',
            // 'credentialless' allows cross-origin resources without CORS headers
            // while still enabling SharedArrayBuffer (needed for FFmpeg.wasm)
            value: isProd ? 'require-corp' : 'credentialless',
          },
        ],
      },
    ];
  },

  // Proxy video requests to backend API in local development
  // In production, CloudFront serves videos directly
  async rewrites() {
    // Skip rewrites in production (CloudFront handles videos)
    if (isProd) {
      return [];
    }

    return [
      {
        source: '/videos/:path*',
        destination: `${apiUrl}/videos/:path*`,
      },
      {
        source: '/confirmations/:path*',
        destination: `${apiUrl}/confirmations/:path*`,
      },
    ];
  },

  // Empty turbopack config to acknowledge we're using Turbopack
  turbopack: {},
};

export default nextConfig;
