import type { Metadata } from 'next';
import Script from 'next/script';
import { AppRouterCacheProvider } from '@mui/material-nextjs/v15-appRouter';
import { ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { theme } from './theme';
import './globals.css';

const GA_MEASUREMENT_ID = 'G-XPC77Q8RWV';

export const metadata: Metadata = {
  metadataBase: new URL('https://rallycut.com'),
  title: {
    default: 'RallyCut - AI-Powered Beach Volleyball Video Analysis',
    template: '%s | RallyCut',
  },
  description:
    'Turn hours of beach volleyball footage into minutes of action. RallyCut uses AI to automatically detect rallies and create highlight reels. No editing skills required.',
  keywords: [
    'beach volleyball',
    'video analysis',
    'rally detection',
    'highlights',
    'AI',
    'video editor',
    'volleyball highlights',
    'sports video',
    'automatic editing',
  ],
  authors: [{ name: 'RallyCut' }],
  creator: 'RallyCut',
  publisher: 'RallyCut',
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      'max-video-preview': -1,
      'max-image-preview': 'large',
      'max-snippet': -1,
    },
  },
  openGraph: {
    title: 'RallyCut - AI-Powered Beach Volleyball Video Analysis',
    description: 'Turn hours of footage into minutes of action with AI-powered rally detection.',
    url: 'https://rallycut.com',
    siteName: 'RallyCut',
    type: 'website',
    locale: 'en_US',
    images: [
      {
        url: '/og-image.png',
        width: 1200,
        height: 630,
        alt: 'RallyCut - AI-Powered Beach Volleyball Video Analysis',
      },
    ],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'RallyCut - AI Beach Volleyball Video Analysis',
    description: 'Turn hours of footage into minutes of action with AI-powered rally detection.',
    images: ['/og-image.png'],
  },
  icons: {
    icon: '/favicon.ico',
    apple: '/apple-touch-icon.png',
  },
  manifest: '/manifest.json',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        {/* Google Analytics */}
        <Script
          src={`https://www.googletagmanager.com/gtag/js?id=${GA_MEASUREMENT_ID}`}
          strategy="afterInteractive"
        />
        <Script id="google-analytics" strategy="afterInteractive">
          {`
            window.dataLayer = window.dataLayer || [];
            function gtag(){dataLayer.push(arguments);}
            gtag('js', new Date());
            gtag('config', '${GA_MEASUREMENT_ID}');
          `}
        </Script>
      </head>
      <body>
        <AppRouterCacheProvider>
          <ThemeProvider theme={theme}>
            <CssBaseline />
            {children}
          </ThemeProvider>
        </AppRouterCacheProvider>
      </body>
    </html>
  );
}
