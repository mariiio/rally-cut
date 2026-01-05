// Design tokens for use in both server and client components
// This file is separate from theme.ts to avoid 'use client' restriction

export const designTokens = {
  colors: {
    // Tertiary - Sandy gold for accents
    tertiary: {
      main: '#FFD166',
      light: '#FFDE8A',
      dark: '#F5BC3C',
    },
    // Extended surface levels
    surface: {
      0: '#0D0E12',
      1: '#151821',
      2: '#1A1E28',
      3: '#1F242F',
      4: '#252B38',
    },
    // Timeline-specific colors
    timeline: {
      background: '#0F1116',
      rallyDefault: 'linear-gradient(180deg, #3B82F6 0%, #2563EB 100%)',
      rallyHover: 'linear-gradient(180deg, #60A5FA 0%, #3B82F6 100%)',
      rallySelected: 'linear-gradient(180deg, #FF6B4A 0%, #E55235 100%)',
      cursor: '#FF6B4A',
      cursorGlow: '0 0 8px rgba(255, 107, 74, 0.6)',
    },
    // Video area
    video: {
      background: '#0A0A0A',
      shadow: '0 8px 32px rgba(0, 0, 0, 0.4)',
    },
  },
  shadows: {
    sm: '0 1px 2px rgba(0, 0, 0, 0.2)',
    md: '0 2px 8px rgba(0, 0, 0, 0.3)',
    lg: '0 4px 16px rgba(0, 0, 0, 0.4)',
    xl: '0 8px 32px rgba(0, 0, 0, 0.5)',
    glow: {
      primary: '0 0 20px rgba(255, 107, 74, 0.4)',
      secondary: '0 0 20px rgba(0, 212, 170, 0.4)',
    },
    focus: '0 0 0 3px rgba(255, 107, 74, 0.3)',
  },
  gradients: {
    primary: 'linear-gradient(135deg, #FF6B4A 0%, #FF8A6F 100%)',
    secondary: 'linear-gradient(135deg, #00D4AA 0%, #4DDFBF 100%)',
    tertiary: 'linear-gradient(135deg, #FFD166 0%, #FFA94D 100%)',
    sunset: 'linear-gradient(135deg, #FF6B4A 0%, #FFD166 50%, #00D4AA 100%)',
    toolbar: 'linear-gradient(180deg, #1A1E28 0%, #151821 100%)',
  },
  transitions: {
    fast: '150ms cubic-bezier(0.4, 0, 0.2, 1)',
    normal: '200ms cubic-bezier(0.4, 0, 0.2, 1)',
    slow: '300ms cubic-bezier(0.4, 0, 0.2, 1)',
  },
  spacing: {
    panel: {
      expanded: {
        left: 340,
        right: 340,
      },
      collapsed: 48,
    },
    header: 56,
    timeline: {
      normal: 240,
      expanded: 320,
    },
  },
  // Mobile-specific design tokens
  mobile: {
    breakpoint: 640,
    touchTarget: 44, // Minimum touch target size (Apple HIG)
    bottomNav: {
      height: 56,
      safeAreaPadding: 'env(safe-area-inset-bottom)',
    },
    header: {
      height: 56,
    },
    rallyItem: {
      minHeight: 52,
    },
    miniTimeline: {
      height: 80,
      handleSize: 44,
      windowDuration: 30, // seconds
    },
  },
};
