'use client';

import { createTheme, alpha } from '@mui/material/styles';

// Extend MUI breakpoints with custom 'mobile' breakpoint
declare module '@mui/material/styles' {
  interface BreakpointOverrides {
    xs: true;
    mobile: true; // Custom phone-only breakpoint at 640px
    sm: true;
    md: true;
    lg: true;
    xl: true;
  }
}

// Beach volleyball theme - Dark with sunset/ocean accents
export const theme = createTheme({
  palette: {
    mode: 'dark',

    // Primary - Sunset Coral/Orange (energetic, action-oriented)
    primary: {
      main: '#FF6B4A',
      light: '#FF8A6F',
      dark: '#E55235',
      contrastText: '#FFFFFF',
    },

    // Secondary - Ocean Teal/Aqua (calm, highlights)
    secondary: {
      main: '#00D4AA',
      light: '#4DDFBF',
      dark: '#00B896',
      contrastText: '#0D0E12',
    },

    // Backgrounds - Dark with warm undertones
    background: {
      default: '#0D0E12',
      paper: '#151821',
    },

    // Text hierarchy
    text: {
      primary: '#F5F5F7',
      secondary: '#A1A7B4',
      disabled: '#5C6370',
    },

    divider: 'rgba(255, 255, 255, 0.08)',

    // Status colors
    success: {
      main: '#4ADE80',
      light: '#86EFAC',
      dark: '#22C55E',
      contrastText: '#0D0E12',
    },

    error: {
      main: '#EF4444',
      light: '#F87171',
      dark: '#DC2626',
      contrastText: '#FFFFFF',
    },

    warning: {
      main: '#F59E0B',
      light: '#FBBF24',
      dark: '#D97706',
      contrastText: '#0D0E12',
    },

    info: {
      main: '#3B82F6',
      light: '#60A5FA',
      dark: '#2563EB',
      contrastText: '#FFFFFF',
    },

    // Action states
    action: {
      active: 'rgba(255, 255, 255, 0.70)',
      hover: 'rgba(255, 255, 255, 0.08)',
      selected: 'rgba(255, 107, 74, 0.16)',
      disabled: 'rgba(255, 255, 255, 0.26)',
      disabledBackground: 'rgba(255, 255, 255, 0.12)',
      focus: 'rgba(255, 107, 74, 0.24)',
    },
  },

  // Typography - Inter for clean, sporty look
  typography: {
    fontFamily: [
      'Inter',
      '-apple-system',
      'BlinkMacSystemFont',
      'Segoe UI',
      'Roboto',
      'Helvetica Neue',
      'Arial',
      'sans-serif',
    ].join(','),

    h1: {
      fontSize: '2.5rem',
      fontWeight: 700,
      lineHeight: 1.2,
      letterSpacing: '-0.02em',
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 700,
      lineHeight: 1.25,
      letterSpacing: '-0.01em',
    },
    h3: {
      fontSize: '1.5rem',
      fontWeight: 600,
      lineHeight: 1.3,
      letterSpacing: '-0.01em',
    },
    h4: {
      fontSize: '1.25rem',
      fontWeight: 600,
      lineHeight: 1.35,
    },
    h5: {
      fontSize: '1.125rem',
      fontWeight: 600,
      lineHeight: 1.4,
    },
    h6: {
      fontSize: '1rem',
      fontWeight: 600,
      lineHeight: 1.4,
    },
    body1: {
      fontSize: '1rem',
      fontWeight: 400,
      lineHeight: 1.5,
    },
    body2: {
      fontSize: '0.875rem',
      fontWeight: 400,
      lineHeight: 1.5,
    },
    subtitle1: {
      fontSize: '1rem',
      fontWeight: 500,
      lineHeight: 1.5,
      letterSpacing: '0.01em',
    },
    subtitle2: {
      fontSize: '0.875rem',
      fontWeight: 500,
      lineHeight: 1.5,
      letterSpacing: '0.01em',
    },
    caption: {
      fontSize: '0.75rem',
      fontWeight: 400,
      lineHeight: 1.4,
      letterSpacing: '0.02em',
    },
    overline: {
      fontSize: '0.6875rem',
      fontWeight: 600,
      lineHeight: 1.5,
      letterSpacing: '0.08em',
      textTransform: 'uppercase' as const,
    },
    button: {
      fontSize: '0.875rem',
      fontWeight: 600,
      lineHeight: 1.5,
      letterSpacing: '0.02em',
      textTransform: 'none' as const,
    },
  },

  // Shape - Friendly rounded corners
  shape: {
    borderRadius: 8,
  },

  // Responsive breakpoints
  breakpoints: {
    values: {
      xs: 0,
      mobile: 640, // Phone-only breakpoint
      sm: 768,
      md: 1024,
      lg: 1440,
      xl: 1920,
    },
  },

  // Component overrides
  components: {
    // Global baseline styles
    MuiCssBaseline: {
      styleOverrides: {
        body: {
          scrollbarColor: '#464C5B #151821',
          '&::-webkit-scrollbar': {
            width: 8,
            height: 8,
          },
          '&::-webkit-scrollbar-track': {
            background: '#151821',
          },
          '&::-webkit-scrollbar-thumb': {
            background: '#464C5B',
            borderRadius: 4,
            '&:hover': {
              background: '#5C6370',
            },
          },
        },
      },
    },

    // Buttons - Gradient primary, clean secondary
    MuiButton: {
      defaultProps: {
        disableElevation: true,
      },
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 600,
          borderRadius: 6,
          transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
        },
        containedPrimary: {
          background: 'linear-gradient(135deg, #FF6B4A 0%, #FF8A6F 100%)',
          boxShadow: '0 2px 8px rgba(255, 107, 74, 0.3)',
          '&:hover': {
            background: 'linear-gradient(135deg, #FF8A6F 0%, #FFA088 100%)',
            boxShadow: '0 4px 16px rgba(255, 107, 74, 0.4)',
            transform: 'translateY(-1px)',
          },
          '&:active': {
            background: 'linear-gradient(135deg, #E55235 0%, #FF6B4A 100%)',
            transform: 'translateY(0)',
          },
        },
        containedSecondary: {
          background: 'linear-gradient(135deg, #00D4AA 0%, #4DDFBF 100%)',
          boxShadow: '0 2px 8px rgba(0, 212, 170, 0.3)',
          '&:hover': {
            background: 'linear-gradient(135deg, #4DDFBF 0%, #66E5CC 100%)',
            boxShadow: '0 4px 16px rgba(0, 212, 170, 0.4)',
            transform: 'translateY(-1px)',
          },
        },
        outlinedPrimary: {
          borderColor: 'rgba(255, 107, 74, 0.5)',
          color: '#FF6B4A',
          '&:hover': {
            borderColor: '#FF6B4A',
            backgroundColor: 'rgba(255, 107, 74, 0.08)',
          },
        },
        outlinedSecondary: {
          borderColor: 'rgba(0, 212, 170, 0.5)',
          color: '#00D4AA',
          '&:hover': {
            borderColor: '#00D4AA',
            backgroundColor: 'rgba(0, 212, 170, 0.08)',
          },
        },
        text: {
          color: '#A1A7B4',
          '&:hover': {
            backgroundColor: 'rgba(255, 255, 255, 0.08)',
            color: '#F5F5F7',
          },
        },
        sizeSmall: {
          padding: '6px 12px',
          fontSize: '0.8125rem',
        },
        sizeMedium: {
          padding: '8px 16px',
        },
        sizeLarge: {
          padding: '12px 24px',
          fontSize: '1rem',
        },
      },
    },

    // Paper - Subtle borders, no default elevation
    MuiPaper: {
      defaultProps: {
        elevation: 0,
      },
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          backgroundColor: '#151821',
          border: '1px solid rgba(255, 255, 255, 0.06)',
          borderRadius: 8,
        },
        elevation1: {
          backgroundColor: '#1A1E28',
          boxShadow: '0 2px 8px rgba(0, 0, 0, 0.3)',
        },
        elevation2: {
          backgroundColor: '#1F242F',
          boxShadow: '0 4px 16px rgba(0, 0, 0, 0.4)',
        },
        elevation3: {
          backgroundColor: '#252B38',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.5)',
        },
      },
    },

    // AppBar - Gradient header
    MuiAppBar: {
      styleOverrides: {
        root: {
          backgroundColor: '#151821',
          backgroundImage: 'linear-gradient(180deg, #1A1E28 0%, #151821 100%)',
          borderBottom: '1px solid rgba(255, 255, 255, 0.06)',
          boxShadow: '0 2px 8px rgba(0, 0, 0, 0.3)',
        },
      },
    },

    // Toolbar
    MuiToolbar: {
      styleOverrides: {
        root: {
          minHeight: '56px !important',
        },
        dense: {
          minHeight: '48px !important',
        },
      },
    },

    // Chips - Compact with good contrast
    MuiChip: {
      styleOverrides: {
        root: {
          borderRadius: 4,
          fontWeight: 500,
          fontSize: '0.75rem',
          height: 24,
          transition: 'all 0.15s ease',
        },
        filled: {
          backgroundColor: 'rgba(255, 255, 255, 0.08)',
          color: '#A1A7B4',
          '&:hover': {
            backgroundColor: 'rgba(255, 255, 255, 0.12)',
          },
        },
        colorPrimary: {
          backgroundColor: 'rgba(255, 107, 74, 0.16)',
          color: '#FF8A6F',
          '&:hover': {
            backgroundColor: 'rgba(255, 107, 74, 0.24)',
          },
        },
        colorSecondary: {
          backgroundColor: 'rgba(0, 212, 170, 0.16)',
          color: '#4DDFBF',
          '&:hover': {
            backgroundColor: 'rgba(0, 212, 170, 0.24)',
          },
        },
        sizeSmall: {
          height: 18,
          fontSize: '0.6875rem',
        },
      },
    },

    // Icon buttons
    MuiIconButton: {
      styleOverrides: {
        root: {
          borderRadius: 6,
          color: '#A1A7B4',
          transition: 'all 0.2s ease',
          '&:hover': {
            backgroundColor: 'rgba(255, 255, 255, 0.08)',
            color: '#F5F5F7',
          },
          '&:active': {
            backgroundColor: 'rgba(255, 255, 255, 0.12)',
          },
          '&.Mui-disabled': {
            color: '#464C5B',
          },
        },
        colorPrimary: {
          color: '#FF6B4A',
          '&:hover': {
            backgroundColor: 'rgba(255, 107, 74, 0.12)',
            color: '#FF8A6F',
          },
        },
        colorSecondary: {
          color: '#00D4AA',
          '&:hover': {
            backgroundColor: 'rgba(0, 212, 170, 0.12)',
            color: '#4DDFBF',
          },
        },
        sizeSmall: {
          padding: 4,
        },
        sizeMedium: {
          padding: 8,
        },
        sizeLarge: {
          padding: 12,
        },
      },
    },

    // Tooltips
    MuiTooltip: {
      styleOverrides: {
        tooltip: {
          backgroundColor: '#252B38',
          color: '#F5F5F7',
          fontSize: '0.75rem',
          fontWeight: 500,
          padding: '6px 12px',
          borderRadius: 6,
          boxShadow: '0 4px 16px rgba(0, 0, 0, 0.4)',
          border: '1px solid rgba(255, 255, 255, 0.08)',
        },
        arrow: {
          color: '#252B38',
        },
      },
    },

    // Popovers
    MuiPopover: {
      styleOverrides: {
        paper: {
          backgroundColor: '#1F242F',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.5)',
          borderRadius: 8,
        },
      },
    },

    // Dialogs
    MuiDialog: {
      styleOverrides: {
        paper: {
          backgroundColor: '#1A1E28',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          boxShadow: '0 16px 48px rgba(0, 0, 0, 0.6)',
          borderRadius: 12,
        },
      },
    },

    MuiDialogTitle: {
      styleOverrides: {
        root: {
          fontSize: '1.125rem',
          fontWeight: 600,
        },
      },
    },

    // Text fields
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            borderRadius: 6,
            '& fieldset': {
              borderColor: 'rgba(255, 255, 255, 0.12)',
            },
            '&:hover fieldset': {
              borderColor: 'rgba(255, 255, 255, 0.24)',
            },
            '&.Mui-focused fieldset': {
              borderColor: '#FF6B4A',
              boxShadow: '0 0 0 3px rgba(255, 107, 74, 0.15)',
            },
          },
        },
      },
    },

    MuiOutlinedInput: {
      styleOverrides: {
        root: {
          borderRadius: 6,
        },
      },
    },

    // Switches
    MuiSwitch: {
      styleOverrides: {
        root: {
          padding: 8,
        },
        track: {
          borderRadius: 10,
          backgroundColor: 'rgba(255, 255, 255, 0.2)',
        },
        thumb: {
          boxShadow: '0 2px 4px rgba(0, 0, 0, 0.3)',
        },
        switchBase: {
          '&.Mui-checked': {
            color: '#FFFFFF',
            '& + .MuiSwitch-track': {
              backgroundColor: '#FF6B4A',
              opacity: 1,
            },
          },
        },
      },
    },

    // Sliders
    MuiSlider: {
      styleOverrides: {
        root: {
          color: '#FF6B4A',
        },
        thumb: {
          width: 14,
          height: 14,
          boxShadow: '0 2px 4px rgba(0, 0, 0, 0.3)',
          '&:hover': {
            boxShadow: '0 0 0 8px rgba(255, 107, 74, 0.16)',
          },
          '&.Mui-active': {
            boxShadow: '0 0 0 12px rgba(255, 107, 74, 0.24)',
          },
        },
        track: {
          height: 4,
          borderRadius: 2,
        },
        rail: {
          height: 4,
          borderRadius: 2,
          backgroundColor: 'rgba(255, 255, 255, 0.12)',
        },
      },
    },

    // Dividers
    MuiDivider: {
      styleOverrides: {
        root: {
          borderColor: 'rgba(255, 255, 255, 0.08)',
        },
      },
    },

    // Lists
    MuiListItem: {
      styleOverrides: {
        root: {
          borderRadius: 6,
          '&:hover': {
            backgroundColor: 'rgba(255, 255, 255, 0.04)',
          },
          '&.Mui-selected': {
            backgroundColor: 'rgba(255, 107, 74, 0.12)',
            '&:hover': {
              backgroundColor: 'rgba(255, 107, 74, 0.16)',
            },
          },
        },
      },
    },

    MuiListItemButton: {
      styleOverrides: {
        root: {
          borderRadius: 6,
          '&:hover': {
            backgroundColor: 'rgba(255, 255, 255, 0.04)',
          },
          '&.Mui-selected': {
            backgroundColor: 'rgba(255, 107, 74, 0.12)',
            '&:hover': {
              backgroundColor: 'rgba(255, 107, 74, 0.16)',
            },
          },
        },
      },
    },

    // Collapse animation
    MuiCollapse: {
      styleOverrides: {
        root: {
          transition: 'height 200ms cubic-bezier(0.4, 0, 0.2, 1)',
        },
      },
    },

    // Badge
    MuiBadge: {
      styleOverrides: {
        badge: {
          fontSize: '0.625rem',
          fontWeight: 600,
          minWidth: 16,
          height: 16,
          borderRadius: 8,
          backgroundColor: '#FF6B4A',
          color: '#FFFFFF',
        },
      },
    },

    // Snackbar
    MuiSnackbar: {
      styleOverrides: {
        root: {
          '& .MuiPaper-root': {
            borderRadius: 8,
          },
        },
      },
    },

    // Alert
    MuiAlert: {
      styleOverrides: {
        root: {
          borderRadius: 8,
        },
        standardSuccess: {
          backgroundColor: 'rgba(74, 222, 128, 0.12)',
          color: '#4ADE80',
        },
        standardError: {
          backgroundColor: 'rgba(239, 68, 68, 0.12)',
          color: '#EF4444',
        },
        standardWarning: {
          backgroundColor: 'rgba(245, 158, 11, 0.12)',
          color: '#F59E0B',
        },
        standardInfo: {
          backgroundColor: 'rgba(59, 130, 246, 0.12)',
          color: '#3B82F6',
        },
      },
    },

    // Tab styling
    MuiTabs: {
      styleOverrides: {
        indicator: {
          backgroundColor: '#FF6B4A',
          height: 2,
        },
      },
    },

    MuiTab: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 500,
          fontSize: '0.875rem',
          minHeight: 40,
          color: '#A1A7B4',
          '&.Mui-selected': {
            color: '#F5F5F7',
          },
          '&:hover': {
            color: '#F5F5F7',
          },
        },
      },
    },

    // Select
    MuiSelect: {
      styleOverrides: {
        select: {
          borderRadius: 6,
        },
      },
    },

    // Menu
    MuiMenu: {
      styleOverrides: {
        paper: {
          backgroundColor: '#1F242F',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.5)',
          borderRadius: 8,
        },
      },
    },

    MuiMenuItem: {
      styleOverrides: {
        root: {
          borderRadius: 4,
          margin: '2px 4px',
          '&:hover': {
            backgroundColor: 'rgba(255, 255, 255, 0.06)',
          },
          '&.Mui-selected': {
            backgroundColor: 'rgba(255, 107, 74, 0.12)',
            '&:hover': {
              backgroundColor: 'rgba(255, 107, 74, 0.16)',
            },
          },
        },
      },
    },

    // Form control label
    MuiFormControlLabel: {
      styleOverrides: {
        label: {
          fontSize: '0.875rem',
        },
      },
    },

    // Circular progress
    MuiCircularProgress: {
      styleOverrides: {
        colorPrimary: {
          color: '#FF6B4A',
        },
        colorSecondary: {
          color: '#00D4AA',
        },
      },
    },

    // Linear progress
    MuiLinearProgress: {
      styleOverrides: {
        root: {
          borderRadius: 4,
          backgroundColor: 'rgba(255, 255, 255, 0.12)',
        },
        barColorPrimary: {
          backgroundColor: '#FF6B4A',
        },
        barColorSecondary: {
          backgroundColor: '#00D4AA',
        },
      },
    },
  },
});

// Custom design tokens for use in components
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
        left: 280,
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
