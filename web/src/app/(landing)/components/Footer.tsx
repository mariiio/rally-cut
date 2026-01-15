'use client';

import { Box, Container, Typography, Grid, Stack, IconButton, Divider } from '@mui/material';
import SportsVolleyballIcon from '@mui/icons-material/SportsVolleyball';
import TwitterIcon from '@mui/icons-material/Twitter';
import InstagramIcon from '@mui/icons-material/Instagram';
import YouTubeIcon from '@mui/icons-material/YouTube';
import Link from 'next/link';
import { designTokens } from '@/app/theme';

const footerLinks = {
  product: [
    { label: 'Features', href: '#features' },
    { label: 'Pricing', href: '#pricing' },
    { label: 'FAQ', href: '#faq' },
  ],
  legal: [
    { label: 'Terms of Service', href: '/terms' },
    { label: 'Privacy Policy', href: '/privacy' },
  ],
  company: [
    { label: 'Contact', href: 'mailto:hello@rallycut.com' },
  ],
};

export function Footer() {
  const scrollToSection = (href: string) => {
    if (href.startsWith('#')) {
      const element = document.querySelector(href);
      if (element) {
        element.scrollIntoView({ behavior: 'smooth' });
      }
    }
  };

  return (
    <Box
      component="footer"
      sx={{
        py: 6,
        bgcolor: designTokens.colors.surface[0],
        borderTop: '1px solid',
        borderColor: 'divider',
      }}
    >
      <Container maxWidth="lg">
        <Grid container spacing={4}>
          {/* Brand */}
          <Grid size={{ xs: 12, md: 4 }}>
            <Stack spacing={2}>
              <Stack direction="row" alignItems="center" spacing={1.5}>
                <SportsVolleyballIcon
                  sx={{
                    fontSize: 28,
                    color: 'primary.main',
                  }}
                />
                <Typography
                  variant="h6"
                  sx={{
                    fontWeight: 700,
                    background: designTokens.gradients.primary,
                    backgroundClip: 'text',
                    WebkitBackgroundClip: 'text',
                    WebkitTextFillColor: 'transparent',
                  }}
                >
                  RallyCut
                </Typography>
              </Stack>
              <Typography variant="body2" color="text.secondary" sx={{ maxWidth: 280 }}>
                AI-powered video analysis for volleyball. Turn hours of footage into minutes of
                highlights.
              </Typography>
              {/* Social Icons */}
              <Stack direction="row" spacing={1}>
                <IconButton
                  size="small"
                  sx={{
                    color: 'text.secondary',
                    '&:hover': { color: 'primary.main' },
                  }}
                >
                  <TwitterIcon fontSize="small" />
                </IconButton>
                <IconButton
                  size="small"
                  sx={{
                    color: 'text.secondary',
                    '&:hover': { color: 'primary.main' },
                  }}
                >
                  <InstagramIcon fontSize="small" />
                </IconButton>
                <IconButton
                  size="small"
                  sx={{
                    color: 'text.secondary',
                    '&:hover': { color: 'primary.main' },
                  }}
                >
                  <YouTubeIcon fontSize="small" />
                </IconButton>
              </Stack>
            </Stack>
          </Grid>

          {/* Product Links */}
          <Grid size={{ xs: 6, sm: 4, md: 2 }}>
            <Typography variant="subtitle2" fontWeight={600} sx={{ mb: 2 }}>
              Product
            </Typography>
            <Stack spacing={1.5}>
              {footerLinks.product.map((link) => (
                <Typography
                  key={link.label}
                  component={link.href.startsWith('#') ? 'span' : Link}
                  href={link.href.startsWith('#') ? undefined : link.href}
                  onClick={
                    link.href.startsWith('#') ? () => scrollToSection(link.href) : undefined
                  }
                  variant="body2"
                  sx={{
                    color: 'text.secondary',
                    textDecoration: 'none',
                    cursor: 'pointer',
                    '&:hover': { color: 'primary.main' },
                  }}
                >
                  {link.label}
                </Typography>
              ))}
            </Stack>
          </Grid>

          {/* Legal Links */}
          <Grid size={{ xs: 6, sm: 4, md: 2 }}>
            <Typography variant="subtitle2" fontWeight={600} sx={{ mb: 2 }}>
              Legal
            </Typography>
            <Stack spacing={1.5}>
              {footerLinks.legal.map((link) => (
                <Typography
                  key={link.label}
                  component={Link}
                  href={link.href}
                  variant="body2"
                  sx={{
                    color: 'text.secondary',
                    textDecoration: 'none',
                    '&:hover': { color: 'primary.main' },
                  }}
                >
                  {link.label}
                </Typography>
              ))}
            </Stack>
          </Grid>

          {/* Company Links */}
          <Grid size={{ xs: 6, sm: 4, md: 2 }}>
            <Typography variant="subtitle2" fontWeight={600} sx={{ mb: 2 }}>
              Company
            </Typography>
            <Stack spacing={1.5}>
              {footerLinks.company.map((link) => (
                <Typography
                  key={link.label}
                  component={Link}
                  href={link.href}
                  variant="body2"
                  sx={{
                    color: 'text.secondary',
                    textDecoration: 'none',
                    '&:hover': { color: 'primary.main' },
                  }}
                >
                  {link.label}
                </Typography>
              ))}
            </Stack>
          </Grid>
        </Grid>

        <Divider sx={{ my: 4 }} />

        {/* Bottom */}
        <Stack
          direction={{ xs: 'column', sm: 'row' }}
          justifyContent="space-between"
          alignItems="center"
          spacing={2}
        >
          <Typography variant="body2" color="text.disabled">
            © {new Date().getFullYear()} RallyCut. All rights reserved.
          </Typography>
          <Typography variant="body2" color="text.disabled">
            Made with ❤️ for volleyball players
          </Typography>
        </Stack>
      </Container>
    </Box>
  );
}
