'use client';

import { Box, Container, Typography, Stack, Divider } from '@mui/material';
import Link from 'next/link';
import { designTokens } from '@/app/designTokens';

// Note: metadata export moved to layout since this is a client component

export default function TermsPage() {
  return (
    <Box
      sx={{
        minHeight: '100vh',
        bgcolor: designTokens.colors.surface[0],
        color: 'text.primary',
        py: 8,
      }}
    >
      <Container maxWidth="md">
        <Stack spacing={4}>
          {/* Header */}
          <Box>
            <Typography
              component={Link}
              href="/"
              sx={{
                color: 'primary.main',
                textDecoration: 'none',
                fontSize: '0.9rem',
                '&:hover': { textDecoration: 'underline' },
              }}
            >
              ← Back to RallyCut
            </Typography>
            <Typography variant="h2" sx={{ fontWeight: 700, mt: 3, mb: 1 }}>
              Terms of Service
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Last updated: January 2, 2025
            </Typography>
          </Box>

          <Divider />

          {/* Content */}
          <Stack spacing={4} sx={{ '& h3': { fontWeight: 600, mb: 1 } }}>
            <Box>
              <Typography variant="h5" component="h3">
                1. Acceptance of Terms
              </Typography>
              <Typography variant="body1" color="text.secondary" sx={{ lineHeight: 1.8 }}>
                By accessing or using RallyCut (&quot;the Service&quot;), you agree to be bound by these
                Terms of Service. If you do not agree to these terms, please do not use the
                Service. You must be at least 13 years old (or 16 in the EU) to use RallyCut.
              </Typography>
            </Box>

            <Box>
              <Typography variant="h5" component="h3">
                2. Description of Service
              </Typography>
              <Typography variant="body1" color="text.secondary" sx={{ lineHeight: 1.8 }}>
                RallyCut is an AI-powered video analysis platform for volleyball. The
                Service allows users to upload videos, automatically detect rallies using machine
                learning, edit video segments, and export highlight reels. The Service operates
                in your web browser and may use cloud processing for video analysis.
              </Typography>
            </Box>

            <Box>
              <Typography variant="h5" component="h3">
                3. User Responsibilities
              </Typography>
              <Typography variant="body1" color="text.secondary" sx={{ lineHeight: 1.8 }}>
                You agree to:
              </Typography>
              <Box component="ul" sx={{ color: 'text.secondary', pl: 3, mt: 1 }}>
                <li>Only upload videos you have the right to use and share</li>
                <li>Not upload content that is illegal, harmful, or violates others&apos; rights</li>
                <li>Not attempt to reverse engineer, hack, or abuse the Service</li>
                <li>Not use the Service for any commercial purpose without authorization</li>
                <li>Provide accurate information when creating an account (if applicable)</li>
              </Box>
            </Box>

            <Box>
              <Typography variant="h5" component="h3">
                4. Intellectual Property
              </Typography>
              <Typography variant="body1" color="text.secondary" sx={{ lineHeight: 1.8 }}>
                <strong>Your Content:</strong> You retain all ownership rights to videos you
                upload. By using the Service, you grant us a limited license to process your
                videos for the purpose of providing the Service.
              </Typography>
              <Typography variant="body1" color="text.secondary" sx={{ lineHeight: 1.8, mt: 2 }}>
                <strong>Our Property:</strong> The RallyCut platform, including its AI models,
                software, design, and branding, are owned by RallyCut. You may not copy,
                modify, or distribute our intellectual property without permission.
              </Typography>
            </Box>

            <Box>
              <Typography variant="h5" component="h3">
                5. Payment Terms
              </Typography>
              <Typography variant="body1" color="text.secondary" sx={{ lineHeight: 1.8 }}>
                <strong>Free Tier:</strong> Basic features are available at no cost with usage
                limitations as described on our pricing page.
              </Typography>
              <Typography variant="body1" color="text.secondary" sx={{ lineHeight: 1.8, mt: 2 }}>
                <strong>Paid Plans:</strong> Subscription fees are billed in advance on a
                monthly or annual basis. You may cancel at any time, and you&apos;ll retain access
                until the end of your billing period. Refunds are provided at our discretion.
              </Typography>
              <Typography variant="body1" color="text.secondary" sx={{ lineHeight: 1.8, mt: 2 }}>
                <strong>Price Changes:</strong> We may change our prices with 30 days notice.
                Existing subscriptions will honor the original price until renewal.
              </Typography>
            </Box>

            <Box>
              <Typography variant="h5" component="h3">
                6. Disclaimers
              </Typography>
              <Typography variant="body1" color="text.secondary" sx={{ lineHeight: 1.8 }}>
                The Service is provided &quot;as is&quot; without warranties of any kind. We do not
                guarantee:
              </Typography>
              <Box component="ul" sx={{ color: 'text.secondary', pl: 3, mt: 1 }}>
                <li>100% accuracy of AI rally detection</li>
                <li>Uninterrupted or error-free service</li>
                <li>That the Service will meet all your requirements</li>
                <li>The security of data stored in your browser</li>
              </Box>
            </Box>

            <Box>
              <Typography variant="h5" component="h3">
                7. Limitation of Liability
              </Typography>
              <Typography variant="body1" color="text.secondary" sx={{ lineHeight: 1.8 }}>
                To the maximum extent permitted by law, RallyCut shall not be liable for any
                indirect, incidental, special, consequential, or punitive damages, including
                loss of data, revenue, or profits. Our total liability shall not exceed the
                amount you paid us in the 12 months preceding the claim.
              </Typography>
            </Box>

            <Box>
              <Typography variant="h5" component="h3">
                8. Termination
              </Typography>
              <Typography variant="body1" color="text.secondary" sx={{ lineHeight: 1.8 }}>
                We may suspend or terminate your access to the Service at any time for
                violation of these terms or for any other reason with reasonable notice. You
                may stop using the Service at any time. Upon termination, your right to use
                the Service ceases, and we may delete your data after a reasonable period.
              </Typography>
            </Box>

            <Box>
              <Typography variant="h5" component="h3">
                9. Changes to Terms
              </Typography>
              <Typography variant="body1" color="text.secondary" sx={{ lineHeight: 1.8 }}>
                We may modify these terms at any time. We&apos;ll notify you of material changes
                through the Service or by email. Continued use after changes constitutes
                acceptance of the new terms.
              </Typography>
            </Box>

            <Box>
              <Typography variant="h5" component="h3">
                10. Governing Law
              </Typography>
              <Typography variant="body1" color="text.secondary" sx={{ lineHeight: 1.8 }}>
                These terms are governed by the laws of Delaware, USA, without regard to
                conflict of law principles. Any disputes shall be resolved in the courts of
                Delaware. For users in the EU, this does not affect your statutory rights
                under local consumer protection laws.
              </Typography>
            </Box>

            <Box>
              <Typography variant="h5" component="h3">
                11. Contact
              </Typography>
              <Typography variant="body1" color="text.secondary" sx={{ lineHeight: 1.8 }}>
                For questions about these terms, contact us at:{' '}
                <Box
                  component="a"
                  href="mailto:hello@rallycut.com"
                  sx={{ color: 'primary.main' }}
                >
                  hello@rallycut.com
                </Box>
              </Typography>
            </Box>
          </Stack>

          <Divider />

          <Typography variant="body2" color="text.disabled" sx={{ textAlign: 'center' }}>
            © {new Date().getFullYear()} RallyCut. All rights reserved.
          </Typography>
        </Stack>
      </Container>
    </Box>
  );
}
