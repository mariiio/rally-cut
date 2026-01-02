'use client';

import { Box, Container, Typography, Stack, Divider } from '@mui/material';
import Link from 'next/link';
import { designTokens } from '@/app/designTokens';

// Note: metadata export moved to layout since this is a client component

export default function PrivacyPage() {
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
              Privacy Policy
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
                1. Introduction
              </Typography>
              <Typography variant="body1" color="text.secondary" sx={{ lineHeight: 1.8 }}>
                RallyCut (&quot;we&quot;, &quot;our&quot;, &quot;us&quot;) respects your privacy and is committed to
                protecting your personal data. This privacy policy explains how we collect,
                use, and safeguard your information when you use our video analysis service.
                We comply with GDPR, CCPA, and other applicable privacy regulations.
              </Typography>
            </Box>

            <Box>
              <Typography variant="h5" component="h3">
                2. Information We Collect
              </Typography>
              <Typography variant="body1" color="text.secondary" sx={{ lineHeight: 1.8 }}>
                <strong>Information you provide:</strong>
              </Typography>
              <Box component="ul" sx={{ color: 'text.secondary', pl: 3, mt: 1 }}>
                <li>Email address (when joining waitlist or creating account)</li>
                <li>Videos you upload for analysis</li>
                <li>Edits and highlights you create</li>
              </Box>
              <Typography variant="body1" color="text.secondary" sx={{ lineHeight: 1.8, mt: 2 }}>
                <strong>Information collected automatically:</strong>
              </Typography>
              <Box component="ul" sx={{ color: 'text.secondary', pl: 3, mt: 1 }}>
                <li>Browser type and version</li>
                <li>Device information</li>
                <li>IP address (anonymized for analytics)</li>
                <li>Usage data (features used, time spent)</li>
                <li>Cookies and similar technologies</li>
              </Box>
            </Box>

            <Box>
              <Typography variant="h5" component="h3">
                3. How We Use Your Information
              </Typography>
              <Typography variant="body1" color="text.secondary" sx={{ lineHeight: 1.8 }}>
                We use your information to:
              </Typography>
              <Box component="ul" sx={{ color: 'text.secondary', pl: 3, mt: 1 }}>
                <li>Provide and improve the video analysis service</li>
                <li>Process your videos through our AI models</li>
                <li>Save your edits and preferences</li>
                <li>Send service-related communications</li>
                <li>Analyze usage patterns to improve the product</li>
                <li>Prevent fraud and abuse</li>
                <li>Comply with legal obligations</li>
              </Box>
            </Box>

            <Box>
              <Typography variant="h5" component="h3">
                4. Video Data
              </Typography>
              <Typography variant="body1" color="text.secondary" sx={{ lineHeight: 1.8 }}>
                <strong>Processing:</strong> Videos are uploaded to our secure cloud infrastructure
                (AWS) for AI analysis. We process videos solely to provide rally detection and
                editing features.
              </Typography>
              <Typography variant="body1" color="text.secondary" sx={{ lineHeight: 1.8, mt: 2 }}>
                <strong>Storage:</strong> Videos are stored for 30 days (free tier) or 90 days
                (paid plans) after upload. After this period, videos are automatically deleted.
              </Typography>
              <Typography variant="body1" color="text.secondary" sx={{ lineHeight: 1.8, mt: 2 }}>
                <strong>AI Training:</strong> We do not use your videos to train our AI models
                without your explicit consent. Processing is limited to providing the service.
              </Typography>
            </Box>

            <Box>
              <Typography variant="h5" component="h3">
                5. Cookies and Tracking
              </Typography>
              <Typography variant="body1" color="text.secondary" sx={{ lineHeight: 1.8 }}>
                We use:
              </Typography>
              <Box component="ul" sx={{ color: 'text.secondary', pl: 3, mt: 1 }}>
                <li><strong>Essential cookies:</strong> Required for the service to function</li>
                <li><strong>Analytics cookies:</strong> Help us understand how you use the service (Google Analytics)</li>
                <li><strong>Preference cookies:</strong> Remember your settings</li>
              </Box>
              <Typography variant="body1" color="text.secondary" sx={{ lineHeight: 1.8, mt: 2 }}>
                We do not use advertising cookies or sell your data to advertisers.
              </Typography>
            </Box>

            <Box>
              <Typography variant="h5" component="h3">
                6. Third-Party Services
              </Typography>
              <Typography variant="body1" color="text.secondary" sx={{ lineHeight: 1.8 }}>
                We use trusted third-party services:
              </Typography>
              <Box component="ul" sx={{ color: 'text.secondary', pl: 3, mt: 1 }}>
                <li><strong>AWS:</strong> Cloud hosting and video storage (US)</li>
                <li><strong>Google Analytics:</strong> Usage analytics</li>
                <li><strong>Stripe:</strong> Payment processing (when applicable)</li>
              </Box>
              <Typography variant="body1" color="text.secondary" sx={{ lineHeight: 1.8, mt: 2 }}>
                These services have their own privacy policies and are contractually bound to
                protect your data.
              </Typography>
            </Box>

            <Box>
              <Typography variant="h5" component="h3">
                7. Data Security
              </Typography>
              <Typography variant="body1" color="text.secondary" sx={{ lineHeight: 1.8 }}>
                We implement appropriate security measures:
              </Typography>
              <Box component="ul" sx={{ color: 'text.secondary', pl: 3, mt: 1 }}>
                <li>Encryption in transit (HTTPS/TLS)</li>
                <li>Encryption at rest for stored data</li>
                <li>Access controls and authentication</li>
                <li>Regular security assessments</li>
              </Box>
            </Box>

            <Box>
              <Typography variant="h5" component="h3">
                8. Your Rights (GDPR)
              </Typography>
              <Typography variant="body1" color="text.secondary" sx={{ lineHeight: 1.8 }}>
                If you are in the EU/EEA, you have the right to:
              </Typography>
              <Box component="ul" sx={{ color: 'text.secondary', pl: 3, mt: 1 }}>
                <li><strong>Access:</strong> Request a copy of your personal data</li>
                <li><strong>Rectification:</strong> Correct inaccurate data</li>
                <li><strong>Erasure:</strong> Request deletion of your data (&quot;right to be forgotten&quot;)</li>
                <li><strong>Portability:</strong> Receive your data in a portable format</li>
                <li><strong>Object:</strong> Object to certain processing activities</li>
                <li><strong>Restrict:</strong> Request limited processing</li>
              </Box>
              <Typography variant="body1" color="text.secondary" sx={{ lineHeight: 1.8, mt: 2 }}>
                To exercise these rights, contact us at{' '}
                <Box component="a" href="mailto:hello@rallycut.com" sx={{ color: 'primary.main' }}>
                  hello@rallycut.com
                </Box>
              </Typography>
            </Box>

            <Box>
              <Typography variant="h5" component="h3">
                9. Your Rights (CCPA)
              </Typography>
              <Typography variant="body1" color="text.secondary" sx={{ lineHeight: 1.8 }}>
                If you are a California resident, you have the right to:
              </Typography>
              <Box component="ul" sx={{ color: 'text.secondary', pl: 3, mt: 1 }}>
                <li><strong>Know:</strong> What personal information we collect and how it&apos;s used</li>
                <li><strong>Delete:</strong> Request deletion of your personal information</li>
                <li><strong>Opt-out:</strong> Opt out of the sale of personal information</li>
                <li><strong>Non-discrimination:</strong> Not be discriminated against for exercising rights</li>
              </Box>
              <Typography variant="body1" color="text.secondary" sx={{ lineHeight: 1.8, mt: 2 }}>
                <strong>Note:</strong> We do not sell personal information to third parties.
              </Typography>
            </Box>

            <Box>
              <Typography variant="h5" component="h3">
                10. Children&apos;s Privacy
              </Typography>
              <Typography variant="body1" color="text.secondary" sx={{ lineHeight: 1.8 }}>
                RallyCut is not intended for children under 13 (or 16 in the EU). We do not
                knowingly collect personal information from children. If you believe we have
                collected data from a child, please contact us immediately.
              </Typography>
            </Box>

            <Box>
              <Typography variant="h5" component="h3">
                11. International Transfers
              </Typography>
              <Typography variant="body1" color="text.secondary" sx={{ lineHeight: 1.8 }}>
                Your data may be transferred to and processed in the United States, where our
                servers are located. For EU users, we rely on Standard Contractual Clauses
                (SCCs) to ensure adequate protection of your data during international transfers.
              </Typography>
            </Box>

            <Box>
              <Typography variant="h5" component="h3">
                12. Changes to This Policy
              </Typography>
              <Typography variant="body1" color="text.secondary" sx={{ lineHeight: 1.8 }}>
                We may update this privacy policy from time to time. We&apos;ll notify you of
                significant changes through the Service or by email. The &quot;Last updated&quot; date
                at the top indicates when the policy was last revised.
              </Typography>
            </Box>

            <Box>
              <Typography variant="h5" component="h3">
                13. Contact Us
              </Typography>
              <Typography variant="body1" color="text.secondary" sx={{ lineHeight: 1.8 }}>
                For privacy-related questions or to exercise your rights, contact us at:
              </Typography>
              <Box sx={{ color: 'text.secondary', mt: 1 }}>
                <Typography variant="body1">
                  Email:{' '}
                  <Box component="a" href="mailto:hello@rallycut.com" sx={{ color: 'primary.main' }}>
                    hello@rallycut.com
                  </Box>
                </Typography>
              </Box>
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
