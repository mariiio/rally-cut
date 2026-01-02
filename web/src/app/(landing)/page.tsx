'use client';

import { Box } from '@mui/material';
import { Navbar } from './components/Navbar';
import { Hero } from './components/Hero';
import { Demo } from './components/Demo';
import { Features } from './components/Features';
import { HowItWorks } from './components/HowItWorks';
import { Pricing } from './components/Pricing';
import { FAQ } from './components/FAQ';
import { FinalCTA } from './components/FinalCTA';
import { Footer } from './components/Footer';
import { designTokens } from '@/app/designTokens';

export default function LandingPage() {
  return (
    <Box
      sx={{
        minHeight: '100vh',
        bgcolor: designTokens.colors.surface[0],
        color: 'text.primary',
      }}
    >
      <Navbar />
      <Hero />
      <Demo />
      <Features />
      <HowItWorks />
      <Pricing />
      <FAQ />
      <FinalCTA />
      <Footer />
    </Box>
  );
}
