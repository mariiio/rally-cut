'use client';

import {
  Box,
  Container,
  Typography,
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import { designTokens } from '@/app/theme';

const faqs = [
  {
    question: 'What video formats do you support?',
    answer:
      'We support MP4, MOV, AVI, WebM, and most common video formats. For best results, use videos recorded at 720p or higher. Large files are automatically optimized for processing.',
  },
  {
    question: 'How accurate is the AI rally detection?',
    answer:
      'Our AI achieves 95%+ accuracy on beach volleyball footage. It uses a specialized VideoMAE model trained specifically on volleyball content. You can always fine-tune any detection using the timeline editor.',
  },
  {
    question: 'Can I use RallyCut on my phone?',
    answer:
      'Yes! RallyCut works in any modern browser on desktop, tablet, or mobile. Upload from your camera roll and edit on the go. The interface adapts to your screen size.',
  },
  {
    question: 'How long does processing take?',
    answer:
      'Processing time depends on video length. A typical 15-minute match takes about 2-3 minutes to analyze. Pro users get priority processing for faster results during peak times.',
  },
  {
    question: 'Do I need to create an account?',
    answer:
      'No! You can start using RallyCut immediately without an account. Your edits are saved in your browser. Account features are coming soon for cloud sync and sharing across devices.',
  },
  {
    question: 'What about other sports?',
    answer:
      'RallyCut is currently optimized for beach volleyball. Support for indoor volleyball and other racket sports is on our roadmap. Let us know what sports you\'d like to see!',
  },
  {
    question: 'Can I cancel my subscription anytime?',
    answer:
      'Yes, you can cancel anytime. You\'ll keep access until the end of your billing period. No questions asked, no hidden fees.',
  },
  {
    question: 'How do I export my videos?',
    answer:
      'Select the rallies or highlights you want, click Export, and we\'ll generate a video file you can download. Exports include smooth fade transitions between clips. Pro users get high-resolution exports without watermarks.',
  },
];

export function FAQ() {
  return (
    <Box
      component="section"
      id="faq"
      sx={{
        py: { xs: 8, md: 12 },
        bgcolor: designTokens.colors.surface[1],
      }}
    >
      <Container maxWidth="md">
        <Box sx={{ textAlign: 'center', mb: 6 }}>
          <Typography
            variant="h2"
            sx={{
              fontSize: { xs: '2rem', md: '2.5rem' },
              fontWeight: 700,
              mb: 2,
            }}
          >
            Frequently Asked Questions
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Got questions? We&apos;ve got answers.
          </Typography>
        </Box>

        <Box>
          {faqs.map((faq, index) => (
            <Accordion
              key={index}
              elevation={0}
              sx={{
                bgcolor: 'transparent',
                border: '1px solid',
                borderColor: 'divider',
                borderRadius: '12px !important',
                mb: 2,
                '&::before': { display: 'none' },
                '&.Mui-expanded': {
                  borderColor: 'primary.main',
                  bgcolor: designTokens.colors.surface[2],
                },
              }}
            >
              <AccordionSummary
                expandIcon={<ExpandMoreIcon sx={{ color: 'primary.main' }} />}
                sx={{
                  px: 3,
                  py: 1,
                  '& .MuiAccordionSummary-content': {
                    my: 2,
                  },
                }}
              >
                <Typography variant="h6" fontWeight={600} fontSize="1rem">
                  {faq.question}
                </Typography>
              </AccordionSummary>
              <AccordionDetails sx={{ px: 3, pb: 3, pt: 0 }}>
                <Typography variant="body1" color="text.secondary" sx={{ lineHeight: 1.7 }}>
                  {faq.answer}
                </Typography>
              </AccordionDetails>
            </Accordion>
          ))}
        </Box>
      </Container>
    </Box>
  );
}
