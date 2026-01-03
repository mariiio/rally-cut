'use client';

import { Box } from '@mui/material';
import { motion, useScroll, useTransform, useReducedMotion } from 'framer-motion';
import { useRef } from 'react';

// Floating orb component
function FloatingOrb({
  color,
  size,
  top,
  left,
  delay = 0,
}: {
  color: string;
  size: number;
  top: string;
  left: string;
  delay?: number;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{
        opacity: [0.3, 0.5, 0.3],
        scale: [1, 1.1, 1],
        x: [0, 20, -10, 0],
        y: [0, -15, 10, 0],
      }}
      transition={{
        duration: 8,
        delay,
        repeat: Infinity,
        ease: 'easeInOut',
      }}
      style={{
        position: 'absolute',
        top,
        left,
        width: size,
        height: size,
        borderRadius: '50%',
        background: `radial-gradient(circle, ${color} 0%, transparent 70%)`,
        filter: 'blur(40px)',
        pointerEvents: 'none',
      }}
    />
  );
}

// Animated grid pattern
function GridPattern() {
  return (
    <Box
      sx={{
        position: 'absolute',
        inset: 0,
        backgroundImage: `
          linear-gradient(rgba(255, 107, 74, 0.03) 1px, transparent 1px),
          linear-gradient(90deg, rgba(255, 107, 74, 0.03) 1px, transparent 1px)
        `,
        backgroundSize: '60px 60px',
        maskImage: 'radial-gradient(ellipse at center, black 0%, transparent 70%)',
        WebkitMaskImage: 'radial-gradient(ellipse at center, black 0%, transparent 70%)',
        pointerEvents: 'none',
      }}
    />
  );
}

// Volleyball icon that floats
function FloatingVolleyball({
  top,
  left,
  size = 40,
  delay = 0,
}: {
  top: string;
  left: string;
  size?: number;
  delay?: number;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{
        opacity: [0.15, 0.25, 0.15],
        y: [0, -20, 0],
        rotate: [0, 10, -5, 0],
      }}
      transition={{
        duration: 6,
        delay,
        repeat: Infinity,
        ease: 'easeInOut',
      }}
      style={{
        position: 'absolute',
        top,
        left,
        pointerEvents: 'none',
      }}
    >
      <svg
        width={size}
        height={size}
        viewBox="0 0 24 24"
        fill="none"
        stroke="rgba(255, 107, 74, 0.4)"
        strokeWidth="1"
      >
        <circle cx="12" cy="12" r="10" />
        <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10" />
        <path d="M12 2a15.3 15.3 0 0 0-4 10 15.3 15.3 0 0 0 4 10" />
        <path d="M2 12h20" />
      </svg>
    </motion.div>
  );
}

export function HeroBackground() {
  const ref = useRef(null);
  const shouldReduceMotion = useReducedMotion();
  const { scrollYProgress } = useScroll({
    target: ref,
    offset: ['start start', 'end start'],
  });

  const y1 = useTransform(scrollYProgress, [0, 1], [0, 150]);
  const y2 = useTransform(scrollYProgress, [0, 1], [0, 100]);
  const opacity = useTransform(scrollYProgress, [0, 0.5], [1, 0]);

  if (shouldReduceMotion) {
    return (
      <Box
        sx={{
          position: 'absolute',
          inset: 0,
          overflow: 'hidden',
          pointerEvents: 'none',
        }}
      >
        {/* Static gradient backgrounds */}
        <Box
          sx={{
            position: 'absolute',
            top: 0,
            left: '50%',
            transform: 'translateX(-50%)',
            width: '200%',
            height: '100%',
            background:
              'radial-gradient(ellipse at 50% 0%, rgba(255, 107, 74, 0.15) 0%, transparent 60%)',
          }}
        />
        <Box
          sx={{
            position: 'absolute',
            bottom: 0,
            left: '50%',
            transform: 'translateX(-50%)',
            width: '200%',
            height: '50%',
            background:
              'radial-gradient(ellipse at 50% 100%, rgba(0, 212, 170, 0.08) 0%, transparent 60%)',
          }}
        />
      </Box>
    );
  }

  return (
    <Box
      ref={ref}
      sx={{
        position: 'absolute',
        inset: 0,
        overflow: 'hidden',
        pointerEvents: 'none',
      }}
    >
      {/* Animated grid */}
      <GridPattern />

      {/* Primary gradient orb (coral - top) */}
      <motion.div style={{ y: y1, opacity }}>
        <FloatingOrb color="rgba(255, 107, 74, 0.4)" size={600} top="-20%" left="30%" delay={0} />
      </motion.div>

      {/* Secondary gradient orb (teal - bottom right) */}
      <motion.div style={{ y: y2, opacity }}>
        <FloatingOrb color="rgba(0, 212, 170, 0.25)" size={400} top="40%" left="70%" delay={2} />
      </motion.div>

      {/* Tertiary orb (gold accent) */}
      <motion.div style={{ y: y1, opacity }}>
        <FloatingOrb color="rgba(255, 209, 102, 0.2)" size={300} top="60%" left="10%" delay={4} />
      </motion.div>

      {/* Floating volleyball icons */}
      <motion.div style={{ opacity }}>
        <FloatingVolleyball top="15%" left="85%" size={50} delay={0} />
        <FloatingVolleyball top="70%" left="5%" size={35} delay={1.5} />
        <FloatingVolleyball top="25%" left="8%" size={25} delay={3} />
        <FloatingVolleyball top="80%" left="90%" size={30} delay={2} />
      </motion.div>
    </Box>
  );
}
