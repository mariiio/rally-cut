'use client';

import { motion, useScroll, useSpring, useReducedMotion } from 'framer-motion';
import { designTokens } from '@/app/theme';

export function ScrollProgress() {
  const { scrollYProgress } = useScroll();
  const shouldReduceMotion = useReducedMotion();

  const scaleX = useSpring(scrollYProgress, {
    stiffness: 100,
    damping: 30,
    restDelta: 0.001,
  });

  if (shouldReduceMotion) {
    return null;
  }

  return (
    <motion.div
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        height: 3,
        background: designTokens.gradients.primary,
        transformOrigin: '0%',
        scaleX,
        zIndex: 9999,
      }}
    />
  );
}
