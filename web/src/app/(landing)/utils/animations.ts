import { Variants, Transition } from 'framer-motion';

// Fade in from below
export const fadeInUp: Variants = {
  hidden: { opacity: 0, y: 30 },
  visible: { opacity: 1, y: 0 },
};

// Simple fade in
export const fadeIn: Variants = {
  hidden: { opacity: 0 },
  visible: { opacity: 1 },
};

// Scale in from smaller
export const scaleIn: Variants = {
  hidden: { opacity: 0, scale: 0.9 },
  visible: { opacity: 1, scale: 1 },
};

// Slide in from left
export const slideInLeft: Variants = {
  hidden: { opacity: 0, x: -50 },
  visible: { opacity: 1, x: 0 },
};

// Slide in from right
export const slideInRight: Variants = {
  hidden: { opacity: 0, x: 50 },
  visible: { opacity: 1, x: 0 },
};

// Container that staggers children
export const staggerContainer: Variants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1,
      delayChildren: 0.1,
    },
  },
};

// Faster stagger for grids
export const staggerContainerFast: Variants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.08,
      delayChildren: 0.05,
    },
  },
};

// Smooth ease-out-expo curve
export const smoothTransition: Transition = {
  duration: 0.6,
  ease: [0.22, 1, 0.36, 1],
};

// Spring physics for interactions
export const springTransition: Transition = {
  type: 'spring',
  stiffness: 100,
  damping: 15,
};

// Snappy spring for hovers
export const snappySpring: Transition = {
  type: 'spring',
  stiffness: 400,
  damping: 25,
};

// Hover and tap props for interactive elements
export const hoverScale = {
  whileHover: { scale: 1.02 },
  whileTap: { scale: 0.98 },
  transition: snappySpring,
};

// Lift effect for cards
export const hoverLift = {
  whileHover: { y: -8 },
  transition: snappySpring,
};

// Glow effect preset (combine with hover)
export const glowOnHover = {
  whileHover: {
    boxShadow: '0 20px 40px rgba(255, 107, 74, 0.15)',
  },
};
