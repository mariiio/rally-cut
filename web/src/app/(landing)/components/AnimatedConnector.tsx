'use client';

import { motion, useInView } from 'framer-motion';
import { useRef } from 'react';

interface AnimatedConnectorProps {
  delay?: number;
}

export function AnimatedConnector({ delay = 0 }: AnimatedConnectorProps) {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: '-50px' });

  return (
    <svg
      ref={ref}
      width="40"
      height="24"
      viewBox="0 0 40 24"
      fill="none"
      style={{
        position: 'absolute',
        right: -20,
        top: '50%',
        transform: 'translateY(-50%)',
      }}
    >
      <motion.path
        d="M0 12H32M32 12L24 4M32 12L24 20"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        initial={{ pathLength: 0, opacity: 0 }}
        animate={isInView ? { pathLength: 1, opacity: 1 } : {}}
        transition={{
          pathLength: { duration: 0.6, delay: delay + 0.3, ease: 'easeOut' },
          opacity: { duration: 0.2, delay },
        }}
        style={{
          color: 'rgba(255, 107, 74, 0.5)',
        }}
      />
    </svg>
  );
}
