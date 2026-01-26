'use client';

import { Box, Typography, useMediaQuery, useTheme } from '@mui/material';
import { motion, useReducedMotion } from 'framer-motion';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import { useState, useEffect } from 'react';
import { designTokens } from '@/app/theme';

type AnimationPhase = 'idle' | 'scanning' | 'revealing' | 'complete';

// --- Segment data ---

interface Segment {
  type: 'rally' | 'dead';
  start: number; // percentage position (0-100)
  width: number; // percentage width
  color: string;
}

// Input timeline: ~60% dead time, ~40% rally time = 100% total
const INPUT_SEGMENTS: Segment[] = [
  { type: 'rally', start: 0, width: 5, color: '#FF6B4A' },
  { type: 'dead', start: 5, width: 8, color: 'rgba(255,255,255,0.08)' },
  { type: 'rally', start: 13, width: 4, color: '#00D4AA' },
  { type: 'dead', start: 17, width: 6, color: 'rgba(255,255,255,0.08)' },
  { type: 'rally', start: 23, width: 7, color: '#FF6B4A' },
  { type: 'dead', start: 30, width: 5, color: 'rgba(255,255,255,0.08)' },
  { type: 'rally', start: 35, width: 3, color: '#FFD166' },
  { type: 'dead', start: 38, width: 9, color: 'rgba(255,255,255,0.08)' },
  { type: 'rally', start: 47, width: 6, color: '#FF6B4A' },
  { type: 'dead', start: 53, width: 7, color: 'rgba(255,255,255,0.08)' },
  { type: 'rally', start: 60, width: 4, color: '#00D4AA' },
  { type: 'dead', start: 64, width: 10, color: 'rgba(255,255,255,0.08)' },
  { type: 'rally', start: 74, width: 5, color: '#FF6B4A' },
  { type: 'dead', start: 79, width: 8, color: 'rgba(255,255,255,0.08)' },
  { type: 'rally', start: 87, width: 6, color: '#FFD166' },
  { type: 'dead', start: 93, width: 7, color: 'rgba(255,255,255,0.08)' },
];

const RALLY_SEGMENTS = INPUT_SEGMENTS.filter((s) => s.type === 'rally');
const RALLY_COUNT = RALLY_SEGMENTS.length;

// Output segments: rallies packed together with no dead-time gaps
const OUTPUT_SEGMENTS: Segment[] = (() => {
  let offset = 0;
  const totalWidth = RALLY_SEGMENTS.reduce((sum, s) => sum + s.width, 0);
  return RALLY_SEGMENTS.map((s) => {
    const seg: Segment = {
      type: 'rally',
      start: (offset / totalWidth) * 100,
      width: (s.width / totalWidth) * 100,
      color: s.color,
    };
    offset += s.width;
    return seg;
  });
})();

// --- Phase timing (ms) ---
const PHASE_TIMING = {
  idle: 1000,
  scanning: 3000,
  revealing: 2000,
  complete: 2000,
  fadeOut: 1000,
};

// --- Sub-components ---

function WindowChrome({ title }: { title: string }) {
  return (
    <Box
      sx={{
        display: 'flex',
        alignItems: 'center',
        gap: 0.75,
        px: 1.5,
        py: 0.75,
        bgcolor: designTokens.colors.surface[2],
        borderBottom: '1px solid',
        borderColor: 'divider',
      }}
    >
      {['#FF5F56', '#FFBD2E', '#27C93F'].map((color) => (
        <Box
          key={color}
          sx={{ width: 7, height: 7, borderRadius: '50%', bgcolor: color }}
        />
      ))}
      <Typography
        variant="caption"
        sx={{ ml: 1, color: 'text.secondary', fontSize: '0.65rem' }}
      >
        {title}
      </Typography>
    </Box>
  );
}

function VideoFrame({ title }: { title: string }) {
  return (
    <Box
      sx={{
        borderRadius: 2,
        overflow: 'hidden',
        bgcolor: designTokens.colors.surface[1],
        border: '1px solid',
        borderColor: 'divider',
        boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4)',
      }}
    >
      <WindowChrome title={title} />
      <Box
        sx={{
          aspectRatio: '16/9',
          bgcolor: designTokens.colors.surface[0],
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          position: 'relative',
        }}
      >
        {/* Subtle gradient for depth */}
        <Box
          sx={{
            position: 'absolute',
            inset: 0,
            background:
              'radial-gradient(ellipse at 50% 40%, rgba(255,255,255,0.03) 0%, transparent 70%)',
          }}
        />
        {/* Play button */}
        <Box
          sx={{
            width: 34,
            height: 34,
            borderRadius: '50%',
            border: '2px solid rgba(255,255,255,0.15)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <Box
            sx={{
              width: 0,
              height: 0,
              borderTop: '7px solid transparent',
              borderBottom: '7px solid transparent',
              borderLeft: '11px solid rgba(255,255,255,0.2)',
              ml: '2px',
            }}
          />
        </Box>
      </Box>
    </Box>
  );
}

function SegmentTimeline({
  segments,
  scanProgress,
  showScan = false,
  staggerReveal = false,
  revealed = true,
  label,
}: {
  segments: Segment[];
  scanProgress?: number;
  showScan?: boolean;
  staggerReveal?: boolean;
  revealed?: boolean;
  label?: string;
}) {
  return (
    <Box sx={{ mt: 1 }}>
      <Box
        sx={{
          height: 24,
          bgcolor: designTokens.colors.surface[0],
          borderRadius: 1,
          position: 'relative',
          overflow: 'hidden',
          border: '1px solid',
          borderColor: 'divider',
          contain: 'layout paint',
        }}
      >
        {/* Waveform background texture */}
        <Box
          sx={{
            position: 'absolute',
            inset: 0,
            opacity: 0.15,
            backgroundImage: `repeating-linear-gradient(
              90deg,
              transparent,
              transparent 2px,
              rgba(255, 255, 255, 0.1) 2px,
              rgba(255, 255, 255, 0.1) 4px
            )`,
          }}
        />

        {/* Scan beam — sweeps left to right during scanning phase */}
        {showScan && scanProgress !== undefined && scanProgress < 1 && (
          <Box
            sx={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              height: '100%',
              transform: `translateX(${scanProgress * 200 - 100}%)`,
              background:
                'linear-gradient(90deg, transparent 0%, rgba(255, 107, 74, 0.3) 50%, transparent 100%)',
              willChange: 'transform',
              pointerEvents: 'none',
              zIndex: 2,
            }}
          />
        )}

        {/* Segments */}
        {segments.map((seg, i) => {
          const isRally = seg.type === 'rally';
          const isDetected =
            isRally && scanProgress !== undefined
              ? scanProgress * 100 >= seg.start + seg.width * 0.5
              : isRally;

          if (staggerReveal) {
            return (
              <motion.div
                key={i}
                initial={{ opacity: 0, scaleX: 0 }}
                animate={{
                  opacity: revealed ? 1 : 0,
                  scaleX: revealed ? 1 : 0,
                }}
                transition={{
                  delay: revealed ? i * 0.1 : 0,
                  duration: 0.3,
                  ease: [0.22, 1, 0.36, 1],
                }}
                style={{
                  position: 'absolute',
                  left: `${seg.start}%`,
                  top: '15%',
                  width: `${seg.width}%`,
                  height: '70%',
                  background: seg.color,
                  borderRadius: 3,
                  transformOrigin: 'left center',
                  willChange: 'transform, opacity',
                }}
              />
            );
          }

          return (
            <Box
              key={i}
              sx={{
                position: 'absolute',
                left: `${seg.start}%`,
                top: '15%',
                width: `${seg.width}%`,
                height: '70%',
                bgcolor: seg.color,
                borderRadius: 0.5,
                opacity: isRally ? (isDetected ? 1 : 0.3) : 1,
                boxShadow:
                  isRally && isDetected
                    ? `0 0 8px ${seg.color}80`
                    : 'none',
                transition: 'opacity 0.3s, box-shadow 0.3s',
                willChange: 'opacity',
              }}
            />
          );
        })}
      </Box>

      {label && (
        <Typography
          variant="caption"
          sx={{
            color: 'text.secondary',
            fontSize: '0.7rem',
            mt: 0.75,
            display: 'block',
            textAlign: 'center',
          }}
        >
          {label}
        </Typography>
      )}
    </Box>
  );
}

function ProcessingConnector({
  active,
  phase,
  vertical,
}: {
  active: boolean;
  phase: AnimationPhase;
  vertical: boolean;
}) {
  const showStats = phase === 'complete';

  // Gap values from the parent flex container (must match)
  // Parent uses gap: { xs: 2, md: 3 } → 16px mobile, 24px desktop
  const gapPx = vertical ? 16 : 24;

  const lineGradient = vertical
    ? active
      ? 'repeating-linear-gradient(180deg, transparent 0px, transparent 4px, rgba(255,107,74,0.3) 4px, rgba(255,107,74,0.3) 8px)'
      : 'repeating-linear-gradient(180deg, transparent 0px, transparent 4px, rgba(255,255,255,0.1) 4px, rgba(255,255,255,0.1) 8px)'
    : active
      ? 'repeating-linear-gradient(90deg, transparent 0px, transparent 4px, rgba(255,107,74,0.3) 4px, rgba(255,107,74,0.3) 8px)'
      : 'repeating-linear-gradient(90deg, transparent 0px, transparent 4px, rgba(255,255,255,0.1) 4px, rgba(255,255,255,0.1) 8px)';

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        flexShrink: 0,
      }}
    >
      {/* AI Badge — above the line */}
      <Box
        sx={{
          position: 'relative',
          zIndex: 1,
          display: 'flex',
          alignItems: 'center',
          gap: 0.75,
          px: 1.5,
          py: 0.75,
          mb: 1.5,
          bgcolor: designTokens.colors.surface[2],
          borderRadius: 2,
          border: '1px solid',
          borderColor: active ? 'rgba(255, 107, 74, 0.3)' : 'divider',
          boxShadow: active ? designTokens.shadows.glow.primary : 'none',
          transition: 'all 0.3s',
        }}
      >
        <motion.div
          animate={
            phase === 'scanning'
              ? { rotate: [0, 15, -15, 0], scale: [1, 1.2, 1, 1.15, 1] }
              : active
                ? { scale: [1, 1.1, 1] }
                : {}
          }
          transition={
            phase === 'scanning'
              ? { duration: 1.5, repeat: Infinity, ease: 'easeInOut' }
              : active
                ? { duration: 2, repeat: Infinity, ease: 'easeInOut' }
                : {}
          }
          style={{ display: 'flex' }}
        >
          <AutoAwesomeIcon
            sx={{
              fontSize: 16,
              color: active ? 'primary.main' : 'text.disabled',
              filter: active ? 'drop-shadow(0 0 3px rgba(255,107,74,0.5))' : 'none',
              transition: 'filter 0.3s',
            }}
          />
        </motion.div>
        <Typography
          variant="caption"
          sx={{
            color: active ? 'primary.light' : 'text.disabled',
            fontWeight: 600,
            fontSize: '0.7rem',
            whiteSpace: 'nowrap',
          }}
        >
          AI Detection
        </Typography>
      </Box>

      {/* Line — flex child between badge and stats, extends into parent gaps */}
      <Box
        sx={{
          position: 'relative',
          ...(vertical
            ? { width: 2, height: 24, alignSelf: 'center' }
            : { height: 2, alignSelf: 'stretch' }),
        }}
      >
        {/* Line background — extends into parent gaps to touch stage edges */}
        <Box
          sx={{
            position: 'absolute',
            ...(vertical
              ? { left: 0, right: 0, top: -gapPx, bottom: -gapPx }
              : { top: 0, bottom: 0, left: -gapPx, right: -gapPx }),
            background: lineGradient,
            transition: 'background 0.3s',
          }}
        />

        {/* Flowing particles */}
        {active && (
          <Box
            sx={{
              position: 'absolute',
              ...(vertical
                ? { left: '50%', top: -gapPx, bottom: -gapPx, width: 20, transform: 'translateX(-50%)' }
                : { top: '50%', left: -gapPx, right: -gapPx, height: 20, transform: 'translateY(-50%)' }),
              overflow: 'hidden',
              pointerEvents: 'none',
            }}
          >
            {[0, 1, 2].map((i) => (
              <motion.div
                key={i}
                animate={
                  vertical
                    ? { y: ['-10%', '110%'], opacity: [0, 1, 1, 0] }
                    : { x: ['-10%', '110%'], opacity: [0, 1, 1, 0] }
                }
                transition={{
                  duration: 1.5,
                  delay: i * 0.5,
                  repeat: Infinity,
                  ease: 'linear',
                }}
                style={{
                  position: 'absolute',
                  ...(vertical
                    ? { left: '50%', marginLeft: -3 }
                    : { top: '50%', marginTop: -3 }),
                  width: 6,
                  height: 6,
                  borderRadius: '50%',
                  background: '#FF6B4A',
                  boxShadow: '0 0 6px rgba(255, 107, 74, 0.6)',
                  willChange: 'transform, opacity',
                }}
              />
            ))}
          </Box>
        )}
      </Box>

      {/* Stat — below the line */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: showStats ? 1 : 0, y: showStats ? 0 : 10 }}
        transition={{ duration: 0.4 }}
        style={{ position: 'relative', zIndex: 1, willChange: 'transform, opacity' }}
      >
        <Typography
          variant="caption"
          sx={{
            color: 'secondary.main',
            fontWeight: 600,
            fontSize: '0.7rem',
            whiteSpace: 'nowrap',
            textAlign: 'center',
            mt: 1.5,
          }}
        >
          91% dead time removed
        </Typography>
      </motion.div>
    </Box>
  );
}

// --- Main Component ---

export function PipelineAnimation() {
  const shouldReduceMotion = useReducedMotion();
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));

  const [phase, setPhase] = useState<AnimationPhase>(
    shouldReduceMotion ? 'complete' : 'idle',
  );
  const [scanProgress, setScanProgress] = useState(shouldReduceMotion ? 1 : 0);
  const [visible, setVisible] = useState(true);

  // Animation cycle
  useEffect(() => {
    if (shouldReduceMotion) return;

    let cancelled = false;
    const timers: ReturnType<typeof setTimeout>[] = [];
    let scanInterval: ReturnType<typeof setInterval> | null = null;

    const schedule = (fn: () => void, ms: number) => {
      const id = setTimeout(fn, ms);
      timers.push(id);
      return id;
    };

    const runCycle = () => {
      if (cancelled) return;

      // Reset to idle
      setPhase('idle');
      setScanProgress(0);
      setVisible(true);

      // -> scanning
      schedule(() => {
        if (cancelled) return;
        setPhase('scanning');

        const scanStart = Date.now();
        scanInterval = setInterval(() => {
          if (cancelled) return;
          const progress = Math.min((Date.now() - scanStart) / PHASE_TIMING.scanning, 1);
          setScanProgress(progress);
          if (progress >= 1 && scanInterval) {
            clearInterval(scanInterval);
            scanInterval = null;
          }
        }, 50);

        // -> revealing
        schedule(() => {
          if (cancelled) return;
          if (scanInterval) {
            clearInterval(scanInterval);
            scanInterval = null;
          }
          setScanProgress(1);
          setPhase('revealing');

          // -> complete
          schedule(() => {
            if (cancelled) return;
            setPhase('complete');

            // -> fade out & restart
            schedule(() => {
              if (cancelled) return;
              setVisible(false);

              schedule(() => {
                if (cancelled) return;
                runCycle();
              }, PHASE_TIMING.fadeOut);
            }, PHASE_TIMING.complete);
          }, PHASE_TIMING.revealing);
        }, PHASE_TIMING.scanning);
      }, PHASE_TIMING.idle);
    };

    runCycle();

    return () => {
      cancelled = true;
      timers.forEach(clearTimeout);
      if (scanInterval) clearInterval(scanInterval);
    };
  }, [shouldReduceMotion]);

  const connectorActive =
    phase === 'scanning' || phase === 'revealing' || phase === 'complete';
  const outputVisible = phase === 'revealing' || phase === 'complete';

  const ralliesDetected =
    phase === 'scanning'
      ? RALLY_SEGMENTS.filter((s) => scanProgress * 100 >= s.start + s.width * 0.5).length
      : phase === 'idle'
        ? 0
        : RALLY_COUNT;

  // Reduced motion: static final state
  if (shouldReduceMotion) {
    return (
      <Box
        sx={{
          display: 'flex',
          flexDirection: { xs: 'column', md: 'row' },
          alignItems: { xs: 'stretch', md: 'center' },
          gap: 3,
          mt: { xs: 3, md: 4 },
          maxWidth: { md: 1080 },
          mx: 'auto',
        }}
      >
        <Box sx={{ flex: 1, minWidth: 0 }}>
          <VideoFrame title="Raw Match" />
          <SegmentTimeline segments={INPUT_SEGMENTS} label="45 min raw footage" />
        </Box>
        <ProcessingConnector active={false} phase="complete" vertical={isMobile} />
        <Box sx={{ flex: 1, minWidth: 0 }}>
          <VideoFrame title="Highlights" />
          <SegmentTimeline
            segments={OUTPUT_SEGMENTS}
            label="4 min of action"
          />
        </Box>
      </Box>
    );
  }

  return (
    <motion.div
      animate={{ opacity: visible ? 1 : 0 }}
      transition={{ duration: visible ? 0.6 : 0.8 }}
      style={{ willChange: 'opacity' }}
    >
      <Box
        sx={{
          display: 'flex',
          flexDirection: { xs: 'column', md: 'row' },
          alignItems: { xs: 'stretch', md: 'center' },
          gap: { xs: 2, md: 3 },
          mt: { xs: 3, md: 4 },
          maxWidth: { md: 1080 },
          mx: 'auto',
        }}
      >
        {/* Input Stage */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
          style={{ flex: 1, minWidth: 0 }}
        >
          <VideoFrame title="Raw Match" />
          <SegmentTimeline
            segments={INPUT_SEGMENTS}
            scanProgress={phase === 'scanning' ? scanProgress : phase === 'idle' ? 0 : 1}
            showScan={phase === 'scanning'}
            label="45 min raw footage"
          />
        </motion.div>

        {/* Connector */}
        <ProcessingConnector
          active={connectorActive}
          phase={phase}
          vertical={isMobile}
        />

        {/* Output Stage */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{
            opacity: outputVisible ? 1 : 0,
            y: outputVisible ? 0 : 20,
          }}
          transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
          style={{ flex: 1, minWidth: 0 }}
        >
          <VideoFrame title="Highlights" />
          <SegmentTimeline
            segments={OUTPUT_SEGMENTS}
            staggerReveal
            revealed={outputVisible}
            label="4 min of action"
          />

          {/* Rally counter */}
          <motion.div
            animate={{
              opacity: outputVisible ? 1 : 0,
              y: outputVisible ? 0 : 10,
            }}
            transition={{ duration: 0.4, delay: outputVisible ? 0.5 : 0 }}
            style={{ willChange: 'transform, opacity' }}
          >
            <Typography
              variant="caption"
              sx={{
                color: 'primary.light',
                fontWeight: 600,
                fontSize: '0.75rem',
                mt: 0.5,
                display: 'block',
                textAlign: 'center',
              }}
            >
              {ralliesDetected} rallies detected
            </Typography>
          </motion.div>
        </motion.div>
      </Box>
    </motion.div>
  );
}
