'use client';

import { ReactNode, useCallback, useEffect, useMemo } from 'react';
import {
  Box,
  Button,
  Chip,
  Stack,
  Switch,
  ToggleButton,
  ToggleButtonGroup,
  Typography,
} from '@mui/material';
import CheckIcon from '@mui/icons-material/Check';
import NavigateBeforeIcon from '@mui/icons-material/NavigateBefore';
import NavigateNextIcon from '@mui/icons-material/NavigateNext';
import { usePlayerTrackingStore } from '@/stores/playerTrackingStore';
import { useEditorStore } from '@/stores/editorStore';
import { usePlayerStore } from '@/stores/playerStore';
import type { ScoreGtEntry, ScoreTeam } from '@/services/api';

interface Props {
  renderSection: (status: ReactNode, body: ReactNode) => ReactNode;
}

// Team identity is stable across the match, but the court side of each
// team flips on every side switch. `nearSideTeam` from the backend tells us
// which team is physically on the near court for the current rally.
const TEAM_COLOR: Record<'A' | 'B', string> = {
  A: '#2196F3', // primary / blue
  B: '#9C27B0', // secondary / purple
};

function sideLabel(team: 'A' | 'B', nearSide: ScoreTeam): string {
  if (nearSide === null) return '';
  return team === nearSide ? 'near court' : 'far court';
}

function TeamSelector({
  value,
  onChange,
  unconfirmed,
  nearSide,
}: {
  value: ScoreTeam;
  onChange: (team: 'A' | 'B') => void;
  unconfirmed?: boolean;
  nearSide: ScoreTeam;
}) {
  // Camera-perspective ordering: whichever team is on the far court renders
  // on top, near court on bottom. When nearSide is unknown, default to B/A.
  const orderedTeams: Array<'A' | 'B'> =
    nearSide === 'A' ? ['B', 'A'] : nearSide === 'B' ? ['A', 'B'] : ['B', 'A'];

  return (
    <ToggleButtonGroup
      exclusive
      orientation="vertical"
      value={value}
      // `onChange` alone doesn't fire when the already-selected button is
      // clicked (exclusive mode), which would make a pre-filled "unconfirmed"
      // choice impossible to confirm with a single click. We attach `onClick`
      // to each ToggleButton below instead so any click calls `onChange`.
      size="small"
      fullWidth
      sx={{
        width: '100%',
        '& .MuiToggleButton-root': {
          py: 0.75,
          textTransform: 'none',
          fontWeight: 600,
          borderWidth: 2,
          lineHeight: 1.15,
          justifyContent: 'flex-start',
        },
        '& .MuiToggleButton-root.Mui-selected': {
          color: '#fff',
          borderStyle: unconfirmed ? 'dashed' : 'solid',
        },
        '& .MuiToggleButton-root[value="A"].Mui-selected': {
          backgroundColor: TEAM_COLOR.A,
          borderColor: TEAM_COLOR.A,
          '&:hover': { backgroundColor: TEAM_COLOR.A },
        },
        '& .MuiToggleButton-root[value="B"].Mui-selected': {
          backgroundColor: TEAM_COLOR.B,
          borderColor: TEAM_COLOR.B,
          '&:hover': { backgroundColor: TEAM_COLOR.B },
        },
      }}
    >
      {orderedTeams.map((team) => {
        const isSelected = value === team;
        const where = sideLabel(team, nearSide);
        return (
          <ToggleButton
            key={team}
            value={team}
            onClick={() => onChange(team)}
          >
            <Stack
              direction="row"
              spacing={0.75}
              alignItems="center"
              justifyContent="center"
              sx={{ width: '100%' }}
            >
              {isSelected && <CheckIcon sx={{ fontSize: 16 }} />}
              <Box
                sx={{
                  width: 10,
                  height: 10,
                  borderRadius: '50%',
                  backgroundColor: TEAM_COLOR[team],
                  border: isSelected ? '1px solid #fff' : 'none',
                  flexShrink: 0,
                }}
              />
              <Stack sx={{ alignItems: 'flex-start', lineHeight: 1.1 }}>
                <Typography component="span" sx={{ fontSize: '0.78rem', fontWeight: 700 }}>
                  Team {team}
                </Typography>
                {where && (
                  <Typography
                    component="span"
                    sx={{
                      fontSize: '0.6rem',
                      opacity: 0.85,
                      textTransform: 'none',
                    }}
                  >
                    {where}
                  </Typography>
                )}
              </Stack>
            </Stack>
          </ToggleButton>
        );
      })}
    </ToggleButtonGroup>
  );
}

export function ScoreGtSection({ renderSection }: Props) {
  const activeMatchId = useEditorStore((s) => s.activeMatchId);
  const rallies = useEditorStore((s) => s.rallies);
  const selectedRallyId = useEditorStore((s) => s.selectedRallyId);
  const selectRally = useEditorStore((s) => s.selectRally);
  const seek = usePlayerStore((s) => s.seek);

  const scoreGtForVideo = usePlayerTrackingStore((s) =>
    activeMatchId ? s.scoreGt[activeMatchId] : undefined,
  );
  const fetchScoreGt = usePlayerTrackingStore((s) => s.fetchScoreGt);
  const saveScoreGt = usePlayerTrackingStore((s) => s.saveScoreGt);
  const setRallySideSwitch = usePlayerTrackingStore((s) => s.setRallySideSwitch);

  useEffect(() => {
    if (activeMatchId && !scoreGtForVideo) {
      void fetchScoreGt(activeMatchId);
    }
  }, [activeMatchId, scoreGtForVideo, fetchScoreGt]);

  const entries: ScoreGtEntry[] = useMemo(() => scoreGtForVideo ?? [], [scoreGtForVideo]);

  const selectedRally = rallies.find((r) => r.id === selectedRallyId);
  const backendRallyId = selectedRally?._backendId ?? null;
  const currentEntryIdx = entries.findIndex((e) => e.rallyId === backendRallyId);
  const currentEntry = currentEntryIdx >= 0 ? entries[currentEntryIdx] : null;
  const isLastRally = entries.length > 0 && currentEntryIdx === entries.length - 1;

  /**
   * GT point winner for rally `i`:
   *   - i < N-1: next rally's gt_serving_team (exact in volleyball)
   *   - i == N-1: the labeler's explicit gt_point_winner (no next rally)
   * Returns null when the value cannot be determined yet.
   */
  const deriveGtPointWinner = (i: number): ScoreTeam => {
    if (i < 0 || i >= entries.length) return null;
    if (i < entries.length - 1) return entries[i + 1].gtServingTeam;
    return entries[i].gtPointWinner;
  };

  // "Labeled" = serving team is set, AND (for the last rally) point winner too.
  const labeledCount = entries.filter(
    (e, i) =>
      e.gtServingTeam !== null &&
      (i < entries.length - 1 ? true : e.gtPointWinner !== null),
  ).length;
  const total = entries.length;

  // Running score walking from the start; null if there is a gap before current.
  const runningScore: { scoreA: number; scoreB: number } | null = (() => {
    if (!currentEntry || currentEntryIdx < 0) return null;
    let scoreA = 0;
    let scoreB = 0;
    for (let i = 0; i <= currentEntryIdx; i++) {
      const winner = deriveGtPointWinner(i);
      if (winner === null) return null; // chain broken before/at current
      if (winner === 'A') scoreA += 1;
      else if (winner === 'B') scoreB += 1;
    }
    return { scoreA, scoreB };
  })();

  // Effective serving team display: GT if set, otherwise predicted (unconfirmed).
  const effectiveServing: ScoreTeam = currentEntry
    ? (currentEntry.gtServingTeam ?? currentEntry.servingTeam)
    : null;
  const servingIsConfirmed = currentEntry?.gtServingTeam !== null;
  const pointWinner = currentEntry?.gtPointWinner ?? null;

  const handleSetServing = useCallback(
    async (team: ScoreTeam) => {
      if (!activeMatchId || !currentEntry) return;
      await saveScoreGt(activeMatchId, currentEntry.rallyId, {
        gtServingTeam: team,
        gtPointWinner: currentEntry.gtPointWinner,
      }).catch(() => {
        /* store already reverted + logged */
      });
    },
    [activeMatchId, currentEntry, saveScoreGt],
  );

  const handleSetWinner = useCallback(
    async (team: ScoreTeam) => {
      if (!activeMatchId || !currentEntry) return;
      // When the user confirms a point winner, also lock in the serving team shown
      // (either the already-confirmed GT or the predicted value).
      const servingToSave = currentEntry.gtServingTeam ?? currentEntry.servingTeam ?? null;
      await saveScoreGt(activeMatchId, currentEntry.rallyId, {
        gtServingTeam: servingToSave,
        gtPointWinner: team,
      }).catch(() => {
        /* store already reverted + logged */
      });
    },
    [activeMatchId, currentEntry, saveScoreGt],
  );

  const jumpToEntry = useCallback(
    (entry: ScoreGtEntry) => {
      const target = rallies.find((r) => r._backendId === entry.rallyId);
      if (target) {
        selectRally(target.id);
        seek(target.start_time);
      } else {
        seek(entry.startMs / 1000);
      }
    },
    [rallies, selectRally, seek],
  );

  const findUnlabeled = useCallback(
    (direction: 'next' | 'prev'): ScoreGtEntry | null => {
      if (entries.length === 0) return null;
      const currentIdx = currentEntry
        ? entries.findIndex((e) => e.rallyId === currentEntry.rallyId)
        : -1;
      // Serving team must be set for every rally; point winner is only
      // required on the last rally (others are derived from the next rally's
      // serving team).
      const isUnlabeled = (e: ScoreGtEntry, i: number) => {
        if (e.gtServingTeam === null) return true;
        if (i === entries.length - 1 && e.gtPointWinner === null) return true;
        return false;
      };
      if (direction === 'next') {
        for (let i = currentIdx + 1; i < entries.length; i++) {
          if (isUnlabeled(entries[i], i)) return entries[i];
        }
        for (let i = 0; i <= currentIdx && i < entries.length; i++) {
          if (isUnlabeled(entries[i], i)) return entries[i];
        }
      } else {
        for (let i = currentIdx - 1; i >= 0; i--) {
          if (isUnlabeled(entries[i], i)) return entries[i];
        }
        for (let i = entries.length - 1; i >= currentIdx && i >= 0; i--) {
          if (isUnlabeled(entries[i], i)) return entries[i];
        }
      }
      return null;
    },
    [entries, currentEntry],
  );

  const handleNextUnlabeled = useCallback(() => {
    const next = findUnlabeled('next');
    if (next) jumpToEntry(next);
  }, [findUnlabeled, jumpToEntry]);

  const handlePrevUnlabeled = useCallback(() => {
    const prev = findUnlabeled('prev');
    if (prev) jumpToEntry(prev);
  }, [findUnlabeled, jumpToEntry]);

  // Keyboard shortcuts: N = next unlabeled, B = previous unlabeled.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const active = document.activeElement;
      if (active && (active.tagName === 'INPUT' || active.tagName === 'TEXTAREA'
        || (active as HTMLElement).isContentEditable)) return;
      if (e.key === 'n' || e.key === 'N') {
        e.preventDefault();
        handleNextUnlabeled();
      } else if (e.key === 'b' || e.key === 'B') {
        e.preventDefault();
        handlePrevUnlabeled();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [handleNextUnlabeled, handlePrevUnlabeled]);

  const status = (
    <Chip
      label={`${labeledCount} / ${total}`}
      size="small"
      sx={{ height: 18, fontSize: '0.65rem' }}
    />
  );

  // Per-rally near-side team (dynamic, accounts for cumulative side switches).
  const nearSide: ScoreTeam = currentEntry?.nearSideTeam ?? null;

  // "Side switched" label in the legend is driven by the same resolved flag
  // as the toggle below, so the two are always in sync.
  const justSwitched = currentEntry?.sideSwitchHere === true;

  const legend = (
    <Box
      sx={{
        display: 'flex',
        flexWrap: 'wrap',
        gap: 1.5,
        px: 1,
        py: 0.5,
        borderRadius: 1,
        bgcolor: 'action.hover',
        alignItems: 'center',
      }}
    >
      {(['A', 'B'] as const).map((team) => {
        const where = sideLabel(team, nearSide);
        return (
          <Stack key={team} direction="row" spacing={0.5} alignItems="center">
            <Box
              sx={{
                width: 10,
                height: 10,
                borderRadius: '50%',
                backgroundColor: TEAM_COLOR[team],
              }}
            />
            <Typography variant="caption" sx={{ fontWeight: 600 }}>
              Team {team}
            </Typography>
            {where && (
              <Typography variant="caption" color="text.secondary">
                ({where})
              </Typography>
            )}
          </Stack>
        );
      })}
      {nearSide === null && currentEntry && (
        <Typography variant="caption" color="text.secondary" sx={{ fontStyle: 'italic' }}>
          court sides unknown — run match analysis to enable
        </Typography>
      )}
      {justSwitched && (
        <Typography
          variant="caption"
          sx={{ fontWeight: 700, color: 'warning.main', ml: 'auto' }}
        >
          ↔ side switched
        </Typography>
      )}
    </Box>
  );

  const body = (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.25 }}>
      {legend}
      {!activeMatchId ? (
        <Typography variant="caption" color="text.secondary">
          Open a video to label scores.
        </Typography>
      ) : !currentEntry ? (
        <Typography variant="caption" color="text.secondary">
          Select a rally to label its score.
        </Typography>
      ) : (
        <>
          <Stack direction="row" spacing={1} alignItems="stretch">
            <Box sx={{ flex: 1, minWidth: 0 }}>
              <Stack
                direction="row"
                alignItems="center"
                spacing={0.5}
                sx={{ mb: 0.5, minHeight: 18 }}
              >
                <Typography variant="caption" color="text.secondary">
                  Serving team
                </Typography>
                {!servingIsConfirmed && effectiveServing && (
                  <Chip
                    label="unconfirmed"
                    size="small"
                    sx={{
                      height: 14,
                      fontSize: '0.58rem',
                      bgcolor: 'warning.light',
                      color: 'warning.contrastText',
                      '& .MuiChip-label': { px: 0.5 },
                    }}
                  />
                )}
              </Stack>
              <TeamSelector
                value={effectiveServing}
                onChange={handleSetServing}
                unconfirmed={!servingIsConfirmed && effectiveServing !== null}
                nearSide={nearSide}
              />
            </Box>

            {isLastRally && (
              <Box sx={{ flex: 1, minWidth: 0 }}>
                <Stack
                  direction="row"
                  alignItems="center"
                  sx={{ mb: 0.5, minHeight: 18 }}
                >
                  <Typography variant="caption" color="text.secondary">
                    Point winner
                  </Typography>
                </Stack>
                <TeamSelector
                  value={pointWinner}
                  onChange={handleSetWinner}
                  nearSide={nearSide}
                />
              </Box>
            )}
          </Stack>

          {!isLastRally && (
            <Typography variant="caption" color="text.secondary" sx={{ fontStyle: 'italic' }}>
              Point winner auto-inferred from the next rally&apos;s serving team.
            </Typography>
          )}

          <Box>
            <Typography
              variant="caption"
              color="text.secondary"
              sx={{ display: 'block' }}
            >
              Running score
            </Typography>
            <Typography variant="h6" sx={{ fontWeight: 700, lineHeight: 1.2 }}>
              {runningScore
                ? `${runningScore.scoreA} – ${runningScore.scoreB}`
                : '—'}
            </Typography>
          </Box>

          <Stack
            direction="row"
            alignItems="center"
            justifyContent="space-between"
            sx={{
              px: 1,
              py: 0.5,
              borderRadius: 1,
              border: '1px solid',
              borderColor: 'divider',
            }}
          >
            <Typography variant="caption" sx={{ fontWeight: 600 }}>
              Side switch at this rally
            </Typography>
            <Switch
              size="small"
              checked={currentEntry.sideSwitchHere}
              onChange={(_, checked) => {
                if (!activeMatchId) return;
                void setRallySideSwitch(activeMatchId, currentEntry.rallyId, checked).catch(
                  () => {
                    /* store already reverted + logged */
                  },
                );
              }}
            />
          </Stack>

          <Stack direction="row" spacing={1} sx={{ mt: 0.5 }}>
            <Button
              size="small"
              variant="outlined"
              startIcon={<NavigateBeforeIcon />}
              onClick={handlePrevUnlabeled}
              disabled={labeledCount >= total}
              sx={{ flex: 1 }}
            >
              Prev unlabeled
            </Button>
            <Button
              size="small"
              variant="contained"
              endIcon={<NavigateNextIcon />}
              onClick={handleNextUnlabeled}
              disabled={labeledCount >= total}
              sx={{ flex: 1 }}
            >
              Next unlabeled
            </Button>
          </Stack>
          <Typography variant="caption" color="text.secondary">
            Shortcuts: <b>N</b> next unlabeled · <b>B</b> previous unlabeled
          </Typography>
        </>
      )}
    </Box>
  );

  return <>{renderSection(status, body)}</>;
}
