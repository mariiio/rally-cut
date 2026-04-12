'use client';

import { useEffect, useState, useCallback } from 'react';
import {
  Box,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  Divider,
  CircularProgress,
  ToggleButton,
  ToggleButtonGroup,
} from '@mui/material';
import type { MatchStats } from '@/services/api';
import { getMatchStatsApi, getMatchAnalysis } from '@/services/api';
import { useEditorStore } from '@/stores/editorStore';
import CourtHeatmap from '@/components/CourtHeatmap';

function pct(value: number): string {
  return `${(value * 100).toFixed(0)}%`;
}

function TeamStatsTable({ stats }: { stats: MatchStats }) {
  const teams = stats.teamStats;
  if (!teams || teams.length === 0) return null;

  return (
    <Box sx={{ mb: 2 }}>
      <Typography variant="subtitle2" sx={{ mb: 0.5, fontWeight: 'bold' }}>
        Team Comparison
      </Typography>
      <TableContainer>
        <Table size="small" sx={{ '& td, & th': { py: 0.25, px: 0.75, fontSize: '0.75rem' } }}>
          <TableHead>
            <TableRow>
              <TableCell />
              {teams.map((t) => (
                <TableCell key={t.team} align="center" sx={{ fontWeight: 'bold' }}>
                  Team {t.team}
                </TableCell>
              ))}
            </TableRow>
          </TableHead>
          <TableBody>
            <TableRow>
              <TableCell sx={{ fontWeight: 500 }}>Points</TableCell>
              {teams.map((t) => (
                <TableCell key={t.team} align="center">{t.pointsWon}</TableCell>
              ))}
            </TableRow>
            <TableRow>
              <TableCell sx={{ fontWeight: 500 }}>Kills</TableCell>
              {teams.map((t) => (
                <TableCell key={t.team} align="center">{t.kills}</TableCell>
              ))}
            </TableRow>
            <TableRow>
              <TableCell sx={{ fontWeight: 500 }}>Kill %</TableCell>
              {teams.map((t) => (
                <TableCell key={t.team} align="center">{pct(t.killPct)}</TableCell>
              ))}
            </TableRow>
            <TableRow>
              <TableCell sx={{ fontWeight: 500 }}>Efficiency</TableCell>
              {teams.map((t) => (
                <TableCell key={t.team} align="center">
                  {(t.attackEfficiency * 100).toFixed(0)}%
                </TableCell>
              ))}
            </TableRow>
            <TableRow>
              <TableCell sx={{ fontWeight: 500 }}>Aces</TableCell>
              {teams.map((t) => (
                <TableCell key={t.team} align="center">{t.aces}</TableCell>
              ))}
            </TableRow>
            <TableRow>
              <TableCell sx={{ fontWeight: 500 }}>Errors</TableCell>
              {teams.map((t) => (
                <TableCell key={t.team} align="center">{t.attackErrors + t.serveErrors}</TableCell>
              ))}
            </TableRow>
            <TableRow>
              <TableCell sx={{ fontWeight: 500 }}>Side-out %</TableCell>
              {teams.map((t) => (
                <TableCell key={t.team} align="center">{pct(t.sideOutPct)}</TableCell>
              ))}
            </TableRow>
            <TableRow>
              <TableCell sx={{ fontWeight: 500 }}>Serve win %</TableCell>
              {teams.map((t) => (
                <TableCell key={t.team} align="center">{pct(t.serveWinPct)}</TableCell>
              ))}
            </TableRow>
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  );
}

function PlayerStatsTable({ stats, playerNames }: { stats: MatchStats; playerNames: Record<string, string> }) {
  const players = stats.playerStats.filter((p) => p.totalActions > 0);
  if (players.length === 0) return null;

  return (
    <Box sx={{ mb: 2 }}>
      <Typography variant="subtitle2" sx={{ mb: 0.5, fontWeight: 'bold' }}>
        Player Stats
      </Typography>
      <TableContainer>
        <Table size="small" sx={{ '& td, & th': { py: 0.25, px: 0.5, fontSize: '0.7rem' } }}>
          <TableHead>
            <TableRow>
              <TableCell>Player</TableCell>
              <TableCell align="center">SRV</TableCell>
              <TableCell align="center">RCV</TableCell>
              <TableCell align="center">SET</TableCell>
              <TableCell align="center">ATK</TableCell>
              <TableCell align="center">K</TableCell>
              <TableCell align="center">E</TableCell>
              <TableCell align="center">Eff</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {players.map((p) => {
              const name = playerNames[String(p.playerId)];
              return (
              <TableRow key={p.playerId}>
                <TableCell>
                  <Chip
                    label={name || `P${p.playerId}`}
                    size="small"
                    color={p.team === 'A' ? 'primary' : p.team === 'B' ? 'secondary' : 'default'}
                    sx={{ height: 18, fontSize: '0.65rem' }}
                  />
                </TableCell>
                <TableCell align="center">{p.serves || '-'}</TableCell>
                <TableCell align="center">{p.receives || '-'}</TableCell>
                <TableCell align="center">{p.sets || '-'}</TableCell>
                <TableCell align="center">{p.attacks || '-'}</TableCell>
                <TableCell align="center">{p.kills || '-'}</TableCell>
                <TableCell align="center">{p.attackErrors || '-'}</TableCell>
                <TableCell align="center">
                  {p.attacks > 0 ? `${(p.attackEfficiency * 100).toFixed(0)}%` : '-'}
                </TableCell>
              </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  );
}

function MatchOverview({ stats }: { stats: MatchStats }) {
  return (
    <Box sx={{ mb: 2 }}>
      {stats.finalScoreA !== undefined && stats.finalScoreB !== undefined && (
        <Box sx={{ textAlign: 'center', mb: 1 }}>
          <Typography variant="h6" sx={{ fontWeight: 'bold' }}>
            {stats.finalScoreA} - {stats.finalScoreB}
          </Typography>
          <Typography variant="caption" color="text.secondary">
            Team A vs Team B
          </Typography>
        </Box>
      )}
      <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', justifyContent: 'center' }}>
        <Chip
          label={`${stats.totalRallies} rallies`}
          size="small"
          variant="outlined"
          sx={{ fontSize: '0.7rem' }}
        />
        <Chip
          label={`${stats.totalContacts} contacts`}
          size="small"
          variant="outlined"
          sx={{ fontSize: '0.7rem' }}
        />
        <Chip
          label={`avg ${stats.avgRallyDurationS.toFixed(1)}s`}
          size="small"
          variant="outlined"
          sx={{ fontSize: '0.7rem' }}
        />
        <Chip
          label={`${stats.avgContactsPerRally.toFixed(1)} contacts/rally`}
          size="small"
          variant="outlined"
          sx={{ fontSize: '0.7rem' }}
        />
        {stats.sideOutRate > 0 && (
          <Chip
            label={`${pct(stats.sideOutRate)} side-out`}
            size="small"
            variant="outlined"
            sx={{ fontSize: '0.7rem' }}
          />
        )}
      </Box>
    </Box>
  );
}

function LandingHeatmapSection({ stats }: { stats: MatchStats }) {
  const [view, setView] = useState<'serve' | 'attack' | 'all'>('all');
  const hm = stats.landingHeatmaps;
  if (!hm) return null;

  const teamA = hm.teamA[view];
  const teamB = hm.teamB[view];
  if ((!teamA || teamA.count === 0) && (!teamB || teamB.count === 0)) return null;

  const titles: Record<string, string> = {
    serve: 'Serve Targets',
    attack: 'Attack Landings',
    all: 'All Landings',
  };

  return (
    <Box sx={{ mb: 1 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
        <Typography variant="subtitle2" sx={{ fontWeight: 'bold' }}>
          Landing Zones
        </Typography>
        <ToggleButtonGroup
          value={view}
          exclusive
          onChange={(_, v) => { if (v) setView(v); }}
          size="small"
          sx={{
            '& .MuiToggleButton-root': {
              py: 0, px: 0.75, fontSize: '0.65rem', height: 20,
              textTransform: 'none',
            },
          }}
        >
          <ToggleButton value="all">All</ToggleButton>
          <ToggleButton value="serve">Serve</ToggleButton>
          <ToggleButton value="attack">Attack</ToggleButton>
        </ToggleButtonGroup>
      </Box>
      <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap' }}>
        {teamA && teamA.count > 0 && (
          <CourtHeatmap
            grid={teamA.grid}
            count={teamA.count}
            points={view === 'all' ? hm.teamA.points : hm.teamA.points.filter(p => p.type === view)}
            title={`Team A — ${titles[view]}`}
          />
        )}
        {teamB && teamB.count > 0 && (
          <CourtHeatmap
            grid={teamB.grid}
            count={teamB.count}
            points={view === 'all' ? hm.teamB.points : hm.teamB.points.filter(p => p.type === view)}
            title={`Team B — ${titles[view]}`}
          />
        )}
      </Box>
    </Box>
  );
}

export function MatchStatsPanel() {
  const [stats, setStats] = useState<MatchStats | null>(null);
  const [playerNames, setPlayerNames] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const activeMatchId = useEditorStore((s) => s.activeMatchId);

  const loadStats = useCallback(async () => {
    if (!activeMatchId) return;
    setLoading(true);
    setError(null);
    try {
      const [result, analysis] = await Promise.all([
        getMatchStatsApi(activeMatchId),
        getMatchAnalysis(activeMatchId).catch(() => null),
      ]);
      setStats(result);
      // Extract player names from matchAnalysisJson
      const names = (analysis as Record<string, unknown> | null)?.playerNames as Record<string, string> | undefined;
      setPlayerNames(names ?? {});
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load stats');
    } finally {
      setLoading(false);
    }
  }, [activeMatchId]);

  useEffect(() => {
    loadStats();
  }, [loadStats]);

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%', p: 4 }}>
        <CircularProgress size={24} />
      </Box>
    );
  }

  if (error) {
    return (
      <Box sx={{ p: 2 }}>
        <Typography variant="body2" color="error">{error}</Typography>
      </Box>
    );
  }

  if (!stats) {
    return (
      <Box sx={{ p: 2, textAlign: 'center' }}>
        <Typography variant="body2" color="text.secondary">
          No match stats available yet. Track all rallies to generate statistics.
        </Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ height: '100%', overflow: 'auto', p: 1 }}>
      <MatchOverview stats={stats} />
      <Divider sx={{ my: 1 }} />
      <TeamStatsTable stats={stats} />
      <Divider sx={{ my: 1 }} />
      <PlayerStatsTable stats={stats} playerNames={playerNames} />
      {stats.landingHeatmaps && (
        <>
          <Divider sx={{ my: 1 }} />
          <LandingHeatmapSection stats={stats} />
        </>
      )}
    </Box>
  );
}
