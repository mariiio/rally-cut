---
name: video-analysis
description: Run RallyCut video analysis CLI commands - cut dead time, profile performance, run tracking, action analysis. Use when processing volleyball videos or configuring rally detection. (project)
allowed-tools: Bash, Read
---

# RallyCut Video Analysis

User-facing video processing commands. For ML evaluation/diagnosis, prefer the `ml-experiment`, `tracking-diagnosis`, `court-detection`, or `contact-detection` skills.

> Run from `analysis/` (i.e. `cd analysis` first).

## Primary commands

```bash
uv run rallycut cut <video.mp4>        # Remove dead time (auto-selects TemporalMaxer)
uv run rallycut profile <video.mp4>    # Performance profiling
```

## Common options for `cut`

```bash
uv run rallycut cut video.mp4 --output out.mp4    # Output file path
uv run rallycut cut video.mp4 --dry-run            # Analyze without generating video
uv run rallycut cut video.mp4 --json segments.json # Export detected segments
uv run rallycut cut video.mp4 --debug              # Timeline visualization + diagnostics
uv run rallycut cut video.mp4 --limit 60           # Analyze first 60s only
uv run rallycut cut video.mp4 --temporal-maxer     # Force TemporalMaxer pipeline
uv run rallycut cut video.mp4 --heuristics         # Force heuristics pipeline
```

## Performance options

- `--no-gpu` — Force CPU processing
- `--proxy/--no-proxy` — Use 480p proxy for faster analysis (default: on)
- `--no-cache` — Force re-analysis even if cached
- `--stride` — Frame sampling interval (auto-adjusted for FPS)

## Tracking & player matching

```bash
uv run rallycut track-players video.mp4            # Player tracking
uv run rallycut detect-court video.mp4             # Court corner detection
uv run rallycut match-players <video-id>           # Cross-rally player matching (1-4)
uv run rallycut remap-track-ids <video-id>         # Remap track IDs to player IDs
uv run rallycut reattribute-actions <video-id>     # Re-attribute player actions (default --min-confidence 0.70)
uv run rallycut repair-identities <video-id>       # Fix within-rally ID switches
```

## Action analysis

```bash
uv run rallycut analyze actions <video-id>         # Action classification subcommand group
uv run rallycut analyze highlights <video-id>      # Highlight ranking subcommand group
```

## Reference crops

```bash
uv run rallycut suggest-reference-crops <video>    # Rank candidate crops per player
uv run rallycut validate-reference-crops <video>   # Validate user-selected crops
uv run rallycut relabel-with-crops <video>         # Replay Pass 2 with current ref crops
```

## Verifying available commands

If a command above doesn't work, run `uv run rallycut --help` to see the actual registered subcommand list — the CLI is the source of truth.
