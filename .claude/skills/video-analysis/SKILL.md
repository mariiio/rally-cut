---
name: video-analysis
description: Run RallyCut video analysis commands - cut dead time, extract highlights, add overlays, profile performance. Use when processing volleyball videos or configuring analysis parameters. (project)
allowed-tools: Bash, Read
---

# RallyCut Video Analysis

These are user-facing video processing commands. For ML evaluation/diagnosis, use the `ml-experiment`, `tracking-diagnosis`, `court-detection`, or `contact-detection` skills instead.

## Primary Commands

```bash
cd analysis
uv run rallycut cut <video.mp4>        # Remove dead time (auto-selects TemporalMaxer)
uv run rallycut highlights <video.mp4> # Extract top rallies
uv run rallycut overlay <video.mp4>    # Ball tracking overlay
uv run rallycut profile <video.mp4>    # Performance profiling
```

## Common Options for `cut`

```bash
uv run rallycut cut video.mp4 --output out.mp4    # Output file path
uv run rallycut cut video.mp4 --dry-run            # Analyze without generating video
uv run rallycut cut video.mp4 --json segments.json # Export detected segments
uv run rallycut cut video.mp4 --debug              # Timeline visualization + diagnostics
uv run rallycut cut video.mp4 --limit 60           # Analyze first 60s only
uv run rallycut cut video.mp4 --temporal-maxer     # Force TemporalMaxer pipeline
uv run rallycut cut video.mp4 --heuristics         # Force heuristics pipeline
```

## Performance Options

- `--no-gpu` - Force CPU processing
- `--proxy/--no-proxy` - Use 480p proxy for faster analysis (default: on)
- `--no-cache` - Force re-analysis even if cached
- `--stride` - Frame sampling interval (auto-adjusted for FPS)

## Other Commands

```bash
uv run rallycut track-players video.mp4            # Player tracking
uv run rallycut detect-court video.mp4             # Court corner detection
uv run rallycut match-players <video-id>           # Cross-rally player matching
uv run rallycut remap-track-ids <video-id>         # Remap track IDs to player IDs
uv run rallycut reattribute-actions <video-id>     # Re-attribute player actions
```
