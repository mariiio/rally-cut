---
name: video-analysis
description: Run RallyCut video analysis commands - cut dead time, extract highlights, add overlays, get statistics. Use when processing volleyball videos or configuring analysis parameters.
allowed-tools: Bash, Read
---

# RallyCut Video Analysis

## Primary Commands
```bash
uv run rallycut cut <video.mp4>        # Remove dead time
uv run rallycut highlights <video.mp4> # Extract top rallies
uv run rallycut overlay <video.mp4>    # Ball tracking overlay
uv run rallycut stats <video.mp4>      # Game statistics
uv run rallycut profile <video.mp4>    # Performance profiling
```

## Common Parameters for `cut`
- `--output`, `-o` - Output file path
- `--dry-run` - Analyze without generating video
- `--json` - Export segments to JSON
- `--stride` - Frame sampling interval (auto-adjusted for FPS)
- `--padding` - Seconds of padding around play segments
- `--min-gap` - Minimum gap to bridge between segments

## Performance Options
- `--no-gpu` - Force CPU processing
- `--proxy/--no-proxy` - Use 480p proxy for faster analysis (default: on)
- `--no-cache` - Force re-analysis even if cached

## Output Formats
- `--json segments.json` - Export detected segments
- `--load-segments segments.json` - Load pre-computed segments
