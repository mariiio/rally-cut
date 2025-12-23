# RallyCut

Beach volleyball video analysis CLI - auto-cut dead time, generate highlights, and add ball tracking overlays.

## Features

- **Auto-cut dead time**: Automatically remove no-play segments from recordings using VideoMAE ML model
- **Highlight generation**: Auto-generate highlight reels from the longest rallies
- **Ball tracking overlay**: Visual ball trajectory overlay on exported videos
- **Quick mode**: Fast motion-based analysis when ML precision isn't needed

## Installation

```bash
# Clone the repository
git clone https://github.com/rallycut/rallycut.git
cd rallycut

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

## Quick Start

```bash
# Remove dead time from a video
uv run rallycut cut match.mp4

# Preview detected segments without generating output
uv run rallycut cut match.mp4 --dry-run

# Generate highlight reel (top 5 rallies)
uv run rallycut highlights match.mp4 -n 5
```

## Commands

### Dead Time Removal (`cut`)

Remove no-play segments from beach volleyball recordings:

```bash
# Basic usage - outputs to match_cut.mp4
uv run rallycut cut match.mp4

# Specify output path
uv run rallycut cut match.mp4 -o clean_match.mp4

# Preview segments only (no video output)
uv run rallycut cut match.mp4 --dry-run

# Save segment data to JSON
uv run rallycut cut match.mp4 --json segments.json

# Load segments from JSON (skip analysis)
uv run rallycut cut match.mp4 --segments segments.json

# Quick mode (motion-based, faster but less accurate)
uv run rallycut cut match.mp4 --quick

# Adjust padding around play segments
uv run rallycut cut match.mp4 --padding 2.0

# Set minimum play duration to filter short rallies
uv run rallycut cut match.mp4 --min-play 3.0

# Analyze faster with larger stride (less accurate)
uv run rallycut cut match.mp4 --stride 32

# Analyze only first 2 minutes (for testing)
uv run rallycut cut match.mp4 --limit 120
```

### Highlight Generation (`highlights`)

Generate highlight reel from top rallies:

```bash
# Generate top 5 highlights
uv run rallycut highlights match.mp4 -n 5

# Preview rankings without generating video
uv run rallycut highlights match.mp4 --dry-run

# Export as individual clips
uv run rallycut highlights match.mp4 --individual -o clips/

# Order by score instead of chronologically
uv run rallycut highlights match.mp4 --by-score

# Adjust padding around rallies
uv run rallycut highlights match.mp4 --padding-start 1.0 --padding-end 2.0
```

### Ball Tracking Overlay (`overlay`)

Add ball trajectory visualization:

```bash
# Add overlay to entire video
uv run rallycut overlay match.mp4

# Process specific time range
uv run rallycut overlay match.mp4 -s 12 -e 25 -o rally1.mp4

# Adjust trail length and smoothing
uv run rallycut overlay match.mp4 --trail 20 --smooth 2.0

# Lower confidence threshold for more detections
uv run rallycut overlay match.mp4 --confidence 0.2
```

### Statistics (`stats`)

Display rally statistics (experimental - best for indoor volleyball):

```bash
# Show statistics table
uv run rallycut stats match.mp4

# Export to JSON
uv run rallycut stats match.mp4 --json stats.json
```

## Command Reference

| Command | Description |
|---------|-------------|
| `rallycut cut <video>` | Remove dead time from video |
| `rallycut highlights <video>` | Generate highlight reel |
| `rallycut overlay <video>` | Add ball tracking overlay |
| `rallycut stats <video>` | Extract game statistics |

## Global Options

All commands support:
- `--gpu / --no-gpu`: Enable/disable GPU acceleration (auto-detected by default)
- `--help`: Show command help

## Requirements

- Python 3.10+
- FFmpeg (for video processing)
- ~500MB disk space for ML models

### Optional

- NVIDIA GPU with CUDA for faster processing
- Apple Silicon (MPS) also supported

## Architecture

```
rallycut/
├── cli/              # Typer CLI commands
├── core/             # Domain models, config, video handling
├── analysis/         # ML pipeline, game state, action detection
├── processing/       # Video cutting, highlights
├── tracking/         # Ball tracking with Kalman filter
├── statistics/       # Stats aggregation
└── output/           # Overlay rendering

lib/volleyball_ml/    # ML model adapters (VideoMAE, YOLO)
```

## Development

```bash
# Install dev dependencies
uv sync --extra dev

# Run tests
uv run python -m pytest

# Run tests with verbose output
uv run python -m pytest -v
```

## ML Models

RallyCut uses two main ML models:

1. **VideoMAE** - Game state classification (SERVICE, PLAY, NO_PLAY)
2. **YOLOv8** - Ball detection for trajectory overlay

Models are adapted from [volleyball_analytics](https://github.com/masouduut94/volleyball_analytics).

## Known Limitations

- Action detection (stats command) is trained on indoor volleyball and has limited accuracy on beach volleyball footage
- Ball tracking works best with clear ball visibility and may lose tracking when ball goes out of frame

## License

MIT
