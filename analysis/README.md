# RallyCut

Volleyball video analysis CLI - auto-cut dead time from recordings.

## Features

- **Auto-cut dead time**: Automatically remove no-play segments from recordings using VideoMAE ML model
- **Proxy mode**: 480p proxy videos for faster ML processing

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
```

## Commands

### Dead Time Removal (`cut`)

Remove no-play segments from volleyball recordings:

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

# Disable proxy for full-resolution analysis (slower)
uv run rallycut cut match.mp4 --no-proxy

# Adjust padding around play segments
uv run rallycut cut match.mp4 --padding 2.0

# Set minimum play duration to filter short rallies
uv run rallycut cut match.mp4 --min-play 3.0

# Analyze faster with larger stride (less accurate)
uv run rallycut cut match.mp4 --stride 32

# Analyze only first 2 minutes (for testing)
uv run rallycut cut match.mp4 --limit 120
```

### Performance Profiling (`profile`)

Analyze performance bottlenecks in the processing pipeline:

```bash
# Profile processing performance
uv run rallycut profile match.mp4

# Profile with specific segment
uv run rallycut profile match.mp4 --limit 60
```

## Command Reference

| Command | Description |
|---------|-------------|
| `rallycut cut <video>` | Remove dead time from video |
| `rallycut profile <video>` | Analyze processing performance |

## Global Options

All commands support:
- `--gpu / --no-gpu`: Enable/disable GPU acceleration (auto-detected by default)
- `--help`: Show command help

## Requirements

- Python 3.11+
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
├── analysis/         # ML pipeline, game state classification
├── processing/       # Video cutting, FFmpeg export
├── statistics/       # Stats aggregation

lib/volleyball_ml/    # ML model adapters (VideoMAE)
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

RallyCut uses **VideoMAE** for game state classification (SERVICE, PLAY, NO_PLAY).

Models are adapted from [volleyball_analytics](https://github.com/masouduut94/volleyball_analytics).

## Known Limitations

- Game state detection accuracy is optimized for standard volleyball footage

## License

MIT
