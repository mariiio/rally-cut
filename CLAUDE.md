# RallyCut

Beach volleyball video analysis CLI. Uses ML (VideoMAE, YOLOv8) to detect game states, remove dead time, generate highlights, and track ball trajectories.

## Stack

- Python 3.11+, uv package manager
- PyTorch + Transformers (VideoMAE), Ultralytics (YOLO)
- Typer CLI, Pydantic config, FFmpeg, OpenCV

## Commands

```bash
# Run CLI
uv run rallycut cut <video.mp4>        # Remove dead time
uv run rallycut highlights <video.mp4> # Extract top rallies
uv run rallycut overlay <video.mp4>    # Ball tracking overlay
uv run rallycut stats <video.mp4>      # Game statistics

# Development
uv run pytest tests                    # Run tests (excludes slow ML tests)
uv run pytest tests --run-slow         # Include slow ML inference tests
uv run mypy rallycut/                  # Type check
uv run ruff check rallycut/            # Lint
```

## Structure

```
rallycut/
├── cli/commands/    # Typer commands (cut, stats, highlights, overlay)
├── core/            # Config (Pydantic), models, Video wrapper, caching
├── analysis/        # GameStateAnalyzer, MotionDetector, TwoPassAnalyzer
├── processing/      # VideoCutter, HighlightScorer, FFmpegExporter
├── tracking/        # BallTracker (YOLO + Kalman filter)
├── service/         # Cloud detection service (Modal deployment)
lib/volleyball_ml/   # ML model wrappers (VideoMAE, YOLO)
tests/
├── unit/            # Fast tests with mocked ML
├── integration/     # Full pipeline tests
└── conftest.py      # @pytest.mark.slow for ML tests
```

## Code Style

- Ruff: E, F, I, N, W, UP rules (line length 100, E501 ignored)
- MyPy: strict mode (`disallow_untyped_defs = true`)
- Type hints required on all functions

## Key Patterns

- **Lazy loading**: ML models loaded on first use, not import
- **Two-pass analysis**: Motion pre-filter (fast) + ML refinement (accurate)
- **Proxy videos**: 480p cached copies for faster ML processing
- **Sequential reading**: Use `video.iter_frames()` not seeking
- **Config**: Nested Pydantic BaseSettings with YAML/env var support

## Testing

- Mark slow tests: `@pytest.mark.slow` for ML inference
- Mock ML models in unit tests to avoid downloads
- Test fixtures in `tests/fixtures/`

## Config

Config loaded from (priority order):
1. `./rallycut.yaml`
2. `~/.config/rallycut/rallycut.yaml`
3. Environment vars: `RALLYCUT_MOTION__HIGH_THRESHOLD`
4. Defaults in `core/config.py`
