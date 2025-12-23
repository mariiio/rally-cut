# Implementation Plan: RallyCut MVP Core

**Feature Branch**: `001-mvp-core`
**Created**: 2025-01-22
**Status**: Ready for Implementation

## Technical Context

### Tech Stack
- **Language**: Python 3.11+
- **CLI Framework**: Typer with Rich for terminal UI
- **ML Framework**: PyTorch 2.0+, Transformers (VideoMAE), Ultralytics (YOLOv8)
- **Video Processing**: OpenCV, FFmpeg (via ffmpeg-python)
- **Configuration**: Pydantic, pydantic-settings
- **Ball Tracking**: Kalman filter (filterpy), scipy
- **Package Manager**: uv

### Dependencies (from pyproject.toml)
```
typer[all]>=0.9.0
rich>=13.0.0
pydantic>=2.0.0
pydantic-settings>=2.0.0
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.24.0
scipy>=1.11.0
filterpy>=1.4.5
ffmpeg-python>=0.2.0
httpx>=0.25.0
platformdirs>=4.0.0
```

### Project Structure
```
rallycut/
├── cli/
│   ├── main.py              # CLI entry point (exists)
│   └── commands/            # Command implementations (skeletons exist)
├── core/
│   ├── video.py             # Video container abstraction
│   ├── models.py            # Domain models (GameState, Action, Rally, etc.)
│   └── config.py            # Configuration management
├── analysis/
│   ├── pipeline.py          # Main analysis orchestrator
│   ├── game_state.py        # VideoMAE game state classifier
│   ├── action_detector.py   # YOLOv8 action detection
│   └── models/
│       └── manager.py       # Model loading & lifecycle
├── processing/
│   ├── cutter.py            # Dead time removal
│   ├── exporter.py          # FFmpeg video export
│   └── highlights.py        # Highlight generation
├── tracking/
│   ├── ball_tracker.py      # Kalman filter tracking
│   └── trajectory.py        # Trajectory smoothing
├── statistics/
│   └── aggregator.py        # Statistics collection
└── output/
    ├── json_export.py       # JSON output
    └── overlay.py           # Video overlay rendering

lib/volleyball_ml/
├── video_mae.py             # Adapted VideoMAE classifier
└── yolo_detector.py         # Adapted YOLO detector
```

## Constitution Check

| Principle | Compliance | Notes |
|-----------|------------|-------|
| Privacy First | ✓ | All processing local, no network calls |
| CLI First | ✓ | Typer CLI fully functional before implementation |
| Modular Design | ✓ | Each component independently testable |
| Beach Volleyball Focus | ✓ | Models tuned for 2v2 play |
| Graceful Degradation | ✓ | CPU fallback, configurable batch sizes |

## Architecture Decisions

### Decision 1: Selective Adaptation over Submodule
- **Decision**: Copy and adapt ML code from volleyball_analytics
- **Rationale**: Avoids pulling in FastAPI/database dependencies
- **Alternatives**: Git submodule (rejected - too many unused deps)

### Decision 2: Chunk-Based Processing
- **Decision**: Process videos in 5-minute chunks
- **Rationale**: Memory efficiency for 1-2 hour videos
- **Trade-off**: Slight complexity for boundary handling

### Decision 3: Lazy Model Loading
- **Decision**: Load models only when needed
- **Rationale**: Fast CLI startup, memory efficiency
- **Implementation**: ModelManager with caching

### Decision 4: FFmpeg for Video Export
- **Decision**: Use FFmpeg directly via subprocess
- **Rationale**: Hardware acceleration, maximum control
- **Alternative**: MoviePy (rejected - slower, less control)

## Implementation Phases

### Phase 1: Core Infrastructure
- Domain models (GameState, Action, Rally, BallPosition)
- Configuration management
- Video abstraction layer

### Phase 2: ML Integration
- Model manager with download/cache
- VideoMAE adapter for game state classification
- YOLOv8 adapter for action detection

### Phase 3: Analysis Pipeline
- Pipeline orchestrator
- Chunk-based processing
- Statistics aggregation

### Phase 4: Video Processing
- Dead time cutter (FFmpeg integration)
- Ball tracker (Kalman filter)
- Trajectory overlay renderer

### Phase 5: CLI Integration
- Wire up existing command skeletons
- Progress reporting
- JSON/table output formatting

## Implementation Status (Updated 2025-12-22)

### Completed
- **Phase 1: Core Infrastructure** - All domain models, config, video abstraction
- **Phase 3: Dead Time Removal** - Full `cut` command with ML and motion detection
- **Phase 4: Statistics** - Complete with volleyball-trained YOLO weights

### Cut Command Features
```bash
rallycut cut video.mp4                    # Full ML processing
rallycut cut video.mp4 --dry-run          # Analysis only, no export
rallycut cut video.mp4 --quick            # Fast motion detection mode
rallycut cut video.mp4 --limit 120        # Test on first 2 minutes
rallycut cut video.mp4 --segments in.json # Use pre-defined segments
rallycut cut video.mp4 --json out.json    # Export detected segments
```

### ML Optimization Implemented
1. **Batch Processing** - Process 8 windows per forward pass
2. **Local Weights** - Load volleyball-trained model from `weights/videomae/game_state_classifier/`
3. **Configurable Stride** - Trade accuracy for speed (8=accurate, 16=fast)
4. **Frame Resizing** - 224x224 input for faster inference

### Recommended Settings
```bash
# High accuracy (slower)
rallycut cut video.mp4 --stride 8 --padding 2.0 --min-play 1.5

# Balanced
rallycut cut video.mp4 --stride 16 --padding 1.5

# Fast preview
rallycut cut video.mp4 --quick --dry-run
```

### Stats Command Features
```bash
rallycut stats video.mp4                   # Detect actions
rallycut stats video.mp4 -o stats.json     # Save to JSON
rallycut stats video.mp4 --limit 120       # First 2 minutes
rallycut stats video.mp4 --segments seg.json  # Use known play segments
```

**Model**: Volleyball-trained YOLO from [volleyball_analytics](https://github.com/masouduut94/volleyball_analytics)
**Classes**: ball, block, receive, set, spike, serve

### Not Yet Implemented
- Highlights command (rally ranking)
- Overlay command (ball tracking)
- Analyze command (full pipeline)
- Model download manager

## External References

- volleyball_analytics: https://github.com/masouduut94/volleyball_analytics
- VideoMAE: https://huggingface.co/docs/transformers/model_doc/videomae
- YOLOv8: https://docs.ultralytics.com/
