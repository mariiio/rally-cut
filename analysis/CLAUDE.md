# RallyCut Analysis

Volleyball video analysis CLI. Uses ML (VideoMAE) to detect game states and remove dead time.

## Stack

- Python 3.11+, uv package manager
- PyTorch + Transformers (VideoMAE)
- Typer CLI, Pydantic config, FFmpeg, OpenCV

## Commands

```bash
# Core commands (binary head + decoder is default when features cached)
uv run rallycut cut <video.mp4>                 # Auto-selects best pipeline
uv run rallycut cut <video.mp4> --heuristics    # Force heuristics (57% F1)
uv run rallycut cut <video.mp4> --binary-head   # Force binary head (84% F1 at IoU=0.4)
uv run rallycut cut <video.mp4> --model beach   # Use beach volleyball model
uv run rallycut profile <video.mp4>             # Performance profiling

# One-time feature extraction (enables 84% F1 pipeline at IoU=0.4)
uv run rallycut train extract-features --stride 48  # Required for binary head

# Useful options for cut
uv run rallycut cut video.mp4 --debug           # Timeline visualization + diagnostics
uv run rallycut cut video.mp4 --profile         # Performance breakdown
uv run rallycut cut video.mp4 --json            # Export segments as JSON
uv run rallycut cut video.mp4 --segments s.json # Load pre-computed segments
uv run rallycut cut video.mp4 --limit 60        # Analyze first 60s only
uv run rallycut cut video.mp4 --no-ball-boundary  # Disable ball-based boundary refinement (on by default)

# Training (beach model fine-tuning) - FULL WORKFLOW
rm -rf training_data/                                 # Clean old local data
uv run rallycut train export-dataset --name beach_v1  # Export labeled data from DB
uv run rallycut train prepare                         # Generate samples + extract frames (~6.6GB)
modal volume rm -r rallycut-training training_data/   # Clean old Modal data (IMPORTANT!)
modal volume put -f rallycut-training training_data/ training_data/  # Upload fresh data
uv run rallycut train modal --epochs 30 --fresh       # Train on T4 GPU (~4hrs, ~$0.59/hr)
uv run rallycut train modal --download                # Download trained model
uv run rallycut train modal --cleanup                 # Delete from Modal (~$0.75/GB/mo)

# Training optimizations (applied by default):
# - 2-class training (SERVICE merged into PLAY) - reduces class confusion
# - Video-level train/val split prevents data leakage (honest validation)
# - 9/12 encoder layers frozen to prevent catastrophic forgetting
# - Lower learning rate (5e-6) for stable fine-tuning
# - Pre-extracted frames for 15x faster training (~1.4s/step vs 20s/step)
# - Use --freeze-layers 0 and --lr 1e-5 for full fine-tuning (not recommended)

# Adding new labeled videos (RECOMMENDED WORKFLOW)
# Step 1: Export and backup
uv run rallycut train export-dataset --name beach_v3  # Export all labeled data from DB
uv run rallycut train push --name beach_v3            # Back up to S3 (deduplicates videos)

# Step 2a: Retrain temporal model (fast, local, recommended first)
uv run rallycut train extract-features --stride 48    # Extract features for new videos
uv run rallycut train temporal --model v1 --epochs 50 # Retrain temporal smoother

# Step 2b: Fine-tune VideoMAE (slow, GPU required, only if needed)
rm -rf training_data/
uv run rallycut train prepare                         # Generate samples from all videos
modal volume rm -r rallycut-training training_data/   # Clean old Modal data
modal volume put -f rallycut-training training_data/ training_data/  # Upload fresh
uv run rallycut train modal --upload-model            # Upload existing model weights
uv run rallycut train modal --resume-from-model --lr 1e-6 --epochs 10 --fresh
uv run rallycut train modal --download --cleanup      # Download and clean up

# S3 backup/restore (survives DB resets and MinIO clears)
uv run rallycut train push --name beach_v2            # Upload dataset to S3
uv run rallycut train pull --name beach_v2            # Download dataset from S3
uv run rallycut train restore --name beach_v2         # Re-import ground truth into DB
uv run rallycut train restore --name beach_v2 --upload-to-app-s3  # + upload to MinIO
uv run rallycut train restore --name beach_v2 --dry-run  # Preview without changes
uv run rallycut train list-remote                     # List datasets backed up in S3

# Training is preemption-resilient:
# - Auto-retries up to 2 times on GPU preemption
# - Checkpoints saved every 100 steps (~5 min max loss)
# - Resumes from latest checkpoint automatically

# Temporal model training (DEPRECATED - use binary head instead)
# Binary head achieves 84% F1 (IoU=0.4) vs temporal's 65% F1
# uv run rallycut train temporal --model v1 --epochs 50

# Binary head training (recommended)
uv run rallycut train export-dataset --name beach_v3  # Export labeled data
uv run rallycut train extract-features --stride 48    # Extract VideoMAE features
uv run rallycut train binary-head --epochs 50         # Train binary head classifier
# Model saved to weights/binary_head/best_binary_head.pt

# Evaluation
uv run rallycut evaluate                              # Evaluate (auto-selects binary head if features cached)
uv run rallycut evaluate --binary-head                # Force binary head evaluation (84% F1 at IoU=0.4)
uv run rallycut evaluate --heuristics                 # Force heuristics evaluation (57% F1)
uv run rallycut evaluate --model beach --iou 0.5      # Evaluate beach model
uv run rallycut evaluate tune-decoder --iou 0.4       # Grid search decoder parameters

# Development
uv run pytest tests                    # Run tests (excludes slow ML tests)
uv run pytest tests --run-slow         # Include slow ML inference tests
uv run mypy rallycut/                  # Type check
uv run ruff check rallycut/            # Lint
```

## Structure

```
rallycut/
├── cli/commands/    # Typer commands (cut, profile, train, evaluate)
├── core/            # Config, models, Video wrapper, caching, profiler
├── analysis/        # GameStateAnalyzer (VideoMAE ML classifier)
├── processing/      # VideoCutter, FFmpegExporter
├── temporal/        # Temporal models and binary head pipeline
│   ├── binary_head.py        # Binary head classifier (recommended)
│   ├── deterministic_decoder.py  # Decoder for binary head output
│   ├── boundary_refinement.py    # Fine-stride boundary refinement
│   ├── features.py           # VideoMAE feature extraction and caching
│   ├── inference.py          # Model loading and inference
│   ├── models.py             # v1/v2/v3 temporal models (deprecated)
│   └── training.py           # Training loop with validation
├── tracking/        # Ball and player tracking
│   ├── ball_tracker.py       # YOLO-based ball detection
│   ├── ball_features.py      # Ball phase detection, server ID, reactivity scoring
│   ├── player_tracker.py     # YOLO + ByteTrack player tracking
│   ├── player_filter.py      # Multi-stage player filtering with court/ball scoring
│   ├── player_features.py    # Appearance extraction (skin tone, jersey, proportions)
│   └── match_tracker.py      # Cross-rally player ID consistency
├── court/           # Court calibration
│   └── calibration.py        # Homography for image→court projection
├── statistics/      # Match statistics aggregation
│   └── aggregator.py         # Rally grouping, action counts
├── labeling/        # Ground truth labeling with Label Studio
│   ├── ground_truth.py      # GroundTruthPosition/Result data structures
│   └── studio_client.py     # Label Studio API client
├── evaluation/      # Ground truth loading, metrics, parameter tuning
├── service/         # Cloud detection (Modal deployment)
│   ├── platforms/modal_app.py       # Modal GPU function
│   └── player_tracking_runner.py    # Local player tracking subprocess
lib/volleyball_ml/   # ML model wrappers (VideoMAE)
tests/
├── unit/            # Fast tests with mocked ML
└── integration/     # Full pipeline tests
```

## Configuration

Nested Pydantic config with YAML/env var support. Key sections:

| Section | Key Settings |
|---------|-------------|
| `game_state` | `stride=48` (frames between samples), `window_size=16`, `batch_size=8`, `temporal_model_path`, `temporal_model_version` |
| `segment` | `min_play_duration=1.0`, `min_gap=5.0`, `rally_continuation=2.0` |
| `proxy` | 480p@30fps normalized for faster ML |

**Loading priority:**
1. `./rallycut.yaml` (current dir)
2. `~/.config/rallycut/rallycut.yaml`
3. Environment: `RALLYCUT_GAME_STATE__STRIDE=32` (nested with `__`)
4. Defaults in `core/config.py`

## Caching

Analysis results cached in `~/.cache/rallycut/analysis/`:
- **Cache key**: SHA256(file_signature + stride + proxy_mode)
- **File signature**: filename + size + mtime + first 1MB hash
- **Invalidation**: Automatic when video changes

Proxy videos cached in `~/.cache/rallycut/proxies/`.

## Model Variants

Two model variants available via `--model` flag:

| Model | Weights | Use Case |
|-------|---------|----------|
| `beach` (default) | `weights/videomae/game_state_classifier/` | Beach volleyball |
| `indoor` | Same as beach (tuned heuristics) | Indoor volleyball courts |

**Note:** Both variants use the same model weights with different post-processing heuristics. Beach has stricter thresholds to prevent over-merging rallies.

Each model has optimized post-processing heuristics defined in `MODEL_PRESETS` (see `core/config.py`).

## Detection Pipelines

Three detection pipelines are available, with binary head + decoder as the recommended default:

| Pipeline | F1 (IoU=0.5) | F1 (IoU=0.4) | Overmerge | Command |
|----------|--------------|--------------|-----------|---------|
| Binary Head + Decoder (default) | ~78% | 84% | 0% | `rallycut cut video.mp4` |
| Heuristics (fallback) | 57% | - | ~10% | `rallycut cut video.mp4 --heuristics` |
| Temporal (deprecated) | 65% | - | ~5% | `rallycut cut video.mp4 --experimental-temporal` |

**Note:** IoU=0.4 better reflects detection accuracy when labeling marks serve toss start (model detects ~2s later when play begins).

**Pipeline auto-selection:**
1. If `--binary-head` flag: use binary head + decoder
2. If `--experimental-temporal` flag: use temporal model (deprecated)
3. If `--heuristics` flag: use heuristics
4. Auto: use binary head if features cached and model exists, else heuristics

**Enabling 84% F1 pipeline (IoU=0.4):**
```bash
# One-time feature extraction
uv run rallycut train extract-features --stride 48
```

## Processing Heuristics

The heuristics pipeline applies three post-processing heuristics to fix ML errors:

1. **Confidence extension**: Extends PLAY segments at boundaries where `play_confidence > threshold`
2. **Rally continuation**: Bridges NO_PLAY gaps within a rally (fixes mid-rally false negatives)
3. **Density filtering**: Removes segments with sparse active windows (filters noise)

**Model-specific thresholds:**

| Parameter | Indoor | Beach |
|-----------|--------|-------|
| `min_play_duration` | 1.0s | 1.0s |
| `rally_continuation_seconds` | 2.0s | 1.5s |
| `boundary_confidence_threshold` | 0.35 | 0.4 |
| `min_active_density` | 0.25 | 0.3 |
| `min_gap_seconds` | 5.0s | 3.0s |

Beach heuristics are tuned to be more discriminative (higher thresholds) to prevent over-merging of rallies.

See `processing/cutter.py` and `core/config.py` for implementation.

## Cloud Detection (Modal)

Production ML runs on Modal GPUs:

```bash
# Deploy
modal deploy rallycut/service/platforms/modal_app.py

# Local testing
uv run python -m rallycut.service.local_runner
```

API triggers Modal via webhook. Results posted back on completion.

## Player Tracking Filtering

Multi-stage filtering to identify active players and exclude non-players. See `tracking/player_filter.py`.

**Stages:**
1. Track length filter (>10% of video)
2. Hard filters: sideline position (<0.20 or >0.80), court presence (>50%), stationary objects
3. Score computation: length, court_presence, ball_proximity, engagement, spread, reactivity
4. Top-4 selection by combined score

**Key insight:** Positively identify players by volleyball behaviors (ball engagement, court coverage, movement) rather than detecting "non-players".

## Track Merging

Merges fragmented ByteTrack IDs using velocity prediction and position/size similarity. Thresholds are video-agnostic (seconds, normalized coordinates). See `tracking/player_filter.py`.

## Ball Phase Detection

Detects game phases (SERVE, ATTACK, DEFENSE, TRANSITION) from ball velocity patterns. Identifies server by finding player closest to ball during velocity spike in first 3 seconds. See `tracking/ball_features.py`.

**Output:** `ballPhases`, `serverInfo`, `ballPositions` for overlay visualization.

## Cross-Rally Player Consistency

Maintains consistent player IDs (1-4) across match using appearance features (skin tone, jersey color, body proportions). Detects side switches by comparing assignment costs. See `tracking/player_features.py` and `tracking/match_tracker.py`.

## Ground Truth Labeling

Label Studio integration for tracking accuracy evaluation. See `labeling/studio_client.py`.

```bash
# Setup (one-time)
pip3 install label-studio
python3 -m label_studio start  # Opens at http://localhost:8080
export LABEL_STUDIO_API_KEY=your_token_here  # From Profile → Account & Settings

# Workflow
uv run rallycut track-players video.mp4 -o tracking.json
uv run rallycut label open video.mp4 -p tracking.json  # Opens browser, shows task ID
# Correct annotations in Label Studio, click Submit
uv run rallycut label save 123 -o ground_truth.json
uv run rallycut compare-tracking tracking.json ground_truth.json

# Evaluate from database (after ground truth is synced)
uv run rallycut evaluate-tracking --all              # Evaluate all labeled rallies
uv run rallycut evaluate-tracking -r <rally-id>      # Evaluate specific rally
uv run rallycut evaluate-tracking -v <video-id>      # Evaluate all in video
uv run rallycut evaluate-tracking --all --per-player # Show per-player breakdown
uv run rallycut evaluate-tracking --all -e           # Show error analysis
uv run rallycut evaluate-tracking --all -o out.json  # Export metrics to JSON
```

| Command | Purpose |
|---------|---------|
| `rallycut label open` | Open video with pre-filled predictions |
| `rallycut label save` | Export annotations as ground truth JSON |
| `rallycut compare-tracking` | Compute MOT metrics from JSON files |
| `rallycut evaluate-tracking` | Evaluate tracking from database with detailed breakdowns |

## Code Style

- Ruff: E, F, I, N, W, UP rules (line length 100, E501 ignored)
- MyPy: strict mode (`disallow_untyped_defs = true`)
- Type hints required on all functions

## Key Patterns

- **Lazy loading**: ML models loaded on first use, not import
- **Proxy videos**: 480p@30fps cached copies for faster ML (10-100x speedup)
- **Training proxies**: Same 480p proxies used for training (45% smaller than originals)
- **Temporal smoothing**: Median filter on ML results fixes isolated errors
- **Sequential reading**: Use `video.iter_frames()` not seeking
- **FPS normalization**: High-FPS videos (60fps+) subsampled to 30fps for VideoMAE

## Caveats

- **First run downloads ~500MB** of models from HuggingFace
- **Stride scales with FPS**: `--auto-stride` adjusts stride proportionally (stride 32 @ 30fps = stride 64 @ 60fps)
- **ONNX disabled on MPS**: CoreML provider has compatibility issues
- **Avoid seeking**: VideoMAE needs sequential frame reading for proper temporal dynamics
