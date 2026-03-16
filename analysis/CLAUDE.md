# RallyCut Analysis

Volleyball video analysis CLI. Uses ML (VideoMAE) to detect game states and remove dead time.

## Stack

- Python 3.11+, uv package manager
- PyTorch + Transformers (VideoMAE)
- Typer CLI, Pydantic config, FFmpeg, OpenCV

## Commands

```bash
# Core commands (TemporalMaxer is default when model+features exist)
uv run rallycut cut <video.mp4>                 # Auto-selects best pipeline
uv run rallycut cut <video.mp4> --temporal-maxer # Force TemporalMaxer (95% LOO F1)
uv run rallycut cut <video.mp4> --heuristics    # Force heuristics (57% F1)
uv run rallycut cut <video.mp4> --model beach   # Use beach volleyball model
uv run rallycut profile <video.mp4>             # Performance profiling

# Pre-extract features (optional — TemporalMaxer auto-extracts on first run)
uv run rallycut train extract-features --stride 12

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

# Step 2a: Retrain TemporalMaxer (fast, local, recommended first)
uv run rallycut train extract-features --stride 12    # Extract features for new videos
uv run rallycut train temporal-maxer --epochs 50      # Retrain TemporalMaxer

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

# Model weight backup (survives local machine loss and Modal volume clears)
uv run rallycut train push-weights                      # Push trained weights (latest)
uv run rallycut train push-weights --name pre-retrain   # Named snapshot + update latest
uv run rallycut train push-weights --include-external   # Also backup external weights
uv run rallycut train push-weights --dry-run            # Preview what would upload
uv run rallycut train pull-weights                      # Pull latest weights
uv run rallycut train pull-weights --name pre-retrain   # Pull specific snapshot
uv run rallycut train list-weights                      # List snapshots in S3

# Training is preemption-resilient:
# - Auto-retries up to 2 times on GPU preemption
# - Checkpoints saved every 100 steps (~5 min max loss)
# - Resumes from latest checkpoint automatically

# WASB HRNet fine-tuning (improve ball detection on beach volleyball)
# IMPORTANT: Always re-export before retraining. The export writes a manifest.json
# that lists GT rally IDs — training uses it to hold GT rallies out of the training
# split. Stale exports will cause the upload step to error (missing manifest).
uv run python -m experiments.pseudo_label_export \
    --output-dir experiments/wasb_pseudo_labels \
    --all-tracked --extract-frames                    # Export pseudo-labels + manifest
uv run rallycut train wasb-modal --upload           # Upload data + pretrained weights to Modal
uv run rallycut train wasb-modal --epochs 30 --fresh  # Train on A10G GPU (~$1.10/hr)
uv run rallycut train wasb-modal --download         # Download fine-tuned model (auto-deletes ONNX)
uv run python scripts/eval_wasb.py                  # Evaluate WASB
uv run rallycut train wasb-modal --cleanup          # Delete from Modal volume

# TemporalMaxer training (95% LOO F1 at IoU=0.4)
uv run rallycut train export-dataset --name beach_v3  # Export labeled data
uv run rallycut train extract-features --stride 12    # Extract VideoMAE features
uv run rallycut train temporal-maxer --epochs 50      # Train TemporalMaxer TAS model
# Model saved to weights/temporal_maxer/best_temporal_maxer.pt

# Evaluation
uv run rallycut evaluate                              # Evaluate (auto-selects TemporalMaxer)
uv run rallycut evaluate --temporal-maxer             # Force TemporalMaxer (95% LOO F1)
uv run rallycut evaluate --heuristics                 # Force heuristics evaluation (57% F1)
uv run rallycut evaluate --model beach --iou 0.5      # Evaluate beach model

# Stratified evaluation (by video characteristics)
uv run python scripts/eval_stratified.py             # Group metrics by brightness/camera/complexity
uv run python scripts/eval_stratified.py --ball-only # Ball metrics only

# Cross-rally player matching & post-processing
uv run rallycut match-players <video-id>                 # Assign consistent player IDs 1-4
uv run rallycut match-players <video-id> -o result.json  # Export assignments to JSON
uv run rallycut match-players <video-id> --num-samples 20  # More frames per track (default: 12)
uv run rallycut match-players <video-id> -q              # Quiet mode
uv run rallycut repair-identities <video-id>             # Fix within-rally ID switches using match profiles
uv run rallycut repair-identities <video-id> --dry-run   # Preview repairs without updating DB
uv run rallycut remap-track-ids <video-id>               # Remap stored track IDs to player IDs
uv run rallycut remap-track-ids <video-id> --dry-run     # Preview changes without updating DB
uv run rallycut reattribute-actions <video-id>           # Re-attribute actions using match-level teams
uv run rallycut reattribute-actions <video-id> --min-confidence 0.80  # Stricter confidence gate

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
├── temporal/        # TemporalMaxer rally detection pipeline
│   ├── temporal_maxer/       # TemporalMaxer TAS model
│   │   ├── model.py          # Multi-scale temporal pyramid architecture
│   │   ├── training.py       # Sequence-level training loop
│   │   └── inference.py      # Inference and segment extraction
│   ├── features.py           # VideoMAE feature extraction and caching
│   └── inference.py          # Segment extraction and anti-overmerge
├── tracking/        # Ball and player tracking
│   ├── ball_tracker.py       # Ball tracker factory and data types
│   ├── ball_filter.py        # Temporal filter for ball tracking post-processing
│   ├── ball_features.py      # Ball features, server ID, reactivity scoring
│   ├── player_tracker.py     # YOLO + BoT-SORT player tracking
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
├── service/         # Cloud services (Modal deployment)
│   ├── platforms/modal_app.py       # Modal GPU detection function
│   ├── platforms/modal_tracking.py  # Modal GPU batch tracking function
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
| `game_state` | `stride=12` (frames between samples), `window_size=16`, `batch_size=8` |
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

Two detection pipelines are available:

| Pipeline | F1 (IoU=0.5) | F1 (IoU=0.4) | Overmerge | Command |
|----------|--------------|--------------|-----------|---------|
| TemporalMaxer (default) | - | 95.2% (LOO) | 0% | `rallycut cut video.mp4 --temporal-maxer` |
| Heuristics (fallback) | 57% | - | ~10% | `rallycut cut video.mp4 --heuristics` |

**Note:** IoU=0.4 better reflects detection accuracy when labeling marks serve toss start (model detects ~2s later when play begins). TemporalMaxer's F1 is honest leave-one-video-out CV (stride=12, 43 videos).

**Pipeline auto-selection:**
1. If `--temporal-maxer` flag: use TemporalMaxer (auto-extracts features if not cached)
2. If `--heuristics` flag: use heuristics
3. Auto: TemporalMaxer if model+features exist, else heuristics

**Cloud (DetectionService/Modal):** Automatically uses TemporalMaxer when weights are in the image. Features are extracted inline on first run and cached for subsequent runs.

**Enabling best pipeline (local):**
```bash
# Train TemporalMaxer (features auto-extracted on first run, or pre-extract for speed)
uv run rallycut train extract-features --stride 12  # Optional: pre-extract for faster first run
uv run rallycut train temporal-maxer --epochs 50
```

## Processing Heuristics

Fallback pipeline (57% F1) applies confidence extension, rally continuation, and density filtering. Model-specific thresholds in `MODEL_PRESETS` (`core/config.py`). Beach uses stricter thresholds to prevent over-merging. See `processing/cutter.py`.

## Cloud Services (Modal)

Production ML runs on Modal GPUs. Two separate Modal apps for independent deploys:

**Detection** (`rallycut-detection`): TemporalMaxer rally detection. Auto-enabled when
`weights/temporal_maxer/best_temporal_maxer.pt` is in the image.

**Batch Tracking** (`rallycut-tracking`): GPU-accelerated player + ball tracking.
Downloads video once, tracks all rallies on T4 GPU (~80 FPS vs 6 FPS local CPU).
Sends per-rally webhooks for progressive results. Set `MODAL_TRACKING_URL` in API env
to enable; falls back to local CPU when unset.

```bash
# Deploy
modal deploy rallycut/service/platforms/modal_app.py       # Detection
modal deploy rallycut/service/platforms/modal_tracking.py  # Batch tracking

# Local testing
uv run python -m rallycut.service.local_runner
```

API triggers Modal via webhook. Results posted back on completion.

## Player Tracking

Uses YOLO for person detection with BoT-SORT for temporal tracking. BoT-SORT with ReID (enabled by default) reduces ID switches when players cross paths. GMC (camera motion compensation) is disabled for fixed tripod cameras.

**YOLO Model Options (imgsz=1280):**
| Model | FPS | F1 | HOTA | Far Recall | Use Case |
|-------|-----|-----|------|------------|----------|
| **yolo11s** | 6.1 | 92.5% | 91.3% | 96.3% | Default - best accuracy/speed tradeoff |
| yolov8n | 7.7 | 79.4% | 80.3% | 63.5% | Faster, lower far-court recall |
| yolo11m | 2.4 | 77.0% | 82.4% | 89.4% | Mid-recall regression, slow |
| yolov8n@640 | 15.3 | 74.1% | 79.2% | 55.3% | Fastest, lowest accuracy |

Default tracker is BoT-SORT (ReID enabled). ByteTrack available as `--tracker bytetrack` (legacy).

```bash
uv run rallycut track-players video.mp4                          # Default (yolo11s + BoT-SORT)
uv run rallycut track-players video.mp4 --yolo-model yolov8n     # Faster, lower accuracy
uv run rallycut track-players video.mp4 --calibration '[...]'    # Court calibration for ROI + teams
```

## Player Tracking Filtering

Multi-stage filtering to identify 4 active players and exclude non-players. Key insight: positively identify players by volleyball behaviors (ball engagement, court coverage, movement) rather than detecting "non-players". Includes stationary background removal, primary track identification with hard/soft filters and safety net, per-frame filtering, and track merging. See `tracking/player_filter.py`.

## Ball Tracking

WASB HRNet is the sole ball tracker (86.4% match rate, 33 FPS ONNX+CoreML). Multi-stage temporal filter handles false positives (static positions, player lock-on, oscillation, exit ghosts, non-ball objects, trajectory blips) and interpolates gaps. All parameters and filter stages are in `BallFilterConfig` in `tracking/ball_filter.py`.

## Cross-Rally Player Consistency

Maintains consistent player IDs (1-4) across a match via appearance-based Hungarian assignment (2 passes). Features: upper/lower body color histograms, dominant clothing color, skin tone. Includes side switch detection and global within-team voting. See `tracking/match_tracker.py` and `tracking/player_features.py`.

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
uv run rallycut evaluate-tracking --all              # All labeled rallies (player metrics)
uv run rallycut evaluate-tracking --all --ball-only  # Ball tracking metrics only
uv run rallycut evaluate-tracking -r <rally-id>      # Specific rally

# Grid search for filter parameters (use --help for grid options)
uv run rallycut evaluate-tracking tune-filter --all --grid quick
uv run rallycut evaluate-tracking tune-ball-filter --all --grid segment-pruning
```


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
