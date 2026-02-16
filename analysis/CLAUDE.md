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
uv run rallycut cut <video.mp4> --temporal-maxer # Force TemporalMaxer (75% LOO F1)
uv run rallycut cut <video.mp4> --binary-head   # Force binary head (84% F1 at IoU=0.4)
uv run rallycut cut <video.mp4> --heuristics    # Force heuristics (57% F1)
uv run rallycut cut <video.mp4> --model beach   # Use beach volleyball model
uv run rallycut profile <video.mp4>             # Performance profiling

# Pre-extract features (optional — TemporalMaxer auto-extracts on first run, required for binary head)
uv run rallycut train extract-features --stride 48

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
uv run rallycut train extract-features --stride 48    # Extract features for new videos
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

# Training is preemption-resilient:
# - Auto-retries up to 2 times on GPU preemption
# - Checkpoints saved every 100 steps (~5 min max loss)
# - Resumes from latest checkpoint automatically

# TrackNet ball tracker training (fine-tune on beach volleyball)
uv run rallycut train tracknet-modal --upload       # Upload training data to Modal
uv run rallycut train tracknet-modal --epochs 30 --fresh  # Train on A10G GPU (~$1.10/hr)
uv run rallycut train tracknet-modal --download     # Download best.pt + last.pt
uv run python scripts/eval_tracknet.py              # Evaluate TrackNet vs VballNet
uv run rallycut train tracknet-modal --cleanup      # Delete from Modal volume

# Temporal model training (DEPRECATED - use TemporalMaxer instead)
# uv run rallycut train temporal --model v1 --epochs 50

# TemporalMaxer training (recommended - 75% LOO F1 at IoU=0.4)
uv run rallycut train export-dataset --name beach_v3  # Export labeled data
uv run rallycut train extract-features --stride 48    # Extract VideoMAE features
uv run rallycut train temporal-maxer --epochs 50      # Train TemporalMaxer TAS model
# Model saved to weights/temporal_maxer/best_temporal_maxer.pt

# Binary head training (alternative - 84% F1 train-on-all, 57.4% LOO)
uv run rallycut train binary-head --epochs 50         # Train binary head classifier

# Evaluation
uv run rallycut evaluate                              # Evaluate (auto-selects TemporalMaxer > binary head)
uv run rallycut evaluate --temporal-maxer             # Force TemporalMaxer (75% LOO F1)
uv run rallycut evaluate --binary-head                # Force binary head evaluation (84% F1 at IoU=0.4)
uv run rallycut evaluate --heuristics                 # Force heuristics evaluation (57% F1)
uv run rallycut evaluate --model beach --iou 0.5      # Evaluate beach model
uv run rallycut evaluate tune-decoder --iou 0.4       # Grid search decoder parameters

# Stratified evaluation (by video characteristics)
uv run python scripts/eval_stratified.py             # Group metrics by brightness/camera/complexity
uv run python scripts/eval_stratified.py --ball-only # Ball metrics only

# Cross-rally player matching
uv run rallycut match-players <video-id>                 # Assign consistent player IDs 1-4
uv run rallycut match-players <video-id> -o result.json  # Export assignments to JSON
uv run rallycut match-players <video-id> --num-samples 20  # More frames per track (default: 12)
uv run rallycut match-players <video-id> -q              # Quiet mode

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
├── temporal/        # Temporal models and TemporalMaxer pipeline
│   ├── temporal_maxer/       # TemporalMaxer TAS model (recommended)
│   │   ├── model.py          # Multi-scale temporal pyramid architecture
│   │   ├── training.py       # Sequence-level training loop
│   │   └── inference.py      # Inference and segment extraction
│   ├── binary_head.py        # Binary head classifier (alternative)
│   ├── deterministic_decoder.py  # Decoder for binary head output
│   ├── boundary_refinement.py    # Fine-stride boundary refinement
│   ├── features.py           # VideoMAE feature extraction and caching
│   ├── inference.py          # Model loading and inference
│   ├── models.py             # v1/v2/v3 temporal models (deprecated)
│   └── training.py           # Training loop with validation
├── tracking/        # Ball and player tracking
│   ├── ball_tracker.py       # VballNet ONNX ball detection (9-frame temporal)
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

Four detection pipelines are available, with TemporalMaxer as the recommended default:

| Pipeline | F1 (IoU=0.5) | F1 (IoU=0.4) | Overmerge | Command |
|----------|--------------|--------------|-----------|---------|
| TemporalMaxer (default) | 71.6% (LOO) | 75.0% (LOO) | 0% | `rallycut cut video.mp4 --temporal-maxer` |
| Binary Head + Decoder | ~78% | 84% | 0% | `rallycut cut video.mp4 --binary-head` |
| Heuristics (fallback) | 57% | - | ~10% | `rallycut cut video.mp4 --heuristics` |
| Temporal (deprecated) | 65% | - | ~5% | `rallycut cut video.mp4 --experimental-temporal` |

**Note:** IoU=0.4 better reflects detection accuracy when labeling marks serve toss start (model detects ~2s later when play begins). Binary head's 84% F1 is train-on-all (inflated); TemporalMaxer's 75% is honest LOO CV.

**Pipeline auto-selection:**
1. If `--temporal-maxer` flag: use TemporalMaxer (auto-extracts features if not cached)
2. If `--binary-head` flag: use binary head + decoder
3. If `--experimental-temporal` flag: use temporal model (deprecated)
4. If `--heuristics` flag: use heuristics
5. Auto: TemporalMaxer if model+features exist, else binary head if features cached, else heuristics

**Cloud (DetectionService/Modal):** Automatically uses TemporalMaxer when weights are in the image. Features are extracted inline on first run and cached for subsequent runs.

**Enabling best pipeline (local):**
```bash
# Train TemporalMaxer (features auto-extracted on first run, or pre-extract for speed)
uv run rallycut train extract-features --stride 48  # Optional: pre-extract for faster first run
uv run rallycut train temporal-maxer --epochs 50
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

Production ML runs on Modal GPUs. DetectionService automatically enables TemporalMaxer when
`weights/temporal_maxer/best_temporal_maxer.pt` is present in the image (bundled via
`modal_app.py`'s `.add_local_dir("weights", "/app/weights")`). Features are extracted inline
on first detection and cached at `~/.cache/rallycut/features/` for subsequent runs.

```bash
# Deploy
modal deploy rallycut/service/platforms/modal_app.py

# Local testing
uv run python -m rallycut.service.local_runner
```

API triggers Modal via webhook. Results posted back on completion.

## Player Tracking

Uses YOLOv8 for person detection with BoT-SORT for temporal tracking. BoT-SORT with ReID (enabled by default) reduces ID switches when players cross paths. GMC (camera motion compensation) is disabled for fixed tripod cameras.

**YOLO Model Options:**
| Model | FPS | F1 | Recall | Use Case |
|-------|-----|-----|--------|----------|
| **yolov8n** | 23 | 88.0% | 82.4% | Default - best speed/accuracy tradeoff |
| yolov8s | 15 | 83.8% | 78.7% | - |
| yolov8m | 7 | 89.2% | 86.4% | Best accuracy (3x slower) |
| yolov8l | 5 | 88.4% | 85.5% | No benefit over medium |
| yolo11n | 23 | 88.5% | 83.1% | Marginal +0.9pp HOTA over yolov8n |
| yolo11s | 14 | 89.0% | 84.2% | Best overall metrics, 1.6x slower |
| yolo11m | 7 | - | - | Untested |
| yolo11l | 5 | - | - | Untested |

**Tracker Options:**
| Tracker | ID | Strengths | Use Case |
|---------|-----|-----------|----------|
| **BoT-SORT** | `botsort` (default) | ReID for identity consistency, reduced ID switches | Recommended for all videos |
| **ByteTrack** | `bytetrack` | Simpler, faster | Legacy compatibility |

**Preprocessing Options:**
| Method | ID | Description | Use Case |
|--------|-----|-------------|----------|
| None | `none` (default) | No preprocessing | Most videos |
| CLAHE | `clahe` | Contrast enhancement via LAB color space | High-contrast sand backgrounds (tested, not beneficial) |

```bash
# Default tracking (yolov8n + BoT-SORT)
uv run rallycut track-players video.mp4

# Use medium model for best accuracy (3x slower)
uv run rallycut track-players video.mp4 --yolo-model yolov8m

# Use ByteTrack instead of BoT-SORT
uv run rallycut track-players video.mp4 --tracker bytetrack

# Tune filter parameters for edge cases
uv run rallycut track-players video.mp4 --min-bbox-area 0.002 --min-bbox-height 0.06

# Compare YOLO model sizes
uv run rallycut evaluate-tracking compare-yolo-models --all
```

## Player Tracking Filtering

Multi-stage filtering to identify active players and exclude non-players. See `tracking/player_filter.py`.

**Primary Track Identification:**
1. **Hard filters** (must pass all):
   - Not on sidelines (x between 0.20-0.80)
   - Not identified as referee (sideline + low movement + no ball proximity)
   - Minimum presence rate (>20% of frames)
   - Court presence >50% (if calibration available)
2. **Soft filters** (for ranking):
   - Stationary tracks (low spread + no ball engagement) are deprioritized but included as fallbacks
3. **Selection priority**:
   - Active tracks (moving or ball-engaged) with high stability
   - Active tracks with lower stability (if needed)
   - Stationary tracks as fallback (if still need more players)

**Per-Frame Filtering:**
1. Bbox size filter (removes small background detections, keeps primary tracks)
2. Play area filter (convex hull of ball positions, keeps primary tracks)
3. Court boundary filter (if calibration available, removes off-court detections before tracking)
4. Distractor/referee/stationary filter (hard 4-player constraint removes non-primary tracks coexisting with all players; excludes referees and stationary tracks)
5. Two-team selection (2 players per court side by Y position)

**Key insight:** Positively identify players by volleyball behaviors (ball engagement, court coverage, movement) rather than detecting "non-players".

## Track Merging

Merges fragmented tracker IDs using velocity prediction and position/size similarity. Thresholds are video-agnostic (seconds, normalized coordinates). BoT-SORT's ReID reduces fragmentation by maintaining identity through brief occlusions. See `tracking/player_filter.py`.

## Ball Tracking Filtering

Multi-stage temporal filter for ball tracking. Default raw mode applies motion energy filtering, stationarity detection, segment pruning, oscillation/blip removal, and interpolation. See `tracking/ball_filter.py`.

### Ball Tracking Model Variants

Multiple ball tracking models are available via the `model` parameter:

| Model | ID | Match Rate | Error | Speed | Recommendation |
|-------|-----|------------|-------|-------|----------------|
| **Ensemble** | `ensemble` (default) | **82.5%** | 28.2px | 33 FPS (ONNX+CoreML) | **Default** - best accuracy |
| **WASB HRNet** | `wasb` | 67.5% | 51.9px | 33 FPS (ONNX+CoreML) | High precision, lower recall |
| **VballNetV2** | `v2` | 41.7% | 92.7px | 100 FPS (CPU) | Fastest, ONNX-only |
| **VballNetFastV1** | `fast` | ~40% | ~95px | 280 FPS (CPU) | Speed priority (3.6x faster) |

WASB (BMVC 2023) is an HRNet pretrained on indoor volleyball. The ensemble uses WASB as primary (high precision) with VballNet as fallback (high recall). WASB uses ONNX+CoreML on Apple Silicon (auto-exported from PyTorch on first use); VballNet uses ONNX.

```python
from rallycut.tracking import create_ball_tracker

# Default: ensemble (best accuracy - 82.5% match rate)
tracker = create_ball_tracker()

# WASB only (high precision, 67.5% match rate)
tracker = create_ball_tracker("wasb")

# VballNet only (fastest, 41.7% match rate)
tracker = create_ball_tracker("v2")
```

CLI support:
```bash
rallycut track-players video.mp4                          # Default ensemble
rallycut track-players video.mp4 --ball-model wasb        # WASB only
rallycut track-players video.mp4 --ball-model v2          # VballNet only (fastest)
rallycut evaluate-tracking compare-ball-models --all      # Compare all variants
```

### Heatmap Decoding Options

The VballNet model outputs heatmaps that are decoded to ball positions. See `tracking/ball_tracker.py`.

**HeatmapDecodingConfig:**
- `threshold=0.3`: Threshold for heatmap binarization. Uses contour method for centroid extraction.

Testing on ground truth (beach volleyball) showed `threshold=0.3` with contour centroid gives best results. With frame offset auto-detection, accuracy reaches **75% match rate** at 50px threshold.

The raw filter pipeline maximizes detection rate and match rate through post-processing (segment pruning, outlier removal, blip removal, interpolation) rather than Kalman smoothing.

**Problems solved:**
- **False positives at static positions**: VballNet frequently detects stationary players as the ball. The motion energy filter (step 0 in raw mode) computes temporal intensity change in a 15x15 patch around each detection. Real ball in flight creates high motion energy; a player standing still has near-zero energy. Positions with `motion_energy < 0.02` are zeroed out. This removes ~5% of false positives before segment pruning runs, improving detection rate from 78.6% to 83.7% and match rate from 47.1% to 48.6%.
- **Player lock-on (stationarity)**: WASB can lock onto a player position for 12+ consecutive frames, producing nearly identical coordinates (spread < 0.5% of screen). Since a volleyball is always in motion during a rally, this is physics-impossible and indicates model failure. The stationarity filter uses a sliding window to detect and zero these blocks. Tight thresholds (0.5% spread, 12 frames minimum) avoid false positives on real slow-ball events like set apex (~0.8% spread) or slow net trajectories (~1.5%). Source-agnostic: applies equally to WASB and VballNet. Default off; enabled only in ensemble mode.
- **False segments at rally boundaries**: VballNet outputs consistent false detections at rally start/end (before temporal context builds up, or after ball exits frame). Segment pruning splits the trajectory at large position jumps (>20% screen) and discards short fragments (<15 frames). Short fragments that are spatially close to an anchor segment (within 10-15% of screen of the nearest anchor endpoint, wider when the anchor contains WASB positions) are recovered rather than discarded — these are real trajectory fragments between interleaved false positives.
- **Interleaved false positives**: VballNet sometimes interleaves single-frame false detections (jumping to player positions) within real trajectory regions. Without anchor-proximity recovery, segment splitting at each jump fragments the real trajectory into tiny segments that all get pruned. The recovery step keeps real fragments and discards only the distant false positive clusters.
- **Oscillating false detections**: After ball exits frame, VballNet can lock onto two players and alternate between them with high confidence. The pattern is cluster-based: positions stay near one player for 2-5 frames, jump to another for 1-2 frames, then back. Oscillation pruning uses spatial clustering to detect this: finds the two furthest-apart positions (poles) in each window, assigns positions to nearest pole, and counts cluster transitions. Both clusters must be compact (within 1.5% of screen of their pole — real VballNet player-locking has ~1% noise) and have ≥3 members. A transition rate ≥25% over 12+ frames triggers trimming.
- **Exit ghost detections**: When the ball exits the frame mid-rally, VballNet produces false detections that smoothly drift from the exit point toward a player position. These "exit ghosts" are missed by other filters (no jump, no oscillation, no gap). Detected by physics: ball consistently approaching a screen edge with velocity > 0.8%/frame MUST exit — any reversal away from that edge is impossible. All subsequent positions are marked as ghosts until a gap > max_interpolation_gap frames.
- **Hovering false detections**: After ball exits frame, VballNet can lock onto a single player position and produce many frames within a tiny radius. Detected by checking short segments (12-36 frames) that appear after a gap > max_interpolation_gap: if the first 12 positions all lie within 5% of screen of their centroid, the segment is dropped. Real ball trajectories always have spread >5% over 12 frames because the ball is in motion.
- **Trajectory blips**: VballNet can briefly lock onto a player position for 2-5 consecutive frames mid-trajectory. Single-frame outlier detection misses these because consecutive false positives validate each other as neighbors. Blip removal uses distant trajectory context (positions ≥5 frames away) with a two-pass approach: first flags suspects, then re-evaluates with clean (non-suspect) context to prevent contamination. Only compact clusters of ≥2 consecutive suspect frames are removed (real bounces have trajectory spread, blips are tightly clustered at a fixed player position).
- **Missing frames**: Linear interpolation fills small gaps (up to 10 frames) between detections.

**Key parameters (BallFilterConfig):**
- `enable_motion_energy_filter=True`: Remove false positives at static positions (low temporal change)
- `motion_energy_threshold=0.02`: Motion energy below this = likely false positive (zeroed)
- `enable_stationarity_filter=False`: Detect and remove player lock-on (12+ frames within 0.5% spread). Default off; enabled in ensemble mode
- `stationarity_min_frames=12`: Minimum consecutive frames to flag as stationary (~0.4s at 30fps)
- `stationarity_max_spread=0.005`: Maximum spread from centroid (0.5% of screen, ~10px at 1920px)
- `enable_segment_pruning=True`: Remove short disconnected false segments
- `segment_jump_threshold=0.20`: Position jump threshold to split segments (20% of screen)
- `min_segment_frames=15`: Minimum frames to keep a segment
- `min_output_confidence=0.05`: Drop VballNet zero-confidence placeholders from output
- `enable_exit_ghost_removal=True`: Remove physics-impossible reversals near screen edges
- `exit_edge_zone=0.10`: Screen edge zone (10%) for exit approach detection
- `exit_approach_frames=3`: Min consecutive frames approaching edge before reversal triggers
- `exit_min_approach_speed=0.008`: Min per-frame speed toward edge (~0.8% of screen)
- `enable_oscillation_pruning=True`: Detect and trim cluster-based oscillation patterns
- `min_oscillation_frames=12`: Sliding window size for cluster transition rate detection
- `oscillation_reversal_rate=0.25`: Cluster transition rate threshold to trigger pruning
- `oscillation_min_displacement=0.03`: Min pole distance (3% of screen) to detect oscillation
- `enable_outlier_removal=True`: Removes flickering and edge artifacts (runs after pruning)
- `enable_blip_removal=True`: Remove multi-frame trajectory blips (consecutive false positives at player positions)
- `blip_context_min_frames=5`: Min frame distance for distant context neighbors
- `blip_max_deviation=0.15`: Max deviation from interpolated trajectory (15% of screen)
- `enable_interpolation=True`: Fill missing frames with linear interpolation
- `max_interpolation_gap=10`: Max frames to interpolate (larger gaps left empty)

**Usage:**
```python
# Default: raw mode with segment pruning (best detection/match rate)
result = tracker.track_video(video_path)

# Disable filtering entirely for raw comparison
result = tracker.track_video(video_path, enable_filtering=False)

# Keep raw positions for debugging
result = tracker.track_video(video_path, preserve_raw=True)
print(result.raw_positions)  # Original detections before filtering
```

**Grid search for optimal parameters:**
```bash
# Segment pruning grid search
uv run rallycut evaluate-tracking tune-ball-filter --all --grid segment-pruning

# Oscillation pruning grid search
uv run rallycut evaluate-tracking tune-ball-filter --all --grid oscillation

# Outlier removal grid search
uv run rallycut evaluate-tracking tune-ball-filter --all --grid outlier

# Ensemble grid search
uv run rallycut evaluate-tracking tune-ball-filter --all --grid ensemble -o results.json
```

### Frame Offset Auto-Detection

Different videos require different frame offsets due to FPS variations (29.97 vs 30.0) and ground truth labeling timing. The evaluation automatically detects the optimal offset (0-5 frames) per video.

```python
from rallycut.evaluation.tracking.ball_metrics import find_optimal_frame_offset

# Find best offset for aligning predictions with ground truth
best_offset, match_rate = find_optimal_frame_offset(
    ground_truth=gt_positions,
    predictions=ball_positions,
    video_width=1920,
    video_height=1080,
)
print(f"Optimal offset: +{best_offset} frames ({match_rate*100:.1f}% match)")
```

## Cross-Rally Player Consistency

Maintains consistent player IDs (1-4) across a match using appearance-based matching.

**How it works:**
1. For each rally, sample ~12 video frames per primary track
2. Extract appearance features (skin tone HSV, jersey color, body proportions)
3. Use Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) to match tracks to accumulated player profiles
4. Detect side switches when swapped assignment cost < 70% of normal cost (requires 3+ rallies)
5. Update profiles with new appearance data after assignment

**Cost function:** 50% skin tone + 30% height + 20% jersey color (lower = better match).

**Key files:** `tracking/player_features.py` (feature extraction, similarity), `tracking/match_tracker.py` (orchestration, Hungarian assignment, side switch detection), `evaluation/tracking/db.py` (`load_rallies_for_video`), `cli/commands/match_players.py` (CLI).

```bash
uv run rallycut match-players <video-id> -o result.json
```

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

# Ball tracking evaluation
uv run rallycut evaluate-tracking --all --ball-only      # Ball tracking metrics only
uv run rallycut evaluate-tracking -r <rally-id> -b       # Ball eval for specific rally
uv run rallycut evaluate-tracking --all -b -o ball.json  # Export ball metrics to JSON

# Grid search for optimal player filter parameters
uv run rallycut evaluate-tracking tune-filter --all --cache-only  # Cache raw positions (one-time)
uv run rallycut evaluate-tracking tune-filter --all --grid quick  # Quick grid search
uv run rallycut evaluate-tracking tune-filter --all --grid farside  # Far-side player optimization
uv run rallycut evaluate-tracking tune-filter --all --grid relaxed  # Comprehensive relaxation
uv run rallycut evaluate-tracking tune-filter --all --grid full --min-rally-f1 0.70  # Full search with constraint
uv run rallycut evaluate-tracking tune-filter --all -o results.json  # Export full results

# Grid search for optimal ball filter parameters
uv run rallycut evaluate-tracking tune-ball-filter --all --grid segment-pruning  # Segment pruning (18 configs)
uv run rallycut evaluate-tracking tune-ball-filter --all --grid oscillation     # Oscillation pruning (18 configs)
uv run rallycut evaluate-tracking tune-ball-filter --all --grid outlier         # Outlier removal + exit detection (18 configs)
uv run rallycut evaluate-tracking tune-ball-filter --all --grid ensemble        # Ensemble search
uv run rallycut evaluate-tracking tune-ball-filter --all -o ball.json           # Export results
```

| Command | Purpose |
|---------|---------|
| `rallycut label open` | Open video with pre-filled predictions |
| `rallycut label save` | Export annotations as ground truth JSON |
| `rallycut compare-tracking` | Compute MOT metrics from JSON files |
| `rallycut evaluate-tracking` | Evaluate player tracking from database |
| `rallycut evaluate-tracking --ball-only` | Evaluate ball tracking with detailed metrics |
| `rallycut evaluate-tracking tune-filter` | Grid search for optimal PlayerFilterConfig parameters |
| `rallycut evaluate-tracking tune-ball-filter` | Grid search for optimal BallFilterConfig parameters |

**Player tracking metrics:**
- MOTA: Multi-Object Tracking Accuracy (combines FP, FN, ID switches)
- HOTA: Higher Order Tracking Accuracy (balances detection and association)
- DetA: Detection Accuracy (how well detections match GT)
- AssA: Association Accuracy (ID consistency across frames)
- Mostly Tracked (MT): GT tracks tracked >80% of their lifespan
- Fragmentation: GT tracks split into multiple prediction tracks

**Ball tracking metrics (`--ball-only`):**
- Detection Rate: % of GT frames with ball detected
- Match Rate: % of GT frames with ball within 50px threshold
- Mean/Median/P90 Error: Position error in pixels
- Accuracy <20px/<50px: % of detections within threshold


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
