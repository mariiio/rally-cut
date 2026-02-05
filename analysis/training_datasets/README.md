# Training Datasets

This directory contains training data for fine-tuning the VideoMAE rally detection model.

## Directory Structure

```
training_datasets/
├── README.md           # This file
├── .gitignore          # Ignores video files, tracks manifests
├── beach_v1/           # Dataset name
│   ├── manifest.json   # Dataset metadata (tracked)
│   ├── ground_truth.json  # Rally annotations (tracked)
│   └── videos/         # Video files (git-ignored)
│       ├── video1.mp4
│       └── video2.MOV
└── beach_v2/           # Another dataset version
    └── ...
```

## Model Variants

RallyCut supports two model variants:

| Model | Path | Purpose | Heuristics |
|-------|------|---------|------------|
| `indoor` | `weights/videomae/game_state_classifier/` | Original model for indoor volleyball | Default values |
| `beach` | `weights/videomae/beach_volleyball/` | Fine-tuned for beach volleyball | Optimized for beach |

**Key differences:**
- Indoor model: original from volleyball_analytics, never modified
- Beach model: fine-tuned on labeled beach volleyball data

## Quick Start

### Using Models

```bash
# Use indoor model (default)
rallycut cut video.mp4 --model indoor

# Use beach model (with optimized heuristics)
rallycut cut video.mp4 --model beach

# Evaluate with beach model
rallycut evaluate --model beach --iou 0.5
```

### Training Workflow

1. **Label videos in web editor**
   - Upload videos to the web app
   - Tag rally start/end points manually

2. **Export dataset from database**
   ```bash
   rallycut train export-dataset --name beach_v1
   ```

3. **Back up to S3** (survives DB resets and MinIO clears)
   ```bash
   rallycut train push --name beach_v1
   ```

4. **Prepare training samples**
   ```bash
   rallycut train prepare --output training_data/
   ```
   This generates 480p@30fps proxy videos for efficient training (VideoMAE only needs 224x224 input).

5. **Run training on Modal GPU**
   ```bash
   rallycut train modal --upload         # Upload training data JSON
   rallycut train modal --upload-videos  # Upload proxy videos (parallel, ~4GB)
   rallycut train modal --epochs 10      # Train on T4 GPU (~$0.59/hr)
   rallycut train modal --download       # Get trained model
   ```

   **Preemption resilience:** Training automatically handles GPU preemption:
   - Auto-retries up to 2 times (3 total attempts)
   - Checkpoints saved every 100 steps (~5 min max progress loss)
   - Resumes from latest checkpoint on restart

6. **Evaluate the trained model**
   ```bash
   rallycut evaluate --model beach --iou 0.5
   ```

7. **Clean up Modal storage** (recommended after training)
   ```bash
   rallycut train modal --cleanup  # Delete videos and model outputs from Modal
   ```
   Modal charges ~$0.75/GB/month for storage. Proxy videos are cached locally at `~/.cache/rallycut/proxies/` - re-upload anytime for retraining.

## S3 Backup

Training datasets can be backed up to AWS S3 so the complete training pipeline survives local DB resets and MinIO volume clears.

### Setup

```bash
export TRAINING_S3_BUCKET=your-bucket    # Required
# Optional (defaults shown):
# export TRAINING_S3_PREFIX=training
# export TRAINING_S3_REGION=us-east-1
```

Uses the default AWS credential chain (`~/.aws/credentials`, env vars, IAM roles) -- **not** the MinIO credentials used by the app.

### S3 Key Structure

Videos are stored by content_hash for deduplication across dataset versions:

```
{prefix}/
  datasets/{name}/
    manifest.json
    ground_truth.json
  videos/
    {content_hash}.mp4      # Shared pool, deduplicated
```

Videos use `StorageClass=STANDARD_IA` (~$0.0125/GB/month). For ~7GB of originals, that's ~$0.09/month.

### Commands

| Command | Purpose |
|---------|---------|
| `rallycut train push --name beach_v2` | Upload dataset to S3 |
| `rallycut train pull --name beach_v2` | Download dataset from S3 |
| `rallycut train restore --name beach_v2` | Re-import ground truth into DB |
| `rallycut train list-remote` | List datasets backed up in S3 |

## Adding New Videos (Recommended Workflow)

When you have new labeled videos to add:

### Step 1: Export and Back Up

1. **Upload and label in web app**
   - Upload new beach volleyball video
   - Use rally editor to mark all rallies
   - Ensure rallies are saved (confidence = NULL for ground truth)

2. **Re-export dataset** (includes all videos, old + new)
   ```bash
   rallycut train export-dataset --name beach_v3
   ```

3. **Back up to S3** (only new videos are uploaded -- existing ones deduplicated by content_hash)
   ```bash
   rallycut train push --name beach_v3
   ```

4. **Commit manifests to git**
   ```bash
   cd training_datasets/beach_v3
   git add manifest.json ground_truth.json
   git commit -m "Add beach_v3 dataset (N videos, M rallies)"
   ```

### Step 2a: Retrain Temporal Model (Fast, Local, Recommended)

The temporal model is a lightweight learned post-processor that smooths VideoMAE predictions. This is the **recommended first step** when adding new data.

```bash
# Extract VideoMAE features (cached, only processes new videos)
rallycut train extract-features --stride 48

# Train temporal smoother (~5 min on CPU)
rallycut train temporal --model v1 --epochs 50

# Model saved to: weights/temporal/best_temporal_model.pt
```

**Why temporal model first?**
- Fast: trains in minutes on CPU (no GPU needed)
- Cheap: no cloud costs
- Effective: learns video-specific patterns from your labeled data
- Non-destructive: doesn't modify the base VideoMAE model

### Step 2b: Fine-tune VideoMAE (Slow, GPU Required, Only If Needed)

Only do this if the temporal model doesn't achieve good enough results, or if you need the model to recognize entirely new visual settings.

```bash
# Prepare training samples
rallycut train prepare --output training_data/

# Upload to Modal
rallycut train modal --upload         # Upload training JSON
rallycut train modal --upload-videos  # Upload proxy videos (only new ones)
rallycut train modal --upload-model   # Upload existing model weights

# Run incremental training (lower LR to preserve knowledge)
rallycut train modal --resume-from-model --lr 1e-6 --epochs 10

# Download and clean up
rallycut train modal --download
rallycut train modal --cleanup
```

**When to fine-tune VideoMAE?**
- Temporal model alone doesn't help (e.g., model doesn't recognize the visual setting at all)
- You have videos with very different visual characteristics (different camera angles, lighting, venues)
- You have 20+ labeled videos (need enough data to avoid overfitting)

**Why be cautious with VideoMAE fine-tuning?**
- Slow: hours on GPU (~$0.59/hr)
- Risk of overfitting with limited data
- Risk of "catastrophic forgetting" (model forgets old patterns)
- Previous beach fine-tuning made model less discriminative

## Disaster Recovery

After a DB reset or fresh machine setup, restore the complete training pipeline:

```bash
# See available backups
rallycut train list-remote

# Download from S3
rallycut train pull --name beach_v2

# Import into DB + upload to MinIO
rallycut train restore --name beach_v2 --upload-to-app-s3

# Videos and labels are now back in the web editor
# Ready to train again:
rallycut train prepare
rallycut train modal --upload --upload-videos --epochs 10
```

Use `--dry-run` to preview what restore would do without making changes:
```bash
rallycut train restore --name beach_v2 --dry-run
```

## Dataset Files

### manifest.json (tracked)

Contains dataset metadata and video list:

```json
{
  "name": "beach_v1",
  "description": "Beach volleyball training data",
  "created": "2026-01-12T10:00:00",
  "videos": [
    {
      "filename": "IMG_1817.MOV",
      "video_id": "uuid-here",
      "content_hash": "sha256...",
      "duration_ms": 1200000,
      "rally_count": 35
    }
  ],
  "stats": {
    "total_videos": 9,
    "total_rallies": 311,
    "total_duration_min": 180
  }
}
```

### ground_truth.json (tracked)

Contains rally annotations per video:

```json
[
  {
    "video_id": "uuid-here",
    "filename": "IMG_1817.MOV",
    "rallies": [
      {"start_ms": 5000, "end_ms": 15000},
      {"start_ms": 25000, "end_ms": 38000}
    ]
  }
]
```

## Temporal Models

The temporal model provides learned post-processing to smooth VideoMAE predictions. Three architectures are available:

| Model | Description | Parameters | Training |
|-------|-------------|------------|----------|
| `v1` (LearnedSmoothing) | 1D convolution + classifier | ~1.5K | Fast, stable |
| `v2` (ConvCRF) | CNN + Conditional Random Field | ~50K | Medium |
| `v3` (BiLSTMCRF) | Bidirectional LSTM + CRF | ~100K | Slow, needs more data |

**Recommendation:** Start with v1. It's the simplest and most stable for small datasets.

```bash
# Train v1 (recommended)
rallycut train temporal --model v1 --epochs 50

# Train v2 (if v1 underfits)
rallycut train temporal --model v2 --epochs 100

# Model version is saved in checkpoint metadata
# Auto-detected when loading - no need to specify version
```

**Model version auto-detection:** When a temporal model is saved, its version (v1/v2/v3) is stored in the checkpoint metadata. When loading, the correct model architecture is automatically instantiated.

## Model-Specific Heuristics

The beach model uses optimized post-processing heuristics:

| Parameter | Indoor | Beach |
|-----------|--------|-------|
| `min_play_duration` | 1.0s | 1.0s |
| `rally_continuation_seconds` | 2.0s | 1.5s |
| `boundary_confidence_threshold` | 0.35 | 0.4 |
| `min_active_density` | 0.25 | 0.3 |
| `min_gap_seconds` | 5.0s | 3.0s |

Beach heuristics are tuned to be more discriminative (higher thresholds) to prevent over-merging of rallies. These heuristics are automatically applied when using `--model beach`.

## Proxy Videos for Efficient Training

Training uses 480p@30fps proxy videos instead of full-resolution originals:

| Aspect | Original | Proxy |
|--------|----------|-------|
| Resolution | 1080p/4K | 480p |
| FPS | Variable (30-60) | Normalized to 30fps |
| Size | ~7GB (9 videos) | ~4GB (45% smaller) |
| Quality | Full | Sufficient (VideoMAE uses 224x224) |

**Why proxies?**
- Faster upload to Modal cloud
- Faster frame decoding during training
- Same ML accuracy (VideoMAE downscales to 224x224 anyway)
- 30fps normalization matches VideoMAE's temporal window (16 frames = 0.53s)

Proxies are generated automatically by `rallycut train prepare` and cached in `~/.cache/rallycut/proxies/`.

## Tips

- **Diverse data**: Include videos from different tournaments, lighting, camera angles
- **Balanced rallies**: Mix short and long rallies
- **Clean labels**: Ensure rally boundaries are accurate
- **Incremental improvement**: Start with few videos, add more as you identify gaps

## Git Workflow

Git and S3 together form a complete backup:

- **Git**: manifest.json + ground_truth.json (small, versioned metadata)
- **S3**: video files (large, deduplicated by content_hash)

```bash
# After exporting and pushing a dataset
cd training_datasets/beach_v2
git add manifest.json ground_truth.json
git commit -m "Add beach_v2 training dataset (N videos, M rallies)"
```

To fully restore: pull manifests from git, pull videos from S3, restore into DB.
