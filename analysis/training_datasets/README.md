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

3. **Prepare training samples**
   ```bash
   rallycut train prepare --output training_data/
   ```
   This generates 480p@30fps proxy videos for efficient training (VideoMAE only needs 224x224 input).

4. **Run training on Modal GPU**
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

5. **Evaluate the trained model**
   ```bash
   rallycut evaluate --model beach --iou 0.5
   ```

6. **Clean up Modal storage** (recommended after training)
   ```bash
   rallycut train modal --cleanup  # Delete videos and model outputs from Modal
   ```
   Modal charges ~$0.75/GB/month for storage. Proxy videos are cached locally at `~/.cache/rallycut/proxies/` - re-upload anytime for retraining.

## Adding New Videos (Incremental Training)

When you have new labeled videos to add to an existing trained model:

1. **Upload and label in web app**
   - Upload new beach volleyball video
   - Use rally editor to mark all rallies
   - Ensure rallies are saved (confidence = NULL for ground truth)

2. **Re-export dataset** (includes all videos, old + new)
   ```bash
   rallycut train export-dataset --name beach_v2
   ```

3. **Commit dataset to git** (backup labels in case DB resets)
   ```bash
   cd training_datasets/beach_v2
   git add manifest.json ground_truth.json
   git commit -m "Add beach_v2 dataset (N videos, M rallies)"
   ```

4. **Prepare training data**
   ```bash
   rallycut train prepare --output training_data/
   ```

5. **Upload to Modal**
   ```bash
   rallycut train modal --upload         # Upload training JSON
   rallycut train modal --upload-videos  # Upload proxy videos (only new ones)
   rallycut train modal --upload-model   # Upload existing model weights
   ```

6. **Run incremental training** (lower learning rate to preserve existing knowledge)
   ```bash
   rallycut train modal --resume-from-model --lr 1e-5 --epochs 5
   ```

7. **Download and clean up**
   ```bash
   rallycut train modal --download
   rallycut train modal --cleanup
   ```

**Why incremental training?**
- Starts from your existing trained model, not from scratch
- Lower learning rate (1e-5 vs 5e-5) prevents "catastrophic forgetting"
- Fewer epochs needed (5 vs 10+) since model already knows most patterns
- Faster and cheaper than full retraining

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

## Model-Specific Heuristics

The beach model uses optimized post-processing heuristics:

| Parameter | Indoor | Beach |
|-----------|--------|-------|
| `min_play_duration` | 1.0s | 0.5s |
| `rally_continuation_seconds` | 2.0s | 3.0s |
| `boundary_confidence_threshold` | 0.35 | 0.25 |
| `min_active_density` | 0.25 | 0.15 |

These heuristics are automatically applied when using `--model beach`.

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

Videos are git-ignored (large files), manifests are tracked:

```bash
# After exporting a new dataset
cd training_datasets/beach_v2
git add manifest.json ground_truth.json
git commit -m "Add beach_v2 training dataset (N videos, M rallies)"
```
