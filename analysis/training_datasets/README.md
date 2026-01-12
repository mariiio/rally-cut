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

4. **Run training on Modal GPU**
   ```bash
   rallycut train modal --upload         # Upload data
   rallycut train modal --upload-videos  # Upload videos
   rallycut train modal --epochs 25      # Train
   rallycut train modal --download       # Get model
   ```

5. **Evaluate the trained model**
   ```bash
   rallycut evaluate --model beach --iou 0.5
   ```

## Adding New Videos

1. **Upload and label in web app**
   - Upload new beach volleyball video
   - Use rally editor to mark all rallies
   - Ensure rallies are saved (confidence = NULL for ground truth)

2. **Re-export dataset**
   ```bash
   rallycut train export-dataset --name beach_v2
   ```

3. **Retrain model**
   ```bash
   rallycut train prepare --output training_data/
   rallycut train modal --epochs 25
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

## Model-Specific Heuristics

The beach model uses optimized post-processing heuristics:

| Parameter | Indoor | Beach |
|-----------|--------|-------|
| `min_play_duration` | 1.0s | 0.5s |
| `rally_continuation_seconds` | 2.0s | 3.0s |
| `boundary_confidence_threshold` | 0.35 | 0.25 |
| `min_active_density` | 0.25 | 0.15 |

These heuristics are automatically applied when using `--model beach`.

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
