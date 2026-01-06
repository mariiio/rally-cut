# RallyCut Lambda Functions

Serverless video processing on AWS Lambda (ARM64 Graviton2, 20% cheaper than x86).

## Functions

| Function | Purpose | Memory | Timeout |
|----------|---------|--------|---------|
| `video-export/` | Extract rallies, concatenate, apply camera effects | 1024 MB | 15 min |
| `video-optimize/` | Poster, optimization, proxy generation | 2048 MB | 10 min |

## Structure

```
lambda/
├── video-export/
│   ├── handler.py       # Lambda entry point
│   ├── Dockerfile       # ARM64 with FFmpeg
│   └── deploy-cli.sh    # Deployment script
└── video-optimize/
    ├── handler.py
    ├── Dockerfile
    └── deploy.sh
```

## Video Export

Extracts rally clips and concatenates them with tier-specific processing:

| Tier | Processing |
|------|------------|
| PREMIUM (no camera) | Stream copy, no re-encode (`-c copy`) |
| PREMIUM (with camera) | Re-encode with crop filter (`libx264 -crf 23`) |
| FREE | Scale to 720p + watermark overlay |

**Camera effects**: Builds FFmpeg crop filter from keyframes with easing interpolation. Generates time-varying x/y/w/h expressions for smooth pan/zoom.

**Progress reporting**:
- 0-80%: Clip extraction (per rally)
- 85%: Concatenation
- 95%: Upload
- Webhooks: `/v1/webhooks/export-progress`, `/v1/webhooks/export-complete`

## Video Optimize

Runs after upload confirmation:

1. **Poster**: 1280px JPEG at 1-second mark
2. **Optimization check**: If bitrate > 8Mbps OR moov atom not in first 32KB
3. **Optimize**: H.264 CRF 23, movflags +faststart (skipped if not needed)
4. **Proxy**: 720p, CRF 28, 96kbps audio

**Outputs**: `{base}_poster.jpg`, `{base}_optimized.mp4`, `{base}_proxy.mp4`

## FFmpeg Settings

```
Video: libx264, crf 23, preset fast, profile high, level 4.1
Audio: AAC 128kbps stereo
Flags: movflags +faststart (for streaming)
```

Watermark: `s3://bucket/assets/watermark.png`, overlaid bottom-right.

## Deployment

```bash
# Export Lambda
cd lambda/video-export
./deploy-cli.sh dev   # or prod

# Optimize Lambda
cd lambda/video-optimize
./deploy.sh dev       # or prod
```

Scripts handle: ECR repo creation, Docker build (arm64), IAM role, Lambda update.

## Environment

- `S3_BUCKET_NAME` - Source/destination bucket
- `API_WEBHOOK_URL` - Completion callback (set by deploy script)

## Caveats

- **15-min timeout**: Max ~20-30 min videos at 720p
- **No resume**: If Lambda times out mid-process, job fails (no partial upload)
- **Sequential clips**: Rally extraction is sequential, not parallel
- **Watermark optional**: Continues without watermark if asset missing from S3
- **Camera filter complexity**: Multi-keyframe interpolation uses piecewise FFmpeg expressions
