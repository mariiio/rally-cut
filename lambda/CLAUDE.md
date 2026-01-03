# RallyCut Lambda Functions

Serverless video processing on AWS Lambda (ARM64).

## Functions

| Function | Purpose |
|----------|---------|
| `video-export/` | Extract rallies, concatenate, apply watermark |
| `video-optimize/` | Poster, optimization, and proxy generation |

## Structure

```
lambda/
├── video-export/
│   ├── Dockerfile       # ARM64 with FFmpeg
│   ├── handler.py       # Lambda entry point
│   └── requirements.txt
└── video-optimize/
    └── ...
```

## Build & Deploy

```bash
cd lambda/video-export
docker build -t rallycut-export .
# Push to ECR, update Lambda function
```

## Video Export Flow

1. API creates ExportJob, invokes Lambda
2. Lambda downloads video from S3
3. Extracts rally segments with FFmpeg
4. Applies watermark (FREE tier) or passes through (PREMIUM)
5. Concatenates segments
6. Uploads result to S3
7. POSTs completion webhook to API

## Video Optimize Flow

After upload confirmation, the video-optimize function processes videos:

1. Downloads original video from S3
2. **Generates poster** (always): 1280px JPEG at 1 second
3. **Checks if optimization needed**: moov atom position + bitrate > 8Mbps
4. **Optimizes video** (if needed): H.264, CRF 23, faststart
5. **Generates proxy** (PREMIUM tier only): 720p, CRF 28 for fast editing
6. Uploads all outputs to S3 with appropriate cache headers
7. POSTs completion webhook with all S3 keys

**Outputs:**
- `{base}_poster.jpg` - Poster image (~50KB)
- `{base}_optimized.mp4` - Full quality optimized video
- `{base}_proxy.mp4` - 720p proxy (PREMIUM only)

## Environment Variables

- `AWS_BUCKET_NAME` - Source/destination bucket
- `API_WEBHOOK_URL` - Completion callback endpoint
