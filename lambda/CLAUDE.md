# RallyCut Lambda Functions

Serverless video processing on AWS Lambda (ARM64).

## Functions

| Function | Purpose |
|----------|---------|
| `video-export/` | Extract rallies, concatenate, apply watermark |
| `video-optimize/` | Faststart + compression for uploaded videos |

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

## Environment Variables

- `AWS_BUCKET_NAME` - Source/destination bucket
- `API_WEBHOOK_URL` - Completion callback endpoint
