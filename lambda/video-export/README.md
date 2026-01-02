# Video Export Lambda

AWS Lambda function for server-side video export processing. Handles clip extraction, watermarking, and concatenation using FFmpeg.

## Features

- **Free Tier**: 720p resolution with RallyCut watermark
- **Premium Tier**: Original quality, no watermark, fast copy (no re-encoding)
- **ARM64 (Graviton2)**: 20% cheaper than x86

## Prerequisites

- Docker
- AWS CLI configured with appropriate permissions

## Deployment

```bash
# Build and push image
docker build --platform linux/arm64 --provenance=false -t rallycut-video-export .
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com
docker tag rallycut-video-export:latest <account>.dkr.ecr.us-east-1.amazonaws.com/rallycut-video-export-dev:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/rallycut-video-export-dev:latest

# Update Lambda
aws lambda update-function-code \
  --function-name rallycut-video-export-dev \
  --image-uri <account>.dkr.ecr.us-east-1.amazonaws.com/rallycut-video-export-dev:latest
```

## Configuration

| Variable | Description |
|----------|-------------|
| `S3_BUCKET_NAME` | S3 bucket for video storage |
| Memory | 1024 MB |
| Timeout | 900s (15 min) |
| Architecture | arm64 |

## API Environment

Add to your API's `.env`:
```
EXPORT_LAMBDA_FUNCTION_NAME=rallycut-video-export-dev
API_BASE_URL=https://api.rallycut.app
```

## Cost Optimization

- ARM64 architecture (20% cheaper)
- 1024 MB memory (minimum for FFmpeg)
- S3 lifecycle: exports auto-delete after 1 day
- CloudWatch logs: 14-day retention
