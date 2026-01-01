# Modal Setup for RallyCut

This guide walks through deploying the ML detection service to Modal.

## Prerequisites

- Python 3.11+
- Modal account and CLI installed (`pip install modal && modal token new`)
- AWS credentials from the [AWS setup](./aws-setup.md)

## 1. Create Modal Secrets

Modal needs AWS credentials to download videos from S3:

```bash
# Create the aws-credentials secret
modal secret create aws-credentials \
  AWS_ACCESS_KEY_ID=<your-access-key-id> \
  AWS_SECRET_ACCESS_KEY=<your-secret-access-key> \
  AWS_REGION=us-east-1 \
  S3_BUCKET_NAME=rallycut-dev
```

## 2. Deploy the Detection Service

From the `analysis/` directory:

```bash
cd analysis
modal deploy rallycut/service/platforms/modal_app.py
```

After deployment, you'll see output like:

```
✓ Created web endpoint detect => https://your-workspace--rallycut-detection-detect.modal.run
✓ Created web endpoint health => https://your-workspace--rallycut-detection-health.modal.run
```

Save the `detect` endpoint URL - this is your `MODAL_FUNCTION_URL`.

## 3. Update API Environment

Add to your `api/.env`:

```bash
# Modal
MODAL_WEBHOOK_SECRET=<generate-a-random-secret>
MODAL_FUNCTION_URL=https://your-workspace--rallycut-detection-detect.modal.run
```

Generate a webhook secret:

```bash
openssl rand -hex 32
```

## 4. Test the Deployment

### Health Check

```bash
curl https://your-workspace--rallycut-detection-health.modal.run
# {"status":"healthy","service":"rallycut-detection"}
```

### Local Test (Direct Modal Call)

```bash
cd analysis
modal run rallycut/service/platforms/modal_app.py \
  --video-url "https://your-bucket.s3.amazonaws.com/test-video.mp4"
```

### End-to-End Test (Via API)

1. Upload a video through the web UI
2. Click "Detect Rallies"
3. Check the API logs for Modal call
4. Wait for webhook callback with results

## Architecture

```
┌─────────────┐     POST /detect      ┌─────────────┐
│   API       │ ──────────────────────>│   Modal     │
│  (Vercel)   │                        │  (GPU T4)   │
└─────────────┘                        └─────────────┘
       │                                      │
       │                                      │ Download
       │                                      ▼
       │                               ┌─────────────┐
       │                               │    S3       │
       │                               │   Bucket    │
       │                               └─────────────┘
       │                                      │
       │      POST /webhooks/detection-complete
       │◄─────────────────────────────────────┘
       │
       ▼
┌─────────────┐
│  Database   │
│  (Rallies)  │
└─────────────┘
```

## Cost Estimate

| Resource | Specification | Cost |
|----------|---------------|------|
| Modal GPU | T4 (16GB VRAM) | ~$0.59/hour |
| Processing Time | ~5 min per 10min video | ~$0.05/video |

With cold start: first job may take ~2 min to initialize.

## Troubleshooting

### "No module named 'rallycut'"

Ensure you're deploying from the `analysis/` directory where the `rallycut/` package exists.

### Webhook not reaching API

1. Check the callback URL is correct (must be publicly accessible)
2. For local development, use ngrok or similar to expose localhost
3. Check Modal logs: `modal logs rallycut-detection`

### S3 download fails

1. Verify AWS credentials in Modal secret
2. Check S3 bucket policy allows the IAM user
3. Ensure the video key matches what the API sent

### GPU out of memory

The T4 has 16GB VRAM. For long videos (>30 min), consider:
- Using `use_proxy: true` in detection config (default)
- Processing in chunks

## Monitoring

View Modal logs:

```bash
modal logs rallycut-detection --follow
```

View deployed functions:

```bash
modal app list
```
