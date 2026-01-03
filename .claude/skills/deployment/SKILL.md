---
name: deployment
description: Deploy RallyCut services - Modal for ML inference, Lambda for video export. Use when deploying updates to cloud services. (project)
allowed-tools: Bash, Read
---

# RallyCut Deployment

## Modal (ML Inference)

```bash
cd analysis
modal deploy rallycut/service/modal_app.py
```

### Verify Deployment
- Check Modal dashboard for function status
- Test: `modal run rallycut/service/modal_app.py::detect_rallies`

## Lambda (Video Export)

```bash
cd lambda/video-export

# Build container
docker build --platform linux/arm64 -t rallycut-export .

# Tag and push to ECR
aws ecr get-login-password | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
docker tag rallycut-export:latest <account>.dkr.ecr.<region>.amazonaws.com/rallycut-export:latest
docker push <account>.dkr.ecr.<region>.amazonaws.com/rallycut-export:latest

# Update Lambda function
aws lambda update-function-code --function-name rallycut-video-export --image-uri <ecr-uri>:latest
```

## Environment Variables

### Modal
Set in Modal dashboard or `modal secret create`:
- `DATABASE_URL`
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
- `WEBHOOK_SECRET`

### Lambda
Set in AWS Lambda console:
- `AWS_BUCKET_NAME`
- `API_WEBHOOK_URL`
