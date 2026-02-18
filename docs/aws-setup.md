# AWS Infrastructure Setup for RallyCut

This guide walks through setting up the AWS infrastructure for RallyCut.

## Prerequisites

- AWS CLI installed and configured (`aws configure`)
- An AWS account with admin access

## 1. Create S3 Bucket

```bash
# Set your environment (dev or prod)
ENV=dev
REGION=us-east-1
BUCKET_NAME=rallycut-${ENV}

# Create the bucket
aws s3 mb s3://${BUCKET_NAME} --region ${REGION}

# Enable versioning (optional but recommended)
aws s3api put-bucket-versioning \
  --bucket ${BUCKET_NAME} \
  --versioning-configuration Status=Enabled

# Set up CORS for direct browser uploads
aws s3api put-bucket-cors --bucket ${BUCKET_NAME} --cors-configuration '{
  "CORSRules": [
    {
      "AllowedHeaders": ["*"],
      "AllowedMethods": ["GET", "PUT", "POST", "HEAD"],
      "AllowedOrigins": ["http://localhost:3000", "https://rallycut.app"],
      "ExposeHeaders": ["ETag"],
      "MaxAgeSeconds": 3000
    }
  ]
}'

# Set lifecycle policy (prefix-based — see Cost Guardrails below)
# Run after bucket creation:
#   ./scripts/setup-aws-guardrails.sh ${ENV} your@email.com

echo "S3 bucket ${BUCKET_NAME} created"
echo "Run ./scripts/setup-aws-guardrails.sh to configure lifecycle + cost alerts"
```

## 2. Create CloudFront Distribution

### 2.1 Create Origin Access Control (OAC)

```bash
# Create OAC for secure S3 access
aws cloudfront create-origin-access-control \
  --origin-access-control-config '{
    "Name": "rallycut-'${ENV}'-oac",
    "Description": "OAC for RallyCut S3 bucket",
    "SigningProtocol": "sigv4",
    "SigningBehavior": "always",
    "OriginAccessControlOriginType": "s3"
  }'
```

Save the `Id` from the output - you'll need it for the distribution.

### 2.2 Create CloudFront Key Pair for Signed Cookies

This requires using the AWS Console or generating locally:

```bash
# Generate RSA key pair locally
openssl genrsa -out cloudfront-private-key.pem 2048
openssl rsa -pubout -in cloudfront-private-key.pem -out cloudfront-public-key.pem

# Upload public key to CloudFront
aws cloudfront create-public-key \
  --public-key-config '{
    "CallerReference": "rallycut-'${ENV}'-'$(date +%s)'",
    "Name": "rallycut-'${ENV}'-key",
    "EncodedKey": "'$(cat cloudfront-public-key.pem | sed ':a;N;$!ba;s/\n/\\n/g')'"
  }'
```

Save the `Id` from the output - this is your `CLOUDFRONT_KEY_PAIR_ID`.

### 2.3 Create Key Group

```bash
# Use the public key ID from above
PUBLIC_KEY_ID=<your-public-key-id>

aws cloudfront create-key-group \
  --key-group-config '{
    "Name": "rallycut-'${ENV}'-keygroup",
    "Items": ["'${PUBLIC_KEY_ID}'"]
  }'
```

Save the `Id` for use in the distribution.

### 2.4 Create CloudFront Distribution

```bash
# Get OAC ID from step 2.1
OAC_ID=<your-oac-id>
KEY_GROUP_ID=<your-key-group-id>

aws cloudfront create-distribution \
  --distribution-config '{
    "CallerReference": "rallycut-'${ENV}'-'$(date +%s)'",
    "Comment": "RallyCut video streaming",
    "Enabled": true,
    "Origins": {
      "Quantity": 1,
      "Items": [
        {
          "Id": "S3-'${BUCKET_NAME}'",
          "DomainName": "'${BUCKET_NAME}'.s3.'${REGION}'.amazonaws.com",
          "OriginAccessControlId": "'${OAC_ID}'",
          "S3OriginConfig": {
            "OriginAccessIdentity": ""
          }
        }
      ]
    },
    "DefaultCacheBehavior": {
      "TargetOriginId": "S3-'${BUCKET_NAME}'",
      "ViewerProtocolPolicy": "redirect-to-https",
      "AllowedMethods": {
        "Quantity": 2,
        "Items": ["GET", "HEAD"],
        "CachedMethods": {
          "Quantity": 2,
          "Items": ["GET", "HEAD"]
        }
      },
      "CachePolicyId": "658327ea-f89d-4fab-a63d-7e88639e58f6",
      "Compress": true,
      "TrustedKeyGroups": {
        "Enabled": true,
        "Quantity": 1,
        "Items": ["'${KEY_GROUP_ID}'"]
      }
    },
    "PriceClass": "PriceClass_100"
  }'
```

Note the `DomainName` from the output (e.g., `d123456789.cloudfront.net`).

### 2.5 Update S3 Bucket Policy

```bash
DISTRIBUTION_ARN=<your-distribution-arn>

aws s3api put-bucket-policy --bucket ${BUCKET_NAME} --policy '{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowCloudFrontServicePrincipal",
      "Effect": "Allow",
      "Principal": {
        "Service": "cloudfront.amazonaws.com"
      },
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::'${BUCKET_NAME}'/*",
      "Condition": {
        "StringEquals": {
          "AWS:SourceArn": "'${DISTRIBUTION_ARN}'"
        }
      }
    }
  ]
}'
```

## 3. Create IAM User for API

```bash
# Create user
aws iam create-user --user-name rallycut-api-${ENV}

# Create policy
aws iam put-user-policy --user-name rallycut-api-${ENV} --policy-name S3Access --policy-document '{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject"
      ],
      "Resource": "arn:aws:s3:::'${BUCKET_NAME}'/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:ListBucket"
      ],
      "Resource": "arn:aws:s3:::'${BUCKET_NAME}'"
    }
  ]
}'

# Create access keys
aws iam create-access-key --user-name rallycut-api-${ENV}
```

**Save the AccessKeyId and SecretAccessKey securely!**

## 4. Update API Environment Variables

Add these to your `api/.env`:

```bash
# AWS
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=<your-access-key-id>
AWS_SECRET_ACCESS_KEY=<your-secret-access-key>

# S3
S3_BUCKET_NAME=rallycut-dev

# CloudFront
CLOUDFRONT_DOMAIN=d123456789.cloudfront.net
CLOUDFRONT_KEY_PAIR_ID=<your-key-pair-id>
CLOUDFRONT_PRIVATE_KEY="-----BEGIN RSA PRIVATE KEY-----
<paste the contents of cloudfront-private-key.pem here>
-----END RSA PRIVATE KEY-----"
```

## 5. Test the Setup

```bash
# Test S3 upload
echo "test" > test.txt
aws s3 cp test.txt s3://${BUCKET_NAME}/test.txt
aws s3 rm s3://${BUCKET_NAME}/test.txt
rm test.txt

# Test CloudFront (after distribution is deployed - takes ~15 min)
curl -I https://d123456789.cloudfront.net/videos/test.mp4
```

## Summary of Values Needed

| Variable | Description |
|----------|-------------|
| `S3_BUCKET_NAME` | e.g., `rallycut-dev` |
| `CLOUDFRONT_DOMAIN` | e.g., `d123456789.cloudfront.net` |
| `CLOUDFRONT_KEY_PAIR_ID` | Public key ID from CloudFront |
| `CLOUDFRONT_PRIVATE_KEY` | Contents of `cloudfront-private-key.pem` |
| `AWS_ACCESS_KEY_ID` | From IAM user creation |
| `AWS_SECRET_ACCESS_KEY` | From IAM user creation |

## Cost Guardrails

Run the guardrails script after initial setup to prevent surprise bills:

```bash
./scripts/setup-aws-guardrails.sh dev your@email.com 10
```

This configures:

| Guardrail | What it does |
|-----------|--------------|
| **AWS Budget** | $10/month with email alerts at 50%, 80%, and 100% forecasted |
| **Lambda concurrency** | Export: 2, Optimize: 3 (prevents runaway invocations) |
| **S3 lifecycle** | `exports/` deleted at 30d, `videos/` tiered at 30d, incomplete uploads aborted at 3d |
| **CloudWatch alarms** | Duration (p99 > 80% timeout) and error (>3 in 5min) alerts per Lambda |

**S3 lifecycle details:**

| Prefix | Transition | Expiration | Rationale |
|--------|-----------|------------|-----------|
| `exports/` | — | Delete at 30d | Regenerable from source videos |
| `videos/` | Intelligent-Tiering at 30d | None (app manages via `cleanupExpiredContent()`) | Paid user content — never auto-delete |
| `analysis/` | Standard-IA at 30d | None | Small files, infrequent access |
| *(all)* | — | Abort incomplete multipart at 3d | Prevents orphaned upload parts |

**Modal:** Set spending limit manually at [modal.com/settings](https://modal.com/settings) → Usage/Billing.

The script is idempotent — safe to re-run with different parameters.

## Cost Estimate (MVP)

| Service | Usage | Monthly Cost |
|---------|-------|--------------|
| S3 | ~10GB storage | ~$0.25 |
| CloudFront | ~10GB transfer | ~$0.85 |
| **Total** | | **~$1.10/mo** |
