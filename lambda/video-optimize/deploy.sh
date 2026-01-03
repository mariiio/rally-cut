#!/bin/bash
set -e

STAGE=${1:-dev}
REGION=${AWS_REGION:-us-east-1}
FUNCTION_NAME="rallycut-video-optimize-${STAGE}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO="rallycut-video-optimize-${STAGE}"
ECR_IMAGE="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO}:latest"
ROLE_NAME="${FUNCTION_NAME}-role"
S3_BUCKET="rallycut-${STAGE}"

echo "🚀 Deploying Video Optimization Lambda: $FUNCTION_NAME"
echo ""

# Step 1: Create ECR repository if needed
echo "📦 Setting up ECR repository..."
if ! aws ecr describe-repositories --repository-names $ECR_REPO --region $REGION 2>/dev/null; then
  aws ecr create-repository --repository-name $ECR_REPO --region $REGION
  echo "   Created ECR repo: $ECR_REPO"
else
  echo "   ECR repo exists: $ECR_REPO"
fi

# Step 2: Build and push Docker image
echo ""
echo "🐳 Building Docker image (ARM64)..."
docker build --platform linux/arm64 -t $ECR_REPO:latest .

echo ""
echo "📤 Pushing to ECR..."
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com
docker tag ${ECR_REPO}:latest $ECR_IMAGE
docker push $ECR_IMAGE

# Step 3: Create IAM role if it doesn't exist
echo ""
echo "📋 Setting up IAM role..."
if ! aws iam get-role --role-name $ROLE_NAME 2>/dev/null; then
  aws iam create-role \
    --role-name $ROLE_NAME \
    --assume-role-policy-document '{
      "Version": "2012-10-17",
      "Statement": [{
        "Effect": "Allow",
        "Principal": {"Service": "lambda.amazonaws.com"},
        "Action": "sts:AssumeRole"
      }]
    }'

  # Attach basic Lambda execution policy
  aws iam attach-role-policy \
    --role-name $ROLE_NAME \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

  # Add S3 permissions (read original, write processed)
  aws iam put-role-policy \
    --role-name $ROLE_NAME \
    --policy-name S3Access \
    --policy-document "{
      \"Version\": \"2012-10-17\",
      \"Statement\": [{
        \"Effect\": \"Allow\",
        \"Action\": [\"s3:GetObject\", \"s3:PutObject\"],
        \"Resource\": \"arn:aws:s3:::${S3_BUCKET}/*\"
      }]
    }"

  echo "   Created role: $ROLE_NAME"
  echo "⏳ Waiting for role to propagate..."
  sleep 10
else
  echo "   Role exists: $ROLE_NAME"
fi

ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"

# Step 4: Create or update Lambda function
echo ""
echo "🔧 Creating/updating Lambda function..."
if aws lambda get-function --function-name $FUNCTION_NAME 2>/dev/null; then
  aws lambda update-function-code \
    --function-name $FUNCTION_NAME \
    --image-uri $ECR_IMAGE
  echo "   Updated function code"
else
  aws lambda create-function \
    --function-name $FUNCTION_NAME \
    --package-type Image \
    --code ImageUri=$ECR_IMAGE \
    --role $ROLE_ARN \
    --timeout 600 \
    --memory-size 2048 \
    --architectures arm64 \
    --environment "Variables={S3_BUCKET_NAME=${S3_BUCKET}}"
  echo "   Created function: $FUNCTION_NAME"
fi

echo ""
echo "✅ Lambda deployed successfully!"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Add to your API .env:"
echo ""
echo "  PROCESSING_LAMBDA_FUNCTION_NAME=$FUNCTION_NAME"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
